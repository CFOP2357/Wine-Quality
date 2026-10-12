"""Script de entrenamiento parametrizado para Wine Quality (regresión),
instrumentado con MLflow Tracking.

    python -m wine_quality.modeling.train                      # defaults (linear)
    python -m wine_quality.modeling.train --model rf --n-estimators 300
    python -m wine_quality.modeling.train --model xgb --learning-rate 0.05 --seed 7
    python -m wine_quality.modeling.train --model tree --normalizer none --scaler none

Cada corrida queda registrada en el experimento MLflow (parámetros, métricas,
figura de evaluación, tabla de métricas y el pipeline entrenado). Interfaz:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Responsabilidades: parse_args (CLI), evaluate (métricas), log_to_mlflow
(tracking), run_experiment (orquestación). La carga/partición vive en
dataset.py, el preprocesamiento en features.py y la selección de modelo en
modeling/models.py (Factory).
"""

import argparse
import os

import mlflow
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from wine_quality import dataset
from wine_quality.config import (
    DEFAULT_SEED,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    TARGET,
)
from wine_quality.features import NORMALIZERS, SCALERS, build_preprocessor, feature_columns
from wine_quality.modeling.models import REGISTRY, build_model, used_params
from wine_quality.plots import evaluation_plot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI del experimento: modelo, semilla, hiperparámetros y preprocesamiento.

    Todos los argumentos tienen default, así el script corre sin argumentos.
    Cada modelo consume solo los hiperparámetros que le aplican (ver REGISTRY);
    los demás se ignoran.
    """
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de regresión para Wine Quality, lo evalúa "
        "en train/val/test y registra la corrida en MLflow. Hiperparámetros no "
        "aplicables al modelo elegido se ignoran.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(REGISTRY),
        default="linear",
        help="Modelo a entrenar: linear (LinearRegression), tree (DecisionTree), "
        "rf (RandomForest) o xgb (XGBoost)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla aleatoria: controla la partición de datos y el modelo",
    )
    # --- hiperparámetros del modelo (superset; cada modelo toma su subconjunto) ---
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="[rf, xgb] Número de árboles del ensamble",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="[tree, rf, xgb] Profundidad máxima de cada árbol; -1 o no pasarlo = "
        "default de la librería (sin límite en tree/rf, 6 en xgb)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="[tree, rf] Mínimo de muestras por hoja",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="[xgb] Tasa de aprendizaje del boosting",
    )
    parser.add_argument(
        "--fit-intercept",
        choices=["true", "false"],
        default="true",
        help="[linear] Ajustar el término independiente "
        "(string para ser compatible con los parámetros planos de MLproject)",
    )
    # --- preprocesamiento (también es parte del experimento) ---
    parser.add_argument(
        "--normalizer",
        choices=sorted(NORMALIZERS),
        default="yeo-johnson",
        help="Normalización de variables numéricas ('none' la omite)",
    )
    parser.add_argument(
        "--scaler",
        choices=sorted(SCALERS),
        default="standard",
        help="Escalado de variables numéricas ('none' lo omite)",
    )
    return parser.parse_args(argv)


def evaluate(estimator, X, y) -> dict[str, float]:
    """MAE, RMSE y R2 de un estimador sobre una partición."""
    pred = estimator.predict(X)
    return {
        "mae": metrics.mean_absolute_error(y, pred),
        "rmse": metrics.root_mean_squared_error(y, pred),
        "r2": metrics.r2_score(y, pred),
    }


def split_features_target(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]


def setup_mlflow() -> None:
    """Configura el tracking según cómo se lanzó el script:

    - `mlflow run`: el proceso padre YA creó la corrida (MLFLOW_RUN_ID en el
      entorno) con su propio tracking URI/experimento; aquí no se toca nada y
      start_run() simplemente se adhiere a esa corrida.
    - Ejecución directa (`python -m ...`): tracking local en SQLite con
      artefactos bajo mlartifacts/ (como en la Actividad 2).
    """
    if os.getenv("MLFLOW_RUN_ID"):
        return
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=(PROJ_ROOT / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_to_mlflow(params, sizes, results, fig_path, model, X_train) -> str:
    """Registra la corrida: parámetros, métricas, artefactos y el modelo entrenado."""
    run_name = f"{params['model']}_seed{params['seed']}"
    with mlflow.start_run(run_name=run_name) as run:
        # El tag funciona también cuando la corrida la creó `mlflow run`
        # (ahí run_name de start_run se ignora al adherirse a la corrida activa)
        mlflow.set_tags(
            {
                "mlflow.runName": run_name,
                "etapa": "experimentacion",
                "problema": "regression",
                "autor": "Yoguel Salazar",
            }
        )
        # Parámetros: configuración efectiva (modelo + preprocesamiento) y datos.
        # `mlflow run` ya registró los parámetros del MLproject en la corrida
        # (como strings crudos); MLflow prohíbe cambiarles el valor, así que
        # solo se agregan las claves nuevas (model_class, n_train, etc.).
        existing = mlflow.get_run(run.info.run_id).data.params
        mlflow.log_params({k: v for k, v in (params | sizes).items() if k not in existing})

        # Métricas: mae/rmse/r2 por partición (modelo y referencia dummy)
        for col in results.columns:
            mlflow.log_metrics({f"{col}_{k}": float(v) for k, v in results[col].items()})

        # Artefactos: figura de evaluación y tabla de métricas
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
        mlflow.log_table(results.reset_index(names="metric"), artifact_file="metrics_table.json")

        # Modelo entrenado (pipeline completo) con firma de entrada/salida
        # (formato skops: deserialización segura; se declaran confiables los tipos
        # de NUESTRO propio pipeline: numpy.dtype de los encoders y, para xgb,
        # las clases de XGBoost)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",  # tree y rf
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
            ],
        )
        logger.success(f"Corrida registrada en MLflow | run_id={run.info.run_id}")
        return run.info.run_id


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    """Flujo completo: datos -> pipeline -> entrenamiento -> evaluación -> registro."""
    hyperparams = {
        "n_estimators": args.n_estimators,
        # -1 (sentinela de MLproject) equivale a no pasarlo -> default de la librería
        "max_depth": None if args.max_depth in (None, -1) else args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "learning_rate": args.learning_rate,
        "fit_intercept": args.fit_intercept == "true",
    }
    params = used_params(args.model, args.seed, **hyperparams)
    params |= {"normalizer": args.normalizer, "scaler": args.scaler}
    logger.info(f"Configuración efectiva: {params}")

    # 1. Datos: carga -> limpieza -> partición 70/15/15 estratificada (con la semilla)
    df_train, df_val, df_test = dataset.prepare(seed=args.seed)
    X_train, y_train = split_features_target(df_train)
    X_val, y_val = split_features_target(df_val)
    X_test, y_test = split_features_target(df_test)
    sizes = {"n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
             "split": "70/15/15 estratificado por quality"}

    # 2. Pipeline ÚNICO preprocesamiento + modelo: todo se ajusta SOLO con train
    num_cols, cat_cols = feature_columns(df_train)
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(num_cols, cat_cols,
                                                normalizer=args.normalizer,
                                                scaler=args.scaler)),
            ("model", build_model(args.model, args.seed, **hyperparams)),
        ]
    )
    model.fit(X_train, y_train)

    # Referencia trivial: predecir la media (¿el modelo aporta algo?)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    # 3. Métricas por partición (modelo y dummy)
    results = pd.DataFrame(
        {
            "train": evaluate(model, X_train, y_train),
            "val": evaluate(model, X_val, y_val),
            "test": evaluate(model, X_test, y_test),
            "dummy_val": evaluate(dummy, X_val, y_val),
            "dummy_test": evaluate(dummy, X_test, y_test),
        }
    ).round(4)
    logger.info(f"Métricas:\n{results}")

    # 4. Figura de diagnóstico sobre test
    fig_path = evaluation_plot(y_test, model.predict(X_test), partition="test")

    # 5. Registro de la corrida en MLflow
    setup_mlflow()
    log_to_mlflow(params, sizes, results, fig_path, model, X_train)

    logger.success(
        f"Entrenamiento completo | modelo={args.model} | "
        f"val_rmse={results.loc['rmse', 'val']:.4f} | val_mae={results.loc['mae', 'val']:.4f}"
    )
    return results


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
