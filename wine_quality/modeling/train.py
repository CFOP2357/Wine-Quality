"""Script de entrenamiento parametrizado para Wine Quality (regresión),
instrumentado con MLflow Tracking.

    python -m wine_quality.modeling.train                      # defaults (linear)
    python -m wine_quality.modeling.train --model rf --n-estimators 300
    python -m wine_quality.modeling.train --model xgb --learning-rate 0.05 --seed 7
    python -m wine_quality.modeling.train --model tree --normalizer none --scaler none

Cada corrida queda registrada en el experimento MLflow (parámetros, métricas,
figura de evaluación, tabla de métricas y el pipeline entrenado). Interfaz:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

El flujo está organizado en objetos con una única responsabilidad:

- ExperimentConfig: la configuración de la corrida (CLI -> valores tipados).
- Trainer: construye el pipeline (preprocesador + modelo del factory) y lo ajusta.
- Evaluator: calcula la tabla de métricas (modelo y referencia dummy).
- MLflowTracker: configura el tracking y registra la corrida.
- run_experiment: orquesta datos -> Trainer -> Evaluator -> figura -> MLflowTracker.

La carga/partición vive en dataset.py, el preprocesamiento en features.py y la
selección de modelo en modeling/models.py (Factory).
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

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
from wine_quality.features import (
    IMPUTERS,
    NORMALIZERS,
    SCALERS,
    build_preprocessor,
    feature_columns,
)
from wine_quality.modeling.models import REGISTRY, build_model, used_params
from wine_quality.plots import evaluation_plot, iteration_curve


# ====================== CONFIGURACIÓN ======================


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración tipada de una corrida: única fuente de verdad de modelo,
    semilla, hiperparámetros y preprocesamiento. Se construye desde la CLI con
    `from_cli()`, que también normaliza los valores compatibles con MLproject
    (sentinela -1 -> None, strings 'true'/'false' -> bool)."""

    model: str = "linear"
    seed: int = DEFAULT_SEED
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 1
    learning_rate: float = 0.1
    fit_intercept: bool = True
    imputer: str = "median"
    normalizer: str = "yeo-johnson"
    scaler: str = "standard"
    overfit_threshold: float = 3.0
    early_stopping: int | None = None

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ExperimentConfig":
        args = cls._parser().parse_args(argv)
        return cls(
            model=args.model,
            seed=args.seed,
            n_estimators=args.n_estimators,
            # -1 (sentinela de MLproject) equivale a no pasarlo -> default de la librería
            max_depth=None if args.max_depth in (None, -1) else args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            learning_rate=args.learning_rate,
            fit_intercept=args.fit_intercept == "true",
            imputer=args.imputer,
            normalizer=args.normalizer,
            scaler=args.scaler,
            overfit_threshold=args.overfit_threshold,
            # -1 (sentinela de MLproject) = sin early stopping
            early_stopping=None if args.early_stopping in (None, -1) else args.early_stopping,
        )

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        """CLI del experimento. Todos los argumentos tienen default, así el
        script corre sin argumentos; cada modelo consume solo los
        hiperparámetros que le aplican (ver REGISTRY) e ignora los demás."""
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
            "--imputer",
            choices=sorted(IMPUTERS),
            default="median",
            help="Imputación de variables numéricas",
        )
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
        parser.add_argument(
            "--overfit-threshold",
            type=float,
            default=3.0,
            help="Umbral (%%) de brecha RMSE val-vs-train para marcar la corrida "
            "como overfitting (diagnóstico: etiqueta y advierte, no descarta)",
        )
        parser.add_argument(
            "--early-stopping",
            type=int,
            default=None,
            help="[xgb] Rondas sin mejora en val para detener el boosting; XGBoost "
            "se queda con la mejor iteración (pon --n-estimators alto, p. ej. 1000). "
            "Genera la curva train-vs-val por iteración como artefacto. -1 = desactivado",
        )
        return parser

    @property
    def hyperparams(self) -> dict:
        """Superset de hiperparámetros del modelo (el factory filtra)."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "learning_rate": self.learning_rate,
            "fit_intercept": self.fit_intercept,
        }

    @property
    def effective_params(self) -> dict:
        """Parámetros que realmente usa la corrida (los que se registran)."""
        params = used_params(self.model, self.seed, **self.hyperparams)
        params |= {
            "imputer": self.imputer,
            "normalizer": self.normalizer,
            "scaler": self.scaler,
        }
        if self.early_stopping is not None:
            params["early_stopping"] = self.early_stopping
        return params

    @property
    def run_name(self) -> str:
        """Nombre autodescriptivo de la corrida: modelo + hiperparámetros que
        ese modelo consume + desviaciones de preprocesamiento respecto al
        default del EDA. Así la vista de corridas de MLflow dice por sí sola
        qué cambió en cada una (la semilla ya está como parámetro).

        Ejemplos: 'rf_ne300_md12_msl1_sin-norm_sin-esc', 'xgb_ne300_md4_lr0.05'.
        """
        abbrev = {
            "n_estimators": "ne",
            "max_depth": "md",
            "min_samples_leaf": "msl",
            "learning_rate": "lr",
        }
        parts = [self.model]
        _, keys, _ = REGISTRY[self.model]
        for key in keys:
            value = self.hyperparams[key]
            if key == "fit_intercept":
                parts.append(f"intercepto-{'si' if value else 'no'}")
            elif key == "max_depth" and value is None:
                parts.append("md-auto")
            else:
                parts.append(f"{abbrev[key]}{value}")
        if self.imputer != "median":
            parts.append(f"imp-{self.imputer}")
        if self.normalizer != "yeo-johnson":
            parts.append("sin-norm" if self.normalizer == "none" else f"norm-{self.normalizer}")
        if self.scaler != "standard":
            parts.append("sin-esc" if self.scaler == "none" else f"esc-{self.scaler}")
        if self.early_stopping is not None:
            parts.append(f"es{self.early_stopping}")
        return "_".join(parts)


# ======================= ENTRENAMIENTO =======================


class Trainer:
    """Construye el pipeline ÚNICO (preprocesamiento + modelo) a partir de la
    configuración y lo ajusta SOLO con la partición de entrenamiento.

    Con --early-stopping (solo xgb) también usa val como eval_set: XGBoost
    evalúa train y val en cada ronda de boosting, se detiene cuando val deja
    de mejorar y conserva la mejor iteración. Eso es *seleccionar* con val
    (legítimo); test no participa. Las curvas quedan en `evals_result_` y la
    mejor ronda en `best_iteration_`."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evals_result_: dict | None = None
        self.best_iteration_: int | None = None

    def build_pipeline(self, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
        return Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        num_cols,
                        cat_cols,
                        imputer=self.config.imputer,
                        normalizer=self.config.normalizer,
                        scaler=self.config.scaler,
                    ),
                ),
                ("model", build_model(self.config.model, self.config.seed, **self.config.hyperparams)),
            ]
        )

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> Pipeline:
        num_cols, cat_cols = feature_columns(df_train)
        pipeline = self.build_pipeline(num_cols, cat_cols)
        X_train, y_train = split_features_target(df_train)

        use_early_stopping = (
            self.config.early_stopping is not None and self.config.model == "xgb"
        )
        if not use_early_stopping:
            if self.config.early_stopping is not None:
                logger.warning(
                    f"--early-stopping solo aplica a xgb; se ignora para {self.config.model}"
                )
            pipeline.fit(X_train, y_train)
            return pipeline

        # Early stopping: el preprocesador se ajusta SOLO con train y val se
        # transforma con él (sin fuga); el modelo recibe ambas como eval_set.
        assert df_val is not None, "early stopping requiere la partición de validación"
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        X_val, y_val = split_features_target(df_val)
        Xt_train = preprocessor.fit_transform(X_train, y_train)
        Xt_val = preprocessor.transform(X_val)

        model.set_params(early_stopping_rounds=self.config.early_stopping, eval_metric="rmse")
        model.fit(Xt_train, y_train, eval_set=[(Xt_train, y_train), (Xt_val, y_val)], verbose=False)

        self.evals_result_ = model.evals_result()
        self.best_iteration_ = int(model.best_iteration)
        logger.info(
            f"Early stopping: mejor iteración = {self.best_iteration_} "
            f"de un máximo de {self.config.n_estimators}"
        )
        # El pipeline queda con ambos pasos ya ajustados: predict funciona igual
        # que sin early stopping (y usa la mejor iteración automáticamente).
        return pipeline


# ========================= EVALUACIÓN =========================


class Evaluator:
    """Calcula la tabla de métricas (MAE, RMSE, R2) por partición, para el
    modelo entrenado y para una referencia trivial (DummyRegressor: ¿el modelo
    aporta algo sobre predecir la media?)."""

    def __init__(self, df_train, df_val, df_test):
        self.partitions = {
            "train": split_features_target(df_train),
            "val": split_features_target(df_val),
            "test": split_features_target(df_test),
        }

    @staticmethod
    def evaluate(estimator, X, y) -> dict[str, float]:
        """MAE, RMSE y R2 de cualquier estimador con .predict() (Strategy)."""
        pred = estimator.predict(X)
        return {
            "mae": metrics.mean_absolute_error(y, pred),
            "rmse": metrics.root_mean_squared_error(y, pred),
            "r2": metrics.r2_score(y, pred),
        }

    def results_table(self, model: Pipeline) -> pd.DataFrame:
        X_train, y_train = self.partitions["train"]
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
        columns = {
            name: self.evaluate(model, X, y) for name, (X, y) in self.partitions.items()
        }
        for name in ("val", "test"):
            X, y = self.partitions[name]
            columns[f"dummy_{name}"] = self.evaluate(dummy, X, y)
        return pd.DataFrame(columns).round(4)

    @staticmethod
    def overfit_gap_pct(results: pd.DataFrame) -> float:
        """Brecha relativa (%) del RMSE val respecto a train: diagnóstico de
        overfitting. Nota: los ensambles de árboles casi memorizan train por
        diseño, así que una brecha grande solo es dañina si además empeora val."""
        train_rmse = results.loc["rmse", "train"]
        val_rmse = results.loc["rmse", "val"]
        return float((val_rmse - train_rmse) / train_rmse * 100)


# ========================== TRACKING ==========================


class MLflowTracker:
    """Configura el tracking de MLflow y registra la corrida completa
    (parámetros, métricas, artefactos y el pipeline entrenado)."""

    def setup(self) -> None:
        """Según cómo se lanzó el script:

        - `mlflow run`: el proceso padre YA creó la corrida (MLFLOW_RUN_ID en
          el entorno) con su propio tracking URI/experimento; aquí no se toca
          nada y start_run() simplemente se adhiere a esa corrida.
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

    def log_run(
        self,
        config: ExperimentConfig,
        sizes: dict,
        results: pd.DataFrame,
        fig_paths: list[Path],
        model: Pipeline,
        X_train: pd.DataFrame,
        overfit_gap: float,
    ) -> str:
        is_overfit = overfit_gap > config.overfit_threshold
        with mlflow.start_run(run_name=config.run_name) as run:
            # El tag funciona también cuando la corrida la creó `mlflow run`
            # (ahí run_name de start_run se ignora al adherirse a la corrida activa)
            mlflow.set_tags(
                {
                    "mlflow.runName": config.run_name,
                    "etapa": "experimentacion",
                    "problema": "regression",
                    "autor": "Yoguel Salazar",
                    "is_overfitting": "si" if is_overfit else "no",
                }
            )
            # Parámetros: configuración efectiva (modelo + preprocesamiento) y datos.
            # `mlflow run` ya registró los parámetros del MLproject en la corrida
            # (como strings crudos); MLflow prohíbe cambiarles el valor, así que
            # solo se agregan las claves nuevas (model_class, n_train, etc.).
            existing = mlflow.get_run(run.info.run_id).data.params
            mlflow.log_params(
                {k: v for k, v in (config.effective_params | sizes).items() if k not in existing}
            )

            # Métricas: mae/rmse/r2 por partición (modelo y referencia dummy)
            # + brecha de overfitting (RMSE val vs train, en %)
            for col in results.columns:
                mlflow.log_metrics({f"{col}_{k}": float(v) for k, v in results[col].items()})
            mlflow.log_metric("overfit_gap_pct", round(overfit_gap, 1))

            # Artefactos: figuras (evaluación y, con early stopping, la curva
            # train-vs-val por iteración) y tabla de métricas
            for fig in fig_paths:
                mlflow.log_artifact(str(fig), artifact_path="figures")
            mlflow.log_table(results.reset_index(names="metric"), artifact_file="metrics_table.json")

            # Modelo entrenado (pipeline completo) con firma de entrada/salida
            # (formato skops: deserialización segura; se declaran confiables los
            # tipos de NUESTRO propio pipeline: numpy.dtype de los encoders y,
            # para xgb, las clases de XGBoost)
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


# ======================== ORQUESTACIÓN ========================


def split_features_target(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Orquesta el flujo: datos -> Trainer -> Evaluator -> figura -> tracking."""
    logger.info(f"Configuración efectiva: {config.effective_params}")

    # 1. Datos: carga -> limpieza -> partición 70/15/15 estratificada (con la semilla)
    df_train, df_val, df_test = dataset.prepare(seed=config.seed)
    sizes = {
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "split": "70/15/15 estratificado por quality",
    }

    # 2. Entrenamiento: pipeline único ajustado SOLO con train
    # (con --early-stopping, val actúa como eval_set para elegir la iteración)
    trainer = Trainer(config)
    model = trainer.train(df_train, df_val)
    if trainer.best_iteration_ is not None:
        sizes["best_iteration"] = trainer.best_iteration_

    # 3. Métricas por partición (modelo y dummy)
    evaluator = Evaluator(df_train, df_val, df_test)
    results = evaluator.results_table(model)
    logger.info(f"Métricas:\n{results}")

    # 4. Diagnóstico de overfitting: brecha relativa RMSE val vs train
    overfit_gap = Evaluator.overfit_gap_pct(results)
    if overfit_gap > config.overfit_threshold:
        logger.warning(
            f"Posible overfitting: RMSE val {overfit_gap:.1f}% peor que train "
            f"(umbral {config.overfit_threshold}%). La corrida queda etiquetada "
            "'is_overfitting=si'; contrasta con val_rmse antes de descartarla."
        )

    # 5. Figuras: diagnóstico sobre test y, con early stopping, la curva
    # train-vs-val por iteración
    X_test, y_test = evaluator.partitions["test"]
    fig_paths = [evaluation_plot(y_test, model.predict(X_test), partition="test")]
    if trainer.evals_result_ is not None:
        fig_paths.append(iteration_curve(trainer.evals_result_, trainer.best_iteration_))

    # 6. Registro de la corrida en MLflow
    tracker = MLflowTracker()
    tracker.setup()
    X_train, _ = evaluator.partitions["train"]
    tracker.log_run(config, sizes, results, fig_paths, model, X_train, overfit_gap)

    logger.success(
        f"Entrenamiento completo | modelo={config.model} | "
        f"val_rmse={results.loc['rmse', 'val']:.4f} | val_mae={results.loc['mae', 'val']:.4f} | "
        f"overfit_gap={overfit_gap:.1f}% | "
        f"is_overfitting={'si' if overfit_gap > config.overfit_threshold else 'no'}"
    )
    return results


def main(argv: list[str] | None = None):
    run_experiment(ExperimentConfig.from_cli(argv))


if __name__ == "__main__":
    main()
