"""Búsqueda del mejor valor de UN hiperparámetro (sweep de una coordenada).

Declara "sobre este hiperparámetro busca; los demás quedan fijos en lo que
diga la CLI": entrena una corrida completa por valor, compara con val_rmse
(test no participa en la selección) y registra todo en MLflow como una corrida
padre `tune_<param>` con una corrida hija anidada por valor, más la curva de
validación (RMSE train vs val contra el hiperparámetro) como artefacto.

    python -m wine_quality.modeling.tune --model xgb --tune-param max_depth \
        --values 3,4,5,6,7 --learning-rate 0.05 --early-stopping 50

    mlflow run . -e tune --env-manager local \
        -P model=xgb -P tune_param=max_depth -P values=3,4,5,6,7
"""

from dataclasses import replace

from loguru import logger
import mlflow

from wine_quality.modeling.train import (
    ExperimentConfig,
    MLflowTracker,
    run_experiment,
)
from wine_quality.plots import validation_curve_plot

# Hiperparámetros buscables y cómo castear cada valor de la lista --values
TUNABLE = {
    "n_estimators": int,
    "max_depth": lambda v: None if int(v) == -1 else int(v),
    "min_samples_leaf": int,
    "learning_rate": float,
    "imputer": str,
    "normalizer": str,
    "scaler": str,
}


def parse_args(argv: list[str] | None = None):
    parser = ExperimentConfig._parser()
    parser.description = (
        "Busca el mejor valor de UN hiperparámetro entrenando una corrida por "
        "valor y comparando val_rmse; el resto de la configuración queda fija."
    )
    parser.add_argument(
        "--tune-param",
        required=True,
        choices=sorted(TUNABLE),
        help="Hiperparámetro sobre el que buscar (los demás quedan fijos)",
    )
    parser.add_argument(
        "--values",
        required=True,
        help="Valores a probar, separados por coma (p. ej. '3,4,5,6,7'; "
        "en max_depth, -1 = default de la librería)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    # La configuración base (los "fijos") reutiliza la normalización de train.py
    base = ExperimentConfig.from_namespace(args)

    cast = TUNABLE[args.tune_param]
    values = [cast(v.strip()) for v in args.values.split(",")]
    logger.info(
        f"Sweep de '{args.tune_param}' sobre {values} | fijos: modelo={base.model}, "
        f"seed={base.seed} (selección por val_rmse; test no participa)"
    )

    tracker = MLflowTracker()
    tracker.setup()
    with mlflow.start_run(run_name=f"tune_{args.tune_param}_{base.model}") as parent:
        mlflow.set_tags(
            {
                "mlflow.runName": f"tune_{args.tune_param}_{base.model}",
                "etapa": "tuning",
                "autor": "Yoguel Salazar",
            }
        )
        trials = []  # (valor, train_rmse, val_rmse)
        for value in values:
            config = replace(base, **{args.tune_param: value})
            logger.info(f"--- Candidato {args.tune_param}={value} ---")
            results = run_experiment(config, nested=True)
            trials.append((value, results.loc["rmse", "train"], results.loc["rmse", "val"]))

        best_value, best_train, best_val = min(trials, key=lambda t: t[2])
        fig = validation_curve_plot(
            args.tune_param,
            [t[0] for t in trials],
            [t[1] for t in trials],
            [t[2] for t in trials],
            best_value,
        )
        # Resumen en la corrida padre (claves nuevas: sin conflicto con las de
        # `mlflow run`, que registra los parámetros del MLproject en el padre)
        existing = mlflow.get_run(parent.info.run_id).data.params
        mlflow.log_params(
            {
                k: v
                for k, v in {
                    "tune_param": args.tune_param,
                    "tune_values": ",".join(str(v) for v in values),
                    "best_value": best_value,
                }.items()
                if k not in existing
            }
        )
        mlflow.log_metric("best_val_rmse", float(best_val))
        mlflow.log_artifact(str(fig), artifact_path="figures")

    logger.success(
        f"Sweep completo | mejor {args.tune_param}={best_value} | "
        f"val_rmse={best_val:.4f} (train_rmse={best_train:.4f})"
    )


if __name__ == "__main__":
    main()
