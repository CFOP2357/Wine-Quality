"""Inferencia batch con un modelo entrenado y registrado en MLflow.

Carga el pipeline completo (preprocesamiento + modelo) de una corrida y predice
la calidad para un CSV con las columnas de entrada (las 11 fisicoquímicas +
'tipo'); si trae la columna 'quality' se ignora como feature.

    python -m wine_quality.modeling.predict --run-id <id> --input data/processed/test.csv

Sin --run-id usa la corrida con mejor val_rmse del experimento (el campeón).
"""

import argparse
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd

from wine_quality.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    TARGET,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predice la calidad del vino con un modelo registrado en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Corrida de MLflow cuyo modelo cargar; por defecto, la de mejor "
        "val_rmse del experimento",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "test.csv",
        help="CSV con las columnas de entrada",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "predictions.csv",
        help="CSV de salida: las entradas + columna 'prediction'",
    )
    return parser.parse_args(argv)


def best_run_id() -> str:
    """Corrida con mejor (menor) val_rmse del experimento."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.val_rmse ASC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No hay corridas en el experimento {EXPERIMENT_NAME!r}")
    run = runs.iloc[0]
    logger.info(
        f"Campeón por val_rmse: {run['tags.mlflow.runName']} "
        f"(val_rmse={run['metrics.val_rmse']:.4f}, run_id={run['run_id']})"
    )
    return run["run_id"]


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_id = args.run_id or best_run_id()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    df = pd.read_csv(args.input)
    X = df.drop(columns=[TARGET], errors="ignore")
    out = df.assign(prediction=model.predict(X))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.success(f"{len(out)} predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
