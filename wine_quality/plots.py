"""Figuras del proyecto (todas sobre reports/figures/):

- evaluation_plot: real vs predicho + residuales de una partición.
- iteration_curve: RMSE train vs val por ronda de boosting (early stopping).
- validation_curve_plot: RMSE train vs val al variar un hiperparámetro (sweep).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: apto para ejecución por script
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from wine_quality.config import FIGURES_DIR


def iteration_curve(evals_result: dict, best_iteration: int, out_dir: Path = FIGURES_DIR) -> Path:
    """Curva RMSE train vs val por ronda de boosting (diagnóstico de overfitting):
    mientras las curvas bajan juntas el modelo aprende señal; donde val toca su
    mínimo y empieza a subir (línea punteada = mejor iteración) y train sigue
    bajando, las curvas se alejan: ahí empieza el sobreajuste."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse = evals_result["validation_1"]["rmse"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_rmse, label="train")
    ax.plot(val_rmse, label="val")
    ax.axvline(
        best_iteration, color="red", ls="--", lw=1, label=f"mejor iteración ({best_iteration})"
    )
    ax.set(
        xlabel="Ronda de boosting",
        ylabel="RMSE",
        title="Curva de entrenamiento: RMSE train vs val por iteración",
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "iteration_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Curva de iteraciones guardada en {path}")
    return path


def validation_curve_plot(
    param_name: str,
    values: list,
    train_rmse: list[float],
    val_rmse: list[float],
    best_value,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Curva de validación: RMSE train vs val al variar UN hiperparámetro
    (sklearn 'validation curve'). Donde las curvas se separan y val empeora,
    ese valor del hiperparámetro ya sobreajusta; el mejor valor es el mínimo
    de la curva de val."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(v) for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels, train_rmse, marker="o", label="train")
    ax.plot(labels, val_rmse, marker="o", label="val")
    ax.axvline(str(best_value), color="red", ls="--", lw=1, label=f"mejor valor ({best_value})")
    ax.set(xlabel=param_name, ylabel="RMSE", title=f"Curva de validación: RMSE vs {param_name}")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"validation_curve_{param_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Curva de validación guardada en {path}")
    return path


def evaluation_plot(y_true, y_pred, partition: str = "test", out_dir: Path = FIGURES_DIR) -> Path:
    """Diagnóstico visual: dispersión real vs predicho + histograma de residuales."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_true, y_pred, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set(
        xlabel="Calidad real",
        ylabel="Calidad predicha",
        title=f"Real vs predicho ({partition})",
    )

    sns.histplot(y_true - y_pred, kde=True, ax=axes[1])
    axes[1].set_title(f"Residuales ({partition})")

    fig.tight_layout()
    path = out_dir / f"evaluation_{partition}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Figura de evaluación guardada en {path}")
    return path
