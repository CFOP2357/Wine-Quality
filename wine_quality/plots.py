"""Figuras de evaluación del modelo (real vs predicho y residuales)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: apto para ejecución por script
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from wine_quality.config import FIGURES_DIR


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
