"""Preprocesamiento de features para Wine Quality.

Devuelve transformadores SIN ajustar: el fit() ocurre dentro del pipeline del
modelo y aprende únicamente de la partición de entrenamiento (evita fuga de
datos; ver scikit-learn Common Pitfalls).

Los pasos numéricos son configurables por corrida (--normalizer, --scaler):
así el preprocesamiento se vuelve un hiperparámetro más del experimento
(p. ej. los árboles no requieren escalado, la regresión lineal sí se beneficia).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

from wine_quality.config import TARGET

# Opciones de cada paso numérico; "none" omite el paso.
IMPUTERS = {
    "median": lambda: SimpleImputer(strategy="median"),
    "mean": lambda: SimpleImputer(strategy="mean"),
    "knn": lambda: KNNImputer(n_neighbors=5),
}
NORMALIZERS = {
    "yeo-johnson": lambda: PowerTransformer(method="yeo-johnson", standardize=False),
    "none": None,
}
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identifica columnas numéricas y categóricas (excluyendo el target)."""
    features = df.drop(columns=[TARGET])
    num_cols = features.select_dtypes(include="number").columns.tolist()
    cat_cols = features.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    imputer: str = "median",
    normalizer: str = "yeo-johnson",
    scaler: str = "standard",
) -> ColumnTransformer:
    """Preprocesador configurable (defaults = decisiones documentadas en el EDA):

    - Numéricas: imputación (mediana por default: robusta a la asimetría) ->
      normalización Yeo-Johnson (corrige colas pesadas sin eliminar outliers) ->
      estandarización (media 0 / desviación 1). Cada paso puede cambiarse u
      omitirse por corrida (--imputer, --normalizer, --scaler).
    - Categóricas: imputación moda -> one-hot ('tipo' en columnas binarias).
    """
    num_steps: list[tuple] = [("imputer", IMPUTERS[imputer]())]
    if NORMALIZERS[normalizer] is not None:
        num_steps.append(("normalizer", NORMALIZERS[normalizer]()))
    if SCALERS[scaler] is not None:
        num_steps.append(("scaler", SCALERS[scaler]()))

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
