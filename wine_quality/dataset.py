"""Carga, limpieza y partición del dataset Wine Quality (UCI).

Migrado del notebook de la Actividad 2: cada función tiene una sola
responsabilidad y todo lo que depende de estadísticas (imputación,
normalización, escalado) vive en el pipeline del modelo, NO aquí.
"""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from wine_quality.config import (
    DEFAULT_SEED,
    PROCESSED_DATA_DIR,
    RED_PATH,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
    WHITE_PATH,
)


def load_raw() -> pd.DataFrame:
    """Lee los CSV de UCI (separador ';') y combina tinto y blanco en un solo
    DataFrame con la columna 'tipo'."""
    assert RED_PATH.exists() and WHITE_PATH.exists(), (
        "Faltan los CSV de Wine Quality en data/raw/."
    )
    df_red = pd.read_csv(RED_PATH, sep=";").assign(tipo="tinto")
    df_white = pd.read_csv(WHITE_PATH, sep=";").assign(tipo="blanco")
    df_raw = pd.concat([df_red, df_white], ignore_index=True)
    logger.info(
        f"Tinto: {len(df_red):,} | Blanco: {len(df_white):,} | "
        f"Total: {df_raw.shape[0]:,} registros, {df_raw.shape[1]} variables"
    )
    return df_raw


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima previa a la partición (decisiones documentadas en el EDA):
    - 'tipo' a category (resto de columnas bien tipadas al cargar).
    - Duplicados exactos fuera ANTES de partir, para evitar fuga entre particiones.
    - Outliers se conservan (mediciones reales); los modera el Yeo-Johnson del pipeline.
    """
    df = df_raw.copy()
    df["tipo"] = df["tipo"].astype("category")
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Duplicados eliminados: {n_dup} | Registros restantes: {len(df):,}")
    return df


def split(df: pd.DataFrame, seed: int = DEFAULT_SEED):
    """Partición estratificada por quality: 70/30 y luego el 30 % a la mitad.

    Las clases 3 y 9 son demasiado raras para estratificarse solas (quality=9
    tiene ~5 muestras); se agrupan con su vecina SOLO en las etiquetas de
    estratificación.
    """
    strat = df[TARGET].clip(lower=4, upper=8)
    df_train, df_temp = train_test_split(
        df, test_size=VAL_SIZE + TEST_SIZE, random_state=seed, stratify=strat
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=seed,
        stratify=df_temp[TARGET].clip(lower=4, upper=8),
    )
    df_train, df_val, df_test = (
        d.reset_index(drop=True) for d in (df_train, df_val, df_test)
    )
    logger.info(
        f"Train: {df_train.shape} ({len(df_train) / len(df):.0%}) | "
        f"Val: {df_val.shape} ({len(df_val) / len(df):.0%}) | "
        f"Test: {df_test.shape} ({len(df_test) / len(df):.0%})"
    )
    return df_train, df_val, df_test


def save_splits(df_train, df_val, df_test) -> None:
    """Guarda las particiones tal como salieron del split (sin transformar)."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        part.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
    logger.success(f"Particiones guardadas en {PROCESSED_DATA_DIR}")


def prepare(seed: int = DEFAULT_SEED, save: bool = True):
    """Flujo completo de datos: carga -> limpieza -> partición [-> guardado]."""
    df = clean(load_raw())
    df_train, df_val, df_test = split(df, seed=seed)
    if save:
        save_splits(df_train, df_val, df_test)
    return df_train, df_val, df_test
