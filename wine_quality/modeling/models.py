"""Fábrica de modelos (patrón Factory + Strategy).

Un único registro mapea el nombre del modelo (CLI) a su clase y a los
hiperparámetros que CONSUME. El script de entrenamiento expone el superset de
hiperparámetros; cada modelo toma su subconjunto e ignora el resto, de modo que
agregar un modelo nuevo es una entrada más en REGISTRY, sin tocar la lógica de
entrenamiento. Todos los estimadores comparten la API fit/predict de sklearn
(Strategy), por lo que el pipeline no cambia.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# nombre -> (clase, hiperparámetros que consume, ¿acepta random_state?)
REGISTRY: dict[str, tuple[type, list[str], bool]] = {
    "linear": (LinearRegression, ["fit_intercept"], False),
    "tree": (DecisionTreeRegressor, ["max_depth", "min_samples_leaf"], True),
    "rf": (
        RandomForestRegressor,
        ["n_estimators", "max_depth", "min_samples_leaf"],
        True,
    ),
    "xgb": (XGBRegressor, ["n_estimators", "max_depth", "learning_rate"], True),
}


def build_model(name: str, seed: int, **hyperparams):
    """Instancia el modelo `name` con el subconjunto de `hyperparams` que consume.

    Valores None se omiten para que aplique el default de la librería
    (p. ej. max_depth=None en árboles, 6 en XGBoost).
    """
    if name not in REGISTRY:
        raise ValueError(f"Modelo desconocido: {name!r}. Opciones: {sorted(REGISTRY)}")
    cls, keys, takes_seed = REGISTRY[name]
    kwargs = {k: hyperparams[k] for k in keys if hyperparams.get(k) is not None}
    if takes_seed:
        kwargs["random_state"] = seed
    return cls(**kwargs)


def used_params(name: str, seed: int, **hyperparams) -> dict:
    """Parámetros efectivos del modelo: los que se registran/reportan.

    Solo incluye los hiperparámetros que el modelo consume, para que la tabla
    comparativa (p. ej. en MLflow) no muestre columnas irrelevantes.
    """
    cls, keys, takes_seed = REGISTRY[name]
    params = {"model": name, "model_class": cls.__name__, "seed": seed}
    params.update({k: hyperparams.get(k) for k in keys})
    return params
