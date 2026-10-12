# Wine-Quality

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Código de experimentación base con MLflow para el dataset **Wine Quality** (UCI).

**TC5061 – Operaciones de Aprendizaje Automático (MLOps)** · Actividad 3
**Autor:** Victor Yoguel Salazar Alanis · A01797950 · Tecnológico de Monterrey

## ¿Qué hace este proyecto?

Convierte el modelo base entrenado en el notebook de la Actividad 2 en **código de
experimentación reutilizable**: un script de entrenamiento parametrizado por línea de
comandos e instrumentado con **MLflow Tracking**. Cada corrida registra sus
hiperparámetros, métricas (MAE, RMSE, R² en train/val/test), artefactos (figura de
evaluación y tabla de métricas) y el pipeline entrenado, de modo que cualquier
configuración es reproducible y comparable desde la interfaz de MLflow.

- **Problema:** regresión de la calidad del vino (`quality`, 3–9) a partir de 11
  variables fisicoquímicas y el tipo de vino (tinto/blanco).
- **Datos:** [Wine Quality (UCI)](https://doi.org/10.24432/C56S3T); los CSV usan `;`
  como separador y se combinan en un solo dataset con la columna `tipo`.
- **Flujo:** carga → limpieza (duplicados fuera antes de partir) → partición 70/15/15
  estratificada → pipeline único (imputación → Yeo-Johnson → estandarización → one-hot
  → modelo) ajustado SOLO con train → evaluación → registro en MLflow.
- **Modelos disponibles:** `linear` (LinearRegression), `tree` (DecisionTree),
  `rf` (RandomForest) y `xgb` (XGBoost), seleccionables por CLI mediante un registro
  tipo *factory* ([`wine_quality/modeling/models.py`](wine_quality/modeling/models.py)):
  cada modelo consume solo los hiperparámetros que le aplican e ignora el resto.

## Instalación

Requiere **Python 3.12**. Desde la raíz del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\activate          # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Coloca los datos crudos en `data/raw/` (no se versionan en git):
`winequality-red.csv` y `winequality-white.csv`, descargados de UCI.

> **Importante:** todos los comandos de este README (`python`, `mlflow run`,
> `mlflow ui`) deben ejecutarse **con el entorno `.venv` activado**. Si tienes otro
> Python/MLflow instalado globalmente, invocar `mlflow` sin activar el entorno usa
> esa otra versión y puede fallar (ver [Solución de problemas](#solución-de-problemas)).

## Cómo ejecutar el entrenamiento

### Opción 1: script directo

El script corre sin argumentos (defaults: `linear`, semilla 42):

```powershell
python -m wine_quality.modeling.train
```

Argumentos disponibles (`--help` muestra esta ayuda):

| Argumento | Aplica a | Default | Descripción |
|---|---|---|---|
| `--model {linear,tree,rf,xgb}` | — | `linear` | Modelo a entrenar |
| `--seed` | todos | `42` | Semilla: controla la partición y el modelo |
| `--n-estimators` | rf, xgb | `200` | Número de árboles del ensamble |
| `--max-depth` | tree, rf, xgb | `-1` | Profundidad máxima (`-1` = default de la librería) |
| `--min-samples-leaf` | tree, rf | `1` | Mínimo de muestras por hoja |
| `--learning-rate` | xgb | `0.1` | Tasa de aprendizaje del boosting |
| `--fit-intercept {true,false}` | linear | `true` | Ajustar término independiente |
| `--imputer {median,mean,knn}` | preprocesamiento | `median` | Imputación de numéricas |
| `--normalizer {yeo-johnson,none}` | preprocesamiento | `yeo-johnson` | Normalización de numéricas |
| `--scaler {standard,minmax,none}` | preprocesamiento | `standard` | Escalado de numéricas |
| `--overfit-threshold` | diagnóstico | `3.0` | % de brecha RMSE val vs train para etiquetar `is_overfitting` |
| `--early-stopping` | xgb | `-1` (off) | Rondas sin mejora en val para detener el boosting y quedarse con la mejor iteración; genera la curva train-vs-val por iteración |

Los hiperparámetros que no aplican al modelo elegido se ignoran (y no se registran),
así la tabla comparativa de MLflow queda limpia.

### Opción 2: `mlflow run` (MLproject)

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow run . --env-manager local --experiment-name TC5061_Actividad3_Experimentacion `
    -P model=rf -P n_estimators=300 -P max_depth=12
```

`--env-manager local` reutiliza el entorno activo (`.venv`); sin esa bandera, MLflow
construye un entorno nuevo a partir de [`python_env.yaml`](python_env.yaml).

### Búsqueda del mejor valor de un hiperparámetro (sweep)

Para buscar el mejor valor de **un** hiperparámetro manteniendo los demás fijos
("sobre este busca, estos déjalos fijos"):

```powershell
python -m wine_quality.modeling.tune --tune-param max_depth --values 3,4,5,6,7 `
    --model xgb --n-estimators 1000 --learning-rate 0.05 --early-stopping 50
```

o con MLproject: `mlflow run . -e tune --env-manager local -P tune_param=max_depth -P values=3,4,5,6,7 -P model=xgb ...`

Entrena una corrida completa por valor (hijas anidadas bajo una corrida padre
`tune_<param>`), selecciona el mejor por `val_rmse` (test no participa) y sube
la **curva de validación** (RMSE train vs val contra el hiperparámetro) como
artefacto del padre. Buscables: `n_estimators`, `max_depth`, `min_samples_leaf`,
`learning_rate`, `imputer`, `normalizer`, `scaler`.

### Corridas de ejemplo (experimento comparativo)

```powershell
python -m wine_quality.modeling.train                                            # 1. baseline lineal
python -m wine_quality.modeling.train --model tree --max-depth 8                 # 2. árbol
python -m wine_quality.modeling.train --model rf   --n-estimators 100            # 3. bosque chico
python -m wine_quality.modeling.train --model rf   --n-estimators 300 --max-depth 12   # 4. bosque grande
python -m wine_quality.modeling.train --model xgb  --n-estimators 150 --learning-rate 0.08  # 5. boosting
python -m wine_quality.modeling.train --model xgb  --n-estimators 300 --learning-rate 0.05  # 6. boosting fino
```

## Ver y comparar los experimentos

El tracking es local: corridas en `mlflow.db` (SQLite) y artefactos en `mlartifacts/`.

```powershell
.venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre <http://127.0.0.1:5000>, entra al experimento
**TC5061_Actividad3_Experimentacion**, selecciona corridas y usa **Compare** para
contrastar hiperparámetros contra métricas (`val_rmse`, `val_mae`, `val_r2`...).
Cada corrida incluye además la figura `figures/evaluation_test.png`, la tabla
`metrics_table.json` y el pipeline entrenado como artefactos.

## Reproducibilidad

- **Semilla:** `--seed` controla la partición estratificada y la aleatoriedad del
  modelo; se registra como parámetro en cada corrida. Misma configuración + misma
  semilla ⇒ mismos resultados.
- **Dependencias:** declaradas con versiones fijadas en [`requirements.txt`](requirements.txt).
- **Datos inmutables:** `data/raw/` nunca se modifica; las particiones generadas se
  guardan en `data/processed/` y todo lo aprendido (imputación, normalización,
  escalado) vive dentro del pipeline, ajustado únicamente con train.
- El notebook de la Actividad 2 se conserva en `notebooks/` **solo como registro del
  EDA**; la lógica reproducible vive en el paquete `wine_quality/`.

## Solución de problemas

**`MlflowException: Detected out-of-date database schema ... run 'mlflow db upgrade'`**
al abrir `mlflow ui`: el comando se ejecutó con un MLflow distinto al del proyecto
(p. ej. uno instalado en el Python global, más antiguo que el de `.venv`, que fue el
que creó `mlflow.db`). **No ejecutes `mlflow db upgrade`** — la base no está
desactualizada, el MLflow lanzado sí. Activa el entorno y vuelve a intentar:

```powershell
.venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

o invoca directamente el ejecutable del entorno:

```powershell
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

## Organización del proyecto

```
├── MLproject          <- Definición del proyecto MLflow (entry points `main` y `tune`)
├── python_env.yaml    <- Entorno que usa `mlflow run` si no se pasa --env-manager local
├── Makefile           <- Comandos de conveniencia del template CCDS
├── README.md          <- Este archivo
├── requirements.txt   <- Dependencias del proyecto
├── pyproject.toml     <- Metadatos del paquete `wine_quality`
│
├── data               <- (no versionado) raw/ = CSV originales de UCI; processed/ = particiones
├── models             <- Modelos serializados (los de MLflow viven en mlartifacts/)
├── notebooks          <- Notebook de la Actividad 2 (EDA y modelo base; solo exploración)
├── references         <- Material explicativo (investigación de patrones de diseño, etc.)
├── reports            <- Salidas generadas
│   └── figures        <- Figuras de evaluación de cada corrida
│
└── wine_quality       <- Código fuente del proyecto
    ├── config.py              <- Rutas, constantes y configuración de MLflow
    ├── dataset.py             <- Carga, limpieza y partición 70/15/15 estratificada
    ├── features.py            <- Preprocesador configurable (imputación/normalización/escalado/one-hot)
    ├── plots.py               <- Figuras: evaluación, curva de iteraciones y curva de validación
    └── modeling
        ├── models.py          <- Registro factory de modelos y sus hiperparámetros
        ├── train.py           <- Script de entrenamiento parametrizado + MLflow Tracking
        ├── tune.py            <- Sweep de un hiperparámetro (corridas anidadas + curva de validación)
        └── predict.py         <- Inferencia batch con un modelo registrado en MLflow
```

## Declaración de uso de inteligencia artificial

Anthropic. (2026). *Claude Code (basado en Claude Opus 5.5 y Claude Fable 5)* [Modelo de
lenguaje grande], utilizado para edición de código y gestión del repositorio.
https://claude.com/claude-code
