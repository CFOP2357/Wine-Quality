# TC5061 · Actividad 3 — Código de experimentación base con MLflow

**Alumno:** Victor Yoguel Salazar Alanis · A01797950
**Curso:** TC5061 – Operaciones de Aprendizaje Automático (MLOps) · Tecnológico de Monterrey
**Repositorio:** <https://github.com/CFOP2357/Wine-Quality>

---

## 1. Ejecución del script

El notebook de la Actividad 2 se migró al paquete `wine_quality/` (carga y partición en
`dataset.py`, preprocesamiento en `features.py`, selección de modelo en `modeling/models.py`
y entrenamiento en `modeling/train.py`); el notebook se conserva solo para el EDA. El script
corre sin argumentos (defaults: regresión lineal, semilla 42) y expone por `argparse` el
modelo, la semilla, 4 hiperparámetros y el preprocesamiento:

```
python -m wine_quality.modeling.train
python -m wine_quality.modeling.train --model rf --n-estimators 300 --max-depth 12 --normalizer none --scaler none
```

También es ejecutable como proyecto MLflow: `mlflow run . --env-manager local -P model=rf -P n_estimators=300`.
Las instrucciones completas (instalación, datos, tabla de argumentos) están en el README del repositorio.

## 2. Experimento registrado

Todas las corridas se agrupan en el experimento **TC5061_Actividad3_Experimentacion**
(tracking local en SQLite). Cada corrida registra, dentro de un `mlflow.start_run()`: los
hiperparámetros efectivos del modelo (`log_params`), 16 métricas (`log_metric`: MAE, RMSE y R²
en train/val/test, la referencia dummy y la brecha de overfitting), dos artefactos
(`log_artifact`: figura real-vs-predicho y tabla de métricas) y el pipeline entrenado completo
(`mlflow.sklearn.log_model`). El experimento contiene **23 corridas**: 7 corridas base
comparando los cuatro modelos (linear, árbol, random forest, XGBoost) y variando su
preprocesamiento, más la búsqueda de hiperparámetros de XGBoost (descenso por coordenadas,
early stopping con `eval_set` y un sweep de `max_depth` con corridas anidadas).

**Protocolo:** los modelos se compararon y seleccionaron únicamente con las métricas de
**validación**; el RMSE de **test** se reporta como estimación de generalización de la
configuración ya elegida y no participó en ninguna selección.

## 3. Evidencias (capturas de MLflow)

**Captura 1 — Tabla comparativa de corridas** (23 corridas del experimento, con
hiperparámetros y métricas de validación; las hijas del sweep aparecen anidadas bajo
`tune_max_depth_xgb`):

![Tabla comparativa de corridas en MLflow](mlflow_screenshots/mlflow_tabla_corridas.png)

**Captura 2 — Detalle de la corrida campeona** `rf_ne300_md12_msl1_sin-norm_sin-esc`:
las 16 métricas registradas, tags y metadatos de la corrida:

![Detalle de la corrida campeona](mlflow_screenshots/mlflow_corrida_campeon.png)

**Captura 3 — Artefactos de la corrida campeona** (carpeta `figures/` y
`metrics_table.json` con las métricas por partición, incluida la referencia dummy):

![Artefactos de la corrida campeona](mlflow_screenshots/mlflow_campeon_artefactos.png)

**Captura 4 — Curva de validación del sweep** `tune_max_depth_xgb` (RMSE train vs val
contra `max_depth`; mejor valor = 5):

![Curva de validación del sweep de max_depth](mlflow_screenshots/mlflow_curva_validacion_sweep.png)

## 4. Análisis de resultados

**Mejor configuración.** El campeón por validación fue **Random Forest con 300 árboles,
profundidad máxima 12 y sin normalización ni escalado** (`rf_ne300_md12_msl1_sin-norm_sin-esc`):
val RMSE **0.6935**, y en test —evaluado una sola vez, ya elegido— RMSE **0.6977**, MAE 0.538 y
R² 0.376, muy por encima de la referencia trivial de predecir la media (test RMSE 0.8831). Las
tres variantes de random forest quedaron estadísticamente empatadas (val RMSE 0.6935–0.6939),
de modo que la evidencia sostiene "random forest con ~200–400 árboles" más que una combinación
exacta. El mejor XGBoost afinado (profundidad 5, lr 0.05, 165 árboles por early stopping) quedó
cerca (val 0.6993) sin alcanzarlo; el lineal (0.7410) y el árbol (0.7483) quedaron claramente atrás.

**Hiperparámetro más influyente.** La decisión que más movió las métricas fue la **familia de
modelo**: pasar de lineal/árbol a un ensamble redujo el val RMSE ~0.05 (≈7%). Dentro de XGBoost,
el hiperparámetro individual más influyente fue **max_depth** (val RMSE 0.7094→0.6993 entre
profundidad 2 y 5, con curva en U confirmada por el sweep), mientras `learning_rate` y
`n_estimators` solo movieron milésimas una vez fijada la profundidad con early stopping. Dos
hallazgos secundarios con valor práctico: quitar la normalización y el escalado **no afectó** a
los modelos de árboles (el campeón no los usa), confirmando que ese preprocesamiento beneficia
a los modelos lineales; y el diagnóstico `overfit_gap_pct` mostró brechas train-val enormes en
los ensambles (67–172%) **sin** costo en validación: memorización benigna, no overfitting dañino
— la curva train-vs-val por iteración lo evidencia (val se aplana, no empeora).

**Limitaciones.** (1) Con 798 registros en validación, diferencias de milésimas de RMSE (como
el empate entre los rf) no son significativas; la selección entre ellos es efectivamente
arbitraria. (2) La validación se reutilizó muchas veces durante la búsqueda, por lo que el val
RMSE del campeón tiene un sesgo optimista de selección; por eso se reporta test (0.6977) como
estimación honesta. (3) Todas las corridas comparten una única partición (semilla 42): no se
midió la varianza entre particiones (validación cruzada quedaría como siguiente paso). (4) El
techo de desempeño (R² ≈ 0.38) sugiere que la calidad percibida depende de factores que las 11
variables fisicoquímicas no capturan, un límite del dataset más que de los modelos.

---

**Reproducibilidad:** semilla fija y registrada por corrida; `requirements.txt` con versiones
fijadas; README con instrucciones exactas de ejecución.

**Declaración de uso de IA:** Anthropic. (2026). *Claude Code (Claude Opus 5.5 y Claude Fable 5)*
[Modelo de lenguaje grande], utilizado para edición de código y gestión del repositorio bajo un
esquema *human-in-the-loop* (decisiones dirigidas, revisadas y aprobadas por el autor).
https://claude.com/claude-code
