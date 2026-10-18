# Spec 002 — Modelado + Política de decisión

- **Fecha:** 2026-10-17
- **Estado:** aprobada (2026-10-17)
- **Sub-proyecto:** 2 de 5 en el plan del día (Cimientos → Datos/EDA → **Modelado + Decisión** → API). Une los
  antiguos sub-proyectos "Modelado" y "Política de decisión" (van encadenados: probabilidad calibrada → umbrales por costo).
- **Depende de:** spec 001 (`data/interim/{train,valid,test}.csv`, `docs/eda_findings.md`), `credit_risk.config`.

## 1. Intención

Entregar un modelo de probabilidad de mora **defendible ante el equipo de riesgo**. En palabras de la usuaria, poder explicar:

- cómo se trataron los valores faltantes y atípicos;
- que se probó un baseline y se comparó con otros modelos y distintos hiperparámetros ("no se tomó el primer modelo");
- que la elección tuvo en cuenta tiempos y costos;
- que la métrica es defendible frente al objetivo de costos del negocio;
- que el modelo deja sustento documentado para decisiones futuras.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos**
- Pipeline de preprocesamiento reproducible, ajustado solo con train, que aplica los tratamientos decididos.
- Baseline (regresión logística) y dos modelos adicionales, con búsqueda de hiperparámetros y tiempos medidos.
- Selección del campeón por costo esperado en validación, con calibración verificada.
- Política de 3 bandas (aprobar / revisión / rechazar) con umbrales por costo y análisis de sensibilidad.
- Evaluación única y final del campeón en test.
- Artefacto del modelo + metadata, y una ficha del modelo (`docs/model_card.md`).

**No-objetivos**
- La API (sub-proyecto siguiente). Aquí solo se deja el artefacto listo para cargarse.
- Feature engineering más allá de los indicadores y recodificaciones decididos.
- Monitoreo de drift.
- Re-entrenar con train + valid antes de test (los umbrales quedan ligados a validación).

**Anti-objetivos**
- Que el test influya en cualquier elección (modelo, hiperparámetros, calibración o umbrales).
- Que algún tratamiento se ajuste con valid o test (medianas, percentiles, escalado).
- Elegir el modelo solo por AUC, sin mirar costo ni calibración.
- Cifras en la ficha o en los reportes que no salgan de una ejecución real.

## 3. Tratamiento de faltantes y atípicos

Todo se implementa como transformadores de scikit-learn dentro del pipeline: **se ajustan con train** y se aplican igual a valid, test y, más adelante, a la API.

### 3.1 Decisiones de la usuaria (de `docs/eda_findings.md`)

| # | Variable | Tratamiento |
|---|---|---|
| U1 | Duplicados contradictorios | Se conservan |
| U2 | Códigos 96/98 | Indicador `has_special_past_due_code` (1 si cualquiera de las 3 variables de atraso es 96/98) y los 3 conteos con 96/98 pasan a 0 |
| U3 | `MonthlyIncome` faltante | Mediana por grupo de edad × dependientes, con indicador de faltante y respaldo |
| U4 | `RevolvingUtilizationOfUnsecuredLines` | Recorte en el p99 de train |
| U5 | `DebtRatio` | Opción B: en filas sin ingreso, recalcular ratio = monto ÷ ingreso imputado; con ingreso positivo, recorte en el p99 de train de las filas con ingreso positivo |

### 3.2 Decisiones de Claude (las 5 preguntas abiertas y detalles de implementación)

| # | Tema | Decisión | Motivo |
|---|---|---|---|
| C1 | `MonthlyIncome = 0` | Se trata como **no disponible**, igual que el nulo | En esas filas `DebtRatio` es un monto (100 % enteros); no se puede calcular el ratio con ingreso 0 |
| C2 | Indicadores de ingreso | **Un solo indicador** `income_unavailable` (nulo o 0) en lugar de `monthly_income_missing` + `debt_ratio_is_amount` | Con C1 ambos serían idénticos; dos columnas iguales rompen la logística (colinealidad perfecta) |
| C3 | `NumberOfDependents` faltante | Imputar **0** (mediana y moda en train) | Sus 2.451 nulos están todos dentro de `income_unavailable`, que ya captura la señal |
| C4 | Grupos para imputar ingreso | Decil de edad (cortes de train) × dependientes agrupados `0, 1, 2, 3, 4+`. Grupo con < 30 filas en train → respaldo: mediana del decil de edad; si no, mediana global | La relación ingreso-edad es en U invertida; con 6+ dependientes hay < 100 filas por grupo |
| C5 | `MonthlyIncome` extremo | Recorte en el p99 de train (≈ 25.000) del ingreso observado, antes de calcular medianas y ratios | Máximo 3.008.750; evita ratios recalculados absurdos |
| C6 | `age < 18` | Se **eliminan** de train, valid y test (1 fila en train) | Edad imposible para un crédito; la API rechazará esas edades, así que la población del modelo no las incluye |
| C7 | Variables de conteo con colas largas | Recorte en el p99 de train **solo en el pipeline de la logística** | A los árboles no les afectan; recortar en p99 (p. ej. atrasos 30–59 en 3) borraría matices útiles para ellos |
| C8 | Escala para la logística | `log1p` en ingreso y ratios + estandarización, solo en la rama de la logística | La logística es sensible a escala y sesgo; los árboles no |

Orden del pipeline común: eliminar `age < 18` (antes del pipeline) → indicador 96/98 y recodificación → `income_unavailable` → recorte de ingreso observado → imputación de dependientes → imputación de ingreso → recálculo/recorte de `DebtRatio` → recorte de `RevolvingUtilization…`.

## 4. Modelos y búsqueda de hiperparámetros

| Modelo | Papel | Preprocesamiento |
|---|---|---|
| Regresión logística | Baseline (interpretable, estándar en riesgo) | Común + rama logística (C7, C8) |
| Random Forest | Alternativa de árboles en bagging | Común |
| LightGBM | Gradient boosting (estándar de la industria en tabular) | Común |

- Búsqueda: `RandomizedSearchCV` con **validación cruzada estratificada de 5 pliegues sobre train**, `scoring="roc_auc"`, `n_iter = 20` por modelo, semilla de `params.yaml`. La logística busca sobre `C` y penalización.
- Se registra por modelo: mejor AUC de CV (media ± desviación), hiperparámetros, **tiempo de búsqueda, tiempo de entrenamiento final y latencia de predicción** por cliente.
- Seguimiento de experimentos con **MLflow local** (`mlruns/`, no versionado): un run por modelo con parámetros, métricas y tiempos. La evidencia versionada es `reports/model_comparison.md` (ver §7).
- Sin `class_weight`/remuestreo: distorsionan la probabilidad y el costo se gestiona con el umbral.

## 5. Calibración, métricas y selección del campeón

- **Calibración:** cada candidato ajustado se calibra con `CalibratedClassifierCV` (isotónica, 5 pliegues sobre train). Se reporta Brier y curva de calibración en valid, antes y después.
- **Métricas en valid** para cada candidato calibrado: ROC-AUC, Gini, KS, PR-AUC, Brier y **costo esperado por cliente** (con el umbral óptimo binario de §6).
- **Criterio de selección (defendible frente al negocio):**
  1. Menor **costo esperado por cliente** en valid (matriz FN = 5, FP = 1 de `params.yaml`).
  2. Si dos modelos difieren en menos de **2 %** de costo, gana el más simple/rápido (logística < RF < LightGBM en complejidad), salvo que su Brier sea peor en más de 10 %.
  3. Se documenta en un ADR (`docs/decisions/ADR-002-seleccion-modelo.md`) por qué AUC guía la búsqueda (no depende del umbral, es estable) pero el **costo** decide.

## 6. Política de decisión (3 bandas)

- **Umbral de rechazo binario `t*`:** minimiza el costo esperado en valid. Se compara con el umbral teórico para probabilidades bien calibradas, `FP / (FP + FN) = 1/6 ≈ 0,167`.
- **Banda de revisión manual** *(decisión de Claude)*: se define por **capacidad de revisión**, `review_capacity` en `params.yaml` (por defecto **0,10** = el 10 % de las solicitudes de valid). `t_low` se elige para que la franja `[t_low, t*)` contenga esa fracción: `p < t_low` → aprobar, `[t_low, t*)` → revisión, `p ≥ t*` → rechazar. La revisión se coloca **justo por debajo de `t*`**: son los aprobados más riesgosos, donde un analista evita el error más caro (FN = 5).
- **Costo de la política:** el costo esperado se calcula para la política binaria (`t*`). Para las 3 bandas se reporta el volumen y la tasa de mora real de cada banda en valid, sin asumir un costo ni un resultado para la revisión humana.
- **Sensibilidad:** tabla de `t*`, `t_low`, tasa de aprobación, tasa de mora entre aprobados y costo esperado para ratios FN:FP de **3, 5, 10 y 20**; y una segunda tabla con volumen y tasa de mora por banda para capacidades de revisión de **5 %, 10 % y 20 %**.
- **Dónde viven:** las entradas de negocio (`costs`, `review_capacity`, ratios de sensibilidad) en `configs/params.yaml`; los umbrales **aprendidos** en la metadata del modelo (§7). *Esto requiere ajustar el principio 7 de la constitución y `docs/business_context.md`, que hoy dicen que los umbrales viven en `params.yaml` (ver §10).*

## 7. Entregables

| Entregable | Ubicación |
|---|---|
| Transformadores del pipeline (común y rama logística) | `src/credit_risk/features/` |
| Definición de candidatos, espacios de búsqueda, entrenamiento, calibración | `src/credit_risk/models/` |
| Métricas (AUC, Gini, KS, PR-AUC, Brier, costo esperado) | `src/credit_risk/evaluation/` |
| Umbrales, bandas y sensibilidad | `src/credit_risk/decision/` |
| Script de entrenamiento reproducible | `uv run poe train` → entrena, selecciona, fija umbrales y guarda el artefacto (no depende del notebook) |
| Artefacto | `models/champion.joblib` (no versionado) + `models/champion_metadata.json` (versionado: modelo, hiperparámetros, features, umbrales, métricas valid y test, tiempos, versiones de librerías, semilla, fecha, hash de los CSV de entrada) |
| Notebook de análisis | `notebooks/02-modelado.ipynb`: lee resultados y artefacto; comparación, curvas ROC y de calibración, costo vs umbral, sensibilidad, importancia global |
| Comparación de modelos | `reports/model_comparison.md` (generado por el script) |
| Figuras | `reports/figures/` |
| Ficha del modelo | `docs/model_card.md`: propósito, datos, tratamientos y su porqué, candidatos y resultados, criterio de selección, umbrales y sensibilidad, resultado en test, importancia global (SHAP para árboles / coeficientes para la logística), limitaciones y riesgos (incluido el uso de `age`), cuándo reentrenar |
| ADR | `docs/decisions/ADR-002-seleccion-modelo.md` |
| Parámetros nuevos | `params.yaml`: `review_capacity`, `cost_sensitivity_ratios`, `review_capacity_sensitivity`, `search.n_iter`, `search.cv_folds` |

Dependencias nuevas: `lightgbm`, `mlflow`, `shap`, `joblib` (vía scikit-learn).

## 8. Test set

Se abre **una sola vez**, al final del script, para el campeón ya calibrado y con umbrales fijados: AUC, Gini, KS, PR-AUC, Brier, costo esperado y distribución por bandas. Se reporta tal cual, aunque sea peor que valid.

## 9. Lo que queda al criterio de quien implemente

Rangos exactos de los espacios de búsqueda (razonables para cada modelo y con tiempo total de búsqueda acotado a ~15 min en un portátil), nombres internos, estilo de figuras y redacción de reportes, respetando esta spec y la constitución.

## 10. Confirmado por la usuaria (2026-10-17)

1. Umbrales aprendidos en la metadata versionada del modelo; `params.yaml` guarda solo las entradas de negocio. Se enmienda el principio 7 de la constitución y `docs/business_context.md`.
2. `review_capacity = 10 %` como supuesto (como el 5:1), con sensibilidad a 5 %, 10 % y 20 %.
3. Eliminar `age < 18` de las tres particiones.

## 11. Criterios de aceptación

- [ ] `uv run poe check` pasa; tests con datos sintéticos para cada transformador, métrica y función de decisión.
- [ ] Ningún transformador usa estadísticas de valid/test (test explícito: ajustar con un DataFrame y transformar otro no cambia lo aprendido).
- [ ] `uv run poe train` termina, registra los 3 candidatos en MLflow y genera `reports/model_comparison.md` y `models/champion_metadata.json`.
- [ ] Re-ejecutar `uv run poe train` con la misma semilla produce el mismo campeón, los mismos umbrales y las mismas métricas.
- [ ] El campeón se elige por costo esperado en valid según §5, y la tabla de comparación lo muestra.
- [ ] El test se usa una sola vez (visible en el código del script).
- [ ] `notebooks/02-modelado.ipynb` se ejecuta de punta a punta.
- [ ] `docs/model_card.md` y `ADR-002` existen y solo citan cifras de la ejecución real.
- [ ] `AGENTS.md` apunta a esta spec y a su plan como activos.
