# Spec 003 — API de inferencia (FastAPI)

- **Fecha:** 2026-10-18
- **Estado:** aprobada (2026-10-18)
- **Sub-proyecto:** 3 del plan del día (Cimientos → Datos/EDA → Modelado + Decisión → **API**)
- **Depende de:** spec 002: `models/champion.joblib` (`{"model", "features", "t_low", "t_star"}`), `models/champion_metadata.json`, `credit_risk.decision.policy` (`assign_band`, `RISK_LEVEL`), `credit_risk.features.treatment`.

## 1. Intención

Exponer el modelo campeón para que otro sistema lo use sin conocer Python, y poder **demostrarlo en vivo desde `/docs`** en la prueba técnica. Guion de la demo (de la usuaria):

1. `GET /health`: la API está viva y el modelo cargado.
2. `POST /predict` con un cliente de bajo riesgo → `approve`, probabilidad baja y 3 razones.
3. El mismo cliente con atrasos de 30–59 días y la tarjeta al 95 % → `review` o `reject`, y razones que cambian de forma lógica.
4. Un cliente inválido (`age: 10` o texto en un campo numérico) → error claro, sin llegar al modelo.
5. Si hay tiempo, `GET /model-info`: versión, umbrales y métricas.

La usuaria quiere entender tres piezas de FastAPI: **validación con Pydantic, carga única del modelo y tests con `TestClient`**. El código debe ser lo bastante claro para explicarlas.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos:** los tres endpoints, validación de entrada, razones por cliente con SHAP, ejemplos de la demo precargados en `/docs`, tests que corren en el CI sin el modelo real.

**No-objetivos:** predicción por lotes, autenticación, Docker, despliegue en la nube, registro de predicciones para monitoreo, reentrenar o cambiar el modelo.

**Anti-objetivos:**
- Que la API aplique un tratamiento distinto al del entrenamiento (debe usar el mismo pipeline del artefacto).
- Que un dato inválido llegue al modelo o produzca un error 500.
- Umbrales, costos o rutas hardcodeados en la API.
- Que las razones expliquen un modelo distinto al que produjo la probabilidad.

## 3. Endpoints y contratos

### `GET /health`
- 200 `{"status": "ok", "model_loaded": true, "model_version": "<trained_at>"}` si el modelo está cargado.
- 503 `{"status": "unavailable", "model_loaded": false, "model_version": null}` si no.

### `POST /predict`
**Entrada (`ClientFeatures`)**: los 10 campos con su **nombre original de Kaggle** en el JSON (alias de Pydantic; en Python, nombres en snake_case).

| Campo JSON | Tipo | Regla |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | float | ≥ 0 |
| `age` | int | 18–120 |
| `NumberOfTime30-59DaysPastDueNotWorse` | int | 0–98 (acepta los códigos 96/98) |
| `DebtRatio` | float | ≥ 0 |
| `MonthlyIncome` | float o `null` | ≥ 0; `null` o 0 → el pipeline lo trata como no disponible |
| `NumberOfOpenCreditLinesAndLoans` | int | 0–98 |
| `NumberOfTimes90DaysLate` | int | 0–98 |
| `NumberRealEstateLoansOrLines` | int | 0–98 |
| `NumberOfTime60-89DaysPastDueNotWorse` | int | 0–98 |
| `NumberOfDependents` | int o `null` | 0–20; `null` → el pipeline lo imputa |

Campos extra → 422 (`extra="forbid"`): evita que un nombre mal escrito se ignore en silencio.

**Salida (`PredictionResponse`):**

| Campo | Tipo | Contenido |
|---|---|---|
| `probability` | float | Probabilidad calibrada de mora severa, 4 decimales |
| `risk_level` | `low` / `medium` / `high` | `RISK_LEVEL[decision]` |
| `decision` | `approve` / `review` / `reject` | `assign_band(p, t_low, t_star)` |
| `top_reasons` | lista de 3 `{feature, description}` | Variables que más **suben** el riesgo de ese cliente (§4) |
| `model_version` | str | `trained_at` de la metadata |

Errores: 422 automático de FastAPI para entrada inválida; 503 si el modelo no está cargado.

### `GET /model-info`
Desde la metadata: `model_version`, `champion`, `thresholds` (`t_low`, `t_star`, `bayes_threshold`), `costs`, `review_capacity`, `test_metrics`, `features`. 503 si no hay modelo.

## 4. Razones por cliente (SHAP)

- El modelo calibrado son **5 copias de LightGBM** (una por pliegue de `CalibratedClassifierCV`), y la probabilidad es su promedio. Las razones se calculan igual: SHAP (`TreeExplainer`) en **cada una de las 5 copias** sobre la fila transformada por su propio preprocesamiento, y se **promedia** la contribución de cada variable.
- Si el campeón fuera la logística, la contribución es coeficiente × valor estandarizado, promediada igual. Debe funcionar para cualquiera de los tres candidatos.
- Se ordenan las contribuciones **positivas** (suben el riesgo) de mayor a menor y se toman 3. Si hay menos de 3 positivas, se completan con las de mayor valor absoluto.
- Las 12 variables del modelo se mapean a su nombre original y a una descripción en español (*decisión de Claude, revisable*):

| Variable del modelo | `feature` devuelto | `description` |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | igual | Uso alto de tarjetas y líneas de crédito |
| `age` | igual | Edad |
| `NumberOfTime30-59DaysPastDueNotWorse` | igual | Atrasos de 30 a 59 días |
| `NumberOfTime60-89DaysPastDueNotWorse` | igual | Atrasos de 60 a 89 días |
| `NumberOfTimes90DaysLate` | igual | Atrasos de 90 días o más |
| `DebtRatio` | igual | Nivel de endeudamiento frente al ingreso |
| `MonthlyIncome` | igual | Ingreso mensual |
| `NumberOfOpenCreditLinesAndLoans` | igual | Número de créditos abiertos |
| `NumberRealEstateLoansOrLines` | igual | Créditos inmobiliarios |
| `NumberOfDependents` | igual | Número de dependientes |
| `income_unavailable` | `MonthlyIncome` | Ingreso no informado |
| `has_special_past_due_code` | `NumberOfTimes90DaysLate` | Historial de atrasos con código especial (96/98) |

  Refinamientos tras la revisión final: (1) `income_unavailable` y `has_special_past_due_code` solo se muestran si su indicador vale 1 para ese cliente, y si falta el ingreso se muestra "Ingreso no informado" en lugar de `MonthlyIncome`; (2) las descripciones no afirman riesgo alto (p. ej. "Uso de tarjetas y líneas de crédito"), porque las razones de relleno pueden bajar el riesgo; (3) los valores SHAP se calculan antes de la calibración isotónica, así que indican dirección y orden, no suman la probabilidad devuelta.
  Si dos razones apuntan al mismo `feature` devuelto, se conserva la de mayor contribución y se toma la siguiente.

## 5. Carga del modelo y configuración

- El artefacto y la metadata se cargan **una sola vez al arrancar** (lifespan de FastAPI) desde `cfg.paths.models`, y quedan en el estado de la app. Si faltan, la app arranca igual y responde 503 (§3).
- La app se construye con una fábrica `create_app(model_dir: Path | None = None)`, para que los tests puedan apuntar a un directorio con un modelo sintético.
- Arranque: `uv run poe serve` → `uvicorn credit_risk.api.main:app` en `localhost:8000`.

## 6. Ejemplos de la demo en `/docs`

Tres ejemplos seleccionables en `POST /predict` (`openapi_examples`), definidos en `api/examples.py`:

| Ejemplo | Perfil | Resultado esperado con el modelo real |
|---|---|---|
| `low_risk` | Uso de tarjeta bajo, sin atrasos, ingreso medio, edad media | `approve` |
| `same_client_with_delays` | Igual que `low_risk`, pero con 2 atrasos de 30–59 días y uso de tarjeta al 95 % | `review` o `reject` |
| `invalid_age` | Igual que `low_risk` con `age: 10` | 422 |

Los valores concretos los fija quien implemente y **se verifican contra el modelo real** en local. Si un ejemplo no da el resultado esperado, se ajustan sus valores (de forma realista), no el modelo ni los umbrales.

## 7. Tests

- Con `TestClient` y un **modelo sintético** pequeño (mismo pipeline y `CalibratedClassifierCV`, entrenado en segundos con datos sintéticos y guardado en `tmp_path` con la misma estructura de artefacto y metadata). El CI no tiene el modelo real.
- Cubren: health con y sin modelo; predict válido (campos, rangos, 3 razones, decisión coherente con los umbrales); 422 por edad < 18, texto en campo numérico, campo faltante obligatorio y campo extra; `null` en `MonthlyIncome` y `NumberOfDependents` aceptado; 503 sin modelo; model-info; razones promediadas sobre las 5 copias; mapeo y deduplicación de razones.
- Verificación local aparte (no en el CI): los 3 ejemplos de §6 contra el modelo real.

## 8. Entregables

| Entregable | Ubicación |
|---|---|
| Contratos Pydantic | `src/credit_risk/api/schemas.py` |
| Carga del artefacto, predicción y razones | `src/credit_risk/api/service.py` |
| App, lifespan y endpoints | `src/credit_risk/api/main.py` |
| Ejemplos de la demo | `src/credit_risk/api/examples.py` |
| Tests | `tests/test_api_*.py` |
| Guion de la demo | sección "API" en `README.md` (arranque y los 4 pasos) |
| Dependencias | `fastapi`, `uvicorn`; dev: `httpx` |
| Tarea | `serve` en `[tool.poe.tasks]` |

## 9. Decisiones de Claude aprobadas por la usuaria (2026-10-18)

Nombres originales de Kaggle con alias; `MonthlyIncome` y `NumberOfDependents` opcionales; ejemplos precargados; razones como objetos `{feature, description}` en español; SHAP promediado sobre las 5 copias; `model_version` en cada respuesta; 503 si falta el modelo; probabilidad con 4 decimales; conteos 0–98 aceptando 96/98; tests con modelo sintético.

## 9.1 Enmienda (2026-10-18, tras la revisión final)

Aprobada por la usuaria tras la revisión de los pendientes de la ejecución:

1. `MonthlyIncome` y `NumberOfDependents` se pueden **omitir** (valor por defecto `null`), además de enviarse como `null`.
2. **Modo estricto** en `ClientFeatures`: números enviados como texto (`"30"`) o booleanos → 422. Los enteros se siguen aceptando en campos decimales.
3. La razón del código especial 96/98 muestra como `feature` la **columna de atraso que trae el código** (la más severa si hay varias), no siempre `NumberOfTimes90DaysLate`. Corrige un supuesto de esta spec: en los datos el código llega en las tres columnas a la vez, pero la API acepta cada una por separado.
4. Los explicadores SHAP se crean **una vez al cargar el modelo** (`LoadedModel.explainers`): la latencia de `/predict` bajó de ≈ 950 ms a ≈ 103 ms en local.
5. Documentado en el `README.md`: las razones indican dirección y orden, no puntos de probabilidad (SHAP se calcula en log-odds de cada copia, antes de la calibración isotónica).

## 10. Criterios de aceptación

- [x] `uv run poe check` pasa, incluidos los tests de la API sin el modelo real.
- [x] `uv run poe serve` arranca y `/docs` muestra los 3 endpoints y los 3 ejemplos.
- [x] Contra el modelo real: `low_risk` → `approve`; `same_client_with_delays` → `review` o `reject`; `invalid_age` → 422.
- [x] La probabilidad de `/predict` coincide con `model.predict_proba` del artefacto para la misma fila.
- [x] Las razones salen del promedio de las 5 copias (test explícito).
- [x] `README.md` explica cómo arrancar la API y el guion de la demo.
- [x] `AGENTS.md` apunta a esta spec y a su plan como activos.
