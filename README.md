# Credit risk — Give Me Some Credit

Modelo de probabilidad de **mora severa (90+ días en los próximos 2 años)** servido por una API
que devuelve probabilidad, nivel de riesgo, decisión sugerida y razones principales.
Contexto de negocio: [docs/business_context.md](docs/business_context.md).

## Inicio rápido

Requisitos: [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
uv run poe check      # lint + tests
```

### Datos

Descarga `cs-training.csv`, `cs-test.csv`, `sampleEntry.csv` y `Data Dictionary.xls` desde
[Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) y colócalos en
`data/raw/`. Los datos no se versionan.

## Estructura

```
configs/params.yaml        parámetros (semilla, rutas, costos)
src/credit_risk/           paquete: config, data, features, models, evaluation, decision, monitoring, api
tests/                     tests (pytest)
notebooks/                 exploración (la lógica vive en src/)
data/{raw,interim,processed,external}/   datos locales, no versionados
docs/                      negocio, constitución, diccionario, ADRs, specs y planes
```

## Cómo se trabaja

Desarrollo guiado por specs con IA: cada sub-proyecto tiene spec → plan → implementación,
trazable en `git log`. Reglas para agentes en [AGENTS.md](AGENTS.md); principios en
[docs/constitution.md](docs/constitution.md); uso de IA en [docs/ai_usage_log.md](docs/ai_usage_log.md).

## API

Requisito: haber entrenado el modelo (`uv run poe train`), que genera `models/champion.joblib`.

```bash
uv run poe serve        # http://127.0.0.1:8000
```

Abre `http://127.0.0.1:8000/docs`. Guion de la demo:

1. `GET /health`: la API está viva y el modelo cargado.
2. `POST /predict` con el ejemplo `low_risk` → `approve`, probabilidad 0.0089 (bajo riesgo) y 3 razones.
3. `POST /predict` con `same_client_with_delays` (2 atrasos de 30–59 días y tarjeta al 95 %) → `reject`, probabilidad 0.1728; las razones pasan a ser los atrasos y el uso de la tarjeta.
4. `POST /predict` con `invalid_age` (`age: 10`) → 422 con el detalle del campo, sin llegar al modelo.

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Estado y versión del modelo (503 si no está cargado) |
| `POST /predict` | Probabilidad, decisión (`approve`/`review`/`reject`), nivel de riesgo y 3 razones |
| `GET /model-info` | Versión, umbrales, costos, métricas de test y variables |

**Cómo leer la respuesta:**
- `MonthlyIncome` y `NumberOfDependents` son opcionales: si no se envían (o llegan como `null`), el pipeline los imputa igual que en el entrenamiento.
- Los campos numéricos deben ir como números JSON: `"30"` (texto) o `true` se rechazan con 422.
- Las **razones** son las variables que más suben el riesgo de ese cliente según SHAP, promediado sobre las 5 copias calibradas del modelo. Indican **dirección y orden**, no cuántos puntos de probabilidad aporta cada una (SHAP se calcula antes de la calibración isotónica).
- Latencia medida en local: unos 100 ms por predicción, con los explicadores SHAP creados una sola vez al cargar el modelo.

## Estado

| # | Sub-proyecto | Estado |
|---|---|---|
| 0 | Cimientos | ✅ |
| 1 | Datos + EDA | ✅ |
| 2 | Modelado | ✅ (hecho junto con el 3) |
| 3 | Política de decisión | ✅ (hecho junto con el 2) |
| 4 | API | ✅ |
| 5 | Monitoreo | pendiente |
