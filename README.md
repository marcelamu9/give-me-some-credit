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

## Estado

| # | Sub-proyecto | Estado |
|---|---|---|
| 0 | Cimientos | ✅ |
| 1 | Datos + EDA | ✅ |
| 2 | Modelado | pendiente |
| 3 | Política de decisión | pendiente |
| 4 | API | pendiente |
| 5 | Monitoreo | pendiente |
