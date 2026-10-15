# Spec 000 — Cimientos del proyecto

- **Fecha:** 2026-10-14
- **Estado:** aprobada (2026-10-14)
- **Sub-proyecto:** 0 de 6 (Cimientos → Datos/EDA → Modelado → Decisión → API → Monitoreo)

## 1. Objetivo

Dejar el repositorio listo para que cada sub-proyecto siguiente pueda arrancar en minutos durante
un live coding: contexto de negocio y reglas escritas, entorno reproducible, calidad automatizada
y dataset correcto disponible. Este sub-proyecto **no** incluye EDA, features ni modelos.

## 2. Contexto y decisiones heredadas

| Tema | Decisión |
|---|---|
| Dataset | Give Me Some Credit (Kaggle), target `SeriousDlqin2yrs` |
| Entorno | Python 3.11, uv, ruff, pytest |
| API (futuro) | FastAPI + Pydantic |
| MLOps (futuro) | MLflow local, logging de predicciones, PSI, GitHub Actions |
| Flujo IA | Superpowers: brainstorming → spec → writing-plans → ejecución |

## 3. Alcance

### 3.1 Control de versiones
- Repo ya inicializado (rama `master`, sin commits) → renombrar a `main`; `.gitattributes` con `eol=lf`; `.gitignore` para Python, `data/**` (excepto `.gitkeep`), `mlruns/`, `.venv/`, notebooks checkpoints, `models/*.pkl|*.joblib`.
- Primer commit = estado actual + esta spec, para que el historial muestre spec → plan → implementación.

### 3.2 Documentación (fuente única por tema)
| Archivo | Contenido | Audiencia |
|---|---|---|
| `CLAUDE.md` | `@AGENTS.md` | Claude Code |
| `AGENTS.md` | Reglas imperativas de trabajo + punteros a docs. Sin contexto de negocio extenso. | Cualquier agente |
| `docs/constitution.md` | Principios no negociables (ver 3.3) | Humanos y agentes |
| `docs/business_context.md` | Target, costos FN/FP supuestos, 3 bandas, salida de la API, consumidores | Humanos y agentes |
| `docs/data_dictionary.md` | 10 variables de GMSC + target, tipos, rangos esperados, problemas conocidos | Humanos y agentes |
| `docs/decisions/ADR-001-dataset.md` | Por qué GMSC y no German Credit | Evaluadores |
| `docs/ai_usage_log.md` | Plantilla: fecha · tarea · qué se delegó · cómo se validó · correcciones | Evaluadores |
| `README.md` | Qué es, cómo instalar/correr (`make`), estructura, estado por sub-proyecto | Evaluadores |

### 3.3 Principios de la constitución
1. Reproducibilidad: `uv.lock` versionado, semilla global en `configs/params.yaml`.
2. El test set se separa una vez y no se usa hasta la evaluación final.
3. Sin leakage: todo ajuste (imputación, escalado, binning) se hace dentro de pipelines entrenados solo con train.
4. Lógica reutilizable en `src/credit_risk/`; los notebooks solo exploran y llaman a `src/`.
5. TDD para código en `src/`; ningún resultado se afirma sin evidencia ejecutada.
6. Un commit por tarea; cada sub-proyecto tiene spec y plan antes de código.
7. Parámetros de negocio (costos, umbrales) en `configs/params.yaml`, nunca hardcodeados.

### 3.4 Estructura y entorno
```
src/credit_risk/{__init__.py, config.py, data/, features/, models/, evaluation/, decision/, monitoring/, api/}
tests/test_config.py
configs/params.yaml
data/{raw,interim,processed,external}/.gitkeep
notebooks/  reports/figures/  models/
pyproject.toml  uv.lock  .python-version  .pre-commit-config.yaml (ruff)
.github/workflows/ci.yml   # uv sync --locked → poe lint → poe test
```
- `config.py`: carga `params.yaml` y expone rutas del proyecto. Es el único código de este sub-proyecto (con su test).
- Tareas con **poethepoet** en `pyproject.toml` (no hay `make` en Windows): `uv sync` (setup), `uv run poe lint`, `uv run poe format`, `uv run poe test`, `uv run poe check`.
- Dependencias base: pandas, numpy, scikit-learn, pyyaml; dev: pytest, ruff, pre-commit, poethepoet. El resto se añade en su sub-proyecto (YAGNI).
- Idioma: documentación en español; código, identificadores y mensajes de commit en inglés.

### 3.5 Datos
- Ya presentes en `data/raw/` (descarga manual desde Kaggle): `cs-training.csv`, `cs-test.csv`,
  `sampleEntry.csv`, `Data Dictionary.xls`. German Credit ya fue retirado; ADR-001 documenta el cambio.
- `data/raw/` es inmutable y no se versiona; el README documenta cómo obtener los archivos.
- Se usa solo `cs-training.csv` (tiene target); `cs-test.csv` no tiene etiquetas y no sirve para evaluar.
- La primera columna del CSV es un índice sin nombre (ID de fila), no una variable.
- `docs/data_dictionary.md` se redacta a partir de `Data Dictionary.xls` + inspección del CSV.

## 4. Fuera de alcance
EDA, split, features, modelos, MLflow, API, monitoreo, Docker, despliegue en la nube.

## 5. Criterios de aceptación
- [x] `uv sync && uv run poe check` pasa en limpio en una máquina nueva.
- [ ] El CI corre en verde en GitHub tras el primer push (push manual por la usuaria; aún no hay remoto).
- [ ] Una sesión nueva de Claude Code muestra `AGENTS.md` cargado vía `CLAUDE.md` (`/memory`).
- [x] `data/raw/cs-training.csv` existe con 150.000 filas y 12 columnas (verificado) y no está versionado en git.
- [x] `git log` muestra al menos: commit de spec, commit de plan, commits de implementación.
- [x] Ningún documento duplica contenido de otro (negocio solo en `business_context.md`, reglas solo en `AGENTS.md`/constitución).

## 6. Supuestos abiertos (a resolver en sub-proyectos posteriores)
- Matriz de costos FN:FP: se propone 5:1 como punto de partida en `business_context.md`, con análisis de sensibilidad en el sub-proyecto 3.
- Métricas finales de selección: propuesta en `business_context.md` (AUC/Gini/KS, PR-AUC, Brier, costo esperado), confirmación en el sub-proyecto 2.
