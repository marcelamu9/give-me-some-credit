# Cimientos del proyecto — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar el repo listo para live coding: git limpio, entorno uv reproducible, calidad automatizada (ruff + pytest + CI), configuración central testeada y documentación de negocio/reglas sin duplicados.

**Architecture:** Paquete `credit_risk` con src-layout instalado en editable por uv/hatchling. Un único módulo de código (`credit_risk/config.py`) carga `configs/params.yaml` y resuelve rutas contra la raíz del proyecto. Las tareas de desarrollo se exponen con poethepoet (no hay `make` en Windows) y el CI ejecuta exactamente las mismas tareas.

**Tech Stack:** Python 3.11 (vía uv), uv, hatchling, pyyaml, pandas, numpy, scikit-learn, pytest, ruff, pre-commit, poethepoet, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-14-cimientos-design.md`

## Global Constraints

- Python: `requires-python = ">=3.11,<3.14"`, `.python-version` = `3.11`.
- Entorno: uv; `uv.lock` versionado; CI usa `uv sync --locked`.
- Tareas: `uv run poe lint | format | test | check` (no Makefile).
- Ruff: `line-length = 100`, `target-version = "py311"`, reglas `E F W I B UP SIM`.
- Dependencias base: pandas, numpy, scikit-learn, pyyaml. Dev: pytest, ruff, pre-commit, poethepoet. Nada más (YAGNI).
- `data/**` nunca se versiona (salvo `.gitkeep`); `data/raw/` es inmutable: no modificar ni mover sus archivos.
- Solo `cs-training.csv` tiene target; `cs-test.csv` no se usa para evaluar.
- Ningún test puede depender de `data/raw/` (el CI no tiene los datos).
- Parámetros de negocio (costos) solo en `configs/params.yaml`.
- Idioma: documentación en español; código, identificadores y mensajes de commit en inglés (Conventional Commits).
- Rama principal: `main`. Un commit por tarea. No hacer `git push` (no hay remoto; lo hace la usuaria).
- Terminal del entorno: Windows. Los comandos de este plan funcionan en Git Bash y PowerShell salvo que se indique.

## Review Focus

1. Ejecutar código desde otro directorio (p. ej. un notebook en `notebooks/`) → las rutas de `load_config()` deben resolverse contra la raíz del proyecto, no contra el cwd. (Test en Task 3.)
2. `params.yaml` vacío o con una clave/sección faltante → `ConfigError` que nombre la clave que falta, no un `KeyError`/`TypeError` críptico. (Tests en Task 3.)
3. Costos no positivos o no numéricos en `params.yaml` → `ConfigError`. (Tests en Task 3.)
4. Archivos de datos que se cuelan en git (incluido `Data Dictionary.xls`) → `git check-ignore` debe confirmarlos ignorados. (Verificación en Task 1.)
5. Finales de línea CRLF en Windows (`core.autocrlf=true`) que hagan fallar `ruff format --check` en el CI Linux → `.gitattributes` con `eol=lf`. (Task 1 + verificación en Task 5.)

---

### Task 1: Higiene del repositorio y commits de spec y plan

**Files:**
- Create: `.gitignore`, `.gitattributes`
- Commit (ya existen): `AGENTS.md`, `README.md`, `docs/constitution.md`, `.claude/settings.json`, `docs/superpowers/specs/2026-10-14-cimientos-design.md`, `docs/superpowers/plans/2026-10-14-cimientos.md`

**Interfaces:**
- Consumes: nada.
- Produces: rama `main` con dos commits (spec, plan); reglas de ignore que tareas posteriores asumen.

- [ ] **Step 1: Renombrar la rama**

Run: `git branch -m master main`
Expected: `git branch --show-current` imprime `main`.

- [ ] **Step 2: Crear `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/

# Notebooks
.ipynb_checkpoints/

# Datos: raw es inmutable y no se versiona
data/**
!data/**/
!data/**/.gitkeep

# Artefactos de modelos y experimentos
mlruns/
mlartifacts/
models/*.pkl
models/*.joblib

# Entorno local
.env
.claude/settings.local.json
```

- [ ] **Step 3: Crear `.gitattributes`**

```gitattributes
* text=auto eol=lf
*.xls binary
*.png binary
```

- [ ] **Step 4: Verificar que los datos quedan ignorados**

Run: `git check-ignore -v "data/raw/cs-training.csv" "data/raw/cs-test.csv" "data/raw/sampleEntry.csv" "data/raw/Data Dictionary.xls"`
Expected: 4 líneas, todas apuntando a la regla `data/**` de `.gitignore`.

Run: `git status --porcelain`
Expected: ningún archivo bajo `data/` aparece.

- [ ] **Step 5: Commit de la spec**

```bash
git add .gitignore .gitattributes AGENTS.md README.md docs/constitution.md .claude/settings.json docs/superpowers/specs/2026-10-14-cimientos-design.md
git commit -m "docs: add foundations spec and repo hygiene"
```

- [ ] **Step 6: Commit del plan**

```bash
git add docs/superpowers/plans/2026-10-14-cimientos.md
git commit -m "docs: add foundations implementation plan"
```

Run: `git log --oneline`
Expected: 2 commits, el de spec primero.

---

### Task 2: Scaffolding del proyecto Python (uv, ruff, pytest, poe, pre-commit)

**Files:**
- Create: `pyproject.toml`, `.python-version` (vía uv), `uv.lock` (vía uv), `.pre-commit-config.yaml`
- Create: `src/credit_risk/__init__.py` y `src/credit_risk/{data,features,models,evaluation,decision,monitoring,api}/__init__.py`
- Create: `.gitkeep` en `data/raw`, `data/interim`, `data/processed`, `data/external`, `notebooks`, `models`, `reports/figures`, `configs`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `.gitignore` de Task 1.
- Produces: paquete importable `credit_risk`; tareas `uv run poe lint|format|test|check` usadas por Task 3, 5 y 6.

- [ ] **Step 1: Fijar Python 3.11**

Run: `uv python pin 3.11`
Expected: crea `.python-version` con `3.11` (uv descarga el intérprete si no existe).

- [ ] **Step 2: Crear la estructura de directorios y paquetes**

En Git Bash:
```bash
mkdir -p src/credit_risk/{data,features,models,evaluation,decision,monitoring,api} tests configs notebooks models reports/figures data/{raw,interim,processed,external}
touch src/credit_risk/__init__.py src/credit_risk/{data,features,models,evaluation,decision,monitoring,api}/__init__.py
touch data/{raw,interim,processed,external}/.gitkeep notebooks/.gitkeep models/.gitkeep reports/figures/.gitkeep configs/.gitkeep
```
No tocar los archivos existentes de `data/raw/`.

Contenido de `src/credit_risk/__init__.py`:
```python
"""Credit risk: probability of serious delinquency (90+ days within 2 years)."""
```
Los `__init__.py` de subpaquetes quedan vacíos.

- [ ] **Step 3: Crear `pyproject.toml`**

```toml
[project]
name = "credit-risk"
version = "0.1.0"
description = "Probabilidad de mora severa (90+ días en 2 años) sobre Give Me Some Credit"
readme = "README.md"
requires-python = ">=3.11,<3.14"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/credit_risk"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.poe.tasks]
_ruff-check = "ruff check ."
_ruff-format-check = "ruff format --check ."
_ruff-format = "ruff format ."
_ruff-fix = "ruff check --fix ."
lint = ["_ruff-check", "_ruff-format-check"]
format = ["_ruff-format", "_ruff-fix"]
test = "pytest"
check = ["lint", "test"]
```

- [ ] **Step 4: Añadir dependencias (genera `uv.lock`)**

Run: `uv add pandas numpy scikit-learn pyyaml`
Run: `uv add --dev pytest ruff pre-commit poethepoet`
Expected: `pyproject.toml` gana `dependencies` y `[dependency-groups] dev`; existe `uv.lock`.

- [ ] **Step 5: Escribir el test de humo**

`tests/test_package.py`:
```python
import importlib

import pytest

SUBPACKAGES = ["data", "features", "models", "evaluation", "decision", "monitoring", "api"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name):
    importlib.import_module(f"credit_risk.{name}")
```

- [ ] **Step 6: Correr tests y lint**

Run: `uv run poe test`
Expected: 7 passed.

Run: `uv run poe lint`
Expected: `All checks passed!` y `files already formatted` (si no, correr `uv run poe format` y repetir).

- [ ] **Step 7: Configurar pre-commit**

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: uv run ruff check --fix
        language: system
        types_or: [python, pyi, jupyter]
      - id: ruff-format
        name: ruff format
        entry: uv run ruff format
        language: system
        types_or: [python, pyi, jupyter]
```

Run: `uv run pre-commit install`
Run: `uv run pre-commit run --all-files`
Expected: ambos hooks `Passed`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .pre-commit-config.yaml src tests configs/.gitkeep notebooks/.gitkeep models/.gitkeep reports/figures/.gitkeep data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/external/.gitkeep
git commit -m "build: scaffold credit_risk package with uv, ruff, pytest and poe"
```

Run: `git show --stat HEAD`
Expected: ningún `.csv`/`.xls` en el commit.

---

### Task 3: Configuración central (`params.yaml` + `config.py`) con TDD

**Files:**
- Create: `configs/params.yaml`, `src/credit_risk/config.py`
- Delete: `configs/.gitkeep` (ya no hace falta)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: paquete `credit_risk` de Task 2.
- Produces (usado por sub-proyectos 1–5):
  - `PROJECT_ROOT: Path`, `DEFAULT_PARAMS_PATH: Path`
  - `class ConfigError(ValueError)`
  - `@dataclass(frozen=True) Paths(raw, interim, processed, external, models, reports: Path)` — todas absolutas
  - `@dataclass(frozen=True) Costs(false_negative: float, false_positive: float)`
  - `@dataclass(frozen=True) Config(seed: int, target: str, training_file: Path, paths: Paths, costs: Costs)`
  - `load_config(path: Path | str = DEFAULT_PARAMS_PATH) -> Config`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from credit_risk.config import PROJECT_ROOT, ConfigError, load_config

VALID_YAML = """\
seed: 7
target: SeriousDlqin2yrs
paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  external: data/external
  models: models
  reports: reports
data:
  training_file: cs-training.csv
costs:
  false_negative: 5
  false_positive: 1
"""


def write_params(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_params_file_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.target == "SeriousDlqin2yrs"
    assert cfg.training_file == PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    assert cfg.costs.false_negative == 5.0
    assert cfg.costs.false_positive == 1.0


def test_relative_paths_resolve_against_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(write_params(tmp_path, VALID_YAML))
    assert cfg.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert cfg.paths.models == PROJECT_ROOT / "models"


def test_absolute_paths_are_kept(tmp_path):
    elsewhere = (tmp_path / "elsewhere").as_posix()
    text = VALID_YAML.replace("raw: data/raw", f'raw: "{elsewhere}"')
    cfg = load_config(write_params(tmp_path, text))
    assert cfg.paths.raw == Path(elsewhere)


def test_costs_are_parsed_as_floats(tmp_path):
    cfg = load_config(write_params(tmp_path, VALID_YAML))
    assert isinstance(cfg.costs.false_negative, float)
    assert cfg.costs.false_negative == 5.0


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_empty_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="empty"):
        load_config(write_params(tmp_path, ""))


def test_missing_top_level_key_is_named(tmp_path):
    text = VALID_YAML.replace("seed: 7\n", "")
    with pytest.raises(ConfigError, match="seed"):
        load_config(write_params(tmp_path, text))


def test_missing_path_key_is_named(tmp_path):
    text = VALID_YAML.replace("  models: models\n", "")
    with pytest.raises(ConfigError, match="models"):
        load_config(write_params(tmp_path, text))


def test_section_that_is_not_a_mapping_is_rejected(tmp_path):
    text = VALID_YAML.replace("costs:\n  false_negative: 5\n  false_positive: 1\n", "costs: null\n")
    with pytest.raises(ConfigError, match="costs"):
        load_config(write_params(tmp_path, text))


@pytest.mark.parametrize(("fn", "fp"), [(0, 1), (5, -1)])
def test_non_positive_costs_are_rejected(tmp_path, fn, fp):
    text = VALID_YAML.replace("false_negative: 5", f"false_negative: {fn}").replace(
        "false_positive: 1", f"false_positive: {fp}"
    )
    with pytest.raises(ConfigError, match="positive"):
        load_config(write_params(tmp_path, text))


def test_non_numeric_cost_is_rejected(tmp_path):
    text = VALID_YAML.replace("false_negative: 5", "false_negative: high")
    with pytest.raises(ConfigError, match="number"):
        load_config(write_params(tmp_path, text))
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL / error de colección con `ModuleNotFoundError: No module named 'credit_risk.config'`.

- [ ] **Step 3: Crear `configs/params.yaml`**

```yaml
# Parámetros del proyecto. Rutas relativas a la raíz del repo.
seed: 42
target: SeriousDlqin2yrs

paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  external: data/external
  models: models
  reports: reports

data:
  training_file: cs-training.csv

# Supuesto de negocio: costo relativo de un falso negativo (aprobar a quien cae en mora)
# frente a un falso positivo (rechazar a un buen cliente). Ver docs/business_context.md.
costs:
  false_negative: 5
  false_positive: 1
```

Run: `git rm configs/.gitkeep`

- [ ] **Step 4: Implementar `src/credit_risk/config.py`**

```python
"""Project configuration: loads configs/params.yaml and resolves paths against the project root."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"

TOP_LEVEL_KEYS = ("seed", "target", "paths", "data", "costs")
PATH_KEYS = ("raw", "interim", "processed", "external", "models", "reports")
DATA_KEYS = ("training_file",)
COST_KEYS = ("false_negative", "false_positive")


class ConfigError(ValueError):
    """Raised when params.yaml is missing or invalid."""


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    external: Path
    models: Path
    reports: Path


@dataclass(frozen=True)
class Costs:
    false_negative: float
    false_positive: float


@dataclass(frozen=True)
class Config:
    seed: int
    target: str
    training_file: Path
    paths: Paths
    costs: Costs


def _require(section: Any, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ConfigError(f"{where} is empty or not a mapping")
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"Missing keys in {where}: {', '.join(missing)}")
    return section


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"costs.{name} must be a number, got {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"costs.{name} must be positive, got {number}")
    return number


def load_config(path: Path | str = DEFAULT_PARAMS_PATH) -> Config:
    """Load and validate params.yaml. Relative paths resolve against PROJECT_ROOT, not the cwd."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    raw = _require(yaml.safe_load(path.read_text(encoding="utf-8")), TOP_LEVEL_KEYS, path.name)
    path_section = _require(raw["paths"], PATH_KEYS, f"{path.name} [paths]")
    data_section = _require(raw["data"], DATA_KEYS, f"{path.name} [data]")
    cost_section = _require(raw["costs"], COST_KEYS, f"{path.name} [costs]")

    paths = Paths(**{key: _resolve(path_section[key]) for key in PATH_KEYS})
    costs = Costs(**{key: _positive_float(cost_section[key], key) for key in COST_KEYS})

    return Config(
        seed=int(raw["seed"]),
        target=str(raw["target"]),
        training_file=paths.raw / data_section["training_file"],
        paths=paths,
        costs=costs,
    )
```

- [ ] **Step 5: Verificar que pasan**

Run: `uv run pytest tests/test_config.py -v`
Expected: 12 passed.

Run: `uv run poe check`
Expected: lint limpio, 19 passed en total.

- [ ] **Step 6: Comprobación manual con los datos reales (solo local, no es test)**

Run: `uv run python -c "from credit_risk.config import load_config; import pandas as pd; c = load_config(); df = pd.read_csv(c.training_file, index_col=0); print(df.shape, round(df[c.target].mean(), 4))"`
Expected: `(150000, 11) 0.0668` (11 columnas + el índice = 12 columnas en el CSV).

- [ ] **Step 7: Commit**

```bash
git add configs/params.yaml src/credit_risk/config.py tests/test_config.py
git commit -m "feat: add validated project config loader"
```

---

### Task 4: Documentación de gobierno y dominio

**Files:**
- Create: `CLAUDE.md`, `docs/business_context.md`, `docs/data_dictionary.md`, `docs/decisions/ADR-001-dataset.md`, `docs/ai_usage_log.md`
- Modify (reescritura completa): `AGENTS.md`, `docs/constitution.md`, `README.md`

**Interfaces:**
- Consumes: comandos `uv run poe ...` (Task 2), `configs/params.yaml` (Task 3).
- Produces: mapa de documentos al que remiten las reglas de `AGENTS.md`.

- [ ] **Step 1: `CLAUDE.md`**

```markdown
@AGENTS.md

## Notas para Claude Code

- Flujo de trabajo: `superpowers:brainstorming` → spec → `superpowers:writing-plans` → `superpowers:executing-plans` (o `subagent-driven-development`).
```

- [ ] **Step 2: Reescribir `AGENTS.md`**

```markdown
# AGENTS.md — reglas de trabajo

Modelo de probabilidad de mora severa (90+ días en los próximos 2 años) sobre el dataset
Give Me Some Credit, servido por una API de inferencia. Contexto completo de negocio:
[docs/business_context.md](docs/business_context.md).

## Antes de tocar código

1. Lee [docs/constitution.md](docs/constitution.md).
2. Lee la spec activa y su plan:
   - Spec activa: `docs/superpowers/specs/2026-10-14-cimientos-design.md`
   - Plan activo: `docs/superpowers/plans/2026-10-14-cimientos.md`
3. Si la tarea no está cubierta por una spec aprobada, no escribas código: propón primero una spec.

## Mapa de documentos

| Necesito… | Dónde |
|---|---|
| Problema, costos, decisión, salida de la API | `docs/business_context.md` |
| Principios no negociables | `docs/constitution.md` |
| Variables y calidad de datos | `docs/data_dictionary.md` |
| Decisiones técnicas y su porqué | `docs/decisions/ADR-*.md` |
| Specs y planes por sub-proyecto | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
| Registro del uso de IA | `docs/ai_usage_log.md` |

## Comandos

- Instalar: `uv sync` y luego `uv run pre-commit install`
- Lint: `uv run poe lint` · Formatear: `uv run poe format`
- Tests: `uv run poe test` · Todo: `uv run poe check`

## Reglas

- No modifiques ni muevas nada en `data/raw/`.
- No uses `cs-test.csv` para evaluar: no tiene etiquetas.
- Obtén rutas y parámetros con `credit_risk.config.load_config()`; nunca rutas absolutas ni constantes de negocio en el código.
- Ningún test puede depender de `data/raw/` (el CI no tiene los datos).
- `uv run poe check` debe pasar antes de cada commit.
- Un commit por tarea, en inglés, con Conventional Commits (`feat:`, `fix:`, `docs:`, `build:`, `ci:`, `test:`).
- Documentación en español; código e identificadores en inglés.
- Al cerrar una tarea hecha con IA, añade una entrada a `docs/ai_usage_log.md`.
- Al cambiar de sub-proyecto, actualiza aquí la spec y el plan activos.
```

- [ ] **Step 3: Escribir `docs/constitution.md`**

```markdown
# Constitución del proyecto

Principios no negociables. Si una spec o un plan los contradice, gana la constitución
y se corrige la spec.

1. **Reproducibilidad.** `uv.lock` versionado; semilla global `seed` en `configs/params.yaml`
   usada en todo proceso aleatorio. Mismo código + mismos datos = mismos resultados.
2. **El test set es sagrado.** Se separa una sola vez (estratificado por el target) y no se mira
   hasta la evaluación final del modelo campeón. La selección de modelo y umbrales usa
   validación (o CV sobre train).
3. **Sin leakage.** Imputación, escalado, binning, calibración y cualquier ajuste se entrenan
   solo con datos de entrenamiento, dentro de pipelines.
4. **Código en `src/`, exploración en notebooks.** La lógica reutilizable vive en
   `src/credit_risk/`; los notebooks solo exploran y llaman a funciones de `src/`.
5. **TDD y evidencia.** Todo código en `src/` nace con su test. Ningún resultado (métrica,
   test verde, mejora) se afirma sin la salida del comando que lo demuestra.
6. **Spec antes que código.** Cada sub-proyecto tiene spec aprobada y plan antes de escribir
   código. Un commit por tarea.
7. **Negocio en configuración.** Costos, umbrales y bandas viven en `configs/params.yaml`,
   nunca hardcodeados.
8. **Datos crudos inmutables.** `data/raw/` no se modifica ni se versiona; las transformaciones
   escriben en `data/interim/` o `data/processed/`.
```

- [ ] **Step 4: Escribir `docs/business_context.md`**

```markdown
# Contexto de negocio

## Problema

Estimar la probabilidad de que un cliente presente **mora severa: 90 días o más de atraso
en los próximos 2 años** (`SeriousDlqin2yrs = 1`), para apoyar la decisión de crédito.

## Tensión de negocio

- **Falso negativo (FN):** aprobar a un cliente que cae en mora → pérdida de capital.
- **Falso positivo (FP):** rechazar a un buen cliente → pérdida de margen y de relación.

**Supuesto de partida:** un FN cuesta **5 veces** lo que un FP (`costs` en
`configs/params.yaml`). Es un supuesto, no un dato: el sub-proyecto 3 incluye análisis de
sensibilidad del umbral frente a este ratio.

## Política de decisión (3 bandas)

| Nivel de riesgo | Decisión sugerida | Regla |
|---|---|---|
| Bajo | Aprobar | probabilidad < umbral inferior |
| Medio | Revisión manual | umbral inferior ≤ probabilidad < umbral superior |
| Alto | Rechazar | probabilidad ≥ umbral superior |

Los umbrales se eligen en el sub-proyecto 3 minimizando el costo esperado con la matriz
de costos anterior, y se guardan en `configs/params.yaml`.

## Contrato de salida de la API (sub-proyecto 4)

| Campo | Tipo | Descripción |
|---|---|---|
| `probability` | float [0, 1] | Probabilidad calibrada de mora severa |
| `risk_level` | `low` / `medium` / `high` | Banda de riesgo |
| `decision` | `approve` / `review` / `reject` | Decisión sugerida |
| `top_reasons` | lista de 3 | Variables que más empujan el riesgo de ese cliente (SHAP local) |

## Consumidores

Sistema de originación de crédito y analistas de riesgo (la banda "revisión manual" llega
a un analista). La decisión es una **sugerencia**; la decisión final es humana o de política.

## Métricas (propuesta, se confirma en el sub-proyecto 2)

- Discriminación para seleccionar modelo: ROC-AUC, Gini (= 2·AUC − 1), KS.
- PR-AUC como apoyo por el desbalance (~6,7 % de positivos).
- Calibración obligatoria (la API expone una probabilidad): Brier score y curva de calibración.
- Negocio: costo esperado por cliente para fijar umbrales.
```

- [ ] **Step 5: Escribir `docs/data_dictionary.md`**

Cifras verificadas el 2026-10-14 sobre `data/raw/cs-training.csv` (150.000 filas; primera columna = índice sin nombre 1..150000, no es variable).

```markdown
# Diccionario de datos — Give Me Some Credit

Fuente: Kaggle "Give Me Some Credit" (`cs-training.csv`, `Data Dictionary.xls`).
Solo `cs-training.csv` tiene target. Primera columna del CSV = ID de fila sin nombre (no es variable).

**Target:** `SeriousDlqin2yrs` — la persona tuvo 90+ días de atraso o peor (1/0).
150.000 filas, 10.026 positivos (**6,68 %**).

| Variable | Descripción | Tipo | Nulos | Rango observado | Notas |
|---|---|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Saldo en tarjetas y líneas personales (sin hipotecas ni cuotas) / suma de límites | ratio | 0 | 0 – 50.708 (p75 0,56) | Valores > 1 posibles; extremos claramente erróneos |
| `age` | Edad en años | entero | 0 | 0 – 109 | 1 fila con edad < 18 (edad 0) |
| `NumberOfTime30-59DaysPastDueNotWorse` | Veces con 30–59 días de atraso (no peor) en 2 años | entero | 0 | 0 – 98 | 269 filas con códigos 96/98 |
| `DebtRatio` | Pagos mensuales de deuda + gastos / ingreso bruto mensual | ratio | 0 | 0 – 329.664 (p75 0,87) | Cola extrema; investigar relación con `MonthlyIncome` nulo |
| `MonthlyIncome` | Ingreso mensual | real | **29.731 (19,8 %)** | 0 – 3.008.750 | Nulo informativo; requiere imputación + indicador |
| `NumberOfOpenCreditLinesAndLoans` | Créditos de cuotas y líneas de crédito abiertas | entero | 0 | 0 – 58 | |
| `NumberOfTimes90DaysLate` | Veces con 90+ días de atraso | entero | 0 | 0 – 98 | 269 filas con códigos 96/98 |
| `NumberRealEstateLoansOrLines` | Hipotecas y créditos inmobiliarios (incl. HELOC) | entero | 0 | 0 – 54 | |
| `NumberOfTime60-89DaysPastDueNotWorse` | Veces con 60–89 días de atraso (no peor) en 2 años | entero | 0 | 0 – 98 | 269 filas con códigos 96/98 |
| `NumberOfDependents` | Dependientes excluyendo al titular | entero | **3.924 (2,6 %)** | 0 – 20 | Se lee como float por los nulos |

## Problemas de calidad conocidos (a tratar en el sub-proyecto 1)

- Códigos 96/98 en las tres variables de atraso: probablemente códigos especiales, no conteos.
- 609 filas duplicadas (sin contar el ID).
- Outliers extremos en `RevolvingUtilizationOfUnsecuredLines` y `DebtRatio`.
- `MonthlyIncome` y `NumberOfDependents` con nulos.
```

- [ ] **Step 6: Escribir `docs/decisions/ADR-001-dataset.md`**

```markdown
# ADR-001: Dataset Give Me Some Credit

- **Fecha:** 2026-10-14
- **Estado:** aceptada

## Contexto

El repo arrancó con UCI German Credit (1.000 filas, target good/bad genérico). El objetivo
del proyecto define el target como mora de 90+ días en 2 años.

## Decisión

Usar **Give Me Some Credit** (Kaggle): su target `SeriousDlqin2yrs` coincide exactamente con
la definición de negocio, tiene 150.000 filas y presenta problemas reales (desbalance ~6,7 %,
nulos, códigos especiales, outliers). German Credit se retiró del repositorio.

## Consecuencias

- Solo `cs-training.csv` sirve para entrenar y evaluar; `cs-test.csv` no tiene etiquetas.
- El test set propio se separa de `cs-training.csv`.
- Los datos se descargan manualmente de Kaggle; no se versionan.
```

- [ ] **Step 7: Escribir `docs/ai_usage_log.md`**

```markdown
# Bitácora de uso de IA

Registro breve de qué se delegó a la IA, cómo se validó y qué se corrigió.

| Fecha | Tarea | Delegado a la IA | Validación humana / evidencia | Correcciones |
|---|---|---|---|---|
| 2026-10-14 | Brainstorming de objetivos y estructura | Revisión de estructura, preguntas de alcance y stack (Claude Opus + superpowers:brainstorming) | Respuestas de la usuaria a cada ronda de preguntas | Detectó que los datos eran German Credit y no GMSC; se reemplazaron |
| 2026-10-14 | Spec y plan de cimientos | Redacción de spec y plan; perfilado de `cs-training.csv` | Revisión de la spec por la usuaria | Makefile → poethepoet (no hay `make` en Windows) |
```

La fila de la ejecución del plan se añade en Task 6, con la evidencia real.

- [ ] **Step 8: Escribir `README.md`**

~~~markdown
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
| 1 | Datos + EDA | pendiente |
| 2 | Modelado | pendiente |
| 3 | Política de decisión | pendiente |
| 4 | API | pendiente |
| 5 | Monitoreo | pendiente |
~~~

- [ ] **Step 9: Verificar ausencia de duplicados y enlaces**

Run: `git grep -n "Objetivo de negocio" -- AGENTS.md CLAUDE.md`
Expected: sin resultados (el negocio ya no vive en AGENTS.md).

Run: `uv run poe check`
Expected: en verde (los docs no rompen nada).

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md AGENTS.md README.md docs/constitution.md docs/business_context.md docs/data_dictionary.md docs/decisions/ADR-001-dataset.md docs/ai_usage_log.md
git commit -m "docs: add governance, business context and data dictionary"
```

---

### Task 5: CI con GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `uv.lock`, `.python-version`, tareas `poe lint` y `poe test` (Task 2).
- Produces: workflow `CI` que corre en cada push y pull request.

- [ ] **Step 1: Crear el workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - name: Install dependencies
        run: uv sync --locked
      - name: Lint
        run: uv run poe lint
      - name: Test
        run: uv run poe test
```

- [ ] **Step 2: Simular el CI localmente desde un clon limpio**

En Git Bash:
```bash
rm -rf /tmp/ci-check && git clone . /tmp/ci-check && cd /tmp/ci-check && uv sync --locked && uv run poe lint && uv run poe test; cd -
```
(Si `/tmp` no está disponible, usar cualquier carpeta temporal fuera del repo.)
Expected: lint limpio y 19 passed **sin** `data/raw/` (el clon no tiene los datos). Confirma también que no hay problemas de CRLF.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run lint and tests with uv on GitHub Actions"
```

---

### Task 6: Verificación final contra la spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-14-cimientos-design.md` (marcar criterios de aceptación cumplidos)
- Modify: `docs/ai_usage_log.md` (fila de ejecución con evidencia real)

- [ ] **Step 1: Ejecutar todos los criterios verificables localmente**

Run: `uv sync && uv run poe check`
Expected: 19 passed, lint limpio.

Run: `git ls-files data`
Expected: solo los cuatro `.gitkeep`.

Run: `git log --oneline`
Expected: commits en orden: spec → plan → scaffold → config → docs → ci.

- [ ] **Step 2: Marcar en la spec los criterios cumplidos**

En la sección 5 cambiar `- [ ]` por `- [x]` en: `uv sync && uv run poe check`, `cs-training.csv … no está versionado`, `git log`, `Ningún documento duplica`. Dejar sin marcar CI en verde y `/memory` (requieren a la usuaria).

- [ ] **Step 3: Registrar la ejecución en `docs/ai_usage_log.md`**

Añadir una fila al final de la tabla, con la salida real del Step 1 (número de tests que pasaron) y cualquier desviación del plan que haya ocurrido:

```markdown
| 2026-10-14 | Ejecución del plan de cimientos | Implementación tarea a tarea (Claude Sonnet + superpowers:executing-plans) | `uv run poe check`: <N> passed, lint limpio | <desviaciones del plan, o "ninguna"> |
```
Sustituye `<N>` y la última celda por los valores reales antes de guardar.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-10-14-cimientos-design.md docs/ai_usage_log.md
git commit -m "docs: mark foundations acceptance criteria"
```

- [ ] **Step 5: Entregar a la usuaria los pasos manuales (no ejecutarlos)**

1. Crear un repo vacío en GitHub y hacer: `git remote add origin <url>` y `git push -u origin main`; confirmar que el workflow `CI` queda en verde.
2. Abrir una sesión nueva de Claude Code y ejecutar `/memory` para confirmar que `AGENTS.md` se carga vía `CLAUDE.md`.
3. Marcar esos dos criterios en la spec.
