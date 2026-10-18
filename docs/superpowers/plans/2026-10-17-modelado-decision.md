# Modelado + Política de decisión — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline de tratamiento ajustado solo con train, tres candidatos (logística baseline, Random Forest, LightGBM) con búsqueda de hiperparámetros, calibración, selección por costo esperado, política de 3 bandas con sensibilidad, evaluación única en test y artefacto documentado.

**Architecture:** Transformadores de scikit-learn en `credit_risk.features` (salida pandas) que se componen con cada modelo en un `Pipeline`. `credit_risk.evaluation` (métricas) y `credit_risk.decision` (umbrales, bandas, sensibilidad) son funciones puras sobre `y`/`p`. `credit_risk.models` busca, calibra y elige; `credit_risk.models.train` orquesta todo, abre el test una vez y escribe artefacto, metadata, reporte y runs de MLflow. El notebook `02-modelado` solo lee esos resultados y dibuja.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, lightgbm, mlflow, shap, joblib, matplotlib, pytest, uv + poe.

**Spec:** `docs/superpowers/specs/2026-10-17-modelado-decision-design.md`

## Global Constraints

- Todo tratamiento se ajusta **solo con train** (`fit`) y se aplica con `transform`; ninguna estadística sale de valid o test.
- Test: se lee y evalúa **una sola vez**, al final de `run_training`, con el campeón ya calibrado y los umbrales ya fijados.
- Costos: FN = `cfg.costs.false_negative` (5), FP = `cfg.costs.false_positive` (1). Nunca hardcodeados.
- Semilla `cfg.seed` en todo proceso aleatorio (búsqueda, modelos, pliegues, muestreos).
- Sin `class_weight` ni remuestreo.
- Nombres de columnas nuevas: `has_special_past_due_code`, `income_unavailable`. `MODEL_FEATURES = [*FEATURES, "has_special_past_due_code", "income_unavailable"]`.
- Nombres de candidatos: `"logistic"`, `"random_forest"`, `"lightgbm"` (en ese orden de complejidad).
- Bandas: `"approve"` / `"review"` / `"reject"`; niveles `"low"` / `"medium"` / `"high"`.
- Dependencias nuevas permitidas: `lightgbm`, `mlflow`, `shap`. Ninguna otra (sin `tabulate`: las tablas Markdown se generan a mano).
- Tests con datos sintéticos; ninguno lee `data/raw/` ni `data/interim/`. `uv run poe check` en verde antes de cada commit.
- Documentación en español; código, identificadores y commits en inglés. Un commit por tarea. No hacer `git push`.

## Review Focus

1. Un cliente de valid/test/API cuyo grupo edad × dependientes no existe (o tiene < 30 filas) en train → se imputa con el respaldo, nunca queda NaN. (Test en Task 2.)
2. Predecir **una sola fila** (como enviará la API) → los cortes de edad y percentiles aprendidos se reutilizan; no se recalculan sobre la fila. (Test en Task 2.)
3. `MonthlyIncome = 0` con `DebtRatio` tipo monto → el ratio recalculado es finito (el ingreso imputado siempre es > 0). (Test en Task 2.)
4. Capacidad de revisión mayor que la fracción de clientes bajo `t*` → `t_low` se acota a 0, sin error. (Test en Task 4.)
5. DataFrame de entrada con columnas extra o en otro orden (p. ej. el target o un JSON de la API) → el preprocesamiento selecciona `FEATURES` en orden y no se rompe. (Test en Task 2.)

---

### Task 1: Configuración, dependencias y gobierno

**Files:**
- Modify: `configs/params.yaml`, `src/credit_risk/config.py`, `tests/test_config.py`, `pyproject.toml` / `uv.lock`, `docs/constitution.md`, `docs/business_context.md`, `AGENTS.md`

**Interfaces:**
- Produces: `Decision(review_capacity: float, cost_sensitivity_ratios: tuple[float, ...], review_capacity_sensitivity: tuple[float, ...])`, `Search(n_iter: int, cv_folds: int)` (dataclasses frozen); campos `Config.decision`, `Config.search`. Claves obligatorias nuevas de nivel superior: `decision`, `search`.

- [ ] **Step 0: Commit de spec y plan**

```bash
git add docs/superpowers/specs/2026-10-17-modelado-decision-design.md
git commit -m "docs: add modeling and decision spec"
git add docs/superpowers/plans/2026-10-17-modelado-decision.md
git commit -m "docs: add modeling and decision implementation plan"
```

- [ ] **Step 1: Tests que fallan.** Añadir a `VALID_YAML` de `tests/test_config.py`:
```yaml
decision:
  review_capacity: 0.10
  cost_sensitivity_ratios: [3, 5, 10, 20]
  review_capacity_sensitivity: [0.05, 0.10, 0.20]
search:
  n_iter: 20
  cv_folds: 5
```
Tests nuevos:
- `test_default_decision_and_search_params`: `load_config()` → `review_capacity == 0.10`, `cost_sensitivity_ratios == (3.0, 5.0, 10.0, 20.0)`, `review_capacity_sensitivity == (0.05, 0.10, 0.20)`, `search.n_iter == 20`, `search.cv_folds == 5`.
- `test_review_capacity_must_be_fraction` (parametrize `0`, `1`, `"x"`) → `ConfigError` con `"between 0 and 1"`.
- `test_search_values_must_be_positive_integers` (parametrize `n_iter: 0`, `cv_folds: 1`) → `ConfigError` con `"must be"` (`cv_folds` ≥ 2, `n_iter` ≥ 1).
- `test_missing_decision_key_is_named`: quitar `review_capacity` → `ConfigError` con `"review_capacity"`.

- [ ] **Step 2: Verificar que fallan.** Run: `uv run pytest tests/test_config.py -v` → fallan los nuevos; los existentes pasan.

- [ ] **Step 3: Implementar.** `params.yaml`: secciones `decision` y `search` con los valores de arriba y comentarios en español (`review_capacity` = fracción de solicitudes que pueden revisar los analistas; supuesto, como el 5:1). `config.py`: reutilizar `_require` y `_fraction`; listas → `tuple[float, ...]` (cada capacidad validada con `_fraction`, cada ratio con `_positive_float`).

- [ ] **Step 4: Verificar.** Run: `uv run pytest tests/test_config.py -v` → todo verde.

- [ ] **Step 5: Dependencias.** Run: `uv add lightgbm mlflow shap`. Añadir en `[tool.poe.tasks]`: `train = "python -m credit_risk.models.train"`.

- [ ] **Step 6: Gobierno.**
  - `docs/constitution.md`, principio 7, nuevo texto: "**Negocio en configuración.** Las entradas de negocio (costos, capacidad de revisión, escenarios de sensibilidad) viven en `configs/params.yaml`; los umbrales que se aprenden del modelo viven en su metadata versionada (`models/champion_metadata.json`). Nada de esto se hardcodea."
  - `docs/business_context.md`: en "Política de decisión", explicar la banda de revisión por capacidad (`review_capacity`, 10 % supuesto) y que los umbrales quedan en la metadata del modelo; quitar las referencias a "sub-proyecto 3" (ahora es el 2) y marcar las métricas como confirmadas según la spec 002 §5.
  - `AGENTS.md`: spec y plan activos → los de este sub-proyecto.

- [ ] **Step 7: Verificar y commit.** Run: `uv run poe check` → verde.
```bash
git add configs/params.yaml src/credit_risk/config.py tests/test_config.py pyproject.toml uv.lock docs/constitution.md docs/business_context.md AGENTS.md
git commit -m "feat: add decision and search parameters and modeling dependencies"
```

---

### Task 2: Transformadores de tratamiento

**Files:**
- Create: `src/credit_risk/features/treatment.py`, `tests/test_features_treatment.py`

**Interfaces:**
- Consumes: `FEATURES`, `PAST_DUE_COLUMNS`, `SPECIAL_PAST_DUE_CODES` de `credit_risk.data.schema`.
- Produces:
  - `MODEL_FEATURES: list[str]` y `COUNT_COLUMNS: list[str]` (las 3 de atraso + `NumberOfOpenCreditLinesAndLoans` + `NumberRealEstateLoansOrLines` + `NumberOfDependents`).
  - `drop_underage(df: pd.DataFrame, min_age: int = 18) -> pd.DataFrame` (filas con `age < 18` fuera; no muta).
  - `SpecialPastDueRecoder()` — sin estado. Selecciona `FEATURES` (en orden), añade `has_special_past_due_code` (int 0/1) y pone a 0 los conteos 96/98.
  - `IncomeImputer(min_group_size: int = 30)` — `fit` aprende: `income_cap_` (p99 de ingresos > 0), `age_edges_` (cortes de deciles de `age`, extremos abiertos con `-inf`/`inf`), `group_medians_` (mediana del ingreso > 0 recortado por decil de edad × dependientes `0,1,2,3,4+`, solo grupos con ≥ `min_group_size` filas), `age_medians_`, `global_median_`. `transform`: añade `income_unavailable` (1 si ingreso NaN o 0), recorta el ingreso observado en `income_cap_`, `NumberOfDependents` NaN → 0, imputa el ingreso no disponible con grupo → edad → global.
  - `RatioTreatment()` — requiere la salida de `IncomeImputer`. `fit` aprende `debt_ratio_cap_` (p99 de `DebtRatio` en filas con `income_unavailable == 0`) y `revolving_cap_` (p99 de `RevolvingUtilization…`). `transform`: en filas `income_unavailable == 1`, `DebtRatio = DebtRatio / MonthlyIncome` (ingreso ya imputado); después recorta `DebtRatio` en `debt_ratio_cap_` (todas las filas) y `RevolvingUtilization…` en `revolving_cap_`.
  - `LogisticScaling()` — rama logística. `fit` aprende el p99 de cada columna de `COUNT_COLUMNS`; `transform` las recorta y aplica `log1p` a `MonthlyIncome`, `RevolvingUtilization…` y `DebtRatio`.
  - `build_preprocessor(kind: Literal["logistic", "tree"]) -> Pipeline` — `"tree"`: recoder → imputer → ratios. `"logistic"`: lo mismo + `LogisticScaling` + `StandardScaler`. Todo con `set_output(transform="pandas")`; columnas de salida = `MODEL_FEATURES`.
  - Todos los transformadores heredan de `BaseEstimator, TransformerMixin`, no mutan la entrada y exponen `get_feature_names_out`.

- [ ] **Step 1: Tests que fallan** (fixture local: ~200 filas sintéticas con `numpy.random.default_rng(0)`, edades 20–80, dependientes 0–6 con algunos NaN, ingresos con NaN y ceros, algunos 96/98).
- `test_drop_underage`: edades `[17, 18, 40]` → quedan `[18, 40]`.
- `test_recoder_flags_and_zeroes_codes`: fila con 98 en las 3 → flag 1 y conteos 0; fila con 2 → flag 0 y conteo 2 intacto.
- `test_recoder_selects_features_in_order`: entrada con target y columnas desordenadas → salida exactamente `FEATURES + ["has_special_past_due_code"]`.
- `test_imputer_flags_unavailable_income`: ingresos `[NaN, 0, 5000]` → `income_unavailable == [1, 1, 0]`.
- `test_imputer_leaves_no_missing`: tras `fit_transform`, `MonthlyIncome` y `NumberOfDependents` sin NaN y todos los ingresos imputados > 0.
- `test_imputer_uses_group_median`: datos donde el grupo (decil de edad de 30 años, 2 dependientes) tiene 40 filas con ingreso 3.000 y el resto 9.000 → un NaN de ese grupo se imputa con 3.000.
- `test_imputer_falls_back_for_unseen_group`: transformar una fila con 9 dependientes y edad 95 → ingreso imputado = mediana del decil de edad (o global si no hay) y no NaN.
- `test_imputer_learns_only_from_fit_data`: `fit` con un frame A, `transform` de un frame B con ingresos de 10⁶ → `income_cap_` igual que antes de transformar B y los ingresos de B recortados al cap de A.
- `test_ratio_recomputed_when_income_unavailable`: fila con ingreso 0, `DebtRatio = 1500` e ingreso imputado 3.000 → `DebtRatio` intermedio 0,5 (y finito).
- `test_ratio_caps`: `DebtRatio` y `RevolvingUtilization…` por encima de sus caps aprendidos quedan exactamente en el cap.
- `test_single_row_transform`: pipeline `"tree"` ajustado en el fixture transforma 1 fila sin error y devuelve 1 fila sin NaN.
- `test_preprocessor_output_columns`: `build_preprocessor("tree")` y `("logistic")` → columnas `MODEL_FEATURES`.
- `test_logistic_scaling_clips_counts_and_logs`: un conteo por encima de su p99 queda en el p99; `MonthlyIncome` transformado == `log1p` del valor previo.
- `test_transformers_do_not_mutate_input`.

- [ ] **Step 2: Verificar que fallan.** Run: `uv run pytest tests/test_features_treatment.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `treatment.py`** con las interfaces de arriba. Para el decil de edad en `transform`, usar `np.digitize`/`pd.cut` con `age_edges_` aprendidos (nunca `qcut` en transform).

- [ ] **Step 4: Verificar.** Run: `uv run poe check` → verde.

- [ ] **Step 5: Commit.**
```bash
git add src/credit_risk/features/treatment.py tests/test_features_treatment.py
git commit -m "feat: add missing-value and outlier treatment transformers"
```

---

### Task 3: Métricas

**Files:**
- Create: `src/credit_risk/evaluation/metrics.py`, `tests/test_evaluation_metrics.py`

**Interfaces:**
- Produces:
  - `gini(y_true, p) -> float` (= 2·AUC − 1).
  - `ks_statistic(y_true, p) -> float` (máx. diferencia entre las CDF de `p` en positivos y negativos).
  - `expected_cost(y_true, p, threshold: float, fn_cost: float, fp_cost: float) -> float` — costo medio por cliente; predice positivo si `p ≥ threshold`.
  - `classification_metrics(y_true, p) -> dict[str, float]` con claves `roc_auc`, `gini`, `ks`, `pr_auc` (average precision), `brier`.

- [ ] **Step 1: Tests que fallan.**
- Separación perfecta `y = [0,0,1,1]`, `p = [0.1,0.2,0.8,0.9]` → `roc_auc == 1`, `gini == 1`, `ks == 1`.
- Predicción aleatoria constante `p = 0.5` → `gini == 0`, `ks == 0`.
- `expected_cost`: `y = [1, 0, 0, 1]`, `p = [0.9, 0.6, 0.1, 0.2]`, umbral 0.5, FN 5, FP 1 → 1 FP + 1 FN → `(1 + 5) / 4 == 1.5`.
- `brier` de `y = [0, 1]`, `p = [0.0, 1.0]` == 0.
- Claves exactas de `classification_metrics`.

- [ ] **Step 2–4:** fallan (`ModuleNotFoundError`) → implementar con `sklearn.metrics` (`roc_auc_score`, `average_precision_score`, `brier_score_loss`, `roc_curve` para KS) → `uv run poe check` verde.

- [ ] **Step 5: Commit.** `git commit -m "feat: add credit risk evaluation metrics"` (con los dos archivos).

---

### Task 4: Política de decisión

**Files:**
- Create: `src/credit_risk/decision/policy.py`, `tests/test_decision_policy.py`

**Interfaces:**
- Consumes: `expected_cost` (Task 3).
- Produces:
  - `THRESHOLD_GRID = np.round(np.arange(0.005, 0.996, 0.005), 3)`.
  - `optimal_threshold(y_true, p, fn_cost, fp_cost, grid=THRESHOLD_GRID) -> tuple[float, float]` → `(t_star, cost)`; en empate, el umbral más bajo.
  - `review_threshold(p, t_star: float, capacity: float) -> float` → `t_low` tal que la fracción de `p` en `[t_low, t_star)` ≈ `capacity`: cuantil de `p` en `max(F(t_star) − capacity, 0)`; acotado a `[0, t_star]`.
  - `assign_band(p, t_low, t_star) -> np.ndarray` con `"approve"`/`"review"`/`"reject"`; `RISK_LEVEL = {"approve": "low", "review": "medium", "reject": "high"}`.
  - `band_summary(y_true, p, t_low, t_star) -> pd.DataFrame` — índice `approve, review, reject`; columnas `rows`, `share`, `default_rate`.
  - `cost_sensitivity(y_true, p, ratios, fp_cost, capacity) -> pd.DataFrame` — una fila por ratio FN:FP; columnas `fn_fp_ratio`, `t_star`, `t_low`, `approval_rate` (approve + review), `default_rate_approved`, `expected_cost`.
  - `capacity_sensitivity(y_true, p, t_star, capacities) -> pd.DataFrame` — una fila por capacidad; columnas `capacity`, `t_low`, más `rows`/`default_rate` de cada banda (`approve_rows`, `approve_default_rate`, …).

- [ ] **Step 1: Tests que fallan.**
- `test_optimal_threshold_matches_bayes_rule`: `p` uniforme en 10.000 puntos y `y ~ Bernoulli(p)` con semilla → `t_star` dentro de ±0,03 de `1/6` para FN 5 / FP 1.
- `test_optimal_threshold_tie_takes_lowest`.
- `test_review_threshold_capacity`: `p = linspace(0, 1, 1001)`, `t_star = 0.5`, capacidad 0,10 → `t_low ≈ 0.4` y la fracción en `[t_low, 0.5)` ≈ 0,10 (±0,005).
- `test_review_threshold_clips_at_zero`: `t_star = 0.05`, capacidad 0,20 → `t_low == 0.0`.
- `test_assign_band_boundaries`: `p = [0.1, 0.4, 0.5]`, `t_low = 0.4`, `t_star = 0.5` → `["approve", "review", "reject"]`.
- `test_band_summary_shares_sum_to_one`.
- `test_cost_sensitivity_threshold_falls_as_fn_cost_rises`: `t_star` no creciente al pasar de ratio 3 → 20.
- `test_capacity_sensitivity_review_grows_with_capacity`.

- [ ] **Step 2–4:** fallan → implementar → `uv run poe check` verde.

- [ ] **Step 5: Commit.** `git commit -m "feat: add cost-based threshold and three-band decision policy"`.

---

### Task 5: Candidatos, búsqueda, calibración y selección

**Files:**
- Create: `src/credit_risk/models/candidates.py`, `src/credit_risk/models/selection.py`, `tests/test_models_candidates.py`, `tests/test_models_selection.py`

**Interfaces:**
- Consumes: `build_preprocessor` (Task 2).
- Produces:
  - `COMPLEXITY_ORDER = ("logistic", "random_forest", "lightgbm")`.
  - `candidate_specs(seed: int) -> dict[str, tuple[Pipeline, dict]]` — por nombre: pipeline (`build_preprocessor(kind)` + modelo, paso final llamado `"model"`) y distribuciones para `RandomizedSearchCV` (`"model__..."`). Logística: `C` log-uniforme 1e-3–1e2, `penalty` l1/l2 con solver `saga`/`liblinear`, `max_iter=2000`. RF: `n_estimators` 100–300, `max_depth` 4–12, `min_samples_leaf` 20–200, `max_features` sqrt/0.5, `n_jobs=-1`. LightGBM: `n_estimators` 200–800, `learning_rate` 0.01–0.1 log, `num_leaves` 15–63, `min_child_samples` 20–200, `subsample` 0.6–1 (`subsample_freq=1`), `colsample_bytree` 0.6–1, `reg_lambda` 0–5, `verbose=-1`. Rangos libres si se respeta ~15 min en total.
  - `@dataclass SearchResult(name: str, estimator: Pipeline, best_params: dict, cv_auc_mean: float, cv_auc_std: float, search_seconds: float)`.
  - `search_candidate(name, pipeline, distributions, X, y, n_iter, cv_folds, seed) -> SearchResult` — `RandomizedSearchCV(scoring="roc_auc", cv=StratifiedKFold(shuffle=True, random_state=seed), refit=True)`.
  - `calibrate(estimator: Pipeline, X, y, cv_folds: int) -> tuple[CalibratedClassifierCV, float]` — `CalibratedClassifierCV(clone(estimator), method="isotonic", cv=cv_folds)`; devuelve el modelo y segundos de ajuste.
  - `prediction_latency_ms(model, X, n_rows: int = 1000, repeats: int = 5) -> float` — latencia mediana por cliente en ms, prediciendo de una fila en una (sobre una muestra de `min(n_rows, len(X))`). Puede usar 200 filas si 1.000 tarda demasiado; documentar en el reporte.
  - `select_champion(table: pd.DataFrame, tolerance: float = 0.02, brier_tolerance: float = 0.10) -> str` — `table` con columnas `model`, `expected_cost`, `brier`. Regla de la spec §5: el de menor costo; entre los que están a ≤ 2 % de ese costo, el más simple según `COMPLEXITY_ORDER`, salvo que su Brier sea > 10 % peor que el del de menor costo.

- [ ] **Step 1: Tests que fallan.**
- `test_candidate_specs_names_and_steps`: claves == `COMPLEXITY_ORDER`; cada pipeline termina en el paso `"model"`; todas las claves de distribución empiezan por `"model__"`.
- `test_search_candidate_runs_on_small_data`: 400 filas sintéticas con señal, `n_iter=2`, `cv_folds=2` para los 3 candidatos → `cv_auc_mean > 0.6` y `search_seconds > 0`.
- `test_calibrate_returns_probabilities`: salida de `predict_proba` en [0, 1] y filas que suman 1.
- `test_prediction_latency_positive`.
- `test_select_champion_lowest_cost`: costos `{logistic: 1.00, random_forest: 0.90, lightgbm: 0.80}` → `"lightgbm"`.
- `test_select_champion_prefers_simpler_within_tolerance`: `{logistic: 0.81, rf: 0.90, lgbm: 0.80}` y Brier parecidos → `"logistic"`.
- `test_select_champion_rejects_simpler_with_bad_brier`: como el anterior pero Brier de logística 20 % peor → `"lightgbm"`.

- [ ] **Step 2–4:** fallan → implementar → `uv run poe check` verde (los tests de búsqueda deben tardar < 30 s en total).

- [ ] **Step 5: Commit.** `git commit -m "feat: add candidate models, search, calibration and champion selection"`.

---

### Task 6: Orquestación del entrenamiento (`uv run poe train`)

**Files:**
- Create: `src/credit_risk/models/train.py`, `src/credit_risk/models/report.py`, `tests/test_models_train.py`

**Interfaces:**
- Consumes: todo lo anterior; `load_partition`, `PARTITIONS`; `Config`.
- Produces:
  - `markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str` (en `report.py`, sin `tabulate`).
  - `run_training(cfg: Config, tracking_uri: str | None = None) -> dict` — devuelve la metadata escrita. Pasos, en orden:
    1. Cargar `train` y `valid`; `drop_underage`; separar `X = df[FEATURES]`, `y = df[cfg.target]`.
    2. Por candidato: `search_candidate` (n_iter y pliegues de `cfg.search`) → `calibrate` → probabilidades en valid → `classification_metrics` + `optimal_threshold` + `expected_cost` + `prediction_latency_ms`; además Brier y AUC en valid **sin** calibrar (de `SearchResult.estimator`). Un run de MLflow por candidato (experimento `"credit-risk-modeling"`) con params, métricas y tiempos.
    3. `select_champion` sobre la tabla de valid.
    4. Con el campeón: `t_star` de valid, `t_low = review_threshold(p_valid, t_star, cfg.decision.review_capacity)`, `band_summary`, `cost_sensitivity`, `capacity_sensitivity`.
    5. **Test, una única vez:** cargar `test`, `drop_underage`, predecir con el campeón, `classification_metrics`, `expected_cost` en `t_star`, `band_summary`.
    6. Guardar `models/champion.joblib` (`{"model", "features": FEATURES, "t_low", "t_star"}`), `models/candidates.joblib` (los 3 calibrados, para el notebook), `models/champion_metadata.json` y `reports/model_comparison.md`.
  - Metadata (claves): `champion`, `selection_rule`, `features`, `model_features`, `thresholds` (`t_star`, `t_low`, `bayes_threshold` = FP/(FP+FN)), `costs`, `review_capacity`, `candidates` (por modelo: `best_params`, `cv_auc_mean`, `cv_auc_std`, `search_seconds`, `calibration_seconds`, `latency_ms`, `valid_metrics`, `valid_metrics_uncalibrated`, `valid_expected_cost`, `valid_t_star`), `valid_bands`, `cost_sensitivity`, `capacity_sensitivity`, `test_metrics`, `test_expected_cost`, `test_bands`, `rows` (por partición, tras `drop_underage`), `data_sha256` (por CSV), `seed`, `library_versions`, `trained_at` (ISO).
  - `reports/model_comparison.md`: tabla de candidatos (CV AUC ± std, AUC/Gini/KS/PR-AUC/Brier en valid, Brier sin calibrar, costo, `t_star`, tiempos, latencia), campeón y regla aplicada, bandas en valid, tablas de sensibilidad, resultados de test. Encabezados en español.
  - `if __name__ == "__main__": run_training(load_config())`.

- [ ] **Step 1: Test que falla** — `test_run_training_end_to_end(tmp_path)`: escribir `train/valid/test.csv` sintéticos (columnas `COLUMNS`, índice `id`, ~600/200/200 filas con señal y una fila con `age = 10`) en `tmp_path/"interim"`; construir `cfg` con `dataclasses.replace` (rutas en `tmp_path`, `search = Search(n_iter=2, cv_folds=2)`); `run_training(cfg, tracking_uri=(tmp_path / "mlruns").as_uri())`. Asserts: existen `champion.joblib`, `champion_metadata.json` y `model_comparison.md`; `metadata["champion"] in COMPLEXITY_ORDER`; `0 <= t_low <= t_star`; las 3 entradas en `candidates`; `test_metrics` con las 5 claves; `rows["train"]` = filas sintéticas − 1 (la de edad 10).
- `test_test_partition_loaded_once(tmp_path, monkeypatch)`: envolver `load_partition` para contar llamadas con `"test"` → exactamente 1.
- `test_markdown_table`: DataFrame 2×2 → cabecera, separador `|---|` y 2 filas.

- [ ] **Step 2–4:** fallan → implementar → `uv run poe check` verde.

- [ ] **Step 5: Entrenamiento real.** Run: `uv run poe train`. Expected: termina (≈ 15 min o menos) y crea los 3 artefactos. Anotar el tiempo total.

- [ ] **Step 6: Reproducibilidad.** Copiar `models/champion_metadata.json`, volver a correr `uv run poe train` y comparar ignorando `trained_at`, `*_seconds` y `latency_ms`: mismo campeón, mismos umbrales y mismas métricas.

- [ ] **Step 7: Commit.**
```bash
git add src/credit_risk/models/train.py src/credit_risk/models/report.py tests/test_models_train.py models/champion_metadata.json reports/model_comparison.md
git commit -m "feat: add reproducible training pipeline with champion selection"
```

---

### Task 7: Notebook de análisis del modelado

**Files:**
- Create: `notebooks/02-modelado.ipynb`, figuras en `reports/figures/`
- Optional: helpers de gráficos en `src/credit_risk/evaluation/plots.py` con su test (si el notebook necesita más que una llamada directa a matplotlib/sklearn).

- [ ] **Step 1: Crear el notebook** (Markdown en español; sin filas de datos; solo agregados y gráficos). Lee `models/champion_metadata.json`, `models/candidates.joblib` y la partición **valid** (el test no se vuelve a evaluar: sus cifras salen de la metadata). Secciones:
  1. Tratamiento aplicado: tabla de decisiones U1–U5 y C1–C8 (de la spec) y un ejemplo agregado del antes/después en valid (nulos de ingreso antes y después, % con `has_special_past_due_code`, % con `income_unavailable`).
  2. Comparación de candidatos: tabla de la metadata + curvas ROC en valid → `roc_valid.png`.
  3. Calibración: curvas de calibración en valid de los 3 (`sklearn.calibration.calibration_curve`) → `calibration_valid.png`.
  4. Costo esperado vs umbral del campeón en valid, marcando `t_star` y `1/6` → `cost_vs_threshold.png`.
  5. Política: bandas en valid, sensibilidad por costo y por capacidad.
  6. Importancia global: SHAP (`TreeExplainer`, muestra de 2.000 filas de valid transformadas) si el campeón es de árboles; coeficientes estandarizados si es la logística → `importance_champion.png`. El estimador base se obtiene de `calibrated.calibrated_classifiers_[0].estimator`.
  7. Resultado en test (de la metadata) frente a valid.

- [ ] **Step 2: Ejecutar.** Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02-modelado.ipynb` → sin errores; no imprime rutas absolutas (asignar `_ = save_figure(...)`).

- [ ] **Step 3: Lint y commit.** `uv run poe lint`.
```bash
git add notebooks/02-modelado.ipynb reports/figures
git commit -m "feat: add modeling analysis notebook and figures"
```

---

### Task 8: Ficha del modelo, ADR y cierre

**Files:**
- Create: `docs/model_card.md`, `docs/decisions/ADR-002-seleccion-modelo.md`
- Modify: `README.md` (Estado), `docs/ai_usage_log.md`, spec (§11)

- [ ] **Step 1: `docs/model_card.md`** (cifras solo de la metadata y del notebook ejecutado). Secciones: Propósito y uso previsto; Datos (particiones y tamaños); Tratamiento de faltantes y atípicos (U1–U5, C1–C8, cada uno con su porqué); Candidatos y búsqueda (espacios, iteraciones, tiempos); Resultados en valid (tabla); Criterio de selección y por qué el costo decide; Umbrales y bandas (`t_star` vs 1/6, `t_low`, capacidad 10 %); Sensibilidad (costo y capacidad); Resultado en test; Importancia de variables; Limitaciones y riesgos (supuestos 5:1 y 10 %; uso de `age`, variable sensible en crédito; población sin menores de 18; códigos 96/98 de significado no documentado; `DebtRatio` como monto); Cuándo reentrenar (p. ej. caída de AUC o deriva de PSI en el monitoreo futuro).

- [ ] **Step 2: `docs/decisions/ADR-002-seleccion-modelo.md`**: contexto, decisión (AUC para buscar, costo esperado para elegir, regla de simplicidad con 2 % y Brier 10 %), alternativas descartadas (solo AUC; F1; `class_weight`), consecuencias.

- [ ] **Step 3:** `README.md`: sub-proyectos "Modelado" y "Política de decisión" → ✅ (indicar que se hicieron juntos). Fila en `docs/ai_usage_log.md` con evidencia real (tests, tiempo de `poe train`, reproducibilidad) y desviaciones. Marcar criterios cumplidos en la spec §11.

- [ ] **Step 4: Verificar y commit.** `uv run poe check` → verde; `git ls-files models` → solo `.gitkeep` y `champion_metadata.json`.
```bash
git add docs/model_card.md docs/decisions/ADR-002-seleccion-modelo.md README.md docs/ai_usage_log.md docs/superpowers/specs/2026-10-17-modelado-decision-design.md
git commit -m "docs: add model card and model selection ADR"
```
