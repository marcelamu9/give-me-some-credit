# Spec 001 — Datos + EDA

- **Fecha:** 2026-10-17
- **Estado:** aprobada (2026-10-17)
- **Sub-proyecto:** 1 de 6 (Cimientos → **Datos/EDA** → Modelado → Decisión → API → Monitoreo)
- **Depende de:** spec 000 (cimientos): `credit_risk.config.load_config()`, `configs/params.yaml`.

## 1. Intención

Entender cómo se comportan los datos antes de modelar: calidad estructural primero y luego
comportamiento estadístico. El resultado es **conocimiento documentado**, no datos tratados:
los hallazgos alimentan la spec de modelado, donde se decidirán los tratamientos.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos**
- Fase 1: diagnóstico de calidad de datos sobre `cs-training.csv` completo.
- Eliminar los duplicados exactos (decisión de la usuaria; único cambio sobre los datos).
- Split estratificado y reproducible en train interno, validación y test.
- Fase 2: EDA estadístico **solo sobre el train interno**.
- Hallazgos resumidos en Markdown, legibles por humanos y por la IA en el siguiente sub-proyecto.

**No-objetivos (se deciden en el sub-proyecto 2)**
- Imputar nulos, recortar o transformar outliers, recodificar 96/98, filtrar edades.
- Feature engineering, selección de variables, IV/WoE, modelos.

**Anti-objetivos (sería un fracaso aunque funcione)**
- Que la validación o el test influyan en algún análisis de la Fase 2.
- Que el notebook contenga lógica que no esté en `src/` y testeada.
- Que el resumen afirme cifras que no salen de una ejecución real del notebook.
- Usar `cs-test.csv` para cualquier análisis con target (no tiene etiquetas).

## 3. Datos de entrada

- `data/raw/cs-training.csv`: 150.000 filas; primera columna = ID sin nombre (1..150000); 10 variables + target `SeriousDlqin2yrs`.
- `data/raw/Data Dictionary.xls`: descripción de variables (ya resumida en `docs/data_dictionary.md`).
- `data/raw/cs-test.csv`: sin etiquetas; **no se usa en este sub-proyecto** (reservado para probar la API).

Referencia (perfilado previo, 2026-10-17; el notebook debe reproducirlo, no copiarlo): 609 duplicados
exactos, 37 grupos de filas con mismas variables y distinto target, 0 columnas vacías, 0 valores negativos.

## 4. Fase 1 — Calidad de datos (sobre `cs-training.csv` completo)

Chequeos estructurales, sin mirar distribuciones ni relación con el target:

| Chequeo | Qué reporta |
|---|---|
| Esquema | Columnas presentes vs. esperadas (las 11 del diccionario); faltantes y sobrantes |
| Tipos | Tipo leído por columna vs. tipo esperado (enteros, reales, binario); columnas enteras leídas como float por nulos |
| Formato | Valores no numéricos en columnas numéricas |
| ID | Unicidad, nulos y valores ≤ 0 en el ID |
| Columnas vacías / constantes | Columnas 100 % nulas o con un único valor |
| Target | Solo valores {0, 1}, sin nulos |
| Valores negativos | Conteo por columna (ninguna variable admite negativos) |
| Valores imposibles o sospechosos por dominio | Imposibles: `age` < 18; códigos 96/98 en las 3 variables de atraso. Sospechosos (posibles pero a revisar): `MonthlyIncome` = 0; ratios (`RevolvingUtilization…`, `DebtRatio`) > 1 |
| Duplicados exactos | Filas idénticas en todas las columnas salvo el ID |
| Duplicados contradictorios | Grupos con las mismas 10 variables y distinto target |

**Única acción:** eliminar duplicados exactos (se conserva la primera aparición). Los
contradictorios y los valores imposibles **se documentan, no se tratan**.

## 5. Split

- Sobre los datos sin duplicados exactos, en dos pasos estratificados por el target:
  1. 20 % **test** (apartado; no se abre hasta la evaluación final del modelo campeón).
  2. Del 80 % restante: 80 % **train interno** / 20 % **validación**.
  Resultado aproximado: 64 % / 16 % / 20 %.
- Semilla: `seed` de `configs/params.yaml`. Proporciones en `params.yaml` (nueva sección `split`).
- Salida: `data/interim/train.csv`, `data/interim/valid.csv`, `data/interim/test.csv`, conservando el ID original.
- Reporta: tamaño y tasa de mora por partición, y número de filas con mismas variables que caen
  en particiones distintas (fuga potencial por duplicados contradictorios).

## 6. Fase 2 — EDA estadístico (solo train interno)

| Análisis | Contenido |
|---|---|
| Target | Conteo y tasa de mora (desbalance) |
| Nulos | % de nulos por variable; tasa de mora con vs. sin nulo (¿nulo informativo?) |
| Outliers | Percentiles (1, 5, 25, 50, 75, 95, 99, máx.) por variable; conteo fuera de 1,5·IQR y por encima del p99 |
| Distribuciones | Histogramas por variable separados por target (escala log en variables muy sesgadas) |
| Correlaciones | Matriz de Spearman entre variables (robusta a outliers y sesgo) |
| Tasa de mora por tramos *(decisión de Claude)* | Tasa de mora por decil de cada variable, para ver relación y monotonía con el riesgo |

Ningún resultado de esta fase modifica los datos.

## 7. Entregables y ubicación

| Entregable | Ubicación |
|---|---|
| Carga de datos y validación de esquema | `src/credit_risk/data/` |
| Chequeos de calidad, deduplicación y split | `src/credit_risk/data/` |
| Funciones de perfilado estadístico y gráficos | `src/credit_risk/eda/` (subpaquete nuevo) |
| Tests (con DataFrames sintéticos, nunca `data/raw/`) | `tests/` |
| Notebook que lee resultados y muestra gráficos | `notebooks/01-eda.ipynb` (Fase 1, split y Fase 2 en secciones) |
| Figuras clave | `reports/figures/` |
| Resumen de hallazgos y preguntas abiertas para modelado | `docs/eda_findings.md` |
| Parámetros del split | `configs/params.yaml` (sección `split`) |

El notebook solo llama a funciones de `src/`. `docs/eda_findings.md` se escribe con las cifras
de una ejecución real del notebook y termina con una lista de **preguntas abiertas para el
sub-proyecto 2** (p. ej. tratamiento de 96/98, imputación de `MonthlyIncome`, outliers).

## 8. Decisiones de la usuaria

- Solo diagnóstico: ningún tratamiento salvo eliminar duplicados exactos.
- Fase 1 sobre el archivo completo; Fase 2 solo sobre el train interno.
- Test con etiquetas apartado de `cs-training.csv` (20 %), porque `cs-test.csv` no tiene etiquetas.
- Train interno / validación 80/20, estratificado.
- Lectura principal en notebook.

## 9. Decisiones de Claude (revisables)

- Resumen en Markdown y figuras exportadas, además del notebook (la IA y los evaluadores leen mal un `.ipynb`).
- Chequeos extra en Fase 1: duplicados contradictorios y valores imposibles por dominio.
- Análisis extra en Fase 2: tasa de mora por deciles.
- Correlación de Spearman en lugar de Pearson.
- Particiones guardadas como CSV (sin añadir dependencias de Parquet).
- Dependencias nuevas: `matplotlib` (principal); `ipykernel` y `nbconvert` (dev) para ejecutar el notebook.
- El notebook no corre en el CI (necesita los datos); se verifica localmente ejecutándolo de punta a punta.

## 9.1 Enmienda (2026-10-17, tras la ejecución)

1. La usuaria pidió ver en el notebook las filas de los grupos contradictorios y de cada chequeo
   de dominio. Se implementó y, tras revisar el perfil agregado, **se retiró**: no hacía falta ver
   filas para decidir. Se mantiene la regla de no mostrar filas de datos.
2. Se conserva (añadido por Claude): perfil por chequeo de dominio (`domain_issue_profile`:
   filas, % del total, tasa de mora frente a la base, positivos en juego) y solapamiento de los
   códigos 96/98 (`past_due_code_overlap`).
3. Se añade a la Fase 2: `DebtRatio` según el estado del ingreso (`debt_ratio_by_income_status`),
   para respaldar con evidencia que sin ingreso la columna guarda un monto y no un ratio.
4. Decisiones preliminares de la usuaria para el sub-proyecto 2, registradas en
   `docs/eda_findings.md`: conservar los duplicados contradictorios y las filas 96/98.

## 10. Lo que queda al criterio de quien implemente

Firmas de funciones, nombres internos de módulos, estilo de los gráficos y redacción del
resumen, siempre que respeten esta spec y la constitución.

## 11. Criterios de aceptación

- [x] `uv run poe check` pasa; los tests nuevos usan datos sintéticos y no leen `data/raw/`.
- [x] El notebook se ejecuta de punta a punta sin errores (`uv run jupyter nbconvert --execute`).
- [x] La Fase 1 reproduce las cifras de referencia de la sección 3 (o explica cualquier diferencia).
- [x] `data/interim/{train,valid,test}.csv` existen, no tienen duplicados exactos, sus tamaños suman las filas deduplicadas y la tasa de mora difiere menos de 0,1 pp entre particiones.
- [x] Ninguna función de la Fase 2 recibe la validación ni el test (visible en el notebook).
- [x] Re-ejecutar el split produce exactamente las mismas particiones.
- [x] `docs/eda_findings.md` existe, cita solo cifras del notebook ejecutado y cierra con preguntas abiertas para el sub-proyecto 2.
- [x] `AGENTS.md` apunta a esta spec y a su plan como activos.
