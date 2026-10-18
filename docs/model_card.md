# Ficha del modelo — probabilidad de mora severa

Todas las cifras salen de `models/champion_metadata.json`, de `reports/model_comparison.md`
(ambos generados por `uv run poe train`) y del notebook ejecutado `notebooks/02-modelado.ipynb`.
Spec: `docs/superpowers/specs/2026-10-17-modelado-decision-design.md`.

## Propósito y uso previsto

Estimar la probabilidad de que un solicitante tenga mora de 90+ días en los próximos 2 años
(`SeriousDlqin2yrs`) y traducirla en una **sugerencia** de decisión: aprobar, revisión manual o
rechazar. La decisión final es humana o de política (ver `docs/business_context.md`). No está
pensado para otras poblaciones ni para fijar tasas o límites de crédito.

## Datos

Partición estratificada de `cs-training.csv` (spec 001), sin filas con `age < 18`:

| Partición | Filas |
|---|---|
| train | 95.608 |
| valid | 23.903 |
| test | 29.879 |

El hash SHA-256 de cada CSV de entrada está en la metadata.

## Tratamiento de faltantes y atípicos

Transformadores de scikit-learn ajustados **solo con train** (`credit_risk.features.treatment`).

| # | Tema | Tratamiento | Porqué |
|---|---|---|---|
| U1 | Duplicados contradictorios | Se conservan | Decisión de negocio |
| U2 | Códigos 96/98 en atrasos | Indicador `has_special_past_due_code` y conteos a 0 | El código marca "desconocido", no 96 atrasos; el indicador conserva la señal |
| U3 | `MonthlyIncome` faltante | Mediana por decil de edad × dependientes | La relación ingreso-edad es en U invertida |
| U4 | `RevolvingUtilization…` | Recorte en el p99 de train | Valores absurdos distorsionan modelos y escalas |
| U5 | `DebtRatio` | Sin ingreso: monto ÷ ingreso imputado; luego recorte en p99 | Sin ingreso el campo es un monto, no un ratio |
| C1 | `MonthlyIncome = 0` | Se trata como no disponible | El ratio no se puede calcular con ingreso 0 |
| C2 | Indicadores de ingreso | Un solo `income_unavailable` | Dos columnas iguales rompen la logística |
| C3 | `NumberOfDependents` faltante | Se imputa 0 | Sus nulos caen dentro de `income_unavailable` |
| C4 | Grupos de imputación | Decil × dependientes `0..4+`; grupo < 30 filas → decil → global | Evita medianas de grupos diminutos |
| C5 | Ingreso extremo | Recorte en el p99 de train | Evita ratios recalculados absurdos |
| C6 | `age < 18` | Se eliminan de las tres particiones | Edad imposible para un crédito |
| C7 | Conteos con colas largas | Recorte en p99, solo en la logística | A los árboles no les afecta |
| C8 | Escala | `log1p` + estandarización, solo en la logística | La logística es sensible a escala y sesgo |

En valid, el 19,63 % de los ingresos eran nulos y el 0,99 % eran 0; tras el tratamiento no
queda ninguno. El 20,62 % de las filas tiene `income_unavailable = 1` y el 0,16 % tiene
`has_special_past_due_code = 1`.

## Candidatos y búsqueda

`RandomizedSearchCV` con 20 iteraciones por modelo, validación cruzada estratificada de 5
pliegues sobre train, `scoring = roc_auc`, semilla 42; sin `class_weight` ni remuestreo.

| Modelo | Espacio buscado | Mejores hiperparámetros |
|---|---|---|
| Logística (baseline) | `C` log-uniforme 1e-3–1e2; penalización L1/L2 | `C` = 0,0083; L1 |
| Random Forest | árboles 100–300; profundidad 4–12; hoja mínima 20–200; `max_features` sqrt/0,5 | 202 árboles; profundidad 11; hoja mínima 40; sqrt |
| LightGBM | árboles 200–800; tasa 0,01–0,1; hojas 15–63; mínimo por hoja 20–200; submuestreo y columnas 0,6–1; `reg_lambda` 0–5 | 227 árboles; tasa 0,040; 21 hojas; mínimo por hoja 60 |

Tiempos medidos: búsqueda de 33,3 s (logística), 177,3 s (Random Forest) y 98,0 s (LightGBM).
Entrenamiento completo con `uv run poe train`: 8 min 10 s y 8 min 7 s en dos ejecuciones.
Cada candidato se calibra con regresión isotónica (5 pliegues sobre train).

## Resultados en valid

| Modelo | AUC CV (± std) | AUC | Gini | KS | PR-AUC | Brier | Brier sin calibrar | Costo esperado | `t*` | Latencia (ms/cliente) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logística | 0,8534 ± 0,0039 | 0,8557 | 0,7113 | 0,5469 | 0,3873 | 0,0498 | 0,0501 | 0,2219 | 0,180 | 50,14 |
| Random Forest | 0,8650 ± 0,0041 | 0,8660 | 0,7319 | 0,5715 | 0,4052 | 0,0492 | 0,0493 | 0,2160 | 0,175 | 179,60 |
| LightGBM | 0,8653 ± 0,0037 | 0,8681 | 0,7362 | 0,5790 | 0,4093 | 0,0490 | 0,0491 | 0,2141 | 0,140 | 37,63 |

La latencia es la mediana por cliente prediciendo de una fila en una, sobre 200 filas de valid
(el modelo calibrado completo, con el preprocesamiento en pandas).

## Criterio de selección

El **costo esperado por cliente** en valid (FN = 5, FP = 1) decide; el AUC solo guía la
búsqueda (ver `docs/decisions/ADR-002-seleccion-modelo.md`). LightGBM tiene el menor costo
(0,2141). Random Forest (0,2160) queda a menos de 2 % de ese costo y es más simple, y su Brier
no es peor en más de 10 %, así que **el campeón es Random Forest**. La logística (0,2219)
queda fuera de la tolerancia de 2 %.

## Umbrales y bandas

- `t*` = 0,175, frente al umbral teórico FP/(FP+FN) = 0,167 para probabilidades bien
  calibradas: coinciden de forma aproximada, como se espera de un modelo calibrado.
- Capacidad de revisión supuesta: 10 % de las solicitudes → `t_low` = 0,081.

| Banda en valid | Clientes | Proporción | Mora observada |
|---|---|---|---|
| Aprobar (`p < 0,081`) | 18.918 | 79,14 % | 2,24 % |
| Revisión (`0,081 ≤ p < 0,175`) | 2.391 | 10,00 % | 11,38 % |
| Rechazar (`p ≥ 0,175`) | 2.594 | 10,85 % | 34,93 % |

El costo esperado se calcula para la política binaria (`t*`). Para las tres bandas se reportan
volumen y mora real, sin asumir un costo ni un resultado para la revisión humana.

## Sensibilidad

**A la razón de costos FN:FP** (valid):

| FN:FP | `t*` | `t_low` | Aprobación | Mora entre aprobados | Costo esperado |
|---|---|---|---|---|---|
| 3 | 0,305 | 0,107 | 93,75 % | 4,17 % | 0,1519 |
| 5 | 0,175 | 0,081 | 89,15 % | 3,26 % | 0,2160 |
| 10 | 0,090 | 0,057 | 81,64 % | 2,45 % | 0,3370 |
| 20 | 0,050 | 0,030 | 69,15 % | 1,63 % | 0,4778 |

**A la capacidad de revisión** (valid; `t*` fijo en 0,175):

| Capacidad | `t_low` | Aprobar | Mora aprobados | Revisión | Mora revisión |
|---|---|---|---|---|---|
| 5 % | 0,110 | 20.111 | 2,68 % | 1.198 | 13,02 % |
| 10 % | 0,081 | 18.918 | 2,24 % | 2.391 | 11,38 % |
| 20 % | 0,050 | 16.528 | 1,63 % | 4.781 | 8,91 % |

## Resultado en test

Evaluado una sola vez, con el campeón calibrado y los umbrales ya fijados en valid.

| Métrica | Valid | Test |
|---|---|---|
| AUC | 0,8660 | 0,8639 |
| Gini | 0,7319 | 0,7277 |
| KS | 0,5715 | 0,5877 |
| PR-AUC | 0,4052 | 0,4011 |
| Brier | 0,0492 | 0,0490 |
| Costo esperado (`t*` = 0,175) | 0,2160 | 0,2137 |

Bandas en test: aprobar 78,61 % (mora 2,12 %), revisión 10,30 % (11,30 %), rechazar 11,08 %
(34,90 %). Los resultados en test son coherentes con los de valid.

## Importancia de variables

SHAP (media absoluta, 2.000 filas de valid) del Random Forest. Orden de mayor a menor:
`RevolvingUtilizationOfUnsecuredLines`, `NumberOfTimes90DaysLate`,
`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `age`,
`DebtRatio`, `NumberOfOpenCreditLinesAndLoans`, `MonthlyIncome`,
`NumberRealEstateLoansOrLines`, `NumberOfDependents`, `has_special_past_due_code`,
`income_unavailable`. Figura: `reports/figures/importance_champion.png`.

## Limitaciones y riesgos

- **Supuestos de negocio:** la razón de costos 5:1 y la capacidad de revisión del 10 % no son
  datos; la sensibilidad muestra cuánto cambian los umbrales (por ejemplo, `t*` de 0,305 a
  0,050 entre 3:1 y 20:1).
- **Uso de `age`:** es una variable sensible en crédito y es la quinta en importancia; debe
  revisarse con el área de riesgo y de cumplimiento antes de un uso real.
- **Población:** se eliminaron los menores de 18 años; la API deberá rechazar esas edades.
- **Códigos 96/98:** su significado no está documentado; se tratan como "desconocido" con un
  indicador.
- **`DebtRatio`:** sin ingreso es un monto; se recalcula con el ingreso imputado, que es una
  estimación.
- **Latencia:** el Random Forest tarda 179,6 ms por cliente en esta medición (una fila a la
  vez, con el preprocesamiento en pandas), frente a 37,6 ms de LightGBM; conviene revisarlo al
  construir la API.
- **Umbrales ligados a valid:** no se reentrenó con train + valid; los umbrales se fijaron con
  el modelo entrenado solo con train.
- **Reproducibilidad:** dos ejecuciones dan el mismo campeón, los mismos umbrales y las mismas
  métricas; la única diferencia observada fue de 1e-17 en el Brier de test.

## Cuándo reentrenar

Ante una caída sostenida del AUC o de la calibración frente a los valores de esta ficha, un
cambio relevante en la mezcla de bandas (por ejemplo, deriva de PSI en el monitoreo futuro) o
un cambio en los costos o en la capacidad de revisión de `configs/params.yaml`.
