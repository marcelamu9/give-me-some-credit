# Comparación de modelos

Generado por `uv run poe train` el 2026-10-18T02:59:33+00:00 (semilla 42). Cifras de validación salvo la última sección.

## Candidatos (validación)

| modelo | AUC CV (± std) | AUC | Gini | KS | PR-AUC | Brier | Brier sin calibrar | costo esperado | t* | búsqueda (s) | calibración (s) | latencia (ms/cliente) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic | 0.8534 ± 0.0039 | 0.8557 | 0.7113 | 0.5469 | 0.3873 | 0.0498 | 0.0501 | 0.2219 | 0.1800 | 35.1000 | 1.7000 | 49.4600 |
| random_forest | 0.8650 ± 0.0041 | 0.8660 | 0.7319 | 0.5715 | 0.4052 | 0.0492 | 0.0493 | 0.2160 | 0.1750 | 202.6000 | 16.0000 | 447.7400 |
| lightgbm | 0.8653 ± 0.0037 | 0.8681 | 0.7362 | 0.5790 | 0.4093 | 0.0490 | 0.0491 | 0.2141 | 0.1400 | 239.9000 | 5.2000 | 93.9000 |

La latencia se midió prediciendo de una fila en una sobre 200 filas de validación (mediana de repeticiones); el costo esperado es por cliente con FN = 5 y FP = 1.

## Campeón

**lightgbm**. Regla: menor costo esperado por cliente en validación; son elegibles los modelos a menos de 2 % de ese costo cuyo Brier no sea más de 10 % peor que el del modelo de menor costo; entre ellos gana la regresión logística (interpretable) y, si no es elegible, el de menor latencia de predicción (más barato de operar).

## Umbrales y bandas (validación)

- `t*` (rechazo) = 0.140 (umbral teórico FP/(FP+FN) = 0.167).

- `t_low` (aprobación) = 0.073, con capacidad de revisión de 10%.

| banda | clientes | proporción | default_rate |
|---|---|---|---|
| approve | 18419 | 0.7706 | 0.0208 |
| review | 2395 | 0.1002 | 0.0923 |
| reject | 3089 | 0.1292 | 0.3224 |

## Sensibilidad al costo (razón FN:FP)

| fn_fp_ratio | t_star | t_low | approval_rate | default_rate_approved | expected_cost |
|---|---|---|---|---|---|
| 3.0000 | 0.2800 | 0.1051 | 0.9339 | 0.0410 | 0.1522 |
| 5.0000 | 0.1400 | 0.0725 | 0.8708 | 0.0291 | 0.2141 |
| 10.0000 | 0.1000 | 0.0590 | 0.8254 | 0.0250 | 0.3349 |
| 20.0000 | 0.0450 | 0.0273 | 0.6710 | 0.0145 | 0.4658 |

## Sensibilidad a la capacidad de revisión

| capacity | t_low | approve_rows | approve_default_rate | review_rows | review_default_rate | reject_rows | reject_default_rate |
|---|---|---|---|---|---|---|---|
| 0.0500 | 0.0973 | 19619 | 0.0248 | 1195 | 0.0987 | 3089 | 0.3224 |
| 0.1000 | 0.0725 | 18419 | 0.0208 | 2395 | 0.0923 | 3089 | 0.3224 |
| 0.2000 | 0.0450 | 16031 | 0.0144 | 4783 | 0.0782 | 3089 | 0.3224 |

## Resultado en test (única evaluación)

| roc_auc | gini | ks | pr_auc | brier |
|---|---|---|---|---|
| 0.8650 | 0.7301 | 0.5891 | 0.4010 | 0.0488 |

Costo esperado en test con `t*` = 0.140: 0.2156.

| banda | clientes | proporción | default_rate |
|---|---|---|---|
| approve | 22861 | 0.7651 | 0.0191 |
| review | 3020 | 0.1011 | 0.1010 |
| reject | 3998 | 0.1338 | 0.3154 |
