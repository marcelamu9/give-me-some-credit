# Comparación de modelos

Generado por `uv run poe train` el 2026-10-18T02:15:43+00:00 (semilla 42). Cifras de validación salvo la última sección.

## Candidatos (validación)

| modelo | AUC CV (± std) | AUC | Gini | KS | PR-AUC | Brier | Brier sin calibrar | costo esperado | t* | búsqueda (s) | calibración (s) | latencia (ms/cliente) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic | 0.8534 ± 0.0039 | 0.8557 | 0.7113 | 0.5469 | 0.3873 | 0.0498 | 0.0501 | 0.2219 | 0.1800 | 32.7000 | 1.7000 | 49.0200 |
| random_forest | 0.8650 ± 0.0041 | 0.8660 | 0.7319 | 0.5715 | 0.4052 | 0.0492 | 0.0493 | 0.2160 | 0.1750 | 177.1000 | 9.1000 | 179.6100 |
| lightgbm | 0.8653 ± 0.0037 | 0.8681 | 0.7362 | 0.5790 | 0.4093 | 0.0490 | 0.0491 | 0.2141 | 0.1400 | 97.7000 | 2.2000 | 37.7400 |

La latencia se midió prediciendo de una fila en una sobre 200 filas de validación (mediana de repeticiones); el costo esperado es por cliente con FN = 5 y FP = 1.

## Campeón

**random_forest**. Regla: menor costo esperado por cliente en validación; entre los modelos a menos de 2 % de ese costo gana el más simple (logística < Random Forest < LightGBM), salvo que su Brier sea más de 10 % peor que el del modelo de menor costo.

## Umbrales y bandas (validación)

- `t*` (rechazo) = 0.175 (umbral teórico FP/(FP+FN) = 0.167).

- `t_low` (aprobación) = 0.081, con capacidad de revisión de 10%.

| banda | clientes | proporción | default_rate |
|---|---|---|---|
| approve | 18918 | 0.7914 | 0.0224 |
| review | 2391 | 0.1000 | 0.1138 |
| reject | 2594 | 0.1085 | 0.3493 |

## Sensibilidad al costo (razón FN:FP)

| fn_fp_ratio | t_star | t_low | approval_rate | default_rate_approved | expected_cost |
|---|---|---|---|---|---|
| 3.0000 | 0.3050 | 0.1065 | 0.9375 | 0.0417 | 0.1519 |
| 5.0000 | 0.1750 | 0.0810 | 0.8915 | 0.0326 | 0.2160 |
| 10.0000 | 0.0900 | 0.0573 | 0.8164 | 0.0245 | 0.3370 |
| 20.0000 | 0.0500 | 0.0300 | 0.6915 | 0.0163 | 0.4778 |

## Sensibilidad a la capacidad de revisión

| capacity | t_low | approve_rows | approve_default_rate | review_rows | review_default_rate | reject_rows | reject_default_rate |
|---|---|---|---|---|---|---|---|
| 0.0500 | 0.1096 | 20111 | 0.0268 | 1198 | 0.1302 | 2594 | 0.3493 |
| 0.1000 | 0.0810 | 18918 | 0.0224 | 2391 | 0.1138 | 2594 | 0.3493 |
| 0.2000 | 0.0500 | 16528 | 0.0163 | 4781 | 0.0891 | 2594 | 0.3493 |

## Resultado en test (única evaluación)

| roc_auc | gini | ks | pr_auc | brier |
|---|---|---|---|---|
| 0.8639 | 0.7277 | 0.5877 | 0.4011 | 0.0490 |

Costo esperado en test con `t*` = 0.175: 0.2137.

| banda | clientes | proporción | default_rate |
|---|---|---|---|
| approve | 23488 | 0.7861 | 0.0212 |
| review | 3079 | 0.1030 | 0.1130 |
| reject | 3312 | 0.1108 | 0.3490 |
