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
