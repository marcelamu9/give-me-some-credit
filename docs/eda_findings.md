# Hallazgos de datos y EDA

Resumen de `notebooks/01-eda.ipynb` ejecutado de punta a punta (spec:
`docs/superpowers/specs/2026-10-17-datos-eda-design.md`). Todas las cifras salen de esa
ejecución. La Fase 1 usa el archivo completo; la Fase 2 usa **solo el train interno**.
Este documento diagnostica: no se imputó, recortó ni recodificó nada.

## 1. Resumen ejecutivo

- Los datos reproducen el perfilado previo: 609 duplicados exactos y 37 grupos de filas con las mismas variables y distinto target. Tras quitar los duplicados exactos quedan 149.391 filas.
- Desbalance: la tasa de mora es 6,70 % en el train interno (6.406 de 95.609 filas), idéntica entre particiones (diferencia máxima 0,0025 pp).
- `MonthlyIncome` tiene 19,6 % de nulos y `NumberOfDependents` 2,6 %. En ambos, la mora entre los nulos es **menor** que entre los presentes (5,48 % vs 7,00 % y 4,45 % vs 6,76 %).
- Los códigos 96/98 aparecen en las 3 variables de atraso (269 filas por variable en el archivo completo) y distorsionan máximos y tramos.
- Hay valores extremos graves: `RevolvingUtilizationOfUnsecuredLines` llega a 50.708, `DebtRatio` a 329.664 y `MonthlyIncome` a 3.008.750; `DebtRatio` > 1 ocurre en 35.137 filas.
- El riesgo es claramente monótono con la utilización de líneas de crédito (2,7 % en el primer decil frente a 23,4 % en el último) y decrece con la edad (11,3 % hasta los 33 años frente a 2,1 % por encima de los 72).
- Ninguna correlación de Spearman entre variables supera 0,47 en valor absoluto; no hay redundancia fuerte.

## 2. Fase 1 — Calidad (archivo completo, 150.000 filas)

| Chequeo | Resultado |
|---|---|
| Esquema | Las 11 columnas esperadas; sin faltantes ni sobrantes |
| Tipos | Todos coinciden salvo `NumberOfDependents`: entera pero leída como float por los nulos |
| Formato | 0 valores no numéricos en todas las columnas |
| ID | Único, sin nulos, sin valores ≤ 0 |
| Vacías / constantes | Ninguna |
| Target | Solo {0, 1}, sin nulos |
| Negativos | 0 en todas las columnas |
| `age` < 18 | 1 fila (imposible) |
| Códigos 96/98 | 269 filas en cada una de las 3 variables de atraso (imposibles) |
| `MonthlyIncome` = 0 | 1.634 filas (sospechoso) |
| `RevolvingUtilization…` > 1 | 3.321 filas (sospechoso) |
| `DebtRatio` > 1 | 35.137 filas (sospechoso) |
| Duplicados exactos | 609 (referencia: 609, coincide) |
| Duplicados contradictorios | 37 grupos (referencia: 37, coincide) |

Única acción: eliminar los 609 duplicados exactos (150.000 → 149.391 filas).

### Perfil de los casos de dominio y duplicados contradictorios (enmienda 2026-10-17)

Tasa de mora base del archivo completo: 6,68 %. "Positivos" = casos de mora que se perderían si se eliminaran esas filas.

| Chequeo | Filas | % filas | Tasa de mora | Positivos |
|---|---|---|---|---|
| `age` < 18 | 1 | 0,00 % | 0 % | 0 |
| Códigos 96/98 | 269 | 0,18 % | **54,6 %** | 147 |
| `MonthlyIncome` = 0 | 1.634 | 1,09 % | 4,0 % | 66 |
| `RevolvingUtilization…` > 1 | 3.321 | 2,21 % | **37,3 %** | 1.237 |
| `DebtRatio` > 1 | 35.137 | 23,42 % | 6,5 % | 2.291 |

- **Los 96/98 van juntos:** las 269 filas tienen el código en las tres variables de atraso a la vez (y siempre el mismo código en las tres). Es un registro especial, no tres errores independientes.
- **Las filas 96/98 son de muy alto riesgo** (54,6 % de mora, 8 veces la base). Eliminarlas quitaría 147 positivos y una señal fuerte: apunta a **tratarlas** (por ejemplo, indicador propio), no a borrarlas.
- **`RevolvingUtilization…` > 1 también concentra riesgo** (37,3 %, 1.237 positivos): eliminarlas sería muy costoso; los valores extremos se tratan, no se descartan.
- **`DebtRatio` > 1 es demasiado frecuente (23 %) y con mora normal** como para ser un error: es una característica de la distribución.
- **Duplicados contradictorios:** 145 filas en 37 grupos, 47 positivos. 137 de esas filas no tienen `MonthlyIncome`, y 10 de los 37 grupos (39 filas) son además registros 96/98 (perfiles casi vacíos: edad, códigos y ceros). Parecen registros con poca información más que errores de captura.

## 3. Split

| Partición | Filas | Tasa de mora |
|---|---|---|
| train | 95.609 | 6,700 % |
| valid | 23.903 | 6,698 % |
| test | 29.879 | 6,700 % |

- Las tres sumas dan 149.391 y no hay duplicados exactos entre particiones. Rehacer el split produce archivos idénticos (mismo hash SHA-256).
- 19 combinaciones de variables aparecen en más de una partición (fuga potencial por filas con mismas variables).

## 4. Fase 2 — EDA (train interno, 95.609 filas)

### Target
- 6.406 positivos, tasa 6,70 %: desbalance de aproximadamente 1 a 14. Figura: [target_distribution.png](../reports/figures/target_distribution.png).

### Nulos
- Solo `MonthlyIncome` (18.736, 19,60 %) y `NumberOfDependents` (2.451, 2,56 %) tienen nulos.
- Mora con nulo vs sin nulo: 5,48 % vs 6,997 % en ingreso y 4,45 % vs 6,76 % en dependientes; el nulo parece informativo. Figura: [missing.png](../reports/figures/missing.png).

### Outliers
- p99 / máximo: `RevolvingUtilization…` 1,09 / 50.708; `DebtRatio` 4.982,68 / 329.664; `MonthlyIncome` 25.000 / 3.008.750; `age` 87 / 109.
- Las variables de atraso tienen p99 de 2 o 3 y máximo 98 (el código de "desconocido").
- Fuera de 1,5·IQR: `DebtRatio` 20.051 filas y `NumberOfTime30-59…` 15.239, porque casi todo son ceros y cualquier valor positivo cuenta como atípico.

### Distribuciones
- Histogramas por target en `reports/figures/dist_<variable>.png` (escala `log1p` en los ratios y en `MonthlyIncome`), por ejemplo [dist_age.png](../reports/figures/dist_age.png).

### Correlaciones (Spearman)
- Las más altas: `NumberOfOpenCreditLinesAndLoans`–`NumberRealEstateLoansOrLines` 0,47; `DebtRatio`–`NumberRealEstateLoansOrLines` 0,40; `MonthlyIncome`–`NumberRealEstateLoansOrLines` 0,39.
- Entre las variables de atraso: 0,25 a 0,31. Figura: [corr_spearman.png](../reports/figures/corr_spearman.png).

### Tasa de mora por deciles
- `RevolvingUtilization…`: sube desde 1,2 % (segundo decil) hasta 23,4 % (último). El primer decil (2,7 %) rompe la monotonía.
- `age`: baja de forma monótona de 11,3 % a 2,1 %.
- `MonthlyIncome`: baja de 9,7 % a 4,5 % a partir del segundo tramo.
- `DebtRatio` no es monótona: sube hasta 11,3 % y cae a 5,9 % y 4,7 % en los dos últimos deciles (valores extremos).
- `NumberOfTimes90DaysLate` y `NumberOfTime60-89…` quedan en un solo tramo porque `qcut` agrupa los empates; el análisis por deciles no sirve para ellas. `NumberOfTime30-59…` queda en 2 tramos: ≥ 2 atrasos tiene 32,15 % de mora frente a 5,31 %.
- Figuras: `reports/figures/decile_<variable>.png`, por ejemplo [decile_age.png](../reports/figures/decile_age.png).

## 5. Preguntas abiertas para el sub-proyecto 2

1. Códigos 96/98 en las 3 variables de atraso (269 filas, siempre juntos, 54,6 % de mora): ¿indicador propio + recodificar los conteos, o categoría aparte? Eliminar perdería 147 positivos.
2. `MonthlyIncome`: ¿imputar o añadir indicador de nulo (la mora es menor entre los nulos)? ¿Qué hacer con los 1.634 ceros?
3. `NumberOfDependents`: nulos informativos (4,45 % de mora); ¿imputar con 0 o con indicador?
4. Ratios con valores extremos (`RevolvingUtilization…`, `DebtRatio`): ¿recorte por percentil o transformación?
5. `age` < 18 (1 fila): ¿filtrar?
6. Duplicados contradictorios (37 grupos, 145 filas, 47 positivos, casi todos sin `MonthlyIncome`) y 19 combinaciones repetidas entre particiones: ¿conservar (son pocos y parecen perfiles con poca información) o eliminar? ¿Efecto sobre la evaluación?
7. Variables correlacionadas (apertura de créditos, préstamos inmobiliarios, ingreso): ¿aportan información redundante?
8. Variables de atraso casi constantes: ¿agrupar en tramos para analizar su relación con el riesgo?
