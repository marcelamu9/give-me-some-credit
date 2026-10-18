# ADR-002: Selección del modelo por costo esperado

- **Fecha:** 2026-10-17
- **Estado:** aceptada

## Contexto

Hay tres candidatos (logística, Random Forest, LightGBM) con métricas de discriminación casi
iguales en validación (AUC de 0,8557 a 0,8681). El negocio no se mide en AUC: un falso negativo
(aprobar a quien cae en mora) cuesta 5 veces un falso positivo (`costs` en `params.yaml`).

## Decisión

- **El AUC guía la búsqueda de hiperparámetros** (`RandomizedSearchCV`, validación cruzada de 5
  pliegues): no depende de un umbral y es estable entre pliegues.
- **El costo esperado por cliente en valid decide el campeón**, calculado con el umbral
  óptimo de cada modelo calibrado.
- **Regla de simplicidad:** si dos modelos difieren en menos de 2 % de costo, gana el más
  simple (logística < Random Forest < LightGBM), salvo que su Brier sea más de 10 % peor que el
  del modelo de menor costo.
- La calibración (isotónica) es obligatoria porque la API expone una probabilidad y el umbral
  óptimo depende de que esté calibrada.

Resultado: LightGBM tiene el menor costo (0,2141); Random Forest (0,2160) queda dentro de la
tolerancia y es más simple, y su Brier no empeora más de 10 %; por eso es el campeón.

## Alternativas descartadas

- **Elegir solo por AUC:** habría elegido LightGBM sin considerar costo ni calibración, y
  ocultaría que la diferencia práctica con Random Forest es de 0,0019 en costo por cliente.
- **Elegir por F1:** trata FN y FP como igual de costosos, contra la matriz de costos del negocio.
- **`class_weight` o remuestreo:** distorsionan la probabilidad; el costo se gestiona con el
  umbral y la probabilidad queda calibrada.

## Consecuencias

- El campeón puede cambiar si cambian `costs` o los datos; la regla se reevalúa con
  `uv run poe train`.
- La tolerancia de 2 % y de 10 % en Brier son criterios de esta decisión, no resultados
  empíricos; se documentan para poder discutirlos.
- Random Forest tuvo la mayor latencia por cliente en la medición (179,6 ms frente a 37,6 ms de
  LightGBM); si la API exige menos, la regla debería incluir latencia.
- La ficha del modelo (`docs/model_card.md`) recoge las cifras y limitaciones.
