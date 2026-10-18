# ADR-002: Selección del modelo por costo esperado

- **Fecha:** 2026-10-17
- **Estado:** aceptada (regla de desempate corregida el 2026-10-17, ver "Revisión")

## Contexto

Hay tres candidatos (logística, Random Forest, LightGBM) con métricas de discriminación casi
iguales en validación (AUC de 0,8557 a 0,8681). El negocio no se mide en AUC: un falso negativo
(aprobar a quien cae en mora) cuesta 5 veces un falso positivo (`costs` en `params.yaml`).

## Decisión

- **El AUC guía la búsqueda de hiperparámetros** (`RandomizedSearchCV`, validación cruzada de 5
  pliegues): no depende de un umbral y es estable entre pliegues.
- **El costo esperado por cliente en valid decide el campeón**, calculado con el umbral
  óptimo de cada modelo calibrado.
- **Regla de desempate operativa:** son elegibles los modelos a menos de 2 % del menor costo
  cuyo Brier no sea más de 10 % peor que el del modelo de menor costo. Entre ellos gana la
  **logística** (interpretable); si no es elegible, el de **menor latencia de predicción** (más
  barato de operar).
- La calibración (isotónica) es obligatoria porque la API expone una probabilidad y el umbral
  óptimo depende de que esté calibrada.

Resultado: LightGBM tiene el menor costo (0,2141) y la menor latencia entre los elegibles;
Random Forest (0,2160) es elegible pero unas 4,8 veces más lento; la logística (0,2219) queda
fuera del 2 %. **El campeón es LightGBM.**

## Revisión

La primera versión de esta decisión usaba un orden fijo de "complejidad" (logística < Random
Forest < LightGBM) y eligió Random Forest, peor que LightGBM en costo, AUC, KS y Brier, y unas
4,8 veces más lento. Ese orden no medía lo que importa al negocio (explicar y operar el modelo),
así que se sustituyó por la regla operativa de arriba (enmienda 10.1 de la spec 002).

## Alternativas descartadas

- **Elegir solo por AUC:** aquí coincide con el resultado (LightGBM), pero no considera costo
  ni calibración, y en otro escenario podría elegir un modelo más caro para el negocio.
- **Orden fijo de complejidad:** se probó y se descartó (ver "Revisión").
- **Elegir por F1:** trata FN y FP como igual de costosos, contra la matriz de costos del negocio.
- **`class_weight` o remuestreo:** distorsionan la probabilidad; el costo se gestiona con el
  umbral y la probabilidad queda calibrada.

## Consecuencias

- El campeón puede cambiar si cambian `costs` o los datos; la regla se reevalúa con
  `uv run poe train`.
- La tolerancia de 2 % y de 10 % en Brier son criterios de esta decisión, no resultados
  empíricos; se documentan para poder discutirlos.
- La latencia medida varía con la carga de la máquina (LightGBM: 37,7 ms en una ejecución y
  93,9 ms en otra), pero el orden entre modelos se mantuvo; si dos modelos elegibles quedaran
  muy cerca en latencia, convendría medirla con más repeticiones.
- La ficha del modelo (`docs/model_card.md`) recoge las cifras y limitaciones.
