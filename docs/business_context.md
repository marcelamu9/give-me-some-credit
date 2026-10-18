# Contexto de negocio

## Problema

Estimar la probabilidad de que un cliente presente **mora severa: 90 días o más de atraso
en los próximos 2 años** (`SeriousDlqin2yrs = 1`), para apoyar la decisión de crédito.

## Tensión de negocio

- **Falso negativo (FN):** aprobar a un cliente que cae en mora → pérdida de capital.
- **Falso positivo (FP):** rechazar a un buen cliente → pérdida de margen y de relación.

**Supuesto de partida:** un FN cuesta **5 veces** lo que un FP (`costs` en
`configs/params.yaml`). Es un supuesto, no un dato: el sub-proyecto 2 incluye análisis de
sensibilidad del umbral frente a este ratio.

## Política de decisión (3 bandas)

| Nivel de riesgo | Decisión sugerida | Regla |
|---|---|---|
| Bajo | Aprobar | probabilidad < umbral inferior |
| Medio | Revisión manual | umbral inferior ≤ probabilidad < umbral superior |
| Alto | Rechazar | probabilidad ≥ umbral superior |

El umbral superior (rechazo) se elige minimizando el costo esperado con la matriz de costos
anterior. La banda de revisión se define por **capacidad de revisión** (`review_capacity` en
`configs/params.yaml`, 10 % de las solicitudes como supuesto): el umbral inferior se fija para
que la franja de revisión, justo por debajo del umbral superior, contenga esa fracción. Los
umbrales aprendidos quedan en la metadata versionada del modelo
(`models/champion_metadata.json`), no en `params.yaml`.

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

## Métricas (confirmadas en la spec 002, §5)

- Discriminación para seleccionar modelo: ROC-AUC, Gini (= 2·AUC − 1), KS.
- PR-AUC como apoyo por el desbalance (~6,7 % de positivos).
- Calibración obligatoria (la API expone una probabilidad): Brier score y curva de calibración.
- Negocio: costo esperado por cliente para fijar umbrales.
