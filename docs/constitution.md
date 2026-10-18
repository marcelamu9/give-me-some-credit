# Constitución del proyecto

Principios no negociables. Si una spec o un plan los contradice, gana la constitución
y se corrige la spec.

1. **Reproducibilidad.** `uv.lock` versionado; semilla global `seed` en `configs/params.yaml`
   usada en todo proceso aleatorio. Mismo código + mismos datos = mismos resultados.
2. **El test set es sagrado.** Se separa una sola vez (estratificado por el target) y no se mira
   hasta la evaluación final del modelo campeón. La selección de modelo y umbrales usa
   validación (o CV sobre train).
3. **Sin leakage.** Imputación, escalado, binning, calibración y cualquier ajuste se entrenan
   solo con datos de entrenamiento, dentro de pipelines.
4. **Código en `src/`, exploración en notebooks.** La lógica reutilizable vive en
   `src/credit_risk/`; los notebooks solo exploran y llaman a funciones de `src/`.
5. **TDD y evidencia.** Todo código en `src/` nace con su test. Ningún resultado (métrica,
   test verde, mejora) se afirma sin la salida del comando que lo demuestra.
6. **Spec antes que código.** Cada sub-proyecto tiene spec aprobada y plan antes de escribir
   código. Un commit por tarea.
7. **Negocio en configuración.** Las entradas de negocio (costos, capacidad de revisión,
   escenarios de sensibilidad) viven en `configs/params.yaml`; los umbrales que se aprenden del
   modelo viven en su metadata versionada (`models/champion_metadata.json`). Nada de esto se
   hardcodea.
8. **Datos crudos inmutables.** `data/raw/` no se modifica ni se versiona; las transformaciones
   escriben en `data/interim/` o `data/processed/`.
