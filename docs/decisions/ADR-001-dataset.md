# ADR-001: Dataset Give Me Some Credit

- **Fecha:** 2026-10-14
- **Estado:** aceptada

## Contexto

El repo arrancó con UCI German Credit (1.000 filas, target good/bad genérico). El objetivo
del proyecto define el target como mora de 90+ días en 2 años.

## Decisión

Usar **Give Me Some Credit** (Kaggle): su target `SeriousDlqin2yrs` coincide exactamente con
la definición de negocio, tiene 150.000 filas y presenta problemas reales (desbalance ~6,7 %,
nulos, códigos especiales, outliers). German Credit se retiró del repositorio.

## Consecuencias

- Solo `cs-training.csv` sirve para entrenar y evaluar; `cs-test.csv` no tiene etiquetas.
- El test set propio se separa de `cs-training.csv`.
- Los datos se descargan manualmente de Kaggle; no se versionan.
