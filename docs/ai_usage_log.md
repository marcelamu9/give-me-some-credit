# Bitácora de uso de IA

Registro breve de qué se delegó a la IA, cómo se validó y qué se corrigió.

| Fecha | Tarea | Delegado a la IA | Validación humana / evidencia | Correcciones |
|---|---|---|---|---|
| 2026-10-14 | Brainstorming de objetivos y estructura | Revisión de estructura, preguntas de alcance y stack (Claude Opus + superpowers:brainstorming) | Respuestas de la usuaria a cada ronda de preguntas | Detectó que los datos eran German Credit y no GMSC; se reemplazaron |
| 2026-10-14 | Spec y plan de cimientos | Redacción de spec y plan; perfilado de `cs-training.csv` | Revisión de la spec por la usuaria | Makefile → poethepoet (no hay `make` en Windows) |
| 2026-10-14 | Ejecución del plan de cimientos | Implementación tarea a tarea (Claude Sonnet + superpowers:executing-plans) | `uv run poe check`: 19 passed, lint limpio; clon limpio también verde | ruff 0.17 formatea bloques de código en `.md` y `poe lint` fallaba sobre el plan: se ejecutó `poe format` (commit `style` aparte); `.superpowers/` añadido a `.gitignore`; trabajo en el repo sin worktree |
| 2026-10-17 | Ejecución del plan Datos + EDA | Implementación tarea a tarea (Claude Sonnet + superpowers:executing-plans): config del split, calidad, split, perfilado, gráficos, notebook y hallazgos | `uv run poe check`: 63 passed, lint limpio; notebook ejecutado de punta a punta; hash idéntico de `data/interim/*.csv` en dos ejecuciones | Trabajo directamente sobre `main` sin worktree; ajustes menores de longitud de línea para ruff; se instaló el kernel de Jupyter a nivel de usuario; el notebook imprime rutas relativas para no versionar rutas locales |
