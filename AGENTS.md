# AGENTS.md — reglas de trabajo

Modelo de probabilidad de mora severa (90+ días en los próximos 2 años) sobre el dataset
Give Me Some Credit, servido por una API de inferencia. Contexto completo de negocio:
[docs/business_context.md](docs/business_context.md).

## Antes de tocar código

1. Lee [docs/constitution.md](docs/constitution.md).
2. Lee la spec activa y su plan:
   - Spec activa: `docs/superpowers/specs/2026-10-17-modelado-decision-design.md`
   - Plan activo: `docs/superpowers/plans/2026-10-17-modelado-decision.md`
3. Si la tarea no está cubierta por una spec aprobada, no escribas código: propón primero una spec.

## Mapa de documentos

| Necesito… | Dónde |
|---|---|
| Problema, costos, decisión, salida de la API | `docs/business_context.md` |
| Principios no negociables | `docs/constitution.md` |
| Variables y calidad de datos | `docs/data_dictionary.md` |
| Decisiones técnicas y su porqué | `docs/decisions/ADR-*.md` |
| Specs y planes por sub-proyecto | `docs/superpowers/specs/`, `docs/superpowers/plans/` |
| Registro del uso de IA | `docs/ai_usage_log.md` |

## Comandos

- Instalar: `uv sync` y luego `uv run pre-commit install`
- Lint: `uv run poe lint` · Formatear: `uv run poe format`
- Tests: `uv run poe test` · Todo: `uv run poe check`

## Reglas

- No modifiques ni muevas nada en `data/raw/`.
- No uses `cs-test.csv` para evaluar: no tiene etiquetas.
- Obtén rutas y parámetros con `credit_risk.config.load_config()`; nunca rutas absolutas ni constantes de negocio en el código.
- Ningún test puede depender de `data/raw/` (el CI no tiene los datos).
- `uv run poe check` debe pasar antes de cada commit.
- Un commit por tarea, en inglés, con Conventional Commits (`feat:`, `fix:`, `docs:`, `build:`, `ci:`, `test:`).
- Documentación en español; código e identificadores en inglés.
- Al cerrar una tarea hecha con IA, añade una entrada a `docs/ai_usage_log.md`.
- Al cambiar de sub-proyecto, actualiza aquí la spec y el plan activos.
