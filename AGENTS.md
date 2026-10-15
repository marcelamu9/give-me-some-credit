# Objetivo del proyecto

Desarrollar un modelo de clasificación que estime la probabilidad de que un cliente presente una mora severa de 90 días o más durante los próximos dos años.

# Objetivo de negocio

Apoyar decisiones de riesgo crediticio equilibrando:

- El costo de aprobar a un cliente con alto riesgo de incumplimiento.
- El costo de rechazar innecesariamente a un buen cliente.

# Contexto de uso esperado

El modelo se consumirá inicialmente mediante una API de inferencia.

La API recibirá las variables de un cliente y devolverá:

- probabilidad estimada de mora severa;
- nivel de riesgo;
- decisión sugerida.

La arquitectura definitiva podrá ajustarse durante el proyecto.


## Reglas

* Lee `docs/constitution.md` y la spec activa (`specs/NNN-*/`) antes de tocar código.
