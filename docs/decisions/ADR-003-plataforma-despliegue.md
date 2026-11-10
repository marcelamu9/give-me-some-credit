# ADR-003: Plataforma de despliegue y modelo dentro de la imagen

- **Fecha:** 2026-11-10
- **Estado:** aceptada

## Contexto

La API de inferencia debe publicarse en AWS con una URL HTTPS para el portafolio. Es una demo sin
tráfico sostenido: los recursos solo deben existir mientras se prueba y después se eliminan. El
modelo pesa 2,8 MB, la API no necesita ningún otro servicio de AWS para responder y las
dependencias de Python (LightGBM, scikit-learn, SHAP) ocupan unos 900 MB.

## Decisión

- **Plataforma: ECS Express Mode.** Con un solo comando crea la arquitectura estándar de
  contenedores: servicio ECS sobre Fargate, balanceador con HTTPS, grupos de seguridad, escalado,
  logs y despliegues canary con reversión automática. No hay servidores que administrar.
- **El modelo va dentro de la imagen.** S3 guarda la copia oficial por versión y el modelo se
  copia a la imagen al construirla. Cada imagen lleva su versión exacta del modelo, identificada
  con un `LABEL`, y la API no necesita permisos de AWS: no tiene rol de tarea.
- **Etiqueta de la imagen: SHA corto del commit.** Relaciona la imagen con el código.
- **Tamaño fijo:** 0,5 vCPU, 1 GB y una sola tarea. Basta para la demo y el costo es predecible.
- **Despliegue desde GitHub Actions con OIDC**, sin claves guardadas. La confianza del rol se
  limita a la rama `main` de este repositorio y usa los identificadores numéricos que GitHub
  incluye en el token, que no cambian aunque se reutilice el nombre del repositorio.

## Alternativas descartadas

- **App Runner:** era la opción más simple, pero dejó de aceptar clientes nuevos el 30-04-2026.
- **ECS clásico sobre Fargate:** es la misma arquitectura, pero exige definir a mano el balanceador,
  los target groups, los grupos de seguridad y el certificado. Para una demo no aporta más control
  útil.
- **Lambda:** cobra solo por uso, pero cada arranque en frío tendría que cargar las dependencias y
  el modelo, lo que penaliza la latencia de una API que se usa de forma esporádica.
- **SageMaker:** resuelve necesidades de muchos modelos (registro, variantes, monitoreo) que este
  proyecto no tiene, con más piezas y más costo.
- **EC2:** obliga a administrar el servidor y cobra la instancia completa.
- **Descargar el modelo de S3 al arrancar:** permitiría cambiar de modelo sin reconstruir la imagen,
  pero daría permisos de AWS a la API y una dependencia más en el arranque.

## Consecuencias

- Cambiar de modelo exige construir y desplegar una imagen nueva. Es aceptable, porque deja cada
  versión desplegada completamente trazable.
- Un despliegue tarda unos 10 minutos por la fase canary. Se acepta a cambio de la reversión
  automática.
- Con una sola tarea, una caída de su zona de disponibilidad interrumpe el servicio hasta que ECS
  lance otra. En producción se usarían al menos dos tareas en zonas distintas.
- La infraestructura se crea y se elimina con scripts, sin infraestructura como código; es el
  siguiente paso natural si el servicio tuviera que mantenerse.
