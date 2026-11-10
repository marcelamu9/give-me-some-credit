# Spec 004 — Despliegue en AWS (Docker → ECR → ECS Express Mode → GitHub Actions)

- **Fecha:** 2026-10-26
- **Estado:** aprobada (2026-10-26)
- **Sub-proyecto:** 4 (Cimientos → Datos/EDA → Modelado + Decisión → API → **Despliegue**)
- **Depende de:** spec 003 (API: `credit_risk.api.main:app`, `/health`, `/predict`, `/model-info`, ejemplos en `credit_risk.api.examples`), `models/champion.joblib` + `models/champion_metadata.json`.

## 1. Intención

Publicar la API en una URL pública de AWS para el portafolio y, sobre todo, que la usuaria **aprenda el despliegue haciéndolo**: Docker, un registro de imágenes, cómputo gestionado, balanceador y CI/CD. Debe poder responder con seguridad "¿cómo desplegarías este modelo?" porque lo hizo de punta a punta.

Condición de costo de la usuaria: los recursos solo existen mientras se prueba; después se eliminan todos.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos**
- Fase 1: imagen Docker de la API con el modelo, probada en local.
- Fase 2: modelo en S3, imagen en ECR, servicio en ECS Express Mode (Fargate + ALB con HTTPS), alerta de presupuesto y limpieza completa.
- Fase 3: repositorio en GitHub y despliegue con GitHub Actions sin claves guardadas (OIDC).
- Un smoke test reutilizable contra cualquier URL (contenedor local o AWS).

**No-objetivos**
- Infraestructura como código (Terraform/CloudFormation/CDK): se menciona como siguiente paso.
- Dominio propio, autenticación, varios entornos (dev/prod), ajuste fino del autoescalado.
- SageMaker, Lambda, Kubernetes.
- Monitoreo de drift (sigue fuera de alcance).

**Anti-objetivos**
- Recursos de AWS que sigan cobrando después de la limpieza.
- Credenciales de AWS en el repositorio, en la imagen o en los secretos de GitHub (se usa OIDC).
- Datos de `data/` dentro de la imagen o del repositorio público.
- Crear recursos en AWS sin que la usuaria lo confirme en ese momento.

## 2.1 Modo de trabajo: sesión guiada

El objetivo es que la usuaria aprenda, así que **ejecuta ella** y Claude guía. No se delega la ejecución del plan a un agente autónomo.

| Parte | Quién | Cómo |
|---|---|---|
| `Dockerfile` | La usuaria lo escribe | Claude explica cada instrucción; ella la escribe, construye y depura |
| Comandos de AWS | La usuaria los ejecuta uno a uno | Claude explica qué hace cada comando y qué debe salir; después se guardan en `deploy/aws/` |
| Consola de AWS | La usuaria la recorre | Claude indica dónde mirar; ella describe lo que ve |
| Repo en GitHub y primer `push` | La usuaria | Revisión conjunta previa de lo que se publica |
| Workflow de GitHub Actions | Claude propone un borrador comentado | Ella lo explica con sus palabras y lo lanza |
| Smoke test y sus tests | Claude | Código de apoyo; Claude explica qué comprueba |

Las tres PARADAS de §4–§6 son puntos de comprobación: no se avanza hasta que la usuaria explica lo aprendido.

## 3. Decisiones

| # | Tema | Decisión | Motivo |
|---|---|---|---|
| D1 | Plataforma | **ECS Express Mode** | App Runner dejó de aceptar clientes nuevos el 30-04-2026; AWS recomienda Express Mode. Crea la arquitectura estándar (ECS + Fargate + ALB) con un solo comando |
| D2 | Región | `us-east-1` | Más barata y con todos los servicios |
| D3 | Modelo | S3 es la fuente oficial (`s3://<bucket>/models/<trained_at>/`); la imagen **incluye** el modelo al construirse | Cada imagen lleva su versión exacta del modelo; el servicio no necesita permisos de S3 |
| D4 | Etiqueta de imagen | SHA corto del commit de git; la versión del modelo va como `LABEL` | Trazabilidad imagen ↔ código ↔ modelo |
| D5 | Puerto y host | Uvicorn en `0.0.0.0:8000`; el `--primary-container` declara el puerto 8000 | `127.0.0.1` dejaría la API inaccesible desde fuera del contenedor |
| D6 | Health check | `/health` (503 sin modelo) | El ALB no envía tráfico a una tarea sin modelo |
| D7 | Tamaño de tarea | 0,5 vCPU / 1 GB, mínimo 1 y máximo 1 tarea | Basta para la demo; costo mínimo |
| D8 | Scripts de AWS | Bash (Git Bash en local, Ubuntu en Actions), en `deploy/aws/`, un script por paso, con cada comando comentado en español | Que la usuaria lea y entienda cada paso; reutilizables en CI |
| D9 | Nombres y etiquetas | Prefijo `credit-risk` en todo; etiqueta `Project=credit-risk` | Limpieza y control de costos fáciles |
| D10 | Despliegue en Actions | Workflow **manual** (`workflow_dispatch`) y separado del CI existente | Como la infraestructura se elimina tras probar, un despliegue automático en cada `push` fallaría; el CI de tests sigue en cada `push` |
| D11 | Repositorio | Público en GitHub (portafolio) | Revisado antes de publicar: sin datos, sin modelo, sin credenciales |

## 4. Fase 1 — Docker en local

- `Dockerfile` en la raíz:
  - Base `python:3.11-slim`; uv para instalar dependencias con `uv sync --locked --no-dev`.
  - Capas en orden de caché: `pyproject.toml` + `uv.lock` → dependencias → `src/` y `configs/` → modelo.
  - Copia solo `models/champion.joblib` y `models/champion_metadata.json` (no `candidates.joblib`).
  - Usuario sin privilegios (no root).
  - `EXPOSE 8000`; `CMD` con Uvicorn en `0.0.0.0:8000`.
  - `HEALTHCHECK` contra `/health`.
- `.dockerignore`: `data/`, `.venv/`, `mlruns/`, `notebooks/`, `reports/`, `tests/`, `docs/`, `.git/`, `models/candidates.joblib`.
- Tareas poe: `docker-build` (etiqueta con el SHA del commit) y `docker-run` (`-p 8000:8000`).
- `python -m credit_risk.deploy.smoke --url <URL>` (módulo en `src/`, solo librería estándar, con tests): comprueba `/health` → 200 con modelo cargado; `low_risk` → `approve`; `same_client_with_delays` → `review` o `reject`; `invalid_age` → 422; e informa la latencia media de `/predict`. Sale con código ≠ 0 si algo falla.
- **PARADA 1:** la usuaria construye y corre la imagen, abre `/docs` desde el contenedor y explica la diferencia entre imagen y contenedor, el papel del `0.0.0.0` y el orden de las capas.

## 5. Fase 2 — AWS

Scripts en `deploy/aws/`, ejecutados por la usuaria (o por Claude **solo con su confirmación explícita en ese momento**):

| Script | Qué hace |
|---|---|
| `00_budget.sh` | Alerta de presupuesto mensual de **5 USD** por correo (AWS Budgets) |
| `01_bucket_and_model.sh` | Crea el bucket S3 (privado, cifrado, versionado) y sube el modelo a `models/<trained_at>/` |
| `02_ecr_push.sh` | Crea el repositorio ECR, hace login de Docker, construye y sube la imagen etiquetada con el SHA |
| `03_roles.sh` | Crea el rol de ejecución de tareas y el rol de infraestructura de Express Mode (políticas gestionadas de AWS) |
| `04_service.sh` | `aws ecs create-express-gateway-service` con la imagen de ECR, los dos roles, puerto 8000, `--health-check-path /health`, CPU/memoria de D7 y escalado 1–1; espera a que esté estable e imprime la URL |
| `05_smoke.sh` | Ejecuta el smoke test (`credit_risk.deploy.smoke`) contra la URL pública |
| `99_teardown.sh` | Elimina el servicio de Express Mode (y con él el ALB y los recursos que creó), el repositorio ECR, el bucket S3 y los roles; al final **lista** los recursos con la etiqueta `Project=credit-risk` y avisa si queda alguno |

`deploy/aws/config.sh` centraliza región, prefijo y nombres. Ningún script contiene identificadores de cuenta ni claves: se obtienen de la sesión de la CLI (`aws sts get-caller-identity`).

- **Comprobaciones previas:** en *Billing → Credits*, confirmar que los créditos cubren ECS/Fargate, ELB, ECR, S3 y CloudWatch; y que la cuenta tiene una **VPC por defecto** en `us-east-1` (Express Mode la usa si no se indica otra red).
- **PARADA 2:** con la URL pública funcionando, la usuaria recorre en la consola de AWS el servicio ECS, la tarea Fargate, el ALB, el target group con su health check, la imagen en ECR, el modelo en S3 y los logs en CloudWatch, y explica qué hace cada pieza. Después se ejecuta `99_teardown.sh` el mismo día.

## 6. Fase 3 — GitHub y CI/CD

- Revisión antes de publicar: `git ls-files` sin `data/`, sin `.joblib` y sin credenciales; los notebooks solo muestran agregados. La usuaria crea el repositorio público y hace el primer `push`.
- `deploy/aws/06_github_oidc.sh`: crea el proveedor OIDC de GitHub en IAM y un rol que solo puede asumir **este** repositorio en la rama `main`, con permisos mínimos: leer el modelo de S3, subir a ECR y actualizar el servicio ECS. Lo elimina `99_teardown.sh`.
- `.github/workflows/deploy.yml` (`workflow_dispatch`): asumir el rol por OIDC → tests (`uv run poe check`) → descargar el modelo de S3 → build → push a ECR con el SHA → actualizar el servicio de Express Mode con la nueva imagen → esperar estabilidad → smoke test contra la URL.
- El `ci.yml` actual no cambia (lint + tests en cada `push`).
- **PARADA 3:** la usuaria lanza el workflow, sigue los pasos en la pestaña *Actions*, comprueba la nueva versión en la URL y explica qué es OIDC y por qué evita guardar claves. Después, teardown.

## 7. Documentación

- `README.md`, sección **Despliegue**: diagrama de la arquitectura, requisitos, pasos de las 3 fases, costo esperado, cómo limpiar y una subsección **"¿Cómo se desplegaría en producción?"** (IaC, varios entornos, autoescalado, monitoreo de drift, model registry, SageMaker como alternativa).
- `docs/decisions/ADR-003-plataforma-despliegue.md`: por qué ECS Express Mode (App Runner sin altas nuevas; alternativas Lambda, ECS clásico, SageMaker, EC2) y por qué el modelo va dentro de la imagen.

## 8. Criterios de aceptación

- [x] `uv run poe check` pasa (incluye tests del smoke test con una API falsa).
- [x] `docker build` produce una imagen sin `data/` ni `candidates.joblib` (verificado listando su contenido) y corre como usuario no root.
- [x] El smoke test pasa contra el contenedor local.
- [x] El smoke test pasa contra la URL pública de AWS.
- [x] Tras `99_teardown.sh`, no queda ningún recurso con la etiqueta `Project=credit-risk` ni servicio de Express Mode, repositorio ECR o bucket del proyecto (ver enmienda 10.1).
- [x] La alerta de presupuesto existe antes de crear el resto de recursos.
- [x] El workflow `deploy.yml` despliega con OIDC, sin secretos de AWS en GitHub.
- [x] README y ADR-003 documentan arquitectura, costos, pasos y limpieza; `AGENTS.md` apunta a esta spec y a su plan.

## 9. Lo que queda al criterio de quien implemente

Detalles del `Dockerfile` (multi-stage o no) siempre que cumpla §4; nombres exactos de recursos dentro del prefijo; redacción de los comentarios de los scripts; formato del informe del smoke test. Los comandos de AWS se verifican contra la documentación oficial vigente, no de memoria.

## 10. Enmiendas

1. **Verificación del teardown:** el índice de etiquetas de AWS lista durante horas recursos ya borrados, así que el criterio se comprueba preguntando a cada servicio; el índice se muestra solo como información.
2. **Merge antes de probar el workflow:** GitHub solo ofrece `workflow_dispatch` para archivos que están en la rama principal, y el rol solo confía en `main`.
3. **Confianza del rol de GitHub:** el `sub` del token incluye los identificadores numéricos del propietario y del repositorio, y la regla los usa.
4. **Permisos del rol de GitHub:** actualizar el servicio también requiere registrar definiciones de tarea y pasar los roles de ECS; se añadieron además consultas del estado del despliegue.
5. **`AWS_ROLE_ARN`** es una variable del repositorio, no un secreto, para no escribir el número de cuenta en el código público.
