# Despliegue en AWS — Plan (sesión guiada)

> **Modo de ejecución:** sesión guiada (spec §2.1). La usuaria ejecuta los comandos y escribe el `Dockerfile`; Claude explica antes de cada paso, revisa la salida y escribe el código de apoyo (smoke test, borradores de scripts y workflow). No se delega a un agente autónomo. Ningún recurso de AWS se crea sin la confirmación de la usuaria en ese momento. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** API pública en AWS (ECS Express Mode) a partir de una imagen Docker con el modelo, desplegable con GitHub Actions por OIDC, y que se pueda crear, probar y eliminar el mismo día.

**Architecture:** Imagen Docker (API + modelo) → ECR → servicio ECS Express Mode (Fargate + ALB con HTTPS) en `us-east-1`; el modelo vive en S3 y se copia a la imagen al construirla. Scripts bash numerados en `deploy/aws/` documentan y repiten cada paso, incluida la limpieza. Un smoke test en `credit_risk.deploy.smoke` (solo librería estándar) verifica cualquier URL.

**Tech Stack:** Docker, uv, AWS CLI v2 (S3, ECR, IAM, ECS Express Mode, Budgets, CloudWatch), Bash (Git Bash / Ubuntu), GitHub Actions + OIDC, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-despliegue-design.md`

## Global Constraints

- Región `us-east-1`; prefijo `credit-risk`; etiqueta `Project=credit-risk` en todo recurso que la admita.
- Tarea Fargate: 0,5 vCPU / 1 GB, escalado mínimo 1 y máximo 1; puerto del contenedor 8000; health check `/health`.
- Imagen: `python:3.11-slim`, dependencias con `uv sync --locked --no-dev`, usuario no root, Uvicorn en `0.0.0.0:8000`, sin `data/` ni `models/candidates.joblib`; etiqueta = SHA corto del commit; `LABEL` con la versión del modelo (`trained_at`).
- Sin identificadores de cuenta ni claves en el repositorio: se obtienen con `aws sts get-caller-identity`. GitHub usa OIDC, nunca secretos de AWS.
- Alerta de presupuesto de 5 USD creada **antes** que cualquier otro recurso. Teardown el mismo día de cada prueba.
- Los comandos de AWS se comprueban contra la documentación oficial vigente antes de ejecutarlos.
- Smoke test sin dependencias externas (solo `urllib`), para que corra en cualquier sitio.
- Documentación en español; código, scripts e identificadores en inglés. Un commit por tarea; rama `feat/despliegue`. `uv run poe check` en verde antes de cada commit.

## Review Focus

1. Teardown con recursos a medio crear (o ejecutado dos veces) → no falla; informa lo que no existía y sigue. (Verificación en Task 3.)
2. Construir la imagen sin `models/champion.joblib` → el build falla con un mensaje claro, no produce una imagen sin modelo. (Verificación en Task 2.)
3. Scripts bash con finales de línea CRLF en Windows → fallan en Linux; `.gitattributes` ya fuerza `eol=lf`, comprobar que los `.sh` lo respetan. (Verificación en Task 3.)
4. URL con `/` final o con `https://` → el smoke test la normaliza. (Test en Task 1.)
5. Servicio desplegado pero sin modelo (503 en `/health`) → el smoke test falla con un mensaje explícito, no con una traza. (Test en Task 1.)

---

### Task 0: Preparación

- [ ] Rama: `git checkout -b feat/despliegue`.
- [ ] Commits de la spec y del plan (`docs: add deployment spec`, `docs: add deployment plan`); `AGENTS.md` apunta a ellos como activos.
- [ ] Docker Desktop abierto: `docker info` responde con `Server Version`.
- [ ] `aws sts get-caller-identity` muestra la cuenta personal de la usuaria.

---

### Task 1: Smoke test (Claude, TDD)

**Files:** Create `src/credit_risk/deploy/__init__.py`, `src/credit_risk/deploy/smoke.py`, `tests/test_deploy_smoke.py`.

**Interfaces — Produces:**
- `Fetch = Callable[[str, str, dict | None], tuple[int, dict]]` — `(method, url, json_body) -> (status, json_response)`.
- `@dataclass CheckResult(name: str, passed: bool, detail: str)`.
- `normalize_url(url: str) -> str` — quita la `/` final; añade `https://` si no hay esquema (salvo `http://localhost` o `http://127.0.0.1`, que se respetan).
- `run_smoke(base_url: str, fetch: Fetch = urllib_fetch, repeats: int = 5) -> list[CheckResult]` — checks: `health` (200 y `model_loaded`), `low_risk_approved`, `delays_raise_decision` (`review` o `reject` y probabilidad mayor que la de `low_risk`), `invalid_age_rejected` (422) y `latency` (media de `repeats` llamadas a `/predict`, siempre `passed=True`, la cifra en `detail`). Usa los ejemplos de `credit_risk.api.examples`.
- `main(argv: list[str] | None = None) -> int` — `--url`; imprime una línea por check y devuelve 0 solo si todos pasan. Ejecutable con `python -m credit_risk.deploy.smoke --url ...`.

- [ ] **Tests que fallan** (con un `fake_fetch` que devuelve respuestas preparadas): todo correcto → todos `passed` y `main` devuelve 0; `/health` 503 → check `health` falla con `detail` que menciona "503"; `low_risk` con `reject` → falla `low_risk_approved`; `invalid_age` con 200 → falla `invalid_age_rejected`; `fetch` que lanza `URLError` → check fallido con mensaje, sin traza; `normalize_url` con `"x.on.aws/"`, `"https://x.on.aws"` y `"http://localhost:8000/"`.
- [ ] Ejecutar → fallan (`ModuleNotFoundError`). Implementar. `uv run poe check` → verde.
- [ ] Prueba real contra la API local: `uv run poe serve` y en otra terminal `uv run python -m credit_risk.deploy.smoke --url http://127.0.0.1:8000` → todos los checks pasan.
- [ ] Commit: `feat: add deployment smoke test`.

---

### Task 2: Fase 1 — Docker en local (la usuaria escribe el `Dockerfile`)

**Files:** Create `Dockerfile`, `.dockerignore`; Modify `pyproject.toml` (tareas poe).

- [ ] **Guía del `Dockerfile`** (Claude explica cada instrucción antes de que la usuaria la escriba): `FROM python:3.11-slim` → instalar uv (copiando el binario desde `ghcr.io/astral-sh/uv`) → `WORKDIR /app` → copiar `pyproject.toml` + `uv.lock` y `uv sync --locked --no-dev --no-install-project` → copiar `src/` y `configs/` y `uv sync --locked --no-dev` → copiar `models/champion.joblib` y `models/champion_metadata.json` a `models/` → `ARG MODEL_VERSION` + `LABEL model_version=$MODEL_VERSION` → crear usuario sin privilegios y `USER` → `EXPOSE 8000` → `HEALTHCHECK` que llama a `/health` con Python estándar (la imagen no trae `curl`) → `CMD` con Uvicorn en `0.0.0.0:8000` usando el entorno ya instalado (sin reinstalar al arrancar).
- [ ] **`.dockerignore`** con lo de la spec §4 (`data/`, `.venv/`, `mlruns/`, `notebooks/`, `reports/`, `tests/`, `docs/`, `.git/`, `models/candidates.joblib`).
- [ ] **Tareas poe:** `docker-build` (construye `credit-risk-api:<sha corto>` pasando `MODEL_VERSION` desde la metadata) y `docker-run` (`docker run --rm -p 8000:8000 credit-risk-api:<sha>`). Si poe no puede calcular el SHA en una sola línea portable, un script pequeño en `deploy/docker_build.sh` y la tarea lo invoca.
- [ ] **Verificaciones:**
  - `uv run poe docker-build` → termina; `docker images credit-risk-api` muestra la imagen y su tamaño (anotarlo).
  - Sin datos ni candidatos: `docker run --rm --entrypoint sh credit-risk-api:<sha> -c "ls /app /app/models; test ! -e /app/data"` → no hay `data/`; en `models/` solo los dos archivos.
  - No root: `docker run --rm --entrypoint whoami credit-risk-api:<sha>` → no es `root`.
  - Etiqueta: `docker inspect --format "{{ index .Config.Labels \"model_version\" }}" credit-risk-api:<sha>` → el `trained_at` del modelo.
  - Sin modelo (Review Focus 2): renombrar temporalmente `models/champion.joblib`, construir → falla en el `COPY` con el nombre del archivo; restaurar el nombre.
  - `uv run poe docker-run` y, en otra terminal, `uv run python -m credit_risk.deploy.smoke --url http://127.0.0.1:8000` → todos los checks pasan; `/docs` abre en el navegador.
- [ ] **PARADA 1:** la usuaria explica imagen frente a contenedor, por qué `0.0.0.0`, por qué el orden de capas acelera los builds y qué pasa si falta el modelo.
- [ ] Commit: `build: add Dockerfile for the inference API`.

---

### Task 3: Fase 2 — AWS (la usuaria ejecuta cada comando)

**Files:** Create `deploy/aws/config.sh`, `00_budget.sh`, `01_bucket_and_model.sh`, `02_ecr_push.sh`, `03_roles.sh`, `04_service.sh`, `05_smoke.sh`, `99_teardown.sh`.

Patrón de cada paso: Claude explica el comando y la salida esperada → la usuaria lo ejecuta y comparte la salida → si es correcta, se guarda en su script (comentado en español, `set -euo pipefail`, lee `config.sh`).

- [ ] **Comprobaciones previas:** *Billing → Credits* cubre ECS/Fargate, ELB, ECR, S3 y CloudWatch; `aws ec2 describe-vpcs --filters Name=isDefault,Values=true --region us-east-1` devuelve una VPC.
- [ ] **`config.sh`:** región, prefijo, nombres (bucket `credit-risk-models-<account_id>`, repo ECR `credit-risk-api`, servicio `credit-risk-api`, roles `credit-risk-ecs-execution` y `credit-risk-ecs-infrastructure`), `ACCOUNT_ID` desde `aws sts get-caller-identity`, `IMAGE_TAG` desde `git rev-parse --short HEAD`, `MODEL_VERSION` desde la metadata.
- [ ] **`00_budget.sh`:** presupuesto mensual de 5 USD con aviso por correo al 80 % (el correo se pasa como argumento, no se guarda). Verificar: `aws budgets describe-budgets` lo lista.
- [ ] **`01_bucket_and_model.sh`:** bucket privado (bloqueo de acceso público), cifrado y versionado, con etiqueta; subir los dos archivos a `models/<MODEL_VERSION>/`. Verificar: `aws s3 ls s3://<bucket>/models/ --recursive`.
- [ ] **`02_ecr_push.sh`:** crear el repositorio (escaneo al subir activado) si no existe, login de Docker, build con `deploy/docker_build.sh` o `poe docker-build`, etiquetar y subir. Verificar: `aws ecr describe-images --repository-name credit-risk-api`.
- [ ] **`03_roles.sh`:** rol de ejecución (confianza `ecs-tasks.amazonaws.com`, política gestionada `AmazonECSTaskExecutionRolePolicy`) y rol de infraestructura de Express Mode (confianza y política gestionada según la guía oficial de Express Mode). Esperar ~1 minuto por la propagación de IAM.
- [ ] **`04_service.sh`:** `aws ecs create-express-gateway-service` con `--service-name credit-risk-api`, la imagen de ECR en `--primary-container` (puerto 8000), los ARN de los dos roles, `--cpu`/`--memory` de las restricciones globales, `--health-check-path /health`, escalado 1–1 y etiquetas; esperar a que esté estable e imprimir la URL. Verificar: la URL responde en `/health`.
- [ ] **`05_smoke.sh`:** `uv run python -m credit_risk.deploy.smoke --url <URL>` → todos los checks pasan. Anotar la latencia.
- [ ] **PARADA 2:** la usuaria recorre en la consola el servicio ECS (Express Mode), la tarea Fargate, el ALB y su target group con el health check, la imagen en ECR, el modelo en S3 y los logs en CloudWatch, y explica qué hace cada pieza y qué costaría mantenerlo encendido.
- [ ] **`99_teardown.sh`** (ejecutar el mismo día): eliminar el servicio de Express Mode y esperar a que desaparezca, el repositorio ECR (con sus imágenes), el bucket (todas las versiones de los objetos) y los roles (con sus políticas); cada borrado tolera que el recurso no exista. Al final: `aws resourcegroupstaggingapi get-resources --tag-filters Key=Project,Values=credit-risk` → vacío, o avisa de lo que quede. Ejecutarlo **dos veces**: la segunda no debe fallar (Review Focus 1). El presupuesto se conserva.
- [ ] `git ls-files --eol deploy/aws/` → todos `lf` (Review Focus 3). `grep` de 12 dígitos seguidos en `deploy/` → sin identificadores de cuenta.
- [ ] Commit: `feat: add AWS deployment and teardown scripts`.

---

### Task 4: Fase 3 — GitHub y CI/CD

**Files:** Create `deploy/aws/06_github_oidc.sh`, `.github/workflows/deploy.yml`; Modify `deploy/aws/99_teardown.sh`.

- [ ] **Revisión antes de publicar (juntas):** `git ls-files | grep -E "^data/|\.joblib$|\.env"` → vacío; buscar claves (`AKIA`, `aws_secret`) en el historial → nada; los notebooks solo muestran agregados; `docs/playbook.md` sigue fuera del repo.
- [ ] La usuaria crea el repositorio público en GitHub, añade el remoto y hace `git push -u origin main` (después del merge de esta rama, o empuja también `feat/despliegue`). El workflow `CI` existente pasa en verde.
- [ ] **`06_github_oidc.sh`:** proveedor OIDC `token.actions.githubusercontent.com` (si no existe) y rol `credit-risk-github-deploy` cuya confianza solo admite `repo:<usuario>/<repo>:ref:refs/heads/main`, con permisos mínimos: `s3:GetObject` en el bucket, push a `credit-risk-api` en ECR y actualizar/describir el servicio de Express Mode (y `iam:PassRole` sobre los dos roles del servicio, si la actualización lo exige). `99_teardown.sh` borra el rol y el proveedor.
- [ ] **`deploy.yml`** (`workflow_dispatch`; borrador comentado de Claude, la usuaria lo explica): `permissions: id-token: write, contents: read` → `aws-actions/configure-aws-credentials` con el rol → `uv sync` + `uv run poe check` → descargar el modelo de S3 a `models/` → build y push a ECR con el SHA → actualizar el servicio de Express Mode con la nueva imagen y esperar estabilidad → smoke test contra la URL. Sin secretos de AWS: solo variables del repositorio (ARN del rol y región).
- [ ] **Prueba:** recrear la infraestructura con `00`–`04` (y `06`), lanzar el workflow desde la pestaña *Actions* y comprobar en `/model-info` y en ECR que la imagen desplegada es la del último commit.
- [ ] **PARADA 3:** la usuaria explica el flujo del workflow, qué es OIDC y por qué evita guardar claves, y qué haría falta para desplegar en cada `push`.
- [ ] Teardown el mismo día (`99_teardown.sh`), con la verificación de etiquetas vacía.
- [ ] Commit: `ci: add manual AWS deployment workflow with OIDC`.

---

### Task 5: Documentación y cierre

**Files:** Modify `README.md`, `docs/ai_usage_log.md`, spec §8; Create `docs/decisions/ADR-003-plataforma-despliegue.md`.

- [ ] **README, sección "Despliegue":** diagrama de la arquitectura; requisitos; pasos de las 3 fases (comandos de `poe` y de `deploy/aws/`); costo observado (de la consola de facturación) y por qué el teardown; tamaño de imagen y latencia en AWS medidos; subsección "¿Cómo se desplegaría en producción?" (IaC, varios entornos, despliegue automático, autoescalado, monitoreo de drift, model registry, SageMaker como alternativa).
- [ ] **ADR-003:** contexto (App Runner sin altas nuevas desde el 30-04-2026), decisión (ECS Express Mode), alternativas (Lambda, ECS clásico con ALB propio, SageMaker, EC2) y consecuencias; y por qué el modelo va dentro de la imagen.
- [ ] Fila en `docs/ai_usage_log.md` (sesión guiada: qué escribió la usuaria, qué propuso Claude, qué falló y cómo se resolvió); marcar los criterios de la spec §8.
- [ ] `uv run poe check` → verde. Commit: `docs: document AWS deployment`. Merge de `feat/despliegue` a `main` y `push`.
