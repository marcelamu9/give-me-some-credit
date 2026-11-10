#!/usr/bin/env bash
# Paso 5: smoke test contra la URL pública del servicio.
# Uso: bash deploy/aws/05_smoke.sh
#
# Si todo responde 503 (incluso la edad inválida, que la API siempre rechaza con 422), el
# balanceador aún no envía tráfico a la tarea: esperar 1-2 minutos y repetir.

set -euo pipefail
source "$(dirname "$0")/config.sh"

SERVICE_ARN="arn:aws:ecs:$AWS_DEFAULT_REGION:$ACCOUNT_ID:service/default/$SERVICE"
URL=$(aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" \
    --query "service.activeConfigurations[-1].ingressPaths[0].endpoint" --output text)

uv run python -m credit_risk.deploy.smoke --url "$URL"
