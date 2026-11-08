#!/usr/bin/env bash
# Paso 4: servicio ECS Express Mode (Fargate + balanceador con HTTPS) con la imagen de ECR.
# Uso: bash deploy/aws/04_service.sh [etiqueta]   (por defecto, el SHA corto de HEAD)
#
# DESDE AQUÍ SE COBRA POR HORA (~0,06 USD/h): crear, probar y ejecutar 99_teardown.sh el mismo día.
# Si el servicio ya existe, despliega la nueva imagen con un update (despliegue canary).

set -euo pipefail
source "$(dirname "$0")/config.sh"

IMAGE_TAG="${1:-$(git rev-parse --short HEAD)}"
IMAGE="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
SERVICE_ARN="arn:aws:ecs:$AWS_DEFAULT_REGION:$ACCOUNT_ID:service/default/$SERVICE"

# La imagen tiene que estar en ECR antes de desplegarla (02_ecr_push.sh).
aws ecr describe-images --repository-name "$ECR_REPO" --image-ids "imageTag=$IMAGE_TAG" \
    >/dev/null

# describe-services devuelve ACTIVE solo si el servicio existe y no se está borrando.
STATUS=$(aws ecs describe-services --cluster default --services "$SERVICE" \
    --query "services[0].status" --output text 2>/dev/null || echo "NONE")

if [ "$STATUS" = "ACTIVE" ]; then
    echo "El servicio $SERVICE ya existe: se despliega la imagen $IMAGE_TAG."
    aws ecs update-express-gateway-service \
        --service-arn "$SERVICE_ARN" \
        --primary-container "{\"image\": \"$IMAGE\", \"containerPort\": 8000}" \
        --monitor-resources DEPLOYMENT --monitor-mode TEXT-ONLY
else
    # --primary-container: imagen y puerto del CMD del Dockerfile (por defecto sería 80).
    # --health-check-path: el balanceador no envía tráfico mientras /health no responda 200.
    # --cpu/--memory en unidades: 512 = 0,5 vCPU; 1024 MiB = 1 GB (D7). Escalado fijo 1-1.
    # Las etiquetas de ECS van en minúscula (key/value), a diferencia de IAM, ECR y S3.
    aws ecs create-express-gateway-service \
        --service-name "$SERVICE" \
        --execution-role-arn "arn:aws:iam::$ACCOUNT_ID:role/$EXECUTION_ROLE" \
        --infrastructure-role-arn "arn:aws:iam::$ACCOUNT_ID:role/$INFRA_ROLE" \
        --primary-container "{\"image\": \"$IMAGE\", \"containerPort\": 8000}" \
        --health-check-path /health \
        --cpu 512 --memory 1024 \
        --scaling-target minTaskCount=1,maxTaskCount=1 \
        --tags "key=$TAG_KEY,value=$TAG_VALUE" \
        --monitor-resources DEPLOYMENT --monitor-mode TEXT-ONLY
fi

# URL pública (HTTPS). Tras crear el servicio puede pasar 1-3 minutos con 503 del balanceador
# mientras termina de enviar tráfico a la tarea; el 503 no viene de la API.
aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" \
    --query "service.activeConfigurations[-1].ingressPaths[0].endpoint" --output text
