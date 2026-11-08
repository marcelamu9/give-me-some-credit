#!/usr/bin/env bash
# Paso 2: repositorio ECR y subida de la imagen etiquetada con el SHA corto del commit (D4).
# Uso: bash deploy/aws/02_ecr_push.sh
#
# Requiere Docker encendido y models/champion.joblib en local (la imagen lo lleva dentro).

set -euo pipefail
source "$(dirname "$0")/config.sh"

REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
IMAGE_TAG=$(git rev-parse --short HEAD)

# Idempotente: crear el repositorio solo si no existe. scanOnPush revisa cada imagen subida en
# busca de vulnerabilidades conocidas (CVE).
if aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1; then
    echo "El repositorio $ECR_REPO ya existe."
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO" \
        --image-scanning-configuration scanOnPush=true \
        --tags "Key=$TAG_KEY,Value=$TAG_VALUE"
fi

# Contraseña temporal (12 h) que se pasa a Docker por la tubería, sin mostrarla en pantalla.
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

# Construye credit-risk-api:<sha> (y :latest) con la versión del modelo como LABEL.
uv run poe docker-build

# docker tag no copia nada: añade el nombre completo registro/repositorio:etiqueta, que le dice
# a Docker adónde subir la imagen.
docker tag "credit-risk-api:$IMAGE_TAG" "$REGISTRY/$ECR_REPO:$IMAGE_TAG"
docker push "$REGISTRY/$ECR_REPO:$IMAGE_TAG"

# Verificación. Docker Desktop sube un índice (lleva la etiqueta), la imagen real y una
# atestación de cómo se construyó; las dos últimas aparecen sin etiqueta y es normal.
aws ecr describe-images --repository-name "$ECR_REPO" \
    --query "imageDetails[].{tags:imageTags,bytes:imageSizeInBytes,type:artifactMediaType}"

# El escaneo se hace sobre la imagen real, no sobre el índice.
IMAGE_DIGEST=$(aws ecr describe-images --repository-name "$ECR_REPO" \
    --query "imageDetails[?artifactMediaType=='application/vnd.oci.image.config.v1+json'] \
             | sort_by(@, &imagePushedAt)[-1].imageDigest" --output text)
echo "Escaneo de $IMAGE_DIGEST (puede tardar 1-2 minutos en completarse):"
aws ecr describe-image-scan-findings --repository-name "$ECR_REPO" \
    --image-id "imageDigest=$IMAGE_DIGEST" \
    --query "{estado:imageScanStatus.status,hallazgos:imageScanFindings.findingSeverityCounts}" ||
    echo "El escaneo aún no está listo; repite este último comando en un minuto."
