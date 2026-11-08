#!/usr/bin/env bash
# Paso 1: bucket S3 privado, cifrado y versionado con la copia oficial del modelo (D3).
# Uso: bash deploy/aws/01_bucket_and_model.sh
#
# El modelo queda en s3://<bucket>/models/<versión>/, donde la versión es el trained_at de la
# metadata en formato compacto (20261018T025933Z): los ':' y el '+' dan problemas en rutas y URLs.

set -euo pipefail
source "$(dirname "$0")/config.sh"

# Idempotente: head-bucket responde bien solo si el bucket existe y es nuestro.
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    echo "El bucket $BUCKET ya existe."
else
    # En us-east-1 no se indica --create-bucket-configuration (en otras regiones sí).
    aws s3api create-bucket --bucket "$BUCKET"
fi

# Versionado: si el modelo se sobrescribe o se borra por error, S3 guarda la versión anterior.
aws s3api put-bucket-versioning --bucket "$BUCKET" \
    --versioning-configuration Status=Enabled
# Etiqueta del proyecto, para que el teardown pueda comprobar que no quedó nada.
aws s3api put-bucket-tagging --bucket "$BUCKET" \
    --tagging "TagSet=[{Key=$TAG_KEY,Value=$TAG_VALUE}]"

# AWS crea los buckets privados y cifrados por defecto desde 2023; aquí se comprueba, no se supone.
aws s3api get-public-access-block --bucket "$BUCKET" --query PublicAccessBlockConfiguration
aws s3api get-bucket-encryption --bucket "$BUCKET" \
    --query "ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm"

MODEL_VERSION=$(uv run python -c "import json, datetime; \
t = json.load(open('models/champion_metadata.json'))['trained_at']; \
print(datetime.datetime.fromisoformat(t).strftime('%Y%m%dT%H%M%SZ'))")

aws s3 cp models/champion.joblib "s3://$BUCKET/models/$MODEL_VERSION/"
aws s3 cp models/champion_metadata.json "s3://$BUCKET/models/$MODEL_VERSION/"

# Verificación: deben aparecer los dos archivos bajo models/<versión>/.
aws s3 ls "s3://$BUCKET" --recursive
