#!/usr/bin/env bash
# Descarga de S3 la versión más reciente del modelo a models/. La usa el workflow de despliegue,
# porque el .joblib no está en git. Uso: bash deploy/aws/fetch_model.sh

set -euo pipefail
source "$(dirname "$0")/config.sh"

# Las versiones son carpetas con fecha compacta (20261018T025933Z/): ordenadas, la última es la
# más reciente. tr quita la barra final y el "\r" de la CLI de Windows.
MODEL_VERSION=$(aws s3 ls "s3://$BUCKET/models/" | awk '{print $2}' | sort | tail -1 | tr -d '/\r')
if [ -z "$MODEL_VERSION" ]; then
    echo "No hay ningún modelo en s3://$BUCKET/models/ (ejecutar 01_bucket_and_model.sh)." >&2
    exit 1
fi

echo "Descargando el modelo $MODEL_VERSION"
aws s3 cp "s3://$BUCKET/models/$MODEL_VERSION/" models/ --recursive
