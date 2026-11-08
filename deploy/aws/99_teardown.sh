#!/usr/bin/env bash
# Limpieza: elimina TODO lo que crearon los pasos 1-4 y comprueba que no quede nada.
# Uso: bash deploy/aws/99_teardown.sh
#
# Idempotente: se puede ejecutar dos veces; lo que ya no existe se salta.
# El presupuesto (paso 0) se conserva a propósito: es gratis y sigue vigilando la cuenta.

set -uo pipefail # sin -e: si un paso falla, se sigue limpiando el resto
source "$(dirname "$0")/config.sh"

SERVICE_ARN="arn:aws:ecs:$AWS_DEFAULT_REGION:$ACCOUNT_ID:service/default/$SERVICE"

echo "== 1. Servicio Express Mode (tareas, balanceador, target groups, grupos de seguridad)"
STATUS=$(aws ecs describe-services --cluster default --services "$SERVICE" \
    --query "services[0].status" --output text 2>/dev/null || echo "NONE")
if [ "$STATUS" = "ACTIVE" ]; then
    # Sin --monitor-resources: en el borrado, el seguimiento puede quedarse mirando aunque el
    # servicio ya esté eliminado. Se espera con el waiter, que termina al quedar INACTIVE.
    aws ecs delete-express-gateway-service --service-arn "$SERVICE_ARN" >/dev/null
    echo "Borrando el servicio (5-10 minutos, sobre todo por el balanceador)..."
    aws ecs wait services-inactive --cluster default --services "$SERVICE"
    echo "Servicio eliminado."
else
    echo "No hay servicio activo."
fi

echo "== 2. Clúster 'default' (solo si lo creó este proyecto y quedó vacío)"
CLUSTER_TAG=$(aws ecs describe-clusters --clusters default --include TAGS \
    --query "clusters[?status=='ACTIVE'] | [0].tags[?key=='$TAG_KEY'] | [0].value" \
    --output text 2>/dev/null)
if [ "$CLUSTER_TAG" = "$TAG_VALUE" ]; then
    aws ecs delete-cluster --cluster default >/dev/null && echo "Clúster eliminado."
else
    echo "El clúster no existe o no es de este proyecto; no se toca."
fi

echo "== 2b. Definiciones de tarea (la 'ficha' de cada versión del contenedor)"
# Se dan de baja (INACTIVE) y después se eliminan; si no, quedan registradas con la etiqueta.
for td in $(aws ecs list-task-definitions --family-prefix "default-$SERVICE" --status ACTIVE \
    --query "taskDefinitionArns[]" --output text); do
    aws ecs deregister-task-definition --task-definition "$td" >/dev/null
done
INACTIVE=$(aws ecs list-task-definitions --family-prefix "default-$SERVICE" --status INACTIVE \
    --query "taskDefinitionArns[]" --output text)
if [ -n "$INACTIVE" ]; then
    # shellcheck disable=SC2086 # se pasan como varios argumentos a propósito
    aws ecs delete-task-definitions --task-definitions $INACTIVE >/dev/null &&
        echo "Definiciones de tarea eliminadas."
fi

echo "== 3. Logs de CloudWatch"
for group in $(aws logs describe-log-groups --log-group-name-prefix "/aws/ecs/default/$SERVICE" \
    --query "logGroups[].logGroupName" --output text); do
    aws logs delete-log-group --log-group-name "$group" && echo "Eliminado $group"
done

echo "== 4. Repositorio ECR (con todas sus imágenes)"
if aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1; then
    aws ecr delete-repository --repository-name "$ECR_REPO" --force >/dev/null &&
        echo "Repositorio eliminado."
else
    echo "No existe."
fi

echo "== 5. Bucket S3 (con versionado: hay que borrar cada versión antes que el bucket)"
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    # Cada línea: clave y versión de un objeto o de una marca de borrado. tr quita el "\r" que
    # añade la CLI de Windows al final de cada línea; sin eso la versión llega mal a AWS.
    aws s3api list-object-versions --bucket "$BUCKET" \
        --query "[Versions, DeleteMarkers][][].[Key, VersionId]" --output text | tr -d '\r' |
        while read -r key version; do
            [ -n "$key" ] && [ "$key" != "None" ] || continue
            aws s3api delete-object --bucket "$BUCKET" --key "$key" --version-id "$version" \
                >/dev/null
        done
    aws s3api delete-bucket --bucket "$BUCKET" && echo "Bucket eliminado."
else
    echo "No existe."
fi

echo "== 6. Roles de IAM (primero se quitan las políticas, después el rol)"
for role in "$EXECUTION_ROLE" "$INFRA_ROLE"; do
    if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
        for policy in $(aws iam list-attached-role-policies --role-name "$role" \
            --query "AttachedPolicies[].PolicyArn" --output text); do
            aws iam detach-role-policy --role-name "$role" --policy-arn "$policy"
        done
        aws iam delete-role --role-name "$role" && echo "Eliminado $role"
    else
        echo "$role no existe."
    fi
done

echo "== Verificación directa (lo que manda: se pregunta a cada servicio)"
FAILED=0
check() { # $1 descripción, $2 lo que devolvió la consulta (vacío = no queda nada)
    local found="${2//$'\r'/}" # sin el "\r" de la CLI de Windows
    if [ -n "$found" ] && [ "$found" != "None" ]; then
        echo "  QUEDA  $1: $found"
        FAILED=1
    else
        echo "  OK     $1"
    fi
}
check "servicio ECS activo" "$(aws ecs describe-services --cluster default --services "$SERVICE" \
    --query "services[?status=='ACTIVE'].serviceName" --output text 2>/dev/null)"
check "tareas corriendo" "$(aws ecs list-tasks --cluster default \
    --query "taskArns[]" --output text 2>/dev/null)"
check "balanceadores de Express Mode" "$(aws elbv2 describe-load-balancers \
    --query "LoadBalancers[?starts_with(LoadBalancerName, 'ecs-express-gateway')].LoadBalancerName" \
    --output text)"
check "grupos de seguridad del proyecto" "$(aws ec2 describe-security-groups \
    --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" --query "SecurityGroups[].GroupId" \
    --output text)"
check "repositorio ECR" "$(aws ecr describe-repositories --repository-names "$ECR_REPO" \
    --query "repositories[].repositoryName" --output text 2>/dev/null)"
check "bucket S3" "$(aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 && echo "$BUCKET")"
check "logs de CloudWatch" "$(aws logs describe-log-groups \
    --log-group-name-prefix "/aws/ecs/default/$SERVICE" --query "logGroups[].logGroupName" \
    --output text)"
check "roles de IAM" "$(aws iam list-roles \
    --query "Roles[?starts_with(RoleName, '$PROJECT-')].RoleName" --output text)"

echo "== Índice de etiquetas $TAG_KEY=$TAG_VALUE (informativo)"
# Este índice se actualiza con retraso: puede seguir listando recursos ya borrados (reglas de
# grupos de seguridad, tareas o servicios INACTIVE) durante horas. La verificación directa de
# arriba es la que cuenta; si aquí aparece algo, repetir el script más tarde para confirmarlo.
aws resourcegroupstaggingapi get-resources --tag-filters "Key=$TAG_KEY,Values=$TAG_VALUE" \
    --query "ResourceTagMappingList[].ResourceARN" --output text | tr -d '\r' | tr '\t' '\n' |
    sed 's/^/  /'

if [ "$FAILED" = 0 ]; then
    echo "OK: no queda ningún recurso del proyecto."
else
    echo "QUEDAN recursos: revisar las líneas QUEDA y volver a ejecutar el script."
    exit 1
fi
