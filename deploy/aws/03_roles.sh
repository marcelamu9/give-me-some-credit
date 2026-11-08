#!/usr/bin/env bash
# Paso 3: los dos roles de IAM que necesita ECS Express Mode. Los roles no tienen costo.
# Uso: bash deploy/aws/03_roles.sh
#
# Un rol tiene dos partes: la política de CONFIANZA (quién puede asumirlo) y la de PERMISOS
# (qué puede hacer quien lo asuma). Se usan las políticas gestionadas que indica la guía oficial:
# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html
#
# La API no recibe ningún rol propio (task role): el modelo va dentro de la imagen y la API no
# llama a AWS. Mínimo privilegio.

set -euo pipefail
source "$(dirname "$0")/config.sh"

# Crea el rol si no existe (idempotente) y le adjunta la política gestionada.
#   $1 nombre del rol, $2 servicio que puede asumirlo, $3 ARN de la política de permisos
create_role() {
    local role="$1" principal="$2" policy_arn="$3"
    if aws iam get-role --role-name "$role" >/dev/null 2>&1; then
        echo "El rol $role ya existe."
    else
        aws iam create-role --role-name "$role" \
            --tags "Key=$TAG_KEY,Value=$TAG_VALUE" \
            --assume-role-policy-document "{
                \"Version\": \"2012-10-17\",
                \"Statement\": [{
                    \"Effect\": \"Allow\",
                    \"Principal\": {\"Service\": \"$principal\"},
                    \"Action\": \"sts:AssumeRole\"
                }]
            }" >/dev/null
        echo "Rol $role creado."
    fi
    # Adjuntar una política que ya está adjunta no da error.
    aws iam attach-role-policy --role-name "$role" --policy-arn "$policy_arn"
}

# Rol de ejecución: lo asume el agente que ARRANCA las tareas (ecs-tasks) para descargar la
# imagen de ECR y escribir los logs en CloudWatch.
create_role "$EXECUTION_ROLE" "ecs-tasks.amazonaws.com" \
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"

# Rol de infraestructura: lo asume ECS para crear el balanceador, el certificado HTTPS, los
# grupos de seguridad y el escalado del servicio Express Mode.
create_role "$INFRA_ROLE" "ecs.amazonaws.com" \
    "arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"

# Verificación: cada rol con su política.
for role in "$EXECUTION_ROLE" "$INFRA_ROLE"; do
    aws iam list-attached-role-policies --role-name "$role" \
        --query "AttachedPolicies[].PolicyName" --output text
done

# IAM es "eventualmente consistente": un rol recién creado puede tardar ~1 minuto en poder
# usarse. Si 04_service.sh falla con un error de assume-role, esperar y repetir.
