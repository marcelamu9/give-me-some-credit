#!/usr/bin/env bash
# Paso 6: permitir que GitHub Actions despliegue SIN guardar claves de AWS (OIDC).
# Uso: bash deploy/aws/06_github_oidc.sh
#
# Cómo funciona: GitHub firma un token que dice "soy el repo X, rama main"; AWS verifica la
# firma con el proveedor OIDC y, si el rol lo permite, entrega credenciales que caducan en 1 h.
# Los roles no tienen costo.

set -euo pipefail
source "$(dirname "$0")/config.sh"

# owner/repo a partir del remoto de git (https://github.com/owner/repo.git o git@github.com:...).
GITHUB_REPO=$(git remote get-url origin | sed -E 's#^.*github\.com[:/]##; s#\.git$##')
OIDC_HOST="token.actions.githubusercontent.com"
PROVIDER_ARN="arn:aws:iam::$ACCOUNT_ID:oidc-provider/$OIDC_HOST"
echo "Repositorio autorizado: $GITHUB_REPO (solo la rama main)"

# 1. Proveedor OIDC: "AWS confía en los tokens que firma GitHub Actions". Uno por cuenta.
#    No hace falta huella (thumbprint): AWS valida a GitHub con sus propias autoridades.
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" \
    >/dev/null 2>&1; then
    echo "El proveedor OIDC de GitHub ya existe."
else
    aws iam create-open-id-connect-provider \
        --url "https://$OIDC_HOST" \
        --client-id-list "sts.amazonaws.com" \
        --tags "Key=$TAG_KEY,Value=$TAG_VALUE" >/dev/null
    echo "Proveedor OIDC creado."
fi

# 2. Rol con confianza limitada a ESTE repositorio y a la rama main (D11 / spec §6).
#    aud: el token es para STS; sub: quién lo pide (un fork u otra rama no coincide).
TRUST="{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
        \"Effect\": \"Allow\",
        \"Principal\": {\"Federated\": \"$PROVIDER_ARN\"},
        \"Action\": \"sts:AssumeRoleWithWebIdentity\",
        \"Condition\": {\"StringEquals\": {
            \"$OIDC_HOST:aud\": \"sts.amazonaws.com\",
            \"$OIDC_HOST:sub\": \"repo:$GITHUB_REPO:ref:refs/heads/main\"
        }}
    }]
}"
if aws iam get-role --role-name "$GITHUB_ROLE" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$GITHUB_ROLE" --policy-document "$TRUST"
    echo "El rol $GITHUB_ROLE ya existe; confianza actualizada."
else
    aws iam create-role --role-name "$GITHUB_ROLE" \
        --tags "Key=$TAG_KEY,Value=$TAG_VALUE" \
        --assume-role-policy-document "$TRUST" >/dev/null
    echo "Rol $GITHUB_ROLE creado."
fi

# 3. Permisos mínimos, cada uno limitado a los recursos de este proyecto.
REGION="$AWS_DEFAULT_REGION"
REPO_ARN="arn:aws:ecr:$REGION:$ACCOUNT_ID:repository/$ECR_REPO"
SERVICE_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:service/default/$SERVICE"
aws iam put-role-policy --role-name "$GITHUB_ROLE" --policy-name "deploy" --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
        {\"Sid\": \"ReadModel\", \"Effect\": \"Allow\",
         \"Action\": [\"s3:ListBucket\"], \"Resource\": \"arn:aws:s3:::$BUCKET\"},
        {\"Sid\": \"ReadModelObjects\", \"Effect\": \"Allow\",
         \"Action\": [\"s3:GetObject\"], \"Resource\": \"arn:aws:s3:::$BUCKET/models/*\"},
        {\"Sid\": \"EcrLogin\", \"Effect\": \"Allow\",
         \"Action\": [\"ecr:GetAuthorizationToken\"], \"Resource\": \"*\"},
        {\"Sid\": \"EcrPush\", \"Effect\": \"Allow\",
         \"Action\": [\"ecr:DescribeRepositories\", \"ecr:DescribeImages\",
                    \"ecr:DescribeImageScanFindings\", \"ecr:BatchGetImage\",
                    \"ecr:BatchCheckLayerAvailability\", \"ecr:InitiateLayerUpload\",
                    \"ecr:UploadLayerPart\", \"ecr:CompleteLayerUpload\", \"ecr:PutImage\"],
         \"Resource\": \"$REPO_ARN\"},
        {\"Sid\": \"UpdateService\", \"Effect\": \"Allow\",
         \"Action\": [\"ecs:DescribeServices\", \"ecs:DescribeExpressGatewayService\",
                    \"ecs:UpdateExpressGatewayService\"],
         \"Resource\": \"$SERVICE_ARN\"},
        {\"Sid\": \"PassExecutionRole\", \"Effect\": \"Allow\",
         \"Action\": [\"iam:PassRole\"],
         \"Resource\": \"arn:aws:iam::$ACCOUNT_ID:role/$EXECUTION_ROLE\"}
    ]
}"

# Lo que hay que poner en el workflow (no es un secreto: sin el token de GitHub no sirve).
echo "ARN del rol para el workflow:"
aws iam get-role --role-name "$GITHUB_ROLE" --query Role.Arn --output text
