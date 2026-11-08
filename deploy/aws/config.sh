# Configuración común de los scripts de deploy/aws/. No se ejecuta sola: cada script la carga con
#   source "$(dirname "$0")/config.sh"
# No contiene números de cuenta ni claves: la cuenta se obtiene de la sesión de la CLI.

# Git Bash (Windows) convierte los argumentos que empiezan por "/" en rutas de Windows:
# "/health" llegaría a AWS como "C:/Program Files/Git/health". Esto lo desactiva; en Linux no
# tiene efecto.
export MSYS_NO_PATHCONV=1

# Región de todos los recursos (D2 de la spec). AWS_DEFAULT_REGION hace que cada comando la use
# sin tener que escribir --region.
export AWS_DEFAULT_REGION="us-east-1"

# Prefijo de todos los nombres y etiqueta que llevan todos los recursos (D9). La etiqueta permite
# al teardown comprobar que no quedó nada.
PROJECT="credit-risk"
TAG_KEY="Project"
TAG_VALUE="$PROJECT"

# Número de cuenta de la sesión actual (el mismo que muestra `aws sts get-caller-identity`).
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Nombres de los recursos.
BUDGET_NAME="${PROJECT}-monthly"
BUDGET_USD="5"
BUCKET="${PROJECT}-models-${ACCOUNT_ID}" # los nombres de bucket son únicos en todo AWS
ECR_REPO="${PROJECT}-api"
SERVICE="${PROJECT}-api"
EXECUTION_ROLE="${PROJECT}-ecs-execution"
INFRA_ROLE="${PROJECT}-ecs-infrastructure"
