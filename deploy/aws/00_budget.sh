#!/usr/bin/env bash
# Paso 0: alerta de presupuesto mensual. Se crea ANTES que cualquier otro recurso.
# Uso: bash deploy/aws/00_budget.sh tu_correo@ejemplo.com
#
# Un presupuesto AVISA por correo, no apaga nada: lo que apaga es 99_teardown.sh.
# Los presupuestos sin acciones automáticas no tienen costo.

# -e: parar en el primer error; -u: error si se usa una variable sin definir;
# -o pipefail: un fallo dentro de una tubería (a | b) también cuenta como error.
set -euo pipefail
source "$(dirname "$0")/config.sh"

EMAIL="${1:?Indica el correo para las alertas: bash deploy/aws/00_budget.sh tu_correo@ejemplo.com}"

# Idempotente: si ya existe, no se crea otra vez.
if aws budgets describe-budget --account-id "$ACCOUNT_ID" --budget-name "$BUDGET_NAME" \
    >/dev/null 2>&1; then
    echo "El presupuesto $BUDGET_NAME ya existe; no se crea de nuevo."
    exit 0
fi

# IncludeCredit=false: sin esto el presupuesto resta los créditos promocionales, vería 0 USD y
# nunca avisaría. Así mide el gasto real aunque después lo pague el crédito.
aws budgets create-budget \
    --account-id "$ACCOUNT_ID" \
    --budget "{
        \"BudgetName\": \"$BUDGET_NAME\",
        \"BudgetLimit\": {\"Amount\": \"$BUDGET_USD\", \"Unit\": \"USD\"},
        \"TimeUnit\": \"MONTHLY\",
        \"BudgetType\": \"COST\",
        \"CostTypes\": {\"IncludeCredit\": false, \"IncludeRefund\": false}
    }" \
    --notifications-with-subscribers "[
        {
            \"Notification\": {\"NotificationType\": \"ACTUAL\", \"ComparisonOperator\": \"GREATER_THAN\",
                             \"Threshold\": 20, \"ThresholdType\": \"PERCENTAGE\"},
            \"Subscribers\": [{\"SubscriptionType\": \"EMAIL\", \"Address\": \"$EMAIL\"}]
        },
        {
            \"Notification\": {\"NotificationType\": \"ACTUAL\", \"ComparisonOperator\": \"GREATER_THAN\",
                             \"Threshold\": 100, \"ThresholdType\": \"PERCENTAGE\"},
            \"Subscribers\": [{\"SubscriptionType\": \"EMAIL\", \"Address\": \"$EMAIL\"}]
        }
    ]"

# Verificación: debe aparecer el nombre del presupuesto.
aws budgets describe-budgets --account-id "$ACCOUNT_ID" --query "Budgets[].BudgetName"
