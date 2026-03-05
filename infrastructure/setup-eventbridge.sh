#!/usr/bin/env bash
# setup-eventbridge.sh — EventBridge rules for main pipeline + alerts.
#
# Main pipeline: single rule that fires at 13:15 and 14:15 UTC. The Lambda
# time-gates and only runs when it's 6:15am PT — DST handled automatically.
#
# Run to migrate from old PDT/PST two-rule setup, or for fresh install.
# Usage: ./infrastructure/setup-eventbridge.sh

set -euo pipefail

FUNCTION_MAIN="alpha-engine-research-runner"
FUNCTION_ALERTS="alpha-engine-research-alerts"
RULE_MAIN="alpha-research-daily"
RULE_ALERTS="alpha-research-alerts"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Setting up EventBridge rules..."

# Remove old rules if they exist (no-op if missing)
for old in alpha-research-pdt alpha-research-pst; do
  if aws events describe-rule --name "$old" --region "$REGION" &>/dev/null; then
    echo "Disabling old rule: $old"
    aws events disable-rule --name "$old" --region "$REGION" 2>/dev/null || true
  fi
done

# Create main pipeline rule (fires 13:15 and 14:15 UTC; Lambda gates on 6:15am PT)
aws events put-rule \
  --name "$RULE_MAIN" \
  --schedule-expression "cron(15 13,14 ? * MON-FRI *)" \
  --state ENABLED \
  --description "6:15am PT daily — Lambda time-gates for DST" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_MAIN" \
  --targets "Id"="1","Arn"="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_MAIN}" \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_MAIN" \
  --statement-id "alpha-research-daily" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_MAIN}" \
  --region "$REGION" 2>/dev/null || true

# Create alerts rule (every 30 min, market hours)
aws events put-rule \
  --name "$RULE_ALERTS" \
  --schedule-expression "cron(0/30 13-21 ? * MON-FRI *)" \
  --state ENABLED \
  --description "Intraday price alerts" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_ALERTS" \
  --targets "Id"="1","Arn"="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_ALERTS}" \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_ALERTS" \
  --statement-id "alpha-research-alerts" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_ALERTS}" \
  --region "$REGION" 2>/dev/null || true

echo "Done. Rules active: $RULE_MAIN, $RULE_ALERTS. No DST swap needed."
