#!/usr/bin/env bash
# setup-eventbridge.sh — EventBridge rules for main pipeline + alerts.
#
# Weekly population refresh: Monday 06:00 UTC (Sunday ~10-11pm PT).
# Lambda receives {"weekly_run": true} in event payload — bypasses
# the 5:45am PT time gate but still runs the full pipeline.
#
# Weekday rule (daily 5:45am PT) is kept but DISABLED by default.
# Re-enable for transition back to daily if needed.
#
# Usage: ./infrastructure/setup-eventbridge.sh

set -euo pipefail

FUNCTION_MAIN="alpha-engine-research-runner"
FUNCTION_ALERTS="alpha-engine-research-alerts"
RULE_DAILY="alpha-research-daily"
RULE_WEEKLY="alpha-research-weekly"
RULE_ALERTS="alpha-research-alerts"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Setting up EventBridge rules..."

# Remove old rules if they exist (no-op if missing)
for old in alpha-research-pdt alpha-research-pst alpha-research-sunday; do
  if aws events describe-rule --name "$old" --region "$REGION" &>/dev/null; then
    echo "Removing old rule: $old"
    aws events remove-targets --rule "$old" --ids "1" --region "$REGION" 2>/dev/null || true
    aws events delete-rule --name "$old" --region "$REGION" 2>/dev/null || true
  fi
done

# ── Weekly rule (Saturday 06:00 UTC = Friday ~10-11pm PT; population refresh) ──
# Saturday gives a weekend buffer to fix any pipeline issues before Monday trading.
aws events put-rule \
  --name "$RULE_WEEKLY" \
  --schedule-expression "cron(0 6 ? * SAT *)" \
  --state ENABLED \
  --description "Saturday 06:00 UTC (Fri night PT) — weekly population refresh" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_WEEKLY" \
  --targets '[{"Id":"1","Arn":"arn:aws:lambda:'"${REGION}"':'"${ACCOUNT_ID}"':function:'"${FUNCTION_MAIN}"'","Input":"{\"weekly_run\": true}"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_MAIN" \
  --statement-id "alpha-research-weekly" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_WEEKLY}" \
  --region "$REGION" 2>/dev/null || true

# ── Weekday rule (DISABLED — kept for rollback to daily if needed) ──
aws events put-rule \
  --name "$RULE_DAILY" \
  --schedule-expression "cron(45 12,13 ? * MON-FRI *)" \
  --state DISABLED \
  --description "5:45am PT weekdays — DISABLED (using weekly schedule)" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_DAILY" \
  --targets '[{"Id":"1","Arn":"arn:aws:lambda:'"${REGION}"':'"${ACCOUNT_ID}"':function:'"${FUNCTION_MAIN}"'"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_MAIN" \
  --statement-id "alpha-research-daily" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_DAILY}" \
  --region "$REGION" 2>/dev/null || true

# ── Alerts rule (every 30 min, market hours, Mon–Fri) ──
aws events put-rule \
  --name "$RULE_ALERTS" \
  --schedule-expression "cron(0/30 13-21 ? * MON-FRI *)" \
  --state ENABLED \
  --description "Intraday price alerts" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_ALERTS" \
  --targets '[{"Id":"1","Arn":"arn:aws:lambda:'"${REGION}"':'"${ACCOUNT_ID}"':function:'"${FUNCTION_ALERTS}"'"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_ALERTS" \
  --statement-id "alpha-research-alerts" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_ALERTS}" \
  --region "$REGION" 2>/dev/null || true

echo ""
echo "Done. Active rules: $RULE_WEEKLY, $RULE_ALERTS"
echo "      Disabled:     $RULE_DAILY (re-enable for daily if needed)"
echo ""
