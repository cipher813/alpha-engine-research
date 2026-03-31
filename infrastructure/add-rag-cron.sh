#!/bin/bash
# Register the weekly RAG ingestion cron job on the always-on EC2 instance.
# Safe to run multiple times — replaces existing entry.
#
# Schedule: Saturdays at 05:00 UTC (Friday ~9-10pm PT)
# Runs 1 hour before Research Lambda (06:00 UTC) so RAG data is fresh.
#
# Pipelines: SEC filings (10-K/10-Q), 8-K material events,
#            thesis history from signals.json, filing change detection.
#
# Secrets sourced from ~/.alpha-engine.env (shared with executor/backtester).
#
# Usage:
#   bash infrastructure/add-rag-cron.sh

set -euo pipefail

ENV_FILE="/home/ec2-user/.alpha-engine.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Create it with RAG_DATABASE_URL, VOYAGE_API_KEY, FINNHUB_API_KEY, then chmod 600."
    exit 1
fi

SOURCE_ENV=". ${ENV_FILE} &&"

CRON_LINE="0 5 * * 6  cd /home/ec2-user/alpha-engine-research && git pull --ff-only >> /var/log/rag-ingestion.log 2>&1 && ${SOURCE_ENV} bash rag/pipelines/run_weekly_ingestion.sh >> /var/log/rag-ingestion.log 2>&1"

# Remove existing RAG ingestion entry, then add new one
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "rag.*ingestion\|run_weekly_ingestion" || true)

{
    echo "$FILTERED"
    echo "$CRON_LINE"
} | crontab -

echo "RAG ingestion cron job registered: Saturdays 05:00 UTC"
echo "  Pipeline: SEC filings + 8-Ks + theses + filing change detection"
echo "  Secrets: sourced from ${ENV_FILE}"
echo "  Log: /var/log/rag-ingestion.log"
echo ""
echo "Current crontab:"
crontab -l
