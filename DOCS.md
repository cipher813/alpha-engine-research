# alpha-engine-research — Operations & Maintenance Guide

---

## Overview

`alpha-engine-research` is an autonomous equity research system that delivers a daily morning email at 6:15am PT on NYSE trading days. It maintains rolling investment theses for a fixed universe of 20 stocks you define, scans the broader S&P 500/400 daily for the 3 strongest buy candidates outside your universe, and synthesizes everything into a single actionable research brief.

The system is designed to compound knowledge over time. Each daily run builds on the prior day's reports rather than starting fresh — the LLM agents read yesterday's thesis, integrate new developments, prune stale content, and write an updated brief. After a few weeks of operation the reports become substantially richer than on day one.

It runs entirely on AWS (Lambda + S3 + SES + EventBridge) with no servers to manage. Cost is approximately $5–6/month, almost entirely Claude API usage.

---

## How It Works

Each morning the pipeline executes two branches in parallel:

**Branch A — Universe Pipeline**
For each of your 20 tracked stocks:
1. Downloads 6 months of price data and computes technical indicators (RSI, MACD, moving averages, momentum)
2. Fetches the latest news headlines and SEC filings
3. Fetches analyst consensus ratings and price targets from Financial Modeling Prep
4. Runs a News Agent (Claude Haiku) — reads prior report, integrates new articles, prunes stale content
5. Runs a Research Agent (Claude Haiku) — reads prior report, integrates new analyst actions
6. Runs a Macro Agent (Claude Sonnet, once globally) — assesses market regime and outputs per-sector modifiers

**Branch B — Scanner Pipeline**
1. Downloads price data for ~900 S&P 500/400 stocks
2. Filters to ~50 candidates via quant screen (momentum path + deep value path)
3. Runs a Scanner Ranking Agent (Claude Sonnet, one call over all 50) → top 10
4. Runs News + Research agents on the top 10
5. Evaluates whether any of the top 10 should replace a current buy candidate (rule-based, tiered rotation thresholds)

Both branches feed into a **Score Aggregator** that computes a 0–100 attractiveness score per stock:
```
Base  = Technical×0.40 + News×0.30 + Research×0.30
Score = Base + macro_shift  (bounded ±10 pts by sector macro modifier)
```

Scores are rated BUY (≥65), HOLD (40–64), or SELL (<40) on a 12-month vs. SPY horizon.

A **Consolidator Agent** (Claude Sonnet) then synthesizes all reports into the final email: macro regime summary, notable developments, universe ratings table, and deep-dive theses on the top 3 buy candidates.

---

## Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | LangGraph | Pipeline state machine with parallel branches |
| LLM | Anthropic Claude (Haiku + Sonnet) | Per-stock agents (Haiku), strategic synthesis (Sonnet) |
| Price data | yfinance | Free OHLCV, commodities, index returns |
| News | Yahoo Finance RSS + SEC EDGAR | Headlines, article excerpts, 8-K filings |
| Analyst data | Financial Modeling Prep API | Consensus ratings, price targets, rating changes |
| Macro data | FRED API | Fed funds rate, CPI, unemployment |
| Compute | AWS Lambda | Two functions: main pipeline (1024MB/600s), alerts (256MB/60s) |
| Storage | AWS S3 | All agent reports, investment theses, SQLite database |
| Email | AWS SES | Daily brief + intraday price alerts |
| Scheduling | AWS EventBridge | DST-aware cron rules |
| Database | SQLite (in S3) | Investment theses, candidate history, score performance |

**Runtime**: Python 3.12 on Lambda, Python 3.9+ locally.

---

## Data Sources

| Data | Source | API Key Required | Free Tier Limits |
|------|--------|-----------------|-----------------|
| Stock prices (OHLCV) | yfinance | No | Unlimited |
| Technical indicators | Computed locally | No | — |
| News headlines | Yahoo Finance RSS | No | Unlimited |
| SEC 8-K filings | EDGAR full-text search | No | Unlimited |
| Analyst consensus, price targets, rating changes, earnings surprises | Financial Modeling Prep (FMP) | Yes | 250 req/day (~73 used daily) |
| Fed funds rate, CPI, unemployment | FRED API | Yes | Unlimited |
| Commodity prices (oil, gold, copper) | yfinance | No | Unlimited |
| Index returns (SPY, QQQ, IWM) | yfinance | No | Unlimited |

**Note on analyst data**: The Research Agent sees structured FMP data only — consensus rating (Buy/Hold/Sell), mean price target, number of analysts, recent rating changes (last 30 days), and the last two earnings surprises. It does not access full analyst report PDFs.

---

## Output

### Daily Email (6:15am PT, trading days only)

Delivered via AWS SES to configured recipients. Structure:

1. **Macro Regime Summary** — current regime (bull/neutral/caution/bear), key macro forces, sector headwinds and tailwinds
2. **Notable Developments** — 3–5 material news/earnings/analyst actions across the universe
3. **Universe Ratings Table** — all 20 tracked stocks with rating, score, and one-sentence rationale (macro impact noted where relevant)
4. **Top 3 Buy Candidates** — 3–4 sentence deep-dive thesis per candidate, with entry status (CONTINUING / NEW_ENTRY / RETURNED)

Scores include staleness flags (`⚠stale`) when unchanged for ≥5 trading days, and a recalibration warning if BUY signal accuracy vs. SPY drops below 55% over the trailing 60 days.

### Intraday Price Alerts

A separate Lambda checks every 30 minutes during market hours (9:30am–4:00pm ET). If any tracked ticker moves ≥5% from prior close, an email alert fires. 60-minute per-ticker cooldown prevents alert fatigue.

### S3 Archive

Every report is permanently retained. Nothing is ever overwritten.

```
s3://{your-bucket}/
├── archive/
│   ├── universe/{TICKER}/latest.md          ← current report
│   ├── universe/{TICKER}/history/{date}.md  ← full history
│   ├── candidates/{TICKER}/                 ← buy candidate reports
│   └── macro/latest.md + history/
├── consolidated/{YYYY-MM-DD}/morning.md     ← final email body
├── signals/{YYYY-MM-DD}/signals.json        ← machine-readable output
├── backups/                                 ← daily DB snapshots
└── research.db                             ← SQLite (all structured data)
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.9+
- AWS CLI configured (`aws configure`)
- Three API keys: Anthropic, FMP, FRED

### Setup

```bash
# Clone and enter project
git clone https://github.com/your-username/alpha-engine-research.git
cd alpha-engine-research

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY

# Configure your stock universe and email
cp config/universe.sample.yaml config/universe.yaml
# Edit config/universe.yaml — set your tickers, sender, and recipient email addresses
```

### Test run (no email, no S3 write)

```bash
python3 main.py --dry-run --skip-scanner
```

### Full local run (sends real email, writes to S3)

```bash
python3 main.py --skip-scanner   # skip scanner on first few runs for speed
python3 main.py                  # full run including scanner
```

Output is printed to terminal and written to `run.log`.

---

## Deployment

### One-time AWS setup (if not already done)

```bash
# Create S3 bucket (choose your own bucket name)
aws s3 mb s3://your-bucket-name --region us-east-1

# Create IAM role
aws iam create-role \
  --role-name alpha-engine-research-role \
  --assume-role-policy-document file://infrastructure/trust-policy.json

aws iam put-role-policy \
  --role-name alpha-engine-research-role \
  --policy-name alpha-engine-research-policy \
  --policy-document file://infrastructure/iam-policy.json
```

Verify sender email in AWS SES console (Verified Identities → Create Identity).

### Deploy Lambda functions

```bash
export LAMBDA_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/alpha-engine-research-role
bash infrastructure/deploy.sh both
```

### Set Lambda environment variables

```bash
aws lambda update-function-configuration \
  --function-name alpha-engine-research-runner \
  --environment "Variables={S3_BUCKET=your-bucket-name,ANTHROPIC_API_KEY=...,FMP_API_KEY=...,FRED_API_KEY=...}"

aws lambda update-function-configuration \
  --function-name alpha-engine-research-alerts \
  --environment "Variables={S3_BUCKET=your-bucket-name,ANTHROPIC_API_KEY=...}"
```

### Create EventBridge rules

```bash
bash infrastructure/setup-eventbridge.sh
```

### Validate setup

```bash
# Check Lambda env vars
aws lambda get-function-configuration --function-name alpha-engine-research-runner --query 'Environment.Variables'

# Check EventBridge rules
aws events list-rules --name-prefix alpha-research --query 'Rules[].{Name:Name,State:State,Schedule:ScheduleExpression}' --output table

# Check rule targets
aws events list-targets-by-rule --rule alpha-research-daily --query 'Targets[].Arn' --output text

# Test invoke (triggers full pipeline + sends email)
aws lambda invoke --function-name alpha-engine-research-runner --payload '{}' --log-type Tail --query 'LogResult' --output text /tmp/out.json | base64 -d
```

### Redeploying after code changes

```bash
bash infrastructure/deploy.sh both
```

Run this any time you change code. Lambda env vars are preserved on redeploy.

---

## Maintenance

### Changing the stock universe

Edit `config/universe.yaml` — add or remove tickers in the `universe` list, specifying the correct `sector` for each. Valid sectors:

```
Technology, Healthcare, Financial, Consumer Discretionary,
Consumer Staples, Energy, Industrials, Materials,
Real Estate, Utilities, Communication Services
```

Redeploy after changing: `bash infrastructure/deploy.sh both`

New tickers will produce "initial reports" on their first run (no prior context). After 2–3 runs they will have compounding context like the rest.

### Adding email recipients

Edit `config/universe.yaml`:
```yaml
email:
  recipients:
    - you@example.com
    - colleague@example.com
```

If SES is in sandbox mode, each recipient email must be individually verified in the SES console. To send to unverified addresses, request production access in the SES console (takes ~24 hours to approve).

### Updating API keys

```bash
aws lambda update-function-configuration \
  --function-name alpha-engine-research-runner \
  --environment "Variables={S3_BUCKET=your-bucket-name,ANTHROPIC_API_KEY=NEW,FMP_API_KEY=NEW,FRED_API_KEY=NEW}"
```

Update `.env` locally as well.

### Reviewing past reports

All reports are in S3:

```bash
# List all consolidated briefs
aws s3 ls s3://your-bucket-name/consolidated/ --recursive

# Download a specific day's brief
aws s3 cp s3://your-bucket-name/consolidated/2026-03-05/morning.md ./

# View a ticker's current report
aws s3 cp s3://your-bucket-name/archive/universe/AAPL/latest.md -
```

### Syncing the database locally

```bash
python3 local/sync_db.py pull    # download from S3
python3 local/sync_db.py push    # upload to S3 (creates backup first)
python3 local/sync_db.py status  # compare local vs S3
```

The SQLite database (`research.db`) contains all investment theses, candidate tenure history, score performance tracking, and news article hashes.

### Monitoring

**CloudWatch Logs** — Lambda logs are automatically written to:
- `/aws/lambda/alpha-engine-research-runner`
- `/aws/lambda/alpha-engine-research-alerts`

View the last run's logs:
```bash
aws logs tail /aws/lambda/alpha-engine-research-runner --since 1h
```

**If an email doesn't arrive**:
1. Check CloudWatch logs for the Lambda function — look for errors
2. Check SES sending activity in the AWS console
3. Run locally to isolate: `python3 main.py --skip-scanner`

**If scores look off**:
- FMP free tier limit is 250 req/day. If the run exceeds this, research scores default to 50. Check FMP API usage at financialmodelingprep.com.
- The performance tracker will flag recalibration if BUY accuracy vs. SPY falls below 55% over 60 days.

### Tuning scoring

Key parameters in `config/universe.yaml`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `rating_thresholds.buy` | 65 | Score required for BUY rating |
| `rating_thresholds.sell` | 40 | Score below which = SELL |
| `scoring_weights.technical` | 0.40 | Weight of technical score in composite |
| `scoring_weights.news` | 0.30 | Weight of news/sentiment score |
| `scoring_weights.research` | 0.30 | Weight of analyst research score |

After tuning, redeploy: `bash infrastructure/deploy.sh both`

---

## Teardown

To shut down the system completely and stop all AWS charges:

### 1. Disable scheduling (immediate — stops future runs)

```bash
aws events disable-rule --name alpha-research-daily
aws events disable-rule --name alpha-research-alerts
```

### 2. Delete EventBridge rules

```bash
# Remove targets first, then delete rules
aws events remove-targets --rule alpha-research-daily --ids 1
aws events delete-rule --name alpha-research-daily

aws events remove-targets --rule alpha-research-alerts --ids 1
aws events delete-rule --name alpha-research-alerts
```

### 3. Delete Lambda functions

```bash
aws lambda delete-function --function-name alpha-engine-research-runner
aws lambda delete-function --function-name alpha-engine-research-alerts
```

### 4. Back up and delete S3 (optional — contains full report archive)

```bash
# Optional: download full archive before deleting
aws s3 sync s3://your-bucket-name ./archive-backup

# Delete all objects and bucket
aws s3 rm s3://your-bucket-name --recursive
aws s3 rb s3://your-bucket-name
```

### 5. Delete IAM role

```bash
aws iam delete-role-policy \
  --role-name alpha-engine-research-role \
  --policy-name alpha-engine-research-policy

aws iam delete-role --role-name alpha-engine-research-role
```

### 6. Remove SES verified identity (optional)

In the AWS SES console: Verified Identities → select identity → Delete.

### Cost after teardown

$0/month. All charges stop immediately upon deleting the Lambda functions and EventBridge rules. S3 storage accrues ~$0.02/month per GB until the bucket is deleted.

---

## Cost Reference

| Service | Monthly cost |
|---------|-------------|
| Claude API (Haiku + Sonnet) | ~$5.00 |
| AWS Lambda | $0 (well within free tier) |
| AWS S3 | ~$0.50 |
| AWS SES | $0 (free tier: 62,000 emails/month) |
| EventBridge | $0 (free tier) |
| **Total** | **~$5.50/month** |

Claude API usage scales slightly with universe size. Adding more tickers to the 20-stock universe increases Haiku usage proportionally; the Sonnet calls (macro, scanner ranking, consolidator) are fixed regardless of universe size.
