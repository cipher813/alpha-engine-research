# alpha-engine-research

Autonomous investment research system for US equities. Maintains rolling investment theses for a configurable universe of stocks, scans the broader S&P 500/400 daily for the 3 strongest buy candidates, and delivers a consolidated morning research brief via email at 6:15am PT on NYSE trading days.

Each daily run builds on the prior day's reports rather than starting from scratch — LLM agents read the previous thesis, integrate new developments, and prune stale content. After a few weeks the reports compound into something substantially richer than day one.

**Cost: ~$5.70/month** (almost entirely Claude API; AWS services fall within free tier)

---

## Quick Start

```bash
git clone https://github.com/your-username/alpha-engine-research.git
cd alpha-engine-research

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY

# Configure stock universe and email
cp config/universe.sample.yaml config/universe.yaml
# Edit config/universe.yaml — set your tickers and email addresses

# Dry run — no email, no S3 write
python3 main.py --dry-run --skip-scanner

# Full run
python3 main.py
```

See [DOCS.md](DOCS.md) for complete deployment instructions (AWS setup, Lambda, EventBridge, SES).

---

## Architecture

```
Config Universe (20 stocks) + 3 Active Buy Candidates
         │
         ▼
Data Fetchers (price/news/analyst/macro — parallel)
         │
         ├── BRANCH A: Universe Pipeline (fan-out per ticker)
         │      ├── Technical Score Engine (no LLM)
         │      ├── News Agent (claude-haiku, per ticker)
         │      ├── Research Agent (claude-haiku, per ticker)
         │      └── Macro Agent (claude-sonnet, global)
         │
         └── BRANCH B: Scanner Pipeline
                ├── Stage 1: Quant Filter (~900 → ~50, no LLM)
                ├── Stage 2: Data Enrichment (no LLM)
                ├── Stage 3: Scanner Ranking Agent (1 Sonnet call, 50 → top 10)
                └── Stage 4: Deep Analysis (Haiku fan-out, top 10)
         │
         ▼ [fan-in join]
Score Aggregator (weighted composite + per-sector macro modifier)
         │
         ▼
Thesis Updater + Candidate Evaluator (rule-based rotation)
         │
         ▼
Consolidator Agent (claude-sonnet, ~500-word email body)
         │
         ▼
Archive Writer (S3 + SQLite)
         │
         ▼
Email Sender (AWS SES, 6:15am PT, NYSE trading days only)
```

---

## Attractiveness Score (0–100)

```
Base  = Technical × 0.40 + News × 0.30 + Research × 0.30
Score = Base + macro_shift   (clipped to [0, 100])

macro_shift = (sector_modifier − 1.0) / 0.30 × 10
  → modifier 0.70 = −10 pts  |  1.0 = 0 pts  |  1.30 = +10 pts
```

Macro uses an **additive bounded shift** (±10 pts max) rather than a multiplier. This preserves conviction-level ratings through moderate macro headwinds — a caution regime (modifier ~0.85) nudges scores down ~5 points rather than compressing all scores by 15%, which would uniformly drop strong-conviction holdings from BUY to HOLD regardless of their individual fundamentals.

| Score | Rating | Interpretation (12-month vs SPY) |
|-------|--------|----------------------------------|
| 65–100 | **BUY** | Expected to match or outperform the market |
| 40–64 | **HOLD** | Expected flat to mild underperformance |
| 0–39 | **SELL** | Expected to underperform the market |

### Technical Score (no LLM)

| Signal | Weight | Notes |
|--------|--------|-------|
| RSI (14-day) | 25% | Regime-aware: overbought threshold raises to 80 in `bull` market |
| MACD signal cross | 20% | Bullish cross above zero line = 100 |
| Price vs 50-day MA | 20% | >5% above = 80, at = 50, >5% below = 30 |
| Price vs 200-day MA | 20% | Same scale |
| 20-day momentum | 15% | Percentile-ranked within S&P 500 universe |

### Per-Sector Macro Modifier (0.70–1.30)

The Macro Agent (claude-sonnet) outputs 11 sector-specific multipliers rather than a single global modifier. This matters because macro conditions affect sectors differently: rising rates hurt Real Estate and Utilities but help Financials; high oil benefits Energy but hurts Consumer Discretionary.

### Score Staleness Tracking

Scores unchanged for ≥5 trading days receive a `⚠stale` flag in the email. This signals the next agent run should scrutinize this ticker for overlooked developments. Staleness does not change the score itself.

### GBM Predictor Integration (activated 2026-03-14)

The GBM predictor (alpha-engine-predictor) veto gate is now active. When the predictor outputs a DOWN prediction with confidence >= 0.65 for a ticker, the executor overrides the ENTER signal to HOLD, preventing entry into declining positions. This adds a quantitative ML filter on top of the LLM-based research signals. Configuration: `predictor.enabled: true` in `config/universe.yaml`.

### Score Performance Feedback Loop

The system tracks whether BUY-rated stocks (score ≥65) actually outperform SPY over subsequent 10 and 30 trading day windows. If accuracy falls below 55% over a trailing 60-day window, a recalibration flag is set in the consolidated report.

---

## Market Scanner

A 5-stage pipeline runs daily in parallel with the universe pipeline:

1. **Quant Filter** (no LLM): ~900 S&P 500+400 stocks → ~50 via two paths:
   - **Momentum path**: tech score ≥60, price ≥$10, volume ≥500k, not in severe downtrend
   - **Deep value path**: RSI <35, oversold, analyst consensus ≥Buy
2. **Data Enrichment** (no LLM): fetch headlines + analyst data for all ~50
3. **Scanner Ranking** (1 Sonnet call): rank all 50 simultaneously → top 10 with rationale
4. **Deep Analysis** (Haiku fan-out): news + research agents for each top-10 candidate
5. **Candidate Evaluator** (rule-based, no LLM): tiered rotation logic

### Rotation Thresholds (tiered by tenure)

| Tenure | Score delta required | Rationale |
|--------|---------------------|-----------|
| ≤3 days | ≥12 pts | Prevent whipsaw rotation of new entries |
| 4–10 days | ≥8 pts | Early tenure — require meaningful gap |
| 11–30 days | ≥5 pts | Established — standard threshold |
| ≥31 days | ≥3 pts | Long-held — scoring has stabilized; low bar |

At most **1 rotation per daily run**.

---

## Archive Strategy

Every agent report and investment thesis is **permanently retained** in S3. The system never overwrites prior data — it only appends. This enables:
- Compounding institutional knowledge: each run updates from prior context, not scratch
- Full lifecycle tracking of buy candidates (promotions, demotions, re-promotions)
- Historical review of any thesis at any date

### S3 Layout

```
s3://alpha-engine-research/
├── archive/
│   ├── universe/{TICKER}/                # latest + history/{YYYY-MM-DD}/
│   ├── candidates/{TICKER}/              # persists after demotion
│   └── macro/                            # latest + history/{YYYY-MM-DD}/
├── consolidated/{YYYY-MM-DD}/morning.md  # final email body
└── research.db                           # SQLite (all structured data)
```

---

## Three-Step Thesis Drafting Protocol

All agents follow this mandatory protocol (§7.3):

1. **Start from existing**: Load archived prior report. Preserve every finding that remains valid.
2. **Add new material findings**: Integrate only material new developments — skip minor noise.
3. **Remove stale content**: Prune resolved events, superseded ratings, outdated commentary. When in doubt, retain.

This produces compounding institutional knowledge rather than daily amnesia.

---

## Project Structure

```
alpha-engine-research/
├── config/
│   └── universe.yaml              # Tracked tickers, weights, email config
├── agents/
│   ├── news_agent.py              # News/sentiment agent (Haiku, per-ticker)
│   ├── research_agent.py          # Analyst research agent (Haiku, per-ticker)
│   ├── macro_agent.py             # Macro environment agent (Sonnet, global)
│   ├── scanner_ranking_agent.py   # Cross-stock ranking (Sonnet, 1 call)
│   └── consolidator.py            # Final report synthesis (Sonnet)
├── data/
│   ├── fetchers/
│   │   ├── price_fetcher.py       # yfinance OHLCV + technical indicators
│   │   ├── news_fetcher.py        # Yahoo Finance RSS + SEC EDGAR 8-K
│   │   ├── analyst_fetcher.py     # FMP analyst consensus + price targets
│   │   └── macro_fetcher.py       # FRED rates + commodity prices via yfinance
│   ├── deduplicator.py            # Article hash tracking + mention_count
│   └── scanner.py                 # S&P 500/400 screener + candidate evaluator
├── scoring/
│   ├── technical.py               # RSI, MACD, MA, momentum → 0–100
│   ├── aggregator.py              # Weighted composite + macro modifier
│   └── performance_tracker.py     # BUY signal realized return tracking
├── thesis/
│   └── updater.py                 # Score → buy/sell/hold + summary
├── archive/
│   ├── manager.py                 # S3 read/write + SQLite CRUD
│   └── models.py                  # Pydantic models
├── emailer/
│   ├── formatter.py               # Markdown → HTML + plain text
│   └── sender.py                  # AWS SES delivery
├── graph/
│   └── research_graph.py          # LangGraph state machine
├── lambda/
│   ├── handler.py                 # Main daily pipeline Lambda
│   └── alerts_handler.py          # Intraday price alert Lambda
├── infrastructure/
│   ├── deploy.sh                  # Build + deploy both Lambda functions
│   ├── iam-policy.json
│   └── trust-policy.json
├── local/
│   ├── run.py                     # Local test runner
│   └── sync_db.py                 # Pull/push research.db to/from S3
└── tests/
    ├── test_scoring.py
    ├── test_agents.py
    ├── test_scanner.py
    ├── test_archive.py
    └── test_graph.py
```

---

## Data Sources

| Data Type | Source | Cost |
|-----------|--------|------|
| Price data (OHLCV) | `yfinance` | Free |
| Technical indicators | Computed from yfinance | Free |
| News headlines + body | Yahoo Finance RSS | Free |
| SEC 8-K filings | EDGAR full-text search API | Free |
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier (250 req/day; usage ~73/day) |
| Macro data | FRED CSV API | Free |
| Commodity prices | yfinance (`CL=F`, `GC=F`, `HG=F`) | Free |
| LLM | Anthropic Claude API | ~$5.09/month |
| Email | AWS SES | ~$0/month (free tier) |
| Storage | AWS S3 | ~$0.50/month |

**Total: ~$5.70/month**

---

## LLM Strategy

| Agent | Model | Rationale |
|-------|-------|-----------|
| News Agent (all tickers) | `claude-haiku-4-5` | Templated incremental update — Haiku handles well at low cost |
| Research Agent (all tickers) | `claude-haiku-4-5` | Same — reads prior context, integrates new data |
| Scanner News/Research (top-10) | `claude-haiku-4-5` | Same implementation, same rationale |
| Scanner Ranking Agent | `claude-sonnet-4-6` | Cross-stock comparative judgment requires Sonnet |
| Macro Agent | `claude-sonnet-4-6` | Nuanced economic interpretation |
| Consolidator | `claude-sonnet-4-6` | Synthesis across all reports — highest visibility output |

---

## Scheduling

A single EventBridge rule (`alpha-research-daily`) fires at 13:15 and 14:15 UTC on weekdays. The Lambda time-gates on PT: it only runs the pipeline when the current time in `America/Los_Angeles` is 6:10–6:25am. DST is handled automatically — no manual rule swap.

| Rule | UTC Cron | Purpose |
|------|----------|---------|
| alpha-research-daily | `cron(15 13,14 ? * MON-FRI *)` | 6:15am PT run; Lambda gates on PT time |
| alpha-research-alerts | `cron(0/30 13-21 ? * MON-FRI *)` | Intraday price alerts |

Use `infrastructure/setup-eventbridge.sh` to create the single-rule setup. NYSE market holidays are detected by `exchange_calendars` in the Lambda handler and cause an early exit. Early-close days (partial sessions) still run.

### Intraday Price Alerts

A separate Lambda (`alpha-engine-research-alerts`) runs every 30 minutes during market hours (9:30am–4:00pm ET) and fires an email alert when any tracked ticker moves ≥5% from prior close. 60-minute per-ticker cooldown prevents alert fatigue.

---

## AWS Infrastructure

```
S3 Bucket:        s3://alpha-engine-research
Lambda (main):    alpha-engine-research-runner    (1024 MB, 600s timeout)
Lambda (alerts):  alpha-engine-research-alerts    (256 MB, 60s timeout)
EventBridge:      alpha-research-daily (cron 13:15,14:15 UTC, MON-FRI — Lambda time-gates to 6:15am PT)
EventBridge Alerts: every 30 min, 13:30–21:00 UTC, MON-FRI
IAM Role:         alpha-engine-research-role
SES Identity:     [configured sender email]
Region:           us-east-1
```

---

## Design Decision Rationale

### Archive-First, Never-Overwrite
Every agent always starts from its prior archived report. LLM-generated research compounds over time — a report integrating 30 days of incremental findings is more accurate than a daily-fresh report. The archive ensures continuity and prevents analytical drift from run to run. Storage cost is negligible (~$0.50/month).

### Per-Sector Macro Modifier (not a single global)
A single global modifier treats rising rates identically for Utilities (hurt — bond proxy) and Financials (helped — net interest margin). Sector-specific modifiers require the Macro Agent to reason about transmission mechanisms, producing systematically more accurate scores. Prompt complexity increase is minimal.

### Hybrid LLM Strategy (Haiku + Sonnet)
Per-stock agents perform a structured, templated task (load → integrate → prune). Haiku handles this reliably at ~3.5× lower cost. Strategic agents (ranking, macro, consolidation) require broader judgment across multiple stocks or complex data — Sonnet adds meaningful quality there. Result: ~$5/month vs. ~$15/month all-Sonnet.

### Scanner Multi-Stage Architecture
900 stocks in one LLM call exceeds context limits and produces poor results. The staged approach mirrors human analyst workflow: screen mechanically first, evaluate shortlist qualitatively. Quant filter (Stage 1) costs nothing and reduces 900→50. Ranking agent (Stage 3) sees all 50 simultaneously for cross-stock comparison — critical for relative judgment. Only top 10 get expensive deep analysis.

### Single Ranking Agent for Cross-Stock Comparison
50 independent per-stock calls produce 50 scores with no cross-stock context. The ranking agent sees all candidates together and can make relative judgments: "stock A has stronger momentum but stock B has a clearer catalyst." One Sonnet call (~$0.02) produces a ranked top-10 more effectively than 50 isolated Haiku calls.

### Rule-Based Candidate Rotation
"Does X's score exceed Y's by ≥N points?" is a math inequality, not a judgment call. LLM rotation would add non-determinism and cost with no quality benefit. The tiered rotation rules (different thresholds by tenure) are more reliably enforced in code than in a prompt.

### Tiered Rotation Thresholds
A flat 5-point threshold causes whipsaw rotation — a stock promoted two days ago can immediately be replaced, then re-promoted the next day. Tiered thresholds protect new entries from immediate replacement while applying a low bar to long-held picks where scoring has stabilized.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover scoring engine, agent JSON extraction, scanner rotation logic, archive CRUD operations, and scheduling logic. All tests run without LLM API calls or AWS access.

---

## Open Items

The following items require your action before the system is fully operational:

### 1. API Keys — Required Before Any Run
- **`ANTHROPIC_API_KEY`** — Set in `.env` (local) and Lambda environment variables. Required for all LLM agents.
- **`FMP_API_KEY`** — Financial Modeling Prep API key for analyst consensus data. Free tier (250 req/day) is sufficient. Register at [financialmodelingprep.com](https://financialmodelingprep.com/). Set in `.env` and Lambda env vars.
- **`FRED_API_KEY`** — Federal Reserve economic data API key. Free. Register at [fred.stlouisfed.org/docs/api/](https://fred.stlouisfed.org/docs/api/api_key.html). Set in `.env` and Lambda env vars.

### 2. Email Configuration — Required Before Email Delivery Works
- **SES sender email**: Update `email.sender` in `config/universe.yaml` with a real email address, then verify it in AWS SES (Console → SES → Verified Identities → Create Identity).
- **SES sandbox exit**: By default, SES sandbox mode only allows sending to verified addresses. Request production access in the AWS SES console to send to arbitrary recipients.
- **Recipient email(s)**: Update `email.recipients` in `config/universe.yaml`.

### 3. AWS Infrastructure Setup — Required Before Lambda Deployment
All steps use the AWS CLI or Console:

```bash
# a. Create S3 bucket
aws s3 mb s3://alpha-engine-research --region us-east-1

# b. Create IAM role
aws iam create-role \
  --role-name alpha-engine-research-role \
  --assume-role-policy-document file://infrastructure/trust-policy.json

aws iam put-role-policy \
  --role-name alpha-engine-research-role \
  --policy-name alpha-engine-research-policy \
  --policy-document file://infrastructure/iam-policy.json

# c. Set LAMBDA_ROLE_ARN and deploy
export LAMBDA_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/alpha-engine-research-role
bash infrastructure/deploy.sh both

# d. Set Lambda environment variables (ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY)
# via AWS Console or:
aws lambda update-function-configuration \
  --function-name alpha-engine-research-runner \
  --environment "Variables={ANTHROPIC_API_KEY=...,FMP_API_KEY=...,FRED_API_KEY=...}"
```

### 4. EventBridge Rules — Required for Scheduled Execution
Run the setup script (creates single DST-aware rule + alerts rule, adds Lambda targets):

```bash
bash infrastructure/setup-eventbridge.sh
```

The script disables the old PDT/PST rules if present and creates `alpha-research-daily`, which fires at 13:15 and 14:15 UTC. The Lambda time-gates and only runs at 6:15am PT — no manual DST swap needed.

### 5. Scanner Universe Timing Validation
The scanner fetches price data for S&P 500+400 (~900 stocks). Current implementation samples the first 150 for timing safety. Test the full ~900-stock download locally before expanding to validate Lambda timeout compliance. If needed, the scanner can be split into a separate Lambda invoked asynchronously.

### 6. First-Run Bootstrap
On the first run, no archived reports exist. All agents will produce "initial reports" from scratch. This is expected and handled by the thesis drafting protocol. The second run will have prior context to build on.

### 7. Candidate Rotation Threshold Tuning
The tiered rotation thresholds (12/8/5/3 points by tenure) are starting assumptions. Review `candidate_tenures` and `scanner_appearances` tables after 2 weeks of live data to assess whether thresholds need adjustment.

### 8. Re-Promoted Candidate Stale Report Handling
When a previously-demoted candidate re-enters the top-10, its prior S3 report is loaded as context. If the prior report is >30 trading days old, agents should apply aggressive staleness pruning. The current prompt instructs this but it can be reinforced by passing a `report_age_days` variable explicitly to the prompt when the gap exceeds 30 days.

### 9. CloudWatch Monitoring
Set up CloudWatch alarms for:
- Lambda errors (`Errors > 0` for either function)
- Lambda duration (`Duration > 540000ms` for main runner — 90s before timeout)
- Lambda throttles

---

## License

MIT — see [LICENSE](LICENSE).
