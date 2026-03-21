# Alpha Engine Research

Autonomous investment research pipeline for US equities. Maintains rolling investment theses for a tracked population of ~25 stocks, scans the S&P 500/400 weekly for candidates to rotate in, and delivers a consolidated morning research brief via email.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

---

## Role in the System

Research is the signal generator. It produces `signals/{date}/signals.json` — the file that the Predictor and Executor both depend on. Runs weekly on Monday (06:00 UTC via EventBridge). Each run builds on prior theses rather than starting from scratch, so reports compound into something substantially richer over time.

The output `universe[]` array contains all tracked population stocks with scores, ratings, and signals. There is no separate buy_candidates category — all stocks are treated uniformly.

---

## Proprietary Source Files

This repo is open-source, but 13 files containing tuned parameters, prompt text, and scoring logic are **gitignored**. You must create these files yourself to run the pipeline. Sample configs and example prompts are provided as starting points.

### Files You Must Create

| File | Purpose | Key Inputs | Key Outputs |
|------|---------|------------|-------------|
| `config.py` | Centralized configuration loader | Config YAML files | Settings dict for all modules |
| `graph/research_graph.py` | LangGraph state machine orchestrator | All agent outputs, config | Orchestrated pipeline execution |
| `agents/news_agent.py` | News sentiment analysis per ticker | Headlines, SEC filings, prior report | News score (0-100), sentiment, catalysts |
| `agents/research_agent.py` | Analyst research analysis per ticker | Analyst data, price data, prior report | Research score (0-100), thesis |
| `agents/macro_agent.py` | Macro environment and sector analysis | FRED data, prior macro report | Market regime, sector modifiers |
| `agents/scanner_ranking_agent.py` | Cross-stock candidate ranking | Quant-filtered candidates | Ranked top-N with path classification |
| `agents/consolidator.py` | Synthesize all analyses into email brief | All agent outputs, scores | Formatted morning brief (markdown) |
| `scoring/technical.py` | Deterministic technical scoring engine | OHLCV, indicators | Technical score (0-100) per ticker (feeds Predictor, not research composite) |
| `scoring/aggregator.py` | Weighted composite + macro modifier | News + research sub-scores, sector modifiers | Final score (0-100) + rating |
| `scoring/performance_tracker.py` | Signal accuracy tracking | Historical scores, price outcomes | Accuracy metrics, recalibration flags |
| `data/scanner.py` | Multi-stage quant filter pipeline | S&P 500/400 universe, price data | Shortlisted candidates (~50) |
| `data/population_selector.py` | Sector-balanced population management | Current population, scored candidates | Updated population with rotation events |
| `thesis/updater.py` | Score to rating + thesis summary | Aggregated scores, prior thesis | Rating (BUY/HOLD/SELL), updated thesis |

### What's Included (Infrastructure)

These files are tracked in git and provide the framework:

- `agents/prompt_loader.py` — Loads prompts from `config/prompts/`
- `agents/token_guard.py` — Token budget validation before LLM calls
- `data/fetchers/` — Price, news, analyst, macro, insider, institutional, options, revision data fetchers (yfinance, Yahoo RSS, FMP, FRED, EDGAR)
- `data/deduplicator.py` — Duplicate headline handling
- `archive/manager.py` — S3 read/write + SQLite CRUD
- `emailer/` — Markdown to HTML email + SES/Gmail delivery
- `lambda/` — AWS Lambda handlers (main pipeline + intraday alerts)
- `infrastructure/` — Deploy scripts, IAM policies
- `health_status.py` — Health monitoring (write/read/check upstream module health)
- `retry.py` — Exponential backoff retry decorator for resilient API calls
- `tests/` — Unit tests (no LLM/AWS required)
- `main.py` — CLI entry point

### Reference Materials

- `config/universe.sample.yaml` — Universe config template with safe defaults
- `config/scoring.sample.yaml` — Scoring config template with equal weights
- `config/prompts.example/*.txt` — Generic prompt templates with placeholder scoring logic
- `.env.example` — Required API keys and AWS config

---

## Quick Start

### Prerequisites

- Python 3.11+
- API keys: `ANTHROPIC_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY`
- AWS credentials with S3 read/write and SES send permission

### Setup

```bash
git clone https://github.com/cipher813/alpha-engine-research.git
cd alpha-engine-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Environment variables
cp .env.example .env
# Edit .env — add API keys, AWS config, email settings

# 2. Configuration files
cp config/universe.sample.yaml config/universe.yaml
cp config/scoring.sample.yaml config/scoring.yaml
cp -r config/prompts.example config/prompts
# Edit all config files — set your tickers, weights, thresholds, and prompts

# 3. Create the 13 proprietary source files listed above
#    Use the sample configs and example prompts as reference

# 4. Dry run — no email, no S3 write
python3 main.py --dry-run --skip-scanner

# 5. Full run
python3 main.py
```

---

## Architecture

```
Data Fetchers (1y OHLCV for ~900 tickers, macro data — parallel batches)
         |
         v
Scanner Pipeline (sequential stages)
    Stage 1: Quant Filter (~900 → ~60, no LLM — RSI, MACD, MA200, ATR%)
    Stage 2: Data Enrichment + Balance Sheet Filter (~60 → ~50, fail-closed)
    Stage 3: Scanner Ranking Agent (1 Sonnet call, ~50 → 35)
         |
         v
Population Agents (per-ticker, parallel via ThreadPoolExecutor)
    +-- Macro Agent (1 Sonnet call, global regime + sector ratings)
    +-- Per-ticker fan-out (35 tickers × 2 agents each):
         +-- News Agent (Haiku) — sentiment, catalysts, thesis update
         +-- Research Agent (Haiku) — analyst consensus, thesis update
         |
         v
Score Aggregator (news + research weighted composite + macro shift + signal boosts ±10 cap)
         |
         v
Thesis Updater → Population Evaluator (sector-balanced rotation, 10-40% weekly turnover)
         |
         v
Consolidator Agent (Sonnet, email body) → Archive Writer (S3 + SQLite) → Email Sender
```

---

## How Scoring Works

Each stock receives a composite attractiveness score (0-100) blending two sub-scores adjusted by a per-sector macro modifier:

```
Base   = (news_score * w_news) + (research_score * w_research)
Shift  = macro_shift + signal_boosts
Score  = Base + Shift    — signal_boosts capped at ±10 pts, final clipped to [0, 100]

Signal boosts: PEAD (±5), analyst revisions (±3), options flow (±4),
               insider activity (±5), short interest (±4), institutional (±3)
               → individual caps preserved, aggregate capped at ±10 pts
```

Scoring weights are auto-tuned weekly by the backtester via `config/scoring_weights.json` on S3.

The macro modifier uses an **additive bounded shift** rather than a multiplier — a caution regime nudges scores down a few points rather than compressing all scores uniformly. Six additional signal boosts (PEAD, analyst revisions, options flow, insider activity, short interest, 13F institutional) are applied on top of the base composite, individually capped and aggregate-capped at ±10 points to prevent transient signal stacking from dominating the core LLM assessment.

- **Per-sector macro modifier**: The Macro Agent outputs sector-specific multipliers because macro conditions affect sectors differently.
- **Ratings**: BUY / HOLD / SELL based on configurable score thresholds.
- **Predictor veto**: High-confidence DOWN predictions from the GBM predictor can override BUY signals.

Technical analysis (RSI, MACD, momentum) is handled by the Predictor (GBM features) and Executor (ATR stops), not the research composite.

---

## Market Scanner

A multi-stage pipeline runs before the population analysis:

1. **Quant Filter** (no LLM): S&P 500+400 to shortlist via volume, volatility (ATR%), and balance sheet screens (D/E, current ratio — fail-closed: candidates with unavailable financial data are excluded)
2. **Data Enrichment** (no LLM): Fetch headlines, analyst data, insider activity, options flow for shortlisted stocks
3. **Scanner Ranking** (1 Sonnet call): Rank all candidates simultaneously via momentum and deep-value paths
4. **Deep Analysis** (Haiku fan-out): News + research agents for each candidate
5. **Population Evaluator** (rule-based): Sector-balanced rotation with min 10% / max ~40% weekly turnover, 2-week tenure protection, and immediate removal on thesis collapse (score < 40)

---

## Configuration Reference

| File | Contents |
|------|----------|
| `config/universe.yaml` | Population size, rating thresholds, scanner settings, rotation tiers, LLM models |
| `config/scoring.yaml` | Composite weights (news/research), macro modifier params |
| `config/prompts/*.txt` | LLM agent prompts (scoring baselines, thesis protocol, output format) |
| `.env` | API keys (Anthropic, FMP, FRED), AWS config, email settings |

Sample/example versions are provided in the repo. Copy them and tune all parameters to your own research.

---

## Data Sources

| Data Type | Source | Cost |
|-----------|--------|------|
| Price data (OHLCV) | yfinance | Free |
| Technical indicators | Computed from price data | Free |
| News headlines | Yahoo Finance RSS | Free |
| SEC 8-K filings | EDGAR full-text search API | Free |
| SEC Form 4 insider transactions | EDGAR (EdgarTools) | Free |
| 13F institutional holdings | SEC EDGAR | Free |
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier |
| EPS revision trends | Financial Modeling Prep (FMP) | Free tier |
| Options chain (IV, put/call) | yfinance | Free |
| Macro data | FRED CSV API | Free |
| LLM | Anthropic Claude API | ~$5/month |
| Email | AWS SES + Gmail SMTP | Free tier |
| Storage | AWS S3 | ~$0.50/month |

---

## Key Files

```
alpha-engine-research/
+-- main.py                           # CLI entry point
+-- health_status.py                  # Health monitoring (write/read/check upstream)
+-- retry.py                          # Exponential backoff retry decorator
+-- config/
|   +-- universe.sample.yaml          # Template — safe defaults
|   +-- scoring.sample.yaml          # Template — safe defaults
|   +-- prompts.example/              # Template prompts
+-- agents/
|   +-- prompt_loader.py              # Loads prompts from config/prompts/
|   +-- token_guard.py                # Token budget validation
+-- data/
|   +-- fetchers/                     # price, news, analyst, macro, insider,
|   |                                 # institutional, options, revision fetchers
|   +-- deduplicator.py              # Duplicate headline handling
+-- scoring/                          # (proprietary) aggregator, technical, performance tracker
+-- archive/
|   +-- manager.py                    # S3 read/write + SQLite CRUD
+-- emailer/                          # Markdown to HTML + SES/Gmail delivery
+-- lambda/                           # AWS Lambda handlers (main + alerts)
+-- infrastructure/                   # Deploy scripts, IAM policies
+-- tests/                            # Unit tests (no LLM/AWS required)
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover scoring engine, agent JSON extraction, scanner rotation logic, archive CRUD, and scheduling logic. All tests run without LLM API calls or AWS access.

---

## Opportunities for Improvement

All originally identified data source gaps have been implemented — see `data/fetchers/` for the full set (insider, institutional, options, revision, short interest, credit spreads, equity breadth).

### LLM Agent Gaps

1. **News sentiment quantification is keyword-only** — recurring themes use word frequency, not semantic analysis. "China tariffs" (bearish) and "China market expansion" (bullish) are indistinguishable. Plan: add LLM-based theme sentiment scoring before passing to the news agent — a single Haiku call per theme batch to classify sentiment direction and magnitude.

2. **Consistency check uses numeric divergence** — `check_consistency()` flags when news and research sub-scores diverge by >30 points (e.g., news > 70 and research < 40). This catches gross mismatches reliably. Plan: optionally add a single Haiku call for full semantic consistency check on flagged tickers.

### Scanner Pipeline Gaps

1. **No per-sector candidate cap in scanner** — the population evaluator enforces sector slots, but the scanner's quant filter could surface 50 Technology stocks. Plan: add configurable per-sector candidate cap (e.g., max 30% from any single sector) in the quant filter stage.

2. **Deep value path could be stronger** — currently checks RSI < 35 + analyst consensus + configurable ATR threshold. Plan: add price-to-book, free cash flow yield, or distance from 52-week low as confirming signals.

3. **No sub-sector (industry group) analysis** — semiconductors and SaaS treated identically as "Technology." Plan: use GICS sub-industry data to provide industry-level context to agents and apply sub-sector concentration limits. *Dependency: requires GICS sub-industry data source (Wikipedia tables or paid provider).*

### Thesis Management

1. **No archive expiration** — old wrong theses remain in archive indefinitely. Plan: add TTL-based archive cleanup that moves theses older than a configurable max age (e.g., 6 months) to a cold archive prefix in S3.

### Survivorship Bias

1. **Wikipedia S&P constituents are current-only** — delisted stocks disappear from universe retroactively. Plan: maintain a local append-only constituents log that tracks additions and removals, so historical backtests use point-in-time membership. *Dependency: no free authoritative source for historical S&P constituent changes — S&P Global charges for this data.*

2. **No tracking of removed stocks** — stocks removed from S&P 500/400 and their subsequent performance are invisible. Plan: when a ticker drops off the constituents list, log the removal date and continue tracking price for 90 days to measure post-removal drift.

3. **Price floor creates look-ahead bias** — recovered penny stocks are included only after recovery. Plan: apply the price floor at the point-in-time of each signal date rather than at scan time, or accept the bias as a feature (we genuinely don't want to trade sub-$10 names).

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
