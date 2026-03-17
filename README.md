# Alpha Engine Research

Autonomous investment research pipeline for US equities. Maintains rolling investment theses for a configurable universe of stocks, scans the broader S&P 500/400 for buy candidates, and delivers a consolidated morning research brief via email.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

---

## Role in the System

Research is the signal generator. It produces `signals/{date}/signals.json` — the file that the Predictor and Executor both depend on. Each weekly run builds on prior theses rather than starting from scratch, so reports compound into something substantially richer over time.

---

## ⚠ Proprietary Source Files

This repo is open-source, but 13 files containing tuned parameters, prompt text, and scoring logic are **gitignored**. You must create these files yourself to run the pipeline. Sample configs and example prompts are provided as starting points.

### Files You Must Create

| File | Purpose | Key Inputs | Key Outputs |
|------|---------|------------|-------------|
| `config.py` | Centralized configuration loader | Config YAML files | Settings dict for all modules |
| `graph/research_graph.py` | LangGraph state machine orchestrator | All agent outputs, config | Orchestrated pipeline execution |
| `agents/news_agent.py` | News sentiment analysis per ticker | Headlines, SEC filings, prior report | News score (0–100), sentiment, catalysts |
| `agents/research_agent.py` | Analyst research analysis per ticker | Analyst data, price data, prior report | Research score (0–100), thesis |
| `agents/macro_agent.py` | Macro environment and sector analysis | FRED data, prior macro report | Market regime, sector modifiers |
| `agents/scanner_ranking_agent.py` | Cross-stock candidate ranking | Quant-filtered candidates | Ranked top-N with path classification |
| `agents/consolidator.py` | Synthesize all analyses into email brief | All agent outputs, scores | Formatted morning brief (markdown) |
| `scoring/technical.py` | Deterministic technical scoring engine | OHLCV, indicators | Technical score (0–100) per ticker |
| `scoring/aggregator.py` | Weighted composite + macro modifier | Sub-scores, sector modifiers | Final score (0–100) + rating |
| `scoring/performance_tracker.py` | Signal accuracy tracking | Historical scores, price outcomes | Accuracy metrics, recalibration flags |
| `data/scanner.py` | Multi-stage quant filter pipeline | S&P 500/400 universe, price data | Shortlisted candidates (~50) |
| `data/population_selector.py` | Sector-balanced universe management | Current universe, candidates | Updated tracked population |
| `thesis/updater.py` | Score → rating + thesis summary | Aggregated scores, prior thesis | Rating (BUY/HOLD/SELL), updated thesis |

### What's Included (Infrastructure)

These files are tracked in git and provide the framework:

- `agents/prompt_loader.py` — Loads prompts from `config/prompts/`
- `data/fetchers/` — Price, news, analyst, macro data fetchers (yfinance, Yahoo RSS, FMP, FRED, EDGAR)
- `data/deduplicator.py` — Duplicate headline handling
- `archive/manager.py` — S3 read/write + SQLite CRUD
- `emailer/` — Markdown → HTML email + SES delivery
- `lambda/` — AWS Lambda handlers (main pipeline + intraday alerts)
- `infrastructure/` — Deploy scripts, IAM policies
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
Config Universe + Active Buy Candidates
         │
         ▼
Data Fetchers (price/news/analyst/macro — parallel)
         │
         ├── BRANCH A: Universe Pipeline (fan-out per ticker)
         │      ├── Technical Score Engine (no LLM)
         │      ├── News Agent (Haiku, per ticker)
         │      ├── Research Agent (Haiku, per ticker)
         │      └── Macro Agent (Sonnet, global)
         │
         └── BRANCH B: Scanner Pipeline
                ├── Stage 1: Quant Filter (~900 → ~50, no LLM)
                ├── Stage 2: Data Enrichment (no LLM)
                ├── Stage 3: Scanner Ranking Agent (1 Sonnet call)
                └── Stage 4: Deep Analysis (Haiku fan-out)
         │
         ▼ [fan-in join]
Score Aggregator (weighted composite + per-sector macro modifier)
         │
         ▼
Thesis Updater + Candidate Evaluator (rule-based rotation)
         │
         ▼
Consolidator Agent (Sonnet, email body)
         │
         ▼
Archive Writer (S3 + SQLite) → Email Sender (AWS SES)
```

---

## How Scoring Works

Each stock receives a composite attractiveness score (0–100) blending sub-scores from multiple sources, adjusted by a per-sector macro modifier:

```
Base  = weighted_sum(sub_scores)    — weights configurable in scoring.yaml
Score = Base + macro_shift          — clipped to [0, 100]
```

The macro modifier uses an **additive bounded shift** rather than a multiplier — a caution regime nudges scores down a few points rather than compressing all scores uniformly.

- **Technical score** (no LLM): Deterministic scoring from RSI, MACD, price vs moving averages, momentum. Supports regime-aware thresholds.
- **Per-sector macro modifier**: The Macro Agent outputs sector-specific multipliers because macro conditions affect sectors differently.
- **Ratings**: BUY / HOLD / SELL based on configurable score thresholds.
- **Predictor veto**: High-confidence DOWN predictions from the GBM predictor can override BUY signals.

---

## Market Scanner

A multi-stage pipeline runs in parallel with the universe pipeline:

1. **Quant Filter** (no LLM): S&P 500+400 → shortlist via configurable screening criteria
2. **Data Enrichment** (no LLM): Fetch headlines + analyst data for shortlisted stocks
3. **Scanner Ranking** (1 Sonnet call): Rank all candidates simultaneously → top N
4. **Deep Analysis** (Haiku fan-out): News + research agents for each candidate
5. **Candidate Evaluator** (rule-based): Tiered rotation logic with tenure-based thresholds

---

## Configuration Reference

| File | Contents |
|------|----------|
| `config/universe.yaml` | Universe size, rating thresholds, scanner settings, rotation tiers, LLM models |
| `config/scoring.yaml` | Technical scoring thresholds, composite weights, macro modifier params |
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
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier |
| Macro data | FRED CSV API | Free |
| LLM | Anthropic Claude API | ~$5/month |
| Email | AWS SES | Free tier |
| Storage | AWS S3 | ~$0.50/month |

---

## Key Files

```
alpha-engine-research/
├── main.py                           # CLI entry point
├── config/
│   ├── universe.sample.yaml          # Template — safe defaults
│   ├── scoring.sample.yaml           # Template — safe defaults
│   └── prompts.example/              # Template prompts
├── agents/
│   └── prompt_loader.py              # Loads prompts from config/prompts/
├── data/
│   ├── fetchers/                     # Price, news, analyst, macro data fetchers
│   └── deduplicator.py              # Duplicate headline handling
├── archive/
│   └── manager.py                    # S3 read/write + SQLite CRUD
├── emailer/                          # Markdown → HTML + SES delivery
├── lambda/                           # AWS Lambda handlers
├── infrastructure/                   # Deploy scripts, IAM policies
└── tests/                            # Unit tests (no LLM/AWS required)
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover scoring engine, agent JSON extraction, scanner rotation logic, archive CRUD, and scheduling logic. All tests run without LLM API calls or AWS access.

---

## Opportunities for Improvement

### Data Source Gaps

| Data Source | Alpha Impact | Cost | Notes |
|-------------|-------------|------|-------|
| Options put/call ratio & IV skew | HIGH — validates/contradicts analyst sentiment | Free (yfinance options chain) | Orthogonal to current momentum-heavy signals |
| Earnings revision trends (week-over-week EPS consensus) | HIGH — "revisions up" is strong bullish signal | Free (diff FMP weekly) | 4-week revision direction as feature |
| SEC Form 4 insider transactions | MEDIUM — cluster buying is 6-12 month signal | Free (EdgarTools) | 3+ C-level buys within 30 days |
| Short interest & borrow rates | MEDIUM — crowded trade risk flag | Free (finviz scrape) | |
| Leading indicators (ISM PMI, jobless claims) | MEDIUM — leads equity drawdowns 3-6 months | Free (FRED) | |
| Credit spreads (HY-Treasury OAS) | MEDIUM — risk-on/off indicator | Free (FRED) | |
| Equity breadth (advance/decline, % above 50d MA) | MEDIUM — confirms/contradicts SPY price | Free (finviz scrape) | |
| 13F institutional holdings changes | LOW — quarterly lag reduces timeliness | Free (SEC EDGAR) | |

### LLM Agent Gaps

1. **Market regime definition is subjective** — macro agent uses bull/neutral/caution/bear labels with no quantitative thresholds. Different runs may classify the same data differently. Plan: define regimes mathematically (e.g., Bull: VIX < 15 AND SPY 30d return > 0; Bear: VIX > 25 OR SPY 30d return < -5%). Supplement LLM judgment with a quantitative floor.

2. **News sentiment quantification is keyword-only** — recurring themes use word frequency, not semantic analysis. "China tariffs" (bearish) and "China market expansion" (bullish) are indistinguishable. Plan: add LLM-based theme sentiment scoring before passing to the news agent — a single Haiku call per theme batch to classify sentiment direction and magnitude.

3. **Prior report length unbounded** — if accumulated thesis is 5000 words, agent's output budget is consumed by context. Plan: truncate prior_report to most recent 500 words or last 3 months of content before injecting into agent prompts.

4. **Consistency check uses regex, not semantics** — `check_consistency()` in aggregator.py uses keyword matching. "Bullish but headwinds may resurface" gets bullish_hits=1, bearish_hits=1, classified as consistent. Plan: use LLM-based consistency check (single Haiku call) or require keyword dominance ratio > 2:1.

### Scanner Pipeline Gaps

1. **No volatility screen** — high-volatility stocks pass if volume threshold met. Plan: add realized volatility filter (e.g., reject if 20d vol > 3x sector median) to prevent scanner from surfacing names with outsized risk.

2. **No debt/balance sheet filter** — near-default companies can pass scanner. Plan: add debt-to-equity or interest coverage screen via FMP fundamentals data to filter out financially distressed names.

3. **No sector concentration limit in candidates** — all 60 candidates could be Technology. Plan: add configurable per-sector candidate cap (e.g., max 30% from any single sector) in the quant filter stage.

4. **Deep value path weak** — only checks RSI < 35 + single analyst "Buy" rating. Plan: strengthen with additional value signals — price-to-book, free cash flow yield, or distance from 52-week low — and require 2+ confirming signals.

5. **No sub-sector (industry group) analysis** — semiconductors and SaaS treated identically as "Technology." Plan: use GICS sub-industry from Wikipedia tables (already fetched) to provide industry-level context to agents and apply sub-sector concentration limits.

### Thesis Management

1. **Stale thesis detection exists but is never surfaced** — `stale_days` counter runs but no visual "STALE" badge appears in output or email. Plan: add stale badge to the morning brief and signals.json output when `stale_days >= threshold`.

2. **No forced thesis refresh trigger** — stale theses accumulate without action. Plan: when `stale_days >= 2 * threshold`, force a full thesis rewrite by clearing the prior report and running agents with a "fresh analysis" prompt override.

3. **No archive expiration** — 6-month-old wrong theses remain in archive indefinitely. Plan: add TTL-based archive cleanup that moves theses older than a configurable max age (e.g., 6 months) to a cold archive prefix in S3.

### Survivorship Bias

1. **Wikipedia S&P constituents are current-only** — delisted stocks disappear from universe retroactively. Plan: maintain a local append-only constituents log that tracks additions and removals, so historical backtests use point-in-time membership.

2. **No tracking of removed stocks** — stocks removed from S&P 500/400 and their subsequent performance are invisible. Plan: when a ticker drops off the constituents list, log the removal date and continue tracking price for 90 days to measure post-removal drift.

3. **Price floor ($10) creates look-ahead bias** — recovered penny stocks are included only after recovery. Plan: apply the price floor at the point-in-time of each signal date rather than at scan time, or accept the bias as a feature (we genuinely don't want to trade sub-$10 names).

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
