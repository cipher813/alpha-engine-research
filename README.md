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

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
