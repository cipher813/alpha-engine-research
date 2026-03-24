# Alpha Engine Research

Autonomous investment research pipeline for US equities. 6 sector teams with ReAct tool-calling agents screen the S&P 900 weekly, maintain rolling investment theses, and submit recommendations to a CIO for population selection. Delivers a consolidated morning research brief via email.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

---

## Role in the System

Research is the signal generator. It produces `signals/{date}/signals.json` — the file that the Predictor and Executor both depend on. Runs weekly on Monday (06:00 UTC via EventBridge). Each sector team builds on prior theses rather than starting from scratch, creating compounding institutional knowledge over time.

**signals.json** contains:
- `universe` — list of all stocks with signal (ENTER/HOLD/EXIT), score, rating, conviction, sector_rating, thesis
- `buy_candidates` — subset with ENTER signals and enriched CIO theses
- `market_regime`, `sector_ratings` — macro context for executor position sizing
- `population` — current 25-stock portfolio ticker list

Research and Predictor Training are **independent** — no data flows between them. They run sequentially on Monday only to spread API load.

---

## Architecture (v2 — Sector Teams)

```
PARALLEL (LangGraph Send() fan-out)
├── Technology Team   (Quant→Qual→PeerReview→2-3 picks + thesis maintenance)
├── Healthcare Team   (same pattern)
├── Financials Team
├── Industrials Team
├── Consumer Team
├── Defensives Team
├── Macro Economist   (regime + sector ratings, reflection loop)
└── Exit Evaluator    (determines exits from current population)
     │
SEQUENTIAL (fan-in)
├── Merge Results     (combine team picks + macro + exits → compute open slots)
├── Score Aggregator  (quant × w_quant + qual × w_qual + macro shift + boosts)
├── CIO               (single Sonnet batch: evaluate all picks on 4 dimensions)
├── Population Entry   (place CIO ADVANCE decisions into population)
├── Consolidator      (morning email: macro, sector allocation, notable developments, universe ratings)
├── Archive Writer    (S3 + SQLite + thesis history + IC audit trail + signals.json)
└── Email Sender      (markdown → HTML via formatter, Gmail SMTP primary, SES fallback)
```

### 6 Sector Teams

| Team | GICS Sectors | ~Stocks |
|------|-------------|---------|
| Technology | Information Technology | ~120 |
| Healthcare | Healthcare | ~110 |
| Financials | Financials | ~140 |
| Industrials | Industrials, Materials | ~215 |
| Consumer | Consumer Disc, Consumer Staples, Comm Services | ~190 |
| Defensives | Energy, Utilities, Real Estate | ~145 |

### Intra-Team Flow

Each team runs autonomously using ReAct tool-calling agents:

1. **Quant Analyst** (Haiku, ReAct with tools, max 8 iterations): Screens the sector universe using tools it chooses — volume filters, technical indicators, analyst consensus, balance sheet, options flow. Produces ranked top 10.
2. **Qualitative Analyst** (Haiku, ReAct with tools, max 8 iterations): Reviews quant's top 5 using news, analyst reports, insider activity, SEC filings, prior theses. Produces holistic `qual_score` (0-100) per stock. May add 0-1 additional candidate.
3. **Peer Review** (Haiku): Quant reviews qual's addition (if any). Joint finalization selects 2-3 recommendations.
4. **Thesis Maintenance**: For held population stocks, update thesis only when material triggers fire (news spike, price move > 2 ATR, analyst revision, earnings proximity, insider cluster, sector regime change).

### CIO Evaluation

The CIO (Sonnet) receives all team recommendations in a single batch call and evaluates on 4 dimensions:

1. **Team conviction** — quant+qual agreement, peer review flags
2. **Macro alignment** — regime fit, sector rating
3. **Portfolio fit** — diversification vs concentration
4. **Catalyst specificity** — time-bound vs vague

Selects top N for open slots. Every decision (ADVANCE, REJECT, NO_ADVANCE_DEADLOCK) saved with full rationale to `thesis_history` table.

---

## Proprietary Source Files

This repo is open-source, but files containing prompts, scoring logic, and orchestration are **gitignored**. You must create these files yourself to run the pipeline.

### Gitignored (Must Create)

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration loader |
| `config/universe.yaml` | Population size, thresholds, LLM models, architecture version |
| `graph/research_graph.py` | v1 LangGraph orchestrator |
| `agents/macro_agent.py` | Macro economist with reflection loop |
| `agents/news_agent.py` | v1 news sentiment agent |
| `agents/research_agent.py` | v1 research analysis agent |
| `agents/scanner_ranking_agent.py` | v1 scanner ranking agent |
| `agents/consolidator.py` | Email brief synthesizer |
| `agents/investment_committee/ic_cio.py` | CIO batch evaluator |
| `agents/sector_teams/quant_analyst.py` | Quant analyst ReAct agent |
| `agents/sector_teams/qual_analyst.py` | Qual analyst ReAct agent |
| `agents/sector_teams/peer_review.py` | Intra-team peer review |
| `agents/sector_teams/sector_team.py` | Team orchestrator |
| `agents/debate_agents.py` | v1 bull/bear/judge agents |
| `agents/synthesis_judge.py` | v1 divergence resolution |
| `scoring/technical.py` | Technical scoring engine |
| `scoring/aggregator.py` | v1 weighted composite |
| `scoring/performance_tracker.py` | Signal accuracy tracking |
| `data/scanner.py` | v1 quant filter pipeline |
| `data/population_selector.py` | Population management + exit/entry handlers |
| `thesis/updater.py` | Thesis record builder |

### Tracked (Infrastructure + Tools)

- `agents/sector_teams/team_config.py` — GICS-to-team mapping, slot allocation
- `agents/sector_teams/quant_tools.py` — LangChain @tool wrappers for quant agent
- `agents/sector_teams/qual_tools.py` — LangChain @tool wrappers for qual agent
- `agents/sector_teams/material_triggers.py` — Thesis update triggers (no LLM)
- `agents/prompt_loader.py` — Loads prompts from `config/prompts/`
- `agents/token_guard.py` — Token budget validation
- `graph/research_graph_v2.py` — v2 LangGraph graph with Send() fan-out
- `scoring/composite.py` — v2 composite scoring (quant × w_quant + qual × w_qual)
- `data/fetchers/` — All 8 data fetchers (price, news, analyst, macro, insider, institutional, options, revision)
- `data/deduplicator.py` — Duplicate headline handling
- `archive/manager.py` — S3 + SQLite CRUD + thesis history + IC audit trail + `load_latest_theses()` for prior week backfill
- `retry.py` — Exponential-backoff retry decorator (used by archive manager + price fetcher)
- `emailer/` — Markdown to HTML email + SES/Gmail delivery
- `lambda/` — AWS Lambda handlers (main pipeline + intraday alerts)
- `infrastructure/` — Deploy scripts, IAM policies

---

## How Scoring Works (v2)

```
composite = quant_score × w_quant + qual_score × w_qual + macro_shift + signal_boosts

Signal boosts: PEAD (±5), analyst revisions (±3), options flow (±4),
               insider activity (±5), short interest (±4), institutional (±3)
               → aggregate capped at ±10 pts
```

- `w_quant` and `w_qual` auto-tuned by backtester via `config/scoring_weights.json`
- Macro shift uses additive bounded formula (not a multiplier)
- Ratings: BUY / HOLD / SELL based on configurable thresholds
- Predictor veto: high-confidence DOWN from GBM can override BUY signals

---

## Population Management

- **Target**: ~25 stocks
- **Exits**: thesis collapse (score < 40, immediate), score degradation (< 45 with tenure > 2 weeks), forced rotation floor (min 10% weekly turnover)
- **Entries**: CIO selects from sector team recommendations to fill open slots
- **Tenure protection**: 2-week minimum (except thesis collapse)
- **Max rotations**: 10 per run (~40% of population)

### Slot Allocation

Based on open slots + macro sector ratings:

| Open Slots | Base | Overweight (+1) | Market Weight (0) | Underweight (-1) |
|------------|------|-----------------|--------------------|--------------------|
| 0 | 0 | 0 | 0 | 0 |
| 1-3 | 1 | 2 | 1 | 0 |
| 4-7 | 2 | 3 | 2 | 1 |
| 8-10 | 3 | 4 | 3 | 2 |

Teams always produce 2-3 picks regardless of allocation. The CIO selects from the full pool.

---

## Weekly Email Format

The morning research brief contains 4 sections:

1. **Macro Regime Summary** — full macro narrative (Fed, yield curve, credit, equity, commodities), key risks to monitor
2. **Sector Allocation** — 11-sector table with overweight/market weight/underweight ratings and rationale
3. **Notable Developments** — high-conviction recommendations, CIO advances, material exits
4. **Universe Ratings** — unified table of all stocks with status column:

| Status | Meaning |
|--------|---------|
| NEW | Entered portfolio this week (CIO advanced) |
| BUY REC | Buy recommendation, not yet held (no open slot) |
| UPDATED | Continuing stock with material trigger, fresh thesis |
| HOLD | Continuing stock, carryover thesis from prior week |
| EXIT | Dropped from portfolio this week |

Sorted by status priority, then score descending. Each stock has rating, score, and thesis rationale.

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
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier (250 req/day) |
| EPS revision trends | Financial Modeling Prep (FMP) | Free tier |
| Options chain (IV, put/call) | yfinance | Free |
| Macro data | FRED CSV API | Free |
| LLM | Anthropic Claude (Haiku + Sonnet) | ~$0.89/week ($46/year) |
| Email | AWS SES + Gmail SMTP | Free tier |
| Storage | AWS S3 | ~$0.50/month |

FMP rate-limited: 1 req/sec with global thread lock, 250/day budget tracked in-process. First 429 response immediately disables all FMP calls for remainder of run (graceful degradation — agents proceed without analyst data). yfinance uses sequential batches of 100 with rate-limited session via `requests-ratelimiter`.

---

## Quick Start

### Prerequisites

- Python 3.12+
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
# Edit universe.yaml — set architecture_version, LLM models, thresholds

# 3. Create gitignored source files (see table above)

# 4. Dry run (skips email + S3 write, still calls APIs/LLM)
python local/run.py --dry-run

# 5. Deploy to Lambda
./infrastructure/deploy.sh main
```

### Offline Mode

Run the full pipeline end-to-end with **zero external calls** — no API keys, no LLM, no S3, no network required. All data fetchers, LLM agents, S3 operations, and email are replaced with synthetic stubs that return structurally valid data.

```bash
# Basic offline run (uses today's date)
python local/run.py --offline

# Offline with a specific date
python local/run.py --offline --date 2026-03-23
```

**What gets stubbed:**

| Category | Stubbed Functions | Synthetic Output |
|----------|-------------------|-----------------|
| Price data | `fetch_price_data`, `fetch_sp500_sp400_with_sectors`, `yf.download` | 252-day synthetic OHLCV for 30 sample tickers |
| News | `fetch_all_news` | 2 synthetic headlines per ticker |
| Analyst | `fetch_analyst_consensus`, `fetch_revisions` | Randomized consensus ratings + targets |
| Macro | `fetch_macro_data`, `compute_market_breadth` | Hardcoded neutral macro environment |
| Insider/Options | `fetch_insider_activity`, `fetch_options_signals`, `fetch_short_interest`, `fetch_institutional_accumulation` | Empty or minimal synthetic data |
| V1 LLM agents | `run_news_agent`, `run_research_agent`, `run_macro_agent_with_reflection`, `run_scanner_ranking_agent`, `run_consolidator_agent`, `run_synthesis_judge`, `run_candidate_debate` | Deterministic scores (40-75 range), stub reports |
| V2 sector teams | `run_sector_team`, `run_quant_analyst`, `run_qual_analyst`, `run_peer_review` | Synthetic picks with randomized quant/qual scores |
| V2 CIO | `run_cio` | Advances candidates up to open slot count |
| S3 / Archive | `download_db`, `upload_db`, `load_predictions_json`, `write_signals_json`, `upload_population_json` | Uses local SQLite (creates empty if none exists), skips all S3 I/O |
| Email | `send_email` | Prints subject line to stdout |

**Use cases:**
- Validate graph topology and node wiring after refactors
- Test scoring/aggregation/population logic changes without burning API credits
- CI pipeline smoke tests
- Develop and debug new graph nodes or scoring features
- Onboard new contributors without requiring API keys or AWS credentials

**How it works:** `local/offline_stubs.py` monkey-patches all external call sites at import time. Stubs are installed before graph modules load, so `from X import Y` bindings in the graph pick up the patched functions. The pipeline runs in ~1 second.

**Combining with other flags:**

```bash
# Offline + skip scanner (V1 only — fastest possible run)
python local/run.py --offline --skip-scanner
```

> **Note:** `--offline` implies `--dry-run` behavior (no email delivery, no S3 writes). You don't need to pass both.

---

## Database Tables

### Core (v1)
`investment_thesis`, `agent_reports`, `population`, `population_history`, `scanner_appearances`, `technical_scores`, `macro_snapshots`, `score_performance`, `news_article_hashes`, `predictor_outcomes`

### v2 Additions
- `stock_archive` — every stock ever analyzed (ticker, sector, team, first/last analyzed, times in population)
- `thesis_history` — every thesis version (author: `team:technology`, `ic:cio`, etc.), full bull/bear/catalysts/risks
- `analyst_resources` — which tools each agent used per ticker (for future refinement)

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests cover scoring engine, agent JSON extraction, rotation logic, archive CRUD, and scheduling. No LLM API calls or AWS access required.

### Local Pipeline Runs

```bash
# Offline mode: synthetic data, no API/LLM/S3 calls
python local/run.py --offline

# No-S3 mode: real APIs, writes signals to local/output/ instead of S3, no email
# Safe for preprod testing — does not affect live executor
python local/run.py --no-s3

# Full local run: real APIs, writes to S3, sends email (same as Lambda)
python local/run.py --local

# Specify date
python local/run.py --offline --date 2026-03-24
```

**Preprod workflow:**
1. `--offline` — verify pipeline logic runs without crashes
2. `--no-s3` — verify with real APIs, inspect `local/output/signals-{date}.json`
3. Validate output: `python ~/Development/alpha-engine/tests/validate_signals.py local/output/signals-{date}.json`
4. `--local` — push to prod S3 and send email

---

## Opportunities

### Validation (needs live data)
- **Quant-only ablation**: Run system without LLM agents to prove multi-agent value
- **Sector balancing hypothesis**: Compare Sharpe with vs without sector constraints
- **Debate conviction vs performance**: Track whether CIO conviction correlates with outperformance (needs 200+ samples, ~20 weeks)
- **Agent tool refinement**: Analyze `analyst_resources` table to understand which tools each sector uses most

### Architecture Enhancements
- **Adaptive slot allocation**: Weight team recommendations by historical accuracy
- **Backtester team performance**: Per-team signal accuracy at 10d/30d, fed back to CIO
- **IC critic**: Add a Haiku critic to challenge CIO selections before finalization
- ~~**Thesis maintenance optimization**: Only update theses for stocks with material news~~ ✅ Implemented — material triggers check all held stocks weekly, carryover thesis for unchanged stocks

### Performance
- ~~**S3 price caching**: Cache weekly price data to avoid re-downloading 900 tickers from yfinance~~ ✅ Predictor maintains S3 price cache; research uses sequential rate-limited batches
- **Batch S3 I/O**: Consolidated JSON alongside per-ticker files
- **LangGraph state optimization**: Return only changed keys from memory-intensive nodes

### Data Gaps
- **Survivorship bias**: Wikipedia S&P constituents are current-only — need append-only log
- **Sub-sector analysis**: Semiconductors and SaaS treated identically as "Technology"
- **Archive TTL**: Old theses remain indefinitely — add configurable expiration

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis and parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
