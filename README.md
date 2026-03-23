# Alpha Engine Research

Autonomous investment research pipeline for US equities. 6 sector teams with ReAct tool-calling agents screen the S&P 900 weekly, maintain rolling investment theses, and submit recommendations to a CIO for population selection. Delivers a consolidated morning research brief via email.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

---

## Role in the System

Research is the signal generator. It produces `signals/{date}/signals.json` — the file that the Predictor and Executor both depend on. Runs weekly on Monday (06:00 UTC via EventBridge). Each sector team builds on prior theses rather than starting from scratch, creating compounding institutional knowledge over time.

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
├── Consolidator      (morning email with exit rationales + CIO decisions)
├── Archive Writer    (S3 + SQLite + thesis history + IC audit trail)
└── Email Sender
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
- `archive/manager.py` — S3 + SQLite CRUD + thesis history + IC audit trail
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

FMP rate-limited with thread-safe 250ms interval + retry with exponential backoff on 429.

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

# 4. Dry run
python local/run.py --dry-run

# 5. Deploy to Lambda
./infrastructure/deploy.sh main
```

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
- **Thesis maintenance optimization**: Only update theses for stocks with material news

### Performance
- **S3 price caching**: Cache weekly price data to avoid re-downloading 900 tickers from yfinance
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
