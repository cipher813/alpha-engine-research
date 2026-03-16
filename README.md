# alpha-engine-research

Autonomous investment research system for US equities. Maintains rolling investment theses for a configurable universe of stocks, scans the broader S&P 500/400 for buy candidates, and delivers a consolidated morning research brief via email on NYSE trading days.

Each daily run builds on the prior day's reports rather than starting from scratch — LLM agents read the previous thesis, integrate new developments, and prune stale content. After a few weeks the reports compound into something substantially richer than day one.

Part of the [Alpha Engine](https://github.com/cipher813) autonomous trading system.

---

## Quick Start

```bash
git clone https://github.com/cipher813/alpha-engine-research.git
cd alpha-engine-research

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY

# Configure stock universe, scoring params, and email
cp config/universe.sample.yaml config/universe.yaml
cp config/scoring.sample.yaml config/scoring.yaml
cp -r config/prompts.example config/prompts
# Edit all config files — set your tickers, weights, thresholds, and prompts

# Dry run — no email, no S3 write
python3 main.py --dry-run --skip-scanner

# Full run
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

## Attractiveness Score (0–100)

Each stock receives a composite attractiveness score blending sub-scores from multiple sources, adjusted by a per-sector macro modifier:

```
Base  = weighted_sum(sub_scores)
Score = Base + macro_shift   (clipped to [0, 100])
```

The macro modifier uses an **additive bounded shift** rather than a multiplier. This preserves conviction-level ratings through moderate macro headwinds — a caution regime nudges scores down a few points rather than compressing all scores uniformly, which would drop strong-conviction holdings from BUY to HOLD regardless of their individual fundamentals.

### Technical Score (no LLM)

Deterministic technical scoring from price-derived indicators: RSI, MACD, price vs moving averages, and momentum. All thresholds and weights are configurable via `config/scoring.yaml`. The engine supports regime-aware thresholds (different parameters for bull/bear/neutral markets).

### Per-Sector Macro Modifier

The Macro Agent outputs sector-specific multipliers rather than a single global modifier. This matters because macro conditions affect sectors differently: rising rates hurt Real Estate and Utilities but help Financials; high oil benefits Energy but hurts Consumer Discretionary.

### GBM Predictor Integration

A GBM predictor (alpha-engine-predictor) veto gate can override BUY signals. When the model predicts DOWN with sufficient confidence, the executor overrides the ENTER signal to HOLD, preventing entry into declining positions. Configuration in `config/universe.yaml`.

### Score Performance Feedback Loop

The system tracks whether BUY-rated stocks actually outperform SPY over configurable time windows. If accuracy falls below a configured threshold, a recalibration flag is set in the consolidated report.

---

## Market Scanner

A multi-stage pipeline runs in parallel with the universe pipeline:

1. **Quant Filter** (no LLM): S&P 500+400 → shortlist via configurable screening criteria (momentum path + deep value path)
2. **Data Enrichment** (no LLM): fetch headlines + analyst data for shortlisted stocks
3. **Scanner Ranking** (1 Sonnet call): rank all candidates simultaneously → top N with rationale
4. **Deep Analysis** (Haiku fan-out): news + research agents for each top candidate
5. **Candidate Evaluator** (rule-based, no LLM): tiered rotation logic

### Rotation Logic

Candidates are protected by tenure-based rotation thresholds (configurable in `config/universe.yaml`). Recent additions require a larger score gap to be replaced, preventing whipsaw rotation. Long-held picks use a lower bar since their scores have stabilized.

---

## Archive Strategy

Every agent report and investment thesis is **permanently retained** in S3. The system never overwrites prior data — it only appends. This enables:
- Compounding institutional knowledge: each run updates from prior context, not scratch
- Full lifecycle tracking of buy candidates (promotions, demotions, re-promotions)
- Historical review of any thesis at any date

---

## Three-Step Thesis Drafting Protocol

All agents follow a mandatory protocol:

1. **Start from existing**: Load archived prior report. Preserve every finding that remains valid.
2. **Add new material findings**: Integrate only material new developments — skip minor noise.
3. **Remove stale content**: Prune resolved events, superseded ratings, outdated commentary.

This produces compounding institutional knowledge rather than daily amnesia.

---

## Configuration

All tunable parameters are in gitignored config files:

| File | Contents |
|------|----------|
| `config/universe.yaml` | Universe size, rating thresholds, scanner settings, rotation tiers, LLM models |
| `config/scoring.yaml` | Technical scoring thresholds, composite weights, macro modifier params |
| `config/prompts/*.txt` | LLM agent prompts (scoring baselines, thesis protocol, output format) |
| `.env` | API keys, AWS config |

Sample/example versions of each file are provided in the repo. Copy them, then tune all parameters to your own research.

---

## Project Structure

```
alpha-engine-research/
├── config/
│   ├── universe.yaml              # Population, weights, thresholds (gitignored)
│   ├── scoring.yaml               # Scoring engine params (gitignored)
│   ├── prompts/                   # LLM agent prompts (gitignored)
│   ├── universe.sample.yaml       # Template — safe defaults
│   ├── scoring.sample.yaml        # Template — safe defaults
│   └── prompts.example/           # Template prompts
├── agents/                        # LLM agent orchestrators
├── data/
│   ├── fetchers/                  # Price, news, analyst, macro data fetchers
│   ├── scanner.py                 # S&P 500/400 screener + candidate evaluator
│   └── population_selector.py     # Sector-balanced universe management
├── scoring/
│   ├── technical.py               # Deterministic technical scoring engine
│   ├── aggregator.py              # Weighted composite + macro modifier
│   └── performance_tracker.py     # Signal accuracy tracking
├── thesis/
│   └── updater.py                 # Score → rating + thesis summary
├── archive/
│   └── manager.py                 # S3 read/write + SQLite CRUD
├── emailer/                       # Markdown → HTML + SES delivery
├── graph/
│   └── research_graph.py          # LangGraph state machine orchestrator
├── lambda/                        # AWS Lambda handlers
├── infrastructure/                # Deploy scripts, IAM policies
├── local/                         # Local test runners
└── tests/                         # Unit tests (no LLM/AWS required)
```

---

## Data Sources

| Data Type | Source | Cost |
|-----------|--------|------|
| Price data (OHLCV) | `yfinance` | Free |
| Technical indicators | Computed from price data | Free |
| News headlines | Yahoo Finance RSS | Free |
| SEC 8-K filings | EDGAR full-text search API | Free |
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier |
| Macro data | FRED CSV API | Free |
| LLM | Anthropic Claude API | ~$5/month |
| Email | AWS SES | Free tier |
| Storage | AWS S3 | ~$0.50/month |

---

## LLM Strategy

Per-stock agents (news, research) use Haiku for cost efficiency on structured, templated tasks. Strategic agents (ranking, macro, consolidation) use Sonnet for cross-stock comparative judgment and nuanced interpretation. This hybrid approach delivers quality where it matters at ~3.5× lower cost than all-Sonnet.

---

## Design Decisions

- **Archive-first, never-overwrite**: LLM research compounds over time — 30 days of incremental findings beats a daily-fresh report
- **Per-sector macro modifiers**: Sector-specific rather than single global — macro affects sectors differently
- **Multi-stage scanner**: Quant filter (free) → ranking (1 LLM call) → deep analysis (N calls) mirrors human analyst workflow
- **Single ranking call**: Cross-stock comparison in one call beats N independent evaluations
- **Rule-based rotation**: Math inequality, not judgment — deterministic and reliable
- **Tiered rotation thresholds**: Protect recent entries from whipsaw, lower bar for stabilized holdings

---

## Related Repos

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (IB Gateway trade execution)
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) — Signal quality analysis & autonomous parameter optimization
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## Setup

See the sample config files and `.env.example` for all required configuration. Full AWS deployment docs are in `DOCS.md` (not tracked — generate from infrastructure scripts).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover scoring engine, agent JSON extraction, scanner rotation logic, archive CRUD operations, and scheduling logic. All tests run without LLM API calls or AWS access.

---

## License

MIT — see [LICENSE](LICENSE).
