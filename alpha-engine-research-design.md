# alpha-engine-research — Design Document

_Drafted: 2026-03-04 · Updated: 2026-03-04 (v1.4)_

---

## 1. Executive Summary

`alpha-engine-research` is a standalone autonomous investment research system that maintains rolling, permanently-archived investment theses for a configurable universe of stocks, scours US equities for the three strongest buy candidates in the market, and delivers a single consolidated morning report via email at 6:15am PT on NYSE trading days (Monday–Friday, excluding federal market holidays). The system is agent-driven, **archive-first**, and **incrementally updated** — every thesis and report is built on top of its prior version, never from scratch, and every version is permanently retained for historical review.

---

## 2. Goals and Non-Goals

### Goals

- Maintain an up-to-date investment thesis (buy/sell/hold + attractiveness score 0–100) for each stock in a user-defined config file (initially 20 stocks).
- Scan US equities daily to identify the three strongest buy candidates from the broader market.
- Maintain and evolve investment theses for those three buy candidates alongside the core universe.
- **Archive every version of every investment thesis permanently** — both for universe stocks and buy candidates — so the full history is available for review at any time.
- **Track the full lifecycle of every buy candidate**: entries, demotions, re-promotions, and the rationale for each transition. A previously-demoted candidate that re-enters the top-3 picks up where its prior thesis left off.
- Every agent report and thesis follows a strict **three-step update protocol**: (1) start from the existing archived version, (2) integrate new material findings, (3) remove stale/outdated content. No agent ever drafts from scratch if prior context exists.
- Deliver a single consolidated email report at 6:15am PT on **NYSE trading days** (Mon–Fri, excluding federal market holidays where the exchange is closed the full day). Early-close days still run.
- Operate completely autonomously once deployed — no manual intervention required.

### Non-Goals

- No trade execution (that remains in `alpha-engine`).
- No portfolio risk management or position sizing.
- No intraday price data or real-time streaming.
- No fundamental deep-dive (10-K/10-Q parsing) in v1 — analyst consensus and headline-level research only.

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          alpha-engine-research                                │
│                                                                              │
│  Config Universe (20 stocks) + 3 Active Buy Candidates                      │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │             Data Fetchers (parallel, all sources)       │                 │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐   │                 │
│  │  │  Technical  │  │  News/RSS/   │  │  Analyst FMP │   │                 │
│  │  │  (yfinance) │  │  SEC EDGAR   │  │  + Macro FRED│   │                 │
│  │  └────────────┘  └──────────────┘  └──────────────┘   │                 │
│  └────────────────────────────────────────────────────────┘                 │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │               Archive Read (prior reports)              │                 │
│  └────────────────────────────────────────────────────────┘                 │
│          │                                                                   │
│          ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                     LangGraph Research Graph                      │       │
│  │                                                                  │       │
│  │  ┌─────────── BRANCH A (universe) ──────────────────────────┐   │       │
│  │  │                                                           │   │       │
│  │  │  [fan-out per ticker — up to 23 parallel]                │   │       │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │   │       │
│  │  │  │ Tech Score │  │ News Agent │  │  Research Agent    │  │   │       │
│  │  │  │ (no LLM)   │  │ ~300 words │  │  ~300 words        │  │   │       │
│  │  │  └────────────┘  └────────────┘  └────────────────────┘  │   │       │
│  │  │                                                           │   │       │
│  │  │  [single global]                                          │   │       │
│  │  │  ┌──────────────────────────┐                            │   │       │
│  │  │  │  Macro Agent ~300 words  │                            │   │       │
│  │  │  └──────────────────────────┘                            │   │       │
│  │  └─────────────────────────────────────┬─────────────────── ┘   │       │
│  │                                         │                        │       │
│  │  ┌─────────── BRANCH B (scanner) ───────│──────────────────┐    │       │
│  │  │                                      │                   │    │       │
│  │  │  Step 1: Quant Filter                │                   │    │       │
│  │  │  [no LLM] ~900 stocks → ~50          │                   │    │       │
│  │  │          │                           │                   │    │       │
│  │  │  Step 2: Scanner Ranking Agent       │                   │    │       │
│  │  │  [1 LLM call] 50 → top 10            │                   │    │       │
│  │  │          │                           │                   │    │       │
│  │  │  Step 3: Deep Analysis               │                   │    │       │
│  │  │  [fan-out, up to 20 parallel]        │                   │    │       │
│  │  │  ┌────────────┐  ┌────────────┐      │                   │    │       │
│  │  │  │ News Agent │  │ Research   │      │                   │    │       │
│  │  │  │ (per top-10│  │ Agent      │      │                   │    │       │
│  │  │  │ candidate) │  │ (per top-10│      │                   │    │       │
│  │  │  └────────────┘  └────────────┘      │                   │    │       │
│  │  └──────────────────────────────────────│───────────────────┘    │       │
│  │                                         │                        │       │
│  │  [both branches complete]               │                        │       │
│  │          └─────────────────────────────-┘                        │       │
│  │                            │                                     │       │
│  │                            ▼                                     │       │
│  │              ┌──────────────────────────┐                        │       │
│  │              │     Score Aggregator      │                        │       │
│  │              │  (weighted composite)     │                        │       │
│  │              └──────────────────────────┘                        │       │
│  │                            │                                     │       │
│  │                            ▼                                     │       │
│  │      ┌─────────────────────────────────────────┐                 │       │
│  │      │  Thesis Updater  │  Candidate Evaluator  │                │       │
│  │      │  (universe)      │  (rotate top-3 if     │                │       │
│  │      │                  │   needed, rule-based)  │                │       │
│  │      └─────────────────────────────────────────┘                 │       │
│  │                            │                                     │       │
│  │                            ▼                                     │       │
│  │              ┌──────────────────────────┐                        │       │
│  │              │   Consolidator Agent      │                        │       │
│  │              │  (~500 words)             │                        │       │
│  │              └──────────────────────────┘                        │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────┐                 │
│  │    Archive Write (updated reports + thesis)            │                 │
│  └───────────────────────────────────────────────────────┘                 │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────┐                 │
│  │       Email Delivery (AWS SES)                         │                 │
│  │       6:15am PT · NYSE trading days only              │                 │
│  └───────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Agent Design

### 4.1 News & Sentiment Agent

**Scope:** One instance per tracked stock (20 universe + 3 buy candidates = up to 23 parallel invocations).

**Inputs:**
- Archived prior news report for this ticker (from S3/db)
- Fresh news headlines + **full article body text** from Yahoo Finance RSS (last 24h) — not just headlines; first 500 chars of article body fetched where available
- SEC 8-K filings (last 24h) from EDGAR
- Stock's current price and recent price events (>3% moves)
- Deduplication hash list from prior run (to skip already-seen stories)

**Task:**
1. Load the archived report (prior context).
2. **Deduplicate incoming articles** against the prior run: skip any story whose headline+source hash was already processed. Only novel articles proceed to analysis. Track `mention_count` for recurring themes (e.g., tariff risk appearing across 5 articles = high salience).
3. Compare incoming headlines against what was already known — use body text excerpts to extract substance beyond the headline where relevant.
4. Add material new developments; discard information older than ~10 trading days or no longer relevant.
5. Assess net news sentiment: positive / negative / neutral, with brief rationale.
6. Output a refreshed ~300-word report covering:
   - Key recent headlines (last 24–48h)
   - Significant events still in the window (earnings, guidance, product launches, regulatory)
   - Net sentiment score: -1.0 to +1.0
   - News-driven contribution to attractiveness (0–100 sub-score)
   - `mention_count` for any recurring theme (≥3 articles) in the JSON output

**Output:** Refreshed markdown report + structured JSON sub-score (including `mention_count` for dominant themes).

---

### 4.2 Analyst Research Agent

**Scope:** One instance per tracked stock (same 23).

**Inputs:**
- Archived prior research report for this ticker
- Latest analyst consensus data: average rating (strong buy / buy / hold / underperform / sell), mean price target, number of analysts, recent rating changes (from Financial Modeling Prep or similar)
- Recent earnings surprise data (beat/miss magnitude)
- Any recent price target changes

**Task:**
1. Load the archived report.
2. Identify new analyst actions since the last run (upgrades, downgrades, target changes).
3. Remove stale rating actions (>30 trading days old).
4. Assess analyst consensus sentiment and price target gap to current price.
5. Output a refreshed ~300-word report covering:
   - Current consensus rating and mean price target
   - Implied upside/downside to current price
   - Notable recent analyst actions
   - Earnings surprise history (last 2 quarters)
   - Research-driven contribution to attractiveness (0–100 sub-score)

**Output:** Refreshed markdown report + structured JSON sub-score.

---

### 4.3 Macro & Market Environment Agent

**Scope:** Single global instance (not per-stock).

**Inputs:**
- Archived prior macro report
- Latest FRED data: fed funds rate, 2yr/10yr treasury yields, yield curve slope, VIX
- Unemployment rate, CPI (monthly updates via FRED)
- Commodity spot prices: crude oil (WTI), gold, copper (via yfinance tickers `CL=F`, `GC=F`, `HG=F`)
- Broad market levels: SPY, QQQ, IWM daily closes and 30-day performance

**Task:**
1. Load the archived report.
2. Identify material macro changes since the last run (rate decisions, yield moves >10bps, VIX spikes, significant commodity moves).
3. Produce a refreshed ~300-word report covering:
   - Current rate environment and trajectory
   - Yield curve status (inverted / normal / steepening)
   - VIX regime (low <15, moderate 15–25, elevated 25–35, panic >35)
   - Commodity signals and what they imply for sectors
   - Broad market trend and risk-on / risk-off assessment
   - VIX regime classification as `market_regime` string: `bull` | `neutral` | `caution` | `bear`
   - Macro tailwind / headwind rating per sector (see below)
4. Output **per-sector macro modifiers** (not a single global modifier): each sector gets its own multiplier based on how the current macro environment specifically affects it (e.g., rising rates hurt rate-sensitive sectors like Real Estate/Utilities more than Energy; high oil benefits Energy/Industrials). Sectors: Technology, Healthcare, Financial, Consumer Discretionary, Consumer Staples, Energy, Industrials, Materials, Real Estate, Utilities, Communication Services.

**Output:** Refreshed markdown report + structured JSON with `sector_modifiers` dict (11 sectors → modifier 0.70–1.30) and `market_regime` string. The scoring aggregator uses each stock's sector tag to look up its specific modifier rather than applying a single global number.

---

### 4.4 Consolidator Agent

**Scope:** Single instance per run, executes after all per-stock agents complete.

**Inputs:**
- All refreshed agent reports (news, research, macro)
- All updated investment theses for 20 universe stocks (buy/sell/hold + score)
- **Full news + research reports** for all three buy candidates (not just summaries) — so the consolidator can synthesize the complete picture for the most important picks
- Sector modifiers + `market_regime` from the Macro Agent
- Previous consolidated report (for context/continuity)

**Task:**
1. Read all inputs.
2. **Thesis/score consistency check** (automated pre-step, not LLM): before the LLM call, the aggregator flags any stock where the written thesis direction and the numeric score are inconsistent (e.g., thesis text is clearly bullish but score is 35 — or thesis is cautious but score is 85). Flag is passed to the consolidator as `consistency_flag: true` for the consolidator to note or reconcile in the report.
3. Produce a single consolidated email-ready report, max ~500 words, covering:
   - **Market environment snapshot** (2–3 sentences from macro report)
   - **Notable developments** (material news or research across the universe, top 3–5 items)
   - **Universe ratings summary** (table: ticker, rating, score, 24h change in score, staleness flag)
   - **Top 3 Buy Candidates** — for each: name, sector, score, short thesis (2–3 sentences), key upside catalyst, key risk. Because the full reports for buy candidates are available, the consolidator should synthesize a richer thesis for these three than it can for universe stocks.
   - **Any candidate rotation** note if one of the 3 was replaced this run
   - **Consistency flag notes** for any stock where thesis and score diverged significantly
4. Ensure the tone is concise, data-driven, and actionable.

**Output:** Formatted markdown/HTML email body.

---

## 5. Attractiveness Score Methodology

The attractiveness score (0–100) is a weighted composite. All sub-scores are independently calculated on a 0–100 scale before weighting.

```
Attractiveness Score =
    (Technical Score   × 0.40)
  + (News Score        × 0.30)
  + (Research Score    × 0.30)
  × Macro Modifier (0.70 – 1.30)

Final score clipped to [0, 100].
```

### 5.1 Technical Score (0–100)

Computed deterministically from price data — no LLM involved in this step.

| Signal | Weight | Scoring |
|---|---|---|
| RSI (14-day) | 25% | **Regime-adjusted:** in `bull` market regime (VIX<15, market uptrend), RSI overbought threshold raises to 80 (strong momentum is a signal, not a warning); in `bear`/`caution` regime, oversold threshold raises to 40 (oversold can signal further decline). Default (neutral): 0=overbought(>70), 100=oversold(<30), linear interpolation. `market_regime` from Macro Agent controls which thresholds are applied. |
| MACD signal cross | 20% | Bullish cross above zero line = 100, bearish = 0, no signal = 50 |
| Price vs. 50-day MA | 20% | >5% above = 80, at = 50, >5% below = 30 |
| Price vs. 200-day MA | 20% | Same scale as 50-day MA |
| Momentum (20-day return) | 15% | Mapped to 0–100 via percentile rank within S&P 500 universe |

### 5.2 News Score (0–100)

Produced by the News Agent using Claude. Maps sentiment + materiality:
- Net positive sentiment, material catalyst → 70–100
- Mixed/neutral sentiment → 40–60
- Net negative sentiment, material catalyst → 0–30
- No material news → defaults to prior score with slight decay toward 50

### 5.3 Research Score (0–100)

Produced by the Research Agent using Claude.

| Condition | Score range |
|---|---|
| Strong Buy consensus + >20% upside target | 80–100 |
| Buy consensus + 10–20% upside | 60–79 |
| Hold consensus / minimal upside | 40–59 |
| Underperform / Sell consensus | 0–39 |
| Recent positive estimate revisions | +5–10 bonus |
| Recent earnings beat | +5 bonus |

### 5.4 Macro Modifier (Per-Sector)

Per-sector multiplier applied after individual stock scores are calculated. Each stock has a `sector` tag in `universe.yaml`; the aggregator looks up the matching sector modifier from the Macro Agent's JSON output. Using a single global modifier ignores the reality that macro conditions affect sectors differently — rising rates are a headwind for Real Estate but a tailwind for Financials; high oil benefits Energy but hurts Consumer Discretionary.

**Representative modifier ranges by macro regime and sector:**

| Sector | Bull (VIX<15, cutting) | Neutral (VIX 15–20) | Caution (VIX 20–28) | Bear (VIX>28) |
|---|---|---|---|---|
| Technology | 1.15–1.25 | 1.00–1.10 | 0.85–0.95 | 0.70–0.80 |
| Healthcare | 1.05–1.15 | 1.00–1.05 | 0.95–1.00 | 0.90–0.95 |
| Financial | 1.10–1.20 | 1.00–1.10 | 0.85–0.95 | 0.70–0.85 |
| Consumer Discretionary | 1.10–1.20 | 0.95–1.05 | 0.80–0.90 | 0.70–0.80 |
| Consumer Staples | 0.95–1.05 | 1.00–1.05 | 1.00–1.10 | 1.05–1.15 |
| Energy | 1.00–1.15 | 1.00–1.10 | 0.95–1.05 | 0.85–0.95 |
| Industrials | 1.05–1.15 | 1.00–1.05 | 0.85–0.95 | 0.75–0.85 |
| Materials | 1.05–1.15 | 1.00–1.05 | 0.85–0.95 | 0.75–0.85 |
| Real Estate | 1.15–1.25 | 0.95–1.05 | 0.80–0.90 | 0.70–0.80 |
| Utilities | 1.05–1.15 | 1.00–1.05 | 1.00–1.10 | 1.00–1.10 |
| Communication Services | 1.10–1.20 | 1.00–1.10 | 0.85–0.95 | 0.75–0.85 |

The Macro Agent outputs exact modifier values (not ranges) for each sector based on current conditions. These are stored in `macro_snapshots.sector_modifiers` (JSON column) and in the State as `sector_modifiers: dict[str, float]`.

### 5.5 Buy/Sell/Hold Rating

Derived directly from the final attractiveness score:

| Score | Rating |
|---|---|
| 70–100 | **BUY** |
| 40–69 | **HOLD** |
| 0–39 | **SELL** |

---

### 5.6 Score Performance Feedback Loop

The system tracks whether high scores (BUY-rated stocks) actually outperform over a subsequent 10-trading-day and 30-trading-day window. This creates a ground-truth signal to validate and recalibrate the scoring model over time.

**Mechanism:**
- On each run, `performance_tracker.py` checks if any stocks scored ≥70 exactly 10 or 30 trading days ago.
- It fetches the current price vs. the price on the scoring date and computes realized return.
- Results are written to the `score_performance` table (see §7.2).
- After 60 trading days of data, a weekly summary shows accuracy rate: what % of BUY-scored stocks outperformed SPY over the 10d and 30d windows.

**Recalibration trigger:** If BUY accuracy falls below 55% over a trailing 60-day window, a recalibration flag is set in the state — the consolidator notes it in the email and the operator is prompted to review the scoring weights.

**Implementation:** `scoring/performance_tracker.py` — invoked at the start of each run, before any agents execute. Reads from `technical_scores` and `investment_thesis` for prior dates, fetches current prices via yfinance, writes to `score_performance`. No LLM involved.

---

### 5.7 Score Staleness Tracking

A score that hasn't materially changed in many days may reflect stale agent analysis rather than a stable investment picture. The system tracks the date of the last material change for each stock's score and flags scores that have not materially updated in ≥5 trading days.

**Mechanism:**
- `last_material_change_date` column in `investment_thesis` — updated when the score changes by ≥3 points OR the agent's `material_changes: true` flag is set.
- In the email, stocks with a staleness gap ≥5 trading days get a `⚠ stale` indicator next to their score in the ratings table (see §10.2).
- Staleness does **not** change the score itself — it signals that the next agent run should scrutinize this ticker more carefully for overlooked developments.

**Rationale:** A score of 68 that has been exactly 68 for 12 trading days may mean the stock is genuinely stable — or it may mean no new material data is being integrated. The flag makes this visible without forcing arbitrary score drift.

---

## 6. Market Scanner — Finding the 3 Buy Candidates

The scanner runs as **Branch B** of the LangGraph graph, in parallel with the universe agent pipeline. It has its own sequence of dedicated agents that funnel from ~900 stocks down to a scored, analyzed shortlist before any rotation decision is made. The scanner does **not** use a single LLM — it uses multiple agents at different stages for different purposes.

### 6.1 Candidate Universe

- Base universe: S&P 500 + S&P 400 (MidCap) = ~900 liquid US-listed stocks
- Source: `yfinance` for price data; static constituent list refreshed monthly from Wikipedia or a free index provider
- Exclusion: any ticker already in the 20-stock config universe

### 6.2 Scanner Pipeline — Four Stages

```
~900 US equities (S&P 500 + S&P 400, excluding universe)
        │
        ▼
Stage 1: Quant Filter [no LLM — pure math]
        RSI, MACD, MA50/200, volume, price floor
        ~900 → ~50 candidates
        │
        ▼
Stage 2: Data Enrichment [no LLM — data fetch]
        Fetch 5 headlines + analyst consensus for all 50
        │
        ▼
Stage 3: Scanner Ranking Agent [1 LLM call — Sonnet]
        Single prompt: all 50 candidates, their technical scores,
        headlines, and analyst data → ranked top 10 with 1-sentence
        rationale per stock
        │
        ▼
Stage 4: Deep Analysis [up to 20 LLM calls — Haiku, parallel fan-out]
        News Agent  (per top-10 candidate, no prior report)
        Research Agent (per top-10 candidate, no prior report)
        → produces full attractiveness scores for each
        │
        ▼
Stage 5: Candidate Evaluator [no LLM — rule-based logic]
        Compare top-10 scores vs. current 3 buy candidates
        Rotate if score delta ≥ 5 pts (max 1 rotation per run)
```

**Why a single ranking agent in Stage 3?** Comparing 50 stocks at once requires a model that can reason across the full set simultaneously — running 50 independent agents would produce 50 scores with no cross-stock context. The ranking agent sees all candidates together and can weigh them relative to each other before the expensive deep analysis in Stage 4. Only the top 10 get full treatment.

**Why separate Deep Analysis agents in Stage 4?** The ranking agent's output is a coarse signal. Stages 3–4 together provide the same depth of analysis as the universe agents, but bootstrapped fresh (no archived prior report). Once a candidate is promoted to an active buy pick, its reports are archived and future runs use the same incremental-update pattern as the universe.

**The Candidate Evaluator (Stage 5) is rule-based, not LLM-driven.** Rotation is a deterministic scoring comparison — using an LLM for a yes/no threshold decision adds cost and non-determinism with no quality benefit.

### 6.3 Scanner Stage Details

**Stage 1 — Quantitative Filter (no LLM)**

Filters to reduce ~900 → ~50 via **two parallel paths** that merge before Stage 2:

**Path A — Momentum Path (default)**
- Average daily volume (20-day) ≥ 500k shares
- Price ≥ $10
- Technical score ≥ 60 (using same scoring as §5.1)
- Not in a confirmed downtrend (price < 200-day MA by > 15%)

**Path B — Deep Value Path** (config-gated, `scanner.deep_value_path: true`)
- Average daily volume (20-day) ≥ 500k shares
- Price ≥ $10
- RSI < 35 (oversold) AND price < 200-day MA (technically weak but potentially bottoming)
- Analyst consensus rating ≥ Buy (high analyst conviction despite technical weakness)
- Up to 10 deep-value candidates admitted alongside the ~40 momentum candidates

**Merge:** Both path outputs are pooled. Deduplicate by ticker. Combined list proceeds to Stage 2 with a `path` field (`momentum` or `deep_value`) so the Stage 3 ranking agent knows which filter admitted each candidate. The ranking prompt explicitly instructs the agent to consider both paths on their own terms — deep value picks are not compared directly against momentum scores.

**Rationale for deep value path:** Momentum-only filtering will systematically miss early-stage recoveries where analyst conviction is high but the technical picture is still weak. Adding a small deep value cohort expands the candidate surface area without materially increasing Stage 4 cost.

**Stage 2 — Data Enrichment (no LLM)**

For all ~50 survivors:
- Last 5 news headlines via Yahoo Finance RSS
- Analyst consensus rating and mean price target via FMP

**Stage 3 — Scanner Ranking Agent (1 Sonnet call)**

Single prompt, ~6,000 input tokens, covering all 50 candidates. Outputs top 10 ranked list with a 1-sentence rationale per stock. This is the only strategic, cross-stock judgment call in the scanner pipeline.

**Stage 4 — Deep Analysis (up to 20 Haiku calls, parallel fan-out)**

News Agent and Research Agent invoked for each of the top 10 candidates. Same agent implementations as the universe pipeline, but `prior_report=None` since these stocks have no archive history. Output: full ~300-word report + sub-score per agent.

**Stage 5 — Candidate Evaluator (rule-based, tiered rotation)**

Rotation thresholds are tiered by how long a candidate has been held. This prevents whipsaw rotation of new entries while still enabling rapid replacement of persistently weak holdings.

| Tenure | Score-delta required to trigger rotation | Rationale |
|---|---|---|
| ≤3 trading days | ≥12 points | New entry — give it time to settle; avoid reversing on day 2 |
| 4–10 trading days | ≥8 points | Early tenure — still evaluating; require meaningful gap |
| 11–30 trading days | ≥5 points | Established — standard threshold |
| ≥31 trading days | ≥3 points | Long-held pick — low bar to rotate; scoring should be stable by now |

**Override conditions (apply regardless of tenure tier):**
- A candidate held ≥10 trading days with score consistently < 60 over 5 consecutive runs is eligible for replacement with any candidate scoring ≥65 (zero-margin rule — weak picks don't get tenure protection).
- If all 3 current candidates score < 55 and a new candidate scores ≥70, emergency rotation is permitted even if tenure is ≤3 days.

**At most 1 rotation per daily morning run** (prevents cascading swaps). All thresholds are configurable in `universe.yaml` under `scanner.rotation_tiers`.

---

## 7. Archive Strategy

The archive is the system's long-term memory. Every agent report and investment thesis is persisted after each run and loaded at the start of the next. The archive grows daily and is **never destructively overwritten** — only appended to and versioned. Prior records exist forever.

The guiding principle is **incremental surgical updates**: agents always start from prior context, add what is new and material, and remove what is stale. Fresh drafts from scratch are only produced on a ticker's absolute first-ever run.

### 7.1 Storage Layout (S3)

```
s3://alpha-engine-research/
├── archive/
│   ├── universe/
│   │   └── {TICKER}/
│   │       ├── thesis.json           # Current investment thesis (latest version)
│   │       ├── news_report.md        # Current news agent report
│   │       ├── research_report.md    # Current research agent report
│   │       └── history/
│   │           └── {YYYY-MM-DD}/     # Full dated snapshot — retained permanently
│   │               ├── thesis.json
│   │               ├── news_report.md
│   │               └── research_report.md
│   │
│   ├── candidates/
│   │   ├── active.json               # Current 3 slots: ticker + entry_date + slot#
│   │   └── {TICKER}/                 # Created for EVERY stock that has EVER been
│   │       ├── thesis.json           # a buy candidate — survives demotion
│   │       ├── news_report.md
│   │       ├── research_report.md
│   │       └── history/
│   │           └── {YYYY-MM-DD}/     # Full dated snapshot — retained permanently
│   │               ├── thesis.json
│   │               ├── news_report.md
│   │               └── research_report.md
│   │
│   └── macro/
│       ├── macro_report.md           # Current macro agent report
│       └── history/
│           └── {YYYY-MM-DD}/
│               └── macro_report.md
│
├── consolidated/
│   └── {YYYY-MM-DD}/
│       └── morning.md                # Final email body for that day
│
└── research.db                       # SQLite for all structured/queryable data
```

**Key design rule:** The `candidates/{TICKER}/` directory is created when a stock first enters the top-3. It is **never deleted** when the candidate is demoted. If the stock is later re-promoted, the system finds the existing directory, loads the prior thesis, and continues updating it — preserving full investment history across all tenure periods.

### 7.2 SQLite Schema (`research.db`)

```sql
-- Investment thesis state per stock per day
CREATE TABLE investment_thesis (
    id                       INTEGER PRIMARY KEY,
    symbol                   TEXT NOT NULL,
    date                     TEXT NOT NULL,           -- YYYY-MM-DD
    run_time                 TEXT NOT NULL,           -- ISO 8601 timestamp
    rating                   TEXT NOT NULL,           -- BUY / HOLD / SELL
    score                    REAL NOT NULL,           -- 0.0 – 100.0
    technical_score          REAL,
    news_score               REAL,
    research_score           REAL,
    macro_modifier           REAL,                    -- sector-specific modifier (from §5.4)
    thesis_summary           TEXT,                    -- 2–3 sentence thesis from agent
    prev_rating              TEXT,                    -- prior day rating (for change detection)
    prev_score               REAL,                    -- prior day score
    last_material_change_date TEXT,                  -- date of last ≥3pt score move or material_changes=true
    stale_days               INTEGER,                 -- trading days since last_material_change_date
    consistency_flag         INTEGER DEFAULT 0,       -- 1 if thesis text direction ≠ score direction
    UNIQUE(symbol, date, run_time)
);

-- Raw agent report content (linkable to thesis)
CREATE TABLE agent_reports (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT,                        -- NULL for macro (global)
    date        TEXT NOT NULL,
    run_time    TEXT NOT NULL,
    agent_type  TEXT NOT NULL,               -- news | research | macro | consolidator
    report_md   TEXT NOT NULL,               -- full markdown report text
    word_count  INTEGER,
    UNIQUE(symbol, date, run_time, agent_type)
);

-- Full lifecycle history of every buy candidate tenure
-- A ticker can appear multiple times if it exits and re-enters
CREATE TABLE candidate_tenures (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    slot            INTEGER NOT NULL,        -- 1, 2, or 3
    entry_date      TEXT NOT NULL,           -- date this tenure began
    exit_date       TEXT,                    -- date demoted (NULL = currently active)
    exit_reason     TEXT,                    -- "replaced_by_XXX" | "score_decline" | "tenure_limit"
    replaced_by     TEXT,                    -- ticker that took this slot (NULL if active)
    peak_score      REAL,                    -- highest attractiveness score during this tenure
    exit_score      REAL,                    -- score at time of demotion
    tenure_days     INTEGER                  -- computed on exit: exit_date - entry_date
);

-- Current active slots (always exactly 3 rows when system is running)
CREATE TABLE active_candidates (
    slot            INTEGER PRIMARY KEY,     -- 1, 2, or 3
    symbol          TEXT NOT NULL,
    entry_date      TEXT NOT NULL,
    prior_tenures   INTEGER NOT NULL DEFAULT 0  -- how many times this ticker was previously a candidate
);

-- Daily scanner appearances — tracks which stocks made top-10 each day
-- even if they weren't selected as one of the 3 active candidates
CREATE TABLE scanner_appearances (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    scanner_rank    INTEGER NOT NULL,        -- 1–10 (rank in that day's top-10)
    scan_path       TEXT,                    -- "momentum" | "deep_value" (which Stage 1 path admitted it)
    tech_score      REAL,
    news_score      REAL,
    research_score  REAL,
    final_score     REAL,
    selected        INTEGER NOT NULL DEFAULT 0,  -- 1 if promoted to active candidate
    selection_reason TEXT,                   -- why selected or not
    UNIQUE(symbol, date)
);

-- Technical scores (daily, computed deterministically)
CREATE TABLE technical_scores (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    rsi_14          REAL,
    macd_signal     REAL,                    -- cross direction: 1.0 / 0.0 / -1.0
    price_vs_ma50   REAL,                    -- pct diff
    price_vs_ma200  REAL,                    -- pct diff
    momentum_20d    REAL,                    -- 20-day return pct
    technical_score REAL,                    -- 0–100 composite
    UNIQUE(symbol, date)
);

-- Macro snapshots
CREATE TABLE macro_snapshots (
    id                  INTEGER PRIMARY KEY,
    date                TEXT NOT NULL UNIQUE,
    fed_funds_rate      REAL,
    treasury_2yr        REAL,
    treasury_10yr       REAL,
    yield_curve_slope   REAL,
    vix                 REAL,
    sp500_close         REAL,
    sp500_30d_return    REAL,
    oil_wti             REAL,
    gold                REAL,
    copper              REAL,
    market_regime       TEXT,                -- bull / neutral / caution / bear
    sector_modifiers    TEXT                 -- JSON: {"Technology": 1.15, "Healthcare": 1.05, ...}
);

-- Score performance tracking — realized return vs. SPY after BUY-scored periods
CREATE TABLE score_performance (
    id              INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    score_date      TEXT NOT NULL,           -- date the BUY score was assigned
    score           REAL NOT NULL,           -- score on score_date (should be ≥70 to appear here)
    price_on_date   REAL,                    -- closing price on score_date
    price_10d       REAL,                    -- closing price 10 trading days later
    price_30d       REAL,                    -- closing price 30 trading days later
    spy_10d_return  REAL,                    -- SPY return over same 10d window (benchmark)
    spy_30d_return  REAL,                    -- SPY return over same 30d window
    return_10d      REAL,                    -- (price_10d / price_on_date) - 1
    return_30d      REAL,                    -- (price_30d / price_on_date) - 1
    beat_spy_10d    INTEGER,                 -- 1 if return_10d > spy_10d_return, else 0
    beat_spy_30d    INTEGER,                 -- 1 if return_30d > spy_30d_return, else 0
    eval_date_10d   TEXT,                    -- actual date 10d measurement was taken
    eval_date_30d   TEXT,                    -- actual date 30d measurement was taken
    UNIQUE(symbol, score_date)
);

-- News article deduplication — stores hashes of processed articles to skip re-processing
CREATE TABLE news_article_hashes (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT NOT NULL,               -- NULL for scanner-wide articles
    article_hash TEXT NOT NULL,              -- SHA-256 of headline + source domain
    first_seen  TEXT NOT NULL,              -- date first processed
    mention_count INTEGER NOT NULL DEFAULT 1, -- how many runs this story has appeared in
    UNIQUE(symbol, article_hash)
);
```

### 7.3 Thesis Drafting Protocol (Mandatory for All Agents)

This protocol applies to every agent that produces a report or thesis — news agents, research agents, macro agent, and thesis updater — without exception.

```
STEP 1 — START FROM EXISTING
  Load the latest archived version of this ticker's report/thesis from S3.
  This is the baseline. Do not discard any existing finding unless you
  can specifically identify why it is now stale or incorrect.
  If NO prior record exists (first-ever run for this ticker), produce
  a fresh report from the available data and note "Initial report."

STEP 2 — ADD NEW MATERIAL FINDINGS
  Compare incoming data (new headlines, analyst actions, price moves,
  macro changes) against the archived baseline.
  Integrate only MATERIAL new developments — things that meaningfully
  change the investment picture. Skip minor noise.

STEP 3 — REMOVE STALE / OUTDATED CONTENT
  Prune the report of content that is no longer relevant:
    - News events that are resolved or have faded in significance (>10 trading days)
    - Analyst ratings that have since been superseded
    - Price targets or guidance that has been revised
    - Macro commentary about conditions that have since changed
  Do not remove something unless you are certain it is stale.
  When in doubt, retain it.
```

All agent prompts encode this protocol explicitly. The goal is compounding institutional knowledge — each run makes the thesis incrementally more accurate and more current, not merely different.

---

### 7.4 Buy Candidate Lifecycle Tracking

Buy candidates can be promoted, demoted, and re-promoted multiple times. The system tracks every phase of this lifecycle.

**Promotion:**
- A stock enters the top-3 when the candidate evaluator determines its score exceeds the weakest active candidate's score by ≥5 points.
- On promotion: create a new row in `candidate_tenures` (entry_date = today), update `active_candidates` slot, set `prior_tenures` = count of prior tenures for this ticker.
- Check if `candidates/{TICKER}/` already exists in S3. If it does, load the existing thesis/reports and pass them to the deep analysis agents as prior context. **The prior thesis is never discarded — it is the starting point for the new tenure.**

**Demotion:**
- A stock exits when a newer candidate's score exceeds it by ≥5 points (or tenure limit reached with score < 65).
- On demotion: set `exit_date`, `exit_score`, `exit_reason`, `replaced_by`, and `tenure_days` on the `candidate_tenures` row.
- The `candidates/{TICKER}/` S3 directory is **left intact**. No files are deleted. The dated history snapshot for today is written before demotion is finalized.
- Remove the ticker from `active_candidates`.

**Re-promotion:**
- When a previously-demoted ticker re-enters the scanner top-10 and is selected:
  - Look up `candidate_tenures` for prior records (`prior_tenures` count).
  - Load the latest `candidates/{TICKER}/thesis.json` and reports from S3 — the last known state from the prior tenure.
  - Pass these to the Stage 4 deep analysis agents as `prior_report` context, not `None`.
  - Start a new `candidate_tenures` row. The `prior_tenures` field on `active_candidates` is incremented.
  - The email report flags: `↩ RETURNED — previously held {N} tenure(s), last demoted {date}`.

**Scanner appearance tracking:**
- Every stock that appears in any day's top-10 is recorded in `scanner_appearances`, whether or not it is selected. This provides a historical ranking signal — a stock that repeatedly appears in the top-10 without being selected is building a case for eventual promotion.

---

### 7.5 Archive Update Protocol

**At the start of each run:**
1. Download `research.db` from S3.
2. Load `active_candidates` table to determine which 3 tickers to treat as candidates.
3. Load prior agent reports from S3 for all universe tickers + active candidates.
4. For any scanner top-10 candidates identified in Branch B, check `candidates/{TICKER}/` in S3 for prior history before running Stage 4 deep analysis.

**At the end of each run:**
1. Write updated reports and thesis to S3 for all universe tickers + active candidates.
2. Write dated history snapshot (`history/{YYYY-MM-DD}/`) for each ticker.
3. Update `research.db` (investment_thesis, agent_reports, candidate_tenures, scanner_appearances) and upload to S3.
4. Write consolidated report to `consolidated/{YYYY-MM-DD}/morning.md`.
5. Backup `research.db` to `s3://alpha-engine-research/backups/research_{YYYYMMDD}.db`.

---

## 8. Data Sources and APIs

| Data Type | Source | Notes |
|---|---|---|
| Price data (daily OHLCV) | `yfinance` | Free, no key required |
| Technical indicators | Computed from yfinance prices | RSI, MACD, MAs via pandas |
| News headlines | Yahoo Finance RSS (`feeds.finance.yahoo.com`) | Free, proven in alpha-engine |
| SEC filings (8-K) | EDGAR full-text search API | Free |
| Analyst consensus | Financial Modeling Prep (FMP) | Free tier: 250 requests/day; paid tier for scale |
| Macro data (rates, VIX) | FRED CSV API | Free, proven in alpha-engine |
| Commodity prices | yfinance (`CL=F`, `GC=F`, `HG=F`) | Free |
| Market index levels | yfinance (`SPY`, `QQQ`, `IWM`) | Free |
| S&P 500/400 constituents | Wikipedia scrape or static CSV | Refresh monthly |
| Email delivery | AWS SES | ~$0.10/1000 emails |
| LLM (all agents) | Claude API (claude-sonnet-4-6) | Anthropic API |

**FMP Free Tier Note:** 250 requests/day. Daily usage: 20 universe + 3 candidates + 50 scanner candidates = 73 calls/day — within the free tier. Paid tier ($14/month) is not needed for v1. See §15.4 for full analysis.

---

## 9. LangGraph Orchestration

### 9.1 State Schema

```python
class ResearchState(TypedDict):
    run_date: str                          # YYYY-MM-DD
    run_time: str                          # ISO 8601

    # Config
    universe_tickers: list[str]            # from config
    candidate_tickers: list[str]           # current 3 buy candidates

    # Fetched raw data (keyed by ticker)
    price_data: dict[str, pd.DataFrame]
    news_headlines: dict[str, list[dict]]
    analyst_data: dict[str, dict]
    macro_data: dict

    # Archive (prior context)
    prior_news_reports: dict[str, str]
    prior_research_reports: dict[str, str]
    prior_macro_report: str
    prior_theses: dict[str, dict]
    active_candidates: list[dict]
    news_article_hashes: dict[str, set]   # ticker → set of previously seen article hashes

    # Agent outputs
    technical_scores: dict[str, dict]
    news_reports: dict[str, str]
    research_reports: dict[str, str]
    macro_report: str
    sector_modifiers: dict[str, float]    # sector → modifier (from Macro Agent, replaces single macro_modifier)
    market_regime: str                    # bull | neutral | caution | bear (from Macro Agent)

    # Score performance
    performance_summary: dict             # accuracy stats from performance_tracker

    # Scored outputs
    investment_theses: dict[str, dict]    # ticker → {score, rating, summary, stale_days, consistency_flag}

    # Scanner outputs
    scanner_filtered: list[dict]          # ~50 tickers after quant filter (includes scan_path field)
    scanner_ranked: list[dict]            # top 10 from ranking agent (ticker + rationale)
    scanner_news_reports: dict[str, str]  # news agent output per top-10 candidate
    scanner_research_reports: dict[str, str]  # research agent output per top-10 candidate
    scanner_scores: dict[str, dict]       # full attractiveness scores for top 10
    new_candidates: list[dict]            # post-rotation final 3 (may be unchanged)

    # Final output
    consolidated_report: str
    email_sent: bool
```

### 9.2 Graph Topology

```
fetch_data (price, news, analyst, macro data for all stocks)
    │
    ├──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  BRANCH A — Universe Pipeline                                    │  BRANCH B — Scanner Pipeline
    │  (1x/day, NYSE trading days only)                                │  (same schedule)
    │                                                                  │
    ├──► [fan-out per ticker, max 10 concurrent]                       ├──► quant_filter
    │        ├── technical_score_engine (no LLM)                      │        [no LLM, ~900 → ~50]
    │        ├── news_agent (LLM, per universe ticker)                 │        │
    │        └── research_agent (LLM, per universe ticker)            │        ▼
    │                                                                  │    scanner_data_enrichment
    ├──► macro_agent (LLM, global)                                     │        [no LLM, fetch headlines]
    │                                                                  │        │
    │  [fan-out join — all per-ticker agents complete]                 │        ▼
    │                                                                  │    scanner_ranking_agent
    │                                                                  │        [1 LLM call, Sonnet]
    │                                                                  │        50 → top 10 ranked
    │                                                                  │        │
    │                                                                  │        ▼
    │                                                                  │    [fan-out, max 10 concurrent]
    │                                                                  │        ├── scanner_news_agent
    │                                                                  │        │   (LLM, per top-10 candidate)
    │                                                                  │        └── scanner_research_agent
    │                                                                  │            (LLM, per top-10 candidate)
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
            score_aggregator
            (weighted composite for all stocks: universe + active candidates + scanner top-10)
                            │
                            ▼
            thesis_updater            candidate_evaluator
            (per universe ticker      (compare top-10 scores vs.
             → buy/sell/hold          current 3 picks, rotate
             + 1-sentence summary)    if score delta ≥ 5 pts)
                            │
                            ▼
            consolidator_agent
            (reads all theses + candidate results + macro report
             → ≤500-word email body)
                            │
                            ▼
            archive_writer (S3 + research.db)
                            │
                            ▼
            email_sender (AWS SES)
                            │
                            ▼
            END
```

**Parallelism notes:**
- Both branches start simultaneously after `fetch_data` using LangGraph `Send` nodes.
- Within Branch A, news + research agents fan out per ticker (up to 23 × 2 = 46 LLM calls, gated by a semaphore at max 10 concurrent).
- Branch B's scanner stages are sequential within the branch (filter → enrich → rank → deep analysis) — you can't deep-analyze until ranking picks the top 10.
- Branch B Stage 4 checks the archive before running: if any top-10 ticker has prior `candidates/{TICKER}/` records in S3, those reports are loaded and passed as `prior_report` context.
- Branch B's Stage 4 (deep analysis) fans out within itself (up to 10 × 2 = 20 LLM calls, same concurrency semaphore shared with Branch A).
- `score_aggregator` waits for both branches to complete before proceeding (LangGraph fan-in join).
- Every run is a full run — there are no partial runs. One complete pipeline executes at 6:15am PT on each NYSE trading day.

---

## 10. Scheduling and Delivery

### 10.1 AWS EventBridge Schedule

Two EventBridge rules (Mon–Fri only), one active per DST season:

| Season | Active months | UTC Cron | PT offset |
|---|---|---|---|
| PDT | Mar second Sun → Nov first Sun | `cron(15 13 ? * MON-FRI *)` | UTC-7 |
| PST | Nov first Sun → Mar second Sun | `cron(15 14 ? * MON-FRI *)` | UTC-8 |

Only one rule is enabled at a time. Swap the enabled/disabled rule at each DST transition (twice a year). A note in the operational runbook will flag the swap dates.

**Market holiday handling:** EventBridge fires Mon–Fri regardless of market holidays. The Lambda handler is responsible for detecting NYSE holidays and exiting cleanly if today is a full-day market closure.

Implementation in `lambda/handler.py`:
```python
from exchange_calendars import get_calendar
import datetime, pytz

def is_trading_day() -> bool:
    nyse = get_calendar("XNYS")
    today = datetime.date.today()
    return nyse.is_session(today)

def handler(event, context):
    if not is_trading_day():
        print(f"Market holiday on {datetime.date.today()} — skipping run.")
        return {"status": "SKIPPED", "reason": "market_holiday"}
    # ... proceed with full pipeline
```

**NYSE full-day closures** (holidays that trigger a skip):
New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas Day — and their observed substitutes when they fall on a weekend.

**Early-close days** (day before July 4th, Black Friday, Christmas Eve) are **not skipped** — the market is open for a partial session and the morning report still runs as normal.

**`exchange_calendars` package:** Lightweight (~2 MB), already maintains an accurate NYSE holiday schedule, handles "observed" substitutions automatically. Add to `requirements.txt`.

### 10.2 Email Format

Delivered via AWS SES as a multipart email (plain text + HTML).

```
Subject: Research Brief — {YYYY-MM-DD} {DAY_OF_WEEK}
         e.g. "Research Brief — 2026-03-04 Wednesday"
         e.g. "Research Brief — 2026-07-05 Friday [Early Close]"  ← on partial-session days

Body Structure:
─────────────────────────────────────────
MARKET ENVIRONMENT
[2–3 sentences from macro agent]

NOTABLE DEVELOPMENTS
• [Item 1]
• [Item 2]
• [Item 3 ...]

UNIVERSE RATINGS — {N} stocks tracked
┌────────┬──────────┬───────┬────────┬────────┐
│ Ticker │  Rating  │ Score │ Δ Score│ Status │
├────────┼──────────┼───────┼────────┼────────┤
│ AAPL   │   HOLD   │  58   │  +2    │        │
│ NVDA   │   BUY    │  81   │  +5    │        │
│ TMO    │   HOLD   │  61   │   0    │ ⚠stale │  ← score unchanged ≥5 trading days
│ ...    │   ...    │  ...  │  ...   │        │
└────────┴──────────┴───────┴────────┴────────┘
⚠stale = score/thesis unchanged for ≥5 trading days; may warrant manual review

TOP 3 BUY CANDIDATES
━━━━━━━━━━━━━━━━━━━
1. [TICKER] — Score: [XX] | [SECTOR]
   [2–3 sentence thesis]
   Catalyst: [key upside driver]
   Risk: [key downside risk]
   ★ NEW ENTRY [if promoted this run]
   ↩ RETURNED — previously held {N} tenure(s), last demoted {date} [if re-promoted]

2. [TICKER] — Score: [XX] | [SECTOR]
   ...

3. [TICKER] — Score: [XX] | [SECTOR]
   ...

─────────────────────────────────────────
Generated by alpha-engine-research | {timestamp}
```

---

## 11. Infrastructure

### 11.1 AWS Resources

```
S3 Bucket:        s3://alpha-engine-research
Lambda Function:  alpha-engine-research-runner   (main daily pipeline)
Lambda Function:  alpha-engine-research-alerts    (intraday price alert — see §11.4)
EventBridge:      alpha-research-pdt (Mar–Nov, MON-FRI), alpha-research-pst (Nov–Mar, MON-FRI)
EventBridge:      alpha-research-alerts (every 30 min, MON-FRI 9:30am–4pm ET)
IAM Role:         alpha-engine-research-role
SES Identity:     [verified sender email]
AWS Region:       us-east-1
```

### 11.2 Lambda Configuration

| Parameter | Value |
|---|---|
| Runtime | Python 3.12 |
| Memory | 1024 MB |
| Timeout | 600 seconds (10 minutes) |
| Architecture | x86_64 |
| Layer | pandas, yfinance, anthropic SDK, langraph |

**Note on Lambda package size:** The scanner stage fetching ~900 tickers via yfinance may be slow in Lambda. If the full morning run approaches the 10-minute timeout, the scanner can be split into a separate Lambda invoked asynchronously from the main function.

### 11.3 IAM Permissions

The Lambda role needs:
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on `alpha-engine-research/*`
- `ses:SendEmail`, `ses:SendRawEmail`
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`

### 11.4 Intraday Price Alert Lambda

A lightweight second Lambda (`alpha-engine-research-alerts`) runs every 30 minutes during market hours (9:30am–4:00pm ET, Mon–Fri, NYSE trading days). It does **not** run the full pipeline — it only checks current prices for the 20 universe stocks + 3 active candidates and fires a concise email alert when a significant intraday move is detected.

**Trigger condition:** Intraday price move ≥ 5% from prior close (configurable in `universe.yaml` under `alerts.price_move_threshold_pct`).

**Alert content:**
```
⚡ PRICE ALERT — {TICKER}: {+/-X.X}% from prior close
Current: ${price} | Prior close: ${prior_close}
Current rating: {rating} | Score: {score}
[Last known news context — 1 sentence from most recent news report]
```

**Implementation:** `lambda/alerts_handler.py` — fetches current prices via yfinance for ~23 tickers, compares against prior close from `research.db` (downloaded from S3 at Lambda cold start), sends SES email if any ticker breaches threshold. No LLM, no agents. Execution time: ~15 seconds. Lambda memory: 256 MB, timeout: 60s.

**Cool-down:** Once an alert fires for a ticker, suppress further alerts for that ticker for 60 minutes (stored in a lightweight `alerts_fired` dict in Lambda memory; lost on cold start, which is acceptable). Prevents alert fatigue on volatile days.

**Cost:** Negligible — ~130 invocations/month × 15s × 256MB = 498 GB-sec/month, within the 400K free tier.

**Why separate Lambda?** The main pipeline runs once at 6:15am and takes ~8 minutes. An intraday alert check doesn't need any of that infrastructure — it only needs current prices, which takes seconds. Keeping it separate prevents any alert logic from bloating or delaying the morning run.

---

## 12. Project Structure

```
alpha-engine-research/
├── config/
│   └── universe.yaml              # Tracked tickers, weights, email recipients
├── agents/
│   ├── __init__.py
│   ├── news_agent.py              # News/sentiment agent (per ticker — reused by scanner)
│   ├── research_agent.py          # Analyst research agent (per ticker — reused by scanner)
│   ├── macro_agent.py             # Macro environment agent (global)
│   ├── scanner_ranking_agent.py   # Single-call cross-stock ranking (scanner Stage 3)
│   └── consolidator.py            # Final report consolidation agent
├── data/
│   ├── __init__.py
│   ├── fetchers/
│   │   ├── price_fetcher.py       # yfinance OHLCV + technical indicators
│   │   ├── news_fetcher.py        # Yahoo Finance RSS + SEC EDGAR (+ article body text fetch)
│   │   ├── analyst_fetcher.py     # FMP analyst consensus + price targets
│   │   └── macro_fetcher.py       # FRED + commodity prices via yfinance
│   ├── deduplicator.py            # News article hash-based deduplication + mention_count tracking
│   └── scanner.py                 # S&P 500/400 screener + top candidate selector (momentum + deep value paths)
├── scoring/
│   ├── __init__.py
│   ├── technical.py               # RSI, MACD, MA, momentum → 0–100 score (regime-aware RSI)
│   ├── aggregator.py              # Weighted composite score + per-sector macro modifier
│   └── performance_tracker.py     # Score feedback loop: realized return vs. SPY after BUY signals
├── thesis/
│   ├── __init__.py
│   └── updater.py                 # Reads scores → produces buy/sell/hold + summary
├── archive/
│   ├── __init__.py
│   ├── manager.py                 # S3 read/write, db read/write
│   └── models.py                  # Pydantic models for thesis, report, candidate
├── email/
│   ├── __init__.py
│   ├── formatter.py               # Markdown → plain text + HTML
│   └── sender.py                  # AWS SES delivery
├── graph/
│   ├── __init__.py
│   └── research_graph.py          # LangGraph state machine definition
├── lambda/
│   ├── handler.py                 # Lambda entry point — main morning pipeline
│   ├── alerts_handler.py          # Lambda entry point — intraday price alert (separate function)
│   └── package/                   # Dependencies bundled for Lambda
├── infrastructure/
│   ├── deploy.sh                  # Build + deploy Lambda
│   ├── trust-policy.json
│   └── iam-policy.json
├── local/
│   ├── run.py                     # Local test runner (bypasses Lambda)
│   └── sync_db.py                 # Pull/push research.db to/from S3
├── tests/
│   ├── test_scoring.py
│   ├── test_agents.py
│   ├── test_scanner.py
│   ├── test_archive.py
│   └── test_graph.py
├── config.py                      # Python constants (mirrors universe.yaml)
├── main.py                        # Local entry point
├── requirements.txt
└── README.md
```

### 12.1 `config/universe.yaml`

```yaml
universe:
  - ticker: ASML
    sector: Technology
  - ticker: RACE
    sector: Consumer Discretionary
  - ticker: CME
    sector: Financial
  - ticker: TSM
    sector: Technology
  - ticker: COST
    sector: Consumer Staples
  - ticker: LLY
    sector: Healthcare
  - ticker: WM
    sector: Industrials
  - ticker: PLTR
    sector: Technology
  - ticker: VST
    sector: Utilities
  - ticker: JPM
    sector: Financial
  - ticker: RKLB
    sector: Industrials
  - ticker: MELI
    sector: Consumer Discretionary
  - ticker: AVGO
    sector: Technology
  - ticker: TMO
    sector: Healthcare
  - ticker: CEG
    sector: Utilities
  - ticker: AON
    sector: Financial
  - ticker: GEV
    sector: Industrials
  - ticker: EQIX
    sector: Real Estate
  - ticker: IBKR
    sector: Financial
  - ticker: BX
    sector: Financial

scoring_weights:
  technical: 0.40
  news: 0.30
  research: 0.30

rating_thresholds:
  buy: 70
  sell: 40

scanner:
  candidate_count: 3
  candidate_universe: sp500_sp400    # sp500 | sp500_sp400 | russell1000
  min_avg_volume: 500000
  min_price: 10

  # Tiered rotation thresholds — see §6.3 Stage 5
  rotation_tiers:
    - max_tenure_days: 3
      min_score_diff: 12             # new entry: require large gap to prevent whipsaw
    - max_tenure_days: 10
      min_score_diff: 8              # early tenure
    - max_tenure_days: 30
      min_score_diff: 5              # established
    - max_tenure_days: 99999
      min_score_diff: 3              # long-held: low bar to rotate

  # Override: weak long-held picks lose tenure protection
  weak_pick_score_threshold: 60      # score below which tenure override applies
  weak_pick_consecutive_runs: 5      # must score below threshold for this many consecutive runs
  emergency_rotation_new_score: 70   # new candidate must score ≥ this for emergency override

  # Deep value path — second quant filter path (RSI-oversold + analyst conviction)
  deep_value_path: true
  deep_value_max_rsi: 35
  deep_value_min_consensus: Buy      # FMP consensus string
  deep_value_max_candidates: 10      # cap on how many deep value picks enter Stage 2

archive:
  retain_history: true               # never delete dated history snapshots
  check_prior_candidate_records: true  # always load prior thesis for re-promoted candidates

alerts:
  enabled: true
  price_move_threshold_pct: 5.0      # % intraday move from prior close to trigger alert
  cooldown_minutes: 60               # suppress repeat alerts per ticker

email:
  recipients:
    - your@email.com
  sender: research@yourdomain.com    # must be SES-verified
  send_time_pt: "06:15"

schedule:
  timezone: America/Los_Angeles
  run_time: "06:15"
  frequency: trading_days            # Mon-Fri, NYSE holidays excluded
  holiday_calendar: XNYS             # exchange_calendars identifier for NYSE
  run_on_early_close: true           # partial-session days still run

llm:
  per_stock_model: claude-haiku-4-5  # news + research agents (universe, candidates, scanner deep)
  strategic_model: claude-sonnet-4-6 # scanner ranking, macro, consolidator
  max_tokens_per_stock: 600
  max_tokens_strategic: 1024
  concurrent_agents: 10              # semaphore limit on simultaneous LLM calls
```

---

## 13. Prompt Design

### 13.1 News Agent Prompt Template

```
You are a financial news analyst maintaining an ongoing intelligence
brief for {ticker} ({company_name}).

PRIOR REPORT (from {prior_date}):
{prior_report}
[If "NONE — initial report", produce a fresh brief from the data below.]

NEW ARTICLES (last 24–48 hours — novel only, duplicates pre-filtered):
{new_articles}
← Format per article: HEADLINE | SOURCE | ARTICLE_EXCERPT (first 400 chars of body)
   Note: articles already seen in the prior run have been deduplicated before reaching you.

RECURRING THEMES (appeared in ≥3 articles this run):
{recurring_themes}
← Format: THEME | mention_count | example_headline
   High mention_count = high market salience; weight accordingly.

NEW SEC FILINGS (last 24–48 hours):
{new_filings}

CURRENT PRICE: ${price} | RECENT MOVE: {price_change_pct}% ({price_change_date})

THESIS DRAFTING PROTOCOL — FOLLOW IN ORDER:
1. START WITH EXISTING: The prior report above is your baseline. Preserve
   every finding that remains valid. Do not rewrite what hasn't changed.
2. ADD NEW FINDINGS: Integrate material new articles or filings. Use the
   article body excerpt (not just the headline) to assess substance.
   Weight recurring themes by their mention_count — a theme appearing in
   10 articles matters more than one appearing in 1.
3. REMOVE STALE CONTENT: Remove events that are resolved, superseded, or
   older than 10 trading days and no longer relevant. When in doubt, retain.

Keep the report to approximately 300 words.
End with a JSON block:
{"news_score": <0-100>, "sentiment": "<positive|neutral|negative>",
 "key_catalyst": "<one sentence>", "prior_date": "{prior_date}",
 "material_changes": <true|false>,
 "dominant_theme": "<recurring theme if any, else null>",
 "dominant_theme_count": <mention_count or 0>}

Output the full refreshed report followed by the JSON block.
```

### 13.2 Research Agent Prompt Template

```
You are a sell-side research analyst maintaining an ongoing analyst
consensus brief for {ticker} ({company_name}).

PRIOR REPORT (from {prior_date}):
{prior_report}
[If "NONE — initial report", produce a fresh brief from the data below.]

CURRENT ANALYST DATA:
- Consensus rating: {consensus_rating} ({num_analysts} analysts)
- Mean price target: ${mean_target} | Current price: ${current_price}
- Implied upside/downside: {upside_pct}%
- Recent rating changes (last 30 days): {rating_changes}
- Last earnings: {earnings_surprise}

THESIS DRAFTING PROTOCOL — FOLLOW IN ORDER:
1. START WITH EXISTING: The prior report above is your baseline.
   Preserve findings that remain accurate.
2. ADD NEW FINDINGS: Integrate any new analyst actions (upgrades,
   downgrades, target changes, initiation of coverage) since the prior
   report. Note if the consensus direction has shifted.
3. REMOVE STALE CONTENT: Remove rating actions older than 30 trading
   days that have been superseded by newer actions. When in doubt, retain.

Keep the report to approximately 300 words.
End with a JSON block:
{"research_score": <0-100>, "consensus_direction": "<bullish|neutral|bearish>",
 "key_upside": "<one sentence>", "key_risk": "<one sentence>",
 "material_changes": <true|false>}

Output the full refreshed report followed by the JSON block.
```

### 13.3 Macro Agent Prompt Template

```
You are a macro economist and market strategist maintaining an ongoing
macro environment brief for US equity investors.

PRIOR REPORT (from {prior_date}):
{prior_report}
[If "NONE — initial report", produce a fresh brief from the data below.]

CURRENT MACRO DATA:
- Fed Funds Rate: {fed_funds}%
- 2yr Yield: {t2yr}% | 10yr Yield: {t10yr}% | Curve: {curve_slope}bps
- VIX: {vix}
- SPY 30d return: {spy_30d}% | QQQ 30d: {qqq_30d}% | IWM 30d: {iwm_30d}%
- WTI Oil: ${oil}/bbl | Gold: ${gold}/oz | Copper: ${copper}/lb
- Latest CPI: {cpi_yoy}% YoY | Unemployment: {unemployment}%
- Next scheduled releases: {upcoming_releases}

THESIS DRAFTING PROTOCOL — FOLLOW IN ORDER:
1. START WITH EXISTING: The prior report is your baseline. Preserve
   the macro thesis and supporting evidence that remains valid.
2. ADD NEW FINDINGS: Note any material changes in rate expectations,
   yield moves >10bps, VIX regime shifts, significant commodity moves,
   or new economic data releases since the prior report.
3. REMOVE STALE CONTENT: Remove commentary about conditions that have
   since changed (e.g., prior VIX spike that has resolved, rate
   expectations that have been revised). When in doubt, retain.

Keep the report to approximately 300 words.
End with a JSON block:
{
  "market_regime": "<bull|neutral|caution|bear>",
  "key_theme": "<one sentence macro thesis>",
  "material_changes": <true|false>,
  "sector_modifiers": {
    "Technology": <0.70-1.30>,
    "Healthcare": <0.70-1.30>,
    "Financial": <0.70-1.30>,
    "Consumer Discretionary": <0.70-1.30>,
    "Consumer Staples": <0.70-1.30>,
    "Energy": <0.70-1.30>,
    "Industrials": <0.70-1.30>,
    "Materials": <0.70-1.30>,
    "Real Estate": <0.70-1.30>,
    "Utilities": <0.70-1.30>,
    "Communication Services": <0.70-1.30>
  }
}

Guidance for sector_modifiers:
- Assign each sector a modifier from 0.70 (strong macro headwind for that sector) to 1.30
  (strong macro tailwind) based on current rates, VIX, yield curve, and commodity prices.
- Rising rates are a headwind for Real Estate and Utilities, a moderate tailwind for Financials.
- High VIX is broadly negative but less severe for Consumer Staples and Healthcare (defensive).
- High oil benefits Energy; hurts Consumer Discretionary and Industrials (input cost).
- Rate-cutting cycle with low VIX benefits rate-sensitive growth sectors (Technology, Real Estate).
- Modifiers should vary across sectors — avoid assigning the same value to all sectors.
- market_regime drives RSI threshold selection in the technical scoring engine (see §5.1).

Output the full refreshed report followed by the JSON block.
```

### 13.4 Scanner Ranking Agent Prompt Template

```
You are an equity research analyst. Your job is to identify the most
attractive near-term buy opportunities from a screened candidate list.

CANDIDATE LIST (passed quantitative filter):
{candidates_table}
← Format: TICKER | SECTOR | PATH | TECH_SCORE | ANALYST_RATING | UPSIDE% | TOP_HEADLINE_1 | TOP_HEADLINE_2
   PATH = "momentum" (strong technical trend) or "deep_value" (oversold + analyst conviction)
   Sorted by tech_score descending within each path. Both paths are present.

CURRENT MACRO REGIME: {market_regime}
← bull | neutral | caution | bear — use to weight momentum vs. value candidates appropriately.
   In bear/caution regimes, deep_value picks with strong analyst backing may be more attractive
   than high-RSI momentum picks. In bull regimes, momentum typically deserves more weight.

Instructions:
1. Review all {n} candidates holistically across both paths.
2. Do NOT evaluate momentum and deep_value candidates by the same technical score standard —
   deep_value picks are admitted specifically because they are technically weak but
   fundamentally supported. Evaluate deep_value candidates on analyst conviction and upside.
3. Rank the top 10 by investment attractiveness across both paths, weighing technical
   momentum, analyst conviction, upside to price target, news catalyst, and macro regime.
4. For each of your top 10, provide:
   - Rank (1–10)
   - Ticker
   - Path (momentum or deep_value)
   - 1-sentence rationale covering why it stands out vs. the field

Output as JSON array:
[{"rank": 1, "ticker": "XXX", "path": "momentum", "rationale": "..."}, ...]
```

### 13.5 Consolidator Prompt Template

```
You are a portfolio research director. Synthesize the following agent
reports into a concise daily research brief for an investor.

MACRO REPORT:
{macro_report}

NEWS REPORTS — UNIVERSE STOCKS (summary only):
{news_reports_by_ticker}
← 1–2 sentence extract per ticker; full reports omitted for space

RESEARCH REPORTS — UNIVERSE STOCKS (summary only):
{research_reports_by_ticker}
← 1–2 sentence extract per ticker

BUY CANDIDATE FULL REPORTS — read these carefully, these are the priority:
{candidate_full_news_reports}
← Full ~300-word news report per buy candidate (not a summary)
{candidate_full_research_reports}
← Full ~300-word research report per buy candidate (not a summary)

UNIVERSE INVESTMENT THESES:
{thesis_table}    ← ticker | rating | score | score_delta | stale_flag | consistency_flag | 1-sentence thesis

TOP 3 BUY CANDIDATES:
{candidates}
← ticker | score | score_delta | 1-sentence thesis | catalyst | risk
   | status: CONTINUING | NEW_ENTRY | RETURNED(N tenures, last demoted DATE)

CONSISTENCY FLAGS (pre-computed):
{consistency_flags}
← List of tickers where thesis direction and score are inconsistent.
   E.g.: "PLTR: thesis is clearly bullish but score is 34 — verify scoring inputs"

Instructions:
1. Write a consolidated research brief, maximum 500 words.
2. Structure: Market Environment → Notable Developments → Universe Ratings Table
   → Top 3 Buy Candidates (with short thesis each).
3. For the Top 3 Buy Candidates, synthesize the FULL reports (not summaries) —
   provide a richer 3–4 sentence thesis per candidate than you do for universe stocks.
4. Flag any candidate that was newly promoted this run (NEW_ENTRY).
5. Flag any candidate that is a re-promotion (RETURNED) with the number of prior
   tenures and last demotion date — this history adds context.
6. If any consistency_flags exist, note them briefly: "Note: {ticker} thesis/score
   inconsistency flagged — review recommended."
7. Add ⚠stale after scores in the ratings table that have stale_flag = true.
8. Be concise, data-driven, and actionable. No filler language.
9. If today is an early-close trading day, note briefly that the market
   closes at 1pm ET and intraday volatility may be elevated on thin volume.

Output the brief in clean markdown, suitable for email.
```

---

## 14. Implementation Phases

### Phase 1 — Foundation (Week 1–2)

- [ ] Initialize GitHub repo `alpha-engine-research`
- [ ] Set up Python project structure, venv, `requirements.txt`
- [ ] Implement `config/universe.yaml` (with sector tags) and `config.py` reader
- [ ] Implement `data/fetchers/`: price_fetcher, news_fetcher (+ body text), macro_fetcher
- [ ] Implement `data/deduplicator.py` — article hash tracking + mention_count
- [ ] Implement `scoring/technical.py` — RSI, MACD, MA scores (with regime-aware RSI hooks)
- [ ] Set up SQLite schema (`research.db` — all tables including `score_performance`, `news_article_hashes`) and S3 bucket
- [ ] Implement `archive/manager.py` — S3 read/write + db CRUD
- [ ] Implement `scoring/performance_tracker.py` — scaffold (no data yet, but schema and logic ready)
- [ ] Write tests for all fetchers, scoring, and deduplicator

### Phase 2 — Agents (Week 3–4)

- [ ] Implement `agents/news_agent.py` with prior-report-aware prompt
- [ ] Implement `agents/research_agent.py` — add FMP API integration
- [ ] Implement `agents/macro_agent.py` — add commodity data
- [ ] Implement `scoring/aggregator.py` — weighted composite + macro modifier
- [ ] Implement `thesis/updater.py` — score → buy/sell/hold + summary
- [ ] Implement `agents/consolidator.py`
- [ ] Run agents locally against live data, validate report quality

### Phase 3 — Scanner (Week 5)

- [ ] Build S&P 500/400 constituent list loader (static monthly CSV)
- [ ] Implement `data/scanner.py` — Stage 1 quantitative filter
- [ ] Add analyst data fetch for scanner candidates (FMP)
- [ ] Implement Stage 3 LLM ranking prompt
- [ ] Implement candidate rotation logic (score delta, tenure check)
- [ ] Validate scanner output end-to-end locally

### Phase 4 — LangGraph + Email (Week 6)

- [ ] Implement `graph/research_graph.py` — full LangGraph topology
- [ ] Implement `email/formatter.py` — markdown → HTML
- [ ] Set up AWS SES, verify sender identity
- [ ] Implement `email/sender.py`
- [ ] Run full graph locally, validate email output

### Phase 5 — Lambda + Scheduling (Week 7)

- [ ] Implement `lambda/handler.py` with DST-aware PT time check
- [ ] Implement `lambda/alerts_handler.py` — intraday price alert Lambda (23 tickers, yfinance, SES)
- [ ] Build `infrastructure/deploy.sh` (deploy both Lambda functions)
- [ ] Deploy main Lambda, set up two EventBridge rules (PDT + PST) for morning pipeline
- [ ] Deploy alerts Lambda, set up EventBridge rule (every 30 min, 9:30–16:00 ET, MON-FRI)
- [ ] Activate the correct DST rule for current season
- [ ] Run a manual invocation, validate full pipeline end-to-end
- [ ] Validate archive write/read cycle: confirm dated snapshots are written
- [ ] Simulate candidate rotation over 3 consecutive runs (verify tiered rotation thresholds)
- [ ] Simulate re-promotion: manually add a past candidate record and confirm it is loaded
- [ ] Test intraday alert: manually set a ticker's prior_close to trigger a ≥5% alert

### Phase 6 — Hardening (Week 8)

- [ ] Add retry logic on Anthropic API rate limit errors
- [ ] Add fallback for FMP outages (graceful degradation to last-known data)
- [ ] Add Lambda timeout guard (scanner can be skipped if <90s remaining)
- [ ] Set up CloudWatch alarms for Lambda errors and duration
- [ ] Finalize README with operational runbook

---

## 15. Cost Estimate (Monthly)

### 15.1 Token Budget Per Run

Token counts are per single full daily run. "Input" includes system prompt + prior archived report + fetched data. "Output" is the generated report + JSON. Branch B scanner inputs are slightly smaller for re-promoted candidates (prior report exists) vs. first-time candidates (no prior report — shorter prompt).

| Agent | Calls/run | Model | Input tokens | Output tokens | Notes |
|---|---|---|---|---|---|
| News Agent (universe) | 20 | Haiku | ~1,100 | ~450 | Prior report + headlines + SEC + prompt |
| Research Agent (universe) | 20 | Haiku | ~900 | ~450 | Prior report + analyst data + prompt |
| Active Candidate News Agent | 3 | Haiku | ~1,100 | ~450 | Same as universe |
| Active Candidate Research Agent | 3 | Haiku | ~900 | ~450 | Same as universe |
| Macro Agent | 1 | Sonnet | ~900 | ~450 | Prior report + FRED + commodities + prompt |
| **Branch A subtotal** | **47** | | **~50,600** | **~21,150** | |
| Scanner Ranking Agent | 1 | Sonnet | ~6,500 | ~800 | 50 candidates × ~120 tok each + prompt |
| Scanner News Agent (top-10) | 10 | Haiku | ~500 | ~450 | Prior report if re-promoted, else shorter fresh prompt |
| Scanner Research Agent (top-10) | 10 | Haiku | ~400 | ~450 | Same |
| **Branch B subtotal** | **21** | | **~14,500** | **~9,800** | |
| Consolidator | 1 | Sonnet | ~4,800 | ~900 | Full buy candidate reports (3 × ~600 tok each) + thesis table + prompt |
| **Daily run total** | **69** | | **~69,900** | **~31,850** | |

---

### 15.2 Selected Configuration ✓

**Tier: Hybrid (Haiku per-stock, Sonnet strategic) · 1×/day · Trading days only (Mon–Fri, NYSE holidays excluded)**

This is the selected approach for v1. Cost is minimized during the development and iteration phase. Individual agents can be upgraded to Sonnet selectively once the product is near-final.

**Model assignments:**

| Agent | Model | Rationale |
|---|---|---|
| News Agent (all tickers) | `claude-haiku-4-5` | Templated incremental update — Haiku is well-suited |
| Research Agent (all tickers) | `claude-haiku-4-5` | Same — reads prior context and integrates new data |
| Scanner News Agent (top-10) | `claude-haiku-4-5` | Same agent implementation, same rationale |
| Scanner Research Agent (top-10) | `claude-haiku-4-5` | Same |
| Scanner Ranking Agent | `claude-sonnet-4-6` | Cross-stock comparative judgment — requires Sonnet |
| Macro Agent | `claude-sonnet-4-6` | Economic interpretation and nuanced regime assessment |
| Consolidator | `claude-sonnet-4-6` | Synthesis across all reports — highest visibility output |

**Monthly schedule:** ~22 trading days/month (52 weeks × 5 days − ~10 NYSE holidays ÷ 12 months)

**Daily token breakdown:**
- Haiku agents (news + research, universe + candidates + scanner deep):
  - 20 universe news: 20 × 1,100 = 22,000 input · 20 × 450 = 9,000 output
  - 20 universe research: 20 × 900 = 18,000 input · 20 × 450 = 9,000 output
  - 3 candidate news: 3 × 1,100 = 3,300 input · 3 × 450 = 1,350 output
  - 3 candidate research: 3 × 900 = 2,700 input · 3 × 450 = 1,350 output
  - 10 scanner news: 10 × 500 = 5,000 input · 10 × 450 = 4,500 output
  - 10 scanner research: 10 × 400 = 4,000 input · 10 × 450 = 4,500 output
  - **Haiku daily: 55,000 input · 29,700 output**
- Sonnet agents (macro + ranking + consolidator):
  - Macro: 900 input · 450 output
  - Scanner Ranking: 6,500 input · 800 output
  - Consolidator: 4,800 input · 900 output (full buy candidate reports included)
  - **Sonnet daily: 12,200 input · 2,150 output**

**Monthly token totals (× 22 trading days):**

| Model | Input/day | Input/month | Output/day | Output/month |
|---|---|---|---|---|
| Haiku | 55,000 | **1.21M** | 29,700 | **653K** |
| Sonnet | 12,200 | **268K** | 2,150 | **47K** |

**Monthly LLM cost:**

| Model | Input tokens | Input cost | Output tokens | Output cost | Subtotal |
|---|---|---|---|---|---|
| Haiku (`claude-haiku-4-5`) | 1.21M | $0.97 | 653K | $2.61 | $3.58 |
| Sonnet (`claude-sonnet-4-6`) | 268K | $0.80 | 47K | $0.71 | $1.51 |
| **LLM total** | | | | | **$5.09** |

**Full monthly cost:**

| Resource | Cost/month | Notes |
|---|---|---|
| Claude API (hybrid) | $5.09 | Per above |
| AWS Lambda | $0 | 22 inv × 600s × 1GB = 13.2K GB-sec, within 400K free tier |
| AWS S3 | $0.50 | ~500MB storage, daily PUTs for reports + db backup |
| AWS SES | $0 | 22 emails/mo, within 3,000/mo free tier from Lambda |
| FMP API | $0 | 73 calls/day, within 250/day free tier (see §15.4) |
| **Total** | **~$5.70/month** | |

---

### 15.3 Upgrade Path

As the product matures, individual agents can be upgraded selectively. Recommended upgrade order (highest impact per dollar):

| Upgrade | Additional cost/month | When to consider |
|---|---|---|
| News Agent → Sonnet | +$4 | If Haiku news reports miss material context or feel thin |
| Research Agent → Sonnet | +$3 | If analyst consensus synthesis is inconsistent |
| Consolidator → Opus | +$5 | When final email quality is the top priority |
| Add midday refresh (news + consolidator only) | +$2.50 | When intraday news responsiveness matters |
| Add afternoon refresh | +$2.50 | Same |

**All-Sonnet equivalent at current schedule (22 trading days/month):**
- Haiku volume on Sonnet: 1.21M × $3/M + 653K × $15/M = $3.63 + $9.80 = $13.43
- Plus existing Sonnet agents: $1.33
- **All-Sonnet LLM: ~$14.76/month → total ~$15/month**

---

### 15.4 FMP API — Free Tier Analysis

FMP's free tier allows 250 requests/day. Daily call count:
- Universe (20) + active candidates (3) = 23 calls
- Scanner Stage 2 enrichment (50 candidates) = 50 calls
- **Total: 73 calls/day — within the 250/day free tier**

The paid tier ($14/month) is not required unless the scanner candidate pool grows beyond ~170 stocks. The free tier is sufficient for v1.

---

### 15.5 Additional Cost Reduction Options

| Option | Monthly savings | Tradeoff |
|---|---|---|
| Reduce scanner deep analysis from top 10 → top 5 | ~$0.80 (Tier 2) | Slightly narrower candidate evaluation |
| Cache macro agent: skip if FRED data unchanged (<1bp) | ~$0.20 | Negligible — macro rarely changes intraday |
| Skip research agents on midday/afternoon (news-only refresh) | ~$1.20 (Tier 2) | Analyst data doesn't change intraday anyway |
| Reduce prior report context from full ~300 words → 150-word summary | ~15% input token savings | Agents have less context; may reduce report continuity |
| Switch Consolidator to Haiku | ~$0.80 (Tier 2) | Consolidated report quality may decrease |

---

## 16. Open Questions / Decisions Required

1. **Email sender domain** — AWS SES requires a verified sender email/domain. Confirm which address to use before Phase 4. SES sandbox mode must be exited to send to non-verified recipients.

2. **Scanner universe size** — Starting with S&P 500 only (~480 stocks, excluding universe) is safest for Lambda timing. S&P 400 adds ~400 more stocks and ~30s to the quant filter stage. Recommend validating timing locally before expanding to S&P 500 + S&P 400.

3. **DST cron management** — Two EventBridge rules (PDT + PST) need to be manually toggled twice a year. Alternative: a single UTC cron + DST logic in the Lambda handler. The manual swap is simpler but error-prone; Lambda-side DST handling is more robust. Decision before Phase 5.

4. **Candidate rotation threshold tuning** — The 5-point score differential and 10-day tenure limit are starting assumptions. Recommend reviewing the `candidate_tenures` and `scanner_appearances` tables after 2 weeks of live data to assess if the thresholds need adjustment.

5. **Re-promotion archive lookup** — When a top-10 scanner candidate has `candidates/{TICKER}/` in S3, the system loads its prior report. Edge case: the prior report may be months old and contain very stale content. Agents should detect this via the `prior_date` field and apply extra aggressive staleness pruning when the gap is >30 trading days.

---

## 17. Relationship to `alpha-engine`

`alpha-engine-research` is intentionally decoupled from `alpha-engine`. It does not share a codebase, database, or AWS resources. The research output (investment theses, attractiveness scores) is designed to be human-consumable via email but can optionally be surfaced as a signal input to `alpha-engine`'s LLM analyst in a future integration phase — e.g., by writing scored theses to a shared S3 location that `alpha-engine` reads at run time.

---

## 18. Design Decision Rationale

This section documents the "why" behind major architectural choices. Intended for the project README once the repo is set up, and as a reference for future contributors navigating non-obvious decisions.

---

### 18.1 Archive-First, Never-Overwrite

**Decision:** Every agent always starts from its prior archived report. No report is ever generated from scratch if a prior version exists. Every prior version is retained permanently in a dated `history/` folder.

**Why:** LLM-generated research compounds over time. A report that has integrated 30 days of incremental findings is more accurate than a fresh report generated from today's data alone. Starting fresh each day also produces drift — the model may interpret the same fact differently depending on other context in the prompt. The archive ensures continuity: a finding from two weeks ago is still present unless the agent explicitly decides it is stale. The cost of storing prior reports is negligible (S3 at ~$0.50/month). The cost of losing compounded institutional knowledge is not recoverable.

---

### 18.2 Per-Sector Macro Modifier (vs. Single Global)

**Decision:** The Macro Agent outputs 11 sector-specific modifiers rather than one global multiplier.

**Why:** A single global modifier treats a rate-hike environment identically for Utilities (hurt by rising rates — bond-proxy) and Financials (helped by rising rates — net interest margin). This produces systematically wrong scores. Sector-specific modifiers require the Macro Agent to reason about the transmission mechanism of macro conditions to each sector — which it can do well at Sonnet quality. The additional prompt complexity (one JSON dict vs. one float) is minimal. This improvement was identified as the second-highest impact architectural change because scoring accuracy affects every stock in the universe every day.

---

### 18.3 Hybrid LLM Model Strategy (Haiku + Sonnet)

**Decision:** Per-stock agents (news, research) use `claude-haiku-4-5`. Strategic agents (macro, scanner ranking, consolidator) use `claude-sonnet-4-6`.

**Why:** Per-stock agents perform a structured, templated task: load prior report, integrate new data, remove stale data, output score. This is a well-defined incremental update — Haiku handles it reliably at a fraction of the cost. Strategic agents perform tasks requiring broader judgment: reasoning across 50 candidates simultaneously (scanner ranking), interpreting complex macro signals (macro agent), or synthesizing 23 reports into a coherent email (consolidator). These tasks benefit from Sonnet's stronger reasoning. The cost savings are significant (~$10/month Sonnet-only → ~$5/month hybrid at identical schedule). The quality difference on per-stock tasks is minimal.

---

### 18.4 Scanner Multi-Stage Architecture (Not a Single LLM)

**Decision:** The scanner is a 5-stage pipeline — quant filter, data enrichment, single ranking call, deep analysis fan-out, rule-based evaluator — rather than a single LLM processing 900 stocks.

**Why:** A single LLM call over 900 stocks would exceed context limits and produce poor results — the model cannot reason well over hundreds of rows in one prompt. The staged approach mirrors how a human analyst actually works: screen mechanically first, then evaluate the shortlist qualitatively. The quant filter (Stage 1) costs nothing and reduces 900→50 without LLM. The ranking agent (Stage 3) sees 50 candidates simultaneously — critical for cross-stock comparison — using one Sonnet call. Only the top 10 get the expensive deep analysis (Stage 4). Stage 5 is rule-based because "is X's score > Y's score by Z points?" is a math operation, not a judgment call.

---

### 18.5 Single Ranking Agent for Cross-Stock Comparison

**Decision:** Stage 3 uses one LLM call over all 50 candidates rather than 50 independent per-stock calls.

**Why:** Independent per-stock calls produce 50 scores with no cross-stock context — the model scores each stock in isolation, with no information about what it's competing against. The ranking agent sees all candidates together and can make relative judgments: "stock A has stronger momentum but stock B has a clearer catalyst." This is the same reason human analysts rank stocks rather than scoring them in isolation. One Sonnet call over 50 candidates (~6,500 input tokens) costs ~$0.02 and produces a ranked top-10 — far more efficient than 50 Haiku calls that produce incomparable independent scores.

---

### 18.6 Rule-Based Candidate Rotation (Not LLM)

**Decision:** The Candidate Evaluator (Stage 5) is deterministic rule-based logic, not an LLM.

**Why:** The rotation decision — "does stock X's score exceed stock Y's score by ≥N points?" — is a math inequality. Using an LLM for this introduces non-determinism (the model might say "yes" or "no" differently based on irrelevant context) and cost, with no quality benefit. LLMs add value for tasks requiring judgment under ambiguity; threshold comparisons are not such tasks. The tiered rotation rules (different score-delta thresholds by tenure) add complexity that would be harder to enforce reliably in a prompt than in code.

---

### 18.7 Tiered Rotation Thresholds

**Decision:** Rotation score-delta requirements vary by tenure (≤3 days: 12pt gap, 4–10 days: 8pt, 11–30 days: 5pt, ≥31 days: 3pt).

**Why:** A flat 5-point threshold causes whipsaw rotation — a stock promoted two days ago can immediately be replaced by a competitor that edges it by 5 points, then re-promoted the following day. This produces an unstable, rapidly cycling candidate list. The tiered thresholds protect new entries from immediate replacement (requiring a large gap to override) while applying a low bar to long-held picks (where the scoring model has had many days to stabilize and a small consistent gap is meaningful). The override conditions (weak pick protection) prevent tenure from becoming a shield for genuinely deteriorating candidates.

---

### 18.8 SQLite/S3 Over a Remote Database

**Decision:** Use SQLite synced to S3 rather than a managed database (Postgres, Snowflake, Supabase, etc.).

**Why:** The system writes ~64 rows/day and stores ~25MB/year. There is exactly one writer (the Lambda function) running once per day. No concurrent access, no real-time queries, no external dashboard consumers. A managed remote database introduces connection management, VPC/security groups, always-on cost, and connector package overhead (100MB+ for some drivers, pushing into Lambda package size limits). SQLite + S3 is zero-cost, zero-ops, and perfectly matched to a single-writer batch workload. Upgrade path when needed: Supabase (free PostgreSQL, pgvector, remote queryability) — not Snowflake (OLAP data warehouse, wrong fit, no individual tier).

---

### 18.9 3-Step Thesis Drafting Protocol (Mandatory for All Agents)

**Decision:** Every agent that produces a report must follow exactly three steps: (1) start from existing, (2) add new, (3) remove stale. This protocol is explicit in every agent's prompt.

**Why:** Without a protocol, LLMs tend to either: (a) ignore the prior report and generate fresh content, losing accumulated findings, or (b) heavily re-use prior content verbatim without assessing staleness, causing the report to drift out of date. The three-step protocol creates a specific cognitive sequence: assess what's still valid first, then assess what's new, then assess what to remove. Step 3 (remove stale) is the hardest step for LLMs to do unprompted — they tend to add rather than prune. Making it an explicit step improves pruning discipline. The instruction "when in doubt, retain" is intentional: false negatives (keeping slightly stale content) are less harmful than false positives (removing valid analysis).

---

### 18.10 Score Performance Feedback Loop

**Decision:** Track whether BUY-scored stocks (score ≥70) actually outperform SPY over 10-day and 30-day windows and surface the accuracy rate.

**Why:** Without feedback, the scoring model is an open loop — there's no signal about whether high scores actually predict outperformance. The feedback loop turns the system from "research generation" to "research validation." After 60 trading days, the accuracy rate provides an empirical basis to recalibrate weights (e.g., if technical score is systematically leading to false positives in a bear market, reduce its weight). The recalibration flag (triggered if accuracy falls below 55%) ensures this data surfaces to the operator rather than being silently buried in a table.

---

### 18.11 Regime-Aware RSI Thresholds

**Decision:** RSI overbought/oversold thresholds are adjusted based on `market_regime` from the Macro Agent.

**Why:** RSI overbought/oversold levels carry different implications in different market regimes. In a bull market, RSI can sustain levels above 70 for weeks without triggering a reversal — treating RSI=72 as a strong sell signal in a bull market produces false negatives (missed upside). In a bear market, RSI can fall to 35 and keep falling — treating RSI=32 as a strong buy signal in a bear market produces false positives (catching falling knives). Regime-aware thresholds let the technical score reflect the reality that momentum signals mean different things in different macro environments. The `market_regime` field from the Macro Agent is the control signal — connecting the macro view to the technical scoring engine directly.

---

### 18.12 News Article Body Text (vs. Headlines Only)

**Decision:** The news fetcher retrieves up to 400 characters of article body text for each headline, not just the headline string.

**Why:** Financial news headlines are often written for clicks, not accuracy. "Company X Misses Earnings" might mean a $0.01 miss on $5 EPS, or a catastrophic miss. The body text provides immediate context. "Company X Beats Expectations for Third Consecutive Quarter" is more actionable when paired with "beating estimates by 7% and raising full-year guidance" from the first paragraph. The additional data volume per article is minimal (~100 extra tokens), the scraping cost is negligible, and the improvement in agent analysis quality is meaningful for high-conviction decisions. The 400-character limit is deliberate — enough to capture the lead paragraph, not enough to bloat the prompt.

---

### 18.13 News Article Deduplication

**Decision:** Articles already processed in a prior run are skipped; recurring themes across multiple articles are tracked via `mention_count`.

**Why:** Yahoo Finance RSS regularly re-serves the same articles across multiple days. Without deduplication, an agent sees the same story on day 1, day 2, and day 3 — and may treat each appearance as a new development, artificially elevating the news score and producing thesis drift. Deduplication ensures agents only process genuinely new information. The `mention_count` field inverts this: a story that appears across 10 different articles on the same day is not "one story" — it reflects broad market consensus on a theme, which is itself a signal. Tracking this separately from deduplication lets the agent weight high-salience themes appropriately.

---

### 18.14 Deep Value Scanner Path

**Decision:** The scanner quant filter has two parallel paths: momentum (strong technicals) and deep value (oversold + analyst conviction), with up to 10 deep value candidates admitted alongside ~40 momentum candidates.

**Why:** A momentum-only filter systematically misses early-stage recoveries — stocks that are still technically weak but where institutional analyst consensus has already turned bullish (often a leading indicator). The deep value path captures these. Limiting it to 10 candidates (vs. ~40 for momentum) reflects the higher false-positive rate of oversold screens — being oversold is a necessary but not sufficient condition for attractiveness. The analyst conviction requirement (consensus ≥ Buy) provides the positive signal that filters out stocks that are oversold for good reason. The ranking agent is aware of which path admitted each candidate and evaluates them on different criteria.

---

### 18.15 Intraday Price Alert Lambda (Separate from Main Pipeline)

**Decision:** A separate Lambda function checks for ≥5% intraday moves every 30 minutes during market hours, completely independent of the morning pipeline.

**Why:** The morning pipeline runs once at 6:15am and takes ~8 minutes. A stock can move 8% between 2pm and 3pm — after the morning report has been delivered and forgotten. Intraday alerts ensure material price events surface promptly without requiring the operator to monitor prices manually. The separation from the main pipeline is important: the alert check is trivially fast (price fetch + threshold comparison, no LLM), and mixing it into the morning pipeline would require either (a) blocking the morning pipeline until market close, which defeats its purpose, or (b) running the full pipeline intraday, which adds significant cost and complexity. A standalone Lambda with its own EventBridge schedule is the clean solution.

---

### 18.16 NYSE Calendar-Aware Scheduling

**Decision:** EventBridge fires Mon–Fri regardless of holidays; the Lambda handler checks `exchange_calendars` to skip non-trading days rather than encoding holidays in EventBridge crons.

**Why:** NYSE holidays are not aligned to a simple cron pattern — they include floating holidays (MLK Day = third Monday of January, Thanksgiving = fourth Thursday of November, etc.), "observed" substitutions (Christmas on Saturday → Friday off), and ad-hoc closures. Encoding these in EventBridge cron expressions is error-prone and requires manual updates every year. The `exchange_calendars` Python library maintains an accurate NYSE holiday schedule and handles observed-day logic automatically. The Lambda-side check is two lines of code and requires no infrastructure changes when the holiday schedule changes. The tradeoff is an occasional "wasted" Lambda cold start on holidays — negligible cost, and the handler exits immediately when the check fails.

---

_End of design document. Version 1.4_
