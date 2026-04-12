"""
Research Graph — Sector-Team Architecture with LangGraph Send() fan-out.

Topology:
  fetch_data
  → dispatch_all        (Send: 6 sector teams + macro + exit evaluator — all parallel)
  → merge_results       (fan-in: team picks + macro + exits → compute open slots)
  → score_aggregator    (composite scores for team recommendations)
  → cio_node            (single Sonnet batch: evaluate all picks, select top N)
  → population_entry_handler
  → consolidator_node
  → archive_writer
  → email_sender_node
  → END

Multi-agent patterns:
  - 6 sector teams: quant (ReAct) → qual (ReAct) → peer review → 2-3 recommendations
  - Macro economist: regime + sector ratings with reflection loop
  - CIO: batch evaluation on 4 dimensions (team conviction, macro alignment, portfolio fit, catalyst specificity)
  - Thesis maintenance: triggered by material events only
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from config import (
    POPULATION_CFG,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
)
from agents.sector_teams.team_config import (
    ALL_TEAM_IDS,
    TEAM_SECTORS,
    SECTOR_TEAM_MAP,
    compute_team_slots,
    get_team_tickers,
)
from agents.sector_teams.sector_team import run_sector_team, SectorTeamContext
from agents.macro_agent import run_macro_agent_with_reflection
from agents.investment_committee.ic_cio import run_cio
from data.population_selector import (
    compute_exits_and_open_slots,
    apply_ic_entries,
)
from scoring.composite import compute_composite_score, normalize_conviction, score_to_rating
from archive.manager import ArchiveManager

logger = logging.getLogger(__name__)


# ── State Schema ──────────────────────────────────────────────────────────────

def _take_last(_left: Any, right: Any) -> Any:
    return right


def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Merge dicts from Send() fan-out — each team writes its team_id key."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    # ── Core run info ────────────────────────────────────────────────────────
    run_date: Annotated[str, _take_last]
    run_time: Annotated[str, _take_last]
    archive_manager: Annotated[Any, _take_last]
    is_early_close: Annotated[bool, _take_last]

    # ── Data (loaded in fetch_data) ──────────────────────────────────────────
    price_data: Annotated[dict[str, Any], _take_last]
    technical_scores: Annotated[dict[str, dict], _take_last]
    scanner_universe: Annotated[list[str], _take_last]
    sector_map: Annotated[dict[str, str], _take_last]
    macro_data: Annotated[dict, _take_last]
    current_population: Annotated[list[dict], _take_last]
    population_tickers: Annotated[list[str], _take_last]
    prior_theses: Annotated[dict[str, dict], _take_last]
    prior_sector_ratings: Annotated[dict[str, dict], _take_last]
    predictions: Annotated[dict[str, dict], _take_last]
    news_data_by_ticker: Annotated[dict[str, dict], _take_last]
    analyst_data_by_ticker: Annotated[dict[str, dict], _take_last]
    insider_data_by_ticker: Annotated[dict[str, dict], _take_last]

    # ── Prior context (memory) ─────────────────────────────────────────────
    prior_macro_report: Annotated[str, _take_last]
    prior_macro_snapshots: Annotated[list[dict], _take_last]
    episodic_memories: Annotated[dict[str, list], _take_last]
    semantic_memories: Annotated[dict[str, list], _take_last]

    # ── Sector team outputs (merged via Send) ────────────────────────────────
    sector_team_outputs: Annotated[dict[str, Any], _merge_dicts]

    # ── Macro output ─────────────────────────────────────────────────────────
    macro_report: Annotated[str, _take_last]
    sector_modifiers: Annotated[dict[str, float], _take_last]
    sector_ratings: Annotated[dict[str, dict], _take_last]
    market_regime: Annotated[str, _take_last]

    # ── Exit evaluator output ────────────────────────────────────────────────
    remaining_population: Annotated[list[dict], _take_last]
    exits: Annotated[list[dict], _take_last]
    open_slots: Annotated[int, _take_last]

    # ── CIO output ───────────────────────────────────────────────────────────
    ic_decisions: Annotated[list[dict], _take_last]
    advanced_tickers: Annotated[list[str], _take_last]
    entry_theses: Annotated[dict[str, dict], _take_last]

    # ── Final population ─────────────────────────────────────────────────────
    new_population: Annotated[list[dict], _take_last]
    population_rotation_events: Annotated[list[dict], _take_last]

    # ── Email ────────────────────────────────────────────────────────────────
    consolidated_report: Annotated[str, _take_last]
    email_sent: Annotated[bool, _take_last]

    # ── Team slot allocation ─────────────────────────────────────────────────
    team_slot_allocation: Annotated[dict[str, int], _take_last]

    # ── Investment theses (combined team + IC) ───────────────────────────────
    investment_theses: Annotated[dict[str, dict], _take_last]

    # ── Dispatch metadata (for Send()) ───────────────────────────────────────
    team_id: Annotated[str, _take_last]  # set by Send() per team


# ── Node Functions ────────────────────────────────────────────────────────────

def fetch_data(state: ResearchState) -> dict:
    """Load all shared data needed by sector teams, macro, and exit evaluator."""
    from data.fetchers.price_fetcher import (
        fetch_price_data, fetch_sp500_sp400_with_sectors, compute_technical_indicators,
    )
    from data.fetchers.macro_fetcher import fetch_macro_data, compute_market_breadth
    from scoring.technical import compute_technical_score

    run_date = state["run_date"]
    am: ArchiveManager = state["archive_manager"]

    logger.info("[fetch_data] starting for %s", run_date)

    # RAG availability check (early, so we know before agents start)
    rag_available = False
    try:
        from rag.db import is_available as _rag_is_available
        rag_available = _rag_is_available()
        logger.info("[fetch_data] RAG database: %s", "available" if rag_available else "UNAVAILABLE")
        # Reset per-run RAG stats
        from agents.sector_teams.qual_tools import reset_rag_stats
        reset_rag_stats()
    except Exception as e:
        logger.warning("[fetch_data] RAG availability check failed: %s", e)

    # Load S&P 900 universe
    scanner_universe, wikipedia_sector_map = fetch_sp500_sp400_with_sectors()
    logger.info("[fetch_data] %d tickers in S&P 900 universe", len(scanner_universe))

    # Load current population
    current_population = am.load_population()
    population_tickers = [p["ticker"] for p in current_population]

    # Build sector map
    sector_map = dict(wikipedia_sector_map)
    for p in current_population:
        sector_map.setdefault(p["ticker"], p.get("sector", "Unknown"))

    all_tickers = list(set(population_tickers + scanner_universe))

    # ── Feature store first: load pre-computed features for ~900 tickers ─────
    # The predictor's daily inference writes technical + interaction features for
    # the full universe. This eliminates the 3-month yfinance bulk fetch for all
    # tickers that have feature store coverage.
    technical_scores = {}
    _fs_features = {}
    _fs_enriched = 0
    try:
        from data.fetchers.feature_store_reader import read_latest_features
        _fs_features = read_latest_features() or {}
        if _fs_features:
            for ticker, fs_row in _fs_features.items():
                indicators = {
                    "rsi_14": fs_row.get("rsi_14", 50.0),
                    "macd_cross": fs_row.get("macd_cross", 0.0),
                    "macd_above_zero": bool(fs_row.get("macd_above_zero", False)),
                    "macd_line_last": fs_row.get("macd_line_last", 0.0),
                    "signal_line_last": 0.0,
                    "current_price": 0.0,  # filled below from daily_closes or yfinance
                    "ma50": None,
                    "ma200": None,
                    "price_vs_ma50": fs_row.get("price_vs_ma50"),
                    "price_vs_ma200": fs_row.get("price_vs_ma200"),
                    "momentum_20d": fs_row.get("momentum_20d"),
                    "momentum_5d": fs_row.get("momentum_5d"),
                    "avg_volume_20d": fs_row.get("avg_volume_20d"),
                    "atr_14_pct": fs_row.get("atr_14_pct"),
                    "dist_from_52w_high": fs_row.get("dist_from_52w_high"),
                    "dist_from_52w_low": fs_row.get("dist_from_52w_low"),
                }
                ts = compute_technical_score(indicators)
                technical_scores[ticker] = {**indicators, "technical_score": ts}
                _fs_enriched += 1
            logger.info("[fetch_data] feature store: %d tickers loaded (skipping yfinance for these)", _fs_enriched)
    except Exception as e:
        logger.debug("[fetch_data] feature store not available: %s", e)

    # ── Load current prices for feature store tickers from daily_closes ──────
    # daily_closes is a single parquet per trading day (~100KB), much cheaper than
    # ~900 yfinance batch calls. Provides current_price for scanner liquidity filter.
    _price_filled = 0
    try:
        from data.fetchers.feature_store_reader import read_latest_daily_closes
        daily_closes = read_latest_daily_closes()
        if daily_closes:
            for ticker in technical_scores:
                if ticker in daily_closes:
                    technical_scores[ticker]["current_price"] = daily_closes[ticker]
                    _price_filled += 1
            logger.info("[fetch_data] daily_closes filled current_price for %d tickers", _price_filled)
    except Exception as e:
        logger.debug("[fetch_data] daily_closes not available: %s", e)

    # ── yfinance fallback: only fetch tickers NOT in feature store ───────────
    # Population tickers always fetched (agents need raw OHLCV for deep analysis).
    # Scanner universe tickers covered by feature store are SKIPPED.
    _fs_covered = set(_fs_features.keys()) if _fs_features else set()
    yf_tickers = [t for t in all_tickers if t not in _fs_covered or t in population_tickers]
    if len(yf_tickers) < len(all_tickers):
        logger.info(
            "[fetch_data] yfinance: fetching %d tickers (skipped %d from feature store)",
            len(yf_tickers), len(all_tickers) - len(yf_tickers),
        )
    price_data = fetch_price_data(yf_tickers, period="3mo") if yf_tickers else {}

    # Fill current_price from yfinance for tickers that had feature store data
    # but no daily_closes price
    for ticker in technical_scores:
        if technical_scores[ticker]["current_price"] == 0.0:
            if ticker in price_data and price_data[ticker] is not None and not price_data[ticker].empty:
                technical_scores[ticker]["current_price"] = float(price_data[ticker]["Close"].iloc[-1])

    # Fill in technical scores for tickers not covered by feature store
    for ticker, df in price_data.items():
        if ticker in technical_scores:
            continue
        if df is not None and len(df) >= 20:
            indicators = compute_technical_indicators(df)
            ts = compute_technical_score(indicators)
            technical_scores[ticker] = {**indicators, "technical_score": ts}

    # ── Macro data ───────────────────────────────────────────────────────────
    macro_data = fetch_macro_data()

    # Market breadth — compute from feature store if available (avoids needing
    # price_data for all ~900 tickers), fall back to price_data computation.
    if _fs_enriched >= 200:
        # Feature store has enough coverage for breadth computation
        above_50d, total_50d = 0, 0
        above_200d, total_200d = 0, 0
        advancers, decliners = 0, 0
        for ticker, ts in technical_scores.items():
            pv50 = ts.get("price_vs_ma50")
            pv200 = ts.get("price_vs_ma200")
            mom5d = ts.get("momentum_5d")
            if pv50 is not None:
                total_50d += 1
                if pv50 > 0:
                    above_50d += 1
            if pv200 is not None:
                total_200d += 1
                if pv200 > 0:
                    above_200d += 1
            if mom5d is not None:
                if mom5d > 0:
                    advancers += 1
                elif mom5d < 0:
                    decliners += 1
        breadth = {
            "pct_above_50d_ma": round(above_50d / total_50d * 100, 1) if total_50d > 0 else None,
            "pct_above_200d_ma": round(above_200d / total_200d * 100, 1) if total_200d > 0 else None,
            "advance_decline_ratio": round(advancers / max(decliners, 1), 2),
            "n_stocks": max(total_50d, total_200d),
        }
        logger.info("[fetch_data] breadth from feature store: %s", breadth)
    else:
        breadth = compute_market_breadth(price_data)
    macro_data.update(breadth)

    # Load prior theses from SQLite (most recent entry per population ticker)
    prior_theses = am.load_latest_theses(population_tickers)

    prior_sector_ratings = state.get("prior_sector_ratings", {})

    # Load prior macro report from S3
    prior_macro_report = ""
    try:
        prior_macro_data = am.load_prior_reports("macro_global", category="macro")
        prior_macro_report = prior_macro_data.get("news_report") or ""
        if prior_macro_report:
            logger.info("[fetch_data] loaded prior macro report (%d chars)", len(prior_macro_report))
    except Exception as e:
        logger.warning("[fetch_data] failed to load prior macro report: %s", e)

    # Load last 3 macro snapshots for structured context
    prior_macro_snapshots = []
    try:
        rows = am.db_conn.execute(
            "SELECT date, market_regime, vix, treasury_10yr, yield_curve_slope, "
            "sp500_30d_return, sector_modifiers, sector_ratings "
            "FROM macro_snapshots ORDER BY date DESC LIMIT 3"
        ).fetchall()
        for r in rows:
            prior_macro_snapshots.append({
                "date": r[0], "market_regime": r[1], "vix": r[2],
                "treasury_10yr": r[3], "yield_curve_slope": r[4],
                "sp500_30d_return": r[5], "sector_modifiers": r[6],
                "sector_ratings": r[7],
            })
    except Exception as e:
        logger.warning("[fetch_data] failed to load macro snapshots: %s", e)

    # Load episodic memories (Phase 2: lessons from failed signals)
    episodic_memories = {}
    try:
        all_sectors = list(set(sector_map.values()))
        episodic_memories = am.load_episodic_memories(
            tickers=population_tickers + scanner_universe[:50],
            sectors=all_sectors,
        )
        if episodic_memories:
            logger.info("[fetch_data] loaded episodic memories for %d tickers", len(episodic_memories))
    except Exception as e:
        logger.debug("[fetch_data] episodic memories not available: %s", e)

    # Load semantic memories (Phase 3: cross-agent observations)
    semantic_memories = {}
    try:
        all_sectors = list(set(sector_map.values()))
        semantic_memories = am.load_semantic_memories(sectors=all_sectors)
        if semantic_memories:
            logger.info("[fetch_data] loaded semantic memories for %d sectors", len(semantic_memories))
    except Exception as e:
        logger.debug("[fetch_data] semantic memories not available: %s", e)

    # Load predictions
    predictions = {}
    try:
        pred_json = am.load_predictions_json()
        if pred_json:
            predictions = pred_json.get("predictions", {})
    except Exception as e:
        logger.debug("[fetch_data] predictions not available: %s", e)

    # Pre-fetch news/analyst/insider data for held population (for material triggers)
    news_data_by_ticker = {}
    analyst_data_by_ticker = {}
    insider_data_by_ticker = {}

    # Note: full data fetching happens within sector team agents via tools.
    # Here we only pre-fetch lightweight data for material trigger checks.
    from data.fetchers.news_fetcher import fetch_all_news
    for ticker in population_tickers:
        try:
            articles = fetch_all_news([ticker])
            news_data_by_ticker[ticker] = {
                "articles": articles.get(ticker, []),
                "article_count": len(articles.get(ticker, [])),
            }
        except Exception as e:
            logger.debug("[fetch_data] news fetch failed for %s: %s", ticker, e)

    logger.info("[fetch_data] done — %d prices, %d tech scores, %d population",
                len(price_data), len(technical_scores), len(population_tickers))

    return {
        "scanner_universe": scanner_universe,
        "sector_map": sector_map,
        "price_data": price_data,
        "technical_scores": technical_scores,
        "macro_data": macro_data,
        "current_population": current_population,
        "population_tickers": population_tickers,
        "prior_theses": prior_theses,
        "prior_sector_ratings": prior_sector_ratings,
        "predictions": predictions,
        "news_data_by_ticker": news_data_by_ticker,
        "analyst_data_by_ticker": analyst_data_by_ticker,
        "insider_data_by_ticker": insider_data_by_ticker,
        "prior_macro_report": prior_macro_report,
        "prior_macro_snapshots": prior_macro_snapshots,
        "episodic_memories": episodic_memories,
        "semantic_memories": semantic_memories,
    }


def dispatch_all(state: ResearchState) -> list:
    """
    Fan-out via Send(): launch 6 sector teams + macro + exit evaluator in parallel.
    Each receives a subset of shared state.
    """
    sends = []

    # 6 sector teams
    for team_id in ALL_TEAM_IDS:
        sends.append(Send("sector_team_node", {
            **state,
            "team_id": team_id,
        }))

    # Macro economist
    sends.append(Send("macro_economist_node", {
        **state,
    }))

    # Exit evaluator
    sends.append(Send("exit_evaluator_node", {
        **state,
    }))

    logger.info("[dispatch] sending %d parallel tasks (6 teams + macro + exits)", len(sends))
    return sends


def sector_team_node(state: ResearchState) -> dict:
    """Run a single sector team (dispatched via Send)."""
    team_id = state.get("team_id", "unknown")
    logger.info("[sector_team:%s] starting", team_id)

    ctx = SectorTeamContext(
        scanner_universe=state.get("scanner_universe", []),
        sector_map=state.get("sector_map", {}),
        price_data=state.get("price_data", {}),
        technical_scores=state.get("technical_scores", {}),
        market_regime=state.get("market_regime", "neutral"),
        prior_theses=state.get("prior_theses", {}),
        held_tickers=state.get("population_tickers", []),
        news_data_by_ticker=state.get("news_data_by_ticker", {}),
        analyst_data_by_ticker=state.get("analyst_data_by_ticker", {}),
        insider_data_by_ticker=state.get("insider_data_by_ticker", {}),
        prior_sector_ratings=state.get("prior_sector_ratings", {}),
        current_sector_ratings=state.get("sector_ratings", {}),
        run_date=state["run_date"],
        episodic_memories=state.get("episodic_memories", {}),
        semantic_memories=state.get("semantic_memories", {}),
    )
    result = run_sector_team(team_id, ctx)

    # Return partial state update — _merge_dicts reducer merges team outputs
    return {
        "sector_team_outputs": {team_id: result},
    }


def macro_economist_node(state: ResearchState) -> dict:
    """Run the macro economist with reflection."""
    logger.info("[macro] starting")
    macro_data = state.get("macro_data", {})
    prior_report = state.get("prior_macro_report", "")

    # Derive prior date from macro_snapshots
    prior_date = ""
    prior_snapshots = state.get("prior_macro_snapshots", [])
    if prior_snapshots:
        prior_date = prior_snapshots[0].get("date", "")

    if prior_report:
        logger.info("[macro] using prior report from %s (%d chars)", prior_date, len(prior_report))
    else:
        logger.info("[macro] no prior report — generating fresh")

    result = run_macro_agent_with_reflection(
        prior_report=prior_report,
        prior_date=prior_date,
        macro_data=macro_data,
        prior_snapshots=prior_snapshots,
    )

    return {
        "macro_report": result.get("report_md", ""),
        "sector_modifiers": result.get("sector_modifiers", {}),
        "sector_ratings": result.get("sector_ratings", {}),
        "market_regime": result.get("market_regime", "neutral"),
    }


def exit_evaluator_node(state: ResearchState) -> dict:
    """Determine exits from current population using prior theses."""
    logger.info("[exit_evaluator] starting")

    # Build investment_theses from prior_theses for score lookup
    investment_theses = {}
    for ticker, thesis in state.get("prior_theses", {}).items():
        investment_theses[ticker] = {
            "long_term_score": thesis.get("score", 50),
            **thesis,
        }

    remaining, exits, open_slots = compute_exits_and_open_slots(
        current_population=state.get("current_population", []),
        investment_theses=investment_theses,
        config=POPULATION_CFG,
        run_date=state.get("run_date"),
    )

    return {
        "remaining_population": remaining,
        "exits": exits,
        "open_slots": open_slots,
    }


def merge_results_node(state: ResearchState) -> dict:
    """Fan-in: merge sector team outputs + macro + exits. Compute slot allocation."""
    logger.info("[merge] merging results")

    sector_ratings = state.get("sector_ratings", {})
    open_slots = state.get("open_slots", 0)

    team_slot_allocation = compute_team_slots(open_slots, sector_ratings)

    logger.info("[merge] %d open slots, allocation: %s", open_slots, team_slot_allocation)

    return {
        "team_slot_allocation": team_slot_allocation,
    }


def score_aggregator(state: ResearchState) -> dict:
    """Compute composite scores for all team recommendations."""
    logger.info("[score_aggregator] starting")

    team_outputs = state.get("sector_team_outputs", {})
    sector_modifiers = state.get("sector_modifiers", {})
    sector_map = state.get("sector_map", {})

    investment_theses = {}

    for team_id, output in team_outputs.items():
        # Score each recommendation
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "")
            sector = sector_map.get(ticker, "Unknown")
            modifier = sector_modifiers.get(sector, 1.0)

            score_result = compute_composite_score(
                quant_score=rec.get("quant_score"),
                qual_score=rec.get("qual_score"),
                sector_modifier=modifier,
            )

            investment_theses[ticker] = {
                "ticker": ticker,
                "sector": sector,
                "team_id": team_id,
                "final_score": score_result["final_score"],
                "quant_score": rec.get("quant_score"),
                "qual_score": rec.get("qual_score"),
                "weighted_base": score_result["weighted_base"],
                "macro_shift": score_result["macro_shift"],
                "bull_case": rec.get("bull_case", ""),
                "bear_case": rec.get("bear_case", ""),
                "catalysts": rec.get("catalysts", []),
                "conviction": normalize_conviction(rec.get("conviction", "medium")),
                "quant_rationale": rec.get("quant_rationale", ""),
                "rating": score_to_rating(
                    score_result["final_score"],
                    buy_threshold=RATING_BUY_THRESHOLD,
                    sell_threshold=RATING_SELL_THRESHOLD,
                ),
                "score_failed": score_result["score_failed"],
            }

        # Merge thesis updates from held stocks
        for ticker, thesis in output.get("thesis_updates", {}).items():
            if ticker not in investment_theses:
                # Skip thesis_updates that lack final_score. Without one,
                # the downstream _build_signals_payload safety gate will
                # downgrade the stock from BUY to HOLD with a
                # "broken thesis" warning (observed for 9 held tickers on
                # the 2026-04-11 run). The upstream cause is usually a
                # held-stock thesis that was saved without final_score —
                # either from a first-time held-stock update path that
                # never ran the aggregator, or a historic archive entry
                # that predates the current schema. Either way, skipping
                # here prevents the broken record from entering
                # investment_theses and fails loudly via the ERROR log so
                # the archive row can be backfilled.
                if thesis.get("final_score") is None:
                    logger.error(
                        "[score_aggregator] thesis_update for %s is missing "
                        "final_score — skipping investment_theses entry to "
                        "prevent broken-thesis downgrade downstream. "
                        "Upstream fix required: backfill the archive thesis "
                        "for this ticker or re-run the aggregator path.",
                        ticker,
                    )
                    continue
                investment_theses[ticker] = {
                    "ticker": ticker,
                    "team_id": team_id,
                    **thesis,
                }

    logger.info("[score_aggregator] scored %d tickers", len(investment_theses))

    return {"investment_theses": investment_theses}


def cio_node(state: ResearchState) -> dict:
    """Run CIO batch evaluation."""
    logger.info("[cio] starting")

    # Collect all team recommendations as candidate list
    team_outputs = state.get("sector_team_outputs", {})
    candidates = []
    for team_id, output in team_outputs.items():
        for rec in output.get("recommendations", []):
            candidates.append({
                **rec,
                "team_id": team_id,
            })

    # Load prior IC decisions for portfolio continuity
    prior_ic = []
    try:
        am = state.get("archive_manager")
        if am and am.db_conn:
            rows = am.db_conn.execute(
                "SELECT ticker, thesis_type, rationale, conviction, score "
                "FROM thesis_history WHERE run_date = ("
                "  SELECT MAX(run_date) FROM thesis_history WHERE run_date < ?"
                ") ORDER BY conviction DESC",
                (state.get("run_date", ""),)
            ).fetchall()
            prior_ic = [
                {"ticker": r[0], "thesis_type": r[1], "rationale": r[2],
                 "conviction": r[3], "score": r[4]}
                for r in rows
            ]
            if prior_ic:
                logger.info("[cio] loaded %d prior IC decisions", len(prior_ic))
    except Exception as e:
        logger.debug("[cio] prior IC decisions not available: %s", e)

    cio_result = run_cio(
        candidates=candidates,
        macro_context={
            "market_regime": state.get("market_regime", "neutral"),
            "macro_report": state.get("macro_report", ""),
        },
        sector_ratings=state.get("sector_ratings", {}),
        current_population=state.get("remaining_population", []),
        open_slots=state.get("open_slots", 0),
        exits=state.get("exits", []),
        run_date=state.get("run_date", ""),
        prior_decisions=prior_ic,
    )

    return {
        "ic_decisions": cio_result.get("decisions", []),
        "advanced_tickers": cio_result.get("advanced_tickers", []),
        "entry_theses": cio_result.get("entry_theses", {}),
    }


def population_entry_handler(state: ResearchState) -> dict:
    """Place IC ADVANCE decisions into population."""
    logger.info("[entry_handler] starting")

    final_pop, entry_events = apply_ic_entries(
        remaining_population=state.get("remaining_population", []),
        ic_decisions=state.get("ic_decisions", []),
        entry_theses=state.get("entry_theses", {}),
        sector_map=state.get("sector_map", {}),
        run_date=state.get("run_date", ""),
    )

    all_events = state.get("exits", []) + entry_events

    return {
        "new_population": final_pop,
        "population_rotation_events": all_events,
    }


def consolidator(state: ResearchState) -> dict:
    """Build the weekly research email with 4 structured sections."""
    logger.info("[consolidator] starting")

    sections = []
    run_date = state.get("run_date", "")

    # ── Section 1: Macro Regime Summary ──────────────────────────────────────
    regime = state.get("market_regime", "neutral")
    sections.append(f"# Daily Research Brief — {run_date}\n")
    sections.append("---\n")
    sections.append("## a. MACRO REGIME SUMMARY\n")
    sections.append(f"**Current Regime: {regime.upper()}**\n")
    macro_report = state.get("macro_report", "")
    if macro_report:
        # Strip code fences that the macro agent sometimes includes
        import re
        macro_report = re.sub(r"```\w*\n?", "", macro_report).strip()
        sections.append(macro_report)
    sections.append("")

    # ── Section 2: Sector Allocation ─────────────────────────────────────────
    sector_ratings = state.get("sector_ratings", {})
    if sector_ratings:
        sections.append("---\n")
        sections.append("## b. SECTOR ALLOCATION\n")
        sections.append("| Sector | Rating | Rationale |")
        sections.append("|--------|--------|-----------|")
        for sector in sorted(sector_ratings):
            sr = sector_ratings[sector]
            rating_raw = sr.get("rating", "market_weight")
            indicator = {"overweight": "\u25b2", "underweight": "\u25bc"}.get(rating_raw, "\u25cf")
            label = f"{indicator} {rating_raw.replace('_', ' ').upper()}"
            rationale = sr.get("rationale", "")
            sections.append(f"| {sector} | {label} | {rationale} |")
        sections.append("")

    # ── Section 3: Notable Developments ──────────────────────────────────────
    notable = _build_notable_developments(state)
    if notable:
        sections.append("---\n")
        sections.append("## c. NOTABLE DEVELOPMENTS\n")
        for item in notable:
            sections.append(f"- {item}")
        sections.append("")

    # ── Section 4: Universe Ratings (unified table) ─────────────────────────
    sections.append("---\n")
    sections.append("## d. UNIVERSE RATINGS\n")

    current_pop = state.get("current_population", [])
    new_pop = state.get("new_population", [])
    current_tickers = {p["ticker"] for p in current_pop} if current_pop else set()
    new_tickers = {p["ticker"] for p in new_pop} if new_pop else set()

    entrant_tickers = new_tickers - current_tickers
    exit_list = state.get("exits", [])
    exit_tickers = {e.get("ticker_out", "") for e in exit_list}

    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    entry_theses = state.get("entry_theses", {})
    team_outputs = state.get("sector_team_outputs", {})

    # Collect tickers with material thesis updates from sector teams
    updated_tickers = set()
    for team_id, output in team_outputs.items():
        for ticker in output.get("thesis_updates", {}):
            updated_tickers.add(ticker)

    pop_lookup = {p["ticker"]: p for p in new_pop}

    # Build unified rows: (ticker, status, rating, score, rationale)
    rows = []

    # Current portfolio stocks
    for p in new_pop:
        ticker = p["ticker"]
        thesis = theses.get(ticker, {})
        prior = prior_theses.get(ticker, {})
        pop_entry = pop_lookup.get(ticker, {})

        rating = thesis.get("rating") or prior.get("rating") or pop_entry.get("long_term_rating", "HOLD")
        score = thesis.get("final_score") or prior.get("score") or pop_entry.get("long_term_score", 0)

        if ticker in entrant_tickers:
            status = "NEW"
            et = entry_theses.get(ticker, {})
            rationale = et.get("bull_case") or thesis.get("bull_case", "New entry")
        elif ticker in updated_tickers and thesis.get("bull_case"):
            status = "UPDATED"
            rationale = thesis.get("bull_case", "")
        else:
            status = "HOLD"
            rationale = prior.get("thesis_summary") or thesis.get("bull_case", "Continuing coverage — no material update")

        rows.append((ticker, status, rating, score, rationale))

    # BUY recommendations not in portfolio (candidates that didn't get a slot)
    buy_candidates = []
    for ticker, thesis in theses.items():
        if thesis.get("rating") == "BUY" and ticker not in new_tickers and ticker not in exit_tickers:
            score = thesis.get("final_score", 0)
            rationale = thesis.get("bull_case", "Buy recommendation — no open slot")
            buy_candidates.append((ticker, "BUY REC", "BUY", score, rationale))

    # Exited stocks
    for e in exit_list:
        ticker = e.get("ticker_out", "?")
        score = e.get("score_out", 0)
        reason = e.get("reason", "Exited from population")
        rows.append((ticker, "EXIT", "SELL", score, reason))

    # Combine: portfolio + buy recs + exits
    rows.extend(buy_candidates)

    # Sort: NEW first, then BUY REC, then UPDATED, then HOLD by score desc, then EXIT
    status_order = {"NEW": 0, "BUY REC": 1, "UPDATED": 2, "HOLD": 3, "EXIT": 4}
    rows.sort(key=lambda r: (status_order.get(r[1], 9), -(r[3] or 0)))

    n_buy_recs = len(buy_candidates)
    sections.append(f"*{len(new_pop)} stocks in portfolio | {len(entrant_tickers)} new | {len(exit_list)} exited | {n_buy_recs} buy candidates*\n")
    sections.append("| Ticker | Status | Rating | Score | Rationale |")
    sections.append("|--------|--------|--------|-------|-----------|")
    for ticker, status, rating, score, rationale in rows:
        score_str = f"{score:.0f}" if score else "—"
        sections.append(f"| {ticker} | {status} | {rating} | {score_str} | {rationale} |")
    sections.append("")

    # Footer
    sections.append("---\n")
    sections.append(f"*Brief generated: {run_date} | Portfolio: {len(new_pop)} stocks*")

    consolidated = "\n".join(sections)
    return {"consolidated_report": consolidated}


def _build_notable_developments(state: ResearchState) -> list[str]:
    """Extract notable developments from team outputs and exits."""
    notable = []

    # High-conviction recommendations
    team_outputs = state.get("sector_team_outputs", {})
    for team_id, output in team_outputs.items():
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "?")
            bull = rec.get("bull_case", "")
            conviction = rec.get("conviction", "")
            if conviction in ("high",) and bull:
                notable.append(f"**{ticker} — High Conviction ({team_id.title()}):** {bull[:200]}")

    # Exits with reasons
    for e in state.get("exits", []):
        ticker = e.get("ticker_out", "?")
        reason = e.get("reason", "")
        if reason:
            notable.append(f"**{ticker} — Exit:** {reason[:200]}")

    # CIO advances
    for d in state.get("ic_decisions", []):
        if d.get("decision") == "ADVANCE":
            ticker = d.get("ticker", "?")
            rationale = d.get("rationale", "")
            if rationale:
                notable.append(f"**{ticker} — CIO Advance:** {rationale[:200]}")

    return notable[:7]


def archive_writer(state: ResearchState) -> dict:
    """Write all data to S3 + SQLite."""
    logger.info("[archive_writer] starting")
    am: ArchiveManager = state["archive_manager"]
    run_date = state["run_date"]
    # Bind once at the top so the scanner evaluations and team candidates
    # blocks below can reference it. Previously lines 945 and 975 referenced
    # a bare `team_outputs` that was never defined in this scope, causing
    # NameError and leaving team_candidates empty — which cascaded downstream
    # to the backtester evaluator on 2026-04-11.
    team_outputs = state.get("sector_team_outputs", {})

    # Save IC decisions
    for decision in state.get("ic_decisions", []):
        try:
            am.save_ic_decision(run_date, decision)
        except Exception as e:
            logger.warning("Failed to save IC decision for %s: %s", decision.get("ticker"), e)

    # Save stock archive entries
    for team_id, output in state.get("sector_team_outputs", {}).items():
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "")
            sector = state.get("sector_map", {}).get(ticker, "Unknown")
            try:
                am.save_stock_archive(ticker, sector, team_id, run_date)
            except Exception as e:
                logger.warning("Failed to save stock archive for %s: %s", ticker, e)

        # Save tool usage as analyst resources
        for tc in output.get("tool_calls", []):
            if tc.get("tool") and tc.get("ticker"):
                try:
                    am.save_analyst_resource(
                        ticker=tc["ticker"],
                        run_date=run_date,
                        agent=f"team:{team_id}",
                        resource_type=tc["tool"],
                    )
                except Exception as e:
                    logger.debug("[archive_writer] tool log failed: %s", e)

    # Save population
    new_pop = state.get("new_population", [])
    try:
        am.save_population(new_pop, run_date)
    except Exception as e:
        logger.warning("Failed to save population: %s", e)

    # Save rotation events
    for event in state.get("population_rotation_events", []):
        try:
            am.log_rotation_event(event, run_date)
        except Exception as e:
            logger.warning("Failed to save rotation event: %s", e)

    # Write signals.json (backward compatible)
    investment_theses = state.get("investment_theses", {})
    try:
        signals_payload = _build_signals_payload(state)
        am.write_signals_json(run_date, state.get("run_time", ""), signals_payload)
    except Exception as e:
        logger.error("Failed to write signals.json: %s", e)

    # Write prices.json snapshot for backtester consumption
    try:
        price_data = state.get("price_data", {})
        if price_data:
            prices_snapshot = {}
            for ticker, df in price_data.items():
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    prices_snapshot[ticker] = {
                        "open": round(float(last.get("Open", 0)), 2),
                        "high": round(float(last.get("High", 0)), 2),
                        "low": round(float(last.get("Low", 0)), 2),
                        "close": round(float(last.get("Close", 0)), 2),
                    }
            if prices_snapshot:
                am.write_prices_json(run_date, prices_snapshot)
                logger.info("[archive_writer] wrote prices.json with %d tickers", len(prices_snapshot))
    except Exception as e:
        logger.warning("Failed to write prices.json: %s", e)

    # Extract semantic memories from this run (Phase 3)
    try:
        from memory.semantic import extract_semantic_memories
        n_semantic = extract_semantic_memories(
            db_conn=am.db_conn,
            sector_team_outputs=state.get("sector_team_outputs", {}),
            macro_report=state.get("macro_report", ""),
            market_regime=state.get("market_regime", "neutral"),
            ic_decisions=state.get("ic_decisions", []),
            run_date=run_date,
        )
        if n_semantic:
            logger.info("[archive_writer] extracted %d semantic memories", n_semantic)
    except Exception as e:
        logger.debug("[archive_writer] semantic extraction skipped: %s", e)

    # ── Evaluation logging ──────────────────────────────────────────────────
    # Log all ~900 stocks with tech indicators for population baseline analysis.
    try:
        scanner_universe = state.get("scanner_universe", [])
        technical_scores = state.get("technical_scores", {})
        sector_map = state.get("sector_map", {})
        # Build set of tickers that any team picked (quant top-10 or recommended)
        team_picked_tickers: set[str] = set()
        for _tid, _out in team_outputs.items():
            for _rec in _out.get("recommendations", []):
                team_picked_tickers.add(_rec.get("ticker", ""))
            for _pick in _out.get("quant_output", {}).get("ranked_picks", []):
                if isinstance(_pick, dict):
                    team_picked_tickers.add(_pick.get("ticker", ""))

        scanner_evals = []
        for ticker in scanner_universe:
            ts = technical_scores.get(ticker, {})
            scanner_evals.append({
                "ticker": ticker,
                "eval_date": run_date,
                "sector": sector_map.get(ticker),
                "tech_score": ts.get("technical_score"),
                "rsi_14": ts.get("rsi_14"),
                "atr_pct": ts.get("atr_pct") or ts.get("atr_14_pct"),
                "price_vs_ma200": ts.get("price_vs_ma200"),
                "current_price": ts.get("current_price"),
                "avg_volume_20d": ts.get("avg_volume_20d"),
                "quant_filter_pass": 1 if ticker in team_picked_tickers else 0,
            })
        am.write_scanner_evaluations(scanner_evals)
        logger.info("[archive_writer] logged %d scanner evaluations", len(scanner_evals))
    except Exception as e:
        logger.warning("Failed to write scanner evaluations: %s", e)

    # Log quant top-10 per team + final recommendations
    try:
        team_candidate_records = []
        for team_id, output in team_outputs.items():
            quant_picks = output.get("quant_output", {}).get("ranked_picks", [])
            recommended_tickers = {
                r.get("ticker", "") for r in output.get("recommendations", [])
            }
            for rank, pick in enumerate(quant_picks, 1):
                if not isinstance(pick, dict) or "ticker" not in pick:
                    continue
                ticker = pick["ticker"]
                # Find qual score from recommendations if available
                qual_score = None
                for rec in output.get("recommendations", []):
                    if rec.get("ticker") == ticker:
                        qual_score = rec.get("qual_score")
                        break
                team_candidate_records.append({
                    "ticker": ticker,
                    "eval_date": run_date,
                    "team_id": team_id,
                    "quant_rank": rank,
                    "quant_score": pick.get("quant_score"),
                    "qual_score": qual_score,
                    "team_recommended": 1 if ticker in recommended_tickers else 0,
                })
        am.write_team_candidates(team_candidate_records)
        logger.info("[archive_writer] logged %d team candidates", len(team_candidate_records))
    except Exception as e:
        logger.warning("Failed to write team candidates: %s", e)

    # Log all CIO decisions (ADVANCE/REJECT/DEADLOCK)
    try:
        cio_eval_records = []
        for decision in state.get("ic_decisions", []):
            ticker = decision.get("ticker", "")
            thesis = investment_theses.get(ticker, {})
            cio_eval_records.append({
                "ticker": ticker,
                "eval_date": run_date,
                "team_id": thesis.get("team_id"),
                "quant_score": thesis.get("quant_score"),
                "qual_score": thesis.get("qual_score"),
                "combined_score": thesis.get("weighted_base"),
                "macro_shift": thesis.get("macro_shift"),
                "final_score": thesis.get("final_score"),
                "cio_decision": decision.get("decision", "UNKNOWN"),
                "cio_conviction": decision.get("conviction"),
                "cio_rank": decision.get("rank"),
                "rationale": decision.get("rationale"),
            })
        am.write_cio_evaluations(cio_eval_records)
        logger.info("[archive_writer] logged %d CIO evaluations", len(cio_eval_records))
    except Exception as e:
        logger.warning("Failed to write CIO evaluations: %s", e)

    # Upload DB
    try:
        am.upload_db(run_date)
    except Exception as e:
        logger.warning("Failed to upload DB: %s", e)

    return {}


def email_sender(state: ResearchState) -> dict:
    """Send the morning email with properly rendered HTML."""
    from emailer.sender import send_email
    from emailer.formatter import format_email
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER

    logger.info("[email_sender] starting")
    consolidated = state.get("consolidated_report", "")
    run_date = state.get("run_date", "")

    if consolidated:
        try:
            subject = f"Alpha Engine Research — {run_date}"
            html_body, plain_body = format_email(consolidated, run_date)
            send_email(
                subject=subject,
                html_body=html_body,
                plain_body=plain_body,
                recipients=EMAIL_RECIPIENTS,
                sender=EMAIL_SENDER,
            )
            return {"email_sent": True}
        except Exception as e:
            logger.error("Email send failed: %s", e)

    return {"email_sent": False}


def _build_signals_payload(state: ResearchState) -> dict:
    """Build backward-compatible signals.json payload.

    Includes both v2 keys (signals, population) and v1 keys (universe, buy_candidates)
    so the executor and predictor can read actionable signals.
    """
    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    pop = state.get("new_population", [])
    pop_tickers = {p["ticker"] for p in pop}
    pop_lookup = {p["ticker"]: p for p in pop}
    sector_map = state.get("sector_map", {})
    sector_ratings = state.get("sector_ratings", {})
    entry_theses = state.get("entry_theses", {})

    # v2 signals dict (keyed by ticker)
    # Signal logic:
    #   ENTER  = BUY-rated AND in population (new entry or reaffirmed hold)
    #   HOLD   = held in population, not BUY-rated (maintain position)
    #   EXIT   = dropped from population (sell)
    #   Stocks not in population and not BUY-rated are excluded (irrelevant to executor)
    advanced_tickers = set(state.get("advanced_tickers", []))
    signals = {}

    # First: tickers with fresh theses from this run
    for ticker, thesis in theses.items():
        rating = thesis.get("rating", "HOLD")
        final_score = thesis.get("final_score")
        in_pop = ticker in pop_tickers

        # Safety gate: a BUY rating with no final_score is a broken thesis
        # (e.g., held-stock LLM update that dropped scoring fields). Downgrade
        # to HOLD so the executor does not attempt to ENTER on null score.
        # Root cause mitigation for the 2026-04-04 incident where
        # LNTH/KR/PR/HAL leaked through as ENTER with score=null.
        if rating == "BUY" and final_score is None:
            logger.warning(
                "[signals] %s has rating=BUY but final_score is None — "
                "downgrading to HOLD (broken thesis)",
                ticker,
            )
            rating = "HOLD"

        # Determine signal
        if rating == "BUY" and ticker in advanced_tickers:
            signal = "ENTER"  # CIO approved new entry
        elif rating == "BUY" and in_pop:
            signal = "ENTER"  # Reaffirm existing BUY position
        elif in_pop:
            signal = "HOLD"   # Held, not BUY-rated
        elif rating == "BUY":
            signal = "ENTER"  # BUY recommendation (candidate, not yet held)
        else:
            continue  # Not held, not recommended — skip

        signals[ticker] = {
            "ticker": ticker,
            "score": thesis.get("final_score"),
            "rating": rating,
            "signal": signal,
            "conviction": normalize_conviction(thesis.get("conviction", "stable")),
            "thesis_summary": thesis.get("bull_case", ""),
            "sector": thesis.get("sector", "Unknown"),
            "team_id": thesis.get("team_id"),
            "quant_score": thesis.get("quant_score"),
            "qual_score": thesis.get("qual_score"),
            "sub_scores": {
                "quant": thesis.get("quant_score"),
                "qual": thesis.get("qual_score"),
            },
        }

    # Second: population tickers without fresh theses — carry over from prior week
    for p in pop:
        ticker = p["ticker"]
        if ticker not in signals:
            prior = prior_theses.get(ticker, {})
            sector = sector_map.get(ticker, p.get("sector", "Unknown"))
            prior_rating = prior.get("rating") or p.get("long_term_rating", "HOLD")
            carried_score = prior.get("score") or p.get("long_term_score")
            # Only emit ENTER if we have a score — unscored holdovers stay HOLD
            if prior_rating == "BUY" and carried_score is not None:
                carried_signal = "ENTER"
            else:
                carried_signal = "HOLD"
            signals[ticker] = {
                "ticker": ticker,
                "score": carried_score,
                "rating": prior_rating,
                "signal": carried_signal,
                "conviction": normalize_conviction(prior.get("conviction") or p.get("conviction", "stable")),
                "thesis_summary": prior.get("thesis_summary", ""),
                "sector": sector,
                "team_id": prior.get("team_id"),
                "quant_score": prior.get("quant_score"),
                "qual_score": prior.get("qual_score"),
                "sub_scores": {
                    "quant": prior.get("quant_score"),
                    "qual": prior.get("qual_score"),
                },
            }

    # Third: exited stocks — explicit EXIT signal so executor knows to sell
    for e in state.get("exits", []):
        ticker = e.get("ticker_out", "")
        if ticker and ticker not in signals:
            sector = sector_map.get(ticker, "Unknown")
            signals[ticker] = {
                "ticker": ticker,
                "score": e.get("score_out", 0),
                "rating": "SELL",
                "signal": "EXIT",
                "conviction": "declining",
                "thesis_summary": e.get("reason", "Exited from population"),
                "sector": sector,
                "team_id": None,
                "quant_score": None,
                "qual_score": None,
                "sub_scores": {"quant": None, "qual": None},
            }

    # v1-compatible universe list (executor reads this)
    universe = []
    for ticker, sig in signals.items():
        sector = sig["sector"]
        sr = sector_ratings.get(sector, {})
        pop_entry = pop_lookup.get(ticker, {})
        universe.append({
            "ticker": ticker,
            "signal": sig["signal"],
            "score": sig["score"],
            "rating": sig["rating"],
            "conviction": sig["conviction"],
            "price_target_upside": pop_entry.get("price_target_upside"),
            "sector_rating": sr.get("rating", "market_weight"),
            "sector": sector,
            "thesis_summary": sig["thesis_summary"],
            "sub_scores": sig.get("sub_scores"),
        })

    # v1-compatible buy_candidates list (ENTER signals with enriched theses)
    buy_candidates = []
    for entry in universe:
        if entry["signal"] == "ENTER":
            candidate = dict(entry)
            et = entry_theses.get(entry["ticker"], {})
            if et:
                candidate["thesis_summary"] = et.get("bull_case", candidate["thesis_summary"])
                candidate["catalysts"] = et.get("catalysts", [])
            buy_candidates.append(candidate)

    return {
        "date": state.get("run_date", ""),
        "time": state.get("run_time", ""),
        "market_regime": state.get("market_regime", "neutral"),
        "sector_modifiers": state.get("sector_modifiers", {}),
        "sector_ratings": sector_ratings,
        "signals": signals,
        "population": [p["ticker"] for p in pop],
        "universe": universe,
        "buy_candidates": buy_candidates,
        "architecture_version": "sector_teams",
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Sector-team graph with Send() fan-out.

    Topology:
      fetch_data → dispatch_all (Send: 6 teams + macro + exit)
      → merge_results → score_aggregator → cio_node
      → population_entry_handler → consolidator → archive → email → END
    """
    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("sector_team_node", sector_team_node)
    graph.add_node("macro_economist_node", macro_economist_node)
    graph.add_node("exit_evaluator_node", exit_evaluator_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("score_aggregator", score_aggregator)
    graph.add_node("cio_node", cio_node)
    graph.add_node("population_entry_handler", population_entry_handler)
    graph.add_node("consolidator_node", consolidator)
    graph.add_node("archive_writer", archive_writer)
    graph.add_node("email_sender_node", email_sender)

    # Entry point
    graph.set_entry_point("fetch_data")

    # Fan-out: fetch_data → dispatch (sends to teams + macro + exit evaluator)
    graph.add_conditional_edges("fetch_data", dispatch_all)

    # Fan-in: all three Send targets converge to merge_results
    graph.add_edge("sector_team_node", "merge_results")
    graph.add_edge("macro_economist_node", "merge_results")
    graph.add_edge("exit_evaluator_node", "merge_results")

    # Sequential post-merge
    graph.add_edge("merge_results", "score_aggregator")
    graph.add_edge("score_aggregator", "cio_node")
    graph.add_edge("cio_node", "population_entry_handler")
    graph.add_edge("population_entry_handler", "consolidator_node")
    graph.add_edge("consolidator_node", "archive_writer")
    graph.add_edge("archive_writer", "email_sender_node")
    graph.add_edge("email_sender_node", END)

    return graph.compile()


def create_initial_state(
    run_date: str,
    archive_manager: ArchiveManager,
    is_early_close: bool = False,
) -> ResearchState:
    return ResearchState(
        run_date=run_date,
        run_time=datetime.now(timezone.utc).isoformat(),
        archive_manager=archive_manager,
        is_early_close=is_early_close,
        email_sent=False,
    )
