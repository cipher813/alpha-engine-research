"""
LangGraph research graph — full pipeline state machine (§9).

Topology:
  fetch_data
    ├── BRANCH A: universe pipeline (fan-out per ticker, macro global)
    └── BRANCH B: scanner pipeline (sequential stages within branch)
  [fan-in join]
  score_aggregator
  thesis_updater + candidate_evaluator
  consolidator_agent
  archive_writer
  email_sender

Parallelism via LangGraph Send nodes; semaphore limits concurrent LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

logger = logging.getLogger(__name__)

import pandas as pd
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from config import (
    UNIVERSE_TICKERS,
    SECTOR_MAP,
    CONCURRENT_AGENTS,
    ROTATION_TIERS,
    WEAK_PICK_SCORE_THRESHOLD,
    WEAK_PICK_CONSECUTIVE_RUNS,
    EMERGENCY_ROTATION_NEW_SCORE,
    DEEP_VALUE_MIN_CONSENSUS,
)
from agents.news_agent import run_news_agent
from agents.research_agent import run_research_agent
from agents.macro_agent import run_macro_agent
from agents.scanner_ranking_agent import run_scanner_ranking_agent
from agents.consolidator import run_consolidator_agent
from data.fetchers.price_fetcher import fetch_price_data, compute_technical_indicators
from data.fetchers.news_fetcher import fetch_all_news
from data.fetchers.analyst_fetcher import fetch_analyst_consensus
from data.fetchers.macro_fetcher import fetch_macro_data
from data.deduplicator import deduplicate_articles, compute_recurring_themes
from data.scanner import (
    get_scanner_universe,
    run_quant_filter,
    confirm_deep_value_with_analyst,
    evaluate_candidate_rotation,
)
from scoring.technical import compute_technical_score, compute_momentum_percentiles
from scoring.aggregator import aggregate_all
from scoring.performance_tracker import run_performance_checks, record_new_buy_scores
from thesis.updater import build_thesis_record
from emailer.formatter import format_email
from emailer.sender import send_email
from archive.manager import ArchiveManager

# ── State schema (§9.1) ───────────────────────────────────────────────────────
# Keys written by both parallel branches (run_universe_agents, run_scanner_pipeline)
# need reducers so LangGraph can merge concurrent updates.

def _take_last(_left: Any, right: Any) -> Any:
    """Reducer: when multiple nodes write the same key, take the last value."""
    return right


class ResearchState(TypedDict, total=False):
    run_date: Annotated[str, _take_last]
    run_time: Annotated[str, _take_last]
    universe_tickers: Annotated[list[str], _take_last]
    candidate_tickers: Annotated[list[str], _take_last]
    price_data: Annotated[dict[str, Any], _take_last]
    news_headlines: Annotated[dict[str, list[dict]], _take_last]
    analyst_data: Annotated[dict[str, dict], _take_last]
    macro_data: Annotated[dict, _take_last]
    prior_news_reports: Annotated[dict[str, str], _take_last]
    prior_research_reports: Annotated[dict[str, str], _take_last]
    prior_macro_report: Annotated[str, _take_last]
    prior_theses: Annotated[dict[str, dict], _take_last]
    active_candidates: Annotated[list[dict], _take_last]
    news_article_hashes: Annotated[dict[str, set], _take_last]
    technical_scores: Annotated[dict[str, dict], _take_last]
    news_reports: Annotated[dict[str, str], _take_last]
    news_scores: Annotated[dict[str, float], _take_last]
    research_reports: Annotated[dict[str, str], _take_last]
    research_scores: Annotated[dict[str, float], _take_last]
    macro_report: Annotated[str, _take_last]
    sector_modifiers: Annotated[dict[str, float], _take_last]
    sector_ratings: Annotated[dict[str, dict], _take_last]
    market_regime: Annotated[str, _take_last]
    score_history: Annotated[dict[str, list[float]], _take_last]
    performance_summary: Annotated[dict, _take_last]
    investment_theses: Annotated[dict[str, dict], _take_last]
    scanner_filtered: Annotated[list[dict], _take_last]
    scanner_ranked: Annotated[list[dict], _take_last]
    scanner_news_reports: Annotated[dict[str, str], _take_last]
    scanner_research_reports: Annotated[dict[str, str], _take_last]
    scanner_scores: Annotated[dict[str, dict], _take_last]
    new_candidates: Annotated[list[dict], _take_last]
    consolidated_report: Annotated[str, _take_last]
    email_sent: Annotated[bool, _take_last]
    archive_manager: Annotated[Any, _take_last]
    is_early_close: Annotated[bool, _take_last]
    _scanner_universe: Annotated[list[str], _take_last]  # internal: scanner ticker list
    _rotation_events: Annotated[list, _take_last]  # internal: rotation log


# ── Semaphore for LLM concurrency ─────────────────────────────────────────────
_llm_semaphore = asyncio.Semaphore(CONCURRENT_AGENTS)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def fetch_data(state: ResearchState) -> ResearchState:
    """
    Node 1: Download all data needed for both branches.
    - Price data for universe + candidates + S&P 500/400
    - News headlines for universe + candidates
    - Analyst data for universe + candidates
    - Macro data from FRED + yfinance
    - Prior reports from archive
    """
    run_date = state["run_date"]
    archive: ArchiveManager = state["archive_manager"]

    # Load active candidates from DB
    active_candidates = archive.load_active_candidates()
    candidate_tickers = [c["symbol"] for c in active_candidates]
    all_tracked = list(set(UNIVERSE_TICKERS + candidate_tickers))

    # Fetch price data for all tracked + S&P 500/400 universe
    scanner_universe = get_scanner_universe(exclude_tickers=all_tracked)
    scanner_universe_sample = scanner_universe
    all_tickers_to_fetch = all_tracked + scanner_universe_sample

    price_data = fetch_price_data(all_tickers_to_fetch, period="1y")

    # Fetch news for tracked tickers
    news_headlines: dict[str, list] = {}
    news_article_hashes: dict[str, set] = {}
    for ticker in all_tracked:
        raw = fetch_all_news(ticker, hours=48)
        news_headlines[ticker] = raw["yahoo"] + raw.get("edgar_8k", [])
        news_article_hashes[ticker] = archive.load_news_hashes(ticker)

    # Fetch analyst data for tracked tickers
    analyst_data: dict[str, dict] = {}
    for ticker in all_tracked:
        try:
            analyst_data[ticker] = fetch_analyst_consensus(ticker)
        except Exception as e:
            analyst_data[ticker] = {"ticker": ticker, "error": str(e)}

    # Fetch macro data
    try:
        macro_data = fetch_macro_data()
    except Exception as e:
        macro_data = {"error": str(e)}

    # Load prior reports from archive
    prior_news_reports: dict[str, str] = {}
    prior_research_reports: dict[str, str] = {}
    prior_theses: dict[str, dict] = {}

    for ticker in all_tracked:
        category = "candidates" if ticker in candidate_tickers else "universe"
        reports = archive.load_prior_reports(ticker, category=category)
        if reports["news_report"]:
            prior_news_reports[ticker] = reports["news_report"]
        if reports["research_report"]:
            prior_research_reports[ticker] = reports["research_report"]
        if reports["thesis"]:
            prior_theses[ticker] = reports["thesis"]

    prior_macro_report_data = archive.load_prior_reports("macro_global", category="macro")
    prior_macro_report = prior_macro_report_data.get("news_report") or ""

    # Load score history for conviction and velocity computation (§A.3, A.4)
    score_history = archive.load_score_history(all_tracked, n=6)

    return {
        **state,
        "universe_tickers": UNIVERSE_TICKERS,
        "candidate_tickers": candidate_tickers,
        "active_candidates": active_candidates,
        "price_data": price_data,
        "news_headlines": news_headlines,
        "news_article_hashes": news_article_hashes,
        "analyst_data": analyst_data,
        "macro_data": macro_data,
        "prior_news_reports": prior_news_reports,
        "prior_research_reports": prior_research_reports,
        "prior_theses": prior_theses,
        "prior_macro_report": prior_macro_report,
        "score_history": score_history,
        "_scanner_universe": scanner_universe_sample,
        "_price_data_full": price_data,
    }


def run_universe_agents(state: ResearchState) -> ResearchState:
    """
    Branch A: Run all per-ticker universe agents (news + research, parallel).
    Also runs the macro agent (global).
    Uses asyncio for concurrency up to CONCURRENT_AGENTS.
    """
    run_date = state["run_date"]
    all_tickers = list(set(state.get("universe_tickers", []) + state.get("candidate_tickers", [])))

    news_reports: dict[str, str] = {}
    news_scores: dict[str, float] = {}
    research_reports: dict[str, str] = {}
    research_scores: dict[str, float] = {}

    # Compute technical scores
    price_data = state.get("price_data", {})
    technical_scores: dict[str, dict] = {}
    momentum_data: dict[str, Optional[float]] = {}

    for ticker in all_tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            indicators = compute_technical_indicators(df)
            if indicators:
                technical_scores[ticker] = indicators
                momentum_data[ticker] = indicators.get("momentum_20d")

    # Compute momentum percentiles across S&P 500 universe for context
    momentum_percentiles = compute_momentum_percentiles(momentum_data)

    for ticker in all_tickers:
        if ticker in technical_scores:
            regime = state.get("market_regime", "neutral")
            percentile = momentum_percentiles.get(ticker)
            ts = compute_technical_score(
                technical_scores[ticker],
                market_regime=regime,
                momentum_percentile=percentile,
            )
            technical_scores[ticker]["technical_score"] = ts

    # Run news + research agents for each ticker
    prior_date_default = "NONE"

    def run_ticker_agents(ticker: str):
        # Deduplicate articles
        prior_hashes = state.get("news_article_hashes", {}).get(ticker, set())
        all_articles = state.get("news_headlines", {}).get(ticker, [])
        novel_articles, _ = deduplicate_articles(all_articles, prior_hashes)
        recurring = compute_recurring_themes(novel_articles)

        prior_news = state.get("prior_news_reports", {}).get(ticker)
        prior_research = state.get("prior_research_reports", {}).get(ticker)
        prior_thesis = state.get("prior_theses", {}).get(ticker, {})
        prior_date = prior_thesis.get("date", prior_date_default)

        tech = technical_scores.get(ticker, {})
        current_price = tech.get("current_price", 0)

        # News agent
        try:
            news_result = run_news_agent(
                ticker=ticker,
                company_name=ticker,  # company name lookup can be added later
                prior_report=prior_news,
                prior_date=prior_date,
                new_articles=novel_articles,
                recurring_themes=recurring,
                sec_filings=[a for a in all_articles if a.get("source") == "SEC EDGAR"],
                current_price=current_price,
                price_change_pct=tech.get("momentum_20d", 0) or 0,
                price_change_date=run_date,
            )
            news_reports[ticker] = news_result["report_md"]
            news_scores[ticker] = news_result["news_score"]
        except Exception as e:
            news_reports[ticker] = f"Error: {e}"
            news_scores[ticker] = 50.0

        # Research agent
        try:
            research_result = run_research_agent(
                ticker=ticker,
                company_name=ticker,
                prior_report=prior_research,
                prior_date=prior_date,
                analyst_data=state.get("analyst_data", {}).get(ticker, {}),
            )
            research_reports[ticker] = research_result["report_md"]
            research_scores[ticker] = research_result["research_score"]
        except Exception as e:
            research_reports[ticker] = f"Error: {e}"
            research_scores[ticker] = 50.0

    # Run agents (simple sequential for now; can be made async for full parallelism)
    for ticker in all_tickers:
        run_ticker_agents(ticker)

    # Macro agent
    _default_sectors = ["Technology", "Healthcare", "Financial", "Consumer Discretionary",
                        "Consumer Staples", "Energy", "Industrials", "Materials",
                        "Real Estate", "Utilities", "Communication Services"]
    try:
        macro_result = run_macro_agent(
            prior_report=state.get("prior_macro_report"),
            prior_date=state.get("run_date", "NONE"),
            macro_data=state.get("macro_data", {}),
        )
        macro_report = macro_result["report_md"]
        sector_modifiers = macro_result["sector_modifiers"]
        sector_ratings = macro_result["sector_ratings"]
        market_regime = macro_result["market_regime"]
    except Exception as e:
        macro_report = f"Macro agent error: {e}"
        sector_modifiers = {s: 1.0 for s in _default_sectors}
        sector_ratings = {s: {"rating": "market_weight", "rationale": "Macro data unavailable"}
                          for s in _default_sectors}
        market_regime = "neutral"

    # Recompute technical scores with correct regime
    for ticker in all_tickers:
        if ticker in technical_scores:
            percentile = momentum_percentiles.get(ticker)
            ts = compute_technical_score(
                technical_scores[ticker],
                market_regime=market_regime,
                momentum_percentile=percentile,
            )
            technical_scores[ticker]["technical_score"] = ts

    return {
        "technical_scores": technical_scores,
        "news_reports": news_reports,
        "news_scores": news_scores,
        "research_reports": research_reports,
        "research_scores": research_scores,
        "macro_report": macro_report,
        "sector_modifiers": sector_modifiers,
        "sector_ratings": sector_ratings,
        "market_regime": market_regime,
    }


def run_scanner(
    run_date: str,
    scanner_universe: list[str],
    price_data: dict,
    archive: ArchiveManager,
    technical_scores: Optional[dict] = None,
    market_regime: str = "neutral",
) -> dict:
    """
    Full scanner pipeline (Stages 1–4). Pure function — no LangGraph state.

    Returns dict with keys:
      scanner_filtered, scanner_ranked, scanner_news_reports,
      scanner_research_reports, scanner_scores
    """
    technical_scores = technical_scores or {}

    # Stage 1: Quant filter
    candidates = run_quant_filter(
        tickers=scanner_universe,
        price_data=price_data,
        technical_scores=technical_scores,
        market_regime=market_regime,
    )
    logger.info("[scanner] stage=1 quant_filter candidates=%d universe=%d", len(candidates), len(scanner_universe))

    # Stage 2: Data enrichment
    analyst_data_scanner: dict[str, dict] = {}
    for c in candidates:
        ticker = c["ticker"]
        try:
            analyst_data_scanner[ticker] = fetch_analyst_consensus(ticker)
            c["analyst_rating"] = analyst_data_scanner[ticker].get("consensus_rating", "Hold")
            c["upside_pct"] = analyst_data_scanner[ticker].get("upside_pct")
        except Exception:
            c["analyst_rating"] = "Hold"
            c["upside_pct"] = None

        try:
            news = fetch_all_news(ticker, hours=24)
            c["headlines"] = [a.get("headline", "") for a in news["yahoo"][:2]]
            c["articles"] = news["yahoo"]
            c["sec_filings"] = news["edgar_8k"]
        except Exception:
            c["headlines"] = []
            c["articles"] = []
            c["sec_filings"] = []

        c["sector"] = SECTOR_MAP.get(ticker, "Technology")

    candidates = confirm_deep_value_with_analyst(
        candidates,
        analyst_data=analyst_data_scanner,
        min_consensus=DEEP_VALUE_MIN_CONSENSUS,
    )
    logger.info("[scanner] stage=2 confirmed_candidates=%d", len(candidates))

    # Stage 3: Ranking agent
    ranked_top10 = run_scanner_ranking_agent(
        candidates=candidates,
        market_regime=market_regime,
    )
    logger.info("[scanner] stage=3 ranked=%d", len(ranked_top10))

    # Stage 4: Deep analysis for top 10
    scanner_news_reports: dict[str, str] = {}
    scanner_research_reports: dict[str, str] = {}
    scanner_scores: dict[str, dict] = {}

    articles_by_ticker = {c["ticker"]: c.get("articles", []) for c in candidates}
    sec_filings_by_ticker = {c["ticker"]: c.get("sec_filings", []) for c in candidates}

    for entry in ranked_top10:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue

        prior_reports = archive.load_prior_reports(ticker, category="candidates")
        prior_news = prior_reports.get("news_report")
        prior_research = prior_reports.get("research_report")
        prior_date = (prior_reports.get("thesis") or {}).get("date", "NONE")

        adata = analyst_data_scanner.get(ticker, {})
        df = price_data.get(ticker, pd.DataFrame())
        current_price_val = float(df["Close"].iloc[-1]) if not df.empty else adata.get("current_price", 0) or 0

        try:
            news_result = run_news_agent(
                ticker=ticker, company_name=ticker,
                prior_report=prior_news, prior_date=prior_date,
                new_articles=articles_by_ticker.get(ticker, []),
                recurring_themes=[],
                sec_filings=sec_filings_by_ticker.get(ticker, []),
                current_price=current_price_val,
                price_change_pct=0, price_change_date=run_date,
            )
            scanner_news_reports[ticker] = news_result["report_md"]
            news_score = news_result["news_score"]
        except Exception as e:
            scanner_news_reports[ticker] = f"Error: {e}"
            news_score = 50.0

        try:
            research_result = run_research_agent(
                ticker=ticker, company_name=ticker,
                prior_report=prior_research, prior_date=prior_date,
                analyst_data=adata,
            )
            scanner_research_reports[ticker] = research_result["report_md"]
            research_score = research_result["research_score"]
        except Exception as e:
            scanner_research_reports[ticker] = f"Error: {e}"
            research_score = 50.0

        tech = technical_scores.get(ticker) or {}
        tech_score = tech.get("technical_score", 50.0)

        scanner_scores[ticker] = {
            "ticker": ticker,
            "tech_score": tech_score,
            "news_score": news_score,
            "research_score": research_score,
            "rank": entry.get("rank"),
            "path": entry.get("path", "momentum"),
        }

    logger.info("[scanner] stage=4 scored=%d tickers=%s", len(scanner_scores), list(scanner_scores.keys()))
    return {
        "scanner_filtered": candidates,
        "scanner_ranked": ranked_top10,
        "scanner_news_reports": scanner_news_reports,
        "scanner_research_reports": scanner_research_reports,
        "scanner_scores": scanner_scores,
    }


def run_scanner_pipeline(state: ResearchState) -> ResearchState:
    """LangGraph node — unpacks state and delegates to run_scanner()."""
    result = run_scanner(
        run_date=state["run_date"],
        scanner_universe=state.get("_scanner_universe", []),
        price_data=state.get("price_data", {}),
        archive=state["archive_manager"],
        technical_scores=state.get("technical_scores", {}),
        market_regime=state.get("market_regime", "neutral"),
    )
    return result


def _compute_price_target_upside(
    analyst_data: dict[str, dict],
    technical_scores: dict[str, dict],
) -> dict[str, float | None]:
    """
    Compute (consensus_price_target / current_price - 1) for each ticker.
    Returns None for any ticker where data is unavailable or price is zero.
    """
    result: dict[str, float | None] = {}
    for ticker, adata in analyst_data.items():
        target = adata.get("price_target") or adata.get("mean_target")
        price = (technical_scores.get(ticker) or {}).get("current_price")
        if target and price and price > 0:
            try:
                result[ticker] = round(float(target) / float(price) - 1.0, 4)
            except (TypeError, ValueError):
                result[ticker] = None
        else:
            result[ticker] = None
    return result


def score_aggregator(state: ResearchState) -> ResearchState:
    """
    Fan-in join: aggregate scores for all stocks after both branches complete.
    """
    run_date = state["run_date"]
    all_tickers = list(set(
        state.get("universe_tickers", []) + state.get("candidate_tickers", [])
    ))

    # Precompute price target upside from analyst data (§A.1)
    price_target_upside = _compute_price_target_upside(
        analyst_data=state.get("analyst_data", {}),
        technical_scores=state.get("technical_scores", {}),
    )

    # Aggregate universe + candidate scores
    aggregated = aggregate_all(
        tickers=all_tickers,
        technical_scores=state.get("technical_scores", {}),
        news_scores=state.get("news_scores", {}),
        research_scores=state.get("research_scores", {}),
        sector_modifiers=state.get("sector_modifiers", {}),
        prior_theses=state.get("prior_theses", {}),
        run_date=run_date,
        score_history=state.get("score_history", {}),
        price_target_upside=price_target_upside,
    )

    # Aggregate scanner candidate scores (sector modifiers applied)
    scanner_scores = state.get("scanner_scores", {})
    sector_modifiers = state.get("sector_modifiers", {})
    for ticker, sdata in scanner_scores.items():
        sector = SECTOR_MAP.get(ticker, "Technology")
        modifier = sector_modifiers.get(sector, 1.0)
        base = (
            sdata["tech_score"] * 0.40
            + sdata["news_score"] * 0.30
            + sdata["research_score"] * 0.30
        )
        final = max(0.0, min(100.0, base * modifier))
        sdata["score"] = round(final, 2)
        sdata["sector"] = sector

    return {**state, "investment_theses": aggregated, "scanner_scores": scanner_scores}


def thesis_updater(state: ResearchState) -> ResearchState:
    """
    Build full thesis records for all universe + candidate tickers.
    """
    run_date = state["run_date"]
    investment_theses = state.get("investment_theses", {})
    full_theses: dict[str, dict] = {}

    for ticker, aggregated in investment_theses.items():
        news_json = {}
        research_json = {}
        # Try to extract JSON from agent reports
        # (agents embed JSON at end of their reports; thesis updater extracts it)
        full_theses[ticker] = build_thesis_record(
            ticker=ticker,
            run_date=run_date,
            aggregated=aggregated,
            agent_outputs={"news_json": news_json, "research_json": research_json},
        )

    return {**state, "investment_theses": full_theses}


def candidate_evaluator(state: ResearchState) -> ResearchState:
    """
    Stage 5: Determine candidate rotation using tiered rule-based logic.
    At most 1 rotation per run.
    """
    run_date = state["run_date"]
    active_candidates = state.get("active_candidates", [])
    scanner_scores = state.get("scanner_scores", {})
    investment_theses = state.get("investment_theses", {})

    # Update active candidate scores from this run
    for cand in active_candidates:
        ticker = cand["symbol"]
        thesis = investment_theses.get(ticker, {})
        cand["score"] = thesis.get("final_score", cand.get("score", 50))
        # Track consecutive low runs
        if cand["score"] < WEAK_PICK_SCORE_THRESHOLD:
            cand["consecutive_low_runs"] = cand.get("consecutive_low_runs", 0) + 1
        else:
            cand["consecutive_low_runs"] = 0

    new_active, rotations = evaluate_candidate_rotation(
        scanner_scores=scanner_scores,
        active_candidates=active_candidates,
        rotation_tiers=ROTATION_TIERS,
        weak_pick_score_threshold=WEAK_PICK_SCORE_THRESHOLD,
        weak_pick_consecutive_runs=WEAK_PICK_CONSECUTIVE_RUNS,
        emergency_rotation_new_score=EMERGENCY_ROTATION_NEW_SCORE,
        run_date=run_date,
    )

    # Annotate new candidates with status for consolidator
    new_tickers = {c["symbol"] for c in new_active}
    old_tickers = {c["symbol"] for c in active_candidates}

    for c in new_active:
        if c["symbol"] not in old_tickers:
            prior_tenures = state.get("archive_manager").db_conn.execute(
                "SELECT COUNT(*) FROM candidate_tenures WHERE symbol = ?", (c["symbol"],)
            ).fetchone()[0] if state.get("archive_manager") else 0
            if prior_tenures > 0:
                c["status"] = f"RETURNED({prior_tenures} prior tenures)"
            else:
                c["status"] = "NEW_ENTRY"
        else:
            c["status"] = "CONTINUING"

    logger.info("[scanner] stage=5 active_candidates=%s rotations=%d",
                [c["symbol"] for c in new_active], len(rotations))
    return {**state, "new_candidates": new_active, "_rotation_events": rotations}


def consolidator_node(state: ResearchState) -> ResearchState:
    """Run the Consolidator Agent to produce the final email body."""
    run_date = state["run_date"]
    new_candidates = state.get("new_candidates", state.get("active_candidates", []))

    # Prepare candidate thesis data with full reports
    candidate_full_news = {
        c["symbol"]: state.get("news_reports", {}).get(c["symbol"], "")
        for c in new_candidates
    }
    candidate_full_research = {
        c["symbol"]: state.get("research_reports", {}).get(c["symbol"], "")
        for c in new_candidates
    }

    # Enrich candidates with thesis data
    for c in new_candidates:
        ticker = c["symbol"]
        thesis = state.get("investment_theses", {}).get(ticker, {})
        c["score"] = thesis.get("final_score", c.get("score", 0))
        c["score_delta"] = thesis.get("score_delta")
        c["thesis_summary"] = thesis.get("thesis_summary", "")
        c["key_catalyst"] = thesis.get("key_catalyst", "")
        c["key_risk"] = thesis.get("key_risk", "")

    # Separate universe vs candidate reports
    candidate_tickers = {c["symbol"] for c in new_candidates}
    universe_news = {k: v for k, v in state.get("news_reports", {}).items() if k not in candidate_tickers}
    universe_research = {k: v for k, v in state.get("research_reports", {}).items() if k not in candidate_tickers}
    universe_theses = {k: v for k, v in state.get("investment_theses", {}).items() if k not in candidate_tickers}

    consolidated = run_consolidator_agent(
        run_date=run_date,
        macro_report=state.get("macro_report", ""),
        universe_news_reports=universe_news,
        universe_research_reports=universe_research,
        candidate_full_news=candidate_full_news,
        candidate_full_research=candidate_full_research,
        investment_theses=universe_theses,
        active_candidates=new_candidates,
        performance_summary=state.get("performance_summary", {}),
        sector_ratings=state.get("sector_ratings", {}),
        is_early_close=state.get("is_early_close", False),
    )

    return {**state, "consolidated_report": consolidated}


def archive_writer(state: ResearchState) -> ResearchState:
    """Write all outputs to S3 and SQLite."""
    run_date = state["run_date"]
    run_time = state["run_time"]
    archive: ArchiveManager = state["archive_manager"]
    investment_theses = state.get("investment_theses", {})
    all_tickers = list(set(
        state.get("universe_tickers", []) + state.get("candidate_tickers", [])
    ))
    candidate_tickers = set(state.get("candidate_tickers", []))

    for ticker in all_tickers:
        category = "candidates" if ticker in candidate_tickers else "universe"
        thesis = investment_theses.get(ticker, {})

        archive.save_reports(
            ticker=ticker,
            run_date=run_date,
            news_report=state.get("news_reports", {}).get(ticker),
            research_report=state.get("research_reports", {}).get(ticker),
            thesis=thesis,
            category=category,
        )

        archive.write_investment_thesis(thesis, run_time=run_time)
        archive.write_agent_report(
            {"symbol": ticker, "date": run_date, "agent_type": "news",
             "report_md": state.get("news_reports", {}).get(ticker, "")},
            run_time=run_time,
        )
        archive.write_agent_report(
            {"symbol": ticker, "date": run_date, "agent_type": "research",
             "report_md": state.get("research_reports", {}).get(ticker, "")},
            run_time=run_time,
        )

        # Technical scores
        tech = state.get("technical_scores", {}).get(ticker, {})
        if tech:
            archive.write_technical_score(ticker, run_date, tech)

        # News hashes
        hashes_for_ticker = [
            a["article_hash"]
            for a in state.get("news_headlines", {}).get(ticker, [])
            if "article_hash" in a
        ]
        archive.upsert_news_hashes(ticker, hashes_for_ticker, run_date)

    # Macro
    archive.save_macro_report(run_date, state.get("macro_report", ""))
    archive.write_agent_report(
        {"symbol": None, "date": run_date, "agent_type": "macro",
         "report_md": state.get("macro_report", "")},
        run_time=run_time,
    )

    macro_data = state.get("macro_data", {})
    macro_data["market_regime"] = state.get("market_regime", "neutral")
    macro_data["sector_modifiers"] = state.get("sector_modifiers", {})
    macro_data["sector_ratings"] = state.get("sector_ratings", {})
    archive.write_macro_snapshot(run_date, macro_data)

    # Consolidated report
    archive.save_consolidated_report(run_date, state.get("consolidated_report", ""))
    archive.write_agent_report(
        {"symbol": None, "date": run_date, "agent_type": "consolidator",
         "report_md": state.get("consolidated_report", "")},
        run_time=run_time,
    )

    # Scanner appearances
    scanner_appearances = []
    for i, entry in enumerate(state.get("scanner_ranked", []), 1):
        ticker = entry.get("ticker", "")
        score_data = state.get("scanner_scores", {}).get(ticker, {})
        scanner_appearances.append({
            "symbol": ticker,
            "date": run_date,
            "scanner_rank": i,
            "scan_path": entry.get("path"),
            "tech_score": score_data.get("tech_score"),
            "news_score": score_data.get("news_score"),
            "research_score": score_data.get("research_score"),
            "final_score": score_data.get("score"),
            "selected": 1 if ticker in {c["symbol"] for c in state.get("new_candidates", [])} else 0,
        })
    archive.write_scanner_appearances(scanner_appearances)

    # Active candidates
    archive.save_active_candidates(state.get("new_candidates", []))

    # Performance tracker — record new BUY scores
    prices = {
        t: state.get("technical_scores", {}).get(t, {}).get("current_price")
        for t in all_tickers
    }
    if archive.db_conn:
        record_new_buy_scores(
            db_conn=archive.db_conn,
            today=run_date,
            investment_theses=investment_theses,
            price_data=prices,
        )

    # Write machine-readable signals.json for executor consumption (§A.1)
    new_candidates_for_signals = state.get("new_candidates", [])
    candidate_tickers_set = {c["symbol"] for c in new_candidates_for_signals}
    # Enrich sector_ratings with modifier field so executor has all sizing inputs
    _sector_modifiers = state.get("sector_modifiers", {})
    _sector_ratings_enriched = {
        sector: {**v, "modifier": round(_sector_modifiers.get(sector, 1.0), 3)}
        for sector, v in state.get("sector_ratings", {}).items()
    }
    signals_payload = {
        "market_regime": state.get("market_regime", "neutral"),
        "sector_ratings": _sector_ratings_enriched,
        "universe": [
            {
                "ticker": t,
                "sector": thesis.get("sector"),
                "rating": thesis.get("rating"),
                "score": thesis.get("final_score"),
                "score_delta_1d": thesis.get("score_delta"),
                "score_velocity_5d": thesis.get("score_velocity_5d"),
                "conviction": thesis.get("conviction", "stable"),
                "signal": thesis.get("signal", "HOLD"),
                "price_target_upside": thesis.get("price_target_upside"),
                "stale": bool(thesis.get("stale_days", 0) >= 5),
            }
            for t, thesis in investment_theses.items()
            if t not in candidate_tickers_set
        ],
        "buy_candidates": [
            {
                "ticker": c["symbol"],
                "sector": investment_theses.get(c["symbol"], {}).get("sector"),
                "rating": investment_theses.get(c["symbol"], {}).get("rating"),
                "score": investment_theses.get(c["symbol"], {}).get("final_score"),
                "score_delta_1d": investment_theses.get(c["symbol"], {}).get("score_delta"),
                "score_velocity_5d": investment_theses.get(c["symbol"], {}).get("score_velocity_5d"),
                "conviction": investment_theses.get(c["symbol"], {}).get("conviction", "stable"),
                "signal": investment_theses.get(c["symbol"], {}).get("signal", "HOLD"),
                "price_target_upside": investment_theses.get(c["symbol"], {}).get("price_target_upside"),
                "tenure_days": (
                    (datetime.strptime(run_date, "%Y-%m-%d").date()
                     - datetime.strptime(c["entry_date"], "%Y-%m-%d").date()).days
                    if c.get("entry_date") else None
                ),
            }
            for c in new_candidates_for_signals
        ],
    }
    archive.write_signals_json(run_date, run_time, signals_payload)

    # Write daily OHLCV price snapshot for backtester consumption
    price_data = state.get("price_data", {})
    prices_snapshot: dict[str, dict] = {}
    for ticker in all_tickers:
        df = price_data.get(ticker)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            prices_snapshot[ticker] = {
                "open": round(float(row["Open"]), 4),
                "close": round(float(row["Close"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
            }
    archive.write_prices_json(run_date, prices_snapshot)

    archive.commit()
    archive.upload_db(run_date=run_date)

    return state


def email_sender_node(state: ResearchState) -> ResearchState:
    """Format and send the daily email via AWS SES."""
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER

    run_date = state["run_date"]
    report = state.get("consolidated_report", "")

    day_of_week = datetime.strptime(run_date, "%Y-%m-%d").strftime("%A")
    is_early_close = state.get("is_early_close", False)
    early_close_tag = " [Early Close]" if is_early_close else ""
    subject = f"Research Brief — {run_date} {day_of_week}{early_close_tag}"

    html_body, plain_body = format_email(report, run_date)

    success = send_email(
        subject=subject,
        html_body=html_body,
        plain_body=plain_body,
        recipients=EMAIL_RECIPIENTS,
        sender=EMAIL_SENDER,
    )

    return {**state, "email_sent": success}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("fetch_data", fetch_data)
    graph.add_node("run_universe_agents", run_universe_agents)
    graph.add_node("run_scanner_pipeline", run_scanner_pipeline)
    graph.add_node("score_aggregator", score_aggregator)
    graph.add_node("thesis_updater", thesis_updater)
    graph.add_node("candidate_evaluator", candidate_evaluator)
    graph.add_node("consolidator_node", consolidator_node)
    graph.add_node("archive_writer", archive_writer)
    graph.add_node("email_sender_node", email_sender_node)

    graph.set_entry_point("fetch_data")

    # Both branches start after fetch_data
    graph.add_edge("fetch_data", "run_universe_agents")
    graph.add_edge("fetch_data", "run_scanner_pipeline")

    # Both branches feed into score_aggregator (fan-in)
    graph.add_edge("run_universe_agents", "score_aggregator")
    graph.add_edge("run_scanner_pipeline", "score_aggregator")

    # Sequential post-aggregation
    graph.add_edge("score_aggregator", "thesis_updater")
    graph.add_edge("thesis_updater", "candidate_evaluator")
    graph.add_edge("candidate_evaluator", "consolidator_node")
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
