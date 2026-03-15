"""
LangGraph research graph — scanner-driven population pipeline.

Topology:
  fetch_data                  (download macro + S&P 900 price data + load population)
  run_scanner_pipeline        (quant filter 900→~50, ranking →30-35, deep analysis)
  run_population_agents       (LLM agents: news + research for ~25 population stocks)
  score_aggregator            (aggregate all scores)
  population_evaluator        (sector-balanced population selection + rotation)
  consolidator_node
  archive_writer              (S3 + SQLite + population/latest.json)

All stocks derived from S&P 900 — no hardcoded universe.
Sector allocation driven by macro agent's sector_modifiers.
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
    POPULATION_CFG,
    CONCURRENT_AGENTS,
    ROTATION_TIERS,
    WEAK_PICK_SCORE_THRESHOLD,
    WEAK_PICK_CONSECUTIVE_RUNS,
    EMERGENCY_ROTATION_NEW_SCORE,
    DEEP_VALUE_MIN_CONSENSUS,
    MIN_PREDICTION_CONFIDENCE,
)
from agents.news_agent import run_news_agent
from agents.research_agent import run_research_agent
from agents.macro_agent import run_macro_agent
from agents.scanner_ranking_agent import run_scanner_ranking_agent
from agents.consolidator import run_consolidator_agent
from data.fetchers.price_fetcher import (
    fetch_price_data,
    compute_technical_indicators,
    fetch_sp500_sp400_with_sectors,
)
from data.fetchers.news_fetcher import fetch_all_news
from data.fetchers.analyst_fetcher import fetch_analyst_consensus
from data.fetchers.macro_fetcher import fetch_macro_data
from data.deduplicator import deduplicate_articles, compute_recurring_themes
from data.scanner import (
    get_scanner_universe,
    run_quant_filter,
    confirm_deep_value_with_analyst,
)
from data.population_selector import select_population
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
    predictions: Annotated[dict[str, dict], _take_last]
    news_reports: Annotated[dict[str, str], _take_last]
    news_scores: Annotated[dict[str, float], _take_last]
    news_scores_lt: Annotated[dict[str, float], _take_last]
    news_jsons: Annotated[dict[str, dict], _take_last]
    research_reports: Annotated[dict[str, str], _take_last]
    research_scores: Annotated[dict[str, float], _take_last]
    research_scores_lt: Annotated[dict[str, float], _take_last]
    research_jsons: Annotated[dict[str, dict], _take_last]
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
    # ── Population-driven fields ──────────────────────────────────────────────
    current_population: Annotated[list[dict], _take_last]  # from SQLite
    population_tickers: Annotated[list[str], _take_last]   # tickers to analyze
    new_population: Annotated[list[dict], _take_last]       # after selection
    population_rotation_events: Annotated[list[dict], _take_last]
    sector_map: Annotated[dict[str, str], _take_last]       # dynamic sector map


# ── Semaphore for LLM concurrency ─────────────────────────────────────────────
_llm_semaphore = asyncio.Semaphore(CONCURRENT_AGENTS)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def fetch_data(state: ResearchState) -> ResearchState:
    """
    Node 1: Download all data needed for the scanner-driven pipeline.
    - Load current population from SQLite (empty on first run)
    - Price data for S&P 500/400 (~900 stocks)
    - Macro data from FRED + yfinance
    - Prior reports from archive for population tickers
    - Predictor predictions from S3
    """
    run_date = state["run_date"]
    archive: ArchiveManager = state["archive_manager"]

    # Load current population from SQLite (empty list on first run)
    current_population = archive.load_population()
    population_tickers = [p["ticker"] for p in current_population]
    logger.info("[fetch_data] loaded %d stocks from current population", len(current_population))

    # Also load legacy active candidates for backward compat during transition
    active_candidates = archive.load_active_candidates()
    candidate_tickers = [c["symbol"] for c in active_candidates]

    # Combine all known tickers for prior report loading
    all_tracked = list(set(population_tickers + candidate_tickers + UNIVERSE_TICKERS))

    # Fetch S&P 900 tickers WITH sector data from Wikipedia
    # This builds the sector_map needed for scanner and population selection
    scanner_universe, wikipedia_sector_map = fetch_sp500_sp400_with_sectors()
    all_tickers_to_fetch = list(set(all_tracked + scanner_universe))

    price_data = fetch_price_data(all_tickers_to_fetch, period="1y")

    # Fetch macro data (needed before scanner for market_regime)
    try:
        macro_data = fetch_macro_data()
    except Exception as e:
        macro_data = {"error": str(e)}

    # Load prior reports and news for population tickers (they'll get LLM analysis)
    # News/analyst data will be fetched later for scanner-selected candidates
    prior_news_reports: dict[str, str] = {}
    prior_research_reports: dict[str, str] = {}
    prior_theses: dict[str, dict] = {}

    for ticker in all_tracked:
        # All population stocks are archived under "universe" category
        reports = archive.load_prior_reports(ticker, category="universe")
        if not reports["news_report"]:
            # Also check candidates category (transition period)
            reports = archive.load_prior_reports(ticker, category="candidates")
        if reports["news_report"]:
            prior_news_reports[ticker] = reports["news_report"]
        if reports["research_report"]:
            prior_research_reports[ticker] = reports["research_report"]
        if reports["thesis"]:
            prior_theses[ticker] = reports["thesis"]

    prior_macro_report_data = archive.load_prior_reports("macro_global", category="macro")
    prior_macro_report = prior_macro_report_data.get("news_report") or ""

    # Load score history for conviction and velocity computation
    score_history = archive.load_score_history(all_tracked, n=6)

    # Load predictor predictions from S3 (best-effort — {} if not yet available)
    try:
        predictions = archive.load_predictions_json()
    except Exception:
        predictions = {}

    # Build dynamic sector map: Wikipedia sectors (primary) + population sectors (overlay)
    sector_map: dict[str, str] = dict(SECTOR_MAP)  # start with any static entries (empty now)
    sector_map.update(wikipedia_sector_map)          # add all ~900 sectors from Wikipedia
    for p in current_population:
        # Population sectors override Wikipedia (in case of manual corrections)
        if p.get("sector") and p["sector"] != "Unknown":
            sector_map[p["ticker"]] = p["sector"]

    return {
        **state,
        "current_population": current_population,
        "population_tickers": population_tickers,
        "universe_tickers": population_tickers,  # backward compat
        "candidate_tickers": candidate_tickers,
        "active_candidates": active_candidates,
        "price_data": price_data,
        "news_headlines": {},   # fetched per-ticker in scanner/agent nodes
        "news_article_hashes": {},
        "analyst_data": {},
        "macro_data": macro_data,
        "prior_news_reports": prior_news_reports,
        "prior_research_reports": prior_research_reports,
        "prior_theses": prior_theses,
        "prior_macro_report": prior_macro_report,
        "score_history": score_history,
        "_scanner_universe": scanner_universe,
        "predictions": predictions,
        "sector_map": sector_map,
    }


def run_population_agents(state: ResearchState) -> ResearchState:
    """
    Run LLM agents (news + research) on the population-candidate tickers
    identified by the scanner.  Also runs the macro agent (global).

    The scanner has already reduced S&P 900 → ~30-35 ranked candidates.
    This node runs full LLM analysis on those candidates to produce
    long_term_score / long_term_rating values for population selection.
    """
    run_date = state["run_date"]
    archive: ArchiveManager = state["archive_manager"]

    # Population tickers to analyze = scanner-ranked candidates (30-35 stocks).
    # These include re-scored incumbents and potential new entries.
    scanner_ranked = state.get("scanner_ranked", [])
    population_tickers = state.get("population_tickers", [])

    # Combine scanner-ranked + current population (incumbents must be re-scored)
    tickers_to_analyze = list(set(
        [e.get("ticker", "") for e in scanner_ranked if e.get("ticker")]
        + population_tickers
    ))
    logger.info("[population_agents] analyzing %d tickers (scanner=%d + population=%d)",
                len(tickers_to_analyze), len(scanner_ranked), len(population_tickers))

    news_reports: dict[str, str] = {}
    news_scores: dict[str, float] = {}
    news_scores_lt: dict[str, float] = {}
    news_jsons: dict[str, dict] = {}
    research_reports: dict[str, str] = {}
    research_scores: dict[str, float] = {}
    research_scores_lt: dict[str, float] = {}
    research_jsons: dict[str, dict] = {}

    # Compute technical scores for analyzed tickers
    price_data = state.get("price_data", {})
    technical_scores: dict[str, dict] = state.get("technical_scores", {})
    momentum_data: dict[str, Optional[float]] = {}

    for ticker in tickers_to_analyze:
        if ticker not in technical_scores:
            df = price_data.get(ticker)
            if df is not None and not df.empty:
                indicators = compute_technical_indicators(df)
                if indicators:
                    technical_scores[ticker] = indicators
        if ticker in technical_scores:
            momentum_data[ticker] = technical_scores[ticker].get("momentum_20d")

    # Compute momentum percentiles across all available tickers
    momentum_percentiles = compute_momentum_percentiles(momentum_data)

    # Macro agent — determines market regime and sector ratings
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
    for ticker in tickers_to_analyze:
        if ticker in technical_scores:
            percentile = momentum_percentiles.get(ticker)
            ts = compute_technical_score(
                technical_scores[ticker],
                market_regime=market_regime,
                momentum_percentile=percentile,
            )
            technical_scores[ticker]["technical_score"] = ts

    # Fetch news, analyst data, and run LLM agents for each ticker
    news_headlines: dict[str, list] = {}
    news_article_hashes: dict[str, set] = {}
    analyst_data: dict[str, dict] = {}

    prior_date_default = "NONE"

    def run_ticker_agents(ticker: str):
        # Fetch news
        try:
            raw = fetch_all_news(ticker, hours=48)
            articles = raw["yahoo"] + raw.get("edgar_8k", [])
        except Exception:
            articles = []
        news_headlines[ticker] = articles
        news_article_hashes[ticker] = archive.load_news_hashes(ticker)

        # Fetch analyst data
        try:
            analyst_data[ticker] = fetch_analyst_consensus(ticker)
        except Exception as e:
            analyst_data[ticker] = {"ticker": ticker, "error": str(e)}

        # Deduplicate articles
        prior_hashes = news_article_hashes.get(ticker, set())
        novel_articles, _ = deduplicate_articles(articles, prior_hashes)
        recurring = compute_recurring_themes(novel_articles)

        prior_news = state.get("prior_news_reports", {}).get(ticker)
        prior_research = state.get("prior_research_reports", {}).get(ticker)
        prior_thesis = state.get("prior_theses", {}).get(ticker, {})
        prior_date = prior_thesis.get("date", prior_date_default)

        tech = technical_scores.get(ticker, {})
        # Merge predictor values if available
        pred = state.get("predictions", {}).get(ticker, {})
        if pred and tech:
            tech.update({
                "p_up": pred.get("p_up"),
                "p_flat": pred.get("p_flat"),
                "p_down": pred.get("p_down"),
                "prediction_confidence": pred.get("prediction_confidence", 0.0),
                "predicted_direction": pred.get("predicted_direction"),
            })
        current_price = tech.get("current_price", 0)

        # News agent
        try:
            news_result = run_news_agent(
                ticker=ticker,
                company_name=ticker,
                prior_report=prior_news,
                prior_date=prior_date,
                new_articles=novel_articles,
                recurring_themes=recurring,
                sec_filings=[a for a in articles if a.get("source") == "SEC EDGAR"],
                current_price=current_price,
                price_change_pct=tech.get("momentum_20d", 0) or 0,
                price_change_date=run_date,
            )
            news_reports[ticker] = news_result["report_md"]
            news_scores[ticker] = news_result["news_score"]
            news_scores_lt[ticker] = news_result["news_score_lt"]
            news_jsons[ticker] = news_result["news_json"]
        except Exception as e:
            news_reports[ticker] = f"Error: {e}"
            news_scores[ticker] = 50.0
            news_scores_lt[ticker] = 50.0
            news_jsons[ticker] = {}

        # Research agent
        try:
            research_result = run_research_agent(
                ticker=ticker,
                company_name=ticker,
                prior_report=prior_research,
                prior_date=prior_date,
                analyst_data=analyst_data.get(ticker, {}),
            )
            research_reports[ticker] = research_result["report_md"]
            research_scores[ticker] = research_result["research_score"]
            research_scores_lt[ticker] = research_result["research_score_lt"]
            research_jsons[ticker] = research_result["research_json"]
        except Exception as e:
            research_reports[ticker] = f"Error: {e}"
            research_scores[ticker] = 50.0
            research_scores_lt[ticker] = 50.0
            research_jsons[ticker] = {}

    for ticker in tickers_to_analyze:
        run_ticker_agents(ticker)

    return {
        "technical_scores": technical_scores,
        "news_reports": news_reports,
        "news_scores": news_scores,
        "news_scores_lt": news_scores_lt,
        "news_jsons": news_jsons,
        "news_headlines": news_headlines,
        "news_article_hashes": news_article_hashes,
        "analyst_data": analyst_data,
        "research_reports": research_reports,
        "research_scores": research_scores,
        "research_scores_lt": research_scores_lt,
        "research_jsons": research_jsons,
        "macro_report": macro_report,
        "sector_modifiers": sector_modifiers,
        "sector_ratings": sector_ratings,
        "market_regime": market_regime,
        "universe_tickers": tickers_to_analyze,  # all analyzed tickers
    }


def run_scanner(
    run_date: str,
    scanner_universe: list[str],
    price_data: dict,
    archive: ArchiveManager,
    technical_scores: Optional[dict] = None,
    market_regime: str = "neutral",
    sector_map: Optional[dict] = None,
) -> dict:
    """
    Scanner pipeline (Stages 1–3). Pure function — no LangGraph state.

    Expanded for population-based architecture:
    - Stage 1: Quant filter (900 → ~60)
    - Stage 2: Data enrichment (analyst + headlines)
    - Stage 3: Ranking agent (→ top 35 for population selection buffer)

    LLM-based deep analysis (news + research agents) is now handled by
    run_population_agents() AFTER this scanner stage, rather than here.

    Returns dict with keys:
      scanner_filtered, scanner_ranked, scanner_scores
    """
    technical_scores = technical_scores or {}
    sector_map = sector_map or SECTOR_MAP

    # Stage 1: Quant filter
    candidates = run_quant_filter(
        tickers=scanner_universe,
        price_data=price_data,
        technical_scores=technical_scores,
        market_regime=market_regime,
    )
    logger.info("[scanner] stage=1 quant_filter candidates=%d universe=%d", len(candidates), len(scanner_universe))

    # Stage 2: Data enrichment — fetch analyst + brief news for ranking
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
        except Exception:
            c["headlines"] = []

        # Use dynamic sector map
        c["sector"] = sector_map.get(ticker, "Unknown")

    candidates = confirm_deep_value_with_analyst(
        candidates,
        analyst_data=analyst_data_scanner,
        min_consensus=DEEP_VALUE_MIN_CONSENSUS,
    )
    logger.info("[scanner] stage=2 confirmed_candidates=%d", len(candidates))

    # Stage 3: Ranking agent — expand from top 10 to top 35
    # Need enough candidates to fill population (25) + buffer for sector balance
    ranked = run_scanner_ranking_agent(
        candidates=candidates,
        market_regime=market_regime,
        top_n=35,  # expanded from 10 for population coverage
    )
    logger.info("[scanner] stage=3 ranked=%d", len(ranked))

    # Build preliminary scanner scores (tech-only; LLM scores come after agents run)
    scanner_scores: dict[str, dict] = {}
    for entry in ranked:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue
        tech = technical_scores.get(ticker) or {}
        scanner_scores[ticker] = {
            "ticker": ticker,
            "tech_score": tech.get("technical_score", 50.0),
            "sector": sector_map.get(ticker, "Unknown"),
            "rank": entry.get("rank"),
            "path": entry.get("path", "momentum"),
        }

    logger.info("[scanner] stage=3 scored=%d tickers", len(scanner_scores))
    return {
        "scanner_filtered": candidates,
        "scanner_ranked": ranked,
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
        sector_map=state.get("sector_map", {}),
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
    Aggregate scores for all analyzed tickers (population + scanner candidates).
    """
    run_date = state["run_date"]
    all_tickers = list(set(state.get("universe_tickers", [])))

    # Precompute price target upside from analyst data
    price_target_upside = _compute_price_target_upside(
        analyst_data=state.get("analyst_data", {}),
        technical_scores=state.get("technical_scores", {}),
    )

    # Aggregate all scores (same formula applies to all tickers now)
    aggregated = aggregate_all(
        tickers=all_tickers,
        technical_scores=state.get("technical_scores", {}),
        news_scores=state.get("news_scores", {}),
        research_scores=state.get("research_scores", {}),
        sector_modifiers=state.get("sector_modifiers", {}),
        prior_theses=state.get("prior_theses", {}),
        sector_map=state.get("sector_map", {}),  # dynamic sector map from Wikipedia
        run_date=run_date,
        score_history=state.get("score_history", {}),
        price_target_upside=price_target_upside,
        news_scores_lt=state.get("news_scores_lt", {}),
        research_scores_lt=state.get("research_scores_lt", {}),
    )

    # Update scanner_scores with sector from dynamic map
    scanner_scores = state.get("scanner_scores", {})
    sector_map = state.get("sector_map", {})
    sector_modifiers = state.get("sector_modifiers", {})
    for ticker, sdata in scanner_scores.items():
        sector = sector_map.get(ticker, sdata.get("sector", "Unknown"))
        sdata["sector"] = sector

    return {**state, "investment_theses": aggregated, "scanner_scores": scanner_scores}


def thesis_updater(state: ResearchState) -> ResearchState:
    """
    Build full thesis records for all universe + candidate tickers.
    """
    run_date = state["run_date"]
    investment_theses = state.get("investment_theses", {})
    full_theses: dict[str, dict] = {}

    news_jsons = state.get("news_jsons", {})
    research_jsons = state.get("research_jsons", {})
    predictions = state.get("predictions", {})

    for ticker, aggregated in investment_theses.items():
        thesis = build_thesis_record(
            ticker=ticker,
            run_date=run_date,
            aggregated=aggregated,
            agent_outputs={
                "news_json": news_jsons.get(ticker, {}),
                "research_json": research_jsons.get(ticker, {}),
            },
        )
        # Inject predictor fields if available
        pred = predictions.get(ticker, {})
        thesis["predicted_direction"] = pred.get("predicted_direction")
        thesis["prediction_confidence"] = pred.get("prediction_confidence")
        thesis["predicted_alpha"] = pred.get("predicted_alpha")

        # ── Option A: GBM confirmation gate ──────────────────────────────────
        # If the research signal is ENTER but the GBM model forecasts DOWN with
        # sufficient confidence, downgrade the signal to HOLD.  This prevents
        # entering a long position against a short-term bearish GBM signal.
        # The veto only fires when:
        #   1. signal == "ENTER"          (research wants to open a position)
        #   2. predicted_direction == "DOWN"   (GBM 5-day forecast is bearish)
        #   3. prediction_confidence >= MIN_PREDICTION_CONFIDENCE  (0.60 default)
        gbm_dir = pred.get("predicted_direction")
        gbm_conf = float(pred.get("prediction_confidence") or 0.0)
        if (
            thesis.get("signal") == "ENTER"
            and gbm_dir == "DOWN"
            and gbm_conf >= MIN_PREDICTION_CONFIDENCE
        ):
            thesis["signal"] = "HOLD"
            thesis["gbm_veto"] = True
            logger.info(
                "[GBM gate] ENTER → HOLD for %s: predicted_direction=DOWN confidence=%.2f",
                ticker, gbm_conf,
            )
        else:
            thesis["gbm_veto"] = False

        full_theses[ticker] = thesis

    return {**state, "investment_theses": full_theses}


def population_evaluator(state: ResearchState) -> ResearchState:
    """
    Build the sector-balanced investment population using the population selector.

    Takes scored candidates (from scanner + LLM agents) and the current
    population, applies sector allocation rules from macro agent ratings,
    and produces the new population with rotation events.
    """
    run_date = state["run_date"]
    investment_theses = state.get("investment_theses", {})
    current_population = state.get("current_population", [])
    sector_ratings = state.get("sector_ratings", {})
    sector_modifiers = state.get("sector_modifiers", {})
    sector_map = state.get("sector_map", {})

    # Build scored candidates from investment_theses (all analyzed tickers)
    scored_candidates: list[dict] = []
    for ticker, thesis in investment_theses.items():
        scored_candidates.append({
            "ticker": ticker,
            "sector": sector_map.get(ticker, thesis.get("sector", "Unknown")),
            "long_term_score": thesis.get("long_term_score", 50.0),
            "long_term_rating": thesis.get("long_term_rating", "HOLD"),
            "conviction": thesis.get("conviction", "stable"),
            "price_target_upside": thesis.get("price_target_upside"),
            "thesis_summary": thesis.get("thesis_summary", ""),
            "sub_scores": {
                "news_lt": thesis.get("news_score_lt", 50.0),
                "research_lt": thesis.get("research_score_lt", 50.0),
            },
            # Pass through all thesis fields for archive
            "final_score": thesis.get("final_score"),
            "news_score": thesis.get("news_score"),
            "research_score": thesis.get("research_score"),
            "technical_score": thesis.get("technical_score"),
        })

    # Enrich sector_ratings with modifier values for population selector
    enriched_sector_ratings = {}
    for sector, rating_data in sector_ratings.items():
        enriched_sector_ratings[sector] = {
            **rating_data,
            "modifier": sector_modifiers.get(sector, 1.0),
        }

    # Run population selection
    new_population, rotation_events = select_population(
        scored_candidates=scored_candidates,
        current_population=current_population,
        sector_ratings=enriched_sector_ratings,
        config=POPULATION_CFG,
        run_date=run_date,
    )

    # Update sector_map with any new tickers
    for p in new_population:
        sector_map[p["ticker"]] = p.get("sector", "Unknown")

    logger.info(
        "[population_evaluator] population=%d rotations=%d events=%s",
        len(new_population), len(rotation_events),
        [e.get("type") for e in rotation_events],
    )

    # Build backward-compatible new_candidates for consolidator
    new_candidates = []
    for p in new_population:
        thesis = investment_theses.get(p["ticker"], {})
        new_candidates.append({
            "symbol": p["ticker"],
            "entry_date": p.get("entry_date", run_date),
            "score": thesis.get("final_score", p.get("long_term_score", 50.0)),
            "score_delta": thesis.get("score_delta"),
            "thesis_summary": thesis.get("thesis_summary", ""),
            "key_catalyst": thesis.get("key_catalyst", ""),
            "key_risk": thesis.get("key_risk", ""),
            "status": "CONTINUING" if p.get("tenure_weeks", 0) > 0 else "NEW_ENTRY",
        })

    return {
        **state,
        "new_population": new_population,
        "population_rotation_events": rotation_events,
        "new_candidates": new_candidates,
        "_rotation_events": rotation_events,
        "sector_map": sector_map,
    }


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
    """Write all outputs to S3 and SQLite, including population data."""
    run_date = state["run_date"]
    run_time = state["run_time"]
    archive: ArchiveManager = state["archive_manager"]
    investment_theses = state.get("investment_theses", {})
    new_population = state.get("new_population", [])
    population_tickers = {p["ticker"] for p in new_population}
    all_tickers = list(set(state.get("universe_tickers", [])))

    # Save reports for all analyzed tickers (all under "universe" category now)
    for ticker in all_tickers:
        thesis = investment_theses.get(ticker, {})

        archive.save_reports(
            ticker=ticker,
            run_date=run_date,
            news_report=state.get("news_reports", {}).get(ticker),
            research_report=state.get("research_reports", {}).get(ticker),
            thesis=thesis,
            category="universe",  # all stocks use "universe" category
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
            "selected": 1 if ticker in population_tickers else 0,
        })
    archive.write_scanner_appearances(scanner_appearances)

    # Build enriched sector ratings (used by population JSON + signals.json)
    _sector_modifiers = state.get("sector_modifiers", {})
    _sector_ratings_enriched = {
        sector: {**v, "modifier": round(_sector_modifiers.get(sector, 1.0), 3)}
        for sector, v in state.get("sector_ratings", {}).items()
    }

    # ── Population persistence ──────────────────────────────────────────────
    # Save population to SQLite + S3 (population/latest.json + population/{date}.json)
    # Include market_regime + sector_ratings so Executor's population_reader can use them
    population_for_save = _build_population_records(state, new_population)
    archive.save_population(
        population_for_save,
        run_date,
        market_regime=state.get("market_regime", "neutral"),
        sector_ratings=_sector_ratings_enriched,
    )

    # Log rotation events
    for event in state.get("population_rotation_events", []):
        archive.log_rotation_event(event, run_date)

    # Legacy: also save active_candidates for backward compatibility
    legacy_candidates = [
        {
            "symbol": p["ticker"],
            "entry_date": p.get("entry_date", run_date),
            "slot": i + 1,
            "score": investment_theses.get(p["ticker"], {}).get("final_score", 50.0),
            "consecutive_low_runs": 0,
        }
        for i, p in enumerate(new_population[:3])  # first 3 for legacy compat
    ]
    archive.save_active_candidates(legacy_candidates)

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

    # ── signals.json (backward compatible for Executor + Predictor) ────────

    # In the new architecture, all population stocks go into "universe"
    # and buy_candidates is populated from population with BUY ratings
    signals_payload = {
        "market_regime": state.get("market_regime", "neutral"),
        "sector_ratings": _sector_ratings_enriched,
        "universe": [
            _build_signal_entry(ticker, investment_theses.get(ticker, {}), run_date=run_date)
            for ticker in sorted(population_tickers)
            if ticker in investment_theses
        ],
        "buy_candidates": [
            _build_signal_entry(ticker, investment_theses.get(ticker, {}), run_date=run_date)
            for ticker in sorted(population_tickers)
            if investment_theses.get(ticker, {}).get("long_term_rating") == "BUY"
        ],
    }
    archive.write_signals_json(run_date, run_time, signals_payload)

    # Write daily OHLCV price snapshot for backtester consumption
    price_data = state.get("price_data", {})
    prices_snapshot: dict[str, dict] = {}
    for ticker in population_tickers:
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


def _build_population_records(state: ResearchState, population: list[dict]) -> list[dict]:
    """Build population records for SQLite + S3 persistence."""
    investment_theses = state.get("investment_theses", {})
    sector_map = state.get("sector_map", {})
    records = []
    for p in population:
        ticker = p["ticker"]
        thesis = investment_theses.get(ticker, {})
        records.append({
            "ticker": ticker,
            "long_term_score": p.get("long_term_score", thesis.get("long_term_score", 50.0)),
            "long_term_rating": p.get("long_term_rating", thesis.get("long_term_rating", "HOLD")),
            "sector": p.get("sector", sector_map.get(ticker, "Unknown")),
            "conviction": p.get("conviction", thesis.get("conviction", "stable")),
            "price_target_upside": p.get("price_target_upside", thesis.get("price_target_upside")),
            "thesis_summary": p.get("thesis_summary", thesis.get("thesis_summary", "")),
            "entry_date": p.get("entry_date"),
            "tenure_weeks": p.get("tenure_weeks", 0),
        })
    return records


def _build_signal_entry(ticker: str, thesis: dict, run_date: str = "") -> dict:
    """Build a single signal entry dict for signals.json."""
    return {
        "ticker": ticker,
        "sector": thesis.get("sector"),
        "rating": thesis.get("rating"),
        "score": thesis.get("final_score"),
        "score_delta_1d": thesis.get("score_delta"),
        "score_velocity_5d": thesis.get("score_velocity_5d"),
        "conviction": thesis.get("conviction", "stable"),
        "signal": thesis.get("signal", "HOLD"),
        "price_target_upside": thesis.get("price_target_upside"),
        "stale": bool(thesis.get("stale_days", 0) >= 5),
        "long_term_score": thesis.get("long_term_score"),
        "long_term_rating": thesis.get("long_term_rating"),
        "sub_scores": {
            "technical": thesis.get("technical_score"),
            "news": thesis.get("news_score"),
            "news_lt": thesis.get("news_score_lt"),
            "research": thesis.get("research_score"),
            "research_lt": thesis.get("research_score_lt"),
        },
        "predicted_direction": thesis.get("predicted_direction"),
        "prediction_confidence": thesis.get("prediction_confidence"),
        "predicted_alpha": thesis.get("predicted_alpha"),
        "gbm_veto": thesis.get("gbm_veto", False),
    }


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
    """
    Scanner-driven population pipeline:
      fetch_data → run_scanner_pipeline → run_population_agents
      → score_aggregator → thesis_updater → population_evaluator
      → consolidator_node → archive_writer → email_sender_node → END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("fetch_data", fetch_data)
    graph.add_node("run_scanner_pipeline", run_scanner_pipeline)
    graph.add_node("run_population_agents", run_population_agents)
    graph.add_node("score_aggregator", score_aggregator)
    graph.add_node("thesis_updater", thesis_updater)
    graph.add_node("population_evaluator", population_evaluator)
    graph.add_node("consolidator_node", consolidator_node)
    graph.add_node("archive_writer", archive_writer)
    graph.add_node("email_sender_node", email_sender_node)

    graph.set_entry_point("fetch_data")

    # Sequential pipeline: scan first, then run LLM agents on selected tickers
    graph.add_edge("fetch_data", "run_scanner_pipeline")
    graph.add_edge("run_scanner_pipeline", "run_population_agents")
    graph.add_edge("run_population_agents", "score_aggregator")

    # Sequential post-aggregation
    graph.add_edge("score_aggregator", "thesis_updater")
    graph.add_edge("thesis_updater", "population_evaluator")
    graph.add_edge("population_evaluator", "consolidator_node")
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
