"""
V2 Research Graph — Sector-Team Architecture with LangGraph Send() fan-out.

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
    ARCHITECTURE_VERSION,
)
from agents.sector_teams.team_config import (
    ALL_TEAM_IDS,
    TEAM_SECTORS,
    SECTOR_TEAM_MAP,
    compute_team_slots,
    get_team_tickers,
)
from agents.sector_teams.sector_team import run_sector_team
from agents.macro_agent import run_macro_agent_with_reflection
from agents.investment_committee.ic_cio import run_cio
from data.population_selector import (
    compute_exits_and_open_slots,
    apply_ic_entries,
)
from scoring.composite import compute_composite_score, score_to_rating
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


class ResearchStateV2(TypedDict, total=False):
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

def fetch_data_v2(state: ResearchStateV2) -> dict:
    """Load all shared data needed by sector teams, macro, and exit evaluator."""
    from data.fetchers.price_fetcher import (
        fetch_price_data, fetch_sp500_sp400_with_sectors, compute_technical_indicators,
    )
    from data.fetchers.macro_fetcher import fetch_macro_data, compute_market_breadth
    from scoring.technical import compute_technical_score

    run_date = state["run_date"]
    am: ArchiveManager = state["archive_manager"]

    logger.info("[v2:fetch_data] starting for %s", run_date)

    # Load S&P 900 universe
    scanner_universe, wikipedia_sector_map = fetch_sp500_sp400_with_sectors()
    logger.info("[v2:fetch_data] %d tickers in S&P 900 universe", len(scanner_universe))

    # Load current population
    current_population = am.load_population()
    population_tickers = [p["ticker"] for p in current_population]

    # Build sector map
    sector_map = dict(wikipedia_sector_map)
    for p in current_population:
        sector_map.setdefault(p["ticker"], p.get("sector", "Unknown"))

    # Fetch price data for all tickers needed
    all_tickers = list(set(population_tickers + scanner_universe))
    price_data = fetch_price_data(all_tickers, period="3mo")

    # Technical indicators for all tickers with price data
    technical_scores = {}
    for ticker, df in price_data.items():
        if df is not None and len(df) >= 20:
            indicators = compute_technical_indicators(df)
            ts = compute_technical_score(indicators)
            technical_scores[ticker] = {**indicators, "technical_score": ts}

    # Macro data
    macro_data = fetch_macro_data()
    breadth = compute_market_breadth(price_data)
    macro_data.update(breadth)

    # Load prior data
    prior_theses = {}
    for t in population_tickers:
        thesis = am.load_structured_thesis(t) if hasattr(am, 'load_structured_thesis') else None
        if thesis:
            prior_theses[t] = thesis

    prior_sector_ratings = state.get("prior_sector_ratings", {})

    # Load predictions
    predictions = {}
    try:
        pred_json = am.load_predictions_json()
        if pred_json:
            predictions = pred_json.get("predictions", {})
    except Exception:
        pass

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
        except Exception:
            pass

    logger.info("[v2:fetch_data] done — %d prices, %d tech scores, %d population",
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
    }


def dispatch_all(state: ResearchStateV2) -> list:
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

    logger.info("[v2:dispatch] sending %d parallel tasks (6 teams + macro + exits)", len(sends))
    return sends


def sector_team_node(state: ResearchStateV2) -> dict:
    """Run a single sector team (dispatched via Send)."""
    team_id = state.get("team_id", "unknown")
    logger.info("[v2:sector_team:%s] starting", team_id)

    result = run_sector_team(
        team_id=team_id,
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
    )

    # Return partial state update — _merge_dicts reducer merges team outputs
    return {
        "sector_team_outputs": {team_id: result},
    }


def macro_economist_node(state: ResearchStateV2) -> dict:
    """Run the macro economist with reflection."""
    logger.info("[v2:macro] starting")
    macro_data = state.get("macro_data", {})
    prior_report = state.get("prior_macro_report", "")
    prior_date = ""

    result = run_macro_agent_with_reflection(
        prior_report=prior_report,
        prior_date=prior_date,
        macro_data=macro_data,
    )

    return {
        "macro_report": result.get("report_md", ""),
        "sector_modifiers": result.get("sector_modifiers", {}),
        "sector_ratings": result.get("sector_ratings", {}),
        "market_regime": result.get("market_regime", "neutral"),
    }


def exit_evaluator_node(state: ResearchStateV2) -> dict:
    """Determine exits from current population using prior theses."""
    logger.info("[v2:exit_evaluator] starting")

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


def merge_results_node(state: ResearchStateV2) -> dict:
    """Fan-in: merge sector team outputs + macro + exits. Compute slot allocation."""
    logger.info("[v2:merge] merging results")

    sector_ratings = state.get("sector_ratings", {})
    open_slots = state.get("open_slots", 0)

    team_slot_allocation = compute_team_slots(open_slots, sector_ratings)

    logger.info("[v2:merge] %d open slots, allocation: %s", open_slots, team_slot_allocation)

    return {
        "team_slot_allocation": team_slot_allocation,
    }


def score_aggregator_v2(state: ResearchStateV2) -> dict:
    """Compute composite scores for all team recommendations."""
    logger.info("[v2:score_aggregator] starting")

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
                "conviction": rec.get("conviction", "medium"),
                "quant_rationale": rec.get("quant_rationale", ""),
                "rating": score_to_rating(score_result["final_score"]),
                "score_failed": score_result["score_failed"],
            }

        # Merge thesis updates from held stocks
        for ticker, thesis in output.get("thesis_updates", {}).items():
            if ticker not in investment_theses:
                investment_theses[ticker] = {
                    "ticker": ticker,
                    "team_id": team_id,
                    **thesis,
                }

    logger.info("[v2:score_aggregator] scored %d tickers", len(investment_theses))

    return {"investment_theses": investment_theses}


def cio_node(state: ResearchStateV2) -> dict:
    """Run CIO batch evaluation."""
    logger.info("[v2:cio] starting")

    # Collect all team recommendations as candidate list
    team_outputs = state.get("sector_team_outputs", {})
    candidates = []
    for team_id, output in team_outputs.items():
        for rec in output.get("recommendations", []):
            candidates.append({
                **rec,
                "team_id": team_id,
            })

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
    )

    return {
        "ic_decisions": cio_result.get("decisions", []),
        "advanced_tickers": cio_result.get("advanced_tickers", []),
        "entry_theses": cio_result.get("entry_theses", {}),
    }


def population_entry_handler(state: ResearchStateV2) -> dict:
    """Place IC ADVANCE decisions into population."""
    logger.info("[v2:entry_handler] starting")

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


def consolidator_v2(state: ResearchStateV2) -> dict:
    """Build the weekly research email with 4 structured sections."""
    logger.info("[v2:consolidator] starting")

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

    # ── Section 4: Universe Ratings ──────────────────────────────────────────
    sections.append("---\n")
    sections.append("## d. UNIVERSE RATINGS\n")

    current_pop = state.get("current_population", [])
    new_pop = state.get("new_population", [])
    current_tickers = {p["ticker"] for p in current_pop} if current_pop else set()
    new_tickers = {p["ticker"] for p in new_pop} if new_pop else set()

    continuing_tickers = current_tickers & new_tickers
    entrant_tickers = new_tickers - current_tickers
    exit_list = state.get("exits", [])

    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    entry_theses = state.get("entry_theses", {})
    team_outputs = state.get("sector_team_outputs", {})

    # Collect tickers with material thesis updates from sector teams
    updated_tickers = set()
    for team_id, output in team_outputs.items():
        for ticker in output.get("thesis_updates", {}):
            updated_tickers.add(ticker)

    # Build a lookup for population entry data (score, rating from prior week)
    pop_lookup = {p["ticker"]: p for p in new_pop}

    # 4a. Continuing Coverage
    if continuing_tickers:
        sections.append(f"### Continuing Coverage ({len(continuing_tickers)} stocks)\n")
        sections.append("| Ticker | Rating | Score | Rationale |")
        sections.append("|--------|--------|-------|-----------|")
        for ticker in sorted(continuing_tickers):
            thesis = theses.get(ticker, {})
            prior = prior_theses.get(ticker, {})
            pop_entry = pop_lookup.get(ticker, {})
            # Prefer fresh thesis, fall back to prior, then population
            rating = thesis.get("rating") or prior.get("rating") or pop_entry.get("long_term_rating", "HOLD")
            score = thesis.get("final_score") or prior.get("score") or pop_entry.get("long_term_score", 0)
            score_str = f"{score:.0f}" if score else "—"
            # Fresh thesis if material update occurred, otherwise carry over prior
            if ticker in updated_tickers and thesis.get("bull_case"):
                rationale = thesis.get("bull_case", "")
            elif prior.get("thesis_summary"):
                rationale = prior["thesis_summary"]
            elif thesis.get("bull_case"):
                rationale = thesis["bull_case"]
            else:
                rationale = "Continuing coverage — no material update"
            sections.append(f"| {ticker} | {rating} | {score_str} | {rationale} |")
        sections.append("")

    # 4b. Entrants
    if entrant_tickers:
        sections.append(f"### Entrants ({len(entrant_tickers)} stocks)\n")
        sections.append("| Ticker | Rating | Score | Rationale |")
        sections.append("|--------|--------|-------|-----------|")
        for ticker in sorted(entrant_tickers):
            thesis = theses.get(ticker, {})
            et = entry_theses.get(ticker, {})
            rating = thesis.get("rating", "BUY")
            score = thesis.get("final_score", 0)
            score_str = f"{score:.0f}" if score else "—"
            # Always fresh thesis for entrants — prefer CIO entry thesis
            rationale = et.get("bull_case") or thesis.get("bull_case", "New entry")
            sections.append(f"| {ticker} | {rating} | {score_str} | {rationale} |")
        sections.append("")

    # 4c. Exits
    if exit_list:
        sections.append(f"### Exits ({len(exit_list)} stocks)\n")
        sections.append("| Ticker | Score | Rationale |")
        sections.append("|--------|-------|-----------|")
        for e in exit_list:
            ticker = e.get("ticker_out", "?")
            score = e.get("score_out", 0)
            score_str = f"{score:.0f}" if score else "—"
            reason = e.get("reason", "Exited")
            sections.append(f"| {ticker} | {score_str} | {reason} |")
        sections.append("")

    # Footer
    sections.append("---\n")
    sections.append(f"*Brief generated: {run_date} | Portfolio: {len(new_pop)} stocks*")

    consolidated = "\n".join(sections)
    return {"consolidated_report": consolidated}


def _build_notable_developments(state: ResearchStateV2) -> list[str]:
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


def archive_writer_v2(state: ResearchStateV2) -> dict:
    """Write all data to S3 + SQLite."""
    logger.info("[v2:archive_writer] starting")
    am: ArchiveManager = state["archive_manager"]
    run_date = state["run_date"]

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
                except Exception:
                    pass

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

    # Upload DB
    try:
        am.upload_db(run_date)
    except Exception as e:
        logger.warning("Failed to upload DB: %s", e)

    return {}


def email_sender_v2(state: ResearchStateV2) -> dict:
    """Send the morning email with properly rendered HTML."""
    from emailer.sender import send_email
    from emailer.formatter import format_email
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER

    logger.info("[v2:email_sender] starting")
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


def _build_signals_payload(state: ResearchStateV2) -> dict:
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

    # v2 signals dict (keyed by ticker) — includes new theses AND population carryovers
    signals = {}
    # First: tickers with fresh theses from this run
    for ticker, thesis in theses.items():
        signals[ticker] = {
            "ticker": ticker,
            "score": thesis.get("final_score"),
            "rating": thesis.get("rating", "HOLD"),
            "signal": "ENTER" if thesis.get("rating") == "BUY" and ticker in pop_tickers else "HOLD",
            "conviction": thesis.get("conviction", "stable"),
            "thesis_summary": thesis.get("bull_case", ""),
            "sector": thesis.get("sector", "Unknown"),
            "team_id": thesis.get("team_id"),
            "quant_score": thesis.get("quant_score"),
            "qual_score": thesis.get("qual_score"),
        }
    # Second: population tickers without fresh theses — carry over from prior week
    for p in pop:
        ticker = p["ticker"]
        if ticker not in signals:
            prior = prior_theses.get(ticker, {})
            sector = sector_map.get(ticker, p.get("sector", "Unknown"))
            signals[ticker] = {
                "ticker": ticker,
                "score": prior.get("score") or p.get("long_term_score"),
                "rating": prior.get("rating") or p.get("long_term_rating", "HOLD"),
                "signal": "HOLD",
                "conviction": prior.get("conviction") or p.get("conviction", "stable"),
                "thesis_summary": prior.get("thesis_summary", ""),
                "sector": sector,
                "team_id": prior.get("team_id"),
                "quant_score": prior.get("quant_score"),
                "qual_score": prior.get("qual_score"),
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
        "architecture_version": "v2_sector_teams",
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph_v2() -> StateGraph:
    """
    V2 sector-team graph with Send() fan-out.

    Topology:
      fetch_data → dispatch_all (Send: 6 teams + macro + exit)
      → merge_results → score_aggregator → cio_node
      → population_entry_handler → consolidator → archive → email → END
    """
    graph = StateGraph(ResearchStateV2)

    # Nodes
    graph.add_node("fetch_data", fetch_data_v2)
    graph.add_node("sector_team_node", sector_team_node)
    graph.add_node("macro_economist_node", macro_economist_node)
    graph.add_node("exit_evaluator_node", exit_evaluator_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("score_aggregator", score_aggregator_v2)
    graph.add_node("cio_node", cio_node)
    graph.add_node("population_entry_handler", population_entry_handler)
    graph.add_node("consolidator_node", consolidator_v2)
    graph.add_node("archive_writer", archive_writer_v2)
    graph.add_node("email_sender_node", email_sender_v2)

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


def create_initial_state_v2(
    run_date: str,
    archive_manager: ArchiveManager,
    is_early_close: bool = False,
) -> ResearchStateV2:
    return ResearchStateV2(
        run_date=run_date,
        run_time=datetime.now(timezone.utc).isoformat(),
        archive_manager=archive_manager,
        is_early_close=is_early_close,
        email_sent=False,
    )
