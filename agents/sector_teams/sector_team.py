"""
Sector Team Orchestrator — wires quant + qual + peer review into one execution unit.

Each team:
  1. Quant screens sector → top 10
  2. Qual reviews quant's top 5 → qual scores + 0-1 additions
  3. Peer review → final 2-3 recommendations
  4. Thesis maintenance for held stocks with material triggers

All 6 teams run in parallel via LangGraph Send().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config import (
    PER_STOCK_MODEL, ANTHROPIC_API_KEY, TEAM_PICKS_PER_RUN,
)
from agents.sector_teams.team_config import (
    TEAM_SECTORS, TEAM_SCREENING_PARAMS, get_team_tickers,
)
from agents.prompt_loader import load_prompt
from agents.sector_teams.quant_analyst import run_quant_analyst
from agents.sector_teams.qual_analyst import run_qual_analyst
from agents.sector_teams.peer_review import run_peer_review
from agents.sector_teams.material_triggers import check_material_triggers
from thesis.structured import build_structured_thesis, format_structured_thesis_for_prompt

log = logging.getLogger(__name__)


@dataclass
class SectorTeamContext:
    """Bundled context for a sector team run — avoids 17-parameter function signatures."""
    scanner_universe: list[str]
    sector_map: dict[str, str]
    price_data: dict[str, Any]
    technical_scores: dict[str, dict]
    market_regime: str
    prior_theses: dict[str, dict]
    held_tickers: list[str]
    news_data_by_ticker: dict[str, Any]
    analyst_data_by_ticker: dict[str, Any]
    insider_data_by_ticker: dict[str, Any]
    prior_sector_ratings: dict[str, dict]
    current_sector_ratings: dict[str, dict]
    run_date: str
    api_key: str | None = None
    episodic_memories: dict[str, list] = field(default_factory=dict)
    semantic_memories: dict[str, list] = field(default_factory=dict)


def run_sector_team(team_id: str, ctx: SectorTeamContext) -> dict:
    """
    Run the full sector team pipeline.

    Returns:
        {
            "team_id": str,
            "recommendations": list[dict],  # final 2-3 picks with quant+qual scores
            "thesis_updates": dict[str, dict],  # updated theses for held stocks
            "quant_output": dict,  # full quant analyst output
            "qual_output": dict,  # full qual analyst output
            "peer_review_output": dict,
            "tool_calls": list[dict],  # combined tool call log
        }
    """
    log.info("[team:%s] starting — %d universe, %d held",
             team_id, len(ctx.scanner_universe), len(ctx.held_tickers))

    # ── Step 1: Get sector tickers ────────────────────────────────────────────
    sector_tickers = get_team_tickers(team_id, ctx.scanner_universe, ctx.sector_map)
    log.info("[team:%s] %d tickers in sector", team_id, len(sector_tickers))

    if not sector_tickers:
        log.warning("[team:%s] no tickers in sector — skipping", team_id)
        return _empty_result(team_id)

    # ── Step 2: Quant analyst screens sector ──────────────────────────────────
    quant_output = run_quant_analyst(
        team_id=team_id,
        sector_tickers=sector_tickers,
        market_regime=ctx.market_regime,
        price_data=ctx.price_data,
        technical_scores=ctx.technical_scores,
        run_date=ctx.run_date,
        api_key=ctx.api_key,
    )

    quant_picks = quant_output.get("ranked_picks", [])
    # Validate picks have required 'ticker' key — LLM output parsing can drop it
    valid_picks = [p for p in quant_picks if isinstance(p, dict) and "ticker" in p]
    if len(valid_picks) < len(quant_picks):
        log.warning(
            "[team:%s] quant produced %d picks but %d lack 'ticker' key — dropped",
            team_id, len(quant_picks), len(quant_picks) - len(valid_picks),
        )
    if not valid_picks:
        log.warning("[team:%s] quant produced no valid picks", team_id)
        return _empty_result(team_id, quant_output=quant_output)

    # ── Step 3: Qual analyst reviews top 5 ────────────────────────────────────
    top5 = valid_picks[:5]

    qual_output = run_qual_analyst(
        team_id=team_id,
        quant_top5=top5,
        prior_theses=ctx.prior_theses,
        market_regime=ctx.market_regime,
        run_date=ctx.run_date,
        api_key=ctx.api_key,
        price_data=ctx.price_data,
        episodic_memories=ctx.episodic_memories,
        semantic_memories=ctx.semantic_memories,
    )

    # ── Step 4: Peer review → final 2-3 ──────────────────────────────────────
    peer_output = run_peer_review(
        team_id=team_id,
        quant_picks=top5,
        qual_assessments=qual_output.get("assessments", []),
        additional_candidate=qual_output.get("additional_candidate"),
        technical_scores=ctx.technical_scores,
        market_regime=ctx.market_regime,
        api_key=ctx.api_key,
    )

    # ── Step 5: Thesis maintenance for held stocks ────────────────────────────
    team_held = [t for t in ctx.held_tickers if ctx.sector_map.get(t, "") in
                 {s for s, tid in _sector_team_inverse().items() if tid == team_id}]

    # Check for sector regime change
    sector_regime_changed = _check_regime_change(
        team_id, ctx.prior_sector_ratings, ctx.current_sector_ratings
    )

    thesis_updates = {}
    for ticker in team_held:
        triggers = check_material_triggers(
            ticker=ticker,
            news_data=ctx.news_data_by_ticker.get(ticker),
            price_data=ctx.price_data.get(ticker),
            analyst_data=ctx.analyst_data_by_ticker.get(ticker),
            insider_data=ctx.insider_data_by_ticker.get(ticker),
            prior_thesis=ctx.prior_theses.get(ticker),
            sector_regime_changed=sector_regime_changed,
            run_date=ctx.run_date,
        )

        if triggers:
            # Material event — update thesis via Haiku
            updated = _update_thesis_for_held_stock(
                ticker, triggers, ctx.prior_theses.get(ticker),
                ctx.news_data_by_ticker.get(ticker),
                ctx.analyst_data_by_ticker.get(ticker),
                ctx.run_date, team_id, ctx.api_key,
            )
            thesis_updates[ticker] = updated
        else:
            # No material event — preserve prior thesis
            if ctx.prior_theses.get(ticker):
                thesis_updates[ticker] = {
                    **ctx.prior_theses[ticker],
                    "stale_days": ctx.prior_theses[ticker].get("stale_days", 0) + 1,
                    "triggers": [],
                }

    # ── Combine tool call logs ────────────────────────────────────────────────
    all_tool_calls = (
        quant_output.get("tool_calls", []) +
        qual_output.get("tool_calls", []) +
        [{"phase": "peer_review", "rationale": peer_output.get("peer_review_rationale", "")}]
    )

    log.info("[team:%s] done — %d recommendations, %d thesis updates",
             team_id, len(peer_output.get("recommendations", [])), len(thesis_updates))

    # Propagate the first analyst error (if any) to the team level so the
    # score_aggregator can hard-fail loudly. Quant errors take precedence —
    # a broken quant stage guarantees a broken qual stage.
    team_error = quant_output.get("error") or qual_output.get("error")
    if team_error:
        team_error = f"[team:{team_id}] {team_error}"

    return {
        "team_id": team_id,
        "recommendations": peer_output.get("recommendations", []),
        "thesis_updates": thesis_updates,
        "quant_output": quant_output,
        "qual_output": qual_output,
        "peer_review_output": peer_output,
        "tool_calls": all_tool_calls,
        "error": team_error,
    }


def _empty_result(team_id: str, quant_output: dict | None = None,
                  error: str | None = None) -> dict:
    # If quant produced an error and no explicit error was passed, surface
    # it so the aggregator sees the failure rather than an empty team that
    # looks identical to "no sector tickers in universe".
    if error is None and quant_output is not None:
        error = quant_output.get("error")
    return {
        "team_id": team_id,
        "recommendations": [],
        "thesis_updates": {},
        "quant_output": quant_output or {},
        "qual_output": {},
        "peer_review_output": {},
        "tool_calls": [],
        "error": error,
    }


def _sector_team_inverse() -> dict[str, str]:
    """Return {gics_sector: team_id} mapping."""
    from agents.sector_teams.team_config import SECTOR_TEAM_MAP
    return SECTOR_TEAM_MAP


def _check_regime_change(
    team_id: str,
    prior_ratings: dict,
    current_ratings: dict,
) -> bool:
    """Check if any of this team's sectors changed regime rating."""
    team_sectors = TEAM_SECTORS.get(team_id, [])
    for sector in team_sectors:
        prior = prior_ratings.get(sector, {}).get("rating", "market_weight")
        current = current_ratings.get(sector, {}).get("rating", "market_weight")
        if prior != current:
            return True
    return False


def _update_thesis_for_held_stock(
    ticker: str,
    triggers: list[str],
    prior_thesis: dict | None,
    news_data: dict | None,
    analyst_data: dict | None,
    run_date: str,
    team_id: str,
    api_key: str | None = None,
) -> dict:
    """Update thesis for a held stock with material triggers (single Haiku call)."""
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=500,
    )

    prior_text = ""
    if prior_thesis:
        prior_text = format_structured_thesis_for_prompt(prior_thesis)

    news_summary = ""
    if news_data:
        articles = news_data.get("articles", [])
        if articles:
            news_summary = "\n".join(
                f"- {a.get('headline', '')}" for a in articles[:5]
            )

    analyst_summary = ""
    if analyst_data:
        analyst_summary = (
            f"Consensus: {analyst_data.get('consensus_rating', 'N/A')}, "
            f"Target: ${analyst_data.get('mean_target', 'N/A')}, "
            f"Upside: {analyst_data.get('upside_pct', 'N/A')}%"
        )

    prompt = load_prompt("sector_team_thesis_update").format(
        team_title=team_id.title(),
        ticker=ticker,
        triggers_csv=", ".join(triggers),
        prior_text=prior_text or "No prior thesis available.",
        news_summary=news_summary or "No significant news.",
        analyst_summary=analyst_summary or "No analyst updates.",
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        import re
        text = response.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            llm_update = json.loads(match.group())
            # Held-stock updates only revise narrative fields. Scoring fields
            # (final_score, quant_score, qual_score, rating, sector, team_id)
            # come from the prior thesis — the LLM is not authoritative on
            # scores and does not re-run the quant tools for held stocks.
            #
            # Strip None values from llm_update BEFORE merging so that an
            # LLM that emits `"final_score": null` (seen on LNTH/LLY/PFE/
            # VRTX/CME/JHG/COKE/HSY/KR in the 2026-04-11 run) can't
            # overwrite valid prior scores with nulls. The merge order
            # `{**prior_thesis, **llm_update}` still lets the LLM override
            # narrative fields with real values, but nulls are dropped.
            #
            # Prior mitigation was a downstream downgrade in
            # research_graph._build_signals_payload that catches broken
            # theses at emit time — this is the upstream fix for the same
            # 2026-04-04 / 2026-04-11 root cause.
            llm_update_clean = {k: v for k, v in llm_update.items() if v is not None}
            if prior_thesis:
                result = {**prior_thesis, **llm_update_clean}
            else:
                result = llm_update_clean
                result["score_failed"] = True
            result["last_updated"] = run_date
            result["triggers"] = triggers
            result["stale_days"] = 0
            return result
        else:
            log.warning(
                "[thesis_update:%s] LLM returned no JSON block — using fallback",
                ticker,
            )
    except Exception as e:
        log.warning("[thesis_update:%s] failed: %s", ticker, e)

    # Fallback: preserve prior thesis
    if prior_thesis:
        return {**prior_thesis, "triggers": triggers, "stale_days": 0}
    return {"triggers": triggers, "stale_days": 0, "score_failed": True}
