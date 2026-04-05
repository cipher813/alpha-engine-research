"""
CIO Agent — evaluates all sector team recommendations in a single batch Sonnet call.

The CIO sees all candidates simultaneously and selects the top N for open slots.
Evaluates on 4 dimensions: team conviction, macro alignment, portfolio fit, catalyst specificity.
Writes entry theses for advanced stocks. All decisions (advance, reject, deadlock) saved.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY
from agents.prompt_loader import load_prompt

log = logging.getLogger(__name__)


def run_cio(
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    current_population: list[dict],
    open_slots: int,
    exits: list[dict],
    run_date: str,
    api_key: Optional[str] = None,
    prior_decisions: list[dict] | None = None,
) -> dict:
    """
    Run the CIO evaluation in a single batch Sonnet call.

    Args:
        candidates: All team recommendations. Each has:
            ticker, team_id, quant_score, qual_score, bull_case, bear_case,
            catalysts, conviction, quant_rationale.
        macro_context: {market_regime, macro_report_summary, ...}
        sector_ratings: {sector: {rating, modifier, rationale}}
        current_population: Current held stocks (for portfolio fit analysis).
        open_slots: Number of population slots available.
        exits: Stocks being removed this week (for context).
        run_date: YYYY-MM-DD.

    Returns:
        {
            "decisions": list[dict],  # one per candidate with decision + rationale
            "advanced_tickers": list[str],
            "entry_theses": dict[str, dict],  # CIO-authored theses for advanced stocks
        }
    """
    if not candidates:
        log.info("[cio] no candidates to evaluate")
        return {"decisions": [], "advanced_tickers": [], "entry_theses": {}}

    if open_slots <= 0:
        log.info("[cio] no open slots — rejecting all %d candidates", len(candidates))
        return {
            "decisions": [
                _reject_decision(c, "no open slots") for c in candidates
            ],
            "advanced_tickers": [],
            "entry_theses": {},
        }

    llm = ChatAnthropic(
        model=STRATEGIC_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
    )

    prompt = _build_cio_prompt(
        candidates, macro_context, sector_ratings,
        current_population, open_slots, exits, run_date,
        prior_decisions=prior_decisions,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        return _parse_cio_response(text, candidates, open_slots)
    except Exception as e:
        log.error("[cio] evaluation failed: %s", e)
        # Fallback: rank by combined score, take top N
        return _fallback_selection(candidates, open_slots)


def _format_prior_decisions(prior_decisions: list[dict] | None) -> str:
    """Format prior IC decisions for prompt injection. Returns empty string if none."""
    if not prior_decisions:
        return ""
    lines = ["PRIOR WEEK IC DECISIONS (for portfolio continuity):"]
    for d in prior_decisions[:10]:
        ticker = d.get("ticker", "?")
        action = d.get("thesis_type", "?")
        rationale = (d.get("rationale", "") or "")[:120]
        lines.append(f"  - {ticker}: {action} — {rationale}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_cio_prompt(
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    population: list[dict],
    open_slots: int,
    exits: list[dict],
    run_date: str,
    prior_decisions: list[dict] | None = None,
) -> str:
    """Build the single batch CIO prompt."""

    # Format candidates
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        team = c.get("team_id", "unknown")
        qs = c.get("quant_score", "?")
        qls = c.get("qual_score", "?")
        conv = c.get("conviction", "?")
        bull = (c.get("bull_case", "") or "")[:150]
        bear = (c.get("bear_case", "") or "")[:150]
        cats = ", ".join(c.get("catalysts", [])[:3]) if c.get("catalysts") else "none specified"

        cand_lines.append(
            f"  {i}. {c['ticker']} [{team}] — Quant: {qs}, Qual: {qls}, Conviction: {conv}\n"
            f"     Bull: {bull}\n"
            f"     Bear: {bear}\n"
            f"     Catalysts: {cats}"
        )
    candidates_text = "\n".join(cand_lines)

    # Format current population by sector
    pop_by_sector = {}
    for p in population:
        sector = p.get("sector", "Unknown")
        pop_by_sector.setdefault(sector, []).append(p.get("ticker", ""))
    pop_text = "\n".join(f"  {s}: {', '.join(ts)}" for s, ts in sorted(pop_by_sector.items()))

    # Format exits
    exit_text = "\n".join(
        f"  - {e.get('ticker_out', e.get('ticker', '?'))}: {e.get('reason', 'unknown')}"
        for e in exits[:10]
    ) if exits else "  None"

    # Format sector ratings
    ratings_text = "\n".join(
        f"  {s}: {r.get('rating', 'market_weight')} (modifier: {r.get('modifier', 1.0):.2f})"
        for s, r in sorted(sector_ratings.items())
    )

    regime = macro_context.get("market_regime", "neutral")

    return load_prompt("ic_cio_evaluation").format(
        run_date=run_date,
        regime=regime,
        open_slots=open_slots,
        ratings_text=ratings_text,
        pop_text=pop_text,
        exit_text=exit_text,
        prior_decisions_block=_format_prior_decisions(prior_decisions),
        candidates_text=candidates_text,
    )


def _parse_cio_response(
    text: str,
    candidates: list[dict],
    open_slots: int,
) -> dict:
    """Parse the CIO's JSON response."""
    # Find JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.warning("[cio] could not parse JSON from response")
        return _fallback_selection(candidates, open_slots)

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("[cio] invalid JSON in response")
        return _fallback_selection(candidates, open_slots)

    decisions = result.get("decisions", [])
    if not decisions:
        log.warning("[cio] no decisions in response")
        return _fallback_selection(candidates, open_slots)

    # Extract advanced tickers and theses
    advanced = []
    entry_theses = {}
    for d in decisions:
        if d.get("decision") == "ADVANCE":
            ticker = d.get("ticker", "")
            advanced.append(ticker)
            if d.get("entry_thesis"):
                entry_theses[ticker] = d["entry_thesis"]

    # Cap at open_slots
    advanced = advanced[:open_slots]

    log.info("[cio] %d advanced, %d rejected, %d deadlocked out of %d candidates",
             len([d for d in decisions if d.get("decision") == "ADVANCE"]),
             len([d for d in decisions if d.get("decision") == "REJECT"]),
             len([d for d in decisions if d.get("decision") == "NO_ADVANCE_DEADLOCK"]),
             len(decisions))

    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
    }


def _fallback_selection(candidates: list[dict], open_slots: int) -> dict:
    """Fallback: rank by combined quant+qual score."""
    scored = []
    for c in candidates:
        qs = c.get("quant_score") or 0
        qls = c.get("qual_score") or 0
        combined = (qs + qls) / 2 if qls else qs
        scored.append((combined, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    decisions = []
    advanced = []
    entry_theses = {}
    for i, (score, c) in enumerate(scored):
        if i < open_slots:
            decisions.append({
                "ticker": c["ticker"],
                "decision": "ADVANCE",
                "rank": i + 1,
                "conviction": int(score),
                "rationale": "Fallback: selected by combined score",
                "entry_thesis": None,
            })
            advanced.append(c["ticker"])
        else:
            decisions.append({
                "ticker": c["ticker"],
                "decision": "REJECT",
                "rank": None,
                "conviction": int(score),
                "rationale": "Fallback: below cutoff",
                "entry_thesis": None,
            })

    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
    }


def _reject_decision(candidate: dict, reason: str) -> dict:
    return {
        "ticker": candidate.get("ticker", ""),
        "decision": "REJECT",
        "rank": None,
        "conviction": 0,
        "rationale": reason,
        "entry_thesis": None,
    }
