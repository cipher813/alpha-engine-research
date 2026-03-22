"""
Structured thesis — fixed-schema format that replaces free-text truncation.

Instead of passing a 2000-char truncated markdown report to agents (which loses
oldest findings), we maintain a compact structured thesis with fixed field sizes.
Fields are updated in-place each run — the thesis never grows beyond ~1200 chars.

Full markdown reports are still archived to S3 for human consumption and the
consolidator. The structured thesis is what agents receive as prior context.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Field size limits (total ~1200 chars)
_MAX_BULL_CASE = 200
_MAX_BEAR_CASE = 200
_MAX_CATALYST = 50
_MAX_RISK = 50
_MAX_CONVICTION_RATIONALE = 100
_MAX_CATALYSTS = 5
_MAX_RISKS = 5


def _truncate(text: str | None, max_chars: int) -> str:
    """Truncate a string to max_chars, preserving word boundaries."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Cut at last space before limit
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..." if cut else text[:max_chars]


def build_structured_thesis(
    news_json: dict,
    research_json: dict,
    aggregated: dict,
    prior_structured: dict | None = None,
) -> dict:
    """
    Build or update a structured thesis from agent outputs.

    If prior_structured exists, preserves fields that haven't changed
    (e.g., bull_case stays if research_json has no new key_upside).

    Args:
        news_json: structured output from news agent
        research_json: structured output from research agent
        aggregated: output from score_aggregator
        prior_structured: prior run's structured thesis (or None on first run)

    Returns:
        dict with fixed fields, total ~1200 chars
    """
    prior = prior_structured or {}

    # Bull case: prefer research upside, fall back to news catalyst
    bull_case = (
        research_json.get("key_upside")
        or news_json.get("key_catalyst")
        or prior.get("bull_case", "")
    )

    # Bear case: prefer research risk, fall back to prior
    bear_case = (
        research_json.get("key_risk")
        or prior.get("bear_case", "")
    )

    # Catalysts: merge new catalyst with prior list, deduplicate, cap at 5
    new_catalyst = news_json.get("key_catalyst") or ""
    prior_catalysts = prior.get("catalysts", [])
    catalysts = []
    seen = set()
    # New catalyst first
    if new_catalyst and new_catalyst.lower() not in seen:
        catalysts.append(_truncate(new_catalyst, _MAX_CATALYST))
        seen.add(new_catalyst.lower())
    # Then prior catalysts
    for c in prior_catalysts:
        if c.lower() not in seen and len(catalysts) < _MAX_CATALYSTS:
            catalysts.append(c)
            seen.add(c.lower())

    # Risks: merge new risk with prior list
    new_risk = research_json.get("key_risk") or ""
    prior_risks = prior.get("risks", [])
    risks = []
    seen_risks = set()
    if new_risk and new_risk.lower() not in seen_risks:
        risks.append(_truncate(new_risk, _MAX_RISK))
        seen_risks.add(new_risk.lower())
    for r in prior_risks:
        if r.lower() not in seen_risks and len(risks) < _MAX_RISKS:
            risks.append(r)
            seen_risks.add(r.lower())

    # Conviction rationale
    conviction = aggregated.get("conviction", "stable")
    signal = aggregated.get("signal", "HOLD")
    score = aggregated.get("final_score", 50)
    conviction_rationale = f"{signal} at {score:.0f} ({conviction} conviction)"

    return {
        "bull_case": _truncate(bull_case, _MAX_BULL_CASE),
        "bear_case": _truncate(bear_case, _MAX_BEAR_CASE),
        "catalysts": catalysts,
        "risks": risks,
        "conviction_rationale": _truncate(conviction_rationale, _MAX_CONVICTION_RATIONALE),
        "last_updated": aggregated.get("date", ""),
    }


def format_structured_thesis_for_prompt(structured: dict) -> str:
    """
    Format a structured thesis as concise text for agent prompts.

    Returns a ~1200-char block that agents can parse quickly.
    """
    if not structured:
        return "NONE — initial analysis"

    lines = []
    if structured.get("bull_case"):
        lines.append(f"BULL CASE: {structured['bull_case']}")
    if structured.get("bear_case"):
        lines.append(f"BEAR CASE: {structured['bear_case']}")

    catalysts = structured.get("catalysts", [])
    if catalysts:
        lines.append("CATALYSTS: " + " | ".join(catalysts))

    risks = structured.get("risks", [])
    if risks:
        lines.append("RISKS: " + " | ".join(risks))

    if structured.get("conviction_rationale"):
        lines.append(f"CONVICTION: {structured['conviction_rationale']}")

    if structured.get("last_updated"):
        lines.append(f"LAST UPDATED: {structured['last_updated']}")

    return "\n".join(lines)
