"""
Analyst Research Agent (§4.2).

One instance per tracked stock (up to 23 parallel invocations).
Uses claude-haiku-4-5 (per-stock model).

Implements the three-step thesis drafting protocol (§7.3).
"""

from __future__ import annotations

import json
import re
from typing import Optional

import anthropic

from config import PER_STOCK_MODEL, MAX_TOKENS_PER_STOCK, ANTHROPIC_API_KEY, PRIOR_REPORT_MAX_CHARS
from agents.token_guard import check_prompt_size

from agents.prompt_loader import load_prompt

_PROMPT_TEMPLATE = load_prompt("research_agent")


def _truncate_prior(text: str, max_chars: int | None = None) -> str:
    """Truncate prior report to last max_chars characters to manage prompt size."""
    if max_chars is None:
        max_chars = PRIOR_REPORT_MAX_CHARS
    if not text or len(text) <= max_chars:
        return text
    return "[...truncated...]\n" + text[-max_chars:]


def _format_rating_changes(changes: list[dict]) -> str:
    if not changes:
        return "None in last 30 days."
    parts = []
    for c in changes[:5]:
        parts.append(
            f"{c.get('date', '')} | {c.get('firm', '')} | "
            f"{c.get('action', '')} | {c.get('from_grade', '')} → {c.get('to_grade', '')}"
        )
    return "\n".join(parts)


def _format_earnings(surprises: list[dict]) -> str:
    if not surprises:
        return "No recent earnings data."
    parts = []
    for e in surprises[:2]:
        surprise_pct = e.get("surprise_pct")
        direction = "beat" if surprise_pct and surprise_pct > 0 else "miss"
        parts.append(
            f"{e.get('date', '')}: {direction} by {abs(surprise_pct or 0):.1f}% "
            f"(actual: {e.get('actual', 'N/A')}, estimated: {e.get('estimated', 'N/A')})"
        )
    return " | ".join(parts)


def _extract_json_from_response(text: str) -> dict:
    match = re.search(r"\{[^{}]*\"research_score_short\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"research_score_short": None, "research_score_long": None, "consensus_direction": "neutral", "material_changes": False, "score_failed": True}


def run_research_agent(
    ticker: str,
    company_name: str,
    prior_report: Optional[str],
    prior_date: str,
    analyst_data: dict,
    api_key: Optional[str] = None,
    insider_summary: str = "",
) -> dict:
    """
    Run the Analyst Research Agent for a single ticker.

    Returns dict with:
      report_md: str — refreshed ~300-word markdown report
      research_json: dict — structured JSON output from agent
      ticker: str
    """
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    prior_text = _truncate_prior(prior_report) if prior_report else "NONE — initial report"

    # Build the base prompt
    prompt_text = _PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        prior_date=prior_date,
        prior_report=prior_text,
        consensus_rating=analyst_data.get("consensus_rating", "N/A"),
        num_analysts=analyst_data.get("num_analysts", "N/A"),
        mean_target=f"{analyst_data.get('mean_target', 'N/A')}",
        current_price=f"{analyst_data.get('current_price', 'N/A')}",
        upside_pct=f"{analyst_data.get('upside_pct', 'N/A')}",
        rating_changes=_format_rating_changes(analyst_data.get("rating_changes", [])),
        earnings_surprise=_format_earnings(analyst_data.get("earnings_surprises", [])),
    )

    # O13: Append insider activity summary if available
    if insider_summary:
        prompt_text += f"\n\n{insider_summary}\n\nIncorporate insider trading activity into your fundamental analysis."

    prompt = prompt_text

    prompt = check_prompt_size(prompt, MAX_TOKENS_PER_STOCK, caller=f"research_agent:{ticker}")

    response = client.messages.create(
        model=PER_STOCK_MODEL,
        max_tokens=MAX_TOKENS_PER_STOCK,
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = response.content[0].text
    research_json = _extract_json_from_response(full_text)
    report_md = re.sub(r"\{[^{}]*\"research_score\"[^{}]*\}", "", full_text, flags=re.DOTALL).strip()

    return {
        "ticker": ticker,
        "report_md": report_md,
        "research_json": research_json,
        "research_score": float(research_json.get("research_score_short", 50)),
        "research_score_lt": float(research_json.get("research_score_long", 50)),
        "material_changes": bool(research_json.get("material_changes", False)),
    }
