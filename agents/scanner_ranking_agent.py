"""
Scanner Ranking Agent — Stage 3 of the market scanner (§6.3).

Single LLM call (claude-sonnet-4-6) over all ~50 quant-filtered candidates.
Outputs ranked top-10 with 1-sentence rationale per stock.

Using one call over 50 candidates enables cross-stock comparative judgment —
running 50 independent calls would produce incomparable isolated scores (§18.5).
"""

from __future__ import annotations

import json
import re
from typing import Optional

import anthropic

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY
from agents.token_guard import check_prompt_size

from agents.prompt_loader import load_prompt

_PROMPT_TEMPLATE = load_prompt("scanner_ranking_agent")


def _build_candidates_table(candidates: list[dict]) -> str:
    """Format candidates into a pipe-delimited table for the prompt."""
    lines = []
    # Sort: momentum first by tech_score, then deep_value
    momentum = sorted(
        [c for c in candidates if c.get("path") == "momentum"],
        key=lambda x: x.get("tech_score", 0), reverse=True,
    )
    deep_val = [c for c in candidates if c.get("path") == "deep_value"]

    for c in momentum + deep_val:
        ticker = c.get("ticker", "")
        sector = c.get("sector", "")
        path = c.get("path", "")
        tech_score = f"{c.get('tech_score', 0):.0f}"
        analyst_rating = c.get("analyst_rating", "N/A")
        upside = f"{c.get('upside_pct', 'N/A')}%"
        headlines = c.get("headlines", [])
        h1 = headlines[0] if len(headlines) > 0 else "N/A"
        h2 = headlines[1] if len(headlines) > 1 else ""
        lines.append(f"{ticker} | {sector} | {path} | {tech_score} | {analyst_rating} | {upside} | {h1} | {h2}")

    return "\n".join(lines)


def _extract_ranking(text: str, top_n: int = 10) -> list[dict]:
    """Extract JSON array of objects from agent response."""
    # Find the first '[' that opens a JSON array of objects (greedy match)
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data[:top_n]
        except json.JSONDecodeError as e:
            print(f"[SCANNER] JSON decode error on regex match: {e}")

    # Fallback: strip markdown code fences and try parsing the whole text
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return data[:top_n]
    except json.JSONDecodeError:
        pass

    print(f"[SCANNER] _extract_ranking failed. Raw response (first 500 chars):\n{text[:500]}")
    return []


def run_scanner_ranking_agent(
    candidates: list[dict],
    market_regime: str = "neutral",
    api_key: Optional[str] = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Run the Scanner Ranking Agent on ~50 quant-filtered candidates.

    Args:
        candidates: list of candidate dicts (from Stage 1+2)
        market_regime: from Macro Agent output
        top_n: number of top candidates to rank (default 10, expanded to 35 for population)

    Returns:
        list of top-N ranked dicts: {rank, ticker, path, rationale}
    """
    if not candidates:
        return []

    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    table = _build_candidates_table(candidates)
    prompt = _PROMPT_TEMPLATE.format(
        candidates_table=table,
        market_regime=market_regime,
        n=len(candidates),
        top_n=top_n,
    )

    prompt = check_prompt_size(prompt, MAX_TOKENS_STRATEGIC, caller="scanner_ranking")

    response = client.messages.create(
        model=STRATEGIC_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        messages=[{"role": "user", "content": prompt}],
    )

    return _extract_ranking(response.content[0].text, top_n=top_n)
