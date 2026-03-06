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

_PROMPT_TEMPLATE = """\
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
[{{"rank": 1, "ticker": "XXX", "path": "momentum", "rationale": "..."}}, ...]
"""


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


def _extract_ranking(text: str) -> list[dict]:
    """Extract JSON array of objects from agent response."""
    # Find the first '[' that opens a JSON array of objects (greedy match)
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data[:10]
        except json.JSONDecodeError as e:
            print(f"[SCANNER] JSON decode error on regex match: {e}")

    # Fallback: strip markdown code fences and try parsing the whole text
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return data[:10]
    except json.JSONDecodeError:
        pass

    print(f"[SCANNER] _extract_ranking failed. Raw response (first 500 chars):\n{text[:500]}")
    return []


def run_scanner_ranking_agent(
    candidates: list[dict],
    market_regime: str = "neutral",
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Run the Scanner Ranking Agent on ~50 quant-filtered candidates.

    Args:
        candidates: list of candidate dicts (from Stage 1+2)
        market_regime: from Macro Agent output

    Returns:
        list of top-10 ranked dicts: {rank, ticker, path, rationale}
    """
    if not candidates:
        return []

    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    table = _build_candidates_table(candidates)
    prompt = _PROMPT_TEMPLATE.format(
        candidates_table=table,
        market_regime=market_regime,
        n=len(candidates),
    )

    response = client.messages.create(
        model=STRATEGIC_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        messages=[{"role": "user", "content": prompt}],
    )

    return _extract_ranking(response.content[0].text)
