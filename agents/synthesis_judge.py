"""
Synthesis Judge — resolves significant disagreement between news and research agents.

When news and research sub-scores diverge significantly for a ticker, the deterministic
weighted average can't reason about *why* they disagree. The synthesis judge receives
both full reports and produces a contextually-reasoned adjusted score.

Runs on the top 5 most divergent tickers per cycle. All divergence scores are tracked
for future threshold analysis.

Model: Sonnet (cross-analysis synthesis requires comparative reasoning).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY
from agents.token_guard import check_prompt_size

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a senior investment analyst acting as a synthesis judge.

Two independent analysts have scored {ticker} ({company_name}) and their assessments
diverge significantly:

NEWS ANALYST SCORE: {news_score}/100
RESEARCH ANALYST SCORE: {research_score}/100
DIVERGENCE: {divergence:.0f} points

--- NEWS ANALYST REPORT ---
{news_report}

--- RESEARCH ANALYST REPORT ---
{research_report}

--- ADDITIONAL CONTEXT ---
Analyst consensus: {consensus_rating} | Target upside: {upside_pct}
Market regime: {market_regime} | Sector rating: {sector_rating}

YOUR TASK:
1. Identify WHY the two analysts disagree — is it stale data, different time horizons,
   or genuinely conflicting signals?
2. Determine which perspective is more relevant for a 5-day to 6-month investment horizon.
3. Produce an adjusted attractiveness score (0-100) that reflects your reasoned synthesis.

Output a JSON block at the end of your response:
{{"adjusted_score": <float 0-100>, "dominant_perspective": "news"|"research"|"balanced", "judge_rationale": "<1-2 sentence explanation>"}}
"""


def _extract_judge_json(text: str) -> dict:
    """Extract JSON from judge response."""
    match = re.search(r"\{[^{}]*\"adjusted_score\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            # Clamp score to valid range
            parsed["adjusted_score"] = max(0.0, min(100.0, float(parsed.get("adjusted_score", 50))))
            return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    logger.warning("[synthesis_judge] JSON extraction failed — returning None")
    return None


def run_synthesis_judge(
    ticker: str,
    company_name: str,
    news_report: str,
    research_report: str,
    news_score: float,
    research_score: float,
    analyst_data: dict,
    macro_context: dict,
    api_key: Optional[str] = None,
) -> dict | None:
    """
    Run the synthesis judge for a single ticker with divergent news/research scores.

    Returns dict with:
      adjusted_score: float (0-100)
      dominant_perspective: "news" | "research" | "balanced"
      judge_rationale: str
    Or None if the judge call fails.
    """
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    divergence = abs(news_score - research_score)
    consensus_rating = analyst_data.get("consensus_rating", "N/A")
    upside_pct = analyst_data.get("upside_pct")
    upside_str = f"{upside_pct:+.0f}%" if upside_pct is not None else "N/A"

    prompt = _PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        news_score=f"{news_score:.0f}",
        research_score=f"{research_score:.0f}",
        divergence=divergence,
        news_report=news_report[:2000] if news_report else "No news report available.",
        research_report=research_report[:2000] if research_report else "No research report available.",
        consensus_rating=consensus_rating,
        upside_pct=upside_str,
        market_regime=macro_context.get("regime", "neutral"),
        sector_rating=macro_context.get("sector_rating", {}).get("rating", "market_weight"),
    )

    prompt = check_prompt_size(prompt, MAX_TOKENS_STRATEGIC, caller=f"synthesis_judge:{ticker}")

    try:
        response = client.messages.create(
            model=STRATEGIC_MODEL,
            max_tokens=MAX_TOKENS_STRATEGIC,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_judge_json(response.content[0].text)
        if result:
            logger.info(
                "[synthesis_judge] %s: news=%s research=%s → adjusted=%s (%s) — %s",
                ticker, news_score, research_score,
                result["adjusted_score"], result.get("dominant_perspective"),
                result.get("judge_rationale", "")[:80],
            )
        return result
    except Exception as e:
        logger.error("[synthesis_judge] %s: LLM call failed: %s", ticker, e)
        return None
