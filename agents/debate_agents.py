"""
Bull/Bear Debate Agents — adversarial analysis for new population candidates.

For stocks being considered for population entry, runs a structured debate:
  1. Bull agent makes the case FOR the stock (Haiku)
  2. Bear agent makes the case AGAINST, seeing bull's arguments (Haiku)
  3. Judge agent synthesizes both and produces a conviction-weighted verdict (Sonnet)

Only runs for NEW candidates (not already in population). Bear sees bull's
arguments to construct specific counter-arguments rather than independent analysis.

Model routing: Haiku for bull/bear (structured extraction), Sonnet for judge
(cross-analysis synthesis requiring comparative reasoning).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config import (
    PER_STOCK_MODEL,
    STRATEGIC_MODEL,
    MAX_TOKENS_PER_STOCK,
    MAX_TOKENS_STRATEGIC,
    ANTHROPIC_API_KEY,
)
from agents.token_guard import check_prompt_size

logger = logging.getLogger(__name__)


_BULL_PROMPT = """You are a BULLISH equity analyst making the strongest possible case
for investing in {ticker}.

AVAILABLE DATA:
News report: {news_report}
Research report: {research_report}
Analyst consensus: {consensus_rating} | Target upside: {upside_pct}
Current tech score: {tech_score}/100

Make the strongest BULL case. Focus on:
- Catalysts that could drive outperformance in the next 1-6 months
- Why analyst consensus supports the thesis
- What the market is underappreciating

Be specific and evidence-based. Do not hedge.

Output JSON:
{{"bull_case": "<2-3 sentence bull thesis>", "key_arguments": ["<argument 1>", "<argument 2>", "<argument 3>"], "conviction": <0-100>}}
"""

_BEAR_PROMPT = """You are a BEARISH equity analyst making the strongest possible case
AGAINST investing in {ticker}.

AVAILABLE DATA:
News report: {news_report}
Research report: {research_report}
Analyst consensus: {consensus_rating} | Target upside: {upside_pct}
Current tech score: {tech_score}/100

THE BULL CASE (from a colleague — you must address these arguments directly):
{bull_case}
Bull arguments: {bull_arguments}

Make the strongest BEAR case. Focus on:
- Specific risks that could cause underperformance
- Why the bull's catalysts may not materialize or are already priced in
- What the market is overlooking on the downside

Be specific and evidence-based. Address each bull argument directly.

Output JSON:
{{"bear_case": "<2-3 sentence bear thesis>", "key_arguments": ["<counter to bull arg 1>", "<counter to bull arg 2>", "<additional risk>"], "conviction": <0-100>}}
"""

_JUDGE_PROMPT = """You are a senior portfolio manager evaluating whether to add {ticker}
to the portfolio. Two analysts have presented opposing views.

BULL ANALYST (conviction: {bull_conviction}/100):
{bull_case}
Arguments: {bull_arguments}

BEAR ANALYST (conviction: {bear_conviction}/100):
{bear_case}
Arguments: {bear_arguments}

MARKET CONTEXT:
Market regime: {market_regime} | Sector rating: {sector_rating}

YOUR TASK:
1. Weigh both cases. Which analyst has the stronger evidence?
2. Consider the market context — does the regime favor or hinder this stock?
3. Produce a verdict and conviction score.

Output JSON:
{{"verdict": "<strong_buy|buy|neutral|avoid>", "conviction_score": <0-100>, "rationale": "<2-3 sentence synthesis>", "key_condition": "<what would change your mind>"}}
"""


def _extract_json(text: str, key: str) -> dict | None:
    """Extract JSON block containing the specified key."""
    pattern = r"\{[^{}]*\"" + re.escape(key) + r"\"[^{}]*\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_bull_agent(
    ticker: str,
    news_report: str,
    research_report: str,
    analyst_data: dict,
    tech_score: float,
    api_key: Optional[str] = None,
) -> dict | None:
    """Run the bull agent for a single ticker. Returns structured bull case."""
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    prompt = _BULL_PROMPT.format(
        ticker=ticker,
        news_report=news_report[:1500] if news_report else "No news report.",
        research_report=research_report[:1500] if research_report else "No research report.",
        consensus_rating=analyst_data.get("consensus_rating", "N/A"),
        upside_pct=f"{analyst_data.get('upside_pct', 0):+.0f}%" if analyst_data.get("upside_pct") else "N/A",
        tech_score=f"{tech_score:.0f}" if tech_score else "N/A",
    )
    prompt = check_prompt_size(prompt, MAX_TOKENS_PER_STOCK, caller=f"bull_agent:{ticker}")

    try:
        response = client.messages.create(
            model=PER_STOCK_MODEL,
            max_tokens=MAX_TOKENS_PER_STOCK,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text, "bull_case")
        if result:
            result["conviction"] = max(0, min(100, int(result.get("conviction", 50))))
        return result
    except Exception as e:
        logger.error("[bull_agent] %s: failed: %s", ticker, e)
        return None


def run_bear_agent(
    ticker: str,
    news_report: str,
    research_report: str,
    analyst_data: dict,
    tech_score: float,
    bull_output: dict,
    api_key: Optional[str] = None,
) -> dict | None:
    """Run the bear agent for a single ticker. Sees bull's arguments."""
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    bull_args = bull_output.get("key_arguments", [])

    prompt = _BEAR_PROMPT.format(
        ticker=ticker,
        news_report=news_report[:1500] if news_report else "No news report.",
        research_report=research_report[:1500] if research_report else "No research report.",
        consensus_rating=analyst_data.get("consensus_rating", "N/A"),
        upside_pct=f"{analyst_data.get('upside_pct', 0):+.0f}%" if analyst_data.get("upside_pct") else "N/A",
        tech_score=f"{tech_score:.0f}" if tech_score else "N/A",
        bull_case=bull_output.get("bull_case", "No bull case provided."),
        bull_arguments="\n".join(f"- {a}" for a in bull_args) if bull_args else "None provided.",
    )
    prompt = check_prompt_size(prompt, MAX_TOKENS_PER_STOCK, caller=f"bear_agent:{ticker}")

    try:
        response = client.messages.create(
            model=PER_STOCK_MODEL,
            max_tokens=MAX_TOKENS_PER_STOCK,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text, "bear_case")
        if result:
            result["conviction"] = max(0, min(100, int(result.get("conviction", 50))))
        return result
    except Exception as e:
        logger.error("[bear_agent] %s: failed: %s", ticker, e)
        return None


def run_judge_agent(
    ticker: str,
    bull_output: dict,
    bear_output: dict,
    macro_context: dict,
    api_key: Optional[str] = None,
) -> dict | None:
    """Run the judge agent to synthesize bull/bear debate. Returns verdict."""
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    prompt = _JUDGE_PROMPT.format(
        ticker=ticker,
        bull_conviction=bull_output.get("conviction", 50),
        bull_case=bull_output.get("bull_case", "No bull case."),
        bull_arguments="\n".join(f"- {a}" for a in bull_output.get("key_arguments", [])),
        bear_conviction=bear_output.get("conviction", 50),
        bear_case=bear_output.get("bear_case", "No bear case."),
        bear_arguments="\n".join(f"- {a}" for a in bear_output.get("key_arguments", [])),
        market_regime=macro_context.get("regime", "neutral"),
        sector_rating=macro_context.get("sector_rating", {}).get("rating", "market_weight"),
    )
    prompt = check_prompt_size(prompt, MAX_TOKENS_STRATEGIC, caller=f"judge_agent:{ticker}")

    try:
        response = client.messages.create(
            model=STRATEGIC_MODEL,
            max_tokens=MAX_TOKENS_STRATEGIC,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _extract_json(response.content[0].text, "verdict")
        if result:
            result["conviction_score"] = max(0, min(100, int(result.get("conviction_score", 50))))
        return result
    except Exception as e:
        logger.error("[judge_agent] %s: failed: %s", ticker, e)
        return None


def run_candidate_debate(
    ticker: str,
    news_report: str,
    research_report: str,
    analyst_data: dict,
    tech_score: float,
    macro_context: dict,
    api_key: Optional[str] = None,
) -> dict | None:
    """
    Orchestrate full bull → bear → judge debate for one candidate ticker.

    Returns combined debate result or None if debate fails entirely.
    """
    # Bull agent first
    bull = run_bull_agent(ticker, news_report, research_report, analyst_data, tech_score, api_key)
    if not bull:
        logger.warning("[debate] %s: bull agent failed — skipping debate", ticker)
        return None

    # Bear agent sees bull's arguments
    bear = run_bear_agent(ticker, news_report, research_report, analyst_data, tech_score, bull, api_key)
    if not bear:
        logger.warning("[debate] %s: bear agent failed — skipping debate", ticker)
        return None

    # Judge synthesizes
    judge = run_judge_agent(ticker, bull, bear, macro_context, api_key)
    if not judge:
        logger.warning("[debate] %s: judge agent failed — returning bull/bear without verdict", ticker)

    result = {
        "ticker": ticker,
        "bull_case": bull.get("bull_case", ""),
        "bull_arguments": bull.get("key_arguments", []),
        "bull_conviction": bull.get("conviction", 50),
        "bear_case": bear.get("bear_case", ""),
        "bear_arguments": bear.get("key_arguments", []),
        "bear_conviction": bear.get("conviction", 50),
    }

    if judge:
        result.update({
            "verdict": judge.get("verdict", "neutral"),
            "conviction_score": judge.get("conviction_score", 50),
            "rationale": judge.get("rationale", ""),
            "key_condition": judge.get("key_condition", ""),
        })

    logger.info(
        "[debate] %s: bull=%d bear=%d → verdict=%s conviction=%d",
        ticker, result["bull_conviction"], result["bear_conviction"],
        result.get("verdict", "N/A"), result.get("conviction_score", 0),
    )

    return result
