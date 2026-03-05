"""
News & Sentiment Agent (§4.1).

One instance per tracked stock (up to 23 parallel invocations).
Uses claude-haiku-4-5 (per-stock model) — templated incremental update.

Implements the three-step thesis drafting protocol (§7.3):
  1. Start from existing archived report
  2. Add material new findings
  3. Remove stale/outdated content
"""

from __future__ import annotations

import json
import re
from typing import Optional

import anthropic

from config import PER_STOCK_MODEL, MAX_TOKENS_PER_STOCK, ANTHROPIC_API_KEY

_PROMPT_TEMPLATE = """\
You are a financial news analyst maintaining an ongoing intelligence
brief for {ticker} ({company_name}).

PRIOR REPORT (from {prior_date}):
{prior_report}
[If "NONE — initial report", produce a fresh brief from the data below.]

NEW ARTICLES (last 24–48 hours — novel only, duplicates pre-filtered):
{new_articles}
← Format per article: HEADLINE | SOURCE | ARTICLE_EXCERPT (first 400 chars of body)
   Note: articles already seen in the prior run have been deduplicated before reaching you.

RECURRING THEMES (appeared in ≥3 articles this run):
{recurring_themes}
← Format: THEME | mention_count | example_headline
   High mention_count = high market salience; weight accordingly.

NEW SEC FILINGS (last 24–48 hours):
{new_filings}

CURRENT PRICE: ${price} | RECENT MOVE: {price_change_pct}% ({price_change_date})

THESIS DRAFTING PROTOCOL — FOLLOW IN ORDER:
1. START WITH EXISTING: The prior report above is your baseline. Preserve
   every finding that remains valid. Do not rewrite what hasn't changed.
2. ADD NEW FINDINGS: Integrate material new articles or filings. Use the
   article body excerpt (not just the headline) to assess substance.
   Weight recurring themes by their mention_count — a theme appearing in
   10 articles matters more than one appearing in 1.
3. REMOVE STALE CONTENT: Remove events that are resolved, superseded, or
   older than 10 trading days and no longer relevant. When in doubt, retain.

Keep the report to approximately 300 words.

news_score (0-100): Expected relative attractiveness for the next ~12 months vs the market.
  Higher = more likely to outperform SPY; lower = more likely to underperform.

End with a JSON block:
{{"news_score": <0-100>, "sentiment": "<positive|neutral|negative>",
 "key_catalyst": "<one sentence>", "prior_date": "{prior_date}",
 "material_changes": <true|false>,
 "dominant_theme": "<recurring theme if any, else null>",
 "dominant_theme_count": <mention_count or 0>}}

Output the full refreshed report followed by the JSON block.
"""


def _format_articles(articles: list[dict]) -> str:
    if not articles:
        return "No new articles."
    lines = []
    for a in articles:
        headline = a.get("headline", "")
        source = a.get("source", "")
        excerpt = a.get("article_excerpt", "")[:400]
        lines.append(f"{headline} | {source} | {excerpt}")
    return "\n".join(lines)


def _format_recurring_themes(themes: list[dict]) -> str:
    if not themes:
        return "None."
    lines = []
    for t in themes:
        lines.append(f"{t['theme']} | {t['mention_count']} | {t.get('example_headline', '')}")
    return "\n".join(lines)


def _format_filings(filings: list[dict]) -> str:
    if not filings:
        return "None."
    lines = []
    for f in filings:
        lines.append(f"{f.get('form_type', '8-K')} | {f.get('date', '')} | {f.get('title', '')}")
    return "\n".join(lines)


def _extract_json_from_response(text: str) -> dict:
    """Extract the trailing JSON block from agent response."""
    match = re.search(r"\{[^{}]*\"news_score\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"news_score": 50, "sentiment": "neutral", "material_changes": False}


def run_news_agent(
    ticker: str,
    company_name: str,
    prior_report: Optional[str],
    prior_date: str,
    new_articles: list[dict],
    recurring_themes: list[dict],
    sec_filings: list[dict],
    current_price: float,
    price_change_pct: float,
    price_change_date: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the News & Sentiment Agent for a single ticker.

    Returns dict with:
      report_md: str — refreshed ~300-word markdown report
      news_json: dict — structured JSON output from agent
      ticker: str
    """
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    prior_text = prior_report or "NONE — initial report"
    articles_text = _format_articles(new_articles)
    themes_text = _format_recurring_themes(recurring_themes)
    filings_text = _format_filings(sec_filings)

    prompt = _PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        prior_date=prior_date,
        prior_report=prior_text,
        new_articles=articles_text,
        recurring_themes=themes_text,
        new_filings=filings_text,
        price=f"{current_price:.2f}",
        price_change_pct=f"{price_change_pct:+.1f}",
        price_change_date=price_change_date,
    )

    response = client.messages.create(
        model=PER_STOCK_MODEL,
        max_tokens=MAX_TOKENS_PER_STOCK,
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = response.content[0].text
    news_json = _extract_json_from_response(full_text)

    # Strip the JSON block from the markdown report
    report_md = re.sub(r"\{[^{}]*\"news_score\"[^{}]*\}", "", full_text, flags=re.DOTALL).strip()

    return {
        "ticker": ticker,
        "report_md": report_md,
        "news_json": news_json,
        "news_score": float(news_json.get("news_score", 50)),
        "material_changes": bool(news_json.get("material_changes", False)),
    }
