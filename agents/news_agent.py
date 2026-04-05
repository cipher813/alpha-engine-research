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

from config import PER_STOCK_MODEL, MAX_TOKENS_PER_STOCK, ANTHROPIC_API_KEY, PRIOR_REPORT_MAX_CHARS
from agents.token_guard import check_prompt_size

from agents.prompt_loader import load_prompt

_PROMPT_TEMPLATE = load_prompt("news_agent")


def _truncate_prior(text: str, max_chars: int | None = None) -> str:
    """Truncate prior report to last max_chars characters to manage prompt size."""
    if max_chars is None:
        max_chars = PRIOR_REPORT_MAX_CHARS
    if not text or len(text) <= max_chars:
        return text
    return "[...truncated...]\n" + text[-max_chars:]


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
    match = re.search(r"\{[^{}]*\"news_score_short\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"news_score_short": None, "news_score_long": None, "sentiment": "neutral", "material_changes": False, "score_failed": True}


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

    prior_text = _truncate_prior(prior_report) if prior_report else "NONE — initial report"
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

    prompt = check_prompt_size(prompt, MAX_TOKENS_PER_STOCK, caller=f"news_agent:{ticker}")

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
        "news_score": float(news_json.get("news_score_short", 50)),
        "news_score_lt": float(news_json.get("news_score_long", 50)),
        "material_changes": bool(news_json.get("material_changes", False)),
    }
