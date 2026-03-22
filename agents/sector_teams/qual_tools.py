"""
Tools for the Qualitative Analyst agent — wrap existing data fetchers for ReAct tool calling.

The qual analyst reviews the quant's top 5 picks, using qualitative data sources
(news, analyst reports, insider activity, SEC filings) to form holistic conviction.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# ── Tool Definitions ──────────────────────────────────────────────────────────

QUAL_TOOLS: list[dict] = [
    {
        "name": "get_news_articles",
        "description": "Get recent news headlines, excerpts, and article hashes for a ticker. Useful for understanding market narrative and sentiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "description": "Lookback days (default 7)", "default": 7},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_analyst_reports",
        "description": "Get analyst consensus, price target, recent rating changes, and earnings surprises for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_insider_activity",
        "description": "Get insider transactions, cluster buying signals, and net sentiment for a ticker. Cluster buying (3+ insiders in 30d) is a strong bullish signal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_sec_filings",
        "description": "Get recent SEC filings (8-K, 10-K, 10-Q) for a ticker. Useful for understanding corporate actions and disclosures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_prior_thesis",
        "description": "Get the prior structured thesis for a ticker (bull case, bear case, catalysts, risks, conviction). Returns None if the stock has never been analyzed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_options_flow",
        "description": "Get options market signals: put/call ratio, IV rank, expected move for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_institutional_activity",
        "description": "Get 13F-based institutional accumulation signals. Useful for understanding if large funds are building positions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "submit_qualitative_assessment",
        "description": "Submit your qualitative assessment for all reviewed stocks. Call this when done analyzing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "assessments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "qual_score": {"type": "number", "description": "0-100 qualitative conviction"},
                            "bull_case": {"type": "string", "description": "Key bullish argument (1-2 sentences)"},
                            "bear_case": {"type": "string", "description": "Key bearish risk (1-2 sentences)"},
                            "catalysts": {"type": "array", "items": {"type": "string"}, "description": "Upcoming catalysts"},
                            "conviction": {"type": "string", "enum": ["high", "medium", "low"]},
                            "resources_used": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Which tools most influenced your assessment (e.g., 'news', 'analyst', 'insider')",
                            },
                        },
                        "required": ["ticker", "qual_score", "bull_case", "bear_case", "conviction"],
                    },
                },
                "additional_candidate": {
                    "type": "object",
                    "description": "Optional: 1 additional stock not in quant's picks that you think deserves consideration",
                    "properties": {
                        "ticker": {"type": "string"},
                        "qual_score": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            },
            "required": ["assessments"],
        },
    },
]


# ── Tool Executors ────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: dict,
    context: dict,
) -> str:
    """Execute a qual tool and return JSON string result."""
    try:
        if tool_name == "get_news_articles":
            return _get_news_articles(tool_input, context)
        elif tool_name == "get_analyst_reports":
            return _get_analyst_reports(tool_input, context)
        elif tool_name == "get_insider_activity":
            return _get_insider_activity(tool_input, context)
        elif tool_name == "get_sec_filings":
            return _get_sec_filings(tool_input, context)
        elif tool_name == "get_prior_thesis":
            return _get_prior_thesis(tool_input, context)
        elif tool_name == "get_options_flow":
            return _get_options_flow(tool_input, context)
        elif tool_name == "get_institutional_activity":
            return _get_institutional_activity(tool_input, context)
        elif tool_name == "submit_qualitative_assessment":
            return _submit_assessment(tool_input, context)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        log.warning("Qual tool %s failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})


def _get_news_articles(inp: dict, ctx: dict) -> str:
    from data.fetchers.news_fetcher import fetch_news_for_ticker
    ticker = inp["ticker"]
    days = inp.get("days", 7)
    try:
        articles = fetch_news_for_ticker(ticker, lookback_days=days)
        # Limit to avoid huge tool responses
        trimmed = []
        for a in articles[:10]:
            trimmed.append({
                "headline": a.get("headline", ""),
                "source": a.get("source", ""),
                "published": a.get("published_utc", ""),
                "excerpt": (a.get("article_excerpt", "") or "")[:300],
            })
        return json.dumps({"ticker": ticker, "article_count": len(articles), "articles": trimmed})
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _get_analyst_reports(inp: dict, ctx: dict) -> str:
    from data.fetchers.analyst_fetcher import fetch_analyst_consensus
    ticker = inp["ticker"]
    try:
        data = fetch_analyst_consensus(ticker)
        return json.dumps({
            "ticker": ticker,
            "consensus_rating": data.get("consensus_rating", "N/A"),
            "num_analysts": data.get("num_analysts", 0),
            "mean_target": data.get("mean_target"),
            "upside_pct": round(data.get("upside_pct", 0), 1) if data.get("upside_pct") else None,
            "rating_changes": data.get("rating_changes", [])[:5],
            "earnings_surprises": data.get("earnings_surprises", [])[:4],
        })
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _get_insider_activity(inp: dict, ctx: dict) -> str:
    from data.fetchers.insider_fetcher import fetch_insider_activity
    ticker = inp["ticker"]
    try:
        data = fetch_insider_activity(ticker)
        return json.dumps({
            "ticker": ticker,
            "cluster_buy": data.get("cluster_buy", False),
            "unique_buyers_30d": data.get("unique_buyers_30d", 0),
            "total_buy_value_30d": data.get("total_buy_value_30d", 0),
            "net_sentiment": data.get("net_sentiment", 0),
            "recent_transactions": data.get("transactions", [])[:5],
        })
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _get_sec_filings(inp: dict, ctx: dict) -> str:
    from data.fetchers.news_fetcher import fetch_sec_filings
    ticker = inp["ticker"]
    try:
        filings = fetch_sec_filings(ticker)
        trimmed = [{"title": f.get("title", ""), "date": f.get("date", ""), "form_type": f.get("form_type", "")}
                    for f in filings[:5]]
        return json.dumps({"ticker": ticker, "filings": trimmed})
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _get_prior_thesis(inp: dict, ctx: dict) -> str:
    prior_theses = ctx.get("prior_theses", {})
    ticker = inp["ticker"]
    thesis = prior_theses.get(ticker)
    if thesis:
        return json.dumps({
            "ticker": ticker,
            "bull_case": thesis.get("bull_case", ""),
            "bear_case": thesis.get("bear_case", ""),
            "catalysts": thesis.get("catalysts", []),
            "risks": thesis.get("risks", []),
            "conviction": thesis.get("conviction_rationale", ""),
            "last_updated": thesis.get("last_updated", ""),
        })
    return json.dumps({"ticker": ticker, "prior_thesis": None})


def _get_options_flow(inp: dict, ctx: dict) -> str:
    from data.fetchers.options_fetcher import fetch_options_data
    ticker = inp["ticker"]
    try:
        data = fetch_options_data(ticker)
        return json.dumps({
            "ticker": ticker,
            "put_call_ratio": round(data.get("put_call_ratio", 1.0), 2),
            "iv_rank": round(data.get("iv_rank", 50), 1),
            "expected_move_pct": round(data.get("expected_move_pct", 0), 2),
        })
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _get_institutional_activity(inp: dict, ctx: dict) -> str:
    from data.fetchers.institutional_fetcher import fetch_institutional_activity
    ticker = inp["ticker"]
    try:
        data = fetch_institutional_activity(ticker)
        return json.dumps({
            "ticker": ticker,
            "n_funds_accumulating": data.get("n_funds_accumulating", 0),
            "accumulation_signal": data.get("accumulation_signal", False),
            "total_new_shares": data.get("total_new_shares", 0),
        })
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


def _submit_assessment(inp: dict, ctx: dict) -> str:
    assessments = inp.get("assessments", [])
    additional = inp.get("additional_candidate")
    if not assessments:
        return json.dumps({"error": "No assessments submitted"})

    ctx["_final_assessments"] = assessments
    ctx["_additional_candidate"] = additional
    return json.dumps({
        "status": "accepted",
        "count": len(assessments),
        "additional": additional.get("ticker") if additional else None,
    })
