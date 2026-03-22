"""
Tools for the Quant Analyst agent — wrap existing data fetchers for ReAct tool calling.

Each tool is defined as an Anthropic tool_use schema + an executor function.
The quant agent calls these iteratively to screen and rank sector candidates.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# ── Tool Definitions (Anthropic tool_use schema) ─────────────────────────────

QUANT_TOOLS: list[dict] = [
    {
        "name": "screen_by_volume",
        "description": "Filter tickers by minimum average daily volume (20-day). Returns tickers meeting the threshold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to screen"},
                "min_volume": {"type": "number", "description": "Minimum 20-day average volume (e.g., 500000)"},
            },
            "required": ["tickers", "min_volume"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": "Get technical indicators for tickers: RSI(14), MACD, price vs MA50/MA200, 20d momentum, ATR%, technical_score (0-100).",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to analyze"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_analyst_consensus",
        "description": "Get analyst ratings, price targets, and earnings surprises for tickers. Returns consensus_rating, num_analysts, mean_target, upside_pct.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to look up"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_balance_sheet",
        "description": "Get key balance sheet metrics: debt_to_equity, current_ratio, market_cap. Some sectors (Financials, Real Estate) have different capital structures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to check"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_price_performance",
        "description": "Get recent price performance: 5d, 20d, 60d, YTD returns and current price for tickers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to check"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_options_flow",
        "description": "Get options market signals: put/call ratio, IV rank, expected move. Useful for gauging market sentiment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "Tickers to check"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "submit_ranked_picks",
        "description": "Submit your final ranked list of top 10 candidates with quant scores. Call this when you have completed your analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "quant_score": {"type": "number", "description": "0-100 quantitative attractiveness"},
                            "rationale": {"type": "string", "description": "1-2 sentence rationale for this pick"},
                            "key_metrics": {"type": "object", "description": "Key metrics that drove the decision"},
                        },
                        "required": ["ticker", "quant_score", "rationale"],
                    },
                    "description": "Ranked list (best first), max 10",
                },
            },
            "required": ["picks"],
        },
    },
]


# ── Tool Executors ────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: dict,
    context: dict,
) -> str:
    """
    Execute a quant tool and return the result as a JSON string.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Input parameters from the agent.
        context: Shared context dict with price_data, sector_map, etc.

    Returns:
        JSON string with tool results.
    """
    try:
        if tool_name == "screen_by_volume":
            return _screen_by_volume(tool_input, context)
        elif tool_name == "get_technical_indicators":
            return _get_technical_indicators(tool_input, context)
        elif tool_name == "get_analyst_consensus":
            return _get_analyst_consensus(tool_input, context)
        elif tool_name == "get_balance_sheet":
            return _get_balance_sheet(tool_input, context)
        elif tool_name == "get_price_performance":
            return _get_price_performance(tool_input, context)
        elif tool_name == "get_options_flow":
            return _get_options_flow(tool_input, context)
        elif tool_name == "submit_ranked_picks":
            return _submit_ranked_picks(tool_input, context)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        log.warning("Tool %s failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})


def _screen_by_volume(inp: dict, ctx: dict) -> str:
    """Filter tickers by 20-day average volume."""
    tickers = inp["tickers"]
    min_vol = inp["min_volume"]
    price_data = ctx.get("price_data", {})

    passing = []
    for t in tickers:
        df = price_data.get(t)
        if df is not None and hasattr(df, "Volume") and len(df) >= 20:
            avg_vol = df["Volume"].tail(20).mean()
            if avg_vol >= min_vol:
                passing.append({"ticker": t, "avg_volume_20d": int(avg_vol)})
        elif df is not None and len(df) >= 20:
            try:
                avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
                if avg_vol >= min_vol:
                    passing.append({"ticker": t, "avg_volume_20d": int(avg_vol)})
            except Exception:
                pass

    return json.dumps({"passing_tickers": len(passing), "tickers": passing[:50]})


def _get_technical_indicators(inp: dict, ctx: dict) -> str:
    """Compute technical indicators for requested tickers."""
    tickers = inp["tickers"]
    price_data = ctx.get("price_data", {})
    technical_scores = ctx.get("technical_scores", {})

    results = {}
    for t in tickers:
        ts = technical_scores.get(t, {})
        if ts:
            results[t] = {
                "rsi_14": round(ts.get("rsi_14", 0), 1),
                "macd_cross": ts.get("macd_cross", False),
                "price_vs_ma50": round(ts.get("price_vs_ma50", 0), 2),
                "price_vs_ma200": round(ts.get("price_vs_ma200", 0), 2),
                "momentum_20d": round(ts.get("momentum_20d", 0), 2),
                "atr_pct": round(ts.get("atr_pct", 0), 2),
                "technical_score": round(ts.get("technical_score", 0), 1),
            }
        else:
            results[t] = {"error": "no data available"}

    return json.dumps(results)


def _get_analyst_consensus(inp: dict, ctx: dict) -> str:
    """Fetch analyst consensus for tickers."""
    from data.fetchers.analyst_fetcher import fetch_analyst_consensus

    tickers = inp["tickers"]
    results = {}
    for t in tickers[:20]:  # cap at 20 to limit API calls
        try:
            data = fetch_analyst_consensus(t)
            results[t] = {
                "consensus_rating": data.get("consensus_rating", "N/A"),
                "num_analysts": data.get("num_analysts", 0),
                "mean_target": data.get("mean_target"),
                "upside_pct": round(data.get("upside_pct", 0), 1) if data.get("upside_pct") else None,
            }
        except Exception as e:
            results[t] = {"error": str(e)}

    return json.dumps(results)


def _get_balance_sheet(inp: dict, ctx: dict) -> str:
    """Fetch balance sheet metrics via yfinance."""
    import yfinance as yf

    tickers = inp["tickers"]
    results = {}
    for t in tickers[:20]:
        try:
            info = yf.Ticker(t).info
            results[t] = {
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margins": info.get("grossMargins"),
            }
        except Exception as e:
            results[t] = {"error": str(e)}

    return json.dumps(results)


def _get_price_performance(inp: dict, ctx: dict) -> str:
    """Compute recent price performance from cached price data."""
    tickers = inp["tickers"]
    price_data = ctx.get("price_data", {})

    results = {}
    for t in tickers:
        df = price_data.get(t)
        if df is None or len(df) < 5:
            results[t] = {"error": "insufficient price data"}
            continue

        close = df["Close"] if "Close" in df.columns else df["Adj Close"]
        current = float(close.iloc[-1])
        results[t] = {"current_price": round(current, 2)}

        for label, days in [("5d", 5), ("20d", 20), ("60d", 60)]:
            if len(close) >= days:
                prior = float(close.iloc[-days])
                results[t][f"return_{label}"] = round((current / prior - 1) * 100, 2)

    return json.dumps(results)


def _get_options_flow(inp: dict, ctx: dict) -> str:
    """Fetch options flow data."""
    from data.fetchers.options_fetcher import fetch_options_data

    tickers = inp["tickers"]
    results = {}
    for t in tickers[:10]:  # cap — options fetch is slow
        try:
            data = fetch_options_data(t)
            results[t] = {
                "put_call_ratio": round(data.get("put_call_ratio", 1.0), 2),
                "iv_rank": round(data.get("iv_rank", 50), 1),
                "expected_move_pct": round(data.get("expected_move_pct", 0), 2),
            }
        except Exception as e:
            results[t] = {"error": str(e)}

    return json.dumps(results)


def _submit_ranked_picks(inp: dict, ctx: dict) -> str:
    """Validate and store the agent's final picks."""
    picks = inp.get("picks", [])
    if not picks:
        return json.dumps({"error": "No picks submitted"})
    if len(picks) > 10:
        picks = picks[:10]

    ctx["_final_picks"] = picks
    return json.dumps({"status": "accepted", "count": len(picks)})
