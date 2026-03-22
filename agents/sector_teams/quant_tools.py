"""
Tools for the Quant Analyst agent — LangChain @tool wrappers around existing fetchers.

Tools are created via factory functions that close over shared context (price_data,
technical_scores). The quant agent calls these via LangGraph's create_react_agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

log = logging.getLogger(__name__)


def create_quant_tools(context: dict) -> list:
    """
    Create LangChain tools for the quant analyst, closing over shared context.

    Args:
        context: Shared data dict with price_data, technical_scores, etc.

    Returns:
        List of LangChain tool callables.
    """
    price_data = context.get("price_data", {})
    technical_scores = context.get("technical_scores", {})

    @tool
    def screen_by_volume(tickers: list[str], min_volume: float) -> str:
        """Filter tickers by minimum 20-day average daily volume. Returns tickers meeting threshold."""
        passing = []
        for t in tickers:
            df = price_data.get(t)
            if df is not None and len(df) >= 20 and "Volume" in df.columns:
                avg_vol = df["Volume"].tail(20).mean()
                if avg_vol >= min_volume:
                    passing.append({"ticker": t, "avg_volume_20d": int(avg_vol)})
        return json.dumps({"passing_tickers": len(passing), "tickers": passing[:50]})

    @tool
    def get_technical_indicators(tickers: list[str]) -> str:
        """Get technical indicators: RSI(14), MACD, price vs MA50/MA200, momentum, ATR%, technical_score (0-100)."""
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

    @tool
    def get_analyst_consensus(tickers: list[str]) -> str:
        """Get analyst ratings, price targets, earnings surprises. Returns consensus_rating, num_analysts, mean_target, upside_pct."""
        from data.fetchers.analyst_fetcher import fetch_analyst_consensus as _fetch

        results = {}
        for t in tickers[:20]:
            try:
                data = _fetch(t)
                results[t] = {
                    "consensus_rating": data.get("consensus_rating", "N/A"),
                    "num_analysts": data.get("num_analysts", 0),
                    "mean_target": data.get("mean_target"),
                    "upside_pct": round(data.get("upside_pct", 0), 1) if data.get("upside_pct") else None,
                }
            except Exception as e:
                results[t] = {"error": str(e)}
        return json.dumps(results)

    @tool
    def get_balance_sheet(tickers: list[str]) -> str:
        """Get balance sheet metrics: debt/equity, current ratio, PE, revenue growth, gross margins."""
        import yfinance as yf

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

    @tool
    def get_price_performance(tickers: list[str]) -> str:
        """Get recent price performance: 5d, 20d, 60d returns and current price."""
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

    @tool
    def get_options_flow(tickers: list[str]) -> str:
        """Get options signals: put/call ratio, IV rank, expected move. Gauges market sentiment."""
        from data.fetchers.options_fetcher import fetch_options_data

        results = {}
        for t in tickers[:10]:
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

    return [
        screen_by_volume,
        get_technical_indicators,
        get_analyst_consensus,
        get_balance_sheet,
        get_price_performance,
        get_options_flow,
    ]
