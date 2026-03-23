"""
Tools for the Qualitative Analyst agent — LangChain @tool wrappers.

The qual analyst reviews the quant's top 5 picks using qualitative data sources.
Tools are created via factory that closes over shared context (prior_theses).
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)


def create_qual_tools(context: dict) -> list:
    """
    Create LangChain tools for the qual analyst, closing over shared context.

    Args:
        context: Dict with prior_theses and other shared data.

    Returns:
        List of LangChain tool callables.
    """
    prior_theses = context.get("prior_theses", {})
    price_data = context.get("price_data", {})

    @tool
    def get_news_articles(ticker: str, days: int = 7) -> str:
        """Get recent news headlines and excerpts for a ticker. Useful for understanding market narrative."""
        from data.fetchers.news_fetcher import fetch_news_for_ticker

        try:
            articles = fetch_news_for_ticker(ticker, lookback_days=days)
            trimmed = [
                {"headline": a.get("headline", ""), "source": a.get("source", ""),
                 "published": a.get("published_utc", ""),
                 "excerpt": (a.get("article_excerpt", "") or "")[:300]}
                for a in articles[:10]
            ]
            return json.dumps({"ticker": ticker, "article_count": len(articles), "articles": trimmed})
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_analyst_reports(ticker: str) -> str:
        """Get analyst consensus, price target, rating changes, earnings surprises for a ticker."""
        from data.fetchers.analyst_fetcher import fetch_analyst_consensus

        try:
            cp = None
            df = price_data.get(ticker)
            if df is not None and not df.empty and "Close" in df.columns:
                cp = float(df["Close"].iloc[-1])
            data = fetch_analyst_consensus(ticker, current_price=cp)
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

    @tool
    def get_insider_activity(ticker: str) -> str:
        """Get insider transactions and cluster buying signals. Cluster buying (3+ insiders in 30d) is strongly bullish."""
        from data.fetchers.insider_fetcher import fetch_insider_activity as _fetch

        try:
            data = _fetch(ticker)
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

    @tool
    def get_sec_filings(ticker: str) -> str:
        """Get recent SEC filings (8-K, 10-K, 10-Q) for corporate actions and disclosures."""
        from data.fetchers.news_fetcher import fetch_sec_filings

        try:
            filings = fetch_sec_filings(ticker)
            trimmed = [{"title": f.get("title", ""), "date": f.get("date", ""),
                        "form_type": f.get("form_type", "")} for f in filings[:5]]
            return json.dumps({"ticker": ticker, "filings": trimmed})
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_prior_thesis(ticker: str) -> str:
        """Get prior structured thesis (bull/bear case, catalysts, risks). Returns null if never analyzed."""
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

    @tool
    def get_options_flow(ticker: str) -> str:
        """Get options market signals: put/call ratio, IV rank, expected move."""
        from data.fetchers.options_fetcher import fetch_options_data

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

    @tool
    def get_institutional_activity(ticker: str) -> str:
        """Get 13F institutional accumulation signals. Shows if large funds are building positions."""
        from data.fetchers.institutional_fetcher import fetch_institutional_activity as _fetch

        try:
            data = _fetch(ticker)
            return json.dumps({
                "ticker": ticker,
                "n_funds_accumulating": data.get("n_funds_accumulating", 0),
                "accumulation_signal": data.get("accumulation_signal", False),
                "total_new_shares": data.get("total_new_shares", 0),
            })
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    return [
        get_news_articles,
        get_analyst_reports,
        get_insider_activity,
        get_sec_filings,
        get_prior_thesis,
        get_options_flow,
        get_institutional_activity,
    ]
