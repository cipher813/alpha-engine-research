"""
Analyst consensus fetcher — Financial Modeling Prep (FMP) API.
Free tier: 250 requests/day. Daily usage ~73 — within free tier (§15.4).

OPEN ITEM: Set FMP_API_KEY environment variable before running.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 10


def _fmp_get(endpoint: str, params: Optional[dict] = None) -> dict | list:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")

    url = f"{_FMP_BASE}/{endpoint}"
    p = {"apikey": api_key}
    if params:
        p.update(params)

    resp = requests.get(url, params=p, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_analyst_consensus(ticker: str) -> dict:
    """
    Fetch analyst consensus rating, mean price target, and number of analysts.
    Returns dict with keys: consensus_rating, mean_target, num_analysts,
    current_price, upside_pct, rating_changes (list of recent actions).
    """
    result = {
        "ticker": ticker,
        "consensus_rating": None,
        "mean_target": None,
        "num_analysts": None,
        "current_price": None,
        "upside_pct": None,
        "rating_changes": [],
        "earnings_surprises": [],
    }

    # Analyst estimates / consensus
    try:
        data = _fmp_get(f"analyst-stock-recommendations/{ticker}", {"limit": 1})
        if isinstance(data, list) and data:
            rec = data[0]
            result["consensus_rating"] = rec.get("analystRatingsStrongBuy") and _derive_consensus(rec)
    except Exception:
        pass

    # Price target consensus
    try:
        data = _fmp_get(f"price-target-consensus/{ticker}")
        if isinstance(data, list) and data:
            pt = data[0]
            result["mean_target"] = pt.get("targetConsensus")
            result["num_analysts"] = pt.get("numberOfAnalysts")
    except Exception:
        pass

    # Current price (from FMP quote)
    try:
        data = _fmp_get(f"quote/{ticker}")
        if isinstance(data, list) and data:
            q = data[0]
            result["current_price"] = q.get("price")
            if result["mean_target"] and result["current_price"]:
                result["upside_pct"] = round(
                    (result["mean_target"] / result["current_price"] - 1) * 100, 1
                )
    except Exception:
        pass

    # Recent rating changes (last 30 days)
    try:
        data = _fmp_get(f"upgrades-downgrades/{ticker}", {"limit": 10})
        if isinstance(data, list):
            result["rating_changes"] = [
                {
                    "date": r.get("publishedDate", "")[:10],
                    "firm": r.get("gradingCompany", ""),
                    "action": r.get("action", ""),
                    "from_grade": r.get("previousGrade", ""),
                    "to_grade": r.get("newGrade", ""),
                }
                for r in data[:5]
            ]
    except Exception:
        pass

    # Earnings surprises (last 2 quarters)
    try:
        data = _fmp_get(f"earnings-surprises/{ticker}", {"limit": 2})
        if isinstance(data, list):
            result["earnings_surprises"] = [
                {
                    "date": r.get("date", "")[:10],
                    "actual": r.get("actualEarningResult"),
                    "estimated": r.get("estimatedEarning"),
                    "surprise_pct": (
                        round(
                            (r["actualEarningResult"] / abs(r["estimatedEarning"]) - 1) * 100, 1
                        )
                        if r.get("estimatedEarning")
                        else None
                    ),
                }
                for r in data[:2]
            ]
    except Exception:
        pass

    return result


def _derive_consensus(rec: dict) -> str:
    """Derive a plain-English consensus string from FMP rating counts."""
    strong_buy = rec.get("analystRatingsStrongBuy", 0) or 0
    buy = rec.get("analystRatingsbuy", 0) or 0
    hold = rec.get("analystRatingsHold", 0) or 0
    sell = rec.get("analystRatingsSell", 0) or 0
    strong_sell = rec.get("analystRatingsStrongSell", 0) or 0
    total = strong_buy + buy + hold + sell + strong_sell
    if total == 0:
        return "Hold"
    bullish = (strong_buy + buy) / total
    bearish = (sell + strong_sell) / total
    if bullish >= 0.7:
        return "Strong Buy"
    if bullish >= 0.5:
        return "Buy"
    if bearish >= 0.5:
        return "Sell"
    if bearish >= 0.3:
        return "Underperform"
    return "Hold"
