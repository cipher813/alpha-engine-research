"""
Analyst consensus fetcher — Financial Modeling Prep (FMP) stable API.
Free tier: 250 requests/day. Daily usage ~73 — within free tier (§15.4).

Uses stable API endpoints (v3 legacy endpoints are no longer supported).
"""

from __future__ import annotations

import os
from typing import Optional

import requests

_FMP_STABLE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 10


def _fmp_get(endpoint: str, params: Optional[dict] = None) -> dict | list:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")

    url = f"{_FMP_STABLE}/{endpoint}"
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
    current_price, upside_pct.
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

    # Analyst grades consensus (strongBuy/buy/hold/sell counts + pre-computed consensus)
    try:
        data = _fmp_get("grades-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            g = data[0]
            result["consensus_rating"] = g.get("consensus")
            total = sum(g.get(k, 0) or 0 for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
            result["num_analysts"] = total or None
    except Exception:
        pass

    # Price target consensus
    try:
        data = _fmp_get("price-target-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            pt = data[0]
            result["mean_target"] = pt.get("targetConsensus")
    except Exception:
        pass

    # Current price
    try:
        data = _fmp_get("quote", {"symbol": ticker})
        if isinstance(data, list) and data:
            q = data[0]
            result["current_price"] = q.get("price")
            if result["mean_target"] and result["current_price"]:
                result["upside_pct"] = round(
                    (result["mean_target"] / result["current_price"] - 1) * 100, 1
                )
    except Exception:
        pass

    return result
