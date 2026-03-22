"""
Analyst consensus fetcher — Financial Modeling Prep (FMP) stable API.
Free tier: 250 requests/day. Daily usage ~73 — within free tier (§15.4).

Uses stable API endpoints for consensus data. Also fetches earnings
surprises from the v3 API for PEAD scoring (O10).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_FMP_STABLE = "https://financialmodelingprep.com/stable"
_FMP_V3 = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 10

# Rate limiter: FMP free tier = 250 req/day, ~5 req/sec safe.
# With 6 sector teams calling in parallel, we need a global lock.
_fmp_lock = threading.Lock()
_fmp_last_call = 0.0
_FMP_MIN_INTERVAL = 0.25  # 250ms between calls = max 4 req/sec
_FMP_MAX_RETRIES = 3
_FMP_RETRY_BACKOFF = 2.0  # seconds, doubles each retry


def _fmp_get(endpoint: str, params: Optional[dict] = None, base: str = _FMP_STABLE) -> dict | list:
    global _fmp_last_call
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")

    url = f"{base}/{endpoint}"
    p = {"apikey": api_key}
    if params:
        p.update(params)

    for attempt in range(_FMP_MAX_RETRIES):
        # Rate limit: ensure minimum interval between calls
        with _fmp_lock:
            now = time.monotonic()
            wait = _FMP_MIN_INTERVAL - (now - _fmp_last_call)
            if wait > 0:
                time.sleep(wait)
            _fmp_last_call = time.monotonic()

        resp = requests.get(url, params=p, timeout=_TIMEOUT)

        if resp.status_code == 429:
            backoff = _FMP_RETRY_BACKOFF * (2 ** attempt)
            logger.warning("FMP 429 for %s — retrying in %.1fs (attempt %d/%d)",
                           endpoint, backoff, attempt + 1, _FMP_MAX_RETRIES)
            time.sleep(backoff)
            continue

        resp.raise_for_status()
        return resp.json()

    # Final attempt failed
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
    except Exception as e:
        logger.warning("FMP grades-consensus failed for %s: %s", ticker, e)

    # Price target consensus
    try:
        data = _fmp_get("price-target-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            pt = data[0]
            result["mean_target"] = pt.get("targetConsensus")
    except Exception as e:
        logger.warning("FMP price-target-consensus failed for %s: %s", ticker, e)

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
    except Exception as e:
        logger.warning("FMP quote failed for %s: %s", ticker, e)

    # O10: Earnings surprises (uses v3 API)
    try:
        data = _fmp_get(f"earning_surprises/{ticker}", base=_FMP_V3)
        if isinstance(data, list) and data:
            surprises = []
            for entry in data[:4]:  # last 4 quarters
                actual = entry.get("actualEarningResult")
                estimated = entry.get("estimatedEarning")
                surprise_pct = None
                if actual is not None and estimated is not None and estimated != 0:
                    surprise_pct = round((actual - estimated) / abs(estimated) * 100, 2)
                surprises.append({
                    "date": entry.get("date", ""),
                    "actual": actual,
                    "estimated": estimated,
                    "surprise_pct": surprise_pct,
                })
            result["earnings_surprises"] = surprises
    except Exception as e:
        logger.debug("FMP earnings surprises failed for %s: %s", ticker, e)

    return result
