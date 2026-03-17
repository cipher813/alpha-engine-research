"""
Macro data fetcher — FRED CSV API (free) + commodity/index prices via yfinance.

OPEN ITEM: Set FRED_API_KEY environment variable before running.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_TIMEOUT = 15


def _fred_latest(series_id: str, api_key: str) -> Optional[float]:
    """Fetch the most recent observation for a FRED series."""
    try:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        resp = requests.get(_FRED_BASE, params=params, timeout=_FRED_TIMEOUT)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        for o in obs:
            val = o.get("value", ".")
            if val != ".":
                return float(val)
    except Exception as e:
        logger.warning("FRED fetch failed for %s: %s", series_id, e)
    return None


def fetch_macro_data() -> dict:
    """
    Fetch current macro data from FRED and yfinance.

    Returns dict with:
      fed_funds_rate, treasury_2yr, treasury_10yr, yield_curve_slope,
      vix, unemployment, cpi_yoy, sp500_close, sp500_30d_return,
      qqq_30d_return, iwm_30d_return, oil_wti, gold, copper
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable not set.")

    # ── FRED data ─────────────────────────────────────────────────────────────
    fred_series = {
        "fed_funds_rate": "FEDFUNDS",
        "treasury_2yr": "DGS2",
        "treasury_10yr": "DGS10",
        "vix": "VIXCLS",
        "unemployment": "UNRATE",
        "cpi_yoy": "CPIAUCSL",  # we'll compute YoY below
    }

    macro = {}
    for key, series_id in fred_series.items():
        macro[key] = _fred_latest(series_id, api_key)

    # Yield curve slope (10yr - 2yr in bps)
    if macro.get("treasury_10yr") and macro.get("treasury_2yr"):
        macro["yield_curve_slope"] = round(
            (macro["treasury_10yr"] - macro["treasury_2yr"]) * 100, 1
        )
    else:
        macro["yield_curve_slope"] = None

    # CPI YoY: compare current to 12 months ago
    macro["cpi_yoy"] = _fred_cpi_yoy(api_key)

    # ── yfinance: commodities + indices ───────────────────────────────────────
    commodity_tickers = ["CL=F", "GC=F", "HG=F"]
    index_tickers = ["SPY", "QQQ", "IWM"]
    all_tickers = commodity_tickers + index_tickers

    try:
        df = yf.download(
            all_tickers,
            period="35d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        def _last_close(ticker: str) -> Optional[float]:
            try:
                s = df[ticker]["Close"].dropna()
                return float(s.iloc[-1]) if not s.empty else None
            except Exception as e:
                logger.debug("_last_close failed for %s: %s", ticker, e)
                return None

        def _return_30d(ticker: str) -> Optional[float]:
            try:
                s = df[ticker]["Close"].dropna()
                if len(s) >= 20:
                    return round(((s.iloc[-1] / s.iloc[-20]) - 1) * 100, 2)
            except Exception as e:
                logger.debug("_return_30d failed for %s: %s", ticker, e)
            return None

        macro["oil_wti"] = _last_close("CL=F")
        macro["gold"] = _last_close("GC=F")
        macro["copper"] = _last_close("HG=F")
        macro["sp500_close"] = _last_close("SPY")
        macro["sp500_30d_return"] = _return_30d("SPY")
        macro["qqq_30d_return"] = _return_30d("QQQ")
        macro["iwm_30d_return"] = _return_30d("IWM")

    except Exception as e:
        logger.warning("yfinance macro download failed: %s", e)
        for k in ["oil_wti", "gold", "copper", "sp500_close",
                  "sp500_30d_return", "qqq_30d_return", "iwm_30d_return"]:
            macro.setdefault(k, None)

    macro["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return macro


def _fred_cpi_yoy(api_key: str) -> Optional[float]:
    """Compute CPI YoY% by comparing latest vs 12 months prior."""
    try:
        params = {
            "series_id": "CPIAUCSL",
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 15,
        }
        resp = requests.get(_FRED_BASE, params=params, timeout=_FRED_TIMEOUT)
        resp.raise_for_status()
        obs = [o for o in resp.json().get("observations", []) if o["value"] != "."]
        if len(obs) < 13:
            return None
        latest = float(obs[0]["value"])
        year_ago = float(obs[12]["value"])
        return round((latest / year_ago - 1) * 100, 2)
    except Exception as e:
        logger.warning("CPI YoY computation failed: %s", e)
        return None
