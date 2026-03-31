"""
Macro data fetcher — FRED CSV API (free) + commodity/index prices via yfinance.

Fetches:
  - Core rates & indicators (Fed Funds, treasuries, VIX, unemployment, CPI)
  - Leading indicators (ISM PMI, Initial Jobless Claims, HY Credit Spread OAS)
  - Commodities (WTI oil, gold, copper) and index returns (SPY, QQQ, IWM)
  - Equity breadth (% above 50d/200d MA, advance/decline) from scanner price data
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_TIMEOUT = 15


def _fred_latest(series_id: str, api_key: str) -> Optional[float]:
    """Fetch the most recent observation for a FRED series (with retry)."""
    last_err = None
    for attempt in range(1, 3):  # 2 attempts
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
            return None
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < 2:
                logger.warning("FRED fetch failed for %s (attempt %d/2): %s — retrying in 3s", series_id, attempt, e)
                time.sleep(3)
            else:
                logger.warning("FRED fetch failed for %s after 2 attempts: %s", series_id, e)
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            return None
    return None


def compute_market_breadth(price_data: dict[str, pd.DataFrame]) -> dict:
    """
    Compute equity breadth metrics from ~900 S&P 500+400 stocks.

    Returns dict with:
      pct_above_50d_ma:  % of stocks trading above their 50-day MA
      pct_above_200d_ma: % of stocks trading above their 200-day MA
      advance_decline_ratio: advancers / decliners over last 5 trading days
      n_stocks: number of stocks with valid data
    """
    above_50d = 0
    total_50d = 0
    above_200d = 0
    total_200d = 0
    advancers = 0
    decliners = 0

    for ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 10:
            continue

        close = df["Close"]
        current = float(close.iloc[-1])

        # 50-day MA breadth
        if len(close) >= 50:
            ma50 = float(close.rolling(50).mean().iloc[-1])
            total_50d += 1
            if current > ma50:
                above_50d += 1

        # 200-day MA breadth
        if len(close) >= 200:
            ma200 = float(close.rolling(200).mean().iloc[-1])
            total_200d += 1
            if current > ma200:
                above_200d += 1

        # 5-day advance/decline
        if len(close) >= 6:
            five_day_return = current / float(close.iloc[-6]) - 1
            if five_day_return > 0:
                advancers += 1
            elif five_day_return < 0:
                decliners += 1

    result = {
        "pct_above_50d_ma": round(above_50d / total_50d * 100, 1) if total_50d > 0 else None,
        "pct_above_200d_ma": round(above_200d / total_200d * 100, 1) if total_200d > 0 else None,
        "advance_decline_ratio": round(advancers / max(decliners, 1), 2),
        "n_stocks": max(total_50d, total_200d),
    }
    logger.info(
        "[breadth] above_50dMA=%.1f%% above_200dMA=%.1f%% A/D=%.2f n=%d",
        result["pct_above_50d_ma"] or 0,
        result["pct_above_200d_ma"] or 0,
        result["advance_decline_ratio"],
        result["n_stocks"],
    )
    return result


def fetch_macro_data() -> dict:
    """
    Fetch current macro data from FRED and yfinance.

    Returns dict with:
      fed_funds_rate, treasury_2yr, treasury_10yr, yield_curve_slope,
      vix, unemployment, cpi_yoy,
      consumer_sentiment, initial_claims, hy_credit_spread_oas,
      sp500_close, sp500_30d_return, qqq_30d_return, iwm_30d_return,
      oil_wti, gold, copper
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
        # Leading indicators
        "consumer_sentiment": "UMCSENT",      # U. Michigan Consumer Sentiment
        "initial_claims": "ICSA",            # Initial Jobless Claims (weekly, thousands)
        "hy_credit_spread_oas": "BAMLH0A0HYM2",  # ICE BofA HY OAS (bps)
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

    # ── Commodities + indices (polygon primary, yfinance fallback) ───────────
    # Polygon free tier doesn't support futures (CL=F etc.) — use ETF proxies.
    # Commodity mapping: CL=F→USO (oil), GC=F→GLD (gold), HG=F→CPER (copper)
    _POLYGON_TICKERS = ["USO", "GLD", "CPER", "SPY", "QQQ", "IWM"]
    _COMMODITY_MAP = {"USO": "oil_wti", "GLD": "gold", "CPER": "copper"}
    _INDEX_MAP = {"SPY": "sp500_close", "QQQ": "qqq", "IWM": "iwm"}

    polygon_ok = False
    try:
        from polygon_client import polygon_client
        client = polygon_client()
        # Fetch 35 trading days for 30d return calc
        from datetime import timedelta as _td
        _end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _start = (datetime.now(timezone.utc) - _td(days=50)).strftime("%Y-%m-%d")

        poly_data: dict[str, pd.DataFrame] = {}
        for t in _POLYGON_TICKERS:
            bars = client.get_daily_bars(t, _start, _end)
            if not bars.empty:
                poly_data[t] = bars

        if len(poly_data) >= 4:  # at least most tickers succeeded
            polygon_ok = True
            for t, key in _COMMODITY_MAP.items():
                if t in poly_data and not poly_data[t].empty:
                    macro[key] = float(poly_data[t]["Close"].iloc[-1])
            if "SPY" in poly_data and not poly_data["SPY"].empty:
                macro["sp500_close"] = float(poly_data["SPY"]["Close"].iloc[-1])
            for t in ["SPY", "QQQ", "IWM"]:
                if t in poly_data and len(poly_data[t]) >= 20:
                    s = poly_data[t]["Close"]
                    ret = round(((float(s.iloc[-1]) / float(s.iloc[-20])) - 1) * 100, 2)
                    macro[f"{_INDEX_MAP.get(t, t.lower())}_30d_return"] = ret
            logger.info("Macro prices fetched from polygon (%d tickers)", len(poly_data))
    except Exception as e:
        logger.warning("Polygon macro fetch failed: %s", e)

    if not polygon_ok:
        # Fallback to yfinance with futures tickers
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
