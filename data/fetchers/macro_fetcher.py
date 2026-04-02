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


def _load_macro_from_s3() -> dict | None:
    """Try to load macro data from alpha-engine-data's S3 output."""
    try:
        import boto3
        import json
        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        prefix = "market_data/"
        s3 = boto3.client("s3")
        ptr = s3.get_object(Bucket=bucket, Key=f"{prefix}latest_weekly.json")
        pointer = json.loads(ptr["Body"].read())
        s3_prefix = pointer.get("s3_prefix", "")
        if not s3_prefix:
            return None
        obj = s3.get_object(Bucket=bucket, Key=f"{s3_prefix}macro.json")
        data = json.loads(obj["Body"].read())
        if data and data.get("fed_funds_rate") is not None:
            logger.info("Loaded macro data from S3 (date=%s)", pointer.get("date"))
            return data
    except Exception as e:
        logger.debug("S3 macro load failed: %s", e)
    return None


def fetch_macro_data() -> dict:
    """
    Fetch current macro data.

    Priority: S3 (alpha-engine-data) → FRED + yfinance.

    Returns dict with:
      fed_funds_rate, treasury_2yr, treasury_10yr, yield_curve_slope,
      vix, unemployment, cpi_yoy,
      consumer_sentiment, initial_claims, hy_credit_spread_oas,
      sp500_close, sp500_30d_return, qqq_30d_return, iwm_30d_return,
      oil_wti, gold, copper
    """
    # S3-first: try pre-collected data from alpha-engine-data
    s3_data = _load_macro_from_s3()
    if s3_data is not None:
        return s3_data

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

    # ── yfinance: commodities + indices ───────────────────────────────────────
    # yfinance batch download is more efficient here (1 call for 6 tickers)
    # than polygon per-ticker calls (6 calls at 5/min rate limit).
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
