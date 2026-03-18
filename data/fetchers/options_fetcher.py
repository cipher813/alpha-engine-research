"""
data/fetchers/options_fetcher.py — Options-derived signals (O12).

Fetches options chain data from yfinance to compute put/call ratio,
IV rank, and expected move. These signals capture market positioning
orthogonal to price momentum.

yfinance options chains are slow (~1-2 sec/ticker). Run in parallel
with other fetchers when possible. Cache results to S3.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_FETCH_DELAY = 0.5  # seconds between yfinance options fetches


def fetch_options_signals(
    tickers: list[str],
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch options-derived signals for a list of tickers.

    For each ticker:
    - Put/call ratio (open interest weighted)
    - ATM implied volatility
    - IV rank approximation (using realized vol percentile)
    - Expected move percentage

    Returns per ticker:
        put_call_ratio: float — put OI / call OI (1.0 = neutral)
        iv_rank: float — IV percentile rank 0-100
        atm_iv: float — at-the-money implied volatility
        expected_move_pct: float — expected move as % of price

    Graceful degradation: returns neutral values on failure.
    """
    try:
        import yfinance
    except ImportError:
        log.warning("yfinance not available for options data")
        return {t: _neutral_result() for t in tickers}

    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            t = yfinance.Ticker(ticker)

            # Get available expiry dates
            try:
                expiries = t.options
            except Exception:
                results[ticker] = _neutral_result()
                continue

            if not expiries:
                results[ticker] = _neutral_result()
                continue

            # Select nearest monthly expiry (20-40 DTE preferred)
            expiry = _select_nearest_monthly(expiries, reference_date)
            if not expiry:
                results[ticker] = _neutral_result()
                continue

            # Fetch the options chain
            try:
                chain = t.option_chain(expiry)
            except Exception:
                results[ticker] = _neutral_result()
                time.sleep(_FETCH_DELAY)
                continue

            calls = chain.calls
            puts = chain.puts

            # Put/call ratio (open interest weighted)
            put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            pc_ratio = round(put_oi / max(call_oi, 1), 3)

            # Get current price for ATM strike selection
            info = t.info if hasattr(t, "info") else {}
            current_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
            if not current_price:
                hist = t.history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0

            # ATM implied volatility
            atm_iv = _get_atm_iv(calls, puts, current_price)

            # IV rank: approximate using 20d realized vol percentile
            # (True IV rank requires historical IV data we don't have)
            iv_rank = _approximate_iv_rank(t, atm_iv)

            # Expected move percentage
            days_to_expiry = _days_to_expiry(expiry, reference_date)
            if atm_iv > 0 and days_to_expiry > 0:
                expected_move_pct = round(
                    atm_iv * np.sqrt(days_to_expiry / 365) * 100, 2
                )
            else:
                expected_move_pct = 0.0

            results[ticker] = {
                "put_call_ratio": pc_ratio,
                "iv_rank": round(iv_rank, 1),
                "atm_iv": round(atm_iv, 4),
                "expected_move_pct": expected_move_pct,
            }

            time.sleep(_FETCH_DELAY)

        except Exception as e:
            log.warning("Options data fetch failed for %s: %s", ticker, e)
            results[ticker] = _neutral_result()

    log.info("Fetched options signals for %d/%d tickers", len(results), len(tickers))
    return results


def _neutral_result() -> dict:
    """Return neutral options data when fetching fails."""
    return {
        "put_call_ratio": 1.0,
        "iv_rank": 50.0,
        "atm_iv": 0.0,
        "expected_move_pct": 0.0,
    }


def _select_nearest_monthly(
    expiries: tuple[str, ...],
    reference_date: Optional[str] = None,
) -> Optional[str]:
    """Select the nearest expiry with 20-40 DTE (monthly preferred)."""
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()

    best = None
    best_dte = float("inf")

    for exp_str in expiries:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - today).days
            if 15 <= dte <= 60:  # wider range than 20-40 for robustness
                if abs(dte - 30) < abs(best_dte - 30):  # prefer closest to 30 DTE
                    best = exp_str
                    best_dte = dte
        except ValueError:
            continue

    # Fallback: just take the nearest expiry > 7 DTE
    if best is None:
        for exp_str in expiries:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - today).days
                if dte > 7:
                    return exp_str
            except ValueError:
                continue

    return best


def _get_atm_iv(calls, puts, current_price: float) -> float:
    """Get at-the-money implied volatility from the options chain."""
    if current_price <= 0:
        return 0.0

    try:
        # Find the strike closest to current price
        if "strike" not in calls.columns or "impliedVolatility" not in calls.columns:
            return 0.0

        call_strikes = calls["strike"].values
        if len(call_strikes) == 0:
            return 0.0

        atm_idx = np.abs(call_strikes - current_price).argmin()
        call_iv = calls.iloc[atm_idx]["impliedVolatility"]

        # Average call and put ATM IV for a better estimate
        if "strike" in puts.columns and "impliedVolatility" in puts.columns:
            put_strikes = puts["strike"].values
            if len(put_strikes) > 0:
                put_atm_idx = np.abs(put_strikes - current_price).argmin()
                put_iv = puts.iloc[put_atm_idx]["impliedVolatility"]
                return float((call_iv + put_iv) / 2)

        return float(call_iv)
    except Exception:
        return 0.0


def _approximate_iv_rank(ticker_obj, current_iv: float) -> float:
    """
    Approximate IV rank using the percentile of current IV vs
    historical 20d realized volatility over the past year.

    True IV rank needs historical IV data; this is a reasonable proxy.
    """
    if current_iv <= 0:
        return 50.0

    try:
        hist = ticker_obj.history(period="1y")
        if hist.empty or len(hist) < 30:
            return 50.0

        # Compute rolling 20d realized vol (annualized)
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 10:
            return 50.0

        # Percentile rank of current IV within historical realized vol distribution
        rank = float((rolling_vol < current_iv).sum() / len(rolling_vol) * 100)
        return min(100.0, max(0.0, rank))
    except Exception:
        return 50.0


def _days_to_expiry(expiry_str: str, reference_date: Optional[str] = None) -> int:
    """Compute calendar days to expiry."""
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return max(0, (exp_date - today).days)
    except ValueError:
        return 30  # default assumption


def cache_options_to_s3(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Cache options data to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/options/{date_str}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        log.info("Cached options data to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to cache options data to S3: %s", e)
