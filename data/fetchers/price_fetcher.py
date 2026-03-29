"""
Price fetcher — downloads daily OHLCV data and computes technical indicators.
Uses yfinance (free, no API key required).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from retry import retry

logger = logging.getLogger(__name__)


class PriceFetchError(RuntimeError):
    """Raised when price data fetch returns insufficient results."""
    pass


_BATCH_SIZE = 100
_BATCH_DELAY_SECS = 2
_MIN_FETCH_RATIO = 0.80
_MIN_EXPECTED_CONSTITUENTS = 800  # S&P 500 (~503) + S&P 400 (~400), minus overlaps

# ── yfinance request counter (per-Lambda-invocation) ─────────────────────────
_yf_request_count = 0


def get_yf_request_count() -> int:
    """Return total yfinance requests made this invocation."""
    return _yf_request_count


def _make_rate_limited_session():
    """Create a requests session with rate limiting (max 2 requests per 5 seconds)."""
    try:
        from requests_ratelimiter import LimiterSession, RequestRate, Limiter, Duration
        rate = RequestRate(2, Duration.SECOND * 5)
        session = LimiterSession(limiter=Limiter(rate))
        session.headers["User-agent"] = "alpha-engine-research/1.0"
        return session
    except ImportError:
        logger.warning("requests_ratelimiter not installed, using default session")
        return None


@retry(max_attempts=2, retryable=(Exception,), label="yfinance")
def _download_batch(tickers: list[str], period: str, session=None) -> dict[str, pd.DataFrame]:
    """Download one batch of tickers and return per-ticker DataFrames."""
    result: dict[str, pd.DataFrame] = {}
    if not tickers:
        return result

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
        session=session,
    )

    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Close"])
        result[ticker] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].copy()
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["Close"])
                result[ticker] = df
            except (KeyError, AttributeError):
                result[ticker] = pd.DataFrame()

    return result


def fetch_price_data(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for a list of tickers in sequential batches of 100
    with a rate-limited session and delay between batches to avoid Yahoo rate limits.
    Returns dict of ticker → DataFrame with columns [Open, High, Low, Close, Volume].
    Raises PriceFetchError if fewer than 80% of tickers return data.
    """
    import time

    if not tickers:
        return {}

    global _yf_request_count

    session = _make_rate_limited_session()
    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    result: dict[str, pd.DataFrame] = {}

    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(_BATCH_DELAY_SECS)
        try:
            result.update(_download_batch(batch, period, session=session))
            _yf_request_count += 1
            n_ok = sum(1 for t in batch if t in result and not result[t].empty)
            logger.info("Batch %d/%d: %d/%d tickers OK", i + 1, len(batches), n_ok, len(batch))
        except Exception as e:
            _yf_request_count += 1
            logger.warning("Batch %d/%d failed (%d tickers): %s", i + 1, len(batches), len(batch), e)

    # ── Data quality gate ─────────────────────────────────────────────────
    n_requested = len(tickers)
    n_fetched = sum(1 for df in result.values() if not df.empty)
    pct = n_fetched / n_requested * 100 if n_requested > 0 else 0

    if n_requested > 0 and n_fetched < n_requested * _MIN_FETCH_RATIO:
        msg = (f"Price fetch failed: {n_fetched}/{n_requested} tickers "
               f"with data ({pct:.0f}%) — minimum {_MIN_FETCH_RATIO:.0%} required")
        logger.error(msg)
        raise PriceFetchError(msg)

    logger.info(
        "Price fetch complete: %d/%d tickers (%.0f%%), %d batches, %d total yf requests this run",
        n_fetched, n_requested, pct, len(batches), _yf_request_count,
    )
    return result


# Wikipedia GICS sector names → internal sector names used throughout the system
_GICS_SECTOR_MAP = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financial",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
}


def fetch_sp500_sp400_with_sectors() -> tuple[list[str], dict[str, str]]:
    """
    Fetch S&P 500 and S&P 400 constituent lists AND GICS sectors from Wikipedia.
    Falls back to cached static CSV if Wikipedia is unavailable.

    Known limitation — survivorship bias:
        Wikipedia provides current-only constituent lists. Stocks that were removed
        from the index (delisted, acquired, or dropped due to market cap decline)
        are not included. This means historical backtests using these lists will
        overstate returns because they exclude stocks that performed poorly enough
        to be removed. Impact is small for this system's weekly horizon: S&P indices
        rebalance quarterly with ~20-30 changes/year. For historical constituent
        data, paid sources (Compustat, Sharadar) would be needed.

    Returns:
        (tickers, sector_map) where:
        - tickers: deduplicated list of ticker symbols
        - sector_map: {ticker: internal_sector_name} for all tickers
    """
    import requests
    from pathlib import Path
    from io import StringIO

    # Writable cache in /tmp (Lambda's /var/task is read-only); fall back to baked-in copy
    baked_cache = Path(__file__).parent.parent.parent / "data" / "constituents_cache.csv"
    cache_path = Path("/tmp/constituents_cache.csv")
    if not cache_path.exists() and baked_cache.exists():
        import shutil
        shutil.copy2(baked_cache, cache_path)

    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp400_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    headers = {
        "User-Agent": "alpha-engine-research/1.0 (https://github.com/; research bot)"
    }

    tickers: list[str] = []
    sector_map: dict[str, str] = {}

    try:
        for url in [sp500_url, sp400_url]:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

            # Find ticker/symbol column
            col = next(
                (c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()),
                df.columns[0],
            )

            # Find GICS sector column
            sector_col = next(
                (c for c in df.columns if "gics" in str(c).lower() and "sector" in str(c).lower()),
                None,
            )
            if sector_col is None:
                # Broader match for tables without "GICS" prefix
                sector_col = next(
                    (c for c in df.columns if "sector" in str(c).lower()),
                    None,
                )

            for _, row in df.iterrows():
                raw_ticker = str(row[col]).strip().replace(".", "-")
                if not raw_ticker or raw_ticker == "nan" or len(raw_ticker) > 6:
                    continue
                tickers.append(raw_ticker)

                if sector_col is not None:
                    raw_sector = str(row[sector_col]).strip()
                    mapped = _GICS_SECTOR_MAP.get(raw_sector, raw_sector)
                    sector_map[raw_ticker] = mapped

        tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order

        # Sanity check: Wikipedia should return ~900 tickers
        if tickers and len(tickers) < _MIN_EXPECTED_CONSTITUENTS:
            logger.error(
                "Wikipedia returned only %d tickers (expected >= %d) — falling back to cache",
                len(tickers), _MIN_EXPECTED_CONSTITUENTS,
            )
            raise ValueError(f"Insufficient tickers from Wikipedia: {len(tickers)}")

        if tickers:
            cache_df = pd.DataFrame({
                "ticker": tickers,
                "sector": [sector_map.get(t, "Unknown") for t in tickers],
            })
            cache_df.to_csv(cache_path, index=False)
            logger.info("Fetched %d S&P 500+400 constituents with sectors from Wikipedia", len(tickers))
        return tickers, sector_map

    except Exception as e:
        logger.warning("Wikipedia fetch failed (%s); trying cache...", e)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            ticker_list = cached["ticker"].tolist()
            if "sector" in cached.columns:
                sector_map = {
                    str(t): str(s)
                    for t, s in zip(cached["ticker"], cached["sector"])
                    if s and str(s) != "nan"
                }
            has_sectors = "with" if sector_map else "without"
            logger.info("Loaded %d tickers from cache (%s sectors)", len(ticker_list), has_sectors)
            return ticker_list, sector_map
        logger.error("No cache found. Scanner will have no universe.")
        return [], {}


def fetch_sp500_sp400_tickers() -> list[str]:
    """
    Fetch S&P 500 and S&P 400 constituent lists from Wikipedia.
    Falls back to cached static CSV if Wikipedia is unavailable.
    Returns deduplicated list of tickers.
    """
    tickers, _ = fetch_sp500_sp400_with_sectors()
    return tickers


def fetch_short_interest(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch short interest data from yfinance for a list of tickers.

    Uses Ticker.info fields: shortPercentOfFloat, shortRatio, sharesShort.
    Only call for analyzed tickers (~35), not the full S&P 900.

    Note: yfinance short interest data is delayed (bi-monthly FINRA reporting).
    Best used as a supplementary signal, not for precise timing.

    Returns {ticker: {short_pct_float, short_ratio, shares_short}}.
    """
    global _yf_request_count

    results: dict[str, dict] = {}
    n_ok = 0
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            _yf_request_count += 1
            short_pct = info.get("shortPercentOfFloat")  # e.g. 0.05 = 5%
            short_ratio = info.get("shortRatio")          # days to cover
            shares_short = info.get("sharesShort")

            # Convert to percentage if present
            if short_pct is not None:
                short_pct = round(short_pct * 100, 2)

            results[ticker] = {
                "short_pct_float": short_pct,
                "short_ratio": short_ratio,
                "shares_short": shares_short,
            }
            if short_pct is not None:
                n_ok += 1
        except Exception as e:
            _yf_request_count += 1
            logger.debug("Short interest fetch failed for %s: %s", ticker, e)
            results[ticker] = {
                "short_pct_float": None,
                "short_ratio": None,
                "shares_short": None,
            }

    logger.info(
        "Short interest: %d/%d tickers with data, %d total yf requests this run",
        n_ok, len(tickers), _yf_request_count,
    )
    return results


def compute_technical_indicators(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute RSI(14), MACD signal, price vs MA50, price vs MA200,
    20-day momentum, and 20-day average volume from a price DataFrame.
    Returns None if insufficient data.
    """
    if df.empty or len(df) < 30:
        return None

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)

    # ── RSI 14 ──────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # ── MACD ────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    macd_cross = 0.0  # no cross
    if len(macd_line) >= 2:
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        if prev_diff < 0 and curr_diff >= 0:
            macd_cross = 1.0   # bullish cross
        elif prev_diff > 0 and curr_diff <= 0:
            macd_cross = -1.0  # bearish cross
    macd_above_zero = bool(macd_line.iloc[-1] > 0)

    # ── Moving Averages ──────────────────────────────────────────────────────
    current_price = float(close.iloc[-1])

    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    price_vs_ma50 = ((current_price / ma50) - 1) * 100 if ma50 else None
    price_vs_ma200 = ((current_price / ma200) - 1) * 100 if ma200 else None

    # ── 20-day Momentum ──────────────────────────────────────────────────────
    momentum_20d = None
    if len(close) >= 21:
        momentum_20d = float(((close.iloc[-1] / close.iloc[-21]) - 1) * 100)

    # ── Average Volume ───────────────────────────────────────────────────────
    avg_volume_20d = None
    if not volume.empty and len(volume) >= 20:
        avg_volume_20d = float(volume.tail(20).mean())

    return {
        "rsi_14": rsi_14,
        "macd_cross": macd_cross,
        "macd_above_zero": macd_above_zero,
        "macd_line_last": float(macd_line.iloc[-1]),
        "signal_line_last": float(signal_line.iloc[-1]),
        "current_price": current_price,
        "ma50": ma50,
        "ma200": ma200,
        "price_vs_ma50": price_vs_ma50,
        "price_vs_ma200": price_vs_ma200,
        "momentum_20d": momentum_20d,
        "avg_volume_20d": avg_volume_20d,
    }
