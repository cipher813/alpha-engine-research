"""
Price fetcher — downloads daily OHLCV data and computes technical indicators.
Uses yfinance (free, no API key required).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetchError(RuntimeError):
    """Raised when price data fetch returns insufficient results."""
    pass


_BATCH_SIZE = 100
_DOWNLOAD_WORKERS = 5


def _download_batch(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
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
    Download daily OHLCV for a list of tickers in parallel batches of 100.
    Returns dict of ticker → DataFrame with columns [Open, High, Low, Close, Volume].
    Missing tickers are silently skipped (empty DataFrame).
    """
    import concurrent.futures

    if not tickers:
        return {}

    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    result: dict[str, pd.DataFrame] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_batch, batch, period): batch for batch in batches}
        failed_batches = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result.update(future.result())
            except Exception as e:
                failed_batches.append(futures[future])
                logger.warning("Batch download failed (%d tickers), will retry: %s", len(futures[future]), e)

    # Retry failed batches sequentially (no parallelism to avoid yfinance race conditions)
    for batch in failed_batches:
        try:
            result.update(_download_batch(batch, period))
        except Exception as e:
            logger.warning("Batch retry failed (%d tickers), skipping: %s", len(batch), e)

    # ── Catastrophic failure gate ──────────────────────────────────────────
    n_requested = len(tickers)
    n_fetched = sum(1 for df in result.values() if not df.empty)
    if n_requested > 0 and n_fetched < max(n_requested * 0.10, 1):
        msg = (f"Price fetch catastrophic failure: {n_fetched}/{n_requested} "
               f"tickers with data ({n_fetched/n_requested*100:.0f}%)")
        logger.error(msg)
        raise PriceFetchError(msg)

    if n_requested > 0 and n_fetched < n_requested * 0.50:
        logger.warning(
            "Price fetch degraded: %d/%d tickers with data (%.0f%%)",
            n_fetched, n_requested, n_fetched / n_requested * 100,
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

    cache_path = Path(__file__).parent.parent.parent / "data" / "constituents_cache.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

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
        if tickers:
            cache_df = pd.DataFrame({
                "ticker": tickers,
                "sector": [sector_map.get(t, "Unknown") for t in tickers],
            })
            cache_df.to_csv(cache_path, index=False)
            print(f"  Fetched {len(tickers)} S&P 500+400 constituents with sectors from Wikipedia.")
        return tickers, sector_map

    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}); trying cache...")
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
            print(f"  Loaded {len(ticker_list)} tickers from cache ({has_sectors} sectors).")
            return ticker_list, sector_map
        print("  No cache found. Scanner will have no universe.")
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
    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
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
        except Exception as e:
            logger.debug("Short interest fetch failed for %s: %s", ticker, e)
            results[ticker] = {
                "short_pct_float": None,
                "short_ratio": None,
                "shares_short": None,
            }
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
