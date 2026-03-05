"""
Technical scoring engine — deterministic, no LLM.

Computes a 0–100 technical attractiveness score from price-derived indicators:
  RSI(14)            25% weight  [regime-aware thresholds]
  MACD signal cross  20% weight
  Price vs 50-day MA 20% weight
  Price vs 200-day MA 20% weight
  20-day momentum    15% weight  [percentile-ranked vs S&P 500]

See §5.1 for full scoring methodology.
"""

from __future__ import annotations

from typing import Optional


# ── Per-signal scoring ────────────────────────────────────────────────────────

def _score_rsi(rsi: float, market_regime: str = "neutral") -> float:
    """
    Score RSI (0–100) with regime-aware overbought/oversold thresholds.

    Bull regime (VIX<15, uptrend): raise overbought threshold to 80.
    Bear/caution regime: raise oversold threshold to 40 (oversold can signal
      further decline, not necessarily a buy).
    Neutral: standard 30/70 thresholds.
    """
    if market_regime == "bull":
        # Strong momentum is a signal; only truly extreme overbought is bad
        overbought = 80
        oversold = 30
        max_oversold_score = 100.0  # oversold in bull = max bullish
    elif market_regime in ("bear", "caution"):
        # Oversold can signal further decline — cap the bullish credit for being oversold.
        # RSI threshold raises to 40 but max score for oversold zone is 65 (not 100).
        overbought = 70
        oversold = 40
        max_oversold_score = 65.0
    else:  # neutral
        overbought = 70
        oversold = 30
        max_oversold_score = 100.0

    if rsi >= overbought:
        return 0.0
    if rsi <= oversold:
        return max_oversold_score
    # Linear interpolation between oversold (max_oversold_score) and overbought (0)
    return max_oversold_score * (overbought - rsi) / (overbought - oversold)


def _score_macd(macd_cross: float, macd_above_zero: bool) -> float:
    """
    Score MACD signal cross.
    Bullish cross above zero = 100
    Bullish cross below zero = 70
    No cross, above zero = 60
    No cross, below zero = 40
    Bearish cross above zero = 30
    Bearish cross below zero = 0
    """
    if macd_cross == 1.0:  # bullish cross
        return 100.0 if macd_above_zero else 70.0
    if macd_cross == -1.0:  # bearish cross
        return 30.0 if macd_above_zero else 0.0
    # No recent cross
    return 60.0 if macd_above_zero else 40.0


def _score_price_vs_ma(pct_diff: Optional[float]) -> float:
    """
    Score price relative to a moving average.
    >5% above MA → 80
    At MA (0%) → 50
    >5% below MA → 30
    Linear interpolation between those anchors.
    More extreme: capped at 100 above, 0 below.
    """
    if pct_diff is None:
        return 50.0

    if pct_diff >= 5:
        # Scale from 80 at +5% to 100 at +20%
        return min(100.0, 80.0 + (pct_diff - 5) * (20.0 / 15.0))
    if pct_diff >= 0:
        # Linear from 50 at 0% to 80 at +5%
        return 50.0 + pct_diff * 6.0
    if pct_diff > -5:
        # Linear from 50 at 0% to 30 at -5%
        return 50.0 + pct_diff * 4.0  # pct_diff is negative → drops to 30
    # Scale from 30 at -5% to 0 at -25%
    return max(0.0, 30.0 - (abs(pct_diff) - 5) * 1.5)


def _score_momentum(momentum_20d: Optional[float], percentile_rank: Optional[float] = None) -> float:
    """
    Score 20-day momentum.
    Ideally uses percentile rank within S&P 500 universe (0–100).
    Falls back to raw return mapping if percentile not available.
    """
    if percentile_rank is not None:
        return float(percentile_rank)

    if momentum_20d is None:
        return 50.0

    # Map raw 20-day return to 0–100
    # +10% return → ~80, 0% → 50, -10% → ~20, capped at extremes
    score = 50.0 + momentum_20d * 3.0
    return max(0.0, min(100.0, score))


# ── Composite score ───────────────────────────────────────────────────────────

def compute_technical_score(
    indicators: dict,
    market_regime: str = "neutral",
    momentum_percentile: Optional[float] = None,
) -> float:
    """
    Compute weighted composite technical score (0–100).

    Args:
        indicators: dict from price_fetcher.compute_technical_indicators()
        market_regime: 'bull' | 'neutral' | 'caution' | 'bear'
        momentum_percentile: percentile rank (0–100) within S&P 500 for 20d return.
                             If None, falls back to raw return mapping.

    Returns: float in [0, 100]
    """
    rsi_score = _score_rsi(
        indicators.get("rsi_14", 50.0),
        market_regime=market_regime,
    )
    macd_score = _score_macd(
        indicators.get("macd_cross", 0.0),
        indicators.get("macd_above_zero", False),
    )
    ma50_score = _score_price_vs_ma(indicators.get("price_vs_ma50"))
    ma200_score = _score_price_vs_ma(indicators.get("price_vs_ma200"))
    momentum_score = _score_momentum(
        indicators.get("momentum_20d"),
        percentile_rank=momentum_percentile,
    )

    composite = (
        rsi_score * 0.25
        + macd_score * 0.20
        + ma50_score * 0.20
        + ma200_score * 0.20
        + momentum_score * 0.15
    )

    return round(max(0.0, min(100.0, composite)), 2)


def compute_momentum_percentiles(
    momentum_data: dict[str, Optional[float]],
) -> dict[str, float]:
    """
    Compute percentile ranks for 20d momentum across a universe of tickers.
    Returns {ticker: percentile_rank_0_to_100}.
    """
    import numpy as np

    valid = [(t, m) for t, m in momentum_data.items() if m is not None]
    if not valid:
        return {t: 50.0 for t in momentum_data}

    tickers, values = zip(*valid)
    values_arr = np.array(values, dtype=float)
    ranks = (values_arr.argsort().argsort() / max(len(values_arr) - 1, 1)) * 100

    result = {t: round(float(r), 1) for t, r in zip(tickers, ranks)}
    # Fill any missing (None momentum) with 50
    for t in momentum_data:
        result.setdefault(t, 50.0)
    return result
