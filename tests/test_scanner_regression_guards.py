"""Scanner regression guards.

Complements ``test_scanner_consumer_contract.py`` (which pins the
feature-store read path) by covering two adjacent regression classes
that the consumer-contract test does NOT catch:

1. **OHLCV-fallback path units**: ``data.fetchers.price_fetcher.
   compute_technical_indicators`` is the fallback that produced the
   ONLY real scanner candidates during the 6-month silent-zero-output
   regression. If it gets accidentally normalized to match the
   feature-store ratio convention, the scanner regresses to zero
   candidates everywhere.

2. **Full-gate count regression**: the consumer-contract test pins
   ``liquidity_pass`` count only. A regression in the volatility gate,
   ``compute_technical_score``, or the momentum/deep-value path
   selection could drop production picks from ~60 → ~5 without firing
   that test. The full-gate count test runs the WHOLE scanner pipeline
   against a realistic synthetic universe and asserts ≥ N picks pass
   all gates.

Filed per the gaps analysis on PR #237. See also the plan doc
`~/Development/alpha-engine-docs/private/feature-store-schema-audit-260525.md`.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── OHLCV-fallback path: raw-units pin ───────────────────────────────────────


def _synthetic_ohlcv(
    n: int = 100,
    seed: int = 0,
    volume_low: int = 1_000_000,
    volume_high: int = 10_000_000,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame that ``compute_technical_indicators``
    can consume. Volume is in raw shares (the production scale).
    """
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(volume_low, volume_high, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_ohlcv_fallback_returns_avg_volume_in_raw_shares():
    """``compute_technical_indicators`` must emit ``avg_volume_20d`` in
    raw shares — NOT normalized.

    This is the ONLY scanner-feeding code path that was producing real
    candidates during the 6-month silent-zero-output regression (the
    feature-store path was broken; the fallback was correct). If a
    future refactor "harmonizes" this to match the feature store's
    normalized convention, the scanner regresses to zero everywhere.
    """
    from data.fetchers.price_fetcher import compute_technical_indicators

    df = _synthetic_ohlcv(n=100, volume_low=1_000_000, volume_high=10_000_000)
    indicators = compute_technical_indicators(df)

    assert indicators is not None
    avg_vol = indicators.get("avg_volume_20d")
    assert avg_vol is not None, (
        "compute_technical_indicators returned avg_volume_20d=None despite "
        "100 rows of synthetic volume — fallback path is broken."
    )
    # Synthetic volume in [1M, 10M] — rolling-20d-mean must be inside that
    # band. The scanner's MIN_AVG_VOLUME gate is 500_000; we want at least
    # an order of magnitude above the gate so the sniff has real teeth.
    assert avg_vol >= 1_000_000, (
        f"OHLCV-fallback avg_volume_20d = {avg_vol:,.0f}; expected raw "
        f"shares (>= 1M). Likely normalized — check "
        f"data/fetchers/price_fetcher.py::compute_technical_indicators."
    )
    assert avg_vol < 100_000_000, (
        f"OHLCV-fallback avg_volume_20d = {avg_vol:,.0f}; "
        "implausibly large — check unit logic."
    )


def test_ohlcv_fallback_passes_scanner_liquidity_gate():
    """End-to-end fallback path: synthetic OHLCV → compute_technical_indicators
    → scanner liquidity gate. Must pass.

    Mirrors the production fallback chain when feature-store rows are
    missing for a ticker.
    """
    from data.fetchers.price_fetcher import compute_technical_indicators
    from data.scanner import run_quant_filter

    tickers = [f"T{i:03d}" for i in range(20)]
    price_data = {t: _synthetic_ohlcv(seed=i) for i, t in enumerate(tickers)}
    sector_map = {t: "Technology" for t in tickers}

    # technical_scores empty → scanner falls through to OHLCV path.
    run_quant_filter(
        tickers=tickers,
        price_data=price_data,
        technical_scores={},
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    liquidity_pass = sum(1 for r in eval_log if r.get("liquidity_pass") == 1)
    assert liquidity_pass >= 15, (
        f"Only {liquidity_pass} / 20 OHLCV-fallback tickers passed the "
        "scanner liquidity gate. The fallback path's avg_volume_20d "
        "units may have regressed."
    )


# ── Full-gate count regression: realistic fixture, all gates exercised ──────


def _bullish_indicators(
    *,
    ticker_seed: int,
    avg_volume_raw: float = 5_000_000.0,
    current_price: float = 50.0,
    atr_pct: float = 0.018,  # raw decimal (1.8%); scanner ×100 = 1.8 < MAX_ATR_PCT
) -> dict:
    """Indicators dict matching the shape returned by
    ``_build_technical_scores_from_feature_store`` for a ticker that
    SHOULD pass all scanner gates (liquidity + volatility + tech_score
    + MA200 floor + momentum path).

    Slight per-seed variation so the technical score isn't identical
    across tickers (deduping by tech_score in the scanner top-N would
    otherwise collapse the candidate count).
    """
    rng = np.random.default_rng(ticker_seed)
    return {
        "rsi_14": 55.0 + rng.uniform(-5, 10),         # 50-65: bullish-ish
        "macd_cross": 1.0,                             # bullish cross
        "macd_above_zero": True,
        "macd_line_last": 0.5,
        "signal_line_last": 0.3,
        "current_price": current_price + rng.uniform(-10, 30),
        "ma50": None,
        "ma200": None,
        "price_vs_ma50": 0.03 + rng.uniform(-0.01, 0.03),   # +2% to +6%
        "price_vs_ma200": 0.08 + rng.uniform(-0.02, 0.05),  # +6% to +13%
        "momentum_20d": 0.04 + rng.uniform(-0.01, 0.04),    # +3% to +8%
        "momentum_5d": 0.015 + rng.uniform(-0.005, 0.015),
        "avg_volume_20d": avg_volume_raw,             # raw shares — scanner-consumer slot
        "atr_14_pct": atr_pct,                         # decimal pct
        "dist_from_52w_high": -0.04,
        "dist_from_52w_low": 0.25,
    }


def test_full_scanner_pipeline_produces_realistic_pick_count():
    """End-to-end count regression. Feed a realistic 100-ticker
    bullish-shaped fixture through the scanner; assert ≥ N pass all
    gates (liquidity + volatility + tech_score + MA200 floor + momentum
    path).

    Production target: ~50-60 picks/cycle. Synthetic fixture is
    cleanly bullish so we expect substantially all 100 to pass —
    asserting ≥ 80 catches a regression in any single gate while
    leaving headroom for incidental fixture noise.

    Catches regressions in:
      - Volatility (ATR) gate — silent if ATR computation broken
      - compute_technical_score thresholds
      - MA200 floor logic
      - Momentum path selection
      - Top-N truncation logic
    """
    from scoring.technical import compute_technical_score
    from data.scanner import run_quant_filter

    tickers = [f"T{i:03d}" for i in range(100)]
    sector_map = {t: "Technology" for t in tickers}

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        indicators = _bullish_indicators(ticker_seed=i)
        tech_score = compute_technical_score(
            indicators, sector=sector_map[ticker],
        )
        technical_scores[ticker] = {**indicators, "technical_score": tech_score}

    candidates = run_quant_filter(
        tickers=tickers,
        price_data={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )

    # The scanner caps total candidates via _momentum_top_n + deep_value.
    # The hard floor we care about is "did the gates pass at scale" —
    # i.e., the eval log shows a healthy candidate-eligible count.
    eval_log = run_quant_filter._last_eval_log
    gate_eligible = sum(
        1 for r in eval_log
        if r.get("liquidity_pass") == 1 and r.get("volatility_pass") == 1
    )
    assert gate_eligible >= 80, (
        f"Only {gate_eligible} / 100 bullish-fixture tickers passed "
        "liquidity + volatility gates. Regression in one of the gate "
        "computations. Check data/scanner.py + scoring/technical.py."
    )

    # And the final post-rank-cutoff candidate list must be non-trivial.
    # Top-N caps to ~50 in production; assert at least 30 to catch a
    # rank-cutoff regression too.
    assert len(candidates) >= 30, (
        f"Scanner returned only {len(candidates)} candidates from a "
        "100-ticker bullish fixture. Either the top-N is too tight or "
        "an upstream gate is failing silently."
    )


def test_scanner_failure_mode_volatility_gate_regression_caught():
    """Negative-control: if every ticker is too volatile (ATR breaks
    the gate), the count test SHOULD fail. Confirms the count guard
    actually exercises the volatility gate.
    """
    from scoring.technical import compute_technical_score
    from data.scanner import run_quant_filter

    tickers = [f"T{i:03d}" for i in range(100)]
    sector_map = {t: "Technology" for t in tickers}

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        # atr_pct 0.10 = 10% decimal → scanner ×100 = 10 > MAX_ATR_PCT default.
        indicators = _bullish_indicators(ticker_seed=i, atr_pct=0.10)
        tech_score = compute_technical_score(
            indicators, sector=sector_map[ticker],
        )
        technical_scores[ticker] = {**indicators, "technical_score": tech_score}

    run_quant_filter(
        tickers=tickers,
        price_data={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    vol_fail_count = sum(
        1 for r in eval_log if r.get("volatility_pass") == 0
    )
    # If the volatility gate were broken (silently passing high-ATR
    # tickers), this would be ~0. We assert it actually fires.
    assert vol_fail_count >= 80, (
        f"Negative control: feeding 100 high-ATR tickers should produce "
        f">= 80 volatility_pass=0 records; got {vol_fail_count}. The "
        "volatility gate may be silently passing through."
    )


def test_scanner_failure_mode_liquidity_gate_regression_caught():
    """Negative-control: if every ticker's raw volume is below the
    MIN_AVG_VOLUME floor, the scanner SHOULD record liquidity_pass=0.
    Confirms the count guard actually exercises the liquidity gate
    (already covered in test_scanner_consumer_contract.py but pinned
    here too for cohesion with the volatility negative-control).
    """
    from scoring.technical import compute_technical_score
    from data.scanner import run_quant_filter

    tickers = [f"T{i:03d}" for i in range(20)]
    sector_map = {t: "Technology" for t in tickers}

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        # Volume of 100K shares — below default 500K MIN_AVG_VOLUME floor.
        indicators = _bullish_indicators(ticker_seed=i, avg_volume_raw=100_000.0)
        tech_score = compute_technical_score(
            indicators, sector=sector_map[ticker],
        )
        technical_scores[ticker] = {**indicators, "technical_score": tech_score}

    run_quant_filter(
        tickers=tickers,
        price_data={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    liq_fail_count = sum(
        1 for r in eval_log if r.get("liquidity_pass") == 0
    )
    assert liq_fail_count == 20, (
        f"Negative control: feeding 20 low-volume tickers should produce "
        f"20 liquidity_pass=0 records; got {liq_fail_count}. Gate broken."
    )
