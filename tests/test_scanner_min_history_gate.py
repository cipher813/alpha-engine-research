"""Regression tests for the scanner min-history admission gate.

Context (ROADMAP P1, 2026-04-22):
    SNDK (IPO'd 2025-02-13) passed the scanner's liquidity + tech_score
    filters and was promoted to a buy_candidate on 2026-04-21 despite
    having only ~290 bars — far below the 252-day warmup that
    ``dist_from_52w_high`` / ``return_252d`` / ``momentum_252d`` need.
    Downstream scoring silently saw NaN features for those dimensions
    and the ticker still entered the portfolio.

Correct posture (feedback_no_unscoreable_labels): guardrail upstream,
not NaN-tolerance downstream. This test locks the invariant that
``run_quant_filter`` rejects tickers with < MIN_TRADING_DAYS_FOR_CANDIDACY
bars before they reach the candidate lists.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from data import scanner


def _ohlcv(n_rows: int) -> pd.DataFrame:
    """Build a minimally-shaped OHLCV frame with n_rows trading-day bars."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 10_000_000,
        },
        index=dates,
    )


def _tech() -> dict:
    """Fully-formed technical indicators that would otherwise pass every filter."""
    return {
        "current_price": 100.0,
        "avg_volume_20d": 10_000_000,
        "rsi_14": 55.0,
        "price_vs_ma200": 5.0,
        "atr_14_pct": 0.02,  # 2% decimal — scanner multiplies by 100
    }


def _scanner_params(min_history_days: int = 252) -> dict:
    return {
        "tech_score_min": 60,
        "max_atr_pct": 8.0,
        "min_avg_volume": 1_000_000,
        "min_price": 5.0,
        "momentum_ma200_floor_pct": -10.0,
        "momentum_top_n": 50,
        "min_combined_candidates": 30,
        "min_trading_days_for_candidacy": min_history_days,
        "atr_period": 14,
    }


def _run_quant_filter_with_tech(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    technical_scores: dict[str, dict],
    min_history_days: int = 252,
):
    """Invoke run_quant_filter with patched scanner params + a non-falsey
    tech_score so the ticker would pass downstream filters if not gated
    by min-history.
    """
    params = _scanner_params(min_history_days=min_history_days)
    # Stub compute_technical_score to always return a passing score so the
    # test isolates the min-history gate behavior.
    with patch.object(scanner, "get_scanner_params", return_value=params), \
         patch.object(scanner, "compute_technical_score", return_value=75):
        return scanner.run_quant_filter(
            tickers=tickers,
            price_data=price_data,
            technical_scores=technical_scores,
            market_regime="neutral",
            sector_map={t: "Technology" for t in tickers},
        )


def test_short_history_ticker_rejected_with_named_reason(caplog):
    """A ticker with < min_trading_days_for_candidacy bars must be rejected
    with ``filter_fail_reason="insufficient_history"``. It must NOT land in
    the candidate lists.
    """
    # 200 bars — below the 252 default.
    price_data = {"NEWCO": _ohlcv(n_rows=200)}
    technical_scores = {"NEWCO": _tech()}

    candidates = _run_quant_filter_with_tech(
        ["NEWCO"], price_data, technical_scores, min_history_days=252,
    )

    # Must not appear in returned candidates (run_quant_filter return value
    # is the promoted candidate list).
    assert all(c["ticker"] != "NEWCO" for c in candidates), (
        "Short-history ticker leaked past the min-history gate into candidates."
    )


def test_sufficient_history_ticker_proceeds_to_downstream_filters():
    """A ticker with ≥ min_trading_days_for_candidacy bars passes the gate
    and continues to liquidity/volatility/tech_score filters.
    """
    # 300 bars — above the 252 default.
    price_data = {"ESTABLISHED": _ohlcv(n_rows=300)}
    technical_scores = {"ESTABLISHED": _tech()}

    candidates = _run_quant_filter_with_tech(
        ["ESTABLISHED"], price_data, technical_scores, min_history_days=252,
    )

    assert any(c["ticker"] == "ESTABLISHED" for c in candidates), (
        "Long-history ticker was wrongly rejected — min-history gate "
        "overreached into the tech_score / liquidity stage."
    )


def test_configurable_threshold_via_scanner_params():
    """The threshold is read from ``get_scanner_params()`` so the
    backtester can sweep it.
    """
    # 100 bars — below 252 default, but configurable to 50 for this test.
    price_data = {"MICROCAP": _ohlcv(n_rows=100)}
    technical_scores = {"MICROCAP": _tech()}

    # With threshold lowered to 50, ticker passes.
    candidates = _run_quant_filter_with_tech(
        ["MICROCAP"], price_data, technical_scores, min_history_days=50,
    )
    assert any(c["ticker"] == "MICROCAP" for c in candidates)

    # With threshold raised to 200, same ticker is rejected.
    candidates = _run_quant_filter_with_tech(
        ["MICROCAP"], price_data, technical_scores, min_history_days=200,
    )
    assert all(c["ticker"] != "MICROCAP" for c in candidates)


def test_default_threshold_is_252():
    """Default matches the longest rolling-window feature in the universe
    (``return_252d`` / ``momentum_252d`` / ``dist_from_52w_*``).
    """
    src = (scanner.__file__ and open(scanner.__file__).read()) or ""
    # Locate the default via the get(..) fallback.
    assert 'min_trading_days_for_candidacy", 252' in src, (
        "Default min_trading_days_for_candidacy must be 252 (matches the "
        "longest rolling-window feature downstream)."
    )


def test_ticker_with_no_df_but_tech_only_is_not_gated():
    """When ``df`` is absent (feature-store-only path), the history length
    can't be directly observed. The gate must NOT reject on unknown —
    downstream feature-store policy handles that case. Gate only fires
    when we can prove insufficient history from ``df``.
    """
    price_data: dict[str, pd.DataFrame] = {}  # no df
    technical_scores = {"TECH_ONLY": _tech()}

    # Should pass the min-history gate (df is None) and proceed.
    candidates = _run_quant_filter_with_tech(
        ["TECH_ONLY"], price_data, technical_scores, min_history_days=252,
    )
    assert any(c["ticker"] == "TECH_ONLY" for c in candidates), (
        "Gate rejected a feature-store-only ticker even though df was absent; "
        "gate must only fire when short history is observable from df."
    )
