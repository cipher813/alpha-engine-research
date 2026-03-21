"""Tests for the technical scoring engine and aggregator."""

import pytest
from scoring.technical import (
    _score_rsi,
    _score_macd,
    _score_price_vs_ma,
    _score_momentum,
    compute_technical_score,
    compute_momentum_percentiles,
)
from scoring.aggregator import (
    compute_attractiveness_score,
    score_to_rating,
    check_consistency,
)


class TestRSIScoring:
    def test_neutral_regime_oversold(self):
        assert _score_rsi(25, "neutral") == 100.0

    def test_neutral_regime_overbought(self):
        assert _score_rsi(75, "neutral") == 0.0

    def test_neutral_regime_midpoint(self):
        score = _score_rsi(50, "neutral")
        assert 45 <= score <= 55  # roughly in middle

    def test_bull_regime_raises_overbought_threshold(self):
        # RSI=75 should not be overbought in bull regime
        bull_score = _score_rsi(75, "bull")
        neutral_score = _score_rsi(75, "neutral")
        assert bull_score > neutral_score

    def test_bear_regime_raises_oversold_threshold(self):
        # RSI=35 should not be as bullish in bear regime
        bear_score = _score_rsi(35, "bear")
        neutral_score = _score_rsi(35, "neutral")
        assert bear_score < neutral_score

    def test_score_in_valid_range(self):
        for rsi in [0, 15, 30, 50, 70, 85, 100]:
            for regime in ["bull", "neutral", "caution", "bear"]:
                score = _score_rsi(rsi, regime)
                assert 0.0 <= score <= 100.0


class TestMACDScoring:
    def test_bullish_cross_above_zero(self):
        assert _score_macd(1.0, True) == 100.0

    def test_bullish_cross_below_zero(self):
        assert _score_macd(1.0, False) == 70.0

    def test_bearish_cross_above_zero(self):
        assert _score_macd(-1.0, True) == 30.0

    def test_bearish_cross_below_zero(self):
        assert _score_macd(-1.0, False) == 0.0

    def test_no_cross_above_zero(self):
        assert _score_macd(0.0, True) == 60.0

    def test_no_cross_below_zero(self):
        assert _score_macd(0.0, False) == 40.0


class TestPriceVsMAScoring:
    def test_none_returns_50(self):
        assert _score_price_vs_ma(None) == 50.0

    def test_at_ma(self):
        assert _score_price_vs_ma(0.0) == 50.0

    def test_far_above(self):
        assert _score_price_vs_ma(10.0) > 80.0
        assert _score_price_vs_ma(10.0) <= 100.0

    def test_far_below(self):
        assert _score_price_vs_ma(-15.0) < 30.0
        assert _score_price_vs_ma(-15.0) >= 0.0

    def test_above_5pct(self):
        score = _score_price_vs_ma(5.0)
        assert score >= 80.0

    def test_valid_range(self):
        for pct in [-30, -10, -5, 0, 5, 10, 25]:
            score = _score_price_vs_ma(pct)
            assert 0.0 <= score <= 100.0


class TestCompositeScore:
    def test_strong_bull_indicators(self):
        indicators = {
            "rsi_14": 25.0,       # oversold = bullish
            "macd_cross": 1.0,    # bullish cross
            "macd_above_zero": True,
            "price_vs_ma50": 6.0,
            "price_vs_ma200": 8.0,
            "momentum_20d": 5.0,
        }
        score = compute_technical_score(indicators, market_regime="neutral")
        assert score > 70.0

    def test_strong_bear_indicators(self):
        indicators = {
            "rsi_14": 80.0,       # overbought = bearish
            "macd_cross": -1.0,   # bearish cross
            "macd_above_zero": False,
            "price_vs_ma50": -10.0,
            "price_vs_ma200": -20.0,
            "momentum_20d": -10.0,
        }
        score = compute_technical_score(indicators, market_regime="neutral")
        assert score < 30.0

    def test_score_clipped_to_range(self):
        indicators = {"rsi_14": 50, "macd_cross": 0, "macd_above_zero": True}
        score = compute_technical_score(indicators)
        assert 0.0 <= score <= 100.0


class TestMomentumPercentiles:
    def test_highest_momentum_gets_high_percentile(self):
        data = {"A": 10.0, "B": 5.0, "C": -5.0}
        result = compute_momentum_percentiles(data)
        assert result["A"] > result["B"] > result["C"]

    def test_handles_none(self):
        data = {"A": 5.0, "B": None}
        result = compute_momentum_percentiles(data)
        assert "A" in result
        assert "B" in result
        assert result["B"] == 50.0


class TestAggregator:
    def test_score_formula(self):
        # With neutral modifier (1.0) the macro_shift is 0, so final == weighted_base
        result = compute_attractiveness_score(
            ticker="AAPL",
            technical_score=80.0,
            news_score=60.0,
            research_score=70.0,
            sector_modifiers={"Technology": 1.0},
            sector_map={"AAPL": "Technology"},
        )
        expected_base = 80.0 * 0.40 + 60.0 * 0.30 + 70.0 * 0.30  # 71.0
        assert abs(result["weighted_base"] - expected_base) < 0.01
        assert abs(result["macro_shift"]) < 0.01       # neutral = no shift
        assert abs(result["final_score"] - expected_base) < 0.01

    def test_macro_shift_additive(self):
        # Modifier 0.70 → shift = -10; modifier 1.30 → shift = +10
        result_headwind = compute_attractiveness_score(
            ticker="X", technical_score=70.0, news_score=70.0, research_score=70.0,
            sector_modifiers={"Technology": 0.70}, sector_map={"X": "Technology"},
        )
        result_tailwind = compute_attractiveness_score(
            ticker="X", technical_score=70.0, news_score=70.0, research_score=70.0,
            sector_modifiers={"Technology": 1.30}, sector_map={"X": "Technology"},
        )
        assert abs(result_headwind["macro_shift"] - (-10.0)) < 0.01
        assert abs(result_tailwind["macro_shift"] - 10.0) < 0.01
        # 20-point spread between extremes
        assert abs(result_tailwind["final_score"] - result_headwind["final_score"] - 20.0) < 0.1

    def test_macro_modifier_applied(self):
        result_no_boost = compute_attractiveness_score(
            ticker="AAPL",
            technical_score=60.0, news_score=60.0, research_score=60.0,
            sector_modifiers={"Technology": 1.0},
            sector_map={"AAPL": "Technology"},
        )
        result_boosted = compute_attractiveness_score(
            ticker="AAPL",
            technical_score=60.0, news_score=60.0, research_score=60.0,
            sector_modifiers={"Technology": 1.2},
            sector_map={"AAPL": "Technology"},
        )
        assert result_boosted["final_score"] > result_no_boost["final_score"]

    def test_score_clipped_at_100(self):
        result = compute_attractiveness_score(
            ticker="X", technical_score=100, news_score=100, research_score=100,
            sector_modifiers={"Technology": 1.3},
            sector_map={"X": "Technology"},
        )
        assert result["final_score"] <= 100.0

    def test_ratings(self):
        assert score_to_rating(75) == "BUY"
        assert score_to_rating(65) == "BUY"   # buy threshold is 65
        assert score_to_rating(64) == "HOLD"
        assert score_to_rating(55) == "HOLD"
        assert score_to_rating(40) == "HOLD"
        assert score_to_rating(39) == "SELL"
        assert score_to_rating(35) == "SELL"

    def test_consistency_flag_divergent_scores(self):
        # News bullish (80), research bearish (30) → flag inconsistency
        assert check_consistency(80, 30, 55) is True

    def test_consistency_flag_aligned_scores(self):
        # Both scores similar → no flag
        assert check_consistency(70, 65, 67) is False

    def test_consistency_flag_none_scores(self):
        # One score missing → no flag (can't detect inconsistency)
        assert check_consistency(None, 65, 65) is False
