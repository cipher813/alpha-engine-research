"""Tests for the technical scoring engine and aggregator."""

import pytest

_technical = pytest.importorskip("scoring.technical", reason="scoring.technical is gitignored")
_score_rsi = _technical._score_rsi
_score_macd = _technical._score_macd
_score_price_vs_ma = _technical._score_price_vs_ma
_score_momentum = _technical._score_momentum
compute_technical_score = _technical.compute_technical_score
compute_momentum_percentiles = _technical.compute_momentum_percentiles

from scoring.composite import (
    compute_composite_score,
    score_to_rating,
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


class TestCompositeScoring:
    def test_score_formula(self):
        # With neutral modifier (1.0) the macro_shift is 0, so final == weighted_base
        result = compute_composite_score(
            quant_score=80.0, qual_score=60.0, sector_modifier=1.0,
        )
        expected_base = 80.0 * 0.50 + 60.0 * 0.50  # 70.0
        assert abs(result["weighted_base"] - expected_base) < 0.1
        assert abs(result["macro_shift"]) < 0.01
        assert abs(result["final_score"] - expected_base) < 0.1

    def test_macro_shift_additive(self):
        result_headwind = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=0.70,
        )
        result_tailwind = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.30,
        )
        assert abs(result_headwind["macro_shift"] - (-10.0)) < 0.1
        assert abs(result_tailwind["macro_shift"] - 10.0) < 0.1
        assert abs(result_tailwind["final_score"] - result_headwind["final_score"] - 20.0) < 0.2

    def test_macro_modifier_applied(self):
        result_neutral = compute_composite_score(
            quant_score=60.0, qual_score=60.0, sector_modifier=1.0,
        )
        result_boosted = compute_composite_score(
            quant_score=60.0, qual_score=60.0, sector_modifier=1.2,
        )
        assert result_boosted["final_score"] > result_neutral["final_score"]

    def test_score_clipped_at_100(self):
        result = compute_composite_score(
            quant_score=100, qual_score=100, sector_modifier=1.3,
        )
        assert result["final_score"] <= 100.0

    def test_score_clipped_at_0(self):
        result = compute_composite_score(
            quant_score=5, qual_score=5, sector_modifier=0.70,
        )
        assert result["final_score"] >= 0.0

    def test_missing_quant_uses_qual(self):
        result = compute_composite_score(
            quant_score=None, qual_score=70.0, sector_modifier=1.0,
        )
        assert result["final_score"] == 70.0
        assert result["score_failed"] is False

    def test_both_missing_returns_failed(self):
        result = compute_composite_score(
            quant_score=None, qual_score=None, sector_modifier=1.0,
        )
        assert result["score_failed"] is True

    def test_boosts_capped(self):
        result = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.0,
            boosts={"pead": 5, "revision": 3, "options": 4, "insider": 5},
        )
        assert result["total_boost"] == 10.0  # capped at 10

    def test_ratings(self):
        assert score_to_rating(75) == "BUY"
        assert score_to_rating(69) == "HOLD"
        assert score_to_rating(70) == "BUY"   # default buy threshold is 70
        assert score_to_rating(55) == "HOLD"
        assert score_to_rating(41) == "HOLD"
        assert score_to_rating(40) == "SELL"  # sell threshold is <= 40
        assert score_to_rating(35) == "SELL"
