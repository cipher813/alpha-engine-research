"""Tests for scoring/aggregator.py — pure logic functions.

Covers: compute_attractiveness_score, score_to_rating, check_consistency,
compute_staleness, compute_conviction, compute_score_velocity_5d,
compute_signal, compute_long_term_score, _compute_pead_boost,
_compute_revision_boost, _trading_days_between, _get_weights.
"""

from unittest.mock import patch, MagicMock

import pytest

from scoring.aggregator import (
    _trading_days_between,
    check_consistency,
    compute_attractiveness_score,
    compute_conviction,
    compute_long_term_score,
    compute_score_velocity_5d,
    compute_signal,
    compute_staleness,
    score_to_rating,
    _compute_pead_boost,
    _compute_revision_boost,
)


# ---------------------------------------------------------------------------
# score_to_rating
# ---------------------------------------------------------------------------


class TestScoreToRating:
    def test_buy(self):
        assert score_to_rating(80) == "BUY"

    def test_hold(self):
        assert score_to_rating(50) == "HOLD"

    def test_sell(self):
        assert score_to_rating(30) == "SELL"

    def test_boundary_buy(self):
        assert score_to_rating(70) == "BUY"

    def test_boundary_sell(self):
        assert score_to_rating(40) == "HOLD"


# ---------------------------------------------------------------------------
# compute_attractiveness_score
# ---------------------------------------------------------------------------


class TestComputeAttractivenessScore:
    def setup_method(self):
        # Reset weight cache so each test gets fresh defaults
        import scoring.aggregator as agg
        agg._weights_cache = None

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_neutral_macro(self, _mock):
        result = compute_attractiveness_score(
            "AAPL", quant_score=80.0, qual_score=70.0,
            sector_modifiers={"Technology": 1.0},
            sector_map={"AAPL": "Technology"},
        )
        assert result["macro_shift"] == 0.0
        assert result["ticker"] == "AAPL"
        assert result["rating"] in ("BUY", "HOLD", "SELL")

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_bullish_macro(self, _mock):
        result = compute_attractiveness_score(
            "AAPL", quant_score=65.0, qual_score=65.0,
            sector_modifiers={"Technology": 1.30},
            sector_map={"AAPL": "Technology"},
        )
        assert result["macro_shift"] == pytest.approx(10.0)

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_bearish_macro(self, _mock):
        result = compute_attractiveness_score(
            "AAPL", quant_score=65.0, qual_score=65.0,
            sector_modifiers={"Technology": 0.70},
            sector_map={"AAPL": "Technology"},
        )
        assert result["macro_shift"] == pytest.approx(-10.0)

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_clamped_to_100(self, _mock):
        result = compute_attractiveness_score(
            "AAPL", quant_score=100.0, qual_score=100.0,
            sector_modifiers={"Technology": 1.30},
            sector_map={"AAPL": "Technology"},
        )
        assert result["final_score"] <= 100.0

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_clamped_to_0(self, _mock):
        result = compute_attractiveness_score(
            "AAPL", quant_score=0.0, qual_score=0.0,
            sector_modifiers={"Technology": 0.70},
            sector_map={"AAPL": "Technology"},
        )
        assert result["final_score"] >= 0.0

    @patch("scoring.aggregator._load_weights_from_s3", return_value=None)
    def test_missing_sector_defaults(self, _mock):
        result = compute_attractiveness_score(
            "UNKNOWN", quant_score=70.0, qual_score=70.0,
            sector_modifiers={},
        )
        assert result["macro_modifier"] == 1.0

    @patch("scoring.aggregator._load_weights_from_s3", return_value={"quant": 0.60, "qual": 0.40})
    def test_s3_weights(self, _mock):
        import scoring.aggregator as agg
        agg._weights_cache = None
        result = compute_attractiveness_score(
            "AAPL", quant_score=100.0, qual_score=0.0,
            sector_modifiers={"Technology": 1.0},
            sector_map={"AAPL": "Technology"},
        )
        assert result["weighted_base"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------


class TestCheckConsistency:
    def test_consistent(self):
        assert check_consistency(75.0, 72.0, 73.5) is False

    def test_inconsistent(self):
        assert check_consistency(85.0, 30.0, 57.5) is True

    def test_none_quant(self):
        assert check_consistency(None, 70.0, 70.0) is False

    def test_none_qual(self):
        assert check_consistency(70.0, None, 70.0) is False

    def test_large_divergence_but_same_direction(self):
        # Both above BUY threshold — not inconsistent even with 35pt gap
        assert check_consistency(95.0, 60.0, 77.5) is False


# ---------------------------------------------------------------------------
# compute_staleness
# ---------------------------------------------------------------------------


class TestComputeStaleness:
    def test_stale(self):
        assert compute_staleness("2026-01-01", 35) is True

    def test_fresh(self):
        assert compute_staleness("2026-04-01", 3) is False

    def test_none_date(self):
        assert compute_staleness(None, 100) is False


# ---------------------------------------------------------------------------
# compute_conviction
# ---------------------------------------------------------------------------


class TestComputeConviction:
    def test_rising(self):
        assert compute_conviction([80, 75, 70]) == "rising"

    def test_declining(self):
        assert compute_conviction([60, 70, 80]) == "declining"

    def test_stable(self):
        assert compute_conviction([75, 80, 75]) == "stable"

    def test_single_score(self):
        assert compute_conviction([75]) == "stable"

    def test_two_scores_rising(self):
        assert compute_conviction([80, 70]) == "rising"

    def test_two_scores_declining(self):
        assert compute_conviction([60, 70]) == "declining"


# ---------------------------------------------------------------------------
# compute_score_velocity_5d
# ---------------------------------------------------------------------------


class TestComputeScoreVelocity:
    def test_positive_velocity(self):
        v = compute_score_velocity_5d([80, 78, 76, 74, 72])
        assert v is not None
        assert v > 0

    def test_negative_velocity(self):
        v = compute_score_velocity_5d([60, 65, 70, 75, 80])
        assert v is not None
        assert v < 0

    def test_single_score(self):
        assert compute_score_velocity_5d([75]) is None

    def test_two_scores(self):
        v = compute_score_velocity_5d([80, 70])
        assert v == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_signal
# ---------------------------------------------------------------------------


class TestComputeSignal:
    def test_buy_enters(self):
        assert compute_signal("BUY", "HOLD", "rising", False) == "ENTER"

    def test_sell_exits(self):
        assert compute_signal("SELL", "BUY", "declining", False) == "EXIT"

    def test_reduce_on_declining(self):
        assert compute_signal("HOLD", "BUY", "declining", False) == "REDUCE"

    def test_reduce_on_material_change(self):
        assert compute_signal("HOLD", "BUY", "stable", True) == "REDUCE"

    def test_hold_default(self):
        assert compute_signal("HOLD", "HOLD", "stable", False) == "HOLD"

    def test_hold_from_none(self):
        assert compute_signal("HOLD", None, "declining", False) == "HOLD"


# ---------------------------------------------------------------------------
# compute_long_term_score
# ---------------------------------------------------------------------------


class TestComputeLongTermScore:
    def test_neutral(self):
        score, rating = compute_long_term_score(80.0, 70.0, {"Tech": 1.0}, "Tech")
        assert score == 75.0
        assert rating == "BUY"

    def test_bullish_macro(self):
        score, rating = compute_long_term_score(60.0, 60.0, {"Tech": 1.30}, "Tech")
        assert score == 70.0  # 60 + 10

    def test_clamped(self):
        score, _ = compute_long_term_score(100.0, 100.0, {"Tech": 1.30}, "Tech")
        assert score <= 100.0


# ---------------------------------------------------------------------------
# _trading_days_between
# ---------------------------------------------------------------------------


class TestTradingDaysBetween:
    def test_same_day(self):
        assert _trading_days_between("2026-04-08", "2026-04-08") == 0

    def test_one_week(self):
        # Mon to Fri = 4 trading days (Tue-Fri)
        result = _trading_days_between("2026-04-06", "2026-04-10")
        assert result == 4

    def test_none_start(self):
        assert _trading_days_between(None, "2026-04-08") == 0

    def test_start_after_end(self):
        assert _trading_days_between("2026-04-10", "2026-04-08") == 0


# ---------------------------------------------------------------------------
# _compute_pead_boost
# ---------------------------------------------------------------------------


class TestPeadBoost:
    def test_no_earnings_data(self):
        assert _compute_pead_boost("AAPL", {}, "2026-04-08") == 0.0

    def test_no_ticker_data(self):
        assert _compute_pead_boost("AAPL", {"MSFT": {}}, "2026-04-08") == 0.0

    def test_old_earnings(self):
        data = {"AAPL": {"earnings_surprises": [{"date": "2025-01-01", "surprise_pct": 10}]}}
        assert _compute_pead_boost("AAPL", data, "2026-04-08") == 0.0

    def test_recent_strong_surprise(self):
        data = {"AAPL": {"earnings_surprises": [{"date": "2026-04-05", "surprise_pct": 15}]}}
        boost = _compute_pead_boost("AAPL", data, "2026-04-08")
        assert boost > 0

    def test_recent_negative_surprise(self):
        data = {"AAPL": {"earnings_surprises": [{"date": "2026-04-05", "surprise_pct": -15}]}}
        boost = _compute_pead_boost("AAPL", data, "2026-04-08")
        assert boost < 0

    def test_invalid_date(self):
        data = {"AAPL": {"earnings_surprises": [{"date": "invalid", "surprise_pct": 10}]}}
        assert _compute_pead_boost("AAPL", data, "2026-04-08") == 0.0


# ---------------------------------------------------------------------------
# _compute_revision_boost
# ---------------------------------------------------------------------------


class TestRevisionBoost:
    def test_no_data(self):
        assert _compute_revision_boost("AAPL", {}) == 0.0

    def test_no_ticker(self):
        assert _compute_revision_boost("AAPL", {"MSFT": {"revision_streak": 3}}) == 0.0
