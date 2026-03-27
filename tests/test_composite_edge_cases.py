"""Edge case tests for composite scoring."""

import pytest
from scoring.composite import compute_composite_score, score_to_rating, normalize_conviction


class TestCompositeEdgeCases:
    def test_extreme_bullish_modifier(self):
        result = compute_composite_score(quant_score=50, qual_score=50, sector_modifier=1.5)
        assert result["macro_shift"] > 10  # beyond normal range
        assert result["final_score"] <= 100

    def test_extreme_bearish_modifier(self):
        result = compute_composite_score(quant_score=50, qual_score=50, sector_modifier=0.5)
        assert result["macro_shift"] < -10
        assert result["final_score"] >= 0

    def test_all_negative_boosts(self):
        result = compute_composite_score(
            quant_score=70, qual_score=70, sector_modifier=1.0,
            boosts={"pead": -5, "revision": -3, "options": -4},
        )
        assert result["total_boost"] == -10.0  # capped at -10
        assert result["final_score"] == 60.0

    def test_single_score_with_boosts(self):
        result = compute_composite_score(
            quant_score=60, qual_score=None, sector_modifier=1.0,
            boosts={"pead": 5},
        )
        assert result["final_score"] == 65.0
        assert result["score_failed"] is False

    def test_zero_scores(self):
        result = compute_composite_score(quant_score=0, qual_score=0, sector_modifier=1.0)
        assert result["final_score"] == 0.0

    def test_perfect_scores(self):
        result = compute_composite_score(quant_score=100, qual_score=100, sector_modifier=1.0)
        assert result["final_score"] == 100.0


class TestNormalizeConviction:
    def test_valid_passthrough(self):
        assert normalize_conviction("rising") == "rising"
        assert normalize_conviction("stable") == "stable"
        assert normalize_conviction("declining") == "declining"

    def test_qual_analyst_mapping(self):
        assert normalize_conviction("high") == "rising"
        assert normalize_conviction("medium") == "stable"
        assert normalize_conviction("low") == "declining"

    def test_numeric_mapping(self):
        assert normalize_conviction(85) == "rising"
        assert normalize_conviction(55) == "stable"
        assert normalize_conviction(20) == "declining"

    def test_none_defaults_stable(self):
        assert normalize_conviction(None) == "stable"

    def test_sentence_defaults_stable(self):
        assert normalize_conviction("I am quite confident in this pick") == "stable"

    def test_case_insensitive(self):
        assert normalize_conviction("RISING") == "rising"
        assert normalize_conviction("Declining") == "declining"


class TestScoreToRating:
    def test_none_returns_hold(self):
        assert score_to_rating(None) == "HOLD"

    def test_custom_thresholds(self):
        assert score_to_rating(65, buy_threshold=60) == "BUY"
        assert score_to_rating(65, buy_threshold=70) == "HOLD"

    def test_boundary_values(self):
        assert score_to_rating(70.0) == "BUY"
        assert score_to_rating(69.9) == "HOLD"
        assert score_to_rating(40.0) == "SELL"
        assert score_to_rating(40.1) == "HOLD"
