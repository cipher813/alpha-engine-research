"""
Tests for thesis/updater.py.
All functions are pure (no I/O, no LLM) — straightforward unit tests.
"""

import pytest
from thesis.updater import build_thesis_record, check_rating_change, _build_summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_aggregated(
    rating="BUY",
    final_score=75.0,
    technical_score=70.0,
    news_score=68.0,
    research_score=72.0,
    macro_modifier=1.05,
    sector="Technology",
    conviction="rising",
    signal="ENTER",
    score_delta=3.2,
    score_velocity_5d=1.1,
    price_target_upside=0.18,
    prior_score=71.8,
    prior_rating="HOLD",
    stale_days=0,
    consistency_flag=0,
    material_changes=True,
    last_material_change_date="2026-03-05",
) -> dict:
    return {
        "rating": rating,
        "final_score": final_score,
        "technical_score": technical_score,
        "news_score": news_score,
        "research_score": research_score,
        "macro_modifier": macro_modifier,
        "sector": sector,
        "conviction": conviction,
        "signal": signal,
        "score_delta": score_delta,
        "score_velocity_5d": score_velocity_5d,
        "price_target_upside": price_target_upside,
        "prior_score": prior_score,
        "prior_rating": prior_rating,
        "stale_days": stale_days,
        "consistency_flag": consistency_flag,
        "material_changes": material_changes,
        "last_material_change_date": last_material_change_date,
    }


# ── build_thesis_record ───────────────────────────────────────────────────────

class TestBuildThesisRecord:
    def test_required_fields_present(self):
        agg = _make_aggregated()
        result = build_thesis_record(
            ticker="PLTR",
            run_date="2026-03-05",
            aggregated=agg,
            agent_outputs={},
        )
        assert result["ticker"] == "PLTR"
        assert result["date"] == "2026-03-05"
        assert result["rating"] == "BUY"
        assert result["final_score"] == 75.0

    def test_executor_fields_populated(self):
        agg = _make_aggregated(conviction="rising", signal="ENTER", score_velocity_5d=1.5)
        result = build_thesis_record("PLTR", "2026-03-05", agg, {})
        assert result["conviction"] == "rising"
        assert result["signal"] == "ENTER"
        assert result["score_velocity_5d"] == 1.5

    def test_agent_outputs_surfaced(self):
        agg = _make_aggregated()
        agent_outputs = {
            "news_json": {"key_catalyst": "AI contract win", "sentiment": "positive"},
            "research_json": {"key_risk": "valuation stretched", "consensus_direction": "bullish"},
        }
        result = build_thesis_record("PLTR", "2026-03-05", agg, agent_outputs)
        assert result["key_catalyst"] == "AI contract win"
        assert result["key_risk"] == "valuation stretched"
        assert result["news_sentiment"] == "positive"
        assert result["consensus_direction"] == "bullish"

    def test_empty_agent_outputs_no_error(self):
        agg = _make_aggregated()
        result = build_thesis_record("PLTR", "2026-03-05", agg, {})
        assert result["key_catalyst"] is None
        assert result["key_risk"] is None

    def test_research_json_upside_used_as_catalyst_fallback(self):
        agg = _make_aggregated()
        agent_outputs = {
            "news_json": {},
            "research_json": {"key_upside": "25% analyst target upside"},
        }
        result = build_thesis_record("PLTR", "2026-03-05", agg, agent_outputs)
        assert result["key_catalyst"] == "25% analyst target upside"

    def test_updated_at_is_set(self):
        agg = _make_aggregated()
        result = build_thesis_record("PLTR", "2026-03-05", agg, {})
        assert "updated_at" in result
        assert result["updated_at"]  # non-empty

    def test_sector_preserved(self):
        agg = _make_aggregated(sector="Healthcare")
        result = build_thesis_record("LLY", "2026-03-05", agg, {})
        assert result["sector"] == "Healthcare"


# ── _build_summary ────────────────────────────────────────────────────────────

class TestBuildSummary:
    def test_basic_buy_summary(self):
        agg = _make_aggregated(rating="BUY", final_score=75.0)
        result = _build_summary("PLTR", agg, {}, {})
        assert "PLTR" in result
        assert "BUY" in result
        assert "75" in result

    def test_bullish_consensus_included(self):
        agg = _make_aggregated()
        result = _build_summary("PLTR", agg, {}, {"consensus_direction": "bullish"})
        assert "bullish" in result.lower()

    def test_bearish_consensus_included(self):
        agg = _make_aggregated()
        result = _build_summary("PLTR", agg, {}, {"consensus_direction": "bearish"})
        assert "bearish" in result.lower()

    def test_catalyst_included(self):
        agg = _make_aggregated()
        result = _build_summary("PLTR", agg, {"key_catalyst": "AI contract"}, {})
        assert "AI contract" in result

    def test_risk_included_for_hold(self):
        agg = _make_aggregated(rating="HOLD", final_score=55.0)
        result = _build_summary("PLTR", agg, {}, {"key_risk": "macro headwind"})
        assert "macro headwind" in result

    def test_risk_excluded_for_sell(self):
        agg = _make_aggregated(rating="SELL", final_score=30.0)
        result = _build_summary("PLTR", agg, {}, {"key_risk": "competition risk"})
        assert "competition risk" not in result

    def test_no_catalyst_no_error(self):
        agg = _make_aggregated()
        result = _build_summary("PLTR", agg, {}, {})
        assert "PLTR" in result

    def test_neutral_consensus_not_mentioned(self):
        agg = _make_aggregated()
        result = _build_summary("PLTR", agg, {}, {"consensus_direction": "neutral"})
        assert "Analyst consensus" not in result


# ── check_rating_change ───────────────────────────────────────────────────────

class TestCheckRatingChange:
    def test_hold_to_buy(self):
        thesis = {"rating": "BUY"}
        prior = {"rating": "HOLD"}
        result = check_rating_change(thesis, prior)
        assert result == "HOLD → BUY"

    def test_buy_to_sell(self):
        thesis = {"rating": "SELL"}
        prior = {"rating": "BUY"}
        result = check_rating_change(thesis, prior)
        assert result == "BUY → SELL"

    def test_no_change_returns_none(self):
        thesis = {"rating": "BUY"}
        prior = {"rating": "BUY"}
        result = check_rating_change(thesis, prior)
        assert result is None

    def test_no_prior_returns_none(self):
        thesis = {"rating": "BUY"}
        result = check_rating_change(thesis, None)
        assert result is None

    def test_prior_uses_prev_rating_fallback(self):
        thesis = {"rating": "BUY"}
        prior = {"prev_rating": "SELL"}  # no "rating" key
        result = check_rating_change(thesis, prior)
        assert result == "SELL → BUY"
