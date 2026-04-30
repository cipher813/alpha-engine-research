"""
Round-trip + validation tests for ``graph.state_schemas``.

These models are not yet wired into ``ResearchState`` — this commit ships them
standalone so they can be reviewed + tested against fixtures of real agent
outputs before the integration commit lands. The ``extra="allow"`` posture
on every model means construction from a real agent dict (which carries
fields not enumerated here) does NOT reject; PR 2 flips ``extra="forbid"``.

Workstream: typed LangGraph state + Pydantic agent outputs + decision-artifact
capture (alpha-engine-research-typed-state-capture-260429.md).
"""

from __future__ import annotations

import pytest

from graph.state_schemas import (
    CIODecision,
    CIOOutput,
    ExitEvaluatorOutput,
    ExitEvent,
    InvestmentThesis,
    MacroEconomistOutput,
    PopulationRotationEvent,
    SectorRecommendation,
    SectorTeamOutput,
    ThesisUpdate,
    ToolCall,
)


# ── ToolCall ──────────────────────────────────────────────────────────────


class TestToolCall:
    def test_minimal_construction(self):
        tc = ToolCall(tool="quant_indicators")
        assert tc.tool == "quant_indicators"
        assert tc.ticker is None
        assert tc.args == {}
        assert tc.result_summary is None

    def test_full_construction(self):
        tc = ToolCall(
            tool="qual_news_search",
            ticker="AAPL",
            args={"hours": 48},
            result_summary="3 articles retrieved",
        )
        assert tc.ticker == "AAPL"
        assert tc.args == {"hours": 48}

    def test_extra_fields_allowed(self):
        tc = ToolCall(tool="x", undocumented_field="value")
        # extra fields preserved on dump
        assert tc.model_dump()["undocumented_field"] == "value"

    def test_tool_optional_for_peer_review_entries(self):
        # 2026-04-30: peer_review appends a phase-tracking entry to
        # tool_calls without a `tool` name (it's an orchestration step,
        # not a tool invocation). Schema must accept tool=None.
        tc = ToolCall(args={"phase": "peer_review"})
        assert tc.tool is None
        assert tc.args["phase"] == "peer_review"


# ── SectorRecommendation ──────────────────────────────────────────────────


class TestSectorRecommendation:
    def test_minimal_valid(self):
        r = SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=65.0)
        assert r.conviction == "medium"
        assert r.bull_case == ""

    def test_score_clamping(self):
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=120, qual_score=50)
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=50, qual_score=-1)

    def test_qual_score_optional(self):
        # 2026-04-30: peer_review can produce a recommendation when the
        # qual analyst returned 0 assessments (qual_score legitimately
        # absent). Schema must accept qual_score=None.
        r = SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=None)
        assert r.qual_score is None
        # And must still clamp when a value IS provided.
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=120)

    def test_conviction_literal(self):
        with pytest.raises(ValueError):
            SectorRecommendation(
                ticker="AAPL", quant_score=70, qual_score=70, conviction="extreme"
            )

    def test_round_trip(self):
        original = SectorRecommendation(
            ticker="NVDA",
            quant_score=85.0,
            qual_score=78.0,
            bull_case="AI tailwind",
            bear_case="Valuation",
            catalysts=["Earnings", "GTC"],
            conviction="high",
        )
        roundtripped = SectorRecommendation.model_validate(original.model_dump())
        assert roundtripped == original


# ── ThesisUpdate ──────────────────────────────────────────────────────────


class TestThesisUpdate:
    def test_all_scores_none_allowed(self):
        # Permitted by design — score_aggregator's recompute path or
        # hard-fail handles the missing-score case.
        t = ThesisUpdate(ticker="CME")
        assert t.final_score is None
        assert t.quant_score is None
        assert t.qual_score is None

    def test_partial_scores_round_trip(self):
        t = ThesisUpdate(ticker="HSY", quant_score=60.0, qual_score=55.0)
        d = t.model_dump()
        assert d["final_score"] is None
        assert d["quant_score"] == 60.0
        assert d["qual_score"] == 55.0

    def test_score_range_enforced(self):
        with pytest.raises(ValueError):
            ThesisUpdate(ticker="AAPL", final_score=110)


# ── SectorTeamOutput ──────────────────────────────────────────────────────


class TestSectorTeamOutput:
    def test_empty_team(self):
        sto = SectorTeamOutput(team_id="technology")
        assert sto.recommendations == []
        assert sto.thesis_updates == {}
        assert sto.tool_calls == []
        assert sto.error is None

    def test_with_recommendations_and_updates(self):
        sto = SectorTeamOutput(
            team_id="financials",
            recommendations=[
                SectorRecommendation(ticker="JPM", quant_score=70, qual_score=65),
                SectorRecommendation(ticker="V", quant_score=72, qual_score=70),
            ],
            thesis_updates={
                "MA": ThesisUpdate(
                    ticker="MA", final_score=68.0, quant_score=70.0, qual_score=66.0
                )
            },
            tool_calls=[ToolCall(tool="quant_indicators", ticker="JPM")],
        )
        assert len(sto.recommendations) == 2
        assert "MA" in sto.thesis_updates

    def test_extra_stub_fields_preserved(self):
        # The offline stub returns quant_output / qual_output / peer_review_output
        # as extra fields. They must round-trip through the model without rejection.
        sto = SectorTeamOutput(
            team_id="technology",
            recommendations=[],
            thesis_updates={},
            tool_calls=[],
            quant_output={"ranked_picks": []},
            qual_output={"assessments": []},
            peer_review_output={"recommendations": []},
        )
        d = sto.model_dump()
        assert "quant_output" in d
        assert "qual_output" in d
        assert "peer_review_output" in d

    def test_construction_from_dict_payload(self):
        # Mirrors the shape the offline stub at local/offline_stubs.py:297-305
        # actually returns. Must validate cleanly.
        payload = {
            "team_id": "technology",
            "recommendations": [
                {"ticker": "AAPL", "quant_score": 65.0, "qual_score": 70.0,
                 "bull_case": "x", "bear_case": "y", "catalysts": [],
                 "conviction": "medium", "team_id": "technology"},
            ],
            "thesis_updates": {},
            "quant_output": {},
            "qual_output": {},
            "peer_review_output": {},
            "tool_calls": [],
        }
        sto = SectorTeamOutput.model_validate(payload)
        assert sto.recommendations[0].ticker == "AAPL"


# ── MacroEconomistOutput ──────────────────────────────────────────────────


class TestMacroEconomistOutput:
    def test_minimal(self):
        m = MacroEconomistOutput()
        assert m.macro_report == ""
        assert m.market_regime == "neutral"
        assert m.sector_modifiers == {}

    def test_modifier_clamp_in_range(self):
        m = MacroEconomistOutput(
            sector_modifiers={"Technology": 1.20, "Energy": 0.85, "Healthcare": 1.0}
        )
        assert m.sector_modifiers["Technology"] == 1.20

    def test_modifier_clamp_above_range(self):
        with pytest.raises(ValueError, match=r"sector_modifiers"):
            MacroEconomistOutput(sector_modifiers={"Technology": 1.50})

    def test_modifier_clamp_below_range(self):
        with pytest.raises(ValueError, match=r"sector_modifiers"):
            MacroEconomistOutput(sector_modifiers={"Energy": 0.50})

    def test_modifier_boundaries_inclusive(self):
        # 0.70 and 1.30 are explicitly the inclusive boundaries
        m = MacroEconomistOutput(
            sector_modifiers={"Low": 0.70, "High": 1.30, "Mid": 1.0}
        )
        assert m.sector_modifiers["Low"] == 0.70
        assert m.sector_modifiers["High"] == 1.30

    def test_regime_literal_enforced(self):
        with pytest.raises(ValueError):
            MacroEconomistOutput(market_regime="euphoric")

    def test_regime_all_valid_values(self):
        for regime in ("bull", "neutral", "bear", "caution"):
            m = MacroEconomistOutput(market_regime=regime)
            assert m.market_regime == regime


# ── ExitEvent + ExitEvaluatorOutput ───────────────────────────────────────


class TestExitEvent:
    def test_minimal(self):
        e = ExitEvent(ticker_out="MA")
        assert e.reason == ""
        assert e.score_out == 0.0

    def test_with_reason_and_score(self):
        e = ExitEvent(ticker_out="MA", reason="min_rotation_floor", score_out=70.0)
        assert e.score_out == 70.0


class TestExitEvaluatorOutput:
    def test_minimal(self):
        eo = ExitEvaluatorOutput()
        assert eo.exits == []
        assert eo.open_slots == 0

    def test_open_slots_non_negative(self):
        with pytest.raises(ValueError):
            ExitEvaluatorOutput(open_slots=-1)


# ── CIODecision + CIOOutput ───────────────────────────────────────────────


class TestCIODecision:
    def test_minimal(self):
        d = CIODecision(ticker="JPM")
        assert d.thesis_type is None

    def test_thesis_type_literal(self):
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", thesis_type="MAYBE")

    def test_full(self):
        d = CIODecision(
            ticker="JPM",
            thesis_type="ADVANCE",
            rationale="Strong",
            conviction=78,
            score=78.0,
        )
        assert d.thesis_type == "ADVANCE"
        assert d.conviction == 78

    def test_conviction_int_range_enforced(self):
        # Path Y: conviction is a 0-100 score; bounds enforced.
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", conviction=120)
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", conviction=-1)

    def test_conviction_optional(self):
        d = CIODecision(ticker="JPM")
        assert d.conviction is None


class TestCIOOutput:
    def test_minimal(self):
        o = CIOOutput()
        assert o.ic_decisions == []
        assert o.advanced_tickers == []
        assert o.entry_theses == {}


# ── InvestmentThesis ──────────────────────────────────────────────────────


class TestInvestmentThesis:
    def test_minimal_required(self):
        t = InvestmentThesis(ticker="AAPL", final_score=70.0, rating="BUY")
        assert t.sector == "Unknown"
        assert t.team_id == ""
        # Storage format from normalize_conviction (executor-compatible)
        assert t.conviction == "stable"

    def test_full(self):
        t = InvestmentThesis(
            ticker="NVDA",
            sector="Technology",
            team_id="technology",
            final_score=82.0,
            quant_score=80.0,
            qual_score=85.0,
            weighted_base=82.5,
            macro_shift=-0.5,
            bull_case="AI",
            bear_case="Valuation",
            catalysts=["Earnings"],
            conviction="rising",  # storage format
            quant_rationale="...",
            rating="BUY",
            score_failed=False,
        )
        assert t.weighted_base == 82.5

    def test_agent_format_conviction_rejected(self):
        # InvestmentThesis is post-normalize_conviction storage; agent format
        # must NOT be accepted at this boundary (use ThesisUpdate for the
        # union-format variant if needed).
        with pytest.raises(ValueError):
            InvestmentThesis(
                ticker="AAPL", final_score=70.0, rating="BUY",
                conviction="medium",
            )

    def test_rating_literal(self):
        with pytest.raises(ValueError):
            InvestmentThesis(ticker="AAPL", final_score=70.0, rating="STRONG_BUY")

    def test_round_trip_with_extras(self):
        # Real score_aggregator output may carry a "date" or "team_id"
        # field added by archive_writer's row construction; extras allowed.
        payload = {
            "ticker": "JPM",
            "sector": "Financials",
            "team_id": "financials",
            "final_score": 68.0,
            "quant_score": 65.0,
            "qual_score": 70.0,
            "weighted_base": 67.5,
            "macro_shift": 0.5,
            "bull_case": "...",
            "bear_case": "...",
            "catalysts": [],
            "conviction": "stable",  # storage format from normalize_conviction
            "quant_rationale": "",
            "rating": "BUY",
            "score_failed": False,
            "date": "2026-04-25",  # extra
        }
        t = InvestmentThesis.model_validate(payload)
        assert t.model_dump()["date"] == "2026-04-25"


# ── PopulationRotationEvent ───────────────────────────────────────────────


class TestPopulationRotationEvent:
    def test_minimal(self):
        e = PopulationRotationEvent()
        assert e.event_type is None
        assert e.reason == ""

    def test_entry_event_shape(self):
        e = PopulationRotationEvent(event_type="entry", ticker_in="JPM", reason="advance")
        assert e.event_type == "entry"
        assert e.ticker_in == "JPM"

    def test_exit_event_shape(self):
        e = PopulationRotationEvent(event_type="exit", ticker_out="MA", reason="floor")
        assert e.event_type == "exit"
        assert e.ticker_out == "MA"

    def test_event_type_literal(self):
        with pytest.raises(ValueError):
            PopulationRotationEvent(event_type="rotation")
