"""Tests for score_aggregator hard-fail behavior on sector team errors.

Regression: prior to PR #25 an exception in a team's ReAct agent was
swallowed and the team returned an empty recommendations list identical
to the legitimate "LLM produced no picks" path. Aggregator must now raise
when any team carries a non-None `error` marker.
"""

from __future__ import annotations

import pytest

from graph.research_graph import score_aggregator


def _state(team_outputs: dict) -> dict:
    return {
        "sector_team_outputs": team_outputs,
        "sector_modifiers": {},
        "sector_map": {},
    }


class TestScoreAggregatorHardFail:
    def test_raises_when_any_team_has_error(self):
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": "RecursionError: exceeded recursion_limit",
            },
        })
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_raises_with_all_failed_teams_listed(self):
        state = _state({
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": "APIError: 529"},
            "defensives": {"recommendations": [], "thesis_updates": {},
                           "error": "JSONDecodeError: malformed"},
        })
        with pytest.raises(RuntimeError) as exc:
            score_aggregator(state)
        msg = str(exc.value)
        assert "healthcare" in msg
        assert "defensives" in msg

    def test_passes_through_when_no_errors(self):
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
            },
        })
        # Should not raise — error=None means the team legitimately had no picks.
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}

    def test_passes_through_when_error_key_absent(self):
        # Backward compat: team_outputs written before this change lack
        # the `error` key entirely — aggregator should not raise.
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {}},
        })
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}


class TestScoreAggregatorPartialTolerance:
    """2026-05-02: sector teams that hit recursion_limit must NOT crash the
    SF. They return ``{partial: True, error: None}``; aggregator logs WARN
    and proceeds. Distinct from real errors which still hard-fail.
    """

    def test_partial_team_does_not_raise(self, caplog):
        import logging
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
            "healthcare": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
            },
        })
        with caplog.at_level(logging.WARNING, logger="research"):
            result = score_aggregator(state)
        assert result == {"investment_theses": {}}
        assert any(
            "partial" in r.message and "technology" in r.message
            for r in caplog.records
        ), f"Expected WARN naming the partial team; got: {[r.message for r in caplog.records]}"

    def test_partial_team_with_error_still_raises(self):
        """Belt-and-suspenders: if a team somehow has BOTH partial=True
        AND a real error, the error path still wins (hard-fail). The
        partial flag is for budget-exhausted runs that produced no
        downstream-incompatible state, not for masking real failures.
        """
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": "APIError: 529",
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
        })
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_all_teams_partial_raises(self):
        """If every team is partial, the CIO has nothing to rank — that's
        a system-wide failure even though no single team errored. Hard-fail
        so operators investigate the systemic cause."""
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
        })
        with pytest.raises(RuntimeError, match="all .* sector teams returned partial"):
            score_aggregator(state)

    def test_mixed_partial_and_full_teams_advances(self):
        """Most realistic scenario: 1-2 teams partial, rest fine. SF
        advances; CIO ranks what it has."""
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["qual:recursion_limit_exhausted"]},
            "consumer": {"recommendations": [], "thesis_updates": {},
                         "error": None, "partial": False},
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": False},
        })
        # Should not raise.
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}
