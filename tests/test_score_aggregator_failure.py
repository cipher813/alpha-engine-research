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
