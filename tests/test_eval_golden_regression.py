"""CI-gated golden-trace eval regression suite (L4579).

The prompt-layer twin of test_schema_contract: a deterministic gate
(no live LLM calls) that fails the build when a rubric prompt, the
RubricEvalLLMOutput schema, or the graph trajectory contract changes
without the goldens being consciously re-blessed via
``scripts/regen_golden_traces.py``.
"""

from __future__ import annotations

import pytest

from evals import trajectory
from evals.golden_trace import (
    current_pin,
    load_eval_pipeline_golden,
    load_graph_topology_golden,
    parse_recorded_response,
    render_rubric_for,
)

_GOLDEN = load_eval_pipeline_golden()
_RUBRICS = _GOLDEN["rubrics"]
_RUBRIC_IDS = [r["rubric_id"] for r in _RUBRICS]


def _pin(rubric_id: str) -> dict:
    return next(r for r in _RUBRICS if r["rubric_id"] == rubric_id)


# ── Render contract ────────────────────────────────────────────────────────


class TestRubricRenderContract:
    @pytest.mark.parametrize("rubric_id", _RUBRIC_IDS)
    def test_rubric_renders_and_interpolates_input(self, rubric_id):
        pin = _pin(rubric_id)
        rendered = render_rubric_for(_GOLDEN, rubric_id, pin["agent_id"])
        assert isinstance(rendered, str) and rendered
        # Proves agent_input was interpolated (a template that dropped the
        # {agent_input} placeholder would not contain the marker).
        assert _GOLDEN["marker"] in rendered, (
            f"{rubric_id}: rendered prompt does not contain the golden "
            f"input marker — the {{agent_input}} placeholder may have been "
            f"removed. Re-bless via scripts/regen_golden_traces.py if "
            f"intentional."
        )


# ── Prompt drift lock (forces conscious re-bless) ──────────────────────────


class TestPromptDriftLock:
    @pytest.mark.parametrize("rubric_id", _RUBRIC_IDS)
    def test_version_and_hash_match_golden(self, rubric_id):
        pin = _pin(rubric_id)
        cur = current_pin(rubric_id, pin["agent_id"])
        assert cur.version == pin["version"], (
            f"{rubric_id}: prompt version drifted "
            f"{pin['version']} -> {cur.version}. A version bump means the "
            f"rubric changed — regenerate goldens (scripts/"
            f"regen_golden_traces.py) and review the eval-quality delta."
        )
        assert cur.prompt_hash == pin["prompt_hash"], (
            f"{rubric_id}: prompt content hash drifted (the rubric text "
            f"changed even if the version did not). Regenerate goldens "
            f"and re-bless consciously, or bump the rubric version."
        )


# ── Parse → score pipeline ─────────────────────────────────────────────────


class TestParsePipeline:
    def test_recorded_response_parses_to_expected_scores(self):
        case = _GOLDEN["parse_case"]
        out = parse_recorded_response(case["recorded_response"])

        actual = [
            {"dimension": d.dimension, "score": d.score}
            for d in out.dimension_scores
        ]
        assert actual == case["expected"]["dimension_scores"], (
            "Parsed dimension scores diverged from the golden — a "
            "RubricEvalLLMOutput schema change or a parser regression. "
            "Re-bless via scripts/regen_golden_traces.py if intentional."
        )
        assert out.overall_reasoning == case["expected"]["overall_reasoning"]


# ── Graph trajectory contract ──────────────────────────────────────────────


class TestGraphTopologyContract:
    def test_required_nodes_match_golden(self):
        golden = load_graph_topology_golden()
        assert list(trajectory.REQUIRED_NODES) == golden["required_nodes"], (
            "evals/trajectory.py REQUIRED_NODES changed — a graph rewire "
            "weakens the runtime trajectory validator unless re-blessed. "
            "Regenerate via scripts/regen_golden_traces.py."
        )

    def test_ordering_constraints_match_golden(self):
        golden = load_graph_topology_golden()
        current = [list(c) for c in trajectory.ORDERING_CONSTRAINTS]
        assert current == golden["ordering_constraints"]

    def test_sector_team_count_matches_golden(self):
        golden = load_graph_topology_golden()
        assert trajectory.EXPECTED_SECTOR_TEAM_COUNT == golden["sector_team_count"]
