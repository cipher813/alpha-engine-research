"""
Unit tests for ``graph.reducers`` — the typed-state successors to the
``research_graph._merge_dicts`` reducer (PR #50, 2026-04-29).

Three reducer behaviors covered: ``take_last`` (overwrite), ``merge_typed_dicts``
(last-write-wins per key + canonical key order), ``reject_on_conflict``
(raise on overlap + canonical key order).

Workstream: typed LangGraph state + Pydantic agent outputs + decision-artifact
capture (alpha-engine-research-typed-state-capture-260429.md).
"""

from __future__ import annotations

import itertools

import pytest

from graph.reducers import merge_typed_dicts, reject_on_conflict, take_last


class TestTakeLast:
    def test_left_replaced_by_right(self):
        assert take_last("old", "new") == "new"

    def test_works_with_none(self):
        assert take_last(None, "x") == "x"
        assert take_last("x", None) is None

    def test_works_with_complex_types(self):
        left = {"a": 1, "b": 2}
        right = {"c": 3}
        assert take_last(left, right) == {"c": 3}


class TestMergeTypedDictsBasics:
    def test_both_none(self):
        assert merge_typed_dicts(None, None) == {}

    def test_left_none(self):
        result = merge_typed_dicts(None, {"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_right_none(self):
        result = merge_typed_dicts({"b": 2, "a": 1}, None)
        assert list(result.keys()) == ["a", "b"]

    def test_disjoint_keys_canonical_order(self):
        left = {"financials": "F", "consumer": "C"}
        right = {"technology": "T", "energy": "E"}
        result = merge_typed_dicts(left, right)
        assert list(result.keys()) == [
            "consumer",
            "energy",
            "financials",
            "technology",
        ]

    def test_overlapping_keys_right_wins(self):
        left = {"technology": "old", "financials": "F"}
        right = {"technology": "new"}
        result = merge_typed_dicts(left, right)
        assert result["technology"] == "new"
        assert list(result.keys()) == ["financials", "technology"]


class TestMergeTypedDictsDeterminism:
    """All permutations of completion order must produce the same result."""

    TEAM_OUTPUTS = [
        {"financials": {"recs": ["JPM"]}},
        {"technology": {"recs": ["AAPL"]}},
        {"healthcare": {"recs": ["UNH"]}},
        {"energy": {"recs": ["XOM"]}},
    ]

    @staticmethod
    def _reduce_in_order(team_outputs: list[dict]) -> dict:
        acc = None
        for partial in team_outputs:
            acc = merge_typed_dicts(acc, partial)
        return acc

    def test_iteration_order_canonical(self):
        result = self._reduce_in_order(self.TEAM_OUTPUTS)
        assert list(result.keys()) == ["energy", "financials", "healthcare", "technology"]

    def test_arbitrary_completion_orders_yield_same_result(self):
        canonical = self._reduce_in_order(self.TEAM_OUTPUTS)
        for perm in itertools.permutations(self.TEAM_OUTPUTS):
            result = self._reduce_in_order(list(perm))
            assert result == canonical
            assert list(result.keys()) == list(canonical.keys())


class TestMergeTypedDictsValuePreservation:
    def test_inner_values_unchanged(self):
        left = {"technology": {"recs": ["AAPL"], "score": 78}}
        right = {"financials": {"recs": ["JPM"], "score": 65}}
        result = merge_typed_dicts(left, right)
        assert result["technology"] == {"recs": ["AAPL"], "score": 78}
        assert result["financials"] == {"recs": ["JPM"], "score": 65}

    def test_does_not_mutate_inputs(self):
        left = {"b": 2, "a": 1}
        right = {"c": 3}
        original_left = list(left.keys())
        original_right = list(right.keys())
        merge_typed_dicts(left, right)
        assert list(left.keys()) == original_left
        assert list(right.keys()) == original_right


class TestRejectOnConflictBasics:
    def test_both_none(self):
        assert reject_on_conflict(None, None) == {}

    def test_left_none(self):
        result = reject_on_conflict(None, {"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_right_none(self):
        result = reject_on_conflict({"b": 2, "a": 1}, None)
        assert list(result.keys()) == ["a", "b"]

    def test_disjoint_keys_canonical_order(self):
        left = {"financials": "F", "consumer": "C"}
        right = {"technology": "T", "energy": "E"}
        result = reject_on_conflict(left, right)
        assert list(result.keys()) == [
            "consumer",
            "energy",
            "financials",
            "technology",
        ]


class TestRejectOnConflictRaises:
    def test_single_overlap(self):
        with pytest.raises(RuntimeError, match=r"keys written by multiple branches"):
            reject_on_conflict({"technology": "T1"}, {"technology": "T2"})

    def test_multiple_overlaps_named(self):
        try:
            reject_on_conflict(
                {"technology": "T1", "healthcare": "H1"},
                {"technology": "T2", "healthcare": "H2", "energy": "E"},
            )
            pytest.fail("expected RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            assert "technology" in msg
            assert "healthcare" in msg
            # energy is disjoint, must NOT appear in the error
            assert "energy" not in msg

    def test_overlap_keys_sorted_in_message(self):
        """Overlap-key list in the error message is sorted, not insertion-ordered,
        so the error is reproducible across runs (matches the canonical-ordering
        invariant that motivates these reducers)."""
        try:
            reject_on_conflict(
                {"technology": "T1", "healthcare": "H1"},
                {"healthcare": "H2", "technology": "T2"},
            )
            pytest.fail("expected RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            # sorted: healthcare comes before technology
            assert msg.index("healthcare") < msg.index("technology")
