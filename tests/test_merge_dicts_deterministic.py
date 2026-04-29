"""
Regression tests for ``graph.research_graph._merge_dicts`` — the reducer used
on ``sector_team_outputs`` for LangGraph's Send fan-out.

The reducer must produce canonical (sorted) key order regardless of the
order in which Send-branch results arrive. Without sorting, downstream
consumers (``score_aggregator``, ``cio_node``, ``archive_writer``) iterate
the merged dict in completion order, and that propagates into
``cio_node``'s ``candidates[:open_slots]`` slice — different completion
order yields different advance decisions, different exits, and different
``signals.json`` across re-runs of the same input.

Diagnosed 2026-04-29 during baseline capture for the typed-state
workstream — see ``tests/fixtures/BASELINE_README.md``.
"""

from __future__ import annotations

from graph.research_graph import _merge_dicts


class TestMergeDictsBasics:
    def test_both_none(self):
        assert _merge_dicts(None, None) == {}

    def test_left_none(self):
        result = _merge_dicts(None, {"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]
        assert result == {"a": 1, "b": 2}

    def test_right_none(self):
        result = _merge_dicts({"b": 2, "a": 1}, None)
        assert list(result.keys()) == ["a", "b"]
        assert result == {"a": 1, "b": 2}

    def test_disjoint_keys(self):
        left = {"financials": "F", "consumer": "C"}
        right = {"technology": "T", "energy": "E"}
        result = _merge_dicts(left, right)
        assert list(result.keys()) == ["consumer", "energy", "financials", "technology"]

    def test_overlapping_keys_right_wins(self):
        left = {"technology": "old", "financials": "F"}
        right = {"technology": "new"}
        result = _merge_dicts(left, right)
        assert result["technology"] == "new"
        assert list(result.keys()) == ["financials", "technology"]


class TestMergeDictsDeterminism:
    """
    Simulate the LangGraph Send fan-out: six sector teams, each producing a
    single-key dict, merged in arbitrary completion order. The reducer is
    called sequentially as each Send branch returns — this matches LangGraph's
    runtime semantics for a typed-dict reducer.
    """

    TEAM_OUTPUTS = [
        {"financials": {"recommendations": ["JPM", "V"]}},
        {"technology": {"recommendations": ["AAPL", "MSFT"]}},
        {"healthcare": {"recommendations": ["UNH", "JNJ"]}},
        {"consumer_discretionary": {"recommendations": ["AMZN", "TSLA"]}},
        {"energy": {"recommendations": ["XOM", "CVX"]}},
        {"industrials": {"recommendations": ["HON", "UNP"]}},
    ]

    @staticmethod
    def _reduce_in_order(team_outputs: list[dict]) -> dict:
        """Sequentially apply ``_merge_dicts`` across the list, mimicking
        LangGraph's reducer call pattern."""
        acc = None
        for partial in team_outputs:
            acc = _merge_dicts(acc, partial)
        return acc

    def test_completion_order_a_vs_b_yields_same_result(self):
        # Order A: insertion order of TEAM_OUTUPS as defined
        order_a = self.TEAM_OUTPUTS
        # Order B: reversed completion (e.g. industrials finishes first)
        order_b = list(reversed(self.TEAM_OUTPUTS))

        result_a = self._reduce_in_order(order_a)
        result_b = self._reduce_in_order(order_b)

        # Bytewise equal — same keys, same insertion order, same values
        assert result_a == result_b
        assert list(result_a.keys()) == list(result_b.keys())

    def test_iteration_order_is_canonical(self):
        # Iteration order must be alphabetical (sorted) regardless of arrival
        # order, so downstream consumers (score_aggregator + cio_node) get
        # deterministic candidate lists.
        result = self._reduce_in_order(self.TEAM_OUTPUTS)
        assert list(result.keys()) == [
            "consumer_discretionary",
            "energy",
            "financials",
            "healthcare",
            "industrials",
            "technology",
        ]

    def test_arbitrary_completion_orders_all_yield_canonical(self):
        import itertools

        # Every permutation of completion order must produce the same
        # canonical-ordered result. 720 permutations for 6 teams — fine.
        canonical = self._reduce_in_order(self.TEAM_OUTPUTS)
        canonical_keys = list(canonical.keys())

        for perm in itertools.permutations(self.TEAM_OUTPUTS):
            result = self._reduce_in_order(list(perm))
            assert result == canonical
            assert list(result.keys()) == canonical_keys


class TestMergeDictsValuePreservation:
    """Sorting must not mutate the inner values — only re-order top-level keys."""

    def test_inner_values_unchanged(self):
        left = {"technology": {"recs": ["AAPL"], "score": 78}}
        right = {"financials": {"recs": ["JPM"], "score": 65}}
        result = _merge_dicts(left, right)
        assert result["technology"] == {"recs": ["AAPL"], "score": 78}
        assert result["financials"] == {"recs": ["JPM"], "score": 65}

    def test_does_not_mutate_inputs(self):
        left = {"b": 2, "a": 1}
        right = {"c": 3}
        original_left_keys = list(left.keys())
        original_right_keys = list(right.keys())
        _merge_dicts(left, right)
        # Inputs preserved
        assert list(left.keys()) == original_left_keys
        assert list(right.keys()) == original_right_keys
