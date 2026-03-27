"""Tests for trajectory validation constants and logic."""

from evals.trajectory import (
    REQUIRED_NODES,
    ORDERING_CONSTRAINTS,
    EXPECTED_SECTOR_TEAM_COUNT,
    validate_trajectory,
)


class TestTrajectoryConstants:
    def test_required_nodes_count(self):
        assert len(REQUIRED_NODES) == 11

    def test_required_nodes_include_key_nodes(self):
        assert "fetch_data" in REQUIRED_NODES
        assert "sector_team_node" in REQUIRED_NODES
        assert "macro_economist_node" in REQUIRED_NODES
        assert "cio_node" in REQUIRED_NODES
        assert "email_sender_node" in REQUIRED_NODES

    def test_ordering_constraints_valid(self):
        node_set = set(REQUIRED_NODES)
        for before, after in ORDERING_CONSTRAINTS:
            assert before in node_set, f"{before} not in REQUIRED_NODES"
            assert after in node_set, f"{after} not in REQUIRED_NODES"

    def test_sector_team_count(self):
        assert EXPECTED_SECTOR_TEAM_COUNT == 6

    def test_fetch_data_is_first(self):
        # fetch_data should appear as "before" in constraints but never as "after"
        # for the initial fan-out edges
        before_set = {b for b, _ in ORDERING_CONSTRAINTS}
        assert "fetch_data" in before_set

    def test_email_sender_is_last(self):
        # email_sender should appear as "after" but never as "before"
        after_set = {a for _, a in ORDERING_CONSTRAINTS}
        before_set = {b for b, _ in ORDERING_CONSTRAINTS}
        assert "email_sender_node" in after_set
        assert "email_sender_node" not in before_set


class TestValidateTrajectoryDisabled:
    def test_returns_none_when_tracing_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        result = validate_trajectory()
        assert result is None

    def test_returns_none_when_tracing_false(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        result = validate_trajectory()
        assert result is None
