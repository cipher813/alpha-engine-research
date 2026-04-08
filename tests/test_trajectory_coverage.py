"""Tests for evals/trajectory.py."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from evals.trajectory import (
    EXPECTED_SECTOR_TEAM_COUNT,
    ORDERING_CONSTRAINTS,
    REQUIRED_NODES,
    validate_trajectory,
)


class TestValidateTrajectory:
    def test_skips_without_tracing_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate_trajectory()
            assert result is None

    @patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
    @patch("langsmith.Client")
    def test_no_runs_found(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.list_runs.return_value = []
        result = validate_trajectory(max_wait_seconds=0)
        assert result is not None
        assert result["passed"] is False
        assert "no_run_found" in result["failures"]

    @patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
    @patch("langsmith.Client")
    def test_all_nodes_present_passes(self, MockClient):
        mock_client = MockClient.return_value

        # Create a mock root run
        root_run = MagicMock()
        root_run.trace_id = "trace-123"
        root_run.start_time = datetime(2026, 4, 8, 12, 0, 0)
        root_run.end_time = datetime(2026, 4, 8, 12, 10, 0)
        mock_client.list_runs.side_effect = [
            [root_run],  # First call: list root runs
            _make_child_runs(),  # Second call: list child runs
        ]

        result = validate_trajectory(max_wait_seconds=0)
        assert result["passed"] is True
        assert len(result["failures"]) == 0
        assert result["duration_ms"] == 600000

    @patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
    @patch("langsmith.Client")
    def test_missing_node_fails(self, MockClient):
        mock_client = MockClient.return_value

        root_run = MagicMock()
        root_run.trace_id = "trace-123"
        root_run.start_time = datetime(2026, 4, 8, 12, 0, 0)
        root_run.end_time = datetime(2026, 4, 8, 12, 10, 0)

        # Omit email_sender_node
        children = _make_child_runs()
        children = [c for c in children if c.name != "email_sender_node"]

        mock_client.list_runs.side_effect = [[root_run], children]

        result = validate_trajectory(max_wait_seconds=0)
        assert result["passed"] is False
        assert any("missing_node" in f for f in result["failures"])

    @patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
    @patch("langsmith.Client")
    def test_wrong_team_count_fails(self, MockClient):
        mock_client = MockClient.return_value

        root_run = MagicMock()
        root_run.trace_id = "trace-123"
        root_run.start_time = datetime(2026, 4, 8, 12, 0, 0)
        root_run.end_time = datetime(2026, 4, 8, 12, 10, 0)

        # Only 3 sector teams instead of 6
        children = _make_child_runs(n_teams=3)
        mock_client.list_runs.side_effect = [[root_run], children]

        result = validate_trajectory(max_wait_seconds=0)
        assert result["passed"] is False
        assert any("sector_team_count" in f for f in result["failures"])

    @patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
    @patch("langsmith.Client")
    def test_ordering_violation_fails(self, MockClient):
        mock_client = MockClient.return_value

        root_run = MagicMock()
        root_run.trace_id = "trace-123"
        root_run.start_time = datetime(2026, 4, 8, 12, 0, 0)
        root_run.end_time = datetime(2026, 4, 8, 12, 10, 0)

        # Make cio_node start before score_aggregator (violation)
        children = _make_child_runs()
        for c in children:
            if c.name == "cio_node":
                c.start_time = datetime(2026, 4, 8, 12, 1, 0)  # Very early
            if c.name == "score_aggregator":
                c.start_time = datetime(2026, 4, 8, 12, 9, 0)  # Very late

        mock_client.list_runs.side_effect = [[root_run], children]

        result = validate_trajectory(max_wait_seconds=0)
        assert result["passed"] is False
        assert any("ordering_violation" in f for f in result["failures"])


def _make_child_runs(n_teams: int = 6) -> list[MagicMock]:
    """Create mock child runs matching the expected trajectory."""
    children = []
    base = datetime(2026, 4, 8, 12, 0, 0)
    offset = 0

    for node in REQUIRED_NODES:
        if node == "sector_team_node":
            for i in range(n_teams):
                child = MagicMock()
                child.name = "sector_team_node"
                child.start_time = base + timedelta(seconds=offset + i)
                children.append(child)
            offset += n_teams
        else:
            child = MagicMock()
            child.name = node
            child.start_time = base + timedelta(seconds=offset)
            children.append(child)
            offset += 10

    return children
