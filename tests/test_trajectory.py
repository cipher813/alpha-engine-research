"""Tests for trajectory validation constants and logic."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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


def _stub_run(trace_id: str = "trace-1") -> MagicMock:
    """Stub root run with end_time so duration computes cleanly."""
    run = MagicMock()
    run.trace_id = trace_id
    run.start_time = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    run.end_time = datetime(2026, 4, 27, 12, 5, 0, tzinfo=timezone.utc)
    return run


def _stub_child(name: str, *, ts_offset_s: float = 0.0) -> MagicMock:
    child = MagicMock()
    child.name = name
    child.start_time = datetime(
        2026, 4, 27, 12, 1, 0, tzinfo=timezone.utc,
    ).replace(second=int(ts_offset_s))
    return child


class TestTraceLagPolling:
    """Regression: the validator was racing the LangSmith async flusher and
    reporting ``missing_node: email_sender_node`` on every research run.
    The fix polls the child-runs query until all REQUIRED_NODES appear
    or a completeness timeout fires."""

    def _all_required_children(self) -> list:
        # One child span per required node, each with the right name and
        # increasing start_time so ordering constraints are satisfied.
        return [
            _stub_child(name, ts_offset_s=i)
            for i, name in enumerate(REQUIRED_NODES)
            # sector_team_node fans out via Send() — emit it 6 times to
            # satisfy EXPECTED_SECTOR_TEAM_COUNT.
            for _ in (range(6) if name == "sector_team_node" else range(1))
        ]

    def test_late_arriving_email_sender_eventually_passes(self, monkeypatch):
        """The trace race the validator was hitting: every node BUT
        ``email_sender_node`` is in the trace on first fetch; on the second
        fetch (after a poll interval), all nodes are present. Validator
        should pass without ever logging a failure."""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        # First fetch: all nodes except email_sender (the race).
        # Second fetch: all nodes including email_sender.
        first_batch = [
            c for c in self._all_required_children()
            if c.name != "email_sender_node"
        ]
        second_batch = self._all_required_children()
        fetch_results = iter([first_batch, second_batch])

        client_mock = MagicMock()
        client_mock.list_runs.side_effect = lambda **kwargs: (
            [_stub_run()] if kwargs.get("is_root") else next(fetch_results)
        )

        with patch("langsmith.Client", return_value=client_mock), \
             patch("evals.trajectory.time.sleep"):
            result = validate_trajectory(completeness_timeout_seconds=10)

        assert result is not None
        assert result["passed"] is True, (
            f"Validator should poll past trace lag; got failures: "
            f"{result.get('failures')}"
        )
        assert result["node_counts"].get("email_sender_node") == 1

    def test_genuinely_missing_node_still_fails_after_timeout(self, monkeypatch):
        """If a node is genuinely missing (not just trace lag), the
        validator must surface it as a failure once the completeness
        timeout fires. Otherwise the poll trains operators to ignore real
        regressions just like the old false-positive did."""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        # Every fetch returns the same incomplete set — node is genuinely
        # missing, not lagging.
        incomplete = [
            c for c in self._all_required_children()
            if c.name != "cio_node"  # mid-flow node, not just terminal
        ]
        client_mock = MagicMock()
        client_mock.list_runs.side_effect = lambda **kwargs: (
            [_stub_run()] if kwargs.get("is_root") else incomplete
        )

        with patch("langsmith.Client", return_value=client_mock), \
             patch("evals.trajectory.time.sleep"):
            result = validate_trajectory(completeness_timeout_seconds=1)

        assert result is not None
        assert result["passed"] is False
        assert any(f.startswith("missing_node: cio_node") for f in result["failures"])

    def test_first_fetch_complete_does_not_sleep(self, monkeypatch):
        """Happy path: when the first child-runs fetch already has all
        required nodes, the validator must not sleep — completeness
        timeout is a fallback, not a delay."""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        complete = self._all_required_children()
        client_mock = MagicMock()
        client_mock.list_runs.side_effect = lambda **kwargs: (
            [_stub_run()] if kwargs.get("is_root") else complete
        )

        sleep_mock = MagicMock()
        with patch("langsmith.Client", return_value=client_mock), \
             patch("evals.trajectory.time.sleep", sleep_mock):
            result = validate_trajectory(completeness_timeout_seconds=30)

        assert result is not None
        assert result["passed"] is True
        # No sleep calls — the completeness loop short-circuits on first
        # fetch when all required nodes are present.
        assert sleep_mock.call_count == 0


class TestNoFalseFailureOnEmailSenderRegression:
    """Locking the specific 2026-04-20 incident: 'Trajectory validation
    FAILED — 1 failures: [missing_node: email_sender_node]' was logged
    after every successful research run. The graph node IS named
    ``email_sender_node`` and IS in REQUIRED_NODES — the old single-fetch
    behavior was racing the async LangSmith flusher. This test ensures
    the polling fix resolves that exact failure mode."""

    def test_email_sender_lag_does_not_become_a_false_alarm(self, monkeypatch):
        from evals.trajectory import REQUIRED_NODES

        # Reproduce the incident shape: every required node EXCEPT
        # email_sender_node lands in the first fetch (LangSmith hasn't
        # flushed the terminal node yet).
        all_minus_email = [
            _stub_child(name, ts_offset_s=i)
            for i, name in enumerate(REQUIRED_NODES)
            for _ in (range(6) if name == "sector_team_node" else range(1))
            if name != "email_sender_node"
        ]
        all_present = all_minus_email + [
            _stub_child("email_sender_node", ts_offset_s=20),
        ]

        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        fetches = iter([all_minus_email, all_minus_email, all_present])
        client_mock = MagicMock()
        client_mock.list_runs.side_effect = lambda **kwargs: (
            [_stub_run()] if kwargs.get("is_root") else next(fetches)
        )

        with patch("langsmith.Client", return_value=client_mock), \
             patch("evals.trajectory.time.sleep"):
            result = validate_trajectory(completeness_timeout_seconds=15)

        assert result is not None, "validator returned None on the lag scenario"
        assert result["passed"] is True, (
            f"validator must wait for email_sender_node to flush rather than "
            f"reporting it as missing on the first fetch. Got: {result['failures']}"
        )
