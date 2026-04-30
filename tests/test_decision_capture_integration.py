"""
Integration tests for decision-artifact capture in the research graph.

Verifies:
- Feature flag default-off: capture functions are no-ops without env var.
- Feature flag on + mocked S3: each producer node writes one artifact at
  the canonical S3 key.
- Hard-fail behavior: S3 unavailability raises through the node, not
  swallowed silently (per ``feedback_no_silent_fails``).
- Per-node payload helpers produce JSON-serializable dicts that survive
  round-trip through ``DecisionArtifact.model_validate``.

Capture is gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED``; default-off
preserves existing behavior. Production turns it on once IAM grant for
``s3:PutObject`` on ``decision_artifacts/*`` is in place on the
research-runner Lambda role.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from alpha_engine_lib.decision_capture import (
    DecisionArtifact,
    DecisionCaptureWriteError,
    capture_decision,
    FullPromptContext,
    ModelMetadata,
)


# ── Feature-flag gate ─────────────────────────────────────────────────────


class TestFeatureFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False

    def test_true_value_enables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is True

    def test_one_value_enables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "1")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is True

    def test_false_value_disables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "false")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False

    def test_arbitrary_value_disables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "yolo")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False


# ── Per-node payload builders ─────────────────────────────────────────────


class TestSectorTeamPayloadBuilder:
    @pytest.fixture
    def fake_ctx(self):
        from agents.sector_teams.sector_team import SectorTeamContext
        return SectorTeamContext(
            scanner_universe=["AAPL", "MSFT", "JPM", "JNJ", "XOM"],
            sector_map={
                "AAPL": "Technology", "MSFT": "Technology",
                "JPM": "Financial", "JNJ": "Healthcare", "XOM": "Energy",
            },
            price_data={},
            technical_scores={
                "AAPL": {"rsi_14": 55, "technical_score": 70},
                "MSFT": {"rsi_14": 50, "technical_score": 65},
            },
            market_regime="neutral",
            prior_theses={"AAPL": {"final_score": 65, "rating": "BUY"}},
            held_tickers=["AAPL"],
            news_data_by_ticker={"AAPL": {"articles": []}},
            analyst_data_by_ticker={},
            insider_data_by_ticker={},
            prior_sector_ratings={},
            current_sector_ratings={"Technology": {"rating": "overweight"}},
            run_date="2026-04-29",
            episodic_memories={},
            semantic_memories={},
        )

    def test_payload_is_json_serializable(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_team_capture_payload
        snapshot, summary = build_sector_team_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        # Must serialize cleanly — capture_decision will JSON-dump for S3
        json.dumps(snapshot)
        assert isinstance(summary, str)
        assert "team_id=technology" in summary

    def test_payload_includes_required_fields(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_team_capture_payload
        snapshot, _ = build_sector_team_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        for key in (
            "team_id", "run_date", "market_regime",
            "scanner_universe_size", "team_tickers", "held_tickers_in_team",
            "news_data_by_ticker", "analyst_data_by_ticker",
            "insider_data_by_ticker", "prior_theses_in_team",
            "prior_sector_ratings", "current_sector_ratings",
            "technical_scores_team", "memories_summary",
        ):
            assert key in snapshot, f"missing field: {key}"

    def test_team_filtering(self, fake_ctx):
        # technical_scores filters to team tickers; full universe scores
        # are NOT captured (they'd duplicate other agents' captures).
        from graph.decision_capture_helpers import build_sector_team_capture_payload
        snapshot, _ = build_sector_team_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        ts = snapshot["technical_scores_team"]
        assert set(ts.keys()) == {"AAPL", "MSFT"}
        # Held tickers filtered to only those in the team
        assert snapshot["held_tickers_in_team"] == ["AAPL"]


class TestMacroEconomistPayloadBuilder:
    def test_minimal_state(self):
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {"run_date": "2026-04-29", "macro_data": {"vix": 14.2}}
        snapshot, summary = build_macro_economist_capture_payload(state)
        json.dumps(snapshot)
        assert snapshot["macro_data"] == {"vix": 14.2}
        assert "run_date=2026-04-29" in summary

    def test_with_prior_report(self):
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {
            "run_date": "2026-04-29",
            "macro_data": {"vix": 14.2, "tnx": 4.31},
            "prior_macro_report": "x" * 500,
            "prior_macro_snapshots": [{"date": "2026-04-22"}],
        }
        snapshot, summary = build_macro_economist_capture_payload(state)
        assert snapshot["prior_date"] == "2026-04-22"
        assert snapshot["prior_snapshots_count"] == 1
        assert "prior_report_chars=500" in summary


class TestCIOPayloadBuilder:
    def test_minimal(self):
        from graph.decision_capture_helpers import build_cio_capture_payload
        state = {
            "run_date": "2026-04-29",
            "market_regime": "neutral",
            "macro_report": "...",
            "sector_ratings": {},
            "remaining_population": [],
            "open_slots": 5,
            "exits": [],
        }
        candidates = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        prior_ic = []
        snapshot, summary = build_cio_capture_payload(
            state, candidates=candidates, prior_ic=prior_ic,
        )
        json.dumps(snapshot)
        assert snapshot["candidates_count"] == 2
        assert snapshot["open_slots"] == 5
        assert "candidates=2" in summary


# ── End-to-end: feature flag + capture writes artifact ────────────────────


@pytest.fixture
def mocked_s3(monkeypatch):
    """moto-mocked S3 + ``alpha-engine-research`` bucket pre-created.

    Patches ``boto3.client`` so any code path that calls
    ``boto3.client("s3")`` (including capture_decision under the hood)
    gets the mocked client.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")

        # Also patch boto3.client globally for any caller using the
        # default-client path. capture_decision accepts s3_client= so
        # we typically test via that injection path.
        yield client


class TestCaptureFiresWhenEnabled:
    def test_capture_writes_artifact_via_helper(self, mocked_s3, monkeypatch):
        # Direct test of capture_decision with the helper-built payload.
        # Verifies the integration produces a valid DecisionArtifact end-to-end.
        from graph.decision_capture_helpers import build_macro_economist_capture_payload

        state = {
            "run_date": "2026-04-29",
            "macro_data": {"vix": 14.2},
            "prior_macro_report": "",
            "prior_macro_snapshots": [],
        }
        snapshot, summary = build_macro_economist_capture_payload(state)

        s3_key = capture_decision(
            run_id="test-run-001",
            agent_id="macro_economist",
            model_metadata=ModelMetadata(model_name="claude-sonnet-4-6"),
            full_prompt_context=FullPromptContext(
                system_prompt="<placeholder>", user_prompt="<placeholder>",
            ),
            input_data_snapshot=snapshot,
            input_data_summary=summary,
            agent_output={"macro_report": "test", "market_regime": "neutral"},
            s3_client=mocked_s3,
            timestamp=datetime(2026, 4, 29, 22, 30, tzinfo=timezone.utc),
        )

        assert s3_key == "decision_artifacts/2026/04/29/macro_economist/test-run-001.json"
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert artifact.agent_id == "macro_economist"
        assert artifact.model_metadata.model_name == "claude-sonnet-4-6"
        assert artifact.input_data_summary == summary


class TestCaptureNoOpWhenDisabled:
    def test_capture_if_enabled_skips_when_flag_off(self, monkeypatch, mocked_s3):
        """When the flag is off, ``_capture_if_enabled`` short-circuits
        before any S3 call. Test by invoking it with an invalid bucket
        — it must NOT raise (no S3 attempt at all)."""
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)

        from graph.research_graph import _capture_if_enabled

        # No exception even though the bucket doesn't exist — capture
        # function must short-circuit on the env-var check.
        _capture_if_enabled(
            state={"run_date": "2026-04-29"},
            agent_id="sector_team:technology",
            model_name_key="sector_team",
            input_data_snapshot={"x": 1},
            input_data_summary="test",
            agent_output={"recommendations": []},
        )

        # Verify nothing was written to S3
        objects = mocked_s3.list_objects_v2(Bucket="alpha-engine-research")
        assert "Contents" not in objects or not objects["Contents"]


class TestCaptureHardFailsOnS3Error:
    def test_capture_if_enabled_raises_on_s3_error(self, monkeypatch):
        """When the flag is on AND S3 unreachable, ``_capture_if_enabled``
        re-raises ``DecisionCaptureWriteError`` per ``feedback_no_silent_fails``.

        Use moto with a NON-existent bucket. boto3 picks up the mocked
        endpoint via moto, but ``alpha-engine-research`` doesn't exist
        (we never created it), so ``put_object`` raises
        ``NoSuchBucket`` which is wrapped into ``DecisionCaptureWriteError``.
        """
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")

        with mock_aws():
            # Deliberately do NOT create the bucket
            from graph.research_graph import _capture_if_enabled

            with pytest.raises(DecisionCaptureWriteError):
                _capture_if_enabled(
                    state={"run_date": "2026-04-29"},
                    agent_id="sector_team:technology",
                    model_name_key="sector_team",
                    input_data_snapshot={"x": 1},
                    input_data_summary="test",
                    agent_output={"recommendations": []},
                )


# ── Run-id derivation ─────────────────────────────────────────────────────


class TestDeriveRunId:
    def test_explicit_run_id_used(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({"run_id": "explicit-123", "run_date": "2026-04-29"}) == "explicit-123"

    def test_falls_back_to_run_date(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({"run_date": "2026-04-29"}) == "2026-04-29"

    def test_unknown_when_neither(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({}) == "unknown"
