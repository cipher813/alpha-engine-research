"""
Unit tests for the LLM-as-judge eval pipeline (PR 2 of ROADMAP P3.1).

Covers:
- ``resolve_rubric_for_agent`` — agent_id → rubric_name mapping (and
  the intentionally-unevaluated cases).
- ``build_eval_s3_key`` — canonical S3 path layout.
- ``evaluate_artifact`` — end-to-end with a mocked judge LLM, asserting
  the rendered prompt, the wrapped artifact metadata, and the cost
  tracker integration.
- ``persist_eval_artifact`` — moto-mocked S3 round-trip, including
  re-validating from S3 bytes.

The judge LLM is mocked across all eval tests — we don't make real
Anthropic calls in the unit suite. Real-LLM smoke tests live with the
SF-wiring PR.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from alpha_engine_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)
from graph.state_schemas import (
    RubricDimensionScore,
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_artifact(agent_id: str, *, run_id: str = "test-run-001") -> DecisionArtifact:
    """Build a DecisionArtifact with shape-realistic input + output."""
    return DecisionArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered at run time>",
        ),
        input_data_snapshot={
            "team_id": "technology",
            "run_date": "2026-05-09",
            "market_regime": "neutral",
            "sector_tickers": ["AAPL", "MSFT"],
            "technical_scores_team": {
                "AAPL": {"rsi_14": 55, "technical_score": 70},
                "MSFT": {"rsi_14": 50, "technical_score": 65},
            },
        },
        input_data_summary="team_id=technology, sector_tickers=2",
        agent_output={
            "ranked_picks": [
                {"ticker": "AAPL", "quant_score": 70, "quant_rationale": "RSI 55, TS 70."},
                {"ticker": "MSFT", "quant_score": 65, "quant_rationale": "RSI 50, TS 65."},
            ],
        },
    )


def _make_llm_output() -> RubricEvalLLMOutput:
    """Build a realistic judge response."""
    return RubricEvalLLMOutput(
        dimension_scores=[
            RubricDimensionScore(
                dimension="numerical_grounding", score=4,
                reasoning="Both picks cite specific RSI + TS values.",
            ),
            RubricDimensionScore(
                dimension="signal_calibration", score=3,
                reasoning="Score gradient is directional but tight.",
            ),
            RubricDimensionScore(
                dimension="ranking_coherence", score=4,
                reasoning="Rank matches scores; reasoning differentiates picks.",
            ),
            RubricDimensionScore(
                dimension="regime_awareness", score=3,
                reasoning="Regime mentioned once but doesn't shape picks.",
            ),
        ],
        overall_reasoning="Solid grounding; regime engagement weakest.",
    )


# ── Rubric mapping ────────────────────────────────────────────────────────


class TestResolveRubricForAgent:
    def test_sector_quant_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_quant:technology") == "eval_rubric_sector_quant"
        assert resolve_rubric_for_agent("sector_quant:financials") == "eval_rubric_sector_quant"

    def test_sector_qual_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_qual:healthcare") == "eval_rubric_sector_qual"

    def test_sector_peer_review_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_peer_review:industrials") == "eval_rubric_sector_peer_review"

    def test_macro_economist_exact_match(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("macro_economist") == "eval_rubric_macro_economist"

    def test_ic_cio_exact_match(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("ic_cio") == "eval_rubric_ic_cio"

    def test_thesis_update_intentionally_unevaluated(self):
        # Held-stock thesis updates have artifacts captured (PR #87)
        # but no rubric — narrower call shape, structured update vs
        # novel analysis. Returns None so callers skip cleanly.
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("thesis_update:technology:AAPL") is None

    def test_unknown_agent_returns_none(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("totally_made_up_agent") is None
        assert resolve_rubric_for_agent("") is None


# ── S3 key shape ──────────────────────────────────────────────────────────


class TestBuildEvalS3Key:
    def test_canonical_path(self):
        from evals.judge import build_eval_s3_key
        ts = datetime(2026, 5, 9, 22, 30, tzinfo=timezone.utc)
        key = build_eval_s3_key(
            judged_agent_id="sector_quant:technology",
            run_id="run-abc-123",
            judge_model="claude-haiku-4-5",
            timestamp=ts,
        )
        assert key == "decision_artifacts/_eval/2026-05-09/sector_quant:technology/run-abc-123.claude-haiku-4-5.json"

    def test_default_timestamp_is_now(self):
        from evals.judge import build_eval_s3_key
        key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_model="claude-haiku-4-5",
        )
        # Today's UTC date partition; we just verify shape, not exact match
        assert "decision_artifacts/_eval/" in key
        assert "/ic_cio/r1.claude-haiku-4-5.json" in key

    def test_judge_model_disambiguates_two_tier(self):
        """Haiku + Sonnet of same (date, agent, run_id) must coexist —
        the judge_model segment is what keeps the two writes from
        clobbering each other (PR 3b two-tier orchestration)."""
        from evals.judge import build_eval_s3_key
        ts = datetime(2026, 5, 9, 22, 30, tzinfo=timezone.utc)
        haiku_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_model="claude-haiku-4-5", timestamp=ts,
        )
        sonnet_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_model="claude-sonnet-4-6", timestamp=ts,
        )
        assert haiku_key != sonnet_key
        assert haiku_key.endswith(".claude-haiku-4-5.json")
        assert sonnet_key.endswith(".claude-sonnet-4-6.json")

    def test_prefix_override_for_judge_only_mode(self):
        """PR 4e: ``judge_only=True`` test runs persist under a
        non-prod prefix. Verify the prefix override propagates."""
        from evals.judge import build_eval_s3_key
        ts = datetime(2026, 5, 9, 22, 30, tzinfo=timezone.utc)
        prod_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_model="claude-haiku-4-5", timestamp=ts,
        )
        test_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_model="claude-haiku-4-5", timestamp=ts,
            prefix="decision_artifacts/_eval_judge_only/",
        )
        assert prod_key.startswith("decision_artifacts/_eval/")
        assert test_key.startswith("decision_artifacts/_eval_judge_only/")
        assert prod_key != test_key


# ── evaluate_artifact end-to-end ──────────────────────────────────────────


class TestEvaluateArtifact:
    def test_unmapped_agent_raises(self):
        from evals.judge import evaluate_artifact
        artifact = _make_artifact("totally_made_up_agent")
        with pytest.raises(ValueError, match="No rubric mapped"):
            evaluate_artifact(artifact)

    def test_full_pipeline_with_mocked_llm(self, monkeypatch):
        from evals import judge as judge_mod

        # PR #106: with_structured_output(include_raw=True) returns a
        # runnable whose .invoke() yields a dict with parsed/raw/parsing_error.
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact,
                judge_model="claude-haiku-4-5",
                api_key="sk-test",
                judged_artifact_s3_key="decision_artifacts/2026/05/09/sector_quant:technology/r1.json",
            )

        # Result wrapping
        assert isinstance(result, RubricEvalArtifact)
        assert result.judged_agent_id == "sector_quant:technology"
        assert result.run_id == artifact.run_id
        assert result.rubric_id == "eval_rubric_sector_quant"
        assert result.judge_model == "claude-haiku-4-5"
        assert result.judged_artifact_s3_key.endswith("/r1.json")
        # Rubric version comes from the loaded prompt's frontmatter; we
        # don't pin to a specific semver here so prompt updates don't
        # break this test.
        assert result.rubric_version  # non-empty
        # Dimension scores propagated
        assert len(result.dimension_scores) == 4
        assert result.dimension_scores[0].dimension == "numerical_grounding"
        # Overall reasoning propagated
        assert "regime engagement" in result.overall_reasoning
        # First-attempt success — should NOT have called invoke more than once
        assert fake_structured.invoke.call_count == 1
        # And must have requested include_raw=True
        fake_llm.with_structured_output.assert_called_once()
        kwargs = fake_llm.with_structured_output.call_args.kwargs
        assert kwargs.get("include_raw") is True

    def test_renders_artifact_payload_into_prompt(self, monkeypatch):
        """Verify the rubric prompt is rendered with the artifact's
        input_data_snapshot + agent_output, not placeholder strings."""
        from evals import judge as judge_mod

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            artifact = _make_artifact("sector_quant:technology")
            judge_mod.evaluate_artifact(artifact, api_key="sk-test")

        # Inspect the rendered prompt passed to invoke.
        call_args = fake_structured.invoke.call_args
        messages = call_args[0][0]
        rendered = messages[0].content
        # Specific values from the snapshot must appear in the rendered prompt
        assert "AAPL" in rendered
        assert "technology" in rendered
        # And specific values from the agent output
        assert "ranked_picks" in rendered
        assert "RSI 55, TS 70." in rendered
        # Substitution variables should NOT remain unrendered
        assert "{agent_input}" not in rendered
        assert "{agent_output}" not in rendered

    def test_retries_on_parse_failure_and_succeeds(self, monkeypatch, caplog):
        """First attempt returns parsing_error; second attempt succeeds.
        Pin: retry loop catches the parse failure, logs WARNING with
        raw payload head, retries, returns the parsed result. Counts
        2 invokes total."""
        import logging
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content='[{"dimension": "x", "score": 4'),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }
        good_resp = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }

        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = [bad_resp, good_resp]
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"), \
             caplog.at_level(logging.WARNING):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
            )

        assert isinstance(result, RubricEvalArtifact)
        assert fake_structured.invoke.call_count == 2
        # Loud-log the parse failure with raw head for diagnostic.
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("parse attempt 1/3 failed" in m for m in warn_msgs)
        assert any("raw head=" in m for m in warn_msgs)

    def test_raises_after_max_retries_exhausted(self, monkeypatch):
        """All 3 attempts return parsing_error → raises RuntimeError
        with diagnostic context. Bounds worst-case latency + makes
        structural failures loud."""
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content="malformed"),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = bad_resp
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="parse attempts failed"):
                judge_mod.evaluate_artifact(
                    artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
                )

        # All 3 attempts fired (default MAX_JUDGE_RETRIES=3).
        assert fake_structured.invoke.call_count == 3

    def test_retry_count_param_overrides_default(self, monkeypatch):
        """``max_retries`` param lets callers tune the budget — useful
        for the test-track flag and for ad-hoc replay scripts that
        want to fail fast."""
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content="x"),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = bad_resp
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError):
                judge_mod.evaluate_artifact(
                    artifact, max_retries=1, api_key="sk-test",
                )

        # Single attempt only.
        assert fake_structured.invoke.call_count == 1


# ── persist_eval_artifact ─────────────────────────────────────────────────


@pytest.fixture
def mocked_s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        yield client


class TestPersistEvalArtifact:
    def test_writes_at_canonical_key(self, mocked_s3):
        from evals.judge import persist_eval_artifact

        artifact = RubricEvalArtifact(
            run_id="run-1",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="sector_quant:technology",
            rubric_id="eval_rubric_sector_quant",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="solid grounding",
        )
        key = persist_eval_artifact(
            artifact, s3_client=mocked_s3, bucket="alpha-engine-research",
        )

        assert key == "decision_artifacts/_eval/2026-05-09/sector_quant:technology/run-1.claude-haiku-4-5.json"
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=key)
        roundtrip = RubricEvalArtifact.model_validate(json.loads(obj["Body"].read()))
        assert roundtrip.judge_model == "claude-haiku-4-5"
        assert roundtrip.rubric_version == "1.0.0"
        assert len(roundtrip.dimension_scores) == 4

    def test_partition_date_matches_artifact_timestamp(self, mocked_s3):
        # Re-derives partition from the artifact's stamped timestamp so
        # replays land at the same key regardless of write-time clock.
        from evals.judge import persist_eval_artifact

        artifact = RubricEvalArtifact(
            run_id="run-2",
            timestamp="2026-04-25T03:14:00.000Z",  # different from "today"
            judged_agent_id="ic_cio",
            rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0",
            judge_model="claude-sonnet-4-6",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="x",
        )
        key = persist_eval_artifact(
            artifact, s3_client=mocked_s3, bucket="alpha-engine-research",
        )
        assert "/2026-04-25/" in key


# ── RubricEvalLLMOutput stringify defense ─────────────────────────────────


class TestRubricEvalLLMOutputStringDefense:
    """Pins the 2026-05-03 fix for an observed Haiku failure mode:
    ``dimension_scores`` returned as a JSON-encoded string instead of
    a structured array. First surfaced in the judge_only smoke against
    Sat 5/3 captures (5/32 evals failed at the schema boundary).
    Mirrors the JointFinalizationOutput defense pattern from PR #99.
    """

    def test_actual_list_passes_through_unchanged(self):
        out = RubricEvalLLMOutput(
            dimension_scores=[
                RubricDimensionScore(
                    dimension="numerical_grounding", score=4,
                    reasoning="cited multiples match filing",
                ),
                RubricDimensionScore(
                    dimension="signal_calibration", score=3,
                    reasoning="confidence within range",
                ),
            ],
            overall_reasoning="balanced",
        )
        assert len(out.dimension_scores) == 2
        assert out.dimension_scores[0].dimension == "numerical_grounding"

    def test_json_string_of_list_is_parsed_and_logged(self, caplog):
        """The exact failure shape Haiku produced 2026-05-03: a string
        whose contents are valid JSON for the expected list."""
        import logging
        payload = json.dumps([
            {"dimension": "numerical_grounding", "score": 4, "reasoning": "ok"},
            {"dimension": "signal_calibration", "score": 3, "reasoning": "tight"},
        ])

        with caplog.at_level(logging.WARNING):
            out = RubricEvalLLMOutput(
                dimension_scores=payload,  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

        assert len(out.dimension_scores) == 2
        assert out.dimension_scores[0].dimension == "numerical_grounding"
        assert any(
            "schema-vs-LLM drift" in rec.message
            or "JSON-string" in rec.message
            for rec in caplog.records
        )

    def test_invalid_json_string_raises_normal_pydantic_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="list_type|valid list"):
            RubricEvalLLMOutput(
                dimension_scores="not even close to valid json",  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

    def test_truncated_jsonlist_string_still_raises_loud(self):
        """Truncated mid-string from max_tokens hit. Validator's
        json.loads can't parse incomplete JSON → falls through to loud
        Pydantic list_type error. Pin so silent rescue can't regress."""
        from pydantic import ValidationError

        truncated = (
            '[\n  {\n    "dimension": "numerical_grounding",\n    "score": 4,\n'
            '    "reasoning": "Extensive citation analysis was attem'
            # truncated mid-string, no closing
        )
        with pytest.raises(ValidationError, match="list_type|valid list"):
            RubricEvalLLMOutput(
                dimension_scores=truncated,  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

    def test_field_description_present(self):
        """Pin the 'structured array, NOT JSON-encoded string' hint
        in the tool-use spec — same pattern as JointFinalizationOutput."""
        field_info = RubricEvalLLMOutput.model_fields["dimension_scores"]
        assert "structured array" in (field_info.description or "")
        assert "NOT" in (field_info.description or "")
