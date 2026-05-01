"""
Unit tests for ``graph.llm_cost_tracker``.

Locks down:

- Callback handler extracts token usage from modern AIMessage shape +
  legacy llm_output shape; hard-fails on missing usage.
- Callback bubbles into the active ``track_llm_cost`` frame; calls outside
  any frame are no-op (logged) instead of raising.
- ``track_llm_cost`` enter/exit balances correctly + populates
  ``ModelMetadata`` + ``FullPromptContext`` on exit.
- Multiple LLM calls within one frame aggregate (ReAct simulation).
- ``recompute_cost`` runs against the cached price table and populates
  ``cost_usd``.
- ``pop_metadata_for`` removes the entry on read (bounded under fan-out).
- Frame stack underflow is detected.
- **PR 3** — per-call JSONL sink: rows buffered + flushed at scope exit
  to ``s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/
  {run_id}/{agent_id}.jsonl``; gated on the same env flag as
  decision_capture; hard-fails on S3 error per
  ``feedback_no_silent_fails``.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from alpha_engine_lib.cost import PriceCard, PriceTable
from alpha_engine_lib.decision_capture import FullPromptContext, ModelMetadata


# ── Test fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def fake_price_table_yaml(tmp_path: Path) -> Path:
    """Write a minimal model_pricing.yaml the tracker can load."""
    yaml_path = tmp_path / "model_pricing.yaml"
    yaml_path.write_text(
        "cards:\n"
        "  - model_name: claude-haiku-4-5\n"
        "    effective_from: 2026-01-01\n"
        "    input_per_1m: 1.0\n"
        "    output_per_1m: 5.0\n"
        "    cache_read_per_1m: 0.1\n"
        "    cache_create_per_1m: 1.25\n"
        "  - model_name: claude-sonnet-4-6\n"
        "    effective_from: 2026-01-01\n"
        "    input_per_1m: 3.0\n"
        "    output_per_1m: 15.0\n"
        "    cache_read_per_1m: 0.3\n"
        "    cache_create_per_1m: 3.75\n"
    )
    return yaml_path


@pytest.fixture(autouse=True)
def reset_tracker_state(monkeypatch, tmp_path):
    """Clear module-level cached price table + frame stack between tests."""
    from graph import llm_cost_tracker

    llm_cost_tracker._reset_price_table_for_tests()
    # Wipe completed metadata + frame stack via fresh ContextVar values.
    llm_cost_tracker._frame_stack.set([])
    llm_cost_tracker._completed_metadata.set({})
    yield
    llm_cost_tracker._reset_price_table_for_tests()


@pytest.fixture
def patched_pricing_path(monkeypatch, fake_price_table_yaml):
    """Point ``_resolve_pricing_path`` at the test yaml."""
    from graph import llm_cost_tracker
    monkeypatch.setattr(
        llm_cost_tracker, "_resolve_pricing_path",
        lambda: fake_price_table_yaml,
    )


# ── Helpers for fake LangChain LLMResult shapes ──────────────────────────


def _make_modern_response(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_create: int = 0,
    model_name: str = "claude-haiku-4-5",
) -> MagicMock:
    """Build a fake LLMResult mimicking modern langchain-anthropic shape."""
    message = MagicMock()
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {
            "cache_read": cache_read,
            "cache_creation": cache_create,
        },
    }
    message.response_metadata = {"model_name": model_name}

    generation = MagicMock()
    generation.message = message

    response = MagicMock()
    response.generations = [[generation]]
    response.llm_output = None
    return response


def _make_legacy_response(
    *,
    input_tokens: int,
    output_tokens: int,
    model_name: str = "claude-haiku-4-5",
) -> MagicMock:
    """Build a fake LLMResult mimicking legacy llm_output shape (no cache fields)."""
    response = MagicMock()
    response.generations = [[]]
    response.llm_output = {
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "model": model_name,
    }
    return response


def _make_empty_response() -> MagicMock:
    """Response with no usage information at all — should hard-fail."""
    response = MagicMock()
    response.generations = [[]]
    response.llm_output = None
    return response


# ── Callback handler ──────────────────────────────────────────────────────


class TestCostTelemetryCallback:
    def test_modern_shape_extracted(self):
        from graph.llm_cost_tracker import CostTelemetryCallback

        cb = CostTelemetryCallback()
        usage = cb._extract_usage(_make_modern_response(
            input_tokens=4000, output_tokens=1200,
            cache_read=2000, cache_create=500,
        ))
        assert usage == {
            "input_tokens": 4000,
            "output_tokens": 1200,
            "cache_read_tokens": 2000,
            "cache_create_tokens": 500,
        }

    def test_legacy_shape_extracted_no_cache(self):
        from graph.llm_cost_tracker import CostTelemetryCallback

        cb = CostTelemetryCallback()
        usage = cb._extract_usage(_make_legacy_response(
            input_tokens=1000, output_tokens=500,
        ))
        assert usage == {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
        }

    def test_empty_shape_hard_fails(self):
        from graph.llm_cost_tracker import CostTelemetryCallback

        cb = CostTelemetryCallback()
        with pytest.raises(RuntimeError, match="no usage metadata"):
            cb._extract_usage(_make_empty_response())

    def test_model_name_extracted_modern(self):
        from graph.llm_cost_tracker import CostTelemetryCallback

        cb = CostTelemetryCallback()
        name = cb._extract_model_name(
            _make_modern_response(
                input_tokens=10, output_tokens=5,
                model_name="claude-sonnet-4-6",
            )
        )
        assert name == "claude-sonnet-4-6"

    def test_no_active_frame_skips_silently(self, caplog):
        from graph.llm_cost_tracker import CostTelemetryCallback

        cb = CostTelemetryCallback()
        cb.on_llm_end(_make_modern_response(input_tokens=10, output_tokens=5))
        # No exception — call outside a frame is a no-op (debug log).
        # No assertion on caplog content; just verify no raise.

    def test_callback_accumulates_into_active_frame(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, _current_frame,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="test_agent",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(
                input_tokens=100, output_tokens=50,
                cache_read=20, cache_create=10,
            ))
            frame = _current_frame()
            assert frame is not None
            assert frame.input_tokens == 100
            assert frame.output_tokens == 50
            assert frame.cache_read_tokens == 20
            assert frame.cache_create_tokens == 10
            assert frame.call_count == 1


# ── Frame lifecycle (track_llm_cost) ──────────────────────────────────────


class TestTrackLlmCostBasics:
    def test_frame_pops_on_exit(self, patched_pricing_path):
        from graph.llm_cost_tracker import track_llm_cost, _frame_stack

        assert _frame_stack.get() == []
        with track_llm_cost(agent_id="agent_a", model_name_fallback="claude-haiku-4-5"):
            assert len(_frame_stack.get()) == 1
        assert _frame_stack.get() == []

    def test_metadata_stashed_on_exit(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, pop_metadata_for,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="agent_a",
            model_name_fallback="claude-haiku-4-5",
            run_type="weekly_research",
            node_name="some_node",
            sector_team_id="technology",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=1000, output_tokens=500))

        pair = pop_metadata_for("agent_a")
        assert pair is not None
        meta, ctx = pair
        assert meta.input_tokens == 1000
        assert meta.output_tokens == 500
        assert meta.model_name == "claude-haiku-4-5"
        assert meta.run_type == "weekly_research"
        assert meta.node_name == "some_node"
        assert meta.sector_team_id == "technology"
        # cost_usd recomputed: 1000 × $1/M + 500 × $5/M = $0.001 + $0.0025 = $0.0035
        assert meta.cost_usd == pytest.approx(0.0035)

    def test_pop_metadata_clears_entry(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, pop_metadata_for,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(agent_id="agent_a", model_name_fallback="claude-haiku-4-5"):
            cb.on_llm_end(_make_modern_response(input_tokens=10, output_tokens=5))

        # First pop returns the pair; second returns None.
        first = pop_metadata_for("agent_a")
        second = pop_metadata_for("agent_a")
        assert first is not None
        assert second is None

    def test_pop_unknown_agent_returns_none(self):
        from graph.llm_cost_tracker import pop_metadata_for
        assert pop_metadata_for("never_tracked") is None

    def test_no_calls_yields_zero_token_metadata(self, patched_pricing_path):
        """Frame closes with zero calls — tokens stay 0 but metadata is still
        stashed (carries the agent_id/run_type context for the capture)."""
        from graph.llm_cost_tracker import track_llm_cost, pop_metadata_for

        with track_llm_cost(agent_id="silent_agent", model_name_fallback="claude-haiku-4-5"):
            pass  # No LLM calls.

        pair = pop_metadata_for("silent_agent")
        assert pair is not None
        meta, _ = pair
        assert meta.input_tokens == 0
        assert meta.output_tokens == 0
        assert meta.cost_usd == 0.0


# ── Multi-call accumulation (ReAct loop simulation) ───────────────────────


class TestMultiCallAccumulation:
    def test_three_calls_aggregated(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, pop_metadata_for,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(agent_id="react_agent", model_name_fallback="claude-haiku-4-5"):
            cb.on_llm_end(_make_modern_response(input_tokens=1000, output_tokens=200))
            cb.on_llm_end(_make_modern_response(input_tokens=1500, output_tokens=300))
            cb.on_llm_end(_make_modern_response(input_tokens=500, output_tokens=400))

        meta, _ = pop_metadata_for("react_agent")
        assert meta.input_tokens == 3000
        assert meta.output_tokens == 900
        # cost = (3000 + 4500) / 1M = 0.0075
        assert meta.cost_usd == pytest.approx((3000 * 1.0 + 900 * 5.0) / 1_000_000)


# ── Prompt context propagation ───────────────────────────────────────────


class TestPromptPropagation:
    def test_prompt_id_and_version_stamped(self, patched_pricing_path, tmp_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, pop_metadata_for,
        )
        from agents.prompt_loader import LoadedPrompt

        prompt = LoadedPrompt(
            name="cio_decision",
            text="You are the CIO. Decide.",
            version="2.3.0",
            hash="deadbeef",
            source_path=tmp_path / "fake.txt",
        )
        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="agent_with_prompt",
            prompt=prompt,
            model_name_fallback="claude-sonnet-4-6",
        ):
            cb.on_llm_end(_make_modern_response(
                input_tokens=100, output_tokens=50,
                model_name="claude-sonnet-4-6",
            ))

        meta, ctx = pop_metadata_for("agent_with_prompt")
        assert meta.prompt_id == "cio_decision"
        assert meta.prompt_version == "2.3.0"
        assert ctx.prompt_version_hash == "deadbeef"
        assert ctx.user_prompt == "You are the CIO. Decide."


# ── Run-type Literal enforcement ─────────────────────────────────────────


class TestRunType:
    def test_default_weekly_research(self, patched_pricing_path):
        from graph.llm_cost_tracker import track_llm_cost, pop_metadata_for

        with track_llm_cost(agent_id="default_run", model_name_fallback="claude-haiku-4-5"):
            pass
        meta, _ = pop_metadata_for("default_run")
        assert meta.run_type == "weekly_research"

    def test_explicit_morning(self, patched_pricing_path):
        from graph.llm_cost_tracker import track_llm_cost, pop_metadata_for

        with track_llm_cost(
            agent_id="morning_agent",
            run_type="morning",
            model_name_fallback="claude-haiku-4-5",
        ):
            pass
        meta, _ = pop_metadata_for("morning_agent")
        assert meta.run_type == "morning"


# ── Exception path: frame still pops when body raises ─────────────────────


class TestExceptionPath:
    def test_frame_pops_when_body_raises(self, patched_pricing_path):
        from graph.llm_cost_tracker import track_llm_cost, _frame_stack

        with pytest.raises(ValueError, match="boom"):
            with track_llm_cost(agent_id="raiser", model_name_fallback="claude-haiku-4-5"):
                raise ValueError("boom")
        assert _frame_stack.get() == []


# ── Pricing path resolution ───────────────────────────────────────────────


class TestPricingPathResolution:
    def test_uses_find_config_with_cost_subdir(self, monkeypatch, tmp_path):
        """Verify _resolve_pricing_path delegates to _find_config(subdir='cost')."""
        from graph import llm_cost_tracker

        captured: dict = {}

        def fake_find(filename, subdir="research"):
            captured["filename"] = filename
            captured["subdir"] = subdir
            yaml_path = tmp_path / "model_pricing.yaml"
            yaml_path.write_text(
                "cards:\n"
                "  - model_name: claude-haiku-4-5\n"
                "    effective_from: 2026-01-01\n"
                "    input_per_1m: 1.0\n"
                "    output_per_1m: 5.0\n"
                "    cache_read_per_1m: 0.1\n"
                "    cache_create_per_1m: 1.25\n"
            )
            return yaml_path

        monkeypatch.setattr(llm_cost_tracker, "_find_config", fake_find)
        path = llm_cost_tracker._resolve_pricing_path()
        assert captured == {"filename": "model_pricing.yaml", "subdir": "cost"}
        assert path.exists()


# ── Price-table cache ─────────────────────────────────────────────────────


class TestPriceTableCache:
    def test_loaded_once_per_process(self, monkeypatch, tmp_path):
        from graph import llm_cost_tracker

        load_count = {"n": 0}

        yaml_path = tmp_path / "model_pricing.yaml"
        yaml_path.write_text(
            "cards:\n"
            "  - model_name: claude-haiku-4-5\n"
            "    effective_from: 2026-01-01\n"
            "    input_per_1m: 1.0\n"
            "    output_per_1m: 5.0\n"
            "    cache_read_per_1m: 0.1\n"
            "    cache_create_per_1m: 1.25\n"
        )
        monkeypatch.setattr(
            llm_cost_tracker, "_resolve_pricing_path", lambda: yaml_path,
        )

        # Wrap the real loader to count calls.
        real_loader = llm_cost_tracker.load_pricing
        def counting_load(path):
            load_count["n"] += 1
            return real_loader(path)
        monkeypatch.setattr(llm_cost_tracker, "load_pricing", counting_load)

        llm_cost_tracker._reset_price_table_for_tests()
        t1 = llm_cost_tracker._load_price_table()
        t2 = llm_cost_tracker._load_price_table()
        t3 = llm_cost_tracker._load_price_table()

        assert t1 is t2 is t3
        assert load_count["n"] == 1


# ── PR 3: per-call JSONL sink ─────────────────────────────────────────────


_TEST_BUCKET = "alpha-engine-research"


@pytest.fixture
def mocked_s3():
    """Stand up a moto-mocked S3 + create the cost-raw bucket."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_TEST_BUCKET)
        yield s3


@pytest.fixture
def capture_enabled(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")


@pytest.fixture
def capture_disabled(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)


def _read_jsonl_object(s3, bucket: str, key: str) -> list[dict]:
    """Helper: read a moto-stored JSONL object back as a list of dicts."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


class TestPerCallRowAccumulation:
    def test_callback_appends_per_call_row(self, patched_pricing_path):
        """Each on_llm_end fires appends one row to frame.per_call_rows."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="test_agent",
            model_name_fallback="claude-haiku-4-5",
        ) as frame:
            cb.on_llm_end(_make_modern_response(input_tokens=100, output_tokens=50))
            cb.on_llm_end(_make_modern_response(input_tokens=200, output_tokens=80))
            assert len(frame.per_call_rows) == 2
            assert frame.per_call_rows[0]["call_seq"] == 1
            assert frame.per_call_rows[1]["call_seq"] == 2
            assert frame.per_call_rows[0]["input_tokens"] == 100
            assert frame.per_call_rows[1]["input_tokens"] == 200

    def test_per_call_row_carries_timestamp(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="test_agent",
            model_name_fallback="claude-haiku-4-5",
        ) as frame:
            cb.on_llm_end(_make_modern_response(input_tokens=10, output_tokens=5))
            ts = frame.per_call_rows[0]["timestamp"]
            assert isinstance(ts, str)
            # Round-trip-able ISO 8601 (UTC).
            from datetime import datetime
            datetime.fromisoformat(ts.replace("Z", "+00:00"))


class TestJsonlFlushHappyPath:
    def test_flush_writes_jsonl_at_canonical_key(
        self, mocked_s3, capture_enabled, patched_pricing_path,
    ):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="sector_team:technology",
            sector_team_id="technology",
            node_name="sector_team_node",
            run_type="weekly_research",
            run_id="2026-05-02",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=4000, output_tokens=1200))

        # Key should be: decision_artifacts/_cost_raw/{date}/{run_id}/{agent_id}.jsonl
        # (date from frame.enter_time = today UTC).
        listing = mocked_s3.list_objects_v2(
            Bucket=_TEST_BUCKET,
            Prefix="decision_artifacts/_cost_raw/",
        )
        keys = [obj["Key"] for obj in listing.get("Contents", [])]
        assert len(keys) == 1
        assert "/2026-05-02/sector_team:technology.jsonl" in keys[0]

        rows = _read_jsonl_object(mocked_s3, _TEST_BUCKET, keys[0])
        assert len(rows) == 1
        row = rows[0]
        # Row carries frame-level dimensions.
        assert row["agent_id"] == "sector_team:technology"
        assert row["sector_team_id"] == "technology"
        assert row["node_name"] == "sector_team_node"
        assert row["run_type"] == "weekly_research"
        assert row["run_id"] == "2026-05-02"
        # Row carries call-level data.
        assert row["call_seq"] == 1
        assert row["input_tokens"] == 4000
        assert row["output_tokens"] == 1200
        assert row["model_name"] == "claude-haiku-4-5"
        # cost_usd computed: (4000*1.0 + 1200*5.0) / 1M = 0.01
        assert row["cost_usd"] == pytest.approx(0.01)
        # Schema versioning present for future migrations.
        assert row["schema_version"] == 1

    def test_react_loop_writes_multiple_rows(
        self, mocked_s3, capture_enabled, patched_pricing_path,
    ):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="sector_quant:tech",
            run_id="2026-05-02",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=1000, output_tokens=200))
            cb.on_llm_end(_make_modern_response(input_tokens=1500, output_tokens=300))
            cb.on_llm_end(_make_modern_response(input_tokens=500, output_tokens=400))

        listing = mocked_s3.list_objects_v2(
            Bucket=_TEST_BUCKET, Prefix="decision_artifacts/_cost_raw/",
        )
        keys = [obj["Key"] for obj in listing.get("Contents", [])]
        rows = _read_jsonl_object(mocked_s3, _TEST_BUCKET, keys[0])
        assert len(rows) == 3
        assert [r["call_seq"] for r in rows] == [1, 2, 3]
        assert [r["input_tokens"] for r in rows] == [1000, 1500, 500]


class TestJsonlFlushGating:
    def test_flag_off_skips_flush(
        self, mocked_s3, capture_disabled, patched_pricing_path,
    ):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="sector_team:tech",
            run_id="2026-05-02",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=10, output_tokens=5))

        # No JSONL written.
        listing = mocked_s3.list_objects_v2(
            Bucket=_TEST_BUCKET, Prefix="decision_artifacts/_cost_raw/",
        )
        assert listing.get("Contents", []) == []

    def test_no_calls_no_flush(
        self, mocked_s3, capture_enabled, patched_pricing_path,
    ):
        """Frames with zero LLM calls don't emit empty JSONLs."""
        from graph.llm_cost_tracker import track_llm_cost

        with track_llm_cost(
            agent_id="silent_agent",
            run_id="2026-05-02",
            model_name_fallback="claude-haiku-4-5",
        ):
            pass

        listing = mocked_s3.list_objects_v2(
            Bucket=_TEST_BUCKET, Prefix="decision_artifacts/_cost_raw/",
        )
        assert listing.get("Contents", []) == []

    def test_no_run_id_skips_flush(
        self, mocked_s3, capture_enabled, patched_pricing_path,
    ):
        """Without run_id we can't compute the partition key — flush is
        skipped (in-process metadata stash still works)."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, pop_metadata_for,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="agent_no_run_id",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=10, output_tokens=5))

        # In-process metadata still landed.
        assert pop_metadata_for("agent_no_run_id") is not None
        # But no JSONL on S3.
        listing = mocked_s3.list_objects_v2(
            Bucket=_TEST_BUCKET, Prefix="decision_artifacts/_cost_raw/",
        )
        assert listing.get("Contents", []) == []


class TestJsonlFlushHardFail:
    def test_s3_put_failure_raises_cost_raw_write_error(
        self, capture_enabled, patched_pricing_path,
    ):
        """When the env flag is on AND S3 unreachable, the flush raises
        instead of silently swallowing per ``feedback_no_silent_fails``."""
        from graph.llm_cost_tracker import (
            CostRawWriteError, CostTelemetryCallback, track_llm_cost,
            _flush_cost_rows_to_s3, _Frame,
        )

        # Construct a frame with one row and call the flush helper directly
        # against a stub S3 client that always raises ClientError.
        stub = MagicMock()
        stub.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no perm"}},
            "PutObject",
        )

        from datetime import datetime, timezone
        frame = _Frame(
            agent_id="ic_cio",
            sector_team_id=None,
            node_name="cio_node",
            run_type="weekly_research",
            prompt=None,
            run_id="2026-05-02",
            enter_time=datetime(2026, 5, 2, tzinfo=timezone.utc),
            per_call_rows=[{
                "schema_version": 1,
                "timestamp": "2026-05-02T13:30:00+00:00",
                "call_seq": 1,
                "model_name": "claude-haiku-4-5",
                "input_tokens": 10, "output_tokens": 5,
                "cache_read_tokens": 0, "cache_create_tokens": 0,
            }],
        )

        with pytest.raises(CostRawWriteError, match="AccessDenied"):
            _flush_cost_rows_to_s3(frame=frame, table=None, s3_client=stub)


class TestS3KeyBuilder:
    def test_key_format(self):
        from datetime import datetime, timezone
        from graph.llm_cost_tracker import _build_cost_raw_s3_key

        key = _build_cost_raw_s3_key(
            capture_dt=datetime(2026, 5, 2, 13, 30, tzinfo=timezone.utc),
            run_id="2026-05-02",
            agent_id="sector_team:technology",
        )
        assert key == "decision_artifacts/_cost_raw/2026-05-02/2026-05-02/sector_team:technology.jsonl"


# ── PR 5: hard ceiling (RunBudgetExceededError) ──────────────────────────


@pytest.fixture(autouse=True)
def reset_run_budget_state(monkeypatch):
    """Clear per-run accumulator + restore default ceiling between tests."""
    from graph import llm_cost_tracker
    llm_cost_tracker._reset_run_cost_totals_for_tests()
    monkeypatch.delenv("ALPHA_ENGINE_RUN_BUDGET_USD", raising=False)
    yield
    llm_cost_tracker._reset_run_cost_totals_for_tests()


class TestRunBudgetCeilingResolution:
    def test_default_ceiling_is_100_usd(self):
        from graph.llm_cost_tracker import _resolve_run_budget_ceiling
        assert _resolve_run_budget_ceiling() == 100.0

    def test_env_override(self, monkeypatch):
        from graph.llm_cost_tracker import _resolve_run_budget_ceiling
        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "5.50")
        assert _resolve_run_budget_ceiling() == 5.50

    def test_zero_disables(self, monkeypatch):
        from graph.llm_cost_tracker import _resolve_run_budget_ceiling
        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "0")
        assert _resolve_run_budget_ceiling() == 0.0

    def test_negative_treated_as_disabled_at_check_site(self, monkeypatch):
        # Resolver returns the negative value as-is; the check at frame
        # exit treats `ceiling <= 0` as disabled.
        from graph.llm_cost_tracker import _resolve_run_budget_ceiling
        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "-1")
        assert _resolve_run_budget_ceiling() == -1.0

    def test_unparseable_returns_zero_with_warn(self, monkeypatch, caplog):
        from graph.llm_cost_tracker import _resolve_run_budget_ceiling
        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "not-a-number")
        with caplog.at_level("WARNING"):
            result = _resolve_run_budget_ceiling()
        assert result == 0.0
        assert any("not a number" in r.message for r in caplog.records)


class TestRunCostAccumulator:
    def test_single_frame_accumulates(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, get_run_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="agent_a", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=1_000_000, output_tokens=0))
        # 1M input × $1/M = $1.00
        assert get_run_cost("run-x") == pytest.approx(1.0)

    def test_multiple_frames_one_run_sum(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, get_run_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="agent_a", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=500_000, output_tokens=0))
        with track_llm_cost(
            agent_id="agent_b", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=300_000, output_tokens=0))
        # 800k × $1/M = $0.80
        assert get_run_cost("run-x") == pytest.approx(0.80)

    def test_separate_runs_isolated(self, patched_pricing_path):
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost, get_run_cost,
        )

        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="a", run_id="run-1",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=1_000_000, output_tokens=0))
        with track_llm_cost(
            agent_id="a", run_id="run-2",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=500_000, output_tokens=0))
        assert get_run_cost("run-1") == pytest.approx(1.0)
        assert get_run_cost("run-2") == pytest.approx(0.5)

    def test_unknown_run_id_returns_zero(self):
        from graph.llm_cost_tracker import get_run_cost
        assert get_run_cost("never-seen") == 0.0


class TestRunBudgetCeilingEnforcement:
    def test_under_ceiling_does_not_raise(self, monkeypatch, patched_pricing_path):
        """1M input tokens at $1/M = $1.00 < $5 ceiling → no raise."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "5.00")
        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="cheap_agent", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=1_000_000, output_tokens=0))
        # Frame closed cleanly.

    def test_over_ceiling_raises_run_budget_exceeded(
        self, monkeypatch, patched_pricing_path,
    ):
        """10M input tokens × $1/M = $10 > $5 ceiling → raise."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, RunBudgetExceededError, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "5.00")
        cb = CostTelemetryCallback()
        with pytest.raises(RunBudgetExceededError) as excinfo:
            with track_llm_cost(
                agent_id="runaway_agent", run_id="run-budget-test",
                model_name_fallback="claude-haiku-4-5",
            ):
                cb.on_llm_end(_make_modern_response(input_tokens=10_000_000, output_tokens=0))
        err = excinfo.value
        assert err.run_id == "run-budget-test"
        assert err.cumulative_cost_usd == pytest.approx(10.0)
        assert err.ceiling_usd == 5.0

    def test_ceiling_fires_after_cumulative_exceeds(
        self, monkeypatch, patched_pricing_path,
    ):
        """Three frames each $0.50 < $1 ceiling individually; cumulative
        $1.50 > $1 trips the ceiling on the 3rd frame's exit."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, RunBudgetExceededError, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "1.00")
        cb = CostTelemetryCallback()

        # Frame 1: $0.50, ok.
        with track_llm_cost(
            agent_id="agent_a", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=500_000, output_tokens=0))
        # Frame 2: another $0.50, total now $1.00, NOT > ceiling.
        with track_llm_cost(
            agent_id="agent_b", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=500_000, output_tokens=0))
        # Frame 3: another $0.50 pushes total to $1.50, > $1.00 → raise.
        with pytest.raises(RunBudgetExceededError):
            with track_llm_cost(
                agent_id="agent_c", run_id="run-x",
                model_name_fallback="claude-haiku-4-5",
            ):
                cb.on_llm_end(_make_modern_response(input_tokens=500_000, output_tokens=0))

    def test_ceiling_zero_disables_check(
        self, monkeypatch, patched_pricing_path,
    ):
        """ALPHA_ENGINE_RUN_BUDGET_USD=0 turns off enforcement."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "0")
        cb = CostTelemetryCallback()
        with track_llm_cost(
            agent_id="big_spender", run_id="run-x",
            model_name_fallback="claude-haiku-4-5",
        ):
            # 1B tokens — astronomical cost; but ceiling=0 disables.
            cb.on_llm_end(_make_modern_response(input_tokens=1_000_000_000, output_tokens=0))
        # No raise.

    def test_no_run_id_skips_ceiling(self, monkeypatch, patched_pricing_path):
        """Frames without run_id can't accumulate per-run cost; ceiling
        check is skipped (the diagnostic cost-on-frame is still computed
        and stashed)."""
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "0.01")  # tiny
        cb = CostTelemetryCallback()
        # No run_id, no per-run accumulation, no ceiling check.
        with track_llm_cost(
            agent_id="anonymous", model_name_fallback="claude-haiku-4-5",
        ):
            cb.on_llm_end(_make_modern_response(input_tokens=10_000_000, output_tokens=0))
        # No raise even though frame cost ($10) >> ceiling ($0.01).

    def test_jsonl_flushes_before_ceiling_raise(
        self, monkeypatch, patched_pricing_path,
    ):
        """When the ceiling fires, the JSONL flush must complete first
        so operators can diagnose the offending calls on S3."""
        import boto3
        from moto import mock_aws
        from graph.llm_cost_tracker import (
            CostTelemetryCallback, RunBudgetExceededError, track_llm_cost,
        )

        monkeypatch.setenv("ALPHA_ENGINE_RUN_BUDGET_USD", "1.00")
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="alpha-engine-research")

            cb = CostTelemetryCallback()
            with pytest.raises(RunBudgetExceededError):
                with track_llm_cost(
                    agent_id="big_agent", run_id="run-fail",
                    model_name_fallback="claude-haiku-4-5",
                ):
                    cb.on_llm_end(_make_modern_response(input_tokens=10_000_000, output_tokens=0))

            # Verify the JSONL landed on S3 BEFORE the raise.
            listing = s3.list_objects_v2(
                Bucket="alpha-engine-research",
                Prefix="decision_artifacts/_cost_raw/",
            )
            assert "Contents" in listing
            keys = [o["Key"] for o in listing["Contents"]]
            assert any("/run-fail/big_agent.jsonl" in k for k in keys)


class TestRunBudgetExceededErrorMessage:
    def test_includes_diagnostic_fields(self):
        from graph.llm_cost_tracker import RunBudgetExceededError
        err = RunBudgetExceededError(
            run_id="abc-123", cumulative_cost_usd=125.50, ceiling_usd=100.00,
        )
        msg = str(err)
        assert "abc-123" in msg
        assert "$125.50" in msg
        assert "$100.00" in msg
        assert "ALPHA_ENGINE_RUN_BUDGET_USD" in msg
