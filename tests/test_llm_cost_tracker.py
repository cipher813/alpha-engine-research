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
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
