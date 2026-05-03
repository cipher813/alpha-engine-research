"""Unit tests for the rolling-mean Lambda handler (PR 4b)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_rolling_mean_handler.py"


def _load_handler_module():
    """Import lambda/eval_rolling_mean_handler.py without using ``lambda``
    as a package name (Python keyword)."""
    module_name = "lambda_eval_rolling_mean_handler"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod():
    mod = _load_handler_module()
    mod._init_done = False
    yield mod


def _ok_summary() -> dict:
    return {
        "combos_discovered": 12,
        "datapoints_emitted": 12,
        "combos_skipped_no_data": 0,
        "failed": [],
        "window_start": "2026-05-09T00:00:00+00:00",
        "window_end": "2026-06-06T00:00:00+00:00",
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["failed"] = [{
        "combo_idx": "5", "stage": "get_metric_data",
        "error": "missing result for query Id",
    }]
    return s


class TestHandler:
    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_ok_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["datapoints_emitted"] == 12

    def test_partial_when_any_failure(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_partial_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"
        assert len(result["summary"]["failed"]) == 1

    def test_error_when_compute_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=RuntimeError("CW throttled")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "CW throttled" in result["error"]

    def test_end_time_iso_passed_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"end_time_iso": "2026-06-06T00:00:00Z"}, context=None,
            )

        assert captured["end_time"] == datetime(
            2026, 6, 6, 0, 0, tzinfo=timezone.utc,
        )

    def test_end_time_defaults_to_none_when_unset(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        # None means rolling_mean will default to now-UTC.
        assert captured.get("end_time") is None
