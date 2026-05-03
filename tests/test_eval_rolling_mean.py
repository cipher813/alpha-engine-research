"""Unit tests for the rolling-4-week-mean derived metric (PR 4b).

Covers:
- ``_list_metric_combos`` — pagination over ListMetrics + dimension
  preservation.
- ``_build_metric_data_queries`` — query shape correctness.
- ``compute_and_emit_4w_mean`` end-to-end with a stubbed CloudWatch
  client: the empty-corpus first-run case, the happy path, the
  partial-no-data case, and the result-mapping path back from
  GetMetricData Ids to dimensions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


def _dims(agent: str, criterion: str, judge: str = "claude-haiku-4-5") -> list[dict]:
    return [
        {"Name": "judged_agent_id", "Value": agent},
        {"Name": "criterion", "Value": criterion},
        {"Name": "judge_model", "Value": judge},
    ]


def _make_cw_with_combos(combos: list[list[dict]], values_by_idx: dict[int, list[float]]):
    """Build a MagicMock CloudWatch client backed by the given combos +
    GetMetricData values keyed by query Id index."""
    cw = MagicMock()

    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator

    cw.get_metric_data.return_value = {
        "MetricDataResults": [
            {"Id": f"m{idx}", "Values": values_by_idx.get(idx, [])}
            for idx in range(len(combos))
        ],
    }
    return cw


# ── _list_metric_combos ───────────────────────────────────────────────────


class TestListMetricCombos:
    def test_paginated_results_aggregate(self):
        from evals.rolling_mean import _list_metric_combos

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Metrics": [{"Dimensions": _dims("a", "c1")}]},
            {"Metrics": [
                {"Dimensions": _dims("a", "c2")},
                {"Dimensions": _dims("b", "c1")},
            ]},
        ]
        cw.get_paginator.return_value = paginator

        out = _list_metric_combos(
            cw, namespace="AlphaEngine/Eval", metric_name="agent_quality_score",
        )
        assert len(out) == 3

    def test_drops_streams_with_no_dimensions(self):
        from evals.rolling_mean import _list_metric_combos

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Metrics": [
                {"Dimensions": _dims("a", "c1")},
                {"Dimensions": []},  # the no-dim aggregate emission, if any
            ]},
        ]
        cw.get_paginator.return_value = paginator

        out = _list_metric_combos(
            cw, namespace="AlphaEngine/Eval", metric_name="agent_quality_score",
        )
        assert len(out) == 1


# ── _build_metric_data_queries ────────────────────────────────────────────


class TestBuildMetricDataQueries:
    def test_one_query_per_combo_with_indexed_id(self):
        from evals.rolling_mean import _build_metric_data_queries

        combos = [_dims("a", "c1"), _dims("a", "c2"), _dims("b", "c1")]
        queries = _build_metric_data_queries(
            combos,
            namespace="AlphaEngine/Eval",
            metric_name="agent_quality_score",
            period_seconds=2419200,
        )
        assert [q["Id"] for q in queries] == ["m0", "m1", "m2"]
        # Stat is Average; Period covers the full window so we get a
        # single mean datapoint per combo.
        for q in queries:
            assert q["MetricStat"]["Stat"] == "Average"
            assert q["MetricStat"]["Period"] == 2419200
            assert q["ReturnData"] is True


# ── compute_and_emit_4w_mean ──────────────────────────────────────────────


class TestComputeAndEmit4wMean:
    def test_empty_corpus_first_run(self):
        from evals.rolling_mean import compute_and_emit_4w_mean

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Metrics": []}]
        cw.get_paginator.return_value = paginator

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 0
        assert result["datapoints_emitted"] == 0
        cw.get_metric_data.assert_not_called()
        cw.put_metric_data.assert_not_called()
        # No combos → no floor emission. The alarm correctly stays in
        # INSUFFICIENT_DATA rather than firing on a None.
        assert result["floor_value"] is None
        assert result["floor_metric_emitted"] is False

    def test_happy_path_emits_one_per_combo(self):
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("ic_cio", "decision_coherence"),
            _dims("ic_cio", "rationale_quality"),
            _dims("macro_economist", "regime_grounding"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [4.2],  # mean over the 4w window
            1: [3.8],
            2: [4.5],
        })

        end = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)
        result = compute_and_emit_4w_mean(end_time=end, cloudwatch_client=cw)

        assert result["combos_discovered"] == 3
        assert result["datapoints_emitted"] == 3
        assert result["combos_skipped_no_data"] == 0
        assert result["failed"] == []

        # Two put_metric_data calls now: per-combo batch + floor.
        assert cw.put_metric_data.call_count == 2
        per_combo_call = cw.put_metric_data.call_args_list[0]
        floor_call = cw.put_metric_data.call_args_list[1]

        # Per-combo batch
        assert per_combo_call.kwargs["Namespace"] == "AlphaEngine/Eval"
        per_combo_metrics = per_combo_call.kwargs["MetricData"]
        assert all(
            d["MetricName"] == "agent_quality_score_4w_mean"
            for d in per_combo_metrics
        )
        values_by_agent = {
            d["Dimensions"][0]["Value"] + "/" + d["Dimensions"][1]["Value"]: d["Value"]
            for d in per_combo_metrics
        }
        assert values_by_agent["ic_cio/decision_coherence"] == 4.2
        assert values_by_agent["ic_cio/rationale_quality"] == 3.8
        assert values_by_agent["macro_economist/regime_grounding"] == 4.5

        # Floor metric: single dimensionless datapoint = MIN across combos.
        floor_metrics = floor_call.kwargs["MetricData"]
        assert len(floor_metrics) == 1
        assert floor_metrics[0]["MetricName"] == "agent_quality_score_4w_mean_min"
        assert "Dimensions" not in floor_metrics[0]
        assert floor_metrics[0]["Value"] == 3.8  # min of 4.2, 3.8, 4.5
        assert result["floor_value"] == 3.8
        assert result["floor_metric_emitted"] is True

    def test_skips_combos_with_no_data_in_window(self):
        """A combo that ListMetrics returns but GetMetricData has no
        Values for (e.g., agent stopped emitting; combo first appeared
        this week with no prior data) should be counted as skipped,
        not failed."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("ic_cio", "decision_coherence"),
            _dims("ic_cio", "deprecated_dim"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [4.0],
            1: [],  # no data
        })

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 1
        assert result["combos_skipped_no_data"] == 1
        assert result["failed"] == []

    def test_window_is_28_days_ending_at_end_time(self):
        from evals.rolling_mean import compute_and_emit_4w_mean, ROLLING_WINDOW_DAYS

        combos = [_dims("a", "c1")]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.0]})

        end = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)
        compute_and_emit_4w_mean(end_time=end, cloudwatch_client=cw)

        # Inspect the GetMetricData call's StartTime/EndTime.
        kwargs = cw.get_metric_data.call_args.kwargs
        assert kwargs["EndTime"] == end
        assert kwargs["StartTime"] == end - timedelta(days=ROLLING_WINDOW_DAYS)

    def test_dimension_shape_preserved_on_derived_emission(self):
        """The derived per-combo metric must carry the SAME dimension
        shape as the source so the dashboard's per-combo trend lines
        work without further translation. (The floor metric is
        intentionally dimensionless — see
        ``test_floor_metric_is_dimensionless``.)"""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant:technology", "numerical_grounding", "claude-sonnet-4-6")]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.5]})

        compute_and_emit_4w_mean(cloudwatch_client=cw)

        # Index 0 = per-combo emission; index 1 = floor.
        per_combo_call = cw.put_metric_data.call_args_list[0]
        emitted = per_combo_call.kwargs["MetricData"][0]
        emitted_dims = {d["Name"]: d["Value"] for d in emitted["Dimensions"]}
        assert emitted_dims == {
            "judged_agent_id": "sector_quant:technology",
            "criterion": "numerical_grounding",
            "judge_model": "claude-sonnet-4-6",
        }

    def test_floor_metric_is_dimensionless(self):
        """The floor metric carries NO dimensions — that's what lets a
        single CloudWatch alarm fire on it (alarms can't use SEARCH
        to reduce across dimensions). One alarm, one stream, one
        threshold."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("a", "c1"),
            _dims("a", "c2"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.5], 1: [3.2]})

        compute_and_emit_4w_mean(cloudwatch_client=cw)

        floor_call = cw.put_metric_data.call_args_list[1]
        floor_metric = floor_call.kwargs["MetricData"][0]
        assert floor_metric["MetricName"] == "agent_quality_score_4w_mean_min"
        assert "Dimensions" not in floor_metric
        # MIN across the two combo means.
        assert floor_metric["Value"] == 3.2

    def test_floor_emitted_only_when_at_least_one_combo_had_data(self):
        """All combos return empty Values → no floor emission even
        though combos were discovered. Alarm stays in
        INSUFFICIENT_DATA rather than getting a None datapoint."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("a", "c1"), _dims("a", "c2")]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [],  # no data
            1: [],
        })

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 0
        assert result["combos_skipped_no_data"] == 2
        assert result["floor_value"] is None
        assert result["floor_metric_emitted"] is False
        # Only the combo discovery happened — no put calls at all
        # because no per-combo data AND no floor to emit.
        cw.put_metric_data.assert_not_called()

    def test_missing_query_result_recorded_as_failure(self):
        """If GetMetricData drops a query result (shouldn't happen
        in practice, but defensive), record it in failed list."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("a", "c1"), _dims("a", "c2")]
        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Metrics": [{"Dimensions": d} for d in combos]}]
        cw.get_paginator.return_value = paginator
        # Only one result returned (m0); m1 missing.
        cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "m0", "Values": [4.0]},
            ],
        }

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 1
        assert len(result["failed"]) == 1
        assert result["failed"][0]["combo_idx"] == "1"
        assert result["failed"][0]["stage"] == "get_metric_data"
