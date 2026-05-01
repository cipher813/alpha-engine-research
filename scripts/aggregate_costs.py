"""
Daily LLM cost aggregator — reads per-call JSONL rows from S3 + writes a
single parquet file for analytics.

Manual CLI for now (PR 3 of the cost-telemetry workstream); SF wiring
is a follow-up. Run after a weekday/Saturday SF completes once the
``decision_artifacts/_cost_raw/{date}/...`` keys have been written by
``graph/llm_cost_tracker._flush_cost_rows_to_s3``::

    python scripts/aggregate_costs.py --date 2026-05-02

Reads:    ``s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/**/*.jsonl``
Writes:   ``s3://alpha-engine-research/decision_artifacts/_cost/{date}/cost.parquet``
Prints:   total cost, breakdown by sector_team / by model / by run_type, plus
          the underlying token totals so cost can be cross-checked against
          a fresh price-table query if rates change later.

Schema posture (matches the JSONL row shape):

- ``schema_version``, ``timestamp``, ``run_id``, ``agent_id``, ``sector_team_id``,
  ``node_name``, ``run_type``, ``prompt_id``, ``prompt_version``,
  ``prompt_version_hash``, ``model_name``, ``call_seq``, ``input_tokens``,
  ``output_tokens``, ``cache_read_tokens``, ``cache_create_tokens``, ``cost_usd``.
- All additive going forward — never rename or remove a column without a
  ``schema_version`` bump per CLAUDE.md S3 contract safety rules.

Workstream design: ``alpha-engine-docs/private/ROADMAP.md`` line ~1708.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import date as date_type
from typing import Any, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"
_INPUT_PREFIX = "decision_artifacts/_cost_raw"
_OUTPUT_PREFIX = "decision_artifacts/_cost"


# ── S3 read helpers ──────────────────────────────────────────────────────


def _list_jsonl_keys(s3_client: Any, bucket: str, prefix: str) -> list[str]:
    """Return all keys under ``prefix`` ending in ``.jsonl``.

    Uses paginated ``ListObjectsV2`` so prefixes with >1000 entries are
    handled correctly. Empty prefix returns an empty list (caller should
    short-circuit).
    """
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if key.endswith(".jsonl"):
                keys.append(key)
    return keys


def _read_jsonl_rows(s3_client: Any, bucket: str, key: str) -> list[dict]:
    """Read a single JSONL object and return its parsed rows.

    Skips blank lines silently (trailing newlines from the writer are
    common and harmless). Raises if a non-blank line fails to parse —
    the JSONL writer is strict + JSON-encoding always round-trips, so
    a parse error indicates corruption worth surfacing.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    rows = []
    for i, line in enumerate(body.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSONL at s3://{bucket}/{key} line {i}: {exc}"
            ) from exc
    return rows


# ── Aggregation ──────────────────────────────────────────────────────────


def aggregate_day(
    s3_client: Any,
    bucket: str,
    target_date: date_type,
    *,
    output_key_override: Optional[str] = None,
) -> Optional[dict]:
    """Read all JSONL files for ``target_date`` and write a parquet.

    Returns a summary dict with ``rows_in``, ``rows_out``, ``output_key``,
    ``total_cost_usd``, ``by_team``, ``by_model``, ``by_run_type``,
    or ``None`` if no JSONL files were found for the date (no parquet
    written in that case — distinguished from an empty-data parquet).
    """
    date_str = target_date.isoformat()
    input_prefix = f"{_INPUT_PREFIX}/{date_str}/"
    keys = _list_jsonl_keys(s3_client, bucket, input_prefix)
    if not keys:
        logger.warning(
            "[aggregate_costs] no JSONL files found at s3://%s/%s — "
            "nothing to aggregate for %s",
            bucket, input_prefix, date_str,
        )
        return None

    logger.info(
        "[aggregate_costs] reading %d JSONL files from s3://%s/%s",
        len(keys), bucket, input_prefix,
    )
    all_rows: list[dict] = []
    for key in keys:
        all_rows.extend(_read_jsonl_rows(s3_client, bucket, key))

    if not all_rows:
        logger.warning(
            "[aggregate_costs] %d JSONL files contained zero rows — "
            "skipping parquet write",
            len(keys),
        )
        return None

    df = pd.DataFrame(all_rows)

    # Write parquet to a buffer + put_object so we don't need s3fs as a dep.
    output_key = output_key_override or f"{_OUTPUT_PREFIX}/{date_str}/cost.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    logger.info(
        "[aggregate_costs] wrote %d rows to s3://%s/%s",
        len(df), bucket, output_key,
    )

    return _build_summary(df, output_key=output_key, files_read=len(keys))


def _build_summary(df: pd.DataFrame, *, output_key: str, files_read: int) -> dict:
    """Compute drilldown breakdowns on the aggregated DataFrame.

    Hard-codes the key dimensions of interest (sector_team, model,
    run_type). Total cost + total tokens are surfaced separately so
    operators can sanity-check cost vs an expected band without needing
    to load the parquet themselves.
    """
    total_cost = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0
    total_input = int(df["input_tokens"].fillna(0).sum()) if "input_tokens" in df.columns else 0
    total_output = int(df["output_tokens"].fillna(0).sum()) if "output_tokens" in df.columns else 0
    total_cache_read = int(df["cache_read_tokens"].fillna(0).sum()) if "cache_read_tokens" in df.columns else 0
    total_cache_create = int(df["cache_create_tokens"].fillna(0).sum()) if "cache_create_tokens" in df.columns else 0

    def _group_sum(col: str) -> dict:
        if col not in df.columns or "cost_usd" not in df.columns:
            return {}
        grouped = df.groupby(col, dropna=False)["cost_usd"].sum().fillna(0)
        return {str(k): float(v) for k, v in grouped.items()}

    return {
        "rows_in": int(len(df)),
        "files_read": files_read,
        "output_key": output_key,
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_create_tokens": total_cache_create,
        "by_sector_team": _group_sum("sector_team_id"),
        "by_model": _group_sum("model_name"),
        "by_run_type": _group_sum("run_type"),
        "by_agent_id": _group_sum("agent_id"),
    }


# ── Pretty-printer ────────────────────────────────────────────────────────


def print_summary(summary: dict, *, target_date: date_type) -> None:
    """Render the summary dict as a human-readable report on stdout.

    Format matches the convention in other alpha-engine reporters
    (markdown-ish with totals first, then drilldowns). Operators copy
    this into the weekly cost-report email in PR 4.
    """
    print(f"# LLM cost report — {target_date.isoformat()}\n")
    print(f"- Files read:               {summary['files_read']}")
    print(f"- Per-call rows:            {summary['rows_in']}")
    print(f"- Output:                   s3://{_DEFAULT_BUCKET}/{summary['output_key']}")
    print(f"- Total cost:               ${summary['total_cost_usd']:.4f}")
    print(f"- Total input tokens:       {summary['total_input_tokens']:,}")
    print(f"- Total output tokens:      {summary['total_output_tokens']:,}")
    print(f"- Total cache_read tokens:  {summary['total_cache_read_tokens']:,}")
    print(f"- Total cache_create tokens:{summary['total_cache_create_tokens']:,}")
    print()
    for label, key in (
        ("By sector team", "by_sector_team"),
        ("By model", "by_model"),
        ("By run_type", "by_run_type"),
        ("By agent_id", "by_agent_id"),
    ):
        breakdown = summary.get(key, {})
        if not breakdown:
            continue
        print(f"## {label}")
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"  {k:<32s} ${v:.4f}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-call cost JSONL files into a daily parquet.",
    )
    parser.add_argument(
        "--date", required=True,
        help="Target date in ISO format (YYYY-MM-DD) — corresponds to the "
             "decision_artifacts/_cost_raw/{date}/ partition.",
    )
    parser.add_argument(
        "--bucket", default=_DEFAULT_BUCKET,
        help=f"S3 bucket (default: {_DEFAULT_BUCKET}).",
    )
    parser.add_argument(
        "--output-key", default=None,
        help="Override the output parquet key. Default: "
             "decision_artifacts/_cost/{date}/cost.parquet.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        target_date = date_type.fromisoformat(args.date)
    except ValueError as exc:
        print(f"error: --date must be ISO YYYY-MM-DD ({exc})", file=sys.stderr)
        return 2

    s3_client = boto3.client("s3")
    summary = aggregate_day(
        s3_client, args.bucket, target_date,
        output_key_override=args.output_key,
    )
    if summary is None:
        print(f"No cost data found for {target_date.isoformat()}", file=sys.stderr)
        return 1

    if not args.quiet:
        print_summary(summary, target_date=target_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
