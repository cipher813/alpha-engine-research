"""LLM-as-judge orchestrator (PR 3b of ROADMAP P3.1, Phase 2 P1).

Fans the judge module out over every captured DecisionArtifact for a
given date partition, applies two-tier sampling (Haiku default + Sonnet
escalation), and persists results to S3.

Two-tier sampling logic (per ROADMAP §1626):

  1. **Haiku for cost on every weekly run.** Every artifact whose
     ``agent_id`` resolves to a rubric is scored with Haiku.

  2. **Sonnet for nuance on a sampled subset.** A Sonnet pass also
     runs for any artifact when *either* of these holds:
       - ``force_sonnet_pass`` was passed in by the caller (used by
         the Saturday SF every 4th run — the run-frequency cadence is
         a SF concern, not a Lambda concern, so this flag is the
         contract surface).
       - The Haiku eval flagged a dimension score below
         ``haiku_escalate_threshold`` (default 3) — Haiku itself said
         the artifact has a concerning gap; Sonnet's nuance is worth
         the cost to confirm or refute.

Per-artifact escalation (rather than batch-level "if any artifact's
Haiku score < 3, re-run all of them with Sonnet") is the deliberate
choice — only the borderline ones get the expensive pass, which keeps
weekly judging cost bounded while preserving diagnostic depth where
it matters.

Eval is observability, NOT a gate. Errors during evaluation of any
single artifact are logged loudly and accumulated in the result
dict's ``failed`` list — the run continues so other artifacts still
get scored. Callers (the Lambda handler) decide whether a non-empty
``failed`` warrants alarming.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import boto3

from alpha_engine_lib.decision_capture import DecisionArtifact

from evals.judge import (
    evaluate_artifact,
    persist_eval_artifact,
    resolve_rubric_for_agent,
)
from evals.metrics import emit_eval_metric
from graph.state_schemas import RubricEvalArtifact

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────


DEFAULT_HAIKU_MODEL = "claude-haiku-4-5"
"""Cost-tier judge — runs on every artifact every weekly run."""

DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"
"""Nuance-tier judge — runs on the sampled subset (force_sonnet_pass
or Haiku-flagged borderline)."""

DEFAULT_HAIKU_ESCALATE_THRESHOLD = 3
"""Any Haiku dimension score strictly below this value escalates the
artifact to a Sonnet pass. 3 is the rubric midpoint — below 3 means
Haiku flagged a real problem, not just an average dimension."""

_BUCKET_DEFAULT = "alpha-engine-research"


# ── Sampling decision ─────────────────────────────────────────────────────


def should_escalate_to_sonnet(
    haiku_eval: RubricEvalArtifact,
    *,
    threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
) -> bool:
    """Per-artifact escalation: True iff any Haiku dimension score is
    below ``threshold``."""
    return any(d.score < threshold for d in haiku_eval.dimension_scores)


# ── Capture-corpus listing ────────────────────────────────────────────────


def _build_capture_prefix(date: str) -> str:
    """``decision_artifacts/{Y}/{M}/{D}/`` — partition layout that
    ``alpha_engine_lib.decision_capture`` writes to."""
    y, m, d = date.split("-")
    return f"decision_artifacts/{y}/{m}/{d}/"


def list_capture_keys(s3: Any, *, date: str, bucket: str) -> list[str]:
    """Enumerate every captured artifact key under the date partition.

    Excludes the ``_eval/`` subtree — those are eval artifacts (output
    of this very orchestrator), not capture artifacts that need scoring.
    Excludes any keys not ending in ``.json`` (defensive — the partition
    should only contain captures, but a stray prefix shouldn't crash
    the run).
    """
    prefix = _build_capture_prefix(date)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/_eval/" in key or not key.endswith(".json"):
                continue
            keys.append(key)
    return keys


def _load_capture_artifact(
    s3: Any, *, key: str, bucket: str,
) -> DecisionArtifact:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return DecisionArtifact(**json.loads(raw))


# ── Orchestration ─────────────────────────────────────────────────────────


def evaluate_corpus(
    *,
    date: str,
    bucket: str = _BUCKET_DEFAULT,
    haiku_model: str = DEFAULT_HAIKU_MODEL,
    sonnet_model: str = DEFAULT_SONNET_MODEL,
    force_sonnet_pass: bool = False,
    haiku_escalate_threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Score every captured artifact under ``date`` per the two-tier
    sampling policy. Returns a summary dict suitable for SF inspection.

    Hard-fails on listing errors (bucket missing, S3 unreachable). Per
    artifact: a load / eval / persist error is logged + appended to
    ``failed`` and the run continues with the next artifact. Eval is
    observability — one rubric or LLM hiccup must not silently halt
    every other agent's eval.

    CloudWatch metric emission (PR 4a, ROADMAP §1634): each persisted
    eval also pushes one ``AlphaEngine/Eval/agent_quality_score``
    datapoint per rubric dimension. Metric write failures are
    observability OF observability — they're caught + counted in
    ``summary['metric_emission_failures']`` but never halt the run.
    Set ``emit_metrics=False`` to disable in tests / local replay.
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (boto3.client("cloudwatch") if emit_metrics else None)
    capture_keys = list_capture_keys(s3, date=date, bucket=bucket)

    haiku_evaluated = 0
    sonnet_evaluated = 0
    skipped_unmapped = 0
    metric_emission_failures = 0
    failed: list[dict[str, str]] = []
    persisted_keys: list[str] = []

    def _try_emit(eval_artifact: RubricEvalArtifact) -> None:
        nonlocal metric_emission_failures
        if not emit_metrics:
            return
        try:
            emit_eval_metric(eval_artifact, cloudwatch_client=cw)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] cloudwatch emit failed for "
                "agent_id=%s judge=%s",
                eval_artifact.judged_agent_id, eval_artifact.judge_model,
            )
            metric_emission_failures += 1

    logger.info(
        "[eval_orchestrator] start date=%s bucket=%s capture_keys=%d "
        "haiku_model=%s sonnet_model=%s force_sonnet=%s threshold=%d",
        date, bucket, len(capture_keys), haiku_model, sonnet_model,
        force_sonnet_pass, haiku_escalate_threshold,
    )

    for key in capture_keys:
        try:
            artifact = _load_capture_artifact(s3, key=key, bucket=bucket)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[eval_orchestrator] load failed for %s", key)
            failed.append({"key": key, "agent_id": "<unknown>", "stage": "load", "error": str(exc)})
            continue

        rubric = resolve_rubric_for_agent(artifact.agent_id)
        if rubric is None:
            skipped_unmapped += 1
            continue

        # Haiku tier — every mapped artifact every run.
        try:
            haiku_eval = evaluate_artifact(
                artifact, judge_model=haiku_model, judged_artifact_s3_key=key,
            )
            haiku_persisted_key = persist_eval_artifact(
                haiku_eval, s3_client=s3, bucket=bucket,
            )
            haiku_evaluated += 1
            persisted_keys.append(haiku_persisted_key)
            _try_emit(haiku_eval)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] haiku eval failed for %s (%s)",
                key, artifact.agent_id,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "haiku", "error": str(exc),
            })
            # Skip the Sonnet escalation if Haiku itself failed —
            # there's no haiku_eval to inspect for the threshold gate.
            continue

        # Sonnet tier — sampled subset.
        escalate = force_sonnet_pass or should_escalate_to_sonnet(
            haiku_eval, threshold=haiku_escalate_threshold,
        )
        if not escalate:
            continue

        try:
            sonnet_eval = evaluate_artifact(
                artifact, judge_model=sonnet_model, judged_artifact_s3_key=key,
            )
            sonnet_persisted_key = persist_eval_artifact(
                sonnet_eval, s3_client=s3, bucket=bucket,
            )
            sonnet_evaluated += 1
            persisted_keys.append(sonnet_persisted_key)
            _try_emit(sonnet_eval)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] sonnet eval failed for %s (%s)",
                key, artifact.agent_id,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "sonnet", "error": str(exc),
            })

    logger.info(
        "[eval_orchestrator] done date=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d failed=%d metric_emission_failures=%d",
        date, haiku_evaluated, sonnet_evaluated, skipped_unmapped,
        len(failed), metric_emission_failures,
    )

    return {
        "date": date,
        "capture_keys_total": len(capture_keys),
        "haiku_evaluated": haiku_evaluated,
        "sonnet_evaluated": sonnet_evaluated,
        "skipped_unmapped": skipped_unmapped,
        "metric_emission_failures": metric_emission_failures,
        "failed": failed,
        "persisted_keys": persisted_keys,
        "haiku_model": haiku_model,
        "sonnet_model": sonnet_model,
        "force_sonnet_pass": force_sonnet_pass,
    }
