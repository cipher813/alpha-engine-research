"""
LLM-as-judge evaluation pipeline (PR 2 of ROADMAP P3.1, Phase 2 P1).

Reads a captured ``DecisionArtifact``, looks up the matching rubric
prompt, sends ``(rubric, artifact_input, artifact_output)`` to a judge
LLM (Haiku for cost on every weekly run; Sonnet for nuance on a
sampled subset wired in PR 3), and persists the structured eval result
to S3.

Eval is observability, NOT a gate. Runs proceed regardless of eval
score; the eval corpus + dashboard surface quality regressions weeks
before they show up in alpha-vs-SPY.

Composes with:
- Decision-artifact capture (alpha_engine_lib.decision_capture).
- Rubric prompts in alpha-engine-config (eval_rubric_*.txt at
  version 1.0.0+, loaded via ``agents.prompt_loader.load_prompt``).
- Cost telemetry — eval LLM calls are tagged ``agent_id="eval_judge"``
  via ``track_llm_cost`` so judging cost is observable + bounded.

Out of scope (lands in PR 3+):
- SF wiring (new ``EvalJudge`` state after Research).
- Two-tier model orchestration (Haiku default, Sonnet on sampled
  subset / on Haiku score < 3).
- CloudWatch metric ``AlphaEngine/Eval/agent_quality_score`` + SNS.
- Streamlit dashboard page.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from alpha_engine_lib.decision_capture import DecisionArtifact

from config import ANTHROPIC_API_KEY, S3_BUCKET
from agents.prompt_loader import load_prompt
from graph.llm_cost_tracker import get_cost_telemetry_callback, track_llm_cost
from graph.state_schemas import (
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
"""Default judge model — Haiku for cost on every weekly run.

Sonnet (``claude-sonnet-4-6``) is used for the nuance-tier sampled
subset; orchestration logic lands in PR 3."""

DEFAULT_MAX_TOKENS = 1500
"""Token cap for the judge response. 1500 covers 4-5 dimension
entries at ~250 tokens each + overall_reasoning. Bump if specific
rubrics consistently truncate."""


# ── Agent → rubric mapping ────────────────────────────────────────────────


def resolve_rubric_for_agent(agent_id: str) -> Optional[str]:
    """Return the rubric prompt name for ``agent_id``, or ``None`` if
    the agent type is intentionally unevaluated.

    Mapping mirrors the captured agent_id taxonomy (see
    research_graph.sector_team_node + cio_node + macro_economist_node):

      sector_quant:{team_id}        → eval_rubric_sector_quant
      sector_qual:{team_id}         → eval_rubric_sector_qual
      sector_peer_review:{team_id}  → eval_rubric_sector_peer_review
      macro_economist               → eval_rubric_macro_economist
      ic_cio                        → eval_rubric_ic_cio
      thesis_update:{team}:{ticker} → None (deferred — narrower call,
                                       structured update not novel
                                       analysis; eval value lower)

    Unknown agent_ids return None so the caller can skip cleanly
    rather than crash on rubric lookup.
    """
    if agent_id.startswith("sector_quant:"):
        return "eval_rubric_sector_quant"
    if agent_id.startswith("sector_qual:"):
        return "eval_rubric_sector_qual"
    if agent_id.startswith("sector_peer_review:"):
        return "eval_rubric_sector_peer_review"
    if agent_id == "macro_economist":
        return "eval_rubric_macro_economist"
    if agent_id == "ic_cio":
        return "eval_rubric_ic_cio"
    return None


# ── Judge call ────────────────────────────────────────────────────────────


def evaluate_artifact(
    artifact: DecisionArtifact,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judged_artifact_s3_key: Optional[str] = None,
) -> RubricEvalArtifact:
    """Judge a single ``DecisionArtifact`` against its rubric.

    Resolves the rubric for ``artifact.agent_id``, renders the rubric
    prompt with the artifact's ``input_data_snapshot`` + ``agent_output``,
    and invokes the judge LLM via ``with_structured_output``. The
    returned ``RubricEvalArtifact`` carries the dimension scores plus
    metadata (rubric_id+version, judge_model, judged_agent_id) so the
    persisted eval can be re-aggregated later.

    Cost telemetry: scoped under ``agent_id="eval_judge"`` so judging
    cost is tracked separately from the agents being judged.

    Raises ``ValueError`` if no rubric is mapped for the artifact's
    agent_id — callers should pre-filter via ``resolve_rubric_for_agent``
    when iterating over a mixed batch.
    """
    rubric_name = resolve_rubric_for_agent(artifact.agent_id)
    if rubric_name is None:
        raise ValueError(
            f"No rubric mapped for agent_id={artifact.agent_id!r}. "
            f"Pre-filter with resolve_rubric_for_agent() if iterating "
            f"a mixed batch."
        )

    loaded_prompt = load_prompt(rubric_name)

    # Render with the artifact's payload. ``json.dumps(..., default=str)``
    # handles any stray types (datetimes, Decimals) that snuck into the
    # captured snapshot.
    rendered = loaded_prompt.format(
        agent_input=json.dumps(
            artifact.input_data_snapshot, indent=2, default=str,
        ),
        agent_output=json.dumps(
            artifact.agent_output, indent=2, default=str,
        ),
    )

    llm = ChatAnthropic(
        model=judge_model,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=max_tokens,
        callbacks=[get_cost_telemetry_callback()],
    )
    structured_llm = llm.with_structured_output(RubricEvalLLMOutput)

    with track_llm_cost(
        agent_id="eval_judge",
        node_name="eval_judge_node",
        run_type="weekly_research",
        prompt=loaded_prompt,
        model_name_fallback=judge_model,
        run_id=artifact.run_id,
    ):
        llm_output: RubricEvalLLMOutput = structured_llm.invoke(
            [HumanMessage(content=rendered)],
            config={"metadata": loaded_prompt.langsmith_metadata()},
        )

    return RubricEvalArtifact(
        run_id=artifact.run_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=loaded_prompt.version,
        judge_model=judge_model,
        dimension_scores=llm_output.dimension_scores,
        overall_reasoning=llm_output.overall_reasoning,
    )


# ── Persistence ───────────────────────────────────────────────────────────


def build_eval_s3_key(
    *,
    judged_agent_id: str,
    run_id: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build the canonical S3 key for an eval artifact.

    Path shape (per ROADMAP §1630):
      ``decision_artifacts/_eval/{YYYY-MM-DD}/{judged_agent_id}/{run_id}.json``

    The date partition is taken from ``timestamp`` (defaults to
    now-UTC) so multiple runs on the same calendar day cluster under
    one prefix. ``run_id`` is the filename so retries with the same
    run_id idempotently overwrite.
    """
    ts = timestamp or datetime.now(timezone.utc)
    date_partition = ts.strftime("%Y-%m-%d")
    return (
        f"decision_artifacts/_eval/{date_partition}/"
        f"{judged_agent_id}/{run_id}.json"
    )


def persist_eval_artifact(
    artifact: RubricEvalArtifact,
    *,
    s3_client: Any = None,
    bucket: str = S3_BUCKET,
) -> str:
    """Write an eval artifact to S3 and return the S3 key.

    Uses the canonical ``decision_artifacts/_eval/...`` path. Hard-fails
    on S3 errors (per ``feedback_no_silent_fails``) — callers should
    handle the exception explicitly if running in best-effort mode.

    The ``s3_client`` parameter accepts an injected client for tests;
    production passes None and the helper builds the default client.
    """
    s3 = s3_client or boto3.client("s3")
    # Re-derive timestamp from the artifact's stamped ISO-8601 so the
    # partition matches what the artifact records — not "now at write
    # time" — keeping replay paths stable.
    artifact_ts = datetime.fromisoformat(artifact.timestamp.replace("Z", "+00:00"))
    key = build_eval_s3_key(
        judged_agent_id=artifact.judged_agent_id,
        run_id=artifact.run_id,
        timestamp=artifact_ts,
    )
    body = artifact.model_dump_json(indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(
        "[eval_judge] persisted eval for agent_id=%s rubric=%s judge=%s → %s",
        artifact.judged_agent_id, artifact.rubric_id,
        artifact.judge_model, key,
    )
    return key
