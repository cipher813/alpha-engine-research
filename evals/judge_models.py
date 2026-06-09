"""Judge-model registry — pinned request IDs + re-anchor protocol.

L4578(a) (eval SOTA hardening). The LLM-as-judge layer scores agent
outputs against rubrics; those scores are compared across weeks via the
4-week rolling mean (``evals/rolling_mean.py``) and, soon, statistical
control bands. That comparison is only valid if the *same* judge model
produced the scores being compared — an unpinned alias that Anthropic
silently repoints to a newer snapshot confounds every cross-time κ /
rolling-mean comparison with a model change nobody recorded.

This module separates three identities the judge previously collapsed
into a single ``judge_model`` string:

* ``logical_key`` — the STABLE identity used for the S3 eval-artifact
  path, the CloudWatch ``judge_model`` dimension, and the custom_id tag.
  It must NOT change when we pin to a more-precise snapshot, or the
  rolling-mean time series would reset for a non-change.
* ``request_model`` — the EXACT string sent to the Anthropic API. Pinned
  to an immutable dated snapshot wherever Anthropic publishes one, so the
  same weights run every week.
* resolved model — the ``model`` field Anthropic returns on the response,
  captured per-artifact as ``RubricEvalArtifact.judge_resolved_model``.
  The authoritative record of what actually ran, and the re-anchor
  trigger.

**Re-anchor protocol (on judge upgrade).** When ``judge_resolved_model``
changes for a given ``logical_key`` — Anthropic ships a new snapshot, or
we deliberately bump ``request_model`` below — judge scores before and
after are NOT comparable. The 4-week rolling mean and any control band
must re-baseline from the change date: treat it as a regime break, not a
quality regression. The procedure is (1) bump ``request_model`` here,
(2) record the date + old→new resolved model in EXPERIMENTS.md, (3) reset
the affected rolling-mean / control-band baselines so the alarm doesn't
fire on the discontinuity.

**Model-ID provenance.** IDs verified against the canonical Anthropic
model catalog (claude-api skill, 2026-06-09). Haiku 4.5 publishes the
dated snapshot ``claude-haiku-4-5-20251001``. Sonnet 4.6 publishes NO
dated snapshot — the alias ``claude-sonnet-4-6`` is itself the canonical
ID and appending a date returns HTTP 404 — so it cannot be pinned at
request time; its drift is caught post-hoc via the resolved-model record
instead. This asymmetry is why pinning is a per-spec property, not a
blanket "append a date suffix" rule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeModelSpec:
    """One judge model's three identities (see module docstring)."""

    logical_key: str
    """Stable identity — S3 path / CloudWatch dimension / custom_id tag.
    Never changes on a snapshot pin."""

    request_model: str
    """Exact string sent to the Anthropic API. A dated snapshot when one
    exists (``pinned=True``), otherwise the alias (``pinned=False``)."""

    tag: str
    """Compact custom_id tag (keeps the Batches API custom_id under its
    64-char ceiling)."""

    pinned: bool
    """True iff ``request_model`` is an immutable dated snapshot. False
    means no snapshot is published and the alias is the canonical ID."""

    pin_note: str
    """Why this spec is (or isn't) pinned — auditable rationale."""


HAIKU = JudgeModelSpec(
    logical_key="claude-haiku-4-5",
    request_model="claude-haiku-4-5-20251001",
    tag="h45",
    pinned=True,
    pin_note=(
        "Pinned to the dated snapshot so the judge can't silently drift "
        "when Anthropic repoints the `claude-haiku-4-5` alias to a newer "
        "snapshot. Verified current via the Anthropic model catalog "
        "(claude-api skill, 2026-06-09)."
    ),
)

SONNET = JudgeModelSpec(
    logical_key="claude-sonnet-4-6",
    request_model="claude-sonnet-4-6",
    tag="s46",
    pinned=False,
    pin_note=(
        "Anthropic publishes NO dated snapshot for Sonnet 4.6 — the alias "
        "IS the canonical ID and appending a date returns HTTP 404 — so it "
        "cannot be pinned at request time. Drift is detected post-hoc via "
        "`judge_resolved_model` + the re-anchor protocol, not prevented."
    ),
)

_SPECS: tuple[JudgeModelSpec, ...] = (HAIKU, SONNET)
_BY_LOGICAL: dict[str, JudgeModelSpec] = {s.logical_key: s for s in _SPECS}
_BY_REQUEST: dict[str, JudgeModelSpec] = {s.request_model: s for s in _SPECS}
_BY_TAG: dict[str, JudgeModelSpec] = {s.tag: s for s in _SPECS}

TAG_BY_LOGICAL: dict[str, str] = {s.logical_key: s.tag for s in _SPECS}
"""Logical-key → custom_id tag. Single source for judge.py's custom_id
codec so the tag map can't drift from this registry."""


def resolve(model: str) -> JudgeModelSpec:
    """Resolve a logical key, request ID, or tag to its ``JudgeModelSpec``.

    Accepts any of the three identities so callers can pass whatever they
    hold — the persisted ``judge_model`` logical key, an explicit request
    ID, or a custom_id tag.

    Raises ``KeyError`` for an unknown model: judge models are a closed,
    audited set, so an unrecognized string is a bug (a typo or an
    un-registered model), not something to paper over with a soft
    fallback. Fail loud per the no-silent-fails rule.
    """
    spec = (
        _BY_LOGICAL.get(model)
        or _BY_REQUEST.get(model)
        or _BY_TAG.get(model)
    )
    if spec is None:
        raise KeyError(
            f"Unknown judge model {model!r}; register it in "
            f"evals/judge_models.py (known logical keys: "
            f"{sorted(_BY_LOGICAL)}). Judge models are a closed, audited "
            f"set — an unrecognized id is a bug, not a fallback."
        )
    return spec


def request_model_for(logical_key: str) -> str:
    """Exact API request string for a logical judge-model key.

    The one indirection both judge transports (sync ``evaluate_artifact``
    and the Batches API ``build_batch_request``) route through so the
    pinned snapshot is applied in exactly one place.
    """
    return resolve(logical_key).request_model
