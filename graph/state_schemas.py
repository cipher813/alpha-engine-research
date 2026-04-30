"""
Pydantic schemas for typed LangGraph state — agent outputs + computed
artifacts referenced from ``ResearchState``.

These schemas are the typed-state successor to the ``dict[str, Any]`` and
``dict[str, dict]`` annotations in ``graph.research_graph.ResearchState``.
They are not yet wired into ``ResearchState`` — that happens in a subsequent
commit of the typed-state workstream once each node's read/write contract
has been verified against its actual returned shape.

**Compatibility posture for this commit (PR 1, commit 1):** every model
uses ``model_config = ConfigDict(extra="allow")`` so a model constructed
from a real agent output (which carries fields not enumerated here — e.g.
``quant_output``, ``qual_output``, ``peer_review_output`` from the
sector-team stub) does NOT reject the extras. PR 2 (where agents are
wrapped with ``with_structured_output()``) flips ``extra="forbid"`` once
the agent contracts are explicit.

Numeric constraints (e.g. ``quant_score ∈ [0, 100]``, sector modifiers
∈ [0.70, 1.30]) ARE enforced even with ``extra="allow"`` — the validators
fire on the named fields regardless of the extras policy.

Workstream context: ``~/Development/alpha-engine-docs/private/alpha-engine-
research-typed-state-capture-260429.md`` (Day-1 design doc § 3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Conviction enum (two formats — agent vs storage) ──────────────────────
# Agent format: returned by sector_quant / sector_qual / cio agents.
# Storage format: produced by ``scoring.composite.normalize_conviction()``
# and used by downstream executor + archive writes.
# ``ThesisUpdate`` accepts the union because prior_theses (loaded from
# archive) carry storage format while cio entry_theses carry agent format.
AgentConvictionLiteral = Literal["high", "medium", "low"]
StoredConvictionLiteral = Literal["rising", "stable", "declining"]
EitherConvictionLiteral = Literal[
    "high", "medium", "low", "rising", "stable", "declining"
]


# ── Atomic agent-output components ────────────────────────────────────────


class ToolCall(BaseModel):
    """One ReAct tool invocation log entry. Diagnostic; not load-bearing.

    ``tool`` is optional because peer-review-orchestration entries are
    appended to ``tool_calls`` to record the phase, but they have no
    underlying tool name (peer review is a synthesis step, not a tool
    invocation). 2026-04-30 warn-mode validation surfaced 3 such entries
    across healthcare/financials/defensives in a real Saturday SF run.
    """

    model_config = ConfigDict(extra="allow")

    tool: str | None = None
    ticker: str | None = None
    args: dict = Field(default_factory=dict)
    result_summary: str | None = None


class SectorRecommendation(BaseModel):
    """One BUY-candidate output from a sector team's quant→qual→peer chain.

    ``qual_score`` is optional because peer-review can produce
    recommendations even when the qual analyst returned zero assessments
    (logged as ``[qual:<team>] completed — 0 assessments, N tool calls``).
    2026-04-30 warn-mode validation surfaced 6 such entries (healthcare 3
    + defensives 3) in a real Saturday SF run.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    quant_score: float = Field(ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    conviction: AgentConvictionLiteral = "medium"
    quant_rationale: str = ""


class ThesisUpdate(BaseModel):
    """
    Per-stock held-position thesis update.

    All score fields are nullable: the held-stock evaluation path
    occasionally produces records missing ``final_score`` (first-time
    update, legacy archive entries predating the current schema). The
    ``score_aggregator`` recompute-or-hard-fail path (alpha-engine-research
    PR #42, 2026-04-22) handles the partial-score case by recomputing
    ``final_score`` from sub-scores when both are present, hard-failing
    only when ALL three score fields are absent.
    """

    model_config = ConfigDict(extra="allow")

    # ``ticker`` is optional because ThesisUpdate values appear as the
    # value-half of a ``dict[str, ThesisUpdate]`` mapping where the key IS
    # the ticker (e.g. ``cio.entry_theses[ticker] = thesis_dict``). The
    # cio agent and held-stock thesis_update path both rely on this
    # convention. score_aggregator's investment_theses path repeats the
    # ticker in the value too, so callers can use either shape.
    ticker: str | None = None
    final_score: float | None = Field(default=None, ge=0, le=100)
    quant_score: float | None = Field(default=None, ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    sector: str | None = None
    rating: Literal["BUY", "HOLD", "SELL"] | None = None
    conviction: EitherConvictionLiteral | None = None
    bull_case: str = ""
    bear_case: str = ""
    thesis_summary: str = ""


# ── Sector team output (one per Send fan-out branch) ──────────────────────


class SectorTeamOutput(BaseModel):
    """
    Wraps a single sector team's full output. Stored under
    ``state['sector_team_outputs'][team_id]`` after Send fan-out merges.
    """

    model_config = ConfigDict(extra="allow")

    team_id: str
    recommendations: list[SectorRecommendation] = Field(default_factory=list)
    thesis_updates: dict[str, ThesisUpdate] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = None


# ── Macro economist output ────────────────────────────────────────────────


REGIME_VALUES = ("bull", "neutral", "bear", "caution")
RegimeLiteral = Literal["bull", "neutral", "bear", "caution"]


class MacroEconomistOutput(BaseModel):
    """
    Stored as four separate keys in ``state`` (``macro_report``,
    ``sector_modifiers``, ``sector_ratings``, ``market_regime``) — this
    schema captures the contract those four fields must satisfy together.
    """

    model_config = ConfigDict(extra="allow")

    macro_report: str = ""
    sector_modifiers: dict[str, float] = Field(default_factory=dict)
    sector_ratings: dict[str, dict] = Field(default_factory=dict)
    market_regime: RegimeLiteral = "neutral"

    @field_validator("sector_modifiers")
    @classmethod
    def clamp_modifiers(cls, v: dict[str, float]) -> dict[str, float]:
        """Each per-sector modifier must lie in the macro-economist invariant
        range [0.70, 1.30] per the agent's prompt-level clamping rule."""
        for sector, m in v.items():
            if not (0.70 <= float(m) <= 1.30):
                raise ValueError(
                    f"sector_modifiers[{sector!r}]={m} outside [0.70, 1.30]"
                )
        return v


# ── Exit evaluator output ─────────────────────────────────────────────────


class ExitEvent(BaseModel):
    """One exit-from-population event."""

    model_config = ConfigDict(extra="allow")

    ticker_out: str
    reason: str = ""
    score_out: float = 0.0


class ExitEvaluatorOutput(BaseModel):
    """
    Stored as three separate keys in ``state`` (``remaining_population``,
    ``exits``, ``open_slots``) — this schema captures the contract.
    """

    model_config = ConfigDict(extra="allow")

    remaining_population: list[dict] = Field(default_factory=list)
    exits: list[ExitEvent] = Field(default_factory=list)
    open_slots: int = Field(default=0, ge=0)


# ── CIO output ────────────────────────────────────────────────────────────


class CIODecision(BaseModel):
    """One per-ticker CIO decision (ADVANCE / REJECT / HOLD).

    ``conviction`` is an integer score (0-100), aligned with what the CIO
    agent prompt actually emits — composite ranking scores like 78, 72, 25.
    2026-04-30 warn-mode validation surfaced 9 violations against the prior
    Literal['high','medium','low'] schema in a real Saturday SF run; every
    decision had a numeric conviction. Path Y of the conviction-semantics
    decision (versatility — int representation generalizes; downstream
    consumers can map to display levels via a level-helper). PR for
    producer-side alignment of SectorRecommendation/InvestmentThesis to
    int convention is the follow-up.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    thesis_type: Literal["ADVANCE", "REJECT", "HOLD"] | None = None
    rationale: str = ""
    conviction: int | None = Field(default=None, ge=0, le=100)
    score: float | None = Field(default=None, ge=0, le=100)


class CIOOutput(BaseModel):
    """
    Stored as three separate keys in ``state`` (``ic_decisions``,
    ``advanced_tickers``, ``entry_theses``).
    """

    model_config = ConfigDict(extra="allow")

    ic_decisions: list[CIODecision] = Field(default_factory=list)
    advanced_tickers: list[str] = Field(default_factory=list)
    entry_theses: dict[str, ThesisUpdate] = Field(default_factory=dict)


# ── Investment thesis (computed by score_aggregator) ──────────────────────


class InvestmentThesis(BaseModel):
    """
    Per-ticker investment thesis. Constructed by ``score_aggregator`` from
    sector_team recommendations + sector modifiers; written to S3 +
    research.db by ``archive_writer``; consumed by ``consolidator`` for
    the email body.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    sector: str = "Unknown"
    team_id: str = ""
    final_score: float = Field(ge=0, le=100)
    quant_score: float | None = Field(default=None, ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    weighted_base: float = 0.0
    macro_shift: float = 0.0
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    # InvestmentThesis is constructed by score_aggregator AFTER
    # ``normalize_conviction()`` runs, so the stored value uses the
    # executor-compatible enum (rising/stable/declining), NOT the
    # agent-input format (high/medium/low). Surfaced 2026-04-29 by
    # warn-mode validation against the original draft schema.
    conviction: StoredConvictionLiteral = "stable"
    quant_rationale: str = ""
    rating: Literal["BUY", "HOLD", "SELL"]
    score_failed: bool = False


# ── Population rotation event (entry/exit log entry) ──────────────────────


class PopulationRotationEvent(BaseModel):
    """
    One row of the population-rotation log. Mixes entry and exit shapes
    (different agents emit different field sets), so we leave the schema
    open and capture only the load-bearing fields.
    """

    model_config = ConfigDict(extra="allow")

    event_type: Literal["entry", "exit"] | None = None
    ticker: str | None = None
    ticker_in: str | None = None
    ticker_out: str | None = None
    reason: str = ""
