"""
LLM cost telemetry tracker — wires Anthropic SDK token counts into the
``DecisionArtifact`` capture stream.

This is the consumer side of the per-run LLM cost telemetry workstream
(ROADMAP P1). The price-table primitive lives in ``alpha_engine_lib.cost``
(v0.2.3+); the capture wrapper lives in ``alpha_engine_lib.decision_capture``.
This module is the glue: a LangChain callback handler that pulls
``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens`` /
``cache_creation_input_tokens`` from every ``ChatAnthropic`` response, plus
a context manager that scopes the accumulation to one logical agent
decision (one node — sector_team, macro_economist, or ic_cio).

Why aggregate per-node rather than per-call:

- The existing decision-capture surface writes ONE artifact per node
  boundary in ``research_graph.py``. A sector_team node fires up to four
  LLM calls (quant ReAct → qual ReAct → peer_review quant addition →
  peer_review joint finalization); summing them yields the total cost
  attributed to that team's decision.
- Per-call granularity (one row per Anthropic API call) is the PR 3
  parquet aggregator's concern. It can either read these artifacts and
  pull the aggregated cost, or walk a side-stream we add later. Out of
  scope here.

Usage pattern (one ChatAnthropic instance, one agent decision)::

    from graph.llm_cost_tracker import (
        get_cost_telemetry_callback, track_llm_cost,
    )
    from agents.prompt_loader import load_prompt

    cb = get_cost_telemetry_callback()
    llm = ChatAnthropic(model=PER_STOCK_MODEL, callbacks=[cb], ...)

    user_prompt = load_prompt("cio_decision")
    with track_llm_cost(
        agent_id="ic_cio",
        node_name="cio_node",
        prompt=user_prompt,
        run_type="weekly_research",
    ):
        result = llm.with_structured_output(CIORawOutput).invoke([msg])

    # later in cio_node, _capture_if_enabled reads the populated metadata:
    metadata = pop_metadata_for("ic_cio")  # ModelMetadata, fully populated

**Hard-fail surface (per ``feedback_no_silent_fails``):**

- ``track_llm_cost`` enter/exit must be balanced — RuntimeError on stack
  underflow.
- The callback hard-fails if ``on_llm_end`` fires with no usage metadata
  on the response. Anthropic responses always carry usage; missing
  fields are an upstream change worth surfacing immediately.
- Price-table lookup hard-fails on unknown model (delegated to
  ``alpha_engine_lib.cost.recompute_cost``).

Thread + async safety: state lives in ``ContextVar`` so it's per-task in
asyncio, per-thread otherwise. ``contextvars.copy_context()`` semantics
take care of LangGraph fan-out propagation.

Workstream design: ``alpha-engine-docs/private/ROADMAP.md`` line ~1708.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from alpha_engine_lib.cost import (
    PriceTable,
    PriceTableLoadError,
    load_pricing,
    recompute_cost,
)
from alpha_engine_lib.decision_capture import (
    FullPromptContext,
    ModelMetadata,
)
from langchain_core.callbacks import BaseCallbackHandler

from agents.prompt_loader import LoadedPrompt
from config import _find_config

logger = logging.getLogger(__name__)


# ── Pricing table loader (cached at module level) ────────────────────────


_PRICING_FILENAME = "model_pricing.yaml"
_PRICING_SUBDIR = "cost"
_price_table: Optional[PriceTable] = None


def _resolve_pricing_path() -> Path:
    """Locate ``cost/model_pricing.yaml`` via the same search order as
    other config files (sibling clone → parent → $GITHUB_WORKSPACE →
    Lambda-staged ``config/``). Uses ``subdir="cost"`` so the file lives
    alongside the existing ``research/`` and ``predictor/`` subtrees in
    the alpha-engine-config repo.
    """
    return _find_config(_PRICING_FILENAME, subdir=_PRICING_SUBDIR)


def _load_price_table() -> PriceTable:
    """Load + cache the price table on first use.

    Hard-fails on missing / malformed pricing yaml. The cached table is
    a module-level singleton — pricing changes require a Lambda redeploy
    or process restart, which matches the cadence at which Anthropic
    publishes new rates.
    """
    global _price_table
    if _price_table is None:
        path = _resolve_pricing_path()
        _price_table = load_pricing(path)
        logger.info(
            "[cost_tracker] loaded price table from %s (%d cards)",
            path, len(_price_table.cards),
        )
    return _price_table


def _reset_price_table_for_tests() -> None:
    """Clear the cached price table — exposed for tests that swap yaml
    fixtures between cases. Not used in production code paths.
    """
    global _price_table
    _price_table = None


# ── Per-frame accumulator (one frame = one ``track_llm_cost`` scope) ─────


RunType = Literal["weekly_research", "morning", "EOD"]


@dataclass
class _Frame:
    """One agent decision's accumulating LLM-cost surface.

    Multiple LLM calls fire within a single ``track_llm_cost`` scope
    (especially for ReAct loops); this frame sums their token counts.
    The model_name is captured from the FIRST call's response and
    locked — mixed-model frames (e.g. agent retries on a different
    model) would corrupt cost recompute, so we surface that explicitly.
    """

    agent_id: str
    sector_team_id: Optional[str]
    node_name: Optional[str]
    run_type: Optional[RunType]
    prompt: Optional[LoadedPrompt]

    model_name: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    call_count: int = 0
    enter_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Stack of frames — supports nested ``track_llm_cost`` (rare, but legal).
_frame_stack: contextvars.ContextVar[list[_Frame]] = contextvars.ContextVar(
    "alpha_engine_cost_tracker_frames", default=[],
)

# Populated metadata, keyed by agent_id — read by ``_capture_if_enabled``
# at the node boundary right after ``track_llm_cost`` exits.
_completed_metadata: contextvars.ContextVar[
    dict[str, tuple[ModelMetadata, FullPromptContext]]
] = contextvars.ContextVar(
    "alpha_engine_cost_tracker_completed", default={},
)


def _current_frame() -> Optional[_Frame]:
    stack = _frame_stack.get()
    return stack[-1] if stack else None


# ── LangChain callback handler ────────────────────────────────────────────


class CostTelemetryCallback(BaseCallbackHandler):
    """Sums tokens from every Anthropic LLM call into the active frame.

    LangChain fires ``on_llm_end(response, run_id, parent_run_id, **kwargs)``
    once per chat-model API call. For ReAct agents, this fires multiple
    times per ``agent.invoke()`` (once per ReAct iteration).

    The handler reads usage from the response in this priority order:

    1. ``response.generations[0][0].message.usage_metadata`` — the modern
       AIMessage shape exposed by langchain-core ≥0.2 / langchain-anthropic
       ≥0.2. Carries ``input_tokens``, ``output_tokens``, and the optional
       ``input_token_details`` sub-dict with ``cache_read`` + ``cache_creation``.
    2. ``response.llm_output["token_usage"]`` — legacy fallback shape
       from older langchain-anthropic. No cache fields available; logs
       a warning if hit because we lose cache visibility.

    No usage on the response → ``RuntimeError`` per ``feedback_no_silent_fails``.
    Anthropic always returns usage; a missing field signals an SDK shape
    change worth investigating.
    """

    def on_llm_end(self, response, **kwargs: Any) -> None:  # type: ignore[override]
        frame = _current_frame()
        if frame is None:
            # No active scope — call happened outside ``track_llm_cost``.
            # Could be a non-research code path or a misuse; log + skip.
            # We don't raise here because the callback is attached at
            # ChatAnthropic construction time and may outlive the scope.
            logger.debug(
                "[cost_tracker] on_llm_end fired with no active frame — "
                "call accumulation skipped",
            )
            return

        usage = self._extract_usage(response)
        frame.input_tokens += usage["input_tokens"]
        frame.output_tokens += usage["output_tokens"]
        frame.cache_read_tokens += usage["cache_read_tokens"]
        frame.cache_create_tokens += usage["cache_create_tokens"]
        frame.call_count += 1

        # Lock model_name on first call. ChatAnthropic doesn't expose
        # the configured model on the response message in every shape,
        # so we pull from the AIMessage's ``response_metadata.model_name``
        # if available, else from llm_output's ``model``.
        if frame.model_name is None:
            frame.model_name = self._extract_model_name(response)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract token counts from an LLMResult.

        Returns a dict with the four fields populated. Hard-fails if the
        response carries no usage information at all.
        """
        # Modern path: AIMessage.usage_metadata.
        try:
            generations = response.generations
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                if message is not None and getattr(message, "usage_metadata", None):
                    um = message.usage_metadata
                    details = um.get("input_token_details") or {}
                    return {
                        "input_tokens": int(um.get("input_tokens", 0)),
                        "output_tokens": int(um.get("output_tokens", 0)),
                        "cache_read_tokens": int(details.get("cache_read", 0)),
                        "cache_create_tokens": int(details.get("cache_creation", 0)),
                    }
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

        # Legacy path: response.llm_output["token_usage"]. No cache fields.
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if token_usage:
            logger.warning(
                "[cost_tracker] response carried legacy token_usage shape — "
                "cache_read + cache_create token counts are not available; "
                "consider upgrading langchain-anthropic for cache visibility",
            )
            return {
                "input_tokens": int(token_usage.get("input_tokens", 0)),
                "output_tokens": int(token_usage.get("output_tokens", 0)),
                "cache_read_tokens": 0,
                "cache_create_tokens": 0,
            }

        raise RuntimeError(
            "[cost_tracker] LLM response carried no usage metadata; "
            "Anthropic responses always include usage — this indicates an "
            "SDK shape change worth investigating. "
            "Per feedback_no_silent_fails, refusing to record a 0-token call."
        )

    @staticmethod
    def _extract_model_name(response: Any) -> Optional[str]:
        """Best-effort model-name extraction from the response.

        Returns None if the response shape doesn't expose it; the frame
        falls back to the ChatAnthropic instance's configured model
        (passed at ``track_llm_cost`` exit via ``recompute_cost``'s lookup).
        """
        try:
            generations = response.generations
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                if message is not None:
                    md = getattr(message, "response_metadata", None) or {}
                    name = md.get("model_name") or md.get("model")
                    if name:
                        return str(name)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            return llm_output.get("model") or llm_output.get("model_name")
        return None


_callback_singleton: Optional[CostTelemetryCallback] = None


def get_cost_telemetry_callback() -> CostTelemetryCallback:
    """Return the process-singleton callback handler.

    Attached to every ``ChatAnthropic(callbacks=[...])`` constructor so a
    single instance accumulates across all agents. The frame stack does
    the per-decision scoping, not the callback identity.
    """
    global _callback_singleton
    if _callback_singleton is None:
        _callback_singleton = CostTelemetryCallback()
    return _callback_singleton


# ── Context manager: one frame per agent decision ────────────────────────


@contextmanager
def track_llm_cost(
    agent_id: str,
    *,
    sector_team_id: Optional[str] = None,
    node_name: Optional[str] = None,
    run_type: Optional[RunType] = "weekly_research",
    prompt: Optional[LoadedPrompt] = None,
    model_name_fallback: Optional[str] = None,
) -> Iterator[_Frame]:
    """Scope LLM token accumulation to one agent decision.

    Pushes a fresh ``_Frame`` onto the stack on enter; on exit, sums the
    accumulated tokens, recomputes USD cost via the cached price table,
    and stashes a populated ``ModelMetadata`` + ``FullPromptContext``
    keyed by ``agent_id`` for ``_capture_if_enabled`` to read.

    Parameters
    ----------
    agent_id
        Identifier matching the ``DecisionArtifact.agent_id`` field —
        e.g. ``"sector_team:technology"``, ``"macro_economist"``,
        ``"ic_cio"``.
    sector_team_id
        For sector-team agents only. Stamped onto ``ModelMetadata`` for
        cost-by-team drilldowns.
    node_name
        LangGraph node name for cost-by-node drilldowns.
    run_type
        ``weekly_research`` (default — the only research-side value) /
        ``morning`` (predictor) / ``EOD`` (executor). Future PRs in the
        cost-telemetry workstream will populate the latter two.
    prompt
        ``LoadedPrompt`` for the user-facing prompt of this decision.
        Stamps ``prompt_id`` + ``prompt_version`` on ``ModelMetadata``;
        also flows into ``FullPromptContext.user_prompt`` and
        ``prompt_version_hash``.
    model_name_fallback
        If the callback can't extract model_name from the response shape,
        use this. Should match what was passed to ``ChatAnthropic(model=...)``.
        Required for ``recompute_cost`` to look up the right card.

    Yields
    ------
    _Frame
        The active frame — useful for tests + diagnostic logging. Most
        callers ignore the yielded value.

    Raises
    ------
    RuntimeError
        If the frame stack underflows on exit (enter/exit imbalance —
        usually a bug in the calling code, e.g. swallowed exception).
        Also propagates ``PriceCardLookupError`` from ``recompute_cost``
        if the resolved model isn't in the price table.
    """
    frame = _Frame(
        agent_id=agent_id,
        sector_team_id=sector_team_id,
        node_name=node_name,
        run_type=run_type,
        prompt=prompt,
    )
    stack = _frame_stack.get()
    new_stack = stack + [frame]
    token = _frame_stack.set(new_stack)
    try:
        yield frame
    finally:
        # Pop the frame regardless of whether the body raised; we still
        # want to surface accumulated cost for partial decisions.
        current = _frame_stack.get()
        if not current or current[-1] is not frame:
            _frame_stack.reset(token)
            raise RuntimeError(
                f"[cost_tracker] frame stack imbalance for agent_id={agent_id!r}; "
                f"track_llm_cost enter/exit not properly nested",
            )
        _frame_stack.reset(token)

    # Build populated metadata + prompt context.
    model_name = frame.model_name or model_name_fallback
    if model_name is None:
        # No model_name resolved — log loudly and skip recompute. The
        # captured ModelMetadata still records token counts; cost_usd
        # stays 0 and PR 3's aggregator can recompute later if the
        # model is ever pinned to the row out-of-band. Keeping this as
        # a warning rather than a raise so a Lambda log shape change
        # doesn't take down a Sat SF run.
        logger.warning(
            "[cost_tracker] could not resolve model_name for agent_id=%s "
            "(call_count=%d); cost_usd left at 0",
            agent_id, frame.call_count,
        )
        model_name = "unknown"

    metadata = ModelMetadata(
        model_name=model_name,
        input_tokens=frame.input_tokens,
        output_tokens=frame.output_tokens,
        cache_read_tokens=frame.cache_read_tokens,
        cache_create_tokens=frame.cache_create_tokens,
        run_type=run_type,
        node_name=node_name,
        sector_team_id=sector_team_id,
        prompt_id=prompt.name if prompt else None,
        prompt_version=prompt.version if prompt else None,
    )

    # Compute cost if we have a real model and the price table is loadable.
    if model_name != "unknown":
        try:
            table = _load_price_table()
            recompute_cost(metadata, table, at=frame.enter_time)
        except (PriceTableLoadError, FileNotFoundError) as exc:
            # Pricing yaml absent or malformed — record tokens, leave
            # cost_usd at 0, log loudly. Cost recompute is a downstream
            # concern and PR 3's aggregator can re-derive from tokens.
            logger.warning(
                "[cost_tracker] price-table load failed for agent_id=%s: %s "
                "(token counts captured; cost_usd left at 0)",
                agent_id, exc,
            )

    # FullPromptContext: capture rendered user_prompt text where we have
    # it (via the LoadedPrompt). System prompts are agent-specific and
    # not always loaded by name; downstream PR can plumb them.
    prompt_context = FullPromptContext(
        system_prompt=(
            f"<system prompt for agent_id={agent_id!r} not yet plumbed; "
            f"see agents/ for the gitignored prompt template files>"
        ),
        user_prompt=prompt.text if prompt else f"<user prompt for {agent_id} not provided>",
        prompt_version_hash=prompt.hash if prompt else None,
    )

    # Stash for the next ``_capture_if_enabled`` call to read.
    completed = dict(_completed_metadata.get())
    completed[agent_id] = (metadata, prompt_context)
    _completed_metadata.set(completed)

    logger.debug(
        "[cost_tracker] frame closed: agent_id=%s model=%s tokens=%d/%d/%d/%d "
        "cost=$%.6f calls=%d prompt=%s/%s",
        agent_id, model_name,
        frame.input_tokens, frame.output_tokens,
        frame.cache_read_tokens, frame.cache_create_tokens,
        metadata.cost_usd, frame.call_count,
        prompt.name if prompt else "<none>",
        prompt.version if prompt else "<none>",
    )


def pop_metadata_for(agent_id: str) -> Optional[tuple[ModelMetadata, FullPromptContext]]:
    """Retrieve and clear the populated ``(ModelMetadata, FullPromptContext)``
    for ``agent_id`` if a recent ``track_llm_cost`` exit has stashed one.

    Returns ``None`` if no metadata is staged — the caller should fall
    back to a placeholder so capture can still write something.
    Pops the entry to keep the dict bounded under heavy fan-out.
    """
    completed = dict(_completed_metadata.get())
    pair = completed.pop(agent_id, None)
    _completed_metadata.set(completed)
    return pair
