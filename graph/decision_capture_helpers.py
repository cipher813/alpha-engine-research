"""
Decision-artifact capture helpers — one ``build_capture_payload(...)`` per
producer node that returns ``(input_data_snapshot, summary)`` tuples.

The actual S3 write happens in ``alpha_engine_lib.decision_capture.capture_decision``.
This module exists so each producer node has a single place to declare
"what does the agent semantically depend on?" — which fields belong in
``input_data_snapshot`` for replay correctness, and what one-line summary
captures the decision context for at-a-glance reading.

Compatibility posture: the snapshots are JSON-serializable dicts (no
``pd.DataFrame``, no Pydantic models — those get ``.to_dict()`` /
``.model_dump()`` at the boundary). The 1MB cap in ``capture_decision``
truncates pathological payloads; steady-state agent inputs are well under
the cap per the Day-1 design doc § 4 PR 3 size estimates.

**Excluded by design** from per-node snapshots: ``price_data``
(``dict[str, pd.DataFrame]``, huge — agents use the already-derived
``technical_scores`` instead). Replay paths can reconstruct ``price_data``
from ArcticDB's universe library via ``run_date`` if needed.

**Placeholder-quality fields** (refined in a follow-up commit):
- ``ModelMetadata`` token counts default to 0 — real counts come from a
  LangChain callback handler integration that lands when agents are
  upgraded to ``with_structured_output()`` (D4 in the workstream).
- ``FullPromptContext`` system_prompt + user_prompt are placeholders —
  the actual prompts live in gitignored ``agents/`` files; plumbing them
  through requires the same agent-upgrade work.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.sector_teams.sector_team import SectorTeamContext


# ── Feature flag ──────────────────────────────────────────────────────────


_DECISION_CAPTURE_ENV_VAR = "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"


def is_decision_capture_enabled() -> bool:
    """Check the env var fresh on each call (allows toggling in tests).

    Returns True iff ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` is set to
    ``"true"`` / ``"1"`` / ``"yes"`` (case-insensitive). Default off — the
    capture path stays dormant until production explicitly opts in via the
    Lambda env var (gated on IAM grant for ``s3:PutObject`` on the
    ``decision_artifacts/*`` prefix).
    """
    return os.environ.get(_DECISION_CAPTURE_ENV_VAR, "").lower() in (
        "true", "1", "yes",
    )


# ── Per-node snapshot builders ────────────────────────────────────────────


def build_sector_team_capture_payload(
    team_id: str,
    ctx: "SectorTeamContext",
    *,
    team_tickers: list[str],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for a sector_team agent invocation.

    Captures everything the sector team's quant + qual + peer-review chain
    semantically depended on EXCEPT the raw OHLCV ``price_data`` (which is
    digested into ``technical_scores`` before any LLM call sees it).
    """
    held_in_team = [t for t in ctx.held_tickers if t in set(team_tickers)]
    prior_theses_in_team = {
        t: dict(ctx.prior_theses[t]) for t in team_tickers if t in ctx.prior_theses
    }
    snapshot: dict[str, Any] = {
        "team_id": team_id,
        "run_date": ctx.run_date,
        "market_regime": ctx.market_regime,
        "scanner_universe_size": len(ctx.scanner_universe),
        "team_tickers": list(team_tickers),
        "held_tickers_in_team": held_in_team,
        # Full per-ticker payloads — the agent had access to all of these
        # at decision time. These are bounded in steady state (S&P 500+400
        # × small per-ticker dicts).
        "news_data_by_ticker": dict(ctx.news_data_by_ticker),
        "analyst_data_by_ticker": dict(ctx.analyst_data_by_ticker),
        "insider_data_by_ticker": dict(ctx.insider_data_by_ticker),
        "prior_theses_in_team": prior_theses_in_team,
        "prior_sector_ratings": dict(ctx.prior_sector_ratings),
        "current_sector_ratings": dict(ctx.current_sector_ratings),
        # technical_scores filtered to team tickers — full dict is ~900
        # entries, the team only sees ~50-200.
        "technical_scores_team": {
            t: dict(ctx.technical_scores.get(t, {})) for t in team_tickers
        },
        "memories_summary": {
            "episodic_count": len(ctx.episodic_memories),
            "semantic_count": len(ctx.semantic_memories),
        },
    }
    summary = (
        f"team_id={team_id}, run_date={ctx.run_date}, regime={ctx.market_regime}, "
        f"team_tickers={len(team_tickers)}, held_in_team={len(held_in_team)}, "
        f"news_tickers={len(ctx.news_data_by_ticker)}, "
        f"prior_theses_in_team={len(prior_theses_in_team)}"
    )
    return snapshot, summary


def build_macro_economist_capture_payload(state: dict) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the macro_economist node."""
    macro_data = state.get("macro_data", {})
    prior_macro_report = state.get("prior_macro_report", "")
    prior_snapshots = state.get("prior_macro_snapshots", []) or []
    prior_date = ""
    if prior_snapshots:
        prior_date = prior_snapshots[0].get("date", "") if isinstance(prior_snapshots[0], dict) else ""

    snapshot: dict[str, Any] = {
        "run_date": state.get("run_date"),
        "macro_data": dict(macro_data) if isinstance(macro_data, dict) else macro_data,
        "prior_macro_report": prior_macro_report,
        "prior_date": prior_date,
        "prior_snapshots_count": len(prior_snapshots),
    }
    summary = (
        f"run_date={state.get('run_date')}, "
        f"macro_data_keys={len(macro_data) if isinstance(macro_data, dict) else 0}, "
        f"prior_report_chars={len(prior_macro_report)}, "
        f"prior_snapshots={len(prior_snapshots)}"
    )
    return snapshot, summary


def build_cio_capture_payload(
    state: dict,
    *,
    candidates: list[dict],
    prior_ic: list[dict],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the cio_node."""
    macro_context = {
        "market_regime": state.get("market_regime", "neutral"),
        "macro_report": state.get("macro_report", ""),
    }
    snapshot: dict[str, Any] = {
        "run_date": state.get("run_date"),
        "candidates_count": len(candidates),
        "candidates": list(candidates),
        "macro_context": macro_context,
        "sector_ratings": dict(state.get("sector_ratings", {})),
        "remaining_population": list(state.get("remaining_population", [])),
        "remaining_population_count": len(state.get("remaining_population", [])),
        "open_slots": state.get("open_slots", 0),
        "exits": list(state.get("exits", [])),
        "prior_ic_decisions": list(prior_ic),
        "prior_ic_count": len(prior_ic),
    }
    summary = (
        f"run_date={state.get('run_date')}, candidates={len(candidates)}, "
        f"open_slots={state.get('open_slots', 0)}, "
        f"remaining_pop={len(state.get('remaining_population', []))}, "
        f"prior_ic={len(prior_ic)}, regime={state.get('market_regime', 'neutral')}"
    )
    return snapshot, summary


# ── Run ID extraction ─────────────────────────────────────────────────────


def derive_run_id(state: dict) -> str:
    """Pick a stable run_id for the current pipeline invocation.

    Today: falls back to ``run_date`` so artifacts from the same day cluster
    under one S3 key prefix per agent. When Lambda's ``aws_request_id``
    is plumbed through state (a follow-up), prefer that — it gives us
    per-invocation uniqueness so retries don't overwrite each other.
    """
    run_id = state.get("run_id")
    if run_id:
        return str(run_id)
    return str(state.get("run_date", "unknown"))
