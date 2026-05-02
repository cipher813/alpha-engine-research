"""Tests for sector-team recursion-limit handling — the 2026-05-02 fix arc.

The Saturday SF Research step halted with ``GraphRecursionError: Recursion
limit of 16 reached`` for 5 of 6 sector teams. Two underlying issues:

1. **Budget off-by-2**: ``recursion_limit = max_iterations × 2`` was correct
   before the 2026-04-30 PR 2.3 ``response_format=...`` flip; the flip
   added one post-loop LLM call that consumed the same budget. Bump to
   ``× 2 + 2``.
2. **Crash on overrun**: budget exhaustion raised through the agent and
   crashed the whole SF instead of being a degraded-but-non-fatal outcome.
   Catch ``GraphRecursionError`` separately, return ``partial=True``,
   ``error=None`` so score_aggregator accepts it.

Tests cover:
- The recursion-limit constant calculation matches the formula above.
- Recursion errors return ``partial=True`` rather than ``error``.
- Other exceptions still flow through ``error`` (preserving hard-fail
  semantics for genuine bugs).
- ``run_sector_team`` bubbles ``partial`` up to the team-level dict.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from langgraph.errors import GraphRecursionError


@pytest.fixture
def fresh_modules():
    """Force-reload the analyst modules. ``test_dry_run.py``'s sentinel
    pattern can leave MagicMocks in place for cross-test runs in some
    pytest orders; reloading guarantees we test the real functions."""
    from agents.sector_teams import quant_analyst, qual_analyst, sector_team
    importlib.reload(quant_analyst)
    importlib.reload(qual_analyst)
    importlib.reload(sector_team)
    yield
    # No teardown — next test that needs them will reload again if it
    # also depends on freshness.


# ── Recursion-limit constants ─────────────────────────────────────────────────


def test_quant_recursion_limit_is_max_iterations_times_2_plus_2():
    """Locks the +2 budget bump that accounts for response_format
    extraction. Regressing this resurrects the 2026-05-02 SF crash."""
    from agents.sector_teams.quant_analyst import _QUANT_RECURSION_LIMIT
    from config import QUANT_MAX_ITERATIONS
    assert _QUANT_RECURSION_LIMIT == QUANT_MAX_ITERATIONS * 2 + 2


def test_qual_recursion_limit_is_max_iterations_times_2_plus_2():
    from agents.sector_teams.qual_analyst import _QUAL_RECURSION_LIMIT
    from config import QUAL_MAX_ITERATIONS
    assert _QUAL_RECURSION_LIMIT == QUAL_MAX_ITERATIONS * 2 + 2


# ── Quant analyst graceful degradation ────────────────────────────────────────


def _quant_kwargs():
    """Common args to invoke run_quant_analyst."""
    return {
        "team_id": "technology",
        "sector_tickers": ["AAPL", "MSFT"],
        "market_regime": "neutral",
        "price_data": {},
        "technical_scores": {},
        "run_date": "2026-05-02",
        "api_key": "test-key",
    }


def test_quant_analyst_returns_partial_on_recursion_error(fresh_modules):
    """The 2026-05-02 scenario: agent.invoke raises GraphRecursionError.
    Must return ``partial=True, error=None`` so score_aggregator treats
    as degraded-not-failed."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit of 18 reached"
    )

    with patch.object(_qa, "create_react_agent", return_value=fake_agent):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None, "recursion error must NOT populate error field"
    assert result["partial"] is True
    assert result["partial_reason"] == "recursion_limit_exhausted"
    assert result["ranked_picks"] == []


def test_quant_analyst_still_errors_on_other_exceptions(fresh_modules):
    """Generic exceptions (API errors, malformed JSON, etc.) must still
    flow through the ``error`` field — preserves hard-fail semantics for
    real bugs."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = RuntimeError("some other failure")

    with patch.object(_qa, "create_react_agent", return_value=fake_agent):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is not None
    assert "RuntimeError" in result["error"]
    assert result.get("partial", False) is False


# ── Qual analyst graceful degradation ─────────────────────────────────────────


def _qual_kwargs():
    return {
        "team_id": "technology",
        "quant_top5": [{"ticker": "AAPL"}],
        "prior_theses": {},
        "market_regime": "neutral",
        "run_date": "2026-05-02",
        "api_key": "test-key",
        "price_data": {},
    }


def test_qual_analyst_returns_partial_on_recursion_error(fresh_modules):
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit of 18 reached"
    )

    with patch.object(_qual, "create_react_agent", return_value=fake_agent):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is None
    assert result["partial"] is True
    assert result["partial_reason"] == "recursion_limit_exhausted"
    assert result["assessments"] == []


def test_qual_analyst_still_errors_on_other_exceptions(fresh_modules):
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = ValueError("schema validation failed")

    with patch.object(_qual, "create_react_agent", return_value=fake_agent):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is not None
    assert "ValueError" in result["error"]
    assert result.get("partial", False) is False


# ── sector_team aggregation: partial bubbles up ───────────────────────────────


def test_sector_team_aggregates_partial_from_quant(fresh_modules):
    """If quant returned partial, the team-level result must surface it
    via ``partial=True`` so score_aggregator sees it."""
    from agents.sector_teams.sector_team import _empty_result

    quant_partial = {
        "ranked_picks": [],
        "error": None,
        "partial": True,
        "partial_reason": "recursion_limit_exhausted",
    }
    result = _empty_result("technology", quant_output=quant_partial)
    assert result["partial"] is True
    assert "quant:recursion_limit_exhausted" in result["partial_reasons"]
    assert result["error"] is None


def test_sector_team_no_partial_when_quant_clean(fresh_modules):
    """Default behavior — clean quant means partial=False at the team level."""
    from agents.sector_teams.sector_team import _empty_result

    result = _empty_result("technology", quant_output={
        "ranked_picks": [], "error": None,
    })
    assert result["partial"] is False
    assert result["partial_reasons"] == []
