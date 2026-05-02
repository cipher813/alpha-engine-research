"""Locks the per-candidate invariant in ``run_cio`` (added 2026-05-02
after the post-PR-D validation invoke caught Sonnet emitting an empty
``decisions`` list when the inline JSON example was stripped).

Three layers of defense in this regression class:

1. **Prompt** (alpha-engine-config #21, ic_cio_evaluation v1.2.1) — the
   ``OUTPUT REQUIREMENT`` block tells Sonnet decisions MUST contain N
   entries for N candidates. LLM-side prevention.
2. **Schema** (graph.state_schemas.CIORawOutput, ``min_length=1``) —
   empty list rejected at the SDK structured-output boundary, surfacing
   as ``parsing_error`` rather than the downstream "empty decisions"
   raise. SDK-side defense.
3. **Runtime** (this test, agents.investment_committee.ic_cio.run_cio) —
   the post-call assertion ``len(decisions_dicts) == len(candidates)``
   catches partial-list responses (decisions for 7 of 9 candidates,
   etc.) that the schema wouldn't flag. Code-side defense.

Strict mode raises with a clear "decisions for N candidates" message;
lax mode logs a WARN and falls through to ``_post_process_cio_decisions``
which tolerates a partial list by treating missing tickers as REJECT.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


def _candidate(ticker: str, score: float = 70.0) -> dict:
    return {
        "ticker": ticker,
        "team_id": "technology",
        "quant_score": score,
        "qual_score": score,
        "bull_case": "",
        "bear_case": "",
        "catalysts": [],
        "conviction": 60,
        "quant_rationale": "",
        "rr_ratio": 1.5,
    }


@pytest.fixture(autouse=True)
def _strict_mode():
    """Force STRICT_VALIDATION=true so the runtime invariant raises
    rather than falling through to the lax-mode WARN. The bug we're
    locking against produced exactly this raise on 2026-05-02."""
    prior = os.environ.get("STRICT_VALIDATION")
    os.environ["STRICT_VALIDATION"] = "true"
    yield
    if prior is None:
        del os.environ["STRICT_VALIDATION"]
    else:
        os.environ["STRICT_VALIDATION"] = prior


def _make_raw_output_with(decisions_count: int):
    """Build a ``CIORawOutput`` carrying ``decisions_count`` synthetic
    decisions. Used as the LLM's structured-output return shape."""
    from graph.state_schemas import CIORawDecision, CIORawOutput

    decisions = [
        CIORawDecision(
            ticker=f"T{i}", decision="REJECT", rank=None, conviction=30,
            rationale="synthetic test decision",
        )
        for i in range(decisions_count)
    ]
    return CIORawOutput(decisions=decisions)


def _patch_llm(monkeypatch, raw_output):
    """Patch ChatAnthropic so the LLM mock returns ``raw_output`` from
    its structured-output ``.invoke``. Bypasses the real Anthropic call
    entirely; only the post-call invariant code path is exercised."""
    from agents.investment_committee import ic_cio

    fake_llm = MagicMock()
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = raw_output
    fake_llm.with_structured_output.return_value = fake_structured

    monkeypatch.setattr(ic_cio, "ChatAnthropic", lambda **kw: fake_llm)


class TestCIOPerCandidateInvariant:
    def test_partial_decisions_raises_in_strict_mode(self, monkeypatch):
        """Sonnet returns 5 decisions for 9 candidates → strict-mode
        raise with message naming both counts. Without this assertion
        the partial list would silently flow into ``_post_process_cio_decisions``
        which would treat the 4 missing candidates as REJECT — masking
        a real LLM behavior regression."""
        from agents.investment_committee.ic_cio import run_cio

        candidates = [_candidate(f"X{i}") for i in range(9)]
        _patch_llm(monkeypatch, _make_raw_output_with(decisions_count=5))

        with pytest.raises(RuntimeError) as exc_info:
            run_cio(
                candidates=candidates,
                macro_context={"market_regime": "neutral"},
                sector_ratings={},
                current_population=[],
                open_slots=2,
                exits=[],
                run_date="2026-05-02",
                max_new_entrants=10,
                min_new_entrants=2,
            )
        msg = str(exc_info.value)
        assert "5 decisions" in msg
        assert "9 candidates" in msg

    def test_full_decisions_passes(self, monkeypatch):
        """N decisions for N candidates → no raise; downstream post-
        processing runs normally."""
        from agents.investment_committee.ic_cio import run_cio

        candidates = [_candidate(f"T{i}") for i in range(3)]
        _patch_llm(monkeypatch, _make_raw_output_with(decisions_count=3))

        result = run_cio(
            candidates=candidates,
            macro_context={"market_regime": "neutral"},
            sector_ratings={},
            current_population=[],
            open_slots=2,
            exits=[],
            run_date="2026-05-02",
            max_new_entrants=10,
            min_new_entrants=2,
        )
        # All 3 stub decisions are REJECT → 0 advanced; just locks
        # that the path didn't raise on a complete decisions list.
        assert "decisions" in result
        assert "advanced_tickers" in result

    def test_empty_decisions_handled_before_invariant(self, monkeypatch):
        """The empty-list case is caught by the existing ``not
        decisions_dicts`` branch (which raises ``CIO structured response
        had empty decisions list``) BEFORE the per-candidate invariant
        runs. Lock that the empty-list error class hasn't drifted —
        empty is its own message, not the per-candidate count message.

        Note: ``CIORawOutput`` itself rejects empty at the schema layer
        (``min_length=1``), so we patch the wrapper to bypass that and
        exercise the run_cio empty-handling branch directly."""
        from agents.investment_committee.ic_cio import run_cio
        from graph.state_schemas import CIORawOutput

        # Construct a CIORawOutput-like that bypasses min_length=1
        # validation by setting decisions on an instance constructed
        # with one item then mutating to empty post-construction
        # (mirrors what would happen if the LLM emitted [] and the SDK
        # parser somehow let it through — defense in depth).
        from graph.state_schemas import CIORawDecision
        raw = CIORawOutput(decisions=[
            CIORawDecision(ticker="X", decision="REJECT")
        ])
        # Mutate after construction to produce the run_cio-side branch.
        object.__setattr__(raw, "decisions", [])
        _patch_llm(monkeypatch, raw)

        candidates = [_candidate(f"Y{i}") for i in range(2)]
        with pytest.raises(RuntimeError) as exc_info:
            run_cio(
                candidates=candidates,
                macro_context={"market_regime": "neutral"},
                sector_ratings={},
                current_population=[],
                open_slots=1,
                exits=[],
                run_date="2026-05-02",
                max_new_entrants=10,
                min_new_entrants=2,
            )
        # Empty case has its own message — distinct from the
        # per-candidate count message. Grep-friendly distinction.
        assert "empty decisions list" in str(exc_info.value).lower()
