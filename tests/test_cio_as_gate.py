"""Tests for the CIO-as-gate change (PR feat/cio-as-gate, 2026-04-30).

Pins two related changes:

1. ``_build_signals_payload`` no longer emits ENTER for BUY-rated team recs
   that the CIO did not advance. Only CIO-advanced names (or held BUY-rated
   reaffirmations) become ENTER.

2. ``run_cio`` now applies a weekly entrant cap derived from
   ``cio.max_new_entrants`` and ``cio.min_new_entrants`` config values,
   rather than truncating purely on ``open_slots``.
"""

from agents.investment_committee.ic_cio import (
    _compute_effective_cap,
    _fallback_selection,
)


def _make_thesis(ticker, rating, score, sector="Technology"):
    return {
        "ticker": ticker,
        "rating": rating,
        "final_score": score,
        "quant_score": score,
        "qual_score": score,
        "sector": sector,
        "team_id": sector.lower(),
        "conviction": "stable",
        "bull_case": "",
    }


def test_unadvanced_buy_does_not_emit_signal():
    """A team rec with rating=BUY that the CIO did NOT advance and
    that is NOT in the population must produce no signal — the previous
    bypass branch let these leak through as ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "ABBV": _make_thesis("ABBV", "BUY", 70.0, sector="Healthcare"),
        },
        "prior_theses": {},
        "new_population": [],  # not held
        "sector_map": {"ABBV": "Healthcare"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],  # CIO did NOT advance
    }

    payload = _build_signals_payload(state)
    signals = payload.get("signals", {})

    assert "ABBV" not in signals, (
        f"Non-advanced BUY rec leaked through to signals: {signals.get('ABBV')}"
    )


def test_advanced_buy_emits_enter():
    """A team rec with rating=BUY that the CIO DID advance must emit ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "MSFT": _make_thesis("MSFT", "BUY", 75.0),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"MSFT": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": ["MSFT"],
    }

    payload = _build_signals_payload(state)
    msft = payload["signals"].get("MSFT")
    assert msft is not None
    assert msft["signal"] == "ENTER"


def test_held_buy_emits_enter_reaffirm():
    """A held BUY-rated name that was NOT advanced this week (but is already
    in population) must still emit ENTER as a reaffirm."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "AAPL": _make_thesis("AAPL", "BUY", 68.0),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "AAPL"}],  # held
        "sector_map": {"AAPL": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],  # not advanced this week — reaffirm path
    }

    payload = _build_signals_payload(state)
    aapl = payload["signals"].get("AAPL")
    assert aapl is not None
    assert aapl["signal"] == "ENTER"


def test_held_non_buy_emits_hold():
    """A held name with rating != BUY must emit HOLD."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "META": _make_thesis("META", "HOLD", 55.0, sector="Communication Services"),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "META"}],
        "sector_map": {"META": "Communication Services"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    meta = payload["signals"].get("META")
    assert meta is not None
    assert meta["signal"] == "HOLD"


def test_compute_effective_cap_population_gap_within_bounds():
    """When open_slots is within [min, max], use it directly."""
    assert _compute_effective_cap(
        open_slots=5, n_candidates=20, max_new_entrants=10, min_new_entrants=2
    ) == 5


def test_compute_effective_cap_population_gap_above_max_clamps_to_max():
    """When open_slots > max, clamp down to max."""
    assert _compute_effective_cap(
        open_slots=15, n_candidates=20, max_new_entrants=10, min_new_entrants=2
    ) == 10


def test_compute_effective_cap_population_gap_below_min_floors_at_min():
    """When open_slots < min, floor to min — exits will rotate names out."""
    assert _compute_effective_cap(
        open_slots=0, n_candidates=20, max_new_entrants=10, min_new_entrants=2
    ) == 2


def test_compute_effective_cap_floor_capped_by_n_candidates():
    """If fewer candidates exist than the configured floor, the effective
    floor is n_candidates — never demand advancing more candidates than exist."""
    assert _compute_effective_cap(
        open_slots=0, n_candidates=1, max_new_entrants=10, min_new_entrants=2
    ) == 1


def test_compute_effective_cap_zero_when_no_candidates():
    """Zero candidates → zero advances regardless of slots/bounds."""
    assert _compute_effective_cap(
        open_slots=10, n_candidates=0, max_new_entrants=10, min_new_entrants=2
    ) == 0


def test_compute_effective_cap_zero_when_max_zero():
    """A max_new_entrants of 0 produces a 0 cap (kill-switch behavior)."""
    assert _compute_effective_cap(
        open_slots=10, n_candidates=20, max_new_entrants=0, min_new_entrants=0
    ) == 0


def test_fallback_selection_truncates_at_effective_cap():
    """The fallback path (LLM failure) must also respect effective_cap."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    result = _fallback_selection(candidates, effective_cap=3)
    assert len(result["advanced_tickers"]) == 3
    # Highest combined score is T0 (90+80)/2 = 85
    assert result["advanced_tickers"] == ["T0", "T1", "T2"]


# ── Offline replay against 2026-04-24 signals fixture ───────────────────────
#
# Demonstrates that today's 27-ENTER output collapses correctly under the new
# gate behavior. signals.json from 2026-04-24 contains 21 population names
# (15 BUY-rated, 6 HOLD-rated) and 27 buy_candidates (all ENTER, all BUY).
# We reconstruct an upstream state and replay _build_signals_payload under
# three CIO-advance scenarios.

import json
from pathlib import Path

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "signals_2026-04-24.json"
)


def _state_from_fixture(advanced_tickers):
    """Reconstruct an upstream graph state from the persisted signals.json,
    then override advanced_tickers to simulate different CIO outcomes."""
    with open(FIXTURE_PATH) as f:
        signals = json.load(f)

    investment_theses = {}
    for u in signals.get("universe", []):
        ticker = u["ticker"]
        investment_theses[ticker] = {
            "ticker": ticker,
            "rating": u.get("rating", "HOLD"),
            "final_score": u.get("score"),
            "quant_score": (u.get("sub_scores") or {}).get("quant"),
            "qual_score": (u.get("sub_scores") or {}).get("qual"),
            "sector": u.get("sector", "Unknown"),
            "team_id": (u.get("sector") or "unknown").lower(),
            "conviction": u.get("conviction", "stable"),
            "bull_case": u.get("thesis_summary", ""),
        }

    pop_tickers = signals.get("population", [])
    return {
        "investment_theses": investment_theses,
        "prior_theses": {},
        "new_population": [{"ticker": t} for t in pop_tickers],
        "sector_map": {t: th["sector"] for t, th in investment_theses.items()},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": list(advanced_tickers),
    }


def _count_signals(payload):
    counts = {"ENTER": 0, "HOLD": 0}
    for sig in payload["signals"].values():
        counts[sig["signal"]] = counts.get(sig["signal"], 0) + 1
    return counts


def test_replay_2026_04_24_no_cio_advances():
    """If the CIO advanced 0 of the 27 candidates this Saturday, ENTER count
    must equal only the held BUY-rated reaffirmations — under the old code
    all 27 would still ENTER via the bypass branch."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    held_buys = sum(
        1 for t in state["new_population"]
        if state["investment_theses"].get(t["ticker"], {}).get("rating") == "BUY"
    )
    assert counts["ENTER"] == held_buys, (
        f"With 0 advances, ENTER must == held-BUY count ({held_buys}); "
        f"got {counts['ENTER']}. Bypass branch may have leaked."
    )


def test_replay_2026_04_24_all_advanced():
    """Sanity: if the CIO advanced all 27 candidates, all 27 BUYs ENTER —
    proves the new code still emits ENTER on the CIO-approved path."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    # Advance every BUY-rated ticker
    all_buy_tickers = [
        t for t, th in state["investment_theses"].items()
        if th.get("rating") == "BUY"
    ]
    state["advanced_tickers"] = all_buy_tickers
    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    assert counts["ENTER"] == len(all_buy_tickers), (
        f"With all BUYs advanced, ENTER must equal BUY count "
        f"({len(all_buy_tickers)}); got {counts['ENTER']}"
    )


def test_replay_2026_04_24_top3_advanced_caps_below_baseline():
    """Tightly capped advance: CIO advances only top 3 net-new BUYs.
    The total ENTER count must drop below the 27-ENTER baseline that the
    bypass branch would have produced when none of those 3 are reaffirms."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    pop_tickers = {p["ticker"] for p in state["new_population"]}

    new_buy_candidates = [
        (t, th.get("final_score") or 0)
        for t, th in state["investment_theses"].items()
        if th.get("rating") == "BUY" and t not in pop_tickers
    ]
    new_buy_candidates.sort(key=lambda x: x[1], reverse=True)
    cap = 3
    state["advanced_tickers"] = [t for t, _ in new_buy_candidates[:cap]]

    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    held_buys = sum(
        1 for p in state["new_population"]
        if state["investment_theses"].get(p["ticker"], {}).get("rating") == "BUY"
    )
    expected_enters = held_buys + min(cap, len(new_buy_candidates))
    assert counts["ENTER"] == expected_enters
    # Must be strictly fewer than the original 27 — bypass branch is gone.
    assert counts["ENTER"] < 27, (
        f"Capped at top-{cap}, ENTER count must be < 27 baseline; got {counts['ENTER']}"
    )
