"""Regression tests for the 2026-04-04 unscored-BUY leak.

Pins two related fixes that prevent held-stock LLM thesis updates from
emitting ENTER signals with score=null:

1. ``_update_thesis_for_held_stock`` must merge the LLM JSON over
   ``prior_thesis`` so scoring fields (final_score, quant_score,
   qual_score, rating, sector, team_id) are preserved.
2. ``_build_signals_payload`` must downgrade BUY → HOLD when
   ``final_score`` is None, as a last line of defense if a broken
   thesis slips past step 1.

Root cause: the held-stock thesis update prompt only revises narrative
fields, but the old code replaced the entire thesis dict with the
LLM's partial JSON, stripping scores. Downstream code then emitted
BUY-rated ENTER signals with null scores (LNTH, KR, PR, HAL in
signals/2026-04-04/signals.json).
"""

from unittest.mock import MagicMock, patch

import pytest


def _fake_llm_factory(response_content: str) -> MagicMock:
    """Build a MagicMock that mimics ChatAnthropic's invoke() response."""
    fake_response = MagicMock()
    fake_response.content = response_content
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def test_held_stock_thesis_update_preserves_prior_scores():
    """LLM thesis updates for held stocks must merge over prior_thesis."""
    from agents.sector_teams import sector_team

    prior = {
        "ticker": "LNTH",
        "sector": "Healthcare",
        "team_id": "healthcare",
        "final_score": 45.0,
        "quant_score": 50.0,
        "qual_score": 40.0,
        "rating": "HOLD",
        "conviction": "stable",
        "bull_case": "old bull",
        "bear_case": "old bear",
    }

    fake_llm = _fake_llm_factory(
        '{"bull_case": "new bull narrative", '
        '"bear_case": "new bear narrative", '
        '"conviction": "declining"}'
    )

    with patch.object(sector_team, "ChatAnthropic", return_value=fake_llm), \
         patch.object(sector_team, "load_prompt") as mock_load, \
         patch.object(sector_team, "format_structured_thesis_for_prompt", return_value=""):
        mock_load.return_value.format.return_value = "prompt"
        result = sector_team._update_thesis_for_held_stock(
            ticker="LNTH",
            triggers=[],
            prior_thesis=prior,
            news_data=None,
            analyst_data=None,
            run_date="2026-04-04",
            team_id="healthcare",
            api_key="test-key",
        )

    # Scores must be preserved from prior
    assert result["final_score"] == 45.0
    assert result["quant_score"] == 50.0
    assert result["qual_score"] == 40.0
    assert result["rating"] == "HOLD"
    assert result["sector"] == "Healthcare"
    assert result["team_id"] == "healthcare"

    # Narrative fields must be updated from LLM
    assert result["bull_case"] == "new bull narrative"
    assert result["bear_case"] == "new bear narrative"
    assert result["conviction"] == "declining"

    # Run metadata must be set
    assert result["last_updated"] == "2026-04-04"
    assert result["stale_days"] == 0


def test_held_stock_thesis_update_strips_null_llm_fields():
    """LLM-provided `null` values must NOT overwrite valid prior fields.

    Observed 2026-04-11: the held-stock thesis update prompt for LNTH,
    LLY, PFE, VRTX, CME, JHG, COKE, HSY, KR returned JSON that
    explicitly included `"final_score": null`. The merge
    `{**prior_thesis, **llm_update}` let the null override the valid
    prior score, triggering the downstream `_build_signals_payload`
    downgrade-to-HOLD safety net for 9 tickers.

    Fix: strip None values from the LLM JSON BEFORE merging so prior
    scoring fields survive.
    """
    from agents.sector_teams import sector_team

    prior = {
        "ticker": "LNTH",
        "sector": "Healthcare",
        "team_id": "healthcare",
        "final_score": 62.0,
        "quant_score": 65.0,
        "qual_score": 58.0,
        "rating": "BUY",
        "conviction": "stable",
    }

    # LLM includes explicit nulls — these would have overwritten the
    # valid prior scores in the pre-fix merge, leaving final_score=None.
    fake_llm = _fake_llm_factory(
        '{"bull_case": "updated bull", '
        '"bear_case": "updated bear", '
        '"final_score": null, '
        '"quant_score": null, '
        '"rating": "BUY"}'
    )

    with patch.object(sector_team, "ChatAnthropic", return_value=fake_llm), \
         patch.object(sector_team, "load_prompt") as mock_load, \
         patch.object(sector_team, "format_structured_thesis_for_prompt", return_value=""):
        mock_load.return_value.format.return_value = "prompt"
        result = sector_team._update_thesis_for_held_stock(
            ticker="LNTH",
            triggers=[],
            prior_thesis=prior,
            news_data=None,
            analyst_data=None,
            run_date="2026-04-11",
            team_id="healthcare",
            api_key="test-key",
        )

    # Scores must be preserved from prior — null LLM values are stripped
    assert result["final_score"] == 62.0
    assert result["quant_score"] == 65.0
    assert result["qual_score"] == 58.0
    # Non-null LLM values still override
    assert result["bull_case"] == "updated bull"
    assert result["bear_case"] == "updated bear"
    # Non-null LLM rating still overrides (this is intentional — the LLM
    # is allowed to change rating narrative even for held stocks)
    assert result["rating"] == "BUY"


def test_held_stock_thesis_update_no_prior_marks_score_failed():
    """If the LLM succeeds but there's no prior thesis, mark score_failed."""
    from agents.sector_teams import sector_team

    fake_llm = _fake_llm_factory('{"bull_case": "bull", "bear_case": "bear"}')

    with patch.object(sector_team, "ChatAnthropic", return_value=fake_llm), \
         patch.object(sector_team, "load_prompt") as mock_load, \
         patch.object(sector_team, "format_structured_thesis_for_prompt", return_value=""):
        mock_load.return_value.format.return_value = "prompt"
        result = sector_team._update_thesis_for_held_stock(
            ticker="NEW",
            triggers=[],
            prior_thesis=None,
            news_data=None,
            analyst_data=None,
            run_date="2026-04-04",
            team_id="healthcare",
            api_key="test-key",
        )

    assert result.get("score_failed") is True
    assert "final_score" not in result


def test_build_signals_downgrades_buy_when_final_score_none():
    """If a thesis has rating=BUY but final_score=None, emit HOLD not ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "LNTH": {
                "ticker": "LNTH",
                "rating": "BUY",
                "final_score": None,  # ← broken thesis
                "quant_score": None,
                "qual_score": None,
                "sector": "Healthcare",
                "team_id": "healthcare",
                "conviction": "declining",
                "bull_case": "",
            }
        },
        "prior_theses": {},
        "new_population": [{"ticker": "LNTH"}],  # held
        "sector_map": {"LNTH": "Healthcare"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    # Payload has multiple keys; the modern signals dict is keyed by ticker
    signals = payload.get("signals", {})
    lnth = signals.get("LNTH")
    assert lnth is not None, f"LNTH missing from signals: {signals}"
    # Must be HOLD, not ENTER — the safety gate downgrades unscored BUYs
    assert lnth["signal"] == "HOLD", (
        f"Unscored BUY should have been downgraded to HOLD, got: {lnth}"
    )


def test_build_signals_allows_scored_buy_through():
    """A BUY with a real final_score must still be emitted as ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "NVDA": {
                "ticker": "NVDA",
                "rating": "BUY",
                "final_score": 82.5,  # ← valid
                "quant_score": 85.0,
                "qual_score": 80.0,
                "sector": "Technology",
                "team_id": "technology",
                "conviction": "rising",
                "bull_case": "AI demand",
            }
        },
        "prior_theses": {},
        "new_population": [{"ticker": "NVDA"}],  # held
        "sector_map": {"NVDA": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    signals = payload.get("signals", {})
    nvda = signals.get("NVDA")
    assert nvda is not None
    assert nvda["signal"] == "ENTER"
    assert nvda["score"] == 82.5
