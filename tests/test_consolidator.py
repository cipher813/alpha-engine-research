"""
Tests for agents/consolidator.py helper functions and run_consolidator_agent.
Helper functions are pure — no mocking needed.
run_consolidator_agent mocks the Anthropic client.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.consolidator import (
    _truncate_report,
    _format_universe_summaries,
    _format_thesis_table,
    _format_candidates,
    _format_consistency_flags,
    _format_sector_ratings,
    _format_performance_summary,
    run_consolidator_agent,
)


# ── _truncate_report ──────────────────────────────────────────────────────────

class TestTruncateReport:
    def test_short_text_unchanged(self):
        text = "Short report."
        assert _truncate_report(text, max_words=10) == text

    def test_long_text_truncated(self):
        text = " ".join([f"word{i}" for i in range(100)])
        result = _truncate_report(text, max_words=5)
        assert result.endswith("...")
        assert len(result.split()) == 5  # 5 words, "..." appended to last word

    def test_exact_limit_not_truncated(self):
        text = "one two three"
        assert _truncate_report(text, max_words=3) == text

    def test_empty_string(self):
        assert _truncate_report("", max_words=10) == ""


# ── _format_universe_summaries ────────────────────────────────────────────────

class TestFormatUniverseSummaries:
    def test_tickers_appear_in_output(self):
        news = {"PLTR": "PLTR had a strong week.", "RKLB": "RKLB launched a rocket."}
        research = {"PLTR": "Analysts bullish on PLTR."}
        news_out, research_out = _format_universe_summaries(news, research)
        assert "PLTR" in news_out
        assert "RKLB" in news_out
        assert "PLTR" in research_out

    def test_empty_reports_return_none_string(self):
        news_out, research_out = _format_universe_summaries({}, {})
        assert news_out == "None."
        assert research_out == "None."

    def test_sorted_alphabetically(self):
        news = {"ZZZT": "last", "AAAA": "first"}
        news_out, _ = _format_universe_summaries(news, {})
        assert news_out.index("AAAA") < news_out.index("ZZZT")


# ── _format_thesis_table ──────────────────────────────────────────────────────

class TestFormatThesisTable:
    def _make_thesis(self, rating="BUY", score=75.0, delta=2.0, stale=False, inconsistent=False):
        return {
            "rating": rating,
            "final_score": score,
            "technical_score": 70.0,
            "news_score": 68.0,
            "research_score": 72.0,
            "score_delta": delta,
            "stale_days": 6 if stale else 0,
            "consistency_flag": 1 if inconsistent else 0,
            "thesis_summary": "Strong momentum.",
        }

    def test_header_row_present(self):
        result = _format_thesis_table({"PLTR": self._make_thesis()})
        assert "Ticker" in result
        assert "Rating" in result

    def test_ticker_in_table(self):
        result = _format_thesis_table({"PLTR": self._make_thesis()})
        assert "PLTR" in result

    def test_stale_flag_shown(self):
        result = _format_thesis_table({"PLTR": self._make_thesis(stale=True)})
        assert "⚠stale" in result

    def test_consistency_flag_shown(self):
        result = _format_thesis_table({"PLTR": self._make_thesis(inconsistent=True)})
        assert "⚠inconsistent" in result

    def test_delta_formatted_with_sign(self):
        result = _format_thesis_table({"PLTR": self._make_thesis(delta=3.0)})
        assert "+3" in result

    def test_none_delta_shows_na(self):
        t = self._make_thesis()
        t["score_delta"] = None
        result = _format_thesis_table({"PLTR": t})
        assert "N/A" in result

    def test_multiple_tickers_sorted(self):
        theses = {
            "ZZZT": self._make_thesis(),
            "AAAA": self._make_thesis(),
        }
        result = _format_thesis_table(theses)
        assert result.index("AAAA") < result.index("ZZZT")


# ── _format_candidates ────────────────────────────────────────────────────────

class TestFormatCandidates:
    def _make_candidate(self, symbol="PLTR", score=75.0, delta=2.0, status="CONTINUING"):
        return {
            "symbol": symbol,
            "score": score,
            "score_delta": delta,
            "thesis_summary": "Strong AI platform.",
            "key_catalyst": "Defense contracts",
            "key_risk": "Valuation",
            "status": status,
        }

    def test_ticker_in_output(self):
        result = _format_candidates([self._make_candidate()])
        assert "PLTR" in result

    def test_status_shown(self):
        result = _format_candidates([self._make_candidate(status="NEW_ENTRY")])
        assert "NEW_ENTRY" in result

    def test_score_shown(self):
        result = _format_candidates([self._make_candidate(score=82.0)])
        assert "82" in result

    def test_delta_with_sign(self):
        result = _format_candidates([self._make_candidate(delta=5.0)])
        assert "+5" in result

    def test_none_delta_shows_na(self):
        c = self._make_candidate()
        c["score_delta"] = None
        result = _format_candidates([c])
        assert "N/A" in result

    def test_empty_list(self):
        assert _format_candidates([]) == ""

    def test_uses_ticker_key_as_fallback(self):
        c = {"ticker": "RKLB", "score": 70.0, "score_delta": 1.0,
             "thesis_summary": "", "key_catalyst": "", "key_risk": "", "status": "NEW_ENTRY"}
        result = _format_candidates([c])
        assert "RKLB" in result


# ── _format_consistency_flags ─────────────────────────────────────────────────

class TestFormatConsistencyFlags:
    def test_no_flags_returns_none_string(self):
        theses = {"PLTR": {"consistency_flag": 0}}
        assert _format_consistency_flags(theses) == "None."

    def test_flag_ticker_appears(self):
        theses = {"PLTR": {"consistency_flag": 1}}
        result = _format_consistency_flags(theses)
        assert "PLTR" in result
        assert "inconsistency" in result

    def test_mixed_flags(self):
        theses = {
            "PLTR": {"consistency_flag": 1},
            "AAPL": {"consistency_flag": 0},
        }
        result = _format_consistency_flags(theses)
        assert "PLTR" in result
        assert "AAPL" not in result


# ── _format_sector_ratings ────────────────────────────────────────────────────

class TestFormatSectorRatings:
    def test_empty_returns_not_available(self):
        assert _format_sector_ratings({}) == "Not available."

    def test_overweight_symbol(self):
        ratings = {"Technology": {"rating": "overweight", "rationale": "AI tailwind"}}
        result = _format_sector_ratings(ratings)
        assert "▲" in result
        assert "OVERWEIGHT" in result

    def test_underweight_symbol(self):
        ratings = {"Real Estate": {"rating": "underweight", "rationale": "Rate pressure"}}
        result = _format_sector_ratings(ratings)
        assert "▼" in result
        assert "UNDERWEIGHT" in result

    def test_market_weight_symbol(self):
        ratings = {"Energy": {"rating": "market_weight", "rationale": "Balanced"}}
        result = _format_sector_ratings(ratings)
        assert "●" in result

    def test_rationale_included(self):
        ratings = {"Healthcare": {"rating": "market_weight", "rationale": "Defensive play"}}
        result = _format_sector_ratings(ratings)
        assert "Defensive play" in result

    def test_header_row_present(self):
        ratings = {"Technology": {"rating": "overweight", "rationale": "x"}}
        result = _format_sector_ratings(ratings)
        assert "Sector" in result and "Rating" in result


# ── _format_performance_summary ───────────────────────────────────────────────

class TestFormatPerformanceSummary:
    def test_empty_dict_returns_no_data(self):
        result = _format_performance_summary({})
        assert "No performance data" in result

    def test_accuracy_10d_included(self):
        result = _format_performance_summary({"accuracy_10d": 62.5, "sample_size": 8})
        assert "62" in result
        assert "10d" in result

    def test_accuracy_30d_included(self):
        result = _format_performance_summary({"accuracy_30d": 58.0, "sample_size": 5})
        assert "58" in result
        assert "30d" in result

    def test_recalibration_flag_shown(self):
        result = _format_performance_summary({
            "accuracy_10d": 40.0, "sample_size": 10, "recalibration_flag": True
        })
        assert "RECALIBRATION" in result

    def test_no_recalibration_flag_absent(self):
        result = _format_performance_summary({
            "accuracy_10d": 65.0, "sample_size": 10, "recalibration_flag": False
        })
        assert "RECALIBRATION" not in result

    def test_sample_size_shown(self):
        result = _format_performance_summary({"sample_size": 42, "accuracy_10d": 60.0})
        assert "42" in result


# ── run_consolidator_agent ────────────────────────────────────────────────────

class TestRunConsolidatorAgent:
    @patch("agents.consolidator.anthropic.Anthropic")
    def test_returns_llm_text(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="Daily report.")]

        result = run_consolidator_agent(
            run_date="2026-03-05",
            macro_report="Markets are stable.",
            universe_news_reports={"PLTR": "PLTR news summary."},
            universe_research_reports={"PLTR": "PLTR research summary."},
            candidate_full_news={},
            candidate_full_research={},
            investment_theses={},
            active_candidates=[],
            performance_summary={},
        )
        assert result == "Daily report."
        assert mock_client.messages.create.called

    @patch("agents.consolidator.anthropic.Anthropic")
    def test_sector_ratings_passed_to_prompt(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="Report.")]

        sector_ratings = {
            "Technology": {"rating": "overweight", "rationale": "AI tailwind"}
        }
        run_consolidator_agent(
            run_date="2026-03-05",
            macro_report="Rates elevated.",
            universe_news_reports={},
            universe_research_reports={},
            candidate_full_news={},
            candidate_full_research={},
            investment_theses={},
            active_candidates=[],
            performance_summary={},
            sector_ratings=sector_ratings,
        )
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "OVERWEIGHT" in prompt_text

    @patch("agents.consolidator.anthropic.Anthropic")
    def test_early_close_note_added(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value.content = [MagicMock(text="Report.")]

        run_consolidator_agent(
            run_date="2026-03-05",
            macro_report="",
            universe_news_reports={},
            universe_research_reports={},
            candidate_full_news={},
            candidate_full_research={},
            investment_theses={},
            active_candidates=[],
            performance_summary={},
            is_early_close=True,
        )
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "early-close" in prompt_text
