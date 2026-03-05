"""
Tests for graph node logic and trading day detection.
Validates scheduler logic without triggering real AWS or LLM calls.
"""

import importlib
import datetime
import pytest
from unittest.mock import patch, MagicMock

# 'lambda' is a reserved keyword; use importlib to load lambda.handler
_lambda_handler = importlib.import_module("lambda.handler")


class TestTradingDayDetection:
    def test_weekday_is_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-03-04 is a Wednesday
        d = datetime.date(2026, 3, 4)
        assert is_trading_day(d) is True

    def test_weekend_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-03-07 is a Saturday
        d = datetime.date(2026, 3, 7)
        assert is_trading_day(d) is False

    def test_christmas_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2025-12-25 is Christmas (NYSE closed)
        d = datetime.date(2025, 12, 25)
        assert is_trading_day(d) is False

    def test_mlk_day_is_not_trading_day(self):
        is_trading_day = _lambda_handler.is_trading_day
        # 2026-01-19 is MLK Day
        d = datetime.date(2026, 1, 19)
        assert is_trading_day(d) is False


class TestHandlerHolidaySkip:
    @patch("lambda.handler._is_scheduled_run_time", return_value=True)
    @patch("lambda.handler.is_trading_day", return_value=False)
    def test_skips_on_holiday(self, mock_trading_day, mock_time):
        handler = _lambda_handler.handler
        result = handler({}, {})
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "market_holiday"


class TestScoreAggregatorNode:
    def test_aggregator_uses_sector_modifiers(self):
        from graph.research_graph import score_aggregator

        state = {
            "run_date": "2026-03-04",
            "run_time": "2026-03-04T06:20:00Z",
            "universe_tickers": ["AAPL"],
            "candidate_tickers": [],
            "technical_scores": {"AAPL": {"technical_score": 70.0}},
            "news_scores": {"AAPL": 65.0},
            "research_scores": {"AAPL": 60.0},
            "sector_modifiers": {"Technology": 1.2},
            "prior_theses": {},
            "market_regime": "bull",
            "scanner_scores": {},
        }

        # Patch SECTOR_MAP to include AAPL
        with patch("graph.research_graph.SECTOR_MAP", {"AAPL": "Technology"}):
            result = score_aggregator(state)

        aapl_thesis = result["investment_theses"]["AAPL"]
        # Expected base = 70*0.4 + 65*0.3 + 60*0.3 = 28 + 19.5 + 18 = 65.5
        # With 1.2 modifier = 65.5 * 1.2 = 78.6
        assert aapl_thesis["final_score"] > 70.0
        assert aapl_thesis["macro_modifier"] == pytest.approx(1.2, abs=0.01)
