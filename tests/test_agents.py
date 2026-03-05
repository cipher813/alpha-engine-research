"""
Tests for agent prompt construction and JSON extraction.
These tests validate the non-LLM parts of agent logic (prompt formatting,
JSON parsing, response extraction) without making real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.news_agent import _format_articles, _format_recurring_themes, _extract_json_from_response
from agents.research_agent import _format_rating_changes, _format_earnings, _extract_json_from_response as research_extract
from agents.macro_agent import _extract_macro_json
from agents.scanner_ranking_agent import _build_candidates_table, _extract_ranking


class TestNewsAgentHelpers:
    def test_format_articles_empty(self):
        assert _format_articles([]) == "No new articles."

    def test_format_articles(self):
        articles = [{
            "headline": "AAPL beats earnings",
            "source": "Reuters",
            "article_excerpt": "Apple Inc reported..."
        }]
        result = _format_articles(articles)
        assert "AAPL beats earnings" in result
        assert "Reuters" in result

    def test_format_recurring_themes_empty(self):
        assert _format_recurring_themes([]) == "None."

    def test_format_recurring_themes(self):
        themes = [{"theme": "tariffs", "mention_count": 5, "example_headline": "Tariff fears mount"}]
        result = _format_recurring_themes(themes)
        assert "tariffs" in result
        assert "5" in result

    def test_extract_json_valid(self):
        text = 'Some report text\n{"news_score": 75, "sentiment": "positive", "key_catalyst": "earnings beat", "prior_date": "2026-03-03", "material_changes": true, "dominant_theme": null, "dominant_theme_count": 0}'
        result = _extract_json_from_response(text)
        assert result["news_score"] == 75
        assert result["sentiment"] == "positive"
        assert result["material_changes"] is True

    def test_extract_json_fallback(self):
        result = _extract_json_from_response("No JSON here.")
        assert result["news_score"] == 50
        assert result["sentiment"] == "neutral"


class TestResearchAgentHelpers:
    def test_format_rating_changes_empty(self):
        assert _format_rating_changes([]) == "None in last 30 days."

    def test_format_rating_changes(self):
        changes = [{
            "date": "2026-03-01",
            "firm": "Goldman",
            "action": "Upgrade",
            "from_grade": "Hold",
            "to_grade": "Buy",
        }]
        result = _format_rating_changes(changes)
        assert "Goldman" in result
        assert "Upgrade" in result

    def test_format_earnings_empty(self):
        assert _format_earnings([]) == "No recent earnings data."

    def test_format_earnings_beat(self):
        surprises = [{
            "date": "2026-01-15",
            "actual": 2.50,
            "estimated": 2.30,
            "surprise_pct": 8.7
        }]
        result = _format_earnings(surprises)
        assert "beat" in result
        assert "8.7" in result

    def test_extract_research_json(self):
        text = 'Report\n{"research_score": 82, "consensus_direction": "bullish", "key_upside": "20% target", "key_risk": "macro", "material_changes": false}'
        result = research_extract(text)
        assert result["research_score"] == 82
        assert result["consensus_direction"] == "bullish"


class TestMacroAgentHelpers:
    def test_extract_macro_json_valid(self):
        text = '''Report text here.
{"market_regime": "bull",
 "key_theme": "Rate cuts incoming",
 "material_changes": true,
 "sector_modifiers": {
   "Technology": 1.15,
   "Healthcare": 1.05,
   "Financial": 1.10,
   "Consumer Discretionary": 1.10,
   "Consumer Staples": 1.00,
   "Energy": 1.05,
   "Industrials": 1.08,
   "Materials": 1.05,
   "Real Estate": 1.20,
   "Utilities": 1.10,
   "Communication Services": 1.12
 }}'''
        result = _extract_macro_json(text)
        assert result["market_regime"] == "bull"
        assert "Technology" in result["sector_modifiers"]
        assert 0.70 <= result["sector_modifiers"]["Technology"] <= 1.30

    def test_extract_macro_json_fallback(self):
        result = _extract_macro_json("No JSON here.")
        assert result["market_regime"] == "neutral"
        assert "Technology" in result["sector_modifiers"]

    def test_sector_modifiers_clamped(self):
        text = '{"market_regime": "bull", "sector_modifiers": {"Technology": 2.0, "Healthcare": 0.5}}'
        result = _extract_macro_json(text)
        assert result["sector_modifiers"].get("Technology", 1.3) <= 1.30
        assert result["sector_modifiers"].get("Healthcare", 0.70) >= 0.70


class TestScannerRankingAgent:
    def test_build_candidates_table(self):
        candidates = [
            {
                "ticker": "NVDA",
                "sector": "Technology",
                "path": "momentum",
                "tech_score": 85,
                "analyst_rating": "Strong Buy",
                "upside_pct": 25,
                "headlines": ["NVDA surges", "AI demand strong"],
            }
        ]
        table = _build_candidates_table(candidates)
        assert "NVDA" in table
        assert "momentum" in table
        assert "85" in table

    def test_extract_ranking_valid(self):
        text = '[{"rank": 1, "ticker": "NVDA", "path": "momentum", "rationale": "Strong AI tailwind"}]'
        result = _extract_ranking(text)
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"

    def test_extract_ranking_fallback(self):
        result = _extract_ranking("No valid JSON here.")
        assert result == []
