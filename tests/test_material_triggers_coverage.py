"""Tests for agents/sector_teams/material_triggers.py."""

import pandas as pd
import pytest

from agents.sector_teams.material_triggers import check_material_triggers


class TestMaterialTriggers:
    def test_no_triggers(self):
        result = check_material_triggers(
            "AAPL", None, None, None, None, None, False, "2026-04-08"
        )
        assert result == []

    def test_news_volume_spike(self):
        news = {"articles": [{"title": f"article {i}"} for i in range(5)]}
        result = check_material_triggers(
            "AAPL", news, None, None, None, None, False, "2026-04-08"
        )
        assert "news_volume_spike" in result

    def test_news_volume_below_threshold(self):
        news = {"articles": [{"title": "one"}]}
        result = check_material_triggers(
            "AAPL", news, None, None, None, None, False, "2026-04-08"
        )
        assert "news_volume_spike" not in result

    def test_news_article_count_field(self):
        news = {"article_count": 5}
        result = check_material_triggers(
            "AAPL", news, None, None, None, None, False, "2026-04-08"
        )
        assert "news_volume_spike" in result

    def test_price_move_gt_2atr(self):
        n = 30
        closes = [150.0] * (n - 1) + [180.0]  # 30pt jump, ATR ~10, 2*ATR=20
        price_data = {
            "Close": pd.Series(closes),
            "High": pd.Series([155.0] * n),
            "Low": pd.Series([145.0] * n),
        }
        result = check_material_triggers(
            "AAPL", None, price_data, None, None, None, False, "2026-04-08"
        )
        assert "price_move_gt_2atr" in result

    def test_price_move_small(self):
        n = 30
        closes = [150.0] * n
        price_data = {
            "Close": pd.Series(closes),
            "High": pd.Series([155.0] * n),
            "Low": pd.Series([145.0] * n),
        }
        result = check_material_triggers(
            "AAPL", None, price_data, None, None, None, False, "2026-04-08"
        )
        assert "price_move_gt_2atr" not in result

    def test_analyst_rating_change(self):
        analyst = {"rating_changes": [{"from": "hold", "to": "buy"}]}
        result = check_material_triggers(
            "AAPL", None, None, analyst, None, None, False, "2026-04-08"
        )
        assert "analyst_rating_change" in result

    def test_analyst_target_revision(self):
        analyst = {"upside_pct": 25.0}
        prior = {"price_target_upside": 10.0}
        result = check_material_triggers(
            "AAPL", None, None, analyst, None, prior, False, "2026-04-08"
        )
        assert "analyst_target_revision" in result

    def test_analyst_target_small_revision(self):
        analyst = {"upside_pct": 12.0}
        prior = {"price_target_upside": 10.0}
        result = check_material_triggers(
            "AAPL", None, None, analyst, None, prior, False, "2026-04-08"
        )
        assert "analyst_target_revision" not in result

    def test_recent_earnings(self):
        analyst = {"earnings_surprises": [{"date": "2026-04-06", "surprise_pct": 5}]}
        result = check_material_triggers(
            "AAPL", None, None, analyst, None, None, False, "2026-04-08"
        )
        assert "recent_earnings" in result

    def test_old_earnings(self):
        analyst = {"earnings_surprises": [{"date": "2026-01-01", "surprise_pct": 5}]}
        result = check_material_triggers(
            "AAPL", None, None, analyst, None, None, False, "2026-04-08"
        )
        assert "recent_earnings" not in result

    def test_insider_cluster(self):
        insider = {"unique_buyers_30d": 3}
        result = check_material_triggers(
            "AAPL", None, None, None, insider, None, False, "2026-04-08"
        )
        assert "insider_cluster" in result

    def test_insider_selling(self):
        insider = {"net_sentiment": -0.8}
        result = check_material_triggers(
            "AAPL", None, None, None, insider, None, False, "2026-04-08"
        )
        assert "insider_selling" in result

    def test_sector_regime_change(self):
        result = check_material_triggers(
            "AAPL", None, None, None, None, None, True, "2026-04-08"
        )
        assert "sector_regime_change" in result

    def test_multiple_triggers(self):
        news = {"articles": [{"title": f"a{i}"} for i in range(5)]}
        analyst = {"rating_changes": [{"from": "hold", "to": "buy"}]}
        result = check_material_triggers(
            "AAPL", news, None, analyst, None, None, True, "2026-04-08"
        )
        assert len(result) >= 3
        assert "news_volume_spike" in result
        assert "analyst_rating_change" in result
        assert "sector_regime_change" in result
