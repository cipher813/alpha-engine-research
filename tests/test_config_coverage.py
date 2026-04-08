"""Tests for config.py — params, caching, and S3 fallback."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

import config as cfg


class TestGetResearchParams:
    def setup_method(self):
        cfg._research_params_cache = None

    @patch("config._load_research_params_from_s3", return_value=None)
    def test_defaults(self, _mock):
        params = cfg.get_research_params()
        assert "pead_window_min_days" in params
        assert "pead_strong_boost" in params
        assert isinstance(params["pead_window_min_days"], int)

    @patch("config._load_research_params_from_s3", return_value={"pead_strong_boost": 99.0})
    def test_s3_override(self, _mock):
        params = cfg.get_research_params()
        assert params["pead_strong_boost"] == 99.0
        # Other defaults should still be present
        assert "pead_window_min_days" in params

    @patch("config._load_research_params_from_s3", return_value=None)
    def test_cached_on_second_call(self, mock_load):
        cfg.get_research_params()
        cfg.get_research_params()
        # Should only load once
        mock_load.assert_called_once()


class TestRp:
    def setup_method(self):
        cfg._research_params_cache = None

    @patch("config._load_research_params_from_s3", return_value=None)
    def test_returns_value(self, _mock):
        assert cfg.rp("pead_window_min_days") is not None

    @patch("config._load_research_params_from_s3", return_value=None)
    def test_raises_on_missing(self, _mock):
        with pytest.raises(KeyError):
            cfg.rp("nonexistent_key")


class TestGetScannerParams:
    def setup_method(self):
        cfg._scanner_params_cache = None

    def test_defaults_without_s3(self):
        params = cfg.get_scanner_params()
        assert "tech_score_min" in params
        assert "max_atr_pct" in params
        assert isinstance(params["tech_score_min"], int)

    def test_cached(self):
        cfg._scanner_params_cache = None
        p1 = cfg.get_scanner_params()
        p2 = cfg.get_scanner_params()
        assert p1 is p2


class TestLoadResearchParamsFromS3:
    def test_no_s3_returns_none(self):
        # Without credentials, S3 call fails → returns None
        result = cfg._load_research_params_from_s3()
        assert result is None


class TestConstants:
    """Verify key config constants are loaded from sample YAML."""

    def test_weights_exist(self):
        assert cfg.WEIGHT_NEWS is not None
        assert cfg.WEIGHT_RESEARCH is not None
        assert cfg.WEIGHT_NEWS + cfg.WEIGHT_RESEARCH == pytest.approx(1.0)

    def test_thresholds(self):
        assert cfg.RATING_BUY_THRESHOLD > cfg.RATING_SELL_THRESHOLD

    def test_staleness(self):
        assert cfg.STALENESS_THRESHOLD_DAYS >= 1

    def test_sector_map_is_dict(self):
        assert isinstance(cfg.SECTOR_MAP, dict)
