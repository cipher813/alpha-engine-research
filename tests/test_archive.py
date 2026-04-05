"""Tests for archive manager (using in-memory SQLite, no S3 calls)."""

import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

ArchiveManager = pytest.importorskip("archive.manager", reason="archive.manager requires gitignored config").ArchiveManager


@pytest.fixture
def archive_in_memory():
    """Create an ArchiveManager with an in-memory SQLite DB and mocked S3."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = ArchiveManager(bucket="test-bucket", local_db_path=db_path)
    manager.s3 = MagicMock()
    manager.s3.get_object.side_effect = Exception("NoSuchKey")
    manager.s3.put_object = MagicMock()
    manager.s3.upload_file = MagicMock()
    manager.s3.download_file = MagicMock(side_effect=Exception("mock"))

    # Initialize with fresh schema
    manager.db_conn = sqlite3.connect(db_path)
    manager.db_conn.row_factory = sqlite3.Row
    manager._ensure_schema()

    yield manager

    manager.close()
    os.unlink(db_path)


class TestArchiveSchema:
    def test_schema_created(self, archive_in_memory):
        conn = archive_in_memory.db_conn
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        expected = {
            "investment_thesis", "agent_reports", "candidate_tenures",
            "active_candidates", "scanner_appearances", "technical_scores",
            "macro_snapshots", "score_performance", "news_article_hashes",
        }
        assert expected.issubset(set(tables))


class TestInvestmentThesisWrite:
    def test_write_and_read_thesis(self, archive_in_memory):
        thesis = {
            "ticker": "NVDA",
            "date": "2026-03-04",
            "rating": "BUY",
            "final_score": 85.5,
            "technical_score": 88.0,
            "quant_score": 80.0,
            "qual_score": 82.0,
            "macro_modifier": 1.15,
            "thesis_summary": "NVDA rates BUY. AI demand is strong.",
            "prior_score": 80.0,
            "prior_rating": "BUY",
            "last_material_change_date": "2026-03-04",
            "stale_days": 0,
            "consistency_flag": 0,
        }
        archive_in_memory.write_investment_thesis(thesis, run_time="2026-03-04T06:20:00Z")

        row = archive_in_memory.db_conn.execute(
            "SELECT * FROM investment_thesis WHERE symbol = 'NVDA'"
        ).fetchone()
        assert row is not None
        assert row["rating"] == "BUY"
        assert abs(row["score"] - 85.5) < 0.01

    def test_load_prior_theses(self, archive_in_memory):
        thesis = {
            "ticker": "AAPL",
            "date": "2026-03-03",
            "rating": "HOLD",
            "final_score": 58.0,
            "technical_score": None,
            "quant_score": None,
            "qual_score": None,
            "macro_modifier": None,
            "thesis_summary": "AAPL rates HOLD.",
            "prior_score": None,
            "prior_rating": None,
            "last_material_change_date": None,
            "stale_days": 2,
            "consistency_flag": 0,
        }
        archive_in_memory.write_investment_thesis(thesis, run_time="2026-03-03T06:20:00Z")

        prior = archive_in_memory.load_prior_theses(["AAPL"])
        assert "AAPL" in prior
        assert prior["AAPL"]["rating"] == "HOLD"


class TestActiveCandidates:
    def test_save_and_load_candidates(self, archive_in_memory):
        candidates = [
            {"slot": 1, "symbol": "NVDA", "entry_date": "2026-02-15", "prior_tenures": 0, "score": 85, "consecutive_low_runs": 0},
            {"slot": 2, "symbol": "MSFT", "entry_date": "2026-02-20", "prior_tenures": 1, "score": 78, "consecutive_low_runs": 0},
            {"slot": 3, "symbol": "AMZN", "entry_date": "2026-03-01", "prior_tenures": 0, "score": 72, "consecutive_low_runs": 0},
        ]
        archive_in_memory.save_active_candidates(candidates)
        loaded = archive_in_memory.load_active_candidates()
        assert len(loaded) == 3
        symbols = {c["symbol"] for c in loaded}
        assert symbols == {"NVDA", "MSFT", "AMZN"}


class TestNewsHashes:
    def test_upsert_and_load_hashes(self, archive_in_memory):
        hashes = ["abc123", "def456"]
        archive_in_memory.upsert_news_hashes("AAPL", hashes, "2026-03-04")

        loaded = archive_in_memory.load_news_hashes("AAPL")
        assert "abc123" in loaded
        assert "def456" in loaded

    def test_mention_count_increments(self, archive_in_memory):
        archive_in_memory.upsert_news_hashes("AAPL", ["abc123"], "2026-03-03")
        archive_in_memory.upsert_news_hashes("AAPL", ["abc123"], "2026-03-04")

        row = archive_in_memory.db_conn.execute(
            "SELECT mention_count FROM news_article_hashes WHERE symbol='AAPL' AND article_hash='abc123'"
        ).fetchone()
        assert row["mention_count"] == 2


class TestTechnicalScoreWrite:
    def test_write_technical_score(self, archive_in_memory):
        data = {
            "rsi_14": 42.5,
            "macd_cross": 1.0,
            "price_vs_ma50": 3.2,
            "price_vs_ma200": -1.5,
            "momentum_20d": 4.0,
            "technical_score": 67.3,
        }
        archive_in_memory.write_technical_score("COST", "2026-03-04", data)

        row = archive_in_memory.db_conn.execute(
            "SELECT * FROM technical_scores WHERE symbol='COST'"
        ).fetchone()
        assert row is not None
        assert abs(row["rsi_14"] - 42.5) < 0.01
