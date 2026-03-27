"""Tests for episodic and semantic memory tables and retrieval."""

import sqlite3
import pytest
from archive.manager import ArchiveManager


@pytest.fixture
def am(tmp_path):
    """Create an ArchiveManager with an in-memory DB."""
    db_path = str(tmp_path / "test.db")
    manager = ArchiveManager(local_db_path=db_path)
    manager.db_conn = sqlite3.connect(db_path)
    manager.db_conn.row_factory = sqlite3.Row
    manager._ensure_schema()
    return manager


class TestEpisodicMemory:
    def test_write_and_read(self, am):
        am.db_conn.execute(
            "INSERT INTO memory_episodes "
            "(ticker, signal_date, score, conviction, thesis_summary, "
            "outcome_10d, outcome_vs_spy, lesson, sector, pattern_tags, created_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NVDA", "2026-03-10", 78, "rising", "AI infrastructure",
             -0.12, -0.09, "Check margin sustainability", "Technology",
             '["earnings"]', "2026-03-10"),
        )
        am.db_conn.commit()

        result = am.load_episodic_memories(
            tickers=["NVDA"], sectors=["Technology"],
        )
        assert "NVDA" in result
        assert result["NVDA"][0]["lesson"] == "Check margin sustainability"

    def test_dedup_by_ticker_date(self, am):
        am.db_conn.execute(
            "INSERT INTO memory_episodes "
            "(ticker, signal_date, score, lesson, created_date) "
            "VALUES ('AAPL', '2026-03-01', 70, 'lesson1', '2026-03-01')"
        )
        am.db_conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            am.db_conn.execute(
                "INSERT INTO memory_episodes "
                "(ticker, signal_date, score, lesson, created_date) "
                "VALUES ('AAPL', '2026-03-01', 75, 'lesson2', '2026-03-01')"
            )

    def test_sector_level_retrieval(self, am):
        am.db_conn.execute(
            "INSERT INTO memory_episodes "
            "(ticker, signal_date, score, lesson, sector, created_date) "
            "VALUES ('MU', '2026-03-05', 72, 'Memory cycle lesson', 'Technology', '2026-03-05')"
        )
        am.db_conn.commit()

        # Query for a different ticker but same sector
        result = am.load_episodic_memories(
            tickers=["INTC"], sectors=["Technology"],
        )
        assert "MU" in result  # MU's sector memory retrieved

    def test_empty_returns_empty(self, am):
        result = am.load_episodic_memories(tickers=["AAPL"], sectors=[])
        assert result == {}


class TestSemanticMemory:
    def test_save_and_load(self, am):
        saved = am.save_semantic_memory(
            category="sector_observation",
            source="team:technology",
            content="Semiconductor inventory correction signals strengthening",
            sector="Technology",
            related_tickers=["NVDA", "AMD"],
            run_date="2026-03-20",
        )
        assert saved is True

        result = am.load_semantic_memories(sectors=["Technology"])
        assert "Technology" in result
        assert "Semiconductor" in result["Technology"][0]["content"]

    def test_duplicate_does_not_crash(self, am):
        am.save_semantic_memory(
            category="macro_reasoning", source="macro",
            content="Yield curve inversion deepening",
            sector=None, related_tickers=None, run_date="2026-03-20",
        )
        # Save same content again — should not raise
        am.save_semantic_memory(
            category="macro_reasoning", source="macro",
            content="Yield curve inversion deepening",
            sector=None, related_tickers=None, run_date="2026-03-25",
        )
        # Should have at most 2 entries (no crash)
        count = am.db_conn.execute("SELECT count(*) FROM memory_semantic").fetchone()[0]
        assert count <= 2

    def test_cross_sector_retrieval(self, am):
        am.save_semantic_memory(
            category="cross_sector", source="cio",
            content="Rotation from growth to value",
            sector=None, related_tickers=["AAPL", "BRK.B"],
            run_date="2026-03-20",
        )
        result = am.load_semantic_memories(sectors=["Technology"])
        assert "_cross_sector" in result

    def test_empty_returns_empty(self, am):
        result = am.load_semantic_memories(sectors=["Energy"])
        assert result == {}
