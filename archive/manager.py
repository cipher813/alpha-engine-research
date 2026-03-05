"""
Archive manager — S3 read/write and SQLite CRUD.

Manages the full archive lifecycle:
  - Download research.db from S3 at run start
  - Read prior agent reports from S3
  - Write updated reports and theses to S3
  - Write dated history snapshots to S3
  - Upload updated research.db to S3 at run end

S3 layout: see §7.1
SQLite schema: see §7.2
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import S3_BUCKET, AWS_REGION

_DB_S3_KEY = "research.db"
_BACKUP_KEY_TPL = "backups/research_{date}.db"


# ── S3 helpers ────────────────────────────────────────────────────────────────

class ArchiveManager:
    def __init__(self, bucket: str = S3_BUCKET, region: str = AWS_REGION, local_db_path: Optional[str] = None):
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)
        self.local_db_path = local_db_path or os.path.join(tempfile.gettempdir(), "research.db")
        self.db_conn: Optional[sqlite3.Connection] = None

    # ── Database lifecycle ────────────────────────────────────────────────────

    def download_db(self) -> sqlite3.Connection:
        """Download research.db from S3 and open connection. Creates schema if new."""
        try:
            self.s3.download_file(self.bucket, _DB_S3_KEY, self.local_db_path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # First run — create fresh DB
                pass
            else:
                raise

        self.db_conn = sqlite3.connect(self.local_db_path)
        self.db_conn.row_factory = sqlite3.Row
        self._ensure_schema()
        return self.db_conn

    def upload_db(self, run_date: str) -> None:
        """Upload research.db to S3 and create a dated backup."""
        if self.db_conn:
            self.db_conn.commit()
        self.s3.upload_file(self.local_db_path, self.bucket, _DB_S3_KEY)
        backup_key = _BACKUP_KEY_TPL.format(date=run_date.replace("-", ""))
        self.s3.upload_file(self.local_db_path, self.bucket, backup_key)

    def _ensure_schema(self) -> None:
        """Create all tables if they don't exist (§7.2)."""
        conn = self.db_conn
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS investment_thesis (
            id                       INTEGER PRIMARY KEY,
            symbol                   TEXT NOT NULL,
            date                     TEXT NOT NULL,
            run_time                 TEXT NOT NULL,
            rating                   TEXT NOT NULL,
            score                    REAL NOT NULL,
            technical_score          REAL,
            news_score               REAL,
            research_score           REAL,
            macro_modifier           REAL,
            thesis_summary           TEXT,
            prev_rating              TEXT,
            prev_score               REAL,
            last_material_change_date TEXT,
            stale_days               INTEGER,
            consistency_flag         INTEGER DEFAULT 0,
            UNIQUE(symbol, date, run_time)
        );

        CREATE TABLE IF NOT EXISTS agent_reports (
            id          INTEGER PRIMARY KEY,
            symbol      TEXT,
            date        TEXT NOT NULL,
            run_time    TEXT NOT NULL,
            agent_type  TEXT NOT NULL,
            report_md   TEXT NOT NULL,
            word_count  INTEGER,
            UNIQUE(symbol, date, run_time, agent_type)
        );

        CREATE TABLE IF NOT EXISTS candidate_tenures (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            slot            INTEGER NOT NULL,
            entry_date      TEXT NOT NULL,
            exit_date       TEXT,
            exit_reason     TEXT,
            replaced_by     TEXT,
            peak_score      REAL,
            exit_score      REAL,
            tenure_days     INTEGER
        );

        CREATE TABLE IF NOT EXISTS active_candidates (
            slot            INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            entry_date      TEXT NOT NULL,
            prior_tenures   INTEGER NOT NULL DEFAULT 0,
            score           REAL,
            consecutive_low_runs INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS scanner_appearances (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            date            TEXT NOT NULL,
            scanner_rank    INTEGER NOT NULL,
            scan_path       TEXT,
            tech_score      REAL,
            news_score      REAL,
            research_score  REAL,
            final_score     REAL,
            selected        INTEGER NOT NULL DEFAULT 0,
            selection_reason TEXT,
            UNIQUE(symbol, date)
        );

        CREATE TABLE IF NOT EXISTS technical_scores (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            date            TEXT NOT NULL,
            rsi_14          REAL,
            macd_signal     REAL,
            price_vs_ma50   REAL,
            price_vs_ma200  REAL,
            momentum_20d    REAL,
            technical_score REAL,
            UNIQUE(symbol, date)
        );

        CREATE TABLE IF NOT EXISTS macro_snapshots (
            id                  INTEGER PRIMARY KEY,
            date                TEXT NOT NULL UNIQUE,
            fed_funds_rate      REAL,
            treasury_2yr        REAL,
            treasury_10yr       REAL,
            yield_curve_slope   REAL,
            vix                 REAL,
            sp500_close         REAL,
            sp500_30d_return    REAL,
            oil_wti             REAL,
            gold                REAL,
            copper              REAL,
            market_regime       TEXT,
            sector_modifiers    TEXT
        );

        CREATE TABLE IF NOT EXISTS score_performance (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            score_date      TEXT NOT NULL,
            score           REAL NOT NULL,
            price_on_date   REAL,
            price_10d       REAL,
            price_30d       REAL,
            spy_10d_return  REAL,
            spy_30d_return  REAL,
            return_10d      REAL,
            return_30d      REAL,
            beat_spy_10d    INTEGER,
            beat_spy_30d    INTEGER,
            eval_date_10d   TEXT,
            eval_date_30d   TEXT,
            UNIQUE(symbol, score_date)
        );

        CREATE TABLE IF NOT EXISTS news_article_hashes (
            id          INTEGER PRIMARY KEY,
            symbol      TEXT NOT NULL,
            article_hash TEXT NOT NULL,
            first_seen  TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 1,
            UNIQUE(symbol, article_hash)
        );
        """)
        # ── Column migrations (existing DBs may lack newer columns) ──────────
        migrations = [
            "ALTER TABLE investment_thesis ADD COLUMN conviction TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN signal TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN score_velocity_5d REAL",
            "ALTER TABLE investment_thesis ADD COLUMN price_target_upside REAL",
            "ALTER TABLE macro_snapshots ADD COLUMN sector_ratings TEXT",
        ]
        for stmt in migrations:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # column already exists
        conn.commit()

    # ── S3 object helpers ─────────────────────────────────────────────────────

    def _s3_get(self, key: str) -> Optional[str]:
        """Download S3 object and return as string. Returns None if not found."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def _s3_put(self, key: str, body: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body.encode("utf-8"))

    # ── Universe archive read/write ───────────────────────────────────────────

    def load_prior_reports(self, ticker: str, category: str = "universe") -> dict:
        """
        Load the latest archived reports for a ticker.
        category: 'universe' or 'candidates'
        Returns dict with 'news_report', 'research_report', 'thesis' (or None).
        """
        base = f"archive/{category}/{ticker}"
        return {
            "news_report": self._s3_get(f"{base}/news_report.md"),
            "research_report": self._s3_get(f"{base}/research_report.md"),
            "thesis": self._load_thesis_json(f"{base}/thesis.json"),
        }

    def _load_thesis_json(self, key: str) -> Optional[dict]:
        raw = self._s3_get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    def save_reports(
        self,
        ticker: str,
        run_date: str,
        news_report: Optional[str],
        research_report: Optional[str],
        thesis: Optional[dict],
        category: str = "universe",
    ) -> None:
        """
        Write updated reports and thesis to S3 for a ticker.
        Creates both the 'latest' file and a dated history snapshot.
        """
        base = f"archive/{category}/{ticker}"
        hist = f"{base}/history/{run_date}"

        if news_report:
            self._s3_put(f"{base}/news_report.md", news_report)
            self._s3_put(f"{hist}/news_report.md", news_report)

        if research_report:
            self._s3_put(f"{base}/research_report.md", research_report)
            self._s3_put(f"{hist}/research_report.md", research_report)

        if thesis:
            thesis_json = json.dumps(thesis, indent=2)
            self._s3_put(f"{base}/thesis.json", thesis_json)
            self._s3_put(f"{hist}/thesis.json", thesis_json)

    def save_macro_report(self, run_date: str, macro_report: str) -> None:
        self._s3_put("archive/macro/macro_report.md", macro_report)
        self._s3_put(f"archive/macro/history/{run_date}/macro_report.md", macro_report)

    def save_consolidated_report(self, run_date: str, report: str) -> None:
        self._s3_put(f"consolidated/{run_date}/morning.md", report)

    # ── Active candidates ─────────────────────────────────────────────────────

    def load_active_candidates(self) -> list[dict]:
        """Load current 3 active candidates from DB."""
        if not self.db_conn:
            return []
        rows = self.db_conn.execute(
            "SELECT slot, symbol, entry_date, prior_tenures, score, consecutive_low_runs FROM active_candidates ORDER BY slot"
        ).fetchall()
        return [dict(r) for r in rows]

    def save_active_candidates(self, candidates: list[dict]) -> None:
        """Overwrite active_candidates table with new state."""
        if not self.db_conn:
            return
        self.db_conn.execute("DELETE FROM active_candidates")
        for c in candidates:
            self.db_conn.execute(
                """INSERT INTO active_candidates (slot, symbol, entry_date, prior_tenures, score, consecutive_low_runs)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (c["slot"], c["symbol"], c["entry_date"], c.get("prior_tenures", 0),
                 c.get("score"), c.get("consecutive_low_runs", 0)),
            )
        self.db_conn.commit()

    # ── DB write helpers ──────────────────────────────────────────────────────

    def load_score_history(self, tickers: list[str], n: int = 6) -> dict[str, list[float]]:
        """
        Return the last n scores (most recent first) for each ticker.
        Used to compute conviction and score_velocity_5d in the aggregator.
        """
        if not self.db_conn:
            return {}
        result = {}
        for ticker in tickers:
            rows = self.db_conn.execute(
                """SELECT score FROM investment_thesis WHERE symbol = ?
                   ORDER BY date DESC, run_time DESC LIMIT ?""",
                (ticker, n),
            ).fetchall()
            result[ticker] = [r[0] for r in rows]
        return result

    def write_signals_json(self, run_date: str, run_time: str, signals: dict) -> None:
        """Write the machine-readable signals.json to S3 for executor consumption (§A.1)."""
        payload = {"date": run_date, "run_time": run_time, **signals}
        self._s3_put(
            f"signals/{run_date}/signals.json",
            json.dumps(payload, indent=2, default=str),
        )

    def write_investment_thesis(self, thesis: dict, run_time: str) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT OR REPLACE INTO investment_thesis
               (symbol, date, run_time, rating, score, technical_score, news_score,
                research_score, macro_modifier, thesis_summary, prev_rating, prev_score,
                last_material_change_date, stale_days, consistency_flag,
                conviction, signal, score_velocity_5d, price_target_upside)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thesis["ticker"], thesis["date"], run_time, thesis["rating"],
                thesis["final_score"], thesis.get("technical_score"),
                thesis.get("news_score"), thesis.get("research_score"),
                thesis.get("macro_modifier"), thesis.get("thesis_summary"),
                thesis.get("prior_rating"), thesis.get("prior_score"),
                thesis.get("last_material_change_date"), thesis.get("stale_days"),
                thesis.get("consistency_flag", 0),
                thesis.get("conviction", "stable"),
                thesis.get("signal", "HOLD"),
                thesis.get("score_velocity_5d"),
                thesis.get("price_target_upside"),
            ),
        )

    def write_agent_report(self, report: dict, run_time: str) -> None:
        if not self.db_conn:
            return
        text = report.get("report_md", "")
        self.db_conn.execute(
            """INSERT OR REPLACE INTO agent_reports
               (symbol, date, run_time, agent_type, report_md, word_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                report.get("symbol"), report["date"], run_time,
                report["agent_type"], text, len(text.split()),
            ),
        )

    def write_technical_score(self, ticker: str, date: str, data: dict) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT OR REPLACE INTO technical_scores
               (symbol, date, rsi_14, macd_signal, price_vs_ma50, price_vs_ma200, momentum_20d, technical_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker, date,
                data.get("rsi_14"), data.get("macd_cross"),
                data.get("price_vs_ma50"), data.get("price_vs_ma200"),
                data.get("momentum_20d"), data.get("technical_score"),
            ),
        )

    def write_macro_snapshot(self, date: str, macro: dict) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT OR REPLACE INTO macro_snapshots
               (date, fed_funds_rate, treasury_2yr, treasury_10yr, yield_curve_slope,
                vix, sp500_close, sp500_30d_return, oil_wti, gold, copper,
                market_regime, sector_modifiers, sector_ratings)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                date,
                macro.get("fed_funds_rate"), macro.get("treasury_2yr"),
                macro.get("treasury_10yr"), macro.get("yield_curve_slope"),
                macro.get("vix"), macro.get("sp500_close"),
                macro.get("sp500_30d_return"), macro.get("oil_wti"),
                macro.get("gold"), macro.get("copper"),
                macro.get("market_regime"),
                json.dumps(macro.get("sector_modifiers", {})),
                json.dumps(macro.get("sector_ratings", {})),
            ),
        )

    def write_scanner_appearances(self, appearances: list[dict]) -> None:
        if not self.db_conn:
            return
        for a in appearances:
            self.db_conn.execute(
                """INSERT OR REPLACE INTO scanner_appearances
                   (symbol, date, scanner_rank, scan_path, tech_score, news_score,
                    research_score, final_score, selected, selection_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    a["symbol"], a["date"], a["scanner_rank"], a.get("scan_path"),
                    a.get("tech_score"), a.get("news_score"), a.get("research_score"),
                    a.get("final_score"), a.get("selected", 0), a.get("selection_reason"),
                ),
            )

    def write_candidate_tenure_entry(self, tenure: dict) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT INTO candidate_tenures (symbol, slot, entry_date)
               VALUES (?, ?, ?)""",
            (tenure["symbol"], tenure["slot"], tenure["entry_date"]),
        )

    def close_candidate_tenure(self, symbol: str, exit_date: str, exit_score: float,
                                exit_reason: str, replaced_by: Optional[str], tenure_days: int,
                                peak_score: float) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """UPDATE candidate_tenures
               SET exit_date=?, exit_score=?, exit_reason=?, replaced_by=?, tenure_days=?, peak_score=?
               WHERE symbol=? AND exit_date IS NULL""",
            (exit_date, exit_score, exit_reason, replaced_by, tenure_days, peak_score, symbol),
        )

    def upsert_news_hashes(self, ticker: str, new_hashes: list[str], today: str) -> None:
        if not self.db_conn:
            return
        for h in new_hashes:
            self.db_conn.execute(
                """INSERT INTO news_article_hashes (symbol, article_hash, first_seen, mention_count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(symbol, article_hash) DO UPDATE SET mention_count = mention_count + 1""",
                (ticker, h, today),
            )

    def load_news_hashes(self, ticker: str) -> set[str]:
        if not self.db_conn:
            return set()
        rows = self.db_conn.execute(
            "SELECT article_hash FROM news_article_hashes WHERE symbol = ?", (ticker,)
        ).fetchall()
        return {r[0] for r in rows}

    def load_prior_theses(self, tickers: list[str]) -> dict[str, dict]:
        """Load the most recent investment_thesis row for each ticker."""
        if not self.db_conn:
            return {}
        result = {}
        for ticker in tickers:
            row = self.db_conn.execute(
                """SELECT * FROM investment_thesis WHERE symbol = ?
                   ORDER BY date DESC, run_time DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
            if row:
                result[ticker] = dict(row)
        return result

    def commit(self) -> None:
        if self.db_conn:
            self.db_conn.commit()

    def close(self) -> None:
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
