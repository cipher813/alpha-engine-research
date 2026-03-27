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

import logging

import boto3
from botocore.exceptions import ClientError

from config import S3_BUCKET, AWS_REGION
from retry import retry

log = logging.getLogger(__name__)

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
            self._s3_download_file(_DB_S3_KEY, self.local_db_path)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                # First run — create fresh DB
                pass
            else:
                raise

        self.db_conn = sqlite3.connect(self.local_db_path)
        self.db_conn.row_factory = sqlite3.Row
        self._ensure_schema()
        return self.db_conn

    @retry(max_attempts=3, retryable=(Exception,), label="s3_download")
    def _s3_download_file(self, key: str, local_path: str) -> None:
        self.s3.download_file(self.bucket, key, local_path)

    @retry(max_attempts=3, retryable=(Exception,), label="s3_upload")
    def _s3_upload_file(self, local_path: str, key: str) -> None:
        self.s3.upload_file(local_path, self.bucket, key)

    def upload_db(self, run_date: str) -> None:
        """Upload research.db to S3 and create a dated backup."""
        if self.db_conn:
            self.db_conn.commit()
        self._s3_upload_file(self.local_db_path, _DB_S3_KEY)
        backup_key = _BACKUP_KEY_TPL.format(date=run_date.replace("-", ""))
        self._s3_upload_file(self.local_db_path, backup_key)

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

        CREATE TABLE IF NOT EXISTS predictor_outcomes (
            id                      INTEGER PRIMARY KEY,
            symbol                  TEXT NOT NULL,
            prediction_date         TEXT NOT NULL,
            predicted_direction     TEXT,
            prediction_confidence   REAL,
            p_up                    REAL,
            p_flat                  REAL,
            p_down                  REAL,
            score_modifier_applied  REAL DEFAULT 0.0,
            actual_5d_return        REAL,
            correct_5d              INTEGER,
            UNIQUE(symbol, prediction_date)
        );

        CREATE TABLE IF NOT EXISTS population (
            id                  INTEGER PRIMARY KEY,
            symbol              TEXT NOT NULL UNIQUE,
            sector              TEXT NOT NULL,
            long_term_score     REAL,
            long_term_rating    TEXT,
            conviction          TEXT DEFAULT 'stable',
            price_target_upside REAL,
            thesis_summary      TEXT,
            entry_date          TEXT,
            tenure_weeks        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS population_history (
            id          INTEGER PRIMARY KEY,
            date        TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            ticker_in   TEXT,
            ticker_out  TEXT,
            sector      TEXT,
            reason      TEXT,
            score_in    REAL,
            score_out   REAL
        );

        CREATE TABLE IF NOT EXISTS stock_archive (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL UNIQUE,
            sector          TEXT NOT NULL,
            sector_team     TEXT NOT NULL,
            first_analyzed  TEXT NOT NULL,
            last_analyzed   TEXT NOT NULL,
            times_in_population INTEGER DEFAULT 0,
            current_status  TEXT DEFAULT 'inactive'
        );

        CREATE TABLE IF NOT EXISTS thesis_history (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL,
            run_date        TEXT NOT NULL,
            author          TEXT NOT NULL,
            thesis_type     TEXT NOT NULL,
            bull_case       TEXT,
            bear_case       TEXT,
            catalysts       TEXT,
            risks           TEXT,
            conviction      INTEGER,
            score           REAL,
            rationale       TEXT
        );

        CREATE TABLE IF NOT EXISTS analyst_resources (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL,
            run_date        TEXT NOT NULL,
            agent           TEXT NOT NULL,
            resource_type   TEXT NOT NULL,
            resource_detail TEXT,
            influence       TEXT DEFAULT 'supporting'
        );

        CREATE TABLE IF NOT EXISTS memory_episodes (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL,
            signal_date     TEXT NOT NULL,
            score           REAL,
            rating          TEXT,
            conviction      TEXT,
            thesis_summary  TEXT,
            outcome_10d     REAL,
            outcome_vs_spy  REAL,
            lesson          TEXT,
            sector          TEXT,
            pattern_tags    TEXT,
            created_date    TEXT NOT NULL,
            UNIQUE(ticker, signal_date)
        );

        CREATE TABLE IF NOT EXISTS memory_semantic (
            id              INTEGER PRIMARY KEY,
            category        TEXT NOT NULL,
            source          TEXT NOT NULL,
            content         TEXT NOT NULL,
            sector          TEXT,
            related_tickers TEXT,
            created_date    TEXT NOT NULL,
            reinforced_date TEXT
        );
        """)
        # ── Column migrations (existing DBs may lack newer columns) ──────────
        migrations = [
            "ALTER TABLE investment_thesis ADD COLUMN conviction TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN signal TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN score_velocity_5d REAL",
            "ALTER TABLE investment_thesis ADD COLUMN price_target_upside REAL",
            "ALTER TABLE macro_snapshots ADD COLUMN sector_ratings TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN predicted_direction TEXT",
            "ALTER TABLE investment_thesis ADD COLUMN prediction_confidence REAL",
        ]
        for stmt in migrations:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    pass  # column already exists — safe to ignore
                else:
                    log.error("Schema migration failed: %s — %s", stmt, e)
                    raise
        conn.commit()

    # ── S3 object helpers ─────────────────────────────────────────────────────

    def _s3_get(self, key: str) -> Optional[str]:
        """Download S3 object and return as string. Returns None if not found."""
        try:
            return self._s3_get_inner(key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    @retry(max_attempts=3, retryable=(Exception,), label="s3_get")
    def _s3_get_inner(self, key: str) -> str:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    @retry(max_attempts=3, retryable=(Exception,), label="s3_put")
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

    def load_latest_theses(self, tickers: list[str]) -> dict[str, dict]:
        """Load the most recent investment thesis per ticker from SQLite.

        Returns {ticker: {rating, score, conviction, thesis_summary, ...}}.
        """
        if not self.db_conn or not tickers:
            return {}
        results = {}
        # SQLite pre-3.24: no window functions. Use GROUP BY + MAX(id) for latest per ticker.
        placeholders = ",".join("?" for _ in tickers)
        try:
            rows = self.db_conn.execute(
                f"""SELECT t.symbol, t.rating, t.score, t.conviction, t.signal,
                           t.thesis_summary, t.technical_score, t.price_target_upside
                    FROM investment_thesis t
                    INNER JOIN (
                        SELECT symbol, MAX(id) as max_id
                        FROM investment_thesis
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    ) latest ON t.id = latest.max_id""",
                tickers,
            ).fetchall()
            for row in rows:
                results[row["symbol"]] = {
                    "rating": row["rating"],
                    "score": row["score"],
                    "conviction": row["conviction"],
                    "signal": row["signal"],
                    "thesis_summary": row["thesis_summary"] or "",
                    "technical_score": row["technical_score"],
                    "price_target_upside": row["price_target_upside"],
                }
        except Exception as e:
            log.warning("Failed to load latest theses from SQLite: %s", e)
        return results

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

    def load_predictions_json(self) -> dict[str, dict]:
        """
        Load predictor/predictions/latest.json from S3.
        Returns dict mapping ticker -> prediction dict. Returns {} on any failure.
        """
        from config import PREDICTOR_PREDICTIONS_KEY
        try:
            raw = self._s3_get(PREDICTOR_PREDICTIONS_KEY)
            if not raw:
                return {}
            data = json.loads(raw)
            predictions = data.get("predictions", [])
            return {p["ticker"]: p for p in predictions if "ticker" in p}
        except Exception:
            return {}

    def write_predictor_outcome(self, symbol: str, prediction_date: str, outcome: dict) -> None:
        """
        Insert or update a predictor_outcomes row.
        Called by the backtester when 5-day outcomes are available.
        """
        if not self.db_conn:
            return
        # UPDATE-then-INSERT pattern (Lambda SQLite is pre-3.24, no ON CONFLICT DO UPDATE)
        cur = self.db_conn.execute(
            """UPDATE predictor_outcomes
               SET actual_5d_return = ?, correct_5d = ?
               WHERE symbol = ? AND prediction_date = ?""",
            (outcome.get("actual_5d_return"), outcome.get("correct_5d"),
             symbol, prediction_date),
        )
        if cur.rowcount == 0:
            self.db_conn.execute(
                """INSERT INTO predictor_outcomes
                   (symbol, prediction_date, predicted_direction, prediction_confidence,
                    p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, correct_5d)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, prediction_date,
                    outcome.get("predicted_direction"),
                    outcome.get("prediction_confidence"),
                    outcome.get("p_up"),
                    outcome.get("p_flat"),
                    outcome.get("p_down"),
                    outcome.get("score_modifier_applied", 0.0),
                    outcome.get("actual_5d_return"),
                    outcome.get("correct_5d"),
                ),
            )

    def write_prices_json(self, run_date: str, prices: dict) -> None:
        """Write daily OHLCV price snapshot to S3 for backtester consumption.

        S3 key: prices/{date}/prices.json
        Format: {"date": "YYYY-MM-DD", "prices": {"TICK": {"open": x, "close": x, "high": x, "low": x}}}
        """
        payload = {"date": run_date, "prices": prices}
        self._s3_put(
            f"prices/{run_date}/prices.json",
            json.dumps(payload, indent=2),
        )

    def write_investment_thesis(self, thesis: dict, run_time: str) -> None:
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT OR REPLACE INTO investment_thesis
               (symbol, date, run_time, rating, score, technical_score, news_score,
                research_score, macro_modifier, thesis_summary, prev_rating, prev_score,
                last_material_change_date, stale_days, consistency_flag,
                conviction, signal, score_velocity_5d, price_target_upside,
                predicted_direction, prediction_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                thesis.get("predicted_direction"),
                thesis.get("prediction_confidence"),
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
        # UPDATE-then-INSERT pattern (Lambda SQLite is pre-3.24, no ON CONFLICT DO UPDATE)
        for h in new_hashes:
            cur = self.db_conn.execute(
                """UPDATE news_article_hashes SET mention_count = mention_count + 1
                   WHERE symbol = ? AND article_hash = ?""",
                (ticker, h),
            )
            if cur.rowcount == 0:
                self.db_conn.execute(
                    """INSERT INTO news_article_hashes (symbol, article_hash, first_seen, mention_count)
                       VALUES (?, ?, ?, 1)""",
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

    # ── Population persistence ─────────────────────────────────────────────────

    def save_population(
        self,
        population: list[dict],
        run_date: str,
        market_regime: str = "neutral",
        sector_ratings: dict | None = None,
    ) -> None:
        """
        Persist current investment population to SQLite for next-run continuity.
        Also writes population/latest.json and population/{date}.json to S3.

        The S3 JSON includes market_regime and sector_ratings so the Executor's
        population_reader.py can consume it directly.
        """
        if not self.db_conn:
            return

        # SQLite: clear and rewrite population table
        self.db_conn.execute("DELETE FROM population")
        for p in population:
            self.db_conn.execute(
                """INSERT INTO population
                   (symbol, sector, long_term_score, long_term_rating,
                    conviction, price_target_upside, thesis_summary, entry_date, tenure_weeks)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p["ticker"], p.get("sector", "Unknown"),
                    p.get("long_term_score", 50.0), p.get("long_term_rating", "HOLD"),
                    p.get("conviction", "stable"), p.get("price_target_upside"),
                    p.get("thesis_summary", ""), p.get("entry_date", run_date),
                    p.get("tenure_weeks", 0),
                ),
            )
        self.db_conn.commit()

        # S3: write population JSON (includes regime + sector_ratings for Executor)
        pop_json = json.dumps({
            "date": run_date,
            "market_regime": market_regime,
            "sector_ratings": sector_ratings or {},
            "population": population,
        }, indent=2, default=str)
        self._s3_put("population/latest.json", pop_json)
        self._s3_put(f"population/{run_date}.json", pop_json)

    def load_population(self) -> list[dict]:
        """
        Load current population from SQLite.
        Returns [] on first run (before any population has been saved).
        """
        if not self.db_conn:
            return []
        try:
            rows = self.db_conn.execute(
                """SELECT symbol, sector, long_term_score, long_term_rating,
                          conviction, price_target_upside, thesis_summary,
                          entry_date, tenure_weeks
                   FROM population ORDER BY long_term_score DESC"""
            ).fetchall()
            return [
                {
                    "ticker": r[0],
                    "sector": r[1],
                    "long_term_score": r[2],
                    "long_term_rating": r[3],
                    "conviction": r[4],
                    "price_target_upside": r[5],
                    "thesis_summary": r[6],
                    "entry_date": r[7],
                    "tenure_weeks": r[8],
                }
                for r in rows
            ]
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return []

    def log_rotation_event(self, event: dict, run_date: str) -> None:
        """Log a population rotation event to the population_history table."""
        if not self.db_conn:
            return
        self.db_conn.execute(
            """INSERT INTO population_history
               (date, event_type, ticker_in, ticker_out, sector, reason,
                score_in, score_out)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_date,
                event.get("type", "UNKNOWN"),
                event.get("ticker_in", event.get("ticker")),
                event.get("ticker_out"),
                event.get("sector", "Unknown"),
                event.get("reason", ""),
                event.get("score_in", event.get("long_term_score")),
                event.get("score_out"),
            ),
        )

    # ── Stock Archive + Thesis History + Analyst Resources ─────────────────────

    def save_stock_archive(
        self, ticker: str, sector: str, team_id: str, run_date: str,
        status: str = "active",
    ) -> None:
        """Upsert the stock_archive record (UPDATE-then-INSERT for Lambda compat)."""
        conn = self.db_conn
        cur = conn.execute(
            "UPDATE stock_archive SET last_analyzed=?, current_status=? WHERE ticker=?",
            (run_date, status, ticker),
        )
        if cur.rowcount == 0:
            conn.execute(
                """INSERT INTO stock_archive
                   (ticker, sector, sector_team, first_analyzed, last_analyzed,
                    times_in_population, current_status)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (ticker, sector, team_id, run_date, run_date, status),
            )
        conn.commit()

    def increment_population_count(self, ticker: str) -> None:
        """Increment times_in_population for a stock entering the population."""
        self.db_conn.execute(
            "UPDATE stock_archive SET times_in_population = times_in_population + 1 WHERE ticker=?",
            (ticker,),
        )
        self.db_conn.commit()

    def save_thesis_history(
        self,
        ticker: str,
        run_date: str,
        author: str,
        thesis_type: str,
        thesis: dict,
    ) -> None:
        """Save a thesis history record. Author format: 'team:technology', 'ic:cio', etc."""
        self.db_conn.execute(
            """INSERT INTO thesis_history
               (ticker, run_date, author, thesis_type, bull_case, bear_case,
                catalysts, risks, conviction, score, rationale)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                run_date,
                author,
                thesis_type,
                thesis.get("bull_case", ""),
                thesis.get("bear_case", ""),
                json.dumps(thesis.get("catalysts", [])),
                json.dumps(thesis.get("risks", [])),
                thesis.get("conviction"),
                thesis.get("score"),
                thesis.get("rationale", ""),
            ),
        )
        self.db_conn.commit()

    def save_ic_decision(self, run_date: str, decision: dict) -> None:
        """Save an IC decision as a thesis_history record."""
        self.save_thesis_history(
            ticker=decision["ticker"],
            run_date=run_date,
            author="ic:cio",
            thesis_type=f"ic_{decision.get('decision', 'unknown').lower()}",
            thesis={
                "bull_case": decision.get("team_recommendation", ""),
                "bear_case": "",
                "catalysts": [],
                "risks": [],
                "conviction": decision.get("conviction"),
                "score": decision.get("score"),
                "rationale": decision.get("cio_rationale", ""),
            },
        )

    def load_stock_history(self, ticker: str) -> list[dict]:
        """Load all thesis_history records for a ticker (for re-entry context)."""
        cur = self.db_conn.execute(
            "SELECT * FROM thesis_history WHERE ticker=? ORDER BY run_date DESC",
            (ticker,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def save_analyst_resource(
        self,
        ticker: str,
        run_date: str,
        agent: str,
        resource_type: str,
        resource_detail: str = "",
        influence: str = "supporting",
    ) -> None:
        """Track which data sources an agent used for a given ticker."""
        self.db_conn.execute(
            """INSERT INTO analyst_resources
               (ticker, run_date, agent, resource_type, resource_detail, influence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticker, run_date, agent, resource_type, resource_detail, influence),
        )
        self.db_conn.commit()

    # ── Memory: Episodic (Phase 2) ───────────────────────────────────────────

    def load_episodic_memories(
        self,
        tickers: list[str],
        sectors: list[str],
        max_per_ticker: int = 3,
        max_age_weeks: int = 12,
    ) -> dict[str, list[dict]]:
        """Load episodic memories for tickers (exact match) and sectors (related stocks)."""
        import json as _json
        from datetime import date, timedelta
        cutoff = str(date.today() - timedelta(weeks=max_age_weeks))

        result: dict[str, list[dict]] = {}

        # Exact ticker matches
        for ticker in tickers:
            rows = self.db_conn.execute(
                "SELECT ticker, signal_date, score, conviction, thesis_summary, "
                "outcome_10d, outcome_vs_spy, lesson, sector, pattern_tags "
                "FROM memory_episodes WHERE ticker = ? AND created_date >= ? "
                "ORDER BY signal_date DESC LIMIT ?",
                (ticker, cutoff, max_per_ticker),
            ).fetchall()
            if rows:
                result[ticker] = [
                    {"ticker": r[0], "signal_date": r[1], "score": r[2],
                     "conviction": r[3], "thesis_summary": r[4],
                     "outcome_10d": r[5], "outcome_vs_spy": r[6],
                     "lesson": r[7], "sector": r[8], "pattern_tags": r[9]}
                    for r in rows
                ]

        # Sector-level memories (for stocks not in exact match)
        for sector in sectors:
            rows = self.db_conn.execute(
                "SELECT ticker, signal_date, score, conviction, thesis_summary, "
                "outcome_10d, outcome_vs_spy, lesson, sector, pattern_tags "
                "FROM memory_episodes WHERE sector = ? AND created_date >= ? "
                "ORDER BY signal_date DESC LIMIT 5",
                (sector, cutoff),
            ).fetchall()
            for r in rows:
                t = r[0]
                if t not in result:
                    result.setdefault(t, []).append(
                        {"ticker": r[0], "signal_date": r[1], "score": r[2],
                         "conviction": r[3], "thesis_summary": r[4],
                         "outcome_10d": r[5], "outcome_vs_spy": r[6],
                         "lesson": r[7], "sector": r[8], "pattern_tags": r[9]}
                    )

        # Prune old memories
        self.db_conn.execute("DELETE FROM memory_episodes WHERE created_date < ?", (cutoff,))
        self.db_conn.commit()

        return result

    # ── Memory: Semantic (Phase 3) ─────────────────────────────────────────

    def load_semantic_memories(
        self,
        sectors: list[str],
        max_age_weeks: int = 8,
        max_per_sector: int = 5,
    ) -> dict[str, list[dict]]:
        """Load semantic memories by sector. Auto-prunes stale entries."""
        from datetime import date, timedelta
        cutoff = str(date.today() - timedelta(weeks=max_age_weeks))

        # Prune stale memories
        self.db_conn.execute("DELETE FROM memory_semantic WHERE created_date < ?", (cutoff,))
        self.db_conn.commit()

        result: dict[str, list[dict]] = {}
        for sector in sectors:
            rows = self.db_conn.execute(
                "SELECT category, source, content, sector, related_tickers, "
                "created_date, reinforced_date "
                "FROM memory_semantic WHERE sector = ? "
                "ORDER BY reinforced_date DESC, created_date DESC LIMIT ?",
                (sector, max_per_sector),
            ).fetchall()
            if rows:
                result[sector] = [
                    {"category": r[0], "source": r[1], "content": r[2],
                     "sector": r[3], "related_tickers": r[4],
                     "created_date": r[5], "reinforced_date": r[6]}
                    for r in rows
                ]

        # Also load cross-sector memories
        cross_rows = self.db_conn.execute(
            "SELECT category, source, content, sector, related_tickers, "
            "created_date, reinforced_date "
            "FROM memory_semantic WHERE category = 'cross_sector' "
            "ORDER BY created_date DESC LIMIT 5"
        ).fetchall()
        if cross_rows:
            result["_cross_sector"] = [
                {"category": r[0], "source": r[1], "content": r[2],
                 "sector": r[3], "related_tickers": r[4],
                 "created_date": r[5], "reinforced_date": r[6]}
                for r in cross_rows
            ]

        return result

    def save_semantic_memory(
        self,
        category: str,
        source: str,
        content: str,
        sector: str | None,
        related_tickers: list[str] | None,
        run_date: str,
    ) -> bool:
        """Save a semantic memory. Returns True if new, False if duplicate."""
        import json as _json
        tickers_json = _json.dumps(related_tickers) if related_tickers else None
        try:
            self.db_conn.execute(
                "INSERT INTO memory_semantic "
                "(category, source, content, sector, related_tickers, created_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (category, source, content, sector, tickers_json, run_date),
            )
            self.db_conn.commit()
            return True
        except Exception:
            # Likely duplicate — update reinforced_date instead
            self.db_conn.execute(
                "UPDATE memory_semantic SET reinforced_date = ? "
                "WHERE category = ? AND source = ? AND content = ?",
                (run_date, category, source, content),
            )
            self.db_conn.commit()
            return False

    def close(self) -> None:
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
