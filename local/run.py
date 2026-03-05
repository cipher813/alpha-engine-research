"""
Local test runner — bypasses Lambda for development and testing.

Usage:
  python local/run.py                    # run today's pipeline
  python local/run.py --date 2026-03-05  # run for a specific date
  python local/run.py --dry-run          # skip email, skip S3 write

Requires environment variables:
  ANTHROPIC_API_KEY
  FMP_API_KEY
  FRED_API_KEY
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (for S3)
  S3_BUCKET (default: alpha-engine-research)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run alpha-engine-research pipeline locally")
    parser.add_argument("--date", type=str, default=None,
                        help="Run date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip email delivery and S3 upload. Print report to stdout.")
    parser.add_argument("--skip-scanner", action="store_true",
                        help="Skip scanner pipeline (Branch B) for faster testing.")
    args = parser.parse_args()

    run_date = args.date or str(datetime.date.today())

    print(f"alpha-engine-research local run — {run_date}")
    if args.dry_run:
        print("DRY RUN: email and S3 writes disabled")

    # Check trading day (use importlib: 'lambda' is a reserved keyword)
    import importlib
    _handler = importlib.import_module("lambda.handler")
    is_trading_day = _handler.is_trading_day
    is_early_close = _handler.is_early_close
    d = datetime.datetime.strptime(run_date, "%Y-%m-%d").date()

    if not is_trading_day(d):
        print(f"NOTE: {run_date} is not an NYSE trading day. Running anyway for testing.")

    early_close = is_early_close(d)

    # Set up archive (use project-root research.db so sync_db push can find it)
    from archive.manager import ArchiveManager
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_db = os.path.join(project_root, "research.db")
    archive = ArchiveManager(local_db_path=local_db)
    print("Downloading research.db from S3...")
    try:
        archive.download_db()
    except Exception as e:
        print(f"S3 download failed ({e}). Starting with empty DB.")
        import sqlite3
        archive.db_conn = sqlite3.connect(archive.local_db_path)
        archive.db_conn.row_factory = sqlite3.Row
        archive._ensure_schema()

    # Performance tracker
    from scoring.performance_tracker import run_performance_checks
    perf_summary = run_performance_checks(archive.db_conn, run_date)
    print(f"Performance summary: {perf_summary}")

    # Build and run graph
    from graph.research_graph import build_graph, create_initial_state
    graph = build_graph()
    state = create_initial_state(
        run_date=run_date,
        archive_manager=archive,
        is_early_close=early_close,
    )
    state["performance_summary"] = perf_summary

    if args.dry_run:
        # Monkey-patch email sender to print instead
        import emailer.sender as sender_mod
        sender_mod.send_email = lambda **kwargs: (print("\n=== EMAIL BODY ===\n" + kwargs.get("plain_body", "")), True)[1]

    if args.skip_scanner:
        # Monkey-patch scanner to return empty
        import graph.research_graph as graph_mod
        original_scanner = graph_mod.run_scanner_pipeline
        def _skip_scanner(state):
            print("  [scanner skipped]")
            return {**state, "scanner_filtered": [], "scanner_ranked": [],
                    "scanner_news_reports": {}, "scanner_research_reports": {},
                    "scanner_scores": {}}
        graph_mod.run_scanner_pipeline = _skip_scanner

    print("Running pipeline...")
    final_state = graph.invoke(state)

    print("\n=== RUN COMPLETE ===")
    print(f"Email sent: {final_state.get('email_sent', False)}")
    print(f"Tickers processed: {len(final_state.get('investment_theses', {}))}")

    print("\n=== CONSOLIDATED REPORT ===")
    print(final_state.get("consolidated_report", ""))

    archive.close()


if __name__ == "__main__":
    main()
