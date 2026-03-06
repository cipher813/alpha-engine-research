"""
Run the scanner pipeline in isolation for testing and debugging.
Calls the production run_scanner() function directly — no duplicated logic.

Usage:
  python local/run_scanner.py
  python local/run_scanner.py --regime caution
  python local/run_scanner.py --tickers 150   # limit for speed
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from archive.manager import ArchiveManager
from config import UNIVERSE_TICKERS, ROTATION_TIERS, WEAK_PICK_SCORE_THRESHOLD, \
    WEAK_PICK_CONSECUTIVE_RUNS, EMERGENCY_ROTATION_NEW_SCORE, SECTOR_MAP
from data.fetchers.price_fetcher import fetch_price_data
from data.scanner import get_scanner_universe, evaluate_candidate_rotation
from graph.research_graph import run_scanner


def main():
    parser = argparse.ArgumentParser(description="Run scanner pipeline in isolation")
    parser.add_argument("--regime", default="caution",
                        choices=["bull", "neutral", "caution", "bear"])
    parser.add_argument("--tickers", type=int, default=0,
                        help="Limit scanner universe to N tickers (0 = all)")
    args = parser.parse_args()

    run_date = str(datetime.date.today())

    # FMP API key check
    fmp_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_key:
        print("WARNING: FMP_API_KEY not set — analyst data will be empty")
    else:
        print(f"FMP_API_KEY loaded (len={len(fmp_key)})")
        from data.fetchers.analyst_fetcher import fetch_analyst_consensus
        t = fetch_analyst_consensus("AAPL")
        print(f"FMP test (AAPL): consensus={t['consensus_rating']} target={t['mean_target']} analysts={t['num_analysts']}")

    # Archive
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive = ArchiveManager(local_db_path=os.path.join(project_root, "research.db"))
    try:
        archive.download_db()
    except Exception as e:
        print(f"S3 download failed ({e}). Using local DB.")
        archive.db_conn = sqlite3.connect(archive.local_db_path)
        archive.db_conn.row_factory = sqlite3.Row
        archive._ensure_schema()

    active_candidates = archive.load_active_candidates()
    print(f"Active candidates in DB: {[c['symbol'] for c in active_candidates]}")

    # Build scanner universe and fetch price data
    all_tracked = list(set(UNIVERSE_TICKERS + [c["symbol"] for c in active_candidates]))
    scanner_universe = get_scanner_universe(exclude_tickers=all_tracked)
    if args.tickers:
        scanner_universe = scanner_universe[:args.tickers]

    print(f"\nFetching price data for {len(scanner_universe)} tickers...")
    price_data = fetch_price_data(scanner_universe, period="1y")

    # Call prod scanner function
    result = run_scanner(
        run_date=run_date,
        scanner_universe=scanner_universe,
        price_data=price_data,
        archive=archive,
        market_regime=args.regime,
    )

    # Compute composite scores (normally done by score_aggregator in full pipeline)
    scanner_scores = result["scanner_scores"]
    for ticker, sdata in scanner_scores.items():
        base = sdata["tech_score"] * 0.40 + sdata["news_score"] * 0.30 + sdata["research_score"] * 0.30
        sdata["score"] = round(max(0.0, min(100.0, base)), 2)

    # Stage 5: rotation / slot fill
    new_active, rotations = evaluate_candidate_rotation(
        scanner_scores=scanner_scores,
        active_candidates=active_candidates,
        rotation_tiers=ROTATION_TIERS,
        weak_pick_score_threshold=WEAK_PICK_SCORE_THRESHOLD,
        weak_pick_consecutive_runs=WEAK_PICK_CONSECUTIVE_RUNS,
        emergency_rotation_new_score=EMERGENCY_ROTATION_NEW_SCORE,
        run_date=run_date,
    )

    print(f"\n{'='*50}")
    print(f"FINAL BUY CANDIDATES ({len(new_active)}):")
    for c in new_active:
        score = scanner_scores.get(c["symbol"], {}).get("score") or c.get("score", 0)
        print(f"  {c['symbol']} | score={score:.1f} | slot={c['slot']} | entry={c['entry_date']}")
    if rotations:
        print("\nRotation events:")
        for r in rotations:
            print(f"  {r['reason']}: {r.get('out_ticker', 'EMPTY')} → {r['in_ticker']}")

    archive.close()


if __name__ == "__main__":
    main()
