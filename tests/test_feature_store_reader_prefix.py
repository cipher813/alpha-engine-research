"""Regression test pinning the staging/daily_closes/ prefix in
``feature_store_reader.read_latest_daily_closes``.

Coordinated with alpha-engine-data PR #112 which migrated the writer
+ in-repo readers from ``predictor/daily_closes/`` →
``staging/daily_closes/``. Hard-cutover, no fallback (per
``feedback_no_silent_fails``) — the reader's list_objects_v2 call must
target the new prefix exactly.

Called from ``graph/research_graph.py:222-223`` in the live Saturday SF
Research Lambda; a regression here would silently scan an empty prefix
(after the 7-day staging lifecycle eats the legacy parquets) and return
None, falling back to the yfinance batch fetch path.
"""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return (Path(__file__).parent.parent / path).read_text()


def test_read_latest_daily_closes_uses_staging_prefix():
    src = _read("data/fetchers/feature_store_reader.py")
    assert 'Prefix="staging/daily_closes/"' in src, (
        "feature_store_reader.read_latest_daily_closes is not scanning "
        "staging/daily_closes/. The 2026-04-29 prefix migration "
        "(alpha-engine-data PR #112) requires this exact prefix. "
        "Hard-cutover; no fallback to legacy predictor/daily_closes/."
    )


def test_no_legacy_prefix_in_feature_store_reader():
    """Belt-and-suspenders: forbid the legacy string anywhere in the file.
    Catches a regression that flips the cutover back to the predictor/
    namespace under any name (string concat, f-string, comment-then-uncomment)."""
    src = _read("data/fetchers/feature_store_reader.py")
    assert "predictor/daily_closes" not in src, (
        "feature_store_reader.py contains 'predictor/daily_closes' — "
        "the prefix was migrated to staging/. No fallback per "
        "feedback_no_silent_fails."
    )
