"""Tests for rag.pipelines._signals_resolver.

The resolver must:
  - pick the lexicographically-latest prefix that *contains signals.json*
  - iterate backward past prefixes that have no signals.json (namespace pollution)
  - raise RuntimeError if no prefix contains signals.json (no silent fallback)
  - raise RuntimeError if no signals/ prefixes exist at all
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _mk_s3_mock(prefixes_with_signals: dict[str, dict | None]):
    """Build a mock boto3 s3 client.

    prefixes_with_signals maps "signals/{date}/" -> parsed signals.json dict,
    or None to indicate the prefix exists but has no signals.json (NoSuchKey).
    """
    s3 = MagicMock()

    class NoSuchKey(Exception):
        pass

    s3.exceptions.NoSuchKey = NoSuchKey

    s3.list_objects_v2.return_value = {
        "CommonPrefixes": [{"Prefix": p} for p in prefixes_with_signals],
    }

    def _get_object(Bucket, Key):
        for prefix, data in prefixes_with_signals.items():
            if Key == f"{prefix}signals.json":
                if data is None:
                    raise NoSuchKey(f"simulated NoSuchKey for {Key}")
                body = MagicMock()
                body.read.return_value = json.dumps(data).encode()
                return {"Body": body}
        raise NoSuchKey(f"unexpected Key {Key}")

    s3.get_object.side_effect = _get_object
    return s3


def test_resolver_picks_latest_prefix_with_signals():
    """When latest prefix has no signals.json, fall through to the most recent one that does."""
    from rag.pipelines import _signals_resolver

    s3 = _mk_s3_mock({
        "signals/2026-04-12/": {"universe": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]},
        "signals/2026-04-13/": None,
        "signals/2026-04-14/": None,
        "signals/2026-04-15/": None,
        "signals/2026-04-16/": None,
        "signals/2026-04-17/": None,
    })

    with patch("boto3.client", return_value=s3):
        tickers = _signals_resolver.load_tickers_from_latest_signals()

    assert tickers == ["AAPL", "MSFT"]


def test_resolver_uses_latest_when_all_good():
    from rag.pipelines import _signals_resolver

    s3 = _mk_s3_mock({
        "signals/2026-04-11/": {"universe": [{"ticker": "OLD"}]},
        "signals/2026-04-12/": {"universe": [{"ticker": "NEW"}]},
    })

    with patch("boto3.client", return_value=s3):
        tickers = _signals_resolver.load_tickers_from_latest_signals()

    assert tickers == ["NEW"]


def test_resolver_raises_when_no_prefixes():
    from rag.pipelines import _signals_resolver

    s3 = _mk_s3_mock({})

    with patch("boto3.client", return_value=s3):
        try:
            _signals_resolver.load_tickers_from_latest_signals()
        except RuntimeError as e:
            assert "No signals/ prefixes" in str(e)
            return
    assert False, "expected RuntimeError"


def test_resolver_raises_when_no_prefix_has_signals_json():
    """No silent fallback: if no prefix has signals.json, hard-fail."""
    from rag.pipelines import _signals_resolver

    s3 = _mk_s3_mock({
        "signals/2026-04-13/": None,
        "signals/2026-04-14/": None,
    })

    with patch("boto3.client", return_value=s3):
        try:
            _signals_resolver.load_tickers_from_latest_signals()
        except RuntimeError as e:
            assert "No signals.json found" in str(e)
            return
    assert False, "expected RuntimeError"


def test_resolver_drops_entries_missing_ticker_field():
    from rag.pipelines import _signals_resolver

    s3 = _mk_s3_mock({
        "signals/2026-04-12/": {"universe": [{"ticker": "AAPL"}, {"score": 0.5}, {"ticker": None}, {"ticker": "GOOG"}]},
    })

    with patch("boto3.client", return_value=s3):
        tickers = _signals_resolver.load_tickers_from_latest_signals()

    assert tickers == ["AAPL", "GOOG"]
