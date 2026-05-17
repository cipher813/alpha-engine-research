"""Tests for get_balance_sheet @tool — fundamentals S3 read (PR2).

yfinance-centralization arc (plan doc:
``alpha-engine-docs/private/yfinance-centralization-260516.md``, item R3 / PR 2).

The tool now reads alpha-engine-data's weekly Finnhub fundamentals snapshot
(``archive/fundamentals/{date}.json``, the data-module's normalized/clipped
schema) instead of ``yfinance.Ticker().info``. The fundamentals dict is
injected via the ``fundamentals_data`` context key (closed over at
tool-creation, mirroring ``factor_profiles``), so these tests use fakes
+ ``monkeypatch`` — NOT ``unittest.mock.patch`` (documented full-suite
bleed in this repo; mirrors ``tests/test_held_thesis_strict.py`` style).

Contract preserved: exact return-schema keys + ``{"error": ...}``
graceful-degrade on a missing ticker/snapshot (the quant agent consumes
this as soft context — it must never raise under all-agents-strict).
"""

from __future__ import annotations

import json

import pandas as pd

from agents.sector_teams import quant_tools
from agents.sector_teams.quant_tools import create_quant_tools

# Data-module normalized/clipped schema (collectors/fundamentals.py output).
_FAKE_FUNDAMENTALS = {
    "NVDA": {
        "pe_ratio": 1.8,
        "pb_ratio": 2.1,
        "debt_to_equity": 0.15,
        "revenue_growth_yoy": 0.62,
        "fcf_yield": 0.03,
        "gross_margin": 0.74,
        "roe": 0.55,
        "current_ratio": 1.2,
    },
}


def _tools(fundamentals_data, tickers=("NVDA", "AAPL")):
    """create_quant_tools with fundamentals injected via context (no S3)."""
    price_data = {t: pd.DataFrame() for t in tickers}
    return create_quant_tools(
        {
            "price_data": price_data,
            "technical_scores": {},
            "factor_profiles": {},
            "fundamentals_data": fundamentals_data,
        }
    )


def _find_tool(tools, name):
    return next(t for t in tools if t.name == name)


# ── Mapped schema from a faked fundamentals JSON ─────────────────────────────


def test_get_balance_sheet_in_tools_list():
    tools = _tools(_FAKE_FUNDAMENTALS)
    assert "get_balance_sheet" in [t.name for t in tools]


def test_returns_mapped_schema_from_fundamentals():
    tools = _tools(_FAKE_FUNDAMENTALS)
    tool = _find_tool(tools, "get_balance_sheet")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    entry = result["NVDA"]

    # Exact return-schema keys preserved (the quant agent's contract).
    assert set(entry.keys()) == {
        "debt_to_equity",
        "current_ratio",
        "market_cap",
        "pe_ratio",
        "forward_pe",
        "price_to_book",
        "revenue_growth",
        "gross_margins",
    }
    # Finnhub-normalized → tool schema mapping.
    assert entry["debt_to_equity"] == 0.15  # already a ratio — NO %/100 scaling
    assert entry["current_ratio"] == 1.2
    assert entry["pe_ratio"] == 1.8
    assert entry["price_to_book"] == 2.1  # pb_ratio → price_to_book
    assert entry["revenue_growth"] == 0.62  # revenue_growth_yoy
    assert entry["gross_margins"] == 0.74  # gross_margin
    # Not persisted by the collector — None per the optional-field contract.
    assert entry["forward_pe"] is None
    assert entry["market_cap"] is None


def test_de_not_rescaled_by_pct_over_100():
    """D/E from Finnhub is already a ratio — the yfinance %/100 scaling
    must NOT be carried over."""
    tools = _tools({"NVDA": {"debt_to_equity": 1.5}})
    tool = _find_tool(tools, "get_balance_sheet")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    assert result["NVDA"]["debt_to_equity"] == 1.5  # not 0.015


# ── Graceful-degrade (must never raise — all-agents-strict) ──────────────────


def test_missing_ticker_degrades_to_error_dict():
    """A ticker absent from the snapshot returns {"error": ...}, no raise."""
    tools = _tools(_FAKE_FUNDAMENTALS, tickers=("NVDA", "AAPL"))
    tool = _find_tool(tools, "get_balance_sheet")
    result = json.loads(tool.invoke({"tickers": ["AAPL"]}))
    assert "AAPL" in result
    assert "error" in result["AAPL"]
    assert "debt_to_equity" not in result["AAPL"]


def test_empty_snapshot_degrades_no_raise():
    """An empty/missing snapshot degrades per-ticker, never raises."""
    tools = _tools({}, tickers=("NVDA",))
    tool = _find_tool(tools, "get_balance_sheet")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    assert "error" in result["NVDA"]


def test_no_yfinance_import_in_module():
    """The module is yfinance-free post-PR2."""
    import inspect

    src = inspect.getsource(quant_tools)
    assert "import yfinance" not in src
    assert "yf.Ticker" not in src


# ── S3 reader fallback/date-resolution (monkeypatched boto, no network) ───────


def test_read_fundamentals_from_s3_scans_for_latest(monkeypatch):
    """When run_date isn't given, the reader scans the prefix and reads
    the most-recent snapshot (mirrors predictor's fetch_alt_data pattern)."""
    captured = {}

    class _FakeBody:
        def read(self):
            return json.dumps(_FAKE_FUNDAMENTALS).encode()

    class _FakeS3:
        def list_objects_v2(self, **kw):
            captured["prefix"] = kw.get("Prefix")
            return {
                "Contents": [
                    {"Key": "archive/fundamentals/2026-05-09.json"},
                    {"Key": "archive/fundamentals/2026-05-16.json"},
                    {"Key": "archive/fundamentals/_index.txt"},
                ]
            }

        def get_object(self, **kw):
            captured["key"] = kw["Key"]
            return {"Body": _FakeBody()}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeS3())
    out = quant_tools.read_fundamentals_from_s3()
    assert out == _FAKE_FUNDAMENTALS
    assert captured["prefix"] == "archive/fundamentals/"
    # Most-recent (reverse-sorted) .json selected, non-json ignored.
    assert captured["key"] == "archive/fundamentals/2026-05-16.json"


def test_read_fundamentals_from_s3_returns_none_on_failure(monkeypatch):
    """Any S3 failure → None (caller graceful-degrades, never raises)."""

    class _BoomS3:
        def list_objects_v2(self, **kw):
            raise RuntimeError("s3 down")

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _BoomS3())
    assert quant_tools.read_fundamentals_from_s3() is None
