"""
Offline stubs — monkey-patches all external calls (LLM, data APIs, S3, email)
so the research pipeline can run end-to-end without any network access.

Usage:
    from local.offline_stubs import install_offline_stubs
    install_offline_stubs()   # call BEFORE importing graph modules

Generates synthetic but structurally valid data so every graph node
receives the dict shapes it expects.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Synthetic price data ─────────────────────────────────────────────────────

_SAMPLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "PG", "HD", "MA", "XOM", "ABBV", "LLY", "COST", "MRK", "PEP",
    "CVX", "KO", "AVGO", "TMO", "WMT", "MCD", "CSCO", "ACN", "ABT", "CRM",
]

_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Communication Services",
    "TSLA": "Consumer Discretionary", "JPM": "Financial", "V": "Financial", "UNH": "Healthcare",
    "JNJ": "Healthcare", "PG": "Consumer Staples", "HD": "Consumer Discretionary",
    "MA": "Financial", "XOM": "Energy", "ABBV": "Healthcare", "LLY": "Healthcare",
    "COST": "Consumer Staples", "MRK": "Healthcare", "PEP": "Consumer Staples",
    "CVX": "Energy", "KO": "Consumer Staples", "AVGO": "Technology", "TMO": "Healthcare",
    "WMT": "Consumer Staples", "MCD": "Consumer Discretionary", "CSCO": "Technology",
    "ACN": "Technology", "ABT": "Healthcare", "CRM": "Technology",
}


def _synthetic_ohlcv(ticker: str, days: int = 252) -> pd.DataFrame:
    """Generate a synthetic 1-year daily OHLCV DataFrame."""
    rng = np.random.default_rng(hash(ticker) % (2**31))
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    base = rng.uniform(50, 500)
    returns = rng.normal(0.0005, 0.015, size=days)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.02, size=days))
    low = close * (1 - rng.uniform(0, 0.02, size=days))
    opn = low + (high - low) * rng.uniform(0.3, 0.7, size=days)
    volume = rng.integers(500_000, 20_000_000, size=days)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


# ── Stub functions ───────────────────────────────────────────────────────────

def _stub_fetch_price_data(tickers, period="1y"):
    logger.info("[offline] stub fetch_price_data: %d tickers", len(tickers))
    return {t: _synthetic_ohlcv(t) for t in tickers}


def _stub_fetch_sp500_sp400():
    logger.info("[offline] stub fetch_sp500_sp400_with_sectors: %d tickers", len(_SAMPLE_TICKERS))
    return list(_SAMPLE_TICKERS), dict(_SECTOR_MAP)


def _stub_fetch_short_interest(tickers):
    logger.info("[offline] stub fetch_short_interest: %d tickers", len(tickers))
    rng = random.Random(42)
    return {t: {"short_pct_float": rng.uniform(1, 15), "short_ratio": rng.uniform(1, 5)} for t in tickers}


def _stub_fetch_all_news(ticker, hours=48):
    logger.info("[offline] stub fetch_all_news: %s", ticker)
    return {
        "yahoo": [
            {"headline": f"[STUB] {ticker} reports strong quarterly results",
             "source": "Yahoo Finance", "published": datetime.now().isoformat(),
             "url": "https://example.com", "excerpt": f"Synthetic news article for {ticker} dry run."},
            {"headline": f"[STUB] Analysts upgrade {ticker} outlook",
             "source": "Yahoo Finance", "published": datetime.now().isoformat(),
             "url": "https://example.com", "excerpt": f"Synthetic upgrade article for {ticker}."},
        ],
        "edgar_8k": [],
    }


def _stub_fetch_analyst_consensus(ticker, current_price=None):
    logger.info("[offline] stub fetch_analyst_consensus: %s", ticker)
    rng = random.Random(hash(ticker))
    price = current_price or rng.uniform(50, 500)
    target = price * rng.uniform(0.9, 1.3)
    return {
        "ticker": ticker,
        "consensus_rating": rng.choice(["Strong Buy", "Buy", "Hold"]),
        "num_analysts": rng.randint(5, 30),
        "mean_target": round(target, 2),
        "current_price": round(price, 2),
        "upside_pct": round((target / price - 1) * 100, 1),
        "rating_changes": "None recent",
        "earnings_surprise": "+2.5%",
    }


def _stub_fetch_macro_data():
    logger.info("[offline] stub fetch_macro_data")
    return {
        "fed_funds": 5.25, "t2yr": 4.60, "t10yr": 4.20,
        "curve_slope": -40, "vix": 16.5,
        "spy_30d": 2.1, "qqq_30d": 3.4, "iwm_30d": 1.2,
        "oil": 78.50, "gold": 2350.0, "copper": 4.25,
        "cpi_yoy": 3.1, "unemployment": 3.9,
        "consumer_sentiment": 67.8, "initial_claims": 215,
        "hy_oas": 340,
        "upcoming_releases": "CPI (next week), FOMC (in 2 weeks)",
    }


def _stub_compute_market_breadth(price_data):
    return {"pct_above_50d": 62.5, "pct_above_200d": 58.0, "adv_dec_ratio": 1.3}


def _stub_fetch_revisions(tickers, reference_date=None):
    logger.info("[offline] stub fetch_revisions: %d tickers", len(tickers))
    return {}


def _stub_fetch_options_signals(tickers, reference_date=None):
    logger.info("[offline] stub fetch_options_signals: %d tickers", len(tickers))
    return {}


def _stub_cache_options_to_s3(data, date):
    pass


def _stub_fetch_insider_activity(tickers, lookback_days=90, reference_date=None):
    logger.info("[offline] stub fetch_insider_activity: %d tickers", len(tickers))
    return {}


def _stub_cache_insider_to_s3(data, date):
    pass


def _stub_fetch_institutional_accumulation(tickers):
    logger.info("[offline] stub fetch_institutional_accumulation: %d tickers", len(tickers))
    return {}


# ── LLM agent stubs ─────────────────────────────────────────────────────────

def _stub_run_news_agent(ticker, company_name, prior_report, prior_date,
                         new_articles, recurring_themes=None, sec_filings=None,
                         current_price=0, price_change_pct=0, price_change_date="",
                         **kwargs):
    logger.info("[offline] stub run_news_agent: %s", ticker)
    rng = random.Random(hash(ticker) + 1)
    score_st = rng.randint(40, 75)
    score_lt = rng.randint(40, 75)
    return {
        "ticker": ticker,
        "report_md": f"[OFFLINE STUB] News report for {ticker}. Sentiment is moderately positive based on synthetic data.",
        "news_json": {
            "news_score_short": score_st,
            "news_score_long": score_lt,
            "sentiment": "moderately_positive",
            "material_changes": False,
        },
        "news_score": float(score_st),
        "news_score_lt": float(score_lt),
    }


def _stub_run_research_agent(ticker, company_name, prior_report, prior_date,
                             analyst_data, insider_summary="", **kwargs):
    logger.info("[offline] stub run_research_agent: %s", ticker)
    rng = random.Random(hash(ticker) + 2)
    score_st = rng.randint(40, 75)
    score_lt = rng.randint(40, 75)
    return {
        "ticker": ticker,
        "report_md": f"[OFFLINE STUB] Research report for {ticker}. Analyst consensus is moderately bullish.",
        "research_json": {
            "research_score_short": score_st,
            "research_score_long": score_lt,
            "consensus_direction": "moderately_bullish",
            "material_changes": False,
        },
        "research_score": float(score_st),
        "research_score_lt": float(score_lt),
    }


def _stub_run_macro_agent_with_reflection(prior_report, prior_date, macro_data,
                                          max_iterations=2, api_key=None, **kwargs):
    logger.info("[offline] stub run_macro_agent_with_reflection")
    from config import ALL_SECTORS
    return {
        "report_md": "[OFFLINE STUB] Macro environment is neutral. Fed on hold, yields stable.",
        "macro_json": {
            "market_regime": "neutral",
            "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
            "sector_ratings": {s: {"rating": "market_weight", "rationale": "Synthetic data"} for s in ALL_SECTORS},
        },
        "market_regime": "neutral",
        "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
        "sector_ratings": {s: {"rating": "market_weight", "rationale": "Synthetic data"} for s in ALL_SECTORS},
    }


def _stub_run_scanner_ranking_agent(candidates, market_regime="neutral",
                                    api_key=None, top_n=10):
    logger.info("[offline] stub run_scanner_ranking_agent: %d candidates, top_n=%d", len(candidates), top_n)
    ranked = []
    for i, c in enumerate(candidates[:top_n]):
        ranked.append({
            "rank": i + 1,
            "ticker": c.get("ticker", ""),
            "path": c.get("path", "momentum"),
            "rationale": f"[OFFLINE] Ranked #{i+1} based on synthetic quant scores",
        })
    return ranked


def _stub_run_consolidator_agent(run_date, macro_report, universe_news_reports,
                                 universe_research_reports, candidate_full_news,
                                 candidate_full_research, investment_theses,
                                 active_candidates, performance_summary,
                                 sector_ratings=None, is_early_close=False,
                                 api_key=None, **kwargs):
    logger.info("[offline] stub run_consolidator_agent")
    n_theses = len(investment_theses)
    n_candidates = len(active_candidates)
    return (
        f"# Alpha Engine Research — {run_date} [OFFLINE DRY RUN]\n\n"
        f"**This report was generated with synthetic data (no API/LLM calls).**\n\n"
        f"## Summary\n"
        f"- Tickers analyzed: {n_theses}\n"
        f"- Buy candidates: {n_candidates}\n"
        f"- Market regime: neutral (synthetic)\n\n"
        f"## Macro\n{macro_report}\n\n"
        f"## Population\n"
        + "\n".join(f"- {t}: score {th.get('final_score', 'N/A')}" for t, th in list(investment_theses.items())[:10])
    )




def _stub_run_macro_agent(prior_report, prior_date, macro_data, api_key=None, **kwargs):
    return _stub_run_macro_agent_with_reflection(prior_report, prior_date, macro_data)


# ── Sector team / CIO stubs ──────────────────────────────────────────────

def _stub_run_quant_analyst(team_id, sector_tickers, market_regime, price_data,
                            technical_scores, run_date, api_key=None, **kwargs):
    logger.info("[offline] stub run_quant_analyst: %s (%d tickers)", team_id, len(sector_tickers))
    rng = random.Random(hash(team_id))
    picks = []
    for i, t in enumerate(sector_tickers[:10]):
        picks.append({
            "ticker": t,
            "quant_score": rng.randint(45, 85),
            "rationale": f"[OFFLINE] Synthetic quant score for {t}",
            "key_metrics": {"rsi": rng.randint(30, 70), "momentum_20d": round(rng.uniform(-5, 10), 1)},
        })
    return {"team_id": team_id, "ranked_picks": picks, "tool_calls": [], "iterations": 0}


def _stub_run_qual_analyst(team_id, quant_top5, prior_theses, market_regime,
                           run_date, api_key=None, price_data=None, **kwargs):
    logger.info("[offline] stub run_qual_analyst: %s (%d picks)", team_id, len(quant_top5))
    rng = random.Random(hash(team_id) + 10)
    assessments = []
    for pick in quant_top5:
        assessments.append({
            "ticker": pick.get("ticker", ""),
            "qual_score": rng.randint(45, 85),
            "bull_case": "[OFFLINE] Strong fundamentals",
            "bear_case": "[OFFLINE] Valuation risk",
            "catalysts": ["Earnings", "Product launch"],
            "risks": ["Competition", "Macro headwinds"],
        })
    return {"team_id": team_id, "assessments": assessments, "additional_candidate": None, "tool_calls": []}


def _stub_run_peer_review(team_id, quant_picks, qual_assessments,
                          additional_candidate=None, technical_scores=None,
                          market_regime="neutral", api_key=None, **kwargs):
    logger.info("[offline] stub run_peer_review: %s", team_id)
    recs = []
    for qa in (qual_assessments or [])[:3]:
        recs.append({
            "ticker": qa.get("ticker", ""),
            "quant_score": 65,
            "qual_score": qa.get("qual_score", 60),
            "combined_score": 63,
            "bull_case": qa.get("bull_case", ""),
            "bear_case": qa.get("bear_case", ""),
            "catalysts": qa.get("catalysts", []),
            "conviction": "medium",
            "quant_rationale": "[OFFLINE] Synthetic peer review",
            "team_id": team_id,
        })
    return {"recommendations": recs, "peer_review_rationale": "[OFFLINE] Synthetic review"}


def _stub_run_sector_team(team_id, scanner_universe, sector_map, price_data,
                          technical_scores, market_regime, prior_theses,
                          held_tickers, news_data_by_ticker, analyst_data_by_ticker,
                          insider_data_by_ticker, prior_sector_ratings,
                          current_sector_ratings, run_date, api_key=None, **kwargs):
    logger.info("[offline] stub run_sector_team: %s", team_id)
    from agents.sector_teams.team_config import get_team_tickers
    sector_tickers = get_team_tickers(team_id, scanner_universe, sector_map)
    if not sector_tickers:
        return {"team_id": team_id, "recommendations": [], "thesis_updates": {},
                "quant_output": {}, "qual_output": {}, "peer_review_output": {}, "tool_calls": []}

    quant = _stub_run_quant_analyst(team_id, sector_tickers, market_regime, price_data, technical_scores, run_date)
    qual = _stub_run_qual_analyst(team_id, quant["ranked_picks"][:5], prior_theses, market_regime, run_date)
    peer = _stub_run_peer_review(team_id, quant["ranked_picks"][:5], qual["assessments"])

    return {
        "team_id": team_id,
        "recommendations": peer["recommendations"],
        "thesis_updates": {},
        "quant_output": quant,
        "qual_output": qual,
        "peer_review_output": peer,
        "tool_calls": [],
    }


def _stub_run_cio(candidates, macro_context, sector_ratings, current_population,
                  open_slots, exits, run_date, api_key=None, **kwargs):
    logger.info("[offline] stub run_cio: %d candidates, %d open slots", len(candidates), open_slots)
    decisions = []
    advanced = []
    entry_theses = {}
    for i, c in enumerate(candidates[:open_slots]):
        ticker = c.get("ticker", f"UNK{i}")
        decisions.append({
            "ticker": ticker,
            "decision": "ADVANCE",
            "rationale": "[OFFLINE] Synthetic CIO advance decision",
            "scores": {"conviction": 3, "macro_alignment": 3, "portfolio_fit": 3, "catalyst": 3},
        })
        advanced.append(ticker)
        entry_theses[ticker] = {
            "bull_case": "[OFFLINE] Synthetic bull case",
            "bear_case": "[OFFLINE] Synthetic bear case",
            "catalysts": ["Earnings"],
            "risks": ["Valuation"],
            "conviction": "medium",
            "conviction_rationale": "Synthetic",
            "score": c.get("combined_score", 60),
        }
    for c in candidates[open_slots:]:
        decisions.append({
            "ticker": c.get("ticker", ""),
            "decision": "REJECT",
            "rationale": "[OFFLINE] No open slots remaining",
        })
    return {"decisions": decisions, "advanced_tickers": advanced, "entry_theses": entry_theses}


# ── S3 / archive stubs ──────────────────────────────────────────────────────

def _stub_download_db(self):
    """Skip S3 download — use local DB or create empty."""
    import sqlite3
    logger.info("[offline] stub download_db — using local DB at %s", self.local_db_path)
    self.db_conn = sqlite3.connect(self.local_db_path)
    self.db_conn.row_factory = sqlite3.Row
    self._ensure_schema()
    return self.db_conn


def _stub_upload_db(self, run_date):
    logger.info("[offline] stub upload_db — skipping S3 upload")


def _stub_load_predictions_json(self):
    logger.info("[offline] stub load_predictions_json — returning empty")
    return {}


def _stub_send_email(**kwargs):
    logger.info("[offline] stub send_email — printing subject only")
    print(f"  [OFFLINE] Email would be sent: {kwargs.get('subject', '(no subject)')}")
    return True


# ── Installer ────────────────────────────────────────────────────────────────

_patches = []


def install_offline_stubs():
    """
    Monkey-patch all external call sites so the pipeline runs fully offline.
    Call this BEFORE importing graph modules.
    """
    logger.info("[offline] Installing offline stubs — no API/LLM calls will be made")

    targets = [
        # Data fetchers
        ("data.fetchers.price_fetcher.fetch_price_data", _stub_fetch_price_data),
        ("data.fetchers.price_fetcher.fetch_sp500_sp400_with_sectors", _stub_fetch_sp500_sp400),
        ("data.fetchers.price_fetcher.fetch_short_interest", _stub_fetch_short_interest),
        ("data.fetchers.news_fetcher.fetch_all_news", _stub_fetch_all_news),
        ("data.fetchers.analyst_fetcher.fetch_analyst_consensus", _stub_fetch_analyst_consensus),
        ("data.fetchers.macro_fetcher.fetch_macro_data", _stub_fetch_macro_data),
        ("data.fetchers.macro_fetcher.compute_market_breadth", _stub_compute_market_breadth),
        ("data.fetchers.revision_fetcher.fetch_revisions", _stub_fetch_revisions),
        ("data.fetchers.options_fetcher.fetch_options_signals", _stub_fetch_options_signals),
        ("data.fetchers.options_fetcher.cache_options_to_s3", _stub_cache_options_to_s3),
        ("data.fetchers.insider_fetcher.fetch_insider_activity", _stub_fetch_insider_activity),
        ("data.fetchers.insider_fetcher.cache_insider_to_s3", _stub_cache_insider_to_s3),

        # LLM agents
        ("agents.news_agent.run_news_agent", _stub_run_news_agent),
        ("agents.research_agent.run_research_agent", _stub_run_research_agent),
        ("agents.macro_agent.run_macro_agent_with_reflection", _stub_run_macro_agent_with_reflection),
        ("agents.macro_agent.run_macro_agent", _stub_run_macro_agent),
        ("agents.scanner_ranking_agent.run_scanner_ranking_agent", _stub_run_scanner_ranking_agent),
        ("agents.consolidator.run_consolidator_agent", _stub_run_consolidator_agent),

        # Sector teams + CIO
        ("agents.sector_teams.quant_analyst.run_quant_analyst", _stub_run_quant_analyst),
        ("agents.sector_teams.qual_analyst.run_qual_analyst", _stub_run_qual_analyst),
        ("agents.sector_teams.peer_review.run_peer_review", _stub_run_peer_review),
        ("agents.sector_teams.sector_team.run_sector_team", _stub_run_sector_team),
        ("agents.investment_committee.ic_cio.run_cio", _stub_run_cio),

        # Email
        ("emailer.sender.send_email", _stub_send_email),
    ]

    # Archive manager methods — patched on the class
    from archive.manager import ArchiveManager
    ArchiveManager._orig_download_db = ArchiveManager.download_db
    ArchiveManager._orig_upload_db = ArchiveManager.upload_db
    ArchiveManager._orig_load_predictions = ArchiveManager.load_predictions_json
    ArchiveManager.download_db = _stub_download_db
    ArchiveManager.upload_db = _stub_upload_db
    ArchiveManager.load_predictions_json = _stub_load_predictions_json

    # Also stub the S3 write methods on ArchiveManager
    for method_name in ("write_signals_json", "write_consolidated_report",
                        "upload_population_json"):
        if hasattr(ArchiveManager, method_name):
            original = getattr(ArchiveManager, method_name)
            setattr(ArchiveManager, f"_orig_{method_name}", original)
            def _make_stub(name):
                def _stub(self, *args, **kwargs):
                    logger.info("[offline] stub %s — skipping S3 write", name)
                return _stub
            setattr(ArchiveManager, method_name, _make_stub(method_name))

    # S3 config override — return defaults (no S3 call)
    import config as cfg_mod
    cfg_mod._research_params_cache = dict(cfg_mod._RP_DEFAULTS)

    # Patch institutional fetcher if importable
    try:
        import data.fetchers.institutional_fetcher as inst_mod
        inst_mod.fetch_institutional_accumulation = _stub_fetch_institutional_accumulation
    except ImportError:
        pass

    # Patch yfinance.download globally to prevent any stray yf calls
    try:
        import yfinance as yf
        def _stub_yf_download(*args, **kwargs):
            logger.info("[offline] stub yf.download — returning empty DataFrame")
            return pd.DataFrame()
        yf.download = _stub_yf_download
    except ImportError:
        pass

    # Apply function patches via direct module attribute replacement.
    # We import each module and replace the function attribute so that
    # callers importing from the module get the stub.
    import importlib
    for target, stub in targets:
        parts = target.rsplit(".", 1)
        mod_path, func_name = parts[0], parts[1]
        try:
            mod = importlib.import_module(mod_path)
            setattr(mod, func_name, stub)
        except (ImportError, AttributeError) as e:
            logger.warning("[offline] could not patch %s: %s", target, e)

    print("OFFLINE MODE: all API/LLM/S3/email calls stubbed with synthetic data")


def patch_graph_modules():
    """
    Patch graph module local name bindings AFTER they've been imported.
    Call this after `from graph.research_graph import ...` in the runner.
    """
    import sys
    _graph_patches = {
        # V1 data fetchers
        "fetch_price_data": _stub_fetch_price_data,
        "fetch_sp500_sp400_with_sectors": _stub_fetch_sp500_sp400,
        "fetch_short_interest": _stub_fetch_short_interest,
        "fetch_all_news": _stub_fetch_all_news,
        "fetch_analyst_consensus": _stub_fetch_analyst_consensus,
        "fetch_macro_data": _stub_fetch_macro_data,
        "compute_market_breadth": _stub_compute_market_breadth,
        "fetch_revisions": _stub_fetch_revisions,
        "fetch_options_signals": _stub_fetch_options_signals,
        "cache_options_to_s3": _stub_cache_options_to_s3,
        "fetch_insider_activity": _stub_fetch_insider_activity,
        "cache_insider_to_s3": _stub_cache_insider_to_s3,
        # V1 LLM agents
        "run_news_agent": _stub_run_news_agent,
        "run_research_agent": _stub_run_research_agent,
        "run_macro_agent_with_reflection": _stub_run_macro_agent_with_reflection,
        "run_scanner_ranking_agent": _stub_run_scanner_ranking_agent,
        "run_consolidator_agent": _stub_run_consolidator_agent,
        "send_email": _stub_send_email,
        # Sector teams + CIO
        "run_sector_team": _stub_run_sector_team,
        "run_cio": _stub_run_cio,
    }
    for mod_name in ("graph.research_graph",):
        mod = sys.modules.get(mod_name)
        if mod:
            for attr, stub in _graph_patches.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, stub)
