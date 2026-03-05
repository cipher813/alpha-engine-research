"""
Pydantic models for archive data structures.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class InvestmentThesis(BaseModel):
    symbol: str
    date: str
    run_time: str
    rating: str                          # BUY / HOLD / SELL
    score: float
    technical_score: Optional[float] = None
    news_score: Optional[float] = None
    research_score: Optional[float] = None
    macro_modifier: Optional[float] = None
    thesis_summary: Optional[str] = None
    prev_rating: Optional[str] = None
    prev_score: Optional[float] = None
    last_material_change_date: Optional[str] = None
    stale_days: Optional[int] = 0
    consistency_flag: int = 0


class AgentReport(BaseModel):
    symbol: Optional[str] = None         # None for macro (global)
    date: str
    run_time: str
    agent_type: str                      # news | research | macro | consolidator
    report_md: str
    word_count: Optional[int] = None


class CandidateTenure(BaseModel):
    symbol: str
    slot: int
    entry_date: str
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None
    replaced_by: Optional[str] = None
    peak_score: Optional[float] = None
    exit_score: Optional[float] = None
    tenure_days: Optional[int] = None


class ActiveCandidate(BaseModel):
    slot: int
    symbol: str
    entry_date: str
    prior_tenures: int = 0
    score: Optional[float] = None
    consecutive_low_runs: int = 0


class ScannerAppearance(BaseModel):
    symbol: str
    date: str
    scanner_rank: int
    scan_path: Optional[str] = None      # momentum | deep_value
    tech_score: Optional[float] = None
    news_score: Optional[float] = None
    research_score: Optional[float] = None
    final_score: Optional[float] = None
    selected: int = 0
    selection_reason: Optional[str] = None


class TechnicalScoreRecord(BaseModel):
    symbol: str
    date: str
    rsi_14: Optional[float] = None
    macd_signal: Optional[float] = None
    price_vs_ma50: Optional[float] = None
    price_vs_ma200: Optional[float] = None
    momentum_20d: Optional[float] = None
    technical_score: Optional[float] = None


class MacroSnapshot(BaseModel):
    date: str
    fed_funds_rate: Optional[float] = None
    treasury_2yr: Optional[float] = None
    treasury_10yr: Optional[float] = None
    yield_curve_slope: Optional[float] = None
    vix: Optional[float] = None
    sp500_close: Optional[float] = None
    sp500_30d_return: Optional[float] = None
    oil_wti: Optional[float] = None
    gold: Optional[float] = None
    copper: Optional[float] = None
    market_regime: Optional[str] = None
    sector_modifiers: Optional[str] = None  # JSON string
