"""
Central configuration reader — mirrors universe.yaml into typed Python constants.
All other modules import from here rather than reading YAML directly.
"""

import os
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "config" / "universe.yaml"


def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── Population (replaces static universe) ────────────────────────────────────
# All stocks are derived from S&P 900 scanner — no hardcoded starting stocks.
# UNIVERSE / UNIVERSE_TICKERS / SECTOR_MAP are loaded dynamically from
# population/latest.json (S3) or SQLite at run time.
# The static list below is kept empty — graph.research_graph loads the active
# population from the archive manager at startup.
POPULATION_CFG: dict = _cfg.get("population", {})
UNIVERSE: list[dict] = _cfg.get("universe", [])  # backward compat (empty after migration)
UNIVERSE_TICKERS: list[str] = [s["ticker"] for s in UNIVERSE]
SECTOR_MAP: dict[str, str] = {s["ticker"]: s["sector"] for s in UNIVERSE}

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORING_WEIGHTS: dict[str, float] = _cfg["scoring_weights"]
# Horizon separation: Research uses news + research only (6–12 month fundamental).
# Technical analysis is handled by Predictor (GBM) and Executor (ATR/time exits).
WEIGHT_NEWS: float = SCORING_WEIGHTS.get("news", 0.50)
WEIGHT_RESEARCH: float = SCORING_WEIGHTS.get("research", 0.50)

RATING_BUY_THRESHOLD: float = _cfg["rating_thresholds"]["buy"]
RATING_SELL_THRESHOLD: float = _cfg["rating_thresholds"]["sell"]

# ── Scanner ───────────────────────────────────────────────────────────────────
SCANNER_CFG: dict = _cfg["scanner"]
CANDIDATE_COUNT: int = SCANNER_CFG["candidate_count"]
CANDIDATE_UNIVERSE: str = SCANNER_CFG["candidate_universe"]
MIN_AVG_VOLUME: int = SCANNER_CFG["min_avg_volume"]
MIN_PRICE: float = SCANNER_CFG["min_price"]
ROTATION_TIERS: list[dict] = SCANNER_CFG["rotation_tiers"]
WEAK_PICK_SCORE_THRESHOLD: float = SCANNER_CFG["weak_pick_score_threshold"]
WEAK_PICK_CONSECUTIVE_RUNS: int = SCANNER_CFG["weak_pick_consecutive_runs"]
EMERGENCY_ROTATION_NEW_SCORE: float = SCANNER_CFG["emergency_rotation_new_score"]
DEEP_VALUE_PATH_ENABLED: bool = SCANNER_CFG["deep_value_path"]
DEEP_VALUE_MAX_RSI: float = SCANNER_CFG["deep_value_max_rsi"]
DEEP_VALUE_MIN_CONSENSUS: str = SCANNER_CFG["deep_value_min_consensus"]
DEEP_VALUE_MAX_CANDIDATES: int = SCANNER_CFG["deep_value_max_candidates"]

# ── Archive ───────────────────────────────────────────────────────────────────
ARCHIVE_CFG: dict = _cfg["archive"]

# ── Alerts ────────────────────────────────────────────────────────────────────
ALERTS_CFG: dict = _cfg["alerts"]
ALERTS_ENABLED: bool = ALERTS_CFG["enabled"]
PRICE_MOVE_THRESHOLD_PCT: float = ALERTS_CFG["price_move_threshold_pct"]
ALERT_COOLDOWN_MINUTES: int = ALERTS_CFG["cooldown_minutes"]

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_CFG: dict = _cfg["email"]
EMAIL_RECIPIENTS: list[str] = EMAIL_CFG["recipients"]
EMAIL_SENDER: str = EMAIL_CFG["sender"]

# ── Schedule ──────────────────────────────────────────────────────────────────
SCHEDULE_CFG: dict = _cfg["schedule"]
HOLIDAY_CALENDAR: str = SCHEDULE_CFG["holiday_calendar"]

# ── Predictor ─────────────────────────────────────────────────────────────────
_pred_cfg: dict = _cfg.get("predictor", {})
PREDICTOR_PREDICTIONS_KEY: str = _pred_cfg.get("s3_predictions_key", "predictor/predictions/latest.json")
# Minimum GBM prediction_confidence required to apply the confirmation gate veto.
# Below this threshold the prediction is treated as low-conviction and ignored.
MIN_PREDICTION_CONFIDENCE: float = float(_pred_cfg.get("min_confidence", 0.60))

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_CFG: dict = _cfg["llm"]
PER_STOCK_MODEL: str = LLM_CFG["per_stock_model"]
STRATEGIC_MODEL: str = LLM_CFG["strategic_model"]
MAX_TOKENS_PER_STOCK: int = LLM_CFG["max_tokens_per_stock"]
MAX_TOKENS_STRATEGIC: int = LLM_CFG["max_tokens_strategic"]
CONCURRENT_AGENTS: int = LLM_CFG["concurrent_agents"]

# ── AWS / Environment ─────────────────────────────────────────────────────────
S3_BUCKET: str = os.environ.get("S3_BUCKET", "alpha-engine-research")
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
FMP_API_KEY: str = os.environ.get("FMP_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# ── Scoring / staleness ───────────────────────────────────────────────────────
STALENESS_THRESHOLD_DAYS: int = 5       # flag if score unchanged >= this many trading days
MATERIAL_SCORE_CHANGE_MIN: float = 3.0  # minimum point change to reset last_material_change_date

# ── All tracked tickers in a run (universe + up to 3 candidates) ──────────────
ALL_SECTORS: list[str] = [
    "Technology", "Healthcare", "Financial", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Materials",
    "Real Estate", "Utilities", "Communication Services",
]
