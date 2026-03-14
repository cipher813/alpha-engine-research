"""
Score aggregator — computes weighted composite attractiveness score.

Formula (§5):
  Base  = tech × 0.40 + news × 0.30 + research × 0.30
  Score = Base + macro_shift   (clipped to [0, 100])

  macro_shift = (sector_modifier - 1.0) / 0.30 × MACRO_MAX_SHIFT_POINTS
  → modifier 0.70 → -10 pts  |  1.0 → 0 pts  |  1.30 → +10 pts

Design rationale: additive bounded shift preserves conviction-level ratings.
A caution regime (modifier ~0.85) nudges scores down ~5 pts rather than
multiplying by 0.85 which would suppress every stock in the universe uniformly.
High-conviction stocks stay BUY through moderate headwinds; only truly marginal
scores flip to HOLD/SELL, which is the desired analyst-aligned behavior.

Uses per-sector macro modifiers from the Macro Agent (§5.4), not a single global shift.
"""

from __future__ import annotations

import json
import logging
import os

from config import (
    WEIGHT_TECHNICAL,
    WEIGHT_NEWS,
    WEIGHT_RESEARCH,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
    SECTOR_MAP,
    STALENESS_THRESHOLD_DAYS,
    MATERIAL_SCORE_CHANGE_MIN,
)

logger = logging.getLogger(__name__)

DEFAULT_SECTOR_MODIFIER = 1.0   # fallback if sector or macro data unavailable

# Module-level cache: populated once per Lambda cold-start by _get_weights().
_weights_cache: dict | None = None


def _load_weights_from_s3() -> dict | None:
    """
    Check S3 for backtester-updated scoring weights.

    Reads s3://{RESEARCH_BUCKET}/config/scoring_weights.json, written by the
    backtester's weight optimizer when it applies an update. Returns None if
    the file doesn't exist or cannot be read — caller falls back to universe.yaml.
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    key = "config/scoring_weights.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        weights = {k: float(data[k]) for k in ("technical", "news", "research") if k in data}
        if len(weights) == 3:
            logger.info(
                "Scoring weights loaded from S3 (updated %s, n=%s): %s",
                data.get("updated_at", "unknown"),
                data.get("n_samples", "?"),
                weights,
            )
            return weights
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            logger.warning("Could not read scoring weights from S3: %s", e)
    except Exception as e:
        logger.warning("Unexpected error loading S3 scoring weights: %s", e)
    return None


def _get_weights() -> tuple[float, float, float]:
    """
    Return current scoring weights, checking S3 override first.

    Result is cached for the lifetime of the Lambda instance (one S3 call
    per cold-start). Falls back to universe.yaml values if S3 file absent.
    """
    global _weights_cache
    if _weights_cache is None:
        s3_weights = _load_weights_from_s3()
        _weights_cache = s3_weights or {
            "technical": WEIGHT_TECHNICAL,
            "news": WEIGHT_NEWS,
            "research": WEIGHT_RESEARCH,
        }
    return _weights_cache["technical"], _weights_cache["news"], _weights_cache["research"]
MACRO_MODIFIER_RANGE = 0.30     # distance from 1.0 to min/max (0.70 and 1.30)
MACRO_MAX_SHIFT_POINTS = 10.0   # max pts added/subtracted by macro shift


def compute_attractiveness_score(
    ticker: str,
    technical_score: float,
    news_score: float,
    research_score: float,
    sector_modifiers: dict[str, float],
    sector_map: dict[str, str] | None = None,
) -> dict:
    """
    Compute the final attractiveness score for a ticker.

    Returns dict with:
      technical_score, news_score, research_score, macro_modifier,
      weighted_base, macro_shift, final_score, rating
    """
    sm = sector_map or SECTOR_MAP
    sector = sm.get(ticker, "Technology")
    macro_modifier = sector_modifiers.get(sector, DEFAULT_SECTOR_MODIFIER)
    macro_modifier = max(0.70, min(1.30, macro_modifier))  # clamp to valid range

    w_tech, w_news, w_research = _get_weights()
    weighted_base = (
        technical_score * w_tech
        + news_score * w_news
        + research_score * w_research
    )

    # Additive bounded shift: (modifier - 1.0) / 0.30 × 10 → range [-10, +10]
    macro_shift = (macro_modifier - 1.0) / MACRO_MODIFIER_RANGE * MACRO_MAX_SHIFT_POINTS
    final_score = max(0.0, min(100.0, weighted_base + macro_shift))

    return {
        "ticker": ticker,
        "sector": sector,
        "technical_score": round(technical_score, 2),
        "news_score": round(news_score, 2),
        "research_score": round(research_score, 2),
        "macro_modifier": round(macro_modifier, 3),
        "macro_shift": round(macro_shift, 2),
        "weighted_base": round(weighted_base, 2),
        "final_score": round(final_score, 2),
        "rating": score_to_rating(final_score),
    }


def score_to_rating(score: float) -> str:
    """Convert numeric score to BUY / HOLD / SELL (§5.5)."""
    if score >= RATING_BUY_THRESHOLD:
        return "BUY"
    if score >= RATING_SELL_THRESHOLD:
        return "HOLD"
    return "SELL"


def check_consistency(thesis_text: str, score: float) -> bool:
    """
    Automated pre-LLM check: flag if thesis text direction is inconsistent
    with numeric score (§4.4 step 2).

    Returns True if inconsistency detected (consistency_flag = 1).
    Heuristic: look for strongly bullish/bearish language vs. score.
    """
    if not thesis_text:
        return False

    text_lower = thesis_text.lower()

    bullish_terms = ["strong buy", "bullish", "upside", "buy", "outperform", "positive catalyst"]
    bearish_terms = ["sell", "downside", "bearish", "underperform", "headwind", "risk", "decline"]

    bullish_hits = sum(1 for t in bullish_terms if t in text_lower)
    bearish_hits = sum(1 for t in bearish_terms if t in text_lower)

    if bullish_hits > bearish_hits and score < 40:
        return True  # clearly bullish text but SELL score
    if bearish_hits > bullish_hits and score > 70:
        return True  # clearly bearish text but BUY score

    return False


def compute_staleness(
    last_material_change_date: str | None,
    trading_days_since: int,
) -> bool:
    """
    Return True if the score is stale (no material change in >= STALENESS_THRESHOLD_DAYS).
    """
    if last_material_change_date is None:
        return False
    return trading_days_since >= STALENESS_THRESHOLD_DAYS


def compute_conviction(score_history: list[float]) -> str:
    """
    Derive conviction from the last 3 scores (most recent first).

    rising:   2+ of the last 2 deltas are positive
    declining: 2+ of the last 2 deltas are negative
    stable:   mixed or insufficient history
    """
    if len(score_history) < 2:
        return "stable"
    # Deltas: [score[0]-score[1], score[1]-score[2]] (recent first)
    deltas = [score_history[i] - score_history[i + 1] for i in range(min(2, len(score_history) - 1))]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if pos > neg:
        return "rising"
    if neg > pos:
        return "declining"
    return "stable"


def compute_score_velocity_5d(score_history: list[float]) -> float | None:
    """
    Average daily score change over last 5 runs (most recent first).
    Returns None if fewer than 2 data points.
    """
    if len(score_history) < 2:
        return None
    window = score_history[:5]
    return round((window[0] - window[-1]) / (len(window) - 1), 3)


def compute_signal(
    rating: str,
    prior_rating: str | None,
    conviction: str,
    material_changes: bool,
) -> str:
    """
    Derive an actionable signal for the executor.

    ENTER:  rating is BUY (executor skips tickers already in portfolio)
    EXIT:   rating is SELL (regardless of prior)
    REDUCE: BUY → HOLD transition with declining conviction or material change
    HOLD:   all other cases
    """
    if rating == "BUY":
        return "ENTER"
    if rating == "SELL":
        return "EXIT"
    if rating == "HOLD" and prior_rating == "BUY" and (conviction == "declining" or material_changes):
        return "REDUCE"
    return "HOLD"


def compute_long_term_score(
    news_score_lt: float,
    research_score_lt: float,
    sector_modifiers: dict[str, float],
    sector: str,
) -> tuple[float, str]:
    """
    Compute a long-term (12-month) composite score and rating.

    Technical indicators are inherently short-term and excluded.
    Weights: news_lt 50%, research_lt 50%, plus the same macro sector shift.

    Returns (long_term_score, long_term_rating).
    """
    macro_modifier = sector_modifiers.get(sector, DEFAULT_SECTOR_MODIFIER)
    macro_modifier = max(0.70, min(1.30, macro_modifier))
    macro_shift = (macro_modifier - 1.0) / MACRO_MODIFIER_RANGE * MACRO_MAX_SHIFT_POINTS
    base = news_score_lt * 0.50 + research_score_lt * 0.50
    score = round(max(0.0, min(100.0, base + macro_shift)), 2)
    return score, score_to_rating(score)


def aggregate_all(
    tickers: list[str],
    technical_scores: dict[str, dict],
    news_scores: dict[str, float],
    research_scores: dict[str, float],
    sector_modifiers: dict[str, float],
    prior_theses: dict[str, dict],
    sector_map: dict[str, str] | None = None,
    run_date: str | None = None,
    score_history: dict[str, list[float]] | None = None,
    price_target_upside: dict[str, float | None] | None = None,
    news_scores_lt: dict[str, float] | None = None,
    research_scores_lt: dict[str, float] | None = None,
) -> dict[str, dict]:
    """
    Run aggregation for all tickers in a single pass.

    Returns {ticker: full_result_dict} including conviction, signal,
    score_velocity_5d, price_target_upside, consistency_flag, and stale_days.

    Args:
        score_history: {ticker: [score_t0, score_t-1, score_t-2, ...]} most-recent first
        price_target_upside: {ticker: float | None} precomputed (price_target/price - 1)
    """
    _score_history = score_history or {}
    _pta = price_target_upside or {}
    _news_lt = news_scores_lt or {}
    _research_lt = research_scores_lt or {}
    results = {}

    for ticker in tickers:
        tech = technical_scores.get(ticker, {})
        tech_score = tech.get("technical_score", 50.0)
        news_score = news_scores.get(ticker, 50.0)
        research_score = research_scores.get(ticker, 50.0)

        result = compute_attractiveness_score(
            ticker=ticker,
            technical_score=tech_score,
            news_score=news_score,
            research_score=research_score,
            sector_modifiers=sector_modifiers,
            sector_map=sector_map,
        )

        # Long-term composite (12-month horizon, no technical component)
        lt_score, lt_rating = compute_long_term_score(
            news_score_lt=_news_lt.get(ticker, 50.0),
            research_score_lt=_research_lt.get(ticker, 50.0),
            sector_modifiers=sector_modifiers,
            sector=result["sector"],
        )

        # Prior data for change tracking
        prior = prior_theses.get(ticker, {})
        prior_score = prior.get("score")
        prior_rating = prior.get("rating")

        score_delta = None
        if prior_score is not None:
            score_delta = round(result["final_score"] - prior_score, 2)

        # Staleness tracking (§5.7)
        last_change_date = prior.get("last_material_change_date")
        stale_days = prior.get("stale_days", 0) or 0

        material_change = False
        if prior_score is None or abs(result["final_score"] - prior_score) >= MATERIAL_SCORE_CHANGE_MIN:
            material_change = True
            stale_days = 0
            last_change_date = run_date
        else:
            stale_days += 1  # approx: will be corrected by actual trading day count in DB

        # Conviction and velocity from score history (§A.4)
        history = _score_history.get(ticker, [])
        conviction = compute_conviction(history)
        velocity_5d = compute_score_velocity_5d(history)

        # Actionable signal for executor — driven by short-term final_score/rating (§A.3)
        signal = compute_signal(
            rating=result["rating"],
            prior_rating=prior_rating,
            conviction=conviction,
            material_changes=material_change,
        )

        # Consistency check
        thesis_summary = prior.get("thesis_summary", "")
        consistency_flag = check_consistency(thesis_summary, result["final_score"])

        results[ticker] = {
            **result,
            "prior_score": prior_score,
            "prior_rating": prior_rating,
            "score_delta": score_delta,
            "last_material_change_date": last_change_date,
            "stale_days": stale_days,
            "material_changes": material_change,
            "consistency_flag": int(consistency_flag),
            "conviction": conviction,
            "score_velocity_5d": velocity_5d,
            "signal": signal,
            "price_target_upside": _pta.get(ticker),
            "long_term_score": lt_score,
            "long_term_rating": lt_rating,
        }

    return results
