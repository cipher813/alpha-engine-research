"""
Composite scoring for sector-team architecture.

Replaces the old news_score × w_news + research_score × w_research formula
with quant_score × w_quant + qual_score × w_qual + macro_shift + boosts.

Weights are loaded from S3 config (auto-tuned by backtester) with YAML defaults.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Default weights — overridden by S3 config/scoring_weights.json
DEFAULT_W_QUANT = 0.50
DEFAULT_W_QUAL = 0.50


def compute_composite_score(
    quant_score: float | None,
    qual_score: float | None,
    sector_modifier: float,
    boosts: dict[str, float] | None = None,
    w_quant: float = DEFAULT_W_QUANT,
    w_qual: float = DEFAULT_W_QUAL,
    max_aggregate_boost: float = 10.0,
) -> dict:
    """
    Compute the composite attractiveness score.

    Args:
        quant_score: Quantitative score (0-100) from quant analyst. None if failed.
        qual_score: Qualitative score (0-100) from qual analyst. None if failed.
        sector_modifier: Macro sector modifier (0.70-1.30).
        boosts: {boost_name: points} from O10-O13 enrichments.
        w_quant: Weight for quant score.
        w_qual: Weight for qual score.
        max_aggregate_boost: Cap on total boost points.

    Returns:
        {
            "final_score": float (0-100),
            "weighted_base": float,
            "macro_shift": float,
            "total_boost": float,
            "score_failed": bool,
        }
    """
    # Handle missing scores
    if quant_score is None and qual_score is None:
        return {
            "final_score": None,
            "weighted_base": None,
            "macro_shift": 0.0,
            "total_boost": 0.0,
            "score_failed": True,
        }

    # If one score is missing, use the other at full weight
    if quant_score is None:
        weighted_base = qual_score
    elif qual_score is None:
        weighted_base = quant_score
    else:
        weighted_base = quant_score * w_quant + qual_score * w_qual

    # Macro shift: (modifier - 1.0) / 0.30 × 10 → range [-10, +10]
    macro_shift = (sector_modifier - 1.0) / 0.30 * 10.0

    # Aggregate boosts with cap
    total_boost = 0.0
    if boosts:
        total_boost = sum(boosts.values())
        total_boost = max(-max_aggregate_boost, min(max_aggregate_boost, total_boost))

    final = weighted_base + macro_shift + total_boost
    final = max(0.0, min(100.0, final))

    return {
        "final_score": round(final, 1),
        "weighted_base": round(weighted_base, 1),
        "macro_shift": round(macro_shift, 1),
        "total_boost": round(total_boost, 1),
        "score_failed": False,
    }


def score_to_rating(score: float | None, buy_threshold: float = 70.0, sell_threshold: float = 40.0) -> str:
    """Convert a composite score to a rating."""
    if score is None:
        return "HOLD"
    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"
