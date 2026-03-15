"""
data/population_selector.py — sector-balanced investment population management.

Replaces the old dual-branch architecture (20 static universe + 3 rotating
candidates) with a scanner-driven population of 20-25 stocks drawn entirely
from S&P 900.

Sector allocation is driven by the macro agent's sector_modifiers:
  - Overweight sectors (modifier >= 1.05): 3+ stocks each
  - Market-weight sectors (0.95 < modifier < 1.05): ~2 stocks each
  - Underweight sectors (modifier <= 0.95): at least 1 stock each

Rotation rules:
  - Stocks stay in population unless thesis degrades (long_term_score drops)
    or a same-sector challenger scores meaningfully higher.
  - Minimum tenure protection (2 weeks) prevents churn, with override for
    thesis collapse (score < 40 → immediate removal).
  - Max 3 rotations per weekly run.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 11 GICS sectors (display names matching our sector labels)
GICS_SECTORS = [
    "Technology",
    "Healthcare",
    "Financial",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]


def classify_sectors(
    sector_ratings: dict[str, dict],
    config: dict,
) -> dict[str, str]:
    """
    Classify each GICS sector as overweight / market_weight / underweight
    based on the macro agent's sector_modifier values.

    Args:
        sector_ratings: {sector_name: {modifier: float, rating: str, ...}}
        config: population config from universe.yaml

    Returns: {sector_name: "overweight" | "market_weight" | "underweight"}
    """
    sa_config = config.get("sector_allocation", {})
    ow_thresh = sa_config.get("overweight_modifier_threshold", 1.05)
    uw_thresh = sa_config.get("underweight_modifier_threshold", 0.95)

    result: dict[str, str] = {}
    for sector in GICS_SECTORS:
        data = sector_ratings.get(sector, {})
        modifier = data.get("modifier", 1.0)
        if modifier >= ow_thresh:
            result[sector] = "overweight"
        elif modifier <= uw_thresh:
            result[sector] = "underweight"
        else:
            result[sector] = "market_weight"

    return result


def compute_sector_slots(
    sector_classes: dict[str, str],
    config: dict,
) -> dict[str, int]:
    """
    Compute target slot count per sector based on classification.

    Starting allocation:
      overweight → overweight_min (default 3)
      market_weight → market_weight_target (default 2)
      underweight → underweight_min (default 1)

    Adjusts to hit target_size:
      If total < target: add extra slots to overweight sectors first
      If total > target: reduce market_weight sectors first

    Returns: {sector_name: slot_count}
    """
    sa_config = config.get("sector_allocation", {})
    ow_min = sa_config.get("overweight_min", 3)
    mw_target = sa_config.get("market_weight_target", 2)
    uw_min = sa_config.get("underweight_min", 1)
    target_size = config.get("target_size", 25)

    slots: dict[str, int] = {}
    for sector, classification in sector_classes.items():
        if classification == "overweight":
            slots[sector] = ow_min
        elif classification == "market_weight":
            slots[sector] = mw_target
        else:
            slots[sector] = uw_min

    total = sum(slots.values())

    # Adjust to hit target_size
    if total < target_size:
        # Add extra slots to overweight sectors (round-robin)
        ow_sectors = [s for s, c in sector_classes.items() if c == "overweight"]
        if not ow_sectors:
            ow_sectors = [s for s, c in sector_classes.items() if c == "market_weight"]
        idx = 0
        while total < target_size and ow_sectors:
            slots[ow_sectors[idx % len(ow_sectors)]] += 1
            total += 1
            idx += 1
    elif total > target_size:
        # Reduce market_weight sectors first (don't go below 1)
        mw_sectors = sorted(
            [s for s, c in sector_classes.items() if c == "market_weight"],
            key=lambda s: slots[s],
            reverse=True,
        )
        idx = 0
        while total > target_size and mw_sectors:
            s = mw_sectors[idx % len(mw_sectors)]
            if slots[s] > 1:
                slots[s] -= 1
                total -= 1
            idx += 1
            if idx >= len(mw_sectors) * 5:  # safety limit
                break

    return slots


def select_population(
    scored_candidates: list[dict],
    current_population: list[dict],
    sector_ratings: dict[str, dict],
    config: dict,
    run_date: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Build the investment population from S&P 900 scanner results.

    Args:
        scored_candidates: list of scored stock dicts from scanner pipeline.
            Each must have: ticker, sector, long_term_score, long_term_rating,
            conviction, price_target_upside, thesis_summary, sub_scores, etc.
        current_population: existing population from prior run (may be empty on first run).
            Each must have: ticker, sector, long_term_score, entry_date, tenure_weeks.
        sector_ratings: from macro agent {sector: {modifier, rating, ...}}
        config: population config section from universe.yaml
        run_date: current run date (YYYY-MM-DD). Defaults to today.

    Returns:
        (new_population, rotation_events)
        - new_population: list of population dicts with all fields
        - rotation_events: list of {type, ticker, sector, reason, ...} dicts
    """
    if run_date is None:
        run_date = str(date.today())

    pop_config = config.get("population", config)
    rotation_config = pop_config.get("rotation", {})

    min_lt_score = rotation_config.get("min_long_term_score", 45)
    challenger_delta = rotation_config.get("challenger_score_delta", 5)
    max_rotations = rotation_config.get("max_rotations_per_run", 3)
    min_tenure = rotation_config.get("min_tenure_weeks", 2)
    collapse_thresh = rotation_config.get("thesis_collapse_threshold", 40)

    # Classify sectors and compute slots
    sector_classes = classify_sectors(sector_ratings, pop_config)
    sector_slots = compute_sector_slots(sector_classes, pop_config)

    # Index current population by ticker
    current_by_ticker = {p["ticker"]: p for p in current_population}

    # Index scored candidates by ticker (includes re-scored incumbents)
    candidates_by_ticker = {c["ticker"]: c for c in scored_candidates}

    # Group all candidates by sector
    candidates_by_sector: dict[str, list[dict]] = {}
    for c in scored_candidates:
        sector = c.get("sector", "Unknown")
        candidates_by_sector.setdefault(sector, []).append(c)

    # Sort each sector's candidates by long_term_score descending
    for sector in candidates_by_sector:
        candidates_by_sector[sector].sort(
            key=lambda c: c.get("long_term_score", 0), reverse=True,
        )

    new_population: list[dict] = []
    rotation_events: list[dict] = []
    rotations_used = 0

    # ── Phase 1: Evaluate incumbents ──
    # Check each current population member — keep, remove, or replace
    incumbents_by_sector: dict[str, list[dict]] = {}
    removed_tickers: set[str] = set()

    for incumbent in current_population:
        ticker = incumbent["ticker"]
        sector = incumbent.get("sector", "Unknown")

        # Get refreshed score from scanner results (if available)
        refreshed = candidates_by_ticker.get(ticker)
        lt_score = refreshed.get("long_term_score", incumbent.get("long_term_score", 50.0)) if refreshed else incumbent.get("long_term_score", 50.0)

        # Compute tenure
        entry_date = incumbent.get("entry_date", run_date)
        try:
            tenure_weeks = (datetime.fromisoformat(run_date) - datetime.fromisoformat(entry_date)).days // 7
        except (ValueError, TypeError):
            tenure_weeks = incumbent.get("tenure_weeks", 0)

        # Check for thesis collapse (immediate removal regardless of tenure)
        if lt_score < collapse_thresh:
            rotation_events.append({
                "type": "REMOVE",
                "ticker": ticker,
                "sector": sector,
                "reason": f"thesis_collapse (lt_score={lt_score:.1f} < {collapse_thresh})",
                "long_term_score": lt_score,
            })
            removed_tickers.add(ticker)
            rotations_used += 1
            continue

        # Check for score below minimum (with tenure protection)
        if lt_score < min_lt_score and tenure_weeks >= min_tenure:
            if rotations_used < max_rotations:
                rotation_events.append({
                    "type": "REMOVE",
                    "ticker": ticker,
                    "sector": sector,
                    "reason": f"score_degraded (lt_score={lt_score:.1f} < {min_lt_score}, tenure={tenure_weeks}w)",
                    "long_term_score": lt_score,
                })
                removed_tickers.add(ticker)
                rotations_used += 1
                continue

        # Incumbent survives — track by sector
        kept = refreshed.copy() if refreshed else incumbent.copy()
        kept["entry_date"] = entry_date
        kept["tenure_weeks"] = tenure_weeks
        incumbents_by_sector.setdefault(sector, []).append(kept)

    # ── Phase 2: Fill sector slots ──
    for sector in GICS_SECTORS:
        target_slots = sector_slots.get(sector, 1)
        sector_incumbents = incumbents_by_sector.get(sector, [])
        sector_candidates = candidates_by_sector.get(sector, [])

        # Sort incumbents by score (keep best)
        sector_incumbents.sort(
            key=lambda x: x.get("long_term_score", 0), reverse=True,
        )

        # Trim incumbents if we have too many for this sector
        while len(sector_incumbents) > target_slots and rotations_used < max_rotations:
            weakest = sector_incumbents.pop()
            ticker = weakest["ticker"]
            tenure = weakest.get("tenure_weeks", 0)
            if tenure >= min_tenure:
                rotation_events.append({
                    "type": "REMOVE",
                    "ticker": ticker,
                    "sector": sector,
                    "reason": f"sector_rebalance (over_allocated, lt_score={weakest.get('long_term_score', 0):.1f})",
                    "long_term_score": weakest.get("long_term_score", 0),
                })
                removed_tickers.add(ticker)
                rotations_used += 1
            else:
                # Keep despite over-allocation (tenure too short)
                sector_incumbents.append(weakest)
                break

        # Check for superior challengers (replace weakest incumbent)
        if sector_incumbents and sector_candidates and rotations_used < max_rotations:
            weakest_incumbent = min(sector_incumbents, key=lambda x: x.get("long_term_score", 0))
            weakest_score = weakest_incumbent.get("long_term_score", 0)
            weakest_tenure = weakest_incumbent.get("tenure_weeks", 0)

            for challenger in sector_candidates:
                c_ticker = challenger["ticker"]
                if c_ticker in removed_tickers:
                    continue
                if any(inc["ticker"] == c_ticker for inc in sector_incumbents):
                    continue  # already an incumbent
                c_score = challenger.get("long_term_score", 0)

                if (
                    c_score > weakest_score + challenger_delta
                    and weakest_tenure >= min_tenure
                    and len(sector_incumbents) >= target_slots
                ):
                    # Replace weakest with challenger
                    rotation_events.append({
                        "type": "REPLACE",
                        "ticker_out": weakest_incumbent["ticker"],
                        "ticker_in": c_ticker,
                        "sector": sector,
                        "reason": f"superior_challenger ({c_score:.1f} > {weakest_score:.1f} + {challenger_delta})",
                        "score_out": weakest_score,
                        "score_in": c_score,
                    })
                    removed_tickers.add(weakest_incumbent["ticker"])
                    sector_incumbents.remove(weakest_incumbent)
                    challenger_entry = challenger.copy()
                    challenger_entry["entry_date"] = run_date
                    challenger_entry["tenure_weeks"] = 0
                    sector_incumbents.append(challenger_entry)
                    rotations_used += 1
                    break  # max 1 replacement per sector per run

        # Fill remaining slots with top candidates
        current_tickers = {inc["ticker"] for inc in sector_incumbents}
        slots_to_fill = target_slots - len(sector_incumbents)

        for challenger in sector_candidates:
            if slots_to_fill <= 0:
                break
            c_ticker = challenger["ticker"]
            if c_ticker in current_tickers or c_ticker in removed_tickers:
                continue
            # New addition — no rotation count needed for fills
            entry = challenger.copy()
            entry["entry_date"] = run_date
            entry["tenure_weeks"] = 0
            sector_incumbents.append(entry)
            current_tickers.add(c_ticker)
            slots_to_fill -= 1

            rotation_events.append({
                "type": "ADD",
                "ticker": c_ticker,
                "sector": sector,
                "reason": f"slot_fill (lt_score={entry.get('long_term_score', 0):.1f})",
                "long_term_score": entry.get("long_term_score", 0),
            })

        new_population.extend(sector_incumbents)

    # Sort final population by long_term_score descending
    new_population.sort(key=lambda p: p.get("long_term_score", 0), reverse=True)

    # Log summary
    sector_counts = {}
    for p in new_population:
        s = p.get("sector", "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    logger.info(
        "Population selection complete: %d stocks across %d sectors | "
        "%d rotations | sector breakdown: %s",
        len(new_population),
        len(sector_counts),
        rotations_used,
        ", ".join(f"{s}={n}" for s, n in sorted(sector_counts.items())),
    )

    return new_population, rotation_events
