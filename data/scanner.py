"""
Market scanner — Stage 1 quantitative filter.

Scans S&P 500 + S&P 400 (~900 stocks) and reduces to ~50 candidates
via momentum and deep-value paths (§6.1, §6.3).

No LLM is used in this module.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import (
    UNIVERSE_TICKERS,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    DEEP_VALUE_PATH_ENABLED,
    DEEP_VALUE_MAX_RSI,
    DEEP_VALUE_MAX_CANDIDATES,
)
from data.fetchers.price_fetcher import (
    fetch_sp500_sp400_tickers,
    fetch_price_data,
    compute_technical_indicators,
)
from scoring.technical import compute_technical_score


def get_scanner_universe(exclude_tickers: Optional[list[str]] = None) -> list[str]:
    """
    Return the full scanner candidate universe (S&P 500 + S&P 400),
    excluding any tickers already in the monitored universe.
    """
    all_tickers = fetch_sp500_sp400_tickers()
    exclude = set(exclude_tickers or []) | set(UNIVERSE_TICKERS)
    return [t for t in all_tickers if t not in exclude]


def run_quant_filter(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    technical_scores: dict[str, dict],
    market_regime: str = "neutral",
) -> list[dict]:
    """
    Stage 1: Apply quantitative filter to reduce ~900 → ~50.

    Returns list of candidate dicts with fields:
      ticker, path (momentum|deep_value), tech_score, rsi_14,
      price, avg_volume_20d, price_vs_ma200
    """
    momentum_candidates = []
    deep_value_candidates = []

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        tech = technical_scores.get(ticker) or compute_technical_indicators(df)
        if tech is None:
            continue

        price = tech.get("current_price", 0)
        avg_vol = tech.get("avg_volume_20d", 0) or 0
        rsi = tech.get("rsi_14", 50)
        price_vs_ma200 = tech.get("price_vs_ma200")

        # Basic liquidity + price floor (both paths)
        if avg_vol < MIN_AVG_VOLUME or price < MIN_PRICE:
            continue

        tech_score = compute_technical_score(tech, market_regime=market_regime)

        candidate = {
            "ticker": ticker,
            "tech_score": tech_score,
            "rsi_14": rsi,
            "current_price": price,
            "avg_volume_20d": avg_vol,
            "price_vs_ma200": price_vs_ma200,
        }

        # ── Momentum path ──────────────────────────────────────────────────
        # Require strong technicals and no severe downtrend
        if (
            tech_score >= 60
            and (price_vs_ma200 is None or price_vs_ma200 > -15)
        ):
            candidate["path"] = "momentum"
            momentum_candidates.append(candidate)

        # ── Deep value path (config-gated) ─────────────────────────────────
        elif DEEP_VALUE_PATH_ENABLED and rsi < DEEP_VALUE_MAX_RSI:
            # RSI oversold + below 200 MA → potential bottoming
            # Analyst conviction check happens in Stage 2 (after FMP fetch)
            candidate["path"] = "deep_value_pending"  # confirmed after analyst data
            deep_value_candidates.append(candidate)

    # Sort momentum candidates by tech_score descending; take top 40
    momentum_candidates.sort(key=lambda x: x["tech_score"], reverse=True)
    momentum_top = momentum_candidates[:40]

    # Deep value: cap at DEEP_VALUE_MAX_CANDIDATES (default 10)
    deep_value_candidates.sort(key=lambda x: x["rsi_14"])  # most oversold first
    deep_value_top = deep_value_candidates[:DEEP_VALUE_MAX_CANDIDATES]

    combined = momentum_top + deep_value_top

    # Deduplicate (a ticker can't be in both paths)
    seen: set[str] = set()
    result = []
    for c in combined:
        if c["ticker"] not in seen:
            seen.add(c["ticker"])
            result.append(c)

    return result


def confirm_deep_value_with_analyst(
    candidates: list[dict],
    analyst_data: dict[str, dict],
    min_consensus: str = "Buy",
) -> list[dict]:
    """
    Stage 2 (partial): For deep_value_pending candidates, confirm analyst conviction.
    Removes candidates that don't meet the analyst threshold; promotes path field.
    """
    _CONSENSUS_RANK = {
        "Strong Buy": 5, "Buy": 4, "Hold": 3, "Underperform": 2, "Sell": 1,
    }
    min_rank = _CONSENSUS_RANK.get(min_consensus, 4)

    result = []
    for c in candidates:
        if c.get("path") != "deep_value_pending":
            result.append(c)
            continue

        adata = analyst_data.get(c["ticker"], {})
        consensus = adata.get("consensus_rating", "Hold")
        rank = _CONSENSUS_RANK.get(consensus, 0)

        if rank >= min_rank:
            c = {**c, "path": "deep_value"}
            result.append(c)
        # else: drop this candidate — analyst conviction too low

    return result


def evaluate_candidate_rotation(
    scanner_scores: dict[str, dict],
    active_candidates: list[dict],
    rotation_tiers: list[dict],
    weak_pick_score_threshold: float,
    weak_pick_consecutive_runs: int,
    emergency_rotation_new_score: float,
    run_date: str,
) -> tuple[list[dict], list[dict]]:
    """
    Stage 5: Determine whether to rotate any of the 3 active candidates.

    Args:
        scanner_scores: {ticker: {score, path, ...}} for top-10 scanner candidates
        active_candidates: current 3 dicts with {symbol, entry_date, slot, consecutive_low_runs}
        rotation_tiers: list of {max_tenure_days, min_score_diff}
        ...

    Returns:
        (new_active_candidates, rotation_events)
        rotation_events: list of {out_ticker, in_ticker, reason}
    """
    from datetime import datetime, date

    def tenure_days(entry_date_str: str) -> int:
        try:
            ed = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            return (date.fromisoformat(run_date) - ed).days
        except Exception:
            return 0

    def required_delta(tenure: int) -> float:
        for tier in sorted(rotation_tiers, key=lambda t: t["max_tenure_days"]):
            if tenure <= tier["max_tenure_days"]:
                return float(tier["min_score_diff"])
        return 3.0

    # Build sorted scanner candidate list by score
    scanner_sorted = sorted(
        [{"ticker": t, **v} for t, v in scanner_scores.items()],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    active = list(active_candidates)  # copy
    rotations = []
    rotations_this_run = 0

    for challenger in scanner_sorted:
        if rotations_this_run >= 1:
            break  # max 1 rotation per run

        c_score = challenger.get("score", 0)
        c_ticker = challenger["ticker"]

        # Find the weakest active candidate
        weakest = min(active, key=lambda x: x.get("score", 0))
        w_score = weakest.get("score", 0)
        tenure = tenure_days(weakest.get("entry_date", run_date))
        delta_needed = required_delta(tenure)
        consec_low = weakest.get("consecutive_low_runs", 0)

        # Emergency override: all 3 weak + strong challenger
        all_weak = all(a.get("score", 100) < 55 for a in active)
        if all_weak and c_score >= emergency_rotation_new_score:
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": "emergency_rotation",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1
            continue

        # Weak pick override: long-held low scorer loses tenure protection
        if (
            tenure >= 10
            and consec_low >= weak_pick_consecutive_runs
            and w_score < weak_pick_score_threshold
            and c_score >= 65
        ):
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": "weak_pick_override",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1
            continue

        # Standard tiered threshold
        if c_score - w_score >= delta_needed:
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": f"score_delta_{c_score - w_score:.1f}_tenure_{tenure}d",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1

    return active, rotations
