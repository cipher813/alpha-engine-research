"""
Peer Review — Intra-team review between Quant and Qual analysts.

1. If qual added a candidate: quant reviews it (do the numbers support?)
2. Joint finalization: produce final 2-3 recommendations with combined scores.

Uses single Haiku calls (no ReAct) — the peer review is a structured evaluation,
not an open-ended exploration.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config import PER_STOCK_MODEL, MAX_TOKENS_PER_STOCK, ANTHROPIC_API_KEY, TEAM_PICKS_PER_RUN
from agents.prompt_loader import load_prompt

log = logging.getLogger(__name__)


def run_peer_review(
    team_id: str,
    quant_picks: list[dict],
    qual_assessments: list[dict],
    additional_candidate: Optional[dict],
    technical_scores: dict,
    market_regime: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run intra-team peer review and produce final recommendations.

    Args:
        team_id: Sector team identifier.
        quant_picks: Quant analyst's top 5 (ticker, quant_score, rationale).
        qual_assessments: Qual analyst's assessments (ticker, qual_score, bull_case, bear_case).
        additional_candidate: Qual's extra candidate (or None).
        technical_scores: {ticker: dict} for quant review of additional.
        market_regime: Current macro regime.

    Returns:
        {
            "team_id": str,
            "recommendations": list[dict],  # final 2-3 picks
            "additional_accepted": bool,
            "peer_review_rationale": str,
        }
    """
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_PER_STOCK,
    )

    # Step 1: If qual added a candidate, quant reviews it
    additional_accepted = False
    if additional_candidate and additional_candidate.get("ticker"):
        additional_accepted = _quant_reviews_addition(
            llm, team_id, additional_candidate, technical_scores
        )

    # Step 2: Joint finalization — select final 2-3
    all_candidates = _merge_candidates(quant_picks, qual_assessments, additional_candidate, additional_accepted)

    if len(all_candidates) <= TEAM_PICKS_PER_RUN:
        # Not enough to need selection — return all
        return {
            "team_id": team_id,
            "recommendations": all_candidates,
            "additional_accepted": additional_accepted,
            "peer_review_rationale": "All candidates advanced (fewer than max picks).",
        }

    # Joint finalization via single Haiku call
    result = _joint_finalization(llm, team_id, all_candidates, market_regime)

    return {
        "team_id": team_id,
        "recommendations": result["picks"],
        "additional_accepted": additional_accepted,
        "peer_review_rationale": result["rationale"],
    }


def _quant_reviews_addition(
    llm: ChatAnthropic,
    team_id: str,
    candidate: dict,
    technical_scores: dict,
) -> bool:
    """Quant reviews qual's additional candidate. Returns True if accepted."""
    ticker = candidate.get("ticker", "")
    ts = technical_scores.get(ticker, {})

    prompt = load_prompt("peer_review_quant_addition").format(
        team_title=team_id.title(),
        ticker=ticker,
        qual_rationale=candidate.get("rationale", "No rationale provided"),
        qual_score=candidate.get("qual_score", "N/A"),
        rsi_14=ts.get("rsi_14", "N/A"),
        macd_cross=ts.get("macd_cross", "N/A"),
        price_vs_ma50=ts.get("price_vs_ma50", "N/A"),
        price_vs_ma200=ts.get("price_vs_ma200", "N/A"),
        momentum_20d=ts.get("momentum_20d", "N/A"),
        atr_pct=ts.get("atr_pct", "N/A"),
        technical_score=ts.get("technical_score", "N/A"),
    )

    # PR 2.2 Step C: flip _quant_reviews_addition to with_structured_output.
    # Strict mode raises on parse failure; lax mode keeps the silent-False
    # fallback (rejecting the qual addition is the conservative editorial
    # behavior — the team can still produce 2-3 picks from the quant set).
    from graph.state_schemas import QuantAcceptanceVerdict
    from strict_mode import is_strict_validation_enabled

    structured_llm = llm.with_structured_output(QuantAcceptanceVerdict)
    try:
        verdict: QuantAcceptanceVerdict = structured_llm.invoke(
            [HumanMessage(content=prompt)]
        )
        log.info(
            "[peer_review:%s] quant %s qual's addition %s: %s",
            team_id,
            "accepted" if verdict.accept else "rejected",
            ticker,
            verdict.reason,
        )
        return verdict.accept
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        log.warning(
            "[peer_review:%s] quant review of %s failed: %s", team_id, ticker, e
        )

    return False


def _merge_candidates(
    quant_picks: list[dict],
    qual_assessments: list[dict],
    additional: Optional[dict],
    additional_accepted: bool,
) -> list[dict]:
    """Merge quant picks with qual assessments into combined candidates."""
    # Build lookup by ticker
    qual_by_ticker = {a["ticker"]: a for a in qual_assessments}

    merged = []
    for qp in quant_picks:
        ticker = qp["ticker"]
        qa = qual_by_ticker.get(ticker, {})
        merged.append({
            "ticker": ticker,
            "quant_score": qp.get("quant_score", 0),
            "quant_rationale": qp.get("rationale", ""),
            "qual_score": qa.get("qual_score"),
            "bull_case": qa.get("bull_case", ""),
            "bear_case": qa.get("bear_case", ""),
            "catalysts": qa.get("catalysts", []),
            "conviction": qa.get("conviction", "medium"),
            "resources_used": qa.get("resources_used", []),
        })

    # Add the additional candidate if accepted
    if additional_accepted and additional and additional.get("ticker"):
        ticker = additional["ticker"]
        if ticker not in {m["ticker"] for m in merged}:
            merged.append({
                "ticker": ticker,
                "quant_score": additional.get("quant_score", 0),
                "quant_rationale": "",
                "qual_score": additional.get("qual_score"),
                "bull_case": additional.get("rationale", ""),
                "bear_case": "",
                "catalysts": [],
                "conviction": "medium",
                "resources_used": [],
                "is_qual_addition": True,
            })

    return merged


def _joint_finalization(
    llm: ChatAnthropic,
    team_id: str,
    candidates: list[dict],
    market_regime: str,
) -> dict:
    """Single Haiku call to select final 2-3 from merged candidates."""
    candidates_text = "\n".join(
        f"  {c['ticker']}: quant={c.get('quant_score', '?')}, qual={c.get('qual_score', '?')}, "
        f"conviction={c.get('conviction', '?')}, bull={c.get('bull_case', '')[:80]}"
        for c in candidates
    )

    prompt = load_prompt("peer_review_joint_finalization").format(
        team_title=team_id.title(),
        market_regime=market_regime,
        candidates_text=candidates_text,
        team_picks_per_run=TEAM_PICKS_PER_RUN,
    )

    # PR 2.2 Step C: flip _joint_finalization to with_structured_output.
    # The combined-score fallback is load-bearing — every team MUST produce
    # 2-3 picks for the merge step downstream — so the lax-mode fallback
    # is preserved. Strict mode raises on parse failure; the operator
    # then has the choice to flip STRICT_VALIDATION=false in 30s and
    # re-run if the failure is an isolated LLM hiccup rather than a
    # systemic schema issue.
    from graph.state_schemas import JointFinalizationOutput
    from strict_mode import is_strict_validation_enabled

    structured_llm = llm.with_structured_output(JointFinalizationOutput)
    try:
        result: JointFinalizationOutput = structured_llm.invoke(
            [HumanMessage(content=prompt)]
        )
        selected = set(result.selected_tickers)
        picks = [c for c in candidates if c["ticker"] in selected]
        return {
            "picks": picks[:TEAM_PICKS_PER_RUN],
            "rationale": result.rationale,
        }
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        log.warning("[peer_review:%s] joint finalization failed: %s", team_id, e)

    # Fallback: sort by combined score and take top N
    for c in candidates:
        qs = c.get("quant_score") or 0
        qls = c.get("qual_score") or 0
        c["_combined"] = (qs + qls) / 2 if qls else qs

    candidates.sort(key=lambda x: x["_combined"], reverse=True)
    return {
        "picks": candidates[:TEAM_PICKS_PER_RUN],
        "rationale": "Fallback: selected by combined quant+qual score.",
    }
