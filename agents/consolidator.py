"""
Consolidator Agent (§4.4).

Single instance per run. Executes after all per-stock agents complete.
Uses claude-sonnet-4-6 (strategic model) for final synthesis.

Produces the email-ready research brief (max ~500 words).
"""

from __future__ import annotations

import re
from typing import Optional

import anthropic

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY
from agents.token_guard import check_prompt_size

from agents.prompt_loader import load_prompt

_PROMPT_TEMPLATE = load_prompt("consolidator")


def _truncate_report(text: str, max_words: int = 60) -> str:
    """Extract first max_words of a report for summary view."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _format_universe_summaries(
    news_reports: dict[str, str],
    research_reports: dict[str, str],
) -> tuple[str, str]:
    """Format 1-2 sentence extracts from universe reports."""
    news_lines = []
    research_lines = []
    for ticker in sorted(news_reports):
        news_lines.append(f"{ticker}: {_truncate_report(news_reports[ticker], 40)}")
    for ticker in sorted(research_reports):
        research_lines.append(f"{ticker}: {_truncate_report(research_reports[ticker], 40)}")
    return "\n".join(news_lines) or "None.", "\n".join(research_lines) or "None."


_GBM_SYMBOL = {"UP": "↑", "DOWN": "↓", "FLAT": "─"}


def _format_thesis_table(investment_theses: dict[str, dict]) -> str:
    """Format the universe ratings table for the consolidator prompt."""
    lines = ["Ticker | Rating | Score | Tech | News | Research | Δ | Signal | GBM | Thesis"]
    for ticker in sorted(investment_theses):
        t = investment_theses[ticker]
        score = f"{t.get('final_score', 0):.0f}"
        tech = f"{t.get('technical_score', 0):.0f}"
        news = f"{t.get('news_score', 0):.0f}"
        research = f"{t.get('research_score', 0):.0f}"
        delta = t.get("score_delta")
        delta_str = f"{delta:+.0f}" if delta is not None else "N/A"
        stale = "⚠stale" if t.get("stale_days", 0) >= 5 else ""
        consistency = "⚠inconsistent" if t.get("consistency_flag") else ""
        flags = " | ".join(filter(None, [stale, consistency]))
        signal = t.get("signal", "HOLD")
        # GBM column: direction symbol + veto marker if signal was downgraded
        gbm_dir = t.get("predicted_direction")
        gbm_sym = _GBM_SYMBOL.get(gbm_dir, "?") if gbm_dir else "?"
        if t.get("gbm_veto"):
            gbm_sym += "✗"   # ✗ = veto fired: ENTER downgraded to HOLD
        thesis = _truncate_report(t.get("thesis_summary", ""), 30)
        row = f"{ticker} | {t.get('rating', '?')} | {score} | {tech} | {news} | {research} | {delta_str} | {signal}"
        row += f" | {gbm_sym}"
        if flags:
            row += f" | {flags}"
        row += f" | {thesis}"
        lines.append(row)
    return "\n".join(lines)


def _format_candidates(candidates: list[dict]) -> str:
    """Format top 3 buy candidates for the prompt."""
    if not candidates:
        return "None active this run."
    lines = []
    for c in candidates:
        ticker = c.get("symbol", c.get("ticker", ""))
        score = f"{c.get('score', 0):.0f}"
        delta = c.get("score_delta")
        delta_str = f"{delta:+.0f}" if delta is not None else "N/A"
        thesis = c.get("thesis_summary", "")
        catalyst = c.get("key_catalyst", "")
        risk = c.get("key_risk", "")
        status = c.get("status", "CONTINUING")
        lines.append(
            f"{ticker} | Score:{score} | Δ{delta_str} | {thesis} | "
            f"Catalyst:{catalyst} | Risk:{risk} | {status}"
        )
    return "\n".join(lines)


def _format_consistency_flags(investment_theses: dict[str, dict]) -> str:
    flags = [
        f"{t}: thesis/score inconsistency flagged"
        for t, v in investment_theses.items()
        if v.get("consistency_flag")
    ]
    return "\n".join(flags) if flags else "None."


def _format_sector_ratings(sector_ratings: dict[str, dict]) -> str:
    """Format sector allocation table for the consolidator prompt."""
    if not sector_ratings:
        return "Not available."
    _symbol = {"overweight": "▲", "underweight": "▼", "market_weight": "●"}
    lines = ["Sector | Rating | Rationale"]
    for sector in sorted(sector_ratings):
        entry = sector_ratings[sector]
        rating = entry.get("rating", "market_weight")
        symbol = _symbol.get(rating, "●")
        rationale = entry.get("rationale", "")
        lines.append(f"{sector} | {symbol} {rating.upper()} | {rationale}")
    return "\n".join(lines)


def _format_performance_summary(perf: dict) -> str:
    if not perf:
        return "No performance data yet (requires 30+ trading days)."
    acc_10d = perf.get("accuracy_10d")
    acc_30d = perf.get("accuracy_30d")
    sample = perf.get("sample_size", 0)
    recal = perf.get("recalibration_flag", False)
    parts = [f"BUY signal accuracy (n={sample}):"]
    if acc_10d is not None:
        parts.append(f"10d vs SPY: {acc_10d:.0f}%")
    if acc_30d is not None:
        parts.append(f"30d vs SPY: {acc_30d:.0f}%")
    if recal:
        parts.append("RECALIBRATION FLAG: true")
    return " | ".join(parts)


def run_consolidator_agent(
    run_date: str,
    macro_report: str,
    universe_news_reports: dict[str, str],
    universe_research_reports: dict[str, str],
    candidate_full_news: dict[str, str],
    candidate_full_research: dict[str, str],
    investment_theses: dict[str, dict],
    active_candidates: list[dict],
    performance_summary: dict,
    sector_ratings: dict[str, dict] | None = None,
    is_early_close: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Run the Consolidator Agent to produce the final email body.

    Returns the consolidated markdown report as a string.
    """
    client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    news_summary, research_summary = _format_universe_summaries(
        universe_news_reports, universe_research_reports
    )
    thesis_table = _format_thesis_table(investment_theses)
    candidates_str = _format_candidates(active_candidates)
    consistency_flags = _format_consistency_flags(investment_theses)
    performance_str = _format_performance_summary(performance_summary)

    # Full reports for buy candidates (not summaries)
    candidate_news_full = "\n\n".join(
        f"=== {t} NEWS ===\n{r}" for t, r in candidate_full_news.items()
    ) or "None."
    candidate_research_full = "\n\n".join(
        f"=== {t} RESEARCH ===\n{r}" for t, r in candidate_full_research.items()
    ) or "None."

    sector_ratings_table = _format_sector_ratings(sector_ratings or {})

    prompt = _PROMPT_TEMPLATE.format(
        macro_report=macro_report or "No macro report available.",
        sector_ratings_table=sector_ratings_table,
        news_reports_by_ticker=news_summary,
        research_reports_by_ticker=research_summary,
        candidate_full_news_reports=candidate_news_full,
        candidate_full_research_reports=candidate_research_full,
        thesis_table=thesis_table,
        candidates=candidates_str,
        consistency_flags=consistency_flags,
        performance_summary=performance_str,
    )

    # Add early-close note to prompt if needed
    if is_early_close:
        prompt += "\n\nNOTE: Today is an early-close trading day (market closes at 1pm ET)."

    prompt = check_prompt_size(prompt, MAX_TOKENS_STRATEGIC, caller="consolidator")

    response = client.messages.create(
        model=STRATEGIC_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
