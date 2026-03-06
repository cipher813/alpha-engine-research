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

_PROMPT_TEMPLATE = """\
You are a portfolio research director. Synthesize the following agent
reports into a concise daily research brief for an investor.

MACRO REPORT:
{macro_report}

SECTOR RATINGS (from macro agent — use this section as the allocation signal table):
{sector_ratings_table}

NEWS REPORTS — UNIVERSE STOCKS (summary only):
{news_reports_by_ticker}
← 1–2 sentence extract per ticker; full reports omitted for space

RESEARCH REPORTS — UNIVERSE STOCKS (summary only):
{research_reports_by_ticker}
← 1–2 sentence extract per ticker

BUY CANDIDATE FULL REPORTS — read these carefully, these are the priority:
{candidate_full_news_reports}
← Full ~300-word news report per buy candidate (not a summary)
{candidate_full_research_reports}
← Full ~300-word research report per buy candidate (not a summary)

UNIVERSE INVESTMENT THESES:
{thesis_table}
← ticker | rating | score | tech | news | research | Δ | thesis_summary
  Rating = BUY (70+) / HOLD (40-69) / SELL (0-39). All represent expected relative return vs SPY over the next ~12 months.

RATING DEFINITIONS (12-month horizon):
- BUY: Expected to match or outperform the market over the next 12 months.
- HOLD: Expected flat to underperformance vs market over the next 12 months.
- SELL: Expected to underperform the market over the next 12 months.

TOP 3 BUY CANDIDATES (from scanner pipeline — separate from tracked universe):
{candidates}
← ticker | score | score_delta | 1-sentence thesis | catalyst | risk
   | status: CONTINUING | NEW_ENTRY | RETURNED(N tenures, last demoted DATE)
   If "None active", omit section (e) entirely — do NOT substitute universe stocks.

CONSISTENCY FLAGS (pre-computed):
{consistency_flags}
← List of tickers where thesis direction and score are inconsistent.
   E.g.: "PLTR: thesis is clearly bullish but score is 34 — verify scoring inputs"

PERFORMANCE TRACKER:
{performance_summary}
← BUY signal accuracy stats (10d and 30d vs. SPY). Flag if recalibration_flag = true.

Instructions:
1. Write a consolidated research brief, maximum 700 words.
2. Structure (strictly in this order):
   a. MACRO REGIME SUMMARY — write this section FIRST, before any ratings.
      Cover: current market regime (bull/neutral/caution/bear), the key macro forces
      driving it (rates, inflation, VIX, growth outlook), and which sectors face
      notable headwinds or tailwinds right now. Be specific — name the sectors and
      explain the mechanism (e.g., "Rising 10yr yields are a headwind for Real Estate
      and Utilities; Financials benefit from steepening curve"). This section primes
      the reader to interpret every stock rating that follows in its proper context.
   b. SECTOR ALLOCATION — render the SECTOR RATINGS table directly from the data
      provided. Format as: Sector | Rating | Rationale.
      Use ▲ for overweight, ▼ for underweight, ● for market_weight.
      This section is the allocation signal table for the executor.
   c. NOTABLE DEVELOPMENTS — 2-4 bullets on the most material news/earnings/analyst
      actions across the universe since the last run.
   d. UNIVERSE RATINGS TABLE — Ticker | Rating | Score | Rationale (1 sentence).
      For each stock, write a brief rationale explaining the rating. Where macro is
      a significant driver, say so (e.g., "Financials tailwind lifts score, strong
      momentum" or "Utilities headwind from rising rates; otherwise solid fundamentals").
   e. TOP 3 BUY CANDIDATES — richer 3–4 sentence thesis per candidate.

3. SECTOR ALLOCATION details:
   - Render every sector in the SECTOR RATINGS table. Do not omit any.
   - Use the rationale from the data verbatim or improve it slightly — do not invent new rationale.
   - The ▲/▼/● symbols must match the rating exactly.

4. UNIVERSE RATINGS TABLE details:
   - Include Ticker | Rating | Score | Rationale
   - Base rationale on tech/news/research scores and thesis_summary above
   - Note macro sector impact where relevant (it explains why two stocks with similar
     fundamentals may have different ratings in the current environment)

5. For section (e): ONLY include it if buy candidates are listed above.
   If the candidates field says "None active", omit section (e) entirely.
   Do NOT substitute universe stocks — the buy candidates come exclusively
   from the scanner pipeline (stocks outside the tracked universe).
   When candidates are present, synthesize the FULL reports (not summaries) —
   provide a richer 3–4 sentence thesis per candidate than you do for universe stocks.
6. Flag any candidate that was newly promoted this run (NEW_ENTRY).
7. Flag any candidate that is a re-promotion (RETURNED) with the number of prior
   tenures and last demotion date — this history adds context.
8. If any consistency_flags exist, note them briefly: "Note: {{ticker}} thesis/score
   inconsistency flagged — review recommended."
9. Add ⚠stale after scores in the ratings table that have stale_flag = true.
10. Be concise, data-driven, and actionable. No filler language.
11. If today is an early-close trading day, note briefly that the market
    closes at 1pm ET and intraday volatility may be elevated on thin volume.
12. If performance_summary shows recalibration_flag = true, add a note at the end:
    "⚠ Scoring recalibration may be needed — BUY signal accuracy has fallen below 55%."

Output the brief in clean markdown, suitable for email.
"""


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


def _format_thesis_table(investment_theses: dict[str, dict]) -> str:
    """Format the universe ratings table for the consolidator prompt."""
    lines = ["Ticker | Rating | Score | Tech | News | Research | Δ | Thesis"]
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
        status = " | ".join(filter(None, [stale, consistency]))
        thesis = _truncate_report(t.get("thesis_summary", ""), 30)
        lines.append(f"{ticker} | {t.get('rating', '?')} | {score} | {tech} | {news} | {research} | {delta_str} | {status} | {thesis}")
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

    response = client.messages.create(
        model=STRATEGIC_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
