"""
Episodic memory extraction — converts failed BUY signal outcomes into lessons.

Runs after performance_tracker populates score_performance with 10d/30d outcomes.
Uses Haiku to extract a 1-2 sentence lesson from each failed signal.

Cost-capped: MAX_EPISODIC_EXTRACTIONS per run to prevent runaway costs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

MAX_EPISODIC_EXTRACTIONS = 15  # hard cap per run (~$0.08 max)


def extract_memories(db_conn: sqlite3.Connection, api_key: str | None = None) -> int:
    """
    Extract episodic memories from recently completed score_performance outcomes.

    Finds BUY signals where beat_spy_10d = 0 (underperformed) that don't yet
    have a memory_episodes entry. Uses Haiku to generate a lesson for each.

    Returns number of new memories created.
    """
    # Find failed signals without existing memories
    rows = db_conn.execute("""
        SELECT sp.symbol, sp.score_date, sp.score, sp.return_10d, sp.spy_10d_return,
               it.conviction, it.thesis_summary, it.signal,
               ms.market_regime, ms.vix
        FROM score_performance sp
        LEFT JOIN investment_thesis it
            ON sp.symbol = it.symbol AND sp.score_date = it.date
        LEFT JOIN macro_snapshots ms
            ON sp.score_date = ms.date
        LEFT JOIN memory_episodes me
            ON sp.symbol = me.ticker AND sp.score_date = me.signal_date
        WHERE sp.beat_spy_10d = 0
            AND me.id IS NULL
        ORDER BY sp.score_date DESC
        LIMIT ?
    """, (MAX_EPISODIC_EXTRACTIONS * 2,)).fetchall()  # fetch more, cap at extraction

    if not rows:
        logger.info("[memory_extractor] no new failed signals to process")
        return 0

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from config import ANTHROPIC_API_KEY, PER_STOCK_MODEL

    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=256,
    )

    n_created = 0
    for r in rows:
        if n_created >= MAX_EPISODIC_EXTRACTIONS:
            logger.warning("[memory_extractor] hit cap of %d extractions", MAX_EPISODIC_EXTRACTIONS)
            break

        symbol, score_date, score, return_10d, spy_return = r[0], r[1], r[2], r[3], r[4]
        conviction, thesis_summary, signal = r[5], r[6], r[7]
        regime, vix = r[8], r[9]

        outcome_vs_spy = (return_10d or 0) - (spy_return or 0)

        prompt = f"""A BUY signal for {symbol} on {score_date} (score: {score}, conviction: {conviction}) underperformed SPY by {outcome_vs_spy:.1%} over 10 days.

Thesis: {(thesis_summary or 'N/A')[:300]}
Market regime: {regime or 'unknown'}, VIX: {vix or '?'}
Stock return: {return_10d:.1%}, SPY return: {spy_return:.1%}

In 1-2 sentences, what lesson should be remembered for future analysis of {symbol} or similar stocks? Also provide 2-3 pattern tags (e.g., "earnings", "margin_compression", "momentum_reversal").

Respond ONLY with JSON: {{"lesson": "...", "pattern_tags": ["...", "..."]}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content
            # Extract JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                lesson = data.get("lesson", "")
                tags = json.dumps(data.get("pattern_tags", []))
            else:
                lesson = text[:200]
                tags = "[]"

            # Get sector from investment_thesis or sector_map
            sector_row = db_conn.execute(
                "SELECT sector FROM stock_archive WHERE ticker = ? LIMIT 1", (symbol,)
            ).fetchone()
            sector = sector_row[0] if sector_row else None

            db_conn.execute(
                "INSERT OR IGNORE INTO memory_episodes "
                "(ticker, signal_date, score, rating, conviction, thesis_summary, "
                "outcome_10d, outcome_vs_spy, lesson, sector, pattern_tags, created_date) "
                "VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, score_date, score, conviction, (thesis_summary or "")[:500],
                 return_10d, outcome_vs_spy, lesson, sector, tags, score_date),
            )
            db_conn.commit()
            n_created += 1
            logger.info("[memory_extractor] created memory for %s (%s): %s",
                       symbol, score_date, lesson[:80])

        except Exception as e:
            logger.warning("[memory_extractor] failed for %s (%s): %s", symbol, score_date, e)

    logger.info("[memory_extractor] created %d new episodic memories", n_created)
    return n_created
