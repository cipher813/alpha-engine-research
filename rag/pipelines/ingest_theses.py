"""Ingest Alpha Engine thesis history into the RAG vector store.

Embeds bull_case, bear_case, catalysts, and risks from the thesis_history
table in research.db. Enables semantic search over prior reasoning across
weeks (e.g., "What were the bearish catalysts for AAPL last quarter?").

Usage:
    # Ingest all thesis records from research.db
    python -m rag.pipelines.ingest_theses --db-path /path/to/research.db

    # Ingest only new records since last run
    python -m rag.pipelines.ingest_theses --db-path /path/to/research.db --since 2026-03-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import date

logger = logging.getLogger(__name__)


def _load_theses(db_path: str, since: str | None = None) -> list[dict]:
    """Load thesis records from SQLite thesis_history table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sql = "SELECT * FROM thesis_history"
    params = []
    if since:
        sql += " WHERE run_date >= ?"
        params.append(since)
    sql += " ORDER BY run_date DESC"

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def _thesis_to_chunks(thesis: dict) -> list[dict]:
    """Convert a thesis record into text chunks with section labels."""
    ticker = thesis.get("ticker", "UNKNOWN")
    run_date = thesis.get("run_date", "")
    author = thesis.get("author", "")

    chunks = []

    # Bull case
    bull = thesis.get("bull_case", "")
    if bull and len(bull) > 50:
        chunks.append({
            "content": f"[{ticker} Bull Case — {run_date} by {author}]\n{bull}",
            "section_label": "bull_case",
        })

    # Bear case
    bear = thesis.get("bear_case", "")
    if bear and len(bear) > 50:
        chunks.append({
            "content": f"[{ticker} Bear Case — {run_date} by {author}]\n{bear}",
            "section_label": "bear_case",
        })

    # Catalysts
    catalysts = thesis.get("catalysts", "")
    if catalysts:
        if isinstance(catalysts, str):
            try:
                catalysts = json.loads(catalysts)
            except json.JSONDecodeError:
                pass
        if isinstance(catalysts, list):
            catalysts_text = "\n".join(f"- {c}" for c in catalysts)
        else:
            catalysts_text = str(catalysts)
        if len(catalysts_text) > 30:
            chunks.append({
                "content": f"[{ticker} Catalysts — {run_date} by {author}]\n{catalysts_text}",
                "section_label": "catalysts",
            })

    # Risks
    risks = thesis.get("risks", "")
    if risks:
        if isinstance(risks, str):
            try:
                risks = json.loads(risks)
            except json.JSONDecodeError:
                pass
        if isinstance(risks, list):
            risks_text = "\n".join(f"- {r}" for r in risks)
        else:
            risks_text = str(risks)
        if len(risks_text) > 30:
            chunks.append({
                "content": f"[{ticker} Risks — {run_date} by {author}]\n{risks_text}",
                "section_label": "risks",
            })

    # Conviction rationale
    rationale = thesis.get("rationale", "") or thesis.get("conviction_rationale", "")
    if rationale and len(rationale) > 30:
        chunks.append({
            "content": f"[{ticker} Conviction Rationale — {run_date} by {author}]\n{rationale}",
            "section_label": "conviction_rationale",
        })

    return chunks


def ingest_theses(
    db_path: str,
    since: str | None = None,
    dry_run: bool = False,
) -> int:
    """Ingest thesis records from research.db into RAG store.

    Returns number of thesis documents ingested.
    """
    from rag.embeddings import embed_texts
    from rag.retrieval import ingest_document, document_exists

    theses = _load_theses(db_path, since)
    logger.info("Loaded %d thesis records from %s", len(theses), db_path)

    ingested = 0
    for thesis in theses:
        ticker = thesis.get("ticker", "")
        run_date_str = thesis.get("run_date", "")
        if not ticker or not run_date_str:
            continue

        try:
            filed_date = date.fromisoformat(run_date_str[:10])
        except ValueError:
            continue

        author = thesis.get("author", "")
        doc_type = "thesis"
        source = "alpha_engine"

        if document_exists(ticker, doc_type, filed_date, source):
            continue

        chunks = _thesis_to_chunks(thesis)
        if not chunks:
            continue

        if dry_run:
            logger.info("[DRY RUN] Would ingest %s thesis %s (%d chunks)", ticker, filed_date, len(chunks))
            ingested += 1
            continue

        # Embed
        embeddings = embed_texts([c["content"] for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        doc_id = ingest_document(
            ticker=ticker,
            sector=thesis.get("sector"),
            doc_type=doc_type,
            source=source,
            filed_date=filed_date,
            title=f"{ticker} thesis — {filed_date} ({author})",
            url=None,
            chunks=chunks,
        )
        if doc_id:
            ingested += 1

    return ingested


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest thesis history into RAG store")
    parser.add_argument("--db-path", required=True, help="Path to research.db")
    parser.add_argument("--since", type=str, help="Only ingest theses from this date forward (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = ingest_theses(args.db_path, since=args.since, dry_run=args.dry_run)
    logger.info("Ingested %d thesis documents", n)


if __name__ == "__main__":
    main()
