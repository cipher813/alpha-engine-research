# RAG — Semantic Retrieval for Research Agents

Hybrid retrieval (vector + Full-Text Search) over SEC filings, earnings transcripts, and thesis history. Provides the qual analyst agents with deep fundamental context beyond headlines and consensus data.

> **Retrieval-only** in this repo. The shared retrieval/db/embeddings/schema code lives in [`alpha_engine_lib.rag`](https://github.com/cipher813/alpha-engine-lib/tree/main/src/alpha_engine_lib/rag) (since lib v0.3.0; hybrid-retrieval API since v0.6.0). RAG **ingestion** lives in [`alpha-engine-data/rag/pipelines/`](https://github.com/cipher813/alpha-engine-data/tree/main/rag/pipelines) and runs as part of the weekly Step Function via that repo's `infrastructure/spot_data_weekly.sh`.

## Architecture

```
Ingestion (weekly)          Neon pgvector + FTS              Qual Analyst Agent
alpha-engine-data    ──→   rag.documents             ──→     @tool query_filings()
SEC + 8-K + theses   ──→   rag.chunks                ──→     hybrid retrieval
                            ├─ HNSW on embedding      ──→     top-k results +
                            └─ GIN on content_tsv             component scores
```

## Retrieval methods

The lib's `retrieve()` API supports three methods (since v0.6.0):

| Method | Strong on | Weak on |
|---|---|---|
| `vector` | Conceptual / paraphrased queries (competitive moat, strategy) | Exact-term surfaces (tickers, $ amounts, filing types) |
| `keyword` | Literal-term matches (PostgreSQL FTS via `ts_rank_cd`) | Conceptual queries lacking literal overlap |
| `hybrid` | Both — blends top_k from each side, normalizes via min-max within the candidate set, returns weighted blend |

**This repo's qual analyst calls `retrieve(method="hybrid", vector_weight=0.7)`** at `agents/sector_teams/qual_tools.py::query_filings`. Per-call component scores (`vector_score` / `keyword_score` / `combined_score`) are emitted in a structured `RAG_RETRIEVE` INFO log line for decision-artifact capture and LangSmith trace observability.

`vector_weight=0.7` is the ROADMAP-spec'd starting default. Empirical calibration may move the value once enough data accumulates — see "Calibration owed" below.

## Retrieval surface used by this repo

| Caller | Imports |
|---|---|
| `agents/sector_teams/qual_tools.py` | `from alpha_engine_lib.rag import retrieve` (qual analyst's `query_filings` tool, hybrid mode) |
| `graph/research_graph.py` | `from alpha_engine_lib.rag import is_available` (gates RAG access at graph startup) |

The lib re-exports `retrieve`, `ingest_document`, `document_exists`, `embed_texts`, `get_connection`, and `is_available`. Schema is shipped as package data (`alpha_engine_lib.rag/schema.sql`); the `0001_content_tsv.sql` migration is shipped at `alpha_engine_lib.rag/migrations/`.

## Environment Variables

| Var | Purpose |
|-----|---------|
| `RAG_DATABASE_URL` | Neon pooled connection string (read by `is_available` and `retrieve`) |
| `VOYAGE_API_KEY` | Voyage embedding API key (used by retrieval-time query embedding) |

## Cost

| Component | Monthly |
|-----------|---------|
| Neon pgvector + FTS (free tier) | $0 |
| Voyage embeddings (~25 stocks × weekly) | ~$0.10 |

## Eval harness

`evals/rag_retrieval.py` + `scripts/run_rag_retrieval_eval.py` ship a 6-condition × 3-cutoff recall@k harness. Empirical calibration of `vector_weight` is the intended use case but the harness is general-purpose for any retrieval regression scan.

```bash
# Curate evals/rag_retrieval_queries.yaml with hand-picked
# (query → expected_chunk_id) pairs (or seed via
# scripts/seed_rag_retrieval_queries.py for a starting point).

python scripts/run_rag_retrieval_eval.py
# → ~/Development/alpha-engine-docs/private/rag-retrieval-eval-{date}.md
```

## Calibration owed

The 2026-05-08 first-pass eval against 30 auto-seeded queries was inconclusive — recall numbers stayed under 0.10 across all conditions due to (a) regex-artifact queries in the seed and (b) a harness/production mismatch where the eval doesn't apply the `tickers=[…]` pre-filter that `query_filings` uses in production. Two paths to genuine calibration:

1. **Refine the seed test set** — replace the regex-artifact entries, extend the YAML schema to carry per-query `tickers` / `doc_types` / `min_date` filters, re-run.
2. **Passive measurement from production** — once the qual analyst has accumulated ~2 weeks of decision artifacts under hybrid mode, mine the `RAG_RETRIEVE` log lines + downstream agent citations to compute "did hybrid surface chunks the agent ended up citing?" against vector. No synthetic test set required.

Path 2 is cleaner. Until calibration lands, `vector_weight=0.7` stays the default.
