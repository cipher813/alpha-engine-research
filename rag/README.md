# RAG — Semantic Retrieval for Research Agents

Vector retrieval over SEC filings, earnings transcripts, and thesis history. Provides the qual analyst agents with deep fundamental context beyond headlines and consensus data.

> **Retrieval-only** in this repo. The shared retrieval/db/embeddings/schema code lives in [`alpha_engine_lib.rag`](https://github.com/cipher813/alpha-engine-lib/tree/main/src/alpha_engine_lib/rag) (since lib v0.3.0). RAG **ingestion** lives in [`alpha-engine-data/rag/pipelines/`](https://github.com/cipher813/alpha-engine-data/tree/main/rag/pipelines) and runs as part of the weekly Step Function via that repo's `infrastructure/spot_data_weekly.sh`.

## Architecture

```
Ingestion (weekly)          Neon pgvector              Qual Analyst Agent
alpha-engine-data    ──→   rag.documents  ──→         @tool query_filings()
SEC + 8-K + theses   ──→   rag.chunks     ──→         semantic search + metadata filter
                            (HNSW index)   ──→         top-k results → agent context
```

## Retrieval surface used by this repo

| Caller | Imports |
|---|---|
| `agents/sector_teams/qual_tools.py` | `from alpha_engine_lib.rag import retrieve` (qual analyst's `query_filings` tool) |
| `graph/research_graph.py` | `from alpha_engine_lib.rag import is_available` (gates RAG access at graph startup) |

The lib re-exports `retrieve`, `ingest_document`, `document_exists`, `embed_texts`, `get_connection`, and `is_available`. Schema is shipped as package data (`alpha_engine_lib.rag/schema.sql`) and loaded by the data repo's `rag/preflight.py` at ingestion time.

## Environment Variables

| Var | Purpose |
|-----|---------|
| `RAG_DATABASE_URL` | Neon pooled connection string (read by `is_available` and `retrieve`) |
| `VOYAGE_API_KEY` | Voyage embedding API key (used by retrieval-time query embedding) |

## Cost

| Component | Monthly |
|-----------|---------|
| Neon pgvector (free tier) | $0 |
| Voyage embeddings (~25 stocks × weekly) | ~$0.10 |
