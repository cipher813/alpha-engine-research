# RAG — Semantic Retrieval for Research Agents

Vector retrieval over SEC filings, earnings transcripts, and thesis history. Provides the qual analyst agents with deep fundamental context beyond headlines and consensus data.

> **As of alpha-engine-lib v0.3.0, the shared retrieval/db/embeddings/schema code lives in [`alpha_engine_lib.rag`](https://github.com/cipher813/alpha-engine-lib/tree/main/src/alpha_engine_lib/rag).** This folder now retains only the ingestion pipelines + tests. Retrieval consumers (`graph/research_graph.py`, `agents/sector_teams/qual_tools.py`) import from the lib.

## Architecture

```
Ingestion (weekly)          Neon pgvector              Qual Analyst Agent
SEC EDGAR 10-K/10-Q  ──→   rag.documents  ──→         @tool query_filings()
FMP Transcripts      ──→   rag.chunks     ──→         semantic search + metadata filter
Thesis History       ──→   (HNSW index)   ──→         top-k results → agent context
```

## Components

| Location | File | Purpose |
|---|---|---|
| `alpha_engine_lib.rag` | `embeddings.py` | Voyage voyage-3-lite wrapper (512d), batch support |
| `alpha_engine_lib.rag` | `db.py` | Neon PostgreSQL connection management |
| `alpha_engine_lib.rag` | `retrieval.py` | Filtered similarity search + document ingestion |
| `alpha_engine_lib.rag` | `schema.sql` | pgvector table definitions + HNSW index |
| this repo | `pipelines/ingest_sec_filings.py` | 10-K/10-Q section extraction + chunking |
| this repo | `pipelines/ingest_earnings_transcripts.py` | FMP transcript ingestion (currently blocked — see below) |
| this repo | `pipelines/ingest_theses.py` | Self-ingestion of thesis_history from research.db |
| this repo | `pipelines/_signals_resolver.py` | Shared S3 signals → ticker-list resolver |

> **Note on canonical ingestion home:** `alpha-engine-data` also has a `rag/pipelines/` directory. Reconciliation between research's pipelines (which use `_signals_resolver`) and data's pipelines (which inline the signals lookup) is a separate cleanup arc. For now, both live; the production Saturday SF state runs whichever target the SSM document points to.

## Ingestion CLI

```bash
# SEC filings — backfill 2 years for all population stocks
.venv/bin/python -m rag.pipelines.ingest_sec_filings --from-signals --lookback-years 2

# Thesis history
.venv/bin/python -m rag.pipelines.ingest_theses --db-path /path/to/research.db

# Dry run (preview without writing)
.venv/bin/python -m rag.pipelines.ingest_sec_filings --tickers AAPL,MSFT --dry-run
```

## Environment Variables

| Var | Purpose |
|-----|---------|
| `RAG_DATABASE_URL` | Neon pooled connection string |
| `VOYAGE_API_KEY` | Voyage embedding API key |
| `FMP_API_KEY` | FMP API key (for transcripts) |

## Active Data Sources

### SEC Filings (10-K / 10-Q)
- Source: EDGAR submissions API (`data.sec.gov/submissions/`)
- Sections extracted: Risk Factors, MD&A, Business, Market Risk Disclosures
- Chunked at ~400 tokens with 50-token overlap
- Dedup by (ticker, doc_type, filed_date, source)

### Thesis History
- Source: `thesis_history` table in research.db
- Embeds bull_case, bear_case, catalysts, risks, conviction_rationale
- Note: currently empty for most records (CIO writes score/conviction but not text fields). Will populate as research agents produce richer theses.

## Future Opportunities

### Earnings Call Transcripts
Pipeline built (`ingest_earnings_transcripts.py`) but **FMP deprecated its transcript endpoint** (legacy-only as of Aug 2025). Alternatives:
- **Alpha Vantage** premium tier (~$50/month) — full transcript text
- **Free transcript sources** — Motley Fool publishes partial transcripts; would need a scraper
- **Whisper + podcast feeds** — some earnings calls are available as audio; could transcribe with Whisper

When a transcript source becomes available, the pipeline is ready — swap `_fetch_transcript()`.

### Analyst Reports
Full analyst reports (Goldman, Morgan Stanley, etc.) are behind paywalls. If access becomes available, the chunking + embedding pipeline generalizes to any long-form document.

## Cost

| Component | Monthly |
|-----------|---------|
| Neon pgvector (free tier) | $0 |
| Voyage embeddings (~25 stocks × weekly) | ~$0.10 |
