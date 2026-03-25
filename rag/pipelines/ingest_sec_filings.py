"""Ingest SEC 10-K and 10-Q filings into the RAG vector store.

Downloads filing HTML from SEC EDGAR, extracts key sections (Risk Factors,
MD&A, Business Description), chunks the text, embeds via Voyage, and stores
in Neon pgvector.

Usage:
    # Ingest recent filings for a list of tickers
    python -m rag.pipelines.ingest_sec_filings --tickers AAPL,MSFT,GOOG

    # Backfill last 2 years for all tickers in a signals file
    python -m rag.pipelines.ingest_sec_filings --from-signals --lookback-years 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
_EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
_SEC_HEADERS = {"User-Agent": "AlphaEngine research@nousergon.ai"}

# Sections to extract from 10-K / 10-Q
_TARGET_SECTIONS = [
    "Risk Factors",
    "Management's Discussion and Analysis",
    "Business",
    "Financial Statements",
    "Quantitative and Qualitative Disclosures About Market Risk",
]

# Chunk config
_CHUNK_SIZE = 400  # tokens (approximate by words * 1.3)
_CHUNK_OVERLAP = 50


def _search_filings(ticker: str, form_types: list[str], lookback_days: int = 730) -> list[dict]:
    """Search SEC EDGAR for recent filings of given types."""
    results = []
    for form_type in form_types:
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": (date.today() - timedelta(days=lookback_days)).isoformat(),
            "enddt": date.today().isoformat(),
            "forms": form_type,
        }
        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                headers=_SEC_HEADERS,
                timeout=15,
            )
            if resp.status_code != 200:
                # Fallback: use EDGAR full-text search API
                resp = requests.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": ticker, "forms": form_type, "dateRange": "custom",
                            "startdt": params["startdt"], "enddt": params["enddt"]},
                    headers=_SEC_HEADERS,
                    timeout=15,
                )
            data = resp.json() if resp.status_code == 200 else {}
            for hit in data.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                results.append({
                    "form_type": form_type,
                    "filed_date": src.get("file_date", ""),
                    "accession_number": src.get("accession_no", "").replace("-", ""),
                    "cik": src.get("entity_id", ""),
                    "company_name": src.get("entity_name", ""),
                    "url": f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id', '')}/{src.get('accession_no', '').replace('-', '')}/",
                })
        except Exception as e:
            logger.warning("EDGAR search failed for %s %s: %s", ticker, form_type, e)
        time.sleep(0.15)  # SEC rate limit: 10 req/sec
    return results


def _download_filing_html(url: str) -> str | None:
    """Download the primary filing document HTML from an EDGAR filing URL."""
    try:
        # Get the filing index page to find the primary document
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        # Find the primary document link (usually the .htm file)
        for row in soup.select("table.tableFile tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                doc_type = cells[3].get_text(strip=True)
                if doc_type in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    link = cells[2].find("a")
                    if link and link.get("href"):
                        doc_url = "https://www.sec.gov" + link["href"]
                        time.sleep(0.15)
                        doc_resp = requests.get(doc_url, headers=_SEC_HEADERS, timeout=60)
                        if doc_resp.status_code == 200:
                            return doc_resp.text
        return None
    except Exception as e:
        logger.warning("Failed to download filing from %s: %s", url, e)
        return None


def _extract_sections(html: str) -> dict[str, str]:
    """Extract target sections from filing HTML.

    Returns dict mapping section_label → text content.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n", strip=True)

    sections = {}
    for section_name in _TARGET_SECTIONS:
        # Try to find section by heading pattern
        pattern = re.compile(
            rf"(?:Item\s+\d+[A-Z]?\.?\s*)?{re.escape(section_name)}",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            start = match.start()
            # Find the next section heading (Item N.)
            next_item = re.search(r"\nItem\s+\d+[A-Z]?\.?\s+[A-Z]", text[start + len(match.group()):])
            end = start + len(match.group()) + next_item.start() if next_item else start + 50000
            section_text = text[start:end].strip()
            # Truncate very long sections (some MD&A sections are 100K+ chars)
            if len(section_text) > 50000:
                section_text = section_text[:50000]
            if len(section_text) > 200:  # skip empty/trivial sections
                sections[section_name] = section_text

    return sections


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by approximate token count.

    Uses word count * 1.3 as token approximation.
    """
    words = text.split()
    # Approximate words per chunk (tokens / 1.3)
    words_per_chunk = int(chunk_size / 1.3)
    overlap_words = int(overlap / 1.3)

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap_words
        if start >= len(words):
            break

    return chunks


def ingest_ticker(
    ticker: str,
    sector: str | None = None,
    form_types: list[str] | None = None,
    lookback_days: int = 730,
    dry_run: bool = False,
) -> int:
    """Ingest SEC filings for a single ticker.

    Returns number of documents ingested.
    """
    from rag.embeddings import embed_texts
    from rag.retrieval import ingest_document, document_exists

    if form_types is None:
        form_types = ["10-K", "10-Q"]

    filings = _search_filings(ticker, form_types, lookback_days)
    logger.info("Found %d filings for %s", len(filings), ticker)

    ingested = 0
    for filing in filings:
        filed_date_str = filing.get("filed_date", "")
        if not filed_date_str:
            continue

        try:
            filed_date = date.fromisoformat(filed_date_str[:10])
        except ValueError:
            continue

        form_type = filing["form_type"]
        if document_exists(ticker, form_type, filed_date, "sec_edgar"):
            logger.debug("Already ingested: %s %s %s", ticker, form_type, filed_date)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would ingest %s %s %s", ticker, form_type, filed_date)
            ingested += 1
            continue

        # Download and parse
        html = _download_filing_html(filing.get("url", ""))
        if not html:
            logger.warning("Could not download %s %s %s", ticker, form_type, filed_date)
            continue

        sections = _extract_sections(html)
        if not sections:
            logger.warning("No sections extracted from %s %s %s", ticker, form_type, filed_date)
            continue

        # Chunk and embed all sections
        all_chunks = []
        for section_label, section_text in sections.items():
            for chunk_text in _chunk_text(section_text):
                all_chunks.append({
                    "content": chunk_text,
                    "section_label": section_label,
                })

        if not all_chunks:
            continue

        # Embed in batch
        embeddings = embed_texts([c["content"] for c in all_chunks])
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        # Store
        doc_id = ingest_document(
            ticker=ticker,
            sector=sector,
            doc_type=form_type,
            source="sec_edgar",
            filed_date=filed_date,
            title=f"{ticker} {form_type} ({filed_date})",
            url=filing.get("url"),
            chunks=all_chunks,
        )
        if doc_id:
            ingested += 1

    return ingested


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest SEC filings into RAG store")
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker list")
    parser.add_argument("--from-signals", action="store_true", help="Load tickers from latest signals.json on S3")
    parser.add_argument("--lookback-years", type=int, default=2, help="Years of filings to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be ingested without writing")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.from_signals:
        import boto3
        s3 = boto3.client("s3")
        # Find the most recent signals file
        resp = s3.list_objects_v2(Bucket="alpha-engine-research", Prefix="signals/", Delimiter="/")
        prefixes = sorted([p["Prefix"] for p in resp.get("CommonPrefixes", [])])
        if not prefixes:
            logger.error("No signals found on S3")
            return
        latest = prefixes[-1]
        obj = s3.get_object(Bucket="alpha-engine-research", Key=f"{latest}signals.json")
        data = json.loads(obj["Body"].read())
        tickers = [s["ticker"] for s in data.get("universe", []) if s.get("ticker")]
        logger.info("Loaded %d tickers from signals", len(tickers))
    else:
        parser.error("Provide --tickers or --from-signals")
        return

    lookback_days = args.lookback_years * 365
    total = 0
    for ticker in tickers:
        n = ingest_ticker(ticker, lookback_days=lookback_days, dry_run=args.dry_run)
        total += n
        logger.info("Ingested %d filings for %s", n, ticker)

    logger.info("Total: %d filings ingested for %d tickers", total, len(tickers))


if __name__ == "__main__":
    main()
