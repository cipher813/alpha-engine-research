"""Resolve tickers from the most recent signals.json on S3.

Centralizes the `--from-signals` lookup shared by all RAG ingest pipelines.
Historically each pipeline open-coded `prefixes[-1]` (lexicographically-latest
signals/{date}/ prefix), which crashes with NoSuchKey whenever the latest
prefix doesn't contain signals.json — a real failure mode when other producers
write into signals/{date}/ (e.g. executor's pre-alpha-engine#64 order_book_summary.json).

This helper iterates sorted prefixes backward until one contains signals.json,
then hard-fails if none do. No silent fallback.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_BUCKET = "alpha-engine-research"


def load_tickers_from_latest_signals() -> list[str]:
    """Return the tickers from the most recent signals/{date}/signals.json.

    Raises RuntimeError if no signals.json exists under any signals/ prefix.
    """
    import boto3

    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=_BUCKET, Prefix="signals/", Delimiter="/")
    prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    if not prefixes:
        raise RuntimeError(f"No signals/ prefixes found in s3://{_BUCKET}/")

    for prefix in reversed(prefixes):
        try:
            obj = s3.get_object(Bucket=_BUCKET, Key=f"{prefix}signals.json")
        except s3.exceptions.NoSuchKey:
            continue
        date_str = prefix.strip("/").split("/")[-1]
        data = json.loads(obj["Body"].read())
        tickers = [t["ticker"] for t in data.get("universe", []) if t.get("ticker")]
        logger.info(
            "Loaded %d tickers from s3://%s/%ssignals.json",
            len(tickers), _BUCKET, prefix,
        )
        return tickers

    raise RuntimeError(
        f"No signals.json found under any of {len(prefixes)} signals/ prefixes in s3://{_BUCKET}/"
    )
