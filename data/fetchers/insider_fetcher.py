"""
data/fetchers/insider_fetcher.py — SEC EDGAR Form 4 insider trading scanner (O13).

Fetches insider transactions from SEC EDGAR via EdgarTools, detects cluster
buying patterns (3+ C-level insiders buying within 30 days), and provides
insider sentiment signals for research scoring.

SEC EDGAR rate limit: 10 req/sec — 0.2s delay between tickers.
Only fetched for buy candidates (~30-50 tickers), not the full S&P 900.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

_RATE_LIMIT_DELAY = 0.25  # stay under 10 req/sec limit
_MAX_FILINGS = 20  # cap Form 4 filings per ticker


def fetch_insider_activity(
    tickers: list[str],
    lookback_days: int = 90,
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch insider trading activity for a list of tickers from SEC EDGAR.

    Uses EdgarTools to parse Form 4 filings and detect cluster buying patterns.

    Returns per ticker:
        cluster_buy: bool — True if 3+ unique insiders bought in last 30 days
        unique_buyers_30d: int — count of unique insiders who bought in last 30 days
        total_buy_value_30d: float — total dollar value of buys in last 30 days
        net_sentiment: float — net buy/sell ratio (-1 to +1)
        transactions: list[dict] — top 10 recent transactions for display
    """
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    start_date = today - timedelta(days=lookback_days)
    results: dict[str, dict] = {}

    try:
        from edgartools import Company
    except ImportError:
        log.warning("edgartools not installed — insider data unavailable. pip install edgartools")
        for ticker in tickers:
            results[ticker] = _empty_result()
        return results

    for ticker in tickers:
        try:
            company = Company(ticker)
            filings = company.get_filings(form="4")

            transactions: list[dict] = []
            for filing in list(filings)[:_MAX_FILINGS]:
                try:
                    filing_date_str = str(filing.filing_date)
                    filing_date = datetime.strptime(filing_date_str[:10], "%Y-%m-%d")

                    if filing_date < start_date:
                        continue

                    days_ago = (today - filing_date).days

                    # Parse the Form 4 XML
                    try:
                        form4 = filing.obj()
                        if hasattr(form4, "transactions"):
                            for txn in form4.transactions:
                                txn_type = "BUY" if getattr(txn, "acquired", False) else "SELL"
                                shares = getattr(txn, "shares", 0) or 0
                                price = getattr(txn, "price", 0) or 0
                                value = shares * price

                                transactions.append({
                                    "date": filing_date_str[:10],
                                    "days_ago": days_ago,
                                    "insider": str(getattr(txn, "reporting_owner", "Unknown")),
                                    "title": str(getattr(txn, "title", "")),
                                    "type": txn_type,
                                    "shares": int(shares),
                                    "value": round(value, 2),
                                })
                        elif hasattr(form4, "owner_name"):
                            # Simpler form4 parsing fallback
                            transactions.append({
                                "date": filing_date_str[:10],
                                "days_ago": days_ago,
                                "insider": str(getattr(form4, "owner_name", "Unknown")),
                                "title": "",
                                "type": "BUY",
                                "shares": 0,
                                "value": 0,
                            })
                    except Exception:
                        # Form 4 parsing can fail for unusual filings
                        pass

                except Exception:
                    continue

            # Cluster detection: unique buyers in last 30 days
            buys_30d = [t for t in transactions if t["type"] == "BUY" and t["days_ago"] <= 30]
            unique_buyers = len(set(t["insider"] for t in buys_30d))
            total_buy_value = sum(t["value"] for t in buys_30d)

            # Net sentiment: ratio of buy value to total value
            total_buys = sum(t["value"] for t in transactions if t["type"] == "BUY")
            total_sells = sum(t["value"] for t in transactions if t["type"] == "SELL")
            total_value = total_buys + total_sells
            if total_value > 0:
                net_sentiment = round((total_buys - total_sells) / total_value, 3)
            else:
                net_sentiment = 0.0

            results[ticker] = {
                "cluster_buy": unique_buyers >= 3,
                "unique_buyers_30d": unique_buyers,
                "total_buy_value_30d": round(total_buy_value, 2),
                "net_sentiment": net_sentiment,
                "transactions": transactions[:10],
            }

            time.sleep(_RATE_LIMIT_DELAY)

        except Exception as e:
            log.warning("Insider data fetch failed for %s: %s", ticker, e)
            results[ticker] = _empty_result()

    log.info("Fetched insider activity for %d/%d tickers", len(results), len(tickers))
    return results


def _empty_result() -> dict:
    """Return neutral insider data when fetching fails."""
    return {
        "cluster_buy": False,
        "unique_buyers_30d": 0,
        "total_buy_value_30d": 0.0,
        "net_sentiment": 0.0,
        "transactions": [],
    }


def format_insider_summary(insider_data: dict) -> str:
    """
    Format insider activity data into a human-readable summary for
    the research agent prompt.

    Returns empty string if no meaningful activity.
    """
    if not insider_data or insider_data.get("unique_buyers_30d", 0) == 0:
        transactions = insider_data.get("transactions", [])
        if not transactions:
            return ""
        # Only sells or no 30d activity
        sells = [t for t in transactions if t["type"] == "SELL"]
        if not sells:
            return ""
        lines = ["Insider Activity (90 days):"]
        lines.append(f"- Net sentiment: {'BEARISH' if insider_data.get('net_sentiment', 0) < -0.3 else 'NEUTRAL'}")
        for t in sells[:3]:
            lines.append(f"  - {t['insider']}: SELL {t['shares']:,} shares (${t['value']:,.0f}) on {t['date']}")
        return "\n".join(lines)

    lines = ["Insider Activity (90 days):"]
    unique_buyers = insider_data["unique_buyers_30d"]
    cluster = insider_data.get("cluster_buy", False)

    if cluster:
        lines.append(f"- {unique_buyers} unique insiders bought in last 30 days (CLUSTER BUYING detected)")
    elif unique_buyers > 0:
        lines.append(f"- {unique_buyers} insider(s) bought in last 30 days")

    # Show top transactions
    for t in insider_data.get("transactions", [])[:5]:
        action = t["type"]
        lines.append(
            f"  - {t['insider']} {action} {t['shares']:,} shares "
            f"(${t['value']:,.0f}) on {t['date']}"
        )

    sentiment = insider_data.get("net_sentiment", 0)
    if sentiment > 0.3:
        lines.append("- Net sentiment: BULLISH")
    elif sentiment < -0.3:
        lines.append("- Net sentiment: BEARISH")
    else:
        lines.append("- Net sentiment: NEUTRAL")

    return "\n".join(lines)


def cache_insider_to_s3(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Cache insider data to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/insider/{date_str}.json"
        # Strip non-serializable transaction details for cache
        cache_data = {}
        for ticker, d in data.items():
            cache_data[ticker] = {
                k: v for k, v in d.items() if k != "transactions"
            }
            cache_data[ticker]["n_transactions"] = len(d.get("transactions", []))
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(cache_data, default=str),
            ContentType="application/json",
        )
        log.info("Cached insider data to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to cache insider data to S3: %s", e)
