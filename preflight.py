"""
Research Lambda preflight: connectivity checks run at the top of each
handler invocation before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into two mode-specific sequences matching the
research Lambdas.

Modes:

- ``"weekly"`` — ``lambda/handler.py``, the weekly research pipeline.
  AWS_REGION + ANTHROPIC_API_KEY + S3 bucket reachable + ArcticDB ``universe``
  library reachable with SPY's last row populated. Phase 7c (2026-04-17) made
  ArcticDB the only price source for the weekly path, so an ArcticDB outage
  is now a hard failure rather than a degraded-mode scenario.
- ``"alerts"`` — ``lambda/alerts_handler.py``, the 30-minute intraday
  price alert Lambda. AWS_REGION + S3 bucket only; alerts do not call
  Anthropic and still read intraday bars from yfinance (ArcticDB is daily
  only — see ROADMAP "Intraday data store investigation").
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from alpha_engine_lib.preflight import BasePreflight

log = logging.getLogger(__name__)


class ResearchPreflight(BasePreflight):
    """Preflight checks for the two research Lambdas."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("weekly", "alerts"):
            raise ValueError(f"ResearchPreflight: unknown mode {mode!r}")
        self.mode = mode

    def _check_arcticdb_universe(self) -> None:
        """Assert ArcticDB is reachable and SPY has fresh data.

        SPY is written by alpha-engine-data's weekly + daily collectors to the
        ``macro`` library (benchmarks/sector ETFs/macro series live there; the
        ``universe`` library holds the ~910 S&P 500+400 constituents). Its
        last-row date is the cleanest proxy for "DataPhase1 has run recently."
        Tolerates short staleness (up to 7 calendar days) because the Saturday
        research pipeline nominally runs after Friday's close — tighter
        freshness is enforced by the predictor's daily inference, not the
        weekly research batch.
        """
        import arcticdb as adb

        region = os.environ.get("AWS_REGION", "us-east-1")
        uri = f"s3s://s3.{region}.amazonaws.com:{self.bucket}?path_prefix=arcticdb&aws_auth=true"
        try:
            arctic = adb.Arctic(uri)
            macro = arctic.get_library("macro")
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB unreachable at {uri}: {exc}"
            ) from exc

        try:
            df = macro.read("SPY", columns=["Close"]).data
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB macro.SPY unreadable: {exc} — DataPhase1 did "
                f"not run or the macro library is broken."
            ) from exc

        if df is None or df.empty:
            raise RuntimeError(
                "ArcticDB macro.SPY has no rows — DataPhase1 has never written."
            )

        last_date = pd.Timestamp(df.index.max()).normalize()
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        stale_days = (today - last_date).days
        if stale_days > 7:
            raise RuntimeError(
                f"ArcticDB macro.SPY last_date={last_date.date()} is "
                f"{stale_days}d stale (>7d threshold) — DataPhase1 has not "
                f"refreshed recently."
            )
        log.info(
            "preflight: ArcticDB macro.SPY last_date=%s (%dd old)",
            last_date.date(), stale_days,
        )

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        if self.mode == "weekly":
            # Without the Anthropic key the graph fails mid-invocation
            # with a less-actionable error; checking here surfaces the
            # misconfiguration before any S3 read or LLM call.
            self.check_env_vars("ANTHROPIC_API_KEY")
        self.check_s3_bucket()
        if self.mode == "weekly":
            self._check_arcticdb_universe()
