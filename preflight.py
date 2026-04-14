"""
Research Lambda preflight: connectivity checks run at the top of each
handler invocation before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into two mode-specific sequences matching the
research Lambdas.

Modes:

- ``"weekly"`` — ``lambda/handler.py``, the weekly research pipeline.
  AWS_REGION + ANTHROPIC_API_KEY + S3 bucket reachable. No ArcticDB
  freshness check here: research reads ArcticDB for price enrichment
  via ``data/fetchers/price_fetcher.py``, but it falls back to
  yfinance on miss, so an ArcticDB outage is a degraded-mode scenario
  rather than a hard failure.
- ``"alerts"`` — ``lambda/alerts_handler.py``, the 30-minute intraday
  price alert Lambda. AWS_REGION + S3 bucket only; alerts do not call
  Anthropic.
"""

from __future__ import annotations

from alpha_engine_lib.preflight import BasePreflight


class ResearchPreflight(BasePreflight):
    """Preflight checks for the two research Lambdas."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("weekly", "alerts"):
            raise ValueError(f"ResearchPreflight: unknown mode {mode!r}")
        self.mode = mode

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        if self.mode == "weekly":
            # Without the Anthropic key the graph fails mid-invocation
            # with a less-actionable error; checking here surfaces the
            # misconfiguration before any S3 read or LLM call.
            self.check_env_vars("ANTHROPIC_API_KEY")
        self.check_s3_bucket()
