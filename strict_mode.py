"""Strict-mode validation env-var helper.

Lives at repo root (alongside ``preflight.py``, ``retry.py``,
``health_status.py``) so both ``graph/research_graph.py`` (state-shape
validators) and the ``agents/`` LLM-extraction sites can import it
without a circular import. ``graph`` depends on ``agents``, so a shared
helper at either layer would create a one-way violation; root level is
neutral.

Default behavior during PR 2 rollout is **False** (warn-mode preserved).
Step F of the PR-2 sequence flips this default to ``True``
(strict-by-default), at which point operators set
``STRICT_VALIDATION=false`` for the emergency override path. The
30-second Lambda-console env-var flip works either way because the
helper reads ``os.environ`` fresh on each call.
"""

from __future__ import annotations

import os


def is_strict_validation_enabled() -> bool:
    """Return ``True`` when typed-state validation should hard-fail
    on schema violations.

    Reads the ``STRICT_VALIDATION`` env var fresh on each call so a
    Lambda console flip takes effect on warm containers without
    redeploy. Truthy values: ``true``, ``1``, ``yes`` (case-insensitive).
    """
    return os.environ.get("STRICT_VALIDATION", "false").lower() in (
        "true", "1", "yes"
    )
