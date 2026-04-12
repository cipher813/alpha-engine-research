"""Monkey-patch LangSmith's serializer to handle pandas DataFrames safely.

## Problem

The research graph state holds `price_data: dict[str, pd.DataFrame]` and
LangSmith's tracer serializes node inputs/outputs to JSON on every chain
start/end. LangSmith's internal `_serialize_json` helper (in
`langsmith._internal._serde`) tries a list of serialization methods in
order:

    _serialization_methods = [
        ("model_dump", {...}),  # Pydantic V2
        ("dict", {}),           # Pydantic V1
        ("to_dict", {}),        # dataclasses-json
    ]

`pd.DataFrame` has a `to_dict()` method that, with default
`orient="dict"`, returns `{column_name: {row_index: value}}` where
`row_index` is `pd.Timestamp` from the DatetimeIndex. LangSmith then
feeds that dict to `orjson.dumps(..., option=OPT_NON_STR_KEYS)`.

`pd.Timestamp` subclasses `datetime.datetime` in Python space, but
orjson's C-level type check is strict (`PyDateTime_DateTimeType` exact
match). It fails to recognize `pd.Timestamp` as a datetime and raises
TypeError. LangSmith's fallback path (`_serde.dumps_json:149-163`) then
calls stdlib `json.dumps`, which rejects all non-primitive dict keys
with:

    TypeError: keys must be str, int, float, bool or None, not Timestamp

Every agent callback crashes, CloudWatch floods with thousands of
warnings, and LangSmith receives zero valid trace data — which in turn
makes `evals/trajectory.py` report 11 false "missing node" failures.

## Fix

`langsmith._internal._serde._serialize_json` is a module-level function
that `dumps_json` looks up by name at call time (LOAD_GLOBAL, not a
captured reference). That means monkey-patching the module attribute
takes effect immediately for all subsequent serialization calls.

This module replaces `_serialize_json` with a wrapper that detects
`pd.DataFrame` and `pd.Series` BEFORE the `to_dict` fallback fires, and
returns a lightweight string summary instead. Everything else is
delegated to the original implementation unchanged.

Call `install()` once at Lambda startup, before the first graph
invocation.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_INSTALLED = False


def install() -> bool:
    """Install the LangSmith pandas serializer patch. Idempotent.

    Returns True if the patch was applied (or was already applied),
    False if langsmith/pandas aren't importable and the patch was
    skipped — e.g., in a stripped test environment.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    try:
        import pandas as pd
    except ImportError:
        log.debug("pandas not available — LangSmith patch skipped")
        return False

    try:
        from langsmith._internal import _serde
    except ImportError:
        log.debug("langsmith._internal._serde not available — patch skipped")
        return False

    original = _serde._serialize_json

    def _patched_serialize_json(obj):
        # DataFrames: return a summary string. Avoids the to_dict path
        # that produces {col: {Timestamp: value}} and trips orjson's
        # strict C-level type check for dict keys.
        if isinstance(obj, pd.DataFrame):
            try:
                cols = list(obj.columns)
                col_preview = ", ".join(map(str, cols[:5]))
                if len(cols) > 5:
                    col_preview += f", ... +{len(cols) - 5}"
                return f"<DataFrame shape={obj.shape} cols=[{col_preview}]>"
            except Exception:
                return f"<DataFrame id={id(obj)}>"
        # Series: same treatment.
        if isinstance(obj, pd.Series):
            return f"<Series len={len(obj)} dtype={obj.dtype}>"
        # Everything else: unchanged — delegate to langsmith's original
        # serializer for Pydantic models, dataclasses-json, etc.
        return original(obj)

    _serde._serialize_json = _patched_serialize_json
    _INSTALLED = True
    log.info("Installed LangSmith pandas DataFrame/Series serializer patch")
    return True
