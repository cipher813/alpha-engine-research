"""Regression tests for graph.langsmith_pandas_patch.

Pins the fix for the 2026-04-11 callback-flood bug where every agent
node crashed with `TypeError: keys must be str, int, float, bool or
None, not Timestamp` because langsmith._internal._serde called
pd.DataFrame.to_dict() on the state and the resulting Timestamp-keyed
dict broke both orjson (strict C-level datetime type check) and the
stdlib json fallback path.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_patch_is_idempotent():
    """install() can be called multiple times safely."""
    from graph.langsmith_pandas_patch import install

    assert install() in (True, False)
    assert install() in (True, False)


def test_serializer_returns_summary_for_dataframe():
    """After patching, _serialize_json returns a safe string for DataFrames."""
    from graph.langsmith_pandas_patch import install

    installed = install()
    if not installed:
        pytest.skip("langsmith or pandas not available")

    from langsmith._internal import _serde

    df = pd.DataFrame(
        {"Open": [1.0, 2.0], "Close": [1.5, 2.5]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )

    result = _serde._serialize_json(df)

    # Must be a plain string (not a dict with Timestamp keys)
    assert isinstance(result, str)
    # Must describe the dataframe shape
    assert "DataFrame" in result
    assert "2" in result  # 2 rows
    # Must be JSON-safe (stdlib json happy)
    import json
    json.dumps(result)  # should not raise


def test_serializer_returns_summary_for_series():
    """After patching, _serialize_json also handles Series safely."""
    from graph.langsmith_pandas_patch import install

    if not install():
        pytest.skip("langsmith or pandas not available")

    from langsmith._internal import _serde

    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]))
    result = _serde._serialize_json(s)

    assert isinstance(result, str)
    assert "Series" in result
    import json
    json.dumps(result)  # should not raise


def test_non_pandas_objects_pass_through():
    """The patch must not break serialization of non-pandas objects."""
    from graph.langsmith_pandas_patch import install

    if not install():
        pytest.skip("langsmith or pandas not available")

    from langsmith._internal import _serde

    # Plain dict with string keys — langsmith's original _serialize_json
    # doesn't know what to do with a raw dict (it expects objects with
    # serialization methods), so it falls to _simple_default which
    # returns str() of the object. The exact return doesn't matter here;
    # we just verify the patch doesn't crash on non-DataFrame input.
    class PlainObj:
        def __repr__(self):
            return "PlainObj()"

    result = _serde._serialize_json(PlainObj())
    # _simple_default returns str(obj) as fallback
    assert result == "PlainObj()"


def test_dumps_json_with_dataframe_in_state():
    """End-to-end: langsmith.dumps_json on a state dict containing DataFrames succeeds.

    This is the regression test for the actual 2026-04-11 bug — state
    containing pd.DataFrame with a DatetimeIndex must serialize cleanly
    through langsmith's dumps_json, not raise the Timestamp key error.
    """
    from graph.langsmith_pandas_patch import install

    if not install():
        pytest.skip("langsmith or pandas not available")

    from langsmith._internal._serde import dumps_json

    state = {
        "run_date": "2026-04-11",
        "price_data": {
            "AAPL": pd.DataFrame(
                {"Open": [190.0, 191.0], "Close": [192.0, 193.5]},
                index=pd.DatetimeIndex(["2026-04-10", "2026-04-11"]),
            ),
            "MSFT": pd.DataFrame(
                {"Open": [440.0], "Close": [445.0]},
                index=pd.DatetimeIndex(["2026-04-11"]),
            ),
        },
        "market_regime": "neutral",
    }

    # Must not raise TypeError about Timestamp keys
    result = dumps_json(state)
    assert isinstance(result, bytes)
    # The summary strings should appear in the serialized output
    assert b"DataFrame" in result
    assert b"neutral" in result
