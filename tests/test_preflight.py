"""
Tests for ResearchPreflight mode composition.

BasePreflight primitives are tested in alpha-engine-lib. These tests
only verify that each research mode calls the expected primitives.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight import ResearchPreflight


class TestResearchPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ResearchPreflight(bucket="b", mode="bogus")

    def test_weekly_mode_checks_anthropic_key(self):
        pf = ResearchPreflight(bucket="b", mode="weekly")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3:
            pf.run()
        # Two check_env_vars calls: AWS_REGION then ANTHROPIC_API_KEY.
        assert env.call_args_list[0].args == ("AWS_REGION",)
        assert env.call_args_list[1].args == ("ANTHROPIC_API_KEY",)
        s3.assert_called_once()

    def test_alerts_mode_skips_anthropic_key(self):
        pf = ResearchPreflight(bucket="b", mode="alerts")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3:
            pf.run()
        assert env.call_count == 1
        assert env.call_args_list[0].args == ("AWS_REGION",)
        s3.assert_called_once()
