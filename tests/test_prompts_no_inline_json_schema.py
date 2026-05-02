"""Locks audit finding F1: no production prompt may carry an inline
``Respond with ONLY a JSON object`` schema example.

Per the prompt audit at ``alpha-engine-docs/private/alpha-engine-research-
prompt-audit-260430.md`` § 2 / F1: every LLM call site uses
``with_structured_output(<PydanticModel>)`` or ``response_format=`` since
the typed-state hard-fail flip arc (PRs #62-#65). The inline JSON schema
example in each prompt body became redundant — and a drift surface
(PR #59/#60 caught a literal-vs-int drift between the prompt example and
the Pydantic schema).

PR B (2026-05-02) stripped these examples from all 10 prompts. This test
prevents resurrection: any future PR that re-introduces a literal JSON
schema in a prompt body will fail here, forcing the contributor to
re-affirm + extend the Pydantic schema instead.

Search paths mirror ``agents.prompt_loader._resolve_prompt_path``: sibling
clone, then ``$GITHUB_WORKSPACE`` (CI), then the Lambda-staged
``<repo>/config/prompts/`` directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agents.prompt_loader import load_prompt

# The 10 production prompts shipped with the research repo. Mirrors the
# inventory in the audit doc § 1.
_PRODUCTION_PROMPTS = (
    "macro_agent",
    "macro_agent_critic",
    "quant_analyst_system",
    "quant_analyst_user",
    "qual_analyst_system",
    "qual_analyst_user",
    "peer_review_quant_addition",
    "peer_review_joint_finalization",
    "sector_team_thesis_update",
    "ic_cio_evaluation",
)

# Patterns that indicate the prompt is re-stating its output schema in prose
# rather than relying on the SDK's structured-output enforcement. Each is
# case-insensitive. Hits trigger a hard-fail with the exact line for triage.
_INLINE_JSON_PATTERNS = (
    r"respond with only a json",
    r"output json only",
    r"end with a json block",
    r"respond with a json object containing your assessments",
    r"respond with your final ranked list as a json array",
    r"output the full refreshed report followed by the json block",
)


def _config_prompts_dir() -> Path | None:
    """Resolve the production prompts directory if available in this env.

    Returns ``None`` when neither the sibling-clone nor the
    ``$GITHUB_WORKSPACE`` checkout nor the Lambda-staged directory has the
    prompts (e.g. a contributor running pytest without staging the config
    repo). Test then ``pytest.skip``s rather than false-failing.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.home() / "alpha-engine-config" / "research" / "prompts",
        repo_root.parent / "alpha-engine-config" / "research" / "prompts",
    ]
    ws = os.environ.get("GITHUB_WORKSPACE")
    if ws:
        candidates.append(
            Path(ws) / "alpha-engine-config" / "research" / "prompts"
        )
    candidates.append(repo_root / "config" / "prompts")
    for c in candidates:
        if c.exists() and any(c.glob("*.txt")):
            return c
    return None


@pytest.fixture(scope="module")
def prompts_dir() -> Path:
    p = _config_prompts_dir()
    if p is None:
        pytest.skip(
            "alpha-engine-config prompts not staged (no sibling clone, "
            "GITHUB_WORKSPACE, or Lambda staging). Skipping production-"
            "prompt content lock."
        )
    return p


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_no_inline_json_schema(name: str, prompts_dir: Path) -> None:
    """Each production prompt must NOT contain an inline JSON schema example."""
    text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    for pattern in _INLINE_JSON_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None, (
            f"Prompt '{name}.txt' contains inline JSON-schema instruction "
            f"matching /{pattern}/ at offset {match.start()} — re-introduces "
            f"audit finding F1 (drift surface vs Pydantic schema). Remove "
            f"the prose example; rely on with_structured_output enforcement."
        )


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_no_template_json_block(name: str, prompts_dir: Path) -> None:
    """No prompt should carry an example with multiple ``{{`` brace pairs.

    The previous schema examples used double-brace-escaped JSON for
    ``str.format()``. A handful of legitimate ``{{`` may appear in valid
    rendering contexts (extremely rare; we set a generous threshold). Three
    or more separate ``{{`` openings in one prompt strongly suggests an
    inline JSON example slipped back in.
    """
    text = (prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    open_braces = text.count("{{")
    assert open_braces < 3, (
        f"Prompt '{name}.txt' has {open_braces} ``{{{{`` escape sequences — "
        f"likely indicates an inline JSON-schema example. Audit finding F1 "
        f"removed these in PR B; re-introducing them resurrects the "
        f"two-sources-of-truth drift risk vs the Pydantic schema."
    )


@pytest.mark.parametrize("name", _PRODUCTION_PROMPTS)
def test_prompt_has_version_frontmatter(name: str, prompts_dir: Path) -> None:
    """Every production prompt must declare its version in frontmatter.

    Loader defaults to ``0.0.0`` when frontmatter is absent — that default
    is for legacy / test prompts only. Production prompts MUST stamp a
    real semver so LangSmith metadata propagation (PR D) and prompt-vs-
    prompt drift detection have a real version to attribute outputs to.
    """
    loaded = load_prompt(name)
    assert loaded.version != "0.0.0", (
        f"Prompt '{name}.txt' is missing ``# version: X.Y.Z`` frontmatter. "
        f"Add the frontmatter line; the loader's default-to-zero only "
        f"applies to legacy/test prompts."
    )
