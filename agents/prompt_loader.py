"""
Prompt loader — reads agent prompt templates from config/prompts/ directory.

Prompts are stored as plain text files (gitignored) to keep proprietary
scoring logic out of the public repository. Each agent calls load_prompt()
at module import time; the template is cached for the process lifetime.

Falls back to ``config/prompts.example/{name}.txt`` when the proprietary
prompt is missing, matching the ``scoring.yaml`` → ``scoring.sample.yaml``
fallback pattern. This keeps CI (which has no proprietary prompts) working
against the sample prompts without committing real templates.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_PROMPTS_DIR = _CONFIG_DIR / "prompts"
_PROMPTS_EXAMPLE_DIR = _CONFIG_DIR / "prompts.example"
_cache: dict[str, str] = {}


def load_prompt(agent_name: str) -> str:
    """
    Load a prompt template from ``config/prompts/{agent_name}.txt``.

    Falls back to ``config/prompts.example/{agent_name}.txt`` if the real
    prompt is missing — this makes CI and fresh clones work without needing
    the proprietary prompts, while production still picks up the real
    templates when they're present.

    Args:
        agent_name: filename without extension (e.g. "macro_agent")

    Returns:
        The prompt template string with {placeholders} for .format().

    Raises:
        FileNotFoundError: if neither the real nor the example prompt exists.
    """
    if agent_name in _cache:
        return _cache[agent_name]

    path = _PROMPTS_DIR / f"{agent_name}.txt"
    if path.exists():
        source = "prompts"
    else:
        example_path = _PROMPTS_EXAMPLE_DIR / f"{agent_name}.txt"
        if example_path.exists():
            logger.warning(
                "config/prompts/%s.txt not found — falling back to "
                "config/prompts.example/%s.txt",
                agent_name, agent_name,
            )
            path = example_path
            source = "prompts.example"
        else:
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                f"Copy config/prompts.example/{agent_name}.txt to "
                f"config/prompts/{agent_name}.txt and customise. "
                f"(Fallback example also missing at {example_path}.)"
            )

    template = path.read_text(encoding="utf-8")
    _cache[agent_name] = template
    logger.debug(
        "Loaded prompt template: %s from %s (%d chars)",
        agent_name, source, len(template),
    )
    return template
