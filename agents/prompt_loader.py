"""
Prompt loader — reads agent prompt templates from config/prompts/ directory.

Prompts are stored as plain text files (gitignored) to keep proprietary
scoring logic out of the public repository. Each agent calls load_prompt()
at module import time; the template is cached for the process lifetime.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"
_cache: dict[str, str] = {}


def load_prompt(agent_name: str) -> str:
    """
    Load a prompt template from config/prompts/{agent_name}.txt.

    Args:
        agent_name: filename without extension (e.g. "macro_agent")

    Returns:
        The prompt template string with {placeholders} for .format().

    Raises:
        FileNotFoundError: if the prompt file is missing.
    """
    if agent_name in _cache:
        return _cache[agent_name]

    path = _PROMPTS_DIR / f"{agent_name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Copy config/prompts.example/{agent_name}.txt to "
            f"config/prompts/{agent_name}.txt and customise."
        )

    template = path.read_text(encoding="utf-8")
    _cache[agent_name] = template
    logger.debug("Loaded prompt template: %s (%d chars)", agent_name, len(template))
    return template
