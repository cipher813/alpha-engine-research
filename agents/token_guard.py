"""
Token limit enforcement for LLM agent calls (M1-13).

Estimates prompt token count and truncates from the middle if the prompt
would exceed the model's context window minus reserved output tokens.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 characters per token for English text."""
    return int(len(text) / 3.5)


def check_prompt_size(
    prompt: str,
    max_output_tokens: int,
    model_context_limit: int = 200_000,
    caller: str = "",
) -> str:
    """
    Check whether *prompt* fits within the model context window after
    reserving space for *max_output_tokens* of output.

    If the prompt is too large, truncate from the middle — keeping the
    first 40 % and last 40 % of the allowed character budget — and log a
    warning.

    Returns the (possibly truncated) prompt.
    """
    max_prompt_tokens = model_context_limit - max_output_tokens
    max_prompt_chars = int(max_prompt_tokens * 3.5)

    if len(prompt) <= max_prompt_chars:
        return prompt

    est_tokens = estimate_tokens(prompt)
    tag = f"[token_guard:{caller}]" if caller else "[token_guard]"
    _logger.warning(
        "%s Prompt too large: ~%d tokens (limit %d). Truncating from middle.",
        tag,
        est_tokens,
        max_prompt_tokens,
    )

    keep_chars = max_prompt_chars
    first_part = int(keep_chars * 0.40)
    last_part = int(keep_chars * 0.40)

    truncated = (
        prompt[:first_part]
        + "\n\n[...TRUNCATED — middle removed to fit context window...]\n\n"
        + prompt[-last_part:]
    )
    return truncated
