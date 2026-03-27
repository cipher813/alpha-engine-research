"""
Shared utilities for extracting data from LangGraph message histories.

Used by ReAct agents (quant_analyst, qual_analyst) to parse tool calls
and final text from the LangGraph message format.
"""

from __future__ import annotations


def extract_tool_calls(messages: list) -> list[dict]:
    """Extract tool call records from LangGraph message history."""
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                calls.append({
                    "tool": tc.get("name", ""),
                    "input_summary": str(tc.get("args", {}))[:200],
                })
        elif hasattr(msg, "type") and msg.type == "tool":
            calls.append({
                "tool": getattr(msg, "name", "unknown"),
                "status": "executed",
            })
    return calls


def get_final_text(messages: list) -> str:
    """Get the last AI message text from a LangGraph message history."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                texts = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(texts)
    return ""
