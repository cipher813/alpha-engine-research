"""
Qualitative Analyst Agent — LangGraph create_react_agent with LangChain tools.

Reviews the quant analyst's top 5 picks with qualitative data:
news, analyst reports, insider activity, SEC filings, prior theses.
Produces a single holistic qual_score (0-100) per stock.
May identify 0-1 additional candidates that quant missed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from config import PER_STOCK_MODEL, ANTHROPIC_API_KEY, QUAL_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.qual_tools import create_qual_tools

log = logging.getLogger(__name__)


def run_qual_analyst(
    team_id: str,
    quant_top5: list[dict],
    prior_theses: dict[str, dict],
    market_regime: str,
    run_date: str,
    api_key: Optional[str] = None,
    price_data: Optional[dict] = None,
    episodic_memories: dict[str, list] | None = None,
    semantic_memories: dict[str, list] | None = None,
) -> dict:
    """
    Run the qual analyst ReAct agent.

    Returns:
        {
            "team_id": str,
            "assessments": list[dict],  # qual_score + bull/bear per stock
            "additional_candidate": dict | None,
            "tool_calls": list[dict],
            "iterations": int,
        }
    """
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=4096,
    )

    tools = create_qual_tools({
        "prior_theses": prior_theses,
        "price_data": price_data or {},
        "episodic_memories": episodic_memories or {},
        "semantic_memories": semantic_memories or {},
    })

    system_prompt = _build_system_prompt(team_id, market_regime, len(quant_top5))

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    picks_text = "\n".join(
        f"  {i+1}. {p['ticker']} (quant_score={p.get('quant_score', '?')}): "
        f"{p.get('rationale', 'no rationale')}"
        for i, p in enumerate(quant_top5)
    )

    user_message = (
        f"Today is {run_date}. Market regime: {market_regime}.\n\n"
        f"The Quant Analyst's top 5 picks for your sector:\n{picks_text}\n\n"
        f"Review each stock using your qualitative tools. For each, produce:\n"
        f"- qual_score (0-100): your holistic qualitative conviction\n"
        f"- bull_case: key bullish argument (1-2 sentences)\n"
        f"- bear_case: key bearish risk (1-2 sentences)\n"
        f"- conviction: high/medium/low\n"
        f"- resources_used: which tools most influenced your assessment\n\n"
        f"You may add 1 additional candidate if you identify a strong story the quant missed.\n\n"
        f"Respond with your final assessment as a JSON object:\n"
        f'{{"assessments": [...], "additional_candidate": {{...}} or null}}'
    )

    log.info("[qual:%s] starting ReAct agent with %d picks", team_id, len(quant_top5))

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": QUAL_MAX_ITERATIONS * 2},
        )

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)
        parsed = _parse_assessments(final_text)

        log.info("[qual:%s] completed — %d assessments, %d tool calls",
                 team_id, len(parsed.get("assessments", [])), len(tool_calls))

        return {
            "team_id": team_id,
            "assessments": parsed.get("assessments", []),
            "additional_candidate": parsed.get("additional_candidate"),
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
        }

    except Exception as e:
        log.error("[qual:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "assessments": [],
            "additional_candidate": None,
            "tool_calls": [],
            "iterations": 0,
        }


def _build_system_prompt(team_id: str, market_regime: str, n_picks: int) -> str:
    return load_prompt("qual_analyst_system").format(
        team_title=team_id.title(),
        n_picks=n_picks,
        market_regime=market_regime,
    )


from agents.langchain_utils import extract_tool_calls as _extract_tool_calls
from agents.langchain_utils import get_final_text as _get_final_text
from agents.json_utils import extract_json_object, extract_json_array


def _parse_assessments(text: str) -> dict:
    """Parse assessments from the agent's final response."""
    # Try full JSON object with assessments key
    result = extract_json_object(text, hint_key='"assessments"')
    if result and "assessments" in result:
        return result

    # Try JSON array directly
    arr = extract_json_array(text)
    if arr:
        return {"assessments": arr, "additional_candidate": None}

    return {"assessments": [], "additional_candidate": None}
