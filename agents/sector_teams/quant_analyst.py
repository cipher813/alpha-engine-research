"""
Quant Analyst Agent — LangGraph create_react_agent with LangChain tools.

Each sector team's quant analyst autonomously screens its sector universe
using ReAct tool-calling. The agent decides its own screening strategy —
different sectors naturally use different tools and thresholds.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from config import PER_STOCK_MODEL, ANTHROPIC_API_KEY, QUANT_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.quant_tools import create_quant_tools
from agents.sector_teams.team_config import TEAM_SCREENING_PARAMS, QUANT_TOP_N, MAX_TICKERS_IN_PROMPT

log = logging.getLogger(__name__)


def run_quant_analyst(
    team_id: str,
    sector_tickers: list[str],
    market_regime: str,
    price_data: dict,
    technical_scores: dict,
    run_date: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the quant analyst ReAct agent for a sector team.

    Returns:
        {
            "team_id": str,
            "ranked_picks": list[dict],
            "tool_calls": list[dict],
            "iterations": int,
        }
    """
    team_params = TEAM_SCREENING_PARAMS.get(team_id, {})

    # Create LLM
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=4096,
    )

    # Create tools with shared context
    tools = create_quant_tools({
        "price_data": price_data,
        "technical_scores": technical_scores,
    })

    # Build system prompt
    system_prompt = _build_system_prompt(team_id, team_params, market_regime, len(sector_tickers))

    # Create ReAct agent via LangGraph
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    # Build input message
    ticker_list = ", ".join(sector_tickers[:MAX_TICKERS_IN_PROMPT])
    user_message = load_prompt("quant_analyst_user").format(
        run_date=run_date,
        market_regime=market_regime,
        universe_size=len(sector_tickers),
        ticker_list=ticker_list,
        quant_top_n=QUANT_TOP_N,
    )

    log.info("[quant:%s] starting ReAct agent with %d tickers", team_id, len(sector_tickers))

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": QUANT_MAX_ITERATIONS * 2},
        )

        # Extract the final response and tool calls from message history
        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)
        picks = _parse_picks_from_response(final_text)

        log.info("[quant:%s] completed — %d picks, %d tool calls",
                 team_id, len(picks), len(tool_calls))

        return {
            "team_id": team_id,
            "ranked_picks": picks,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
        }

    except Exception as e:
        log.error("[quant:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "ranked_picks": [],
            "tool_calls": [],
            "iterations": 0,
        }


def _build_system_prompt(
    team_id: str,
    team_params: dict,
    market_regime: str,
    universe_size: int,
) -> str:
    focus_metrics = team_params.get("focus_metrics", [])
    focus_str = ", ".join(focus_metrics) if focus_metrics else "standard technical and fundamental metrics"

    return load_prompt("quant_analyst_system").format(
        team_title=team_id.title(),
        universe_size=universe_size,
        quant_top_n=QUANT_TOP_N,
        focus_str=focus_str,
        market_regime=market_regime,
    )


from agents.langchain_utils import extract_tool_calls as _extract_tool_calls
from agents.langchain_utils import get_final_text as _get_final_text
from agents.json_utils import extract_json_array


def _parse_picks_from_response(text: str) -> list[dict]:
    """Parse ranked picks from the agent's final response."""
    result = extract_json_array(text)
    if result:
        return result[:QUANT_TOP_N]
    return []
