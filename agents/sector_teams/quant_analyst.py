"""
Quant Analyst Agent — LangGraph create_react_agent with LangChain tools.

Each sector team's quant analyst autonomously screens its sector universe
using ReAct tool-calling. The agent decides its own screening strategy —
different sectors naturally use different tools and thresholds.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import PER_STOCK_MODEL, ANTHROPIC_API_KEY, QUANT_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.quant_tools import create_quant_tools
from agents.sector_teams.team_config import TEAM_SCREENING_PARAMS, QUANT_TOP_N, MAX_TICKERS_IN_PROMPT
from graph.llm_cost_tracker import get_cost_telemetry_callback

log = logging.getLogger(__name__)

# LangGraph state-transition budget. Each ReAct round = 1 LLM message + 1 tool
# response = 2 transitions, so QUANT_MAX_ITERATIONS rounds need 2× that. The
# extra +2 accounts for the post-loop ``response_format`` extraction call
# that ``create_react_agent(..., response_format=QuantAnalystOutput)`` adds
# (PR 2.3 / commit 84a1ce3, 2026-04-30) — that call sits inside the same
# subgraph and counts against ``recursion_limit``. Without it, an agent that
# legitimately uses all 8 rounds blows the budget on the structured-extraction
# pass. 2026-05-02 SF Research step halted on exactly this for 5 of 6 sector
# teams.
_QUANT_RECURSION_LIMIT = QUANT_MAX_ITERATIONS * 2 + 2


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

    # Create LLM. Cost-telemetry callback aggregates token usage across
    # the ReAct loop's multiple Anthropic calls into the active
    # ``track_llm_cost`` frame.
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=4096,
        callbacks=[get_cost_telemetry_callback()],
    )

    # Create tools with shared context
    tools = create_quant_tools({
        "price_data": price_data,
        "technical_scores": technical_scores,
    })

    # Build system prompt
    system_prompt = _build_system_prompt(team_id, team_params, market_regime, len(sector_tickers))

    # Create ReAct agent via LangGraph. PR 2.3 Step D adds response_format
    # so the ReAct loop ends with one extra Anthropic call that returns the
    # parsed Pydantic model directly (available as result['structured_response']).
    # This retires the _parse_picks_from_response regex parser.
    from graph.state_schemas import QuantAnalystOutput
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        response_format=QuantAnalystOutput,
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
        # Token usage from this ReAct loop's multiple Anthropic calls
        # accumulates into the active ``track_llm_cost`` frame opened
        # by the outer ``sector_team_node`` in research_graph.py.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": _QUANT_RECURSION_LIMIT},
        )

        # Extract the final response and tool calls from message history.
        # picks now come from response_format's structured_response Pydantic
        # model (PR 2.3 Step D); final_text is kept only for the diagnostic
        # logging path below when picks are empty.
        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)
        structured = result.get("structured_response")
        if structured is None:
            # response_format failed to populate — treat as agent failure
            # so the team's `error` field surfaces to score_aggregator.
            raise RuntimeError(
                "create_react_agent did not populate structured_response — "
                "the response_format extraction call failed (no QuantAnalystOutput)"
            )
        # Convert QuantPick Pydantic models to dicts for downstream
        # consumers (peer_review, score_aggregator) that use dict-access.
        picks = [p.model_dump() for p in structured.ranked_picks]

        log.info("[quant:%s] completed — %d picks, %d tool calls",
                 team_id, len(picks), len(tool_calls))

        # Diagnostic logging for the "no valid picks" case. 2-3 sector
        # teams have been returning zero picks per weekly run since at
        # least 2026-04-04 and we don't know whether it's (a) the LLM
        # producing no JSON, (b) _parse_picks_from_response failing to
        # extract it, (c) the ReAct agent hitting the recursion limit
        # before producing final text, or (d) all tool calls failing
        # and the LLM having no data to work with. Log enough context
        # to tell these apart on the next run.
        if not picks:
            last_tool = tool_calls[-1].get("tool") if tool_calls else "<none>"
            recursion_limit_hit = len(tool_calls) >= QUANT_MAX_ITERATIONS * 2 - 1
            text_tail = (final_text[-500:] if final_text else "<empty>").replace("\n", " ")
            log.warning(
                "[quant:%s] produced 0 picks — tool_calls=%d "
                "(recursion_limit_hit=%s) last_tool=%s "
                "final_text_tail=%r",
                team_id,
                len(tool_calls),
                recursion_limit_hit,
                last_tool,
                text_tail,
            )

        return {
            "team_id": team_id,
            "ranked_picks": picks,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
            "error": None,
            "partial": False,
        }

    except GraphRecursionError as e:
        # Budget exhausted before the agent reached a stop condition.
        # Treat as a degraded-but-non-fatal outcome: this team contributes
        # zero picks but doesn't crash the SF — score_aggregator will see
        # ``partial=True`` and accept the empty contribution with a WARN.
        # The +2 budget bump should already prevent the ``response_format``
        # extraction call from blowing the budget on its own; if we still
        # hit this, the agent legitimately needs more than 8 ReAct rounds
        # for this sector + run, which is a tunable observation, not a
        # crash-the-pipeline emergency.
        log.warning(
            "[quant:%s] recursion budget (%d transitions) exhausted before "
            "stop condition — accepting partial result (0 picks). "
            "score_aggregator will proceed with this team excluded.",
            team_id, _QUANT_RECURSION_LIMIT,
        )
        return {
            "team_id": team_id,
            "ranked_picks": [],
            "tool_calls": [],
            "iterations": _QUANT_RECURSION_LIMIT,
            "error": None,
            "partial": True,
            "partial_reason": "recursion_limit_exhausted",
        }

    except Exception as e:
        # Record the error so downstream (score_aggregator) can hard-fail
        # loudly instead of treating an exception as equivalent to the LLM
        # legitimately producing zero picks. Recursion budget exhaustion
        # is handled separately above as a partial outcome, not an error.
        log.error("[quant:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "ranked_picks": [],
            "tool_calls": [],
            "iterations": 0,
            "error": f"{type(e).__name__}: {e}",
            "partial": False,
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
