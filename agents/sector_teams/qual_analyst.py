"""
Qualitative Analyst Agent — LangGraph create_react_agent with LangChain tools.

Reviews the quant analyst's top 5 picks with qualitative data:
news, analyst reports, insider activity, SEC filings, prior theses.
Produces a single holistic qual_score (0-100) per stock.
May identify 0-1 additional candidates that quant missed.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import PER_STOCK_MODEL, ANTHROPIC_API_KEY, QUAL_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.qual_tools import create_qual_tools
from graph.llm_cost_tracker import get_cost_telemetry_callback

log = logging.getLogger(__name__)

# +2 over (max_iter × 2) accounts for the post-loop ``response_format``
# extraction call inside ``create_react_agent``. See quant_analyst.py for
# the full rationale (PR 2.3 / commit 84a1ce3, 2026-04-30).
_QUAL_RECURSION_LIMIT = QUAL_MAX_ITERATIONS * 2 + 2


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
        callbacks=[get_cost_telemetry_callback()],
    )

    tools = create_qual_tools({
        "prior_theses": prior_theses,
        "price_data": price_data or {},
        "episodic_memories": episodic_memories or {},
        "semantic_memories": semantic_memories or {},
    })

    system_prompt = _build_system_prompt(team_id, market_regime, len(quant_top5))

    # PR 2.3 Step D: response_format ends the ReAct loop with a typed
    # Pydantic model in result['structured_response'], retiring the
    # _parse_assessments regex parser.
    from graph.state_schemas import QualAnalystOutput
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        response_format=QualAnalystOutput,
    )

    picks_text = "\n".join(
        f"  {i+1}. {p['ticker']} (quant_score={p.get('quant_score', '?')}): "
        f"{p.get('rationale', 'no rationale')}"
        for i, p in enumerate(quant_top5)
    )

    user_message = load_prompt("qual_analyst_user").format(
        run_date=run_date,
        market_regime=market_regime,
        picks_text=picks_text,
    )

    log.info("[qual:%s] starting ReAct agent with %d picks", team_id, len(quant_top5))

    try:
        # Token usage from this ReAct loop's multiple Anthropic calls
        # accumulates into the active ``track_llm_cost`` frame opened
        # by the outer ``sector_team_node`` in research_graph.py.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": _QUAL_RECURSION_LIMIT},
        )

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        structured = result.get("structured_response")
        if structured is None:
            raise RuntimeError(
                "create_react_agent did not populate structured_response — "
                "the response_format extraction call failed (no QualAnalystOutput)"
            )
        # Convert QualAssessment Pydantic models to dicts for downstream
        # peer_review consumption (which uses dict-access patterns).
        assessments = [a.model_dump() for a in structured.assessments]
        additional_candidate = (
            structured.additional_candidate.model_dump()
            if structured.additional_candidate is not None
            else None
        )

        log.info("[qual:%s] completed — %d assessments, %d tool calls",
                 team_id, len(assessments), len(tool_calls))

        return {
            "team_id": team_id,
            "assessments": assessments,
            "additional_candidate": additional_candidate,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
            "error": None,
            "partial": False,
        }

    except GraphRecursionError as e:
        # Budget exhausted before the agent reached a stop condition.
        # Mirrors quant_analyst's policy: degraded-but-non-fatal — this
        # team contributes zero assessments but doesn't crash the SF.
        log.warning(
            "[qual:%s] recursion budget (%d transitions) exhausted before "
            "stop condition — accepting partial result (0 assessments). "
            "score_aggregator will proceed with this team excluded.",
            team_id, _QUAL_RECURSION_LIMIT,
        )
        return {
            "team_id": team_id,
            "assessments": [],
            "additional_candidate": None,
            "tool_calls": [],
            "iterations": _QUAL_RECURSION_LIMIT,
            "error": None,
            "partial": True,
            "partial_reason": "recursion_limit_exhausted",
        }

    except Exception as e:
        # Record the error so downstream (score_aggregator) can hard-fail
        # instead of silently treating an exception the same as an LLM
        # legitimately producing zero assessments. Recursion budget
        # exhaustion is handled separately above as partial.
        log.error("[qual:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "assessments": [],
            "additional_candidate": None,
            "tool_calls": [],
            "iterations": 0,
            "error": f"{type(e).__name__}: {e}",
            "partial": False,
        }


def _build_system_prompt(team_id: str, market_regime: str, n_picks: int) -> str:
    return load_prompt("qual_analyst_system").format(
        team_title=team_id.title(),
        n_picks=n_picks,
        market_regime=market_regime,
    )


from agents.langchain_utils import extract_tool_calls as _extract_tool_calls
