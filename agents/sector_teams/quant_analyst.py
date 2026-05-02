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
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import PER_STOCK_MODEL, ANTHROPIC_API_KEY, QUANT_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.quant_tools import create_quant_tools
from agents.sector_teams.team_config import TEAM_SCREENING_PARAMS, QUANT_TOP_N, MAX_TICKERS_IN_PROMPT
from graph.llm_cost_tracker import get_cost_telemetry_callback
from strict_mode import is_strict_validation_enabled

log = logging.getLogger(__name__)

# LangGraph state-transition budget. Each ReAct round = 1 LLM message + 1 tool
# response = 2 transitions, so QUANT_MAX_ITERATIONS rounds need 2× that.
# The +2 buffer is RETAINED defensively (was added 2026-05-02 to cover
# response_format's extra extraction call inside the subgraph). After the
# 2026-05-02 refactor that decouples the structured-output extraction from
# the ReAct loop, the +2 is no longer load-bearing — but the cost is one
# transition slot of headroom and removing it offers no benefit. Keep as
# defensive margin.
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

    # Create ReAct agent via LangGraph. The ReAct loop runs until the model
    # produces a final-text answer (no more tool calls). The structured-
    # output extraction is decoupled and runs as a separate
    # ``with_structured_output`` call after the loop — see "structured
    # extraction" block below. Mirrors the convention in macro_agent.py /
    # peer_review.py / ic_cio.py.
    #
    # 2026-05-02 refactor rationale: the prior ``response_format=
    # QuantAnalystOutput`` mechanism inside ``create_react_agent`` adds a
    # post-loop extraction call to the LangGraph subgraph. That call is
    # not constrained — Haiku occasionally returns markdown-fenced JSON
    # text instead of using the structured-output tool, which crashes the
    # SF with a Pydantic ``ValidationError`` (input_value is the entire
    # string-with-fences assigned to ``ranked_picks``). Decoupling lets us
    # drive ``with_structured_output`` directly with ``include_raw=True``
    # and the strict-mode parsing-error contract, which is the established
    # pattern across every other LLM-output site in this codebase.
    from graph.state_schemas import QuantAnalystOutput
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    # Build input message
    ticker_list = ", ".join(sector_tickers[:MAX_TICKERS_IN_PROMPT])
    user_prompt = load_prompt("quant_analyst_user")
    user_message = user_prompt.format(
        run_date=run_date,
        market_regime=market_regime,
        universe_size=len(sector_tickers),
        ticker_list=ticker_list,
        quant_top_n=QUANT_TOP_N,
    )
    # System prompt's metadata anchors LangSmith trace attribution; the
    # user prompt's version + hash piggyback so a future drift in either
    # half of the prompt-pair is independently grep-able.
    system_prompt_loaded = load_prompt("quant_analyst_system")
    _ls_metadata = {
        **system_prompt_loaded.langsmith_metadata(),
        "user_prompt_version": user_prompt.version,
        "user_prompt_hash": user_prompt.hash[:12],
    }

    log.info("[quant:%s] starting ReAct agent with %d tickers", team_id, len(sector_tickers))

    try:
        # Token usage from this ReAct loop's multiple Anthropic calls
        # accumulates into the active ``track_llm_cost`` frame opened
        # by the outer ``sector_team_node`` in research_graph.py.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={
                "recursion_limit": _QUANT_RECURSION_LIMIT,
                "metadata": _ls_metadata,
            },
        )

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)

        # ── Decoupled structured-output extraction ──────────────────────
        # Drives ``with_structured_output(include_raw=True)`` directly so
        # the strict-mode parsing-error contract is honored — no markdown-
        # fence-text confusion possible because the extraction call is
        # constrained at the API boundary (Anthropic tool-use). Mirrors
        # macro_agent.py:184 / peer_review.py / ic_cio.py.
        if not final_text or not final_text.strip():
            raise RuntimeError(
                f"[quant:{team_id}] ReAct loop produced empty final_text — "
                f"nothing to extract structured picks from. tool_calls={len(tool_calls)}"
            )
        structured_llm = llm.with_structured_output(
            QuantAnalystOutput, include_raw=True,
        )
        extract_msg = HumanMessage(content=(
            "Extract the final ranked picks from this analyst's answer "
            "into the structured schema. Use only what's in the text — "
            "do not invent picks. If the analyst produced no picks, "
            "return an empty list.\n\n"
            f"--- ANALYST ANSWER ---\n{final_text}"
        ))
        extract_resp = structured_llm.invoke(
            [extract_msg],
            config={"metadata": _ls_metadata},
        )
        parsed: QuantAnalystOutput | None = extract_resp.get("parsed")
        parsing_error = extract_resp.get("parsing_error")
        if parsing_error is not None:
            msg = (
                f"[quant:{team_id}] structured-output parse failed: "
                f"{type(parsing_error).__name__}: {parsing_error}"
            )
            if is_strict_validation_enabled():
                raise RuntimeError(msg)
            log.warning("%s — falling back to empty picks (lax mode)", msg)
            parsed = QuantAnalystOutput()
        assert parsed is not None
        # Convert QuantPick Pydantic models to dicts for downstream
        # consumers (peer_review, score_aggregator) that use dict-access.
        picks = [p.model_dump() for p in parsed.ranked_picks]

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
