"""
Trajectory validation for the research graph.

After each pipeline run, validates that all expected LangGraph nodes executed
in the correct order. Uses LangSmith traces collected during execution.

This module never blocks the pipeline — all errors are caught and logged.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# ── Reference trajectory from graph/research_graph.py ────────────────────────

REQUIRED_NODES = [
    "fetch_data",
    "sector_team_node",
    "macro_economist_node",
    "exit_evaluator_node",
    "merge_results",
    "score_aggregator",
    "cio_node",
    "population_entry_handler",
    "consolidator_node",
    "archive_writer",
    "email_sender_node",
]

# (before, after) — "before" must appear earlier in the trace than "after"
ORDERING_CONSTRAINTS = [
    ("fetch_data", "sector_team_node"),
    ("fetch_data", "macro_economist_node"),
    ("fetch_data", "exit_evaluator_node"),
    ("sector_team_node", "merge_results"),
    ("macro_economist_node", "merge_results"),
    ("exit_evaluator_node", "merge_results"),
    ("merge_results", "score_aggregator"),
    ("score_aggregator", "cio_node"),
    ("cio_node", "population_entry_handler"),
    ("population_entry_handler", "consolidator_node"),
    ("consolidator_node", "archive_writer"),
    ("archive_writer", "email_sender_node"),
]

EXPECTED_SECTOR_TEAM_COUNT = 6


def validate_trajectory(
    project_name: str = "alpha-research",
    max_wait_seconds: int = 15,
) -> dict | None:
    """
    Validate the most recent LangGraph run's trajectory against the reference.

    Queries LangSmith for the latest completed run in the project, extracts
    the child span node names, and checks:
      1. All required nodes are present
      2. sector_team_node appears exactly 6 times (one per Send)
      3. Ordering constraints are satisfied

    Args:
        project_name: LangSmith project name (matches LANGCHAIN_PROJECT env var)
        max_wait_seconds: Max time to wait for traces to propagate to LangSmith

    Returns:
        {"passed": bool, "failures": list[str], "node_counts": dict, "duration_ms": int}
        or None if tracing is not enabled or validation could not run.
    """
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        logger.info("Trajectory validation skipped — LANGCHAIN_TRACING_V2 not set")
        return None

    try:
        from langsmith import Client
    except ImportError:
        logger.warning("Trajectory validation skipped — langsmith not installed")
        return None

    client = Client()
    failures: list[str] = []

    # Wait for the most recent run to appear in LangSmith
    run = None
    for attempt in range(max_wait_seconds // 3 + 1):
        if attempt > 0:
            time.sleep(3)
        try:
            runs = list(client.list_runs(
                project_name=project_name,
                is_root=True,
                limit=1,
            ))
            if runs:
                run = runs[0]
                break
        except Exception as e:
            logger.warning("LangSmith query attempt %d failed: %s", attempt + 1, e)

    if run is None:
        logger.warning("No runs found in LangSmith project '%s'", project_name)
        return {"passed": False, "failures": ["no_run_found"], "node_counts": {}, "duration_ms": 0}

    # Fetch child spans (graph node executions)
    try:
        child_runs = list(client.list_runs(
            project_name=project_name,
            trace_id=run.trace_id,
            is_root=False,
        ))
    except Exception as e:
        logger.warning("Failed to fetch child runs: %s", e)
        return {"passed": False, "failures": [f"fetch_children_failed: {e}"], "node_counts": {}, "duration_ms": 0}

    # Extract node names and their earliest start times
    node_names: list[str] = []
    node_first_start: dict[str, float] = {}
    for child in child_runs:
        name = child.name
        if name and child.start_time:
            node_names.append(name)
            ts = child.start_time.timestamp()
            if name not in node_first_start or ts < node_first_start[name]:
                node_first_start[name] = ts

    # Count occurrences
    node_counts: dict[str, int] = {}
    for name in node_names:
        node_counts[name] = node_counts.get(name, 0) + 1

    # Check 1: All required nodes present
    for required in REQUIRED_NODES:
        if required not in node_counts:
            failures.append(f"missing_node: {required}")

    # Check 2: sector_team_node count
    team_count = node_counts.get("sector_team_node", 0)
    if team_count != EXPECTED_SECTOR_TEAM_COUNT:
        failures.append(
            f"sector_team_count: expected {EXPECTED_SECTOR_TEAM_COUNT}, got {team_count}"
        )

    # Check 3: Ordering constraints
    for before, after in ORDERING_CONSTRAINTS:
        t_before = node_first_start.get(before)
        t_after = node_first_start.get(after)
        if t_before is not None and t_after is not None:
            if t_before > t_after:
                failures.append(f"ordering_violation: {before} started after {after}")

    # Compute duration
    duration_ms = 0
    if run.end_time and run.start_time:
        duration_ms = int((run.end_time - run.start_time).total_seconds() * 1000)

    passed = len(failures) == 0

    if passed:
        logger.info(
            "Trajectory validation PASSED — %d nodes, %d sector teams, %dms",
            len(node_names), team_count, duration_ms,
        )
    else:
        logger.error(
            "Trajectory validation FAILED — %d failures: %s",
            len(failures), failures,
        )

    return {
        "passed": passed,
        "failures": failures,
        "node_counts": node_counts,
        "duration_ms": duration_ms,
    }
