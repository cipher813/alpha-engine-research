"""
Lambda entry point — main research pipeline.

Weekly (primary): triggered by EventBridge Monday 06:00 UTC (Sunday ~10-11pm PT).
EventBridge passes {"weekly_run": true} — bypasses the 5:45am PT time gate.

Weekday (disabled, available for rollback): EventBridge at 12:45+13:45 UTC
(5:45am PT after DST time gate). Checks for market holidays.

Pass {"force": true} to bypass all gates (manual testing).
"""

from __future__ import annotations

import datetime
import os
import time

import pytz

from exchange_calendars import get_calendar


def is_trading_day(date: datetime.date | None = None) -> bool:
    """Return True if date (default: today) is an NYSE trading day."""
    nyse = get_calendar("XNYS")
    d = date or datetime.date.today()
    return nyse.is_session(d)


def is_early_close(date: datetime.date | None = None) -> bool:
    """
    Return True if the NYSE has an early close today (partial session).
    Early closes: day before July 4th, Black Friday, Christmas Eve.
    These still run — the morning report executes normally.
    """
    nyse = get_calendar("XNYS")
    d = date or datetime.date.today()
    try:
        # exchange_calendars exposes early close dates
        session = nyse.schedule.loc[str(d)] if str(d) in nyse.schedule.index else None
        if session is not None:
            close_time = session["market_close"]
            # NYSE standard close is 4pm ET = 21:00 UTC
            standard_close_utc_hour = 21
            if close_time.hour < standard_close_utc_hour:
                return True
    except Exception:
        pass
    return False


def _is_scheduled_run_time() -> bool:
    """
    Return True if current PT time is within the 5:40–5:55am run window.
    Used by the weekday EventBridge rule (12:45+13:45 UTC).
    Only the invocation that lands in 5:45am PT proceeds.
    """
    pt = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
    return pt.hour == 5 and 40 <= pt.minute <= 55


def handler(event, context):
    """
    AWS Lambda handler for the research pipeline.

    Gate logic:
      - force=True  → bypass all gates (manual testing)
      - weekly_run=True → bypass time gate (Monday 06:00 UTC weekly schedule)
      - Otherwise → require 5:40-5:55am PT time window AND NYSE trading day

    Returns:
        dict with status: "OK" | "SKIPPED" | "ERROR"
    """
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    force = event.get("force", False)
    weekly = event.get("weekly_run", False)
    fd = None

    # Time gate: weekly runs and force bypass; weekday runs require 5:40-5:55am PT
    if not force and not weekly and not _is_scheduled_run_time():
        return {"status": "SKIPPED", "reason": "wrong_time"}

    today = datetime.date.today()

    # Trading day gate: force bypasses; weekly runs on Monday (always a trading day
    # unless it's a rare Monday holiday like MLK Day or Presidents' Day)
    if not force and not is_trading_day(today):
        if weekly:
            print(f"Monday holiday on {today} — running anyway (weekly population refresh).")
        else:
            print(f"Market holiday on {today} — skipping run.")
            return {"status": "SKIPPED", "reason": "market_holiday", "date": str(today)}

    early_close = is_early_close(today) if not weekly else False
    run_date = str(today)

    # Idempotency gate: skip if signals already written for this date
    if not force:
        try:
            import boto3
            from botocore.exceptions import ClientError
            s3 = boto3.client("s3")
            s3.head_object(Bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                           Key=f"signals/{run_date}/signals.json")
            print(f"Signals already exist for {run_date} — skipping (use force=True to override)")
            return {"status": "SKIPPED", "reason": "already_run", "date": run_date}
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                print(f"WARNING: S3 idempotency check failed: {e} — proceeding with run")
        except Exception as e:
            print(f"WARNING: S3 idempotency check failed: {e} — proceeding with run")

    run_type = "weekly population refresh" if weekly else "weekday"
    print(f"Starting alpha-engine-research run for {run_date} ({run_type})"
          + (" [early close]" if early_close else ""))

    _health_start = time.time()

    # Import pipeline (deferred to reduce cold-start time)
    try:
        from archive.manager import ArchiveManager
        from graph.research_graph import build_graph, create_initial_state

        # ── Validate required env vars (fail fast, not 30 min in) ─────
        from config import ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY
        _missing = []
        if not ANTHROPIC_API_KEY:
            _missing.append("ANTHROPIC_API_KEY")
        if not FMP_API_KEY:
            _missing.append("FMP_API_KEY")
        if not FRED_API_KEY:
            _missing.append("FRED_API_KEY")
        if _missing:
            msg = f"Missing required env vars: {', '.join(_missing)}"
            print(f"FATAL: {msg}")
            return {"statusCode": 500, "body": msg}

        archive = ArchiveManager()
        archive.download_db()

        # Run performance tracker before agents
        from scoring.performance_tracker import run_performance_checks
        perf_summary = run_performance_checks(archive.db_conn, run_date)

        # Build and run the LangGraph pipeline
        graph = build_graph()
        initial_state = create_initial_state(
            run_date=run_date,
            archive_manager=archive,
            is_early_close=early_close,
        )
        initial_state["performance_summary"] = perf_summary
        initial_state["flow_doctor"] = None

        # Extract episodic memories from newly completed signal outcomes
        try:
            from memory.episodic import extract_memories
            n_memories = extract_memories(archive.db_conn)
            if n_memories:
                print(f"Extracted {n_memories} new episodic memories from outcomes")
        except Exception as _me:
            print(f"WARNING: memory extraction skipped: {_me}")

        final_state = graph.invoke(initial_state)

        # ── Trajectory validation (Phase 2 eval) ──────────────────
        _trajectory_result = None
        try:
            from evals.trajectory import validate_trajectory
            _trajectory_result = validate_trajectory(
                project_name=os.environ.get("LANGCHAIN_PROJECT", "alpha-research"),
            )
            if _trajectory_result and not _trajectory_result["passed"]:
                import logging as _logging
                _logging.getLogger("evals.trajectory").error(
                    "Trajectory validation failed: %s", _trajectory_result["failures"]
                )
        except Exception as _te:
            print(f"WARNING: trajectory validation skipped: {_te}")

        archive.close()

        # Write health status on success
        try:
            from health_status import write_health
            _population = final_state.get("new_population", [])
            _rotations = final_state.get("population_rotation_events", [])
            write_health(
                bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                module_name="research",
                status="ok",
                run_date=run_date,
                duration_seconds=time.time() - _health_start,
                summary={
                    "n_population": len(_population) if isinstance(_population, list) else 0,
                    "n_rotations": len(_rotations) if isinstance(_rotations, list) else 0,
                    "market_regime": final_state.get("market_regime", "unknown"),
                },
            )
        except Exception as he:
            print(f"WARNING: health status write failed: {he}")

        print(f"Run complete. Email sent: {final_state.get('email_sent', False)}")
        return {
            "status": "OK",
            "date": run_date,
            "email_sent": final_state.get("email_sent", False),
            "early_close": early_close,
            "weekly_run": weekly,
            "trajectory_passed": _trajectory_result["passed"] if _trajectory_result else None,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Pipeline error: {e}\n{tb}")

        # Write health status on failure
        try:
            from health_status import write_health
            write_health(
                bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                module_name="research",
                status="failed",
                run_date=run_date,
                duration_seconds=time.time() - _health_start,
                error=str(e),
            )
        except Exception as he:
            print(f"WARNING: health status write failed: {he}")

        return {"status": "ERROR", "date": run_date, "error": str(e)}
