"""
Lambda entry point — main daily morning pipeline (§10.1).

Triggered by EventBridge on NYSE trading days (Mon–Fri, 5:45am PT).
Checks for market holidays and skips if NYSE is closed.
Invokes the full LangGraph research pipeline.
"""

from __future__ import annotations

import datetime
import os
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
    Allows a single EventBridge rule to fire at both 12:45 and 13:45 UTC;
    only the invocation that lands in 5:45am PT proceeds.
    """
    pt = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
    return pt.hour == 5 and 40 <= pt.minute <= 55


def handler(event, context):
    """
    AWS Lambda handler for the daily morning research pipeline.

    Triggered by EventBridge at 12:45 and 13:45 UTC (Mon–Fri). Only runs
    when current PT time is 5:40–5:55am; the other invocation exits early.
    No DST rule swap required.

    Returns:
        dict with status: "OK" | "SKIPPED" | "ERROR"
    """
    # Time gate: only run when it's 5:45am PT (handles DST automatically)
    # Pass {"force": true} in the event payload to bypass for manual runs.
    if not event.get("force") and not _is_scheduled_run_time():
        return {"status": "SKIPPED", "reason": "wrong_time"}

    today = datetime.date.today()

    # Check market holiday
    if not is_trading_day(today):
        print(f"Market holiday on {today} — skipping run.")
        return {"status": "SKIPPED", "reason": "market_holiday", "date": str(today)}

    early_close = is_early_close(today)
    run_date = str(today)

    print(f"Starting alpha-engine-research run for {run_date}"
          + (" [early close]" if early_close else ""))

    # Import pipeline (deferred to reduce cold-start time)
    try:
        from graph.research_graph import build_graph, create_initial_state
        from archive.manager import ArchiveManager

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

        final_state = graph.invoke(initial_state)

        archive.close()

        print(f"Run complete. Email sent: {final_state.get('email_sent', False)}")
        return {
            "status": "OK",
            "date": run_date,
            "email_sent": final_state.get("email_sent", False),
            "early_close": early_close,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Pipeline error: {e}\n{tb}")
        return {"status": "ERROR", "date": run_date, "error": str(e)}
