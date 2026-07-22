#!/usr/bin/env python3
"""
Canyon v9 - Step 144: Dynamic Daily Workflow
============================================

Research-only. No broker connection. No live orders.

This step turns risk gates, sector cycles, timeframe routes, option routes, and
monitor events into a real daily workflow. It is intentionally dynamic: if there
are many risk items, event reviews, sector links, or option blockers, the
workflow grows instead of staying stuck at four generic boxes.

Outputs:
  daily_workflow_steps.csv
  daily_workflow_queue.csv
  daily_workflow_ticker_explain.csv
  daily_workflow_state.json
  daily_workflow_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    df_to_markdown,
    now_str,
    read_csv_safe,
    read_json_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_SECTOR_DETAIL = ROOT / "sector_timeframe_ticker_detail.csv"
IN_SECTOR_OPTION = ROOT / "sector_timeframe_option_route.csv"
IN_RISK_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"
IN_SECTOR_CYCLE = ROOT / "sector_cycle_state.csv"
IN_KEY_LINKS = ROOT / "key_sector_linkage.csv"
IN_EVENTS = ROOT / "desk_monitor_events.csv"
IN_EVENT_DOSSIER = ROOT / "event_research_dossier.csv"
IN_EVENT_REL_WATCHLIST = ROOT / "event_signal_reliability_watchlist.csv"

OUT_STEPS = ROOT / "daily_workflow_steps.csv"
OUT_QUEUE = ROOT / "daily_workflow_queue.csv"
OUT_EXPLAIN = ROOT / "daily_workflow_ticker_explain.csv"
OUT_STATE = ROOT / "daily_workflow_state.json"
OUT_REPORT = ROOT / "daily_workflow_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def severity_rank(value: Any) -> int:
    raw = text(value).upper()
    if "CRITICAL" in raw or "HARD" in raw or "REDUCE_ONLY" in raw:
        return 0
    if "SIZE_DOWN" in raw or "WARNING" in raw or "REVIEW" in raw:
        return 1
    if "INFO" in raw or "WATCH" in raw:
        return 2
    return 3


def status_bucket(row: pd.Series) -> str:
    risk = text(row.get("risk_action")).upper()
    event = text(row.get("event_gate")).upper()
    option = text(row.get("option_route")).upper()
    action = text(row.get("sector_adjusted_desk_action")).upper()
    if "REDUCE_ONLY" in risk or "RISK FIRST" in action:
        return "Risk first"
    if "SIZE_DOWN" in risk or "TINY" in action:
        return "Tiny research only"
    if "MISSING" in event or "REVIEW" in event:
        return "Event check"
    if "NO NEW OPTION" in option or "NO CALL" in option:
        return "Option blocked"
    if "PUT" in option or "HEDGE" in option:
        return "Hedge review"
    if "WATCH" in action:
        return "Watch"
    return "Research"


def priority_label(rank: int) -> str:
    if rank <= 8:
        return "High"
    if rank <= 18:
        return "Medium"
    return "Normal"


def source_trace(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        raw = text(part)
        if not raw:
            continue
        for item in raw.replace("+", ";").split(";"):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return "; ".join(values)


def build_queue() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    routes = read_csv_safe(IN_SECTOR_ROUTE)
    details = read_csv_safe(IN_SECTOR_DETAIL)
    options = read_csv_safe(IN_SECTOR_OPTION)
    risk_queue = read_csv_safe(IN_RISK_QUEUE)
    sector_cycle = read_csv_safe(IN_SECTOR_CYCLE)
    key_links = read_csv_safe(IN_KEY_LINKS)
    events = read_csv_safe(IN_EVENTS)
    event_dossier = read_csv_safe(IN_EVENT_DOSSIER)
    event_reliability = read_csv_safe(IN_EVENT_REL_WATCHLIST)
    risk_state = read_json_safe(IN_RISK_DESK, default={})

    queue_rows: list[dict[str, Any]] = []
    explain_rows: list[dict[str, Any]] = []

    if not routes.empty:
        risk_by_ticker = (
            risk_queue.drop_duplicates("ticker", keep="first").set_index("ticker")
            if not risk_queue.empty and "ticker" in risk_queue.columns
            else pd.DataFrame()
        )
        event_by_ticker = (
            event_dossier.drop_duplicates("ticker", keep="first").set_index("ticker")
            if not event_dossier.empty and "ticker" in event_dossier.columns
            else pd.DataFrame()
        )
        if not event_reliability.empty and "target_ticker" in event_reliability.columns:
            event_reliability = event_reliability.copy()
            event_reliability["target_ticker"] = event_reliability["target_ticker"].astype(str).str.upper().str.strip()
            event_rel_by_ticker = event_reliability.drop_duplicates("target_ticker", keep="first").set_index("target_ticker")
            event_rel_counts = event_reliability["target_ticker"].value_counts().to_dict()
        else:
            event_rel_by_ticker = pd.DataFrame()
            event_rel_counts = {}
        event_counts = {}
        if not events.empty and "ticker" in events.columns:
            event_counts = events["ticker"].fillna("PORTFOLIO").astype(str).value_counts().to_dict()

        for _, row in routes.iterrows():
            ticker = text(row.get("ticker"))
            if not ticker:
                continue
            rrow = risk_by_ticker.loc[ticker] if not risk_by_ticker.empty and ticker in risk_by_ticker.index else pd.Series(dtype=object)
            erow = event_by_ticker.loc[ticker] if not event_by_ticker.empty and ticker in event_by_ticker.index else pd.Series(dtype=object)
            relrow = event_rel_by_ticker.loc[ticker] if not event_rel_by_ticker.empty and ticker in event_rel_by_ticker.index else pd.Series(dtype=object)
            bucket = status_bucket(row)
            rel_status = text(relrow.get("calibrated_reliability_status")).upper()
            rel_action = text(relrow.get("calibrated_research_action")).upper()
            if bucket not in {"Risk first", "Tiny research only"} and (
                "LOW_SAMPLE" in rel_status
                or "UNPROVEN" in rel_status
                or "PENDING" in text(relrow.get("model_seen_audit_status")).upper()
                or "WATCH_ONLY" in rel_action
                or "DO_NOT_UPGRADE" in rel_action
            ):
                bucket = "Event check"
            rank = (
                severity_rank(row.get("risk_action")) * 10
                + severity_rank(row.get("event_gate")) * 3
                + (0 if "Risk" in bucket else 2 if "Event" in bucket else 4)
            )
            current_weight = safe_float(rrow.get("current_weight_pct"), np.nan)
            recommended_weight = safe_float(rrow.get("recommended_risk_weight_pct"), np.nan)
            reduction = safe_float(rrow.get("risk_reduction_pct_of_current"), np.nan)
            workflow_action = text(rrow.get("required_next_action"))
            if not workflow_action:
                if bucket == "Risk first":
                    workflow_action = "Handle risk reduction before considering any new idea."
                elif bucket == "Tiny research only":
                    workflow_action = "Treat as tiny paper/research size only until risk improves."
                elif bucket == "Event check":
                    if rel_action:
                        workflow_action = "Do not upgrade from news alone. Confirm price, volume, risk, event source, and local reliability first."
                    else:
                        workflow_action = "Verify earnings, filings, headline source, and missing evidence."
                elif bucket == "Option blocked":
                    workflow_action = "Do not open bullish options; wait for risk/event gates to clear."
                elif bucket == "Hedge review":
                    workflow_action = "Review defensive put/hedge research only."
                else:
                    workflow_action = "Review route and source evidence before adding to focus list."

            why = text(row.get("why"))
            source = source_trace(
                row.get("source_file"),
                rrow.get("source_file"),
                erow.get("source_file"),
                "event_signal_reliability_watchlist.csv" if rel_action or rel_status else "",
            )
            bucket_base_rank = {
                "Risk first": 0,
                "Tiny research only": 6,
                "Event check": 12,
                "Option blocked": 16,
                "Hedge review": 18,
                "Watch": 24,
                "Research": 28,
            }.get(bucket, 30)
            rank = (
                bucket_base_rank
                + severity_rank(row.get("risk_action"))
                + severity_rank(row.get("event_gate"))
                + max(0, 3 - int(event_counts.get(ticker, 0)))
            )
            queue_rows.append({
                "priority_rank": rank,
                "priority": priority_label(rank),
                "ticker": ticker,
                "sector": text(row.get("sector")),
                "workflow_bucket": bucket,
                "what_to_do": workflow_action,
                "sector_cycle_state": text(row.get("sector_cycle_state")),
                "linked_sector": text(row.get("linked_sector")),
                "linked_sector_cycle_state": text(row.get("linked_sector_cycle_state")),
                "best_horizon": text(row.get("best_horizon_after_sector")),
                "sector_adjusted_action": text(row.get("sector_adjusted_desk_action")),
                "option_route": text(row.get("option_route")),
                "risk_action": text(row.get("risk_action")),
                "event_gate": text(row.get("event_gate")),
                "monitor_event_count": int(event_counts.get(ticker, 0)),
                "event_reliability_count": int(event_rel_counts.get(ticker, 0)),
                "event_reliability_status": text(relrow.get("calibrated_reliability_status")),
                "event_reliability_action": text(relrow.get("calibrated_research_action")),
                "calibrated_event_score": round(safe_float(relrow.get("calibrated_event_score"), np.nan), 3) if np.isfinite(safe_float(relrow.get("calibrated_event_score"), np.nan)) else np.nan,
                "calibrated_reliability_score": round(safe_float(relrow.get("calibrated_reliability_score"), np.nan), 3) if np.isfinite(safe_float(relrow.get("calibrated_reliability_score"), np.nan)) else np.nan,
                "event_calibration_note": text(relrow.get("calibration_note")),
                "event_calibration_headline": text(relrow.get("headline")),
                "current_weight_pct": round(current_weight, 3) if np.isfinite(current_weight) else np.nan,
                "recommended_weight_pct": round(recommended_weight, 3) if np.isfinite(recommended_weight) else np.nan,
                "risk_reduction_pct": round(reduction, 3) if np.isfinite(reduction) else np.nan,
                "what_to_watch": text(row.get("what_to_watch")),
                "what_would_change": text(row.get("what_would_change")),
                "why": why,
                "source_files": source,
                "next_dashboard_section": "Time Frames -> Sector-Aware Timeframe Router",
                "research_only": True,
            })
            explain_rows.append({
                "ticker": ticker,
                "plain_english_summary": (
                    f"{ticker}: {bucket}. "
                    f"Sector cycle is {text(row.get('sector_cycle_state'))}; "
                    f"best horizon is {text(row.get('best_horizon_after_sector'))}; "
                    f"option route is {text(row.get('option_route'))}."
                ),
                "risk_evidence": text(rrow.get("reason_stack")) or text(row.get("risk_action")),
                "event_evidence": text(erow.get("catalysts")) or text(erow.get("required_next_action")) or text(row.get("event_gate")),
                "event_reliability_evidence": (
                    f"{text(relrow.get('calibrated_research_action'))}; "
                    f"{text(relrow.get('calibrated_reliability_status'))}; "
                    f"{text(relrow.get('calibration_note'))}; "
                    f"{text(relrow.get('headline'))}"
                ).strip("; "),
                "sector_evidence": text(row.get("why")),
                "option_evidence": text(row.get("option_route")),
                "source_files": source,
                "research_only": True,
            })

    queue = pd.DataFrame(queue_rows)
    if not queue.empty:
        queue = queue.sort_values(["priority_rank", "monitor_event_count", "ticker"], ascending=[True, False, True]).reset_index(drop=True)

    explain = pd.DataFrame(explain_rows)

    step_rows: list[dict[str, Any]] = []

    def add_step(station: str, status: str, count: int, what: str, why: str, source: str, next_section: str) -> None:
        step_rows.append({
            "step_order": len(step_rows) + 1,
            "status": status,
            "station": station,
            "items": int(count),
            "what_to_do": what,
            "why_this_exists": why,
            "source_files": source,
            "next_dashboard_section": next_section,
            "research_only": True,
        })

    master_action = text(risk_state.get("master_risk_action") or risk_state.get("overall_status"))
    risk_first = queue[queue["workflow_bucket"].isin(["Risk first", "Tiny research only"])] if not queue.empty else pd.DataFrame()
    event_check = queue[queue["workflow_bucket"].eq("Event check")] if not queue.empty else pd.DataFrame()
    event_reliability_review = (
        queue[
            queue.get("event_reliability_action", pd.Series(dtype=str)).astype(str).str.len().gt(0)
            | queue.get("event_reliability_status", pd.Series(dtype=str)).astype(str).str.upper().str.contains("LOW_SAMPLE|UNPROVEN|PENDING", na=False)
        ]
        if not queue.empty
        else pd.DataFrame()
    )
    option_blocked = queue[queue["workflow_bucket"].isin(["Option blocked", "Hedge review"])] if not queue.empty else pd.DataFrame()
    active_sectors = sector_cycle[sector_cycle.get("cycle_state", pd.Series(dtype=str)).astype(str).str.contains("Leadership|Early improvement|Crowded", case=False, na=False)] if not sector_cycle.empty else pd.DataFrame()
    link_rows = key_links.head(12) if not key_links.empty else pd.DataFrame()
    monitor_critical = events[events.get("severity", pd.Series(dtype=str)).astype(str).str.upper().isin(["CRITICAL", "WARNING"])] if not events.empty else pd.DataFrame()

    add_step(
        "Risk Desk",
        "REVIEW" if not risk_first.empty else "OK",
        len(risk_first),
        "Resolve risk-first and tiny-research names before new ideas.",
        f"Master risk state is {master_action or 'not available'}; risk gates can block sector and option signals.",
        "risk_desk_overview.json; risk_desk_ticker_action_queue.csv; sector_timeframe_route.csv",
        "Performance -> Risk Desk Summary",
    )
    add_step(
        "Sector Map",
        "REVIEW" if not active_sectors.empty else "WAIT",
        len(active_sectors),
        "Use sector cycle to decide where research attention belongs today.",
        "Leadership, crowded leadership, and early improvement sectors guide attention, not sizing.",
        "sector_cycle_state.csv; key_sector_linkage.csv",
        "News Room -> Sector Cycle and Links",
    )
    add_step(
        "Ticker Route",
        "REVIEW" if not queue.empty else "WAIT",
        len(queue),
        "Work through ticker queue by priority. Open each ticker explanation before acting.",
        "Combines sector cycle, risk gate, event gate, and timeframe route.",
        "sector_timeframe_route.csv; daily_workflow_ticker_explain.csv",
        "Time Frames -> Sector-Aware Timeframe Router",
    )
    add_step(
        "Event Check",
        "REVIEW" if not event_check.empty else "OK",
        len(event_check),
        "Clear missing earnings/news/SEC evidence before upgrading any route.",
        "Event gaps can turn a good-looking setup into a no-trade.",
        "event_research_dossier.csv; stock_news.json",
        "News Room -> Event Research",
    )
    add_step(
        "News Reliability",
        "REVIEW" if not event_reliability_review.empty else "OK",
        len(event_reliability_review),
        "Use calibrated news only as context until local proof, price reaction, and source timing are clean.",
        "Step166 grades whether each event/news signal is locally proven, low-sample, unproven, or still pending after first seen.",
        "event_signal_reliability_watchlist.csv; event_signal_reliability_adjusted_panel.csv; event_signal_local_audit_returns.csv",
        "News Room -> Event Reliability Calibration",
    )
    add_step(
        "Options Route",
        "REVIEW" if not option_blocked.empty else "OK",
        len(option_blocked),
        "Check whether route is no-new-option, hedge-only, or defined-risk review.",
        "Options cannot override risk, event, liquidity, or execution gates.",
        "sector_timeframe_option_route.csv; options_playbook.csv",
        "Time Frames -> Sector-adjusted option route",
    )
    add_step(
        "Sector Links",
        "WATCH" if not link_rows.empty else "WAIT",
        len(link_rows),
        "Review key leader-to-follower and news/theme read-through links.",
        "Major catalysts can travel across suppliers, peers, customers, and correlated sectors.",
        "key_sector_linkage.csv; news_supply_chain_readthrough.csv",
        "News Room -> Important sector links",
    )
    add_step(
        "Desk Monitor",
        "REVIEW" if not monitor_critical.empty else "OK",
        len(monitor_critical),
        "Review price breaks, volume spikes, correlation breaks, spread widening, and news shocks.",
        "Monitor events can force wait/review even when route scores look good.",
        "desk_monitor_events.csv; desk_monitor_ticker_state.csv",
        "Alerts -> Desk Monitor",
    )

    steps = pd.DataFrame(step_rows)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "workflow_steps": int(len(steps)),
        "queue_rows": int(len(queue)),
        "risk_first_rows": int(len(risk_first)),
        "event_check_rows": int(len(event_check)),
        "event_reliability_review_rows": int(len(event_reliability_review)),
        "option_review_rows": int(len(option_blocked)),
        "sector_focus_rows": int(len(active_sectors)),
        "key_link_rows": int(len(link_rows)),
        "monitor_review_rows": int(len(monitor_critical)),
        "master_risk_state": master_action or "NO_DATA",
        "top_queue_tickers": queue["ticker"].head(8).tolist() if not queue.empty and "ticker" in queue.columns else [],
        "outputs": {
            "steps": OUT_STEPS.name,
            "queue": OUT_QUEUE.name,
            "explain": OUT_EXPLAIN.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
        "logic": "Dynamic workflow. The number of rows changes with today's risk, sector, event, option, and monitor evidence.",
    }
    return steps, queue, explain, state


def main() -> int:
    steps, queue, explain, state = build_queue()
    steps.to_csv(OUT_STEPS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    explain.to_csv(OUT_EXPLAIN, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Workflow steps: {state.get('workflow_steps', 0)}",
        f"- Queue rows: {state.get('queue_rows', 0)}",
        f"- Risk-first rows: {state.get('risk_first_rows', 0)}",
        f"- Event-check rows: {state.get('event_check_rows', 0)}",
        f"- News-reliability review rows: {state.get('event_reliability_review_rows', 0)}",
        f"- Option-review rows: {state.get('option_review_rows', 0)}",
        f"- Sector-focus rows: {state.get('sector_focus_rows', 0)}",
        f"- Key-link rows: {state.get('key_link_rows', 0)}",
        f"- Monitor-review rows: {state.get('monitor_review_rows', 0)}",
        f"- Master risk state: {state.get('master_risk_state', 'NO_DATA')}",
        "",
        "## Workflow Steps",
        "",
        df_to_markdown(steps, max_rows=40),
        "",
        "## Ticker Queue",
        "",
        df_to_markdown(queue, max_rows=80),
        "",
        "## Ticker Explanations",
        "",
        df_to_markdown(explain, max_rows=80),
        "",
        "## Product Truth",
        "",
        "This workflow is not an order system. It is a daily research operating checklist. No broker connection and no live order path exist.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 144 - Dynamic Daily Workflow", sections)

    print(f"wrote {OUT_STEPS.name} rows={len(steps)}")
    print(f"wrote {OUT_QUEUE.name} rows={len(queue)}")
    print(f"wrote {OUT_EXPLAIN.name} rows={len(explain)}")
    print(f"risk_first_rows={state.get('risk_first_rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
