#!/usr/bin/env python3
"""
Canyon v9 Step 180 - Action Readiness Detail Cards.

Research-only. No broker connection. No live orders.

Step179 produces ticker drilldown tables. Step180 turns those tables into a
card-ready PM deck so the dashboard can show one readable card per ticker:
  - current state
  - first blocker
  - current route authority
  - first source file to open
  - next checks
  - route conflicts against older ticker-room outputs

This step is presentation and audit structure only. It cannot trade, rebalance,
write to the paper ledger, or override risk.

Outputs:
  action_readiness_detail_cards.csv
  action_readiness_detail_card_panels.csv
  action_readiness_card_deck_summary.csv
  action_readiness_card_deck_state.json
  action_readiness_card_deck_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_CARDS = ROOT / "action_readiness_detail_cards.csv"
OUT_PANELS = ROOT / "action_readiness_detail_card_panels.csv"
OUT_SUMMARY = ROOT / "action_readiness_card_deck_summary.csv"
OUT_STATE = ROOT / "action_readiness_card_deck_state.json"
OUT_REPORT = ROOT / "action_readiness_card_deck_report.md"


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def as_upper(value: Any, default: str = "") -> str:
    text = as_text(value, default)
    return text.upper() if text else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def compact(value: Any, limit: int = 260) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.DataFrame]:
    if df.empty or ticker_col not in df.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for ticker, sub in df.groupby(df[ticker_col].astype(str).str.upper()):
        if ticker:
            out[ticker] = sub.copy()
    return out


def card_status(row: pd.Series) -> str:
    conflict = as_upper(row.get("decision_route_conflict_status"))
    stage = as_upper(row.get("current_stage"))
    first_status = as_upper(row.get("first_gate_status"))
    if conflict == "ROUTE_CONFLICT_REVIEW":
        return "Conflict review"
    if "RISK_REPAIR_REQUIRED" in stage:
        return "Risk blocked"
    if "BLOCKED" in first_status:
        return "Blocked"
    if "TRIGGER" in stage:
        return "Trigger watch"
    return "Review"


def urgency(status: str, row: pd.Series) -> str:
    if status in {"Conflict review", "Risk blocked", "Blocked"}:
        return "High"
    if as_upper(row.get("current_stage")) == "TRIGGER_WATCH":
        return "Watch"
    return "Review"


def card_bias(row: pd.Series) -> str:
    route = as_upper(row.get("route_after_all_gates_clear"))
    if "NO NEW EXPOSURE" in route:
        return "No new exposure"
    if "PUT" in route or "HEDGE" in route:
        return "Put / hedge only after gates clear"
    if "UNDERLYING" in route:
        return "Underlying only after gates clear"
    if "CALL" in route:
        return "Defined-risk call only after gates clear"
    return "Research only"


def split_next_checks(text: Any) -> list[str]:
    raw = as_text(text)
    if not raw:
        return []
    parts: list[str] = []
    current = ""
    for token in raw.replace(" 2.", "\n2.").replace(" 3.", "\n3.").splitlines():
        token = token.strip()
        if not token:
            continue
        if token[:2] in {"1.", "2.", "3."}:
            if current:
                parts.append(current.strip())
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        parts.append(current.strip())
    return [compact(p, 230) for p in parts[:3]]


def source_trace_summary(source_rows: pd.DataFrame, ticker: str) -> str:
    if source_rows.empty:
        return "No source trace rows found."
    parts = []
    for _, src in source_rows.head(4).iterrows():
        parts.append(
            f"{as_text(src.get('evidence_area'), 'Evidence')}: "
            f"{as_text(src.get('source_status'), 'NO_STATUS')} "
            f"from {as_text(src.get('source_file'), 'NO_SOURCE')}"
        )
    return " | ".join(parts)


def blocker_summary(blockers: pd.DataFrame) -> str:
    if blockers.empty:
        return "No open blockers in blocker explainer."
    parts = []
    for _, b in blockers.head(3).iterrows():
        parts.append(
            f"{as_text(b.get('gate_name'), 'Gate')}={as_text(b.get('gate_status'), 'NO_STATUS')}"
        )
    return "; ".join(parts)


def checklist_summary(checklist: pd.DataFrame) -> str:
    if checklist.empty:
        return "No manual checklist rows found."
    parts = []
    for _, row in checklist.head(3).iterrows():
        parts.append(
            f"{int(safe_float(row.get('check_order'), 0))}. {as_text(row.get('check_name'), 'Check')} -> {as_text(row.get('source_file'), 'NO_SOURCE')}"
        )
    return " | ".join(parts)


def build_cards(
    drilldown: pd.DataFrame,
    source_trace: pd.DataFrame,
    blockers: pd.DataFrame,
    checklist: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if drilldown.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    source_idx = by_ticker(source_trace)
    blocker_idx = by_ticker(blockers)
    checklist_idx = by_ticker(checklist)

    card_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []

    for _, row in drilldown.sort_values("drilldown_rank").iterrows():
        ticker = as_upper(row.get("ticker"))
        status = card_status(row)
        pri = urgency(status, row)
        checks = split_next_checks(row.get("next_3_checks"))
        while len(checks) < 3:
            checks.append("Recheck source freshness and keep research-only guardrails.")

        ticker_sources = source_idx.get(ticker, pd.DataFrame())
        ticker_blockers = blocker_idx.get(ticker, pd.DataFrame())
        ticker_checklist = checklist_idx.get(ticker, pd.DataFrame())
        conflict_status = as_text(row.get("decision_route_conflict_status"), "NO_ROUTE_CONFLICT")
        conflict_note = as_text(row.get("decision_route_conflict_note"), "")
        conflict_flag = conflict_status == "ROUTE_CONFLICT_REVIEW"

        headline = f"{ticker}: {as_text(row.get('first_blocking_gate'), 'First gate')} is the next blocker"
        if conflict_flag:
            headline = f"{ticker}: route conflict needs review"
        elif as_upper(row.get("current_stage")) == "RISK_REPAIR_REQUIRED":
            headline = f"{ticker}: risk repair required before any route review"

        card_rows.append({
            "ticker": ticker,
            "card_rank": row.get("drilldown_rank"),
            "card_status": status,
            "review_priority": pri,
            "headline": headline,
            "subheadline": compact(row.get("why_blocked_plain_english"), 360),
            "current_stage": row.get("current_stage"),
            "readiness_score": row.get("readiness_score"),
            "action_bias": card_bias(row),
            "route_authority": "Step178/179 current readiness route is authoritative when older ticker-room text conflicts.",
            "route_after_all_gates_clear": row.get("route_after_all_gates_clear"),
            "option_permission_after_repair": row.get("option_permission_after_repair"),
            "first_blocking_gate": row.get("first_blocking_gate"),
            "first_gate_status": row.get("first_gate_status"),
            "first_source_to_open": row.get("first_source_to_open"),
            "first_clear_condition": compact(row.get("first_clear_condition"), 420),
            "next_check_1": checks[0],
            "next_check_2": checks[1],
            "next_check_3": checks[2],
            "trigger_to_watch": row.get("trigger_to_watch"),
            "risk_summary": compact(row.get("risk_summary"), 260),
            "monitor_summary": compact(row.get("monitor_summary"), 260),
            "option_summary": compact(row.get("option_summary"), 260),
            "event_news_summary": compact(row.get("event_news_summary"), 260),
            "sector_portfolio_summary": compact(row.get("sector_portfolio_summary"), 300),
            "decision_route_conflict_status": conflict_status,
            "decision_route_conflict_note": compact(conflict_note, 420),
            "source_trace_summary": source_trace_summary(ticker_sources, ticker),
            "open_blocker_summary": blocker_summary(ticker_blockers),
            "manual_checklist_summary": checklist_summary(ticker_checklist),
            "do_not_do": row.get("do_not_do"),
            "source_files": row.get("source_trace_files"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        panel_specs = [
            (
                "Current state",
                status,
                f"{ticker} / {as_text(row.get('current_stage'), 'NO_STAGE')}",
                compact(row.get("why_blocked_plain_english"), 480),
                "action_readiness_ticker_drilldown.csv",
            ),
            (
                "First source",
                row.get("first_gate_status"),
                as_text(row.get("first_source_to_open"), "NO_SOURCE"),
                compact(row.get("first_clear_condition"), 480),
                as_text(row.get("first_source_to_open"), "NO_SOURCE"),
            ),
            (
                "Route authority",
                conflict_status,
                as_text(row.get("route_after_all_gates_clear"), "NO_ROUTE"),
                compact(conflict_note or "No direct route conflict found.", 480),
                "action_readiness_ticker_drilldown.csv; ticker_decision_room.csv",
            ),
            (
                "Source trace",
                "Review",
                f"{len(ticker_sources)} source rows",
                source_trace_summary(ticker_sources, ticker),
                "action_readiness_source_trace.csv",
            ),
            (
                "Open blockers",
                "Blocked" if not ticker_blockers.empty else "Review",
                f"{len(ticker_blockers)} open blocker rows",
                blocker_summary(ticker_blockers),
                "action_readiness_blocker_explainer.csv",
            ),
            (
                "Manual checks",
                "Review",
                f"{len(ticker_checklist)} checklist rows",
                checklist_summary(ticker_checklist),
                "action_readiness_manual_checklist.csv",
            ),
        ]
        for order, (panel, panel_status, title, detail, sources) in enumerate(panel_specs, 1):
            panel_rows.append({
                "ticker": ticker,
                "panel_order": order,
                "panel": panel,
                "panel_status": panel_status,
                "title": title,
                "lead": detail,
                "detail": compact(row.get("do_not_do"), 360) if panel == "Manual checks" else detail,
                "source_files": sources,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

    cards = pd.DataFrame(card_rows)
    panels = pd.DataFrame(panel_rows)
    summary_rows = []
    for col in ["card_status", "current_stage", "decision_route_conflict_status"]:
        if col in cards.columns:
            counts = cards[col].value_counts(dropna=False).reset_index()
            counts.columns = ["bucket", "count"]
            counts.insert(0, "summary_scope", col)
            summary_rows.extend(counts.to_dict("records"))
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["research_only"] = True
        summary["no_broker_connection"] = True
        summary["no_live_orders"] = True
    return cards, panels, summary


def build_state(cards: pd.DataFrame, panels: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    if cards.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_ACTION_READINESS_CARD_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    first = cards.sort_values("card_rank").iloc[0]
    conflicts = int(cards["decision_route_conflict_status"].astype(str).eq("ROUTE_CONFLICT_REVIEW").sum())
    blocked = int(cards["card_status"].astype(str).isin(["Blocked", "Risk blocked", "Conflict review"]).sum())
    return {
        "date": today_str(),
        "overall_status": "ACTION_READINESS_DETAIL_CARDS_ACTIVE",
        "card_rows": int(len(cards)),
        "panel_rows": int(len(panels)),
        "summary_rows": int(len(summary)),
        "blocked_or_conflict_cards": blocked,
        "route_conflict_cards": conflicts,
        "top_card_ticker": as_text(first.get("ticker")),
        "top_card_status": as_text(first.get("card_status")),
        "top_card_source": as_text(first.get("first_source_to_open")),
        "truth": "This is a presentation deck only. It cannot trade, rebalance, or override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "cards": OUT_CARDS.name,
            "panels": OUT_PANELS.name,
            "summary": OUT_SUMMARY.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    drilldown = read_csv_safe(ROOT / "action_readiness_ticker_drilldown.csv")
    source_trace = read_csv_safe(ROOT / "action_readiness_source_trace.csv")
    blockers = read_csv_safe(ROOT / "action_readiness_blocker_explainer.csv")
    checklist = read_csv_safe(ROOT / "action_readiness_manual_checklist.csv")

    cards, panels, summary = build_cards(drilldown, source_trace, blockers, checklist)
    state = build_state(cards, panels, summary)

    cards.to_csv(OUT_CARDS, index=False)
    panels.to_csv(OUT_PANELS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_json(OUT_STATE, state)

    card_cols = [c for c in [
        "ticker", "card_status", "review_priority", "headline",
        "current_stage", "first_blocking_gate", "first_source_to_open",
        "route_after_all_gates_clear", "decision_route_conflict_status",
        "next_check_1",
    ] if c in cards.columns]
    panel_cols = [c for c in [
        "ticker", "panel_order", "panel", "panel_status",
        "title", "lead", "source_files",
    ] if c in panels.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Card rows: {state.get('card_rows')}\n"
        f"- Panel rows: {state.get('panel_rows')}\n"
        f"- Route conflict cards: {state.get('route_conflict_cards')}\n"
        f"- Top ticker/source: {state.get('top_card_ticker')} / {state.get('top_card_source')}\n",
        "## Detail cards\n" + df_to_markdown(cards[card_cols] if card_cols else cards, 40),
        "## Detail panels\n" + df_to_markdown(panels[panel_cols] if panel_cols else panels, 80),
        "## Summary\n" + df_to_markdown(summary, 40),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Cards are display surfaces. They do not create permission.\n"
        "- Step178/179 route authority overrides older ticker-room phrasing when a conflict is detected.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 180 - Action Readiness Detail Cards", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 180 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Cards: {state.get('card_rows')}")
    print(f"Panels: {state.get('panel_rows')}")
    print(f"Route conflicts: {state.get('route_conflict_cards')}")
    print(f"Top ticker/source: {state.get('top_card_ticker')} / {state.get('top_card_source')}")
    print("Outputs:")
    for path in [OUT_CARDS, OUT_PANELS, OUT_SUMMARY, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
