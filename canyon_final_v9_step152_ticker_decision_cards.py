#!/usr/bin/env python3
"""
Canyon v9 - Step 152: Ticker Decision Cards
===========================================

Research-only. No broker connection. No live orders.

Step151 creates ticker decision rooms. Step152 reshapes those rooms into
professional card-ready data: one summary card per ticker and one panelized
evidence chain per ticker.

Outputs:
  ticker_decision_cards.csv
  ticker_decision_card_panels.csv
  ticker_decision_card_state.json
  ticker_decision_card_report.md
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
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_ROOM = ROOT / "ticker_decision_room.csv"
IN_NEWS = ROOT / "ticker_decision_room_news.csv"
IN_TICKETS = ROOT / "conditional_action_tickets.csv"
IN_EVIDENCE = ROOT / "ticker_evidence_binder.csv"
IN_MONITOR = ROOT / "desk_monitor_events.csv"

OUT_CARDS = ROOT / "ticker_decision_cards.csv"
OUT_PANELS = ROOT / "ticker_decision_card_panels.csv"
OUT_STATE = ROOT / "ticker_decision_card_state.json"
OUT_REPORT = ROOT / "ticker_decision_card_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def shorten(value: Any, limit: int = 520) -> str:
    raw = text(value)
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "..."


def one_by_ticker(df: pd.DataFrame, key: str = "ticker") -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[key] = out[key].astype(str).str.upper().str.strip()
    out = out[out[key] != ""]
    if out.empty:
        return pd.DataFrame()
    return out.drop_duplicates(key, keep="first").set_index(key)


def row_at(indexed: pd.DataFrame, ticker: str) -> pd.Series:
    if indexed.empty or ticker not in indexed.index:
        return pd.Series(dtype=object)
    row = indexed.loc[ticker]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def normalize_ticker_column(df: pd.DataFrame, column: str = "ticker") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[column] = out[column].astype(str).str.upper().str.strip()
    out = out[out[column] != ""]
    return out


def first_nonempty(*values: Any) -> str:
    for value in values:
        raw = text(value)
        if raw:
            return raw
    return ""


def status_band(room_status: str, decision_now: str, risk_gate: str) -> str:
    raw = f"{room_status} {decision_now} {risk_gate}".upper()
    if any(token in raw for token in ["BLOCKED", "WAIT", "REDUCE", "SIZE_DOWN"]):
        return "Blocked"
    if any(token in raw for token in ["MONITOR", "SOURCE REVIEW"]):
        return "Review first"
    if "CALL WATCH" in raw:
        return "Call watch"
    if "HEDGE" in raw or "PUT" in raw:
        return "Hedge watch"
    if "TINY" in raw:
        return "Tiny paper"
    return "Research"


def action_bias(route: str, option_side: str, vehicle: str) -> str:
    raw = f"{route} {option_side} {vehicle}".upper()
    if "CALL" in raw:
        return "Bullish if gates clear"
    if "PUT" in raw or "HEDGE" in raw:
        return "Protective / bearish if gates clear"
    if "STOCK" in raw or "ETF" in raw or "TINY" in raw:
        return "Tiny equity if gates clear"
    return "No vehicle yet"


def urgency(
    room_status: str,
    monitor_count: Any,
    event_gate: str,
    news_direction: str,
    calibrated_status: str = "",
    calibrated_action: str = "",
) -> str:
    monitor = safe_float(monitor_count)
    raw = f"{room_status} {event_gate} {news_direction} {calibrated_status} {calibrated_action}".upper()
    if monitor > 0 or "BLOCKED" in raw:
        return "High review"
    if "MISSING" in raw or "REVIEW" in raw or "LOW_SAMPLE" in raw or "UNPROVEN" in raw or "PENDING" in raw:
        return "Source review"
    if "BULLISH" in raw or "BEARISH" in raw:
        return "Watch"
    return "Low"


def panel_status(panel: str, row: pd.Series) -> str:
    if panel == "Decision":
        return status_band(row.get("room_status"), row.get("decision_now"), row.get("risk_gate"))
    if panel == "Risk":
        return "Blocked" if upper(row.get("risk_gate")) in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCKED"} else text(row.get("risk_gate")) or "Review"
    if panel == "News":
        calibrated_status = upper(row.get("top_news_reliability_status"))
        calibrated_action = upper(row.get("top_news_calibrated_action"))
        if "LOW_SAMPLE" in calibrated_status:
            return "Small-sample review"
        if "UNPROVEN" in calibrated_status:
            return "Unproven local context"
        if "WATCH_ONLY" in calibrated_action:
            return "Confirmation required"
        if calibrated_action:
            return calibrated_action.replace("_", " ").title()
        return text(row.get("top_news_direction")) or "No mapped news"
    if panel == "Options":
        return action_bias(row.get("route_after_gates_clear"), row.get("option_side_after_clear"), row.get("primary_vehicle_after_clear"))
    if panel == "Timing":
        return "Trigger required"
    if panel == "Sources":
        return "Source trail"
    return "Review"


def join_evidence(evidence_rows: pd.DataFrame, limit: int = 420) -> str:
    if evidence_rows.empty:
        return ""
    pieces: list[str] = []
    for _, row in evidence_rows.head(4).iterrows():
        pieces.append(f"{text(row.get('layer'))}: {text(row.get('evidence'))}")
    return shorten(" | ".join(pieces), limit)


def build_cards() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    room = read_csv_safe(IN_ROOM)
    news = normalize_ticker_column(read_csv_safe(IN_NEWS))
    tickets = one_by_ticker(read_csv_safe(IN_TICKETS))
    evidence = normalize_ticker_column(read_csv_safe(IN_EVIDENCE))
    monitor = normalize_ticker_column(read_csv_safe(IN_MONITOR))

    if room.empty or "ticker" not in room.columns:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_DECISION_ROOM",
            "card_rows": 0,
            "panel_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    room = room.copy()
    room["ticker"] = room["ticker"].astype(str).str.upper().str.strip()
    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []

    for _, r in room.iterrows():
        ticker = text(r.get("ticker")).upper()
        if not ticker:
            continue
        ticket = row_at(tickets, ticker)
        news_rows = news[news["ticker"] == ticker].copy() if not news.empty else pd.DataFrame()
        ev_rows = evidence[evidence["ticker"] == ticker].copy() if not evidence.empty else pd.DataFrame()
        mon_rows = monitor[monitor["ticker"] == ticker].copy() if not monitor.empty else pd.DataFrame()

        headline = f"{ticker} - {text(r.get('room_status')) or 'Research'}"
        subheadline = shorten(text(r.get("plain_english_summary")), 300)
        status = status_band(r.get("room_status"), r.get("decision_now"), r.get("risk_gate"))
        bias = action_bias(r.get("route_after_gates_clear"), r.get("option_side_after_clear"), r.get("primary_vehicle_after_clear"))
        review = urgency(
            r.get("room_status"),
            r.get("monitor_alert_count"),
            r.get("event_gate"),
            r.get("top_news_direction"),
            r.get("top_news_reliability_status"),
            r.get("top_news_calibrated_action"),
        )
        primary_route = first_nonempty(r.get("route_after_gates_clear"), r.get("primary_vehicle_after_clear"), "Research only")
        proof = first_nonempty(r.get("proof_needed"), ticket.get("required_proof_before_upgrade"))
        source_count = len([p for p in text(r.get("source_trail")).replace(" / ", ";").split(";") if p.strip()])

        rows.append({
            "ticker": ticker,
            "card_status": status,
            "review_priority": review,
            "action_bias": bias,
            "decision_quality_score": r.get("decision_quality_score", ""),
            "headline": headline,
            "subheadline": subheadline,
            "decision_now": r.get("decision_now", ""),
            "primary_route": primary_route,
            "vehicle_after_clear": r.get("primary_vehicle_after_clear", ""),
            "option_side_after_clear": r.get("option_side_after_clear", ""),
            "short_term_plan": r.get("short_term_plan", ""),
            "medium_term_plan": r.get("medium_term_plan", ""),
            "long_term_plan": r.get("long_term_plan", ""),
            "trigger_to_watch": r.get("trigger_to_watch", ""),
            "invalidation": r.get("invalidation", ""),
            "main_blocker": r.get("main_blocker", ""),
            "top_news_direction": r.get("top_news_direction", ""),
            "top_news_headline": r.get("top_news_headline", ""),
            "top_news_calibrated_score": r.get("top_news_calibrated_score", ""),
            "top_news_reliability_score": r.get("top_news_reliability_score", ""),
            "top_news_reliability_status": r.get("top_news_reliability_status", ""),
            "top_news_calibrated_action": r.get("top_news_calibrated_action", ""),
            "top_news_calibration_note": r.get("top_news_calibration_note", ""),
            "top_news_calibration_source": r.get("top_news_calibration_source", ""),
            "negative_news_vulnerability": r.get("negative_news_vulnerability", ""),
            "theme": r.get("theme", ""),
            "chain_role": r.get("chain_role", ""),
            "monitor_alert_count": r.get("monitor_alert_count", 0),
            "mapped_news_rows": len(news_rows),
            "source_count_estimate": source_count,
            "research_only": True,
            "no_broker_connection": True,
        })

        panel_specs = [
            (
                1,
                "Decision",
                "What the desk should do now",
                first_nonempty(r.get("decision_now"), "Research only. No live order."),
                first_nonempty(r.get("plain_english_summary"), subheadline),
                "ticker_decision_room.csv",
            ),
            (
                2,
                "Risk",
                "Why this can be blocked or reduced",
                first_nonempty(r.get("main_blocker"), r.get("risk_gate"), "No blocker found"),
                first_nonempty(r.get("proof_needed"), proof, r.get("no_go_conditions")),
                "final_risk_gate.csv; decision_conflict_summary.csv; desk_monitor_events.csv",
            ),
            (
                3,
                "Timing",
                "Short, medium, and long-term route",
                f"Short: {text(r.get('short_term_plan'))} | Medium: {text(r.get('medium_term_plan'))} | Long: {text(r.get('long_term_plan'))}",
                f"Trigger: {text(r.get('trigger_to_watch'))} | Invalidation: {text(r.get('invalidation'))}",
                "sector_timeframe_route.csv; conditional_action_tickets.csv",
            ),
            (
                4,
                "Options",
                "Call, put, hedge, or stock/ETF route",
                f"{primary_route}; vehicle={text(r.get('primary_vehicle_after_clear'))}; side={text(r.get('option_side_after_clear'))}",
                first_nonempty(ticket.get("option_structure_after_clear"), r.get("option_structure_after_clear"), r.get("no_go_conditions")),
                "options_playbook.csv; conditional_action_tickets.csv",
            ),
            (
                5,
                "News",
                "Headline logic and read-through",
                first_nonempty(
                    r.get("top_news_calibrated_action"),
                    r.get("top_news_direction"),
                    "No mapped news",
                ),
                shorten(
                    f"{text(r.get('top_news_headline'))} | "
                    f"Reliability: {text(r.get('top_news_reliability_status')) or 'not calibrated'} "
                    f"score={text(r.get('top_news_reliability_score')) or 'N/A'} | "
                    f"{text(r.get('top_news_logic'))} | "
                    f"{text(r.get('top_news_calibration_note')) or text(r.get('top_news_action_hint'))}",
                    760,
                ),
                "ticker_decision_room_news.csv; event_signal_reliability_adjusted_panel.csv; news_impact_targets.csv; news_supply_chain_readthrough.csv",
            ),
            (
                6,
                "Sector",
                "Theme and supply-chain context",
                f"Theme={text(r.get('theme')) or 'N/A'}; role={text(r.get('chain_role')) or 'N/A'}; sector cycle={text(r.get('sector_cycle_state')) or 'N/A'}",
                first_nonempty(r.get("supply_chain_readthrough"), "No supply-chain read-through mapped yet."),
                "theme_candidate_enrichment.csv; sector_timeframe_route.csv",
            ),
            (
                7,
                "Evidence",
                "Source-backed evidence trail",
                first_nonempty(r.get("evidence_snapshot"), "No evidence snapshot"),
                first_nonempty(join_evidence(ev_rows), r.get("conflict_snapshot"), "No detailed evidence rows found."),
                "ticker_evidence_binder.csv; ticker_evidence_summary.csv",
            ),
            (
                8,
                "Sources",
                "Files behind this card",
                f"Mapped news rows={len(news_rows)}; monitor alerts={len(mon_rows)}; source count estimate={source_count}",
                first_nonempty(r.get("source_trail"), "No source trail"),
                "ticker_decision_room.csv",
            ),
        ]

        for order, panel, title, lead, detail, sources in panel_specs:
            panel_rows.append({
                "ticker": ticker,
                "panel_order": order,
                "panel": panel,
                "panel_status": panel_status(panel, r),
                "title": title,
                "lead": shorten(lead, 420),
                "detail": shorten(detail, 760),
                "source_files": sources,
                "research_only": True,
                "no_broker_connection": True,
            })

    cards = pd.DataFrame(rows)
    panels = pd.DataFrame(panel_rows)
    if not cards.empty:
        priority_order = {
            "High review": 0,
            "Source review": 1,
            "Watch": 2,
            "Low": 3,
        }
        status_order = {
            "Blocked": 0,
            "Review first": 1,
            "Call watch": 2,
            "Hedge watch": 3,
            "Tiny paper": 4,
            "Research": 5,
        }
        cards["_priority_rank"] = cards["review_priority"].map(priority_order).fillna(9)
        cards["_status_rank"] = cards["card_status"].map(status_order).fillna(9)
        cards["_score"] = pd.to_numeric(cards["decision_quality_score"], errors="coerce").fillna(0.0)
        cards = cards.sort_values(["_priority_rank", "_status_rank", "_score", "ticker"], ascending=[True, True, False, True])
        cards = cards.drop(columns=["_priority_rank", "_status_rank", "_score"]).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(cards) else "NO_CARD_ROWS",
        "card_rows": int(len(cards)),
        "panel_rows": int(len(panels)),
        "blocked_cards": int((cards["card_status"] == "Blocked").sum()) if not cards.empty else 0,
        "high_review_cards": int((cards["review_priority"] == "High review").sum()) if not cards.empty else 0,
        "with_news_cards": int((cards["mapped_news_rows"].astype(float) > 0).sum()) if not cards.empty else 0,
        "with_calibrated_news_cards": int((cards.get("top_news_calibrated_action", pd.Series(dtype=str)).astype(str).str.len() > 0).sum()) if not cards.empty else 0,
        "outputs": {
            "cards": OUT_CARDS.name,
            "panels": OUT_PANELS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return cards, panels, state


def main() -> int:
    cards, panels, state = build_cards()
    cards.to_csv(OUT_CARDS, index=False)
    panels.to_csv(OUT_PANELS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Cards: {state.get('card_rows', 0)}",
        f"- Panels: {state.get('panel_rows', 0)}",
        f"- Blocked cards: {state.get('blocked_cards', 0)}",
        f"- High-review cards: {state.get('high_review_cards', 0)}",
        f"- Cards with mapped news: {state.get('with_news_cards', 0)}",
        f"- Cards with calibrated news: {state.get('with_calibrated_news_cards', 0)}",
        "",
        "## Decision Cards",
        "",
        df_to_markdown(cards, max_rows=80),
        "",
        "## Decision Card Panels",
        "",
        df_to_markdown(panels, max_rows=180),
        "",
        "## Product Truth",
        "",
        "These cards are research summaries only. They do not approve trades, route orders, or connect to a broker.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 152 - Ticker Decision Cards", sections)

    print(f"wrote {OUT_CARDS.name} rows={len(cards)}")
    print(f"wrote {OUT_PANELS.name} rows={len(panels)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
