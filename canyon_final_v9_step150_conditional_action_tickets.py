#!/usr/bin/env python3
"""
Canyon v9 - Step 150: Conditional Action Tickets
================================================

Research-only. No broker connection. No live orders.

Step149 ranks gate-clear candidates. Step150 turns those candidates into
plain-English if/then action tickets: current stance, short/medium/long
workflow, vehicle type, option side, trigger, invalidation, source trail,
and no-go conditions.

Outputs:
  conditional_action_tickets.csv
  conditional_action_tickets_top.csv
  conditional_action_ticket_state.json
  conditional_action_ticket_report.md
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

IN_RANKING = ROOT / "gate_clear_candidate_ranking.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_EVIDENCE_SUMMARY = ROOT / "ticker_evidence_summary.csv"
IN_CONFLICT_SUMMARY = ROOT / "decision_conflict_summary.csv"
IN_GATE_SUMMARY = ROOT / "gate_upgrade_ticker_summary.csv"
IN_WORKFLOW = ROOT / "daily_workflow_queue.csv"

OUT_TICKETS = ROOT / "conditional_action_tickets.csv"
OUT_TOP = ROOT / "conditional_action_tickets_top.csv"
OUT_STATE = ROOT / "conditional_action_ticket_state.json"
OUT_REPORT = ROOT / "conditional_action_ticket_report.md"


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


def shorten(value: Any, limit: int = 540) -> str:
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


def contains(value: Any, token: str) -> bool:
    return token.upper() in upper(value)


def clean_gate_list(value: Any) -> str:
    raw = text(value)
    if not raw or upper(raw) == "NONE":
        return "No open gate in full-clear scenario"
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return "; ".join(dict.fromkeys(parts)) if parts else raw


def vehicle_for(lane: str, option_side: str, full_option: str) -> str:
    lane_u = upper(lane)
    option_u = upper(option_side or full_option)
    if "BULLISH OPTION" in lane_u or ("CALL" in option_u and "CALL" in upper(full_option)):
        return "Defined-risk call spread research"
    if "HEDGE" in lane_u or "PUT" in option_u:
        return "Put spread or protective hedge research"
    if "EQUITY" in lane_u:
        return "Tiny stock or ETF paper review"
    if "SECTOR" in lane_u or "EVENT" in lane_u:
        return "Source review before vehicle choice"
    return "No new exposure; research backlog"


def permission_now(lane: str, current_gates: str, option_permission: str) -> str:
    if current_gates and upper(current_gates) != "NONE":
        return "No action now; gates are still open."
    if contains(lane, "BULLISH OPTION") and contains(option_permission, "BLOCKED"):
        return "No option now; option permission is still blocked."
    if contains(lane, "EQUITY"):
        return "Tiny paper review only after manual checks."
    if contains(lane, "HEDGE"):
        return "Protection research only; not bullish permission."
    return "Research only; no live order."


def proof_needed(lane: str, event_gate: str, current_gates: str) -> str:
    pieces: list[str] = []
    gates = clean_gate_list(current_gates)
    if gates != "No open gate in full-clear scenario":
        pieces.append(f"Clear gates: {gates}.")
    if upper(event_gate) != "CLEAR":
        pieces.append("Resolve event/news/earnings source review.")
    if contains(lane, "OPTION"):
        pieces.append("Confirm spread/liquidity manually and use defined-risk structures only.")
    if contains(lane, "HEDGE"):
        pieces.append("Confirm the hedge is tied to portfolio risk, not a standalone bearish bet.")
    pieces.append("No broker path exists; any action remains paper/research only.")
    return " ".join(pieces)


def trigger_for(option_side: str, ranking: pd.Series, option: pd.Series) -> str:
    side = upper(option_side)
    if "CALL" in side:
        trigger = text(option.get("call_trigger"))
    elif "PUT" in side:
        trigger = text(option.get("put_trigger"))
    else:
        trigger = ""
    return trigger or text(ranking.get("price_trigger_to_watch")) or "No price trigger yet; keep in research queue."


def invalidation_for(option_side: str, option: pd.Series, ranking: pd.Series) -> str:
    invalidation = text(option.get("option_invalidation"))
    if invalidation:
        return invalidation
    if contains(option_side, "CALL"):
        return "Invalidate bullish research if price loses support or risk gate worsens."
    if contains(option_side, "PUT"):
        return "Invalidate hedge research if stress signals cool and price reclaims broken support."
    blocker = text(ranking.get("main_blocker"))
    return f"Invalidate if blocker worsens: {blocker}" if blocker else "Invalidate if upstream gate deteriorates."


def no_go_for(lane: str, ranking: pd.Series, option: pd.Series) -> str:
    conditions = [
        "No live orders.",
        "No broker connection.",
        "No naked weekly options.",
    ]
    blockers = text(ranking.get("current_gates"))
    if blockers:
        conditions.append(f"Do not upgrade while these gates remain: {blockers}.")
    option_no_go = text(option.get("no_go_conditions"))
    if option_no_go:
        conditions.append(option_no_go)
    if contains(lane, "BULLISH OPTION"):
        conditions.append("Do not treat call edge as permission until risk, event, execution, and price gates all clear.")
    if contains(lane, "HEDGE"):
        conditions.append("Do not size hedge without checking total portfolio exposure and cost.")
    return " ".join(dict.fromkeys(conditions))


def build_ticket_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranking = read_csv_safe(IN_RANKING)
    sector = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    evidence = one_by_ticker(read_csv_safe(IN_EVIDENCE_SUMMARY))
    conflict = one_by_ticker(read_csv_safe(IN_CONFLICT_SUMMARY))
    gate = one_by_ticker(read_csv_safe(IN_GATE_SUMMARY))
    workflow = one_by_ticker(read_csv_safe(IN_WORKFLOW))

    if ranking.empty or "ticker" not in ranking.columns:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_GATE_CLEAR_RANKING",
            "ticket_rows": 0,
            "top_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    ranking = ranking.copy()
    ranking["ticker"] = ranking["ticker"].astype(str).str.upper().str.strip()
    rows: list[dict[str, Any]] = []

    for _, rr in ranking.iterrows():
        ticker = text(rr.get("ticker")).upper()
        if not ticker:
            continue
        sec = row_at(sector, ticker)
        opt = row_at(options, ticker)
        ev = row_at(evidence, ticker)
        cf = row_at(conflict, ticker)
        gs = row_at(gate, ticker)
        wf = row_at(workflow, ticker)

        lane = text(rr.get("candidate_lane"))
        option_side = text(rr.get("option_side") or opt.get("option_side"))
        full_option = text(rr.get("full_clear_option_permission") or gs.get("full_clear_option_permission"))
        vehicle = vehicle_for(lane, option_side, full_option)
        current_gates = text(rr.get("current_gates") or gs.get("current_gates"))
        event_gate = text(rr.get("event_gate") or wf.get("event_gate"))
        trigger = trigger_for(option_side, rr, opt)
        invalidation = invalidation_for(option_side, opt, rr)
        no_go = no_go_for(lane, rr, opt)
        sources = "; ".join(dict.fromkeys(
            p for p in [
                text(rr.get("source_files")),
                text(sec.get("source_file")),
                text(opt.get("source_file")),
                text(ev.get("source_files")),
            ] if p
        ))

        ticket_score = (
            safe_float(rr.get("readiness_score"))
            + max(safe_float(rr.get("call_score")), safe_float(rr.get("put_score"))) * 0.10
            - safe_float(rr.get("high_conflicts")) * 3.0
            - safe_float(rr.get("gate_count")) * 1.2
        )

        rows.append({
            "ticket_rank": int(safe_float(rr.get("overall_rank"), len(rows) + 1)),
            "ticker": ticker,
            "desk_lane": lane,
            "ticket_score": round(max(0.0, min(100.0, ticket_score)), 2),
            "current_permission": permission_now(lane, current_gates, text(rr.get("option_permission_now"))),
            "primary_vehicle_after_clear": vehicle,
            "option_side_after_clear": option_side or "N/A",
            "option_structure_after_clear": text(opt.get("option_structure")) or full_option or "N/A",
            "short_term_plan": text(sec.get("short_decision")) or "Research queue only",
            "medium_term_plan": text(sec.get("medium_decision")) or "Research queue only",
            "long_term_plan": text(sec.get("long_decision")) or "Research queue only",
            "trigger_to_watch": trigger,
            "invalidation": invalidation,
            "required_proof_before_upgrade": proof_needed(lane, event_gate, current_gates),
            "main_blocker": text(rr.get("main_blocker")) or clean_gate_list(current_gates),
            "why_this_ticket_exists": shorten(text(rr.get("why_ranked_here"))),
            "next_check": text(rr.get("next_check")),
            "sector_context": shorten(text(sec.get("why")) or text(rr.get("sector_cycle_state"))),
            "event_status": event_gate or "N/A",
            "evidence_snapshot": (
                f"Evidence rows={text(ev.get('evidence_rows')) or 'N/A'}; "
                f"risk={text(ev.get('risk_rows')) or 'N/A'}; "
                f"event/news={text(ev.get('event_rows')) or 'N/A'}; "
                f"options={text(ev.get('option_rows')) or 'N/A'}."
            ),
            "conflict_snapshot": (
                f"conflicts={text(cf.get('conflict_count')) or '0'}; "
                f"high={text(cf.get('high_conflicts')) or '0'}; "
                f"top={text(cf.get('top_conflict')) or 'N/A'}."
            ),
            "no_go_conditions": shorten(no_go, 700),
            "source_trail": shorten(sources, 700),
            "research_only": True,
            "no_broker_connection": True,
        })

    tickets = pd.DataFrame(rows)
    if not tickets.empty:
        tickets = tickets.sort_values(
            ["ticket_rank", "ticket_score", "ticker"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    top = tickets.head(20).copy() if not tickets.empty else pd.DataFrame()
    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(tickets) else "NO_TICKET_ROWS",
        "ticket_rows": int(len(tickets)),
        "top_rows": int(len(top)),
        "call_research_tickets": int((tickets["primary_vehicle_after_clear"].str.contains("call", case=False, na=False)).sum()) if not tickets.empty else 0,
        "put_or_hedge_tickets": int((tickets["primary_vehicle_after_clear"].str.contains("put|hedge", case=False, regex=True, na=False)).sum()) if not tickets.empty else 0,
        "tiny_equity_tickets": int((tickets["primary_vehicle_after_clear"].str.contains("stock|ETF", case=False, regex=True, na=False)).sum()) if not tickets.empty else 0,
        "outputs": {
            "tickets": OUT_TICKETS.name,
            "top": OUT_TOP.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return tickets, top, state


def main() -> int:
    tickets, top, state = build_ticket_rows()
    tickets.to_csv(OUT_TICKETS, index=False)
    top.to_csv(OUT_TOP, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Ticket rows: {state.get('ticket_rows', 0)}",
        f"- Call research tickets: {state.get('call_research_tickets', 0)}",
        f"- Put/hedge tickets: {state.get('put_or_hedge_tickets', 0)}",
        f"- Tiny equity tickets: {state.get('tiny_equity_tickets', 0)}",
        "",
        "## Top Conditional Tickets",
        "",
        df_to_markdown(top, max_rows=30),
        "",
        "## Full Conditional Ticket Book",
        "",
        df_to_markdown(tickets, max_rows=160),
        "",
        "## Product Truth",
        "",
        "These are if/then research tickets. They do not create orders, approve option trades, or bypass risk/event/execution gates.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 150 - Conditional Action Tickets", sections)

    print(f"wrote {OUT_TICKETS.name} rows={len(tickets)}")
    print(f"wrote {OUT_TOP.name} rows={len(top)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
