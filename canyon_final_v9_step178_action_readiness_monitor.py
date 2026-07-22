#!/usr/bin/env python3
"""
Canyon v9 Step 178 - Action Readiness Monitor.

Research-only. No broker connection. No live orders.

Step177 says which repair path should be reviewed. Step178 turns that into a
monitorable gate system:
  - which gate is currently blocking each ticker
  - what source file created the gate
  - what condition would clear it
  - where the ticker could move after each gate clears

This step does not trade, rebalance, write to the paper ledger, or override
risk. It only creates a PM-readable readiness monitor.

Outputs:
  action_readiness_monitor.csv
  action_readiness_gate_matrix.csv
  action_readiness_next_move_queue.csv
  action_readiness_transition_map.csv
  action_readiness_state.json
  action_readiness_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_MONITOR = ROOT / "action_readiness_monitor.csv"
OUT_GATES = ROOT / "action_readiness_gate_matrix.csv"
OUT_QUEUE = ROOT / "action_readiness_next_move_queue.csv"
OUT_TRANSITIONS = ROOT / "action_readiness_transition_map.csv"
OUT_STATE = ROOT / "action_readiness_state.json"
OUT_REPORT = ROOT / "action_readiness_report.md"


GATE_ORDER = [
    "risk_repair_gate",
    "monitor_gate",
    "spread_tca_gate",
    "event_proof_gate",
    "iv_greeks_gamma_gate",
    "price_trigger_gate",
    "route_gate",
]

GATE_LABELS = {
    "risk_repair_gate": "Risk repair gate",
    "monitor_gate": "Price/volume monitor gate",
    "spread_tca_gate": "Spread and TCA gate",
    "event_proof_gate": "Event proof gate",
    "iv_greeks_gamma_gate": "IV, Greeks, and gamma gate",
    "price_trigger_gate": "Price trigger gate",
    "route_gate": "Manual route gate",
}

STATUS_RANK = {
    "BLOCKED": 0,
    "REVIEW": 1,
    "WAIT_FOR_TRIGGER": 2,
    "CLEAR": 3,
}


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


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return as_upper(value) in {"TRUE", "1", "YES", "Y"}


def by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        ticker = as_upper(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def split_flags(value: Any) -> list[str]:
    text = as_text(value)
    if not text:
        return []
    text = text.replace(",", ";")
    return [part.strip() for part in text.split(";") if part.strip()]


def text_has(value: Any, *needles: str) -> bool:
    text = as_upper(value)
    return any(needle.upper() in text for needle in needles)


def best_option_rows(option_blockers: pd.DataFrame) -> dict[str, pd.Series]:
    if option_blockers.empty or "ticker" not in option_blockers.columns:
        return {}
    df = option_blockers.copy()
    if "blocker_count" in df.columns:
        df["_blocker_count"] = pd.to_numeric(df["blocker_count"], errors="coerce").fillna(99)
    else:
        df["_blocker_count"] = 99
    if "route_quality_score" in df.columns:
        df["_quality"] = pd.to_numeric(df["route_quality_score"], errors="coerce").fillna(0)
    else:
        df["_quality"] = 0
    df = df.sort_values(["ticker", "_blocker_count", "_quality"], ascending=[True, True, False])
    return by_ticker(df)


def best_event_rows(event_ranking: pd.DataFrame) -> dict[str, pd.Series]:
    if event_ranking.empty or "target_ticker" not in event_ranking.columns:
        return {}
    df = event_ranking.copy()
    if "best_event_score" in df.columns:
        df["_score"] = pd.to_numeric(df["best_event_score"], errors="coerce").fillna(0)
    else:
        df["_score"] = 0
    df = df.sort_values(["target_ticker", "_score"], ascending=[True, False])
    return by_ticker(df, "target_ticker")


def gate_row(
    ticker: str,
    gate_id: str,
    status: str,
    severity: str,
    current_value: str,
    source_file: str,
    what_would_clear: str,
    clears_to: str,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "gate_order": GATE_ORDER.index(gate_id) + 1,
        "gate_id": gate_id,
        "gate_name": GATE_LABELS[gate_id],
        "gate_status": status,
        "gate_severity": severity,
        "current_value": current_value,
        "what_would_clear": what_would_clear,
        "clears_to": clears_to,
        "evidence": evidence,
        "source_file": source_file,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def choose_status_from_monitor(monitor: pd.Series | None, remaining: str) -> tuple[str, str, str, str]:
    if monitor is None:
        return (
            "REVIEW",
            "MEDIUM",
            "No monitor row found",
            "desk_monitor_ticker_state.csv",
        )
    severity = as_upper(monitor.get("max_monitor_severity"), as_upper(monitor.get("max_severity"), "NO_DATA"))
    price_state = as_upper(monitor.get("price_break_state"), "NO_DATA")
    volume_state = as_upper(monitor.get("volume_spike_state"), "NO_DATA")
    vol_state = as_upper(monitor.get("volatility_regime_state"), "NO_DATA")
    current = (
        f"severity={severity}; price={price_state}; volume={volume_state}; "
        f"volatility={vol_state}; events={as_text(monitor.get('event_count'), 'NO_DATA')}"
    )
    if "CRITICAL" in severity:
        return "BLOCKED", "HIGH", current, "desk_monitor_ticker_state.csv"
    if "WARNING" in severity or "monitor" in remaining.lower():
        return "REVIEW", "MEDIUM", current, "desk_monitor_ticker_state.csv"
    return "CLEAR", "LOW", current, "desk_monitor_ticker_state.csv"


def choose_spread_status(monitor: pd.Series | None, option_row: pd.Series | None, remaining: str) -> tuple[str, str, str, str]:
    spread_blocker = "spread" in remaining.lower() or "tca" in remaining.lower()
    if option_row is not None:
        spread_blocker = spread_blocker or safe_bool(option_row.get("spread_data_blocker")) or safe_bool(option_row.get("execution_blocker"))
    if monitor is None:
        return "REVIEW", "MEDIUM", "No monitor spread row found", "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv"
    spread_status = as_upper(monitor.get("spread_status"), "NO_DATA")
    spread_bps = as_text(monitor.get("spread_bps"), "NO_DATA")
    current = f"spread_status={spread_status}; spread_bps={spread_bps}; bid={as_text(monitor.get('bid'), 'NO_DATA')}; ask={as_text(monitor.get('ask'), 'NO_DATA')}"
    if "DATA_GAP" in spread_status or spread_blocker:
        return "BLOCKED", "HIGH", current, "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv"
    if "WIDE" in spread_status or "REVIEW" in spread_status:
        return "REVIEW", "MEDIUM", current, "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv"
    return "CLEAR", "LOW", current, "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv"


def choose_event_status(event_row: pd.Series | None, option_row: pd.Series | None, remaining: str) -> tuple[str, str, str, str, str]:
    event_blocker = "event" in remaining.lower() or "news" in remaining.lower()
    if option_row is not None:
        event_blocker = event_blocker or safe_bool(option_row.get("event_proof_blocker"))
    if event_row is None:
        current = "No event read-through ranking row found"
        status = "REVIEW" if not event_blocker else "BLOCKED"
        return status, "MEDIUM", current, "event_readthrough_target_ranking.csv; event_research_gate.csv", ""

    tone = as_text(event_row.get("top_tone"), "NO_DATA")
    decision = as_text(event_row.get("top_decision"), "NO_DATA")
    score = as_text(event_row.get("best_event_score"), "NO_DATA")
    headline = as_text(event_row.get("top_headline"), "NO_DATA")
    proof = as_text(event_row.get("proof_required"), "Validate causal link and event-time price reaction.")
    current = f"decision={decision}; tone={tone}; score={score}; headline={headline[:180]}"
    if event_blocker or text_has(decision, "CONTEXT", "WATCH", "VALIDATION", "NO_DIRECTIONAL"):
        return "REVIEW", "MEDIUM", current, "event_readthrough_target_ranking.csv; event_research_gate.csv", proof
    return "CLEAR", "LOW", current, "event_readthrough_target_ranking.csv; event_research_gate.csv", proof


def choose_option_greeks_status(option_row: pd.Series | None, remaining: str) -> tuple[str, str, str, str]:
    greeks_blocker = "iv" in remaining.lower() or "gamma" in remaining.lower() or "option" in remaining.lower()
    if option_row is not None:
        greeks_blocker = greeks_blocker or safe_bool(option_row.get("greeks_iv_gamma_blocker")) or safe_bool(option_row.get("high_iv_blocker")) or safe_bool(option_row.get("backtest_sanity_blocker"))
    if option_row is None:
        return "REVIEW", "MEDIUM", "No option blocker attribution row found", "option_unlock_blocker_attribution.csv; options_greeks_book_risk.csv"
    current = (
        f"vehicle={as_text(option_row.get('final_vehicle_decision'), 'NO_DATA')}; "
        f"side={as_text(option_row.get('final_option_side'), 'NO_DATA')}; "
        f"first_blocker={as_text(option_row.get('first_blocker'), 'NO_DATA')}; "
        f"blocker_count={as_text(option_row.get('blocker_count'), 'NO_DATA')}"
    )
    if greeks_blocker:
        return "REVIEW", "MEDIUM", current, "option_unlock_blocker_attribution.csv; options_greeks_book_risk.csv"
    return "CLEAR", "LOW", current, "option_unlock_blocker_attribution.csv; options_greeks_book_risk.csv"


def trigger_status(row: pd.Series, gate_row_data: pd.Series | None, monitor: pd.Series | None) -> tuple[str, str, str, str, str]:
    trigger = as_text(row.get("trigger_to_watch"))
    if not trigger and gate_row_data is not None:
        trigger = as_text(gate_row_data.get("price_trigger_to_watch"))
    if monitor is not None:
        current = (
            f"latest_close={as_text(monitor.get('latest_close'), 'NO_DATA')}; "
            f"price_break={as_text(monitor.get('price_break_state'), 'NO_DATA')}; "
            f"volume={as_text(monitor.get('volume_spike_state'), 'NO_DATA')}"
        )
    else:
        current = "No latest monitor price row found"
    if not trigger:
        return "REVIEW", "MEDIUM", current, "gate_clear_candidate_ranking.csv; desk_monitor_ticker_state.csv", "Define a clear price trigger before any manual review."
    return "WAIT_FOR_TRIGGER", "LOW", current, "gate_clear_candidate_ranking.csv; desk_monitor_ticker_state.csv", trigger


def build_ticker_gates(
    row: pd.Series,
    monitor_idx: dict[str, pd.Series],
    option_idx: dict[str, pd.Series],
    gate_idx: dict[str, pd.Series],
    event_idx: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    ticker = as_upper(row.get("ticker"))
    remaining = as_text(row.get("remaining_non_risk_blockers"))
    repair_status = as_upper(row.get("ticker_repair_status_after_scenario"))
    route = as_text(row.get("route_after_risk_repair"), "Watch only")
    option_permission = as_text(row.get("option_permission_after_repair"), "NO_DATA")
    monitor = monitor_idx.get(ticker)
    option_row = option_idx.get(ticker)
    gate_data = gate_idx.get(ticker)
    event = event_idx.get(ticker)

    if repair_status == "TICKER_RISK_REPAIRED":
        risk_gate = gate_row(
            ticker,
            "risk_repair_gate",
            "CLEAR",
            "LOW",
            f"{repair_status}; target_weight={as_text(row.get('risk_target_weight_pct'), 'NO_DATA')}%; repair_weight={as_text(row.get('recommended_repair_weight_pct'), 'NO_DATA')}%",
            "risk_repair_recommendation_board.csv; risk_repair_ticker_plan.csv",
            "Already simulated as repaired under the recommended scenario; still requires manual execution and rerun confirmation.",
            "Non-risk gates can be reviewed.",
            as_text(row.get("pm_note")),
        )
    else:
        risk_gate = gate_row(
            ticker,
            "risk_repair_gate",
            "BLOCKED",
            "HIGH",
            f"{repair_status}; target_weight={as_text(row.get('risk_target_weight_pct'), 'NO_DATA')}%; repair_weight={as_text(row.get('recommended_repair_weight_pct'), 'NO_DATA')}%",
            "risk_repair_recommendation_board.csv; risk_repair_ticker_plan.csv",
            "Reduce or hold the ticker until the recommended scenario leaves it <= individual risk target, then rerun Steps 176-178.",
            "Non-risk gates can be reviewed only after risk repair clears.",
            as_text(row.get("pm_note")),
        )

    monitor_status, monitor_sev, monitor_current, monitor_source = choose_status_from_monitor(monitor, remaining)
    spread_status, spread_sev, spread_current, spread_source = choose_spread_status(monitor, option_row, remaining)
    event_status, event_sev, event_current, event_source, proof = choose_event_status(event, option_row, remaining)
    option_status, option_sev, option_current, option_source = choose_option_greeks_status(option_row, remaining)
    trig_status, trig_sev, trig_current, trig_source, trigger = trigger_status(row, gate_data, monitor)

    gates = [
        risk_gate,
        gate_row(
            ticker,
            "monitor_gate",
            monitor_status,
            monitor_sev,
            monitor_current,
            monitor_source,
            "Price break, volume spike, volatility shift, spread widening, correlation/news shock, or risk breach must calm or be explained with source evidence.",
            "Spread/TCA and event proof can move to active review.",
            as_text(row.get("current_main_blocker")),
        ),
        gate_row(
            ticker,
            "spread_tca_gate",
            spread_status,
            spread_sev,
            spread_current,
            spread_source,
            "Real bid/ask/spread or TCA estimate must be present; DATA_GAP cannot unlock options or size.",
            "Option route can be checked for IV/Greeks and structure.",
            as_text(option_row.get("no_go_reasons")) if option_row is not None else "",
        ),
        gate_row(
            ticker,
            "event_proof_gate",
            event_status,
            event_sev,
            event_current,
            event_source,
            proof or "Validate headline source, timestamp, event-time price reaction, and causal link before promotion.",
            "News/read-through can be used as context after validation.",
            as_text(row.get("next_required_proof")),
        ),
        gate_row(
            ticker,
            "iv_greeks_gamma_gate",
            option_status,
            option_sev,
            option_current,
            option_source,
            "IV, Greeks, gamma/kill-zone, suspicious backtest, and defined-risk structure must pass manual review.",
            "Only defined-risk option research can proceed; no naked weekly options.",
            option_permission,
        ),
        gate_row(
            ticker,
            "price_trigger_gate",
            trig_status,
            trig_sev,
            trig_current,
            trig_source,
            f"Wait for trigger: {trigger}. Require price and volume confirmation before manual review.",
            "Manual route gate can be checked after trigger confirmation.",
            trigger,
        ),
    ]

    hard_block_before_route = any(g["gate_status"] == "BLOCKED" for g in gates)
    if hard_block_before_route:
        route_status = "BLOCKED"
        route_sev = "HIGH"
        route_clear = "Clear the first blocking gate before route review."
    elif any(g["gate_status"] == "REVIEW" for g in gates):
        route_status = "REVIEW"
        route_sev = "MEDIUM"
        route_clear = "Finish remaining manual checks before any paper/option research route can reopen."
    else:
        route_status = "WAIT_FOR_TRIGGER"
        route_sev = "LOW"
        route_clear = "If trigger confirms, route is ready for manual paper-only review."

    gates.append(gate_row(
        ticker,
        "route_gate",
        route_status,
        route_sev,
        f"route={route}; option_permission={option_permission}",
        "risk_repair_strategy_reopen_map.csv; conditional_action_tickets.csv; ticker_decision_room.csv",
        route_clear,
        route,
        "No broker connection. No live orders. Research-only route review.",
    ))
    return gates


def first_blocking_gate(gates: list[dict[str, Any]]) -> dict[str, Any]:
    for gate_id in GATE_ORDER:
        gate = next(g for g in gates if g["gate_id"] == gate_id)
        if gate["gate_status"] in {"BLOCKED", "REVIEW", "WAIT_FOR_TRIGGER"}:
            return gate
    return gates[-1]


def current_stage(first_gate: dict[str, Any], route: str, risk_clear: bool) -> str:
    gate_id = first_gate["gate_id"]
    status = first_gate["gate_status"]
    if not risk_clear or gate_id == "risk_repair_gate":
        return "RISK_REPAIR_REQUIRED"
    if status == "WAIT_FOR_TRIGGER":
        return "TRIGGER_WATCH"
    if gate_id in {"monitor_gate", "spread_tca_gate", "event_proof_gate", "iv_greeks_gamma_gate"}:
        return "NON_RISK_GATES_REQUIRED"
    route_up = route.upper()
    if "HEDGE" in route_up or "PUT" in route_up:
        return "HEDGE_RESEARCH_REVIEW_READY"
    if "UNDERLYING" in route_up:
        return "UNDERLYING_REVIEW_READY"
    return "WATCH_ONLY"


def next_stage_for_route(route: str, option_permission: str) -> str:
    route_up = f"{route} {option_permission}".upper()
    if "HEDGE" in route_up or "PUT" in route_up:
        return "Put or hedge research review after gates clear"
    if "CALL" in route_up:
        return "Defined-risk call research review after gates clear"
    if "UNDERLYING" in route_up:
        return "Tiny underlying paper review after gates clear"
    return "Watch only; no action route reopened"


def readiness_score(row: pd.Series, gates: list[dict[str, Any]]) -> float:
    base = min(45.0, safe_float(row.get("repair_priority_score")) / 3.0)
    gate_points = sum({"CLEAR": 10.0, "WAIT_FOR_TRIGGER": 6.0, "REVIEW": 3.0, "BLOCKED": 0.0}.get(g["gate_status"], 0.0) for g in gates)
    gate_points = gate_points / max(len(gates), 1)
    blocker_penalty = max(0, len([g for g in gates if g["gate_status"] == "BLOCKED"]) - 1) * 4.0
    return round(float(np.clip(base + gate_points - blocker_penalty, 0.0, 100.0)), 2)


def build_outputs(
    board: pd.DataFrame,
    reopen: pd.DataFrame,
    monitor_df: pd.DataFrame,
    option_blockers: pd.DataFrame,
    gate_rank: pd.DataFrame,
    event_ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if board.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    monitor_idx = by_ticker(monitor_df)
    option_idx = best_option_rows(option_blockers)
    gate_idx = by_ticker(gate_rank)
    event_idx = best_event_rows(event_ranking)
    reopen_idx = by_ticker(reopen)

    monitor_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for _, row in board.sort_values("repair_rank").iterrows():
        ticker = as_upper(row.get("ticker"))
        gates = build_ticker_gates(row, monitor_idx, option_idx, gate_idx, event_idx)
        first = first_blocking_gate(gates)
        risk_clear = next(g for g in gates if g["gate_id"] == "risk_repair_gate")["gate_status"] == "CLEAR"
        route = as_text(row.get("route_after_risk_repair"), "Watch only")
        option_permission = as_text(row.get("option_permission_after_repair"), "NO_DATA")
        stage = current_stage(first, route, risk_clear)
        reopen_row = reopen_idx.get(ticker)
        reopen_bucket = as_text(reopen_row.get("reopen_bucket"), "NO_DATA") if reopen_row is not None else "NO_DATA"

        gate_rows.extend(gates)
        score = readiness_score(row, gates)
        monitor_rows.append({
            "ticker": ticker,
            "sector": row.get("sector"),
            "repair_rank": row.get("repair_rank"),
            "current_stage": stage,
            "readiness_score": score,
            "first_blocking_gate": first["gate_name"],
            "first_gate_status": first["gate_status"],
            "first_gate_severity": first["gate_severity"],
            "nearest_clear_condition": first["what_would_clear"],
            "next_possible_stage": next_stage_for_route(route, option_permission),
            "route_after_all_gates_clear": route,
            "option_permission_after_repair": option_permission,
            "trigger_to_watch": row.get("trigger_to_watch"),
            "remaining_non_risk_blockers": row.get("remaining_non_risk_blockers"),
            "risk_repair_status": row.get("ticker_repair_status_after_scenario"),
            "reopen_bucket": reopen_bucket,
            "source_files": (
                "risk_repair_recommendation_board.csv; action_readiness_gate_matrix.csv; "
                "desk_monitor_ticker_state.csv; option_unlock_blocker_attribution.csv; "
                "event_readthrough_target_ranking.csv"
            ),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        queue_rows.append({
            "queue_rank": 0,
            "ticker": ticker,
            "repair_rank": row.get("repair_rank"),
            "priority": "HIGH" if first["gate_status"] == "BLOCKED" or stage == "RISK_REPAIR_REQUIRED" else ("MEDIUM" if first["gate_status"] == "REVIEW" else "WATCH"),
            "current_stage": stage,
            "next_move": first["gate_name"],
            "what_to_check": first["what_would_clear"],
            "source_file_to_open": first["source_file"],
            "clears_to": first["clears_to"],
            "route_after_clear": route,
            "pm_note": (
                f"{ticker}: first unblock is {first['gate_name']} ({first['gate_status']}). "
                f"Do not use option/underlying route until this clears."
            ),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        status_by_gate = {g["gate_id"]: g["gate_status"] for g in gates}
        transition_rows.append({
            "ticker": ticker,
            "current_stage": stage,
            "if_risk_repaired_then": "Review monitor/spread/event gates" if status_by_gate["risk_repair_gate"] == "BLOCKED" else "Risk repair already simulated; non-risk gates dominate.",
            "if_monitor_clears_then": "Check spread/TCA data and event proof" if status_by_gate["monitor_gate"] != "CLEAR" else "Monitor already clear enough for next gate.",
            "if_spread_tca_clears_then": "Check event proof and IV/Greeks" if status_by_gate["spread_tca_gate"] != "CLEAR" else "Spread/TCA gate already clear enough for next gate.",
            "if_event_proof_clears_then": "Use news/read-through only as validated context" if status_by_gate["event_proof_gate"] != "CLEAR" else "Event proof gate already clear enough for next gate.",
            "if_iv_greeks_gamma_clears_then": "Wait for price trigger before any option research review" if status_by_gate["iv_greeks_gamma_gate"] != "CLEAR" else "IV/Greeks gate already clear enough for trigger watch.",
            "if_price_trigger_confirms_then": next_stage_for_route(route, option_permission),
            "final_manual_review_route": route,
            "source_files": "action_readiness_gate_matrix.csv; risk_repair_strategy_reopen_map.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    queue = pd.DataFrame(queue_rows)
    if not queue.empty:
        priority_order = {"HIGH": 0, "MEDIUM": 1, "WATCH": 2}
        queue["_priority_order"] = queue["priority"].map(priority_order).fillna(9)
        queue["_repair_rank"] = pd.to_numeric(queue["repair_rank"], errors="coerce").fillna(999)
        queue["_stage_order"] = queue["current_stage"].map({
            "NON_RISK_GATES_REQUIRED": 0,
            "RISK_REPAIR_REQUIRED": 1,
            "TRIGGER_WATCH": 2,
            "HEDGE_RESEARCH_REVIEW_READY": 3,
            "UNDERLYING_REVIEW_READY": 3,
            "WATCH_ONLY": 4,
        }).fillna(9)
        queue = queue.sort_values(["_priority_order", "_repair_rank", "_stage_order", "ticker"]).drop(columns=["_priority_order", "_repair_rank", "_stage_order"]).reset_index(drop=True)
        queue["queue_rank"] = np.arange(1, len(queue) + 1)

    gates_df = pd.DataFrame(gate_rows)
    monitor_out = pd.DataFrame(monitor_rows).sort_values(["first_gate_severity", "repair_rank"], ascending=[True, True]).reset_index(drop=True)
    transitions = pd.DataFrame(transition_rows)
    return monitor_out, gates_df, queue, transitions


def build_state(monitor: pd.DataFrame, gates: pd.DataFrame, queue: pd.DataFrame) -> dict[str, Any]:
    if monitor.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_ACTION_READINESS_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    stage_counts = monitor["current_stage"].value_counts().to_dict()
    first = queue.iloc[0] if not queue.empty else pd.Series(dtype=object)
    risk_gate_pass = int(gates[(gates["gate_id"] == "risk_repair_gate") & (gates["gate_status"] == "CLEAR")].shape[0]) if not gates.empty else 0
    blocked_gate_count = int(gates["gate_status"].astype(str).eq("BLOCKED").sum()) if not gates.empty else 0
    review_gate_count = int(gates["gate_status"].astype(str).eq("REVIEW").sum()) if not gates.empty else 0
    return {
        "date": today_str(),
        "overall_status": "ACTION_READINESS_MONITOR_ACTIVE",
        "ticker_count": int(len(monitor)),
        "risk_gate_pass_count": risk_gate_pass,
        "risk_repair_required_count": int(stage_counts.get("RISK_REPAIR_REQUIRED", 0)),
        "non_risk_gates_required_count": int(stage_counts.get("NON_RISK_GATES_REQUIRED", 0)),
        "trigger_watch_count": int(stage_counts.get("TRIGGER_WATCH", 0)),
        "review_ready_count": int(stage_counts.get("HEDGE_RESEARCH_REVIEW_READY", 0) + stage_counts.get("UNDERLYING_REVIEW_READY", 0)),
        "blocked_gate_count": blocked_gate_count,
        "review_gate_count": review_gate_count,
        "top_next_move_ticker": as_text(first.get("ticker")),
        "top_next_move": as_text(first.get("next_move")),
        "truth": "This is a readiness monitor only. It cannot trade, rebalance, or override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "monitor": OUT_MONITOR.name,
            "gate_matrix": OUT_GATES.name,
            "next_move_queue": OUT_QUEUE.name,
            "transition_map": OUT_TRANSITIONS.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    board = read_csv_safe(ROOT / "risk_repair_recommendation_board.csv")
    reopen = read_csv_safe(ROOT / "risk_repair_strategy_reopen_map.csv")
    monitor_df = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    option_blockers = read_csv_safe(ROOT / "option_unlock_blocker_attribution.csv")
    gate_rank = read_csv_safe(ROOT / "gate_clear_candidate_ranking.csv")
    event_ranking = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")

    monitor, gates, queue, transitions = build_outputs(
        board,
        reopen,
        monitor_df,
        option_blockers,
        gate_rank,
        event_ranking,
    )
    state = build_state(monitor, gates, queue)

    monitor.to_csv(OUT_MONITOR, index=False)
    gates.to_csv(OUT_GATES, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    transitions.to_csv(OUT_TRANSITIONS, index=False)
    write_json(OUT_STATE, state)

    monitor_cols = [c for c in [
        "ticker", "current_stage", "readiness_score", "first_blocking_gate",
        "first_gate_status", "nearest_clear_condition",
        "next_possible_stage", "route_after_all_gates_clear",
    ] if c in monitor.columns]
    queue_cols = [c for c in [
        "queue_rank", "ticker", "priority", "current_stage", "next_move",
        "what_to_check", "source_file_to_open", "clears_to",
    ] if c in queue.columns]
    gate_cols = [c for c in [
        "ticker", "gate_order", "gate_name", "gate_status", "gate_severity",
        "current_value", "what_would_clear", "source_file",
    ] if c in gates.columns]
    transition_cols = [c for c in [
        "ticker", "current_stage", "if_risk_repaired_then",
        "if_monitor_clears_then", "if_spread_tca_clears_then",
        "if_price_trigger_confirms_then", "final_manual_review_route",
    ] if c in transitions.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Tickers monitored: {state.get('ticker_count')}\n"
        f"- Risk gate pass count: {state.get('risk_gate_pass_count')}\n"
        f"- Risk repair still required: {state.get('risk_repair_required_count')}\n"
        f"- Non-risk gates required: {state.get('non_risk_gates_required_count')}\n"
        f"- Top next move: {state.get('top_next_move_ticker')} / {state.get('top_next_move')}\n",
        "## Action readiness monitor\n" + df_to_markdown(monitor[monitor_cols] if monitor_cols else monitor, 40),
        "## Next move queue\n" + df_to_markdown(queue[queue_cols] if queue_cols else queue, 40),
        "## Gate matrix\n" + df_to_markdown(gates[gate_cols] if gate_cols else gates, 80),
        "## Transition map\n" + df_to_markdown(transitions[transition_cols] if transition_cols else transitions, 40),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Risk repair is necessary but not sufficient for options or paper review.\n"
        "- Missing spread/TCA/event/source data blocks promotion; it never upgrades a ticker.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 178 - Action Readiness Monitor", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 178 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Tickers monitored: {state.get('ticker_count')}")
    print(f"Risk gate pass count: {state.get('risk_gate_pass_count')}")
    print(f"Top next move: {state.get('top_next_move_ticker')} / {state.get('top_next_move')}")
    print("Outputs:")
    for path in [OUT_MONITOR, OUT_GATES, OUT_QUEUE, OUT_TRANSITIONS, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
