#!/usr/bin/env python3
"""
Canyon v9 Step 177 - Risk Repair Recommendation Board.

Research-only. No broker connection. No live orders.

Step176 simulates repair paths. Step177 turns the recommended repair path into
a PM-facing queue:
  - which tickers must be repaired first
  - why they are first
  - what remains blocked after risk repair
  - which strategy or option research route can be reviewed after gates clear

This step does not trade, rebalance, send orders, or modify the paper ledger.

Outputs:
  risk_repair_recommendation_board.csv
  risk_repair_priority_queue.csv
  risk_repair_strategy_reopen_map.csv
  risk_repair_pm_playbook.csv
  risk_repair_recommendation_state.json
  risk_repair_recommendation_report.md
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


OUT_BOARD = ROOT / "risk_repair_recommendation_board.csv"
OUT_QUEUE = ROOT / "risk_repair_priority_queue.csv"
OUT_REOPEN = ROOT / "risk_repair_strategy_reopen_map.csv"
OUT_PLAYBOOK = ROOT / "risk_repair_pm_playbook.csv"
OUT_STATE = ROOT / "risk_repair_recommendation_state.json"
OUT_REPORT = ROOT / "risk_repair_recommendation_report.md"


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


def by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        ticker = as_upper(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def preferred_scenario(state: dict[str, Any], summary: pd.DataFrame) -> str:
    scenario = as_text(state.get("recommended_repair_scenario"))
    if scenario:
        return scenario
    if not summary.empty and "overall_repair_status" in summary.columns:
        repaired = summary[summary["overall_repair_status"].astype(str).eq("RISK_REPAIRED_FOR_MANUAL_REVIEW")]
        if not repaired.empty:
            return as_text(repaired.sort_values("scenario_rank").iloc[0].get("scenario"))
    return "HARD_RISK_REPAIR_70"


def remaining_flags(text: Any) -> list[str]:
    raw = as_text(text)
    if not raw or raw.lower() == "none":
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def route_after_repair(option_projection: str, permission: str, gate_row: pd.Series | None, risk_repaired: bool) -> str:
    if not risk_repaired:
        return "No new exposure; ticker is still above its individual risk target in the recommended scenario"
    projection = as_upper(option_projection)
    permission = as_upper(permission)
    gate_route = as_text(gate_row.get("full_clear_route"), "") if gate_row is not None else ""
    if "DEFINED_RISK_CALL" in projection or "OPTION_RESEARCH" in permission and "CALL" in projection:
        return "Defined-risk call review after non-risk gates clear"
    if "HEDGE" in projection or "PUT_OR_HEDGE" in permission:
        return "Put / hedge research only after monitor, spread, and event proof clear"
    if "UNDERLYING" in permission or "NON_RISK_BLOCKERS" in projection:
        return "Underlying paper review only after non-risk gates clear"
    if gate_route:
        return f"{gate_route} after risk and non-risk gates clear"
    return "Watch only; no option or paper route is reopened"


def primary_repair_action(
    reduction: float,
    original_status: str,
    ticker_repair_status: str,
    option_permission: str,
) -> str:
    if reduction > 0 and "REDUCE_ONLY" in original_status:
        return "MANDATORY_REPAIR_TO_RISK_TARGET"
    if reduction > 0:
        return "SIZE_DOWN_TO_REPAIR_PATH"
    if "STILL_ABOVE" in ticker_repair_status:
        return "HOLD_NO_NEW_EXPOSURE_SECONDARY_REPAIR"
    if "HEDGE" in option_permission:
        return "RISK_REPAIRED_HEDGE_REVIEW_ONLY"
    return "RISK_REPAIRED_MANUAL_REVIEW"


def repair_priority_score(row: pd.Series, option_row: pd.Series, gate_row: pd.Series | None, optimizer_row: pd.Series | None) -> float:
    score = 0.0
    original_status = as_upper(row.get("original_risk_unlock_status"))
    if "REDUCE_ONLY" in original_status:
        score += 45.0
    elif "SIZE_DOWN" in original_status:
        score += 22.0

    reduction = safe_float(row.get("reduction_pct_points"))
    score += min(28.0, reduction * 3.5)

    first_lock = as_upper(row.get("first_risk_lock"))
    if "SINGLE" in first_lock:
        score += 10.0
    if "MONITOR" in first_lock:
        score += 6.0

    remaining = "; ".join(remaining_flags(option_row.get("remaining_non_risk_blockers")))
    if "monitor" in remaining.lower():
        score += 5.0
    if "spread" in remaining.lower():
        score += 4.0
    if "event" in remaining.lower():
        score += 4.0

    if gate_row is not None:
        score += min(8.0, safe_float(gate_row.get("readiness_score")) / 10.0)
        if "Bullish option" in as_text(gate_row.get("candidate_lane")):
            score += 3.0
    if optimizer_row is not None:
        cycle = as_upper(optimizer_row.get("subsector_cycle_phase"))
        if "DOWNCYCLE" in cycle or "LATE" in cycle or "CROWDED" in cycle:
            score += 5.0
    return round(score, 2)


def build_recommendation_board(
    scenario: str,
    ticker_plan: pd.DataFrame,
    option_projection: pd.DataFrame,
    repair_summary: pd.DataFrame,
    unlock_board: pd.DataFrame,
    gate_rank: pd.DataFrame,
    conditional: pd.DataFrame,
    decision_room: pd.DataFrame,
    thesis: pd.DataFrame,
    optimizer: pd.DataFrame,
) -> pd.DataFrame:
    plan = ticker_plan[ticker_plan["scenario"].astype(str).eq(scenario)].copy() if not ticker_plan.empty else pd.DataFrame()
    opts = option_projection[option_projection["scenario"].astype(str).eq(scenario)].copy() if not option_projection.empty else pd.DataFrame()
    if plan.empty:
        return pd.DataFrame()

    opt_idx = by_ticker(opts)
    unlock_idx = by_ticker(unlock_board)
    gate_idx = by_ticker(gate_rank)
    ticket_idx = by_ticker(conditional)
    room_idx = by_ticker(decision_room)
    thesis_idx = by_ticker(thesis)
    optimizer_idx = by_ticker(optimizer)

    summary_row = repair_summary[repair_summary["scenario"].astype(str).eq(scenario)].iloc[0] if not repair_summary.empty and "scenario" in repair_summary.columns and repair_summary["scenario"].astype(str).eq(scenario).any() else pd.Series(dtype=object)

    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        ticker = as_upper(row.get("ticker"))
        opt = opt_idx.get(ticker, pd.Series(dtype=object))
        unlock = unlock_idx.get(ticker, pd.Series(dtype=object))
        gate = gate_idx.get(ticker)
        ticket = ticket_idx.get(ticker, pd.Series(dtype=object))
        room = room_idx.get(ticker, pd.Series(dtype=object))
        thesis_row = thesis_idx.get(ticker, pd.Series(dtype=object))
        optz = optimizer_idx.get(ticker)

        reduction = safe_float(row.get("reduction_pct_points"))
        original_status = as_upper(row.get("original_risk_unlock_status"))
        ticker_repair = as_upper(row.get("ticker_repair_status"))
        risk_repaired = ticker_repair == "TICKER_RISK_REPAIRED"
        option_permission = as_text(opt.get("option_permission_projection"), "NO_DATA")
        action = primary_repair_action(reduction, original_status, ticker_repair, option_permission)
        route = route_after_repair(as_text(opt.get("option_projection")), option_permission, gate, risk_repaired)
        score = repair_priority_score(row, opt, gate, optz)
        remaining = as_text(opt.get("remaining_non_risk_blockers"), "none")

        if action == "MANDATORY_REPAIR_TO_RISK_TARGET":
            desk_lane = "Repair first"
        elif action == "HOLD_NO_NEW_EXPOSURE_SECONDARY_REPAIR":
            desk_lane = "Secondary watch"
        elif "HEDGE" in action or "HEDGE" in option_permission:
            desk_lane = "Hedge research"
        else:
            desk_lane = "Manual review"

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector"),
            "recommended_scenario": scenario,
            "repair_priority_score": score,
            "desk_lane": desk_lane,
            "primary_repair_action": action,
            "current_weight_pct": row.get("current_weight_pct"),
            "recommended_repair_weight_pct": row.get("simulated_weight_pct"),
            "risk_target_weight_pct": row.get("risk_target_weight_pct"),
            "reduction_pct_points": reduction,
            "reduction_pct_of_current": row.get("reduction_pct_of_current"),
            "original_risk_unlock_status": row.get("original_risk_unlock_status"),
            "ticker_repair_status_after_scenario": row.get("ticker_repair_status"),
            "first_risk_lock": row.get("first_risk_lock"),
            "strategy_sleeve": thesis_row.get("strategy_sleeve"),
            "strategy_posture": thesis_row.get("strategy_posture"),
            "thesis_quality_score": thesis_row.get("thesis_quality_score"),
            "top_signal": optz.get("top_signal") if optz is not None else "",
            "subsector_cycle_phase": optz.get("subsector_cycle_phase") if optz is not None else "",
            "leadership_handoff_signal": optz.get("leadership_handoff_signal") if optz is not None else "",
            "candidate_lane": gate.get("candidate_lane") if gate is not None else ticket.get("desk_lane"),
            "readiness_score": gate.get("readiness_score") if gate is not None else np.nan,
            "current_main_blocker": gate.get("main_blocker") if gate is not None else room.get("main_blocker"),
            "trigger_to_watch": gate.get("price_trigger_to_watch") if gate is not None else ticket.get("trigger_to_watch"),
            "route_after_risk_repair": route,
            "option_projection_after_repair": opt.get("option_projection"),
            "option_permission_after_repair": option_permission,
            "remaining_non_risk_blockers": remaining,
            "next_required_proof": opt.get("required_next_proof") or unlock.get("unlock_sequence") or ticket.get("required_proof_before_upgrade"),
            "pm_note": (
                f"{ticker}: {action}. Cut/hold target is {safe_float(row.get('simulated_weight_pct')):.2f}% "
                f"under {scenario}. After risk repair: {route}. Remaining blockers: {remaining}."
            ),
            "scenario_gross_pct": summary_row.get("gross_exposure_pct"),
            "scenario_annual_vol_pct": summary_row.get("annual_vol_pct"),
            "scenario_repair_status": summary_row.get("overall_repair_status"),
            "source_files": "risk_repair_ticker_plan.csv; risk_repair_option_projection.csv; gate_clear_candidate_ranking.csv; institutional_strategy_thesis_board.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["repair_priority_score", "reduction_pct_points"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "repair_rank", np.arange(1, len(out) + 1))
    return out


def build_priority_queue(board: pd.DataFrame, state: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    scenario = as_text(state.get("recommended_repair_scenario"), board["recommended_scenario"].iloc[0])
    scenario_row = summary[summary["scenario"].astype(str).eq(scenario)].iloc[0] if not summary.empty and summary["scenario"].astype(str).eq(scenario).any() else pd.Series(dtype=object)

    hard = board[board["primary_repair_action"].astype(str).eq("MANDATORY_REPAIR_TO_RISK_TARGET")]
    secondary = board[board["primary_repair_action"].astype(str).eq("HOLD_NO_NEW_EXPOSURE_SECONDARY_REPAIR")]
    hedge = board[board["option_permission_after_repair"].astype(str).str.contains("HEDGE|PUT", case=False, na=False)]
    underlying = board[board["option_permission_after_repair"].astype(str).str.contains("UNDERLYING", case=False, na=False)]
    nonrisk_blockers = sorted({
        item
        for text in board["remaining_non_risk_blockers"].fillna("")
        for item in remaining_flags(text)
        if item.lower() != "none"
    })

    rows = [
        {
            "queue_rank": 1,
            "station": "Risk repair first",
            "priority": "P0",
            "status": "REQUIRED_BEFORE_NEW_IDEAS",
            "ticker_count": int(len(hard)),
            "tickers": ", ".join(hard["ticker"].tolist()) if not hard.empty else "none",
            "what_to_do": "Bring hard-risk tickers to their recommended repair weights in the research model before reviewing new upside ideas.",
            "why_it_matters": f"{scenario} moves gross to {scenario_row.get('gross_exposure_pct', 'NO_DATA')}% and vol to {scenario_row.get('annual_vol_pct', 'NO_DATA')}%.",
            "done_when": "Scenario status remains RISK_REPAIRED_FOR_MANUAL_REVIEW after rerun.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        },
        {
            "queue_rank": 2,
            "station": "No-new-exposure watchlist",
            "priority": "P1",
            "status": "HOLD_OR_REVIEW_ONLY",
            "ticker_count": int(len(secondary)),
            "tickers": ", ".join(secondary["ticker"].tolist()) if not secondary.empty else "none",
            "what_to_do": "Do not add exposure to secondary size-down names; decide later whether to run full ticker-risk-target repair.",
            "why_it_matters": "The recommended path repairs portfolio risk but not every ticker's individual risk target.",
            "done_when": "Each secondary ticker either clears monitor/event/spread checks or moves to TICKER_RISK_TARGET scenario.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        },
        {
            "queue_rank": 3,
            "station": "Non-risk blocker cleanup",
            "priority": "P1",
            "status": "SOURCE_AND_MONITOR_REVIEW",
            "ticker_count": int(len(board)),
            "tickers": ", ".join(board["ticker"].head(8).tolist()) + ("..." if len(board) > 8 else ""),
            "what_to_do": "Clear monitor, spread/TCA, event proof, IV/Greeks, and source reliability blockers before reopening any route.",
            "why_it_matters": "Risk repair alone does not make a call, put, or paper trade valid.",
            "done_when": "Remaining blocker column is none or documented with source-backed manual proof.",
            "blocker_types": "; ".join(nonrisk_blockers) if nonrisk_blockers else "none",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        },
        {
            "queue_rank": 4,
            "station": "Route reopen review",
            "priority": "P2",
            "status": "MANUAL_REVIEW_ONLY",
            "ticker_count": int(len(hedge) + len(underlying)),
            "tickers": ", ".join(pd.concat([hedge["ticker"], underlying["ticker"]]).drop_duplicates().tolist()) if (not hedge.empty or not underlying.empty) else "none",
            "what_to_do": "Review hedge-only and underlying-only routes after risk repair. Do not treat this as an automatic trade list.",
            "why_it_matters": "The board currently projects hedge/underlying review, not clean bullish call permission.",
            "done_when": "Route has clean risk, event proof, spread/TCA, trigger, and monitor state.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        },
    ]
    return pd.DataFrame(rows)


def build_reopen_map(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in board.iterrows():
        permission = as_upper(row.get("option_permission_after_repair"))
        remaining = remaining_flags(row.get("remaining_non_risk_blockers"))
        risk_repaired = as_upper(row.get("ticker_repair_status_after_scenario")) == "TICKER_RISK_REPAIRED"
        if not risk_repaired:
            reopen_bucket = "WATCH_ONLY_RISK_STILL_LOCKED"
        elif "HEDGE" in permission or "PUT" in permission:
            reopen_bucket = "HEDGE_RESEARCH_AFTER_NON_RISK_GATES"
        elif "UNDERLYING" in permission:
            reopen_bucket = "UNDERLYING_REVIEW_AFTER_NON_RISK_GATES"
        elif "OPTION" in permission and "NO_NEW" not in permission:
            reopen_bucket = "DEFINED_RISK_OPTION_REVIEW_AFTER_NON_RISK_GATES"
        else:
            reopen_bucket = "WATCH_ONLY"
        rows.append({
            "ticker": row.get("ticker"),
            "repair_rank": row.get("repair_rank"),
            "recommended_scenario": row.get("recommended_scenario"),
            "risk_repair_status": row.get("ticker_repair_status_after_scenario"),
            "reopen_bucket": reopen_bucket,
            "route_after_risk_repair": row.get("route_after_risk_repair"),
            "strategy_sleeve": row.get("strategy_sleeve"),
            "candidate_lane": row.get("candidate_lane"),
            "option_permission_after_repair": row.get("option_permission_after_repair"),
            "remaining_gate_count": len(remaining),
            "remaining_gates": "; ".join(remaining) if remaining else "none",
            "trigger_to_watch": row.get("trigger_to_watch"),
            "source_files": row.get("source_files"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["reopen_bucket", "repair_rank"]).reset_index(drop=True)


def build_playbook(board: pd.DataFrame, queue: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    hard = board[board["primary_repair_action"].astype(str).eq("MANDATORY_REPAIR_TO_RISK_TARGET")]
    scenario = as_text(state.get("recommended_repair_scenario"), board["recommended_scenario"].iloc[0])
    rows = [
        {
            "step_no": 1,
            "playbook_step": "Confirm repair scenario",
            "decision": scenario,
            "what_to_check": "Use Step176 scenario summary. Confirm gross, vol, VaR, and CVaR are inside limits.",
            "do_not_do": "Do not treat the scenario as an order ticket.",
            "source_file": "risk_repair_scenario_summary.csv",
        },
        {
            "step_no": 2,
            "playbook_step": "Repair hard-risk names first",
            "decision": ", ".join(hard["ticker"].tolist()) if not hard.empty else "none",
            "what_to_check": "Hard-risk names should be at or below their recommended repair weights before upside review.",
            "do_not_do": "Do not add exposure to hard-risk names while original status is REDUCE_ONLY_LOCKED.",
            "source_file": "risk_repair_recommendation_board.csv",
        },
        {
            "step_no": 3,
            "playbook_step": "Re-run risk and monitor",
            "decision": "rerun 173 174 175 176 177 172",
            "what_to_check": "Risk repaired status should persist; monitor/spread/event blockers should be reviewed separately.",
            "do_not_do": "Do not unlock options only because portfolio-level risk improved.",
            "source_file": "run_daily_all_log.csv",
        },
        {
            "step_no": 4,
            "playbook_step": "Review route reopen map",
            "decision": "hedge/underlying/manual only",
            "what_to_check": "Only routes with clean non-risk gates can be moved to paper review.",
            "do_not_do": "Do not use naked weekly calls; do not bypass event proof or spread/TCA.",
            "source_file": "risk_repair_strategy_reopen_map.csv",
        },
        {
            "step_no": 5,
            "playbook_step": "Record decision",
            "decision": "paper research log only",
            "what_to_check": "Document why a ticker stays blocked, moves to hedge research, or becomes underlying paper review.",
            "do_not_do": "No broker connection. No live orders. No automatic paper ledger edits from this step.",
            "source_file": "risk_repair_pm_playbook.csv",
        },
    ]
    out = pd.DataFrame(rows)
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def build_state(board: pd.DataFrame, queue: pd.DataFrame, reopen: pd.DataFrame, scenario: str, repair_state: dict[str, Any]) -> dict[str, Any]:
    if board.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_REPAIR_RECOMMENDATION_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    hard_count = int(board["primary_repair_action"].astype(str).eq("MANDATORY_REPAIR_TO_RISK_TARGET").sum())
    secondary_count = int(board["primary_repair_action"].astype(str).eq("HOLD_NO_NEW_EXPOSURE_SECONDARY_REPAIR").sum())
    hedge_count = int(reopen["reopen_bucket"].astype(str).str.contains("HEDGE", na=False).sum()) if not reopen.empty else 0
    underlying_count = int(reopen["reopen_bucket"].astype(str).str.contains("UNDERLYING", na=False).sum()) if not reopen.empty else 0
    first = board.sort_values("repair_rank").iloc[0]
    return {
        "date": today_str(),
        "overall_status": "RISK_REPAIR_RECOMMENDATION_ACTIVE",
        "recommended_scenario": scenario,
        "recommended_scenario_status": repair_state.get("recommended_repair_status", "NO_DATA"),
        "recommended_scenario_gross_pct": repair_state.get("recommended_scenario_gross_pct", "NO_DATA"),
        "recommended_scenario_annual_vol_pct": repair_state.get("recommended_scenario_annual_vol_pct", "NO_DATA"),
        "ticker_count": int(len(board)),
        "hard_repair_count": hard_count,
        "secondary_watch_count": secondary_count,
        "hedge_research_after_repair_count": hedge_count,
        "underlying_review_after_repair_count": underlying_count,
        "top_repair_ticker": as_text(first.get("ticker")),
        "top_repair_action": as_text(first.get("primary_repair_action")),
        "queue_steps": int(len(queue)),
        "truth": "This is a repair recommendation board only. It cannot trade, rebalance, or override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "board": OUT_BOARD.name,
            "queue": OUT_QUEUE.name,
            "reopen_map": OUT_REOPEN.name,
            "playbook": OUT_PLAYBOOK.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    repair_state = read_json_safe(ROOT / "risk_repair_state.json")
    repair_summary = read_csv_safe(ROOT / "risk_repair_scenario_summary.csv")
    ticker_plan = read_csv_safe(ROOT / "risk_repair_ticker_plan.csv")
    option_projection = read_csv_safe(ROOT / "risk_repair_option_projection.csv")
    unlock_board = read_csv_safe(ROOT / "risk_unlock_action_board.csv")
    gate_rank = read_csv_safe(ROOT / "gate_clear_candidate_ranking.csv")
    conditional = read_csv_safe(ROOT / "conditional_action_tickets.csv")
    decision_room = read_csv_safe(ROOT / "ticker_decision_room.csv")
    thesis = read_csv_safe(ROOT / "institutional_strategy_thesis_board.csv")
    optimizer = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")

    scenario = preferred_scenario(repair_state, repair_summary)
    board = build_recommendation_board(
        scenario,
        ticker_plan,
        option_projection,
        repair_summary,
        unlock_board,
        gate_rank,
        conditional,
        decision_room,
        thesis,
        optimizer,
    )
    queue = build_priority_queue(board, repair_state, repair_summary)
    reopen = build_reopen_map(board)
    playbook = build_playbook(board, queue, repair_state)
    state = build_state(board, queue, reopen, scenario, repair_state)

    board.to_csv(OUT_BOARD, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    reopen.to_csv(OUT_REOPEN, index=False)
    playbook.to_csv(OUT_PLAYBOOK, index=False)
    write_json(OUT_STATE, state)

    board_cols = [c for c in [
        "repair_rank", "ticker", "desk_lane", "primary_repair_action",
        "current_weight_pct", "recommended_repair_weight_pct",
        "reduction_pct_points", "route_after_risk_repair",
        "remaining_non_risk_blockers", "pm_note",
    ] if c in board.columns]
    queue_cols = [c for c in [
        "queue_rank", "station", "priority", "status", "ticker_count",
        "tickers", "what_to_do", "why_it_matters", "done_when",
    ] if c in queue.columns]
    reopen_cols = [c for c in [
        "ticker", "reopen_bucket", "route_after_risk_repair",
        "option_permission_after_repair", "remaining_gates", "trigger_to_watch",
    ] if c in reopen.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Recommended scenario: {state.get('recommended_scenario')} ({state.get('recommended_scenario_status')})\n"
        f"- Gross / vol after scenario: {state.get('recommended_scenario_gross_pct')}% / {state.get('recommended_scenario_annual_vol_pct')}%\n"
        f"- Hard repair count: {state.get('hard_repair_count')}\n"
        f"- Top repair ticker: {state.get('top_repair_ticker')} ({state.get('top_repair_action')})\n",
        "## Repair recommendation board\n" + df_to_markdown(board[board_cols] if board_cols else board, 40),
        "## Priority queue\n" + df_to_markdown(queue[queue_cols] if queue_cols else queue, 20),
        "## Strategy reopen map\n" + df_to_markdown(reopen[reopen_cols] if reopen_cols else reopen, 40),
        "## PM playbook\n" + df_to_markdown(playbook, 20),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- This step recommends a repair sequence; it does not alter holdings.\n"
        "- Risk repair does not automatically unlock options. Non-risk gates still matter.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 177 - Risk Repair Recommendation Board", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 177 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Scenario: {state.get('recommended_scenario')}")
    print(f"Hard repair count: {state.get('hard_repair_count')}")
    print(f"Top repair ticker: {state.get('top_repair_ticker')}")
    print("Outputs:")
    for path in [OUT_BOARD, OUT_QUEUE, OUT_REOPEN, OUT_PLAYBOOK, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
