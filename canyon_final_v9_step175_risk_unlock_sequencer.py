#!/usr/bin/env python3
"""
Canyon v9 Step 175 - Risk Unlock Sequencer.

Research-only. No broker connection. No live orders.

Step131 shows the current institutional risk desk. Step174 shows that option
routes are mostly locked by risk. Step175 connects those two boards:
  - Which risk component is locking each ticker?
  - How much weight must be reduced before the ticker can be reviewed again?
  - What has to clear before calls, puts, or underlying paper research can be
    considered?

This is a de-risking and explanation board, not a trading system. It cannot
create orders and it cannot upgrade any idea above the active risk gate.

Outputs:
  risk_unlock_action_board.csv
  risk_unlock_component_attribution.csv
  risk_unlock_sizing_ladder.csv
  risk_unlock_option_bridge.csv
  risk_unlock_state.json
  risk_unlock_report.md
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


OUT_ACTION = ROOT / "risk_unlock_action_board.csv"
OUT_COMPONENTS = ROOT / "risk_unlock_component_attribution.csv"
OUT_LADDER = ROOT / "risk_unlock_sizing_ladder.csv"
OUT_OPTION_BRIDGE = ROOT / "risk_unlock_option_bridge.csv"
OUT_STATE = ROOT / "risk_unlock_state.json"
OUT_REPORT = ROOT / "risk_unlock_report.md"


HARD_LOCK_WORDS = {
    "REDUCE_ONLY",
    "BLOCK",
    "BLOCKED",
    "NO_NEW",
    "NO NEW",
}
SIZE_DOWN_WORDS = {
    "SIZE_DOWN",
    "REVIEW",
    "WARNING",
    "DATA_GAP",
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


def pct_display(value: Any, digits: int = 2) -> str:
    return f"{safe_float(value):.{digits}f}%"


def by_ticker(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "ticker" not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        ticker = as_upper(row.get("ticker"))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def lock_level(status: Any) -> str:
    text = as_upper(status)
    if any(word in text for word in HARD_LOCK_WORDS):
        return "HARD_LOCK"
    if any(word in text for word in SIZE_DOWN_WORDS):
        return "SIZE_DOWN"
    if text in {"CLEAR", "OK", "PASS", "NAN", ""}:
        return "CLEAR"
    return "REVIEW"


def component_requirement(component: str, status: str, row: pd.Series, overview: dict[str, Any]) -> str:
    level = lock_level(status)
    if component == "master_risk_action":
        return (
            "Portfolio risk must move from SIZE_DOWN/REVIEW toward CLEAR. "
            f"Gross exposure target is {safe_float(overview.get('recommended_gross_exposure')) * 100:.0f}% "
            f"and annual vol target is {safe_float(overview.get('target_vol_pct')):.1f}%."
        )
    if component == "single_name_action":
        return "Single-name VaR/CVaR must fall inside budget; reduce ticker weight before new exposure."
    if component == "earnings_gap_action":
        return "Earnings/gap risk must clear; avoid adding before event risk is timestamped and priced."
    if component == "gap_down_action":
        return "Gap-down model must move away from REDUCE_ONLY; event and stop-level risk need review."
    if component == "kelly_status":
        return "Kelly sizing must support the position; current alpha confidence is not enough for full size."
    if component == "liquidity_crisis_status":
        return "Liquidity crisis exit capacity must be acceptable under stressed volume assumptions."
    if component == "sector_status":
        return "Sector/factor concentration must be below active-risk budget."
    if component == "monitor_status":
        return "Live monitor alerts must calm or be explained by source-backed evidence."
    if component == "execution_status":
        return "Spread/TCA/fill assumptions must be known; data gaps cannot unlock options."
    if component == "event_proof_status":
        return "News/event proof must have source, timestamp, ticker mapping, and price/volume confirmation."
    if level == "CLEAR":
        return "No active blocker from this component."
    return "Manual risk review required before this component can be treated as clear."


def status_rank(status: str) -> int:
    level = lock_level(status)
    return {"HARD_LOCK": 0, "SIZE_DOWN": 1, "REVIEW": 2, "CLEAR": 3}.get(level, 2)


def first_risk_lock(row: pd.Series, monitor_row: pd.Series | None, call_row: pd.Series | None) -> str:
    checks = [
        ("single_name_action", row.get("single_name_action")),
        ("earnings_gap_action", row.get("earnings_gap_action")),
        ("gap_down_action", row.get("gap_down_action")),
        ("final_risk_action", row.get("final_risk_action")),
        ("master_risk_action", row.get("master_risk_action")),
        ("sector_status", row.get("sector_status")),
        ("kelly_status", row.get("kelly_status")),
        ("liquidity_crisis_status", row.get("liquidity_crisis_status")),
    ]
    for label, status in checks:
        if lock_level(status) == "HARD_LOCK":
            return label.replace("_", " ").title()
    monitor_text = ""
    if monitor_row is not None:
        monitor_text = as_upper(monitor_row.get("max_monitor_severity") or monitor_row.get("max_severity"))
    if monitor_text in {"CRITICAL", "WARNING"}:
        return "Live Monitor"
    for label, status in checks:
        if lock_level(status) == "SIZE_DOWN":
            return label.replace("_", " ").title()
    if call_row is not None and "RISK" in as_upper(call_row.get("call_unlock_status")):
        return "Option Risk Bridge"
    return "No hard risk lock"


def build_unlock_sequence(
    ticker: str,
    row: pd.Series,
    current_weight: float,
    target_weight: float,
    overview: dict[str, Any],
    monitor_row: pd.Series | None,
    call_row: pd.Series | None,
) -> str:
    reduction = max(0.0, current_weight - target_weight)
    master_action = as_upper(row.get("master_risk_action"), "NO_DATA")
    monitor_text = ""
    if monitor_row is not None:
        monitor_text = as_upper(monitor_row.get("max_monitor_severity") or monitor_row.get("max_severity"), "NO_DATA")
    call_status = as_upper(call_row.get("call_unlock_status"), "NO_CALL_DATA") if call_row is not None else "NO_CALL_DATA"
    annual_vol = safe_float(overview.get("annual_vol_pct"))
    target_vol = safe_float(overview.get("target_vol_pct"))
    gross_target = safe_float(overview.get("recommended_gross_exposure")) * 100.0

    steps = [
        f"1. Reduce {ticker} by about {reduction:.2f} percentage points, from {current_weight:.2f}% toward <= {target_weight:.2f}%.",
        f"2. Keep total gross near {gross_target:.0f}% until master risk action is no worse than REVIEW/CLEAR.",
        f"3. Annual vol is {annual_vol:.1f}% versus {target_vol:.1f}% target; do not unlock bullish leverage while this is above target.",
        "4. Clear single-name, earnings gap, gap-down, Kelly, sector, and liquidity checks.",
    ]
    if monitor_text in {"CRITICAL", "WARNING"}:
        steps.append(f"5. Monitor is {monitor_text}; wait for price/volume/spread/news shock to calm or be explained.")
    else:
        steps.append("5. Monitor must remain calm after the reduction.")
    if "CALL_LOCKED" in call_status or "RISK" in call_status:
        steps.append("6. Calls stay locked until risk repair is complete; only then review defined-risk call spreads manually.")
    else:
        steps.append("6. Options still require manual spread, IV, event-source, and trigger checks.")
    if master_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        steps.append("7. If portfolio-level risk worsens, move this ticker to no-new-exposure rather than adding.")
    return " ".join(steps)


def unlock_status(row: pd.Series, monitor_row: pd.Series | None) -> str:
    final_action = as_upper(row.get("final_risk_action"), "NO_DATA")
    if lock_level(final_action) == "HARD_LOCK":
        return "REDUCE_ONLY_LOCKED"
    if monitor_row is not None:
        monitor_text = as_upper(monitor_row.get("max_monitor_severity") or monitor_row.get("max_severity"))
        if monitor_text == "CRITICAL":
            return "MONITOR_CRITICAL_LOCKED"
    if lock_level(final_action) == "SIZE_DOWN" or as_upper(row.get("master_risk_action")) == "SIZE_DOWN":
        return "SIZE_DOWN_LOCKED"
    if as_upper(row.get("status_bucket")) in {"HARD RISK", "RISK"}:
        return "RISK_REVIEW_LOCKED"
    return "CLEAR_MANUAL_ONLY"


def option_implication(status: str, call_row: pd.Series | None, put_row: pd.Series | None) -> str:
    call_status = as_upper(call_row.get("call_unlock_status"), "NO_CALL_DATA") if call_row is not None else "NO_CALL_DATA"
    put_status = as_upper(put_row.get("hedge_unlock_status"), "NO_PUT_DATA") if put_row is not None else "NO_PUT_DATA"
    if status in {"REDUCE_ONLY_LOCKED", "MONITOR_CRITICAL_LOCKED"}:
        return "No bullish calls. Risk reduction or hedge research only; no live orders."
    if status == "SIZE_DOWN_LOCKED":
        return "Underlying paper review only after size-down; options remain manual-only."
    if "CALL_RESEARCH_UNLOCKED" in call_status:
        return "Defined-risk call spread research only after trigger, TCA, IV, and event proof checks."
    if "HEDGE" in put_status or "PUT" in put_status:
        return "Put/hedge remains research-only protection, not a leverage route."
    return "Manual research only; no option route is automatically unlocked."


def build_action_board(
    risk_queue: pd.DataFrame,
    monitor: pd.DataFrame,
    call_unlock: pd.DataFrame,
    put_unlock: pd.DataFrame,
    optimizer: pd.DataFrame,
    overview: dict[str, Any],
) -> pd.DataFrame:
    if risk_queue.empty:
        return pd.DataFrame()

    mon_idx = by_ticker(monitor)
    call_idx = by_ticker(call_unlock)
    put_idx = by_ticker(put_unlock)
    opt_idx = by_ticker(optimizer)

    rows: list[dict[str, Any]] = []
    for _, row in risk_queue.iterrows():
        ticker = as_upper(row.get("ticker"))
        if not ticker:
            continue
        monitor_row = mon_idx.get(ticker)
        call_row = call_idx.get(ticker)
        put_row = put_idx.get(ticker)
        optimizer_row = opt_idx.get(ticker)

        current_weight = safe_float(row.get("current_weight_pct"))
        risk_target = safe_float(row.get("recommended_risk_weight_pct"))
        opt_target = safe_float(optimizer_row.get("final_optimizer_weight_pct")) if optimizer_row is not None else risk_target
        sequencer_target = max(0.0, risk_target)
        reduction_pp = max(0.0, current_weight - sequencer_target)
        reduction_pct = reduction_pp / current_weight if current_weight > 0 else 0.0

        status = unlock_status(row, monitor_row)
        first_lock = first_risk_lock(row, monitor_row, call_row)
        monitor_status = as_upper(monitor_row.get("max_monitor_severity") or monitor_row.get("max_severity"), "NO_DATA") if monitor_row is not None else "NO_DATA"
        spread_status = as_upper(monitor_row.get("spread_status"), "NO_DATA") if monitor_row is not None else "NO_DATA"
        call_status = as_text(call_row.get("call_unlock_status"), "NO_CALL_DATA") if call_row is not None else "NO_CALL_DATA"
        put_status = as_text(put_row.get("hedge_unlock_status"), "NO_PUT_DATA") if put_row is not None else "NO_PUT_DATA"

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector"),
            "current_weight_pct": round(current_weight, 4),
            "risk_target_weight_pct": round(risk_target, 4),
            "optimizer_target_weight_pct": round(opt_target, 4) if np.isfinite(opt_target) else np.nan,
            "sequencer_target_weight_pct": round(sequencer_target, 4),
            "reduction_needed_pct_points": round(reduction_pp, 4),
            "reduction_needed_pct_of_current": round(reduction_pct, 4),
            "risk_unlock_status": status,
            "first_risk_lock": first_lock,
            "final_risk_action": row.get("final_risk_action"),
            "master_risk_action": row.get("master_risk_action"),
            "single_name_action": row.get("single_name_action"),
            "earnings_gap_action": row.get("earnings_gap_action"),
            "gap_down_action": row.get("gap_down_action"),
            "kelly_status": row.get("kelly_status"),
            "sector_status": row.get("sector_status"),
            "liquidity_crisis_status": row.get("liquidity_crisis_status"),
            "monitor_status": monitor_status,
            "spread_status": spread_status,
            "call_unlock_status": call_status,
            "put_hedge_unlock_status": put_status,
            "option_implication": option_implication(status, call_row, put_row),
            "unlock_sequence": build_unlock_sequence(
                ticker,
                row,
                current_weight,
                sequencer_target,
                overview,
                monitor_row,
                call_row,
            ),
            "do_not_do": "Do not add new bullish leverage while risk_unlock_status is locked. No broker connection. No live orders.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
            "source_file": "risk_desk_ticker_action_queue.csv; call_unlock_board.csv; institutional_optimizer_bridge.csv; desk_monitor_ticker_state.csv",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {
        "REDUCE_ONLY_LOCKED": 0,
        "MONITOR_CRITICAL_LOCKED": 1,
        "SIZE_DOWN_LOCKED": 2,
        "RISK_REVIEW_LOCKED": 3,
        "CLEAR_MANUAL_ONLY": 4,
    }
    out["_rank"] = out["risk_unlock_status"].map(order).fillna(5)
    out = out.sort_values(["_rank", "reduction_needed_pct_points"], ascending=[True, False]).drop(columns=["_rank"])
    return out.reset_index(drop=True)


def build_component_attribution(
    action_board: pd.DataFrame,
    risk_queue: pd.DataFrame,
    monitor: pd.DataFrame,
    call_unlock: pd.DataFrame,
    option_blockers: pd.DataFrame,
    overview: dict[str, Any],
) -> pd.DataFrame:
    if risk_queue.empty:
        return pd.DataFrame()

    mon_idx = by_ticker(monitor)
    call_idx = by_ticker(call_unlock)
    blocker_idx = {}
    if not option_blockers.empty and "ticker" in option_blockers.columns:
        for ticker, sub in option_blockers.groupby(option_blockers["ticker"].astype(str).str.upper()):
            blocker_idx[ticker] = sub

    rows: list[dict[str, Any]] = []
    for _, row in risk_queue.iterrows():
        ticker = as_upper(row.get("ticker"))
        if not ticker:
            continue
        monitor_row = mon_idx.get(ticker)
        call_row = call_idx.get(ticker)
        monitor_status = as_upper(monitor_row.get("max_monitor_severity") or monitor_row.get("max_severity"), "NO_DATA") if monitor_row is not None else "NO_DATA"
        spread_status = as_upper(monitor_row.get("spread_status"), "NO_DATA") if monitor_row is not None else "NO_DATA"
        event_status = "CLEAR"
        if ticker in blocker_idx and "event_proof_blocker" in blocker_idx[ticker].columns:
            event_status = "BLOCKED" if bool(blocker_idx[ticker]["event_proof_blocker"].fillna(False).astype(bool).any()) else "CLEAR"
        execution_status = "DATA_GAP" if spread_status == "DATA_GAP" else "CLEAR"
        components = [
            ("master_risk_action", row.get("master_risk_action"), "institutional_risk_gate_state.json"),
            ("single_name_action", row.get("single_name_action"), "single_name_risk_budget.csv"),
            ("earnings_gap_action", row.get("earnings_gap_action"), "earnings_calendar.csv; gap_down_risk.csv"),
            ("gap_down_action", row.get("gap_down_action"), "gap_down_risk.csv"),
            ("kelly_status", row.get("kelly_status"), "kelly_position_sizing.csv"),
            ("liquidity_crisis_status", row.get("liquidity_crisis_status"), "liquidity_crisis_model.csv"),
            ("sector_status", row.get("sector_status"), "sector_active_exposure.csv"),
            ("monitor_status", monitor_status, "desk_monitor_ticker_state.csv"),
            ("execution_status", execution_status, "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv"),
            ("event_proof_status", event_status, "option_unlock_blocker_attribution.csv"),
            ("call_unlock_status", call_row.get("call_unlock_status") if call_row is not None else "NO_CALL_DATA", "call_unlock_board.csv"),
        ]
        for component, status, source in components:
            rows.append({
                "ticker": ticker,
                "sector": row.get("sector"),
                "component": component,
                "component_status": as_text(status, "NO_DATA"),
                "lock_level": lock_level(status),
                "unlock_requirement": component_requirement(component, as_text(status, "NO_DATA"), row, overview),
                "source_file": source,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_rank"] = out["lock_level"].map({"HARD_LOCK": 0, "SIZE_DOWN": 1, "REVIEW": 2, "CLEAR": 3}).fillna(2)
    return out.sort_values(["ticker", "_rank", "component"]).drop(columns=["_rank"]).reset_index(drop=True)


def build_sizing_ladder(action_board: pd.DataFrame, overview: dict[str, Any]) -> pd.DataFrame:
    if action_board.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    gross_target = safe_float(overview.get("recommended_gross_exposure")) * 100.0
    for _, row in action_board.iterrows():
        current = safe_float(row.get("current_weight_pct"))
        target = safe_float(row.get("sequencer_target_weight_pct"))
        stages = [
            ("CURRENT_BOOK", current, "Current research book exposure; not an add signal."),
            ("RISK_TARGET", target, "First acceptable target from risk/optimizer gates."),
            ("HALF_RISK_TARGET", target * 0.5, "Stricter level if monitor/event/execution remains blocked."),
            ("ZERO_NEW_EXPOSURE", 0.0, "No new exposure; watch only until risk repair is complete."),
        ]
        for step_no, (stage, weight, note) in enumerate(stages, start=1):
            reduction_pp = max(0.0, current - weight)
            if stage == "CURRENT_BOOK":
                allowed = "OBSERVE_CURRENT_RISK"
                option_permission = "NO_NEW_OPTION"
            elif stage == "RISK_TARGET":
                allowed = "RISK_REPAIR_TARGET"
                option_permission = "UNDERLYING_REVIEW_ONLY"
            elif stage == "HALF_RISK_TARGET":
                allowed = "DEFENSIVE_REVIEW"
                option_permission = "HEDGE_RESEARCH_ONLY"
            else:
                allowed = "WATCH_ONLY"
                option_permission = "NO_OPTION"
            rows.append({
                "ticker": row.get("ticker"),
                "sector": row.get("sector"),
                "step_no": step_no,
                "ladder_stage": stage,
                "target_weight_pct": round(weight, 4),
                "reduction_from_current_pct_points": round(reduction_pp, 4),
                "portfolio_gross_target_pct": round(gross_target, 2),
                "allowed_research_action": allowed,
                "option_permission": option_permission,
                "why": note,
                "risk_unlock_status": row.get("risk_unlock_status"),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_option_bridge(action_board: pd.DataFrame, call_unlock: pd.DataFrame, put_unlock: pd.DataFrame) -> pd.DataFrame:
    if action_board.empty:
        return pd.DataFrame()
    call_idx = by_ticker(call_unlock)
    put_idx = by_ticker(put_unlock)
    rows: list[dict[str, Any]] = []
    for _, row in action_board.iterrows():
        ticker = as_upper(row.get("ticker"))
        call_row = call_idx.get(ticker)
        put_row = put_idx.get(ticker)
        call_status = as_text(call_row.get("call_unlock_status"), "NO_CALL_DATA") if call_row is not None else "NO_CALL_DATA"
        put_status = as_text(put_row.get("hedge_unlock_status"), "NO_PUT_DATA") if put_row is not None else "NO_PUT_DATA"
        rows.append({
            "ticker": ticker,
            "sector": row.get("sector"),
            "risk_unlock_status": row.get("risk_unlock_status"),
            "risk_first_lock": row.get("first_risk_lock"),
            "call_unlock_status": call_status,
            "call_first_blocker": as_text(call_row.get("first_blocker"), "NO_CALL_DATA") if call_row is not None else "NO_CALL_DATA",
            "put_hedge_unlock_status": put_status,
            "put_first_blocker": as_text(put_row.get("first_blocker"), "NO_PUT_DATA") if put_row is not None else "NO_PUT_DATA",
            "option_after_risk_repair": row.get("option_implication"),
            "next_required_proof": row.get("unlock_sequence"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_summary(action_board: pd.DataFrame, overview: dict[str, Any], call_unlock: pd.DataFrame) -> dict[str, Any]:
    if action_board.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_RISK_UNLOCK_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    reduce_count = int(action_board["risk_unlock_status"].astype(str).eq("REDUCE_ONLY_LOCKED").sum())
    size_count = int(action_board["risk_unlock_status"].astype(str).eq("SIZE_DOWN_LOCKED").sum())
    monitor_count = int(action_board["monitor_status"].astype(str).str.upper().eq("CRITICAL").sum()) if "monitor_status" in action_board.columns else 0
    call_locked_by_risk = 0
    if not call_unlock.empty and "call_unlock_status" in call_unlock.columns:
        call_locked_by_risk = int(call_unlock["call_unlock_status"].astype(str).str.upper().str.contains("RISK", na=False).sum())
    total_current = float(action_board["current_weight_pct"].sum())
    total_target = float(action_board["sequencer_target_weight_pct"].sum())
    total_reduction = max(0.0, total_current - total_target)
    status = "RISK_UNLOCK_REPAIR_REQUIRED" if reduce_count or size_count or monitor_count else "RISK_UNLOCK_MANUAL_REVIEW"
    return {
        "date": today_str(),
        "overall_status": status,
        "ticker_count": int(len(action_board)),
        "reduce_only_locked_count": reduce_count,
        "size_down_locked_count": size_count,
        "monitor_critical_locked_count": monitor_count,
        "call_locked_by_risk_count": call_locked_by_risk,
        "total_current_weight_pct": round(total_current, 4),
        "total_sequencer_target_weight_pct": round(total_target, 4),
        "total_reduction_needed_pct_points": round(total_reduction, 4),
        "master_risk_action": overview.get("master_risk_action", "NO_DATA"),
        "recommended_gross_exposure_pct": round(safe_float(overview.get("recommended_gross_exposure")) * 100.0, 2),
        "annual_vol_pct": round(safe_float(overview.get("annual_vol_pct")), 4),
        "target_vol_pct": round(safe_float(overview.get("target_vol_pct")), 4),
        "truth": "Risk repair comes before option unlock. This board is research-only and cannot place orders.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "action_board": OUT_ACTION.name,
            "component_attribution": OUT_COMPONENTS.name,
            "sizing_ladder": OUT_LADDER.name,
            "option_bridge": OUT_OPTION_BRIDGE.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    risk_queue = read_csv_safe(ROOT / "risk_desk_ticker_action_queue.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    call_unlock = read_csv_safe(ROOT / "call_unlock_board.csv")
    put_unlock = read_csv_safe(ROOT / "put_hedge_unlock_board.csv")
    option_blockers = read_csv_safe(ROOT / "option_unlock_blocker_attribution.csv")
    optimizer = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    overview = read_json_safe(ROOT / "risk_desk_overview.json")

    action_board = build_action_board(risk_queue, monitor, call_unlock, put_unlock, optimizer, overview)
    components = build_component_attribution(action_board, risk_queue, monitor, call_unlock, option_blockers, overview)
    ladder = build_sizing_ladder(action_board, overview)
    option_bridge = build_option_bridge(action_board, call_unlock, put_unlock)
    state = build_summary(action_board, overview, call_unlock)

    action_board.to_csv(OUT_ACTION, index=False)
    components.to_csv(OUT_COMPONENTS, index=False)
    ladder.to_csv(OUT_LADDER, index=False)
    option_bridge.to_csv(OUT_OPTION_BRIDGE, index=False)
    write_json(OUT_STATE, state)

    top_cols = [c for c in [
        "ticker", "sector", "current_weight_pct", "sequencer_target_weight_pct",
        "reduction_needed_pct_points", "risk_unlock_status", "first_risk_lock",
        "call_unlock_status", "option_implication",
    ] if c in action_board.columns]
    comp_cols = [c for c in [
        "ticker", "component", "component_status", "lock_level", "unlock_requirement",
    ] if c in components.columns]
    ladder_cols = [c for c in [
        "ticker", "step_no", "ladder_stage", "target_weight_pct",
        "allowed_research_action", "option_permission",
    ] if c in ladder.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Master risk action: {state.get('master_risk_action')}\n"
        f"- Recommended gross exposure: {state.get('recommended_gross_exposure_pct')}%\n"
        f"- Annual vol / target vol: {state.get('annual_vol_pct')}% / {state.get('target_vol_pct')}%\n"
        f"- Total current weight: {state.get('total_current_weight_pct')}%\n"
        f"- Sequencer target weight: {state.get('total_sequencer_target_weight_pct')}%\n"
        f"- Required reduction: {state.get('total_reduction_needed_pct_points')} percentage points\n"
        f"- Calls locked by risk: {state.get('call_locked_by_risk_count')}\n",
        "## Risk unlock action board\n" + df_to_markdown(action_board[top_cols] if top_cols else action_board, 30),
        "## Component attribution\n" + df_to_markdown(components[comp_cols] if comp_cols else components, 60),
        "## Sizing ladder\n" + df_to_markdown(ladder[ladder_cols] if ladder_cols else ladder, 60),
        "## Option bridge\n" + df_to_markdown(option_bridge, 30),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Risk can reduce or block. Risk cannot upgrade.\n"
        "- Options cannot override the active risk gate.\n"
        "- Missing spread, event, or source data cannot unlock a route.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 175 - Risk Unlock Sequencer", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 175 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Tickers: {state.get('ticker_count')}")
    print(f"Required reduction: {state.get('total_reduction_needed_pct_points')} pct points")
    print("Outputs:")
    for path in [OUT_ACTION, OUT_COMPONENTS, OUT_LADDER, OUT_OPTION_BRIDGE, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
