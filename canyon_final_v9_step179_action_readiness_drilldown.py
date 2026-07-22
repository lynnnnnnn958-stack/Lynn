#!/usr/bin/env python3
"""
Canyon v9 Step 179 - Action Readiness Drilldown.

Research-only. No broker connection. No live orders.

Step178 produces a gate monitor. Step179 turns each ticker into a PM-readable
drilldown:
  - why the ticker is blocked or waiting
  - which source file created the signal
  - which proof would clear the next gate
  - what must not be done before gates clear

This is a source-trace and explanation layer only. It cannot trade, rebalance,
write to a ledger, or override risk.

Outputs:
  action_readiness_ticker_drilldown.csv
  action_readiness_source_trace.csv
  action_readiness_blocker_explainer.csv
  action_readiness_manual_checklist.csv
  action_readiness_drilldown_state.json
  action_readiness_drilldown_report.md
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


OUT_DRILLDOWN = ROOT / "action_readiness_ticker_drilldown.csv"
OUT_SOURCE_TRACE = ROOT / "action_readiness_source_trace.csv"
OUT_BLOCKERS = ROOT / "action_readiness_blocker_explainer.csv"
OUT_CHECKLIST = ROOT / "action_readiness_manual_checklist.csv"
OUT_STATE = ROOT / "action_readiness_drilldown_state.json"
OUT_REPORT = ROOT / "action_readiness_drilldown_report.md"


GATE_PRIORITY = [
    "risk_repair_gate",
    "monitor_gate",
    "spread_tca_gate",
    "event_proof_gate",
    "iv_greeks_gamma_gate",
    "price_trigger_gate",
    "route_gate",
]


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


def compact(value: Any, limit: int = 360) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        ticker = as_upper(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def best_option_rows(option_blockers: pd.DataFrame) -> dict[str, pd.Series]:
    if option_blockers.empty or "ticker" not in option_blockers.columns:
        return {}
    df = option_blockers.copy()
    df["_blocker_count"] = pd.to_numeric(df.get("blocker_count", 99), errors="coerce").fillna(99)
    df["_quality"] = pd.to_numeric(df.get("route_quality_score", 0), errors="coerce").fillna(0)
    df = df.sort_values(["ticker", "_blocker_count", "_quality"], ascending=[True, True, False])
    return by_ticker(df)


def best_event_rows(event_ranking: pd.DataFrame) -> dict[str, pd.Series]:
    if event_ranking.empty or "target_ticker" not in event_ranking.columns:
        return {}
    df = event_ranking.copy()
    df["_score"] = pd.to_numeric(df.get("best_event_score", 0), errors="coerce").fillna(0)
    df = df.sort_values(["target_ticker", "_score"], ascending=[True, False])
    return by_ticker(df, "target_ticker")


def gate_sort_key(row: pd.Series) -> tuple[int, int]:
    gate = as_text(row.get("gate_id"))
    status = as_upper(row.get("gate_status"))
    priority = GATE_PRIORITY.index(gate) if gate in GATE_PRIORITY else 99
    status_rank = {"BLOCKED": 0, "REVIEW": 1, "WAIT_FOR_TRIGGER": 2, "CLEAR": 3}.get(status, 9)
    return status_rank, priority


def ticker_gates(gates: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if gates.empty or "ticker" not in gates.columns:
        return pd.DataFrame()
    out = gates[gates["ticker"].astype(str).str.upper().eq(ticker)].copy()
    if out.empty:
        return out
    out["_sort"] = out.apply(gate_sort_key, axis=1)
    out = out.sort_values("_sort").drop(columns=["_sort"])
    return out


def open_gates(gates: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = ticker_gates(gates, ticker)
    if out.empty or "gate_status" not in out.columns:
        return out
    return out[~out["gate_status"].astype(str).str.upper().eq("CLEAR")].copy()


def build_why_text(row: pd.Series, open_gate_df: pd.DataFrame) -> str:
    ticker = as_upper(row.get("ticker"))
    stage = as_text(row.get("current_stage"), "NO_DATA")
    route = as_text(row.get("route_after_all_gates_clear"), "NO_DATA")
    first_gate = as_text(row.get("first_blocking_gate"), "NO_DATA")
    first_status = as_text(row.get("first_gate_status"), "NO_DATA")
    trigger = as_text(row.get("trigger_to_watch"), "NO_TRIGGER")
    if stage == "RISK_REPAIR_REQUIRED":
        return (
            f"{ticker} is not ready because the recommended repair scenario still leaves the ticker above its individual risk target. "
            f"The first gate is {first_gate} ({first_status}). Route after all gates clear would be: {route}. "
            f"Trigger is only a watch item, not permission: {trigger}."
        )
    if stage == "NON_RISK_GATES_REQUIRED":
        open_names = "; ".join(open_gate_df["gate_name"].astype(str).head(4).to_list()) if not open_gate_df.empty else first_gate
        return (
            f"{ticker} has simulated risk repair, but it is still not ready. The open gates are: {open_names}. "
            f"Do not treat the post-repair route as permission until these gates clear. Planned route after all gates clear: {route}."
        )
    if stage == "TRIGGER_WATCH":
        return (
            f"{ticker} is in trigger watch. Gates are close enough for manual review, but price must confirm: {trigger}. "
            "No action is implied before confirmation."
        )
    return (
        f"{ticker} is in {stage}. Keep it paper/research only and use the route only after manual proof checks."
    )


def next_three_checks(open_gate_df: pd.DataFrame) -> str:
    if open_gate_df.empty:
        return "1. Confirm route manually. 2. Recheck source freshness. 3. Keep research-only guardrails."
    parts = []
    for i, (_, gate) in enumerate(open_gate_df.head(3).iterrows(), 1):
        parts.append(
            f"{i}. {as_text(gate.get('gate_name'), 'Gate')}: {compact(gate.get('what_would_clear'), 170)} "
            f"(source: {as_text(gate.get('source_file'), 'NO_SOURCE')})"
        )
    return " ".join(parts)


def source_row(
    ticker: str,
    area: str,
    source_file: str,
    status: str,
    evidence_summary: str,
    why_it_matters: str,
    proof_required: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "evidence_area": area,
        "source_file": source_file,
        "source_status": status,
        "evidence_summary": compact(evidence_summary, 520),
        "why_it_matters": compact(why_it_matters, 420),
        "proof_required": compact(proof_required, 520),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def build_source_trace_for_ticker(
    ticker: str,
    readiness: pd.Series,
    repair: pd.Series | None,
    market_monitor: pd.Series | None,
    option_row: pd.Series | None,
    event_row: pd.Series | None,
    decision: pd.Series | None,
    optimizer: pd.Series | None,
    gate_candidate: pd.Series | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if repair is not None:
        rows.append(source_row(
            ticker,
            "Risk repair",
            "risk_repair_recommendation_board.csv",
            as_text(repair.get("ticker_repair_status_after_scenario"), "NO_DATA"),
            (
                f"current={as_text(repair.get('current_weight_pct'), 'NO_DATA')}%; "
                f"repair={as_text(repair.get('recommended_repair_weight_pct'), 'NO_DATA')}%; "
                f"target={as_text(repair.get('risk_target_weight_pct'), 'NO_DATA')}%; "
                f"action={as_text(repair.get('primary_repair_action'), 'NO_DATA')}"
            ),
            "Risk is the first permission gate. Options or new exposure cannot override it.",
            as_text(repair.get("next_required_proof"), "Confirm repair path and rerun risk gates."),
        ))
    else:
        rows.append(source_row(
            ticker,
            "Risk repair",
            "risk_repair_recommendation_board.csv",
            "MISSING",
            "No repair row found.",
            "Missing risk repair evidence blocks promotion.",
            "Create or rerun Step177/178 before any route review.",
        ))

    if market_monitor is not None:
        rows.append(source_row(
            ticker,
            "Price and volume monitor",
            "desk_monitor_ticker_state.csv",
            as_text(market_monitor.get("max_monitor_severity"), as_text(market_monitor.get("max_severity"), "NO_DATA")),
            (
                f"close={as_text(market_monitor.get('latest_close'), 'NO_DATA')}; "
                f"price_break={as_text(market_monitor.get('price_break_state'), 'NO_DATA')}; "
                f"volume={as_text(market_monitor.get('volume_spike_state'), 'NO_DATA')}; "
                f"volatility={as_text(market_monitor.get('volatility_regime_state'), 'NO_DATA')}; "
                f"spread={as_text(market_monitor.get('spread_status'), 'NO_DATA')}"
            ),
            "A shocky monitor state can invalidate a clean-looking signal.",
            "Monitor severity must calm or be explained with price/volume/news source evidence.",
        ))
    else:
        rows.append(source_row(ticker, "Price and volume monitor", "desk_monitor_ticker_state.csv", "MISSING", "No monitor row found.", "Missing monitor evidence blocks route promotion.", "Rerun Step119/178."))

    if option_row is not None:
        rows.append(source_row(
            ticker,
            "Option route and blockers",
            "option_unlock_blocker_attribution.csv",
            as_text(option_row.get("first_blocker"), "NO_DATA"),
            (
                f"vehicle={as_text(option_row.get('final_vehicle_decision'), 'NO_DATA')}; "
                f"side={as_text(option_row.get('final_option_side'), 'NO_DATA')}; "
                f"blockers={as_text(option_row.get('blocker_count'), 'NO_DATA')}; "
                f"no_go={as_text(option_row.get('no_go_reasons'), 'NO_DATA')}"
            ),
            "Option route is allowed only after risk, spread, IV/Greeks, event proof, and trigger checks.",
            as_text(option_row.get("required_confirmation"), "Manual spread, IV, liquidity, and trigger confirmation required."),
        ))
    else:
        rows.append(source_row(ticker, "Option route and blockers", "option_unlock_blocker_attribution.csv", "MISSING", "No option blocker row found.", "Missing option evidence blocks option route.", "Rerun Step174/178."))

    if event_row is not None:
        rows.append(source_row(
            ticker,
            "News and event read-through",
            "event_readthrough_target_ranking.csv",
            as_text(event_row.get("top_decision"), "NO_DATA"),
            (
                f"tone={as_text(event_row.get('top_tone'), 'NO_DATA')}; "
                f"score={as_text(event_row.get('best_event_score'), 'NO_DATA')}; "
                f"headline={as_text(event_row.get('top_headline'), 'NO_DATA')}"
            ),
            "A headline is not enough. The system needs a mapped causal link and event-time price reaction.",
            as_text(event_row.get("proof_required"), "Validate causal link and event-time reaction."),
        ))
    else:
        rows.append(source_row(ticker, "News and event read-through", "event_readthrough_target_ranking.csv", "NO_TARGET_ROW", "No top event row found for this ticker.", "No mapped event means news cannot upgrade the ticker.", "Use news only as context until source proof exists."))

    if decision is not None:
        decision_text = " ".join(
            as_text(decision.get(col))
            for col in [
                "route_after_gates_clear",
                "primary_vehicle_after_clear",
                "option_side_after_clear",
                "option_structure_after_clear",
            ]
        )
        current_route = as_upper(readiness.get("route_after_all_gates_clear"))
        conflict = (
            ("PUT" in current_route or "HEDGE" in current_route)
            and "CALL" in as_upper(decision_text)
        ) or (
            "UNDERLYING" in current_route
            and ("CALL" in as_upper(decision_text) or "PUT" in as_upper(decision_text))
        )
        rows.append(source_row(
            ticker,
            "Ticker decision room",
            "ticker_decision_room.csv",
            "CONFLICT_REVIEW" if conflict else as_text(decision.get("room_status"), "NO_DATA"),
            (
                f"decision={as_text(decision.get('decision_now'), 'NO_DATA')}; "
                f"route={as_text(decision.get('route_after_gates_clear'), 'NO_DATA')}; "
                f"news={as_text(decision.get('top_news_direction'), 'NO_DATA')}; "
                f"blocker={as_text(decision.get('main_blocker'), 'NO_DATA')}"
            ),
            "Decision room combines source trails into a human-readable ticker page.",
            as_text(decision.get("proof_needed"), "Open ticker decision room and verify source trail."),
        ))
    else:
        rows.append(source_row(ticker, "Ticker decision room", "ticker_decision_room.csv", "MISSING", "No decision room row found.", "Missing ticker room reduces explanation quality.", "Rerun Step151/179."))

    if optimizer is not None:
        rows.append(source_row(
            ticker,
            "Portfolio optimizer",
            "institutional_optimizer_bridge.csv",
            as_text(optimizer.get("final_optimizer_status"), "NO_DATA"),
            (
                f"final_weight={as_text(optimizer.get('final_optimizer_weight_pct'), 'NO_DATA')}%; "
                f"risk_target={as_text(optimizer.get('risk_gated_target_pct'), 'NO_DATA')}%; "
                f"cycle={as_text(optimizer.get('subsector_cycle_phase'), 'NO_DATA')}; "
                f"why_not_more={as_text(optimizer.get('why_not_more'), 'NO_DATA')}"
            ),
            "A ticker can have a signal and still be too large for portfolio constraints.",
            "Confirm active risk, sector concentration, liquidity, and correlation before size changes.",
        ))
    else:
        rows.append(source_row(ticker, "Portfolio optimizer", "institutional_optimizer_bridge.csv", "MISSING", "No optimizer row found.", "Missing optimizer evidence blocks sizing confidence.", "Rerun Step157/179."))

    if gate_candidate is not None:
        rows.append(source_row(
            ticker,
            "Gate-clear candidate ranking",
            "gate_clear_candidate_ranking.csv",
            as_text(gate_candidate.get("candidate_lane"), "NO_DATA"),
            (
                f"readiness={as_text(gate_candidate.get('readiness_score'), 'NO_DATA')}; "
                f"current_gates={as_text(gate_candidate.get('current_gates'), 'NO_DATA')}; "
                f"next_check={as_text(gate_candidate.get('next_check'), 'NO_DATA')}"
            ),
            "This shows what the ticker would become only after gates clear.",
            as_text(gate_candidate.get("next_check"), "Use only after risk/event/execution/price gates clear."),
        ))
    else:
        rows.append(source_row(ticker, "Gate-clear candidate ranking", "gate_clear_candidate_ranking.csv", "MISSING", "No gate-clear candidate row found.", "Missing upgrade path blocks route confidence.", "Rerun Step149/179."))

    return rows


def decision_route_conflict(readiness_route: Any, decision: pd.Series | None) -> tuple[str, str]:
    if decision is None:
        return "NO_DECISION_ROOM", "No ticker decision room row was found."
    route = as_upper(readiness_route)
    decision_text = " ".join(
        as_text(decision.get(col))
        for col in [
            "route_after_gates_clear",
            "primary_vehicle_after_clear",
            "option_side_after_clear",
            "option_structure_after_clear",
            "short_term_plan",
            "medium_term_plan",
            "long_term_plan",
        ]
    )
    decision_up = as_upper(decision_text)
    if ("PUT" in route or "HEDGE" in route) and "CALL" in decision_up:
        return (
            "ROUTE_CONFLICT_REVIEW",
            "Older ticker room mentions call research, but the current readiness route is put/hedge only after risk, monitor, spread, and event proof clear. Use Step178/179 as the current route authority.",
        )
    if "UNDERLYING" in route and ("CALL" in decision_up or "PUT" in decision_up):
        return (
            "ROUTE_CONFLICT_REVIEW",
            "Older ticker room mentions an option route, but the current readiness route is underlying-only after non-risk gates clear. Use Step178/179 as the current route authority.",
        )
    return "NO_ROUTE_CONFLICT", "Current readiness route and ticker room are not directly contradictory."


def build_blocker_rows(gates: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return pd.DataFrame()
    readiness_idx = by_ticker(readiness)
    rows = []
    for _, gate in gates.iterrows():
        ticker = as_upper(gate.get("ticker"))
        status = as_upper(gate.get("gate_status"))
        if status == "CLEAR":
            continue
        ready = readiness_idx.get(ticker, pd.Series(dtype=object))
        rows.append({
            "ticker": ticker,
            "current_stage": ready.get("current_stage"),
            "gate_order": gate.get("gate_order"),
            "gate_name": gate.get("gate_name"),
            "gate_status": gate.get("gate_status"),
            "gate_severity": gate.get("gate_severity"),
            "plain_english_reason": compact(
                f"{as_text(gate.get('gate_name'), 'Gate')} is {status}. Current evidence: "
                f"{as_text(gate.get('current_value'), 'NO_DATA')}. Clear condition: "
                f"{as_text(gate.get('what_would_clear'), 'NO_CLEAR_CONDITION')}",
                700,
            ),
            "source_file": gate.get("source_file"),
            "what_would_clear": gate.get("what_would_clear"),
            "do_not_do_before_clear": "No live orders, no broker path, no naked weekly options, no upgrade from one headline, and no new exposure while risk is locked.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_checklist_rows(drilldown: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    if drilldown.empty:
        return pd.DataFrame()
    rows = []
    blocker_idx = {t: df for t, df in blockers.groupby("ticker")} if not blockers.empty else {}
    for _, row in drilldown.iterrows():
        ticker = as_upper(row.get("ticker"))
        b = blocker_idx.get(ticker, pd.DataFrame())
        top_blockers = b.head(3)
        base_checks = [
            (
                "Open first source file",
                as_text(row.get("first_source_to_open"), "NO_SOURCE"),
                as_text(row.get("first_clear_condition"), "Review first clear condition."),
            ),
            (
                "Confirm risk permission",
                "risk_repair_recommendation_board.csv",
                "Risk repair must be simulated clear and manually confirmed before route promotion.",
            ),
            (
                "Confirm monitor is calm",
                "desk_monitor_ticker_state.csv",
                "Price, volume, volatility, spread, and event shock status must be calm or explained.",
            ),
            (
                "Validate event and source proof",
                "event_readthrough_target_ranking.csv; ticker_decision_room.csv",
                "Headline must have source, timestamp, mapped causal link, and event-time price reaction.",
            ),
            (
                "Check option and execution route",
                "option_unlock_blocker_attribution.csv; options_tca_no_go_audit.csv",
                "Spread/TCA, IV, Greeks, gamma, and defined-risk structure must pass. No naked weekly options.",
            ),
        ]
        for order, (name, source, condition) in enumerate(base_checks, 1):
            status = "BLOCKED" if order == 1 and not top_blockers.empty else "REVIEW"
            rows.append({
                "ticker": ticker,
                "check_order": order,
                "check_name": name,
                "check_status": status,
                "source_file": source,
                "pass_condition": condition,
                "why_this_check_exists": compact(row.get("why_blocked_plain_english"), 420),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    return pd.DataFrame(rows)


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readiness = read_csv_safe(ROOT / "action_readiness_monitor.csv")
    gates = read_csv_safe(ROOT / "action_readiness_gate_matrix.csv")
    queue = read_csv_safe(ROOT / "action_readiness_next_move_queue.csv")
    repair = read_csv_safe(ROOT / "risk_repair_recommendation_board.csv")
    market_monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    option_blockers = read_csv_safe(ROOT / "option_unlock_blocker_attribution.csv")
    event_ranking = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    decision_room = read_csv_safe(ROOT / "ticker_decision_room.csv")
    optimizer = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    gate_candidates = read_csv_safe(ROOT / "gate_clear_candidate_ranking.csv")

    if readiness.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    repair_idx = by_ticker(repair)
    market_idx = by_ticker(market_monitor)
    option_idx = best_option_rows(option_blockers)
    event_idx = best_event_rows(event_ranking)
    decision_idx = by_ticker(decision_room)
    optimizer_idx = by_ticker(optimizer)
    candidate_idx = by_ticker(gate_candidates)
    queue_idx = by_ticker(queue)

    drill_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for _, ready in readiness.iterrows():
        ticker = as_upper(ready.get("ticker"))
        open_gate_df = open_gates(gates, ticker)
        first_gate = open_gate_df.iloc[0] if not open_gate_df.empty else pd.Series(dtype=object)
        queue_row = queue_idx.get(ticker, pd.Series(dtype=object))
        repair_row = repair_idx.get(ticker)
        market_row = market_idx.get(ticker)
        option_row = option_idx.get(ticker)
        event_row = event_idx.get(ticker)
        decision_row = decision_idx.get(ticker)
        optimizer_row = optimizer_idx.get(ticker)
        candidate_row = candidate_idx.get(ticker)

        source_rows.extend(build_source_trace_for_ticker(
            ticker,
            ready,
            repair_row,
            market_row,
            option_row,
            event_row,
            decision_row,
            optimizer_row,
            candidate_row,
        ))

        risk_summary = "NO_RISK_ROW"
        if repair_row is not None:
            risk_summary = (
                f"{as_text(repair_row.get('primary_repair_action'), 'NO_ACTION')}; "
                f"current {as_text(repair_row.get('current_weight_pct'), 'NO_DATA')}% -> "
                f"repair {as_text(repair_row.get('recommended_repair_weight_pct'), 'NO_DATA')}%; "
                f"target {as_text(repair_row.get('risk_target_weight_pct'), 'NO_DATA')}%"
            )

        monitor_summary = "NO_MONITOR_ROW"
        if market_row is not None:
            monitor_summary = (
                f"severity={as_text(market_row.get('max_monitor_severity'), 'NO_DATA')}; "
                f"price={as_text(market_row.get('price_break_state'), 'NO_DATA')}; "
                f"volume={as_text(market_row.get('volume_spike_state'), 'NO_DATA')}; "
                f"vol={as_text(market_row.get('volatility_regime_state'), 'NO_DATA')}; "
                f"spread={as_text(market_row.get('spread_status'), 'NO_DATA')}"
            )

        option_summary = "NO_OPTION_ROW"
        if option_row is not None:
            option_summary = (
                f"vehicle={as_text(option_row.get('final_vehicle_decision'), 'NO_DATA')}; "
                f"side={as_text(option_row.get('final_option_side'), 'NO_DATA')}; "
                f"first_blocker={as_text(option_row.get('first_blocker'), 'NO_DATA')}; "
                f"blockers={as_text(option_row.get('blocker_count'), 'NO_DATA')}"
            )

        event_summary = "NO_EVENT_TARGET_ROW"
        if event_row is not None:
            event_summary = (
                f"{as_text(event_row.get('top_tone'), 'NO_TONE')} / "
                f"{as_text(event_row.get('top_decision'), 'NO_DECISION')}; "
                f"{compact(event_row.get('top_headline'), 180)}"
            )

        sector_summary = "NO_OPTIMIZER_ROW"
        if optimizer_row is not None:
            sector_summary = (
                f"subsector={as_text(optimizer_row.get('subsector'), 'NO_DATA')}; "
                f"cycle={as_text(optimizer_row.get('subsector_cycle_phase'), 'NO_DATA')}; "
                f"handoff={as_text(optimizer_row.get('leadership_handoff_signal'), 'NO_DATA')}; "
                f"why_not_more={compact(optimizer_row.get('why_not_more'), 170)}"
            )

        conflict_status, conflict_note = decision_route_conflict(ready.get("route_after_all_gates_clear"), decision_row)
        decision_summary = compact(decision_row.get("plain_english_summary") if decision_row is not None else "", 420)
        if conflict_status == "ROUTE_CONFLICT_REVIEW":
            decision_summary = compact(f"{conflict_note} Older ticker room summary: {decision_summary}", 520)

        drill_rows.append({
            "ticker": ticker,
            "drilldown_rank": as_text(queue_row.get("queue_rank"), as_text(ready.get("repair_rank"), "999")),
            "sector": ready.get("sector"),
            "current_stage": ready.get("current_stage"),
            "readiness_score": ready.get("readiness_score"),
            "why_blocked_plain_english": build_why_text(ready, open_gate_df),
            "first_blocking_gate": ready.get("first_blocking_gate"),
            "first_gate_status": ready.get("first_gate_status"),
            "first_source_to_open": as_text(first_gate.get("source_file"), as_text(queue_row.get("source_file_to_open"), "NO_SOURCE")),
            "first_clear_condition": as_text(first_gate.get("what_would_clear"), as_text(ready.get("nearest_clear_condition"), "NO_CLEAR_CONDITION")),
            "next_3_checks": next_three_checks(open_gate_df),
            "route_after_all_gates_clear": ready.get("route_after_all_gates_clear"),
            "option_permission_after_repair": ready.get("option_permission_after_repair"),
            "trigger_to_watch": ready.get("trigger_to_watch"),
            "risk_summary": risk_summary,
            "monitor_summary": monitor_summary,
            "option_summary": option_summary,
            "event_news_summary": event_summary,
            "sector_portfolio_summary": sector_summary,
            "decision_route_conflict_status": conflict_status,
            "decision_route_conflict_note": conflict_note,
            "decision_room_summary": decision_summary,
            "source_trace_files": (
                "action_readiness_source_trace.csv; action_readiness_gate_matrix.csv; "
                "risk_repair_recommendation_board.csv; desk_monitor_ticker_state.csv; "
                "option_unlock_blocker_attribution.csv; event_readthrough_target_ranking.csv; "
                "ticker_decision_room.csv; institutional_optimizer_bridge.csv"
            ),
            "do_not_do": "No broker connection. No live orders. No naked weekly options. Do not upgrade from one headline. Do not add exposure while risk or source gates are blocked.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    drilldown = pd.DataFrame(drill_rows)
    drilldown["drilldown_rank_num"] = pd.to_numeric(drilldown["drilldown_rank"], errors="coerce").fillna(999)
    drilldown = drilldown.sort_values(["drilldown_rank_num", "ticker"]).drop(columns=["drilldown_rank_num"]).reset_index(drop=True)
    source_trace = pd.DataFrame(source_rows)
    blockers = build_blocker_rows(gates, readiness)
    checklist = build_checklist_rows(drilldown, blockers)
    return drilldown, source_trace, blockers, checklist


def build_state(
    drilldown: pd.DataFrame,
    source_trace: pd.DataFrame,
    blockers: pd.DataFrame,
    checklist: pd.DataFrame,
) -> dict[str, Any]:
    if drilldown.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_ACTION_READINESS_DRILLDOWN_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    stage_counts = drilldown["current_stage"].value_counts().to_dict()
    first = drilldown.iloc[0]
    return {
        "date": today_str(),
        "overall_status": "ACTION_READINESS_DRILLDOWN_ACTIVE",
        "ticker_count": int(len(drilldown)),
        "source_trace_rows": int(len(source_trace)),
        "open_blocker_rows": int(len(blockers)),
        "manual_checklist_rows": int(len(checklist)),
        "risk_repair_required_count": int(stage_counts.get("RISK_REPAIR_REQUIRED", 0)),
        "non_risk_gates_required_count": int(stage_counts.get("NON_RISK_GATES_REQUIRED", 0)),
        "top_ticker": as_text(first.get("ticker")),
        "top_first_gate": as_text(first.get("first_blocking_gate")),
        "top_first_source": as_text(first.get("first_source_to_open")),
        "truth": "This is a drilldown and source trace only. It cannot trade, rebalance, or override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "ticker_drilldown": OUT_DRILLDOWN.name,
            "source_trace": OUT_SOURCE_TRACE.name,
            "blocker_explainer": OUT_BLOCKERS.name,
            "manual_checklist": OUT_CHECKLIST.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    drilldown, source_trace, blockers, checklist = build_outputs()
    state = build_state(drilldown, source_trace, blockers, checklist)

    drilldown.to_csv(OUT_DRILLDOWN, index=False)
    source_trace.to_csv(OUT_SOURCE_TRACE, index=False)
    blockers.to_csv(OUT_BLOCKERS, index=False)
    checklist.to_csv(OUT_CHECKLIST, index=False)
    write_json(OUT_STATE, state)

    drill_cols = [c for c in [
        "ticker", "current_stage", "readiness_score", "why_blocked_plain_english",
        "first_blocking_gate", "first_source_to_open", "first_clear_condition",
        "next_3_checks", "route_after_all_gates_clear",
    ] if c in drilldown.columns]
    source_cols = [c for c in [
        "ticker", "evidence_area", "source_file", "source_status",
        "evidence_summary", "proof_required",
    ] if c in source_trace.columns]
    blocker_cols = [c for c in [
        "ticker", "gate_name", "gate_status", "gate_severity",
        "plain_english_reason", "source_file", "what_would_clear",
    ] if c in blockers.columns]
    checklist_cols = [c for c in [
        "ticker", "check_order", "check_name", "check_status",
        "source_file", "pass_condition",
    ] if c in checklist.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Tickers: {state.get('ticker_count')}\n"
        f"- Source trace rows: {state.get('source_trace_rows')}\n"
        f"- Open blocker rows: {state.get('open_blocker_rows')}\n"
        f"- Manual checklist rows: {state.get('manual_checklist_rows')}\n"
        f"- Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}\n",
        "## Ticker drilldown\n" + df_to_markdown(drilldown[drill_cols] if drill_cols else drilldown, 40),
        "## Source trace\n" + df_to_markdown(source_trace[source_cols] if source_cols else source_trace, 80),
        "## Blocker explainer\n" + df_to_markdown(blockers[blocker_cols] if blocker_cols else blockers, 80),
        "## Manual checklist\n" + df_to_markdown(checklist[checklist_cols] if checklist_cols else checklist, 80),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Source trace explains evidence; it does not create permission.\n"
        "- Missing source proof blocks upgrade and must never be treated as positive evidence.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 179 - Action Readiness Drilldown", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 179 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Tickers: {state.get('ticker_count')}")
    print(f"Source trace rows: {state.get('source_trace_rows')}")
    print(f"Open blocker rows: {state.get('open_blocker_rows')}")
    print(f"Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}")
    print("Outputs:")
    for path in [OUT_DRILLDOWN, OUT_SOURCE_TRACE, OUT_BLOCKERS, OUT_CHECKLIST, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
