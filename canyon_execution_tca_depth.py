#!/usr/bin/env python3
"""
Canyon v9 Execution / TCA Depth Desk.

This is a workstream-deepening module, not another numbered trading step.

Purpose:
  - combine execution playbook, cost stress, monitor shocks, risk gates, and
    options TCA no-go evidence into one PM-readable decision board
  - answer whether a ticker is executable as paper research, needs manual
    spread/liquidity proof, should be reduced only, or should not be touched
  - keep execution strictly research-only with no broker connection and no live
    order path
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


OUT_BOARD = ROOT / "execution_tca_decision_board.csv"
OUT_CARDS = ROOT / "execution_tca_ticker_cards.csv"
OUT_SOURCE_GUIDE = ROOT / "execution_tca_source_guide.csv"
OUT_STATE = ROOT / "execution_tca_state.json"
OUT_REPORT = ROOT / "execution_tca_report.md"


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def clean_ticker(value: Any) -> str:
    return clean_text(value).upper()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def first_row(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty or "ticker" not in df.columns:
        return pd.Series(dtype=object)
    mask = df["ticker"].astype(str).str.upper().eq(ticker)
    if not mask.any():
        return pd.Series(dtype=object)
    return df.loc[mask].iloc[0]


def status_rank(label: str) -> int:
    raw = clean_text(label).upper()
    if any(x in raw for x in ["BLOCK", "DATA_GAP", "NO_GO"]):
        return 4
    if any(x in raw for x in ["REDUCE_ONLY", "SIZE_DOWN", "CRITICAL"]):
        return 3
    if any(x in raw for x in ["REVIEW", "WARNING", "WAIT"]):
        return 2
    if any(x in raw for x in ["CLEAR", "PASS", "OK"]):
        return 0
    return 1


def score_execution(
    risk_action: str,
    playbook_status: str,
    cost_status: str,
    monitor_severity: str,
    spread_status: str,
    base_cost: float,
    stress_cost: float,
    fill_rate: float,
    option_no_go_count: float,
) -> float:
    score = 100.0
    # This is an execution-readiness score, not permission. A name can score
    # above zero and still be blocked if risk or spread proof is not clean.
    risk_raw = clean_text(risk_action).upper()
    play_raw = clean_text(playbook_status).upper()
    cost_raw = clean_text(cost_status).upper()
    monitor_raw = clean_text(monitor_severity).upper()
    spread_raw = clean_text(spread_status).upper()
    if "REDUCE_ONLY" in risk_raw:
        score -= 35.0
    elif "SIZE_DOWN" in risk_raw:
        score -= 18.0
    elif "BLOCK" in risk_raw:
        score -= 40.0
    else:
        score -= status_rank(risk_action) * 5.0
    if "DATA_GAP" in play_raw:
        score -= 24.0
    else:
        score -= status_rank(playbook_status) * 6.0
    if "DATA_GAP" in cost_raw:
        score -= 18.0
    else:
        score -= status_rank(cost_status) * 5.0
    if "CRITICAL" in monitor_raw:
        score -= 16.0
    elif "WARNING" in monitor_raw:
        score -= 8.0
    else:
        score -= status_rank(monitor_severity) * 3.0
    if "DATA_GAP" in spread_raw:
        score -= 22.0
    else:
        score -= status_rank(spread_status) * 5.0
    if np.isfinite(base_cost):
        score -= max(0.0, base_cost - 15.0) * 0.28
    else:
        score -= 8.0
    if np.isfinite(stress_cost):
        score -= max(0.0, stress_cost - 30.0) * 0.18
    else:
        score -= 8.0
    if np.isfinite(fill_rate):
        score -= max(0.0, 95.0 - fill_rate) * 0.35
    else:
        score -= 12.0
    if np.isfinite(option_no_go_count):
        score -= min(option_no_go_count, 8.0) * 1.5
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def decide_verdict(
    risk_action: str,
    playbook_status: str,
    cost_status: str,
    monitor_severity: str,
    spread_status: str,
    option_decision: str,
    score: float,
) -> tuple[str, str]:
    risk = clean_text(risk_action).upper()
    play = clean_text(playbook_status).upper()
    cost = clean_text(cost_status).upper()
    monitor = clean_text(monitor_severity).upper()
    spread = clean_text(spread_status).upper()
    option = clean_text(option_decision).upper()

    if "REDUCE_ONLY" in risk:
        return "RISK_REDUCTION_ONLY", "Risk gate dominates. No new exposure; only reduction or hedge research can be reviewed."
    if "DATA_GAP" in {play, cost, spread} or "DATA_GAP" in spread:
        return "MANUAL_SPREAD_LIQUIDITY_CHECK", "Manual bid/ask, spread, ADV, and fill-risk proof are required before any paper route."
    if "CRITICAL" in monitor:
        return "WAIT_FOR_MONITOR_TO_CALM", "Monitor shock is active. Wait or explain price/volume/spread/news/correlation shock first."
    if "BLOCK" in play or "BLOCK" in cost:
        return "NO_NEW_PAPER_ACTION", "Execution layer blocks new exposure until cost, fill, and gate proof improve."
    if "PUT_OR_HEDGE" in option:
        return "PUT_OR_HEDGE_RESEARCH_ONLY", "Only defined-risk put or hedge research may be reviewed after manual execution checks."
    if score < 55:
        return "NO_NEW_PAPER_ACTION", "Execution score is too low for new paper action."
    if score < 70 or "REVIEW" in play or "REVIEW" in cost:
        return "TINY_PAPER_AFTER_MANUAL_CHECKS", "Only tiny paper sizing after manual spread, liquidity, and source checks."
    return "EXECUTION_RESEARCH_READY", "Research execution assumptions are acceptable, but still no broker and no live order path."


def build_board() -> pd.DataFrame:
    cost = read_csv_safe(ROOT / "execution_cost_model.csv")
    trade = read_csv_safe(ROOT / "execution_trade_plan.csv")
    risk = read_csv_safe(ROOT / "final_risk_gate.csv")
    readiness = read_csv_safe(ROOT / "action_readiness_monitor.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    options = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    option_audit = read_csv_safe(ROOT / "options_tca_no_go_audit.csv")
    slices = read_csv_safe(ROOT / "execution_slicing_schedule.csv")
    bridge = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")

    ticker_source = []
    for df in [trade, cost, risk, readiness, monitor, bridge]:
        if not df.empty and "ticker" in df.columns:
            ticker_source.extend(df["ticker"].dropna().astype(str).str.upper().tolist())
    tickers = sorted(dict.fromkeys(ticker_source))
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        c = first_row(cost, ticker)
        t = first_row(trade, ticker)
        r = first_row(risk, ticker)
        rd = first_row(readiness, ticker)
        m = first_row(monitor, ticker)
        b = first_row(bridge, ticker)

        opt_rows = pd.DataFrame()
        if not options.empty and "ticker" in options.columns:
            opt_rows = options[options["ticker"].astype(str).str.upper().eq(ticker)].copy()
        if not opt_rows.empty and "route_quality_score" in opt_rows.columns:
            opt_rows["_sort"] = pd.to_numeric(opt_rows["route_quality_score"], errors="coerce").fillna(-999)
            opt = opt_rows.sort_values("_sort", ascending=False).iloc[0]
        elif not opt_rows.empty:
            opt = opt_rows.iloc[0]
        else:
            opt = pd.Series(dtype=object)

        audit_rows = pd.DataFrame()
        if not option_audit.empty and "ticker" in option_audit.columns:
            audit_rows = option_audit[option_audit["ticker"].astype(str).str.upper().eq(ticker)].copy()
        option_no_go = int(audit_rows["status"].astype(str).str.upper().isin(["BLOCK_OR_REVIEW", "BLOCK", "REVIEW"]).sum()) if not audit_rows.empty and "status" in audit_rows.columns else 0

        slice_rows = pd.DataFrame()
        if not slices.empty and "ticker" in slices.columns:
            slice_rows = slices[slices["ticker"].astype(str).str.upper().eq(ticker)].copy()
        first_slice = clean_text(slice_rows.iloc[0].get("instruction")) if not slice_rows.empty else "No slicing row."

        risk_action = clean_text(r.get("final_risk_action"), clean_text(t.get("final_risk_action"), "NO_DATA"))
        playbook_status = clean_text(t.get("execution_playbook_status"), "NO_DATA")
        cost_status = clean_text(c.get("execution_cost_status"), clean_text(b.get("execution_status"), "NO_DATA"))
        monitor_severity = clean_text(m.get("max_monitor_severity"), clean_text(c.get("monitor_severity"), "NO_DATA"))
        spread_status = clean_text(m.get("spread_status"), clean_text(t.get("spread_status"), "NO_DATA"))
        option_decision = clean_text(opt.get("final_vehicle_decision"), clean_text(rd.get("option_permission_after_repair"), "NO_OPTION_DATA"))
        option_side = clean_text(opt.get("final_option_side"), "NONE")

        base_cost = safe_float(c.get("base_cost_bps"), safe_float(t.get("total_tca_cost_bps")))
        stress_cost = safe_float(c.get("stress_cost_bps"), np.nan)
        fill_rate = safe_float(c.get("expected_fill_rate_pct"), safe_float(t.get("expected_fill_rate_pct")))
        participation = safe_float(c.get("participation_rate_pct"), safe_float(t.get("participation_rate_pct")))
        spread_bps = safe_float(c.get("spread_bps"), safe_float(t.get("spread_bps_est")))
        trade_notional = safe_float(c.get("trade_notional_dollars"), safe_float(t.get("trade_notional_dollars")))
        expected_cost = safe_float(c.get("base_cost_dollars"), safe_float(t.get("expected_cost_dollars")))
        stress_cost_dollars = safe_float(c.get("stress_cost_dollars"), np.nan)
        score = score_execution(
            risk_action,
            playbook_status,
            cost_status,
            monitor_severity,
            spread_status,
            base_cost,
            stress_cost,
            fill_rate,
            float(option_no_go),
        )
        verdict, plain_reason = decide_verdict(
            risk_action,
            playbook_status,
            cost_status,
            monitor_severity,
            spread_status,
            option_decision,
            score,
        )

        blocker_bits = []
        if status_rank(risk_action) >= 3:
            blocker_bits.append(f"risk={risk_action}")
        if status_rank(playbook_status) >= 3:
            blocker_bits.append(f"execution={playbook_status}")
        if status_rank(cost_status) >= 3:
            blocker_bits.append(f"cost={cost_status}")
        if status_rank(monitor_severity) >= 3:
            blocker_bits.append(f"monitor={monitor_severity}")
        if status_rank(spread_status) >= 2:
            blocker_bits.append(f"spread={spread_status}")
        if option_no_go:
            blocker_bits.append(f"option_no_go_checks={option_no_go}")
        if not blocker_bits:
            blocker_bits.append("no primary execution blocker")

        next_check = (
            "Collect manual bid/ask snapshot, confirm spread, avoid open/close auction, "
            "verify ADV/participation, then rerun execution and action-readiness modules."
        )
        if "REDUCE_ONLY" in risk_action.upper():
            next_check = "Repair risk sizing first. Execution review cannot promote new exposure while risk is reduce-only."
        elif "DATA_GAP" in spread_status.upper() or "DATA_GAP" in playbook_status.upper() or "DATA_GAP" in cost_status.upper():
            next_check = "Fix spread/liquidity data gap first with a manual quote snapshot or better intraday quote source."
        elif "CRITICAL" in monitor_severity.upper():
            next_check = "Wait for monitor shock to calm or document why the shock is explained."

        rows.append({
            "ticker": ticker,
            "sector": clean_text(t.get("sector"), clean_text(b.get("sector"), clean_text(m.get("sector"), "Unknown"))),
            "execution_verdict": verdict,
            "execution_score_0_100": score,
            "plain_reason": plain_reason,
            "primary_blockers": "; ".join(blocker_bits),
            "risk_action": risk_action,
            "execution_playbook_status": playbook_status,
            "execution_cost_status": cost_status,
            "monitor_severity": monitor_severity,
            "spread_status": spread_status,
            "trade_direction": clean_text(c.get("direction"), clean_text(t.get("direction"), "NO_DATA")),
            "trade_notional_dollars": round(trade_notional, 2) if np.isfinite(trade_notional) else np.nan,
            "participation_rate_pct": round(participation, 6) if np.isfinite(participation) else np.nan,
            "spread_bps": round(spread_bps, 4) if np.isfinite(spread_bps) else np.nan,
            "base_cost_bps": round(base_cost, 4) if np.isfinite(base_cost) else np.nan,
            "stress_cost_bps": round(stress_cost, 4) if np.isfinite(stress_cost) else np.nan,
            "base_cost_dollars": round(expected_cost, 2) if np.isfinite(expected_cost) else np.nan,
            "stress_cost_dollars": round(stress_cost_dollars, 2) if np.isfinite(stress_cost_dollars) else np.nan,
            "expected_fill_rate_pct": round(fill_rate, 2) if np.isfinite(fill_rate) else np.nan,
            "option_route": option_decision,
            "option_side": option_side,
            "option_no_go_checks": option_no_go,
            "trigger_to_watch": clean_text(opt.get("trigger_to_watch"), clean_text(rd.get("trigger_to_watch"), "NO_TRIGGER")),
            "invalidation": clean_text(opt.get("invalidation"), "NO_INVALIDATION"),
            "first_slice_instruction": first_slice,
            "next_manual_check": next_check,
            "source_files": "execution_cost_model.csv; execution_trade_plan.csv; final_risk_gate.csv; desk_monitor_ticker_state.csv; options_execution_route_matrix.csv; options_tca_no_go_audit.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    board = pd.DataFrame(rows)
    if not board.empty:
        verdict_order = {
            "RISK_REDUCTION_ONLY": 0,
            "MANUAL_SPREAD_LIQUIDITY_CHECK": 1,
            "WAIT_FOR_MONITOR_TO_CALM": 2,
            "NO_NEW_PAPER_ACTION": 3,
            "PUT_OR_HEDGE_RESEARCH_ONLY": 4,
            "TINY_PAPER_AFTER_MANUAL_CHECKS": 5,
            "EXECUTION_RESEARCH_READY": 6,
        }
        board["_rank"] = board["execution_verdict"].map(verdict_order).fillna(9)
        board = board.sort_values(["_rank", "execution_score_0_100", "ticker"], ascending=[True, True, True]).drop(columns=["_rank"])
    return board


def humanize_code(value: Any) -> str:
    raw = clean_text(value, "No data").upper()
    mapping = {
        "RISK_REDUCTION_ONLY": "Risk first: do not add exposure",
        "MANUAL_SPREAD_LIQUIDITY_CHECK": "Needs spread and liquidity check",
        "WAIT_FOR_MONITOR_TO_CALM": "Wait for the live monitor to calm",
        "NO_NEW_PAPER_ACTION": "No new paper action",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "Only put or hedge research",
        "TINY_PAPER_AFTER_MANUAL_CHECKS": "Tiny paper only after manual checks",
        "EXECUTION_RESEARCH_READY": "Execution assumptions look acceptable for paper research",
        "REDUCE_ONLY": "Risk says reduce only",
        "SIZE_DOWN": "Risk says smaller size only",
        "DATA_GAP": "Missing spread or liquidity proof",
        "CRITICAL": "Active live-style warning",
        "WARNING": "Warning",
        "OK": "OK",
        "DOWN": "Downside or risk-reduction route",
        "UP": "Upside route",
        "FLAT": "No directional route",
        "WATCH_EVENT_PROOF_FIRST": "Watch only: confirm the event first",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "Put or hedge research only",
        "CALL_RESEARCH_ONLY": "Call research only",
        "NONE": "No option",
        "PUT": "Put",
        "CALL": "Call",
    }
    if raw in mapping:
        return mapping[raw]
    return raw.replace("_", " ").title()


def human_cost_line(row: pd.Series) -> str:
    base = safe_float(row.get("base_cost_bps"), np.nan)
    stress = safe_float(row.get("stress_cost_bps"), np.nan)
    fill = safe_float(row.get("expected_fill_rate_pct"), np.nan)
    if not np.isfinite(base) and not np.isfinite(stress) and not np.isfinite(fill):
        return "Execution cost is not proven yet."
    parts = []
    if np.isfinite(base):
        if base < 12:
            parts.append(f"normal estimated cost ({base:.1f} bps)")
        elif base < 25:
            parts.append(f"noticeable estimated cost ({base:.1f} bps)")
        else:
            parts.append(f"expensive estimated cost ({base:.1f} bps)")
    if np.isfinite(stress):
        if stress >= 40:
            parts.append(f"stress cost can jump to {stress:.1f} bps")
        else:
            parts.append(f"stress cost about {stress:.1f} bps")
    if np.isfinite(fill):
        if fill >= 95:
            parts.append(f"fill quality looks high ({fill:.0f}%)")
        elif fill >= 85:
            parts.append(f"fill quality needs care ({fill:.0f}%)")
        else:
            parts.append(f"fill quality is weak ({fill:.0f}%)")
    return "; ".join(parts) + "."


def human_route_line(row: pd.Series) -> str:
    direction = humanize_code(row.get("trade_direction"))
    option_route = humanize_code(row.get("option_route"))
    side = humanize_code(row.get("option_side"))
    if "No Option" in side or side == "No data":
        return f"Route: {direction}. Options: {option_route}."
    return f"Route: {direction}. Options: {option_route}; side to review: {side}."


def human_blocker_line(row: pd.Series) -> str:
    raw = clean_text(row.get("primary_blockers"))
    if not raw:
        return "No primary blocker listed."
    blockers = []
    for part in raw.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "risk":
            blockers.append(f"risk gate: {humanize_code(value)}")
        elif key == "execution":
            blockers.append(f"execution proof: {humanize_code(value)}")
        elif key == "cost":
            blockers.append(f"cost model: {humanize_code(value)}")
        elif key == "monitor":
            blockers.append(f"live monitor: {humanize_code(value)}")
        elif key == "spread":
            blockers.append(f"spread data: {humanize_code(value)}")
        elif key == "option_no_go_checks":
            try:
                n = int(float(value))
            except Exception:
                n = 0
            blockers.append(f"{n} option no-go checks")
        else:
            blockers.append(f"{key}: {humanize_code(value)}")
    if not blockers:
        return humanize_code(raw)
    return "What blocks it now: " + "; ".join(blockers) + "."


def build_cards(board: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if board.empty:
        return pd.DataFrame(rows)
    for _, row in board.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        verdict = clean_text(row.get("execution_verdict"))
        score = safe_float(row.get("execution_score_0_100"), 0.0)
        if verdict == "RISK_REDUCTION_ONLY":
            action = "Do not add. Repair or reduce risk first."
        elif verdict == "MANUAL_SPREAD_LIQUIDITY_CHECK":
            action = "Open manual spread/liquidity check before any paper route."
        elif verdict == "WAIT_FOR_MONITOR_TO_CALM":
            action = "Wait for monitor shock to calm or document the cause."
        elif verdict == "PUT_OR_HEDGE_RESEARCH_ONLY":
            action = "Only defined-risk put/hedge research after manual checks."
        elif verdict == "TINY_PAPER_AFTER_MANUAL_CHECKS":
            action = "Tiny paper only after spread, fill, and source proof."
        elif verdict == "EXECUTION_RESEARCH_READY":
            action = "Execution assumptions acceptable for research paper workflow."
        else:
            action = "No new paper action until blockers clear."

        rows.append({
            "ticker": ticker,
            "card_status": verdict,
            "score": score,
            "headline": f"{ticker}: {action}",
            "cost_line": human_cost_line(row),
            "route_line": human_route_line(row),
            "blocker_line": human_blocker_line(row),
            "manual_check": clean_text(row.get("next_manual_check")),
            "trigger": clean_text(row.get("trigger_to_watch")),
            "source_files": clean_text(row.get("source_files")),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_source_guide() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source_area": "Cost and fill model",
            "file": "execution_cost_model.csv",
            "what_it_answers": "Base/stress TCA, spread proxy, market impact, expected fill rate, and execution cost status.",
        },
        {
            "source_area": "Trade plan and slicing",
            "file": "execution_trade_plan.csv; execution_slicing_schedule.csv",
            "what_it_answers": "Trade direction, paper notional, allowed participation, estimated days, and slicing assumption.",
        },
        {
            "source_area": "Risk permission",
            "file": "final_risk_gate.csv; action_readiness_monitor.csv",
            "what_it_answers": "Whether risk allows new exposure, reduction only, or non-risk gate review.",
        },
        {
            "source_area": "Live-style monitor",
            "file": "desk_monitor_ticker_state.csv; desk_monitor_events.csv",
            "what_it_answers": "Price break, volume spike, volatility shift, spread widening, and risk breach context.",
        },
        {
            "source_area": "Options execution route",
            "file": "options_execution_route_matrix.csv; options_tca_no_go_audit.csv",
            "what_it_answers": "Call/put/hedge/no-option route, no-go checks, trigger, invalidation, and manual confirmation.",
        },
    ])


def write_outputs(board: pd.DataFrame, cards: pd.DataFrame, source_guide: pd.DataFrame) -> None:
    board.to_csv(OUT_BOARD, index=False)
    cards.to_csv(OUT_CARDS, index=False)
    source_guide.to_csv(OUT_SOURCE_GUIDE, index=False)

    if board.empty:
        state = {
            "date": today_str(),
            "status": "NO_EXECUTION_TCA_DATA",
            "overall_execution_tca_score": 0.0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    else:
        verdict_counts = board["execution_verdict"].value_counts().to_dict()
        score = round(float(pd.to_numeric(board["execution_score_0_100"], errors="coerce").fillna(0.0).mean()), 1)
        data_gap_count = int(board["execution_verdict"].astype(str).eq("MANUAL_SPREAD_LIQUIDITY_CHECK").sum())
        risk_reduce_count = int(board["execution_verdict"].astype(str).eq("RISK_REDUCTION_ONLY").sum())
        monitor_wait_count = int(board["execution_verdict"].astype(str).eq("WAIT_FOR_MONITOR_TO_CALM").sum())
        ready_count = int(board["execution_verdict"].astype(str).eq("EXECUTION_RESEARCH_READY").sum())
        if ready_count:
            status = "EXECUTION_RESEARCH_READY_WITH_MANUAL_CHECKS"
        elif risk_reduce_count or data_gap_count or monitor_wait_count:
            status = "EXECUTION_BLOCKED_OR_REPAIR_FIRST"
        else:
            status = "EXECUTION_REVIEW_REQUIRED"
        state = {
            "date": today_str(),
            "status": status,
            "overall_execution_tca_score": score,
            "ticker_rows": int(len(board)),
            "risk_reduction_only_count": risk_reduce_count,
            "manual_spread_liquidity_check_count": data_gap_count,
            "wait_for_monitor_count": monitor_wait_count,
            "execution_research_ready_count": ready_count,
            "top_blocker_ticker": clean_text(board.iloc[0].get("ticker")),
            "top_blocker": clean_text(board.iloc[0].get("execution_verdict")),
            "verdict_counts": verdict_counts,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
            "truth": "Execution/TCA depth is a paper-research feasibility layer. It cannot place trades or override risk gates.",
            "outputs": {
                "board": OUT_BOARD.name,
                "cards": OUT_CARDS.name,
                "source_guide": OUT_SOURCE_GUIDE.name,
                "report": OUT_REPORT.name,
            },
        }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Status: **{state.get('status')}**",
        f"- Overall execution/TCA score: **{state.get('overall_execution_tca_score')}/100**",
        f"- Risk reduction only: **{state.get('risk_reduction_only_count', 0)}**",
        f"- Manual spread/liquidity checks: **{state.get('manual_spread_liquidity_check_count', 0)}**",
        f"- Wait for monitor: **{state.get('wait_for_monitor_count', 0)}**",
        f"- Execution research ready: **{state.get('execution_research_ready_count', 0)}**",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        "## Decision Board",
        "",
        df_to_markdown(board, max_rows=80),
        "",
        "## Ticker Cards",
        "",
        df_to_markdown(cards, max_rows=80),
        "",
        "## Source Guide",
        "",
        df_to_markdown(source_guide, max_rows=20),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Execution / TCA Depth Desk", sections)


def main() -> None:
    board = build_board()
    cards = build_cards(board)
    source_guide = build_source_guide()
    write_outputs(board, cards, source_guide)
    print("Canyon execution/TCA depth desk complete.")
    if not board.empty:
        print(f"Rows: {len(board)} | score: {pd.to_numeric(board['execution_score_0_100'], errors='coerce').mean():.1f}")
        print("Top verdict:", board.iloc[0]["ticker"], board.iloc[0]["execution_verdict"])
    print(f"Outputs: {OUT_BOARD.name}, {OUT_CARDS.name}, {OUT_STATE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
