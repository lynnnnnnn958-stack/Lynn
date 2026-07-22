#!/usr/bin/env python3
"""
Canyon v9 Step 174 - Options Unlock Board.

Research-only. No broker connection. No live orders.

Step173 explains the current option/execution route. Step174 explains why a
route is locked and what would need to change before a call, put/hedge, or
underlying-only route can be promoted.

It is intentionally conservative:
  - Risk gate is checked first.
  - Call unlocks require risk, event proof, execution, monitor, and IV/Greeks.
  - Put/hedge routes are labeled as risk research only, not bullish trades.

Outputs:
  option_unlock_blocker_attribution.csv
  call_unlock_board.csv
  put_hedge_unlock_board.csv
  option_unlock_summary.csv
  option_unlock_state.json
  option_unlock_report.md
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


OUT_ATTRIBUTION = ROOT / "option_unlock_blocker_attribution.csv"
OUT_CALL = ROOT / "call_unlock_board.csv"
OUT_PUT = ROOT / "put_hedge_unlock_board.csv"
OUT_SUMMARY = ROOT / "option_unlock_summary.csv"
OUT_STATE = ROOT / "option_unlock_state.json"
OUT_REPORT = ROOT / "option_unlock_report.md"


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def blocker_flags(row: pd.Series) -> dict[str, bool]:
    risk_text = f"{row.get('risk_action', '')} {row.get('gate_status', '')}".upper()
    no_go = as_upper(row.get("no_go_reasons"), "")
    return {
        "risk_blocker": (
            safe_float(row.get("risk_score"), 0) < 45
            or "REDUCE_ONLY" in risk_text
            or "NO NEW" in risk_text
            or "NO_NEW" in risk_text
        ),
        "execution_blocker": safe_float(row.get("execution_score"), 0) < 50,
        "greeks_iv_gamma_blocker": safe_float(row.get("greeks_score"), 0) < 50,
        "monitor_blocker": safe_float(row.get("monitor_score"), 0) < 50,
        "event_proof_blocker": safe_float(row.get("event_score"), 0) < 45,
        "backtest_sanity_blocker": "REVIEW" in as_upper(row.get("backtest_sanity_status"), ""),
        "spread_data_blocker": "SPREAD STATE IS DATA_GAP" in no_go or "LIVE SPREAD IS MISSING" in no_go,
        "high_iv_blocker": "IV RANK" in no_go and "HIGH" in no_go,
    }


def first_blocker(flags: dict[str, bool]) -> str:
    order = [
        ("risk_blocker", "Risk gate"),
        ("monitor_blocker", "Live monitor shock"),
        ("execution_blocker", "Execution/TCA"),
        ("event_proof_blocker", "News/event proof"),
        ("greeks_iv_gamma_blocker", "IV/Greeks/Gamma"),
        ("spread_data_blocker", "Spread data"),
        ("backtest_sanity_blocker", "Options backtest sanity"),
        ("high_iv_blocker", "High IV"),
    ]
    for key, label in order:
        if flags.get(key):
            return label
    return "No hard blocker"


def unlock_sequence(flags: dict[str, bool], row: pd.Series, target: str) -> str:
    steps: list[str] = []
    if flags.get("risk_blocker"):
        steps.append("1. Risk gate must move beyond REDUCE_ONLY / no-new-exposure / tiny-only.")
    if flags.get("monitor_blocker"):
        steps.append("2. Active monitor shock must calm or be explained.")
    if flags.get("execution_blocker") or flags.get("spread_data_blocker"):
        steps.append("3. Confirm real spread, fill rate, stress TCA, and avoid open/close auction.")
    if flags.get("event_proof_blocker"):
        steps.append("4. Require source timestamp, model-seen proof, and price/volume confirmation.")
    if flags.get("greeks_iv_gamma_blocker") or flags.get("high_iv_blocker"):
        steps.append("5. IV/Greeks/gamma must allow a defined-risk spread; no naked weekly premium.")
    if flags.get("backtest_sanity_blocker"):
        steps.append("6. Treat option backtest as warning only until proxy metrics are repaired.")
    if not steps:
        steps.append("Manual check: route is research-only and still requires trigger, liquidity, and spread review.")

    trigger = as_text(row.get("trigger_to_watch"), "")
    if trigger:
        steps.append(f"Trigger proof: {trigger}")
    if target == "CALL":
        steps.append("Final call rule: only defined-risk call spread, never naked weekly OTM chase.")
    elif target == "PUT":
        steps.append("Final put rule: hedge/downside research only, not bullish leverage.")
    return " ".join(steps)


def call_candidate_type(row: pd.Series) -> str:
    side = as_upper(row.get("base_option_side"), "NONE")
    event_bias = as_upper(row.get("event_bias"), "")
    trigger = as_upper(row.get("trigger_to_watch"), "")
    best_bt = as_upper(row.get("best_option_backtest_strategy"), "")
    if side == "CALL":
        return "DIRECT_CALL_CANDIDATE"
    if "BULLISH" in event_bias or "CALL" in as_upper(row.get("base_option_route"), ""):
        return "EVENT_CALL_WATCH"
    if "ABOVE" in trigger or "HIGH" in trigger or "BULL_CALL" in best_bt:
        return "LATENT_UPSIDE_TRIGGER"
    return "NOT_A_CALL_CANDIDATE"


def call_status(row: pd.Series, flags: dict[str, bool], candidate: str) -> str:
    if candidate == "NOT_A_CALL_CANDIDATE":
        return "NO_CALL_THESIS"
    if flags.get("risk_blocker"):
        return "CALL_LOCKED_BY_RISK"
    if flags.get("monitor_blocker"):
        return "CALL_LOCKED_BY_MONITOR"
    if flags.get("execution_blocker") or flags.get("spread_data_blocker"):
        return "CALL_LOCKED_BY_EXECUTION"
    if flags.get("event_proof_blocker"):
        return "CALL_LOCKED_BY_EVENT_PROOF"
    if flags.get("greeks_iv_gamma_blocker") or flags.get("high_iv_blocker"):
        return "CALL_LOCKED_BY_IV_GREEKS"
    if flags.get("backtest_sanity_blocker"):
        return "CALL_REVIEW_BACKTEST_PROXY"
    if safe_float(row.get("route_quality_score"), 0) >= 62:
        return "CALL_RESEARCH_UNLOCKED_MANUAL_ONLY"
    return "CALL_WATCH_NEEDS_MORE_SCORE"


def put_candidate_type(row: pd.Series) -> str:
    side = as_upper(row.get("base_option_side"), "NONE")
    final_side = as_upper(row.get("final_option_side"), "NONE")
    final_decision = as_upper(row.get("final_vehicle_decision"), "")
    risk = as_upper(row.get("risk_action"), "")
    if final_side == "PUT" or "PUT_OR_HEDGE" in final_decision:
        return "ACTIVE_PUT_OR_HEDGE_RESEARCH"
    if side == "PUT":
        return "LATENT_PUT_OR_HEDGE"
    if "REDUCE_ONLY" in risk:
        return "RISK_REDUCTION_WATCH"
    return "NO_PUT_HEDGE_THESIS"


def build_attribution(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        flags = blocker_flags(row)
        flag_count = int(sum(flags.values()))
        rows.append({
            "ticker": row.get("ticker"),
            "sector": row.get("sector"),
            "horizon": row.get("horizon"),
            "route_quality_score": row.get("route_quality_score"),
            "final_vehicle_decision": row.get("final_vehicle_decision"),
            "final_option_side": row.get("final_option_side"),
            "call_candidate_type": call_candidate_type(row),
            "put_candidate_type": put_candidate_type(row),
            "blocker_count": flag_count,
            "first_blocker": first_blocker(flags),
            **flags,
            "no_go_reasons": row.get("no_go_reasons"),
            "trigger_to_watch": row.get("trigger_to_watch"),
            "required_confirmation": row.get("required_confirmation"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def build_call_board(matrix: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, sub in matrix.groupby("ticker"):
        callish = sub.copy()
        callish["call_candidate_type"] = callish.apply(call_candidate_type, axis=1)
        callish["call_candidate_rank"] = callish["call_candidate_type"].map({
            "DIRECT_CALL_CANDIDATE": 3,
            "EVENT_CALL_WATCH": 2,
            "LATENT_UPSIDE_TRIGGER": 1,
            "NOT_A_CALL_CANDIDATE": 0,
        }).fillna(0)
        callish = callish.sort_values(["call_candidate_rank", "route_quality_score"], ascending=[False, False])
        best = callish.iloc[0]
        flags = blocker_flags(best)
        candidate = call_candidate_type(best)
        status = call_status(best, flags, candidate)
        rows.append({
            "ticker": ticker,
            "sector": best.get("sector"),
            "call_candidate_type": candidate,
            "call_unlock_status": status,
            "best_call_horizon": best.get("horizon"),
            "route_quality_score": best.get("route_quality_score"),
            "base_option_side": best.get("base_option_side"),
            "risk_action": best.get("risk_action"),
            "risk_score": best.get("risk_score"),
            "execution_score": best.get("execution_score"),
            "greeks_score": best.get("greeks_score"),
            "monitor_score": best.get("monitor_score"),
            "event_score": best.get("event_score"),
            "first_blocker": first_blocker(flags),
            "blocker_count": int(sum(flags.values())),
            "call_unlock_sequence": unlock_sequence(flags, best, "CALL"),
            "call_structure_if_unlocked": "Defined-risk call debit spread; no naked weekly calls.",
            "premium_budget_if_unlocked_bps": 8.0 if safe_float(best.get("route_quality_score"), 0) < 62 else 15.0,
            "trigger_to_watch": best.get("trigger_to_watch"),
            "invalidation": best.get("invalidation"),
            "no_go_reasons": best.get("no_go_reasons"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["call_candidate_type", "route_quality_score"], ascending=[True, False]).reset_index(drop=True)


def build_put_board(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, sub in matrix.groupby("ticker"):
        work = sub.copy()
        work["put_candidate_type"] = work.apply(put_candidate_type, axis=1)
        work["put_rank"] = work["put_candidate_type"].map({
            "ACTIVE_PUT_OR_HEDGE_RESEARCH": 3,
            "LATENT_PUT_OR_HEDGE": 2,
            "RISK_REDUCTION_WATCH": 1,
            "NO_PUT_HEDGE_THESIS": 0,
        }).fillna(0)
        work = work.sort_values(["put_rank", "route_quality_score"], ascending=[False, False])
        best = work.iloc[0]
        flags = blocker_flags(best)
        put_type = put_candidate_type(best)
        if put_type == "ACTIVE_PUT_OR_HEDGE_RESEARCH":
            hedge_status = "HEDGE_RESEARCH_ONLY_MANUAL_CHECKS"
        elif put_type == "LATENT_PUT_OR_HEDGE":
            hedge_status = "PUT_HEDGE_LOCKED_NEEDS_GATES"
        elif put_type == "RISK_REDUCTION_WATCH":
            hedge_status = "RISK_REDUCTION_ONLY_NO_OPTION"
        else:
            hedge_status = "NO_PUT_HEDGE_THESIS"
        rows.append({
            "ticker": ticker,
            "sector": best.get("sector"),
            "put_candidate_type": put_type,
            "hedge_unlock_status": hedge_status,
            "best_put_horizon": best.get("horizon"),
            "route_quality_score": best.get("route_quality_score"),
            "final_vehicle_decision": best.get("final_vehicle_decision"),
            "risk_action": best.get("risk_action"),
            "first_blocker": first_blocker(flags),
            "blocker_count": int(sum(flags.values())),
            "hedge_unlock_sequence": unlock_sequence(flags, best, "PUT"),
            "put_structure_if_unlocked": "Put debit spread or protective hedge research only.",
            "premium_budget_if_unlocked_bps": 5.0 if flags.get("risk_blocker") else 10.0,
            "trigger_to_watch": best.get("trigger_to_watch"),
            "invalidation": best.get("invalidation"),
            "no_go_reasons": best.get("no_go_reasons"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["put_candidate_type", "route_quality_score"], ascending=[True, False]).reset_index(drop=True)


def build_summary(call_board: pd.DataFrame, put_board: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "tickers_total",
            "value": int(call_board["ticker"].nunique()) if not call_board.empty else 0,
            "meaning": "Tickers with option unlock analysis.",
        },
        {
            "metric": "direct_call_candidates",
            "value": int(call_board["call_candidate_type"].astype(str).eq("DIRECT_CALL_CANDIDATE").sum()) if not call_board.empty else 0,
            "meaning": "Tickers whose base route contains a call candidate.",
        },
        {
            "metric": "call_unlocked_manual_only",
            "value": int(call_board["call_unlock_status"].astype(str).eq("CALL_RESEARCH_UNLOCKED_MANUAL_ONLY").sum()) if not call_board.empty else 0,
            "meaning": "Call candidates that passed the model gates, still manual research only.",
        },
        {
            "metric": "call_locked_by_risk",
            "value": int(call_board["call_unlock_status"].astype(str).eq("CALL_LOCKED_BY_RISK").sum()) if not call_board.empty else 0,
            "meaning": "Call candidates where L8 risk blocks bullish options first.",
        },
        {
            "metric": "put_or_hedge_research",
            "value": int(put_board["hedge_unlock_status"].astype(str).eq("HEDGE_RESEARCH_ONLY_MANUAL_CHECKS").sum()) if not put_board.empty else 0,
            "meaning": "Tickers with put/hedge research route; not bullish trades.",
        },
        {
            "metric": "risk_blocker_rows",
            "value": int(attribution["risk_blocker"].sum()) if not attribution.empty else 0,
            "meaning": "Horizon rows blocked by risk.",
        },
        {
            "metric": "execution_blocker_rows",
            "value": int(attribution["execution_blocker"].sum()) if not attribution.empty else 0,
            "meaning": "Horizon rows blocked by TCA/spread/fill risk.",
        },
        {
            "metric": "event_proof_blocker_rows",
            "value": int(attribution["event_proof_blocker"].sum()) if not attribution.empty else 0,
            "meaning": "Horizon rows blocked by news/event proof.",
        },
        {
            "metric": "monitor_blocker_rows",
            "value": int(attribution["monitor_blocker"].sum()) if not attribution.empty else 0,
            "meaning": "Horizon rows blocked by live monitor shock.",
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(attribution: pd.DataFrame, call_board: pd.DataFrame, put_board: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    attribution.to_csv(OUT_ATTRIBUTION, index=False)
    call_board.to_csv(OUT_CALL, index=False)
    put_board.to_csv(OUT_PUT, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    metric = dict(zip(summary["metric"], summary["value"])) if not summary.empty else {}
    state = {
        "date": today_str(),
        "ticker_count": int(metric.get("tickers_total", 0)),
        "direct_call_candidates": int(metric.get("direct_call_candidates", 0)),
        "call_unlocked_manual_only": int(metric.get("call_unlocked_manual_only", 0)),
        "call_locked_by_risk": int(metric.get("call_locked_by_risk", 0)),
        "put_or_hedge_research": int(metric.get("put_or_hedge_research", 0)),
        "risk_blocker_rows": int(metric.get("risk_blocker_rows", 0)),
        "execution_blocker_rows": int(metric.get("execution_blocker_rows", 0)),
        "event_proof_blocker_rows": int(metric.get("event_proof_blocker_rows", 0)),
        "monitor_blocker_rows": int(metric.get("monitor_blocker_rows", 0)),
        "overall_status": "OPTION_UNLOCK_BOARD_ACTIVE" if len(attribution) else "NO_OPTION_UNLOCK_DATA",
        "truth": "This board explains why option routes are locked. It does not unlock live trading.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "attribution": OUT_ATTRIBUTION.name,
            "call_board": OUT_CALL.name,
            "put_board": OUT_PUT.name,
            "summary": OUT_SUMMARY.name,
            "report": OUT_REPORT.name,
        },
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Desk truth",
        "This board explains why option routes are locked and what must change before any call or put/hedge research route can be considered. It is research-only.",
        "",
        "## Summary",
        df_to_markdown(summary),
        "",
        "## Call unlock board",
        df_to_markdown(call_board[[
            "ticker", "call_candidate_type", "call_unlock_status", "best_call_horizon",
            "route_quality_score", "first_blocker", "blocker_count", "call_unlock_sequence",
        ]], max_rows=30) if not call_board.empty else "No rows.",
        "",
        "## Put / hedge unlock board",
        df_to_markdown(put_board[[
            "ticker", "put_candidate_type", "hedge_unlock_status", "best_put_horizon",
            "route_quality_score", "first_blocker", "blocker_count", "hedge_unlock_sequence",
        ]], max_rows=30) if not put_board.empty else "No rows.",
        "",
        "## Blocker attribution",
        df_to_markdown(attribution.head(80), max_rows=80),
        "",
        "## Non-negotiable",
        "- No broker connection. No live orders.",
        "- Call unlock requires risk, event proof, TCA/spread, monitor, and IV/Greeks to clear.",
        "- Put/hedge rows are risk-research rows, not bullish permission.",
        "- No naked weekly options.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 174 - Options Unlock Board", sections)
    return state


def main() -> None:
    matrix = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    attribution = build_attribution(matrix)
    call_board = build_call_board(matrix, attribution)
    put_board = build_put_board(matrix)
    summary = build_summary(call_board, put_board, attribution)
    state = write_outputs(attribution, call_board, put_board, summary)

    print("Step 174 complete.")
    print(f"Tickers: {state['ticker_count']}; direct call candidates: {state['direct_call_candidates']}")
    print(f"Call unlocked manual-only: {state['call_unlocked_manual_only']}; call locked by risk: {state['call_locked_by_risk']}")
    print(f"Put/hedge research: {state['put_or_hedge_research']}")
    print(f"Wrote: {OUT_ATTRIBUTION.name}, {OUT_CALL.name}, {OUT_PUT.name}, {OUT_SUMMARY.name}, {OUT_STATE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
