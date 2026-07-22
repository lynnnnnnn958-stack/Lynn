#!/usr/bin/env python3
"""
Canyon v9 - Step 167: Horizon and Vehicle Router
================================================

Research-only. No broker connection. No live orders.

Step167 makes the decision layer easier to read by separating each ticker into
short-term, medium-term, and long-term research routes. It also spells out the
vehicle clearly: stock/ETF paper only, call research, put/hedge research, or no
new exposure. Options never override risk, event, execution, or news-reliability
gates.

Outputs:
  horizon_vehicle_matrix.csv
  horizon_vehicle_summary.csv
  option_route_clarity_board.csv
  horizon_vehicle_state.json
  horizon_vehicle_report.md
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

IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_SECTOR_OPTION = ROOT / "sector_timeframe_option_route.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_CARDS = ROOT / "ticker_decision_cards.csv"
IN_ROOM = ROOT / "ticker_decision_room.csv"
IN_QUEUE = ROOT / "daily_workflow_queue.csv"
IN_EVENT_RELIABILITY = ROOT / "event_signal_reliability_watchlist.csv"

OUT_MATRIX = ROOT / "horizon_vehicle_matrix.csv"
OUT_SUMMARY = ROOT / "horizon_vehicle_summary.csv"
OUT_OPTION_BOARD = ROOT / "option_route_clarity_board.csv"
OUT_STATE = ROOT / "horizon_vehicle_state.json"
OUT_REPORT = ROOT / "horizon_vehicle_report.md"


HORIZONS = [
    ("Short-term", "1-5 trading days", "short_score_after", "short_decision", "short_term_plan"),
    ("Medium-term", "2-8 weeks", "medium_score_after", "medium_decision", "medium_term_plan"),
    ("Long-term", "3-12 months", "long_score_after", "long_decision", "long_term_plan"),
]


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    raw = str(value).strip()
    return "" if raw.lower() == "nan" else raw


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def shorten(value: Any, limit: int = 620) -> str:
    raw = text(value)
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "..."


def first_nonempty(*values: Any) -> str:
    for value in values:
        raw = text(value)
        if raw:
            return raw
    return ""


def normalize_ticker(df: pd.DataFrame, column: str = "ticker") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[column] = out[column].astype(str).str.upper().str.strip()
    out = out[out[column] != ""]
    return out


def one_by_ticker(df: pd.DataFrame, column: str = "ticker") -> pd.DataFrame:
    out = normalize_ticker(df, column)
    if out.empty:
        return pd.DataFrame()
    return out.drop_duplicates(column, keep="first").set_index(column)


def row_at(indexed: pd.DataFrame, ticker: str) -> pd.Series:
    if indexed.empty or ticker not in indexed.index:
        return pd.Series(dtype=object)
    row = indexed.loc[ticker]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def source_trace(*parts: Any) -> str:
    seen: list[str] = []
    for part in parts:
        raw = text(part)
        if not raw:
            continue
        for piece in raw.replace("+", ";").split(";"):
            item = piece.strip()
            if item and item not in seen:
                seen.append(item)
    return shorten("; ".join(seen), 760)


def gate_level(risk_action: Any, event_gate: Any, news_status: Any, monitor_count: Any) -> str:
    risk = upper(risk_action)
    event = upper(event_gate)
    news = upper(news_status)
    monitors = safe_float(monitor_count, 0.0)
    if "REDUCE_ONLY" in risk or "RISK FIRST" in risk:
        return "No new exposure"
    if "SIZE_DOWN" in risk:
        return "Tiny paper only"
    if "MISSING" in event or "REVIEW" in event:
        return "Source review first"
    if "LOW_SAMPLE" in news or "UNPROVEN" in news or "PENDING" in news:
        return "News confirmation first"
    if monitors > 0:
        return "Monitor first"
    return "Research review allowed"


def call_status(opt: pd.Series, sector_opt: pd.Series, risk_action: Any, event_gate: Any) -> str:
    permission = upper(opt.get("option_permission"))
    route = upper(first_nonempty(sector_opt.get("option_route"), opt.get("option_answer")))
    side = upper(first_nonempty(sector_opt.get("option_side"), opt.get("option_side")))
    risk = upper(risk_action)
    event = upper(event_gate)
    if "REDUCE_ONLY" in risk:
        return "No call. Risk gate blocks bullish exposure."
    if "SIZE_DOWN" in risk:
        return "No call now. Risk gate only allows tiny paper research."
    if "MISSING" in event or "REVIEW" in event:
        return "No call now. Event/source review first."
    if "CALL_BLOCKED" in permission:
        return "Call edge exists, but current gates block it."
    if "CALL" in permission or "CALL" in side or "CALL" in route:
        return "Defined-risk call spread research after all gates clear."
    return "No call edge."


def put_status(opt: pd.Series, sector_opt: pd.Series, risk_action: Any, event_gate: Any) -> str:
    permission = upper(opt.get("option_permission"))
    route = upper(first_nonempty(sector_opt.get("option_route"), opt.get("option_answer")))
    side = upper(first_nonempty(sector_opt.get("option_side"), opt.get("option_side")))
    risk = upper(risk_action)
    event = upper(event_gate)
    if "MISSING" in event or "REVIEW" in event:
        return "Put/hedge review only after source check."
    if "PUT" in permission or "HEDGE" in permission or "PUT" in side or "HEDGE" in route:
        return "Put spread or protective hedge research only."
    if "REDUCE_ONLY" in risk:
        return "Risk reduction first; hedge only if tied to portfolio risk."
    return "No put edge."


def gate_stack(
    risk_action: Any,
    event_gate: Any,
    news_status: Any,
    news_action: Any,
    monitor_count: Any,
    no_go_conditions: Any = "",
) -> list[str]:
    risk = upper(risk_action)
    event = upper(event_gate)
    news = upper(news_status)
    news_act = upper(news_action)
    monitors = safe_float(monitor_count, 0.0)
    no_go = text(no_go_conditions)
    stack: list[str] = []
    if "REDUCE_ONLY" in risk:
        stack.append("L8 Risk: reduce-only blocks new exposure")
    elif "SIZE_DOWN" in risk:
        stack.append("L8 Risk: size-down allows at most tiny paper research")
    if "MISSING" in event or "REVIEW" in event:
        stack.append("L5 Event: source/earnings/news review is not clear")
    if "LOW_SAMPLE" in news:
        stack.append("L5 News: local audit sample is too small")
    if "UNPROVEN" in news:
        stack.append("L5 News: local context is not proven yet")
    if "PENDING" in news:
        stack.append("L5 News: post first-seen price window is pending")
    if "WATCH_ONLY" in news_act or "DO_NOT_UPGRADE" in news_act:
        stack.append("L5 News: headline cannot upgrade the route by itself")
    if monitors > 0:
        stack.append("Monitor: active price/news/volume alert needs review")
    if no_go:
        stack.append("Execution/Options: no-go conditions must be checked")
    return stack or ["No blocking gate detected"]


def unlock_checklist(
    risk_action: Any,
    event_gate: Any,
    news_status: Any,
    news_action: Any,
    monitor_count: Any,
    trigger: Any,
    no_go_conditions: Any = "",
) -> str:
    risk = upper(risk_action)
    event = upper(event_gate)
    news = upper(news_status)
    news_act = upper(news_action)
    monitors = safe_float(monitor_count, 0.0)
    checks: list[str] = []
    if "REDUCE_ONLY" in risk:
        checks.append("risk gate must move from REDUCE_ONLY to SIZE_DOWN or CLEAR")
    if "SIZE_DOWN" in risk:
        checks.append("risk gate must clear or portfolio size must fall to the recommended tiny paper budget")
    if "MISSING" in event or "REVIEW" in event:
        checks.append("event/news/earnings source review must clear")
    if any(token in news for token in ["LOW_SAMPLE", "UNPROVEN", "PENDING"]):
        checks.append("news signal needs price/volume confirmation and more local audit evidence")
    if "WATCH_ONLY" in news_act or "DO_NOT_UPGRADE" in news_act:
        checks.append("do not upgrade from the headline alone")
    if monitors > 0:
        checks.append("active monitor alert must calm or be explained")
    if text(no_go_conditions):
        checks.append("spread, IV, liquidity, and no-go conditions must be manually checked")
    if text(trigger):
        checks.append(f"price must confirm around: {text(trigger)}")
    return "; ".join(checks) if checks else "No major blocking checklist item detected; still manual research only."


def option_use_case(call: str, put: str, gate: str) -> str:
    raw = f"{call} {put} {gate}".upper()
    if "NO NEW EXPOSURE" in upper(gate):
        return "Risk reduction first"
    if "NO CALL NOW" in raw and ("PUT SPREAD" in raw or "PROTECTIVE HEDGE" in raw):
        return "No bullish option; hedge research only"
    if "NO CALL NOW" in raw:
        return "Call idea blocked; stock/ETF paper context only"
    if "DEFINED-RISK CALL" in raw:
        return "Defined-risk call watch after gates clear"
    if "PUT SPREAD" in raw or "PROTECTIVE HEDGE" in raw:
        return "Put or protective hedge research"
    return "No clean option use case"


def action_permission_state(gate: str, vehicle: str) -> str:
    gate_u = upper(gate)
    vehicle_u = upper(vehicle)
    if "NO NEW EXPOSURE" in gate_u:
        return "Blocked"
    if "TINY PAPER" in gate_u:
        return "Tiny paper only"
    if "REVIEW" in gate_u or "CONFIRMATION" in gate_u or "MONITOR" in gate_u:
        return "Wait for proof"
    if "CALL" in vehicle_u or "PUT" in vehicle_u or "HEDGE" in vehicle_u:
        return "Option research only after final manual gates"
    return "Research review allowed"


def route_confidence(score: float, gate: str, news_status: Any, monitor_count: Any, vehicle: str) -> float:
    score_part = max(0.0, min(35.0, (score if np.isfinite(score) else 0.0) * 0.45))
    out = 45.0 + score_part
    gate_u = upper(gate)
    news = upper(news_status)
    monitors = safe_float(monitor_count, 0.0)
    if "NO NEW EXPOSURE" in gate_u:
        out = min(out, 18.0)
    elif "TINY PAPER" in gate_u:
        out = min(out, 38.0)
    elif "REVIEW" in gate_u or "CONFIRMATION" in gate_u:
        out = min(out, 42.0)
    if any(token in news for token in ["LOW_SAMPLE", "UNPROVEN", "PENDING"]):
        out -= 8.0
    if monitors > 0:
        out -= min(10.0, monitors * 2.0)
    if "NO VEHICLE" in upper(vehicle):
        out -= 5.0
    return round(max(0.0, min(100.0, out)), 1)


def horizon_consensus(actions: dict[str, str], vehicles: dict[str, str]) -> str:
    action_set = {upper(v) for v in actions.values() if text(v)}
    vehicle_set = {upper(v) for v in vehicles.values() if text(v)}
    if len(action_set) <= 1 and len(vehicle_set) <= 1:
        return "Same route across all horizons"
    if any("CALL" in v for v in vehicle_set) and any("STOCK" in v for v in vehicle_set):
        return "Mixed vehicle: option should be tactical, not the whole thesis"
    if any("PUT" in v or "HEDGE" in v for v in vehicle_set):
        return "Risk/hedge route appears in at least one horizon"
    return "Different horizon plans; open the matrix before acting"


def primary_research_question(gate: str, call: str, put: str, news_status: Any) -> str:
    gate_u = upper(gate)
    if "NO NEW EXPOSURE" in gate_u:
        return "Can portfolio risk be reduced before considering any new idea?"
    if "TINY PAPER" in gate_u:
        return "Is this worth tiny stock/ETF paper research despite the risk gate?"
    if "UNPROVEN" in upper(news_status) or "LOW_SAMPLE" in upper(news_status):
        return "Does price and volume confirm the news read-through, or is it just noise?"
    if "DEFINED-RISK CALL" in upper(call):
        return "Is there enough confirmed upside edge for defined-risk call research?"
    if "PUT SPREAD" in upper(put) or "PROTECTIVE HEDGE" in upper(put):
        return "Is this hedge tied to real portfolio risk, or just a standalone bearish bet?"
    return "Is there enough evidence to keep this ticker on the research desk?"


def timeframe_fit_warning(horizon: str, vehicle: str, gate: str) -> str:
    vehicle_u = upper(vehicle)
    if horizon == "Long-term" and "CALL" in vehicle_u:
        return "Long-term thesis should not rely on short-dated call premium."
    if horizon == "Short-term" and "NO VEHICLE" in vehicle_u:
        return "Short-term idea has no usable vehicle until gates clear."
    if "TINY PAPER" in upper(gate):
        return "Risk gate caps all horizons at tiny paper research."
    return ""


def vehicle_for_horizon(
    horizon: str,
    plan: str,
    risk_action: Any,
    event_gate: Any,
    opt: pd.Series,
    sector_opt: pd.Series,
    news_status: Any,
    monitor_count: Any,
) -> tuple[str, str, str]:
    gate = gate_level(risk_action, event_gate, news_status, monitor_count)
    plan_u = upper(plan)
    option_side = upper(first_nonempty(sector_opt.get("option_side"), opt.get("option_side")))
    option_route = upper(first_nonempty(sector_opt.get("option_route"), opt.get("option_answer")))
    permission = upper(opt.get("option_permission"))

    if gate == "No new exposure":
        return "No new exposure", "No vehicle", "Risk gate blocks new exposure."
    if gate == "Tiny paper only":
        return "Tiny stock/ETF paper only", "Stock/ETF", "Risk gate requires reduced/tiny research sizing."
    if gate in {"Source review first", "News confirmation first", "Monitor first"}:
        return "Wait for confirmation", "No vehicle", f"{gate}; do not upgrade from score alone."
    if "RISK FIRST" in plan_u or "NO NEW EXPOSURE" in plan_u:
        return "No new exposure", "No vehicle", "Horizon plan says risk first."
    if "TINY" in plan_u:
        return "Tiny stock/ETF paper only", "Stock/ETF", "Horizon plan allows only tiny paper research."
    if "CALL" in option_side or "CALL" in option_route or "CALL" in permission:
        if horizon == "Long-term":
            return "Stock/ETF research first; call only as tactical overlay", "Stock/ETF + optional call watch", "Long-term thesis should not be expressed mainly through short-dated options."
        return "Defined-risk call spread research", "Call", "Call route only after risk, event, liquidity, and price gates clear."
    if "PUT" in option_side or "HEDGE" in option_route or "PUT" in permission or "HEDGE" in permission:
        return "Put spread or protective hedge research", "Put/Hedge", "Use only as risk-linked hedge or bearish research sleeve."
    return "Stock/ETF research only", "Stock/ETF", "No clean options edge; use equity research context."


def route_rank(row: pd.Series) -> int:
    gate = upper(row.get("gate_status"))
    vehicle = upper(row.get("vehicle_type"))
    score = safe_float(row.get("horizon_score"), 0.0)
    base = {
        "NO NEW EXPOSURE": 0,
        "TINY PAPER ONLY": 1,
        "SOURCE REVIEW FIRST": 2,
        "NEWS CONFIRMATION FIRST": 3,
        "MONITOR FIRST": 4,
        "RESEARCH REVIEW ALLOWED": 5,
    }.get(gate, 6)
    vehicle_bonus = 0 if vehicle in {"NONE", "NO VEHICLE"} else 1 if "STOCK" in vehicle else 2
    return int(base * 100 - vehicle_bonus * 5 - score)


def build_router() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sector_route = normalize_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    sector_opt = one_by_ticker(read_csv_safe(IN_SECTOR_OPTION))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    cards = one_by_ticker(read_csv_safe(IN_CARDS))
    room = one_by_ticker(read_csv_safe(IN_ROOM))
    queue = one_by_ticker(read_csv_safe(IN_QUEUE))
    event_rel = read_csv_safe(IN_EVENT_RELIABILITY)
    if not event_rel.empty and "target_ticker" in event_rel.columns:
        event_rel = event_rel.copy()
        event_rel["target_ticker"] = event_rel["target_ticker"].astype(str).str.upper().str.strip()
        event_rel_by_ticker = event_rel.drop_duplicates("target_ticker", keep="first").set_index("target_ticker")
    else:
        event_rel_by_ticker = pd.DataFrame()

    tickers = list(dict.fromkeys(
        list(sector_route["ticker"].dropna().astype(str).str.upper()) if not sector_route.empty else []
        + list(cards.index if not cards.empty else [])
        + list(room.index if not room.empty else [])
        + list(queue.index if not queue.empty else [])
    ))

    matrix_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []

    sr_by_ticker = one_by_ticker(sector_route) if not sector_route.empty else pd.DataFrame()
    for ticker in tickers:
        sr = row_at(sr_by_ticker, ticker)
        so = row_at(sector_opt, ticker)
        opt = row_at(options, ticker)
        card = row_at(cards, ticker)
        dr = row_at(room, ticker)
        q = row_at(queue, ticker)
        rel = row_at(event_rel_by_ticker, ticker)

        risk_action = first_nonempty(sr.get("risk_action"), opt.get("final_risk_action"), q.get("risk_action"), dr.get("risk_gate"))
        event_gate = first_nonempty(sr.get("event_gate"), opt.get("event_gate"), q.get("event_gate"), dr.get("event_gate"))
        news_status = first_nonempty(card.get("top_news_reliability_status"), dr.get("top_news_reliability_status"), rel.get("calibrated_reliability_status"), q.get("event_reliability_status"))
        news_action = first_nonempty(card.get("top_news_calibrated_action"), dr.get("top_news_calibrated_action"), rel.get("calibrated_research_action"), q.get("event_reliability_action"))
        monitor_count = first_nonempty(card.get("monitor_alert_count"), dr.get("monitor_alert_count"), q.get("monitor_event_count"))
        gate = gate_level(risk_action, event_gate, news_status, monitor_count)
        call = call_status(opt, so, risk_action, event_gate)
        put = put_status(opt, so, risk_action, event_gate)
        sector = first_nonempty(sr.get("sector"), card.get("theme"), dr.get("sector"), q.get("sector"))
        option_side = first_nonempty(so.get("option_side"), opt.get("option_side"), card.get("option_side_after_clear"), dr.get("option_side_after_clear"))
        option_structure = first_nonempty(so.get("option_structure"), opt.get("option_structure"), card.get("vehicle_after_clear"), dr.get("option_structure_after_clear"))
        trigger = first_nonempty(card.get("trigger_to_watch"), dr.get("trigger_to_watch"), opt.get("call_trigger"), so.get("call_trigger"), sr.get("what_to_watch"))
        invalidation = first_nonempty(card.get("invalidation"), dr.get("invalidation"), opt.get("option_invalidation"), so.get("put_trigger"))
        blocker = first_nonempty(card.get("main_blocker"), dr.get("main_blocker"), sr.get("primary_blocker"), opt.get("primary_blocker"), q.get("workflow_bucket"))
        no_go = first_nonempty(opt.get("no_go_conditions"), so.get("no_go_conditions"), dr.get("no_go_conditions"))
        stack_items = gate_stack(risk_action, event_gate, news_status, news_action, monitor_count, no_go)
        override_stack = " > ".join(stack_items)
        checklist = unlock_checklist(risk_action, event_gate, news_status, news_action, monitor_count, trigger, no_go)
        use_case = option_use_case(call, put, gate)
        research_question = primary_research_question(gate, call, put, news_status)

        horizon_actions: dict[str, str] = {}
        horizon_vehicles: dict[str, str] = {}
        horizon_scores: dict[str, float] = {}
        for horizon, window, score_col, decision_col, card_col in HORIZONS:
            score = safe_float(sr.get(score_col), np.nan)
            plan = first_nonempty(sr.get(decision_col), card.get(card_col), dr.get(card_col), q.get("sector_adjusted_action"))
            clear_action, vehicle, reason = vehicle_for_horizon(
                horizon,
                plan,
                risk_action,
                event_gate,
                opt,
                so,
                news_status,
                monitor_count,
            )
            horizon_actions[horizon] = clear_action
            horizon_vehicles[horizon] = vehicle
            horizon_scores[horizon] = score
            confidence = route_confidence(score, gate, news_status, monitor_count, vehicle)
            matrix_rows.append({
                "ticker": ticker,
                "sector": sector,
                "horizon": horizon,
                "time_window": window,
                "horizon_score": round(score, 2) if np.isfinite(score) else np.nan,
                "horizon_plan_from_model": plan,
                "clear_action": clear_action,
                "vehicle_type": vehicle,
                "permission_state": action_permission_state(gate, vehicle),
                "route_confidence_score": confidence,
                "call_status": call,
                "put_status": put,
                "option_use_case": use_case,
                "option_side": option_side,
                "option_structure": option_structure,
                "option_expiry_bucket": opt.get("option_expiry_bucket", ""),
                "gate_status": gate,
                "risk_action": risk_action,
                "event_gate": event_gate,
                "news_reliability_status": news_status,
                "news_calibrated_action": news_action,
                "monitor_event_count": safe_float(monitor_count, 0.0),
                "trigger_to_watch": trigger,
                "invalidation": invalidation,
                "main_blocker": blocker,
                "override_stack": override_stack,
                "required_confirmations": checklist,
                "primary_research_question": research_question,
                "timeframe_fit_warning": timeframe_fit_warning(horizon, vehicle, gate),
                "why_this_route": shorten(reason),
                "source_files": source_trace(
                    sr.get("source_file"),
                    opt.get("source_file"),
                    so.get("source_file"),
                    card.get("top_news_calibration_source"),
                    q.get("source_files"),
                    "event_signal_reliability_watchlist.csv" if news_action else "",
                ),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

        best_horizon = first_nonempty(sr.get("best_horizon_after_sector"), q.get("best_horizon"))
        if not best_horizon:
            valid_scores = {h: s for h, s in horizon_scores.items() if np.isfinite(s)}
            best_horizon = max(valid_scores, key=valid_scores.get) if valid_scores else "Needs review"
        consensus = horizon_consensus(horizon_actions, horizon_vehicles)
        confidence_values = [
            route_confidence(score, gate, news_status, monitor_count, horizon_vehicles.get(horizon, ""))
            for horizon, score in horizon_scores.items()
        ]
        decision_depth_score = round(float(np.nanmean(confidence_values)), 1) if confidence_values else 0.0
        summary_rows.append({
            "ticker": ticker,
            "sector": sector,
            "gate_status": gate,
            "best_horizon": best_horizon,
            "horizon_consensus": consensus,
            "decision_depth_score": decision_depth_score,
            "next_best_action": first_nonempty(q.get("what_to_do"), card.get("decision_now"), dr.get("decision_now")),
            "primary_research_question": research_question,
            "short_action": horizon_actions.get("Short-term", ""),
            "short_vehicle": horizon_vehicles.get("Short-term", ""),
            "medium_action": horizon_actions.get("Medium-term", ""),
            "medium_vehicle": horizon_vehicles.get("Medium-term", ""),
            "long_action": horizon_actions.get("Long-term", ""),
            "long_vehicle": horizon_vehicles.get("Long-term", ""),
            "current_desk_action": first_nonempty(card.get("decision_now"), dr.get("decision_now"), q.get("what_to_do")),
            "option_route": first_nonempty(sr.get("option_route"), so.get("option_route"), card.get("primary_route"), dr.get("route_after_gates_clear")),
            "option_side": option_side,
            "call_status": call,
            "put_status": put,
            "option_use_case": use_case,
            "risk_action": risk_action,
            "event_gate": event_gate,
            "news_reliability_status": news_status,
            "news_calibrated_action": news_action,
            "top_news_headline": first_nonempty(card.get("top_news_headline"), dr.get("top_news_headline"), rel.get("headline"), q.get("event_calibration_headline")),
            "trigger_to_watch": trigger,
            "main_blocker": blocker,
            "override_stack": override_stack,
            "unlock_checklist": checklist,
            "plain_english": shorten(
                f"{ticker}: {gate}. Best horizon is {best_horizon}. "
                f"{research_question} "
                f"Short={horizon_actions.get('Short-term', 'N/A')}; "
                f"Medium={horizon_actions.get('Medium-term', 'N/A')}; "
                f"Long={horizon_actions.get('Long-term', 'N/A')}. "
                f"Call: {call} Put: {put}"
            ),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        option_rows.append({
            "ticker": ticker,
            "sector": sector,
            "gate_status": gate,
            "option_route": first_nonempty(sr.get("option_route"), so.get("option_route"), opt.get("option_answer"), card.get("primary_route")),
            "option_permission": opt.get("option_permission", ""),
            "option_side": option_side,
            "option_structure": option_structure,
            "option_expiry_bucket": opt.get("option_expiry_bucket", ""),
            "call_status": call,
            "put_status": put,
            "option_use_case": use_case,
            "call_allowed_now": "Yes" if call.startswith("Defined-risk call") and gate == "Research review allowed" else "No",
            "put_or_hedge_allowed_now": "Yes" if put.startswith("Put spread") and gate in {"Research review allowed", "Monitor first"} else "No",
            "call_trigger": first_nonempty(opt.get("call_trigger"), so.get("call_trigger"), trigger),
            "put_trigger": first_nonempty(opt.get("put_trigger"), so.get("put_trigger")),
            "option_invalidation": first_nonempty(opt.get("option_invalidation"), invalidation),
            "no_go_conditions": no_go,
            "what_would_change": first_nonempty(opt.get("what_would_change"), so.get("option_reason"), sr.get("what_would_change"), q.get("what_would_change")),
            "call_unlock_checklist": checklist,
            "put_or_hedge_unlock_checklist": checklist,
            "override_stack": override_stack,
            "risk_action": risk_action,
            "event_gate": event_gate,
            "news_reliability_status": news_status,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    matrix = pd.DataFrame(matrix_rows)
    summary = pd.DataFrame(summary_rows)
    option_board = pd.DataFrame(option_rows)
    if not matrix.empty:
        matrix["_rank"] = matrix.apply(route_rank, axis=1)
        matrix = matrix.sort_values(["_rank", "ticker", "horizon"]).drop(columns=["_rank"]).reset_index(drop=True)
    if not summary.empty:
        gate_order = {
            "No new exposure": 0,
            "Tiny paper only": 1,
            "Source review first": 2,
            "News confirmation first": 3,
            "Monitor first": 4,
            "Research review allowed": 5,
        }
        summary["_gate_rank"] = summary["gate_status"].map(gate_order).fillna(9)
        summary = summary.sort_values(["_gate_rank", "ticker"]).drop(columns=["_gate_rank"]).reset_index(drop=True)
    if not option_board.empty:
        option_board = option_board.sort_values(["gate_status", "ticker"]).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "status": "READY" if len(summary) else "NO_ROWS",
        "tickers": int(len(summary)),
        "matrix_rows": int(len(matrix)),
        "option_board_rows": int(len(option_board)),
        "no_new_exposure_count": int((summary.get("gate_status", pd.Series(dtype=str)) == "No new exposure").sum()) if not summary.empty else 0,
        "tiny_paper_only_count": int((summary.get("gate_status", pd.Series(dtype=str)) == "Tiny paper only").sum()) if not summary.empty else 0,
        "source_or_news_review_count": int(summary.get("gate_status", pd.Series(dtype=str)).isin(["Source review first", "News confirmation first"]).sum()) if not summary.empty else 0,
        "call_watch_count": int(summary.get("call_status", pd.Series(dtype=str)).astype(str).str.contains("Defined-risk call|Call edge exists", case=False, na=False).sum()) if not summary.empty else 0,
        "call_blocked_now_count": int(summary.get("call_status", pd.Series(dtype=str)).astype(str).str.contains("No call now|No call\\.", case=False, na=False).sum()) if not summary.empty else 0,
        "put_or_hedge_count": int(summary.get("put_status", pd.Series(dtype=str)).astype(str).str.contains("Put spread|protective hedge|hedge only", case=False, na=False).sum()) if not summary.empty else 0,
        "avg_decision_depth_score": round(float(pd.to_numeric(summary.get("decision_depth_score", pd.Series(dtype=float)), errors="coerce").mean()), 1) if not summary.empty else 0.0,
        "same_route_all_horizons_count": int(summary.get("horizon_consensus", pd.Series(dtype=str)).astype(str).str.contains("Same route", na=False).sum()) if not summary.empty else 0,
        "mixed_horizon_count": int(len(summary) - summary.get("horizon_consensus", pd.Series(dtype=str)).astype(str).str.contains("Same route", na=False).sum()) if not summary.empty else 0,
        "outputs": {
            "matrix": OUT_MATRIX.name,
            "summary": OUT_SUMMARY.name,
            "option_board": OUT_OPTION_BOARD.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return matrix, summary, option_board, state


def main() -> int:
    matrix, summary, option_board, state = build_router()
    matrix.to_csv(OUT_MATRIX, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    option_board.to_csv(OUT_OPTION_BOARD, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Tickers: {state.get('tickers', 0)}",
        f"- Matrix rows: {state.get('matrix_rows', 0)}",
        f"- No-new-exposure rows: {state.get('no_new_exposure_count', 0)}",
        f"- Tiny-paper-only rows: {state.get('tiny_paper_only_count', 0)}",
        f"- Source/news review rows: {state.get('source_or_news_review_count', 0)}",
        f"- Call watch rows: {state.get('call_watch_count', 0)}",
        f"- Call blocked now rows: {state.get('call_blocked_now_count', 0)}",
        f"- Put/hedge rows: {state.get('put_or_hedge_count', 0)}",
        f"- Avg decision depth score: {state.get('avg_decision_depth_score', 0)}",
        f"- Same-route all horizons: {state.get('same_route_all_horizons_count', 0)}",
        f"- Mixed-horizon rows: {state.get('mixed_horizon_count', 0)}",
        "",
        "## Horizon Vehicle Summary",
        "",
        df_to_markdown(summary, max_rows=90),
        "",
        "## Horizon Vehicle Matrix",
        "",
        df_to_markdown(matrix, max_rows=180),
        "",
        "## Option Route Clarity Board",
        "",
        df_to_markdown(option_board, max_rows=120),
        "",
        "## Product Truth",
        "",
        "This is a research router. It does not approve trades, route orders, or connect to a broker.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 167 - Horizon and Vehicle Router", sections)

    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"wrote {OUT_MATRIX.name} rows={len(matrix)}")
    print(f"wrote {OUT_OPTION_BOARD.name} rows={len(option_board)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
