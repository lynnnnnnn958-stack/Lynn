#!/usr/bin/env python3
"""
Canyon v9 Step 195 - Institutional Promotion Gate.

Research-only. No broker connection. No live orders.

This step turns the existing research desks into one final PM decision layer.
It does not add another shallow score. It answers, per ticker:

1. What are we allowed to do now?
2. What is the short / medium / long-term read?
3. Is stock / ETF, call, put, or hedge research allowed?
4. What is the first blocker?
5. What proof would unlock the next step?

Outputs:
  institutional_promotion_gate_state.json
  institutional_promotion_gate.csv
  institutional_ticker_drilldown_cards.csv
  institutional_horizon_route_matrix.csv
  institutional_vehicle_permission_matrix.csv
  institutional_promotion_queue.csv
  institutional_promotion_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "institutional_promotion_gate_state.json"
OUT_GATE = ROOT / "institutional_promotion_gate.csv"
OUT_DRILLDOWN = ROOT / "institutional_ticker_drilldown_cards.csv"
OUT_HORIZON = ROOT / "institutional_horizon_route_matrix.csv"
OUT_VEHICLE = ROOT / "institutional_vehicle_permission_matrix.csv"
OUT_QUEUE = ROOT / "institutional_promotion_queue.csv"
OUT_REPORT = ROOT / "institutional_promotion_report.md"

HORIZONS = [
    ("Short-term", "0-10 trading days", "short_term_plan"),
    ("Medium-term", "2-8 weeks", "medium_term_plan"),
    ("Long-term", "3-12 months", "long_term_plan"),
]


def as_text(value: Any, default: str = "") -> str:
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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    value_float = safe_float(value, np.nan)
    if not np.isfinite(value_float):
        return default
    return int(value_float)


def shorten(value: Any, limit: int = 260) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def plain(value: Any) -> str:
    text = as_text(value, "No data")
    replacements = {
        "ALREADY_CLOSED_DO_NOT_REPEAT": "already closed; do not repeat",
        "BLOCKED": "blocked",
        "CALL_RESEARCH_ONLY": "call research only",
        "CAUSAL_REVIEW_REQUIRED": "causal links need review",
        "CLEAR": "clear",
        "DATA_GAP": "missing data",
        "DOWNSTREAM_BENEFICIARY": "customer or downstream winner",
        "NO_DATA": "no data",
        "NO_GO": "not allowed",
        "NO_LIVE_ORDERS": "no live orders",
        "NOT_IN_RISK_BOOK_REVIEW": "not in risk book; needs review",
        "PAPER_ONLY": "paper only",
        "PEER_READ_THROUGH": "related peer",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "PUT_RESEARCH_ONLY": "put research only",
        "REDUCE_ONLY": "reduce only",
        "RESEARCH_ONLY": "research only",
        "RISK_BLOCKED": "risk blocks action",
        "RISK_REDUCTION_FIRST": "reduce risk first",
        "RISK_REDUCTION_ONLY": "risk reduction only",
        "SIZE_DOWN": "use smaller size",
        "STOCK_OR_ETF_RESEARCH_ONLY": "stock or ETF research only",
        "TINY_PAPER_ONLY": "tiny paper only",
        "TINY_STOCK_OR_ETF_PAPER_ONLY": "tiny stock or ETF paper only",
        "UNKNOWN_NEEDS_DATA": "unknown; needs data",
        "UPSTREAM_BENEFICIARY": "supplier or upstream winner",
        "VULNERABLE_TARGET": "possible loser",
        "WAIT_EVENT_PROOF_FIRST": "wait for event proof first",
        "WATCH_EVENT_PROOF_FIRST": "watch event proof first",
        "WATCH_FOR_CONFIRMATION": "watch for confirmation",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = text.replace("_", " ")
    cleanup = {
        "NOT IN RISK BOOK needs review": "not in risk book; needs review",
        "UNKNOWN needs data": "unknown; needs data",
    }
    for raw, friendly in cleanup.items():
        text = text.replace(raw, friendly)
    return " ".join(text.split())


def first_row_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""]
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby(ticker_col, sort=False)}


def option_rows_by_ticker(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    return {ticker: grp.copy() for ticker, grp in work.groupby("ticker", sort=False)}


def risk_gate_read(risk_row: pd.Series | None) -> tuple[str, str, float]:
    if risk_row is None or risk_row.empty:
        return "research only", "Ticker is not in the risk book yet.", 0.0
    action = as_text(risk_row.get("final_risk_action"), "REVIEW").upper()
    allowed = safe_float(risk_row.get("recommended_risk_weight_pct"), np.nan)
    current = safe_float(risk_row.get("current_weight_pct"), np.nan)
    if "SEED" in action:
        return "review only", "Risk seed exists, but manual approval is still required before sizing.", 0.0
    if "REDUCE" in action:
        return "risk reduction first", "Risk says reduce only. Do not add exposure.", 0.0
    if "SIZE" in action:
        max_weight = allowed if np.isfinite(allowed) else 0.25
        return "tiny paper review only", "Risk says use smaller size before any idea can move forward.", max_weight
    if "CLEAR" in action:
        max_weight = allowed if np.isfinite(allowed) else min(current if np.isfinite(current) else 1.0, 1.0)
        return "research usable after checks", "Risk is not the first blocker, but other gates still apply.", max_weight
    return "review only", "Risk needs review before sizing.", 0.0


def option_permission_for_ticker(option_rows: pd.DataFrame) -> dict[str, str]:
    if option_rows.empty:
        return {
            "call_permission": "Blocked. No option route file for this ticker.",
            "put_permission": "Blocked. No option route file for this ticker.",
            "hedge_permission": "Blocked. No option route file for this ticker.",
            "option_summary": "No option route evidence.",
        }

    decisions = " ".join(option_rows.get("final_vehicle_decision", pd.Series(dtype=str)).astype(str).tolist()).upper()
    sides = " ".join(option_rows.get("final_option_side", pd.Series(dtype=str)).astype(str).tolist()).upper()
    no_go_count = int(pd.to_numeric(option_rows.get("no_go_count", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
    reason = shorten(option_rows.get("no_go_reasons", pd.Series([""])).astype(str).iloc[0], 240)

    call = "Blocked. Calls need risk, proof, spread, and price confirmation."
    put = "Blocked. Puts need risk, proof, spread, and price confirmation."
    hedge = "Blocked. Hedge research needs manual spread and risk proof."

    if "CALL" in decisions or "CALL" in sides:
        call = "Call research only after all gates clear."
    if "PUT" in decisions or "PUT" in sides or "HEDGE" in decisions:
        put = "Put research only after all gates clear."
        hedge = "Hedge research only after all gates clear."
    if no_go_count > 0:
        call = "Blocked. Option no-go checks are still open."
        if "PUT" in sides or "HEDGE" in decisions:
            put = "Hedge or put research only; no trade permission."
            hedge = "Hedge research only; manual quote and risk proof required."

    return {
        "call_permission": call,
        "put_permission": put,
        "hedge_permission": hedge,
        "option_summary": f"{no_go_count} option blocker(s). {plain(reason)}",
    }


def choose_final_answer(
    ticker: str,
    risk_status: str,
    readiness: pd.Series | None,
    portfolio: pd.Series | None,
    execution: pd.Series | None,
    news: pd.Series | None,
    master: pd.Series | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    next_step = "Read the ticker drilldown and fill missing proof."
    click = "Home"
    permission = "Study only"
    route = "Wait for confirmation"
    max_weight = 0.0

    risk_action = risk_status.lower()
    if "risk reduction" in risk_action:
        blockers.append("Risk says reduce only")
        permission = "Do not add"
        route = "Risk reduction first"
        next_step = "Open Risk and reduce or explain the risk breach before any new idea."
        click = "Risk"
    elif "tiny" in risk_action:
        blockers.append("Risk says smaller size")
        permission = "Tiny paper review only"
        route = "Stock or ETF research only"
        click = "Risk"
    elif "research only" in risk_action:
        blockers.append("Ticker is not in the risk book yet")
        permission = "Study only"
        route = "Research only until risk book entry exists"
        next_step = "Create or verify the risk-book entry before judging size, calls, puts, or hedges."
        click = "Risk"
    elif "review only" in risk_action:
        blockers.append("Risk needs manual review")
        permission = "Study only"
        route = "Research only until risk review clears"
        next_step = "Open Risk and resolve the risk review before route promotion."
        click = "Risk"

    if readiness is not None and not readiness.empty:
        first_gate = plain(readiness.get("first_blocking_gate"))
        gate_status = plain(readiness.get("first_gate_status"))
        if "blocked" in gate_status.lower() or "risk" in gate_status.lower():
            blockers.append(first_gate)
            if permission != "Do not add":
                permission = "Study only"
            next_step = as_text(readiness.get("first_clear_condition"), next_step)
            click = "Today"

    if portfolio is not None and not portfolio.empty:
        decision = plain(portfolio.get("portfolio_v2_decision"))
        confidence = safe_float(portfolio.get("confidence_0_100"), 0.0)
        max_weight = max(max_weight, safe_float(portfolio.get("robust_weight_v2_pct"), 0.0))
        if "no new" in decision.lower() or confidence < 50:
            blockers.append("Optimizer does not allow new exposure")
            permission = "Do not add" if "risk" in " ".join(blockers).lower() else "Study only"
            next_step = as_text(portfolio.get("what_would_unlock"), next_step)
            click = "Performance"

    if execution is not None and not execution.empty:
        exec_perm = plain(execution.get("execution_permission"))
        exec_status = plain(execution.get("execution_status"))
        if "no new" in exec_perm.lower():
            blockers.append("Trading cost or liquidity blocks new exposure")
            permission = "Do not add"
            route = "No new exposure"
            next_step = as_text(execution.get("what_to_do"), next_step)
            click = "Performance"
        elif "manual quote" in exec_perm.lower():
            blockers.append("Manual spread and liquidity check needed")
            if permission not in {"Do not add"}:
                permission = "Study only"
            next_step = as_text(execution.get("what_to_do"), next_step)
            click = "Performance"
        elif exec_status:
            blockers.append(f"Execution: {exec_status}")

    if news is not None and not news.empty:
        open_items = safe_int(news.get("open_proof_items"), 0)
        contradicted = safe_int(news.get("contradicted_edges"), 0)
        if contradicted:
            blockers.append("News story contradicts price action")
            permission = "Study only"
            route = "Do not trade the headline"
            next_step = as_text(news.get("proof_needed"), "Resolve the contradicted news link first.")
            click = "News"
        elif open_items > 0:
            blockers.append("News link still needs proof")
            if permission not in {"Do not add"}:
                permission = "Study only"
            next_step = as_text(news.get("proof_needed"), "Prove the news-to-stock link before action.")
            click = "News"

    if master is not None and not master.empty:
        master_action = plain(master.get("master_action"))
        if "tiny" in master_action.lower() and permission == "Watch list":
            permission = "Tiny paper review only"
            route = "Tiny stock or ETF paper review"

    if not blockers:
        permission = "Watch list"
        route = "Research usable after final manual check"
        next_step = "Check trigger, invalidation, spread, news proof, and risk one more time."
        click = "Ideas"
    else:
        first = blockers[0]
        if first == "Risk says reduce only":
            next_step = "Open Risk and reduce or explain the risk breach before any new idea."
            click = "Risk"
        elif first == "Risk says smaller size":
            next_step = "Open Risk and confirm the smaller allowed size before any paper review."
            click = "Risk"
        elif first == "Ticker is not in the risk book yet":
            next_step = "Create or verify the risk-book entry before judging size, calls, puts, or hedges."
            click = "Risk"
        elif first == "Risk needs manual review":
            next_step = "Open Risk and resolve the risk review before route promotion."
            click = "Risk"
        elif "Price/volume" in first:
            click = "Today"
        elif "News" in first:
            click = "News"

    if permission == "Do not add":
        max_weight = 0.0
        route = "No new exposure"
    elif permission == "Study only":
        max_weight = 0.0
    elif permission == "Tiny paper review only":
        max_weight = min(max_weight if max_weight > 0 else 0.25, 0.25)

    return {
        "final_permission": permission,
        "primary_route_now": route,
        "max_paper_weight_pct": round(max_weight, 4),
        "first_blocker": blockers[0] if blockers else "Final manual check",
        "blocker_count": len(blockers),
        "why_now": "; ".join(dict.fromkeys(blockers)) if blockers else "No hard blocker found, but this remains research-only.",
        "next_step": shorten(next_step, 300),
        "where_to_click": click,
    }


def stock_permission(final_permission: str, risk_status: str) -> str:
    if final_permission == "Do not add":
        return "No new stock or ETF. Reduce or observe only."
    if final_permission == "Tiny paper review only":
        return "Tiny stock or ETF paper review only after proof."
    if "tiny" in risk_status.lower():
        return "Tiny stock or ETF paper review only."
    return "Study first; manual approval needed before paper sizing."


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ticker_cards = read_csv_safe(ROOT / "ticker_decision_cards.csv")
    readiness = read_csv_safe(ROOT / "action_readiness_detail_cards.csv")
    risk = read_csv_safe(ROOT / "final_risk_gate.csv")
    risk_seed = read_csv_safe(ROOT / "risk_book_seed_entries.csv")
    if not risk_seed.empty and "ticker" in risk_seed.columns:
        if risk.empty:
            risk = risk_seed.copy()
        else:
            existing_tickers = set(risk.get("ticker", pd.Series(dtype=str)).dropna().map(clean_ticker).tolist())
            add_seed = risk_seed[~risk_seed["ticker"].map(clean_ticker).isin(existing_tickers)].copy()
            if not add_seed.empty:
                risk = pd.concat([risk, add_seed], ignore_index=True)
    portfolio = read_csv_safe(ROOT / "depth5_portfolio_optimizer_v2.csv")
    execution = read_csv_safe(ROOT / "depth5_execution_liquidity_desk.csv")
    news = read_csv_safe(ROOT / "depth5_news_causal_proof_system.csv")
    options = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    master = read_csv_safe(ROOT / "master_10_layer_decision_matrix_v2.csv")

    ticker_map = first_row_by_ticker(ticker_cards)
    readiness_map = first_row_by_ticker(readiness)
    risk_map = first_row_by_ticker(risk)
    portfolio_map = first_row_by_ticker(portfolio)
    execution_map = first_row_by_ticker(execution)
    news_map = first_row_by_ticker(news)
    master_map = first_row_by_ticker(master)
    options_map = option_rows_by_ticker(options)

    tickers = sorted(set(ticker_map) | set(readiness_map) | set(risk_map) | set(portfolio_map) | set(execution_map) | set(news_map) | set(master_map) | set(options_map))
    gate_rows: list[dict[str, Any]] = []
    drilldown_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    vehicle_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []

    for ticker in tickers:
        trow = ticker_map.get(ticker, pd.Series(dtype=object))
        rrow = risk_map.get(ticker, pd.Series(dtype=object))
        ready_row = readiness_map.get(ticker, pd.Series(dtype=object))
        port_row = portfolio_map.get(ticker, pd.Series(dtype=object))
        exec_row = execution_map.get(ticker, pd.Series(dtype=object))
        news_row = news_map.get(ticker, pd.Series(dtype=object))
        master_row = master_map.get(ticker, pd.Series(dtype=object))
        opt_rows = options_map.get(ticker, pd.DataFrame())

        risk_status, risk_reason, risk_max = risk_gate_read(rrow)
        decision = choose_final_answer(ticker, risk_status, ready_row, port_row, exec_row, news_row, master_row)
        decision["max_paper_weight_pct"] = min(max(decision["max_paper_weight_pct"], 0.0), risk_max if risk_max > 0 else decision["max_paper_weight_pct"])
        option_perm = option_permission_for_ticker(opt_rows)

        score_inputs = [
            safe_float(trow.get("decision_quality_score"), np.nan),
            safe_float(ready_row.get("readiness_score"), np.nan),
            safe_float(port_row.get("confidence_0_100"), np.nan),
            safe_float(news_row.get("causal_confidence_0_100"), np.nan),
            safe_float(master_row.get("stack_score_avg"), np.nan),
        ]
        available_scores = [x for x in score_inputs if np.isfinite(x)]
        confidence = float(np.mean(available_scores)) if available_scores else 0.0
        confidence -= min(decision["blocker_count"] * 7.5, 35.0)
        confidence = round(float(np.clip(confidence, 0.0, 100.0)), 1)

        sector = as_text(trow.get("theme"), as_text(rrow.get("sector"), as_text(port_row.get("sector"), "Unknown")))
        headline = as_text(news_row.get("top_headline"), as_text(trow.get("top_news_headline"), "No mapped headline"))
        trigger = as_text(trow.get("trigger_to_watch"), as_text(ready_row.get("trigger_to_watch"), "Wait for price confirmation."))
        invalidation = as_text(trow.get("invalidation"), "If price/news/risk contradicts the idea, stop reviewing.")

        final_gate = {
            "ticker": ticker,
            "sector_or_theme": sector,
            "final_permission": decision["final_permission"],
            "primary_route_now": decision["primary_route_now"],
            "confidence_0_100": confidence,
            "max_paper_weight_pct": decision["max_paper_weight_pct"],
            "first_blocker": decision["first_blocker"],
            "why_now": decision["why_now"],
            "next_step": decision["next_step"],
            "where_to_click": decision["where_to_click"],
            "risk_status": risk_status,
            "risk_reason": risk_reason,
            "news_headline": headline,
            "trigger_to_watch": trigger,
            "invalidation": invalidation,
            "source_files": "ticker_decision_cards.csv; action_readiness_detail_cards.csv; final_risk_gate.csv; depth5_*; options_execution_route_matrix.csv; event_readthrough_target_ranking.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        gate_rows.append(final_gate)

        stock_perm = stock_permission(decision["final_permission"], risk_status)
        vehicle_rows.append({
            "ticker": ticker,
            "stock_or_etf": stock_perm,
            "call": option_perm["call_permission"] if decision["final_permission"] not in {"Do not add"} else "Blocked. Calls cannot override risk.",
            "put": option_perm["put_permission"],
            "hedge": option_perm["hedge_permission"],
            "option_reason": option_perm["option_summary"],
            "max_paper_weight_pct": decision["max_paper_weight_pct"],
            "source_files": "options_execution_route_matrix.csv; final_risk_gate.csv; depth5_execution_liquidity_desk.csv",
        })

        short_answer = as_text(trow.get("short_term_plan"), "Wait for confirmation.")
        medium_answer = as_text(trow.get("medium_term_plan"), "Wait for confirmation.")
        long_answer = as_text(trow.get("long_term_plan"), "Wait for confirmation.")
        horizon_plan_map = {
            "Short-term": short_answer,
            "Medium-term": medium_answer,
            "Long-term": long_answer,
        }
        for horizon, time_window, _ in HORIZONS:
            opt = pd.Series(dtype=object)
            if not opt_rows.empty and "horizon" in opt_rows.columns:
                match = opt_rows[opt_rows["horizon"].astype(str).str.lower() == horizon.lower()]
                if not match.empty:
                    opt = match.iloc[0]
            route_decision = plain(opt.get("final_vehicle_decision", decision["primary_route_now"]))
            side = plain(opt.get("final_option_side", "None"))
            no_go = safe_int(opt.get("no_go_count"), 0)
            if decision["final_permission"] == "Do not add":
                route_decision = "No new exposure"
            elif no_go > 0:
                route_decision = f"{route_decision}. Option blockers still open."
            horizon_rows.append({
                "ticker": ticker,
                "horizon": horizon,
                "time_window": time_window,
                "plain_view": plain(horizon_plan_map[horizon]),
                "allowed_vehicle": route_decision,
                "option_side": side,
                "why_this_horizon": shorten(plain(opt.get("why_this_route", decision["why_now"])), 260),
                "trigger_to_watch": trigger,
                "invalidation": invalidation,
                "source_files": "ticker_decision_cards.csv; options_execution_route_matrix.csv",
            })

        drilldown_rows.append({
            "ticker": ticker,
            "top_answer": f"{ticker}: {decision['final_permission']}. {decision['primary_route_now']}.",
            "why": decision["why_now"],
            "short_term": plain(short_answer),
            "medium_term": plain(medium_answer),
            "long_term": plain(long_answer),
            "stock_or_etf": stock_perm,
            "call": vehicle_rows[-1]["call"],
            "put": vehicle_rows[-1]["put"],
            "hedge": vehicle_rows[-1]["hedge"],
            "first_blocker": decision["first_blocker"],
            "proof_needed": decision["next_step"],
            "where_to_click": decision["where_to_click"],
            "news_to_read": headline,
            "trigger_to_watch": trigger,
            "invalidation": invalidation,
            "source_files": final_gate["source_files"],
        })

        if decision["final_permission"] in {"Do not add", "Study only"} or decision["blocker_count"] > 0:
            priority = "P1" if decision["final_permission"] == "Do not add" else "P2"
            queue_rows.append({
                "priority": priority,
                "ticker": ticker,
                "work": decision["first_blocker"],
                "why_it_matters": decision["why_now"],
                "next_step": decision["next_step"],
                "where_to_click": decision["where_to_click"],
                "source_files": final_gate["source_files"],
            })

    gate = pd.DataFrame(gate_rows)
    if not gate.empty:
        order = {"Do not add": 0, "Study only": 1, "Tiny paper review only": 2, "Watch list": 3}
        gate["_rank"] = gate["final_permission"].map(order).fillna(4)
        gate = gate.sort_values(["_rank", "confidence_0_100", "ticker"], ascending=[True, False, True]).drop(columns=["_rank"]).reset_index(drop=True)
    drilldown = pd.DataFrame(drilldown_rows)
    horizon = pd.DataFrame(horizon_rows)
    vehicle = pd.DataFrame(vehicle_rows)
    queue = pd.DataFrame(queue_rows)
    if not queue.empty:
        queue["_rank"] = queue["priority"].map({"P1": 1, "P2": 2, "P3": 3}).fillna(4)
        queue = queue.sort_values(["_rank", "ticker"]).drop(columns=["_rank"]).reset_index(drop=True)

    state = {
        "date": today_str(),
        "status": "INSTITUTIONAL_PROMOTION_GATE_ACTIVE",
        "ticker_count": len(gate),
        "do_not_add_count": int((gate["final_permission"] == "Do not add").sum()) if not gate.empty else 0,
        "study_only_count": int((gate["final_permission"] == "Study only").sum()) if not gate.empty else 0,
        "tiny_paper_review_count": int((gate["final_permission"] == "Tiny paper review only").sum()) if not gate.empty else 0,
        "watch_list_count": int((gate["final_permission"] == "Watch list").sum()) if not gate.empty else 0,
        "queue_count": len(queue),
        "paper_allowed_now_count": int((gate["max_paper_weight_pct"] > 0).sum()) if not gate.empty else 0,
        "options_allowed_now_count": 0,
        "plain_answer": "Final gate is active. Most names remain research-only because risk, proof, execution, or news-chain checks still block promotion.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return gate, drilldown, horizon, vehicle, queue, state


def main() -> None:
    gate, drilldown, horizon, vehicle, queue, state = build_outputs()
    for df, path in [
        (gate, OUT_GATE),
        (drilldown, OUT_DRILLDOWN),
        (horizon, OUT_HORIZON),
        (vehicle, OUT_VEHICLE),
        (queue, OUT_QUEUE),
    ]:
        df.to_csv(path, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Final Promotion Gate\n\n" + df_to_markdown(gate.head(30)),
        "## Ticker Drilldown Cards\n\n" + df_to_markdown(drilldown.head(30)),
        "## Horizon Route Matrix\n\n" + df_to_markdown(horizon.head(60)),
        "## Vehicle Permission Matrix\n\n" + df_to_markdown(vehicle.head(30)),
        "## Promotion Queue\n\n" + df_to_markdown(queue.head(40)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 195 - Institutional Promotion Gate", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Tickers: {state['ticker_count']}")
    print(f"[OK] Do not add: {state['do_not_add_count']} | Study only: {state['study_only_count']} | Tiny paper review: {state['tiny_paper_review_count']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
