#!/usr/bin/env python3
"""
Canyon v9 Step 173 - Options Execution Route Engine.

Research-only. No broker connection. No live orders.

This step deepens the weakest option/execution area. It combines:
  - short/medium/long horizon routes,
  - call vs put vs no-option permission,
  - risk gate dominance,
  - options Greeks / IV / gamma heat,
  - execution cost / spread / fill risk,
  - live monitor events,
  - event read-through context,
  - suspicious options backtest sanity checks.

The output is a desk-style route matrix. It never creates orders. Missing or
weak evidence can only block, downsize, or require review.

Outputs:
  options_execution_route_matrix.csv
  options_trade_permission_summary.csv
  options_tca_no_go_audit.csv
  options_execution_playbook.csv
  options_execution_state.json
  options_execution_report.md
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


OUT_MATRIX = ROOT / "options_execution_route_matrix.csv"
OUT_SUMMARY = ROOT / "options_trade_permission_summary.csv"
OUT_AUDIT = ROOT / "options_tca_no_go_audit.csv"
OUT_PLAYBOOK = ROOT / "options_execution_playbook.csv"
OUT_STATE = ROOT / "options_execution_state.json"
OUT_REPORT = ROOT / "options_execution_report.md"

MODEL_ACCOUNT_VALUE = 100000.0


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace(".", "-")


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


def bool_text(value: Any) -> bool:
    return str(value or "").strip().upper() in {"TRUE", "YES", "Y", "1"}


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, dict[str, Any]]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].map(clean_ticker)
    work = work[work[ticker_col] != ""].drop_duplicates(ticker_col, keep="first")
    return work.set_index(ticker_col).to_dict(orient="index")


def load_option_backtest_quality() -> pd.DataFrame:
    bt = read_csv_safe(ROOT / "options_backtest_results.csv")
    if bt.empty or "ticker" not in bt.columns:
        return pd.DataFrame(columns=[
            "ticker", "best_option_backtest_strategy", "best_option_backtest_sharpe",
            "backtest_sanity_status", "backtest_sanity_note",
        ])
    bt = bt.copy()
    bt["ticker"] = bt["ticker"].map(clean_ticker)
    bt["sharpe_num"] = pd.to_numeric(bt.get("sharpe"), errors="coerce")
    bt["ann_return_num"] = pd.to_numeric(bt.get("ann_return"), errors="coerce")
    bt["max_drawdown_num"] = pd.to_numeric(bt.get("max_drawdown"), errors="coerce")
    rows: list[dict[str, Any]] = []
    for ticker, sub in bt.groupby("ticker"):
        if not ticker:
            continue
        best = sub.sort_values("sharpe_num", ascending=False).head(1)
        best_row = best.iloc[0] if not best.empty else pd.Series(dtype=object)
        impossible = bool(
            (sub["ann_return_num"].abs() > 3.0).any()
            or sub["max_drawdown_num"].isna().any()
            or sub["sharpe_num"].abs().gt(8.0).any()
        )
        rows.append({
            "ticker": ticker,
            "best_option_backtest_strategy": as_text(best_row.get("strategy"), "NO_DATA"),
            "best_option_backtest_sharpe": round(safe_float(best_row.get("sharpe_num"), np.nan), 3),
            "backtest_sanity_status": "REVIEW_PROXY_BACKTEST" if impossible else "USABLE_RESEARCH_CONTEXT",
            "backtest_sanity_note": (
                "Option backtest has extreme/NaN metrics; use as a warning label only, not sizing evidence."
                if impossible else
                "Option backtest is usable as research context, still not permission to trade."
            ),
        })
    return pd.DataFrame(rows)


def score_risk(risk_action: str, gate_status: str) -> tuple[float, str]:
    text = f"{risk_action} {gate_status}".upper()
    if "REDUCE_ONLY" in text or "NO NEW EXPOSURE" in text or "NO_NEW" in text:
        return 8.0, "Risk gate blocks bullish exposure."
    if "SIZE_DOWN" in text or "TINY" in text:
        return 38.0, "Risk gate allows at most tiny paper research."
    if "CLEAR" in text or "OK" in text:
        return 76.0, "Risk gate is not the main blocker."
    return 42.0, "Risk gate is unclear; keep review-only."


def score_execution(row: dict[str, Any]) -> tuple[float, list[str]]:
    status = as_upper(row.get("execution_cost_status"), "NO_EXECUTION_DATA")
    base = safe_float(row.get("base_cost_bps"), np.nan)
    stress = safe_float(row.get("stress_cost_bps"), np.nan)
    spread = safe_float(row.get("spread_bps"), np.nan)
    fill = safe_float(row.get("expected_fill_rate_pct"), np.nan)
    liquidity = as_upper(row.get("liquidity_label"), "")
    reasons: list[str] = []

    score = 42.0
    if status == "CLEAR":
        score = 78.0
    elif "REVIEW" in status:
        score = 58.0
        reasons.append("execution status requires review")
    elif "SIZE_DOWN" in status:
        score = 34.0
        reasons.append("execution status says size down")
    elif "BLOCK" in status:
        score = 12.0
        reasons.append("execution status blocks route")
    else:
        reasons.append("execution data incomplete")

    if np.isfinite(base):
        if base > 45:
            score = min(score, 20.0)
            reasons.append(f"base TCA {base:.1f}bps is too high")
        elif base > 25:
            score = min(score, 45.0)
            reasons.append(f"base TCA {base:.1f}bps needs review")
    if np.isfinite(stress):
        if stress > 75:
            score = min(score, 18.0)
            reasons.append(f"stress TCA {stress:.1f}bps is too high")
        elif stress > 45:
            score = min(score, 42.0)
            reasons.append(f"stress TCA {stress:.1f}bps needs review")
    if np.isfinite(spread):
        if spread > 20:
            score = min(score, 20.0)
            reasons.append(f"spread {spread:.1f}bps is too wide")
        elif spread > 10:
            score = min(score, 50.0)
            reasons.append(f"spread {spread:.1f}bps needs manual check")
    else:
        score = min(score, 55.0)
        reasons.append("live spread is missing")
    if np.isfinite(fill) and fill < 80:
        score = min(score, 35.0)
        reasons.append(f"expected fill rate {fill:.1f}% is weak")
    if liquidity in {"LOW", "POOR", "DATA_GAP"}:
        score = min(score, 35.0)
        reasons.append(f"liquidity is {liquidity}")
    return float(np.clip(score, 0, 85)), reasons


def score_greeks(row: dict[str, Any]) -> tuple[float, list[str], str]:
    status = as_upper(row.get("greeks_status"), "NO_GREEKS_DATA")
    iv_rank = safe_float(row.get("iv_rank"), np.nan)
    gamma_score = safe_float(row.get("gamma_score"), np.nan)
    heat = safe_float(row.get("options_heat_score"), np.nan)
    squeeze = bool_text(row.get("squeeze_risk"))
    reasons: list[str] = []
    score = 42.0
    if status == "CLEAR":
        score = 76.0
    elif "REVIEW" in status:
        score = 55.0
        reasons.append("Greeks status requires review")
    elif "BLOCK" in status:
        score = 18.0
        reasons.append("Greeks status blocks option route")
    else:
        reasons.append("Greeks data incomplete")

    option_style = "DEFINED_RISK_SPREAD_ONLY"
    if np.isfinite(iv_rank):
        if iv_rank >= 70:
            score = min(score, 38.0)
            reasons.append(f"IV rank {iv_rank:.1f} is high; avoid naked long premium")
            option_style = "SPREAD_OR_NO_OPTION"
        elif iv_rank <= 25:
            option_style = "DEFINED_RISK_DEBIT_SPREAD_OK_IF_GATES_CLEAR"
    if np.isfinite(gamma_score) and gamma_score >= 70:
        score = min(score, 45.0)
        reasons.append(f"gamma score {gamma_score:.1f} creates pin/squeeze risk")
    if np.isfinite(heat) and heat >= 75:
        score = min(score, 40.0)
        reasons.append(f"options heat {heat:.1f} is crowded")
    if squeeze:
        score = min(score, 42.0)
        reasons.append("squeeze risk is a catalyst only, not permission")
    return float(np.clip(score, 0, 85)), reasons, option_style


def score_monitor(row: dict[str, Any]) -> tuple[float, list[str]]:
    severity = as_upper(row.get("max_monitor_severity", row.get("max_severity")), "NO_MONITOR_DATA")
    price_break = as_upper(row.get("price_break_state"), "NO_DATA")
    volume = as_upper(row.get("volume_spike_state"), "NO_DATA")
    vol = as_upper(row.get("volatility_regime_state"), "NO_DATA")
    spread = as_upper(row.get("spread_status"), "NO_DATA")
    reasons: list[str] = []
    score = 72.0
    if severity == "CRITICAL":
        score = 22.0
        reasons.append("monitor severity is CRITICAL")
    elif severity == "WARNING":
        score = 45.0
        reasons.append("monitor severity is WARNING")
    elif "NO" in severity:
        score = 50.0
        reasons.append("monitor data incomplete")
    for name, value in [("price break", price_break), ("volume spike", volume), ("volatility regime", vol), ("spread", spread)]:
        if value in {"CRITICAL", "WARNING", "DATA_GAP"}:
            reasons.append(f"{name} state is {value}")
            if value == "CRITICAL":
                score = min(score, 20.0)
            elif value == "WARNING":
                score = min(score, 45.0)
            else:
                score = min(score, 55.0)
    return float(np.clip(score, 0, 80)), reasons


def score_event(row: dict[str, Any], horizon_row: dict[str, Any]) -> tuple[float, list[str], str]:
    rel = as_upper(horizon_row.get("news_reliability_status", row.get("news_reliability_status")), "NO_EVENT_DATA")
    top_decision = as_upper(row.get("top_decision"), "")
    directional = as_upper(row.get("directional_route"), "")
    reasons: list[str] = []
    score = 50.0
    if "RELIABLE" in rel or "ENOUGH" in rel:
        score = 72.0
    elif "UNPROVEN" in rel or "LOW_SAMPLE" in rel:
        score = 34.0
        reasons.append(f"news reliability is {rel}")
    elif "NO_EVENT" in rel or "NO_DATA" in rel:
        score = 48.0
        reasons.append("no mapped event proof")
    if "WATCH" in top_decision:
        reasons.append("event route is watch-for-confirmation")
    if "PUT" in directional or "HEDGE" in directional:
        event_bias = "DOWNSIDE_OR_HEDGE"
    elif "CALL" in directional:
        event_bias = "BULLISH_READTHROUGH"
    else:
        event_bias = "NO_DIRECTIONAL_EVENT"
    return float(np.clip(score, 0, 80)), reasons, event_bias


def premium_budget_bps(final_permission: str, risk_action: str, score: float, horizon: str) -> float:
    text = f"{final_permission} {risk_action}".upper()
    if "BLOCK" in text or "NO_OPTION" in text or "NO BULLISH" in text or "NO_BULLISH" in text:
        return 0.0
    if "REDUCE_ONLY" in text or "HEDGE" in text or "PUT" in text:
        return 5.0 if score < 55 else 10.0
    if "CALL" in text:
        base = 8.0 if score < 62 else 15.0
        if str(horizon).lower().startswith("short"):
            base = min(base, 8.0)
        return base
    if "TINY" in text:
        return 5.0
    return 0.0


def final_route(row: dict[str, Any]) -> tuple[str, str, str, str]:
    risk_action = as_upper(row.get("risk_action"), "")
    gate = as_upper(row.get("gate_status"), "")
    horizon = as_upper(row.get("horizon"), "")
    option_side = as_upper(row.get("option_side", row.get("base_option_side")), "NONE")
    event_bias = as_upper(row.get("event_bias"), "")
    risk_score = safe_float(row.get("risk_score"), 0)
    exec_score = safe_float(row.get("execution_score"), 0)
    greeks_score = safe_float(row.get("greeks_score"), 0)
    monitor_score = safe_float(row.get("monitor_score"), 0)
    event_score = safe_float(row.get("event_score"), 0)
    total = safe_float(row.get("route_quality_score"), 0)

    if "REDUCE_ONLY" in risk_action or "NO NEW EXPOSURE" in gate:
        if option_side == "PUT" or "HEDGE" in event_bias:
            return (
                "PUT_OR_HEDGE_RESEARCH_ONLY",
                "PUT",
                "Put debit spread or protective hedge research only; no bullish option.",
                "Risk reduction first; hedge research only after manual spread and IV checks.",
            )
        return (
            "NO_BULLISH_OPTION_RISK_REDUCTION_FIRST",
            "NONE",
            "No option. Reduce existing paper risk first.",
            "L8 risk blocks any bullish option route.",
        )
    if min(exec_score, monitor_score) < 35:
        return (
            "WAIT_EXECUTION_OR_MONITOR_REVIEW",
            "NONE",
            "No option until execution/monitor shock clears.",
            "Execution or live monitor risk is too high for option research.",
        )
    if greeks_score < 38:
        return (
            "NO_OPTION_GREEKS_OR_IV_BLOCK",
            "NONE",
            "No option until IV/Greeks risk improves.",
            "Options structure risk is not clean enough.",
        )
    if event_score < 38:
        return (
            "WATCH_EVENT_PROOF_FIRST",
            "NONE",
            "No option until event proof and price/volume confirmation improve.",
            "News/event evidence is not reliable enough.",
        )
    if option_side == "CALL" and total >= 62 and risk_score >= 55:
        structure = "Defined-risk call debit spread; no naked weekly calls."
        if "SHORT" in horizon:
            structure = "Small defined-risk call spread watch only after trigger; no weekly OTM chase."
        return (
            "DEFINED_RISK_CALL_SPREAD_WATCH",
            "CALL",
            structure,
            "Call route requires trigger, spread check, IV check, and risk gate remaining clear.",
        )
    if option_side == "PUT" and total >= 50:
        return (
            "PUT_OR_HEDGE_RESEARCH_ONLY",
            "PUT",
            "Put debit spread or protective hedge research only.",
            "Put route is for risk reduction or downside research, not speculative leverage.",
        )
    if "TINY" in gate or total >= 50:
        return (
            "TINY_STOCK_OR_ETF_PAPER_ONLY",
            "NONE",
            "Use tiny stock/ETF paper route; no option.",
            "Vehicle choice favors underlying paper because option proof is incomplete.",
        )
    return (
        "NO_OPTION_WAIT",
        "NONE",
        "No option. Wait for gates and trigger proof.",
        "Combined route score is too low.",
    )


def build_matrix() -> pd.DataFrame:
    horizon = read_csv_safe(ROOT / "horizon_vehicle_matrix.csv")
    option = one_by_ticker(read_csv_safe(ROOT / "option_route_clarity_board.csv"))
    greeks = one_by_ticker(read_csv_safe(ROOT / "options_greeks_book_risk.csv"))
    execution = one_by_ticker(read_csv_safe(ROOT / "execution_cost_model.csv"))
    risk = one_by_ticker(read_csv_safe(ROOT / "risk_desk_ticker_action_queue.csv"))
    events = one_by_ticker(read_csv_safe(ROOT / "event_readthrough_target_ranking.csv"), "target_ticker")
    monitor = one_by_ticker(read_csv_safe(ROOT / "desk_monitor_ticker_state.csv"))
    bt_quality = one_by_ticker(load_option_backtest_quality())

    if horizon.empty:
        tickers = sorted(set(option) | set(greeks) | set(execution) | set(risk) | set(monitor))
        rows = []
        for ticker in tickers:
            for h, window in [("Short-term", "1-5 trading days"), ("Medium-term", "2-8 weeks"), ("Long-term", "3-12 months")]:
                rows.append({"ticker": ticker, "horizon": h, "time_window": window})
        horizon = pd.DataFrame(rows)

    rows: list[dict[str, Any]] = []
    for _, hrow in horizon.iterrows():
        ticker = clean_ticker(hrow.get("ticker"))
        if not ticker:
            continue
        opt = option.get(ticker, {})
        gre = greeks.get(ticker, {})
        exe = execution.get(ticker, {})
        rsk = risk.get(ticker, {})
        evt = events.get(ticker, {})
        mon = monitor.get(ticker, {})
        bt = bt_quality.get(ticker, {})

        risk_action = as_upper(hrow.get("risk_action", opt.get("risk_action", rsk.get("final_risk_action"))), "UNKNOWN")
        gate_status = as_text(hrow.get("gate_status", opt.get("gate_status")), "UNKNOWN")
        risk_score, risk_note = score_risk(risk_action, gate_status)
        execution_score, execution_reasons = score_execution(exe)
        greeks_score, greeks_reasons, option_style = score_greeks(gre)
        monitor_score, monitor_reasons = score_monitor(mon)
        event_score, event_reasons, event_bias = score_event(evt, hrow.to_dict())
        horizon_score = safe_float(hrow.get("horizon_score"), 35.0)
        backtest_status = as_upper(bt.get("backtest_sanity_status"), "NO_OPTION_BACKTEST")
        backtest_penalty = 8.0 if "REVIEW" in backtest_status else 0.0

        route_quality = (
            0.28 * risk_score
            + 0.22 * execution_score
            + 0.18 * greeks_score
            + 0.14 * monitor_score
            + 0.10 * event_score
            + 0.08 * max(0.0, min(horizon_score, 100.0))
            - backtest_penalty
        )
        route_quality = float(np.clip(route_quality, 0.0, 85.0))

        base_row: dict[str, Any] = {
            "ticker": ticker,
            "sector": as_text(hrow.get("sector", opt.get("sector", rsk.get("sector"))), "Unknown"),
            "horizon": as_text(hrow.get("horizon"), "Unknown"),
            "time_window": as_text(hrow.get("time_window"), ""),
            "risk_action": risk_action,
            "gate_status": gate_status,
            "base_option_side": as_upper(hrow.get("option_side", opt.get("option_side")), "NONE"),
            "base_option_route": as_text(hrow.get("option_use_case", opt.get("option_use_case")), "NO_OPTION_CONTEXT"),
            "event_bias": event_bias,
            "risk_score": round(risk_score, 1),
            "execution_score": round(execution_score, 1),
            "greeks_score": round(greeks_score, 1),
            "monitor_score": round(monitor_score, 1),
            "event_score": round(event_score, 1),
            "horizon_score": round(horizon_score, 1),
            "route_quality_score": round(route_quality, 1),
            "iv_rank": safe_float(gre.get("iv_rank"), np.nan),
            "gamma_score": safe_float(gre.get("gamma_score"), np.nan),
            "options_heat_score": safe_float(gre.get("options_heat_score"), np.nan),
            "greeks_status": as_upper(gre.get("greeks_status"), "NO_GREEKS_DATA"),
            "base_cost_bps": safe_float(exe.get("base_cost_bps"), np.nan),
            "stress_cost_bps": safe_float(exe.get("stress_cost_bps"), np.nan),
            "spread_bps": safe_float(exe.get("spread_bps"), np.nan),
            "expected_fill_rate_pct": safe_float(exe.get("expected_fill_rate_pct"), np.nan),
            "execution_cost_status": as_upper(exe.get("execution_cost_status"), "NO_EXECUTION_DATA"),
            "monitor_severity": as_upper(mon.get("max_monitor_severity", mon.get("max_severity")), "NO_MONITOR_DATA"),
            "price_break_state": as_upper(mon.get("price_break_state"), "NO_DATA"),
            "volume_spike_state": as_upper(mon.get("volume_spike_state"), "NO_DATA"),
            "volatility_regime_state": as_upper(mon.get("volatility_regime_state"), "NO_DATA"),
            "spread_status": as_upper(mon.get("spread_status"), "NO_DATA"),
            "backtest_sanity_status": backtest_status,
            "best_option_backtest_strategy": as_text(bt.get("best_option_backtest_strategy"), "NO_DATA"),
            "best_option_backtest_sharpe": safe_float(bt.get("best_option_backtest_sharpe"), np.nan),
            "trigger_to_watch": as_text(hrow.get("trigger_to_watch", opt.get("call_trigger")), ""),
            "invalidation": as_text(hrow.get("invalidation", opt.get("option_invalidation")), ""),
        }
        final_permission, final_side, final_structure, route_reason = final_route(base_row)
        no_go_reasons = []
        no_go_reasons.extend([risk_note])
        no_go_reasons.extend(execution_reasons)
        no_go_reasons.extend(greeks_reasons)
        no_go_reasons.extend(monitor_reasons)
        no_go_reasons.extend(event_reasons)
        if "REVIEW" in backtest_status:
            no_go_reasons.append(as_text(bt.get("backtest_sanity_note"), "option backtest requires review"))
        if not no_go_reasons:
            no_go_reasons = ["No hard blocker found; still paper/research only."]
        budget_bps = premium_budget_bps(final_permission, risk_action, route_quality, base_row["horizon"])

        base_row.update({
            "final_vehicle_decision": final_permission,
            "final_option_side": final_side,
            "final_option_structure": final_structure,
            "option_style_rule": option_style,
            "premium_budget_bps_of_model_account": round(budget_bps, 2),
            "max_premium_dollars_model_account": round(MODEL_ACCOUNT_VALUE * budget_bps / 10000.0, 2),
            "no_go_count": len([x for x in no_go_reasons if x]),
            "no_go_reasons": "; ".join(dict.fromkeys([x for x in no_go_reasons if x]))[:1200],
            "required_confirmation": as_text(hrow.get("required_confirmations", opt.get("call_unlock_checklist")), "Risk, event, price, spread, IV, and liquidity checks must clear."),
            "why_this_route": route_reason,
            "source_files": "horizon_vehicle_matrix.csv | option_route_clarity_board.csv | options_greeks_book_risk.csv | execution_cost_model.csv | risk_desk_ticker_action_queue.csv | desk_monitor_ticker_state.csv | event_readthrough_target_ranking.csv | options_backtest_results.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        rows.append(base_row)
    return pd.DataFrame(rows)


def build_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, sub in matrix.groupby("ticker"):
        sub = sub.sort_values("route_quality_score", ascending=False)
        best = sub.iloc[0]
        call_rows = int(sub["final_option_side"].astype(str).str.upper().eq("CALL").sum())
        put_rows = int(sub["final_option_side"].astype(str).str.upper().eq("PUT").sum())
        no_option_rows = int(sub["final_option_side"].astype(str).str.upper().eq("NONE").sum())
        if call_rows > 0:
            desk_bias = "CALL_WATCH_AFTER_GATES"
        elif put_rows > 0:
            desk_bias = "PUT_OR_HEDGE_RESEARCH"
        elif str(best.get("final_vehicle_decision", "")).upper().startswith("TINY"):
            desk_bias = "TINY_UNDERLYING_ONLY"
        else:
            desk_bias = "NO_OPTION_WAIT"
        rows.append({
            "ticker": ticker,
            "sector": best.get("sector"),
            "desk_option_bias": desk_bias,
            "best_horizon": best.get("horizon"),
            "best_time_window": best.get("time_window"),
            "best_route_quality_score": best.get("route_quality_score"),
            "best_vehicle_decision": best.get("final_vehicle_decision"),
            "best_option_side": best.get("final_option_side"),
            "best_option_structure": best.get("final_option_structure"),
            "max_premium_budget_bps": float(sub["premium_budget_bps_of_model_account"].max()),
            "call_horizon_count": call_rows,
            "put_horizon_count": put_rows,
            "no_option_horizon_count": no_option_rows,
            "top_no_go_reasons": best.get("no_go_reasons"),
            "required_confirmation": best.get("required_confirmation"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows).sort_values(["best_route_quality_score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def build_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    checks = [
        ("risk_gate", "risk_score", 45, "Risk gate must not allow options to override L8."),
        ("execution_cost", "execution_score", 50, "TCA/spread/fill risk must be small enough to preserve signal edge."),
        ("greeks_iv_gamma", "greeks_score", 50, "Greeks, IV, and gamma heat must not turn the option into hidden leverage."),
        ("monitor_events", "monitor_score", 50, "Live shocks must calm or be explained before option research."),
        ("event_proof", "event_score", 45, "News/event read-through needs proof and price/volume confirmation."),
    ]
    for _, row in matrix.iterrows():
        for check, col, threshold, why in checks:
            score = safe_float(row.get(col), 0.0)
            rows.append({
                "ticker": row.get("ticker"),
                "horizon": row.get("horizon"),
                "check": check,
                "score": round(score, 1),
                "threshold": threshold,
                "status": "PASS" if score >= threshold else "BLOCK_OR_REVIEW",
                "why_it_matters": why,
                "evidence": row.get("no_go_reasons"),
                "final_vehicle_decision": row.get("final_vehicle_decision"),
                "research_only": True,
            })
    return pd.DataFrame(rows)


def build_playbook(matrix: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        ticker = row["ticker"]
        sub = matrix[matrix["ticker"] == ticker].sort_values("route_quality_score", ascending=False)
        best = sub.iloc[0] if not sub.empty else pd.Series(dtype=object)
        bias = as_upper(row.get("desk_option_bias"), "")
        if "CALL" in bias:
            first_step = "Wait for upside trigger; then review defined-risk call spread only."
        elif "PUT" in bias or "HEDGE" in bias:
            first_step = "Use only hedge/downside research framing; do not treat this as a bullish idea."
        elif "TINY" in bias:
            first_step = "Use tiny underlying paper only; skip options until proof improves."
        else:
            first_step = "Do not use options; repair gates first."
        rows.append({
            "ticker": ticker,
            "desk_option_bias": row.get("desk_option_bias"),
            "first_step": first_step,
            "short_term_rule": route_for_horizon(sub, "Short-term"),
            "medium_term_rule": route_for_horizon(sub, "Medium-term"),
            "long_term_rule": route_for_horizon(sub, "Long-term"),
            "trigger": best.get("trigger_to_watch", ""),
            "invalidation": best.get("invalidation", ""),
            "do_not_do": "No live orders. No broker. No naked weekly options. Do not let gamma/news override risk or execution gates.",
            "manual_checks": best.get("required_confirmation", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows)


def route_for_horizon(sub: pd.DataFrame, horizon: str) -> str:
    if sub.empty or "horizon" not in sub.columns:
        return "No data."
    row = sub[sub["horizon"].astype(str).str.upper().eq(horizon.upper())]
    if row.empty:
        return "No data."
    r = row.iloc[0]
    return f"{r.get('final_vehicle_decision')}: {r.get('final_option_structure')}"


def write_outputs(matrix: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame, playbook: pd.DataFrame) -> dict[str, Any]:
    matrix.to_csv(OUT_MATRIX, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    playbook.to_csv(OUT_PLAYBOOK, index=False)

    state = {
        "date": today_str(),
        "route_rows": int(len(matrix)),
        "tickers": int(summary["ticker"].nunique()) if not summary.empty else 0,
        "call_watch_tickers": int(summary["desk_option_bias"].astype(str).str.contains("CALL", na=False).sum()) if not summary.empty else 0,
        "put_or_hedge_tickers": int(summary["desk_option_bias"].astype(str).str.contains("PUT|HEDGE", na=False).sum()) if not summary.empty else 0,
        "no_option_tickers": int(summary["desk_option_bias"].astype(str).str.contains("NO_OPTION", na=False).sum()) if not summary.empty else 0,
        "tiny_underlying_tickers": int(summary["desk_option_bias"].astype(str).str.contains("TINY", na=False).sum()) if not summary.empty else 0,
        "blocked_or_review_checks": int(audit["status"].astype(str).eq("BLOCK_OR_REVIEW").sum()) if not audit.empty else 0,
        "overall_status": "OPTIONS_EXECUTION_RESEARCH_BOARD_ACTIVE" if len(matrix) else "NO_ROUTE_DATA",
        "truth": "This step clarifies option/execution research routes only. It cannot create live orders and cannot override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "matrix": OUT_MATRIX.name,
            "summary": OUT_SUMMARY.name,
            "audit": OUT_AUDIT.name,
            "playbook": OUT_PLAYBOOK.name,
            "report": OUT_REPORT.name,
        },
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Desk truth",
        (
            "This is an options/execution research board, not a trade blotter. "
            "It separates short, medium, and long horizon routes and makes call/put/no-option permissions explicit."
        ),
        "",
        "## Summary",
        df_to_markdown(summary.head(30), max_rows=30),
        "",
        "## Route matrix",
        df_to_markdown(matrix[[
            "ticker", "horizon", "route_quality_score", "final_vehicle_decision",
            "final_option_side", "final_option_structure", "premium_budget_bps_of_model_account",
            "risk_action", "execution_cost_status", "greeks_status", "monitor_severity",
            "no_go_reasons",
        ]].head(60), max_rows=60) if not matrix.empty else "No rows.",
        "",
        "## No-go audit",
        df_to_markdown(audit.head(60), max_rows=60),
        "",
        "## Non-negotiable rules",
        "- No broker connection. No live orders.",
        "- No naked weekly options.",
        "- Gamma squeeze is context only, never permission.",
        "- Risk gate, execution cost, source proof, and monitor shocks dominate options.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 173 - Options Execution Route Engine", sections)
    return state


def main() -> None:
    matrix = build_matrix()
    summary = build_summary(matrix)
    audit = build_audit(matrix)
    playbook = build_playbook(matrix, summary)
    state = write_outputs(matrix, summary, audit, playbook)
    print("Step 173 complete.")
    print(f"Route rows: {state['route_rows']}; tickers: {state['tickers']}")
    print(f"Call watch: {state['call_watch_tickers']}; put/hedge: {state['put_or_hedge_tickers']}; no-option: {state['no_option_tickers']}")
    print(f"Blocked/review checks: {state['blocked_or_review_checks']}")
    print(f"Wrote: {OUT_MATRIX.name}, {OUT_SUMMARY.name}, {OUT_AUDIT.name}, {OUT_PLAYBOOK.name}, {OUT_STATE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
