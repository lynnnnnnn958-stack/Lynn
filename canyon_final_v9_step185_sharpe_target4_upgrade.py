#!/usr/bin/env python3
"""
Canyon v9 Step 185 - Sharpe 4 Target Upgrade Desk.

Research-only. No broker connection. No live orders.

This step does not "make Sharpe 4" by changing a headline metric. It builds a
control desk for the user's target Sharpe of 4.0:

  - current headline Sharpe
  - credibility-adjusted planning Sharpe
  - gap to target
  - what must improve: signal IC, live proof, risk repair, execution cost,
    portfolio construction, point-in-time data, and walk-forward validation

Outputs:
  sharpe_target4_state.json
  sharpe_target4_driver_attribution.csv
  sharpe_target4_action_queue.csv
  sharpe_target4_policy.csv
  sharpe_target4_ticker_gate.csv
  sharpe_target4_report.md
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


TARGET_SHARPE = 4.0

OUT_STATE = ROOT / "sharpe_target4_state.json"
OUT_DRIVERS = ROOT / "sharpe_target4_driver_attribution.csv"
OUT_QUEUE = ROOT / "sharpe_target4_action_queue.csv"
OUT_POLICY = ROOT / "sharpe_target4_policy.csv"
OUT_TICKERS = ROOT / "sharpe_target4_ticker_gate.csv"
OUT_REPORT = ROOT / "sharpe_target4_report.md"


def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clamp(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return low
    return float(min(max(value, low), high))


def metric_value(summary: pd.DataFrame, name: str, default: float = np.nan) -> float:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return default
    row = summary[summary["metric"].astype(str).str.lower().eq(name.lower())]
    if row.empty:
        return default
    return safe_float(row.iloc[0]["value"], default)


def pct_to_float(value: Any, default: float = np.nan) -> float:
    out = safe_float(value, default)
    if np.isfinite(out) and abs(out) > 1.5:
        out /= 100.0
    return out


def status_from_gap(current: float, target: float = TARGET_SHARPE) -> str:
    if not np.isfinite(current):
        return "NO_DATA"
    if current >= target:
        return "TARGET_MET_BUT_VERIFY"
    if current >= 3.0:
        return "CLOSE_BUT_NEEDS_PROOF"
    if current >= 2.0:
        return "MID_STAGE"
    if current >= 1.0:
        return "EARLY_STAGE"
    return "REPAIR_FIRST"


def multiplier_from_score(score: float, floor: float = 0.35, cap: float = 1.0) -> float:
    if not np.isfinite(score):
        return floor
    return clamp(floor + (cap - floor) * score / 100.0, floor, cap)


def performance_baseline() -> dict[str, Any]:
    summary = read_csv_safe(ROOT / "backtest_summary.csv")
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    walk = read_csv_safe(ROOT / "backtest_walk_forward_proxy.csv")
    optimizer_report = ROOT / "portfolio_optimizer_report.md"

    headline_sharpe = metric_value(summary, "Annualised Sharpe")
    sortino = metric_value(summary, "Annualised Sortino")
    max_dd = pct_to_float(metric_value(summary, "Max Drawdown"))
    total_return = pct_to_float(metric_value(summary, "Total Return (Strategy)"))
    months = int(safe_float(metric_value(summary, "Periods Tested (months)"), 0) or 0)

    ann_return = np.nan
    ann_vol = np.nan
    monthly_mean = np.nan
    monthly_vol = np.nan
    turnover_median = np.nan
    cost_median_bps = np.nan
    if not monthly.empty and "strategy_ret" in monthly.columns:
        ret = pd.to_numeric(monthly["strategy_ret"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not ret.empty:
            monthly_mean = float(ret.mean())
            monthly_vol = float(ret.std(ddof=1))
            ann_return = float((1.0 + monthly_mean) ** 12 - 1.0)
            ann_vol = float(monthly_vol * np.sqrt(12.0))
            if not np.isfinite(headline_sharpe) and ann_vol > 0:
                headline_sharpe = ann_return / ann_vol
        if "turnover_pct" in monthly.columns:
            turnover_median = safe_float(pd.to_numeric(monthly["turnover_pct"], errors="coerce").median())
        if "tc_cost_bps" in monthly.columns:
            cost_median_bps = safe_float(pd.to_numeric(monthly["tc_cost_bps"], errors="coerce").median())

    walk_median = np.nan
    walk_min = np.nan
    walk_oos = np.nan
    if not walk.empty and "sharpe_proxy" in walk.columns:
        w = pd.to_numeric(walk["sharpe_proxy"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not w.empty:
            walk_median = float(w.median())
            walk_min = float(w.min())
        if "window" in walk.columns:
            oos = walk[walk["window"].astype(str).str.contains("oos|second", case=False, na=False)]
            if not oos.empty:
                walk_oos = safe_float(pd.to_numeric(oos["sharpe_proxy"], errors="coerce").median())

    optimizer_sharpe = np.nan
    if optimizer_report.exists():
        text = optimizer_report.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "| Max-Sharpe |" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    optimizer_sharpe = safe_float(parts[3])
                break

    return {
        "headline_sharpe": round(headline_sharpe, 3) if np.isfinite(headline_sharpe) else np.nan,
        "sortino": round(sortino, 3) if np.isfinite(sortino) else np.nan,
        "max_drawdown_pct": round(max_dd * 100, 2) if np.isfinite(max_dd) else np.nan,
        "total_return_pct": round(total_return * 100, 2) if np.isfinite(total_return) else np.nan,
        "months": months,
        "annual_return_est_pct": round(ann_return * 100, 2) if np.isfinite(ann_return) else np.nan,
        "annual_vol_est_pct": round(ann_vol * 100, 2) if np.isfinite(ann_vol) else np.nan,
        "monthly_mean_pct": round(monthly_mean * 100, 3) if np.isfinite(monthly_mean) else np.nan,
        "monthly_vol_pct": round(monthly_vol * 100, 3) if np.isfinite(monthly_vol) else np.nan,
        "walk_forward_median_sharpe": round(walk_median, 3) if np.isfinite(walk_median) else np.nan,
        "walk_forward_min_sharpe": round(walk_min, 3) if np.isfinite(walk_min) else np.nan,
        "walk_forward_oos_sharpe": round(walk_oos, 3) if np.isfinite(walk_oos) else np.nan,
        "optimizer_max_sharpe": round(optimizer_sharpe, 3) if np.isfinite(optimizer_sharpe) else np.nan,
        "median_monthly_turnover_pct": round(turnover_median, 1) if np.isfinite(turnover_median) else np.nan,
        "median_backtest_cost_bps": round(cost_median_bps, 2) if np.isfinite(cost_median_bps) else np.nan,
    }


def credibility_adjustment(headline_sharpe: float) -> dict[str, Any]:
    cred = read_json_safe(ROOT / "backtest_credibility_state.json", {})
    signal = read_json_safe(ROOT / "signal_validation_state.json", {})
    pit = read_json_safe(ROOT / "pit_truth_state.json", {})
    tca = read_json_safe(ROOT / "execution_tca_state.json", {})
    opt = read_json_safe(ROOT / "institutional_optimizer_state.json", {})

    cred_score = safe_float(cred.get("overall_credibility_score"), 50)
    pit_score = safe_float(pit.get("pit_truth_score"), 50)
    tca_score = safe_float(tca.get("overall_execution_tca_score"), 25)
    opt_score = safe_float(opt.get("institutional_optimizer_score"), 50)

    signal_blocked = int(safe_float(signal.get("policy_blocked_signals"), 0) or 0)
    signal_down = int(safe_float(signal.get("p2_signal_reviews"), 0) or 0)
    live_ic_available = bool(signal.get("live_ic_available", False))
    live_ic_obs = int(safe_float(signal.get("live_ic_observations"), 0) or 0)
    signal_mult = 1.0 - 0.03 * signal_blocked - 0.015 * signal_down
    if not live_ic_available or live_ic_obs < 120:
        signal_mult = min(signal_mult, 0.72)
    signal_mult = clamp(signal_mult, 0.35, 1.0)

    tca_mult = multiplier_from_score(tca_score, floor=0.50, cap=1.0)
    if str(tca.get("status", "")).upper().startswith("EXECUTION_BLOCKED"):
        tca_mult = min(tca_mult, 0.60)

    opt_mult = multiplier_from_score(opt_score, floor=0.55, cap=1.0)
    if str(opt.get("overall_status", "")).upper() != "CLEAR":
        opt_mult = min(opt_mult, 0.82)

    cred_mult = multiplier_from_score(cred_score, floor=0.45, cap=1.0)
    pit_mult = multiplier_from_score(pit_score, floor=0.45, cap=1.0)

    combined = cred_mult * pit_mult * signal_mult * tca_mult * opt_mult
    planning_sharpe = float(headline_sharpe) * combined if np.isfinite(headline_sharpe) else np.nan

    return {
        "backtest_credibility_score": round(cred_score, 1),
        "pit_truth_score": round(pit_score, 1),
        "execution_tca_score": round(tca_score, 1),
        "optimizer_score": round(opt_score, 1),
        "blocked_signals": signal_blocked,
        "downweighted_signals": signal_down,
        "live_ic_observations": live_ic_obs,
        "credibility_multiplier": round(cred_mult, 3),
        "pit_multiplier": round(pit_mult, 3),
        "signal_multiplier": round(signal_mult, 3),
        "execution_multiplier": round(tca_mult, 3),
        "optimizer_multiplier": round(opt_mult, 3),
        "combined_multiplier": round(combined, 3),
        "credibility_adjusted_planning_sharpe": round(planning_sharpe, 3) if np.isfinite(planning_sharpe) else np.nan,
    }


def build_driver_attribution(base: dict[str, Any], adj: dict[str, Any]) -> pd.DataFrame:
    current = safe_float(base.get("headline_sharpe"))
    planning = safe_float(adj.get("credibility_adjusted_planning_sharpe"))
    ann_ret = safe_float(base.get("annual_return_est_pct")) / 100.0
    ann_vol = safe_float(base.get("annual_vol_est_pct")) / 100.0
    gap = TARGET_SHARPE - current if np.isfinite(current) else np.nan

    target_return_same_vol = TARGET_SHARPE * ann_vol if np.isfinite(ann_vol) else np.nan
    target_vol_same_return = ann_ret / TARGET_SHARPE if np.isfinite(ann_ret) else np.nan

    rows = [
        {
            "driver": "Headline Sharpe gap",
            "current_value": round(current, 3) if np.isfinite(current) else np.nan,
            "target_value": TARGET_SHARPE,
            "gap": round(gap, 3) if np.isfinite(gap) else np.nan,
            "plain_english": "This is the visible backtest gap. Do not treat it as live performance.",
            "status": status_from_gap(current),
            "source_files": "backtest_summary.csv / backtest_monthly_perf.csv",
        },
        {
            "driver": "Credibility-adjusted planning Sharpe",
            "current_value": round(planning, 3) if np.isfinite(planning) else np.nan,
            "target_value": TARGET_SHARPE,
            "gap": round(TARGET_SHARPE - planning, 3) if np.isfinite(planning) else np.nan,
            "plain_english": "This haircuts the headline Sharpe for prototype data, thin live IC, execution gaps, and optimizer review status.",
            "status": status_from_gap(planning),
            "source_files": "backtest_credibility_state.json / signal_validation_state.json / execution_tca_state.json",
        },
        {
            "driver": "Return needed if volatility stays the same",
            "current_value": round(ann_ret * 100, 2) if np.isfinite(ann_ret) else np.nan,
            "target_value": round(target_return_same_vol * 100, 2) if np.isfinite(target_return_same_vol) else np.nan,
            "gap": round((target_return_same_vol - ann_ret) * 100, 2) if np.isfinite(target_return_same_vol) and np.isfinite(ann_ret) else np.nan,
            "plain_english": "This shows how unrealistic it is to chase Sharpe 4 through return only.",
            "status": "NOT_A_SOLO_PATH",
            "source_files": "backtest_monthly_perf.csv",
        },
        {
            "driver": "Volatility needed if return stays the same",
            "current_value": round(ann_vol * 100, 2) if np.isfinite(ann_vol) else np.nan,
            "target_value": round(target_vol_same_return * 100, 2) if np.isfinite(target_vol_same_return) else np.nan,
            "gap": round((ann_vol - target_vol_same_return) * 100, 2) if np.isfinite(target_vol_same_return) and np.isfinite(ann_vol) else np.nan,
            "plain_english": "This shows how much cleaner the path must become if returns do not improve.",
            "status": "REQUIRES_RISK_AND_SIGNAL_REPAIR",
            "source_files": "backtest_monthly_perf.csv / risk_repair_state.json",
        },
        {
            "driver": "Signal IC quality",
            "current_value": round(adj.get("signal_multiplier", np.nan), 3),
            "target_value": 0.95,
            "gap": round(0.95 - safe_float(adj.get("signal_multiplier")), 3),
            "plain_english": "Blocked and down-weighted signals must stay out until live IC and regime decay prove them.",
            "status": "REPAIR_REQUIRED",
            "source_files": "signal_validation_state.json / research_signal_weight_policy.csv",
        },
        {
            "driver": "Execution and TCA quality",
            "current_value": round(adj.get("execution_multiplier", np.nan), 3),
            "target_value": 0.95,
            "gap": round(0.95 - safe_float(adj.get("execution_multiplier")), 3),
            "plain_english": "A Sharpe target is not credible while spread, fill, and TCA checks are blocked or manual.",
            "status": "REPAIR_REQUIRED",
            "source_files": "execution_tca_state.json / backtest_execution_reality_check.csv",
        },
        {
            "driver": "Point-in-time proof",
            "current_value": round(adj.get("pit_multiplier", np.nan), 3),
            "target_value": 0.95,
            "gap": round(0.95 - safe_float(adj.get("pit_multiplier")), 3),
            "plain_english": "A high Sharpe can be fake if the historical data was not available at the decision time.",
            "status": "REVIEW_REQUIRED",
            "source_files": "pit_truth_state.json / pit_backtest_readiness_gates.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_policy() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "policy_area": "Sharpe 4 claim rule",
            "rule": "Do not claim Sharpe 4 unless headline Sharpe, walk-forward OOS Sharpe, and credibility-adjusted planning Sharpe all clear.",
            "current_gate": "BLOCKED_FOR_CLAIM",
            "target_gate": "headline >= 4.0; OOS >= 3.0; credibility-adjusted >= 2.5; no P0 data/execution/signal blockers",
            "why": "A single optimized backtest Sharpe can be selection bias.",
        },
        {
            "policy_area": "Signal admission",
            "rule": "Only admit signals with positive IC, stable decay, regime bucket support, and live observations.",
            "current_gate": "SIGNAL_REPAIR_REQUIRED",
            "target_gate": "no BLOCK_SIGNAL in active model; live IC observations >= 250 per promoted signal",
            "why": "Sharpe 4 requires edge quality, not more weak signals.",
        },
        {
            "policy_area": "Turnover and cost",
            "rule": "Turnover must be capped unless expected alpha survives TCA and failed-fill stress.",
            "current_gate": "EXECUTION_REPAIR_REQUIRED",
            "target_gate": "median monthly turnover <= 45%; median model cost <= 7 bps; current TCA <= backtest cost + 3 bps",
            "why": "High turnover can erase Sharpe after spread and slippage.",
        },
        {
            "policy_area": "Risk scaling",
            "rule": "Use volatility target and drawdown state before increasing gross exposure.",
            "current_gate": "RISK_REPAIR_FIRST",
            "target_gate": "risk repair scenario clear; gross/vol within target; no REDUCE_ONLY names in active ideas",
            "why": "Sharpe improves more from avoiding bad risk than from chasing every signal.",
        },
        {
            "policy_area": "Point-in-time evidence",
            "rule": "Historical evidence must prove what was known before the decision timestamp.",
            "current_gate": "PIT_REVIEW_REQUIRED",
            "target_gate": "PIT score >= 85; no missing timestamp / membership / delisted-name gates",
            "why": "Look-ahead and survivorship bias can make Sharpe meaningless.",
        },
    ])


def build_action_queue(base: dict[str, Any], adj: dict[str, Any]) -> pd.DataFrame:
    turnover = safe_float(base.get("median_monthly_turnover_pct"))
    cost = safe_float(base.get("median_backtest_cost_bps"))
    rows = [
        {
            "priority": "P0",
            "workstream": "Risk repair before Sharpe chasing",
            "action": "Keep REDUCE_ONLY and SIZE_DOWN names from driving upside ideas. Use the risk repair scenario before reviewing new calls or size.",
            "expected_sharpe_effect": "Reduces left-tail months and volatility drag.",
            "done_when": "final_risk_gate has no REDUCE_ONLY names in the idea queue, or they are explicitly risk-reduction only.",
            "source_files": "final_risk_gate.csv / risk_repair_priority_queue.csv",
            "status": "ACTIVE",
        },
        {
            "priority": "P0",
            "workstream": "Signal repair",
            "action": "Remove blocked signals from active scoring; keep down-weighted signals as research-only until decay and live IC improve.",
            "expected_sharpe_effect": "Raises average edge quality and reduces noisy trades.",
            "done_when": "policy_blocked_signals = 0 for active model and live IC observations are collected.",
            "source_files": "signal_validation_state.json / research_signal_weight_policy.csv",
            "status": "REPAIR_REQUIRED",
        },
        {
            "priority": "P0",
            "workstream": "Execution cost realism",
            "action": "Cut turnover and require spread/TCA proof before an idea can count toward the Sharpe target.",
            "expected_sharpe_effect": "Prevents paper alpha from disappearing after costs.",
            "done_when": f"median turnover <= 45% and median cost <= 7 bps. Current median turnover {turnover:.1f}% and cost {cost:.1f} bps.",
            "source_files": "backtest_execution_reality_check.csv / execution_tca_state.json",
            "status": "REPAIR_REQUIRED",
        },
        {
            "priority": "P1",
            "workstream": "Walk-forward validation",
            "action": "Promote only rules that survive non-overlapping out-of-sample windows and the latest 12/24 month slices.",
            "expected_sharpe_effect": "Reduces overfit Sharpe and improves repeatability.",
            "done_when": "true frozen-signal OOS Sharpe >= 3.0 and no window below 1.5 before any Sharpe 4 claim.",
            "source_files": "backtest_walk_forward_proxy.csv / backtest_credibility_scorecard.csv",
            "status": "NEEDS_DEEPER_TEST",
        },
        {
            "priority": "P1",
            "workstream": "Point-in-time truth",
            "action": "Close event timestamp, historical universe, delisted-name, split/dividend, and fundamentals as-of gaps.",
            "expected_sharpe_effect": "Turns prototype Sharpe into evidence that can be trusted.",
            "done_when": "PIT score >= 85 and no P0/P1 source timing gates remain.",
            "source_files": "pit_truth_state.json / pit_backtest_readiness_gates.csv",
            "status": "REVIEW_REQUIRED",
        },
        {
            "priority": "P2",
            "workstream": "Sharpe 4 second-stage alpha",
            "action": "Only after P0/P1 gates improve, test additional alpha sleeves: event-confirmed momentum, software catch-up vs semi late-cycle, and defensive put/hedge overlays.",
            "expected_sharpe_effect": "Adds return only after the base is cleaner.",
            "done_when": "new sleeve has OOS IC, cost-aware backtest, and drawdown attribution.",
            "source_files": "event_readthrough_decision_board.csv / sector_cycle_linkage_report.md / options_execution_route_matrix.csv",
            "status": "WAIT_FOR_BASE_REPAIR",
        },
    ]
    return pd.DataFrame(rows)


def build_ticker_gate() -> pd.DataFrame:
    bridge = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    if bridge.empty:
        return pd.DataFrame(columns=[
            "ticker", "sharpe4_role", "current_blocker", "can_help_target4", "plain_next_step", "source_files"
        ])
    rows: list[dict[str, Any]] = []
    for _, row in bridge.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        signal_action = str(row.get("signal_validation_action", "")).upper()
        risk_action = str(row.get("final_risk_action", "")).upper()
        status = str(row.get("final_optimizer_status", "")).upper()
        tca_status = str(row.get("execution_status", "")).upper()
        constraints = str(row.get("binding_constraints", ""))
        weight = safe_float(row.get("final_optimizer_weight_pct"), 0)

        blockers = []
        if "BLOCK" in signal_action:
            blockers.append("signal blocked")
        if "REDUCE" in risk_action or "SIZE" in risk_action:
            blockers.append("risk repair")
        if "BLOCK" in status or "REVIEW" in status or "SIZE_DOWN" in status:
            blockers.append("optimizer review")
        if tca_status and tca_status not in {"CLEAR", "NAN"}:
            blockers.append("execution/TCA")
        if weight <= 0:
            blockers.append("zero final weight")

        if not blockers:
            role = "Potential contributor"
            can_help = "YES_AFTER_CONFIRMATION"
            next_step = "Can be considered only after price trigger, spread/TCA, and source proof remain clear."
        elif "risk repair" in blockers:
            role = "Risk repair first"
            can_help = "NO"
            next_step = "Do not use this to chase Sharpe. Repair risk first; upside review comes later."
        elif "signal blocked" in blockers:
            role = "Signal repair first"
            can_help = "NO"
            next_step = "Do not count this toward Sharpe 4 until the mapped signal is repaired and retested."
        else:
            role = "Review only"
            can_help = "NOT_YET"
            next_step = "Keep as research context until blocker evidence changes."

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", ""),
            "sharpe4_role": role,
            "can_help_target4": can_help,
            "final_optimizer_weight_pct": round(weight, 4),
            "current_blocker": "; ".join(dict.fromkeys(blockers)) if blockers else "none",
            "plain_next_step": next_step,
            "binding_constraints": constraints,
            "source_files": row.get("source_file", "institutional_optimizer_bridge.csv"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"NO": 0, "NOT_YET": 1, "YES_AFTER_CONFIRMATION": 2}
        out["_order"] = out["can_help_target4"].map(order).fillna(9)
        out = out.sort_values(["_order", "final_optimizer_weight_pct"], ascending=[True, False]).drop(columns=["_order"])
    return out


def write_report(base: dict[str, Any], adj: dict[str, Any], drivers: pd.DataFrame, queue: pd.DataFrame, policy: pd.DataFrame, tickers: pd.DataFrame) -> None:
    headline = safe_float(base.get("headline_sharpe"))
    planning = safe_float(adj.get("credibility_adjusted_planning_sharpe"))
    gap = TARGET_SHARPE - headline if np.isfinite(headline) else np.nan
    status = status_from_gap(headline)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
        "## Current Answer",
        "",
        f"- Target Sharpe: **{TARGET_SHARPE:.2f}**",
        f"- Current headline Sharpe: **{headline:.2f}**" if np.isfinite(headline) else "- Current headline Sharpe: **No data**",
        f"- Gap to target: **{gap:.2f}**" if np.isfinite(gap) else "- Gap to target: **No data**",
        f"- Credibility-adjusted planning Sharpe: **{planning:.2f}**" if np.isfinite(planning) else "- Credibility-adjusted planning Sharpe: **No data**",
        f"- Status: **{status}**",
        "",
        "Plain English: do not chase a Sharpe 4 label. The system must first prove cleaner signals, lower execution drag, stronger point-in-time truth, and more stable out-of-sample behavior.",
        ]),
        "## Driver Attribution\n\n" + df_to_markdown(drivers),
        "## Action Queue\n\n" + df_to_markdown(queue),
        "## Policy\n\n" + df_to_markdown(policy),
        "## Ticker Gate\n\n" + df_to_markdown(tickers.head(20)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 185 - Sharpe 4 Target Upgrade Desk", sections)


def main() -> None:
    base = performance_baseline()
    adj = credibility_adjustment(safe_float(base.get("headline_sharpe")))
    drivers = build_driver_attribution(base, adj)
    queue = build_action_queue(base, adj)
    policy = build_policy()
    tickers = build_ticker_gate()

    drivers.to_csv(OUT_DRIVERS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    policy.to_csv(OUT_POLICY, index=False)
    tickers.to_csv(OUT_TICKERS, index=False)

    headline = safe_float(base.get("headline_sharpe"))
    planning = safe_float(adj.get("credibility_adjusted_planning_sharpe"))
    state = {
        "date": today_str(),
        "target_sharpe": TARGET_SHARPE,
        "current_headline_sharpe": round(headline, 3) if np.isfinite(headline) else None,
        "gap_to_target": round(TARGET_SHARPE - headline, 3) if np.isfinite(headline) else None,
        "credibility_adjusted_planning_sharpe": round(planning, 3) if np.isfinite(planning) else None,
        "target_status": status_from_gap(headline),
        "claim_allowed": bool(np.isfinite(headline) and headline >= TARGET_SHARPE and np.isfinite(planning) and planning >= 2.5),
        "base_metrics": base,
        "credibility_adjustments": adj,
        "p0_action_count": int((queue["priority"] == "P0").sum()) if not queue.empty else 0,
        "ticker_gate_rows": int(len(tickers)),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "This is a target desk. It does not guarantee future Sharpe and does not promote overfit metrics.",
    }
    write_json(OUT_STATE, state)
    write_report(base, adj, drivers, queue, policy, tickers)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Current headline Sharpe: {state['current_headline_sharpe']}")
    print(f"[OK] Credibility-adjusted planning Sharpe: {state['credibility_adjusted_planning_sharpe']}")
    print(f"[OK] Gap to target: {state['gap_to_target']}")


if __name__ == "__main__":
    main()
