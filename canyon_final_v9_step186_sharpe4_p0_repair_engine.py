#!/usr/bin/env python3
"""
Canyon v9 Step 186 - Sharpe 4 P0 Repair Engine.

Research-only. No broker connection. No live orders.

Step185 showed the gap to Sharpe 4. Step186 turns the first repair layer into
concrete files:

  - risk repair weights before any Sharpe chasing
  - active-vs-research signal permissions
  - turnover and execution-cost caps
  - a plain-English P0 repair report

This does not claim Sharpe 4. It creates the control pack required before the
system can credibly try to improve Sharpe.

Outputs:
  sharpe4_p0_repair_state.json
  sharpe4_p0_ticker_repair_plan.csv
  sharpe4_p0_signal_policy_enforced.csv
  sharpe4_p0_execution_budget.csv
  sharpe4_p0_blocker_summary.csv
  sharpe4_p0_repair_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


TARGET_MONTHLY_TURNOVER_PCT = 45.0
TARGET_TCA_BPS = 10.0
LIVE_IC_MIN_OBS_FOR_ACTIVE_USE = 120

OUT_STATE = ROOT / "sharpe4_p0_repair_state.json"
OUT_TICKERS = ROOT / "sharpe4_p0_ticker_repair_plan.csv"
OUT_SIGNALS = ROOT / "sharpe4_p0_signal_policy_enforced.csv"
OUT_EXECUTION = ROOT / "sharpe4_p0_execution_budget.csv"
OUT_BLOCKERS = ROOT / "sharpe4_p0_blocker_summary.csv"
OUT_REPORT = ROOT / "sharpe4_p0_repair_report.md"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct_value(value: Any, default: float = 0.0) -> float:
    """Return a value that already lives in percentage-point units.

    The source files use columns named *_pct, but values like 0.546 mean
    0.546%, not 54.6%. Do not auto-convert small numbers here.
    """
    x = safe_float(value, default)
    if not np.isfinite(x):
        return default
    return x


def clean_status(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"", "NAN", "NONE"}:
        return ""
    return text


def first_existing(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return row.get(name)
    return default


def load_execution_context() -> pd.DataFrame:
    board = read_csv_safe(ROOT / "execution_tca_decision_board.csv")
    cost = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    cards = read_csv_safe(ROOT / "execution_tca_ticker_cards.csv")

    out = pd.DataFrame()
    if not board.empty and "ticker" in board.columns:
        out = board.copy()
        out["ticker"] = out["ticker"].apply(clean_ticker)
    if not cost.empty and "ticker" in cost.columns:
        cost = cost.copy()
        cost["ticker"] = cost["ticker"].apply(clean_ticker)
        keep = [c for c in [
            "ticker", "target_trade_dollars", "participation_rate_pct",
            "spread_bps_est", "total_tca_cost_bps", "execution_status",
            "avg_20d_dollar_volume",
        ] if c in cost.columns]
        out = out.merge(cost[keep], on="ticker", how="outer", suffixes=("", "_raw_est"))
    if not cards.empty and "ticker" in cards.columns:
        cards = cards.copy()
        cards["ticker"] = cards["ticker"].apply(clean_ticker)
        keep = [c for c in ["ticker", "headline", "cost_line", "blocker_line", "manual_check"] if c in cards.columns]
        out = out.merge(cards[keep], on="ticker", how="outer", suffixes=("", "_card"))
    return out


def build_ticker_repair_plan() -> pd.DataFrame:
    risk = read_csv_safe(ROOT / "final_risk_gate.csv")
    optimizer = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    execution = load_execution_context()

    if risk.empty or "ticker" not in risk.columns:
        return pd.DataFrame()

    base = risk.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)
    if not optimizer.empty and "ticker" in optimizer.columns:
        optimizer = optimizer.copy()
        optimizer["ticker"] = optimizer["ticker"].apply(clean_ticker)
        keep = [c for c in [
            "ticker", "subsector", "subsector_cycle_phase", "leadership_handoff_signal",
            "top_signal", "validation_signal", "signal_validation_action",
            "signal_policy_weight_multiplier", "final_optimizer_weight_pct",
            "final_optimizer_status", "binding_constraints", "why_not_more",
        ] if c in optimizer.columns]
        base = base.merge(optimizer[keep], on="ticker", how="left", suffixes=("", "_optimizer"))

    if not execution.empty and "ticker" in execution.columns:
        keep = [c for c in [
            "ticker", "execution_verdict", "execution_score_0_100", "primary_blockers",
            "trade_notional_dollars", "participation_rate_pct", "spread_bps",
            "spread_bps_est", "base_cost_bps", "stress_cost_bps",
            "total_tca_cost_bps", "expected_fill_rate_pct", "option_route",
            "option_side", "option_no_go_checks", "headline", "manual_check",
        ] if c in execution.columns]
        base = base.merge(execution[keep], on="ticker", how="left", suffixes=("", "_exec"))

    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        current = pct_value(row.get("current_weight_pct"))
        recommended = pct_value(row.get("recommended_risk_weight_pct"))
        optimizer_weight = pct_value(row.get("final_optimizer_weight_pct"), recommended)
        final_risk = clean_status(row.get("final_risk_action"))
        signal_action = clean_status(row.get("signal_validation_action"))
        execution_verdict = clean_status(row.get("execution_verdict"))
        no_go_checks = int(safe_float(row.get("option_no_go_checks"), 0) or 0)

        if final_risk == "REDUCE_ONLY":
            clean_weight = min(recommended, optimizer_weight if optimizer_weight > 0 else recommended)
            sharpe_alpha_weight = 0.0
            permission = "Risk reduction only"
            new_exposure = "No"
            option_permission = "No calls or new puts; hedge research only after manual proof"
        elif final_risk == "SIZE_DOWN":
            clean_weight = min(recommended, optimizer_weight if optimizer_weight > 0 else recommended)
            sharpe_alpha_weight = 0.0 if no_go_checks > 0 or "DATA_GAP" in execution_verdict else clean_weight
            permission = "Size down before review"
            new_exposure = "No"
            option_permission = "No options until spread, event, and risk checks clear"
        else:
            clean_weight = min(current, optimizer_weight if optimizer_weight > 0 else current)
            sharpe_alpha_weight = clean_weight
            permission = "Review for contribution"
            new_exposure = "Only after trigger and execution proof"
            option_permission = "Only if options route is separately clear"

        reduction = max(0.0, current - clean_weight)
        reduction_dollars = reduction / 100.0 * MODEL_ACCOUNT_VALUE
        active_gap = max(0.0, current - sharpe_alpha_weight)

        blockers = []
        if final_risk:
            blockers.append(f"risk={final_risk}")
        if signal_action in {"BLOCK_SIGNAL", "DOWNWEIGHT"}:
            blockers.append(f"signal={signal_action}")
        if execution_verdict and execution_verdict not in {"CLEAR", "READY"}:
            blockers.append(f"execution={execution_verdict}")
        if no_go_checks > 0:
            blockers.append(f"options_no_go={no_go_checks}")

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", ""),
            "subsector": row.get("subsector", ""),
            "cycle_phase": row.get("subsector_cycle_phase", ""),
            "current_weight_pct": round(current, 4),
            "p0_clean_weight_pct": round(clean_weight, 4),
            "sharpe4_alpha_weight_pct": round(sharpe_alpha_weight, 4),
            "required_weight_cut_pct": round(reduction, 4),
            "required_weight_cut_dollars": round(reduction_dollars, 2),
            "current_weight_not_counted_for_sharpe4_pct": round(active_gap, 4),
            "p0_permission": permission,
            "new_exposure_allowed": new_exposure,
            "option_permission": option_permission,
            "risk_action": final_risk or "NO_DATA",
            "signal_action": signal_action or "NO_SIGNAL_MAP",
            "execution_verdict": execution_verdict or "NO_EXECUTION_DATA",
            "estimated_tca_bps": round(safe_float(first_existing(row, ["base_cost_bps", "total_tca_cost_bps"]), np.nan), 2),
            "stress_tca_bps": round(safe_float(row.get("stress_cost_bps"), np.nan), 2),
            "expected_fill_rate_pct": round(safe_float(row.get("expected_fill_rate_pct"), np.nan), 1),
            "blocker_stack": "; ".join(blockers) if blockers else "none",
            "plain_next_step": (
                "Repair or cut risk weight first. Do not use this name to chase Sharpe 4 until risk, signal, execution, and option checks are clean."
                if blockers else
                "Can enter the Sharpe 4 research queue only after trigger and source proof are still clean."
            ),
            "source_files": "final_risk_gate.csv / institutional_optimizer_bridge.csv / execution_tca_decision_board.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["required_weight_cut_pct", "current_weight_pct"], ascending=[False, False])
    return out


def build_signal_policy() -> pd.DataFrame:
    policy = read_csv_safe(ROOT / "research_signal_weight_policy.csv")
    state = read_json_safe(ROOT / "signal_validation_state.json", {})
    live_obs_global = int(safe_float(state.get("live_ic_observations"), 0) or 0)
    if policy.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in policy.iterrows():
        signal = str(row.get("signal", "")).strip()
        action = clean_status(row.get("recommended_signal_action"))
        live_obs = int(safe_float(row.get("live_observations"), live_obs_global) or 0)
        research_mult = safe_float(row.get("weight_multiplier"), 0.0)

        if action == "BLOCK_SIGNAL":
            active_mult = 0.0
            active_use = "Blocked from active Sharpe 4 model"
            repair_action = "Set active weight to zero; rewrite or retest the signal before reuse."
        elif action == "DOWNWEIGHT":
            active_mult = 0.0 if live_obs < LIVE_IC_MIN_OBS_FOR_ACTIVE_USE else min(research_mult, 0.35)
            active_use = "Research only until live IC proves it" if active_mult == 0 else "Tiny active use with live IC monitor"
            repair_action = "Collect live IC observations by horizon and regime; keep as research-only before that."
        else:
            active_mult = min(max(research_mult, 0.0), 1.0)
            active_use = "Allowed with monitor"
            repair_action = "Keep monitoring decay, regime failure, and live-vs-backtest drift."

        rows.append({
            "signal": signal,
            "original_signal_action": action or "NO_DATA",
            "research_weight_multiplier": round(research_mult, 3),
            "sharpe4_active_multiplier": round(active_mult, 3),
            "active_use": active_use,
            "allowed_horizon": row.get("allowed_horizon", ""),
            "baseline_mean_ic": row.get("baseline_mean_ic", np.nan),
            "best_horizon": row.get("best_horizon", ""),
            "worst_horizon": row.get("worst_horizon", ""),
            "live_observations": live_obs,
            "repair_action": repair_action,
            "proof_required": row.get("proof_required", ""),
            "source_files": row.get("source_files", "research_signal_weight_policy.csv"),
            "research_only": True,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"Blocked from active Sharpe 4 model": 0, "Research only until live IC proves it": 1, "Tiny active use with live IC monitor": 2, "Allowed with monitor": 3}
        out["_order"] = out["active_use"].map(order).fillna(9)
        out = out.sort_values(["_order", "sharpe4_active_multiplier", "signal"]).drop(columns=["_order"])
    return out


def build_execution_budget(ticker_plan: pd.DataFrame) -> pd.DataFrame:
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    reality = read_csv_safe(ROOT / "backtest_execution_reality_check.csv")
    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    execution_state = read_json_safe(ROOT / "execution_tca_state.json", {})

    median_turnover = np.nan
    p90_turnover = np.nan
    median_backtest_cost = np.nan
    if not monthly.empty:
        if "turnover_pct" in monthly.columns:
            turnover = pd.to_numeric(monthly["turnover_pct"], errors="coerce").dropna()
            if not turnover.empty:
                median_turnover = float(turnover.median())
                p90_turnover = float(turnover.quantile(0.90))
        if "tc_cost_bps" in monthly.columns:
            costs = pd.to_numeric(monthly["tc_cost_bps"], errors="coerce").dropna()
            if not costs.empty:
                median_backtest_cost = float(costs.median())

    current_avg_tca = np.nan
    current_med_tca = np.nan
    if not tca.empty and "total_tca_cost_bps" in tca.columns:
        vals = pd.to_numeric(tca["total_tca_cost_bps"], errors="coerce").dropna()
        if not vals.empty:
            current_avg_tca = float(vals.mean())
            current_med_tca = float(vals.median())

    current_gross = safe_float(ticker_plan.get("current_weight_pct", pd.Series(dtype=float)).sum(), np.nan) if not ticker_plan.empty else np.nan
    clean_gross = safe_float(ticker_plan.get("p0_clean_weight_pct", pd.Series(dtype=float)).sum(), np.nan) if not ticker_plan.empty else np.nan
    alpha_gross = safe_float(ticker_plan.get("sharpe4_alpha_weight_pct", pd.Series(dtype=float)).sum(), np.nan) if not ticker_plan.empty else np.nan
    cut_dollars = safe_float(ticker_plan.get("required_weight_cut_dollars", pd.Series(dtype=float)).sum(), 0.0) if not ticker_plan.empty else 0.0

    turnover_multiplier = TARGET_MONTHLY_TURNOVER_PCT / median_turnover if np.isfinite(median_turnover) and median_turnover > 0 else np.nan
    turnover_multiplier = min(1.0, turnover_multiplier) if np.isfinite(turnover_multiplier) else np.nan

    rows = [
        {
            "budget_area": "Monthly turnover",
            "current_value": round(median_turnover, 2) if np.isfinite(median_turnover) else np.nan,
            "target_value": TARGET_MONTHLY_TURNOVER_PCT,
            "status": "REPAIR_REQUIRED" if np.isfinite(median_turnover) and median_turnover > TARGET_MONTHLY_TURNOVER_PCT else "CLEAR",
            "p0_rule": "Cap monthly turnover before any Sharpe 4 claim.",
            "repair_instruction": f"Use at most {turnover_multiplier:.2f}x of current turnover pace." if np.isfinite(turnover_multiplier) else "Need turnover data.",
            "source_files": "backtest_monthly_perf.csv",
        },
        {
            "budget_area": "90th percentile turnover",
            "current_value": round(p90_turnover, 2) if np.isfinite(p90_turnover) else np.nan,
            "target_value": 70.0,
            "status": "REPAIR_REQUIRED" if np.isfinite(p90_turnover) and p90_turnover > 70.0 else "CLEAR",
            "p0_rule": "Stop high-churn months from dominating performance.",
            "repair_instruction": "Throttle rebalances; do not rotate full sleeves unless risk is being reduced.",
            "source_files": "backtest_monthly_perf.csv",
        },
        {
            "budget_area": "Current TCA vs target",
            "current_value": round(current_avg_tca, 2) if np.isfinite(current_avg_tca) else np.nan,
            "target_value": TARGET_TCA_BPS,
            "status": "REPAIR_REQUIRED" if np.isfinite(current_avg_tca) and current_avg_tca > TARGET_TCA_BPS else "CLEAR",
            "p0_rule": "Execution cost must be below the target before alpha is counted.",
            "repair_instruction": "Require spread/liquidity proof and smaller slices; do not count high-cost ideas toward Sharpe 4.",
            "source_files": "institutional_tca_cost_estimates.csv / execution_tca_state.json",
        },
        {
            "budget_area": "Backtest cost realism",
            "current_value": round(median_backtest_cost, 2) if np.isfinite(median_backtest_cost) else np.nan,
            "target_value": TARGET_TCA_BPS,
            "status": "WEAK" if np.isfinite(current_avg_tca) and np.isfinite(median_backtest_cost) and current_avg_tca > median_backtest_cost + 3.0 else "REVIEW",
            "p0_rule": "Do not trust a Sharpe number when current TCA is much higher than backtest cost.",
            "repair_instruction": "Use current TCA or stress TCA in backtest until the gap closes.",
            "source_files": "backtest_execution_reality_check.csv",
        },
        {
            "budget_area": "Gross exposure after P0 repair",
            "current_value": round(current_gross, 2) if np.isfinite(current_gross) else np.nan,
            "target_value": round(clean_gross, 2) if np.isfinite(clean_gross) else np.nan,
            "status": "RISK_REPAIR_REQUIRED" if np.isfinite(clean_gross) and np.isfinite(current_gross) and clean_gross < current_gross else "CLEAR",
            "p0_rule": "Use clean risk-gated weights before any new idea review.",
            "repair_instruction": f"Research-model cut required: about ${cut_dollars:,.0f} on a ${MODEL_ACCOUNT_VALUE:,.0f} model account.",
            "source_files": "final_risk_gate.csv / sharpe4_p0_ticker_repair_plan.csv",
        },
        {
            "budget_area": "Alpha gross allowed for Sharpe 4",
            "current_value": round(current_gross, 2) if np.isfinite(current_gross) else np.nan,
            "target_value": round(alpha_gross, 2) if np.isfinite(alpha_gross) else np.nan,
            "status": "BLOCKED" if np.isfinite(alpha_gross) and alpha_gross <= 0 else "REVIEW",
            "p0_rule": "Risk-reduction-only names do not count as Sharpe 4 alpha contributors.",
            "repair_instruction": "First clear risk, signal, and execution blockers; then rebuild alpha sleeves from clean candidates.",
            "source_files": "sharpe4_p0_ticker_repair_plan.csv",
        },
        {
            "budget_area": "Execution desk status",
            "current_value": execution_state.get("overall_execution_tca_score", np.nan),
            "target_value": 70.0,
            "status": str(execution_state.get("status", "NO_DATA")),
            "p0_rule": "The execution layer cannot be blocked while claiming a high-Sharpe target.",
            "repair_instruction": "Fix spread data gaps, failed-fill assumptions, and risk-reduction routing before new exposure.",
            "source_files": "execution_tca_state.json",
        },
    ]

    if not reality.empty:
        for _, row in reality.iterrows():
            rows.append({
                "budget_area": str(row.get("check", "Execution check")),
                "current_value": row.get("stress_cost_bps", np.nan),
                "target_value": TARGET_TCA_BPS,
                "status": row.get("status", "REVIEW"),
                "p0_rule": str(row.get("evidence", "")),
                "repair_instruction": "Keep this as a P0 blocker until the evidence improves.",
                "source_files": row.get("source_file", "backtest_execution_reality_check.csv"),
            })
    return pd.DataFrame(rows)


def build_blocker_summary(ticker_plan: pd.DataFrame, signal_policy: pd.DataFrame, execution_budget: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hard_risk = int(ticker_plan["risk_action"].astype(str).str.contains("REDUCE_ONLY", na=False).sum()) if not ticker_plan.empty and "risk_action" in ticker_plan.columns else 0
    size_down = int(ticker_plan["risk_action"].astype(str).str.contains("SIZE_DOWN", na=False).sum()) if not ticker_plan.empty and "risk_action" in ticker_plan.columns else 0
    blocked_signals = int(signal_policy["original_signal_action"].astype(str).str.contains("BLOCK_SIGNAL", na=False).sum()) if not signal_policy.empty else 0
    research_only_signals = int(signal_policy["active_use"].astype(str).str.contains("Research only", na=False).sum()) if not signal_policy.empty else 0
    execution_repairs = int(execution_budget["status"].astype(str).str.upper().str.contains("REPAIR|WEAK|BLOCK|DATA_GAP", na=False).sum()) if not execution_budget.empty else 0
    alpha_names = int((ticker_plan.get("sharpe4_alpha_weight_pct", pd.Series(dtype=float)) > 0).sum()) if not ticker_plan.empty else 0

    rows.extend([
        {
            "blocker": "Hard risk names",
            "count": hard_risk,
            "status": "P0_BLOCKER" if hard_risk else "CLEAR",
            "plain_english": "These names are reduce-only and cannot be used to chase Sharpe 4.",
            "source_files": "final_risk_gate.csv",
        },
        {
            "blocker": "Size-down names",
            "count": size_down,
            "status": "P0_REVIEW" if size_down else "CLEAR",
            "plain_english": "These names must be cut to a smaller research weight before review.",
            "source_files": "final_risk_gate.csv",
        },
        {
            "blocker": "Blocked signals",
            "count": blocked_signals,
            "status": "P0_BLOCKER" if blocked_signals else "CLEAR",
            "plain_english": "These signals get zero active weight in the Sharpe 4 model.",
            "source_files": "research_signal_weight_policy.csv",
        },
        {
            "blocker": "Research-only signals",
            "count": research_only_signals,
            "status": "P0_REVIEW" if research_only_signals else "CLEAR",
            "plain_english": "These may be studied, but not counted as active alpha until live IC proof exists.",
            "source_files": "research_signal_weight_policy.csv / signal_validation_state.json",
        },
        {
            "blocker": "Execution budget repairs",
            "count": execution_repairs,
            "status": "P0_BLOCKER" if execution_repairs else "CLEAR",
            "plain_english": "Turnover, TCA, spread data, or fill assumptions still block a credible Sharpe 4 claim.",
            "source_files": "backtest_execution_reality_check.csv / execution_tca_state.json",
        },
        {
            "blocker": "Names allowed to count as Sharpe 4 alpha now",
            "count": alpha_names,
            "status": "NO_ALPHA_NAMES_YET" if alpha_names == 0 else "REVIEW",
            "plain_english": "If this is zero, the desk is still in repair mode, not alpha-chasing mode.",
            "source_files": "sharpe4_p0_ticker_repair_plan.csv",
        },
    ])
    return pd.DataFrame(rows)


def write_report(state: dict[str, Any], tickers: pd.DataFrame, signals: pd.DataFrame, execution: pd.DataFrame, blockers: pd.DataFrame) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- P0 repair status: **{state['p0_repair_status']}**",
            f"- Current gross exposure in repair book: **{state['current_gross_pct']:.2f}%**",
            f"- P0 clean gross after repair: **{state['p0_clean_gross_pct']:.2f}%**",
            f"- Alpha gross allowed for Sharpe 4 now: **{state['sharpe4_alpha_gross_allowed_pct']:.2f}%**",
            f"- Blocked active signals: **{state['blocked_signal_count']}**",
            f"- Median monthly turnover: **{state['median_monthly_turnover_pct']:.2f}%**, target **{TARGET_MONTHLY_TURNOVER_PCT:.2f}%**",
            "",
            "Plain English: the model is still in repair mode. Do not look for new calls, puts, or larger size until risk, signal, and execution blockers clear.",
        ]),
        "## Blocker Summary\n\n" + df_to_markdown(blockers),
        "## Ticker Repair Plan\n\n" + df_to_markdown(tickers.head(20)),
        "## Signal Policy Enforced\n\n" + df_to_markdown(signals),
        "## Execution Budget\n\n" + df_to_markdown(execution),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 186 - Sharpe 4 P0 Repair Engine", sections)


def main() -> None:
    tickers = build_ticker_repair_plan()
    signals = build_signal_policy()
    execution = build_execution_budget(tickers)
    blockers = build_blocker_summary(tickers, signals, execution)

    tickers.to_csv(OUT_TICKERS, index=False)
    signals.to_csv(OUT_SIGNALS, index=False)
    execution.to_csv(OUT_EXECUTION, index=False)
    blockers.to_csv(OUT_BLOCKERS, index=False)

    current_gross = safe_float(tickers.get("current_weight_pct", pd.Series(dtype=float)).sum(), 0.0) if not tickers.empty else 0.0
    clean_gross = safe_float(tickers.get("p0_clean_weight_pct", pd.Series(dtype=float)).sum(), 0.0) if not tickers.empty else 0.0
    alpha_gross = safe_float(tickers.get("sharpe4_alpha_weight_pct", pd.Series(dtype=float)).sum(), 0.0) if not tickers.empty else 0.0
    required_cut = safe_float(tickers.get("required_weight_cut_pct", pd.Series(dtype=float)).sum(), 0.0) if not tickers.empty else 0.0

    blocked_signal_count = int(signals["original_signal_action"].astype(str).str.contains("BLOCK_SIGNAL", na=False).sum()) if not signals.empty else 0
    research_only_signal_count = int(signals["active_use"].astype(str).str.contains("Research only", na=False).sum()) if not signals.empty else 0
    active_signal_count = int((signals["sharpe4_active_multiplier"] > 0).sum()) if not signals.empty and "sharpe4_active_multiplier" in signals.columns else 0
    hard_risk_count = int(tickers["risk_action"].astype(str).str.contains("REDUCE_ONLY", na=False).sum()) if not tickers.empty and "risk_action" in tickers.columns else 0

    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    turnover = pd.to_numeric(monthly.get("turnover_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    median_turnover = float(turnover.median()) if not turnover.empty else np.nan
    tca_vals = pd.to_numeric(read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv").get("total_tca_cost_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    avg_tca = float(tca_vals.mean()) if not tca_vals.empty else np.nan

    p0_clear = (
        hard_risk_count == 0
        and blocked_signal_count == 0
        and active_signal_count > 0
        and np.isfinite(median_turnover)
        and median_turnover <= TARGET_MONTHLY_TURNOVER_PCT
        and np.isfinite(avg_tca)
        and avg_tca <= TARGET_TCA_BPS
    )

    state = {
        "date": today_str(),
        "p0_repair_status": "P0_CLEAR_FOR_NEXT_STAGE" if p0_clear else "P0_REPAIR_REQUIRED",
        "current_gross_pct": round(current_gross, 4),
        "p0_clean_gross_pct": round(clean_gross, 4),
        "sharpe4_alpha_gross_allowed_pct": round(alpha_gross, 4),
        "required_gross_cut_pct": round(required_cut, 4),
        "required_model_cut_dollars": round(required_cut / 100.0 * MODEL_ACCOUNT_VALUE, 2),
        "hard_risk_ticker_count": hard_risk_count,
        "blocked_signal_count": blocked_signal_count,
        "research_only_signal_count": research_only_signal_count,
        "active_signal_count": active_signal_count,
        "median_monthly_turnover_pct": round(median_turnover, 4) if np.isfinite(median_turnover) else None,
        "target_monthly_turnover_pct": TARGET_MONTHLY_TURNOVER_PCT,
        "avg_current_tca_bps": round(avg_tca, 4) if np.isfinite(avg_tca) else None,
        "target_tca_bps": TARGET_TCA_BPS,
        "claim_allowed_after_p0": False,
        "why_claim_still_blocked": (
            "P0 repair is not clear yet."
            if not p0_clear else
            "P0 is clear, but Sharpe 4 still needs walk-forward, live IC, and point-in-time proof before any claim."
        ),
        "next_stage_after_p0": "Walk-forward OOS, live IC, PIT truth, and cost-aware backtest retest.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "Step186 is a repair-control layer. It cannot place trades and cannot make Sharpe 4 true by itself.",
    }
    write_json(OUT_STATE, state)
    write_report(state, tickers, signals, execution, blockers)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] P0 status: {state['p0_repair_status']}")
    print(f"[OK] Current gross: {state['current_gross_pct']}%")
    print(f"[OK] P0 clean gross: {state['p0_clean_gross_pct']}%")
    print(f"[OK] Alpha gross allowed now: {state['sharpe4_alpha_gross_allowed_pct']}%")
    print(f"[OK] Blocked signals: {state['blocked_signal_count']}")


if __name__ == "__main__":
    main()
