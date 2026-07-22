#!/usr/bin/env python3
"""
Canyon v9 - Step 131: Risk Desk Summary
=======================================

Research-only. No broker connection. No live orders.

Step111-118 create the risk controls. This step turns those controls into a
single desk summary that the dashboard can show without forcing the user to
open ten CSV files.

Outputs:
  risk_desk_overview.json
  risk_desk_breach_table.csv
  risk_desk_ticker_action_queue.csv
  risk_desk_component_health.csv
  risk_desk_summary_report.md
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

OUT_OVERVIEW = ROOT / "risk_desk_overview.json"
OUT_BREACH = ROOT / "risk_desk_breach_table.csv"
OUT_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
OUT_HEALTH = ROOT / "risk_desk_component_health.csv"
OUT_REPORT = ROOT / "risk_desk_summary_report.md"

INPUTS = {
    "single_name": "single_name_risk_budget.csv",
    "sector": "sector_active_exposure.csv",
    "theme_factor": "theme_factor_exposure.csv",
    "macro_sensitivity": "portfolio_macro_sensitivity.csv",
    "macro_stress": "macro_scenario_stress.csv",
    "drawdown": "drawdown_control_state.json",
    "vol_target": "vol_target_state.json",
    "kelly": "kelly_position_sizing.csv",
    "paper_nav": "paper_nav_curve.csv",
    "ticker_attribution": "return_attribution_by_ticker.csv",
    "slippage": "slippage_model_report.csv",
    "correlation": "holdings_correlation_matrix.csv",
    "beta": "portfolio_beta_report.csv",
    "crisis_correlation": "crisis_correlation_stress.csv",
    "tail_hedge": "tail_hedge_budget.csv",
    "liquidity_crisis": "liquidity_crisis_simulation.csv",
    "earnings_gap": "earnings_gap_down_risk.csv",
    "crisis_override": "crisis_correlation_override.csv",
    "risk_budget": "institutional_risk_budget_summary.csv",
    "portfolio_var": "portfolio_var_cvar_summary.csv",
    "factor_exposure": "factor_exposure_decomposition.csv",
    "final_gate": "final_risk_gate.csv",
    "master_state": "institutional_risk_gate_state.json",
}


ACTION_RANK = {
    "CLEAR": 0,
    "OK": 0,
    "WATCH": 1,
    "REVIEW": 2,
    "MISSING_DATA_REVIEW": 3,
    "SIZE_DOWN": 4,
    "REDUCE_ONLY": 5,
    "BLOCK_NEW": 6,
    "BLOCKED": 6,
}


def read_csv_safe(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_safe(name: str) -> dict[str, Any]:
    path = ROOT / name
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def numeric(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def rank_action(value: Any) -> int:
    return ACTION_RANK.get(str(value).upper(), 2)


def pct(value: Any) -> float:
    x = numeric(value)
    if not np.isfinite(x):
        return np.nan
    return x * 100.0


def dollars(value: Any) -> float:
    x = numeric(value)
    return round(x, 2) if np.isfinite(x) else np.nan


def status_bucket(value: Any) -> str:
    text = str(value).upper()
    if text in {"BLOCK_NEW", "BLOCKED", "REDUCE_ONLY"}:
        return "Hard risk"
    if text in {"SIZE_DOWN", "MISSING_DATA_REVIEW"}:
        return "Size down"
    if text == "REVIEW":
        return "Review"
    return "Clear"


def plain_next_step(action: Any, reason: Any) -> str:
    action_text = str(action).upper()
    reason_text = str(reason or "")
    if action_text in {"BLOCK_NEW", "BLOCKED"}:
        return "Do not add new exposure. Resolve the listed risk driver first."
    if action_text == "REDUCE_ONLY":
        return "Research only reduction. New buying is blocked by the risk gate."
    if action_text == "SIZE_DOWN":
        return "Cut paper size to the recommended risk weight before considering any idea."
    if action_text == "MISSING_DATA_REVIEW":
        return "Do not upgrade. Fill missing data before using this signal."
    if action_text == "REVIEW":
        return "Manual review required before paper action."
    if "earnings" in reason_text.lower():
        return "Check earnings date, implied move, and gap risk before action."
    return "Risk clear at current thresholds."


def source_health() -> pd.DataFrame:
    rows = []
    now = time.time()
    for component, fname in INPUTS.items():
        path = ROOT / fname
        exists = path.exists() and path.stat().st_size > 0
        if not exists:
            rows.append({
                "component": component,
                "file": fname,
                "status": "MISSING",
                "rows": 0,
                "age_minutes": np.nan,
                "required_next_action": "Run the upstream step or inspect why the file was not produced.",
            })
            continue
        rows_count = np.nan
        if path.suffix == ".csv":
            df = read_csv_safe(fname)
            rows_count = len(df)
            status = "OK" if rows_count > 0 else "EMPTY"
        else:
            status = "OK"
        age_min = (now - path.stat().st_mtime) / 60.0
        if age_min > 24 * 60:
            status = "STALE"
        rows.append({
            "component": component,
            "file": fname,
            "status": status,
            "rows": int(rows_count) if np.isfinite(rows_count) else np.nan,
            "age_minutes": round(age_min, 1),
            "required_next_action": "OK" if status == "OK" else "Refresh and verify this component.",
        })
    return pd.DataFrame(rows)


def build_breach_table(budget: pd.DataFrame) -> pd.DataFrame:
    if budget.empty:
        return pd.DataFrame()
    out = budget.copy()
    if "used_pct" in out.columns:
        out["used_pct"] = pd.to_numeric(out["used_pct"], errors="coerce")
        out["used_pct_display"] = (out["used_pct"] * 100.0).round(1)
    out["severity_rank"] = out.get("status", pd.Series("", index=out.index)).map(rank_action)
    out["status_bucket"] = out.get("status", pd.Series("", index=out.index)).map(status_bucket)
    out["required_next_action"] = out.apply(
        lambda r: plain_next_step(r.get("status", ""), r.get("budget_item", "")),
        axis=1,
    )
    out = out.sort_values(["severity_rank", "used_pct"], ascending=[False, False]).reset_index(drop=True)
    cols = [
        "scope", "budget_item", "current_value", "limit_value", "used_pct",
        "used_pct_display", "status", "status_bucket", "action_if_breached",
        "required_next_action", "source_file",
    ]
    return out[[c for c in cols if c in out.columns]]


def build_action_queue(final_gate: pd.DataFrame, single: pd.DataFrame, earnings: pd.DataFrame, kelly: pd.DataFrame) -> pd.DataFrame:
    if final_gate.empty:
        return pd.DataFrame()
    q = final_gate.copy()
    q["ticker"] = q["ticker"].astype(str).str.upper()

    if not single.empty and "ticker" in single.columns:
        single_keep = [c for c in [
            "ticker", "var_95_1d", "cvar_95_1d", "var_95_5d", "cvar_95_5d",
            "risk_budget_used_pct", "earnings_days_to_event", "implied_move",
            "liquidity_label", "single_name_stop_level",
        ] if c in single.columns]
        q = q.merge(single[single_keep], on="ticker", how="left", suffixes=("", "_single"))

    if not earnings.empty and "ticker" in earnings.columns:
        e_keep = [c for c in ["ticker", "gap_down_action", "estimated_gap_loss_model_account", "data_status"] if c in earnings.columns]
        q = q.merge(earnings[e_keep], on="ticker", how="left", suffixes=("", "_earnings"))

    if not kelly.empty and "ticker" in kelly.columns:
        k_keep = [c for c in ["ticker", "recommended_kelly_weight_pct", "kelly_status", "ic_sample_confidence"] if c in kelly.columns]
        q = q.merge(kelly[k_keep], on="ticker", how="left", suffixes=("", "_kelly"))

    q["severity_rank"] = q.get("final_risk_action", pd.Series("", index=q.index)).map(rank_action)
    q["status_bucket"] = q.get("final_risk_action", pd.Series("", index=q.index)).map(status_bucket)
    q["required_next_action"] = q.apply(
        lambda r: plain_next_step(r.get("final_risk_action", ""), r.get("reason_stack", "")),
        axis=1,
    )
    for col in [
        "current_weight_pct", "recommended_risk_weight_pct", "risk_reduction_pct_of_current",
        "var_95_1d", "cvar_95_1d", "var_95_5d", "cvar_95_5d",
        "risk_budget_used_pct", "implied_move", "estimated_gap_loss_model_account",
        "recommended_kelly_weight_pct",
    ]:
        if col in q.columns:
            q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.sort_values(["severity_rank", "current_weight_pct"], ascending=[False, False]).reset_index(drop=True)
    cols = [
        "ticker", "sector", "current_action", "current_weight_pct",
        "recommended_risk_weight_pct", "risk_reduction_pct_of_current",
        "final_risk_action", "status_bucket", "master_risk_action",
        "single_name_action", "earnings_gap_action", "gap_down_action",
        "kelly_status", "liquidity_crisis_status", "sector_status",
        "var_95_1d", "cvar_95_1d", "risk_budget_used_pct",
        "earnings_days_to_event", "implied_move", "estimated_gap_loss_model_account",
        "liquidity_label", "single_name_stop_level", "recommended_kelly_weight_pct",
        "reason_stack", "required_next_action", "source_file",
    ]
    return q[[c for c in cols if c in q.columns]]


def build_overview(
    budget: pd.DataFrame,
    breach: pd.DataFrame,
    queue: pd.DataFrame,
    portfolio_var: pd.DataFrame,
    master_state: dict[str, Any],
    drawdown: dict[str, Any],
    vol_state: dict[str, Any],
    crisis: pd.DataFrame,
    macro: pd.DataFrame,
    tail: pd.DataFrame,
    health: pd.DataFrame,
) -> dict[str, Any]:
    var_row = portfolio_var.iloc[0].to_dict() if not portfolio_var.empty else {}
    queue_actions = queue.get("final_risk_action", pd.Series(dtype=str)).astype(str).str.upper()
    breach_status = breach.get("status", pd.Series(dtype=str)).astype(str).str.upper()
    worst_macro = {}
    if not macro.empty and "conservative_portfolio_impact" in macro.columns:
        m = macro.copy()
        m["conservative_portfolio_impact"] = pd.to_numeric(m["conservative_portfolio_impact"], errors="coerce")
        if m["conservative_portfolio_impact"].notna().any():
            worst_macro = m.sort_values("conservative_portfolio_impact").iloc[0].to_dict()
    crisis_row = crisis.iloc[0].to_dict() if not crisis.empty else {}

    overview = {
        "run_time": datetime.now().replace(microsecond=0).isoformat(),
        "master_risk_action": master_state.get("master_risk_action", "NO_DATA"),
        "master_exposure_multiplier": numeric(master_state.get("master_exposure_multiplier", np.nan)),
        "normal_gross_exposure": numeric(var_row.get("gross_exposure", np.nan)),
        "recommended_gross_exposure": numeric(var_row.get("gross_exposure", np.nan)) * numeric(master_state.get("master_exposure_multiplier", np.nan)),
        "annual_vol_pct": pct(var_row.get("annual_vol", np.nan)),
        "target_vol_pct": pct(vol_state.get("target_annual_vol", np.nan)),
        "vol_exposure_multiplier": numeric(vol_state.get("vol_exposure_multiplier", np.nan)),
        "var_95_1d_pct": pct(var_row.get("var_95_1d", np.nan)),
        "cvar_95_1d_pct": pct(var_row.get("cvar_95_1d", np.nan)),
        "var_95_20d_pct": pct(var_row.get("var_95_20d", np.nan)),
        "cvar_95_20d_pct": pct(var_row.get("cvar_95_20d", np.nan)),
        "var_95_1d_dollars": dollars(var_row.get("var_95_1d_dollars", np.nan)),
        "cvar_95_1d_dollars": dollars(var_row.get("cvar_95_1d_dollars", np.nan)),
        "drawdown_pct": pct(drawdown.get("drawdown_pct", np.nan)),
        "drawdown_action": drawdown.get("drawdown_action", "NO_DATA"),
        "crisis_vol_increase_ratio": numeric(crisis_row.get("vol_increase_ratio", np.nan)),
        "crisis_action": crisis_row.get("stress_action", "NO_DATA"),
        "worst_macro_scenario": worst_macro.get("scenario", "NO_DATA"),
        "worst_macro_impact_pct": pct(worst_macro.get("conservative_portfolio_impact", np.nan)),
        "tail_hedge_rows": int(len(tail)),
        "ticker_reduce_only_count": int(queue_actions.isin(["REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"]).sum()),
        "ticker_size_down_count": int(queue_actions.isin(["SIZE_DOWN", "MISSING_DATA_REVIEW"]).sum()),
        "ticker_review_count": int((queue_actions == "REVIEW").sum()),
        "budget_hard_breach_count": int(breach_status.isin(["REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"]).sum()),
        "budget_size_down_count": int(breach_status.isin(["SIZE_DOWN", "MISSING_DATA_REVIEW"]).sum()),
        "missing_component_count": int(health.get("status", pd.Series(dtype=str)).astype(str).isin(["MISSING", "EMPTY", "STALE"]).sum()),
        "top_risk_drivers": master_state.get("master_reason_stack", [])[:8],
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Risk can reduce or block. Risk cannot upgrade. Options cannot override risk. Missing data cannot improve decisions.",
    }
    return overview


def df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_csv(index=False)


def write_report(overview: dict[str, Any], breach: pd.DataFrame, queue: pd.DataFrame, health: pd.DataFrame) -> None:
    lines = [
        "# Canyon v9 - Step 131 Risk Desk Summary",
        f"Generated: {overview.get('run_time')}",
        "",
        "Research-only. No broker connection. No live orders.",
        "",
        "## Desk Overview",
        f"- Master risk action: {overview.get('master_risk_action')}",
        f"- Master exposure multiplier: {overview.get('master_exposure_multiplier')}",
        f"- 1d VaR 95%: {overview.get('var_95_1d_pct'):.2f}%" if np.isfinite(numeric(overview.get("var_95_1d_pct"))) else "- 1d VaR 95%: no data",
        f"- 1d CVaR 95%: {overview.get('cvar_95_1d_pct'):.2f}%" if np.isfinite(numeric(overview.get("cvar_95_1d_pct"))) else "- 1d CVaR 95%: no data",
        f"- Worst macro scenario: {overview.get('worst_macro_scenario')} ({overview.get('worst_macro_impact_pct'):.2f}%)" if np.isfinite(numeric(overview.get("worst_macro_impact_pct"))) else f"- Worst macro scenario: {overview.get('worst_macro_scenario')}",
        "",
        "## Budget Breaches",
        df_to_md(breach, 40),
        "",
        "## Ticker Action Queue",
        df_to_md(queue, 40),
        "",
        "## Component Health",
        df_to_md(health, 50),
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("[step131] building consolidated risk desk summary")
    budget = read_csv_safe(INPUTS["risk_budget"])
    portfolio_var = read_csv_safe(INPUTS["portfolio_var"])
    final_gate = read_csv_safe(INPUTS["final_gate"])
    single = read_csv_safe(INPUTS["single_name"])
    earnings = read_csv_safe(INPUTS["earnings_gap"])
    kelly = read_csv_safe(INPUTS["kelly"])
    crisis = read_csv_safe(INPUTS["crisis_correlation"])
    macro = read_csv_safe(INPUTS["macro_stress"])
    tail = read_csv_safe(INPUTS["tail_hedge"])
    master_state = read_json_safe(INPUTS["master_state"])
    drawdown = read_json_safe(INPUTS["drawdown"])
    vol_state = read_json_safe(INPUTS["vol_target"])

    health = source_health()
    breach = build_breach_table(budget)
    queue = build_action_queue(final_gate, single, earnings, kelly)
    overview = build_overview(
        budget, breach, queue, portfolio_var, master_state, drawdown,
        vol_state, crisis, macro, tail, health,
    )

    OUT_OVERVIEW.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    breach.to_csv(OUT_BREACH, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    health.to_csv(OUT_HEALTH, index=False)
    write_report(overview, breach, queue, health)

    print(f"[step131] wrote {OUT_OVERVIEW.name}")
    print(f"[step131] wrote {OUT_BREACH.name}: {len(breach)} rows")
    print(f"[step131] wrote {OUT_QUEUE.name}: {len(queue)} rows")
    print(f"[step131] wrote {OUT_HEALTH.name}: {len(health)} rows")
    print(
        f"[step131] master={overview.get('master_risk_action')} "
        f"mult={overview.get('master_exposure_multiplier')} "
        f"hard_tickers={overview.get('ticker_reduce_only_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
