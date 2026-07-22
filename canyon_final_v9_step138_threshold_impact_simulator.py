#!/usr/bin/env python3
"""
Canyon v9 - Step 138: Threshold Impact Simulator
================================================

Research-only. No broker connection. No live orders.

Step137 creates draft-only historical threshold recommendations. Step138 asks:
"If a reviewer approved this change, what would it do to the current risk desk?"

This step does not apply any policy. It simulates impact only.

Outputs:
  risk_threshold_impact_simulation.csv
  risk_threshold_impact_decision_table.csv
  risk_threshold_impact_state.json
  risk_threshold_impact_report.md
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
    read_json_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_RECS = ROOT / "risk_historical_threshold_recommendations.csv"
IN_QUEUE = ROOT / "risk_historical_threshold_review_queue.csv"
IN_BREACH = ROOT / "risk_desk_breach_table.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"

OUT_SIM = ROOT / "risk_threshold_impact_simulation.csv"
OUT_DECISIONS = ROOT / "risk_threshold_impact_decision_table.csv"
OUT_STATE = ROOT / "risk_threshold_impact_state.json"
OUT_REPORT = ROOT / "risk_threshold_impact_report.md"


SEVERITY_RANK = {
    "CLEAR": 0,
    "WATCH": 1,
    "REVIEW": 2,
    "SIZE_DOWN": 3,
    "REDUCE_ONLY": 4,
    "BLOCK_NEW": 4,
    "BLOCKED": 5,
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def fmt_value(value: float, policy_type: str) -> str:
    if not np.isfinite(value):
        return "NA"
    if policy_type == "risk_multiplier":
        return f"{value:.2f}x"
    return f"{value * 100:.2f}%"


def status_from_usage(usage: float) -> str:
    if not np.isfinite(usage):
        return "NO_DATA"
    if usage >= 1.50:
        return "REDUCE_ONLY"
    if usage >= 1.00:
        return "SIZE_DOWN"
    if usage >= 0.85:
        return "REVIEW"
    if usage >= 0.70:
        return "WATCH"
    return "CLEAR"


def exposure_multiplier_for_status(status: str, current_master_multiplier: float) -> float:
    status = str(status).upper()
    if status in {"BLOCKED"}:
        return 0.0
    if status in {"REDUCE_ONLY", "BLOCK_NEW"}:
        return min(current_master_multiplier, 0.50)
    if status == "SIZE_DOWN":
        return min(current_master_multiplier, 0.70)
    return current_master_multiplier


def rank(status: str) -> int:
    return SEVERITY_RANK.get(str(status).upper(), 2)


def lookup_breach_row(breach: pd.DataFrame, control: str) -> dict[str, Any]:
    if breach.empty or "budget_item" not in breach.columns:
        return {}
    m = breach["budget_item"].astype(str).str.lower().eq(str(control).lower())
    if not m.any():
        return {}
    return breach.loc[m].iloc[0].to_dict()


def simulate_row(row: pd.Series, breach: pd.DataFrame, risk_state: dict[str, Any]) -> dict[str, Any]:
    control = str(row.get("control", ""))
    policy_type = str(row.get("policy_type", "loss_budget"))
    action = str(row.get("threshold_action", ""))
    master_action = str(risk_state.get("master_risk_action", "NO_DATA") if risk_state else "NO_DATA")
    current_master_multiplier = safe_float(risk_state.get("master_exposure_multiplier", 1.0) if risk_state else 1.0, 1.0)

    current_value = abs(safe_float(row.get("current_value")))
    current_limit = abs(safe_float(row.get("current_policy_hard")))
    proxy_hard = abs(safe_float(row.get("proxy_history_hard")))
    proxy_warning = abs(safe_float(row.get("proxy_history_warning")))

    current_usage = current_value / current_limit if current_limit > 0 else np.nan
    current_status = status_from_usage(current_usage)

    blocked_loosen = action == "NO_LOOSEN_DURING_RISK_REDUCTION"
    simulated_limit = current_limit if blocked_loosen else proxy_hard
    simulated_warning = current_limit * 0.85 if blocked_loosen else proxy_warning
    simulated_usage = current_value / simulated_limit if simulated_limit > 0 else np.nan
    simulated_status = status_from_usage(simulated_usage)

    current_mult = exposure_multiplier_for_status(current_status, current_master_multiplier)
    simulated_mult = exposure_multiplier_for_status(simulated_status, current_master_multiplier)
    severity_delta = rank(simulated_status) - rank(current_status)

    if blocked_loosen:
        impact = "NO_POLICY_CHANGE_ALLOWED"
        decision = "Do not loosen this policy while the risk desk is already reducing exposure."
    elif np.isfinite(simulated_limit) and np.isfinite(current_limit) and simulated_limit < current_limit:
        if severity_delta > 0:
            impact = "WOULD_CREATE_NEW_BREACH"
            decision = "Tightening would create a stricter active breach; require explicit approval and implementation timing."
        else:
            impact = "TIGHTEN_NO_NEW_BREACH"
            decision = "Tightening looks conservative and does not create a new active breach at current values."
    elif np.isfinite(simulated_limit) and np.isfinite(current_limit) and simulated_limit > current_limit:
        if severity_delta < 0:
            impact = "WOULD_RELAX_CURRENT_BREACH"
            decision = "Loosening would reduce current breach pressure; review only after risk state normalizes."
        else:
            impact = "LOOSEN_NO_CURRENT_EFFECT"
            decision = "Loosening would not change current control status, but should still wait for approval."
    else:
        impact = "NO_LIMIT_DELTA"
        decision = "No meaningful threshold impact in this simulation."

    breach_row = lookup_breach_row(breach, control)
    return {
        "review_priority": row.get("review_priority", ""),
        "ticket_id": row.get("ticket_id", ""),
        "risk_area": row.get("risk_area", ""),
        "control": control,
        "plain_english_name": row.get("plain_english_name", control),
        "threshold_action": action,
        "impact_category": impact,
        "current_value": current_value,
        "current_value_display": fmt_value(current_value, policy_type),
        "current_limit": current_limit,
        "current_limit_display": fmt_value(current_limit, policy_type),
        "simulated_limit": simulated_limit,
        "simulated_limit_display": fmt_value(simulated_limit, policy_type),
        "simulated_warning": simulated_warning,
        "current_usage": current_usage,
        "simulated_usage": simulated_usage,
        "current_status": current_status,
        "simulated_status": simulated_status,
        "severity_delta": severity_delta,
        "current_master_exposure_multiplier": current_master_multiplier,
        "control_current_exposure_multiplier": current_mult,
        "control_simulated_exposure_multiplier": simulated_mult,
        "master_risk_action": master_action,
        "upstream_breach_status": breach_row.get("status", ""),
        "upstream_breach_bucket": breach_row.get("status_bucket", ""),
        "approval_required": True,
        "can_auto_apply": False,
        "decision_note": decision,
        "source_file": row.get("source_file", ""),
        "research_only": True,
    }


def build_simulation(recs: pd.DataFrame, breach: pd.DataFrame, risk_state: dict[str, Any]) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame()
    rows = [simulate_row(row, breach, risk_state) for _, row in recs.iterrows()]
    out = pd.DataFrame(rows)
    return out.sort_values(["review_priority", "severity_delta", "control"], ascending=[True, False, True]).reset_index(drop=True)


def build_decisions(sim: pd.DataFrame) -> pd.DataFrame:
    if sim.empty:
        return pd.DataFrame()
    cols = [
        "review_priority", "ticket_id", "control", "plain_english_name",
        "threshold_action", "impact_category", "current_limit_display",
        "simulated_limit_display", "current_status", "simulated_status",
        "decision_note", "approval_required", "can_auto_apply", "research_only",
    ]
    return sim[[c for c in cols if c in sim.columns]].copy()


def build_state(sim: pd.DataFrame) -> dict[str, Any]:
    if sim.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    impact = sim["impact_category"].astype(str).str.upper()
    if impact.eq("WOULD_CREATE_NEW_BREACH").any():
        overall = "APPROVAL_TIMING_REQUIRED"
    elif impact.eq("NO_POLICY_CHANGE_ALLOWED").any():
        overall = "DE_RISK_FIRST"
    elif impact.eq("TIGHTEN_NO_NEW_BREACH").any():
        overall = "SAFE_TIGHTEN_REVIEW"
    else:
        overall = "SIMULATION_READY"
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Simulate threshold-policy impact only. No limit is changed automatically.",
        "controls_simulated": int(len(sim)),
        "blocked_policy_changes": int(impact.eq("NO_POLICY_CHANGE_ALLOWED").sum()),
        "safe_tighten_candidates": int(impact.eq("TIGHTEN_NO_NEW_BREACH").sum()),
        "new_breach_if_approved": int(impact.eq("WOULD_CREATE_NEW_BREACH").sum()),
        "would_relax_current_breach": int(impact.eq("WOULD_RELAX_CURRENT_BREACH").sum()),
        "auto_apply_count": int(sim.get("can_auto_apply", pd.Series(dtype=bool)).astype(bool).sum()),
        "outputs": {
            "simulation": OUT_SIM.name,
            "decision_table": OUT_DECISIONS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    recs = read_csv_safe(IN_RECS)
    breach = read_csv_safe(IN_BREACH)
    risk_state = read_json_safe(IN_RISK_DESK)

    sim = build_simulation(recs, breach, risk_state)
    decisions = build_decisions(sim)
    state = build_state(sim)

    sim.to_csv(OUT_SIM, index=False)
    decisions.to_csv(OUT_DECISIONS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Controls simulated: {state.get('controls_simulated', 0)}",
        f"- Blocked policy changes: {state.get('blocked_policy_changes', 0)}",
        f"- Safe tighten candidates: {state.get('safe_tighten_candidates', 0)}",
        f"- New breach if approved: {state.get('new_breach_if_approved', 0)}",
        f"- Auto-apply count: {state.get('auto_apply_count', 0)}",
        "",
        "## Impact Simulation",
        "",
        df_to_markdown(sim, max_rows=50),
        "",
        "## Decision Table",
        "",
        df_to_markdown(decisions, max_rows=50),
        "",
        "## Product Truth",
        "",
        "This is a threshold impact simulation. It does not apply policy changes and it cannot override the active risk desk.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 138 - Threshold Impact Simulator", sections)

    print(f"wrote {OUT_SIM.name} rows={len(sim)}")
    print(f"wrote {OUT_DECISIONS.name} rows={len(decisions)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
