#!/usr/bin/env python3
"""
Canyon v9 - Step 137: Historical Threshold Review
=================================================

Research-only. No broker connection. No live orders.

Step136 turns missing-history controls into local proxy history. Step137 turns
that history into a draft-only threshold review layer. It does not edit any
upstream risk limit. It only says whether a human reviewer should consider
tightening, loosening, or keeping the current policy.

Outputs:
  risk_historical_threshold_recommendations.csv
  risk_historical_threshold_deltas.csv
  risk_historical_threshold_review_queue.csv
  risk_historical_threshold_state.json
  risk_historical_threshold_report.md
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

IN_HISTORY = ROOT / "risk_historical_evidence_summary.csv"
IN_BRIDGE = ROOT / "risk_historical_policy_bridge.csv"
IN_CALIBRATION = ROOT / "risk_threshold_calibration.csv"
IN_OPEN_TICKETS = ROOT / "risk_policy_open_tickets.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"

OUT_RECS = ROOT / "risk_historical_threshold_recommendations.csv"
OUT_DELTAS = ROOT / "risk_historical_threshold_deltas.csv"
OUT_QUEUE = ROOT / "risk_historical_threshold_review_queue.csv"
OUT_STATE = ROOT / "risk_historical_threshold_state.json"
OUT_REPORT = ROOT / "risk_historical_threshold_report.md"


CONTROL_POLICY_INTENT = {
    "Crisis-correlation volatility budget": {
        "policy_type": "risk_multiplier",
        "human_name": "Crisis correlation limit",
        "default_limit": 1.50,
        "minimum_hard": 1.25,
        "maximum_hard": 2.50,
        "warning_ratio": 0.90,
        "review_owner": "Risk committee",
    },
    "Drawdown budget": {
        "policy_type": "loss_budget",
        "human_name": "Portfolio drawdown limit",
        "default_limit": 0.10,
        "minimum_hard": 0.05,
        "maximum_hard": 0.25,
        "warning_ratio": 0.70,
        "review_owner": "Risk manager plus PM",
    },
    "Liquidity crisis liquidation budget": {
        "policy_type": "loss_budget",
        "human_name": "Liquidity crisis loss limit",
        "default_limit": 0.05,
        "minimum_hard": 0.005,
        "maximum_hard": 0.08,
        "warning_ratio": 0.75,
        "review_owner": "Risk manager plus execution review",
    },
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


def clip(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(min(max(value, lo), hi))


def lookup_row(df: pd.DataFrame, control: str) -> dict[str, Any]:
    if df.empty or "control" not in df.columns:
        return {}
    m = df["control"].astype(str).str.lower().eq(control.lower())
    if not m.any():
        return {}
    return df.loc[m].iloc[0].to_dict()


def action_for_delta(
    control: str,
    current_limit: float,
    proposed_hard: float,
    current_value: float,
    historical_percentile: float,
    master_action: str,
) -> tuple[str, str, int]:
    """
    Return action, rationale, and review priority.

    The guardrail is deliberate: if the risk desk is already reducing risk, a
    looser historical proxy limit can be reviewed later but must not be used as
    an excuse to loosen the book today.
    """
    if not np.isfinite(current_limit) or current_limit <= 0:
        return (
            "ADD_EXPLICIT_LIMIT_REVIEW",
            "No usable current limit exists. Add a draft warning/hard threshold before enforcement.",
            2,
        )

    delta_pct = (proposed_hard - current_limit) / current_limit if np.isfinite(proposed_hard) else np.nan
    desk_de_risking = str(master_action).upper() in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}

    if np.isfinite(delta_pct) and delta_pct <= -0.15:
        return (
            "TIGHTEN_LIMIT_REVIEW",
            "Historical proxy hard limit is materially below the current policy. Consider phased tightening after review.",
            2,
        )

    if np.isfinite(delta_pct) and delta_pct >= 0.20:
        if desk_de_risking or (np.isfinite(current_value) and np.isfinite(historical_percentile) and historical_percentile >= 90):
            return (
                "NO_LOOSEN_DURING_RISK_REDUCTION",
                "Proxy history could justify a looser hard limit, but the current book is already high-risk. De-risk first, then review.",
                1,
            )
        return (
            "LOOSEN_LIMIT_REVIEW",
            "Historical proxy hard limit is materially above the current policy. Review false-positive risk before loosening.",
            3,
        )

    if np.isfinite(historical_percentile) and historical_percentile >= 90:
        return (
            "KEEP_LIMIT_SIZE_DOWN",
            "Current value is in the high historical proxy percentile. Keep policy unchanged and reduce exposure.",
            2,
        )

    return (
        "KEEP_CURRENT_LIMIT",
        "Current policy is close enough to proxy history. Keep monitoring realized breach rates.",
        5,
    )


def build_recommendations(
    history: pd.DataFrame,
    calibration: pd.DataFrame,
    open_tickets: pd.DataFrame,
    bridge: pd.DataFrame,
    risk_state: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master_action = str(risk_state.get("master_risk_action", "NO_DATA") if risk_state else "NO_DATA")

    if history.empty:
        return pd.DataFrame()

    for _, hrow in history.iterrows():
        control = str(hrow.get("control", ""))
        intent = CONTROL_POLICY_INTENT.get(control, {
            "policy_type": "loss_budget",
            "human_name": control,
            "default_limit": np.nan,
            "minimum_hard": 0.0,
            "maximum_hard": np.inf,
            "warning_ratio": 0.75,
            "review_owner": "Risk manager",
        })
        policy_type = intent["policy_type"]
        cal = lookup_row(calibration, control)
        ticket = lookup_row(open_tickets, control)
        bridge_row = lookup_row(bridge, control)

        current_limit = safe_float(cal.get("current_limit"), safe_float(ticket.get("current_limit"), intent["default_limit"]))
        current_value = safe_float(cal.get("current_value"), safe_float(hrow.get("current_value")))
        proposed_raw = safe_float(hrow.get("proposed_hard"))
        proposed_warning_raw = safe_float(hrow.get("proposed_warning"))
        hard = clip(proposed_raw, safe_float(intent["minimum_hard"], 0.0), safe_float(intent["maximum_hard"], np.inf))
        warning = clip(
            proposed_warning_raw if np.isfinite(proposed_warning_raw) else hard * safe_float(intent["warning_ratio"], 0.75),
            0.0,
            hard if np.isfinite(hard) else np.inf,
        )
        historical_percentile = safe_float(hrow.get("historical_percentile"))
        action, rationale, priority = action_for_delta(
            control=control,
            current_limit=current_limit,
            proposed_hard=hard,
            current_value=current_value,
            historical_percentile=historical_percentile,
            master_action=master_action,
        )
        delta = hard - current_limit if np.isfinite(hard) and np.isfinite(current_limit) else np.nan
        delta_pct = delta / current_limit if np.isfinite(delta) and np.isfinite(current_limit) and current_limit > 0 else np.nan
        rows.append({
            "review_priority": priority,
            "risk_area": hrow.get("risk_area", ticket.get("risk_area", "")),
            "control": control,
            "plain_english_name": intent["human_name"],
            "policy_type": policy_type,
            "current_policy_hard": current_limit,
            "proxy_history_warning": warning,
            "proxy_history_hard": hard,
            "hard_limit_delta": delta,
            "hard_limit_delta_pct": delta_pct,
            "current_value": current_value,
            "current_value_display": fmt_value(current_value, policy_type),
            "current_policy_display": fmt_value(current_limit, policy_type),
            "proxy_hard_display": fmt_value(hard, policy_type),
            "historical_p80": safe_float(hrow.get("historical_p80")),
            "historical_p95": safe_float(hrow.get("historical_p95")),
            "worst_observed": safe_float(hrow.get("worst_observed")),
            "historical_percentile": historical_percentile,
            "sample_n": int(safe_float(hrow.get("sample_n"), 0)),
            "lookback_start": hrow.get("lookback_start", ""),
            "lookback_end": hrow.get("lookback_end", ""),
            "historical_status": hrow.get("historical_status", ""),
            "threshold_action": action,
            "review_owner": intent["review_owner"],
            "approval_required": True,
            "can_auto_apply": False,
            "rationale": rationale,
            "risk_desk_master_action": master_action,
            "ticket_id": ticket.get("ticket_id", bridge_row.get("ticket_id", "")),
            "source_file": hrow.get("source_file", ""),
            "research_only": True,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["review_priority", "threshold_action", "control"]).reset_index(drop=True)


def build_deltas(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame()
    cols = [
        "control", "plain_english_name", "policy_type", "current_policy_hard",
        "proxy_history_warning", "proxy_history_hard", "hard_limit_delta",
        "hard_limit_delta_pct", "threshold_action", "approval_required",
        "can_auto_apply", "rationale",
    ]
    return recs[[c for c in cols if c in recs.columns]].copy()


def build_queue(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame()
    actionable = recs[recs["threshold_action"].astype(str).ne("KEEP_CURRENT_LIMIT")].copy()
    if actionable.empty:
        return pd.DataFrame(columns=[
            "review_priority", "ticket_id", "control", "threshold_action",
            "review_owner", "approval_required", "can_auto_apply",
            "required_next_action", "rationale", "research_only",
        ])
    actionable["required_next_action"] = actionable["threshold_action"].map({
        "TIGHTEN_LIMIT_REVIEW": "Review and, if approved, phase in the tighter historical-proxy hard limit.",
        "LOOSEN_LIMIT_REVIEW": "Review false-positive evidence before loosening this limit.",
        "NO_LOOSEN_DURING_RISK_REDUCTION": "Do not loosen now. De-risk first, then rerun Step136/137.",
        "KEEP_LIMIT_SIZE_DOWN": "Keep limit unchanged and reduce exposure until current value is below watch level.",
        "ADD_EXPLICIT_LIMIT_REVIEW": "Draft a warning and hard limit, then approve before enforcement.",
    }).fillna("Manual risk review required.")
    cols = [
        "review_priority", "ticket_id", "risk_area", "control",
        "plain_english_name", "threshold_action", "current_policy_display",
        "proxy_hard_display", "sample_n", "review_owner",
        "approval_required", "can_auto_apply", "required_next_action",
        "rationale", "source_file", "research_only",
    ]
    return actionable[[c for c in cols if c in actionable.columns]].reset_index(drop=True)


def build_state(recs: pd.DataFrame, queue: pd.DataFrame) -> dict[str, Any]:
    if recs.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    actions = recs["threshold_action"].astype(str).str.upper()
    if actions.eq("NO_LOOSEN_DURING_RISK_REDUCTION").any():
        overall = "DE_RISK_BEFORE_POLICY_CHANGE"
    elif actions.isin(["TIGHTEN_LIMIT_REVIEW", "LOOSEN_LIMIT_REVIEW", "ADD_EXPLICIT_LIMIT_REVIEW"]).any():
        overall = "THRESHOLD_REVIEW_REQUIRED"
    else:
        overall = "THRESHOLDS_OK"
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Draft-only historical threshold review. No risk limit is changed automatically.",
        "controls_checked": int(len(recs)),
        "review_queue_items": int(len(queue)),
        "tighten_candidates": int(actions.eq("TIGHTEN_LIMIT_REVIEW").sum()),
        "loosen_candidates": int(actions.eq("LOOSEN_LIMIT_REVIEW").sum()),
        "blocked_loosen_candidates": int(actions.eq("NO_LOOSEN_DURING_RISK_REDUCTION").sum()),
        "keep_current_count": int(actions.eq("KEEP_CURRENT_LIMIT").sum()),
        "auto_apply_count": int(recs.get("can_auto_apply", pd.Series(dtype=bool)).astype(bool).sum()),
        "outputs": {
            "recommendations": OUT_RECS.name,
            "deltas": OUT_DELTAS.name,
            "review_queue": OUT_QUEUE.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    history = read_csv_safe(IN_HISTORY)
    bridge = read_csv_safe(IN_BRIDGE)
    calibration = read_csv_safe(IN_CALIBRATION)
    open_tickets = read_csv_safe(IN_OPEN_TICKETS)
    risk_state = read_json_safe(IN_RISK_DESK)

    recs = build_recommendations(history, calibration, open_tickets, bridge, risk_state)
    deltas = build_deltas(recs)
    queue = build_queue(recs)
    state = build_state(recs, queue)

    recs.to_csv(OUT_RECS, index=False)
    deltas.to_csv(OUT_DELTAS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Controls checked: {state.get('controls_checked', 0)}",
        f"- Review queue items: {state.get('review_queue_items', 0)}",
        f"- Tighten candidates: {state.get('tighten_candidates', 0)}",
        f"- Loosen candidates: {state.get('loosen_candidates', 0)}",
        f"- Blocked loosen candidates: {state.get('blocked_loosen_candidates', 0)}",
        f"- Auto-apply count: {state.get('auto_apply_count', 0)}",
        "",
        "## Historical Threshold Recommendations",
        "",
        df_to_markdown(recs, max_rows=50),
        "",
        "## Threshold Deltas",
        "",
        df_to_markdown(deltas, max_rows=50),
        "",
        "## Review Queue",
        "",
        df_to_markdown(queue, max_rows=50),
        "",
        "## Product Truth",
        "",
        "These are draft-only policy suggestions. They are not executed, not broker-connected, and cannot override the active risk desk.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 137 - Historical Threshold Review", sections)

    print(f"wrote {OUT_RECS.name} rows={len(recs)}")
    print(f"wrote {OUT_QUEUE.name} rows={len(queue)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
