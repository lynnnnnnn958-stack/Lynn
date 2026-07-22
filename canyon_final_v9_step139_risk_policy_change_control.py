#!/usr/bin/env python3
"""
Canyon v9 - Step 139: Risk Policy Change Control
================================================

Research-only. No broker connection. No live orders.

Step138 simulates threshold-change impact. Step139 turns those simulations into
a controlled change ledger:

  - what is blocked
  - what can move to approval review
  - what tests are required before implementation
  - what rollback trigger would stop the change

This step preserves manual approval fields across reruns. It still does not
apply any threshold change.

Outputs:
  risk_policy_change_control.csv
  risk_policy_change_readiness.csv
  risk_policy_implementation_plan.csv
  risk_policy_change_state.json
  risk_policy_change_report.md
"""

from __future__ import annotations

import hashlib
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

IN_IMPACT = ROOT / "risk_threshold_impact_simulation.csv"
IN_DECISIONS = ROOT / "risk_threshold_impact_decision_table.csv"
IN_OPEN_TICKETS = ROOT / "risk_policy_open_tickets.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"

OUT_CONTROL = ROOT / "risk_policy_change_control.csv"
OUT_READINESS = ROOT / "risk_policy_change_readiness.csv"
OUT_PLAN = ROOT / "risk_policy_implementation_plan.csv"
OUT_STATE = ROOT / "risk_policy_change_state.json"
OUT_REPORT = ROOT / "risk_policy_change_report.md"


MANUAL_FIELDS = [
    "approval_status",
    "approved_by",
    "approval_date",
    "approval_notes",
    "implementation_status",
    "implementation_owner",
    "implementation_date",
    "implementation_notes",
    "rollback_status",
    "rollback_notes",
]

CONTROL_COLUMNS = [
    "change_id",
    "ticket_id",
    "created_at",
    "last_seen_at",
    "currently_active",
    "review_priority",
    "risk_area",
    "control",
    "plain_english_name",
    "threshold_action",
    "impact_category",
    "change_status",
    "approval_status",
    "approved_by",
    "approval_date",
    "approval_notes",
    "implementation_status",
    "implementation_owner",
    "implementation_date",
    "implementation_notes",
    "rollback_status",
    "rollback_notes",
    "current_limit_display",
    "simulated_limit_display",
    "current_status",
    "simulated_status",
    "master_risk_action",
    "can_request_approval",
    "can_stage_dry_run",
    "can_apply_live",
    "required_approval",
    "required_tests",
    "rollback_trigger",
    "required_next_action",
    "decision_note",
    "source_file",
    "research_only",
]


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def slug(value: str) -> str:
    out = []
    for ch in value.upper():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    text = "".join(out).strip("_")
    return text[:42] if text else "UNKNOWN"


def change_id(row: pd.Series) -> str:
    raw = "|".join([
        clean_text(row.get("ticket_id")),
        clean_text(row.get("control")),
        clean_text(row.get("threshold_action")),
        clean_text(row.get("simulated_limit_display")),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8].upper()
    return f"RPC-{slug(clean_text(row.get('control')))}-{digest}"


def change_status(row: pd.Series) -> tuple[str, bool, bool, str]:
    impact = clean_text(row.get("impact_category")).upper()
    action = clean_text(row.get("threshold_action")).upper()
    master = clean_text(row.get("master_risk_action")).upper()

    if impact == "NO_POLICY_CHANGE_ALLOWED" or action == "NO_LOOSEN_DURING_RISK_REDUCTION":
        return (
            "BLOCKED_DE_RISK_FIRST",
            False,
            False,
            "Do not request approval while the master risk desk is reducing exposure.",
        )
    if impact == "WOULD_CREATE_NEW_BREACH":
        return (
            "BLOCKED_TIMING_REQUIRED",
            False,
            False,
            "This change would create a stricter active breach; review implementation timing first.",
        )
    if impact == "WOULD_RELAX_CURRENT_BREACH":
        return (
            "BLOCKED_RISK_RELAXATION",
            False,
            False,
            "This would reduce breach pressure; wait until the book is de-risked.",
        )
    if impact == "TIGHTEN_NO_NEW_BREACH" and action == "TIGHTEN_LIMIT_REVIEW":
        return (
            "READY_FOR_APPROVAL",
            True,
            True,
            "Ready for human approval and dry-run staging. It still cannot auto-apply.",
        )
    if master in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        return (
            "MONITOR_UNTIL_RISK_NORMALIZES",
            False,
            False,
            "Risk desk is still defensive; keep monitoring before policy implementation.",
        )
    return (
        "MONITOR",
        False,
        False,
        "No actionable policy change from this simulation.",
    )


def approval_route(row: pd.Series) -> str:
    control = clean_text(row.get("control")).lower()
    action = clean_text(row.get("threshold_action")).upper()
    if "liquidity" in control:
        return "Risk manager + execution review"
    if "correlation" in control or "macro" in control:
        return "Risk committee"
    if "drawdown" in control:
        return "Risk manager + PM"
    if action == "TIGHTEN_LIMIT_REVIEW":
        return "Risk manager"
    return "Risk manager review"


def required_tests(row: pd.Series) -> str:
    impact = clean_text(row.get("impact_category")).upper()
    if impact == "TIGHTEN_NO_NEW_BREACH":
        return (
            "Run steps 111-118/131-139 twice; confirm no new active breach; "
            "confirm Evidence Pack links source rows; browser-check Risk Desk tabs."
        )
    return (
        "Re-run after de-risking; confirm master risk action is no longer SIZE_DOWN/REDUCE_ONLY; "
        "then rerun Step136-139 before review."
    )


def rollback_trigger(row: pd.Series) -> str:
    impact = clean_text(row.get("impact_category")).upper()
    if impact == "TIGHTEN_NO_NEW_BREACH":
        return (
            "Rollback or pause if new breach count rises above 0, liquidity status worsens, "
            "or manual reviewer rejects the threshold."
        )
    return "No rollout allowed; blocked items have no rollback path because they cannot be staged."


def normalize_existing(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=CONTROL_COLUMNS)
    out = existing.copy()
    for col in CONTROL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    if "change_id" in out.columns:
        out["change_id"] = out["change_id"].astype(str)
        out = out[out["change_id"].str.strip() != ""].copy()
        out = out.drop_duplicates("change_id", keep="last")
    return out[CONTROL_COLUMNS].reset_index(drop=True)


def default_manual_fields(status: str) -> dict[str, Any]:
    if status == "READY_FOR_APPROVAL":
        approval_status = "PENDING_APPROVAL"
        implementation_status = "NOT_STAGED"
    elif status.startswith("BLOCKED"):
        approval_status = "BLOCKED"
        implementation_status = "BLOCKED"
    else:
        approval_status = "MONITOR"
        implementation_status = "NOT_STAGED"
    return {
        "approval_status": approval_status,
        "approved_by": "",
        "approval_date": "",
        "approval_notes": "",
        "implementation_status": implementation_status,
        "implementation_owner": "",
        "implementation_date": "",
        "implementation_notes": "",
        "rollback_status": "NOT_NEEDED",
        "rollback_notes": "",
    }


def build_current_controls(impact: pd.DataFrame, risk_state: dict[str, Any]) -> pd.DataFrame:
    if impact.empty:
        return pd.DataFrame(columns=CONTROL_COLUMNS)
    rows = []
    now = now_str()
    for _, row in impact.iterrows():
        status, can_request, can_stage, next_action = change_status(row)
        manual = default_manual_fields(status)
        rows.append({
            "change_id": change_id(row),
            "ticket_id": row.get("ticket_id", ""),
            "created_at": now,
            "last_seen_at": now,
            "currently_active": True,
            "review_priority": int(safe_float(row.get("review_priority"), 9)),
            "risk_area": row.get("risk_area", ""),
            "control": row.get("control", ""),
            "plain_english_name": row.get("plain_english_name", ""),
            "threshold_action": row.get("threshold_action", ""),
            "impact_category": row.get("impact_category", ""),
            "change_status": status,
            **manual,
            "current_limit_display": row.get("current_limit_display", ""),
            "simulated_limit_display": row.get("simulated_limit_display", ""),
            "current_status": row.get("current_status", ""),
            "simulated_status": row.get("simulated_status", ""),
            "master_risk_action": risk_state.get("master_risk_action", row.get("master_risk_action", "")) if risk_state else row.get("master_risk_action", ""),
            "can_request_approval": can_request,
            "can_stage_dry_run": can_stage,
            "can_apply_live": False,
            "required_approval": approval_route(row),
            "required_tests": required_tests(row),
            "rollback_trigger": rollback_trigger(row),
            "required_next_action": next_action,
            "decision_note": row.get("decision_note", ""),
            "source_file": row.get("source_file", ""),
            "research_only": True,
        })
    out = pd.DataFrame(rows)
    return out[CONTROL_COLUMNS].sort_values(["review_priority", "change_status", "control"]).reset_index(drop=True)


def merge_controls(current: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_existing(existing)
    existing_by_id = {str(r["change_id"]): r for _, r in existing.iterrows()}
    current_ids = set(current["change_id"].astype(str)) if not current.empty else set()
    rows: list[dict[str, Any]] = []

    for _, row in current.iterrows():
        cid = str(row["change_id"])
        out = row.to_dict()
        if cid in existing_by_id:
            prev = existing_by_id[cid]
            out["created_at"] = prev.get("created_at") or out["created_at"]
            for field in MANUAL_FIELDS:
                prev_value = prev.get(field, "")
                if clean_text(prev_value):
                    out[field] = prev_value
        rows.append(out)

    for _, row in existing.iterrows():
        cid = str(row["change_id"])
        if cid in current_ids:
            continue
        out = row.to_dict()
        out["currently_active"] = False
        if clean_text(out.get("approval_status")).startswith("PENDING"):
            out["approval_status"] = "STALE_PENDING_REVIEW"
        if clean_text(out.get("implementation_status")) in {"NOT_STAGED", "STAGED"}:
            out["implementation_status"] = "STALE_REVIEW"
        out["required_next_action"] = "This change is no longer in the latest impact simulation. Review before closing."
        rows.append(out)

    result = pd.DataFrame(rows)
    for col in CONTROL_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    result["can_apply_live"] = False
    return result[CONTROL_COLUMNS].sort_values(["currently_active", "review_priority", "change_status", "control"], ascending=[False, True, True, True]).reset_index(drop=True)


def build_readiness(control: pd.DataFrame) -> pd.DataFrame:
    if control.empty:
        return pd.DataFrame()
    rows = []
    for status, sub in control.groupby("change_status", dropna=False):
        rows.append({
            "change_status": status,
            "count": int(len(sub)),
            "can_request_approval_count": int(sub["can_request_approval"].astype(bool).sum()) if "can_request_approval" in sub.columns else 0,
            "can_stage_dry_run_count": int(sub["can_stage_dry_run"].astype(bool).sum()) if "can_stage_dry_run" in sub.columns else 0,
            "can_apply_live_count": int(sub["can_apply_live"].astype(bool).sum()) if "can_apply_live" in sub.columns else 0,
            "controls": "; ".join(sub["control"].dropna().astype(str).head(10).tolist()),
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["can_request_approval_count", "count"], ascending=[False, False]).reset_index(drop=True)


def build_plan(control: pd.DataFrame) -> pd.DataFrame:
    if control.empty:
        return pd.DataFrame()
    rows = []
    for _, row in control.iterrows():
        status = clean_text(row.get("change_status"))
        if status == "READY_FOR_APPROVAL":
            phase = "PHASE_0_APPROVAL_THEN_DRY_RUN"
        elif status.startswith("BLOCKED"):
            phase = "BLOCKED"
        else:
            phase = "MONITOR"
        rows.append({
            "change_id": row.get("change_id", ""),
            "ticket_id": row.get("ticket_id", ""),
            "control": row.get("control", ""),
            "phase": phase,
            "current_limit_display": row.get("current_limit_display", ""),
            "simulated_limit_display": row.get("simulated_limit_display", ""),
            "owner": row.get("required_approval", ""),
            "pre_implementation_tests": row.get("required_tests", ""),
            "rollback_trigger": row.get("rollback_trigger", ""),
            "next_action": row.get("required_next_action", ""),
            "can_apply_live": False,
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["phase", "control"]).reset_index(drop=True)


def build_state(control: pd.DataFrame, readiness: pd.DataFrame) -> dict[str, Any]:
    if control.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    statuses = control["change_status"].astype(str).str.upper()
    ready = int(statuses.eq("READY_FOR_APPROVAL").sum())
    blocked = int(statuses.str.startswith("BLOCKED").sum())
    stale = int((~control["currently_active"].astype(bool)).sum())
    if blocked:
        overall = "BLOCKED_CHANGES_PRESENT"
    elif ready:
        overall = "APPROVAL_READY"
    else:
        overall = "MONITOR_ONLY"
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Change-control ledger only. No threshold is applied automatically.",
        "changes_total": int(len(control)),
        "active_changes": int(control["currently_active"].astype(bool).sum()),
        "ready_for_approval": ready,
        "blocked_changes": blocked,
        "stale_changes": stale,
        "can_apply_live_count": int(control["can_apply_live"].astype(bool).sum()),
        "readiness_rows": int(len(readiness)),
        "outputs": {
            "change_control": OUT_CONTROL.name,
            "readiness": OUT_READINESS.name,
            "implementation_plan": OUT_PLAN.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    impact = read_csv_safe(IN_IMPACT)
    existing = read_csv_safe(OUT_CONTROL)
    risk_state = read_json_safe(IN_RISK_DESK)

    current = build_current_controls(impact, risk_state)
    control = merge_controls(current, existing)
    readiness = build_readiness(control)
    plan = build_plan(control)
    state = build_state(control, readiness)

    control.to_csv(OUT_CONTROL, index=False)
    readiness.to_csv(OUT_READINESS, index=False)
    plan.to_csv(OUT_PLAN, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Total changes: {state.get('changes_total', 0)}",
        f"- Active changes: {state.get('active_changes', 0)}",
        f"- Ready for approval: {state.get('ready_for_approval', 0)}",
        f"- Blocked changes: {state.get('blocked_changes', 0)}",
        f"- Can apply live: {state.get('can_apply_live_count', 0)}",
        "",
        "## Change Control Ledger",
        "",
        df_to_markdown(control, max_rows=50),
        "",
        "## Readiness",
        "",
        df_to_markdown(readiness, max_rows=20),
        "",
        "## Implementation Plan",
        "",
        df_to_markdown(plan, max_rows=50),
        "",
        "## Product Truth",
        "",
        "This is a local change-control ledger. It preserves manual approval fields, but it does not apply thresholds and it cannot send orders.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 139 - Risk Policy Change Control", sections)

    print(f"wrote {OUT_CONTROL.name} rows={len(control)}")
    print(f"wrote {OUT_PLAN.name} rows={len(plan)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
