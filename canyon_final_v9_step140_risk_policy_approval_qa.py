#!/usr/bin/env python3
"""
Canyon v9 - Step 140: Risk Policy Approval QA
=============================================

Research-only. No broker connection. No live orders.

Step139 builds a change-control ledger. Step140 checks whether a policy change
is actually ready for the next governance stage:

  - blocked changes stay blocked
  - ready-for-approval changes wait for human approval
  - approved changes still only move to dry-run review, never live apply

This step does not apply thresholds. It creates approval packets and a manual
approval template that can be edited by a reviewer.

Outputs:
  risk_policy_approval_checklist.csv
  risk_policy_approval_packet.csv
  risk_policy_manual_approval_template.csv
  risk_policy_approval_state.json
  risk_policy_approval_report.md
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

IN_CHANGE = ROOT / "risk_policy_change_control.csv"
IN_READINESS = ROOT / "risk_policy_change_readiness.csv"
IN_PLAN = ROOT / "risk_policy_implementation_plan.csv"
IN_IMPACT = ROOT / "risk_threshold_impact_simulation.csv"

OUT_CHECKLIST = ROOT / "risk_policy_approval_checklist.csv"
OUT_PACKET = ROOT / "risk_policy_approval_packet.csv"
OUT_TEMPLATE = ROOT / "risk_policy_manual_approval_template.csv"
OUT_STATE = ROOT / "risk_policy_approval_state.json"
OUT_REPORT = ROOT / "risk_policy_approval_report.md"


TEMPLATE_COLUMNS = [
    "change_id",
    "ticket_id",
    "control",
    "approval_status",
    "approved_by",
    "approval_date",
    "approval_notes",
    "implementation_owner",
    "implementation_notes",
    "rollback_notes",
    "reviewer_action",
    "research_only",
]


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def boolish(value: Any) -> bool:
    return text(value).upper() in {"TRUE", "1", "YES", "Y", "APPROVED"}


def file_has_rows(path: Path) -> bool:
    df = read_csv_safe(path)
    return not df.empty


def existing_template() -> pd.DataFrame:
    existing = read_csv_safe(OUT_TEMPLATE)
    if existing.empty:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    for col in TEMPLATE_COLUMNS:
        if col not in existing.columns:
            existing[col] = ""
    return existing[TEMPLATE_COLUMNS].drop_duplicates("change_id", keep="last")


def template_from_change(change: pd.DataFrame) -> pd.DataFrame:
    if change.empty:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    previous = existing_template()
    prev_by_id = {str(r["change_id"]): r for _, r in previous.iterrows()} if not previous.empty else {}
    rows = []
    for _, row in change.iterrows():
        cid = text(row.get("change_id"))
        prev = prev_by_id.get(cid, {})
        approval_status = text(prev.get("approval_status")) or text(row.get("approval_status")) or "PENDING_APPROVAL"
        rows.append({
            "change_id": cid,
            "ticket_id": row.get("ticket_id", ""),
            "control": row.get("control", ""),
            "approval_status": approval_status,
            "approved_by": text(prev.get("approved_by")) or text(row.get("approved_by")),
            "approval_date": text(prev.get("approval_date")) or text(row.get("approval_date")),
            "approval_notes": text(prev.get("approval_notes")) or text(row.get("approval_notes")),
            "implementation_owner": text(prev.get("implementation_owner")) or text(row.get("implementation_owner")),
            "implementation_notes": text(prev.get("implementation_notes")) or text(row.get("implementation_notes")),
            "rollback_notes": text(prev.get("rollback_notes")) or text(row.get("rollback_notes")),
            "reviewer_action": text(prev.get("reviewer_action")) or reviewer_action(row),
            "research_only": True,
        })
    return pd.DataFrame(rows)[TEMPLATE_COLUMNS]


def reviewer_action(row: pd.Series) -> str:
    status = text(row.get("change_status")).upper()
    if status.startswith("BLOCKED"):
        return "No approval allowed. De-risk first, then rerun Step136-140."
    if status == "READY_FOR_APPROVAL":
        return "Fill approved_by, approval_date, and approval_notes only after human review."
    return "Monitor. No approval request needed."


def manual_fields_for(change_id: str, template: pd.DataFrame, change_row: pd.Series) -> dict[str, str]:
    if not template.empty and "change_id" in template.columns:
        m = template["change_id"].astype(str).eq(change_id)
        if m.any():
            row = template.loc[m].iloc[0]
            return {col: text(row.get(col)) for col in TEMPLATE_COLUMNS}
    return {
        "approval_status": text(change_row.get("approval_status")),
        "approved_by": text(change_row.get("approved_by")),
        "approval_date": text(change_row.get("approval_date")),
        "approval_notes": text(change_row.get("approval_notes")),
        "implementation_owner": text(change_row.get("implementation_owner")),
        "implementation_notes": text(change_row.get("implementation_notes")),
        "rollback_notes": text(change_row.get("rollback_notes")),
    }


def qa_stage(row: pd.Series, manual: dict[str, str], checks: dict[str, bool]) -> tuple[str, str]:
    change_status = text(row.get("change_status")).upper()
    approval_status = text(manual.get("approval_status")).upper()

    if change_status.startswith("BLOCKED"):
        return "BLOCKED", "Change is blocked by risk state and cannot request approval."
    if not boolish(row.get("can_request_approval")):
        return "NOT_APPROVAL_ELIGIBLE", "Change-control row is not eligible for approval request."
    if not checks["has_required_evidence"]:
        return "MISSING_EVIDENCE", "Required source files or evidence links are missing."
    if approval_status != "APPROVED":
        return "WAITING_FOR_MANUAL_APPROVAL", "Approval fields are not complete."
    if not (manual.get("approved_by") and manual.get("approval_date") and manual.get("approval_notes")):
        return "APPROVAL_FIELDS_INCOMPLETE", "Approval status is approved but reviewer/date/notes are incomplete."
    if not checks["has_rollback_trigger"]:
        return "MISSING_ROLLBACK_TRIGGER", "Rollback trigger is missing."
    return "READY_FOR_DRY_RUN", "Manual approval is complete; next stage is dry-run only, not live apply."


def build_checklist(change: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    if change.empty:
        return pd.DataFrame()
    impact_ready = file_has_rows(IN_IMPACT)
    plan_ready = file_has_rows(IN_PLAN)
    rows = []
    for _, row in change.iterrows():
        cid = text(row.get("change_id"))
        manual = manual_fields_for(cid, template, row)
        checks = {
            "is_active": boolish(row.get("currently_active")),
            "has_impact_simulation": impact_ready,
            "has_implementation_plan": plan_ready,
            "has_required_tests": bool(text(row.get("required_tests"))),
            "has_rollback_trigger": bool(text(row.get("rollback_trigger"))),
            "has_required_evidence": bool(text(row.get("source_file"))) and impact_ready and plan_ready,
            "manual_approved": text(manual.get("approval_status")).upper() == "APPROVED",
            "manual_reviewer_present": bool(manual.get("approved_by")),
            "manual_date_present": bool(manual.get("approval_date")),
            "manual_notes_present": bool(manual.get("approval_notes")),
        }
        stage, note = qa_stage(row, manual, checks)
        rows.append({
            "change_id": cid,
            "ticket_id": row.get("ticket_id", ""),
            "review_priority": row.get("review_priority", ""),
            "control": row.get("control", ""),
            "change_status": row.get("change_status", ""),
            "threshold_action": row.get("threshold_action", ""),
            "impact_category": row.get("impact_category", ""),
            "approval_status": manual.get("approval_status", ""),
            "approval_stage": stage,
            "approval_note": note,
            "is_active": checks["is_active"],
            "has_impact_simulation": checks["has_impact_simulation"],
            "has_implementation_plan": checks["has_implementation_plan"],
            "has_required_tests": checks["has_required_tests"],
            "has_rollback_trigger": checks["has_rollback_trigger"],
            "has_required_evidence": checks["has_required_evidence"],
            "manual_approved": checks["manual_approved"],
            "manual_reviewer_present": checks["manual_reviewer_present"],
            "manual_date_present": checks["manual_date_present"],
            "manual_notes_present": checks["manual_notes_present"],
            "can_request_approval": boolish(row.get("can_request_approval")),
            "can_stage_dry_run_after_approval": stage == "READY_FOR_DRY_RUN",
            "can_apply_live": False,
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["review_priority", "approval_stage", "control"]).reset_index(drop=True)


def build_packet(change: pd.DataFrame, checklist: pd.DataFrame) -> pd.DataFrame:
    if change.empty or checklist.empty:
        return pd.DataFrame()
    ck_by_id = {str(r["change_id"]): r for _, r in checklist.iterrows()}
    rows = []
    for _, row in change.iterrows():
        cid = text(row.get("change_id"))
        ck = ck_by_id.get(cid, {})
        rows.append({
            "change_id": cid,
            "ticket_id": row.get("ticket_id", ""),
            "control": row.get("control", ""),
            "plain_english_name": row.get("plain_english_name", ""),
            "approval_stage": ck.get("approval_stage", ""),
            "change_status": row.get("change_status", ""),
            "approval_status": ck.get("approval_status", ""),
            "current_limit_display": row.get("current_limit_display", ""),
            "simulated_limit_display": row.get("simulated_limit_display", ""),
            "current_status": row.get("current_status", ""),
            "simulated_status": row.get("simulated_status", ""),
            "required_approval": row.get("required_approval", ""),
            "required_tests": row.get("required_tests", ""),
            "rollback_trigger": row.get("rollback_trigger", ""),
            "decision_note": row.get("decision_note", ""),
            "approval_note": ck.get("approval_note", ""),
            "can_apply_live": False,
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["approval_stage", "control"]).reset_index(drop=True)


def build_state(checklist: pd.DataFrame) -> dict[str, Any]:
    if checklist.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    stages = checklist["approval_stage"].astype(str).str.upper()
    blocked = int(stages.eq("BLOCKED").sum())
    waiting = int(stages.eq("WAITING_FOR_MANUAL_APPROVAL").sum())
    ready_dry = int(stages.eq("READY_FOR_DRY_RUN").sum())
    incomplete = int(stages.str.contains("INCOMPLETE|MISSING", regex=True).sum())
    if incomplete:
        overall = "APPROVAL_QA_GAPS"
    elif blocked and waiting:
        overall = "BLOCKED_AND_WAITING_APPROVAL"
    elif waiting:
        overall = "WAITING_FOR_MANUAL_APPROVAL"
    elif ready_dry:
        overall = "DRY_RUN_READY"
    elif blocked:
        overall = "BLOCKED_ONLY"
    else:
        overall = "MONITOR_ONLY"
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Approval QA only. It can move an approved item to dry-run readiness, never live apply.",
        "changes_checked": int(len(checklist)),
        "blocked_count": blocked,
        "waiting_manual_approval_count": waiting,
        "ready_for_dry_run_count": ready_dry,
        "qa_gap_count": incomplete,
        "can_apply_live_count": int(checklist["can_apply_live"].astype(bool).sum()),
        "outputs": {
            "checklist": OUT_CHECKLIST.name,
            "packet": OUT_PACKET.name,
            "manual_template": OUT_TEMPLATE.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    change = read_csv_safe(IN_CHANGE)
    template = template_from_change(change)
    checklist = build_checklist(change, template)
    packet = build_packet(change, checklist)
    state = build_state(checklist)

    checklist.to_csv(OUT_CHECKLIST, index=False)
    packet.to_csv(OUT_PACKET, index=False)
    template.to_csv(OUT_TEMPLATE, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Changes checked: {state.get('changes_checked', 0)}",
        f"- Blocked: {state.get('blocked_count', 0)}",
        f"- Waiting manual approval: {state.get('waiting_manual_approval_count', 0)}",
        f"- Ready for dry run: {state.get('ready_for_dry_run_count', 0)}",
        f"- QA gaps: {state.get('qa_gap_count', 0)}",
        f"- Can apply live: {state.get('can_apply_live_count', 0)}",
        "",
        "## Approval Checklist",
        "",
        df_to_markdown(checklist, max_rows=50),
        "",
        "## Approval Packet",
        "",
        df_to_markdown(packet, max_rows=50),
        "",
        "## Manual Approval Template",
        "",
        df_to_markdown(template, max_rows=50),
        "",
        "## Product Truth",
        "",
        "This is an approval QA gate. It never applies thresholds and never enables live orders.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 140 - Risk Policy Approval QA", sections)

    print(f"wrote {OUT_CHECKLIST.name} rows={len(checklist)}")
    print(f"wrote {OUT_PACKET.name} rows={len(packet)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
