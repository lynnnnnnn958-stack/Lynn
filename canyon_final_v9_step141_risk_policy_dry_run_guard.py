#!/usr/bin/env python3
"""
Canyon v9 - Step 141: Risk Policy Dry-Run Guard
================================================

Research-only. No broker connection. No live orders.

Step140 decides whether a policy change is blocked, waiting for human approval,
or ready for dry-run review. Step141 turns that QA result into a shadow rollout
guard:

  - blocked changes remain blocked
  - unapproved changes remain waiting
  - approved changes can only enter a dry-run plan
  - live application is always false

This step does not apply thresholds. It creates the checklist a reviewer would
use before allowing a policy change to run in shadow mode.

Outputs:
  risk_policy_dry_run_readiness.csv
  risk_policy_dry_run_plan.csv
  risk_policy_dry_run_monitor.csv
  risk_policy_dry_run_state.json
  risk_policy_dry_run_report.md
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

IN_APPROVAL = ROOT / "risk_policy_approval_checklist.csv"
IN_PACKET = ROOT / "risk_policy_approval_packet.csv"
IN_TEMPLATE = ROOT / "risk_policy_manual_approval_template.csv"
IN_CHANGE = ROOT / "risk_policy_change_control.csv"
IN_BREACH = ROOT / "risk_desk_breach_table.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"
IN_EVIDENCE = ROOT / "risk_policy_evidence_pack.csv"

OUT_READINESS = ROOT / "risk_policy_dry_run_readiness.csv"
OUT_PLAN = ROOT / "risk_policy_dry_run_plan.csv"
OUT_MONITOR = ROOT / "risk_policy_dry_run_monitor.csv"
OUT_STATE = ROOT / "risk_policy_dry_run_state.json"
OUT_REPORT = ROOT / "risk_policy_dry_run_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def boolish(value: Any) -> bool:
    return text(value).upper() in {"TRUE", "1", "YES", "Y", "APPROVED"}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct(value: Any) -> str:
    x = safe_float(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x * 100:.2f}%"


def rows_by_id(df: pd.DataFrame, key: str) -> dict[str, pd.Series]:
    if df.empty or key not in df.columns:
        return {}
    return {text(r.get(key)): r for _, r in df.drop_duplicates(key, keep="last").iterrows()}


def breach_counts(breach: pd.DataFrame) -> dict[str, int]:
    if breach.empty or "status" not in breach.columns:
        return {"hard_or_reduce": 0, "size_down": 0, "review": 0}
    status = breach["status"].astype(str).str.upper()
    return {
        "hard_or_reduce": int(status.str.contains("REDUCE_ONLY|BLOCK|HARD", regex=True).sum()),
        "size_down": int(status.eq("SIZE_DOWN").sum()),
        "review": int(status.eq("REVIEW").sum()),
    }


def evidence_grade_for(ticket_id: str, evidence: pd.DataFrame) -> str:
    if evidence.empty or "ticket_id" not in evidence.columns:
        return "NO_EVIDENCE_PACK"
    m = evidence["ticket_id"].astype(str).eq(ticket_id)
    if not m.any():
        return "NO_MATCHING_EVIDENCE"
    return text(evidence.loc[m, "evidence_grade"].iloc[0]) if "evidence_grade" in evidence.columns else "EVIDENCE_PRESENT"


def dry_run_stage(row: pd.Series, packet_row: pd.Series | dict[str, Any], risk_state: dict[str, Any]) -> tuple[str, str]:
    approval_stage = text(row.get("approval_stage")).upper()
    approval_status = text(row.get("approval_status")).upper()
    change_status = text(row.get("change_status")).upper()
    impact = text(row.get("impact_category")).upper()
    master = text(risk_state.get("master_risk_action")).upper()

    if approval_stage == "BLOCKED" or change_status.startswith("BLOCKED"):
        return "BLOCKED_NOT_ELIGIBLE", "This change is blocked. De-risk first, then rerun Step136-141."
    if "MISSING" in approval_stage or "INCOMPLETE" in approval_stage:
        return "QA_GAP_FIX_FIRST", "Approval QA found missing evidence or incomplete manual fields."
    if approval_stage == "WAITING_FOR_MANUAL_APPROVAL" or approval_status != "APPROVED":
        return "WAITING_MANUAL_APPROVAL", "Human approval fields are not complete; do not stage dry-run."
    if approval_stage != "READY_FOR_DRY_RUN":
        return "MONITOR_ONLY", "No dry-run stage is available from the current approval state."
    if impact == "TIGHTEN_NO_NEW_BREACH":
        if master in {"SIZE_DOWN", "REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            return "DRY_RUN_ALLOWED_DEFENSIVE_ONLY", "Approved tightening can run in shadow mode, but the book is defensive and cannot expand risk."
        return "DRY_RUN_READY", "Approved tightening can run in shadow mode for observation only."
    return "DRY_RUN_REVIEW_REQUIRED", "Approved item still needs reviewer confirmation because impact category is not a simple conservative tighten."


def plan_for_stage(stage: str, row: pd.Series, packet_row: pd.Series | dict[str, Any], risk_state: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    current_limit = text(packet_row.get("current_limit_display")) or text(row.get("current_limit_display"))
    simulated_limit = text(packet_row.get("simulated_limit_display")) or text(row.get("simulated_limit_display"))
    master = text(risk_state.get("master_risk_action")) or "NO_DATA"
    common_pause = (
        "Pause if master risk action worsens, any new hard/reduce-only breach appears, "
        "liquidity worsens, evidence pack breaks, or reviewer withdraws approval."
    )
    if stage.startswith("BLOCKED"):
        return {
            "dry_run_phase": "NOT_STARTED",
            "shadow_days_required": 0,
            "shadow_limit_display": current_limit,
            "monitored_metrics": "master_risk_action; blocked_change_status",
            "success_criteria": "Current book is de-risked and Step136-141 reruns without blocked status.",
            "pause_trigger": "Already blocked.",
            "next_action": "Do not stage. Reduce risk first.",
        }
    if stage == "WAITING_MANUAL_APPROVAL":
        return {
            "dry_run_phase": "NOT_STARTED",
            "shadow_days_required": 0,
            "shadow_limit_display": current_limit,
            "monitored_metrics": "approval_status; approved_by; approval_date; approval_notes",
            "success_criteria": "Human reviewer completes approval template and Step140 returns READY_FOR_DRY_RUN.",
            "pause_trigger": "Manual reviewer rejects or leaves approval fields incomplete.",
            "next_action": "Wait for manual approval; rerun Step140-141 after template update.",
        }
    if stage == "QA_GAP_FIX_FIRST":
        return {
            "dry_run_phase": "NOT_STARTED",
            "shadow_days_required": 0,
            "shadow_limit_display": current_limit,
            "monitored_metrics": "required_evidence; required_tests; rollback_trigger",
            "success_criteria": "All QA gaps are fixed and Step140 returns READY_FOR_DRY_RUN.",
            "pause_trigger": "Evidence or rollback fields remain incomplete.",
            "next_action": "Fix QA gaps before dry-run.",
        }
    if stage in {"DRY_RUN_ALLOWED_DEFENSIVE_ONLY", "DRY_RUN_READY"}:
        return {
            "dry_run_phase": "PHASE_1_SHADOW_OBSERVE",
            "shadow_days_required": 5,
            "shadow_limit_display": simulated_limit,
            "monitored_metrics": (
                "current_status; simulated_status; master_risk_action; hard_or_reduce_breach_count; "
                "size_down_breach_count; evidence_grade"
            ),
            "success_criteria": (
                "For at least 5 daily runs, simulated status stays no worse than current status, "
                "no new hard/reduce-only breach appears, and Evidence Pack remains usable."
            ),
            "pause_trigger": common_pause,
            "next_action": (
                f"Run shadow review only. Current master risk action is {master}; "
                f"hard/reduce breaches={counts.get('hard_or_reduce', 0)}, size-down breaches={counts.get('size_down', 0)}."
            ),
        }
    return {
        "dry_run_phase": "REVIEW_REQUIRED",
        "shadow_days_required": 0,
        "shadow_limit_display": current_limit,
        "monitored_metrics": "approval_stage; impact_category; reviewer_decision",
        "success_criteria": "Reviewer explicitly confirms the impact category is safe for dry-run.",
        "pause_trigger": "Any uncertainty in impact category or evidence.",
        "next_action": "Manual review required before dry-run.",
    }


def build_outputs(
    approval: pd.DataFrame,
    packet: pd.DataFrame,
    template: pd.DataFrame,
    change: pd.DataFrame,
    breach: pd.DataFrame,
    risk_state: dict[str, Any],
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if approval.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    packet_by_id = rows_by_id(packet, "change_id")
    change_by_id = rows_by_id(change, "change_id")
    template_by_id = rows_by_id(template, "change_id")
    counts = breach_counts(breach)
    readiness_rows = []
    plan_rows = []
    monitor_rows = []

    for _, row in approval.iterrows():
        cid = text(row.get("change_id"))
        ticket_id = text(row.get("ticket_id"))
        packet_row = packet_by_id.get(cid, {})
        change_row = change_by_id.get(cid, {})
        template_row = template_by_id.get(cid, {})
        stage, stage_note = dry_run_stage(row, packet_row, risk_state)
        plan = plan_for_stage(stage, row, packet_row, risk_state, counts)
        evidence_grade = evidence_grade_for(ticket_id, evidence)
        can_enter = stage in {"DRY_RUN_READY", "DRY_RUN_ALLOWED_DEFENSIVE_ONLY"}

        readiness_rows.append({
            "change_id": cid,
            "ticket_id": ticket_id,
            "review_priority": row.get("review_priority", ""),
            "control": row.get("control", ""),
            "change_status": row.get("change_status", ""),
            "approval_stage": row.get("approval_stage", ""),
            "approval_status": row.get("approval_status", ""),
            "dry_run_stage": stage,
            "dry_run_note": stage_note,
            "evidence_grade": evidence_grade,
            "manual_reviewer": text(template_row.get("approved_by")),
            "manual_approval_date": text(template_row.get("approval_date")),
            "can_enter_dry_run": can_enter,
            "can_apply_live": False,
            "research_only": True,
        })

        plan_rows.append({
            "change_id": cid,
            "ticket_id": ticket_id,
            "control": row.get("control", ""),
            "dry_run_phase": plan["dry_run_phase"],
            "shadow_days_required": plan["shadow_days_required"],
            "current_limit_display": text(packet_row.get("current_limit_display")) or text(change_row.get("current_limit_display")),
            "shadow_limit_display": plan["shadow_limit_display"],
            "monitored_metrics": plan["monitored_metrics"],
            "success_criteria": plan["success_criteria"],
            "pause_trigger": plan["pause_trigger"],
            "next_action": plan["next_action"],
            "can_apply_live": False,
            "research_only": True,
        })

        monitor_rows.append({
            "change_id": cid,
            "ticket_id": ticket_id,
            "control": row.get("control", ""),
            "observation_time": now_str(),
            "dry_run_stage": stage,
            "master_risk_action": risk_state.get("master_risk_action", "NO_DATA"),
            "master_exposure_multiplier": risk_state.get("master_exposure_multiplier", np.nan),
            "recommended_gross_exposure": risk_state.get("recommended_gross_exposure", np.nan),
            "hard_or_reduce_breach_count": counts["hard_or_reduce"],
            "size_down_breach_count": counts["size_down"],
            "review_breach_count": counts["review"],
            "current_status": text(packet_row.get("current_status")),
            "simulated_status": text(packet_row.get("simulated_status")),
            "evidence_grade": evidence_grade,
            "monitor_status": "READY_TO_OBSERVE" if can_enter else "NOT_OBSERVING",
            "required_next_observation": plan["next_action"],
            "can_apply_live": False,
            "research_only": True,
        })

    readiness = pd.DataFrame(readiness_rows).sort_values(["review_priority", "dry_run_stage", "control"]).reset_index(drop=True)
    plan = pd.DataFrame(plan_rows).sort_values(["dry_run_phase", "control"]).reset_index(drop=True)
    monitor = pd.DataFrame(monitor_rows).sort_values(["monitor_status", "control"]).reset_index(drop=True)
    return readiness, plan, monitor


def build_state(readiness: pd.DataFrame, monitor: pd.DataFrame) -> dict[str, Any]:
    if readiness.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    stages = readiness["dry_run_stage"].astype(str).str.upper()
    ready = int(readiness["can_enter_dry_run"].astype(bool).sum())
    waiting = int(stages.eq("WAITING_MANUAL_APPROVAL").sum())
    blocked = int(stages.str.startswith("BLOCKED").sum())
    qa_gap = int(stages.str.contains("QA_GAP|MISSING|INCOMPLETE", regex=True).sum())
    if qa_gap:
        overall = "DRY_RUN_QA_GAPS"
    elif ready:
        overall = "DRY_RUN_READY_RESEARCH_ONLY"
    elif waiting and blocked:
        overall = "WAITING_APPROVAL_AND_BLOCKED"
    elif waiting:
        overall = "WAITING_MANUAL_APPROVAL"
    elif blocked:
        overall = "BLOCKED_ONLY"
    else:
        overall = "MONITOR_ONLY"
    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Dry-run guard only. It can stage shadow observation after manual approval, but can never apply live thresholds.",
        "changes_checked": int(len(readiness)),
        "dry_run_ready_count": ready,
        "waiting_manual_approval_count": waiting,
        "blocked_count": blocked,
        "qa_gap_count": qa_gap,
        "monitor_rows": int(len(monitor)),
        "can_apply_live_count": int(readiness["can_apply_live"].astype(bool).sum()),
        "outputs": {
            "readiness": OUT_READINESS.name,
            "plan": OUT_PLAN.name,
            "monitor": OUT_MONITOR.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    approval = read_csv_safe(IN_APPROVAL)
    packet = read_csv_safe(IN_PACKET)
    template = read_csv_safe(IN_TEMPLATE)
    change = read_csv_safe(IN_CHANGE)
    breach = read_csv_safe(IN_BREACH)
    risk_state = read_json_safe(IN_RISK_DESK)
    evidence = read_csv_safe(IN_EVIDENCE)

    readiness, plan, monitor = build_outputs(approval, packet, template, change, breach, risk_state, evidence)
    state = build_state(readiness, monitor)

    readiness.to_csv(OUT_READINESS, index=False)
    plan.to_csv(OUT_PLAN, index=False)
    monitor.to_csv(OUT_MONITOR, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Changes checked: {state.get('changes_checked', 0)}",
        f"- Dry-run ready: {state.get('dry_run_ready_count', 0)}",
        f"- Waiting manual approval: {state.get('waiting_manual_approval_count', 0)}",
        f"- Blocked: {state.get('blocked_count', 0)}",
        f"- QA gaps: {state.get('qa_gap_count', 0)}",
        f"- Can apply live: {state.get('can_apply_live_count', 0)}",
        "",
        "## Dry-Run Readiness",
        "",
        df_to_markdown(readiness, max_rows=50),
        "",
        "## Dry-Run Plan",
        "",
        df_to_markdown(plan, max_rows=50),
        "",
        "## Dry-Run Monitor",
        "",
        df_to_markdown(monitor, max_rows=50),
        "",
        "## Product Truth",
        "",
        "This is a shadow rollout guard. It never applies thresholds and never enables live orders.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 141 - Risk Policy Dry-Run Guard", sections)

    print(f"wrote {OUT_READINESS.name} rows={len(readiness)}")
    print(f"wrote {OUT_PLAN.name} rows={len(plan)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
