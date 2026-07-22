#!/usr/bin/env python3
"""
Canyon v9 - Step 134: Risk Policy Decision Log
==============================================

Research-only. No broker connection. No live orders.

Step133 creates draft-only risk policy review items. Step134 turns those items
into stable review tickets and a decision log. It preserves manual decision
fields across reruns so the log can become a lightweight governance record.

This step does not approve, reject, or apply any threshold change by itself.

Outputs:
  risk_policy_decision_log.csv
  risk_policy_open_tickets.csv
  risk_policy_decision_state.json
  risk_policy_decision_report.md
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

IN_QUEUE = ROOT / "risk_policy_review_queue.csv"
IN_STATE = ROOT / "risk_policy_review_state.json"

OUT_LOG = ROOT / "risk_policy_decision_log.csv"
OUT_OPEN = ROOT / "risk_policy_open_tickets.csv"
OUT_STATE = ROOT / "risk_policy_decision_state.json"
OUT_REPORT = ROOT / "risk_policy_decision_report.md"

MANUAL_FIELDS = [
    "decision_status",
    "decision",
    "reviewer",
    "decision_date",
    "decision_reason",
    "manual_notes",
]

BASE_COLUMNS = [
    "ticket_id",
    "created_at",
    "last_seen_at",
    "currently_in_queue",
    "review_bucket",
    "review_priority",
    "risk_area",
    "control",
    "calibration_status",
    "calibration_mode",
    "policy_decision_candidate",
    "approval_required",
    "decision_status",
    "decision",
    "reviewer",
    "decision_date",
    "decision_reason",
    "manual_notes",
    "current_value",
    "current_limit",
    "proposed_warning_limit",
    "proposed_hard_limit",
    "target_hard_limit_after_more_history",
    "historical_or_cross_sectional_percentile",
    "sample_n",
    "implementation_status",
    "can_auto_apply",
    "policy_rationale",
    "required_next_action",
    "source_file",
    "research_only",
]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def slug(value: str) -> str:
    out = []
    for ch in value.upper():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    text = "".join(out).strip("_")
    return text[:42] if text else "UNKNOWN"


def ticket_id(row: pd.Series) -> str:
    raw = "|".join([
        clean_text(row.get("risk_area")),
        clean_text(row.get("control")),
        clean_text(row.get("policy_decision_candidate")),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8].upper()
    return f"RP-{int(safe_float(row.get('review_priority'), 9)):02d}-{slug(clean_text(row.get('control')))}-{digest}"


def review_bucket(row: pd.Series) -> str:
    priority = int(safe_float(row.get("review_priority"), 9))
    decision = clean_text(row.get("policy_decision_candidate")).upper()
    status = clean_text(row.get("calibration_status")).upper()
    if priority == 1:
        return "De-risk first"
    if decision in {"TIGHTEN_LIMIT_REVIEW", "LOOSEN_LIMIT_REVIEW", "ADD_DRAFT_LIMIT_REVIEW"}:
        return "Policy change review"
    if decision == "KEEP_LIMIT_COLLECT_HISTORY" or status == "MISSING_HISTORY":
        return "History needed"
    if decision == "KEEP_LIMIT_SIZE_DOWN":
        return "Monitor and size down"
    return "Monitor"


def default_decision_status(bucket: str) -> str:
    if bucket == "De-risk first":
        return "PENDING_DE_RISK_REVIEW"
    if bucket == "Policy change review":
        return "PENDING_POLICY_REVIEW"
    if bucket == "History needed":
        return "PENDING_DATA_HISTORY"
    if bucket == "Monitor and size down":
        return "PENDING_SIZE_REVIEW"
    return "PENDING_MONITORING"


def normalize_bool(value: Any) -> bool:
    text = clean_text(value).upper()
    if text in {"TRUE", "1", "YES", "Y"}:
        return True
    return False


def normalize_existing(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    out = existing.copy()
    for col in BASE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    if "ticket_id" in out.columns:
        out["ticket_id"] = out["ticket_id"].astype(str)
        out = out[out["ticket_id"].str.strip() != ""].copy()
        out = out.drop_duplicates("ticket_id", keep="last")
    return out[BASE_COLUMNS].reset_index(drop=True)


def build_current_ticket_rows(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    rows = []
    now = now_str()
    for _, row in queue.iterrows():
        bucket = review_bucket(row)
        rows.append({
            "ticket_id": ticket_id(row),
            "created_at": now,
            "last_seen_at": now,
            "currently_in_queue": True,
            "review_bucket": bucket,
            "review_priority": int(safe_float(row.get("review_priority"), 9)),
            "risk_area": row.get("risk_area", ""),
            "control": row.get("control", ""),
            "calibration_status": row.get("calibration_status", ""),
            "calibration_mode": row.get("calibration_mode", ""),
            "policy_decision_candidate": row.get("policy_decision_candidate", ""),
            "approval_required": row.get("approval_required", ""),
            "decision_status": default_decision_status(bucket),
            "decision": "UNDECIDED",
            "reviewer": "",
            "decision_date": "",
            "decision_reason": "",
            "manual_notes": "",
            "current_value": safe_float(row.get("current_value")),
            "current_limit": safe_float(row.get("current_limit")),
            "proposed_warning_limit": safe_float(row.get("proposed_warning_limit")),
            "proposed_hard_limit": safe_float(row.get("proposed_hard_limit")),
            "target_hard_limit_after_more_history": safe_float(row.get("target_hard_limit_after_more_history")),
            "historical_or_cross_sectional_percentile": safe_float(row.get("historical_or_cross_sectional_percentile")),
            "sample_n": int(safe_float(row.get("sample_n"), 0)),
            "implementation_status": "DRAFT_ONLY",
            "can_auto_apply": False,
            "policy_rationale": row.get("policy_rationale", ""),
            "required_next_action": row.get("required_next_action", ""),
            "source_file": row.get("source_file", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows)[BASE_COLUMNS]


def merge_with_existing(current: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_existing(existing)
    if current.empty and existing.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    existing_by_id = {str(r["ticket_id"]): r for _, r in existing.iterrows()}
    current_ids = set(current["ticket_id"].astype(str)) if not current.empty else set()
    merged_rows: list[dict[str, Any]] = []

    for _, row in current.iterrows():
        tid = str(row["ticket_id"])
        out = row.to_dict()
        if tid in existing_by_id:
            prev = existing_by_id[tid]
            out["created_at"] = prev.get("created_at") or out["created_at"]
            for field in MANUAL_FIELDS:
                prev_value = prev.get(field, "")
                if clean_text(prev_value):
                    out[field] = prev_value
        merged_rows.append(out)

    for _, row in existing.iterrows():
        tid = str(row["ticket_id"])
        if tid in current_ids:
            continue
        out = row.to_dict()
        out["currently_in_queue"] = False
        if clean_text(out.get("decision_status")).startswith("PENDING"):
            out["decision_status"] = "STALE_PENDING_REVIEW"
            out["required_next_action"] = "This ticket is no longer in the latest Step133 queue. Review before closing."
        merged_rows.append(out)

    out_df = pd.DataFrame(merged_rows)
    for col in BASE_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df["review_priority"] = pd.to_numeric(out_df["review_priority"], errors="coerce").fillna(9).astype(int)
    out_df["currently_in_queue"] = out_df["currently_in_queue"].apply(normalize_bool)
    out_df["can_auto_apply"] = False
    out_df = out_df.sort_values(
        ["currently_in_queue", "review_priority", "risk_area", "control"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return out_df[BASE_COLUMNS]


def build_open_tickets(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    status = log["decision_status"].astype(str).str.upper()
    open_mask = log["currently_in_queue"].astype(bool) & ~status.isin({
        "APPROVED_KEEP",
        "APPROVED_TIGHTEN",
        "APPROVED_LOOSEN",
        "REJECTED",
        "CLOSED",
    })
    return log.loc[open_mask].copy().reset_index(drop=True)


def build_state(log: pd.DataFrame, open_tickets: pd.DataFrame, policy_state: dict[str, Any]) -> dict[str, Any]:
    if log.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    status = open_tickets["decision_status"].astype(str).str.upper() if not open_tickets.empty else pd.Series(dtype=str)
    priority = pd.to_numeric(open_tickets.get("review_priority", pd.Series(dtype=float)), errors="coerce")
    auto_apply_count = int(log["can_auto_apply"].astype(bool).sum()) if "can_auto_apply" in log.columns else 0

    if (priority == 1).any():
        overall = "OPEN_DE_RISK_TICKETS"
    elif status.str.contains("POLICY", regex=False).any():
        overall = "OPEN_POLICY_TICKETS"
    elif len(open_tickets) > 0:
        overall = "OPEN_MONITORING_TICKETS"
    else:
        overall = "NO_OPEN_TICKETS"

    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Stable risk policy ticket log. Manual decision fields are preserved across reruns. Nothing auto-applies.",
        "total_tickets": int(len(log)),
        "open_tickets": int(len(open_tickets)),
        "priority_1_open_tickets": int((priority == 1).sum()) if len(priority) else 0,
        "pending_policy_review": int(status.str.contains("POLICY", regex=False).sum()) if len(status) else 0,
        "pending_data_history": int(status.str.contains("DATA_HISTORY", regex=False).sum()) if len(status) else 0,
        "auto_apply_count": auto_apply_count,
        "can_auto_apply_anything": auto_apply_count > 0,
        "upstream_policy_status": policy_state.get("overall_status", "NO_DATA") if policy_state else "NO_DATA",
        "outputs": {
            "decision_log": OUT_LOG.name,
            "open_tickets": OUT_OPEN.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    queue = read_csv_safe(IN_QUEUE)
    policy_state = read_json_safe(IN_STATE)
    existing = read_csv_safe(OUT_LOG)

    current = build_current_ticket_rows(queue)
    log = merge_with_existing(current, existing)
    open_tickets = build_open_tickets(log)
    state = build_state(log, open_tickets, policy_state)

    log.to_csv(OUT_LOG, index=False)
    open_tickets.to_csv(OUT_OPEN, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Total tickets: {state.get('total_tickets', 0)}",
        f"- Open tickets: {state.get('open_tickets', 0)}",
        f"- Priority 1 open tickets: {state.get('priority_1_open_tickets', 0)}",
        f"- Pending policy review: {state.get('pending_policy_review', 0)}",
        f"- Pending data history: {state.get('pending_data_history', 0)}",
        f"- Auto-apply count: {state.get('auto_apply_count', 0)}",
        "",
        "## Open Tickets",
        "",
        df_to_markdown(open_tickets, max_rows=40),
        "",
        "## Full Decision Log",
        "",
        df_to_markdown(log, max_rows=60),
        "",
        "## How to Use This Log",
        "",
        "Manual fields are intentionally preserved across reruns: `decision_status`, `decision`, `reviewer`, `decision_date`, `decision_reason`, and `manual_notes`. Fill those fields after review. The script will update current market evidence but keep the human decision record.",
        "",
        "## Product Truth",
        "",
        "This is a lightweight local governance ledger, not a regulated approval system. It is still research-only and paper-only.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 134 - Risk Policy Decision Log", sections)

    print(f"wrote {OUT_LOG.name} rows={len(log)}")
    print(f"wrote {OUT_OPEN.name} rows={len(open_tickets)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
