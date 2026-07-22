#!/usr/bin/env python3
"""
Canyon v9 - Step 133: Risk Policy Review
========================================

Research-only. No broker connection. No live orders.

Step132 says whether a threshold is calibrated, too loose, too tight, or
missing history. Step133 turns that into a controlled policy-review queue.

This step never edits Step111-118 limits. It creates draft-only proposals that
must be reviewed by a human before any policy change is made.

Outputs:
  risk_policy_review_queue.csv
  risk_policy_threshold_proposals.csv
  risk_policy_review_state.json
  risk_policy_review_report.md
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

IN_CALIBRATION = ROOT / "risk_threshold_calibration.csv"
IN_SCORECARD = ROOT / "risk_calibration_scorecard.csv"
IN_RISK_DESK = ROOT / "risk_desk_overview.json"

OUT_QUEUE = ROOT / "risk_policy_review_queue.csv"
OUT_PROPOSALS = ROOT / "risk_policy_threshold_proposals.csv"
OUT_STATE = ROOT / "risk_policy_review_state.json"
OUT_REPORT = ROOT / "risk_policy_review_report.md"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def fmt_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value * 100:.2f}%"


def clipped_positive(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return max(float(value), 0.0)


def review_priority(status: str, mode: str) -> int:
    status = str(status).upper()
    mode = str(mode).upper()
    if status == "ACTIVE_BREACH":
        return 1
    if status in {"NEEDS_POLICY_LIMIT", "TOO_LOOSE_REVIEW", "TOO_TIGHT_REVIEW"}:
        return 2
    if status == "REVIEW_HIGH_CURRENT_RISK":
        return 3
    if status == "MISSING_HISTORY" or mode == "POINT_ESTIMATE":
        return 4
    return 5


def approval_required(status: str, area: str, mode: str) -> str:
    status = str(status).upper()
    area = str(area)
    mode = str(mode).upper()
    if status == "ACTIVE_BREACH":
        return "Risk manager plus PM sign-off"
    if status in {"TOO_LOOSE_REVIEW", "TOO_TIGHT_REVIEW", "NEEDS_POLICY_LIMIT"}:
        return "Risk committee review"
    if "Macro" in area or "Correlation" in area:
        return "Risk committee review"
    if mode in {"POINT_ESTIMATE", "SCENARIO_LIBRARY"}:
        return "Data and risk research review"
    return "Risk manager review"


def decision_candidate(status: str) -> str:
    status = str(status).upper()
    if status == "ACTIVE_BREACH":
        return "KEEP_CURRENT_LIMIT_AND_DE_RISK"
    if status == "TOO_LOOSE_REVIEW":
        return "TIGHTEN_LIMIT_REVIEW"
    if status == "TOO_TIGHT_REVIEW":
        return "LOOSEN_LIMIT_REVIEW"
    if status == "NEEDS_POLICY_LIMIT":
        return "ADD_DRAFT_LIMIT_REVIEW"
    if status == "MISSING_HISTORY":
        return "KEEP_LIMIT_COLLECT_HISTORY"
    if status == "REVIEW_HIGH_CURRENT_RISK":
        return "KEEP_LIMIT_SIZE_DOWN"
    return "KEEP_CURRENT_LIMIT"


def proposed_limits(row: pd.Series) -> tuple[float, float, float]:
    current_value = safe_float(row.get("current_value"))
    current_limit = safe_float(row.get("current_limit"))
    calibrated_warning = safe_float(row.get("calibrated_warning"))
    calibrated_hard = safe_float(row.get("calibrated_hard"))
    status = str(row.get("calibration_status", "")).upper()

    if status == "NEEDS_POLICY_LIMIT":
        warning = calibrated_warning
        hard = calibrated_hard
        target = calibrated_hard
    elif status == "TOO_LOOSE_REVIEW":
        target = calibrated_hard
        # Do not create an instant breach only because the policy was tightened.
        hard = max(calibrated_hard, current_value * 1.05) if np.isfinite(current_value) and np.isfinite(calibrated_hard) else calibrated_hard
        warning = min(current_limit, calibrated_warning) if np.isfinite(current_limit) and np.isfinite(calibrated_warning) else calibrated_warning
    elif status == "TOO_TIGHT_REVIEW":
        target = calibrated_hard
        hard = max(current_limit, calibrated_hard) if np.isfinite(current_limit) and np.isfinite(calibrated_hard) else calibrated_hard
        warning = max(current_limit * 0.8, calibrated_warning) if np.isfinite(current_limit) and np.isfinite(calibrated_warning) else calibrated_warning
    elif status == "ACTIVE_BREACH":
        warning = current_limit * 0.85 if np.isfinite(current_limit) else calibrated_warning
        hard = current_limit
        target = current_limit
    else:
        warning = current_limit * 0.85 if np.isfinite(current_limit) else calibrated_warning
        hard = current_limit if np.isfinite(current_limit) else calibrated_hard
        target = hard

    return clipped_positive(warning), clipped_positive(hard), clipped_positive(target)


def policy_rationale(row: pd.Series, decision: str) -> str:
    status = str(row.get("calibration_status", ""))
    mode = str(row.get("calibration_mode", ""))
    control = str(row.get("control", ""))
    current_value = safe_float(row.get("current_value"))
    current_limit = safe_float(row.get("current_limit"))
    percentile = safe_float(row.get("historical_or_cross_sectional_percentile"))
    sample_n = int(safe_float(row.get("sample_n"), 0))

    base = (
        f"{control}: status={status}, mode={mode}, current={fmt_pct(current_value)}, "
        f"limit={fmt_pct(current_limit)}, percentile={percentile:.1f}%, sample={sample_n}."
    )
    if decision == "KEEP_CURRENT_LIMIT_AND_DE_RISK":
        return base + " Current exposure breaches the limit; reduce risk first and do not loosen policy during a breach."
    if decision == "TIGHTEN_LIMIT_REVIEW":
        return base + " The current limit looks loose versus the calibrated hard level; tighten only after review."
    if decision == "LOOSEN_LIMIT_REVIEW":
        return base + " The current limit may over-block normal conditions; loosen only after review."
    if decision == "ADD_DRAFT_LIMIT_REVIEW":
        return base + " No explicit policy limit exists; add a draft warning/hard threshold before using this as a hard gate."
    if decision == "KEEP_LIMIT_COLLECT_HISTORY":
        return base + " Evidence is not enough for calibration; keep current policy and collect point-in-time history."
    if decision == "KEEP_LIMIT_SIZE_DOWN":
        return base + " The current risk is high in the available distribution; size down rather than changing the limit."
    return base + " The current limit is acceptable as a first-pass policy; monitor realized breach rates."


def next_action(decision: str) -> str:
    if decision == "KEEP_CURRENT_LIMIT_AND_DE_RISK":
        return "Do not change the limit now. De-risk the current book, then re-run Step132/133."
    if decision == "TIGHTEN_LIMIT_REVIEW":
        return "Create a reviewed policy ticket to phase in the tighter hard limit."
    if decision == "LOOSEN_LIMIT_REVIEW":
        return "Create a reviewed policy ticket with false-positive evidence before loosening."
    if decision == "ADD_DRAFT_LIMIT_REVIEW":
        return "Draft a new warning and hard limit, then require approval before enforcement."
    if decision == "KEEP_LIMIT_COLLECT_HISTORY":
        return "Collect point-in-time observations and keep this marked prototype until history is sufficient."
    if decision == "KEEP_LIMIT_SIZE_DOWN":
        return "Keep the policy limit unchanged and reduce exposure until risk falls below watch level."
    return "Keep current policy. Recheck in the next daily risk run."


def build_policy_queue(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in calibration.iterrows():
        status = str(row.get("calibration_status", "")).upper()
        mode = str(row.get("calibration_mode", ""))
        area = str(row.get("risk_area", ""))
        decision = decision_candidate(status)
        warning, hard, target = proposed_limits(row)
        rows.append({
            "review_priority": review_priority(status, mode),
            "risk_area": area,
            "control": row.get("control", ""),
            "calibration_status": status,
            "calibration_mode": mode,
            "policy_decision_candidate": decision,
            "current_value": safe_float(row.get("current_value")),
            "current_limit": safe_float(row.get("current_limit")),
            "proposed_warning_limit": warning,
            "proposed_hard_limit": hard,
            "target_hard_limit_after_more_history": target,
            "historical_or_cross_sectional_percentile": safe_float(row.get("historical_or_cross_sectional_percentile")),
            "sample_n": int(safe_float(row.get("sample_n"), 0)),
            "approval_required": approval_required(status, area, mode),
            "implementation_status": "DRAFT_ONLY",
            "can_auto_apply": False,
            "policy_rationale": policy_rationale(row, decision),
            "required_next_action": next_action(decision),
            "source_file": row.get("source_file", ""),
            "research_only": True,
        })

    out = pd.DataFrame(rows)
    sort_cols = ["review_priority", "historical_or_cross_sectional_percentile", "risk_area", "control"]
    return out.sort_values(sort_cols, ascending=[True, False, True, True]).reset_index(drop=True)


def build_threshold_proposals(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()
    cols = [
        "risk_area", "control", "policy_decision_candidate",
        "current_limit", "proposed_warning_limit", "proposed_hard_limit",
        "target_hard_limit_after_more_history", "approval_required",
        "implementation_status", "can_auto_apply", "required_next_action",
    ]
    out = queue[[c for c in cols if c in queue.columns]].copy()
    out["proposal_note"] = out["policy_decision_candidate"].map({
        "KEEP_CURRENT_LIMIT_AND_DE_RISK": "No threshold change while breached.",
        "TIGHTEN_LIMIT_REVIEW": "Candidate for phased tightening.",
        "LOOSEN_LIMIT_REVIEW": "Candidate for loosening only with false-positive evidence.",
        "ADD_DRAFT_LIMIT_REVIEW": "Candidate for adding an explicit limit.",
        "KEEP_LIMIT_COLLECT_HISTORY": "Keep current placeholder and collect history.",
        "KEEP_LIMIT_SIZE_DOWN": "Keep policy and reduce exposure.",
        "KEEP_CURRENT_LIMIT": "No policy change proposed.",
    }).fillna("Review required.")
    return out


def build_state(queue: pd.DataFrame, calibration_state: dict[str, Any], risk_desk_state: dict[str, Any]) -> dict[str, Any]:
    if queue.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }

    decisions = queue["policy_decision_candidate"].astype(str)
    priority_1 = int((queue["review_priority"] == 1).sum())
    policy_change_candidates = int(decisions.isin([
        "TIGHTEN_LIMIT_REVIEW", "LOOSEN_LIMIT_REVIEW", "ADD_DRAFT_LIMIT_REVIEW",
    ]).sum())
    needs_history = int(decisions.eq("KEEP_LIMIT_COLLECT_HISTORY").sum())
    auto_apply_count = int(queue["can_auto_apply"].astype(bool).sum())

    if priority_1 > 0:
        overall = "DE_RISK_FIRST"
    elif policy_change_candidates > 0:
        overall = "POLICY_REVIEW_REQUIRED"
    elif needs_history > 0:
        overall = "DATA_HISTORY_REQUIRED"
    else:
        overall = "POLICY_OK"

    return {
        "run_time": now_str(),
        "overall_status": overall,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Draft-only policy review. No threshold is changed automatically.",
        "controls_in_queue": int(len(queue)),
        "priority_1_items": priority_1,
        "policy_change_candidates": policy_change_candidates,
        "needs_history_count": needs_history,
        "auto_apply_count": auto_apply_count,
        "can_auto_apply_anything": auto_apply_count > 0,
        "risk_desk_master_action": risk_desk_state.get("master_risk_action", "NO_DATA") if risk_desk_state else "NO_DATA",
        "calibration_status": calibration_state.get("overall_status", "NO_DATA") if calibration_state else "NO_DATA",
        "outputs": {
            "review_queue": OUT_QUEUE.name,
            "threshold_proposals": OUT_PROPOSALS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    calibration = read_csv_safe(IN_CALIBRATION)
    scorecard = read_csv_safe(IN_SCORECARD)
    calibration_state = read_json_safe(ROOT / "risk_threshold_calibration_state.json")
    risk_desk_state = read_json_safe(IN_RISK_DESK)

    queue = build_policy_queue(calibration)
    proposals = build_threshold_proposals(queue)
    state = build_state(queue, calibration_state, risk_desk_state)

    queue.to_csv(OUT_QUEUE, index=False)
    proposals.to_csv(OUT_PROPOSALS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Controls in queue: {state.get('controls_in_queue', 0)}",
        f"- Priority 1 de-risk-first items: {state.get('priority_1_items', 0)}",
        f"- Policy change candidates: {state.get('policy_change_candidates', 0)}",
        f"- Needs more history: {state.get('needs_history_count', 0)}",
        f"- Auto-apply count: {state.get('auto_apply_count', 0)}",
        "",
        "## Policy Review Queue",
        "",
        df_to_markdown(queue, max_rows=40),
        "",
        "## Threshold Proposals",
        "",
        df_to_markdown(proposals, max_rows=40),
    ]
    if not scorecard.empty:
        sections.extend([
            "",
            "## Step132 Calibration Scorecard Input",
            "",
            df_to_markdown(scorecard, max_rows=20),
        ])
    sections.extend([
        "",
        "## Product Truth",
        "",
        "This is a draft-only policy workflow. It protects the system from silently moving risk thresholds after a calibration run. Any real threshold change still needs human approval and a separate implementation step.",
    ])
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 133 - Risk Policy Review", sections)

    print(f"wrote {OUT_QUEUE.name} rows={len(queue)}")
    print(f"wrote {OUT_PROPOSALS.name} rows={len(proposals)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
