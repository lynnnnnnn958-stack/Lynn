#!/usr/bin/env python3
"""
Canyon v9 Step 207 - PM Evidence Proof-to-Acceptance Bridge.

Research-only. No broker connection. No live orders.

Step206 captures outside-source proof. This step bridges verified proof back to
the Step204 evidence acceptance queue by creating a manual patch file. It does
not edit Step204, does not approve a ticker, and does not unlock size or
options. A human still has to copy/accept the matching evidence row.

Outputs:
  pm_evidence_proof_acceptance_bridge_state.json
  pm_evidence_proof_acceptance_bridge.csv
  pm_evidence_proof_acceptance_patch.csv
  pm_evidence_proof_acceptance_conflicts.csv
  pm_evidence_proof_acceptance_bridge_report.md
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "pm_evidence_proof_acceptance_bridge_state.json"
OUT_BRIDGE = ROOT / "pm_evidence_proof_acceptance_bridge.csv"
OUT_PATCH = ROOT / "pm_evidence_proof_acceptance_patch.csv"
OUT_CONFLICTS = ROOT / "pm_evidence_proof_acceptance_conflicts.csv"
OUT_REPORT = ROOT / "pm_evidence_proof_acceptance_bridge_report.md"


BRIDGE_COLUMNS = [
    "ticker",
    "field_name",
    "bridge_state",
    "matching_step204_row_found",
    "current_step204_decision",
    "proposed_step204_decision",
    "step204_suggestion_id",
    "step204_suggested_value",
    "observed_value",
    "source_name",
    "source_url",
    "reviewer",
    "review_date",
    "proof_note",
    "next_step",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

PATCH_COLUMNS = [
    "ticker",
    "field_name",
    "step204_suggestion_id",
    "acceptance_status",
    "reviewer",
    "review_date",
    "human_note",
    "observed_value",
    "source_name",
    "source_url",
    "copy_instruction",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

CONFLICT_COLUMNS = [
    "ticker",
    "field_name",
    "conflict_reason",
    "current_step204_decision",
    "step204_suggested_value",
    "observed_value",
    "source_name",
    "reviewer",
    "review_date",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def short(value: Any, limit: int = 320) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_decision(value: Any) -> str:
    text = as_text(value, "Needs human decision")
    key = text.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ACCEPT": "Accept",
        "ACCEPTED": "Accept",
        "REJECT": "Reject",
        "REJECTED": "Reject",
        "NEEDS_OUTSIDE_CONFIRMATION": "Needs outside confirmation",
        "NEEDS_EXTERNAL_CONFIRMATION": "Needs outside confirmation",
        "IGNORE": "Ignore",
        "IGNORED": "Ignore",
        "NEEDS_HUMAN_DECISION": "Needs human decision",
        "PENDING": "Needs human decision",
    }
    return aliases.get(key, text)


def find_step204_match(ticker: str, field: str, observed_value: str, step204: pd.DataFrame) -> pd.Series | None:
    if step204.empty:
        return None
    work = step204.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    candidates = work[(work["ticker"] == ticker) & (work["field_name"].astype(str) == field)].copy()
    if candidates.empty:
        return None

    observed = observed_value.strip()
    if observed:
        exact = candidates[candidates["suggested_value"].astype(str).str.strip() == observed]
        if not exact.empty:
            return exact.iloc[0]
    return candidates.iloc[0]


def conflict_reasons(ready_row: pd.Series, step204_row: pd.Series | None) -> list[str]:
    reasons: list[str] = []
    if step204_row is None:
        return ["No matching Step204 evidence row was found for this ticker and field."]

    decision = normalize_decision(step204_row.get("acceptance_status"))
    if decision in {"Reject", "Ignore"}:
        reasons.append("Step204 row is currently rejected or ignored. Human must reopen it before accepting proof.")
    if decision == "Accept":
        reasons.append("Step204 row is already accepted. Confirm no duplicate acceptance is needed.")

    observed = as_text(ready_row.get("observed_value"), "")
    suggested = as_text(step204_row.get("suggested_value"), "")
    if observed and suggested and observed != suggested:
        reasons.append("Observed proof value differs from the Step204 suggested value. Human must choose the correct wording.")

    for col, label in [
        ("source_name", "source name"),
        ("observed_value", "observed value"),
        ("reviewer", "reviewer"),
        ("review_date", "review date"),
    ]:
        if not as_text(ready_row.get(col), ""):
            reasons.append(f"Verified proof is missing {label}.")
    return reasons


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ready = read_csv_safe(ROOT / "pm_evidence_source_proof_ready_for_acceptance.csv")
    proof_state = read_csv_safe(ROOT / "pm_evidence_source_proof_status.csv")
    step204 = read_csv_safe(ROOT / "pm_review_evidence_acceptance_input.csv")

    if ready.empty:
        bridge = pd.DataFrame(columns=BRIDGE_COLUMNS)
        patch = pd.DataFrame(columns=PATCH_COLUMNS)
        conflicts = pd.DataFrame(columns=CONFLICT_COLUMNS)
        proof_rows = int(len(proof_state)) if not proof_state.empty else 0
        state = {
            "date": today_str(),
            "status": "NO_VERIFIED_SOURCE_PROOF_READY",
            "ready_proof_rows": 0,
            "bridge_rows": 0,
            "patch_rows": 0,
            "conflict_count": 0,
            "step204_rows": int(len(step204)),
            "proof_status_rows": proof_rows,
            "plain_answer": (
                "Proof-to-acceptance bridge is active, but no verified outside-source proof is ready yet. "
                "Fill and verify Step206 proof rows first; this step will then create a manual Step204 acceptance patch."
            ),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return bridge, patch, conflicts, state

    bridge_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for _, ready_row in ready.iterrows():
        ticker = clean_ticker(ready_row.get("ticker"))
        field = as_text(ready_row.get("field_name"), "")
        observed = as_text(ready_row.get("observed_value"), "")
        match = find_step204_match(ticker, field, observed, step204)
        reasons = conflict_reasons(ready_row, match)

        current_decision = normalize_decision(match.get("acceptance_status")) if match is not None else "No matching row"
        suggestion_id = as_text(match.get("suggestion_id"), "") if match is not None else ""
        suggested = as_text(match.get("suggested_value"), "") if match is not None else ""
        source_name = as_text(ready_row.get("source_name"), "")
        source_url = as_text(ready_row.get("source_url"), "")
        reviewer = as_text(ready_row.get("reviewer"), "")
        review_date = as_text(ready_row.get("review_date"), "")
        proof_note = as_text(ready_row.get("proof_note"), "")

        bridge_state = "Ready for manual Step204 accept" if not reasons else "Needs bridge review"
        next_step = (
            "Copy the patch row into pm_review_evidence_acceptance_input.csv, then rerun Step204."
            if bridge_state == "Ready for manual Step204 accept"
            else "Resolve bridge conflicts before changing Step204."
        )

        bridge_rows.append({
            "ticker": ticker,
            "field_name": field,
            "bridge_state": bridge_state,
            "matching_step204_row_found": "Yes" if match is not None else "No",
            "current_step204_decision": current_decision,
            "proposed_step204_decision": "Accept",
            "step204_suggestion_id": suggestion_id,
            "step204_suggested_value": short(suggested, 260),
            "observed_value": short(observed, 260),
            "source_name": source_name,
            "source_url": source_url,
            "reviewer": reviewer,
            "review_date": review_date,
            "proof_note": short(proof_note, 260),
            "next_step": next_step,
            "source_files": "pm_evidence_source_proof_ready_for_acceptance.csv; pm_review_evidence_acceptance_input.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        if reasons:
            for reason in reasons:
                conflict_rows.append({
                    "ticker": ticker,
                    "field_name": field,
                    "conflict_reason": reason,
                    "current_step204_decision": current_decision,
                    "step204_suggested_value": short(suggested, 260),
                    "observed_value": short(observed, 260),
                    "source_name": source_name,
                    "reviewer": reviewer,
                    "review_date": review_date,
                    "source_files": "pm_evidence_source_proof_ready_for_acceptance.csv; pm_review_evidence_acceptance_input.csv",
                    "research_only": True,
                    "no_broker_connection": True,
                    "no_live_orders": True,
                })
            continue

        note_parts = [
            "Verified outside-source proof.",
            f"Source: {source_name}.",
            f"Observed value: {observed}.",
        ]
        if source_url:
            note_parts.append(f"URL/file: {source_url}.")
        if proof_note:
            note_parts.append(f"Note: {proof_note}.")
        patch_rows.append({
            "ticker": ticker,
            "field_name": field,
            "step204_suggestion_id": suggestion_id,
            "acceptance_status": "Accept",
            "reviewer": reviewer,
            "review_date": review_date,
            "human_note": short(" ".join(note_parts), 520),
            "observed_value": observed,
            "source_name": source_name,
            "source_url": source_url,
            "copy_instruction": "Manually copy these values into the matching Step204 acceptance row; then rerun Step204.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    bridge = pd.DataFrame(bridge_rows, columns=BRIDGE_COLUMNS)
    patch = pd.DataFrame(patch_rows, columns=PATCH_COLUMNS)
    conflicts = pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS)
    state = {
        "date": today_str(),
        "status": "PM_EVIDENCE_PROOF_ACCEPTANCE_BRIDGE_ACTIVE",
        "ready_proof_rows": int(len(ready)),
        "bridge_rows": int(len(bridge)),
        "patch_rows": int(len(patch)),
        "conflict_count": int(len(conflicts)),
        "step204_rows": int(len(step204)),
        "proof_status_rows": int(len(proof_state)) if not proof_state.empty else 0,
        "plain_answer": (
            f"Proof-to-acceptance bridge is active. {len(ready)} verified proof rows were checked against Step204. "
            f"{len(patch)} manual acceptance patch rows are ready, and {len(conflicts)} conflicts need review."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return bridge, patch, conflicts, state


def main() -> None:
    bridge, patch, conflicts, state = build_outputs()
    bridge.to_csv(OUT_BRIDGE, index=False)
    patch.to_csv(OUT_PATCH, index=False)
    conflicts.to_csv(OUT_CONFLICTS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Bridge Rows\n\n" + df_to_markdown(bridge.head(120)),
        "## Manual Step204 Patch\n\n" + df_to_markdown(patch.head(120)),
        "## Conflicts\n\n" + df_to_markdown(conflicts.head(120)),
        "## How To Use\n\nIf patch rows exist, copy them manually into `pm_review_evidence_acceptance_input.csv`, then rerun Step204. Do not use this bridge as approval to add size or options.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 207 - PM Evidence Proof-to-Acceptance Bridge", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Ready proof rows: {state['ready_proof_rows']}")
    print(f"[OK] Manual patch rows: {state['patch_rows']}")
    print(f"[OK] Conflicts: {state['conflict_count']}")
    print("[OK] Step204 input changed: False")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
