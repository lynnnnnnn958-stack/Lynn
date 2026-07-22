#!/usr/bin/env python3
"""
Canyon v9 Step 214 - Proof Intake Safe Apply.

Research-only. No broker connection. No live orders.

Step213 tells the user what proof to fill. Step214 creates one clean intake
sheet for human proof entry and safely applies only rows explicitly marked
APPLY. If no row is marked APPLY, it only refreshes the template and preview.

This step can update pm_evidence_source_proof_input.csv, but only after:
  1. apply_decision is APPLY
  2. required human fields are present
  3. news proof has price/volume reaction checks
  4. a backup of the original proof input is written first

Outputs:
  quant_fund_proof_intake_state.json
  quant_fund_proof_intake_template.csv
  quant_fund_proof_intake_user_entry.csv
  quant_fund_proof_intake_apply_preview.csv
  quant_fund_proof_intake_applied_rows.csv
  quant_fund_proof_intake_rejected_rows.csv
  quant_fund_proof_intake_audit.csv
  quant_fund_proof_intake_report.md
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    now_str,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "quant_fund_proof_intake_state.json"
OUT_TEMPLATE = ROOT / "quant_fund_proof_intake_template.csv"
OUT_USER_ENTRY = ROOT / "quant_fund_proof_intake_user_entry.csv"
OUT_PREVIEW = ROOT / "quant_fund_proof_intake_apply_preview.csv"
OUT_APPLIED = ROOT / "quant_fund_proof_intake_applied_rows.csv"
OUT_REJECTED = ROOT / "quant_fund_proof_intake_rejected_rows.csv"
OUT_AUDIT = ROOT / "quant_fund_proof_intake_audit.csv"
OUT_REPORT = ROOT / "quant_fund_proof_intake_report.md"
PROOF_INPUT = ROOT / "pm_evidence_source_proof_input.csv"


HUMAN_COLUMNS = [
    "apply_decision",
    "proof_status",
    "source_name",
    "source_url_or_file",
    "observed_value",
    "observed_time",
    "price_reaction_checked",
    "volume_reaction_checked",
    "reviewer",
    "review_date",
    "proof_note",
]

INTAKE_COLUMNS = [
    "entry_rank",
    "proof_id",
    "ticker",
    "proof_type",
    "plain_task",
    "question_to_answer",
    "source_to_open",
    "fields_to_fill_now",
    "good_example",
    "bad_example",
    "local_source_files",
    "editable_file",
    "apply_decision",
    "proof_status",
    "source_name",
    "source_url_or_file",
    "observed_value",
    "observed_time",
    "price_reaction_checked",
    "volume_reaction_checked",
    "reviewer",
    "review_date",
    "proof_note",
    "after_apply",
    "do_not_do",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

PREVIEW_COLUMNS = [
    "proof_id",
    "ticker",
    "proof_type",
    "apply_decision",
    "validation_state",
    "will_apply",
    "missing_or_problem",
    "updated_fields",
    "next_step",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

AUDIT_COLUMNS = [
    "timestamp",
    "proof_id",
    "ticker",
    "audit_event",
    "detail",
    "backup_file",
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


def short(value: Any, limit: int = 260) -> str:
    text = " ".join(as_text(value, "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def guard_flags(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def normalize_apply(value: Any) -> str:
    text = as_text(value, "WAIT").upper().replace("-", "_").replace(" ", "_")
    if text in {"APPLY", "YES", "Y", "UPDATE"}:
        return "APPLY"
    if text in {"REJECT", "REJECTED"}:
        return "REJECT"
    if text in {"IGNORE", "NOT_NEEDED", "NOT NEEDED"}:
        return "IGNORE"
    return "WAIT"


def normalize_proof_status(value: Any, apply_decision: str) -> str:
    text = as_text(value, "").strip()
    key = text.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "VERIFY": "Verified",
        "VERIFIED": "Verified",
        "ACCEPT": "Verified",
        "NEEDS_PROOF": "Needs proof",
        "PENDING": "Needs proof",
        "REJECT": "Rejected",
        "REJECTED": "Rejected",
        "SOURCE_UNAVAILABLE": "Source unavailable",
        "NOT_NEEDED": "Not needed",
        "IGNORE": "Not needed",
    }
    if key in aliases:
        return aliases[key]
    if apply_decision == "APPLY":
        return "Verified"
    return "Needs proof"


def yes_no_or_blank(value: Any) -> str:
    text = as_text(value, "").lower()
    if text in {"yes", "y", "true", "checked", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "unchecked", "0"}:
        return "No"
    return ""


def build_template() -> pd.DataFrame:
    cards = read_csv_safe(ROOT / "quant_fund_proof_fill_cards.csv")
    if cards.empty:
        return pd.DataFrame(columns=INTAKE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for idx, row in cards.iterrows():
        proof_id = as_text(row.get("proof_id"), "")
        if not proof_id:
            continue
        rows.append(guard_flags({
            "entry_rank": int(idx) + 1,
            "proof_id": proof_id,
            "ticker": clean_ticker(row.get("ticker")),
            "proof_type": as_text(row.get("proof_type"), ""),
            "plain_task": short(row.get("plain_task"), 220),
            "question_to_answer": short(row.get("question_to_answer"), 320),
            "source_to_open": short(row.get("source_to_open"), 360),
            "fields_to_fill_now": as_text(row.get("fields_to_fill_now"), ""),
            "good_example": short(row.get("good_example"), 240),
            "bad_example": short(row.get("bad_example"), 240),
            "local_source_files": as_text(row.get("local_source_files"), ""),
            "editable_file": "pm_evidence_source_proof_input.csv",
            "apply_decision": "WAIT",
            "proof_status": "Needs proof",
            "source_name": "",
            "source_url_or_file": "",
            "observed_value": "",
            "observed_time": "",
            "price_reaction_checked": "",
            "volume_reaction_checked": "",
            "reviewer": "",
            "review_date": "",
            "proof_note": "",
            "after_apply": "Rerun Steps 206, 207, 204, 212, 213, and 214.",
            "do_not_do": "Do not mark APPLY unless a human can inspect the source.",
        }))
    return pd.DataFrame(rows, columns=INTAKE_COLUMNS)


def existing_by_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if df.empty or "proof_id" not in df.columns:
        return out
    for _, row in df.iterrows():
        pid = as_text(row.get("proof_id"), "")
        if pid and pid not in out:
            out[pid] = row
    return out


def merge_user_entry(template: pd.DataFrame) -> pd.DataFrame:
    old = read_csv_safe(OUT_USER_ENTRY)
    if template.empty:
        return pd.DataFrame(columns=INTAKE_COLUMNS)
    old_map = existing_by_id(old)
    rows: list[dict[str, Any]] = []
    for _, row in template.iterrows():
        out = row.to_dict()
        old_row = old_map.get(as_text(row.get("proof_id"), ""))
        if old_row is not None:
            for col in HUMAN_COLUMNS:
                if col in old_row.index:
                    old_value = as_text(old_row.get(col), "")
                    if old_value:
                        out[col] = old_value
        out["apply_decision"] = normalize_apply(out.get("apply_decision"))
        out["proof_status"] = normalize_proof_status(out.get("proof_status"), out["apply_decision"])
        rows.append(guard_flags(out))
    return pd.DataFrame(rows, columns=INTAKE_COLUMNS)


def validate_entry(row: pd.Series) -> tuple[str, list[str]]:
    apply_decision = normalize_apply(row.get("apply_decision"))
    if apply_decision != "APPLY":
        return "Waiting", []

    missing: list[str] = []
    for col, label in [
        ("source_name", "source name"),
        ("observed_value", "observed value"),
        ("reviewer", "reviewer"),
        ("review_date", "review date"),
    ]:
        if not as_text(row.get(col), ""):
            missing.append(label)

    if not as_text(row.get("source_url_or_file"), "") and not as_text(row.get("local_source_files"), ""):
        missing.append("source URL or source file")

    ptype = as_text(row.get("proof_type"), "")
    if ptype == "News proof":
        if not as_text(row.get("observed_time"), ""):
            missing.append("news/source time")
        if yes_no_or_blank(row.get("price_reaction_checked")) not in {"Yes", "No"}:
            missing.append("price reaction checked Yes/No")
        if yes_no_or_blank(row.get("volume_reaction_checked")) not in {"Yes", "No"}:
            missing.append("volume reaction checked Yes/No")

    if missing:
        return "Cannot apply yet", missing
    return "Ready to apply", []


def build_preview(user_entry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in user_entry.iterrows():
        apply_decision = normalize_apply(row.get("apply_decision"))
        state, missing = validate_entry(row)
        will_apply = state == "Ready to apply"
        updated = [
            "proof_status",
            "source_name",
            "source_url",
            "observed_value",
            "observed_time",
            "price_reaction_checked",
            "volume_reaction_checked",
            "reviewer",
            "review_date",
            "proof_note",
        ] if will_apply else []
        rows.append(guard_flags({
            "proof_id": as_text(row.get("proof_id"), ""),
            "ticker": clean_ticker(row.get("ticker")),
            "proof_type": as_text(row.get("proof_type"), ""),
            "apply_decision": apply_decision,
            "validation_state": state,
            "will_apply": "Yes" if will_apply else "No",
            "missing_or_problem": "; ".join(missing) if missing else ("No apply request yet" if apply_decision == "WAIT" else "None"),
            "updated_fields": "; ".join(updated) if updated else "No update",
            "next_step": (
                "This row will update the proof input file, then rerun proof checks."
                if will_apply else
                "Fill missing fields or leave apply decision as WAIT."
            ),
        }))
    return pd.DataFrame(rows, columns=PREVIEW_COLUMNS)


def apply_valid_rows(user_entry: pd.DataFrame, preview: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    proof_input = read_csv_safe(PROOF_INPUT)
    if proof_input.empty:
        return (
            pd.DataFrame(columns=PREVIEW_COLUMNS),
            preview[preview["apply_decision"].astype(str).eq("APPLY")].copy(),
            pd.DataFrame(columns=AUDIT_COLUMNS),
            "",
        )

    ready_ids = set(preview.loc[preview["will_apply"].astype(str).eq("Yes"), "proof_id"].astype(str))
    if not ready_ids:
        audit = pd.DataFrame([guard_flags({
            "timestamp": now_str(),
            "proof_id": "",
            "ticker": "",
            "audit_event": "NO_APPLY_ROWS",
            "detail": "No valid APPLY rows were found. Proof input file was not changed.",
            "backup_file": "",
        })], columns=AUDIT_COLUMNS)
        rejected = preview[preview["apply_decision"].astype(str).eq("APPLY") & ~preview["will_apply"].astype(str).eq("Yes")].copy()
        return pd.DataFrame(columns=PREVIEW_COLUMNS), rejected, audit, ""

    stamp = now_str().replace(":", "").replace(" ", "_").replace("-", "")
    backup = ROOT / f"pm_evidence_source_proof_input_backup_step214_{stamp}.csv"
    proof_input.to_csv(backup, index=False)

    user_map = existing_by_id(user_entry)
    applied_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    work = proof_input.copy()
    if "proof_id" not in work.columns:
        work["proof_id"] = ""

    for idx, input_row in work.iterrows():
        pid = as_text(input_row.get("proof_id"), "")
        if pid not in ready_ids:
            continue
        entry = user_map.get(pid)
        if entry is None:
            continue
        apply_decision = normalize_apply(entry.get("apply_decision"))
        status = normalize_proof_status(entry.get("proof_status"), apply_decision)
        work.at[idx, "proof_status"] = status
        work.at[idx, "source_name"] = as_text(entry.get("source_name"), "")
        work.at[idx, "source_url"] = as_text(entry.get("source_url_or_file"), "")
        work.at[idx, "observed_value"] = as_text(entry.get("observed_value"), "")
        work.at[idx, "observed_time"] = as_text(entry.get("observed_time"), "")
        work.at[idx, "price_reaction_checked"] = yes_no_or_blank(entry.get("price_reaction_checked"))
        work.at[idx, "volume_reaction_checked"] = yes_no_or_blank(entry.get("volume_reaction_checked"))
        work.at[idx, "reviewer"] = as_text(entry.get("reviewer"), "")
        work.at[idx, "review_date"] = as_text(entry.get("review_date"), "")
        work.at[idx, "proof_note"] = as_text(entry.get("proof_note"), "")

        applied_rows.append(guard_flags({
            "proof_id": pid,
            "ticker": clean_ticker(input_row.get("ticker")),
            "proof_type": as_text(entry.get("proof_type"), ""),
            "apply_decision": "APPLY",
            "validation_state": "Applied",
            "will_apply": "Applied",
            "missing_or_problem": "None",
            "updated_fields": "proof_status; source_name; source_url; observed_value; observed_time; price_reaction_checked; volume_reaction_checked; reviewer; review_date; proof_note",
            "next_step": "Rerun Steps 206, 207, 204, 212, 213, and 214.",
        }))
        audit_rows.append(guard_flags({
            "timestamp": now_str(),
            "proof_id": pid,
            "ticker": clean_ticker(input_row.get("ticker")),
            "audit_event": "APPLIED_TO_PROOF_INPUT",
            "detail": "Human intake row marked APPLY and passed validation.",
            "backup_file": backup.name,
        }))

    work.to_csv(PROOF_INPUT, index=False)
    applied = pd.DataFrame(applied_rows, columns=PREVIEW_COLUMNS)
    rejected = preview[preview["apply_decision"].astype(str).eq("APPLY") & ~preview["proof_id"].astype(str).isin(ready_ids)].copy()
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return applied, rejected, audit, backup.name


def build_state(template: pd.DataFrame, user_entry: pd.DataFrame, preview: pd.DataFrame, applied: pd.DataFrame, rejected: pd.DataFrame, audit: pd.DataFrame, backup_file: str) -> dict[str, Any]:
    apply_requests = int((preview["apply_decision"].astype(str) == "APPLY").sum()) if not preview.empty else 0
    waiting = int((preview["apply_decision"].astype(str) == "WAIT").sum()) if not preview.empty else 0
    first = user_entry.iloc[0].to_dict() if not user_entry.empty else {}
    if len(applied):
        answer = f"Proof Intake Safe Apply updated {len(applied)} proof rows. Rerun Steps 206, 207, 204, 212, 213, and 214."
    elif apply_requests:
        answer = f"Proof Intake Safe Apply found {apply_requests} APPLY request(s), but {len(rejected)} need more fields before update."
    else:
        answer = (
            "Proof Intake Safe Apply is ready. No rows are marked APPLY yet, so the proof input file was not changed. "
            "Fill the intake sheet, set apply decision to APPLY for finished rows, then rerun Step214."
        )
    return {
        "date": today_str(),
        "status": "Active",
        "template_rows": int(len(template)),
        "user_entry_rows": int(len(user_entry)),
        "apply_request_count": apply_requests,
        "applied_count": int(len(applied)),
        "rejected_count": int(len(rejected)),
        "waiting_count": waiting,
        "backup_file": backup_file,
        "first_ticker": as_text(first.get("ticker"), ""),
        "first_task": as_text(first.get("plain_task"), ""),
        "user_entry_file": OUT_USER_ENTRY.name,
        "plain_answer": answer,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    template = build_template()
    template.to_csv(OUT_TEMPLATE, index=False)

    user_entry = merge_user_entry(template)
    user_entry.to_csv(OUT_USER_ENTRY, index=False)

    preview = build_preview(user_entry)
    applied, rejected, audit, backup_file = apply_valid_rows(user_entry, preview)

    preview.to_csv(OUT_PREVIEW, index=False)
    applied.to_csv(OUT_APPLIED, index=False)
    rejected.to_csv(OUT_REJECTED, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    state = build_state(template, user_entry, preview, applied, rejected, audit, backup_file)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## How To Use\n\n"
        "1. Open quant_fund_proof_intake_user_entry.csv.\n"
        "2. Fill source name, source URL or file, observed value, reviewer, and review date.\n"
        "3. For news proof, also fill source time plus price and volume reaction checks.\n"
        "4. Set apply_decision to APPLY only for rows you want written back.\n"
        "5. Rerun Step214, then rerun Steps 206, 207, 204, 212, and 213.",
        "## Apply Preview\n\n" + df_to_markdown(preview.head(120)),
        "## Applied Rows\n\n" + df_to_markdown(applied.head(80)),
        "## Rejected APPLY Rows\n\n" + df_to_markdown(rejected.head(80)),
        "## Intake Sheet Preview\n\n" + df_to_markdown(user_entry.head(40)),
        "## Audit\n\n" + df_to_markdown(audit.head(80)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Proof Intake Safe Apply", sections)
    print(
        "Step214 complete: "
        f"{len(user_entry)} intake rows, {state['apply_request_count']} apply requests, "
        f"{len(applied)} applied, {len(rejected)} rejected."
    )


if __name__ == "__main__":
    main()
