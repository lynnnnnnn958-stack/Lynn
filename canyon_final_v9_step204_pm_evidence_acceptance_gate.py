#!/usr/bin/env python3
"""
Canyon v9 Step 204 - PM Evidence Acceptance Gate.

Research-only. No broker connection. No live orders.

Step203 creates evidence suggestions. This step turns those suggestions into a
human acceptance queue. It preserves human edits, separates accepted evidence
from undecided evidence, and writes a ready-to-copy patch draft. It never edits
the official PM review input, never approves a ticker, and never unlocks options.

Outputs:
  pm_review_evidence_acceptance_state.json
  pm_review_evidence_acceptance_input.csv
  pm_review_evidence_acceptance_status.csv
  pm_review_evidence_acceptance_ready_patch.csv
  pm_review_evidence_acceptance_conflicts.csv
  pm_review_evidence_acceptance_report.md
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
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


OUT_STATE = ROOT / "pm_review_evidence_acceptance_state.json"
OUT_INPUT = ROOT / "pm_review_evidence_acceptance_input.csv"
OUT_STATUS = ROOT / "pm_review_evidence_acceptance_status.csv"
OUT_READY_PATCH = ROOT / "pm_review_evidence_acceptance_ready_patch.csv"
OUT_CONFLICTS = ROOT / "pm_review_evidence_acceptance_conflicts.csv"
OUT_REPORT = ROOT / "pm_review_evidence_acceptance_report.md"


ACCEPTANCE_STATUSES = {
    "Needs human decision",
    "Accept",
    "Reject",
    "Needs outside confirmation",
    "Ignore",
}

HUMAN_COLUMNS = [
    "acceptance_status",
    "reviewer",
    "review_date",
    "human_note",
]

INPUT_COLUMNS = [
    "suggestion_id",
    "suggestion_status",
    "ticker",
    "field_name",
    "suggested_value",
    "confidence",
    "acceptance_status",
    "reviewer",
    "review_date",
    "human_note",
    "official_value",
    "will_fill_draft",
    "human_confirmation_needed",
    "how_to_decide",
    "rationale",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

PATCH_FIELDS = [
    "thesis_plain",
    "earnings_date",
    "expected_event_move_pct",
    "event_size_policy",
    "liquidity_snapshot_date",
    "bid_ask_spread_bps",
    "avg_daily_dollar_volume_check",
    "sector_confirmed",
    "crowding_check",
    "news_proof_note",
    "execution_proof_note",
    "paper_stop_pct",
    "option_route_requested",
    "decision_note",
    "last_updated",
]

PATCH_OUTPUT_COLUMNS = [
    "ticker",
    "patch_status",
    "accepted_field_count",
    "blocked_field_count",
] + PATCH_FIELDS + [
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

PM_APPROVAL_REQUIRED_FIELDS = {
    "reviewer",
    "review_date",
    "approved_cap_pct",
    "paper_stop_pct",
    "thesis_plain",
    "earnings_date",
    "expected_event_move_pct",
    "event_size_policy",
    "liquidity_snapshot_date",
    "bid_ask_spread_bps",
    "avg_daily_dollar_volume_check",
    "sector_confirmed",
    "crowding_check",
    "decision_note",
}


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


def is_filled(value: Any) -> bool:
    return bool(as_text(value, ""))


def short(value: Any, limit: int = 360) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def norm_status(value: Any) -> str:
    text = as_text(value, "Needs human decision").strip()
    key = text.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ACCEPT": "Accept",
        "ACCEPTED": "Accept",
        "YES": "Accept",
        "Y": "Accept",
        "REJECT": "Reject",
        "REJECTED": "Reject",
        "NO": "Reject",
        "N": "Reject",
        "NEEDS_EXTERNAL_CONFIRMATION": "Needs outside confirmation",
        "NEEDS_OUTSIDE_CONFIRMATION": "Needs outside confirmation",
        "OUTSIDE_CONFIRMATION": "Needs outside confirmation",
        "EXTERNAL_CONFIRMATION": "Needs outside confirmation",
        "EXTERNAL": "Needs outside confirmation",
        "IGNORE": "Ignore",
        "IGNORED": "Ignore",
        "NEEDS_HUMAN_DECISION": "Needs human decision",
        "PENDING": "Needs human decision",
        "": "Needs human decision",
    }
    if key in aliases:
        return aliases[key]
    if text in ACCEPTANCE_STATUSES:
        return text
    return "Needs human decision"


def suggestion_id(row: pd.Series) -> str:
    ticker = clean_ticker(row.get("ticker"))
    field = as_text(row.get("field_name"), "").lower()
    value = as_text(row.get("suggested_value"), "")
    source = as_text(row.get("source_files"), "")
    raw = "|".join([ticker, field, value, source])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{ticker}-{field}-{digest}"


def one_by_ticker(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby("ticker", sort=False)}


def existing_by_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "suggestion_id" not in df.columns:
        return {}
    work = df.copy()
    work["suggestion_id"] = work["suggestion_id"].map(lambda x: as_text(x, ""))
    work = work[work["suggestion_id"] != ""].copy()
    return {sid: grp.iloc[0] for sid, grp in work.groupby("suggestion_id", sort=False)}


def how_to_decide(field: str, confidence: str) -> str:
    field = as_text(field, "")
    confidence = as_text(confidence, "")
    if field == "thesis_plain":
        return "Accept only if the story is understandable and the linked news actually belongs to this ticker."
    if field == "earnings_date":
        return "Accept only after checking the next earnings date from a reliable calendar."
    if field == "expected_event_move_pct":
        return "Accept only after checking the option-implied move or a documented fallback."
    if field == "event_size_policy":
        return "Accept only if the event rule is conservative enough for the earnings/news window."
    if field in {"liquidity_snapshot_date", "bid_ask_spread_bps", "avg_daily_dollar_volume_check"}:
        return "Accept only after checking a current quote or liquidity file."
    if field in {"sector_confirmed", "crowding_check"}:
        return "Accept only if the sector, peers, and concentration read are reasonable."
    if field == "news_proof_note":
        return "Accept only if the headline, source, timing, and price reaction can be verified."
    if field == "execution_proof_note":
        return "Accept only if the trading-cost note is supported by spread or liquidity data."
    if field == "paper_stop_pct":
        return "Accept only if the stop is small enough for the risk seed and not wider than the system stop."
    if field == "option_route_requested":
        return "Accept NO only. Calls and puts require a separate options gate."
    if field == "decision_note":
        return "Accept only if it clearly says this is still review-only."
    if field == "last_updated":
        return "Accept if this matches the latest evidence review date."
    if confidence.lower() == "low":
        return "Low-confidence suggestion: verify outside the model before accepting."
    return "Accept only if the source file and rationale make sense to a human reviewer."


def build_acceptance_input(suggestions: pd.DataFrame, official: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    official_map = one_by_ticker(official)
    existing_map = existing_by_id(existing)
    current_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    if not suggestions.empty:
        work = suggestions.copy()
        if "ticker" in work.columns:
            work["ticker"] = work["ticker"].apply(clean_ticker)
        else:
            work["ticker"] = ""
        work = work[work["ticker"] != ""].copy()

        for _, src in work.iterrows():
            sid = suggestion_id(src)
            current_ids.add(sid)
            old = existing_map.get(sid)
            ticker = clean_ticker(src.get("ticker"))
            field = as_text(src.get("field_name"), "")
            official_row = official_map.get(ticker, pd.Series(dtype=object))

            row = {
                "suggestion_id": sid,
                "suggestion_status": "Suggested today",
                "ticker": ticker,
                "field_name": field,
                "suggested_value": as_text(src.get("suggested_value"), ""),
                "confidence": as_text(src.get("confidence"), ""),
                "acceptance_status": "Needs human decision",
                "reviewer": "",
                "review_date": "",
                "human_note": "",
                "official_value": as_text(official_row.get(field), ""),
                "will_fill_draft": as_text(src.get("will_fill_draft"), ""),
                "human_confirmation_needed": "Yes",
                "how_to_decide": how_to_decide(field, src.get("confidence")),
                "rationale": short(src.get("rationale"), 420),
                "source_files": as_text(src.get("source_files"), ""),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            }
            if old is not None:
                for col in HUMAN_COLUMNS:
                    if col in old.index and is_filled(old.get(col)):
                        row[col] = old.get(col)
            row["acceptance_status"] = norm_status(row["acceptance_status"])
            rows.append(row)

    if not existing.empty and "suggestion_id" in existing.columns:
        for _, old in existing.iterrows():
            sid = as_text(old.get("suggestion_id"), "")
            if not sid or sid in current_ids:
                continue
            row = {col: as_text(old.get(col), "") for col in INPUT_COLUMNS}
            row["suggestion_id"] = sid
            row["suggestion_status"] = "No longer suggested today"
            row["acceptance_status"] = norm_status(row.get("acceptance_status"))
            row["research_only"] = True
            row["no_broker_connection"] = True
            row["no_live_orders"] = True
            rows.append(row)

    out = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    if not out.empty:
        out["acceptance_status"] = out["acceptance_status"].apply(norm_status)
        out = out.drop_duplicates("suggestion_id", keep="first").reset_index(drop=True)
    return out


def conflict_rows_for_acceptance(row: pd.Series) -> list[dict[str, Any]]:
    status = norm_status(row.get("acceptance_status"))
    ticker = clean_ticker(row.get("ticker"))
    field = as_text(row.get("field_name"), "")
    suggested = as_text(row.get("suggested_value"), "")
    official = as_text(row.get("official_value"), "")
    confidence = as_text(row.get("confidence"), "")
    out: list[dict[str, Any]] = []

    if status == "Accept":
        if confidence.lower() == "low":
            out.append({
                "ticker": ticker,
                "field_name": field,
                "acceptance_status": status,
                "conflict_reason": "Accepted evidence is low confidence. Check an outside source before copying it.",
                "suggested_value": suggested,
                "official_value": official,
                "source_files": as_text(row.get("source_files"), ""),
            })
        if not is_filled(row.get("reviewer")) or not is_filled(row.get("review_date")):
            out.append({
                "ticker": ticker,
                "field_name": field,
                "acceptance_status": status,
                "conflict_reason": "Accepted evidence needs reviewer name and review date.",
                "suggested_value": suggested,
                "official_value": official,
                "source_files": as_text(row.get("source_files"), ""),
            })
        if field == "option_route_requested" and suggested.upper() not in {"NO", "NONE", "NO_OPTIONS"}:
            out.append({
                "ticker": ticker,
                "field_name": field,
                "acceptance_status": status,
                "conflict_reason": "Options cannot be accepted through this gate. Calls and puts need the separate options gate.",
                "suggested_value": suggested,
                "official_value": official,
                "source_files": as_text(row.get("source_files"), ""),
            })
        if official and suggested and official.strip() != suggested.strip():
            out.append({
                "ticker": ticker,
                "field_name": field,
                "acceptance_status": status,
                "conflict_reason": "Official PM review file already has a different value. Human must choose which one is correct.",
                "suggested_value": suggested,
                "official_value": official,
                "source_files": as_text(row.get("source_files"), ""),
            })

    if status in {"Reject", "Needs outside confirmation"} and field in PM_APPROVAL_REQUIRED_FIELDS:
        out.append({
            "ticker": ticker,
            "field_name": field,
            "acceptance_status": status,
            "conflict_reason": "This is a required PM review field. If it is rejected or still needs outside confirmation, the ticker cannot move forward.",
            "suggested_value": suggested,
            "official_value": official,
            "source_files": as_text(row.get("source_files"), ""),
        })

    return out


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    suggestions = read_csv_safe(ROOT / "pm_review_evidence_autofill_suggestions.csv")
    official = read_csv_safe(ROOT / "risk_seed_pm_review_input.csv")
    existing = read_csv_safe(OUT_INPUT)

    if suggestions.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_AUTOFILL_SUGGESTIONS",
            "plain_answer": "Step204 needs Step203 evidence suggestions first.",
            "suggestion_count": 0,
            "accepted_count": 0,
            "undecided_count": 0,
            "ready_patch_rows": 0,
            "conflict_count": 0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, empty, state

    acceptance = build_acceptance_input(suggestions, official, existing)

    conflict_rows: list[dict[str, Any]] = []
    for _, row in acceptance.iterrows():
        conflict_rows.extend(conflict_rows_for_acceptance(row))
    conflicts = pd.DataFrame(conflict_rows, columns=[
        "ticker",
        "field_name",
        "acceptance_status",
        "conflict_reason",
        "suggested_value",
        "official_value",
        "source_files",
    ])

    accepted = acceptance[acceptance["acceptance_status"] == "Accept"].copy() if not acceptance.empty else pd.DataFrame()
    conflict_keys = set()
    if not conflicts.empty:
        conflict_keys = {
            (clean_ticker(row.get("ticker")), as_text(row.get("field_name"), ""), as_text(row.get("suggested_value"), ""))
            for _, row in conflicts.iterrows()
        }

    patch_rows: list[dict[str, Any]] = []
    if not accepted.empty:
        for ticker, grp in accepted.groupby("ticker", sort=False):
            row = {
                "ticker": ticker,
                "patch_status": "Ready for manual copy",
                "accepted_field_count": int(len(grp)),
                "blocked_field_count": 0,
                "source_files": "; ".join(sorted({as_text(x, "") for x in grp.get("source_files", pd.Series(dtype=str)) if as_text(x, "")})[:8]),
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            }
            blocked = 0
            for _, src in grp.iterrows():
                key = (clean_ticker(src.get("ticker")), as_text(src.get("field_name"), ""), as_text(src.get("suggested_value"), ""))
                if key in conflict_keys:
                    blocked += 1
                    continue
                field = as_text(src.get("field_name"), "")
                if field in PATCH_FIELDS:
                    row[field] = as_text(src.get("suggested_value"), "")
            row["blocked_field_count"] = blocked
            if blocked:
                row["patch_status"] = "Needs conflict review before manual copy"
            patch_rows.append(row)

    ready_patch = pd.DataFrame(patch_rows)
    if not ready_patch.empty:
        ordered_cols = [
            "ticker",
            "patch_status",
            "accepted_field_count",
            "blocked_field_count",
        ] + [c for c in PATCH_FIELDS if c in ready_patch.columns] + [
            "source_files",
            "research_only",
            "no_broker_connection",
            "no_live_orders",
        ]
        ready_patch = ready_patch[ordered_cols]
    else:
        ready_patch = pd.DataFrame(columns=PATCH_OUTPUT_COLUMNS)

    status_rows: list[dict[str, Any]] = []
    if not acceptance.empty:
        ticker_conflicts = conflicts.groupby("ticker").size().to_dict() if not conflicts.empty else {}
        for ticker, grp in acceptance.groupby("ticker", sort=False):
            accepted_count = int((grp["acceptance_status"] == "Accept").sum())
            rejected_count = int((grp["acceptance_status"] == "Reject").sum())
            outside_count = int((grp["acceptance_status"] == "Needs outside confirmation").sum())
            ignored_count = int((grp["acceptance_status"] == "Ignore").sum())
            undecided_count = int((grp["acceptance_status"] == "Needs human decision").sum())
            conflict_count = int(ticker_conflicts.get(ticker, 0))
            if accepted_count and not conflict_count:
                next_step = "Accepted evidence is ready for a human to copy into the official PM review file."
            elif conflict_count:
                next_step = "Fix the conflict before copying anything into the official PM review file."
            elif outside_count:
                next_step = "Collect outside confirmation for the required fields."
            else:
                next_step = "Decide whether to accept, reject, or ignore each evidence suggestion."
            status_rows.append({
                "ticker": ticker,
                "suggestion_count": int(len(grp)),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "needs_external_confirmation_count": outside_count,
                "ignored_count": ignored_count,
                "undecided_count": undecided_count,
                "conflict_count": conflict_count,
                "accepted_fields": ", ".join(grp.loc[grp["acceptance_status"] == "Accept", "field_name"].astype(str).tolist()[:12]),
                "next_step": next_step,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })
    status = pd.DataFrame(status_rows)

    accepted_count = int((acceptance.get("acceptance_status", pd.Series(dtype=str)) == "Accept").sum()) if not acceptance.empty else 0
    rejected_count = int((acceptance.get("acceptance_status", pd.Series(dtype=str)) == "Reject").sum()) if not acceptance.empty else 0
    outside_count = int((acceptance.get("acceptance_status", pd.Series(dtype=str)) == "Needs outside confirmation").sum()) if not acceptance.empty else 0
    undecided_count = int((acceptance.get("acceptance_status", pd.Series(dtype=str)) == "Needs human decision").sum()) if not acceptance.empty else 0
    conflict_count = int(len(conflicts))

    state = {
        "date": today_str(),
        "status": "PM_EVIDENCE_ACCEPTANCE_GATE_ACTIVE",
        "suggestion_count": int(len(acceptance)),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "needs_external_confirmation_count": outside_count,
        "undecided_count": undecided_count,
        "ready_patch_rows": int(len(ready_patch)),
        "conflict_count": conflict_count,
        "plain_answer": (
            f"Evidence acceptance gate is active. {len(acceptance)} suggestions are available for human review. "
            f"{accepted_count} are accepted, {outside_count} need outside confirmation, and {undecided_count} are still undecided. "
            "Nothing has been copied into the official PM review file."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return acceptance, status, ready_patch, conflicts, state


def main() -> None:
    acceptance, status, ready_patch, conflicts, state = build_outputs()
    acceptance.to_csv(OUT_INPUT, index=False)
    status.to_csv(OUT_STATUS, index=False)
    ready_patch.to_csv(OUT_READY_PATCH, index=False)
    conflicts.to_csv(OUT_CONFLICTS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## How To Use\n\nOpen `pm_review_evidence_acceptance_input.csv`. For each row, set `acceptance_status` to Accept, Reject, Needs outside confirmation, or Ignore. Add reviewer and review_date when accepting. Then rerun Step204.",
        "## Acceptance Queue\n\n" + df_to_markdown(acceptance.head(180)),
        "## Ticker Status\n\n" + df_to_markdown(status.head(120)),
        "## Ready Patch Draft\n\n" + df_to_markdown(ready_patch.head(120)),
        "## Conflicts\n\n" + df_to_markdown(conflicts.head(180)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 204 - PM Evidence Acceptance Gate", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Suggestions in review: {state['suggestion_count']}")
    print(f"[OK] Accepted: {state['accepted_count']}")
    print(f"[OK] Undecided: {state['undecided_count']}")
    print(f"[OK] Ready patch rows: {state['ready_patch_rows']}")
    print(f"[OK] Conflicts: {state['conflict_count']}")
    print("[OK] Official PM review input changed: False")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
