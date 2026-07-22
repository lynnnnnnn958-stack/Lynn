#!/usr/bin/env python3
"""
Canyon v9 Step 192 - Manual Proof Review Gate.

Research-only. No broker connection. No live orders.

Step191 creates a fillable proof template. Step192 reads that template after
manual evidence is entered, then decides whether each ticker is still blocked,
needs more proof, can be removed, or can move to watch-only review.

Important:
  - Passing this gate never creates a live order.
  - Passing this gate never allows calls or puts.
  - Passing this gate only means "human can review this name on the watch list."

Outputs:
  sharpe4_manual_proof_review_state.json
  sharpe4_manual_proof_review_gate.csv
  sharpe4_watch_only_review_queue.csv
  sharpe4_manual_proof_missing_fields.csv
  sharpe4_manual_proof_review_report.md
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


IN_TEMPLATE = ROOT / "sharpe4_manual_proof_input_template.csv"
IN_TASKS = ROOT / "sharpe4_proof_workbench_ticker_tasks.csv"
IN_PROMOTION = ROOT / "sharpe4_risk_book_promotion_gate.csv"

OUT_STATE = ROOT / "sharpe4_manual_proof_review_state.json"
OUT_GATE = ROOT / "sharpe4_manual_proof_review_gate.csv"
OUT_WATCH = ROOT / "sharpe4_watch_only_review_queue.csv"
OUT_MISSING = ROOT / "sharpe4_manual_proof_missing_fields.csv"
OUT_REPORT = ROOT / "sharpe4_manual_proof_review_report.md"


MANUAL_COLS = [
    "source_name",
    "source_url_or_file",
    "source_date_or_timestamp",
    "key_numbers",
    "evidence_summary",
    "pass_fail_review",
    "reviewer_notes",
    "next_gate_request",
    "date_recorded",
]

EVIDENCE_COLS = [
    "source_name",
    "source_url_or_file",
    "source_date_or_timestamp",
    "key_numbers",
    "evidence_summary",
    "pass_fail_review",
    "reviewer_notes",
    "date_recorded",
]

BASE_REQUIRED = [
    "source_name",
    "source_url_or_file",
    "source_date_or_timestamp",
    "evidence_summary",
    "pass_fail_review",
    "next_gate_request",
]

NUMBER_REQUIRED_BUCKETS = {
    "Earnings and gap proof",
    "Tail-risk stop proof",
    "Crowding and overlap proof",
    "Spread and fill proof",
    "Event reaction proof",
}

BLOCKER_BY_BUCKET = {
    "Earnings and gap proof": {"earnings proof", "earnings"},
    "Tail-risk stop proof": {"tail risk"},
    "Crowding and overlap proof": {"crowding"},
    "Spread and fill proof": {"tca", "liquidity", "spread"},
    "Event reaction proof": {"event proof", "event"},
    "Source proof": {"source proof", "source"},
}


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def compact(text: Any, limit: int = 220) -> str:
    out = " ".join(as_text(text).split())
    if len(out) <= limit:
        return out
    return out[: limit - 3].rstrip() + "..."


def has_any_manual(row: pd.Series) -> bool:
    return any(as_text(row.get(col)) for col in EVIDENCE_COLS)


def required_fields_for(bucket: str) -> list[str]:
    fields = list(BASE_REQUIRED)
    if bucket in NUMBER_REQUIRED_BUCKETS:
        fields.insert(3, "key_numbers")
    return fields


def missing_fields(row: pd.Series) -> list[str]:
    return [col for col in required_fields_for(as_text(row.get("work_bucket"))) if not as_text(row.get(col))]


def parse_blockers(status_reason: str) -> list[str]:
    text = as_text(status_reason).lower()
    if not text:
        return []
    text = text.replace("blocked by", "").replace(".", "")
    parts = []
    for raw in text.replace(";", ",").split(","):
        item = raw.strip()
        if item:
            parts.append(item)
    return parts


def addressed_blockers(bucket: str) -> set[str]:
    return BLOCKER_BY_BUCKET.get(as_text(bucket), set())


def review_signal(pass_fail_review: str, next_gate_request: str) -> str:
    review = as_text(pass_fail_review).lower()
    request = as_text(next_gate_request).lower()
    combined = f"{review} {request}"

    if "remove" in request:
        return "REMOVE"
    if any(x in combined for x in ["fail", "failed", "reject", "rejected", "invalid", "not pass"]):
        return "FAIL"
    if any(x in combined for x in ["pass", "passed", "approve", "approved", "valid", "ok"]):
        if "keep blocked" in request:
            return "PASS_KEEP_BLOCKED"
        if "watch" in request or "move" in request:
            return "PASS_MOVE"
        return "PASS_REVIEWED"
    if "keep blocked" in request:
        return "KEEP_BLOCKED"
    return "NO_DECISION"


def final_review(row: pd.Series) -> dict[str, str]:
    if not has_any_manual(row):
        return {
            "manual_review_status": "Not reviewed yet",
            "review_reason": "No manual proof has been entered yet.",
            "remaining_blockers": as_text(row.get("status_reason"), "No blocker detail"),
            "final_permission": "Research only. No paper size. No calls or puts.",
            "plain_next_step": "Fill the proof template row first.",
        }

    miss = missing_fields(row)
    if miss:
        return {
            "manual_review_status": "Needs more proof",
            "review_reason": "Missing required fields: " + ", ".join(miss) + ".",
            "remaining_blockers": as_text(row.get("status_reason"), "Manual proof incomplete"),
            "final_permission": "Research only. No paper size. No calls or puts.",
            "plain_next_step": "Complete the missing fields before asking for a watch-only review.",
        }

    signal = review_signal(row.get("pass_fail_review"), row.get("next_gate_request"))
    blockers = parse_blockers(row.get("status_reason"))
    addressed = addressed_blockers(as_text(row.get("work_bucket")))
    remaining = [b for b in blockers if b not in addressed]
    remaining_text = ", ".join(remaining) if remaining else "No unresolved hard blocker from the current proof item."

    if signal == "REMOVE":
        return {
            "manual_review_status": "Remove from queue review",
            "review_reason": "Manual review asks to remove this ticker from the proof queue.",
            "remaining_blockers": remaining_text,
            "final_permission": "Remove or archive from this proof queue. No paper size. No options.",
            "plain_next_step": "Archive the idea unless a new source creates a fresh thesis.",
        }
    if signal == "FAIL":
        return {
            "manual_review_status": "Proof failed; keep blocked",
            "review_reason": "The entered proof failed or was rejected.",
            "remaining_blockers": remaining_text,
            "final_permission": "Research only. No paper size. No calls or puts.",
            "plain_next_step": "Keep blocked or remove from queue.",
        }
    if signal in {"PASS_KEEP_BLOCKED", "KEEP_BLOCKED"}:
        return {
            "manual_review_status": "Proof accepted; keep blocked",
            "review_reason": "The proof is recorded, but the requested next gate is to keep the ticker blocked.",
            "remaining_blockers": remaining_text,
            "final_permission": "Research only. No paper size. No calls or puts.",
            "plain_next_step": "Use the proof as context, but do not promote the ticker.",
        }
    if signal in {"PASS_MOVE", "PASS_REVIEWED"}:
        if remaining:
            return {
                "manual_review_status": "Proof accepted; still blocked",
                "review_reason": "This proof item passed, but another hard blocker remains.",
                "remaining_blockers": remaining_text,
                "final_permission": "Research only. No paper size. No calls or puts.",
                "plain_next_step": "Work the next blocker before watch-only review.",
            }
        return {
            "manual_review_status": "Ready for watch-only review",
            "review_reason": "Required proof is complete and no unresolved hard blocker is left from this gate.",
            "remaining_blockers": remaining_text,
            "final_permission": "Watch-only human review allowed. Still no paper size. Options remain blocked.",
            "plain_next_step": "Review on the watch list; then run a separate risk and options gate before any paper sizing.",
        }

    return {
        "manual_review_status": "Needs reviewer decision",
        "review_reason": "Proof fields are filled, but pass/fail and next gate request are not clear.",
        "remaining_blockers": remaining_text,
        "final_permission": "Research only. No paper size. No calls or puts.",
        "plain_next_step": "Write Pass, Fail, Keep blocked, Move to watch-only review, or Remove from queue.",
    }


def human_status(status: str) -> str:
    text = as_text(status)
    mapping = {
        "Not reviewed yet": "Not checked yet",
        "Needs more proof": "Missing evidence",
        "Needs reviewer decision": "Need a clear yes/no decision",
        "Proof failed; keep blocked": "The evidence did not support it",
        "Proof accepted; keep blocked": "Evidence saved, but still do not touch",
        "Proof accepted; still blocked": "One issue fixed, another issue remains",
        "Ready for watch-only review": "You may study this name next",
        "Remove from queue review": "Probably remove this idea",
    }
    return mapping.get(text, text or "Not checked yet")


def human_touch_permission(status: str) -> str:
    text = as_text(status)
    if text == "Ready for watch-only review":
        return "You may study it. Do not paper trade it yet. Do not use calls or puts."
    if text == "Remove from queue review":
        return "Do not spend more time unless a new source appears."
    return "Do not touch it yet. Read only."


def human_why(row: pd.Series, review: dict[str, str]) -> str:
    status = as_text(review.get("manual_review_status"))
    bucket = as_text(row.get("work_bucket"))
    blocker = as_text(review.get("remaining_blockers"))

    if status == "Not reviewed yet":
        if bucket == "Earnings and gap proof":
            return "We have not written down the next earnings date and possible jump risk yet."
        if bucket == "Tail-risk stop proof":
            return "We have not written the maximum loss rule yet."
        if bucket == "Crowding and overlap proof":
            return "We have not checked whether this is just the same bet as another stock."
        if bucket == "Event reaction proof":
            return "We have not proven the news actually moved this stock or its peers."
        return "The evidence row is still blank."
    if status == "Needs more proof":
        return as_text(review.get("review_reason"), "The evidence row is incomplete.")
    if status == "Ready for watch-only review":
        return "The required evidence is filled, and this specific blocker is cleared."
    if status == "Proof accepted; still blocked":
        return f"The evidence helped, but another problem remains: {blocker}."
    if status == "Proof failed; keep blocked":
        return "The evidence did not support the idea."
    if status == "Remove from queue review":
        return "The review says this idea should be removed from the current queue."
    return as_text(review.get("review_reason"), "The review still needs a clearer decision.")


def human_next_step(row: pd.Series, review: dict[str, str]) -> str:
    status = as_text(review.get("manual_review_status"))
    bucket = as_text(row.get("work_bucket"))
    ticker = clean_ticker(row.get("ticker"))

    if status == "Not reviewed yet":
        if bucket == "Earnings and gap proof":
            return f"For {ticker}, find the next earnings date, expected move, and whether to avoid holding through the report."
        if bucket == "Tail-risk stop proof":
            return f"For {ticker}, write the max loss, stop level, and smallest possible starter size."
        if bucket == "Crowding and overlap proof":
            return f"For {ticker}, compare it with its closest peer and set one combined exposure limit."
        if bucket == "Event reaction proof":
            return f"For {ticker}, compare the news time with the 1-day and 3-day price/volume reaction."
        return f"For {ticker}, fill the source, timestamp, key numbers, and short evidence summary."
    if status == "Needs more proof":
        return "Fill the missing fields before asking the system to move this name forward."
    if status == "Ready for watch-only review":
        return "Put it on the study list, then run separate risk and options checks before any paper idea."
    if status == "Proof accepted; still blocked":
        return "Work the next remaining blocker before this can move forward."
    if status == "Proof failed; keep blocked":
        return "Keep blocked or remove it from the queue."
    if status == "Remove from queue review":
        return "Archive it unless a new source creates a fresh reason to look again."
    return "Write a clear Pass, Fail, Keep blocked, Move to study list, or Remove decision."


def human_do_not_do(status: str) -> str:
    text = as_text(status)
    if text == "Ready for watch-only review":
        return "Do not paper trade yet. Do not buy calls. Do not buy puts."
    if text == "Remove from queue review":
        return "Do not force a trade idea from this."
    return "Do not paper trade. Do not buy calls. Do not buy puts. Do not count it as a good signal yet."


def load_review_base() -> pd.DataFrame:
    template = read_csv_safe(IN_TEMPLATE)
    tasks = read_csv_safe(IN_TASKS)
    promotion = read_csv_safe(IN_PROMOTION)

    if template.empty or "ticker" not in template.columns:
        return pd.DataFrame()

    base = template.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)
    for col in ["work_bucket", "proof_to_collect"] + MANUAL_COLS:
        if col not in base.columns:
            base[col] = ""

    for df, cols in [
        (tasks, [
            "ticker", "risk_snapshot", "option_gate", "source_headline",
            "exact_next_step", "still_forbidden",
        ]),
        (promotion, [
            "ticker", "promotion_status", "current_permission", "status_reason",
            "risk_level", "earnings", "correlation", "liquidity",
            "daily_cvar_95_pct", "estimated_tca_bps",
        ]),
    ]:
        if df.empty or "ticker" not in df.columns:
            continue
        work = df.copy()
        work["ticker"] = work["ticker"].apply(clean_ticker)
        keep = [c for c in cols if c in work.columns]
        base = base.merge(work[keep].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_source"))

    return base


def build_review_gate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = load_review_base()
    if base.empty:
        state = {
            "date": today_str(),
            "status": "NO_MANUAL_PROOF_TEMPLATE",
            "total_rows": 0,
            "reviewed_rows": 0,
            "ready_for_watch_only_review_count": 0,
            "paper_sizing_allowed_now_count": 0,
            "options_allowed_now_count": 0,
            "plain_english": "No manual proof template was found. Run Step191 first.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), state

    rows = []
    missing_rows = []
    for _, row in base.iterrows():
        review = final_review(row)
        miss = missing_fields(row) if has_any_manual(row) else []
        if miss:
            for field in miss:
                missing_rows.append({
                    "ticker": row.get("ticker"),
                    "work_bucket": row.get("work_bucket"),
                    "missing_field": field,
                    "why_it_matters": "The system cannot promote a ticker from a vague note. It needs source, timestamp, numbers, summary, and review decision.",
                    "research_only": True,
                })

        rows.append({
            "ticker": row.get("ticker"),
            "work_bucket": row.get("work_bucket"),
            "proof_to_collect": row.get("proof_to_collect"),
            "manual_review_status": review["manual_review_status"],
            "plain_status": human_status(review["manual_review_status"]),
            "can_i_touch_it": human_touch_permission(review["manual_review_status"]),
            "why_in_plain_english": human_why(row, review),
            "next_step_plain": human_next_step(row, review),
            "what_not_to_do": human_do_not_do(review["manual_review_status"]),
            "review_reason": review["review_reason"],
            "remaining_blockers": review["remaining_blockers"],
            "final_permission": review["final_permission"],
            "plain_next_step": review["plain_next_step"],
            "source_name": row.get("source_name", ""),
            "source_url_or_file": row.get("source_url_or_file", ""),
            "source_date_or_timestamp": row.get("source_date_or_timestamp", ""),
            "key_numbers": row.get("key_numbers", ""),
            "evidence_summary": compact(row.get("evidence_summary"), 260),
            "pass_fail_review": row.get("pass_fail_review", ""),
            "next_gate_request": row.get("next_gate_request", ""),
            "risk_snapshot": row.get("risk_snapshot", ""),
            "option_gate": row.get("option_gate", ""),
            "source_headline": row.get("source_headline", ""),
            "source_files": "manual_proof_input_template / proof_workbench_tasks / promotion_gate",
            "research_only": True,
        })

    gate = pd.DataFrame(rows)
    status_order = {
        "Ready for watch-only review": 0,
        "Proof accepted; still blocked": 1,
        "Needs reviewer decision": 2,
        "Needs more proof": 3,
        "Proof accepted; keep blocked": 4,
        "Proof failed; keep blocked": 5,
        "Remove from queue review": 6,
        "Not reviewed yet": 7,
    }
    if not gate.empty:
        gate["_status_order"] = gate["manual_review_status"].map(status_order).fillna(9)
        gate = gate.sort_values(["_status_order", "ticker"]).drop(columns=["_status_order"]).reset_index(drop=True)

    watch = gate[gate["manual_review_status"].eq("Ready for watch-only review")].copy() if not gate.empty else pd.DataFrame()
    if not watch.empty:
        watch = watch[[
            "ticker", "work_bucket", "plain_status", "can_i_touch_it",
            "why_in_plain_english", "next_step_plain", "what_not_to_do",
            "source_name", "source_date_or_timestamp",
            "key_numbers", "evidence_summary", "risk_snapshot", "option_gate",
        ]].copy()

    missing = pd.DataFrame(missing_rows)
    if missing.empty:
        missing = pd.DataFrame(columns=[
            "ticker", "work_bucket", "missing_field", "why_it_matters", "research_only",
        ])
    reviewed_rows = int(base.apply(has_any_manual, axis=1).sum())
    ready_count = int(gate["manual_review_status"].eq("Ready for watch-only review").sum()) if not gate.empty else 0
    accepted_still_blocked = int(gate["manual_review_status"].eq("Proof accepted; still blocked").sum()) if not gate.empty else 0
    needs_more = int(gate["manual_review_status"].isin(["Needs more proof", "Needs reviewer decision"]).sum()) if not gate.empty else 0
    not_reviewed = int(gate["manual_review_status"].eq("Not reviewed yet").sum()) if not gate.empty else 0

    state = {
        "date": today_str(),
        "status": "MANUAL_PROOF_REVIEW_ACTIVE",
        "total_rows": int(len(gate)),
        "reviewed_rows": reviewed_rows,
        "ready_for_watch_only_review_count": ready_count,
        "accepted_but_still_blocked_count": accepted_still_blocked,
        "needs_more_or_decision_count": needs_more,
        "not_reviewed_count": not_reviewed,
        "paper_sizing_allowed_now_count": 0,
        "options_allowed_now_count": 0,
        "plain_english": "This page answers one question: can this name move forward? If evidence is blank, the answer is no. Even when a name moves forward, it is only for study, not for paper trading or options.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return gate, watch, missing, state


def write_report(gate: pd.DataFrame, watch: pd.DataFrame, missing: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Status: **{state['status']}**",
            f"- Template rows: **{state['total_rows']}**",
            f"- Reviewed rows: **{state['reviewed_rows']}**",
            f"- Ready for watch-only review: **{state['ready_for_watch_only_review_count']}**",
            f"- Accepted but still blocked: **{state.get('accepted_but_still_blocked_count', 0)}**",
            f"- Needs more proof or reviewer decision: **{state.get('needs_more_or_decision_count', 0)}**",
            f"- Not reviewed: **{state.get('not_reviewed_count', 0)}**",
            f"- Paper sizing allowed now: **{state['paper_sizing_allowed_now_count']}**",
            f"- Options allowed now: **{state['options_allowed_now_count']}**",
            "",
            state["plain_english"],
        ]),
        "## Manual Proof Review Gate\n\n" + df_to_markdown(gate.head(40)),
        "## Watch-Only Review Queue\n\n" + df_to_markdown(watch.head(40)),
        "## Missing Fields\n\n" + df_to_markdown(missing.head(80)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 192 - Manual Proof Review Gate", sections)


def main() -> None:
    gate, watch, missing, state = build_review_gate()
    gate.to_csv(OUT_GATE, index=False)
    watch.to_csv(OUT_WATCH, index=False)
    missing.to_csv(OUT_MISSING, index=False)
    write_json(OUT_STATE, state)
    write_report(gate, watch, missing, state)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Template rows: {state['total_rows']}")
    print(f"[OK] Reviewed rows: {state['reviewed_rows']}")
    print(f"[OK] Ready for watch-only review: {state['ready_for_watch_only_review_count']}")
    print(f"[OK] Paper sizing allowed now: {state['paper_sizing_allowed_now_count']}")
    print(f"[OK] Options allowed now: {state['options_allowed_now_count']}")


if __name__ == "__main__":
    main()
