#!/usr/bin/env python3
"""
Canyon v9 Step 211 - Proof Collection Workbench.

Research-only. No broker connection. No live orders.

The current flow is proof-first. This step turns proof blockers into a plain
workbench: what question to answer, what source is acceptable, which fields
must be filled, where the editable row lives, and what to rerun after proof is
verified. It does not fetch outside data and does not approve evidence.

Outputs:
  quant_fund_proof_workbench_state.json
  quant_fund_proof_task_cards.csv
  quant_fund_proof_ticker_queue.csv
  quant_fund_proof_field_guide.csv
  quant_fund_proof_user_instructions.csv
  quant_fund_proof_workbench_quality_check.csv
  quant_fund_proof_workbench_report.md
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


OUT_STATE = ROOT / "quant_fund_proof_workbench_state.json"
OUT_TASKS = ROOT / "quant_fund_proof_task_cards.csv"
OUT_QUEUE = ROOT / "quant_fund_proof_ticker_queue.csv"
OUT_GUIDE = ROOT / "quant_fund_proof_field_guide.csv"
OUT_INSTRUCTIONS = ROOT / "quant_fund_proof_user_instructions.csv"
OUT_QA = ROOT / "quant_fund_proof_workbench_quality_check.csv"
OUT_REPORT = ROOT / "quant_fund_proof_workbench_report.md"


TASK_COLUMNS = [
    "task_rank",
    "ticker",
    "proof_type",
    "question_to_answer",
    "why_this_matters",
    "acceptable_source",
    "fields_to_fill",
    "editable_file",
    "proof_id",
    "suggested_value",
    "after_you_fill",
    "current_ticker_answer",
    "do_not_do",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

QUEUE_COLUMNS = [
    "queue_rank",
    "ticker",
    "open_proof_tasks",
    "first_proof_type",
    "first_question",
    "first_source",
    "first_fields_to_fill",
    "why_this_ticker_first",
    "where_to_click",
    "after_done",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

GUIDE_COLUMNS = [
    "proof_type",
    "plain_goal",
    "good_sources",
    "must_fill",
    "what_counts_as_done",
    "what_does_not_count",
    "next_step_after_verified",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

INSTRUCTION_COLUMNS = [
    "step",
    "instruction",
    "why",
    "done_when",
    "do_not_do",
    "file_or_page",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

QA_COLUMNS = [
    "check",
    "status",
    "bad_rows",
    "what_it_checked",
    "fix_hint",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


RAW_TOKENS = ["DATA_GAP", "SIZE_DOWN", "NO_GO", "NEEDS_REVIEW", "PENDING_MANUAL_CHECKS"]


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


def short(value: Any, limit: int = 280) -> str:
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


def plain(value: Any) -> str:
    text = as_text(value, "")
    replacements = {
        "DATA_GAP": "missing data",
        "SIZE_DOWN": "risk says smaller size",
        "NO_GO": "not allowed yet",
        "NEEDS_REVIEW": "needs human review",
        "PENDING_MANUAL_CHECKS": "needs manual checks",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    return " ".join(text.split())


def normalize_question(value: Any) -> str:
    return " ".join(as_text(value, "").lower().replace("?", "").split())


def proof_type_from(row: pd.Series) -> str:
    group = as_text(row.get("field_group"), "")
    if group:
        return group
    question = as_text(row.get("blocker"), "") + " " + as_text(row.get("required_question"), "")
    q = question.lower()
    if "spread" in q or "bid/ask" in q or "liquidity" in q or "trading" in q:
        return "Trading cost"
    if "headline" in q or "news" in q or "timestamp" in q or "price/volume reaction" in q:
        return "News proof"
    if "event" in q or "options market" in q or "expected move" in q:
        return "Event risk"
    return "General proof"


def required_fields(missing_proof: Any, proof_type: str) -> str:
    text = as_text(missing_proof, "")
    if text:
        return plain(text)
    if proof_type == "News proof":
        return "source name; observed value; reviewer; review date; source or observation time; price reaction checked; volume reaction checked"
    if proof_type in {"Trading cost", "Event risk"}:
        return "source name; observed value; reviewer; review date"
    return "source name; observed value; reviewer; review date"


def load_maps() -> tuple[dict[tuple[str, str], pd.Series], dict[str, pd.Series]]:
    proof_input = read_csv_safe(ROOT / "pm_evidence_source_proof_input.csv")
    ticker_cards = read_csv_safe(ROOT / "quant_fund_ticker_flow_cards.csv")
    proof_map: dict[tuple[str, str], pd.Series] = {}
    card_map: dict[str, pd.Series] = {}

    if not proof_input.empty:
        for _, row in proof_input.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            question = normalize_question(row.get("required_question"))
            if ticker and question:
                proof_map[(ticker, question)] = row

    if not ticker_cards.empty:
        for _, row in ticker_cards.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            if ticker and ticker not in card_map:
                card_map[ticker] = row
    return proof_map, card_map


def build_task_rows() -> pd.DataFrame:
    blockers = read_csv_safe(ROOT / "quant_fund_flow_blocker_queue.csv")
    gaps = read_csv_safe(ROOT / "pm_evidence_source_proof_gap_queue.csv")
    proof_map, card_map = load_maps()

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    source_frames: list[tuple[pd.DataFrame, str]] = []
    if not blockers.empty:
        proof_blockers = blockers[blockers.get("blocker_type", "").astype(str).str.lower().eq("outside proof")].copy()
        source_frames.append((proof_blockers, "quant_fund_flow_blocker_queue.csv"))
    if not gaps.empty:
        source_frames.append((gaps, "pm_evidence_source_proof_gap_queue.csv"))

    rank = 1
    for frame, source_name in source_frames:
        if frame.empty:
            continue
        for _, raw in frame.iterrows():
            ticker = clean_ticker(raw.get("ticker"))
            question = as_text(raw.get("blocker"), "") or as_text(raw.get("required_question"), "")
            if not ticker or not question:
                continue
            key = (ticker, normalize_question(question))
            if key in seen:
                continue
            seen.add(key)

            gap_match = pd.Series(dtype=object)
            if not gaps.empty:
                gwork = gaps.copy()
                gwork["ticker"] = gwork["ticker"].apply(clean_ticker)
                gwork["_q"] = gwork["required_question"].apply(normalize_question)
                match = gwork[(gwork["ticker"] == ticker) & (gwork["_q"] == key[1])]
                if not match.empty:
                    gap_match = match.iloc[0]

            proof_row = proof_map.get(key)
            proof_type = proof_type_from(gap_match if not gap_match.empty else raw)
            missing = as_text(gap_match.get("missing_proof"), "") if not gap_match.empty else ""
            preferred_source = (
                as_text(gap_match.get("preferred_source"), "")
                or as_text(raw.get("source_files"), "")
                or "Use the most direct source available, then record it."
            )
            acceptable = (
                as_text(gap_match.get("acceptable_proof"), "")
                or "Record the source, observed value, reviewer, and date."
            )
            fields = required_fields(missing, proof_type)
            proof_id = as_text(proof_row.get("proof_id"), "") if proof_row is not None else ""
            suggested = short(proof_row.get("suggested_value"), 220) if proof_row is not None else ""
            card = card_map.get(ticker)
            current_answer = short(card.get("front_answer"), 180) if card is not None else "Ticker is blocked until proof is verified."
            why = (
                "This proof is required before the system can turn a model suggestion into accepted evidence. "
                "Without it, the ticker stays research-only."
            )
            if proof_type == "Trading cost":
                why = "Trading cost proof is required because spread, liquidity, and fill risk can erase a signal."
            elif proof_type == "News proof":
                why = "News proof is required because a headline cannot create size unless the source, timing, and price reaction are verified."
            elif proof_type == "Event risk":
                why = "Event-risk proof is required because earnings or option-implied moves can dominate the normal signal."

            rows.append(guard_flags({
                "task_rank": rank,
                "ticker": ticker,
                "proof_type": proof_type,
                "question_to_answer": short(question, 240),
                "why_this_matters": why,
                "acceptable_source": short(preferred_source + " " + acceptable, 340),
                "fields_to_fill": fields,
                "editable_file": "pm_evidence_source_proof_input.csv",
                "proof_id": proof_id,
                "suggested_value": suggested,
                "after_you_fill": "Set Proof Status to Verified only if the source is real, then rerun Steps 206, 207, 204, 209, and 210.",
                "current_ticker_answer": current_answer,
                "do_not_do": "Do not use model text as proof. Do not add stock, call, put, or size while this proof is missing.",
                "source_files": as_text(raw.get("source_files"), source_name),
            }))
            rank += 1

    tasks = pd.DataFrame(rows, columns=TASK_COLUMNS)
    if tasks.empty:
        return tasks

    type_order = {"Event risk": 0, "Trading cost": 1, "News proof": 2, "General proof": 3}
    tasks["_type_sort"] = tasks["proof_type"].map(type_order).fillna(9)
    tasks = tasks.sort_values(["task_rank", "_type_sort", "ticker"]).drop(columns=["_type_sort"])
    tasks["task_rank"] = range(1, len(tasks) + 1)
    return tasks


def build_ticker_queue(tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    rows = []
    grouped = tasks.groupby("ticker", sort=False)
    rank = 1
    for ticker, group in grouped:
        first = group.iloc[0]
        rows.append(guard_flags({
            "queue_rank": rank,
            "ticker": ticker,
            "open_proof_tasks": int(len(group)),
            "first_proof_type": as_text(first.get("proof_type")),
            "first_question": short(first.get("question_to_answer"), 220),
            "first_source": short(first.get("acceptable_source"), 220),
            "first_fields_to_fill": short(first.get("fields_to_fill"), 180),
            "why_this_ticker_first": short(first.get("why_this_matters"), 240),
            "where_to_click": "Risk -> Source Proof Desk",
            "after_done": "Rerun Steps 206, 207, 204, 209, and 210.",
        }))
        rank += 1
    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)


def build_field_guide() -> pd.DataFrame:
    rows = [
        {
            "proof_type": "Event risk",
            "plain_goal": "Prove the event or option-implied move is real enough to respect.",
            "good_sources": "Earnings calendar, company IR date, options expected-move source, or verified event-risk file.",
            "must_fill": "source name; observed value; reviewer; review date",
            "what_counts_as_done": "A source names the event or expected move and a human records the value and date.",
            "what_does_not_count": "A model sentence saying event risk exists.",
            "next_step_after_verified": "Rerun proof bridge, then check whether PM evidence can be accepted.",
        },
        {
            "proof_type": "Trading cost",
            "plain_goal": "Prove spread, liquidity, or quote timing before paper sizing.",
            "good_sources": "Fresh quote snapshot, Yahoo quote page, local liquidity cache, or better intraday quote source.",
            "must_fill": "source name; observed value; reviewer; review date; observed time if available",
            "what_counts_as_done": "Bid/ask spread, liquidity date, or dollar-volume observation is recorded.",
            "what_does_not_count": "Old proxy spread with no current quote check.",
            "next_step_after_verified": "Rerun execution/risk proof flow before any route review.",
        },
        {
            "proof_type": "News proof",
            "plain_goal": "Prove the headline, timestamp, linked ticker, and price/volume reaction.",
            "good_sources": "Original article, publisher timestamp, source URL, linked-stock map, price and volume after headline.",
            "must_fill": "source name; observed value; reviewer; review date; source or observation time; price reaction checked; volume reaction checked",
            "what_counts_as_done": "The headline source, timing, and reaction are recorded, with direct or read-through link explained.",
            "what_does_not_count": "A headline copied without source time or market reaction.",
            "next_step_after_verified": "Rerun event causal chain and proof acceptance bridge.",
        },
        {
            "proof_type": "General proof",
            "plain_goal": "Turn an uncertain model suggestion into a human-reviewed evidence row.",
            "good_sources": "Direct source file, vendor page, company filing, quote page, or original news article.",
            "must_fill": "source name; observed value; reviewer; review date",
            "what_counts_as_done": "A human can point to the source and explain the observed value.",
            "what_does_not_count": "A guess or unverified model summary.",
            "next_step_after_verified": "Rerun Steps 206, 207, 204, 209, and 210.",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=GUIDE_COLUMNS)


def build_instructions(tasks: pd.DataFrame) -> pd.DataFrame:
    first = tasks.iloc[0].to_dict() if not tasks.empty else {}
    ticker = as_text(first.get("ticker"), "the first ticker")
    proof_type = as_text(first.get("proof_type"), "proof task")
    rows = [
        {
            "step": 1,
            "instruction": f"Start with {ticker}: {proof_type}.",
            "why": "The flow navigator says proof is the first bottleneck.",
            "done_when": "You know the exact question that must be answered.",
            "do_not_do": "Do not jump to calls, puts, or new size.",
            "file_or_page": "Home -> Proof Collection Workbench",
        },
        {
            "step": 2,
            "instruction": "Find a real source for the task.",
            "why": "Accepted evidence needs a source, not a model-generated sentence.",
            "done_when": "You have the source name, source URL or file, observed value, and date.",
            "do_not_do": "Do not use a vague headline screenshot if timing and source are unclear.",
            "file_or_page": "News source, quote page, local proof file, or company/source page",
        },
        {
            "step": 3,
            "instruction": "Fill the matching row in pm_evidence_source_proof_input.csv.",
            "why": "Step206 reads this file and decides whether proof is ready for acceptance.",
            "done_when": "Proof Status is Verified and required fields are filled truthfully.",
            "do_not_do": "Do not mark Verified if source name, observed value, reviewer, or date is blank.",
            "file_or_page": "pm_evidence_source_proof_input.csv",
        },
        {
            "step": 4,
            "instruction": "Rerun Steps 206, 207, 204, 209, and 210.",
            "why": "The source proof must flow into acceptance bridge and then back into the website cards.",
            "done_when": "Ready proof or acceptance patch rows appear, and ticker cards update.",
            "do_not_do": "Do not manually change final route without rerunning the gates.",
            "file_or_page": "Terminal or Run Daily System",
        },
        {
            "step": 5,
            "instruction": "If a patch row appears, manually review before accepting Step204 evidence.",
            "why": "The bridge prepares a patch but does not approve evidence automatically.",
            "done_when": "A human accepts, rejects, or writes a note.",
            "do_not_do": "Do not let proof acceptance unlock live orders. Live orders are disabled.",
            "file_or_page": "Risk -> Proof-to-Acceptance Bridge",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=INSTRUCTION_COLUMNS)


def build_quality(tasks: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    checks = [
        (
            "Every proof task has a question",
            int(tasks["question_to_answer"].astype(str).str.strip().eq("").sum()) if not tasks.empty else 0,
            "A proof card is useless without the exact question.",
            "Fill from blocker or required_question.",
        ),
        (
            "Every proof task has acceptable source guidance",
            int(tasks["acceptable_source"].astype(str).str.strip().eq("").sum()) if not tasks.empty else 0,
            "The user needs to know what source counts.",
            "Fill preferred_source and acceptable_proof.",
        ),
        (
            "Every proof task has fields to fill",
            int(tasks["fields_to_fill"].astype(str).str.strip().eq("").sum()) if not tasks.empty else 0,
            "The user needs the exact missing fields.",
            "Fill from missing_proof or proof type defaults.",
        ),
        (
            "Ticker queue exists",
            0 if not queue.empty else 1,
            "The page needs a ticker order.",
            "Regenerate tasks from blocker and proof queues.",
        ),
        (
            "No obvious raw status tokens in proof cards",
            count_raw_tokens(tasks),
            "Proof cards should read like English.",
            "Add replacements in plain().",
        ),
    ]
    rows = []
    for check, bad, what, hint in checks:
        rows.append(guard_flags({
            "check": check,
            "status": "PASS" if bad == 0 else "REVIEW",
            "bad_rows": int(bad),
            "what_it_checked": what,
            "fix_hint": hint,
        }))
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def count_raw_tokens(tasks: pd.DataFrame) -> int:
    if tasks.empty:
        return 0
    cols = ["question_to_answer", "why_this_matters", "acceptable_source", "fields_to_fill", "after_you_fill", "do_not_do"]
    count = 0
    for _, row in tasks.iterrows():
        text = " ".join(as_text(row.get(c), "") for c in cols)
        if any(tok in text for tok in RAW_TOKENS):
            count += 1
    return count


def build_state(tasks: pd.DataFrame, queue: pd.DataFrame, qa: pd.DataFrame) -> dict[str, Any]:
    first = tasks.iloc[0].to_dict() if not tasks.empty else {}
    review_count = int((qa["status"] != "PASS").sum()) if not qa.empty and "status" in qa.columns else 0
    first_ticker = as_text(first.get("ticker"), "")
    first_question = as_text(first.get("question_to_answer"), "No proof task is open.")
    return {
        "date": today_str(),
        "status": "Active",
        "proof_task_count": int(len(tasks)),
        "ticker_queue_count": int(len(queue)),
        "quality_review_count": review_count,
        "first_ticker": first_ticker,
        "first_proof_type": as_text(first.get("proof_type"), ""),
        "first_question": first_question,
        "plain_answer": (
            f"Proof workbench is active. Start with {first_ticker}: {first_question} "
            f"{len(tasks)} proof tasks are open."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    tasks = build_task_rows()
    queue = build_ticker_queue(tasks)
    guide = build_field_guide()
    instructions = build_instructions(tasks)
    qa = build_quality(tasks, queue)
    state = build_state(tasks, queue, qa)

    tasks.to_csv(OUT_TASKS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    guide.to_csv(OUT_GUIDE, index=False)
    instructions.to_csv(OUT_INSTRUCTIONS, index=False)
    qa.to_csv(OUT_QA, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Proof Task Cards\n\n" + df_to_markdown(tasks.head(120)),
        "## Ticker Queue\n\n" + df_to_markdown(queue.head(80)),
        "## Field Guide\n\n" + df_to_markdown(guide),
        "## User Instructions\n\n" + df_to_markdown(instructions),
        "## Quality Check\n\n" + df_to_markdown(qa),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Proof Collection Workbench", sections)
    print(
        "Step211 complete: "
        f"{len(tasks)} proof tasks, {len(queue)} ticker queue rows, "
        f"{len(guide)} guide rows, {len(qa)} QA checks."
    )


if __name__ == "__main__":
    main()
