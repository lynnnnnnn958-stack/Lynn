#!/usr/bin/env python3
"""
Canyon v9 Step 215 - Proof Closure Tracker.

Research-only. No broker connection. No live orders.

Step214 lets a human fill and safely apply proof. Step215 tracks whether that
proof actually closes the loop: intake -> quality gate -> verified source proof
-> proof-to-acceptance bridge -> Step204 evidence acceptance.

Outputs:
  quant_fund_proof_closure_state.json
  quant_fund_proof_closure_ticker_status.csv
  quant_fund_proof_closure_stage_counts.csv
  quant_fund_proof_closure_next_actions.csv
  quant_fund_proof_closure_unblock_candidates.csv
  quant_fund_proof_closure_report.md
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "quant_fund_proof_closure_state.json"
OUT_TICKERS = ROOT / "quant_fund_proof_closure_ticker_status.csv"
OUT_STAGE_COUNTS = ROOT / "quant_fund_proof_closure_stage_counts.csv"
OUT_NEXT_ACTIONS = ROOT / "quant_fund_proof_closure_next_actions.csv"
OUT_UNBLOCK = ROOT / "quant_fund_proof_closure_unblock_candidates.csv"
OUT_REPORT = ROOT / "quant_fund_proof_closure_report.md"


TICKER_COLUMNS = [
    "ticker",
    "closure_state",
    "plain_status",
    "next_action",
    "where_to_go",
    "proof_rows",
    "missing_proof_rows",
    "quality_ready_rows",
    "intake_apply_requests",
    "intake_applied_rows",
    "verified_source_rows",
    "bridge_patch_rows",
    "bridge_conflicts",
    "step204_accepted_rows",
    "first_question",
    "first_source",
    "why_this_matters",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

STAGE_COLUMNS = [
    "stage_order",
    "stage_name",
    "row_count",
    "plain_meaning",
    "next_if_zero",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

ACTION_COLUMNS = [
    "action_rank",
    "ticker",
    "action",
    "why",
    "page_or_file",
    "done_when",
    "do_not_do",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

UNBLOCK_COLUMNS = [
    "ticker",
    "unblock_state",
    "what_would_unlock",
    "remaining_blocker",
    "proof_progress_score",
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


def short(value: Any, limit: int = 300) -> str:
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


def count_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, int]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""]
    return work.groupby(ticker_col).size().astype(int).to_dict()


def rows_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if df.empty or ticker_col not in df.columns:
        return out
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""]
    for ticker, grp in work.groupby(ticker_col, sort=False):
        out[ticker] = grp.iloc[0]
    return out


def accepted_step204_counts() -> dict[str, int]:
    df = read_csv_safe(ROOT / "pm_review_evidence_acceptance_input.csv")
    if df.empty or "ticker" not in df.columns or "acceptance_status" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    accepted = work[work["acceptance_status"].astype(str).str.lower().eq("accept")]
    return count_by_ticker(accepted)


def build_ticker_status() -> pd.DataFrame:
    fill_plan = read_csv_safe(ROOT / "quant_fund_proof_fill_ticker_plan.csv")
    quality = read_csv_safe(ROOT / "quant_fund_proof_quality_gate.csv")
    proof_status = read_csv_safe(ROOT / "pm_evidence_source_proof_status.csv")
    ready = read_csv_safe(ROOT / "pm_evidence_source_proof_ready_for_acceptance.csv")
    bridge = read_csv_safe(ROOT / "pm_evidence_proof_acceptance_bridge.csv")
    patch = read_csv_safe(ROOT / "pm_evidence_proof_acceptance_patch.csv")
    conflicts = read_csv_safe(ROOT / "pm_evidence_proof_acceptance_conflicts.csv")
    intake_preview = read_csv_safe(ROOT / "quant_fund_proof_intake_apply_preview.csv")
    applied = read_csv_safe(ROOT / "quant_fund_proof_intake_applied_rows.csv")

    tickers: set[str] = set()
    for df in [fill_plan, quality, proof_status, ready, bridge, patch, conflicts, intake_preview, applied]:
        if not df.empty and "ticker" in df.columns:
            tickers.update(clean_ticker(x) for x in df["ticker"].tolist() if clean_ticker(x))

    fill_first = rows_by_ticker(fill_plan)
    proof_status_first = rows_by_ticker(proof_status)
    quality_counts = count_by_ticker(quality)
    quality_ready = count_by_ticker(quality[quality["can_send_to_acceptance_bridge"].astype(str).str.lower().eq("yes")]) if not quality.empty and "can_send_to_acceptance_bridge" in quality.columns else {}
    ready_counts = count_by_ticker(ready)
    bridge_counts = count_by_ticker(bridge)
    patch_counts = count_by_ticker(patch)
    conflict_counts = count_by_ticker(conflicts)
    intake_apply = count_by_ticker(intake_preview[intake_preview["apply_decision"].astype(str).str.upper().eq("APPLY")]) if not intake_preview.empty and "apply_decision" in intake_preview.columns else {}
    intake_applied = count_by_ticker(applied)
    accepted_counts = accepted_step204_counts()

    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        frow = fill_first.get(ticker, pd.Series(dtype=object))
        psrow = proof_status_first.get(ticker, pd.Series(dtype=object))
        proof_rows = int(quality_counts.get(ticker, 0))
        ready_rows = int(quality_ready.get(ticker, 0))
        verified_source_rows = int(ready_counts.get(ticker, 0))
        patch_rows = int(patch_counts.get(ticker, 0))
        bridge_rows = int(bridge_counts.get(ticker, 0))
        conflicts_n = int(conflict_counts.get(ticker, 0))
        apply_req = int(intake_apply.get(ticker, 0))
        applied_n = int(intake_applied.get(ticker, 0))
        accepted_n = int(accepted_counts.get(ticker, 0))
        needs_proof = int(pd.to_numeric(pd.Series([psrow.get("needs_proof_count")]), errors="coerce").fillna(0).iloc[0]) if not psrow.empty else proof_rows

        if accepted_n > 0 and needs_proof == 0:
            closure_state = "Evidence accepted"
            next_action = "Return to the final permission and promotion gates."
            where = "Home -> Final permission / promotion gate"
        elif conflicts_n > 0:
            closure_state = "Bridge conflict"
            next_action = "Resolve the bridge conflict before touching Step204."
            where = "System -> Proof-to-acceptance bridge conflicts"
        elif patch_rows > 0:
            closure_state = "Copy acceptance patch"
            next_action = "Copy the manual patch into Step204 acceptance input, then rerun Step204."
            where = "System -> Proof-to-acceptance bridge patch"
        elif bridge_rows > 0 or verified_source_rows > 0 or ready_rows > 0:
            closure_state = "Ready for acceptance bridge"
            next_action = "Rerun Steps 206 and 207, then inspect the manual Step204 patch."
            where = "System -> Proof-to-acceptance bridge"
        elif applied_n > 0:
            closure_state = "Applied, rerun proof checks"
            next_action = "Rerun Steps 206, 207, 204, 212, 213, 214, and 215."
            where = "Home -> Run Daily System Now"
        elif apply_req > 0:
            closure_state = "Fix intake apply row"
            next_action = "The APPLY row is missing required fields. Fix it in the user entry sheet."
            where = "quant_fund_proof_intake_user_entry.csv"
        elif needs_proof > 0 or proof_rows > 0:
            closure_state = "Fill proof first"
            next_action = "Fill the user entry sheet. Leave unfinished rows as WAIT; mark only finished rows APPLY."
            where = "quant_fund_proof_intake_user_entry.csv"
        else:
            closure_state = "No open proof blocker"
            next_action = "Monitor; no proof closure work is visible for this ticker."
            where = "Home"

        first_question = as_text(frow.get("first_question"), "") or as_text(psrow.get("first_missing_proof"), "")
        first_source = as_text(frow.get("first_source_to_open"), "")
        why = as_text(frow.get("why_this_ticker_first"), "")
        plain = f"{ticker}: {closure_state}. {next_action}"
        rows.append(guard_flags({
            "ticker": ticker,
            "closure_state": closure_state,
            "plain_status": short(plain, 360),
            "next_action": next_action,
            "where_to_go": where,
            "proof_rows": proof_rows,
            "missing_proof_rows": needs_proof,
            "quality_ready_rows": ready_rows,
            "intake_apply_requests": apply_req,
            "intake_applied_rows": applied_n,
            "verified_source_rows": verified_source_rows,
            "bridge_patch_rows": patch_rows,
            "bridge_conflicts": conflicts_n,
            "step204_accepted_rows": accepted_n,
            "first_question": short(first_question, 260),
            "first_source": short(first_source, 300),
            "why_this_matters": short(why, 320),
        }))

    if not rows:
        return pd.DataFrame(columns=TICKER_COLUMNS)

    order = {
        "Bridge conflict": 0,
        "Copy acceptance patch": 1,
        "Ready for acceptance bridge": 2,
        "Applied, rerun proof checks": 3,
        "Fix intake apply row": 4,
        "Fill proof first": 5,
        "Evidence accepted": 6,
        "No open proof blocker": 7,
    }
    out = pd.DataFrame(rows, columns=TICKER_COLUMNS)
    out["_order"] = out["closure_state"].map(order).fillna(9)
    out = out.sort_values(["_order", "missing_proof_rows", "proof_rows", "ticker"], ascending=[True, False, False, True]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_stage_counts(tickers: pd.DataFrame) -> pd.DataFrame:
    intake_state = read_json_safe(ROOT / "quant_fund_proof_intake_state.json", {})
    quality = read_csv_safe(ROOT / "quant_fund_proof_quality_gate.csv")
    ready = read_csv_safe(ROOT / "pm_evidence_source_proof_ready_for_acceptance.csv")
    patch = read_csv_safe(ROOT / "pm_evidence_proof_acceptance_patch.csv")
    conflicts = read_csv_safe(ROOT / "pm_evidence_proof_acceptance_conflicts.csv")
    step204_state = read_json_safe(ROOT / "pm_review_evidence_acceptance_state.json", {})

    quality_ready = int((quality.get("can_send_to_acceptance_bridge", pd.Series(dtype=str)).astype(str).str.lower() == "yes").sum()) if not quality.empty else 0
    rows = [
        (1, "User entry sheet", int(intake_state.get("user_entry_rows", 0)), "Rows available for human source entry.", "Run Step213 first."),
        (2, "Apply requests", int(intake_state.get("apply_request_count", 0)), "Rows the user marked APPLY.", "Fill the user entry sheet and mark finished rows APPLY."),
        (3, "Applied to proof input", int(intake_state.get("applied_count", 0)), "Rows written back to the proof input file.", "Fix APPLY rows or leave them WAIT."),
        (4, "Quality-ready proof", quality_ready, "Rows that passed Step212 quality checks.", "Fill source, observed value, reviewer, and review date."),
        (5, "Verified source proof", int(len(ready)), "Rows ready for Step207 bridge.", "Rerun Steps 206 and 207 after applying proof."),
        (6, "Manual Step204 patch", int(len(patch)), "Patch rows ready to copy into Step204.", "Resolve source proof or bridge conflicts."),
        (7, "Bridge conflicts", int(len(conflicts)), "Rows that need bridge review before Step204.", "Resolve conflicts before accepting evidence."),
        (8, "Accepted evidence", int(step204_state.get("accepted_count", 0)), "Evidence rows accepted in Step204.", "Copy patch rows manually, then rerun Step204."),
    ]
    return pd.DataFrame([guard_flags({
        "stage_order": order,
        "stage_name": name,
        "row_count": count,
        "plain_meaning": meaning,
        "next_if_zero": next_if_zero,
    }) for order, name, count, meaning, next_if_zero in rows], columns=STAGE_COLUMNS)


def build_next_actions(tickers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if tickers.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    for rank, (_, row) in enumerate(tickers.head(12).iterrows(), start=1):
        rows.append(guard_flags({
            "action_rank": rank,
            "ticker": row.get("ticker", ""),
            "action": row.get("next_action", ""),
            "why": row.get("plain_status", ""),
            "page_or_file": row.get("where_to_go", ""),
            "done_when": done_when(as_text(row.get("closure_state"), "")),
            "do_not_do": "Do not add size, calls, puts, or final approval just because a proof row exists.",
        }))
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def done_when(state: str) -> str:
    if state == "Fill proof first":
        return "The row is filled in the intake sheet and marked APPLY."
    if state == "Fix intake apply row":
        return "The APPLY row passes validation and Step214 applies it."
    if state == "Applied, rerun proof checks":
        return "Step212 says the proof row can go to the acceptance bridge."
    if state == "Ready for acceptance bridge":
        return "Step207 creates either a manual Step204 patch or a clear conflict."
    if state == "Copy acceptance patch":
        return "Step204 shows accepted evidence after manual copy."
    if state == "Bridge conflict":
        return "Conflict file is empty after rerun."
    return "No immediate proof action remains."


def build_unblock_candidates(tickers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if tickers.empty:
        return pd.DataFrame(columns=UNBLOCK_COLUMNS)
    for _, row in tickers.iterrows():
        proof_rows = float(row.get("proof_rows", 0) or 0)
        missing = float(row.get("missing_proof_rows", 0) or 0)
        ready = float(row.get("quality_ready_rows", 0) or 0)
        applied = float(row.get("intake_applied_rows", 0) or 0)
        patch = float(row.get("bridge_patch_rows", 0) or 0)
        accepted = float(row.get("step204_accepted_rows", 0) or 0)
        progress = 0.0
        if proof_rows > 0:
            progress += max(0.0, min(35.0, (proof_rows - missing) / proof_rows * 35.0))
        progress += min(15.0, applied * 5.0)
        progress += min(20.0, ready * 10.0)
        progress += min(20.0, patch * 10.0)
        progress += min(10.0, accepted * 2.0)
        state = as_text(row.get("closure_state"), "")
        rows.append(guard_flags({
            "ticker": row.get("ticker", ""),
            "unblock_state": "Closest to unlock" if progress >= 50 else "Still blocked",
            "what_would_unlock": row.get("next_action", ""),
            "remaining_blocker": row.get("closure_state", ""),
            "proof_progress_score": round(progress, 1),
        }))
    out = pd.DataFrame(rows, columns=UNBLOCK_COLUMNS)
    return out.sort_values(["proof_progress_score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def build_state(tickers: pd.DataFrame, counts: pd.DataFrame, actions: pd.DataFrame, unblock: pd.DataFrame) -> dict[str, Any]:
    closure_counts = tickers["closure_state"].value_counts().to_dict() if not tickers.empty else {}
    first = actions.iloc[0].to_dict() if not actions.empty else {}
    fill_first = int(closure_counts.get("Fill proof first", 0))
    patch_ready = int(closure_counts.get("Copy acceptance patch", 0))
    bridge_conflict = int(closure_counts.get("Bridge conflict", 0))
    accepted = int(closure_counts.get("Evidence accepted", 0))
    if patch_ready:
        answer = f"Proof closure is active. {patch_ready} ticker(s) have a manual Step204 patch ready to copy."
    elif bridge_conflict:
        answer = f"Proof closure is active. {bridge_conflict} ticker(s) have bridge conflicts to resolve first."
    elif fill_first:
        answer = f"Proof closure is active. {fill_first} ticker(s) still need proof filled before anything can be accepted."
    else:
        answer = f"Proof closure is active. {accepted} ticker(s) have accepted evidence; monitor final permission gates."
    return {
        "date": today_str(),
        "status": "Active",
        "ticker_count": int(len(tickers)),
        "fill_first_count": fill_first,
        "patch_ready_count": patch_ready,
        "bridge_conflict_count": bridge_conflict,
        "accepted_evidence_ticker_count": accepted,
        "top_action_ticker": as_text(first.get("ticker"), ""),
        "top_action": as_text(first.get("action"), ""),
        "plain_answer": answer,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    tickers = build_ticker_status()
    counts = build_stage_counts(tickers)
    actions = build_next_actions(tickers)
    unblock = build_unblock_candidates(tickers)
    state = build_state(tickers, counts, actions, unblock)

    tickers.to_csv(OUT_TICKERS, index=False)
    counts.to_csv(OUT_STAGE_COUNTS, index=False)
    actions.to_csv(OUT_NEXT_ACTIONS, index=False)
    unblock.to_csv(OUT_UNBLOCK, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Next Actions\n\n" + df_to_markdown(actions.head(20)),
        "## Ticker Closure Status\n\n" + df_to_markdown(tickers.head(80)),
        "## Stage Counts\n\n" + df_to_markdown(counts),
        "## Unblock Candidates\n\n" + df_to_markdown(unblock.head(80)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Proof Closure Tracker", sections)
    print(
        "Step215 complete: "
        f"{len(tickers)} tickers, {state['fill_first_count']} still need proof, "
        f"{state['patch_ready_count']} patch-ready."
    )


if __name__ == "__main__":
    main()
