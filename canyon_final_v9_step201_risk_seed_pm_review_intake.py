#!/usr/bin/env python3
"""
Canyon v9 Step 201 - Risk Seed PM Review Intake.

Research-only. No broker connection. No live orders.

Step200 ranks review-only risk seeds. Step201 creates a durable manual review
template, preserves human edits, validates the evidence, and explains what is
still missing before a ticker can even be considered for tiny paper review.

Outputs:
  risk_seed_pm_review_state.json
  risk_seed_pm_review_input.csv
  risk_seed_pm_review_status.csv
  risk_seed_pm_review_todo.csv
  risk_seed_pm_review_audit.csv
  risk_seed_pm_review_report.md
"""
from __future__ import annotations

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


OUT_STATE = ROOT / "risk_seed_pm_review_state.json"
OUT_INPUT = ROOT / "risk_seed_pm_review_input.csv"
OUT_STATUS = ROOT / "risk_seed_pm_review_status.csv"
OUT_TODO = ROOT / "risk_seed_pm_review_todo.csv"
OUT_AUDIT = ROOT / "risk_seed_pm_review_audit.csv"
OUT_REPORT = ROOT / "risk_seed_pm_review_report.md"


INPUT_COLUMNS = [
    "ticker",
    "approval_lane",
    "risk_level",
    "system_seed_cap_pct",
    "system_stop_pct",
    "review_status",
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
    "news_proof_note",
    "execution_proof_note",
    "option_route_requested",
    "decision_note",
    "last_updated",
]

STATUS_HELP = (
    "Use one of: NEEDS_REVIEW, WATCH_ONLY, REJECT, APPROVE_TINY_PAPER_REVIEW. "
    "Approval still does not create a trade; it only lets the next gate consider a tiny paper review."
)


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def is_filled(value: Any) -> bool:
    return bool(as_text(value, ""))


def normal_status(value: Any) -> str:
    text = as_text(value, "NEEDS_REVIEW").upper().replace(" ", "_")
    allowed = {"NEEDS_REVIEW", "WATCH_ONLY", "REJECT", "APPROVE_TINY_PAPER_REVIEW"}
    return text if text in allowed else "NEEDS_REVIEW"


def one_by_ticker(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "ticker" not in df.columns:
        return {}
    work = df.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    return {ticker: grp.iloc[0] for ticker, grp in work.groupby("ticker", sort=False)}


def blocker_types_by_ticker(blockers: pd.DataFrame) -> dict[str, set[str]]:
    if blockers.empty or "ticker" not in blockers.columns:
        return {}
    work = blockers.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    out: dict[str, set[str]] = {}
    for ticker, grp in work.groupby("ticker", sort=False):
        out[ticker] = {as_text(x, "") for x in grp.get("blocker_type", pd.Series(dtype=str)) if as_text(x, "")}
    return out


def build_input_template(rank: pd.DataFrame, existing: pd.DataFrame, carry_forward: pd.DataFrame | None = None) -> pd.DataFrame:
    existing_map = one_by_ticker(existing)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _, src in rank.iterrows():
        ticker = clean_ticker(src.get("ticker"))
        if not ticker:
            continue
        seen.add(ticker)
        old = existing_map.get(ticker)
        row = {col: "" for col in INPUT_COLUMNS}
        row.update({
            "ticker": ticker,
            "approval_lane": as_text(src.get("approval_lane"), ""),
            "risk_level": as_text(src.get("risk_level"), ""),
            "system_seed_cap_pct": safe_float(src.get("starter_cap_if_approved_pct"), 0.0),
            "system_stop_pct": safe_float(src.get("paper_stop_if_ever_tested_pct"), np.nan),
            "review_status": "NEEDS_REVIEW",
            "option_route_requested": "NO",
        })
        if old is not None:
            for col in INPUT_COLUMNS:
                if col in old.index and is_filled(old.get(col)):
                    row[col] = old.get(col)
        rows.append(row)

    carry = carry_forward if carry_forward is not None else pd.DataFrame()
    if not carry.empty and "ticker" in carry.columns:
        carry = carry.copy()
        carry["ticker"] = carry["ticker"].apply(clean_ticker)
        carry = carry[carry["ticker"] != ""].drop_duplicates("ticker", keep="first")
        for _, src in carry.iterrows():
            ticker = clean_ticker(src.get("ticker"))
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            old = existing_map.get(ticker)
            row = {col: "" for col in INPUT_COLUMNS}
            row.update({
                "ticker": ticker,
                "approval_lane": as_text(src.get("approval_lane"), "Carry-forward review row"),
                "risk_level": as_text(src.get("risk_level"), "Unknown"),
                "system_seed_cap_pct": safe_float(src.get("system_seed_cap_pct"), safe_float(src.get("max_tiny_paper_review_pct"), 0.0)),
                "system_stop_pct": safe_float(src.get("system_stop_pct"), np.nan),
                "review_status": "NEEDS_REVIEW",
                "option_route_requested": "NO",
            })
            if old is not None:
                for col in INPUT_COLUMNS:
                    if col in old.index and is_filled(old.get(col)):
                        row[col] = old.get(col)
            rows.append(row)

    out = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    if not out.empty:
        out["ticker"] = out["ticker"].apply(clean_ticker)
        out = out.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return out


def missing_for_blockers(row: pd.Series, blocker_types: set[str]) -> list[str]:
    missing: list[str] = []

    status = normal_status(row.get("review_status"))
    if status == "APPROVE_TINY_PAPER_REVIEW":
        base_required = [
            ("reviewer", "reviewer name"),
            ("review_date", "review date"),
            ("approved_cap_pct", "approved cap"),
            ("paper_stop_pct", "paper stop"),
            ("thesis_plain", "plain-English thesis"),
            ("decision_note", "decision note"),
        ]
        for col, label in base_required:
            if not is_filled(row.get(col)):
                missing.append(label)

    if "Earnings gap" in blocker_types or status == "APPROVE_TINY_PAPER_REVIEW":
        for col, label in [
            ("earnings_date", "next earnings date"),
            ("expected_event_move_pct", "expected earnings/event move"),
            ("event_size_policy", "event size policy"),
        ]:
            if not is_filled(row.get(col)):
                missing.append(label)

    if "Liquidity" in blocker_types or "Execution proof" in blocker_types or status == "APPROVE_TINY_PAPER_REVIEW":
        for col, label in [
            ("liquidity_snapshot_date", "liquidity snapshot date"),
            ("bid_ask_spread_bps", "bid/ask spread"),
            ("avg_daily_dollar_volume_check", "dollar-volume check"),
        ]:
            if not is_filled(row.get(col)):
                missing.append(label)

    if "Sector classification" in blocker_types or status == "APPROVE_TINY_PAPER_REVIEW":
        for col, label in [
            ("sector_confirmed", "sector/theme confirmation"),
            ("crowding_check", "crowding check"),
        ]:
            if not is_filled(row.get(col)):
                missing.append(label)

    if "News proof" in blocker_types and not is_filled(row.get("news_proof_note")):
        missing.append("news proof note")
    if "Execution proof" in blocker_types and not is_filled(row.get("execution_proof_note")):
        missing.append("execution proof note")

    return sorted(set(missing))


def validate_review(row: pd.Series, rank_row: pd.Series | None, blocker_types: set[str]) -> tuple[str, str, float, list[str], str]:
    status = normal_status(row.get("review_status"))
    risk_level = as_text(row.get("risk_level"), as_text(rank_row.get("risk_level") if rank_row is not None else "", "Unknown"))
    lane = as_text(row.get("approval_lane"), as_text(rank_row.get("approval_lane") if rank_row is not None else "", "Unknown"))
    system_cap = safe_float(row.get("system_seed_cap_pct"), safe_float(rank_row.get("starter_cap_if_approved_pct") if rank_row is not None else 0, 0))
    approved_cap = safe_float(row.get("approved_cap_pct"), np.nan)
    spread_bps = safe_float(row.get("bid_ask_spread_bps"), np.nan)
    option_route = as_text(row.get("option_route_requested"), "NO").upper()
    missing = missing_for_blockers(row, blocker_types)

    if status == "REJECT":
        return "Rejected by reviewer", "Rejected. Do not promote unless a new review is opened.", 100.0, [], "Rejected"
    if status == "WATCH_ONLY":
        return "Watch only", "Keep watching. No paper size, no calls, no puts.", 85.0, [], "Watchlist only"
    if status != "APPROVE_TINY_PAPER_REVIEW":
        return "Not reviewed yet", "Fill the PM review template before any promotion discussion.", 0.0, missing, STATUS_HELP

    hard_reasons: list[str] = []
    if "very high" in risk_level.lower():
        hard_reasons.append("risk level is very high")
    if lane == "High-risk sandbox only":
        hard_reasons.append("system lane is high-risk sandbox")
    if np.isfinite(approved_cap) and approved_cap > max(system_cap, 0.0):
        hard_reasons.append("approved cap is larger than the system seed cap")
    if np.isfinite(spread_bps) and spread_bps > 35:
        hard_reasons.append("spread is too wide for clean tiny paper review")
    if option_route not in {"NO", "NONE", "NO_OPTIONS"}:
        hard_reasons.append("options were requested before the option gate cleared")

    proof_fields = [
        "reviewer", "review_date", "approved_cap_pct", "paper_stop_pct", "thesis_plain",
        "earnings_date", "expected_event_move_pct", "event_size_policy",
        "liquidity_snapshot_date", "bid_ask_spread_bps", "avg_daily_dollar_volume_check",
        "sector_confirmed", "crowding_check", "decision_note",
    ]
    filled_count = sum(1 for col in proof_fields if is_filled(row.get(col)))
    proof_score = round(100.0 * filled_count / max(len(proof_fields), 1), 1)
    if missing:
        return "Incomplete approval", "Approval request is missing proof: " + "; ".join(missing[:8]), proof_score, missing, "Incomplete"
    if hard_reasons:
        return "Approval blocked", "Cannot promote: " + "; ".join(hard_reasons), proof_score, hard_reasons, "Blocked"
    return (
        "Ready for final gate check",
        "PM review packet is complete. Next gate may consider tiny paper review only; options remain blocked.",
        100.0,
        [],
        "Tiny paper review only if Final PM Gate also clears",
    )


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rank = read_csv_safe(ROOT / "risk_seed_approval_rank.csv")
    blockers = read_csv_safe(ROOT / "risk_seed_blocker_matrix.csv")
    existing = read_csv_safe(OUT_INPUT)

    if rank.empty:
        empty = pd.DataFrame()
        state = {
            "date": today_str(),
            "status": "NO_RISK_SEED_APPROVAL_RANK",
            "plain_answer": "Step201 needs Step200 approval rank first.",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return empty, empty, empty, state

    bridge = read_csv_safe(ROOT / "pm_review_final_gate_bridge.csv")
    next_actions = read_csv_safe(ROOT / "pm_review_final_gate_next_actions.csv")
    history = read_csv_safe(ROOT / "risk_book_seed_entries_history.csv")
    carry_frames = []
    for carry in [existing, bridge, next_actions, history]:
        if not carry.empty and "ticker" in carry.columns:
            carry_frames.append(carry.copy())
    carry_forward = pd.concat(carry_frames, ignore_index=True, sort=False) if carry_frames else pd.DataFrame()

    template = build_input_template(rank, existing, carry_forward)
    template.to_csv(OUT_INPUT, index=False)

    rank_map = one_by_ticker(rank)
    blocker_map = blocker_types_by_ticker(blockers)
    status_rows: list[dict[str, Any]] = []
    todo_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for _, row in template.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        rank_row = rank_map.get(ticker)
        types = blocker_map.get(ticker, set())
        review_state, allowed_next, proof_score, missing, short_state = validate_review(row, rank_row, types)
        status = normal_status(row.get("review_status"))

        status_rows.append({
            "ticker": ticker,
            "review_status": status,
            "review_state": review_state,
            "allowed_next_state": allowed_next,
            "proof_score_0_100": proof_score,
            "approval_lane": as_text(row.get("approval_lane"), ""),
            "risk_level": as_text(row.get("risk_level"), ""),
            "system_seed_cap_pct": safe_float(row.get("system_seed_cap_pct"), 0.0),
            "approved_cap_pct": safe_float(row.get("approved_cap_pct"), np.nan),
            "paper_stop_pct": safe_float(row.get("paper_stop_pct"), np.nan),
            "open_blocker_types": "; ".join(sorted(types)) if types else "Final manual check",
            "missing_fields_plain": "; ".join(missing) if missing else "No missing fields for this review state.",
            "option_rule": "Options remain blocked until the separate options, spread, IV, event, and risk gates clear.",
            "source_files": "risk_seed_pm_review_input.csv; risk_seed_approval_rank.csv; risk_seed_blocker_matrix.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        if missing or short_state in {"Blocked", "Incomplete"}:
            todo_rows.append({
                "ticker": ticker,
                "priority": "P1" if status == "APPROVE_TINY_PAPER_REVIEW" else "P2",
                "what_to_fix": "; ".join(missing[:10]) if missing else allowed_next,
                "why_it_matters": "The system cannot let a risk seed become even tiny paper review without this evidence.",
                "where_to_fill": "risk_seed_pm_review_input.csv",
                "research_only": True,
            })

        for col in INPUT_COLUMNS:
            value = row.get(col)
            audit_rows.append({
                "ticker": ticker,
                "field": col,
                "filled": "Yes" if is_filled(value) else "No",
                "value_preview": as_text(value, "")[:140],
            })

    status_df = pd.DataFrame(status_rows)
    todo_df = pd.DataFrame(todo_rows)
    audit_df = pd.DataFrame(audit_rows)

    completed = int((status_df["review_state"] == "Ready for final gate check").sum()) if not status_df.empty else 0
    incomplete_approvals = int((status_df["review_state"] == "Incomplete approval").sum()) if not status_df.empty else 0
    blocked = int((status_df["review_state"] == "Approval blocked").sum()) if not status_df.empty else 0
    not_started = int((status_df["review_state"] == "Not reviewed yet").sum()) if not status_df.empty else 0
    watch_only = int((status_df["review_state"] == "Watch only").sum()) if not status_df.empty else 0
    rejected = int((status_df["review_state"] == "Rejected by reviewer").sum()) if not status_df.empty else 0

    state = {
        "date": today_str(),
        "status": "RISK_SEED_PM_REVIEW_INTAKE_ACTIVE",
        "template_rows": int(len(template)),
        "ready_for_final_gate_check_count": completed,
        "incomplete_approval_count": incomplete_approvals,
        "approval_blocked_count": blocked,
        "not_started_count": not_started,
        "watch_only_count": watch_only,
        "rejected_count": rejected,
        "todo_count": int(len(todo_df)),
        "plain_answer": (
            f"PM review intake is active. {len(template)} review rows exist. "
            f"{completed} are complete enough for the next gate, {incomplete_approvals} approval requests are incomplete, "
            f"and {not_started} have not been reviewed yet."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return status_df, todo_df, audit_df, state


def main() -> None:
    status, todo, audit, state = build_outputs()
    status.to_csv(OUT_STATUS, index=False)
    todo.to_csv(OUT_TODO, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## How To Use\n\nOpen `risk_seed_pm_review_input.csv`, fill one ticker at a time, then rerun Step201. Use only plain evidence. Do not use this file as a trade ticket.",
        "## Review Status\n\n" + df_to_markdown(status.head(90)),
        "## What To Fix\n\n" + df_to_markdown(todo.head(120)),
        "## Field Audit\n\n" + df_to_markdown(audit.head(160)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 201 - Risk Seed PM Review Intake", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Template rows: {state['template_rows']}")
    print(f"[OK] Ready for final gate check: {state['ready_for_final_gate_check_count']}")
    print(f"[OK] Not reviewed yet: {state['not_started_count']}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
