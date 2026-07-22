#!/usr/bin/env python3
"""
Canyon v9 Step 183 - Ticker Reviewability Progress.

Research-only. No broker connection. No live orders.

Step181 gives the front-door verdict and Step182 gives the one-name memo.
Step183 answers the next practical question:

  "What exactly has to clear before this ticker is worth manual review?"

It converts the existing Action Readiness gate matrix into:
  - a per-ticker reviewability score
  - a per-gate checklist with source files
  - a four-step ladder from current state to forbidden actions

This is not a buy/sell score. It cannot grant permission, size a position,
trade options, or bypass risk/source gates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_PROGRESS = ROOT / "ticker_reviewability_progress.csv"
OUT_CHECKLIST = ROOT / "ticker_reviewability_checklist.csv"
OUT_LADDER = ROOT / "ticker_reviewability_ladder.csv"
OUT_STATE = ROOT / "ticker_reviewability_state.json"
OUT_REPORT = ROOT / "ticker_reviewability_report.md"


GATE_WEIGHTS = {
    "risk_repair_gate": 25.0,
    "monitor_gate": 20.0,
    "spread_tca_gate": 15.0,
    "event_proof_gate": 15.0,
    "iv_greeks_gamma_gate": 10.0,
    "price_trigger_gate": 10.0,
    "route_gate": 5.0,
}

STATUS_MULTIPLIERS = {
    "CLEAR": 1.00,
    "REVIEW": 0.50,
    "WAIT_FOR_TRIGGER": 0.35,
    "WAIT": 0.35,
    "NO_DATA": 0.00,
    "DATA_GAP": 0.00,
    "BLOCKED": 0.00,
}

PROOF_QUESTIONS = {
    "risk_repair_gate": "Does the ticker fit the single-name and portfolio risk budget after repair?",
    "monitor_gate": "Are price break, volume spike, volatility shift, spread widening, correlation break, news shock, and risk breach calm or explained?",
    "spread_tca_gate": "Do we have real spread or TCA evidence before any size or option structure is reviewed?",
    "event_proof_gate": "Is the news/event mapped to the correct ticker with a causal read-through and timestamp evidence?",
    "iv_greeks_gamma_gate": "Do IV, Greeks, gamma, kill-zone, and defined-risk structure pass manual review?",
    "price_trigger_gate": "Has the price trigger confirmed with price and volume evidence?",
    "route_gate": "After every prior gate clears, what route remains research-only and what stays forbidden?",
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


def as_upper(value: Any, default: str = "") -> str:
    text = as_text(value, default)
    return text.upper() if text else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def compact(value: Any, limit: int = 360) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def status_multiplier(status: str) -> float:
    raw = as_upper(status, "NO_DATA")
    if "CLEAR" in raw:
        return STATUS_MULTIPLIERS["CLEAR"]
    if "WAIT_FOR_TRIGGER" in raw:
        return STATUS_MULTIPLIERS["WAIT_FOR_TRIGGER"]
    if "REVIEW" in raw:
        return STATUS_MULTIPLIERS["REVIEW"]
    if "WAIT" in raw:
        return STATUS_MULTIPLIERS["WAIT"]
    if "DATA_GAP" in raw or "NO_DATA" in raw:
        return 0.0
    if "BLOCK" in raw:
        return 0.0
    return 0.0


def is_unfinished(status: str) -> bool:
    return status_multiplier(status) < 0.999


def is_high_blocker(row: pd.Series) -> bool:
    status = as_upper(row.get("gate_status"), "NO_DATA")
    severity = as_upper(row.get("gate_severity"), "")
    return ("BLOCK" in status or "DATA_GAP" in status or "NO_DATA" in status) and severity in {"HIGH", "CRITICAL"}


def plain_status(status: str) -> str:
    raw = as_upper(status, "NO_DATA")
    if "CLEAR" in raw:
        return "clear"
    if "REVIEW" in raw:
        return "needs proof review"
    if "WAIT" in raw:
        return "waiting for trigger"
    if "DATA_GAP" in raw or "NO_DATA" in raw:
        return "missing source data"
    if "BLOCK" in raw:
        return "blocked"
    return "unknown"


def reviewability_status(score: float, high_blockers: int, first_gate_id: str) -> tuple[str, str]:
    if first_gate_id == "risk_repair_gate" and high_blockers > 0:
        return (
            "HARD_BLOCKED_RISK_FIRST",
            "Risk repair is the first unfinished proof. Do not review route, options, or size first.",
        )
    if high_blockers > 0:
        return (
            "BLOCKED_SOURCE_OR_MONITOR",
            "A high-severity gate is still blocked. Build the missing source proof before manual review.",
        )
    if score >= 75:
        return (
            "READY_FOR_MANUAL_REVIEW",
            "Evidence is mostly complete, but the route remains research-only and manual.",
        )
    if score >= 45:
        return (
            "PROOF_BUILDING",
            "Enough context exists to keep researching, but not enough to treat it as review-ready.",
        )
    if score >= 20:
        return (
            "EARLY_PROOF",
            "Some gates are partly supported, but the first missing proof still controls the path.",
        )
    return (
        "HARD_BLOCKED",
        "The ticker lacks enough gate evidence for a useful manual review.",
    )


def first_memo_row(memo: pd.DataFrame, ticker: str) -> pd.Series | None:
    if memo.empty or "ticker" not in memo.columns:
        return None
    sub = memo[memo["ticker"].astype(str).str.upper().eq(ticker)]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_checklist(gates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in gates.iterrows():
        ticker = as_upper(row.get("ticker"))
        gate_id = as_text(row.get("gate_id"), "unknown_gate")
        weight = GATE_WEIGHTS.get(gate_id, 5.0)
        status = as_text(row.get("gate_status"), "NO_DATA")
        multiplier = status_multiplier(status)
        earned = round(weight * multiplier, 2)
        rows.append({
            "ticker": ticker,
            "check_order": int(safe_float(row.get("gate_order"), 999)),
            "gate_id": gate_id,
            "gate_name": as_text(row.get("gate_name"), gate_id),
            "gate_status": status,
            "plain_status": plain_status(status),
            "gate_severity": as_text(row.get("gate_severity"), "NO_DATA"),
            "max_points": weight,
            "earned_points": earned,
            "proof_question": PROOF_QUESTIONS.get(gate_id, "What evidence would make this gate reliable?"),
            "current_value": compact(row.get("current_value"), 420),
            "what_would_clear": compact(row.get("what_would_clear"), 520),
            "source_file": compact(row.get("source_file"), 260),
            "is_blocking_now": bool(is_unfinished(status)),
            "is_high_blocker": bool(is_high_blocker(row)),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    return pd.DataFrame(rows)


def next_three_proofs(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "No checklist rows found."
    unfinished = checks[checks["gate_status"].astype(str).map(is_unfinished)].copy()
    if unfinished.empty:
        return "All listed gates are clear; manual review is still research-only."
    parts: list[str] = []
    for _, row in unfinished.sort_values("check_order").head(3).iterrows():
        parts.append(
            f"{row['gate_name']}: {compact(row['what_would_clear'], 180)}"
        )
    return " | ".join(parts)


def build_progress(gates: pd.DataFrame, checklist: pd.DataFrame, memo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tickers = sorted(set(gates["ticker"].astype(str).str.upper())) if "ticker" in gates.columns else []
    for ticker in tickers:
        gate_rows = gates[gates["ticker"].astype(str).str.upper().eq(ticker)].copy()
        check_rows = checklist[checklist["ticker"].astype(str).str.upper().eq(ticker)].copy()
        check_rows = check_rows.sort_values("check_order")

        total = float(check_rows["max_points"].sum()) if "max_points" in check_rows.columns else 0.0
        earned = float(check_rows["earned_points"].sum()) if "earned_points" in check_rows.columns else 0.0
        score = round((earned / total * 100.0) if total > 0 else 0.0, 1)

        unfinished = check_rows[check_rows["gate_status"].astype(str).map(is_unfinished)].copy()
        first_gate = unfinished.iloc[0] if not unfinished.empty else None
        first_gate_id = as_text(first_gate.get("gate_id")) if first_gate is not None else "ALL_CLEAR"
        high_blockers = int(check_rows["is_high_blocker"].sum()) if "is_high_blocker" in check_rows.columns else 0
        remaining = int(len(unfinished))
        review_rows = check_rows[
            check_rows["gate_status"].astype(str).str.upper().str.contains("REVIEW|WAIT", na=False)
        ]
        clear_rows = check_rows[
            check_rows["gate_status"].astype(str).str.upper().str.contains("CLEAR", na=False)
        ]
        status, status_explain = reviewability_status(score, high_blockers, first_gate_id)

        memo_row = first_memo_row(memo, ticker)
        route = "Research-only after gates clear."
        options = "No option route can be used before gates clear."
        current_verdict = status_explain
        short_term = "Research only."
        medium_term = "Research only."
        long_term = "Research only."
        conflict_status = "NO_MEMO"
        if memo_row is not None:
            route = as_text(memo_row.get("route_if_every_gate_clears"), route)
            options = as_text(memo_row.get("options_answer"), options)
            current_verdict = as_text(memo_row.get("current_verdict"), current_verdict)
            short_term = as_text(memo_row.get("short_term_view"), short_term)
            medium_term = as_text(memo_row.get("medium_term_view"), medium_term)
            long_term = as_text(memo_row.get("long_term_view"), long_term)
            conflict_status = as_text(memo_row.get("conflict_status"), "NO_ROUTE_CONFLICT")

        rows.append({
            "ticker": ticker,
            "reviewability_score_0_100": score,
            "reviewability_status": status,
            "plain_english_status": status_explain,
            "first_unfinished_gate": as_text(first_gate.get("gate_name")) if first_gate is not None else "All listed gates clear",
            "first_unfinished_gate_id": first_gate_id,
            "first_source_to_open": as_text(first_gate.get("source_file")) if first_gate is not None else "Manual research log",
            "what_would_clear_next": as_text(first_gate.get("what_would_clear")) if first_gate is not None else "Manual confirmation only; no live order path.",
            "remaining_gates_count": remaining,
            "high_blockers_count": high_blockers,
            "review_or_wait_gates_count": int(len(review_rows)),
            "clear_gates_count": int(len(clear_rows)),
            "next_three_proofs": next_three_proofs(check_rows),
            "route_if_every_gate_clears": route,
            "options_permission_plain": options,
            "current_verdict": current_verdict,
            "short_term_view": short_term,
            "medium_term_view": medium_term,
            "long_term_view": long_term,
            "conflict_status": conflict_status,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["high_blockers_count", "reviewability_score_0_100", "remaining_gates_count"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        out.insert(0, "review_rank", range(1, len(out) + 1))
    return out


def build_ladder(progress: pd.DataFrame, checklist: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in progress.iterrows():
        ticker = as_text(row.get("ticker"))
        check_rows = checklist[checklist["ticker"].astype(str).eq(ticker)].sort_values("check_order")
        first_source = as_text(row.get("first_source_to_open"), "NO_SOURCE")
        next_clear = as_text(row.get("what_would_clear_next"), "NO_CLEAR_PATH")
        options = as_text(row.get("options_permission_plain"), "No option route.")
        route = as_text(row.get("route_if_every_gate_clears"), "Research-only route.")
        unresolved_sources = "; ".join(
            check_rows.loc[check_rows["is_blocking_now"].astype(bool), "source_file"].dropna().astype(str).head(4).tolist()
        ) or first_source
        rows.extend([
            {
                "ticker": ticker,
                "ladder_order": 1,
                "ladder_step": "Current state",
                "plain_answer": f"{row.get('reviewability_status')}: {row.get('current_verdict')}",
                "source_files": first_source,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            },
            {
                "ticker": ticker,
                "ladder_order": 2,
                "ladder_step": "First proof to collect",
                "plain_answer": f"Open {first_source}. Clear condition: {next_clear}",
                "source_files": first_source,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            },
            {
                "ticker": ticker,
                "ladder_order": 3,
                "ladder_step": "What unlocks manual review",
                "plain_answer": f"After unfinished gates clear, only this research route can be reviewed: {route}",
                "source_files": unresolved_sources,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            },
            {
                "ticker": ticker,
                "ladder_order": 4,
                "ladder_step": "Still forbidden",
                "plain_answer": f"No broker connection, no live orders, no naked weekly options, and no headline-only upgrade. Options note: {options}",
                "source_files": "ticker_research_memo.csv; action_readiness_gate_matrix.csv",
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            },
        ])
    return pd.DataFrame(rows)


def main() -> None:
    gates = read_csv_safe(ROOT / "action_readiness_gate_matrix.csv")
    memo = read_csv_safe(ROOT / "ticker_research_memo.csv")

    if gates.empty or "ticker" not in gates.columns:
        empty_progress = pd.DataFrame(columns=[
            "review_rank", "ticker", "reviewability_score_0_100",
            "reviewability_status", "first_unfinished_gate", "first_source_to_open",
        ])
        empty_checklist = pd.DataFrame()
        empty_ladder = pd.DataFrame()
        empty_progress.to_csv(OUT_PROGRESS, index=False)
        empty_checklist.to_csv(OUT_CHECKLIST, index=False)
        empty_ladder.to_csv(OUT_LADDER, index=False)
        write_json(OUT_STATE, {
            "status": "TICKER_REVIEWABILITY_PROGRESS_NO_DATA",
            "date": today_str(),
            "reason": "action_readiness_gate_matrix.csv missing or empty",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        write_markdown_report(
            OUT_REPORT,
            "Canyon v9 Step 183 - Ticker Reviewability Progress",
            ["No gate matrix rows found. Run Steps 178-182 first."],
        )
        print("Step183 complete: no gate data")
        return

    gates = gates.copy()
    gates["ticker"] = gates["ticker"].astype(str).str.upper()
    if "gate_order" not in gates.columns:
        gates["gate_order"] = range(1, len(gates) + 1)

    checklist = build_checklist(gates)
    progress = build_progress(gates, checklist, memo)
    ladder = build_ladder(progress, checklist)

    progress.to_csv(OUT_PROGRESS, index=False)
    checklist.to_csv(OUT_CHECKLIST, index=False)
    ladder.to_csv(OUT_LADDER, index=False)

    ready = int(progress["reviewability_status"].eq("READY_FOR_MANUAL_REVIEW").sum()) if not progress.empty else 0
    high_blocked = int((progress["high_blockers_count"] > 0).sum()) if not progress.empty else 0
    avg_score = round(float(progress["reviewability_score_0_100"].mean()), 1) if not progress.empty else 0.0
    top = progress.iloc[0].to_dict() if not progress.empty else {}

    state = {
        "status": "TICKER_REVIEWABILITY_PROGRESS_ACTIVE",
        "date": today_str(),
        "ticker_rows": int(len(progress)),
        "checklist_rows": int(len(checklist)),
        "ladder_rows": int(len(ladder)),
        "average_reviewability_score_0_100": avg_score,
        "ready_for_manual_review_tickers": ready,
        "high_blocked_tickers": high_blocked,
        "top_reviewable_ticker": as_text(top.get("ticker"), "NO_DATA"),
        "top_reviewability_score_0_100": top.get("reviewability_score_0_100", "NO_DATA"),
        "top_first_unfinished_gate": as_text(top.get("first_unfinished_gate"), "NO_DATA"),
        "top_first_source_to_open": as_text(top.get("first_source_to_open"), "NO_DATA"),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        f"- Average reviewability score: {avg_score}/100",
        f"- Ready for manual review: {ready}",
        f"- Tickers with high blocked gates: {high_blocked}",
        f"- Top ticker to study: {state['top_reviewable_ticker']}",
        "",
        "## Important guardrail",
        "This is not an opportunity score. It only measures whether enough evidence exists for manual review. No broker connection. No live orders.",
        "",
        "## Reviewability progress",
        df_to_markdown(progress.head(15), max_rows=15),
        "",
        "## First checklist rows",
        df_to_markdown(checklist.head(30), max_rows=30),
    ]
    write_markdown_report(
        OUT_REPORT,
        "Canyon v9 Step 183 - Ticker Reviewability Progress",
        sections,
    )
    print(
        f"Step183 complete: {len(progress)} tickers, {len(checklist)} checks, "
        f"{len(ladder)} ladder rows, avg score {avg_score}"
    )


if __name__ == "__main__":
    main()
