#!/usr/bin/env python3
"""
Canyon v9 Step 189 - Sharpe 4 Risk-Book Promotion Gate.

Research-only. No broker connection. No live orders.

Step188 creates risk-book intake cards. Step189 makes those cards actionable:
for each candidate, it names the single next proof to collect and states what
is still forbidden.

Outputs:
  sharpe4_risk_book_promotion_state.json
  sharpe4_risk_book_promotion_gate.csv
  sharpe4_risk_book_manual_proof_queue.csv
  sharpe4_risk_book_promotion_report.md
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


OUT_STATE = ROOT / "sharpe4_risk_book_promotion_state.json"
OUT_GATE = ROOT / "sharpe4_risk_book_promotion_gate.csv"
OUT_QUEUE = ROOT / "sharpe4_risk_book_manual_proof_queue.csv"
OUT_REPORT = ROOT / "sharpe4_risk_book_promotion_report.md"


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def merge_inputs() -> pd.DataFrame:
    cards = read_csv_safe(ROOT / "sharpe4_risk_book_candidate_cards.csv")
    var_liq = read_csv_safe(ROOT / "sharpe4_risk_book_var_liquidity.csv")
    event = read_csv_safe(ROOT / "sharpe4_risk_book_event_route.csv")
    corr = read_csv_safe(ROOT / "sharpe4_risk_book_correlation_proxy.csv")

    if cards.empty or "ticker" not in cards.columns:
        return pd.DataFrame()
    out = cards.copy()
    out["ticker"] = out["ticker"].apply(clean_ticker)
    for df, cols in [
        (var_liq, [
            "ticker", "annual_vol_pct", "daily_cvar_95_pct", "five_day_cvar_95_pct",
            "price_risk", "estimated_tca_bps", "price_data_status",
        ]),
        (event, [
            "ticker", "event_score", "event_role", "event_headline", "earnings_status",
            "earnings_date", "days_to_earnings", "iv_rank", "event_route", "option_answer",
        ]),
        (corr, [
            "ticker", "corr_to_spy", "corr_to_qqq", "corr_to_smh",
            "highest_peer", "highest_peer_corr", "correlation_risk",
        ]),
    ]:
        if df.empty or "ticker" not in df.columns:
            continue
        work = df.copy()
        work["ticker"] = work["ticker"].apply(clean_ticker)
        keep = [c for c in cols if c in work.columns]
        out = out.merge(work[keep].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_detail"))
    return out


def first_proof(row: pd.Series) -> tuple[str, str, str]:
    risk = as_text(row.get("risk_level"))
    earnings = as_text(row.get("earnings"))
    corr = as_text(row.get("correlation"))
    liq = as_text(row.get("liquidity"))
    cvar = safe_float(row.get("daily_cvar_95_pct"))
    tca = safe_float(row.get("estimated_tca_bps"))
    event_headline = as_text(row.get("event_headline"))

    if "missing" in earnings.lower():
        return (
            "Fill earnings date and gap risk",
            "Open an earnings calendar source, record next report date, consensus move, and whether the name must be flat/reduced before the event.",
            "Earnings date is not proven, so the model cannot know gap risk.",
        )
    if risk == "Very high" or (np.isfinite(cvar) and cvar >= 8.0):
        return (
            "Write a tail-risk stop plan",
            "Record 1-day CVaR, 5-day CVaR, maximum starter cap, and the exact stop/invalidation rule before any watch upgrade.",
            "Tail risk is too high for a normal starter position.",
        )
    if corr in {"Crowded", "Factor heavy"}:
        peer = as_text(row.get("highest_peer"), "peer")
        return (
            "Check crowding against peers",
            f"Treat this as the same exposure bucket as {peer}; set a combined cap before calling it diversification.",
            "The ticker may just be another version of the same technology / semiconductor bet.",
        )
    if liq in {"Review", "Thin", "Data missing"} or (np.isfinite(tca) and tca > 10):
        return (
            "Capture live spread proof",
            "Record bid/ask spread, expected fill quality, and whether a tiny paper slice would exceed the TCA budget.",
            "Paper Sharpe can disappear if spread and failed-fill assumptions are too optimistic.",
        )
    if not event_headline:
        return (
            "Find source event proof",
            "Attach the source headline and explain exactly why it helps or hurts this ticker.",
            "There is no readable event source attached to the candidate.",
        )
    return (
        "Validate event-time reaction",
        "Check whether the ticker actually moved with the event, whether volume confirmed it, and whether the move faded.",
        "A headline is not enough; the price/volume reaction must prove the causal link.",
    )


def promotion_status(row: pd.Series, proof: str) -> tuple[str, str, str]:
    risk = as_text(row.get("risk_level"))
    corr = as_text(row.get("correlation"))
    earnings = as_text(row.get("earnings"))
    tca = safe_float(row.get("estimated_tca_bps"))

    hard = []
    if risk == "Very high":
        hard.append("tail risk")
    if corr == "Crowded":
        hard.append("crowding")
    if "missing" in earnings.lower():
        hard.append("earnings proof")
    if np.isfinite(tca) and tca > 12:
        hard.append("TCA")

    if hard:
        return (
            "Blocked from paper review",
            "No paper size. No call/put route. Research only.",
            "Blocked by " + ", ".join(hard) + ".",
        )
    if proof == "Validate event-time reaction":
        return (
            "Can become watch-only after proof",
            "Still no paper size. It may enter watch-only if event reaction, spread, and risk cap are documented.",
            "The remaining blocker is proof, not basic data quality.",
        )
    return (
        "Needs manual proof first",
        "Research only. Complete the named proof before any watch upgrade.",
        "The model needs one missing fact before the next gate.",
    )


def option_gate(row: pd.Series, status: str) -> str:
    raw = as_text(row.get("options_now")) or as_text(row.get("option_answer"))
    risk = as_text(row.get("risk_level"))
    iv = safe_float(row.get("iv_rank"))
    if status != "Can become watch-only after proof":
        return "Options blocked now. Do not look for calls or puts until the risk-book gate changes."
    if risk in {"High", "Very high"} or (np.isfinite(iv) and iv >= 70):
        return "Options still research-only. High risk or rich IV means no weekly calls; defined-risk only after proof."
    if "call" in raw.lower():
        return "Call research may be reviewed after proof, but only defined-risk and no weekly chase."
    if "put" in raw.lower():
        return "Put or hedge research may be reviewed after proof, but only defined-risk."
    return "Stock-first route. Options remain blocked until a separate options gate is clean."


def build_gate() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = merge_inputs()
    if base.empty:
        state = {
            "date": today_str(),
            "status": "NO_RISK_BOOK_INTAKE",
            "candidate_count": 0,
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    rows = []
    queue = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        proof, instruction, why = first_proof(row)
        status, permission, status_reason = promotion_status(row, proof)
        opt = option_gate(row, status)
        rows.append({
            "ticker": ticker,
            "promotion_status": status,
            "current_permission": permission,
            "first_proof_to_collect": proof,
            "why_this_is_first": why,
            "proof_instruction": instruction,
            "option_gate": opt,
            "risk_level": row.get("risk_level", ""),
            "earnings": row.get("earnings", ""),
            "correlation": row.get("correlation", ""),
            "liquidity": row.get("liquidity", ""),
            "daily_cvar_95_pct": row.get("daily_cvar_95_pct", np.nan),
            "estimated_tca_bps": row.get("estimated_tca_bps", np.nan),
            "event_headline": row.get("event_headline", ""),
            "status_reason": status_reason,
            "do_not_do": "Do not call this a Sharpe 4 contributor, do not size, and do not use options until this gate is upgraded by evidence.",
            "source_files": "sharpe4_risk_book_candidate_cards.csv / var_liquidity / event_route / correlation_proxy",
            "research_only": True,
        })
        queue.append({
            "priority": "P1" if status == "Blocked from paper review" else "P2",
            "ticker": ticker,
            "task": proof,
            "how_to_do_it": instruction,
            "why": why,
            "done_when": "The proof is written into the risk book and can be traced to a source file.",
            "still_forbidden": "No paper size and no options before gate review.",
            "research_only": True,
        })

    gate = pd.DataFrame(rows)
    status_order = {
        "Blocked from paper review": 0,
        "Needs manual proof first": 1,
        "Can become watch-only after proof": 2,
    }
    gate["_status_order"] = gate["promotion_status"].map(status_order).fillna(9)
    gate = gate.sort_values(["_status_order", "ticker"]).drop(columns=["_status_order"]).reset_index(drop=True)
    queue_df = pd.DataFrame(queue)
    queue_df["_priority_order"] = queue_df["priority"].map({"P1": 0, "P2": 1}).fillna(9)
    queue_df = queue_df.sort_values(["_priority_order", "ticker"]).drop(columns=["_priority_order"]).reset_index(drop=True)

    state = {
        "date": today_str(),
        "status": "PROMOTION_GATE_ACTIVE",
        "candidate_count": int(len(gate)),
        "blocked_from_paper_review_count": int(gate["promotion_status"].eq("Blocked from paper review").sum()),
        "needs_manual_proof_count": int(gate["promotion_status"].eq("Needs manual proof first").sum()),
        "can_become_watch_only_after_proof_count": int(gate["promotion_status"].eq("Can become watch-only after proof").sum()),
        "paper_sizing_allowed_now_count": 0,
        "options_allowed_now_count": 0,
        "plain_english": "This is a proof queue, not a buy list. A ticker cannot graduate until the first proof item is filled and reviewed.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return gate, queue_df, state


def write_report(gate: pd.DataFrame, queue: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Status: **{state['status']}**",
            f"- Candidates: **{state['candidate_count']}**",
            f"- Blocked from paper review: **{state['blocked_from_paper_review_count']}**",
            f"- Can become watch-only after proof: **{state['can_become_watch_only_after_proof_count']}**",
            f"- Paper sizing allowed now: **{state['paper_sizing_allowed_now_count']}**",
            f"- Options allowed now: **{state['options_allowed_now_count']}**",
            "",
            state["plain_english"],
        ]),
        "## Promotion Gate\n\n" + df_to_markdown(gate.head(30)),
        "## Manual Proof Queue\n\n" + df_to_markdown(queue.head(30)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 189 - Sharpe 4 Risk-Book Promotion Gate", sections)


def main() -> None:
    gate, queue, state = build_gate()
    gate.to_csv(OUT_GATE, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    write_json(OUT_STATE, state)
    write_report(gate, queue, state)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Candidates: {state['candidate_count']}")
    print(f"[OK] Blocked from paper review: {state['blocked_from_paper_review_count']}")
    print(f"[OK] Watch-only after proof: {state['can_become_watch_only_after_proof_count']}")
    print(f"[OK] Paper sizing allowed now: {state['paper_sizing_allowed_now_count']}")


if __name__ == "__main__":
    main()
