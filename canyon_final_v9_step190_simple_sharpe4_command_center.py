#!/usr/bin/env python3
"""
Canyon v9 Step 190 - Simple Sharpe 4 Command Center.

Research-only. No broker connection. No live orders.

The previous Sharpe 4 repair steps are useful, but they create too many tables.
This step makes a human-readable front panel:

  - what is the answer right now?
  - what is forbidden?
  - what is the next proof to collect?
  - which names are worth reading first?

Outputs:
  sharpe4_simple_command_state.json
  sharpe4_simple_today_cards.csv
  sharpe4_simple_candidate_queue.csv
  sharpe4_simple_report.md
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
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "sharpe4_simple_command_state.json"
OUT_CARDS = ROOT / "sharpe4_simple_today_cards.csv"
OUT_QUEUE = ROOT / "sharpe4_simple_candidate_queue.csv"
OUT_REPORT = ROOT / "sharpe4_simple_report.md"


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


def shorten(text: Any, limit: int = 180) -> str:
    clean = " ".join(as_text(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def plain_status(status: Any) -> str:
    text = as_text(status).replace("_", " ").title()
    return text or "No Data"


def build_state() -> dict[str, Any]:
    target = read_json_safe(ROOT / "sharpe_target4_state.json", {})
    p0 = read_json_safe(ROOT / "sharpe4_p0_repair_state.json", {})
    intake = read_json_safe(ROOT / "sharpe4_risk_book_intake_state.json", {})
    promo = read_json_safe(ROOT / "sharpe4_risk_book_promotion_state.json", {})

    current_sharpe = safe_float(target.get("current_headline_sharpe"), np.nan)
    proof_sharpe = safe_float(target.get("credibility_adjusted_planning_sharpe"), np.nan)
    alpha_allowed = safe_float(p0.get("sharpe4_alpha_gross_allowed_pct"), 0.0)
    paper_allowed = safe_float(promo.get("paper_sizing_allowed_now_count"), 0.0)
    options_allowed = safe_float(promo.get("options_allowed_now_count"), 0.0)
    blocked_review = int(safe_float(promo.get("blocked_from_paper_review_count"), 0))
    watch_after_proof = int(safe_float(promo.get("can_become_watch_only_after_proof_count"), 0))

    if paper_allowed == 0 and options_allowed == 0:
        answer = "No new paper size and no options today."
        mode = "Proof First"
    else:
        answer = "Some gates may be open, but still review manually."
        mode = "Manual Review"

    if alpha_allowed == 0:
        first_job = "Fix the proof and risk book before trying to improve Sharpe."
    else:
        first_job = "Review only the names that cleared proof and risk gates."

    return {
        "date": today_str(),
        "mode": mode,
        "answer": answer,
        "first_job": first_job,
        "current_headline_sharpe": round(current_sharpe, 2) if np.isfinite(current_sharpe) else None,
        "proof_adjusted_sharpe": round(proof_sharpe, 2) if np.isfinite(proof_sharpe) else None,
        "target_sharpe": 4.0,
        "alpha_gross_allowed_pct": round(alpha_allowed, 2),
        "paper_sizing_allowed_now_count": int(paper_allowed),
        "options_allowed_now_count": int(options_allowed),
        "risk_book_candidates": int(safe_float(intake.get("candidate_count"), 0)),
        "blocked_from_paper_review_count": blocked_review,
        "watch_only_after_proof_count": watch_after_proof,
        "main_warning": "A higher Sharpe cannot be claimed by relabeling the same backtest. It needs cleaner data, lower cost, better risk control, and proof that signals work out of sample.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def build_cards(state: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "card": "Answer now",
            "value": state["answer"],
            "why_it_matters": "This stops the dashboard from turning a research idea into a trade too early.",
            "what_to_open": "Performance > Simple Command Center",
        },
        {
            "card": "First job",
            "value": state["first_job"],
            "why_it_matters": "The current book is still in cleanup mode; chasing new calls would make the system noisier.",
            "what_to_open": "Manual proof queue",
        },
        {
            "card": "Sharpe truth",
            "value": f"Headline {state.get('current_headline_sharpe')} vs target {state.get('target_sharpe')}; proof-adjusted {state.get('proof_adjusted_sharpe')}.",
            "why_it_matters": "The target is a goal, not a result. The proof-adjusted number is the honest planning number.",
            "what_to_open": "Detailed Sharpe diagnostics",
        },
        {
            "card": "What is forbidden",
            "value": f"Paper sizing: {state.get('paper_sizing_allowed_now_count')}; options: {state.get('options_allowed_now_count')}.",
            "why_it_matters": "If both are zero, do not search for calls, puts, or new size yet.",
            "what_to_open": "Risk-book promotion gate",
        },
        {
            "card": "Candidate queue",
            "value": f"{state.get('risk_book_candidates')} names checked; {state.get('blocked_from_paper_review_count')} blocked; {state.get('watch_only_after_proof_count')} can become watch-only after proof.",
            "why_it_matters": "The list is a proof queue, not a buy list.",
            "what_to_open": "Simple candidate queue",
        },
    ]
    return pd.DataFrame(rows)


def simpler_reason(row: pd.Series) -> str:
    status = as_text(row.get("promotion_status"))
    first = as_text(row.get("first_proof_to_collect"))
    risk = as_text(row.get("risk_level"))
    earnings = as_text(row.get("earnings"))
    corr = as_text(row.get("correlation"))
    if "earnings" in first.lower():
        return "The next earnings date or gap risk is not proven."
    if "tail" in first.lower():
        return "The stock can move too much; write the loss limit first."
    if "crowding" in first.lower():
        return "It may be the same exposure as another tech or semiconductor name."
    if status == "Can become watch-only after proof":
        return "The basic gate is not terrible, but the event still needs price and volume proof."
    if risk or earnings or corr:
        return f"Risk: {risk or 'unknown'}; earnings: {earnings or 'unknown'}; correlation: {corr or 'unknown'}."
    return "The risk book is not complete."


def build_candidate_queue() -> pd.DataFrame:
    gate = read_csv_safe(ROOT / "sharpe4_risk_book_promotion_gate.csv")
    cards = read_csv_safe(ROOT / "sharpe4_risk_book_candidate_cards.csv")
    if gate.empty or "ticker" not in gate.columns:
        return pd.DataFrame()
    gate = gate.copy()
    gate["ticker"] = gate["ticker"].apply(clean_ticker)

    if not cards.empty and "ticker" in cards.columns:
        c = cards.copy()
        c["ticker"] = c["ticker"].apply(clean_ticker)
        keep = [x for x in ["ticker", "plain_thesis", "short_term", "medium_term", "long_term", "main_blockers"] if x in c.columns]
        gate = gate.merge(c[keep].drop_duplicates("ticker"), on="ticker", how="left")

    priority_map = {
        "Blocked from paper review": 1,
        "Needs manual proof first": 2,
        "Can become watch-only after proof": 3,
    }
    gate["_priority"] = gate["promotion_status"].map(priority_map).fillna(9)
    gate = gate.sort_values(["_priority", "ticker"]).drop(columns=["_priority"]).reset_index(drop=True)

    rows = []
    for _, row in gate.iterrows():
        status = as_text(row.get("promotion_status"))
        if status == "Blocked from paper review":
            simple_status = "Not ready"
        elif status == "Can become watch-only after proof":
            simple_status = "May watch after proof"
        else:
            simple_status = "Needs proof"

        rows.append({
            "ticker": clean_ticker(row.get("ticker")),
            "simple_status": simple_status,
            "first_proof": as_text(row.get("first_proof_to_collect"), "Complete risk proof"),
            "why": simpler_reason(row),
            "what_to_do_next": shorten(row.get("proof_instruction"), 220),
            "do_not_do": "Do not size. Do not use options. Do not count it as Sharpe 4 alpha yet.",
            "short_term_read": shorten(row.get("short_term"), 160),
            "medium_term_read": shorten(row.get("medium_term"), 160),
            "source_headline": shorten(row.get("event_headline"), 150),
            "source_files": "sharpe4_risk_book_promotion_gate.csv / sharpe4_risk_book_candidate_cards.csv",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def write_report(state: dict[str, Any], cards: pd.DataFrame, queue: pd.DataFrame) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Mode: **{state['mode']}**",
            f"- Answer: **{state['answer']}**",
            f"- First job: **{state['first_job']}**",
            f"- Headline Sharpe: **{state.get('current_headline_sharpe')}**",
            f"- Proof-adjusted Sharpe: **{state.get('proof_adjusted_sharpe')}**",
            f"- Paper sizing allowed now: **{state['paper_sizing_allowed_now_count']}**",
            f"- Options allowed now: **{state['options_allowed_now_count']}**",
            "",
            state["main_warning"],
        ]),
        "## Today Cards\n\n" + df_to_markdown(cards),
        "## Simple Candidate Queue\n\n" + df_to_markdown(queue.head(30)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 190 - Simple Sharpe 4 Command Center", sections)


def main() -> None:
    state = build_state()
    cards = build_cards(state)
    queue = build_candidate_queue()

    cards.to_csv(OUT_CARDS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    write_json(OUT_STATE, state)
    write_report(state, cards, queue)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Mode: {state['mode']}")
    print(f"[OK] Paper sizing allowed now: {state['paper_sizing_allowed_now_count']}")
    print(f"[OK] Options allowed now: {state['options_allowed_now_count']}")
    print(f"[OK] Simple queue rows: {len(queue)}")


if __name__ == "__main__":
    main()
