#!/usr/bin/env python3
"""
Canyon v9 Step 193 - PM Morning Brief.

Research-only. No broker connection. No live orders.

This is not a new signal. It is the operating layer a portfolio manager should
see first: risk posture, what changed, what can move forward, what is blocked,
and which evidence must be checked before any paper or options idea.

Outputs:
  pm_morning_brief_state.json
  pm_morning_brief_cards.csv
  pm_morning_brief_focus_queue.csv
  pm_morning_brief_news_to_verify.csv
  pm_morning_brief_report.md
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


OUT_STATE = ROOT / "pm_morning_brief_state.json"
OUT_CARDS = ROOT / "pm_morning_brief_cards.csv"
OUT_QUEUE = ROOT / "pm_morning_brief_focus_queue.csv"
OUT_NEWS = ROOT / "pm_morning_brief_news_to_verify.csv"
OUT_REPORT = ROOT / "pm_morning_brief_report.md"


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default


def shorten(value: Any, limit: int = 210) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def severity_rank(value: Any) -> int:
    text = as_text(value).upper()
    if any(x in text for x in ["CRITICAL", "REDUCE_ONLY", "HARD", "BLOCK"]):
        return 0
    if any(x in text for x in ["SIZE_DOWN", "WARNING", "REVIEW"]):
        return 1
    if any(x in text for x in ["WATCH", "MONITOR"]):
        return 2
    return 3


def plain_risk_task(value: Any) -> str:
    text = as_text(value)
    upper = text.upper()
    if "RESEARCH ONLY REDUCTION" in upper or "NEW BUYING" in upper:
        return "Do not add new exposure. Review or reduce this risk first."
    if "CUT PAPER SIZE" in upper:
        return "Use a smaller paper size before considering any new idea."
    return text or "Review the risk breach before considering new ideas."


def plain_budget_reason(value: Any) -> str:
    text = as_text(value)
    mapping = {
        "Single-name tail-risk budget": "One stock could lose too much if it moves sharply.",
        "Macro scenario loss budget": "A bad macro scenario could hurt the whole portfolio too much.",
        "Annual volatility target": "The portfolio is moving more than the target allows.",
        "Crisis-correlation volatility budget": "In a crisis, many stocks may fall together, so diversification may not help.",
        "Earnings gap-loss budget": "Earnings-day jumps could create a loss that is too large.",
    }
    return mapping.get(text, f"{text or 'A risk budget'} is over or near its limit.")


def risk_posture(breaches: pd.DataFrame, monitor: pd.DataFrame) -> dict[str, Any]:
    hard = 0
    size_down = 0
    if not breaches.empty:
        text = breaches.astype(str).agg(" ".join, axis=1).str.upper()
        hard = int(text.str.contains("REDUCE_ONLY|HARD|BLOCK|CRITICAL", regex=True).sum())
        size_down = int(text.str.contains("SIZE_DOWN|REDUCE", regex=True).sum())
    critical_events = 0
    if not monitor.empty and "severity" in monitor.columns:
        critical_events = int(monitor["severity"].astype(str).str.upper().eq("CRITICAL").sum())

    if hard or critical_events:
        return {
            "risk_mode": "Risk first",
            "risk_answer": "Do not add exposure. Clean the risk queue before trusting new ideas.",
            "risk_color": "red",
            "hard_breaches": hard,
            "size_down_breaches": size_down,
            "critical_events": critical_events,
        }
    if size_down:
        return {
            "risk_mode": "Size down",
            "risk_answer": "New exposure needs a smaller risk budget. No aggressive options.",
            "risk_color": "gray",
            "hard_breaches": hard,
            "size_down_breaches": size_down,
            "critical_events": critical_events,
        }
    return {
        "risk_mode": "Risk not blocking",
        "risk_answer": "No hard risk block was found in the current local files.",
        "risk_color": "green",
        "hard_breaches": hard,
        "size_down_breaches": size_down,
        "critical_events": critical_events,
    }


def desk_answer(risk: dict[str, Any], sharpe: dict[str, Any], manual: dict[str, Any]) -> str:
    paper_allowed = safe_int(sharpe.get("paper_sizing_allowed_now_count")) + safe_int(manual.get("paper_sizing_allowed_now_count"))
    options_allowed = safe_int(sharpe.get("options_allowed_now_count")) + safe_int(manual.get("options_allowed_now_count"))
    reviewed = safe_int(manual.get("reviewed_rows"))
    ready = safe_int(manual.get("ready_for_watch_only_review_count"))

    if risk["risk_mode"] in {"Risk first", "Size down"}:
        return "Risk first. No new paper trades and no options today."
    if paper_allowed == 0 and options_allowed == 0:
        if reviewed == 0:
            return "Research only. Fill evidence before moving any name forward."
        if ready:
            return "Some names may be studied, but paper trades and options are still blocked."
        return "Evidence is being reviewed, but nothing is cleared for paper or options."
    return "Some research routes may be reviewed, but every name still needs final risk and execution checks."


def build_cards(state: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "card": "Desk answer",
            "value": state["desk_answer"],
            "why_it_matters": "This is the answer to read before looking at individual tickers.",
            "color": "red" if "No new" in state["desk_answer"] or "Research only" in state["desk_answer"] else "gray",
        },
        {
            "card": "Risk mode",
            "value": state["risk_mode"],
            "why_it_matters": state["risk_answer"],
            "color": state["risk_color"],
        },
        {
            "card": "Evidence status",
            "value": f"{state['reviewed_rows']} filled / {state['not_reviewed_count']} blank",
            "why_it_matters": "Blank evidence means the ticker stays read-only.",
            "color": "gray",
        },
        {
            "card": "Study list",
            "value": str(state["ready_for_watch_only_review_count"]),
            "why_it_matters": "Study list means research only, not paper trading.",
            "color": "green" if state["ready_for_watch_only_review_count"] else "gray",
        },
        {
            "card": "Current Sharpe",
            "value": f"{state['current_headline_sharpe']:.2f}",
            "why_it_matters": f"Honest version after proof is {state['proof_adjusted_sharpe']:.2f}. Do not call the model strong until proof improves.",
            "color": "gray",
        },
    ]
    return pd.DataFrame(rows)


def build_focus_queue(
    breaches: pd.DataFrame,
    manual_gate: pd.DataFrame,
    validation: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not breaches.empty:
        work = breaches.copy()
        if "status" in work.columns:
            work["_rank"] = work["status"].apply(severity_rank)
            work = work.sort_values("_rank")
        for _, row in work.head(5).iterrows():
            rows.append({
                "rank": len(rows) + 1,
                "bucket": "Risk first",
                "ticker": clean_ticker(row.get("ticker")),
                "plain_task": plain_risk_task(row.get("required_next_action")),
                "why": plain_budget_reason(row.get("budget_item")),
                "do_not_do": "Do not add exposure until this is understood.",
                "source": as_text(row.get("source_file"), "risk_desk_breach_table.csv"),
            })

    if not manual_gate.empty:
        work = manual_gate.copy()
        if "manual_review_status" in work.columns:
            order = {
                "Not reviewed yet": 0,
                "Needs more proof": 1,
                "Needs reviewer decision": 2,
                "Proof accepted; still blocked": 3,
            }
            work["_rank"] = work["manual_review_status"].map(order).fillna(9)
            work = work.sort_values(["_rank", "ticker"])
        for _, row in work.head(6).iterrows():
            rows.append({
                "rank": len(rows) + 1,
                "bucket": "Evidence first",
                "ticker": clean_ticker(row.get("ticker")),
                "plain_task": as_text(row.get("next_step_plain"), "Fill the evidence row first."),
                "why": as_text(row.get("why_in_plain_english"), "The ticker cannot move forward without evidence."),
                "do_not_do": as_text(row.get("what_not_to_do"), "Do not paper trade. Do not use options."),
                "source": "sharpe4_manual_proof_review_gate.csv",
            })

    if not validation.empty:
        work = validation.copy()
        if "priority" in work.columns:
            work["_rank"] = work["priority"].apply(severity_rank)
            work = work.sort_values("_rank")
        for _, row in work.head(5).iterrows():
            rows.append({
                "rank": len(rows) + 1,
                "bucket": "News proof",
                "ticker": clean_ticker(row.get("target_ticker")),
                "plain_task": as_text(row.get("required_next_action"), "Verify source timing and price reaction."),
                "why": shorten(row.get("validation_note") or row.get("issue"), 190),
                "do_not_do": "Do not turn this headline into a trade until price/volume confirms the link.",
                "source": "event_causal_validation_queue.csv",
            })

    if not events.empty:
        work = events.copy()
        if "best_event_score" in work.columns:
            work["_score"] = pd.to_numeric(work["best_event_score"], errors="coerce").fillna(0)
            work = work.sort_values("_score", ascending=False)
        for _, row in work.head(3).iterrows():
            tickers = as_text(row.get("top_beneficiaries") or row.get("top_vulnerable_targets"), "No mapped tickers")
            rows.append({
                "rank": len(rows) + 1,
                "bucket": "News to read",
                "ticker": as_text(row.get("source_news_ticker"), ""),
                "plain_task": shorten(row.get("headline"), 190),
                "why": f"Mapped targets: {shorten(tickers, 150)}.",
                "do_not_do": "Do not trust the mapping until the evidence queue is checked.",
                "source": "event_readthrough_event_summary.csv",
            })

    return pd.DataFrame(rows)


def build_news_verify(validation: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not validation.empty:
        work = validation.copy()
        if "priority" in work.columns:
            work["_rank"] = work["priority"].apply(severity_rank)
            work = work.sort_values("_rank")
        for _, row in work.head(12).iterrows():
            rows.append({
                "ticker": clean_ticker(row.get("target_ticker")),
                "headline": shorten(row.get("headline"), 220),
                "why_to_check": shorten(row.get("validation_note") or row.get("issue"), 220),
                "next_step": as_text(row.get("required_next_action"), "Verify source timing and post-event price/volume reaction."),
                "source": as_text(row.get("publisher"), "") + " " + as_text(row.get("published"), ""),
                "link": as_text(row.get("link"), ""),
                "research_only": True,
            })
    elif not events.empty:
        for _, row in events.head(12).iterrows():
            rows.append({
                "ticker": as_text(row.get("source_news_ticker"), ""),
                "headline": shorten(row.get("headline"), 220),
                "why_to_check": "Headline mapped to stocks, but causal proof still needs review.",
                "next_step": as_text(row.get("top_required_proof"), "Check source timing and price/volume reaction."),
                "source": as_text(row.get("publisher"), "") + " " + as_text(row.get("published"), ""),
                "link": as_text(row.get("link"), ""),
                "research_only": True,
            })
    return pd.DataFrame(rows)


def main() -> None:
    sharpe = read_json_safe(ROOT / "sharpe4_simple_command_state.json", {})
    manual = read_json_safe(ROOT / "sharpe4_manual_proof_review_state.json", {})
    subsector = read_json_safe(ROOT / "institutional_subsector_cycle_state.json", {})
    event_state = read_json_safe(ROOT / "event_readthrough_state.json", {})

    breaches = read_csv_safe(ROOT / "risk_desk_breach_table.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_events.csv")
    manual_gate = read_csv_safe(ROOT / "sharpe4_manual_proof_review_gate.csv")
    validation = read_csv_safe(ROOT / "event_causal_validation_queue.csv")
    events = read_csv_safe(ROOT / "event_readthrough_event_summary.csv")

    risk = risk_posture(breaches, monitor)
    answer = desk_answer(risk, sharpe, manual)

    current_sharpe = safe_float(sharpe.get("current_headline_sharpe"), 0.0)
    proof_sharpe = safe_float(sharpe.get("proof_adjusted_sharpe"), 0.0)
    reviewed = safe_int(manual.get("reviewed_rows"))
    not_reviewed = safe_int(manual.get("not_reviewed_count"))
    ready = safe_int(manual.get("ready_for_watch_only_review_count"))
    event_count = safe_int(event_state.get("event_count"), len(events))
    validation_count = len(validation)

    state = {
        "date": today_str(),
        "status": "PM_BRIEF_ACTIVE",
        "desk_answer": answer,
        "risk_mode": risk["risk_mode"],
        "risk_answer": risk["risk_answer"],
        "risk_color": risk["risk_color"],
        "hard_breaches": risk["hard_breaches"],
        "size_down_breaches": risk["size_down_breaches"],
        "critical_events": risk["critical_events"],
        "reviewed_rows": reviewed,
        "not_reviewed_count": not_reviewed,
        "ready_for_watch_only_review_count": ready,
        "current_headline_sharpe": current_sharpe,
        "proof_adjusted_sharpe": proof_sharpe,
        "event_count": event_count,
        "news_validation_rows": validation_count,
        "subsector_note": as_text(subsector.get("logic"), "No subsector cycle note available."),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "plain_english": "Read this first. It converts the system's risk, news, evidence, and performance files into a PM morning brief.",
    }

    cards = build_cards(state)
    queue = build_focus_queue(breaches, manual_gate, validation, events)
    news_verify = build_news_verify(validation, events)

    cards.to_csv(OUT_CARDS, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    news_verify.to_csv(OUT_NEWS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Desk Answer",
            "",
            f"**{answer}**",
            "",
            f"- Risk mode: **{state['risk_mode']}**",
            f"- Hard breaches: **{state['hard_breaches']}**",
            f"- Evidence filled / blank: **{reviewed} / {not_reviewed}**",
            f"- Names ready for study only: **{ready}**",
            f"- Headline Sharpe / proof-adjusted Sharpe: **{current_sharpe:.2f} / {proof_sharpe:.2f}**",
            f"- News events / validation rows: **{event_count} / {validation_count}**",
        ]),
        "## Brief Cards\n\n" + df_to_markdown(cards),
        "## PM Focus Queue\n\n" + df_to_markdown(queue.head(25)),
        "## News To Verify\n\n" + df_to_markdown(news_verify.head(25)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 193 - PM Morning Brief", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Desk answer: {answer}")
    print(f"[OK] Focus queue rows: {len(queue)}")
    print(f"[OK] News verify rows: {len(news_verify)}")
    print(f"[OK] Research-only: True")


if __name__ == "__main__":
    main()
