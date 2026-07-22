#!/usr/bin/env python3
"""
Canyon v9 Step 181 - Deep Decision Desk.

Research-only. No broker connection. No live orders.

The dashboard now has many strong modules, but a user can get lost if every raw
table appears at once. Step181 builds a readable "start here" layer from the
latest risk, readiness, route, news, sector, and institutional-depth outputs.

It does not create permission. It explains:
  - the current portfolio-level verdict
  - which tickers deserve the first deep review
  - why each one is blocked or conflicted
  - which source file proves the reason
  - what order to read the dashboard in

Outputs:
  deep_decision_desk_questions.csv
  deep_decision_desk_ticker_map.csv
  deep_decision_desk_source_guide.csv
  deep_decision_desk_reading_path.csv
  deep_decision_desk_state.json
  deep_decision_desk_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_QUESTIONS = ROOT / "deep_decision_desk_questions.csv"
OUT_TICKERS = ROOT / "deep_decision_desk_ticker_map.csv"
OUT_SOURCES = ROOT / "deep_decision_desk_source_guide.csv"
OUT_PATH = ROOT / "deep_decision_desk_reading_path.csv"
OUT_STATE = ROOT / "deep_decision_desk_state.json"
OUT_REPORT = ROOT / "deep_decision_desk_report.md"


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


def compact(value: Any, limit: int = 280) -> str:
    text = " ".join(as_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def count_status(df: pd.DataFrame, col: str, needle: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).str.upper().str.contains(needle.upper(), na=False).sum())


def count_exact_status(df: pd.DataFrame, col: str, value: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).str.upper().eq(value.upper()).sum())


def join_unique(values: pd.Series, limit: int = 4) -> str:
    out: list[str] = []
    for value in values.astype(str):
        text = as_text(value)
        if text and text not in out:
            out.append(text)
    return " | ".join(out[:limit])


def status_plain(status: str, stage: str) -> str:
    raw = as_upper(status)
    stg = as_upper(stage)
    if raw == "CONFLICT REVIEW":
        return "Route conflict: old idea and current risk route disagree."
    if raw == "RISK BLOCKED" or "RISK_REPAIR_REQUIRED" in stg:
        return "Risk blocked: do not add exposure until the risk repair gate clears."
    if raw == "BLOCKED":
        return "Blocked after risk repair: monitor, spread, event, or route proof is still missing."
    if "WATCH" in raw or "TRIGGER" in stg:
        return "Watch only: wait for the trigger and source proof."
    return "Research review: read the sources before changing the paper plan."


def first_question_for_gate(gate: str) -> str:
    raw = as_upper(gate)
    if "RISK" in raw:
        return "Does the recommended repair leave this ticker below its single-name risk target?"
    if "PRICE" in raw or "VOLUME" in raw or "MONITOR" in raw:
        return "Has the price, volume, volatility, spread, or shock monitor calmed enough to trust the setup?"
    if "SPREAD" in raw or "TCA" in raw:
        return "Do we have realistic spread and cost evidence before using options or size?"
    if "EVENT" in raw or "NEWS" in raw:
        return "Is the news source reliable enough, and is the ticker-specific read-through proven?"
    if "ROUTE" in raw:
        return "Does the current route agree with the latest risk, event, and option gates?"
    return "What proof must clear before this ticker moves from research to action review?"


def build_ticker_map(cards: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    if cards.empty:
        return pd.DataFrame()

    queue_by_ticker: dict[str, pd.Series] = {}
    if not queue.empty and "ticker" in queue.columns:
        for _, row in queue.iterrows():
            ticker = as_upper(row.get("ticker"))
            if ticker and ticker not in queue_by_ticker:
                queue_by_ticker[ticker] = row

    rows: list[dict[str, Any]] = []
    work = cards.copy()
    if "card_rank" in work.columns:
        work = work.sort_values("card_rank")
    elif "readiness_score" in work.columns:
        work = work.sort_values("readiness_score", ascending=False)

    for _, row in work.iterrows():
        ticker = as_upper(row.get("ticker"))
        qrow = queue_by_ticker.get(ticker)
        status = as_text(row.get("card_status"), "Review")
        stage = as_text(row.get("current_stage"), "NO_STAGE")
        gate = as_text(row.get("first_blocking_gate"), "NO_GATE")
        route = as_text(row.get("route_after_all_gates_clear"), as_text(row.get("action_bias"), "Research only"))
        first_source = as_text(row.get("first_source_to_open"), "NO_SOURCE")
        next_check = as_text(row.get("next_check_1"), "Recheck source freshness and risk gate.")
        if qrow is not None:
            next_check = as_text(qrow.get("what_to_check"), next_check)
            route = as_text(qrow.get("route_after_clear"), route)

        rows.append({
            "read_order": len(rows) + 1,
            "ticker": ticker,
            "simple_status": status_plain(status, stage),
            "card_status": status,
            "current_stage": stage,
            "first_question": first_question_for_gate(gate),
            "first_blocker": gate,
            "first_source_to_open": first_source,
            "what_to_check_next": compact(next_check, 320),
            "route_if_every_gate_clears": compact(route, 300),
            "trigger_to_watch": compact(row.get("trigger_to_watch"), 220),
            "why_this_matters": compact(row.get("subheadline"), 420),
            "deep_reason": compact(
                " | ".join([
                    as_text(row.get("risk_summary")),
                    as_text(row.get("monitor_summary")),
                    as_text(row.get("event_news_summary")),
                    as_text(row.get("sector_portfolio_summary")),
                ]),
                560,
            ),
            "conflict_status": as_text(row.get("decision_route_conflict_status"), "NO_ROUTE_CONFLICT"),
            "do_not_do": compact(row.get("do_not_do"), 320),
            "source_files": as_text(row.get("source_files"), "NO_SOURCE_FILES"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    return pd.DataFrame(rows)


def build_questions(
    cards: pd.DataFrame,
    ticker_map: pd.DataFrame,
    readiness_state: dict[str, Any],
    depth_state: dict[str, Any],
    risk_repair_state: dict[str, Any],
    workflow_state: dict[str, Any],
) -> pd.DataFrame:
    card_rows = int(len(cards))
    risk_blocked = count_exact_status(cards, "card_status", "Risk blocked")
    blocked = count_exact_status(cards, "card_status", "Blocked")
    conflicts = count_exact_status(cards, "card_status", "Conflict review")
    top_tickers = join_unique(ticker_map.get("ticker", pd.Series(dtype=str)).head(5)) if not ticker_map.empty else "NO_DATA"
    top_source = as_text(readiness_state.get("top_next_move_ticker"), "")
    top_next = top_tickers.split(" | ")[0] if top_tickers else top_source

    first_blockers = ""
    if not cards.empty and "first_blocking_gate" in cards.columns:
        blocker_counts = cards["first_blocking_gate"].astype(str).value_counts().head(3)
        first_blockers = " | ".join(f"{idx}: {int(val)}" for idx, val in blocker_counts.items())

    verdict = "Research-only. No new exposure until gates clear."
    if card_rows and risk_blocked + blocked + conflicts >= card_rows:
        verdict = "Research-only. Every current candidate is blocked or conflicted."

    questions = [
        {
            "read_order": 1,
            "question": "Can I act today?",
            "short_answer": verdict,
            "why_it_matters": (
                f"{card_rows} candidates are in the current readiness deck; "
                f"{risk_blocked} are risk blocked, {blocked} are blocked after risk repair, "
                f"and {conflicts} have route conflicts."
            ),
            "where_to_look": "deep_decision_desk_ticker_map.csv; action_readiness_detail_cards.csv",
        },
        {
            "read_order": 2,
            "question": "What should I read first?",
            "short_answer": f"Start with {top_tickers}.",
            "why_it_matters": (
                "These names sit at the top of the current readiness deck. They are not buy signals; "
                "they are the first names to investigate because they define today\'s risk and route problem."
            ),
            "where_to_look": "action_readiness_next_move_queue.csv; action_readiness_detail_cards.csv",
        },
        {
            "read_order": 3,
            "question": "What is blocking the system?",
            "short_answer": first_blockers or "NO_BLOCKER_DATA",
            "why_it_matters": (
                "The first blocker tells you which proof must clear before any option, paper trade, "
                "or portfolio route deserves attention."
            ),
            "where_to_look": "action_readiness_blocker_explainer.csv; action_readiness_gate_matrix.csv",
        },
        {
            "read_order": 4,
            "question": "Is this close to a top quant system?",
            "short_answer": (
                f"Current readiness is {depth_state.get('overall_readiness_pct', 'NO_DATA')}%; "
                f"gap is {depth_state.get('overall_gap_to_top_quant_pct', 'NO_DATA')}%."
            ),
            "why_it_matters": (
                "The dashboard is a strong local research prototype, but the remaining gap is still data truth, "
                "backtest credibility, execution/TCA, and options book risk."
            ),
            "where_to_look": "institutional_depth_module_scorecard.csv; institutional_depth_gap_matrix.csv",
        },
        {
            "read_order": 5,
            "question": "What must I not do?",
            "short_answer": "Do not treat headlines, option routes, or old ticker-room text as permission.",
            "why_it_matters": (
                "Risk, source proof, monitor calm, spread/TCA, event proof, and manual route review must all clear first. "
                "No broker connection and no live orders are enabled."
            ),
            "where_to_look": "action_readiness_manual_checklist.csv; options_tca_no_go_audit.csv",
        },
        {
            "read_order": 6,
            "question": "What changed from a messy dashboard?",
            "short_answer": "Raw tables now have a front-door reading order.",
            "why_it_matters": (
                f"The latest run has {workflow_state.get('queue_rows', 'NO_DATA')} workflow queue rows and "
                f"risk repair status {risk_repair_state.get('overall_status', 'NO_DATA')}. "
                "Step181 turns that into a smaller decision map before you open the raw audit tables."
            ),
            "where_to_look": "deep_decision_desk_reading_path.csv; daily_workflow_queue.csv",
        },
    ]
    out = pd.DataFrame(questions)
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def build_source_guide() -> pd.DataFrame:
    rows = [
        (1, "Start here", "deep_decision_desk_questions.csv", "Six plain-English questions that summarize the day.", "Read before opening raw tables."),
        (2, "Ticker map", "deep_decision_desk_ticker_map.csv", "One row per important ticker: status, first blocker, source, route after clear.", "Use this to pick what to investigate first."),
        (3, "Readiness cards", "action_readiness_detail_cards.csv", "The richer card deck built from Step179.", "Explains current route authority and conflicts."),
        (4, "Blockers", "action_readiness_blocker_explainer.csv", "Plain-English explanation of every open gate.", "Use when a ticker is blocked."),
        (5, "Source trace", "action_readiness_source_trace.csv", "Which file produced each risk, route, monitor, or news signal.", "Use when you ask 'where did this come from?'"),
        (6, "Risk repair", "risk_repair_recommendation_board.csv", "Which tickers need reduction or no new exposure.", "Risk still overrides options."),
        (7, "Monitor shock", "desk_monitor_ticker_state.csv", "Price break, volume spike, volatility shift, spread widening, correlation/news shock, risk breach.", "This is the first source for AMAT/FIX/MU/APP."),
        (8, "Event chain", "event_readthrough_decision_board.csv", "News-to-target and supply-chain read-through logic.", "Useful for SpaceX/RKLB-style theme links, but it is not permission."),
        (9, "Time and option route", "sector_timeframe_option_route.csv", "Short, medium, long, call, put, hedge, or no-option route.", "Read after risk/source gates."),
        (10, "Top quant gap", "institutional_depth_module_scorecard.csv", "Readiness versus top institutional standards.", "Use for development priorities, not trading permission."),
    ]
    out = pd.DataFrame(rows, columns=["read_order", "area", "source_file", "what_it_answers", "when_to_open_it"])
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def build_reading_path(ticker_map: pd.DataFrame) -> pd.DataFrame:
    top = join_unique(ticker_map.get("ticker", pd.Series(dtype=str)).head(4)) if not ticker_map.empty else "NO_DATA"
    rows = [
        (1, "Overall verdict", "Am I allowed to add risk today?", "Read the verdict and the six questions.", "deep_decision_desk_questions.csv", "Do not start with options."),
        (2, "First ticker review", "Which names define today?", f"Open the ticker map for {top}.", "deep_decision_desk_ticker_map.csv", "These are research priorities, not trades."),
        (3, "First blocker", "What must clear first?", "Open the first source file listed for the ticker.", "action_readiness_blocker_explainer.csv", "Do not skip the first blocked gate."),
        (4, "Evidence origin", "Where did the signal come from?", "Open source trace and manual checklist.", "action_readiness_source_trace.csv", "A signal with no source is not proof."),
        (5, "Route after proof", "If all gates clear, what vehicle is even allowed?", "Then read route, option, and timeframe files.", "sector_timeframe_option_route.csv; options_execution_route_matrix.csv", "Risk still dominates calls/puts."),
        (6, "Raw audit", "What are the underlying details?", "Only after the summary, open raw workflow and audit tables.", "daily_workflow_queue.csv; institutional_depth_control_map.csv", "Raw tables are drilldown, not the front door."),
    ]
    out = pd.DataFrame(rows, columns=["read_order", "station", "question", "what_to_do", "source_files", "guardrail"])
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def build_state(
    questions: pd.DataFrame,
    ticker_map: pd.DataFrame,
    source_guide: pd.DataFrame,
    reading_path: pd.DataFrame,
    depth_state: dict[str, Any],
) -> dict[str, Any]:
    if ticker_map.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_DEEP_DECISION_DESK_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }

    blocked_or_conflict = int(
        ticker_map["card_status"].astype(str).isin(["Risk blocked", "Blocked", "Conflict review"]).sum()
    )
    conflicts = int(ticker_map["card_status"].astype(str).eq("Conflict review").sum())
    first = ticker_map.sort_values("read_order").iloc[0]
    verdict = "Research-only. Every current candidate is blocked or conflicted." if blocked_or_conflict >= len(ticker_map) else "Research review required."
    return {
        "date": today_str(),
        "overall_status": "DEEP_DECISION_DESK_ACTIVE",
        "overall_verdict": verdict,
        "question_rows": int(len(questions)),
        "ticker_rows": int(len(ticker_map)),
        "blocked_or_conflict_tickers": blocked_or_conflict,
        "route_conflict_tickers": conflicts,
        "source_rows": int(len(source_guide)),
        "reading_path_rows": int(len(reading_path)),
        "top_ticker": as_text(first.get("ticker")),
        "top_first_source": as_text(first.get("first_source_to_open")),
        "overall_readiness_pct": depth_state.get("overall_readiness_pct", "NO_DATA"),
        "overall_gap_to_top_quant_pct": depth_state.get("overall_gap_to_top_quant_pct", "NO_DATA"),
        "truth": "This is a front-door explanation layer only. It cannot trade, rebalance, or override risk.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "questions": OUT_QUESTIONS.name,
            "ticker_map": OUT_TICKERS.name,
            "source_guide": OUT_SOURCES.name,
            "reading_path": OUT_PATH.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    cards = read_csv_safe(ROOT / "action_readiness_detail_cards.csv")
    queue = read_csv_safe(ROOT / "action_readiness_next_move_queue.csv")
    readiness_state = read_json_safe(ROOT / "action_readiness_state.json", {})
    depth_state = read_json_safe(ROOT / "institutional_depth_state.json", {})
    risk_repair_state = read_json_safe(ROOT / "risk_repair_recommendation_state.json", {})
    workflow_state = read_json_safe(ROOT / "daily_workflow_state.json", {})

    ticker_map = build_ticker_map(cards, queue)
    questions = build_questions(cards, ticker_map, readiness_state, depth_state, risk_repair_state, workflow_state)
    source_guide = build_source_guide()
    reading_path = build_reading_path(ticker_map)
    state = build_state(questions, ticker_map, source_guide, reading_path, depth_state)

    questions.to_csv(OUT_QUESTIONS, index=False)
    ticker_map.to_csv(OUT_TICKERS, index=False)
    source_guide.to_csv(OUT_SOURCES, index=False)
    reading_path.to_csv(OUT_PATH, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Verdict: {state.get('overall_verdict')}\n"
        f"- Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}\n"
        f"- Blocked or conflict tickers: {state.get('blocked_or_conflict_tickers')}\n"
        f"- Gap to top quant: {state.get('overall_gap_to_top_quant_pct')}%\n",
        "## Six front-door questions\n" + df_to_markdown(questions, 20),
        "## Ticker map\n" + df_to_markdown(ticker_map, 30),
        "## Source guide\n" + df_to_markdown(source_guide, 20),
        "## Reading path\n" + df_to_markdown(reading_path, 20),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- This step reduces dashboard confusion. It does not reduce risk.\n"
        "- Options, calls, puts, and old ticker-room text cannot override Step178/179 readiness gates.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 181 - Deep Decision Desk", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 181 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Verdict: {state.get('overall_verdict')}")
    print(f"Ticker rows: {state.get('ticker_rows')}")
    print(f"Blocked/conflict: {state.get('blocked_or_conflict_tickers')}")
    print(f"Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}")
    print("Outputs:")
    for path in [OUT_QUESTIONS, OUT_TICKERS, OUT_SOURCES, OUT_PATH, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
