#!/usr/bin/env python3
"""
Canyon v9 Step 209 - Quant Fund Flow Navigator.

Research-only. No broker connection. No live orders.

Step208 defines the institutional operating flow. Step209 makes that flow
actionable: every ticker gets a current state, a first blocker, a next click,
and a plain-English route. This is the dashboard navigation layer, not a broker
or execution engine.

Outputs:
  quant_fund_flow_navigator_state.json
  quant_fund_flow_current_state.csv
  quant_fund_flow_blocker_queue.csv
  quant_fund_flow_next_clicks.csv
  quant_fund_flow_stage_contracts.csv
  quant_fund_flow_pm_command_center.json
  quant_fund_flow_navigator_report.md
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


OUT_STATE = ROOT / "quant_fund_flow_navigator_state.json"
OUT_CURRENT = ROOT / "quant_fund_flow_current_state.csv"
OUT_BLOCKERS = ROOT / "quant_fund_flow_blocker_queue.csv"
OUT_NEXT_CLICKS = ROOT / "quant_fund_flow_next_clicks.csv"
OUT_CONTRACTS = ROOT / "quant_fund_flow_stage_contracts.csv"
OUT_COMMAND = ROOT / "quant_fund_flow_pm_command_center.json"
OUT_REPORT = ROOT / "quant_fund_flow_navigator_report.md"


CURRENT_COLUMNS = [
    "ticker",
    "current_state",
    "operating_mode",
    "can_take_new_risk",
    "first_blocker",
    "next_click",
    "next_action",
    "stock_or_etf_route",
    "option_route",
    "option_side",
    "short_term_route",
    "medium_term_route",
    "long_term_route",
    "why",
    "proof_needed",
    "risk_read",
    "execution_read",
    "news_read",
    "trigger_to_watch",
    "source_trail",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

BLOCKER_COLUMNS = [
    "priority",
    "ticker",
    "blocker_type",
    "blocker",
    "what_to_do",
    "where_to_click",
    "why_it_matters",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

NEXT_CLICK_COLUMNS = [
    "order",
    "page",
    "panel",
    "what_to_read",
    "why_now",
    "done_when",
    "do_not_do",
    "source_files",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

CONTRACT_COLUMNS = [
    "stage",
    "input_contract",
    "output_contract",
    "pass_condition",
    "fail_condition",
    "owner",
    "dashboard_page",
    "active_files",
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


def short(value: Any, limit: int = 340) -> str:
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


def normalize_key(value: Any) -> str:
    return as_text(value, "").upper().replace("-", "_").replace(" ", "_")


def plain_status(value: Any) -> str:
    text = as_text(value, "")
    key = normalize_key(text)
    replacements = {
        "SIZE_DOWN": "risk says smaller size",
        "DATA_GAP": "missing data",
        "NO_GO": "not allowed yet",
        "BLOCKED": "blocked",
        "NEEDS_REVIEW": "needs human review",
        "PENDING_MANUAL_CHECKS": "needs manual checks",
        "WATCH_EVENT_PROOF_FIRST": "watch until event proof is checked",
        "WAIT_EXECUTION_OR_MONITOR_REVIEW": "wait until execution or monitor risk improves",
        "TINY_STOCK_OR_ETF_PAPER_ONLY": "tiny stock or ETF paper review only",
        "NO_OPTION_WAIT": "no options yet",
        "TINY_UNDERLYING_ONLY": "tiny underlying paper review only",
    }
    if key in replacements:
        return replacements[key]
    if "_" in text and key == text:
        return text.replace("_", " ").title()
    return text


def humanize_gate_text(value: Any) -> str:
    text = as_text(value, "")
    if not text:
        return ""

    labels = {
        "master": "portfolio risk",
        "single": "single-stock risk",
        "single_name": "single-stock risk",
        "earnings_gap": "earnings gap risk",
        "kelly": "position-size math",
        "sector": "sector concentration",
        "risk": "risk",
        "execution": "execution data",
        "cost": "trading cost",
        "monitor": "live monitor",
        "spread": "spread data",
        "option_no_go_checks": "option checks",
        "liquidity": "liquidity",
    }
    parts = [p.strip() for p in text.replace("|", ";").split(";") if p.strip()]
    readable: list[str] = []
    for part in parts:
        if "=" in part:
            raw_key, raw_val = part.split("=", 1)
        elif ":" in part:
            raw_key, raw_val = part.split(":", 1)
        else:
            readable.append(plain_status(part))
            continue
        key = raw_key.strip().lower().replace(" ", "_")
        label = labels.get(key, key.replace("_", " "))
        value_text = plain_status(raw_val)
        if key == "option_no_go_checks":
            readable.append(f"{label}: {value_text} checks still block options")
        else:
            readable.append(f"{label}: {value_text}")
    return "; ".join(readable) if readable else plain_status(text)


def first_row_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, pd.Series]:
    if df.empty or ticker_col not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get(ticker_col))
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def collect_tickers(*frames: pd.DataFrame) -> list[str]:
    tickers: set[str] = set()
    for df in frames:
        if df is None or df.empty or "ticker" not in df.columns:
            continue
        for value in df["ticker"].dropna().tolist():
            ticker = clean_ticker(value)
            if ticker:
                tickers.add(ticker)
    return sorted(tickers)


def choose_state(
    ticker: str,
    bridge_row: pd.Series | None,
    proof_row: pd.Series | None,
    promotion_row: pd.Series | None,
    risk_row: pd.Series | None,
    execution_row: pd.Series | None,
    option_row: pd.Series | None,
    card_row: pd.Series | None,
) -> tuple[str, str, str, str, str]:
    """Return state, mode, risk permission, first blocker, next click."""
    proof_need = int(float(proof_row.get("needs_proof_count", 0) or 0)) if proof_row is not None else 0
    bridge_decision = as_text(bridge_row.get("bridge_decision"), "") if bridge_row is not None else ""
    bridge_blocker = as_text(bridge_row.get("first_blocking_gate"), "") if bridge_row is not None else ""
    risk_action = normalize_key(risk_row.get("final_risk_action")) if risk_row is not None else ""
    exec_verdict = normalize_key(execution_row.get("execution_verdict")) if execution_row is not None else ""
    exec_blockers = normalize_key(execution_row.get("primary_blockers")) if execution_row is not None else ""
    promotion_permission = as_text(promotion_row.get("final_permission"), "") if promotion_row is not None else ""
    card_status = normalize_key(card_row.get("card_status")) if card_row is not None else ""
    options_vehicle = normalize_key(option_row.get("best_vehicle_decision")) if option_row is not None else ""

    if proof_need > 0:
        return (
            "Needs outside proof",
            "Proof first",
            "No new risk",
            "Outside proof",
            "Risk",
        )
    if "PM REVIEW" in normalize_key(bridge_blocker) or "BLOCKED" in normalize_key(bridge_decision):
        return (
            "Needs PM review",
            "Review first",
            "No new risk",
            bridge_blocker or "PM review",
            "Risk",
        )
    if risk_action in {"SIZE_DOWN", "BLOCKED", "NO_GO"} or "RISK" in normalize_key(bridge_blocker):
        return (
            "Risk blocked",
            "Risk first",
            "No new risk",
            "Risk budget",
            "Risk",
        )
    if "DATA_GAP" in exec_blockers or "MANUAL" in exec_verdict:
        return (
            "Needs execution proof",
            "Execution proof first",
            "No new risk",
            "Spread or liquidity proof",
            "Risk",
        )
    if options_vehicle in {"NO_OPTION_WAIT", "WATCH_EVENT_PROOF_FIRST", "WAIT_EXECUTION_OR_MONITOR_REVIEW"}:
        return (
            "Watch only",
            "Watchlist",
            "Watch only",
            "Options or event route not ready",
            "Ideas",
        )
    if promotion_permission.lower() in {"do not add", "study only"} or card_status == "BLOCKED":
        return (
            "Study only",
            "Research only",
            "No new risk",
            as_text(promotion_row.get("first_blocker"), "Promotion gate") if promotion_row is not None else "Promotion gate",
            "Ideas",
        )

    max_paper = 0.0
    if bridge_row is not None:
        try:
            max_paper = float(bridge_row.get("max_tiny_paper_review_pct", 0) or 0)
        except Exception:
            max_paper = 0.0
    if max_paper > 0:
        return (
            "Tiny paper review allowed",
            "Paper review",
            "Only tiny paper review",
            "Keep size tiny and manual",
            "Live / Paper",
        )
    return (
        "Research candidate",
        "Research",
        "No new risk until final gate confirms",
        "Final gate confirmation",
        "Ideas",
    )


def build_current_state() -> pd.DataFrame:
    bridge = read_csv_safe(ROOT / "pm_review_final_gate_bridge.csv")
    proof = read_csv_safe(ROOT / "pm_evidence_source_proof_status.csv")
    promotion = read_csv_safe(ROOT / "institutional_promotion_gate.csv")
    risk = read_csv_safe(ROOT / "final_risk_gate.csv")
    execution = read_csv_safe(ROOT / "execution_tca_decision_board.csv")
    options = read_csv_safe(ROOT / "options_trade_permission_summary.csv")
    option_route = read_csv_safe(ROOT / "options_execution_route_matrix.csv")
    cards = read_csv_safe(ROOT / "ticker_decision_cards.csv")
    workflow = read_csv_safe(ROOT / "daily_workflow_ticker_explain.csv")

    bridge_map = first_row_by_ticker(bridge)
    proof_map = first_row_by_ticker(proof)
    promotion_map = first_row_by_ticker(promotion)
    risk_map = first_row_by_ticker(risk)
    execution_map = first_row_by_ticker(execution)
    options_map = first_row_by_ticker(options)
    option_route_map = first_row_by_ticker(option_route)
    cards_map = first_row_by_ticker(cards)
    workflow_map = first_row_by_ticker(workflow)

    tickers = collect_tickers(bridge, proof, promotion, risk, execution, options, option_route, cards)
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        bridge_row = bridge_map.get(ticker)
        proof_row = proof_map.get(ticker)
        promotion_row = promotion_map.get(ticker)
        risk_row = risk_map.get(ticker)
        execution_row = execution_map.get(ticker)
        option_row = options_map.get(ticker)
        option_route_row = option_route_map.get(ticker)
        card_row = cards_map.get(ticker)
        workflow_row = workflow_map.get(ticker)

        current_state, mode, can_risk, first_blocker, next_click = choose_state(
            ticker, bridge_row, proof_row, promotion_row, risk_row, execution_row, option_row, card_row
        )

        proof_needed = ""
        if proof_row is not None:
            proof_needed = as_text(proof_row.get("first_missing_proof"), "")
        if not proof_needed and bridge_row is not None:
            proof_needed = as_text(bridge_row.get("pm_review_status"), "")

        risk_read = ""
        if risk_row is not None:
            risk_read = as_text(risk_row.get("reason_stack"), "")
        if not risk_read and promotion_row is not None:
            risk_read = as_text(promotion_row.get("risk_reason"), "")

        execution_read = ""
        if execution_row is not None:
            execution_read = as_text(execution_row.get("plain_reason"), "")

        news_read = ""
        if promotion_row is not None:
            news_read = as_text(promotion_row.get("news_headline"), "")
        if not news_read and card_row is not None:
            news_read = as_text(card_row.get("top_news_headline"), "")

        option_route_text = ""
        option_side = "None"
        if option_row is not None:
            option_route_text = as_text(option_row.get("best_vehicle_decision"), "")
            option_side = as_text(option_row.get("best_option_side"), "None")
        elif option_route_row is not None:
            option_route_text = as_text(option_route_row.get("final_vehicle_decision"), "")
            option_side = as_text(option_route_row.get("final_option_side"), "None")

        stock_route = "No new exposure"
        if promotion_row is not None:
            stock_route = as_text(promotion_row.get("primary_route_now"), stock_route)
        if not stock_route and card_row is not None:
            stock_route = as_text(card_row.get("primary_route"), "No new exposure")

        next_action = ""
        if proof_row is not None and int(float(proof_row.get("needs_proof_count", 0) or 0)) > 0:
            next_action = as_text(proof_row.get("next_step"), "")
        if not next_action and bridge_row is not None:
            next_action = as_text(bridge_row.get("next_step"), "")
        if not next_action and execution_row is not None:
            next_action = as_text(execution_row.get("next_manual_check"), "")
        if not next_action and promotion_row is not None:
            next_action = as_text(promotion_row.get("next_step"), "")
        if not next_action:
            next_action = "Read the final gate, then fix the first blocker before considering size."

        trigger = ""
        for row in [promotion_row, execution_row, option_route_row, card_row]:
            if row is not None:
                trigger = as_text(row.get("trigger_to_watch"), "")
                if trigger:
                    break

        why_parts = []
        if first_blocker:
            why_parts.append(f"First blocker: {plain_status(first_blocker)}.")
        if proof_needed:
            why_parts.append(f"Proof needed: {short(proof_needed, 160)}.")
        if risk_read:
            why_parts.append(f"Risk: {short(humanize_gate_text(risk_read), 180)}.")
        if execution_read:
            why_parts.append(f"Execution: {short(execution_read, 170)}.")

        source_trail = []
        for row in [bridge_row, proof_row, promotion_row, risk_row, execution_row, option_row, option_route_row, card_row, workflow_row]:
            if row is not None:
                src = as_text(row.get("source_files"), "") or as_text(row.get("source_file"), "")
                if src and src not in source_trail:
                    source_trail.append(src)

        rows.append(guard_flags({
            "ticker": ticker,
            "current_state": current_state,
            "operating_mode": mode,
            "can_take_new_risk": can_risk,
            "first_blocker": plain_status(first_blocker),
            "next_click": next_click,
            "next_action": short(next_action, 260),
            "stock_or_etf_route": short(plain_status(stock_route), 180),
            "option_route": short(plain_status(option_route_text), 180) if option_route_text else "No option route yet",
            "option_side": plain_status(option_side),
            "short_term_route": short(as_text(card_row.get("short_term_plan"), "") if card_row is not None else "", 120),
            "medium_term_route": short(as_text(card_row.get("medium_term_plan"), "") if card_row is not None else "", 120),
            "long_term_route": short(as_text(card_row.get("long_term_plan"), "") if card_row is not None else "", 120),
            "why": short(" ".join(why_parts), 420),
            "proof_needed": short(proof_needed, 260),
            "risk_read": short(humanize_gate_text(risk_read), 260),
            "execution_read": short(execution_read, 260),
            "news_read": short(news_read, 220),
            "trigger_to_watch": short(trigger, 180),
            "source_trail": short("; ".join(source_trail), 420),
        }))

    df = pd.DataFrame(rows, columns=CURRENT_COLUMNS)
    if df.empty:
        return df
    priority = {
        "Needs outside proof": 1,
        "Needs PM review": 2,
        "Risk blocked": 3,
        "Needs execution proof": 4,
        "Watch only": 5,
        "Study only": 6,
        "Tiny paper review allowed": 7,
        "Research candidate": 8,
    }
    df["_sort"] = df["current_state"].map(priority).fillna(99)
    df = df.sort_values(["_sort", "ticker"]).drop(columns=["_sort"])
    return df


def build_blocker_queue(current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    proof_gaps = read_csv_safe(ROOT / "pm_evidence_source_proof_gap_queue.csv")
    final_actions = read_csv_safe(ROOT / "pm_review_final_gate_next_actions.csv")
    execution = read_csv_safe(ROOT / "execution_tca_decision_board.csv")
    options_route = read_csv_safe(ROOT / "options_execution_route_matrix.csv")

    for _, row in proof_gaps.head(30).iterrows() if not proof_gaps.empty else []:
        rows.append(guard_flags({
            "priority": "P0" if as_text(row.get("review_priority"), "").lower() == "high" else "P1",
            "ticker": clean_ticker(row.get("ticker")),
            "blocker_type": "Outside proof",
            "blocker": short(row.get("required_question"), 220),
            "what_to_do": short(row.get("next_step"), 240),
            "where_to_click": "Risk",
            "why_it_matters": "The system cannot accept this evidence until a human source, value, reviewer, and date are filled.",
            "source_files": as_text(row.get("source_files"), "pm_evidence_source_proof_gap_queue.csv"),
        }))

    for _, row in final_actions.head(30).iterrows() if not final_actions.empty else []:
        rows.append(guard_flags({
            "priority": as_text(row.get("priority"), "P1"),
            "ticker": clean_ticker(row.get("ticker")),
            "blocker_type": "Final gate",
            "blocker": short(row.get("bridge_decision"), 180),
            "what_to_do": short(row.get("what_to_do_next"), 240),
            "where_to_click": as_text(row.get("where_to_click"), "Risk"),
            "why_it_matters": short(row.get("why_it_matters"), 240),
            "source_files": as_text(row.get("source_files"), "pm_review_final_gate_next_actions.csv"),
        }))

    if not execution.empty:
        for _, row in execution.head(25).iterrows():
            blockers = as_text(row.get("primary_blockers"), "")
            if "DATA_GAP" in blockers or "MANUAL" in normalize_key(row.get("execution_verdict")):
                rows.append(guard_flags({
                    "priority": "P1",
                    "ticker": clean_ticker(row.get("ticker")),
                    "blocker_type": "Execution / liquidity",
                    "blocker": short(row.get("plain_reason"), 220),
                    "what_to_do": short(row.get("next_manual_check"), 240),
                    "where_to_click": "Risk",
                    "why_it_matters": "Execution cost can erase a signal before the research idea ever becomes useful.",
                    "source_files": as_text(row.get("source_files"), "execution_tca_decision_board.csv"),
                }))

    if not options_route.empty:
        for _, row in options_route.head(25).iterrows():
            no_go = int(float(row.get("no_go_count", 0) or 0))
            if no_go > 0:
                rows.append(guard_flags({
                    "priority": "P2",
                    "ticker": clean_ticker(row.get("ticker")),
                    "blocker_type": "Options",
                    "blocker": f"{no_go} option checks still say no.",
                    "what_to_do": short(row.get("required_confirmation"), 260),
                    "where_to_click": "Ideas",
                    "why_it_matters": short(row.get("why_this_route"), 240),
                    "source_files": as_text(row.get("source_files"), "options_execution_route_matrix.csv"),
                }))

    if not current.empty:
        for _, row in current.head(40).iterrows():
            if as_text(row.get("current_state")) in {"Risk blocked", "Needs PM review", "Needs outside proof"}:
                rows.append(guard_flags({
                    "priority": "P1",
                    "ticker": clean_ticker(row.get("ticker")),
                    "blocker_type": as_text(row.get("current_state")),
                    "blocker": short(row.get("first_blocker"), 180),
                    "what_to_do": short(row.get("next_action"), 240),
                    "where_to_click": as_text(row.get("next_click"), "Risk"),
                    "why_it_matters": short(row.get("why"), 260),
                    "source_files": as_text(row.get("source_trail"), "quant_fund_flow_current_state.csv"),
                }))

    queue = pd.DataFrame(rows, columns=BLOCKER_COLUMNS)
    if queue.empty:
        return queue
    queue = queue.drop_duplicates(subset=["ticker", "blocker_type", "blocker"], keep="first")
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    blocker_order = {
        "Outside proof": 0,
        "Final gate": 1,
        "Risk blocked": 2,
        "Needs PM review": 3,
        "Needs outside proof": 4,
        "Execution / liquidity": 5,
        "Options": 6,
    }
    queue["_priority_sort"] = queue["priority"].map(priority_order).fillna(9)
    queue["_blocker_sort"] = queue["blocker_type"].map(blocker_order).fillna(9)
    queue = queue.sort_values(["_priority_sort", "ticker", "_blocker_sort"]).drop(columns=["_priority_sort", "_blocker_sort"])
    return queue


def build_next_clicks(current: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    state_counts = current["current_state"].value_counts().to_dict() if not current.empty else {}
    proof_count = int(state_counts.get("Needs outside proof", 0))
    risk_count = int(state_counts.get("Risk blocked", 0))
    exec_count = int(state_counts.get("Needs execution proof", 0))
    watch_count = int(state_counts.get("Watch only", 0))
    top_ticker = as_text(blockers.iloc[0].get("ticker"), "the first blocked ticker") if not blockers.empty else "the highest-priority ticker"

    rows = [
        {
            "order": 1,
            "page": "Home",
            "panel": "Today Flow Navigator",
            "what_to_read": "Read the operating mode, first blocker, and first ticker.",
            "why_now": "This is the control tower. It prevents jumping into random tables.",
            "done_when": "You can say what the system wants fixed first.",
            "do_not_do": "Do not start with options or new ideas if the mode says proof or risk first.",
            "source_files": "quant_fund_flow_pm_command_center.json",
        },
        {
            "order": 2,
            "page": "Risk",
            "panel": "Source Proof Desk",
            "what_to_read": f"Start with {top_ticker}; {proof_count} tickers still need outside proof.",
            "why_now": "Unverified proof is the biggest blocker before PM acceptance.",
            "done_when": "Source name, observed value, reviewer, and review date are filled for the first proof item.",
            "do_not_do": "Do not accept model-generated text as proof.",
            "source_files": "pm_evidence_source_proof_gap_queue.csv",
        },
        {
            "order": 3,
            "page": "Risk",
            "panel": "Risk Desk",
            "what_to_read": f"Check the {risk_count} risk-blocked tickers and whether they need size-down or no exposure.",
            "why_now": "Risk vetoes signals, news, and options.",
            "done_when": "You know whether the idea can be tiny paper, watch-only, or blocked.",
            "do_not_do": "Do not let a good headline override VaR, concentration, or event gap risk.",
            "source_files": "final_risk_gate.csv; institutional_risk_master_gate outputs",
        },
        {
            "order": 4,
            "page": "Risk",
            "panel": "Execution Cost / Liquidity",
            "what_to_read": f"Check spread and liquidity for {exec_count} execution-proof tickers.",
            "why_now": "A signal can disappear after spread, market impact, and fill risk.",
            "done_when": "Manual quote or better intraday source clears the spread/liquidity data gap.",
            "do_not_do": "Do not size a ticker with missing spread or liquidity proof.",
            "source_files": "execution_tca_decision_board.csv; execution_cost_model.csv",
        },
        {
            "order": 5,
            "page": "Ideas",
            "panel": "Options Route",
            "what_to_read": f"Only after proof/risk/execution gates: review {watch_count} watch-only option routes.",
            "why_now": "Calls and puts are route decisions, not substitutes for proof.",
            "done_when": "Each ticker says no option, call research, put research, spread only, or wait for trigger.",
            "do_not_do": "Do not search for weekly calls or puts while a ticker is blocked.",
            "source_files": "options_trade_permission_summary.csv; options_execution_route_matrix.csv",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=NEXT_CLICK_COLUMNS)


def build_stage_contracts() -> pd.DataFrame:
    stages = read_csv_safe(ROOT / "quant_fund_operating_flow_stages.csv")
    if stages.empty:
        rows = [
            {
                "stage": "Data",
                "input_contract": "A source file must exist and have a current timestamp.",
                "output_contract": "Clean data status.",
                "pass_condition": "Fresh enough to use today.",
                "fail_condition": "Missing or stale data.",
                "owner": "Data engineer",
                "dashboard_page": "System",
                "active_files": "data_reliability_scorecard.csv",
            }
        ]
    else:
        rows = []
        for _, row in stages.iterrows():
            rows.append({
                "stage": as_text(row.get("stage_name")),
                "input_contract": short(row.get("primary_inputs"), 260),
                "output_contract": short(row.get("primary_outputs"), 260),
                "pass_condition": short(row.get("hard_gate"), 260),
                "fail_condition": short(row.get("if_gate_fails"), 260),
                "owner": as_text(row.get("owner_role"), ""),
                "dashboard_page": as_text(row.get("app_page"), ""),
                "active_files": short(row.get("active_files"), 340),
            })
    return pd.DataFrame([guard_flags(r) for r in rows], columns=CONTRACT_COLUMNS)


def build_command_center(current: pd.DataFrame, blockers: pd.DataFrame, next_clicks: pd.DataFrame) -> dict[str, Any]:
    counts = current["current_state"].value_counts().to_dict() if not current.empty else {}
    top_blocker = blockers.iloc[0].to_dict() if not blockers.empty else {}
    first_page = as_text(top_blocker.get("where_to_click"), "Home")
    first_ticker = as_text(top_blocker.get("ticker"), "No ticker")
    first_action = as_text(top_blocker.get("what_to_do"), "Run the daily system, then read the first blocker.")
    proof_count = int(counts.get("Needs outside proof", 0))
    risk_count = int(counts.get("Risk blocked", 0))
    exec_count = int(counts.get("Needs execution proof", 0))

    if proof_count > 0:
        mode = "Proof first"
        can_risk = "No new risk"
    elif risk_count > 0:
        mode = "Risk first"
        can_risk = "No new risk"
    elif exec_count > 0:
        mode = "Execution proof first"
        can_risk = "No new risk until spread and liquidity are checked"
    else:
        mode = "Research review"
        can_risk = "Only after final gate confirms"

    plain_answer = (
        f"Today mode is {mode}. Start on {first_page} with {first_ticker}. "
        f"First action: {short(first_action, 220)}"
    )

    return {
        "date": today_str(),
        "status": "Active",
        "today_mode": mode,
        "can_take_new_risk": can_risk,
        "ticker_count": int(len(current)),
        "blocker_count": int(len(blockers)),
        "proof_first_count": proof_count,
        "risk_blocked_count": risk_count,
        "execution_proof_count": exec_count,
        "first_page": first_page,
        "first_ticker": first_ticker,
        "first_action": first_action,
        "plain_answer": plain_answer,
        "next_clicks": next_clicks.head(5).to_dict("records") if not next_clicks.empty else [],
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    current = build_current_state()
    blockers = build_blocker_queue(current)
    next_clicks = build_next_clicks(current, blockers)
    contracts = build_stage_contracts()
    command = build_command_center(current, blockers, next_clicks)
    state = {
        "date": today_str(),
        "status": "Active",
        "ticker_count": int(len(current)),
        "blocker_count": int(len(blockers)),
        "next_click_count": int(len(next_clicks)),
        "stage_contract_count": int(len(contracts)),
        "today_mode": command["today_mode"],
        "first_page": command["first_page"],
        "first_ticker": command["first_ticker"],
        "can_take_new_risk": command["can_take_new_risk"],
        "plain_answer": command["plain_answer"],
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    return current, blockers, next_clicks, contracts, state, command


def main() -> None:
    current, blockers, next_clicks, contracts, state, command = build_outputs()
    current.to_csv(OUT_CURRENT, index=False)
    blockers.to_csv(OUT_BLOCKERS, index=False)
    next_clicks.to_csv(OUT_NEXT_CLICKS, index=False)
    contracts.to_csv(OUT_CONTRACTS, index=False)
    write_json(OUT_STATE, state)
    write_json(OUT_COMMAND, command)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## PM Command Center\n\n" + command["plain_answer"],
        "## Current State By Ticker\n\n" + df_to_markdown(current.head(80)),
        "## Blocker Queue\n\n" + df_to_markdown(blockers.head(120)),
        "## Next Clicks\n\n" + df_to_markdown(next_clicks.head(20)),
        "## Stage Contracts\n\n" + df_to_markdown(contracts.head(40)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Quant Fund Flow Navigator", sections)
    print(
        "Step209 complete: "
        f"{len(current)} ticker states, {len(blockers)} blockers, "
        f"{len(next_clicks)} next-click rows, {len(contracts)} contracts."
    )


if __name__ == "__main__":
    main()
