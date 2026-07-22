#!/usr/bin/env python3
"""
Canyon v9 Step 182 - Ticker Research Memo Desk.

Research-only. No broker connection. No live orders.

Step181 gives the front-door verdict. Step182 turns each ticker into a compact
research memo so the user can understand one name at a time:
  - what the ticker/theme is
  - why it matters
  - why it is blocked now
  - what evidence source produced the signal
  - what would change the decision
  - short / medium / long view
  - call / put / no-option answer

This is an explanation layer only. It cannot trade, rebalance, write to a
paper ledger, or override risk.

Outputs:
  ticker_research_memo.csv
  ticker_research_memo_panels.csv
  ticker_research_memo_source_pack.csv
  ticker_research_memo_state.json
  ticker_research_memo_report.md
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


OUT_MEMO = ROOT / "ticker_research_memo.csv"
OUT_PANELS = ROOT / "ticker_research_memo_panels.csv"
OUT_SOURCE_PACK = ROOT / "ticker_research_memo_source_pack.csv"
OUT_STATE = ROOT / "ticker_research_memo_state.json"
OUT_REPORT = ROOT / "ticker_research_memo_report.md"


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


def first_row(df: pd.DataFrame, ticker: str, ticker_col: str = "ticker") -> pd.Series | None:
    if df.empty or ticker_col not in df.columns:
        return None
    sub = df[df[ticker_col].astype(str).str.upper().eq(ticker)]
    if sub.empty:
        return None
    return sub.iloc[0]


def rows_for(df: pd.DataFrame, ticker: str, ticker_col: str = "ticker") -> pd.DataFrame:
    if df.empty or ticker_col not in df.columns:
        return pd.DataFrame()
    return df[df[ticker_col].astype(str).str.upper().eq(ticker)].copy()


def join_rows(df: pd.DataFrame, cols: list[str], limit: int = 4) -> str:
    if df.empty:
        return "No rows found."
    parts: list[str] = []
    for _, row in df.head(limit).iterrows():
        bits = []
        for col in cols:
            if col in row.index:
                text = as_text(row.get(col))
                if text:
                    bits.append(text)
        if bits:
            parts.append(" / ".join(bits))
    return " | ".join(parts) if parts else "No usable rows found."


def infer_memo_status(row: pd.Series) -> str:
    status = as_upper(row.get("card_status"))
    stage = as_upper(row.get("current_stage"))
    conflict = as_upper(row.get("conflict_status"))
    if conflict == "ROUTE_CONFLICT_REVIEW" or status == "CONFLICT REVIEW":
        return "Conflict memo"
    if status == "RISK BLOCKED" or "RISK_REPAIR_REQUIRED" in stage:
        return "Risk-blocked memo"
    if status == "BLOCKED":
        return "Blocked memo"
    return "Research memo"


def options_answer(route_row: pd.Series | None, memo_row: pd.Series) -> tuple[str, str, str]:
    route = as_text(memo_row.get("route_if_every_gate_clears"))
    call_answer = ""
    put_answer = ""
    option_side = ""
    option_structure = ""
    no_go = ""
    if route_row is not None:
        call_answer = as_text(route_row.get("call_answer"))
        put_answer = as_text(route_row.get("put_answer"))
        option_side = as_text(route_row.get("option_side"))
        option_structure = as_text(route_row.get("option_structure"))
        no_go = as_text(route_row.get("no_go_conditions"))

    raw = " ".join([route, call_answer, put_answer, option_side, option_structure]).upper()
    if "NO NEW EXPOSURE" in raw or "NO NEW OPTION" in raw or "NO OPTION" in raw:
        answer = "No option now. Use only research/watch until gates clear."
    elif "PUT" in raw or "HEDGE" in raw:
        answer = "Put / hedge research only after monitor, spread/TCA, event proof, and manual route gates clear."
    elif "CALL" in raw:
        answer = "Defined-risk call research only after risk, source, monitor, spread/TCA, and trigger gates clear."
    elif "UNDERLYING" in raw:
        answer = "Underlying-only paper review after non-risk gates clear; no option route now."
    else:
        answer = "No clear option route. Keep it as research-only."

    structure = option_structure or option_side or "NO_OPTION_STRUCTURE"
    why = no_go or call_answer or put_answer or route or "Risk/source gates control the option answer."
    return answer, structure, compact(why, 420)


def horizon_view(route_row: pd.Series | None, decision_card: pd.Series | None, memo_row: pd.Series) -> tuple[str, str, str]:
    if decision_card is not None:
        short = as_text(decision_card.get("short_term_plan"))
        medium = as_text(decision_card.get("medium_term_plan"))
        long = as_text(decision_card.get("long_term_plan"))
        if short or medium or long:
            return short or "Research only", medium or "Research only", long or "Research only"

    route = as_text(memo_row.get("route_if_every_gate_clears"))
    trigger = as_text(memo_row.get("trigger_to_watch"))
    if "No new exposure" in route:
        return (
            "No new exposure; repair risk first.",
            "Reopen only after risk repair and monitor calm.",
            "Keep as research candidate until source and portfolio proof improve.",
        )
    if "Put" in route or "hedge" in route:
        return (
            f"Watch hedge trigger only: {trigger}",
            "Use put/hedge research only if spread/TCA and event proof clear.",
            "Do not convert a short-term hedge idea into a long holding thesis without fundamentals and cycle support.",
        )
    if "Underlying" in route:
        return (
            f"Underlying paper watch only: {trigger}",
            "Review after non-risk gates clear.",
            "Longer-term work requires business quality, valuation, and event proof.",
        )
    return (
        f"Wait for trigger: {trigger}",
        "Research only until risk/source gates clear.",
        "No long-term thesis upgrade without clean evidence and portfolio room.",
    )


def what_is_this(route_row: pd.Series | None, cycle_row: pd.Series | None, ticker: str) -> str:
    sector = ""
    subsector = ""
    cycle = ""
    handoff = ""
    linked = ""
    if route_row is not None:
        sector = as_text(route_row.get("sector"))
        subsector = as_text(route_row.get("subsector"))
        cycle = as_text(route_row.get("subsector_cycle_phase"), as_text(route_row.get("sector_cycle_state")))
        handoff = as_text(route_row.get("leadership_handoff_signal"))
        linked = as_text(route_row.get("linked_sector"))
    if cycle_row is not None:
        sector = sector or as_text(cycle_row.get("sector"))
        subsector = subsector or as_text(cycle_row.get("subsector"))
        cycle = cycle or as_text(cycle_row.get("subsector_cycle_phase"))
        handoff = handoff or as_text(cycle_row.get("leadership_handoff_signal"))
    bits = [ticker]
    if sector:
        bits.append(f"sector={sector}")
    if subsector:
        bits.append(f"subsector={subsector}")
    if cycle:
        bits.append(f"cycle={cycle}")
    if handoff:
        bits.append(f"handoff={handoff}")
    if linked:
        bits.append(f"linked sector={linked}")
    return "; ".join(bits)


def event_answer(event_rows: pd.DataFrame, source_rows: pd.DataFrame) -> str:
    if not event_rows.empty:
        top = event_rows.iloc[0]
        headline = as_text(top.get("headline"))
        decision = as_text(top.get("readthrough_decision"))
        proof = as_text(top.get("proof_required"))
        why = as_text(top.get("why_this_target"))
        return compact(f"{decision}: {headline}. {why} Proof needed: {proof}", 620)
    if not source_rows.empty:
        event_src = source_rows[
            source_rows.get("evidence_area", pd.Series(dtype=str)).astype(str).str.contains("News|event", case=False, na=False)
        ]
        if not event_src.empty:
            return compact(join_rows(event_src, ["source_status", "evidence_summary", "proof_required"], 2), 620)
    return "No mapped event row found. Do not upgrade from a headline."


def source_pack_rows(
    ticker: str,
    source_rows: pd.DataFrame,
    blocker_rows: pd.DataFrame,
    checklist_rows: pd.DataFrame,
    event_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, src in source_rows.iterrows():
        rows.append({
            "ticker": ticker,
            "source_type": "source_trace",
            "source_file": as_text(src.get("source_file")),
            "status": as_text(src.get("source_status")),
            "what_it_says": compact(src.get("evidence_summary"), 420),
            "proof_required": compact(src.get("proof_required"), 420),
        })
    for _, b in blocker_rows.iterrows():
        rows.append({
            "ticker": ticker,
            "source_type": "blocker",
            "source_file": as_text(b.get("source_file")),
            "status": as_text(b.get("gate_status")),
            "what_it_says": compact(b.get("plain_english_reason"), 420),
            "proof_required": compact(b.get("what_would_clear"), 420),
        })
    for _, c in checklist_rows.iterrows():
        rows.append({
            "ticker": ticker,
            "source_type": "manual_check",
            "source_file": as_text(c.get("source_file")),
            "status": as_text(c.get("check_status")),
            "what_it_says": compact(c.get("why_this_check_exists"), 420),
            "proof_required": compact(c.get("pass_condition"), 420),
        })
    for _, e in event_rows.head(5).iterrows():
        rows.append({
            "ticker": ticker,
            "source_type": "event_readthrough",
            "source_file": as_text(e.get("source_files"), "event_readthrough_decision_board.csv"),
            "status": as_text(e.get("readthrough_decision")),
            "what_it_says": compact(f"{as_text(e.get('headline'))} -> {as_text(e.get('target_ticker'))}: {as_text(e.get('why_this_target'))}", 420),
            "proof_required": compact(e.get("proof_required"), 420),
        })
    return rows


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ticker_map = read_csv_safe(ROOT / "deep_decision_desk_ticker_map.csv")
    cards = read_csv_safe(ROOT / "action_readiness_detail_cards.csv")
    panels = read_csv_safe(ROOT / "action_readiness_detail_card_panels.csv")
    source_trace = read_csv_safe(ROOT / "action_readiness_source_trace.csv")
    blockers = read_csv_safe(ROOT / "action_readiness_blocker_explainer.csv")
    checklist = read_csv_safe(ROOT / "action_readiness_manual_checklist.csv")
    route = read_csv_safe(ROOT / "sector_timeframe_option_route.csv")
    decision_cards = read_csv_safe(ROOT / "ticker_decision_cards.csv")
    event_board = read_csv_safe(ROOT / "event_readthrough_decision_board.csv")
    event_ranking = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    cycle = read_csv_safe(ROOT / "subsector_ticker_cycle_map.csv")
    why_not_more = read_csv_safe(ROOT / "institutional_optimizer_why_not_more.csv")

    if ticker_map.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    memo_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    pack_rows: list[dict[str, Any]] = []

    for _, base in ticker_map.sort_values("read_order").iterrows():
        ticker = as_upper(base.get("ticker"))
        if not ticker:
            continue
        card = first_row(cards, ticker)
        route_row = first_row(route, ticker)
        decision_card = first_row(decision_cards, ticker)
        cycle_row = first_row(cycle, ticker)
        why_row = first_row(why_not_more, ticker)
        src_rows = rows_for(source_trace, ticker)
        blocker_rows = rows_for(blockers, ticker)
        checklist_rows = rows_for(checklist, ticker)
        panel_src_rows = rows_for(panels, ticker)
        event_rows = rows_for(event_board, ticker, "target_ticker")
        if event_rows.empty:
            event_rows = rows_for(event_ranking, ticker)

        memo_status = infer_memo_status(base)
        short_view, medium_view, long_view = horizon_view(route_row, decision_card, base)
        opt_answer, opt_structure, opt_reason = options_answer(route_row, base)
        event_text = event_answer(event_rows, src_rows)
        source_count = len(src_rows) + len(blocker_rows) + len(checklist_rows) + len(event_rows)

        risk_answer = compact(
            join_rows(src_rows[src_rows.get("evidence_area", pd.Series(dtype=str)).astype(str).str.contains("Risk", case=False, na=False)],
                      ["source_status", "evidence_summary", "proof_required"], 2)
            if not src_rows.empty else as_text(base.get("simple_status")),
            620,
        )
        source_answer = compact(
            join_rows(src_rows, ["evidence_area", "source_file", "source_status"], 5),
            520,
        )
        blocker_answer = compact(
            join_rows(blocker_rows, ["gate_name", "gate_status", "plain_english_reason"], 4)
            if not blocker_rows.empty else as_text(base.get("why_this_matters")),
            700,
        )
        checklist_answer = compact(
            join_rows(checklist_rows, ["check_order", "check_name", "check_status", "source_file", "pass_condition"], 5),
            700,
        )
        sector_answer = what_is_this(route_row, cycle_row, ticker)
        if why_row is not None:
            sector_answer = compact(
                f"{sector_answer}; why not more={as_text(why_row.get('primary_reason_not_more'), as_text(why_row.get('what_would_allow_more')))}",
                520,
            )

        memo_rows.append({
            "memo_rank": len(memo_rows) + 1,
            "ticker": ticker,
            "memo_status": memo_status,
            "current_verdict": as_text(base.get("simple_status")),
            "what_is_this": sector_answer,
            "why_it_matters": compact(base.get("why_this_matters"), 620),
            "why_blocked_now": blocker_answer,
            "first_source_to_open": as_text(base.get("first_source_to_open")),
            "what_source_says": source_answer,
            "what_would_change_my_mind": compact(base.get("what_to_check_next"), 620),
            "short_term_view": compact(short_view, 360),
            "medium_term_view": compact(medium_view, 360),
            "long_term_view": compact(long_view, 360),
            "options_answer": opt_answer,
            "options_structure_after_clear": opt_structure,
            "options_reason": opt_reason,
            "risk_answer": risk_answer,
            "news_event_answer": event_text,
            "sector_cycle_answer": sector_answer,
            "manual_checklist_answer": checklist_answer,
            "trigger_to_watch": as_text(base.get("trigger_to_watch")),
            "conflict_status": as_text(base.get("conflict_status"), "NO_ROUTE_CONFLICT"),
            "do_not_do": as_text(base.get("do_not_do")),
            "source_count": source_count,
            "source_files": as_text(base.get("source_files")),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

        panel_specs = [
            ("Current verdict", memo_status, as_text(base.get("simple_status")), compact(base.get("why_this_matters"), 520), "deep_decision_desk_ticker_map.csv"),
            ("What this is", "Context", sector_answer, "Sector, subsector, cycle, and linked-theme context.", "sector_timeframe_option_route.csv; subsector_ticker_cycle_map.csv"),
            ("Why blocked now", as_text(base.get("card_status")), as_text(base.get("first_blocker")), blocker_answer, "action_readiness_blocker_explainer.csv"),
            ("What source proves it", "Source trail", as_text(base.get("first_source_to_open")), source_answer, "action_readiness_source_trace.csv"),
            ("What would change my mind", "Proof needed", as_text(base.get("what_to_check_next")), checklist_answer, "action_readiness_manual_checklist.csv"),
            ("Short / medium / long", "Horizon split", f"Short: {short_view}", f"Medium: {medium_view} | Long: {long_view}", "ticker_decision_cards.csv; sector_timeframe_option_route.csv"),
            ("Options answer", "Vehicle", opt_answer, f"Structure after clear: {opt_structure}. Reason: {opt_reason}", "sector_timeframe_option_route.csv; options_execution_route_matrix.csv"),
            ("News and read-through", "Event proof", event_text, "A mapped headline is context until event-time reaction and causal chain are verified.", "event_readthrough_decision_board.csv; event_readthrough_target_ranking.csv"),
        ]
        for order, (panel, panel_status, headline, detail, sources) in enumerate(panel_specs, 1):
            panel_rows.append({
                "ticker": ticker,
                "panel_order": order,
                "panel": panel,
                "panel_status": panel_status,
                "headline": compact(headline, 260),
                "detail": compact(detail, 720),
                "source_files": sources,
                "research_only": True,
                "no_broker_connection": True,
                "no_live_orders": True,
            })

        pack_rows.extend(source_pack_rows(ticker, src_rows, blocker_rows, checklist_rows, event_rows))
        if panel_src_rows.empty:
            continue

    source_pack = pd.DataFrame(pack_rows)
    if not source_pack.empty:
        source_pack.insert(0, "source_rank", range(1, len(source_pack) + 1))
        source_pack["research_only"] = True
        source_pack["no_broker_connection"] = True
        source_pack["no_live_orders"] = True

    return pd.DataFrame(memo_rows), pd.DataFrame(panel_rows), source_pack


def build_state(memo: pd.DataFrame, panels: pd.DataFrame, source_pack: pd.DataFrame) -> dict[str, Any]:
    if memo.empty:
        return {
            "date": today_str(),
            "overall_status": "NO_TICKER_RESEARCH_MEMO_DATA",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        }
    status_counts = memo["memo_status"].value_counts().to_dict()
    return {
        "date": today_str(),
        "overall_status": "TICKER_RESEARCH_MEMO_ACTIVE",
        "memo_rows": int(len(memo)),
        "panel_rows": int(len(panels)),
        "source_pack_rows": int(len(source_pack)),
        "conflict_memos": int(status_counts.get("Conflict memo", 0)),
        "risk_blocked_memos": int(status_counts.get("Risk-blocked memo", 0)),
        "blocked_memos": int(status_counts.get("Blocked memo", 0)),
        "top_ticker": as_text(memo.sort_values("memo_rank").iloc[0].get("ticker")),
        "top_first_source": as_text(memo.sort_values("memo_rank").iloc[0].get("first_source_to_open")),
        "truth": "Ticker memos are explanation surfaces only. They cannot create permission or override gates.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "outputs": {
            "memo": OUT_MEMO.name,
            "panels": OUT_PANELS.name,
            "source_pack": OUT_SOURCE_PACK.name,
            "report": OUT_REPORT.name,
        },
    }


def write_outputs() -> dict[str, Any]:
    memo, panels, source_pack = build_outputs()
    state = build_state(memo, panels, source_pack)

    memo.to_csv(OUT_MEMO, index=False)
    panels.to_csv(OUT_PANELS, index=False)
    source_pack.to_csv(OUT_SOURCE_PACK, index=False)
    write_json(OUT_STATE, state)

    memo_cols = [c for c in [
        "memo_rank", "ticker", "memo_status", "current_verdict",
        "what_is_this", "why_blocked_now", "first_source_to_open",
        "short_term_view", "medium_term_view", "long_term_view",
        "options_answer", "news_event_answer",
    ] if c in memo.columns]
    panel_cols = [c for c in [
        "ticker", "panel_order", "panel", "panel_status",
        "headline", "detail", "source_files",
    ] if c in panels.columns]

    sections = [
        "## Command conclusion\n"
        f"- Overall status: {state.get('overall_status')}\n"
        f"- Memo rows: {state.get('memo_rows')}\n"
        f"- Panel rows: {state.get('panel_rows')}\n"
        f"- Source rows: {state.get('source_pack_rows')}\n"
        f"- Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}\n",
        "## Ticker memos\n" + df_to_markdown(memo[memo_cols] if memo_cols else memo, 30),
        "## Memo panels\n" + df_to_markdown(panels[panel_cols] if panel_cols else panels, 120),
        "## Source pack\n" + df_to_markdown(source_pack, 120),
        "## Guardrails\n"
        "- Research-only; no broker connection; no live orders.\n"
        "- Short/medium/long views are research workflow labels, not trade instructions.\n"
        "- Calls/puts are not allowed unless every risk, source, monitor, spread/TCA, event, and manual route gate clears.\n",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 182 - Ticker Research Memo Desk", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Step 182 complete.")
    print(f"Status: {state.get('overall_status')}")
    print(f"Memo rows: {state.get('memo_rows')}")
    print(f"Panels: {state.get('panel_rows')}")
    print(f"Source pack rows: {state.get('source_pack_rows')}")
    print(f"Top ticker/source: {state.get('top_ticker')} / {state.get('top_first_source')}")
    print("Outputs:")
    for path in [OUT_MEMO, OUT_PANELS, OUT_SOURCE_PACK, OUT_STATE, OUT_REPORT]:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
