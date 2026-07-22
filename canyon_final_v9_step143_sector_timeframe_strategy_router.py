#!/usr/bin/env python3
"""
Canyon v9 - Step 143: Sector-Aware Timeframe Strategy Router
============================================================

Research-only. No broker connection. No live orders.

This step takes the Step142 sector cycle/linkage map and injects it into the
short-term / medium-term / long-term decision layer from Step128. The goal is to
answer, ticker by ticker:

  - Does the sector cycle support this idea or warn against it?
  - Is the best route short-term, medium-term, long-term, or risk-first only?
  - Are options allowed, blocked, or only a hedge research route?
  - Which source files created the conclusion?

Outputs:
  sector_timeframe_route.csv
  sector_timeframe_ticker_detail.csv
  sector_timeframe_option_route.csv
  sector_timeframe_router_state.json
  sector_timeframe_router_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    df_to_markdown,
    now_str,
    read_csv_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_SUMMARY = ROOT / "ticker_timeframe_summary.csv"
IN_MATRIX = ROOT / "timeframe_decision_matrix.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_STRATEGY = ROOT / "strategy_route_playbook.csv"
IN_SECTOR_CYCLE = ROOT / "sector_cycle_state.csv"
IN_SECTOR_LINKAGE = ROOT / "key_sector_linkage.csv"
IN_SUBSECTOR_CYCLE = ROOT / "subsector_ticker_cycle_map.csv"
IN_RISK_GATE = ROOT / "final_risk_gate.csv"
IN_RISK_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"

OUT_ROUTE = ROOT / "sector_timeframe_route.csv"
OUT_DETAIL = ROOT / "sector_timeframe_ticker_detail.csv"
OUT_OPTIONS = ROOT / "sector_timeframe_option_route.csv"
OUT_STATE = ROOT / "sector_timeframe_router_state.json"
OUT_REPORT = ROOT / "sector_timeframe_router_report.md"


SOURCE_STACK = (
    "ticker_timeframe_summary.csv; timeframe_decision_matrix.csv; "
    "options_playbook.csv; strategy_route_playbook.csv; sector_cycle_state.csv; "
    "key_sector_linkage.csv; subsector_ticker_cycle_map.csv; final_risk_gate.csv; "
    "event_research_dossier.csv"
)


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def norm_sector(value: Any) -> str:
    raw = text(value)
    aliases = {
        "Healthcare": "Health Care",
        "Health Care": "Health Care",
        "Tech": "Technology",
        "Consumer Disc": "Consumer Discretionary",
        "Communication": "Communication Services",
        "Semiconductor": "Semiconductors",
    }
    return aliases.get(raw, raw or "Unknown")


def contains_ticker(value: Any, ticker: str) -> bool:
    raw = text(value).upper()
    target = text(ticker).upper()
    if not raw or not target:
        return False
    tokens = [x.strip().upper() for x in raw.replace(";", ",").split(",")]
    return target in tokens


def cycle_adjustment(cycle_state: str) -> dict[str, float]:
    state = text(cycle_state).lower()
    if "crowded leadership" in state:
        return {"short": -4.0, "medium": -2.0, "long": -2.0, "label": "cycle strong but crowded"}
    if "leadership expansion" in state:
        return {"short": 8.0, "medium": 10.0, "long": 6.0, "label": "sector leadership expansion"}
    if "leadership" in state:
        return {"short": 6.0, "medium": 8.0, "long": 5.0, "label": "sector leadership"}
    if "early improvement" in state:
        return {"short": 4.0, "medium": 8.0, "long": 4.0, "label": "sector early improvement"}
    if "fading" in state:
        return {"short": -6.0, "medium": -4.0, "long": -2.0, "label": "sector fading"}
    if "downcycle" in state or "laggard" in state:
        return {"short": -8.0, "medium": -6.0, "long": -4.0, "label": "sector downcycle"}
    if "event pressure" in state:
        return {"short": -6.0, "medium": -6.0, "long": -3.0, "label": "sector event pressure"}
    return {"short": 0.0, "medium": 0.0, "long": 0.0, "label": "sector neutral"}


def timeframe_decision(score: float, timeframe: str, risk_action: str, event_gate: str, cycle_state: str) -> str:
    risk_u = text(risk_action).upper()
    event_u = text(event_gate).upper()
    cycle_l = text(cycle_state).lower()
    if any(x in risk_u for x in ["REDUCE_ONLY", "BLOCK", "NO_NEW"]):
        return "Risk first - no new exposure"
    if "SIZE_DOWN" in risk_u:
        return "Tiny research only"
    if "MISSING" in event_u or "REVIEW" in event_u:
        return "Wait for event check"
    if "late-cycle" in cycle_l or "chase risk" in cycle_l:
        return "Wait for pullback or de-risking"
    if "catch-up handoff" in cycle_l:
        if timeframe in {"Medium-term", "Long-term"} and score >= 45:
            return "Handoff research watch"
        if timeframe == "Short-term":
            return "Wait for price confirmation"
    if "crowded leadership" in cycle_l:
        return "Wait for pullback or de-risking"
    if timeframe == "Short-term":
        if score >= 48:
            return "Short-term watch"
        return "No short-term setup"
    if timeframe == "Medium-term":
        if score >= 52:
            return "Medium-term research"
        return "No medium-term setup"
    if score >= 50:
        return "Long-term research"
    return "No long-term thesis yet"


def best_horizon(short_score: float, med_score: float, long_score: float) -> str:
    scores = {
        "Short-term": short_score,
        "Medium-term": med_score,
        "Long-term": long_score,
    }
    return max(scores, key=scores.get)


def load_first_by_ticker(path: Path) -> pd.DataFrame:
    df = read_csv_safe(path)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    return df.copy().drop_duplicates(subset=["ticker"], keep="first")


def sector_cycle_for_ticker(ticker: str, sector: str, cycle: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if cycle.empty:
        return None, None
    work = cycle.copy()
    work["sector"] = work.get("sector", pd.Series(dtype=str)).map(norm_sector)
    primary = work[work["sector"].astype(str).eq(norm_sector(sector))].head(1)
    primary_row = primary.iloc[0] if not primary.empty else None

    mention_cols = [c for c in ["top_news_tickers", "top_alpha_names", "top_portfolio_tickers"] if c in work.columns]
    linked = pd.DataFrame()
    if mention_cols:
        mask = pd.Series(False, index=work.index)
        for col in mention_cols:
            mask = mask | work[col].map(lambda x: contains_ticker(x, ticker))
        linked = work[mask].copy()
    if not linked.empty:
        linked["cycle_score"] = pd.to_numeric(linked.get("cycle_score", 0), errors="coerce").fillna(0)
        if primary_row is not None:
            linked = linked[linked["sector"].astype(str) != text(primary_row.get("sector"))]
        linked = linked.sort_values("cycle_score", ascending=False)
    linked_row = linked.iloc[0] if not linked.empty else None
    return primary_row, linked_row


def subsector_overlay_for_ticker(ticker: str, subsector_cycle: pd.DataFrame) -> pd.Series | None:
    if subsector_cycle.empty or "ticker" not in subsector_cycle.columns:
        return None
    rows = subsector_cycle[subsector_cycle["ticker"].astype(str).str.upper() == text(ticker).upper()].head(1)
    return rows.iloc[0] if not rows.empty else None


def linkage_for_ticker(ticker: str, linkage: pd.DataFrame) -> str:
    if linkage.empty:
        return ""
    ticker_u = text(ticker).upper()
    mentions = []
    for _, row in linkage.iterrows():
        if contains_ticker(row.get("representative_tickers", ""), ticker_u):
            theme = text(row.get("catalyst_theme")) or text(row.get("primary_sector"))
            sector = text(row.get("linked_sector"))
            action = text(row.get("desk_action"))
            mentions.append(f"{theme} -> {sector}: {action}")
        if len(mentions) >= 3:
            break
    return " | ".join(mentions)


def classify_option_route(row: pd.Series, opt_row: pd.Series | None, cycle_state: str, risk_action: str, event_gate: str) -> tuple[str, str]:
    risk_u = text(risk_action).upper()
    event_u = text(event_gate).upper()
    cycle_l = text(cycle_state).lower()
    permission = text(opt_row.get("option_permission")) if opt_row is not None else text(row.get("option_permission"))
    side = text(opt_row.get("option_side")) if opt_row is not None else text(row.get("option_side"))
    structure = text(opt_row.get("option_structure")) if opt_row is not None else text(row.get("option_structure"))

    if any(x in risk_u for x in ["REDUCE_ONLY", "BLOCK", "NO_NEW"]):
        return "No new option - risk blocked", "Risk gate blocks new exposure before option logic."
    if "MISSING" in event_u or "REVIEW" in event_u:
        return "No new option - event review first", "Event source coverage is not clean enough for option risk."
    if "SIZE_DOWN" in risk_u:
        if "PUT" in side or "HEDGE" in permission:
            return "Put or hedge research only", "Risk is size-down; only defensive option research is allowed."
        return "No new call - size down risk", "Bullish options are blocked while portfolio risk is size-down."
    if ("late-cycle" in cycle_l or "chase risk" in cycle_l) and "CALL" in side:
        return "No call - late-cycle chase risk", "Subsector leadership is hot/late; do not chase fresh calls into strength."
    if "crowded leadership" in cycle_l and "CALL" in side:
        return "No call - crowded sector", "Sector tape is strong, but crowded leadership reduces chase quality."
    if "downcycle" in cycle_l or "fading" in cycle_l:
        if "PUT" in side or "HEDGE" in permission:
            return "Put hedge research only", "Sector cycle is weak; bullish options are not the clean route."
        return "No option - weak sector cycle", "Sector cycle does not support premium buying."
    if "CALL" in side and any(x in permission for x in ["CALL_SPREAD_RESEARCH", "CALL_WATCH_ONLY"]):
        return "Defined-risk call review", structure or "Defined-risk call spread review"
    if "PUT" in side:
        return "Put hedge review", structure or "Put hedge research only"
    return "No option - stock or wait", structure or "No option route selected."


def sector_adjusted_desk_action(best: str, risk_action: str, event_gate: str, cycle_state: str, option_route: str) -> str:
    risk_u = text(risk_action).upper()
    event_u = text(event_gate).upper()
    cycle_l = text(cycle_state).lower()
    if any(x in risk_u for x in ["REDUCE_ONLY", "BLOCK", "NO_NEW"]):
        return "Risk first - reduce or block"
    if "SIZE_DOWN" in risk_u:
        return "Tiny research only"
    if "MISSING" in event_u or "REVIEW" in event_u:
        return "Wait for event evidence"
    if "late-cycle" in cycle_l or "chase risk" in cycle_l:
        if "PUT" in option_route.upper() or "HEDGE" in option_route.upper():
            return "Late-cycle hedge research"
        return "Late-cycle leader - wait/de-risk"
    if "catch-up handoff" in cycle_l:
        return "Handoff watch - research only"
    if "crowded leadership" in cycle_l:
        return "Strong sector but crowded - wait"
    if "downcycle" in cycle_l or "fading" in cycle_l:
        if "PUT" in option_route.upper() or "HEDGE" in option_route.upper():
            return "Defensive hedge research"
        return "Watch only - weak sector"
    if best == "Short-term":
        return "Short-term watch"
    if best == "Medium-term":
        return "Medium-term research"
    return "Long-term research"


def build_router() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = load_first_by_ticker(IN_SUMMARY)
    strategy = load_first_by_ticker(IN_STRATEGY)
    options = load_first_by_ticker(IN_OPTIONS)
    risk = load_first_by_ticker(IN_RISK_GATE)
    queue = load_first_by_ticker(IN_RISK_QUEUE)
    event = load_first_by_ticker(IN_EVENT)
    picks = load_first_by_ticker(IN_PICKS)
    matrix = read_csv_safe(IN_MATRIX)
    cycle = read_csv_safe(IN_SECTOR_CYCLE)
    linkage = read_csv_safe(IN_SECTOR_LINKAGE)
    subsector_cycle = read_csv_safe(IN_SUBSECTOR_CYCLE)

    tickers = sorted(set().union(*[
        set(df["ticker"].dropna().astype(str).tolist())
        for df in [summary, strategy, options, risk, event, picks]
        if not df.empty and "ticker" in df.columns
    ]))

    route_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []

    for ticker in tickers:
        srow = summary[summary["ticker"].astype(str) == ticker].head(1)
        strow = strategy[strategy["ticker"].astype(str) == ticker].head(1)
        orow = options[options["ticker"].astype(str) == ticker].head(1)
        rrow = risk[risk["ticker"].astype(str) == ticker].head(1)
        qrow = queue[queue["ticker"].astype(str) == ticker].head(1)
        erow = event[event["ticker"].astype(str) == ticker].head(1)
        prow = picks[picks["ticker"].astype(str) == ticker].head(1)

        base = srow.iloc[0] if not srow.empty else (prow.iloc[0] if not prow.empty else pd.Series(dtype=object))
        strategy_base = strow.iloc[0] if not strow.empty else pd.Series(dtype=object)
        option_base = orow.iloc[0] if not orow.empty else None
        risk_base = rrow.iloc[0] if not rrow.empty else pd.Series(dtype=object)
        event_base = erow.iloc[0] if not erow.empty else pd.Series(dtype=object)

        sector = norm_sector(base.get("sector") or strategy_base.get("sector") or risk_base.get("sector") or event_base.get("sector"))
        risk_action = text(risk_base.get("final_risk_action")) or text(base.get("risk_action")) or text(strategy_base.get("risk_action"))
        queue_master_risk = text(qrow.iloc[0].get("master_risk_action")) if not qrow.empty else ""
        master_risk = text(risk_base.get("master_risk_action")) or queue_master_risk
        event_gate = text(event_base.get("event_gate")) or text(base.get("event_gate")) or text(strategy_base.get("event_gate"))
        primary_cycle, linked_cycle = sector_cycle_for_ticker(ticker, sector, cycle)
        primary_state = text(primary_cycle.get("cycle_state")) if primary_cycle is not None else "NO_SECTOR_CYCLE"
        linked_state = text(linked_cycle.get("cycle_state")) if linked_cycle is not None else ""
        linked_sector = text(linked_cycle.get("sector")) if linked_cycle is not None else ""
        adj = cycle_adjustment(primary_state)
        linked_adj = cycle_adjustment(linked_state) if linked_state else {"short": 0.0, "medium": 0.0, "long": 0.0, "label": ""}
        sub_row = subsector_overlay_for_ticker(ticker, subsector_cycle)
        subsector = text(sub_row.get("subsector")) if sub_row is not None else ""
        subsector_phase = text(sub_row.get("subsector_cycle_phase")) if sub_row is not None else ""
        handoff_signal = text(sub_row.get("leadership_handoff_signal")) if sub_row is not None else ""
        subsector_action = text(sub_row.get("subsector_action_bias")) if sub_row is not None else ""
        subsector_label = text(sub_row.get("subsector_adjustment_label")) if sub_row is not None else ""
        option_overlay = text(sub_row.get("option_permission_overlay")) if sub_row is not None else ""
        sub_short_adj = safe_float(sub_row.get("subsector_short_adjustment"), 0.0) if sub_row is not None else 0.0
        sub_med_adj = safe_float(sub_row.get("subsector_medium_adjustment"), 0.0) if sub_row is not None else 0.0
        sub_long_adj = safe_float(sub_row.get("subsector_long_adjustment"), 0.0) if sub_row is not None else 0.0
        combined_cycle_state = " | ".join([x for x in [primary_state, subsector_phase] if x])

        short_base = safe_float(base.get("short_score"), safe_float(strategy_base.get("best_score"), 0.0))
        med_base = safe_float(base.get("medium_score"), 0.0)
        long_base = safe_float(base.get("long_score"), 0.0)
        short_adj = short_base + adj["short"] + linked_adj["short"] * 0.35 + sub_short_adj
        med_adj = med_base + adj["medium"] + linked_adj["medium"] * 0.35 + sub_med_adj
        long_adj = long_base + adj["long"] + linked_adj["long"] * 0.35 + sub_long_adj
        best = best_horizon(short_adj, med_adj, long_adj)

        short_decision = timeframe_decision(short_adj, "Short-term", risk_action, event_gate, combined_cycle_state)
        med_decision = timeframe_decision(med_adj, "Medium-term", risk_action, event_gate, combined_cycle_state)
        long_decision = timeframe_decision(long_adj, "Long-term", risk_action, event_gate, combined_cycle_state)
        option_route, option_reason = classify_option_route(base, option_base, combined_cycle_state, risk_action, event_gate)
        desk_action = sector_adjusted_desk_action(best, risk_action, event_gate, combined_cycle_state, option_route)
        link_context = linkage_for_ticker(ticker, linkage)

        why_parts = [
            f"Primary sector cycle: {primary_state}",
            f"sector adjustment: {adj['label']}",
        ]
        if linked_state:
            why_parts.append(f"linked cycle: {linked_sector} = {linked_state}")
        if subsector_phase:
            why_parts.append(f"subsector: {subsector} = {subsector_phase}")
        if handoff_signal:
            why_parts.append(f"handoff: {handoff_signal}")
        if subsector_label:
            why_parts.append(f"subsector adjustment: {subsector_label}")
        if risk_action:
            why_parts.append(f"risk gate: {risk_action}")
        if event_gate:
            why_parts.append(f"event gate: {event_gate}")
        if link_context:
            why_parts.append(f"linkage: {link_context}")

        route_rows.append({
            "ticker": ticker,
            "sector": sector,
            "sector_cycle_state": primary_state,
            "sector_cycle_score": round(safe_float(primary_cycle.get("cycle_score")) if primary_cycle is not None else 0.0, 2),
            "subsector": subsector,
            "subsector_cycle_phase": subsector_phase,
            "leadership_handoff_signal": handoff_signal,
            "subsector_action_bias": subsector_action,
            "subsector_adjustment_label": subsector_label,
            "linked_sector": linked_sector,
            "linked_sector_cycle_state": linked_state,
            "best_horizon_after_sector": best,
            "sector_adjusted_desk_action": desk_action,
            "short_score_before": round(short_base, 2),
            "short_score_after": round(short_adj, 2),
            "short_decision": short_decision,
            "medium_score_before": round(med_base, 2),
            "medium_score_after": round(med_adj, 2),
            "medium_decision": med_decision,
            "long_score_before": round(long_base, 2),
            "long_score_after": round(long_adj, 2),
            "long_decision": long_decision,
            "option_route": option_route,
            "option_side": text(option_base.get("option_side")) if option_base is not None else text(base.get("option_side")),
            "option_structure": text(option_base.get("option_structure")) if option_base is not None else text(base.get("option_structure")),
            "option_permission_overlay": option_overlay,
            "risk_action": risk_action,
            "master_risk_action": master_risk,
            "event_gate": event_gate,
            "primary_blocker": text(strategy_base.get("primary_blocker")) or option_reason,
            "what_to_watch": text(strategy_base.get("entry_trigger")) or (text(option_base.get("call_trigger")) if option_base is not None else ""),
            "what_would_change": text(strategy_base.get("what_would_change")) or (text(option_base.get("what_would_change")) if option_base is not None else ""),
            "linkage_context": link_context,
            "why": "; ".join(why_parts),
            "source_file": SOURCE_STACK,
            "research_only": True,
        })

        option_rows.append({
            "ticker": ticker,
            "sector": sector,
            "sector_cycle_state": primary_state,
            "subsector": subsector,
            "subsector_cycle_phase": subsector_phase,
            "leadership_handoff_signal": handoff_signal,
            "linked_sector": linked_sector,
            "linked_sector_cycle_state": linked_state,
            "option_route": option_route,
            "option_reason": option_reason,
            "option_permission_overlay": option_overlay,
            "option_permission_before_sector": text(option_base.get("option_permission")) if option_base is not None else text(base.get("option_permission")),
            "option_side": text(option_base.get("option_side")) if option_base is not None else text(base.get("option_side")),
            "option_structure": text(option_base.get("option_structure")) if option_base is not None else text(base.get("option_structure")),
            "call_answer": text(option_base.get("call_answer")) if option_base is not None else "",
            "put_answer": text(option_base.get("put_answer")) if option_base is not None else "",
            "call_trigger": text(option_base.get("call_trigger")) if option_base is not None else "",
            "put_trigger": text(option_base.get("put_trigger")) if option_base is not None else "",
            "no_go_conditions": text(option_base.get("no_go_conditions")) if option_base is not None else "",
            "risk_action": risk_action,
            "event_gate": event_gate,
            "source_file": SOURCE_STACK,
            "research_only": True,
        })

        for timeframe, before, after, decision, base_action in [
            ("Short-term", short_base, short_adj, short_decision, text(base.get("short_action"))),
            ("Medium-term", med_base, med_adj, med_decision, text(base.get("medium_action"))),
            ("Long-term", long_base, long_adj, long_decision, text(base.get("long_action"))),
        ]:
            detail_rows.append({
                "ticker": ticker,
                "sector": sector,
                "timeframe": timeframe,
                "score_before_sector": round(before, 2),
                "sector_adjustment": round(after - before, 2),
                "score_after_sector": round(after, 2),
                "original_action": base_action,
                "sector_adjusted_decision": decision,
                "sector_cycle_state": primary_state,
                "subsector": subsector,
                "subsector_cycle_phase": subsector_phase,
                "leadership_handoff_signal": handoff_signal,
                "linked_sector": linked_sector,
                "linked_sector_cycle_state": linked_state,
                "risk_action": risk_action,
                "event_gate": event_gate,
                "reason": "; ".join(why_parts),
                "source_file": SOURCE_STACK,
                "research_only": True,
            })

    route = pd.DataFrame(route_rows)
    detail = pd.DataFrame(detail_rows)
    opt = pd.DataFrame(option_rows)
    if not route.empty:
        route["_rank"] = (
            pd.to_numeric(route["short_score_after"], errors="coerce").fillna(0)
            + pd.to_numeric(route["medium_score_after"], errors="coerce").fillna(0)
            + pd.to_numeric(route["long_score_after"], errors="coerce").fillna(0)
        )
        action_rank = {
            "Risk first - reduce or block": 0,
            "Tiny research only": 1,
            "Wait for event evidence": 2,
            "Late-cycle leader - wait/de-risk": 3,
            "Late-cycle hedge research": 4,
            "Handoff watch - research only": 5,
            "Strong sector but crowded - wait": 3,
            "Short-term watch": 6,
            "Medium-term research": 7,
            "Long-term research": 8,
            "Defensive hedge research": 9,
            "Watch only - weak sector": 10,
        }
        route["_action_rank"] = route["sector_adjusted_desk_action"].map(action_rank).fillna(9)
        route = route.sort_values(["_action_rank", "_rank"], ascending=[True, False]).drop(columns=["_rank", "_action_rank"]).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "tickers": int(len(route)),
        "sector_supported_tickers": int(route["sector_cycle_state"].ne("NO_SECTOR_CYCLE").sum()) if not route.empty else 0,
        "linked_sector_tickers": int(route["linked_sector"].astype(str).ne("").sum()) if not route.empty else 0,
        "subsector_supported_tickers": int(route.get("subsector_cycle_phase", pd.Series(dtype=str)).astype(str).ne("").sum()) if not route.empty else 0,
        "late_cycle_tickers": int(route.get("subsector_cycle_phase", pd.Series(dtype=str)).astype(str).str.contains("late-cycle|chase risk", case=False, na=False).sum()) if not route.empty else 0,
        "handoff_watch_tickers": int(route.get("leadership_handoff_signal", pd.Series(dtype=str)).astype(str).str.contains("handoff", case=False, na=False).sum()) if not route.empty else 0,
        "risk_first_count": int(route["sector_adjusted_desk_action"].astype(str).str.contains("Risk first|Tiny research", case=False, na=False).sum()) if not route.empty else 0,
        "short_watch_count": int(route["sector_adjusted_desk_action"].astype(str).eq("Short-term watch").sum()) if not route.empty else 0,
        "medium_research_count": int(route["sector_adjusted_desk_action"].astype(str).eq("Medium-term research").sum()) if not route.empty else 0,
        "long_research_count": int(route["sector_adjusted_desk_action"].astype(str).eq("Long-term research").sum()) if not route.empty else 0,
        "defensive_hedge_count": int(route["sector_adjusted_desk_action"].astype(str).eq("Defensive hedge research").sum()) if not route.empty else 0,
        "no_new_option_count": int(opt["option_route"].astype(str).str.contains("No new|No option|No call", case=False, na=False).sum()) if not opt.empty else 0,
        "call_review_count": int(opt["option_route"].astype(str).str.contains("call review", case=False, na=False).sum()) if not opt.empty else 0,
        "put_or_hedge_count": int(opt["option_route"].astype(str).str.contains("put|hedge", case=False, na=False).sum()) if not opt.empty else 0,
        "outputs": {
            "route": OUT_ROUTE.name,
            "detail": OUT_DETAIL.name,
            "options": OUT_OPTIONS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
        "logic": "Sector and subsector cycle modify route selection, but they cannot override risk gates, event checks, liquidity, or no-live-order constraints.",
    }
    return route, detail, opt, state


def main() -> int:
    route, detail, opt, state = build_router()
    route.to_csv(OUT_ROUTE, index=False)
    detail.to_csv(OUT_DETAIL, index=False)
    opt.to_csv(OUT_OPTIONS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Tickers routed: {state.get('tickers', 0)}",
        f"- Sector-supported tickers: {state.get('sector_supported_tickers', 0)}",
        f"- Linked-sector tickers: {state.get('linked_sector_tickers', 0)}",
        f"- Subsector-supported tickers: {state.get('subsector_supported_tickers', 0)}",
        f"- Late-cycle subsector tickers: {state.get('late_cycle_tickers', 0)}",
        f"- Handoff-watch tickers: {state.get('handoff_watch_tickers', 0)}",
        f"- Risk-first or tiny-research routes: {state.get('risk_first_count', 0)}",
        f"- Short-term watch: {state.get('short_watch_count', 0)}",
        f"- Medium-term research: {state.get('medium_research_count', 0)}",
        f"- Long-term research: {state.get('long_research_count', 0)}",
        f"- Defensive hedge research: {state.get('defensive_hedge_count', 0)}",
        f"- No-new-option routes: {state.get('no_new_option_count', 0)}",
        f"- Call review routes: {state.get('call_review_count', 0)}",
        f"- Put or hedge routes: {state.get('put_or_hedge_count', 0)}",
        "",
        "## Sector-Aware Route",
        "",
        df_to_markdown(route, max_rows=60),
        "",
        "## Option Route",
        "",
        df_to_markdown(opt, max_rows=60),
        "",
        "## Ticker Timeframe Detail",
        "",
        df_to_markdown(detail, max_rows=90),
        "",
        "## Product Truth",
        "",
        "This router is research-only. Sector strength can focus attention, but it cannot authorize live trades or override risk gates.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 143 - Sector-Aware Timeframe Strategy Router", sections)

    print(f"wrote {OUT_ROUTE.name} rows={len(route)}")
    print(f"wrote {OUT_DETAIL.name} rows={len(detail)}")
    print(f"wrote {OUT_OPTIONS.name} rows={len(opt)}")
    print(f"risk_first_count={state.get('risk_first_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
