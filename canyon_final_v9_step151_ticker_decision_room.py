#!/usr/bin/env python3
"""
Canyon v9 - Step 151: Ticker Decision Room
==========================================

Research-only. No broker connection. No live orders.

Step151 builds one simple, source-backed decision room per ticker. It combines
conditional action tickets, risk gates, option route, sector cycle, event/news
state, supply-chain read-through, monitor alerts, and conflict evidence.

Outputs:
  ticker_decision_room.csv
  ticker_decision_room_news.csv
  ticker_decision_room_state.json
  ticker_decision_room_report.md
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

IN_TICKETS = ROOT / "conditional_action_tickets.csv"
IN_CANDIDATES = ROOT / "gate_clear_candidate_ranking.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"
IN_RISK = ROOT / "final_risk_gate.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_EVIDENCE = ROOT / "ticker_evidence_summary.csv"
IN_CONFLICT = ROOT / "decision_conflict_summary.csv"
IN_MONITOR = ROOT / "desk_monitor_events.csv"
IN_NEWS_TARGETS = ROOT / "news_impact_targets.csv"
IN_NEWS_READTHROUGH = ROOT / "news_supply_chain_readthrough.csv"
IN_EVENT_RELIABILITY = ROOT / "event_signal_reliability_adjusted_panel.csv"
IN_THEME = ROOT / "theme_candidate_enrichment.csv"

OUT_ROOM = ROOT / "ticker_decision_room.csv"
OUT_NEWS = ROOT / "ticker_decision_room_news.csv"
OUT_STATE = ROOT / "ticker_decision_room_state.json"
OUT_REPORT = ROOT / "ticker_decision_room_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def shorten(value: Any, limit: int = 520) -> str:
    raw = text(value)
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "..."


def one_by_ticker(df: pd.DataFrame, key: str = "ticker") -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[key] = out[key].astype(str).str.upper().str.strip()
    out = out[out[key] != ""]
    if out.empty:
        return pd.DataFrame()
    return out.drop_duplicates(key, keep="first").set_index(key)


def row_at(indexed: pd.DataFrame, ticker: str) -> pd.Series:
    if indexed.empty or ticker not in indexed.index:
        return pd.Series(dtype=object)
    row = indexed.loc[ticker]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def normalize_ticker_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[column] = out[column].astype(str).str.upper().str.strip()
    out = out[out[column] != ""]
    return out


def contains(value: Any, token: str) -> bool:
    return token.upper() in upper(value)


def first_nonempty(*values: Any) -> str:
    for value in values:
        raw = text(value)
        if raw:
            return raw
    return ""


def join_unique(values: list[Any], sep: str = "; ", limit: int = 620) -> str:
    seen: list[str] = []
    for value in values:
        raw = text(value)
        if raw and raw not in seen:
            seen.append(raw)
    return shorten(sep.join(seen), limit)


def route_bucket(ticket: pd.Series, candidate: pd.Series, option: pd.Series) -> str:
    vehicle = upper(ticket.get("primary_vehicle_after_clear"))
    lane = upper(ticket.get("desk_lane") or candidate.get("candidate_lane"))
    side = upper(ticket.get("option_side_after_clear") or option.get("option_side"))
    if "CALL" in vehicle or "CALL" in side or "BULLISH OPTION" in lane:
        return "Call research after gates clear"
    if "PUT" in vehicle or "HEDGE" in vehicle or "PUT" in side or "HEDGE" in lane:
        return "Put or hedge research"
    if "STOCK" in vehicle or "ETF" in vehicle or "EQUITY" in lane:
        return "Tiny stock or ETF paper review"
    return "Research only"


def decision_now(ticket: pd.Series, risk: pd.Series, monitor_count: int) -> str:
    permission = upper(ticket.get("current_permission"))
    risk_action = upper(risk.get("final_risk_action") or risk.get("master_risk_action"))
    if "NO ACTION" in permission:
        return "Wait. Do not add exposure now."
    if "REDUCE" in risk_action or "SIZE_DOWN" in risk_action:
        return "Risk first. Keep size at or below risk budget."
    if monitor_count > 0:
        return "Monitor first. Price or news shock is active."
    if "TINY" in permission:
        return "Tiny paper review only."
    return "Research only. No live order."


def bad_news_vulnerability(row: pd.Series) -> str:
    labels: list[str] = []
    if safe_float(row.get("valuation_vulnerability")) >= 70:
        labels.append("high valuation")
    if safe_float(row.get("weakness_vulnerability")) >= 70:
        labels.append("weak price action")
    if safe_float(row.get("risk_vulnerability")) >= 70:
        labels.append("risk gate pressure")
    if safe_float(row.get("volatility_vulnerability")) >= 70:
        labels.append("high volatility")
    if safe_float(row.get("option_vulnerability")) >= 70:
        labels.append("option-sensitive")
    return ", ".join(labels) if labels else "no extreme negative-news vulnerability flag"


def calibrated_action_label(row: pd.Series) -> str:
    action = upper(row.get("calibrated_research_action"))
    status = upper(row.get("calibrated_reliability_status"))
    if "CALL_RESEARCH" in action:
        return "Calibrated bullish watch"
    if "PUT" in action or "HEDGE" in action:
        return "Calibrated defensive watch"
    if "WATCH_ONLY" in action:
        return "Watch only. Needs price and volume confirmation"
    if "SMALL_SAMPLE" in action or "LOW_SAMPLE" in status:
        return "Small-sample news review"
    if "DO_NOT_UPGRADE" in action or "UNPROVEN" in status:
        return "Context only. Not enough local proof"
    if action:
        return action.replace("_", " ").title()
    return ""


def news_direction(row: pd.Series) -> str:
    calibrated = calibrated_action_label(row)
    if calibrated:
        return calibrated
    tone = upper(row.get("market_tone"))
    impact = safe_float(row.get("impact_score"))
    route = upper(row.get("suggested_research_route"))
    if tone == "POSITIVE" or impact > 1.0 or "POSITIVE" in route:
        return "Bullish read-through"
    if tone == "NEGATIVE" or impact < -1.0 or "AVOID" in route or "NEGATIVE" in route:
        return "Bearish risk"
    if tone == "MIXED" or "MANUAL" in route:
        return "Mixed / manual review"
    return "Context only"


def score_news(row: pd.Series) -> float:
    calibrated_score = safe_float(row.get("calibrated_event_score"), np.nan)
    reliability_score = safe_float(row.get("calibrated_reliability_score"), np.nan)
    if np.isfinite(calibrated_score) or np.isfinite(reliability_score):
        status = upper(row.get("calibrated_reliability_status"))
        action = upper(row.get("calibrated_research_action"))
        status_penalty = 0.0
        if "LOW_SAMPLE" in status:
            status_penalty += 8.0
        if "UNPROVEN" in status:
            status_penalty += 12.0
        if "PENDING" in status or "PENDING" in upper(row.get("model_seen_audit_status")):
            status_penalty += 6.0
        action_boost = 8.0 if "WATCH_ONLY" in action else 4.0 if "SMALL_SAMPLE" in action else 0.0
        return max(
            0.0,
            28.0
            + abs(calibrated_score if np.isfinite(calibrated_score) else 0.0) * 15.0
            + max(0.0, (reliability_score if np.isfinite(reliability_score) else 40.0) - 35.0) * 0.8
            + action_boost
            - status_penalty,
        )
    return (
        abs(safe_float(row.get("impact_score"))) * 12.0
        + safe_float(row.get("total_vulnerability")) * 0.45
        + safe_float(row.get("predicted_score")) * 18.0
        + safe_float(row.get("alpha_score")) * 0.08
    )


def collect_news_rows(
    news_targets: pd.DataFrame,
    news_readthrough: pd.DataFrame,
    event_reliability: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source_df, source_type in [
        (event_reliability, "calibrated event reliability"),
        (news_targets, "news target"),
        (news_readthrough, "supply-chain read-through"),
    ]:
        if source_df.empty or "target_ticker" not in source_df.columns:
            continue
        sub = source_df[source_df["target_ticker"] == ticker].copy()
        if sub.empty:
            continue
        sub["decision_room_source_type"] = source_type
        if source_type == "calibrated event reliability":
            sub["source_file"] = "event_signal_reliability_adjusted_panel.csv"
            if "news_logic" not in sub.columns:
                sub["news_logic"] = (
                    "Calibrated from local event audit. "
                    + sub.get("calibrated_reliability_status", "").astype(str)
                    + "; "
                    + sub.get("calibration_note", "").astype(str)
                )
            if "action_hint" not in sub.columns:
                sub["action_hint"] = sub.get("calibrated_research_action", "")
            if "target_reason" not in sub.columns:
                sub["target_reason"] = (
                    sub.get("relation_layer", "").astype(str)
                    + " / "
                    + sub.get("target_relation", "").astype(str)
                )
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for col in ["published", "headline"]:
        if col not in out.columns:
            out[col] = ""
    out["news_direction"] = out.apply(news_direction, axis=1)
    out["negative_vulnerability_summary"] = out.apply(bad_news_vulnerability, axis=1)
    out["news_priority_score"] = out.apply(score_news, axis=1)
    out = out.sort_values(["news_priority_score", "published", "headline"], ascending=[False, False, True])
    return out


def build_news_output(
    news_targets: pd.DataFrame,
    news_readthrough: pd.DataFrame,
    event_reliability: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        sub = collect_news_rows(news_targets, news_readthrough, event_reliability, ticker).head(8)
        for rank, (_, nr) in enumerate(sub.iterrows(), start=1):
            rows.append({
                "ticker": ticker,
                "news_rank": rank,
                "news_direction": nr.get("news_direction", ""),
                "headline": nr.get("headline", ""),
                "published": nr.get("published", ""),
                "publisher": nr.get("publisher", ""),
                "source_news_ticker": nr.get("source_news_ticker", ""),
                "target_relation": nr.get("target_relation", ""),
                "theme": nr.get("theme", ""),
                "chain_role": nr.get("chain_role", ""),
                "impact_score": nr.get("impact_score", ""),
                "news_logic": nr.get("news_logic", ""),
                "action_hint": nr.get("action_hint", ""),
                "calibrated_event_score": nr.get("calibrated_event_score", ""),
                "calibrated_reliability_score": nr.get("calibrated_reliability_score", ""),
                "calibrated_reliability_status": nr.get("calibrated_reliability_status", ""),
                "calibrated_research_action": nr.get("calibrated_research_action", ""),
                "calibration_source": nr.get("calibration_source", ""),
                "calibration_observed_rows": nr.get("calibration_observed_rows", ""),
                "calibration_hit_rate": nr.get("calibration_hit_rate", ""),
                "model_seen_audit_status": nr.get("model_seen_audit_status", ""),
                "calibration_note": nr.get("calibration_note", ""),
                "decision_room_source_type": nr.get("decision_room_source_type", ""),
                "negative_vulnerability_summary": nr.get("negative_vulnerability_summary", ""),
                "target_reason": nr.get("target_reason", ""),
                "link": nr.get("link", ""),
                "source_file": nr.get("source_file", ""),
                "research_only": True,
                "no_broker_connection": True,
            })
    return pd.DataFrame(rows)


def room_status(decision: str, route: str, event_gate: str, monitor_count: int) -> str:
    if "Wait" in decision or "Risk first" in decision:
        return "Blocked now"
    if monitor_count > 0:
        return "Monitor first"
    if upper(event_gate) not in {"", "CLEAR"}:
        return "Source review first"
    if "Call research" in route:
        return "Call watch after gates clear"
    if "Put or hedge" in route:
        return "Hedge watch"
    if "Tiny" in route:
        return "Tiny paper watch"
    return "Research only"


def build_decision_room() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tickets = one_by_ticker(read_csv_safe(IN_TICKETS))
    candidates = one_by_ticker(read_csv_safe(IN_CANDIDATES))
    picks = one_by_ticker(read_csv_safe(IN_PICKS))
    risk = one_by_ticker(read_csv_safe(IN_RISK))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    sector = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    evidence = one_by_ticker(read_csv_safe(IN_EVIDENCE))
    conflict = one_by_ticker(read_csv_safe(IN_CONFLICT))
    theme = one_by_ticker(read_csv_safe(IN_THEME))
    monitor = normalize_ticker_column(read_csv_safe(IN_MONITOR), "ticker")
    news_targets = normalize_ticker_column(read_csv_safe(IN_NEWS_TARGETS), "target_ticker")
    news_readthrough = normalize_ticker_column(read_csv_safe(IN_NEWS_READTHROUGH), "target_ticker")
    event_reliability = normalize_ticker_column(read_csv_safe(IN_EVENT_RELIABILITY), "target_ticker")

    tickers = list(dict.fromkeys(
        list(tickets.index if not tickets.empty else [])
        + list(candidates.index if not candidates.empty else [])
        + list(theme.index if not theme.empty else [])
    ))

    if not tickers:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_TICKERS",
            "room_rows": 0,
            "news_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    news_output = build_news_output(news_targets, news_readthrough, event_reliability, tickers)
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        ticket = row_at(tickets, ticker)
        candidate = row_at(candidates, ticker)
        pick = row_at(picks, ticker)
        rk = row_at(risk, ticker)
        opt = row_at(options, ticker)
        sec = row_at(sector, ticker)
        ev = row_at(event, ticker)
        evid = row_at(evidence, ticker)
        cf = row_at(conflict, ticker)
        th = row_at(theme, ticker)
        monitor_rows = monitor[monitor["ticker"] == ticker].copy() if not monitor.empty else pd.DataFrame()
        news_rows = news_output[news_output["ticker"] == ticker].copy() if not news_output.empty else pd.DataFrame()
        top_news = news_rows.head(1).iloc[0] if not news_rows.empty else pd.Series(dtype=object)

        route = route_bucket(ticket, candidate, opt)
        now_decision = decision_now(ticket, rk, len(monitor_rows))
        event_gate = first_nonempty(ticket.get("event_status"), ev.get("event_gate"))
        status = room_status(now_decision, route, event_gate, len(monitor_rows))
        alpha_score = safe_float(pick.get("alpha_score") or candidate.get("alpha_score"), 0.0)
        readiness = safe_float(candidate.get("readiness_score") or ticket.get("ticket_score"), 0.0)
        decision_quality_score = max(0.0, min(100.0, readiness + alpha_score * 0.15 - len(monitor_rows) * 5.0 - safe_float(cf.get("high_conflicts")) * 5.0))

        bad_news_summary = text(top_news.get("negative_vulnerability_summary"))
        if not bad_news_summary and not news_rows.empty:
            bad_news_summary = join_unique(news_rows["negative_vulnerability_summary"].head(3).tolist())

        rows.append({
            "ticker": ticker,
            "room_status": status,
            "decision_quality_score": round(decision_quality_score, 2),
            "decision_now": now_decision,
            "route_after_gates_clear": route,
            "current_permission": first_nonempty(ticket.get("current_permission"), "Research only. No live order."),
            "primary_vehicle_after_clear": text(ticket.get("primary_vehicle_after_clear")),
            "option_side_after_clear": text(ticket.get("option_side_after_clear") or opt.get("option_side")),
            "option_structure_after_clear": text(ticket.get("option_structure_after_clear") or opt.get("option_structure")),
            "short_term_plan": text(ticket.get("short_term_plan") or sec.get("short_decision")),
            "medium_term_plan": text(ticket.get("medium_term_plan") or sec.get("medium_decision")),
            "long_term_plan": text(ticket.get("long_term_plan") or sec.get("long_decision")),
            "trigger_to_watch": first_nonempty(ticket.get("trigger_to_watch"), opt.get("call_trigger"), candidate.get("price_trigger_to_watch")),
            "invalidation": text(ticket.get("invalidation") or opt.get("option_invalidation")),
            "risk_gate": first_nonempty(rk.get("final_risk_action"), rk.get("master_risk_action"), sec.get("risk_action")),
            "event_gate": event_gate,
            "sector": first_nonempty(pick.get("sector"), sec.get("sector"), th.get("theme")),
            "sector_cycle_state": first_nonempty(sec.get("sector_cycle_state"), candidate.get("sector_cycle_state")),
            "theme": first_nonempty(th.get("theme"), top_news.get("theme")),
            "chain_role": first_nonempty(th.get("chain_role"), top_news.get("chain_role")),
            "top_news_direction": text(top_news.get("news_direction")),
            "top_news_headline": text(top_news.get("headline")),
            "top_news_logic": shorten(text(top_news.get("news_logic"))),
            "top_news_action_hint": shorten(text(top_news.get("action_hint"))),
            "top_news_calibrated_score": top_news.get("calibrated_event_score", ""),
            "top_news_reliability_score": top_news.get("calibrated_reliability_score", ""),
            "top_news_reliability_status": text(top_news.get("calibrated_reliability_status")),
            "top_news_calibrated_action": text(top_news.get("calibrated_research_action")),
            "top_news_calibration_note": shorten(text(top_news.get("calibration_note")), 620),
            "top_news_calibration_source": text(top_news.get("source_file") or top_news.get("decision_room_source_type")),
            "negative_news_vulnerability": shorten(bad_news_summary),
            "supply_chain_readthrough": shorten(first_nonempty(
                th.get("status_reason"),
                top_news.get("target_reason"),
                sec.get("linkage_context"),
            )),
            "monitor_alert_count": int(len(monitor_rows)),
            "monitor_top_alert": text(monitor_rows.head(1)["title"].iloc[0]) if not monitor_rows.empty and "title" in monitor_rows.columns else "",
            "main_blocker": first_nonempty(ticket.get("main_blocker"), candidate.get("main_blocker"), cf.get("top_conflict")),
            "proof_needed": shorten(text(ticket.get("required_proof_before_upgrade"))),
            "no_go_conditions": shorten(text(ticket.get("no_go_conditions")), 760),
            "evidence_snapshot": first_nonempty(ticket.get("evidence_snapshot"), f"Evidence rows={text(evid.get('evidence_rows')) or 'N/A'}"),
            "conflict_snapshot": first_nonempty(ticket.get("conflict_snapshot"), text(cf.get("top_conflict"))),
            "source_trail": shorten(join_unique([
                ticket.get("source_trail"),
                opt.get("source_file"),
                ev.get("source_file"),
                th.get("source_file"),
                top_news.get("source_file"),
                "event_signal_reliability_adjusted_panel.csv" if text(top_news.get("calibrated_research_action")) else "",
            ], limit=760), 760),
            "plain_english_summary": shorten(
                f"{ticker}: {now_decision} If gates clear, route is {route}. "
                f"Short/medium/long: {text(ticket.get('short_term_plan') or sec.get('short_decision'))} / "
                f"{text(ticket.get('medium_term_plan') or sec.get('medium_decision'))} / "
                f"{text(ticket.get('long_term_plan') or sec.get('long_decision'))}. "
                f"Top news: {text(top_news.get('news_direction')) or 'No mapped news'}"
                f"{' (' + text(top_news.get('calibrated_reliability_status')) + ')' if text(top_news.get('calibrated_reliability_status')) else ''}."
            ),
            "research_only": True,
            "no_broker_connection": True,
        })

    room = pd.DataFrame(rows)
    if not room.empty:
        status_order = {
            "Blocked now": 0,
            "Monitor first": 1,
            "Source review first": 2,
            "Call watch after gates clear": 3,
            "Hedge watch": 4,
            "Tiny paper watch": 5,
            "Research only": 6,
        }
        room["_status_rank"] = room["room_status"].map(status_order).fillna(9)
        room = room.sort_values(["_status_rank", "decision_quality_score", "ticker"], ascending=[True, False, True])
        room = room.drop(columns=["_status_rank"]).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(room) else "NO_ROOM_ROWS",
        "room_rows": int(len(room)),
        "news_rows": int(len(news_output)),
        "calibrated_news_rows": int((news_output.get("calibrated_research_action", pd.Series(dtype=str)).astype(str).str.len() > 0).sum()) if not news_output.empty else 0,
        "blocked_now_count": int((room["room_status"] == "Blocked now").sum()) if not room.empty else 0,
        "call_watch_count": int((room["room_status"] == "Call watch after gates clear").sum()) if not room.empty else 0,
        "hedge_watch_count": int((room["room_status"] == "Hedge watch").sum()) if not room.empty else 0,
        "with_news_count": int((room["top_news_headline"].astype(str).str.len() > 0).sum()) if not room.empty else 0,
        "with_calibrated_news_count": int((room.get("top_news_calibrated_action", pd.Series(dtype=str)).astype(str).str.len() > 0).sum()) if not room.empty else 0,
        "outputs": {
            "room": OUT_ROOM.name,
            "news": OUT_NEWS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return room, news_output, state


def main() -> int:
    room, news, state = build_decision_room()
    room.to_csv(OUT_ROOM, index=False)
    news.to_csv(OUT_NEWS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Decision room rows: {state.get('room_rows', 0)}",
        f"- News mapping rows: {state.get('news_rows', 0)}",
        f"- Calibrated news rows: {state.get('calibrated_news_rows', 0)}",
        f"- Blocked now: {state.get('blocked_now_count', 0)}",
        f"- Call watch after gates clear: {state.get('call_watch_count', 0)}",
        f"- Hedge watch: {state.get('hedge_watch_count', 0)}",
        f"- Tickers with mapped news: {state.get('with_news_count', 0)}",
        f"- Tickers with calibrated top news: {state.get('with_calibrated_news_count', 0)}",
        "",
        "## Ticker Decision Rooms",
        "",
        df_to_markdown(room, max_rows=80),
        "",
        "## News Logic Map",
        "",
        df_to_markdown(news, max_rows=120),
        "",
        "## Product Truth",
        "",
        "Each room is a research view. It is not a trade recommendation, not an order ticket, and not broker-connected.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 151 - Ticker Decision Room", sections)

    print(f"wrote {OUT_ROOM.name} rows={len(room)}")
    print(f"wrote {OUT_NEWS.name} rows={len(news)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
