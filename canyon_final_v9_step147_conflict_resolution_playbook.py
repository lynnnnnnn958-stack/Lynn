#!/usr/bin/env python3
"""
Canyon v9 - Step 147: Conflict Resolution Playbook
==================================================

Research-only. No broker connection. No live orders.

Step146 finds conflicts. Step147 turns each conflict into an explicit gate
clearance playbook: what is blocked, what must be proven, what would upgrade the
route, what would downgrade it, and which dashboard/source to open next.

Outputs:
  conflict_resolution_playbook.csv
  conflict_resolution_ticker_summary.csv
  conflict_resolution_state.json
  conflict_resolution_report.md
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

IN_CONFLICT_MATRIX = ROOT / "decision_conflict_matrix.csv"
IN_CONFLICT_SUMMARY = ROOT / "decision_conflict_summary.csv"
IN_WORKFLOW_QUEUE = ROOT / "daily_workflow_queue.csv"
IN_FINAL_RISK = ROOT / "final_risk_gate.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_OPTION_ROUTE = ROOT / "sector_timeframe_option_route.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_MONITOR = ROOT / "desk_monitor_events.csv"
IN_NEWS = ROOT / "news_impact_targets.csv"

OUT_PLAYBOOK = ROOT / "conflict_resolution_playbook.csv"
OUT_SUMMARY = ROOT / "conflict_resolution_ticker_summary.csv"
OUT_STATE = ROOT / "conflict_resolution_state.json"
OUT_REPORT = ROOT / "conflict_resolution_report.md"


SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
PRIORITY_RANK = {"Immediate": 0, "High": 1, "Medium": 2, "Low": 3}


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def upper(value: Any) -> str:
    return text(value).upper()


def safe_float(value: Any, default: float = np.nan) -> float:
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


def has_any(value: Any, words: list[str]) -> bool:
    raw = upper(value)
    return any(w.upper() in raw for w in words)


def monitor_titles(monitor: pd.DataFrame, ticker: str) -> str:
    if monitor.empty or "ticker" not in monitor.columns:
        return ""
    rows = monitor[monitor["ticker"].astype(str).str.upper().str.strip() == ticker]
    if rows.empty or "title" not in rows.columns:
        return ""
    return "; ".join(rows["title"].dropna().astype(str).head(4).tolist())


def news_headline(news: pd.DataFrame, ticker: str) -> str:
    if news.empty or "target_ticker" not in news.columns:
        return ""
    rows = news[news["target_ticker"].astype(str).str.upper().str.strip() == ticker].copy()
    if rows.empty:
        return ""
    if "total_vulnerability" in rows.columns:
        rows["_v"] = pd.to_numeric(rows["total_vulnerability"], errors="coerce").fillna(0)
        rows = rows.sort_values("_v", ascending=False)
    row = rows.iloc[0]
    return f"{row.get('headline', '')}; tone={row.get('market_tone', '')}; vulnerability={safe_float(row.get('total_vulnerability'), 0):.1f}"


def priority_for(severity: str, route_override: str, conflict_type: str) -> str:
    if severity == "Critical":
        return "Immediate"
    if severity == "High" or has_any(route_override, ["RISK", "MONITOR", "MANUAL EVENT"]):
        return "High"
    if has_any(conflict_type, ["Call edge", "Event", "News", "Sector"]):
        return "Medium"
    return "Low"


def rule_for_conflict(
    conflict: pd.Series,
    workflow: pd.Series,
    risk: pd.Series,
    event: pd.Series,
    option_route: pd.Series,
    options: pd.Series,
    monitor_note: str,
    news_note: str,
) -> dict[str, str]:
    conflict_type = text(conflict.get("conflict_type"))
    route_override = text(conflict.get("route_override"))
    severity = text(conflict.get("severity"))

    risk_action = text(risk.get("final_risk_action") or workflow.get("risk_action"))
    recommended = safe_float(risk.get("recommended_risk_weight_pct") or workflow.get("recommended_weight_pct"), np.nan)
    current = safe_float(risk.get("current_weight_pct") or workflow.get("current_weight_pct"), np.nan)
    event_gate = text(event.get("event_gate") or workflow.get("event_gate") or option_route.get("event_gate"))
    missing = text(event.get("missing_research_sources"))
    risks = text(event.get("risks"))
    call_trigger = text(option_route.get("call_trigger") or options.get("call_trigger"))
    put_trigger = text(option_route.get("put_trigger") or options.get("put_trigger"))
    no_go = text(option_route.get("no_go_conditions") or options.get("no_go_conditions"))
    what_to_watch = text(workflow.get("what_to_watch"))
    best_horizon = text(workflow.get("best_horizon"))
    sector_cycle = text(workflow.get("sector_cycle_state") or option_route.get("sector_cycle_state"))

    base = {
        "current_gate": "Research gate",
        "next_desk_action": "Open the source files and keep this research-only until the blocking evidence clears.",
        "must_clear": "The blocking layer must update from block/review to clear.",
        "proof_needed": "Latest source files must be regenerated by the daily runner.",
        "upgrade_trigger": "Only upgrade after the blocking layer clears and price confirms.",
        "downgrade_trigger": "Downgrade if a new risk, event, monitor, or news block appears.",
        "allowed_after_clear": "Watchlist review only.",
        "option_permission_after_clear": "No option research unless risk, event, liquidity, spread, and price gates are clean.",
        "timeframe_after_clear": best_horizon or "Wait for route update",
        "manual_checklist": "Confirm data freshness, source coverage, price level, exposure, and no-live-order policy.",
    }

    if conflict_type == "Alpha signal vs risk gate":
        return {
            **base,
            "current_gate": "Risk gate",
            "next_desk_action": f"Keep current research size at or below recommended risk weight ({recommended:.2f}%) before any new idea.",
            "must_clear": "final_risk_action must improve to CLEAR or WATCH; portfolio risk breach must disappear; recommended risk weight must no longer imply forced reduction.",
            "proof_needed": "final_risk_gate.csv plus institutional risk budget outputs must be fresh after Step118.",
            "upgrade_trigger": "Risk gate CLEAR and current weight is at or below allowed weight; no active risk-limit breach in desk monitor.",
            "downgrade_trigger": "Any REDUCE_ONLY, SIZE_DOWN, or new risk-limit breach keeps the ticker in risk-first mode.",
            "allowed_after_clear": "Move from Risk first to Watch or Tiny paper only after risk clears.",
            "option_permission_after_clear": "Options still require L5 event and L7 option gates; no automatic call approval.",
            "timeframe_after_clear": best_horizon or "Start with short review, then medium-term only if risk remains clear.",
            "manual_checklist": f"Compare current weight {current:.2f}% vs recommended {recommended:.2f}%; check reason_stack and portfolio VaR/CVaR.",
        }

    if conflict_type == "Call edge vs risk/event gate":
        return {
            **base,
            "current_gate": "Option gate",
            "next_desk_action": "No bullish call trade. Keep it as a defined-risk call-spread watch item only.",
            "must_clear": "final_risk_action must be CLEAR, event_gate must be CLEAR, execution/spread data must be manually clean, and price must hold the call trigger.",
            "proof_needed": "options_playbook.csv, sector_timeframe_option_route.csv, final_risk_gate.csv, and event_research_dossier.csv must all agree.",
            "upgrade_trigger": f"Risk CLEAR + event CLEAR + price confirms {call_trigger or 'the call trigger'} + no no-go condition.",
            "downgrade_trigger": f"Event review, risk SIZE_DOWN/REDUCE_ONLY, failed call trigger, or no-go condition: {no_go}",
            "allowed_after_clear": "Defined-risk call spread research only; no naked weekly calls and no live orders.",
            "option_permission_after_clear": "Call spread research, 2-6 week tenor, sized only inside risk budget.",
            "timeframe_after_clear": "Short-term to medium-term option research only after all gates clear.",
            "manual_checklist": "Verify IV rank, spread width, liquidity, event date, and price trigger before even paper-tracking the option.",
        }

    if conflict_type == "Bullish alpha vs defensive option route":
        return {
            **base,
            "current_gate": "Hedge-only option route",
            "next_desk_action": "Treat puts as hedge research, not as a bearish directional trade.",
            "must_clear": "Bullish route needs risk CLEAR and price confirmation; defensive option route remains separate from alpha score.",
            "proof_needed": "options_playbook.csv must show call permission or neutral option route after risk and event gates clear.",
            "upgrade_trigger": f"Price confirms upside trigger {call_trigger or what_to_watch}; risk and event clear.",
            "downgrade_trigger": f"Breakdown trigger {put_trigger or what_to_watch}; hedge route can remain active.",
            "allowed_after_clear": "Tiny paper equity/ETF review first; options only after L7/L8/L5 clear.",
            "option_permission_after_clear": "Defined-risk call or no option depending on updated L7 route.",
            "timeframe_after_clear": best_horizon or "Use the route playbook horizon.",
            "manual_checklist": "Separate hedge purpose from directional thesis; do not let a put route cancel risk controls.",
        }

    if conflict_type == "Strong alpha vs crowded sector":
        return {
            **base,
            "current_gate": "Sector crowding gate",
            "next_desk_action": "Do not add several correlated names just because each has a high score.",
            "must_clear": "Sector and factor exposure must be under concentration limits; correlated names must fit inside one risk budget.",
            "proof_needed": "sector_timeframe_route.csv, sector budget outputs, and factor/correlation outputs must be reviewed together.",
            "upgrade_trigger": "Sector remains leadership without crowding, risk gate clears, and portfolio has available sector budget.",
            "downgrade_trigger": "Crowded leadership, sector SIZE_DOWN, or correlation cluster breach keeps size capped.",
            "allowed_after_clear": "One best representative name or ETF sleeve, not all names in the same crowded cluster.",
            "option_permission_after_clear": "Options only for the chosen representative and only after risk/event gates clear.",
            "timeframe_after_clear": "Shorter horizon while sector is crowded; medium-term only after crowding cools.",
            "manual_checklist": f"Check sector_cycle_state={sector_cycle}; compare this ticker against same-sector alternatives.",
        }

    if conflict_type == "Buy signal vs weak sector cycle":
        return {
            **base,
            "current_gate": "Sector trend gate",
            "next_desk_action": "Wait for sector confirmation before treating the single-name buy score as actionable.",
            "must_clear": "Sector cycle must improve from downcycle/fading to neutral or leadership; price must confirm.",
            "proof_needed": "sector_cycle_state.csv and sector_timeframe_route.csv must show a better cycle state.",
            "upgrade_trigger": f"Sector improves plus ticker confirms {what_to_watch or call_trigger or 'the price trigger'}.",
            "downgrade_trigger": "Sector remains downcycle/laggard or ticker breaks support.",
            "allowed_after_clear": "Watch first, then tiny paper only if risk also clears.",
            "option_permission_after_clear": "No bullish options from a weak-sector setup without risk/event/price confirmation.",
            "timeframe_after_clear": "Short-term watch until sector cycle improves.",
            "manual_checklist": "Check whether the stock is truly idiosyncratic or just fighting a weak sector tide.",
        }

    if conflict_type == "Theme linkage vs blocker":
        return {
            **base,
            "current_gate": "Theme-readthrough gate",
            "next_desk_action": "Use the theme link as context only; do not let peer/theme news override ticker-specific gates.",
            "must_clear": "Ticker-specific risk, event, price, and source checks must confirm the read-through.",
            "proof_needed": "key_sector_linkage.csv plus ticker evidence binder rows from L5/L6/L8.",
            "upgrade_trigger": "Theme remains positive and the ticker has direct evidence: price/volume confirmation, risk clear, event clear.",
            "downgrade_trigger": "Theme link is peer-only, risk remains blocked, or event source coverage is missing.",
            "allowed_after_clear": "Add to theme watchlist; choose the strongest liquid representative, not every peer.",
            "option_permission_after_clear": "No option permission from theme alone.",
            "timeframe_after_clear": best_horizon or "Theme watch until direct ticker evidence appears.",
            "manual_checklist": "Map upstream/downstream linkage and verify whether this ticker is a direct beneficiary or only a sympathy move.",
        }

    if conflict_type == "Positive signal vs event data gap":
        return {
            **base,
            "current_gate": "Event source gate",
            "next_desk_action": "Complete event research before increasing size or researching options.",
            "must_clear": "event_gate must be CLEAR; earnings date, guidance, SEC, insider, and raw news gaps must be resolved.",
            "proof_needed": f"event_research_dossier.csv must remove or explain missing sources: {missing or 'missing source list'}",
            "upgrade_trigger": "Event source coverage improves, no earnings blackout/gap risk, and catalysts are not contradicted by guidance tone.",
            "downgrade_trigger": f"Any unresolved event risk stays blocking: {risks or 'event risk remains unresolved'}",
            "allowed_after_clear": "Tiny paper or watch review only after L5 clears; L8 still controls size.",
            "option_permission_after_clear": "Options can be reviewed only after L5 and L8 both clear.",
            "timeframe_after_clear": "Avoid short-dated options around unresolved event windows.",
            "manual_checklist": "Open raw news, earnings date, guidance, SEC filing, insider rows, and analyst revision rows.",
        }

    if conflict_type == "Price/monitor shock vs current idea":
        return {
            **base,
            "current_gate": "Monitor shock gate",
            "next_desk_action": "Resolve the monitor event before using the alpha score.",
            "must_clear": "No active CRITICAL monitor event; price must either reclaim broken support or confirm breakout with volume and risk clearance.",
            "proof_needed": "desk_monitor_events.csv and latest price/volume cache must show the shock is resolved or confirmed.",
            "upgrade_trigger": f"Monitor clears and price confirms: {what_to_watch or call_trigger or put_trigger}",
            "downgrade_trigger": monitor_note or "Any fresh price break, volume shock, spread widening, or risk breach.",
            "allowed_after_clear": "Watch or tiny paper only after monitor clears; risk gate still controls size.",
            "option_permission_after_clear": "No short-dated option while monitor shock is unresolved.",
            "timeframe_after_clear": "Short-term monitoring first; longer-term only after shock stabilizes.",
            "manual_checklist": "Check price break direction, volume spike, spread widening, and whether the move happened near earnings/news.",
        }

    if conflict_type == "Negative news vs current route":
        return {
            **base,
            "current_gate": "News risk gate",
            "next_desk_action": "Open the headline and decide if it is direct, peer, or sector read-through before any add.",
            "must_clear": "Headline must be verified, affected ticker relationship must be explicit, and vulnerability must not exceed risk tolerance.",
            "proof_needed": "news_impact_targets.csv and stock_news.json source link must be reviewed.",
            "upgrade_trigger": "Headline is stale/irrelevant or price absorbs it without risk breach; event and monitor gates remain clear.",
            "downgrade_trigger": news_note or "Direct negative headline with high vulnerability or price confirmation lower.",
            "allowed_after_clear": "Manual watch only; do not upgrade from headline alone.",
            "option_permission_after_clear": "No option route from news until price, event, and risk gates confirm.",
            "timeframe_after_clear": "Intraday/short-term watch until news impact is digested.",
            "manual_checklist": "Classify direct vs peer read-through, valuation sensitivity, weakness risk, and affected supply-chain layer.",
        }

    if conflict_type == "High vulnerability vs blocked evidence":
        return {
            **base,
            "current_gate": "Evidence complexity gate",
            "next_desk_action": "Open the evidence binder and resolve critical/block rows before considering a thesis.",
            "must_clear": "Critical/block evidence count must fall or be explicitly explained by a human review.",
            "proof_needed": "ticker_evidence_binder.csv plus decision_conflict_matrix.csv for the selected ticker.",
            "upgrade_trigger": "Critical/block rows resolved, source files are fresh, and no unresolved news/event/risk conflict remains.",
            "downgrade_trigger": "New negative news, monitor shock, or risk block adds to the evidence stack.",
            "allowed_after_clear": "Research note only; action depends on the highest remaining gate.",
            "option_permission_after_clear": "No option permission from evidence complexity alone.",
            "timeframe_after_clear": "Manual research queue.",
            "manual_checklist": "Read all high-severity binder rows before trusting any aggregate score.",
        }

    return base


def build_playbook() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conflicts = read_csv_safe(IN_CONFLICT_MATRIX)
    conflict_summary = one_by_ticker(read_csv_safe(IN_CONFLICT_SUMMARY))
    workflow = one_by_ticker(read_csv_safe(IN_WORKFLOW_QUEUE))
    risk = one_by_ticker(read_csv_safe(IN_FINAL_RISK))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    option_route = one_by_ticker(read_csv_safe(IN_OPTION_ROUTE))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    monitor = read_csv_safe(IN_MONITOR)
    news = read_csv_safe(IN_NEWS)

    if conflicts.empty or "ticker" not in conflicts.columns:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_CONFLICT_MATRIX",
            "playbook_rows": 0,
            "summary_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    rows: list[dict[str, Any]] = []
    conflicts = conflicts.copy()
    conflicts["ticker"] = conflicts["ticker"].astype(str).str.upper().str.strip()
    conflicts = conflicts[conflicts["ticker"] != ""]

    for _, conflict in conflicts.iterrows():
        ticker = text(conflict.get("ticker")).upper()
        wf = row_at(workflow, ticker)
        rk = row_at(risk, ticker)
        ev = row_at(event, ticker)
        optr = row_at(option_route, ticker)
        opt = row_at(options, ticker)
        mnote = monitor_titles(monitor, ticker)
        nnote = news_headline(news, ticker)
        rule = rule_for_conflict(conflict, wf, rk, ev, optr, opt, mnote, nnote)
        priority = priority_for(text(conflict.get("severity")), text(conflict.get("route_override")), text(conflict.get("conflict_type")))

        rows.append({
            "ticker": ticker,
            "priority": priority,
            "priority_rank": PRIORITY_RANK.get(priority, 9),
            "conflict_type": text(conflict.get("conflict_type")),
            "severity": text(conflict.get("severity")),
            "route_override": text(conflict.get("route_override")),
            "current_gate": rule["current_gate"],
            "next_desk_action": shorten(rule["next_desk_action"]),
            "must_clear": shorten(rule["must_clear"]),
            "proof_needed": shorten(rule["proof_needed"]),
            "upgrade_trigger": shorten(rule["upgrade_trigger"]),
            "downgrade_trigger": shorten(rule["downgrade_trigger"]),
            "allowed_after_clear": shorten(rule["allowed_after_clear"]),
            "option_permission_after_clear": shorten(rule["option_permission_after_clear"]),
            "timeframe_after_clear": rule["timeframe_after_clear"],
            "manual_checklist": shorten(rule["manual_checklist"]),
            "source_files": shorten(text(conflict.get("source_files"))),
            "next_dashboard_section": text(conflict.get("next_dashboard_section")),
            "research_only": True,
            "no_broker_connection": True,
        })

    playbook = pd.DataFrame(rows)
    if not playbook.empty:
        playbook = playbook.sort_values(["priority_rank", "ticker", "conflict_type"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for ticker, tdf in playbook.groupby("ticker", sort=False):
        tdf = tdf.sort_values(["priority_rank", "conflict_type"])
        top = tdf.iloc[0]
        cs = row_at(conflict_summary, ticker)
        gates = []
        for gate in tdf["current_gate"].dropna().astype(str).tolist():
            if gate and gate not in gates:
                gates.append(gate)
        required = []
        for item in tdf["must_clear"].dropna().astype(str).tolist():
            if item and item not in required:
                required.append(item)
        option_permissions = []
        for item in tdf["option_permission_after_clear"].dropna().astype(str).tolist():
            if item and item not in option_permissions:
                option_permissions.append(item)
        summary_rows.append({
            "ticker": ticker,
            "priority": top.get("priority", ""),
            "playbook_rows": int(len(tdf)),
            "conflict_count": int(cs.get("conflict_count", len(tdf))) if not cs.empty else int(len(tdf)),
            "top_gate": top.get("current_gate", ""),
            "top_conflict": top.get("conflict_type", ""),
            "action_now": top.get("next_desk_action", ""),
            "unlock_path": shorten(" | ".join(required[:3]), 700),
            "allowed_after_clear": top.get("allowed_after_clear", ""),
            "option_after_clear": shorten(" | ".join(option_permissions[:2]), 500),
            "manual_sources_to_open": top.get("source_files", ""),
            "next_dashboard_section": top.get("next_dashboard_section", ""),
            "research_only": True,
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["priority", "playbook_rows", "ticker"], ascending=[True, False, True])
        summary["priority_rank"] = summary["priority"].map(lambda x: PRIORITY_RANK.get(text(x), 9))
        summary = summary.sort_values(["priority_rank", "playbook_rows", "ticker"], ascending=[True, False, True])
        summary = summary.drop(columns=["priority_rank"]).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(playbook) else "NO_PLAYBOOK_ROWS",
        "tickers": int(summary["ticker"].nunique()) if not summary.empty and "ticker" in summary.columns else 0,
        "playbook_rows": int(len(playbook)),
        "summary_rows": int(len(summary)),
        "immediate_rows": int((playbook["priority"] == "Immediate").sum()) if not playbook.empty else 0,
        "high_rows": int((playbook["priority"] == "High").sum()) if not playbook.empty else 0,
        "medium_rows": int((playbook["priority"] == "Medium").sum()) if not playbook.empty else 0,
        "risk_gate_rows": int((playbook["current_gate"] == "Risk gate").sum()) if not playbook.empty else 0,
        "option_gate_rows": int((playbook["current_gate"] == "Option gate").sum()) if not playbook.empty else 0,
        "event_gate_rows": int((playbook["current_gate"] == "Event source gate").sum()) if not playbook.empty else 0,
        "monitor_gate_rows": int((playbook["current_gate"] == "Monitor shock gate").sum()) if not playbook.empty else 0,
        "outputs": {
            "playbook": OUT_PLAYBOOK.name,
            "summary": OUT_SUMMARY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return playbook, summary, state


def main() -> int:
    playbook, summary, state = build_playbook()
    playbook.to_csv(OUT_PLAYBOOK, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Tickers: {state.get('tickers', 0)}",
        f"- Playbook rows: {state.get('playbook_rows', 0)}",
        f"- High rows: {state.get('high_rows', 0)}",
        f"- Risk gate rows: {state.get('risk_gate_rows', 0)}",
        f"- Option gate rows: {state.get('option_gate_rows', 0)}",
        f"- Event gate rows: {state.get('event_gate_rows', 0)}",
        f"- Monitor gate rows: {state.get('monitor_gate_rows', 0)}",
        "",
        "## Ticker Summary",
        "",
        df_to_markdown(summary, max_rows=80),
        "",
        "## Resolution Playbook",
        "",
        df_to_markdown(playbook, max_rows=180),
        "",
        "## Product Truth",
        "",
        "This playbook defines research gates. It is not an order ticket and cannot send trades.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 147 - Conflict Resolution Playbook", sections)

    print(f"wrote {OUT_PLAYBOOK.name} rows={len(playbook)}")
    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
