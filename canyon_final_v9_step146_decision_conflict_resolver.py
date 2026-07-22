#!/usr/bin/env python3
"""
Canyon v9 - Step 146: Decision Conflict Resolver
================================================

Research-only. No broker connection. No live orders.

This step turns the evidence binder into a decision-conflict matrix. It does
not create new trade recommendations. It explains where strong evidence is
being blocked by risk, event, sector, option, or monitor conditions.

Outputs:
  decision_conflict_matrix.csv
  decision_conflict_summary.csv
  decision_conflict_state.json
  decision_conflict_report.md
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

IN_WORKFLOW_QUEUE = ROOT / "daily_workflow_queue.csv"
IN_EVIDENCE_BINDER = ROOT / "ticker_evidence_binder.csv"
IN_EVIDENCE_SUMMARY = ROOT / "ticker_evidence_summary.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_SECTOR_OPTION = ROOT / "sector_timeframe_option_route.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_FINAL_RISK = ROOT / "final_risk_gate.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_MONITOR = ROOT / "desk_monitor_events.csv"
IN_NEWS_IMPACT = ROOT / "news_impact_targets.csv"
IN_NEWS_WATCH = ROOT / "news_target_watchlist.csv"

OUT_MATRIX = ROOT / "decision_conflict_matrix.csv"
OUT_SUMMARY = ROOT / "decision_conflict_summary.csv"
OUT_STATE = ROOT / "decision_conflict_state.json"
OUT_REPORT = ROOT / "decision_conflict_report.md"


SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


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


def collect_tickers(*frames: pd.DataFrame) -> list[str]:
    tickers: set[str] = set()
    for df in frames:
        if df.empty:
            continue
        for col in ["ticker", "target_ticker"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str).str.upper().str.strip()
                tickers.update(v for v in vals if v)
    return sorted(tickers)


def has_any(value: Any, words: list[str]) -> bool:
    raw = upper(value)
    return any(w.upper() in raw for w in words)


def row_at(indexed: pd.DataFrame, ticker: str) -> pd.Series:
    if indexed.empty or ticker not in indexed.index:
        return pd.Series(dtype=object)
    row = indexed.loc[ticker]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def news_rows_for(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if news.empty or "target_ticker" not in news.columns:
        return pd.DataFrame()
    target = ticker.upper()
    out = news[news["target_ticker"].astype(str).str.upper().str.strip() == target].copy()
    if "total_vulnerability" in out.columns:
        out["_sort_metric"] = pd.to_numeric(out["total_vulnerability"], errors="coerce").fillna(0)
        out = out.sort_values("_sort_metric", ascending=False)
    return out


def monitor_rows_for(monitor: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if monitor.empty or "ticker" not in monitor.columns:
        return pd.DataFrame()
    target = ticker.upper()
    out = monitor[monitor["ticker"].astype(str).str.upper().str.strip() == target].copy()
    if "severity" in out.columns:
        out["_severity_rank"] = out["severity"].map(lambda x: SEVERITY_RANK.get(text(x).title(), 9))
        out = out.sort_values("_severity_rank", ascending=True)
    return out


def add_conflict(
    rows: list[dict[str, Any]],
    ticker: str,
    conflict_type: str,
    severity: str,
    conflict_status: str,
    bullish_evidence: Any,
    blocking_evidence: Any,
    resolution: Any,
    owner_layer: str,
    source_files: str,
    next_section: str,
    route_override: str = "",
) -> None:
    rows.append({
        "ticker": ticker,
        "conflict_type": conflict_type,
        "severity": severity,
        "severity_rank": SEVERITY_RANK.get(severity, 9),
        "conflict_status": conflict_status,
        "bullish_evidence": shorten(bullish_evidence),
        "blocking_evidence": shorten(blocking_evidence),
        "resolution": shorten(resolution),
        "route_override": route_override or conflict_status,
        "owner_layer": owner_layer,
        "source_files": source_files,
        "next_dashboard_section": next_section,
        "research_only": True,
        "no_broker_connection": True,
    })


def build_conflicts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    workflow = read_csv_safe(IN_WORKFLOW_QUEUE)
    evidence = read_csv_safe(IN_EVIDENCE_BINDER)
    evidence_summary = one_by_ticker(read_csv_safe(IN_EVIDENCE_SUMMARY))
    sector_route = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    sector_option = one_by_ticker(read_csv_safe(IN_SECTOR_OPTION))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    final_risk = one_by_ticker(read_csv_safe(IN_FINAL_RISK))
    picks = one_by_ticker(read_csv_safe(IN_PICKS))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    monitor = read_csv_safe(IN_MONITOR)
    news_impact = read_csv_safe(IN_NEWS_IMPACT)
    news_watch = read_csv_safe(IN_NEWS_WATCH)

    tickers = collect_tickers(
        workflow,
        read_csv_safe(IN_EVIDENCE_SUMMARY),
        read_csv_safe(IN_SECTOR_ROUTE),
        read_csv_safe(IN_OPTIONS),
        read_csv_safe(IN_FINAL_RISK),
        read_csv_safe(IN_PICKS),
        read_csv_safe(IN_EVENT),
        monitor,
        news_impact,
        news_watch,
    )

    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        pick = row_at(picks, ticker)
        risk = row_at(final_risk, ticker)
        route = row_at(sector_route, ticker)
        opt_route = row_at(sector_option, ticker)
        opt = row_at(options, ticker)
        ev = row_at(event, ticker)
        es = row_at(evidence_summary, ticker)
        mrows = monitor_rows_for(monitor, ticker)
        nrows = news_rows_for(news_impact, ticker)
        wrow = pd.Series(dtype=object)
        if not workflow.empty and "ticker" in workflow.columns:
            wdf = workflow[workflow["ticker"].astype(str).str.upper().str.strip() == ticker]
            if not wdf.empty:
                wrow = wdf.iloc[0]

        action = upper(pick.get("action", ""))
        alpha_score = safe_float(pick.get("alpha_score"), np.nan)
        alpha_rank = safe_float(pick.get("alpha_rank"), np.nan)
        bullish_alpha = action in {"BUY", "STRONG BUY"} or alpha_score >= 70
        strong_alpha = action == "STRONG BUY" or alpha_score >= 80

        risk_action = text(risk.get("final_risk_action") or route.get("risk_action") or wrow.get("risk_action"))
        master_risk_action = text(risk.get("master_risk_action") or route.get("master_risk_action"))
        reason_stack = text(risk.get("reason_stack"))
        risk_block = has_any(risk_action, ["REDUCE_ONLY", "BLOCK", "RISK FIRST"])
        risk_size_down = has_any(risk_action, ["SIZE_DOWN"])
        risk_reduction_pct = safe_float(risk.get("risk_reduction_pct_of_current"), np.nan)

        event_gate = text(ev.get("event_gate") or route.get("event_gate") or opt.get("event_gate") or wrow.get("event_gate"))
        event_problem = has_any(event_gate, ["REVIEW", "MISSING", "HIGH"])
        event_risks = text(ev.get("risks"))
        event_missing = text(ev.get("missing_research_sources"))
        event_catalysts = text(ev.get("catalysts"))

        sector_cycle = text(route.get("sector_cycle_state") or opt_route.get("sector_cycle_state") or wrow.get("sector_cycle_state"))
        linked_cycle = text(route.get("linked_sector_cycle_state") or opt_route.get("linked_sector_cycle_state"))
        linked_sector = text(route.get("linked_sector") or opt_route.get("linked_sector"))
        sector_action = text(route.get("sector_adjusted_desk_action") or wrow.get("sector_adjusted_action"))
        best_horizon = text(route.get("best_horizon_after_sector") or wrow.get("best_horizon"))
        sector_crowded = has_any(sector_cycle, ["CROWDED"])
        sector_down = has_any(sector_cycle, ["DOWNCYCLE", "LAGGARD", "FADING"])
        linked_leader = has_any(linked_cycle, ["LEADERSHIP"])

        call_score = safe_float(opt.get("call_score"), np.nan)
        put_score = safe_float(opt.get("put_score"), np.nan)
        option_permission = text(opt.get("option_permission") or opt_route.get("option_permission_before_sector"))
        option_route = text(opt_route.get("option_route") or route.get("option_route") or opt.get("option_answer") or wrow.get("option_route"))
        option_reason = text(opt.get("option_reason") or opt_route.get("option_reason"))
        option_side = text(opt.get("option_side") or opt_route.get("option_side"))
        call_edge = call_score >= 55 or has_any(option_permission, ["CALL"])
        put_or_hedge = put_score >= 50 or has_any(option_route, ["PUT", "HEDGE"]) or has_any(option_side, ["PUT"])
        no_new_option = has_any(option_route, ["NO NEW", "BLOCKED", "EVENT REVIEW", "RISK BLOCKED", "SIZE DOWN"])

        monitor_titles = "; ".join(mrows.get("title", pd.Series(dtype=object)).dropna().astype(str).head(3).tolist())
        critical_monitor = not mrows.empty and mrows.get("severity", pd.Series(dtype=object)).astype(str).str.upper().str.contains("CRITICAL|HIGH").any()
        price_break = not mrows.empty and mrows.get("monitor", pd.Series(dtype=object)).astype(str).str.upper().str.contains("PRICE_BREAK").any()

        negative_news = nrows[nrows.get("market_tone", pd.Series(dtype=object)).astype(str).str.upper().str.contains("NEGATIVE")] if not nrows.empty and "market_tone" in nrows.columns else pd.DataFrame()
        vulnerable_news = nrows[nrows.get("total_vulnerability", pd.Series(dtype=object)).apply(lambda x: safe_float(x, 0) >= 55)] if not nrows.empty and "total_vulnerability" in nrows.columns else pd.DataFrame()
        top_news = nrows.iloc[0] if not nrows.empty else pd.Series(dtype=object)

        critical_blocks = safe_float(es.get("critical_or_block_rows"), 0)
        evidence_rows = safe_float(es.get("evidence_rows"), 0)

        if bullish_alpha and (risk_block or risk_size_down):
            severity = "High" if strong_alpha or risk_block else "Medium"
            add_conflict(
                rows,
                ticker,
                "Alpha signal vs risk gate",
                severity,
                "Risk gate overrides alpha",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}; rank={alpha_rank:.0f}" if np.isfinite(alpha_score) else action,
                f"final_risk_action={risk_action}; master={master_risk_action}; reduction={risk_reduction_pct:.1%}; {reason_stack}",
                "Keep as research-only. Do not add size until risk gate clears or recommended risk weight rises.",
                "L8 Portfolio Risk",
                "daily_picks_filtered.csv; final_risk_gate.csv",
                "Risk -> Final Risk Gate",
                "Risk first",
            )

        if call_edge and (risk_block or risk_size_down or event_problem or no_new_option):
            blockers = []
            if risk_block or risk_size_down:
                blockers.append(f"risk={risk_action}")
            if event_problem:
                blockers.append(f"event={event_gate}")
            if no_new_option:
                blockers.append(f"route={option_route}")
            severity = "High" if strong_alpha and event_problem else "Medium"
            add_conflict(
                rows,
                ticker,
                "Call edge vs risk/event gate",
                severity,
                "Call blocked",
                f"call_score={call_score:.1f}; permission={option_permission}; side={option_side}",
                "; ".join(blockers) + (f"; reason={option_reason}" if option_reason else ""),
                "Do not research bullish calls until event source coverage is clean and risk gate is CLEAR. Use defined-risk watchlist only.",
                "L7 Options + L8 Risk + L5 Events",
                "options_playbook.csv; sector_timeframe_option_route.csv; final_risk_gate.csv; event_research_dossier.csv",
                "Time Frames -> Sector-Aware Timeframe Router",
                "No call now",
            )

        if put_or_hedge and bullish_alpha:
            add_conflict(
                rows,
                ticker,
                "Bullish alpha vs defensive option route",
                "Medium",
                "Defensive option route",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}" if np.isfinite(alpha_score) else action,
                f"option_route={option_route}; put_score={put_score:.1f}; risk={risk_action}",
                "Treat puts as hedge research only, not as a bearish conviction unless price and news confirm.",
                "L7 Options",
                "daily_picks_filtered.csv; options_playbook.csv; sector_timeframe_option_route.csv",
                "Time Frames -> Options Route",
                "Hedge only",
            )

        if sector_crowded and bullish_alpha:
            add_conflict(
                rows,
                ticker,
                "Strong alpha vs crowded sector",
                "Medium",
                "Crowding check needed",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}; sector={text(route.get('sector') or wrow.get('sector'))}",
                f"sector_cycle_state={sector_cycle}; sector_action={sector_action}; risk={risk_action}",
                "Do not treat ten similar technology names as ten independent bets. Check sector and factor exposure before increasing size.",
                "L3 Sector + L8 Risk",
                "sector_timeframe_route.csv; final_risk_gate.csv; daily_picks_filtered.csv",
                "Research Room -> Sector Cycle and Links",
                "Size cap",
            )

        if sector_down and bullish_alpha:
            add_conflict(
                rows,
                ticker,
                "Buy signal vs weak sector cycle",
                "Medium",
                "Sector trend disagrees",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}; best_horizon={best_horizon}",
                f"sector_cycle_state={sector_cycle}; sector_action={sector_action}",
                "Require price confirmation and smaller research size because the sector cycle is not helping.",
                "L3 Sector Rotation",
                "sector_timeframe_route.csv; daily_picks_filtered.csv",
                "Research Room -> Sector Cycle and Links",
                "Wait for confirmation",
            )

        if linked_leader and (risk_block or risk_size_down or event_problem):
            add_conflict(
                rows,
                ticker,
                "Theme linkage vs blocker",
                "Medium",
                "Theme support is not permission",
                f"linked_sector={linked_sector}; linked_cycle={linked_cycle}",
                f"risk={risk_action}; event={event_gate}; option_route={option_route}",
                "Keep theme read-through as context. It cannot override risk, event, or execution gates.",
                "L3 Sector Links + L5 Events",
                "sector_timeframe_route.csv; key_sector_linkage.csv; event_research_dossier.csv",
                "Research Room -> Sector Cycle and Links",
                "Context only",
            )

        if event_problem and (bullish_alpha or call_edge):
            add_conflict(
                rows,
                ticker,
                "Positive signal vs event data gap",
                "High" if strong_alpha and call_edge else "Medium",
                "Event review required",
                f"catalysts={event_catalysts}; alpha={alpha_score:.1f}; call_score={call_score:.1f}",
                f"event_gate={event_gate}; risks={event_risks}; missing={event_missing}",
                "Manually verify earnings date, guidance, SEC, insider, and raw news before increasing size or researching options.",
                "L5 Events",
                "event_research_dossier.csv; daily_picks_filtered.csv; options_playbook.csv",
                "Research Room -> Event Research Dossier",
                "Manual event review",
            )

        if (price_break or critical_monitor) and (bullish_alpha or risk_size_down):
            add_conflict(
                rows,
                ticker,
                "Price/monitor shock vs current idea",
                "High" if price_break else "Medium",
                "Monitor shock active",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}; workflow={text(wrow.get('workflow_bucket'))}",
                monitor_titles or "Critical monitor event active",
                "Review price break, stop level, and risk gate before any paper add. Monitor events move ahead of alpha ranking.",
                "L6 Price + L8 Risk",
                "desk_monitor_events.csv; daily_workflow_queue.csv; daily_picks_filtered.csv",
                "Run System -> Today Workflow",
                "Monitor first",
            )

        if not negative_news.empty and (bullish_alpha or risk_size_down):
            add_conflict(
                rows,
                ticker,
                "Negative news vs current route",
                "High" if strong_alpha else "Medium",
                "News risk check",
                f"{action or 'RESEARCH'} alpha_score={alpha_score:.1f}; route={text(wrow.get('workflow_bucket'))}",
                f"{top_news.get('headline', '')}; tone={top_news.get('market_tone', '')}; vulnerability={safe_float(top_news.get('total_vulnerability'), 0):.1f}",
                "Treat bad news as a valuation and weakness stress test. High-multiple or weak names need a no-add review first.",
                "L5 News + L8 Risk",
                "news_impact_targets.csv; news_target_watchlist.csv; daily_picks_filtered.csv",
                "News Room -> News Impact Targeting",
                "News review",
            )

        if not vulnerable_news.empty and critical_blocks > 0:
            add_conflict(
                rows,
                ticker,
                "High vulnerability vs blocked evidence",
                "Medium",
                "Complex evidence stack",
                f"evidence_rows={evidence_rows:.0f}; critical_or_block_rows={critical_blocks:.0f}",
                f"headline={top_news.get('headline', '')}; vulnerability={safe_float(top_news.get('total_vulnerability'), 0):.1f}",
                "Open the evidence binder before acting. This ticker has enough conflicting data to require a human research pass.",
                "Cross-layer QA",
                "ticker_evidence_summary.csv; news_impact_targets.csv",
                "Run System -> Evidence Binder",
                "Human review",
            )

    matrix = pd.DataFrame(rows)
    if not matrix.empty:
        matrix = matrix.sort_values(["severity_rank", "ticker", "conflict_type"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        tdf = matrix[matrix["ticker"] == ticker] if not matrix.empty else pd.DataFrame()
        if tdf.empty:
            continue
        top = tdf.iloc[0]
        source_files: list[str] = []
        for raw in tdf["source_files"].dropna().astype(str).tolist():
            for part in raw.replace("/", ";").split(";"):
                part = part.strip()
                if part and part not in source_files:
                    source_files.append(part)
        summary_rows.append({
            "ticker": ticker,
            "conflict_count": int(len(tdf)),
            "critical_conflicts": int((tdf["severity"] == "Critical").sum()),
            "high_conflicts": int((tdf["severity"] == "High").sum()),
            "medium_conflicts": int((tdf["severity"] == "Medium").sum()),
            "low_conflicts": int((tdf["severity"] == "Low").sum()),
            "top_conflict": top.get("conflict_type", ""),
            "route_override": top.get("route_override", ""),
            "first_resolution": top.get("resolution", ""),
            "source_files": "; ".join(source_files[:16]),
            "research_only": True,
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["high_conflicts", "medium_conflicts", "conflict_count", "ticker"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "tickers_checked": int(len(tickers)),
        "conflict_rows": int(len(matrix)),
        "summary_rows": int(len(summary)),
        "critical_conflicts": int((matrix["severity"] == "Critical").sum()) if not matrix.empty else 0,
        "high_conflicts": int((matrix["severity"] == "High").sum()) if not matrix.empty else 0,
        "medium_conflicts": int((matrix["severity"] == "Medium").sum()) if not matrix.empty else 0,
        "low_conflicts": int((matrix["severity"] == "Low").sum()) if not matrix.empty else 0,
        "alpha_vs_risk_count": int(matrix["conflict_type"].eq("Alpha signal vs risk gate").sum()) if not matrix.empty else 0,
        "call_edge_blocked_count": int(matrix["conflict_type"].eq("Call edge vs risk/event gate").sum()) if not matrix.empty else 0,
        "monitor_shock_count": int(matrix["conflict_type"].eq("Price/monitor shock vs current idea").sum()) if not matrix.empty else 0,
        "news_conflict_count": int(matrix["conflict_type"].str.contains("News", case=False, na=False).sum()) if not matrix.empty else 0,
        "status": "READY" if len(matrix) else "NO_CONFLICTS_FOUND",
        "outputs": {
            "matrix": OUT_MATRIX.name,
            "summary": OUT_SUMMARY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return matrix, summary, state


def main() -> int:
    matrix, summary, state = build_conflicts()
    matrix.to_csv(OUT_MATRIX, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Tickers checked: {state.get('tickers_checked', 0)}",
        f"- Conflict rows: {state.get('conflict_rows', 0)}",
        f"- High conflicts: {state.get('high_conflicts', 0)}",
        f"- Medium conflicts: {state.get('medium_conflicts', 0)}",
        f"- Alpha vs risk conflicts: {state.get('alpha_vs_risk_count', 0)}",
        f"- Call edge blocked conflicts: {state.get('call_edge_blocked_count', 0)}",
        f"- Monitor shock conflicts: {state.get('monitor_shock_count', 0)}",
        f"- News conflicts: {state.get('news_conflict_count', 0)}",
        "",
        "## Conflict Summary",
        "",
        df_to_markdown(summary, max_rows=80),
        "",
        "## Conflict Matrix",
        "",
        df_to_markdown(matrix, max_rows=160),
        "",
        "## Product Truth",
        "",
        "This resolver is a cross-layer research QA layer. It never overrides the no-broker and no-live-order policy.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 146 - Decision Conflict Resolver", sections)

    print(f"wrote {OUT_MATRIX.name} rows={len(matrix)}")
    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
