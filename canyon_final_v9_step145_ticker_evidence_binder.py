#!/usr/bin/env python3
"""
Canyon v9 - Step 145: Ticker Evidence Binder
============================================

Research-only. No broker connection. No live orders.

This step builds a ticker-level evidence binder behind the dynamic workflow.
Instead of showing only a list of source files, it creates explicit rows showing
which layer, source file, status, evidence, and action contributed to the
current route.

Outputs:
  ticker_evidence_binder.csv
  ticker_evidence_summary.csv
  ticker_evidence_state.json
  ticker_evidence_report.md
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
IN_WORKFLOW_EXPLAIN = ROOT / "daily_workflow_ticker_explain.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_SECTOR_DETAIL = ROOT / "sector_timeframe_ticker_detail.csv"
IN_SECTOR_OPTION = ROOT / "sector_timeframe_option_route.csv"
IN_RISK_QUEUE = ROOT / "risk_desk_ticker_action_queue.csv"
IN_FINAL_RISK = ROOT / "final_risk_gate.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_MONITOR = ROOT / "desk_monitor_events.csv"
IN_NEWS_TARGET = ROOT / "news_target_watchlist.csv"
IN_NEWS_IMPACT = ROOT / "news_impact_targets.csv"
IN_SUPPLY_CHAIN = ROOT / "news_supply_chain_readthrough.csv"
IN_SECTOR_CYCLE = ROOT / "sector_cycle_state.csv"
IN_KEY_LINKS = ROOT / "key_sector_linkage.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"

OUT_BINDER = ROOT / "ticker_evidence_binder.csv"
OUT_SUMMARY = ROOT / "ticker_evidence_summary.csv"
OUT_STATE = ROOT / "ticker_evidence_state.json"
OUT_REPORT = ROOT / "ticker_evidence_report.md"


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def shorten(value: Any, limit: int = 420) -> str:
    raw = text(value)
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "..."


def contains_ticker(value: Any, ticker: str) -> bool:
    target = text(ticker).upper()
    raw = text(value).upper()
    if not target or not raw:
        return False
    tokens = [x.strip().upper() for x in raw.replace(";", ",").replace("|", ",").split(",")]
    return target in tokens or f" {target} " in f" {raw} "


def status_rank(status: Any) -> int:
    raw = text(status).upper()
    if any(x in raw for x in ["CRITICAL", "REDUCE_ONLY", "RISK FIRST", "BLOCK"]):
        return 0
    if any(x in raw for x in ["SIZE_DOWN", "REVIEW", "MISSING", "WARNING", "CROWDED"]):
        return 1
    if any(x in raw for x in ["WATCH", "HEDGE", "FADING", "DOWNCYCLE"]):
        return 2
    if any(x in raw for x in ["CLEAR", "OK", "LEADERSHIP", "POSITIVE"]):
        return 3
    return 4


def one_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    return df.copy().drop_duplicates("ticker", keep="first").set_index("ticker")


def add_row(
    rows: list[dict[str, Any]],
    ticker: str,
    layer: str,
    evidence_type: str,
    status: Any,
    evidence: Any,
    action: Any = "",
    source_file: str = "",
    source_provider: str = "",
    source_timestamp: str = "",
    metric: str = "",
    next_section: str = "",
) -> None:
    if not text(evidence) and not text(status) and not text(action):
        return
    rows.append({
        "ticker": ticker,
        "layer": layer,
        "evidence_type": evidence_type,
        "status": text(status),
        "severity_rank": status_rank(status),
        "evidence": shorten(evidence),
        "action": shorten(action, 320),
        "metric": shorten(metric, 220),
        "source_file": source_file,
        "source_provider": source_provider,
        "source_timestamp": source_timestamp,
        "next_dashboard_section": next_section,
        "research_only": True,
        "no_broker_connection": True,
    })


def build_binder() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    workflow = read_csv_safe(IN_WORKFLOW_QUEUE)
    workflow_explain = one_by_ticker(read_csv_safe(IN_WORKFLOW_EXPLAIN))
    sector_route = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    sector_detail = read_csv_safe(IN_SECTOR_DETAIL)
    sector_option = one_by_ticker(read_csv_safe(IN_SECTOR_OPTION))
    risk_queue = one_by_ticker(read_csv_safe(IN_RISK_QUEUE))
    final_risk = one_by_ticker(read_csv_safe(IN_FINAL_RISK))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    monitor = read_csv_safe(IN_MONITOR)
    news_target = read_csv_safe(IN_NEWS_TARGET)
    news_impact = read_csv_safe(IN_NEWS_IMPACT)
    supply = read_csv_safe(IN_SUPPLY_CHAIN)
    sector_cycle = read_csv_safe(IN_SECTOR_CYCLE)
    key_links = read_csv_safe(IN_KEY_LINKS)
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    picks = one_by_ticker(read_csv_safe(IN_PICKS))

    if workflow.empty or "ticker" not in workflow.columns:
        return pd.DataFrame(), pd.DataFrame(), {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "tickers": 0,
            "binder_rows": 0,
            "status": "NO_WORKFLOW_QUEUE",
        }

    rows: list[dict[str, Any]] = []
    tickers = workflow["ticker"].dropna().astype(str).unique().tolist()

    for _, wf in workflow.iterrows():
        ticker = text(wf.get("ticker")).upper()
        if not ticker:
            continue
        sector = text(wf.get("sector"))

        add_row(
            rows,
            ticker,
            "Workflow",
            "Daily route",
            wf.get("workflow_bucket"),
            wf.get("why"),
            wf.get("what_to_do"),
            "daily_workflow_queue.csv",
            "Canyon local workflow",
            "",
            f"priority={wf.get('priority')}; best_horizon={wf.get('best_horizon')}",
            "Run System -> Today Workflow",
        )

        if ticker in workflow_explain.index:
            e = workflow_explain.loc[ticker]
            add_row(
                rows,
                ticker,
                "Workflow",
                "Plain-English summary",
                wf.get("workflow_bucket"),
                e.get("plain_english_summary"),
                "Use this as the first read before opening source rows.",
                "daily_workflow_ticker_explain.csv",
                "Canyon local workflow",
                "",
                "",
                "Run System -> Source trail for this ticker",
            )

        if ticker in picks.index:
            p = picks.loc[ticker]
            signal_cols = [c for c in p.index if str(c).startswith("sig_")]
            best_signal = text(p.get("top_signal"))
            metric = "; ".join(
                f"{col.replace('sig_', '')}={safe_float(p.get(col), 0):.1f}"
                for col in signal_cols[:8]
            )
            add_row(
                rows,
                ticker,
                "L1/L4/L6 Alpha",
                "Alpha score",
                p.get("action"),
                f"alpha_score={safe_float(p.get('alpha_score'), 0):.2f}; rank={p.get('alpha_rank')}; top_signal={best_signal}",
                "Alpha can nominate research candidates, but cannot override risk gates.",
                "daily_picks_filtered.csv",
                "Canyon local signal stack",
                "",
                metric,
                "Today's Picks",
            )

        if ticker in risk_queue.index:
            r = risk_queue.loc[ticker]
            add_row(
                rows,
                ticker,
                "L8 Risk",
                "Risk desk queue",
                r.get("final_risk_action"),
                r.get("reason_stack"),
                r.get("required_next_action"),
                "risk_desk_ticker_action_queue.csv",
                "Canyon risk framework",
                "",
                (
                    f"current_weight={r.get('current_weight_pct')}%; "
                    f"recommended={r.get('recommended_risk_weight_pct')}%; "
                    f"VaR={r.get('var_95_1d')}; CVaR={r.get('cvar_95_1d')}"
                ),
                "Performance -> Risk Desk Summary",
            )

        if ticker in final_risk.index:
            fr = final_risk.loc[ticker]
            add_row(
                rows,
                ticker,
                "L8 Risk",
                "Final risk gate",
                fr.get("final_risk_action"),
                fr.get("reason_stack"),
                "Final risk gate constrains all downstream route decisions.",
                "final_risk_gate.csv",
                "Canyon risk framework",
                "",
                (
                    f"master={fr.get('master_risk_action')}; single={fr.get('single_name_action')}; "
                    f"earnings_gap={fr.get('earnings_gap_action')}; sector={fr.get('sector_status')}"
                ),
                "Performance -> Risk Desk Summary",
            )

        if ticker in sector_route.index:
            sr = sector_route.loc[ticker]
            add_row(
                rows,
                ticker,
                "L3 Sector / Route",
                "Sector-aware route",
                sr.get("sector_adjusted_desk_action"),
                sr.get("why"),
                sr.get("what_would_change"),
                "sector_timeframe_route.csv",
                "Canyon sector router",
                "",
                (
                    f"sector_cycle={text(sr.get('sector_cycle_state'))}; linked_sector={text(sr.get('linked_sector'))}; "
                    f"best_horizon={text(sr.get('best_horizon_after_sector'))}"
                ),
                "Time Frames -> Sector-Aware Timeframe Router",
            )

        if not sector_detail.empty and "ticker" in sector_detail.columns:
            drows = sector_detail[sector_detail["ticker"].astype(str).str.upper() == ticker].head(3)
            for _, d in drows.iterrows():
                add_row(
                    rows,
                    ticker,
                    "Timeframe",
                    text(d.get("timeframe")),
                    d.get("sector_adjusted_decision"),
                    d.get("reason"),
                    "Compare before/after sector-adjusted scores.",
                    "sector_timeframe_ticker_detail.csv",
                    "Canyon sector router",
                    "",
                    f"before={d.get('score_before_sector')}; adjustment={d.get('sector_adjustment')}; after={d.get('score_after_sector')}",
                    "Time Frames -> Sector Router",
                )

        if ticker in sector_option.index:
            so = sector_option.loc[ticker]
            add_row(
                rows,
                ticker,
                "L7 Options",
                "Sector-adjusted option route",
                so.get("option_route"),
                so.get("option_reason"),
                f"Call: {so.get('call_answer')}; Put: {so.get('put_answer')}; No-go: {so.get('no_go_conditions')}",
                "sector_timeframe_option_route.csv",
                "Canyon options router",
                "",
                f"side={so.get('option_side')}; structure={so.get('option_structure')}",
                "Time Frames -> Sector-adjusted option route",
            )

        if ticker in options.index:
            op = options.loc[ticker]
            add_row(
                rows,
                ticker,
                "L7 Options",
                "Original options playbook",
                op.get("option_permission"),
                op.get("option_reason"),
                op.get("what_would_change"),
                "options_playbook.csv",
                "Canyon options playbook",
                "",
                f"call_score={op.get('call_score')}; put_score={op.get('put_score')}; iv_rank={op.get('iv_rank')}",
                "Time Frames -> Options",
            )

        if ticker in event.index:
            ev = event.loc[ticker]
            add_row(
                rows,
                ticker,
                "L5 Events",
                "Event research dossier",
                ev.get("event_gate"),
                f"Catalysts: {ev.get('catalysts')}; Risks: {ev.get('risks')}; Missing: {ev.get('missing_research_sources')}",
                ev.get("required_next_action"),
                "event_research_dossier.csv",
                "Canyon event research",
                "",
                (
                    f"earnings={ev.get('earnings_date')}; days_until={ev.get('days_until_earnings')}; "
                    f"coverage={ev.get('event_source_coverage_pct')}%; event_score={ev.get('event_research_score')}"
                ),
                "News Room -> Event Research",
            )

        if not monitor.empty and "ticker" in monitor.columns:
            mrows = monitor[monitor["ticker"].fillna("").astype(str).str.upper().isin([ticker, "PORTFOLIO"])].head(8)
            for _, m in mrows.iterrows():
                add_row(
                    rows,
                    ticker,
                    text(m.get("source_layer")) or "Desk Monitor",
                    text(m.get("monitor")) or "Monitor event",
                    m.get("severity"),
                    f"{m.get('title')}: {m.get('detail')}",
                    m.get("action"),
                    text(m.get("source_file")) or "desk_monitor_events.csv",
                    text(m.get("source_provider")) or "Canyon desk monitor",
                    text(m.get("run_time")),
                    f"{m.get('metric_1_name')}={m.get('metric_1_value')}; {m.get('metric_2_name')}={m.get('metric_2_value')}",
                    "Alerts -> Desk Monitor",
                )

        if not news_target.empty and "target_ticker" in news_target.columns:
            nrows = news_target[news_target["target_ticker"].astype(str).str.upper() == ticker].head(5)
            for _, n in nrows.iterrows():
                add_row(
                    rows,
                    ticker,
                    "L5 News",
                    "News target watchlist",
                    n.get("suggested_research_route"),
                    f"{n.get('top_headline')}; {n.get('why_this_ticker')}",
                    "Treat as news-context research, not an automatic trade.",
                    "news_target_watchlist.csv",
                    "Yahoo Finance via yfinance + Canyon mapping",
                    "",
                    f"negative={n.get('negative_headline_count')}; positive={n.get('positive_headline_count')}; vulnerability={n.get('max_negative_vulnerability')}",
                    "News Room -> News Target Map",
                )

        if not news_impact.empty and "target_ticker" in news_impact.columns:
            irows = news_impact[news_impact["target_ticker"].astype(str).str.upper() == ticker].head(5)
            for _, n in irows.iterrows():
                add_row(
                    rows,
                    ticker,
                    "L5 News",
                    "Headline impact target",
                    n.get("market_tone"),
                    f"{n.get('headline')}; Logic: {n.get('news_logic')}; Target reason: {n.get('target_reason')}",
                    n.get("action_hint") or n.get("suggested_research_route"),
                    "news_impact_targets.csv",
                    text(n.get("publisher")) or "Yahoo Finance via yfinance",
                    text(n.get("published")),
                    f"impact={n.get('impact_score')}; total_vulnerability={n.get('total_vulnerability')}",
                    "News Room -> News Target Map",
                )

        if not supply.empty and "target_ticker" in supply.columns:
            srows = supply[supply["target_ticker"].astype(str).str.upper() == ticker].head(5)
            for _, n in srows.iterrows():
                add_row(
                    rows,
                    ticker,
                    "L5 News / Sector Link",
                    "Supply-chain read-through",
                    n.get("market_tone"),
                    f"{n.get('theme')} / {n.get('chain_role')}: {n.get('headline')}; {n.get('target_reason')}",
                    n.get("action_hint") or n.get("suggested_research_route"),
                    "news_supply_chain_readthrough.csv",
                    text(n.get("publisher")) or "Yahoo Finance via yfinance",
                    text(n.get("published")),
                    f"impact={n.get('impact_score')}; option_side={n.get('option_side')}",
                    "News Room -> Supply Chain Read-Through",
                )

        if not sector_cycle.empty and "sector" in sector_cycle.columns:
            sc = sector_cycle[sector_cycle["sector"].astype(str) == sector].head(1)
            if sc.empty:
                mention_mask = pd.Series(False, index=sector_cycle.index)
                for col in ["top_news_tickers", "top_alpha_names", "top_portfolio_tickers"]:
                    if col in sector_cycle.columns:
                        mention_mask = mention_mask | sector_cycle[col].map(lambda x: contains_ticker(x, ticker))
                sc = sector_cycle[mention_mask].head(2)
            for _, s in sc.iterrows():
                add_row(
                    rows,
                    ticker,
                    "L3 Sector",
                    "Sector cycle",
                    s.get("cycle_state"),
                    s.get("cycle_note"),
                    "Use as attention context only; risk gate still controls sizing.",
                    "sector_cycle_state.csv",
                    "Canyon sector cycle",
                    "",
                    f"sector={s.get('sector')}; score={s.get('cycle_score')}; label={s.get('rotation_label')}; catalysts={s.get('catalyst_balance')}",
                    "News Room -> Sector Cycle and Links",
                )

        if not key_links.empty:
            mask = pd.Series(False, index=key_links.index)
            if "representative_tickers" in key_links.columns:
                mask = mask | key_links["representative_tickers"].map(lambda x: contains_ticker(x, ticker))
            if sector:
                for col in ["primary_sector", "linked_sector"]:
                    if col in key_links.columns:
                        mask = mask | key_links[col].astype(str).eq(sector)
            for _, k in key_links[mask].head(5).iterrows():
                add_row(
                rows,
                ticker,
                "L3 Sector Link",
                k.get("link_source"),
                k.get("linkage_type"),
                    f"{text(k.get('primary_sector'))} -> {text(k.get('linked_sector'))}; {text(k.get('evidence_note'))}; {text(k.get('top_headline'))}",
                k.get("desk_action"),
                "key_sector_linkage.csv",
                "Canyon sector linkage",
                    "",
                    f"corr_60d={k.get('corr_60d')}; score_gap={k.get('cycle_score_gap')}",
                    "News Room -> Important sector links",
                )

    binder = pd.DataFrame(rows)
    if not binder.empty:
        binder = binder.sort_values(["ticker", "severity_rank", "layer", "evidence_type"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        tdf = binder[binder["ticker"].astype(str).str.upper() == text(ticker).upper()] if not binder.empty else pd.DataFrame()
        source_files = []
        if not tdf.empty and "source_file" in tdf.columns:
            for item in tdf["source_file"].dropna().astype(str):
                for part in item.replace("+", ";").split(";"):
                    part = part.strip()
                    if part and part not in source_files:
                        source_files.append(part)
        summary_rows.append({
            "ticker": ticker,
            "evidence_rows": int(len(tdf)),
            "risk_rows": int(tdf["layer"].astype(str).str.contains("Risk", case=False, na=False).sum()) if not tdf.empty else 0,
            "event_rows": int(tdf["layer"].astype(str).str.contains("Event|News", case=False, na=False).sum()) if not tdf.empty else 0,
            "sector_rows": int(tdf["layer"].astype(str).str.contains("Sector", case=False, na=False).sum()) if not tdf.empty else 0,
            "option_rows": int(tdf["layer"].astype(str).str.contains("Option", case=False, na=False).sum()) if not tdf.empty else 0,
            "monitor_rows": int(tdf["evidence_type"].astype(str).str.contains("PRICE|VOLUME|VOLATILITY|SPREAD|CORRELATION|NEWS|EARNINGS|RISK", case=False, na=False).sum()) if not tdf.empty else 0,
            "critical_or_block_rows": int(tdf["status"].astype(str).str.contains("CRITICAL|REDUCE_ONLY|BLOCK|RISK FIRST", case=False, na=False).sum()) if not tdf.empty else 0,
            "top_evidence": shorten(tdf.iloc[0].get("evidence") if not tdf.empty else "", 300),
            "source_files": "; ".join(source_files[:18]),
            "research_only": True,
        })
    summary = pd.DataFrame(summary_rows)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "tickers": int(len(tickers)),
        "binder_rows": int(len(binder)),
        "summary_rows": int(len(summary)),
        "risk_rows": int(summary["risk_rows"].sum()) if not summary.empty else 0,
        "event_rows": int(summary["event_rows"].sum()) if not summary.empty else 0,
        "sector_rows": int(summary["sector_rows"].sum()) if not summary.empty else 0,
        "option_rows": int(summary["option_rows"].sum()) if not summary.empty else 0,
        "critical_or_block_rows": int(summary["critical_or_block_rows"].sum()) if not summary.empty else 0,
        "status": "READY" if len(binder) else "NO_EVIDENCE_ROWS",
        "outputs": {
            "binder": OUT_BINDER.name,
            "summary": OUT_SUMMARY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return binder, summary, state


def main() -> int:
    binder, summary, state = build_binder()
    binder.to_csv(OUT_BINDER, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Tickers: {state.get('tickers', 0)}",
        f"- Binder rows: {state.get('binder_rows', 0)}",
        f"- Risk rows: {state.get('risk_rows', 0)}",
        f"- Event/news rows: {state.get('event_rows', 0)}",
        f"- Sector rows: {state.get('sector_rows', 0)}",
        f"- Option rows: {state.get('option_rows', 0)}",
        f"- Critical/block rows: {state.get('critical_or_block_rows', 0)}",
        "",
        "## Evidence Summary",
        "",
        df_to_markdown(summary, max_rows=80),
        "",
        "## Evidence Binder",
        "",
        df_to_markdown(binder, max_rows=140),
        "",
        "## Product Truth",
        "",
        "This binder explains research evidence. It is not an order ticket and cannot send trades.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 145 - Ticker Evidence Binder", sections)

    print(f"wrote {OUT_BINDER.name} rows={len(binder)}")
    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
