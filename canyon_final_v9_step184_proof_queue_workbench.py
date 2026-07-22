#!/usr/bin/env python3
"""
Canyon v9 Step 184 - Proof Queue Workbench.

Research-only. No broker connection. No live orders.

Step183 tells us why each ticker is not reviewable yet. Step184 turns that
single-name evidence into a daily workbench:
  - which proof station should be worked first
  - which tickers are affected by that station
  - which source file to open
  - what counts as proof
  - what to rerun after the proof is collected

This is a workflow and source-routing layer only. It cannot trade, grant
permission, rebalance, or override risk.
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


OUT_STATIONS = ROOT / "proof_queue_station_summary.csv"
OUT_TICKERS = ROOT / "proof_queue_ticker_cards.csv"
OUT_PLAN = ROOT / "proof_queue_daily_plan.csv"
OUT_STATE = ROOT / "proof_queue_state.json"
OUT_REPORT = ROOT / "proof_queue_report.md"


STATION_META = {
    "risk_repair_gate": {
        "order": 1,
        "station_id": "risk_repair_first",
        "station_name": "Risk Repair First",
        "plain_goal": "Get the ticker back inside single-name and portfolio risk limits before any thesis review.",
        "why": "Risk must control the research path. If this is not cleared, options, size, and route review are noise.",
        "source_hint": "risk_repair_recommendation_board.csv; risk_repair_ticker_plan.csv",
        "rerun_after": "Run Steps 176, 177, 178, 183, 184 after any manual risk repair update.",
        "do_not_do": "Do not review calls, puts, size, or route while risk repair is first blocker.",
    },
    "monitor_gate": {
        "order": 2,
        "station_id": "monitor_shock_first",
        "station_name": "Monitor Shock First",
        "plain_goal": "Explain or wait out price break, volume spike, volatility shift, spread widening, correlation/news shock, or risk breach.",
        "why": "A live monitor shock can make a good-looking thesis untradeable or turn options into a trap.",
        "source_hint": "desk_monitor_ticker_state.csv; desk_monitor_events.csv",
        "rerun_after": "Run Steps 119, 178, 179, 180, 181, 182, 183, 184 after monitor evidence changes.",
        "do_not_do": "Do not treat a calm price alone as enough. Spread, event, and correlation shock still matter.",
    },
    "spread_tca_gate": {
        "order": 3,
        "station_id": "spread_tca_first",
        "station_name": "Spread And TCA First",
        "plain_goal": "Collect bid/ask/spread or TCA proof before reviewing size or options.",
        "why": "A signal can be correct but unusable if spread and market impact eat the edge.",
        "source_hint": "desk_monitor_ticker_state.csv; options_tca_no_go_audit.csv; execution_cost_model.csv",
        "rerun_after": "Run Steps 125, 158, 178, 183, 184 after spread/TCA evidence changes.",
        "do_not_do": "Do not unlock options or sizing from DATA_GAP spread evidence.",
    },
    "event_proof_gate": {
        "order": 4,
        "station_id": "event_proof_first",
        "station_name": "Event Proof First",
        "plain_goal": "Prove that the news/event belongs to the ticker and has a causal read-through.",
        "why": "A headline can be directionally interesting but still wrong for the target ticker or timing.",
        "source_hint": "event_readthrough_target_ranking.csv; event_research_gate.csv; event_causal_chain.csv",
        "rerun_after": "Run Steps 160, 164, 165, 166, 171, 178, 181, 182, 183, 184 after event proof changes.",
        "do_not_do": "Do not upgrade from one headline without ticker mapping, timestamp, and reaction evidence.",
    },
    "iv_greeks_gamma_gate": {
        "order": 5,
        "station_id": "iv_greeks_gamma_first",
        "station_name": "IV, Greeks, Gamma First",
        "plain_goal": "Check IV, Greeks, gamma, kill-zone, and defined-risk option structure.",
        "why": "The option route can be right directionally but wrong structurally.",
        "source_hint": "option_unlock_blocker_attribution.csv; options_greeks_book_risk.csv",
        "rerun_after": "Run Steps 82, 90, 173, 174, 178, 183, 184 after option evidence changes.",
        "do_not_do": "Do not use naked weekly options or long premium without IV/spread proof.",
    },
    "price_trigger_gate": {
        "order": 6,
        "station_id": "price_trigger_first",
        "station_name": "Price Trigger First",
        "plain_goal": "Wait for the trigger and confirm it with price and volume.",
        "why": "A setup is not reviewable if the trigger is not active yet.",
        "source_hint": "gate_clear_candidate_ranking.csv; desk_monitor_ticker_state.csv",
        "rerun_after": "Run Steps 119, 178, 181, 182, 183, 184 after the trigger changes.",
        "do_not_do": "Do not front-run a trigger as if WAIT means buy.",
    },
    "route_gate": {
        "order": 7,
        "station_id": "manual_route_first",
        "station_name": "Manual Route First",
        "plain_goal": "Resolve final route only after all earlier gates are clear.",
        "why": "Route review is the last manual step, not the first permission source.",
        "source_hint": "risk_repair_strategy_reopen_map.csv; conditional_action_tickets.csv; ticker_decision_room.csv",
        "rerun_after": "Run Steps 150, 151, 152, 178, 181, 182, 183, 184 after route evidence changes.",
        "do_not_do": "Do not use route language as permission while prior gates remain blocked.",
    },
}


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


def station_meta(gate_id: str) -> dict[str, Any]:
    return STATION_META.get(gate_id, {
        "order": 99,
        "station_id": "other_proof",
        "station_name": "Other Proof",
        "plain_goal": "Collect the first missing proof before manual review.",
        "why": "Missing proof blocks review.",
        "source_hint": "ticker_reviewability_checklist.csv",
        "rerun_after": "Run Steps 178, 183, 184.",
        "do_not_do": "Do not promote a ticker from missing evidence.",
    })


def build_ticker_cards(progress: pd.DataFrame, checklist: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if progress.empty:
        return pd.DataFrame()
    for _, row in progress.iterrows():
        ticker = as_text(row.get("ticker")).upper()
        gate_id = as_text(row.get("first_unfinished_gate_id"), "unknown_gate")
        meta = station_meta(gate_id)
        checks = checklist[checklist["ticker"].astype(str).str.upper().eq(ticker)].copy() if not checklist.empty and "ticker" in checklist.columns else pd.DataFrame()
        unfinished = checks[checks.get("is_blocking_now", pd.Series(dtype=bool)).astype(bool)].copy() if not checks.empty and "is_blocking_now" in checks.columns else pd.DataFrame()
        next_sources = "; ".join(unfinished["source_file"].dropna().astype(str).head(3).tolist()) if not unfinished.empty and "source_file" in unfinished.columns else as_text(row.get("first_source_to_open"))
        rows.append({
            "ticker": ticker,
            "station_order": int(meta["order"]),
            "station_id": meta["station_id"],
            "station_name": meta["station_name"],
            "reviewability_score_0_100": safe_float(row.get("reviewability_score_0_100")),
            "reviewability_status": as_text(row.get("reviewability_status")),
            "first_unfinished_gate": as_text(row.get("first_unfinished_gate")),
            "first_source_to_open": as_text(row.get("first_source_to_open")),
            "source_files_to_open": next_sources,
            "evidence_to_collect": compact(row.get("what_would_clear_next"), 520),
            "next_three_proofs": compact(row.get("next_three_proofs"), 700),
            "route_if_every_gate_clears": compact(row.get("route_if_every_gate_clears"), 420),
            "options_permission_plain": compact(row.get("options_permission_plain"), 420),
            "why_this_station_matters": meta["why"],
            "rerun_after_proof": meta["rerun_after"],
            "do_not_do": meta["do_not_do"],
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["station_order", "reviewability_score_0_100", "ticker"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        out["ticker_rank_in_station"] = out.groupby("station_id").cumcount() + 1
    return out


def build_station_summary(ticker_cards: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ticker_cards.empty:
        return pd.DataFrame()
    for station_id, group in ticker_cards.groupby("station_id", sort=False):
        gate_id = ""
        meta = None
        for candidate, data in STATION_META.items():
            if data["station_id"] == station_id:
                gate_id = candidate
                meta = data
                break
        if meta is None:
            meta = station_meta(gate_id)
        tickers = group["ticker"].astype(str).tolist()
        sources = []
        for source in group["source_files_to_open"].dropna().astype(str).head(5).tolist():
            sources.extend([part.strip() for part in source.split(";") if part.strip()])
        source_files = "; ".join(dict.fromkeys(sources)) or meta["source_hint"]
        rows.append({
            "station_order": int(meta["order"]),
            "station_id": station_id,
            "station_name": meta["station_name"],
            "plain_goal": meta["plain_goal"],
            "ticker_count": int(len(group)),
            "tickers": ", ".join(tickers),
            "average_reviewability_score_0_100": round(float(group["reviewability_score_0_100"].mean()), 1),
            "top_ticker": tickers[0] if tickers else "NO_DATA",
            "first_source_to_open": as_text(group["first_source_to_open"].iloc[0], meta["source_hint"]),
            "source_files": source_files,
            "exact_work": compact(group["evidence_to_collect"].iloc[0], 520),
            "why_this_station_matters": meta["why"],
            "rerun_after_proof": meta["rerun_after"],
            "do_not_do": meta["do_not_do"],
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("station_order").reset_index(drop=True)
    return out


def build_daily_plan(stations: pd.DataFrame, ticker_cards: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stations.empty:
        return pd.DataFrame()
    order = 1
    for _, station in stations.iterrows():
        station_id = as_text(station.get("station_id"))
        group = ticker_cards[ticker_cards["station_id"].astype(str).eq(station_id)].copy()
        tickers = ", ".join(group["ticker"].astype(str).head(8).tolist())
        if len(group) > 8:
            tickers += f" + {len(group) - 8} more"
        rows.append({
            "work_order": order,
            "station_name": station.get("station_name"),
            "tickers_to_check": tickers,
            "do_this": station.get("plain_goal"),
            "open_source": station.get("first_source_to_open"),
            "success_condition": station.get("exact_work"),
            "why_it_matters": station.get("why_this_station_matters"),
            "rerun_after": station.get("rerun_after_proof"),
            "do_not_do": station.get("do_not_do"),
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        order += 1
    rows.append({
        "work_order": order,
        "station_name": "After Proof Collection",
        "tickers_to_check": "Only tickers whose first blocker changed",
        "do_this": "Rerun the listed steps, then read Start Here again before opening ticker memos.",
        "open_source": "proof_queue_station_summary.csv; ticker_reviewability_progress.csv",
        "success_condition": "A ticker can move only when its first unfinished gate changes or clears.",
        "why_it_matters": "This keeps the website from becoming a pile of stale tables.",
        "rerun_after": "Run Steps 178, 181, 182, 183, 184.",
        "do_not_do": "Do not manually edit a dashboard conclusion without updating the source file and rerunning the engines.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    })
    return pd.DataFrame(rows)


def main() -> None:
    progress = read_csv_safe(ROOT / "ticker_reviewability_progress.csv")
    checklist = read_csv_safe(ROOT / "ticker_reviewability_checklist.csv")

    if progress.empty or "ticker" not in progress.columns:
        for path in [OUT_STATIONS, OUT_TICKERS, OUT_PLAN]:
            pd.DataFrame().to_csv(path, index=False)
        write_json(OUT_STATE, {
            "status": "PROOF_QUEUE_NO_DATA",
            "date": today_str(),
            "reason": "ticker_reviewability_progress.csv missing or empty",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })
        write_markdown_report(
            OUT_REPORT,
            "Canyon v9 Step 184 - Proof Queue Workbench",
            ["No reviewability rows found. Run Step 183 first."],
        )
        print("Step184 complete: no reviewability data")
        return

    ticker_cards = build_ticker_cards(progress, checklist)
    stations = build_station_summary(ticker_cards)
    daily_plan = build_daily_plan(stations, ticker_cards)

    stations.to_csv(OUT_STATIONS, index=False)
    ticker_cards.to_csv(OUT_TICKERS, index=False)
    daily_plan.to_csv(OUT_PLAN, index=False)

    top_station = stations.iloc[0].to_dict() if not stations.empty else {}
    state = {
        "status": "PROOF_QUEUE_WORKBENCH_ACTIVE",
        "date": today_str(),
        "station_rows": int(len(stations)),
        "ticker_card_rows": int(len(ticker_cards)),
        "daily_plan_rows": int(len(daily_plan)),
        "top_station": as_text(top_station.get("station_name"), "NO_DATA"),
        "top_station_ticker_count": int(safe_float(top_station.get("ticker_count"), 0)),
        "top_station_first_source": as_text(top_station.get("first_source_to_open"), "NO_DATA"),
        "top_station_tickers": as_text(top_station.get("tickers"), "NO_DATA"),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        f"- Stations: {len(stations)}",
        f"- Ticker cards: {len(ticker_cards)}",
        f"- Top station: {state['top_station']}",
        "",
        "## Guardrail",
        "This workbench routes research proof. It is not a buy/sell list and cannot unlock trades.",
        "",
        "## Station summary",
        df_to_markdown(stations, max_rows=20),
        "",
        "## Daily plan",
        df_to_markdown(daily_plan, max_rows=20),
        "",
        "## Ticker cards",
        df_to_markdown(ticker_cards.head(30), max_rows=30),
    ]
    write_markdown_report(
        OUT_REPORT,
        "Canyon v9 Step 184 - Proof Queue Workbench",
        sections,
    )
    print(
        f"Step184 complete: {len(stations)} stations, {len(ticker_cards)} ticker cards, "
        f"{len(daily_plan)} plan rows"
    )


if __name__ == "__main__":
    main()
