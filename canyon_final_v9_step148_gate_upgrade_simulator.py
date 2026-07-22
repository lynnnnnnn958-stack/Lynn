#!/usr/bin/env python3
"""
Canyon v9 - Step 148: Gate Upgrade Simulator
============================================

Research-only. No broker connection. No live orders.

Step147 defines what must clear. Step148 simulates what the route would become
under specific gate-clearance scenarios. It does not change positions, send
orders, or grant automatic option permission.

Outputs:
  gate_upgrade_simulation.csv
  gate_upgrade_ticker_summary.csv
  gate_upgrade_state.json
  gate_upgrade_report.md
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

IN_RESOLUTION_PLAYBOOK = ROOT / "conflict_resolution_playbook.csv"
IN_RESOLUTION_SUMMARY = ROOT / "conflict_resolution_ticker_summary.csv"
IN_WORKFLOW_QUEUE = ROOT / "daily_workflow_queue.csv"
IN_FINAL_RISK = ROOT / "final_risk_gate.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_OPTION_ROUTE = ROOT / "sector_timeframe_option_route.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"

OUT_SIM = ROOT / "gate_upgrade_simulation.csv"
OUT_SUMMARY = ROOT / "gate_upgrade_ticker_summary.csv"
OUT_STATE = ROOT / "gate_upgrade_state.json"
OUT_REPORT = ROOT / "gate_upgrade_report.md"


SCENARIOS = [
    {
        "scenario": "Current gates",
        "rank": 0,
        "cleared": set(),
        "description": "No assumed gate improvement; this is the current route.",
    },
    {
        "scenario": "Risk clears only",
        "rank": 1,
        "cleared": {"Risk gate"},
        "description": "Portfolio/single-name risk gate improves but other blockers may remain.",
    },
    {
        "scenario": "Event clears only",
        "rank": 2,
        "cleared": {"Event source gate"},
        "description": "Event research source coverage clears but risk/price/sector may still block.",
    },
    {
        "scenario": "Monitor clears only",
        "rank": 3,
        "cleared": {"Monitor shock gate"},
        "description": "Price/volume/risk monitor shock stabilizes but other gates may remain.",
    },
    {
        "scenario": "Price trigger confirms",
        "rank": 4,
        "cleared": {"Price trigger"},
        "description": "Ticker hits the relevant price trigger, without assuming risk/event clearance.",
    },
    {
        "scenario": "Risk + Event clear",
        "rank": 5,
        "cleared": {"Risk gate", "Event source gate"},
        "description": "Risk and event gates clear together; price/monitor/sector may still matter.",
    },
    {
        "scenario": "Full gate clear",
        "rank": 6,
        "cleared": {
            "Risk gate",
            "Event source gate",
            "Monitor shock gate",
            "Sector crowding gate",
            "Sector trend gate",
            "News risk gate",
            "Theme-readthrough gate",
            "Evidence complexity gate",
            "Hedge-only option route",
            "Price trigger",
            "Execution/spread check",
        },
        "description": "Risk, event, monitor, sector/news, price, and execution checks are assumed clean.",
    },
]

GATE_ORDER = [
    "Risk gate",
    "Event source gate",
    "Monitor shock gate",
    "Option gate",
    "Sector crowding gate",
    "Sector trend gate",
    "News risk gate",
    "Theme-readthrough gate",
    "Evidence complexity gate",
    "Hedge-only option route",
]


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


def gate_set_for(rows: pd.DataFrame) -> set[str]:
    gates = set()
    if rows.empty or "current_gate" not in rows.columns:
        return gates
    for raw in rows["current_gate"].dropna().astype(str).tolist():
        raw = raw.strip()
        if raw:
            gates.add(raw)
    return gates


def price_trigger_for(workflow: pd.Series, option_route: pd.Series, options: pd.Series) -> str:
    call_trigger = text(option_route.get("call_trigger") or options.get("call_trigger"))
    put_trigger = text(option_route.get("put_trigger") or options.get("put_trigger"))
    watch = text(workflow.get("what_to_watch"))
    if call_trigger and has_any(option_route.get("option_side") or options.get("option_side"), ["CALL"]):
        return call_trigger
    if put_trigger and has_any(option_route.get("option_side") or options.get("option_side"), ["PUT"]):
        return put_trigger
    return call_trigger or put_trigger or watch


def option_edge(options: pd.Series, option_route: pd.Series) -> tuple[str, bool, bool]:
    call_score = safe_float(options.get("call_score"), np.nan)
    put_score = safe_float(options.get("put_score"), np.nan)
    side = text(options.get("option_side") or option_route.get("option_side"))
    permission = text(options.get("option_permission") or option_route.get("option_permission_before_sector"))
    call_edge = call_score >= 55 or has_any(permission, ["CALL"]) or has_any(side, ["CALL"])
    put_edge = put_score >= 50 or has_any(side, ["PUT"]) or has_any(option_route.get("option_route"), ["PUT", "HEDGE"])
    if call_edge:
        return "CALL", True, put_edge
    if put_edge:
        return "PUT", call_edge, True
    return "NONE", False, False


def remaining_gates(gates: set[str], cleared: set[str]) -> list[str]:
    rem = set(gates)
    for gate in cleared:
        if gate in rem:
            rem.remove(gate)
    if "Option gate" in rem:
        option_dependencies_clear = {
            "Risk gate",
            "Event source gate",
            "Price trigger",
            "Execution/spread check",
        }.issubset(cleared)
        if option_dependencies_clear:
            rem.remove("Option gate")
    return [gate for gate in GATE_ORDER if gate in rem] + sorted(g for g in rem if g not in GATE_ORDER)


def route_for(
    rem: list[str],
    cleared: set[str],
    workflow: pd.Series,
    risk: pd.Series,
    event: pd.Series,
    options: pd.Series,
    option_route: pd.Series,
    sector_route: pd.Series,
    pick: pd.Series,
) -> dict[str, str]:
    risk_action = text(risk.get("final_risk_action") or workflow.get("risk_action"))
    event_gate = text(event.get("event_gate") or workflow.get("event_gate") or option_route.get("event_gate"))
    current_route = text(workflow.get("workflow_bucket"))
    current_option_route = text(option_route.get("option_route") or workflow.get("option_route"))
    horizon = text(sector_route.get("best_horizon_after_sector") or workflow.get("best_horizon"))
    sector_state = text(sector_route.get("sector_cycle_state") or workflow.get("sector_cycle_state"))
    action = text(pick.get("action"))
    alpha = safe_float(pick.get("alpha_score"), np.nan)
    side, call_edge, put_edge = option_edge(options, option_route)

    if "Risk gate" in rem:
        return {
            "simulated_route": "Risk first",
            "simulated_action": "No new exposure. Keep research size at or below risk budget.",
            "simulated_option_permission": "No new option research",
            "timeframe": "Risk review first",
            "confidence": "High",
            "why": f"Risk remains blocking: {risk_action}. Current route stays {current_route}.",
        }
    if "Event source gate" in rem:
        return {
            "simulated_route": "Event review",
            "simulated_action": "Watch only until event source coverage is clean.",
            "simulated_option_permission": "No options around unresolved event risk",
            "timeframe": "Event window review",
            "confidence": "High",
            "why": f"Event gate remains {event_gate}; source coverage must clear before option or size review.",
        }
    if "Monitor shock gate" in rem:
        return {
            "simulated_route": "Monitor first",
            "simulated_action": "Wait for price/volume/spread shock to stabilize.",
            "simulated_option_permission": "No short-dated option while monitor shock is active",
            "timeframe": "Short-term monitoring",
            "confidence": "Medium",
            "why": "Monitor shock still outranks alpha ranking and route score.",
        }
    if "Sector crowding gate" in rem:
        return {
            "simulated_route": "Sector budget check",
            "simulated_action": "Choose one representative name; do not add the whole crowded cluster.",
            "simulated_option_permission": "No options until sector/factor exposure is under budget",
            "timeframe": "Shorter horizon while crowded",
            "confidence": "Medium",
            "why": f"Sector remains crowded: {sector_state}.",
        }
    if "Sector trend gate" in rem:
        return {
            "simulated_route": "Sector watch",
            "simulated_action": "Wait for sector trend confirmation before upgrading the ticker.",
            "simulated_option_permission": "No bullish options against a weak sector cycle",
            "timeframe": "Watch until sector improves",
            "confidence": "Medium",
            "why": f"Sector trend remains a blocker: {sector_state}.",
        }
    if "News risk gate" in rem:
        return {
            "simulated_route": "News review",
            "simulated_action": "Open headline and classify direct vs peer/theme read-through.",
            "simulated_option_permission": "No option permission from news alone",
            "timeframe": "Intraday or short-term watch",
            "confidence": "Medium",
            "why": "News risk remains unresolved.",
        }
    if "Theme-readthrough gate" in rem:
        return {
            "simulated_route": "Theme context only",
            "simulated_action": "Keep theme link as context until direct ticker evidence confirms.",
            "simulated_option_permission": "No option permission from theme alone",
            "timeframe": "Theme watch",
            "confidence": "Medium",
            "why": "Theme linkage is not ticker-specific permission.",
        }
    if "Evidence complexity gate" in rem:
        return {
            "simulated_route": "Human evidence review",
            "simulated_action": "Read evidence binder before trusting the aggregate score.",
            "simulated_option_permission": "No option permission from unresolved evidence stack",
            "timeframe": "Manual research queue",
            "confidence": "Medium",
            "why": "Critical or blocked evidence remains unresolved.",
        }
    if "Option gate" in rem:
        return {
            "simulated_route": "Watch, option still blocked",
            "simulated_action": "Equity/ETF watch only; option dependencies are not all clear.",
            "simulated_option_permission": "Option gate still blocked",
            "timeframe": horizon or "Watch",
            "confidence": "Medium",
            "why": "Option gate still needs risk, event, price, and execution/spread checks to align.",
        }

    if call_edge and {"Risk gate", "Event source gate", "Price trigger", "Execution/spread check"}.issubset(cleared):
        return {
            "simulated_route": "Option research unlocked",
            "simulated_action": "Defined-risk call-spread research can be reviewed, still paper-only.",
            "simulated_option_permission": "Defined-risk call spread research only; no naked weekly calls",
            "timeframe": "Short-term to medium-term option research",
            "confidence": "Medium",
            "why": f"Risk/event/price/execution are assumed clear and call edge exists. Current option route was {current_option_route}.",
        }
    if put_edge and {"Risk gate", "Event source gate", "Price trigger"}.issubset(cleared):
        return {
            "simulated_route": "Hedge research unlocked",
            "simulated_action": "Put or hedge structure can be reviewed as protection, not automatic bearish trade.",
            "simulated_option_permission": "Defined-risk hedge research only",
            "timeframe": "Short-term hedge review",
            "confidence": "Medium",
            "why": "Defensive option edge exists and major gates are assumed clear.",
        }
    if action in {"BUY", "STRONG BUY"} or alpha >= 70:
        return {
            "simulated_route": "Tiny paper candidate",
            "simulated_action": "Move to watch or tiny paper review after manual source check.",
            "simulated_option_permission": "No automatic options; equity/ETF paper route first",
            "timeframe": horizon or "Watch",
            "confidence": "Medium",
            "why": f"Main blockers are cleared in this scenario and alpha remains supportive ({alpha:.1f}).",
        }
    return {
        "simulated_route": "Watch",
        "simulated_action": "Keep on watchlist; no action without fresh alpha and price confirmation.",
        "simulated_option_permission": "No option research",
        "timeframe": horizon or "Watch",
        "confidence": "Low",
        "why": "Gates are clear in this scenario but there is no strong enough action signal.",
    }


def build_simulation() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    playbook = read_csv_safe(IN_RESOLUTION_PLAYBOOK)
    resolution_summary = one_by_ticker(read_csv_safe(IN_RESOLUTION_SUMMARY))
    workflow = one_by_ticker(read_csv_safe(IN_WORKFLOW_QUEUE))
    risk = one_by_ticker(read_csv_safe(IN_FINAL_RISK))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    option_route = one_by_ticker(read_csv_safe(IN_OPTION_ROUTE))
    sector_route = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    picks = one_by_ticker(read_csv_safe(IN_PICKS))

    if playbook.empty or "ticker" not in playbook.columns:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_RESOLUTION_PLAYBOOK",
            "simulation_rows": 0,
            "summary_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    playbook = playbook.copy()
    playbook["ticker"] = playbook["ticker"].astype(str).str.upper().str.strip()
    tickers = sorted(t for t in playbook["ticker"].dropna().unique().tolist() if t)

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        tplay = playbook[playbook["ticker"] == ticker]
        gates = gate_set_for(tplay)
        wf = row_at(workflow, ticker)
        rk = row_at(risk, ticker)
        ev = row_at(event, ticker)
        opt = row_at(options, ticker)
        optr = row_at(option_route, ticker)
        sr = row_at(sector_route, ticker)
        pick = row_at(picks, ticker)
        rs = row_at(resolution_summary, ticker)
        trigger = price_trigger_for(wf, optr, opt)
        current_risk = text(rk.get("final_risk_action") or wf.get("risk_action"))
        current_event = text(ev.get("event_gate") or wf.get("event_gate") or optr.get("event_gate"))
        current_route = text(wf.get("workflow_bucket"))
        current_option = text(optr.get("option_route") or wf.get("option_route"))
        summary_unlock = text(rs.get("unlock_path"))

        for scenario in SCENARIOS:
            cleared = set(scenario["cleared"])
            rem = remaining_gates(gates, cleared)
            route = route_for(rem, cleared, wf, rk, ev, opt, optr, sr, pick)
            rows.append({
                "ticker": ticker,
                "scenario": scenario["scenario"],
                "scenario_rank": scenario["rank"],
                "scenario_description": scenario["description"],
                "current_workflow": current_route,
                "current_risk_action": current_risk,
                "current_event_gate": current_event,
                "current_option_route": current_option,
                "current_gates": "; ".join(g for g in GATE_ORDER if g in gates),
                "cleared_gates": "; ".join(sorted(cleared)) if cleared else "None",
                "remaining_gates": "; ".join(rem) if rem else "None",
                "simulated_route": route["simulated_route"],
                "simulated_action": route["simulated_action"],
                "simulated_option_permission": route["simulated_option_permission"],
                "simulated_timeframe": route["timeframe"],
                "simulation_confidence": route["confidence"],
                "price_trigger_to_watch": trigger,
                "why": shorten(route["why"]),
                "unlock_path_from_step147": shorten(summary_unlock),
                "source_files": "conflict_resolution_playbook.csv; daily_workflow_queue.csv; final_risk_gate.csv; event_research_dossier.csv; options_playbook.csv; sector_timeframe_option_route.csv",
                "research_only": True,
                "no_broker_connection": True,
            })

    sim = pd.DataFrame(rows)
    if not sim.empty:
        sim = sim.sort_values(["ticker", "scenario_rank"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        tdf = sim[sim["ticker"] == ticker] if not sim.empty else pd.DataFrame()
        if tdf.empty:
            continue
        full = tdf[tdf["scenario"] == "Full gate clear"].head(1)
        risk_event = tdf[tdf["scenario"] == "Risk + Event clear"].head(1)
        risk_only = tdf[tdf["scenario"] == "Risk clears only"].head(1)
        best = full.iloc[0] if not full.empty else tdf.iloc[-1]
        source = risk_event.iloc[0] if not risk_event.empty else best
        options_unlocked = int(tdf["simulated_route"].astype(str).str.contains("Option research unlocked|Hedge research unlocked", case=False, na=False).sum())
        still_blocked_after_risk = text(risk_only.iloc[0].get("remaining_gates")) if not risk_only.empty else "N/A"
        summary_rows.append({
            "ticker": ticker,
            "current_workflow": text(tdf.iloc[0].get("current_workflow")),
            "current_gates": text(tdf.iloc[0].get("current_gates")),
            "after_risk_clear": text(risk_only.iloc[0].get("simulated_route")) if not risk_only.empty else "N/A",
            "still_blocked_after_risk": still_blocked_after_risk,
            "after_risk_event_clear": text(source.get("simulated_route")),
            "full_clear_route": text(best.get("simulated_route")),
            "full_clear_action": text(best.get("simulated_action")),
            "full_clear_option_permission": text(best.get("simulated_option_permission")),
            "options_unlocked_scenarios": options_unlocked,
            "price_trigger_to_watch": text(best.get("price_trigger_to_watch")),
            "unlock_path_from_step147": text(best.get("unlock_path_from_step147")),
            "research_only": True,
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["options_unlocked_scenarios", "ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)

    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(sim) else "NO_SIMULATION_ROWS",
        "tickers": int(len(tickers)),
        "simulation_rows": int(len(sim)),
        "summary_rows": int(len(summary)),
        "scenarios_per_ticker": int(len(SCENARIOS)),
        "option_research_unlocked_rows": int(sim["simulated_route"].astype(str).str.contains("Option research unlocked", case=False, na=False).sum()) if not sim.empty else 0,
        "hedge_research_unlocked_rows": int(sim["simulated_route"].astype(str).str.contains("Hedge research unlocked", case=False, na=False).sum()) if not sim.empty else 0,
        "risk_first_rows": int(sim["simulated_route"].astype(str).str.contains("Risk first", case=False, na=False).sum()) if not sim.empty else 0,
        "event_review_rows": int(sim["simulated_route"].astype(str).str.contains("Event review", case=False, na=False).sum()) if not sim.empty else 0,
        "monitor_first_rows": int(sim["simulated_route"].astype(str).str.contains("Monitor first", case=False, na=False).sum()) if not sim.empty else 0,
        "outputs": {
            "simulation": OUT_SIM.name,
            "summary": OUT_SUMMARY.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return sim, summary, state


def main() -> int:
    sim, summary, state = build_simulation()
    sim.to_csv(OUT_SIM, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Tickers: {state.get('tickers', 0)}",
        f"- Simulation rows: {state.get('simulation_rows', 0)}",
        f"- Scenarios per ticker: {state.get('scenarios_per_ticker', 0)}",
        f"- Option research unlocked rows: {state.get('option_research_unlocked_rows', 0)}",
        f"- Hedge research unlocked rows: {state.get('hedge_research_unlocked_rows', 0)}",
        f"- Risk-first rows: {state.get('risk_first_rows', 0)}",
        f"- Event-review rows: {state.get('event_review_rows', 0)}",
        "",
        "## Ticker Summary",
        "",
        df_to_markdown(summary, max_rows=80),
        "",
        "## Simulation Matrix",
        "",
        df_to_markdown(sim, max_rows=180),
        "",
        "## Product Truth",
        "",
        "This simulator is a conditional research model. It does not approve live trades, does not connect to a broker, and does not bypass manual review.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 148 - Gate Upgrade Simulator", sections)

    print(f"wrote {OUT_SIM.name} rows={len(sim)}")
    print(f"wrote {OUT_SUMMARY.name} rows={len(summary)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
