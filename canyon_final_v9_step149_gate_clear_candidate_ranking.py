#!/usr/bin/env python3
"""
Canyon v9 - Step 149: Gate-Clear Candidate Ranking
==================================================

Research-only. No broker connection. No live orders.

Step148 simulates what happens when gates clear. Step149 ranks the simulated
candidates by lane so the desk can answer: "If conditions improve tomorrow,
which names deserve attention first, and why?"

Outputs:
  gate_clear_candidate_ranking.csv
  gate_clear_candidate_top5.csv
  gate_clear_candidate_state.json
  gate_clear_candidate_report.md
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

IN_GATE_SUMMARY = ROOT / "gate_upgrade_ticker_summary.csv"
IN_GATE_SIM = ROOT / "gate_upgrade_simulation.csv"
IN_WORKFLOW = ROOT / "daily_workflow_queue.csv"
IN_PICKS = ROOT / "daily_picks_filtered.csv"
IN_RISK = ROOT / "final_risk_gate.csv"
IN_OPTIONS = ROOT / "options_playbook.csv"
IN_EVENT = ROOT / "event_research_dossier.csv"
IN_SECTOR_ROUTE = ROOT / "sector_timeframe_route.csv"
IN_CONFLICT_SUMMARY = ROOT / "decision_conflict_summary.csv"
IN_RESOLUTION_SUMMARY = ROOT / "conflict_resolution_ticker_summary.csv"

OUT_RANKING = ROOT / "gate_clear_candidate_ranking.csv"
OUT_TOP5 = ROOT / "gate_clear_candidate_top5.csv"
OUT_STATE = ROOT / "gate_clear_candidate_state.json"
OUT_REPORT = ROOT / "gate_clear_candidate_report.md"


LANE_ORDER = {
    "Bullish option watch": 0,
    "Equity tiny paper watch": 1,
    "Hedge / protection watch": 2,
    "Sector or event watch": 3,
    "Still blocked after risk clear": 4,
    "Research backlog": 5,
}


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


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not np.isfinite(value):
        return low
    return max(low, min(high, value))


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


def gate_count(value: Any) -> int:
    raw = text(value)
    if not raw or raw.upper() == "NONE":
        return 0
    return len([x for x in raw.split(";") if x.strip()])


def lane_for(row: pd.Series, option_side: str, still_blocked_after_risk: str) -> str:
    full_route = text(row.get("full_clear_route"))
    after_risk = text(row.get("after_risk_clear"))
    after_risk_event = text(row.get("after_risk_event_clear"))
    full_option = text(row.get("full_clear_option_permission"))

    if has_any(full_route, ["OPTION RESEARCH UNLOCKED"]) and has_any(full_option, ["CALL"]):
        return "Bullish option watch"
    if has_any(full_route, ["TINY PAPER"]):
        return "Equity tiny paper watch"
    if has_any(full_route, ["HEDGE"]) or has_any(option_side, ["PUT"]):
        return "Hedge / protection watch"
    if has_any(after_risk, ["EVENT", "SECTOR", "MONITOR"]) or has_any(after_risk_event, ["EVENT", "SECTOR", "MONITOR"]):
        return "Sector or event watch"
    if still_blocked_after_risk and upper(still_blocked_after_risk) != "NONE":
        return "Still blocked after risk clear"
    return "Research backlog"


def route_score(full_route: str, full_option: str) -> float:
    if has_any(full_route, ["OPTION RESEARCH UNLOCKED"]) and has_any(full_option, ["CALL"]):
        return 24.0
    if has_any(full_route, ["TINY PAPER"]):
        return 19.0
    if has_any(full_route, ["HEDGE"]):
        return 14.0
    if has_any(full_route, ["WATCH"]):
        return 8.0
    return 4.0


def sector_adjustment(sector_state: str, linked_state: str) -> float:
    score = 0.0
    if has_any(sector_state, ["LEADERSHIP EXPANSION"]):
        score += 10.0
    elif has_any(sector_state, ["CROWDED LEADERSHIP"]):
        score += 3.0
    elif has_any(sector_state, ["NEUTRAL"]):
        score += 2.0
    elif has_any(sector_state, ["DOWNCYCLE", "LAGGARD", "FADING"]):
        score -= 8.0
    if has_any(linked_state, ["LEADERSHIP"]):
        score += 5.0
    return score


def blocker_label(row: pd.Series, resolution: pd.Series) -> str:
    still = text(row.get("still_blocked_after_risk"))
    current = text(row.get("current_gates"))
    top_gate = text(resolution.get("top_gate"))
    if still and upper(still) != "NONE":
        return still
    if top_gate:
        return top_gate
    return current or "No blocker after full-clear scenario"


def build_ranking() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gate_summary = read_csv_safe(IN_GATE_SUMMARY)
    gate_sim = read_csv_safe(IN_GATE_SIM)
    workflow = one_by_ticker(read_csv_safe(IN_WORKFLOW))
    picks = one_by_ticker(read_csv_safe(IN_PICKS))
    risk = one_by_ticker(read_csv_safe(IN_RISK))
    options = one_by_ticker(read_csv_safe(IN_OPTIONS))
    event = one_by_ticker(read_csv_safe(IN_EVENT))
    sector = one_by_ticker(read_csv_safe(IN_SECTOR_ROUTE))
    conflict = one_by_ticker(read_csv_safe(IN_CONFLICT_SUMMARY))
    resolution = one_by_ticker(read_csv_safe(IN_RESOLUTION_SUMMARY))

    if gate_summary.empty or "ticker" not in gate_summary.columns:
        state = {
            "run_time": now_str(),
            "research_only": True,
            "no_broker_connection": True,
            "status": "NO_GATE_SUMMARY",
            "ranking_rows": 0,
            "top5_rows": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), state

    gate_summary = gate_summary.copy()
    gate_summary["ticker"] = gate_summary["ticker"].astype(str).str.upper().str.strip()
    rows: list[dict[str, Any]] = []

    for _, gs in gate_summary.iterrows():
        ticker = text(gs.get("ticker")).upper()
        if not ticker:
            continue
        wf = row_at(workflow, ticker)
        pick = row_at(picks, ticker)
        rk = row_at(risk, ticker)
        opt = row_at(options, ticker)
        ev = row_at(event, ticker)
        sec = row_at(sector, ticker)
        cf = row_at(conflict, ticker)
        rs = row_at(resolution, ticker)

        alpha = safe_float(pick.get("alpha_score"), 50.0)
        alpha_rank = safe_float(pick.get("alpha_rank"), np.nan)
        action = text(pick.get("action"))
        sector_name = text(pick.get("sector") or wf.get("sector") or sec.get("sector"))
        current_weight = safe_float(rk.get("current_weight_pct") or wf.get("current_weight_pct"), 0.0)
        recommended_weight = safe_float(rk.get("recommended_risk_weight_pct") or wf.get("recommended_weight_pct"), 0.0)
        risk_reduction = safe_float(rk.get("risk_reduction_pct_of_current"), 1.0)
        event_gate = text(ev.get("event_gate") or wf.get("event_gate"))
        event_coverage = safe_float(ev.get("event_source_coverage_pct"), 0.0)
        event_risk = safe_float(ev.get("event_risk_score"), 50.0)
        call_score = safe_float(opt.get("call_score"), 0.0)
        put_score = safe_float(opt.get("put_score"), 0.0)
        option_side = text(opt.get("option_side"))
        option_permission = text(opt.get("option_permission"))
        sector_state = text(sec.get("sector_cycle_state") or wf.get("sector_cycle_state"))
        linked_state = text(sec.get("linked_sector_cycle_state") or wf.get("linked_sector_cycle_state"))
        full_route = text(gs.get("full_clear_route"))
        full_option = text(gs.get("full_clear_option_permission"))
        still_after_risk = text(gs.get("still_blocked_after_risk"))
        current_gates = text(gs.get("current_gates"))
        gates = gate_count(current_gates)
        high_conflicts = safe_float(cf.get("high_conflicts"), 0.0)
        conflict_count = safe_float(cf.get("conflict_count"), 0.0)
        options_unlocked = safe_float(gs.get("options_unlocked_scenarios"), 0.0)
        price_trigger = text(gs.get("price_trigger_to_watch"))
        unlock_path = text(gs.get("unlock_path_from_step147"))

        lane = lane_for(gs, option_side, still_after_risk)
        base = route_score(full_route, full_option)
        alpha_component = clip(alpha) * 0.34
        option_component = max(call_score, put_score) * 0.12
        event_component = clip(event_coverage) * 0.08 - clip(event_risk) * 0.05
        risk_component = max(0.0, 100.0 - clip(risk_reduction * 100.0)) * 0.10
        sector_component = sector_adjustment(sector_state, linked_state)
        conflict_penalty = high_conflicts * 4.0 + max(0.0, conflict_count - high_conflicts) * 1.2
        gate_penalty = gates * 2.0
        current_weight_penalty = max(0.0, current_weight - max(recommended_weight, 0.01)) * 0.7
        readiness = clip(
            base
            + alpha_component
            + option_component
            + event_component
            + risk_component
            + sector_component
            - conflict_penalty
            - gate_penalty
            - current_weight_penalty,
            0.0,
            100.0,
        )

        if lane == "Bullish option watch":
            next_check = "Risk/event/price/execution must all clear before defined-risk call-spread research."
        elif lane == "Equity tiny paper watch":
            next_check = "Risk gate and price confirmation decide whether this can move to tiny paper review."
        elif lane == "Hedge / protection watch":
            next_check = "Treat as protection research only; do not read this as bullish permission."
        elif lane == "Sector or event watch":
            next_check = "Open sector/event sources first; current score is not enough."
        else:
            next_check = "Keep in research backlog until higher gates clear."

        rows.append({
            "ticker": ticker,
            "candidate_lane": lane,
            "lane_rank": LANE_ORDER.get(lane, 99),
            "readiness_score": round(readiness, 2),
            "alpha_score": round(alpha, 2),
            "alpha_rank": int(alpha_rank) if np.isfinite(alpha_rank) else "",
            "action": action,
            "sector": sector_name,
            "sector_cycle_state": sector_state,
            "linked_sector_cycle_state": linked_state,
            "current_workflow": text(gs.get("current_workflow")),
            "current_gates": current_gates,
            "gate_count": gates,
            "after_risk_clear": text(gs.get("after_risk_clear")),
            "after_risk_event_clear": text(gs.get("after_risk_event_clear")),
            "full_clear_route": full_route,
            "full_clear_action": text(gs.get("full_clear_action")),
            "full_clear_option_permission": full_option,
            "option_side": option_side,
            "option_permission_now": option_permission,
            "call_score": round(call_score, 2),
            "put_score": round(put_score, 2),
            "event_gate": event_gate,
            "event_source_coverage_pct": round(event_coverage, 2),
            "event_risk_score": round(event_risk, 2),
            "current_weight_pct": round(current_weight, 3),
            "recommended_weight_pct": round(recommended_weight, 3),
            "risk_reduction_pct": round(risk_reduction, 3),
            "conflict_count": int(conflict_count),
            "high_conflicts": int(high_conflicts),
            "options_unlocked_scenarios": int(options_unlocked),
            "price_trigger_to_watch": price_trigger,
            "main_blocker": blocker_label(gs, rs),
            "why_ranked_here": shorten(
                f"Lane={lane}; full-clear route={full_route}; alpha={alpha:.1f}; "
                f"risk reduction={risk_reduction:.1%}; sector={sector_state}; event={event_gate}; "
                f"high conflicts={high_conflicts:.0f}."
            ),
            "next_check": next_check,
            "unlock_path_from_step147": shorten(unlock_path),
            "source_files": "gate_upgrade_ticker_summary.csv; gate_upgrade_simulation.csv; daily_picks_filtered.csv; final_risk_gate.csv; options_playbook.csv; event_research_dossier.csv; sector_timeframe_route.csv",
            "research_only": True,
            "no_broker_connection": True,
        })

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            ["lane_rank", "readiness_score", "alpha_score", "ticker"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        ranking["overall_rank"] = np.arange(1, len(ranking) + 1)
        cols = ["overall_rank"] + [c for c in ranking.columns if c != "overall_rank"]
        ranking = ranking[cols]

    top_rows: list[pd.DataFrame] = []
    if not ranking.empty:
        top_rows.append(ranking.head(5).assign(top_bucket="Overall top 5"))
        for lane in LANE_ORDER:
            lane_rows = ranking[ranking["candidate_lane"] == lane].head(5).copy()
            if not lane_rows.empty:
                lane_rows["top_bucket"] = f"{lane} top 5"
                top_rows.append(lane_rows)
    top5 = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    if not top5.empty:
        cols = ["top_bucket"] + [c for c in top5.columns if c != "top_bucket"]
        top5 = top5[cols]

    lane_counts = ranking["candidate_lane"].value_counts().to_dict() if not ranking.empty else {}
    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "status": "READY" if len(ranking) else "NO_RANKING_ROWS",
        "ranking_rows": int(len(ranking)),
        "top5_rows": int(len(top5)),
        "overall_top_5": ranking["ticker"].head(5).tolist() if not ranking.empty else [],
        "bullish_option_watch_count": int(lane_counts.get("Bullish option watch", 0)),
        "equity_tiny_paper_watch_count": int(lane_counts.get("Equity tiny paper watch", 0)),
        "hedge_watch_count": int(lane_counts.get("Hedge / protection watch", 0)),
        "sector_event_watch_count": int(lane_counts.get("Sector or event watch", 0)),
        "still_blocked_count": int(lane_counts.get("Still blocked after risk clear", 0)),
        "research_backlog_count": int(lane_counts.get("Research backlog", 0)),
        "outputs": {
            "ranking": OUT_RANKING.name,
            "top5": OUT_TOP5.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    return ranking, top5, state


def main() -> int:
    ranking, top5, state = build_ranking()
    ranking.to_csv(OUT_RANKING, index=False)
    top5.to_csv(OUT_TOP5, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Status: {state.get('status', 'NO_DATA')}",
        f"- Ranking rows: {state.get('ranking_rows', 0)}",
        f"- Top5 rows: {state.get('top5_rows', 0)}",
        f"- Overall top 5: {', '.join(state.get('overall_top_5', []))}",
        f"- Bullish option watch: {state.get('bullish_option_watch_count', 0)}",
        f"- Equity tiny paper watch: {state.get('equity_tiny_paper_watch_count', 0)}",
        f"- Hedge watch: {state.get('hedge_watch_count', 0)}",
        "",
        "## Top Buckets",
        "",
        df_to_markdown(top5, max_rows=80),
        "",
        "## Full Ranking",
        "",
        df_to_markdown(ranking, max_rows=160),
        "",
        "## Product Truth",
        "",
        "This ranking is a research attention queue. It is not a buy list, not an order ticket, and not an option approval engine.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 149 - Gate-Clear Candidate Ranking", sections)

    print(f"wrote {OUT_RANKING.name} rows={len(ranking)}")
    print(f"wrote {OUT_TOP5.name} rows={len(top5)}")
    print(f"status={state.get('status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
