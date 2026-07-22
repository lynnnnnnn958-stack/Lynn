#!/usr/bin/env python3
"""
Canyon v9 - Step 135: Risk Policy Evidence Pack
===============================================

Research-only. No broker connection. No live orders.

Step134 creates stable policy tickets. Step135 attaches an evidence pack to
each ticket so a reviewer can see why the ticket exists, which files created it,
what the strongest numbers are, and what evidence is still missing.

Outputs:
  risk_policy_evidence_pack.csv
  risk_policy_evidence_items.csv
  risk_policy_evidence_state.json
  risk_policy_evidence_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    df_to_markdown,
    now_str,
    read_csv_safe,
    read_json_safe,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

IN_TICKETS = ROOT / "risk_policy_open_tickets.csv"

OUT_PACK = ROOT / "risk_policy_evidence_pack.csv"
OUT_ITEMS = ROOT / "risk_policy_evidence_items.csv"
OUT_STATE = ROOT / "risk_policy_evidence_state.json"
OUT_REPORT = ROOT / "risk_policy_evidence_report.md"


CSV_SOURCES = {
    "risk_desk_breach_table.csv": ROOT / "risk_desk_breach_table.csv",
    "risk_desk_ticker_action_queue.csv": ROOT / "risk_desk_ticker_action_queue.csv",
    "risk_threshold_calibration.csv": ROOT / "risk_threshold_calibration.csv",
    "risk_policy_review_queue.csv": ROOT / "risk_policy_review_queue.csv",
    "portfolio_var_cvar_summary.csv": ROOT / "portfolio_var_cvar_summary.csv",
    "single_name_risk_budget.csv": ROOT / "single_name_risk_budget.csv",
    "sector_active_exposure.csv": ROOT / "sector_active_exposure.csv",
    "factor_exposure_decomposition.csv": ROOT / "factor_exposure_decomposition.csv",
    "portfolio_beta_report.csv": ROOT / "portfolio_beta_report.csv",
    "macro_scenario_stress.csv": ROOT / "macro_scenario_stress.csv",
    "portfolio_macro_sensitivity.csv": ROOT / "portfolio_macro_sensitivity.csv",
    "crisis_correlation_stress.csv": ROOT / "crisis_correlation_stress.csv",
    "crisis_correlation_override.csv": ROOT / "crisis_correlation_override.csv",
    "earnings_gap_down_risk.csv": ROOT / "earnings_gap_down_risk.csv",
    "liquidity_crisis_simulation.csv": ROOT / "liquidity_crisis_simulation.csv",
    "kelly_position_sizing.csv": ROOT / "kelly_position_sizing.csv",
    "paper_nav_curve.csv": ROOT / "paper_nav_curve.csv",
    "risk_historical_evidence_summary.csv": ROOT / "risk_historical_evidence_summary.csv",
    "risk_historical_policy_bridge.csv": ROOT / "risk_historical_policy_bridge.csv",
    "risk_historical_correlation_windows.csv": ROOT / "risk_historical_correlation_windows.csv",
    "risk_historical_drawdown_windows.csv": ROOT / "risk_historical_drawdown_windows.csv",
    "risk_historical_liquidity_ticker_stress.csv": ROOT / "risk_historical_liquidity_ticker_stress.csv",
    "risk_historical_threshold_recommendations.csv": ROOT / "risk_historical_threshold_recommendations.csv",
    "risk_historical_threshold_review_queue.csv": ROOT / "risk_historical_threshold_review_queue.csv",
    "risk_historical_threshold_deltas.csv": ROOT / "risk_historical_threshold_deltas.csv",
    "risk_threshold_impact_simulation.csv": ROOT / "risk_threshold_impact_simulation.csv",
    "risk_threshold_impact_decision_table.csv": ROOT / "risk_threshold_impact_decision_table.csv",
    "risk_policy_change_control.csv": ROOT / "risk_policy_change_control.csv",
    "risk_policy_change_readiness.csv": ROOT / "risk_policy_change_readiness.csv",
    "risk_policy_implementation_plan.csv": ROOT / "risk_policy_implementation_plan.csv",
    "risk_policy_approval_checklist.csv": ROOT / "risk_policy_approval_checklist.csv",
    "risk_policy_approval_packet.csv": ROOT / "risk_policy_approval_packet.csv",
    "risk_policy_manual_approval_template.csv": ROOT / "risk_policy_manual_approval_template.csv",
    "risk_policy_dry_run_readiness.csv": ROOT / "risk_policy_dry_run_readiness.csv",
    "risk_policy_dry_run_plan.csv": ROOT / "risk_policy_dry_run_plan.csv",
    "risk_policy_dry_run_monitor.csv": ROOT / "risk_policy_dry_run_monitor.csv",
}

JSON_SOURCES = {
    "risk_desk_overview.json": ROOT / "risk_desk_overview.json",
    "risk_policy_decision_state.json": ROOT / "risk_policy_decision_state.json",
    "risk_policy_review_state.json": ROOT / "risk_policy_review_state.json",
    "risk_threshold_calibration_state.json": ROOT / "risk_threshold_calibration_state.json",
    "institutional_risk_gate_state.json": ROOT / "institutional_risk_gate_state.json",
    "drawdown_control_state.json": ROOT / "drawdown_control_state.json",
    "vol_target_state.json": ROOT / "vol_target_state.json",
    "risk_policy_approval_state.json": ROOT / "risk_policy_approval_state.json",
    "risk_policy_dry_run_state.json": ROOT / "risk_policy_dry_run_state.json",
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def pct(value: Any) -> str:
    x = safe_float(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x * 100:.2f}%"


def compact_row(row: pd.Series, max_fields: int = 8) -> str:
    parts = []
    for col in row.index[:max_fields]:
        val = row.get(col)
        if pd.isna(val):
            continue
        parts.append(f"{col}={val}")
    return "; ".join(parts)


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    csvs = {name: read_csv_safe(path) for name, path in CSV_SOURCES.items()}
    jsons = {name: read_json_safe(path) for name, path in JSON_SOURCES.items()}
    return csvs, jsons


def add_item(
    items: list[dict[str, Any]],
    ticket: pd.Series,
    evidence_type: str,
    source_file: str,
    evidence_key: str,
    evidence_value: Any,
    evidence_note: str,
    rank: int,
) -> None:
    items.append({
        "ticket_id": ticket.get("ticket_id", ""),
        "control": ticket.get("control", ""),
        "risk_area": ticket.get("risk_area", ""),
        "review_priority": ticket.get("review_priority", ""),
        "evidence_type": evidence_type,
        "source_file": source_file,
        "evidence_key": evidence_key,
        "evidence_value": evidence_value,
        "evidence_note": evidence_note,
        "evidence_rank": rank,
        "research_only": True,
    })


def add_json_items(items: list[dict[str, Any]], ticket: pd.Series, jsons: dict[str, dict[str, Any]], source: str, keys: list[str], start_rank: int) -> int:
    data = jsons.get(source, {}) or {}
    rank = start_rank
    for key in keys:
        if key in data:
            add_item(items, ticket, "state", source, key, data.get(key), "State value used by the risk desk.", rank)
            rank += 1
    return rank


def add_matching_control_rows(items: list[dict[str, Any]], ticket: pd.Series, csvs: dict[str, pd.DataFrame], source: str, control_col: str, value: str, start_rank: int) -> int:
    df = csvs.get(source, pd.DataFrame())
    if df.empty or control_col not in df.columns:
        return start_rank
    m = df[control_col].astype(str).str.lower().eq(str(value).lower())
    rank = start_rank
    for _, row in df.loc[m].head(5).iterrows():
        add_item(items, ticket, "matched_row", source, control_col, compact_row(row, max_fields=10), "Direct row matching this policy control.", rank)
        rank += 1
    return rank


def add_top_rows(
    items: list[dict[str, Any]],
    ticket: pd.Series,
    csvs: dict[str, pd.DataFrame],
    source: str,
    sort_col: str,
    note: str,
    start_rank: int,
    ascending: bool = False,
    abs_sort: bool = False,
    limit: int = 5,
) -> int:
    df = csvs.get(source, pd.DataFrame())
    if df.empty or sort_col not in df.columns:
        return start_rank
    work = df.copy()
    metric = pd.to_numeric(work[sort_col], errors="coerce")
    work["_sort_metric"] = metric.abs() if abs_sort else metric
    work = work.dropna(subset=["_sort_metric"]).sort_values("_sort_metric", ascending=ascending)
    rank = start_rank
    for _, row in work.head(limit).iterrows():
        key = text(row.get("ticker") or row.get("sector") or row.get("scenario") or row.get("factor") or row.get("stress_name") or sort_col)
        add_item(items, ticket, "top_row", source, key, compact_row(row.drop(labels=["_sort_metric"], errors="ignore"), max_fields=10), note, rank)
        rank += 1
    return rank


def evidence_for_ticket(ticket: pd.Series, csvs: dict[str, pd.DataFrame], jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    control = text(ticket.get("control"))
    risk_area = text(ticket.get("risk_area"))
    source_file = text(ticket.get("source_file"))
    rank = 1

    add_item(items, ticket, "ticket", "risk_policy_open_tickets.csv", "policy_ticket", compact_row(ticket, max_fields=14), "Current open policy ticket.", rank)
    rank += 1
    rank = add_matching_control_rows(items, ticket, csvs, "risk_threshold_calibration.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_review_queue.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_desk_breach_table.csv", "budget_item", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_historical_evidence_summary.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_historical_policy_bridge.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_historical_threshold_recommendations.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_historical_threshold_review_queue.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_historical_threshold_deltas.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_threshold_impact_simulation.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_threshold_impact_decision_table.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_change_control.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_implementation_plan.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_approval_checklist.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_approval_packet.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_manual_approval_template.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_dry_run_readiness.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_dry_run_plan.csv", "control", control, rank)
    rank = add_matching_control_rows(items, ticket, csvs, "risk_policy_dry_run_monitor.csv", "control", control, rank)

    rank = add_json_items(
        items,
        ticket,
        jsons,
        "risk_desk_overview.json",
        ["master_risk_action", "master_exposure_multiplier", "normal_gross_exposure", "recommended_gross_exposure", "var_95_1d_pct", "cvar_95_1d_pct", "worst_macro_scenario", "worst_macro_impact_pct"],
        rank,
    )
    rank = add_json_items(
        items,
        ticket,
        jsons,
        "risk_policy_approval_state.json",
        ["overall_status", "changes_checked", "blocked_count", "waiting_manual_approval_count", "ready_for_dry_run_count", "qa_gap_count", "can_apply_live_count"],
        rank,
    )
    rank = add_json_items(
        items,
        ticket,
        jsons,
        "risk_policy_dry_run_state.json",
        ["overall_status", "changes_checked", "dry_run_ready_count", "waiting_manual_approval_count", "blocked_count", "qa_gap_count", "can_apply_live_count"],
        rank,
    )

    lc = control.lower()
    la = risk_area.lower()
    if "single-name" in lc or "single" in la:
        rank = add_top_rows(items, ticket, csvs, "single_name_risk_budget.csv", "risk_budget_used_pct", "Worst single-name risk-budget usage.", rank, limit=8)
        rank = add_top_rows(items, ticket, csvs, "single_name_risk_budget.csv", "cvar_95_1d", "Worst single-name 1d CVaR.", rank, limit=8)
        rank = add_top_rows(items, ticket, csvs, "risk_desk_ticker_action_queue.csv", "risk_reduction_pct_of_current", "Tickers with largest required risk reduction.", rank, limit=8)
    if "macro" in lc or "macro" in la:
        rank = add_top_rows(items, ticket, csvs, "macro_scenario_stress.csv", "conservative_portfolio_impact", "Worst macro scenario impact.", rank, abs_sort=True, limit=8)
        rank = add_top_rows(items, ticket, csvs, "portfolio_macro_sensitivity.csv", "estimated_20d_portfolio_impact", "Largest factor sensitivity estimates.", rank, abs_sort=True, limit=8)
    if "volatility" in lc or "var" in lc or "cvar" in lc or "tail risk" in la:
        rank = add_top_rows(items, ticket, csvs, "portfolio_var_cvar_summary.csv", "cvar_95_1d", "Portfolio VaR/CVaR summary.", rank, limit=3)
        rank = add_json_items(items, ticket, jsons, "vol_target_state.json", ["annual_vol", "target_vol", "vol_exposure_multiplier", "vol_action"], rank)
    if "earnings" in lc or "event" in la:
        rank = add_top_rows(items, ticket, csvs, "earnings_gap_down_risk.csv", "estimated_gap_loss_model_account", "Largest estimated earnings gap losses.", rank, limit=8)
    if "sector" in lc or "sector" in la:
        rank = add_top_rows(items, ticket, csvs, "sector_active_exposure.csv", "cap_used_pct", "Largest sector cap usage.", rank, limit=8)
    if "factor" in lc or "beta" in lc or "factor" in la:
        rank = add_top_rows(items, ticket, csvs, "portfolio_beta_report.csv", "abs_beta", "Largest proxy factor betas.", rank, limit=8)
        rank = add_top_rows(items, ticket, csvs, "factor_exposure_decomposition.csv", "estimated_20d_impact", "Largest estimated factor impacts.", rank, abs_sort=True, limit=8)
    if "correlation" in lc or "correlation" in la:
        rank = add_top_rows(items, ticket, csvs, "crisis_correlation_stress.csv", "vol_increase_ratio", "Crisis-correlation stress estimate.", rank, limit=5)
        rank = add_top_rows(items, ticket, csvs, "crisis_correlation_override.csv", "max_pair_corr", "Crisis correlation override evidence.", rank, limit=5)
        rank = add_top_rows(items, ticket, csvs, "risk_historical_correlation_windows.csv", "stress_vol_ratio", "Worst historical proxy correlation-stress windows.", rank, limit=8)
    if "liquidity" in lc or "liquidity" in la:
        rank = add_top_rows(items, ticket, csvs, "liquidity_crisis_simulation.csv", "estimated_liquidation_loss", "Largest liquidity-crisis liquidation losses.", rank, limit=8)
        rank = add_top_rows(items, ticket, csvs, "risk_historical_liquidity_ticker_stress.csv", "historical_stress_loss_model_account", "Largest historical proxy liquidity-stress losses.", rank, limit=8)
    if "drawdown" in lc or "drawdown" in la:
        rank = add_json_items(items, ticket, jsons, "drawdown_control_state.json", ["drawdown_pct", "drawdown_action", "current_nav", "high_water_mark", "exposure_multiplier"], rank)
        rank = add_top_rows(items, ticket, csvs, "paper_nav_curve.csv", "drawdown", "Paper NAV drawdown history.", rank, ascending=True, limit=5)
        rank = add_top_rows(items, ticket, csvs, "risk_historical_drawdown_windows.csv", "drawdown_loss", "Worst historical proxy current-book drawdown dates.", rank, limit=8)

    if source_file and source_file not in CSV_SOURCES and source_file not in JSON_SOURCES:
        add_item(items, ticket, "source_reference", source_file, "source_file", source_file, "Referenced source from upstream ticket; file may be a composite source string.", rank)

    return items


def evidence_grade(item_count: int, ticket: pd.Series, items: pd.DataFrame | None = None) -> str:
    mode = text(ticket.get("calibration_mode")).upper()
    status = text(ticket.get("calibration_status")).upper()
    sample_n = int(safe_float(ticket.get("sample_n"), 0))
    if status == "MISSING_HISTORY" and items is not None and not items.empty:
        source_text = " ".join(items.get("source_file", pd.Series(dtype=str)).dropna().astype(str).str.lower().tolist())
        value_text = " ".join(items.get("evidence_value", pd.Series(dtype=str)).dropna().astype(str).str.upper().tolist())
        if "risk_historical_" in source_text and "PROXY_HISTORY_USABLE" in value_text:
            return "HISTORICAL_PROXY"
    if item_count >= 8 and mode == "HISTORICAL_ROLLING" and sample_n >= 180:
        return "STRONG"
    if item_count >= 6 and mode in {"CURRENT_CROSS_SECTION", "SCENARIO_LIBRARY"}:
        return "USABLE"
    if item_count >= 4 and status == "MISSING_HISTORY":
        return "NEEDS_HISTORY"
    if item_count >= 3:
        return "PARTIAL"
    return "THIN"


def build_pack(tickets: pd.DataFrame, all_items: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rows = []
    for _, ticket in tickets.iterrows():
        tid = text(ticket.get("ticket_id"))
        sub = all_items[all_items["ticket_id"].astype(str) == tid] if not all_items.empty else pd.DataFrame()
        primary = sub.sort_values("evidence_rank").head(1)
        sources = sorted(set(sub["source_file"].dropna().astype(str))) if not sub.empty and "source_file" in sub.columns else []
        grade = evidence_grade(len(sub), ticket, sub)
        rows.append({
            "ticket_id": tid,
            "review_priority": ticket.get("review_priority", ""),
            "review_bucket": ticket.get("review_bucket", ""),
            "control": ticket.get("control", ""),
            "risk_area": ticket.get("risk_area", ""),
            "decision_status": ticket.get("decision_status", ""),
            "policy_decision_candidate": ticket.get("policy_decision_candidate", ""),
            "evidence_grade": grade,
            "evidence_item_count": int(len(sub)),
            "primary_evidence": primary["evidence_value"].iloc[0] if not primary.empty else "",
            "source_files": "; ".join(sources[:10]),
            "evidence_gap": evidence_gap(grade, ticket),
            "review_question": review_question(ticket),
            "required_next_action": ticket.get("required_next_action", ""),
            "research_only": True,
        })
    return pd.DataFrame(rows).sort_values(["review_priority", "evidence_grade", "control"]).reset_index(drop=True)


def evidence_gap(grade: str, ticket: pd.Series) -> str:
    mode = text(ticket.get("calibration_mode")).upper()
    if grade in {"STRONG", "USABLE"}:
        return "Evidence is sufficient for review, but still research-only."
    if grade == "HISTORICAL_PROXY":
        return "Local proxy history is now attached; still needs true point-in-time institutional data before production use."
    if mode == "POINT_ESTIMATE":
        return "Needs repeated point-in-time observations or crisis-window history."
    if mode == "CURRENT_CROSS_SECTION":
        return "Needs historical pre-trade snapshots, not only today's cross-section."
    if mode == "SCENARIO_LIBRARY":
        return "Needs scenario library backtest and regime-conditioned validation."
    return "Needs more direct source rows."


def review_question(ticket: pd.Series) -> str:
    decision = text(ticket.get("policy_decision_candidate")).upper()
    control = text(ticket.get("control"))
    if decision == "KEEP_CURRENT_LIMIT_AND_DE_RISK":
        return f"Does the current book need immediate size reduction before any {control} policy change is considered?"
    if decision == "TIGHTEN_LIMIT_REVIEW":
        return f"Should the {control} hard limit be tightened after reviewing false-positive risk?"
    if decision == "ADD_DRAFT_LIMIT_REVIEW":
        return f"What warning and hard limits should become the first explicit policy for {control}?"
    if decision == "KEEP_LIMIT_COLLECT_HISTORY":
        return f"What history is required before {control} can be treated as calibrated?"
    return f"Should {control} remain unchanged for the next daily run?"


def build_state(pack: pd.DataFrame, items: pd.DataFrame) -> dict[str, Any]:
    if pack.empty:
        return {
            "run_time": now_str(),
            "overall_status": "NO_DATA",
            "research_only": True,
            "no_broker_connection": True,
        }
    grades = pack["evidence_grade"].astype(str).str.upper()
    if grades.isin(["THIN", "PARTIAL"]).any():
        status = "EVIDENCE_GAPS"
    elif grades.eq("NEEDS_HISTORY").any():
        status = "NEEDS_HISTORY"
    elif grades.eq("HISTORICAL_PROXY").any():
        status = "HISTORICAL_PROXY_REVIEW"
    else:
        status = "EVIDENCE_READY"
    return {
        "run_time": now_str(),
        "overall_status": status,
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Attach evidence to each policy ticket. Evidence supports review; it does not approve or apply policy changes.",
        "tickets_with_evidence": int(len(pack)),
        "evidence_items": int(len(items)),
        "strong_count": int((grades == "STRONG").sum()),
        "usable_count": int((grades == "USABLE").sum()),
        "needs_history_count": int((grades == "NEEDS_HISTORY").sum()),
        "historical_proxy_count": int((grades == "HISTORICAL_PROXY").sum()),
        "partial_or_thin_count": int(grades.isin(["PARTIAL", "THIN"]).sum()),
        "outputs": {
            "evidence_pack": OUT_PACK.name,
            "evidence_items": OUT_ITEMS.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }


def main() -> int:
    tickets = read_csv_safe(IN_TICKETS)
    csvs, jsons = load_sources()
    all_items: list[dict[str, Any]] = []

    if not tickets.empty:
        for _, ticket in tickets.iterrows():
            all_items.extend(evidence_for_ticket(ticket, csvs, jsons))

    items = pd.DataFrame(all_items)
    if not items.empty:
        items = items.sort_values(["review_priority", "ticket_id", "evidence_rank"]).reset_index(drop=True)
    pack = build_pack(tickets, items)
    state = build_state(pack, items)

    pack.to_csv(OUT_PACK, index=False)
    items.to_csv(OUT_ITEMS, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state.get('overall_status', 'NO_DATA')}**",
        f"- Tickets with evidence: {state.get('tickets_with_evidence', 0)}",
        f"- Evidence items: {state.get('evidence_items', 0)}",
        f"- Strong evidence: {state.get('strong_count', 0)}",
        f"- Usable evidence: {state.get('usable_count', 0)}",
        f"- Historical proxy evidence: {state.get('historical_proxy_count', 0)}",
        f"- Needs history: {state.get('needs_history_count', 0)}",
        f"- Partial/thin: {state.get('partial_or_thin_count', 0)}",
        "",
        "## Evidence Pack",
        "",
        df_to_markdown(pack, max_rows=50),
        "",
        "## Evidence Items",
        "",
        df_to_markdown(items, max_rows=120),
        "",
        "## Product Truth",
        "",
        "This evidence pack supports human review. It does not approve, reject, or apply any policy change.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 135 - Risk Policy Evidence Pack", sections)

    print(f"wrote {OUT_PACK.name} rows={len(pack)}")
    print(f"wrote {OUT_ITEMS.name} rows={len(items)}")
    print(f"overall_status={state.get('overall_status', 'NO_DATA')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
