#!/usr/bin/env python3
"""
Canyon v9 Step 172 - Institutional Depth Upgrade Engine.

Research-only. No broker connection. No live orders.

This step is a cross-module control layer. It does not pretend Canyon is a
top-tier institutional quant platform. It audits the actual local outputs and
turns the user's 13 gap areas into a measurable workbench:

  module -> current readiness -> evidence -> weak control -> next upgrade

Outputs:
  institutional_depth_module_scorecard.csv
  institutional_depth_control_map.csv
  institutional_depth_upgrade_queue.csv
  institutional_depth_gap_matrix.csv
  institutional_depth_state.json
  institutional_depth_report.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_SCORECARD = ROOT / "institutional_depth_module_scorecard.csv"
OUT_CONTROL_MAP = ROOT / "institutional_depth_control_map.csv"
OUT_QUEUE = ROOT / "institutional_depth_upgrade_queue.csv"
OUT_GAP = ROOT / "institutional_depth_gap_matrix.csv"
OUT_STATE = ROOT / "institutional_depth_state.json"
OUT_REPORT = ROOT / "institutional_depth_report.md"


HARD_WORDS = (
    "BLOCK",
    "BLOCKER",
    "FAIL",
    "FAILED",
    "CRITICAL",
    "REDUCE_ONLY",
    "NO_NEW",
    "NO DATA",
    "NO_DATA",
    "MISSING",
    "NOT_IN",
)
REVIEW_WORDS = (
    "REVIEW",
    "WEAK",
    "WATCH",
    "SIZE_DOWN",
    "MANUAL",
    "CONTEXT",
    "PROTOTYPE",
    "LOW_SAMPLE",
    "UNPROVEN",
)
CLEAR_WORDS = (
    "CLEAR",
    "OK",
    "PASS",
    "READY",
    "ACTIVE",
    "CONFIRMED",
    "OPERATIONAL",
)


@dataclass(frozen=True)
class ControlSpec:
    name: str
    files: tuple[str, ...]
    min_rows: int
    target_state: str
    institution_gap: str
    priority: str
    layer: str


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    module: str
    user_baseline_low: float
    user_baseline_high: float
    readiness_cap_pct: float
    top_institution_requires: str
    controls: tuple[ControlSpec, ...]


def file_age_hours(path: Path) -> float:
    if not path.exists():
        return np.inf
    return max(0.0, (time.time() - path.stat().st_mtime) / 3600.0)


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def status_from_score(score: float) -> str:
    if score >= 75:
        return "STRONG_PROTOTYPE"
    if score >= 62:
        return "ACTIVE_PROTOTYPE"
    if score >= 50:
        return "DEVELOPING"
    if score >= 35:
        return "WEAK"
    return "MISSING_OR_BLOCKED"


def readiness_band(score: float) -> str:
    if score >= 75:
        return "Near institutional prototype"
    if score >= 62:
        return "Strong research prototype"
    if score >= 50:
        return "Usable but incomplete"
    if score >= 35:
        return "Early prototype"
    return "Major gap"


def scan_status_text(text: str) -> tuple[int, int, int]:
    upper = str(text or "").upper()
    hard = sum(1 for w in HARD_WORDS if w in upper)
    review = sum(1 for w in REVIEW_WORDS if w in upper)
    clear = sum(1 for w in CLEAR_WORDS if w in upper)
    return hard, review, clear


def status_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        low = str(col).lower()
        if any(key in low for key in ("status", "action", "gate", "permission", "severity", "decision", "risk")):
            out.append(col)
    return out


def csv_file_score(path: Path, min_rows: int) -> dict[str, Any]:
    df = read_csv_safe(path)
    if df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "file_score": 12.0 if exists_nonempty(path) else 0.0,
            "flag_summary": "empty or unreadable",
            "sample_evidence": "No usable rows.",
        }

    rows = int(len(df))
    cols = int(len(df.columns))
    row_score = min(35.0, 35.0 * rows / max(float(min_rows), 1.0))
    col_score = min(10.0, cols / 2.0)
    age = file_age_hours(path)
    age_score = 10.0 if age <= 48 else (6.0 if age <= 24 * 7 else 2.0)

    s_cols = status_columns(df)
    if s_cols:
        texts = df[s_cols].head(250).apply(lambda r: " | ".join(str(x) for x in r.to_list()), axis=1)
        hard = review = clear = 0
        for text in texts:
            h, r, c = scan_status_text(text)
            hard += h
            review += r
            clear += c
        denom = max(len(texts), 1)
        hard_rate = min(1.0, hard / denom)
        review_rate = min(1.0, review / denom)
        clear_rate = min(1.0, clear / denom)
        status_score = 22.0 - 15.0 * hard_rate - 7.0 * review_rate + 5.0 * clear_rate
        status_score = float(np.clip(status_score, 3.0, 25.0))
        flag_summary = f"status scan: hard={hard}, review={review}, clear={clear}"
    else:
        status_score = 16.0
        flag_summary = "no explicit status column"

    sample_cols = [c for c in df.columns[:5]]
    sample_evidence = f"{rows} rows, {cols} columns"
    if sample_cols:
        first = df.iloc[0][sample_cols].to_dict()
        sample_evidence += f"; first row keys={first}"
    score = 15.0 + row_score + col_score + age_score + status_score
    return {
        "rows": rows,
        "columns": cols,
        "file_score": round(float(np.clip(score, 0.0, 92.0)), 1),
        "flag_summary": flag_summary,
        "sample_evidence": sample_evidence[:700],
    }


def json_file_score(path: Path) -> dict[str, Any]:
    data = read_json_safe(path, {})
    if not data:
        return {
            "rows": 0,
            "columns": 0,
            "file_score": 12.0 if exists_nonempty(path) else 0.0,
            "flag_summary": "empty or unreadable json",
            "sample_evidence": "No usable JSON state.",
        }
    age = file_age_hours(path)
    age_score = 10.0 if age <= 48 else (6.0 if age <= 24 * 7 else 2.0)
    text = " | ".join(str(v) for v in data.values())
    hard, review, clear = scan_status_text(text)
    status_score = float(np.clip(23.0 - 4.0 * hard - 2.0 * review + 2.0 * clear, 4.0, 25.0))
    score = 35.0 + min(15.0, len(data) * 1.5) + age_score + status_score
    return {
        "rows": 1,
        "columns": len(data),
        "file_score": round(float(np.clip(score, 0.0, 90.0)), 1),
        "flag_summary": f"json status scan: hard={hard}, review={review}, clear={clear}",
        "sample_evidence": str({k: data.get(k) for k in list(data.keys())[:6]})[:700],
    }


def text_file_score(path: Path, min_rows: int) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {
            "rows": 0,
            "columns": 0,
            "file_score": 8.0,
            "flag_summary": "text file unreadable",
            "sample_evidence": "File exists but could not be read as text.",
        }
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = len(lines)
    age = file_age_hours(path)
    age_score = 10.0 if age <= 48 else (6.0 if age <= 24 * 7 else 2.0)
    expected_lines = max(60, min_rows * 20) if path.suffix.lower() == ".py" else max(20, min_rows * 8)
    line_score = min(35.0, 35.0 * line_count / float(expected_lines))
    hard, review, clear = scan_status_text(text[:25000])
    status_score = float(np.clip(18.0 - 1.2 * hard - 0.8 * review + 0.6 * clear, 4.0, 24.0))
    keyword_bonus = 0.0
    low = text.lower()
    for key in ("research-only", "no broker", "no live order", "streamlit", "outputs", "scorecard", "dashboard"):
        if key in low:
            keyword_bonus += 1.5
    score = 20.0 + line_score + age_score + status_score + min(keyword_bonus, 10.0)
    return {
        "rows": line_count,
        "columns": 1,
        "file_score": round(float(np.clip(score, 0.0, 90.0)), 1),
        "flag_summary": f"text scan: hard={hard}, review={review}, clear={clear}, lines={line_count}",
        "sample_evidence": f"{path.name} has {line_count} non-empty lines and {len(text)} characters."[:700],
    }


def assess_file(file_name: str, min_rows: int) -> dict[str, Any]:
    path = ROOT / file_name
    suffix = path.suffix.lower()
    exists = exists_nonempty(path)
    if not exists:
        return {
            "file": file_name,
            "exists": False,
            "age_hours": np.inf,
            "rows": 0,
            "columns": 0,
            "file_score": 0.0,
            "flag_summary": "missing",
            "sample_evidence": "Missing file.",
        }
    if suffix == ".json":
        base = json_file_score(path)
    elif suffix == ".csv":
        base = csv_file_score(path, min_rows)
    else:
        base = text_file_score(path, min_rows)
    base.update({
        "file": file_name,
        "exists": True,
        "age_hours": round(file_age_hours(path), 2),
    })
    return base


def assess_control(module_id: str, module: str, control: ControlSpec) -> dict[str, Any]:
    file_scores = [assess_file(f, control.min_rows) for f in control.files]
    present = [f for f in file_scores if f["exists"]]
    if not present:
        score = 0.0
        status = "MISSING"
        evidence = "Missing all required files."
        weakest = control.institution_gap
    else:
        score = float(np.mean([f["file_score"] for f in present]))
        missing_count = len(file_scores) - len(present)
        if missing_count:
            score = max(0.0, score - min(18.0, missing_count * 6.0))
        status = status_from_score(score)
        evidence = "; ".join(f"{f['file']}={f['rows']} rows" for f in present[:4])
        weakest = "; ".join(f["file"] for f in file_scores if not f["exists"]) or min(present, key=lambda f: f["file_score"])["flag_summary"]

    return {
        "module_id": module_id,
        "module": module,
        "control": control.name,
        "layer": control.layer,
        "control_score_pct": round(score, 1),
        "control_status": status,
        "priority": control.priority,
        "target_state": control.target_state,
        "institution_gap": control.institution_gap,
        "evidence_files": " | ".join(control.files),
        "present_files": " | ".join(f["file"] for f in present) if present else "",
        "missing_files": " | ".join(f["file"] for f in file_scores if not f["exists"]),
        "evidence": evidence,
        "weakest_point": weakest,
        "freshest_age_hours": min([f["age_hours"] for f in present], default=np.inf),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def specs() -> tuple[ModuleSpec, ...]:
    return (
        ModuleSpec(
            "M01",
            "Investment research framework",
            55,
            60,
            70,
            "Analyst-grade thesis, source-backed proof, reject/size rules, and clear promotion gates.",
            (
                ControlSpec("Research promotion gate", ("research_promotion_gate.csv", "research_promotion_component_scores.csv"), 20, "Each ticker has promotion status, blockers, and proof required.", "Promotion still depends on local/proxy evidence and manual checks.", "P1", "Research"),
                ControlSpec("Ticker decision room", ("ticker_decision_room.csv", "ticker_evidence_summary.csv", "action_readiness_ticker_drilldown.csv", "action_readiness_source_trace.csv", "action_readiness_detail_cards.csv", "action_readiness_detail_card_panels.csv"), 40, "Ticker pages explain decision, sources, gates, horizon routes, current readiness blockers, source conflicts, and card-level PM summaries.", "Some tickers still lack mapped news/fundamental source evidence.", "P1", "Research"),
                ControlSpec("Strategy thesis board", ("institutional_strategy_thesis_board.csv", "institutional_strategy_action_playbook.csv"), 20, "Base/bull/bear/no-trade case exists before paper sizing.", "Thesis quality is rule-based and needs analyst-grade fundamental support.", "P2", "Research"),
            ),
        ),
        ModuleSpec(
            "M02",
            "Dashboard / PM workbench",
            55,
            65,
            72,
            "A PM can understand daily priorities, risk, signals, sources, and system health without reading code.",
            (
                ControlSpec("Main dashboard", ("canyon_final_v9_step86_dashboard_v3.py", "action_readiness_detail_cards.csv", "action_readiness_card_deck_summary.csv"), 1, "Single Streamlit workbench with daily run, research, risk, news, system views, and readiness detail cards.", "Needs cleaner professional workflow and less raw-table overload.", "P1", "Dashboard"),
                ControlSpec("Run audit trail", ("run_daily_all_log.csv", "run_daily_all_report.md"), 20, "Every run logs step status and failure notes.", "Does not yet have full dependency graph or auto-remediation.", "P1", "Dashboard"),
                ControlSpec("System output inventory", ("institutional_depth_module_scorecard.csv", "run_daily_all_log.csv"), 10, "Dashboard exposes which modules are active vs prototype.", "Needs persistent user-facing QA history.", "P2", "Dashboard"),
            ),
        ),
        ModuleSpec(
            "M03",
            "Multi-layer signal system",
            50,
            60,
            68,
            "Signals must be horizon-aware, regime-aware, failure-aware, and prevented from overriding risk.",
            (
                ControlSpec("Signal IC and decay", ("signal_horizon_regime_policy.csv", "signal_regime_ic_matrix.csv", "signal_failure_by_market_bucket.csv"), 50, "Signals have allowed use, horizon, decay, and failure mode policy.", "Some signals still have thin sample or no proven regime bucket.", "P1", "Signals"),
                ControlSpec("Live-vs-backtest drift", ("signal_live_vs_backtest_drift.csv", "live_ic_observation_ledger.csv"), 20, "Live observations are separated from backtest evidence.", "Real live IC history is still shallow.", "P1", "Signals"),
                ControlSpec("Signal aggregation policy", ("research_signal_weight_policy.csv", "signal_downgrade_queue.csv"), 10, "Weak signals get downweighted before promotion.", "No full Bayesian uncertainty or ensemble calibration yet.", "P2", "Signals"),
            ),
        ),
        ModuleSpec(
            "M04",
            "Risk framework",
            45,
            55,
            70,
            "Risk must dominate alpha through VaR/CVaR, drawdown, liquidity, earnings, correlation, and hard limits.",
            (
                ControlSpec("Risk budget and VaR/CVaR", ("institutional_risk_budget_summary.csv", "portfolio_var_cvar_summary.csv", "risk_desk_ticker_action_queue.csv", "risk_unlock_action_board.csv", "risk_unlock_sizing_ladder.csv", "risk_repair_scenario_summary.csv", "risk_repair_metric_impact.csv", "risk_repair_recommendation_board.csv", "risk_repair_priority_queue.csv", "action_readiness_monitor.csv", "action_readiness_next_move_queue.csv", "action_readiness_ticker_drilldown.csv", "action_readiness_manual_checklist.csv"), 20, "Single-name and portfolio risk are size-gated, have explicit unlock sizing steps, include scenario repair simulation, PM repair recommendation queue, gate-by-gate action readiness monitor, and manual checklist.", "Thresholds still need deeper historical calibration and intraday risk.", "P1", "Risk"),
                ControlSpec("Risk policy governance", ("risk_policy_approval_packet.csv", "risk_policy_decision_log.csv", "risk_policy_dry_run_monitor.csv"), 20, "Risk rule changes have evidence, review, and dry-run logs.", "No human approval workflow integration yet.", "P2", "Risk"),
                ControlSpec("Extreme and correlation protection", ("crowding_risk_matrix.csv", "signal_correlation.csv", "risk_historical_correlation_windows.csv"), 20, "Correlation and crowding are visible before sizing.", "No vendor-grade factor model or crisis correlation engine.", "P1", "Risk"),
            ),
        ),
        ModuleSpec(
            "M05",
            "Sector / Theme / Subsector cycle",
            45,
            55,
            68,
            "Sector calls must separate leader strength, late-cycle exhaustion, handoff, breadth, and theme linkage.",
            (
                ControlSpec("Subsector cycle board", ("institutional_subsector_cycle_board.csv", "subsector_ticker_cycle_map.csv"), 14, "Subsectors have phase, exhaustion, catch-up, and ticker mapping.", "Needs longer sector-specific validation history.", "P1", "Sector"),
                ControlSpec("Rotation thesis validation", ("subsector_rotation_validation.csv", "subsector_rotation_observation_history.csv"), 5, "Semis-late-cycle and software-handoff thesis are validated against evidence.", "Still a rules/proxy thesis, not an institutional analyst model.", "P1", "Sector"),
                ControlSpec("Sector route bridge", ("sector_timeframe_route.csv", "sector_timeframe_option_route.csv", "key_sector_linkage.csv"), 10, "Sector cycle is connected to timeframe and vehicle route.", "Needs stronger subsector supply-chain and factor-neutral overlays.", "P2", "Sector"),
            ),
        ),
        ModuleSpec(
            "M06",
            "News / Event / Industry read-through",
            35,
            45,
            64,
            "News should map source, timestamp, causal chain, beneficiaries, vulnerable peers, and industry read-through.",
            (
                ControlSpec("Event causal chain", ("event_causal_chain_map.csv", "event_causal_chain_edges.csv", "event_causal_validation_queue.csv"), 50, "Headlines are mapped to causal links and validation queue.", "Many causal links are still hypotheses needing price reaction proof.", "P1", "Events"),
                ControlSpec("Event reliability calibration", ("event_signal_reliability_adjusted_panel.csv", "event_signal_reliability_by_bucket.csv", "event_signal_reliability_watchlist.csv"), 50, "Event signal reliability is calibrated and weak buckets are flagged.", "Samples remain thin and source timestamp quality is uneven.", "P1", "Events"),
                ControlSpec("Read-through decision board", ("event_readthrough_decision_board.csv", "event_readthrough_target_ranking.csv", "event_readthrough_chain_ladder.csv"), 100, "Events now rank beneficiary/vulnerable targets with proof required.", "This is a router, not a verified causal model.", "P1", "Events"),
            ),
        ),
        ModuleSpec(
            "M07",
            "Options / Call / Put / Gamma",
            30,
            40,
            58,
            "Options need clear call/put permission, gamma context, Greeks risk, IV/spread no-go rules, and no weekly chase.",
            (
                ControlSpec("Option route clarity", ("option_route_clarity_board.csv", "horizon_vehicle_matrix.csv", "options_execution_route_matrix.csv", "options_trade_permission_summary.csv", "call_unlock_board.csv", "put_hedge_unlock_board.csv", "risk_unlock_option_bridge.csv", "risk_repair_option_projection.csv", "risk_repair_strategy_reopen_map.csv", "action_readiness_gate_matrix.csv", "action_readiness_transition_map.csv", "action_readiness_blocker_explainer.csv", "action_readiness_source_trace.csv"), 20, "Each ticker has short/medium/long call/put/no-option route, trigger, no-go reasons, unlock sequence, risk-repair bridge, post-repair option projection, route reopen map, gate-by-gate readiness transition map, blocker explainer, and source trace.", "Options chain depth, IV rank, and spread data still need stronger live checks.", "P1", "Options"),
                ControlSpec("Gamma and kill zone", ("options_gamma_report.md", "gamma_squeeze_candidates.csv", "option_kill_zone_risk.csv"), 10, "Gamma squeeze and kill-zone risk are separated from final action.", "Dealer positioning is still OI/volume proxy, not true dealer inventory.", "P1", "Options"),
                ControlSpec("Options Greeks book risk", ("options_greeks_book_risk.csv", "options_backtest_results.csv", "options_tca_no_go_audit.csv", "options_execution_playbook.csv", "option_unlock_blocker_attribution.csv"), 10, "Greeks, IV, gamma heat, TCA no-go checks, suspicious backtest flags, and blocker attribution constrain option routes.", "Full live options chain, real spread, and dealer inventory data are still incomplete.", "P1", "Options"),
            ),
        ),
        ModuleSpec(
            "M08",
            "Portfolio construction",
            35,
            45,
            64,
            "Portfolio must allocate by sleeve, active risk, sector/factor constraints, turnover, and optimizer explainability.",
            (
                ControlSpec("Optimizer bridge", ("institutional_optimizer_bridge.csv", "institutional_optimizer_why_not_more.csv", "institutional_optimizer_constraint_ladder.csv"), 20, "Optimizer explains target, constraints, and why size cannot be larger.", "Still risk-gated research optimizer, not robust production optimization.", "P1", "Portfolio"),
                ControlSpec("Sleeve and target weights", ("institutional_sleeve_allocations.csv", "institutional_target_weights.csv", "institutional_portfolio_construction_plan.csv"), 20, "Weights are mapped into sleeves and research target plan.", "Expected-return uncertainty and robust optimization are still shallow.", "P2", "Portfolio"),
                ControlSpec("Active risk budget", ("institutional_optimizer_active_risk_budget.csv", "strategy_risk_budget_bridge.csv"), 8, "Active risk buckets are visible before sizing.", "No full Barra/Axioma style active risk model.", "P1", "Portfolio"),
            ),
        ),
        ModuleSpec(
            "M09",
            "Backtest credibility",
            25,
            35,
            60,
            "Backtests need point-in-time truth, walk-forward validation, no look-ahead, survivorship control, and execution realism.",
            (
                ControlSpec("Credibility scorecard", ("backtest_credibility_scorecard.csv", "backtest_credibility_blockers.csv", "backtest_credibility_evidence.csv"), 7, "Backtest is labeled by credibility controls before use.", "Still not allowed as sizing evidence until PIT and execution hard gates close.", "P1", "Backtest"),
                ControlSpec("Bias guard", ("backtest_bias_guard.csv", "institutional_backtest_integrity_audit.csv"), 8, "Look-ahead and survivorship risks are explicitly tracked.", "Historical membership, delisted names, and exact model-seen timestamps still need vendor proof.", "P1", "Backtest"),
                ControlSpec("Walk-forward and execution reality", ("backtest_walk_forward_proxy.csv", "backtest_execution_reality_check.csv"), 4, "Walk-forward and execution realism are at least separated from headline performance.", "Proxy walk-forward is not the same as frozen signal history.", "P1", "Backtest"),
            ),
        ),
        ModuleSpec(
            "M10",
            "ML / signal IC / decay",
            25,
            35,
            58,
            "ML needs feature lineage, true out-of-sample IC, decay curves, failure regimes, and live observation feedback.",
            (
                ControlSpec("ML scores and performance", ("ml_signal_scores.csv", "ml_backtest_perf.csv"), 20, "ML outputs exist and are separated from rule signals.", "ML still fails institutional proof without stable live IC and feature lineage.", "P1", "ML"),
                ControlSpec("Signal decay deep dive", ("signal_decay_analysis.csv", "signal_failure_deep_dive.csv", "signal_horizon_regime_policy.csv"), 12, "Decay and failure modes constrain signal use.", "Sample depth and regime buckets remain uneven.", "P1", "ML"),
                ControlSpec("Live IC ledger", ("live_ic_observation_ledger.csv", "signal_live_vs_backtest_drift.csv"), 10, "Live learning is tracked apart from backtest.", "Live history is not long enough for automatic promotion.", "P1", "ML"),
            ),
        ),
        ModuleSpec(
            "M11",
            "Execution / TCA",
            20,
            30,
            55,
            "Execution model needs spread, impact, participation, auction risk, failed-fill assumptions, and capacity limits.",
            (
                ControlSpec("Execution cost model", ("execution_cost_model.csv", "execution_cost_stress_scenarios.csv", "execution_cost_constraint_audit.csv", "options_tca_no_go_audit.csv"), 20, "TCA estimates, spread, fill risk, and option no-go checks constrain paper routes.", "No real fills, broker venue data, or intraday order book depth.", "P1", "Execution"),
                ControlSpec("Trade plan and slicing", ("execution_trade_plan.csv", "execution_slicing_schedule.csv", "options_execution_playbook.csv"), 20, "Order slicing and option route playbooks are visible for research.", "Still hypothetical and no live order path is enabled.", "P2", "Execution"),
                ControlSpec("Capacity limits", ("institutional_execution_capacity_limits.csv", "institutional_tca_cost_estimates.csv"), 20, "Liquidity and capacity limits are estimated.", "Needs real spread/impact calibration.", "P1", "Execution"),
            ),
        ),
        ModuleSpec(
            "M12",
            "Point-in-time data truth",
            20,
            30,
            55,
            "Every historical decision needs what was known, when it was known, vendor/source ID, and model-read timestamp.",
            (
                ControlSpec("PIT truth readiness", ("pit_truth_scorecard.csv", "pit_backtest_readiness_gates.csv", "pit_source_risk_register.csv"), 4, "PIT readiness gates separate local/proxy data from true history.", "Vendor-grade point-in-time data is still mostly missing.", "P1", "Data Truth"),
                ControlSpec("Event time truth", ("event_time_truth_ledger.csv", "event_time_quality_audit.csv", "event_time_repair_queue.csv"), 20, "Event timestamps are audited and repair queue exists.", "Many event timestamps still need first-seen/model-read proof.", "P1", "Data Truth"),
                ControlSpec("PIT seed store", ("pit_store_build_audit.csv", "pit_fundamentals.csv", "pit_store_state.json"), 5, "Local PIT seed store exists for controlled future improvement.", "Seed store is not a substitute for vendor-grade historical data.", "P2", "Data Truth"),
            ),
        ),
        ModuleSpec(
            "M13",
            "Live monitoring / alerts",
            30,
            40,
            58,
            "Monitoring should detect price breaks, volume spikes, vol shifts, spread widening, correlation break, news shock, earnings surprise, and risk breach.",
            (
                ControlSpec("Desk monitor events", ("desk_monitor_events.csv", "desk_monitor_ticker_state.csv", "desk_monitor_summary.json"), 20, "Core live-monitor events are produced and tied to source layer.", "No persistent alert acknowledgement or escalation workflow.", "P1", "Monitoring"),
                ControlSpec("Realtime alert matrix", ("institutional_realtime_alert_matrix.csv", "daily_alerts.json", "daily_alerts.md"), 8, "Alert types are listed with source readiness and event count.", "No live notification routing beyond local dashboard/report.", "P2", "Monitoring"),
                ControlSpec("Risk breach monitor", ("risk_desk_breach_table.csv", "risk_desk_component_health.csv"), 8, "Risk breaches and component health are visible.", "Needs intraday refresh and automated QA for stale sources.", "P1", "Monitoring"),
            ),
        ),
    )


def module_score(module: ModuleSpec, controls: pd.DataFrame) -> dict[str, Any]:
    sub = controls[controls["module_id"] == module.module_id].copy()
    baseline = (module.user_baseline_low + module.user_baseline_high) / 2.0
    if sub.empty:
        evidence_score = 0.0
        coverage = 0.0
        p1_weak = 0
        strongest = "No controls found."
        weakest = "No controls found."
    else:
        weights = sub["priority"].map({"P1": 1.25, "P2": 1.0, "P3": 0.8}).fillna(1.0).astype(float)
        evidence_score = float(np.average(sub["control_score_pct"].astype(float), weights=weights))
        coverage = float((sub["control_score_pct"].astype(float) >= 50).mean() * 100.0)
        p1_weak = int(((sub["priority"] == "P1") & (sub["control_score_pct"].astype(float) < 55)).sum())
        strongest_row = sub.sort_values("control_score_pct", ascending=False).iloc[0]
        weakest_row = sub.sort_values("control_score_pct", ascending=True).iloc[0]
        strongest = f"{strongest_row['control']}: {strongest_row['evidence']}"
        weakest = f"{weakest_row['control']}: {weakest_row['institution_gap']}"

    # Conservative readiness: user baseline remains the anchor. Evidence can
    # improve it, but hard caps prevent proxy data from looking institutional.
    raw = 0.58 * baseline + 0.42 * evidence_score
    depth_bonus = min(4.0, max(0.0, coverage - 45.0) / 18.0)
    penalty = min(8.0, p1_weak * 2.0)
    updated = min(module.readiness_cap_pct, max(baseline, raw + depth_bonus - penalty))
    gap = max(0.0, 100.0 - updated)

    if updated < 45:
        next_priority = "P1_NOW"
    elif p1_weak > 0:
        next_priority = "P1_REPAIR"
    elif updated < 62:
        next_priority = "P2_DEEPEN"
    else:
        next_priority = "P2_POLISH"

    next_action = "Run and repair the weakest P1 control before adding new model complexity."
    if not sub.empty:
        weak_p1 = sub[(sub["priority"] == "P1")].sort_values("control_score_pct").head(1)
        if not weak_p1.empty:
            r = weak_p1.iloc[0]
            next_action = f"{r['target_state']} Gap: {r['institution_gap']}"

    return {
        "module_id": module.module_id,
        "module": module.module,
        "user_baseline_pct": round(baseline, 1),
        "evidence_score_pct": round(evidence_score, 1),
        "updated_readiness_pct": round(updated, 1),
        "gap_to_top_quant_pct": round(gap, 1),
        "readiness_band": readiness_band(updated),
        "module_status": status_from_score(updated),
        "control_coverage_pct": round(coverage, 1),
        "weak_p1_controls": p1_weak,
        "readiness_cap_pct": module.readiness_cap_pct,
        "top_institution_requires": module.top_institution_requires,
        "strongest_evidence": strongest[:900],
        "weakest_gap": weakest[:900],
        "next_priority": next_priority,
        "next_action": next_action[:900],
        "prototype_truth": "Research prototype only. No broker connection. No live orders. Missing data never upgrades a decision.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def build_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    module_specs = specs()
    control_rows: list[dict[str, Any]] = []
    for module in module_specs:
        for control in module.controls:
            control_rows.append(assess_control(module.module_id, module.module, control))
    control_map = pd.DataFrame(control_rows)

    scorecard = pd.DataFrame([module_score(m, control_map) for m in module_specs])
    scorecard = scorecard.sort_values(["updated_readiness_pct", "module_id"], ascending=[True, True]).reset_index(drop=True)

    queue_rows: list[dict[str, Any]] = []
    for _, row in control_map.sort_values(["priority", "control_score_pct"]).iterrows():
        score = safe_float(row["control_score_pct"], 0.0)
        if row["priority"] == "P1" or score < 55:
            queue_rows.append({
                "priority_rank": len(queue_rows) + 1,
                "priority": "P1_NOW" if score < 45 else ("P1_REPAIR" if row["priority"] == "P1" else "P2_DEEPEN"),
                "module": row["module"],
                "control": row["control"],
                "control_score_pct": row["control_score_pct"],
                "current_status": row["control_status"],
                "what_to_fix": row["institution_gap"],
                "target_state": row["target_state"],
                "evidence_files": row["evidence_files"],
                "missing_files": row["missing_files"],
                "why_it_matters": "This control is required before Canyon can move closer to a top-tier institutional workflow.",
                "research_only": True,
            })
    queue = pd.DataFrame(queue_rows)

    gap_rows = []
    for module in module_specs:
        mrow = scorecard[scorecard["module_id"] == module.module_id].iloc[0]
        sub = control_map[control_map["module_id"] == module.module_id].sort_values("control_score_pct")
        weak_controls = "; ".join(sub.head(2)["control"].tolist()) if not sub.empty else "No controls"
        missing = "; ".join([x for x in sub["missing_files"].astype(str).tolist() if x][:2])
        gap_rows.append({
            "module_id": module.module_id,
            "module": module.module,
            "top_institution_requires": module.top_institution_requires,
            "canyon_now_has": mrow["strongest_evidence"],
            "largest_remaining_gap": mrow["weakest_gap"],
            "weak_controls": weak_controls,
            "missing_or_partial_files": missing,
            "updated_readiness_pct": mrow["updated_readiness_pct"],
            "gap_to_top_quant_pct": mrow["gap_to_top_quant_pct"],
            "next_action": mrow["next_action"],
            "research_only": True,
        })
    gap = pd.DataFrame(gap_rows)

    readiness = float(scorecard["updated_readiness_pct"].mean()) if not scorecard.empty else 0.0
    weakest_modules = scorecard.sort_values("updated_readiness_pct").head(4)["module"].tolist()
    p1_items = int((queue["priority"].astype(str).str.startswith("P1")).sum()) if not queue.empty else 0
    strong_controls = int((control_map["control_score_pct"].astype(float) >= 62).sum()) if not control_map.empty else 0
    weak_controls = int((control_map["control_score_pct"].astype(float) < 50).sum()) if not control_map.empty else 0
    state = {
        "date": today_str(),
        "overall_readiness_pct": round(readiness, 1),
        "overall_gap_to_top_quant_pct": round(100.0 - readiness, 1),
        "modules_scored": int(len(scorecard)),
        "controls_scored": int(len(control_map)),
        "strong_controls": strong_controls,
        "weak_controls": weak_controls,
        "p1_queue_items": p1_items,
        "weakest_modules": weakest_modules,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "This is a conservative local-output audit. It improves module visibility and upgrade discipline, but it is not vendor-grade institutional production infrastructure.",
        "outputs": {
            "scorecard": OUT_SCORECARD.name,
            "control_map": OUT_CONTROL_MAP.name,
            "queue": OUT_QUEUE.name,
            "gap_matrix": OUT_GAP.name,
            "report": OUT_REPORT.name,
        },
    }
    return scorecard, control_map, queue, gap, state


def write_outputs(scorecard: pd.DataFrame, control_map: pd.DataFrame, queue: pd.DataFrame, gap: pd.DataFrame, state: dict[str, Any]) -> None:
    scorecard.to_csv(OUT_SCORECARD, index=False)
    control_map.to_csv(OUT_CONTROL_MAP, index=False)
    queue.to_csv(OUT_QUEUE, index=False)
    gap.to_csv(OUT_GAP, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "## One-page truth",
        (
            f"Overall readiness is **{state['overall_readiness_pct']}%**. "
            f"The remaining gap to a top-tier quant institution is about **{state['overall_gap_to_top_quant_pct']}%**. "
            "This score is intentionally conservative because local/yfinance/proxy data, proxy backtests, "
            "manual execution assumptions, and incomplete options Greeks cannot be treated as institutional proof."
        ),
        "",
        "## Module scorecard",
        df_to_markdown(scorecard[[
            "module",
            "user_baseline_pct",
            "updated_readiness_pct",
            "gap_to_top_quant_pct",
            "module_status",
            "next_priority",
            "next_action",
        ]], max_rows=20),
        "",
        "## Priority upgrade queue",
        df_to_markdown(queue.head(20), max_rows=20),
        "",
        "## Control map",
        df_to_markdown(control_map[[
            "module",
            "control",
            "control_score_pct",
            "control_status",
            "priority",
            "evidence",
            "weakest_point",
        ]], max_rows=50),
        "",
        "## Non-negotiable limits",
        "- Research-only; no broker connection; no live orders.",
        "- Missing files and stale evidence can only reduce confidence, not upgrade it.",
        "- Options/gamma signals cannot override L8 risk, execution cost, event proof, or point-in-time data truth.",
        "- Backtest and ML scores cannot become sizing evidence until PIT, survivorship, and execution controls are repaired.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 172 - Institutional Depth Upgrade Engine", sections)


def main() -> None:
    scorecard, control_map, queue, gap, state = build_all()
    write_outputs(scorecard, control_map, queue, gap, state)

    print("Step 172 complete.")
    print(f"Overall readiness: {state['overall_readiness_pct']}%")
    print(f"Gap to top quant: {state['overall_gap_to_top_quant_pct']}%")
    print(f"Modules scored: {state['modules_scored']}; controls scored: {state['controls_scored']}")
    print(f"P1 queue items: {state['p1_queue_items']}")
    print(f"Wrote: {OUT_SCORECARD.name}, {OUT_CONTROL_MAP.name}, {OUT_QUEUE.name}, {OUT_GAP.name}, {OUT_STATE.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
