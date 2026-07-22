#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 125: Institutional Execution Playbook
======================================================

Research-only execution feasibility layer.

This module does not send orders, does not connect to a broker, and does not
produce live trading instructions. It turns the current target-weight change
into a research execution plan with participation, spread/impact, auction risk,
failed-fill assumptions, and slicing assumptions.

Inputs:
  institutional_target_weights.csv
  portfolio_turnover_budget.csv
  institutional_tca_cost_estimates.csv
  institutional_execution_capacity_limits.csv
  desk_monitor_ticker_state.csv
  event_research_gate.csv

Outputs:
  execution_trade_plan.csv
  execution_slicing_schedule.csv
  execution_assumption_audit.csv
  execution_playbook_state.json
  execution_playbook_report.md
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PORTFOLIO_DOLLARS = 100_000.0

OUT_PLAN = ROOT / "execution_trade_plan.csv"
OUT_SLICES = ROOT / "execution_slicing_schedule.csv"
OUT_AUDIT = ROOT / "execution_assumption_audit.csv"
OUT_STATE = ROOT / "execution_playbook_state.json"
OUT_REPORT = ROOT / "execution_playbook_report.md"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 10:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean_ticker(value) -> str:
    return str(value).strip().upper()


def as_float(value, default=np.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def first_value(row: pd.Series, *names: str, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return row.get(name)
    return default


def classify_execution(
    direction: str,
    final_risk_action: str,
    event_gate: str,
    target_status: str,
    source_execution_status: str,
    participation_pct: float,
    total_cost_bps: float,
    spread_status: str,
    adv: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    status = "CLEAR"

    if not np.isfinite(adv) or adv <= 0:
        return "DATA_GAP", ["missing average dollar volume"]

    if direction == "UP" and final_risk_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        return "BLOCK_NEW", ["risk gate allows reduction only"]
    if direction == "UP" and event_gate in {"BLOCK_NEW", "BLOCKED"}:
        return "BLOCK_NEW", ["event gate blocks new exposure"]

    if final_risk_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        status = "REDUCE_ONLY"
        reasons.append("final risk gate")
    elif final_risk_action == "SIZE_DOWN" or target_status == "SIZE_DOWN":
        status = "SIZE_DOWN"
        reasons.append("risk target size-down")

    if event_gate in {"REVIEW", "SIZE_DOWN"}:
        status = "SIZE_DOWN" if status == "CLEAR" else status
        reasons.append(f"event gate {event_gate.lower()}")

    if source_execution_status in {"SIZE_DOWN", "BLOCK_NEW"}:
        status = "SIZE_DOWN" if source_execution_status == "SIZE_DOWN" else "BLOCK_NEW"
        reasons.append(f"source TCA status {source_execution_status.lower()}")

    if np.isfinite(total_cost_bps):
        if total_cost_bps >= 80:
            status = "BLOCK_NEW" if direction == "UP" else "SIZE_DOWN"
            reasons.append("very high expected transaction cost")
        elif total_cost_bps >= 45 and status == "CLEAR":
            status = "SIZE_DOWN"
            reasons.append("high expected transaction cost")
        elif total_cost_bps >= 25 and status == "CLEAR":
            status = "REVIEW"
            reasons.append("moderate expected transaction cost")

    if np.isfinite(participation_pct):
        if participation_pct >= 5.0:
            status = "BLOCK_NEW" if direction == "UP" else "SIZE_DOWN"
            reasons.append("participation exceeds daily cap")
        elif participation_pct >= 2.0 and status in {"CLEAR", "REVIEW"}:
            status = "SIZE_DOWN"
            reasons.append("large participation rate")
        elif participation_pct >= 1.0 and status == "CLEAR":
            status = "REVIEW"
            reasons.append("participation needs care")

    if spread_status in {"DATA_GAP", "WIDE", "WARNING"} and status == "CLEAR":
        status = "REVIEW"
        reasons.append("spread requires manual check")

    if not reasons:
        reasons.append("within research execution limits")
    return status, reasons


def expected_fill_rate(status: str, spread_status: str) -> float:
    base = {
        "CLEAR": 0.985,
        "REVIEW": 0.94,
        "SIZE_DOWN": 0.86,
        "REDUCE_ONLY": 0.90,
        "BLOCK_NEW": 0.0,
        "DATA_GAP": 0.0,
    }.get(status, 0.85)
    if spread_status in {"DATA_GAP", "WIDE", "WARNING"} and base > 0:
        base -= 0.05
    return max(0.0, min(base, 1.0))


def build_trade_plan() -> pd.DataFrame:
    target = read_csv_safe(ROOT / "institutional_target_weights.csv")
    turnover = read_csv_safe(ROOT / "portfolio_turnover_budget.csv")
    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    capacity = read_csv_safe(ROOT / "institutional_execution_capacity_limits.csv")
    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    event_gate = read_csv_safe(ROOT / "event_research_gate.csv")

    if target.empty or "ticker" not in target.columns:
        return pd.DataFrame()

    base = target.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)

    merge_sets = [
        (turnover, ["ticker", "turnover_pct", "direction"], ""),
        (tca, [
            "ticker", "avg_20d_dollar_volume", "participation_rate_pct",
            "spread_bps_est", "auction_risk_bps", "failed_fill_buffer_bps",
            "total_tca_cost_bps", "execution_status",
        ], "_tca"),
        (capacity, ["ticker", "max_daily_participation_pct", "capacity_status"], "_capacity"),
        (monitor, ["ticker", "spread_status", "volume_spike_state", "volatility_regime_state", "price_break_state", "max_monitor_severity"], "_monitor"),
        (event_gate, ["ticker", "event_risk_score", "event_research_score"], "_event"),
    ]
    for df, cols, suffix in merge_sets:
        if not df.empty and "ticker" in df.columns:
            tmp = df.copy()
            tmp["ticker"] = tmp["ticker"].apply(clean_ticker)
            keep = [c for c in cols if c in tmp.columns]
            base = base.merge(tmp[keep], on="ticker", how="left", suffixes=("", suffix))

    rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        direction = str(row.get("direction", "FLAT")).upper()
        turnover_pct = abs(as_float(row.get("turnover_pct"), 0.0))
        target_weight_pct = as_float(row.get("target_weight_pct"), 0.0)
        current_weight_pct = as_float(row.get("current_weight_pct"), 0.0)
        trade_notional = turnover_pct / 100.0 * MODEL_PORTFOLIO_DOLLARS
        adv = as_float(first_value(row, "avg_20d_dollar_volume", "avg_20d_dollar_volume_tca"))
        participation_pct = trade_notional / adv * 100.0 if np.isfinite(adv) and adv > 0 else np.nan
        spread_bps = as_float(row.get("spread_bps_est"), 25.0)
        auction_bps = as_float(row.get("auction_risk_bps"), 10.0)
        failed_fill_bps = as_float(row.get("failed_fill_buffer_bps"), 5.0)
        total_tca_bps = as_float(row.get("total_tca_cost_bps"), spread_bps / 2.0 + auction_bps + failed_fill_bps)
        final_risk_action = str(row.get("final_risk_action", "REVIEW")).upper()
        event_status = str(row.get("event_gate", "REVIEW")).upper()
        target_status = str(row.get("target_status", "REVIEW")).upper()
        source_exec = str(row.get("execution_status", "REVIEW")).upper()
        spread_status = str(row.get("spread_status", "REVIEW")).upper()

        status, reasons = classify_execution(
            direction=direction,
            final_risk_action=final_risk_action,
            event_gate=event_status,
            target_status=target_status,
            source_execution_status=source_exec,
            participation_pct=participation_pct,
            total_cost_bps=total_tca_bps,
            spread_status=spread_status,
            adv=adv,
        )

        max_daily_participation_pct = min(as_float(row.get("max_daily_participation_pct"), 5.0), 5.0)
        if status in {"SIZE_DOWN", "REVIEW"}:
            max_daily_participation_pct = min(max_daily_participation_pct, 2.0)
        if status == "BLOCK_NEW":
            max_daily_participation_pct = 0.0
        max_daily_notional = adv * max_daily_participation_pct / 100.0 if np.isfinite(adv) else 0.0
        allowed_trade_notional = min(trade_notional, max_daily_notional) if max_daily_notional > 0 else 0.0
        if status == "BLOCK_NEW" and direction == "UP":
            allowed_trade_notional = 0.0
        days_to_complete = math.ceil(trade_notional / max_daily_notional) if max_daily_notional > 0 and trade_notional > 0 else 0

        if auction_bps >= 12 or spread_status in {"DATA_GAP", "WIDE", "WARNING"}:
            auction_policy = "Avoid open/close auction; use mid-day VWAP-style research assumption."
        else:
            auction_policy = "Avoid market orders; use patient VWAP-style research assumption."

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "sleeve": row.get("sleeve", "Unknown"),
            "direction": direction,
            "current_weight_pct": current_weight_pct,
            "target_weight_pct": target_weight_pct,
            "turnover_pct": turnover_pct,
            "trade_notional_dollars": round(trade_notional, 2),
            "avg_20d_dollar_volume": adv,
            "participation_rate_pct": participation_pct,
            "spread_bps_est": spread_bps,
            "auction_risk_bps": auction_bps,
            "failed_fill_buffer_bps": failed_fill_bps,
            "total_tca_cost_bps": total_tca_bps,
            "expected_cost_dollars": trade_notional * total_tca_bps / 10000.0 if np.isfinite(total_tca_bps) else np.nan,
            "max_daily_participation_pct": max_daily_participation_pct,
            "allowed_trade_notional_dollars": round(allowed_trade_notional, 2),
            "estimated_days_to_complete": days_to_complete,
            "expected_fill_rate_pct": expected_fill_rate(status, spread_status) * 100.0,
            "execution_playbook_status": status,
            "final_risk_action": final_risk_action,
            "event_gate": event_status,
            "source_execution_status": source_exec,
            "spread_status": spread_status,
            "price_break_state": row.get("price_break_state", ""),
            "volume_spike_state": row.get("volume_spike_state", ""),
            "volatility_regime_state": row.get("volatility_regime_state", ""),
            "max_monitor_severity": row.get("max_monitor_severity", ""),
            "auction_policy": auction_policy,
            "reason": "; ".join(reasons),
            "source_file": "institutional_target_weights.csv / portfolio_turnover_budget.csv / institutional_tca_cost_estimates.csv / desk_monitor_ticker_state.csv",
            "research_only": True,
        })
    return pd.DataFrame(rows)


def build_slicing_schedule(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()
    buckets = ["mid_morning", "late_morning", "midday", "early_afternoon", "late_afternoon"]
    rows = []
    for _, row in plan.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        trade_notional = as_float(row.get("allowed_trade_notional_dollars"), 0.0)
        if trade_notional <= 0:
            rows.append({
                "ticker": ticker,
                "slice_id": 0,
                "time_bucket": "none",
                "slice_notional_dollars": 0.0,
                "slice_participation_pct": 0.0,
                "instruction": "No new trade allowed by research execution playbook.",
                "source_file": "execution_trade_plan.csv",
            })
            continue
        participation = max(as_float(row.get("participation_rate_pct"), 0.0), 0.0)
        slices = int(min(8, max(1, math.ceil(participation / 0.35))))
        slice_notional = trade_notional / slices
        adv = as_float(row.get("avg_20d_dollar_volume"), np.nan)
        slice_participation = slice_notional / adv * 100.0 if np.isfinite(adv) and adv > 0 else np.nan
        for i in range(slices):
            rows.append({
                "ticker": ticker,
                "slice_id": i + 1,
                "time_bucket": buckets[i % len(buckets)],
                "slice_notional_dollars": round(slice_notional, 2),
                "slice_participation_pct": slice_participation,
                "instruction": "Research slicing assumption only; no broker connection, no live order.",
                "source_file": "execution_trade_plan.csv",
            })
    return pd.DataFrame(rows)


def build_assumption_audit(plan: pd.DataFrame) -> pd.DataFrame:
    blocked = int(plan["execution_playbook_status"].astype(str).str.upper().isin(["BLOCK_NEW", "DATA_GAP"]).sum()) if not plan.empty else 0
    review = int(plan["execution_playbook_status"].astype(str).str.upper().isin(["REVIEW", "SIZE_DOWN", "REDUCE_ONLY"]).sum()) if not plan.empty else 0
    data_gap = int(plan["spread_status"].astype(str).str.upper().eq("DATA_GAP").sum()) if not plan.empty and "spread_status" in plan.columns else 0
    return pd.DataFrame([
        {
            "control": "No live execution",
            "status": "PASS",
            "evidence": "The playbook writes CSV/JSON/MD only and has no broker dependency.",
            "required_next_action": "Keep broker APIs out of the research app.",
            "source_file": "canyon_final_v9_step125_execution_playbook.py",
        },
        {
            "control": "Participation limit",
            "status": "PASS" if blocked == 0 else "REVIEW",
            "evidence": "Each ticker receives max daily participation and estimated days-to-complete.",
            "required_next_action": "Calibrate participation caps with real fill history before relying on it.",
            "source_file": "execution_trade_plan.csv",
        },
        {
            "control": "Spread and auction risk",
            "status": "REVIEW" if data_gap else "PASS",
            "evidence": f"{data_gap} ticker rows have spread data gaps or need manual spread checks.",
            "required_next_action": "Add live bid/ask snapshots or paid intraday quote history.",
            "source_file": "desk_monitor_ticker_state.csv",
        },
        {
            "control": "Failed-fill assumption",
            "status": "REVIEW",
            "evidence": "Uses a conservative buffer but no real historical order/fill log exists.",
            "required_next_action": "Once paper execution logs exist, compare expected vs actual fill outcomes.",
            "source_file": "institutional_tca_cost_estimates.csv",
        },
        {
            "control": "Order slicing",
            "status": "REVIEW" if review else "PASS",
            "evidence": "Slicing schedule is deterministic and research-only.",
            "required_next_action": "Later add auction/open/close exclusions and market-condition-aware scheduling.",
            "source_file": "execution_slicing_schedule.csv",
        },
    ])


def build_state(plan: pd.DataFrame, audit: pd.DataFrame) -> dict:
    if plan.empty:
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "overall_status": "NO_DATA",
            "execution_readiness_score": 0.0,
            "research_only": True,
            "no_broker_connection": True,
            "truth": "No execution playbook could be built because target weights were missing.",
        }
    status = plan["execution_playbook_status"].astype(str).str.upper()
    blocked = int(status.isin(["BLOCK_NEW", "DATA_GAP"]).sum())
    review = int(status.isin(["REVIEW", "SIZE_DOWN", "REDUCE_ONLY"]).sum())
    clear = int((status == "CLEAR").sum())
    weighted_cost = np.average(
        pd.to_numeric(plan["total_tca_cost_bps"], errors="coerce").fillna(0.0),
        weights=np.maximum(pd.to_numeric(plan["trade_notional_dollars"], errors="coerce").fillna(0.0), 1.0),
    )
    avg_fill = float(pd.to_numeric(plan["expected_fill_rate_pct"], errors="coerce").fillna(0.0).mean())
    score = 100.0 - blocked * 12.0 - review * 3.0 - max(0.0, weighted_cost - 15.0) * 0.4
    score = max(0.0, min(100.0, score))
    if blocked:
        overall = "BLOCKER"
    elif score < 60:
        overall = "WEAK"
    elif score < 75:
        overall = "REVIEW"
    else:
        overall = "PASS"
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "overall_status": overall,
        "execution_readiness_score": round(score, 1),
        "clear_trades": clear,
        "review_or_size_down_trades": review,
        "blocked_or_data_gap_trades": blocked,
        "weighted_expected_cost_bps": round(float(weighted_cost), 2),
        "average_expected_fill_rate_pct": round(avg_fill, 1),
        "trade_plan_rows": int(len(plan)),
        "assumption_controls": int(len(audit)),
        "research_only": True,
        "no_broker_connection": True,
        "truth": "Research execution playbook only. It is not an order ticket and cannot send trades.",
    }


def write_report(plan: pd.DataFrame, audit: pd.DataFrame, state: dict) -> None:
    lines = [
        "# Canyon v9 Step 125 — Execution Playbook",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Research-only execution feasibility layer. No broker connection. No live orders.",
        "",
        f"- Overall status: {state.get('overall_status')}",
        f"- Execution readiness score: {state.get('execution_readiness_score')}",
        f"- Weighted expected cost: {state.get('weighted_expected_cost_bps')} bps",
        f"- Average expected fill rate: {state.get('average_expected_fill_rate_pct')}%",
        f"- Blocked/data-gap trades: {state.get('blocked_or_data_gap_trades')}",
        f"- Review/size-down trades: {state.get('review_or_size_down_trades')}",
        "",
        "## Output files",
        "",
        "- `execution_trade_plan.csv`",
        "- `execution_slicing_schedule.csv`",
        "- `execution_assumption_audit.csv`",
        "- `execution_playbook_state.json`",
        "",
        "## Product truth",
        "",
        "This is an execution research model. It uses local/yfinance-derived liquidity and static TCA assumptions unless better quote/fill data is added.",
    ]
    if not plan.empty:
        counts = plan["execution_playbook_status"].astype(str).value_counts().to_dict()
        lines.extend(["", "## Status counts", ""])
        for key, val in counts.items():
            lines.append(f"- {key}: {val}")
    if not audit.empty:
        lines.extend(["", "## Assumption audit", ""])
        for _, row in audit.iterrows():
            lines.append(f"- {row.get('control')}: {row.get('status')} — {row.get('required_next_action')}")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plan = build_trade_plan()
    slices = build_slicing_schedule(plan)
    audit = build_assumption_audit(plan)
    state = build_state(plan, audit)

    plan.to_csv(OUT_PLAN, index=False)
    slices.to_csv(OUT_SLICES, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    write_json(OUT_STATE, state)
    write_report(plan, audit, state)

    print(f"[step125] wrote {OUT_PLAN.name}: {len(plan)} rows")
    print(f"[step125] status={state.get('overall_status')} score={state.get('execution_readiness_score')}")


if __name__ == "__main__":
    main()
