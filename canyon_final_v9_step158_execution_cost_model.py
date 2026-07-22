#!/usr/bin/env python3
"""
Canyon v9 Step 158 - Execution Cost Stress Model.

Research-only. No broker connection. No live orders.

This step upgrades execution from a single cost estimate into scenario-aware
TCA: base cost, wide-spread cost, liquidity-shock cost, auction-risk cost, and
failed-fill buffer. It uses the Step157 risk-gated optimizer bridge, not raw
alpha weights.

Outputs:
  execution_cost_model.csv
  execution_cost_stress_scenarios.csv
  execution_cost_constraint_audit.csv
  execution_cost_model_state.json
  execution_cost_model_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_MODEL = ROOT / "execution_cost_model.csv"
OUT_SCENARIOS = ROOT / "execution_cost_stress_scenarios.csv"
OUT_AUDIT = ROOT / "execution_cost_constraint_audit.csv"
OUT_STATE = ROOT / "execution_cost_model_state.json"
OUT_REPORT = ROOT / "execution_cost_model_report.md"

PARTICIPATION_REVIEW = 1.0
PARTICIPATION_SIZE_DOWN = 2.0
PARTICIPATION_BLOCK = 5.0
BASE_COST_REVIEW = 25.0
BASE_COST_SIZE_DOWN = 45.0
BASE_COST_BLOCK = 80.0
STRESS_COST_SIZE_DOWN = 90.0
STRESS_COST_BLOCK = 150.0


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def pct_to_weight(value: Any) -> float:
    x = safe_float(value, 0.0)
    if abs(x) > 1.5:
        x = x / 100.0
    return max(0.0, x)


def spread_from_liquidity(label: str) -> float:
    label = str(label).upper()
    return {
        "HIGH": 2.0,
        "GOOD": 5.0,
        "FAIR": 12.0,
        "THIN": 25.0,
        "LOW": 50.0,
        "MISSING": 25.0,
    }.get(label, 25.0)


def severity_rank(label: str) -> int:
    return {"OK": 0, "INFO": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3, "DATA_GAP": 2}.get(str(label).upper(), 1)


def cost_status(base_cost: float, stress_cost: float, participation: float, data_gap: bool, severity: str) -> str:
    if data_gap:
        return "DATA_GAP"
    if participation >= PARTICIPATION_BLOCK or base_cost >= BASE_COST_BLOCK or stress_cost >= STRESS_COST_BLOCK:
        return "BLOCK_NEW"
    if participation >= PARTICIPATION_SIZE_DOWN or base_cost >= BASE_COST_SIZE_DOWN or stress_cost >= STRESS_COST_SIZE_DOWN or severity_rank(severity) >= 3:
        return "SIZE_DOWN"
    if participation >= PARTICIPATION_REVIEW or base_cost >= BASE_COST_REVIEW or severity_rank(severity) >= 2:
        return "REVIEW"
    return "CLEAR"


def build_base() -> pd.DataFrame:
    bridge = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    if bridge.empty:
        bridge = read_csv_safe(ROOT / "institutional_target_weights.csv")
    if bridge.empty or "ticker" not in bridge.columns:
        return pd.DataFrame()
    base = bridge.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)

    if "final_optimizer_weight" in base.columns:
        base["target_weight"] = pd.to_numeric(base["final_optimizer_weight"], errors="coerce").fillna(0.0)
    elif "target_weight" in base.columns:
        base["target_weight"] = pd.to_numeric(base["target_weight"], errors="coerce").fillna(0.0)
    else:
        base["target_weight"] = base.get("target_weight_pct", 0).apply(pct_to_weight) if "target_weight_pct" in base.columns else 0.0

    if "current_weight_pct" in base.columns:
        base["current_weight"] = base["current_weight_pct"].apply(pct_to_weight)
    elif "current_weight" not in base.columns:
        base["current_weight"] = 0.0
    else:
        base["current_weight"] = pd.to_numeric(base["current_weight"], errors="coerce").fillna(0.0)

    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    if not tca.empty and "ticker" in tca.columns:
        tca = tca.copy()
        tca["ticker"] = tca["ticker"].apply(clean_ticker)
        keep = [c for c in [
            "ticker", "avg_20d_dollar_volume", "participation_rate_pct",
            "spread_bps_est", "half_spread_cost_bps", "market_impact_bps",
            "auction_risk_bps", "failed_fill_buffer_bps", "total_tca_cost_bps",
            "execution_status",
        ] if c in tca.columns]
        base = base.merge(tca[keep], on="ticker", how="left", suffixes=("", "_tca"))

    liq = read_csv_safe(ROOT / "intraday_liquidity_proxy.csv")
    if not liq.empty and "ticker" in liq.columns:
        liq = liq.copy()
        liq["ticker"] = liq["ticker"].apply(clean_ticker)
        keep = [c for c in ["ticker", "avg_20d_dollar_volume", "liquidity_label", "median_20d_volume"] if c in liq.columns]
        base = base.merge(liq[keep], on="ticker", how="left", suffixes=("", "_liq"))

    monitor = read_csv_safe(ROOT / "desk_monitor_ticker_state.csv")
    if not monitor.empty and "ticker" in monitor.columns:
        monitor = monitor.copy()
        monitor["ticker"] = monitor["ticker"].apply(clean_ticker)
        keep = [c for c in [
            "ticker", "price_break_state", "volume_spike_state",
            "volatility_regime_state", "spread_status", "spread_bps",
            "max_monitor_severity",
        ] if c in monitor.columns]
        base = base.merge(monitor[keep], on="ticker", how="left", suffixes=("", "_monitor"))

    return base


def build_model() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = build_base()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    scenario_rows = []
    for _, row in base.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        current_w = pct_to_weight(row.get("current_weight"))
        target_w = pct_to_weight(row.get("target_weight"))
        trade_notional = abs(target_w - current_w) * MODEL_ACCOUNT_VALUE
        direction = "UP" if target_w > current_w else ("DOWN" if target_w < current_w else "FLAT")

        adv = safe_float(row.get("avg_20d_dollar_volume"), np.nan)
        if not np.isfinite(adv):
            adv = safe_float(row.get("avg_20d_dollar_volume_liq"), np.nan)
        liquidity_label = str(row.get("liquidity_label", "MISSING"))
        participation = trade_notional / adv * 100.0 if np.isfinite(adv) and adv > 0 else np.nan

        spread = safe_float(row.get("spread_bps_est"), np.nan)
        monitor_spread = safe_float(row.get("spread_bps"), np.nan)
        if not np.isfinite(spread) and np.isfinite(monitor_spread):
            spread = monitor_spread
        if not np.isfinite(spread):
            spread = spread_from_liquidity(liquidity_label)
        spread_source = "estimated_or_live" if np.isfinite(row.get("spread_bps_est", np.nan)) or np.isfinite(monitor_spread) else "liquidity_proxy"

        severity = str(row.get("max_monitor_severity", row.get("max_monitor_severity_monitor", "OK"))).upper()
        if severity in {"", "NAN"}:
            severity = "OK"
        sev = severity_rank(severity)
        impact = safe_float(row.get("market_impact_bps"), np.nan)
        if not np.isfinite(impact):
            p = max(participation, 0.0001) if np.isfinite(participation) else 1.0
            impact = 4.0 * np.sqrt(p / 1.0)
        auction = safe_float(row.get("auction_risk_bps"), np.nan)
        if not np.isfinite(auction):
            auction = 4.0 + 3.0 * sev
        failed_fill = safe_float(row.get("failed_fill_buffer_bps"), np.nan)
        if not np.isfinite(failed_fill):
            failed_fill = 3.0 + 2.0 * sev

        half_spread = spread / 2.0
        base_cost = half_spread + impact + auction + failed_fill
        wide_spread_cost = spread + impact * 1.25 + auction + failed_fill
        liquidity_shock_cost = spread * 1.5 + impact * 3.0 + auction * 1.5 + failed_fill * 1.5
        auction_stress_cost = half_spread + impact * 1.25 + auction * 2.5 + failed_fill * 1.5
        failed_fill_cost = half_spread + impact * 1.25 + auction + failed_fill * 3.0
        stress_cost = max(wide_spread_cost, liquidity_shock_cost, auction_stress_cost, failed_fill_cost)
        data_gap = not np.isfinite(adv) or adv <= 0
        status = cost_status(base_cost, stress_cost, participation if np.isfinite(participation) else 999, data_gap, severity)
        if direction == "DOWN" and status == "BLOCK_NEW":
            status = "SIZE_DOWN"

        expected_fill = {
            "CLEAR": 0.985,
            "REVIEW": 0.94,
            "SIZE_DOWN": 0.82,
            "BLOCK_NEW": 0.0,
            "DATA_GAP": 0.0,
        }.get(status, 0.85)

        if data_gap:
            instruction = "Manual liquidity check before any paper execution assumption."
        elif status == "BLOCK_NEW":
            instruction = "Do not add exposure; research-only until cost/liquidity clears."
        elif status == "SIZE_DOWN":
            instruction = "Use smaller paper size or staged reduction; avoid open/close auction."
        elif status == "REVIEW":
            instruction = "Manual spread and volume check; prefer mid-day VWAP-style assumption."
        else:
            instruction = "Execution cost acceptable for research paper workflow."

        base_cost_dollars = trade_notional * base_cost / 10000.0
        stress_cost_dollars = trade_notional * stress_cost / 10000.0
        rows.append({
            "ticker": ticker,
            "direction": direction,
            "current_weight_pct": round(current_w * 100, 4),
            "target_weight_pct": round(target_w * 100, 4),
            "trade_notional_dollars": round(trade_notional, 2),
            "avg_20d_dollar_volume": adv,
            "liquidity_label": liquidity_label,
            "participation_rate_pct": round(participation, 6) if np.isfinite(participation) else np.nan,
            "spread_bps": round(spread, 4),
            "spread_source": spread_source,
            "market_impact_bps": round(impact, 4),
            "auction_risk_bps": round(auction, 4),
            "failed_fill_buffer_bps": round(failed_fill, 4),
            "base_cost_bps": round(base_cost, 4),
            "stress_cost_bps": round(stress_cost, 4),
            "base_cost_dollars": round(base_cost_dollars, 2),
            "stress_cost_dollars": round(stress_cost_dollars, 2),
            "expected_fill_rate_pct": round(expected_fill * 100, 1),
            "monitor_severity": severity,
            "price_break_state": row.get("price_break_state", ""),
            "volume_spike_state": row.get("volume_spike_state", ""),
            "volatility_regime_state": row.get("volatility_regime_state", ""),
            "execution_cost_status": status,
            "execution_instruction": instruction,
            "source_file": "institutional_optimizer_bridge.csv / institutional_tca_cost_estimates.csv / intraday_liquidity_proxy.csv / desk_monitor_ticker_state.csv",
            "research_only": True,
        })

        scenarios = {
            "base": base_cost,
            "wide_spread": wide_spread_cost,
            "liquidity_shock": liquidity_shock_cost,
            "auction_stress": auction_stress_cost,
            "failed_fill": failed_fill_cost,
        }
        for scenario, cost in scenarios.items():
            scenario_rows.append({
                "ticker": ticker,
                "scenario": scenario,
                "cost_bps": round(cost, 4),
                "cost_dollars": round(trade_notional * cost / 10000.0, 2),
                "status": cost_status(base_cost, cost, participation if np.isfinite(participation) else 999, data_gap, severity),
                "source_file": "execution_cost_model.csv",
            })

    return pd.DataFrame(rows), pd.DataFrame(scenario_rows)


def build_audit(model: pd.DataFrame) -> pd.DataFrame:
    if model.empty:
        return pd.DataFrame()
    total_notional = float(model["trade_notional_dollars"].sum())
    weighted_base = float((model["base_cost_bps"] * model["trade_notional_dollars"]).sum() / max(total_notional, 1e-12))
    weighted_stress = float((model["stress_cost_bps"] * model["trade_notional_dollars"]).sum() / max(total_notional, 1e-12))
    max_participation = safe_float(model["participation_rate_pct"].max())
    data_gaps = int(model["execution_cost_status"].astype(str).str.upper().eq("DATA_GAP").sum())
    blocked_or_size = int(model["execution_cost_status"].astype(str).str.upper().isin({"BLOCK_NEW", "SIZE_DOWN"}).sum())
    spread_proxy_rows = int(model["spread_source"].astype(str).eq("liquidity_proxy").sum())
    rows = [
        ("ADV coverage", len(model) - data_gaps, len(model), "CLEAR" if data_gaps == 0 else "REVIEW", "Every modeled trade should have average dollar volume.", "intraday_liquidity_proxy.csv"),
        ("Spread source quality", spread_proxy_rows, 0, "CLEAR" if spread_proxy_rows == 0 else "REVIEW", "Proxy spread is usable for research but not enough for institutional TCA.", "desk_monitor_ticker_state.csv"),
        ("Weighted base cost", weighted_base, BASE_COST_REVIEW, "CLEAR" if weighted_base <= BASE_COST_REVIEW else ("REVIEW" if weighted_base <= BASE_COST_SIZE_DOWN else "SIZE_DOWN"), "Base execution cost should stay below signal edge.", "execution_cost_model.csv"),
        ("Weighted stress cost", weighted_stress, STRESS_COST_SIZE_DOWN, "CLEAR" if weighted_stress <= STRESS_COST_SIZE_DOWN else ("SIZE_DOWN" if weighted_stress <= STRESS_COST_BLOCK else "BLOCK_NEW"), "Stress TCA should not make paper sizing meaningless.", "execution_cost_stress_scenarios.csv"),
        ("Max participation", max_participation, PARTICIPATION_REVIEW, "CLEAR" if np.isfinite(max_participation) and max_participation <= PARTICIPATION_REVIEW else "REVIEW", "Avoid assumptions that require too much daily volume.", "execution_cost_model.csv"),
        ("Blocked or size-down names", blocked_or_size, 0, "CLEAR" if blocked_or_size == 0 else "REVIEW", "Names requiring size-down should not be blindly promoted.", "execution_cost_model.csv"),
        ("Fill history", 0, 1, "REVIEW", "No real broker/fill log exists; expected fill rates are assumptions.", "paper/research only"),
        ("No live execution", 1, 1, "PASS", "No broker connection and no live order path.", "code policy"),
    ]
    return pd.DataFrame([{
        "control": name,
        "current_value": current,
        "target_or_limit": limit,
        "status": status,
        "evidence": note,
        "source_file": source,
    } for name, current, limit, status, note, source in rows])


def write_outputs(model: pd.DataFrame, scenarios: pd.DataFrame, audit: pd.DataFrame) -> None:
    model.to_csv(OUT_MODEL, index=False)
    scenarios.to_csv(OUT_SCENARIOS, index=False)
    audit.to_csv(OUT_AUDIT, index=False)

    status_scores = {"PASS": 95, "CLEAR": 90, "REVIEW": 70, "SIZE_DOWN": 45, "BLOCK_NEW": 15, "DATA_GAP": 25}
    model_score = float(model["execution_cost_status"].astype(str).str.upper().map(status_scores).fillna(50).mean()) if not model.empty else 20.0
    audit_score = float(audit["status"].astype(str).str.upper().map(status_scores).fillna(50).mean()) if not audit.empty else 20.0
    score = 0.65 * model_score + 0.35 * audit_score
    weighted_base = 0.0
    weighted_stress = 0.0
    if not model.empty:
        notional = float(model["trade_notional_dollars"].sum())
        weighted_base = float((model["base_cost_bps"] * model["trade_notional_dollars"]).sum() / max(notional, 1e-12))
        weighted_stress = float((model["stress_cost_bps"] * model["trade_notional_dollars"]).sum() / max(notional, 1e-12))
    if score >= 80:
        overall = "EXECUTION_RESEARCH_READY"
    elif score >= 65:
        overall = "EXECUTION_REVIEW_REQUIRED"
    elif score >= 45:
        overall = "EXECUTION_SIZE_DOWN_REQUIRED"
    else:
        overall = "EXECUTION_BLOCKER"
    state = {
        "date": today_str(),
        "execution_cost_model_score": round(score, 1),
        "overall_status": overall,
        "trade_rows": int(len(model)),
        "weighted_base_cost_bps": round(weighted_base, 2),
        "weighted_stress_cost_bps": round(weighted_stress, 2),
        "data_gap_rows": int(model["execution_cost_status"].astype(str).str.upper().eq("DATA_GAP").sum()) if not model.empty else 0,
        "review_or_size_down_rows": int(model["execution_cost_status"].astype(str).str.upper().isin({"REVIEW", "SIZE_DOWN"}).sum()) if not model.empty else 0,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "Scenario TCA for research/paper sizing only. It does not connect to a broker and does not place trades.",
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Execution cost score: **{state['execution_cost_model_score']}/100**",
        f"- Weighted base cost: **{state['weighted_base_cost_bps']} bps**",
        f"- Weighted stress cost: **{state['weighted_stress_cost_bps']} bps**",
        "",
        state["truth"],
        "",
        "## Constraint Audit",
        "",
        df_to_markdown(audit, max_rows=30),
        "",
        "## Ticker Cost Model",
        "",
        df_to_markdown(model, max_rows=80),
        "",
        "## Stress Scenarios",
        "",
        df_to_markdown(scenarios, max_rows=100),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 158 - Execution Cost Stress Model", sections)


def main() -> None:
    model, scenarios = build_model()
    audit = build_audit(model)
    write_outputs(model, scenarios, audit)
    from canyon_final_v9_risk_framework_lib import read_json_safe
    payload = read_json_safe(OUT_STATE, {})
    print("Canyon v9 Step158 execution cost stress model complete.")
    print(f"Overall: {payload.get('overall_status')} ({payload.get('execution_cost_model_score')}/100)")
    print(f"Weighted base/stress cost: {payload.get('weighted_base_cost_bps')} / {payload.get('weighted_stress_cost_bps')} bps")
    print(f"Outputs: {OUT_MODEL.name}, {OUT_SCENARIOS.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
