#!/usr/bin/env python3
"""
Canyon v9 Step 169 - Research Promotion Gate.

Research-only. No broker connection. No live orders.

This is the institutional "prove it before sizing" layer. It does not create
new alpha. It decides whether existing research can be promoted from:

  1. reduce risk only
  2. research only
  3. tiny paper only
  4. manual-approval paper candidate

The gate combines risk, optimizer, signal IC/decay, backtest credibility,
point-in-time data truth, event reliability, execution cost, and live monitor
state. Missing evidence can reduce or block a decision. It cannot upgrade it.

Outputs:
  research_promotion_gate.csv
  research_promotion_component_scores.csv
  research_signal_weight_policy.csv
  institutional_gap_to_top_quant.csv
  research_promotion_state.json
  research_promotion_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    now_str,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_GATE = ROOT / "research_promotion_gate.csv"
OUT_COMPONENTS = ROOT / "research_promotion_component_scores.csv"
OUT_SIGNAL_POLICY = ROOT / "research_signal_weight_policy.csv"
OUT_GAP = ROOT / "institutional_gap_to_top_quant.csv"
OUT_STATE = ROOT / "research_promotion_state.json"
OUT_REPORT = ROOT / "research_promotion_report.md"


TOP_SIGNAL_TO_VALIDATION_SIGNAL = {
    "momentum": "mom_12m_skip1m",
    "quality": "quality_hist",
    "revision": "rev_growth_yoy",
    "surprise": "eps_growth_yoy",
    "regime_ml": "",
    "ml_ensemble": "",
    "sentiment": "",
    "squeeze": "",
    "insider": "",
    "options": "",
}

STATUS_SCORE = {
    "CLEAR": 82,
    "OK": 78,
    "PASS": 78,
    "READY": 75,
    "WATCH": 62,
    "REVIEW": 55,
    "WEAK": 42,
    "SIZE_DOWN": 38,
    "DATA_GAP": 35,
    "LOW_SAMPLE_REVIEW": 35,
    "UNPROVEN_LOCAL_CONTEXT": 32,
    "BLOCK_NEW": 22,
    "BLOCKER": 18,
    "BLOCKED": 18,
    "REDUCE_ONLY": 15,
    "NO_DATA": 25,
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", ""))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def one_by_ticker(df: pd.DataFrame, ticker_col: str = "ticker") -> dict[str, dict[str, Any]]:
    if df.empty or ticker_col not in df.columns:
        return {}
    work = df.copy()
    work[ticker_col] = work[ticker_col].apply(clean_ticker)
    work = work[work[ticker_col] != ""].drop_duplicates(ticker_col, keep="first")
    return work.set_index(ticker_col).to_dict(orient="index")


def score_from_status(status: Any, default: float = 45.0) -> float:
    text = str(status or "").upper().strip()
    if not text:
        return default
    for key, score in STATUS_SCORE.items():
        if key in text:
            return float(score)
    return default


def score_from_signal_action(action: Any) -> float:
    action_u = str(action or "").upper()
    if action_u == "KEEP_CORE":
        return 84.0
    if action_u == "KEEP_WITH_MONITOR":
        return 70.0
    if action_u == "USE_ONLY_AT_SHORT_HORIZON":
        return 58.0
    if action_u == "REVIEW_SAMPLE_SIZE":
        return 46.0
    if action_u == "DOWNWEIGHT":
        return 42.0
    if action_u == "BLOCK_SIGNAL":
        return 18.0
    if action_u == "NO_DATA":
        return 25.0
    return 40.0


def score_from_tca(row: dict[str, Any]) -> float:
    status = str(row.get("execution_status", "")).upper()
    bps = safe_float(row.get("total_tca_cost_bps"), np.nan)
    score = score_from_status(status, 65.0)
    if np.isfinite(bps):
        if bps >= 80:
            score = min(score, 18.0)
        elif bps >= 45:
            score = min(score, 35.0)
        elif bps >= 25:
            score = min(score, 55.0)
        else:
            score = min(max(score, 70.0), 82.0)
    return score


def score_from_monitor(row: dict[str, Any]) -> float:
    severity = str(row.get("max_monitor_severity", row.get("max_severity", ""))).upper()
    if "CRITICAL" in severity:
        return 18.0
    if "WARNING" in severity:
        return 48.0
    if "DATA_GAP" in severity:
        return 35.0
    return 74.0


def score_from_event(row: dict[str, Any], event_state: dict[str, Any]) -> float:
    if not row:
        return 50.0
    status = str(row.get("reliability_status", "")).upper()
    score = safe_float(row.get("reliability_score"), score_from_status(status, 45.0))
    model_seen = safe_float(event_state.get("model_seen_1d_coverage_pct"), 0.0)
    if model_seen <= 0:
        score = min(score, 45.0)
    if "LOW_SAMPLE" in status:
        score = min(score, 42.0)
    if "UNPROVEN" in status:
        score = min(score, 35.0)
    return max(10.0, min(float(score), 85.0))


def top_blockers(component_rows: list[dict[str, Any]], max_items: int = 4) -> str:
    weak = [r for r in component_rows if safe_float(r.get("component_score"), 100.0) < 50]
    weak = sorted(weak, key=lambda r: safe_float(r.get("component_score"), 100.0))
    return "; ".join(f"{r['component']}: {r['status']}" for r in weak[:max_items]) or "No hard blocker; still research-only."


def proof_required(blockers: str) -> str:
    text = blockers.lower()
    if "risk" in text:
        return "Risk gate must improve first; do not let alpha, news, or options jump the line."
    if "signal" in text:
        return "Repair or down-weight the signal, then collect live IC observations before increasing size."
    if "backtest" in text or "point-in-time" in text:
        return "Add point-in-time proof, survivorship controls, and frozen walk-forward signal history."
    if "event" in text or "news" in text:
        return "Require model-seen timestamps and enough event reaction samples before trusting news as alpha."
    if "execution" in text:
        return "Prove expected spread, impact, and fill risk are smaller than the signal edge."
    if "monitor" in text:
        return "Wait for live price, volume, spread, and volatility alerts to calm or be explained."
    return "Manual review still required because the system is research-only and has no live order path."


def promotion_status(score: float, risk_action: str, gate_status: str) -> tuple[str, str, float]:
    risk_u = str(risk_action or "").upper()
    gate_u = str(gate_status or "").upper()
    if "REDUCE_ONLY" in risk_u or "NO NEW EXPOSURE" in gate_u:
        return "Reduce risk only", "No new buying; reduce or protect existing exposure.", 0.0
    if score < 40:
        return "Do not size", "Keep in research notes only.", 0.0
    if score < 55:
        return "Research only", "No paper sizing until blockers are repaired.", 0.15
    if score < 70:
        return "Tiny paper only", "Tiny paper sizing only after manual checks.", 0.35
    return "Manual approval paper candidate", "Paper candidate only; no live order path exists.", 0.60


def build_signal_policy() -> pd.DataFrame:
    failure = read_csv_safe(ROOT / "signal_failure_deep_dive.csv")
    drift = read_csv_safe(ROOT / "signal_live_vs_backtest_drift.csv")
    decay = read_csv_safe(ROOT / "signal_decay_analysis.csv")
    if failure.empty:
        return pd.DataFrame(columns=[
            "signal", "signal_policy", "allowed_horizon", "signal_score",
            "weight_multiplier", "why", "proof_required",
        ])

    drift_map = {}
    if not drift.empty and "signal" in drift.columns:
        drift_map = drift.drop_duplicates("signal").set_index("signal").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    for _, row in failure.iterrows():
        signal = str(row.get("signal", ""))
        action = str(row.get("recommended_signal_action", "NO_DATA"))
        score = score_from_signal_action(action)
        live = drift_map.get(signal, {})
        live_obs = int(safe_float(live.get("live_observations"), 0))
        drift_status = str(live.get("drift_status", "DATA_GAP"))
        if live_obs < 30:
            score = min(score, 64.0)

        sub = decay[decay.get("signal", pd.Series(dtype=str)).astype(str).eq(signal)] if not decay.empty and "signal" in decay.columns else pd.DataFrame()
        usable_horizons = []
        if not sub.empty:
            for _, drow in sub.iterrows():
                if str(drow.get("status", "")).upper() in {"STRONG", "USABLE", "WEAK"} and safe_float(drow.get("mean_ic"), -1) > 0:
                    usable_horizons.append(f"{int(safe_float(drow.get('horizon_days'), 0))}d")
        allowed = ", ".join(usable_horizons) if usable_horizons else "None proven"

        if action == "BLOCK_SIGNAL":
            policy = "Blocked until repaired"
            multiplier = 0.0
        elif action == "DOWNWEIGHT":
            policy = "Down-weight only"
            multiplier = 0.35
        elif action == "REVIEW_SAMPLE_SIZE":
            policy = "Sample-size review"
            multiplier = 0.25
        elif action == "USE_ONLY_AT_SHORT_HORIZON":
            policy = "Short horizon only"
            multiplier = 0.30
        elif action == "KEEP_WITH_MONITOR":
            policy = "Keep with monitor"
            multiplier = 0.60
        elif action == "KEEP_CORE":
            policy = "Core research signal"
            multiplier = 0.80
        else:
            policy = "No-data review"
            multiplier = 0.0

        if live_obs < 30:
            multiplier = min(multiplier, 0.50)

        rows.append({
            "signal": signal,
            "signal_policy": policy,
            "recommended_signal_action": action,
            "allowed_horizon": allowed,
            "signal_score": round(score, 1),
            "weight_multiplier": round(multiplier, 2),
            "baseline_mean_ic": row.get("baseline_mean_ic"),
            "baseline_n_obs": row.get("baseline_n_obs"),
            "best_horizon": row.get("best_horizon"),
            "best_horizon_ic": row.get("best_horizon_ic"),
            "worst_horizon": row.get("worst_horizon"),
            "worst_horizon_ic": row.get("worst_horizon_ic"),
            "live_observations": live_obs,
            "drift_status": drift_status,
            "why": row.get("reason", ""),
            "proof_required": row.get("required_next_action", "Collect live validation before promotion."),
            "source_files": "signal_failure_deep_dive.csv / signal_decay_analysis.csv / signal_live_vs_backtest_drift.csv",
        })
    return pd.DataFrame(rows).sort_values(["weight_multiplier", "signal_score", "signal"], ascending=[True, True, True]).reset_index(drop=True)


def build_gap_scoreboard() -> pd.DataFrame:
    bt = read_json_safe(ROOT / "backtest_credibility_state.json", {})
    sig = read_json_safe(ROOT / "signal_validation_state.json", {})
    pit = read_json_safe(ROOT / "pit_truth_state.json", {})
    opt = read_json_safe(ROOT / "institutional_optimizer_state.json", {})
    exec_state = read_json_safe(ROOT / "execution_cost_model_state.json", {})
    event = read_json_safe(ROOT / "event_signal_reliability_state.json", {})
    risk = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})

    rows = [
        ("Product dashboard", 62, "Usable internal research dashboard; still not a Bloomberg-grade terminal.", "canyon_final_v9_step86_dashboard_v3.py"),
        ("Strategy thesis", 45, "Step167/168 explain thesis and vehicle, but thesis quality is still mostly rule/proxy driven.", "institutional_strategy_thesis_board.csv"),
        ("Risk management", 48 if risk else 30, "Good first-pass gate; thresholds still need historical calibration and intraday/options book risk.", "institutional_risk_gate_state.json"),
        ("Portfolio construction", 38 if opt else 25, f"Optimizer state {opt.get('overall_status', 'NO_DATA')}; still review-required.", "institutional_optimizer_state.json"),
        ("Signal research", 28 if int(sig.get("p1_signal_repairs", 0) or 0) else 38, f"Signal state {sig.get('overall_status', 'NO_DATA')}; live IC still thin.", "signal_validation_state.json"),
        ("Backtest credibility", 22 if bt.get("overall_status") == "PROTOTYPE_ONLY" else 35, f"Backtest state {bt.get('overall_status', 'NO_DATA')}; local evidence is not production evidence.", "backtest_credibility_state.json"),
        ("Point-in-time data", 24 if pit.get("overall_status") == "PIT_REVIEW_REQUIRED" else 35, f"PIT state {pit.get('overall_status', 'NO_DATA')}; vendor-grade proof still missing.", "pit_truth_state.json"),
        ("News and event causality", 34 if event else 25, f"Event state {event.get('overall_status', 'NO_DATA')}; model-seen coverage is {event.get('model_seen_1d_coverage_pct', 0)}%.", "event_signal_reliability_state.json"),
        ("Execution cost and TCA", 18 if exec_state.get("overall_status") == "EXECUTION_REVIEW_REQUIRED" else 25, f"Execution state {exec_state.get('overall_status', 'NO_DATA')}; no real fills or market-impact calibration.", "execution_cost_model_state.json"),
        ("Live trading readiness", 3, "Intentionally near zero: no broker connection and no live order path by design.", "research-only policy"),
    ]
    out = pd.DataFrame(rows, columns=["section", "top_quant_readiness_pct", "assessment", "source_files"])
    out["gap_to_top_quant_pct"] = 100 - out["top_quant_readiness_pct"]
    out["status"] = pd.cut(
        out["top_quant_readiness_pct"],
        bins=[-1, 20, 40, 60, 75, 100],
        labels=["Very early", "Prototype", "Developing", "Strong prototype", "Near institutional"],
    ).astype(str)
    return out


def component(name: str, score: float, status: str, evidence: str, source: str) -> dict[str, Any]:
    return {
        "component": name,
        "component_score": round(float(score), 1),
        "status": status,
        "evidence": evidence,
        "source_files": source,
    }


def build_gate(signal_policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    strategy = one_by_ticker(read_csv_safe(ROOT / "institutional_strategy_thesis_board.csv"))
    horizon = one_by_ticker(read_csv_safe(ROOT / "horizon_vehicle_summary.csv"))
    optimizer = one_by_ticker(read_csv_safe(ROOT / "institutional_optimizer_bridge.csv"))
    risk = one_by_ticker(read_csv_safe(ROOT / "risk_desk_ticker_action_queue.csv"))
    daily = one_by_ticker(read_csv_safe(ROOT / "daily_picks_filtered.csv"))
    tca = one_by_ticker(read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv"))
    monitor = one_by_ticker(read_csv_safe(ROOT / "desk_monitor_ticker_state.csv"))
    event_rel = one_by_ticker(read_csv_safe(ROOT / "event_signal_reliability_by_ticker.csv"), "target_ticker")

    signal_map = {}
    if not signal_policy.empty and "signal" in signal_policy.columns:
        signal_map = signal_policy.drop_duplicates("signal").set_index("signal").to_dict(orient="index")

    bt_state = read_json_safe(ROOT / "backtest_credibility_state.json", {})
    pit_state = read_json_safe(ROOT / "pit_truth_state.json", {})
    event_state = read_json_safe(ROOT / "event_signal_reliability_state.json", {})
    risk_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})

    backtest_score = safe_float(bt_state.get("overall_credibility_score"), 45.0)
    if str(bt_state.get("overall_status", "")).upper() == "PROTOTYPE_ONLY":
        backtest_score = min(backtest_score, 58.0)
    pit_score = safe_float(pit_state.get("pit_truth_score"), 45.0)
    if str(pit_state.get("overall_status", "")).upper() == "PIT_REVIEW_REQUIRED":
        pit_score = min(pit_score, 62.0)

    tickers = sorted(set(strategy) | set(horizon) | set(optimizer) | set(risk) | set(daily))
    gate_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    weights = {
        "Risk gate": 0.25,
        "Signal proof": 0.20,
        "Backtest credibility": 0.15,
        "Point-in-time proof": 0.12,
        "Optimizer fit": 0.10,
        "Event reliability": 0.08,
        "Execution cost": 0.06,
        "Live monitor": 0.04,
    }

    for ticker in tickers:
        s = strategy.get(ticker, {})
        h = horizon.get(ticker, {})
        o = optimizer.get(ticker, {})
        r = risk.get(ticker, {})
        d = daily.get(ticker, {})
        e = event_rel.get(ticker, {})
        x = tca.get(ticker, {})
        m = monitor.get(ticker, {})

        sector = s.get("sector") or h.get("sector") or o.get("sector") or r.get("sector") or d.get("sector") or "Unknown"
        top_signal = clean_text(o.get("top_signal")) or clean_text(d.get("top_signal"))
        top_signal = top_signal.lower()
        validation_signal = TOP_SIGNAL_TO_VALIDATION_SIGNAL.get(top_signal, "")
        signal_row = signal_map.get(validation_signal, {}) if validation_signal else {}
        if signal_row:
            signal_score = safe_float(signal_row.get("signal_score"), 40.0)
            signal_status = str(signal_row.get("signal_policy", "Needs validation"))
            signal_evidence = f"{validation_signal}: {signal_row.get('why', '')}"
            signal_multiplier = safe_float(signal_row.get("weight_multiplier"), 0.25)
        elif top_signal:
            signal_score = 38.0
            signal_status = "No IC bridge"
            signal_evidence = f"Top signal {top_signal} is not mapped to a validated IC/decay policy."
            signal_multiplier = 0.20
        else:
            signal_score = 32.0
            signal_status = "Missing top signal"
            signal_evidence = "No top signal found for promotion review."
            signal_multiplier = 0.0

        risk_action = str(r.get("final_risk_action") or o.get("final_risk_action") or "")
        gate_status = str(h.get("gate_status") or s.get("gate_status") or "")
        risk_score = score_from_status(risk_action or gate_status, 45.0)
        if "SIZE_DOWN" in str(risk_state.get("master_risk_action", "")).upper():
            risk_score = min(risk_score, 48.0)
        opt_status = str(o.get("final_optimizer_status", "DATA_GAP"))
        optimizer_score = score_from_status(opt_status, 45.0)
        event_score = score_from_event(e, event_state)
        execution_score = score_from_tca(x)
        monitor_score = score_from_monitor(m)

        comps = [
            component("Risk gate", risk_score, risk_action or gate_status or "No risk row", r.get("reason_stack", "") or h.get("main_blocker", ""), "risk_desk_ticker_action_queue.csv / horizon_vehicle_summary.csv"),
            component("Signal proof", signal_score, signal_status, signal_evidence, "research_signal_weight_policy.csv / signal_failure_deep_dive.csv"),
            component("Backtest credibility", backtest_score, bt_state.get("overall_status", "NO_DATA"), "Local/proxy backtest credibility, capped for prototype status.", "backtest_credibility_state.json"),
            component("Point-in-time proof", pit_score, pit_state.get("overall_status", "NO_DATA"), "PIT readiness for using historical evidence in sizing.", "pit_truth_state.json"),
            component("Optimizer fit", optimizer_score, opt_status, o.get("binding_constraints", ""), "institutional_optimizer_bridge.csv"),
            component("Event reliability", event_score, e.get("reliability_status", "No ticker event row"), e.get("reliability_reason", ""), "event_signal_reliability_by_ticker.csv"),
            component("Execution cost", execution_score, x.get("execution_status", "No TCA row"), f"{x.get('total_tca_cost_bps', '')} bps estimated cost.", "institutional_tca_cost_estimates.csv"),
            component("Live monitor", monitor_score, m.get("max_monitor_severity", "No monitor row"), "Price, volume, volatility, spread, and alert state.", "desk_monitor_ticker_state.csv"),
        ]

        raw_score = sum(c["component_score"] * weights[c["component"]] for c in comps)
        score = float(raw_score)
        if "SIZE_DOWN" in str(risk_state.get("master_risk_action", "")).upper():
            score = min(score, 64.0)
        if str(bt_state.get("overall_status", "")).upper() == "PROTOTYPE_ONLY":
            score = min(score, 67.0)
        if str(event_state.get("model_seen_1d_coverage_pct", 0)) == "0.0":
            score = min(score, 66.0)
        if risk_score <= 20:
            score = min(score, 35.0)
        if signal_score <= 20:
            score = min(score, 48.0)

        status, permission, base_multiplier = promotion_status(score, risk_action, gate_status)
        multiplier = min(base_multiplier, signal_multiplier if signal_multiplier > 0 else base_multiplier)
        if "SIZE_DOWN" in str(risk_state.get("master_risk_action", "")).upper():
            multiplier = min(multiplier, 0.50)
        final_opt_pct = safe_float(o.get("final_optimizer_weight_pct"), 0.0)
        risk_target_pct = safe_float(o.get("risk_gated_target_pct", r.get("recommended_risk_weight_pct", 0.0)), 0.0)
        base_cap = min(v for v in [final_opt_pct, risk_target_pct] if np.isfinite(v)) if any(np.isfinite(v) for v in [final_opt_pct, risk_target_pct]) else 0.0
        max_paper_weight_pct = round(max(0.0, base_cap * multiplier), 4)
        blockers = top_blockers(comps)

        for c in comps:
            component_rows.append({
                "ticker": ticker,
                "sector": sector,
                **c,
                "research_only": True,
            })

        gate_rows.append({
            "ticker": ticker,
            "sector": sector,
            "promotion_status": status,
            "sizing_permission": permission,
            "promotion_score": round(score, 1),
            "max_paper_weight_pct": max_paper_weight_pct,
            "current_weight_pct": r.get("current_weight_pct", s.get("current_weight_pct", "")),
            "risk_target_weight_pct": risk_target_pct,
            "optimizer_target_weight_pct": final_opt_pct,
            "master_risk_action": risk_state.get("master_risk_action", "NO_DATA"),
            "top_signal": top_signal,
            "validation_signal": validation_signal or "Not mapped",
            "signal_policy": signal_status,
            "best_horizon": h.get("best_horizon", s.get("best_horizon", "")),
            "horizon_consensus": h.get("horizon_consensus", s.get("horizon_consensus", "")),
            "option_route": h.get("option_route", s.get("option_expression", "")),
            "call_status": h.get("call_status", s.get("call_status", "")),
            "put_status": h.get("put_status", s.get("put_status", "")),
            "event_reliability_status": e.get("reliability_status", "No ticker event row"),
            "execution_status": x.get("execution_status", "No TCA row"),
            "monitor_status": m.get("max_monitor_severity", "No monitor row"),
            "top_blockers": blockers,
            "first_real_proof_required": proof_required(blockers),
            "one_line_verdict": f"{ticker}: {status}. {permission}",
            "source_files": "risk_desk_ticker_action_queue.csv / institutional_optimizer_bridge.csv / signal_validation_state.json / backtest_credibility_state.json / pit_truth_state.json / event_signal_reliability_by_ticker.csv",
            "research_only": True,
            "no_broker_connection": True,
            "no_live_orders": True,
        })

    gate = pd.DataFrame(gate_rows)
    comps = pd.DataFrame(component_rows)
    if not gate.empty:
        order = {
            "Reduce risk only": 0,
            "Do not size": 1,
            "Research only": 2,
            "Tiny paper only": 3,
            "Manual approval paper candidate": 4,
        }
        gate["status_rank"] = gate["promotion_status"].map(order).fillna(9)
        gate = gate.sort_values(["status_rank", "promotion_score"], ascending=[True, False]).drop(columns=["status_rank"]).reset_index(drop=True)

    counts = gate.get("promotion_status", pd.Series(dtype=str)).value_counts().to_dict() if not gate.empty else {}
    state = {
        "date": today_str(),
        "generated_at": now_str(),
        "overall_status": "RESEARCH_PROMOTION_REVIEW",
        "tickers_reviewed": int(len(gate)),
        "avg_promotion_score": round(float(pd.to_numeric(gate.get("promotion_score", pd.Series(dtype=float)), errors="coerce").mean()), 1) if not gate.empty else 0.0,
        "promotion_counts": counts,
        "manual_approval_candidates": int(counts.get("Manual approval paper candidate", 0)),
        "tiny_paper_only": int(counts.get("Tiny paper only", 0)),
        "research_only_or_worse": int(counts.get("Reduce risk only", 0) + counts.get("Do not size", 0) + counts.get("Research only", 0)),
        "max_total_paper_weight_pct": round(float(pd.to_numeric(gate.get("max_paper_weight_pct", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()), 2) if not gate.empty else 0.0,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "This gate decides whether local research evidence is strong enough for paper sizing. It does not allow live orders and cannot override risk.",
    }
    return gate, comps, state


def write_report(gate: pd.DataFrame, comps: pd.DataFrame, signal_policy: pd.DataFrame, gap: pd.DataFrame, state: dict[str, Any]) -> None:
    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Tickers reviewed: **{state['tickers_reviewed']}**",
        f"- Average promotion score: **{state['avg_promotion_score']}/100**",
        f"- Manual approval candidates: **{state['manual_approval_candidates']}**",
        f"- Tiny paper only: **{state['tiny_paper_only']}**",
        f"- Research-only or worse: **{state['research_only_or_worse']}**",
        f"- Max total paper weight allowed by this gate: **{state['max_total_paper_weight_pct']}%**",
        "",
        state["truth"],
        "",
        "## Research Promotion Gate",
        "",
        df_to_markdown(gate, max_rows=80),
        "",
        "## Weak Components",
        "",
        df_to_markdown(comps[pd.to_numeric(comps.get("component_score", pd.Series(dtype=float)), errors="coerce") < 50], max_rows=120),
        "",
        "## Signal Weight Policy",
        "",
        df_to_markdown(signal_policy, max_rows=80),
        "",
        "## Gap To Top Quant",
        "",
        df_to_markdown(gap, max_rows=40),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 169 - Research Promotion Gate", sections)


def main() -> None:
    signal_policy = build_signal_policy()
    gate, comps, state = build_gate(signal_policy)
    gap = build_gap_scoreboard()

    signal_policy.to_csv(OUT_SIGNAL_POLICY, index=False)
    gate.to_csv(OUT_GATE, index=False)
    comps.to_csv(OUT_COMPONENTS, index=False)
    gap.to_csv(OUT_GAP, index=False)
    write_json(OUT_STATE, state)
    write_report(gate, comps, signal_policy, gap, state)

    print("Canyon v9 Step169 research promotion gate complete.")
    print(f"Overall: {state['overall_status']} | tickers: {state['tickers_reviewed']} | avg score: {state['avg_promotion_score']}/100")
    print(f"Max total paper weight: {state['max_total_paper_weight_pct']}% | manual candidates: {state['manual_approval_candidates']}")
    print(f"Outputs: {OUT_GATE.name}, {OUT_COMPONENTS.name}, {OUT_SIGNAL_POLICY.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
