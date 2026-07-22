#!/usr/bin/env python3
"""
Canyon v9 Step 194 - Institutional Depth 5 Workbench.

Research-only. No broker connection. No live orders.

This step does not add another shallow signal. It consolidates five deep
institutional modules into one PM-readable workbench:

1. Backtest Credibility Center
2. Signal IC / Decay / Failure Lab
3. Portfolio Optimizer 2.0
4. Execution Cost / Liquidity Desk
5. News-to-Industry Causal Proof System

Outputs:
  institutional_depth5_state.json
  institutional_depth5_module_scorecard.csv
  depth5_backtest_credibility_center.csv
  depth5_signal_ic_decay_failure_lab.csv
  depth5_portfolio_optimizer_v2.csv
  depth5_execution_liquidity_desk.csv
  depth5_news_causal_proof_system.csv
  institutional_depth5_priority_queue.csv
  institutional_depth5_report.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_STATE = ROOT / "institutional_depth5_state.json"
OUT_SCORECARD = ROOT / "institutional_depth5_module_scorecard.csv"
OUT_BACKTEST = ROOT / "depth5_backtest_credibility_center.csv"
OUT_SIGNAL = ROOT / "depth5_signal_ic_decay_failure_lab.csv"
OUT_PORTFOLIO = ROOT / "depth5_portfolio_optimizer_v2.csv"
OUT_EXECUTION = ROOT / "depth5_execution_liquidity_desk.csv"
OUT_NEWS = ROOT / "depth5_news_causal_proof_system.csv"
OUT_QUEUE = ROOT / "institutional_depth5_priority_queue.csv"
OUT_REPORT = ROOT / "institutional_depth5_report.md"


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    x = safe_float(value, np.nan)
    if not np.isfinite(x):
        return default
    return int(x)


def clamp_score(value: Any) -> float:
    x = safe_float(value, 0.0)
    return round(float(np.clip(x, 0.0, 100.0)), 1)


def status_from_score(score: float, hard_block: bool = False) -> str:
    if hard_block:
        return "BLOCKED"
    if score >= 85:
        return "Sizing evidence"
    if score >= 70:
        return "Research usable"
    if score >= 55:
        return "Prototype only"
    if score >= 40:
        return "Repair required"
    return "Not reliable yet"


def plain_status(value: Any) -> str:
    text = as_text(value, "No data")
    replacements = {
        "DATA_GAP": "missing data",
        "SIZE_DOWN": "use smaller size",
        "REDUCE_ONLY": "reduce only",
        "BLOCK_SIGNAL": "block this signal",
        "DOWNWEIGHT": "down-weight this signal",
        "THIN_SAMPLE": "sample is too small",
        "NEGATIVE": "negative evidence",
        "WEAK": "weak evidence",
        "REVIEW": "needs review",
        "CLEAR": "clear",
        "PASS": "pass",
        "PROTOTYPE_ONLY": "prototype only",
        "SIGNAL_REPAIR_REQUIRED": "signal repair required",
        "EXECUTION_SIZE_DOWN_REQUIRED": "execution needs smaller size",
        "CAUSAL_REVIEW_REQUIRED": "causal links need review",
        "CONTRADICTED_REVIEW_REQUIRED": "price disagrees with the story",
        "HYPOTHESIS_NEEDS_VALIDATION": "hypothesis needs proof",
        "WATCH_FOR_CONFIRMATION": "watch, but prove it first",
        "CALL_RESEARCH_ONLY": "call research only",
        "PUT_OR_HEDGE_RESEARCH_ONLY": "put or hedge research only",
        "RISK_REDUCTION_ONLY": "risk reduction only",
        "MANUAL_SPREAD_LIQUIDITY_CHECK": "manual spread and liquidity check",
        "NOT_IN_RISK_BOOK_REVIEW": "not in risk book; needs review",
        "UNKNOWN_NEEDS_DATA": "unknown; needs data",
        "STOCK_OR_ETF_RESEARCH_ONLY": "stock or ETF research only",
        "PEER_READ_THROUGH": "peer read-through",
        "UPSTREAM_BENEFICIARY": "supplier or upstream winner",
        "DOWNSTREAM_BENEFICIARY": "customer or downstream winner",
        "BENEFICIARY": "direct possible winner",
        "VULNERABLE_TARGET": "possible loser",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    text = text.replace("_", " ")
    cleanup = {
        "NOT IN RISK BOOK needs review": "not in risk book; needs review",
        "UNKNOWN needs data": "unknown; needs data",
        "WATCH EVENT PROOF FIRST": "watch event proof first",
    }
    for raw, friendly in cleanup.items():
        text = text.replace(raw, friendly)
    return text


def shorten(value: Any, limit: int = 220) -> str:
    text = " ".join(as_text(value, "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def module_card(
    module: str,
    score: float,
    status: str,
    can_use_for_sizing: str,
    biggest_gap: str,
    next_action: str,
    source_files: str,
) -> dict[str, Any]:
    return {
        "module": module,
        "score_0_100": clamp_score(score),
        "status": status,
        "can_use_for_sizing": can_use_for_sizing,
        "biggest_gap": biggest_gap,
        "next_action": next_action,
        "source_files": source_files,
        "research_only": True,
    }


def build_backtest_center() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scorecard = read_csv_safe(ROOT / "backtest_credibility_scorecard.csv")
    blockers = read_csv_safe(ROOT / "backtest_credibility_blockers.csv")
    walk = read_csv_safe(ROOT / "backtest_walk_forward_proxy.csv")
    reality = read_csv_safe(ROOT / "backtest_execution_reality_check.csv")
    pit = read_csv_safe(ROOT / "pit_truth_scorecard.csv")
    state = read_json_safe(ROOT / "backtest_credibility_state.json", {})

    rows: list[dict[str, Any]] = []
    if not scorecard.empty:
        for _, row in scorecard.iterrows():
            status = as_text(row.get("status"), "No data")
            score = safe_float(row.get("score_0_100"), np.nan)
            can_size = "No" if status.upper() in {"WEAK", "BLOCKER", "DATA_GAP"} else "Only as research evidence"
            rows.append({
                "check": as_text(row.get("category"), "Backtest check"),
                "score_0_100": round(score, 1) if np.isfinite(score) else np.nan,
                "plain_status": plain_status(status),
                "can_use_for_sizing": can_size,
                "what_it_means": shorten(row.get("why_it_matters"), 260),
                "missing_proof": shorten(row.get("next_required_action"), 280),
                "source_files": as_text(row.get("source_files"), "backtest files"),
            })

    if not walk.empty:
        weak = int(walk.get("status", pd.Series(dtype=str)).astype(str).str.upper().str.contains("WEAK|BLOCK|DATA", regex=True).sum())
        rows.append({
            "check": "Walk-forward stability",
            "score_0_100": 82.0 if weak == 0 else 55.0,
            "plain_status": "research usable" if weak == 0 else "needs review",
            "can_use_for_sizing": "Only as research evidence",
            "what_it_means": f"{len(walk)} historical windows were checked. This is useful, but still a proxy until signals are frozen before each test.",
            "missing_proof": "Replace proxy walk-forward with frozen train/test folds and a true holdout period.",
            "source_files": "backtest_walk_forward_proxy.csv",
        })

    if not reality.empty:
        weak_reality = int(reality.get("status", pd.Series(dtype=str)).astype(str).str.upper().str.contains("WEAK|REVIEW|BLOCK", regex=True).sum())
        rows.append({
            "check": "Execution realism",
            "score_0_100": 45.0 if weak_reality else 75.0,
            "plain_status": "repair required" if weak_reality else "research usable",
            "can_use_for_sizing": "No" if weak_reality else "Only as research evidence",
            "what_it_means": "Backtest returns still need more realistic fills, slippage, auction risk, and missed-fill assumptions.",
            "missing_proof": "Add current TCA, bid/ask spread, participation, failed fills, and delayed entry to backtest PnL.",
            "source_files": "backtest_execution_reality_check.csv",
        })

    overall_score = clamp_score(state.get("overall_credibility_score", np.nan))
    status = status_from_score(overall_score)
    weak_count = safe_int(state.get("weak_count"), 0)
    blocker_rows = len(blockers) if not blockers.empty else safe_int(state.get("blocker_rows"), 0)
    biggest_gap = "Execution realism, signal IC proof, and point-in-time data are still not institutional-grade."
    if not blockers.empty:
        first = blockers.sort_values("priority").iloc[0]
        biggest_gap = f"{as_text(first.get('category'), 'Backtest blocker')}: {shorten(first.get('next_required_action'), 180)}"
    summary = {
        "score": overall_score,
        "status": status,
        "can_use_for_sizing": "No. Use for research only." if overall_score < 85 else "Yes, with risk controls.",
        "biggest_gap": biggest_gap,
        "next_action": "Close weak controls, add true point-in-time data, and rerun out-of-sample tests before trusting sizing.",
        "weak_count": weak_count,
        "blocker_rows": blocker_rows,
    }

    queue: list[dict[str, Any]] = []
    if not blockers.empty:
        for _, row in blockers.head(5).iterrows():
            queue.append({
                "priority": as_text(row.get("priority"), "P2"),
                "module": "Backtest Credibility Center",
                "item": as_text(row.get("category"), "Backtest blocker"),
                "why_it_matters": shorten(row.get("evidence"), 240),
                "next_action": shorten(row.get("next_required_action"), 260),
                "source_files": as_text(row.get("source_files"), "backtest files"),
            })
    return pd.DataFrame(rows), summary, queue


def build_signal_lab() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    decay = read_csv_safe(ROOT / "signal_decay_analysis.csv")
    queue = read_csv_safe(ROOT / "signal_downgrade_queue.csv")
    regime = read_csv_safe(ROOT / "signal_regime_ic_matrix.csv")
    drift = read_csv_safe(ROOT / "signal_live_vs_backtest_drift.csv")
    state = read_json_safe(ROOT / "signal_validation_state.json", {})

    rows: list[dict[str, Any]] = []
    if not decay.empty:
        work = decay.copy()
        work["mean_ic_num"] = pd.to_numeric(work.get("mean_ic"), errors="coerce")
        for sig, grp in work.groupby("signal"):
            best = grp.sort_values("mean_ic_num", ascending=False).iloc[0]
            worst = grp.sort_values("mean_ic_num", ascending=True).iloc[0]
            action_row = queue[queue["signal"].astype(str) == str(sig)].head(1) if not queue.empty and "signal" in queue.columns else pd.DataFrame()
            action = as_text(action_row.iloc[0].get("recommended_signal_action"), "Keep under review") if not action_row.empty else "Keep under review"
            live_row = drift[drift["signal"].astype(str) == str(sig)].head(1) if not drift.empty and "signal" in drift.columns else pd.DataFrame()
            drift_status = as_text(live_row.iloc[0].get("drift_status"), "No live proof yet") if not live_row.empty else "No live proof yet"
            rows.append({
                "signal": sig,
                "best_horizon": f"{safe_int(best.get('horizon_days'))}d",
                "best_mean_ic": safe_float(best.get("mean_ic")),
                "worst_horizon": f"{safe_int(worst.get('horizon_days'))}d",
                "worst_mean_ic": safe_float(worst.get("mean_ic")),
                "sample_windows": safe_int(best.get("n_obs")),
                "recommended_action": plain_status(action),
                "live_vs_backtest_status": plain_status(drift_status),
                "failure_mode": (
                    shorten(action_row.iloc[0].get("reason"), 180)
                    if not action_row.empty
                    else "No explicit downgrade row, but still needs live forward proof."
                ),
                "source_files": "signal_decay_analysis.csv; signal_downgrade_queue.csv; signal_live_vs_backtest_drift.csv",
            })

    p1 = safe_int(state.get("p1_signal_repairs"), 0)
    p2 = safe_int(state.get("p2_signal_reviews"), 0)
    live = safe_int(state.get("live_ic_observations"), 0)
    usable = int(decay.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin(["USABLE", "STRONG"]).sum()) if not decay.empty else 0
    total_decay = len(decay)
    score = max(20.0, 68.0 - p1 * 5.0 - p2 * 2.0 - (12.0 if live == 0 else 0.0) + min(8.0, usable * 1.5))
    status = status_from_score(score)
    summary = {
        "score": clamp_score(score),
        "status": status,
        "can_use_for_sizing": "No. Use only as a research weight haircut." if p1 else "Only with live IC monitoring.",
        "biggest_gap": f"{p1} signals need blocking/repair and {p2} need down-weight review; live IC observations are {live}.",
        "next_action": "Block negative or data-gap signals, keep collecting live IC, and require horizon-specific signal weights.",
        "p1_signal_repairs": p1,
        "p2_signal_reviews": p2,
        "usable_decay_rows": usable,
        "decay_rows": total_decay,
    }

    out_queue: list[dict[str, Any]] = []
    if not queue.empty:
        for _, row in queue.head(8).iterrows():
            out_queue.append({
                "priority": as_text(row.get("queue_priority"), "P1"),
                "module": "Signal IC / Decay / Failure Lab",
                "item": as_text(row.get("signal"), "signal"),
                "why_it_matters": f"{plain_status(row.get('recommended_signal_action'))}: {shorten(row.get('reason'), 180)}",
                "next_action": shorten(row.get("required_next_action"), 240),
                "source_files": "signal_downgrade_queue.csv",
            })
    return pd.DataFrame(rows), summary, out_queue


def portfolio_confidence(row: pd.Series) -> tuple[str, float, str]:
    risk = as_text(row.get("final_risk_action")).upper()
    signal = as_text(row.get("signal_validation_action")).upper()
    execution = as_text(row.get("execution_status")).upper()
    constraints = as_text(row.get("binding_constraints"))
    score = 100.0
    if "REDUCE" in risk:
        score -= 35
    elif "SIZE" in risk:
        score -= 20
    if "BLOCK" in signal:
        score -= 30
    elif "DOWN" in signal:
        score -= 15
    if "DATA" in execution or execution in {"NAN", ""}:
        score -= 12
    if "risk" in constraints.lower():
        score -= 10
    score = float(np.clip(score, 0, 100))
    if score >= 75:
        label = "Candidate after gates"
    elif score >= 50:
        label = "Tiny research only"
    else:
        label = "No new exposure"
    why = []
    if "REDUCE" in risk:
        why.append("risk says reduce only")
    elif "SIZE" in risk:
        why.append("risk says smaller size")
    if "BLOCK" in signal:
        why.append("signal is blocked")
    if "DATA" in execution or execution in {"NAN", ""}:
        why.append("execution data is weak or missing")
    if not why:
        why.append("all major gates are not blocking")
    return label, round(score, 1), "; ".join(why)


def build_portfolio_optimizer_v2() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    bridge = read_csv_safe(ROOT / "institutional_optimizer_bridge.csv")
    constraints = read_csv_safe(ROOT / "institutional_optimizer_constraint_audit.csv")
    active = read_csv_safe(ROOT / "institutional_optimizer_active_risk_budget.csv")
    state = read_json_safe(ROOT / "institutional_optimizer_state.json", {})

    rows: list[dict[str, Any]] = []
    if not bridge.empty:
        for _, row in bridge.iterrows():
            label, confidence, why = portfolio_confidence(row)
            final_weight = safe_float(row.get("final_optimizer_weight_pct"), 0.0)
            robust_weight = final_weight
            if confidence < 50:
                robust_weight = 0.0
            elif confidence < 75:
                robust_weight = min(final_weight, 0.25)
            rows.append({
                "ticker": clean_ticker(row.get("ticker")),
                "sector": as_text(row.get("sector"), "Unknown"),
                "sleeve": as_text(row.get("sleeve"), "Research"),
                "current_weight_pct": safe_float(row.get("current_weight_pct"), 0.0),
                "math_optimizer_wants_pct": safe_float(row.get("math_optimizer_weight_pct"), 0.0),
                "risk_allows_pct": safe_float(row.get("risk_gated_target_pct"), 0.0),
                "final_weight_pct": final_weight,
                "robust_weight_v2_pct": round(robust_weight, 4),
                "portfolio_v2_decision": label,
                "confidence_0_100": confidence,
                "why": why,
                "what_would_unlock": shorten(plain_status(row.get("why_not_more")), 240),
                "source_files": "institutional_optimizer_bridge.csv; institutional_optimizer_constraint_audit.csv",
            })

    gross = safe_float(state.get("final_gross_pct"), 0.0)
    score = clamp_score(state.get("institutional_optimizer_score", 0.0))
    risk_dominates = safe_int(state.get("risk_gate_dominates_count"), 0)
    constraint_flags = safe_int(state.get("constraint_flags"), 0)
    no_new = int(pd.Series([r.get("portfolio_v2_decision") for r in rows]).astype(str).str.contains("No new").sum()) if rows else 0
    summary = {
        "score": score,
        "status": status_from_score(score),
        "can_use_for_sizing": "Only for research/paper planning; risk gate dominates." if risk_dominates else "Research usable with gates.",
        "biggest_gap": f"Risk gate dominates {risk_dominates} tickers; {constraint_flags} optimizer constraints still flag review.",
        "next_action": "Move from raw ranking to robust weights: risk cap, signal confidence, sector budget, correlation, TCA, and turnover all bind before sizing.",
        "final_gross_pct": gross,
        "no_new_exposure_count": no_new,
    }

    out_queue: list[dict[str, Any]] = []
    if not constraints.empty:
        weak = constraints[constraints.get("status", pd.Series(dtype=str)).astype(str).str.upper().str.contains("REVIEW|SIZE|BLOCK|DATA", regex=True, na=False)]
        for _, row in weak.head(5).iterrows():
            out_queue.append({
                "priority": "P1" if "SIZE" in as_text(row.get("status")).upper() else "P2",
                "module": "Portfolio Optimizer 2.0",
                "item": as_text(row.get("constraint"), "optimizer constraint"),
                "why_it_matters": shorten(row.get("note"), 220),
                "next_action": f"Reduce or cap this exposure before increasing gross. Current {row.get('current_value')} vs limit {row.get('limit_value')}.",
                "source_files": as_text(row.get("source_file"), "institutional_optimizer_constraint_audit.csv"),
            })
    return pd.DataFrame(rows), summary, out_queue


def build_execution_desk() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    model = read_csv_safe(ROOT / "execution_cost_model.csv")
    board = read_csv_safe(ROOT / "execution_tca_decision_board.csv")
    scenarios = read_csv_safe(ROOT / "execution_cost_stress_scenarios.csv")
    state = read_json_safe(ROOT / "execution_cost_model_state.json", {})

    rows: list[dict[str, Any]] = []
    if not model.empty:
        extra = pd.DataFrame()
        if not board.empty and "ticker" in board.columns:
            extra = board.copy()
            extra["ticker"] = extra["ticker"].apply(clean_ticker)
        for _, row in model.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            b = extra[extra["ticker"] == ticker].head(1) if not extra.empty else pd.DataFrame()
            verdict = as_text(b.iloc[0].get("execution_verdict"), as_text(row.get("execution_cost_status"), "Review")) if not b.empty else as_text(row.get("execution_cost_status"), "Review")
            status_upper = verdict.upper() + " " + as_text(row.get("execution_cost_status")).upper()
            if "RISK" in status_upper or "BLOCK" in status_upper:
                permission = "No new exposure"
            elif "SIZE" in status_upper or "REVIEW" in status_upper:
                permission = "Manual quote and smaller paper size only"
            else:
                permission = "Research usable after manual quote"
            rows.append({
                "ticker": ticker,
                "direction": as_text(row.get("direction"), "FLAT"),
                "execution_permission": permission,
                "execution_status": plain_status(verdict),
                "trade_notional_dollars": round(safe_float(row.get("trade_notional_dollars"), 0.0), 2),
                "participation_rate_pct": round(safe_float(row.get("participation_rate_pct"), np.nan), 4),
                "spread_bps": round(safe_float(row.get("spread_bps"), np.nan), 2),
                "base_cost_bps": round(safe_float(row.get("base_cost_bps"), np.nan), 2),
                "stress_cost_bps": round(safe_float(row.get("stress_cost_bps"), np.nan), 2),
                "expected_fill_rate_pct": round(safe_float(row.get("expected_fill_rate_pct"), np.nan), 1),
                "liquidity_read": as_text(row.get("liquidity_label"), "No liquidity label"),
                "monitor_status": plain_status(row.get("monitor_severity")),
                "what_to_do": as_text(row.get("execution_instruction"), "Check live bid/ask spread and volume before any paper assumption."),
                "source_files": "execution_cost_model.csv; execution_tca_decision_board.csv",
            })

    score = clamp_score(state.get("execution_cost_model_score", 0.0))
    data_gap = safe_int(state.get("data_gap_rows"), 0)
    review_rows = safe_int(state.get("review_or_size_down_rows"), 0)
    weighted_stress = safe_float(state.get("weighted_stress_cost_bps"), np.nan)
    summary = {
        "score": score,
        "status": status_from_score(score),
        "can_use_for_sizing": "No automatic sizing. Manual quote required." if review_rows or data_gap else "Research usable after quote check.",
        "biggest_gap": f"{review_rows} tickers need review/size-down; weighted stress cost is {weighted_stress:.2f} bps.",
        "next_action": "Add real bid/ask snapshots, participation limits, failed-fill assumptions, and open/close auction rules.",
        "review_or_size_down_rows": review_rows,
        "data_gap_rows": data_gap,
    }

    out_queue: list[dict[str, Any]] = []
    if not board.empty:
        bad = board[board.get("execution_verdict", pd.Series(dtype=str)).astype(str).str.upper().str.contains("RISK|BLOCK|DATA|SIZE", regex=True, na=False)]
        for _, row in bad.head(8).iterrows():
            out_queue.append({
                "priority": "P1",
                "module": "Execution Cost / Liquidity Desk",
                "item": clean_ticker(row.get("ticker")),
                "why_it_matters": shorten(row.get("plain_reason"), 220),
                "next_action": shorten(row.get("next_manual_check"), 240),
                "source_files": as_text(row.get("source_files"), "execution_tca_decision_board.csv"),
            })
    return pd.DataFrame(rows), summary, out_queue


def build_news_causal_system() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    edges = read_csv_safe(ROOT / "event_causal_chain_edges.csv")
    queue = read_csv_safe(ROOT / "event_causal_validation_queue.csv")
    ranking = read_csv_safe(ROOT / "event_readthrough_target_ranking.csv")
    time_ledger = read_csv_safe(ROOT / "event_time_truth_ledger.csv")
    chain_state = read_json_safe(ROOT / "event_causal_chain_state.json", {})
    rel_state = read_json_safe(ROOT / "event_signal_reliability_state.json", {})

    rows: list[dict[str, Any]] = []
    if not ranking.empty:
        rank = ranking.copy()
        rank["target_ticker"] = rank["target_ticker"].apply(clean_ticker)
        validation_counts = pd.DataFrame()
        if not queue.empty:
            q = queue.copy()
            q["target_ticker"] = q["target_ticker"].apply(clean_ticker)
            validation_counts = q.groupby("target_ticker").size().reset_index(name="open_proof_items")
        edge_counts = pd.DataFrame()
        if not edges.empty:
            e = edges.copy()
            e["target_ticker"] = e["target_ticker"].apply(clean_ticker)
            edge_counts = e.groupby("target_ticker").agg(
                causal_edges=("edge_id", "count"),
                avg_confidence=("causal_confidence_score", "mean"),
                contradicted_edges=("causal_chain_status", lambda s: s.astype(str).str.contains("CONTRADICTED|PRICE_DISAGREES", regex=True).sum()),
            ).reset_index()
        if not validation_counts.empty:
            rank = rank.merge(validation_counts, on="target_ticker", how="left")
        if not edge_counts.empty:
            rank = rank.merge(edge_counts, on="target_ticker", how="left")
        for _, row in rank.head(30).iterrows():
            open_proof = safe_int(row.get("open_proof_items"), 0)
            contrad = safe_int(row.get("contradicted_edges"), 0)
            confidence = safe_float(row.get("avg_confidence"), safe_float(row.get("best_event_score"), np.nan))
            if contrad:
                permission = "Contradicted; review manually"
            elif open_proof:
                permission = "Research only until proof is closed"
            elif confidence >= 80:
                permission = "Watch-list research only"
            else:
                permission = "Context only"
            rows.append({
                "ticker": clean_ticker(row.get("target_ticker")),
                "tone": plain_status(row.get("top_tone")),
                "best_event_score": safe_float(row.get("best_event_score"), np.nan),
                "causal_confidence_0_100": round(confidence, 1) if np.isfinite(confidence) else np.nan,
                "open_proof_items": open_proof,
                "contradicted_edges": contrad,
                "causal_permission": permission,
                "industry_chain_read": plain_status(row.get("top_target_role")),
                "route_before_risk": plain_status(row.get("directional_route")),
                "risk_gate": plain_status(row.get("final_risk_action")),
                "top_headline": shorten(row.get("top_headline"), 180),
                "proof_needed": shorten(row.get("proof_required"), 240),
                "source_files": "event_readthrough_target_ranking.csv; event_causal_chain_edges.csv; event_causal_validation_queue.csv",
            })

    avg_conf = safe_float(chain_state.get("average_causal_confidence"), 0.0)
    validation_rows = safe_int(chain_state.get("validation_queue_rows"), len(queue) if not queue.empty else 0)
    contradicted = safe_int(chain_state.get("contradicted_edge_count"), 0)
    reliable = safe_int(rel_state.get("reliable_bucket_count"), 0)
    institutional = bool(rel_state.get("can_support_institutional_backtest", False))
    score = avg_conf
    if validation_rows:
        score = min(score, 50.0)
    if contradicted:
        score = min(score, 45.0)
    if not institutional:
        score = min(score, 48.0)
    if reliable == 0:
        score = min(score, 42.0)
    summary = {
        "score": clamp_score(score),
        "status": status_from_score(score),
        "can_use_for_sizing": "No. It can explain hypotheses, but it cannot size trades yet.",
        "biggest_gap": f"{validation_rows} causal proof rows are open; {contradicted} links are contradicted; reliable event buckets: {reliable}.",
        "next_action": "Require source time, model-read time, 1d/5d price reaction, chain role, and risk gate before any event idea can move forward.",
        "validation_rows": validation_rows,
        "contradicted_edge_count": contradicted,
        "reliable_bucket_count": reliable,
    }

    out_queue: list[dict[str, Any]] = []
    if not queue.empty:
        for _, row in queue.head(10).iterrows():
            out_queue.append({
                "priority": "P1" if "P1" in as_text(row.get("priority")) else "P2",
                "module": "News-to-Industry Causal Proof System",
                "item": clean_ticker(row.get("target_ticker")),
                "why_it_matters": f"{shorten(row.get('headline'), 140)} | {shorten(row.get('validation_note'), 120)}",
                "next_action": shorten(row.get("required_next_action"), 260),
                "source_files": "event_causal_validation_queue.csv",
            })
    return pd.DataFrame(rows), summary, out_queue


def priority_rank(value: Any) -> int:
    text = as_text(value).upper()
    if "P0" in text:
        return 0
    if "P1" in text:
        return 1
    if "P2" in text:
        return 2
    if "P3" in text:
        return 3
    return 4


def main() -> None:
    backtest, backtest_summary, q_backtest = build_backtest_center()
    signal, signal_summary, q_signal = build_signal_lab()
    portfolio, portfolio_summary, q_portfolio = build_portfolio_optimizer_v2()
    execution, execution_summary, q_execution = build_execution_desk()
    news, news_summary, q_news = build_news_causal_system()

    module_rows = [
        module_card(
            "Backtest Credibility Center",
            backtest_summary["score"],
            backtest_summary["status"],
            backtest_summary["can_use_for_sizing"],
            backtest_summary["biggest_gap"],
            backtest_summary["next_action"],
            "backtest_credibility_state.json; backtest_credibility_scorecard.csv",
        ),
        module_card(
            "Signal IC / Decay / Failure Lab",
            signal_summary["score"],
            signal_summary["status"],
            signal_summary["can_use_for_sizing"],
            signal_summary["biggest_gap"],
            signal_summary["next_action"],
            "signal_validation_state.json; signal_decay_analysis.csv; signal_downgrade_queue.csv",
        ),
        module_card(
            "Portfolio Optimizer 2.0",
            portfolio_summary["score"],
            portfolio_summary["status"],
            portfolio_summary["can_use_for_sizing"],
            portfolio_summary["biggest_gap"],
            portfolio_summary["next_action"],
            "institutional_optimizer_state.json; institutional_optimizer_bridge.csv",
        ),
        module_card(
            "Execution Cost / Liquidity Desk",
            execution_summary["score"],
            execution_summary["status"],
            execution_summary["can_use_for_sizing"],
            execution_summary["biggest_gap"],
            execution_summary["next_action"],
            "execution_cost_model_state.json; execution_cost_model.csv; execution_tca_decision_board.csv",
        ),
        module_card(
            "News-to-Industry Causal Proof System",
            news_summary["score"],
            news_summary["status"],
            news_summary["can_use_for_sizing"],
            news_summary["biggest_gap"],
            news_summary["next_action"],
            "event_causal_chain_state.json; event_causal_validation_queue.csv; event_readthrough_target_ranking.csv",
        ),
    ]
    modules = pd.DataFrame(module_rows)
    queue = pd.DataFrame(q_backtest + q_signal + q_portfolio + q_execution + q_news)
    if not queue.empty:
        queue["_rank"] = queue["priority"].apply(priority_rank)
        queue = queue.sort_values(["_rank", "module", "item"]).drop(columns=["_rank"]).reset_index(drop=True)

    for df, path in [
        (modules, OUT_SCORECARD),
        (backtest, OUT_BACKTEST),
        (signal, OUT_SIGNAL),
        (portfolio, OUT_PORTFOLIO),
        (execution, OUT_EXECUTION),
        (news, OUT_NEWS),
        (queue, OUT_QUEUE),
    ]:
        df.to_csv(path, index=False)

    overall_score = float(modules["score_0_100"].mean()) if not modules.empty else 0.0
    sizing_ready = int(modules["can_use_for_sizing"].astype(str).str.contains("^Yes", regex=True).sum()) if not modules.empty else 0
    repair_modules = int(modules["status"].astype(str).str.contains("Repair|Not reliable|Prototype|BLOCK", regex=True).sum()) if not modules.empty else 0
    state = {
        "date": today_str(),
        "status": "DEPTH5_WORKBENCH_ACTIVE",
        "overall_score_0_100": round(overall_score, 1),
        "overall_status": status_from_score(overall_score),
        "sizing_ready_modules": sizing_ready,
        "repair_or_prototype_modules": repair_modules,
        "priority_queue_rows": len(queue),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "plain_answer": "The five deep modules are active, but they still support research and paper planning only. They do not clear live trading or automatic sizing.",
        "module_scores": {row["module"]: row["score_0_100"] for _, row in modules.iterrows()},
    }
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Five Module Scorecard\n\n" + df_to_markdown(modules),
        "## Priority Queue\n\n" + df_to_markdown(queue.head(30)),
        "## Backtest Credibility Center\n\n" + df_to_markdown(backtest.head(20)),
        "## Signal IC / Decay / Failure Lab\n\n" + df_to_markdown(signal.head(20)),
        "## Portfolio Optimizer 2.0\n\n" + df_to_markdown(portfolio.head(20)),
        "## Execution Cost / Liquidity Desk\n\n" + df_to_markdown(execution.head(20)),
        "## News-to-Industry Causal Proof System\n\n" + df_to_markdown(news.head(20)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 194 - Institutional Depth 5 Workbench", sections)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Overall score: {state['overall_score_0_100']}/100 ({state['overall_status']})")
    print(f"[OK] Priority queue rows: {len(queue)}")
    print("[OK] Research-only: True")


if __name__ == "__main__":
    main()
