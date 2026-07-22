#!/usr/bin/env python3
"""
Canyon v9 Step 157 - Institutional Portfolio Optimizer Bridge.

Research-only. No broker connection. No live orders.

Step63 produces mathematical optimizer weights. Step123 produces risk-gated
target weights. Step157 bridges the two: it shows what the optimizer wants,
what the risk budget allows, which constraints bind, and the final research
weight after risk, sector, correlation, signal-validation, and execution gates.

Outputs:
  institutional_optimizer_bridge.csv
  institutional_optimizer_constraint_audit.csv
  institutional_optimizer_sector_allocations.csv
  institutional_optimizer_why_not_more.csv
  institutional_optimizer_active_risk_budget.csv
  institutional_optimizer_constraint_ladder.csv
  institutional_optimizer_state.json
  institutional_optimizer_report.md
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


OUT_BRIDGE = ROOT / "institutional_optimizer_bridge.csv"
OUT_AUDIT = ROOT / "institutional_optimizer_constraint_audit.csv"
OUT_SECTOR = ROOT / "institutional_optimizer_sector_allocations.csv"
OUT_WHY_NOT_MORE = ROOT / "institutional_optimizer_why_not_more.csv"
OUT_ACTIVE_RISK = ROOT / "institutional_optimizer_active_risk_budget.csv"
OUT_LADDER = ROOT / "institutional_optimizer_constraint_ladder.csv"
OUT_STATE = ROOT / "institutional_optimizer_state.json"
OUT_REPORT = ROOT / "institutional_optimizer_report.md"

SINGLE_NAME_CAP = 0.04
TECH_SECTOR_CAP = 0.28
DEFAULT_SECTOR_CAP = 0.25
TURNOVER_BUDGET = 0.35
SPY_BETA_LIMIT = 1.10
AVG_CORR_REVIEW = 0.35
AVG_CORR_SIZE_DOWN = 0.50
TCA_REVIEW_BPS = 25.0
TCA_SIZE_DOWN_BPS = 45.0
TCA_BLOCK_BPS = 80.0


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


def status_from_score(score: float) -> str:
    if score >= 80:
        return "READY_WITH_GATES"
    if score >= 65:
        return "REVIEW_REQUIRED"
    if score >= 45:
        return "SIZE_DOWN_REQUIRED"
    return "BLOCK_NEW_RESEARCH"


def constraint_status(current: float, limit: float, direction: str = "max") -> str:
    if not np.isfinite(current):
        return "DATA_GAP"
    if direction == "min":
        if current >= limit:
            return "CLEAR"
        if current >= limit * 0.75:
            return "REVIEW"
        return "SIZE_DOWN"
    if current <= limit:
        return "CLEAR"
    if current <= limit * 1.25:
        return "REVIEW"
    return "SIZE_DOWN"


def load_corr_summary(tickers: list[str]) -> pd.DataFrame:
    corr = read_csv_safe(ROOT / "holdings_correlation_matrix.csv", index_col=0)
    tickers = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    if corr.empty or not tickers:
        return pd.DataFrame({"ticker": tickers, "avg_abs_corr_to_book": np.nan, "max_abs_corr_to_book": np.nan, "corr_status": "DATA_GAP"})
    corr.index = [clean_ticker(x) for x in corr.index]
    corr.columns = [clean_ticker(x) for x in corr.columns]
    rows = []
    for ticker in tickers:
        if ticker not in corr.index:
            rows.append({"ticker": ticker, "avg_abs_corr_to_book": np.nan, "max_abs_corr_to_book": np.nan, "corr_status": "DATA_GAP"})
            continue
        vals = pd.to_numeric(corr.loc[ticker], errors="coerce")
        vals = vals.drop(labels=[ticker], errors="ignore").dropna().abs()
        avg_corr = float(vals.mean()) if not vals.empty else np.nan
        max_corr = float(vals.max()) if not vals.empty else np.nan
        if not np.isfinite(avg_corr):
            status = "DATA_GAP"
        elif avg_corr >= AVG_CORR_SIZE_DOWN:
            status = "SIZE_DOWN"
        elif avg_corr >= AVG_CORR_REVIEW:
            status = "REVIEW"
        else:
            status = "CLEAR"
        rows.append({
            "ticker": ticker,
            "avg_abs_corr_to_book": round(avg_corr, 4) if np.isfinite(avg_corr) else np.nan,
            "max_abs_corr_to_book": round(max_corr, 4) if np.isfinite(max_corr) else np.nan,
            "corr_status": status,
        })
    return pd.DataFrame(rows)


def load_math_optimizer_weights() -> pd.DataFrame:
    math_w = read_csv_safe(ROOT / "portfolio_optimized_weights.csv")
    if math_w.empty or "ticker" not in math_w.columns:
        return pd.DataFrame(columns=["ticker", "math_optimizer_weight"])
    math_w = math_w.copy()
    math_w["ticker"] = math_w["ticker"].apply(clean_ticker)
    preferred = next((c for c in ["turnover_aware", "risk_parity", "max_sharpe", "inv_vol"] if c in math_w.columns), None)
    if preferred is None:
        return pd.DataFrame(columns=["ticker", "math_optimizer_weight"])
    out = math_w[["ticker", preferred]].rename(columns={preferred: "math_optimizer_weight"})
    out["math_optimizer_source"] = preferred
    out["math_optimizer_weight"] = pd.to_numeric(out["math_optimizer_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(out["math_optimizer_weight"].sum())
    if total > 0:
        out["math_optimizer_weight"] = out["math_optimizer_weight"] / total
    return out


def load_signal_validation() -> pd.DataFrame:
    queue = read_csv_safe(ROOT / "signal_downgrade_queue.csv")
    if queue.empty or "signal" not in queue.columns:
        return pd.DataFrame(columns=["validation_signal", "recommended_signal_action", "queue_priority"])
    keep = [c for c in ["signal", "recommended_signal_action", "queue_priority", "required_next_action"] if c in queue.columns]
    out = queue[keep].copy().rename(columns={"signal": "validation_signal"})
    out["validation_signal"] = out["validation_signal"].astype(str)
    return out


def load_signal_policy() -> pd.DataFrame:
    policy = read_csv_safe(ROOT / "signal_horizon_regime_policy.csv")
    if policy.empty or "signal" not in policy.columns:
        return pd.DataFrame(columns=["validation_signal", "allowed_use", "weight_multiplier"])
    keep = [c for c in ["signal", "allowed_use", "weight_multiplier", "allowed_market_regimes", "why"] if c in policy.columns]
    out = policy[keep].copy().rename(columns={
        "signal": "validation_signal",
        "why": "signal_policy_why",
    })
    out["validation_signal"] = out["validation_signal"].astype(str)
    out["weight_multiplier"] = pd.to_numeric(out.get("weight_multiplier", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    return out


def load_subsector_overlay() -> pd.DataFrame:
    sub = read_csv_safe(ROOT / "subsector_ticker_cycle_map.csv")
    if sub.empty or "ticker" not in sub.columns:
        return pd.DataFrame(columns=["ticker", "subsector", "subsector_cycle_phase"])
    sub = sub.copy()
    sub["ticker"] = sub["ticker"].apply(clean_ticker)
    keep = [c for c in [
        "ticker", "subsector", "subsector_cycle_phase", "leadership_handoff_signal",
        "subsector_action_bias", "subsector_adjustment_label",
        "subsector_short_adjustment", "subsector_medium_adjustment", "subsector_long_adjustment",
    ] if c in sub.columns]
    return sub[keep].drop_duplicates(subset=["ticker"], keep="first")


def build_bridge() -> pd.DataFrame:
    target = read_csv_safe(ROOT / "institutional_target_weights.csv")
    if target.empty:
        target = read_csv_safe(ROOT / "daily_picks_filtered.csv")
        if target.empty:
            return pd.DataFrame()
        target = target.copy()
        target["target_weight"] = target.get("weight_pct", 0).apply(pct_to_weight) if "weight_pct" in target.columns else 1.0 / max(len(target), 1)
        target["target_weight_pct"] = target["target_weight"] * 100
        target["current_weight_pct"] = target.get("weight_pct", target["target_weight_pct"])
        target["target_status"] = "RAW_ALPHA_ONLY"
        target["final_risk_action"] = "MISSING_RISK_GATE"
        target["reason"] = "institutional_target_weights.csv missing; fallback to daily_picks_filtered.csv"

    work = target.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    work = work[work["ticker"] != ""].copy()
    for col in ["target_weight", "target_weight_pct", "current_weight_pct", "alpha_score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    daily = read_csv_safe(ROOT / "daily_picks_filtered.csv")
    if not daily.empty and "ticker" in daily.columns:
        daily = daily.copy()
        daily["ticker"] = daily["ticker"].apply(clean_ticker)
        keep = [c for c in ["ticker", "top_signal", "action", "score_trend", "rank_change"] if c in daily.columns]
        work = work.merge(daily[keep], on="ticker", how="left", suffixes=("", "_daily"))

    math_w = load_math_optimizer_weights()
    if not math_w.empty:
        work = work.merge(math_w, on="ticker", how="left")
    else:
        work["math_optimizer_weight"] = np.nan
        work["math_optimizer_source"] = "missing"

    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    if not tca.empty and "ticker" in tca.columns:
        tca = tca.copy()
        tca["ticker"] = tca["ticker"].apply(clean_ticker)
        keep = [c for c in ["ticker", "total_tca_cost_bps", "execution_status", "avg_20d_dollar_volume", "participation_rate_pct"] if c in tca.columns]
        work = work.merge(tca[keep], on="ticker", how="left", suffixes=("", "_tca"))

    sigval = load_signal_validation()
    work["validation_signal"] = work.get("top_signal", "").fillna("").astype(str).str.lower().map(TOP_SIGNAL_TO_VALIDATION_SIGNAL).fillna("")
    if not sigval.empty:
        work = work.merge(sigval, on="validation_signal", how="left")
    else:
        work["recommended_signal_action"] = ""
        work["queue_priority"] = ""
        work["required_next_action"] = ""

    sigpolicy = load_signal_policy()
    if not sigpolicy.empty:
        work = work.merge(sigpolicy, on="validation_signal", how="left")
    else:
        work["allowed_use"] = ""
        work["weight_multiplier"] = 1.0
        work["allowed_market_regimes"] = ""
        work["signal_policy_why"] = ""

    subsector = load_subsector_overlay()
    if not subsector.empty:
        work = work.merge(subsector, on="ticker", how="left")
    else:
        work["subsector"] = ""
        work["subsector_cycle_phase"] = ""
        work["leadership_handoff_signal"] = ""
        work["subsector_action_bias"] = ""

    corr = load_corr_summary(work["ticker"].tolist())
    work = work.merge(corr, on="ticker", how="left")

    risk_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    allowed_gross = safe_float(risk_state.get("master_exposure_multiplier"), 0.70)
    master_action = str(risk_state.get("master_risk_action", "REVIEW"))
    if allowed_gross <= 0 or not np.isfinite(allowed_gross):
        allowed_gross = 0.70

    rows = []
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        risk_target = pct_to_weight(row.get("target_weight", row.get("target_weight_pct", 0.0)))
        current_weight = pct_to_weight(row.get("current_weight_pct", 0.0))
        math_weight = safe_float(row.get("math_optimizer_weight"), np.nan)
        math_scaled = math_weight * allowed_gross if np.isfinite(math_weight) else risk_target
        alpha_request = max(current_weight, risk_target, math_scaled if np.isfinite(math_scaled) else 0.0)
        if not np.isfinite(math_scaled) or math_scaled <= 1e-8:
            # A zero historical optimizer weight is not a hard veto. The risk
            # layer already made the sizing conservative; use the optimizer as
            # a down-weighting reference unless a separate risk/signal gate blocks.
            math_cap = risk_target
        else:
            math_cap = max(math_scaled, risk_target * 0.50)
        proposed = min(risk_target, math_cap, SINGLE_NAME_CAP)
        reasons = []
        binding = []

        if risk_target < math_scaled - 1e-8:
            binding.append("risk budget")
            reasons.append("risk-gated target below math optimizer")
        elif np.isfinite(math_scaled) and math_scaled < risk_target * 0.50:
            binding.append("math optimizer low")
            reasons.append("math optimizer suggests below half of risk target")
        if proposed >= SINGLE_NAME_CAP - 1e-8:
            binding.append("single-name cap")
        risk_action = str(row.get("final_risk_action", "")).upper()
        target_status = str(row.get("target_status", "")).upper()
        if risk_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"} or target_status in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            proposed = min(proposed, risk_target)
            binding.append("hard risk gate")
            reasons.append(f"risk action {risk_action or target_status}")
        elif risk_action in {"SIZE_DOWN", "MISSING_DATA_REVIEW"} or target_status in {"SIZE_DOWN", "MISSING_DATA_REVIEW"}:
            proposed *= 0.90
            binding.append("risk size-down")

        signal_action = str(row.get("recommended_signal_action", "")).upper()
        signal_policy_multiplier = safe_float(row.get("weight_multiplier"), 1.0)
        if np.isfinite(signal_policy_multiplier) and signal_policy_multiplier < 0.999:
            proposed *= max(0.0, min(1.0, signal_policy_multiplier))
            binding.append("signal policy")
            reasons.append(f"signal policy multiplier {signal_policy_multiplier:.2f}")
        if signal_action == "BLOCK_SIGNAL":
            proposed *= 0.50
            binding.append("signal repair")
            reasons.append("top signal maps to blocked validation signal")
        elif signal_action in {"DOWNWEIGHT", "REVIEW_SAMPLE_SIZE"}:
            proposed *= 0.75
            binding.append("signal validation")
            reasons.append(f"signal validation {signal_action}")
        elif signal_action == "USE_ONLY_AT_SHORT_HORIZON":
            proposed *= 0.85
            binding.append("horizon decay")

        subsector_phase = str(row.get("subsector_cycle_phase", "")).lower()
        if "late-cycle" in subsector_phase or "chase risk" in subsector_phase:
            proposed *= 0.75
            binding.append("subsector late-cycle")
            reasons.append("subsector is a hot/late leader; do not chase size")
        elif "downcycle" in subsector_phase or "laggard" in subsector_phase:
            proposed *= 0.80
            binding.append("subsector downcycle")
            reasons.append("subsector relative trend is weak")

        tca_bps = safe_float(row.get("total_tca_cost_bps"), np.nan)
        if np.isfinite(tca_bps):
            if tca_bps >= TCA_BLOCK_BPS:
                proposed = 0.0
                binding.append("execution cost block")
            elif tca_bps >= TCA_SIZE_DOWN_BPS:
                proposed *= 0.50
                binding.append("execution cost")
            elif tca_bps >= TCA_REVIEW_BPS:
                proposed *= 0.85
                binding.append("execution review")

        corr_status = str(row.get("corr_status", "DATA_GAP")).upper()
        if corr_status == "SIZE_DOWN":
            proposed *= 0.75
            binding.append("correlation cluster")
        elif corr_status == "REVIEW":
            proposed *= 0.90
            binding.append("correlation review")

        proposed = max(0.0, min(proposed, risk_target, SINGLE_NAME_CAP))
        if proposed <= 1e-8:
            final_status = "BLOCK_NEW"
        elif proposed < risk_target * 0.65:
            final_status = "SIZE_DOWN"
        elif proposed < risk_target * 0.95:
            final_status = "REVIEW"
        else:
            final_status = "CLEAR"

        if not reasons:
            reasons.append("optimizer within current risk and execution gates")
        if not binding:
            binding.append("none")
        max_feasible = max(0.0, min(risk_target, SINGLE_NAME_CAP))
        if signal_action == "BLOCK_SIGNAL":
            max_feasible = min(max_feasible, risk_target * 0.50)
        if "late-cycle" in subsector_phase or "chase risk" in subsector_phase:
            max_feasible = min(max_feasible, risk_target * 0.75)
        if np.isfinite(tca_bps) and tca_bps >= TCA_BLOCK_BPS:
            max_feasible = 0.0
        why_not_more = "; ".join(dict.fromkeys(reasons))
        if proposed >= max_feasible - 1e-8 and max_feasible < alpha_request - 1e-8:
            why_not_more = f"max feasible reached: {why_not_more}"

        rows.append({
            "ticker": ticker,
            "sector": row.get("sector", "Unknown"),
            "subsector": row.get("subsector", ""),
            "subsector_cycle_phase": row.get("subsector_cycle_phase", ""),
            "leadership_handoff_signal": row.get("leadership_handoff_signal", ""),
            "sleeve": row.get("sleeve", "Unassigned"),
            "top_signal": row.get("top_signal", ""),
            "validation_signal": row.get("validation_signal", ""),
            "signal_validation_action": row.get("recommended_signal_action", ""),
            "signal_policy_allowed_use": row.get("allowed_use", ""),
            "signal_policy_weight_multiplier": round(signal_policy_multiplier, 3) if np.isfinite(signal_policy_multiplier) else np.nan,
            "current_weight_pct": round(current_weight * 100, 4),
            "alpha_requested_weight_pct": round(alpha_request * 100, 4) if np.isfinite(alpha_request) else np.nan,
            "math_optimizer_weight_pct": round(math_scaled * 100, 4) if np.isfinite(math_scaled) else np.nan,
            "risk_gated_target_pct": round(risk_target * 100, 4),
            "max_feasible_weight_pct": round(max_feasible * 100, 4),
            "final_optimizer_weight_pct": round(proposed * 100, 4),
            "final_optimizer_weight": proposed,
            "optimizer_gap_vs_math_pct": round((math_scaled - proposed) * 100, 4) if np.isfinite(math_scaled) else np.nan,
            "optimizer_gap_vs_risk_target_pct": round((risk_target - proposed) * 100, 4),
            "avg_abs_corr_to_book": row.get("avg_abs_corr_to_book"),
            "max_abs_corr_to_book": row.get("max_abs_corr_to_book"),
            "corr_status": row.get("corr_status", "DATA_GAP"),
            "total_tca_cost_bps": tca_bps,
            "execution_status": row.get("execution_status", ""),
            "final_risk_action": row.get("final_risk_action", ""),
            "target_status": row.get("target_status", ""),
            "final_optimizer_status": final_status,
            "binding_constraints": "; ".join(dict.fromkeys(binding)),
            "reason": "; ".join(dict.fromkeys(reasons)),
            "why_not_more": why_not_more,
            "master_risk_action": master_action,
            "math_optimizer_source": row.get("math_optimizer_source", "missing"),
            "source_file": "portfolio_optimized_weights.csv / institutional_target_weights.csv / signal_downgrade_queue.csv / institutional_tca_cost_estimates.csv / holdings_correlation_matrix.csv",
        })

    bridge = pd.DataFrame(rows)
    if bridge.empty:
        return bridge

    # Preserve the risk-down cash stance. Do not force scale-up to fill the book.
    return bridge.sort_values(["final_optimizer_status", "final_optimizer_weight_pct"], ascending=[True, False]).reset_index(drop=True)


def build_sector_allocations(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    sector = bridge.groupby("sector", dropna=False)["final_optimizer_weight"].sum().reset_index()
    sector["sector"] = sector["sector"].fillna("Unknown").astype(str)
    sector["final_sector_weight_pct"] = sector["final_optimizer_weight"] * 100
    sector["sector_cap_pct"] = np.where(sector["sector"].eq("Technology"), TECH_SECTOR_CAP * 100, DEFAULT_SECTOR_CAP * 100)
    sector["sector_status"] = np.where(sector["final_sector_weight_pct"] <= sector["sector_cap_pct"] + 1e-9, "CLEAR", "SIZE_DOWN")
    sector["source_file"] = "institutional_optimizer_bridge.csv"
    return sector.drop(columns=["final_optimizer_weight"]).sort_values("final_sector_weight_pct", ascending=False).reset_index(drop=True)


def build_why_not_more(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    rows = []
    for _, row in bridge.iterrows():
        requested = safe_float(row.get("alpha_requested_weight_pct"), safe_float(row.get("current_weight_pct"), 0.0))
        final = safe_float(row.get("final_optimizer_weight_pct"), 0.0)
        max_feasible = safe_float(row.get("max_feasible_weight_pct"), final)
        gap = max(0.0, requested - final)
        if gap <= 0.05:
            primary = "No major gap"
        else:
            constraints = str(row.get("binding_constraints", "")).lower()
            if "risk" in constraints:
                primary = "Risk budget"
            elif "signal" in constraints or "horizon" in constraints:
                primary = "Signal proof"
            elif "late-cycle" in constraints or "subsector" in constraints:
                primary = "Subsector cycle"
            elif "execution" in constraints:
                primary = "Execution cost"
            elif "correlation" in constraints:
                primary = "Correlation"
            else:
                primary = "Optimizer/risk target"
        rows.append({
            "ticker": row.get("ticker"),
            "sector": row.get("sector"),
            "subsector": row.get("subsector", ""),
            "requested_weight_pct": round(requested, 4),
            "max_feasible_weight_pct": round(max_feasible, 4),
            "final_weight_pct": round(final, 4),
            "unfilled_request_pct": round(gap, 4),
            "primary_reason_not_more": primary,
            "binding_constraints": row.get("binding_constraints", ""),
            "why_not_more": row.get("why_not_more", row.get("reason", "")),
            "what_would_allow_more": (
                "Risk gate improves, event/source rows clear, signal policy allows the horizon, "
                "subsector heat cools, and execution/correlation constraints stay clear."
            ),
            "research_only": True,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("unfilled_request_pct", ascending=False).reset_index(drop=True)


def build_active_risk_budget(bridge: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    final_gross = float(bridge["final_optimizer_weight"].sum())
    cash = max(0.0, 1.0 - final_gross)
    rows = []

    def add(bucket: str, current: float, limit: float, status: str, note: str, source: str) -> None:
        rows.append({
            "budget_bucket": bucket,
            "current_pct": round(current * 100, 4) if np.isfinite(current) else np.nan,
            "limit_pct": round(limit * 100, 4) if np.isfinite(limit) else np.nan,
            "remaining_pct": round((limit - current) * 100, 4) if np.isfinite(current) and np.isfinite(limit) else np.nan,
            "status": status,
            "note": note,
            "source_file": source,
        })

    risk_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    allowed_gross = safe_float(risk_state.get("master_exposure_multiplier"), 0.70)
    add("Gross exposure", final_gross, allowed_gross, constraint_status(final_gross, allowed_gross), "Master risk state controls total gross.", "institutional_risk_gate_state.json")
    add("Cash reserve", cash, 0.20, constraint_status(cash, 0.20, direction="min"), "Risk-down states should keep dry powder.", "institutional_optimizer_bridge.csv")

    for sec_name, cap in [("Technology", TECH_SECTOR_CAP), ("Semiconductors", 0.10), ("Software / Cloud", 0.12), ("AI Infrastructure / Hardware", 0.10)]:
        if sec_name == "Technology":
            current = safe_float(sector.loc[sector["sector"].astype(str).eq(sec_name), "final_sector_weight_pct"].iloc[0] / 100.0) if not sector.empty and sector["sector"].astype(str).eq(sec_name).any() else 0.0
            source = "institutional_optimizer_sector_allocations.csv"
        else:
            current = float(bridge.loc[bridge.get("subsector", pd.Series(dtype=str)).astype(str).eq(sec_name), "final_optimizer_weight"].sum())
            source = "institutional_optimizer_bridge.csv / subsector_ticker_cycle_map.csv"
        add(sec_name, current, cap, constraint_status(current, cap), f"{sec_name} budget prevents hidden theme concentration.", source)

    factor = read_csv_safe(ROOT / "portfolio_beta_report.csv")
    if not factor.empty and {"factor", "portfolio_beta"}.issubset(factor.columns):
        for factor_name, limit in [("SPY_beta", SPY_BETA_LIMIT), ("semiconductor_beta_SMH", 0.65), ("QQQ_growth_beta", 1.00)]:
            mask = factor["factor"].astype(str).eq(factor_name)
            if mask.any():
                beta = abs(safe_float(factor.loc[mask, "portfolio_beta"].iloc[0], np.nan))
                add(factor_name, beta, limit, constraint_status(beta, limit), "Factor exposure budget is explicit; beta can cap allocation even when alpha is high.", "portfolio_beta_report.csv")

    late_cycle_weight = float(bridge.loc[bridge.get("subsector_cycle_phase", pd.Series(dtype=str)).astype(str).str.contains("late-cycle|chase risk", case=False, na=False), "final_optimizer_weight"].sum())
    add("Late-cycle leader sleeve", late_cycle_weight, 0.08, constraint_status(late_cycle_weight, 0.08), "Hot/late-cycle leaders should not consume the whole book.", "subsector_ticker_cycle_map.csv")

    p1_signal_weight = float(bridge.loc[bridge.get("signal_validation_action", pd.Series(dtype=str)).astype(str).eq("BLOCK_SIGNAL"), "final_optimizer_weight"].sum())
    add("Blocked-signal exposure", p1_signal_weight, 0.00, "REVIEW" if p1_signal_weight > 1e-8 else "CLEAR", "Blocked signal exposure should be zero unless manually justified.", "signal_horizon_regime_policy.csv")
    return pd.DataFrame(rows)


def build_constraint_ladder(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    rows = []
    for _, row in bridge.iterrows():
        ticker = row.get("ticker")
        current = safe_float(row.get("current_weight_pct"), 0.0)
        math_w = safe_float(row.get("math_optimizer_weight_pct"), np.nan)
        risk_target = safe_float(row.get("risk_gated_target_pct"), np.nan)
        feasible = safe_float(row.get("max_feasible_weight_pct"), np.nan)
        final = safe_float(row.get("final_optimizer_weight_pct"), np.nan)
        stages = [
            ("Current book", current, "Starting point from current research book."),
            ("Math optimizer", math_w, "What the raw optimizer wanted before gates."),
            ("Risk target", risk_target, "Risk-gated target from risk budget."),
            ("Max feasible", feasible, "Hard cap after single-name, signal, sector, event, and execution constraints."),
            ("Final research weight", final, "Final research-only optimizer bridge weight."),
        ]
        prev = np.nan
        for stage, weight, note in stages:
            rows.append({
                "ticker": ticker,
                "stage": stage,
                "weight_pct": round(weight, 4) if np.isfinite(weight) else np.nan,
                "delta_from_prior_pct": round(weight - prev, 4) if np.isfinite(weight) and np.isfinite(prev) else np.nan,
                "binding_constraints": row.get("binding_constraints", "") if stage in {"Max feasible", "Final research weight"} else "",
                "note": note,
                "research_only": True,
            })
            prev = weight
    return pd.DataFrame(rows)


def build_constraint_audit(bridge: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    risk_state = read_json_safe(ROOT / "institutional_risk_gate_state.json", {})
    allowed_gross = safe_float(risk_state.get("master_exposure_multiplier"), 0.70)
    final_gross = float(bridge["final_optimizer_weight"].sum()) if not bridge.empty else 0.0
    max_single = float(bridge["final_optimizer_weight"].max()) if not bridge.empty else 0.0
    turnover = float((bridge["final_optimizer_weight_pct"] / 100.0 - bridge["current_weight_pct"] / 100.0).abs().sum()) if not bridge.empty else np.nan
    avg_corr = safe_float(bridge["avg_abs_corr_to_book"].mean()) if not bridge.empty and "avg_abs_corr_to_book" in bridge.columns else np.nan
    max_tca = safe_float(bridge["total_tca_cost_bps"].max()) if not bridge.empty and "total_tca_cost_bps" in bridge.columns else np.nan
    tech_weight = safe_float(sector.loc[sector["sector"].eq("Technology"), "final_sector_weight_pct"].iloc[0] / 100.0) if not sector.empty and sector["sector"].eq("Technology").any() else 0.0

    factor = read_csv_safe(ROOT / "factor_exposure_decomposition.csv")
    spy_beta = np.nan
    if not factor.empty and {"factor", "portfolio_beta"}.issubset(factor.columns):
        mask = factor["factor"].astype(str).eq("SPY_beta")
        if mask.any():
            spy_beta = safe_float(factor.loc[mask, "portfolio_beta"].iloc[0])

    signal_state = read_json_safe(ROOT / "signal_validation_state.json", {})
    p1_repairs = int(signal_state.get("p1_signal_repairs", 0) or 0)

    checks = [
        ("Allowed gross exposure", final_gross, allowed_gross, "max", "Final optimizer gross must not exceed the master risk multiplier.", "institutional_risk_gate_state.json"),
        ("Single-name max", max_single, SINGLE_NAME_CAP, "max", "No one ticker should dominate the research book.", "institutional_optimizer_bridge.csv"),
        ("Technology sector cap", tech_weight, TECH_SECTOR_CAP, "max", "Avoid hidden mega-tech concentration.", "institutional_optimizer_sector_allocations.csv"),
        ("Turnover budget", turnover, TURNOVER_BUDGET, "max", "Turnover should not eat the signal.", "institutional_target_weights.csv / portfolio_turnover_budget.csv"),
        ("SPY beta review", abs(spy_beta), SPY_BETA_LIMIT, "max", "Portfolio beta must be explicit and reviewed.", "factor_exposure_decomposition.csv"),
        ("Average correlation", avg_corr, AVG_CORR_REVIEW, "max", "Crowded/high-correlation books are less diversified than they look.", "holdings_correlation_matrix.csv"),
        ("Execution cost ceiling", max_tca, TCA_SIZE_DOWN_BPS, "max", "Expected transaction costs should not dominate signal edge.", "institutional_tca_cost_estimates.csv"),
        ("Signal P1 repairs", float(p1_repairs), 0.0, "max", "Optimizer should know when core signals require repair.", "signal_validation_state.json"),
        ("Cash reserve", max(0.0, 1.0 - final_gross), 0.20, "min", "Risk-down states should preserve cash instead of forcing exposure.", "institutional_optimizer_bridge.csv"),
    ]

    rows = []
    for name, current, limit, direction, note, source in checks:
        status = constraint_status(current, limit, direction=direction)
        if name == "Signal P1 repairs" and current > 0:
            status = "REVIEW"
        rows.append({
            "constraint": name,
            "current_value": round(float(current), 6) if np.isfinite(current) else np.nan,
            "limit_value": round(float(limit), 6) if np.isfinite(limit) else np.nan,
            "status": status,
            "note": note,
            "source_file": source,
        })
    return pd.DataFrame(rows)


def write_outputs(bridge: pd.DataFrame, audit: pd.DataFrame, sector: pd.DataFrame, why_not_more: pd.DataFrame, active_risk: pd.DataFrame, ladder: pd.DataFrame) -> None:
    bridge.to_csv(OUT_BRIDGE, index=False)
    audit.to_csv(OUT_AUDIT, index=False)
    sector.to_csv(OUT_SECTOR, index=False)
    why_not_more.to_csv(OUT_WHY_NOT_MORE, index=False)
    active_risk.to_csv(OUT_ACTIVE_RISK, index=False)
    ladder.to_csv(OUT_LADDER, index=False)

    status_scores = {"CLEAR": 90, "REVIEW": 70, "SIZE_DOWN": 45, "BLOCK_NEW": 20, "DATA_GAP": 35}
    audit_score = float(audit["status"].astype(str).str.upper().map(status_scores).fillna(50).mean()) if not audit.empty else 20.0
    bridge_score = float(bridge["final_optimizer_status"].astype(str).str.upper().map(status_scores).fillna(50).mean()) if not bridge.empty else 20.0
    score = 0.55 * audit_score + 0.45 * bridge_score
    final_gross = float(bridge["final_optimizer_weight"].sum()) if not bridge.empty else 0.0
    state = {
        "date": today_str(),
        "institutional_optimizer_score": round(score, 1),
        "overall_status": status_from_score(score),
        "tickers_optimized": int(len(bridge)),
        "final_gross_pct": round(final_gross * 100, 2),
        "cash_reserve_pct": round(max(0.0, 1.0 - final_gross) * 100, 2),
        "constraint_flags": int((audit["status"].astype(str).str.upper() != "CLEAR").sum()) if not audit.empty else 0,
        "risk_gate_dominates_count": int(bridge["binding_constraints"].astype(str).str.contains("risk budget|hard risk gate|risk size-down", regex=True, na=False).sum()) if not bridge.empty else 0,
        "signal_validation_bind_count": int(bridge["binding_constraints"].astype(str).str.contains("signal|horizon", regex=True, na=False).sum()) if not bridge.empty else 0,
        "subsector_cycle_bind_count": int(bridge["binding_constraints"].astype(str).str.contains("subsector|late-cycle", regex=True, na=False).sum()) if not bridge.empty else 0,
        "why_not_more_rows": int(len(why_not_more)),
        "active_risk_flags": int((active_risk["status"].astype(str).str.upper() != "CLEAR").sum()) if not active_risk.empty else 0,
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
        "truth": "This is a risk-gated research optimizer bridge. It does not submit orders and does not allow math optimizer weights to override risk limits.",
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Optimizer score: **{state['institutional_optimizer_score']}/100**",
        f"- Final gross: **{state['final_gross_pct']}%**",
        f"- Cash reserve: **{state['cash_reserve_pct']}%**",
        f"- Constraint flags: **{state['constraint_flags']}**",
        "",
        state["truth"],
        "",
        "## Constraint Audit",
        "",
        df_to_markdown(audit, max_rows=40),
        "",
        "## Optimizer Bridge",
        "",
        df_to_markdown(bridge, max_rows=80),
        "",
        "## Why Not More",
        "",
        df_to_markdown(why_not_more, max_rows=80),
        "",
        "## Active Risk Budget",
        "",
        df_to_markdown(active_risk, max_rows=40),
        "",
        "## Constraint Ladder",
        "",
        df_to_markdown(ladder, max_rows=120),
        "",
        "## Sector Allocations",
        "",
        df_to_markdown(sector, max_rows=30),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 157 - Institutional Portfolio Optimizer Bridge", sections)


def main() -> None:
    bridge = build_bridge()
    sector = build_sector_allocations(bridge)
    audit = build_constraint_audit(bridge, sector)
    why_not_more = build_why_not_more(bridge)
    active_risk = build_active_risk_budget(bridge, sector)
    ladder = build_constraint_ladder(bridge)
    write_outputs(bridge, audit, sector, why_not_more, active_risk, ladder)
    state = read_json_safe(OUT_STATE, {})
    print("Canyon v9 Step157 institutional portfolio optimizer bridge complete.")
    print(f"Overall: {state.get('overall_status')} ({state.get('institutional_optimizer_score')}/100)")
    print(f"Final gross: {state.get('final_gross_pct')}% | cash: {state.get('cash_reserve_pct')}% | constraints: {state.get('constraint_flags')}")
    print(f"Outputs: {OUT_BRIDGE.name}, {OUT_WHY_NOT_MORE.name}, {OUT_ACTIVE_RISK.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
