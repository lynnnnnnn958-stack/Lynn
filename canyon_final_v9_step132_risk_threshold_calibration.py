#!/usr/bin/env python3
"""
Canyon v9 - Step 132: Risk Threshold Calibration
================================================

Research-only. No broker connection. No live orders.

Step111-118 define the current institutional risk controls. Step131 summarizes
them. This step checks whether those risk thresholds look too loose, too tight,
or not yet calibratable from the data currently available.

Important: this step does not mutate upstream risk limits. It only produces
recommendations for human review.

Outputs:
  risk_threshold_calibration.csv
  risk_calibration_scorecard.csv
  risk_threshold_calibration_state.json
  risk_threshold_calibration_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    TRADING_DAYS,
    df_to_markdown,
    load_current_book,
    now_str,
    portfolio_return_series,
    read_csv_safe,
    read_json_safe,
    var_cvar,
    write_json,
    write_markdown_report,
)


ROOT = Path(__file__).parent

OUT_CALIBRATION = ROOT / "risk_threshold_calibration.csv"
OUT_SCORECARD = ROOT / "risk_calibration_scorecard.csv"
OUT_STATE = ROOT / "risk_threshold_calibration_state.json"
OUT_REPORT = ROOT / "risk_threshold_calibration_report.md"

MIN_HISTORY_OBS = 180
DEFAULT_ROLLING_WINDOW = 252


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_pct(value: Any) -> float:
    x = safe_float(value)
    if not np.isfinite(x):
        return np.nan
    return x / 100.0 if abs(x) > 2.0 else x


def percentile_rank(value: float, sample: pd.Series) -> float:
    s = pd.to_numeric(sample, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0 or not np.isfinite(value):
        return np.nan
    return float((s <= value).mean() * 100.0)


def q(sample: pd.Series, quantile: float) -> float:
    s = pd.to_numeric(sample, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(quantile))


def latest(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if df.empty or col not in df.columns:
        return default
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return default
    return float(s.iloc[-1])


def budget_row(budget: pd.DataFrame, item: str) -> dict[str, Any]:
    if budget.empty or "budget_item" not in budget.columns:
        return {}
    m = budget["budget_item"].astype(str).str.lower().eq(item.lower())
    if not m.any():
        return {}
    return budget.loc[m].iloc[0].to_dict()


def budget_current_limit(budget: pd.DataFrame, item: str, fallback_limit: float = np.nan) -> tuple[float, float, str]:
    row = budget_row(budget, item)
    current = safe_float(row.get("current_value"), np.nan)
    limit = safe_float(row.get("limit_value"), fallback_limit)
    source = str(row.get("source_file", "policy fallback"))
    return current, limit, source


def rolling_portfolio_metrics() -> tuple[pd.DataFrame, int, int]:
    book = load_current_book(prefer_filtered=True)
    returns = portfolio_return_series(book, lookback=1500)
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 80:
        return pd.DataFrame(), len(returns), 0

    window = min(DEFAULT_ROLLING_WINDOW, max(63, len(returns) // 2))
    rows = []
    for end in range(window, len(returns) + 1):
        sample = returns.iloc[end - window:end]
        v95, c95 = var_cvar(sample, alpha=0.95)
        rows.append({
            "date": sample.index[-1],
            "var_95_1d": v95,
            "cvar_95_1d": c95,
            "annual_vol": float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(sample) >= 20 else np.nan,
        })
    hist = pd.DataFrame(rows)
    return hist, len(returns), window


def policy_status(
    current_value: float,
    current_limit: float,
    calibrated_warning: float,
    calibrated_hard: float,
    current_percentile: float,
    sample_n: int,
    min_n: int,
    allow_policy_looseness_check: bool = True,
) -> str:
    if sample_n < min_n:
        return "MISSING_HISTORY"
    if not np.isfinite(current_limit):
        return "NEEDS_POLICY_LIMIT"
    if np.isfinite(current_value) and current_value > current_limit:
        return "ACTIVE_BREACH"
    if allow_policy_looseness_check:
        if np.isfinite(calibrated_hard) and current_limit > calibrated_hard * 1.35:
            return "TOO_LOOSE_REVIEW"
        if np.isfinite(calibrated_warning) and current_limit < calibrated_warning * 0.70:
            return "TOO_TIGHT_REVIEW"
    if np.isfinite(current_percentile) and current_percentile >= 90.0:
        return "REVIEW_HIGH_CURRENT_RISK"
    return "CALIBRATED_OK"


def next_action(status: str, control: str) -> str:
    if status == "ACTIVE_BREACH":
        return f"Keep the current risk gate active for {control}; do not let ticker ideas override this breach."
    if status == "TOO_LOOSE_REVIEW":
        return f"Review whether the hard limit for {control} should be pulled closer to the calibrated hard level."
    if status == "TOO_TIGHT_REVIEW":
        return f"Review whether {control} is over-blocking normal conditions or whether current size is too large."
    if status == "REVIEW_HIGH_CURRENT_RISK":
        return f"Current {control} is in the high historical percentile; size down until it normalizes."
    if status in {"MISSING_HISTORY", "NEEDS_POLICY_LIMIT"}:
        return f"Collect more point-in-time history before treating {control} as institutionally calibrated."
    return f"{control} is usable as a first-pass threshold. Keep monitoring realized breach rates."


def make_row(
    area: str,
    control: str,
    current_value: float,
    current_limit: float,
    sample: pd.Series,
    sample_n: int,
    basis: str,
    source_file: str,
    min_n: int = MIN_HISTORY_OBS,
    calibration_mode: str = "HISTORICAL_ROLLING",
    allow_policy_looseness_check: bool = True,
) -> dict[str, Any]:
    calibrated_warning = q(sample, 0.75)
    calibrated_hard = q(sample, 0.90)
    current_percentile = percentile_rank(current_value, sample)
    status = policy_status(
        current_value=current_value,
        current_limit=current_limit,
        calibrated_warning=calibrated_warning,
        calibrated_hard=calibrated_hard,
        current_percentile=current_percentile,
        sample_n=sample_n,
        min_n=min_n,
        allow_policy_looseness_check=allow_policy_looseness_check,
    )
    return {
        "risk_area": area,
        "control": control,
        "current_value": current_value,
        "current_limit": current_limit,
        "calibrated_warning": calibrated_warning,
        "calibrated_hard": calibrated_hard,
        "historical_or_cross_sectional_percentile": current_percentile,
        "sample_n": int(sample_n),
        "calibration_mode": calibration_mode,
        "calibration_status": status,
        "recommended_policy": next_action(status, control),
        "basis": basis,
        "source_file": source_file,
        "research_only": True,
    }


def build_calibration_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    budget = read_csv_safe(ROOT / "institutional_risk_budget_summary.csv")
    portfolio_var = read_csv_safe(ROOT / "portfolio_var_cvar_summary.csv")
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")
    macro = read_csv_safe(ROOT / "macro_scenario_stress.csv")
    crisis = read_csv_safe(ROOT / "crisis_correlation_stress.csv")
    earnings = read_csv_safe(ROOT / "earnings_gap_down_risk.csv")
    liquidity = read_csv_safe(ROOT / "liquidity_crisis_simulation.csv")
    beta = read_csv_safe(ROOT / "portfolio_beta_report.csv")
    dd_state = read_json_safe(ROOT / "drawdown_control_state.json")

    hist, history_obs, rolling_window = rolling_portfolio_metrics()
    rows: list[dict[str, Any]] = []

    current, limit, source = budget_current_limit(budget, "Portfolio 1d VaR 95%", 0.02)
    if not np.isfinite(current):
        current = latest(portfolio_var, "var_95_1d")
    rows.append(make_row(
        "Portfolio tail risk", "Portfolio 1d VaR 95%",
        current, limit,
        hist.get("var_95_1d", pd.Series(dtype=float)),
        len(hist),
        f"Rolling {rolling_window}d current-book return distribution.",
        source,
        calibration_mode="HISTORICAL_ROLLING",
    ))

    current, limit, source = budget_current_limit(budget, "Portfolio 1d CVaR 95%", 0.035)
    if not np.isfinite(current):
        current = latest(portfolio_var, "cvar_95_1d")
    rows.append(make_row(
        "Portfolio tail risk", "Portfolio 1d CVaR 95%",
        current, limit,
        hist.get("cvar_95_1d", pd.Series(dtype=float)),
        len(hist),
        f"Rolling {rolling_window}d current-book return distribution.",
        source,
        calibration_mode="HISTORICAL_ROLLING",
    ))

    current, limit, source = budget_current_limit(budget, "Annual volatility target", 0.15)
    if not np.isfinite(current):
        current = latest(portfolio_var, "annual_vol")
    rows.append(make_row(
        "Volatility target", "Annual volatility target",
        current, limit,
        hist.get("annual_vol", pd.Series(dtype=float)),
        len(hist),
        f"Rolling {rolling_window}d current-book annualized volatility distribution.",
        source,
        calibration_mode="HISTORICAL_ROLLING",
    ))

    current, limit, source = budget_current_limit(budget, "Single-name tail-risk budget", 1.0)
    if not single.empty and "risk_budget_used_pct" in single.columns:
        sample = pd.to_numeric(single["risk_budget_used_pct"], errors="coerce")
        current = max(current, float(sample.max(skipna=True))) if np.isfinite(current) else float(sample.max(skipna=True))
    else:
        sample = pd.Series(dtype=float)
    rows.append(make_row(
        "Single-name risk", "Single-name tail-risk budget",
        current, limit, sample, int(sample.dropna().shape[0]),
        "Current-book cross-section of ticker risk-budget usage.",
        source,
        min_n=10,
        calibration_mode="CURRENT_CROSS_SECTION",
        allow_policy_looseness_check=False,
    ))

    cvar_sample = pd.to_numeric(single.get("cvar_95_1d", pd.Series(dtype=float)), errors="coerce") if not single.empty else pd.Series(dtype=float)
    rows.append(make_row(
        "Single-name risk", "Single-name 1d CVaR 95%",
        float(cvar_sample.max(skipna=True)) if not cvar_sample.dropna().empty else np.nan,
        np.nan,
        cvar_sample,
        int(cvar_sample.dropna().shape[0]),
        "Current-book cross-section. Needs true historical pre-trade calibration.",
        "single_name_risk_budget.csv",
        min_n=20,
        calibration_mode="CURRENT_CROSS_SECTION",
        allow_policy_looseness_check=False,
    ))

    current, limit, source = budget_current_limit(budget, "Earnings gap-loss budget", 0.01)
    if not earnings.empty and "estimated_gap_loss_model_account" in earnings.columns:
        e_loss = pd.to_numeric(earnings["estimated_gap_loss_model_account"], errors="coerce") / MODEL_ACCOUNT_VALUE
        current = max(current, float(e_loss.max(skipna=True))) if np.isfinite(current) else float(e_loss.max(skipna=True))
    else:
        e_loss = pd.Series(dtype=float)
    rows.append(make_row(
        "Event gap risk", "Earnings gap-loss budget",
        current, limit, e_loss, int(e_loss.dropna().shape[0]),
        "Current-book earnings gap loss estimates as percent of model account.",
        source,
        min_n=10,
        calibration_mode="CURRENT_CROSS_SECTION",
        allow_policy_looseness_check=True,
    ))

    current, limit, source = budget_current_limit(budget, "Sector concentration budget", 1.0)
    if not sector.empty and "cap_used_pct" in sector.columns:
        s_used = pd.to_numeric(sector["cap_used_pct"], errors="coerce") / 100.0
        current = max(current, float(s_used.max(skipna=True))) if np.isfinite(current) else float(s_used.max(skipna=True))
    else:
        s_used = pd.Series(dtype=float)
    rows.append(make_row(
        "Sector concentration", "Sector concentration budget",
        current, limit, s_used, int(s_used.dropna().shape[0]),
        "Current-book sector cap usage cross-section.",
        source,
        min_n=5,
        calibration_mode="CURRENT_CROSS_SECTION",
        allow_policy_looseness_check=False,
    ))

    current, limit, source = budget_current_limit(budget, "Factor beta budget", 1.25)
    if not beta.empty and "abs_beta" in beta.columns:
        b_abs = pd.to_numeric(beta["abs_beta"], errors="coerce")
        current = max(current, float(b_abs.max(skipna=True))) if np.isfinite(current) else float(b_abs.max(skipna=True))
    else:
        b_abs = pd.Series(dtype=float)
    rows.append(make_row(
        "Factor exposure", "Factor beta budget",
        current, limit, b_abs, int(b_abs.dropna().shape[0]),
        "Current proxy-factor beta cross-section. Not a full Barra/Axioma model.",
        source,
        min_n=5,
        calibration_mode="CURRENT_CROSS_SECTION",
        allow_policy_looseness_check=False,
    ))

    current, limit, source = budget_current_limit(budget, "Macro scenario loss budget", 0.15)
    limit = abs(limit)
    if not macro.empty and "conservative_portfolio_impact" in macro.columns:
        m_loss = pd.to_numeric(macro["conservative_portfolio_impact"], errors="coerce").abs()
        current = max(abs(current), float(m_loss.max(skipna=True))) if np.isfinite(current) else float(m_loss.max(skipna=True))
    else:
        m_loss = pd.Series(dtype=float)
    rows.append(make_row(
        "Macro stress", "Macro scenario loss budget",
        abs(current), limit, m_loss, int(m_loss.dropna().shape[0]),
        "Hand-built macro scenario library; needs historical scenario calibration.",
        source,
        min_n=5,
        calibration_mode="SCENARIO_LIBRARY",
        allow_policy_looseness_check=False,
    ))

    current, limit, source = budget_current_limit(budget, "Crisis-correlation volatility budget", 1.5)
    if not crisis.empty and "vol_increase_ratio" in crisis.columns:
        c_ratio = pd.to_numeric(crisis["vol_increase_ratio"], errors="coerce")
        current = max(current, float(c_ratio.max(skipna=True))) if np.isfinite(current) else float(c_ratio.max(skipna=True))
    else:
        c_ratio = pd.Series(dtype=float)
    rows.append(make_row(
        "Correlation stress", "Crisis-correlation volatility budget",
        current, limit, c_ratio, int(c_ratio.dropna().shape[0]),
        "Current crisis-correlation override. Needs rolling crisis-window history.",
        source,
        min_n=5,
        calibration_mode="POINT_ESTIMATE",
        allow_policy_looseness_check=False,
    ))

    current, limit, source = budget_current_limit(budget, "Liquidity crisis liquidation budget", 0.05)
    if not liquidity.empty and "estimated_liquidation_loss" in liquidity.columns:
        l_loss = pd.to_numeric(liquidity["estimated_liquidation_loss"], errors="coerce") / MODEL_ACCOUNT_VALUE
        portfolio_liq_loss = float(l_loss.sum(skipna=True)) if not l_loss.dropna().empty else np.nan
        current = max(current, portfolio_liq_loss) if np.isfinite(current) and np.isfinite(portfolio_liq_loss) else portfolio_liq_loss
    else:
        l_loss = pd.Series(dtype=float)
    rows.append(make_row(
        "Liquidity stress", "Liquidity crisis liquidation budget",
        current, limit, pd.Series([current]),
        1 if np.isfinite(current) else 0,
        "Current-book portfolio liquidation loss point estimate. Needs crisis-period liquidity history.",
        source,
        min_n=20,
        calibration_mode="POINT_ESTIMATE",
        allow_policy_looseness_check=False,
    ))

    dd_current = abs(clean_pct(dd_state.get("drawdown_pct", np.nan))) if dd_state else np.nan
    if not np.isfinite(dd_current):
        dd_current, _, source = budget_current_limit(budget, "Drawdown budget", 0.10)
        dd_current = abs(dd_current)
    dd_limit = safe_float(dd_state.get("hard_stop_drawdown", np.nan), np.nan) if dd_state else np.nan
    if not np.isfinite(dd_limit):
        _, dd_limit, source = budget_current_limit(budget, "Drawdown budget", 0.10)
    rows.append(make_row(
        "Drawdown control", "Drawdown budget",
        dd_current, dd_limit,
        hist.get("var_95_1d", pd.Series(dtype=float)) * 0.0 + dd_current,
        1,
        "Current drawdown state only. Needs live NAV history for true calibration.",
        "drawdown_control_state.json",
        min_n=20,
        calibration_mode="POINT_ESTIMATE",
        allow_policy_looseness_check=False,
    ))

    calibration = pd.DataFrame(rows)
    for col in [
        "current_value", "current_limit", "calibrated_warning", "calibrated_hard",
        "historical_or_cross_sectional_percentile",
    ]:
        calibration[col] = pd.to_numeric(calibration[col], errors="coerce")
    calibration["limit_vs_calibrated_hard"] = calibration["current_limit"] / calibration["calibrated_hard"].replace(0, np.nan)
    calibration = calibration.sort_values(
        ["calibration_status", "historical_or_cross_sectional_percentile"],
        ascending=[True, False],
    ).reset_index(drop=True)

    meta = {
        "portfolio_history_observations": int(history_obs),
        "rolling_window": int(rolling_window),
        "book_source": "load_current_book(prefer_filtered=True)",
    }
    return calibration, meta


def status_penalty(status: str) -> int:
    penalties = {
        "ACTIVE_BREACH": 28,
        "TOO_LOOSE_REVIEW": 20,
        "REVIEW_HIGH_CURRENT_RISK": 16,
        "MISSING_HISTORY": 16,
        "NEEDS_POLICY_LIMIT": 16,
        "TOO_TIGHT_REVIEW": 10,
        "CALIBRATED_OK": 0,
    }
    return penalties.get(str(status).upper(), 12)


def area_status(statuses: list[str]) -> str:
    values = {str(x).upper() for x in statuses}
    if "ACTIVE_BREACH" in values:
        return "ACTIVE_BREACH"
    if "TOO_LOOSE_REVIEW" in values or "TOO_TIGHT_REVIEW" in values:
        return "POLICY_REVIEW"
    if "REVIEW_HIGH_CURRENT_RISK" in values:
        return "HIGH_CURRENT_RISK"
    if "MISSING_HISTORY" in values or "NEEDS_POLICY_LIMIT" in values:
        return "NEEDS_HISTORY"
    return "CALIBRATED_OK"


def build_scorecard(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame()
    rows = []
    for area, grp in calibration.groupby("risk_area", sort=False):
        statuses = grp["calibration_status"].astype(str).tolist()
        penalty = sum(status_penalty(s) for s in statuses)
        score = max(0, 100 - penalty)
        rows.append({
            "risk_area": area,
            "score": score,
            "status": area_status(statuses),
            "controls": len(grp),
            "active_breaches": int((grp["calibration_status"] == "ACTIVE_BREACH").sum()),
            "policy_reviews": int(grp["calibration_status"].astype(str).str.contains("TOO_", regex=False).sum()),
            "missing_history": int(grp["calibration_status"].isin(["MISSING_HISTORY", "NEEDS_POLICY_LIMIT"]).sum()),
            "main_next_action": grp.sort_values("historical_or_cross_sectional_percentile", ascending=False)["recommended_policy"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["score", "risk_area"], ascending=[True, True]).reset_index(drop=True)


def main() -> int:
    calibration, meta = build_calibration_table()
    scorecard = build_scorecard(calibration)

    calibration.to_csv(OUT_CALIBRATION, index=False)
    scorecard.to_csv(OUT_SCORECARD, index=False)

    statuses = calibration["calibration_status"].astype(str).str.upper() if not calibration.empty else pd.Series(dtype=str)
    state = {
        "run_time": now_str(),
        "research_only": True,
        "no_broker_connection": True,
        "logic": "Calibrate and recommend only. This step does not change upstream risk limits.",
        "overall_status": (
            "ACTIVE_BREACH" if (statuses == "ACTIVE_BREACH").any()
            else "POLICY_REVIEW" if statuses.str.contains("TOO_", regex=False).any()
            else "NEEDS_HISTORY" if statuses.isin(["MISSING_HISTORY", "NEEDS_POLICY_LIMIT"]).any()
            else "CALIBRATED_OK"
        ),
        "controls_checked": int(len(calibration)),
        "active_breach_count": int((statuses == "ACTIVE_BREACH").sum()),
        "policy_review_count": int(statuses.str.contains("TOO_", regex=False).sum()),
        "missing_history_count": int(statuses.isin(["MISSING_HISTORY", "NEEDS_POLICY_LIMIT"]).sum()),
        "review_high_current_risk_count": int((statuses == "REVIEW_HIGH_CURRENT_RISK").sum()),
        **meta,
        "outputs": {
            "calibration_table": OUT_CALIBRATION.name,
            "scorecard": OUT_SCORECARD.name,
            "state": OUT_STATE.name,
            "report": OUT_REPORT.name,
        },
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Summary",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Controls checked: {state['controls_checked']}",
        f"- Active breaches: {state['active_breach_count']}",
        f"- Policy reviews: {state['policy_review_count']}",
        f"- Missing history / policy limits: {state['missing_history_count']}",
        f"- Portfolio history observations: {state['portfolio_history_observations']}",
        "",
        "## Calibration Scorecard",
        "",
        df_to_markdown(scorecard, max_rows=20),
        "",
        "## Threshold Calibration Table",
        "",
        df_to_markdown(calibration, max_rows=40),
        "",
        "## Product Truth",
        "",
        "This is a first-pass calibration layer. It is useful for deciding what deserves risk review, but it is not yet a full institutional threshold research process because most event, liquidity, crisis-correlation, and drawdown controls still need deeper point-in-time history.",
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 132 - Risk Threshold Calibration", sections)

    print(f"wrote {OUT_CALIBRATION.name} rows={len(calibration)}")
    print(f"wrote {OUT_SCORECARD.name} rows={len(scorecard)}")
    print(f"overall_status={state['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
