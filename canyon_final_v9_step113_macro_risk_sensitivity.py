#!/usr/bin/env python3
"""
Canyon v9 Step 113 - Portfolio macro sensitivity and scenario stress.

Research-only. No broker connection. No live orders.

Outputs:
  portfolio_macro_sensitivity.csv
  macro_scenario_stress.csv
  macro_risk_sensitivity_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    FACTOR_PROXIES,
    ROOT,
    beta_to_factor,
    df_to_markdown,
    get_returns,
    load_current_book,
    portfolio_return_series,
    read_csv_safe,
    read_json_safe,
    source_age,
    worst_status,
    write_markdown_report,
)


OUT_SENS = ROOT / "portfolio_macro_sensitivity.csv"
OUT_SCEN = ROOT / "macro_scenario_stress.csv"
OUT_MD = ROOT / "macro_risk_sensitivity_report.md"

MODEL_FACTOR_WEIGHTS = {
    "SPY_beta": 0.30,
    "QQQ_growth_beta": 0.18,
    "rates_beta_TLT": 0.15,
    "usd_beta_UUP": 0.08,
    "oil_energy_beta_XLE": 0.10,
    "gold_beta_GLD": 0.05,
    "credit_beta_HYG": 0.10,
    "semiconductor_beta_SMH": 0.04,
}

SCENARIOS = [
    {
        "scenario": "2022-style rate hike shock",
        "description": "Rates up, growth multiple compression, credit weaker",
        "SPY_beta": -0.12,
        "QQQ_growth_beta": -0.22,
        "rates_beta_TLT": -0.16,
        "usd_beta_UUP": 0.06,
        "oil_energy_beta_XLE": 0.04,
        "gold_beta_GLD": -0.03,
        "credit_beta_HYG": -0.08,
        "semiconductor_beta_SMH": -0.25,
    },
    {
        "scenario": "March 2020 liquidity crash",
        "description": "Equity shock, credit shock, liquidity disappears, correlations rise",
        "SPY_beta": -0.24,
        "QQQ_growth_beta": -0.20,
        "rates_beta_TLT": 0.08,
        "usd_beta_UUP": 0.05,
        "oil_energy_beta_XLE": -0.35,
        "gold_beta_GLD": -0.06,
        "credit_beta_HYG": -0.12,
        "semiconductor_beta_SMH": -0.28,
    },
    {
        "scenario": "Q4 2018 growth drawdown",
        "description": "Fed tightening plus risk-off growth unwind",
        "SPY_beta": -0.17,
        "QQQ_growth_beta": -0.22,
        "rates_beta_TLT": 0.04,
        "usd_beta_UUP": 0.03,
        "oil_energy_beta_XLE": -0.18,
        "gold_beta_GLD": 0.04,
        "credit_beta_HYG": -0.07,
        "semiconductor_beta_SMH": -0.24,
    },
    {
        "scenario": "2008 crisis proxy",
        "description": "Deep equity and credit drawdown, crisis correlations",
        "SPY_beta": -0.38,
        "QQQ_growth_beta": -0.36,
        "rates_beta_TLT": 0.14,
        "usd_beta_UUP": 0.08,
        "oil_energy_beta_XLE": -0.30,
        "gold_beta_GLD": 0.02,
        "credit_beta_HYG": -0.20,
        "semiconductor_beta_SMH": -0.42,
    },
    {
        "scenario": "2023 regional bank shock",
        "description": "Credit stress, financials selloff, rates down",
        "SPY_beta": -0.08,
        "QQQ_growth_beta": -0.02,
        "rates_beta_TLT": 0.08,
        "usd_beta_UUP": 0.02,
        "oil_energy_beta_XLE": -0.10,
        "gold_beta_GLD": 0.05,
        "credit_beta_HYG": -0.10,
        "semiconductor_beta_SMH": -0.05,
    },
    {
        "scenario": "AI / semiconductor unwind",
        "description": "Crowded AI factor reverses while broad index only partly falls",
        "SPY_beta": -0.08,
        "QQQ_growth_beta": -0.16,
        "rates_beta_TLT": 0.02,
        "usd_beta_UUP": 0.02,
        "oil_energy_beta_XLE": -0.04,
        "gold_beta_GLD": 0.02,
        "credit_beta_HYG": -0.04,
        "semiconductor_beta_SMH": -0.32,
    },
    {
        "scenario": "10y yield +100bp proxy",
        "description": "Duration shock: TLT -10%, growth and credit weaker",
        "SPY_beta": -0.06,
        "QQQ_growth_beta": -0.10,
        "rates_beta_TLT": -0.10,
        "usd_beta_UUP": 0.03,
        "oil_energy_beta_XLE": 0.01,
        "gold_beta_GLD": -0.04,
        "credit_beta_HYG": -0.04,
        "semiconductor_beta_SMH": -0.12,
    },
]


def load_macro_context() -> dict:
    ctx = {}
    macro_csv = read_csv_safe(ROOT / "macro_signals.csv")
    if not macro_csv.empty:
        last = macro_csv.tail(1).iloc[0].to_dict()
        ctx.update({f"macro_{k}": v for k, v in last.items()})
    macro_json = read_json_safe(ROOT / "macro_signals.json", {})
    if isinstance(macro_json, dict):
        ctx.update({f"macro_json_{k}": v for k, v in macro_json.items()})
    return ctx


def build_macro_sensitivity() -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    p_ret = portfolio_return_series(book, lookback=756)
    factor_returns = get_returns(list(FACTOR_PROXIES.values()), lookback=756)
    rows = []
    for factor_name, proxy in FACTOR_PROXIES.items():
        beta = np.nan
        factor_20d_return = np.nan
        if not p_ret.empty and proxy in factor_returns.columns:
            beta = beta_to_factor(p_ret, factor_returns[proxy])
            fr = factor_returns[proxy].dropna().tail(20)
            if not fr.empty:
                factor_20d_return = float((1.0 + fr).prod() - 1.0)
        abs_beta = abs(beta) if np.isfinite(beta) else np.nan
        if not np.isfinite(beta):
            status = "MISSING_DATA_REVIEW"
        elif abs_beta >= 1.30:
            status = "SIZE_DOWN"
        elif abs_beta >= 0.90:
            status = "REVIEW"
        else:
            status = "CLEAR"
        rows.append({
            "factor": factor_name,
            "proxy": proxy,
            "portfolio_beta": beta,
            "abs_beta": abs_beta,
            "factor_20d_return": factor_20d_return,
            "estimated_20d_portfolio_impact": beta * factor_20d_return if np.isfinite(beta) and np.isfinite(factor_20d_return) else np.nan,
            "model_weight": MODEL_FACTOR_WEIGHTS.get(factor_name, 0.0),
            "sensitivity_status": status,
            "source_file": "sp500_price_cache.csv/backtest_price_cache.csv",
        })
    return pd.DataFrame(rows)


def build_scenario_stress(sens: pd.DataFrame) -> pd.DataFrame:
    beta_map = {}
    status_map = {}
    if not sens.empty:
        beta_map = dict(zip(sens["factor"], sens["portfolio_beta"]))
        status_map = dict(zip(sens["factor"], sens["sensitivity_status"]))
    rows = []
    for scen in SCENARIOS:
        impacts = []
        missing = []
        for factor_name in MODEL_FACTOR_WEIGHTS:
            beta = beta_map.get(factor_name, np.nan)
            shock = float(scen.get(factor_name, 0.0))
            model_weight = MODEL_FACTOR_WEIGHTS.get(factor_name, 0.0)
            if np.isfinite(beta):
                impacts.append(beta * shock * model_weight)
            else:
                missing.append(factor_name)
        impact = float(np.nansum(impacts)) if impacts else np.nan
        # The model is proxy-based; add a conservative floor for severe scenarios.
        severity_floor = {
            "2022-style rate hike shock": -0.10,
            "March 2020 liquidity crash": -0.16,
            "Q4 2018 growth drawdown": -0.10,
            "2008 crisis proxy": -0.22,
            "2023 regional bank shock": -0.06,
            "AI / semiconductor unwind": -0.08,
            "10y yield +100bp proxy": -0.06,
        }.get(scen["scenario"], -0.05)
        conservative_impact = min(impact, severity_floor) if np.isfinite(impact) else severity_floor
        if conservative_impact <= -0.15:
            action = "REDUCE_ONLY"
        elif conservative_impact <= -0.10:
            action = "SIZE_DOWN"
        elif conservative_impact <= -0.05:
            action = "REVIEW"
        else:
            action = "CLEAR"
        factor_status = worst_status([status_map.get(k, "CLEAR") for k in MODEL_FACTOR_WEIGHTS])
        rows.append({
            "scenario": scen["scenario"],
            "description": scen["description"],
            "proxy_model_impact": impact,
            "conservative_portfolio_impact": conservative_impact,
            "scenario_action": worst_status([action, factor_status if missing else "CLEAR"]),
            "missing_factor_betas": ", ".join(missing),
            "source_file": "portfolio_macro_sensitivity.csv; macro_signals.csv; macro_signals.json",
            **{f"shock_{k}": scen.get(k, 0.0) for k in MODEL_FACTOR_WEIGHTS},
        })
    return pd.DataFrame(rows)


def write_report(sens: pd.DataFrame, scen: pd.DataFrame) -> None:
    ctx = load_macro_context()
    macro_signal = ctx.get("macro_macro_signal", ctx.get("macro_json_macro_signal", "UNKNOWN"))
    macro_score = ctx.get("macro_macro_score", ctx.get("macro_json_macro_score", "UNKNOWN"))
    sections = [
        "## Summary",
        "",
        f"- Macro signal: {macro_signal}",
        f"- Macro score: {macro_score}",
        f"- Sensitivity rows: {len(sens)}",
        f"- Scenario rows: {len(scen)}",
        f"- Macro source age: {source_age(ROOT / 'macro_signals.csv')}",
        "",
        "## Logic",
        "",
        "- Factor beta is estimated from local price caches only.",
        "- Missing beta is a review item, not a reason to improve an action.",
        "- Scenario impacts use a conservative floor because crisis relationships are unstable.",
        "",
        "## Macro sensitivity",
        "",
        df_to_markdown(sens) if not sens.empty else "No sensitivity rows.",
        "",
        "## Scenario stress",
        "",
        df_to_markdown(scen[["scenario", "conservative_portfolio_impact", "scenario_action", "missing_factor_betas"]])
        if not scen.empty else "No scenario rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 113 - Macro Risk Sensitivity", sections)


def main() -> None:
    sens = build_macro_sensitivity()
    scen = build_scenario_stress(sens)
    sens.to_csv(OUT_SENS, index=False)
    scen.to_csv(OUT_SCEN, index=False)
    write_report(sens, scen)
    print(f"[step113] wrote {OUT_SENS.name}: {len(sens)} rows")
    print(f"[step113] wrote {OUT_SCEN.name}: {len(scen)} rows")
    print(f"[step113] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
