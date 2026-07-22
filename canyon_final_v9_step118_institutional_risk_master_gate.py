#!/usr/bin/env python3
"""
Canyon v9 Step 118 - Institutional risk master budget and final gate.

Research-only. No broker connection. No live orders.

This is the missing "one desk" layer: it combines Step111-117 into a single
portfolio-level risk budget, explicit VaR/CVaR summary, factor exposure table,
and per-ticker final risk gate. Component risk modules can reduce or block a
paper idea; none of them can upgrade a ticker.

Outputs:
  institutional_risk_budget_summary.csv
  portfolio_var_cvar_summary.csv
  factor_exposure_decomposition.csv
  final_risk_gate.csv
  institutional_risk_gate_state.json
  institutional_risk_master_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    annualized_vol,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    portfolio_return_series,
    read_csv_safe,
    read_json_safe,
    today_str,
    var_cvar,
    worst_status,
    write_json,
    write_markdown_report,
)


OUT_BUDGET = ROOT / "institutional_risk_budget_summary.csv"
OUT_VAR = ROOT / "portfolio_var_cvar_summary.csv"
OUT_FACTOR = ROOT / "factor_exposure_decomposition.csv"
OUT_GATE = ROOT / "final_risk_gate.csv"
OUT_STATE = ROOT / "institutional_risk_gate_state.json"
OUT_MD = ROOT / "institutional_risk_master_report.md"


def action_rank(label: str) -> int:
    order = {
        "CLEAR": 0,
        "OK": 0,
        "WATCH": 1,
        "REVIEW": 2,
        "MISSING_DATA_REVIEW": 3,
        "SIZE_DOWN": 4,
        "REDUCE_ONLY": 5,
        "BLOCK_NEW": 6,
        "BLOCKED": 6,
    }
    return order.get(str(label).upper(), 2)


def status_from_used(used_pct: float, warn: float = 0.80, hard: float = 1.00) -> str:
    if not np.isfinite(used_pct):
        return "MISSING_DATA_REVIEW"
    if used_pct >= hard * 1.50:
        return "REDUCE_ONLY"
    if used_pct >= hard:
        return "SIZE_DOWN"
    if used_pct >= warn:
        return "REVIEW"
    return "CLEAR"


def numeric(value, default=np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def exposure_multiplier_from_action(action: str) -> float:
    action = str(action).upper()
    if action in {"BLOCK_NEW", "BLOCKED", "REDUCE_ONLY"}:
        return 0.50
    if action == "SIZE_DOWN":
        return 0.70
    if action in {"REVIEW", "MISSING_DATA_REVIEW"}:
        return 0.85
    return 1.00


def build_portfolio_var_cvar(book: pd.DataFrame) -> pd.DataFrame:
    p = portfolio_return_series(book, lookback=756)
    if p.empty or len(p) < 40:
        base = {
            "date": today_str(),
            "gross_exposure": float(book["weight"].sum()) if not book.empty else np.nan,
            "annual_vol": np.nan,
            "var_95_1d": np.nan,
            "cvar_95_1d": np.nan,
            "var_99_1d": np.nan,
            "cvar_99_1d": np.nan,
            "status": "MISSING_DATA_REVIEW",
            "source_file": "current book; local price caches",
        }
    else:
        var95, cvar95 = var_cvar(p, 0.95)
        var99, cvar99 = var_cvar(p, 0.99)
        vol = annualized_vol(p)
        status = worst_status([
            status_from_used(var95 / 0.020 if np.isfinite(var95) else np.nan),
            status_from_used(cvar95 / 0.035 if np.isfinite(cvar95) else np.nan),
            status_from_used(vol / 0.150 if np.isfinite(vol) else np.nan),
        ])
        base = {
            "date": today_str(),
            "gross_exposure": float(book["weight"].sum()) if not book.empty else np.nan,
            "annual_vol": vol,
            "var_95_1d": var95,
            "cvar_95_1d": cvar95,
            "var_99_1d": var99,
            "cvar_99_1d": cvar99,
            "status": status,
            "source_file": "current book; sp500_price_cache.csv/backtest_price_cache.csv",
        }
    for horizon in [5, 20]:
        scale = np.sqrt(horizon)
        for key in ["var_95_1d", "cvar_95_1d", "var_99_1d", "cvar_99_1d"]:
            out_key = key.replace("_1d", f"_{horizon}d")
            base[out_key] = base[key] * scale if np.isfinite(base.get(key, np.nan)) else np.nan
    for key in list(base):
        if key.startswith(("var_", "cvar_")):
            base[key + "_dollars"] = base[key] * MODEL_ACCOUNT_VALUE if np.isfinite(base[key]) else np.nan
    return pd.DataFrame([base])


def build_factor_exposure_decomposition() -> pd.DataFrame:
    macro = read_csv_safe(ROOT / "portfolio_macro_sensitivity.csv")
    theme = read_csv_safe(ROOT / "theme_factor_exposure.csv")
    rows = []
    if not macro.empty:
        for _, row in macro.iterrows():
            beta = pd.to_numeric(pd.Series([row.get("portfolio_beta", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "exposure_type": "macro_beta",
                "factor": row.get("factor", ""),
                "proxy": row.get("proxy", ""),
                "portfolio_beta": beta,
                "portfolio_weight": np.nan,
                "factor_20d_return": row.get("factor_20d_return", np.nan),
                "estimated_20d_impact": row.get("estimated_20d_portfolio_impact", np.nan),
                "status": row.get("sensitivity_status", "REVIEW"),
                "source_file": "portfolio_macro_sensitivity.csv",
            })
    if not theme.empty:
        for _, row in theme.iterrows():
            if str(row.get("exposure_type", "")) == "market_factor":
                continue
            weight = pd.to_numeric(pd.Series([row.get("portfolio_weight", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "exposure_type": "theme_weight",
                "factor": row.get("factor_or_theme", ""),
                "proxy": "",
                "portfolio_beta": np.nan,
                "portfolio_weight": weight,
                "factor_20d_return": np.nan,
                "estimated_20d_impact": np.nan,
                "status": row.get("exposure_status", "REVIEW"),
                "source_file": "theme_factor_exposure.csv",
            })
    return pd.DataFrame(rows)


def build_budget_summary(book: pd.DataFrame, var_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")
    macro = read_csv_safe(ROOT / "macro_scenario_stress.csv")
    crisis = read_csv_safe(ROOT / "crisis_correlation_stress.csv")
    liq = read_csv_safe(ROOT / "liquidity_crisis_simulation.csv")
    gap = read_csv_safe(ROOT / "earnings_gap_down_risk.csv")
    dd = read_json_safe(ROOT / "drawdown_control_state.json", {})
    vol = read_json_safe(ROOT / "vol_target_state.json", {})

    rows = []
    gross = float(book["weight"].sum()) if not book.empty else np.nan
    rows.append({
        "scope": "PORTFOLIO",
        "budget_item": "Total gross exposure",
        "current_value": gross,
        "limit_value": 1.00,
        "used_pct": gross / 1.00 if np.isfinite(gross) else np.nan,
        "status": status_from_used(gross / 1.00 if np.isfinite(gross) else np.nan, warn=0.90, hard=1.05),
        "action_if_breached": "SIZE_DOWN",
        "source_file": "current book",
    })

    if not var_df.empty:
        row = var_df.iloc[0]
        target_vol_current = numeric(vol.get("estimated_annual_vol", row.get("annual_vol", np.nan)))
        rows.extend([
            {
                "scope": "PORTFOLIO",
                "budget_item": "Portfolio 1d VaR 95%",
                "current_value": row.get("var_95_1d", np.nan),
                "limit_value": 0.020,
                "used_pct": row.get("var_95_1d", np.nan) / 0.020 if np.isfinite(row.get("var_95_1d", np.nan)) else np.nan,
                "status": status_from_used(row.get("var_95_1d", np.nan) / 0.020 if np.isfinite(row.get("var_95_1d", np.nan)) else np.nan),
                "action_if_breached": "SIZE_DOWN",
                "source_file": "portfolio_var_cvar_summary.csv",
            },
            {
                "scope": "PORTFOLIO",
                "budget_item": "Portfolio 1d CVaR 95%",
                "current_value": row.get("cvar_95_1d", np.nan),
                "limit_value": 0.035,
                "used_pct": row.get("cvar_95_1d", np.nan) / 0.035 if np.isfinite(row.get("cvar_95_1d", np.nan)) else np.nan,
                "status": status_from_used(row.get("cvar_95_1d", np.nan) / 0.035 if np.isfinite(row.get("cvar_95_1d", np.nan)) else np.nan),
                "action_if_breached": "SIZE_DOWN",
                "source_file": "portfolio_var_cvar_summary.csv",
            },
            {
                "scope": "PORTFOLIO",
                "budget_item": "Annual volatility target",
                "current_value": target_vol_current,
                "limit_value": 0.150,
                "used_pct": target_vol_current / 0.150 if np.isfinite(target_vol_current) else np.nan,
                "status": worst_status([
                    str(vol.get("vol_action", "")),
                    status_from_used(target_vol_current / 0.150 if np.isfinite(target_vol_current) else np.nan, warn=1.00, hard=1.05),
                ]),
                "action_if_breached": "APPLY_VOL_MULTIPLIER",
                "source_file": "vol_target_state.json; portfolio_var_cvar_summary.csv",
            },
        ])

    max_single = single["risk_budget_used_pct"].max() if not single.empty and "risk_budget_used_pct" in single.columns else np.nan
    rows.append({
        "scope": "TICKER_MAX",
        "budget_item": "Single-name tail-risk budget",
        "current_value": max_single,
        "limit_value": 1.00,
        "used_pct": max_single,
        "status": status_from_used(max_single),
        "action_if_breached": "SIZE_DOWN_OR_REDUCE_ONLY",
        "source_file": "single_name_risk_budget.csv",
    })

    max_sector = sector["cap_used_pct"].max() / 100.0 if not sector.empty and "cap_used_pct" in sector.columns else np.nan
    rows.append({
        "scope": "PORTFOLIO",
        "budget_item": "Sector concentration budget",
        "current_value": max_sector,
        "limit_value": 1.00,
        "used_pct": max_sector,
        "status": status_from_used(max_sector, warn=0.85, hard=1.00),
        "action_if_breached": "BLOCK_NEW_OR_SIZE_DOWN",
        "source_file": "sector_active_exposure.csv",
    })

    max_factor_beta = factor_df["portfolio_beta"].abs().max() if not factor_df.empty and "portfolio_beta" in factor_df.columns else np.nan
    rows.append({
        "scope": "PORTFOLIO",
        "budget_item": "Factor beta budget",
        "current_value": max_factor_beta,
        "limit_value": 1.25,
        "used_pct": max_factor_beta / 1.25 if np.isfinite(max_factor_beta) else np.nan,
        "status": status_from_used(max_factor_beta / 1.25 if np.isfinite(max_factor_beta) else np.nan, warn=0.80, hard=1.00),
        "action_if_breached": "SIZE_DOWN",
        "source_file": "factor_exposure_decomposition.csv",
    })

    worst_macro = macro["conservative_portfolio_impact"].min() if not macro.empty and "conservative_portfolio_impact" in macro.columns else np.nan
    macro_used = abs(worst_macro) / 0.15 if np.isfinite(worst_macro) else np.nan
    rows.append({
        "scope": "PORTFOLIO_STRESS",
        "budget_item": "Macro scenario loss budget",
        "current_value": worst_macro,
        "limit_value": -0.15,
        "used_pct": macro_used,
        "status": status_from_used(macro_used, warn=0.65, hard=1.00),
        "action_if_breached": "REDUCE_ONLY",
        "source_file": "macro_scenario_stress.csv",
    })

    dd_abs = float(dd.get("drawdown_abs_pct", np.nan)) if dd else np.nan
    rows.append({
        "scope": "PORTFOLIO",
        "budget_item": "Drawdown budget",
        "current_value": dd_abs,
        "limit_value": 0.10,
        "used_pct": dd_abs / 0.10 if np.isfinite(dd_abs) else np.nan,
        "status": dd.get("drawdown_action", status_from_used(dd_abs / 0.10 if np.isfinite(dd_abs) else np.nan)),
        "action_if_breached": "APPLY_DRAWDOWN_CIRCUIT",
        "source_file": "drawdown_control_state.json",
    })

    crisis_ratio = crisis["vol_increase_ratio"].max() if not crisis.empty and "vol_increase_ratio" in crisis.columns else np.nan
    crisis_status = "MISSING_DATA_REVIEW"
    if not crisis.empty and "stress_action" in crisis.columns:
        crisis_status = worst_status(crisis["stress_action"].dropna().astype(str).tolist())
    elif np.isfinite(crisis_ratio):
        crisis_status = status_from_used(crisis_ratio / 1.50, warn=0.85, hard=1.00)
    rows.append({
        "scope": "PORTFOLIO_STRESS",
        "budget_item": "Crisis-correlation volatility budget",
        "current_value": crisis_ratio,
        "limit_value": 1.50,
        "used_pct": crisis_ratio / 1.50 if np.isfinite(crisis_ratio) else np.nan,
        "status": crisis_status,
        "action_if_breached": "SIZE_DOWN",
        "source_file": "crisis_correlation_stress.csv",
    })

    max_gap = gap["estimated_gap_loss_model_account"].max() / MODEL_ACCOUNT_VALUE if not gap.empty and "estimated_gap_loss_model_account" in gap.columns else np.nan
    rows.append({
        "scope": "TICKER_MAX",
        "budget_item": "Earnings gap-loss budget",
        "current_value": max_gap,
        "limit_value": 0.010,
        "used_pct": max_gap / 0.010 if np.isfinite(max_gap) else np.nan,
        "status": status_from_used(max_gap / 0.010 if np.isfinite(max_gap) else np.nan),
        "action_if_breached": "BLOCK_NEW_OR_SIZE_DOWN",
        "source_file": "earnings_gap_down_risk.csv",
    })

    max_liq_loss = np.nan
    if not liq.empty and "estimated_liquidation_loss" in liq.columns:
        liq_loss = pd.to_numeric(liq["estimated_liquidation_loss"], errors="coerce").dropna()
        if not liq_loss.empty:
            max_liq_loss = float(liq_loss.sum()) / MODEL_ACCOUNT_VALUE
    rows.append({
        "scope": "PORTFOLIO_STRESS",
        "budget_item": "Liquidity crisis liquidation budget",
        "current_value": max_liq_loss,
        "limit_value": 0.050,
        "used_pct": max_liq_loss / 0.050 if np.isfinite(max_liq_loss) else np.nan,
        "status": status_from_used(max_liq_loss / 0.050 if np.isfinite(max_liq_loss) else np.nan),
        "action_if_breached": "SIZE_DOWN",
        "source_file": "liquidity_crisis_simulation.csv",
    })

    return pd.DataFrame(rows)


def build_master_state(budget: pd.DataFrame) -> dict:
    if budget.empty:
        statuses = ["MISSING_DATA_REVIEW"]
        master_budget = budget
    elif "scope" in budget.columns:
        master_budget = budget[~budget["scope"].astype(str).str.startswith("TICKER")].copy()
        statuses = master_budget["status"].dropna().astype(str).tolist()
    else:
        master_budget = budget
        statuses = budget["status"].dropna().astype(str).tolist()
    if not statuses:
        statuses = ["MISSING_DATA_REVIEW"]
    master_action = worst_status(statuses)
    dd = read_json_safe(ROOT / "drawdown_control_state.json", {})
    vol = read_json_safe(ROOT / "vol_target_state.json", {})
    macro = read_csv_safe(ROOT / "macro_scenario_stress.csv")
    crisis = read_csv_safe(ROOT / "crisis_correlation_stress.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")

    multipliers = [
        float(dd.get("drawdown_exposure_multiplier", 1.0) or 1.0),
        float(vol.get("vol_exposure_multiplier", 1.0) or 1.0),
        exposure_multiplier_from_action(master_action),
    ]
    if not macro.empty and "conservative_portfolio_impact" in macro.columns:
        worst_loss = float(macro["conservative_portfolio_impact"].min())
        if worst_loss <= -0.25:
            multipliers.append(0.50)
        elif worst_loss <= -0.15:
            multipliers.append(0.70)
        elif worst_loss <= -0.10:
            multipliers.append(0.85)
    if not crisis.empty and "vol_increase_ratio" in crisis.columns:
        ratio = float(crisis["vol_increase_ratio"].max())
        if ratio >= 2.0:
            multipliers.append(0.75)
        elif ratio >= 1.5:
            multipliers.append(0.85)
    if not sector.empty and "cap_status" in sector.columns:
        sector_action = worst_status(sector["cap_status"].dropna().astype(str).tolist())
        multipliers.append(exposure_multiplier_from_action(sector_action))

    final_mult = min(multipliers)
    if master_action in {"CLEAR", "OK"} and final_mult < 0.95:
        master_action = "REVIEW"
    if master_action == "REDUCE_ONLY":
        hard_current = master_budget[
            master_budget["budget_item"].astype(str).isin([
                "Portfolio 1d VaR 95%",
                "Portfolio 1d CVaR 95%",
                "Drawdown budget",
                "Liquidity crisis liquidation budget",
            ])
            & (master_budget["status"].astype(str).str.upper() == "REDUCE_ONLY")
        ]
        if hard_current.empty:
            master_action = "SIZE_DOWN"
    reasons = []
    if not master_budget.empty:
        for _, row in master_budget.iterrows():
            if action_rank(str(row.get("status", ""))) >= action_rank("REVIEW"):
                reasons.append(f"{row.get('budget_item')}: {row.get('status')}")
    return {
        "date": today_str(),
        "master_risk_action": master_action,
        "master_exposure_multiplier": round(float(final_mult), 6),
        "component_multipliers": multipliers,
        "master_reason_stack": reasons,
        "no_broker_connection": True,
        "research_only": True,
        "logic": "Risk can reduce/block. Risk cannot upgrade. Options cannot override risk. Missing data cannot improve decisions.",
    }


def build_final_gate(book: pd.DataFrame, state: dict) -> pd.DataFrame:
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    gap = read_csv_safe(ROOT / "earnings_gap_down_risk.csv")
    kelly = read_csv_safe(ROOT / "kelly_position_sizing.csv")
    liq = read_csv_safe(ROOT / "liquidity_crisis_simulation.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")

    if book.empty:
        return pd.DataFrame()

    single_map = single.set_index("ticker").to_dict("index") if not single.empty and "ticker" in single.columns else {}
    gap_map = gap.set_index("ticker").to_dict("index") if not gap.empty and "ticker" in gap.columns else {}
    kelly_map = kelly.set_index("ticker").to_dict("index") if not kelly.empty and "ticker" in kelly.columns else {}
    liq_map = liq.set_index("ticker").to_dict("index") if not liq.empty and "ticker" in liq.columns else {}
    sector_status = sector.set_index("sector")["cap_status"].to_dict() if not sector.empty and {"sector", "cap_status"}.issubset(sector.columns) else {}

    rows = []
    master_mult = float(state.get("master_exposure_multiplier", 1.0))
    master_action = str(state.get("master_risk_action", "REVIEW"))
    for _, row in book.iterrows():
        ticker = clean_ticker(row["ticker"])
        sector_name = row.get("sector", "Unknown")
        current_weight = float(row.get("weight", 0.0))
        s = single_map.get(ticker, {})
        g = gap_map.get(ticker, {})
        k = kelly_map.get(ticker, {})
        l = liq_map.get(ticker, {})

        labels = [
            master_action,
            str(s.get("single_name_action", "REVIEW")),
            str(g.get("gap_down_action", "REVIEW")),
            str(k.get("kelly_status", "REVIEW")),
            str(l.get("liquidity_crisis_status", "REVIEW")),
            str(sector_status.get(sector_name, "CLEAR")),
        ]
        final_action = worst_status(labels)

        kelly_weight = numeric(k.get("recommended_kelly_weight", np.nan))
        liquidity_max = numeric(s.get("max_liquidity_weight", np.nan))
        max_allowed = current_weight * master_mult
        if np.isfinite(kelly_weight):
            max_allowed = min(max_allowed, float(kelly_weight))
        if np.isfinite(liquidity_max):
            max_allowed = min(max_allowed, float(liquidity_max))

        if final_action in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
            recommended = min(max_allowed, current_weight * 0.50)
        elif final_action in {"SIZE_DOWN", "MISSING_DATA_REVIEW"}:
            recommended = min(max_allowed, current_weight * 0.70)
        elif final_action == "REVIEW":
            recommended = min(max_allowed, current_weight * 0.85)
        else:
            recommended = min(max_allowed, current_weight)

        reasons = []
        reason_pairs = [
            ("master", master_action),
            ("single", s.get("single_name_action", "")),
            ("earnings_gap", g.get("gap_down_action", "")),
            ("kelly", k.get("kelly_status", "")),
            ("liquidity_crisis", l.get("liquidity_crisis_status", "")),
            ("sector", sector_status.get(sector_name, "")),
        ]
        for name, label in reason_pairs:
            if action_rank(str(label)) >= action_rank("REVIEW"):
                reasons.append(f"{name}:{label}")

        rows.append({
            "ticker": ticker,
            "sector": sector_name,
            "current_action": row.get("action", ""),
            "current_weight": current_weight,
            "current_weight_pct": current_weight * 100.0,
            "master_risk_action": master_action,
            "single_name_action": s.get("single_name_action", ""),
            "earnings_gap_action": g.get("gap_down_action", ""),
            "kelly_status": k.get("kelly_status", ""),
            "liquidity_crisis_status": l.get("liquidity_crisis_status", ""),
            "sector_status": sector_status.get(sector_name, "CLEAR"),
            "final_risk_action": final_action,
            "max_allowed_weight": max_allowed,
            "recommended_risk_weight": recommended,
            "recommended_risk_weight_pct": recommended * 100.0,
            "risk_reduction_pct_of_current": 1.0 - recommended / current_weight if current_weight > 0 else np.nan,
            "reason_stack": "; ".join(reasons),
            "source_file": "institutional_risk_budget_summary.csv; Step111-117 outputs",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["risk_rank"] = out["final_risk_action"].apply(action_rank)
        out = out.sort_values(["risk_rank", "current_weight"], ascending=[False, False]).drop(columns=["risk_rank"])
    return out.reset_index(drop=True)


def write_report(budget: pd.DataFrame, var_df: pd.DataFrame, factor: pd.DataFrame, gate: pd.DataFrame, state: dict) -> None:
    sections = [
        "## Master State",
        "",
        f"- Master risk action: {state.get('master_risk_action')}",
        f"- Master exposure multiplier: {state.get('master_exposure_multiplier')}",
        "- Research only. No broker connection. No live orders.",
        "",
        "## Logic",
        "",
        "- This is the combined risk gate for Step111-117.",
        "- It can reduce, review, block, or mark reduce-only.",
        "- It cannot upgrade any ticker action.",
        "- Options are not allowed to override the risk gate.",
        "- Missing data cannot improve a decision.",
        "",
        "## Portfolio VaR / CVaR",
        "",
        df_to_markdown(var_df),
        "",
        "## Risk Budget Summary",
        "",
        df_to_markdown(budget),
        "",
        "## Factor Exposure Decomposition",
        "",
        df_to_markdown(factor, max_rows=40),
        "",
        "## Final Risk Gate",
        "",
        df_to_markdown(gate, max_rows=40),
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 118 - Institutional Risk Master Gate", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    var_df = build_portfolio_var_cvar(book)
    factor = build_factor_exposure_decomposition()
    budget = build_budget_summary(book, var_df, factor)
    state = build_master_state(budget)
    gate = build_final_gate(book, state)

    budget.to_csv(OUT_BUDGET, index=False)
    var_df.to_csv(OUT_VAR, index=False)
    factor.to_csv(OUT_FACTOR, index=False)
    gate.to_csv(OUT_GATE, index=False)
    write_json(OUT_STATE, state)
    write_report(budget, var_df, factor, gate, state)

    print(f"[step118] wrote {OUT_BUDGET.name}: {len(budget)} rows")
    print(f"[step118] wrote {OUT_VAR.name}: {len(var_df)} rows")
    print(f"[step118] wrote {OUT_FACTOR.name}: {len(factor)} rows")
    print(f"[step118] wrote {OUT_GATE.name}: {len(gate)} rows")
    print(f"[step118] wrote {OUT_STATE.name}")
    print(f"[step118] wrote {OUT_MD.name}")
    print(f"[step118] master action={state.get('master_risk_action')} multiplier={state.get('master_exposure_multiplier')}")


if __name__ == "__main__":
    main()
