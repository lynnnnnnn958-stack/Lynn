#!/usr/bin/env python3
"""
Canyon v9 Step 117 - Extreme-market protection.

Research-only. No broker connection. No live orders.

This module creates a research risk plan for tail events. It does not place or
recommend live trades. It identifies when protection, de-risking, or manual
review is required.

Outputs:
  tail_hedge_budget.csv
  liquidity_crisis_simulation.csv
  earnings_gap_down_risk.csv
  crisis_correlation_override.csv
  extreme_market_protection_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    load_liquidity_proxy,
    read_csv_safe,
    read_json_safe,
    worst_status,
    write_markdown_report,
)


OUT_TAIL = ROOT / "tail_hedge_budget.csv"
OUT_LIQ = ROOT / "liquidity_crisis_simulation.csv"
OUT_GAP = ROOT / "earnings_gap_down_risk.csv"
OUT_CORR = ROOT / "crisis_correlation_override.csv"
OUT_MD = ROOT / "extreme_market_protection_report.md"


def _state_action() -> tuple[str, dict]:
    dd = read_json_safe(ROOT / "drawdown_control_state.json", {})
    vol = read_json_safe(ROOT / "vol_target_state.json", {})
    macro = read_csv_safe(ROOT / "macro_scenario_stress.csv")
    actions = []
    if dd:
        actions.append(str(dd.get("drawdown_action", "CLEAR")))
    if vol:
        actions.append(str(vol.get("vol_action", "CLEAR")))
    if not macro.empty and "scenario_action" in macro.columns:
        actions.extend(macro["scenario_action"].dropna().astype(str).tolist())
    return worst_status(actions or ["CLEAR"]), {"drawdown": dd, "vol": vol}


def build_tail_hedge_budget(book: pd.DataFrame) -> pd.DataFrame:
    state, meta = _state_action()
    gross = float(book["weight"].sum()) if not book.empty else 0.0
    risk_level = state
    if risk_level in {"REDUCE_ONLY", "BLOCK_NEW", "BLOCKED"}:
        budget_pct = 0.020
    elif risk_level == "SIZE_DOWN":
        budget_pct = 0.015
    elif risk_level == "REVIEW":
        budget_pct = 0.010
    else:
        budget_pct = 0.005
    budget_dollars = budget_pct * MODEL_ACCOUNT_VALUE
    rows = [
        {
            "hedge_sleeve": "Index crash insurance",
            "proxy": "SPY put spread / put proxy",
            "research_budget_pct": budget_pct,
            "research_budget_dollars": budget_dollars * 0.50,
            "trigger_condition": "Portfolio stress scenario <= -10% or drawdown circuit active",
            "risk_state": risk_level,
            "allowed_action": "RESEARCH_ONLY_NO_ORDER",
            "source_file": "macro_scenario_stress.csv; drawdown_control_state.json",
        },
        {
            "hedge_sleeve": "Growth / Nasdaq insurance",
            "proxy": "QQQ put spread / put proxy",
            "research_budget_pct": budget_pct,
            "research_budget_dollars": budget_dollars * 0.30,
            "trigger_condition": "High QQQ/growth beta or AI unwind scenario",
            "risk_state": risk_level,
            "allowed_action": "RESEARCH_ONLY_NO_ORDER",
            "source_file": "portfolio_macro_sensitivity.csv; portfolio_beta_report.csv",
        },
        {
            "hedge_sleeve": "Semiconductor crowding insurance",
            "proxy": "SMH or SOXX put spread proxy",
            "research_budget_pct": budget_pct,
            "research_budget_dollars": budget_dollars * 0.20,
            "trigger_condition": "AI / semiconductor theme weight above cap or stress scenario active",
            "risk_state": risk_level,
            "allowed_action": "RESEARCH_ONLY_NO_ORDER",
            "source_file": "theme_factor_exposure.csv; crisis_correlation_stress.csv",
        },
    ]
    out = pd.DataFrame(rows)
    out["gross_research_exposure"] = gross
    out["note"] = "Planning budget only. No broker connection and no live order path."
    return out


def build_liquidity_crisis_simulation(book: pd.DataFrame) -> pd.DataFrame:
    tickers = book["ticker"].apply(clean_ticker).tolist() if not book.empty and "ticker" in book.columns else []
    liq = load_liquidity_proxy(tickers)
    rows = []
    for _, row in book.iterrows():
        ticker = clean_ticker(row["ticker"])
        weight = float(row.get("weight", 0.0))
        notional = weight * MODEL_ACCOUNT_VALUE
        adv = np.nan
        label = "MISSING"
        if not liq.empty and "ticker" in liq.columns:
            match = liq[liq["ticker"] == ticker]
            if not match.empty:
                adv = pd.to_numeric(pd.Series([match.iloc[0].get("avg_20d_dollar_volume", np.nan)]), errors="coerce").iloc[0]
                label = str(match.iloc[0].get("liquidity_label", "UNKNOWN"))
        daily_sale_capacity = adv * 0.10 if np.isfinite(adv) else np.nan
        days_to_exit = notional / daily_sale_capacity if np.isfinite(daily_sale_capacity) and daily_sale_capacity > 0 else np.nan
        if not np.isfinite(days_to_exit):
            shock = np.nan
            status = "MISSING_DATA_REVIEW"
            estimated_loss = np.nan
        elif days_to_exit > 5:
            shock = -0.15
            status = "BLOCK_NEW"
            estimated_loss = notional * abs(shock)
        elif days_to_exit > 2:
            shock = -0.08
            status = "SIZE_DOWN"
            estimated_loss = notional * abs(shock)
        elif days_to_exit > 1:
            shock = -0.04
            status = "REVIEW"
            estimated_loss = notional * abs(shock)
        else:
            shock = -0.01
            status = "CLEAR"
            estimated_loss = notional * abs(shock)
        rows.append({
            "ticker": ticker,
            "weight": weight,
            "notional_model_account": notional,
            "adv_dollar": adv,
            "liquidity_label": label,
            "daily_sale_capacity_10pct_adv": daily_sale_capacity,
            "days_to_exit_at_10pct_adv": days_to_exit,
            "liquidity_crisis_price_shock": shock,
            "estimated_liquidation_loss": estimated_loss,
            "liquidity_crisis_status": status,
            "source_file": "intraday_liquidity_proxy.csv; current book",
        })
    return pd.DataFrame(rows).sort_values("estimated_liquidation_loss", ascending=False).reset_index(drop=True)


def build_earnings_gap_down_risk(book: pd.DataFrame) -> pd.DataFrame:
    single = read_csv_safe(ROOT / "single_name_risk_budget.csv")
    if single.empty:
        return pd.DataFrame()
    rows = []
    keep = [c for c in [
        "ticker", "weight", "earnings_days_to_event", "implied_move",
        "earnings_risk_label", "single_name_action", "source_detail",
    ] if c in single.columns]
    work = single[keep].copy()
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker", ""))
        weight = pd.to_numeric(pd.Series([row.get("weight", np.nan)]), errors="coerce").iloc[0]
        days = pd.to_numeric(pd.Series([row.get("earnings_days_to_event", np.nan)]), errors="coerce").iloc[0]
        implied = pd.to_numeric(pd.Series([row.get("implied_move", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(implied):
            implied = 0.08
            data_status = "IMPLIED_MOVE_MISSING_REVIEW"
        else:
            data_status = "OK"
        gap_loss = MODEL_ACCOUNT_VALUE * weight * implied if np.isfinite(weight) else np.nan
        if not np.isfinite(days):
            action = "REVIEW"
        elif days < 0:
            action = "REVIEW"
        elif days <= 1:
            action = "BLOCK_NEW"
        elif np.isfinite(days) and days <= 5:
            action = "SIZE_DOWN"
        elif np.isfinite(days) and days <= 14:
            action = "REVIEW"
        else:
            action = "CLEAR"
        if implied >= 0.15 and np.isfinite(days) and 0 <= days <= 14:
            action = worst_status([action, "SIZE_DOWN"])
        elif implied >= 0.25 and np.isfinite(days) and days > 14:
            action = worst_status([action, "REVIEW"])
        rows.append({
            "ticker": ticker,
            "weight": weight,
            "earnings_days_to_event": days,
            "implied_move_or_fallback": implied,
            "estimated_gap_loss_model_account": gap_loss,
            "earnings_risk_label": row.get("earnings_risk_label", ""),
            "single_name_action": row.get("single_name_action", ""),
            "gap_down_action": worst_status([action, str(row.get("single_name_action", "CLEAR"))]),
            "data_status": data_status,
            "source_file": "single_name_risk_budget.csv",
        })
    return pd.DataFrame(rows).sort_values("estimated_gap_loss_model_account", ascending=False).reset_index(drop=True)


def build_crisis_correlation_override(book: pd.DataFrame) -> pd.DataFrame:
    crisis = read_csv_safe(ROOT / "crisis_correlation_stress.csv")
    sector = read_csv_safe(ROOT / "sector_active_exposure.csv")
    theme = read_csv_safe(ROOT / "theme_factor_exposure.csv")
    actions = []
    if not crisis.empty and "stress_action" in crisis.columns:
        actions.extend(crisis["stress_action"].dropna().astype(str).tolist())
    if not sector.empty and "cap_status" in sector.columns:
        actions.extend(sector["cap_status"].dropna().astype(str).tolist())
    if not theme.empty and "exposure_status" in theme.columns:
        actions.extend(theme["exposure_status"].dropna().astype(str).tolist())
    action = worst_status(actions or ["CLEAR"])
    gross = float(book["weight"].sum()) if not book.empty else 0.0
    return pd.DataFrame([{
        "override_name": "crisis correlation breakdown",
        "assumption": "Diversification benefit is reduced; pairwise correlations move toward 1.0 in stress.",
        "normal_gross_exposure": gross,
        "crisis_correlation_assumption": 1.0,
        "minimum_correlation_floor": 0.85,
        "override_action": action,
        "allowed_action": "Reduce risk or review only; no upgrade allowed.",
        "source_file": "crisis_correlation_stress.csv; sector_active_exposure.csv; theme_factor_exposure.csv",
    }])


def write_report(tail: pd.DataFrame, liq: pd.DataFrame, gap: pd.DataFrame, corr: pd.DataFrame) -> None:
    sections = [
        "## Summary",
        "",
        f"- Tail hedge planning rows: {len(tail)}",
        f"- Liquidity crisis rows: {len(liq)}",
        f"- Earnings gap rows: {len(gap)}",
        f"- Correlation override rows: {len(corr)}",
        "",
        "## Logic",
        "",
        "- This is a research protection plan only; it does not create orders.",
        "- Tail hedge budget is a planning allowance, not a trade ticket.",
        "- Liquidity crisis assumes only 10% of ADV can be sold per day.",
        "- Earnings gap risk assumes stops cannot protect against overnight moves.",
        "",
        "## Tail hedge budget",
        "",
        df_to_markdown(tail) if not tail.empty else "No tail rows.",
        "",
        "## Liquidity crisis",
        "",
        df_to_markdown(liq, max_rows=20) if not liq.empty else "No liquidity rows.",
        "",
        "## Earnings gap-down risk",
        "",
        df_to_markdown(gap, max_rows=20) if not gap.empty else "No earnings gap rows.",
        "",
        "## Correlation override",
        "",
        df_to_markdown(corr) if not corr.empty else "No correlation override rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 117 - Extreme Market Protection", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    tail = build_tail_hedge_budget(book)
    liq = build_liquidity_crisis_simulation(book)
    gap = build_earnings_gap_down_risk(book)
    corr = build_crisis_correlation_override(book)
    tail.to_csv(OUT_TAIL, index=False)
    liq.to_csv(OUT_LIQ, index=False)
    gap.to_csv(OUT_GAP, index=False)
    corr.to_csv(OUT_CORR, index=False)
    write_report(tail, liq, gap, corr)
    print(f"[step117] wrote {OUT_TAIL.name}: {len(tail)} rows")
    print(f"[step117] wrote {OUT_LIQ.name}: {len(liq)} rows")
    print(f"[step117] wrote {OUT_GAP.name}: {len(gap)} rows")
    print(f"[step117] wrote {OUT_CORR.name}: {len(corr)} rows")
    print(f"[step117] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
