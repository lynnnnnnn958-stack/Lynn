#!/usr/bin/env python3
"""
Canyon v9 Step 111 - Single-name risk budget.

Research-only. No broker connection. No live orders.

Purpose:
  Add stock-level VaR/CVaR, earnings-event risk, liquidity constraints, and
  single-name stop/circuit logic. Missing data is conservative: it can mark a
  ticker REVIEW/SIZE_DOWN, but it never upgrades a ticker.

Outputs:
  single_name_risk_budget.csv
  single_name_risk_budget_report.md
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
    get_latest_prices,
    get_returns,
    load_current_book,
    load_earnings_calendar,
    load_liquidity_proxy,
    load_options_signals,
    pct,
    source_age,
    var_cvar,
    worst_status,
    write_markdown_report,
)


OUT_CSV = ROOT / "single_name_risk_budget.csv"
OUT_MD = ROOT / "single_name_risk_budget_report.md"


def _first_numeric(row: pd.Series, cols: list[str]) -> float:
    for col in cols:
        if col in row.index:
            try:
                x = float(row[col])
                if np.isfinite(x):
                    return x
            except Exception:
                pass
    return np.nan


def _earnings_lookup(ticker: str, ec: pd.DataFrame, opt: pd.DataFrame) -> dict:
    out = {
        "earnings_days_to_event": np.nan,
        "earnings_date": "",
        "earnings_risk_label": "REVIEW",
        "implied_move": np.nan,
        "earnings_source": "missing",
    }

    rows = ec[ec["ticker"] == ticker].copy() if not ec.empty and "ticker" in ec.columns else pd.DataFrame()
    if not rows.empty:
        days_col = next((c for c in ["days_until", "days_until_earnings", "days_to_earnings"] if c in rows.columns), None)
        if days_col:
            rows[days_col] = pd.to_numeric(rows[days_col], errors="coerce")
            future = rows[rows[days_col].notna() & (rows[days_col] >= -1)].sort_values(days_col)
            use = future.iloc[0] if not future.empty else rows.sort_values(days_col, ascending=False).iloc[0]
            out["earnings_days_to_event"] = float(use[days_col]) if pd.notna(use[days_col]) else np.nan
        if "earnings_date" in rows.columns:
            val = rows.iloc[0].get("earnings_date", "")
            out["earnings_date"] = "" if pd.isna(val) else str(val)[:10]
        if "risk_flag" in rows.columns:
            flag = str(rows.iloc[0].get("risk_flag", "")).upper()
            if flag in {"HIGH", "MEDIUM", "LOW"}:
                out["earnings_risk_label"] = {"HIGH": "SIZE_DOWN", "MEDIUM": "REVIEW", "LOW": "CLEAR"}[flag]
        out["earnings_source"] = "earnings_calendar.csv"

    opt_rows = opt[opt["ticker"] == ticker].copy() if not opt.empty and "ticker" in opt.columns else pd.DataFrame()
    if not opt_rows.empty:
        use = opt_rows.iloc[0]
        opt_days = _first_numeric(use, ["days_to_earnings"])
        if np.isfinite(opt_days) and (not np.isfinite(out["earnings_days_to_event"]) or opt_days >= 0):
            out["earnings_days_to_event"] = opt_days
            out["earnings_source"] = "options_signals.csv"
        atm_iv = _first_numeric(use, ["atm_iv"])
        days = out["earnings_days_to_event"]
        horizon = max(7.0, min(float(days), 45.0)) if np.isfinite(days) and days > 0 else 7.0
        if np.isfinite(atm_iv) and atm_iv > 0:
            out["implied_move"] = float(atm_iv) * np.sqrt(horizon / 365.0)

    days = out["earnings_days_to_event"]
    implied_move = out["implied_move"]
    if not np.isfinite(days):
        base_label = "REVIEW"
    elif days < 0:
        # Stale calendar rows are a data-quality issue, not an imminent event.
        base_label = "REVIEW"
    elif days <= 1:
        base_label = "BLOCK_NEW"
    elif days <= 5:
        base_label = "SIZE_DOWN"
    elif days <= 14:
        base_label = "REVIEW"
    else:
        base_label = "CLEAR"
    out["earnings_risk_label"] = base_label

    if np.isfinite(implied_move) and implied_move >= 0.15:
        if np.isfinite(days) and 0 <= days <= 14:
            out["earnings_risk_label"] = worst_status([out["earnings_risk_label"], "SIZE_DOWN"])
        elif np.isfinite(days) and days > 14:
            out["earnings_risk_label"] = worst_status([out["earnings_risk_label"], "REVIEW"])
    return out


def _liquidity_lookup(ticker: str, weight: float, liq: pd.DataFrame) -> dict:
    out = {
        "adv_dollar": np.nan,
        "days_to_liquidate": np.nan,
        "liquidity_label": "MISSING_DATA_REVIEW",
        "max_liquidity_weight": 0.02,
        "liquidity_source": "missing",
    }
    if liq.empty or "ticker" not in liq.columns:
        return out
    rows = liq[liq["ticker"] == ticker]
    if rows.empty:
        return out
    row = rows.iloc[0]
    adv = _first_numeric(row, ["avg_20d_dollar_volume"])
    out["adv_dollar"] = adv
    out["liquidity_source"] = "intraday_liquidity_proxy.csv"
    if not np.isfinite(adv) or adv <= 0:
        return out

    position_dollar = float(weight) * MODEL_ACCOUNT_VALUE
    out["days_to_liquidate"] = position_dollar / max(adv * 0.10, 1.0)
    if adv >= 2_000_000_000:
        out["liquidity_label"] = "CLEAR"
        out["max_liquidity_weight"] = 0.15
    elif adv >= 500_000_000:
        out["liquidity_label"] = "CLEAR"
        out["max_liquidity_weight"] = 0.10
    elif adv >= 100_000_000:
        out["liquidity_label"] = "REVIEW"
        out["max_liquidity_weight"] = 0.05
    elif adv >= 25_000_000:
        out["liquidity_label"] = "SIZE_DOWN"
        out["max_liquidity_weight"] = 0.03
    else:
        out["liquidity_label"] = "BLOCK_NEW"
        out["max_liquidity_weight"] = 0.01

    if out["days_to_liquidate"] > 5:
        out["liquidity_label"] = worst_status([out["liquidity_label"], "BLOCK_NEW"])
    elif out["days_to_liquidate"] > 2:
        out["liquidity_label"] = worst_status([out["liquidity_label"], "SIZE_DOWN"])
    if weight > out["max_liquidity_weight"]:
        out["liquidity_label"] = worst_status([out["liquidity_label"], "SIZE_DOWN"])
    return out


def build_single_name_risk_budget() -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return pd.DataFrame()

    tickers = book["ticker"].apply(clean_ticker).tolist()
    rets = get_returns(tickers, lookback=504)
    latest_prices = get_latest_prices(tickers)
    ec = load_earnings_calendar()
    opt = load_options_signals()
    liq = load_liquidity_proxy(tickers)

    rows = []
    for _, item in book.iterrows():
        ticker = clean_ticker(item["ticker"])
        weight = float(item.get("weight", 0.0))
        series = rets[ticker].dropna() if ticker in rets.columns else pd.Series(dtype=float)
        var_95_1d, cvar_95_1d = var_cvar(series, alpha=0.95)
        var_99_1d, cvar_99_1d = var_cvar(series, alpha=0.99)
        ann_vol = annualized_vol(series)
        daily_vol = float(series.std(ddof=1)) if len(series) >= 20 else np.nan

        if len(series) < 60:
            price_risk_label = "MISSING_DATA_REVIEW"
        elif np.isfinite(cvar_95_1d) and cvar_95_1d >= 0.09:
            price_risk_label = "REDUCE_ONLY"
        elif np.isfinite(cvar_95_1d) and cvar_95_1d >= 0.06:
            price_risk_label = "SIZE_DOWN"
        elif np.isfinite(ann_vol) and ann_vol >= 0.65:
            price_risk_label = "SIZE_DOWN"
        else:
            price_risk_label = "CLEAR"

        earn = _earnings_lookup(ticker, ec, opt)
        liq_row = _liquidity_lookup(ticker, weight, liq)

        implied_move = earn["implied_move"]
        stop_pct = np.nan
        if np.isfinite(daily_vol):
            stop_pct = max(0.06, min(0.25, daily_vol * 2.5))
        if np.isfinite(implied_move) and np.isfinite(earn["earnings_days_to_event"]) and earn["earnings_days_to_event"] <= 7:
            stop_pct = max(stop_pct if np.isfinite(stop_pct) else 0.06, min(0.25, implied_move * 0.8))

        price = latest_prices.get(ticker, np.nan)
        stop_level = price * (1.0 - stop_pct) if np.isfinite(price) and np.isfinite(stop_pct) else np.nan
        position_cvar_95_1d = weight * cvar_95_1d if np.isfinite(cvar_95_1d) else np.nan
        risk_budget_limit = 0.0030
        risk_budget_used_pct = position_cvar_95_1d / risk_budget_limit if np.isfinite(position_cvar_95_1d) else np.nan
        budget_label = "CLEAR"
        if not np.isfinite(risk_budget_used_pct):
            budget_label = "MISSING_DATA_REVIEW"
        elif risk_budget_used_pct > 2.0:
            budget_label = "REDUCE_ONLY"
        elif risk_budget_used_pct > 1.0:
            budget_label = "SIZE_DOWN"

        final_action = worst_status([
            price_risk_label,
            earn["earnings_risk_label"],
            liq_row["liquidity_label"],
            budget_label,
        ])
        if final_action == "OK":
            final_action = "CLEAR"

        rows.append({
            "ticker": ticker,
            "weight": round(weight, 6),
            "weight_pct": round(weight * 100.0, 2),
            "alpha_score": item.get("alpha_score", np.nan),
            "action": item.get("action", ""),
            "sector": item.get("sector", "Unknown"),
            "theme": item.get("theme", "Unknown"),
            "daily_vol": daily_vol,
            "annual_vol": ann_vol,
            "var_95_1d": var_95_1d,
            "cvar_95_1d": cvar_95_1d,
            "var_99_1d": var_99_1d,
            "cvar_99_1d": cvar_99_1d,
            "var_95_5d": var_95_1d * np.sqrt(5) if np.isfinite(var_95_1d) else np.nan,
            "cvar_95_5d": cvar_95_1d * np.sqrt(5) if np.isfinite(cvar_95_1d) else np.nan,
            "position_cvar_95_1d": position_cvar_95_1d,
            "risk_budget_limit": risk_budget_limit,
            "risk_budget_used_pct": risk_budget_used_pct,
            "price_risk_label": price_risk_label,
            "earnings_days_to_event": earn["earnings_days_to_event"],
            "earnings_date": earn["earnings_date"],
            "implied_move": implied_move,
            "earnings_risk_label": earn["earnings_risk_label"],
            "adv_dollar": liq_row["adv_dollar"],
            "days_to_liquidate": liq_row["days_to_liquidate"],
            "liquidity_label": liq_row["liquidity_label"],
            "max_liquidity_weight": liq_row["max_liquidity_weight"],
            "single_name_stop_pct": stop_pct,
            "single_name_stop_level": stop_level,
            "single_name_action": final_action,
            "source_file": item.get("source_file", ""),
            "source_detail": ";".join([
                str(item.get("source_file", "")),
                earn["earnings_source"],
                liq_row["liquidity_source"],
                "sp500_price_cache.csv/backtest_price_cache.csv",
            ]),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["single_name_action", "risk_budget_used_pct", "cvar_95_1d"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def write_report(df: pd.DataFrame) -> None:
    if df.empty:
        write_markdown_report(OUT_MD, "Canyon v9 Step 111 - Single-name Risk Budget", [
            "No current research book was found. Run Step 87 first.",
        ])
        return

    counts = df["single_name_action"].value_counts().to_dict()
    top_cols = [
        "ticker", "weight_pct", "single_name_action", "cvar_95_1d",
        "earnings_days_to_event", "implied_move", "liquidity_label",
        "risk_budget_used_pct",
    ]
    top = df[[c for c in top_cols if c in df.columns]].head(15)
    sections = [
        "## Summary",
        "",
        f"- Rows: {len(df)}",
        f"- Actions: {counts}",
        f"- Price source age: {source_age(ROOT / 'sp500_price_cache.csv')}",
        f"- Earnings source age: {source_age(ROOT / 'earnings_calendar.csv')}",
        f"- Options source age: {source_age(ROOT / 'options_signals.csv')}",
        "",
        "## Logic",
        "",
        "- Missing price, earnings, or liquidity data becomes REVIEW/SIZE_DOWN, never CLEAR.",
        "- Single-name risk can reduce, block, or force manual review; it cannot upgrade an idea.",
        "- Earnings gap risk is evaluated before any paper action.",
        "",
        "## Highest-risk rows",
        "",
        df_to_markdown(top),
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 111 - Single-name Risk Budget", sections)


def main() -> None:
    df = build_single_name_risk_budget()
    df.to_csv(OUT_CSV, index=False)
    write_report(df)
    print(f"[step111] wrote {OUT_CSV.name}: {len(df)} rows")
    print(f"[step111] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
