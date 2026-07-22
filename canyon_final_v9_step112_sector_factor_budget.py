#!/usr/bin/env python3
"""
Canyon v9 Step 112 - Sector, industry, and factor risk budget.

Research-only. No broker connection. No live orders.

Outputs:
  sector_active_exposure.csv
  theme_factor_exposure.csv
  sector_factor_budget_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ETF_COMPONENT_MAP,
    FACTOR_PROXIES,
    ROOT,
    SECTOR_BENCHMARK_WEIGHTS,
    beta_to_factor,
    clean_ticker,
    df_to_markdown,
    get_returns,
    infer_theme,
    load_current_book,
    pct,
    portfolio_return_series,
    source_age,
    worst_status,
    write_markdown_report,
)


OUT_SECTOR = ROOT / "sector_active_exposure.csv"
OUT_THEME = ROOT / "theme_factor_exposure.csv"
OUT_MD = ROOT / "sector_factor_budget_report.md"

SECTOR_CAP = 0.35
ACTIVE_OVERWEIGHT_CAP = 0.12
THEME_CAPS = {
    "AI / Semiconductors": 0.25,
    "Mega-cap Technology": 0.40,
    "Technology": 0.40,
    "Growth / Nasdaq": 0.35,
    "Broad Market": 1.00,
}


def _benchmark_weight(sector: str) -> float:
    return float(SECTOR_BENCHMARK_WEIGHTS.get(str(sector), 0.03))


def build_sector_active_exposure(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    work = book.copy()
    work["sector"] = work["sector"].fillna("Unknown")
    rows = []
    for sector, grp in work.groupby("sector", dropna=False):
        weight = float(grp["weight"].sum())
        bench = _benchmark_weight(str(sector))
        active = weight - bench
        cap_used = weight / SECTOR_CAP
        tickers = grp.sort_values("weight", ascending=False)["ticker"].head(8).tolist()
        labels = []
        if weight > SECTOR_CAP:
            labels.append("BLOCK_NEW")
        elif cap_used > 0.85:
            labels.append("SIZE_DOWN")
        elif active > ACTIVE_OVERWEIGHT_CAP:
            labels.append("REVIEW")
        else:
            labels.append("CLEAR")
        rows.append({
            "sector": sector,
            "portfolio_weight": weight,
            "portfolio_weight_pct": weight * 100.0,
            "benchmark_weight": bench,
            "benchmark_weight_pct": bench * 100.0,
            "active_weight": active,
            "active_weight_pct": active * 100.0,
            "sector_cap": SECTOR_CAP,
            "cap_used_pct": cap_used * 100.0,
            "cap_status": worst_status(labels),
            "top_tickers": ", ".join(tickers),
            "source_file": ", ".join(sorted(work["source_file"].dropna().unique())),
        })
    return pd.DataFrame(rows).sort_values("portfolio_weight", ascending=False).reset_index(drop=True)


def _theme_from_row(row: pd.Series) -> str:
    ticker = clean_ticker(row.get("ticker", ""))
    sector = str(row.get("sector", ""))
    return infer_theme(ticker, sector)


def build_theme_factor_exposure(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    work = book.copy()
    work["theme"] = work.apply(_theme_from_row, axis=1)
    rows = []

    for theme, grp in work.groupby("theme", dropna=False):
        weight = float(grp["weight"].sum())
        cap = float(THEME_CAPS.get(str(theme), 0.30))
        status = "CLEAR"
        if weight > cap:
            status = "BLOCK_NEW"
        elif weight > cap * 0.85:
            status = "SIZE_DOWN"
        elif weight > cap * 0.70:
            status = "REVIEW"
        etfs = [t for t in grp["ticker"].tolist() if t in ETF_COMPONENT_MAP]
        components = [t for t in grp["ticker"].tolist() if t not in ETF_COMPONENT_MAP]
        overlap_status = "CLEAR"
        if etfs and components and weight > 0.20:
            overlap_status = "REVIEW"
        rows.append({
            "exposure_type": "theme",
            "factor_or_theme": theme,
            "portfolio_weight": weight,
            "portfolio_weight_pct": weight * 100.0,
            "cap": cap,
            "cap_used_pct": weight / cap * 100.0 if cap > 0 else np.nan,
            "exposure_status": worst_status([status, overlap_status]),
            "etf_overlap_flag": bool(etfs and components),
            "etfs_in_theme": ", ".join(etfs),
            "top_tickers": ", ".join(grp.sort_values("weight", ascending=False)["ticker"].head(10).tolist()),
            "source_file": ", ".join(sorted(work["source_file"].dropna().unique())),
        })

    p_ret = portfolio_return_series(work, lookback=504)
    factor_tickers = list(FACTOR_PROXIES.values())
    factor_returns = get_returns(factor_tickers, lookback=504)
    for factor_name, proxy in FACTOR_PROXIES.items():
        beta = np.nan
        if not p_ret.empty and proxy in factor_returns.columns:
            beta = beta_to_factor(p_ret, factor_returns[proxy])
        abs_beta = abs(beta) if np.isfinite(beta) else np.nan
        if not np.isfinite(beta):
            status = "MISSING_DATA_REVIEW"
        elif abs_beta >= 1.35:
            status = "SIZE_DOWN"
        elif abs_beta >= 1.05:
            status = "REVIEW"
        else:
            status = "CLEAR"
        rows.append({
            "exposure_type": "market_factor",
            "factor_or_theme": factor_name,
            "proxy": proxy,
            "portfolio_weight": np.nan,
            "portfolio_weight_pct": np.nan,
            "cap": np.nan,
            "cap_used_pct": np.nan,
            "factor_beta": beta,
            "exposure_status": status,
            "etf_overlap_flag": False,
            "etfs_in_theme": "",
            "top_tickers": "",
            "source_file": "sp500_price_cache.csv/backtest_price_cache.csv",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["exposure_type", "portfolio_weight"], ascending=[True, False]).reset_index(drop=True)


def write_report(sector_df: pd.DataFrame, theme_df: pd.DataFrame) -> None:
    if sector_df.empty and theme_df.empty:
        write_markdown_report(OUT_MD, "Canyon v9 Step 112 - Sector and Factor Budget", [
            "No current research book was found. Run Step 87 first.",
        ])
        return
    sector_flags = sector_df["cap_status"].value_counts().to_dict() if not sector_df.empty else {}
    theme_flags = theme_df["exposure_status"].value_counts().to_dict() if not theme_df.empty else {}
    sections = [
        "## Summary",
        "",
        f"- Sector statuses: {sector_flags}",
        f"- Theme/factor statuses: {theme_flags}",
        f"- Price source age: {source_age(ROOT / 'sp500_price_cache.csv')}",
        "",
        "## Logic",
        "",
        "- Sector cap is 35% by default.",
        "- Active overweight above 12 percentage points is flagged for review.",
        "- ETF plus component overlap is flagged so QQQ/XLK/SMH plus many components cannot hide as diversified exposure.",
        "- Factor beta is proxy-based and should reduce confidence when missing.",
        "",
        "## Sector exposure",
        "",
        df_to_markdown(sector_df, max_rows=20) if not sector_df.empty else "No sector rows.",
        "",
        "## Theme and factor exposure",
        "",
        df_to_markdown(theme_df, max_rows=30) if not theme_df.empty else "No theme rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 112 - Sector and Factor Budget", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    sector_df = build_sector_active_exposure(book)
    theme_df = build_theme_factor_exposure(book)
    sector_df.to_csv(OUT_SECTOR, index=False)
    theme_df.to_csv(OUT_THEME, index=False)
    write_report(sector_df, theme_df)
    print(f"[step112] wrote {OUT_SECTOR.name}: {len(sector_df)} rows")
    print(f"[step112] wrote {OUT_THEME.name}: {len(theme_df)} rows")
    print(f"[step112] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
