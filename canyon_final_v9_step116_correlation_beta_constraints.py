#!/usr/bin/env python3
"""
Canyon v9 Step 116 - Holdings correlation, portfolio beta, and crisis constraints.

Research-only. No broker connection. No live orders.

Outputs:
  holdings_correlation_matrix.csv
  portfolio_beta_report.csv
  crisis_correlation_stress.csv
  sector_overlap_analysis.csv
  correlation_beta_constraints_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    FACTOR_PROXIES,
    ROOT,
    beta_to_factor,
    clean_ticker,
    df_to_markdown,
    get_returns,
    load_current_book,
    portfolio_return_series,
    portfolio_vol,
    source_age,
    worst_status,
    write_markdown_report,
)


OUT_CORR = ROOT / "holdings_correlation_matrix.csv"
OUT_BETA = ROOT / "portfolio_beta_report.csv"
OUT_CRISIS = ROOT / "crisis_correlation_stress.csv"
OUT_OVERLAP = ROOT / "sector_overlap_analysis.csv"
OUT_MD = ROOT / "correlation_beta_constraints_report.md"

CORR_WARNING = 0.75
CRISIS_CORR_FLOOR = 0.85


def build_corr_matrix(book: pd.DataFrame) -> pd.DataFrame:
    tickers = book["ticker"].apply(clean_ticker).tolist() if not book.empty else []
    rets = get_returns(tickers, lookback=252)
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame()
    return rets.corr().round(4)


def build_beta_report(book: pd.DataFrame) -> pd.DataFrame:
    p_ret = portfolio_return_series(book, lookback=756)
    factor_returns = get_returns(list(FACTOR_PROXIES.values()), lookback=756)
    rows = []
    for factor, proxy in FACTOR_PROXIES.items():
        beta = np.nan
        if not p_ret.empty and proxy in factor_returns.columns:
            beta = beta_to_factor(p_ret, factor_returns[proxy])
        if not np.isfinite(beta):
            status = "MISSING_DATA_REVIEW"
        elif abs(beta) > 1.30:
            status = "SIZE_DOWN"
        elif abs(beta) > 0.95:
            status = "REVIEW"
        else:
            status = "CLEAR"
        rows.append({
            "factor": factor,
            "proxy": proxy,
            "portfolio_beta": beta,
            "abs_beta": abs(beta) if np.isfinite(beta) else np.nan,
            "beta_status": status,
            "source_file": "sp500_price_cache.csv/backtest_price_cache.csv",
        })
    return pd.DataFrame(rows)


def build_crisis_stress(book: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    tickers = book["ticker"].apply(clean_ticker).tolist()
    rets = get_returns(tickers, lookback=252)
    common = [t for t in tickers if t in rets.columns]
    if len(common) < 2:
        return pd.DataFrame([{
            "stress_name": "missing correlation data",
            "normal_annual_vol": np.nan,
            "crisis_annual_vol": np.nan,
            "vol_increase_ratio": np.nan,
            "high_corr_pair_count": np.nan,
            "max_pair_corr": np.nan,
            "stress_action": "MISSING_DATA_REVIEW",
            "source_file": "sp500_price_cache.csv/backtest_price_cache.csv",
        }])

    weights = book.set_index("ticker").loc[common, "weight"].astype(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    sub = rets[common].dropna(how="all")
    cov = sub.cov() * 252.0
    w = weights.values.reshape(-1, 1)
    normal_var = float((w.T @ cov.values @ w).item())
    normal_vol = float(np.sqrt(max(normal_var, 0.0)))

    vols = np.sqrt(np.diag(cov.values))
    crisis_corr = np.full((len(common), len(common)), CRISIS_CORR_FLOOR)
    np.fill_diagonal(crisis_corr, 1.0)
    crisis_cov = np.outer(vols, vols) * crisis_corr
    crisis_var = float((w.T @ crisis_cov @ w).item())
    crisis_vol = float(np.sqrt(max(crisis_var, 0.0)))
    vol_ratio = crisis_vol / normal_vol if normal_vol > 0 else np.nan

    pair_corrs = []
    if not corr.empty:
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                val = corr.loc[a, b]
                if np.isfinite(val):
                    pair_corrs.append(float(val))
    high_count = int(sum(abs(x) >= CORR_WARNING for x in pair_corrs))
    max_corr = max(pair_corrs, key=lambda x: abs(x)) if pair_corrs else np.nan

    status = "CLEAR"
    if high_count >= 5 or (np.isfinite(vol_ratio) and vol_ratio >= 1.50):
        status = "SIZE_DOWN"
    elif high_count > 0 or (np.isfinite(vol_ratio) and vol_ratio >= 1.20):
        status = "REVIEW"

    return pd.DataFrame([{
        "stress_name": "crisis correlation floor 0.85",
        "normal_annual_vol": normal_vol,
        "crisis_annual_vol": crisis_vol,
        "vol_increase_ratio": vol_ratio,
        "high_corr_pair_count": high_count,
        "max_pair_corr": max_corr,
        "stress_action": status,
        "source_file": "holdings_correlation_matrix.csv; price cache",
    }])


def build_sector_overlap(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    rows = []
    for label_col in ["sector", "theme"]:
        for label, grp in book.groupby(label_col, dropna=False):
            weight = float(grp["weight"].sum())
            count = int(len(grp))
            top_tickers = ", ".join(grp.sort_values("weight", ascending=False)["ticker"].head(10).tolist())
            if weight >= 0.50 or count >= 12:
                status = "SIZE_DOWN"
            elif weight >= 0.35 or count >= 8:
                status = "REVIEW"
            else:
                status = "CLEAR"
            rows.append({
                "overlap_type": label_col,
                "bucket": label,
                "ticker_count": count,
                "portfolio_weight": weight,
                "portfolio_weight_pct": weight * 100.0,
                "overlap_status": status,
                "top_tickers": top_tickers,
                "source_file": ", ".join(sorted(book["source_file"].dropna().unique())),
            })
    return pd.DataFrame(rows).sort_values("portfolio_weight", ascending=False).reset_index(drop=True)


def write_report(corr: pd.DataFrame, beta: pd.DataFrame, crisis: pd.DataFrame, overlap: pd.DataFrame) -> None:
    sections = [
        "## Summary",
        "",
        f"- Correlation matrix size: {corr.shape[0]} x {corr.shape[1]}",
        f"- Beta rows: {len(beta)}",
        f"- Crisis rows: {len(crisis)}",
        f"- Price source age: {source_age(ROOT / 'sp500_price_cache.csv')}",
        "",
        "## Logic",
        "",
        "- Holdings correlation is not the same as signal correlation.",
        "- Crisis mode assumes correlations can jump toward 0.85.",
        "- High beta or high pair correlation can reduce exposure but cannot upgrade an action.",
        "",
        "## Portfolio beta",
        "",
        df_to_markdown(beta) if not beta.empty else "No beta rows.",
        "",
        "## Crisis stress",
        "",
        df_to_markdown(crisis) if not crisis.empty else "No crisis rows.",
        "",
        "## Sector and theme overlap",
        "",
        df_to_markdown(overlap, max_rows=30) if not overlap.empty else "No overlap rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 116 - Correlation and Beta Constraints", sections)


def main() -> None:
    book = load_current_book(prefer_filtered=True)
    corr = build_corr_matrix(book)
    beta = build_beta_report(book)
    crisis = build_crisis_stress(book, corr)
    overlap = build_sector_overlap(book)
    corr.to_csv(OUT_CORR)
    beta.to_csv(OUT_BETA, index=False)
    crisis.to_csv(OUT_CRISIS, index=False)
    overlap.to_csv(OUT_OVERLAP, index=False)
    write_report(corr, beta, crisis, overlap)
    print(f"[step116] wrote {OUT_CORR.name}: {corr.shape[0]} x {corr.shape[1]}")
    print(f"[step116] wrote {OUT_BETA.name}: {len(beta)} rows")
    print(f"[step116] wrote {OUT_CRISIS.name}: {len(crisis)} rows")
    print(f"[step116] wrote {OUT_OVERLAP.name}: {len(overlap)} rows")
    print(f"[step116] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
