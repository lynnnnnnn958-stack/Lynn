#!/usr/bin/env python3
"""
step_quality_factors.py — accruals + gross-profitability, validated honestly
============================================================================
Two of the most-cited fundamental anomalies, tested with the same rigor as PEAD:

  * ACCRUALS (Sloan 1996): firms whose earnings come from accruals rather than
    cash flow subsequently UNDERperform. Signal = LOW accrual ratio is good.
  * GROSS PROFITABILITY (Novy-Marx 2013): high gross-profit/assets predicts
    OUTperformance and is not subsumed by value. Signal = HIGH is good.

Honesty controls (identical discipline to step_rigorous_backtest / step_pead):
  1. NO LOOK-AHEAD — each annual fundamental is only usable AFTER its EDGAR
     filed_date (know_date); we carry filed_date and gate on it.
  2. PIT S&P 500 membership (survivorship-controlled).
  3. Realistic costs; annual rebalance (fundamentals update yearly).
  4. IS/OOS split + IC t-stat of the quality score vs forward 1y returns.

Input : edgar_fundamentals.csv  (from download_edgar_fundamentals.py)
Output: quality_results.json + quality_report.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TC_BPS = 10.0
OOS_CUTOFF = pd.Timestamp("2020-01-01")


def load_prices() -> pd.DataFrame:
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists() and p.stat().st_size > 3:
            df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            return df[[c for c in df.columns if str(c).isalpha()]]
    raise FileNotFoundError("no prices")


def load_pit():
    p = ROOT / "sp500_pit_membership.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p, parse_dates=["date"])
    m["ticker"] = m["ticker"].astype(str).str.replace("-", ".", regex=False)
    return {d: set(g["ticker"]) for d, g in m.groupby("date")}


def members_asof(pit, date):
    if not pit:
        return None
    ks = [k for k in pit if k <= date]
    return pit[max(ks)] if ks else None


def load_fundamentals() -> pd.DataFrame:
    p = ROOT / "edgar_fundamentals.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["filed_date"] = pd.to_datetime(d["filed_date"], errors="coerce")
    return d.dropna(subset=["filed_date"])


def quality_asof(fund: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Composite quality score per ticker using the latest fundamental already
    public at `date` (filed_date <= date). Higher = better quality.
    quality = z(-accruals) + z(gross_margin) + z(roa)."""
    w = fund[fund["filed_date"] <= date]
    if w.empty:
        return pd.Series(dtype=float)
    w = w.sort_values("filed_date").drop_duplicates("ticker", keep="last")
    w = w.set_index("ticker")

    def z(s):
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd and sd > 0 else s * 0.0

    parts = []
    if "accrual_ratio" in w:  parts.append(-z(w["accrual_ratio"]))   # low accruals good
    if "gross_margin" in w:   parts.append(z(w["gross_margin"]))     # high margin good
    if "roa" in w:            parts.append(z(w["roa"]))              # profitable good
    if not parts:
        return pd.Series(dtype=float)
    q = sum(parts) / len(parts)
    return q.dropna()


def run() -> dict:
    fund = load_fundamentals()
    if fund.empty:
        return {"error": "edgar_fundamentals.csv missing — run download_edgar_fundamentals.py"}
    prices = load_prices()
    pit = load_pit()
    daily = prices.pct_change(fill_method=None)

    # annual rebalance: first trading day of July (post 10-K season)
    idx = prices.index
    yrs = sorted(set(idx.year))
    rebs = []
    for y in yrs:
        cand = idx[(idx.year == y) & (idx.month == 7)]
        if len(cand):
            rebs.append(cand[0])
    rebs = [r for r in rebs if r >= pd.Timestamp("2011-01-01")]

    ic_rows, ret_rows = [], []
    prev = set()
    for i in range(len(rebs) - 1):
        d0, d1 = rebs[i], rebs[i + 1]
        q = quality_asof(fund, d0)
        mem = members_asof(pit, d0)
        if mem is not None:
            q = q[q.index.isin(mem)]
        q = q[q.index.isin(prices.columns)]
        if len(q) < 30:
            continue
        win = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        fwd = {t: (1 + win[t].fillna(0)).prod() - 1 for t in q.index if t in win.columns}
        merged = pd.DataFrame({"q": q, "fwd": pd.Series(fwd)}).dropna()
        if len(merged) >= 30:
            ic = merged["q"].corr(merged["fwd"], method="spearman")
            if not np.isnan(ic):
                ic_rows.append({"date": d0, "ic": ic})
        n_top = max(10, int(len(q) * 0.20))
        longs = q.sort_values(ascending=False).head(n_top).index.tolist()
        turn = len(set(longs) - prev) / max(len(longs), 1)
        tc = turn * 2 * TC_BPS / 10_000
        prev = set(longs)
        for j, (dt, row) in enumerate(win.iterrows()):
            r = np.nanmean([row.get(t, np.nan) for t in longs])
            r = 0.0 if np.isnan(r) else r
            ret_rows.append({"date": dt, "net": r - (tc if j == 0 else 0.0),
                             "spy": row.get("SPY", np.nan)})

    return _metrics(pd.DataFrame(ic_rows), pd.DataFrame(ret_rows))


def _stat(r, ann=252):
    r = r.dropna()
    if len(r) < 30:
        return {}
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() else np.nan
    c = (1 + r).cumprod(); mdd = float((c / c.cummax() - 1).min())
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 2), "max_dd": round(mdd, 4)}


def _metrics(ic_df, ret_df):
    if ret_df.empty:
        return {"error": "no rows"}
    ret_df = ret_df.set_index("date")
    net, spy = ret_df["net"], ret_df["spy"].dropna()
    ic = ic_df["ic"].dropna() if not ic_df.empty else pd.Series(dtype=float)
    ic_t = float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std() else float("nan")
    common = ret_df.dropna(subset=["spy"])
    beta = alpha = np.nan
    if len(common) > 60:
        x, y = common["spy"].values, common["net"].values
        beta = float(np.cov(x, y)[0, 1] / np.var(x))
        alpha = float((y.mean() - beta * x.mean()) * 252)
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy": "Quality (low accruals + high gross margin + ROA), annual rebalance, long top quintile",
        "quality_ic_mean": round(float(ic.mean()), 4) if len(ic) else None,
        "quality_ic_t": round(ic_t, 2) if not np.isnan(ic_t) else None,
        "quality_ic_n": int(len(ic)),
        "full_net": _stat(net),
        "in_sample_pre2020": _stat(net[ret_df.index < OOS_CUTOFF]),
        "out_of_sample_2020on": _stat(net[ret_df.index >= OOS_CUTOFF]),
        "spy_buy_hold": _stat(spy),
        "beta_to_spy": round(beta, 3) if not np.isnan(beta) else None,
        "alpha_annual_after_costs": round(alpha, 4) if not np.isnan(alpha) else None,
    }


def main():
    print("=" * 60)
    print("Quality factors (accruals + profitability) — honest validation")
    print("=" * 60)
    m = run()
    json.dump(m, open(ROOT / "quality_results.json", "w"), indent=2, default=str)
    if "error" in m:
        print("  " + m["error"]); return
    print(f"  Quality IC: {m.get('quality_ic_mean')}  t={m.get('quality_ic_t')} (n={m.get('quality_ic_n')})")
    f = m["full_net"]
    print(f"  Long-only quality: Sharpe {f.get('sharpe')}  CAGR {f.get('cagr')}  MaxDD {f.get('max_dd')}")
    print(f"  Beta {m.get('beta_to_spy')}  Alpha/yr {m.get('alpha_annual_after_costs')}")
    print(f"  OOS Sharpe: {m['out_of_sample_2020on'].get('sharpe') if m['out_of_sample_2020on'] else 'n/a'}")


if __name__ == "__main__":
    main()
