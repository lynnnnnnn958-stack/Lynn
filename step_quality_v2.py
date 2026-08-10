#!/usr/bin/env python3
"""
step_quality_v2.py — gross-profits-to-assets (Novy-Marx), quarterly, honest
===========================================================================
Upgrade of the annual quality test (which had only 15 data points, t=0.93).
Uses QUARTERLY fundamentals for ~4x more independent observations → far tighter
statistics. Signal = Novy-Marx gross-profits-to-assets (the single strongest
quality factor) on a trailing-twelve-month basis, plus TTM ROA.

Honesty controls: PIT filed_date (no look-ahead) · PIT S&P membership
(survivorship-controlled) · realistic costs · IS/OOS split · IC t-stat.

Input : quarterly_fundamentals.csv (step_edgar_quarterly_fundamentals.py)
Output: quality_v2_results.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TC_BPS = 10.0
OOS_CUTOFF = pd.Timestamp("2020-01-01")


def load_prices():
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


def prep(fund: pd.DataFrame) -> pd.DataFrame:
    """TTM gross profit + TTM net income per (ticker, quarter), carrying filed_date."""
    f = fund.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["filed_date"] = pd.to_datetime(f["filed_date"])
    f = f.sort_values(["ticker", "period_end"])
    out = []
    for tk, g in f.groupby("ticker"):
        g = g.drop_duplicates("period_end").sort_values("period_end").reset_index(drop=True)
        if len(g) < 5:
            continue
        g["gp_ttm"] = g["gross_profit"].rolling(4).sum()
        g["ni_ttm"] = g["net_income"].rolling(4).sum() if "net_income" in g else np.nan
        g["gpa"] = g["gp_ttm"] / g["assets"].replace(0, np.nan)      # Novy-Marx
        g["roa"] = g["ni_ttm"] / g["assets"].replace(0, np.nan)
        g = g.dropna(subset=["gpa"])
        out.append(g[["ticker", "period_end", "filed_date", "gpa", "roa"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def quality_asof(q: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    w = q[q["filed_date"] <= date]
    if w.empty:
        return pd.Series(dtype=float)
    w = w.sort_values("filed_date").drop_duplicates("ticker", keep="last").set_index("ticker")

    def z(s):
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd and sd > 0 else s * 0.0
    score = 0.7 * z(w["gpa"]) + 0.3 * z(w["roa"])
    return score.dropna()


def run():
    fp = ROOT / "quarterly_fundamentals.csv"
    if not fp.exists():
        return {"error": "quarterly_fundamentals.csv missing"}
    q = prep(pd.read_csv(fp))
    if q.empty:
        return {"error": "no quality computed"}
    prices = load_prices()
    pit = load_pit()
    daily = prices.pct_change(fill_method=None)

    # QUARTERLY rebalance (first trading day of Jan/Apr/Jul/Oct)
    idx = prices.index
    md = pd.DataFrame({"d": idx})
    md["q"] = md["d"].dt.to_period("Q")
    rebs = [g["d"].iloc[0] for _, g in md.groupby("q")]
    rebs = [r for r in rebs if r >= pd.Timestamp("2011-01-01")]

    ic_rows, ret_rows = [], []
    prev = set()
    for i in range(len(rebs) - 1):
        d0, d1 = rebs[i], rebs[i + 1]
        s = quality_asof(q, d0)
        mem = members_asof(pit, d0)
        if mem is not None:
            s = s[s.index.isin(mem)]
        s = s[s.index.isin(prices.columns)]
        if len(s) < 30:
            continue
        win = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        fwd = {t: (1 + win[t].fillna(0)).prod() - 1 for t in s.index if t in win.columns}
        merged = pd.DataFrame({"q": s, "fwd": pd.Series(fwd)}).dropna()
        if len(merged) >= 30:
            ic = merged["q"].corr(merged["fwd"], method="spearman")
            if not np.isnan(ic):
                ic_rows.append({"date": d0, "ic": ic})
        n_top = max(10, int(len(s) * 0.20))
        longs = s.sort_values(ascending=False).head(n_top).index.tolist()
        turn = len(set(longs) - prev) / max(len(longs), 1)
        tc = turn * 2 * TC_BPS / 10_000
        prev = set(longs)
        for j, (dt, row) in enumerate(win.iterrows()):
            r = np.nanmean([row.get(t, np.nan) for t in longs])
            r = 0.0 if np.isnan(r) else r
            ret_rows.append({"date": dt, "net": r - (tc if j == 0 else 0.0), "spy": row.get("SPY", np.nan)})

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
        "strategy": "Gross-profits-to-assets (Novy-Marx TTM) + ROA, QUARTERLY rebalance, long top quintile",
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
    print("Quality v2 — gross-profits-to-assets, QUARTERLY (more data points)")
    print("=" * 60)
    m = run()
    json.dump(m, open(ROOT / "quality_v2_results.json", "w"), indent=2, default=str)
    if "error" in m:
        print("  " + m["error"]); return
    print(f"  Quality IC: {m.get('quality_ic_mean')}  t={m.get('quality_ic_t')} (n={m.get('quality_ic_n')} quarters)")
    f = m["full_net"]
    print(f"  Long-only: Sharpe {f.get('sharpe')}  CAGR {f.get('cagr')}  MaxDD {f.get('max_dd')}")
    print(f"  Beta {m.get('beta_to_spy')}  Alpha/yr {m.get('alpha_annual_after_costs')}")
    print(f"  IS Sharpe {m['in_sample_pre2020'].get('sharpe') if m['in_sample_pre2020'] else 'n/a'} | "
          f"OOS Sharpe {m['out_of_sample_2020on'].get('sharpe') if m['out_of_sample_2020on'] else 'n/a'}")


if __name__ == "__main__":
    main()
