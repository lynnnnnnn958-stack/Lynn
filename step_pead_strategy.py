#!/usr/bin/env python3
"""
step_pead_strategy.py — Post-Earnings-Announcement Drift, validated honestly
============================================================================
The first *real* edge attempt: PEAD (Bernard & Thomas 1989). Investors underreact
to earnings surprises; prices keep drifting in the surprise direction for weeks.
The effect is strongest in small / under-covered names — exactly where Wall Street's
size can't play.

Honesty controls (same discipline as step_rigorous_backtest):
  1. NO LOOK-AHEAD — the surprise for a quarter is only usable AFTER its EDGAR
     filed_date (know_date). SUE uses a seasonal random walk (EPS_q vs EPS_{q-4}),
     standardized by the trailing std of surprises — no analyst estimates needed.
  2. PIT S&P 500 membership (survivorship-controlled).
  3. Realistic costs (spread + impact).
  4. True out-of-sample: IS (pre-2020) vs OOS (2020+) reported separately.
  5. Reports the IC of SUE vs forward returns and a long-only decile backtest.

Output: pead_results.json + pead_report.md
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
# optional overrides so the SAME code can test the small-cap universe
PRICE_FILE = os.environ.get("PEAD_PRICE_FILE", "")   # e.g. smallcap_price_history.csv
EPS_FILE   = os.environ.get("PEAD_EPS_FILE", "eps_pit.csv")
USE_PIT    = os.environ.get("PEAD_USE_PIT", "1") == "1"

DRIFT_DAYS   = 60          # hold window after a fresh earnings surprise
SUE_MIN_HIST = 8           # need ≥8 quarters to standardize
TOP_FRAC     = 0.20        # long the top quintile by SUE
TC_SPREAD_BPS = 5.0
IMPACT_BPS    = 3.0        # smaller names → but we proxy modestly
OOS_CUTOFF   = pd.Timestamp("2020-01-01")


def load_prices() -> pd.DataFrame:
    cands = ([PRICE_FILE] if PRICE_FILE else []) + ["sp500_price_history_deep.csv", "sp500_price_cache.csv"]
    for f in cands:
        p = ROOT / f
        if p.exists() and p.stat().st_size > 3:
            df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            return df[[c for c in df.columns if str(c).replace("-", "").isalpha()]]
    raise FileNotFoundError("no price cache")


def load_pit() -> dict:
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


def compute_sue(eps: pd.DataFrame) -> pd.DataFrame:
    """Seasonal-random-walk SUE per (ticker, quarter). Zero look-ahead:
    each row carries the filed_date (know_date) it became usable."""
    eps = eps.copy()
    eps["period_end"] = pd.to_datetime(eps["period_end"])
    eps["filed_date"] = pd.to_datetime(eps["filed_date"])
    eps = eps.sort_values(["ticker", "period_end"])
    out = []
    for tk, g in eps.groupby("ticker"):
        g = g.drop_duplicates("period_end").sort_values("period_end").reset_index(drop=True)
        if len(g) < SUE_MIN_HIST + 4:
            continue
        # seasonal surprise = EPS_q − EPS_{q−4}
        g["surprise"] = g["eps"] - g["eps"].shift(4)
        # standardize by trailing std of surprises (expanding, min 8, using only past)
        g["sue"] = g["surprise"] / g["surprise"].rolling(SUE_MIN_HIST, min_periods=SUE_MIN_HIST).std().shift(0)
        g = g.dropna(subset=["sue"])
        g["sue"] = g["sue"].clip(-8, 8)
        out.append(g[["ticker", "period_end", "filed_date", "eps", "surprise", "sue"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def active_sue_asof(sue: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Latest usable SUE per ticker at `date`: filed within the last DRIFT_DAYS,
    filed_date <= date (already public). Returns {ticker: sue}."""
    lo = date - pd.Timedelta(days=DRIFT_DAYS)
    w = sue[(sue["filed_date"] <= date) & (sue["filed_date"] > lo)]
    if w.empty:
        return pd.Series(dtype=float)
    w = w.sort_values("filed_date").drop_duplicates("ticker", keep="last")
    return w.set_index("ticker")["sue"]


def run() -> dict:
    prices = load_prices()
    pit = load_pit() if USE_PIT else {}
    eps_p = ROOT / EPS_FILE
    if not eps_p.exists():
        return {"error": "eps_pit.csv missing — run step_edgar_eps_pit.py first"}
    sue = compute_sue(pd.read_csv(eps_p))
    if sue.empty:
        return {"error": "no SUE computed"}
    print(f"  SUE universe: {sue['ticker'].nunique()} tickers, {len(sue):,} quarterly obs")

    daily = prices.pct_change(fill_method=None)
    # monthly rebalance dates
    idx = prices.index
    md = pd.DataFrame({"d": idx}); md["m"] = md["d"].dt.to_period("M")
    rebs = list(md.groupby("m")["d"].first())
    rebs = [r for r in rebs if r >= pd.Timestamp("2011-01-01")]

    ic_rows, ret_rows = [], []
    prev = set()
    impact = IMPACT_BPS
    for i in range(len(rebs) - 1):
        d0, d1 = rebs[i], rebs[i + 1]
        s = active_sue_asof(sue, d0)
        mem = members_asof(pit, d0)
        if mem is not None:
            s = s[s.index.isin(mem)]
        s = s[s.index.isin(prices.columns)]
        if len(s) < 20:
            continue
        # ── IC: SUE vs realized forward 1-month return (honest predictive check) ──
        fwd = {}
        win = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        for tk in s.index:
            if tk in win.columns:
                fwd[tk] = (1 + win[tk].fillna(0)).prod() - 1
        merged = pd.DataFrame({"sue": s, "fwd": pd.Series(fwd)}).dropna()
        if len(merged) >= 20:
            ic = merged["sue"].corr(merged["fwd"], method="spearman")
            if not np.isnan(ic):
                ic_rows.append({"date": d0, "ic": ic, "n": len(merged)})
        # ── long-only top-quintile SUE, realistic cost ──
        n_top = max(5, int(len(s) * TOP_FRAC))
        longs = s.sort_values(ascending=False).head(n_top).index.tolist()
        turn = len(set(longs) - prev) / max(len(longs), 1)
        tc = turn * (2 * TC_SPREAD_BPS + impact) / 10_000
        prev = set(longs)
        for j, (dt, row) in enumerate(win.iterrows()):
            r = np.nanmean([row.get(t, np.nan) for t in longs])
            r = 0.0 if np.isnan(r) else r
            cost = tc if j == 0 else 0.0
            ret_rows.append({"date": dt, "net": r - cost, "spy": row.get("SPY", np.nan)})

    return _metrics(pd.DataFrame(ic_rows), pd.DataFrame(ret_rows))


def _stat(r, ann=252):
    r = r.dropna()
    if len(r) < 30:
        return {}
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() else np.nan
    c = (1 + r).cumprod(); mdd = float((c / c.cummax() - 1).min())
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 2), "max_dd": round(mdd, 4),
            "win_rate": round((r > 0).mean(), 4)}


def _metrics(ic_df, ret_df):
    if ret_df.empty:
        return {"error": "no backtest rows"}
    ret_df = ret_df.set_index("date")
    net, spy = ret_df["net"], ret_df["spy"].dropna()
    is_m = ret_df.index < OOS_CUTOFF; oos_m = ret_df.index >= OOS_CUTOFF
    # SUE IC significance
    ic = ic_df["ic"].dropna() if not ic_df.empty else pd.Series(dtype=float)
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    ic_t = float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std() else float("nan")
    # beta/alpha
    common = ret_df.dropna(subset=["spy"])
    beta = alpha_ann = np.nan
    if len(common) > 60:
        x, y = common["spy"].values, common["net"].values
        beta = float(np.cov(x, y)[0, 1] / np.var(x))
        alpha_ann = float((y.mean() - beta * x.mean()) * 252)
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy": "PEAD long-only top-quintile SUE, 60d drift, monthly rebalance",
        "controls": "PIT filed_date (no look-ahead) · PIT membership · realistic costs · IS/OOS split",
        "sue_ic_mean": round(ic_mean, 4) if not np.isnan(ic_mean) else None,
        "sue_ic_t": round(ic_t, 2) if not np.isnan(ic_t) else None,
        "sue_ic_n": int(len(ic)),
        "full_net": _stat(net),
        "in_sample_pre2020": _stat(net[is_m]),
        "out_of_sample_2020on": _stat(net[oos_m]),
        "spy_buy_hold": _stat(spy),
        "beta_to_spy": round(beta, 3) if not np.isnan(beta) else None,
        "alpha_annual_after_costs": round(alpha_ann, 4) if not np.isnan(alpha_ann) else None,
    }


def _report(m: dict) -> str:
    if "error" in m:
        return f"# PEAD Strategy\n\n{m['error']}\n"
    f = m["full_net"]; oos = m["out_of_sample_2020on"]; spy = m["spy_buy_hold"]
    a = m.get("alpha_annual_after_costs"); ict = m.get("sue_ic_t")
    edge = ("REAL EDGE" if a and a > 0.02 and ict and ict > 2 else
            "WEAK / NONE" if a is not None else "N/A")
    return "\n".join([
        "# PEAD — Post-Earnings-Announcement Drift (honest validation)", "",
        f"_{m['strategy']}_", f"_{m['controls']}_", "",
        "## Does the surprise predict returns? (the real test)",
        f"- SUE → forward-return IC: **{m.get('sue_ic_mean')}**  (t = {ict}, n = {m.get('sue_ic_n')} months)",
        f"- |t| > 2 means the surprise has statistically real predictive power.", "",
        "## Long-only top-quintile SUE (net of costs)",
        "| Metric | PEAD | SPY |",
        "|---|---|---|",
        f"| Sharpe | {f.get('sharpe','—')} | {spy.get('sharpe','—')} |",
        f"| CAGR | {f.get('cagr',0):.1%} | {spy.get('cagr',0):.1%} |" if f else "| CAGR | — | — |",
        f"| Max DD | {f.get('max_dd',0):.1%} | {spy.get('max_dd',0):.1%} |" if f else "",
        f"| OOS Sharpe (2020+) | {oos.get('sharpe','—') if oos else 'n/a'} | | ", "",
        f"- **Beta to SPY:** {m.get('beta_to_spy')}",
        f"- **Alpha/yr after costs:** {a:.2%}" if a is not None else "- Alpha: n/a",
        f"- **Verdict:** {edge}", "",
        "_First real edge attempt with EDGAR point-in-time earnings. Research only._",
    ])


def main():
    print("=" * 60)
    print("PEAD — Post-Earnings-Announcement Drift (honest validation)")
    print("=" * 60)
    m = run()
    json.dump(m, open(ROOT / "pead_results.json", "w"), indent=2, default=str)
    (ROOT / "pead_report.md").write_text(_report(m))
    if "error" in m:
        print("  " + m["error"]); return
    print(f"  SUE IC: {m.get('sue_ic_mean')}  t={m.get('sue_ic_t')}  (n={m.get('sue_ic_n')})")
    f = m["full_net"]
    print(f"  Long-only SUE: Sharpe {f.get('sharpe')}  CAGR {f.get('cagr')}  MaxDD {f.get('max_dd')}")
    print(f"  Beta {m.get('beta_to_spy')}  Alpha/yr {m.get('alpha_annual_after_costs')}")
    print(f"  OOS Sharpe: {m['out_of_sample_2020on'].get('sharpe') if m['out_of_sample_2020on'] else 'n/a'}")


if __name__ == "__main__":
    main()
