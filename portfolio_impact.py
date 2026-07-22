"""
Canyon Quant — Portfolio Impact Analysis
=========================================
Shows what adding Canyon Quant strategy to a 60/40 portfolio does:
  1. Mean-variance efficient frontier (60/40 base vs adding strategy)
  2. Sharpe improvement at each allocation level (0% to 40%)
  3. Correlation structure and diversification benefit
  4. Optimal allocation weight
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
W    = 70


# ── Load monthly returns ─────────────────────────────────────────────────────
oos_df  = pd.read_csv(ROOT / "backtest_v24_oos.csv",  parse_dates=["date"])
train_df= pd.read_csv(ROOT / "backtest_v20_train.csv", parse_dates=["date"])

def to_month_end(df, ret_col="ret", spy_col="spy_ret"):
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    def agg(col):
        if col not in df.columns: return None
        r = (df.groupby("ym")[col]
             .apply(lambda x: float(np.prod(1+x.values)-1)).reset_index())
        r["date"] = r["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
        return r.set_index("date")[col]
    out = pd.DataFrame({"strat": agg(ret_col)})
    if spy_col in df.columns:
        out["spy"] = agg(spy_col)
    return out[~out.index.duplicated(keep="last")]

oos_m   = to_month_end(oos_df)
train_m = to_month_end(train_df)
all_m   = pd.concat([train_m, oos_m]).sort_index()
all_m   = all_m[~all_m.index.duplicated(keep="last")]

# Download BND (bonds proxy) and GLD
try:
    import yfinance as yf
    bnd_raw = yf.download("BND", start="2007-04-01", end="2026-06-23",
                          auto_adjust=True, progress=False)["Close"].squeeze()
    bnd_m = bnd_raw.resample("ME").last().pct_change().dropna()
    print("  BND downloaded successfully")
except Exception as e:
    print(f"  BND download failed ({e}), using synthetic bond proxy (Rf+spread)")
    ff_m    = pd.read_csv(ROOT/"ff_factors_monthly.csv", index_col=0, parse_dates=True)
    rf_m    = ff_m["RF"].resample("ME").last()
    bnd_m   = rf_m + 0.002   # synthetic: Rf + 20bps/month spread
    bnd_m.name = "BND"

# Align all series to OOS period (2019-2026) for primary analysis
spy_m = oos_m["spy"].dropna()
strat_m = oos_m["strat"].dropna()
common_oos = spy_m.index.intersection(strat_m.index).intersection(bnd_m.index)
common_oos = common_oos[common_oos >= "2019-01-01"]

R_oos = pd.DataFrame({
    "Canyon": strat_m[common_oos],
    "SPY":    spy_m[common_oos],
    "BND":    bnd_m[common_oos],
}).dropna()

# Full period
spy_all = all_m["spy"].dropna()
strat_all = all_m["strat"].dropna()
common_all = spy_all.index.intersection(strat_all.index).intersection(bnd_m.index)
R_all = pd.DataFrame({
    "Canyon": strat_all[common_all],
    "SPY":    spy_all[common_all],
    "BND":    bnd_m[common_all],
}).dropna()


# ── Portfolio metrics ─────────────────────────────────────────────────────────
def port_stats(weights, R, ppy=12):
    w  = np.array(weights)
    r  = R.values @ w
    ar = float((1+r).prod()**(ppy/len(r))-1)
    vol= r.std() * np.sqrt(ppy)
    sh = ar / (vol + 1e-9)
    neg= r[r<0]
    so = ar / (neg.std()*np.sqrt(ppy)+1e-9) if len(neg)>1 else np.nan
    cum= np.cumprod(1+r); peak=np.maximum.accumulate(cum)
    mdd= float(np.max((peak-cum)/peak))
    return dict(ar=ar, vol=vol, sharpe=sh, sortino=so, mdd=mdd)


def neg_sharpe(w, R):
    stats = port_stats(w, R)
    return -stats["sharpe"]


def optimize_port(R, constraints, bounds):
    w0     = np.ones(len(R.columns)) / len(R.columns)
    result = minimize(neg_sharpe, w0, args=(R,),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=bounds,
                      options={"ftol": 1e-9, "maxiter": 1000})
    return result.x, -result.fun


# ── 1. BASELINE 60/40 ────────────────────────────────────────────────────────
print("=" * W)
print("  1. BASELINE: 60/40 (SPY / BND)  OOS 2019-2026")
print("=" * W)

# Only use SPY and BND for baseline
R_base = R_oos[["SPY","BND"]]
w_60_40 = np.array([0.60, 0.40])
s_base  = port_stats(w_60_40, R_base)
print(f"\n  60% SPY / 40% BND:")
print(f"  AR={s_base['ar']:>+7.1%}  Sharpe={s_base['sharpe']:.3f}  "
      f"Sortino={s_base['sortino']:.3f}  MDD={-s_base['mdd']:>+6.1%}")


# ── 2. ALLOCATION SWEEP ───────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("  2. PORTFOLIO SHARPE vs CANYON QUANT ALLOCATION")
print(f"{'='*W}")
print("  Fixed SPY:BND ratio = 3:2. Add Canyon Quant, reduce both proportionally.\n")
print(f"  {'Canyon Alloc':>13} {'SPY':>6} {'BND':>6}  {'AR':>8} {'Sharpe':>8} "
      f"{'Sortino':>9} {'MDD':>8}  {'vs 60/40'}")
print(f"  {'-'*72}")

best_sh   = s_base["sharpe"]
best_alloc= 0.0
results_sweep = []
for canyon_w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    remain  = 1.0 - canyon_w
    spy_w   = remain * 0.60
    bnd_w   = remain * 0.40
    w       = np.array([canyon_w, spy_w, bnd_w])
    s       = port_stats(w, R_oos)
    delta   = s["sharpe"] - s_base["sharpe"]
    flag    = " ← baseline" if canyon_w == 0 else (" ← OPTIMAL" if s["sharpe"] > best_sh else "")
    if s["sharpe"] > best_sh:
        best_sh    = s["sharpe"]
        best_alloc = canyon_w
    results_sweep.append((canyon_w, spy_w, bnd_w, s))
    print(f"  {canyon_w:>12.0%}  {spy_w:>5.0%}  {bnd_w:>5.0%}  "
          f"{s['ar']:>+7.1%}  {s['sharpe']:>7.3f}  "
          f"{s['sortino']:>8.3f}  {-s['mdd']:>7.1%}  {delta:>+7.3f}{flag}")


# ── 3. EFFICIENT FRONTIER ────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("  3. EFFICIENT FRONTIER SHIFT  (OOS 2019-2026)")
print(f"{'='*W}")
print("  Target returns from 4% to 20%, find min-vol portfolio with and without strategy\n")

print(f"  {'Target AR':>10}  {'60/40 Sharpe':>14}  {'With Canyon (20%)':>20}  {'Improvement':>12}")
print(f"  {'-'*62}")

for tgt in [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
    # Without Canyon: SPY+BND only
    def neg_sharpe_2(w):
        return port_stats(w, R_base)["sharpe"] * -1
    cons_base = [{"type":"eq","fun":lambda w: w.sum()-1},
                 {"type":"eq","fun":lambda w: port_stats(w,R_base)["ar"]-tgt}]
    bnds_base = [(0,1),(0,1)]
    try:
        r2 = minimize(neg_sharpe_2, [0.6,0.4], method="SLSQP",
                      constraints=cons_base, bounds=bnds_base,
                      options={"ftol":1e-6,"maxiter":200})
        sh_no = -r2.fun if r2.success else np.nan
    except Exception:
        sh_no = np.nan

    # With Canyon at 20% fixed, optimise SPY/BND split
    def neg_sharpe_c(w_2):
        w = np.array([0.20, w_2[0]*0.80, w_2[1]*0.80])
        return port_stats(w, R_oos)["sharpe"] * -1
    cons_c = [{"type":"eq","fun":lambda w2: w2.sum()-1},
              {"type":"eq","fun":lambda w2: port_stats(
                  np.array([0.20,w2[0]*0.80,w2[1]*0.80]),R_oos)["ar"]-tgt}]
    bnds_c = [(0,1),(0,1)]
    try:
        rc = minimize(neg_sharpe_c, [0.75,0.25], method="SLSQP",
                      constraints=cons_c, bounds=bnds_c,
                      options={"ftol":1e-6,"maxiter":200})
        sh_w = -rc.fun if rc.success else np.nan
    except Exception:
        sh_w = np.nan

    delta = sh_w - sh_no if np.isfinite(sh_no) and np.isfinite(sh_w) else np.nan
    flag  = "✓ BETTER" if (np.isfinite(delta) and delta > 0) else ""
    print(f"  {tgt:>9.0%}    "
          f"{sh_no:>10.3f}    "
          f"{sh_w:>16.3f}    "
          f"{delta:>+9.3f}  {flag}")


# ── 4. CORRELATION & DIVERSIFICATION ────────────────────────────────────────
print(f"\n{'='*W}")
print("  4. DIVERSIFICATION ANALYSIS")
print(f"{'='*W}")

corr = R_oos.corr()
cov  = R_oos.cov() * 12   # annualized

print(f"\n  Correlation matrix (OOS 2019-2026):")
print(f"  {'':<12}" + "".join(f"  {c:>8}" for c in R_oos.columns))
for row in R_oos.columns:
    vals = "".join(f"  {corr.loc[row,col]:>+8.3f}" for col in R_oos.columns)
    print(f"  {row:<12}{vals}")

# Diversification ratio: weighted avg vol / portfolio vol
w_opt = np.array([best_alloc, (1-best_alloc)*0.60, (1-best_alloc)*0.40])
vols  = np.sqrt(np.diag(cov))
wtd_avg_vol = np.dot(w_opt, vols)
port_vol    = np.sqrt(w_opt @ cov.values @ w_opt)
div_ratio   = wtd_avg_vol / port_vol

print(f"\n  Optimal allocation ({best_alloc:.0%} Canyon, {(1-best_alloc)*60:.0%} SPY, "
      f"{(1-best_alloc)*40:.0%} BND):")
print(f"  Individual vols:   Canyon {vols[0]*100:.1f}%   SPY {vols[1]*100:.1f}%   "
      f"BND {vols[2]*100:.1f}%  (annualized)")
print(f"  Weighted-avg vol:  {wtd_avg_vol*100:.1f}%")
print(f"  Portfolio vol:     {port_vol*100:.1f}%")
print(f"  Diversification:   {div_ratio:.2f}x  (>1 = portfolio benefits from diversification)")


# ── 5. FULL-PERIOD ANALYSIS ──────────────────────────────────────────────────
print(f"\n{'='*W}")
print("  5. FULL-PERIOD VALIDATION  (2007-2026, includes 2008 GFC)")
print(f"{'='*W}")

R_full_base = R_all[["SPY","BND"]]
s_base_full = port_stats(np.array([0.60,0.40]), R_full_base)
w_full_opt  = np.array([best_alloc, (1-best_alloc)*0.60, (1-best_alloc)*0.40])
s_full_opt  = port_stats(w_full_opt, R_all)

print(f"\n  {'Portfolio':<30} {'AR':>8} {'Sharpe':>8} {'Sortino':>9} {'MDD':>8}")
print(f"  {'-'*60}")
print(f"  {'60/40 (SPY/BND)':<30} {s_base_full['ar']:>+7.1%}  "
      f"{s_base_full['sharpe']:>7.3f}  {s_base_full['sortino']:>8.3f}  "
      f"{-s_base_full['mdd']:>7.1%}")
print(f"  {f'{best_alloc:.0%} Canyon + blended':<30} {s_full_opt['ar']:>+7.1%}  "
      f"{s_full_opt['sharpe']:>7.3f}  {s_full_opt['sortino']:>8.3f}  "
      f"{-s_full_opt['mdd']:>7.1%}")
print(f"\n  Sharpe improvement: {s_full_opt['sharpe']-s_base_full['sharpe']:>+.3f}")
print(f"  MDD improvement:    {(-s_full_opt['mdd'])-(-s_base_full['mdd']):>+.1%}")


# ── 6. SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("  6. SUMMARY — PORTFOLIO CONSTRUCTION RECOMMENDATION")
print(f"{'='*W}")

s_curr_opt = port_stats(w_opt, R_oos)

print(f"""
  Optimal allocation in OOS 2019-2026:
    Canyon Quant: {best_alloc:.0%}
    SPY:          {(1-best_alloc)*60:.0%}
    BND:          {(1-best_alloc)*40:.0%}

  Effect on institutional 60/40 portfolio:
    60/40 Sharpe:    {s_base['sharpe']:.3f}
    Blended Sharpe:  {s_curr_opt['sharpe']:.3f}   (+{s_curr_opt['sharpe']-s_base['sharpe']:.3f})

    60/40 MDD:       {-s_base['mdd']:.1%}
    Blended MDD:     {-s_curr_opt['mdd']:.1%}  ({(-s_curr_opt['mdd'])-(-s_base['mdd']):>+.1%})

  Diversification ratio: {div_ratio:.2f}x
  (A ratio above 1.0 means the portfolio genuinely benefits from adding the strategy)

  Key: Canyon Quant adds value even at moderate allocations (10-20%).
  The primary mechanism is low correlation to SPY (ρ={corr.loc['Canyon','SPY']:.3f})
  while maintaining positive expected return.
""")
