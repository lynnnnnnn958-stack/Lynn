"""
Improvement 1 — Parameter Sensitivity Scan
===========================================
Tests whether v11's performance is at a stable plateau
(real signal) or a sharp peak (overfitted).

Grid: top_n × momentum_lookback × etf_max × ma_period
81 combinations, each run over all rebalancing periods.
"""
import warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
HOLD_M = 21
WARMUP = 252
TCOST  = 5

# ── Load once ──────────────────────────────────────────────────────────
print("Loading data...")
prices  = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
macro   = pd.read_csv(ROOT/"macro_regime.csv", index_col=0, parse_dates=True).sort_index()

sector_path = ROOT/"sp500_sectors.csv"
sector_map  = pd.read_csv(sector_path, index_col=0).squeeze().to_dict() if sector_path.exists() else {}

common  = prices.index.intersection(letf_df.index).intersection(macro.index)
prices  = prices.reindex(common)
letf_df = letf_df.reindex(common).ffill().bfill()
spy     = spy.reindex(common).ffill()
macro   = macro.reindex(common).ffill()
tqqq    = letf_df["TQQQ"].values
tqqq_r  = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq>0, tqqq, np.nan)))])

price_arr = prices.values.astype(float)
spy_arr   = spy.values.astype(float)
col_idx   = {c: i for i, c in enumerate(prices.columns)}

rebal_all = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))
print(f"  {len(rebal_all)} rebalancing periods  {len(prices.columns)} stocks")

# ── V11 base params (center point) ─────────────────────────────────────
V11 = dict(top_n=25, mom_lk=12, etf_max=0.60, ma_p=200,
           vol_win=20, vol_target=0.18,
           stock_bull=0.40, stock_bear=0.20,
           vix_thresh=30.0, hyg_win=20)

# ── Fast stock score (vectorized per period) ────────────────────────────
def fast_stock_score(t, top_n, mom_lk):
    """Returns (selected_tickers, weights_dict)"""
    if t < WARMUP: return None, None

    p    = price_arr[:t+1]
    now  = p[-1]
    p1m  = p[-22]
    p3m  = p[-64]  if t >= 63  else p[0]
    p13m = p[-(mom_lk*21+21)] if t >= mom_lk*21+20 else p[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        mom1  = np.where(p13m>0, now/p13m-1, np.nan).clip(-1, 1)
        mom3  = np.where(p3m>0,  now/p3m-1,  np.nan).clip(-1, 1)
        mom1m = np.where(p1m>0,  now/p1m-1,  np.nan).clip(-0.5, 0.5)

    vol20 = np.nanstd(np.diff(np.log(np.where(p[-22:]>0, p[-22:], np.nan)), axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    def rk(arr):
        valid = np.isfinite(arr)
        result = np.full_like(arr, np.nan)
        result[valid] = (arr[valid].argsort().argsort() + 1) / valid.sum() * 100
        return result

    score = (rk(mom1)*0.35 + rk(mom3)*0.20 + rk(invvol)*0.25 +
             rk(mom1m)*0.10 + rk(above)*0.10)

    # quality filter
    delta = np.diff(np.log(np.where(p[-16:]>0, p[-16:], np.nan)), axis=0)
    gains = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss  = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14 = 100 - 100/(1 + gains/(loss+1e-9))
    dist  = np.where(sma200>0, now/sma200-1, 0)
    vol75 = np.nanpercentile(vol20, 75)
    quality = (rsi14 < 75) & (dist < 0.5) & (vol20 < vol75)

    valid_cols = np.where(quality & np.isfinite(score))[0]
    candidate_idx = valid_cols[np.argsort(score[valid_cols])[::-1][:top_n*2]]
    if len(candidate_idx) < 10:
        candidate_idx = np.where(np.isfinite(score))[0]
        candidate_idx = candidate_idx[np.argsort(score[candidate_idx])[::-1][:top_n]]

    # sector constraint
    selected_idx, seen_sectors = [], {}
    tickers = list(prices.columns)
    for ci in candidate_idx:
        if len(selected_idx) >= top_n: break
        sec = sector_map.get(tickers[ci], "Unknown")
        seen_sectors[sec] = seen_sectors.get(sec, 0) + 1
        if seen_sectors[sec] <= max(1, int(top_n * 0.35)):
            selected_idx.append(ci)
    if len(selected_idx) < 10:
        selected_idx = candidate_idx[:top_n].tolist()

    # inv-vol weighting
    sel_vol = vol20[selected_idx].clip(min=0.001)
    raw_w   = 1.0 / sel_vol
    raw_w   = raw_w / raw_w.sum()
    raw_w   = np.clip(raw_w, 0, 0.15)
    raw_w   = raw_w / raw_w.sum()
    weights = {tickers[i]: float(w) for i, w in zip(selected_idx, raw_w)}
    return [tickers[i] for i in selected_idx], weights

def stock_ret(t, sel, wts, sw):
    if not sel or sw == 0: return 0.0
    t1 = min(t+HOLD_M, len(prices)-1)
    r  = 0.0
    for tk, wt in wts.items():
        ci = col_idx.get(tk)
        if ci is None: continue
        p0, p1 = price_arr[t, ci], price_arr[t1, ci]
        if p0 > 0 and np.isfinite(p1):
            r += wt * (p1/p0 - 1)
    return r * sw

def etf_ret(t, etf_max, ma_p, vol_win=20, vol_target=0.18):
    above_ma = t < ma_p or float(spy_arr[t]) > float(np.mean(spy_arr[t-ma_p+1:t+1]))
    if not above_ma: return 0.0, 0.0
    w = tqqq_r[max(0,t-vol_win):t]
    w = w[np.isfinite(w)]
    vol = float(np.std(w)*np.sqrt(252)) if len(w) >= 5 else 0.40
    ew  = float(np.clip(vol_target/(vol+0.001), 0, etf_max))
    # TQQQ 50MA gate
    if not (t < 50 or float(tqqq[t]) > float(np.mean(tqqq[t-50:t]))):
        ew *= 0.5
    t1   = min(t+HOLD_M, len(tqqq)-1)
    ep   = float(tqqq[t1]); e0 = float(tqqq[t])
    er   = (ep/e0 - 1) if e0 > 0 and np.isfinite(ep) else 0.0
    return ew, er

def run_params(top_n, mom_lk, etf_max, ma_p):
    rets, spy_rets = [], []
    for t in rebal_all:
        above = t < ma_p or float(spy_arr[t]) > float(np.mean(spy_arr[t-ma_p+1:t+1]))
        sw = V11['stock_bull'] if above else V11['stock_bear']
        sel, wts = fast_stock_score(t, top_n, mom_lk)
        sr  = stock_ret(t, sel, wts, sw)
        ew, er = etf_ret(t, etf_max, ma_p)
        ret = sr + ew * er - TCOST/10_000
        t1  = min(t+HOLD_M, len(spy_arr)-1)
        spy_r = float(spy_arr[t1])/float(spy_arr[t]) - 1
        rets.append(ret); spy_rets.append(spy_r)
    r  = np.array(rets)
    n  = len(r); ppy = 252/HOLD_M
    tot = float(np.prod(1+r))
    ar  = float(tot**(ppy/n) - 1)
    av  = float(r.std()*np.sqrt(ppy))
    sr  = ar / (av + 1e-9)
    cum = np.cumprod(1+r)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.max((peak-cum)/peak))
    return dict(ar=ar, sr=sr, mdd=mdd)

# ── Center point (v11) ──────────────────────────────────────────────────
print("\nRunning v11 center point...")
center = run_params(25, 12, 0.60, 200)
print(f"  v11 baseline: AR={center['ar']:+.1%}  Sharpe={center['sr']:.2f}  MDD={-center['mdd']:.1%}")

# ── Grid search ──────────────────────────────────────────────────────────
grid = {
    'top_n':    [15, 20, 25, 30, 35],
    'mom_lk':   [9,  11, 12, 13, 15],
    'etf_max':  [0.40, 0.50, 0.60, 0.70],
    'ma_p':     [150, 180, 200, 220, 250],
}

print(f"\nRunning {len(grid['top_n'])*len(grid['mom_lk'])*len(grid['etf_max'])*len(grid['ma_p'])} param combinations...")
t0 = time.time()
results = []
n_done  = 0
total   = len(grid['top_n'])*len(grid['mom_lk'])*len(grid['etf_max'])*len(grid['ma_p'])

for tn, ml, em, mp in product(grid['top_n'], grid['mom_lk'], grid['etf_max'], grid['ma_p']):
    r = run_params(tn, ml, em, mp)
    results.append(dict(top_n=tn, mom_lk=ml, etf_max=em, ma_p=mp, **r))
    n_done += 1
    if n_done % 50 == 0:
        print(f"  {n_done}/{total}  elapsed={time.time()-t0:.0f}s")

df_grid = pd.DataFrame(results)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s")

# ── Analysis ──────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  PARAMETER SENSITIVITY ANALYSIS")
print("═"*70)

print(f"\n  Center (v11): AR={center['ar']:+.1%}  Sharpe={center['sr']:.2f}")
print(f"  Grid range:   AR [{df_grid['ar'].min():+.1%}, {df_grid['ar'].max():+.1%}]"
      f"  Sharpe [{df_grid['sr'].min():.2f}, {df_grid['sr'].max():.2f}]")
print(f"  Sharpe std across all 500 combos: {df_grid['sr'].std():.3f}")

center_sharpe = center['sr']
within_5pct = (df_grid['sr'] >= center_sharpe * 0.95).mean()
print(f"  % combos within 5% of center Sharpe: {within_5pct:.0%}")
plateau_verdict = "PLATEAU (stable signal)" if within_5pct > 0.50 else "SHARP PEAK (overfitted)"
print(f"  Verdict: {plateau_verdict}")

# Sensitivity by each parameter (fix others at v11, vary one)
print(f"\n  Single-parameter sensitivity (others fixed at v11 values):")
print(f"  {'Param':<12} {'Value':<8} {'Sharpe':>8}  {'AR':>8}  {'MDD':>8}")
print("  " + "─"*46)

for param, vals, center_val in [
    ('top_n',   [15, 20, 25, 30, 35], 25),
    ('mom_lk',  [9, 11, 12, 13, 15],  12),
    ('etf_max', [0.40, 0.50, 0.60, 0.70], 0.60),
    ('ma_p',    [150, 180, 200, 220, 250], 200),
]:
    for v in vals:
        mask = True
        for p2, cv in [('top_n',25),('mom_lk',12),('etf_max',0.60),('ma_p',200)]:
            if p2 != param:
                mask = mask & (df_grid[p2] == cv)
        row = df_grid[mask & (df_grid[param] == v)]
        if len(row):
            r = row.iloc[0]
            marker = " ◀ v11" if v == center_val else ""
            print(f"  {param:<12} {str(v):<8} {r['sr']:>8.3f}  {r['ar']:>+7.1%}  {-r['mdd']:>+7.1%}{marker}")
    print()

# Top 5 and bottom 5
print(f"\n  Top 5 parameter combos by Sharpe:")
print(f"  {'top_n':>6} {'mom_lk':>7} {'etf_max':>8} {'ma_p':>6}  {'Sharpe':>8}  {'AR':>8}")
for _, row in df_grid.nlargest(5, 'sr').iterrows():
    print(f"  {int(row.top_n):>6} {int(row.mom_lk):>7} {row.etf_max:>8.0%} {int(row.ma_p):>6}  "
          f"{row.sr:>8.3f}  {row.ar:>+7.1%}")

print(f"\n  Worst 5 parameter combos by Sharpe:")
for _, row in df_grid.nsmallest(5, 'sr').iterrows():
    print(f"  {int(row.top_n):>6} {int(row.mom_lk):>7} {row.etf_max:>8.0%} {int(row.ma_p):>6}  "
          f"{row.sr:>8.3f}  {row.ar:>+7.1%}")

# Heatmap: top_n vs etf_max (fix mom_lk=12, ma_p=200)
print(f"\n  Sharpe Heatmap — top_n vs etf_max (mom_lk=12, ma_p=200):")
print(f"  {'top_n \\ etf_max':>14}", end="")
for em in grid['etf_max']:
    print(f"  {em:.0%}", end="")
print()
for tn in grid['top_n']:
    print(f"  {tn:>14}", end="")
    for em in grid['etf_max']:
        mask = ((df_grid.top_n==tn) & (df_grid.etf_max==em) &
                (df_grid.mom_lk==12) & (df_grid.ma_p==200))
        r = df_grid[mask]
        val = f"{r.iloc[0]['sr']:.2f}" if len(r) else " -- "
        print(f"  {val:>5}", end="")
    print()

df_grid.to_csv(ROOT/"param_sensitivity.csv", index=False)
print(f"\n  Saved: param_sensitivity.csv ({len(df_grid)} combinations)")
print("═"*70)
