"""
Attribution Analysis: Identify "low-quality stocks that were killed"
====================================================================
Goal: Diagnose the drag sources in the equity leg, then add a quality
filter to exclude them.

Quality filter logic:
  Momentum-selected stocks may be "bubble stocks" — briefly driven up
  but fundamentally empty.
  Characteristics: extreme RSI overbought + price far above 200MA + abnormally high volatility.
  Such stocks often fall 2-3x harder than normal stocks during market pullbacks.

Filter rules (checked at entry; all must pass before buying):
  1. RSI(14) < 75      — no chasing highs; exclude extreme overbought
  2. Price < 200MA × 1.5 — no stocks >50% above MA (bubble characteristic)
  3. 20-day volatility < 75th percentile of market — exclude high-risk high-beta stocks
  4. 1-month return not in worst 20% (prevent momentum reversal)
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

HOLD_M = 21
TOP_M  = 25
TCOST  = 5
WARMUP = 252
SPY_WIN = 200

print("Loading data...")
prices = pd.read_csv(ROOT / "sp500_price_8yr.csv",
                     index_col=0, parse_dates=True).sort_index()
spy    = prices["SPY"].copy()
prices = prices.drop(columns=["SPY"])
print(f"  {prices.shape}")

def rk(s): return s.rank(pct=True, na_option="bottom") * 100

def regime(t):
    if t < SPY_WIN: return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean() else "BEAR"

# ── Compute signals ────────────────────────────────────────────────────────────
def compute_signals(t):
    """Return (momentum_score, quality_flags) as two DataFrames"""
    if t < WARMUP:
        return None, None
    p    = prices.iloc[:t+1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])

    # ─ Momentum score (original strategy)
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    mom_score = (rk((p1m/p13m-1).clip(-1,1)) * 0.35 +
                 rk((now/p3m-1).clip(-1,1))   * 0.20 +
                 rk(invvol)                    * 0.25 +
                 rk((now/p1m-1).clip(-0.5,0.5))* 0.10 +
                 rk((now>sma200).astype(float)) * 0.10)

    # ─ Quality metrics
    # RSI(14)
    delta  = p.pct_change().iloc[-15:]
    gains  = delta.clip(lower=0).iloc[-14:].mean()
    losses = (-delta.clip(upper=0)).iloc[-14:].mean()
    rsi14  = 100 - 100 / (1 + gains / (losses + 1e-9))

    # Price deviation from 200MA
    dist_200ma = now / (sma200 + 1e-9) - 1   # >0 = how far above 200MA

    # 20-day volatility (individual stocks)
    vol20  = p.pct_change().iloc[-21:].std()

    # 1-month return (exclude momentum-reversal candidates)
    ret1m  = now / p1m - 1

    quality = pd.DataFrame({
        "rsi14":      rsi14,
        "dist_200ma": dist_200ma,
        "vol20":      vol20,
        "ret1m":      ret1m,
        "mom_score":  mom_score,
    }).dropna()

    return mom_score.dropna(), quality


def apply_quality_filter(quality: pd.DataFrame) -> pd.Index:
    """
    Quality filter: exclude stocks with the following characteristics
      1. RSI > 75 (extreme overbought, high momentum-reversal risk)
      2. Price more than 50% above 200MA (price bubble)
      3. 20-day volatility > 75th percentile (high-risk high-beta)
    """
    vol_75pct = quality["vol20"].quantile(0.75)
    mask = (
        (quality["rsi14"]      < 75) &        # not extreme overbought
        (quality["dist_200ma"] < 0.50) &      # not >50% above 200MA
        (quality["vol20"]      < vol_75pct)   # not high volatility
    )
    return quality[mask].index


# ══════════════════════════════════════════════════════════════════════
# Two backtests: original vs quality-filtered
# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))

def stock_ret_detail(t, selected):
    """Return per-stock returns (for attribution)"""
    t_end = min(t + HOLD_M, len(prices) - 1)
    ret_map = {}
    for tk in selected:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1):
            ret_map[tk] = p1 / p0 - 1
    return ret_map

rows_orig, rows_filt = [], []
worst_positions = []   # Record worst-period holding details

for t in rebal:
    t_date = prices.index[t]
    reg    = regime(t)
    scale  = 1.0 if reg == "BULL" else 0.5

    mom, qual = compute_signals(t)
    if mom is None:
        continue

    t_end   = min(t + HOLD_M, len(prices) - 1)
    spy_r   = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1
    tc      = TCOST / 10_000

    # ── Original: pure momentum Top-25
    sel_orig = mom.nlargest(TOP_M).index.tolist()
    ret_orig_map = stock_ret_detail(t, sel_orig)
    r_orig   = np.mean(list(ret_orig_map.values())) * scale - tc

    # ── Quality version: quality-filtered then momentum Top-25
    valid_tks  = apply_quality_filter(qual)
    mom_filt   = mom[mom.index.isin(valid_tks)]
    n_after    = len(mom_filt)
    if n_after >= TOP_M:
        sel_filt = mom_filt.nlargest(TOP_M).index.tolist()
    else:
        sel_filt = mom_filt.nlargest(max(10, n_after)).index.tolist()

    ret_filt_map = stock_ret_detail(t, sel_filt)
    r_filt   = np.mean(list(ret_filt_map.values())) * scale - tc if ret_filt_map else 0.0

    rows_orig.append(dict(date=t_date.strftime("%Y-%m-%d"), ret=r_orig,
                          spy_ret=spy_r, regime=reg, hold=HOLD_M, n_sel=len(sel_orig)))
    rows_filt.append(dict(date=t_date.strftime("%Y-%m-%d"), ret=r_filt,
                          spy_ret=spy_r, regime=reg, hold=HOLD_M, n_sel=len(sel_filt),
                          n_universe=n_after))

    # Record worst period: original return < -5%
    if r_orig < -0.05:
        worst_map = ret_orig_map
        worst_positions.append({
            "date":    t_date.strftime("%Y-%m-%d"),
            "regime":  reg,
            "port_ret": r_orig,
            "spy_ret": spy_r,
            "stocks":  {tk: {"ret": v,
                             "rsi14": float(qual.loc[tk, "rsi14"]) if tk in qual.index else np.nan,
                             "dist_200ma": float(qual.loc[tk, "dist_200ma"]) if tk in qual.index else np.nan,
                             "vol20": float(qual.loc[tk, "vol20"]) if tk in qual.index else np.nan}
                        for tk, v in sorted(worst_map.items(), key=lambda x: x[1])[:10]}
        })

df_orig = pd.DataFrame(rows_orig)
df_filt = pd.DataFrame(rows_filt)

# ══════════════════════════════════════════════════════════════════════
def stats(df):
    r   = df["ret"]
    n, ppy = len(r), 252/HOLD_M
    tot = float((1+r).prod())
    ar  = float(tot**(ppy/n)-1)
    av  = float(r.std()*np.sqrt(ppy))
    cum = (1+r).cumprod()
    mdd = float(-((cum-cum.cummax())/cum.cummax()).min())
    sr  = ar/(av+1e-9)
    neg = r[r<0]
    so  = ar/(neg.std()*np.sqrt(ppy)+1e-9) if len(neg)>1 else 0
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, sortino=so, total=tot-1)

s_orig = stats(df_orig)
s_filt = stats(df_filt)

W = 72
print("\n" + "═"*W)
print("  Attribution Analysis — Quality Filter Effect")
print("  Filter: RSI>75 | Deviation from 200MA>50% | Volatility>75pct → Excluded")
print("═"*W)

print(f"\n  {'Metric':<20} {'Original (Pure Momentum)':>24} {'Quality-Filtered':>18} {'Improvement':>12}")
print("  " + "─"*58)
for name, k in [("Annual Return","ar"),("Sharpe","sr"),("Sortino","sortino"),
                 ("Max Drawdown","mdd"),("Total Return","total")]:
    vo, vf = s_orig[k], s_filt[k]
    diff   = vf - vo
    if k == "mdd":
        print(f"  {name:<20} {-vo:>+13.1%} {-vf:>+13.1%} {-diff:>+9.1%}")
    elif k in ("sr","sortino"):
        print(f"  {name:<20} {vo:>14.2f} {vf:>14.2f} {diff:>+9.2f}")
    else:
        print(f"  {name:<20} {vo:>+13.1%} {vf:>+13.1%} {diff:>+9.1%}")

# Year-by-year
print(f"\n  {'Year':>6}  {'Original':>9}  {'Quality':>9}  {'SPY':>9}  {'Improvement':>12}")
print("  " + "─"*50)
df_orig["year"] = pd.to_datetime(df_orig["date"]).dt.year
df_filt["year"] = pd.to_datetime(df_filt["date"]).dt.year
for yr in sorted(df_orig["year"].unique()):
    go = df_orig[df_orig["year"]==yr]
    gf = df_filt[df_filt["year"]==yr]
    ro = float((1+go["ret"]).prod()-1)
    rf = float((1+gf["ret"]).prod()-1)
    rs = float((1+go["spy_ret"]).prod()-1)
    print(f"  {yr:>6}  {ro:>+8.1%}  {rf:>+8.1%}  {rs:>+8.1%}  {rf-ro:>+7.1%}")

# Worst-period attribution
print(f"\n  Worst holding period details (equity portfolio fell >5%)")
print(f"  {len(worst_positions)} periods total; showing 5 worst:")
worst_positions.sort(key=lambda x: x["port_ret"])
for wp in worst_positions[:5]:
    print(f"\n  ─── {wp['date']}  Portfolio={wp['port_ret']:+.1%}  SPY={wp['spy_ret']:+.1%}")
    print(f"  {'Ticker':<6}  {'Return':>8}  {'RSI14':>7}  {'Dist200MA':>10}  {'20dVol':>8}  Issue")
    for tk, info in list(wp["stocks"].items())[:8]:
        rsi  = info["rsi14"]
        dist = info["dist_200ma"]
        vol  = info["vol20"] * np.sqrt(252) if np.isfinite(info["vol20"]) else np.nan
        flags = []
        if np.isfinite(rsi)  and rsi  > 75:  flags.append("Overbought RSI")
        if np.isfinite(dist) and dist > 0.5:  flags.append("Dist200MA>50%")
        if np.isfinite(vol)  and vol  > 0.6:  flags.append("High Volatility")
        flag_str = "+".join(flags) if flags else "—"
        print(f"  {tk:<6}  {info['ret']:>+7.1%}  "
              f"{rsi:>6.0f}%  {dist:>+7.0%}  {vol:>7.0%}  {flag_str}")

# Filter effect quantification
print(f"\n  Filter exclusion (average per period)")
avg_universe = df_filt["n_universe"].mean()
avg_sel_orig = df_orig["n_sel"].mean()
avg_sel_filt = df_filt["n_sel"].mean()
total_tks    = len(prices.columns)
print(f"  Full universe        : {total_tks} stocks")
print(f"  After filter         : {avg_universe:.0f} stocks (excluded {total_tks-avg_universe:.0f}, "
      f"{1-avg_universe/total_tks:.0%})")
print(f"  Final selected       : {avg_sel_filt:.0f} stocks → Top 25")

print("\n" + "═"*W)

# Save quality-filtered holdings
df_filt.to_csv(ROOT / "stock_quality_filtered.csv", index=False)
print("  Saved: stock_quality_filtered.csv")
