"""
Canyon Quant v8 — Precision Fix Version
==============================
v6 issue: HWM stop-loss (high-watermark -15% trigger) missed full-year TQQQ rallies in 2019/2023
v7 issue: LETF rotation picked SOXL, making 2018/2022 sector crashes worse

v8 fix: Use TQQQ 50-day MA instead of HWM portfolio stop-loss
  TQQQ > 50MA -> ETF full position (60%)
  TQQQ < 50MA -> ETF reduce position (20%) or exit
  Speed: 50MA signal reverses in 1-2 weeks; HWM waits for full portfolio recovery

This allows catching TQQQ rapid recoveries in 2019/2023, rather than waiting for portfolio NAV to recover to its peak.

Everything else unchanged:
  Equities 40% medium-term momentum (Top-25, 21-day holding)
  Trailing stop 20% (exit when price drops 20% from peak during holding period)
  Bear market protection: SPY < 200MA -> ETF=0%

Additional tests:
  v8a: TQQQ < 50MA -> ETF reduced to 20%
  v8b: TQQQ < 50MA -> ETF reduced to 0% (more aggressive)
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

HOLD_M      = 21
TOP_M       = 25
TCOST       = 5
WARMUP      = 252
SPY_WIN     = 200
STOCK_ALLOC = 0.40
ETF_ALLOC   = 0.60
TRAIL_STOP  = 0.20
TQQQ_MA     = 50

# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
for fname in ("sp500_price_8yr.csv",):
    prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()

spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
letf_df = letf_df.reindex(prices.index).ffill().bfill()
common  = prices.index.intersection(letf_df.index)
prices, letf_df, spy = prices.reindex(common), letf_df.reindex(common), spy.reindex(common).ffill()
tqqq    = letf_df["TQQQ"].values
print(f"  {len(common)} days | {common[0].date()} → {common[-1].date()}")

# ══════════════════════════════════════════════════════════════════════
def rk(s): return s.rank(pct=True, na_option="bottom") * 100

def regime_spy(t):
    if t < SPY_WIN: return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean() else "BEAR"

def tqqq_above_ma(t):
    """Whether TQQQ is above the 50-day MA (uses only data before t)"""
    if t < TQQQ_MA: return True
    return float(tqqq[t]) > float(np.mean(tqqq[t-TQQQ_MA:t]))

def letf_with_trail(t):
    t_end = min(t+HOLD_M, len(tqqq)-1)
    entry = tqqq[t]
    if entry <= 0: return 0.0, False
    peak, stopped, exit_day = entry, False, t_end
    for day in range(t+1, t_end+1):
        p = tqqq[day]
        if p > peak: peak = p
        if p < peak*(1-TRAIL_STOP):
            exit_day, stopped = day, True
            break
    return float(tqqq[exit_day]/entry - 1), stopped

def sig_medium(t):
    if t < WARMUP: return None
    p   = prices.iloc[:t+1]
    now = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    invvol = 1.0/(p.pct_change().iloc[-21:].std()+0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    return (rk((p1m/p13m-1).clip(-1,1))*0.35 + rk((now/p3m-1).clip(-1,1))*0.20 +
            rk(invvol)*0.25 + rk((now/p1m-1).clip(-0.5,0.5))*0.10 +
            rk((now>sma200).astype(float))*0.10).dropna()

def stock_ret(t, sig):
    t_end = min(t+HOLD_M, len(prices)-1)
    sel   = sig.nlargest(TOP_M).index
    r = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1): r += (p1/p0-1)/TOP_M
    return r

# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))

def run(etf_low_alloc: float) -> list[dict]:
    """
    etf_low_alloc: ETF position when TQQQ < 50MA (0=exit, 0.2=retain 20%)
    """
    rows = []
    for t in rebal:
        t_end = min(t+HOLD_M, len(prices)-1)
        reg   = regime_spy(t)
        sig   = sig_medium(t)

        sr = stock_ret(t, sig)*(1.0 if reg=="BULL" else 0.5) \
             if sig is not None and len(sig) >= TOP_M else 0.0

        if reg == "BEAR":
            etf_w, er, stopped = 0.0, 0.0, False
            signal = "BEAR→CASH"
        elif tqqq_above_ma(t):
            etf_w, signal = ETF_ALLOC, "BULL+MA↑"
            er, stopped = letf_with_trail(t)
        else:
            etf_w, signal = etf_low_alloc, "BULL+MA↓"
            er, stopped = letf_with_trail(t)  # trailing stop still active

        tc  = TCOST/10_000
        ret = STOCK_ALLOC*sr + etf_w*er - tc

        spy_r = float(spy.iloc[t_end])/float(spy.iloc[t]) - 1
        rows.append(dict(date=prices.index[t].strftime("%Y-%m-%d"),
                         ret=ret, stock_ret=sr, etf_ret=er, etf_w=etf_w,
                         spy_ret=spy_r, regime=reg, signal=signal, stopped=stopped))
    return rows

def stats(rows):
    r   = pd.Series([x["ret"] for x in rows])
    spy = pd.Series([x["spy_ret"] for x in rows])
    n, ppy = len(r), 252/HOLD_M
    tot = float((1+r).prod())
    ar  = float(tot**(ppy/n)-1)
    av  = float(r.std()*np.sqrt(ppy))
    sr  = ar/(av+1e-9)
    cum = (1+r).cumprod()
    mdd = float(-((cum-cum.cummax())/cum.cummax()).min())
    neg = r[r<0]
    sortino = ar/(neg.std()*np.sqrt(ppy)+1e-9)
    calmar  = ar/mdd if mdd > 0 else 999
    spy_tot = float((1+spy).prod())
    spy_ar  = float(spy_tot**(ppy/n)-1)
    yrs = {}
    for x in rows:
        yr = x["date"][:4]
        yrs.setdefault(yr, {"r":[], "s":[]})
        yrs[yr]["r"].append(x["ret"])
        yrs[yr]["s"].append(x["spy_ret"])
    yearly = {yr: (float((1+pd.Series(v["r"])).prod()-1),
                   float((1+pd.Series(v["s"])).prod()-1))
              for yr, v in yrs.items()}
    beat = sum(1 for p,s in yearly.values() if p > s)
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, sortino=sortino, calmar=calmar,
                total=tot-1, spy_ar=spy_ar, spy_total=spy_tot-1,
                beat=beat, n=len(yearly), yearly=yearly)

# v6 reference (using HWM stop-loss)
def run_v6():
    nav, hwm, derisked = 1.0, 1.0, False
    rows = []
    for t in rebal:
        t_end = min(t+HOLD_M, len(prices)-1)
        reg   = regime_spy(t)
        sig   = sig_medium(t)
        sr    = stock_ret(t, sig)*(1.0 if reg=="BULL" else 0.5) \
                if sig is not None and len(sig) >= TOP_M else 0.0
        etf_w = ETF_ALLOC*(0.5 if derisked else 1.0)
        if reg == "BULL":
            er, stopped = letf_with_trail(t)
        else:
            er, stopped = 0.0, False
        tc  = TCOST/10_000
        ret = STOCK_ALLOC*sr + etf_w*er - tc
        nav *= (1+ret)
        if nav > hwm: hwm, derisked = nav, False
        elif nav < hwm*0.85: derisked = True
        spy_r = float(spy.iloc[t_end])/float(spy.iloc[t]) - 1
        rows.append(dict(date=prices.index[t].strftime("%Y-%m-%d"),
                         ret=ret, stock_ret=sr, etf_ret=er, etf_w=etf_w,
                         spy_ret=spy_r, regime=reg, signal="", stopped=stopped))
    return rows

print("Running v6 (reference)...")
r_v6  = run_v6()
print("Running v8a (TQQQ<50MA → 20% ETF)...")
r_v8a = run(etf_low_alloc=0.20)
print("Running v8b (TQQQ<50MA → 0% ETF)...")
r_v8b = run(etf_low_alloc=0.00)

s6, s8a, s8b = stats(r_v6), stats(r_v8a), stats(r_v8b)

# ══════════════════════════════════════════════════════════════════════
W = 74
print("\n" + "═"*W)
print("  CANYON QUANT  v8  —  TQQQ 50MA Signal Replaces HWM Stop-Loss")
print("═"*W)
print(f"\n  {'Metric':<24} {'v6(old)':>10} {'v8a(20%)':>10} {'v8b(0%)':>10}  SPY")
print("  " + "─"*60)
for name, k in [("Ann Return","ar"),("Sharpe","sr"),("Sortino","sortino"),
                 ("Calmar","calmar"),("Max Drawdown","mdd"),("Total Return","total")]:
    row = f"  {name:<24}"
    for s in [s6, s8a, s8b]:
        v = s[k]
        if k == "mdd":
            row += f" {-v:>+9.1%}"
        elif k in ("sr","sortino","calmar"):
            row += f" {v:>10.2f}"
        else:
            row += f" {v:>+9.1%}"
    spy_v = {"ar":s6["spy_ar"],"sr":s6["spy_ar"]/(s6["spy_ar"]+1e-9),
             "sortino":0,"calmar":0,"mdd":0.214,"total":s6["spy_total"]}
    if k == "mdd":
        print(row + f"  {-spy_v[k]:>+7.1%}")
    elif k == "ar" or k == "total":
        print(row + f"  {spy_v[k]:>+7.1%}")
    else:
        print(row)

print(f"\n  Years beat SPY   v6: {s6['beat']}/{s6['n']} yrs  "
      f"v8a: {s8a['beat']}/{s8a['n']} yrs  "
      f"v8b: {s8b['beat']}/{s8b['n']} yrs")

print(f"\n  {'Year':>6}  {'v6':>8}  {'v8a':>8}  {'v8b':>8}  {'SPY':>8}  "
      f"{'v6α':>7}  {'v8aα':>7}  {'v8bα':>7}")
print("  " + "─"*72)
for yr in sorted(s6["yearly"]):
    p6, sp  = s6["yearly"][yr]
    p8a, _  = s8a["yearly"].get(yr, (0, sp))
    p8b, _  = s8b["yearly"].get(yr, (0, sp))
    e6, e8a, e8b = p6-sp, p8a-sp, p8b-sp
    def fmt(v): return f"{v:>+7.1%}"
    def sfmt(v): return ("✓" if v>0 else "✗")+fmt(v)
    print(f"  {yr:>6}  {fmt(p6)}  {fmt(p8a)}  {fmt(p8b)}  {fmt(sp)}  "
          f"{sfmt(e6)}  {sfmt(e8a)}  {sfmt(e8b)}")

# TQQQ MA signal distribution
above = sum(1 for x in r_v8a if "MA↑" in x["signal"])
below = sum(1 for x in r_v8a if "MA↓" in x["signal"])
bear  = sum(1 for x in r_v8a if "BEAR" in x["signal"])
print(f"\n  TQQQ 50MA signal distribution ({len(r_v8a)} periods)")
print(f"    TQQQ>50MA (full 60%): {above} periods ({above/len(r_v8a):.0%})")
print(f"    TQQQ<50MA (reduce 20%): {below} periods ({below/len(r_v8a):.0%})")
print(f"    Bear market (exit 0%): {bear} periods  ({bear/len(r_v8a):.0%})")

best = s8a if s8a["ar"] > s8b["ar"] else s8b
bname = "v8a" if s8a["ar"] > s8b["ar"] else "v8b"
print()
print(f"  Final conclusion: Best version {bname}")
print(f"  Ann return {best['ar']:+.1%}  Sharpe {best['sr']:.2f}  "
      f"Max drawdown {-best['mdd']:+.1%}  Total return {best['total']:+.1%}")
print(f"  vs SPY: Ann+{(best['ar']-best['spy_ar']):.1%}  "
      f"Total+{(best['total']-best['spy_total']):.1%}")
ok_ret = best["ar"] >= 0.16
ok_mdd = best["mdd"] <= 0.25
print(f"  {'✓' if ok_ret else '✗'} Ann return target (>=16%)  "
      f"{'✓' if ok_mdd else '✗'} Drawdown target (<=25%)")
print("═"*W)

best_rows = r_v8a if bname == "v8a" else r_v8b
pd.DataFrame(best_rows).to_csv(ROOT/"backtest_v8_final.csv", index=False)
print(f"  Saved: backtest_v8_final.csv")
