"""
Canyon Quant v10 — Full Integrated Version
================================
Equities 40%  x Quality-filtered momentum stock selection
  Filter: RSI(14)<75  |  Price<200MA*1.5  |  Volatility<75pct
  Signal: 12M momentum 35% + 3M momentum 20% + Low vol 25% + 1M momentum 10% + Trend 10%

Leveraged ETF 60%  x Three-layer defense
  Layer1 Vol-targeting position: TQQQ 20-day vol -> weight=min(60%, 18%/vol)
  Layer2 Dual signal gate: SPY>200MA AND TQQQ>50MA -> full position; else halve or exit
  Layer3 Trailing stop 20%: exit when price drops 20% from peak during holding period
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
TQQQ_MA_WIN = 50
VOL_WIN     = 20
VOL_TARGET  = 0.18
MAX_ETF_W   = 0.60
TRAIL_STOP  = 0.20
STOCK_BULL  = 0.40
STOCK_BEAR  = 0.20

# ── Load Data ──────────────────────────────────────────────────────────
print("Loading data...")
prices  = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
letf_df = letf_df.reindex(prices.index).ffill().bfill()
common  = prices.index.intersection(letf_df.index)
prices, letf_df, spy = prices.reindex(common), letf_df.reindex(common), spy.reindex(common).ffill()
tqqq      = letf_df["TQQQ"].values
tqqq_logr = np.concatenate([[np.nan], np.diff(np.log(np.where(tqqq>0, tqqq, np.nan)))])
print(f"  {len(common)} days  {common[0].date()} → {common[-1].date()}")

# ── Signal Utilities ──────────────────────────────────────────────────────────
def rk(s): return s.rank(pct=True, na_option="bottom") * 100

def spy_bull(t):
    return t < SPY_WIN or float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean()

def tqqq_bull(t):
    return t < TQQQ_MA_WIN or float(tqqq[t]) > np.mean(tqqq[t-TQQQ_MA_WIN:t])

def tqqq_vol(t):
    w = tqqq_logr[max(0,t-VOL_WIN):t]
    w = w[np.isfinite(w)]
    return float(np.std(w)*np.sqrt(252)) if len(w)>=5 else 0.40

def etf_alloc(t):
    if not spy_bull(t):  return 0.0, "BEAR"
    vol  = tqqq_vol(t)
    cap  = min(MAX_ETF_W, VOL_TARGET/(vol+0.001))
    return (cap if tqqq_bull(t) else cap*0.5), ("FULL" if tqqq_bull(t) else "HALF")

def tqqq_ret(t, w):
    if w <= 0: return 0.0, False
    t1 = min(t+HOLD_M, len(tqqq)-1)
    e  = tqqq[t]
    if e <= 0 or not np.isfinite(e): return 0.0, False
    peak, stop, ed = e, False, t1
    for d in range(t+1, t1+1):
        p = tqqq[d]
        if np.isfinite(p):
            peak = max(peak, p)
            if p < peak*(1-TRAIL_STOP):
                ed, stop = d, True; break
    ep = tqqq[ed]
    return (float(ep/e-1) if np.isfinite(ep) and ep>0 else 0.0), stop

# ── Equities: Quality Filter + Momentum ─────────────────────────────────────────────
def stock_signal(t):
    if t < WARMUP: return None
    p    = prices.iloc[:t+1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t>=63  else p.iloc[0]
    p13m = p.iloc[-274] if t>=273 else (p.iloc[-252] if t>=252 else p.iloc[0])

    # Momentum score
    invvol = 1.0/(p.pct_change().iloc[-21:].std()+0.005)
    sma200 = p.iloc[-200:].mean() if t>=199 else p.mean()
    mom = (rk((p1m/p13m-1).clip(-1,1))   *0.35 +
           rk((now/p3m-1).clip(-1,1))     *0.20 +
           rk(invvol)                      *0.25 +
           rk((now/p1m-1).clip(-0.5,0.5)) *0.10 +
           rk((now>sma200).astype(float))  *0.10).dropna()

    # Quality filter
    delta = p.pct_change().iloc[-15:]
    gains = delta.clip(lower=0).iloc[-14:].mean()
    loss  = (-delta.clip(upper=0)).iloc[-14:].mean()
    rsi14 = 100 - 100/(1+gains/(loss+1e-9))
    dist  = now/(sma200+1e-9) - 1
    vol20 = p.pct_change().iloc[-21:].std()
    vol75 = vol20.quantile(0.75)

    quality_mask = (rsi14 < 75) & (dist < 0.50) & (vol20 < vol75)
    valid = mom[mom.index.isin(quality_mask[quality_mask].index)]
    pool  = valid if len(valid) >= TOP_M else mom  # fall back to unfiltered if too few pass quality screen
    return pool.nlargest(TOP_M).index.tolist()

def stock_period_ret(t, sel):
    t1 = min(t+HOLD_M, len(prices)-1)
    r  = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk,np.nan))
        p1 = float(prices.iloc[t1].get(tk,np.nan))
        if p0>0 and np.isfinite(p1): r += (p1/p0-1)/len(sel)
    return r

# ── Main Backtest ────────────────────────────────────────────────────────────
rebal   = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))
records = []
print(f"Running {len(rebal)} periods (v10: quality filter + vol targeting + dual gate + trail stop)...")

for t in rebal:
    t_date = prices.index[t]
    t1     = min(t+HOLD_M, len(prices)-1)
    is_bull = spy_bull(t)
    sw      = STOCK_BULL if is_bull else STOCK_BEAR

    sel  = stock_signal(t)
    sr   = stock_period_ret(t, sel)*sw if sel else 0.0

    ew, sig = etf_alloc(t)
    er, stopped = tqqq_ret(t, ew)

    tc  = TCOST/10_000
    ret = sr + ew*er - tc
    spy_r = float(spy.iloc[t1])/float(spy.iloc[t])-1

    records.append(dict(
        date=t_date.strftime("%Y-%m-%d"), ret=round(ret,6),
        stock_ret=round(sr,6), etf_ret=round(er,6),
        etf_w=round(ew,3), spy_ret=round(spy_r,6),
        tqqq_vol=round(tqqq_vol(t),4), signal=sig, stopped=stopped,
        hold=HOLD_M))

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# ── Statistics ──────────────────────────────────────────────────────────────
def stats(r, spy_r=None):
    r = pd.Series(r)
    n, ppy = len(r), 252/HOLD_M
    tot = float((1+r).prod())
    ar  = float(tot**(ppy/n)-1)
    av  = float(r.std()*np.sqrt(ppy))
    sr  = ar/(av+1e-9)
    cum = (1+r).cumprod()
    mdd = float(-((cum-cum.cummax())/cum.cummax()).min())
    neg = r[r<0]
    so  = ar/(neg.std()*np.sqrt(ppy)+1e-9) if len(neg)>1 else 0
    cal = ar/mdd if mdd>0 else 999
    out = dict(ar=ar, av=av, sr=sr, mdd=mdd, sortino=so, calmar=cal, total=tot-1, n=n)
    if spy_r is not None:
        sr2 = pd.Series(spy_r)
        st  = float((1+sr2).prod())
        out.update(spy_ar=float(st**(ppy/n)-1), spy_total=st-1)
    return out

s10  = stats(df["ret"], df["spy_ret"])

# v6 reference (quick re-run)
def run_v6():
    nav, hwm, der = 1.0, 1.0, False
    rr, ss = [], []
    for t in rebal:
        t1 = min(t+HOLD_M, len(prices)-1)
        is_b = spy_bull(t)
        sel  = stock_signal(t)
        sw   = STOCK_BULL if is_b else STOCK_BEAR
        sr   = stock_period_ret(t, sel)*sw if sel else 0.0
        ew   = MAX_ETF_W*(0.5 if der else 1.0)
        er   = 0.0
        if is_b:
            er, _ = tqqq_ret(t, ew)
        ret  = STOCK_BULL*(stock_period_ret(t,sel) if sel else 0)*( 1 if is_b else 0.5) + ew*er - TCOST/10_000
        nav *= (1+ret)
        if nav>hwm: hwm, der = nav, False
        elif nav<hwm*0.85: der = True
        rr.append(ret); ss.append(float(spy.iloc[t1])/float(spy.iloc[t])-1)
    return rr, ss

print("Running v6 reference...")
rv6, sv6 = run_v6()
s6 = stats(rv6, sv6)

# ── Output ──────────────────────────────────────────────────────────────
W = 76
print("\n"+"═"*W)
print("  CANYON QUANT v10  —  Quality Filter + Vol Targeting + Dual Signal Gate + Trailing Stop")
print("═"*W)

print(f"\n  {'Metric':<22} {'v6':>10} {'v10':>10} {'Improvement':>9}   SPY")
print("  "+"─"*58)
spy_ref = dict(ar=s10["spy_ar"], av=0.159, sr=s10["spy_ar"]/0.159,
               sortino=0, calmar=s10["spy_ar"]/0.214, mdd=0.214, total=s10["spy_total"])
for name, k, is_mdd in [
    ("Ann Return","ar",False),("Ann Vol","av",False),("Sharpe","sr",False),
    ("Sortino","sortino",False),("Calmar","calmar",False),
    ("Max Drawdown","mdd",True),("Total Return","total",False)]:
    v6v, v10v = s6[k], s10[k]
    dif = v10v - v6v
    spv = spy_ref[k]
    if k in ("sr","sortino","calmar"):
        print(f"  {name:<22} {v6v:>10.2f} {v10v:>10.2f} {dif:>+8.2f}   {spv:.2f}")
    elif is_mdd:
        print(f"  {name:<22} {-v6v:>+9.1%} {-v10v:>+9.1%} {-dif:>+8.1%}  {-spv:>+7.1%}")
    else:
        print(f"  {name:<22} {v6v:>+9.1%} {v10v:>+9.1%} {dif:>+8.1%}  {spv:>+7.1%}")

print(f"\n  {'Year':>6}  {'v6':>8}  {'v10':>8}  {'SPY':>8}  "
      f"{'v6α':>7}  {'v10α':>7}  ETF Wt")
print("  "+"─"*66)

v6y = {}
for i, t in enumerate(rebal):
    yr = prices.index[t].year
    v6y.setdefault(yr,[]).append(rv6[i])

for yr in sorted(df["year"].unique()):
    g   = df[df["year"]==yr]
    p10 = float((1+g["ret"]).prod()-1)
    sp  = float((1+g["spy_ret"]).prod()-1)
    p6  = float((1+pd.Series(v6y.get(yr,[0]))).prod()-1) if yr in v6y else 0
    e6  = p6-sp; e10 = p10-sp
    aw  = float(g["etf_w"].mean())
    print(f"  {yr:>6}  {p6:>+7.1%}  {p10:>+7.1%}  {sp:>+7.1%}  "
          f"{'✓' if e6>0 else '✗'}{e6:>+5.1%}  {'✓' if e10>0 else '✗'}{e10:>+5.1%}  {aw:.0%}")

beat = sum(1 for yr in df["year"].unique()
           if float((1+df[df["year"]==yr]["ret"]).prod()-1) >
              float((1+df[df["year"]==yr]["spy_ret"]).prod()-1))
print(f"\n  Years beat SPY: v6 {sum(1 for yr in v6y if float((1+pd.Series(v6y[yr])).prod()-1)>float((1+df[df['year']==yr]['spy_ret']).prod()-1) if yr in v6y)}/9  ->  v10 {beat}/9")

print(f"\n  TQQQ avg position: {df['etf_w'].mean():.0%}  "
      f"Trailing stop triggered: {df['stopped'].sum()}/{len(df)} periods  "
      f"Vol range: {df['tqqq_vol'].min():.0%}–{df['tqqq_vol'].max():.0%}")

print(f"\n  ▸ Goal Check")
checks = [
    ("Ann Return >= 16%",   s10["ar"]    >= 0.16),
    ("Max Drawdown <= 25%",   s10["mdd"]   <= 0.25),
    ("Sharpe > 0.65",    s10["sr"]    >  0.65),
    ("Beat SPY Total Return",    s10["total"] >  s10["spy_total"]),
]
for name, ok in checks:
    print(f"    {'✓' if ok else '✗'}  {name}")
all_ok = all(v for _,v in checks)
print(f"\n  {'[ALL GOALS MET]' if all_ok else '[SOME GOALS UNMET]'}")
print("═"*W)

df.to_csv(ROOT/"backtest_v10_final.csv", index=False)
print("  Saved: backtest_v10_final.csv")
