"""
Canyon Quant — Final Strategy  v6.0
=====================================
Confirmed parameters:
  Equity allocation  : 40%  (medium-term momentum Top-25, 21-day rebalancing)
  Leveraged ETF      : 60%  TQQQ  (3x Nasdaq 100)
  Trailing stop      : 20%  (TQQQ exits to cash when falling 20% from intraperiod high)
  High-water-mark stop: 15%  (ETF halved when portfolio NAV pulls back 15% from peak)
  Bear protection    : SPY breaks below 200-day MA → ETF fully in cash
  Transaction cost   : 5 bps one-way

Target: annualized 16-24%, max drawdown ≤ 25%
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════
HOLD_M      = 21
TOP_M       = 25
TCOST       = 5
WARMUP      = 252
SPY_WIN     = 200
STOCK_ALLOC = 0.40
ETF_ALLOC   = 0.60
TRAIL_STOP  = 0.20   # trailing stop 20%
HWM_STOP    = 0.15   # high-water-mark stop 15%
LETF        = "TQQQ"

# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
for fname in ("sp500_price_8yr.csv",):
    prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()

spy_full = prices["SPY"].copy()
prices   = prices.drop(columns=["SPY"])

letf_df  = pd.read_csv(ROOT / "letf_prices.csv", index_col=0, parse_dates=True).sort_index()
letf_df  = letf_df.reindex(prices.index).ffill().bfill()

common   = prices.index.intersection(letf_df.index)
prices   = prices.reindex(common)
letf_df  = letf_df.reindex(common)
spy      = spy_full.reindex(common).ffill()
tqqq_arr = letf_df[LETF].values

print(f"  {len(common)} days | {common[0].date()} → {common[-1].date()}")

# ══════════════════════════════════════════════════════════════════════
def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def regime(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean() else "BEAR"

def sig_medium(t: int) -> pd.Series | None:
    if t < WARMUP:
        return None
    p    = prices.iloc[:t+1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    return (rk((p1m/p13m-1).clip(-1,1))   * 0.35 +
            rk((now/p3m-1).clip(-1,1))     * 0.20 +
            rk(invvol)                      * 0.25 +
            rk((now/p1m-1).clip(-0.5,0.5)) * 0.10 +
            rk((now>sma200).astype(float))  * 0.10).dropna()

def stock_ret(t: int, sig: pd.Series) -> float:
    t_end = min(t+HOLD_M, len(prices)-1)
    sel   = sig.nlargest(TOP_M).index
    r = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1):
            r += (p1/p0-1)/TOP_M
    return r

def tqqq_with_trail(t: int) -> tuple[float, bool]:
    t_end = min(t+HOLD_M, len(tqqq_arr)-1)
    entry = tqqq_arr[t]
    if entry <= 0:
        return 0.0, False
    peak, stopped, exit_day = entry, False, t_end
    for day in range(t+1, t_end+1):
        p = tqqq_arr[day]
        if p > peak:
            peak = p
        if p < peak*(1-TRAIL_STOP):
            exit_day, stopped = day, True
            break
    return float(tqqq_arr[exit_day]/entry - 1), stopped

# ══════════════════════════════════════════════════════════════════════
# Main backtest
# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))
nav, hwm, derisked = 1.0, 1.0, False
records = []

print(f"Running {len(rebal)} periods...")
for t in rebal:
    t_end  = min(t+HOLD_M, len(prices)-1)
    t_date = prices.index[t]
    reg    = regime(t)
    sig    = sig_medium(t)

    # Equities
    sr = stock_ret(t, sig) * (1.0 if reg=="BULL" else 0.5) \
         if sig is not None and len(sig) >= TOP_M else 0.0

    # ETF position (halved when HWM stop triggered)
    etf_w = ETF_ALLOC * (0.5 if derisked else 1.0)

    # LETF (trailing stop in bull, cash in bear)
    if reg == "BULL":
        er, stopped = tqqq_with_trail(t)
    else:
        er, stopped = 0.0, False

    tc  = TCOST/10_000
    ret = STOCK_ALLOC*sr + etf_w*er - tc
    nav *= (1+ret)

    # Update high-water-mark
    if nav > hwm:
        hwm, derisked = nav, False
    elif nav < hwm*(1-HWM_STOP):
        derisked = True

    spy_r = float(spy.iloc[t_end])/float(spy.iloc[t]) - 1

    records.append(dict(
        date      = t_date.strftime("%Y-%m-%d"),
        nav       = round(nav, 6),
        ret       = round(ret, 6),
        stock_ret = round(sr, 6),
        etf_ret   = round(er, 6),
        spy_ret   = round(spy_r, 6),
        regime    = reg,
        derisked  = derisked,
        stopped   = stopped,
    ))

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")

# ══════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════
r   = df["ret"]
n   = len(r)
ppy = 252/HOLD_M
tot = float((1+r).prod())
ar  = float(tot**(ppy/n)-1)
av  = float(r.std()*np.sqrt(ppy))
sr  = ar/(av+1e-9)
cum = (1+r).cumprod()
mdd = float(-((cum-cum.cummax())/cum.cummax()).min())

# Sortino (penalizes downside volatility only)
neg = r[r < 0]
downvol = float(neg.std()*np.sqrt(ppy)) if len(neg) > 0 else 1e-9
sortino = ar/downvol

# Calmar = annualized return / max drawdown
calmar = ar/mdd if mdd > 0 else np.inf

# SPY over same period
spy_r  = df["spy_ret"]
spy_tot = float((1+spy_r).prod())
spy_ar  = float(spy_tot**(ppy/n)-1)
spy_av  = float(spy_r.std()*np.sqrt(ppy))
spy_sr  = spy_ar/(spy_av+1e-9)
spy_cum = (1+spy_r).cumprod()
spy_mdd = float(-((spy_cum-spy_cum.cummax())/spy_cum.cummax()).min())
spy_neg = spy_r[spy_r < 0]
spy_sortino = spy_ar/(spy_neg.std()*np.sqrt(ppy))
spy_calmar  = spy_ar/spy_mdd

# Monthly return distribution
monthly = df.groupby("month")["ret"].apply(lambda x: float((1+x).prod()-1))

# ══════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════
W = 70
print("\n" + "═"*W)
print(f"  CANYON QUANT  —  Final Strategy v6.0")
print(f"  40% Stocks (Momentum) + 60% TQQQ (Stop-Loss Protected)")
print("═"*W)
print(f"  Backtest : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
print(f"  Periods  : {n} × {HOLD_M}d  |  Stop-loss triggers: "
      f"{df['stopped'].sum()} / {n}")
print(f"  De-risked periods (HWM): {df['derisked'].sum()}")
print()

print(f"  {'Metric':<26} {'Strategy':>12} {'SPY':>10} {'vs SPY':>10}")
print("  " + "─"*58)
rows = [
    ("Annual Return",    ar,       spy_ar,       True),
    ("Annual Volatility",av,       spy_av,       False),
    ("Sharpe Ratio",     sr,       spy_sr,       True),
    ("Sortino Ratio",    sortino,  spy_sortino,  True),
    ("Calmar Ratio",     calmar,   spy_calmar,   True),
    ("Max Drawdown",    -mdd,     -spy_mdd,      True),
    ("Total Return",     tot-1,    spy_tot-1,    True),
    ("Win Rate",         float((r>0).mean()), float((spy_r>0).mean()), True),
]
for name, sv, bv, higher_better in rows:
    diff = sv - bv
    if name in ("Sharpe Ratio","Sortino Ratio","Calmar Ratio","Win Rate"):
        print(f"  {name:<26} {sv:>12.2f} {bv:>10.2f} {diff:>+9.2f}")
    else:
        sign = "✓" if (diff > 0) == higher_better else "✗"
        print(f"  {name:<26} {sv:>+11.1%} {bv:>+9.1%} {diff:>+9.1%} {sign}")

print()
print(f"  Year-by-Year")
print(f"  {'Year':>6}  {'Strategy':>9}  {'SPY':>9}  {'Excess':>8}  "
      f"{'Regime':>6}  {'Stop':>4}  {'DeRisk':>6}")
print("  " + "─"*62)
for yr in sorted(df["year"].unique()):
    g   = df[df["year"]==yr]
    pr  = float((1+g["ret"]).prod()-1)
    sr_ = float((1+g["spy_ret"]).prod()-1)
    reg = g["regime"].mode().iloc[0]
    nst = int(g["stopped"].sum())
    ndr = int(g["derisked"].sum())
    exc = pr - sr_
    sym = "✓" if exc > 0 else "✗"
    print(f"  {yr:>6}  {pr:>+8.1%}  {sr_:>+8.1%}  {exc:>+7.1%} {sym}  "
          f"{reg:>6}  {nst:>4}  {ndr:>6}")

print()
print(f"  Monthly Return Distribution ({len(monthly)} months)")
print(f"  Mean    : {monthly.mean():>+.2%}")
print(f"  Median  : {monthly.median():>+.2%}")
print(f"  Std Dev : {monthly.std():>.2%}")
print(f"  Best    : {monthly.max():>+.2%}  ({monthly.idxmax()})")
print(f"  Worst   : {monthly.min():>+.2%}  ({monthly.idxmin()})")
print(f"  >0%     : {(monthly>0).mean():.1%}")
print(f"  >2%/mo  : {(monthly>0.02).mean():.1%}")
print(f"  <-5%    : {(monthly<-0.05).mean():.1%}  (tail risk months)")

print()
print(f"  Strategy Parameters (Confirmed)")
print("  " + "─"*40)
print(f"  Stock alloc     : {STOCK_ALLOC:.0%}  (Top-{TOP_M} momentum, {HOLD_M}d hold)")
print(f"  ETF alloc       : {ETF_ALLOC:.0%}  ({LETF} — 3x Nasdaq 100)")
print(f"  Trailing stop   : {TRAIL_STOP:.0%}  (from intraperiod high)")
print(f"  HWM stop        : {HWM_STOP:.0%}  (ETF halved until new NAV high)")
print(f"  Bear protection : SPY < {SPY_WIN}MA → ETF = 100% cash")
print(f"  Transaction cost: {TCOST} bps one-way")

print()
print(f"  Interpretation")
print("  " + "─"*40)
if ar >= 0.20:
    print(f"  Annual return {ar:.1%} → exceeds 20% target ✓")
elif ar >= 0.16:
    print(f"  Annual return {ar:.1%} → within 16-24% accepted range ✓")
else:
    print(f"  Annual return {ar:.1%} → below target range")
print(f"  Max drawdown  {-mdd:.1%} → {'within' if mdd<=0.25 else 'exceeds'} -25% limit "
      f"{'✓' if mdd<=0.25 else '✗'}")
print(f"  Beats SPY total return: {'YES ✓' if tot-1 > spy_tot-1 else 'NO ✗'} "
      f"({tot-1:+.1%} vs {spy_tot-1:+.1%})")

# Save
df.to_csv(ROOT/"backtest_final_v6.csv", index=False)
print(f"\n  Saved: backtest_final_v6.csv")
print("═"*W)
