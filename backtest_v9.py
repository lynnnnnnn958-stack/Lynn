"""
Canyon Quant v9 — Three-Layer Defense System
=================================

Core issue: Leveraged ETF risk is not that it "will drop," but that it "drops too fast, too deep, leaving no reaction time"
  TQQQ 2022: -79%, averaging -10% per month
  Trailing stop 20%: Only protects on exit day, but 20% is already lost from entry to stop
  SPY 200MA: Signal too slow; SPY only broke below in March 2022, by which time Nasdaq had already fallen 30%

Three-layer defense (layered approach):

  Layer 1 — Volatility Targeting Position
    Core logic: High vol = more risk = hold less; Low vol = stable market = hold more
    TQQQ 20-day realized vol recalculated daily, auto-adjusts position
    Target: ETF contribution to portfolio volatility <= 18%
    Effect: Early 2022 TQQQ vol spike -> position auto-reduced -> much smaller loss
    Formula: etf_w = min(0.60, 0.18 / tqqq_vol_annualized)

  Layer 2 — Dual Signal Gate (both signals must be green for full position)
    Signal A (macro): SPY > 200MA -> bull market
    Signal B (instrument): TQQQ > 50MA -> leveraged ETF itself trending up
    Rules:
      A green B green -> vol-targeted full position (max 60%)
      A green B red -> position halved
      A red (any B) -> exit to cash
    Fast to exit (any red triggers reduction), slow to re-enter (both green required, conservative)

  Layer 3 — Intra-period Trailing Stop (20%)
    Checked daily; exit immediately when price drops 20% from intra-period peak
    After exit, hold cash for remainder of period; re-evaluate at next rebalance date

Equity sleeve (40%):
    Medium-term momentum selection, Top-25, 21-day holding
    Bear market (SPY<200MA): equities reduced to 20% (remainder in cash)
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════
HOLD_M       = 21
TOP_M        = 25
TCOST        = 5           # bps one-way
WARMUP       = 252
SPY_MA_WIN   = 200         # Macro regime signal
TQQQ_MA_WIN  = 50          # TQQQ trend signal
VOL_WIN      = 20          # Volatility calculation window (days)
VOL_TARGET   = 0.18        # Target ETF vol contribution (annualized 18%)
MAX_ETF_W    = 0.60        # ETF maximum position cap
TRAIL_STOP   = 0.20        # Trailing stop threshold

# Equity position
STOCK_BULL   = 0.40        # Bull market equity position
STOCK_BEAR   = 0.20        # Bear market equity position (remainder in cash)

# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
for fname in ("sp500_price_8yr.csv",):
    prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()

spy    = prices["SPY"].copy()
prices = prices.drop(columns=["SPY"])

letf_df = pd.read_csv(ROOT / "letf_prices.csv",
                       index_col=0, parse_dates=True).sort_index()
letf_df = letf_df.reindex(prices.index).ffill().bfill()

common  = prices.index.intersection(letf_df.index)
prices  = prices.reindex(common)
letf_df = letf_df.reindex(common)
spy     = spy.reindex(common).ffill()
tqqq    = letf_df["TQQQ"].values

# Pre-compute TQQQ daily returns (for volatility)
tqqq_ret = np.diff(np.log(np.where(tqqq > 0, tqqq, np.nan)))
tqqq_ret = np.concatenate([[np.nan], tqqq_ret])

print(f"  {len(common)} days | {common[0].date()} → {common[-1].date()}")


# ══════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════
def spy_above_200ma(t: int) -> bool:
    if t < SPY_MA_WIN:
        return True
    return float(spy.iloc[t]) > float(spy.iloc[t - SPY_MA_WIN + 1:t + 1].mean())

def tqqq_above_50ma(t: int) -> bool:
    if t < TQQQ_MA_WIN:
        return True
    return float(tqqq[t]) > float(np.mean(tqqq[t - TQQQ_MA_WIN:t]))

def tqqq_realized_vol(t: int) -> float:
    """20-day realized volatility (annualized), using only data before t"""
    if t < VOL_WIN + 1:
        return 0.40          # Default 40% annualized, conservative
    window = tqqq_ret[t - VOL_WIN:t]
    window = window[np.isfinite(window)]
    if len(window) < 5:
        return 0.40
    return float(np.std(window) * np.sqrt(252))

def etf_weight(t: int) -> tuple[float, str]:
    """
    Three signals combined to determine ETF position:
      1. SPY 200MA (macro)
      2. TQQQ 50MA (trend)
      3. Vol-targeting position (risk)
    Returns: (position, signal description)
    """
    if not spy_above_200ma(t):
        return 0.0, "BEAR_SPY"           # Bear market: exit to cash

    vol = tqqq_realized_vol(t)
    vol_cap = min(MAX_ETF_W, VOL_TARGET / (vol + 0.001))  # vol-target position cap

    if tqqq_above_50ma(t):
        w = vol_cap                       # Both green: vol-targeted full position
        sig = f"BULL_FULL  vol={vol:.0%} w={w:.0%}"
    else:
        w = vol_cap * 0.50               # One green (TQQQ<50MA): halved
        sig = f"BULL_HALF  vol={vol:.0%} w={w:.0%}"

    return float(np.clip(w, 0, MAX_ETF_W)), sig

def letf_period_ret(t: int, etf_w: float) -> tuple[float, bool]:
    """
    TQQQ holding return with trailing stop
    Checked daily; exit when price drops 20% from intra-period peak; hold cash for remainder
    """
    if etf_w <= 0:
        return 0.0, False
    t_end = min(t + HOLD_M, len(tqqq) - 1)
    entry = tqqq[t]
    if entry <= 0 or not np.isfinite(entry):
        return 0.0, False

    peak, stopped, exit_day = entry, False, t_end
    for day in range(t + 1, t_end + 1):
        p = tqqq[day]
        if np.isfinite(p):
            if p > peak:
                peak = p
            if p < peak * (1 - TRAIL_STOP):
                exit_day, stopped = day, True
                break

    exit_price = tqqq[exit_day]
    if not np.isfinite(exit_price) or exit_price <= 0:
        return 0.0, stopped
    return float(exit_price / entry - 1), stopped

def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def sig_medium(t: int) -> pd.Series | None:
    if t < WARMUP:
        return None
    p    = prices.iloc[:t + 1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    return (rk((p1m / p13m - 1).clip(-1, 1))    * 0.35 +
            rk((now / p3m - 1).clip(-1, 1))      * 0.20 +
            rk(invvol)                             * 0.25 +
            rk((now / p1m - 1).clip(-0.5, 0.5))  * 0.10 +
            rk((now > sma200).astype(float))      * 0.10).dropna()

def stock_period_ret(t: int, sig: pd.Series) -> float:
    t_end = min(t + HOLD_M, len(prices) - 1)
    sel   = sig.nlargest(TOP_M).index
    r = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1):
            r += (p1 / p0 - 1) / TOP_M
    return r


# ══════════════════════════════════════════════════════════════════════
# Main Backtest
# ══════════════════════════════════════════════════════════════════════
rebal   = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
records = []

print(f"Running {len(rebal)} rebalance periods...")
for t in rebal:
    t_end  = min(t + HOLD_M, len(prices) - 1)
    t_date = prices.index[t]

    # ETF position (three-layer signal)
    ew, sig_str = etf_weight(t)

    # Equity position (halved in bear market)
    is_bear  = not spy_above_200ma(t)
    stock_w  = STOCK_BEAR if is_bear else STOCK_BULL
    sig_full = sig_medium(t)
    sr = stock_period_ret(t, sig_full) \
         if sig_full is not None and len(sig_full) >= TOP_M else 0.0

    # TQQQ return (with trailing stop)
    er, stopped = letf_period_ret(t, ew)

    tc    = TCOST / 10_000
    ret   = stock_w * sr + ew * er - tc

    spy_r = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1

    records.append(dict(
        date      = t_date.strftime("%Y-%m-%d"),
        ret       = round(ret, 6),
        stock_ret = round(sr, 6),
        stock_w   = round(stock_w, 3),
        etf_ret   = round(er, 6),
        etf_w     = round(ew, 3),
        spy_ret   = round(spy_r, 6),
        tqqq_vol  = round(tqqq_realized_vol(t), 4),
        signal    = sig_str,
        stopped   = stopped,
    ))

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year


# ══════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════
def full_stats(r_series: pd.Series, spy_series: pd.Series,
               label: str = "") -> dict:
    r   = r_series.dropna()
    n   = len(r)
    ppy = 252 / HOLD_M
    tot = float((1 + r).prod())
    ar  = float(tot ** (ppy / n) - 1)
    av  = float(r.std() * np.sqrt(ppy))
    sr  = ar / (av + 1e-9)
    cum = (1 + r).cumprod()
    mdd = float(-((cum - cum.cummax()) / cum.cummax()).min())
    neg = r[r < 0]
    sortino = ar / (neg.std() * np.sqrt(ppy) + 1e-9) if len(neg) > 1 else 0
    calmar  = ar / mdd if mdd > 0 else 999
    # SPY
    spy_tot = float((1 + spy_series).prod())
    spy_ar  = float(spy_tot ** (ppy / n) - 1)
    return dict(label=label, ar=ar, av=av, sr=sr, mdd=mdd,
                sortino=sortino, calmar=calmar, total=tot-1,
                spy_ar=spy_ar, spy_total=spy_tot-1, n=n)

s9  = full_stats(df["ret"], df["spy_ret"], "v9")

# v6 reference (re-run)
def run_v6_ref():
    nav, hwm, derisked = 1.0, 1.0, False
    rows_r, rows_s = [], []
    for t in rebal:
        t_end = min(t+HOLD_M, len(prices)-1)
        reg   = "BULL" if spy_above_200ma(t) else "BEAR"
        sig   = sig_medium(t)
        sr    = stock_period_ret(t, sig)*(1.0 if reg=="BULL" else 0.5) \
                if sig is not None and len(sig)>=TOP_M else 0.0
        etf_w = MAX_ETF_W*(0.5 if derisked else 1.0)
        if reg == "BULL":
            er, _ = letf_period_ret(t, etf_w)
        else:
            er = 0.0
        tc  = TCOST/10_000
        ret = STOCK_BULL*sr + etf_w*er - tc
        nav *= (1+ret)
        if nav > hwm: hwm, derisked = nav, False
        elif nav < hwm*0.85: derisked = True
        rows_r.append(ret)
        rows_s.append(float(spy.iloc[t_end])/float(spy.iloc[t])-1)
    return rows_r, rows_s

print("Running v6 reference...")
r6, s6 = run_v6_ref()
sv6 = full_stats(pd.Series(r6), pd.Series(s6), "v6")

# ══════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════
W = 76
print("\n" + "═"*W)
print("  CANYON QUANT v9  —  Three-Layer Defense System")
print("  Layer1: Vol Targeting  Layer2: Dual Signal Gate  Layer3: Trailing Stop 20%")
print("═"*W)

print(f"\n  {'Metric':<22} {'v6 (old)':>10} {'v9 (new)':>10} {'Change':>10}   SPY")
print("  " + "─"*60)
metrics = [
    ("Ann Return",  "ar",      "+.1%",  True),
    ("Ann Vol",     "av",      "+.1%",  False),
    ("Sharpe",      "sr",      ".2f",   True),
    ("Sortino",     "sortino", ".2f",   True),
    ("Calmar",      "calmar",  ".2f",   True),
    ("Max Drawdown","mdd",     "DD",    True),
    ("Total Return","total",   "+.1%",  True),
]
spy_ref = dict(ar=sv6["spy_ar"], av=0, sr=sv6["spy_ar"]/0.159,
               sortino=0, calmar=sv6["spy_ar"]/0.214,
               mdd=0.214, total=sv6["spy_total"])

for name, k, fmt, hi_better in metrics:
    v6v = sv6[k]
    v9v = s9[k]
    dif = v9v - v6v
    spy_v = spy_ref.get(k, 0)

    def ffmt(v, f):
        if f == "+.1%": return f"{v:>+9.1%}"
        if f == "DD":   return f"{-v:>+9.1%}"
        return f"{v:>10.2f}"

    sym = "↑" if (dif > 0) == hi_better else "↓"
    if fmt == "DD":
        diff_str = f"{dif:>+9.1%} {sym}"
    elif fmt == "+.1%":
        diff_str = f"{dif:>+9.1%} {sym}"
    else:
        diff_str = f"{dif:>+9.2f} {sym}"

    spy_str = ffmt(spy_v, fmt)
    print(f"  {name:<22} {ffmt(v6v,fmt)} {ffmt(v9v,fmt)} {diff_str}  {spy_str}")

# Annual comparison
print(f"\n  {'Year':>6}  {'v6':>8}  {'v9':>8}  {'SPY':>8}  "
      f"{'v6α':>7}  {'v9α':>7}  {'ETF Wt':>6}")
print("  " + "─"*66)

v6_by_yr = {}
for i, t in enumerate(rebal):
    yr = prices.index[t].year
    v6_by_yr.setdefault(yr, []).append(r6[i])

for yr in sorted(df["year"].unique()):
    g   = df[df["year"] == yr]
    p9  = float((1+g["ret"]).prod()-1)
    sp  = float((1+g["spy_ret"]).prod()-1)
    p6  = float((1+pd.Series(v6_by_yr.get(yr,[0]))).prod()-1)
    e6  = p6 - sp
    e9  = p9 - sp
    avg_w = float(g["etf_w"].mean())
    s6sym = "✓" if e6 > 0 else "✗"
    s9sym = "✓" if e9 > 0 else "✗"
    print(f"  {yr:>6}  {p6:>+7.1%}  {p9:>+7.1%}  {sp:>+7.1%}  "
          f"{s6sym}{e6:>+5.1%}  {s9sym}{e9:>+5.1%}  {avg_w:.0%}")

# Signal distribution
print(f"\n  ETF weight distribution (v9, {len(df)} periods)")
df["signal_cat"] = df["signal"].apply(
    lambda x: "BEAR_FLAT" if "BEAR" in x else (
              "DUAL_FULL" if "FULL" in x else "SINGLE_HALF"))
for cat, grp in df.groupby("signal_cat"):
    n  = len(grp)
    aw = grp["etf_w"].mean()
    ar_g = float((1+grp["ret"]).prod()-1)
    print(f"    {cat:<12}  {n:>3} periods ({n/len(df):.0%})  "
          f"avg ETF weight {aw:.0%}  segment total return {ar_g:+.1%}")

print(f"\n  Trailing stop triggered: {df['stopped'].sum()}/{len(df)} periods")
print(f"  Avg TQQQ weight: {df['etf_w'].mean():.1%}  "
      f"vol range: {df['tqqq_vol'].min():.0%} - {df['tqqq_vol'].max():.0%}")

# Target check
print()
print("  ▸ Target Check")
targets = [
    ("Annualized return >= 16%",  s9["ar"]     >= 0.16),
    ("Max drawdown <= 25%",       s9["mdd"]    <= 0.25),
    ("Beat SPY total return",     s9["total"]  >  s9["spy_total"]),
    ("Sharpe > 0.60",             s9["sr"]     >  0.60),
]
all_pass = all(v for _, v in targets)
for name, ok in targets:
    print(f"    {'✓' if ok else '✗'}  {name}")
print(f"\n  {'[ALL PASS]' if all_pass else '[SOME FAILED]'}")
print("═"*W)

df.to_csv(ROOT / "backtest_v9_final.csv", index=False)
print(f"  Saved: backtest_v9_final.csv")
