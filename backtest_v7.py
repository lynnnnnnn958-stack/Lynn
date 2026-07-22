"""
Canyon Quant v7 — Improved version
==========================
Core changes vs v6:

  [Fix] Removed high-water-mark stop (HWM stop)
         Reason: In the 2019/2023 bull years, HWM stop kept ETF locked at 30% allocation for a full year,
               missing TQQQ +114% (2019) and +198% (2023) gains.

  [New] LETF momentum rotation
         Each period picks the strongest 20-day momentum LETF from {TQQQ, SOXL, UPRO}
         SOXL (3x semiconductor) significantly outperformed TQQQ in 2023/2024

  [Retained] Trailing stop 20% (checked daily; exit when down 20% from intra-period high)
  [Retained] Bear protection: SPY < 200MA → full cash
  [Retained] Equities 40% medium-term momentum strategy
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
MOM_WIN     = 20          # LETF momentum window (days)
LETF_CANDS  = ["TQQQ", "SOXL", "UPRO"]   # Rotation candidates

# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
for fname in ("sp500_price_8yr.csv",):
    prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()

spy = prices["SPY"].copy()
prices = prices.drop(columns=["SPY"])

letf_df = pd.read_csv(ROOT / "letf_prices.csv", index_col=0, parse_dates=True).sort_index()
letf_df = letf_df.reindex(prices.index).ffill().bfill()
common  = prices.index.intersection(letf_df.index)
prices  = prices.reindex(common)
letf_df = letf_df.reindex(common)
spy     = spy.reindex(common).ffill()

print(f"  {len(common)} days | {common[0].date()} → {common[-1].date()}")

# ══════════════════════════════════════════════════════════════════════
def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def regime(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t-SPY_WIN+1:t+1].mean() else "BEAR"

def best_letf(t: int) -> str:
    """Pick the LETF with strongest 20-day momentum (uses only data before t; no lookahead)."""
    if t < MOM_WIN:
        return "TQQQ"
    best, best_mom = "TQQQ", -999.0
    for tk in LETF_CANDS:
        if tk not in letf_df.columns:
            continue
        p0 = float(letf_df[tk].iloc[t - MOM_WIN])
        p1 = float(letf_df[tk].iloc[t])
        if p0 > 0:
            mom = p1 / p0 - 1
            if mom > best_mom:
                best_mom = mom
                best = tk
    return best

def letf_with_trail(tk: str, t: int) -> tuple[float, bool]:
    """LETF return with trailing stop: exit when down TRAIL_STOP% from intra-period high."""
    t_end  = min(t + HOLD_M, len(letf_df) - 1)
    prices_tk = letf_df[tk].values
    entry  = prices_tk[t]
    if entry <= 0:
        return 0.0, False
    peak, stopped, exit_day = entry, False, t_end
    for day in range(t + 1, t_end + 1):
        p = prices_tk[day]
        if p > peak:
            peak = p
        if p < peak * (1 - TRAIL_STOP):
            exit_day, stopped = day, True
            break
    return float(prices_tk[exit_day] / entry - 1), stopped

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
# v6 baseline (with HWM, fixed TQQQ) vs v7 comparison
# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))

def run(use_rotation: bool, use_hwm: bool, hwm_thresh: float = 0.15) -> list[dict]:
    nav, hwm, derisked = 1.0, 1.0, False
    rows = []
    for t in rebal:
        t_end = min(t + HOLD_M, len(prices) - 1)
        reg   = regime(t)
        sig   = sig_medium(t)

        sr = stock_ret(t, sig) * (1.0 if reg == "BULL" else 0.5) \
             if sig is not None and len(sig) >= TOP_M else 0.0

        etf_w = ETF_ALLOC * (0.5 if (use_hwm and derisked) else 1.0)

        if reg == "BULL":
            tk = best_letf(t) if use_rotation else "TQQQ"
            er, stopped = letf_with_trail(tk, t)
        else:
            tk, er, stopped = "CASH", 0.0, False

        tc  = TCOST / 10_000
        ret = STOCK_ALLOC * sr + etf_w * er - tc
        nav *= (1 + ret)

        if use_hwm:
            if nav > hwm:
                hwm, derisked = nav, False
            elif nav < hwm * (1 - hwm_thresh):
                derisked = True

        spy_r = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1
        rows.append(dict(date=prices.index[t].strftime("%Y-%m-%d"),
                         nav=nav, ret=ret, stock_ret=sr, etf_ret=er,
                         spy_ret=spy_r, regime=reg, letf=tk, stopped=stopped))
    return rows

print("Running v6 (HWM on, fixed TQQQ)...")
rows_v6 = run(use_rotation=False, use_hwm=True)
print("Running v7 (no HWM, LETF rotation)...")
rows_v7 = run(use_rotation=True,  use_hwm=False)

# ══════════════════════════════════════════════════════════════════════
def stats(rows: list[dict]) -> dict:
    r   = pd.Series([x["ret"] for x in rows])
    spy = pd.Series([x["spy_ret"] for x in rows])
    n   = len(r)
    ppy = 252 / HOLD_M
    tot = float((1+r).prod())
    ar  = float(tot**(ppy/n)-1)
    av  = float(r.std()*np.sqrt(ppy))
    sr  = ar/(av+1e-9)
    cum = (1+r).cumprod()
    mdd = float(-((cum-cum.cummax())/cum.cummax()).min())
    neg = r[r<0]
    sortino = ar/(neg.std()*np.sqrt(ppy)+1e-9)
    calmar  = ar/mdd if mdd>0 else 999
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
    beat_spy = sum(1 for p, s in yearly.values() if p > s)
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, sortino=sortino, calmar=calmar,
                total=tot-1, spy_ar=spy_ar, spy_total=spy_tot-1,
                beat_spy=beat_spy, total_years=len(yearly), yearly=yearly)

s6 = stats(rows_v6)
s7 = stats(rows_v7)

# ══════════════════════════════════════════════════════════════════════
W = 72
print("\n" + "═"*W)
print("  CANYON QUANT  v6  →  v7  Comparison")
print("═"*W)
print(f"  {'Metric':<24} {'v6 (old)':>10} {'v7 (new)':>10} {'Improvement':>10}")
print("  " + "─"*56)
for name, k, fmt in [
    ("Annualized Return", "ar",  "+.1%"),
    ("Annualized Vol",    "av",  "+.1%"),
    ("Sharpe",        "sr",      ".2f"),
    ("Sortino",       "sortino", ".2f"),
    ("Calmar",        "calmar",  ".2f"),
    ("Max Drawdown",      "mdd", "+.1%"),
    ("Total Return",      "total","+.1%"),
]:
    v6v = s6[k]
    v7v = s7[k]
    diff = v7v - v6v
    if fmt == "+.1%":
        if k == "mdd":
            print(f"  {name:<24} {-v6v:>+9.1%} {-v7v:>+9.1%} {-diff:>+9.1%}")
        else:
            print(f"  {name:<24} {v6v:>+9.1%} {v7v:>+9.1%} {diff:>+9.1%}")
    else:
        print(f"  {name:<24} {v6v:>10.2f} {v7v:>10.2f} {diff:>+9.2f}")

print(f"\n  {'Years Beating SPY':<24} {s6['beat_spy']}/{s6['total_years']} yrs"
      f"   →   {s7['beat_spy']}/{s7['total_years']} yrs")

print(f"\n  {'Year':>6}  {'v6':>8}  {'v7':>8}  {'SPY':>8}  {'v6 Alpha':>9}  {'v7 Alpha':>9}")
print("  " + "─"*56)
for yr in sorted(s6["yearly"]):
    p6, sp6 = s6["yearly"][yr]
    p7, sp7 = s7["yearly"].get(yr, (0, 0))
    spy_y   = sp6
    e6 = p6 - spy_y
    e7 = p7 - spy_y
    sym6 = "✓" if e6 > 0 else "✗"
    sym7 = "✓" if e7 > 0 else "✗"
    print(f"  {yr:>6}  {p6:>+7.1%}  {p7:>+7.1%}  {spy_y:>+7.1%}  "
          f"{e6:>+6.1%}{sym6}  {e7:>+6.1%}{sym7}")

print(f"\n  v7 LETF allocation distribution ({len(rows_v7)} periods)")
letf_counts = {}
for x in rows_v7:
    letf_counts[x["letf"]] = letf_counts.get(x["letf"], 0) + 1
for tk, cnt in sorted(letf_counts.items(), key=lambda x: -x[1]):
    pct = cnt / len(rows_v7)
    bar = "█" * int(pct * 30)
    print(f"  {tk:<6}  {bar:<30}  {cnt:>3} periods ({pct:.0%})")

print(f"\n  Trailing stop triggered: {sum(1 for x in rows_v7 if x['stopped'])} / {len(rows_v7)} periods")

print()
print(f"  Verified: {'✓ Ann.Ret' if s7['ar']>=0.16 else '✗ Ann.Ret'} {s7['ar']:+.1%}  "
      f"{'✓ MDD' if s7['mdd']<=0.25 else '✗ MDD'} {-s7['mdd']:+.1%}  "
      f"{'✓ Beat SPY' if s7['total']>s7['spy_total'] else '✗ Lag SPY'} "
      f"({s7['total']:+.1%} vs {s7['spy_total']:+.1%})")
print("═"*W)

import pandas as pd
pd.DataFrame(rows_v7).to_csv(ROOT/"backtest_v7.csv", index=False)
print("  Saved: backtest_v7.csv")
