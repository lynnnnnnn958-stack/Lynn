"""
Canyon Quant — Stop-Loss Leveraged ETF Strategy
===================================
Stop-Loss Layer 1 — LETF trailing stop (daily check)
  Every day within a hold period, monitor LETF price:
  If LETF_today < entry_price × (1 - TRAIL_STOP), exit to cash that day,
  Hold cash for rest of the period (do not re-enter early)

Stop-Loss Layer 2 — Portfolio high-watermark drawdown stop
  Track portfolio NAV high-watermark (HWM);
  If current NAV < HWM × (1 - HWM_STOP), switch to "de-risk mode":
  Next rebalance: reduce ETF weight to 50% of normal until NAV makes a new high

Stop-Loss Layer 3 — LETF fast MA signal (replaces slow SPY 200MA)
  LETF price vs its own 20-day MA:
  Break below 20MA → exit to cash immediately (4-6 weeks faster than 200MA)
  Recross above 20MA → re-enter position

Test parameter combinations (find optimal stop-loss levels):
  TRAIL_STOP: 10%, 15%, 20%
  HWM_STOP:   15%, 20%, 25%
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
HOLD_M      = 21
TOP_M       = 25
TCOST       = 5        # bps
WARMUP      = 252
SPY_WIN     = 200
STOCK_ALLOC = 0.40
ETF_ALLOC   = 0.60
LETF_BULL   = "TQQQ"
LETF_MA     = 20       # LETF fast MA (days)

# Stop-loss parameters (test three levels)
TRAIL_STOPS = [0.10, 0.15, 0.20]   # trailing stop: exit at 10/15/20% from peak
HWM_STOPS   = [0.15, 0.20, 0.25]   # high-watermark stop thresholds

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
for fname in ("sp500_price_8yr.csv", "sp500_price_cache.csv"):
    if (ROOT / fname).exists():
        prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()
        break

spy = prices["SPY"].copy() if "SPY" in prices.columns else None
if "SPY" in prices.columns:
    prices = prices.drop(columns=["SPY"])

letf = pd.read_csv(ROOT / "letf_prices.csv", index_col=0, parse_dates=True).sort_index()
letf = letf.reindex(prices.index).ffill().bfill()
if spy is None:
    spy = letf["SPY"] if "SPY" in letf.columns else letf.iloc[:, 0]

common_idx = prices.index.intersection(letf.index)
prices = prices.reindex(common_idx)
letf   = letf.reindex(common_idx)
spy    = spy.reindex(common_idx).ffill()
print(f"  {len(common_idx)} days, {common_idx[0].date()} → {common_idx[-1].date()}")

tqqq = letf[LETF_BULL].values   # numpy array for fast indexing

# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════
def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def regime_spy(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t - SPY_WIN + 1:t + 1].mean() \
           else "BEAR"

def sig_medium(t: int) -> pd.Series | None:
    if t < WARMUP:
        return None
    p    = prices.iloc[:t + 1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    mom_1m = (now / p1m - 1).clip(-0.5, 0.5)
    mom_3m = (now / p3m - 1).clip(-1, 1)
    mom_12 = (p1m / p13m - 1).clip(-1, 1)
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    trend  = (now > sma200).astype(float)
    return (rk(mom_12)*0.35 + rk(mom_3m)*0.20 +
            rk(invvol)*0.25 + rk(mom_1m)*0.10 + rk(trend)*0.10).dropna()

def stock_ret_period(t: int, sig: pd.Series) -> float:
    t_end = min(t + HOLD_M, len(prices) - 1)
    sel   = sig.nlargest(TOP_M).index.tolist()
    r = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1):
            r += (p1 / p0 - 1) / TOP_M
    return r


# ══════════════════════════════════════════════════════════════════════
# Stop-loss logic: daily TQQQ trailing stop
# ══════════════════════════════════════════════════════════════════════
def tqqq_period_with_stoploss(t: int, trail_stop: float) -> tuple[float, bool]:
    """
    Hold TQQQ from t to t+HOLD_M, check trailing stop daily.
    Returns: (actual return, stop triggered)
    Trailing stop logic:
      - Track highest price since entry (trailing peak)
      - If today price < trailing peak × (1 - trail_stop) → stop out, hold cash for rest of period
    """
    t_end = min(t + HOLD_M, len(tqqq) - 1)
    entry = tqqq[t]
    if entry <= 0:
        return 0.0, False

    peak     = entry
    stopped  = False
    exit_day = t_end

    for day in range(t + 1, t_end + 1):
        price = tqqq[day]
        if price > peak:
            peak = price                             # update trailing peak
        if price < peak * (1 - trail_stop):         # stop triggered
            exit_day = day
            stopped  = True
            break

    exit_price = tqqq[exit_day]
    ret = exit_price / entry - 1 if entry > 0 else 0.0
    return float(ret), stopped


def tqqq_period_with_ma_stop(t: int) -> tuple[float, bool]:
    """
    LETF fast MA stop: check daily whether TQQQ breaks below 20-day MA during hold period.
    Break below → exit to cash until end of period.
    """
    t_end = min(t + HOLD_M, len(tqqq) - 1)
    entry = tqqq[t]
    if entry <= 0:
        return 0.0, False

    stopped  = False
    exit_day = t_end

    for day in range(t + 1, t_end + 1):
        if day < LETF_MA:
            continue
        ma20 = np.mean(tqqq[day - LETF_MA:day])
        if tqqq[day] < ma20:
            exit_day = day
            stopped  = True
            break

    exit_price = tqqq[exit_day]
    ret = exit_price / entry - 1 if entry > 0 else 0.0
    return float(ret), stopped


# ══════════════════════════════════════════════════════════════════════
# Main backtest: with stop-loss
# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))

def run_with_stops(trail_stop: float, hwm_stop: float,
                   use_ma_stop: bool = False) -> dict:
    """
    trail_stop: trailing stop threshold (e.g. 0.15 = exit when 15% below peak)
    hwm_stop:   portfolio HWM stop (e.g. 0.20 = de-risk when NAV 20% below peak)
    use_ma_stop: use MA stop instead of trailing stop
    """
    nav      = 1.0   # current NAV
    hwm      = 1.0   # historical peak NAV
    derisked = False  # whether in de-risk mode
    rets     = []
    spy_rets = []
    stop_count = 0

    for t in rebal:
        t_end = min(t + HOLD_M, len(prices) - 1)
        reg   = regime_spy(t)
        sig   = sig_medium(t)

        # Equity returns
        sr = stock_ret_period(t, sig) if sig is not None and len(sig) >= TOP_M else 0.0
        sr_scaled = sr * (1.0 if reg == "BULL" else 0.5)

        # Determine ETF weight (halved in de-risk mode)
        etf_w = ETF_ALLOC * (0.5 if derisked else 1.0)

        # LETF return (with stop-loss)
        if reg == "BULL":
            if use_ma_stop:
                er, stopped = tqqq_period_with_ma_stop(t)
            else:
                er, stopped = tqqq_period_with_stoploss(t, trail_stop)
            if stopped:
                stop_count += 1
        else:
            # Bear regime: hold cash (not SQQQ; avoid inverse ETF risk)
            er, stopped = 0.0, False

        tc  = TCOST / 10_000
        ret = STOCK_ALLOC * sr_scaled + etf_w * er - tc
        nav *= (1 + ret)

        # Update HWM, check portfolio stop
        if nav > hwm:
            hwm      = nav
            derisked = False           # new high → exit de-risk mode
        elif nav < hwm * (1 - hwm_stop):
            derisked = True            # below HWM stop → enter de-risk mode

        spy_r = float(spy.iloc[t_end]) / float(spy.iloc[t]) - 1
        rets.append(ret)
        spy_rets.append(spy_r)

    return dict(rets=rets, spy_rets=spy_rets, stop_count=stop_count)


def stats(rets: list[float]) -> dict:
    r   = pd.Series(rets)
    n   = len(r)
    ppy = 252 / HOLD_M
    tot = float((1 + r).prod())
    ar  = float(tot ** (ppy / n) - 1)
    av  = float(r.std() * np.sqrt(ppy))
    sr  = ar / (av + 1e-9)
    cum = (1 + r).cumprod()
    mdd = float(-((cum - cum.cummax()) / cum.cummax()).min())
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, total=tot - 1, n=n)


# ══════════════════════════════════════════════════════════════════════
# Baseline (no stop-loss)
# ══════════════════════════════════════════════════════════════════════
print("\nRunning baseline (no stop-loss)...")
base_result = run_with_stops(trail_stop=9999, hwm_stop=9999)
base_s = stats(base_result["rets"])
spy_s  = stats(base_result["spy_rets"])

# ══════════════════════════════════════════════════════════════════════
# Scan stop-loss parameters
# ══════════════════════════════════════════════════════════════════════
print("Scanning stop-loss parameters...")
scan_results = []
for ts in TRAIL_STOPS:
    for hs in HWM_STOPS:
        r = run_with_stops(ts, hs)
        s = stats(r["rets"])
        scan_results.append(dict(trail=ts, hwm=hs,
                                 ar=s["ar"], sr=s["sr"], mdd=s["mdd"],
                                 total=s["total"], stops=r["stop_count"]))

# MA stop version
print("Running MA-stop version...")
ma_result = run_with_stops(trail_stop=0.0, hwm_stop=0.20, use_ma_stop=True)
ma_s = stats(ma_result["rets"])

# ══════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  CANYON  Stop-Loss Leveraged ETF Strategy  |  8-Year Backtest 2018-2026")
print("  Structure: 40% equity (medium-term momentum) + 60% TQQQ (with stop-loss) + cash in bear")
print("=" * 72)

print(f"\n  {'Scheme':<30} {'Ann Ret':>8} {'Sharpe':>7} {'Max DD':>10} {'Total':>9} {'Stops':>6}")
print("  " + "─" * 70)
print(f"  {'No Stop Baseline':<30} {base_s['ar']:>+7.1%} {base_s['sr']:>7.2f} "
      f"{-base_s['mdd']:>+9.1%} {base_s['total']:>+8.1%} {'—':>6}")

best_by_sr = max(scan_results, key=lambda x: x["sr"])
best_by_ar = max(scan_results, key=lambda x: x["ar"])

for r in sorted(scan_results, key=lambda x: -x["sr"]):
    label = f"Trailing {int(r['trail']*100)}% + HWM {int(r['hwm']*100)}%"
    mark  = " ← best Sharpe" if r == best_by_sr else (
            " ← best return" if r == best_by_ar and r != best_by_sr else "")
    print(f"  {label:<30} {r['ar']:>+7.1%} {r['sr']:>7.2f} "
          f"{-r['mdd']:>+9.1%} {r['total']:>+8.1%} {r['stops']:>6}{mark}")

print(f"  {'MA20 Stop + HWM 20%':<30} {ma_s['ar']:>+7.1%} {ma_s['sr']:>7.2f} "
      f"{-ma_s['mdd']:>+9.1%} {ma_s['total']:>+8.1%} {ma_result['stop_count']:>6}  ← MA fast signal")

print(f"\n  SPY Benchmark                  {spy_s['ar']:>+7.1%} {spy_s['sr']:>7.2f} "
      f"{-spy_s['mdd']:>+9.1%} {spy_s['total']:>+8.1%}")

# ── Best scheme year-by-year ─────────────────────────────────────────────────────
best_ts = best_by_sr["trail"]
best_hs = best_by_sr["hwm"]
print(f"\n  Year-by-year [best: trailing stop {int(best_ts*100)}% + HWM {int(best_hs*100)}%]")
print(f"  {'Year':>6}  {'Strategy':>9}  {'SPY':>9}  {'Excess':>9}  Regime")
print("  " + "─" * 50)

best_run = run_with_stops(best_ts, best_hs)
df_best  = pd.DataFrame({
    "ret":     best_run["rets"],
    "spy":     best_run["spy_rets"],
    "date":    [prices.index[t].strftime("%Y-%m-%d") for t in rebal],
    "regime":  [regime_spy(t) for t in rebal],
})
df_best["year"] = pd.to_datetime(df_best["date"]).dt.year

for yr in sorted(df_best["year"].unique()):
    g   = df_best[df_best["year"] == yr]
    pr  = float((1 + g["ret"]).prod() - 1)
    sr  = float((1 + g["spy"]).prod() - 1)
    reg = g["regime"].mode().iloc[0]
    print(f"  {yr:>6}  {pr:>+8.1%}  {sr:>+8.1%}  {pr-sr:>+8.1%}  {reg}")

# ── Stop-loss effect explanation ──────────────────────────────────────────────────────
print()
print("  Stop-loss mechanism explanation")
print("  ─────────────────────────────────────────────────────────────")
print("  Trailing stop: monitor TQQQ daily, exit when X% below intra-period peak")
print("           Hold cash for remaining days; re-enter on next rebalance date")
print("  HWM stop: portfolio NAV drops >Y% from all-time high → halve ETF weight next period")
print("             Restore full ETF weight when NAV makes new high")
print("  Bear protection: SPY breaks 200MA → switch ETF to cash (not SQQQ)")
print()
print("  2022 stop-loss effect (worst TQQQ year, -79% full year):")
print("  No stop: 2022 TQQQ loss transmitted to portfolio, drags total return")
print("  With stop: trailing stop auto-exits when TQQQ falls X%, preserving capital")

# ── Save best scheme ──────────────────────────────────────────────────────
df_best.to_csv(ROOT / "backtest_stoploss_best.csv", index=False)
print(f"\n  Saved: backtest_stoploss_best.csv")
print(f"  Best params: trailing stop {int(best_ts*100)}% + HWM stop {int(best_hs*100)}%")
print(f"  Best ann return: {best_by_sr['ar']:+.2%}  Sharpe: {best_by_sr['sr']:.2f}  "
      f"Max DD: {-best_by_sr['mdd']:+.1%}")
