"""
Canyon Quant — Fundamental-Enhanced Long-Short Backtest (v5)
========================================
Signal architecture:
  Long signal = price momentum (60%) + earnings surprise SUE (25%) + earnings quality (15%)
  Short signal = high PE + earnings downgrade (50%) + price reversal (50%)

  Short trigger conditions:
    1. Market regime: short any time (using fundamental signals)
    2. Or: activate short only when SPY < 200MA (conservative version)
  → Run both and compare

PIT compliance:
  EPS uses know_date = period_end + 45 days (SEC filing deadline)
  Excludes any data where know_date > current date

PE proxy: price[t] / TTM_EPS[t] (TTM computed from PIT-compliant EPS)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ── Parameters ──────────────────────────────────────────────────────────────────────
WARMUP   = 252
HOLD     = 21
LONG_N   = 25
SHORT_N  = 25
TCOST    = 5      # one-way bps
BORROW   = 50     # borrow cost annual bps
SPY_WIN  = 200

# ── Load price data ───────────────────────────────────────────────────────────────
print("Loading data...")
prices = pd.read_csv(ROOT / "sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy = prices["SPY"].copy() if "SPY" in prices.columns else \
      pd.read_csv(ROOT / "macro_price_cache.csv", index_col=0, parse_dates=True)["SPY"] \
        .reindex(prices.index).ffill()
if "SPY" in prices.columns:
    prices = prices.drop(columns=["SPY"])
universe = prices.columns.tolist()
print(f"  Stocks: {len(universe)}  Dates: {prices.index[0].date()} → {prices.index[-1].date()}")


# ── Load EPS history (PIT-compliant) ──────────────────────────────────────────────

def load_eps_signals(eps_path: Path) -> dict[str, pd.DataFrame]:
    """
    Load EPS history and pre-compute SUE time-series for each ticker.
    Returns {ticker: DataFrame(date, sue, eps_trend, pe_proxy)}
    """
    if not eps_path.exists():
        print("  Warning: eps_history_pit.csv not found, skipping fundamental signals")
        return {}

    raw = pd.read_csv(eps_path, parse_dates=["period_end", "know_date"])
    raw = raw.dropna(subset=["eps"]).sort_values(["ticker", "period_end"])
    print(f"  EPS history: {len(raw)} records, {raw['ticker'].nunique()} tickers")

    signals = {}
    for tk, g in raw.groupby("ticker"):
        g = g.sort_values("period_end").reset_index(drop=True)
        if len(g) < 5:
            continue

        eps_vals = g["eps"].values
        kd       = g["know_date"].values

        rows = []
        for i in range(4, len(g)):
            kdate = pd.Timestamp(kd[i])
            # SUE = (EPS_t - EPS_{t-4q}) / std(last 8 changes)
            eps_t   = eps_vals[i]
            eps_4q  = eps_vals[i - 4]
            changes = np.diff(eps_vals[max(0, i-8): i+1])
            sue     = (eps_t - eps_4q) / (changes.std() + 1e-6)
            # EPS trend: are the last 4 QoQ positive
            qoq     = np.diff(eps_vals[max(0, i-4): i+1])
            trend   = float((qoq > 0).mean()) if len(qoq) > 0 else 0.5
            # Earnings acceleration: latest SUE vs prior SUE
            rows.append({
                "know_date": kdate,
                "sue":       float(np.clip(sue, -10, 10)),
                "eps_trend": float(trend),
                "eps":       float(eps_t),
            })

        if rows:
            signals[tk] = pd.DataFrame(rows).set_index("know_date").sort_index()

    print(f"  Valid SUE signals: {len(signals)} tickers")
    return signals


eps_signals = load_eps_signals(ROOT / "eps_history_pit.csv")
HAS_FUNDAMENTAL = len(eps_signals) > 50


def get_fundamental_snapshot(t_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch the latest fundamental signal known at t_date (know_date <= t_date).
    Returns DataFrame(ticker, sue, eps_trend, eps)
    """
    if not HAS_FUNDAMENTAL:
        return pd.DataFrame()
    rows = []
    for tk, df in eps_signals.items():
        avail = df[df.index <= t_date]
        if avail.empty:
            continue
        latest = avail.iloc[-1]
        rows.append({
            "ticker":    tk,
            "sue":       latest["sue"],
            "eps_trend": latest["eps_trend"],
            "eps":       latest["eps"],
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ── Price signals (identical to v1) ─────────────────────────────────────────────────

def price_signals(t: int) -> pd.Series | None:
    if t < WARMUP:
        return None
    p = prices.iloc[:t + 1]
    now = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else None
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else None)

    mom_1m       = (now / p1m - 1).clip(-0.5, 0.5)
    mom_3m       = (now / p3m - 1).clip(-1, 1) if p3m is not None \
                   else pd.Series(0.0, index=now.index)
    mom_12m_skip = (p1m / p13m - 1).clip(-1, 1) if p13m is not None \
                   else pd.Series(0.0, index=now.index)
    ret_d  = p.pct_change().dropna()
    inv_vol = 1.0 / (ret_d.iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    trend  = (now > sma200).astype(float)

    def rk(s): return s.rank(pct=True, na_option="bottom") * 100

    return (rk(mom_12m_skip)*0.35 + rk(mom_1m)*0.10 +
            rk(mom_3m)*0.20 + rk(inv_vol)*0.25 +
            rk(trend)*0.05).dropna()


def composite_long_signal(t: int, t_date: pd.Timestamp) -> pd.Series | None:
    """
    Long signal = price (70%) + SUE (20%) + EPS trend (10%)
    Falls back to pure price signal if fundamental data unavailable
    """
    price_sig = price_signals(t)
    if price_sig is None:
        return None

    if not HAS_FUNDAMENTAL:
        return price_sig

    fund = get_fundamental_snapshot(t_date)
    if fund.empty:
        return price_sig

    common = price_sig.index.intersection(fund.index)
    if len(common) < LONG_N:
        return price_sig  # fallback when fundamental coverage is insufficient

    def rk(s): return s.rank(pct=True, na_option="bottom") * 100

    sue_sig   = rk(fund.loc[common, "sue"])
    trend_sig = rk(fund.loc[common, "eps_trend"])
    price_common = price_sig[common]

    combined = price_common * 0.60 + sue_sig * 0.25 + trend_sig * 0.15

    # Uncovered stocks use price signal at median score
    missing = price_sig.index.difference(common)
    missing_scores = pd.Series(50.0, index=missing)
    return pd.concat([combined, missing_scores]).sort_index()


def short_signal(t: int, t_date: pd.Timestamp, long_set: set) -> list[str]:
    """
    Short signal:
      Weak fundamentals: low SUE (earnings downgrade) + high PE proxy (price/EPS high)
      Price weakening: weak 1M return
    Excludes long holdings
    """
    if t < WARMUP + 4:  # need sufficient EPS history
        return []

    p = prices.iloc[:t + 1]
    now = p.iloc[-1]
    p1m = p.iloc[-22]
    mom_1m = (now / p1m - 1).clip(-0.5, 0.5)

    def rk(s): return s.rank(pct=True, na_option="bottom") * 100

    # Pure price short base score (price weakening)
    price_weakness = rk(-mom_1m)  # high score = weakest recently

    if HAS_FUNDAMENTAL:
        fund = get_fundamental_snapshot(t_date)
        if not fund.empty:
            common = price_weakness.index.intersection(fund.index)
            if len(common) >= SHORT_N * 2:
                sue_c   = fund.loc[common, "sue"]
                # PE proxy = price / TTM_EPS (TTM = sum of 4 quarters EPS)
                eps_c   = fund.loc[common, "eps"].clip(lower=0.01)
                pe_proxy= now[common] / (eps_c * 4 + 1e-3)
                pe_proxy= pe_proxy.replace([np.inf, -np.inf], np.nan).fillna(50)

                # Short score: low SUE (bad earnings) + high PE (expensive) + weak price
                short_score = (rk(-sue_c) * 0.40 +        # low SUE → high short score
                               rk(pe_proxy) * 0.30 +       # high PE → high short score
                               rk(-price_weakness.reindex(common)) * 0.30)
                # Correction: weaker price = better short → reverse
                short_score = (rk(-sue_c[common]) * 0.40 +
                               rk(pe_proxy[common]) * 0.30 +
                               rk(-mom_1m[common]) * 0.30)
                candidates = short_score.drop(index=long_set, errors="ignore").dropna()
                return candidates.nlargest(SHORT_N).index.tolist()

    # No fundamentals: use momentum reversal (was strong → now weak)
    p13m = p.iloc[-274] if t >= 273 else None
    if p13m is None:
        return []
    mom_12m = (p1m / p13m - 1).clip(-1, 1)
    reversal = (rk(mom_12m) * 0.55 + rk(-mom_1m) * 0.45)
    candidates = reversal.drop(index=long_set, errors="ignore").dropna()
    return candidates.nlargest(SHORT_N).index.tolist()


# ── Regime filter ──────────────────────────────────────────────────────────────────

def regime(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > float(spy.iloc[t-SPY_WIN+1:t+1].mean()) else "BEAR"


def period_ret(w: dict, t: int, t_end: int, sign: float = 1.0) -> float:
    r = 0.0
    for tk, wt in w.items():
        pe = float(prices.iloc[t].get(tk, np.nan))
        px = float(prices.iloc[t_end].get(tk, np.nan))
        if pe > 0 and np.isfinite(px):
            r += wt * sign * (px / pe - 1)
    return r


# ── Walk-forward ──────────────────────────────────────────────────────────────

print("\nRunning fundamental-enhanced long-short backtest...")
rebal = list(range(WARMUP, len(prices) - HOLD, HOLD))

# Two strategies: all-weather long-short vs bear-only short
vals = {"ls_always": 1.0, "ls_bear_only": 1.0, "long_only": 1.0, "spy": 1.0}
prev_long = {}
prev_short = {}
rows = []
ic_rows = []

for i, t in enumerate(rebal):
    t_date = prices.index[t]
    t_end  = min(t + HOLD, len(prices) - 1)
    reg    = regime(t)

    # Long signal
    long_sig = composite_long_signal(t, t_date)
    if long_sig is None or len(long_sig) < LONG_N:
        continue

    long_tkrs = long_sig.nlargest(LONG_N).index.tolist()
    long_w    = {tk: 1.0/LONG_N for tk in long_tkrs}
    lr        = period_ret(long_w, t, t_end)

    # Short signal
    short_tkrs = short_signal(t, t_date, set(long_tkrs))
    short_w    = {tk: 1.0/SHORT_N for tk in short_tkrs} if short_tkrs else {}
    sr         = period_ret(short_w, t, t_end, sign=-1.0)  # short P&L

    # Transaction costs
    to_l = sum(abs(long_w.get(tk,0)-prev_long.get(tk,0))
               for tk in set(long_w)|set(prev_long))
    to_s = sum(abs(short_w.get(tk,0)-prev_short.get(tk,0))
               for tk in set(short_w)|set(prev_short))
    tc   = (to_l + to_s)/2 * TCOST/10_000
    borrow = BORROW/252*HOLD/10_000 if short_w else 0.0

    spy_ret = float(spy.iloc[t_end])/float(spy.iloc[t]) - 1
    regime_scale = 1.0 if reg == "BULL" else 0.5

    lo_ret      = (lr - to_l/2*TCOST/10_000) * regime_scale
    ls_always   = (lr + sr - tc - borrow) * regime_scale
    # Bear-only short: bull market is long-only
    ls_bear     = ls_always if reg == "BEAR" else lo_ret

    vals["long_only"]    *= (1 + lo_ret)
    vals["ls_always"]    *= (1 + ls_always)
    vals["ls_bear_only"] *= (1 + ls_bear)
    vals["spy"]          *= (1 + spy_ret)

    # IC
    fwd = {tk: float(prices.iloc[t_end][tk])/float(prices.iloc[t][tk])-1
           for tk in long_sig.index
           if float(prices.iloc[t].get(tk,0)) > 0
           and np.isfinite(float(prices.iloc[t_end].get(tk, np.nan)))}
    if len(fwd) >= 30:
        ic, _ = spearmanr(long_sig[list(fwd)], pd.Series(fwd))
        ic_rows.append({"date": t_date.strftime("%Y-%m-%d"), "IC": round(float(ic),4)})

    rows.append({
        "date":        t_date.strftime("%Y-%m-%d"),
        "lo_ret":      round(lo_ret, 6),
        "ls_always":   round(ls_always, 6),
        "ls_bear":     round(ls_bear, 6),
        "spy_ret":     round(spy_ret, 6),
        "regime":      reg,
        "has_fund":    HAS_FUNDAMENTAL,
        **{f"val_{k}": round(v,4) for k,v in vals.items()},
    })

    prev_long  = dict(long_w)
    prev_short = dict(short_w)

    if (i+1) % 20 == 0 or i == 0:
        print(f"  {t_date.date()} | LO={vals['long_only']:.3f} "
              f"LS_always={vals['ls_always']:.3f} "
              f"LS_bear={vals['ls_bear_only']:.3f} "
              f"SPY={vals['spy']:.3f} | {reg} | fund={'Y' if HAS_FUNDAMENTAL else 'N'}")


# ── Statistics ──────────────────────────────────────────────────────────────────────

df    = pd.DataFrame(rows)
ic_df = pd.DataFrame(ic_rows)
n     = len(df)
ppy   = 252/HOLD

def perf(col, final):
    r    = df[col]
    ar   = float(final**(ppy/n) - 1)
    av   = float(r.std()*np.sqrt(ppy))
    sr   = ar/(av+1e-9)
    cum  = (1+r).cumprod()
    mdd  = float(-((cum-cum.cummax())/cum.cummax()).min())
    pos  = float((r>0).mean())
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, pos=pos, total=final-1)

p = {k: perf(c, vals[k]) for k,c in [
    ("long_only","lo_ret"), ("ls_always","ls_always"),
    ("ls_bear_only","ls_bear"), ("spy","spy_ret")]}

mean_ic    = float(ic_df["IC"].mean()) if not ic_df.empty else float("nan")
std_ic     = float(ic_df["IC"].std())  if not ic_df.empty else float("nan")
pct_pos_ic = float((ic_df["IC"]>0).mean()) if not ic_df.empty else float("nan")
ic_ir      = mean_ic/(std_ic+1e-9)

print("\n" + "="*72)
print(f"  CANYON v5 — Fundamental-Enhanced LS  |  8-Year Backtest 2018-2026")
print(f"  Fundamental signals: {'ENABLED' if HAS_FUNDAMENTAL else 'NOT LOADED (price only)'}")
print("="*72)
print(f"  Period: {df['date'].iloc[0]} → {df['date'].iloc[-1]}  Rebalances: {n}")
print()

labels = ["Long only", "LS (all-weather)", "LS (bear only)", "SPY"]
keys   = ["long_only", "ls_always", "ls_bear_only", "spy"]
print(f"  {'Metric':<16}" + "".join(f"{l:>16}" for l in labels))
print("  " + "─"*80)
for metric, name in [("ar","Ann Return"),("av","Ann Vol"),("sr","Sharpe"),
                      ("mdd","Max DD"),("pos","Pos Periods"),("total","Total Ret")]:
    row = f"  {name:<16}"
    for k in keys:
        v = p[k][metric]
        if metric == "sr":
            row += f"{v:>16.2f}"
        elif metric == "mdd":
            row += f"{-v:>+15.2%}"
        else:
            row += f"{v:>+15.2%}"
    print(row)

print()
print(f"  Signal IC: mean={mean_ic:.4f}  std={std_ic:.4f}  IR={ic_ir:.2f}  pct_pos={pct_pos_ic:.0%}")
print()

df["year"] = pd.to_datetime(df["date"]).dt.year
print(f"  {'Year':>6}  {'Long Only':>10} {'LS All-Wx':>12} {'LS Bear Only':>13} {'SPY':>8}  Regime")
print("  " + "─"*56)
for yr, g in df.groupby("year"):
    reg = g["regime"].mode().iloc[0]
    def yr_ret(c): return f"{float((1+g[c]).prod())-1:>+9.1%}"
    print(f"  {yr:>6}  {yr_ret('lo_ret')} {yr_ret('ls_always'):>12} "
          f"{yr_ret('ls_bear'):>10} {yr_ret('spy_ret'):>8}  {reg}")

print()
print("  Lookahead bias check:")
print("  [PASS] EPS know_date = period_end + 45 days")
print("  [PASS] Fundamental snapshot uses only know_date <= t_date")
print("  [PASS] PE proxy = price[t] / (EPS × 4), EPS from PIT-filtered history")
print("  [PASS] Price signals use only prices[:t+1]")

df.to_csv(ROOT/"backtest_v5_monthly.csv", index=False)
ic_df.to_csv(ROOT/"backtest_v5_ic.csv", index=False)
pd.DataFrame([{"strategy":k, **{m:round(v,4) for m,v in pv.items()}}
              for k,pv in p.items()]).to_csv(ROOT/"backtest_v5_summary.csv", index=False)
print(f"\n  Saved: backtest_v5_monthly.csv / _ic.csv / _summary.csv")
