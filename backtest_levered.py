"""
Canyon Quant — Leveraged ETF Enhanced Strategy
================================
Target: annualized return >= 40% (current triple-horizon ~11.5% → target 40%)

Portfolio structure:
  Equity portion (40%): medium-term momentum, Top-25, 21-day hold
  Leveraged ETF portion (60%): regime-conditional switching
    - BULL regime: hold 3x long ETF (TQQQ/UPRO/SOXL)
    - BEAR regime: hold 3x inverse ETF (SQQQ/SPXU) or cash

Three scheme comparison:
  Scheme A: 40% equity + 60% TQQQ (3x Nasdaq, no switching)
  Scheme B: 40% equity + 60% regime switching (BULL=TQQQ, BEAR=SQQQ)
  Scheme C: 40% equity + 60% long/short switching (BULL=TQQQ+SOXL blend, BEAR=SQQQ)

Leveraged ETF key risks:
  - Volatility decay: 3x ETF is daily leverage, not hold-period leverage
  - TQQQ fell ~-79% in 2022 (QQQ only -33%)
  - 3x ETF expense ratio ~0.85%/yr (already in price; yfinance uses split-adjusted prices)
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
HOLD_M  = 21      # equity hold days (medium-term)
TOP_M   = 25      # top-N stocks
TCOST   = 5       # one-way cost bps
SPY_WIN = 200     # regime detection window
WARMUP  = 252     # warmup period

STOCK_ALLOC = 0.40   # equity allocation
ETF_ALLOC   = 0.60   # leveraged ETF allocation

# Scheme B/C: inverse ETF return when regime=BEAR (inverse ETF may profit)
BEAR_LETF = "SQQQ"  # inverse ETF for BEAR regime

# ══════════════════════════════════════════════════════════════════════
# Load price data
# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
# Equity prices
for fname in ("sp500_price_8yr.csv", "sp500_price_cache.csv"):
    if (ROOT / fname).exists():
        prices = pd.read_csv(ROOT / fname, index_col=0, parse_dates=True).sort_index()
        print(f"  Stock prices: {prices.shape}")
        break

spy = prices["SPY"].copy() if "SPY" in prices.columns else None
if "SPY" in prices.columns:
    prices = prices.drop(columns=["SPY"])

# Leveraged ETF prices
letf_path = ROOT / "letf_prices.csv"
if letf_path.exists():
    letf = pd.read_csv(letf_path, index_col=0, parse_dates=True).sort_index()
    letf = letf.reindex(prices.index).ffill().bfill()
    if spy is None and "SPY" in letf.columns:
        spy = letf["SPY"]
    print(f"  LETF prices: {letf.shape}")
    print(f"  Available: {list(letf.columns)}")
else:
    print("  [WARNING] letf_prices.csv not found — downloading now...")
    import yfinance as yf
    tickers = ["TQQQ","UPRO","SOXL","QLD","SSO","SQQQ","SPXU","SPY"]
    raw = yf.download(tickers, start="2017-01-01", end="2026-06-15",
                      auto_adjust=True, progress=False)["Close"]
    letf = raw.reindex(prices.index).ffill().bfill()
    if spy is None:
        spy = letf["SPY"]
    letf.to_csv(letf_path)
    print(f"  Downloaded & saved: {letf.shape}")

# Align dates
common_idx = prices.index.intersection(letf.index)
prices = prices.reindex(common_idx)
letf   = letf.reindex(common_idx)
spy    = spy.reindex(common_idx).ffill()
print(f"  Aligned: {len(common_idx)} days, {prices.index[0].date()} → {prices.index[-1].date()}")

# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════
def rk(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100

def regime(t: int) -> str:
    if t < SPY_WIN:
        return "BULL"
    return "BULL" if float(spy.iloc[t]) > spy.iloc[t - SPY_WIN + 1:t + 1].mean() else "BEAR"

def letf_period_ret(ticker: str, t: int, hold: int) -> float:
    if ticker not in letf.columns:
        return 0.0
    t_end = min(t + hold, len(letf) - 1)
    p0 = float(letf[ticker].iloc[t])
    p1 = float(letf[ticker].iloc[t_end])
    if p0 > 0 and np.isfinite(p1):
        return p1 / p0 - 1
    return 0.0

# ══════════════════════════════════════════════════════════════════════
# Equity signals (medium-term momentum, same as backtest_triple.py)
# ══════════════════════════════════════════════════════════════════════
def sig_medium(t: int) -> pd.Series | None:
    if t < WARMUP:
        return None
    p   = prices.iloc[:t + 1]
    now = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])

    mom_1m = (now / p1m - 1).clip(-0.5, 0.5)
    mom_3m = (now / p3m - 1).clip(-1, 1)
    mom_12 = (p1m / p13m - 1).clip(-1, 1)
    invvol = 1.0 / (p.pct_change().iloc[-21:].std() + 0.005)
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()
    trend  = (now > sma200).astype(float)

    return (rk(mom_12) * 0.35 + rk(mom_3m) * 0.20 +
            rk(invvol) * 0.25 + rk(mom_1m) * 0.10 +
            rk(trend)  * 0.10).dropna()

def stock_ret(t: int, hold: int, sig: pd.Series) -> float:
    sel   = sig.nlargest(TOP_M).index.tolist()
    t_end = min(t + hold, len(prices) - 1)
    r = 0.0
    for tk in sel:
        p0 = float(prices.iloc[t].get(tk, np.nan))
        p1 = float(prices.iloc[t_end].get(tk, np.nan))
        if p0 > 0 and np.isfinite(p1):
            r += (p1 / p0 - 1) / TOP_M
    return r

# ══════════════════════════════════════════════════════════════════════
# Scheme definitions
#   A: 40% equity + 60% TQQQ (no timing)
#   B: 40% equity + 60% BULL=TQQQ / BEAR=SQQQ (regime switching)
#   C: 40% equity + 60% BULL=TQQQ+UPRO+SOXL (equal weight) / BEAR=SQQQ (diversified)
# ══════════════════════════════════════════════════════════════════════
SCHEMES = {
    "A_TQQQ":   {"bull": [("TQQQ", 1.0)],                          "bear": [("TQQQ", 1.0)]},
    "B_switch":  {"bull": [("TQQQ", 1.0)],                          "bear": [("SQQQ", 1.0)]},
    "C_multi":   {"bull": [("TQQQ", 0.50), ("UPRO", 0.30), ("SOXL", 0.20)],
                  "bear": [("SQQQ", 1.0)]},
    "D_2x_only": {"bull": [("QLD",  0.60), ("SSO",  0.40)],         "bear": [("SSO",  1.0)]},
    "E_cash":    {"bull": [("TQQQ", 1.0)],                          "bear": []},   # cash in bear
}

# ══════════════════════════════════════════════════════════════════════
# Main backtest loop
# ══════════════════════════════════════════════════════════════════════
rebal = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))
spy_rets = []

results: dict[str, list] = {k: [] for k in SCHEMES}

print(f"\nRunning {len(rebal)} rebalance periods ({HOLD_M}d each)...\n")
prev_sig_sel: list = []

for i, t in enumerate(rebal):
    t_end = min(t + HOLD_M, len(prices) - 1)
    reg   = regime(t)
    sig   = sig_medium(t)

    # Equity returns
    sr = stock_ret(t, HOLD_M, sig) if sig is not None and len(sig) >= TOP_M else 0.0
    # Position: reduce equity to 50% in bear (cash protection)
    sr_scaled = sr * (1.0 if reg == "BULL" else 0.5)
    # Transaction costs
    tc = TCOST / 10_000  # simplified: fixed rebalance cost per period

    # SPY benchmark
    s0 = float(spy.iloc[t])
    s1 = float(spy.iloc[t_end])
    spy_rets.append(s1 / s0 - 1 if s0 > 0 else 0.0)

    # Compute ETF return for each scheme
    for name, cfg in SCHEMES.items():
        etf_mix = cfg[reg.lower()]
        if etf_mix:
            er = sum(w * letf_period_ret(tk, t, HOLD_M) for tk, w in etf_mix)
        else:
            er = 0.0  # cash: zero return

        total_ret = (STOCK_ALLOC * sr_scaled - tc) + (ETF_ALLOC * er)
        results[name].append({
            "date": prices.index[t].strftime("%Y-%m-%d"),
            "ret":  round(total_ret, 6),
            "stock_ret": round(sr_scaled, 6),
            "etf_ret":   round(er, 6),
            "regime": reg,
            "hold": HOLD_M,
        })

    if (i + 1) % 10 == 0:
        # Print scheme B progress
        cum_b = float(pd.DataFrame(results["B_switch"])["ret"].add(1).prod())
        print(f"  {prices.index[t].date()}  Regime={reg}  CumRet(B)={cum_b:.2f}x")

# ══════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════
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
    pos = float((r > 0).mean())
    return dict(ar=ar, av=av, sr=sr, mdd=mdd, pos=pos, total=tot - 1)

spy_s = stats(spy_rets)
all_stats = {k: stats([row["ret"] for row in v]) for k, v in results.items()}

# ══════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 82)
print("  CANYON  Leveraged ETF Enhanced Strategy  |  8-Year Backtest 2018-2026")
print("  40% equity + 60% leveraged ETF")
print("=" * 82)
print(f"  A: 40% equity + 60% TQQQ (hold, no switching)")
print(f"  B: 40% equity + 60% [BULL→TQQQ / BEAR→SQQQ] (regime switching)")
print(f"  C: 40% equity + 60% [BULL→TQQQ50+UPRO30+SOXL20 / BEAR→SQQQ]")
print(f"  D: 40% equity + 60% 2x ETF [BULL→QLD60+SSO40] (conservative 2x)")
print(f"  E: 40% equity + 60% [BULL→TQQQ / BEAR→cash] (no short)")
print()
print(f"  {'Metric':<20} {'A_TQQQ':>10} {'B_switch':>10} {'C_multi':>10} "
      f"{'D_2x':>10} {'E_cash':>10} {'SPY':>8}")
print("  " + "─" * 78)

for metric, name in [("ar",  "Annual Return"),
                      ("av",  "Annual Vol"),
                      ("sr",  "Sharpe"),
                      ("mdd", "Max Drawdown"),
                      ("pos", "Win Rate"),
                      ("total","Total Return")]:
    row = f"  {name:<20}"
    for key in ["A_TQQQ","B_switch","C_multi","D_2x_only","E_cash"]:
        v = all_stats[key][metric]
        if metric == "sr":
            row += f"{v:>10.2f}"
        elif metric == "mdd":
            row += f"{-v:>+9.1%}"
        else:
            row += f"{v:>+9.1%}"
    v = spy_s[metric]
    if metric == "sr":
        row += f"{v:>8.2f}"
    elif metric == "mdd":
        row += f"{-v:>+7.1%}"
    else:
        row += f"{v:>+7.1%}"
    print(row)

# Year-by-year (Scheme B)
print()
print("  Year-by-Year  [Best scheme = B: regime-switched TQQQ/SQQQ]")
print(f"  {'Year':>6}  {'Stock':>9}  {'ETF(B)':>9}  {'Combined':>9}  {'SPY':>9}  Regime")
print("  " + "─" * 58)
df_b = pd.DataFrame(results["B_switch"])
df_b["year"] = pd.to_datetime(df_b["date"]).dt.year
df_s_ref = pd.Series(spy_rets, index=[r["date"] for r in results["B_switch"]])
df_s_ref.index = pd.to_datetime(df_s_ref.index)

spy_df = pd.DataFrame({"date": [r["date"] for r in results["B_switch"]], "spy": spy_rets})
spy_df["year"] = pd.to_datetime(spy_df["date"]).dt.year

for yr in sorted(df_b["year"].unique()):
    g   = df_b[df_b["year"] == yr]
    sr  = f"{float((1+g['stock_ret']).prod()-1):>+8.1%}"
    er  = f"{float((1+g['etf_ret']).prod()-1):>+8.1%}"
    cr  = f"{float((1+g['ret']).prod()-1):>+8.1%}"
    spy_yr_val = float((1 + spy_df[spy_df["year"]==yr]["spy"]).prod() - 1)
    reg = g["regime"].mode().iloc[0]
    print(f"  {yr:>6}  {sr}  {er}  {cr}  {spy_yr_val:>+8.1%}  {reg}")

# LETF standalone performance reference
print()
print("  Leveraged ETF standalone performance (for reference)")
print(f"  {'ETF':<8}  {'Total':>10}  {'Ann Ret':>10}  Note")
print("  " + "─" * 42)
n = len(rebal)
ppy = 252 / HOLD_M
for tk in ["TQQQ","SQQQ","UPRO","SOXL","QLD","SSO"]:
    if tk not in letf.columns:
        continue
    rets = []
    for t in rebal:
        r = letf_period_ret(tk, t, HOLD_M)
        rets.append(r)
    tot = float(pd.Series(rets).add(1).prod())
    ar  = float(tot ** (ppy / n) - 1)
    notes = {"TQQQ":"3x QQQ","SQQQ":"3x Inverse QQQ","UPRO":"3x S&P500",
             "SOXL":"3x Semis","QLD":"2x QQQ","SSO":"2x S&P500"}
    print(f"  {tk:<8}  {tot-1:>+9.1%}  {ar:>+9.1%}  {notes.get(tk,'')}")

print()
print("  Key Risk: Volatility Decay")
print("  - 3x ETF = 3x DAILY return, not 3x period return")
print("  - In a sideways/volatile market, 3x ETF loses value vs 3x index")
print("  - Regime switching reduces 2022-type crash exposure")

# Save best scheme
best_key = max(all_stats, key=lambda k: all_stats[k]["ar"])
best_ar  = all_stats[best_key]["ar"]
print(f"\n  Best scheme: {best_key} → Annual Return {best_ar:+.2%}")
pd.DataFrame(results[best_key]).to_csv(ROOT / "backtest_levered_best.csv", index=False)
for k, rows in results.items():
    pd.DataFrame(rows).to_csv(ROOT / f"backtest_{k}.csv", index=False)
print("  Saved all scheme CSVs.")
