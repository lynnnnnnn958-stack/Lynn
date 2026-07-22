"""
Factor IC Tracker — Institutional Signal Validation
====================================================
Measures Information Coefficient (IC) for each factor:
  IC = Spearman correlation between factor score and next-period return
  IC > 0.03 over 100+ periods = statistically significant signal
  IR = IC.mean() / IC.std() = Information Ratio (annualized → multiply by √12 for monthly)

Institutions use rolling IC to:
  1. Confirm a signal is working (not just in-sample)
  2. Detect signal decay (when alpha gets arb'd away)
  3. IC-weight factor combination
"""

import warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(".")
HOLD_M = 21

# ── Load price data ───────────────────────────────────────────────────────
prices = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy    = prices["SPY"].copy()
prices = prices.drop(columns=["SPY"])

WARMUP = 252
rebal  = list(range(WARMUP, len(prices) - HOLD_M, HOLD_M))


def rk(s): return s.rank(pct=True, na_option="bottom")


def compute_period_signals(t: int) -> pd.DataFrame:
    """Compute all factor scores for period t. Returns DataFrame(ticker × factor)."""
    p    = prices.iloc[:t + 1]
    now  = p.iloc[-1]
    p1m  = p.iloc[-22]
    p3m  = p.iloc[-64]  if t >= 63  else p.iloc[0]
    p13m = p.iloc[-274] if t >= 273 else (p.iloc[-252] if t >= 252 else p.iloc[0])
    sma200 = p.iloc[-200:].mean() if t >= 199 else p.mean()

    valid = now.dropna().index

    out = pd.DataFrame(index=valid)
    out["mom_12_1"]  = rk((p1m  / p13m - 1).clip(-1, 1))[valid]   # 12-1M price momentum
    out["mom_3m"]    = rk((now  / p3m  - 1).clip(-1, 1))[valid]   # 3M price momentum
    out["mom_1m"]    = rk((now  / p1m  - 1).clip(-0.5,0.5))[valid] # 1M reversal

    vol20 = p.pct_change().iloc[-21:].std()[valid]
    out["inv_vol"]   = rk(1.0 / vol20.clip(lower=0.001))           # vol-inverse (low vol = high)
    out["above_200"] = rk((now[valid] > sma200[valid]).astype(float))

    # RSI quality (low RSI = oversold = higher future return)
    gains = p.pct_change().iloc[-15:].clip(lower=0).iloc[-14:].mean()[valid]
    loss  = (-p.pct_change().iloc[-15:].clip(upper=0)).iloc[-14:].mean()[valid]
    rsi   = 100 - 100 / (1 + gains / (loss + 1e-9))
    out["low_rsi"]   = rk(-rsi)  # inverted: low RSI = high score

    return out


def compute_period_returns(t: int) -> pd.Series:
    """Compute actual forward returns for period t → t+HOLD_M."""
    t1  = min(t + HOLD_M, len(prices) - 1)
    r   = (prices.iloc[t1] / prices.iloc[t] - 1).dropna()
    return r


# ── Compute rolling IC for each factor ────────────────────────────────────

print("Computing factor ICs over all rebalancing periods...")
print(f"  {len(rebal)} periods × {prices.shape[1]} stocks\n")

ic_records = []

for i, t in enumerate(rebal):
    signals = compute_period_signals(t)
    fwd_ret = compute_period_returns(t)
    common  = signals.index.intersection(fwd_ret.index)
    if len(common) < 50:
        continue

    date = prices.index[t]
    row  = {"date": date, "n": len(common)}

    for factor in signals.columns:
        f = signals[factor].loc[common]
        r = fwd_ret.loc[common]
        if f.std() < 1e-6:
            continue
        ic_val, p_val = spearmanr(f, r)
        row[f"ic_{factor}"]  = float(ic_val)
        row[f"p_{factor}"]   = float(p_val)

    ic_records.append(row)

ic_df = pd.DataFrame(ic_records).set_index("date")
ic_df.to_csv("factor_ic_history.csv")

# ── Summary statistics ─────────────────────────────────────────────────────

ic_cols = [c for c in ic_df.columns if c.startswith("ic_")]
factor_names = {
    "ic_mom_12_1":  "Price Momentum 12-1M",
    "ic_mom_3m":    "Price Momentum 3M",
    "ic_mom_1m":    "Short-term Reversal 1M",
    "ic_inv_vol":   "Low Volatility",
    "ic_above_200": "Above 200MA",
    "ic_low_rsi":   "Low RSI (oversold)",
}

print("=" * 65)
print("  Factor IC Analysis — Canyon Quant Signal Validation")
print("=" * 65)
print(f"  {'Factor':<28} {'Mean IC':>8} {'Std IC':>8} {'IR':>8} {'t-stat':>8} {'Hit%':>7}")
print(f"  {'─'*65}")

for col in ic_cols:
    if col not in ic_df.columns:
        continue
    s = ic_df[col].dropna()
    if len(s) < 10:
        continue
    mean_ic = s.mean()
    std_ic  = s.std()
    ir      = mean_ic / (std_ic + 1e-9)
    t_stat  = mean_ic / (std_ic / np.sqrt(len(s)) + 1e-9)
    hit_pct = (s > 0).mean()
    name    = factor_names.get(col, col.replace("ic_", ""))
    sig     = "**" if abs(t_stat) > 2.0 else ("*" if abs(t_stat) > 1.5 else "")
    print(f"  {name:<28} {mean_ic:>+8.3f} {std_ic:>8.3f} {ir:>+8.3f} {t_stat:>+7.2f} {hit_pct:>6.0%} {sig}")

print(f"\n  ** = statistically significant (|t| > 2.0)")
print(f"  *  = marginally significant   (|t| > 1.5)")

# Rolling IC plot data
print(f"\n  Rolling 12-period IC (most recent 3 years):")
print(f"  {'Date':<12}", end="")
for col in ic_cols[:4]:
    name = col.replace("ic_","")[:8]
    print(f"  {name:>9}", end="")
print()

roll = ic_df[ic_cols].rolling(12).mean().dropna()
for dt, row in roll.iloc[-12:].iterrows():
    print(f"  {str(dt)[:10]:<12}", end="")
    for col in ic_cols[:4]:
        v = row.get(col, np.nan)
        print(f"  {v:>+9.3f}", end="")
    print()

# IC correlation between factors (want LOW correlation for diversification)
print(f"\n  Factor correlation matrix (want low correlations):")
corr = ic_df[ic_cols].corr()
for r_col in ic_cols:
    rn = r_col.replace("ic_", "")[:8]
    print(f"  {rn:<10}", end="")
    for c_col in ic_cols:
        v = corr.loc[r_col, c_col]
        print(f"  {v:>+6.2f}", end="")
    print()

print(f"\n  Saved: factor_ic_history.csv ({len(ic_df)} periods)")
print()

# EDGAR fundamentals IC if available
edgar_path = ROOT / "edgar_fundamentals.csv"
if edgar_path.exists():
    print("  Testing EDGAR fundamental factors...")
    edgar = pd.read_csv(edgar_path, parse_dates=["filed_date", "fiscal_year_end"])

    fund_ic_records = []
    for t in rebal[50:]:  # skip early warmup
        as_of = prices.index[t]
        valid = edgar[edgar["filed_date"] < as_of]
        if valid.empty:
            continue
        latest = valid.sort_values("filed_date").drop_duplicates("ticker", keep="last")

        fwd_ret = compute_period_returns(t)
        common  = set(latest["ticker"]) & set(fwd_ret.index)
        if len(common) < 30:
            continue

        lt = latest.set_index("ticker").loc[list(common)]
        fr = fwd_ret.loc[list(common)]

        row2 = {"date": as_of, "n": len(common)}
        for col in ["accrual_ratio", "gross_margin", "roa"]:
            if col not in lt.columns:
                continue
            f  = lt[col].dropna()
            if col == "accrual_ratio":
                f = -f  # invert: low accruals = better
            fr2 = fr.reindex(f.index).dropna()
            f   = f.reindex(fr2.index)
            if len(f) < 20:
                continue
            ic_val, p_val = spearmanr(f, fr2)
            row2[f"ic_{col}"] = float(ic_val)
        fund_ic_records.append(row2)

    if fund_ic_records:
        fund_ic = pd.DataFrame(fund_ic_records).set_index("date")
        print(f"\n  {'Factor':<28} {'Mean IC':>8} {'t-stat':>8} {'Hit%':>7}")
        print(f"  {'─'*52}")
        for col in [c for c in fund_ic.columns if c.startswith("ic_")]:
            s = fund_ic[col].dropna()
            if len(s) < 5:
                continue
            mean_ic = s.mean()
            t_stat  = mean_ic / (s.std() / np.sqrt(len(s)) + 1e-9)
            hit_pct = (s > 0).mean()
            name    = col.replace("ic_","").replace("_"," ")
            print(f"  {name:<28} {mean_ic:>+8.3f} {t_stat:>+8.2f} {hit_pct:>6.0%}")
