"""
Analyst Revision Momentum Factor
=================================
Downloads yfinance upgrades_downgrades for all S&P500 tickers.
Builds revision score: net upgrades minus downgrades per ticker.
Runs IC analysis vs forward returns.

Factor: sum of daily revision scores over past 60 days
  +1.0 = upgrade action or Strong Buy/Buy/Outperform grade
  -1.0 = downgrade action or Sell/Underperform grade
  +0.5 = initiation with positive grade
  0    = maintain / neutral / reiterate
"""
from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT  = Path(__file__).parent
DELAY = 0.12

POSITIVE_GRADES = {
    "strong buy", "buy", "outperform", "overweight",
    "positive", "accumulate", "market outperform", "add",
    "sector outperform", "top pick", "conviction buy"
}
NEGATIVE_GRADES = {
    "sell", "underperform", "underweight", "negative",
    "reduce", "market underperform", "avoid", "conviction sell"
}

def grade_score(to_grade, from_grade, action):
    """Score a single analyst action: +1, +0.5, 0, -0.5, -1"""
    tg = str(to_grade).lower().strip()
    fg = str(from_grade).lower().strip() if pd.notna(from_grade) else ""
    ac = str(action).lower().strip() if pd.notna(action) else ""

    if ac == "up":    return 1.0
    if ac == "down":  return -1.0

    if tg in POSITIVE_GRADES:
        score = 0.5
        if fg in NEGATIVE_GRADES: score = 1.0   # upgrade from negative
        if ac == "init": score = 0.5
        return score
    if tg in NEGATIVE_GRADES:
        score = -0.5
        if fg in POSITIVE_GRADES: score = -1.0  # downgrade from positive
        return score
    return 0.0

# ── Load tickers ─────────────────────────────────────────────────────────────
p25 = pd.read_csv(ROOT / "sp500_price_25yr.csv", index_col=0, parse_dates=True)
tickers = [c for c in p25.columns if c != "SPY"]
print(f"Downloading analyst upgrades/downgrades for {len(tickers)} tickers...")
print(f"Delay: {DELAY}s/ticker\n")

# ── Download ──────────────────────────────────────────────────────────────────
records = []
n_ok, n_fail, n_empty = 0, 0, 0

for i, tk in enumerate(tickers):
    try:
        t   = yf.Ticker(tk)
        ud  = t.upgrades_downgrades
        if ud is None or len(ud) == 0:
            n_empty += 1
        else:
            # Normalize index to datetime
            if not isinstance(ud.index, pd.DatetimeIndex):
                ud.index = pd.to_datetime(ud.index)
            ud = ud.sort_index()
            ud = ud.loc["2015-01-01":]  # only keep recent history

            for dt, row in ud.iterrows():
                to_g   = row.get("ToGrade",   "")
                from_g = row.get("FromGrade", "")
                action = row.get("Action",    "")
                firm   = row.get("Firm",      "")
                sc     = grade_score(to_g, from_g, action)
                records.append(dict(
                    date=dt.date(), ticker=tk,
                    to_grade=to_g, from_grade=from_g,
                    action=action, firm=firm,
                    rev_score=sc,
                ))
            n_ok += 1
    except Exception:
        n_fail += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(tickers)}]  ok={n_ok}  empty={n_empty}  "
              f"fail={n_fail}  records={len(records)}")
    time.sleep(DELAY)

print(f"\nCompleted: ok={n_ok}  empty={n_empty}  fail={n_fail}")
print(f"Total records: {len(records)}")

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df.to_csv(ROOT / "analyst_revisions.csv", index=False)
print(f"Saved: analyst_revisions.csv  "
      f"({len(df)} rows, {df.ticker.nunique()} tickers, "
      f"{df.date.dt.year.min()}-{df.date.dt.year.max()})")

# ── Action distribution ───────────────────────────────────────────────────────
print(f"\nAction breakdown:")
for ac, cnt in df["action"].value_counts().head(10).items():
    print(f"  {ac:<12}  {cnt:>6} ({cnt/len(df):.1%})")

print(f"\nScore distribution:")
for sc, cnt in df["rev_score"].value_counts().items():
    print(f"  {float(sc):>+5.1f}  {cnt:>6} ({cnt/len(df):.1%})")

# ── IC Analysis ───────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("IC ANALYSIS — Analyst Revision vs Forward Returns")
print(f"{'─'*60}")

prices = pd.read_csv(ROOT / "sp500_price_25yr.csv",
                     index_col=0, parse_dates=True).sort_index()
spy    = prices.pop("SPY")
HOLD_M = 21

# Monthly: for each month-end, build revision score (past 60 days),
# then measure forward 21-day return
monthly_dates = prices.resample("ME").last().index

ics = []
for mo in monthly_dates:
    if mo < pd.Timestamp("2016-01-01"): continue   # warmup
    window_start = mo - pd.DateOffset(days=60)

    # Revision score per ticker (sum of actions in 60-day window)
    recent = df[(df.date >= window_start) & (df.date <= mo)]
    if len(recent) < 10: continue
    rev_agg = recent.groupby("ticker")["rev_score"].sum()

    # Forward 21-day return
    future = prices.loc[mo:]
    if len(future) < 3: continue
    fwd_idx = future.index[min(HOLD_M, len(future) - 1)]
    fwd_r   = future.loc[fwd_idx] / prices.loc[mo] - 1

    # Align and compute IC
    common  = rev_agg.index.intersection(fwd_r.index)
    if len(common) < 10: continue
    x = rev_agg.loc[common].values.astype(float)
    y = fwd_r[common].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10: continue
    ic, _ = spearmanr(x[ok], y[ok])
    if np.isfinite(ic):
        ics.append(ic)

if ics:
    arr = np.array(ics)
    t_  = arr.mean() / arr.std() * np.sqrt(len(arr))
    print(f"  Periods : {len(arr)}")
    print(f"  Mean IC : {arr.mean():+.4f}")
    print(f"  IC Std  : {arr.std():.4f}")
    print(f"  t-stat  : {t_:+.2f}  {'*** SIGNIFICANT' if abs(t_) > 2 else '(marginal)'}")
    print(f"  Hit rate: {(arr > 0).mean():.0%}")

    # By year
    month_labels = [mo for mo in monthly_dates
                    if mo >= pd.Timestamp("2016-01-01")]
    ml = month_labels[:len(ics)]
    by_yr = {}
    for mo, ic in zip(ml, ics):
        yr = mo.year
        by_yr.setdefault(yr, []).append(ic)
    print(f"\n  Year-by-year IC:")
    for yr in sorted(by_yr):
        arr_yr = np.array(by_yr[yr])
        print(f"  {yr}: IC={arr_yr.mean():+.4f}  hit={( arr_yr>0).mean():.0%}  "
              f"n={len(arr_yr)}")
else:
    print("  Insufficient data for IC analysis")

print("\nDone.")
