"""
Download SUE (Standardized Unexpected Earnings) via yfinance earnings_dates.
Output: sue_data.csv  columns: ticker, date, sue, eps_actual, eps_estimate

sue = (actual - estimate) / abs(estimate)   clipped to [-5, 5]
date = earnings announcement date (point-in-time, no lookahead)
"""
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT  = Path(__file__).parent
DELAY = 0.20   # 5 req/sec

prices_df = pd.read_csv(ROOT / "sp500_price_8yr.csv", index_col=0, parse_dates=True)
tickers   = [c for c in prices_df.columns if c != "SPY"]
print(f"Downloading quarterly earnings_dates for {len(tickers)} tickers...")

records = []
n_ok, n_fail = 0, 0

for i, tk in enumerate(tickers):
    try:
        t  = yf.Ticker(tk)
        ed = t.earnings_dates   # columns: EPS Estimate, Reported EPS, Surprise(%)
        time.sleep(DELAY)

        if ed is None or ed.empty:
            n_fail += 1
            continue

        ed = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
        if ed.empty:
            n_fail += 1
            continue

        for dt, row in ed.iterrows():
            actual = float(row["Reported EPS"])
            est    = float(row["EPS Estimate"])
            denom  = abs(est) if abs(est) > 0.01 else 0.01
            sue    = float(np.clip((actual - est) / denom, -5, 5))
            # Use the announcement date (no lookahead — it's the release date)
            ann_date = pd.Timestamp(dt).normalize()
            records.append(dict(
                ticker       = tk,
                date         = ann_date.strftime("%Y-%m-%d"),
                sue          = round(sue, 4),
                eps_actual   = round(actual, 4),
                eps_estimate = round(est, 4),
            ))
        n_ok += 1

    except Exception:
        n_fail += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(tickers)}]  ok={n_ok}  fail={n_fail}  records={len(records)}")

print(f"\nCompleted: ok={n_ok}  fail={n_fail}  total_records={len(records)}")

if len(records) >= 100:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_csv(ROOT / "sue_data.csv", index=False)
    print(f"Saved: sue_data.csv  ({len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].dt.year.min()}-{df['date'].dt.year.max()})")

    # Quick IC analysis
    print("\nSUE IC analysis (all available quarters):")
    price_monthly = prices_df.resample("ME").last()
    ic_list = []
    for mo in sorted(df["date"].dt.to_period("M").unique()):
        mo_start = mo.to_timestamp()
        mo_end   = (mo + 1).to_timestamp()
        batch    = df[(df["date"] >= mo_start) & (df["date"] < mo_end)]
        batch    = batch.sort_values("date").groupby("ticker").last()
        if len(batch) < 10: continue

        # Forward 1-month return
        idx      = price_monthly.index
        cur_idx  = idx[idx >= mo_start]
        nxt_idx  = idx[idx >= mo_end]
        if len(cur_idx) == 0 or len(nxt_idx) == 0: continue
        fwd_r    = price_monthly.loc[nxt_idx[0]] / price_monthly.loc[cur_idx[0]] - 1
        common   = batch.index.intersection(fwd_r.index)
        if len(common) < 5: continue
        from scipy.stats import spearmanr
        ic, _ = spearmanr(batch.loc[common, "sue"].values, fwd_r[common].values)
        if np.isfinite(ic):
            ic_list.append(ic)

    if ic_list:
        arr = np.array(ic_list)
        t_stat = arr.mean() / arr.std() * np.sqrt(len(arr))
        print(f"  Periods:  {len(arr)}")
        print(f"  Mean IC:  {arr.mean():+.4f}")
        print(f"  IC Std:   {arr.std():.4f}")
        print(f"  t-stat:   {t_stat:+.2f}  ({'SIGNIFICANT' if abs(t_stat)>2 else 'NOT SIGNIFICANT'})")
        print(f"  Hit rate: {(arr>0).mean():.0%}")
else:
    print("Insufficient records — not saving")

print("\nDone.")
