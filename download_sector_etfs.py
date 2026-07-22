"""
Sector ETF Prices + RSP (Equal-Weight S&P500)
Improvement 6: Sector rotation overlay
Improvement 8: RSP as equal-weight benchmark
"""
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLY":  "Consumer Discretionary",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
    "XLB":  "Materials",
    "RSP":  "Equal-Weight S&P500 (benchmark)",
    "QQQ":  "Nasdaq-100 (benchmark)",
}

tickers = list(SECTOR_ETFS.keys())
print(f"Downloading sector ETFs + RSP: {tickers}")

df = yf.download(tickers, start="2000-01-01", end="2026-06-22",
                 auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    prices = df["Close"]
else:
    prices = df

prices = prices.sort_index()
# Forward fill up to 3 days
prices = prices.ffill(limit=3)
prices.to_csv(ROOT / "sector_etf_prices.csv")
print(f"Saved: sector_etf_prices.csv  shape={prices.shape}")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Coverage:")
for tk in tickers:
    if tk in prices.columns:
        cov = prices[tk].notna().sum()
        first = prices[tk].first_valid_index()
        print(f"  {tk:<6} {SECTOR_ETFS[tk]:<35}  {cov} days from {first.date() if first else 'N/A'}")
    else:
        print(f"  {tk:<6} MISSING")

# ── Sector momentum IC analysis ───────────────────────────────────────────────
print("\nSector Momentum IC Analysis (monthly, 2003-2026):")
sp_prices = prices.drop(columns=["RSP", "QQQ"], errors="ignore")
sector_cols = [c for c in sp_prices.columns if c in SECTOR_ETFS]
HOLD_M = 21

monthly = sp_prices[sector_cols].resample("ME").last()
ics = []
for i in range(12, len(monthly) - 2):
    mo = monthly.iloc[i]
    mo_3m_ago = monthly.iloc[i - 3] if i >= 3 else monthly.iloc[0]
    mom3m = mo / mo_3m_ago - 1
    fwd   = monthly.iloc[i + 1] / mo - 1
    ok    = np.isfinite(mom3m.values) & np.isfinite(fwd.values)
    if ok.sum() < 5: continue
    ic, _ = spearmanr(mom3m.values[ok], fwd.values[ok])
    if np.isfinite(ic): ics.append(ic)

if ics:
    arr = np.array(ics)
    t_  = arr.mean() / arr.std() * np.sqrt(len(arr))
    print(f"  Sector 3M momentum IC: {arr.mean():+.4f}  t={t_:+.2f}  "
          f"hit={(arr>0).mean():.0%}  "
          f"{'SIGNIFICANT' if abs(t_)>2 else 'marginal'}")

# RSP vs SPY comparison stats
if "RSP" in prices.columns:
    p25_path = ROOT / "sp500_price_25yr.csv"
    if p25_path.exists():
        p25 = pd.read_csv(p25_path, index_col=0, parse_dates=True)["SPY"]
        aligned = prices["RSP"].reindex(p25.index).ffill()
        common  = p25.index[p25.notna() & aligned.notna()]
        if len(common) > 252:
            spy_r = p25[common].pct_change().dropna()
            rsp_r = aligned[common].pct_change().dropna()
            common2 = spy_r.index.intersection(rsp_r.index)
            ppy     = 252
            spy_ar  = float((1 + spy_r[common2]).prod() ** (ppy / len(spy_r[common2])) - 1)
            rsp_ar  = float((1 + rsp_r[common2]).prod() ** (ppy / len(rsp_r[common2])) - 1)
            print(f"\nRSP vs SPY (2003-2026):")
            print(f"  SPY (cap-weight)  AR={spy_ar:+.1%}")
            print(f"  RSP (equal-weight) AR={rsp_ar:+.1%}")
            print(f"  Our strategy vs RSP will show true alpha vs "
                  f"size/equal-weight benchmark")

print("\nDone.")
