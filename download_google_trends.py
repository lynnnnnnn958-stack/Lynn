"""
Google Trends Retail Attention Signal
Improvement 5: Contrarian retail sentiment
Method:
  1. Market-level: "stock market crash", "recession", "bear market" → fear index
  2. Per-stock: search interest for top 100 tickers vs anchor "S&P 500"
     - High attention: retail crowded → contrarian sell signal
     - Low attention: underfollowed → potential alpha

Note: pytrends returns relative interest (0-100).
All comparisons anchored to "S&P 500" as baseline.
"""
import time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

warnings.filterwarnings("ignore")
ROOT  = Path(__file__).parent
DELAY = 5.0   # pytrends needs longer delays to avoid rate limiting

pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 30),
                    retries=3, backoff_factor=2.0)

# ── 1. Market-level fear/greed sentiment ─────────────────────────────────────
print("Downloading market-level Google Trends...")
MARKET_TERMS = ["stock market crash", "recession", "bear market",
                "buy stocks", "sell stocks"]

try:
    pytrends.build_payload(MARKET_TERMS[:5], cat=0,
                           timeframe="2004-01-01 2026-06-22", geo="US")
    market_trends = pytrends.interest_over_time()
    if "isPartial" in market_trends.columns:
        market_trends = market_trends.drop(columns=["isPartial"])
    market_trends.index.name = "date"
    market_trends.to_csv(ROOT / "gtrends_market.csv")
    print(f"  Market trends: {market_trends.shape}")
    print(f"  Columns: {market_trends.columns.tolist()}")

    # Fear index: (crash + recession + bear) - (buy stocks)
    fear_cols   = ["stock market crash", "recession", "bear market"]
    greed_cols  = ["buy stocks"]
    avail_fear  = [c for c in fear_cols  if c in market_trends.columns]
    avail_greed = [c for c in greed_cols if c in market_trends.columns]
    if avail_fear:
        market_trends["fear_index"]  = market_trends[avail_fear].mean(axis=1)
        market_trends["greed_index"] = market_trends[avail_greed].mean(axis=1) if avail_greed else 0
        market_trends["sentiment"]   = (market_trends["greed_index"] -
                                        market_trends["fear_index"])
        print(f"  Fear index range: "
              f"{market_trends['fear_index'].min():.0f} - "
              f"{market_trends['fear_index'].max():.0f}")
        market_trends.to_csv(ROOT / "gtrends_market.csv")
    time.sleep(DELAY)
except Exception as e:
    print(f"  Market trends failed: {e}")

# ── 2. Per-stock retail attention ─────────────────────────────────────────────
print("\nDownloading per-stock Google Trends (top 100 by score)...")
p25_path = ROOT / "sp500_price_25yr.csv"
if not p25_path.exists():
    print("  sp500_price_25yr.csv not found — skipping per-stock")
else:
    p25     = pd.read_csv(p25_path, index_col=0, parse_dates=True)
    tickers = [c for c in p25.columns if c != "SPY"]
    # Focus on most recent high-volume names for better signal
    recent  = p25.tail(252)
    vol     = recent.pct_change().std()
    top100  = vol.nlargest(100).index.tolist()
    print(f"  Top 100 tickers by recent volatility: {top100[:10]}...")

    ANCHOR  = "S&P 500"   # anchor term for normalization
    results = {}
    n_ok, n_fail = 0, 0

    # Batch into groups of 4 (+ anchor = 5 terms max)
    batch_size = 4
    for i in range(0, len(top100), batch_size):
        batch = top100[i: i + batch_size]
        terms = batch + [ANCHOR]
        try:
            pytrends.build_payload(terms, cat=0,
                                   timeframe="2018-01-01 2026-06-22", geo="US")
            df = pytrends.interest_over_time()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            if ANCHOR in df.columns:
                anchor_vals = df[ANCHOR].replace(0, np.nan)
                for tk in batch:
                    if tk in df.columns:
                        # Normalize: interest relative to S&P 500 anchor
                        normalized = df[tk] / anchor_vals * 100
                        results[tk] = normalized
                        n_ok += 1
                    else:
                        n_fail += 1
            time.sleep(DELAY)
        except Exception as e:
            n_fail += len(batch)
            time.sleep(DELAY * 2)

        if (i // batch_size + 1) % 5 == 0:
            print(f"  [{min(i+batch_size, len(top100))}/{len(top100)}]  "
                  f"ok={n_ok}  fail={n_fail}")

    if results:
        stock_trends = pd.DataFrame(results)
        stock_trends.index.name = "date"
        stock_trends.to_csv(ROOT / "gtrends_stocks.csv")
        print(f"\n  Saved: gtrends_stocks.csv  {stock_trends.shape}")
        print(f"  ok={n_ok}  fail={n_fail}")

        # IC analysis: high attention → lower future return?
        from scipy.stats import spearmanr
        p25_prices = p25.drop(columns=["SPY"], errors="ignore")
        ics = []
        for mo_ts in stock_trends.resample("ME").last().index:
            if mo_ts < pd.Timestamp("2019-01-01"): continue
            win = mo_ts - pd.DateOffset(weeks=4)
            recent_tr = stock_trends.loc[win:mo_ts].mean()

            cur_idx = p25_prices.index[p25_prices.index <= mo_ts]
            fut_idx = p25_prices.index[p25_prices.index > mo_ts]
            if len(cur_idx) == 0 or len(fut_idx) < 21: continue
            fwd_r = p25_prices.loc[fut_idx[20]] / p25_prices.loc[cur_idx[-1]] - 1
            common = recent_tr.index.intersection(fwd_r.index)
            if len(common) < 5: continue
            x = recent_tr[common].values.astype(float)
            y = fwd_r[common].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5: continue
            ic, _ = spearmanr(x[ok], y[ok])
            if np.isfinite(ic): ics.append(ic)

        if ics:
            arr = np.array(ics)
            t_  = arr.mean() / arr.std() * np.sqrt(len(arr))
            print(f"\n  Google Trends attention IC (2019-2026):")
            print(f"  Mean IC: {arr.mean():+.4f}  t={t_:+.2f}  "
                  f"hit={(arr>0).mean():.0%}  "
                  f"{'SIGNIFICANT' if abs(t_)>2 else 'marginal'}")
            print(f"  Note: negative IC = contrarian (high attention → underperform)")
    else:
        print("  No per-stock data collected.")

print("\nDone.")
