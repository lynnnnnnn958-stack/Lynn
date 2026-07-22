"""Lightweight real-time price refresh — runs every 5 minutes during market hours.
Updates: price_refresh_desk.csv, heatmap_data.json, paper position P&L.
"""
import pathlib, time, json
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent
t0 = time.time()

alpha  = pd.read_csv(ROOT / "alpha_scores.csv")
tickers = alpha["ticker"].dropna().tolist()
print(f"[price_refresh_rt] {len(tickers)} tickers …")

# ── Batch download: last 2 trading days ───────────────────────────────────────
BATCH = 250
parts = []
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    try:
        raw = yf.download(batch, period="2d", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            continue
        closes = (raw["Close"]
                  if isinstance(raw.columns, pd.MultiIndex)
                  else raw[["Close"]].rename(columns={"Close": batch[0]}))
        parts.append(closes)
    except Exception as e:
        print(f"  batch {i//BATCH+1} error: {e}")
    time.sleep(0.2)

if not parts:
    print("No data received — aborting.")
    raise SystemExit(1)

prices = pd.concat(parts, axis=1).sort_index()
if len(prices) < 2:
    print("Need at least 2 rows of price data.")
    raise SystemExit(1)

today_close = prices.iloc[-1]
prev_close  = prices.iloc[-2]
today_date  = str(prices.index[-1].date())

# ── Write price_refresh_desk.csv ──────────────────────────────────────────────
rows = []
for ticker in tickers:
    lp = today_close.get(ticker)
    pc = prev_close.get(ticker)
    if lp is None or (hasattr(lp, "__float__") and pd.isna(float(lp))):
        continue
    lp = float(lp)
    pc = float(pc) if (pc is not None and not pd.isna(float(pc))) else lp
    chg = (lp - pc) / pc * 100 if pc > 0 else 0.0
    rows.append({
        "ticker":        ticker,
        "last_price":    round(lp, 2),
        "prev_close":    round(pc, 2),
        "daily_chg_pct": round(chg, 3),
        "days_stale":    0,
        "updated_date":  today_date,
    })

price_df = pd.DataFrame(rows)
price_df.to_csv(ROOT / "price_refresh_desk.csv", index=False)
print(f"  price_refresh_desk.csv  {len(price_df)} tickers  {today_date}")

# ── Update heatmap_data.json ──────────────────────────────────────────────────
mktcap = pd.read_csv(ROOT / "mktcap_snapshot.csv")
hm = (alpha[["ticker","sector","alpha_score","alpha_rank","signal","crowding_level"]]
      .merge(price_df[["ticker","last_price","daily_chg_pct","days_stale"]], on="ticker", how="left")
      .merge(mktcap[["ticker","market_cap_usd"]], on="ticker", how="left"))
hm["daily_chg_pct"]  = hm["daily_chg_pct"].fillna(0).astype(float)
hm["market_cap_usd"] = hm["market_cap_usd"].fillna(hm["market_cap_usd"].median()).astype(float)
hm["sector"]         = hm["sector"].fillna("Other").astype(str)
hm["alpha_score"]    = hm["alpha_score"].fillna(50).astype(float)
with open(ROOT / "heatmap_data.json", "w") as f:
    json.dump(hm.to_dict(orient="records"), f)
print(f"  heatmap_data.json       {len(hm)} tickers")

print(f"  Done in {time.time()-t0:.1f}s")
