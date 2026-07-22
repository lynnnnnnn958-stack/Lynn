"""Unusual options flow detector — top 50 tickers by alpha score.
Saves options_flow.json with unusual call/put activity.
Runs in ~90 seconds. Designed for the 5-min real-time refresh cycle.
"""
import json, pathlib, time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent
t0 = time.time()

alpha = pd.read_csv(ROOT / "alpha_scores.csv").sort_values("alpha_rank")
tickers = alpha["ticker"].head(50).tolist()
print(f"[options_flow] scanning {len(tickers)} tickers …")

today = datetime.now().date()
cutoff_30d = today + timedelta(days=30)
cutoff_90d = today + timedelta(days=90)

rows = []
errors = 0

for ticker in tickers:
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options  # list of expiry date strings
        if not expiries:
            continue

        # Pick nearest expiry within 90 days for short-dated flow
        near = [e for e in expiries
                if today < datetime.strptime(e, "%Y-%m-%d").date() <= cutoff_90d]
        if not near:
            continue
        expiry = near[0]

        chain = tk.option_chain(expiry)
        calls = chain.calls
        puts  = chain.puts

        for df_side, side in [(calls, "CALL"), (puts, "PUT")]:
            if df_side.empty:
                continue
            df_side = df_side.copy()
            df_side = df_side[df_side["volume"].notna() & df_side["openInterest"].notna()]
            df_side = df_side[df_side["volume"] > 0]
            df_side["vol_oi_ratio"] = df_side["volume"] / (df_side["openInterest"] + 1)
            df_side["premium_est"]  = df_side["lastPrice"] * df_side["volume"] * 100

            # Unusual = vol/OI > 1.5 AND premium > $50k AND not deep OTM
            unusual = df_side[
                (df_side["vol_oi_ratio"] > 1.5) &
                (df_side["premium_est"] > 50_000) &
                (df_side["lastPrice"] > 0.10)
            ].copy()

            for _, row in unusual.iterrows():
                rows.append({
                    "ticker":       ticker,
                    "side":         side,
                    "expiry":       expiry,
                    "strike":       float(row["strike"]),
                    "last_price":   float(row["lastPrice"]),
                    "volume":       int(row["volume"]),
                    "open_interest":int(row["openInterest"]),
                    "vol_oi_ratio": round(float(row["vol_oi_ratio"]), 2),
                    "premium_est":  round(float(row["premium_est"])),
                    "iv":           round(float(row.get("impliedVolatility", 0) or 0), 3),
                    "days_to_exp":  (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days,
                })
    except Exception:
        errors += 1
    time.sleep(0.15)

# Sort by premium (biggest bets first)
rows.sort(key=lambda r: r["premium_est"], reverse=True)

# Also compute per-ticker call/put sentiment
ticker_sentiment = {}
for r in rows:
    t = r["ticker"]
    if t not in ticker_sentiment:
        ticker_sentiment[t] = {"calls": 0, "puts": 0, "call_prem": 0, "put_prem": 0}
    ts = ticker_sentiment[t]
    if r["side"] == "CALL":
        ts["calls"]     += r["volume"]
        ts["call_prem"] += r["premium_est"]
    else:
        ts["puts"]     += r["volume"]
        ts["put_prem"] += r["premium_est"]

sentiment_list = []
for t, s in ticker_sentiment.items():
    total_prem = s["call_prem"] + s["put_prem"]
    call_pct   = s["call_prem"] / total_prem * 100 if total_prem > 0 else 50
    sentiment_list.append({
        "ticker":    t,
        "call_pct":  round(call_pct, 1),
        "put_pct":   round(100 - call_pct, 1),
        "total_prem":total_prem,
        "bias":      "BULLISH" if call_pct > 60 else ("BEARISH" if call_pct < 40 else "NEUTRAL"),
    })
sentiment_list.sort(key=lambda x: x["total_prem"], reverse=True)

out = {
    "as_of":      str(today),
    "tickers_scanned": len(tickers),
    "unusual_count":   len(rows),
    "top_flows":       rows[:40],
    "ticker_sentiment":sentiment_list[:20],
    "errors":          errors,
}
with open(ROOT / "options_flow.json", "w") as f:
    json.dump(out, f)

print(f"  {len(rows)} unusual flows found across {len(ticker_sentiment)} tickers")
print(f"  options_flow.json saved  ({time.time()-t0:.1f}s)")
