"""Generate heatmap_data.json for the S&P 500 sector heatmap dashboard tab."""
import json, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).parent

price  = pd.read_csv(ROOT / "price_refresh_desk.csv")
alpha  = pd.read_csv(ROOT / "alpha_scores.csv")
mktcap = pd.read_csv(ROOT / "mktcap_snapshot.csv")

df = (alpha[["ticker","sector","alpha_score","alpha_rank","signal","crowding_level"]]
      .merge(price[["ticker","last_price","daily_chg_pct","days_stale"]], on="ticker", how="left")
      .merge(mktcap[["ticker","market_cap_usd"]], on="ticker", how="left"))

df["daily_chg_pct"]  = df["daily_chg_pct"].fillna(0).astype(float)
df["market_cap_usd"] = df["market_cap_usd"].fillna(df["market_cap_usd"].median()).astype(float)
df["sector"]         = df["sector"].fillna("Other").astype(str)
df["alpha_score"]    = df["alpha_score"].fillna(50).astype(float)
df["last_price"]     = df["last_price"].fillna(0).astype(float)
df["days_stale"]     = df["days_stale"].fillna(0).astype(int)

records = df.to_dict(orient="records")

out_path = ROOT / "heatmap_data.json"
with open(out_path, "w") as f:
    json.dump(records, f)

by_sector = df.groupby("sector")["market_cap_usd"].sum().sort_values(ascending=False)
print(f"heatmap_data.json: {len(records)} tickers, {df['sector'].nunique()} sectors")
print(by_sector.apply(lambda v: f"${v/1e9:.0f}B").head(11).to_string())
