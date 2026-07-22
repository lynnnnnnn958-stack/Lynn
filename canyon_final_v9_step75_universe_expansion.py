"""
Canyon v9  Step 75 — Universe Expansion + Extended Backtest (2000-2025)
=======================================================================
Fixes two fundamental flaws in the original ML backtest:
  1. Survivorship bias  : 40 hand-picked tickers → S&P 500 (~500 stocks)
  2. Bull-market overfit: 2020-2025 only → 2000-2025 (25 years, 3 regimes)

Key outputs
-----------
  ic_by_regime_full.csv          IC breakdown: BULL / BEAR / SIDEWAYS / ALL
  extended_backtest_perf.csv     Monthly portfolio returns 2001-2025
  extended_backtest_summary.csv  Summary stats
  universe_expansion_report.md   Full report
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CACHE_PRICES   = ROOT / "sp500_price_cache.csv"
CACHE_TICKERS  = ROOT / "sp500_tickers.json"
CACHE_REGIME   = ROOT / "regime_history.csv"   # from step76 if available
TC_BPS         = 0.0010   # 10 bps transaction cost

# ── fallback universe if Wikipedia scrape fails ─────────────────────────────
FALLBACK_100 = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","UNH","JPM",
    "V","XOM","LLY","JNJ","MA","AVGO","PG","HD","COST","MRK","CVX","ABBV",
    "CRM","NFLX","BAC","KO","PEP","ADBE","WMT","TMO","ORCL","CSCO","DIS",
    "ACN","MCD","VZ","ABT","DHR","INTC","TXN","QCOM","NEE","IBM","PM","MDT",
    "GE","HON","RTX","AMGN","GILD","SBUX","AXP","BMY","SPGI","BLK","GS",
    "MS","CAT","BA","MMM","DE","UPS","FDX","SYK","ISRG","BSX","ZTS","REGN",
    "VRTX","MRNA","BIIB","ILMN","ALGN","EW","CI","HUM","CVS","WBA","MCK",
    "ABC","ANTM","CNC","MOH","LMT","NOC","GD","HII","TDG","SPG","PLD",
    "AMT","CCI","EQIX","DLR","PSA","EXR","ARE","VTR","WPC","O",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Universe sourcing
# ─────────────────────────────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents; cache 24h."""
    cache = CACHE_TICKERS
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < 86400:
            data = json.loads(cache.read_text())
            return data["tickers"]

    try:
        import ssl, urllib.request
        from io import StringIO
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as r:
            html = r.read().decode("utf-8")
        tables = pd.read_html(StringIO(html))
        df = tables[0]
        tickers = df["Symbol"].tolist()
        tickers = [t.replace(".", "-").strip() for t in tickers if isinstance(t, str)]
        cache.write_text(json.dumps({"tickers": tickers, "fetched": datetime.now().isoformat()}))
        return tickers
    except Exception as e:
        print(f"  Wikipedia scrape failed ({e}), using fallback 100 tickers")
        return FALLBACK_100


# ─────────────────────────────────────────────────────────────────────────────
# 2. Price data download
# ─────────────────────────────────────────────────────────────────────────────
def download_prices(tickers: list[str], start: str = "2000-01-01",
                    refresh: bool = False) -> pd.DataFrame:
    """Download adjusted close prices with disk caching."""
    import yfinance as yf

    if CACHE_PRICES.exists() and not refresh:
        age = time.time() - CACHE_PRICES.stat().st_mtime
        if age < 86400:
            print(f"  Loading prices from cache ({CACHE_PRICES.name}) …")
            prices = pd.read_csv(CACHE_PRICES, index_col=0, parse_dates=True)
            print(f"  Cache: {len(prices)} rows × {len(prices.columns)} tickers")
            return prices

    print(f"  Downloading prices for {len(tickers)} tickers from {start} …")
    chunk_size = 50
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    frames = []

    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx+1}/{len(chunks)}: {len(chunk)} tickers …", end=" ", flush=True)
        try:
            raw = yf.download(chunk, start=start, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                lvl1 = raw.columns.get_level_values(1).unique().tolist()
                if "Close" in lvl0:
                    close = raw["Close"]                  # (price_type, ticker) → default
                elif "Close" in lvl1:
                    close = raw.xs("Close", axis=1, level=1)  # (ticker, price_type)
                else:
                    close = None
            else:
                # Single ticker download — wrap as DataFrame with ticker as column name
                if "Close" in raw.columns:
                    close = raw[["Close"]]
                    # rename column to ticker for consistency
                    if len(chunk) == 1:
                        close.columns = chunk
                    else:
                        close = None
                else:
                    close = None
            if close is not None and not close.empty:
                frames.append(close)
                print(f"OK ({close.shape[1]} OK)")
            else:
                print("no data")
        except Exception as e:
            print(f"FAIL ({e})")
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("No price data downloaded")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    # keep tickers with at least 1000 valid days
    valid = prices.count() >= 1000
    prices = prices.loc[:, valid]
    print(f"  Total: {len(prices.columns)} tickers with sufficient history")

    prices.to_csv(CACHE_PRICES)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(prices: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Build cross-sectional feature panel."""
    rets = prices.pct_change()

    mom_1m   = prices.pct_change(21).shift(1)
    mom_3m   = prices.pct_change(63).shift(1)
    mom_6m   = prices.pct_change(126).shift(1)
    mom_12m  = prices.pct_change(252).shift(1)
    mom_1m_l = prices.pct_change(21).shift(22)
    mom_12m_skip1m = (mom_12m - mom_1m_l).shift(1)

    ma200    = prices.rolling(200).mean()
    trend_200 = (prices / ma200 - 1).shift(1)

    vol21 = rets.rolling(21).std() * np.sqrt(252)
    inv_vol = (1 / vol21.replace(0, np.nan)).shift(1)

    spy_ma200 = spy_close.rolling(200).mean()
    spy_regime = (spy_close > spy_ma200).astype(int).shift(1)

    rsi_df = prices.apply(lambda s: compute_rsi(s).shift(1))

    rows = []
    dates = prices.index[prices.index >= "2000-06-01"]
    for date in dates:
        for tk in prices.columns:
            try:
                row = {
                    "date":           date,
                    "ticker":         tk,
                    "mom_1m":         mom_1m.loc[date, tk] if tk in mom_1m.columns else np.nan,
                    "mom_3m":         mom_3m.loc[date, tk] if tk in mom_3m.columns else np.nan,
                    "mom_6m":         mom_6m.loc[date, tk] if tk in mom_6m.columns else np.nan,
                    "mom_12m_skip1m": mom_12m_skip1m.loc[date, tk] if tk in mom_12m_skip1m.columns else np.nan,
                    "trend_200":      trend_200.loc[date, tk] if tk in trend_200.columns else np.nan,
                    "rsi_14":         rsi_df.loc[date, tk] if tk in rsi_df.columns else np.nan,
                    "inv_vol":        inv_vol.loc[date, tk] if tk in inv_vol.columns else np.nan,
                    "spy_regime":     float(spy_regime.get(date, 0)),
                }
                rows.append(row)
            except Exception:
                continue

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    # cross-sectional ranks
    for feat in ["mom_12m_skip1m", "trend_200"]:
        panel[f"rank_{feat.split('_')[0]}"] = (
            panel.groupby("date")[feat].rank(pct=True)
        )
    panel = panel.rename(columns={"rank_mom_12m_skip1m": "rank_mom"})

    return panel.dropna(subset=["mom_1m", "mom_3m", "inv_vol"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Regime labelling (simple; step76 will do this properly)
# ─────────────────────────────────────────────────────────────────────────────
def label_regime(date: pd.Timestamp, spy_close: pd.Series, vix_close: pd.Series) -> str:
    """Simple rule-based regime at a given date."""
    try:
        spy_val = spy_close.loc[:date].iloc[-1]
        spy_ma  = spy_close.loc[:date].iloc[-200:].mean()
        vix_val = vix_close.loc[:date].iloc[-1] if not vix_close.empty else 20.0

        above_ma = spy_val > spy_ma
        if above_ma and vix_val < 20:
            return "BULL"
        elif not above_ma and vix_val > 25:
            return "BEAR"
        else:
            return "SIDEWAYS"
    except Exception:
        return "BULL"


def load_regime_history() -> dict[str, str]:
    """Load regime_history.csv from step76 if available."""
    if CACHE_REGIME.exists():
        # regime_history.csv has Date as the index column (capital D)
        df = pd.read_csv(CACHE_REGIME, index_col=0, parse_dates=True)
        df.index.name = "date"
        df = df.reset_index()
        return dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["regime"]))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward(panel: pd.DataFrame, prices: pd.DataFrame,
                 spy_close: pd.Series, vix_close: pd.Series) -> tuple:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    features = ["mom_1m","mom_3m","mom_6m","mom_12m_skip1m",
                "trend_200","rsi_14","inv_vol","rank_mom","spy_regime"]

    # monthly rebalance dates
    all_dates = panel["date"].sort_values().unique()
    rebalance_dates = pd.date_range("2001-01-31", all_dates[-1], freq="BME")
    rebalance_dates = [d for d in rebalance_dates if d in set(all_dates)]

    regime_map = load_regime_history()

    records_ic, records_perf = [], []
    train_days = 252

    for i, rebdate in enumerate(rebalance_dates):
        if i % 50 == 0:
            print(f"  Rebalance {i+1}/{len(rebalance_dates)}  {rebdate.date()} …")

        date_str = rebdate.strftime("%Y-%m-%d")
        regime = regime_map.get(date_str) or label_regime(rebdate, spy_close, vix_close)

        # training window
        past_dates = all_dates[all_dates < rebdate]
        train_dates = past_dates[-train_days:] if len(past_dates) >= train_days else past_dates
        if len(train_dates) < 60:
            continue

        train = panel[panel["date"].isin(train_dates)].copy()

        # compute target: 21d forward return
        future_prices = prices.shift(-21)
        def fwd_ret(row):
            try:
                ep = prices.loc[row["date"], row["ticker"]]
                fp = future_prices.loc[row["date"], row["ticker"]]
                return fp / ep - 1
            except Exception:
                return np.nan
        train["fwd_ret"] = train.apply(fwd_ret, axis=1)
        train = train.dropna(subset=["fwd_ret"] + features)
        if len(train) < 50:
            continue

        X_tr = train[features].values
        y_tr = train["fwd_ret"].values

        # fit + predict
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_tr)
        mdl = Ridge(alpha=1.0)
        mdl.fit(X_sc, y_tr)

        test = panel[panel["date"] == rebdate].dropna(subset=features)
        if test.empty:
            continue
        X_te = scaler.transform(test[features].values)
        test = test.copy()
        test["score"] = mdl.predict(X_te)

        # compute IC
        test["fwd_ret"] = test.apply(fwd_ret, axis=1)
        valid = test.dropna(subset=["fwd_ret", "score"])
        if len(valid) < 10:
            continue
        ic, _ = spearmanr(valid["score"], valid["fwd_ret"])

        records_ic.append({
            "date": rebdate, "regime": regime, "ic": ic,
            "n_tickers": len(valid),
        })

        # portfolio return: equal-weight top 20
        top = test.nlargest(20, "score")["ticker"].tolist()
        if not top:
            continue
        next_date = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else None
        if next_date is None:
            continue
        try:
            ret_series = prices.loc[rebdate:next_date, top].pct_change().iloc[1:].mean(axis=1)
            port_ret = (1 + ret_series).prod() - 1 - TC_BPS * 2
        except Exception:
            port_ret = np.nan
        records_perf.append({"date": rebdate, "regime": regime, "portfolio_ret": port_ret})

    ic_df   = pd.DataFrame(records_ic)
    perf_df = pd.DataFrame(records_perf)
    return ic_df, perf_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary reporting
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr, ttest_1samp
    rows = []
    for regime in ["BULL", "BEAR", "SIDEWAYS", "ALL"]:
        subset = ic_df if regime == "ALL" else ic_df[ic_df["regime"] == regime]
        if subset.empty:
            continue
        ics = subset["ic"].dropna().values
        if len(ics) < 3:
            continue
        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics))
        t_stat  = float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics))) if np.std(ics) > 0 else 0)
        pos_pct = float((ics > 0).mean() * 100)
        rows.append({
            "regime":       regime,
            "n_months":     len(ics),
            "mean_ic":      round(mean_ic, 4),
            "std_ic":       round(std_ic, 4),
            "t_stat":       round(t_stat, 2),
            "positive_pct": round(pos_pct, 1),
            "assessment":   "STRONG" if abs(t_stat) > 2 and mean_ic > 0.03 else
                            "WEAK"   if mean_ic > 0 else "NEGATIVE",
        })
    return pd.DataFrame(rows)


def write_report(ic_summary: pd.DataFrame, perf_df: pd.DataFrame,
                 n_tickers: int, ts: str) -> None:
    lines = [
        "# Canyon v9 Step 75 — Universe Expansion Report",
        f"Generated: {ts}",
        "",
        f"## Universe",
        f"- Tickers with sufficient history: **{n_tickers}**",
        f"- Backtest range: 2001-01 to {perf_df['date'].max().strftime('%Y-%m') if not perf_df.empty else 'N/A'}",
        "",
        "## IC by Market Regime",
        "",
        "| Regime | N Months | Mean IC | t-stat | Positive % | Assessment |",
        "|--------|----------|---------|--------|------------|------------|",
    ]
    for _, r in ic_summary.iterrows():
        lines.append(
            f"| {r['regime']:8s} | {r['n_months']:8d} | "
            f"{r['mean_ic']:+.4f} | {r['t_stat']:+.2f} | "
            f"{r['positive_pct']:.1f}% | {r['assessment']} |"
        )

    if not perf_df.empty:
        bull_r  = perf_df[perf_df["regime"]=="BULL"]["portfolio_ret"].mean()
        bear_r  = perf_df[perf_df["regime"]=="BEAR"]["portfolio_ret"].mean()
        all_r   = perf_df["portfolio_ret"].mean()
        lines += [
            "",
            "## Average Monthly Portfolio Return",
            f"- BULL months: {bull_r*100:+.2f}%" if not np.isnan(bull_r) else "- BULL months: N/A",
            f"- BEAR months: {bear_r*100:+.2f}%" if not np.isnan(bear_r) else "- BEAR months: N/A",
            f"- ALL months:  {all_r*100:+.2f}%",
            "",
            "## Key Finding",
            "The BEAR IC is the most important number.",
            "- If BEAR IC > 0: momentum partly works even in downturns",
            "- If BEAR IC < 0: the model needs regime-conditional logic (Step 77)",
        ]

    (ROOT / "universe_expansion_report.md").write_text("\n".join(lines))
    print("  [universe_expansion_report.md] OK")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 75 — Universe Expansion")
    parser.add_argument("--refresh", action="store_true", help="Re-download price data")
    parser.add_argument("--fast",    action="store_true", help="Use 100-ticker fallback only")
    args = parser.parse_args()

    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*62}")
    print("Canyon v9  Step 75 — Universe Expansion + Extended Backtest")
    print(f"{'='*62}\n")

    # ── [1] Universe ─────────────────────────────────────────────────────────
    print("[1/6] Fetching S&P 500 tickers …")
    if args.fast:
        tickers = FALLBACK_100
        print(f"  Fast mode: using {len(tickers)} fallback tickers")
    else:
        tickers = get_sp500_tickers()
        print(f"  {len(tickers)} tickers fetched")

    # ── [2] Prices ───────────────────────────────────────────────────────────
    print("\n[2/6] Price data …")
    prices = download_prices(tickers, refresh=args.refresh)

    # SPY and VIX
    import yfinance as yf
    print("  Fetching SPY and ^VIX …")
    spy_raw = yf.download("SPY", start="2000-01-01", auto_adjust=True)
    spy_close = spy_raw["Close"].squeeze() if "Close" in spy_raw.columns else spy_raw.iloc[:, 0]
    try:
        vix_raw   = yf.download("^VIX", start="2000-01-01", auto_adjust=True)
        vix_close = vix_raw["Close"].squeeze() if "Close" in vix_raw.columns else vix_raw.iloc[:, 0]
    except Exception:
        vix_close = pd.Series(dtype=float)
    print(f"  SPY: {len(spy_close)} days  |  VIX: {len(vix_close)} days")

    # History coverage
    first_dates = prices.apply(lambda c: c.first_valid_index())
    n_2000 = (first_dates <= "2001-01-01").sum()
    n_2005 = (first_dates <= "2005-01-01").sum()
    n_2010 = (first_dates <= "2010-01-01").sum()
    print(f"  History coverage: back to 2000={n_2000}, 2005={n_2005}, 2010={n_2010} tickers")

    # ── [3] Features ─────────────────────────────────────────────────────────
    print(f"\n[3/6] Building features for {len(prices.columns)} tickers …")
    panel = build_features(prices, spy_close)
    print(f"  Panel: {len(panel):,} rows  ({panel['date'].nunique()} dates, {panel['ticker'].nunique()} tickers)")

    # ── [4] Walk-forward ─────────────────────────────────────────────────────
    print("\n[4/6] Running walk-forward (this takes a few minutes) …")
    ic_df, perf_df = walk_forward(panel, prices, spy_close, vix_close)
    print(f"  Done: {len(ic_df)} rebalances with valid IC")

    # ── [5] IC by regime ─────────────────────────────────────────────────────
    print("\n[5/6] IC by regime …")
    ic_summary = compute_ic_summary(ic_df)

    print(f"\n  {'Regime':10s}  {'N':>6s}  {'Mean IC':>8s}  {'t-stat':>7s}  {'Pos%':>6s}  Assessment")
    print(f"  {'-'*60}")
    for _, r in ic_summary.iterrows():
        print(f"  {r['regime']:10s}  {r['n_months']:>6d}  {r['mean_ic']:>+8.4f}  "
              f"{r['t_stat']:>+7.2f}  {r['positive_pct']:>5.1f}%  {r['assessment']}")

    # ── [6] Outputs ───────────────────────────────────────────────────────────
    print("\n[6/6] Writing outputs …")
    ic_df.to_csv(ROOT / "ic_by_regime_raw.csv", index=False)
    ic_summary.to_csv(ROOT / "ic_by_regime_full.csv", index=False)
    print("  [ic_by_regime_full.csv] OK")

    if not perf_df.empty:
        perf_df.to_csv(ROOT / "extended_backtest_perf.csv", index=False)
        print("  [extended_backtest_perf.csv] OK")

    # save last-date features for step77
    last_date = panel["date"].max()
    panel[panel["date"] == last_date].to_csv(
        ROOT / "universe_expanded_features.csv", index=False
    )
    print("  [universe_expanded_features.csv] OK")

    # summary
    summary_rows = ic_summary.copy()
    summary_rows["generated"] = ts
    summary_rows.to_csv(ROOT / "extended_backtest_summary.csv", index=False)
    print("  [extended_backtest_summary.csv] OK")

    write_report(ic_summary, perf_df, len(prices.columns), ts)

    print(f"\n{'='*62}")
    print(f"Step 75 complete — {len(prices.columns)} tickers, {len(ic_df)} rebalances")
    print(f"Runtime: {time.time()-t0:.0f}s")
    print(f"{'='*62}\n")

    # Print the key answer
    bear_row = ic_summary[ic_summary["regime"] == "BEAR"]
    if not bear_row.empty:
        bear_ic = bear_row.iloc[0]["mean_ic"]
        print(f">>> KEY RESULT: BEAR market IC = {bear_ic:+.4f}")
        if bear_ic < -0.02:
            print(">>> CONFIRMED: Model fails in bear markets. Step 77 regime-conditional is needed.")
        elif bear_ic > 0.02:
            print(">>> SURPRISING: Model still works in bear markets (may be data-dependent).")
        else:
            print(">>> NEUTRAL: Model has near-zero edge in bear markets.")


if __name__ == "__main__":
    main()
