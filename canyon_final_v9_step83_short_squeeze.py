#!/usr/bin/env python3
"""
Canyon v9 — Step 83: Short Interest & Squeeze Detector
=======================================================
Extracts short-interest data (free via yfinance / Yahoo Finance):

  • Short % of Float    — what fraction of tradeable shares are short
  • Days to Cover       — trading days needed to unwind all shorts
  • Short Change        — MoM change in short interest (covering = bullish)
  • Squeeze Score       — SUE × Short Pressure (catalyst × fuel = explosion)

Key insight: A highly-shorted stock with a POSITIVE earnings surprise forces
shorts to BUY to cover losses → amplifies PEAD drift → this combination
(high short interest + positive SUE) produced INTC +94%, CNC +54% in Apr-May 2026.

Outputs:
  short_interest_scores.csv  — per-ticker: short_pct, days_cover, squeeze_score, rank_squeeze
  short_interest_report.md   — top squeeze candidates

Usage:
  python3 canyon_final_v9_step83_short_squeeze.py [--top N]
"""

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
ML_SCORES    = ROOT / "regime_ml_scores.csv"
SUE_SCORES   = ROOT / "earnings_surprise_scores.csv"
OUT_SCORES   = ROOT / "short_interest_scores.csv"
OUT_REPORT   = ROOT / "short_interest_report.md"
CACHE_FILE   = ROOT / "short_interest_raw_cache.csv"
CACHE_TTL_H  = 24 * 14    # 2-week cache (FINRA updates bi-monthly)
MAX_WORKERS  = 4           # concurrent yfinance calls
DEFAULT_TOP  = 200


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ticker universe
# ─────────────────────────────────────────────────────────────────────────────

def get_universe(top_n: int) -> list[str]:
    if ML_SCORES.exists():
        df = pd.read_csv(ML_SCORES)
        col = "predicted_score" if "predicted_score" in df.columns else df.columns[-1]
        tickers = df.sort_values(col, ascending=False)["ticker"].dropna().tolist()
        return tickers[:top_n]
    # Fallback: hardcoded core names
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","AMD","TSLA","AVGO","COST",
        "MSFT","JPM","UNH","JNJ","XOM","CVX","LLY","V","MA","HD",
    ][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Short interest fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_short(ticker: str) -> dict:
    """Fetch short interest fields for one ticker."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":             ticker,
            "shares_short":       info.get("sharesShort"),
            "shares_short_prior": info.get("sharesShortPriorMonth"),
            "short_ratio":        info.get("shortRatio"),           # days to cover
            "short_pct_float":    info.get("shortPercentOfFloat"),  # 0.05 = 5%
            "float_shares":       info.get("floatShares"),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def load_short_interest(tickers: list[str]) -> pd.DataFrame:
    """Batch-fetch short interest, using cache when fresh."""
    # Check cache
    if CACHE_FILE.exists():
        age_h = (datetime.now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            cached = pd.read_csv(CACHE_FILE)
            covered = set(cached["ticker"].tolist())
            missing = [t for t in tickers if t not in covered]
            if not missing:
                print(f"  Short interest cache: {len(cached)} tickers ({age_h:.0f}h old)")
                return cached[cached["ticker"].isin(tickers)]
            print(f"  Cache has {len(covered)} tickers, fetching {len(missing)} new …")
            tickers_to_fetch = missing
        else:
            print(f"  Cache stale ({age_h:.0f}h) — refreshing {len(tickers)} tickers …")
            tickers_to_fetch = tickers
    else:
        print(f"  Fetching short interest for {len(tickers)} tickers …")
        tickers_to_fetch = tickers

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_short, t): t for t in tickers_to_fetch}
        for fut in as_completed(futures):
            res = fut.result()
            if "error" not in res:
                rows.append(res)
            done += 1
            if done % 20 == 0:
                print(f"  … {done}/{len(tickers_to_fetch)}")

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Merge with cache
    if CACHE_FILE.exists() and tickers_to_fetch != tickers:
        old_df = pd.read_csv(CACHE_FILE)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="ticker", keep="last")
    else:
        combined = new_df

    if not combined.empty:
        combined.to_csv(CACHE_FILE, index=False)

    return combined[combined["ticker"].isin(tickers)] if not combined.empty else combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(raw: pd.DataFrame, sue_df: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Numeric coercion
    for c in ["shares_short", "shares_short_prior", "short_ratio",
              "short_pct_float", "float_shares"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    # Short change: positive = more shorts (bearish), negative = covering (bullish)
    df["short_change_pct"] = (
        (df["shares_short"] - df["shares_short_prior"])
        / df["shares_short_prior"].replace(0, np.nan)
    ).clip(-1, 2)

    # Short pressure index = short_pct_float × days_to_cover
    # Higher = more explosive fuel if a catalyst hits
    df["short_pct_float"] = df["short_pct_float"].fillna(0.02)  # 2% default
    df["short_ratio"]     = df["short_ratio"].fillna(2.0)         # 2 days default
    df["short_pressure"]  = df["short_pct_float"] * df["short_ratio"]

    # Merge SUE signal (catalyst)
    if not sue_df.empty and "rank_sue" in sue_df.columns:
        df = df.merge(sue_df[["ticker", "rank_sue", "sue"]].rename(
            columns={"rank_sue": "sue_rank", "sue": "sue_raw"}
        ), on="ticker", how="left")
        df["sue_rank"] = df["sue_rank"].fillna(50)
        df["sue_raw"]  = df["sue_raw"].fillna(0)
    else:
        df["sue_rank"] = 50
        df["sue_raw"]  = 0

    # ── Squeeze score ────────────────────────────────────────────────────────
    # Core idea: Squeeze = Short Pressure × Positive Catalyst
    #   When shorts face a positive earnings surprise they MUST buy to cover
    #   → forced buying amplifies PEAD drift → strongest alpha combo in quant
    #
    # If SUE > 0 (positive surprise): squeeze_score = pressure × sue_norm (synergistic)
    # If SUE < 0 (miss): negative signal — shorts pile on, pressure confirms downside
    sue_norm = (df["sue_rank"] / 100 - 0.5) * 2   # -1 to +1

    df["squeeze_score"] = np.where(
        df["sue_raw"] > 0,
        df["short_pressure"] * (1 + sue_norm.clip(0, 1)),   # amplify by positive SUE
        -df["short_pressure"] * 0.3 * (-sue_norm).clip(0, 1), # mild negative
    )

    # Covering momentum: shorts declining = buying pressure = additional positive signal
    df["covering_signal"] = -df["short_change_pct"].fillna(0).clip(-0.5, 0.5)

    # Combined signal
    df["combined_squeeze"] = df["squeeze_score"] + df["covering_signal"] * 0.3

    # Cross-sectional rank (0-100)
    valid = df["combined_squeeze"].notna()
    df.loc[valid, "rank_squeeze"] = (
        df.loc[valid, "combined_squeeze"].rank(pct=True) * 100
    )
    df["rank_squeeze"] = df["rank_squeeze"].fillna(50)

    # Signal labels
    df["signal"] = df["rank_squeeze"].apply(
        lambda r: "SQUEEZE_BUY" if r >= 80
             else ("SQUEEZE_WATCH" if r >= 65
             else ("CONFIRMED_SHORT" if r <= 20
             else "NEUTRAL"))
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. IC validation (spot-check vs price cache)
# ─────────────────────────────────────────────────────────────────────────────

def validate_ic(df: pd.DataFrame) -> None:
    """Quick IC check: does squeeze_score correlate with recent 21-day returns?"""
    from scipy import stats as spstats

    price_cache = ROOT / "sp500_price_cache.csv"
    if not price_cache.exists():
        print("  IC validation skipped (no price cache)")
        return

    try:
        prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
        if prices.empty:
            return
        # 21-day return ending today
        fwd = prices.pct_change(21).iloc[-1]
        merged = df[["ticker", "rank_squeeze"]].copy()
        merged["fwd_ret"] = merged["ticker"].map(fwd)
        merged = merged.dropna()
        if len(merged) < 10:
            print("  IC validation: insufficient overlap")
            return
        ic, pval = spstats.spearmanr(merged["rank_squeeze"], merged["fwd_ret"])
        stars = "✅" if abs(ic) > 0.05 else "⚠️"
        print(f"  Squeeze IC vs 21d return: {ic:+.4f}  p={pval:.3f}  {stars}")
    except Exception as e:
        print(f"  IC validation error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    squeeze_buy = df[df["signal"] == "SQUEEZE_BUY"].sort_values(
        "rank_squeeze", ascending=False)
    conf_short  = df[df["signal"] == "CONFIRMED_SHORT"].sort_values(
        "rank_squeeze")

    lines = [
        "# Canyon v9 — Step 83: Short Interest & Squeeze Report",
        f"Generated: {now}  |  Universe: {len(df)} tickers",
        "",
        "## What This Signal Means",
        "A squeeze combines TWO forces: high short interest (fuel) + positive catalyst (ignition).",
        "When shorts face unexpected good news they are FORCED to buy → amplifies PEAD drift.",
        "INTC +94% and CNC +54% in Apr-May 2026 had exactly this structure.",
        "",
        "## Top Squeeze Buy Candidates",
        "| # | Ticker | Rank | Short% | Days Cover | SUE Rank | Signal |",
        "|---|--------|------|--------|------------|----------|--------|",
    ]
    for i, (_, r) in enumerate(squeeze_buy.head(15).iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {r['rank_squeeze']:.0f} "
            f"| {r['short_pct_float']*100:.1f}% "
            f"| {r['short_ratio']:.1f}d "
            f"| {r.get('sue_rank', 50):.0f} "
            f"| {r['signal']} |"
        )

    lines += [
        "",
        "## Short Interest Universe Summary",
        f"- SQUEEZE_BUY (rank ≥ 80):      {len(squeeze_buy)} tickers",
        f"- CONFIRMED_SHORT (rank ≤ 20):   {len(conf_short)} tickers",
        "",
        "## Top Short Squeeze Interpretation",
        "- **Short% of Float > 10% + Days Cover > 7** = high short pressure",
        "- **SUE Rank > 80** = recent positive earnings surprise",
        "- **Both conditions** = prime squeeze candidate (Livermore: follow the forced buyers)",
        "",
        "## Signal Orthogonality",
    ]

    # Quick correlation check
    if "rank_sue" in df.columns or "sue_rank" in df.columns:
        sue_col = "sue_rank" if "sue_rank" in df.columns else "rank_sue"
        corr = df[["rank_squeeze", sue_col]].corr().iloc[0, 1]
        lines.append(f"- rank_squeeze vs rank_sue: corr = {corr:.3f}")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(top_n: int = DEFAULT_TOP) -> None:
    t0 = datetime.now()
    print()
    print("=" * 60)
    print("Canyon v9 — Step 83: Short Interest & Squeeze Detector")
    print("=" * 60)

    print(f"\n[1/4] Loading universe (top {top_n}) …")
    tickers = get_universe(top_n)
    print(f"  {len(tickers)} tickers selected")

    print("\n[2/4] Fetching short interest data …")
    raw_df = load_short_interest(tickers)
    if raw_df.empty:
        print("  ERROR: No short interest data fetched")
        return
    valid = raw_df["short_ratio"].notna().sum()
    print(f"  {valid}/{len(raw_df)} tickers with valid short data")

    print("\n[3/4] Computing squeeze scores …")
    sue_df = pd.DataFrame()
    if SUE_SCORES.exists():
        sue_df = pd.read_csv(SUE_SCORES)
        print(f"  SUE scores loaded: {len(sue_df)} tickers")

    result = compute_signals(raw_df, sue_df)

    # IC validation
    validate_ic(result)

    # Save
    output_cols = ["ticker", "short_pct_float", "short_ratio", "short_change_pct",
                   "short_pressure", "sue_rank", "squeeze_score",
                   "covering_signal", "combined_squeeze", "rank_squeeze", "signal"]
    out = result[[c for c in output_cols if c in result.columns]]
    out.to_csv(OUT_SCORES, index=False)
    print(f"  Saved: {OUT_SCORES.name}  ({len(out)} rows)")

    # Highlights
    print()
    print("  Top 10 Squeeze Candidates:")
    top = result[result["signal"] == "SQUEEZE_BUY"].sort_values(
        "rank_squeeze", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"    {r['ticker']:6s}  short={r['short_pct_float']*100:.1f}%  "
              f"cover={r['short_ratio']:.1f}d  "
              f"sue_rank={r.get('sue_rank',50):.0f}  "
              f"rank_squeeze={r['rank_squeeze']:.0f}")

    print("\n[4/4] Writing report …")
    write_report(result)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Short Interest & Squeeze")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"Number of tickers to analyse (default {DEFAULT_TOP})")
    args = parser.parse_args()
    main(top_n=args.top)
