#!/usr/bin/env python3
"""
canyon_final_v9_step80_earnings_revision.py
============================================
Analyst Earnings Revision Alpha Signal

THESIS:
  Stocks where analysts are UPGRADING outerperform by 2-4% over 3 months.
  This is one of the most consistently documented alpha sources in academic
  literature (Womack 1996, Jegadeesh et al. 2004).

  Source: yfinance recommendations_summary (free, no API key needed)
    - Tracks # of Strong Buy / Buy / Hold / Sell / Strong Sell per month
    - Revision score = net upgrade momentum over rolling 1-month window

SIGNAL CONSTRUCTION:
  bull_score(t)  = strongBuy(0m) + buy(0m)
  bull_score(-1) = strongBuy(-1m) + buy(-1m)
  bear_score(t)  = sell(0m) + strongSell(0m)
  bear_score(-1) = sell(-1m) + strongSell(-1m)

  revision_raw = [bull_score(t) - bull_score(-1)] - [bear_score(t) - bear_score(-1)]
  revision_norm = revision_raw / max(total_analysts, 1)  → range roughly [-1, +1]
  rank_revision = percentile rank across universe

IC PROPERTIES (known from literature):
  - 1-month IC:  ~0.04–0.06  (short-term mean reversion after announcement)
  - 3-month IC:  ~0.06–0.10  (PEAD — post-earnings announcement drift)
  - Orthogonal to momentum: adds independent alpha when combined

OUTPUTS:
  earnings_revision_scores.csv   — ticker, revision_raw, revision_norm, rank_revision,
                                   bull_now, bear_now, bull_chg, bear_chg, n_analysts, signal
  earnings_revision_report.md    — coverage + top signals

Usage:
  python3 canyon_final_v9_step80_earnings_revision.py
  python3 canyon_final_v9_step80_earnings_revision.py --top 200   # larger universe
  python3 canyon_final_v9_step80_earnings_revision.py --fast       # skip slow tickers
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).parent
SP500_JSON    = ROOT / "sp500_tickers.json"
NDX100_JSON   = ROOT / "nasdaq100_tickers.json"
OUT_SCORES    = ROOT / "earnings_revision_scores.csv"
OUT_REPORT    = ROOT / "earnings_revision_report.md"
CACHE_FILE    = ROOT / "revision_raw_cache.csv"

CACHE_TTL_H   = 8     # hours before re-fetching
TOP_N_DEFAULT = 100   # default universe size
RATE_SLEEP    = 0.15  # seconds between yfinance calls

# Grade → sentiment bucket
BULLISH_GRADES = {
    "Strong Buy", "Buy", "Outperform", "Overweight", "Positive",
    "Add", "Long-Term Buy", "Conviction Buy", "Market Outperform",
    "Sector Outperform", "Accumulate",
}
BEARISH_GRADES = {
    "Sell", "Strong Sell", "Underperform", "Underweight", "Negative",
    "Reduce", "Avoid", "Market Underperform", "Sector Underperform",
}


# ─────────────────────────────────────────────────────────────────────────────
# TICKER LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_tickers(top_n: int = TOP_N_DEFAULT) -> list[str]:
    """Load from sp500_tickers.json, fall back to NDX 100."""
    for path in [SP500_JSON, NDX100_JSON]:
        if path.exists():
            d = json.loads(path.read_text())
            raw = d["tickers"] if isinstance(d, dict) else d
            tickers = [t for t in raw if t not in ("SPY", "QQQ")]
            return tickers[:top_n]
    # Hard fallback
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","NFLX",
        "AMD","ADBE","CSCO","QCOM","INTU","TXN","MU","ADI","KLAC","LRCX",
        "PANW","CRWD","AMD","CRM","ORCL","INTU","PYPL","COIN",
    ][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one(ticker: str) -> dict | None:
    """Fetch analyst recommendations for one ticker. Returns dict or None."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        rec = t.recommendations_summary

        if rec is None or rec.empty:
            return None

        # Ensure required columns exist
        for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
            if col not in rec.columns:
                rec[col] = 0

        # 0m = most recent, -1m = prior month
        row_now  = rec.iloc[0] if len(rec) >= 1 else None
        row_prev = rec.iloc[1] if len(rec) >= 2 else None

        if row_now is None:
            return None

        bull_now  = int(row_now.get("strongBuy", 0)) + int(row_now.get("buy", 0))
        bear_now  = int(row_now.get("sell", 0))      + int(row_now.get("strongSell", 0))
        hold_now  = int(row_now.get("hold", 0))
        total_now = bull_now + bear_now + hold_now

        bull_prev = bear_prev = 0
        if row_prev is not None:
            bull_prev = int(row_prev.get("strongBuy", 0)) + int(row_prev.get("buy", 0))
            bear_prev = int(row_prev.get("sell", 0))      + int(row_prev.get("strongSell", 0))

        bull_chg = bull_now - bull_prev
        bear_chg = bear_now - bear_prev
        revision_raw = bull_chg - bear_chg

        return {
            "ticker":       ticker,
            "bull_now":     bull_now,
            "bear_now":     bear_now,
            "hold_now":     hold_now,
            "n_analysts":   total_now,
            "bull_chg":     bull_chg,
            "bear_chg":     bear_chg,
            "revision_raw": revision_raw,
        }

    except Exception:
        return None


def fetch_all(tickers: list[str], fast: bool = False) -> pd.DataFrame:
    """Fetch analyst data for all tickers, using cache if fresh."""
    # Check cache
    if CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            df = pd.read_csv(CACHE_FILE)
            cached_tickers = set(df["ticker"])
            missing = [t for t in tickers if t not in cached_tickers]
            if not missing:
                print(f"  Cache hit ({age_h:.1f}h old) — {len(df)} tickers")
                return df[df["ticker"].isin(tickers)].reset_index(drop=True)
            print(f"  Cache partial — fetching {len(missing)} new tickers")
            tickers_to_fetch = missing
        else:
            print(f"  Cache stale ({age_h:.1f}h) — re-fetching all")
            tickers_to_fetch = tickers
    else:
        tickers_to_fetch = tickers

    print(f"  Fetching analyst data for {len(tickers_to_fetch)} tickers …")
    rows = []
    n_ok = n_skip = 0
    for i, tkr in enumerate(tickers_to_fetch, 1):
        result = _fetch_one(tkr)
        if result:
            rows.append(result)
            n_ok += 1
        else:
            n_skip += 1
        if fast and i > 60:
            break
        time.sleep(RATE_SLEEP)
        if i % 25 == 0:
            print(f"    [{i}/{len(tickers_to_fetch)}] {n_ok} OK  {n_skip} skip")

    new_df = pd.DataFrame(rows)

    # Merge with existing cache
    if CACHE_FILE.exists() and len(tickers_to_fetch) < len(tickers):
        old_df = pd.read_csv(CACHE_FILE)
        combined = pd.concat([
            old_df[~old_df["ticker"].isin(new_df["ticker"])],
            new_df
        ]).reset_index(drop=True)
    else:
        combined = new_df

    if not combined.empty:
        combined.to_csv(CACHE_FILE, index=False)

    return combined[combined["ticker"].isin(tickers)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized revision score and rank.
    revision_norm = revision_raw / max(n_analysts, 1)
    rank_revision  = percentile rank [0,100]
    """
    df = df.copy()

    # Normalise by analyst coverage
    df["revision_norm"] = df["revision_raw"] / df["n_analysts"].clip(lower=1)

    # Percentile rank (0=worst, 100=best)
    df["rank_revision"] = df["revision_norm"].rank(pct=True) * 100

    # Bull ratio — fraction of analysts that are bullish
    df["bull_ratio"] = df["bull_now"] / df["n_analysts"].clip(lower=1)
    df["rank_bull_ratio"] = df["bull_ratio"].rank(pct=True) * 100

    # Combined score: 60% revision momentum + 40% absolute bull ratio
    df["revision_score"] = (
        df["rank_revision"]  * 0.60 +
        df["rank_bull_ratio"] * 0.40
    )

    # Signal labels
    top_q  = df["revision_score"].quantile(0.80)
    bot_q  = df["revision_score"].quantile(0.20)
    df["signal"] = "HOLD"
    df.loc[df["revision_score"] >= top_q, "signal"] = "UPGRADE"    # top 20%
    df.loc[df["revision_score"] <= bot_q, "signal"] = "DOWNGRADE"  # bottom 20%

    return df.sort_values("revision_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# IC VALIDATION (requires price data from step75/77 cache)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_ic(df: pd.DataFrame) -> None:
    """
    Quick sanity-check: does revision_score predict forward returns?
    Uses sp500_price_cache.csv if available.
    """
    price_cache = ROOT / "sp500_price_cache.csv"
    if not price_cache.exists():
        print("  IC validation skipped — sp500_price_cache.csv not found")
        return

    try:
        from scipy import stats as spstats
        prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)

        latest = prices.index[-1]
        tickers = df["ticker"].tolist()
        tickers_ok = [t for t in tickers if t in prices.columns]
        if len(tickers_ok) < 10:
            print("  IC validation skipped — insufficient price overlap")
            return

        rev_scores = df.set_index("ticker")["revision_score"]

        results = []
        for fwd_days, label in [(21, "1M"), (63, "3M")]:
            fwd_idx = prices.index.searchsorted(latest) + fwd_days
            if fwd_idx >= len(prices):
                continue
            fwd_date = prices.index[min(fwd_idx, len(prices) - 1)]
            fwd_ret  = prices.loc[fwd_date, tickers_ok] / prices.loc[latest, tickers_ok] - 1

            common = list(set(tickers_ok) & set(fwd_ret.dropna().index))
            if len(common) < 10:
                continue

            x = rev_scores.reindex(common).values
            y = fwd_ret.reindex(common).values
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 5:
                continue

            ic, pval = spstats.spearmanr(x, y)
            tstat = ic * np.sqrt(len(x) - 2) / np.sqrt(max(1 - ic**2, 1e-9))
            sig = "✅" if abs(tstat) > 2 else "❌"
            results.append((label, ic, tstat, len(x)))
            print(f"  IC vs {label} fwd ret: IC={ic:+.3f}  t={tstat:+.2f}  n={len(x)}  {sig}")

        if not results:
            print("  IC validation: not enough forward data yet (need 3+ months of new prices)")

    except Exception as e:
        print(f"  IC validation error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame) -> None:
    upgrades   = df[df["signal"] == "UPGRADE"]
    downgrades = df[df["signal"] == "DOWNGRADE"]
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Canyon v9 — Step 80: Earnings Revision Report",
        f"Generated: {now_str}",
        "",
        "## Thesis",
        "Analyst upgrades predict 2-4% outperformance over 3 months (PEAD effect).",
        "Signal = net upgrade momentum (bull change - bear change) + bull ratio.",
        "",
        f"## Coverage: {len(df)} tickers",
        f"- UPGRADE signals: {len(upgrades)} (top 20%)",
        f"- DOWNGRADE signals: {len(downgrades)} (bottom 20%)",
        f"- HOLD: {len(df) - len(upgrades) - len(downgrades)}",
        "",
        "## Top UPGRADE Signals",
        "| # | Ticker | Score | Revision | Bull Now | Bear Now | N Analysts |",
        "|---|--------|-------|----------|----------|----------|------------|",
    ]
    for i, (_, row) in enumerate(upgrades.head(15).iterrows(), 1):
        chg_str = f"{row['bull_chg']:+d}/{row['bear_chg']:+d}"
        lines.append(
            f"| {i} | {row['ticker']} | {row['revision_score']:.1f} | "
            f"{row['revision_raw']:+.0f} ({chg_str}) | "
            f"{row['bull_now']} | {row['bear_now']} | {row['n_analysts']} |"
        )

    lines += [
        "",
        "## Top DOWNGRADE Signals",
        "| # | Ticker | Score | Revision | Bull Now | Bear Now | N Analysts |",
        "|---|--------|-------|----------|----------|----------|------------|",
    ]
    for i, (_, row) in enumerate(downgrades.tail(15).sort_values("revision_score").iterrows(), 1):
        chg_str = f"{row['bull_chg']:+d}/{row['bear_chg']:+d}"
        lines.append(
            f"| {i} | {row['ticker']} | {row['revision_score']:.1f} | "
            f"{row['revision_raw']:+.0f} ({chg_str}) | "
            f"{row['bull_now']} | {row['bear_now']} | {row['n_analysts']} |"
        )

    lines += [
        "",
        "## IC Notes",
        "- 1M IC target: 0.04–0.06 (short-term reaction to upgrade)",
        "- 3M IC target: 0.06–0.10 (PEAD drift — strongest known free alpha)",
        "- Revision signal is orthogonal to price momentum → additive alpha",
        "",
        "---",
        "*Generated by canyon_final_v9_step80_earnings_revision.py*",
    ]

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(top_n: int = TOP_N_DEFAULT, fast: bool = False) -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 80: Earnings Revision      ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # [1] Load universe
    tickers = load_tickers(top_n)
    print(f"[1/4] Universe: {len(tickers)} tickers")

    # [2] Fetch analyst data
    print("[2/4] Fetching analyst recommendations …")
    raw_df = fetch_all(tickers, fast=fast)
    if raw_df.empty:
        print("  No analyst data fetched — check internet connection")
        return
    print(f"  Coverage: {len(raw_df)} / {len(tickers)} tickers")

    # [3] Build revision signals
    print("[3/4] Building revision signals …")
    df = build_signals(raw_df)
    df.to_csv(OUT_SCORES, index=False)
    print(f"  Saved {OUT_SCORES.name} ({len(df)} rows)")

    # Show top 10
    print("\n  Top 10 UPGRADE signals:")
    top10 = df[df["signal"] == "UPGRADE"].head(10)
    for _, row in top10.iterrows():
        print(f"    {row['ticker']:6s}  score={row['revision_score']:.1f}  "
              f"revision={row['revision_raw']:+.0f}  "
              f"bull={row['bull_now']} bear={row['bear_now']}  "
              f"n={row['n_analysts']}")

    # [4] IC validation
    print("\n[4/4] IC validation …")
    _validate_ic(df)

    write_report(df)
    print()
    print(f"  Scores : {OUT_SCORES}")
    print(f"  Report : {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Step 80 — Earnings Revision")
    parser.add_argument("--top",  type=int, default=TOP_N_DEFAULT,
                        help=f"Universe size (default {TOP_N_DEFAULT})")
    parser.add_argument("--fast", action="store_true",
                        help="Stop after 60 tickers (quick test)")
    args = parser.parse_args()
    run(top_n=args.top, fast=args.fast)
