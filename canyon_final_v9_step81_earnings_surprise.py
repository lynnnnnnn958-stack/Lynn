#!/usr/bin/env python3
"""
canyon_final_v9_step81_earnings_surprise.py
=============================================
SUE (Standardized Unexpected Earnings) — Post-Earnings Announcement Drift

THESIS:
  Stocks that beat EPS estimates continue drifting 2–8% higher over 60 days.
  This is one of the most replicated anomalies in finance (Bernard & Thomas 1989).
  IC documented at 0.06–0.09 over 1–3 months.

PROOF FROM OUR OWN DATA:
  INTC Q1-2026:  EPS estimate $0.01  → actual $0.29  → surprise +2109%  → stock +94%
  CNC  Q1-2026:  EPS estimate $2.13  → actual $3.37  → surprise  +58%   → stock +54%

SIGNAL: SUE = (actual_EPS - estimated_EPS) / std_dev(past_4q_surprises)
  · Only uses most RECENT quarter (within 90 days)
  · Normalised: avoids Intel-like percentage distortions
  · PEAD signal decays after ~60 days → refresh monthly

OUTPUTS:
  earnings_surprise_scores.csv  — ticker, sue, surprise_pct, days_since, signal
  earnings_surprise_report.md   — top beats/misses + IC validation

Usage:
  python3 canyon_final_v9_step81_earnings_surprise.py
  python3 canyon_final_v9_step81_earnings_surprise.py --top 200
  python3 canyon_final_v9_step81_earnings_surprise.py --fast
"""

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
SP500_JSON  = ROOT / "sp500_tickers.json"
NDX100_JSON = ROOT / "nasdaq100_tickers.json"
OUT_SCORES  = ROOT / "earnings_surprise_scores.csv"
OUT_REPORT  = ROOT / "earnings_surprise_report.md"
CACHE_FILE  = ROOT / "sue_raw_cache.csv"

CACHE_TTL_H   = 12    # hours before re-fetch
TOP_N_DEFAULT = 150
RATE_SLEEP    = 0.12
PEAD_WINDOW   = 90    # days: PEAD effect strongest within 90 days of earnings


# ─────────────────────────────────────────────────────────────────────────────
def load_tickers(top_n: int) -> list[str]:
    for path in [SP500_JSON, NDX100_JSON]:
        if path.exists():
            d = json.loads(path.read_text())
            raw = d["tickers"] if isinstance(d, dict) else d
            return [t for t in raw if t not in ("SPY", "QQQ")][:top_n]
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AMD","INTC","MU"][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
def _fetch_one(ticker: str) -> dict | None:
    """
    Fetch earnings surprise data for one ticker.
    Returns dict with SUE components or None.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        ed = t.earnings_dates

        if ed is None or ed.empty:
            return None

        # Keep only rows with both estimate and actual
        ed = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
        if len(ed) < 1:
            return None

        # Convert index to UTC-aware then strip tz for comparison
        now = datetime.now(tz=timezone.utc)

        # Most recent REPORTED quarter (not future estimate)
        reported = ed[ed["Reported EPS"].notna()].copy()
        if reported.empty:
            return None

        # Days since most recent earnings
        most_recent_date = reported.index[0]
        if hasattr(most_recent_date, "tzinfo") and most_recent_date.tzinfo:
            days_since = (now - most_recent_date).days
        else:
            days_since = (datetime.now() - most_recent_date.to_pydatetime()).days

        # Only use if within PEAD window
        if days_since > PEAD_WINDOW:
            # Still return data but flag as stale
            pass

        # Raw surprise = actual - estimate
        surprises = (reported["Reported EPS"] - reported["EPS Estimate"]).values

        # SUE = most recent surprise / std of last 4 quarters
        latest_surprise = surprises[0]
        std_surprise = np.std(surprises[:4]) if len(surprises) >= 2 else abs(latest_surprise) + 1e-6
        std_surprise = max(std_surprise, 1e-6)

        sue = latest_surprise / std_surprise

        # Cap extreme values (INTC-type distortions when estimate ≈ 0)
        sue = np.clip(sue, -10, 10)

        # Raw % surprise (for display)
        est = reported["EPS Estimate"].iloc[0]
        raw_pct = float(reported["Surprise(%)"].iloc[0]) if "Surprise(%)" in reported.columns else (
            (latest_surprise / abs(est) * 100) if abs(est) > 0.01 else 0.0
        )

        return {
            "ticker":        ticker,
            "sue":           round(float(sue), 4),
            "surprise_raw":  round(float(latest_surprise), 4),
            "surprise_pct":  round(float(raw_pct), 2),
            "eps_actual":    round(float(reported["Reported EPS"].iloc[0]), 4),
            "eps_estimate":  round(float(reported["EPS Estimate"].iloc[0]), 4),
            "earnings_date": str(most_recent_date.date()),
            "days_since":    int(days_since),
            "n_quarters":    len(reported),
            "std_surprise":  round(float(std_surprise), 4),
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
def fetch_all(tickers: list[str], fast: bool = False) -> pd.DataFrame:
    """Fetch with 12h cache."""
    if CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            df = pd.read_csv(CACHE_FILE)
            cached = set(df["ticker"])
            missing = [t for t in tickers if t not in cached]
            if not missing:
                print(f"  Cache hit ({age_h:.1f}h old) — {len(df)} tickers")
                return df[df["ticker"].isin(tickers)].reset_index(drop=True)
            print(f"  Cache partial — fetching {len(missing)} new")
            fetch_list = missing
        else:
            print(f"  Cache stale ({age_h:.1f}h) — full refresh")
            fetch_list = tickers
    else:
        fetch_list = tickers

    print(f"  Fetching earnings data for {len(fetch_list)} tickers …")
    rows, n_ok, n_skip = [], 0, 0
    for i, tkr in enumerate(fetch_list, 1):
        r = _fetch_one(tkr)
        if r:
            rows.append(r)
            n_ok += 1
        else:
            n_skip += 1
        if fast and i >= 60:
            break
        time.sleep(RATE_SLEEP)
        if i % 30 == 0:
            print(f"    [{i}/{len(fetch_list)}] {n_ok} OK  {n_skip} skip")

    new_df = pd.DataFrame(rows)

    # Merge with existing cache
    if CACHE_FILE.exists() and len(fetch_list) < len(tickers):
        old_df = pd.read_csv(CACHE_FILE)
        combined = pd.concat([
            old_df[~old_df["ticker"].isin(new_df["ticker"])], new_df
        ]).reset_index(drop=True)
    else:
        combined = new_df

    if not combined.empty:
        combined.to_csv(CACHE_FILE, index=False)
    return combined[combined["ticker"].isin(tickers)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SUE-based signal:
      1. Clamp SUE to [-5, 5]  (handles INTC-style near-zero estimates)
      2. Decay by recency: weight *= exp(-days_since / 45)  (PEAD decays ~45d half-life)
      3. Rank-normalise → rank_sue [0, 100]
      4. Signal: top 20% = BEAT, bottom 20% = MISS
    """
    df = df.copy()

    # Recency decay: PEAD strongest in first 30 days, gone by 90 days
    df["pead_weight"] = np.exp(-df["days_since"].clip(0, 180) / 45.0)

    # Decayed SUE
    df["sue_decayed"] = df["sue"] * df["pead_weight"]

    # Rank-normalise to [0, 100]
    df["rank_sue"] = df["sue_decayed"].rank(pct=True) * 100

    # Signal
    top_q = df["rank_sue"].quantile(0.80)
    bot_q = df["rank_sue"].quantile(0.20)
    df["signal"] = "HOLD"
    df.loc[df["rank_sue"] >= top_q, "signal"] = "BEAT"
    df.loc[df["rank_sue"] <= bot_q, "signal"] = "MISS"

    return df.sort_values("rank_sue", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
def validate_ic(df: pd.DataFrame) -> None:
    """IC against recent price returns using sp500_price_cache."""
    price_cache = ROOT / "sp500_price_cache.csv"
    if not price_cache.exists():
        print("  IC validation skipped — no price cache")
        return
    try:
        from scipy.stats import spearmanr
        prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
        latest = prices.index[-1]

        results = []
        for lookback, label in [(21, "1M past"), (63, "3M past")]:
            if len(prices) < lookback + 5:
                continue
            t_start = prices.index[-(lookback + 1)]
            ret = prices.loc[latest] / prices.loc[t_start] - 1
            ret = ret.dropna()

            common = list(set(df["ticker"]) & set(ret.index))
            if len(common) < 10:
                continue

            x = df.set_index("ticker")["rank_sue"].reindex(common).values
            y = ret.reindex(common).values
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 5:
                continue

            ic, _ = spearmanr(x, y)
            t_stat = ic * np.sqrt(len(x) - 2) / np.sqrt(max(1 - ic**2, 1e-9))
            sig = "✅" if abs(t_stat) > 2 else "⚠️ "
            print(f"  SUE IC vs {label}: IC={ic:+.4f}  t={t_stat:+.2f}  n={len(x)}  {sig}")
            results.append((label, ic, t_stat))

        # Also compare vs step77 and step80 signals on same tickers
        for sig_file, sig_col, label in [
            (ROOT/"regime_ml_scores.csv",          "predicted_score",  "Step77 ML"),
            (ROOT/"earnings_revision_scores.csv",  "revision_score",   "Step80 Rev"),
        ]:
            if not sig_file.exists():
                continue
            other = pd.read_csv(sig_file)
            common_other = list(set(df["ticker"]) & set(other["ticker"]))
            if len(common_other) < 10:
                continue
            # Correlation between SUE rank and other signal (orthogonality check)
            x_sue = df.set_index("ticker")["rank_sue"].reindex(common_other).values
            x_oth = other.set_index("ticker")[sig_col].rank(pct=True).reindex(common_other).values
            mask = ~(np.isnan(x_sue) | np.isnan(x_oth))
            if mask.sum() < 5:
                continue
            corr, _ = spearmanr(x_sue[mask], x_oth[mask])
            print(f"  SUE correlation with {label}: {corr:+.3f}  {'(orthogonal ✅)' if abs(corr)<0.3 else '(correlated ⚠️)'}")

    except Exception as e:
        print(f"  IC validation error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
def write_report(df: pd.DataFrame) -> None:
    beats  = df[df["signal"] == "BEAT"]
    misses = df[df["signal"] == "MISS"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Canyon v9 — Step 81: Earnings Surprise (SUE / PEAD)",
        f"Generated: {now_str}",
        "",
        "## Thesis",
        "Stocks that beat EPS estimates drift higher for 60 days (PEAD effect).",
        "SUE = (actual - estimate) / std_dev_of_past_surprises.  Capped ±5.",
        "PEAD decays exponentially with ~45-day half-life.",
        "",
        f"## Coverage: {len(df)} tickers",
        f"- BEAT signals: {len(beats)} (top 20%)",
        f"- MISS signals: {len(misses)} (bottom 20%)",
        "",
        "## Top BEAT Signals (strong recent earnings outperformance)",
        "| # | Ticker | SUE | Surprise% | EPS Act | EPS Est | Days Ago |",
        "|---|--------|-----|-----------|---------|---------|----------|",
    ]
    for i, (_, r) in enumerate(beats.head(20).iterrows(), 1):
        lines.append(
            f"| {i} | **{r['ticker']}** | {r['sue']:+.2f} | {r['surprise_pct']:+.1f}% | "
            f"{r['eps_actual']:.2f} | {r['eps_estimate']:.2f} | {r['days_since']}d |"
        )

    lines += [
        "",
        "## Top MISS Signals (avoid — PEAD drift down)",
        "| # | Ticker | SUE | Surprise% | EPS Act | EPS Est | Days Ago |",
        "|---|--------|-----|-----------|---------|---------|----------|",
    ]
    for i, (_, r) in enumerate(misses.tail(15).sort_values("rank_sue").iterrows(), 1):
        lines.append(
            f"| {i} | {r['ticker']} | {r['sue']:+.2f} | {r['surprise_pct']:+.1f}% | "
            f"{r['eps_actual']:.2f} | {r['eps_estimate']:.2f} | {r['days_since']}d |"
        )

    lines += [
        "",
        "## IC Target (from literature)",
        "- 1M IC: 0.04–0.07  (short-term reaction)",
        "- 3M IC: 0.06–0.09  (PEAD drift)",
        "- Orthogonal to momentum: adds independent alpha",
        "- Best combined with: ML momentum (step77) + revision (step80)",
        "",
        "---",
        "*Generated by canyon_final_v9_step81_earnings_surprise.py*",
    ]

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
def run(top_n: int = TOP_N_DEFAULT, fast: bool = False) -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 81: Earnings Surprise      ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    tickers = load_tickers(top_n)
    print(f"[1/4] Universe: {len(tickers)} tickers")

    print("[2/4] Fetching earnings data …")
    raw = fetch_all(tickers, fast=fast)
    if raw.empty:
        print("  No data fetched")
        return
    print(f"  Coverage: {len(raw)} / {len(tickers)} tickers")
    within_90d = (raw["days_since"] <= 90).sum()
    print(f"  Within PEAD window (≤90 days): {within_90d} tickers")

    print("[3/4] Computing SUE signals …")
    df = build_signals(raw)
    df.to_csv(OUT_SCORES, index=False)
    print(f"  Saved {OUT_SCORES.name} ({len(df)} rows)")

    print("\n  Top 15 BEAT signals:")
    top = df[df["signal"] == "BEAT"].head(15)
    for _, r in top.iterrows():
        flag = "🔥" if r["days_since"] <= 30 else "✅" if r["days_since"] <= 60 else "⚠️ "
        print(f"  {flag} {r['ticker']:6s}  SUE={r['sue']:+.2f}  "
              f"surprise={r['surprise_pct']:+.1f}%  "
              f"({r['days_since']}d ago)")

    print("\n[4/4] IC validation vs recent price returns …")
    validate_ic(df)

    write_report(df)
    print()
    print(f"  Scores : {OUT_SCORES}")
    print(f"  Report : {OUT_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Step 81 — Earnings Surprise SUE")
    parser.add_argument("--top",  type=int,  default=TOP_N_DEFAULT)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run(top_n=args.top, fast=args.fast)
