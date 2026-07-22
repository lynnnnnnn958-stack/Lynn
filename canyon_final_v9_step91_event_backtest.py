"""
Canyon v9 — Step 91: Earnings Event Strategy Backtest
======================================================
Backtests 3 event-driven options strategies around earnings events:
  1. Pre_Earnings_Straddle  — buy ATM straddle T-5 to T-1
  2. Post_Earnings_Short_Straddle — sell ATM straddle T+0 to T+5 (IV crush)
  3. Post_Earnings_Drift    — directional stock trade T+1 to T+10 / T+21 (PEAD)

Outputs
-------
  event_backtest_results.csv   — one row per event × strategy
  event_backtest_report.md     — full analysis: win rates, P&L, sector breakdown
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

PRICE_FILE     = ROOT / "sp500_price_cache.csv"
SURPRISE_FILE  = ROOT / "earnings_surprise_scores.csv"
REVISION_FILE  = ROOT / "earnings_revision_scores.csv"
REGIME_FILE    = ROOT / "regime_ml_scores.csv"

OUT_RESULTS    = ROOT / "event_backtest_results.csv"
OUT_REPORT     = ROOT / "event_backtest_report.md"

# ---------------------------------------------------------------------------
# Black-Scholes helpers (inline — no external dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2) * poly
    return approx if x >= 0 else 1.0 - approx


def bs_straddle(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """ATM straddle price = Black-Scholes call + put."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * _norm_cdf(d1)  - K * math.exp(-r * T) * _norm_cdf(d2)
    put  = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return call + put


def realized_vol_at(prices: pd.Series, date_idx: int, window: int = 21) -> float:
    """Annualised realised volatility ending at date_idx (exclusive)."""
    if date_idx < window + 1:
        return 0.30
    sub = prices.iloc[date_idx - window: date_idx]
    sub = sub.replace(0, np.nan).dropna()
    if len(sub) < 5:
        return 0.30
    log_ret = np.log(sub.values[1:] / sub.values[:-1])
    if len(log_ret) < 4:
        return 0.30
    vol = float(np.std(log_ret, ddof=1) * math.sqrt(252))
    return max(vol, 0.05)   # floor at 5 % to avoid degenerate option prices

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices() -> Optional[pd.DataFrame]:
    """Load sp500_price_cache.csv.  Returns DataFrame with DatetimeIndex."""
    if not PRICE_FILE.exists():
        print(f"[ERROR] Price file not found: {PRICE_FILE}")
        return None
    df = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    # Drop columns with > 95 % NaN
    df = df.dropna(axis=1, thresh=int(len(df) * 0.05))
    print(f"[Prices] Loaded {len(df)} trading days × {df.shape[1]} tickers "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load_earnings_calendar(years: int) -> Tuple[pd.DataFrame, str]:
    """
    Load earnings events.  Try earnings_surprise_scores.csv first, then
    earnings_revision_scores.csv.

    Returns (DataFrame with columns [ticker, earnings_date, surprise_pct],
             source name string).
    """
    date_col_candidates = ["earnings_date", "report_date", "date",
                           "earnings_date_str", "quarter_end"]

    def _try_load(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        # Identify date column
        found_date_col = None
        for c in date_col_candidates:
            if c in df.columns:
                found_date_col = c
                break
        if found_date_col is None:
            return None
        df = df.rename(columns={found_date_col: "earnings_date"})
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
        df = df[df["earnings_date"].notna()]
        # Keep surprise_pct if present
        if "surprise_pct" not in df.columns:
            df["surprise_pct"] = np.nan
        return df[["ticker", "earnings_date", "surprise_pct"]].copy()

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    df_surprise = _try_load(SURPRISE_FILE)
    df_revision = _try_load(REVISION_FILE)

    if df_surprise is not None and len(df_surprise) > 0:
        df = df_surprise.copy()
        source = "earnings_surprise_scores.csv"
    elif df_revision is not None and len(df_revision) > 0:
        df = df_revision.copy()
        df["surprise_pct"] = np.nan
        source = "earnings_revision_scores.csv"
    else:
        return pd.DataFrame(columns=["ticker", "earnings_date", "surprise_pct"]), "none"

    # Filter to last N years
    df = df[df["earnings_date"] >= cutoff].copy()

    # Deduplicate: keep most-recent per ticker per quarter
    df["quarter"] = df["earnings_date"].dt.to_period("Q")
    df = (df.sort_values("earnings_date")
            .drop_duplicates(subset=["ticker", "quarter"], keep="last")
            .drop(columns=["quarter"])
            .reset_index(drop=True))

    print(f"[Earnings] Loaded {len(df)} events from {source} "
          f"({df['ticker'].nunique()} tickers, last {years} yr)")
    return df, source


def load_sector_map() -> Dict[str, str]:
    """Return {ticker: sector} from regime_ml_scores.csv."""
    if not REGIME_FILE.exists():
        return {}
    df = pd.read_csv(REGIME_FILE)
    if "ticker" not in df.columns or "sector" not in df.columns:
        return {}
    return dict(zip(df["ticker"], df["sector"]))

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def offset_idx(dates: pd.DatetimeIndex, base_idx: int, offset: int) -> Optional[int]:
    """Return index = base_idx + offset, clamped to valid range."""
    new = base_idx + offset
    if 0 <= new < len(dates):
        return new
    return None


def find_date_idx(dates: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    """
    Find the index of the first trading day >= target.
    Returns -1 if target is beyond the last date.
    """
    pos = dates.searchsorted(target)
    if pos >= len(dates):
        return -1
    return int(pos)

# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_pre_earnings_straddle(
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    earn_idx: int,
    r: float = 0.05,
) -> Optional[dict]:
    """
    Buy ATM straddle at T-5, exit at T-1.
    Returns dict with trade metrics, or None if insufficient data.
    """
    idx_entry = offset_idx(dates, earn_idx, -5)
    idx_exit  = offset_idx(dates, earn_idx, -1)
    if idx_entry is None or idx_exit is None or idx_entry >= idx_exit:
        return None

    spot_entry = prices.iloc[idx_entry]
    spot_exit  = prices.iloc[idx_exit]
    if pd.isna(spot_entry) or pd.isna(spot_exit) or spot_entry <= 0 or spot_exit <= 0:
        return None

    rv = realized_vol_at(prices, idx_entry, window=21)
    iv_entry = rv * 1.30   # pre-earnings premium
    iv_exit  = rv * 1.30 * 1.10   # IV peaks just before earnings

    T_entry = 7 / 365
    T_exit  = 2 / 365
    K = spot_entry   # ATM: strike = spot at entry

    price_entry = bs_straddle(spot_entry, K, T_entry, r, iv_entry)
    price_exit  = bs_straddle(spot_exit,  K, T_exit,  r, iv_exit)

    if price_entry <= 0:
        return None

    pnl_pct = (price_exit - price_entry) / price_entry * 100.0

    return {
        "entry_date":  dates[idx_entry].date().isoformat(),
        "exit_date":   dates[idx_exit].date().isoformat(),
        "entry_price": round(price_entry, 4),
        "exit_price":  round(price_exit,  4),
        "spot_entry":  round(spot_entry,  2),
        "iv_entry":    round(iv_entry,    4),
        "pnl_pct":     round(pnl_pct,     4),
        "won":         int(pnl_pct > 0),
    }


def strategy_post_earnings_short_straddle(
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    earn_idx: int,
    r: float = 0.05,
) -> Optional[dict]:
    """
    SELL ATM straddle at T+0 close, buy back at T+5.
    Profit = IV crush (premium collected > buy-back cost).
    """
    idx_entry = earn_idx
    idx_exit  = offset_idx(dates, earn_idx, +5)
    if idx_exit is None or idx_entry >= idx_exit:
        return None

    spot_entry = prices.iloc[idx_entry]
    spot_exit  = prices.iloc[idx_exit]
    if pd.isna(spot_entry) or pd.isna(spot_exit) or spot_entry <= 0 or spot_exit <= 0:
        return None

    rv = realized_vol_at(prices, idx_entry, window=21)
    iv_entry = rv * 1.20   # still elevated morning of earnings
    iv_exit  = rv * 0.85   # post-crush

    T_entry = 10 / 365
    T_exit  =  5 / 365
    K = spot_entry   # fixed strike at entry spot

    price_entry = bs_straddle(spot_entry, K, T_entry, r, iv_entry)
    price_exit  = bs_straddle(spot_exit,  K, T_exit,  r, iv_exit)

    if price_entry <= 0:
        return None

    # Selling straddle: P&L is premium received minus buy-back cost
    pnl_pct = (price_entry - price_exit) / price_entry * 100.0

    return {
        "entry_date":  dates[idx_entry].date().isoformat(),
        "exit_date":   dates[idx_exit].date().isoformat(),
        "entry_price": round(price_entry, 4),
        "exit_price":  round(price_exit,  4),
        "spot_entry":  round(spot_entry,  2),
        "iv_entry":    round(iv_entry,    4),
        "pnl_pct":     round(pnl_pct,     4),
        "won":         int(pnl_pct > 0),
    }


def strategy_post_earnings_drift(
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    earn_idx: int,
    surprise_pct: float,
    min_surprise: float = 2.0,
    hold_days: int = 10,
) -> Optional[dict]:
    """
    PEAD (Post-Earnings Announcement Drift): directional stock trade.
    Long if surprise_pct > +min_surprise, Short if < -min_surprise.
    Entry T+1 open ≈ T+1 close (daily data), exit at T+hold_days.
    """
    if pd.isna(surprise_pct):
        return None
    if abs(surprise_pct) < min_surprise:
        return None

    direction = 1 if surprise_pct > 0 else -1

    idx_entry = offset_idx(dates, earn_idx, +1)
    idx_exit  = offset_idx(dates, earn_idx, +hold_days)
    if idx_entry is None or idx_exit is None or idx_entry >= idx_exit:
        return None

    price_entry = prices.iloc[idx_entry]
    price_exit  = prices.iloc[idx_exit]
    if pd.isna(price_entry) or pd.isna(price_exit) or price_entry <= 0:
        return None

    pnl_pct = (price_exit - price_entry) / price_entry * direction * 100.0

    return {
        "entry_date":    dates[idx_entry].date().isoformat(),
        "exit_date":     dates[idx_exit].date().isoformat(),
        "entry_price":   round(price_entry, 2),
        "exit_price":    round(price_exit,  2),
        "direction":     direction,
        "surprise_pct":  round(surprise_pct, 2),
        "hold_days":     hold_days,
        "pnl_pct":       round(pnl_pct, 4),
        "won":           int(pnl_pct > 0),
    }

# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    prices_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    sector_map: Dict[str, str],
    min_surprise: float = 2.0,
) -> pd.DataFrame:
    """
    Iterate over all earnings events and all strategies.
    Returns a DataFrame of trade records.
    """
    records: List[dict] = []
    dates = prices_df.index
    available_tickers = set(prices_df.columns)

    total_events   = len(earnings_df)
    processed      = 0
    skipped_nodata = 0
    skipped_nodate = 0

    print(f"\n[Backtest] Running {total_events} earnings events × 3 strategies ...")

    for _, row in earnings_df.iterrows():
        ticker      = row["ticker"]
        earn_date   = pd.Timestamp(row["earnings_date"])
        surprise    = row.get("surprise_pct", np.nan)
        sector      = sector_map.get(ticker, "Unknown")

        processed += 1
        if processed % 50 == 0:
            print(f"  ... {processed}/{total_events} events processed "
                  f"({len(records)} trades so far)")

        # Must have price data for this ticker
        if ticker not in available_tickers:
            skipped_nodata += 1
            continue

        ticker_prices = prices_df[ticker].dropna()
        if len(ticker_prices) < 30:
            skipped_nodata += 1
            continue

        # Re-index on full dates for positional access
        ticker_series = prices_df[ticker]

        # Find earnings date index in the full date index
        earn_idx = find_date_idx(dates, earn_date)
        if earn_idx < 0 or earn_idx >= len(dates):
            skipped_nodate += 1
            continue

        # Require at least 25 prior days (for realized vol calc)
        if earn_idx < 25:
            skipped_nodate += 1
            continue

        base_record = {
            "ticker":       ticker,
            "earnings_date": earn_date.date().isoformat(),
            "sector":       sector,
            "surprise_pct": round(float(surprise), 2) if not pd.isna(surprise) else None,
        }

        # ---- Strategy 1: Pre-earnings straddle ----
        res1 = strategy_pre_earnings_straddle(ticker_series, dates, earn_idx)
        if res1 is not None:
            rec = {**base_record, "strategy": "Pre_Earnings_Straddle", **res1}
            records.append(rec)

        # ---- Strategy 2: Post-earnings short straddle ----
        res2 = strategy_post_earnings_short_straddle(ticker_series, dates, earn_idx)
        if res2 is not None:
            rec = {**base_record, "strategy": "Post_Earnings_Short_Straddle", **res2}
            records.append(rec)

        # ---- Strategy 3a: Drift, 10-day hold ----
        res3a = strategy_post_earnings_drift(
            ticker_series, dates, earn_idx, surprise,
            min_surprise=min_surprise, hold_days=10)
        if res3a is not None:
            rec = {**base_record, "strategy": "Post_Earnings_Drift_10d", **res3a}
            records.append(rec)

        # ---- Strategy 3b: Drift, 21-day hold ----
        res3b = strategy_post_earnings_drift(
            ticker_series, dates, earn_idx, surprise,
            min_surprise=min_surprise, hold_days=21)
        if res3b is not None:
            rec = {**base_record, "strategy": "Post_Earnings_Drift_21d", **res3b}
            records.append(rec)

    print(f"[Backtest] Complete: {len(records)} trade records generated.")
    print(f"           Skipped (no price data): {skipped_nodata}")
    print(f"           Skipped (date out of range): {skipped_nodate}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Sort for clean output
    df = df.sort_values(["strategy", "ticker", "earnings_date"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _stats(group: pd.Series) -> dict:
    """Compute summary stats for a series of P&L percentages."""
    n = len(group)
    if n == 0:
        return {"n": 0, "win_rate": np.nan, "mean_pnl": np.nan,
                "median_pnl": np.nan, "std_pnl": np.nan, "sharpe": np.nan}
    won = group > 0
    mean  = group.mean()
    std   = group.std(ddof=1) if n > 1 else 0.0
    return {
        "n":          n,
        "win_rate":   round(won.mean() * 100, 1),
        "mean_pnl":   round(mean, 2),
        "median_pnl": round(group.median(), 2),
        "std_pnl":    round(std, 2),
        "sharpe":     round(mean / std, 3) if std > 0 else np.nan,
    }


def _best_worst(df_strat: pd.DataFrame) -> Tuple[str, str]:
    """Return (best_event_str, worst_event_str) for display."""
    if df_strat.empty:
        return "N/A", "N/A"
    best_row  = df_strat.loc[df_strat["pnl_pct"].idxmax()]
    worst_row = df_strat.loc[df_strat["pnl_pct"].idxmin()]

    def _fmt(r: pd.Series) -> str:
        return (f"{r['ticker']} on {r['earnings_date']} "
                f"({r['pnl_pct']:+.1f}%)")

    return _fmt(best_row), _fmt(worst_row)


def _surprise_bin(surprise_pct) -> str:
    if pd.isna(surprise_pct):
        return "unknown"
    v = abs(float(surprise_pct))
    if v >= 5:
        return "large (≥5%)"
    if v >= 2:
        return "medium (2-5%)"
    return "small (<2%)"


def build_analysis(results_df: pd.DataFrame) -> str:
    """Build the full Markdown report string."""
    lines: List[str] = []

    lines.append("# Canyon v9 — Step 91: Earnings Event Backtest Report")
    lines.append("")
    lines.append(f"**Total trade records:** {len(results_df)}")
    lines.append(f"**Tickers covered:** {results_df['ticker'].nunique()}")
    lines.append(f"**Earnings events covered:** "
                 f"{results_df[['ticker','earnings_date']].drop_duplicates().shape[0]}")
    lines.append("")

    strategies = results_df["strategy"].unique().tolist()

    # ------------------------------------------------------------------ #
    # Per-strategy summary
    # ------------------------------------------------------------------ #
    lines.append("## 1. Strategy Summary")
    lines.append("")

    for strat in strategies:
        sub = results_df[results_df["strategy"] == strat]
        st  = _stats(sub["pnl_pct"])
        best, worst = _best_worst(sub)

        lines.append(f"### {strat}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Events (N) | {st['n']} |")
        lines.append(f"| Win Rate | {st['win_rate']}% |")
        lines.append(f"| Mean P&L | {st['mean_pnl']:+.2f}% |")
        lines.append(f"| Median P&L | {st['median_pnl']:+.2f}% |")
        lines.append(f"| Std Dev P&L | {st['std_pnl']:.2f}% |")
        lines.append(f"| Sharpe (P&L) | {st['sharpe']} |")
        lines.append(f"| Best event | {best} |")
        lines.append(f"| Worst event | {worst} |")
        lines.append("")

    # ------------------------------------------------------------------ #
    # By sector
    # ------------------------------------------------------------------ #
    lines.append("## 2. Results by Sector")
    lines.append("")

    for strat in strategies:
        sub = results_df[results_df["strategy"] == strat]
        if sub.empty:
            continue
        lines.append(f"### {strat} — Sector Breakdown")
        lines.append("")
        lines.append("| Sector | N | Win Rate | Mean P&L | Sharpe |")
        lines.append("|--------|---|----------|----------|--------|")
        for sector, grp in sub.groupby("sector"):
            st = _stats(grp["pnl_pct"])
            lines.append(f"| {sector} | {st['n']} | {st['win_rate']}% | "
                         f"{st['mean_pnl']:+.2f}% | {st['sharpe']} |")
        lines.append("")

    # ------------------------------------------------------------------ #
    # By surprise magnitude
    # ------------------------------------------------------------------ #
    lines.append("## 3. Results by Earnings Surprise Magnitude")
    lines.append("")

    results_df = results_df.copy()
    results_df["surprise_bin"] = results_df["surprise_pct"].apply(_surprise_bin)

    for strat in strategies:
        sub = results_df[results_df["strategy"] == strat]
        if sub.empty:
            continue
        lines.append(f"### {strat} — Surprise Magnitude Breakdown")
        lines.append("")
        lines.append("| Surprise Bin | N | Win Rate | Mean P&L | Sharpe |")
        lines.append("|-------------|---|----------|----------|--------|")
        for bin_name, grp in sub.groupby("surprise_bin"):
            st = _stats(grp["pnl_pct"])
            lines.append(f"| {bin_name} | {st['n']} | {st['win_rate']}% | "
                         f"{st['mean_pnl']:+.2f}% | {st['sharpe']} |")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Pre vs Post earnings comparison
    # ------------------------------------------------------------------ #
    lines.append("## 4. Pre vs Post-Earnings Comparison")
    lines.append("")
    lines.append("| Group | N | Win Rate | Mean P&L | Sharpe |")
    lines.append("|-------|---|----------|----------|--------|")

    pre_mask  = results_df["strategy"].str.startswith("Pre_")
    post_mask = results_df["strategy"].str.startswith("Post_")

    for label, mask in [("Pre-earnings strategies", pre_mask),
                        ("Post-earnings strategies", post_mask)]:
        sub = results_df[mask]
        st  = _stats(sub["pnl_pct"])
        lines.append(f"| {label} | {st['n']} | {st['win_rate']}% | "
                     f"{st['mean_pnl']:+.2f}% | {st['sharpe']} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Drift: does larger surprise → better drift?
    # ------------------------------------------------------------------ #
    drift_df = results_df[results_df["strategy"].isin(
        ["Post_Earnings_Drift_10d", "Post_Earnings_Drift_21d"])]
    if not drift_df.empty and drift_df["surprise_pct"].notna().any():
        lines.append("## 5. PEAD Drift: Surprise Size vs Return")
        lines.append("")
        lines.append("*Does a larger earnings surprise predict a bigger drift?*")
        lines.append("")
        lines.append("| Hold | Surprise Bin | N | Win Rate | Mean P&L |")
        lines.append("|------|-------------|---|----------|----------|")
        for strat in ["Post_Earnings_Drift_10d", "Post_Earnings_Drift_21d"]:
            sub = drift_df[drift_df["strategy"] == strat]
            for bin_name, grp in sub.groupby("surprise_bin"):
                st = _stats(grp["pnl_pct"])
                hold_label = "10d" if "10d" in strat else "21d"
                lines.append(f"| {hold_label} | {bin_name} | {st['n']} | "
                             f"{st['win_rate']}% | {st['mean_pnl']:+.2f}% |")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Top / Bottom 10 individual trades
    # ------------------------------------------------------------------ #
    lines.append("## 6. Top 10 Individual Trades")
    lines.append("")
    top10 = results_df.nlargest(10, "pnl_pct")[
        ["ticker", "earnings_date", "strategy", "sector", "surprise_pct", "pnl_pct"]]
    lines.append(top10.to_markdown(index=False))
    lines.append("")

    lines.append("## 7. Bottom 10 Individual Trades")
    lines.append("")
    bot10 = results_df.nsmallest(10, "pnl_pct")[
        ["ticker", "earnings_date", "strategy", "sector", "surprise_pct", "pnl_pct"]]
    lines.append(bot10.to_markdown(index=False))
    lines.append("")

    # ------------------------------------------------------------------ #
    # Methodology notes
    # ------------------------------------------------------------------ #
    lines.append("## 8. Methodology Notes")
    lines.append("")
    lines.append("- **IV approximation**: realised 21-day volatility × earnings premium factor.")
    lines.append("  Pre-earnings IV = RV × 1.30; Earnings-day sell IV = RV × 1.20; "
                 "Post-crush IV = RV × 0.85.")
    lines.append("- **Black-Scholes straddle** used for option pricing (call + put).")
    lines.append("- **Strategy 1 (Pre-straddle)**: Entry T-5, exit T-1 before earnings close.")
    lines.append("- **Strategy 2 (Short straddle)**: Sell at T+0 close (IV elevated), "
                 "buy back T+5 (post-crush).")
    lines.append("- **Strategy 3 (PEAD drift)**: Directional stock trade T+1 → T+10 / T+21; "
                 "skipped if |surprise_pct| < threshold.")
    lines.append("- Prices from `sp500_price_cache.csv`; daily close used for all entries/exits.")
    lines.append("- Results are approximations — no transaction costs, slippage, or pin risk modelled.")
    lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canyon v9 Step 91 — Earnings Event Strategy Backtest")
    p.add_argument("--years",        type=int,   default=2,
                   help="How many years of earnings history to backtest (default: 2)")
    p.add_argument("--min-surprise", type=float, default=2.0,
                   help="Min |surprise_pct| for drift strategy (default: 2.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 66)
    print("Canyon v9 — Step 91: Earnings Event Strategy Backtest")
    print("=" * 66)
    print(f"  Years lookback : {args.years}")
    print(f"  Min surprise   : {args.min_surprise}%")
    print(f"  Output dir     : {ROOT}")
    print()

    # ---- Load data ----
    prices_df = load_prices()
    if prices_df is None:
        print("[FATAL] Cannot proceed without price data.")
        print(f"        Expected: {PRICE_FILE}")
        sys.exit(1)

    earnings_df, source = load_earnings_calendar(years=args.years)
    if earnings_df.empty:
        print()
        print("[WARNING] No earnings data found.")
        print("  Looked for:")
        print(f"    {SURPRISE_FILE}")
        print(f"    {REVISION_FILE}")
        print()
        print("  To generate earnings data, run one of these prior steps:")
        print("    python canyon_final_v9_step81_earnings_surprise.py")
        print("    python canyon_final_v9_step80_earnings_revision.py")
        print()
        print("  Required columns: ticker, earnings_date (or report_date/date), "
              "surprise_pct (optional)")
        sys.exit(0)

    sector_map = load_sector_map()
    if not sector_map:
        print("[Sectors] No sector map found — sector analysis will show 'Unknown'")
    else:
        print(f"[Sectors] Loaded {len(sector_map)} ticker → sector mappings")

    # ---- Run backtest ----
    results_df = run_backtest(
        prices_df    = prices_df,
        earnings_df  = earnings_df,
        sector_map   = sector_map,
        min_surprise = args.min_surprise,
    )

    if results_df.empty:
        print("\n[WARNING] No trade records generated.")
        print("  Possible causes:")
        print("  - Earnings dates are outside the price data range")
        print("  - Tickers in earnings data are not in price cache")
        print(f"  Price range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
        sys.exit(0)

    # ---- Write results CSV ----
    results_df.to_csv(OUT_RESULTS, index=False)
    print(f"\n[Output] Results CSV   → {OUT_RESULTS}")
    print(f"         {len(results_df)} rows × {len(results_df.columns)} columns")

    # ---- Build and write report ----
    report_md = build_analysis(results_df)
    OUT_REPORT.write_text(report_md, encoding="utf-8")
    print(f"[Output] Report MD     → {OUT_REPORT}")

    # ---- Console summary ----
    print()
    print("=" * 66)
    print("RESULTS SUMMARY")
    print("=" * 66)
    for strat in sorted(results_df["strategy"].unique()):
        sub = results_df[results_df["strategy"] == strat]
        st  = _stats(sub["pnl_pct"])
        print(f"  {strat:<38}  "
              f"N={st['n']:>4}  "
              f"WinRate={st['win_rate']:>5.1f}%  "
              f"MeanPnL={st['mean_pnl']:>+7.2f}%  "
              f"Sharpe={st['sharpe']}")
    print("=" * 66)
    print()


if __name__ == "__main__":
    main()
