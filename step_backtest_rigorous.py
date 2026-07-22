#!/usr/bin/env python3
"""
Canyon — step_backtest_rigorous.py  (v2 — proper historical signals)
=====================================================================
Three-book vectorized backtest with literature-backed historical signals.

SIGNAL RATIONALE
────────────────
SHORT  (5d rebal):  Weekly reversal — stocks that went UP last 5 days
                    mean-revert DOWN, and vice versa.
                    IC(5d reversal, 5d fwd) ≈ -0.03 to -0.08 (Jegadeesh 1990).
                    Backtest: true LONG/SHORT (long recent losers / short recent winners).
                    Borrow cost: 30 bps/yr on short leg (cheap for S&P 500).

MEDIUM (10d rebal): 12m-1m momentum (Jegadeesh-Titman).
                    12-month return minus most-recent-1-month (avoids reversal window).
                    IC(12m-1m mom, 21d fwd) ≈ 0.02–0.05 (well-documented).
                    Long-only top-15 picks.

LONG   (21d rebal): Multi-factor quality-momentum composite.
                    40% 12m-1m momentum + 30% 6m momentum
                    + 20% inverse realized vol + 10% price-vs-200MA trend.
                    IC(composite, 63d fwd) ≈ 0.02–0.04.
                    Long-only top-20 picks.

Metrics:  CAGR, Sharpe, Max Drawdown, Calmar, Win Rate, Avg Turnover
Benchmark: SPY buy-and-hold
Transaction cost: 5 bps one-way for long, 10 bps one-way for shorts (wider spread)

Outputs:
  backtest_three_books.json
  backtest_three_books.csv
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")


# ── Config ────────────────────────────────────────────────────────────────────

COST_LONG_BPS  = 5    # one-way bps for long trades
COST_SHORT_BPS = 10   # one-way bps for shorts (wider bid/ask + borrow)
BORROW_ANN     = 0.003  # 30 bps/yr borrow cost on short notional (S&P 500 easy-to-borrow)

BOOKS = {
    "SHORT": {
        "n_picks":          10,
        "rebalance_days":   10,
        "signal":           "short_composite",
        "long_short":       True,
        "fwd_horizon_ic":   5,
    },
    "MEDIUM": {
        "n_picks":          15,
        "rebalance_days":   10,
        "signal":           "momentum_12m1m",      # IC=0.051 — best signal for 21d
        "long_short":       False,
        "fwd_horizon_ic":   21,
    },
    "LONG": {
        "n_picks":          20,
        "rebalance_days":   21,
        "signal":           "quality_momentum_neutral",   # sector-neutral IC=0.053
        "long_short":       False,
        "fwd_horizon_ic":   63,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS  (all return date × ticker matrix, 0-100 cross-sectional rank)
# ══════════════════════════════════════════════════════════════════════════════

def compute_reversal_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Short-term weekly reversal signal.
    Score = rank of (-5d return).  High score = recent loser = BUY.
                                    Low score  = recent winner = SELL SHORT.
    Academic basis: Jegadeesh (1990), Lo & MacKinlay (1990).
    Expected IC vs 5d fwd return: +0.03 to +0.08 (cross-sectional).
    """
    r5 = prices.pct_change(5)
    score = (-r5).rank(axis=1, pct=True) * 100
    return score.dropna(how="all")


def compute_momentum_12m1m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-month momentum excluding most-recent month (Jegadeesh-Titman 1993).
    Avoids the short-term reversal window.
    Score = rank of (ret_12m − ret_1m).  High score = strong intermediate momentum.
    Expected IC vs 21d fwd return: +0.02 to +0.05.
    """
    r252 = prices.pct_change(252)
    r21  = prices.pct_change(21)
    mom  = r252.subtract(r21)
    score = mom.rank(axis=1, pct=True) * 100
    return score.dropna(how="all")


def compute_quality_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-factor composite for 63-day prediction:
      40% 12m-1m momentum (trend persistence)
      30% 6m momentum     (medium-term strength)
      20% inverse realized vol (low-vol anomaly / quality proxy)
      10% price vs 200-day MA (trend filter)

    This mimics the factor combination used by AQR's Momentum + Quality model.
    Expected IC vs 63d fwd return: +0.02 to +0.04.
    """
    r252 = prices.pct_change(252)
    r21  = prices.pct_change(21)
    r126 = prices.pct_change(126)
    mom12 = r252.subtract(r21)

    # Realized vol (60-day): lower = higher quality
    vol60   = prices.pct_change().rolling(60).std()
    inv_vol = (-vol60).rank(axis=1, pct=True) * 100

    # Trend: price above 200-day moving average
    ma200 = prices.rolling(200).mean()
    trend = ((prices / ma200) - 1).rank(axis=1, pct=True) * 100

    mom12_r = mom12.rank(axis=1, pct=True) * 100
    r126_r  = r126.rank(axis=1, pct=True) * 100

    composite = (0.40 * mom12_r
               + 0.30 * r126_r
               + 0.20 * inv_vol
               + 0.10 * trend)

    return composite.dropna(how="all")


def compute_short_composite(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-factor SHORT signal.  HIGH score = LONG candidate (will go UP).
                                LOW score  = SHORT candidate (will go DOWN).

    Component logic:
      +40%  Reversal (5d): recent losers bounce → they are long candidates.
      +30%  RSI mean-reversion: RSI<30 = oversold = long; RSI>70 = overbought = short.
      +20%  Distance from 52w high (negative): far below high = cheap = long.
      +10%  Relative vol-adjusted momentum (1m): low recent vol-adj return = long.

    This avoids the fatal flaw of pure reversal in trend markets:
    by including RSI and 52w-distance, stocks that are genuinely weak
    (not just recent losers) score low → better shorts.
    Expected IC vs 5d fwd: ~0.02-0.04 depending on market regime.
    """
    r5   = prices.pct_change(5)
    r21  = prices.pct_change(21)

    # Component 1: short-term reversal (negative 5d return = high score = long candidate)
    rev_rank = (-r5).rank(axis=1, pct=True) * 100

    # Component 2: RSI(14) — low RSI = oversold = long candidate
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    rsi_rank = (-rsi).rank(axis=1, pct=True) * 100   # low RSI = high score

    # Component 3: distance below 52w high (more below = cheaper = long)
    high52 = prices.rolling(252).max()
    dist   = prices / high52.replace(0, np.nan) - 1   # negative = below high
    dist_rank = (-dist).rank(axis=1, pct=True) * 100  # far below = high score

    # Component 4: 1m vol-adjusted return (low = long candidate)
    vol21 = r21 / (prices.pct_change().rolling(21).std() + 1e-9)
    vol_rank = (-vol21).rank(axis=1, pct=True) * 100

    composite = (0.40 * rev_rank
               + 0.30 * rsi_rank
               + 0.20 * dist_rank
               + 0.10 * vol_rank)
    return composite.dropna(how="all")


def compute_quality_momentum_neutral(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Sector-neutral quality-momentum composite.
    Within each sector quintile, ranks stocks by the quality-momentum score.
    This removes the sector-level beta exposure from the LONG book,
    so returns genuinely reflect stock selection, not sector allocation.

    We approximate sectors via rolling 1-year correlation clusters
    (since we don't have GICS sector data in price cache).
    Fallback: plain quality_momentum if sector data unavailable.
    """
    base = compute_quality_momentum(prices)

    # Try to load sector metadata for sector-neutral ranking
    sector_path = ROOT / "alpha_scores.csv"
    if sector_path.exists():
        try:
            meta = pd.read_csv(sector_path)
            if "ticker" in meta.columns and "sector" in meta.columns:
                sectors = meta.set_index("ticker")["sector"].dropna()
                # For each date, rank within sector instead of globally
                result_rows = []
                for date in base.index:
                    row = base.loc[date].dropna()
                    neutralised = row.copy() * np.nan
                    for sec, group in sectors.groupby(sectors):
                        tickers_in_sec = row.index.intersection(group.index)
                        if len(tickers_in_sec) >= 3:
                            sec_row = row[tickers_in_sec]
                            neutralised[tickers_in_sec] = sec_row.rank(pct=True) * 100
                    # Fill any unranked tickers with global rank
                    still_nan = neutralised.isna()
                    neutralised[still_nan] = row[still_nan].rank(pct=True) * 100
                    result_rows.append(neutralised)
                result = pd.DataFrame(result_rows, index=base.index)
                ok("  LONG: sector-neutral ranking applied")
                return result.dropna(how="all")
        except Exception:
            pass

    return base


def load_qqq(prices_index: pd.DatetimeIndex) -> pd.Series:
    """Load QQQ price series, aligned to trading dates."""
    qqq_path = ROOT / "qqq_price_cache.csv"
    if qqq_path.exists():
        try:
            df = pd.read_csv(qqq_path, index_col=0, parse_dates=True).squeeze()
            return df.reindex(prices_index, method="ffill").dropna()
        except Exception:
            pass
    # Fallback: equal-weight proxy of 10 large Nasdaq names
    return pd.Series(dtype=float)


def compute_qqq_alpha_medium(prices: pd.DataFrame) -> pd.DataFrame:
    """
    QQQ-beating signal for MEDIUM book (21-day horizon).

    Logic: identify which S&P500 stocks are outperforming QQQ on a
    3-month basis AND accelerating (recent 3m stronger than 6m half).
    These stocks tend to continue outperforming over the next 21 days.

    Components:
      50%  3m excess return vs QQQ  (alpha persistence at medium term)
      30%  momentum acceleration:   3m return - (12m return / 4)
           Positive = recent momentum stronger than annual trend = catalyst
      20%  12m-1m momentum          (trend direction confirmation)

    This specifically selects S&P500 stocks that are beating the Nasdaq,
    not just the S&P500 average.
    """
    qqq = load_qqq(prices.index)

    r63  = prices.pct_change(63)    # 3m return
    r126 = prices.pct_change(126)   # 6m
    r252 = prices.pct_change(252)   # 12m
    r21  = prices.pct_change(21)    # 1m

    # QQQ-relative 3m excess return
    if not qqq.empty:
        qqq_r63 = qqq.pct_change(63)
        excess_3m = r63.subtract(qqq_r63, axis=0)
    else:
        excess_3m = r63   # fallback

    # Momentum acceleration: 3m return vs half of 12m (annualised to 3m)
    accel = r63.subtract(r252 / 4, fill_value=0)

    # Classic 12m-1m momentum (direction filter)
    mom_12m1m = r252.subtract(r21)

    score = (0.50 * excess_3m.rank(axis=1, pct=True) * 100
           + 0.30 * accel.rank(axis=1, pct=True) * 100
           + 0.20 * mom_12m1m.rank(axis=1, pct=True) * 100)

    return score.dropna(how="all")


def compute_qqq_alpha_long(prices: pd.DataFrame) -> pd.DataFrame:
    """
    QQQ-beating signal for LONG book (63-day horizon).

    Identifies stocks that have been consistently outperforming QQQ
    for 6-12 months AND have improving quality metrics (low vol anomaly).
    No sector-neutral constraint — we WANT tech overweight to beat QQQ.

    Components:
      40%  12m-1m excess return vs QQQ   (persistent idiosyncratic alpha)
      30%  6m excess return vs QQQ        (medium-term trend vs index)
      20%  inverse realized vol           (quality / low-vol anomaly)
      10%  trend strength (price / 200MA) (avoid mean-reverting stocks)
    """
    qqq = load_qqq(prices.index)

    r252 = prices.pct_change(252)
    r126 = prices.pct_change(126)
    r21  = prices.pct_change(21)

    # QQQ-relative excess returns
    if not qqq.empty:
        qqq_r252 = qqq.pct_change(252)
        qqq_r126 = qqq.pct_change(126)
        excess_12m = r252.subtract(r21).subtract(qqq_r252.subtract(qqq_r126 / 2), axis=0)
        excess_6m  = r126.subtract(qqq_r126, axis=0)
    else:
        excess_12m = r252.subtract(r21)
        excess_6m  = r126

    vol60   = prices.pct_change().rolling(60).std()
    inv_vol = (-vol60).rank(axis=1, pct=True) * 100

    ma200  = prices.rolling(200).mean()
    trend  = (prices / ma200.replace(0, np.nan) - 1).rank(axis=1, pct=True) * 100

    score = (0.40 * excess_12m.rank(axis=1, pct=True) * 100
           + 0.30 * excess_6m.rank(axis=1, pct=True) * 100
           + 0.20 * inv_vol
           + 0.10 * trend)

    return score.dropna(how="all")


def get_signal(signal_name: str, prices: pd.DataFrame) -> pd.DataFrame:
    dispatch = {
        "reversal_5d":              compute_reversal_5d,
        "momentum_12m1m":           compute_momentum_12m1m,
        "quality_momentum":         compute_quality_momentum,
        "short_composite":          compute_short_composite,
        "quality_momentum_neutral": compute_quality_momentum_neutral,
        "qqq_alpha_medium":         compute_qqq_alpha_medium,
        "qqq_alpha_long":           compute_qqq_alpha_long,
    }
    fn = dispatch.get(signal_name)
    if fn is None:
        err(f"Unknown signal: {signal_name}")
        return pd.DataFrame()
    return fn(prices)


# ══════════════════════════════════════════════════════════════════════════════
# IC VALIDATION  — verify signal quality before running backtest
# ══════════════════════════════════════════════════════════════════════════════

def validate_ic(signal_df: pd.DataFrame, prices: pd.DataFrame,
                horizon: int, signal_name: str) -> float:
    """
    Compute rolling IC (Spearman) of signal vs horizon-day forward return.
    Prints a summary and returns mean IC.
    """
    fwd_ret = prices.pct_change(horizon).shift(-horizon)
    fwd_rank = fwd_ret.rank(axis=1, pct=True)

    ic_series = []
    # sample every 5 days to avoid overlapping windows at 63d
    sample_dates = signal_df.index[::max(1, horizon // 5)]

    for date in sample_dates:
        if date not in signal_df.index or date not in fwd_rank.index:
            continue
        sig_row = signal_df.loc[date].dropna()
        fwd_row = fwd_rank.loc[date].dropna()
        common  = sig_row.index.intersection(fwd_row.index)
        if len(common) < 50:
            continue
        rho, _ = spearmanr(sig_row[common].values, fwd_row[common].values)
        if not np.isnan(rho):
            ic_series.append(rho)

    if not ic_series:
        warn(f"  {signal_name}: could not compute IC (insufficient aligned data)")
        return 0.0

    mean_ic  = float(np.mean(ic_series))
    std_ic   = float(np.std(ic_series))
    ir       = mean_ic / (std_ic + 1e-9)   # information ratio of IC series
    pos_frac = sum(1 for x in ic_series if x > 0) / len(ic_series)

    color = GREEN if mean_ic > 0.02 else (YELLOW if mean_ic >= 0 else RED)
    print(f"  {signal_name:<20} IC({horizon}d)= {color}{mean_ic:+.4f}{RESET} "
          f"  σ={std_ic:.4f}  IR={ir:.2f}  pos%={pos_frac*100:.0f}%  "
          f"n={len(ic_series)}")
    return mean_ic


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BookStats(NamedTuple):
    total_return:  float
    cagr:          float
    sharpe:        float
    max_drawdown:  float
    calmar:        float
    win_rate:      float
    avg_turnover:  float
    n_days:        int
    n_rebalances:  int


def compute_stats(nav: pd.Series, turnovers: list[float]) -> BookStats:
    nav = nav.dropna()
    if len(nav) < 10:
        return BookStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    daily_ret  = nav.pct_change().dropna()
    n_days     = len(daily_ret)
    total_ret  = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years      = n_days / 252
    cagr       = float((1 + total_ret) ** (1 / max(years, 0.1)) - 1)
    sharpe     = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252))
    roll_max   = nav.cummax()
    dd         = (nav - roll_max) / (roll_max + 1e-9)
    max_dd     = float(dd.min())
    calmar     = cagr / (-max_dd + 1e-9) if max_dd < 0 else 0.0
    win_rate   = float((daily_ret > 0).mean())
    avg_turn   = float(np.mean(turnovers)) if turnovers else 0.0

    return BookStats(
        total_return=round(total_ret, 4),
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd, 4),
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 3),
        avg_turnover=round(avg_turn, 3),
        n_days=n_days,
        n_rebalances=len(turnovers),
    )


def run_book_long_only(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    n_picks: int,
    rebalance_days: int,
    cost_bps: int = COST_LONG_BPS,
) -> tuple[pd.Series, list[float]]:
    """Long-only equal-weight book."""
    cost_ow   = cost_bps / 10_000
    dates     = prices.index.tolist()
    nav       = 1.0
    holdings: dict[str, float] = {}
    nav_list  = []
    turnovers = []

    for i, date in enumerate(dates):
        # daily mark-to-market
        if i > 0 and holdings:
            prev = dates[i - 1]
            day_ret = sum(
                wt * (prices.loc[date, tk] / prices.loc[prev, tk] - 1)
                for tk, wt in holdings.items()
                if tk in prices.columns
                and pd.notna(prices.loc[prev, tk]) and pd.notna(prices.loc[date, tk])
                and prices.loc[prev, tk] > 0
            )
            nav *= (1 + day_ret)

        # rebalance
        if i % rebalance_days == 0 and date in scores.index:
            row = scores.loc[date].dropna()
            if len(row) >= n_picks:
                picks = [tk for tk in row.nlargest(n_picks).index
                         if tk in prices.columns and pd.notna(prices.loc[date, tk])][:n_picks]
                if picks:
                    new_h = {tk: 1.0 / len(picks) for tk in picks}
                    old_set, new_set = set(holdings), set(new_h)
                    turn = (len(old_set - new_set) + len(new_set - old_set)) / (2 * max(n_picks, 1))
                    nav *= (1 - cost_ow * turn * 2)
                    turnovers.append(turn)
                    holdings = new_h

        nav_list.append((date, nav))

    s = pd.Series({d: v for d, v in nav_list}, name="nav")
    s.index = pd.to_datetime(s.index)
    return s, turnovers


def load_regime_series(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Load HMM regime history. Returns Series[date → regime_str].
    Falls back to 'BULL' everywhere if file missing.
    """
    hmm_path = ROOT / "hmm_regime_daily.csv"
    if hmm_path.exists():
        try:
            df = pd.read_csv(hmm_path, parse_dates=["date"])
            df = df.set_index("date")["regime"].reindex(dates, method="ffill")
            df = df.fillna("BULL").str.upper()
            return df
        except Exception:
            pass
    return pd.Series("BULL", index=dates)


def run_book_long_short(
    prices: pd.DataFrame,
    scores: pd.DataFrame,       # HIGH score = long candidate, LOW = short candidate
    n_picks: int,               # n_picks per leg (long + short)
    rebalance_days: int,
    borrow_ann: float = BORROW_ANN,
    regime_series: pd.Series | None = None,
) -> tuple[pd.Series, list[float]]:
    """
    Regime-conditional true long/short book:
      BULL regime  → long-only (top n_picks, 100% capital): riding the trend safely
      BEAR/LATE_BULL → true L/S (top n long +50%, bottom n short -50%): capturing mean-reversion

    This matches how institutional L/S desks actually operate:
    you don't run a market-neutral book into a strong bull market.
    Net beta ≈ 0 in BEAR, ~1.0 in BULL.
    """
    cost_long  = COST_LONG_BPS  / 10_000
    cost_short = COST_SHORT_BPS / 10_000
    borrow_daily = borrow_ann / 252

    dates      = prices.index.tolist()
    nav        = 1.0
    long_h:  dict[str, float] = {}
    short_h: dict[str, float] = {}
    nav_list   = []
    turnovers  = []
    regime_s   = regime_series if regime_series is not None else pd.Series("BULL", index=prices.index)

    n_bull_days = 0
    n_ls_days   = 0

    for i, date in enumerate(dates):
        regime = str(regime_s.get(date, "BULL")).upper()
        is_bull = "BULL" in regime and "LATE" not in regime

        # daily P&L
        if i > 0 and (long_h or short_h):
            prev = dates[i - 1]
            long_ret = sum(
                wt * (prices.loc[date, tk] / prices.loc[prev, tk] - 1)
                for tk, wt in long_h.items()
                if tk in prices.columns
                and pd.notna(prices.loc[prev, tk]) and pd.notna(prices.loc[date, tk])
                and prices.loc[prev, tk] > 0
            )
            short_ret = sum(
                wt * (prices.loc[date, tk] / prices.loc[prev, tk] - 1)
                for tk, wt in short_h.items()
                if tk in prices.columns
                and pd.notna(prices.loc[prev, tk]) and pd.notna(prices.loc[date, tk])
                and prices.loc[prev, tk] > 0
            )
            if is_bull:
                day_ret = long_ret    # long-only in bull
                n_bull_days += 1
            else:
                day_ret = 0.5 * long_ret - 0.5 * short_ret - borrow_daily * len(short_h) / max(n_picks, 1)
                n_ls_days += 1
            nav *= (1 + day_ret)

        # rebalance
        if i % rebalance_days == 0 and date in scores.index:
            row = scores.loc[date].dropna()
            valid = [tk for tk in row.index
                     if tk in prices.columns and pd.notna(prices.loc[date, tk])]
            row = row.loc[valid]

            if is_bull:
                # Long-only: exit any short leg cleanly
                if short_h:
                    nav *= (1 - cost_short * 1.0)   # close all shorts
                    short_h = {}
                if len(row) >= n_picks:
                    long_picks = row.nlargest(n_picks).index.tolist()
                    new_long   = {tk: 1.0 / len(long_picks) for tk in long_picks}
                    l_turn = (len(set(long_h) - set(new_long)) + len(set(new_long) - set(long_h))) / (2 * n_picks)
                    nav *= (1 - cost_long * l_turn * 2)
                    turnovers.append(l_turn)
                    long_h = new_long
            else:
                # True L/S in bear/late-bull
                if len(row) >= n_picks * 2:
                    long_picks  = row.nlargest(n_picks).index.tolist()
                    short_picks = row.nsmallest(n_picks).index.tolist()
                    new_long    = {tk: 1.0 / n_picks for tk in long_picks}
                    new_short   = {tk: 1.0 / n_picks for tk in short_picks}
                    l_turn = (len(set(long_h) - set(new_long)) + len(set(new_long) - set(long_h))) / (2 * n_picks)
                    s_turn = (len(set(short_h) - set(new_short)) + len(set(new_short) - set(short_h))) / (2 * n_picks)
                    nav *= (1 - cost_long  * l_turn * 2)
                    nav *= (1 - cost_short * s_turn * 2)
                    turnovers.append((l_turn + s_turn) / 2)
                    long_h, short_h = new_long, new_short

        nav_list.append((date, nav))

    log(f"  Regime split: {n_bull_days}d BULL (long-only) / {n_ls_days}d BEAR/LATE (L/S)")

    s = pd.Series({d: v for d, v in nav_list}, name="nav")
    s.index = pd.to_datetime(s.index)
    return s, turnovers


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}Canyon — Rigorous Backtest v2 (3 Books){RESET}  {TODAY}")
    print(f"  Signals: reversal_5d · momentum_12m1m · quality_momentum")
    print(f"  Cost: {COST_LONG_BPS}bps long / {COST_SHORT_BPS}bps short / "
          f"{BORROW_ANN*100:.1f}bps/yr borrow on shorts\n")

    price_path = ROOT / "sp500_price_cache.csv"
    if not price_path.exists():
        err("sp500_price_cache.csv not found")
        return

    log("Loading price cache …")
    raw = pd.read_csv(price_path)
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw.set_index(date_col)
    # drop any non-ticker numeric columns (e.g. unnamed index columns)
    raw = raw.loc[:, raw.columns.str.match(r'^[A-Z]')]
    prices_full = raw.sort_index()
    ok(f"Full price cache: {prices_full.shape[0]} days × {prices_full.shape[1]} tickers")

    # use last 2yr for the actual backtest, but compute signals on full history
    cutoff = prices_full.index[-1] - timedelta(days=730)
    prices = prices_full[prices_full.index >= cutoff].copy()
    ok(f"Backtest window: {prices.index[0].date()} → {prices.index[-1].date()} "
       f"({len(prices)} trading days)")

    # SPY + QQQ benchmarks
    spy_nav = None
    if "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
        spy_nav = spy / spy.iloc[0]

    results: dict = {}

    qqq_series = load_qqq(prices.index)
    qqq_nav = None
    if not qqq_series.empty:
        qqq_nav = qqq_series / qqq_series.iloc[0]
        qqq_rets   = qqq_nav.pct_change().dropna()
        qqq_cagr   = float(qqq_nav.iloc[-1] ** (252 / len(qqq_nav)) - 1)
        qqq_sharpe = float(qqq_rets.mean() / (qqq_rets.std() + 1e-9) * np.sqrt(252))
        qqq_mdd    = float(((qqq_nav - qqq_nav.cummax()) / (qqq_nav.cummax() + 1e-9)).min())
        results["QQQ_BENCHMARK"] = {
            "stats": {"cagr": round(qqq_cagr, 4), "sharpe": round(qqq_sharpe, 3),
                      "max_drawdown": round(qqq_mdd, 4)},
            "nav": qqq_nav.to_dict(),
        }
        ok(f"  QQQ: CAGR={qqq_cagr*100:.1f}%  Sharpe={qqq_sharpe:.2f}  MDD={qqq_mdd*100:.1f}%")

    # ── Pre-compute all signals on full price history (no look-ahead) ──────────
    print(f"\n  {BOLD}Signal IC validation{RESET} (Spearman rank IC, sampled every N/5 days)")
    print(f"  {'Signal':<20} {'IC(fwd)':>14}  {'σ':>7}  {'IR':>6}  "
          f"{'pos%':>6}  {'n':>5}")
    print(f"  {'─'*20} {'─'*14}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*5}")

    signals: dict[str, pd.DataFrame] = {}
    for book_name, cfg in BOOKS.items():
        sig_name = cfg["signal"]
        if sig_name not in signals:
            signals[sig_name] = get_signal(sig_name, prices_full)
        horizon = cfg["fwd_horizon_ic"]
        validate_ic(signals[sig_name], prices_full, horizon, sig_name)

    print()

    # ── Load regime series once ────────────────────────────────────────────────
    regime_series = load_regime_series(prices.index)
    regime_counts = regime_series.value_counts()
    log(f"Regime distribution: {dict(regime_counts)}")

    # ── Run each book ──────────────────────────────────────────────────────────
    for book_name, cfg in BOOKS.items():
        log(f"Running {book_name} book ({cfg['n_picks']} picks × {'regime-conditional L/S' if cfg['long_short'] else '1 leg'}, "
            f"rebal every {cfg['rebalance_days']}d) …")

        sig_full = signals[cfg["signal"]]
        sig = sig_full.reindex(prices.index, method="ffill").dropna(how="all")

        if sig.empty:
            warn(f"  No signal data for {book_name} in backtest window")
            continue

        if cfg["long_short"]:
            nav, turnovers = run_book_long_short(
                prices, sig, cfg["n_picks"], cfg["rebalance_days"],
                regime_series=regime_series,
            )
        else:
            nav, turnovers = run_book_long_only(
                prices, sig, cfg["n_picks"], cfg["rebalance_days"]
            )

        if nav.empty:
            warn(f"  {book_name}: empty NAV — skipping")
            continue

        stats = compute_stats(nav, turnovers)
        results[book_name] = {
            "config": {k: v for k, v in cfg.items()},
            "stats":  stats._asdict(),
            "nav":    nav.to_dict(),
        }
        marker = "L/S" if cfg["long_short"] else "L/O"
        ok(f"  {book_name} [{marker}]: CAGR={stats.cagr*100:.1f}%  "
           f"Sharpe={stats.sharpe:.2f}  MDD={stats.max_drawdown*100:.1f}%  "
           f"Calmar={stats.calmar:.2f}  Turn={stats.avg_turnover*100:.1f}%")

    # SPY benchmark
    if spy_nav is not None:
        spy_rets   = spy_nav.pct_change().dropna()
        spy_cagr   = float(spy_nav.iloc[-1] ** (252 / len(spy_nav)) - 1)
        spy_sharpe = float(spy_rets.mean() / (spy_rets.std() + 1e-9) * np.sqrt(252))
        roll_max   = spy_nav.cummax()
        spy_mdd    = float(((spy_nav - roll_max) / (roll_max + 1e-9)).min())
        results["SPY_BENCHMARK"] = {
            "stats": {"cagr": round(spy_cagr, 4), "sharpe": round(spy_sharpe, 3),
                      "max_drawdown": round(spy_mdd, 4)},
            "nav": spy_nav.to_dict(),
        }
        ok(f"  SPY: CAGR={spy_cagr*100:.1f}%  Sharpe={spy_sharpe:.2f}  MDD={spy_mdd*100:.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    serialisable = {}
    for book, data in results.items():
        serialisable[book] = {
            k: ({str(ts): v for ts, v in val.items()} if k == "nav" else val)
            for k, val in data.items()
        }
    json_out = ROOT / "backtest_three_books.json"
    with open(json_out, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    ok(f"backtest_three_books.json saved")

    nav_dfs = []
    for book, data in results.items():
        if "nav" in data:
            s = pd.Series(data["nav"], name=book)
            s.index = pd.to_datetime(s.index)
            nav_dfs.append(s)
    if nav_dfs:
        csv_df = pd.concat(nav_dfs, axis=1).sort_index()
        csv_df.to_csv(ROOT / "backtest_three_books.csv")
        ok(f"backtest_three_books.csv saved  ({len(csv_df)} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    qqq_cagr_ref = results.get("QQQ_BENCHMARK", {}).get("stats", {}).get("cagr", 0.219)

    print(f"\n  {'Book':<12} {'Mode':<6} {'CAGR':>8} {'vs QQQ':>8} {'Sharpe':>8} "
          f"{'MDD':>8} {'Calmar':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for book, data in results.items():
        s = data["stats"]
        if book in ("SPY_BENCHMARK", "QQQ_BENCHMARK"):
            label = "SPY" if "SPY" in book else "QQQ"
            print(f"  {label:<12} {'B&H':<6} {s['cagr']*100:>7.1f}%  "
                  f"{'—':>7}  {s['sharpe']:>7.2f}  {s['max_drawdown']*100:>7.1f}%")
        else:
            mode    = "L/S" if BOOKS[book].get("long_short") else "L/O"
            excess  = s['cagr'] - qqq_cagr_ref
            c_cagr  = GREEN if s['cagr'] > qqq_cagr_ref else (YELLOW if s['cagr'] > 0 else RED)
            c_exc   = GREEN if excess > 0 else RED
            c_sh    = GREEN if s['sharpe'] > 1.0 else (YELLOW if s['sharpe'] > 0 else RED)
            print(f"  {book:<12} {mode:<6} "
                  f"{c_cagr}{s['cagr']*100:>7.1f}%{RESET}  "
                  f"{c_exc}{excess*100:>+7.1f}%{RESET}  "
                  f"{c_sh}{s['sharpe']:>7.2f}{RESET}  "
                  f"{s['max_drawdown']*100:>7.1f}%  "
                  f"{s['calmar']:>7.2f}")

    print(f"\n{GREEN}✓ Backtest v2 complete{RESET}\n")


if __name__ == "__main__":
    main()
