"""
Canyon v9 — Step 92: Stock Strategy Backtest
=============================================
Runs 4 independent strategy backtests on S&P 500 history and compares
each vs SPY benchmark.

Strategies:
  1. Breakout_52w      — 52-week breakout, 10 positions
  2. MA_Crossover      — Golden/Death cross, 15 positions
  3. Sector_Rotation   — Monthly top-3 sector ETFs by 63-day return
  4. RSI_MeanReversion — RSI<30 in uptrend, 8 positions

Usage:
  python canyon_final_v9_step92_stock_strategy_backtest.py [--strategy all] [--top 100] [--years 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
PRICE_CACHE = ROOT / "sp500_price_cache.csv"
REGIME_SCORES = ROOT / "regime_ml_scores.csv"
SECTOR_ETF_CACHE = ROOT / "sector_etf_price_cache.csv"
OUTPUT_CSV = ROOT / "strategy_backtest_results.csv"
OUTPUT_MD = ROOT / "strategy_backtest_report.md"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLI", "XLE", "XLB", "XLRE", "XLU", "XLC", "XLP"]
TX_COST = 0.0005   # one-way; round-trip = 0.10%
RISK_FREE = 0.05   # annual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def annualised_return(daily_rets: pd.Series) -> float:
    """CAGR from a series of daily returns (not log)."""
    total = (1 + daily_rets).prod()
    n_years = len(daily_rets) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def sharpe(daily_rets: pd.Series, rf: float = RISK_FREE) -> float:
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess = daily_rets - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def max_drawdown(daily_rets: pd.Series) -> float:
    cum = (1 + daily_rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def calmar(ann_ret: float, mdd: float) -> float:
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def monthly_returns(daily_rets: pd.Series) -> pd.Series:
    """Compound daily returns into monthly (period end)."""
    return (1 + daily_rets).resample("ME").prod() - 1


def load_prices(years: int) -> pd.DataFrame:
    """Load price cache, filter to last *years* years, forward-fill gaps."""
    if not PRICE_CACHE.exists():
        print(f"[ERROR] Price cache not found: {PRICE_CACHE}")
        sys.exit(1)
    print(f"[INFO] Loading price cache from {PRICE_CACHE} …")
    prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    cutoff = prices.index[-1] - pd.DateOffset(years=years)
    prices = prices.loc[cutoff:]
    prices = prices.ffill().bfill()
    print(f"[INFO] Price data: {prices.index[0].date()} → {prices.index[-1].date()}  "
          f"({len(prices)} days, {prices.shape[1]} tickers)")
    return prices


def load_universe(top_n: int) -> list[str]:
    """Return top-N tickers by predicted_score from regime_ml_scores.csv."""
    if not REGIME_SCORES.exists():
        print(f"[WARN] {REGIME_SCORES} not found — using all tickers in price cache.")
        return []
    scores = pd.read_csv(REGIME_SCORES)
    scores = scores.dropna(subset=["predicted_score"])
    scores = scores.sort_values("predicted_score", ascending=False)
    return scores["ticker"].head(top_n).tolist()


def ensure_spy(prices: pd.DataFrame) -> pd.Series:
    """Return SPY price series, fetching from yfinance if missing."""
    if "SPY" in prices.columns:
        return prices["SPY"]
    print("[INFO] SPY not in price cache — fetching from yfinance …")
    try:
        import yfinance as yf
        spy_df = yf.download("SPY", start=prices.index[0], end=prices.index[-1] + timedelta(days=1),
                             auto_adjust=True, progress=False)
        spy = spy_df["Close"].squeeze()
        spy.name = "SPY"
        spy.index = pd.to_datetime(spy.index)
        return spy.reindex(prices.index).ffill().bfill()
    except Exception as exc:
        print(f"[WARN] Could not fetch SPY: {exc}. Using first available column as benchmark.")
        return prices.iloc[:, 0]


def ensure_sector_etfs(prices: pd.DataFrame, years: int) -> pd.DataFrame:
    """Return price DataFrame for sector ETFs, fetching if missing."""
    available = [t for t in SECTOR_ETFS if t in prices.columns]
    missing = [t for t in SECTOR_ETFS if t not in prices.columns]

    frames = {}
    for t in available:
        frames[t] = prices[t]

    if missing:
        if SECTOR_ETF_CACHE.exists():
            print(f"[INFO] Loading sector ETF cache from {SECTOR_ETF_CACHE}")
            cached = pd.read_csv(SECTOR_ETF_CACHE, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index)
            for t in missing:
                if t in cached.columns:
                    frames[t] = cached[t].reindex(prices.index).ffill().bfill()
                    missing = [x for x in missing if x != t]

    if missing:
        print(f"[INFO] Fetching missing sector ETFs from yfinance: {missing}")
        try:
            import yfinance as yf
            start = prices.index[0] - timedelta(days=30)
            end = prices.index[-1] + timedelta(days=1)
            dl = yf.download(missing, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(dl.columns, pd.MultiIndex):
                closes = dl["Close"]
            else:
                closes = dl[["Close"]] if "Close" in dl.columns else dl
            closes.index = pd.to_datetime(closes.index)
            for t in missing:
                if t in closes.columns:
                    frames[t] = closes[t].reindex(prices.index).ffill().bfill()
            # Save for future use
            new_cache = pd.DataFrame(frames)[missing]
            new_cache.to_csv(SECTOR_ETF_CACHE)
            print(f"[INFO] Saved sector ETF cache to {SECTOR_ETF_CACHE}")
        except Exception as exc:
            print(f"[WARN] Could not fetch sector ETFs: {exc}. "
                  "Sector rotation will use only available ETFs.")

    available_frames = {k: v for k, v in frames.items() if v is not None}
    if not available_frames:
        return pd.DataFrame(index=prices.index)
    return pd.DataFrame(available_frames).reindex(prices.index).ffill().bfill()


def filter_universe(universe: list[str], prices: pd.DataFrame) -> list[str]:
    """Keep only tickers present in price DataFrame."""
    cols = set(prices.columns)
    return [t for t in universe if t in cols]


# ---------------------------------------------------------------------------
# Strategy 1: Breakout_52w
# ---------------------------------------------------------------------------

def run_breakout_52w(prices: pd.DataFrame, universe: list[str],
                     spy_rets: pd.Series) -> dict:
    """52-week breakout strategy."""
    print("[Breakout_52w] Running …")
    tickers = filter_universe(universe, prices)
    if not tickers:
        print("[Breakout_52w] No valid tickers. Skipping.")
        return {}

    px = prices[tickers].copy()
    max_positions = 10
    hold_days = 21          # calendar days
    ma_window = 20

    # Precompute signals
    rolling_max_252 = px.rolling(252, min_periods=200).max()
    ma_20 = px.rolling(ma_window, min_periods=ma_window).mean()

    # State
    portfolio: dict[str, dict] = {}   # ticker -> {entry_price, entry_date}
    daily_rets: list[float] = []
    trade_results: list[float] = []
    dates = px.index

    for i in range(252, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # ----- Exit logic -----
        to_exit = []
        for ticker, pos in portfolio.items():
            curr_price = px.loc[date, ticker]
            curr_ma20 = ma_20.loc[date, ticker]
            days_held = (date - pos["entry_date"]).days

            if (curr_price < curr_ma20 or days_held >= hold_days
                    or np.isnan(curr_price)):
                to_exit.append(ticker)

        for ticker in to_exit:
            pos = portfolio.pop(ticker)
            exit_price = px.loc[date, ticker]
            if np.isnan(exit_price) or pos["entry_price"] == 0:
                continue
            trade_ret = (exit_price / pos["entry_price"] - 1) - 2 * TX_COST
            trade_results.append(float(trade_ret))

        # ----- Entry logic -----
        slots = max_positions - len(portfolio)
        if slots > 0:
            close_today = px.loc[date]
            roll_max_prev = rolling_max_252.loc[prev_date]  # avoid look-ahead
            # breakout: today's close > yesterday's 252-day max
            signal_mask = (close_today > roll_max_prev) & close_today.notna()
            candidates = [t for t in tickers
                          if signal_mask.get(t, False) and t not in portfolio]
            # sort by how far above rolling max (momentum)
            scores = []
            for t in candidates:
                rm = roll_max_prev.get(t, np.nan)
                if rm and rm > 0:
                    scores.append((t, close_today[t] / rm))
            scores.sort(key=lambda x: x[1], reverse=True)
            for ticker, _ in scores[:slots]:
                entry_price = px.loc[date, ticker]
                if np.isnan(entry_price) or entry_price == 0:
                    continue
                portfolio[ticker] = {
                    "entry_price": float(entry_price),
                    "entry_date": date,
                }

        # ----- Daily P&L -----
        if portfolio and i > 0:
            pos_rets = []
            for ticker, pos in portfolio.items():
                p0 = px.loc[prev_date, ticker]
                p1 = px.loc[date, ticker]
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    pos_rets.append(p1 / p0 - 1)
            daily_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
        else:
            daily_ret = 0.0

        daily_rets.append(daily_ret)

    daily_ret_series = pd.Series(daily_rets, index=dates[252:], name="Breakout_52w")
    return _compile_results("Breakout_52w", daily_ret_series, spy_rets, trade_results)


# ---------------------------------------------------------------------------
# Strategy 2: MA_Crossover
# ---------------------------------------------------------------------------

def run_ma_crossover(prices: pd.DataFrame, universe: list[str],
                     spy_rets: pd.Series) -> dict:
    """Golden/Death cross strategy."""
    print("[MA_Crossover] Running …")
    tickers = filter_universe(universe, prices)
    if not tickers:
        print("[MA_Crossover] No valid tickers. Skipping.")
        return {}

    px = prices[tickers].copy()
    max_positions = 15
    hold_days_td = 63  # trading days

    ma_50 = px.rolling(50, min_periods=45).mean()
    ma_200 = px.rolling(200, min_periods=180).mean()

    # Golden cross: ma50 > ma200 AND previous day ma50 <= ma200
    golden_cross = (ma_50 > ma_200) & (ma_50.shift(1) <= ma_200.shift(1))
    death_cross = (ma_50 < ma_200) & (ma_50.shift(1) >= ma_200.shift(1))

    portfolio: dict[str, dict] = {}
    daily_rets: list[float] = []
    trade_results: list[float] = []
    dates = px.index

    for i in range(200, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # ----- Exit -----
        to_exit = []
        for ticker, pos in portfolio.items():
            days_held_td = pos["days_held"]
            if (death_cross.loc[date, ticker]
                    or days_held_td >= hold_days_td
                    or np.isnan(px.loc[date, ticker])):
                to_exit.append(ticker)

        for ticker in to_exit:
            pos = portfolio.pop(ticker)
            exit_price = px.loc[date, ticker]
            if np.isnan(exit_price) or pos["entry_price"] == 0:
                continue
            trade_ret = (exit_price / pos["entry_price"] - 1) - 2 * TX_COST
            trade_results.append(float(trade_ret))

        # Increment held days
        for pos in portfolio.values():
            pos["days_held"] += 1

        # ----- Entry -----
        slots = max_positions - len(portfolio)
        if slots > 0:
            candidates = [
                t for t in tickers
                if golden_cross.loc[prev_date, t] and t not in portfolio
                and not np.isnan(px.loc[date, t])
            ]
            for ticker in candidates[:slots]:
                portfolio[ticker] = {
                    "entry_price": float(px.loc[date, ticker]),
                    "days_held": 0,
                }

        # ----- Daily P&L -----
        if portfolio:
            pos_rets = []
            for ticker in portfolio:
                p0 = px.loc[prev_date, ticker]
                p1 = px.loc[date, ticker]
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    pos_rets.append(p1 / p0 - 1)
            daily_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
        else:
            daily_ret = 0.0

        daily_rets.append(daily_ret)

    daily_ret_series = pd.Series(daily_rets, index=dates[200:], name="MA_Crossover")
    return _compile_results("MA_Crossover", daily_ret_series, spy_rets, trade_results)


# ---------------------------------------------------------------------------
# Strategy 3: Sector_Rotation
# ---------------------------------------------------------------------------

def run_sector_rotation(prices: pd.DataFrame, spy_rets: pd.Series,
                        years: int) -> dict:
    """Monthly top-3 sector ETF rotation."""
    print("[Sector_Rotation] Running …")
    sector_px = ensure_sector_etfs(prices, years)
    available_etfs = [t for t in SECTOR_ETFS if t in sector_px.columns]
    if len(available_etfs) < 3:
        print(f"[Sector_Rotation] Only {len(available_etfs)} sector ETFs available "
              f"(need ≥3). Skipping.")
        return {}

    px = sector_px[available_etfs].copy()
    dates = px.index

    # Monthly rebalance dates (last trading day of each month)
    monthly_ends = px.resample("ME").last().index
    monthly_ends = monthly_ends[monthly_ends >= dates[0]]

    portfolio: dict[str, float] = {}   # ticker -> entry_price
    daily_rets: list[float] = []
    trade_results: list[float] = []
    current_holdings: list[str] = []

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # Check if we're at a month-end rebalance
        if date in monthly_ends or i == 1:
            # Compute 63-day returns for ranking
            lookback_idx = max(0, i - 63)
            lb_date = dates[lookback_idx]
            rets_63 = {}
            for t in available_etfs:
                p0 = px.loc[lb_date, t]
                p1 = px.loc[date, t]
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    rets_63[t] = p1 / p0 - 1
            ranked = sorted(rets_63, key=lambda t: rets_63[t], reverse=True)
            new_holdings = ranked[:3]

            # Exit old positions
            for ticker in current_holdings:
                if ticker not in new_holdings:
                    exit_price = px.loc[date, ticker]
                    ep = portfolio.get(ticker, 0)
                    if ep > 0 and not np.isnan(exit_price):
                        trade_ret = (exit_price / ep - 1) - 2 * TX_COST
                        trade_results.append(float(trade_ret))

            # Enter new positions
            for ticker in new_holdings:
                if ticker not in current_holdings:
                    portfolio[ticker] = float(px.loc[date, ticker])

            # Remove exited
            for ticker in current_holdings:
                if ticker not in new_holdings:
                    portfolio.pop(ticker, None)

            current_holdings = new_holdings

        # Daily P&L
        if current_holdings:
            pos_rets = []
            for ticker in current_holdings:
                p0 = px.loc[prev_date, ticker]
                p1 = px.loc[date, ticker]
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    pos_rets.append(p1 / p0 - 1)
            daily_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
        else:
            daily_ret = 0.0

        daily_rets.append(daily_ret)

    daily_ret_series = pd.Series(daily_rets, index=dates[1:], name="Sector_Rotation")
    return _compile_results("Sector_Rotation", daily_ret_series, spy_rets, trade_results)


# ---------------------------------------------------------------------------
# Strategy 4: RSI_MeanReversion
# ---------------------------------------------------------------------------

def run_rsi_mean_reversion(prices: pd.DataFrame, universe: list[str],
                           spy_rets: pd.Series) -> dict:
    """RSI < 30 in uptrend mean reversion."""
    print("[RSI_MeanReversion] Running …")
    tickers = filter_universe(universe, prices)
    if not tickers:
        print("[RSI_MeanReversion] No valid tickers. Skipping.")
        return {}

    px = prices[tickers].copy()
    max_positions = 8
    hold_days_max = 10
    rsi_period = 14
    ma200_window = 200
    rsi_exit = 60
    stop_pct = 0.97

    # Precompute indicators
    print("[RSI_MeanReversion]   Computing RSI and MA200 …")
    rsi_df = px.apply(lambda col: rsi(col, rsi_period))
    ma200_df = px.rolling(ma200_window, min_periods=180).mean()

    portfolio: dict[str, dict] = {}
    daily_rets: list[float] = []
    trade_results: list[float] = []
    dates = px.index

    for i in range(ma200_window, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # ----- Exit -----
        to_exit = []
        for ticker, pos in portfolio.items():
            curr_price = px.loc[date, ticker]
            curr_rsi = rsi_df.loc[date, ticker]
            curr_ma200 = ma200_df.loc[date, ticker]
            days_held = pos["days_held"]

            stop_level = pos["entry_price"] * stop_pct
            if (curr_rsi > rsi_exit
                    or days_held >= hold_days_max
                    or curr_price < stop_level
                    or curr_price < curr_ma200
                    or np.isnan(curr_price)):
                to_exit.append(ticker)

        for ticker in to_exit:
            pos = portfolio.pop(ticker)
            exit_price = px.loc[date, ticker]
            if np.isnan(exit_price) or pos["entry_price"] == 0:
                continue
            trade_ret = (exit_price / pos["entry_price"] - 1) - 2 * TX_COST
            trade_results.append(float(trade_ret))

        # Increment held days
        for pos in portfolio.values():
            pos["days_held"] += 1

        # ----- Entry -----
        slots = max_positions - len(portfolio)
        if slots > 0:
            # Signal on PREVIOUS day to avoid look-ahead; enter at today's close
            close_prev = px.loc[prev_date]
            rsi_prev = rsi_df.loc[prev_date]
            ma200_prev = ma200_df.loc[prev_date]

            candidates = []
            for t in tickers:
                if t in portfolio:
                    continue
                r = rsi_prev.get(t, np.nan)
                m = ma200_prev.get(t, np.nan)
                c = close_prev.get(t, np.nan)
                if (not np.isnan(r) and not np.isnan(m) and not np.isnan(c)
                        and r < 30 and c > m):
                    # Sort by most oversold first
                    candidates.append((t, float(r)))

            candidates.sort(key=lambda x: x[1])
            for ticker, _ in candidates[:slots]:
                entry_price = px.loc[date, ticker]
                if np.isnan(entry_price) or entry_price == 0:
                    continue
                portfolio[ticker] = {
                    "entry_price": float(entry_price),
                    "days_held": 0,
                }

        # ----- Daily P&L -----
        if portfolio:
            pos_rets = []
            for ticker in portfolio:
                p0 = px.loc[prev_date, ticker]
                p1 = px.loc[date, ticker]
                if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                    pos_rets.append(p1 / p0 - 1)
            daily_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
        else:
            daily_ret = 0.0

        daily_rets.append(daily_ret)

    daily_ret_series = pd.Series(daily_rets, index=dates[ma200_window:],
                                 name="RSI_MeanReversion")
    return _compile_results("RSI_MeanReversion", daily_ret_series, spy_rets, trade_results)


# ---------------------------------------------------------------------------
# Results compilation
# ---------------------------------------------------------------------------

def _compile_results(name: str, daily_rets: pd.Series,
                     spy_rets: pd.Series, trade_results: list[float]) -> dict:
    """Compute full metric set and monthly rows for a strategy."""
    if daily_rets.empty:
        return {}

    # Align with spy
    spy_aligned = spy_rets.reindex(daily_rets.index).fillna(0.0)

    ann_ret = annualised_return(daily_rets)
    sr = sharpe(daily_rets)
    mdd = max_drawdown(daily_rets)
    cal = calmar(ann_ret, mdd)
    spy_ann = annualised_return(spy_aligned)
    alpha = ann_ret - spy_ann

    n_trades = len(trade_results)
    win_rate = float(np.mean([r > 0 for r in trade_results])) if trade_results else 0.0
    avg_trade = float(np.mean(trade_results)) if trade_results else 0.0

    metrics = {
        "strategy": name,
        "ann_return": ann_ret,
        "sharpe": sr,
        "max_drawdown": mdd,
        "calmar": cal,
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade,
        "n_trades": n_trades,
        "spy_ann_return": spy_ann,
        "alpha": alpha,
    }

    # Monthly rows
    monthly_strat = monthly_returns(daily_rets)
    monthly_spy = monthly_returns(spy_aligned)

    cum_strat = (1 + daily_rets).cumprod()
    cum_spy = (1 + spy_aligned).cumprod()

    rows = []
    for dt in monthly_strat.index:
        # Get last cumulative value for this month
        month_end = dt
        try:
            cum_val = float(cum_strat.loc[:month_end].iloc[-1])
        except Exception:
            cum_val = np.nan
        rows.append({
            "date": dt.strftime("%Y-%m"),
            "strategy": name,
            "ret": float(monthly_strat[dt]),
            "cum_ret": cum_val,
            "spy_ret": float(monthly_spy.get(dt, 0.0)),
            "alpha": float(monthly_strat[dt]) - float(monthly_spy.get(dt, 0.0)),
        })

    return {"metrics": metrics, "monthly_rows": rows, "daily_rets": daily_rets}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def fmt_pct_plain(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_comparison_table(all_metrics: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Strategy':<22} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} "
        f"{'Calmar':>7} {'WinRate':>8} {'Trades':>7} {'Alpha':>8}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  STRATEGY COMPARISON vs SPY BENCHMARK")
    print("=" * len(header))
    print(header)
    print(sep)
    for m in all_metrics:
        print(
            f"{m['strategy']:<22} "
            f"{fmt_pct(m['ann_return']):>8} "
            f"{m['sharpe']:>7.2f} "
            f"{fmt_pct(m['max_drawdown']):>8} "
            f"{m['calmar']:>7.2f} "
            f"{fmt_pct_plain(m['win_rate']):>8} "
            f"{m['n_trades']:>7d} "
            f"{fmt_pct(m['alpha']):>8}"
        )
    print(sep)
    spy_ann = all_metrics[0]["spy_ann_return"] if all_metrics else 0.0
    print(f"{'SPY Benchmark':<22} {fmt_pct(spy_ann):>8}")
    print("=" * len(header))
    print()


def build_report_md(all_metrics: list[dict], years: int, top_n: int) -> str:
    lines = []
    lines.append("# Canyon v9 — Step 92: Strategy Backtest Report")
    lines.append("")
    lines.append(f"**Backtest period:** last {years} year(s)  ")
    lines.append(f"**Universe:** top {top_n} tickers by predicted_score  ")
    lines.append(f"**Transaction cost:** {TX_COST*100:.2f}% one-way ({TX_COST*200:.2f}% round-trip)  ")
    lines.append(f"**Risk-free rate:** {RISK_FREE*100:.1f}%  ")
    lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| Strategy | Ann. Return | Sharpe | Max DD | Calmar | Win Rate | # Trades | Alpha |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for m in all_metrics:
        lines.append(
            f"| {m['strategy']} "
            f"| {fmt_pct(m['ann_return'])} "
            f"| {m['sharpe']:.2f} "
            f"| {fmt_pct(m['max_drawdown'])} "
            f"| {m['calmar']:.2f} "
            f"| {fmt_pct_plain(m['win_rate'])} "
            f"| {m['n_trades']} "
            f"| {fmt_pct(m['alpha'])} |"
        )
    if all_metrics:
        spy_ann = all_metrics[0]["spy_ann_return"]
        lines.append(
            f"| **SPY Benchmark** | {fmt_pct(spy_ann)} | — | — | — | — | — | 0.00% |"
        )
    lines.append("")

    lines.append("## Strategy Details")
    lines.append("")
    for m in all_metrics:
        lines.append(f"### {m['strategy']}")
        lines.append("")
        lines.append(f"- **Annualised Return:** {fmt_pct(m['ann_return'])}")
        lines.append(f"- **SPY Annualised Return (same period):** {fmt_pct(m['spy_ann_return'])}")
        lines.append(f"- **Alpha:** {fmt_pct(m['alpha'])}")
        lines.append(f"- **Sharpe Ratio:** {m['sharpe']:.3f}")
        lines.append(f"- **Max Drawdown:** {fmt_pct(m['max_drawdown'])}")
        lines.append(f"- **Calmar Ratio:** {m['calmar']:.3f}")
        lines.append(f"- **Win Rate:** {fmt_pct_plain(m['win_rate'])}")
        lines.append(f"- **Avg Trade Return:** {fmt_pct(m['avg_trade_ret'])}")
        lines.append(f"- **Number of Trades:** {m['n_trades']}")
        lines.append("")

    lines.append("## Strategy Descriptions")
    lines.append("")
    lines.append("### Breakout_52w")
    lines.append("Buys when a stock makes a new 52-week (252-day) closing high.")
    lines.append("Exits when price falls below the 20-day MA or after 21 calendar days.")
    lines.append("Max 10 equal-weight positions. Daily scanning.")
    lines.append("")
    lines.append("### MA_Crossover")
    lines.append("Golden cross (50d MA crosses above 200d MA) entry.")
    lines.append("Death cross or 63 trading days exit. Max 15 equal-weight positions.")
    lines.append("")
    lines.append("### Sector_Rotation")
    lines.append("Monthly rebalance into top-3 sector ETFs by 63-day total return.")
    lines.append("Uses XLK, XLF, XLV, XLY, XLI, XLE, XLB, XLRE, XLU, XLC, XLP.")
    lines.append("")
    lines.append("### RSI_MeanReversion")
    lines.append("RSI(14) < 30 AND price > 200-day MA. Max 8 equal-weight positions.")
    lines.append("Exit when RSI > 60, held 10 trading days, or price < 200d MA × 0.97.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step92 — 4-strategy stock backtest vs SPY"
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "breakout", "ma_cross", "sector_rot", "rsi_rev"],
        default="all",
        help="Which strategy to run (default: all)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        metavar="N",
        help="Universe size from regime_ml_scores (default: 100)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        metavar="N",
        help="Years of backtest history (default: 2)",
    )
    args = parser.parse_args()

    run_all = args.strategy == "all"

    # ----- Load data -----
    prices = load_prices(args.years)
    universe_raw = load_universe(args.top)
    if universe_raw:
        print(f"[INFO] Universe: {len(universe_raw)} tickers from regime_ml_scores")
    else:
        universe_raw = list(prices.columns)
        print(f"[INFO] Using all {len(universe_raw)} tickers from price cache")

    # Trim universe to top N
    universe = universe_raw[: args.top]

    spy_series = ensure_spy(prices)
    spy_rets = spy_series.pct_change().fillna(0.0)

    # ----- Run strategies -----
    all_results: list[dict] = []

    if run_all or args.strategy == "breakout":
        r = run_breakout_52w(prices, universe, spy_rets)
        if r:
            all_results.append(r)

    if run_all or args.strategy == "ma_cross":
        r = run_ma_crossover(prices, universe, spy_rets)
        if r:
            all_results.append(r)

    if run_all or args.strategy == "sector_rot":
        r = run_sector_rotation(prices, spy_rets, args.years)
        if r:
            all_results.append(r)

    if run_all or args.strategy == "rsi_rev":
        r = run_rsi_mean_reversion(prices, universe, spy_rets)
        if r:
            all_results.append(r)

    if not all_results:
        print("[ERROR] No strategy results produced. Check input data.")
        sys.exit(1)

    # ----- Output CSV -----
    all_monthly_rows: list[dict] = []
    all_metrics: list[dict] = []
    for r in all_results:
        all_monthly_rows.extend(r["monthly_rows"])
        all_metrics.append(r["metrics"])

    monthly_df = pd.DataFrame(all_monthly_rows)
    monthly_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Monthly results saved to {OUTPUT_CSV}")

    # ----- Output Markdown -----
    report_text = build_report_md(all_metrics, args.years, args.top)
    OUTPUT_MD.write_text(report_text, encoding="utf-8")
    print(f"[INFO] Report saved to {OUTPUT_MD}")

    # ----- Print comparison table -----
    print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
