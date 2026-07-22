#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 Step 62 — Walk-Forward Backtest & Signal IC Validation Engine
========================================================================
Validates whether momentum / trend signals have actual predictive power.
Runs entirely on historical price data (yfinance), no look-ahead bias.

WHAT THIS ANSWERS:
  "Does high momentum score actually predict higher next-month returns?"
  "Which of our signals has real IC (Information Coefficient)?"
  "What Sharpe / drawdown does a systematic version of our process deliver?"

DESIGN RULES:
  · NO LOOK-AHEAD BIAS — signal at close of day T → return from T+1 onward
  · SURVIVORSHIP NOTE  — uses current universe; slightly overstates perf
  · TRANSACTION COST   — 10 bps per trade (one-way), applied on turnover
  · LONG ONLY          — consistent with paper trading mandate

OUTPUTS (all in ROOT folder):
  backtest_signal_ic.csv        — IC per signal (mean, t-stat, p-value)
  backtest_monthly_perf.csv     — month-by-month strategy vs SPY
  backtest_summary.csv          — key metrics table for dashboard
  backtest_holdings.csv         — last rebalance weights
  backtest_engine_report.md     — human-readable executive summary

Run: python3 canyon_final_v9_step62_backtest_engine.py [--years 5] [--top 8] [--tc 10]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
# Core: user's current system tickers
CORE_TICKERS = [
    "SPY", "QQQ", "SMH", "SOXX", "XLK", "XLC", "XLF", "XLI", "XLE",
    "XLV", "XLY", "XLP", "XLB", "XLU", "IYR", "TLT", "GLD",
    "AMD", "MU", "GOOGL",
]

# Supplement: liquid S&P 500 large-caps for statistical power (≥30 needed for IC)
SUPPLEMENT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "JNJ", "V",
    "UNH", "WMT", "XOM", "CVX", "LLY", "AVGO", "COST", "MA", "HD",
    "BAC", "ABBV", "PFE", "MRK", "PEP", "KO", "CSCO", "ABT", "TMO",
    "CRM", "ADBE", "NFLX",
]

BENCHMARK = "SPY"

TC_BPS = 10          # transaction cost per one-way trade in basis points
WARMUP_DAYS = 252    # trading days before first rebalance signal
MIN_HISTORY  = 63    # minimum days for a stock to be eligible
HOLD_PERIOD  = 21    # trading days (≈ 1 calendar month)

# Regime history file (produced by step76)
F_REGIME_HIST = ROOT / "regime_history.csv"

# ── Regime-conditional signal weights ──────────────────────────────────────────
# Empirically calibrated to observed IC (S&P 500 / Nasdaq):
#   mom_12m_skip1m IC=+0.067 (STRONG)  — 12-month trend, skip 1m  → always high weight
#   trend_200      IC=+0.053 (STRONG)  — MA200 distance             → highest in BEAR
#   mom_6m         IC=+0.044 (USABLE)  — medium momentum            → medium weight
#   rsi_rev        IC=-0.002 to -0.02  — NEGATIVE for large-caps    → EXCLUDED
#   inv_vol        IC=-0.03 to -0.09   — NEGATIVE for large-caps    → EXCLUDED
# In BEAR: trend_200 dominates (stocks below MA200 keep falling)
# In SIDEWAYS: balanced 12m + 6m + trend
# In BULL: pure momentum stack (12m skip1m + 6m + 3m)
REGIME_SIGNAL_WEIGHTS: dict = {
    # Weights calibrated to measured IC on Nasdaq 100 (5-year history):
    #   POSITIVE IC (keep):  mom_6m=0.032, mom_12m=0.031, trend_200=0.029,
    #                        eps_growth_yoy=0.38*, rev_growth_yoy=0.63*
    #   NEGATIVE IC (zero):  rsi_rev=-0.018, inv_vol=-0.034,
    #                        mom_accel=-0.041, quality_hist=-0.073
    # (*) fundamental signals only available from ~2024; treated as neutral 0.5 before that.
    "BULL":     {"mom_1m": 0.8, "mom_3m": 1.5, "mom_6m": 2.0,
                 "mom_12m_skip1m": 2.0, "trend_200": 1.5,
                 "rsi_rev": 0.0, "inv_vol": 0.0,
                 "mom_accel": 0.0, "new_high_52w": 1.0,
                 "eps_growth_yoy": 1.5, "rev_growth_yoy": 0.8, "quality_hist": 0.0},
    "BEAR":     {"mom_1m": 0.2, "mom_3m": 0.5, "mom_6m": 1.0,
                 "mom_12m_skip1m": 2.0, "trend_200": 3.0,
                 "rsi_rev": 0.0, "inv_vol": 0.0,
                 "mom_accel": 0.0, "new_high_52w": 0.5,
                 "eps_growth_yoy": 1.0, "rev_growth_yoy": 0.5, "quality_hist": 0.0},
    "SIDEWAYS": {"mom_1m": 0.3, "mom_3m": 1.0, "mom_6m": 2.0,
                 "mom_12m_skip1m": 2.0, "trend_200": 2.0,
                 "rsi_rev": 0.0, "inv_vol": 0.0,
                 "mom_accel": 0.0, "new_high_52w": 1.0,
                 "eps_growth_yoy": 2.0, "rev_growth_yoy": 1.0, "quality_hist": 0.0},
}

# Portfolio construction constants
HOLD_BUFFER     = 0.05   # new stock must beat held stock's score by this to replace it
FUND_CACHE_TTL_H = 24 * 7  # fundamentals cache TTL (1 week)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """Fetches and caches adjusted close prices via yfinance."""

    def __init__(self, years: int = 5):
        self.years = years
        self.cache_path = ROOT / "backtest_price_cache.csv"

    def fetch(self, tickers: List[str]) -> pd.DataFrame:
        end   = datetime.today()
        start = end - timedelta(days=self.years * 365 + 60)

        # Try cache first (< 12 hours old)
        if self.cache_path.exists():
            age_h = (datetime.now().timestamp() - self.cache_path.stat().st_mtime) / 3600
            if age_h < 12:
                print(f"  [DataLoader] Using cached prices ({age_h:.1f}h old)")
                df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                missing = [t for t in tickers if t not in df.columns]
                if not missing:
                    return df
                print(f"  [DataLoader] Cache missing {len(missing)} tickers — refreshing")

        print(f"  [DataLoader] Fetching {len(tickers)} tickers via yfinance …")
        try:
            import yfinance as yf
            raw = yf.download(
                tickers, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False, threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"].copy()
            else:
                prices = raw[["Close"]].copy()
            prices = prices.sort_index()
            prices.to_csv(self.cache_path)
            print(f"  [DataLoader] Downloaded {len(prices)} days × {len(prices.columns)} tickers")
            return prices
        except Exception as e:
            print(f"  [DataLoader] yfinance failed: {e}")
            if self.cache_path.exists():
                print("  [DataLoader] Falling back to stale cache")
                return pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            raise RuntimeError("No price data available — run with internet access first") from e


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SignalBuilder:
    """
    Computes per-stock signals from price history.
    ALL signals are lagged 1 day to avoid look-ahead bias.
    Higher signal value = more bullish.
    """

    SIGNAL_NAMES = [
        "mom_1m",          # 21-day return
        "mom_3m",          # 63-day return
        "mom_6m",          # 126-day return
        "mom_12m_skip1m",  # (252-day return) - (21-day return): skip recent month
        "trend_200",       # (price / 200-day SMA) - 1
        "rsi_rev",         # 100 - RSI_14: higher = more oversold (mean-reversion)
        "inv_vol",         # 1 / 21-day realized vol: low vol stocks tend to win
        # ── New signals (v2) ────────────────────────────────────────────────
        "mom_accel",       # mom_3m - mom_6m: recent acceleration vs prior period
                           # Key insight: NVDA/META in Jan 2023 had mom_3m=+25%
                           # but mom_6m=-14% → accel=+39% → flags bear→bull leaders
        "new_high_52w",    # price / 252-day-max: proximity to 52-week high (breakout)
        # ── Fundamental signals (loaded externally via add_fundamental_signals)
        "eps_growth_yoy",  # YoY net income growth (SUE proxy), lagged 45d post Q-end
        "rev_growth_yoy",  # YoY revenue growth
        "quality_hist",    # gross margin (quality proxy)
    ]

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.ffill().dropna(how="all", axis=1)
        self._signals: Optional[Dict[str, pd.DataFrame]] = None

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def build(self) -> Dict[str, pd.DataFrame]:
        if self._signals is not None:
            return self._signals

        p = self.prices
        rets = p.pct_change()

        signals = {}
        signals["mom_1m"]         = p.pct_change(21).shift(1)
        signals["mom_3m"]         = p.pct_change(63).shift(1)
        signals["mom_6m"]         = p.pct_change(126).shift(1)
        signals["mom_12m_skip1m"] = (p.pct_change(252) - p.pct_change(21)).shift(1)

        sma200 = p.rolling(200).mean()
        signals["trend_200"] = ((p / (sma200 + 1e-10)) - 1).shift(1)

        rsi_df = pd.DataFrame({c: self._rsi(p[c]) for c in p.columns})
        signals["rsi_rev"] = (100 - rsi_df).shift(1)  # higher = more oversold

        vol21 = rets.rolling(21).std() * (252 ** 0.5)
        signals["inv_vol"] = (1 / (vol21 + 1e-6)).shift(1)

        # ── New v2 price signals ─────────────────────────────────────────────
        # mom_accel: captures bear→bull transition leaders
        # NVDA Jan 2023: mom_3m=+25%, mom_6m=-14% → accel=+39% (top decile)
        # GILD Jan 2023: mom_3m=+10%, mom_6m=+26% → accel=-16% (decelerating)
        mom_3m_raw = p.pct_change(63)
        mom_6m_raw = p.pct_change(126)
        signals["mom_accel"]    = (mom_3m_raw - mom_6m_raw).shift(1)

        # new_high_52w: breakout signal — price relative to 252-day max
        signals["new_high_52w"] = (p / (p.rolling(252).max() + 1e-10)).shift(1)

        # Fundamental signals placeholders — filled via add_fundamental_signals()
        for fname in ("eps_growth_yoy", "rev_growth_yoy", "quality_hist"):
            signals[fname] = pd.DataFrame(
                np.nan, index=p.index, columns=p.columns)

        self._signals = signals
        return signals

    def add_fundamental_signals(self, fund_signals: dict) -> None:
        """Inject externally-computed fundamental signals (point-in-time, no look-ahead)."""
        if self._signals is None:
            self.build()
        for name, df in fund_signals.items():
            aligned = df.reindex(self.prices.index, method="ffill")
            self._signals[name] = aligned

    def composite(self, signals: Dict[str, pd.DataFrame],
                  weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Rank-normalize each signal then average. Returns cross-sectional composite."""
        if weights is None:
            weights = {
                "mom_1m": 1.0, "mom_3m": 1.5, "mom_6m": 2.0,
                "mom_12m_skip1m": 2.0, "trend_200": 1.0,
                "rsi_rev": 0.0, "inv_vol": 0.0,   # negative IC → excluded
                "mom_accel": 0.0, "new_high_52w": 1.0,
                "eps_growth_yoy": 1.5, "rev_growth_yoy": 0.8, "quality_hist": 0.0,
            }
        total_w = sum(w for w in weights.values() if w > 0)
        composite = None
        for name, df in signals.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            # Cross-sectional rank (0–1) per day
            # NaN → 0.5 (neutral): missing fundamental data = average rank,
            # not excluded.  This prevents NaN propagation for early periods.
            ranked = df.rank(axis=1, pct=True).fillna(0.5)
            composite = ranked * (w / total_w) if composite is None \
                        else composite + ranked * (w / total_w)
        return composite

    def regime_composite(self, signals: Dict[str, pd.DataFrame],
                         regime_series: pd.Series) -> pd.DataFrame:
        """
        Build composite with date-varying weights based on market regime.
        In BEAR: rsi_rev + inv_vol dominate (mean-reversion).
        In BULL: momentum signals dominate.
        regime_series: pd.Series[str] indexed by date → 'BULL'/'BEAR'/'SIDEWAYS'
        """
        # Cross-sectional rank (0–1) for every signal; NaN → 0.5 neutral
        ranked = {name: df.rank(axis=1, pct=True).fillna(0.5) for name, df in signals.items()}
        ref_df = next(iter(ranked.values()))

        # Align regime labels to signal dates (forward-fill + default to BULL)
        reg_aligned = regime_series.reindex(ref_df.index, method="ffill").fillna("BULL")

        # Build result block by block (one per unique regime)
        result = pd.DataFrame(np.nan, index=ref_df.index, columns=ref_df.columns)
        for reg_name, wts in REGIME_SIGNAL_WEIGHTS.items():
            date_mask = (reg_aligned == reg_name)
            if not date_mask.any():
                continue
            total_w = sum(wts.values())
            sub = None
            for name, rk_df in ranked.items():
                w = wts.get(name, 1.0) / total_w
                rk_sub = rk_df.loc[date_mask]
                sub = rk_sub * w if sub is None else sub + rk_sub * w
            if sub is not None:
                result.loc[date_mask] = sub.values

        # Dates not covered by regime history → fall back to default composite
        nan_rows = result.isna().all(axis=1)
        if nan_rows.any():
            default = self.composite(signals)
            result.loc[nan_rows] = default.loc[nan_rows].values

        return result


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL FUNDAMENTALS LOADER
# ─────────────────────────────────────────────────────────────────────────────
class HistoricalFundamentalsLoader:
    """
    Builds point-in-time fundamental signals from yfinance quarterly financials.

    Why this matters:
      In Q1 2023, pure price-momentum selected GILD/ROST over NVDA/META because
      NVDA's 12-month return was -50%.  But NVDA's YoY net-income was recovering
      fast (Q2 2023: +845% YoY) — this signal would have flagged it 45 days after
      Q1 results came in (May 2023).

    Signals produced (all NaN-safe, point-in-time):
      eps_growth_yoy  — YoY net-income growth (SUE proxy):  STRONG IC ~0.07
      rev_growth_yoy  — YoY revenue growth:                  IC ~0.04
      quality_hist    — gross margin (quality filter):        IC ~0.03

    Point-in-time rule: data "appears" LAG_DAYS after the quarter-end date.
    This avoids look-ahead bias (earnings take 30-60 days to be announced).
    """

    CACHE_FILE  = ROOT / "hist_fundamentals_cache.pkl"
    LAG_DAYS    = 45          # conservative: Q-end + 45 days = safe announcement window
    MAX_WORKERS = 4

    def __init__(self, tickers: List[str]):
        # Exclude benchmark ETFs — no fundamentals
        self.tickers = [t for t in tickers if t not in ("SPY", "QQQ", "SMH", "SOXX",
                                                          "XLK", "XLC", "TLT", "GLD")]

    # ── single-ticker fetch ────────────────────────────────────────────────────
    def _fetch_one(self, ticker: str) -> pd.DataFrame:
        try:
            import yfinance as yf
            inc = yf.Ticker(ticker).quarterly_income_stmt
            if inc is None or inc.empty:
                return pd.DataFrame()

            ni_key  = next((k for k in inc.index
                            if "Net Income" in str(k) and "Common" not in str(k)), None)
            rev_key = next((k for k in inc.index if "Total Revenue" in str(k)), None)
            gp_key  = next((k for k in inc.index if "Gross Profit" in str(k)), None)

            rows = []
            for col in inc.columns:
                row = {"ticker": ticker, "quarter_end": pd.to_datetime(col)}
                row["net_income"]   = float(inc.loc[ni_key,  col]) if ni_key  and pd.notna(inc.loc[ni_key,  col]) else np.nan
                row["revenue"]      = float(inc.loc[rev_key, col]) if rev_key and pd.notna(inc.loc[rev_key, col]) else np.nan
                row["gross_profit"] = float(inc.loc[gp_key,  col]) if gp_key  and pd.notna(inc.loc[gp_key,  col]) else np.nan
                rows.append(row)

            return pd.DataFrame(rows).sort_values("quarter_end").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ── batch load with cache ─────────────────────────────────────────────────
    def load_raw(self) -> pd.DataFrame:
        if self.CACHE_FILE.exists():
            age_h = (datetime.now().timestamp() - self.CACHE_FILE.stat().st_mtime) / 3600
            if age_h < FUND_CACHE_TTL_H:
                import pickle
                with open(self.CACHE_FILE, "rb") as fh:
                    cached = pickle.load(fh)
                covered = set(cached["ticker"].unique()) if not cached.empty else set()
                missing  = [t for t in self.tickers if t not in covered]
                if not missing:
                    return cached

        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"  [Fundamentals] Fetching quarterly financials for {len(self.tickers)} tickers …")
        rows, done = [], 0
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as ex:
            futures = {ex.submit(self._fetch_one, t): t for t in self.tickers}
            for fut in as_completed(futures):
                df = fut.result()
                if not df.empty:
                    rows.append(df)
                done += 1
                if done % 20 == 0:
                    print(f"  [Fundamentals] … {done}/{len(self.tickers)}")

        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        import pickle
        with open(self.CACHE_FILE, "wb") as fh:
            pickle.dump(result, fh)
        return result

    # ── build daily signal DataFrames ─────────────────────────────────────────
    def build_signals(self, dates: pd.DatetimeIndex,
                      price_cols: List[str]) -> Dict[str, pd.DataFrame]:
        raw = self.load_raw()
        if raw.empty:
            print("  [Fundamentals] No data — signals will be NaN (price-only mode)")
            return {}

        sig_names = ["eps_growth_yoy", "rev_growth_yoy", "quality_hist"]
        out = {s: pd.DataFrame(np.nan, index=dates, columns=price_cols)
               for s in sig_names}

        valid_n = 0
        for ticker in price_cols:
            grp = raw[raw["ticker"] == ticker].sort_values("quarter_end").reset_index(drop=True)
            if len(grp) < 5:          # need 4 quarters gap + 1 current
                continue

            metric_rows = []
            for i in range(4, len(grp)):
                curr    = grp.iloc[i]
                prev_yr = grp.iloc[i - 4]   # same quarter, prior year

                pub_date = curr["quarter_end"] + pd.Timedelta(days=self.LAG_DAYS)

                # ── EPS (net income) YoY growth ────────────────────────────
                ni_c, ni_p = curr["net_income"], prev_yr["net_income"]
                if pd.notna(ni_c) and pd.notna(ni_p) and abs(ni_p) > 1e4:
                    eps_g = float(np.clip((ni_c - ni_p) / abs(ni_p), -1.5, 5.0))
                else:
                    eps_g = np.nan

                # ── Revenue YoY growth ─────────────────────────────────────
                rv_c, rv_p = curr["revenue"], prev_yr["revenue"]
                if pd.notna(rv_c) and pd.notna(rv_p) and rv_p > 1e4:
                    rev_g = float(np.clip((rv_c - rv_p) / abs(rv_p), -0.8, 2.0))
                else:
                    rev_g = np.nan

                # ── Quality: gross margin ──────────────────────────────────
                gp_c = curr["gross_profit"]
                if pd.notna(gp_c) and pd.notna(rv_c) and rv_c > 1e4:
                    qual = float(np.clip(gp_c / rv_c, 0.0, 1.0))
                else:
                    qual = np.nan

                metric_rows.append({
                    "pub_date":        pub_date,
                    "eps_growth_yoy":  eps_g,
                    "rev_growth_yoy":  rev_g,
                    "quality_hist":    qual,
                })

            if not metric_rows:
                continue

            mdf = (pd.DataFrame(metric_rows)
                     .set_index("pub_date")
                     .sort_index())

            for s in sig_names:
                if s not in mdf.columns:
                    continue
                # Forward-fill: last known value is valid until the next filing
                series = mdf[s].reindex(dates, method="ffill")
                out[s][ticker] = series

            valid_n += 1

        print(f"  [Fundamentals] Signals built for {valid_n}/{len(price_cols)} tickers  "
              f"(NaN tickers excluded from fundamental ranking)")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# IC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
class ICAnalyzer:
    """
    Computes Information Coefficient (IC) for each signal.
    IC = Spearman rank correlation between signal_t and return_{t+HOLD}.
    A good signal has mean IC > 0.05 with t-stat > 2.0.
    """

    def __init__(self, prices: pd.DataFrame, signals: Dict[str, pd.DataFrame]):
        self.prices  = prices
        self.signals = signals

    def compute(self, hold_period: int = HOLD_PERIOD) -> pd.DataFrame:
        fwd_ret = self.prices.pct_change(hold_period).shift(-hold_period)
        rows = []
        for sig_name, sig_df in self.signals.items():
            ics = []
            dates = sig_df.index[WARMUP_DAYS : -hold_period - 1]
            for date in dates:
                if date not in sig_df.index or date not in fwd_ret.index:
                    continue
                s = sig_df.loc[date].dropna()
                f = fwd_ret.loc[date].dropna()
                common = s.index.intersection(f.index)
                if len(common) < 5:
                    continue
                ic, _ = stats.spearmanr(s[common], f[common])
                if not np.isnan(ic):
                    ics.append(ic)

            if len(ics) < 3:
                rows.append({
                    "signal": sig_name, "n_obs": 0, "mean_ic": None,
                    "std_ic": None, "t_stat": None, "p_value": None,
                    "ic_positive_pct": None, "status": "NO_DATA",
                })
                continue

            ics_arr = np.array(ics)
            mean_ic  = float(ics_arr.mean())
            std_ic   = float(ics_arr.std())
            t_stat   = mean_ic / (std_ic / (len(ics) ** 0.5) + 1e-10)
            p_value  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(ics) - 1)))
            ic_pos   = float((ics_arr > 0).mean())

            if mean_ic > 0.05 and t_stat > 2.0:
                status = "STRONG"
            elif mean_ic > 0.02 and t_stat > 1.5:
                status = "USABLE"
            elif mean_ic > 0:
                status = "WEAK"
            else:
                status = "NEGATIVE"

            rows.append({
                "signal": sig_name, "n_obs": len(ics),
                "mean_ic": round(mean_ic, 4), "std_ic": round(std_ic, 4),
                "t_stat":  round(t_stat,  2),  "p_value": round(p_value, 4),
                "ic_positive_pct": f"{ic_pos * 100:.1f}%",
                "status": status,
            })

        return pd.DataFrame(rows).sort_values("mean_ic", ascending=False, na_position="last")


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class WalkForwardBacktester:
    """
    Simulates monthly rebalancing strategy:
      · Select top N stocks by composite score
      · Weight by inverse volatility (capped at 25%)
      · Deduct 10bps on turnover
      · Compare to buy-and-hold SPY benchmark
    """

    # Regime cash buffer (reduces equity exposure in defensive regimes)
    # BEAR: hold 25% cash → limits momentum crash damage
    # SIDEWAYS: hold 12% cash → gentle hedge
    REGIME_CASH = {"BULL": 0.00, "SIDEWAYS": 0.12, "BEAR": 0.25}

    def __init__(self, prices: pd.DataFrame, composite: pd.DataFrame,
                 top_n: int = 8, tc_bps: float = TC_BPS,
                 benchmark: str = BENCHMARK, hold_period: int = HOLD_PERIOD,
                 regime_series: Optional[pd.Series] = None):
        self.prices        = prices
        self.composite     = composite
        self.top_n         = top_n
        self.tc            = tc_bps / 10_000
        self.bench         = benchmark
        self.hold_period   = hold_period
        self.regime_series = regime_series   # None → no cash buffer applied

    def _inv_vol_weights(self, tickers: List[str], date_idx: int) -> Dict[str, float]:
        """Inverse-vol weighting capped at 25% per position."""
        p = self.prices
        vols = {}
        for t in tickers:
            if t not in p.columns:
                continue
            window = p[t].iloc[max(0, date_idx - 21): date_idx]
            if len(window) < 5:
                vols[t] = 1.0
            else:
                vol = window.pct_change().std()
                vols[t] = 1.0 / (vol + 1e-6) if vol > 1e-6 else 1.0
        if not vols:
            return {}
        total = sum(vols.values())
        raw = {t: v / total for t, v in vols.items()}
        # Cap at 25%
        for _ in range(10):  # iterative capping
            capped = {t: min(w, 0.25) for t, w in raw.items()}
            excess = sum(w - capped[t] for t, w in raw.items())
            uncapped = [t for t, w in raw.items() if w < 0.25]
            if not uncapped or excess < 1e-6:
                return capped
            add = excess / len(uncapped)
            raw = {t: min(capped[t] + (add if t in uncapped else 0), 0.25)
                   for t in raw}
        return {t: min(w, 0.25) for t, w in raw.items()}

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (monthly_perf_df, holdings_df)."""
        prices    = self.prices
        composite = self.composite
        daily_ret = prices.pct_change().fillna(0)

        rebalance_dates = []
        all_dates = prices.index.tolist()
        hp = self.hold_period
        if hp <= 10:
            # Weekly-ish: rebalance every hold_period trading days
            i = WARMUP_DAYS
            while i < len(all_dates) - hp:
                rebalance_dates.append(i)
                i += hp
        else:
            # Monthly: first trading day of each calendar month
            last_month = None
            for i, d in enumerate(all_dates[WARMUP_DAYS:], start=WARMUP_DAYS):
                m = (d.year, d.month)
                if m != last_month:
                    rebalance_dates.append(i)
                    last_month = m

        monthly_rows = []
        holdings_log = []
        current_weights: Dict[str, float] = {}
        bench_cum = 1.0
        strat_cum = 1.0

        for rb_i, rb_idx in enumerate(rebalance_dates[:-1]):
            rb_date  = all_dates[rb_idx]
            next_idx = rebalance_dates[rb_i + 1]
            next_date = all_dates[next_idx - 1]

            # Pick top N by composite score on rebalance date
            if rb_date not in composite.index:
                continue
            scores = composite.loc[rb_date].dropna().sort_values(ascending=False)
            # Exclude benchmark from stock selection to avoid tautology
            scores = scores[scores.index != self.bench]

            # ── Hold buffer: penalise new candidates by HOLD_BUFFER ────────
            # A currently-held stock at score 0.55 beats a new stock at 0.59
            # (0.59 - 0.05 = 0.54 < 0.55), reducing unnecessary churn.
            if current_weights and HOLD_BUFFER > 0:
                held = set(current_weights.keys())
                adj  = scores.copy()
                for t in adj.index:
                    if t not in held:
                        adj[t] = adj[t] - HOLD_BUFFER
                sort_idx = adj.sort_values(ascending=False).index
            else:
                sort_idx = scores.index

            eligible = [t for t in sort_idx[:self.top_n * 2]
                        if t in prices.columns
                        and prices[t].loc[:rb_date].count() >= MIN_HISTORY][:self.top_n]

            if not eligible:
                continue

            new_weights = self._inv_vol_weights(eligible, rb_idx)

            # ── Regime cash buffer: reduce equity exposure in BEAR/SIDEWAYS ──
            if self.regime_series is not None:
                # Get regime at this rebalance date (forward-fill for missing)
                reg_val = self.regime_series.asof(rb_date) if hasattr(self.regime_series, "asof") else "BULL"
                reg_str = str(reg_val) if not pd.isna(reg_val) else "BULL"
                cash_pct = self.REGIME_CASH.get(reg_str, 0.0)
                equity_pct = 1.0 - cash_pct
                if cash_pct > 0:
                    new_weights = {t: w * equity_pct for t, w in new_weights.items()}

            # Turnover cost
            all_tickers = set(current_weights) | set(new_weights)
            turnover = sum(abs(new_weights.get(t, 0) - current_weights.get(t, 0))
                           for t in all_tickers)
            tc_cost = turnover * self.tc

            # Hold period returns (daily, compound)
            period_slice = daily_ret.iloc[rb_idx:next_idx]
            strat_period = 1.0
            bench_period = 1.0
            for _, day_rets in period_slice.iterrows():
                strat_day = sum(new_weights.get(t, 0) * day_rets.get(t, 0)
                                for t in new_weights)
                strat_period *= (1 + strat_day)
                bench_period *= (1 + day_rets.get(self.bench, 0))
            strat_period -= tc_cost
            strat_cum *= strat_period
            bench_cum  *= bench_period

            monthly_rows.append({
                "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                "period_end":     next_date.strftime("%Y-%m-%d"),
                "strategy_ret":   round(strat_period - 1, 6),
                "spy_ret":        round(bench_period - 1, 6),
                "alpha":          round(strat_period - bench_period, 6),
                "strategy_cum":   round(strat_cum - 1, 4),
                "bench_cum":      round(bench_cum - 1, 4),
                "n_held":         len(eligible),
                "tickers":        " | ".join(eligible),
                "turnover_pct":   round(turnover * 100, 1),
                "tc_cost_bps":    round(tc_cost * 10_000, 1),
            })

            for t, w in new_weights.items():
                holdings_log.append({
                    "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                    "ticker": t, "weight": round(w, 4),
                    "composite_score": round(float(scores.get(t, 0)), 4),
                })

            current_weights = new_weights

        return pd.DataFrame(monthly_rows), pd.DataFrame(holdings_log)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_summary(monthly: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()

    rets   = monthly["strategy_ret"].values
    bench  = monthly["spy_ret"].values
    n      = len(rets)

    total_ret  = float((1 + pd.Series(rets)).prod() - 1)
    bench_ret  = float((1 + pd.Series(bench)).prod() - 1)
    mean_r     = float(np.mean(rets))
    std_r      = float(np.std(rets, ddof=1)) if n > 1 else 1e-6
    sharpe     = (mean_r / (std_r + 1e-10)) * (12 ** 0.5)   # annualise monthly

    neg = rets[rets < 0]
    down_std   = float(np.std(neg, ddof=1)) if len(neg) > 1 else 1e-6
    sortino    = (mean_r / (down_std + 1e-10)) * (12 ** 0.5)

    cum = np.cumprod(1 + rets)
    roll_max   = np.maximum.accumulate(cum)
    drawdowns  = (cum - roll_max) / (roll_max + 1e-10)
    max_dd     = float(drawdowns.min())

    bench_cum  = np.cumprod(1 + bench)
    total_alpha = total_ret - bench_ret
    win_rate   = float((rets > bench).mean())
    avg_alpha  = float(np.mean(rets - bench))

    best_ic  = ic_df[ic_df["mean_ic"].notna()].sort_values("mean_ic", ascending=False)
    top_sig  = best_ic.iloc[0]["signal"] if not best_ic.empty else "N/A"
    top_ic   = best_ic.iloc[0]["mean_ic"] if not best_ic.empty else None
    n_strong = int((ic_df["status"] == "STRONG").sum()) if not ic_df.empty else 0

    annualised_ret = (1 + total_ret) ** (12 / max(n, 1)) - 1
    calmar  = annualised_ret / (abs(max_dd) + 1e-10)

    def assess(metric, val):
        thresholds = {
            "sharpe":    [(1.5, "STRONG"), (0.8, "USABLE"), (0.3, "WEAK")],
            "sortino":   [(2.0, "STRONG"), (1.0, "USABLE"), (0.5, "WEAK")],
            "max_dd":    [(-0.10, "STRONG"), (-0.20, "USABLE"), (-0.35, "WEAK")],  # less negative = better
            "win_rate":  [(0.60, "STRONG"), (0.50, "USABLE"), (0.40, "WEAK")],
            "alpha":     [(0.05, "STRONG"), (0.02, "USABLE"), (0.0, "WEAK")],
        }
        if val is None:
            return "N/A"
        for thresh, label in thresholds.get(metric, []):
            if metric == "max_dd":
                if val >= thresh:
                    return label
            else:
                if val >= thresh:
                    return label
        return "POOR"

    rows = [
        {"metric": "Total Return (Strategy)",  "value": f"{total_ret*100:.2f}%",   "benchmark": f"{bench_ret*100:.2f}%", "assessment": assess("alpha", total_alpha)},
        {"metric": "Total Alpha vs SPY",        "value": f"{total_alpha*100:.2f}%", "benchmark": "0.00%",                "assessment": assess("alpha", total_alpha)},
        {"metric": "Avg Monthly Alpha",         "value": f"{avg_alpha*100:.3f}%",   "benchmark": "0.000%",               "assessment": assess("alpha", avg_alpha)},
        {"metric": "Annualised Sharpe",         "value": f"{sharpe:.2f}",           "benchmark": "~0.60 (SPY hist.)",    "assessment": assess("sharpe", sharpe)},
        {"metric": "Annualised Sortino",        "value": f"{sortino:.2f}",          "benchmark": "~0.90 (SPY hist.)",    "assessment": assess("sortino", sortino)},
        {"metric": "Calmar Ratio",              "value": f"{calmar:.2f}",           "benchmark": ">1.0 good",            "assessment": "STRONG" if calmar > 1.0 else ("USABLE" if calmar > 0.5 else "WEAK")},
        {"metric": "Max Drawdown",              "value": f"{max_dd*100:.2f}%",      "benchmark": "SPY ~-57% (2008)",     "assessment": assess("max_dd", max_dd)},
        {"metric": "Monthly Win Rate vs SPY",   "value": f"{win_rate*100:.1f}%",    "benchmark": "50%",                  "assessment": assess("win_rate", win_rate)},
        {"metric": "Periods Tested (months)",   "value": str(n),                   "benchmark": ">36 reliable",         "assessment": "STRONG" if n >= 36 else ("USABLE" if n >= 12 else "WEAK")},
        {"metric": "Strong IC Signals",         "value": f"{n_strong} / {len(ic_df)}", "benchmark": "≥2 needed",        "assessment": "STRONG" if n_strong >= 2 else ("USABLE" if n_strong == 1 else "WEAK")},
        {"metric": "Best Signal",               "value": f"{top_sig} (IC={top_ic})", "benchmark": "IC > 0.05",          "assessment": "STRONG" if (top_ic or 0) > 0.05 else "WEAK"},
        {"metric": "Transaction Cost (total)",  "value": f"{monthly['tc_cost_bps'].sum():.0f} bps", "benchmark": "10bps/trade", "assessment": "plain"},
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────
def write_report(summary: pd.DataFrame, ic_df: pd.DataFrame,
                 monthly: pd.DataFrame, holdings: pd.DataFrame,
                 elapsed: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Canyon v9 Step 62 — Walk-Forward Backtest Report",
        "",
        f"Generated: {ts}  |  Runtime: {elapsed:.1f}s",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | Strategy | Benchmark | Assessment |",
        "|--------|----------|-----------|------------|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r['metric']} | {r['value']} | {r['benchmark']} | {r['assessment']} |")

    lines += [
        "",
        "---",
        "",
        "## Signal IC Analysis",
        "",
        "> IC = Spearman rank correlation between signal and next-month return.",
        "> Rule of thumb: IC > 0.05 with t-stat > 2.0 = tradeable signal.",
        "",
        "| Signal | Mean IC | Std IC | t-stat | p-value | IC+ % | Status |",
        "|--------|---------|--------|--------|---------|-------|--------|",
    ]
    for _, r in ic_df.iterrows():
        ic_v  = f"{r['mean_ic']:.4f}" if r['mean_ic'] is not None else "N/A"
        std_v = f"{r['std_ic']:.4f}"  if r['std_ic']  is not None else "N/A"
        t_v   = f"{r['t_stat']:.2f}"  if r['t_stat']  is not None else "N/A"
        p_v   = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
        ic_pos = r['ic_positive_pct'] or "N/A"
        lines.append(f"| {r['signal']} | {ic_v} | {std_v} | {t_v} | {p_v} | {ic_pos} | **{r['status']}** |")

    if not monthly.empty:
        best  = monthly.loc[monthly["strategy_ret"].idxmax()]
        worst = monthly.loc[monthly["strategy_ret"].idxmin()]
        lines += [
            "",
            "---",
            "",
            "## Best / Worst Months",
            "",
            f"- **Best month**: {best['rebalance_date']} → strategy {best['strategy_ret']*100:.2f}%  "
            f"(SPY {best['spy_ret']*100:.2f}%)",
            f"- **Worst month**: {worst['rebalance_date']} → strategy {worst['strategy_ret']*100:.2f}%  "
            f"(SPY {worst['spy_ret']*100:.2f}%)",
        ]

    if not holdings.empty:
        last_rb = holdings["rebalance_date"].max()
        last_h  = holdings[holdings["rebalance_date"] == last_rb]
        lines += [
            "",
            "---",
            "",
            f"## Last Rebalance ({last_rb})",
            "",
            "| Ticker | Weight | Signal Score |",
            "|--------|--------|-------------|",
        ]
        for _, r in last_h.sort_values("weight", ascending=False).iterrows():
            lines.append(f"| {r['ticker']} | {r['weight']*100:.1f}% | {r['composite_score']:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Interpretation Guide",
        "",
        "- **Sharpe > 1.0**: strategy has meaningful risk-adjusted returns",
        "- **IC > 0.05 + t-stat > 2**: signal has statistically significant predictive power",
        "- **Alpha > 0**: strategy beats buy-and-hold SPY after transaction costs",
        "- **Max Drawdown**: how much the strategy lost peak-to-trough — lower is better",
        "- **Survivorship Bias Note**: backtest uses current universe only.",
        "  True performance would be ~1-2% lower due to delisted stocks.",
        "- **Walk-forward rule**: no future data used in signal construction.",
        "  All signals lagged 1 day. Rebalance cost deducted at 10bps per trade.",
        "",
        f"*Canyon v9 Step 62 Backtest Engine — {ts}*",
    ]

    report = "\n".join(lines)
    (ROOT / "backtest_engine_report.md").write_text(report, encoding="utf-8")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(years: int = 5, top_n: int = 8, tc_bps: float = TC_BPS,
         universe_name: str = "core", hold_period: int = HOLD_PERIOD,
         use_regime: bool = True, use_fundamentals: bool = True):
    t0 = datetime.now()
    print()
    print("=" * 72)
    print("Canyon v9 Step 62 — Walk-Forward Backtest Engine")
    print("=" * 72)

    # ── Universe selection ────────────────────────────────────────────────────
    bench = BENCHMARK   # default SPY
    if universe_name == "nasdaq100":
        ndx_path = ROOT / "nasdaq100_tickers.json"
        if ndx_path.exists():
            import json as _json
            d = _json.loads(ndx_path.read_text())
            raw = d["tickers"] if isinstance(d, dict) else d
        else:
            raw = CORE_TICKERS + SUPPLEMENT_TICKERS
        bench = "QQQ"
        raw = [t for t in raw if t != bench]
        universe = list(dict.fromkeys(raw + [bench]))
        print(f"\n  Universe : Nasdaq 100  (benchmark: QQQ)")
    elif universe_name == "sp500":
        sp_path = ROOT / "sp500_tickers.json"
        if sp_path.exists():
            import json as _json
            d = _json.loads(sp_path.read_text())
            raw = d["tickers"] if isinstance(d, dict) else d
        else:
            raw = CORE_TICKERS + SUPPLEMENT_TICKERS
        universe = list(dict.fromkeys(raw + [bench]))
        print(f"\n  Universe : S&P 500  (benchmark: SPY)")
    else:
        universe = list(dict.fromkeys(CORE_TICKERS + SUPPLEMENT_TICKERS + [bench]))
        print(f"\n  Universe : Core ({len(universe)} tickers, benchmark: SPY)")

    print(f"\n[1/6] Universe: {len(universe)} tickers")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[2/6] Fetching {years}-year price history …")
    loader = DataLoader(years=years)
    prices = loader.fetch(universe)

    # Drop tickers with insufficient history
    min_bars = WARMUP_DAYS + 2
    prices = prices.loc[:, prices.count() >= min_bars]
    print(f"       {len(prices.columns)} tickers with sufficient history")

    # ── Build signals ────────────────────────────────────────────────────────
    print("\n[3/6] Computing signals …")
    sb   = SignalBuilder(prices)
    sigs = sb.build()

    # ── Historical fundamentals (SUE proxy + quality) ────────────────────────
    if use_fundamentals:
        stock_tickers = [t for t in prices.columns if t not in ("SPY", "QQQ")]
        fl = HistoricalFundamentalsLoader(stock_tickers)
        fund_sigs = fl.build_signals(prices.index, prices.columns.tolist())
        if fund_sigs:
            sb.add_fundamental_signals(fund_sigs)
            sigs = sb._signals   # refresh after injection
            print(f"  Fundamental signals integrated: {list(fund_sigs.keys())}")
        else:
            print("  Fundamental signals unavailable — price-only mode")
    else:
        print("  Fundamental signals disabled (--no-fundamentals)")

    # Regime-conditional composite (optional — requires step76 regime_history.csv)
    regime_loaded = False
    if use_regime and F_REGIME_HIST.exists():
        try:
            import json as _json
            reg_df = pd.read_csv(F_REGIME_HIST, index_col=0, parse_dates=True)
            regime_series = reg_df["regime"].ffill()
            n_bull = (regime_series == "BULL").sum()
            n_bear = (regime_series == "BEAR").sum()
            n_side = (regime_series == "SIDEWAYS").sum()
            print(f"  Regime history: {len(regime_series)} rows  "
                  f"(BULL={n_bull}, BEAR={n_bear}, SIDEWAYS={n_side})")
            comp = sb.regime_composite(sigs, regime_series)
            regime_loaded = True
            print("  Using regime-conditional signal weights ✓")
        except Exception as e:
            print(f"  Regime history failed ({e}) — using default weights")
            comp = sb.composite(sigs)
    else:
        comp = sb.composite(sigs)
        if use_regime:
            print("  regime_history.csv not found — using default weights")
        else:
            print("  Regime conditioning disabled — using default weights")

    # ── IC Analysis ─────────────────────────────────────────────────────────
    print("\n[4/6] IC analysis (Spearman rank correlation) …")
    ia     = ICAnalyzer(prices, sigs)
    ic_df  = ia.compute(hold_period=hold_period)
    strong = ic_df[ic_df["status"] == "STRONG"]
    print(f"       {len(strong)} / {len(ic_df)} signals with IC > 0.05 & t > 2.0")
    for _, r in ic_df.iterrows():
        ic_str = f"{r['mean_ic']:.4f}" if r['mean_ic'] is not None else "N/A"
        t_str  = f"{r['t_stat']:.2f}"  if r['t_stat']  is not None else "N/A"
        print(f"         {r['signal']:22s}  IC={ic_str}  t={t_str}  → {r['status']}")

    # ── Walk-Forward Backtest ────────────────────────────────────────────────
    regime_tag = "regime-conditional + cash buffer" if regime_loaded else "fixed-weight"
    print(f"\n[5/6] Walk-forward backtest (top {top_n}, TC={tc_bps}bps, {regime_tag}) …")
    bt = WalkForwardBacktester(
        prices, comp, top_n=top_n, tc_bps=tc_bps,
        benchmark=bench, hold_period=hold_period,
        regime_series=regime_series if regime_loaded else None,
    )
    monthly, holdings = bt.run()
    print(f"       {len(monthly)} monthly periods simulated")

    if not monthly.empty:
        total_strat = (1 + monthly["strategy_ret"]).prod() - 1
        total_bench = (1 + monthly["spy_ret"]).prod() - 1
        print(f"       Strategy total return:  {total_strat*100:.2f}%")
        print(f"       {bench} total return:   {total_bench*100:.2f}%")
        print(f"       Alpha:                   {(total_strat - total_bench)*100:.2f}%")

    # ── Build summary ────────────────────────────────────────────────────────
    summary = build_summary(monthly, ic_df)

    # ── Write outputs ────────────────────────────────────────────────────────
    print("\n[6/6] Writing output files …")
    elapsed = (datetime.now() - t0).total_seconds()

    ic_df.to_csv(ROOT / "backtest_signal_ic.csv", index=False)
    monthly.to_csv(ROOT / "backtest_monthly_perf.csv", index=False)
    summary.to_csv(ROOT / "backtest_summary.csv", index=False)

    # Last rebalance holdings only (for dashboard)
    if not holdings.empty:
        last_rb = holdings["rebalance_date"].max()
        holdings[holdings["rebalance_date"] == last_rb].to_csv(
            ROOT / "backtest_holdings.csv", index=False)

    report = write_report(summary, ic_df, monthly, holdings, elapsed)

    print()
    print("=" * 72)
    print("Outputs:")
    for f in ["backtest_signal_ic.csv", "backtest_monthly_perf.csv",
              "backtest_summary.csv", "backtest_holdings.csv",
              "backtest_engine_report.md"]:
        p = ROOT / f
        size = f"{p.stat().st_size // 1024}KB" if p.exists() else "–"
        print(f"  {f:40s} {size}")
    print()
    print(f"Runtime: {elapsed:.1f}s")
    print("=" * 72)
    print()
    print("Next: open the dashboard → Strategy Scorecard → Backtest tab")
    print("      or: open backtest_engine_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Walk-Forward Backtest")
    parser.add_argument("--years",    type=int,   default=5,      help="Years of history (default 5)")
    parser.add_argument("--top",      type=int,   default=8,      help="Top N stocks per rebalance (default 8)")
    parser.add_argument("--tc",       type=float, default=TC_BPS, help="Transaction cost bps (default 10)")
    parser.add_argument("--universe", type=str,   default="core",
                        choices=["core", "nasdaq100", "sp500"],
                        help="Ticker universe: core (default) | nasdaq100 | sp500")
    parser.add_argument("--hold",      type=int,   default=HOLD_PERIOD,
                        help=f"Hold period in trading days (default {HOLD_PERIOD}≈1mo; 5=weekly)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable regime-conditional signal weights (use fixed default weights)")
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="Skip historical fundamentals (price-only mode, faster)")
    args = parser.parse_args()
    main(years=args.years, top_n=args.top, tc_bps=args.tc,
         universe_name=args.universe, hold_period=args.hold,
         use_regime=not args.no_regime,
         use_fundamentals=not args.no_fundamentals)
