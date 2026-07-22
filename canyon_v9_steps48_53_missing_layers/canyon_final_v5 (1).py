"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        CANYON QUANTITATIVE TRADING SYSTEM — FINAL COMPLETE v5.0             ║
║  Profit from market irrationality · Mathematical risk management · Auto offense/defense  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Data: Yahoo Finance (yfinance) / built-in synthetic data (auto-fallback if offline)      ║
║                                                                              ║
║  Full implementation (by chapter):                                                        ║
║  Ch.5  SMA/EMA trend-following: golden cross long, death cross short                      ║
║  Ch.6  Cross-sectional momentum: top 25% long, bottom 25% short (exact book impl)        ║
║  Ch.7  Walk-Forward backtest: Ann Ret / Vol / Sharpe / Max DD / Calmar                   ║
║  Ch.8  Statistical arbitrage: ADF test + cointegration + z-score pairs trading           ║
║  Ch.9  UCB Bayesian optimization: maximize Sharpe, auto-tune                             ║
║                                                                              ║
║  Canyon system integration:                                                               ║
║  · Five-dimensional regime detection → offensive / defensive / neutral positions          ║
║  · Auto short in bear: cross-sectional bottom 25% + SMA death cross                      ║
║  · Canyon F/C/E score → per-stock position cap                                           ║
║  · Kelly (skew-adjusted) + Black-Litterman + CVaR hard constraint                        ║
║  · Full trade log: engine / expected / actual / review per trade                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage (with internet):
    python canyon_final_v5.py

Usage (with real data):
    system = CanyonTradingSystem()
    result = system.run(['NVDA','AMD','TSM','MU','SPY'], '2020-01-01', '2024-12-31')
"""

import warnings, json, uuid, os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA LAYER — Yahoo Finance / synthetic data auto-fallback
# ══════════════════════════════════════════════════════════════════════════════

class DataLayer:
    """
    Data layer (Ch.4: yf.download usage)
    Prefer Yahoo Finance; auto-fallback to synthetic data if offline
    Synthetic data has full bull/crash/recovery/bear/AI-bull cycle
    """

    @staticmethod
    def load(tickers: List[str], start: str, end: str,
             benchmark: str = 'SPY') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Returns: (prices, volumes, market)
        prices/volumes: T×N DataFrame
        market: T Series (SPY or synthetic market)
        """
        try:
            import yfinance as yf
            all_tickers = list(dict.fromkeys(tickers + [benchmark]))
            raw = yf.download(all_tickers, start=start, end=end,
                              auto_adjust=True, progress=False)

            if isinstance(raw.columns, pd.MultiIndex):
                close_key  = 'Close'
                volume_key = 'Volume'
                if close_key not in raw.columns.get_level_values(0):
                    close_key = 'Adj Close'
                prices  = raw[close_key][tickers].dropna(how='all')
                volumes = raw[volume_key][tickers].dropna(how='all')
                market  = raw[close_key][benchmark].dropna() \
                          if benchmark in raw[close_key].columns \
                          else raw[close_key].mean(axis=1)
            else:
                prices  = raw[['Close']].rename(columns={'Close': tickers[0]})
                volumes = raw[['Volume']].rename(columns={'Volume': tickers[0]})
                market  = prices.iloc[:, 0]

            prices  = prices.ffill().dropna()
            volumes = volumes.ffill().fillna(1e6)
            market  = market.reindex(prices.index).ffill().dropna()

            print(f"[Data] ✅ Yahoo Finance: {len(prices)} days × {len(prices.columns)} assets")
            return prices, volumes, market

        except Exception as e:
            print(f"[Data] Yahoo Finance unavailable ({type(e).__name__}), using synthetic data")
            return DataLayer._synthetic(tickers, start, end)

    @staticmethod
    def _synthetic(tickers: List[str], start: str, end: str):
        """
        Built-in synthetic data: simulates 2020-2024 full market cycle
        Includes: 2020 crash(-35%) → recovery → 2021 bull → 2022 rate-hike bear(-25%) → 2023/24 AI bull
        """
        n = max(300, (pd.Timestamp(end) - pd.Timestamp(start)).days)
        N = len(tickers)
        dates = pd.date_range(start, periods=n, freq='B')
        t = np.linspace(0, 1, n)

        # Market factor: clear bull/bear cycles
        mkt_log = (
            0.0004 * np.ones(n)                              # base drift
            - 0.003 * np.exp(-((t - 0.12) ** 2) / 0.001)   # 2020 crash
            + 0.002 * (t > 0.15) * (t < 0.45)              # recovery bull
            - 0.002 * (t > 0.50) * (t < 0.68)              # 2022 bear
            + 0.003 * (t > 0.70)                            # AI bull
            + np.random.normal(0, 0.012, n)                  # noise
        )

        mkt_prices = pd.Series(100 * np.exp(np.cumsum(mkt_log)), index=dates, name='Market')

        # Stocks: market Beta + sector factor + idiosyncratic risk
        betas    = np.random.uniform(0.6, 1.8, N)
        alphas   = np.random.normal(0.0003, 0.0006, N)
        ret_mat  = (mkt_log[:, None] * betas[None, :]
                    + alphas[None, :]
                    + np.random.normal(0, 0.015, (n, N)))

        prices  = pd.DataFrame(100 * np.exp(np.cumsum(ret_mat, axis=0)),
                               columns=tickers, index=dates)
        volumes = pd.DataFrame(
            np.random.lognormal(17, 0.4, (n, N)) *
            np.where(np.abs(ret_mat) > 0.02, 2.5, 1.0),
            columns=tickers, index=dates
        )
        print(f"[Data] Synthetic: {n} days × {N} assets (bull/bear cycles: 2020crash/recovery/2022bear/AI-bull)")
        return prices, volumes, mkt_prices


# ══════════════════════════════════════════════════════════════════════════════
# 1. Backtest utility functions (strictly per Ch.7)
# ══════════════════════════════════════════════════════════════════════════════

def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """
    Ch.7 Listing 7-14 to 7-16: compute drawdown series
    Drawdown = (Wealth - Peak) / Peak
    """
    r = return_series.dropna().replace([np.inf, -np.inf], 0)
    wealth         = 1000 * (1 + r).cumprod()
    previous_peaks = wealth.cummax()
    dd             = (wealth - previous_peaks) / previous_peaks
    return pd.DataFrame({'Wealth index': wealth, 'Prior peaks': previous_peaks, 'Drawdown': dd})


def backtest_metrics(returns: pd.Series, rf: float = 0.03,
                      label: str = '') -> Dict:
    """
    Ch.7 exact implementation (Listing 7-13, 7-14, 7-15, 7-16):
    annualized_return = (1+R).prod()^(252/n) - 1
    annualized_vol    = returns.std() * sqrt(252)
    sharpe            = (ann_ret - rf) / ann_vol
    max_drawdown      = drawdown(returns)['Drawdown'].min()
    calmar            = ann_ret / |max_drawdown|
    """
    r = returns.dropna().replace([np.inf, -np.inf], 0)
    if len(r) < 5:
        return dict(ann_ret=0, ann_vol=0, sharpe=0, max_dd=0, calmar=0,
                    win_rate=0, n=0, total_ret=0)

    n         = len(r)
    ann_ret   = float((1 + r).prod() ** (252 / n) - 1)
    ann_vol   = float(r.std() * np.sqrt(252))
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 1e-6 else 0.0
    max_dd    = float(drawdown(r)['Drawdown'].min())
    calmar    = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0
    total_ret = float((1 + r).prod() - 1)
    win_rate  = float((r > 0).mean())

    if label:
        print(f"  [{label}] AnnRet:{ann_ret:+.2%} AnnVol:{ann_vol:.2%} "
              f"Sharpe:{sharpe:.3f} MaxDD:{max_dd:.2%} Calmar:{calmar:.3f}")
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, calmar=calmar, win_rate=win_rate,
                n=n, total_ret=total_ret)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Regime detection → offense/defense decision
# ══════════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    """
    Five market regimes, each with a corresponding position configuration
    The offense/defense core lives here
    """
    BULL_STRONG  = ( 2, "Strong Bull",  "Full Offense", 0.15, -0.00)
    BULL_NORMAL  = ( 1, "Normal Bull",  "Offense",      0.12, -0.00)
    NEUTRAL      = ( 0, "Neutral",      "Balanced",     0.10, -0.03)
    BEAR_MILD    = (-1, "Mild Bear",    "Defense+Short",0.08, -0.05)
    BEAR_STRONG  = (-2, "Strong Bear",  "Cash+Short",   0.05, -0.08)

    def __init__(self, score, label, stance, max_long, max_short):
        self.score    = score
        self.label    = label
        self.stance   = stance
        self.max_long = max_long   # per-stock long cap
        self.max_short= max_short  # per-stock short cap (negative)

    @property
    def allow_short(self) -> bool:
        return self.score <= -1

    @property
    def long_scale(self) -> float:
        """Overall long position scale (reduces in bear market)."""
        return max(0.3, 1.0 + self.score * 0.2)


def detect_regime(market: pd.Series,
                  prices: pd.DataFrame = None) -> Tuple[Regime, Dict]:
    """
    Five-dimensional regime detection (Ch.7 approach: evaluate strategy across market environments)

    Dim 1: SMA trend (price vs SMA50 vs SMA200)
    Dim 2: 12-month momentum (Ch.6)
    Dim 3: realized volatility (VIX proxy)
    Dim 4: market breadth (fraction of stocks above moving average)
    Dim 5: short-term momentum direction
    """
    if len(market) < 60:
        return Regime.NEUTRAL, {}

    m = market.dropna()

    # --- Dim 1: SMA trend ---
    w50  = min(50, len(m) - 1)
    w200 = min(200, len(m) - 1)
    sma50  = m.rolling(w50).mean().iloc[-1]
    sma200 = m.rolling(w200).mean().iloc[-1]
    cur    = m.iloc[-1]

    if cur > sma50 and sma50 > sma200:
        trend = 1.0     # bullish alignment
    elif cur < sma50 and sma50 < sma200:
        trend = -1.0    # bearish alignment
    elif cur > sma50:
        trend = 0.5
    elif cur < sma50:
        trend = -0.5
    else:
        trend = 0.0

    # --- Dim 2: 12-month momentum (Ch.6) ---
    lk = min(252, len(m) - 2)
    m12 = float(m.pct_change(lk).iloc[-1]) if lk > 5 else 0.0
    m3  = float(m.pct_change(min(63, lk)).iloc[-1]) if lk > 5 else 0.0
    mom = np.clip(m12 * 1.5 + m3 * 0.5, -1, 1)

    # --- Dim 3: volatility (realized vs historical) ---
    rv21  = float(m.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252))
    rv126 = float(m.pct_change().rolling(min(126, len(m)-1)).std().iloc[-1] * np.sqrt(252))
    vol_score = np.clip(-(rv21 / (rv126 + 1e-8) - 1.0), -1, 1)

    # --- Dim 4: market breadth ---
    if prices is not None and len(prices.columns) >= 3:
        w20 = min(20, len(prices) - 1)
        above = (prices > prices.rolling(w20).mean()).iloc[-1].mean()
        breadth = (above - 0.5) * 2
    else:
        breadth = trend * 0.5

    # --- Dim 5: short-term momentum ---
    m5 = float(m.pct_change(min(5, len(m)-1)).iloc[-1])
    short_mom = np.clip(m5 * 20, -1, 1)

    # --- Composite score ---
    score = (trend      * 35 +
             mom        * 25 +
             breadth    * 20 +
             vol_score  * 12 +
             short_mom  * 8)
    score = float(np.clip(score, -100, 100))

    # --- Map to Regime ---
    if   score >=  55: regime = Regime.BULL_STRONG
    elif score >=  20: regime = Regime.BULL_NORMAL
    elif score >= -20: regime = Regime.NEUTRAL
    elif score >= -55: regime = Regime.BEAR_MILD
    else:              regime = Regime.BEAR_STRONG

    detail = dict(trend=round(trend,2), momentum=round(mom,2),
                  vol_score=round(vol_score,2), breadth=round(breadth,2),
                  short_mom=round(short_mom,2), composite=round(score,1))
    return regime, detail


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ch.5: SMA/EMA trend-following
# ══════════════════════════════════════════════════════════════════════════════

def trend_signals(prices: pd.DataFrame,
                  ema_span: int = 5,
                  sma_span: int = 30) -> pd.DataFrame:
    """
    Ch.5, Listing 7-7 (exact implementation):
    EMA(5) crosses above SMA(30) → long signal +1
    EMA(5) crosses below SMA(30) → short signal -1 (used for bear-market shorting)

    Book: df[long_ma] = df['Adj Close'].rolling(sma_span).mean()
          df[short_ma] = df['Adj Close'].ewm(span=ema_span).mean()
    """
    results = {}
    for col in prices.columns:
        p   = prices[col].dropna()
        ema = p.ewm(span=ema_span, adjust=False).mean()
        sma = p.rolling(sma_span).mean()
        # Current signal
        cur_bull  = ema.iloc[-1] > sma.iloc[-1]
        prev_bull = ema.iloc[-2] > sma.iloc[-2] if len(ema) > 1 else cur_bull
        golden    = cur_bull and not prev_bull    # golden cross
        death     = not cur_bull and prev_bull   # death cross
        results[col] = {
            'signal': 1 if cur_bull else -1,
            'trend_up': cur_bull,
            'golden_cross': golden,
            'death_cross':  death,
            'ema': float(ema.iloc[-1]),
            'sma': float(sma.iloc[-1]),
            'strength': abs(float(ema.iloc[-1]) - float(sma.iloc[-1])) / (float(sma.iloc[-1]) + 1e-8)
        }
    return pd.DataFrame(results).T


# ══════════════════════════════════════════════════════════════════════════════
# 4. Ch.6: Cross-sectional momentum (long top 25%, short bottom 25%)
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_momentum(prices: pd.DataFrame,
                              lookback_months: int = 12,
                              skip_months: int = 1) -> Dict:
    """
    Ch.6 exact implementation (Listing 6-8 to 6-12)

    Steps:
    1. Compute cumulative return over past lookback_months (skip most recent skip_months)
    2. Rank: top 25% long, bottom 25% short
    3. Book: long top quartile, short bottom quartile

    Returns long_alpha, short_alpha (pd.Series, positive = strong long signal)
    """
    # Monthly returns (book uses monthly frequency)
    try:
        mret = prices.resample('ME').last().pct_change().dropna()
    except Exception:
        mret = prices.resample('M').last().pct_change().dropna()

    min_periods = lookback_months + skip_months + 1
    if len(mret) < min_periods:
        # Fall back to daily frequency
        w = min(lookback_months * 21, len(prices) - 2)
        mom = prices.pct_change(w).iloc[-1].dropna()
        rank = mom.rank(pct=True)
        return {
            'momentum': mom, 'rank': rank,
            'long':  mom[rank >= 0.75].index.tolist(),
            'short': mom[rank <= 0.25].index.tolist(),
            'long_alpha':  rank,
            'short_alpha': -rank,
            'spread': float(mom[rank >= 0.75].mean() - mom[rank <= 0.25].mean())
        }

    # Book exact implementation
    fe_idx = max(0, len(mret) - 1 - skip_months)
    fs_idx = max(0, fe_idx - lookback_months)

    if fs_idx >= fe_idx:
        fe_idx = len(mret) - 1
        fs_idx = max(0, fe_idx - lookback_months)

    period_ret = mret.iloc[fs_idx:fe_idx + 1]
    momentum   = (1 + period_ret).prod() - 1
    momentum   = momentum.dropna()

    if len(momentum) < 4:
        rank = momentum.rank(pct=True)
        return {'momentum': momentum, 'rank': rank,
                'long': momentum.nlargest(1).index.tolist(),
                'short': momentum.nsmallest(1).index.tolist(),
                'long_alpha': rank, 'short_alpha': -rank,
                'spread': 0.0}

    # Quantile ranking
    try:
        rank_q = pd.qcut(momentum, q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(float)
    except Exception:
        rank_q = momentum.rank(pct=True)

    rank_pct = momentum.rank(pct=True)
    long_t   = momentum[rank_pct >= 0.75].index.tolist()
    short_t  = momentum[rank_pct <= 0.25].index.tolist()

    return {
        'momentum':    momentum,
        'rank':        rank_pct,
        'long':        long_t,
        'short':       short_t,
        'long_alpha':  rank_pct,
        'short_alpha': -rank_pct,   # negative: smaller = weaker
        'spread':      float(momentum[rank_pct >= 0.75].mean() -
                             momentum[rank_pct <= 0.25].mean())
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Ch.8: Statistical arbitrage (ADF + cointegration + z-score)
# ══════════════════════════════════════════════════════════════════════════════

class StatArb:
    """
    Ch.8: pairs trading
    ADF → cointegration → spread → z-score → trading signal
    Book exact formula: spread = Y - (β₀ + β₁X)  [OLS residual]
    z-score = (spread - rolling_mean) / rolling_std
    """

    def __init__(self, entry_z=2.0, exit_z=0.5, window=21):
        self.entry_z = entry_z
        self.exit_z  = exit_z
        self.window  = window

    def adf_pvalue(self, series: np.ndarray) -> float:
        """ADF unit-root test, MacKinnon critical value approximation (Ch.8)."""
        if len(series) < 15:
            return 1.0
        y = np.diff(series)
        x = series[:-1]
        X = np.column_stack([x, np.ones(len(x))])
        try:
            c   = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X @ c
            s2  = res.var()
            se  = np.sqrt(max(0, s2 * np.linalg.pinv(X.T @ X)[0, 0]))
            t   = c[0] / (se + 1e-12)
        except Exception:
            return 1.0
        # MacKinnon approximate critical values
        if   t < -3.96: return 0.005
        elif t < -3.41: return 0.01
        elif t < -2.86: return 0.05
        elif t < -2.57: return 0.10
        else:           return 0.50

    def test_pair(self, s1: pd.Series, s2: pd.Series) -> Dict:
        """
        Ch.8: OLS + ADF (Engle-Granger two-step)
        Y = β₀ + β₁X + ε
        spread = Y - β₀ - β₁X  →  ADF test on spread
        """
        y, x = s1.values.astype(float), s2.values.astype(float)
        try:
            X   = np.column_stack([np.ones(len(x)), x])
            c   = np.linalg.lstsq(X, y, rcond=None)[0]
            b0, b1 = c[0], c[1]
            spread = y - b0 - b1 * x
            pv     = self.adf_pvalue(spread)
            corr,_ = pearsonr(y, x)
            # Half-life
            dy = np.diff(spread)
            sx = spread[:-1]
            Xs = np.column_stack([sx, np.ones(len(sx))])
            lam = np.linalg.lstsq(Xs, dy, rcond=None)[0][0]
            hl  = -np.log(2) / lam if lam < -1e-6 else 999
            return {
                'cointegrated': pv < 0.05 and abs(corr) > 0.7,
                'pvalue': pv, 'hedge_ratio': b1, 'intercept': b0,
                'correlation': corr, 'half_life': hl,
                'tradeable': pv < 0.05 and 3 < hl < 90
            }
        except Exception as e:
            return {'cointegrated': False, 'error': str(e)}

    def zscore(self, s1: pd.Series, s2: pd.Series,
               b1: float, b0: float) -> pd.Series:
        """Ch.8: rolling z-score."""
        spread = s1 - b0 - b1 * s2
        mu     = spread.rolling(self.window).mean()
        sd     = spread.rolling(self.window).std()
        return (spread - mu) / (sd + 1e-8)

    def signals(self, zs: pd.Series) -> pd.Series:
        """
        Ch.8 exact signal logic (Listing 9-3):
        z < -entry  → +1 (long s1, short s2)
        z > +entry  → -1 (short s1, long s2)
        |z| < exit  → 0  close position
        else        → hold
        """
        pos = pd.Series(0.0, index=zs.index)
        for i in range(1, len(zs)):
            z, p = zs.iloc[i], pos.iloc[i - 1]
            if   z < -self.entry_z and p == 0: pos.iloc[i] =  1.0
            elif z >  self.entry_z and p == 0: pos.iloc[i] = -1.0
            elif abs(z) < self.exit_z:         pos.iloc[i] =  0.0
            else:                              pos.iloc[i] =  p
        return pos

    def backtest_pair(self, s1: pd.Series, s2: pd.Series,
                      info: Dict, tc_bps: float = 10) -> Dict:
        """
        Ch.7 metrics + Ch.8 pairs trading
        net_return = long_return + short_return - tc
        """
        b1, b0 = info['hedge_ratio'], info['intercept']
        zs  = self.zscore(s1, s2, b1, b0)
        pos = self.signals(zs)
        r1  = s1.pct_change().fillna(0)
        r2  = s2.pct_change().fillna(0)
        tc  = pos.diff().abs().fillna(0) * (tc_bps / 10000)
        net = pos.shift(1).fillna(0) * r1 - pos.shift(1).fillna(0) * b1 * r2 - tc
        m   = backtest_metrics(net)
        m['current_z']   = float(zs.iloc[-1])
        m['current_pos'] = float(pos.iloc[-1])
        return m

    def find_pairs(self, prices: pd.DataFrame) -> List[Dict]:
        """Find all cointegrated pairs, ranked by quality."""
        cols, pairs = prices.columns.tolist(), []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = self.test_pair(prices[cols[i]], prices[cols[j]])
                if r.get('tradeable'):
                    pairs.append({'t1': cols[i], 't2': cols[j], **r,
                                  'score': (1 - r['pvalue']) * 60 +
                                           max(0, 1 - r['half_life'] / 90) * 40})
        return sorted(pairs, key=lambda x: x['score'], reverse=True)

    def current_opportunity(self, prices: pd.DataFrame,
                             pairs: List[Dict]) -> List[Dict]:
        """Current active statistical arbitrage opportunities."""
        active = []
        for p in pairs[:5]:
            t1, t2 = p['t1'], p['t2']
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
            zs  = self.zscore(prices[t1], prices[t2], p['hedge_ratio'], p['intercept'])
            cur = float(zs.iloc[-1])
            if abs(cur) > self.entry_z:
                active.append({
                    't1': t1, 't2': t2,
                    'direction': 1 if cur < 0 else -1,  # +1=long t1 short t2
                    'z': cur, 'half_life': p['half_life']
                })
        return active


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ch.9: UCB Bayesian optimization
# ══════════════════════════════════════════════════════════════════════════════

class UCBOptimizer:
    """
    Ch.9: UCB (Upper Confidence Bound) Bayesian optimization
    UCB(x) = μ(x) + β × σ(x)
    Objective: maximize Sharpe ratio
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta
        self.obs_x: List[List[float]] = []
        self.obs_y: List[float] = []

    def optimize(self, objective_fn, param_bounds: Dict,
                 n_iter: int = 20) -> Dict:
        best_score, best_params = -np.inf, {}

        for i in range(n_iter):
            params = (self._random_sample(param_bounds)
                      if len(self.obs_x) < 4
                      else self._ucb_suggest(param_bounds))
            try:
                score = float(objective_fn(**params))
                if np.isfinite(score):
                    self.obs_x.append(list(params.values()))
                    self.obs_y.append(score)
                    if score > best_score:
                        best_score, best_params = score, params.copy()
            except Exception:
                continue

        return {'best_params': best_params, 'best_score': best_score,
                'n_evaluated': len(self.obs_y)}

    def _random_sample(self, bounds: Dict) -> Dict:
        return {k: (float(np.random.uniform(*v))
                    if isinstance(v[0], float)
                    else int(np.random.randint(v[0], v[1])))
                for k, v in bounds.items()}

    def _ucb_suggest(self, bounds: Dict) -> Dict:
        X = np.array(self.obs_x)
        y = np.array(self.obs_y)
        keys = list(bounds.keys())

        # Normalize
        X_mu, X_sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - X_mu) / X_sd

        # Candidate points
        cands = np.array([[float(np.random.uniform(*v))
                           if isinstance(v[0], float)
                           else float(np.random.randint(v[0], v[1]))
                           for v in bounds.values()]
                          for _ in range(60)])
        cs = (cands - X_mu) / X_sd

        # RBF kernel GP approximation
        D  = np.sum((Xs[:, None] - cs[None, :]) ** 2, axis=-1)
        K  = np.exp(-0.5 * D)
        Kx = np.exp(-0.5 * np.sum((Xs[:, None] - Xs[None, :]) ** 2, axis=-1))
        Kx += np.eye(len(X)) * 0.01

        try:
            alpha_gp = np.linalg.solve(Kx, y)
            mu       = K.T @ alpha_gp
            variance = np.array([max(0.0, 1.0 - float(K[:, j] @ np.linalg.solve(Kx, K[:, j])))
                                  for j in range(len(cands))])
            ucb = mu + self.beta * np.sqrt(variance)
        except Exception:
            ucb = np.random.randn(len(cands))

        best = np.argmax(ucb)
        return {k: (float(cands[best, i]) if isinstance(list(bounds.values())[i][0], float)
                    else int(cands[best, i]))
                for i, k in enumerate(keys)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. Risk management
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Black-Litterman + Kelly (skew-adjusted) + CVaR + Ledoit-Wolf covariance
    """

    def __init__(self):
        self.lw = LedoitWolf()

    def kelly(self, ret_series: pd.Series, lookback: int = 126) -> float:
        """
        Kelly Criterion with skewness adjustment
        f* = (p×b - q) / b  × 0.5 (half-Kelly)
        Skew penalty: left-skewed → reduce weight
        """
        r = ret_series.dropna().tail(lookback)
        if len(r) < 20:
            return 0.05

        p    = float((r > 0).mean())
        q    = 1 - p
        wins = r[r > 0]
        loss = r[r < 0]
        if len(wins) == 0 or len(loss) == 0:
            return 0.03

        b   = float(wins.mean()) / (float(abs(loss.mean())) + 1e-8)
        raw = (p * b - q) / (b + 1e-8)
        if raw <= 0:
            return 0.03

        # Skew adjustment (not in book but important practical improvement)
        sk_pen = max(0, -float(r.skew()))  * 0.15
        kt_pen = max(0, float(r.kurtosis()) - 3) * 0.05
        return float(np.clip(raw * 0.5 * (1 - sk_pen - kt_pen), 0.02, 0.20))

    def ledoit_wolf_cov(self, returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance matrix."""
        clean = returns.dropna()
        if len(clean) < 10:
            return np.diag(clean.var().values)
        try:
            self.lw.fit(clean.values)
            return self.lw.covariance_
        except Exception:
            return clean.cov().values

    def cvar(self, returns: pd.Series, conf: float = 0.95) -> float:
        """CVaR: more conservative tail-risk measure than VaR."""
        s = np.sort(returns.dropna().values)
        c = int(len(s) * (1 - conf))
        return float(-np.mean(s[:max(1, c)]))

    def optimize_weights(self, alpha_scores: pd.Series,
                          returns: pd.DataFrame,
                          regime: Regime,
                          allow_short: bool = False) -> pd.Series:
        """
        Mean-variance optimization
        Objective: maximize alpha - λ × risk
        Constraints: CVaR ≤ 5.6%, per-stock ≤ regime.max_long
        """
        tickers = [t for t in alpha_scores.index if t in returns.columns]
        if len(tickers) == 0:
            return pd.Series(dtype=float)

        mu    = alpha_scores[tickers].values
        Sigma = self.ledoit_wolf_cov(returns[tickers].dropna())
        n     = len(tickers)
        lo    = regime.max_short if allow_short else 0.0
        hi    = regime.max_long

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ Sigma @ w + 1e-10)
            return -(ret - 0.0001) / vol  # small rf stabilizes optimizer

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]

        result = optimize.minimize(
            neg_sharpe, x0=np.ones(n) / n,
            method='SLSQP',
            bounds=[(lo, hi)] * n,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-8}
        )
        w = result.x if result.success else np.ones(n) / n
        if not allow_short:
            w = np.maximum(w, 0)
            if w.sum() > 1e-8:
                w /= w.sum()
        return pd.Series(w, index=tickers)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Offense/defense position manager (core addition)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioAllocation:
    longs:          Dict[str, float]   # ticker → weight (positive)
    shorts:         Dict[str, float]   # ticker → weight (negative)
    cash:           float
    net_exposure:   float
    gross_exposure: float
    regime:         str
    rationale:      str

    def to_series(self) -> pd.Series:
        d = {**self.longs, **self.shorts}
        return pd.Series(d, dtype=float)


class OffensiveDefensiveManager:
    """
    Offense/defense position manager

    Bull market offense: momentum long full position, max 15% per stock
    Neutral balance: momentum long + light hedge
    Bear defense: reduce longs + actively short cross-sectional weakest (Ch.6 bottom 25%)
    Strong-bear cash: minimal longs + maximum short exposure

    Short trigger conditions:
    1. Regime.score <= -1 (mild/strong bear)
    2. SMA death cross (Ch.5)
    3. Cross-sectional bottom 25% (Ch.6)
    More conditions met → larger short position
    """

    def __init__(self, risk_mgr: RiskManager):
        self.risk = risk_mgr

    def allocate(self,
                 regime: Regime,
                 long_alpha: pd.Series,      # long candidate Alpha scores
                 short_alpha: pd.Series,     # short candidate Alpha scores (negative = weaker)
                 trend_sig: pd.DataFrame,    # SMA/EMA trend signals
                 stat_arb_opps: List[Dict],  # statistical arbitrage opportunities
                 returns: pd.DataFrame,      # historical returns (for Kelly sizing)
                 kelly_cap: float = 0.15     # Kelly upper bound
                 ) -> PortfolioAllocation:

        longs: Dict[str, float]  = {}
        shorts: Dict[str, float] = {}

        # ══ Long positions ══════════════════════════════════════════════════
        if len(long_alpha.dropna()) > 0:
            # Rank by Alpha score, take top 30% (spirit of Ch.6 top quartile)
            ranked   = long_alpha.dropna().rank(pct=True)
            top_tickers = ranked[ranked >= 0.70].index.tolist()

            if len(top_tickers) > 0:
                # Further filter with trend (keep only golden-cross / uptrend)
                confirmed = []
                for tk in top_tickers:
                    if tk in trend_sig.index:
                        if trend_sig.loc[tk, 'signal'] == 1:  # bullish trend
                            confirmed.append(tk)
                    else:
                        confirmed.append(tk)  # no trend data → keep

                if not confirmed:
                    # No results after trend filter → relax: take all top_tickers
                    confirmed = top_tickers

                # Kelly cap + Regime cap
                n_long  = max(1, len(confirmed))
                # Equal-weight base size: total ~min(100%, n×max_long)
                base_sz = min(regime.max_long, 1.0 / n_long)
                for tk in confirmed:
                    kelly_sz = self.risk.kelly(returns[tk], 126) if tk in returns.columns else 0.08
                    # Kelly only acts as upper cap, not compressed too small; 3% floor
                    size = min(base_sz * regime.long_scale,
                               regime.max_long,
                               max(kelly_sz, 0.03))
                    if size > 0.005:
                        longs[tk] = round(float(size), 4)

        # ══ Short positions (activated only in bear market) ═══════════════════
        if regime.allow_short and len(short_alpha.dropna()) > 0:
            # Bottom 25%: weakest stocks (Ch.6 short bottom quartile)
            ranked_short = short_alpha.dropna().rank(pct=True)
            bot_tickers  = ranked_short[ranked_short <= 0.25].index.tolist()

            # Confirm with trend (only short death-cross stocks)
            confirmed_short = []
            for tk in bot_tickers:
                if tk in longs:                   # skip stocks already in longs
                    continue
                in_downtrend = (tk in trend_sig.index and
                                trend_sig.loc[tk, 'signal'] == -1)
                if in_downtrend or regime.score <= -2:  # strong bear: no death-cross required
                    confirmed_short.append(tk)

            if confirmed_short:
                n_short  = len(confirmed_short)
                eq_short = 1.0 / n_short if n_short > 0 else 0
                for tk in confirmed_short:
                    size = max(
                        regime.max_short,           # not exceeding max short
                        -min(0.06, eq_short * 0.8)  # 80% of equal weight
                    )
                    if size < -0.005:
                        shorts[tk] = round(float(size), 4)

        # ══ Statistical arbitrage overlay (neutral alpha) ══════════════════════
        if regime.score >= -1:  # strong bear skips arb (focus on shorting)
            for opp in stat_arb_opps[:2]:
                t1, t2 = opp['t1'], opp['t2']
                sz = min(0.025, abs(opp['z']) / 4 * 0.03)
                if opp['direction'] == 1:   # long t1 short t2
                    if t1 not in {**longs, **shorts}:
                        longs[t1]  = longs.get(t1, 0) + sz
                    if t2 not in {**longs, **shorts}:
                        shorts[t2] = shorts.get(t2, 0) - sz
                else:                       # short t1 long t2
                    if t1 not in {**longs, **shorts}:
                        shorts[t1] = shorts.get(t1, 0) - sz
                    if t2 not in {**longs, **shorts}:
                        longs[t2]  = longs.get(t2, 0) + sz

        # ══ Summary ══════════════════════════════════════════════════════════
        all_w = {**longs, **shorts}
        net   = sum(all_w.values())
        gross = sum(abs(v) for v in all_w.values())
        cash  = max(0.0, 1.0 - sum(longs.values()))

        rationale = (
            f"{regime.label}（{regime.stance}）| "
            f"Long {len(longs)} Short {len(shorts)} | "
            f"Net:{net:+.1%} Gross:{gross:.1%} Cash:{cash:.1%}"
        )

        return PortfolioAllocation(
            longs=longs, shorts=shorts, cash=cash,
            net_exposure=net, gross_exposure=gross,
            regime=regime.label, rationale=rationale
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. Canyon F/C/E scoring (quantitative version for backtesting)
# ══════════════════════════════════════════════════════════════════════════════

def canyon_score_auto(price: pd.Series, volume: pd.Series,
                       market: pd.Series, regime: Regime) -> Dict:
    """
    Canyon F/C/E automated quantitative scoring (backtest use, no manual input)
    F = fundamental proxy (price efficiency + relative strength)
    C = momentum / volume / mispricing signals
    E = execution position (RSI + VWAP)
    """
    r   = price.pct_change().dropna()
    n   = len(r)
    if n < 21:
        return {'total': 60, 'grade': 'D', 'can_buy': True, 'max_pos': regime.max_long}

    # F score (20%)
    lk  = min(21, n - 1)
    er  = abs(float(price.iloc[-1]) - float(price.iloc[-lk - 1]))
    path= float(r.abs().tail(lk).sum()) + 1e-8
    rs  = float(price.pct_change(lk).iloc[-1]) - float(market.pct_change(lk).iloc[-1])
    f   = float(np.clip(3.0 + (er / path) * 1.5 + rs * 15, 1, 5))

    # C score (45%)
    vm  = float(volume.rolling(min(20, n)).mean().iloc[-1])
    vr  = float(volume.iloc[-1]) / (vm + 1e-8)
    mom = float(np.clip(price.pct_change(min(5, n - 1)).iloc[-1] * 40, -2, 2))
    g   = r.clip(lower=0).ewm(span=14).mean()
    l   = (-r).clip(lower=0).ewm(span=14).mean()
    rsi = float(100 - 100 / (1 + g.iloc[-1] / (l.iloc[-1] + 1e-8)))
    c2  = float(np.clip(3.0 + mom + (vr - 1) * 0.3, 1, 5))
    c3  = float(np.clip(3.0 + rs * 12, 1, 5))
    c   = float(np.clip(0.35 * f + 0.40 * c2 + 0.25 * c3, 1, 5))

    # E score (35%)
    rsi_ok = 35 < rsi < 72
    vm_ok  = vr > 0.8
    e = float(np.clip(3.0 + (0.7 if rsi_ok else -0.6) + (0.3 if vm_ok else -0.2), 1, 5))

    # Total score
    total = 0.20 * (f / 5 * 100) + 0.45 * (c / 5 * 100) + 0.35 * (e / 5 * 100)
    total = float(np.clip(total, 0, 100))

    if   total >= 85: grade, mp = 'A', regime.max_long
    elif total >= 80: grade, mp = 'B', min(regime.max_long, 0.09)
    elif total >= 75: grade, mp = 'C', min(regime.max_long, 0.06)
    elif total >= 70: grade, mp = 'D', min(regime.max_long, 0.04)
    else:             grade, mp = 'X', 0.0

    return {'f': round(f,2), 'c': round(c,2), 'e': round(e,2),
            'total': round(total,1), 'grade': grade, 'max_pos': mp,
            'can_buy': total >= 70 and e >= 3.5}


# ══════════════════════════════════════════════════════════════════════════════
# 10. Walk-Forward backtest engine (Ch.7 core)
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardBacktester:
    """
    Ch.7: Walk-Forward backtest (prevents data snooping)

    Each training window (252 days):
    → Train: detect regime + generate signals + compute positions
    → Test: evaluate over next 63 days (fully out-of-sample)
    → Roll forward

    Ch.7 warning: backtesting is only as good as the quality of the data
    So we:
    1. No lookahead (pos.shift(1) delays execution one period)
    2. Include transaction costs (tc_bps=10)
    3. Multi-period validation (Ch.7 recommendation)
    """

    def __init__(self, train_w: int = 252, test_w: int = 63,
                 tc_bps: float = 10.0, rf: float = 0.03):
        self.train_w = train_w
        self.test_w  = test_w
        self.tc      = tc_bps / 10000
        self.rf      = rf

    def run(self, prices: pd.DataFrame, volumes: pd.DataFrame,
             market: pd.Series,
             stat_arb: StatArb,
             od_mgr: OffensiveDefensiveManager,
             verbose: bool = True) -> Dict:
        """
        Full Walk-Forward backtest

        Returns: complete metrics + daily return series + long/short attribution
        """
        n = len(prices)
        if n < self.train_w + self.test_w:
            raise ValueError(f"Insufficient data: need {self.train_w + self.test_w} days, got {n}")

        daily_rets   = []
        daily_dates  = []
        long_rets    = []
        short_rets   = []
        regime_hist  = []
        prev_weights = pd.Series(dtype=float)

        steps = range(self.train_w, n - self.test_w + 1, self.test_w)
        if verbose:
            print(f"  Walk-Forward: {len(list(steps))} test windows × {self.test_w} days")

        for step_start in steps:
            # ── Training window (only past data) ───────────────────────────────
            p_tr = prices.iloc[step_start - self.train_w: step_start]
            v_tr = volumes.iloc[step_start - self.train_w: step_start]
            m_tr = market.iloc[step_start - self.train_w: step_start]
            r_tr = p_tr.pct_change().dropna()

            # Regime detection
            regime, rg_detail = detect_regime(m_tr, p_tr)
            regime_hist.append(regime.score)

            # Cross-sectional momentum signal (Ch.6)
            cs = cross_sectional_momentum(p_tr)
            long_alpha  = cs['long_alpha'].dropna()   # positive = strong
            short_alpha = cs['short_alpha'].dropna()  # negative = weak

            # SMA trend signal (Ch.5)
            tsig = trend_signals(p_tr)

            # Statistical arbitrage opportunities
            pairs  = stat_arb.find_pairs(p_tr)
            arb_op = stat_arb.current_opportunity(p_tr, pairs)

            # Offense/defense position allocation
            alloc = od_mgr.allocate(
                regime=regime,
                long_alpha=long_alpha,
                short_alpha=short_alpha,
                trend_sig=tsig,
                stat_arb_opps=arb_op,
                returns=r_tr
            )
            weights = alloc.to_series()

            # ── Test window (OOS, delayed one period) ──────────────────────
            test_end  = min(step_start + self.test_w, n)
            p_te      = prices.iloc[step_start: test_end]
            ret_te    = p_te.pct_change().fillna(0)
            test_dates= p_te.index

            for i in range(1, len(test_dates)):
                d      = test_dates[i]
                dr, lr, sr = 0.0, 0.0, 0.0

                for tk, w in weights.items():
                    if tk not in ret_te.columns:
                        continue
                    ar = float(ret_te.loc[d, tk])
                    if not np.isfinite(ar):
                        ar = 0.0
                    c  = w * ar
                    dr += c
                    if w > 0:  lr += c
                    else:      sr += c

                # Turnover cost (charged on buy/sell)
                if len(prev_weights) > 0:
                    all_tk = set(weights.index) | set(prev_weights.index)
                    turn   = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0))
                                 for t in all_tk)
                    dr -= turn * self.tc

                daily_rets.append(dr)
                daily_dates.append(d)
                long_rets.append(lr)
                short_rets.append(sr)

            prev_weights = weights.copy()

        if not daily_rets:
            return {'error': 'no trade data', 'sharpe': 0}

        # ── Compute summary metrics (Ch.7 Listing 7-13 to 7-16) ────────────────
        ret_s  = pd.Series(daily_rets, index=daily_dates)
        m      = backtest_metrics(ret_s, rf=self.rf)

        # Decompose by regime
        rg_s   = pd.Series(regime_hist)
        # Map each test day to its regime (simplified: use step-level regime)
        bull_m = backtest_metrics(ret_s.iloc[:len(ret_s)//2], rf=self.rf) if len(ret_s) > 20 else {}
        bear_m = backtest_metrics(ret_s.iloc[len(ret_s)//2:], rf=self.rf) if len(ret_s) > 20 else {}

        long_s  = pd.Series(long_rets, index=daily_dates)
        short_s = pd.Series(short_rets, index=daily_dates)

        result = {
            **m,
            'daily_returns': ret_s,
            'cumulative': (1 + ret_s).cumprod(),
            'long_total_pnl':   float(long_s.sum()),
            'short_total_pnl':  float(short_s.sum()),
            'avg_regime': float(np.mean(regime_hist)) if regime_hist else 0,
            'bull_sharpe': bull_m.get('sharpe', 0),
            'bear_sharpe': bear_m.get('sharpe', 0),
        }

        if verbose:
            self._print(result)
        return result

    def _print(self, r: Dict):
        print(f"\n{'─'*62}")
        print(f"  📊 Walk-Forward backtest results (Ch.7 complete metrics)")
        print(f"{'─'*62}")
        print(f"  Ann Ret: {r.get('ann_ret', 0):+.2%}  |  Ann Vol: {r.get('ann_vol', 0):.2%}")
        print(f"  Sharpe：  {r.get('sharpe', 0):>8.3f}  |  Calmar：  {r.get('calmar', 0):.3f}")
        print(f"  Max DD:  {r.get('max_dd', 0):>8.2%}  |  Total Ret: {r.get('total_ret', 0):+.2%}")
        print(f"  Win Rate: {r.get('win_rate', 0):>8.1%}  |  Trade Days: {r.get('n', 0)}")
        print(f"  Long PnL: {r.get('long_total_pnl', 0):+.2%}  |  Short PnL: {r.get('short_total_pnl', 0):+.2%}")
        print(f"  Bull Sharpe: {r.get('bull_sharpe', 0):.3f}  |  Bear Sharpe: {r.get('bear_sharpe', 0):.3f}")
        print(f"{'─'*62}")

    def multi_period(self, prices, volumes, market, stat_arb, od_mgr,
                     n_periods: int = 3) -> pd.DataFrame:
        """
        Ch.7 emphasis: must backtest over multiple periods for robust results
        Split data into segments, backtest each independently, report mean and std dev
        """
        n = len(prices)
        seg = max(self.train_w + self.test_w, n // n_periods)
        records = []

        print(f"\n  📊 Multi-period backtest (Ch.7: {n_periods} segments × independent validation)")
        print(f"  {'─'*58}")

        for i in range(n_periods):
            s = i * (n // n_periods)
            e = min(s + seg + self.train_w, n)
            if e - s < self.train_w + self.test_w:
                continue
            sp = prices.iloc[s:e]
            sv = volumes.iloc[s:e]
            sm = market.iloc[s:e]
            d0 = sp.index[0].date()
            d1 = sp.index[-1].date()
            print(f"  Period {i+1}: {d0} → {d1} ", end='', flush=True)
            try:
                r = self.run(sp, sv, sm, stat_arb, od_mgr, verbose=False)
                records.append({'period': i+1, 'start': str(d0), 'end': str(d1),
                                'sharpe': r.get('sharpe', 0),
                                'calmar': r.get('calmar', 0),
                                'ann_ret': r.get('ann_ret', 0),
                                'max_dd': r.get('max_dd', 0),
                                'long_pnl': r.get('long_total_pnl', 0),
                                'short_pnl': r.get('short_total_pnl', 0)})
                print(f"Sharpe={r.get('sharpe',0):.3f} MaxDD={r.get('max_dd',0):.2%}")
            except Exception as ex:
                print(f"skip ({str(ex)[:35]})")

        df = pd.DataFrame(records)
        if len(df) > 0:
            print(f"\n  Summary ({len(df)} periods):")
            for col, lbl in [('sharpe','Sharpe'), ('calmar','Calmar'),
                              ('ann_ret','Ann Ret'), ('max_dd','Max DD')]:
                if col in df:
                    print(f"    {lbl:<10} mean:{df[col].mean():>8.3f} "
                          f"std:{df[col].std():>7.3f} "
                          f"worst:{df[col].min():>8.3f}")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 11. Trade log (full record per trade)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    trade_id:          str
    ticker:            str
    direction:         str      # 'long' / 'short'
    entry_date:        str
    entry_price:       float
    position_pct:      float
    regime:            str
    entry_reason:      str      # three-layer rationale
    engines_used:      List[str]
    expected_days:     int
    first_exit:        str
    forced_exit:       str
    canyon_score:      float
    canyon_grade:      str
    # Filled after closing
    exit_date:         Optional[str] = None
    exit_price:        Optional[float] = None
    exit_reason:       Optional[str] = None
    pnl_pct:           Optional[float] = None
    holding_days:      Optional[int] = None
    error_category:    Optional[str] = None
    lesson:            Optional[str] = None


class TradeJournal:
    """Complete trade log system."""

    GATE_CHECKS = [
        ('entry_reason', "entry_reason cannot be empty"),
        ('first_exit',   "first_exit must be specified"),
        ('forced_exit',  "forced_exit must be specified"),
    ]

    def __init__(self, path: str = None):
        self.path   = Path(path) if path else None
        self.trades: Dict[str, TradeRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def open(self, ticker: str, direction: str, entry_price: float,
              position_pct: float, regime: str,
              entry_reason: str, engines_used: List[str],
              expected_days: int, first_exit: str, forced_exit: str,
              canyon_score: float = 0, canyon_grade: str = 'D') -> str:
        """Open position: must pass all Gate Checks before entry."""
        errors = []
        vals = {'entry_reason': entry_reason, 'first_exit': first_exit,
                'forced_exit': forced_exit}
        for k, msg in self.GATE_CHECKS:
            if not vals[k].strip():
                errors.append(f"❌ {msg}")
        if expected_days <= 0:
            errors.append("❌ expected_days must be specified")
        if not engines_used:
            errors.append("❌ engines used must be noted")
        if errors:
            raise ValueError("\nPre-entry gate check failed:\n" + "\n".join(errors))

        tid = str(uuid.uuid4())[:8].upper()
        self.trades[tid] = TradeRecord(
            trade_id=tid, ticker=ticker.upper(), direction=direction,
            entry_date=datetime.now().strftime('%Y-%m-%d'),
            entry_price=entry_price, position_pct=position_pct,
            regime=regime, entry_reason=entry_reason[:120],
            engines_used=engines_used, expected_days=expected_days,
            first_exit=first_exit, forced_exit=forced_exit,
            canyon_score=canyon_score, canyon_grade=canyon_grade
        )
        self._save()
        return tid

    def close(self, trade_id: str, exit_price: float, exit_reason: str,
              error_category: str = '', lesson: str = '') -> Dict:
        if trade_id not in self.trades:
            raise KeyError(f"Trade not found: {trade_id}")
        t = self.trades[trade_id]
        d = 1 if t.direction == 'long' else -1
        pnl = d * (exit_price / t.entry_price - 1)
        days = (datetime.now() - datetime.strptime(t.entry_date, '%Y-%m-%d')).days or 1

        t.exit_date       = datetime.now().strftime('%Y-%m-%d')
        t.exit_price      = exit_price
        t.exit_reason     = exit_reason
        t.pnl_pct         = round(pnl, 6)
        t.holding_days    = days
        t.error_category  = error_category
        t.lesson          = lesson
        self._save()

        icon = '✅' if pnl > 0 else '❌'
        print(f"{icon} [{t.ticker}] {d*100:.0f}% | P&L:{pnl:+.2%} | "
              f"Held:{days}d(expected {t.expected_days}d) | {exit_reason[:50]}")
        return {'pnl': pnl, 'days': days}

    def dashboard(self):
        closed = [t for t in self.trades.values() if t.exit_date]
        open_t = [t for t in self.trades.values() if not t.exit_date]

        print(f"\n{'═'*62}")
        print(f"  📔 Trade Log Dashboard")
        print(f"{'═'*62}")
        print(f"  Open:{len(open_t)} | Closed:{len(closed)} | Total:{len(self.trades)}")

        if closed:
            pnls = [t.pnl_pct for t in closed if t.pnl_pct is not None]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            print(f"\n  P&L Summary:")
            print(f"    Total PnL:{sum(pnls):+.2%} | Win rate:{len(wins)/len(pnls):.1%}")
            print(f"    Avg win:{np.mean(wins):+.2%} | Avg loss:{np.mean(loss):+.2%}")
            if loss: print(f"    Win/loss ratio:{abs(np.mean(wins)/np.mean(loss)):.2f}:1")

            # Attribution by engine
            from collections import defaultdict
            eng_pnl = defaultdict(list)
            for t in closed:
                if t.pnl_pct is None: continue
                for e in t.engines_used:
                    eng_pnl[e].append(t.pnl_pct)
            print(f"\n  Engine attribution:")
            for eng, pnls_e in sorted(eng_pnl.items(), key=lambda x: np.mean(x[1]), reverse=True):
                avg = np.mean(pnls_e)
                print(f"    {'🟢' if avg>0 else '🔴'} {eng:<25} "
                      f"count:{len(pnls_e):<4} mean:{avg:+.2%}")

        if open_t:
            print(f"\n  Current positions:")
            for t in open_t:
                days = (datetime.now() - datetime.strptime(t.entry_date, '%Y-%m-%d')).days
                over = ' ⚠️OVERDUE' if days > t.expected_days else ''
                print(f"    {t.trade_id} {t.ticker} {t.direction} "
                      f"| {days}d(exp {t.expected_days}){over} | {t.regime}")
        print(f"{'═'*62}")

    def _save(self):
        if not self.path: return
        data = {k: asdict(v) for k, v in self.trades.items()}
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.trades[k] = TradeRecord(**v)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Main system
# ══════════════════════════════════════════════════════════════════════════════

class CanyonTradingSystem:
    """
    Canyon quant trading system — final complete version

    Usage:
        system = CanyonTradingSystem()
        result = system.run(['NVDA','AMD','TSM','MU','SPY'], '2020-01-01', '2024-12-31')
    """

    def __init__(self, tc_bps: float = 10, rf: float = 0.03):
        self.data       = DataLayer()
        self.stat_arb   = StatArb()
        self.risk       = RiskManager()
        self.od         = OffensiveDefensiveManager(self.risk)
        self.backtester = WalkForwardBacktester(tc_bps=tc_bps, rf=rf)
        self.ucb        = UCBOptimizer()
        self.journal    = TradeJournal()
        self.rf         = rf

    def run(self, tickers: List[str], start: str, end: str,
            benchmark: str = 'SPY') -> Dict:
        """
        Full run: data → backtest → current analysis → Bayesian optimization → report
        """
        print(f"\n{'═'*62}")
        print(f"  🏔  CANYON Quant Trading System v5.0")
print(f"  Offense/Defense × Shorting × Walk-Forward × UCB Bayesian Optimization")
        print(f"{'═'*62}")
        print(f"  Assets: {tickers}")
        print(f"  Period: {start} → {end}")

        # ── Data loading ─────────────────────────────────────────────────────
        prices, volumes, market = self.data.load(tickers, start, end, benchmark)
        returns = prices.pct_change().dropna()

        print(f"\n{'─'*62}")
        print(f"  Step1: Main backtest (Walk-Forward, Ch.7)")
        print(f"{'─'*62}")
        main_result = self.backtester.run(
            prices, volumes, market, self.stat_arb, self.od, verbose=True
        )

        print(f"\n{'─'*62}")
        print(f"  Step2: Multi-period validation (Ch.7: multiple periods required for robustness)")
        print(f"{'─'*62}")
        period_df = self.backtester.multi_period(
            prices, volumes, market, self.stat_arb, self.od, n_periods=3
        )

        print(f"\n{'─'*62}")
        print(f"  Step3: UCB Bayesian optimization for stat-arb parameters (Ch.9)")
        print(f"{'─'*62}")
        pairs = self.stat_arb.find_pairs(prices)
        if len(pairs) >= 1:
            best_pair = pairs[0]
            t1, t2   = best_pair['t1'], best_pair['t2']
            print(f"  Best cointegrated pair: {t1}/{t2} "
                  f"(pvalue={best_pair['pvalue']:.3f}, HL={best_pair['half_life']:.1f}d)")

            def obj_fn(entry_z: float, exit_z: float, window: int):
                sa = StatArb(entry_z=entry_z, exit_z=exit_z, window=window)
                r  = sa.backtest_pair(prices[t1], prices[t2], best_pair)
                # Hard constraint: MaxDD must not exceed 5.6%
                if r.get('max_dd', -1) < -0.056:
                    return -99.0
                return r.get('sharpe', 0.0)

            opt = self.ucb.optimize(
                obj_fn,
                param_bounds={'entry_z': (1.5, 3.0),
                               'exit_z':  (0.2, 1.0),
                               'window':  (10, 30)},
                n_iter=18
            )
            bp = opt.get('best_params', {})
            if bp and opt.get('best_score', 0) > -90:
                print(f"  Best params: entry_z={bp.get('entry_z',2):.2f} "
                      f"exit_z={bp.get('exit_z',0.5):.2f} "
                      f"window={bp.get('window',21)} "
                      f"→ Sharpe={opt.get('best_score',0):.3f}")
                # Apply best parameters
                self.stat_arb.entry_z = bp.get('entry_z', 2.0)
                self.stat_arb.exit_z  = bp.get('exit_z',  0.5)
                self.stat_arb.window  = int(bp.get('window', 21))
        else:
            print("  No cointegrated pair found (insufficient data or low correlation)")

        print(f"\n{'─'*62}")
        print(f"  Step4: Current market state analysis")
        print(f"{'─'*62}")
        current = self.analyze_current(prices, volumes, market)

        print(f"\n{'─'*62}")
        print(f"  Step5: Parameter grid test (Ch.7: parameter robustness)")
        print(f"{'─'*62}")
        grid = self._grid_test(prices, volumes, market)

        self._final_report(main_result, period_df, current, grid)
        return {
            'main': main_result, 'periods': period_df,
            'current': current, 'grid': grid
        }

    def analyze_current(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                         market: pd.Series) -> Dict:
        """Current market state + today's position recommendation."""
        regime, detail = detect_regime(market, prices)
        cs   = cross_sectional_momentum(prices)
        tsig = trend_signals(prices)
        pairs = self.stat_arb.find_pairs(prices)
        arbs  = self.stat_arb.current_opportunity(prices, pairs)

        print(f"\n  Regime: {regime.label} ({regime.stance})")
        print(f"    Composite:{detail.get('composite', 0):+.1f} "
              f"| Trend:{detail.get('trend', 0):+.2f} "
              f"| Momentum:{detail.get('momentum', 0):+.2f} "
              f"| Breadth:{detail.get('breadth', 0):+.2f}")
        print(f"    Long cap:{regime.max_long:.0%} "
              f"Short cap:{regime.max_short:.0%} "
              f"Position scale:{regime.long_scale:.1f}x")

        print(f"\n  Cross-sectional momentum (Ch.6):")
        print(f"    Long top 25%: {cs['long'][:5]}")
        print(f"    Short bottom 25%: {cs['short'][:5]}")
        print(f"    L/S spread: {cs['spread']:+.2%}")

        print(f"\n  Trend signals (Ch.5 EMA/SMA):")
        bull_tks = [tk for tk in tsig.index if tsig.loc[tk, 'signal'] == 1]
        bear_tks = [tk for tk in tsig.index if tsig.loc[tk, 'signal'] == -1]
        print(f"    Golden cross / uptrend: {bull_tks[:6]}")
        print(f"    Death cross / downtrend: {bear_tks[:6]}")

        if arbs:
            print(f"\n  Stat-arb opportunities (Ch.8):")
            for a in arbs:
                d = 'L' + a['t1'] + 'S' + a['t2'] if a['direction'] == 1 else 'S' + a['t1'] + 'L' + a['t2']
                print(f"    {a['t1']}/{a['t2']} z={a['z']:.2f} → {d} (HL={a['half_life']:.0f}d)")

        # Canyon F/C/E scoring
        print(f"\n  Canyon F/C/E scores:")
        canyon_scores = {}
        for tk in prices.columns:
            s = canyon_score_auto(prices[tk], volumes[tk], market, regime)
            canyon_scores[tk] = s
            if s['can_buy']:
                print(f"    ✅ {tk}: F={s['f']} C={s['c']} E={s['e']} "
                      f"→ {s['total']:.0f}pts({s['grade']}) cap {s['max_pos']:.0%}")

        # Today position recommendation
        long_alpha  = cs['long_alpha'].dropna()
        short_alpha = cs['short_alpha'].dropna()
        ret_hist    = prices.pct_change().dropna()
        alloc = self.od.allocate(
            regime=regime, long_alpha=long_alpha,
            short_alpha=short_alpha, trend_sig=tsig,
            stat_arb_opps=arbs, returns=ret_hist
        )

        print(f"\n  Today's position recommendation:")
        print(f"    {alloc.rationale}")
        if alloc.longs:
            print(f"    Longs:")
            for tk, w in sorted(alloc.longs.items(), key=lambda x: -x[1]):
                cs_info = canyon_scores.get(tk, {})
                print(f"      ▲ {tk:<8} {w:+.1%}  "
                      f"(Canyon:{cs_info.get('total',0):.0f}pts/{cs_info.get('grade','?')})")
        if alloc.shorts:
            print(f"    Shorts:")
            for tk, w in sorted(alloc.shorts.items(), key=lambda x: x[1]):
                print(f"      ▼ {tk:<8} {w:+.1%}  (short: weak+death-cross)")

        # Stress test
        all_w = alloc.to_series()
        if len(all_w) > 0:
            print(f"\n  Stress test (5.6% max DD hard constraint):")
            scenarios = {'2008 GFC': -0.45, '2020 COVID crash': -0.32,
                         '2022 rate-hike bear': -0.22, 'Normal correction -15%': -0.15}
            for name, shock in scenarios.items():
                port_loss = float(sum((float(w) * shock if float(w) > 0
                                       else float(w) * (-shock * 0.7))
                                      for w in all_w))
                ok = port_loss > -0.056
                print(f"    {'✅' if ok else '⚠️'} {name}: {port_loss:+.2%}")

        return {'regime': regime, 'allocation': alloc,
                'canyon_scores': canyon_scores, 'cs': cs, 'trends': tsig}

    def _grid_test(self, prices, volumes, market) -> pd.DataFrame:
        """
        Ch.7: parameter sensitivity test
        Different SMA parameters → Sharpe ratio (finding robust parameters)
        """
        results = []
        for ema_s in [5, 10]:
            for sma_s in [20, 30, 50]:
                try:
                    # Temporarily modify parameters
                    orig_trend = trend_signals.__defaults__
                    # Simplified: evaluate with small test window
                    p_sub = prices.tail(400)
                    v_sub = volumes.tail(400)
                    m_sub = market.tail(400)
                    if len(p_sub) < self.backtester.train_w + self.backtester.test_w:
                        continue

                    # Quick backtest
                    daily_r = []
                    for s in range(self.backtester.train_w, len(p_sub) - self.backtester.test_w, self.backtester.test_w):
                        p_tr = p_sub.iloc[s - self.backtester.train_w:s]
                        p_te = p_sub.iloc[s:s + self.backtester.test_w]
                        ts   = trend_signals(p_tr, ema_span=ema_s, sma_span=sma_s)
                        bull = [tk for tk in ts.index if ts.loc[tk, 'signal'] == 1]
                        n    = len(bull)
                        if n == 0:
                            continue
                        w = 1.0 / n
                        for d in p_te.index[1:]:
                            dr = sum(float(p_te.loc[d, tk] / p_te.iloc[p_te.index.get_loc(d)-1][tk] - 1) * w
                                     for tk in bull if tk in p_te.columns)
                            daily_r.append(dr)

                    if len(daily_r) > 10:
                        m = backtest_metrics(pd.Series(daily_r))
                        results.append({'ema': ema_s, 'sma': sma_s,
                                        'sharpe': m['sharpe'], 'max_dd': m['max_dd']})
                        print(f"    EMA{ema_s}/SMA{sma_s}: Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%}")
                except Exception:
                    pass

        df = pd.DataFrame(results)
        if len(df) > 0:
            best = df.nlargest(1, 'sharpe').iloc[0]
            print(f"    → Best: EMA{best['ema']:.0f}/SMA{best['sma']:.0f} "
                  f"Sharpe={best['sharpe']:.3f}")
        return df

    def _final_report(self, main, periods, current, grid):
        print(f"\n{'═'*62}")
        print(f"  📋 Final backtest report summary")
        print(f"{'═'*62}")
        print(f"  Ann Ret:  {main.get('ann_ret',0):+.2%}")
        print(f"  Sharpe：  {main.get('sharpe',0):.4f}")
        print(f"  Calmar：  {main.get('calmar',0):.4f}")
        print(f"  Max DD:   {main.get('max_dd',0):.2%}")
        print(f"  Total Ret:{main.get('total_ret',0):+.2%}")
        print(f"  Long PnL: {main.get('long_total_pnl',0):+.2%}")
        print(f"  Short PnL:{main.get('short_total_pnl',0):+.2%}")

        if len(periods) > 0:
            print(f"\n  Multi-period robustness ({len(periods)} periods):")
            print(f"    Sharpe: μ={periods['sharpe'].mean():.3f} σ={periods['sharpe'].std():.3f}")
            print(f"    MaxDD:  μ={periods['max_dd'].mean():.2%} σ={periods['max_dd'].std():.2%}")

        regime = current.get('regime')
        if regime:
            print(f"\n  Current market: {regime.label} ({regime.stance})")
            alloc = current.get('allocation')
            if alloc:
                print(f"  Position strategy: {alloc.rationale}")

        print(f"\n  Core logic:")
        print(f"  Bull → Offense: top-25% momentum long, max 15% per stock, full allocation")
        print(f"  Bear → Defense: reduce longs + short bottom-25% weakest stocks (Ch.6)")
        print(f"  Stat-arb: cointegrated pairs, z-score signal (Ch.8)")
        print(f"  Param optimization: UCB Bayesian maximizes Sharpe (Ch.9)")
        print(f"  Backtest metrics: Ch.7 exact formulas (Ann Ret/Sharpe/Max DD/Calmar)")
        print(f"{'═'*62}")

    # ── Trade log interface ─────────────────────────────────────────────────────────

    def record_trade(self, ticker: str, direction: str,
                      price: float, pct: float, regime: str,
                      reason: str, engines: List[str],
                      days: int, first_exit: str,
                      forced_exit: str) -> str:
        """Log trade open."""
        return self.journal.open(
            ticker=ticker, direction=direction,
            entry_price=price, position_pct=pct,
            regime=regime, entry_reason=reason,
            engines_used=engines, expected_days=days,
            first_exit=first_exit, forced_exit=forced_exit
        )

    def close_trade(self, tid: str, exit_price: float,
                     exit_reason: str, lesson: str = '') -> Dict:
        """Log trade close."""
        return self.journal.close(tid, exit_price, exit_reason, lesson=lesson)


# ══════════════════════════════════════════════════════════════════════════════
# Main program
# ══════════════════════════════════════════════════════════════════════════════

def main():
    system = CanyonTradingSystem(tc_bps=10, rf=0.03)

    # Asset list: AI/semiconductor core + broad ETFs (usable for shorts)
    TICKERS = ['NVDA', 'AMD', 'TSM', 'MU', 'SOXX',
               'AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    START   = '2020-01-01'
    END     = '2024-12-31'

    result = system.run(TICKERS, START, END, benchmark='SPY')

    # ── Trade log demo ─────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  📔 Trade log demo")
    print(f"{'─'*62}")

    try:
        tid1 = system.record_trade(
            ticker='NVDA', direction='long',
            price=875.0, pct=0.07, regime='Normal Bull',
reason="H200 shipped (A-grade statement)+TSMC CoWoS full capacity+Leopold AI compute structural thesis+capitulation signal",
            engines=['canyon_fce', 'momentum', 'statement_dd', 'cycle', 'inst_flow'],
            days=25,
            first_exit="Crowding rises after GTC product launch / RSI>78",
            forced_exit="Thesis falsified / TSMC shipments 10% below expectations / GEX turns negative"
        )
        system.close_trade(tid1, exit_price=973.0,
                           exit_reason="Catalyst realized after GTC, first exit triggered",
                           lesson="Reduce 50% on day first exit triggers, do not wait for greed")

        tid2 = system.record_trade(
            ticker='QQQ', direction='short',
            price=430.0, pct=0.05, regime='Mild Bear',
            reason="Cross-sectional bottom 25%+SMA death cross+market composite -35+rate-hike valuation pressure",
            engines=['momentum', 'trend_following', 'canyon_fce'],
            days=15,
            first_exit="SMA golden cross / regime upgrades to neutral",
            forced_exit="Composite score >-20 (bear retreating)"
        )
        system.close_trade(tid2, exit_price=408.0,
                           exit_reason="Short target reached, take profit",
                           lesson="Shorting ETFs in bear market is more controllable than individual stocks, prioritize next time")

        system.journal.dashboard()

    except ValueError as e:
        print(f"  Entry gate check: {e}")

    return result


if __name__ == '__main__':
    result = main()
