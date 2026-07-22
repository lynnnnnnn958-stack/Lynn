#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9  Step 100 — Walk-Forward Out-of-Sample Backtest
==========================================================
Strictly separates in-sample (IS) and out-of-sample (OOS) periods.

  In-sample  : rebalance dates < OOS_CUTOFF  (default 2020-01-01)
  Out-of-sample : rebalance dates >= OOS_CUTOFF  (locked test set)

The model is NEVER tuned on post-cutoff data.  Predictions on the OOS
period are generated using a rolling training window (last LOOKBACK_DAYS
of past data only) — no future data is ever used in training.

This is the closest to real-world performance: the OOS IC and Sharpe
answer whether the signal would have worked on data the system has
genuinely never seen before.

Feature set and no-lookahead rules are identical to Step 66.

Outputs:
  wf_oos_predictions.csv      all predictions, column period=IS|OOS
  wf_oos_ic_by_period.csv     IC per signal for IS and OOS separately
  wf_oos_backtest_perf.csv    monthly portfolio perf with period label
  wf_oos_equity_curve.csv     cumulative return by period (for chart)
  wf_oos_summary.csv          IS vs OOS metric comparison table
  wf_oos_report.md            narrative markdown report

Usage:
  python canyon_final_v9_step100_walk_forward_oos.py
  python canyon_final_v9_step100_walk_forward_oos.py --cutoff 2020-01-01
  python canyon_final_v9_step100_walk_forward_oos.py --top 8 --tc 10
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── constants ─────────────────────────────────────────────────────────────────
OOS_CUTOFF_DEFAULT = "2020-01-01"
LOOKBACK_DAYS  = 252
HOLD_PERIOD    = 21
WARMUP_DAYS    = 504
TOP_N          = 8
TC_BPS         = 10
ANN_FACTOR     = 252

FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_pct",
    "high_52w", "rev_1w", "vol_ratio", "ema_cross",
    "rank_mom", "rank_trend",
]

FUNDAMENTAL_FEATURES = [
    "accruals", "rev_growth", "gross_margin_chg", "roe", "debt_change", "pb_ratio",
]

UNIVERSE = list(dict.fromkeys([
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP",
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
    "MSFT","NVDA","AMZN","META","GOOGL","JPM","XOM","CVX",
    "JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY","TMO","COST",
    "V","MA","HD","PYPL","NFLX","INTC","QCOM","TXN","AVGO","CRM","ADBE",
]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price loader
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers: list[str]) -> pd.DataFrame:
    cache_path = ROOT / "backtest_price_cache.csv"
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=WARMUP_DAYS + LOOKBACK_DAYS + 365 * 26)

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            available = [t for t in tickers if t in cached.columns]
            if available and age_h < 24:
                print(f"  [cache] {len(available)}/{len(tickers)} tickers, age={age_h:.1f}h")
                return cached[available].dropna(how="all")
        except Exception as e:
            print(f"  [cache] {e}")

    try:
        import yfinance as yf
        print(f"  [yfinance] Downloading {len(tickers)} tickers …")

        raw = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all")
        prices.to_csv(cache_path)
        print(f"  [yfinance] {prices.shape[1]} tickers × {len(prices)} days")
        return prices
    except Exception as e:
        raise RuntimeError(f"Cannot load prices: {e}")


def load_fundamentals() -> Optional[pd.DataFrame]:
    """Load fundamental signals panel from step165 output (if available)."""
    path = ROOT / "fundamental_signals_daily.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        print(f"  [fundamentals] {df['ticker'].nunique()} tickers, "
              f"{df['date'].nunique()} days, signals: {FUNDAMENTAL_FEATURES}")
        return df
    except Exception as e:
        print(f"  [fundamentals] Could not load: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature builder  (identical no-lookahead rules as Step 66)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = (ema_fast - ema_slow) / (close.abs() + 1e-10)
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pct(close: pd.Series, window: int = 20) -> pd.Series:
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = ma - 2 * std
    upper = ma + 2 * std
    return (close - lower) / (upper - lower + 1e-10)


def _vol_ratio(log_r: pd.Series, short: int = 10, long: int = 63) -> pd.Series:
    return log_r.rolling(short).std() / (log_r.rolling(long).std() + 1e-10)


def build_features(prices: pd.DataFrame,
                   fund_panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    all_records: list[pd.DataFrame] = []
    spy_prices = prices.get("SPY", None)

    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        s = prices[ticker].dropna()
        if len(s) < WARMUP_DAYS:
            continue

        mom_1m         = s.pct_change(21).shift(1)
        mom_3m         = s.pct_change(63).shift(1)
        mom_6m         = s.pct_change(126).shift(1)
        mom_252        = s.pct_change(252).shift(1)
        mom_21         = s.pct_change(21).shift(1)
        mom_12m_skip1m = (mom_252 - mom_21)
        trend_200      = (s / s.rolling(200).mean() - 1).shift(1)
        rsi_14         = _rsi(s, 14).shift(1)

        macd_line, signal_line, histogram = _macd(s)
        macd        = macd_line.shift(1)
        macd_signal = signal_line.shift(1)
        macd_hist   = histogram.shift(1)

        bb_pct    = _bollinger_pct(s).shift(1)

        # 52-week high proximity (George & Hwang 2004)
        high_52w  = (s / s.rolling(252).max()).shift(1)
        # 1-week short-term reversal (Jegadeesh 1990)
        rev_1w    = s.pct_change(5).shift(1)
        # volatility regime: short vol / long vol
        log_r     = np.log(s / s.shift(1))
        vol_ratio = _vol_ratio(log_r).shift(1)
        # EMA50/EMA200 golden-cross ratio
        ema_cross = (_ema(s, 50) / (_ema(s, 200) + 1e-10) - 1).shift(1)

        fwd_ret = np.log(s.shift(-HOLD_PERIOD) / s)

        feat_dict = {
            "ticker":         ticker,
            "mom_1m":         mom_1m,
            "mom_3m":         mom_3m,
            "mom_6m":         mom_6m,
            "mom_12m_skip1m": mom_12m_skip1m,
            "trend_200":      trend_200,
            "rsi_14":         rsi_14,
            "macd":           macd,
            "macd_signal":    macd_signal,
            "macd_hist":      macd_hist,
            "bb_pct":         bb_pct,
            "high_52w":       high_52w,
            "rev_1w":         rev_1w,
            "vol_ratio":      vol_ratio,
            "ema_cross":      ema_cross,
            "forward_ret":    fwd_ret,
        }

        feat_df = pd.DataFrame(feat_dict, index=s.index)
        feat_df.index.name = "date"

        # Merge fundamental signals if available
        if fund_panel is not None and ticker in fund_panel["ticker"].values:
            tk_fund = (fund_panel[fund_panel["ticker"] == ticker]
                       .set_index("date")
                       .reindex(s.index, method="ffill"))
            for fsig in FUNDAMENTAL_FEATURES:
                if fsig in tk_fund.columns:
                    feat_df[fsig] = tk_fund[fsig].values

        all_records.append(feat_df)

    if not all_records:
        return pd.DataFrame()

    panel = pd.concat(all_records).reset_index()
    panel = panel.dropna(subset=FEATURES[:8])
    panel["rank_mom"]   = panel.groupby("date")["mom_12m_skip1m"].rank(pct=True)
    panel["rank_trend"] = panel.groupby("date")["trend_200"].rank(pct=True)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-forward train/predict  (tagged IS vs OOS)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_predict(
    panel: pd.DataFrame,
    oos_cutoff: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    For every monthly rebalance date:
      - Train on the previous `lookback` trading-days of data (strict past)
      - Predict cross-sectional scores
      - Tag result as period=IS (before cutoff) or period=OOS (on/after cutoff)

    No model selection is done on OOS data — the model architecture,
    features, and hyperparameters are fixed before seeing OOS data.
    """
    dates = panel["date"].sort_values().unique()
    rebalance_dates: list = []
    prev_month = None
    for d in dates:
        m = pd.Timestamp(d).month
        if m != prev_month:
            rebalance_dates.append(d)
            prev_month = m

    min_date = dates[min(lookback, len(dates) - 1)]
    rebalance_dates = [d for d in rebalance_dates if d >= min_date]

    all_preds: list[dict] = []
    scaler_r = StandardScaler()
    scaler_f = StandardScaler()
    scaler_l = StandardScaler()

    print(f"  [WF-OOS] {len(rebalance_dates)} rebalance dates  cutoff={oos_cutoff.date()}")
    oos_count = sum(1 for d in rebalance_dates if pd.Timestamp(d) >= oos_cutoff)
    print(f"  [WF-OOS] IS={len(rebalance_dates) - oos_count}  OOS={oos_count}")

    for idx, reb_date in enumerate(rebalance_dates):
        reb_ts = pd.Timestamp(reb_date)
        period = "OOS" if reb_ts >= oos_cutoff else "IS"

        train_end   = reb_ts - pd.Timedelta(days=1)
        train_start = reb_ts - pd.Timedelta(days=lookback * 1.6)

        train_mask = (
            (panel["date"] >= train_start) &
            (panel["date"] <= train_end) &
            (panel["forward_ret"].notna())
        )
        train_df = panel[train_mask].copy()

        if len(train_df) < 200:
            continue

        train_df["y_ranked"] = train_df.groupby("date")["forward_ret"].rank(pct=True)
        X_train = train_df[FEATURES].values
        y_train = train_df["y_ranked"].values

        pred_mask = panel["date"] == reb_date
        pred_df   = panel[pred_mask].dropna(subset=FEATURES)
        if pred_df.empty:
            continue

        X_pred = pred_df[FEATURES].values

        # Ridge
        try:
            scaler_r.fit(X_train)
            ridge = Ridge(alpha=50.0)
            ridge.fit(scaler_r.transform(X_train), y_train)
            ridge_scores = ridge.predict(scaler_r.transform(X_pred))
        except Exception:
            ridge_scores = np.zeros(len(pred_df))

        # Random Forest
        try:
            scaler_f.fit(X_train)
            rf = RandomForestRegressor(
                n_estimators=80, max_depth=4,
                min_samples_leaf=10, random_state=42, n_jobs=-1,
            )
            rf.fit(scaler_f.transform(X_train), y_train)
            rf_scores = rf.predict(scaler_f.transform(X_pred))
        except Exception:
            rf_scores = np.zeros(len(pred_df))

        # LightGBM (optional)
        if _HAS_LGB:
            try:
                scaler_l.fit(X_train)
                lgb = LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=4,
                    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
                lgb.fit(scaler_l.transform(X_train), y_train)
                lgb_scores = lgb.predict(scaler_l.transform(X_pred))
            except Exception:
                lgb_scores = np.zeros(len(pred_df))
        else:
            lgb_scores = np.zeros(len(pred_df))

        if _HAS_LGB and lgb_scores.any():
            ensemble = 0.30 * ridge_scores + 0.30 * rf_scores + 0.40 * lgb_scores
        else:
            ensemble = 0.50 * ridge_scores + 0.50 * rf_scores

        for i, (_, row) in enumerate(pred_df.iterrows()):
            all_preds.append({
                "rebalance_date": str(reb_date)[:10],
                "ticker":         row["ticker"],
                "period":         period,
                "is_oos":         period == "OOS",
                "ridge_score":    float(ridge_scores[i]),
                "rf_score":       float(rf_scores[i]),
                "lgbm_score":     float(lgb_scores[i]),
                "ensemble_score": float(ensemble[i]),
                "n_train":        len(train_df),
            })

        if (idx + 1) % 12 == 0 or idx == len(rebalance_dates) - 1:
            print(f"  [WF-OOS] {idx+1}/{len(rebalance_dates)}  {str(reb_date)[:10]}  "
                  f"period={period}  n_train={len(train_df)}")

    return pd.DataFrame(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# 4. IC computation by period
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic_for_period(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    period: str,
) -> list[dict]:
    sub_preds = preds[preds["period"] == period]
    if sub_preds.empty:
        return []

    reb_dates = sub_preds["rebalance_date"].unique()
    panel_sub = panel[
        panel["date"].astype(str).str[:10].isin([str(d)[:10] for d in reb_dates])
    ].copy()
    panel_sub["rebalance_date"] = panel_sub["date"].astype(str).str[:10]

    merged = sub_preds.merge(
        panel_sub[["rebalance_date", "ticker", "forward_ret"] + FEATURES],
        on=["rebalance_date", "ticker"], how="inner",
    ).dropna(subset=["forward_ret"])

    if merged.empty:
        return []

    signal_cols = FEATURES + ["ridge_score", "rf_score", "lgbm_score", "ensemble_score"]
    records: list[dict] = []
    for col in signal_cols:
        if col not in merged.columns:
            continue
        ics: list[float] = []
        for _, grp in merged.groupby("rebalance_date"):
            if len(grp) < 5:
                continue
            x = pd.to_numeric(grp[col], errors="coerce")
            y = pd.to_numeric(grp["forward_ret"], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 5:
                continue
            ic, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ics.append(ic)

        if not ics:
            records.append({
                "signal": col, "period": period,
                "n_obs": 0, "mean_ic": 0.0, "t_stat": 0.0,
                "ic_positive_pct": "—", "status": "NO_DATA",
            })
            continue

        arr = np.array(ics)
        n   = len(arr)
        mean_ic = float(arr.mean())
        std_ic  = float(arr.std())
        t_stat  = float(mean_ic / (std_ic / np.sqrt(n) + 1e-10))
        _, p_val = ttest_1samp(arr, 0) if n > 2 else (None, 1.0)
        ic_pos_pct = f"{(arr > 0).mean() * 100:.1f}%"

        if   mean_ic > 0.05 and abs(t_stat) > 2.0: status = "STRONG"
        elif mean_ic > 0.03 and abs(t_stat) > 2.0: status = "USABLE"
        elif mean_ic > 0.0  and abs(t_stat) > 1.0: status = "WEAK"
        elif mean_ic < 0.0:                          status = "NEGATIVE"
        else:                                         status = "WEAK"

        records.append({
            "signal":          col,
            "period":          period,
            "n_obs":           n,
            "mean_ic":         round(mean_ic, 4),
            "std_ic":          round(std_ic, 4),
            "t_stat":          round(t_stat, 2),
            "p_value":         round(float(p_val), 4),
            "ic_positive_pct": ic_pos_pct,
            "status":          status,
        })

    return records


def compute_ic_by_period(panel: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = _compute_ic_for_period(panel, preds, "IS")
    rows += _compute_ic_for_period(panel, preds, "OOS")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Portfolio backtest by period
# ─────────────────────────────────────────────────────────────────────────────

def backtest_by_period(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = TOP_N,
    tc_bps: float = TC_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (perf_df, equity_curve_df)."""
    if preds.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    reb_dates = sorted(preds["rebalance_date"].unique())
    records: list[dict] = []
    prev_holdings: set[str] = set()

    for i, reb_date in enumerate(reb_dates[:-1]):
        next_date = reb_dates[i + 1]
        period    = preds.loc[preds["rebalance_date"] == reb_date, "period"].iloc[0]

        grp = preds[preds["rebalance_date"] == reb_date].copy()
        grp["_score"] = pd.to_numeric(grp["ensemble_score"], errors="coerce").fillna(0)
        top = grp.nlargest(top_n, "_score")["ticker"].tolist()
        if not top:
            continue

        reb_ts  = pd.Timestamp(reb_date)
        next_ts = pd.Timestamp(next_date)
        avail   = prices.index[(prices.index >= reb_ts) & (prices.index <= next_ts)]
        if len(avail) < 2:
            continue

        t_start, t_end = avail[0], avail[-1]
        rets: list[float] = []
        for tk in top:
            if tk in prices.columns:
                p_s = prices[tk].loc[t_start]
                p_e = prices[tk].loc[t_end]
                if pd.notna(p_s) and pd.notna(p_e) and p_s > 0:
                    rets.append(float(p_e / p_s - 1))
        if not rets:
            continue

        port_ret = float(np.mean(rets))
        spy_ret  = 0.0
        if "SPY" in prices.columns:
            s_s = prices["SPY"].loc[t_start]
            s_e = prices["SPY"].loc[t_end]
            if pd.notna(s_s) and s_s > 0:
                spy_ret = float(s_e / s_s - 1)

        new_set  = set(top)
        turnover = len(new_set.symmetric_difference(prev_holdings)) / max(len(new_set | prev_holdings), 1)
        tc       = turnover * tc_bps / 10000
        net_ret  = port_ret - tc

        records.append({
            "rebalance_date": reb_date,
            "period_end":     str(next_date)[:10],
            "period":         period,
            "ml_ret":         round(net_ret, 6),
            "spy_ret":        round(spy_ret, 6),
            "alpha":          round(net_ret - spy_ret, 6),
            "n_held":         len(top),
            "turnover_pct":   round(turnover * 100, 1),
            "tickers":        " | ".join(sorted(top)),
        })
        prev_holdings = new_set

    perf_df = pd.DataFrame(records)
    if perf_df.empty:
        return perf_df, pd.DataFrame()

    # Equity curves — separate IS and OOS but joined into one long DataFrame
    equity_rows: list[dict] = []
    is_nav, oos_nav = 1.0, 1.0
    spy_nav = 1.0

    for _, row in perf_df.iterrows():
        ml_r  = float(pd.to_numeric(row["ml_ret"],  errors="coerce") or 0)
        spy_r = float(pd.to_numeric(row["spy_ret"], errors="coerce") or 0)
        p     = row["period"]
        if p == "IS":
            is_nav  = is_nav  * (1 + ml_r)
        else:
            oos_nav = oos_nav * (1 + ml_r)
        spy_nav = spy_nav * (1 + spy_r)
        equity_rows.append({
            "rebalance_date": row["rebalance_date"],
            "period":         p,
            "ml_nav_is":      is_nav  if p == "IS"  else None,
            "ml_nav_oos":     oos_nav if p == "OOS" else None,
            "spy_nav":        spy_nav,
        })

    equity_df = pd.DataFrame(equity_rows)
    return perf_df, equity_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary metrics  (IS vs OOS side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def _period_stats(
    perf_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    period: str,
) -> dict:
    sub = perf_df[perf_df["period"] == period] if not perf_df.empty else pd.DataFrame()
    ic_sub = ic_df[(ic_df["period"] == period) & (ic_df["signal"] == "ensemble_score")] if not ic_df.empty else pd.DataFrame()

    stats: dict = {"period": period}
    if not sub.empty:
        rets  = pd.to_numeric(sub["ml_ret"],  errors="coerce").dropna()
        spy_r = pd.to_numeric(sub["spy_ret"], errors="coerce").dropna()
        n     = len(rets)
        if n:
            total    = float((1 + rets).prod() - 1)
            spy_tot  = float((1 + spy_r).prod() - 1) if not spy_r.empty else 0.0
            alpha    = total - spy_tot
            sharpe   = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(12))
            cum      = (1 + rets).cumprod()
            mdd      = float((cum / cum.cummax() - 1).min())
            win_rate = float((rets > spy_r).mean() * 100) if len(spy_r) == n else 0.0
            stats.update({
                "n_periods":   n,
                "total_ret":   round(total * 100, 2),
                "spy_ret":     round(spy_tot * 100, 2),
                "alpha":       round(alpha * 100, 2),
                "sharpe":      round(sharpe, 3),
                "max_drawdown": round(mdd * 100, 2),
                "win_rate_pct": round(win_rate, 1),
            })
    if not ic_sub.empty:
        stats["ensemble_ic"]     = float(ic_sub["mean_ic"].iloc[0])
        stats["ensemble_t_stat"] = float(ic_sub["t_stat"].iloc[0])
        stats["ic_status"]       = str(ic_sub["status"].iloc[0])
    return stats


def build_summary(
    perf_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    oos_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    is_stats  = _period_stats(perf_df, ic_df, "IS")
    oos_stats = _period_stats(perf_df, ic_df, "OOS")

    metrics = [
        ("Periods (months)",   "n_periods",    None),
        ("ML Total Return %",  "total_ret",    None),
        ("SPY Total Return %", "spy_ret",       None),
        ("Alpha vs SPY %",     "alpha",         0),
        ("Annualised Sharpe",  "sharpe",        0.5),
        ("Max Drawdown %",     "max_drawdown",  -20),
        ("Win Rate vs SPY %",  "win_rate_pct",  50),
        ("Ensemble IC",        "ensemble_ic",   0.03),
        ("IC t-stat",          "ensemble_t_stat", 2.0),
        ("IC Status",          "ic_status",     None),
    ]

    rows: list[dict] = []
    for label, key, good_threshold in metrics:
        is_val  = is_stats.get(key,  "—")
        oos_val = oos_stats.get(key, "—")
        if is_val  != "—" and isinstance(is_val,  float): is_val  = f"{is_val:.3f}"
        if oos_val != "—" and isinstance(oos_val, float): oos_val = f"{oos_val:.3f}"
        rows.append({
            "metric":       label,
            "in_sample":    str(is_val),
            "out_of_sample": str(oos_val),
            "cutoff":       str(oos_cutoff.date()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    summary_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    oos_cutoff: pd.Timestamp,
    ts: str,
) -> None:
    is_ic  = ic_df[(ic_df["period"] == "IS")  & (ic_df["signal"] == "ensemble_score")]
    oos_ic = ic_df[(ic_df["period"] == "OOS") & (ic_df["signal"] == "ensemble_score")]

    is_ic_val  = float(is_ic["mean_ic"].iloc[0])  if not is_ic.empty  else float("nan")
    oos_ic_val = float(oos_ic["mean_ic"].iloc[0]) if not oos_ic.empty else float("nan")
    degradation = is_ic_val - oos_ic_val if not (np.isnan(is_ic_val) or np.isnan(oos_ic_val)) else float("nan")

    if not np.isnan(oos_ic_val):
        if oos_ic_val > 0.05:
            verdict = "STRONG — Model generalises well to unseen post-2020 data."
        elif oos_ic_val > 0.03:
            verdict = "USABLE — Positive OOS IC with some statistical support."
        elif oos_ic_val > 0:
            verdict = "WEAK — Positive but not statistically significant on unseen data."
        else:
            verdict = "NEGATIVE — Model does not hold up on out-of-sample data."
    else:
        verdict = "INSUFFICIENT DATA"

    lines = [
        "# Canyon v9 — Walk-Forward Out-of-Sample Backtest (Step 100)",
        f"Generated: {ts}",
        f"OOS Cutoff: {oos_cutoff.date()}  (in-sample < cutoff, out-of-sample ≥ cutoff)",
        "",
        "## Key Finding",
        f"> **{verdict}**",
        "",
        f"| Metric | In-Sample (pre-{oos_cutoff.year}) | Out-of-Sample (post-{oos_cutoff.year}) | Change |",
        "|---|---|---|---|",
        f"| Ensemble IC | {is_ic_val:+.4f} | {oos_ic_val:+.4f} | {degradation:+.4f} (IC decay) |",
        "",
        "## IS vs OOS Comparison",
        "",
        "| Metric | In-Sample | Out-of-Sample | OOS Cutoff |",
        "|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"| {row['metric']} | {row['in_sample']} | {row['out_of_sample']} | {row['cutoff']} |")

    lines += [
        "",
        "## IC Breakdown by Period",
        "",
        "| Signal | Period | Mean IC | t-stat | IC+ Rate | Status |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in ic_df.iterrows():
        lines.append(
            f"| {row['signal']} | {row['period']} | {row['mean_ic']:+.4f} | "
            f"{row['t_stat']:+.2f} | {row.get('ic_positive_pct','—')} | {row['status']} |"
        )

    lines += [
        "",
        "## Methodology",
        f"- OOS cutoff: {oos_cutoff.date()} — all data on or after this date is the locked test set",
        f"- Walk-forward training window: rolling {LOOKBACK_DAYS} trading days",
        "- Monthly rebalance; top-8 equal-weight long portfolio; 10bps one-way TC",
        "- Features: 10 price-derived signals (same as Step 66, no look-ahead)",
        "- Models: Ridge, Random Forest, Ensemble (and LightGBM if installed)",
        "- NO model selection or hyperparameter tuning on OOS data",
        "",
        "## Limitations",
        "- Survivorship bias in universe (current S&P tickers only; delisted not included)",
        "- Small universe (≤ 40 tickers) means limited cross-sectional variation",
        "- OOS period spans COVID crash, recovery, rate hike cycle — extreme regimes",
    ]

    p = ROOT / "wf_oos_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 100 — Walk-Forward OOS Backtest")
    parser.add_argument("--cutoff",          default=OOS_CUTOFF_DEFAULT,
                        help="OOS cutoff date (default: 2020-01-01)")
    parser.add_argument("--top",             type=int,   default=TOP_N)
    parser.add_argument("--tc",              type=float, default=TC_BPS)
    parser.add_argument("--lookback",        type=int,   default=LOOKBACK_DAYS)
    parser.add_argument("--use-fundamentals",action="store_true",
                        help="Merge fundamental signals from step165 into feature set")
    parser.add_argument("--sizing", choices=["equal","risk_parity","score_wtd","min_var"],
                        default="equal",
                        help="Portfolio sizing method (default: equal)")
    args = parser.parse_args()

    oos_cutoff = pd.Timestamp(args.cutoff)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*64}")
    print(f"Canyon v9 Step 100 — Walk-Forward Out-of-Sample Backtest")
    print(f"OOS cutoff : {oos_cutoff.date()}  (pre-cutoff = IS, post-cutoff = OOS)")
    print(f"Top N      : {args.top}   TC: {args.tc} bps   Lookback: {args.lookback}d")
    print(f"{'='*64}")

    # 1. Prices
    print("\n[1/6] Loading prices …")
    prices = load_prices(UNIVERSE)
    prices = prices.loc[:, prices.count() >= WARMUP_DAYS]
    print(f"      {prices.shape[1]} tickers × {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # 1b. Fundamentals (optional)
    fund_panel = None
    if getattr(args, "use_fundamentals", False):
        print("\n[1b] Loading fundamental signals (step165) …")
        fund_panel = load_fundamentals()
        if fund_panel is not None:
            active_signals = [f for f in FUNDAMENTAL_FEATURES
                              if f in fund_panel.columns and
                              fund_panel[f].notna().any()]
            # Extend FEATURES list with active fundamental signals
            global FEATURES
            FEATURES = FEATURES + [f for f in active_signals if f not in FEATURES]
            print(f"      Active fundamental signals: {active_signals}")
            print(f"      Total features: {len(FEATURES)}")
        else:
            print("      No fundamental data found — run step165 first.")

    # 2. Features
    print("\n[2/6] Building features panel …")
    panel = build_features(prices, fund_panel=fund_panel)
    if panel.empty:
        print("  ERROR: Empty panel — not enough price history.")
        return
    print(f"      Panel: {panel.shape}  "
          f"({panel['ticker'].nunique()} tickers × {panel['date'].nunique()} dates)")

    # 3. Walk-forward predict
    print("\n[3/6] Walk-forward predictions (IS + OOS) …")
    preds = walk_forward_predict(panel, oos_cutoff, lookback=args.lookback)
    if preds is None or preds.empty or "period" not in preds.columns:
        print("  WARN: walk-forward produced no predictions (insufficient history) — skipping OOS report")
        return
    is_count  = (preds["period"] == "IS").sum()
    oos_count = (preds["period"] == "OOS").sum()
    print(f"      Total predictions: {len(preds)}  IS={is_count}  OOS={oos_count}")

    # 4. IC by period
    print("\n[4/6] Computing IC by period …")
    ic_df = compute_ic_by_period(panel, preds)
    if not ic_df.empty:
        for _, row in ic_df[ic_df["signal"] == "ensemble_score"].iterrows():
            bar = "█" * max(0, int(row["mean_ic"] * 200))
            print(f"      Ensemble [{row['period']}]  IC={row['mean_ic']:+.4f}  "
                  f"t={row['t_stat']:+.2f}  {row['status']:8s}  {bar}")

    # 5. Portfolio backtest
    print("\n[5/6] Portfolio backtest by period …")
    perf_df, equity_df = backtest_by_period(preds, prices, top_n=args.top, tc_bps=args.tc)
    if not perf_df.empty:
        for period in ["IS", "OOS"]:
            sub = perf_df[perf_df["period"] == period]
            if not sub.empty:
                total = float((1 + pd.to_numeric(sub["ml_ret"], errors="coerce").fillna(0)).prod() - 1)
                spy   = float((1 + pd.to_numeric(sub["spy_ret"], errors="coerce").fillna(0)).prod() - 1)
                print(f"      [{period}] {len(sub)} months  ML={total*100:+.2f}%  "
                      f"SPY={spy*100:+.2f}%  Alpha={((total-spy)*100):+.2f}%")

    # 6. Outputs
    print("\n[6/6] Writing outputs …")
    summary_df = build_summary(perf_df, ic_df, oos_cutoff)

    preds.to_csv(ROOT / "wf_oos_predictions.csv",    index=False)
    ic_df.to_csv(ROOT / "wf_oos_ic_by_period.csv",   index=False)
    perf_df.to_csv(ROOT / "wf_oos_backtest_perf.csv", index=False)
    equity_df.to_csv(ROOT / "wf_oos_equity_curve.csv", index=False)
    summary_df.to_csv(ROOT / "wf_oos_summary.csv",   index=False)
    write_report(summary_df, ic_df, oos_cutoff, ts)

    print(f"  wf_oos_predictions.csv    ({len(preds)} rows)")
    print(f"  wf_oos_ic_by_period.csv   ({len(ic_df)} rows)")
    print(f"  wf_oos_backtest_perf.csv  ({len(perf_df)} rows)")
    print(f"  wf_oos_equity_curve.csv   ({len(equity_df)} rows)")
    print(f"  wf_oos_summary.csv        ({len(summary_df)} rows)")
    print(f"  wf_oos_report.md")

    elapsed = time.time() - t0
    print(f"\n{'─'*64}")
    print("WALK-FORWARD OOS RESULTS")
    print(f"{'─'*64}")
    if not ic_df.empty:
        for period in ["IS", "OOS"]:
            sub = ic_df[(ic_df["period"] == period) & (ic_df["signal"] == "ensemble_score")]
            if not sub.empty:
                r = sub.iloc[0]
                print(f"  Ensemble IC [{period}] : {r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  {r['status']}")
    if not perf_df.empty:
        for period in ["IS", "OOS"]:
            sub = perf_df[perf_df["period"] == period]
            if not sub.empty:
                ml_r  = pd.to_numeric(sub["ml_ret"],  errors="coerce").fillna(0)
                spy_r = pd.to_numeric(sub["spy_ret"], errors="coerce").fillna(0)
                total = float((1 + ml_r).prod() - 1)
                spy   = float((1 + spy_r).prod() - 1)
                print(f"  Portfolio [{period}]  : ML={total*100:+.2f}%  SPY={spy*100:+.2f}%  Alpha={(total-spy)*100:+.2f}%")
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
