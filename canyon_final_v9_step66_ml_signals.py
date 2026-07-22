"""
Canyon v9  Step 66 — Walk-Forward ML Signal Generator
=======================================================
Replaces equal-weighted rule-based signals with machine-learning learned weights.

Pipeline:
  1. Compute 10 features per ticker per day (7 base signals + 3 regime features)
  2. Walk-forward train/predict:
       Train window  : rolling 252 trading days (≈1 year)
       Predict window: next 21 trading days (out-of-sample)
       Rebalance     : monthly, same schedule as Step 62
  3. Three models trained in parallel:
       Ridge           — linear combination, L2 regularised (interpretable)
       Random Forest   — 100 trees, captures non-linear interactions
       Ensemble        — equal-weight average of Ridge + RF predictions
  4. IC evaluation per model vs individual base signals
  5. Portfolio backtest with ML-composite weights vs inv-vol baseline

Features (X, computed at T close, no look-ahead):
  mom_1m           21-day log-return
  mom_3m           63-day log-return
  mom_6m           126-day log-return
  mom_12m_skip1m   252d − 21d momentum (skip-1-month)
  trend_200        price / 200-day MA − 1
  rsi_14           100 − RSI(14)  [reversal]
  inv_vol          1 / 21-day realised vol
  rank_mom         cross-sectional rank of mom_12m_skip1m
  rank_trend       cross-sectional rank of trend_200
  spy_regime       SPY above/below 200-day MA (1/0, regime indicator)

Target (y):
  Forward 21-day log-return, cross-sectionally ranked (rank ∈ [0,1])

No look-ahead: all price-based features use .shift(1), forward return uses .shift(-HOLD_PERIOD)

Outputs:
  ml_signal_scores.csv         ticker × rebalance_date ML composite scores
  ml_feature_importance.csv    feature importance from Random Forest
  ml_ic_comparison.csv         IC per signal vs ML models
  ml_backtest_perf.csv         ML portfolio monthly returns vs SPY + inv-vol
  ml_signal_report.md          full markdown report

Usage:
  python canyon_final_v9_step66_ml_signals.py
  python canyon_final_v9_step66_ml_signals.py --lookback 504    # 2-year train
  python canyon_final_v9_step66_ml_signals.py --top 8           # top 8 holdings
  python canyon_final_v9_step66_ml_signals.py --tc 10           # 10bps transaction cost
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
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── constants ─────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 252      # training window (trading days)
HOLD_PERIOD   = 21       # holding period for IC / forward return
WARMUP_DAYS   = 504      # days of price data needed before first prediction
TOP_N         = 8        # long portfolio size
TC_BPS        = 10       # one-way transaction cost in bps
ANN_FACTOR    = 252
RF_ANNUAL     = 0.053

_FEATURES_DEFAULT = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "rsi_14", "inv_vol",
    "rank_mom", "rank_trend", "spy_regime",
]


def _load_features() -> list[str]:
    """
    W20: Load pruned feature list from lgb_pruned_features.json (created by W19 SHAP analysis).
    Falls back to _FEATURES_DEFAULT if file not found or invalid.
    Only returns features that are in the known superset (safety check).
    """
    _ALL_KNOWN = {
        "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
        "trend_200", "rsi_14", "inv_vol",
        "rank_mom", "rank_trend", "spy_regime",
        "yc_spread", "hyd_proxy", "dxy_proxy",
        "neg_mom_1m", "neg_mom_3m", "neg_rsi_dev",
        "quality_score", "rank_sentiment",
        "rank_sue", "revision_score", "rank_options",
    }
    import json as _json
    p = Path(__file__).parent / "lgb_pruned_features.json"
    if p.exists():
        try:
            data = _json.load(open(p))
            pruned = [f for f in data.get("pruned_features", []) if f in _ALL_KNOWN]
            if len(pruned) >= 4:
                print(f"  [step66] W20: Using {len(pruned)} SHAP-pruned features: {pruned}")
                return pruned
        except Exception:
            pass
    return list(_FEATURES_DEFAULT)


FEATURES: list[str] = _load_features()

CORE_TICKERS = [
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP",
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
]
SUPPLEMENT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","JPM","XOM","CVX",
    "JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY","TMO","COST",
    "V","MA","HD","PYPL","NFLX","INTC","QCOM","TXN","AVGO","CRM","ADBE",
]
UNIVERSE = list(dict.fromkeys(CORE_TICKERS + SUPPLEMENT_TICKERS))  # deduplicate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Load from Step 62 cache or download fresh."""
    cache_path = ROOT / "backtest_price_cache.csv"
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=WARMUP_DAYS + LOOKBACK_DAYS + 120)

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            available = [t for t in tickers if t in cached.columns]
            if available and age_h < 24:
                print(f"  [cache] {len(available)}/{len(tickers)} tickers, {age_h:.1f}h old")
                return cached[available].dropna(how="all")
        except Exception as e:
            print(f"  [cache] Error: {e}")

    try:
        import yfinance as yf
        print(f"  [yfinance] Downloading {len(tickers)} tickers …")
        raw = yf.download(
            tickers, start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all")
        prices.to_csv(cache_path)
        print(f"  [yfinance] {prices.shape[1]} tickers, {len(prices)} days")
        return prices
    except Exception as e:
        raise RuntimeError(f"Cannot load prices: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Returns long-format DataFrame with columns:
      date, ticker, mom_1m … spy_regime, forward_ret (target)
    All features computed at close-of-day T (.shift(1) applied), so
    they are known at market open T+1.  forward_ret = return from T+1 close
    to T+HOLD_PERIOD+1 close (no look-ahead).
    """
    all_records: list[dict] = []
    spy_prices = prices.get("SPY", None)

    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        s = prices[ticker].dropna()
        if len(s) < WARMUP_DAYS:
            continue

        log_r = np.log(s / s.shift(1))

        # ── signals (shift 1 = no look-ahead) ────────────────────────────────
        mom_1m         = s.pct_change(21).shift(1)
        mom_3m         = s.pct_change(63).shift(1)
        mom_6m         = s.pct_change(126).shift(1)
        mom_252        = s.pct_change(252).shift(1)
        mom_21         = s.pct_change(21).shift(1)
        mom_12m_skip1m = (mom_252 - mom_21).shift(0)   # already shifted above
        trend_200      = (s / s.rolling(200).mean() - 1).shift(1)
        rsi_14         = _rsi(s, 14).shift(1)
        vol_21         = log_r.rolling(21).std().shift(1)
        inv_vol        = (1 / (vol_21 + 1e-10)).shift(0)

        # ── forward return (target) ───────────────────────────────────────────
        fwd_ret = np.log(s.shift(-HOLD_PERIOD) / s)   # T→T+21, look-ahead OK for target only

        # ── regime from SPY ───────────────────────────────────────────────────
        if spy_prices is not None:
            spy_ma200   = spy_prices.rolling(200).mean()
            spy_regime  = (spy_prices > spy_ma200).astype(float).shift(1)
        else:
            spy_regime = pd.Series(1.0, index=s.index)

        # W20: Bear-regime derived features (always computed, used if in pruned FEATURES)
        neg_mom_1m  = -mom_1m
        neg_mom_3m  = -mom_3m
        neg_rsi_dev = -(rsi_14 - 50)

        feat_df = pd.DataFrame({
            "ticker":         ticker,
            "mom_1m":         mom_1m,
            "mom_3m":         mom_3m,
            "mom_6m":         mom_6m,
            "mom_12m_skip1m": mom_12m_skip1m,
            "trend_200":      trend_200,
            "rsi_14":         rsi_14,
            "inv_vol":        inv_vol,
            "spy_regime":     spy_regime.reindex(s.index).fillna(1.0),
            "neg_mom_1m":     neg_mom_1m,
            "neg_mom_3m":     neg_mom_3m,
            "neg_rsi_dev":    neg_rsi_dev,
            "forward_ret":    fwd_ret,
        }, index=s.index)
        feat_df.index.name = "date"
        all_records.append(feat_df)

    if not all_records:
        return pd.DataFrame()

    panel = pd.concat(all_records).reset_index()
    # Dropna only on the core 7 price features (always available)
    _core7 = [f for f in ["mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
                           "trend_200", "rsi_14", "inv_vol"] if f in panel.columns]
    panel = panel.dropna(subset=_core7)

    # ── cross-sectional rank features (computed per date) ─────────────────────
    panel["rank_mom"]   = panel.groupby("date")["mom_12m_skip1m"].rank(pct=True)
    panel["rank_trend"] = panel.groupby("date")["trend_200"].rank(pct=True)

    # W20: Merge macro features if any FEATURES need them
    _macro_feats = [f for f in FEATURES if f in ("yc_spread", "hyd_proxy", "dxy_proxy")]
    if _macro_feats:
        fred_path = ROOT / "fred_macro_daily.csv"
        if fred_path.exists():
            try:
                macro_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
                macro_feat = pd.DataFrame(index=macro_df.index)
                if "TERM"   in macro_df.columns: macro_feat["yc_spread"] = macro_df["TERM"]
                if "HY"     in macro_df.columns:
                    hy = macro_df["HY"]
                    macro_feat["hyd_proxy"] = (hy - hy.rolling(252, min_periods=20).mean()) / \
                                              (hy.rolling(252, min_periods=20).std() + 1e-9)
                if "DOLLAR" in macro_df.columns: macro_feat["dxy_proxy"] = macro_df["DOLLAR"].pct_change(63)
                macro_feat = macro_feat.fillna(method="ffill").reset_index().rename(columns={"index": "date"})
                macro_feat["date"] = pd.to_datetime(macro_feat["date"])
                panel["date"] = pd.to_datetime(panel["date"])
                panel = panel.merge(macro_feat, on="date", how="left")
                for mf in _macro_feats:
                    if mf not in panel.columns:
                        panel[mf] = 0.0
                    panel[mf] = panel[mf].fillna(0.0)
            except Exception:
                for mf in _macro_feats:
                    panel[mf] = 0.0
        else:
            for mf in _macro_feats:
                panel[mf] = 0.0

    # W20: Merge pre-computed signal scores if in FEATURES
    _signal_map = {
        "quality_score":  ("fundamental_quality_rank.csv", "quality_score"),
        "rank_sentiment": ("finbert_sentiment.csv",         "rank_sentiment"),
        "rank_sue":       ("earnings_surprise_scores.csv",  "rank_sue"),
        "revision_score": ("earnings_revision_scores.csv",  "revision_score"),
        "rank_options":   ("options_signals.csv",           "rank_options"),
    }
    for feat_name, (fname, col) in _signal_map.items():
        if feat_name not in FEATURES:
            continue
        p = ROOT / fname
        if p.exists():
            try:
                sig_df = pd.read_csv(p)
                if "ticker" in sig_df.columns and col in sig_df.columns:
                    sig_map = sig_df.set_index("ticker")[col].to_dict()
                    panel[feat_name] = panel["ticker"].map(sig_map).fillna(50.0)
                else:
                    panel[feat_name] = 50.0
            except Exception:
                panel[feat_name] = 50.0
        else:
            panel[feat_name] = 50.0

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-forward ML trainer
# ─────────────────────────────────────────────────────────────────────────────

def _cross_rank_target(y: pd.Series) -> pd.Series:
    """Cross-sectionally rank the target to reduce outlier impact."""
    return y.rank(pct=True)


def walk_forward_train_predict(
    panel: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS,
    hold: int     = HOLD_PERIOD,
) -> pd.DataFrame:
    """
    For each rebalance date:
      1. Take training data from date−lookback to date (exclusive)
      2. Drop rows with NaN forward_ret in train set
      3. Fit Ridge + RandomForest on ranked target
      4. Predict cross-sectional scores for the rebalance date
    Returns predictions_df: date, ticker, ridge_score, rf_score, lgbm_score, ensemble_score
    """
    dates = panel["date"].sort_values().unique()
    # Monthly rebalance dates (first trading day of each month)
    rebalance_dates = []
    prev_month = None
    for d in dates:
        m = pd.Timestamp(d).month
        if m != prev_month:
            rebalance_dates.append(d)
            prev_month = m

    # Warm up: skip first WARMUP_DAYS worth of dates
    min_date = dates[min(lookback, len(dates) - 1)]
    rebalance_dates = [d for d in rebalance_dates if d >= min_date]

    all_predictions: list[dict] = []
    scaler_ridge = StandardScaler()
    scaler_rf    = StandardScaler()
    scaler_lgb   = StandardScaler()

    print(f"  [ML] Walk-forward: {len(rebalance_dates)} rebalance dates, "
          f"lookback={lookback}d, hold={hold}d")

    for idx, reb_date in enumerate(rebalance_dates):
        # Training window: [reb_date - lookback, reb_date)
        train_end   = pd.Timestamp(reb_date) - pd.Timedelta(days=1)
        train_start = pd.Timestamp(reb_date) - pd.Timedelta(days=lookback * 1.6)  # calendar days

        train_mask = (
            (panel["date"] >= train_start) &
            (panel["date"] <= train_end) &
            (panel["forward_ret"].notna())
        )
        train_df = panel[train_mask].copy()

        if len(train_df) < 200:
            continue   # not enough training data

        # Cross-sectionally rank the target within each date
        train_df["y_ranked"] = train_df.groupby("date")["forward_ret"].rank(pct=True)

        X_train = train_df[FEATURES].values
        y_train = train_df["y_ranked"].values

        # Prediction set: exact rebalance date
        pred_mask = panel["date"] == reb_date
        pred_df   = panel[pred_mask].dropna(subset=FEATURES)
        if pred_df.empty:
            continue

        X_pred = pred_df[FEATURES].values

        # ── Ridge ──────────────────────────────────────────────────────────────
        try:
            scaler_ridge.fit(X_train)
            X_tr_sc = scaler_ridge.transform(X_train)
            X_pr_sc = scaler_ridge.transform(X_pred)
            ridge = Ridge(alpha=50.0)
            ridge.fit(X_tr_sc, y_train)
            ridge_scores = ridge.predict(X_pr_sc)
        except Exception:
            ridge_scores = np.zeros(len(pred_df))

        # ── Random Forest ──────────────────────────────────────────────────────
        try:
            scaler_rf.fit(X_train)
            X_tr_rf = scaler_rf.transform(X_train)
            X_pr_rf = scaler_rf.transform(X_pred)
            rf = RandomForestRegressor(
                n_estimators=80, max_depth=4,
                min_samples_leaf=10, random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr_rf, y_train)
            rf_scores = rf.predict(X_pr_rf)
            # Feature importance (last rebalance date only for speed)
            if idx == len(rebalance_dates) - 1:
                _fi = dict(zip(FEATURES, rf.feature_importances_))
            else:
                _fi = None
        except Exception:
            rf_scores = np.zeros(len(pred_df))
            _fi = None

        # ── LightGBM ───────────────────────────────────────────────────────────
        if _HAS_LGB:
            try:
                scaler_lgb.fit(X_train)
                X_tr_lgb = scaler_lgb.transform(X_train)
                X_pr_lgb = scaler_lgb.transform(X_pred)
                lgb = LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                lgb.fit(X_tr_lgb, y_train)
                lgb_scores = lgb.predict(X_pr_lgb)
                # Override feature importance with LightGBM (more reliable than RF)
                if idx == len(rebalance_dates) - 1:
                    _fi = dict(zip(FEATURES, lgb.feature_importances_))
            except Exception:
                lgb_scores = np.zeros(len(pred_df))
        else:
            lgb_scores = np.zeros(len(pred_df))

        # ── Ensemble (Ridge 30% + RF 30% + LightGBM 40%) ───────────────────────
        if _HAS_LGB and lgb_scores.any():
            ensemble_scores = 0.30 * ridge_scores + 0.30 * rf_scores + 0.40 * lgb_scores
        else:
            ensemble_scores = 0.50 * ridge_scores + 0.50 * rf_scores

        for i, (_, row) in enumerate(pred_df.iterrows()):
            all_predictions.append({
                "rebalance_date":  str(reb_date)[:10],
                "ticker":          row["ticker"],
                "ridge_score":     float(ridge_scores[i]),
                "rf_score":        float(rf_scores[i]),
                "lgbm_score":      float(lgb_scores[i]),
                "ensemble_score":  float(ensemble_scores[i]),
                "n_train":         len(train_df),
            })

        if (idx + 1) % 10 == 0 or idx == len(rebalance_dates) - 1:
            print(f"  [ML] {idx+1}/{len(rebalance_dates)} done — "
                  f"{reb_date!s:.10s}  n_train={len(train_df)}")

    preds_df = pd.DataFrame(all_predictions)
    return preds_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. IC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_table(panel: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman IC for each base signal and each ML model.
    IC = corr(signal_at_T, forward_ret_T+1..T+21) per rebalance date.
    """
    # Merge predictions with panel forward_ret
    reb_dates = preds_df["rebalance_date"].unique()
    panel_sub = panel[panel["date"].astype(str).str[:10].isin([str(d)[:10] for d in reb_dates])].copy()
    panel_sub["rebalance_date"] = panel_sub["date"].astype(str).str[:10]

    merged = preds_df.merge(
        panel_sub[["rebalance_date","ticker","forward_ret"] + FEATURES],
        on=["rebalance_date","ticker"], how="inner",
    )
    merged = merged.dropna(subset=["forward_ret"])

    if merged.empty:
        return pd.DataFrame()

    signal_cols = FEATURES + ["ridge_score", "rf_score", "lgbm_score", "ensemble_score"]
    records = []
    for col in signal_cols:
        if col not in merged.columns:
            continue
        ics = []
        for date, grp in merged.groupby("rebalance_date"):
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
            records.append({"signal": col, "n_obs": 0, "mean_ic": 0.0,
                            "std_ic": 0.0, "t_stat": 0.0, "p_value": 1.0,
                            "ic_positive_pct": "—", "status": "NO_DATA"})
            continue

        arr = np.array(ics)
        n   = len(arr)
        mean_ic = float(arr.mean())
        std_ic  = float(arr.std())
        t_stat  = float(mean_ic / (std_ic / np.sqrt(n) + 1e-10))
        p_val   = float(2 * (1 - min(abs(t_stat), 10) / 10))  # approx
        from scipy.stats import ttest_1samp
        _, p_val = ttest_1samp(arr, 0)

        ic_pos_pct = f"{(arr > 0).mean() * 100:.1f}%"

        if   mean_ic > 0.05 and abs(t_stat) > 2.0:   status = "STRONG"
        elif mean_ic > 0.03 and abs(t_stat) > 2.0:   status = "USABLE"
        elif mean_ic > 0.0  and abs(t_stat) > 1.0:   status = "WEAK"
        elif mean_ic < 0.0:                           status = "NEGATIVE"
        else:                                          status = "WEAK"

        records.append({
            "signal":         col,
            "n_obs":          n,
            "mean_ic":        round(mean_ic, 4),
            "std_ic":         round(std_ic, 4),
            "t_stat":         round(t_stat, 2),
            "p_value":        round(float(p_val), 4),
            "ic_positive_pct": ic_pos_pct,
            "status":         status,
        })

    ic_df = pd.DataFrame(records)
    if not ic_df.empty:
        ic_df = ic_df.sort_values("mean_ic", ascending=False).reset_index(drop=True)
    return ic_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ML-weighted portfolio backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest_ml_portfolio(
    preds_df: pd.DataFrame,
    prices:   pd.DataFrame,
    top_n:    int = TOP_N,
    tc_bps:   float = TC_BPS,
) -> pd.DataFrame:
    """
    At each rebalance date:
      - Rank tickers by ensemble_score (descending)
      - Take top_n tickers
      - Equal-weight (1/top_n per ticker)
      - Compute period return to next rebalance date (using actual prices)
      - Apply 10bps TC on estimated turnover
    Returns monthly performance DataFrame.
    """
    if preds_df.empty or prices.empty:
        return pd.DataFrame()

    reb_dates = sorted(preds_df["rebalance_date"].unique())
    records   = []
    prev_holdings: set[str] = set()

    for i, reb_date in enumerate(reb_dates[:-1]):
        next_date = reb_dates[i + 1]

        # Top N by ensemble score
        grp = preds_df[preds_df["rebalance_date"] == reb_date].copy()
        grp["_score"] = pd.to_numeric(grp["ensemble_score"], errors="coerce").fillna(0)
        top = grp.nlargest(top_n, "_score")["ticker"].tolist()

        if not top:
            continue

        # Get prices for period
        reb_ts  = pd.Timestamp(reb_date)
        next_ts = pd.Timestamp(next_date)

        # Find actual available trading dates
        avail_dates = prices.index[
            (prices.index >= reb_ts) & (prices.index <= next_ts)
        ]
        if len(avail_dates) < 2:
            continue

        t_start = avail_dates[0]
        t_end   = avail_dates[-1]

        # Portfolio return (equal weight)
        rets = []
        for tk in top:
            if tk in prices.columns:
                p_s = prices[tk].loc[t_start]
                p_e = prices[tk].loc[t_end]
                if pd.notna(p_s) and pd.notna(p_e) and p_s > 0:
                    rets.append(float(p_e / p_s - 1))
        if not rets:
            continue
        port_ret = float(np.mean(rets))

        # SPY benchmark
        spy_ret = 0.0
        if "SPY" in prices.columns:
            s_s = prices["SPY"].loc[t_start]
            s_e = prices["SPY"].loc[t_end]
            if pd.notna(s_s) and s_s > 0:
                spy_ret = float(s_e / s_s - 1)

        # Transaction cost
        new_set  = set(top)
        turnover = len(new_set.symmetric_difference(prev_holdings)) / max(len(new_set | prev_holdings), 1)
        tc       = turnover * tc_bps / 10000
        net_ret  = port_ret - tc

        records.append({
            "rebalance_date": reb_date,
            "period_end":     str(next_date)[:10],
            "ml_ret":         round(net_ret, 6),
            "spy_ret":        round(spy_ret, 6),
            "alpha":          round(net_ret - spy_ret, 6),
            "n_held":         len(top),
            "turnover_pct":   round(turnover * 100, 1),
            "tc_cost_bps":    round(tc * 10000, 1),
            "tickers":        " | ".join(sorted(top)),
        })
        prev_holdings = new_set

    df = pd.DataFrame(records)
    if not df.empty:
        df["ml_cum"]    = (1 + df["ml_ret"]).cumprod() - 1
        df["bench_cum"] = (1 + df["spy_ret"]).cumprod() - 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Train RF on entire dataset and extract feature importance."""
    train = panel.dropna(subset=FEATURES + ["forward_ret"]).copy()
    if len(train) < 500:
        return pd.DataFrame()

    train["y_ranked"] = train.groupby("date")["forward_ret"].rank(pct=True)
    X = train[FEATURES].values
    y = train["y_ranked"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=150, max_depth=5,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    rf.fit(X_sc, y)

    ridge = Ridge(alpha=50.0)
    ridge.fit(X_sc, y)

    rows = []
    for feat, fi, coef in zip(FEATURES, rf.feature_importances_, ridge.coef_):
        rows.append({
            "feature":    feat,
            "rf_importance":    round(float(fi), 5),
            "ridge_coef":       round(float(coef), 5),
            "rf_rank":          0,  # filled below
        })
    df = pd.DataFrame(rows).sort_values("rf_importance", ascending=False).reset_index(drop=True)
    df["rf_rank"] = range(1, len(df) + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_summary(perf_df: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not perf_df.empty and "ml_ret" in perf_df.columns:
        ml_rets = pd.to_numeric(perf_df["ml_ret"], errors="coerce").dropna()
        spy_rets = pd.to_numeric(perf_df["spy_ret"], errors="coerce").dropna()
        n = len(ml_rets)

        if n > 0:
            total_ret = float((1 + ml_rets).prod() - 1)
            spy_total = float((1 + spy_rets).prod() - 1) if not spy_rets.empty else 0.0
            alpha     = total_ret - spy_total
            sharpe    = float(ml_rets.mean() / (ml_rets.std() + 1e-10) * np.sqrt(12))
            sortino_d = ml_rets[ml_rets < 0].std()
            sortino   = float(ml_rets.mean() / (sortino_d + 1e-10) * np.sqrt(12))
            cum       = (1 + ml_rets).cumprod()
            roll_max  = cum.cummax()
            dd        = (cum / roll_max - 1)
            max_dd    = float(dd.min())
            calmar    = abs(total_ret / (max_dd + 1e-10)) if max_dd < 0 else 0.0
            win_rate  = float((ml_rets > spy_rets).mean() * 100) if len(spy_rets) == n else 0.0

            rows += [
                ("ML Total Return",          f"{total_ret*100:.2f}%",  f"{spy_total*100:.2f}% (SPY)", _assess(total_ret, [0.30, 0.15, 0.05])),
                ("Total Alpha vs SPY",        f"{alpha*100:+.2f}%",     "0.00%",                        _assess(alpha, [0.15, 0.05, 0.0])),
                ("Annualised Sharpe",         f"{sharpe:.2f}",          "~0.60 (SPY hist.)",            _assess(sharpe, [1.0, 0.6, 0.3])),
                ("Annualised Sortino",        f"{sortino:.2f}",         "~0.90 (SPY hist.)",            _assess(sortino, [1.5, 0.9, 0.5])),
                ("Calmar Ratio",              f"{calmar:.2f}",          ">1.0 good",                    _assess(calmar, [1.0, 0.5, 0.2])),
                ("Max Drawdown",              f"{max_dd*100:.2f}%",     "lower is better",              "RISK" if max_dd < -0.20 else "USABLE"),
                ("Monthly Win Rate vs SPY",   f"{win_rate:.1f}%",       "50%",                          _assess(win_rate / 100, [0.55, 0.50, 0.45])),
                ("Periods Tested (months)",   str(n),                   ">36 reliable",                 _assess(n, [36, 24, 12])),
            ]

    if not ic_df.empty:
        ml_ic_rows = ic_df[ic_df["signal"].isin(["ensemble_score","ridge_score","rf_score"])]
        best_ml = ml_ic_rows.loc[ml_ic_rows["mean_ic"].idxmax()] if not ml_ic_rows.empty else None
        if best_ml is not None:
            rows.append(("Best ML Signal IC", f"{best_ml['mean_ic']:.4f} ({best_ml['signal']})",
                          "IC > 0.05 = STRONG", best_ml.get("status", "—")))
        base_ic_rows = ic_df[~ic_df["signal"].isin(["ensemble_score","ridge_score","rf_score"])]
        best_base = base_ic_rows.loc[base_ic_rows["mean_ic"].idxmax()] if not base_ic_rows.empty else None
        if best_base is not None:
            rows.append(("Best Base Signal IC", f"{best_base['mean_ic']:.4f} ({best_base['signal']})",
                          "IC > 0.05 = STRONG", best_base.get("status", "—")))
        if best_ml is not None and best_base is not None:
            improvement = best_ml["mean_ic"] - best_base["mean_ic"]
            rows.append(("ML IC Improvement", f"{improvement:+.4f}",
                          "Positive = ML adds value", "STRONG" if improvement > 0.01 else
                          ("USABLE" if improvement > 0 else "NEGATIVE")))

    return pd.DataFrame(rows, columns=["metric", "value", "benchmark", "assessment"])


def _assess(val, thresholds: list) -> str:
    if val >= thresholds[0]: return "STRONG"
    if val >= thresholds[1]: return "USABLE"
    if val >= thresholds[2]: return "WEAK"
    return "POOR"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(summary_df: pd.DataFrame, ic_df: pd.DataFrame,
                 fi_df: pd.DataFrame, ts: str) -> None:
    lines = [
        "# Canyon v9 — ML Signal Generator Report (Step 66)",
        f"Generated: {ts}",
        "",
        "## Summary",
        "",
        "| Metric | ML Value | Benchmark | Assessment |",
        "|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"| {row['metric']} | {row['value']} | {row['benchmark']} | {row['assessment']} |")

    lines += [
        "",
        "## IC Comparison: ML Models vs Base Signals",
        "",
        "| Signal | Mean IC | t-stat | IC+ Rate | Status |",
        "|---|---|---|---|---|",
    ]
    for _, row in ic_df.iterrows():
        lines.append(
            f"| {row['signal']} | {row['mean_ic']:+.4f} | {row['t_stat']:+.2f} | "
            f"{row.get('ic_positive_pct','—')} | {row['status']} |"
        )

    if not fi_df.empty:
        lines += [
            "",
            "## Feature Importance (Random Forest)",
            "",
            "| Rank | Feature | RF Importance | Ridge Coef |",
            "|---|---|---|---|",
        ]
        for _, row in fi_df.iterrows():
            lines.append(
                f"| {int(row['rf_rank'])} | {row['feature']} | "
                f"{row['rf_importance']:.4f} | {row['ridge_coef']:+.4f} |"
            )

    lines += [
        "",
        "## Model Architecture",
        "- **Ridge**: α=50, L2-regularised linear model on 10 features, cross-sectionally ranked target",
        "- **Random Forest**: 80 trees, max_depth=4, min_samples_leaf=10",
        "- **Ensemble**: equal-average of Ridge and RF predictions",
        "- **Training**: rolling 252-day window, monthly rebalance",
        "- **No look-ahead**: features use .shift(1), target uses forward 21-day return",
        "",
        "## Limitations",
        "- Survivorship bias: universe = current tickers (still trading)",
        "- 252-day training window is relatively short for complex models",
        "- Feature set is price-derived only (no fundamental, no macro)",
        "- Random Forest may overfit on small universe (consider regularisation)",
        "- Next step: add fundamental features (P/E, ROE, earnings revision)",
    ]

    p = ROOT / "ml_signal_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 66 — ML Signal Generator")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--top",      type=int, default=TOP_N)
    parser.add_argument("--tc",       type=float, default=TC_BPS)
    args = parser.parse_args()

    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*62}")
    print(f"Canyon v9 Step 66 — Walk-Forward ML Signal Generator")
    print(f"Lookback: {args.lookback}d  TopN: {args.top}  TC: {args.tc}bps")
    print(f"{'='*62}")

    # ── 1. Prices ─────────────────────────────────────────────────────────────
    print("\n[1/7] Loading prices …")
    prices = load_prices(UNIVERSE)
    prices = prices.loc[:, prices.count() >= WARMUP_DAYS]
    print(f"      {prices.shape[1]} tickers  ×  {len(prices)} days")

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("\n[2/7] Building features panel …")
    panel = build_features(prices)
    if panel.empty:
        print("  ERROR: Empty panel — not enough price history.")
        return
    print(f"      Panel shape: {panel.shape}  "
          f"({panel['ticker'].nunique()} tickers  ×  {panel['date'].nunique()} dates)")

    # ── 3. Walk-forward train/predict ─────────────────────────────────────────
    print("\n[3/7] Walk-forward ML training …")
    preds_df = walk_forward_train_predict(panel, lookback=args.lookback)
    print(f"      {len(preds_df)} predictions across "
          f"{preds_df['rebalance_date'].nunique()} rebalance dates")

    # ── 4. IC comparison ──────────────────────────────────────────────────────
    print("\n[4/7] Computing IC table …")
    ic_df = compute_ic_table(panel, preds_df)
    if not ic_df.empty:
        print(f"      IC results for {len(ic_df)} signals:")
        for _, row in ic_df.iterrows():
            bar = "█" * max(0, int(row["mean_ic"] * 200))
            print(f"        {row['signal']:20s}  IC={row['mean_ic']:+.4f}  "
                  f"t={row['t_stat']:+.2f}  {row['status']:8s}  {bar}")

    # ── 5. Portfolio backtest ─────────────────────────────────────────────────
    print("\n[5/7] ML portfolio backtest …")
    perf_df = backtest_ml_portfolio(preds_df, prices, top_n=args.top, tc_bps=args.tc)
    if not perf_df.empty:
        total_ml  = float((1 + pd.to_numeric(perf_df["ml_ret"], errors="coerce").fillna(0)).prod() - 1)
        total_spy = float((1 + pd.to_numeric(perf_df["spy_ret"], errors="coerce").fillna(0)).prod() - 1)
        print(f"      {len(perf_df)} periods  |  "
              f"ML={total_ml*100:.2f}%  SPY={total_spy*100:.2f}%  "
              f"Alpha={((total_ml - total_spy)*100):+.2f}%")

    # ── 6. Feature importance ─────────────────────────────────────────────────
    print("\n[6/7] Feature importance (full dataset RF) …")
    fi_df = compute_feature_importance(panel, lookback=args.lookback)
    if not fi_df.empty:
        print("      Top features by RF importance:")
        for _, row in fi_df.head(5).iterrows():
            bar = "█" * max(1, int(row["rf_importance"] * 200))
            print(f"        {row['feature']:20s}  {row['rf_importance']:.4f}  {bar}")

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    print("\n[7/7] Writing outputs …")
    summary_df = build_ml_summary(perf_df, ic_df)

    preds_df.to_csv(ROOT / "ml_signal_scores.csv", index=False)
    print(f"  [scores]      ml_signal_scores.csv")

    if not ic_df.empty:
        ic_df.to_csv(ROOT / "ml_ic_comparison.csv", index=False)
        print(f"  [ic]          ml_ic_comparison.csv")

    if not perf_df.empty:
        perf_df.to_csv(ROOT / "ml_backtest_perf.csv", index=False)
        print(f"  [backtest]    ml_backtest_perf.csv")

    if not fi_df.empty:
        fi_df.to_csv(ROOT / "ml_feature_importance.csv", index=False)
        print(f"  [importance]  ml_feature_importance.csv")

    if not summary_df.empty:
        summary_df.to_csv(ROOT / "ml_summary.csv", index=False)

    write_report(summary_df, ic_df, fi_df, ts)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'─'*62}")
    print("ML SIGNAL GENERATOR RESULTS")
    print(f"{'─'*62}")
    if not ic_df.empty:
        ml_rows = ic_df[ic_df["signal"].isin(["ensemble_score","ridge_score","rf_score"])]
        base_rows = ic_df[~ic_df["signal"].isin(["ensemble_score","ridge_score","rf_score"])]
        best_ml_ic   = ml_rows["mean_ic"].max()   if not ml_rows.empty   else 0.0
        best_base_ic = base_rows["mean_ic"].max() if not base_rows.empty else 0.0
        print(f"  Best ML IC    : {best_ml_ic:+.4f}")
        print(f"  Best Base IC  : {best_base_ic:+.4f}")
        print(f"  IC Improvement: {(best_ml_ic - best_base_ic):+.4f}")
    if not perf_df.empty:
        total_ml  = float((1 + pd.to_numeric(perf_df["ml_ret"],  errors="coerce").fillna(0)).prod() - 1)
        total_spy = float((1 + pd.to_numeric(perf_df["spy_ret"], errors="coerce").fillna(0)).prod() - 1)
        print(f"\n  ML Portfolio  : {total_ml*100:+.2f}%")
        print(f"  SPY           : {total_spy*100:+.2f}%")
        print(f"  Net Alpha     : {(total_ml - total_spy)*100:+.2f}%")
    print(f"\n  Runtime       : {elapsed:.1f}s")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
