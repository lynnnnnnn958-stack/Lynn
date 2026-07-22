#!/usr/bin/env python3
"""
Canyon — step_ml_alpha.py
==========================
LightGBM alpha signals for three holding horizons:
  short   → predict 5-day forward return   (rebalance weekly)
  medium  → predict 21-day forward return  (rebalance bi-weekly)
  long    → predict 63-day forward return  (rebalance monthly)

Feature set: technical factors computed from sp500_price_cache.csv
  (momentum 1m/3m/6m/12m, RSI, trend-200, inv-vol, short-rev,
   52w-high ratio, idiosyncratic momentum vs SPY)
  + available fundamental signals merged from existing CSVs

Training protocol:
  - Walk-forward: train on all data up to (today − 126 days), predict today
  - Cross-sectional rank-normalise features and targets to [0,1] by date
    before fitting (eliminates macro-level shifts as spurious signal)
  - Winsorise targets at 5th/95th percentile to reduce outlier influence
  - LightGBM DART with early stopping on a held-out validation slice

Outputs:
  ml_alpha_scores.csv   ticker | ml_short | ml_medium | ml_long (all 0-100)
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
TODAY      = datetime.now().strftime("%Y-%m-%d")
MODELS_DIR = ROOT / "model_cache"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_MAX_AGE_DAYS = 7   # retrain if model file is older than this

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")


# ── Config ───────────────────────────────────────────────────────────────────

HORIZONS = {
    "ml_short":  5,    # trading days
    "ml_medium": 21,
    "ml_long":   63,
}
MIN_HISTORY_DAYS   = 252   # need at least 1 year to compute long-window features
TRAIN_CUTOFF_DAYS  = 84    # reserve last 84 trading days from training (≈ 4 months)
FEATURE_COLS: list[str] = []   # filled after compute_features()


# ── Feature engineering from price matrix ────────────────────────────────────

def _ret(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    return prices.pct_change(n)

def _rolling_rank(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Cross-sectional rank within each rolling window — slow but avoids lookahead."""
    return df.rank(axis=1, pct=True)

def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-format DataFrame: index = (date, ticker), columns = features.
    Only dates with ≥ MIN_HISTORY_DAYS of price history are included.
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    spy = prices["SPY"] if "SPY" in prices.columns else None

    # --- raw returns ---
    r1d  = _ret(prices, 1)
    r5d  = _ret(prices, 5)
    r21d = _ret(prices, 21)
    r63d = _ret(prices, 63)
    r252d= _ret(prices, 252)

    # short-term reversal (1-week) — negative = contrarian
    rev5 = -r5d

    # momentum skip-1-month (12m-1m)
    r21d_lag  = r21d.shift(21)
    mom_skip  = r252d - r21d_lag if r21d_lag is not None else r252d

    # trend (price / 200d SMA − 1)
    sma200 = prices.rolling(200, min_periods=150).mean()
    trend  = (prices / sma200) - 1

    # RSI-14
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi14 = 100 - 100 / (1 + rs)

    # 52-week high ratio
    high52 = prices.rolling(252, min_periods=100).max()
    ratio52 = prices / (high52 + 1e-9)

    # inverse vol (21-day)
    vol21 = r1d.rolling(21, min_periods=10).std()
    inv_vol = 1.0 / (vol21 + 1e-9)

    # idiosyncratic momentum vs SPY
    if spy is not None:
        spy_r21 = r21d[["SPY"]] if "SPY" in r21d.columns else None
        # simple beta using 63-day rolling corr/vol
        spy_vol  = r1d["SPY"].rolling(63, min_periods=21).std()
        cov_mat  = r1d.rolling(63, min_periods=21).cov(r1d["SPY"])
        beta     = cov_mat.div(spy_vol**2 + 1e-9, axis=0)
        idio_mom = r21d.subtract(beta.multiply(r21d["SPY"], axis=0), axis=0)
    else:
        idio_mom = r21d.copy()

    # --- cross-sectional rank each feature by date ---
    def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    features = {
        "f_mom_1m":   cs_rank(r21d),
        "f_mom_3m":   cs_rank(r63d),
        "f_mom_6m":   cs_rank(r252d / 2),        # approx 6m via half-year ret
        "f_mom_12m":  cs_rank(r252d),
        "f_mom_skip": cs_rank(mom_skip),
        "f_rev5":     cs_rank(rev5),
        "f_trend200": cs_rank(trend),
        "f_rsi14":    cs_rank(rsi14),
        "f_ratio52w": cs_rank(ratio52),
        "f_inv_vol":  cs_rank(inv_vol),
        "f_idio_mom": cs_rank(idio_mom),
    }

    # stack into long format
    frames = []
    for fname, fdf in features.items():
        melted = fdf.stack(future_stack=True).rename(fname)
        frames.append(melted)

    combined = pd.concat(frames, axis=1)
    combined.index.names = ["date", "ticker"]
    combined = combined.dropna(how="all")

    # only keep dates beyond MIN_HISTORY_DAYS
    all_dates = prices.index
    if len(all_dates) > MIN_HISTORY_DAYS:
        start_date = all_dates[MIN_HISTORY_DAYS]
        combined = combined[combined.index.get_level_values("date") >= start_date]

    global FEATURE_COLS
    FEATURE_COLS = [c for c in combined.columns if c.startswith("f_")]

    return combined


def compute_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward return shifted back so it aligns with the signal date."""
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    fwd = prices.pct_change(horizon).shift(-horizon)
    # winsorise
    q05 = fwd.stack(future_stack=True).quantile(0.05)
    q95 = fwd.stack(future_stack=True).quantile(0.95)
    fwd = fwd.clip(lower=q05, upper=q95)
    # cs-rank
    fwd = fwd.rank(axis=1, pct=True)
    stacked = fwd.stack(future_stack=True)
    stacked.index.names = ["date", "ticker"]
    return stacked


# ── LightGBM training ────────────────────────────────────────────────────────

def train_and_predict(
    features: pd.DataFrame,
    target: pd.Series,
    all_dates: pd.DatetimeIndex,
    signal_name: str = "",
    retrain: bool = True,
) -> pd.Series:
    """
    Walk-forward: train on dates[:−TRAIN_CUTOFF_DAYS], predict on latest date.
    If retrain=False, loads cached model from MODELS_DIR and skips training.
    Returns a Series indexed by ticker with predicted rank (0-100).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        err("lightgbm not installed — run: pip install lightgbm")
        return pd.Series(dtype=float)

    latest  = all_dates[-1]
    pred_idx = features.index[features.index.get_level_values("date") == latest]

    if pred_idx.empty:
        warn("  No prediction rows for latest date — skipping")
        return pd.Series(dtype=float)

    X_pred = features.loc[pred_idx, FEATURE_COLS].fillna(0.5)

    # ── Inference-only mode: load saved model ────────────────────────────
    if not retrain:
        model = _load_model(signal_name)
        if model is not None:
            pred = pd.Series(model.predict(X_pred), index=pred_idx)
            pred.index = pred.index.get_level_values("ticker")
            return (pred.rank(pct=True) * 100).round(2)
        warn(f"  No cached model for {signal_name} — falling back to full train")

    # ── Full training ─────────────────────────────────────────────────────
    cutoff = all_dates[-TRAIN_CUTOFF_DAYS] if len(all_dates) > TRAIN_CUTOFF_DAYS else all_dates[0]

    common_idx = features.index.intersection(target.index)
    train_idx  = common_idx[common_idx.get_level_values("date") < cutoff]

    if len(train_idx) < 500:
        warn(f"  Insufficient training data ({len(train_idx)} rows) — skipping ML")
        return pd.Series(dtype=float)

    X_train = features.loc[train_idx, FEATURE_COLS].fillna(0.5)
    y_train = target.loc[train_idx].fillna(0.5)

    # validation slice: last 10% of training
    val_n   = max(1, int(len(X_train) * 0.10))
    X_val   = X_train.iloc[-val_n:]
    y_val   = y_train.iloc[-val_n:]
    X_tr    = X_train.iloc[:-val_n]
    y_tr    = y_train.iloc[:-val_n]

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "dart",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples":20,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "n_estimators":     300,
        "verbose":          -1,
        "random_state":     42,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    if signal_name:
        _save_model(model, signal_name)

    pred = pd.Series(model.predict(X_pred), index=pred_idx)
    pred.index = pred.index.get_level_values("ticker")

    # cross-sectionally rank the predictions → 0-100 score
    score = pred.rank(pct=True) * 100
    return score.round(2)


# ── Merge fundamental signals ─────────────────────────────────────────────────

def load_fundamental_context() -> pd.DataFrame:
    """Load whatever fundamental CSVs exist → one row per ticker."""
    dfs = []

    maps = [
        ("fundamental_quality_rank.csv",  "quality_score",     "f_quality"),
        ("earnings_revision_scores.csv",   "revision_score",    "f_revision"),
        ("earnings_surprise_scores.csv",   "rank_sue",          "f_sue"),
        ("momentum_scores.csv",            "momentum_score",    "f_momentum"),
        ("accrual_scores.csv",             "accrual_score",     "f_accruals"),
        ("piotroski_scores.csv",           "piotroski_score",   "f_piotroski"),
    ]
    for fname, col, new_col in maps:
        fpath = ROOT / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, usecols=lambda c: c in ("ticker", col))
            if "ticker" not in df.columns or col not in df.columns:
                continue
            df = df[["ticker", col]].rename(columns={col: new_col})
            df = df.set_index("ticker")[new_col]
            df = (df.rank(pct=True)).rename(new_col)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def _should_retrain() -> bool:
    """Return True if models need retraining (Monday, or model file > MODEL_MAX_AGE_DAYS old)."""
    today = datetime.now()
    if today.weekday() == 0:    # 0 = Monday
        log("Monday detected — forcing full LightGBM retrain")
        return True
    # Check age of saved models
    for sig in HORIZONS:
        mpath = MODELS_DIR / f"lgb_{sig}.pkl"
        if not mpath.exists():
            log(f"No cached model for {sig} — will train")
            return True
        age = (today - datetime.fromtimestamp(mpath.stat().st_mtime)).days
        if age >= MODEL_MAX_AGE_DAYS:
            log(f"Model {sig} is {age}d old (>{MODEL_MAX_AGE_DAYS}d) — retraining")
            return True
    ok(f"Cached models are fresh — using inference-only mode (save ~8 min)")
    return False


def _save_model(model, signal_name: str):
    try:
        import joblib
        joblib.dump(model, MODELS_DIR / f"lgb_{signal_name}.pkl")
    except Exception as e:
        warn(f"Could not save model {signal_name}: {e}")


def _load_model(signal_name: str):
    try:
        import joblib
        p = MODELS_DIR / f"lgb_{signal_name}.pkl"
        return joblib.load(p) if p.exists() else None
    except Exception:
        return None


def main():
    print(f"\n{BOLD}Canyon — ML Alpha (LightGBM){RESET}  {TODAY}")

    # 1. Load prices
    price_path = ROOT / "sp500_price_cache.csv"
    if not price_path.exists():
        err("sp500_price_cache.csv not found — run step_daily_price_signals.py first")
        return
    log("Loading price cache …")
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    all_dates = prices.index
    ok(f"Price cache: {len(all_dates)} days × {prices.shape[1]} tickers")

    if len(all_dates) < MIN_HISTORY_DAYS + 63 + 10:
        err(f"Need ≥ {MIN_HISTORY_DAYS + 73} days of price history — got {len(all_dates)}")
        return

    retrain = _should_retrain()

    # 2. Compute features
    log("Computing technical features …")
    if retrain:
        feat_df = compute_features(prices)
    else:
        # Inference-only: only compute the last TRAIN_CUTOFF_DAYS + 70 dates
        slim_prices = prices.iloc[-(TRAIN_CUTOFF_DAYS + 70):]
        feat_df = compute_features(slim_prices)
    ok(f"Feature matrix: {len(feat_df):,} rows × {len(FEATURE_COLS)} features")

    # 3. Merge fundamental context into latest-date features
    fund_ctx = load_fundamental_context()
    if not fund_ctx.empty:
        latest = all_dates[-1]
        idx_latest = feat_df.index.get_level_values("date") == latest
        fund_cols = fund_ctx.columns.tolist()
        # broadcast fundamental values to latest date rows
        latest_tickers = feat_df.loc[idx_latest].index.get_level_values("ticker")
        for col in fund_cols:
            if col not in feat_df.columns:
                feat_df[col] = np.nan
        feat_df.loc[feat_df.index.get_level_values("date") == latest, fund_cols] = \
            fund_ctx.reindex(latest_tickers.to_list())[fund_cols].values
        # also add to FEATURE_COLS for prediction
        for col in fund_cols:
            if col not in FEATURE_COLS:
                FEATURE_COLS.append(col)
        ok(f"Merged {len(fund_cols)} fundamental features for latest date")

    # 4. Train + predict for each horizon
    results: dict[str, pd.Series] = {}
    for signal_name, horizon in HORIZONS.items():
        action = "Training" if retrain else "Scoring (cached)"
        log(f"{action} LightGBM for {signal_name} (horizon={horizon}d) …")
        fwd = compute_forward_returns(prices, horizon) if retrain else pd.Series(dtype=float)
        score = train_and_predict(feat_df, fwd, all_dates, signal_name=signal_name, retrain=retrain)
        if not score.empty:
            results[signal_name] = score
            ok(f"{signal_name}: {len(score)} tickers scored")
        else:
            warn(f"{signal_name}: no predictions generated")

    if not results:
        err("No ML signals produced — exiting")
        return

    # 5. Combine and save
    out = pd.DataFrame(results)
    out.index.name = "ticker"

    # Fill missing with neutral 50
    for col in HORIZONS.keys():
        if col not in out.columns:
            out[col] = 50.0
    out = out.fillna(50.0)

    out_path = ROOT / "ml_alpha_scores.csv"
    out.reset_index().to_csv(out_path, index=False)
    ok(f"ml_alpha_scores.csv → {len(out)} tickers  [short / medium / long]")

    # Quick IC check: correlate today's ml_short with 5d forward return
    try:
        fwd5 = compute_forward_returns(prices, 5)
        latest = all_dates[-1]
        fwd5_latest = fwd5[fwd5.index.get_level_values("date") == latest]
        fwd5_latest.index = fwd5_latest.index.get_level_values("ticker")
        merged = pd.DataFrame({"score": out["ml_short"], "fwd": fwd5_latest})
        merged = merged.dropna()
        if len(merged) > 10:
            ic = merged["score"].corr(merged["fwd"], method="spearman")
            log(f"  ml_short IC (Spearman vs 5d fwd rank): {ic:.3f}  (note: target is in the future — IC here uses today's fwd as proxy)")
    except Exception:
        pass

    print(f"\n{GREEN}✓ ML alpha complete{RESET}\n")


if __name__ == "__main__":
    main()
