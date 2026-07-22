#!/usr/bin/env python3
"""
Canyon v9 — Step 77: Regime-Conditional ML Model
=================================================
Tests whether training separate models per regime improves performance.

Step 75 trained ONE model on all data.  This step trains THREE models:
  • BULL model   : standard momentum features
  • BEAR model   : adds mean-reversion & low-vol proxies (neg_mom_1m, neg_rsi_dev)
  • SIDEWAYS model: mean-reversion focused

Walk-forward comparison: IC_all (step75) vs IC_regime (step77)

Data inputs:
  sp500_price_cache.csv  — from step75 (prices already downloaded)
  regime_history.csv     — from step76

Outputs:
  regime_ml_scores.csv   — today's LONG/HOLD/SHORT_AVOID per regime
  regime_backtest_ic.csv — per-period IC with regime + model type
  regime_ic_summary.csv  — BULL/BEAR/SIDEWAYS IC comparison table
  regime_current.json    — update with ML thresholds
  regime_ml_report.md    — narrative report

Usage:
  python3 canyon_final_v9_step77_regime_ml.py          # full run
  python3 canyon_final_v9_step77_regime_ml.py --fast   # 5-year quick test
  python3 canyon_final_v9_step77_regime_ml.py --score  # current scores only
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# LightGBM with Ridge fallback
try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

CACHE_PRICES   = ROOT / "sp500_price_cache.csv"
F_REGIME       = ROOT / "regime_history.csv"
IC_SUMMARY_IN  = ROOT / "ic_by_regime_full.csv"   # step75 summary for comparison

OUT_SCORES     = ROOT / "regime_ml_scores.csv"
OUT_IC_YEAR    = ROOT / "regime_backtest_ic.csv"
OUT_IC_SUM     = ROOT / "regime_ic_summary.csv"
OUT_CURRENT    = ROOT / "regime_current.json"
OUT_REPORT     = ROOT / "regime_ml_report.md"
OUT_SNAP       = ROOT / "score_history.csv"    # daily snapshot for future backtesting
SECTOR_CACHE   = ROOT / "sector_cache.json"

MIN_TRAIN      = 80
ALPHA          = 1.0
USE_LGBM       = True   # set False to force Ridge

# LightGBM params (tuned for financial IC, small sample sizes)
LGBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5,
    random_state=42, verbose=-1, n_jobs=-1,
)

# Base features (same as step66/75)
BASE_FEATS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "rsi_14", "inv_vol", "rank_mom", "rank_trend",
    # Macro features (added when available)
    "yc_spread", "hyd_proxy", "dxy_proxy",
]

# Extra features added for BEAR/SIDEWAYS models
BEAR_FEATS  = BASE_FEATS + ["neg_mom_1m", "neg_mom_3m", "neg_rsi_dev"]
SIDE_FEATS  = BASE_FEATS + ["neg_mom_1m", "neg_rsi_dev"]

MACRO_CACHE  = ROOT / "macro_features.csv"    # date-indexed macro data

# Step 78/79 augmented features (added to today's signal only — no lookahead in backtest)
AUG_FEATS_78 = ["quality_score"]       # fundamental quality (Step 78)
AUG_FEATS_79 = ["rank_sentiment"]      # FinBERT news sentiment (Step 79)
FUND_SCORES  = ROOT / "fundamental_deep_scores.csv"
SENT_SCORES  = ROOT / "finbert_sentiment.csv"
SUE_SCORES   = ROOT / "earnings_surprise_scores.csv"    # step81
REV_SCORES   = ROOT / "earnings_revision_scores.csv"    # step80
OPT_SCORES   = ROOT / "options_signals.csv"             # step82 — PCR + unusual calls
SQZ_SCORES   = ROOT / "short_interest_scores.csv"       # step83 — squeeze detector
INS_SCORES   = ROOT / "insider_signal_scores.csv"       # step85 — SEC Form 4 insider buying

# ── Sub-regime thresholds for LATE_BULL detection ────────────────────────────
LATE_BULL_RSI      = 70.0
LATE_BULL_RET20    = 0.04

# ── 8-Signal Regime-Adaptive Blend ───────────────────────────────────────────
# IC sources (estimated / measured):
#   ML momentum   IC≈0.10–0.20  | SUE          IC≈0.19
#   Revision      IC≈~0.08      | Quality      IC≈0.08
#   Sentiment     IC≈0.10       | Options      IC≈0.12 (live test)
#   Squeeze       IC≈TBD        | Insider      IC≈0.08 (CEO buying = signal)
#
# LATE_BULL: Options + Insider critical (PCR crowding & exec selling = exit warning)
# SIDEWAYS:  Squeeze + SUE dominate (catalyst × fuel = rotation winners)
# BEAR:      ML + Quality dominate; insider buying = contrarian signal
# BULL:      ML + SUE core; insider adds edge at stock-picking level
REGIME_BLEND: dict[str, dict[str, float]] = {
    "BULL":      {"ml": 0.35, "sue": 0.21, "revision": 0.14,
                  "quality": 0.06, "sentiment": 0.04,
                  "options": 0.07, "squeeze": 0.06, "insider": 0.07},
    "LATE_BULL": {"ml": 0.23, "sue": 0.23, "revision": 0.18,
                  "quality": 0.09, "sentiment": 0.04,
                  "options": 0.10, "squeeze": 0.05, "insider": 0.08},
    "BEAR":      {"ml": 0.46, "sue": 0.14, "revision": 0.07,
                  "quality": 0.10, "sentiment": 0.03,
                  "options": 0.07, "squeeze": 0.05, "insider": 0.08},
    "SIDEWAYS":  {"ml": 0.25, "sue": 0.27, "revision": 0.11,
                  "quality": 0.07, "sentiment": 0.03,
                  "options": 0.09, "squeeze": 0.09, "insider": 0.09},
    "UNKNOWN":   {"ml": 0.37, "sue": 0.20, "revision": 0.11,
                  "quality": 0.07, "sentiment": 0.04,
                  "options": 0.08, "squeeze": 0.06, "insider": 0.07},
}


def detect_effective_regime(base_regime: str) -> str:
    """
    Elevate BULL → LATE_BULL when market is overbought (RSI > 70 or 20d ret > 4%).
    LATE_BULL uses more defensive weights (less ML momentum, more SUE + quality).
    """
    if base_regime != "BULL":
        return base_regime
    if not OUT_CURRENT.exists():
        return base_regime
    try:
        state = json.loads(OUT_CURRENT.read_text())
        rsi    = float(state.get("rsi",    0))
        ret20  = float(state.get("ret_20d", 0))
        if rsi >= LATE_BULL_RSI or ret20 >= LATE_BULL_RET20:
            return "LATE_BULL"
    except Exception:
        pass
    return base_regime


# ─────────────────────────────────────────────────────────────
# 1.  LOAD PRICES & REBUILD FEATURES
# ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    if not CACHE_PRICES.exists():
        raise FileNotFoundError(
            "sp500_price_cache.csv not found. Run step75 first:\n"
            "  python3 canyon_final_v9_step75_universe_expansion.py"
        )
    prices = pd.read_csv(CACHE_PRICES, index_col=0, parse_dates=True)
    print(f"  Prices: {len(prices):,} days × {len(prices.columns)} tickers")
    return prices


def rsi_series(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build full historical feature panel with forward returns."""
    rets = prices.pct_change()

    mom_1m   = prices.pct_change(21)
    mom_3m   = prices.pct_change(63)
    mom_6m   = prices.pct_change(126)
    mom_12m  = prices.pct_change(252)
    skip_1m  = prices.pct_change(21).shift(22)
    mom_12m_skip1m = mom_12m - skip_1m

    ma200    = prices.rolling(200).mean()
    trend_200 = (prices - ma200) / ma200.replace(0, np.nan)

    inv_vol  = 1.0 / rets.rolling(21).std().replace(0, np.nan)
    fwd_ret  = prices.pct_change(21).shift(-21)

    rsi_df = prices.apply(lambda c: rsi_series(c))

    frames = []
    for ticker in prices.columns:
        df_t = pd.DataFrame({
            "date":             prices.index,
            "ticker":           ticker,
            "mom_1m":           mom_1m[ticker].values,
            "mom_3m":           mom_3m[ticker].values,
            "mom_6m":           mom_6m[ticker].values,
            "mom_12m_skip1m":   mom_12m_skip1m[ticker].values,
            "trend_200":        trend_200[ticker].values,
            "rsi_14":           rsi_df[ticker].values,
            "inv_vol":          inv_vol[ticker].values,
            "fwd_ret":          fwd_ret[ticker].values,
        })
        frames.append(df_t)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["mom_1m", "trend_200", "rsi_14", "inv_vol"])

    # Cross-sectional ranks (daily)
    for date, grp in panel.groupby("date"):
        idx = grp.index
        if len(idx) > 1:
            panel.loc[idx, "rank_mom"]   = grp["mom_12m_skip1m"].rank(pct=True)
            panel.loc[idx, "rank_trend"] = grp["trend_200"].rank(pct=True)

    # Bear/sideways extra features
    panel["neg_mom_1m"]  = -panel["mom_1m"]
    panel["neg_mom_3m"]  = -panel["mom_3m"]
    panel["neg_rsi_dev"] = -(panel["rsi_14"] - 50)

    panel = panel.dropna(subset=["rank_mom", "rank_trend"])
    return panel


def merge_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge macro date-level features into the ticker panel (no lookahead)."""
    if macro.empty:
        for c in ["yc_spread", "hyd_proxy", "dxy_proxy"]:
            panel[c] = np.nan
        return panel
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)
    macro_reset = macro.reset_index().rename(columns={"index": "date"})
    panel = panel.merge(macro_reset, on="date", how="left")
    # Forward-fill macro (macro data has trading-day gaps)
    for c in ["yc_spread", "hyd_proxy", "dxy_proxy"]:
        if c in panel.columns:
            panel[c] = panel[c].ffill()
    return panel


# ─────────────────────────────────────────────────────────────
# 2.  REGIME DATA
# ─────────────────────────────────────────────────────────────

def load_regime() -> pd.DataFrame:
    if not F_REGIME.exists():
        raise FileNotFoundError("regime_history.csv not found. Run step76 first.")
    df = pd.read_csv(F_REGIME, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.reset_index()[["date", "regime", "score"]]
    return df.sort_values("date")


# ─────────────────────────────────────────────────────────────
# 2b. MACRO FEATURES  (yield curve, credit spread, dollar)
# ─────────────────────────────────────────────────────────────

def load_macro_features(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Downloads macro time-series and builds 3 features (no lookahead):
      yc_spread  : 10Y - 3M Treasury yield spread (positive = normal, negative = inverted)
      hyd_proxy  : HYG ETF 21d momentum (falling = credit stress)
      dxy_proxy  : UUP ETF 21d momentum (rising = USD strength / risk-off)

    Returns DataFrame indexed by date with these 3 columns.
    Uses MACRO_CACHE to avoid re-downloading.
    """
    if MACRO_CACHE.exists():
        try:
            df = pd.read_csv(MACRO_CACHE, index_col=0, parse_dates=True)
            # refresh if stale (>1 day)
            age_days = (pd.Timestamp.now() - df.index.max()).days
            if age_days <= 1:
                print(f"  Macro cache: {len(df)} rows (fresh)")
                return df
        except Exception:
            pass

    print("  Downloading macro data (^TNX, ^IRX, HYG, UUP) …")
    import yfinance as yf
    macro_raw = {}

    for sym, label in [("^TNX", "tnx"), ("^IRX", "irx"), ("HYG", "hyg"), ("UUP", "uup")]:
        try:
            data = yf.download(sym, start=start, auto_adjust=True, progress=False)
            close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            macro_raw[label] = close.squeeze()
        except Exception as e:
            print(f"    Warning: could not fetch {sym}: {e}")

    if not macro_raw:
        print("  Macro features unavailable — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(macro_raw)
    df = df.ffill().dropna(how="all")

    # Feature engineering
    if "tnx" in df.columns and "irx" in df.columns:
        df["yc_spread"] = df["tnx"] - df["irx"]   # 10Y - 3M; negative = inverted
    else:
        df["yc_spread"] = np.nan

    if "hyg" in df.columns:
        df["hyd_proxy"] = df["hyg"].pct_change(21)  # HYG 21d momentum; falling = credit stress
    else:
        df["hyd_proxy"] = np.nan

    if "uup" in df.columns:
        df["dxy_proxy"] = df["uup"].pct_change(21)  # UUP 21d momentum; rising = dollar strength
    else:
        df["dxy_proxy"] = np.nan

    result = df[["yc_spread", "hyd_proxy", "dxy_proxy"]].dropna(how="all")
    result.index.name = "date"
    result.to_csv(MACRO_CACHE)
    print(f"  Macro features: {len(result)} rows  "
          f"(yc_spread={result['yc_spread'].iloc[-1]:.2f}, "
          f"hyd={result['hyd_proxy'].iloc[-1]:.3f}, "
          f"dxy={result['dxy_proxy'].iloc[-1]:.3f})")
    return result


# ─────────────────────────────────────────────────────────────
# 3.  WALK-FORWARD (regime-conditional vs all-data baseline)
# ─────────────────────────────────────────────────────────────

def make_model():
    """Return LightGBM regressor if available, else Ridge fallback."""
    if USE_LGBM and _LGBM_AVAILABLE:
        return LGBMRegressor(**LGBM_PARAMS), False   # (model, needs_scaler)
    return Ridge(alpha=ALPHA), True


def fit_predict(X_tr, y_tr, X_te, needs_scaler: bool, model):
    """Fit model and return predictions. Handles scaler for Ridge."""
    if needs_scaler:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def ic(y_true, y_pred):
    if len(y_true) < 4:
        return np.nan
    r, _ = spstats.spearmanr(y_pred, y_true)
    return float(r) if not np.isnan(r) else np.nan


def walk_forward(panel: pd.DataFrame, df_regime: pd.DataFrame,
                 min_years: int = 3) -> pd.DataFrame:
    """
    For each monthly rebalance:
      1. Identify current regime
      2. Fit model_all  (trained on ALL past data, base features)
      3. Fit model_cond (trained on past data of SAME regime + regime features)
      4. Record IC(all) vs IC(cond) for that period
    """
    # Merge regime into panel
    regime_map = dict(zip(df_regime["date"], df_regime["regime"]))
    panel = panel.copy()
    panel["regime"] = panel["date"].map(regime_map)
    panel["regime"] = panel["regime"].ffill().fillna("BULL")

    # Feature sets per regime
    def get_feats(regime, df):
        if regime == "BEAR":
            return [f for f in BEAR_FEATS if f in df.columns]
        elif regime == "SIDEWAYS":
            return [f for f in SIDE_FEATS if f in df.columns]
        else:
            return [f for f in BASE_FEATS if f in df.columns]

    # Monthly rebalance dates
    all_dates = sorted(panel["date"].unique())
    first = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=min_years)
    rebal_dates = [d for d in all_dates if pd.Timestamp(d) >= first][::21]

    records = []
    for i, t_date in enumerate(rebal_dates):
        t = pd.Timestamp(t_date)

        # current regime
        rr = df_regime[df_regime["date"] <= t]
        if rr.empty:
            continue
        cur_regime = rr.iloc[-1]["regime"]

        # base features (for all-data model)
        base_f = [f for f in BASE_FEATS if f in panel.columns]
        # regime features (for conditional model)
        reg_f  = get_feats(cur_regime, panel)

        # Training sets
        hist = panel[panel["date"] < t].dropna(subset=base_f + ["fwd_ret"])
        hist_r = hist[hist["regime"] == cur_regime]

        if len(hist) < MIN_TRAIN:
            continue

        # Test cross-section
        test = panel[panel["date"] == t_date].dropna(subset=base_f)
        if len(test) < 4:
            continue

        # --- Model A: all-data baseline ---
        m_a, ns_a = make_model()
        preds_a = fit_predict(hist[base_f].values, hist["fwd_ret"].values,
                              test[base_f].fillna(0).values, ns_a, m_a)
        ic_all = ic(test["fwd_ret"].values if "fwd_ret" in test.columns else np.zeros(len(test)),
                    preds_a)

        # --- Model B: regime-conditional ---
        train_r = hist_r if len(hist_r) >= MIN_TRAIN else hist
        reg_f_avail = [f for f in reg_f if f in train_r.columns and f in test.columns]
        train_r_clean = train_r.dropna(subset=reg_f_avail + ["fwd_ret"])
        if len(train_r_clean) < MIN_TRAIN:
            ic_cond = ic_all  # fallback
        else:
            m_b, ns_b = make_model()
            preds_b = fit_predict(train_r_clean[reg_f_avail].values,
                                  train_r_clean["fwd_ret"].values,
                                  test[reg_f_avail].fillna(0).values, ns_b, m_b)
            ic_cond = ic(test["fwd_ret"].values if "fwd_ret" in test.columns else np.zeros(len(test)),
                         preds_b)

        records.append({
            "date":    t_date,
            "regime":  cur_regime,
            "n_all":   len(hist),
            "n_cond":  len(train_r),
            "ic_all":  round(ic_all,  4) if ic_all  is not None else np.nan,
            "ic_cond": round(ic_cond, 4) if ic_cond is not None else np.nan,
        })

        if i % 30 == 0:
            print(f"  {t_date}  {cur_regime:10s}  "
                  f"IC_all={ic_all:+.3f}  IC_cond={ic_cond:+.3f}"
                  if ic_all is not None else f"  {t_date}  {cur_regime:10s}  IC=NaN")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 4.  IC SUMMARY COMPARISON
# ─────────────────────────────────────────────────────────────

def summarize(ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["BULL", "BEAR", "SIDEWAYS"]:
        sub = ic_df[ic_df["regime"] == regime].dropna(subset=["ic_all", "ic_cond"])
        if len(sub) < 2:
            continue

        for model, col in [("ALL_DATA", "ic_all"), ("REGIME_COND", "ic_cond")]:
            vals = sub[col].dropna()
            if len(vals) < 2:
                continue
            mean_ic = vals.mean()
            std_ic  = vals.std()
            t_stat  = mean_ic / (std_ic / np.sqrt(len(vals))) if std_ic > 0 else 0
            rows.append({
                "regime":      regime,
                "model":       model,
                "n_periods":   len(vals),
                "ic_mean":     round(mean_ic, 4),
                "ic_std":      round(std_ic, 4),
                "t_stat":      round(t_stat, 2),
                "pct_pos":     round((vals > 0).mean() * 100, 1),
                "meaningful":  "YES" if abs(t_stat) > 2 else "NO",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 5.  CURRENT SCORING
# ─────────────────────────────────────────────────────────────

def _load_aug_features() -> pd.DataFrame:
    """Load Step 78 quality + Step 79 sentiment scores for today's signals."""
    frames = []
    if FUND_SCORES.exists():
        try:
            fd = pd.read_csv(FUND_SCORES)[["ticker", "quality_score"]]
            fd["quality_score"] = pd.to_numeric(fd["quality_score"], errors="coerce")
            # rank-normalise to 0-100 so it's on same scale as other rank features
            fd["quality_score"] = fd["quality_score"].rank(pct=True) * 100
            frames.append(fd.set_index("ticker"))
            print(f"  Step 78 quality: {len(fd)} tickers loaded")
        except Exception as e:
            print(f"  Step 78 quality skipped: {e}")
    if SENT_SCORES.exists():
        try:
            sd = pd.read_csv(SENT_SCORES)[["ticker", "rank_sentiment"]]
            sd["rank_sentiment"] = pd.to_numeric(sd["rank_sentiment"], errors="coerce")
            frames.append(sd.set_index("ticker"))
            print(f"  Step 79 sentiment: {len(sd)} tickers loaded")
        except Exception as e:
            print(f"  Step 79 sentiment skipped: {e}")
    if SUE_SCORES.exists():
        try:
            sue = pd.read_csv(SUE_SCORES)[["ticker", "rank_sue"]]
            sue["rank_sue"] = pd.to_numeric(sue["rank_sue"], errors="coerce")
            frames.append(sue.set_index("ticker"))
            print(f"  Step 81 SUE: {len(sue)} tickers loaded")
        except Exception as e:
            print(f"  Step 81 SUE skipped: {e}")
    if REV_SCORES.exists():
        try:
            rev = pd.read_csv(REV_SCORES)[["ticker", "rank_revision"]]
            rev["rank_revision"] = pd.to_numeric(rev["rank_revision"], errors="coerce")
            frames.append(rev.set_index("ticker"))
            print(f"  Step 80 Revision: {len(rev)} tickers loaded")
        except Exception as e:
            print(f"  Step 80 Revision skipped: {e}")
    if OPT_SCORES.exists():
        try:
            opt_raw = pd.read_csv(OPT_SCORES)
            keep_cols = [c for c in ["ticker", "rank_options", "pcr_crowded", "pcr_vol", "uoa_bear_flag"] if c in opt_raw.columns]
            opt = opt_raw[keep_cols].copy()
            opt["rank_options"] = pd.to_numeric(opt["rank_options"], errors="coerce")
            if "pcr_crowded" not in opt.columns:
                if "pcr_vol" in opt.columns:
                    pcr = pd.to_numeric(opt["pcr_vol"], errors="coerce")
                    opt["pcr_crowded"] = (pcr < 0.35).astype(float).fillna(0)
                elif "uoa_bear_flag" in opt.columns:
                    opt["pcr_crowded"] = pd.to_numeric(opt["uoa_bear_flag"], errors="coerce").fillna(0)
                else:
                    opt["pcr_crowded"] = 0.0
            else:
                opt["pcr_crowded"] = pd.to_numeric(opt["pcr_crowded"], errors="coerce").fillna(0)
            opt = opt[["ticker", "rank_options", "pcr_crowded"]]
            frames.append(opt.set_index("ticker"))
            n_crowd = int(opt["pcr_crowded"].sum())
            print(f"  Step 82 Options: {len(opt)} tickers loaded  ({n_crowd} crowded-call alerts)")
        except Exception as e:
            print(f"  Step 82 Options skipped: {e}")
    if SQZ_SCORES.exists():
        try:
            sqz = pd.read_csv(SQZ_SCORES)[["ticker", "rank_squeeze"]]
            sqz["rank_squeeze"] = pd.to_numeric(sqz["rank_squeeze"], errors="coerce")
            frames.append(sqz.set_index("ticker"))
            print(f"  Step 83 Squeeze: {len(sqz)} tickers loaded")
        except Exception as e:
            print(f"  Step 83 Squeeze skipped: {e}")
    if INS_SCORES.exists():
        try:
            ins = pd.read_csv(INS_SCORES)[["ticker", "rank_insider"]]
            ins["rank_insider"] = pd.to_numeric(ins["rank_insider"], errors="coerce")
            frames.append(ins.set_index("ticker"))
            n_buy = int((ins["rank_insider"] >= 80).sum())
            print(f"  Step 85 Insider: {len(ins)} tickers loaded  ({n_buy} INSIDER_BUY alerts)")
        except Exception as e:
            print(f"  Step 85 Insider skipped: {e}")
    if frames:
        import functools
        aug = functools.reduce(lambda a, b: a.join(b, how="outer"), frames)
        return aug.reset_index()
    return pd.DataFrame(columns=["ticker"])


def save_score_snapshot(scores: pd.DataFrame, regime: str) -> None:
    """
    Append today's full blended scores to score_history.csv.
    This builds a real historical record that can be used for future backtesting
    — replacing the current price-momentum-only Step 62 proxy.

    Columns saved: date, ticker, predicted_score, signal, regime,
                   + all individual signal ranks (rank_sue, rank_revision, etc.)
    """
    if scores.empty:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check if today is already saved (avoid duplicates on multiple --score runs)
    if OUT_SNAP.exists():
        tail = pd.read_csv(OUT_SNAP, usecols=["date"]).tail(1)
        if not tail.empty and str(tail.iloc[0]["date"])[:10] == today_str:
            return   # already saved today

    snap = scores.copy()
    snap["date"]   = today_str
    snap["regime"] = regime

    # Keep only useful columns
    keep = ["date", "ticker", "predicted_score", "signal", "regime"]
    for col in ["rank_sue", "rank_revision", "quality_score", "rank_sentiment",
                "rank_options", "rank_squeeze", "crowding_level"]:
        if col in snap.columns:
            keep.append(col)

    snap = snap[[c for c in keep if c in snap.columns]]

    write_header = not OUT_SNAP.exists()
    snap.to_csv(OUT_SNAP, mode="a", header=write_header, index=False)

    days_saved = len(pd.read_csv(OUT_SNAP, usecols=["date"])["date"].unique())
    print(f"  Snapshot → score_history.csv  ({len(snap)} tickers, "
          f"day {days_saved} of history)")


def score_today(panel: pd.DataFrame, df_regime: pd.DataFrame) -> pd.DataFrame:
    base_regime = df_regime.iloc[-1]["regime"]
    cur_regime  = detect_effective_regime(base_regime)
    if cur_regime != base_regime:
        print(f"  Sub-regime detected: {base_regime} → {cur_regime} "
              f"(overbought — reducing momentum weight)")
    latest = panel["date"].max()
    test   = panel[panel["date"] == latest].copy()

    # Feature set driven by underlying base regime (not LATE_BULL — same features as BULL)
    ml_regime = "BULL" if cur_regime == "LATE_BULL" else cur_regime
    if ml_regime == "BEAR":
        feat_list = [f for f in BEAR_FEATS if f in panel.columns]
    elif ml_regime == "SIDEWAYS":
        feat_list = [f for f in SIDE_FEATS if f in panel.columns]
    else:
        feat_list = [f for f in BASE_FEATS if f in panel.columns]

    test = test[["ticker"] + feat_list].dropna()
    print(f"  Scoring {len(test)} tickers ({latest.date()} — {cur_regime} regime) …")

    # ── Augment with Step 78 quality + Step 79 sentiment (current snapshot only) ──
    aug = _load_aug_features()
    aug_cols = [c for c in aug.columns if c != "ticker"]
    if not aug.empty and aug_cols:
        test = test.merge(aug[["ticker"] + aug_cols], on="ticker", how="left")
        for c in aug_cols:
            test[c] = test[c].fillna(test[c].median())  # fill missing with median
        feat_list = feat_list + [c for c in aug_cols if c in test.columns]
        print(f"  Augmented features: {aug_cols}")

    # Aug cols exist only in today's snapshot — train on base features only
    # (no lookahead: historical panel has no quality/sentiment history)
    base_feat_list = [f for f in feat_list if f not in aug_cols] if aug_cols else feat_list
    train_all    = panel[panel["date"] < latest].dropna(subset=base_feat_list + ["fwd_ret"])

    # Make sure regime column exists on train_all
    # Use ml_regime (base regime) for training — LATE_BULL trains on BULL history
    if "regime" in train_all.columns:
        train_regime = train_all[train_all["regime"] == ml_regime]
    else:
        train_regime = train_all

    train = train_regime if len(train_regime) >= MIN_TRAIN else train_all
    if len(train) < MIN_TRAIN:
        print("  Insufficient history — using all available data")
        train = train_all

    model, needs_scaler = make_model()
    model_name = "LightGBM" if (USE_LGBM and _LGBM_AVAILABLE) else "Ridge"
    print(f"  Model: {model_name}")
    # Train and predict on base features (aug cols blended in post-processing below)
    scores = fit_predict(
        train[base_feat_list].fillna(0).values, train["fwd_ret"].values,
        test[base_feat_list].fillna(0).values, needs_scaler, model,
    )

    test = test.copy()
    test["predicted_score"] = scores

    # ── Regime-adaptive blend (post-processing, no lookahead) ────────────────
    # Weights calibrated to per-regime walk-forward IC:
    #   BEAR:     ML×60% (IC=0.1994, strongest) + quality as drawdown shield
    #   SIDEWAYS: SUE×38% (captures rotation/catalyst events, IC=0.1848)
    #   BULL:     Revision×18% (upgrades chase price in trending markets)
    if aug_cols:
        wts = REGIME_BLEND.get(cur_regime, REGIME_BLEND["UNKNOWN"])
        ml_score = test["predicted_score"]
        ml_rank  = ml_score.rank(pct=True)

        # Start with ML weight (always present)
        combined   = ml_rank * wts["ml"]
        blend_desc = [f"ML×{int(wts['ml']*100)}%"]
        avail_wt   = wts["ml"]

        signal_map = [
            ("rank_sue",       "sue",      "SUE"),
            ("rank_revision",  "revision", "Rev"),
            ("quality_score",  "quality",  "Qual"),
            ("rank_sentiment", "sentiment","Sent"),
            ("rank_options",   "options",  "Opt"),    # step82: PCR + smart money
            ("rank_squeeze",   "squeeze",  "Sqz"),    # step83: squeeze detector
            ("rank_insider",   "insider",  "Ins"),    # step85: SEC Form 4 insider buying
        ]
        for col, key, label in signal_map:
            if col in test.columns and key in wts:
                w = wts[key]
                ranked = test[col].rank(pct=True)
                combined  += ranked * w
                avail_wt  += w
                blend_desc.append(f"{label}×{int(w*100)}%")

        # Renormalise if some signals are missing (avail_wt < 1.0)
        if avail_wt > 0:
            combined = combined / avail_wt
        test["predicted_score"] = combined
        print(f"  Score blending [{cur_regime}]: {' + '.join(blend_desc)}")

    test["regime"] = cur_regime

    # ── Soros Crowding Detector ───────────────────────────────────────────────
    # When ≥3 signals are simultaneously in top 10%, the thesis is KNOWN TO THE
    # MARKET — consensus at maximum is a contrarian warning, not a buy signal.
    # Apply a score penalty; flag with crowding_level for operator review.
    # rank_options is HIGH when PCR is high (fear = opportunity) so it's ANTI-crowding.
    # But pcr_crowded=1 means TOO MANY CALLS (everyone bullish) → crowding risk.
    # We use pcr_crowded as a direct crowding indicator, not rank_options.
    CROWD_COLS = ["rank_sue", "rank_revision", "rank_sentiment",
                  "rank_mom", "rank_trend"]
    crowd_cols_present = [c for c in CROWD_COLS if c in test.columns]
    if crowd_cols_present:
        def _crowding(row):
            return sum(row.get(c, 0) > 90 for c in crowd_cols_present)
        test["crowding_score"] = test.apply(
            lambda row: sum((row[c] > 90) for c in crowd_cols_present
                            if not pd.isna(row[c])), axis=1
        )
        # Penalty: each crowded signal above 2 reduces score by 12%
        # (Soros: "when everyone is cheering, look for the exit")
        crowd_penalty = test["crowding_score"].clip(lower=2) - 2  # 0,1,2,3
        test["crowding_penalty"] = crowd_penalty * 0.12
        test["predicted_score"]  = test["predicted_score"] * (1 - test["crowding_penalty"])

        # Label crowding level
        # If options data available, pcr_crowded (PCR < 0.35) counts as an extra crowd signal
        if "pcr_crowded" in test.columns:
            test["crowding_score"] = test["crowding_score"] + test["pcr_crowded"].fillna(0)

        test["crowding_level"] = test["crowding_score"].map(
            lambda n: "VERY_CROWDED" if n >= 4 else
                      ("CROWDED"      if n >= 3 else
                       ("WATCH"       if n == 2 else "CLEAR"))
        )
        crowded_n = (test["crowding_level"].isin(["CROWDED","VERY_CROWDED"])).sum()
        watch_n   = (test["crowding_level"] == "WATCH").sum()
        if crowded_n or watch_n:
            print(f"  Crowding alert: {crowded_n} CROWDED/VERY_CROWDED  "
                  f"{watch_n} WATCH (Soros: consensus = exit risk)")
    else:
        test["crowding_score"] = 0
        test["crowding_level"] = "CLEAR"

    test = test.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    # ── Concentration filter (Livermore principle) ────────────────────────────
    # Don't hold 160 positions — that's an index fund.
    # Only output high-conviction positions: score > threshold AND top N.
    SCORE_MIN  = 0.82          # minimum blended score for LONG
    MAX_LONG   = 25            # Livermore: "watch the basket" — concentrate
    MAX_SHORT  = 15            # avoid list stays smaller

    test["signal"] = "HOLD"
    long_candidates = test[test["predicted_score"] >= SCORE_MIN]
    long_idx = long_candidates.index[:MAX_LONG]
    test.loc[long_idx, "signal"] = "LONG"

    # SHORT_AVOID: bottom N by score (no score minimum needed — these are warnings)
    short_idx = test.index[-MAX_SHORT:]
    test.loc[short_idx, "signal"] = "SHORT_AVOID"

    # ── Sector diversification: max 3 per GICS sector in LONG ──────────────
    smap = get_sector_map(test["ticker"].tolist())
    test["sector"] = test["ticker"].map(smap).fillna("Other")
    long_mask   = test["signal"] == "LONG"
    sector_cnt: dict[str, int] = {}
    MAX_PER_SECTOR = 3
    UNCAPPED_SECTORS = {"Other", "Broad"}   # heterogeneous catch-all — no cap needed
    for idx in test[long_mask].index:
        sec = test.at[idx, "sector"]
        if sec in UNCAPPED_SECTORS:
            continue
        cnt = sector_cnt.get(sec, 0)
        if cnt >= MAX_PER_SECTOR:
            test.at[idx, "signal"] = "HOLD"
        else:
            sector_cnt[sec] = cnt + 1

    kept_long = (test["signal"] == "LONG").sum()
    print(f"  High-conviction LONG signals: {kept_long} "
          f"(score ≥ {SCORE_MIN}, max {MAX_LONG}, sector-capped)")

    return test


# Sector map for diversification (GICS simplified — extend as needed)
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","GOOG":"Tech","META":"Tech",
    "NVDA":"Semiconductors","AMD":"Semiconductors","INTC":"Semiconductors",
    "MU":"Semiconductors","QCOM":"Semiconductors","TXN":"Semiconductors",
    "AVGO":"Semiconductors","ADI":"Semiconductors","AMAT":"Semiconductors",
    "LRCX":"Semiconductors","KLAC":"Semiconductors","MRVL":"Semiconductors",
    "ADBE":"Tech","CRM":"Tech","ORCL":"Tech","SAP":"Tech","INTU":"Tech",
    "CDNS":"Tech","SNPS":"Tech","ADSK":"Tech","CTSH":"Tech","EPAM":"Tech",
    "ANET":"Tech","CSCO":"Tech","FFIV":"Tech","GLW":"Tech","CIEN":"Tech",
    "DELL":"Tech","HPE":"Tech","CDW":"Tech",
    # Cloud / SaaS / Internet
    "AMZN":"Consumer Disc","NFLX":"Consumer Disc","BKNG":"Consumer Disc",
    "EXPE":"Consumer Disc","ABNB":"Consumer Disc","DASH":"Consumer Disc",
    "APP":"Tech","DDOG":"Tech","CRWD":"Tech","PANW":"Tech","ZS":"Tech",
    "FTNT":"Tech","OKTA":"Tech","SNOW":"Tech","COIN":"Financials",
    # Financials
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","C":"Financials",
    "GS":"Financials","MS":"Financials","BLK":"Financials","BX":"Financials",
    "APO":"Financials","ARES":"Financials","AMP":"Financials","AXP":"Financials",
    "COF":"Financials","SCHW":"Financials","CB":"Financials","ALL":"Financials",
    "AIG":"Financials","AFL":"Financials","ACGL":"Financials","AIZ":"Financials",
    "CINF":"Financials","CBOE":"Financials","CME":"Financials","ICE":"Financials",
    "AJG":"Financials","AON":"Financials","BRO":"Financials","MMC":"Financials",
    "CFG":"Financials","FITB":"Financials","BAC":"Financials","BNY":"Financials",
    # Healthcare
    "UNH":"Healthcare","JNJ":"Healthcare","LLY":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","BMY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "BIIB":"Healthcare","VRTX":"Healthcare","REGN":"Healthcare","CI":"Healthcare",
    "CNC":"Healthcare","ELV":"Healthcare","HUM":"Healthcare","CVS":"Healthcare",
    "ABT":"Healthcare","MDT":"Healthcare","BSX":"Healthcare","SYK":"Healthcare",
    "EW":"Healthcare","BDX":"Healthcare","DHR":"Healthcare","BAX":"Healthcare",
    "DXCM":"Healthcare","ALGN":"Healthcare","DVA":"Healthcare","CRL":"Healthcare",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","APA":"Energy","DVN":"Energy",
    "EOG":"Energy","FANG":"Energy","MRO":"Energy","EQT":"Energy","EXE":"Energy",
    "BKR":"Energy","HAL":"Energy","SLB":"Energy","CF":"Materials","ADM":"Materials",
    # Industrials
    "CAT":"Industrials","DE":"Industrials","HON":"Industrials","GE":"Industrials",
    "RTX":"Industrials","BA":"Industrials","LMT":"Industrials","NOC":"Industrials",
    "EMR":"Industrials","ETN":"Industrials","AME":"Industrials","DOV":"Industrials",
    "FIX":"Industrials","EME":"Industrials","CARR":"Industrials","AXON":"Industrials",
    "CHRW":"Industrials","EXPD":"Industrials","CBRE":"Industrials","FDX":"Industrials",
    # Consumer
    "COST":"Consumer Staples","WMT":"Consumer Staples","PG":"Consumer Staples",
    "KO":"Consumer Staples","PEP":"Consumer Staples","MO":"Consumer Staples",
    "PM":"Consumer Staples","CL":"Consumer Staples","MDLZ":"Consumer Staples",
    "CAG":"Consumer Staples","CPB":"Consumer Staples","GIS":"Consumer Staples",
    "TSLA":"Consumer Disc","GM":"Consumer Disc","F":"Consumer Disc",
    "HD":"Consumer Disc","LOW":"Consumer Disc","TJX":"Consumer Disc",
    "ROST":"Consumer Disc","BBWI":"Consumer Disc","BBY":"Consumer Disc",
    "CMG":"Consumer Disc","MCD":"Consumer Disc","YUM":"Consumer Disc",
    "SBUX":"Consumer Disc","DRI":"Consumer Disc","DPZ":"Consumer Disc",
    "AZO":"Consumer Disc","ORLY":"Consumer Disc","CASY":"Consumer Disc",
    "CVNA":"Consumer Disc","DECK":"Consumer Disc",
    # Utilities / REITs
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "EXC":"Utilities","AEP":"Utilities","XEL":"Utilities","ED":"Utilities",
    "EIX":"Utilities","ETR":"Utilities","DTE":"Utilities","CMS":"Utilities",
    "AEE":"Utilities","LNT":"Utilities","EVRG":"Utilities","ES":"Utilities",
    "AWK":"Utilities","CNP":"Utilities","ATO":"Utilities",
    "AMT":"REITs","PLD":"REITs","EQIX":"REITs","CCI":"REITs","DLR":"REITs",
    "AVB":"REITs","EQR":"REITs","EXR":"REITs","ARE":"REITs","BXP":"REITs",
    "CPT":"REITs","FRT":"REITs","ESS":"REITs",
    # Materials
    "LIN":"Materials","APD":"Materials","ECL":"Materials","ALB":"Materials",
    "DD":"Materials","DOW":"Materials","BG":"Materials","ADM":"Materials",
    "BALL":"Materials","AVY":"Materials","AMCR":"Materials","IP":"Materials",
    # Broad / ETF
    "SPY":"Broad","QQQ":"Broad","SMH":"Broad","XLK":"Broad",
}

# GICS sector name simplification map
_GICS_SIMPLIFY = {
    "Information Technology": "Tech",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Disc",
    "Consumer Staples": "Consumer Staples",
    "Financials": "Financials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "REITs",
    "Communication Services": "Communication",
}


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Return a ticker→sector mapping for the given universe.
    Priority: hardcoded SECTOR_MAP > sector_cache.json (7-day TTL from Wikipedia).
    Missing tickers fall back to 'Other'.
    """
    import time
    result: dict[str, str] = dict(SECTOR_MAP)

    # Identify tickers not covered by the hardcoded map
    missing = [t for t in tickers if t not in result]
    if not missing:
        return result

    # Try cache
    cache_ok = False
    cache_data: dict[str, str] = {}
    if SECTOR_CACHE.exists():
        try:
            cache_data = json.loads(SECTOR_CACHE.read_text())
            age = time.time() - SECTOR_CACHE.stat().st_mtime
            if age < 7 * 86400:  # 1-week TTL
                cache_ok = True
        except Exception:
            cache_data = {}

    if cache_ok:
        for t in missing:
            if t in cache_data:
                result[t] = cache_data[t]
        return result

    # Cache stale / absent — try yfinance for missing tickers (silent fallback)
    import yfinance as yf
    new_data: dict[str, str] = {}
    for t in missing[:50]:   # cap at 50 to avoid slow startup
        try:
            info = yf.Ticker(t).fast_info
            gics = getattr(info, "sector", None) or ""
            new_data[t] = _GICS_SIMPLIFY.get(gics, "Other") if gics else "Other"
        except Exception:
            new_data[t] = "Other"
    if new_data:
        cache_data.update(new_data)
        try:
            SECTOR_CACHE.write_text(json.dumps(cache_data, indent=2))
        except Exception:
            pass
        for t in missing:
            if t in cache_data:
                result[t] = cache_data[t]

    return result


# ─────────────────────────────────────────────────────────────
# 6.  REPORT
# ─────────────────────────────────────────────────────────────

def write_report(ic_df: pd.DataFrame, summary: pd.DataFrame,
                 scores: pd.DataFrame, step75_summary: pd.DataFrame) -> None:

    cur_regime = "N/A"
    if OUT_CURRENT.exists():
        try:
            cur_regime = json.loads(OUT_CURRENT.read_text()).get("regime", "N/A")
        except Exception:
            pass

    lines = [
        "# Canyon v9 — Step 77: Regime-Conditional ML Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Thesis",
        "A single model trained on all market environments may have near-zero or",
        "negative IC in bear markets because momentum features flip sign (high",
        "recent momentum = first to crash).  Training separate models per regime",
        "allows BEAR model to learn mean-reversion and quality features.",
        "",
        f"## Current Regime: **{cur_regime}**",
        "",
        "## Comparison: All-Data Model vs Regime-Conditional Model",
        "| Regime | Model | IC | t-stat | Pos% | Meaningful? |",
        "|--------|-------|----|--------|------|-------------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['regime']} | {row['model']} | "
            f"{row['ic_mean']:+.4f} | {row['t_stat']:+.2f} | "
            f"{row['pct_pos']:.1f}% | "
            f"{'✅ YES' if row['meaningful']=='YES' else '❌ NO'} |"
        )

    # Step 75 comparison
    if not step75_summary.empty and "mean_ic" in step75_summary.columns:
        lines += [
            "",
            "## Step 75 Baseline (reference — single model, 97 tickers, 25 years)",
            "| Regime | IC | t-stat | Assessment |",
            "|--------|-----|--------|-----------|",
        ]
        for _, row in step75_summary.iterrows():
            lines.append(
                f"| {row['regime']} | {row['mean_ic']:+.4f} | {row['t_stat']:+.2f} | {row['assessment']} |"
            )

    lines += [
        "",
        "## What the Numbers Mean",
        "- t-stat > 2 → statistically significant positive edge (p < 0.05)",
        "- IC > 0 in BEAR with t > 2 → model truly has positive expected value in downturns",
        "- REGIME_COND IC > ALL_DATA IC in BEAR → regime-conditioning is additive",
        "",
        "## Today's Top LONG Signals (High-Conviction, max 25)",
        f"**Regime: {cur_regime}** — score ≥ 0.82, sector-capped, crowding-penalised",
        "| # | Ticker | Score | Sector | Signal |",
        "|---|--------|-------|--------|--------|",
    ]
    if not scores.empty:
        long_df = scores[scores["signal"] == "LONG"].head(25)
        for i, (_, row) in enumerate(long_df.iterrows(), 1):
            crowd_flag = ""
            crowd_lv = row.get("crowding_level", "CLEAR")
            if crowd_lv == "VERY_CROWDED":
                crowd_flag = " 🔴 VERY_CROWDED"
            elif crowd_lv == "CROWDED":
                crowd_flag = " 🟠 CROWDED"
            elif crowd_lv == "WATCH":
                crowd_flag = " 🟡 WATCH"
            lines.append(
                f"| {i} | {row['ticker']} | {row.get('predicted_score',0):.4f} "
                f"| {row.get('sector','?')} | LONG{crowd_flag} |"
            )

    # Crowding summary
    if not scores.empty and "crowding_level" in scores.columns:
        very_c = scores[scores["crowding_level"] == "VERY_CROWDED"]["ticker"].tolist()
        crowded = scores[scores["crowding_level"] == "CROWDED"]["ticker"].tolist()
        watch   = scores[scores["crowding_level"] == "WATCH"]["ticker"].tolist()
        if very_c or crowded:
            lines += [
                "",
                "## ⚠️ Crowding Alerts (Soros: consensus = exit risk)",
                "> When ≥3 of 5 signals are simultaneously top 10%, the trade is KNOWN.",
                "> Score has been penalised. Consider reducing or avoiding these positions.",
                "",
            ]
            if very_c:
                lines.append(f"- 🔴 **VERY_CROWDED** (≥4 signals top 10%): {', '.join(very_c)}")
            if crowded:
                lines.append(f"- 🟠 **CROWDED** (3 signals top 10%): {', '.join(crowded)}")
            if watch:
                lines.append(f"- 🟡 **WATCH** (2 signals top 10%): {', '.join(watch)}")

    # Regime blend weight table
    lines += [
        "",
        "## 7-Signal Regime-Adaptive Blend Weights",
        "| Regime | ML | SUE | Rev | Qual | Sent | Opt | Sqz | Rationale |",
        "|--------|----|-----|-----|------|------|-----|-----|-----------|",
    ]
    rationale = {
        "BULL":      "Opt=smart-money confirmation; Sqz=squeeze fuel",
        "LATE_BULL": "Opt=PCR crowding guard (Soros); ML reduced (overbought)",
        "BEAR":      "ML strongest (IC=0.1994); Opt=fear=contrarian opportunity",
        "SIDEWAYS":  "SUE+Sqz=rotation catalyst×fuel; highest combined alpha",
        "UNKNOWN":   "Balanced fallback across all 7 signals",
    }
    for reg, wts in REGIME_BLEND.items():
        cur_marker = " ◀" if reg == cur_regime else ""
        lines.append(
            f"| **{reg}{cur_marker}** "
            f"| {int(wts['ml']*100)}% "
            f"| {int(wts.get('sue',0)*100)}% "
            f"| {int(wts.get('revision',0)*100)}% "
            f"| {int(wts.get('quality',0)*100)}% "
            f"| {int(wts.get('sentiment',0)*100)}% "
            f"| {int(wts.get('options',0)*100)}% "
            f"| {int(wts.get('squeeze',0)*100)}% "
            f"| {rationale[reg]} |"
        )

    lines += [
        "",
        "## Feature Sets by Regime",
        "- **BULL**: momentum (mom_1m → mom_12m), trend, RSI, inv_vol, ranks",
        "- **BEAR**: all base + neg_mom_1m, neg_mom_3m, neg_rsi_dev (mean-reversion proxies)",
        "- **SIDEWAYS**: all base + neg_mom_1m, neg_rsi_dev (short-term mean reversion)",
        "",
        "## Next Steps",
        "If BEAR IC is positive but not significant (t < 2):",
        "  → Normal — we only have ~35 bear-regime months in 25 years",
        "  → Solution: add fundamental quality features (Step 78) for deeper BEAR signal",
        "  → The regime-conditional model is still better than one that has NEGATIVE BEAR IC",
    ]

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────

def run_full(fast: bool = False) -> None:
    print("\n╔════════════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 77: Regime-Conditional ML Model  ║")
    print("╚════════════════════════════════════════════════════╝\n")

    print("[1/5] Loading price data …")
    prices = load_prices()

    if fast:
        cutoff = prices.index.max() - pd.DateOffset(years=5)
        prices = prices[prices.index >= cutoff]
        print(f"  Fast mode: {prices.index.min().date()} → {prices.index.max().date()}")

    print("[2/5] Building feature panel …")
    panel = build_panel(prices)
    print(f"  Panel: {len(panel):,} rows  "
          f"({panel['date'].nunique()} dates, {panel['ticker'].nunique()} tickers)")

    print("[2b/5] Loading macro features (yield curve, credit, dollar) …")
    macro = load_macro_features(start=str(prices.index.min().date()))
    panel = merge_macro(panel, macro)
    macro_ok = [c for c in ["yc_spread","hyd_proxy","dxy_proxy"] if panel[c].notna().any()]
    print(f"  Macro features available: {macro_ok if macro_ok else 'none (will use price-only)'}")
    model_tag = "LightGBM" if (USE_LGBM and _LGBM_AVAILABLE) else "Ridge"
    print(f"  ML model: {model_tag}")

    print("[3/5] Loading regime history …")
    df_regime = load_regime()
    # Merge regime into panel
    regime_map = dict(zip(df_regime["date"], df_regime["regime"]))
    panel["regime"] = panel["date"].map(regime_map)
    panel["regime"] = panel["regime"].ffill().fillna("BULL")

    print("[4/5] Walk-forward IC comparison …")
    ic_df = walk_forward(panel, df_regime, min_years=2 if fast else 3)
    if not ic_df.empty:
        ic_df.to_csv(OUT_IC_YEAR, index=False)
        print(f"  regime_backtest_ic.csv  ({len(ic_df)} periods)")

        summary = summarize(ic_df)
        summary.to_csv(OUT_IC_SUM, index=False)

        print("\n  >>> KEY COMPARISON — Regime-Conditional vs All-Data <<<")
        if not summary.empty:
            for regime in ["BULL", "BEAR", "SIDEWAYS"]:
                sub = summary[summary["regime"] == regime]
                if sub.empty:
                    continue
                print(f"\n  {regime}:")
                for _, row in sub.iterrows():
                    flag = "✅" if row["meaningful"] == "YES" else "⚠ "
                    print(f"    {row['model']:12s}: IC={row['ic_mean']:+.4f}  "
                          f"t={row['t_stat']:+.2f}  {flag}")
    else:
        summary = pd.DataFrame()
        print("  No IC data — check price cache and regime files")

    # Load step75 summary for comparison
    step75_sum = pd.DataFrame()
    if IC_SUMMARY_IN.exists():
        step75_sum = pd.read_csv(IC_SUMMARY_IN)

    print("\n[5/5] Scoring today's universe …")
    scores = score_today(panel, df_regime)
    if not scores.empty:
        scores.to_csv(OUT_SCORES, index=False)
        print(f"  regime_ml_scores.csv  ({len(scores)} tickers)")
        cur_regime = df_regime.iloc[-1]["regime"]
        save_score_snapshot(scores, cur_regime)
        print(f"\n  Top LONG signals ({cur_regime} regime):")
        for _, row in scores[scores["signal"] == "LONG"].head(5).iterrows():
            sc = row.get("predicted_score", 0)
            print(f"    {row['ticker']:6s}  score={sc:+.4f}")

    # Update regime_current.json
    if OUT_CURRENT.exists():
        try:
            current = json.loads(OUT_CURRENT.read_text())
            thresholds = {"BULL": 0.55, "BEAR": 0.65, "SIDEWAYS": 0.60}
            current["ml_threshold"]  = thresholds.get(current.get("regime", "BULL"), 0.55)
            current["regime_ml_run"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not summary.empty:
                bear_row = summary[(summary["regime"] == "BEAR") &
                                   (summary["model"] == "REGIME_COND")]
                if not bear_row.empty:
                    current["bear_ic_regime_cond"] = round(bear_row.iloc[0]["ic_mean"], 4)
                    current["bear_t_stat"]         = round(bear_row.iloc[0]["t_stat"], 2)
            OUT_CURRENT.write_text(json.dumps(current, indent=2))
            print(f"  regime_current.json updated")
        except Exception as e:
            print(f"  Warning: could not update regime_current.json: {e}")

    write_report(ic_df if not ic_df.empty else pd.DataFrame(),
                 summary, scores, step75_sum)

    # ── VERDICT ──
    print("\n" + "═" * 60)
    if not summary.empty:
        bear_cond = summary[(summary["regime"] == "BEAR") &
                            (summary["model"] == "REGIME_COND")]
        bear_all  = summary[(summary["regime"] == "BEAR") &
                            (summary["model"] == "ALL_DATA")]

        ic_cond = bear_cond.iloc[0]["ic_mean"] if not bear_cond.empty else 0
        t_cond  = bear_cond.iloc[0]["t_stat"]  if not bear_cond.empty else 0
        ic_all  = bear_all.iloc[0]["ic_mean"]  if not bear_all.empty else 0

        if ic_cond > 0 and t_cond > 2:
            print(f"  ✅ BEAR IC (regime model) = {ic_cond:+.4f}  t={t_cond:+.2f}")
            print("     Model has STATISTICALLY SIGNIFICANT positive edge in bear markets.")
        elif ic_cond > 0:
            print(f"  ⚠  BEAR IC (regime model) = {ic_cond:+.4f}  t={t_cond:+.2f}")
            print("     Positive but not yet statistically significant (need t > 2).")
            if ic_cond > ic_all:
                print(f"     Regime-conditioning improved BEAR IC: {ic_all:+.4f} → {ic_cond:+.4f}")
        else:
            print(f"  ❌ BEAR IC (regime model) = {ic_cond:+.4f}  t={t_cond:+.2f}")
            print("     Add Step 78 (fundamental features) to strengthen the BEAR signal.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Step 77")
    parser.add_argument("--fast",  action="store_true", help="5-year only")
    parser.add_argument("--score", action="store_true", help="Score today only")
    args = parser.parse_args()

    if args.score:
        print("[Score-only mode]")
        prices    = load_prices()
        panel     = build_panel(prices)
        df_regime = load_regime()
        regime_map = dict(zip(df_regime["date"], df_regime["regime"]))
        panel["regime"] = panel["date"].map(regime_map)
        panel["regime"] = panel["regime"].ffill().fillna("BULL")
        scores = score_today(panel, df_regime)
        if not scores.empty:
            scores.to_csv(OUT_SCORES, index=False)
            cur_regime = df_regime.iloc[-1]["regime"]
            save_score_snapshot(scores, cur_regime)
            print(scores.head(15)[["ticker", "predicted_score", "signal"]].to_string(index=False))
    else:
        run_full(fast=args.fast)
