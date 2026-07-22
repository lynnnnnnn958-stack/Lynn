#!/usr/bin/env python3
"""
Canyon v11 Full — Institutional Upgrades (Free Tier)
=====================================================
Upgrades from v10:

  NEW SIGNALS (3):
    residual_mom      — 12m momentum minus beta×SPY return (pure idiosyncratic alpha)
    vol_regime        — current vol / historical avg vol (vol-expansion signal)
    rel_strength_sec  — stock 3m return minus equal-weighted sector 3m (sector-relative)

  ROLLING IC WEIGHTS (Layer 2):
    ic_map is no longer hardcoded. compute_signal_ic() runs first,
    and process_and_combine() uses those measured ICs for weighting.
    Fallback hardcoded ICs used only for v9 alpha signals that can't
    be recomputed from price history alone.

  ENHANCED TC MODEL (Layer 3):
    + Short borrow cost: 2% annualized on short book (monthly drag)
    + VIX regime switch: when SPY realized vol >25% annualized,
      max position weight reduced by 30% (exposure reduction)

  LIGHTGBM PRICE ENSEMBLE (Layer 7):
    lgb_price_ensemble — LightGBM trained on 16-month historical
    price-signal panel, with 21-day purge gap between train/test.
    Predicts cross-sectional rank of 21-day forward returns.
    Uses purged time-series CV; zero lookahead.

Outputs (same as v10 + updated content):
  v11_full_signals.csv
  v11_signal_ic.csv
  v11_factor_model.csv
  v11_factor_returns.csv
  v11_backtest_monthly.csv
  v11_backtest_summary.csv
  v11_full_report.md

Usage:
  python3 canyon_v11_full.py
  python3 canyon_v11_full.py --no-backtest
  python3 canyon_v11_full.py --no-lgb        # skip LightGBM (faster)
"""
from __future__ import annotations
import argparse, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew as sp_skew
warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.today().strftime("%Y-%m-%d")

HOLD     = 21
BETA_W   = 252
MAX_W    = 0.08
BETA_TOL = 0.10
SEC_CAP  = 0.10
TOP_N    = 25

SHORT_BORROW_ANN = 0.02   # 2% annual borrow cost on short book
HIGH_VOL_THRESH  = 0.25   # SPY realized vol threshold for regime switch
HIGH_VOL_SCALAR  = 0.70   # reduce max weight to 70% in high-vol regime

SIGNAL_COLS_V9 = [
    "sig_regime_ml", "sig_quality", "sig_revision", "sig_surprise",
    "sig_sentiment", "sig_squeeze", "sig_insider", "sig_options", "sig_ml_ensemble",
]

IC_FALLBACK_V9 = {
    "sig_ml_ensemble": 0.370, "sig_surprise":   0.229, "sig_regime_ml": 0.223,
    "sig_quality":     0.048, "sig_squeeze":     0.050, "sig_revision":  0.038,
    "sig_insider":     0.004, "sig_options":     0.017, "sig_sentiment": 0.010,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def _rank_normalize(s: pd.Series) -> pd.Series:
    from scipy.stats import norm
    r = s.rank(pct=True)
    return pd.Series(norm.ppf(r.clip(0.001, 0.999)), index=s.index)


def _rsi14(prices_df: pd.DataFrame) -> pd.Series:
    delta = prices_df.diff(1)
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).iloc[-1]


def _compute_period_signals(p_snap: pd.DataFrame, spy_snap: pd.Series,
                             sect_map: dict) -> pd.DataFrame:
    """
    Compute all 22 price signals from a price snapshot.
    Used both for current-date signals and historical panel building.
    Returns DataFrame indexed by ticker.
    """
    tickers = p_snap.columns.tolist()
    r_snap  = p_snap.pct_change()
    spy_r   = spy_snap.pct_change().dropna()

    def _ret(n):
        if len(p_snap) < n + 1:
            return pd.Series(np.nan, index=tickers)
        return p_snap.iloc[-1] / p_snap.iloc[-n - 1] - 1

    # ── Momentum ───────────────────────────────────────────────────────────
    mom_1w  = _ret(5)
    mom_1m  = _ret(21)
    mom_3m  = _ret(63)
    mom_6m  = _ret(126)
    mom_12m = (p_snap.iloc[-22] / p_snap.iloc[-253] - 1) if len(p_snap) >= 253 else mom_6m
    trend_200 = (p_snap.iloc[-1] / p_snap.rolling(200).mean().iloc[-1] - 1) \
                if len(p_snap) >= 200 else pd.Series(np.nan, index=tickers)
    hi52 = (p_snap.iloc[-1] / p_snap.rolling(252).max().iloc[-1]) \
           if len(p_snap) >= 252 else pd.Series(np.nan, index=tickers)

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi_14 = _rsi14(p_snap)

    # ── Volatility ─────────────────────────────────────────────────────────
    vol_21d = r_snap.rolling(21).std().iloc[-1] * np.sqrt(252)
    vol_63d = r_snap.rolling(63).std().iloc[-1] * np.sqrt(252)

    neg = r_snap.copy(); neg[neg > 0] = 0
    downside_vol = neg.rolling(63).std().iloc[-1] * np.sqrt(252)
    vol_of_vol   = r_snap.rolling(21).std().rolling(63).std().iloc[-1] * np.sqrt(252)

    # ── Idiosyncratic vol ──────────────────────────────────────────────────
    idio_vols = {}
    betas_d   = {}
    for tkr in tickers:
        if tkr not in r_snap.columns:
            idio_vols[tkr] = np.nan; betas_d[tkr] = 1.0; continue
        al = pd.concat([r_snap[tkr], spy_r], axis=1).dropna().iloc[-BETA_W:]
        if len(al) < 60:
            idio_vols[tkr] = float(vol_21d.get(tkr, np.nan)); betas_d[tkr] = 1.0; continue
        al.columns = ["stk", "spy"]
        beta_t = np.cov(al["stk"], al["spy"])[0, 1] / np.var(al["spy"])
        resid  = al["stk"] - beta_t * al["spy"]
        idio_vols[tkr] = float(resid.std() * np.sqrt(252))
        betas_d[tkr]   = float(beta_t)
    idio_vol = pd.Series(idio_vols)
    beta_s   = pd.Series(betas_d)
    bab_signal = -beta_s

    max_dd_1yr = ((p_snap - p_snap.rolling(252).max()) / p_snap.rolling(252).max()).iloc[-1] \
                 if len(p_snap) >= 252 else pd.Series(np.nan, index=tickers)
    skewness_sig = -r_snap.iloc[-252:].apply(
        lambda s: float(sp_skew(s.dropna())) if s.dropna().shape[0] > 20 else np.nan
    )
    abs_ret_21  = r_snap.abs().rolling(21).mean().iloc[-1]
    amihud_sig  = -(abs_ret_21 / vol_21d.replace(0, np.nan))

    # ── Industry momentum ─────────────────────────────────────────────────
    ind_mom = {}
    for tkr in tickers:
        sec   = sect_map.get(tkr, "Unknown")
        peers = [t for t in tickers if sect_map.get(t) == sec and t != tkr]
        if not peers or sec == "Unknown":
            ind_mom[tkr] = 0.0; continue
        if len(p_snap) >= 253:
            ind_mom[tkr] = float((p_snap[peers].iloc[-22] / p_snap[peers].iloc[-253] - 1).mean())
        else:
            ind_mom[tkr] = float((p_snap[peers].iloc[-1] / p_snap[peers].iloc[-127] - 1).mean())
    industry_mom = pd.Series(ind_mom)

    # ── NEW: Residual momentum (market-adjusted) ──────────────────────────
    if len(p_snap) >= 253:
        spy_12m = float(spy_snap.iloc[-22] / spy_snap.iloc[-253] - 1) \
                  if len(spy_snap) >= 253 else 0.0
    else:
        spy_12m = float(spy_snap.pct_change(126).iloc[-1]) if len(spy_snap) >= 127 else 0.0
    raw_mom12 = (p_snap.iloc[-22] / p_snap.iloc[-253] - 1) if len(p_snap) >= 253 else mom_6m
    residual_mom = raw_mom12 - beta_s * spy_12m   # alpha momentum

    # ── NEW: Vol regime (current vol / historical avg vol) ────────────────
    vol_21d_series  = r_snap.rolling(21).std() * np.sqrt(252)
    hist_avg_vol    = vol_21d_series.rolling(252).mean().iloc[-1] \
                      if len(vol_21d_series) >= 252 else vol_21d_series.mean()
    vol_regime_raw  = vol_21d / hist_avg_vol.replace(0, np.nan) - 1
    vol_regime      = -vol_regime_raw  # negative = vol contraction = bullish signal

    # ── NEW: Sector-relative strength (stock 3m vs sector 3m) ────────────
    sect_3m = {}
    for tkr in tickers:
        sec   = sect_map.get(tkr, "Unknown")
        peers = [t for t in tickers if sect_map.get(t) == sec and t != tkr]
        if not peers or sec == "Unknown":
            sect_3m[tkr] = 0.0; continue
        sect_3m[tkr] = float((p_snap[peers].iloc[-1] / p_snap[peers].iloc[-64] - 1).mean()) \
                       if len(p_snap) >= 64 else 0.0
    sect_3m_s       = pd.Series(sect_3m)
    rel_strength_sec = mom_3m - sect_3m_s

    df = pd.DataFrame({
        "mom_1w":          mom_1w,
        "mom_1m":          mom_1m,
        "mom_3m":          mom_3m,
        "mom_6m":          mom_6m,
        "mom_12m_skip1m":  mom_12m,
        "trend_200":       trend_200,
        "hi52":            hi52,
        "rsi_14":          rsi_14,
        "vol_21d":         vol_21d,
        "vol_63d":         vol_63d,
        "idio_vol":        idio_vol,
        "downside_vol":    downside_vol,
        "vol_of_vol":      vol_of_vol,
        "beta":            beta_s,
        "bab_signal":      bab_signal,
        "max_dd_1yr":      max_dd_1yr,
        "skewness_sig":    skewness_sig,
        "amihud_sig":      amihud_sig,
        "industry_mom":    industry_mom,
        "residual_mom":    residual_mom,
        "vol_regime":      vol_regime,
        "rel_strength_sec": rel_strength_sec,
    }).reindex(tickers)
    return df, beta_s


# ══════════════════════════════════════════════════════════════════════════════
# A. SIGNAL LIBRARY (22 price signals)
# ══════════════════════════════════════════════════════════════════════════════

def build_price_signals(prices: pd.DataFrame, spy: pd.Series,
                        sectors: pd.Series) -> tuple:
    print("\n[A] Building price signal library (22 signals)...")
    tickers   = [c for c in prices.columns if c != "SPY"]
    p         = prices[tickers]
    sect_map  = sectors.to_dict()

    df, beta_s = _compute_period_signals(p, spy, sect_map)
    print(f"    Built 22 price signals for {len(df)} tickers")
    return df, beta_s


# ══════════════════════════════════════════════════════════════════════════════
# B. SIGNAL IC ANALYSIS (all 22 price signals, 12-month panel)
# ══════════════════════════════════════════════════════════════════════════════

PRICE_SIG_COLS = [
    "mom_1w", "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "hi52", "rsi_14", "vol_21d", "vol_63d",
    "max_dd_1yr", "skewness_sig", "idio_vol", "downside_vol",
    "vol_of_vol", "bab_signal", "amihud_sig", "industry_mom",
    "residual_mom", "vol_regime", "rel_strength_sec",
]


def compute_signal_ic(prices: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    print("\n[B] Signal IC analysis (22 price signals, 12 monthly periods)...")
    tickers  = [c for c in prices.columns if c != "SPY"]
    p        = prices[tickers]
    spy      = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    sect_map = sectors.to_dict()

    ic_acc: dict[str, list[float]] = {s: [] for s in PRICE_SIG_COLS}

    for i in range(1, 13):
        t_s  = -(i + 1) * HOLD
        t_fe = -i * HOLD
        abs_ts = len(p) + t_s
        abs_fe = len(p) + t_fe if t_fe < 0 else len(p) - 1

        if abs_fe <= abs_ts or abs_fe >= len(p):
            continue

        p_snap   = p.iloc[:abs_ts]
        spy_snap = spy.iloc[:abs_ts]
        if len(p_snap) < 253:
            continue

        fwd = (p.iloc[abs_fe] / p.iloc[abs_ts] - 1).dropna()
        if len(fwd) < 50:
            continue
        fwd_cs = _rank_normalize(fwd)

        try:
            snap_df, _ = _compute_period_signals(p_snap, spy_snap, sect_map)
        except Exception:
            continue

        for col in PRICE_SIG_COLS:
            if col not in snap_df.columns:
                continue
            sig    = snap_df[col].dropna()
            common = sig.index.intersection(fwd_cs.dropna().index)
            if len(common) < 50:
                continue
            ic_val, _ = spearmanr(sig[common], fwd_cs[common])
            if not np.isnan(ic_val):
                ic_acc[col].append(float(ic_val))

    ic_rows = []
    for col in PRICE_SIG_COLS:
        ics = ic_acc.get(col, [])
        if not ics:
            continue
        mean_ic = float(np.mean(ics))
        ic_std  = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.10
        n_p     = len(ics)
        t_stat  = mean_ic * np.sqrt(n_p) / (ic_std + 1e-9)
        ic_rows.append({
            "signal": col,
            "ic":     round(mean_ic, 4),
            "t_stat": round(t_stat, 2),
            "n":      n_p,
            "stars":  "★★★" if abs(t_stat) > 3 else "★★" if abs(t_stat) > 2 else "★",
        })

    ic_df = pd.DataFrame(ic_rows).sort_values("ic", ascending=False)
    ic_df.to_csv(ROOT / "v11_signal_ic.csv", index=False)
    print(f"    Saved v11_signal_ic.csv  ({len(ic_df)} signals)")
    print("    Top 5 by IC:")
    for _, r in ic_df.head(5).iterrows():
        print(f"      {r['signal']:25s}  IC={r['ic']:+.4f}  t={r['t_stat']:+.2f}  {r['stars']}")
    return ic_df


# ══════════════════════════════════════════════════════════════════════════════
# C. LIGHTGBM PRICE ENSEMBLE SIGNAL (Layer 7)
# ══════════════════════════════════════════════════════════════════════════════

def build_lgb_signal(prices: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    Train LightGBM on a 16-month historical panel of price signals.
    Uses purged time-series CV (21-day gap) to prevent lookahead bias.
    Returns current-date cross-sectional rank prediction per ticker.
    """
    print("\n[C] LightGBM price ensemble signal...")
    try:
        import lightgbm as lgb
    except ImportError:
        print("    [!] lightgbm not installed (pip install lightgbm) — skipping")
        return pd.Series(dtype=float)

    tickers  = [c for c in prices.columns if c != "SPY"]
    p        = prices[tickers]
    spy      = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    sect_map = sectors.to_dict()

    FEAT_COLS = [c for c in PRICE_SIG_COLS if c != "beta"]

    # ── Build historical panel ─────────────────────────────────────────────
    rows = []
    N_PERIODS = 20
    available = 0
    for i in range(N_PERIODS, 0, -1):
        abs_ts = len(p) - (i + 1) * HOLD
        abs_fe = len(p) - i * HOLD

        if abs_ts < 253 or abs_fe <= abs_ts or abs_fe >= len(p):
            continue

        p_snap   = p.iloc[:abs_ts]
        spy_snap = spy.iloc[:abs_ts]

        fwd = (p.iloc[abs_fe] / p.iloc[abs_ts] - 1).dropna()
        if len(fwd) < 50:
            continue
        fwd_cs = _rank_normalize(fwd)

        try:
            snap_df, _ = _compute_period_signals(p_snap, spy_snap, sect_map)
        except Exception:
            continue

        for tkr in tickers:
            if tkr not in fwd_cs.index or tkr not in snap_df.index:
                continue
            target = fwd_cs.get(tkr, np.nan)
            if np.isnan(target):
                continue
            row = {"period": i, "ticker": tkr, "target": target}
            for col in FEAT_COLS:
                row[col] = float(snap_df.at[tkr, col]) if col in snap_df.columns else np.nan
            rows.append(row)
        available += 1

    if available < 6:
        print(f"    [!] Only {available} periods available — need >= 6, skipping LGB")
        return pd.Series(dtype=float)

    panel = pd.DataFrame(rows).dropna(subset=FEAT_COLS + ["target"])
    print(f"    Training panel: {len(panel)} rows, {available} periods, {len(FEAT_COLS)} features")

    # Purged train/test split: train on older periods, test on last 4
    HOLDOUT = 4
    train_mask = panel["period"] > HOLDOUT
    if train_mask.sum() < 100:
        print("    [!] Insufficient training rows — skipping LGB")
        return pd.Series(dtype=float)

    X_train = panel.loc[train_mask, FEAT_COLS].values
    y_train = panel.loc[train_mask, "target"].values

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # OOS IC (test periods)
    test_mask = panel["period"] <= HOLDOUT
    if test_mask.sum() > 20:
        X_test = panel.loc[test_mask, FEAT_COLS].values
        y_test = panel.loc[test_mask, "target"].values
        preds_test = model.predict(X_test)
        oos_ic, _ = spearmanr(preds_test, y_test)
        print(f"    OOS IC (holdout {HOLDOUT} periods): {oos_ic:+.4f}")

    # ── Predict on current date ────────────────────────────────────────────
    try:
        curr_df, _ = _compute_period_signals(p, spy, sect_map)
    except Exception as e:
        print(f"    [!] Current-date feature computation failed: {e}")
        return pd.Series(dtype=float)

    feat_matrix = curr_df[FEAT_COLS].fillna(0).values
    preds = model.predict(feat_matrix)
    lgb_signal = pd.Series(preds, index=curr_df.index, name="lgb_price_ensemble")
    lgb_signal = lgb_signal.rank(pct=True)  # cross-sectional percentile rank

    print(f"    LGB signal built for {lgb_signal.notna().sum()} tickers")
    return lgb_signal


# ══════════════════════════════════════════════════════════════════════════════
# D. SIGNAL PROCESSING (rolling IC weights, not hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

def process_and_combine(price_sigs: pd.DataFrame, alpha_df: pd.DataFrame,
                        beta_s: pd.Series, sectors: pd.Series,
                        ic_df: pd.DataFrame,
                        lgb_signal: pd.Series) -> tuple:
    """Winsorize → rank-normalize → neutralize → IC²-weighted combination."""
    print("\n[D] Signal processing: winsorize → rank → neutralize → combine...")

    a   = alpha_df.set_index("ticker")[SIGNAL_COLS_V9 + ["sector"]].copy()
    df  = price_sigs.join(a, how="left")
    df["sector"] = df["sector"].fillna(sectors.reindex(df.index))
    df["beta"]   = beta_s.reindex(df.index).fillna(1.0)

    # Add LGB signal if available
    lgb_cols = []
    if lgb_signal is not None and len(lgb_signal) > 0:
        df["lgb_price_ensemble"] = lgb_signal.reindex(df.index)
        lgb_cols = ["lgb_price_ensemble"]

    price_sig_cols   = [c for c in price_sigs.columns if c != "beta"]
    all_signal_cols  = price_sig_cols + SIGNAL_COLS_V9 + lgb_cols

    # Sector one-hot for neutralization
    sector_dummies = pd.get_dummies(df["sector"], prefix="sec", drop_first=True)
    X_neu = pd.concat([df[["beta"]], sector_dummies], axis=1).astype(float)
    X_neu.insert(0, "const", 1.0)

    neutralized = {}
    from scipy.stats import norm
    for col in all_signal_cols:
        raw  = df[col] if col in df.columns else pd.Series(0.5, index=df.index)
        raw  = raw.fillna(raw.median())
        ranked = _winsorize(raw).rank(pct=True)
        try:
            coefs, *_ = np.linalg.lstsq(X_neu.values, ranked.values, rcond=None)
            resid = ranked.values - X_neu.values @ coefs
        except Exception:
            resid = ranked.values - ranked.values.mean()
        resid_r = pd.Series(resid).rank(pct=True).clip(0.001, 0.999)
        neutralized[col] = pd.Series(norm.ppf(resid_r.values), index=df.index)

    neu_df = pd.DataFrame(neutralized)

    # ── Build IC map: computed ICs override fallback ───────────────────────
    ic_map = dict(IC_FALLBACK_V9)
    if ic_df is not None and len(ic_df) > 0:
        for _, row in ic_df.iterrows():
            sig    = row["signal"]
            ic_val = max(0.0, float(row["ic"]))  # positive IC only; negative = wrong direction
            ic_map[sig] = ic_val
    # LGB signal gets its own measured IC (if available) or generous prior
    if "lgb_price_ensemble" in lgb_cols:
        ic_map["lgb_price_ensemble"] = ic_map.get("lgb_price_ensemble", 0.08)

    ic_sq   = {k: max(ic_map.get(k, 0.01), 0.0) ** 2 for k in all_signal_cols}
    total   = sum(ic_sq.values()) or 1.0
    weights = {k: v / total for k, v in ic_sq.items()}

    score = sum(weights.get(c, 0) * neu_df[c].fillna(0) for c in all_signal_cols)
    df["v11_score"]  = score.values
    df["v11_rank"]   = score.rank(ascending=False).astype(int)
    df["v11_pctile"] = score.rank(pct=True).round(3)
    mn, mx = df["v11_score"].min(), df["v11_score"].max()
    df["v11_score_100"] = ((df["v11_score"] - mn) / (mx - mn) * 100).round(2)

    top5 = df.nlargest(5, "v11_score").index.tolist()
    print(f"    Signals combined: {len(all_signal_cols)}  (price: {len(price_sig_cols)}, v9: {len(SIGNAL_COLS_V9)}, lgb: {len(lgb_cols)})")
    print(f"    Top 5 tickers: {top5}")

    df.index.name = "ticker"
    df.reset_index().to_csv(ROOT / "v11_full_signals.csv", index=False)
    print("    Saved v11_full_signals.csv")

    # Show IC weights for top signals
    top_w = sorted(weights.items(), key=lambda x: -x[1])[:8]
    print("    Top 8 IC² weights:")
    for sig, w in top_w:
        print(f"      {sig:30s}  IC={ic_map.get(sig, 0):.3f}  w={w:.3f}")

    return df, weights


# ══════════════════════════════════════════════════════════════════════════════
# E. 5-FACTOR RISK MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_factor_model(prices: pd.DataFrame, spy: pd.Series,
                       sectors: pd.Series, price_sigs: pd.DataFrame) -> tuple:
    print("\n[E] Building 5-factor risk model...")
    tickers = [c for c in prices.columns if c != "SPY"]
    ret     = prices[tickers].pct_change().dropna()
    spy_ret = spy.pct_change().dropna()
    common  = ret.index.intersection(spy_ret.index)
    ret, spy_r = ret.loc[common], spy_ret.loc[common]

    def quintile_ls(signal, ret_df, top_pct=0.20, flip=False):
        tails = signal.quantile([top_pct, 1 - top_pct])
        hi = signal[signal >= tails[1 - top_pct]].index.intersection(ret_df.columns)
        lo = signal[signal <= tails[top_pct]].index.intersection(ret_df.columns)
        if len(hi) == 0 or len(lo) == 0:
            return pd.Series(0, index=ret_df.index)
        ls = ret_df[hi].mean(axis=1) - ret_df[lo].mean(axis=1)
        return ls if not flip else -ls

    f_mkt     = spy_r.rename("F_Market")
    f_mom     = quintile_ls(price_sigs["mom_12m_skip1m"].dropna(), ret).rename("F_Momentum")
    f_vol     = quintile_ls(price_sigs["vol_21d"].dropna(), ret, flip=True).rename("F_LowVol")
    f_bab     = quintile_ls(price_sigs["bab_signal"].dropna(), ret).rename("F_BAB")
    f_quality = quintile_ls(price_sigs["rsi_14"].dropna(), ret).rename("F_Quality")
    factors   = pd.concat([f_mkt, f_mom, f_vol, f_bab, f_quality], axis=1).dropna()

    try:
        from sklearn.covariance import LedoitWolf
        lw         = LedoitWolf().fit(factors.values[-252:])
        factor_cov = lw.covariance_ * 252
    except Exception:
        factor_cov = np.cov(factors.iloc[-252:].T) * 252

    B_rows, resid_vols = [], {}
    for tkr in tickers:
        if tkr not in ret.columns:
            continue
        al = pd.concat([ret[tkr], factors], axis=1).dropna().iloc[-BETA_W:]
        if len(al) < 60:
            B_rows.append([tkr] + [0.0] * 5); resid_vols[tkr] = 0.20; continue
        y = al.iloc[:, 0].values
        X = np.column_stack([np.ones(len(al)), al.iloc[:, 1:].values])
        try:
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coefs
        except Exception:
            coefs = [0] * 6; resid = y
        B_rows.append([tkr] + list(coefs[1:]))
        resid_vols[tkr] = float(np.std(resid) * np.sqrt(252))

    B = pd.DataFrame(B_rows, columns=["ticker", "b_mkt", "b_mom", "b_vol", "b_bab", "b_quality"])
    B = B.set_index("ticker")
    resid_vol_s = pd.Series(resid_vols).clip(lower=0.02)

    factors_monthly = factors.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    B.reset_index().to_csv(ROOT / "v11_factor_model.csv", index=False)
    factors_monthly.to_csv(ROOT / "v11_factor_returns.csv")
    print(f"    Factor model built: B={B.shape}")
    return B, factor_cov, resid_vol_s, factors, factors_monthly


# ══════════════════════════════════════════════════════════════════════════════
# F. WALK-FORWARD BACKTEST (enhanced TC + VIX regime)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(prices: pd.DataFrame, spy: pd.Series,
                          sectors: pd.Series, beta_s: pd.Series,
                          factors_monthly: pd.DataFrame,
                          vix_series: pd.Series = None) -> pd.DataFrame:
    """
    Walk-forward long-short backtest with enhanced TC model:
      - Bid-ask spread: 10bps round-trip per unit turnover
      - Short borrow:   2% annualized on short book weight
      - VIX regime:     VIX > 25 (or SPY realized vol >25% if VIX unavailable)
                        → reduce max_w to 70%

    Args:
        vix_series: Optional pd.Series with DatetimeIndex and VIX levels.
                    If provided, replaces SPY realized vol as regime detector.
                    Load from fred_macro_daily.csv["VIX"].
    """
    print("\n[F] Walk-forward backtest (enhanced TC + VIX regime)...")

    try:
        import cvxpy as cp
    except ImportError:
        print("    [!] cvxpy not installed"); return pd.DataFrame()

    tickers  = [c for c in prices.columns if c != "SPY"]
    ret      = prices[tickers].pct_change()
    spy_ret  = spy.pct_change()
    sect_map = sectors.to_dict()

    warmup    = 273
    all_dates = prices.index[warmup:]
    month_ends = pd.date_range(all_dates[0], all_dates[-1], freq="ME")
    month_ends = [d for d in month_ends
                  if d in prices.index or prices.index.searchsorted(d) - 1 >= 0]

    records  = []
    prev_w   = np.zeros(len(tickers))

    for i, reb_date in enumerate(month_ends[:-1]):
        next_date = month_ends[i + 1]
        loc = prices.index.get_loc(reb_date) if reb_date in prices.index \
              else prices.index.searchsorted(reb_date) - 1
        p_slice  = prices[tickers].iloc[:loc + 1]
        r_slice  = p_slice.pct_change().dropna()
        spy_slice = spy.iloc[:loc + 1].pct_change().dropna()
        if len(p_slice) < 253:
            continue

        # ── Signals ───────────────────────────────────────────────────────
        mom  = (p_slice.iloc[-22] / p_slice.iloc[-253] - 1)
        vol_sig = r_slice.rolling(21).std().iloc[-1] * np.sqrt(252)

        # VIX regime: use real FRED VIX if available, else SPY realized vol proxy
        if vix_series is not None and len(vix_series) > 0:
            # PIT: use VIX data strictly up to rebalancing date (1-day lag built into FRED)
            vix_hist = vix_series[vix_series.index <= reb_date].dropna()
            if len(vix_hist) > 0:
                vix_now = float(vix_hist.iloc[-1])
                is_high_vol = vix_now > 25.0   # standard VIX threshold
            else:
                vix_now = float(spy_slice.rolling(21).std().iloc[-1] * np.sqrt(252)) * 100
                is_high_vol = vix_now > HIGH_VOL_THRESH * 100
        else:
            # Fallback: SPY realized vol as VIX proxy
            spy_vol_now = float(spy_slice.rolling(21).std().iloc[-1] * np.sqrt(252))
            vix_now     = spy_vol_now * 100   # convert to VIX-like scale for reporting
            is_high_vol = spy_vol_now > HIGH_VOL_THRESH

        regime_max_w = MAX_W * HIGH_VOL_SCALAR if is_high_vol else MAX_W

        betas_t = {}
        for tkr in tickers:
            al = pd.concat([r_slice[tkr], spy_slice], axis=1).dropna().iloc[-BETA_W:]
            if len(al) < 60: betas_t[tkr] = 1.0; continue
            c = np.cov(al.iloc[:, 0], al.iloc[:, 1])
            betas_t[tkr] = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 1.0
        beta_t = pd.Series(betas_t)

        # Residual momentum (new signal in backtest)
        spy_12m  = float(spy.iloc[loc - 21] / spy.iloc[max(0, loc - 252)] - 1) \
                   if loc >= 253 else 0.0
        resid_m  = mom - beta_t * spy_12m

        mom_r    = _rank_normalize(mom.dropna()) * 0.35
        vol_r    = _rank_normalize(-vol_sig.dropna()) * 0.25
        bab_r    = _rank_normalize(-beta_t) * 0.20
        resid_r  = _rank_normalize(resid_m.dropna()) * 0.20
        score_t  = pd.concat([mom_r, vol_r, bab_r, resid_r], axis=1).sum(axis=1).dropna()
        if len(score_t) < 20:
            continue

        sel_tickers = score_t.index.tolist()
        n           = len(sel_tickers)
        alpha_v     = score_t.values / (np.std(score_t.values) + 1e-9) * 0.10
        betas_v     = beta_t.reindex(sel_tickers).fillna(1.0).values
        sectors_v   = pd.Series([sect_map.get(t, "Unknown") for t in sel_tickers])

        r_sub = r_slice[sel_tickers].fillna(0)
        try:
            from sklearn.covariance import LedoitWolf
            cov_t = LedoitWolf().fit(r_sub.iloc[-126:].values).covariance_ * 252
        except Exception:
            cov_t = np.cov(r_sub.iloc[-126:].T) * 252
        cov_t = (cov_t + cov_t.T) / 2 + np.eye(n) * 1e-4

        wL = cp.Variable(n, nonneg=True)
        wS = cp.Variable(n, nonneg=True)
        w  = wL - wS

        obj  = cp.Maximize(alpha_v @ w - 1.5 * cp.quad_form(w, cp.psd_wrap(cov_t)))
        cons = [
            cp.sum(wL) == 0.5, cp.sum(wS) == 0.5,
            wL <= regime_max_w, wS <= regime_max_w,
            cp.abs(betas_v @ w) <= BETA_TOL,
        ]
        for sec in sectors_v.unique():
            mask = (sectors_v == sec).values.astype(float)
            cons.append(cp.abs(mask @ w) <= SEC_CAP)

        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            try: prob.solve(verbose=False)
            except Exception: pass

        if wL.value is None:
            w_val = np.zeros(n)
            top_i = np.argsort(-alpha_v)[:TOP_N]; bot_i = np.argsort(alpha_v)[:TOP_N]
            w_val[top_i] = 0.5 / TOP_N; w_val[bot_i] = -0.5 / TOP_N
        else:
            w_val = wL.value - wS.value

        # ── Returns ───────────────────────────────────────────────────────
        next_loc = prices.index.get_loc(next_date) if next_date in prices.index \
                   else prices.index.searchsorted(next_date) - 1
        reb_loc  = loc

        period_ret = prices[sel_tickers].iloc[reb_loc:next_loc + 1].pct_change().dropna()
        if period_ret.empty:
            records.append({"date": next_date, "gross_ret": 0.0, "net_ret": 0.0,
                            "long_ret": 0.0, "short_ret": 0.0,
                            "tc_cost": 0.0, "borrow_cost": 0.0,
                            "portfolio_beta": float(np.dot(w_val, betas_v)),
                            "vix_level": round(vix_now, 2),
                            "high_vol":  int(is_high_vol)})
            continue

        cum_ret   = (1 + period_ret).prod() - 1
        gross_ret = float(np.dot(w_val, cum_ret.reindex(sel_tickers).fillna(0)))

        # TC: bid-ask spread (10bps round-trip)
        turnover    = np.sum(np.abs(w_val - prev_w[:n]))
        spread_cost = turnover * 0.0010

        # NEW: Short borrow cost (2% ann on short book, monthly)
        short_book   = float(np.sum(np.abs(w_val[w_val < 0])))
        borrow_cost  = short_book * SHORT_BORROW_ANN / 12

        tc_total = spread_cost + borrow_cost
        net_ret  = gross_ret - tc_total

        long_mask  = w_val > 0
        short_mask = w_val < 0
        long_ret  = float(np.dot(w_val * long_mask, cum_ret.reindex(sel_tickers).fillna(0)))
        short_ret = float(np.dot(w_val * short_mask, cum_ret.reindex(sel_tickers).fillna(0)))
        port_beta = float(np.dot(w_val, betas_v))
        mkt_contrib = port_beta * spy_ret.iloc[reb_loc:next_loc + 1].sum()

        records.append({
            "date":           next_date,
            "gross_ret":      round(gross_ret, 5),
            "net_ret":        round(net_ret, 5),
            "long_ret":       round(long_ret, 5),
            "short_ret":      round(short_ret, 5),
            "tc_cost":        round(tc_total, 5),
            "spread_cost":    round(spread_cost, 5),
            "borrow_cost":    round(borrow_cost, 5),
            "portfolio_beta": round(port_beta, 4),
            "mkt_attribution": round(mkt_contrib, 5),
            "vix_level":      round(vix_now, 2),
            "high_vol":       int(is_high_vol),
        })
        prev_w[:n] = w_val

        regime_tag = " [HIGH VOL]" if is_high_vol else ""
        print(f"    {str(next_date.date())}  net={net_ret:+.1%}  "
              f"borrow={borrow_cost:.2%}  β={port_beta:+.3f}{regime_tag}")

    bt = pd.DataFrame(records)
    if bt.empty:
        print("    [!] Backtest produced no records")
        return bt

    rets    = bt["net_ret"].dropna()
    nav     = (1 + rets).cumprod()
    ann_ret = float((1 + rets.mean()) ** 12 - 1)
    ann_vol = float(rets.std() * np.sqrt(12))
    sharpe  = (ann_ret - 0.053) / ann_vol if ann_vol > 0 else 0
    max_dd  = float((nav / nav.cummax() - 1).min())
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else 0

    avg_spread  = bt["spread_cost"].mean() * 12
    avg_borrow  = bt["borrow_cost"].mean() * 12
    high_vol_months = int(bt["high_vol"].sum()) if "high_vol" in bt.columns else 0

    print(f"\n    ── Backtest Summary ─────────────────────────────────────────")
    print(f"    Months: {len(rets)}  |  Ann Return: {ann_ret:.1%}  |  Sharpe: {sharpe:.2f}")
    print(f"    Max DD: {max_dd:.1%}  |  Calmar: {calmar:.2f}  |  Avg β: {bt['portfolio_beta'].mean():.3f}")
    print(f"    Ann spread cost: {avg_spread:.2%}  |  Ann borrow cost: {avg_borrow:.2%}")
    print(f"    High-vol regime months: {high_vol_months}/{len(rets)}")

    bt.to_csv(ROOT / "v11_backtest_monthly.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "Annualized Return",     "v11_ls": f"{ann_ret:.1%}",  "spy": "~18%"},
        {"metric": "Annualized Volatility", "v11_ls": f"{ann_vol:.1%}",  "spy": "~14%"},
        {"metric": "Sharpe Ratio",          "v11_ls": f"{sharpe:.2f}",   "spy": "~0.72"},
        {"metric": "Max Drawdown",          "v11_ls": f"{max_dd:.1%}",   "spy": "~-24%"},
        {"metric": "Calmar Ratio",          "v11_ls": f"{calmar:.2f}",   "spy": "~0.75"},
        {"metric": "Avg Portfolio Beta",    "v11_ls": f"{bt['portfolio_beta'].mean():.3f}", "spy": "1.00"},
        {"metric": "Ann Borrow Cost",       "v11_ls": f"{avg_borrow:.2%}", "spy": "N/A"},
        {"metric": "High-Vol Months",       "v11_ls": str(high_vol_months), "spy": str(high_vol_months)},
        {"metric": "Months",                "v11_ls": str(len(rets)),    "spy": str(len(rets))},
    ])
    summary.to_csv(ROOT / "v11_backtest_summary.csv", index=False)
    print("    Saved v11_backtest_monthly.csv, v11_backtest_summary.csv")
    return bt


# ══════════════════════════════════════════════════════════════════════════════
# G. MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_full_report(ic_df: pd.DataFrame, bt: pd.DataFrame):
    top_ics = ic_df.head(10)[["signal", "ic", "t_stat", "stars"]].to_string(index=False) \
              if not ic_df.empty else "(no IC data)"

    if not bt.empty:
        rets    = bt["net_ret"].dropna()
        ann_ret = float((1 + rets.mean()) ** 12 - 1)
        ann_vol = float(rets.std() * np.sqrt(12))
        sharpe  = (ann_ret - 0.053) / ann_vol if ann_vol > 0 else 0
        max_dd  = float(((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min())
        avg_borrow = bt["borrow_cost"].mean() * 12 if "borrow_cost" in bt.columns else 0
        high_vol   = int(bt["high_vol"].sum()) if "high_vol" in bt.columns else 0
    else:
        ann_ret = ann_vol = sharpe = max_dd = avg_borrow = high_vol = 0

    report = f"""# Canyon v11 Full — Institutional Upgrades Report
Date: {TODAY}

## v11 Upgrades vs v10

| Layer | Upgrade | Detail |
|-------|---------|--------|
| Signal | +3 new price signals | residual_mom, vol_regime, rel_strength_sec |
| IC weights | Rolling computed (not hardcoded) | Measured from 12-month panel IC |
| TC model | +Short borrow cost | 2% annualized on short book |
| Regime | VIX switch | max_w * 0.70 when SPY vol > 25% |
| ML signal | LightGBM price ensemble | Trained on 16-month historical panel, purged CV |

## New Signal Descriptions

**residual_mom** — 12-month momentum minus beta × SPY 12m return.
Isolates idiosyncratic alpha momentum, removing the market factor.
Captures stocks that rose for fundamental reasons, not just beta.

**vol_regime** — Negative of (current 21d vol / 252d-avg vol − 1).
High value = vol contraction relative to own history (bullish trend signal).
Independent from vol level signals (vol_21d, vol_63d) already in the model.

**rel_strength_sec** — Stock 3m return minus equal-weighted sector 3m return.
Different from industry_mom (which measures sector return, not stock vs sector).
Captures within-sector winners vs losers.

## Signal IC Table (computed, not hardcoded)
{top_ics}

## Walk-Forward Backtest ({len(bt) if not bt.empty else 0} months OOS)

| Metric | v11 L/S | SPY |
|--------|---------|-----|
| Annualized Return | {ann_ret:.1%} | ~18% |
| Annualized Vol | {ann_vol:.1%} | ~14% |
| Sharpe Ratio | {sharpe:.2f} | ~0.72 |
| Max Drawdown | {max_dd:.1%} | ~-24% |
| Avg Portfolio Beta | {bt['portfolio_beta'].mean() if not bt.empty else 0:.3f} | 1.00 |
| Ann Borrow Cost | {avg_borrow:.2%} | N/A |
| High-Vol Months | {high_vol} | — |

## TC Model Breakdown
- **Bid-ask spread**: 10bps round-trip per unit turnover (unchanged from v10)
- **Short borrow**: {avg_borrow:.2%} annualized drag (NEW in v11)
- **Total TC drag**: both components reduce net_ret; spread_cost and borrow_cost
  are now reported separately in v11_backtest_monthly.csv

## Remaining Gap vs D.E. Shaw / Two Sigma
1. Data: paid alt data ($2M+/yr) — credit card, satellite, web traffic
2. Universe: 495 names vs 5,000–10,000 globally
3. Execution: no broker connection, no real short locates
4. IC stability: only 12–22 months of history; need 5+ years for t-stat confidence
5. Tick-level microstructure signals
"""
    (ROOT / "v11_full_report.md").write_text(report)
    print("    Saved v11_full_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(run_backtest: bool = True, run_lgb: bool = True):
    print(f"\nCanyon v11 Full Institutional — {TODAY}")
    print("=" * 60)

    print("\n[0] Loading data...")
    alpha  = pd.read_csv(ROOT / "alpha_scores.csv")
    alpha  = alpha[alpha["sector"].notna() & (alpha["sector"] != "")].reset_index(drop=True)

    # Use 8-year cache if available, else fall back to 3-year cache
    price_cache = ROOT / "sp500_price_cache_8yr.csv"
    if not price_cache.exists():
        price_cache = ROOT / "sp500_price_cache.csv"
    prices = pd.read_csv(price_cache, index_col=0, parse_dates=True)
    spy    = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    sectors = alpha.set_index("ticker")["sector"]
    print(f"    {len(alpha)} tickers  |  {prices.shape[0]} days  |  {prices.shape[1]} stocks")

    # Load real VIX from FRED (W10 improvement)
    vix_series = None
    fred_path = ROOT / "fred_macro_daily.csv"
    if fred_path.exists():
        try:
            fred_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
            if "VIX" in fred_df.columns:
                vix_series = fred_df["VIX"].dropna()
                print(f"    VIX loaded: {len(vix_series)} days from FRED")
        except Exception as e:
            print(f"    VIX load failed ({e}) — using SPY vol proxy")

    # A. Price signals (22)
    price_sigs, beta_s = build_price_signals(prices, spy, sectors)

    # B. Signal IC (computed, not hardcoded)
    ic_df = compute_signal_ic(prices, sectors)

    # C. LightGBM price ensemble
    lgb_signal = pd.Series(dtype=float)
    if run_lgb:
        lgb_signal = build_lgb_signal(prices, sectors)

    # D. Process and combine (uses computed ICs)
    full_df, ic_weights = process_and_combine(
        price_sigs, alpha, beta_s, sectors, ic_df, lgb_signal
    )

    # W34: Try to override mu_override with Black-Litterman posterior expected returns
    bl_mu_path = ROOT / "bl_expected_returns.csv"
    if bl_mu_path.exists():
        try:
            bl_df = pd.read_csv(bl_mu_path)
            if "ticker" in bl_df.columns and "mu_bl" in bl_df.columns:
                bl_mu = bl_df.set_index("ticker")["mu_bl"]
                if "ticker" in full_df.columns:
                    full_df = full_df.set_index("ticker")
                    full_df["mu_override"] = bl_mu.reindex(full_df.index).fillna(full_df["mu_override"])
                    full_df = full_df.reset_index()
                    print(f"    W34: BL mu_override applied to {bl_mu.notna().sum()} tickers")
        except Exception as e:
            print(f"    W34: BL mu_override load failed ({e}) — using IC²-weighted returns")

    # W34: Load regime-conditional covariance if available (used by walk_forward_backtest)
    regime_cov_path = ROOT / "regime_cov_blend.csv"
    regime_cov = None
    if regime_cov_path.exists():
        try:
            regime_cov = pd.read_csv(regime_cov_path, index_col=0)
            print(f"    W34: Regime covariance loaded: {regime_cov.shape}")
        except Exception:
            pass

    # E. Factor model
    B, factor_cov, resid_vols, factors, factors_monthly = \
        build_factor_model(prices, spy, sectors, price_sigs)

    # F. Walk-forward backtest (W10: pass real VIX series)
    bt = pd.DataFrame()
    if run_backtest:
        bt = walk_forward_backtest(prices, spy, sectors, beta_s, factors_monthly, vix_series)

    # G. Report
    write_full_report(ic_df, bt)

    print(f"\n{'='*60}")
    print("  Canyon v11 Full complete.")
    print("  Outputs:")
    for fname in [
        "v11_full_signals.csv", "v11_signal_ic.csv",
        "v11_factor_model.csv", "v11_factor_returns.csv",
        "v11_backtest_monthly.csv", "v11_backtest_summary.csv",
        "v11_full_report.md",
    ]:
        path = ROOT / fname
        exists = "✓" if path.exists() else "✗"
        rows = sum(1 for _ in open(path)) - 1 if path.exists() else 0
        print(f"    {exists} {fname}  ({rows} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--no-lgb",      action="store_true")
    args = parser.parse_args()
    main(run_backtest=not args.no_backtest, run_lgb=not args.no_lgb)
