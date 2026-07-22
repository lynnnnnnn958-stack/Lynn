"""
Improvement 8 — LightGBM Factor Synthesis
==========================================
Walk-forward ML factor combination:
  - Training: rolling 60-period window
  - Prediction: 1-period out-of-sample (40 predictions total)
  - Features: all 6 price factors + EDGAR fundamentals (where available)
  - Target: forward 21-day return rank (percentile 0-100)
  - Compare: LGBM IC vs best single-factor IC

Then run a full backtest using LGBM stock ranking as selection signal.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available; using Ridge regression fallback")
    from sklearn.linear_model import Ridge

ROOT   = Path(__file__).parent
HOLD_M = 21
WARMUP = 252

# ── Data Load ─────────────────────────────────────────────────────────
print("Loading data...")
prices  = pd.read_csv(ROOT/"sp500_price_8yr.csv", index_col=0, parse_dates=True).sort_index()
spy     = prices["SPY"].copy()
prices  = prices.drop(columns=["SPY"])
sector_map = pd.read_csv(ROOT/"sp500_sectors.csv", index_col=0).squeeze().to_dict() \
              if (ROOT/"sp500_sectors.csv").exists() else {}

common  = prices.index.intersection(spy.index)
prices  = prices.reindex(common)
spy     = spy.reindex(common).ffill()
price_arr = prices.values.astype(float)
tickers   = list(prices.columns)
N = len(tickers)
col_idx = {c: i for i, c in enumerate(tickers)}

# EDGAR fundamental scores
edgar_df = pd.read_csv(ROOT/"edgar_fundamentals.csv") \
           if (ROOT/"edgar_fundamentals.csv").exists() else pd.DataFrame()
edgar_df["filed_date"] = pd.to_datetime(edgar_df["filed_date"], errors="coerce")

rebal_all = list(range(WARMUP, len(prices)-HOLD_M, HOLD_M))
print(f"  {len(rebal_all)} rebalancing periods  {N} stocks")

# ── Compute factor features for each period × stock ──────────────────
print("Computing factor features for all periods...")

def get_edgar_scores(as_of_date):
    """Point-in-time EDGAR scores as of as_of_date."""
    if edgar_df.empty:
        return {}
    available = edgar_df[edgar_df["filed_date"] <= as_of_date]
    if available.empty:
        return {}
    latest = available.sort_values("filed_date").groupby("ticker").last()
    scores = {}
    for tk in latest.index:
        row = latest.loc[tk]
        acc = float(row.get("accrual_ratio", np.nan))
        gm  = float(row.get("gross_margin",  np.nan))
        roa = float(row.get("roa",            np.nan))
        scores[tk] = dict(accrual=acc, gross_margin=gm, roa=roa)
    return scores

all_sectors = sorted(set(sector_map.values()))

# Sector one-hot for stacking
sector_codes = {s: i for i, s in enumerate(all_sectors)}

# Build feature matrix: shape (n_periods, N_stocks, n_features)
FEATURE_NAMES = ["mom_12_1", "mom_3m", "mom_1m", "inv_vol", "above_200", "rsi_signal",
                 "accrual", "gross_margin", "roa",
                 "sector_momentum"]   # 10 features

print(f"  Feature set: {FEATURE_NAMES}")

# Store all features and targets period-by-period
period_features = []   # list of (N, 10) arrays
period_targets  = []   # list of (N,) arrays (forward return rank)
period_dates    = []

for j, t in enumerate(rebal_all):
    p   = price_arr[:t+1]
    now = p[-1]
    p1m = p[-22]
    p3m = p[-64]  if t >= 63  else p[0]
    p_lk= p[-273] if t >= 272 else (p[-252] if t >= 252 else p[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        mom_12_1 = np.where(p_lk>0, now/p_lk-1, np.nan).clip(-1,1)
        mom_3m   = np.where(p3m>0,  now/p3m-1,  np.nan).clip(-1,1)
        mom_1m   = np.where(p1m>0,  now/p1m-1,  np.nan).clip(-0.5,0.5)

    log_p  = np.log(np.where(p[-22:]>0, p[-22:], np.nan))
    vol20  = np.nanstd(np.diff(log_p, axis=0), axis=0)
    invvol = 1.0 / (vol20 + 0.005)
    sma200 = np.nanmean(p[-200:], axis=0) if t >= 199 else np.nanmean(p, axis=0)
    above  = (now > sma200).astype(float)

    delta = np.diff(log_p, axis=0)
    gains = np.nanmean(np.clip(delta[-14:], 0, None), axis=0)
    loss  = np.nanmean(np.clip(-delta[-14:], 0, None), axis=0)
    rsi14 = 100 - 100/(1 + gains/(loss+1e-9))
    rsi_signal = (100 - rsi14).clip(0, 100) / 100   # invert: low RSI → high signal

    # Forward returns (target)
    t1  = min(t+HOLD_M, len(price_arr)-1)
    p0  = price_arr[t]; p1 = price_arr[t1]
    with np.errstate(divide='ignore', invalid='ignore'):
        fwd_r = np.where((p0>0)&(p1>0), p1/p0-1, np.nan)

    # EDGAR scores
    as_of = prices.index[t]
    edgar_scores = get_edgar_scores(as_of)
    accrual_arr  = np.array([edgar_scores.get(tk, {}).get("accrual",np.nan) for tk in tickers])
    gm_arr       = np.array([edgar_scores.get(tk, {}).get("gross_margin",np.nan) for tk in tickers])
    roa_arr      = np.array([edgar_scores.get(tk, {}).get("roa",np.nan) for tk in tickers])

    # Sector momentum: average 3m momentum within each sector
    sec_mom = np.full(N, np.nan)
    for sec in all_sectors:
        sec_idxs = [col_idx[tk] for tk in tickers if sector_map.get(tk)==sec and tk in col_idx]
        if not sec_idxs: continue
        avg_m = np.nanmean(mom_3m[sec_idxs])
        for ci in sec_idxs:
            sec_mom[ci] = avg_m

    # Stack features (N × 10)
    feat = np.column_stack([
        mom_12_1, mom_3m, mom_1m, invvol, above, rsi_signal,
        accrual_arr, gm_arr, roa_arr, sec_mom
    ])

    # Rank targets (cross-sectional)
    valid = np.isfinite(fwd_r)
    target_rank = np.full(N, np.nan)
    if valid.sum() >= 20:
        target_rank[valid] = (fwd_r[valid].argsort().argsort() + 1) / valid.sum() * 100

    period_features.append(feat)
    period_targets.append(target_rank)
    period_dates.append(prices.index[t])

    if (j+1) % 20 == 0:
        print(f"  Period {j+1}/{len(rebal_all)}")

print(f"\nFeature matrix: {len(period_features)} periods × {N} stocks × {len(FEATURE_NAMES)} features")

# ── Walk-forward ML prediction ────────────────────────────────────────
TRAIN_WINDOW = 60   # periods for training
MIN_TRAIN    = 40

predictions  = []   # (period_idx, stock_idx, predicted_rank)
ic_history   = []

# Also compute linear IC for comparison
single_factor_ics = {f: [] for f in FEATURE_NAMES}

print(f"\nWalk-forward LGBM (train={TRAIN_WINDOW} periods, OOS starts at period {TRAIN_WINDOW+1})...")

for j in range(TRAIN_WINDOW, len(rebal_all)):
    # Build training data
    X_train, y_train = [], []
    for k in range(j-TRAIN_WINDOW, j):
        feat   = period_features[k]
        target = period_targets[k]
        valid  = np.isfinite(target) & np.all(~np.isnan(feat), axis=1)
        # Impute NaN features with column median
        feat_clean = feat.copy()
        for col in range(feat.shape[1]):
            col_vals = feat_clean[:, col]
            nan_mask = ~np.isfinite(col_vals)
            if nan_mask.any():
                med = np.nanmedian(col_vals)
                col_vals[nan_mask] = med if np.isfinite(med) else 0.0
                feat_clean[:, col] = col_vals
        X_train.append(feat_clean[valid])
        y_train.append(target[valid])

    if not X_train: continue
    X_tr = np.vstack(X_train)
    y_tr = np.concatenate(y_train)

    # Test period
    feat_test  = period_features[j].copy()
    target_test = period_targets[j]
    # Impute test features
    for col in range(feat_test.shape[1]):
        nan_m = ~np.isfinite(feat_test[:, col])
        if nan_m.any():
            med = np.nanmedian(X_tr[:, col])
            feat_test[nan_m, col] = med if np.isfinite(med) else 0.0

    valid_test = np.isfinite(target_test)
    if valid_test.sum() < 20: continue

    # Single factor ICs (no ML)
    for fi, fname in enumerate(FEATURE_NAMES):
        f_vals  = feat_test[:, fi]
        f_valid = np.isfinite(f_vals) & valid_test
        if f_valid.sum() >= 10:
            ic, _ = spearmanr(f_vals[f_valid], target_test[f_valid])
            if np.isfinite(ic):
                single_factor_ics[fname].append(ic)

    # Train model
    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            num_leaves=8, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1
        )
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=10.0)

    model.fit(X_tr, y_tr)
    pred = model.predict(feat_test)

    # IC for this period
    ic, _ = spearmanr(pred[valid_test], target_test[valid_test])
    if np.isfinite(ic):
        ic_history.append(ic)

    # Store predictions
    for i, p_val in enumerate(pred):
        predictions.append((j, i, float(p_val)))

    if (j - TRAIN_WINDOW + 1) % 10 == 0:
        recent_ic = np.mean(ic_history[-10:]) if ic_history else 0
        print(f"  OOS period {j-TRAIN_WINDOW+1}/{len(rebal_all)-TRAIN_WINDOW}  "
              f"Last-10 IC: {recent_ic:+.4f}")

# ── IC Analysis ──────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  LGBM FACTOR SYNTHESIS — IC Analysis")
print("═"*70)

ic_arr = np.array(ic_history)
if len(ic_arr) > 0:
    t_stat = ic_arr.mean() / (ic_arr.std() / np.sqrt(len(ic_arr)) + 1e-9)
    print(f"\n  LGBM Combined Signal:")
    print(f"    OOS periods: {len(ic_arr)}")
    print(f"    Mean IC:     {ic_arr.mean():+.4f}")
    print(f"    IC Std:      {ic_arr.std():.4f}")
    print(f"    t-stat:      {t_stat:+.2f}")
    print(f"    Hit rate:    {(ic_arr>0).mean():.0%}")
    sig = "** SIGNIFICANT" if abs(t_stat) > 2.0 else ("* MARGINAL" if abs(t_stat) > 1.5 else "not significant")
    print(f"    Result:      {sig}")

print(f"\n  Single-Factor IC (same OOS window, comparison):")
print(f"  {'Factor':<18} {'Mean IC':>9}  {'t-stat':>8}  {'Hit%':>6}")
print("  " + "─"*45)
for fname, ics in single_factor_ics.items():
    if not ics: continue
    a = np.array(ics)
    t = a.mean() / (a.std()/np.sqrt(len(a)) + 1e-9)
    marker = " **" if abs(t) > 2.0 else (" *" if abs(t) > 1.5 else "")
    print(f"  {fname:<18} {a.mean():>+9.4f}  {t:>+8.2f}  {(a>0).mean():>5.0%}{marker}")

if len(ic_arr) > 0:
    lgbm_tstat = ic_arr.mean() / (ic_arr.std() / np.sqrt(len(ic_arr)) + 1e-9)
    max_single_t = max(abs(np.array(ics).mean() / (np.array(ics).std()/np.sqrt(len(ics))+1e-9))
                       for ics in single_factor_ics.values() if ics)
    print(f"\n  LGBM improvement over best single factor:")
    print(f"    LGBM t-stat:          {lgbm_tstat:+.2f}")
    print(f"    Best single-factor:   {max_single_t:+.2f}")
    print(f"    Gain:                 {lgbm_tstat - max_single_t:+.2f}")

# ── Feature importance ────────────────────────────────────────────────
if HAS_LGBM and len(period_features) > TRAIN_WINDOW:
    # Train on last TRAIN_WINDOW periods for feature importance
    X_fi, y_fi = [], []
    for k in range(len(rebal_all)-TRAIN_WINDOW, len(rebal_all)):
        feat   = period_features[k].copy()
        target = period_targets[k]
        valid  = np.isfinite(target)
        for col in range(feat.shape[1]):
            nan_m = ~np.isfinite(feat[:, col])
            if nan_m.any():
                feat[nan_m, col] = np.nanmedian(feat[:, col]) or 0.0
        X_fi.append(feat[valid]); y_fi.append(target[valid])
    X_fi = np.vstack(X_fi); y_fi = np.concatenate(y_fi)
    m_fi = lgb.LGBMRegressor(n_estimators=200, max_depth=3, num_leaves=8,
                               random_state=42, verbose=-1)
    m_fi.fit(X_fi, y_fi)
    imp = m_fi.feature_importances_
    print(f"\n  Feature importance (trained on last {TRAIN_WINDOW} periods):")
    for fname, im in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
        bar = "█" * int(im/max(imp)*20)
        print(f"  {fname:<18} {im:>6.0f}  {bar}")

# ── Backtest using LGBM predictions as stock selection signal ─────────
if len(predictions) > 0 and len(ic_history) > 5:
    print(f"\n  Running backtest with LGBM stock selection...")
    letf_df = pd.read_csv(ROOT/"letf_prices.csv", index_col=0, parse_dates=True).sort_index()
    common2 = prices.index.intersection(letf_df.index)
    prices2 = prices.reindex(common2)
    tqqq2   = letf_df["TQQQ"].reindex(common2).ffill().values.astype(float)
    spy2    = spy.reindex(common2).ffill().values.astype(float)
    price2  = prices2.values.astype(float)

    spy_ma2 = np.full(len(spy2), np.nan)
    for i in range(199, len(spy2)):
        spy_ma2[i] = np.mean(spy2[i-199:i+1])

    # Build LGBM prediction dict: {period_idx: {stock_idx: pred_rank}}
    pred_dict = {}
    for pidx, sidx, pval in predictions:
        pred_dict.setdefault(pidx, {})[sidx] = pval

    TOP_N_LGBM = 25
    records = []

    for j, t in enumerate(rebal_all):
        if j not in pred_dict:
            # Before OOS: use fixed v11 signal (not LGBM)
            continue

        t1 = min(t+HOLD_M, len(price2)-1)
        preds = pred_dict[j]

        # Top-25 by LGBM predicted rank
        sorted_by_pred = sorted(preds.items(), key=lambda x: -x[1])[:TOP_N_LGBM*2]
        sel_idx = [sidx for sidx, _ in sorted_by_pred[:TOP_N_LGBM] if sidx < N]

        # Stock return (equal weight)
        sr = 0.0
        for sidx in sel_idx:
            p0, p1 = price2[t, sidx], price2[t1, sidx]
            if p0 > 0 and np.isfinite(p1):
                sr += (p1/p0 - 1) / len(sel_idx)

        # ETF (same as v17)
        bull = t < 200 or float(spy2[t]) > float(spy_ma2[t]) if np.isfinite(spy_ma2[t]) else True
        if bull:
            w  = tqqq2[max(0,t-20):t]
            w  = np.diff(np.log(np.where(w>0, w, np.nan)))
            w  = w[np.isfinite(w)]
            vol = float(np.std(w)*np.sqrt(252)) if len(w)>=5 else 0.40
            ew  = float(np.clip(0.18/(vol+0.001), 0, 0.40))
            p0t, p1t = float(tqqq2[t]), float(tqqq2[t1])
            er  = (p1t/p0t-1) if p0t>0 and np.isfinite(p1t) else 0.0
        else:
            ew, er = 0.0, 0.0

        sw  = 0.40 if bull else 0.20
        ret = sw*sr + ew*er - 5/10_000
        spy_r = float(spy2[t1])/float(spy2[t]) - 1
        records.append(dict(date=prices2.index[t].strftime("%Y-%m-%d"),
                             ret=round(float(ret),6), spy_ret=round(float(spy_r),6)))

    if records:
        df_lgbm = pd.DataFrame(records)
        r   = np.array(df_lgbm["ret"])
        n   = len(r); ppy = 252/HOLD_M
        tot = float(np.prod(1+r))
        ar  = float(tot**(ppy/n)-1)
        av  = float(r.std()*np.sqrt(ppy))
        sr  = ar/(av+1e-9)
        cum = np.cumprod(1+r)
        mdd = float(np.max((np.maximum.accumulate(cum)-cum)/np.maximum.accumulate(cum)))
        print(f"\n  LGBM Backtest (OOS periods {TRAIN_WINDOW+1}-{len(rebal_all)}):")
        print(f"    AR: {ar:+.1%}  Sharpe: {sr:.2f}  MDD: {-mdd:.1%}")
        print(f"    Periods: {n}  ({df_lgbm['date'].iloc[0]} → {df_lgbm['date'].iloc[-1]})")
        df_lgbm.to_csv(ROOT/"backtest_v17_lgbm.csv", index=False)
        print(f"    Saved: backtest_v17_lgbm.csv")

print("\n" + "═"*70)

# Save IC history
pd.DataFrame({"date": period_dates[TRAIN_WINDOW:], "lgbm_ic": ic_history[:len(period_dates[TRAIN_WINDOW:])]}).to_csv(
    ROOT/"lgbm_ic_history.csv", index=False)
print(f"  Saved: lgbm_ic_history.csv")
