#!/usr/bin/env python3
"""
step_ml_alpha_v2.py — multi-signal machine-learning alpha, validated honestly
=============================================================================
This is what actual quant funds do: instead of testing one textbook rule at a
time (all of which tested ~0), COMBINE many weak signals with a gradient-boosted
model that finds non-linear interactions — then validate with iron discipline.

Features (all point-in-time, no look-ahead), cross-sectionally rank-normalized
each month so the model learns RELATIVE position:
  price   : momentum 12-1, 1-month reversal, 6-month, 63d volatility, 52w-high
  quality : gross-profits-to-assets (TTM, PIT), ROA (TTM, PIT)
  earnings: SUE seasonal surprise (PIT filed_date)

Honesty controls — the ML-specific ones that stop the #1 finance-ML disaster:
  * WALK-FORWARD with EMBARGO: to predict month t we train ONLY on months whose
    outcome (t→t+1 return) was already realized BEFORE t. Zero temporal leakage.
  * Features gated on PIT (filed_date ≤ rebalance date).
  * PIT S&P membership (survivorship-controlled), realistic costs, IS/OOS split.
  * Reports the model's out-of-sample IC t-stat + feature importances.

Output: ml_alpha_v2_results.json
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OOS_CUTOFF = pd.Timestamp("2020-01-01")
TC_BPS = 10.0
TOP_FRAC = 0.10
MIN_TRAIN_MONTHS = 36          # need history before first prediction
EMBARGO_MONTHS = 1            # gap so training outcomes are known pre-trade


# ── loaders ──────────────────────────────────────────────────────────────────
def load_prices():
    p = ROOT / "sp500_price_history_deep.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return df[[c for c in df.columns if str(c).isalpha()]]


def load_pit():
    p = ROOT / "sp500_pit_membership.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p, parse_dates=["date"])
    m["ticker"] = m["ticker"].astype(str).str.replace("-", ".", regex=False)
    return {d: set(g["ticker"]) for d, g in m.groupby("date")}


def members_asof(pit, date):
    if not pit:
        return None
    ks = [k for k in pit if k <= date]
    return pit[max(ks)] if ks else None


def load_quality():
    import step_quality_v2 as Q
    fp = ROOT / "quarterly_fundamentals.csv"
    return Q.prep(pd.read_csv(fp)) if fp.exists() else pd.DataFrame()


def load_sue():
    import step_pead_strategy as P
    fp = ROOT / "eps_pit.csv"
    return P.compute_sue(pd.read_csv(fp)) if fp.exists() else pd.DataFrame()


def _latest_asof(df, date, valcol):
    w = df[df["filed_date"] <= date]
    if w.empty:
        return pd.Series(dtype=float)
    w = w.sort_values("filed_date").drop_duplicates("ticker", keep="last")
    return w.set_index("ticker")[valcol]


# ── feature panel ────────────────────────────────────────────────────────────
def build_features(prices, qual, sue, d0):
    hist = prices.loc[:d0]
    if len(hist) < 260:
        return pd.DataFrame()
    px_t = hist.iloc[-1]
    feats = {}
    feats["mom_12_1"] = hist.iloc[-21] / hist.iloc[-252] - 1
    feats["mom_6"]    = hist.iloc[-1] / hist.iloc[-126] - 1
    feats["rev_1m"]   = -(hist.iloc[-1] / hist.iloc[-21] - 1)          # reversal
    logr = np.log(hist / hist.shift(1))
    feats["vol_63"]   = -(logr.tail(63).std())                        # low-vol good → negate
    feats["high52"]   = px_t / hist.tail(252).max()
    df = pd.DataFrame(feats)
    if not qual.empty:
        df["gpa"] = _latest_asof(qual, d0, "gpa")
        df["roa"] = _latest_asof(qual, d0, "roa")
    if not sue.empty:
        # only surprises filed within 90d (fresh) else 0
        s = sue[(sue["filed_date"] <= d0) & (sue["filed_date"] > d0 - pd.Timedelta(days=90))]
        s = s.sort_values("filed_date").drop_duplicates("ticker", keep="last")
        df["sue"] = s.set_index("ticker")["sue"]
    return df


def rank_norm(df):
    """cross-sectional rank to [-0.5,0.5] per column (relative position)."""
    return df.rank(pct=True) - 0.5


def run():
    import lightgbm as lgb
    prices = load_prices()
    pit = load_pit()
    qual = load_quality()
    sue = load_sue()
    daily = prices.pct_change(fill_method=None)

    idx = prices.index
    md = pd.DataFrame({"d": idx}); md["m"] = md["d"].dt.to_period("M")
    rebs = [g["d"].iloc[0] for _, g in md.groupby("m")]
    rebs = [r for r in rebs if r >= pd.Timestamp("2011-01-01")]

    # assemble monthly cross-sections: X features, y = next-month return
    months = []
    for i in range(len(rebs) - 1):
        d0, d1 = rebs[i], rebs[i + 1]
        X = build_features(prices, qual, sue, d0)
        if X.empty:
            continue
        mem = members_asof(pit, d0)
        if mem is not None:
            X = X[X.index.isin(mem)]
        X = X[X.index.isin(prices.columns)].dropna(thresh=4)  # need ≥4 features
        if len(X) < 40:
            continue
        win = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        fwd = pd.Series({t: (1 + win[t].fillna(0)).prod() - 1 for t in X.index if t in win.columns})
        Xn = rank_norm(X).fillna(0.0)
        y = fwd.reindex(Xn.index)
        months.append({"d0": d0, "d1": d1, "X": Xn, "y": y, "win": win})

    feat_cols = list(months[0]["X"].columns)
    ic_rows, ret_rows, imp_acc = [], [], np.zeros(len(feat_cols))
    n_models = 0
    prev = set()
    for i in range(len(months)):
        # walk-forward: train on months whose OUTCOME is known before trading month i
        train = [m for j, m in enumerate(months) if j <= i - 1 - EMBARGO_MONTHS]
        if len(train) < MIN_TRAIN_MONTHS:
            continue
        Xtr = pd.concat([m["X"] for m in train]); ytr = pd.concat([m["y"] for m in train]).reindex(Xtr.index)
        ok = ytr.notna()
        Xtr, ytr = Xtr[ok], ytr[ok]
        model = lgb.LGBMRegressor(n_estimators=120, num_leaves=15, learning_rate=0.05,
                                  min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=1.0, verbose=-1)
        model.fit(Xtr[feat_cols], ytr)
        imp_acc += model.feature_importances_; n_models += 1

        mth = months[i]
        pred = pd.Series(model.predict(mth["X"][feat_cols]), index=mth["X"].index)
        realized = mth["y"]
        merged = pd.DataFrame({"pred": pred, "fwd": realized}).dropna()
        if len(merged) >= 30:
            ic = merged["pred"].corr(merged["fwd"], method="spearman")
            if not np.isnan(ic):
                ic_rows.append({"date": mth["d0"], "ic": ic})
        # trade top-decile
        n_top = max(10, int(len(pred) * TOP_FRAC))
        longs = pred.sort_values(ascending=False).head(n_top).index.tolist()
        turn = len(set(longs) - prev) / max(len(longs), 1)
        tc = turn * 2 * TC_BPS / 10_000
        prev = set(longs)
        for k, (dt, row) in enumerate(mth["win"].iterrows()):
            r = np.nanmean([row.get(t, np.nan) for t in longs])
            r = 0.0 if np.isnan(r) else r
            ret_rows.append({"date": dt, "net": r - (tc if k == 0 else 0.0), "spy": row.get("SPY", np.nan)})

    imp = dict(zip(feat_cols, (imp_acc / max(n_models, 1)).round(1))) if n_models else {}
    return _metrics(pd.DataFrame(ic_rows), pd.DataFrame(ret_rows), imp, n_models)


def _stat(r, ann=252):
    r = r.dropna()
    if len(r) < 30:
        return {}
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() else np.nan
    c = (1 + r).cumprod(); mdd = float((c / c.cummax() - 1).min())
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 2), "max_dd": round(mdd, 4)}


def _metrics(ic_df, ret_df, imp, n_models):
    if ret_df.empty:
        return {"error": "no rows"}
    ret_df = ret_df.set_index("date")
    net, spy = ret_df["net"], ret_df["spy"].dropna()
    ic = ic_df["ic"].dropna() if not ic_df.empty else pd.Series(dtype=float)
    ic_t = float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std() else float("nan")
    common = ret_df.dropna(subset=["spy"])
    beta = alpha = np.nan
    if len(common) > 60:
        x, y = common["spy"].values, common["net"].values
        beta = float(np.cov(x, y)[0, 1] / np.var(x))
        alpha = float((y.mean() - beta * x.mean()) * 252)
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy": "LightGBM multi-signal, walk-forward w/ embargo, long top decile",
        "n_models_trained": n_models,
        "oos_ic_mean": round(float(ic.mean()), 4) if len(ic) else None,
        "oos_ic_t": round(ic_t, 2) if not np.isnan(ic_t) else None,
        "oos_ic_n": int(len(ic)),
        "feature_importance": imp,
        "full_net": _stat(net),
        "in_sample_pre2020": _stat(net[ret_df.index < OOS_CUTOFF]),
        "out_of_sample_2020on": _stat(net[ret_df.index >= OOS_CUTOFF]),
        "spy_buy_hold": _stat(spy),
        "beta_to_spy": round(beta, 3) if not np.isnan(beta) else None,
        "alpha_annual_after_costs": round(alpha, 4) if not np.isnan(alpha) else None,
    }


def main():
    print("=" * 62)
    print("ML alpha v2 — LightGBM multi-signal, honest walk-forward")
    print("=" * 62)
    m = run()
    json.dump(m, open(ROOT / "ml_alpha_v2_results.json", "w"), indent=2, default=str)
    if "error" in m:
        print("  " + m["error"]); return
    print(f"  models trained (walk-forward): {m['n_models_trained']}")
    print(f"  OOS prediction IC: {m['oos_ic_mean']}  t={m['oos_ic_t']}  (n={m['oos_ic_n']} months)")
    f = m["full_net"]
    print(f"  Long top-decile: Sharpe {f.get('sharpe')}  CAGR {f.get('cagr')}  MaxDD {f.get('max_dd')}")
    print(f"  Beta {m.get('beta_to_spy')}  Alpha/yr after costs {m.get('alpha_annual_after_costs')}")
    print(f"  IS Sharpe {m['in_sample_pre2020'].get('sharpe') if m['in_sample_pre2020'] else 'n/a'} | "
          f"OOS Sharpe {m['out_of_sample_2020on'].get('sharpe') if m['out_of_sample_2020on'] else 'n/a'}")
    print(f"  feature importance: {m['feature_importance']}")


if __name__ == "__main__":
    main()
