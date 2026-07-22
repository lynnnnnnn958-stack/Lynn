"""
Canyon v9  Step 68 — Fundamental Features + Enhanced ML
=========================================================
Adds balance-sheet and valuation signals to the price-only ML model
from Step 66.  Fundamental data sourced from yfinance .info (free,
always available) with column-level graceful degradation.

New features added to ML pipeline:
  pe_ratio        Trailing P/E (lower = cheaper)
  pb_ratio        Price-to-book
  ps_ratio        Price-to-sales
  roe             Return on equity (%)
  debt_equity     Total debt / equity
  earnings_growth YoY EPS growth (%)
  revenue_growth  YoY revenue growth (%)
  profit_margin   Net profit margin (%)
  analyst_score   Analyst consensus: 1=StrongBuy … 5=StrongSell (inverted → higher=better)
  eps_fwd_yield   Forward EPS / Price (earnings yield)

IC comparison: price-only vs price+fundamental model.

Outputs:
  fundamental_features.csv       raw fundamental data per ticker
  enhanced_ml_scores.csv         combined model scores (latest rebalance)
  fundamental_ic_comparison.csv  IC: base vs price-only ML vs enhanced ML
  fundamental_report.md          full markdown report

Usage:
  python canyon_final_v9_step68_fundamental_features.py
  python canyon_final_v9_step68_fundamental_features.py --refresh   # force re-download
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

FUNDAMENTAL_CACHE   = ROOT / "fundamental_cache.json"
FUNDAMENTAL_TTL_H   = 24     # hours before re-download
LOOKBACK_DAYS       = 252
FEATURES_PRICE = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "rsi_14", "inv_vol",
    "rank_mom", "rank_trend", "spy_regime",
]
FEATURES_FUNDAMENTAL = [
    "pe_ratio", "pb_ratio", "ps_ratio", "roe",
    "debt_equity", "earnings_growth", "revenue_growth",
    "profit_margin", "analyst_score", "eps_fwd_yield",
    # ── Institutional-grade signals (added v9.1) ──────────────────────────────
    "fcf_yield",       # Free Cash Flow / Market Cap  — quality screen
    "roe_trend",       # QoQ earnings growth proxy for ROE direction
    "analyst_upside",  # (Target Price / Current Price) − 1
]
FEATURES_ALL = FEATURES_PRICE + FEATURES_FUNDAMENTAL

UNIVERSE = [
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP",
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
    "CVX","XOM","JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY",
    "TMO","COST","V","MA","HD","NFLX","INTC","QCOM","TXN","AVGO",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fundamental data loader
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def fetch_fundamentals_yf(ticker: str) -> dict:
    """Pull fundamental metrics from yfinance .info."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}

    price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), np.nan)
    eps_fwd = _safe_float(info.get("forwardEps"), np.nan)
    eps_fwd_yield = eps_fwd / price if (np.isfinite(eps_fwd) and np.isfinite(price) and price > 0) else np.nan

    # Analyst score: 1=StrongBuy … 5=StrongSell → invert → higher = better
    rec_mean = _safe_float(info.get("recommendationMean"), np.nan)
    analyst_score = (6.0 - rec_mean) / 5.0 if np.isfinite(rec_mean) else np.nan  # → [0.2, 1.0]

    # ── Institutional-grade signals (v9.1) ────────────────────────────────────
    # FCF Yield: trailing free cash flow / market cap (quality / value hybrid)
    fcf     = _safe_float(info.get("freeCashflow"), np.nan)
    mktcap  = _safe_float(info.get("marketCap"), np.nan)
    fcf_yield = fcf / mktcap if (np.isfinite(fcf) and np.isfinite(mktcap) and mktcap > 0) else np.nan

    # ROE Trend: quarterly earnings growth (YoY) — proxy for ROE direction
    # earningsQuarterlyGrowth > 0 means earnings improving → ROE likely rising
    roe_trend = _safe_float(info.get("earningsQuarterlyGrowth"), np.nan)

    # Analyst Upside: (mean price target / current price) − 1
    target_price = _safe_float(info.get("targetMeanPrice"), np.nan)
    analyst_upside = (target_price / price - 1.0) if (
        np.isfinite(target_price) and np.isfinite(price) and price > 0
    ) else np.nan

    return {
        "ticker":          ticker,
        "as_of":           datetime.now().strftime("%Y-%m-%d"),
        "pe_ratio":        _safe_float(info.get("trailingPE")),
        "pb_ratio":        _safe_float(info.get("priceToBook")),
        "ps_ratio":        _safe_float(info.get("priceToSalesTrailing12Months")),
        "roe":             _safe_float(info.get("returnOnEquity")),
        "debt_equity":     _safe_float(info.get("debtToEquity")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "revenue_growth":  _safe_float(info.get("revenueGrowth")),
        "profit_margin":   _safe_float(info.get("profitMargins")),
        "analyst_score":   analyst_score,
        "eps_fwd_yield":   eps_fwd_yield,
        "fcf_yield":       fcf_yield,
        "roe_trend":       roe_trend,
        "analyst_upside":  analyst_upside,
        "market_cap":      mktcap,
        "sector":          str(info.get("sector", "")),
    }


def load_fundamentals(tickers: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """
    Load from JSON cache (24h TTL) or re-download from yfinance.
    Returns DataFrame with one row per ticker.
    """
    now = time.time()
    cached: dict = {}

    if FUNDAMENTAL_CACHE.exists() and not force_refresh:
        try:
            raw_cache = json.loads(FUNDAMENTAL_CACHE.read_text())
            cache_ts  = raw_cache.get("_ts", 0)
            age_h     = (now - cache_ts) / 3600
            if age_h < FUNDAMENTAL_TTL_H:
                cached = raw_cache.get("data", {})
                print(f"  [fund cache] {len(cached)} tickers, {age_h:.1f}h old")
        except Exception:
            pass

    new_tickers = [t for t in tickers if t not in cached]
    if new_tickers:
        print(f"  [yfinance] Fetching fundamentals for {len(new_tickers)} tickers …")
        for i, tk in enumerate(new_tickers):
            data = fetch_fundamentals_yf(tk)
            cached[tk] = data
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(new_tickers)} done")
        # Save cache
        FUNDAMENTAL_CACHE.write_text(json.dumps({"_ts": now, "data": cached}, default=str))

    rows = [cached[t] for t in tickers if t in cached and cached[t]]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge fundamentals with price panel
# ─────────────────────────────────────────────────────────────────────────────

def merge_fundamentals(panel: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fundamental features into panel (broadcast as time-invariant within each ticker).
    Fundamentals updated quarterly; treating as constant for the ~1-year ML window
    is a reasonable approximation — flag as caveat in report.
    """
    if fund_df.empty:
        return panel

    fund_cols = ["ticker"] + [c for c in FEATURES_FUNDAMENTAL if c in fund_df.columns]
    fund_sub  = fund_df[fund_cols].copy()

    # Normalise fundamentals cross-sectionally (rank ∈ [0,1]) to reduce outlier impact
    for col in FEATURES_FUNDAMENTAL:
        if col in fund_sub.columns:
            fund_sub[col] = pd.to_numeric(fund_sub[col], errors="coerce")
            # Rank transform to [0,1]
            fund_sub[f"r_{col}"] = fund_sub[col].rank(pct=True, na_option="keep")

    merged = panel.merge(fund_sub, on="ticker", how="left")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-run walk-forward with combined features
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_enhanced(
    panel: pd.DataFrame,
    feature_cols: list[str],
    lookback: int = LOOKBACK_DAYS,
    label: str = "enhanced",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward ML using `feature_cols`. Returns (preds_df, ic_df).
    Mirrors Step 66 logic but accepts arbitrary feature set.
    """
    dates = panel["date"].sort_values().unique()
    reb_dates = []
    prev_month = None
    for d in dates:
        m = pd.Timestamp(d).month
        if m != prev_month:
            reb_dates.append(d)
            prev_month = m

    min_date = dates[min(lookback, len(dates) - 1)]
    reb_dates = [d for d in reb_dates if d >= min_date]

    all_preds = []
    scaler = StandardScaler()

    for reb_date in reb_dates:
        train_end   = pd.Timestamp(reb_date) - pd.Timedelta(days=1)
        train_start = pd.Timestamp(reb_date) - pd.Timedelta(days=lookback * 1.6)

        train_mask = (
            (panel["date"] >= train_start) &
            (panel["date"] <= train_end) &
            (panel["forward_ret"].notna())
        )
        train_df = panel[train_mask].dropna(subset=feature_cols)
        if len(train_df) < 200:
            continue
        train_df = train_df.copy()
        train_df["y_ranked"] = train_df.groupby("date")["forward_ret"].rank(pct=True)
        X_tr = train_df[feature_cols].fillna(train_df[feature_cols].median()).values
        y_tr = train_df["y_ranked"].values

        pred_mask = panel["date"] == reb_date
        pred_df   = panel[pred_mask].dropna(subset=FEATURES_PRICE)   # price features required
        if pred_df.empty:
            continue
        X_pr = pred_df[feature_cols].fillna(train_df[feature_cols].median()).values

        try:
            scaler.fit(X_tr)
            rf = RandomForestRegressor(n_estimators=60, max_depth=4,
                                       min_samples_leaf=10, random_state=42, n_jobs=-1)
            rf.fit(scaler.transform(X_tr), y_tr)
            scores = rf.predict(scaler.transform(X_pr))
        except Exception:
            scores = np.zeros(len(pred_df))

        for i, (_, row) in enumerate(pred_df.iterrows()):
            all_preds.append({
                "rebalance_date": str(reb_date)[:10],
                "ticker":         row["ticker"],
                f"{label}_score": float(scores[i]),
            })

    preds_df = pd.DataFrame(all_preds)
    if preds_df.empty:
        return preds_df, pd.DataFrame()

    # IC calculation
    panel_sub = panel.copy()
    panel_sub["rebalance_date"] = panel_sub["date"].astype(str).str[:10]
    merged = preds_df.merge(
        panel_sub[["rebalance_date","ticker","forward_ret"]],
        on=["rebalance_date","ticker"], how="inner",
    ).dropna(subset=["forward_ret"])

    ics = []
    for date, grp in merged.groupby("rebalance_date"):
        sc_col = f"{label}_score"
        if sc_col not in grp.columns or len(grp) < 5:
            continue
        x = pd.to_numeric(grp[sc_col], errors="coerce")
        y = pd.to_numeric(grp["forward_ret"], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        ic, _ = spearmanr(x[mask], y[mask])
        if not np.isnan(ic):
            ics.append(ic)

    if not ics:
        ic_row = {"model": label, "mean_ic": 0.0, "t_stat": 0.0, "status": "NO_DATA"}
    else:
        arr = np.array(ics)
        from scipy.stats import ttest_1samp
        mean_ic = float(arr.mean())
        t_stat  = float(mean_ic / (arr.std() / np.sqrt(len(arr)) + 1e-10))
        _, pv   = ttest_1samp(arr, 0)
        status  = ("STRONG" if mean_ic > 0.05 and abs(t_stat) > 2.0
                   else "USABLE" if mean_ic > 0.03 and abs(t_stat) > 2.0
                   else "WEAK" if mean_ic > 0.0 else "NEGATIVE")
        ic_row  = {"model": label, "mean_ic": round(mean_ic, 4),
                   "std_ic": round(float(arr.std()), 4),
                   "t_stat": round(t_stat, 2), "p_value": round(float(pv), 4),
                   "n_periods": len(ics), "status": status}

    return preds_df, pd.DataFrame([ic_row])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Report + output writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(fund_df: pd.DataFrame, ic_compare: pd.DataFrame,
                 n_missing: int, ts: str) -> None:
    lines = [
        "# Canyon v9 — Fundamental Features Report (Step 68)",
        f"Generated: {ts}",
        "",
        "## IC Improvement: Price-Only vs Price+Fundamental",
        "",
        "| Model | Mean IC | t-stat | Status |",
        "|---|---|---|---|",
    ]
    for _, row in ic_compare.iterrows():
        lines.append(
            f"| {row['model']} | {float(row.get('mean_ic',0)):+.4f} | "
            f"{float(row.get('t_stat',0)):+.2f} | {row.get('status','—')} |"
        )

    lines += [
        "",
        "## Fundamental Data Coverage",
        "",
        f"- Tickers with fundamental data: {len(fund_df) - n_missing}/{len(fund_df)}",
        f"- Tickers missing data: {n_missing}",
        f"- Data source: yfinance .info (free tier, updated ~daily)",
        "",
        "## Features Added",
        "",
        "| Feature | Description | Edge Rationale |",
        "|---|---|---|",
        "| pe_ratio | Trailing P/E | Value premium: low P/E tends to outperform |",
        "| pb_ratio | Price-to-book | Value factor (Fama-French HML) |",
        "| ps_ratio | Price-to-sales | Low P/S = value in growth sectors |",
        "| roe | Return on equity | Quality factor: high ROE = durable earnings |",
        "| debt_equity | Debt/equity | Low leverage → lower distress risk |",
        "| earnings_growth | YoY EPS growth | Earnings momentum signal |",
        "| revenue_growth | YoY revenue | Top-line growth quality |",
        "| profit_margin | Net margin | Operating efficiency |",
        "| analyst_score | Consensus buy/sell | Analyst revision momentum |",
        "| eps_fwd_yield | Forward EPS/Price | Earnings yield = cheap vs expensive |",
        "",
        "## Caveat",
        "- Fundamentals treated as time-invariant within training window (point-in-time).",
        "  In production, use point-in-time fundamental databases (Compustat, FactSet)",
        "  to avoid look-ahead bias from restated financials.",
        "- ETFs (SPY, QQQ, XLK …) have limited fundamental data; features default to NaN.",
        "- Median imputation used for missing fundamentals during training.",
    ]

    p = ROOT / "fundamental_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 68 — Fundamental Features")
    parser.add_argument("--refresh", action="store_true", help="Force re-download fundamentals")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*62}")
    print("Canyon v9 Step 68 — Fundamental Features + Enhanced ML")
    print(f"{'='*62}")

    # ── 1. Price panel ────────────────────────────────────────────────────────
    print("\n[1/5] Loading prices and building feature panel …")
    from canyon_final_v9_step67_shap_explainer import (
        load_prices, build_feature_panel
    )
    prices = load_prices()
    prices = prices.loc[:, prices.count() >= 504]
    panel  = build_feature_panel(prices)
    panel  = panel[panel["forward_ret"].notna() | True]  # keep all rows
    print(f"      Panel: {panel.shape[0]} rows, {panel['ticker'].nunique()} tickers")

    # ── 2. Fundamentals ───────────────────────────────────────────────────────
    print("\n[2/5] Fetching fundamental data …")
    stock_tickers = [t for t in UNIVERSE if t not in ["SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP"]]
    fund_df = load_fundamentals(stock_tickers, force_refresh=args.refresh)
    if fund_df.empty:
        print("  WARNING: No fundamental data available — running price-only model")
        fund_df_all = pd.DataFrame()
    else:
        fund_df_all = load_fundamentals(UNIVERSE, force_refresh=False)
        n_missing = int(fund_df_all[FEATURES_FUNDAMENTAL].isna().all(axis=1).sum()) if not fund_df_all.empty else 0
        print(f"      {len(fund_df_all)} tickers loaded, {n_missing} with no fundamental data")

        # Coverage summary
        for col in FEATURES_FUNDAMENTAL[:5]:
            if col in fund_df_all.columns:
                n_ok = int(fund_df_all[col].notna().sum())
                print(f"      {col:18s}: {n_ok}/{len(fund_df_all)} tickers")

    # ── 3. Merge + enhanced panel ─────────────────────────────────────────────
    print("\n[3/5] Merging fundamentals into feature panel …")
    panel_enhanced = merge_fundamentals(panel, fund_df_all) if not fund_df_all.empty else panel.copy()
    # Use ranked versions of fundamentals to reduce outlier impact
    enhanced_features = FEATURES_PRICE + [
        f"r_{c}" for c in FEATURES_FUNDAMENTAL
        if f"r_{c}" in panel_enhanced.columns
    ]
    print(f"      Enhanced feature count: {len(enhanced_features)} "
          f"(+{len(enhanced_features) - len(FEATURES_PRICE)} fundamental)")

    # ── 4. Walk-forward IC comparison ─────────────────────────────────────────
    print("\n[4/5] Walk-forward IC comparison …")
    # Price-only
    print("  Running price-only model …")
    _, ic_price = walk_forward_enhanced(panel, FEATURES_PRICE, args.lookback, "price_only")
    # Enhanced
    print("  Running price+fundamental model …")
    preds_enh, ic_enh = walk_forward_enhanced(
        panel_enhanced, enhanced_features, args.lookback, "enhanced"
    )

    # Compare
    ic_compare = pd.concat([ic_price, ic_enh], ignore_index=True) if not ic_price.empty else ic_enh

    print("\n  IC Comparison:")
    if not ic_compare.empty:
        for _, row in ic_compare.iterrows():
            print(f"    {str(row.get('model','—')):15s}  IC={float(row.get('mean_ic',0)):+.4f}  "
                  f"t={float(row.get('t_stat',0)):+.2f}  {row.get('status','—')}")

    # ── 5. Write outputs ──────────────────────────────────────────────────────
    print("\n[5/5] Writing outputs …")
    n_missing = 0
    if not fund_df_all.empty:
        fund_df_all.to_csv(ROOT / "fundamental_features.csv", index=False)
        print(f"  [fundamental_features.csv]    OK")
        n_missing = int(fund_df_all[FEATURES_FUNDAMENTAL].isna().all(axis=1).sum()) if FEATURES_FUNDAMENTAL[0] in fund_df_all.columns else 0

    if not preds_enh.empty:
        preds_enh.to_csv(ROOT / "enhanced_ml_scores.csv", index=False)
        print(f"  [enhanced_ml_scores.csv]      OK")

    if not ic_compare.empty:
        ic_compare.to_csv(ROOT / "fundamental_ic_comparison.csv", index=False)
        print(f"  [fundamental_ic_comparison.csv] OK")

    write_report(fund_df_all if not fund_df_all.empty else pd.DataFrame(columns=["ticker"]),
                 ic_compare, n_missing, ts)

    # Summary
    print(f"\n{'─'*62}")
    print("FUNDAMENTAL FEATURES RESULTS")
    print(f"{'─'*62}")
    if not ic_compare.empty:
        price_ic = ic_compare[ic_compare["model"] == "price_only"]["mean_ic"].values
        enh_ic   = ic_compare[ic_compare["model"] == "enhanced"]["mean_ic"].values
        if len(price_ic) and len(enh_ic):
            improvement = float(enh_ic[0]) - float(price_ic[0])
            print(f"  Price-only IC:   {float(price_ic[0]):+.4f}")
            print(f"  Enhanced IC:     {float(enh_ic[0]):+.4f}")
            print(f"  Improvement:     {improvement:+.4f}  "
                  f"({'fundamental adds edge' if improvement > 0.005 else 'marginal/no improvement'})")
    print(f"\n  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
