"""
Canyon v9  Step 67 — SHAP Explainability Engine
=================================================
Answers: "Why did the ML model rank this ticker first?"

Uses TreeExplainer (exact SHAP for Random Forest) and LinearExplainer
(Ridge) to attribute each prediction to individual features.

Outputs:
  shap_values_rf.csv       N_tickers × N_features SHAP matrix (latest rebalance)
  shap_values_ridge.csv    Ridge SHAP matrix
  shap_summary.csv         Mean |SHAP| per feature across all tickers (global)
  shap_per_ticker.csv      Per-ticker: top driving features + direction
  shap_report.md           Full markdown report

Usage:
  python canyon_final_v9_step67_shap_explainer.py
  python canyon_final_v9_step67_shap_explainer.py --date 2026-04-01
  python canyon_final_v9_step67_shap_explainer.py --ticker NVDA
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    "trend_200", "rsi_14", "inv_vol",
    "rank_mom", "rank_trend", "spy_regime",
]
LOOKBACK_DAYS = 252
WARMUP_DAYS   = 504


# ─────────────────────────────────────────────────────────────────────────────
# Re-use feature building from Step 66
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def load_prices() -> pd.DataFrame:
    cache_path = ROOT / "backtest_price_cache.csv"
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df.dropna(how="all")
    try:
        import yfinance as yf
        from canyon_final_v9_step66_ml_signals import UNIVERSE
        raw = yf.download(UNIVERSE, period="5y", auto_adjust=True)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all")
        prices.to_csv(cache_path)
        return prices
    except Exception as e:
        raise RuntimeError(f"Cannot load prices: {e}")


def build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Rebuild feature panel (same logic as Step 66)."""
    all_recs = []
    spy = prices.get("SPY")
    for ticker in prices.columns:
        if ticker == "SPY":
            continue
        s = prices[ticker].dropna()
        if len(s) < WARMUP_DAYS:
            continue
        log_r = np.log(s / s.shift(1))
        mom_252 = s.pct_change(252).shift(1)
        mom_21  = s.pct_change(21).shift(1)
        feat = pd.DataFrame({
            "ticker":         ticker,
            "mom_1m":         s.pct_change(21).shift(1),
            "mom_3m":         s.pct_change(63).shift(1),
            "mom_6m":         s.pct_change(126).shift(1),
            "mom_12m_skip1m": mom_252 - mom_21,
            "trend_200":      (s / s.rolling(200).mean() - 1).shift(1),
            "rsi_14":         _rsi(s, 14).shift(1),
            "inv_vol":        (1 / (log_r.rolling(21).std().shift(1) + 1e-10)),
            "spy_regime":     ((spy > spy.rolling(200).mean()).astype(float).shift(1).reindex(s.index).fillna(1)
                               if spy is not None else pd.Series(1.0, index=s.index)),
            "forward_ret":    np.log(s.shift(-21) / s),
        }, index=s.index)
        feat.index.name = "date"
        all_recs.append(feat)
    if not all_recs:
        return pd.DataFrame()
    panel = pd.concat(all_recs).reset_index()
    panel = panel.dropna(subset=FEATURES[:7])
    panel["rank_mom"]   = panel.groupby("date")["mom_12m_skip1m"].rank(pct=True)
    panel["rank_trend"] = panel.groupby("date")["trend_200"].rank(pct=True)
    return panel


def train_models(panel: pd.DataFrame, lookback: int) -> tuple:
    """Train RF + Ridge on full lookback window. Returns (rf, ridge, scaler, X_cols)."""
    train = panel.dropna(subset=FEATURES + ["forward_ret"]).copy()
    train = train.tail(lookback * len(panel["ticker"].unique()))
    train["y_ranked"] = train.groupby("date")["forward_ret"].rank(pct=True)

    X = train[FEATURES].values
    y = train["y_ranked"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_sc, y)

    ridge = Ridge(alpha=50.0)
    ridge.fit(X_sc, y)

    return rf, ridge, scaler


def compute_shap(
    panel: pd.DataFrame,
    rf, ridge, scaler,
    target_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute SHAP values for all tickers at `target_date` (or latest date).
    Returns dict with keys 'rf', 'ridge', 'summary', 'per_ticker'.
    """
    if target_date is None:
        target_date = str(panel["date"].max())[:10]

    pred_mask = panel["date"].astype(str).str[:10] == target_date[:10]
    pred_df   = panel[pred_mask].dropna(subset=FEATURES).copy()
    if pred_df.empty:
        # Fall back to latest available date
        all_dates = sorted(panel["date"].astype(str).str[:10].unique())
        for d in reversed(all_dates):
            pred_mask = panel["date"].astype(str).str[:10] == d
            pred_df   = panel[pred_mask].dropna(subset=FEATURES)
            if not pred_df.empty:
                target_date = d
                break
    if pred_df.empty:
        return {}

    X_pred  = pred_df[FEATURES].values
    X_sc    = scaler.transform(X_pred)
    tickers = pred_df["ticker"].tolist()

    # ── RF SHAP (TreeExplainer) ───────────────────────────────────────────────
    print(f"  [SHAP] TreeExplainer for RF ({len(tickers)} tickers) …")
    try:
        # Build background from training data
        bg_mask  = panel["date"].astype(str).str[:10] < target_date[:10]
        bg_panel = panel[bg_mask].dropna(subset=FEATURES).tail(5000)
        bg_X     = scaler.transform(bg_panel[FEATURES].values)
        bg_summary = shap.kmeans(bg_X, 50)   # 50-point summary background

        explainer_rf = shap.TreeExplainer(rf, bg_summary)
        shap_rf_raw  = explainer_rf.shap_values(X_sc)
        if isinstance(shap_rf_raw, list):
            shap_rf_raw = shap_rf_raw[0]
        rf_df = pd.DataFrame(shap_rf_raw, columns=FEATURES)
        rf_df.insert(0, "ticker", tickers)
        rf_df.insert(1, "date",   target_date)
    except Exception as e:
        print(f"    [SHAP] RF error: {e} — using permutation approximation")
        # Fallback: scale feature * weight approximation
        coefs = np.array(rf.feature_importances_)
        shap_approx = (X_sc - X_sc.mean(axis=0)) * coefs
        rf_df = pd.DataFrame(shap_approx, columns=FEATURES)
        rf_df.insert(0, "ticker", tickers)
        rf_df.insert(1, "date",   target_date)

    # ── Ridge SHAP (LinearExplainer) ─────────────────────────────────────────
    print(f"  [SHAP] LinearExplainer for Ridge …")
    try:
        explainer_ridge = shap.LinearExplainer(ridge, X_sc, feature_perturbation="correlation_dependent")
        shap_ridge_raw  = explainer_ridge.shap_values(X_sc)
        ridge_df = pd.DataFrame(shap_ridge_raw, columns=FEATURES)
        ridge_df.insert(0, "ticker", tickers)
        ridge_df.insert(1, "date",   target_date)
    except Exception as e:
        print(f"    [SHAP] Ridge error: {e} — using coefficient × feature")
        shap_ridge = X_sc * ridge.coef_
        ridge_df = pd.DataFrame(shap_ridge, columns=FEATURES)
        ridge_df.insert(0, "ticker", tickers)
        ridge_df.insert(1, "date",   target_date)

    # ── Global summary: mean |SHAP| per feature ───────────────────────────────
    summary_rows = []
    for feat in FEATURES:
        rf_mean_abs    = float(rf_df[feat].abs().mean()) if feat in rf_df else 0.0
        ridge_mean_abs = float(ridge_df[feat].abs().mean()) if feat in ridge_df else 0.0
        summary_rows.append({
            "feature":          feat,
            "rf_mean_abs_shap": round(rf_mean_abs, 5),
            "ridge_mean_abs_shap": round(ridge_mean_abs, 5),
            "avg_abs_shap":     round((rf_mean_abs + ridge_mean_abs) / 2, 5),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("avg_abs_shap", ascending=False).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)

    # ── Per-ticker explanation ────────────────────────────────────────────────
    per_ticker_rows = []
    for i, ticker in enumerate(tickers):
        rf_shaps = {f: float(rf_df.loc[rf_df["ticker"] == ticker, f].iloc[0])
                    for f in FEATURES if ticker in rf_df["ticker"].values}
        if not rf_shaps:
            continue
        # Top positive and negative drivers
        sorted_shaps = sorted(rf_shaps.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_shaps[:3]
        top_pos = [f"{f}={v:+.4f}" for f, v in sorted_shaps if v > 0][:2]
        top_neg = [f"{f}={v:+.4f}" for f, v in sorted_shaps if v < 0][:2]

        rf_score_idx = pred_df[pred_df["ticker"] == ticker].index
        raw_rf_pred  = rf.predict(scaler.transform(
            pred_df.loc[rf_score_idx, FEATURES].values
        )).item() if len(rf_score_idx) else 0.0

        per_ticker_rows.append({
            "date":           target_date,
            "ticker":         ticker,
            "rf_pred_score":  round(float(raw_rf_pred), 4),
            "top_driver_1":   top3[0][0] if len(top3) > 0 else "—",
            "shap_1":         round(top3[0][1], 4) if len(top3) > 0 else 0.0,
            "top_driver_2":   top3[1][0] if len(top3) > 1 else "—",
            "shap_2":         round(top3[1][1], 4) if len(top3) > 1 else 0.0,
            "top_driver_3":   top3[2][0] if len(top3) > 2 else "—",
            "shap_3":         round(top3[2][1], 4) if len(top3) > 2 else 0.0,
            "positive_drivers": " | ".join(top_pos),
            "negative_drivers": " | ".join(top_neg),
        })

    per_ticker_df = pd.DataFrame(per_ticker_rows).sort_values("rf_pred_score", ascending=False).reset_index(drop=True)

    return {
        "rf":         rf_df,
        "ridge":      ridge_df,
        "summary":    summary_df,
        "per_ticker": per_ticker_df,
        "date":       target_date,
        "tickers":    tickers,
    }


def write_report(results: dict, ts: str) -> None:
    summary  = results.get("summary", pd.DataFrame())
    per_tick = results.get("per_ticker", pd.DataFrame())
    date     = results.get("date", "—")

    lines = [
        "# Canyon v9 — SHAP Explainability Report (Step 67)",
        f"Generated: {ts}  |  Rebalance date: {date}",
        "",
        "## Global Feature Impact (Mean |SHAP|)",
        "",
        "| Rank | Feature | RF |SHAP| | Ridge |SHAP| | Avg |",
        "|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        bar = "█" * max(1, int(float(row.get("avg_abs_shap", 0)) * 1000))
        lines.append(
            f"| {int(row.get('rank',0))} | {row['feature']} | "
            f"{float(row.get('rf_mean_abs_shap',0)):.5f} | "
            f"{float(row.get('ridge_mean_abs_shap',0)):.5f} | "
            f"{float(row.get('avg_abs_shap',0)):.5f} {bar} |"
        )

    if not per_tick.empty:
        lines += [
            "",
            "## Per-Ticker Explanations (Top 10 by Score)",
            "",
            "| Ticker | Score | Driver 1 | SHAP 1 | Driver 2 | SHAP 2 | Driver 3 | SHAP 3 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in per_tick.head(10).iterrows():
            lines.append(
                f"| {row['ticker']} | {float(row.get('rf_pred_score',0)):+.4f} | "
                f"{row.get('top_driver_1','—')} | {float(row.get('shap_1',0)):+.4f} | "
                f"{row.get('top_driver_2','—')} | {float(row.get('shap_2',0)):+.4f} | "
                f"{row.get('top_driver_3','—')} | {float(row.get('shap_3',0)):+.4f} |"
            )

    lines += [
        "",
        "## How to Read SHAP Values",
        "- **Positive SHAP** = feature pushes predicted score **up** (bullish contribution)",
        "- **Negative SHAP** = feature pushes predicted score **down** (bearish drag)",
        "- **|SHAP|** = magnitude of feature impact regardless of direction",
        "- Sum of all SHAP values ≈ model prediction − baseline",
        "",
        "## Example Interpretation",
        "If NVDA has mom_12m_skip1m SHAP = +0.15, it means:",
        "'NVDA's 12-month momentum is strong enough to push its cross-sectional rank",
        "up by approximately 0.15 (on a 0–1 scale) above what a no-information baseline would predict.'",
    ]

    p = ROOT / "shap_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 67 — SHAP Explainability")
    parser.add_argument("--date",   default=None, help="Target rebalance date YYYY-MM-DD")
    parser.add_argument("--ticker", default=None, help="Print SHAP waterfall for specific ticker")
    args = parser.parse_args()

    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print("Canyon v9 Step 67 — SHAP Explainability Engine")
    print(f"{'='*60}")

    print("\n[1/4] Loading prices and building features …")
    prices = load_prices()
    prices = prices.loc[:, prices.count() >= WARMUP_DAYS]
    panel  = build_feature_panel(prices)
    print(f"      Panel: {panel.shape[0]} rows, {panel['ticker'].nunique()} tickers")

    print("\n[2/4] Training RF + Ridge (full dataset) …")
    rf, ridge, scaler = train_models(panel, lookback=LOOKBACK_DAYS)
    print("      Models trained.")

    print("\n[3/4] Computing SHAP values …")
    results = compute_shap(panel, rf, ridge, scaler, target_date=args.date)
    if not results:
        print("  ERROR: No data for requested date.")
        return

    date  = results["date"]
    print(f"      SHAP computed for {len(results['tickers'])} tickers on {date}")

    # Print global summary
    print(f"\n  Global feature importance (mean |SHAP|):")
    for _, row in results["summary"].iterrows():
        bar = "█" * max(1, int(float(row.get("avg_abs_shap", 0)) * 2000))
        print(f"    {row['feature']:20s}  {float(row.get('avg_abs_shap',0)):.5f}  {bar}")

    # Print top tickers
    per_tick = results.get("per_ticker", pd.DataFrame())
    if not per_tick.empty:
        print(f"\n  Top 5 tickers with drivers:")
        for _, row in per_tick.head(5).iterrows():
            print(f"    {row['ticker']:8s}  score={float(row.get('rf_pred_score',0)):+.4f}  "
                  f"↑{row.get('positive_drivers','—')[:35]}  "
                  f"↓{row.get('negative_drivers','—')[:35]}")

    # Specific ticker waterfall
    if args.ticker and not per_tick.empty:
        tk_row = per_tick[per_tick["ticker"] == args.ticker.upper()]
        if not tk_row.empty:
            r = tk_row.iloc[0]
            print(f"\n  SHAP Waterfall: {args.ticker.upper()} ({date})")
            rf_row = results["rf"][results["rf"]["ticker"] == args.ticker.upper()]
            if not rf_row.empty:
                vals = {f: float(rf_row[f].iloc[0]) for f in FEATURES}
                sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)
                print(f"  {'Feature':22s}  {'SHAP':>8s}  {'Direction':10s}")
                print(f"  {'-'*44}")
                for feat, val in sorted_vals:
                    direction = "▲ bullish" if val > 0 else "▼ bearish"
                    bar_len = max(1, min(20, int(abs(val) * 200)))
                    bar_col = "▌" if val > 0 else "░"
                    bar = bar_col * bar_len
                    print(f"  {feat:22s}  {val:+8.4f}  {bar:<20s}  {direction}")

    print("\n[4/4] Writing outputs …")
    results["rf"].to_csv(ROOT / "shap_values_rf.csv", index=False)
    results["ridge"].to_csv(ROOT / "shap_values_ridge.csv", index=False)
    results["summary"].to_csv(ROOT / "shap_summary.csv", index=False)
    results["per_ticker"].to_csv(ROOT / "shap_per_ticker.csv", index=False)
    write_report(results, ts)
    print(f"  [shap_values_rf.csv]     OK")
    print(f"  [shap_values_ridge.csv]  OK")
    print(f"  [shap_summary.csv]       OK")
    print(f"  [shap_per_ticker.csv]    OK")

    print(f"\n{'='*60}")
    print(f"SHAP complete — {len(results['tickers'])} tickers explained in {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
