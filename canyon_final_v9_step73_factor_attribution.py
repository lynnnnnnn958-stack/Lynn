"""
Canyon v9  Step 73 — Barra-Style Factor Attribution
=====================================================
Decomposes Canyon v9 portfolio returns and risk into 5 systematic factors:

  1. Market   — SPY daily returns (beta exposure)
  2. Momentum — Equal-weight top-quintile 12m-momentum stocks
  3. Low-Vol  — Equal-weight bottom-quintile 21d-volatility stocks
  4. Value    — 1/PE rank (from fundamental_features.csv)
  5. Quality  — ROE cross-sectional rank

Attribution outputs:
  factor_attribution.csv      — factor loadings, return & variance contributions
  factor_marginal_risk.csv    — per-position marginal risk contribution (MRC)
  factor_report.md            — markdown summary

Usage:
  python canyon_final_v9_step73_factor_attribution.py
  python canyon_final_v9_step73_factor_attribution.py --refresh   # force re-download
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── file paths ────────────────────────────────────────────────────────────────
ML_SCORES_CSV       = ROOT / "ml_signal_scores.csv"
FUNDAMENTAL_CSV     = ROOT / "fundamental_features.csv"
POSITIONS_CSV       = ROOT / "paper_sim_positions.csv"
FACTOR_CACHE_CSV    = ROOT / "factor_cache.csv"
ATTRIBUTION_CSV     = ROOT / "factor_attribution.csv"
MRC_CSV             = ROOT / "factor_marginal_risk.csv"
REPORT_MD           = ROOT / "factor_report.md"

LOOKBACK_DAYS   = 252
CACHE_TTL_H     = 24
ANN_FACTOR      = 252
FACTOR_NAMES    = ["Market", "Momentum", "Low-Vol", "Value", "Quality"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Universe loader
# ─────────────────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    """Get unique tickers from ml_signal_scores.csv, always include SPY."""
    if not ML_SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing {ML_SCORES_CSV} — run step66 first.")
    df = pd.read_csv(ML_SCORES_CSV)
    tickers = df["ticker"].unique().tolist()
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
# 2. Price fetcher with 24h disk cache
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Return adjusted-close price DataFrame for `tickers`, covering `days` trading days.
    Uses FACTOR_CACHE_CSV with 24h TTL.
    """
    # Try cache first
    if FACTOR_CACHE_CSV.exists():
        age_h = (time.time() - FACTOR_CACHE_CSV.stat().st_mtime) / 3600
        if age_h < CACHE_TTL_H:
            try:
                cached = pd.read_csv(FACTOR_CACHE_CSV, index_col=0, parse_dates=True)
                available = [t for t in tickers if t in cached.columns]
                if len(available) >= max(3, len(tickers) * 0.7):
                    print(f"  [cache] {len(available)}/{len(tickers)} tickers, {age_h:.1f}h old")
                    # Trim to requested window
                    return cached[available].iloc[-days:].dropna(how="all")
            except Exception as exc:
                print(f"  [cache] Read error ({exc}), re-downloading …")

    # Download fresh
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    end_dt   = datetime.today()
    # Over-request by 40% to ensure we have enough trading days after weekends/holidays
    cal_days = int(days * 1.4) + 30
    start_dt = end_dt - timedelta(days=cal_days)

    print(f"  [yfinance] Downloading {len(tickers)} tickers ({days} trading days) …")
    try:
        raw = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    prices = prices.dropna(how="all")
    # Persist cache
    prices.to_csv(FACTOR_CACHE_CSV)
    print(f"  [yfinance] {prices.shape[1]} tickers, {len(prices)} rows saved")

    available = [t for t in tickers if t in prices.columns]
    return prices[available].iloc[-days:].dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Factor return builder
# ─────────────────────────────────────────────────────────────────────────────

def build_factor_returns(prices: pd.DataFrame, fund_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Construct daily factor return series for the 5 Barra factors.

    Returns a DataFrame indexed by date with columns:
      Market, Momentum, Low-Vol, Value, Quality
    """
    # Daily log returns (drop first row which is NaN)
    rets = np.log(prices / prices.shift(1)).iloc[1:]

    universe_tickers = [c for c in rets.columns if c != "SPY"]
    uni_rets = rets[universe_tickers]

    # ── Factor 1: Market ──────────────────────────────────────────────────────
    if "SPY" not in rets.columns:
        raise KeyError("SPY not in price data — needed for Market factor.")
    f_market = rets["SPY"].rename("Market")

    # ── Factor 2: Momentum ───────────────────────────────────────────────────
    # 12m cumulative return ending at each date (computed on full price series)
    # Use only universe tickers (excluding SPY)
    # For simplicity, compute over the full available price history
    roll_12m = prices[universe_tickers].pct_change(252).shift(1)   # avoid look-ahead
    # On each date, identify top-quintile momentum stocks; equal-weight their next-day return
    def _quintile_ew_return(rets_row: pd.Series, rank_row: pd.Series) -> float:
        """Equal-weight return of top-quintile stocks by rank_row."""
        valid = rank_row.dropna()
        if len(valid) < 5:
            return np.nan
        threshold = valid.quantile(0.80)   # top 20% = top quintile
        top = valid[valid >= threshold].index
        top_rets = rets_row[top].dropna()
        return float(top_rets.mean()) if len(top_rets) > 0 else np.nan

    # Vectorise: align roll_12m to rets index
    roll_aligned = roll_12m.reindex(rets.index)
    mom_series = pd.Series(index=rets.index, dtype=float, name="Momentum")
    for dt in rets.index:
        mom_series.loc[dt] = _quintile_ew_return(uni_rets.loc[dt], roll_aligned.loc[dt])

    # ── Factor 3: Low-Vol ────────────────────────────────────────────────────
    # 21-day rolling vol; bottom-quintile (lowest vol) stocks
    roll_21vol = uni_rets.rolling(21).std().shift(1)
    roll_vol_aligned = roll_21vol.reindex(rets.index)

    lvol_series = pd.Series(index=rets.index, dtype=float, name="Low-Vol")
    for dt in rets.index:
        vol_row = roll_vol_aligned.loc[dt].dropna()
        if len(vol_row) < 5:
            lvol_series.loc[dt] = np.nan
            continue
        threshold = vol_row.quantile(0.20)   # bottom 20% = lowest vol
        low_vol = vol_row[vol_row <= threshold].index
        low_rets = uni_rets.loc[dt, low_vol].dropna()
        lvol_series.loc[dt] = float(low_rets.mean()) if len(low_rets) > 0 else np.nan

    # ── Factor 4: Value (1/PE rank) ──────────────────────────────────────────
    val_series = pd.Series(index=rets.index, dtype=float, name="Value")
    if fund_df is not None and "pe_ratio" in fund_df.columns:
        # Build static cross-sectional rank once (fundamental data is point-in-time snapshot)
        fund_clean = fund_df.set_index("ticker") if "ticker" in fund_df.columns else fund_df
        pe_col = "inv_pe" if "inv_pe" in fund_clean.columns else "pe_ratio"
        pe_vals = fund_clean[pe_col].dropna()
        if pe_col == "pe_ratio":
            # Convert P/E → 1/PE (higher = cheaper = more valuable)
            pe_vals = 1.0 / pe_vals.replace(0, np.nan)
        pe_rank = pe_vals.rank(pct=True)   # cross-sectional percentile rank

        # Long: top-quintile Value stocks (highest 1/PE = cheapest)
        universe_in_fund = [t for t in universe_tickers if t in pe_rank.index]
        if len(universe_in_fund) >= 5:
            top_val_thresh = pe_rank[universe_in_fund].quantile(0.80)
            top_val_tickers = pe_rank[universe_in_fund][
                pe_rank[universe_in_fund] >= top_val_thresh
            ].index.tolist()
            # Return series: equal-weight top-value tickers
            val_cols = [t for t in top_val_tickers if t in uni_rets.columns]
            if val_cols:
                val_series = uni_rets[val_cols].mean(axis=1).rename("Value")
            else:
                val_series = pd.Series(0.0, index=rets.index, name="Value")
        else:
            print("  [Value factor] Insufficient PE data — filling with zeros")
            val_series = pd.Series(0.0, index=rets.index, name="Value")
    else:
        print("  [Value factor] No PE data in fundamental_features.csv — filling with zeros")
        val_series = pd.Series(0.0, index=rets.index, name="Value")

    # ── Factor 5: Quality (ROE rank) ─────────────────────────────────────────
    qual_series = pd.Series(index=rets.index, dtype=float, name="Quality")
    if fund_df is not None and "roe" in fund_df.columns:
        fund_clean = fund_df.set_index("ticker") if "ticker" in fund_df.columns else fund_df
        roe_vals = fund_clean["roe"].dropna()
        roe_rank = roe_vals.rank(pct=True)

        universe_in_fund = [t for t in universe_tickers if t in roe_rank.index]
        if len(universe_in_fund) >= 5:
            top_qual_thresh = roe_rank[universe_in_fund].quantile(0.80)
            top_qual_tickers = roe_rank[universe_in_fund][
                roe_rank[universe_in_fund] >= top_qual_thresh
            ].index.tolist()
            qual_cols = [t for t in top_qual_tickers if t in uni_rets.columns]
            if qual_cols:
                qual_series = uni_rets[qual_cols].mean(axis=1).rename("Quality")
            else:
                qual_series = pd.Series(0.0, index=rets.index, name="Quality")
        else:
            print("  [Quality factor] Insufficient ROE data — filling with zeros")
            qual_series = pd.Series(0.0, index=rets.index, name="Quality")
    else:
        print("  [Quality factor] No ROE data in fundamental_features.csv — filling with zeros")
        qual_series = pd.Series(0.0, index=rets.index, name="Quality")

    # ── Assemble factor DataFrame ─────────────────────────────────────────────
    factor_df = pd.concat(
        [f_market, mom_series, lvol_series, val_series, qual_series], axis=1
    )
    factor_df.columns = FACTOR_NAMES
    factor_df = factor_df.dropna(how="all")
    return factor_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Portfolio return builder
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_returns(
    prices: pd.DataFrame,
    universe_tickers: list[str],
) -> tuple[pd.Series, dict[str, float]]:
    """
    Build daily portfolio return series.
    If paper_sim_positions.csv has weights → use those.
    Else → equal-weight all universe tickers.

    Returns (port_returns, weights_dict)
    """
    # Universe tickers only (exclude SPY from portfolio construction)
    port_tickers = [t for t in universe_tickers if t in prices.columns]

    weights: dict[str, float] = {}

    if POSITIONS_CSV.exists():
        try:
            pos_df = pd.read_csv(POSITIONS_CSV)
            if "ticker" in pos_df.columns and "cost_basis" in pos_df.columns:
                pos_df = pos_df[pos_df["ticker"].isin(port_tickers)].copy()
                if len(pos_df) > 0:
                    total_cb = pos_df["cost_basis"].sum()
                    if total_cb > 0:
                        pos_df["weight"] = pos_df["cost_basis"] / total_cb
                        weights = dict(zip(pos_df["ticker"], pos_df["weight"]))
                        print(f"  [portfolio] Using {len(weights)} positions from paper_sim_positions.csv")
        except Exception as exc:
            print(f"  [portfolio] Could not load positions ({exc}), falling back to equal-weight")

    if not weights:
        n = len(port_tickers)
        weights = {t: 1.0 / n for t in port_tickers}
        print(f"  [portfolio] Equal-weight, {n} tickers")

    # Compute daily returns for each held ticker
    all_tickers_in_weights = [t for t in weights if t in prices.columns]
    if not all_tickers_in_weights:
        raise RuntimeError("No portfolio tickers found in price data.")

    rets = np.log(prices / prices.shift(1)).iloc[1:]
    port_rets_components = []
    for tkr in all_tickers_in_weights:
        w = weights[tkr]
        port_rets_components.append(rets[tkr] * w)

    port_series = pd.concat(port_rets_components, axis=1).sum(axis=1)
    port_series.name = "portfolio"
    return port_series, weights


# ─────────────────────────────────────────────────────────────────────────────
# 5. OLS regression
# ─────────────────────────────────────────────────────────────────────────────

def run_ols(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    OLS: y = alpha + X @ betas + residuals
    Uses np.linalg.lstsq.

    Returns (betas, alpha, r2, residuals)
      betas  : shape (n_factors,)
      alpha  : intercept (scalar)
      r2     : coefficient of determination
      residuals : shape (n_obs,)
    """
    # Add intercept column
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])  # shape (n, 1+n_factors)

    coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    alpha  = float(coeffs[0])
    betas  = coeffs[1:]

    y_hat     = X_aug @ coeffs
    residuals = y - y_hat
    ss_res    = float(np.sum(residuals**2))
    ss_tot    = float(np.sum((y - np.mean(y))**2))
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return betas, alpha, r2, residuals


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decompose return and variance contributions
# ─────────────────────────────────────────────────────────────────────────────

def decompose_contributions(
    betas: np.ndarray,
    factor_rets: pd.DataFrame,
    factor_cov: np.ndarray,
    alpha: float,
    r2: float,
    residuals: np.ndarray,
) -> dict:
    """
    Compute per-factor:
      - annualised return contribution = beta_i * mean(F_i) * 252
      - variance contribution          = beta_i² * var(F_i) * 252

    Also computes idiosyncratic variance = var(residuals) * 252.

    Returns a dict with keys matching FACTOR_NAMES plus 'Alpha'.
    """
    means = factor_rets.mean().values          # shape (n_factors,)
    vols  = factor_rets.var().values           # shape (n_factors,)

    return_contribs  = betas * means * ANN_FACTOR           # annualised
    variance_contribs = (betas**2) * vols * ANN_FACTOR       # annualised variance

    total_factor_var  = float(betas @ factor_cov @ betas) * ANN_FACTOR
    idio_var          = float(np.var(residuals, ddof=1)) * ANN_FACTOR
    total_var         = total_factor_var + idio_var
    if total_var < 1e-12:
        total_var = 1e-12

    # Percent of total variance
    variance_pct = variance_contribs / total_var * 100
    idio_var_pct = idio_var / total_var * 100

    result = {
        "betas":            betas,
        "return_contribs":  return_contribs,
        "variance_contribs": variance_contribs,
        "variance_pct":     variance_pct,
        "alpha_annual":     alpha * ANN_FACTOR,
        "idio_var":         idio_var,
        "idio_var_pct":     idio_var_pct,
        "total_var":        total_var,
        "r2":               r2,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Marginal Risk Contribution per position
# ─────────────────────────────────────────────────────────────────────────────

def compute_marginal_risk(
    weights: dict[str, float],
    tickers_in_prices: list[str],
    betas_per_stock: np.ndarray | None,
    factor_cov: np.ndarray,
    port_vol: float,
) -> pd.DataFrame:
    """
    MRC_i = w_i * (Σ_factor @ w)_i / σ_portfolio

    where Σ_factor is the factor covariance matrix mapped to stock-space
    via stock betas (each stock proxied by portfolio-level betas for simplicity).

    Returns DataFrame with columns: ticker, weight, mrc, mrc_pct
    """
    if port_vol < 1e-8:
        return pd.DataFrame(columns=["ticker", "weight", "mrc", "mrc_pct"])

    tickers = [t for t in weights if t in tickers_in_prices]
    w_arr   = np.array([weights[t] for t in tickers])

    # Map weights to factor space: Bw = B @ w  (B is n_factors x n_stocks)
    # Since we don't have per-stock betas, we approximate each stock's
    # factor loading as proportional to the portfolio-level betas scaled by
    # beta_sum — standard simplified Barra approximation.
    n_f = factor_cov.shape[0]
    n_s = len(tickers)

    if betas_per_stock is not None and betas_per_stock.shape == (n_s, n_f):
        B = betas_per_stock.T    # shape (n_f, n_s)
    else:
        # Fallback: replicate portfolio-level betas for every stock
        # This is the "portfolio-as-single-asset" proxy
        if betas_per_stock is not None and len(betas_per_stock) == n_f:
            beta_col = betas_per_stock.reshape(-1, 1)
        else:
            beta_col = np.ones((n_f, 1))
        B = np.tile(beta_col, (1, n_s))   # shape (n_f, n_s)

    # Portfolio variance in factor space: w^T B^T Σ B w
    Bw    = B @ w_arr                    # shape (n_f,)
    SBw   = factor_cov @ Bw              # shape (n_f,)
    mrc_f = B.T @ SBw                    # shape (n_s,) — factor component of MRC

    mrc = w_arr * mrc_f / port_vol       # scaled by 1/σ_p

    mrc_df = pd.DataFrame({
        "ticker":  tickers,
        "weight":  w_arr,
        "mrc":     mrc,
    })
    total_mrc = mrc_df["mrc"].abs().sum()
    mrc_df["mrc_pct"] = (mrc_df["mrc"] / total_mrc * 100) if total_mrc > 1e-12 else 0.0
    mrc_df = mrc_df.sort_values("mrc_pct", ascending=False).reset_index(drop=True)
    return mrc_df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pretty-print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(attr: dict, factor_names: list[str]) -> None:
    width = 62
    sep   = "─" * width

    print(f"\n  FACTOR ATTRIBUTION — last {LOOKBACK_DAYS} days")
    print(f"  {sep}")
    print(f"  {'Factor':<12} {'Beta':>7}  {'Return Contrib':>15}  {'Risk Contrib':>13}")
    print(f"  {sep}")

    for i, fname in enumerate(factor_names):
        beta  = attr["betas"][i]
        rc    = attr["return_contribs"][i]
        vpct  = attr["variance_pct"][i]
        sign  = "+" if rc >= 0 else ""
        print(
            f"  {fname:<12} {beta:>7.3f}  {sign}{rc*100:>13.1f}%  {vpct:>11.1f}%"
        )

    # Alpha row
    alp  = attr["alpha_annual"]
    sign = "+" if alp >= 0 else ""
    idio = attr["idio_var_pct"]
    print(f"  {'Alpha (α)':<12} {'':>7}  {sign}{alp*100:>13.1f}%  {idio:>11.1f}%")

    print(f"  {sep}")
    print(f"  R²  =  {attr['r2']:.4f}")
    print(f"  {sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_attribution_csv(attr: dict, factor_names: list[str]) -> None:
    rows = []
    for i, fname in enumerate(factor_names):
        rows.append({
            "factor":             fname,
            "beta":               float(attr["betas"][i]),
            "return_contrib_ann": float(attr["return_contribs"][i]),
            "variance_contrib_ann": float(attr["variance_contribs"][i]),
            "variance_pct":       float(attr["variance_pct"][i]),
        })
    # Alpha row
    rows.append({
        "factor":             "Alpha",
        "beta":               np.nan,
        "return_contrib_ann": float(attr["alpha_annual"]),
        "variance_contrib_ann": float(attr["idio_var"]),
        "variance_pct":       float(attr["idio_var_pct"]),
    })
    # Summary row
    rows.append({
        "factor":             "R2",
        "beta":               np.nan,
        "return_contrib_ann": np.nan,
        "variance_contrib_ann": np.nan,
        "variance_pct":       float(attr["r2"]),
    })
    pd.DataFrame(rows).to_csv(ATTRIBUTION_CSV, index=False)
    print(f"  [saved] {ATTRIBUTION_CSV}")


def save_report_md(
    attr: dict,
    factor_names: list[str],
    mrc_df: pd.DataFrame,
    run_date: str,
) -> None:
    lines = [
        f"# Canyon v9 — Factor Attribution Report",
        f"",
        f"**Run date:** {run_date}  ",
        f"**Lookback:** {LOOKBACK_DAYS} trading days",
        f"",
        f"## Factor Loadings & Contributions",
        f"",
        f"| Factor | Beta | Return Contrib (ann.) | Variance % |",
        f"|--------|------|----------------------|------------|",
    ]
    for i, fname in enumerate(factor_names):
        beta = attr["betas"][i]
        rc   = attr["return_contribs"][i]
        vpct = attr["variance_pct"][i]
        sign = "+" if rc >= 0 else ""
        lines.append(f"| {fname} | {beta:.3f} | {sign}{rc*100:.1f}% | {vpct:.1f}% |")

    alp  = attr["alpha_annual"]
    sign = "+" if alp >= 0 else ""
    lines.append(f"| Alpha (α) | — | {sign}{alp*100:.1f}% | {attr['idio_var_pct']:.1f}% |")
    lines += [
        f"",
        f"**R² = {attr['r2']:.4f}**",
        f"",
    ]

    if mrc_df is not None and len(mrc_df) > 0:
        lines += [
            f"## Marginal Risk Contribution (Top 10 Positions)",
            f"",
            f"| Rank | Ticker | Weight | MRC | MRC % |",
            f"|------|--------|--------|-----|-------|",
        ]
        for _, row in mrc_df.head(10).iterrows():
            lines.append(
                f"| {int(row.name)+1} | {row['ticker']} | {row['weight']:.2%} "
                f"| {row['mrc']:.6f} | {row['mrc_pct']:.1f}% |"
            )
        lines.append("")

    lines += [
        f"---",
        f"*Generated by canyon_final_v9_step73_factor_attribution.py*",
    ]

    REPORT_MD.write_text("\n".join(lines))
    print(f"  [saved] {REPORT_MD}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(refresh: bool = False) -> None:
    run_date = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*65}")
    print(f"  Canyon v9 — Factor Attribution (Step 73)   {run_date}")
    print(f"{'='*65}")

    # ── Force cache refresh if requested ─────────────────────────────────────
    if refresh and FACTOR_CACHE_CSV.exists():
        FACTOR_CACHE_CSV.unlink()
        print("  [refresh] Factor cache cleared.")

    # ── Load universe ─────────────────────────────────────────────────────────
    print("\n[1] Loading universe …")
    universe = load_universe()
    print(f"  Universe: {len(universe)} tickers")

    # ── Load fundamental data ─────────────────────────────────────────────────
    print("\n[2] Loading fundamental data …")
    fund_df: pd.DataFrame | None = None
    if FUNDAMENTAL_CSV.exists():
        try:
            fund_df = pd.read_csv(FUNDAMENTAL_CSV)
            available_cols = fund_df.columns.tolist()
            print(f"  Fundamental columns: {available_cols}")
        except Exception as exc:
            print(f"  [warn] Could not load fundamental_features.csv: {exc}")
    else:
        print("  [warn] fundamental_features.csv not found — Value/Quality factors set to zero")

    # ── Fetch prices ──────────────────────────────────────────────────────────
    print("\n[3] Fetching prices …")
    # Request extra days so rolling windows don't eat all our data
    prices = fetch_prices(universe, days=LOOKBACK_DAYS + 30)
    print(f"  Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # Trim to exactly LOOKBACK_DAYS rows of returns (need LOOKBACK_DAYS+1 price rows)
    prices_trim = prices.iloc[-(LOOKBACK_DAYS + 1):]

    # ── Build factor returns ──────────────────────────────────────────────────
    print("\n[4] Building factor returns …")
    factor_df = build_factor_returns(prices_trim, fund_df)
    # Drop rows where any key factor is NaN (market must always be present)
    factor_df = factor_df.dropna(subset=["Market"])
    # Fill remaining NaN factor values with 0 (sparse data)
    factor_df = factor_df.fillna(0.0)
    print(f"  Factor returns: {factor_df.shape[0]} days × {len(FACTOR_NAMES)} factors")

    # ── Build portfolio returns ───────────────────────────────────────────────
    print("\n[5] Building portfolio returns …")
    port_series, weights = build_portfolio_returns(prices_trim, universe)

    # Align portfolio returns to factor return dates
    common_idx = factor_df.index.intersection(port_series.index)
    if len(common_idx) < 30:
        raise RuntimeError(
            f"Only {len(common_idx)} common dates between portfolio and factor returns. "
            "Check price data availability."
        )
    factor_aligned = factor_df.loc[common_idx]
    port_aligned   = port_series.loc[common_idx]
    print(f"  Aligned: {len(common_idx)} days")

    # ── OLS regression ────────────────────────────────────────────────────────
    print("\n[6] Running OLS regression …")
    y = port_aligned.values
    X = factor_aligned.values

    betas, alpha, r2, residuals = run_ols(y, X)
    print(f"  Betas: {dict(zip(FACTOR_NAMES, betas.round(4)))}")
    print(f"  Alpha (daily): {alpha:.6f}  |  R² = {r2:.4f}")

    # ── Decompose contributions ───────────────────────────────────────────────
    print("\n[7] Decomposing return & variance contributions …")
    factor_cov = np.cov(X.T, ddof=1)   # shape (n_factors, n_factors)
    if factor_cov.ndim == 0:
        factor_cov = factor_cov.reshape(1, 1)

    attr = decompose_contributions(betas, factor_aligned, factor_cov, alpha, r2, residuals)

    # Portfolio annualised vol (from daily returns)
    port_vol_ann = float(np.std(y, ddof=1)) * np.sqrt(ANN_FACTOR)
    print(f"  Portfolio annualised vol: {port_vol_ann*100:.1f}%")

    # ── Marginal risk contribution ────────────────────────────────────────────
    print("\n[8] Computing marginal risk contributions …")
    tickers_in_prices = list(prices.columns)
    mrc_df = compute_marginal_risk(
        weights=weights,
        tickers_in_prices=tickers_in_prices,
        betas_per_stock=betas,     # portfolio-level betas used as proxy
        factor_cov=factor_cov,
        port_vol=port_vol_ann,
    )
    if len(mrc_df) > 0:
        mrc_df.to_csv(MRC_CSV, index=False)
        print(f"  [saved] {MRC_CSV}  ({len(mrc_df)} positions)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(attr, FACTOR_NAMES)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("[9] Saving outputs …")
    save_attribution_csv(attr, FACTOR_NAMES)
    save_report_md(attr, FACTOR_NAMES, mrc_df, run_date)

    print(f"\n  Done.  Results in {ROOT}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 73 — Barra-style factor attribution"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of price cache",
    )
    args = parser.parse_args()
    main(refresh=args.refresh)
