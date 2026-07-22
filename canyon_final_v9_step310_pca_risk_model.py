#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 310: Barra-Lite PCA Statistical Risk Model
============================================================
Replaces the crude 3-factor OLS neutralisation from Step 260 with a
proper statistical risk model following the Barra USE3 methodology:

  Σ_total = B · Σ_F · B^T + D

  B      N×K factor loading matrix  (from PCA on correlation matrix)
  Σ_F    K×K factor return covariance  (63-day rolling, Ledoit-Wolf)
  D      N×N diagonal specific-risk matrix  (63-day rolling residuals)

Five modules:

  A — PCA Factor Extraction
      Fit on full IS period (≤ OOS_CUTOFF).  Extract K factors that
      explain ≥ VAR_TARGET of cross-sectional return variance.

  B — Daily Factor Returns + Rolling Covariance
      Project daily returns onto factor space.  Estimate 63-day rolling
      Σ_F with Ledoit-Wolf shrinkage for numerical stability.

  C — Specific Risk (Idiosyncratic)
      Compute residuals after factor projection.  63-day rolling variance.
      Apply 2% annualised floor (no stock can have zero idio risk).

  D — Portfolio Risk Decomposition
      Take the latest ML portfolio (from cpcv_predictions.csv or
      wf_oos_predictions.csv).  Decompose total risk into:
      systematic (factor) risk and specific (idiosyncratic) risk.

  E — Factor Exposure Report
      Per-stock R² (% of variance explained by factors).
      Top factor loadings for each stock.
      Monthly portfolio factor exposures over OOS period.

Outputs:
  pca_factor_loadings.csv      B matrix (ticker × factor_k)
  pca_factor_returns.csv       daily K factor returns
  pca_factor_cov.csv           latest K×K factor covariance
  pca_specific_risk.csv        per-stock annualised specific risk
  pca_portfolio_decomp.csv     systematic vs specific risk over time
  pca_r2_by_stock.csv          % variance explained per stock
  pca_report.md                narrative institutional report

Usage:
  .venv/bin/python canyon_final_v9_step310_pca_risk_model.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── constants ──────────────────────────────────────────────────────────────────
OOS_CUTOFF   = pd.Timestamp("2020-01-01")
VAR_TARGET   = 0.80       # target cumulative variance explained by factors
K_MAX        = 30         # max number of factors
K_MIN        = 5          # min number of factors
COV_WINDOW   = 63         # rolling covariance window (trading days)
SPEC_FLOOR   = 0.02       # minimum annualised idio vol (2%)
ANN_FACTOR   = 252
MAX_WEIGHT   = 0.08       # 8% single-name cap for portfolio
PREDS_FILE   = "cpcv_predictions.csv"   # prefer purged preds
PREDS_ALT    = "wf_oos_predictions.csv" # fallback

UNIVERSE = list(dict.fromkeys([
    "SPY", "QQQ", "SMH", "SOXX", "XLK", "XLE", "XLF", "XLV", "XLU", "XLP",
    "NVDA", "TSLA", "AMD", "MU", "GOOGL", "AMZN", "MSFT", "AAPL", "META", "JPM",
    "XOM", "CVX", "JNJ", "WMT", "KO", "PEP", "MRK", "ABBV", "UNH", "LLY", "TMO",
    "COST", "V", "MA", "HD", "PYPL", "NFLX", "INTC", "QCOM", "TXN", "AVGO",
    "CRM", "ADBE",
]))


# =============================================================================
# A. Load prices and compute returns
# =============================================================================

def load_prices() -> pd.DataFrame:
    cache = ROOT / "backtest_price_cache.csv"
    if not cache.exists():
        raise FileNotFoundError("backtest_price_cache.csv not found — run step290 first")
    prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    avail  = [t for t in UNIVERSE if t in prices.columns]
    return prices[avail].dropna(how="all")


def clean_returns(prices: pd.DataFrame, min_coverage: float = 0.80) -> pd.DataFrame:
    """Daily log-returns, drop tickers with < min_coverage non-NaN."""
    rets = np.log(prices / prices.shift(1))
    coverage = rets.notna().mean()
    good = coverage[coverage >= min_coverage].index.tolist()
    rets = rets[good].copy()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    return rets


# =============================================================================
# B. PCA Factor Model
# =============================================================================

def fit_pca_model(
    rets: pd.DataFrame,
    is_end: pd.Timestamp,
    var_target: float = VAR_TARGET,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> tuple[np.ndarray, list[str], int, np.ndarray]:
    """
    Fit PCA on IS correlation matrix.

    Returns:
      B          (N, K) factor loading matrix
      tickers    list of N ticker names (ordered)
      K          number of factors chosen
      expl_var   explained variance ratio per factor
    """
    is_rets = rets[rets.index <= is_end].copy()
    is_rets = is_rets.dropna(how="all")

    # Fill sparse NaN with cross-sectional median return on that day
    cs_median = is_rets.median(axis=1)
    for col in is_rets.columns:
        is_rets[col] = is_rets[col].fillna(cs_median)

    tickers = list(is_rets.columns)
    N = len(tickers)

    # Standardise each stock's return series (correlation-based PCA)
    scaler = StandardScaler()
    X = scaler.fit_transform(is_rets.values)   # (T, N)

    pca = PCA(n_components=k_max, random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.argmax(cumvar >= var_target)) + 1
    K = max(k_min, min(k_max, K))

    # B: (N, K)  — columns are factor loadings for each stock
    B = pca.components_[:K].T   # shape (N, K)

    expl_var = pca.explained_variance_ratio_[:K]
    print(f"  PCA: K={K} factors explain "
          f"{cumvar[K-1]:.1%} of IS variance  (N={N} stocks, T={len(is_rets)} days)")

    return B, tickers, K, expl_var


# =============================================================================
# C. Daily Factor Returns
# =============================================================================

def compute_factor_returns(
    rets: pd.DataFrame,
    B: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Project daily returns onto factor space.
    F_t = B^T · r_t   (shape K × 1 per day, since B columns orthonormal from PCA)

    Actually B from sklearn PCA components_ is already orthonormal, so:
      F_t = B^T r_t_standardised  ≈  projections onto principal directions
    We use the raw (un-standardised) returns here for interpretability.
    """
    # Align returns to tickers used in PCA
    r = rets[tickers].copy()
    cs_median = r.median(axis=1)
    for col in r.columns:
        r[col] = r[col].fillna(cs_median)

    R = r.values   # (T, N)
    F = R @ B      # (T, K) — factor return matrix

    col_names = [f"factor_{k+1:02d}" for k in range(B.shape[1])]
    fret_df = pd.DataFrame(F, index=r.index, columns=col_names)
    return fret_df


# =============================================================================
# D. Rolling Factor Covariance (Ledoit-Wolf)
# =============================================================================

def rolling_factor_cov(
    fret: pd.DataFrame,
    window: int = COV_WINDOW,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    For each date, estimate K×K factor covariance using Ledoit-Wolf shrinkage
    on the trailing `window` days.
    Returns dict: date -> covariance matrix.
    """
    cov_dict: dict[pd.Timestamp, np.ndarray] = {}
    F = fret.values
    idx = fret.index
    K = F.shape[1]

    for i in range(window, len(F)):
        window_F = F[i - window: i]
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(window_F)
            cov_dict[idx[i]] = lw.covariance_
        except Exception:
            cov_dict[idx[i]] = np.cov(window_F.T)

    return cov_dict


# =============================================================================
# E. Specific Risk (Idiosyncratic)
# =============================================================================

def compute_specific_risk(
    rets: pd.DataFrame,
    B: np.ndarray,
    tickers: list[str],
    window: int = COV_WINDOW,
    floor_ann: float = SPEC_FLOOR,
) -> pd.DataFrame:
    """
    Residual = actual return − factor reconstruction.
    Specific risk = rolling std of residuals.
    Returns DataFrame (date × ticker) of annualised specific risk.
    """
    r = rets[tickers].copy()
    cs_median = r.median(axis=1)
    for col in r.columns:
        r[col] = r[col].fillna(cs_median)

    R       = r.values                  # (T, N)
    F       = R @ B                     # (T, K)
    R_hat   = F @ B.T                   # (T, N) factor reconstruction
    E       = R - R_hat                  # (T, N) residuals

    resid_df = pd.DataFrame(E, index=r.index, columns=tickers)

    # Rolling annualised specific vol
    spec_vol = resid_df.rolling(window).std() * np.sqrt(ANN_FACTOR)
    spec_vol = spec_vol.clip(lower=floor_ann)   # apply floor

    return spec_vol


# =============================================================================
# F. Portfolio Risk Decomposition
# =============================================================================

def load_portfolio_weights(latest_date: pd.Timestamp) -> dict[str, float]:
    """Load top-8 ML portfolio weights from predictions CSV."""
    for fname in [PREDS_FILE, PREDS_ALT]:
        path = ROOT / fname
        if path.exists():
            preds = pd.read_csv(path, parse_dates=["rebalance_date"])
            oos   = preds[preds["rebalance_date"] >= OOS_CUTOFF]
            if oos.empty:
                continue
            latest = oos["rebalance_date"].max()
            top    = (oos[oos["rebalance_date"] == latest]
                      .nlargest(8, "ensemble_score")["ticker"].tolist())
            w = {t: 1.0 / len(top) for t in top}
            print(f"  Portfolio ({latest.date()}): {top}")
            return w
    return {}


def decompose_portfolio_risk(
    weights: dict[str, float],
    B: np.ndarray,
    tickers: list[str],
    sigma_F: np.ndarray,
    spec_vol_row: pd.Series,
) -> dict[str, float]:
    """
    σ²_total = w^T B Σ_F B^T w  +  w^T D w
             = σ²_systematic    +  σ²_specific

    Returns dict of risk components (annualised vol, not variance).
    """
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    held = [t for t in weights if t in ticker_to_idx]

    if not held:
        return {}

    w_vec = np.zeros(len(tickers))
    for t in held:
        w_vec[ticker_to_idx[t]] = weights[t]

    # Annualised daily spec variance
    D_vec = np.array([
        (spec_vol_row.get(t, SPEC_FLOOR) / np.sqrt(ANN_FACTOR)) ** 2
        for t in tickers
    ])

    # Systematic component: w^T B Σ_F B^T w
    Bw            = B.T @ w_vec          # K × 1
    var_sys_daily = float(Bw @ sigma_F @ Bw)
    var_sys_ann   = var_sys_daily * ANN_FACTOR

    # Specific component: w^T D w
    var_spec_ann = float((w_vec ** 2 @ D_vec) * ANN_FACTOR)

    var_total_ann = var_sys_ann + var_spec_ann
    vol_total_ann = np.sqrt(max(var_total_ann, 0))

    return {
        "vol_total_ann":    round(vol_total_ann, 4),
        "vol_systematic":   round(np.sqrt(max(var_sys_ann, 0)), 4),
        "vol_specific":     round(np.sqrt(max(var_spec_ann, 0)), 4),
        "pct_systematic":   round(var_sys_ann / max(var_total_ann, 1e-10) * 100, 1),
        "pct_specific":     round(var_spec_ann / max(var_total_ann, 1e-10) * 100, 1),
    }


def decompose_all_rebalances(
    preds_path: Path,
    B: np.ndarray,
    tickers: list[str],
    cov_dict: dict,
    spec_vol: pd.DataFrame,
) -> pd.DataFrame:
    """Run risk decomposition for every OOS rebalance date."""
    if not preds_path.exists():
        return pd.DataFrame()

    preds = pd.read_csv(preds_path, parse_dates=["rebalance_date"])
    oos   = preds[preds["rebalance_date"] >= OOS_CUTOFF]
    if oos.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for reb_date in sorted(oos["rebalance_date"].unique()):
        top = (oos[oos["rebalance_date"] == reb_date]
               .nlargest(8, "ensemble_score")["ticker"].tolist())
        if not top:
            continue

        weights = {t: 1.0 / len(top) for t in top}

        # Nearest available covariance date
        avail_cov = [d for d in cov_dict if d <= reb_date]
        if not avail_cov:
            continue
        cov_date = max(avail_cov)
        sigma_F  = cov_dict[cov_date]

        # Nearest available spec vol date
        avail_spec = spec_vol.index[spec_vol.index <= reb_date]
        if len(avail_spec) == 0:
            continue
        spec_row = spec_vol.loc[avail_spec[-1]]

        d = decompose_portfolio_risk(weights, B, tickers, sigma_F, spec_row)
        if d:
            d["rebalance_date"] = reb_date
            d["holdings"] = ",".join(top)
            rows.append(d)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# G. R² by stock
# =============================================================================

def r2_by_stock(
    rets: pd.DataFrame,
    B: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """% of each stock's total variance explained by the K factors."""
    r = rets[tickers].copy()
    cs_med = r.median(axis=1)
    for col in r.columns:
        r[col] = r[col].fillna(cs_med)
    r = r.fillna(0.0)                   # zero-fill residual NaN (early history)

    R     = r.values.astype(float)
    F     = R @ B
    R_hat = F @ B.T
    E     = R - R_hat

    r_var = np.nanvar(R, axis=0)
    e_var = np.nanvar(E, axis=0)
    r2_vals = np.where(r_var > 1e-12, 1 - e_var / r_var, np.nan)
    df = pd.DataFrame({
        "ticker":    tickers,
        "r2":        np.round(r2_vals, 4),
        "idio_pct":  np.round((1 - r2_vals) * 100, 1),
        "sys_pct":   np.round(r2_vals * 100, 1),
    }).sort_values("r2", ascending=False)
    return df


# =============================================================================
# H. Report
# =============================================================================

def generate_report(
    K: int,
    expl_var: np.ndarray,
    r2_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
    tickers: list[str],
    ts: str,
) -> str:
    lines = [
        "# Canyon v9 — Step 310: Barra-Lite PCA Risk Model",
        f"Generated: {ts}",
        "",
        "## Model Architecture",
        "",
        "```",
        "Σ_total = B · Σ_F · B^T + D",
        "",
        f"  B     {len(tickers)}×{K} factor loadings  (PCA on IS correlation matrix)",
        f"  Σ_F   {K}×{K} factor covariance  (63-day rolling, Ledoit-Wolf shrinkage)",
        f"  D     {len(tickers)}×{len(tickers)} diagonal specific risk",
        "```",
        "",
        "## Factor Summary",
        "",
        f"| K | Variance Explained | Methodology |",
        "|---|:-----------------:|-------------|",
        f"| {K} | {np.cumsum(expl_var)[-1]:.1%} | PCA on correlation matrix (IS: pre-2020) |",
        "",
        "### Top 10 Factor Contributions",
        "",
        "| Factor | Expl. Var | Cumulative |",
        "|--------|:---------:|:----------:|",
    ]
    cumvar = np.cumsum(expl_var)
    for k in range(min(10, K)):
        lines.append(f"| Factor {k+1:02d} | {expl_var[k]:.2%} | {cumvar[k]:.2%} |")

    # R² summary
    if not r2_df.empty:
        lines += [
            "",
            "## Stock-Level R² (% Variance Explained by Factors)",
            "",
            f"- Mean R²: **{r2_df['r2'].mean():.1%}**  (industry benchmark: 60–75%)",
            f"- Median R²: {r2_df['r2'].median():.1%}",
            f"- Most systematic: {r2_df.iloc[0]['ticker']} ({r2_df.iloc[0]['sys_pct']:.1f}%)",
            f"- Most idiosyncratic: {r2_df.iloc[-1]['ticker']} "
            f"({r2_df.iloc[-1]['idio_pct']:.1f}% idio)",
            "",
            "| Ticker | R² | Systematic % | Idiosyncratic % |",
            "|--------|:--:|:------------:|:---------------:|",
        ]
        for _, row in r2_df.iterrows():
            lines.append(
                f"| {row['ticker']} | {row['r2']:.3f} | "
                f"{row['sys_pct']:.1f}% | {row['idio_pct']:.1f}% |"
            )

    # Portfolio decomposition
    if not decomp_df.empty:
        avg_sys  = decomp_df["vol_systematic"].mean()
        avg_spec = decomp_df["vol_specific"].mean()
        avg_tot  = decomp_df["vol_total_ann"].mean()
        avg_pct  = decomp_df["pct_systematic"].mean()

        lines += [
            "",
            "## OOS Portfolio Risk Decomposition",
            "",
            f"Average over {len(decomp_df)} OOS rebalance dates:",
            "",
            f"| Component | Avg Annualised Vol | Avg % of Total |",
            f"|-----------|:-----------------:|:--------------:|",
            f"| Total     | {avg_tot:.1%}             | 100%           |",
            f"| Systematic (factor) | {avg_sys:.1%}   | {avg_pct:.1f}% |",
            f"| Specific (idio)     | {avg_spec:.1%}  | {100-avg_pct:.1f}% |",
            "",
            "> A well-diversified portfolio should have systematic risk ≈ 60–80%.",
            "> High specific risk (> 40%) means concentrated idiosyncratic bets.",
            "",
            "### Recent Rebalance Dates",
            "",
            "| Date | Total Vol | Systematic | Specific | Sys% |",
            "|------|:---------:|:----------:|:--------:|:----:|",
        ]
        for _, row in decomp_df.tail(8).iterrows():
            lines.append(
                f"| {str(row['rebalance_date'])[:10]} | "
                f"{row['vol_total_ann']:.1%} | "
                f"{row['vol_systematic']:.1%} | "
                f"{row['vol_specific']:.1%} | "
                f"{row['pct_systematic']:.0f}% |"
            )

    lines += [
        "",
        "## How This Changes the Pipeline",
        "",
        "1. **Step 320 (MVO Optimizer)** now has a proper covariance matrix.",
        "   Previously score-weighted → now mean-variance optimal with factor constraints.",
        "",
        "2. **Factor exposure limits**: cap any single-factor loading",
        "   |w^T b_k| ≤ limit_k to prevent over-concentration in one factor.",
        "",
        "3. **Risk budgeting**: systematic vs specific risk split tells the PM",
        "   whether P&L is coming from factor bets or stock selection.",
        "",
        "4. **Tracking error** to SPY is now computable:",
        "   σ_TE = √[(w−w_SPY)^T Σ (w−w_SPY)]",
        "",
        "## Reference",
        "",
        "- Barra USE3 Technical Document (1998).",
        "- Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional",
        "  covariance matrices. *Journal of Multivariate Analysis*, 88, 365-411.",
        "- Connor & Korajczyk (1988). Risk and return in an equilibrium APT.",
    ]
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 310: Barra-Lite PCA Risk Model")
    print("=" * 70)

    # A. Prices
    print("\n[A] Loading prices …")
    prices = load_prices()
    rets   = clean_returns(prices)
    print(f"  Returns: {rets.shape[1]} tickers × {len(rets)} days  "
          f"({rets.index.min().date()} → {rets.index.max().date()})")

    # B. PCA factor model
    print("\n[B] Fitting PCA factor model on IS data (pre-2020) …")
    B, tickers, K, expl_var = fit_pca_model(rets, is_end=OOS_CUTOFF)

    # Save factor loadings
    load_df = pd.DataFrame(
        B, index=tickers,
        columns=[f"factor_{k+1:02d}" for k in range(K)]
    )
    load_df.index.name = "ticker"
    load_df.to_csv(ROOT / "pca_factor_loadings.csv")
    print(f"  Saved: pca_factor_loadings.csv  ({len(tickers)} × {K})")

    # C. Daily factor returns
    print("\n[C] Computing daily factor returns …")
    fret = compute_factor_returns(rets, B, tickers)
    fret.to_csv(ROOT / "pca_factor_returns.csv")
    print(f"  Factor returns: {fret.shape}  ({fret.index.min().date()} → {fret.index.max().date()})")

    # Summarise factor return stats
    fret_ann_vol = fret.std() * np.sqrt(ANN_FACTOR)
    print("  Factor annualised vols (top 5):")
    for k, v in fret_ann_vol.head(5).items():
        print(f"    {k}: {v:.2%}")

    # D. Rolling Ledoit-Wolf factor covariance
    print("\n[D] Rolling factor covariance (Ledoit-Wolf, 63-day window) …")
    cov_dict = rolling_factor_cov(fret)
    latest_cov_date = max(cov_dict.keys())
    latest_sigma_F  = cov_dict[latest_cov_date]

    # Save latest covariance
    cov_df = pd.DataFrame(
        latest_sigma_F,
        index=[f"factor_{k+1:02d}" for k in range(K)],
        columns=[f"factor_{k+1:02d}" for k in range(K)],
    )
    cov_df.to_csv(ROOT / "pca_factor_cov.csv")
    print(f"  Latest factor cov: {K}×{K}  as of {latest_cov_date.date()}")

    # E. Specific risk
    print("\n[E] Computing specific risk …")
    spec_vol = compute_specific_risk(rets, B, tickers)
    spec_today = spec_vol.iloc[-1].sort_values(ascending=False)
    spec_vol.to_csv(ROOT / "pca_specific_risk.csv")
    print("  Latest specific risk (annualised, top 5 highest):")
    for tkr, v in spec_today.head(5).items():
        print(f"    {tkr}: {v:.1%}")

    # F. R² by stock
    print("\n[F] Computing R² per stock …")
    r2_df = r2_by_stock(rets, B, tickers)
    r2_df.to_csv(ROOT / "pca_r2_by_stock.csv", index=False)
    print(f"  Mean R²: {r2_df['r2'].mean():.1%}  "
          f"(range: {r2_df['r2'].min():.1%} – {r2_df['r2'].max():.1%})")

    # G. Portfolio risk decomposition
    print("\n[G] Portfolio risk decomposition (OOS rebalances) …")
    preds_path = ROOT / PREDS_FILE
    if not preds_path.exists():
        preds_path = ROOT / PREDS_ALT

    decomp_df = decompose_all_rebalances(preds_path, B, tickers, cov_dict, spec_vol)
    if not decomp_df.empty:
        decomp_df.to_csv(ROOT / "pca_portfolio_decomp.csv", index=False)
        avg_sys = decomp_df["pct_systematic"].mean()
        avg_vol = decomp_df["vol_total_ann"].mean()
        print(f"  {len(decomp_df)} OOS rebalances  |  "
              f"avg total vol: {avg_vol:.1%}  |  "
              f"avg systematic: {avg_sys:.1f}%")
    else:
        print("  No OOS predictions found for decomposition")

    # H. Report
    report = generate_report(K, expl_var, r2_df, decomp_df, tickers, ts)
    (ROOT / "pca_report.md").write_text(report)

    print("\n" + "=" * 70)
    print("Step 310 Complete — Barra-Lite PCA Risk Model")
    print("=" * 70)
    outputs = [
        "pca_factor_loadings.csv", "pca_factor_returns.csv",
        "pca_factor_cov.csv", "pca_specific_risk.csv",
        "pca_portfolio_decomp.csv", "pca_r2_by_stock.csv", "pca_report.md",
    ]
    print("\nOutputs:")
    for f in outputs:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "not created"
        print(f"  {f:<42} {sz}")


if __name__ == "__main__":
    main()
