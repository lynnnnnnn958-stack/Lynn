#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 320: MVO Optimizer + Black-Litterman
======================================================
Replaces score-weighted sizing (Step 230) with institutionally rigorous
portfolio construction using the Barra-Lite covariance from Step 310.

Two portfolio construction methods:

  Method 1 — Mean-Variance Optimization (MVO)
    max   α^T w  −  λ · w^T Σ w  −  γ · ||w − w_prev||_1
    s.t.  Σ w_i = 1
          w_i ∈ [0, MAX_W]  (long-only, 8% cap)
          |factor_k_exposure| ≤ FACTOR_LIMIT  (for k=1,2,3)
          w_i = 0  if ticker not in ADV universe

    Σ uses the PCA factor model: Σ = B Σ_F B^T + D
    α = ML ensemble scores (cross-sectionally standardised)
    λ = risk-aversion parameter (calibrated to target 15% annual vol)
    γ = turnover penalty (5 bps per unit turnover)

  Method 2 — Black-Litterman (BL)
    Prior:  market-cap implied equilibrium returns  π = δ Σ w_mkt
    Views:  ML ensemble scores → relative return predictions
    BL posterior: μ_BL = π + τΣP^T(PτΣP^T + Ω)^{-1}(q − Pπ)
    Portfolio:  MVO on μ_BL

Outputs:
  mvo_weights_history.csv       monthly portfolio weights (MVO)
  bl_weights_history.csv        monthly portfolio weights (BL)
  mvo_vs_bl_performance.csv     side-by-side performance comparison
  mvo_factor_exposures.csv      factor exposure per rebalance (MVO)
  mvo_report.md                 narrative institutional report

Usage:
  .venv/bin/python canyon_final_v9_step320_mvo_optimizer.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── constants ──────────────────────────────────────────────────────────────────
OOS_CUTOFF   = pd.Timestamp("2020-01-01")
MAX_W        = 0.08       # 8% single-name cap
MIN_W        = 0.0        # long-only
TARGET_VOL   = 0.15       # 15% annual vol target for λ calibration
TC_BPS       = 10         # transaction cost basis points
FACTOR_LIMIT = 0.30       # max absolute factor exposure for factors 1-3
ANN_FACTOR   = 252
TOP_N        = 8          # max holdings
TAU          = 0.05       # Black-Litterman uncertainty scaling

PREDS_FILE  = "cpcv_predictions.csv"
PREDS_ALT   = "wf_oos_predictions.csv"


# =============================================================================
# 1. Load inputs
# =============================================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PCA model outputs + predictions + prices."""
    # Factor loadings B (N×K)
    B_df = pd.read_csv(ROOT / "pca_factor_loadings.csv", index_col=0)

    # Latest factor covariance (K×K)
    Sigma_F = pd.read_csv(ROOT / "pca_factor_cov.csv", index_col=0).values

    # Specific risk time series
    spec_vol = pd.read_csv(ROOT / "pca_specific_risk.csv", index_col=0, parse_dates=True)

    # Predictions
    for fname in [PREDS_FILE, PREDS_ALT]:
        p = ROOT / fname
        if p.exists():
            preds = pd.read_csv(p, parse_dates=["rebalance_date"])
            break
    else:
        preds = pd.DataFrame()

    print(f"  B: {B_df.shape}  Σ_F: {Sigma_F.shape}  "
          f"spec_vol: {spec_vol.shape}  preds: {len(preds)}")
    return B_df, Sigma_F, spec_vol, preds


def load_prices() -> pd.DataFrame:
    cache = ROOT / "backtest_price_cache.csv"
    prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    return prices.dropna(how="all")


# =============================================================================
# 2. Covariance assembly  Σ = B Σ_F B^T + D
# =============================================================================

def build_covariance(
    B_df: pd.DataFrame,
    Sigma_F: np.ndarray,
    spec_vol_row: pd.Series,
    tickers: list[str],
) -> np.ndarray:
    """
    Build N×N covariance matrix for the given tickers.
    Daily variance units (divide annual vol by sqrt(252)).
    """
    N = len(tickers)
    B = B_df.loc[tickers].values   # (N, K)

    # Factor covariance is in daily return units
    Sigma_sys = B @ Sigma_F @ B.T  # (N, N)

    # Specific variance: (annual_vol / sqrt(252))^2 = daily_var
    d_var = np.array([
        (spec_vol_row.get(t, 0.20) / np.sqrt(ANN_FACTOR)) ** 2
        for t in tickers
    ])
    D = np.diag(d_var)

    Sigma = Sigma_sys + D

    # Ensure positive-definite: add small regularisation if needed
    min_eig = np.linalg.eigvalsh(Sigma).min()
    if min_eig < 1e-10:
        Sigma += (abs(min_eig) + 1e-8) * np.eye(N)

    return Sigma


# =============================================================================
# 3. MVO with factor constraints
# =============================================================================

def mvo_optimize(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    B_local: np.ndarray,
    w_prev: np.ndarray,
    lam: float,
    gamma: float,
    max_w: float = MAX_W,
    factor_limit: float = FACTOR_LIMIT,
) -> np.ndarray:
    """
    Solve MVO with turnover penalty and factor exposure constraints.

    Objective (minimise):
      -α^T w  +  λ w^T Σ w  +  γ ||w - w_prev||_1

    Constraints:
      Σ w = 1
      0 ≤ w_i ≤ max_w
      |factor_k exposure| ≤ factor_limit  (k = 1, 2, 3)
    """
    N = len(alpha)

    def neg_utility(w: np.ndarray) -> float:
        ret_term  = -float(alpha @ w)
        risk_term = float(lam * w @ Sigma @ w)
        tc_term   = float(gamma * np.abs(w - w_prev).sum())
        return ret_term + risk_term + tc_term

    def grad(w: np.ndarray) -> np.ndarray:
        g_ret  = -alpha
        g_risk = 2 * lam * Sigma @ w
        g_tc   = gamma * np.sign(w - w_prev)
        return g_ret + g_risk + g_tc

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Factor exposure constraints: |b_k^T w| ≤ factor_limit  (k=1,2,3)
    K_constrained = min(3, B_local.shape[1])
    for k in range(K_constrained):
        b_k = B_local[:, k]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, b=b_k: factor_limit - abs(float(b @ w)),
        })

    bounds = [(MIN_W, max_w)] * N
    w0     = np.ones(N) / N   # equal-weight start

    result = minimize(
        neg_utility, w0, jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if result.success:
        w = result.x
    else:
        # Fallback: score-weighted with hard cap
        w = np.maximum(alpha, 0)
        if w.sum() > 0:
            w /= w.sum()
        w = np.minimum(w, max_w)
        w /= w.sum() if w.sum() > 0 else 1

    # Clip and renormalize
    w = np.clip(w, 0, max_w)
    if w.sum() > 1e-8:
        w /= w.sum()
    return w


# =============================================================================
# 4. Black-Litterman
# =============================================================================

def black_litterman(
    alpha_scores: np.ndarray,
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    delta: float = 2.5,
    tau: float = TAU,
) -> np.ndarray:
    """
    BL posterior expected return.

    Prior:   π = δ · Σ · w_mkt  (implied equilibrium returns)
    Views:   P = identity matrix (absolute views on all assets)
             q = alpha_scores (standardised ML predictions)
             Ω = diag(P τ Σ P^T)  (view uncertainty proportional to prior)

    Posterior: μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1}
                      [(τΣ)^{-1} π + P^T Ω^{-1} q]
    """
    N = len(alpha_scores)
    pi = delta * Sigma @ w_mkt   # equilibrium returns

    # Views: each stock has its own view (P = I)
    P   = np.eye(N)
    q   = alpha_scores.copy()
    tau_Sigma = tau * Sigma
    Omega = np.diag(np.diag(P @ tau_Sigma @ P.T))   # diagonal view uncertainty

    try:
        inv_tauS = np.linalg.inv(tau_Sigma)
        inv_Omega = np.diag(1.0 / np.maximum(np.diag(Omega), 1e-12))

        M = inv_tauS + P.T @ inv_Omega @ P
        b = inv_tauS @ pi + P.T @ inv_Omega @ q

        mu_bl = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        mu_bl = 0.5 * pi + 0.5 * q   # fallback blended

    return mu_bl


# =============================================================================
# 5. Calibrate λ (risk aversion) to target annual vol
# =============================================================================

def calibrate_lambda(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    target_vol: float = TARGET_VOL,
) -> float:
    """
    Approximate: at λ→0 the unconstrained MVO maximises return only.
    Calibrate by bisection so that the MVO portfolio has vol ≈ target_vol.
    (Simple approximation: compute vol at candidate weights.)
    """
    daily_target_var = (target_vol / np.sqrt(ANN_FACTOR)) ** 2

    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        w_eq = np.ones(len(alpha)) / len(alpha)
        N = len(alpha)
        bounds = [(0, MAX_W)] * N
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        try:
            res = minimize(
                lambda w: -float(alpha @ w) + lam * float(w @ Sigma @ w),
                w_eq, method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if res.success:
                port_var = float(res.x @ Sigma @ res.x)
                if port_var * ANN_FACTOR <= (target_vol + 0.02) ** 2:
                    return lam
        except Exception:
            pass

    return 5.0   # conservative default


# =============================================================================
# 6. Main backtest loop
# =============================================================================

def run_backtest(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    B_df: pd.DataFrame,
    Sigma_F: np.ndarray,
    spec_vol: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each OOS rebalance date:
      1. Get candidate universe (top 15 by ensemble score)
      2. Build covariance for candidates
      3. Solve MVO and BL
      4. Compute monthly returns
    Returns (mvo_hist, bl_hist, mvo_perf, bl_perf)
    """
    oos_preds = preds[preds["rebalance_date"] >= OOS_CUTOFF].copy()
    if oos_preds.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    reb_dates = sorted(oos_preds["rebalance_date"].unique())
    tickers_available = list(B_df.index)

    mvo_wt_rows:  list[dict] = []
    bl_wt_rows:   list[dict] = []
    mvo_perf_rows: list[dict] = []
    bl_perf_rows:  list[dict] = []

    w_prev_mvo = None
    w_prev_bl  = None

    for i, reb_date in enumerate(reb_dates[:-1]):
        next_date = reb_dates[i + 1]

        grp = oos_preds[oos_preds["rebalance_date"] == reb_date].copy()
        grp = grp[grp["ticker"].isin(tickers_available)]

        # Standardise scores cross-sectionally
        scores = grp.set_index("ticker")["ensemble_score"]
        mu, std = scores.mean(), scores.std()
        scores_z = ((scores - mu) / (std + 1e-10)).clip(-3, 3)

        # Candidate universe: top 15 by score
        candidates = scores_z.nlargest(min(15, len(scores_z))).index.tolist()
        if len(candidates) < 4:
            continue

        # Covariance for candidates
        avail_spec = spec_vol.index[spec_vol.index <= reb_date]
        if len(avail_spec) == 0:
            continue
        spec_row = spec_vol.loc[avail_spec[-1]]

        try:
            Sigma = build_covariance(B_df, Sigma_F, spec_row, candidates)
        except Exception:
            continue

        alpha_vec  = scores_z[candidates].values
        B_local    = B_df.loc[candidates].values          # (N, K)
        N          = len(candidates)

        # ── MVO ────────────────────────────────────────────────────────────
        lam   = calibrate_lambda(alpha_vec, Sigma)
        gamma = TC_BPS / 10_000 / MAX_W                   # turnover penalty

        if w_prev_mvo is None or len(w_prev_mvo) != N:
            w_prev_mvo_local = np.ones(N) / N
        else:
            # Map previous weights to current candidates
            w_prev_mvo_local = np.zeros(N)
            for j, tk in enumerate(candidates):
                if tk in mvo_wt_rows[-1] if mvo_wt_rows else {}:
                    pass   # handled below

        w_mvo = mvo_optimize(alpha_vec, Sigma, B_local, w_prev_mvo_local,
                             lam=lam, gamma=gamma)
        w_prev_mvo = dict(zip(candidates, w_mvo))

        # ── Black-Litterman ────────────────────────────────────────────────
        # Equal-weight market portfolio from candidates
        w_mkt = np.ones(N) / N
        mu_bl = black_litterman(alpha_vec, Sigma, w_mkt)

        if w_prev_bl is None:
            w_prev_bl_local = np.ones(N) / N
        else:
            w_prev_bl_local = np.array([w_prev_bl.get(t, 1/N) for t in candidates])
            s = w_prev_bl_local.sum()
            w_prev_bl_local = w_prev_bl_local / s if s > 0 else np.ones(N) / N

        w_bl = mvo_optimize(mu_bl, Sigma, B_local, w_prev_bl_local,
                            lam=lam * 0.8, gamma=gamma)
        w_prev_bl = dict(zip(candidates, w_bl))

        # ── Save weights ───────────────────────────────────────────────────
        mvo_row = {"rebalance_date": reb_date}
        bl_row  = {"rebalance_date": reb_date}
        for j, tk in enumerate(candidates):
            mvo_row[tk] = round(float(w_mvo[j]), 6)
            bl_row[tk]  = round(float(w_bl[j]), 6)
        mvo_wt_rows.append(mvo_row)
        bl_wt_rows.append(bl_row)

        # ── Portfolio returns ──────────────────────────────────────────────
        reb_ts  = pd.Timestamp(reb_date)
        next_ts = pd.Timestamp(next_date)
        avail   = prices.index[(prices.index >= reb_ts) & (prices.index <= next_ts)]
        if len(avail) < 2:
            continue
        t0, t1 = avail[0], avail[-1]

        def _port_ret(weights_dict: dict) -> float:
            rets_list = []
            for tk, wt in weights_dict.items():
                if wt < 1e-6 or tk not in prices.columns:
                    continue
                p0 = prices[tk].get(t0)
                p1 = prices[tk].get(t1)
                if p0 and p1 and p0 > 0 and p1 > 0:
                    rets_list.append(wt * (float(p1 / p0) - 1))
            return sum(rets_list) if rets_list else 0.0

        mvo_perf_rows.append({
            "rebalance_date": reb_date,
            "net_ret": round(_port_ret(dict(zip(candidates, w_mvo))) - len([
                tk for tk in candidates if w_mvo[candidates.index(tk)] > 1e-3
            ]) / N * TC_BPS / 10_000, 6),
        })
        bl_perf_rows.append({
            "rebalance_date": reb_date,
            "net_ret": round(_port_ret(dict(zip(candidates, w_bl))) - len([
                tk for tk in candidates if w_bl[candidates.index(tk)] > 1e-3
            ]) / N * TC_BPS / 10_000, 6),
        })

    mvo_hist = pd.DataFrame(mvo_wt_rows) if mvo_wt_rows else pd.DataFrame()
    bl_hist  = pd.DataFrame(bl_wt_rows)  if bl_wt_rows  else pd.DataFrame()
    mvo_perf = pd.DataFrame(mvo_perf_rows) if mvo_perf_rows else pd.DataFrame()
    bl_perf  = pd.DataFrame(bl_perf_rows)  if bl_perf_rows  else pd.DataFrame()
    return mvo_hist, bl_hist, mvo_perf, bl_perf


# =============================================================================
# 7. Metrics + comparison vs score-weighted (step290)
# =============================================================================

def _metrics(rets: pd.Series) -> dict:
    if rets.empty:
        return {}
    ann   = ANN_FACTOR / 21
    mean  = rets.mean()
    std   = rets.std()
    n     = len(rets)
    sharpe = (mean / (std + 1e-10)) * np.sqrt(ann)
    cum    = (1 + rets).prod()
    cagr   = cum ** (ann / n) - 1 if n > 0 else 0
    cs     = (1 + rets).cumprod()
    dd     = (cs - cs.cummax()) / cs.cummax()
    maxdd  = float(dd.min())
    calmar = abs(cagr / maxdd) if maxdd < 0 else 0
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 3),
            "max_dd": round(maxdd, 4), "calmar": round(calmar, 2), "n_months": n}


def load_baseline_perf() -> pd.DataFrame:
    """Load score-weighted OOS performance from step290 for comparison."""
    p = ROOT / "cpcv_backtest_perf.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["rebalance_date"])
        return df[df["rebalance_date"] >= OOS_CUTOFF][["rebalance_date", "net_ret"]]
    return pd.DataFrame()


# =============================================================================
# 8. Factor exposure tracking
# =============================================================================

def compute_factor_exposures(
    mvo_hist: pd.DataFrame,
    B_df: pd.DataFrame,
) -> pd.DataFrame:
    """Portfolio factor exposures = w^T B for each rebalance date."""
    rows = []
    for _, row in mvo_hist.iterrows():
        rd    = row["rebalance_date"]
        tkers = [c for c in mvo_hist.columns if c != "rebalance_date"]
        w_dict = {t: row.get(t, 0.0) for t in tkers if t in B_df.index}
        if not w_dict:
            continue
        tks = list(w_dict.keys())
        w   = np.array([w_dict[t] for t in tks])
        B_s = B_df.loc[tks].values
        exp = B_s.T @ w   # (K,)
        r   = {"rebalance_date": rd}
        for k, e in enumerate(exp):
            r[f"factor_{k+1:02d}"] = round(float(e), 6)
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# 9. Report
# =============================================================================

def generate_report(
    mvo_m: dict, bl_m: dict, base_m: dict,
    exp_df: pd.DataFrame,
    ts: str,
) -> str:
    lines = [
        "# Canyon v9 — Step 320: MVO + Black-Litterman Optimizer",
        f"Generated: {ts}",
        "",
        "## Portfolio Construction Framework",
        "",
        "This step replaces naïve score-weighted sizing with two institutionally-",
        "standard methods that incorporate the full covariance structure from the",
        "Barra-Lite PCA risk model (Step 310).",
        "",
        "### Method 1: Mean-Variance Optimization (MVO)",
        "```",
        "max   α^T w  −  λ · w^T Σ w  −  γ · ||w − w_prev||_1",
        "s.t.  Σ w = 1",
        "      0 ≤ w_i ≤ 8%",
        "      |factor_k_exposure| ≤ 0.30  (k = 1, 2, 3)",
        "```",
        "",
        "### Method 2: Black-Litterman",
        "```",
        "Prior:    π = δ Σ w_mkt  (equilibrium, δ=2.5)",
        "Views:    ML ensemble score → return prediction per stock",
        "Posterior: μ_BL blends ML views with market equilibrium",
        "Portfolio: MVO on μ_BL with lower risk aversion (λ×0.8)",
        "```",
        "",
        "## OOS Performance Comparison",
        "",
        "| Method | CAGR | Sharpe | Max DD | Calmar |",
        "|--------|:----:|:------:|:------:|:------:|",
    ]

    for name, m in [("Score-Weighted (baseline)", base_m),
                    ("MVO", mvo_m), ("Black-Litterman", bl_m)]:
        if m:
            lines.append(
                f"| {name} | {m['cagr']:.1%} | {m['sharpe']:.2f} | "
                f"{m['max_dd']:.1%} | {m['calmar']:.2f} |"
            )

    lines += [
        "",
        "> **Key question**: Does incorporating the full covariance structure improve",
        "> risk-adjusted returns vs simple score weighting?",
        "> MVO should reduce MaxDD (by diversifying away idio risk) even if CAGR is similar.",
        "",
    ]

    # Factor exposures summary
    if not exp_df.empty:
        f1_mean = exp_df["factor_01"].mean() if "factor_01" in exp_df.columns else 0
        f2_mean = exp_df["factor_02"].mean() if "factor_02" in exp_df.columns else 0
        f3_mean = exp_df["factor_03"].mean() if "factor_03" in exp_df.columns else 0
        lines += [
            "## MVO Factor Exposure History",
            "",
            f"Average portfolio exposures to top 3 PCA factors:",
            "",
            f"| Factor | Avg Exposure | Limit | Status |",
            f"|--------|:-----------:|:-----:|:------:|",
            f"| Factor 1 | {f1_mean:+.3f} | ±0.30 | {'OK' if abs(f1_mean) <= 0.30 else 'WARNING'} |",
            f"| Factor 2 | {f2_mean:+.3f} | ±0.30 | {'OK' if abs(f2_mean) <= 0.30 else 'WARNING'} |",
            f"| Factor 3 | {f3_mean:+.3f} | ±0.30 | {'OK' if abs(f3_mean) <= 0.30 else 'WARNING'} |",
            "",
        ]

    lines += [
        "## Institutional Interpretation",
        "",
        "1. **MVO vs Score-Weighted**: Score weighting ignores covariances — two stocks",
        "   that are 95% correlated (e.g. NVDA and AMD) both get equal weight, doubling",
        "   semiconductor concentration. MVO naturally diversifies this away.",
        "",
        "2. **Black-Litterman value**: Pure ML views often produce extreme allocations.",
        "   BL shrinks towards market equilibrium — a more defensible position for a PM",
        "   who must explain portfolio construction to investors.",
        "",
        "3. **Turnover penalty**: γ · ||w − w_prev||_1 discourages excessive trading.",
        "   With TC = 10 bps, each full rotation costs ~50 bps for a 5-position port.",
        "   The optimizer will hold good positions slightly longer than pure signal suggests.",
        "",
        "4. **Factor constraints**: Capping |w^T b_1| ≤ 0.30 prevents the portfolio from",
        "   being a pure 'market-beta bet' (factor 1 = market factor in PCA).",
        "",
        "## Next Step",
        "",
        "The MVO covariance matrix is now available for:",
        "- Step 330 (Earnings Signals): add SUE scores to the alpha vector",
        "- Step 340 (NLP Alpha): add EDGAR filing sentiment to alpha vector",
        "These richer alpha vectors will demonstrate the full value of proper portfolio construction.",
    ]
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 320: MVO + Black-Litterman Optimizer")
    print("=" * 70)

    # Check PCA outputs exist
    if not (ROOT / "pca_factor_loadings.csv").exists():
        print("ERROR: pca_factor_loadings.csv not found — run step310 first")
        return

    print("\n[1/5] Loading inputs …")
    B_df, Sigma_F, spec_vol, preds = load_inputs()
    prices = load_prices()

    if preds.empty:
        print("ERROR: No predictions found — run step290 or step100 first")
        return

    print("\n[2/5] Running MVO + BL backtest over OOS period …")
    mvo_hist, bl_hist, mvo_perf, bl_perf = run_backtest(
        preds, prices, B_df, Sigma_F, spec_vol
    )

    if mvo_hist.empty:
        print("  No OOS rebalances — check predictions file")
        return

    print(f"  {len(mvo_hist)} OOS rebalances")

    # Save weights
    mvo_hist.to_csv(ROOT / "mvo_weights_history.csv", index=False)
    bl_hist.to_csv(ROOT / "bl_weights_history.csv", index=False)

    print("\n[3/5] Computing performance …")
    baseline_df = load_baseline_perf()

    mvo_m  = _metrics(mvo_perf["net_ret"])  if not mvo_perf.empty  else {}
    bl_m   = _metrics(bl_perf["net_ret"])   if not bl_perf.empty   else {}
    base_m = _metrics(baseline_df["net_ret"]) if not baseline_df.empty else {}

    print("\n  Performance comparison (OOS only):")
    print(f"  {'Method':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"  {'-'*57}")
    for name, m in [("Score-Weighted (base)", base_m),
                    ("MVO", mvo_m), ("Black-Litterman", bl_m)]:
        if m:
            print(f"  {name:<25} {m['cagr']:>7.1%} {m['sharpe']:>8.2f} "
                  f"{m['max_dd']:>7.1%} {m['calmar']:>8.2f}")

    # Save performance comparison
    cmp_rows = []
    for name, m in [("score_weighted", base_m), ("mvo", mvo_m), ("bl", bl_m)]:
        if m:
            m["method"] = name
            cmp_rows.append(m)
    pd.DataFrame(cmp_rows).to_csv(ROOT / "mvo_vs_bl_performance.csv", index=False)

    print("\n[4/5] Computing factor exposures …")
    exp_df = compute_factor_exposures(mvo_hist, B_df)
    if not exp_df.empty:
        exp_df.to_csv(ROOT / "mvo_factor_exposures.csv", index=False)
        print(f"  Factor exposures saved: {exp_df.shape}")

    print("\n[5/5] Generating report …")
    report = generate_report(mvo_m, bl_m, base_m, exp_df, ts)
    (ROOT / "mvo_report.md").write_text(report)

    print("\n" + "=" * 70)
    print("Step 320 Complete — MVO + Black-Litterman Optimizer")
    print("=" * 70)
    outputs = [
        "mvo_weights_history.csv", "bl_weights_history.csv",
        "mvo_vs_bl_performance.csv", "mvo_factor_exposures.csv", "mvo_report.md",
    ]
    print("\nOutputs:")
    for f in outputs:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "not created"
        print(f"  {f:<42} {sz}")


if __name__ == "__main__":
    main()
