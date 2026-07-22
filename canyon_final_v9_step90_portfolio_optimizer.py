#!/usr/bin/env python3
"""
Canyon v9 — Step 90: Mean-Variance Portfolio Optimizer (Barra-integrated)
=========================================================================
Replaces the greedy sector-neutral + greedy factor-neutral passes with a
single quadratic program (QP) using the Barra factor covariance matrix.

Objective (risk-adjusted):
  minimize  -w'α  +  λ · w'Vw
  where:
    α = IC-weighted composite scores  (expected return proxy)
    V = B·F·B' + Δ                    (Barra factor + specific covariance)
    λ = risk-aversion scalar           (default 2.0)

Constraints:
  1. Bounds:      w_i ∈ [0.02, 0.10] for longs
                  w_i ∈ [-0.10, -0.02] for shorts
  2. Net neutral: |Σ w_i| ≤ 0.10
  3. Gross long:  Σ_{long} w_i ≤ 0.55
  4. Gross short: Σ_{short} |w_i| ≤ 0.55
  5. Sector cap:  Σ_{i∈sector} |w_i| ≤ 0.30 (per side)
  6. Factor expo: |B_f' w| ≤ tol_f  for beta (0.20), size (0.40), momentum (0.40)
  7. Turnover:    Σ |w_i - w_i^prev| ≤ 0.60 (max 60% book turnover per rebalance)

Solver: scipy.optimize.minimize(method='SLSQP') — no extra dependencies needed.

Falls back to greedy ranking if optimization fails (infeasible/timeout).

Integration with step500:
  from canyon_final_v9_step90_portfolio_optimizer import run_optimizer
  weights, meta = run_optimizer(composite_df, sigs, top_long_greedy, top_short_greedy)
  # meta: {sharpe_ex_ante, factor_share, vol_est, n_longs, n_shorts, status}

Outputs (standalone run):
  portfolio_weights_today.csv   — w_i, side, sector, alpha_score, weight
  portfolio_optimizer_report.md — optimization summary

Usage:
  python3 canyon_final_v9_step90_portfolio_optimizer.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
OUT_WEIGHTS = ROOT / "portfolio_weights_today.csv"
OUT_REPORT  = ROOT / "portfolio_optimizer_report.md"

# ── Optimizer configuration ───────────────────────────────────────────────────

RISK_AVERSION  = 2.0     # λ: higher → more conservative weights
MAX_POS        = 0.10    # maximum weight per position
MIN_POS        = 0.02    # minimum weight if included
MAX_GROSS      = 0.55    # max gross long or short exposure
MAX_NET        = 0.10    # max |net| exposure (long + short sum)
SECTOR_CAP     = 0.30    # max per-sector (each side)
TURNOVER_CAP   = 0.60    # max fraction of book that can change per rebalance

FACTOR_TOLS = {
    "market_beta": 0.20,
    "size":        0.40,
    "momentum":    0.40,
    "low_vol":     0.50,
    "value":       0.50,
}

N_LONG  = 15
N_SHORT = 15


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_barra_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load B (exposures), F (factor cov), Δ (specific risk) from step88."""
    expo_path  = ROOT / "factor_exposures.csv"
    fcov_path  = ROOT / "factor_cov.csv"
    srisk_path = ROOT / "specific_risk.csv"

    expo, fcov, srisk = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    if expo_path.exists():
        try:
            expo = pd.read_csv(expo_path).set_index("ticker")
        except Exception:
            pass
    if fcov_path.exists():
        try:
            fcov = pd.read_csv(fcov_path, index_col=0)
        except Exception:
            pass
    if srisk_path.exists():
        try:
            srisk = pd.read_csv(srisk_path).set_index("ticker")["specific_vol"]
        except Exception:
            pass

    return expo, fcov, srisk


def load_prev_weights() -> pd.Series:
    """Previous portfolio weights for turnover constraint."""
    if not OUT_WEIGHTS.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(OUT_WEIGHTS)
        if "ticker" in df.columns and "weight" in df.columns:
            return df.set_index("ticker")["weight"]
    except Exception:
        pass
    return pd.Series(dtype=float)


def load_sector_map() -> dict[str, str]:
    for fname in ("regime_ml_scores.csv", "alpha_scores.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "ticker" in df.columns and "sector" in df.columns:
            return df.set_index("ticker")["sector"].to_dict()
    return {}


# ── Build Barra covariance ────────────────────────────────────────────────────

def build_cov_matrix(
    tickers:  list[str],
    expo:     pd.DataFrame,
    fcov:     pd.DataFrame,
    srisk:    pd.Series,
) -> np.ndarray:
    """V = B F B' + Δ  (n_stocks × n_stocks, annualized)."""
    n = len(tickers)

    if expo.empty or fcov.empty:
        # Fallback: diagonal using specific risk only (no factor cross-terms)
        spec = np.array([float(srisk.get(t, 0.20))**2 for t in tickers])
        return np.diag(spec)

    # Align factor columns between expo and fcov
    common_f = [f for f in expo.columns if f in fcov.columns]
    if not common_f:
        spec = np.array([float(srisk.get(t, 0.20))**2 for t in tickers])
        return np.diag(spec)

    B = expo.reindex(tickers)[common_f].fillna(0.0).values   # (n, k)
    F = fcov.loc[common_f, common_f].values                   # (k, k)
    spec = np.array([float(srisk.get(t, 0.20))**2 for t in tickers])

    V = B @ F @ B.T + np.diag(spec)
    # Ensure positive semi-definite (numerical fix)
    min_eig = np.linalg.eigvalsh(V).min()
    if min_eig < 1e-8:
        V += (abs(min_eig) + 1e-6) * np.eye(n)

    return V


# ── Core optimizer ────────────────────────────────────────────────────────────

def run_optimizer(
    composite_df:   pd.DataFrame,
    sigs:           dict,
    greedy_long:    list[str],
    greedy_short:   list[str],
    expo:           pd.DataFrame | None = None,
    fcov:           pd.DataFrame | None = None,
    srisk:          pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    """
    Run MVO portfolio optimizer.

    Returns:
      weights  — pd.Series(ticker → weight), positive=long, negative=short
      meta     — dict with optimization diagnostics
    """
    # ── Load Barra matrices if not provided
    if expo is None or fcov is None or srisk is None:
        expo, fcov, srisk = load_barra_matrices()

    prev_weights = load_prev_weights()
    sector_map   = load_sector_map()

    # ── Candidate universe: use greedy long+short + next-best buffer
    alpha_col = "composite_bl" if "composite_bl" in composite_df.columns else "composite"
    if alpha_col not in composite_df.columns:
        alpha_col = composite_df.columns[0]

    all_ranked  = composite_df[alpha_col].sort_values(ascending=False)
    buffer      = 10    # extra candidates beyond N_LONG/N_SHORT

    long_pool   = all_ranked.head(N_LONG  + buffer).index.tolist()
    short_pool  = all_ranked.tail(N_SHORT + buffer).index.tolist()

    # Remove overlaps; prefer greedy picks to stay in their pool
    long_pool   = list(dict.fromkeys(greedy_long  + [t for t in long_pool  if t not in greedy_long]))[:N_LONG  + buffer]
    short_pool  = list(dict.fromkeys(greedy_short + [t for t in short_pool if t not in greedy_short]))[:N_SHORT + buffer]

    tickers  = long_pool + short_pool
    n        = len(tickers)
    n_l      = len(long_pool)
    n_s      = len(short_pool)

    # ── Alpha vector (objective: maximize w'α)
    alpha = all_ranked.reindex(tickers).fillna(0.0).values.astype(float)
    alpha_l2 = np.linalg.norm(alpha) + 1e-9
    alpha    = alpha / alpha_l2   # normalize to unit norm for numerical stability

    # ── Covariance matrix
    V = build_cov_matrix(tickers, expo, fcov, srisk)

    # ── Bounds: long ∈ [min_pos, max_pos], short ∈ [-max_pos, -min_pos]
    lb = np.array([MIN_POS]*n_l + [-MAX_POS]*n_s)
    ub = np.array([MAX_POS]*n_l + [-MIN_POS]*n_s)

    # ── Initial guess: equal-weight within each side
    w0 = np.array([1.0/N_LONG]*n_l + [-1.0/N_SHORT]*n_s)
    w0 = np.clip(w0, lb, ub)

    # ── Constraints
    constraints = []

    # 1. Net exposure
    constraints.append({
        "type": "ineq",
        "fun":  lambda w: MAX_NET - abs(w.sum()),
    })

    # 2. Gross long ≤ MAX_GROSS
    constraints.append({
        "type": "ineq",
        "fun":  lambda w: MAX_GROSS - w[:n_l].sum(),
    })

    # 3. Gross short ≤ MAX_GROSS (|short side|)
    constraints.append({
        "type": "ineq",
        "fun":  lambda w: MAX_GROSS - (-w[n_l:]).sum(),
    })

    # 4. Sector caps (per side)
    sectors_seen: set[str] = set()
    for ticker, sector in sector_map.items():
        if sector in sectors_seen:
            continue
        sectors_seen.add(sector)
        long_idx  = [i for i, t in enumerate(long_pool)  if sector_map.get(t) == sector]
        short_idx = [n_l + i for i, t in enumerate(short_pool) if sector_map.get(t) == sector]
        if long_idx:
            constraints.append({
                "type": "ineq",
                "fun":  lambda w, idx=long_idx: SECTOR_CAP - sum(w[i] for i in idx),
            })
        if short_idx:
            constraints.append({
                "type": "ineq",
                "fun":  lambda w, idx=short_idx: SECTOR_CAP - sum(-w[i] for i in idx),
            })

    # 5. Factor exposure bounds
    if not expo.empty:
        for fname, tol in FACTOR_TOLS.items():
            if fname not in expo.columns:
                continue
            b_f = expo.reindex(tickers)[fname].fillna(0.0).values
            constraints.append({
                "type": "ineq",
                "fun":  lambda w, bf=b_f, t=tol: t - abs(float(bf @ w)),
            })

    # 6. Turnover constraint
    prev_w = prev_weights.reindex(tickers).fillna(0.0).values
    constraints.append({
        "type": "ineq",
        "fun":  lambda w, pw=prev_w: TURNOVER_CAP - np.abs(w - pw).sum(),
    })

    # ── Objective: minimize -w'α + λ * w'Vw
    def objective(w: np.ndarray) -> float:
        return float(-w @ alpha + RISK_AVERSION * w @ V @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return -alpha + 2.0 * RISK_AVERSION * V @ w

    # ── Solve
    try:
        result = minimize(
            fun=objective,
            x0=w0,
            jac=jac,
            method="SLSQP",
            bounds=Bounds(lb=lb, ub=ub),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )
        status   = "optimal" if result.success else "suboptimal"
        w_final  = result.x.copy()
    except Exception as exc:
        print(f"  [Optimizer] SLSQP failed: {exc} — using greedy fallback")
        w_final = w0.copy()
        status  = "fallback_error"

    # ── Post-processing: zero out near-zero positions
    threshold = MIN_POS * 0.5
    w_final[np.abs(w_final) < threshold] = 0.0

    # Re-normalize to target gross = 1.0 (50% long + 50% short)
    gross_l = w_final[:n_l].sum()
    gross_s = (-w_final[n_l:]).sum()
    if gross_l > 0:
        w_final[:n_l] *= (0.5 / gross_l)
    if gross_s > 0:
        w_final[n_l:] *= (0.5 / gross_s)

    # ── Build output Series
    weights = pd.Series(dict(zip(tickers, w_final))).sort_values(ascending=False)
    weights = weights[weights.abs() > 1e-4]

    # ── Diagnostics
    w_arr  = weights.reindex(tickers).fillna(0.0).values
    port_var   = float(w_arr @ V @ w_arr)
    port_vol   = float(np.sqrt(max(port_var, 0.0)))
    alpha_exp  = float(w_arr @ alpha) * alpha_l2   # un-normalize
    sharpe_ex  = alpha_exp / (port_vol + 1e-9) if port_vol > 0 else 0.0

    meta = {
        "status":           status,
        "n_longs":          int((weights > 0.005).sum()),
        "n_shorts":         int((weights < -0.005).sum()),
        "portfolio_vol":    round(port_vol, 4),
        "alpha_exposure":   round(alpha_exp, 4),
        "sharpe_ex_ante":   round(sharpe_ex, 3),
        "gross_long":       round(w_final[:n_l].sum(), 4),
        "gross_short":      round((-w_final[n_l:]).sum(), 4),
        "net_exposure":     round(w_final.sum(), 4),
        "turnover":         round(float(np.abs(w_final - prev_w).sum()), 4),
        "date":             datetime.now().strftime("%Y-%m-%d"),
    }

    return weights, meta


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_weights(weights: pd.Series, meta: dict,
                 composite_df: pd.DataFrame) -> None:
    alpha_col = next((c for c in ("composite_bl","composite") if c in composite_df.columns),
                     composite_df.columns[0])
    sector_map = load_sector_map()
    rows = []
    for tk, w in weights.items():
        if abs(w) < 1e-4:
            continue
        rows.append({
            "ticker":      tk,
            "weight":      round(w, 6),
            "side":        "LONG" if w > 0 else "SHORT",
            "alpha_score": round(float(composite_df[alpha_col].get(tk, 0)), 4),
            "sector":      sector_map.get(tk, ""),
            "date":        meta["date"],
        })
    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    df.to_csv(OUT_WEIGHTS, index=False)
    print(f"  [Optimizer] Saved {len(df)} positions → {OUT_WEIGHTS.name}")


def write_report(weights: pd.Series, meta: dict) -> None:
    longs  = weights[weights >  0.005].sort_values(ascending=False)
    shorts = weights[weights < -0.005].sort_values(ascending=True)

    def _pct(v): return f"{float(v):.2%}" if v == v else "—"

    long_rows  = "".join(f"| **{t}** | {_pct(w)} |\n"
                         for t, w in longs.items())
    short_rows = "".join(f"| **{t}** | {_pct(abs(w))} |\n"
                         for t, w in shorts.items())

    status_icon = "✓" if meta["status"] == "optimal" else "⚠"
    report = f"""# Portfolio Optimizer Report — {meta['date']}

## Optimization Result

| Metric | Value |
|--------|-------|
| Status | {status_icon} {meta['status']} |
| Ex-ante Portfolio Vol | {_pct(meta['portfolio_vol'])} |
| Ex-ante Sharpe | {meta['sharpe_ex_ante']:.3f} |
| Gross Long | {_pct(meta['gross_long'])} |
| Gross Short | {_pct(meta['gross_short'])} |
| Net Exposure | {_pct(meta['net_exposure'])} |
| Turnover | {_pct(meta['turnover'])} |
| Positions | {meta['n_longs']} long · {meta['n_shorts']} short |

## Long Positions (Optimized Weights)

| Ticker | Weight |
|--------|:------:|
{long_rows}
## Short Positions (Optimized Weights)

| Ticker | Weight |
|--------|:------:|
{short_rows}
## Constraints Applied

- Risk aversion λ = {RISK_AVERSION}
- Max position: ±{MAX_POS:.0%}
- Sector cap: {SECTOR_CAP:.0%} per side
- Beta tolerance: ±{FACTOR_TOLS['market_beta']:.2f}
- Turnover cap: {TURNOVER_CAP:.0%} per rebalance
- Covariance: Barra B·F·B' + Δ  (step88)

---
*Optimizer: scipy SLSQP. Fallback: equal-weight greedy picks.*
"""
    OUT_REPORT.write_text(report)
    print(f"  [Optimizer] Report → {OUT_REPORT.name}")


# ── Standalone entry point ────────────────────────────────────────────────────

def run_standalone() -> None:
    """Run optimizer from today's composite scores (no step500 context)."""
    alpha_path = ROOT / "alpha_scores.csv"
    if not alpha_path.exists():
        print("  [Optimizer] No alpha_scores.csv found — run step500 first")
        return

    df = pd.read_csv(alpha_path)
    if "ticker" not in df.columns:
        print("  [Optimizer] alpha_scores.csv missing ticker column")
        return

    score_col = next((c for c in ("alpha_score","composite","sig_fundamental")
                      if c in df.columns), None)
    if not score_col:
        print("  [Optimizer] No score column found")
        return

    df = df.set_index("ticker")
    composite_df = pd.DataFrame({"composite": pd.to_numeric(df[score_col], errors="coerce")})
    composite_df = composite_df.dropna().sort_values("composite", ascending=False)

    greedy_long  = composite_df.head(N_LONG).index.tolist()
    greedy_short = composite_df.tail(N_SHORT).index.tolist()

    expo, fcov, srisk = load_barra_matrices()
    weights, meta = run_optimizer(
        composite_df, {},
        greedy_long, greedy_short,
        expo=expo, fcov=fcov, srisk=srisk,
    )

    save_weights(weights, meta, composite_df)
    write_report(weights, meta)

    print(f"\n  Status:      {meta['status']}")
    print(f"  Vol (ex-ante): {meta['portfolio_vol']:.1%}")
    print(f"  Sharpe:        {meta['sharpe_ex_ante']:.3f}")
    print(f"  Net exposure:  {meta['net_exposure']:+.3f}")
    print(f"  Turnover:      {meta['turnover']:.1%}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Canyon v9 — Portfolio Optimizer  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60 + "\n")
    run_standalone()
    print("\n" + "=" * 60)
    print("Step 90 Complete")
    print("=" * 60)
