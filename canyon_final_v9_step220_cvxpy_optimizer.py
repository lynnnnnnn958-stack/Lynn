#!/usr/bin/env python3
"""
Canyon v9  Step 220 — cvxpy Mean-Variance Portfolio Optimizer
=============================================================
Replaces the rule-based weight truncation in step63 with proper
convex optimization using cvxpy + CLARABEL solver.

MATH
----
Solve:
    maximize   α^T w  −  λ · w^T Σ w  −  γ · ‖w − w_prev‖₁

    subject to:
        Σᵢ wᵢ = 1                      (fully invested)
        wᵢ ≥ 0                          (long-only)
        wᵢ ≤ MAX_POSITION               (concentration limit)
        Σᵢ∈sector_j wᵢ ≤ SECTOR_CAP   (sector limit)

Where:
    α   = signal-adjusted expected excess return (from alpha_scores.csv)
    Σ   = LedoitWolf-shrunk covariance matrix (1-year lookback)
    λ   = risk-aversion (tuned to target annual vol ≈ 15%)
    γ   = turnover penalty (reduces unnecessary rebalancing)

Three portfolios solved:
    1. Alpha-MV:      α-driven MV with turnover penalty (main output)
    2. Max-Sharpe:    max expected return per unit risk
    3. Min-Variance:  lowest risk regardless of alpha

INPUTS
------
    alpha_scores.csv        — step87 aggregated alpha (0–100 per ticker)
    sp500_price_cache.csv   — daily prices for covariance estimation
    sector_map.csv          — GICS sector per ticker
    cvxpy_prev_weights.csv  — previous run weights (for turnover penalty)

OUTPUTS
-------
    cvxpy_weights.csv           — optimized weights (3 portfolios)
    cvxpy_optimizer_report.md   — full explanation with diagnostics
    cvxpy_prev_weights.csv      — saved for next run's turnover penalty
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False
    print("[ERROR] cvxpy not installed. Run: pip install cvxpy")

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ROOT = Path(__file__).parent

# ── Parameters ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 252          # covariance estimation window (1 year)
MAX_POSITION    = 0.08         # max weight per ticker (8%)
SECTOR_CAP      = 0.35         # max sector weight (35%)
MIN_TICKERS     = 10           # minimum tickers to run optimizer
TOP_N           = 40           # only optimize over top-N alpha tickers
RISK_FREE       = 0.053        # annual risk-free rate
LAMBDA_RISK     = 2.0          # risk-aversion parameter
GAMMA_TURNOVER  = 0.005        # turnover penalty (0.5% per unit of ∆w)
ANN_FACTOR      = 252          # trading days per year

# Alpha → expected return mapping:
#   alpha_score 50 (neutral) = 0 excess return
#   alpha_score 100 (max)    = +ALPHA_SCALE annualized excess return
#   alpha_score 0  (min)     = -ALPHA_SCALE annualized excess return
ALPHA_SCALE     = 0.25         # 25% annualized at max alpha


# =============================================================================
# 1. Data loaders
# =============================================================================

def load_alpha() -> pd.DataFrame:
    path = ROOT / "alpha_scores.csv"
    if not path.exists():
        raise FileNotFoundError("alpha_scores.csv not found — run step87 first.")
    df = pd.read_csv(path)
    req = {"ticker", "alpha_score"}
    if not req.issubset(df.columns):
        raise ValueError(f"alpha_scores.csv missing columns: {req - set(df.columns)}")
    df = df.dropna(subset=["ticker","alpha_score"])
    df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce")
    return df.dropna(subset=["alpha_score"])


def load_prices() -> pd.DataFrame:
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
        raise FileNotFoundError("sp500_price_cache.csv not found.")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index().dropna(axis=1, how="all")


def load_sector_map() -> dict[str, str]:
    """Returns ticker → GICS sector mapping."""
    # Try sector_map.csv first
    for fname in ["sector_map.csv", "sp500_sector_map.csv"]:
        path = ROOT / fname
        if path.exists():
            df = pd.read_csv(path)
            if "ticker" in df.columns and "sector" in df.columns:
                return dict(zip(df["ticker"], df["sector"]))
    # Fall back to alpha_scores.csv if it has sector
    alpha = load_alpha()
    if "sector" in alpha.columns:
        return dict(zip(alpha["ticker"], alpha["sector"].fillna("Unknown")))
    return {}


def load_prev_weights(tickers: list[str]) -> pd.Series:
    """Previous run weights for turnover penalty. Returns zeros if missing."""
    path = ROOT / "cvxpy_prev_weights.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, index_col=0)
            col = "alpha_mv" if "alpha_mv" in df.columns else df.columns[0]
            w = df[col].reindex(tickers).fillna(0.0)
            # Re-normalise in case the ticker list changed
            w = w.clip(lower=0)
            total = w.sum()
            return w / total if total > 0 else pd.Series(0.0, index=tickers)
        except Exception:
            pass
    return pd.Series(0.0, index=tickers)


# =============================================================================
# 2. Covariance estimation
# =============================================================================

def estimate_covariance(prices: pd.DataFrame, tickers: list[str],
                        lookback: int = LOOKBACK_DAYS) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance matrix.
    If sklearn not available, falls back to sample covariance.
    """
    avail = [t for t in tickers if t in prices.columns]
    px = prices[avail].tail(lookback + 1)

    log_rets = np.log(px / px.shift(1)).dropna()
    if len(log_rets) < 30:
        raise ValueError(f"Insufficient price history: {len(log_rets)} days")

    R = log_rets.values   # (T × N)

    if _HAS_SKLEARN:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R)
        cov_daily = lw.covariance_
    else:
        cov_daily = np.cov(R.T)

    return cov_daily * ANN_FACTOR   # annualised


# =============================================================================
# 3. Alpha → expected return mapping
# =============================================================================

def alpha_to_mu(alpha_scores: pd.Series, scale: float = ALPHA_SCALE) -> pd.Series:
    """
    Map alpha score (0–100) to annualised expected excess return.

        μ = (alpha_score − 50) / 50 × scale

    Score 100 → +scale  (e.g. +25% pa)
    Score 50  → 0       (neutral)
    Score 0   → −scale  (e.g. −25% pa)
    """
    return (alpha_scores - 50.0) / 50.0 * scale


# =============================================================================
# 4. cvxpy optimizers
# =============================================================================

def solve_alpha_mv(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    sector_matrix: np.ndarray,
    lambda_risk: float = LAMBDA_RISK,
    gamma_tc: float = GAMMA_TURNOVER,
    max_pos: float = MAX_POSITION,
    sector_cap: float = SECTOR_CAP,
) -> np.ndarray | None:
    """
    Alpha-driven mean-variance with turnover penalty.

    maximize  μ^T w  −  λ · w^T Σ w  −  γ · ‖w − w_prev‖₁
    """
    n = len(mu)
    w = cp.Variable(n, name="w")

    # Turnover L1 via auxiliary variable
    z = cp.Variable(n, name="z_abs")   # z ≥ |w − w_prev|

    portfolio_return = mu @ w
    portfolio_risk   = cp.quad_form(w, cov)
    turnover_cost    = gamma_tc * cp.sum(z)

    objective = cp.Maximize(portfolio_return - lambda_risk * portfolio_risk - turnover_cost)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_pos,
        z >= w - w_prev,
        z >= w_prev - w,
    ]
    # Sector caps
    if sector_matrix.shape[0] > 0:
        constraints.append(sector_matrix @ w <= sector_cap)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None

    if w.value is None or prob.status not in ("optimal","optimal_inaccurate"):
        return None
    return np.clip(w.value, 0, None)


def solve_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = RISK_FREE,
    max_pos: float = MAX_POSITION,
) -> np.ndarray | None:
    """
    Maximum Sharpe ratio via Markowitz-Tobin parameterisation.
    Uses the auxiliary variable y = w/κ trick.
    """
    n = len(mu)
    mu_ex = mu - rf  # excess return over risk-free

    # If no positive expected return, can't form max-Sharpe
    if np.all(mu_ex <= 0):
        return None

    y = cp.Variable(n, name="y")   # y = w / κ
    kappa = cp.Variable(name="kappa", nonneg=True)

    objective  = cp.Minimize(cp.quad_form(y, cov))
    constraints = [
        mu_ex @ y == 1,
        cp.sum(y) == kappa,
        y >= 0,
        y <= max_pos * kappa,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None

    if y.value is None or kappa.value is None or kappa.value < 1e-8:
        return None
    w = y.value / kappa.value
    return np.clip(w, 0, None)


def solve_min_variance(
    cov: np.ndarray,
    max_pos: float = MAX_POSITION,
) -> np.ndarray | None:
    """Minimum variance portfolio."""
    n = cov.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= 0, w <= max_pos]
    )
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if w.value is None:
        return None
    return np.clip(w.value, 0, None)


# =============================================================================
# 5. Diagnostics
# =============================================================================

def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                    rf: float = RISK_FREE) -> dict:
    ret    = float(mu @ w)
    var    = float(w @ cov @ w)
    vol    = float(np.sqrt(max(var, 1e-12)))
    sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
    n_eff  = 1.0 / float(np.sum(w**2)) if np.sum(w**2) > 0 else 0
    return {"exp_return": ret, "vol": vol, "sharpe": sharpe, "n_eff": n_eff}


def rank_ic(alpha: pd.Series, fwd_ret: pd.Series) -> float:
    """Rank information coefficient between alpha scores and forward returns."""
    common = alpha.index.intersection(fwd_ret.index)
    if len(common) < 10:
        return float("nan")
    rho, _ = spearmanr(alpha[common], fwd_ret[common])
    return float(rho)


# =============================================================================
# 6. Sector constraint matrix builder
# =============================================================================

def build_sector_matrix(tickers: list[str],
                        sector_map: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """
    Returns (S × N) matrix where S[j, i] = 1 if ticker i is in sector j.
    """
    sectors = sorted({sector_map.get(t, "Unknown") for t in tickers})
    S = np.zeros((len(sectors), len(tickers)))
    for j, sec in enumerate(sectors):
        for i, tkr in enumerate(tickers):
            if sector_map.get(tkr, "Unknown") == sec:
                S[j, i] = 1.0
    return S, sectors


# =============================================================================
# 7. Report writer
# =============================================================================

def write_report(
    tickers: list[str],
    alpha_mv_w: np.ndarray | None,
    sharpe_w:   np.ndarray | None,
    minvar_w:   np.ndarray | None,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha_scores: pd.Series,
    sector_map: dict[str, str],
    n_tickers_total: int,
) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Canyon v9 — cvxpy Portfolio Optimizer Report (Step 220)",
        f"Generated: {ts}\n",
        "## What this optimizer does",
        "",
        "Replaces rule-based weight truncation with formal mean-variance optimisation (cvxpy + CLARABEL).",
        "",
        "**Formula (Alpha-MV portfolio):**",
        "```",
        "maximize  α^T w  −  λ · w^T Σ w  −  γ · ‖w − w_prev‖₁",
        "subject to:  Σ wᵢ = 1,  wᵢ ≥ 0,  wᵢ ≤ 8%,  sector ≤ 35%",
        "```",
        "",
        f"- **λ (risk aversion)** = {LAMBDA_RISK}  — higher = more conservative",
        f"- **γ (turnover penalty)** = {GAMMA_TURNOVER}  — reduces needless rebalancing",
        f"- **Alpha scale** = {ALPHA_SCALE*100:.0f}% annualised at max alpha score",
        "",
        "## Universe",
        "",
        f"- Total alpha-scored tickers: {n_tickers_total}",
        f"- Tickers entering optimizer (top {TOP_N} by alpha): {len(tickers)}",
        f"- Price history window: {LOOKBACK_DAYS} days",
        "",
        "## Portfolio Comparison",
        "",
        "| Portfolio | Expected Return | Annual Vol | Sharpe | Eff. N |",
        "|-----------|----------------|------------|--------|--------|",
    ]

    for label, w in [("Alpha-MV (main)", alpha_mv_w),
                     ("Max-Sharpe",       sharpe_w),
                     ("Min-Variance",     minvar_w)]:
        if w is not None:
            s = portfolio_stats(w, mu, cov)
            lines.append(
                f"| {label} | {s['exp_return']*100:+.1f}% | "
                f"{s['vol']*100:.1f}% | {s['sharpe']:.2f} | {s['n_eff']:.1f} |"
            )
        else:
            lines.append(f"| {label} | — | — | — | — |")

    lines += ["", "## Top Holdings — Alpha-MV Portfolio", ""]
    if alpha_mv_w is not None:
        w_series = pd.Series(alpha_mv_w, index=tickers).sort_values(ascending=False)
        lines.append("| # | Ticker | Weight | Alpha Score | Sector | μ (exp. ret.) |")
        lines.append("|---|--------|--------|-------------|--------|---------------|")
        for rank, (tkr, wt) in enumerate(w_series[w_series > 0.001].items(), 1):
            a = float(alpha_scores.get(tkr, 50))
            m = float(mu[tickers.index(tkr)])
            sec = sector_map.get(tkr, "—")[:20]
            lines.append(f"| {rank} | **{tkr}** | {wt*100:.2f}% | {a:.1f} | {sec} | {m*100:+.1f}% |")

    lines += ["", "## Sector Weights — Alpha-MV Portfolio", ""]
    if alpha_mv_w is not None:
        sect_w: dict[str, float] = {}
        for i, tkr in enumerate(tickers):
            sec = sector_map.get(tkr, "Unknown")
            sect_w[sec] = sect_w.get(sec, 0.0) + float(alpha_mv_w[i])
        lines.append("| Sector | Weight |")
        lines.append("|--------|--------|")
        for sec, sw in sorted(sect_w.items(), key=lambda x: -x[1]):
            if sw > 0.001:
                flag = " ⚠️" if sw > SECTOR_CAP * 0.9 else ""
                lines.append(f"| {sec} | {sw*100:.1f}%{flag} |")

    lines += [
        "",
        "## Methodology Notes",
        "",
        "1. **Covariance**: Ledoit-Wolf shrinkage (reduces estimation error in small samples).",
        "2. **Alpha signal**: `μ = (alpha_score − 50) / 50 × 25%` — neutral score = 0 expected excess return.",
        "3. **Turnover penalty**: L1 penalty on weight changes prevents excessive rebalancing.",
        "4. **Solver**: CLARABEL (interior-point, production-grade convex solver).",
        "",
        "⚠️ **Data warning**: Covariance estimated from survivorship-biased price cache.",
        "Expected returns use proxy alpha scores, not statistically validated forward IC.",
        "Treat all outputs as research-grade, not production-grade.",
    ]

    (ROOT / "cvxpy_optimizer_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 8. Main pipeline
# =============================================================================

def run(top_n: int = TOP_N, dry_run: bool = False) -> dict:
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 220: cvxpy Optimizer  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"{'='*65}")

    if not _HAS_CVXPY:
        print("[ABORT] cvxpy not installed.")
        return {}

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading alpha scores & prices …")
    alpha_df  = load_alpha()
    prices    = load_prices()
    sector_map = load_sector_map()

    # Select top-N tickers by alpha score
    top_alpha = (
        alpha_df
        .sort_values("alpha_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    # Keep only tickers with sufficient price history
    has_prices = [t for t in top_alpha["ticker"] if t in prices.columns
                  and prices[t].notna().sum() >= LOOKBACK_DAYS + 10]
    top_alpha  = top_alpha[top_alpha["ticker"].isin(has_prices)].head(top_n)
    tickers    = list(top_alpha["ticker"])
    n_tickers_total = len(alpha_df)

    print(f"  Top-{top_n} alpha → {len(tickers)} tickers with sufficient price history")
    if len(tickers) < MIN_TICKERS:
        print(f"  [SKIP] Only {len(tickers)} tickers — need ≥ {MIN_TICKERS}")
        return {}

    # ── 2. Covariance ─────────────────────────────────────────────────────────
    print("\n[2/6] Estimating LedoitWolf covariance …")
    try:
        cov = estimate_covariance(prices, tickers)
        print(f"  Σ shape: {cov.shape}  (annualised)  "
              f"avg vol: {np.sqrt(np.diag(cov)).mean()*100:.1f}%")
    except Exception as e:
        print(f"  [ERROR] Covariance failed: {e}")
        return {}

    # ── 3. Expected returns ────────────────────────────────────────────────────
    print("\n[3/6] Mapping alpha scores → expected returns …")
    alpha_s = top_alpha.set_index("ticker")["alpha_score"]
    mu      = alpha_to_mu(alpha_s[tickers]).values
    print(f"  μ range: {mu.min()*100:+.1f}% to {mu.max()*100:+.1f}%  "
          f"(scale = ±{ALPHA_SCALE*100:.0f}%)")

    # ── 4. Previous weights ────────────────────────────────────────────────────
    w_prev = load_prev_weights(tickers).values

    # ── 5. Sector constraint matrix ────────────────────────────────────────────
    print("\n[4/6] Building sector constraints …")
    S, sectors = build_sector_matrix(tickers, sector_map)
    print(f"  {len(sectors)} sectors · cap = {SECTOR_CAP*100:.0f}%")

    # ── 6. Solve three portfolios ──────────────────────────────────────────────
    print("\n[5/6] Solving cvxpy optimizations …")

    if dry_run:
        print("  [DRY-RUN] Skipping solver calls.")
        return {"status": "dry_run"}

    print("  → Alpha-MV portfolio …", end=" ", flush=True)
    w_amv = solve_alpha_mv(mu, cov, w_prev, S)
    print("OK" if w_amv is not None else "FAILED")

    print("  → Max-Sharpe portfolio …", end=" ", flush=True)
    w_ms = solve_max_sharpe(mu, cov)
    print("OK" if w_ms is not None else "FAILED")

    print("  → Min-Variance portfolio …", end=" ", flush=True)
    w_mv = solve_min_variance(cov)
    print("OK" if w_mv is not None else "FAILED")

    # ── 7. Write outputs ───────────────────────────────────────────────────────
    print("\n[6/6] Writing outputs …")

    # Normalise weights to sum to 1
    def _norm(w):
        if w is None: return None
        s = w.sum()
        return w / s if s > 1e-8 else None

    w_amv = _norm(w_amv)
    w_ms  = _norm(w_ms)
    w_mv  = _norm(w_mv)

    rows = []
    for i, tkr in enumerate(tickers):
        rows.append({
            "ticker":   tkr,
            "alpha_mv": round(float(w_amv[i]), 6) if w_amv is not None else None,
            "max_sharpe": round(float(w_ms[i]), 6) if w_ms is not None else None,
            "min_var":  round(float(w_mv[i]), 6) if w_mv is not None else None,
            "alpha_score": float(alpha_s.get(tkr, 50)),
            "mu_annualized": round(float(mu[i]), 4),
            "sector":   sector_map.get(tkr, "Unknown"),
        })

    df_out = pd.DataFrame(rows).sort_values("alpha_mv", ascending=False)
    df_out.to_csv(ROOT / "cvxpy_weights.csv", index=False)
    print(f"  [written] cvxpy_weights.csv  ({len(df_out)} rows)")

    # Save as new prev weights for next run
    df_out.set_index("ticker")[["alpha_mv"]].rename(
        columns={"alpha_mv": "alpha_mv"}
    ).to_csv(ROOT / "cvxpy_prev_weights.csv")
    print(f"  [written] cvxpy_prev_weights.csv")

    # Report
    write_report(tickers, w_amv, w_ms, w_mv, mu, cov, alpha_s, sector_map, n_tickers_total)
    print(f"  [written] cvxpy_optimizer_report.md")

    # Summary stats
    result = {}
    if w_amv is not None:
        s = portfolio_stats(w_amv, mu, cov)
        result = {**s, "n_tickers": int((w_amv > 0.001).sum()), "status": "OK"}
        print(f"\n  Alpha-MV: ret={s['exp_return']*100:+.1f}%  "
              f"vol={s['vol']*100:.1f}%  sharpe={s['sharpe']:.2f}  "
              f"n={result['n_tickers']} holdings")

    print(f"\n{'─'*65}")
    top5 = df_out[df_out["alpha_mv"].notna()].head(5)
    print("  Top 5 holdings (Alpha-MV):")
    for _, r in top5.iterrows():
        print(f"    {r['ticker']:6s}  {r['alpha_mv']*100:5.2f}%  α={r['alpha_score']:.1f}  {r['sector'][:20]}")
    print(f"{'─'*65}\n")

    return result


# =============================================================================
# 9. CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cvxpy MV Portfolio Optimizer — Canyon v9 Step 220")
    parser.add_argument("--top",     type=int,   default=TOP_N,          help=f"Top-N alpha tickers (default {TOP_N})")
    parser.add_argument("--lambda",  type=float, default=LAMBDA_RISK,    dest="lam",  help="Risk aversion λ")
    parser.add_argument("--gamma",   type=float, default=GAMMA_TURNOVER, help="Turnover penalty γ")
    parser.add_argument("--max-pos", type=float, default=MAX_POSITION,   help="Max position weight")
    parser.add_argument("--dry-run", action="store_true", help="Skip solver, validate data only")
    args = parser.parse_args()

    LAMBDA_RISK    = args.lam
    GAMMA_TURNOVER = args.gamma
    MAX_POSITION   = args.max_pos

    result = run(top_n=args.top, dry_run=args.dry_run)
    import sys
    sys.exit(0 if result.get("status") in ("OK","dry_run") else 1)
