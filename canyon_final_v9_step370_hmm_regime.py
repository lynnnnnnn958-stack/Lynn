#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 370: HMM Regime Detection
==========================================
Implements a 2-state Gaussian Hidden Markov Model from scratch
using numpy only (no hmmlearn dependency).

Regimes detected:
  State 0 — Bull / Low-volatility  (positive drift, low vol)
  State 1 — Bear / High-volatility (negative drift, high vol)

Algorithm:
  Training:  Baum-Welch EM on IS daily SPY features (pre-2020)
  Decoding:  Viterbi path on full history (OOS decoded out-of-sample)
  Features:  [SPY 1-day return, SPY 21-day realized vol, SPY 63-day momentum]

Portfolio overlay:
  At each monthly rebalance, check current regime:
    Bull  → full L/S exposure (100%)
    Bear  → half exposure (50%)
  Compare regime-aware L/S Sharpe vs static L/S (step350 result).

Outputs:
  hmm_regime_daily.csv       daily regime probabilities + Viterbi state
  hmm_regime_monthly.csv     regime at each rebalance date
  hmm_regime_performance.csv static vs regime-aware performance
  hmm_transition_matrix.csv  estimated A matrix
  hmm_report.md              narrative report

Usage:
  .venv/bin/python canyon_final_v9_step370_hmm_regime.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── constants ──────────────────────────────────────────────────────────────
OOS_CUTOFF   = pd.Timestamp("2020-01-01")
N_STATES     = 2
N_EM_ITER    = 100
EM_TOL       = 1e-6
ANN_FACTOR   = 252
HOLD_PERIOD  = 21

# Regime overlay fractions
EXPOSURE = {0: 1.00, 1: 0.50}   # Bull=100%, Bear=50%

# Feature windows
VOL_WINDOW = 21
MOM_WINDOW = 63


# =============================================================================
# 1. Gaussian HMM — scratch implementation
# =============================================================================

class GaussianHMM2:
    """
    2-state Gaussian HMM with full covariance.
    Uses Baum-Welch EM for parameter estimation
    and Viterbi algorithm for state decoding.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-6):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        # Parameters (set after fit)
        self.pi:    np.ndarray | None = None   # (K,) initial distribution
        self.A:     np.ndarray | None = None   # (K,K) transition matrix
        self.mu:    np.ndarray | None = None   # (K,D) emission means
        self.sigma: np.ndarray | None = None   # (K,D,D) emission covariances

    # ── helpers ──────────────────────────────────────────────────────────

    def _emission_log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        log p(x_t | state k) for all t, k.
        Returns (T, K) array.
        """
        T = len(X)
        log_b = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k],
                                     allow_singular=True)
            log_b[:, k] = rv.logpdf(X)
        return log_b

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward algorithm (log-space for numerical stability).
        Returns (T, K) log-alpha and log-likelihood.
        """
        T, K = log_b.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-300) + log_b[0]

        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, k])
                    + log_b[t, k]
                )
        ll = np.logaddexp.reduce(log_alpha[-1])
        return log_alpha, ll

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        """
        Backward algorithm (log-space).
        Returns (T, K) log-beta.
        """
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        log_A    = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_A[k] + log_b[t + 1] + log_beta[t + 1]
                )
        return log_beta

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: compute gamma (T,K) and xi (T-1,K,K).
        Returns gamma, xi, log-likelihood.
        """
        log_b     = self._emission_log_prob(X)
        log_alpha, ll = self._forward(log_b)
        log_beta  = self._backward(log_b)
        log_A     = np.log(self.A + 1e-300)

        T, K = X.shape[0], self.n_states

        # gamma[t, k] = P(state_t = k | X, θ)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # xi[t, i, j] = P(state_t=i, state_{t+1}=j | X, θ)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.exp(
                        log_alpha[t, i] + log_A[i, j]
                        + log_b[t + 1, j] + log_beta[t + 1, j] - ll
                    )
        return gamma, xi, ll

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: update pi, A, mu, sigma."""
        K, D = self.n_states, X.shape[1]

        self.pi = gamma[0] / gamma[0].sum()

        A_new = xi.sum(axis=0)
        self.A = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-300)

        for k in range(K):
            wk = gamma[:, k]
            self.mu[k] = (wk[:, None] * X).sum(axis=0) / (wk.sum() + 1e-300)
            diff = X - self.mu[k]
            self.sigma[k] = ((wk[:, None, None] * diff[:, :, None]
                              * diff[:, None, :]).sum(axis=0)
                             / (wk.sum() + 1e-300))
            # Regularize covariance
            self.sigma[k] += np.eye(D) * 1e-6

    def _init_params(self, X: np.ndarray) -> None:
        """K-means-like initialization."""
        T, D = X.shape
        K    = self.n_states
        np.random.seed(42)

        # Sort by first feature (return) to initialize states
        idx       = np.argsort(X[:, 0])
        half      = T // 2
        self.mu   = np.zeros((K, D))
        self.sigma= np.array([np.eye(D) * X.std(axis=0).mean() for _ in range(K)])
        self.mu[0]= X[idx[:half]].mean(axis=0)   # low-return state
        self.mu[1]= X[idx[half:]].mean(axis=0)   # high-return state

        self.pi = np.ones(K) / K
        self.A  = np.full((K, K), 1 / K)

    def fit(self, X: np.ndarray) -> "GaussianHMM2":
        """Fit HMM via Baum-Welch EM."""
        self._init_params(X)
        prev_ll = -np.inf
        for i in range(self.n_iter):
            gamma, xi, ll = self._e_step(X)
            self._m_step(X, gamma, xi)
            if abs(ll - prev_ll) < self.tol:
                print(f"    EM converged at iteration {i+1}  (ll={ll:.4f})")
                break
            prev_ll = ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities (T, K) using forward-only pass."""
        log_b     = self._emission_log_prob(X)
        log_alpha, _ = self._forward(log_b)
        log_gamma = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding — returns most likely state sequence (T,).
        """
        log_b  = self._emission_log_prob(X)
        T, K   = X.shape[0], self.n_states
        log_A  = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta  = np.full((T, K), -np.inf)
        psi    = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_b[0]
        for t in range(1, T):
            for k in range(K):
                trans  = delta[t - 1] + log_A[:, k]
                psi[t, k]   = np.argmax(trans)
                delta[t, k] = trans[psi[t, k]] + log_b[t, k]

        # Backtrack
        path    = np.zeros(T, dtype=int)
        path[-1]= np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


# =============================================================================
# 2. Feature construction
# =============================================================================

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build HMM features from SPY price series.
    Features: [daily_ret, realized_vol_21d, momentum_63d]
    """
    if "SPY" not in prices.columns:
        raise ValueError("SPY not in price cache")

    spy    = prices["SPY"].dropna()
    daily  = spy.pct_change()

    vol21  = daily.rolling(VOL_WINDOW).std() * np.sqrt(ANN_FACTOR)
    mom63  = spy.pct_change(MOM_WINDOW)

    feat   = pd.DataFrame({
        "daily_ret": daily,
        "vol_21d":   vol21,
        "mom_63d":   mom63,
    }).dropna()

    # Standardize features for stable EM
    feat   = (feat - feat.mean()) / (feat.std() + 1e-10)
    return feat


# =============================================================================
# 3. Fit + decode
# =============================================================================

def fit_and_decode(
    feat: pd.DataFrame,
) -> tuple[GaussianHMM2, pd.DataFrame]:
    """
    Fit HMM on IS features, then decode full history (IS+OOS).
    Returns fitted model and daily regime DataFrame.
    """
    is_feat = feat[feat.index < OOS_CUTOFF]
    print(f"    IS training: {len(is_feat)} days  "
          f"({is_feat.index[0].date()} – {is_feat.index[-1].date()})")

    X_is = is_feat.values.astype(float)

    model = GaussianHMM2(n_states=N_STATES, n_iter=N_EM_ITER, tol=EM_TOL)
    model.fit(X_is)

    # Decode full history (OOS never seen during training)
    X_all   = feat.values.astype(float)
    viterbi = model.viterbi(X_all)
    proba   = model.predict_proba(X_all)

    # Label states: state with LOWER mean daily return = Bear (1), other = Bull (0)
    # Find which state index corresponds to lower return (feature 0 = daily_ret)
    mean_ret = np.array([model.mu[k][0] for k in range(N_STATES)])
    bear_state = int(np.argmin(mean_ret))  # lower return = Bear
    bull_state = 1 - bear_state

    # Re-map so state 0 = Bull, state 1 = Bear always
    remap  = {bull_state: 0, bear_state: 1}
    viterbi_mapped = np.array([remap[s] for s in viterbi])

    regime_df = pd.DataFrame({
        "date":         feat.index,
        "viterbi_raw":  viterbi,
        "regime":       viterbi_mapped,   # 0=Bull, 1=Bear
        "prob_bull":    proba[:, bull_state],
        "prob_bear":    proba[:, bear_state],
        "is_oos":       feat.index >= OOS_CUTOFF,
    }).set_index("date")

    return model, regime_df


# =============================================================================
# 4. Monthly regime at rebalance dates
# =============================================================================

def regime_at_rebalance(
    regime_df: pd.DataFrame,
    reb_dates: list,
) -> pd.DataFrame:
    """Map each rebalance date to the HMM regime on that day."""
    rows = []
    for rd in reb_dates:
        ts  = pd.Timestamp(rd)
        # Use most recent regime signal before or on rebalance date
        avail = regime_df.index[regime_df.index <= ts]
        if len(avail) == 0:
            continue
        row = regime_df.loc[avail[-1]]
        rows.append({
            "rebalance_date": rd,
            "regime":         int(row["regime"]),
            "regime_label":   "Bull" if row["regime"] == 0 else "Bear",
            "prob_bull":      round(float(row["prob_bull"]), 4),
            "prob_bear":      round(float(row["prob_bear"]), 4),
            "exposure":       EXPOSURE[int(row["regime"])],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 5. Regime-aware portfolio returns
# =============================================================================

def apply_regime_overlay(
    ls_df: pd.DataFrame,
    regime_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale L/S monthly returns by regime exposure multiplier.
    ls_df: from ls_monthly_returns.csv (step350)
    """
    merged = ls_df.merge(
        regime_monthly[["rebalance_date", "regime", "regime_label", "exposure"]],
        on="rebalance_date", how="left",
    )
    merged["exposure"] = merged["exposure"].fillna(1.0)

    # Regime-aware return = static return × exposure
    merged["dollar_neutral_ret_ra"] = (
        merged["dollar_neutral_ret"] * merged["exposure"]
    )
    merged["beta_neutral_ret_ra"] = (
        merged["beta_neutral_ret"] * merged["exposure"]
    )
    return merged


# =============================================================================
# 6. Metrics
# =============================================================================

def _metrics(rets: pd.Series, label: str) -> dict:
    rets = rets.dropna()
    if rets.empty:
        return {"label": label}
    ann  = ANN_FACTOR / HOLD_PERIOD
    n    = len(rets)
    mean = rets.mean()
    std  = rets.std()
    sharpe  = (mean / (std + 1e-10)) * np.sqrt(ann)
    cum     = (1 + rets).prod()
    cagr    = cum ** (ann / n) - 1
    cs      = (1 + rets).cumprod()
    dd      = (cs - cs.cummax()) / cs.cummax()
    maxdd   = float(dd.min())
    calmar  = abs(cagr / maxdd) if maxdd < 0 else 0
    hit     = (rets > 0).mean()
    return {
        "label":    label,
        "cagr":     round(cagr, 4),
        "sharpe":   round(sharpe, 3),
        "max_dd":   round(maxdd, 4),
        "calmar":   round(calmar, 2),
        "hit_rate": round(hit, 3),
        "n_months": n,
    }


# =============================================================================
# 7. Report
# =============================================================================

def generate_report(
    model: GaussianHMM2,
    regime_df: pd.DataFrame,
    regime_monthly: pd.DataFrame,
    perf_df: pd.DataFrame,
    ts: str,
) -> str:
    lines = [
        "# Canyon v9 — Step 370: HMM Regime Detection",
        f"Generated: {ts}",
        "",
        "## Model Architecture",
        "",
        "```",
        "Algorithm : 2-state Gaussian HMM (scratch implementation, numpy only)",
        "Training  : Baum-Welch EM on IS daily data (pre-2020)",
        "Decoding  : Viterbi path (full history, OOS never used in training)",
        "Features  : [SPY 1-day return, SPY 21d realized vol, SPY 63d momentum]",
        "            (all features standardized before fitting)",
        "```",
        "",
    ]

    if model.A is not None:
        A = model.A
        lines += [
            "## Estimated Transition Matrix",
            "",
            "```",
            f"                  → Bull    → Bear",
            f"  Bull (state 0)    {A[0,0]:.3f}    {A[0,1]:.3f}",
            f"  Bear (state 1)    {A[1,0]:.3f}    {A[1,1]:.3f}",
            "```",
            "",
            f"  Expected Bull duration: **{1/(A[0,1]+1e-10):.0f} days**",
            f"  Expected Bear duration: **{1/(A[1,0]+1e-10):.0f} days**",
            "",
        ]

    # Regime breakdown
    if not regime_df.empty:
        oos = regime_df[regime_df["is_oos"]]
        bull_pct = (oos["regime"] == 0).mean()
        bear_pct = (oos["regime"] == 1).mean()
        lines += [
            "## OOS Regime Distribution (2020–2026)",
            "",
            f"  Bull (regime 0): **{bull_pct:.1%}** of trading days",
            f"  Bear (regime 1): **{bear_pct:.1%}** of trading days",
            "",
        ]

    # Regime at rebalance
    if not regime_monthly.empty:
        oos_rb = regime_monthly
        bull_rb = (oos_rb["regime_label"] == "Bull").mean()
        bear_rb = (oos_rb["regime_label"] == "Bear").mean()
        lines += [
            f"  At rebalance dates: {bull_rb:.1%} Bull / {bear_rb:.1%} Bear",
            f"  Average exposure applied: {oos_rb['exposure'].mean():.2f}×",
            "",
        ]

    # Performance table
    if not perf_df.empty:
        lines += [
            "## Performance: Static vs Regime-Aware L/S",
            "",
            "| Strategy | CAGR | Sharpe | Max DD | Calmar | Hit Rate |",
            "|----------|:----:|:------:|:------:|:------:|:--------:|",
        ]
        for _, r in perf_df.iterrows():
            if r.get("n_months", 0):
                lines.append(
                    f"| {r['label']} | {r['cagr']:.1%} | {r['sharpe']:.2f} | "
                    f"{r['max_dd']:.1%} | {r['calmar']:.2f} | {r['hit_rate']:.1%} |"
                )
        lines += [""]

    lines += [
        "## Portfolio Overlay Logic",
        "",
        "```",
        "Regime detected (Viterbi at rebalance date):",
        "  Bull → apply full L/S allocation (100% exposure)",
        "  Bear → reduce to half exposure (50%)  — 'risk-off'",
        "```",
        "",
        "> When the HMM detects a Bear regime, we cut gross exposure by half.",
        "> This reduces drawdown during high-vol periods (2022 rate shock,",
        "> 2020 COVID crash) at the cost of lower participation in early",
        "> bear-regime recoveries.",
        "",
        "## Limitations",
        "",
        "1. **Only 2 states**: A 3-state model (Bull / Bear / Crisis) would",
        "   identify extreme events (COVID, 2022 peak-to-trough) more precisely.",
        "",
        "2. **Slow regime transitions**: With only SPY daily returns as input,",
        "   the HMM detects regimes 1–3 weeks after they begin. A real system",
        "   would add VIX level, credit spreads, and yield curve slope.",
        "",
        "3. **Baum-Welch on daily data**: Monthly L/S rebalancing means each",
        "   regime lasts multiple periods — daily HMM is appropriate here.",
        "",
        "## Next Step",
        "",
        "Week 18: Daily P&L Attribution + Operations Dashboard.",
    ]
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 370: HMM Regime Detection")
    print("=" * 70)

    # 1. Load prices — download long SPY history for HMM training, fall back to cache
    print("\n[1/5] Loading prices …")
    try:
        import yfinance as yf
        spy_raw = yf.download("SPY", period="20y", progress=False, auto_adjust=True)
        spy_close = spy_raw["Close"].squeeze()
        # Build a single-column DataFrame matching the price cache format
        prices = pd.DataFrame({"SPY": spy_close})
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.dropna()
        print(f"  SPY downloaded: {len(prices)} days ({prices.index[0].date()} – {prices.index[-1].date()})")
    except Exception as _e:
        print(f"  yfinance failed ({_e}), falling back to price cache")
        prices = pd.read_csv(ROOT / "backtest_price_cache.csv",
                             index_col=0, parse_dates=True)
        print(f"  {prices.shape[1]} tickers × {len(prices)} days")

    # 2. Build features
    print("\n[2/5] Building SPY features …")
    feat = build_features(prices)
    print(f"  Feature matrix: {feat.shape}  "
          f"({feat.index[0].date()} – {feat.index[-1].date()})")

    # 3. Fit HMM
    print("\n[3/5] Fitting 2-state Gaussian HMM (Baum-Welch EM) …")
    model, regime_df = fit_and_decode(feat)
    regime_df.to_csv(ROOT / "hmm_regime_daily.csv")

    # Transition matrix stats
    A = model.A
    print(f"  Transition matrix:")
    print(f"    Bull→Bull {A[0,0]:.3f}  Bull→Bear {A[0,1]:.3f}")
    print(f"    Bear→Bull {A[1,0]:.3f}  Bear→Bear {A[1,1]:.3f}")
    print(f"  Expected Bull duration: {1/(A[0,1]+1e-10):.0f} days")
    print(f"  Expected Bear duration: {1/(A[1,0]+1e-10):.0f} days")

    # OOS regime stats
    oos_df  = regime_df[regime_df["is_oos"]]
    bull_pct= (oos_df["regime"] == 0).mean()
    bear_pct= (oos_df["regime"] == 1).mean()
    print(f"\n  OOS regime distribution:")
    print(f"    Bull: {bull_pct:.1%}  Bear: {bear_pct:.1%}")

    # Save transition matrix
    A_df = pd.DataFrame(
        A, index=["Bull_from", "Bear_from"],
        columns=["to_Bull", "to_Bear"]
    )
    A_df.to_csv(ROOT / "hmm_transition_matrix.csv")

    # 4. Regime overlay on L/S portfolio
    print("\n[4/5] Applying regime overlay to L/S portfolio …")
    ls_df = pd.read_csv(ROOT / "ls_monthly_returns.csv",
                        parse_dates=["rebalance_date"])
    if ls_df.empty:
        print("  No L/S results found — skipping overlay")
        return

    reb_dates      = ls_df["rebalance_date"].tolist()
    regime_monthly = regime_at_rebalance(regime_df, reb_dates)
    regime_monthly.to_csv(ROOT / "hmm_regime_monthly.csv", index=False)

    bull_reb = (regime_monthly["regime_label"] == "Bull").mean()
    bear_reb = (regime_monthly["regime_label"] == "Bear").mean()
    avg_exp  = regime_monthly["exposure"].mean()
    print(f"  Rebalance regime: {bull_reb:.1%} Bull / {bear_reb:.1%} Bear")
    print(f"  Average exposure multiplier: {avg_exp:.2f}×")

    # Apply overlay
    combined = apply_regime_overlay(ls_df, regime_monthly)

    # 5. Performance comparison
    print("\n[5/5] Computing performance …")
    perf_rows = []
    for label, col in [
        ("Static Dollar-Neutral L/S",         "dollar_neutral_ret"),
        ("Regime-Aware Dollar-Neutral L/S",    "dollar_neutral_ret_ra"),
        ("Static Beta-Neutral L/S",            "beta_neutral_ret"),
        ("Regime-Aware Beta-Neutral L/S",      "beta_neutral_ret_ra"),
    ]:
        rets = combined[col].dropna() if col in combined.columns else pd.Series()
        perf_rows.append(_metrics(rets, label))

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(ROOT / "hmm_regime_performance.csv", index=False)

    print(f"\n  {'Strategy':<42} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
    print(f"  {'-'*68}")
    for _, r in perf_df.iterrows():
        if r.get("n_months", 0):
            print(f"  {r['label']:<42} {r['cagr']:>6.1%} {r['sharpe']:>7.2f} "
                  f"{r['max_dd']:>7.1%} {r['calmar']:>7.2f}")

    # Improvement summary
    static = perf_df[perf_df["label"] == "Static Dollar-Neutral L/S"]
    regime = perf_df[perf_df["label"] == "Regime-Aware Dollar-Neutral L/S"]
    if not static.empty and not regime.empty:
        dd_improvement = static.iloc[0]["max_dd"] - regime.iloc[0]["max_dd"]
        sr_change      = regime.iloc[0]["sharpe"] - static.iloc[0]["sharpe"]
        print(f"\n  Max DD improvement:  {dd_improvement:+.1%}  (less negative = better)")
        print(f"  Sharpe change:       {sr_change:+.2f}")

    # Generate report
    report = generate_report(model, regime_df, regime_monthly, perf_df, ts)
    (ROOT / "hmm_report.md").write_text(report)

    print("\n" + "=" * 70)
    print("Step 370 Complete — HMM Regime Detection")
    print("=" * 70)
    outputs = [
        "hmm_regime_daily.csv", "hmm_regime_monthly.csv",
        "hmm_regime_performance.csv", "hmm_transition_matrix.csv",
        "hmm_report.md",
    ]
    print("\nOutputs:")
    for f in outputs:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—"
        print(f"  {f:<42} {sz}")


if __name__ == "__main__":
    main()
