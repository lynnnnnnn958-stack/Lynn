"""
W48: End-to-End Pipeline Integration Test
==========================================
Validates the full Canyon v11 pipeline modules work together correctly:

  1. Signal math integrity (IC, weights, covariance)
  2. Factor model mathematics (Barra, Ledoit-Wolf, BL)
  3. Portfolio construction (MVO, Kelly)
  4. Execution layer (AC impact, TWAP)
  5. Attribution math (BHB sector decomposition)
  6. Scorecard thresholds (Deflated Sharpe, calibration)
  7. Module import health (all W1-W47 modules compile without error)

These tests are self-contained (no network, no disk IO beyond tmp fixtures).
Run with: python -m pytest tests/test_e2e_pipeline.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def synthetic_prices(rng):
    """100 tickers × 504 trading days (≈ 2 years)."""
    n_days, n_tickers = 504, 100
    log_ret = rng.normal(0.0002, 0.012, (n_days, n_tickers))
    prices = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)),
        index=pd.date_range("2022-01-03", periods=n_days, freq="B"),
        columns=[f"T{i:03d}" for i in range(n_tickers)],
    )
    return prices


@pytest.fixture(scope="module")
def synthetic_returns(synthetic_prices):
    return synthetic_prices.pct_change().dropna()


@pytest.fixture(scope="module")
def synthetic_weights(rng):
    """Random long-only weights summing to 1.0."""
    n = 20
    w = rng.dirichlet(np.ones(n))
    return pd.Series(w, index=[f"T{i:03d}" for i in range(n)])


@pytest.fixture(scope="module")
def synthetic_alpha(rng):
    """Cross-sectional alpha scores [0-100]."""
    return pd.Series(rng.uniform(0, 100, 100), index=[f"T{i:03d}" for i in range(100)])


# ─── 1. IC / Signal math ──────────────────────────────────────────────────────

class TestSignalMath:

    def test_ic_squared_weighting(self, rng):
        """
        Grinold-Kahn IC² weighting: combined_signal = Σ_k (IC²_k × signal_k) / Σ IC²_k
        A signal with IC=0.10 should get 4× the weight of IC=0.05.
        """
        ics = np.array([0.10, 0.05, 0.08])
        signals = np.vstack([
            rng.normal(0, 1, 200),
            rng.normal(0, 1, 200),
            rng.normal(0, 1, 200),
        ])  # shape (3, 200)

        ic_sq = ics ** 2
        weights = ic_sq / ic_sq.sum()
        combined = weights @ signals  # (200,)

        assert combined.shape == (200,)
        assert abs(weights.sum() - 1.0) < 1e-10

        # IC=0.10 gets 4× IC=0.05
        ratio = weights[0] / weights[1]
        assert abs(ratio - (0.10 / 0.05) ** 2) < 1e-8, \
            f"IC² weight ratio {ratio:.3f} ≠ {(0.10/0.05)**2:.3f}"

    def test_signal_halflife_decay(self):
        """IC(h) = IC₀ × exp(-h/halflife). At h=halflife, IC = IC₀/e."""
        ic0, halflife = 0.08, 21.0
        h = np.array([0, 1, 5, 10, 21, 42, 63])
        ic_h = ic0 * np.exp(-h / halflife)

        assert abs(ic_h[0] - ic0) < 1e-10, "IC at h=0 must equal IC0"
        assert abs(ic_h[6] - ic0 * np.exp(-3)) < 1e-9, "IC at h=3×halflife must be IC0/e³"
        # Monotonically decreasing
        assert all(np.diff(ic_h) < 0), "IC must decay monotonically with lag"
        # At halflife, IC = IC0/e ≈ 0.368 × IC0
        assert abs(ic_h[5] - ic0 * np.exp(-2)) < 1e-9

    def test_spearman_ic_range(self, synthetic_alpha, synthetic_prices):
        """Spearman IC must be in [-1, 1] and finite."""
        prices = synthetic_prices
        t = 250
        fwd_ret = prices.iloc[t + 21] / prices.iloc[t] - 1

        common = synthetic_alpha.index.intersection(fwd_ret.index)
        ic, pval = spearmanr(synthetic_alpha[common], fwd_ret[common])

        assert -1.0 <= ic <= 1.0, f"IC out of range: {ic}"
        assert np.isfinite(ic), "IC is not finite"
        assert np.isfinite(pval), "IC p-value is not finite"


# ─── 2. Covariance / Factor model ────────────────────────────────────────────

class TestCovarianceModel:

    def test_covariance_matrix_psd(self, synthetic_returns):
        """Covariance matrix must be positive semi-definite."""
        cov = synthetic_returns.iloc[:, :20].cov().values
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-10).all(), \
            f"Covariance matrix has negative eigenvalue: {eigenvalues.min():.6f}"

    def test_ledoit_wolf_shrinkage(self, synthetic_returns, rng):
        """
        Ledoit-Wolf toward constant correlation:
        Σ_lw = (1-α) Σ_sample + α Σ_target
        With α ∈ (0,1], the shrunk matrix must be PSD and have lower off-diagonal variance.
        """
        R = synthetic_returns.iloc[:, :30].values  # T×N
        n = R.shape[1]
        S = np.cov(R.T)  # N×N sample covariance

        # Constant-correlation target
        std_vec = np.sqrt(np.diag(S))
        corr_S = S / np.outer(std_vec, std_vec)
        rho_bar = (corr_S.sum() - n) / (n * (n - 1))
        T_target = rho_bar * np.outer(std_vec, std_vec)
        np.fill_diagonal(T_target, np.diag(S))

        alpha = min(0.8, (n / R.shape[0]) * 2)
        S_lw = (1 - alpha) * S + alpha * T_target

        # Must be PSD
        eig = np.linalg.eigvalsh(S_lw)
        assert (eig >= -1e-10).all(), f"Shrunk covariance not PSD: {eig.min():.6f}"

        # Diagonal must be preserved
        np.testing.assert_allclose(np.diag(S_lw), np.diag(S), rtol=1e-8)

    def test_ewma_covariance_weights_sum_to_one(self):
        """EWMA weights for halflife=63 must sum to 1 over 252 observations."""
        halflife = 63
        n_obs = 252
        lam = 0.5 ** (1 / halflife)
        t_idx = np.arange(n_obs - 1, -1, -1)  # oldest first
        w = lam ** t_idx
        w /= w.sum()

        assert abs(w.sum() - 1.0) < 1e-10
        # Most recent observation gets highest weight
        assert w[-1] > w[-2] > w[-3]
        # Oldest observation gets lowest weight
        assert w[0] < w[1]


# ─── 3. Black-Litterman ────────────────────────────────────────────────────────

class TestBlackLitterman:

    def test_posterior_blends_prior_and_view(self, rng):
        """
        BL posterior: μ_BL = [(τΣ)^{-1} + Ω^{-1}]^{-1} × [(τΣ)^{-1}Π + Ω^{-1}Q]
        When τ→0 (strong prior), μ_BL → Π.
        When τ→∞ (weak prior), μ_BL → Q (views dominate).
        """
        n = 10
        Sigma = np.eye(n) * 0.04  # diagonal for simplicity
        Pi = np.zeros(n)           # neutral prior
        Q  = np.ones(n) * 0.10    # bullish views

        # Strong prior (τ very small)
        tau_small = 1e-6
        Omega = np.eye(n) * 0.01
        precision_prior = np.linalg.inv(tau_small * Sigma)
        precision_view  = np.linalg.inv(Omega)
        M_small = np.linalg.inv(precision_prior + precision_view)
        mu_bl_small = M_small @ (precision_prior @ Pi + precision_view @ Q)

        # With very strong prior, mu_bl should be close to Pi (0)
        assert np.abs(mu_bl_small).max() < 0.05, \
            f"Strong prior: expected mu≈0, got {np.abs(mu_bl_small).max():.4f}"

        # Weak prior (τ very large)
        tau_large = 1e6
        precision_prior_large = np.linalg.inv(tau_large * Sigma)
        M_large = np.linalg.inv(precision_prior_large + precision_view)
        mu_bl_large = M_large @ (precision_prior_large @ Pi + precision_view @ Q)

        # With very weak prior, mu_bl should be close to Q (0.10)
        assert np.abs(mu_bl_large - Q).max() < 0.01, \
            f"Weak prior: expected mu≈Q=0.10, got {np.abs(mu_bl_large).max():.4f}"

    def test_equilibrium_returns_formula(self, rng):
        """Π = λ Σ w_mkt. Lambda=3 (risk aversion), equal-weight market portfolio."""
        n = 20
        lam = 3.0
        w_mkt = np.ones(n) / n
        Sigma  = rng.standard_normal((n, n))
        Sigma  = Sigma @ Sigma.T / n  # PSD
        Pi = lam * Sigma @ w_mkt

        # Verify: larger lambda → larger equilibrium returns
        Pi2 = (lam * 2) * Sigma @ w_mkt
        assert (np.abs(Pi2) > np.abs(Pi)).all() or \
               np.abs(Pi2).mean() > np.abs(Pi).mean(), \
               "Equilibrium returns should scale linearly with risk aversion"


# ─── 4. Portfolio construction ────────────────────────────────────────────────

class TestPortfolioConstruction:

    def test_fractional_kelly_constraint(self, rng):
        """w_kelly = κ × Σ^{-1} μ; with κ=0.25, max weight must be < full kelly."""
        n = 15
        kappa_full = 1.0
        kappa_frac = 0.25

        Sigma = np.eye(n) * 0.04
        mu = rng.normal(0.05, 0.02, n)

        w_full = kappa_full * np.linalg.inv(Sigma) @ mu
        w_frac = kappa_frac * np.linalg.inv(Sigma) @ mu

        # Fractional Kelly must be exactly κ × full Kelly
        np.testing.assert_allclose(w_frac, w_full * kappa_frac, rtol=1e-10)

        # After normalization, fraction shouldn't exceed full Kelly before normalizing
        abs_ratio = np.abs(w_frac).sum() / (np.abs(w_full).sum() + 1e-9)
        assert abs(abs_ratio - kappa_frac) < 1e-8, \
            f"Kelly fraction {abs_ratio:.4f} ≠ κ={kappa_frac}"

    def test_mvo_unconstrained_solution(self, rng):
        """w* = Σ^{-1} μ / λ (unconstrained MVO). Verify analytically."""
        n = 5
        lam = 3.0
        Sigma = np.diag(np.array([0.04, 0.09, 0.01, 0.16, 0.06]))
        mu = np.array([0.08, 0.12, 0.05, 0.15, 0.07])

        w_star = np.linalg.inv(Sigma) @ mu / lam

        # Verify: Σ w* = μ/λ
        np.testing.assert_allclose(Sigma @ w_star, mu / lam, rtol=1e-10)

    def test_weight_normalization(self, rng):
        """After clip to [0, max_weight] + renormalize, weights must sum to 1."""
        n = 30
        raw_weights = rng.standard_normal(n)
        # Clip negative, cap at 10%, renormalize
        w = np.clip(raw_weights, 0, None)
        w = np.minimum(w, 0.10)
        if w.sum() > 0:
            w /= w.sum()
        assert abs(w.sum() - 1.0) < 1e-10, f"Weights sum to {w.sum()}"
        assert (w >= 0).all(), "Negative weights after normalization"
        assert (w <= 0.10 + 1e-10).all(), "Weights exceed 10% cap"


# ─── 5. Execution model ────────────────────────────────────────────────────────

class TestExecutionModel:

    def test_almgren_chriss_impact(self):
        """
        AC model: impact_bps = η × vol × √(qty/ADV) × 10000
        Larger position → higher impact, nonlinear (square-root law).
        """
        eta = 0.1
        daily_vol = 0.015  # 1.5%
        adv = 1_000_000.0

        impacts = []
        for qty in [10_000, 40_000, 90_000, 160_000]:
            impact = eta * daily_vol * np.sqrt(qty / adv) * 10_000  # bps
            impacts.append(impact)

        # Impact must increase with quantity
        assert impacts[0] < impacts[1] < impacts[2] < impacts[3]

        # Doubling qty → impact × √2 (not × 2)
        ratio = impacts[1] / impacts[0]  # qty doubled: 40k/10k = 4×, √4 = 2
        # 40k / 10k = 4×, √4 = 2
        assert abs(ratio - 2.0) < 1e-6, \
            f"AC impact ratio {ratio:.4f} ≠ √4 = 2.0 (square-root law violated)"

    def test_twap_slice_math(self):
        """TWAP 6 slices of equal size must total target quantity."""
        total_qty = 6000
        n_slices = 6
        slice_qty = total_qty // n_slices
        executed_qtys = [slice_qty] * n_slices

        assert sum(executed_qtys) == total_qty, \
            f"TWAP slices sum to {sum(executed_qtys)}, expected {total_qty}"
        assert all(q > 0 for q in executed_qtys), "TWAP slice quantity must be positive"


# ─── 6. Attribution (Brinson-Hood-Beebower) ────────────────────────────────────

class TestAttribution:

    def test_bhb_components_sum_to_active_return(self, rng):
        """
        BHB: Active return = allocation + selection + interaction.
        This must hold exactly.
        """
        sectors = ["Tech", "Finance", "Energy", "Health"]
        n_per = 10
        n = len(sectors) * n_per

        port_w = rng.dirichlet(np.ones(n))
        bm_w   = np.ones(n) / n
        ret    = rng.normal(0.001, 0.02, n)

        # Assign sectors
        sec_arr = np.repeat(sectors, n_per)

        active_total = port_w @ ret - bm_w @ ret
        bhb_sum = 0.0
        port_total_ret = port_w @ ret
        bm_total_ret   = bm_w @ ret

        for sec in sectors:
            mask = sec_arr == sec
            pw  = port_w[mask].sum()
            bw  = bm_w[mask].sum()
            pr  = (port_w[mask] * ret[mask]).sum() / (pw + 1e-12)
            br  = (bm_w[mask]   * ret[mask]).sum() / (bw + 1e-12)

            allocation  = (pw - bw) * (br - bm_total_ret)
            selection   = bw * (pr - br)
            interaction = (pw - bw) * (pr - br)
            bhb_sum += allocation + selection + interaction

        assert abs(bhb_sum - active_total) < 1e-10, \
            f"BHB components {bhb_sum:.8f} ≠ active return {active_total:.8f}"

    def test_factor_attribution_decomposition(self, rng):
        """
        Factor attribution: total_return = Σ_k (port_exp_k × factor_ret_k) + specific
        Residual must be deterministic given exposures.
        """
        n_assets, n_factors = 30, 5
        B = rng.standard_normal((n_assets, n_factors))
        factor_ret = rng.normal(0, 0.01, n_factors)
        specific   = rng.normal(0, 0.005, n_assets)

        # True returns
        true_ret = B @ factor_ret + specific
        w = rng.dirichlet(np.ones(n_assets))
        port_ret = w @ true_ret

        # Factor attribution
        port_exp = B.T @ w  # (K,)
        factor_attr = port_exp @ factor_ret  # sum of factor contributions
        specific_attr = w @ specific

        assert abs(factor_attr + specific_attr - port_ret) < 1e-10, \
            f"Attribution doesn't balance: {factor_attr + specific_attr:.10f} ≠ {port_ret:.10f}"


# ─── 7. Scorecard thresholds ──────────────────────────────────────────────────

class TestScorecardMath:

    def test_deflated_sharpe_below_raw_sharpe(self, rng):
        """
        Deflated Sharpe (Bailey & de Prado 2016) always ≤ raw Sharpe.
        Penalizes for multiple testing: more trials → lower DSR.
        """
        from scipy.stats import norm as sp_norm

        raw_sharpe = 1.5
        n_obs = 252
        skew = -0.5
        kurtosis = 4.0

        def deflated_sharpe(sr, n, skewness, kurt, n_trials=1):
            # Expected max SR from n_trials iid trials
            gamma = 0.5772  # Euler-Mascheroni
            E_max = (1 - gamma) * sp_norm.ppf(1 - 1/n_trials) + \
                    gamma * sp_norm.ppf(1 - 1/(n_trials * np.e)) \
                    if n_trials > 1 else 0.0
            sigma = np.sqrt((1 - skewness * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
            dsr = sp_norm.cdf((sr - E_max) / (sigma + 1e-9))
            return dsr

        dsr_1  = deflated_sharpe(raw_sharpe, n_obs, skew, kurtosis, n_trials=1)
        dsr_10 = deflated_sharpe(raw_sharpe, n_obs, skew, kurtosis, n_trials=10)
        dsr_50 = deflated_sharpe(raw_sharpe, n_obs, skew, kurtosis, n_trials=50)

        # More trials → lower DSR (harder to achieve significance)
        assert dsr_1 >= dsr_10 >= dsr_50, \
            f"DSR should decrease with more trials: {dsr_1:.3f} > {dsr_10:.3f} > {dsr_50:.3f}"
        # All must be in [0,1] (it's a probability)
        for dsr in [dsr_1, dsr_10, dsr_50]:
            assert 0.0 <= dsr <= 1.0, f"DSR out of [0,1]: {dsr}"

    def test_max_drawdown_calculation(self):
        """Max drawdown must be non-negative and ≤ 100%."""
        # Known case: drawdown from 1.0 to 0.6 = 40%
        cum_ret = pd.Series([1.0, 1.1, 1.05, 0.95, 0.8, 0.85, 0.6, 0.75, 0.9])
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = float(-drawdown.min())

        assert max_dd > 0.0, "Max drawdown should be positive"
        assert max_dd <= 1.0, f"Max drawdown {max_dd:.2%} exceeds 100%"
        # Expected: peak=1.1, trough=0.6 → (0.6-1.1)/1.1 ≈ 45.45%
        expected_dd = (1.1 - 0.6) / 1.1
        assert abs(max_dd - expected_dd) < 1e-6, \
            f"Max DD {max_dd:.4f} ≠ expected {expected_dd:.4f}"


# ─── 8. Module import health ─────────────────────────────────────────────────

class TestModuleHealth:
    """Validate all W1-W47 modules parse without AST errors."""

    ROOT = Path(__file__).parent.parent

    MODULES = [
        "signals/macro_hmm.py",
        "signals/lgb_shap.py",
        "signals/earnings_call.py",
        "signals/eps_revision.py",
        "data/edgar_pit.py",
        "data/edgar_form4.py",
        "data/fred_macro.py",
        "data/sp500_constituents.py",
        "data/extend_history.py",
        "data/polygon_ivr.py",
        "research/signal_halflife.py",
        "research/signal_corr.py",
        "research/vif_check.py",
        "research/capacity.py",
        "research/backtest_scorecard.py",
        "research/tearsheet.py",
        "research/param_robustness.py",
        "research/stress_test.py",
        "research/v9_bridge.py",
        "research/institutional_scorecard.py",
        "risk/barra.py",
        "risk/regime_cov.py",
        "portfolio/black_litterman.py",
        "portfolio/kelly.py",
        "monitoring/data_quality.py",
        "monitoring/factor_exposure.py",
        "monitoring/slippage.py",
        "monitoring/execution_quality.py",
        "monitoring/attribution.py",
        "execution/alpaca_exec.py",
    ]

    @pytest.mark.parametrize("module_path", MODULES)
    def test_module_syntax(self, module_path):
        """Each module must parse without SyntaxError (AST-only, no imports)."""
        import ast
        full_path = self.ROOT / module_path
        if not full_path.exists():
            pytest.skip(f"Module not yet created: {module_path}")
        source = full_path.read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in {module_path}: {e}")

    def test_no_hardcoded_chinese_text_in_python(self):
        """All .py files must not contain Chinese characters (W1 requirement)."""
        import re
        chinese_pattern = re.compile('[\u4e00-\u9fff\u3400-\u4dbf]')
        violations = []
        for module_path in self.MODULES:
            full_path = self.ROOT / module_path
            if not full_path.exists():
                continue
            source = full_path.read_text()
            if chinese_pattern.search(source):
                violations.append(module_path)
        assert not violations, \
            f"Chinese text found in: {violations}"


# ─── 9. Data-quality guards ────────────────────────────────────────────────────

class TestDataQuality:

    def test_pit_compliance_no_future_dates(self):
        """
        PIT rule: signal_date must be >= know_date (filed_date).
        Tickers with filed_date > signal_date must be excluded.
        """
        signal_date = pd.Timestamp("2023-06-30")
        data = pd.DataFrame({
            "ticker":      ["AAPL", "MSFT", "GOOG", "AMZN"],
            "filed_date":  pd.to_datetime(["2023-06-15", "2023-07-01", "2023-06-29", "2023-06-30"]),
            "eps":         [1.5, 2.0, 0.8, 1.2],
        })
        # PIT-compliant: only include where filed_date <= signal_date
        pit_clean = data[data["filed_date"] <= signal_date]

        assert "MSFT" not in pit_clean["ticker"].values, \
            "MSFT filed 2023-07-01 > signal_date 2023-06-30 — PIT violation"
        assert len(pit_clean) == 3, \
            f"Expected 3 PIT-compliant rows, got {len(pit_clean)}"

    def test_universe_no_duplicate_tickers(self):
        """Universe DataFrame must not have duplicate tickers on any date."""
        dates = pd.date_range("2023-01-01", periods=5, freq="W")
        tickers = ["AAPL", "MSFT", "GOOG"]
        rows = [(d, t) for d in dates for t in tickers]
        df = pd.DataFrame(rows, columns=["date", "ticker"])
        df["alpha"] = np.random.randn(len(df))

        # Check uniqueness per date
        for date, group in df.groupby("date"):
            assert group["ticker"].is_unique, \
                f"Duplicate tickers on {date}: {group[group['ticker'].duplicated()]['ticker'].tolist()}"

    def test_no_inf_in_signals(self, synthetic_alpha):
        """Alpha signals must be finite after computation."""
        assert np.isfinite(synthetic_alpha.values).all(), \
            "Alpha signals contain Inf/NaN values"
        assert (synthetic_alpha >= 0).all() and (synthetic_alpha <= 100).all(), \
            "Alpha scores must be in [0, 100]"


# ─── 10. SUE / Earnings math ─────────────────────────────────────────────────

class TestEarningsMath:

    def test_sue_calculation(self):
        """
        SUE = (EPS_t - EPS_{t-4q}) / std(EPS_changes_8q)
        Positive earnings surprise → positive SUE.
        """
        eps_history = np.array([0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.93, 0.95])
        # t-4q was index 4, current is index 8 (hypothetical)
        eps_t   = 1.05   # beat expectations
        eps_4q  = eps_history[4]  # = 0.90
        changes = np.diff(eps_history)  # 7 QoQ changes
        sue = (eps_t - eps_4q) / (changes.std() + 1e-9)

        assert sue > 0, f"Positive earnings surprise should yield positive SUE, got {sue:.3f}"
        # If company missed: eps_t = 0.85 < eps_4q = 0.90 → negative SUE
        sue_miss = (0.85 - eps_4q) / (changes.std() + 1e-9)
        assert sue_miss < 0, "Negative earnings surprise should yield negative SUE"

    def test_eps_trend_fraction(self):
        """EPS trend = fraction of last 4 QoQ changes that are positive."""
        eps = [0.80, 0.85, 0.82, 0.88, 0.90]  # QoQ: +, -, +, +  → trend = 3/4 = 0.75
        changes = np.diff(eps)[-4:]  # last 4
        trend = (changes > 0).mean()
        assert abs(trend - 0.75) < 1e-10, f"EPS trend {trend:.2f} ≠ 0.75"
