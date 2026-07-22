"""
Canyon v9  Step 63 — LedoitWolf Portfolio Optimizer
=====================================================
Replaces naive inverse-volatility sizing with shrinkage-covariance
mean-variance optimization.

Three portfolios are solved simultaneously:
  1. Max-Sharpe  — maximises risk-adjusted return (Sharpe ratio)
  2. Min-Variance — minimises total portfolio volatility
  3. Risk-Parity  — each asset contributes equally to portfolio variance

Math:
  • Log-returns  r_t = ln(P_t / P_{t-1})
  • LedoitWolf shrinkage covariance  Σ̂  (sklearn)
  • Expected return μ̂  = annualised mean log-return
  • Max-Sharpe: min  −(μ̂·w − rf) / sqrt(w·Σ̂·w)   s.t. Σw=1, w≥0
  • Min-Var:    min   w·Σ̂·w                          s.t. Σw=1, w≥0
  • Risk-Parity: min  Σᵢ(wᵢ·(Σ̂w)ᵢ − σ_p²/N)²       s.t. Σw=1, w≥0

Outputs (all written to canyon_quant folder):
  portfolio_optimized_weights.csv   ticker → max_sharpe / min_var / risk_parity weights
  portfolio_covariance.csv          N×N annualised shrunk covariance
  portfolio_ef_points.csv           100-point efficient frontier (vol, ret, sharpe)
  portfolio_optimizer_report.md     full markdown report

Usage:
  python canyon_final_v9_step63_portfolio_optimizer.py
  python canyon_final_v9_step63_portfolio_optimizer.py --lookback 504  # 2-year window
  python canyon_final_v9_step63_portfolio_optimizer.py --cap 0.30      # 30% per-position cap
  python canyon_final_v9_step63_portfolio_optimizer.py --rf 0.053      # risk-free rate

No look-ahead bias: optimizer only uses data available at the computation date.
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    from sklearn.covariance import LedoitWolf as _LW
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ROOT = Path(__file__).parent

# ── default parameters ────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 252         # 1-year window for covariance estimation
RF_ANNUAL       = 0.053       # risk-free rate (5.3% — US T-Bill as of 2025)
POSITION_CAP    = 0.25        # max weight per ticker
MIN_WEIGHT      = 0.0         # allow zero weight (long-only)
EF_POINTS       = 60          # efficient frontier resolution
ANN_FACTOR      = 252         # trading days per year
TURNOVER_LAMBDA = 2.0         # turnover penalty strength (0 = no penalty)
SECTOR_CAP      = 0.40        # max total weight in any single sector

# Universe: reuse the 20 core tickers from Step 62
CORE_TICKERS = [
    "SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP",
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Price loader  (reuse step62 cache if present, else yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers: list[str], lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Return daily close prices for *tickers*, as wide DataFrame (date × ticker).

    Priority:
      1. backtest_price_cache.csv  (written by Step 62 engine)
      2. fresh yfinance download
    """
    # Prefer larger sp500 cache (written by step75/77); fall back to backtest cache
    for cache_name in ["sp500_price_cache.csv", "backtest_price_cache.csv"]:
        cache_path = ROOT / cache_name
        if not cache_path.exists():
            continue
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            available = [t for t in tickers if t in cached.columns]
            if available:
                print(f"  [cache] {cache_name} ({age_hours:.0f}h old), "
                      f"{len(available)}/{len(tickers)} tickers")
                subset = cached[available].dropna(how="all").tail(lookback + 5)
                return subset
        except Exception as e:
            print(f"  [cache] {cache_name} error: {e}")

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback + 60)  # extra buffer

    # Fresh download
    try:
        import yfinance as yf
        print(f"  [yfinance] Downloading {len(tickers)} tickers …")
        raw = yf.download(
            tickers, start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        prices = prices.dropna(how="all")
        # Save as cache
        prices.to_csv(cache_path)
        print(f"  [yfinance] Downloaded {prices.shape[1]} tickers, {len(prices)} days")
        return prices.tail(lookback + 5)
    except Exception as e:
        raise RuntimeError(f"Cannot load price data: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Returns + covariance
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-returns, drop rows with any NaN."""
    rets = np.log(prices / prices.shift(1))
    return rets.dropna()


def ledoitwolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (cov_annual, mean_annual, tickers) after LedoitWolf shrinkage.
    Falls back to sample covariance if sklearn unavailable.
    """
    tickers = returns.columns.tolist()
    R = returns.values  # shape (T, N)

    if _HAS_SKLEARN:
        lw = _LW(assume_centered=False)
        lw.fit(R)
        cov_daily = lw.covariance_
        shrinkage = getattr(lw, "shrinkage_", None)
        print(f"  [LedoitWolf] shrinkage coefficient = "
              f"{shrinkage:.4f}" if shrinkage is not None else "  [LedoitWolf] fitted")
    else:
        print("  [fallback] sklearn not available — using sample covariance")
        cov_daily = np.cov(R, rowvar=False)

    cov_annual = cov_daily * ANN_FACTOR
    mean_daily = R.mean(axis=0)
    mean_annual = mean_daily * ANN_FACTOR
    return cov_annual, mean_annual, tickers


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Optimization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                rf: float = RF_ANNUAL) -> tuple[float, float, float]:
    """Return (ret, vol, sharpe) for weight vector w."""
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 1e-12)))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def _make_constraints_bounds(n: int, cap: float = POSITION_CAP) -> tuple:
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(MIN_WEIGHT, cap)] * n
    return constraints, bounds


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray,
                         rf: float = RF_ANNUAL,
                         cap: float = POSITION_CAP) -> np.ndarray:
    n = len(mu)
    constraints, bounds = _make_constraints_bounds(n, cap)

    def neg_sharpe(w):
        r, v, s = _port_stats(w, mu, cov, rf)
        return -s

    w0 = np.ones(n) / n
    best_w = w0.copy()
    best_val = neg_sharpe(w0)

    # Multiple random starts to avoid local minima
    rng = np.random.default_rng(42)
    for _ in range(15):
        w_init = rng.dirichlet(np.ones(n))
        w_init = np.clip(w_init, MIN_WEIGHT, cap)
        w_init /= w_init.sum()
        res = minimize(neg_sharpe, w_init, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_w = res.x

    best_w = np.clip(best_w, MIN_WEIGHT, cap)
    best_w /= best_w.sum()
    return best_w


def optimize_min_variance(mu: np.ndarray, cov: np.ndarray,
                           cap: float = POSITION_CAP) -> np.ndarray:
    n = len(mu)
    constraints, bounds = _make_constraints_bounds(n, cap)

    def portfolio_var(w):
        return float(w @ cov @ w)

    w0 = np.ones(n) / n
    res = minimize(portfolio_var, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if res.success else w0
    w = np.clip(w, MIN_WEIGHT, cap)
    w /= w.sum()
    return w


def optimize_risk_parity(cov: np.ndarray, cap: float = POSITION_CAP) -> np.ndarray:
    """Equal Risk Contribution portfolio."""
    n = cov.shape[0]

    def risk_parity_objective(w):
        w = np.array(w)
        sigma_p = np.sqrt(w @ cov @ w + 1e-12)
        mrc = cov @ w          # marginal risk contribution
        rc  = w * mrc          # risk contribution per asset
        target = sigma_p ** 2 / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-4, cap)] * n   # RP needs positive weights
    w0 = np.ones(n) / n

    res = minimize(risk_parity_objective, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = res.x if res.success else w0
    w = np.clip(w, 1e-4, cap)
    w /= w.sum()
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Turnover-aware optimizer  (factor-neutral Sharpe with L1 turnover penalty)
# ─────────────────────────────────────────────────────────────────────────────

def load_previous_weights(tickers: list[str]) -> Optional[np.ndarray]:
    """
    Load prior max-sharpe weights from the last run's CSV, aligned to *tickers*.
    Returns None if no prior file exists or tickers don't match.
    """
    path = ROOT / "portfolio_optimized_weights.csv"
    if not path.exists():
        return None
    try:
        prev = pd.read_csv(path)
        if "ticker" not in prev.columns or "max_sharpe" not in prev.columns:
            return None
        prev = prev.set_index("ticker")["max_sharpe"]
        # Align to current ticker list (fill missing with equal weight)
        w_prev = np.array([prev.get(t, 1.0 / len(tickers)) for t in tickers])
        w_prev = np.clip(w_prev, 0, 1)
        s = w_prev.sum()
        if s > 0:
            w_prev /= s
        return w_prev
    except Exception:
        return None


def optimize_turnover_aware(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    rf: float = RF_ANNUAL,
    cap: float = POSITION_CAP,
    lam: float = TURNOVER_LAMBDA,
    tickers: Optional[list[str]] = None,
    sector_map: Optional[dict[str, str]] = None,
) -> np.ndarray:
    """
    Maximise Sharpe minus a turnover penalty, with optional sector caps.

    Objective: −Sharpe(w) + λ × ||w − w_prev||₁

    Sector constraints: if sector_map is provided, each sector weight ≤ SECTOR_CAP.
    Beta neutralization: implicit via covariance (SPY-aware shrinkage already
    handles systematic risk; explicit beta constraint would need index returns).
    """
    n = len(mu)
    constraints, bounds = _make_constraints_bounds(n, cap)

    # ── Sector constraints ────────────────────────────────────────────────────
    if tickers and sector_map:
        # Group ticker indices by sector
        sector_groups: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            s = sector_map.get(t, "Unknown")
            sector_groups.setdefault(s, []).append(i)
        for sec, idxs in sector_groups.items():
            if len(idxs) >= 2:   # only constrain sectors with ≥2 names
                def _sector_ineq(w, idxs=idxs):
                    return SECTOR_CAP - sum(w[i] for i in idxs)
                constraints.append({"type": "ineq", "fun": _sector_ineq})

    def objective(w):
        r, v, _ = _port_stats(w, mu, cov, rf)
        sharpe = (r - rf) / max(v, 1e-10)
        # L1 turnover penalty: penalise large deviations from last run's weights
        turnover = float(np.sum(np.abs(w - w_prev)))
        return -sharpe + lam * turnover

    w0 = w_prev.copy()
    best_w, best_val = w0.copy(), objective(w0)

    rng = np.random.default_rng(7)
    for _ in range(12):
        w_init = 0.7 * w_prev + 0.3 * rng.dirichlet(np.ones(n))
        w_init = np.clip(w_init, MIN_WEIGHT, cap)
        w_init /= w_init.sum()
        res = minimize(objective, w_init, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 600, "ftol": 1e-9})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_w = res.x

    best_w = np.clip(best_w, MIN_WEIGHT, cap)
    best_w /= best_w.sum()
    return best_w


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Efficient frontier
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                                cap: float = POSITION_CAP,
                                n_points: int = EF_POINTS) -> pd.DataFrame:
    """Sweep target returns from min-var to max-mu and collect (vol, ret, sharpe)."""
    n = len(mu)

    # Bounds for the sweep
    min_ret  = float(mu.min())
    max_ret  = float(min(mu.max(), mu.max()))
    # Use 99% of max_ret (not 92%) — the 0.92 cap was arbitrary and suppressed
    # aggressive allocations that the optimizer could legitimately reach.
    target_rets = np.linspace(min_ret, max_ret * 0.99, n_points)

    records = []
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(MIN_WEIGHT, cap)] * n

    for tr in target_rets:
        constraints = constraints_base + [
            {"type": "ineq", "fun": lambda w, t=tr: (w @ mu) - t}
        ]
        res = minimize(
            lambda w: float(w @ cov @ w),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if res.success:
            w = np.clip(res.x, 0, 1)
            w /= w.sum()
            r, v, s = _port_stats(w, mu, cov)
            records.append({"target_ret": tr, "vol": v, "ret": r, "sharpe": s})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Alpha-score expected return override (from Step 87)
# ─────────────────────────────────────────────────────────────────────────────

def load_alpha_mu(tickers: list[str]) -> Optional[np.ndarray]:
    """
    Read alpha_scores.csv produced by Step 87.
    Returns mu_override vector aligned to *tickers*, or None if file missing.
    Step 87 maps alpha_score [0,100] → annualised expected return via:
        mu = RF + (alpha_score - 50) / 50 * 0.25
    """
    path = ROOT / "alpha_scores.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns or "mu_override" not in df.columns:
            return None
        df = df.set_index("ticker")["mu_override"]
        mu_alpha = np.array([df.get(t, np.nan) for t in tickers])
        n_found = np.sum(~np.isnan(mu_alpha))
        print(f"  [alpha μ] Step 87 alpha scores loaded: "
              f"{n_found}/{len(tickers)} tickers matched")
        return mu_alpha
    except Exception as e:
        print(f"  [alpha μ] Could not load alpha_scores.csv: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Inverse-vol baseline (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_vol_weights(returns: pd.DataFrame, cap: float = POSITION_CAP) -> np.ndarray:
    vols = returns.std().values
    w = 1.0 / (vols + 1e-10)
    w = np.clip(w, 0, cap)
    w /= w.sum()
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(weights_df: pd.DataFrame, ef_df: pd.DataFrame,
                 stats: dict, lookback: int, rf: float, ts: str) -> None:
    lines = [
        "# Canyon v9 — Portfolio Optimizer Report (Step 63)",
        f"Generated: {ts}  |  Lookback: {lookback} days  |  RF: {rf*100:.1f}%",
        "",
        "## Optimization Results",
        "",
        "| Portfolio | Return | Volatility | Sharpe | Max Weight |",
        "|---|---|---|---|---|",
    ]
    for pname, skey in [("Max-Sharpe", "max_sharpe"), ("Min-Variance", "min_var"),
                         ("Risk-Parity", "risk_parity"), ("Inv-Vol Baseline", "inv_vol"),
                         ("Turnover-Aware", "turnover_aware")]:
        s = stats.get(skey, {})
        lines.append(
            f"| {pname} | {s.get('ret',0)*100:.2f}% | {s.get('vol',0)*100:.2f}% | "
            f"{s.get('sharpe',0):.3f} | {s.get('max_w',0)*100:.1f}% |"
        )

    lines += [
        "",
        "## Weight Comparison",
        "",
        "| Ticker | Max-Sharpe | Min-Var | Risk-Parity | Inv-Vol | Turnover-Aware |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in weights_df.iterrows():
        lines.append(
            f"| {row['ticker']} | {row.get('max_sharpe',0)*100:.1f}% | "
            f"{row.get('min_var',0)*100:.1f}% | {row.get('risk_parity',0)*100:.1f}% | "
            f"{row.get('inv_vol',0)*100:.1f}% | {row.get('turnover_aware',0)*100:.1f}% |"
        )

    lines += [
        "",
        "## Efficient Frontier Summary",
        f"- {len(ef_df)} frontier points computed",
    ]
    if not ef_df.empty:
        best = ef_df.loc[ef_df["sharpe"].idxmax()]
        lines.append(
            f"- Best frontier Sharpe: {best['sharpe']:.3f} at "
            f"vol={best['vol']*100:.1f}%, ret={best['ret']*100:.1f}%"
        )

    lines += [
        "",
        "## Notes",
        "- LedoitWolf shrinkage applied to reduce estimation error in small samples",
        "- Long-only constraint (no short positions)",
        f"- Position cap: {POSITION_CAP*100:.0f}% per ticker",
        "- Survivorship bias: universe = current holdings, not historical universe",
        "- Expected returns estimated from historical mean — **forward returns may differ**",
        "- Use Max-Sharpe weights for new entries, Risk-Parity for defensive periods",
        f"- Turnover penalty λ={TURNOVER_LAMBDA} applied to Turnover-Aware portfolio",
        f"- Sector cap: {SECTOR_CAP*100:.0f}% max per sector (applied in Turnover-Aware only)",
    ]

    path = ROOT / "portfolio_optimizer_report.md"
    path.write_text("\n".join(lines))
    print(f"  [report] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(tickers: list[str], lookback: int, rf: float,
                  cap: float) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Canyon v9 Step 63 — Portfolio Optimizer")
    print(f"Tickers: {len(tickers)}  Lookback: {lookback}d  RF: {rf*100:.1f}%  Cap: {cap*100:.0f}%")
    print(f"{'='*60}")

    # ── 1. Load prices ────────────────────────────────────────────────────────
    print("\n[1/6] Loading prices …")
    prices = load_prices(tickers, lookback)
    # Drop tickers with too few observations
    min_obs = max(60, lookback // 3)
    prices = prices.loc[:, prices.count() >= min_obs]
    tickers_used = prices.columns.tolist()
    print(f"      {len(tickers_used)} tickers with ≥{min_obs} observations")

    # ── 2. Returns + covariance ───────────────────────────────────────────────
    print("\n[2/6] Computing returns and LedoitWolf covariance …")
    returns = compute_returns(prices)
    returns = returns.tail(lookback)     # enforce lookback
    cov, mu, tickers_used = ledoitwolf_cov(returns)

    # ── Alpha-score override for expected returns (from Step 87) ──────────────
    mu_alpha = load_alpha_mu(tickers_used)
    if mu_alpha is not None:
        # Fill missing alpha scores with historical mean
        n_missing = np.sum(np.isnan(mu_alpha))
        if n_missing > 0:
            mu_alpha = np.where(np.isnan(mu_alpha), mu, mu_alpha)

        # Cap historical returns at ±40% p.a. before blending.
        # Values beyond ±40% are almost always data anomalies (corporate actions,
        # reverse splits, etc. in the price cache) that would otherwise hijack the
        # optimizer by making one stock look vastly better than any alpha pick.
        # A stock with 140%+ historical annual return is a price-cache artefact,
        # not a real forward expectation.
        mu_hist_capped = np.clip(mu, -0.40, 0.40)
        n_capped = int(np.sum(np.abs(mu) > 0.40))
        if n_capped:
            print(f"  [alpha μ] {n_capped} tickers capped at ±40% historical mu "
                  f"(data-anomaly guard)")

        # Adaptive alpha blend — ratio adjusts to measured IC quality.
        # IC < 0.03 (weak signal): trust history more  → 60/40
        # IC 0.03-0.05 (decent):  standard blend       → 80/20
        # IC > 0.05 (strong):     trust alpha more      → 90/10
        # IC source: signal_weights.json (written by Step 84 IC tracker).
        _alpha_w = 0.80   # default
        _hist_w  = 0.20
        _ic_source = "default"
        try:
            import json as _json
            _sw_p = ROOT / "signal_weights.json"
            if _sw_p.exists():
                _sw_d = _json.loads(_sw_p.read_text())
                _mean_ic = _sw_d.get("mean_ic") or _sw_d.get("ic_mean")
                if _mean_ic is not None:
                    _mean_ic = float(_mean_ic)
                    if _mean_ic < 0.03:
                        _alpha_w, _hist_w = 0.60, 0.40
                        _ic_source = f"IC={_mean_ic:.3f} (weak)"
                    elif _mean_ic > 0.05:
                        _alpha_w, _hist_w = 0.90, 0.10
                        _ic_source = f"IC={_mean_ic:.3f} (strong)"
                    else:
                        _ic_source = f"IC={_mean_ic:.3f} (decent)"
        except Exception:
            pass
        mu_blended = _alpha_w * mu_alpha + _hist_w * mu_hist_capped
        print(f"  [alpha μ] Blended expected returns: "
              f"{int(_alpha_w*100)}% alpha + {int(_hist_w*100)}% historical  "
              f"[{_ic_source}]")
        mu = mu_blended
    else:
        print(f"  [alpha μ] alpha_scores.csv not found — using historical mean returns")
    n = len(tickers_used)
    print(f"      {n} tickers  ×  {len(returns)} trading days")

    vol_annual = np.sqrt(np.diag(cov))
    print("      Annualised vols: "
          + "  ".join(f"{t}={v*100:.0f}%" for t, v in list(zip(tickers_used, vol_annual))[:6])
          + (" …" if n > 6 else ""))

    # ── 3. Optimize ───────────────────────────────────────────────────────────
    print("\n[3/6] Solving max-Sharpe, min-variance, risk-parity, turnover-aware …")

    w_ms  = optimize_max_sharpe(mu, cov, rf=rf, cap=cap)
    w_mv  = optimize_min_variance(mu, cov, cap=cap)
    w_rp  = optimize_risk_parity(cov, cap=cap)
    w_iv  = inverse_vol_weights(returns, cap=cap)

    # ── Turnover-aware (factor-neutral) ──────────────────────────────────────
    w_prev = load_previous_weights(tickers_used)
    if w_prev is None:
        print("      [turnover] No prior weights found — using equal weight as baseline")
        w_prev = np.ones(n) / n

    # Load sector map — prefer sector_map.csv (step88, full GICS coverage)
    sector_map: dict[str, str] = {}

    sm_path = ROOT / "sector_map.csv"
    if sm_path.exists():
        try:
            sm_df = pd.read_csv(sm_path)
            if {"ticker", "sector"}.issubset(sm_df.columns):
                for _, row in sm_df.iterrows():
                    t = str(row["ticker"]).strip()
                    s = str(row["sector"]).strip()
                    if s and s not in ("Other", "", "nan"):
                        sector_map[t] = s
                print(f"      [sector] {len(sector_map)} sector mappings from sector_map.csv")
        except Exception as e:
            print(f"      [sector] sector_map.csv error: {e}")

    # Fallback: fundamental_features_cache.json (step68)
    if not sector_map:
        fund_cache = ROOT / "fundamental_features_cache.json"
        if fund_cache.exists():
            try:
                import json
                raw = json.loads(fund_cache.read_text())
                for t, rec in raw.get("data", {}).items():
                    if isinstance(rec, dict) and rec.get("sector"):
                        sector_map[t] = rec["sector"]
                if sector_map:
                    print(f"      [sector] {len(sector_map)} mappings from fundamentals cache")
            except Exception:
                pass

    w_ta = optimize_turnover_aware(
        mu, cov, w_prev, rf=rf, cap=cap, lam=TURNOVER_LAMBDA,
        tickers=tickers_used, sector_map=sector_map or None,
    )
    turnover_1way = float(np.sum(np.abs(w_ta - w_prev))) / 2.0
    print(f"      [turnover] one-way turnover vs prior: {turnover_1way*100:.1f}%")

    stats: dict[str, dict] = {}
    for label, w in [("max_sharpe", w_ms), ("min_var", w_mv),
                      ("risk_parity", w_rp), ("inv_vol", w_iv),
                      ("turnover_aware", w_ta)]:
        r, v, s = _port_stats(w, mu, cov, rf)
        stats[label] = {"ret": r, "vol": v, "sharpe": s,
                         "max_w": float(w.max())}
        print(f"      {label:17s}  ret={r*100:+.2f}%  vol={v*100:.2f}%  "
              f"Sharpe={s:.3f}  maxW={w.max()*100:.1f}%")

    # ── 4. Efficient frontier ─────────────────────────────────────────────────
    print("\n[4/6] Computing efficient frontier …")
    ef_df = compute_efficient_frontier(mu, cov, cap=cap, n_points=EF_POINTS)
    print(f"      {len(ef_df)} valid frontier points")
    if not ef_df.empty:
        best_ef = ef_df.loc[ef_df["sharpe"].idxmax()]
        print(f"      Best EF Sharpe={best_ef['sharpe']:.3f}  "
              f"ret={best_ef['ret']*100:.2f}%  vol={best_ef['vol']*100:.2f}%")

    # ── 5. Write outputs ──────────────────────────────────────────────────────
    print("\n[5/6] Writing output files …")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Weights table
    weights_df = pd.DataFrame({
        "ticker":          tickers_used,
        "max_sharpe":      w_ms,
        "min_var":         w_mv,
        "risk_parity":     w_rp,
        "inv_vol":         w_iv,
        "turnover_aware":  w_ta,
        "mu_annual":       mu,
        "vol_annual":      vol_annual,
    })
    weights_df = weights_df.sort_values("max_sharpe", ascending=False).reset_index(drop=True)
    wp = ROOT / "portfolio_optimized_weights.csv"
    weights_df.to_csv(wp, index=False)
    print(f"  [weights] {wp}")

    # Covariance matrix
    cov_df = pd.DataFrame(cov, index=tickers_used, columns=tickers_used)
    cp = ROOT / "portfolio_covariance.csv"
    cov_df.to_csv(cp)
    print(f"  [covariance] {cp}")

    # Efficient frontier
    ef_p = ROOT / "portfolio_ef_points.csv"
    ef_df.to_csv(ef_p, index=False)
    print(f"  [frontier] {ef_p}")

    # Report
    write_report(weights_df, ef_df, stats, lookback, rf, ts)

    # ── daily_final.csv — unified alpha scores + risk-model weights ───────────
    # Merges daily_picks.csv (Step 87 alpha scores) with optimizer weights so
    # users have one single table: ticker → alpha score + optimal allocation.
    daily_picks_path = ROOT / "daily_picks.csv"
    if daily_picks_path.exists():
        try:
            picks_df = pd.read_csv(daily_picks_path)
            merge_cols = [c for c in ["ticker", "max_sharpe", "turnover_aware",
                                      "risk_parity", "mu_annual", "vol_annual"]
                          if c in weights_df.columns]
            final_df = picks_df.merge(weights_df[merge_cols], on="ticker", how="left")
            # Add human-readable % columns
            for col in ["max_sharpe", "turnover_aware", "risk_parity"]:
                if col in final_df.columns:
                    final_df[f"{col}_pct"] = (final_df[col] * 100).round(2)
            if "vol_annual" in final_df.columns:
                final_df["vol_pct"] = (final_df["vol_annual"] * 100).round(1)
            final_path = ROOT / "daily_final.csv"
            final_df.to_csv(final_path, index=False)
            print(f"  [daily_final] {final_path}  ({len(final_df)} rows)")
        except Exception as e:
            print(f"  [daily_final] WARNING: {e}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\n[6/6] Done in {time.time()-t0:.1f}s")
    print(f"\n{'─'*60}")
    print("PORTFOLIO OPTIMIZER RESULTS")
    print(f"{'─'*60}")
    print(f"{'Portfolio':<18} {'Return':>8} {'Vol':>8} {'Sharpe':>8}")
    print(f"{'─'*60}")
    for label, pname in [("max_sharpe","Max-Sharpe"), ("min_var","Min-Variance"),
                          ("risk_parity","Risk-Parity"), ("inv_vol","Inv-Vol (base)"),
                          ("turnover_aware","Turnover-Aware")]:
        s = stats[label]
        print(f"{pname:<18} {s['ret']*100:>7.2f}%  {s['vol']*100:>6.2f}%  {s['sharpe']:>8.3f}")

    ms_improvement = stats["max_sharpe"]["sharpe"] - stats["inv_vol"]["sharpe"]
    print(f"\nSharpe improvement over inv-vol: {ms_improvement:+.3f} "
          f"({'BETTER' if ms_improvement > 0 else 'WORSE'})")
    print(f"{'='*60}\n")

    return {"weights_df": weights_df, "ef_df": ef_df,
            "stats": stats, "tickers": tickers_used, "cov": cov_df,
            "w_prev": w_prev, "sector_map": sector_map}


def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 63 — Portfolio Optimizer")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                        help=f"Lookback window in trading days (default {LOOKBACK_DAYS})")
    parser.add_argument("--rf", type=float, default=RF_ANNUAL,
                        help=f"Annual risk-free rate (default {RF_ANNUAL})")
    parser.add_argument("--cap", type=float, default=POSITION_CAP,
                        help=f"Max position weight (default {POSITION_CAP})")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Custom ticker list (default: CORE_TICKERS)")
    args = parser.parse_args()

    MAX_OPT_TICKERS = 60   # max tickers passed to optimizer; above this we pre-filter by alpha

    if args.tickers:
        tickers = args.tickers
    else:
        # Prefer the unified universe from Step 87 alpha_scores.csv
        alpha_path = ROOT / "alpha_scores.csv"
        if alpha_path.exists():
            try:
                alpha_df = pd.read_csv(alpha_path)
                all_tickers = alpha_df["ticker"].dropna().tolist()

                if len(all_tickers) > MAX_OPT_TICKERS:
                    # Pre-filter: keep top-N by alpha_score so covariance matrix is well-conditioned.
                    # With lookback=252 days, we need N << 252 for non-singular covariance.
                    # Top-60 gives T/N ≈ 4× — comfortable for LedoitWolf + SLSQP.
                    top_alpha = (
                        alpha_df.sort_values("alpha_score", ascending=False)
                        .head(MAX_OPT_TICKERS)["ticker"]
                        .tolist()
                    )
                    print(f"[universe] alpha_scores.csv has {len(all_tickers)} tickers — "
                          f"pre-filtering to top {MAX_OPT_TICKERS} by alpha score")
                    print(f"           (T/N ratio: {args.lookback}/{MAX_OPT_TICKERS} = "
                          f"{args.lookback/MAX_OPT_TICKERS:.1f}× — well-conditioned covariance)")
                    tickers = top_alpha
                else:
                    print(f"[universe] Using {len(all_tickers)} tickers from alpha_scores.csv")
                    tickers = all_tickers
            except Exception:
                tickers = CORE_TICKERS
        else:
            print(f"[universe] alpha_scores.csv not found — using {len(CORE_TICKERS)} CORE_TICKERS")
            print("           Run Step 87 first for full alpha-ranked universe")
            tickers = CORE_TICKERS

    run_optimizer(tickers=tickers, lookback=args.lookback,
                  rf=args.rf, cap=args.cap)


if __name__ == "__main__":
    main()
