"""
W26-W27: Barra-Style 10-Factor Risk Model
==========================================
Builds a simplified Barra USE3-style factor model using free, publicly available data.

The 10 factors (W26: Size/Value/Growth, W27: Leverage/Liquidity + 5 more):

  Style factors (from price + EDGAR PIT data):
    1. Size         — log(market_cap), negative loading = small-cap tilt
    2. Value        — book-to-price (B/P), from EDGAR total equity / market_cap
    3. Growth       — 3-year EPS growth rate, from EDGAR
    4. Leverage     — total debt / total assets, from EDGAR PIT
    5. Liquidity    — average daily $ volume (21-day), from price cache

  Momentum factors (from price cache):
    6. Momentum     — 12-1 month price momentum (skip 1 month)
    7. Volatility   — 63-day realized volatility (low = low risk)
    8. Beta         — 252-day rolling beta to SPY

  Sector factors (from S&P 500 sector map):
    9. Sector_Tech   — binary: GICS sector = Information Technology
    10. Sector_Fin   — binary: GICS sector = Financials

  (Additional sector factors handled by sector_map exposure in risk attribution)

Uses:
  - Cross-sectional regression: r_i = Σ f_k * β_{ik} + ε_i
  - Factor covariance matrix Σ_F = cov(factor returns over rolling window)
  - Specific risk = residual std dev from factor regression

Outputs:
  barra_factor_exposures.csv  — ticker × 10 factor exposures
  barra_factor_cov.csv        — 10×10 factor covariance matrix
  barra_specific_risk.csv     — ticker × specific_vol (annualised %)

Usage:
    from risk.barra import build_barra_model, get_factor_exposures, compute_portfolio_risk
    model = build_barra_model()
    risk  = compute_portfolio_risk(weights, model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent

# Factor names in canonical order
FACTOR_NAMES = [
    "size", "value", "growth", "leverage", "liquidity",
    "momentum", "volatility", "beta",
    "sector_tech", "sector_fin",
]

N_FACTORS    = len(FACTOR_NAMES)
COV_WINDOW   = 252   # trading days for factor covariance estimation
BETA_WINDOW  = 252
EWMA_HALFLIFE = 63   # exponential weighting for covariance (63d halflife)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Factor exposure construction
# ─────────────────────────────────────────────────────────────────────────────

def _compute_size(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """log(market_cap) proxy = log(price * shares outstanding from EDGAR)."""
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        # Fallback: use log(price) as size proxy (highly correlated with market cap)
        p = prices[prices.index <= as_of]
        if p.empty:
            return pd.Series(dtype=float)
        last_price = p.iloc[-1]
        return np.log(last_price.clip(lower=0.01)).dropna()

    pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
    pit_df = pit_df[pit_df["know_date"] <= as_of].copy()

    if "concept" in pit_df.columns:
        shares = pit_df[pit_df["concept"] == "shares_out"].copy()
        shares = shares.sort_values("know_date").groupby("ticker").last()[["value"]]
        shares = shares.rename(columns={"value": "shares_out"})
    elif "shares_out" in pit_df.columns:
        shares = pit_df.sort_values("know_date").groupby("ticker").last()[["shares_out"]]
    else:
        shares = pd.DataFrame()

    p = prices[prices.index <= as_of]
    if p.empty:
        return pd.Series(dtype=float)
    last_price = p.iloc[-1]

    if not shares.empty:
        mktcap = last_price * shares["shares_out"].reindex(last_price.index)
        return np.log(mktcap.clip(lower=1e6)).dropna()
    else:
        return np.log(last_price.clip(lower=0.01)).dropna()


def _compute_value(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Book-to-price = total_equity / market_cap (from EDGAR PIT)."""
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.Series(dtype=float)

    pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
    pit_df = pit_df[pit_df["know_date"] <= as_of].copy()

    p = prices[prices.index <= as_of]
    if p.empty:
        return pd.Series(dtype=float)
    last_price = p.iloc[-1]

    try:
        if "concept" in pit_df.columns:
            liab_df  = pit_df[pit_df["concept"] == "total_liabilities"].sort_values("know_date")
            assets_df = pit_df[pit_df["concept"] == "total_assets"].sort_values("know_date")
            liab  = liab_df.groupby("ticker")["value"].last()
            assets = assets_df.groupby("ticker")["value"].last()
            book_equity = assets - liab
        elif "total_assets" in pit_df.columns and "total_liabilities" in pit_df.columns:
            last = pit_df.sort_values("know_date").groupby("ticker").last()
            book_equity = last["total_assets"] - last["total_liabilities"]
        else:
            return pd.Series(dtype=float)

        bp = book_equity / last_price.reindex(book_equity.index)
        return bp.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _compute_growth(as_of: pd.Timestamp) -> pd.Series:
    """3-year EPS growth proxy from EDGAR PIT (SUE z-score as growth measure)."""
    eps_path = ROOT / "eps_revision_scores.csv"
    if eps_path.exists():
        df = pd.read_csv(eps_path)
        if "ticker" in df.columns and "sue_score" in df.columns:
            return df.set_index("ticker")["sue_score"].dropna()

    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.Series(dtype=float)

    pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
    pit_df = pit_df[pit_df["know_date"] <= as_of].copy()

    try:
        if "concept" in pit_df.columns:
            eps_df = pit_df[pit_df["concept"] == "eps_basic"].copy()
        elif "eps_basic" in pit_df.columns:
            eps_df = pit_df.copy()
            eps_df = eps_df.rename(columns={"eps_basic": "value"})
        else:
            return pd.Series(dtype=float)

        eps_df = eps_df.sort_values(["ticker", "period_end"])
        growth = {}
        for tkr, g in eps_df.groupby("ticker"):
            if len(g) >= 4:
                recent = float(g["value"].iloc[-1])
                old    = float(g["value"].iloc[-5]) if len(g) >= 5 else float(g["value"].iloc[0])
                if abs(old) > 0.01:
                    growth[tkr] = (recent - old) / abs(old)
        return pd.Series(growth).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _compute_leverage(as_of: pd.Timestamp) -> pd.Series:
    """Total debt / total assets from EDGAR PIT."""
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.Series(dtype=float)

    pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
    pit_df = pit_df[pit_df["know_date"] <= as_of].copy()

    try:
        if "concept" in pit_df.columns:
            debt_df   = pit_df[pit_df["concept"] == "total_liabilities"].sort_values("know_date")
            assets_df = pit_df[pit_df["concept"] == "total_assets"].sort_values("know_date")
            debt   = debt_df.groupby("ticker")["value"].last()
            assets = assets_df.groupby("ticker")["value"].last()
        elif "total_liabilities" in pit_df.columns and "total_assets" in pit_df.columns:
            last   = pit_df.sort_values("know_date").groupby("ticker").last()
            debt   = last["total_liabilities"]
            assets = last["total_assets"]
        else:
            return pd.Series(dtype=float)

        leverage = debt / assets.where(assets > 0, np.nan)
        return leverage.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _compute_liquidity(prices: pd.DataFrame, as_of: pd.Timestamp,
                       window: int = 21) -> pd.Series:
    """Average daily dollar volume (21d). Higher = more liquid."""
    p = prices[prices.index <= as_of]
    if len(p) < window:
        return pd.Series(dtype=float)
    price_slice = p.iloc[-window:]
    # Use price as dollar-volume proxy (actual volume not available in price cache)
    return price_slice.mean().dropna()


def _compute_momentum(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """12-1 month momentum (skip 1 month)."""
    p = prices[prices.index <= as_of]
    if len(p) < 273:
        return pd.Series(dtype=float)
    return (p.iloc[-1] / p.iloc[-252] - 1) - (p.iloc[-1] / p.iloc[-22] - 1)


def _compute_volatility(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """63-day realized volatility (annualised). Lower = lower risk."""
    p = prices[prices.index <= as_of]
    if len(p) < 63:
        return pd.Series(dtype=float)
    return p.pct_change().iloc[-63:].std() * np.sqrt(252)


def _compute_beta(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """252-day rolling beta to SPY."""
    p = prices[prices.index <= as_of]
    if len(p) < BETA_WINDOW or "SPY" not in p.columns:
        return pd.Series(1.0, index=[c for c in prices.columns if c != "SPY"])

    rets = p.pct_change().iloc[-BETA_WINDOW:].dropna()
    spy_ret = rets["SPY"]
    betas = {}
    spy_var = spy_ret.var()
    for col in rets.columns:
        if col == "SPY":
            continue
        cov = rets[col].cov(spy_ret)
        betas[col] = cov / (spy_var + 1e-12)
    return pd.Series(betas).dropna()


def _compute_sector_factors(as_of: pd.Timestamp, tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    """Sector binary factors: sector_tech, sector_fin."""
    sector_map = {}
    try:
        from data.sp500_constituents import build_constituent_history, get_sector_map
        history = build_constituent_history()
        sector_map = get_sector_map(history, as_of).to_dict()
    except Exception:
        # Fallback: read from sp500_constituents_history.csv if available
        p = ROOT / "sp500_constituents_history.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "ticker" in df.columns and "sector" in df.columns:
                sector_map = df.set_index("ticker")["sector"].to_dict()

    tech_sigs = {t: 1.0 if "Information Technology" in sector_map.get(t, "") else 0.0
                 for t in tickers}
    fin_sigs  = {t: 1.0 if "Financials" in sector_map.get(t, "") else 0.0
                 for t in tickers}
    return pd.Series(tech_sigs), pd.Series(fin_sigs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Factor model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def _z_score(s: pd.Series) -> pd.Series:
    mu, std = s.mean(), s.std()
    return (s - mu) / (std + 1e-9) if std > 1e-9 else s * 0


def build_factor_exposures(
    as_of: Optional[pd.Timestamp] = None,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the factor exposure matrix B (N tickers × K factors).

    Each factor is z-scored cross-sectionally (mean=0, std=1) after winsorizing.

    Returns DataFrame: index=ticker, columns=FACTOR_NAMES.
    """
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    if prices is None:
        for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
            p = ROOT / fname
            if p.exists():
                prices = pd.read_csv(p, index_col=0, parse_dates=True)
                break
    if prices is None:
        raise FileNotFoundError("No price cache found")

    tickers = [c for c in prices.columns if c != "SPY"]

    # Compute raw factor exposures
    raw = {
        "size":       _compute_size(prices[tickers + (["SPY"] if "SPY" in prices.columns else [])], as_of),
        "value":      _compute_value(prices, as_of),
        "growth":     _compute_growth(as_of),
        "leverage":   _compute_leverage(as_of),
        "liquidity":  _compute_liquidity(prices[tickers], as_of),
        "momentum":   _compute_momentum(prices[tickers], as_of),
        "volatility": _compute_volatility(prices[tickers], as_of),
        "beta":       _compute_beta(prices, as_of),
    }

    sector_tech, sector_fin = _compute_sector_factors(as_of, tickers)
    raw["sector_tech"] = sector_tech
    raw["sector_fin"]  = sector_fin

    # Build DataFrame, winsorize + z-score each factor
    B = pd.DataFrame(index=tickers)
    for fname in FACTOR_NAMES:
        s = raw.get(fname, pd.Series(dtype=float))
        if s.empty:
            B[fname] = 0.0
            continue
        s = s.reindex(tickers)
        if fname not in ("sector_tech", "sector_fin"):
            s = _winsorize(s.dropna())
            s = _z_score(s)
        B[fname] = s.fillna(0.0)

    return B


def compute_factor_returns(
    prices: pd.DataFrame,
    n_periods: int = COV_WINDOW,
) -> pd.DataFrame:
    """
    Compute factor returns via cross-sectional regression at each period.

    Returns DataFrame: index=date, columns=FACTOR_NAMES (factor returns).
    """
    rets = prices.pct_change().dropna()
    dates = rets.index[-n_periods:] if len(rets) >= n_periods else rets.index
    tickers = [c for c in prices.columns if c != "SPY"]

    factor_rets = []
    print(f"  [Barra] Computing factor returns over {len(dates)} periods...")

    for t_idx in range(0, len(dates), 21):  # monthly frequency
        as_of = pd.Timestamp(dates[t_idx])
        price_t = prices[prices.index <= as_of]
        if len(price_t) < 252:
            continue

        try:
            B = build_factor_exposures(as_of=as_of, prices=price_t)
        except Exception:
            continue

        # Forward 21-day stock return
        fwd_idx = t_idx + 21
        if fwd_idx >= len(dates):
            continue

        t_end = pd.Timestamp(dates[fwd_idx])
        r = (prices.loc[t_end, tickers] / prices.loc[as_of, tickers] - 1).dropna()

        common = B.index.intersection(r.index)
        if len(common) < 30:
            continue

        B_c = B.loc[common].values
        r_c = r[common].values

        # WLS: equal weights (could use market cap weights)
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(B_c, r_c)
            fret = dict(zip(FACTOR_NAMES, reg.coef_))
            fret["date"] = as_of
            factor_rets.append(fret)
        except Exception:
            continue

    if not factor_rets:
        return pd.DataFrame(columns=["date"] + FACTOR_NAMES)

    return pd.DataFrame(factor_rets).set_index("date")


def compute_factor_covariance(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute factor covariance matrix using EWMA (exponential weighting).
    Halflife = EWMA_HALFLIFE trading days.
    """
    if factor_returns.empty:
        return pd.DataFrame(np.eye(N_FACTORS), index=FACTOR_NAMES, columns=FACTOR_NAMES)

    available_factors = [f for f in FACTOR_NAMES if f in factor_returns.columns]
    fr = factor_returns[available_factors]

    # EWMA covariance
    n = len(fr)
    decay = np.exp(-np.log(2) / EWMA_HALFLIFE)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    centered = fr - (fr * weights[:, None]).sum()
    cov = (fr.values * weights[:, None]).T @ fr.values
    cov_df = pd.DataFrame(cov, index=available_factors, columns=available_factors)

    # Fill in zeros for unavailable factors
    full_cov = pd.DataFrame(0.0, index=FACTOR_NAMES, columns=FACTOR_NAMES)
    for f1 in available_factors:
        for f2 in available_factors:
            full_cov.loc[f1, f2] = cov_df.loc[f1, f2]
    # Set diagonal to 1e-6 for unavailable factors (small non-zero for numerical stability)
    for f in FACTOR_NAMES:
        if f not in available_factors:
            full_cov.loc[f, f] = 1e-6

    return full_cov


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_barra_model(
    as_of: Optional[pd.Timestamp] = None,
    force_refresh: bool = False,
) -> dict:
    """
    Build the complete Barra factor model for the current date.

    Returns dict with:
        B          — factor exposure matrix (DataFrame: ticker × factor)
        Sigma_F    — factor covariance matrix (DataFrame: factor × factor)
        spec_var   — specific variance (Series: ticker)
        factor_rets — factor return history (DataFrame: date × factor)
    """
    import time as _time

    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    exp_path = ROOT / "barra_factor_exposures.csv"
    cov_path = ROOT / "barra_factor_cov.csv"
    spec_path = ROOT / "barra_specific_risk.csv"

    if not force_refresh and exp_path.exists() and cov_path.exists():
        age = (_time.time() - exp_path.stat().st_mtime) / 86400
        if age < 1.5:
            B       = pd.read_csv(exp_path, index_col=0)
            Sigma_F = pd.read_csv(cov_path, index_col=0)
            spec    = pd.read_csv(spec_path, index_col=0).squeeze() if spec_path.exists() else pd.Series(dtype=float)
            print(f"  [Barra] Loaded from cache ({age:.1f}d old): {B.shape}")
            return {"B": B, "Sigma_F": Sigma_F, "spec_var": spec, "factor_rets": pd.DataFrame()}

    # Load prices
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            break
    else:
        raise FileNotFoundError("No price cache found")

    print(f"  [Barra] Building factor exposures as of {as_of.date()}...")
    B = build_factor_exposures(as_of=as_of, prices=prices)
    print(f"  [Barra] Factor exposures: {B.shape} tickers × {N_FACTORS} factors")

    print("  [Barra] Computing factor return history...")
    factor_rets = compute_factor_returns(prices, n_periods=COV_WINDOW)
    Sigma_F = compute_factor_covariance(factor_rets)

    # Specific variance: variance of stock returns unexplained by factors
    tickers = B.index.tolist()
    rets = prices[tickers].pct_change().iloc[-252:].dropna()
    spec_vols = {}
    B_arr = B.values
    for i, tkr in enumerate(tickers):
        if tkr not in rets.columns:
            spec_vols[tkr] = 0.20  # fallback: 20% annual vol
            continue
        r_stock = rets[tkr].dropna()
        if len(r_stock) < 21:
            spec_vols[tkr] = 0.20
            continue
        total_vol = float(r_stock.std() * np.sqrt(252))
        # Factor-explained variance
        b_i = B_arr[i]
        factor_var = float(b_i @ Sigma_F.values @ b_i)
        specific_var = max(total_vol ** 2 - factor_var, 0.001)
        spec_vols[tkr] = float(np.sqrt(specific_var))

    spec = pd.Series(spec_vols, name="specific_vol")

    # Save outputs
    B.to_csv(exp_path)
    Sigma_F.to_csv(cov_path)
    spec.to_frame().to_csv(spec_path)
    print(f"  [Barra] Saved → {exp_path.name}, {cov_path.name}, {spec_path.name}")

    return {"B": B, "Sigma_F": Sigma_F, "spec_var": spec, "factor_rets": factor_rets}


def get_factor_exposures(tickers: list[str]) -> pd.DataFrame:
    """Load factor exposures for specific tickers from cache."""
    p = ROOT / "barra_factor_exposures.csv"
    if not p.exists():
        return pd.DataFrame()
    B = pd.read_csv(p, index_col=0)
    return B.reindex(tickers).fillna(0.0)


def compute_portfolio_risk(
    weights: pd.Series,
    model: Optional[dict] = None,
) -> dict:
    """
    Compute total, factor, and specific portfolio risk.

    Args:
        weights: Portfolio weights (Series: ticker → weight, sum = 1).
        model:   Barra model dict. If None, loads from cache files.

    Returns dict:
        total_vol        — annualised portfolio volatility
        factor_vol       — component from systematic factors
        specific_vol     — component from idiosyncratic risk
        factor_exposures — portfolio-level factor exposures
    """
    if model is None:
        exp_path = ROOT / "barra_factor_exposures.csv"
        cov_path = ROOT / "barra_factor_cov.csv"
        spec_path = ROOT / "barra_specific_risk.csv"
        if not all(p.exists() for p in [exp_path, cov_path]):
            return {}
        B       = pd.read_csv(exp_path, index_col=0)
        Sigma_F = pd.read_csv(cov_path, index_col=0)
        spec    = pd.read_csv(spec_path, index_col=0).squeeze() if spec_path.exists() else pd.Series(dtype=float)
    else:
        B, Sigma_F, spec = model["B"], model["Sigma_F"], model["spec_var"]

    common_tickers = weights.index.intersection(B.index)
    if len(common_tickers) == 0:
        return {}

    w = weights[common_tickers].values
    B_p = B.loc[common_tickers].values

    # Portfolio factor exposures
    port_exposures = B_p.T @ w  # shape (K,)

    # Factor risk contribution
    Sigma = Sigma_F.values
    factor_var = float(port_exposures @ Sigma @ port_exposures)

    # Specific risk contribution
    spec_aligned = spec.reindex(common_tickers).fillna(0.20)
    specific_var = float((w ** 2 * spec_aligned.values ** 2).sum())

    total_var = factor_var + specific_var

    return {
        "total_vol":        float(np.sqrt(max(total_var, 0))),
        "factor_vol":       float(np.sqrt(max(factor_var, 0))),
        "specific_vol":     float(np.sqrt(max(specific_var, 0))),
        "factor_pct":       factor_var / (total_var + 1e-12),
        "specific_pct":     specific_var / (total_var + 1e-12),
        "factor_exposures": dict(zip(FACTOR_NAMES, port_exposures.tolist())),
        "n_tickers":        len(common_tickers),
    }


if __name__ == "__main__":
    print("W26-W27: Barra 10-Factor Risk Model")
    print("=" * 50)
    model = build_barra_model(force_refresh=True)
    B = model["B"]
    print(f"\nFactor exposure summary (first 5 tickers):")
    print(B.head(5).round(2).to_string())
    print(f"\nFactor covariance matrix:")
    print(model["Sigma_F"].round(4).to_string())

    # Test portfolio risk computation
    n = min(25, len(B))
    w = pd.Series(1.0 / n, index=B.index[:n])
    risk = compute_portfolio_risk(w, model)
    print(f"\nEqual-weight portfolio ({n} stocks):")
    print(f"  Total vol:    {risk.get('total_vol', 0):.1%}")
    print(f"  Factor vol:   {risk.get('factor_vol', 0):.1%}  ({risk.get('factor_pct', 0):.0%})")
    print(f"  Specific vol: {risk.get('specific_vol', 0):.1%}  ({risk.get('specific_pct', 0):.0%})")
    print(f"  Factor exposures: {risk.get('factor_exposures', {})}")
