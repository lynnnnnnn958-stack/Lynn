#!/usr/bin/env python3
"""
Canyon v9 — Step 88: Simplified Barra-Style Factor Risk Model
=============================================================
Builds a 10-factor + 8-sector risk model (simplified Barra GEMX).
Replaces the simple beta cap with a proper factor covariance decomposition.

Factors (10 style + 8 sector = 18 total):
  Style:
    1. market_beta   — systematic beta vs SPY (60-day rolling)
    2. size          — log(market_cap)
    3. value         — earnings yield (E/P) or book/price
    4. momentum      — 12-month - 1-month return (standard UMD)
    5. quality       — low accruals proxy (from accruals_scores.csv or ROE)
    6. low_vol       — inverse 60-day realized volatility
    7. short_interest — days-to-cover (short squeeze intensity)
    8. analyst_revision — analyst upgrade/downgrade momentum
  Sector (dummy variables, 8 GICS):
    9.  tech_dummy       XLK
    10. health_dummy     XLV
    11. financial_dummy  XLF
    12. consumer_dummy   XLY
    13. industrial_dummy XLI
    14. energy_dummy     XLE
    15. materials_dummy  XLB
    16. utility_dummy    XLU

Portfolio risk decomposition:
  Total variance = w' (B F B' + Δ) w
  Factor risk    = w' B F B' w
  Specific risk  = w' Δ w

Outputs:
  factor_exposures.csv      — B matrix (n_stocks × n_factors), normalized z-scores
  factor_cov.csv            — F matrix (n_factors × n_factors), annualized
  specific_risk.csv         — per-ticker annualized idiosyncratic vol
  portfolio_risk_decomp.csv — factor vs specific risk for top long/short portfolio
  factor_risk_report.md     — daily risk attribution report

Usage:
  python3 canyon_final_v9_step88_factor_risk_model.py
  python3 canyon_final_v9_step88_factor_risk_model.py --refresh
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
OUT_EXPO   = ROOT / "factor_exposures.csv"
OUT_FCOV   = ROOT / "factor_cov.csv"
OUT_SRISK  = ROOT / "specific_risk.csv"
OUT_DECOMP = ROOT / "portfolio_risk_decomp.csv"
OUT_REPORT = ROOT / "factor_risk_report.md"

FRESHNESS_DAYS = 1       # regenerate daily (prices change daily)
LOOKBACK_DAYS  = 252     # 1 year for factor covariance estimation
BETA_WINDOW    = 60      # rolling beta window
VOL_WINDOW     = 60      # realized vol window
MOM_LONG       = 252     # 12-month momentum lookback
MOM_SHORT      = 21      # skip 1 month (standard UMD)
MIN_STOCKS     = 20

SECTORS = {
    "XLK": "tech",
    "XLV": "health",
    "XLF": "financial",
    "XLY": "consumer",
    "XLI": "industrial",
    "XLE": "energy",
    "XLB": "materials",
    "XLU": "utility",
}

SECTOR_TICKERS: dict[str, list[str]] = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","ORCL","INTC","CSCO","AMAT","TXN",
             "QCOM","NOW","INTU","CRM","IBM","MU","ADI","KLAC","LRCX","MRVL"],
    "XLV": ["UNH","JNJ","LLY","ABT","MRK","TMO","AMGN","DHR","SYK","MDT",
             "ISRG","VRTX","ELV","CVS","CI","HCA","REGN","BIIB","GILD","IDXX"],
    "XLF": ["JPM","BAC","WFC","GS","MS","BLK","AXP","C","BK","MMC",
             "CB","PGR","AON","SCHW","ICE","CME","TRV","MET","PRU","AFL"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","BKNG","LOW","CMG",
             "YUM","ORLY","AZO","DHI","GM","F","LEN","ROST","MAR","HLT"],
    "XLI": ["HON","UNP","CAT","GE","BA","DE","WM","ETN","CSX","NOC",
             "RTX","LMT","GD","FDX","UPS","EMR","PH","ROP","CARR","OTIS"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","PXD",
             "HES","DVN","BKR","HAL","APA","FANG","MRO","CTRA","PR","DT"],
    "XLB": ["LIN","APD","SHW","ECL","NEM","FCX","NUE","CF","MOS","IFF",
             "PPG","DD","ALB","RPM","CE","EMN","WLK","PKG","IP","SEE"],
    "XLU": ["NEE","DUK","SO","D","AEP","EXC","SRE","PEG","XEL","ED",
             "WEC","DTE","ETR","FE","PPL","AEE","CMS","NI","LNT","EVRG"],
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices(tickers: list[str], days: int = 300) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(["SPY"] + tickers, period=f"{days}d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].dropna(how="all", axis=1)
        return raw.dropna(how="all", axis=1)
    except Exception:
        pass

    # Fallback: price cache
    for fname in ("backtest_price_cache.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            avail = [t for t in ["SPY"] + tickers if t in df.columns]
            return df[avail].tail(days)
    return pd.DataFrame()


def load_market_caps() -> pd.Series:
    """Market cap from yfinance fast_info for universe tickers."""
    caps: dict[str, float] = {}
    import yfinance as yf
    from collections import defaultdict

    all_tickers = [tk for tks in SECTOR_TICKERS.values() for tk in tks]
    for tk in all_tickers[:50]:  # limit to avoid rate limits
        try:
            info = yf.Ticker(tk).fast_info
            mc = getattr(info, "market_cap", None)
            if mc and mc > 0:
                caps[tk] = float(mc)
        except Exception:
            pass
    return pd.Series(caps)


def load_accruals() -> pd.Series:
    """Load accruals z-score (proxy for quality; lower accruals = higher quality)."""
    for fname in ("accruals_scores.csv", "accruals_snapshot.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "ticker" not in df.columns:
            continue
        df = df.set_index("ticker")
        col = next((c for c in ("accrual_zscore","accruals_z","z_score")
                    if c in df.columns), None)
        if col:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def load_xbrl_fundamentals() -> pd.DataFrame:
    """Load XBRL fundamentals from step89 output."""
    p = ROOT / "xbrl_fundamentals.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "ticker" in df.columns:
            return df.set_index("ticker")
    except Exception:
        pass
    return pd.DataFrame()


def load_analyst_revisions() -> pd.Series:
    """Analyst revision signal from analyst_revisions.csv or alpha_scores."""
    for p_name, col in [("analyst_revisions.csv", "revision_score"),
                         ("alpha_scores.csv",       "sig_revision")]:
        p = ROOT / p_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "ticker" in df.columns and col in df.columns:
            return pd.to_numeric(
                df.set_index("ticker")[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def load_short_interest() -> pd.Series:
    """Short interest from alpha_scores or options_signals."""
    for p_name, col in [("alpha_scores.csv",    "short_interest_ratio"),
                         ("options_signals.csv", "short_ratio")]:
        p = ROOT / p_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "ticker" in df.columns and col in df.columns:
            return pd.to_numeric(
                df.set_index("ticker")[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


# ── Factor exposure computation ───────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    """Winsorize at ±3σ then z-score."""
    mu, sd = s.mean(), s.std()
    if sd < 1e-9:
        return s * 0.0
    s2 = s.clip(mu - 3*sd, mu + 3*sd)
    return ((s2 - s2.mean()) / s2.std()).round(4)


def compute_factor_exposures(
    prices:     pd.DataFrame,
    mkt_caps:   pd.Series,
    accruals:   pd.Series,
    revisions:  pd.Series,
    short_int:  pd.Series,
    xbrl:       pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns B matrix: DataFrame(index=ticker, columns=factor_names).
    All exposures are cross-sectionally z-scored.
    """
    if "SPY" not in prices.columns or len(prices) < BETA_WINDOW + 10:
        return pd.DataFrame()

    rets = prices.pct_change(fill_method=None).dropna()
    spy  = rets["SPY"]
    stk_rets = rets.drop(columns=["SPY"], errors="ignore")
    tickers  = stk_rets.columns.tolist()

    rows: dict[str, dict] = {tk: {} for tk in tickers}

    # ── 1. Market beta (60-day rolling, use last window)
    if len(rets) >= BETA_WINDOW:
        r_win  = rets.tail(BETA_WINDOW)
        spy_w  = r_win["SPY"]
        spy_var = float(spy_w.var())
        for tk in tickers:
            if tk in r_win.columns:
                cov = float(r_win[tk].cov(spy_w))
                rows[tk]["market_beta"] = cov / (spy_var + 1e-12)
            else:
                rows[tk]["market_beta"] = 1.0
    else:
        for tk in tickers:
            rows[tk]["market_beta"] = 1.0

    # ── 2. Size (log market cap)
    log_cap = np.log(mkt_caps.clip(lower=1e6) + 1)
    for tk in tickers:
        rows[tk]["size"] = float(log_cap.get(tk, np.nan))

    # ── 3. Momentum (12m - 1m, skip last month)
    if len(prices) >= MOM_LONG + 5:
        px_long  = prices.iloc[-(MOM_LONG+1)]
        px_short = prices.iloc[-(MOM_SHORT+1)]
        px_now   = prices.iloc[-1]
        for tk in tickers:
            if tk in prices.columns:
                ret_long  = float(px_now[tk] / px_long[tk] - 1)  if px_long[tk]  > 0 else np.nan
                ret_short = float(px_now[tk] / px_short[tk] - 1) if px_short[tk] > 0 else np.nan
                mom = ret_long - ret_short if not (np.isnan(ret_long) or np.isnan(ret_short)) else np.nan
                rows[tk]["momentum"] = mom
            else:
                rows[tk]["momentum"] = np.nan
    else:
        for tk in tickers:
            rows[tk]["momentum"] = np.nan

    # ── 4. Low volatility (inverse of 60-day realized vol)
    if len(rets) >= VOL_WINDOW:
        vol = stk_rets.tail(VOL_WINDOW).std() * np.sqrt(252)
        for tk in tickers:
            rows[tk]["low_vol"] = float(-vol.get(tk, np.nan))   # negative: low vol = positive factor
    else:
        for tk in tickers:
            rows[tk]["low_vol"] = np.nan

    # ── 5. Quality: prefer XBRL ROE/gross-margin; fallback to accruals
    xbrl_roe = xbrl["sig_quality_roe"] if (xbrl is not None and "sig_quality_roe" in xbrl.columns) else pd.Series(dtype=float)
    xbrl_gm  = xbrl["sig_quality_gm"]  if (xbrl is not None and "sig_quality_gm"  in xbrl.columns) else pd.Series(dtype=float)
    for tk in tickers:
        q_roe = float(xbrl_roe.get(tk, np.nan)) if not xbrl_roe.empty else np.nan
        q_gm  = float(xbrl_gm.get(tk, np.nan))  if not xbrl_gm.empty  else np.nan
        if not np.isnan(q_roe) and not np.isnan(q_gm):
            rows[tk]["quality"] = 0.5 * q_roe + 0.5 * q_gm
        elif not np.isnan(q_roe):
            rows[tk]["quality"] = q_roe
        else:
            rows[tk]["quality"] = float(-accruals.get(tk, np.nan))

    # ── 6. Analyst revisions
    for tk in tickers:
        rows[tk]["analyst_revision"] = float(revisions.get(tk, np.nan))

    # ── 7. Short interest (days-to-cover; high = potential squeeze)
    for tk in tickers:
        rows[tk]["short_interest"] = float(short_int.get(tk, np.nan))

    # ── 8. Value: prefer XBRL earnings yield (E/P); fallback to 1/beta
    xbrl_ep  = xbrl["sig_value_ep"]  if (xbrl is not None and "sig_value_ep"  in xbrl.columns) else pd.Series(dtype=float)
    xbrl_fcf = xbrl["sig_value_fcf"] if (xbrl is not None and "sig_value_fcf" in xbrl.columns) else pd.Series(dtype=float)
    for tk in tickers:
        ep  = float(xbrl_ep.get(tk, np.nan))  if not xbrl_ep.empty  else np.nan
        fcf = float(xbrl_fcf.get(tk, np.nan)) if not xbrl_fcf.empty else np.nan
        if not np.isnan(ep) and not np.isnan(fcf):
            rows[tk]["value"] = 0.6 * ep + 0.4 * fcf
        elif not np.isnan(ep):
            rows[tk]["value"] = ep
        else:
            beta = rows[tk].get("market_beta", 1.0)
            rows[tk]["value"] = 1.0 / max(abs(beta), 0.1)  # last-resort proxy

    # Build DataFrame
    df = pd.DataFrame(rows).T
    df.index.name = "ticker"

    # ── Sector dummies
    tk_to_sector = {tk: sector
                    for sector, tks in SECTOR_TICKERS.items()
                    for tk in tks}
    for etf, name in SECTORS.items():
        col = f"sector_{name}"
        df[col] = df.index.map(lambda t: 1.0 if tk_to_sector.get(t) == etf else 0.0)

    # ── Cross-sectional z-score for continuous factors
    style_factors = ["market_beta","size","momentum","low_vol",
                     "quality","analyst_revision","short_interest","value"]
    for col in style_factors:
        if col in df.columns:
            valid = pd.to_numeric(df[col], errors="coerce")
            df[col] = _zscore(valid.fillna(valid.median()))

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ── Factor returns + covariance ───────────────────────────────────────────────

def compute_factor_returns_and_cov(
    rets:         pd.DataFrame,
    expo:         pd.DataFrame,
    lookback:     int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate factor returns via cross-sectional OLS regression of stock returns
    on factor exposures (Barra method).
    Returns:
      factor_rets : DataFrame(index=date, columns=factors)
      factor_cov  : DataFrame(factors × factors), annualized covariance
    """
    if expo.empty or rets.empty:
        return pd.DataFrame(), pd.DataFrame()

    common_tickers = list(set(expo.index) & set(rets.columns))
    if len(common_tickers) < MIN_STOCKS:
        return pd.DataFrame(), pd.DataFrame()

    B    = expo.loc[common_tickers].values.astype(float)    # (n_stocks, n_factors)
    cols = expo.columns.tolist()
    n_f  = len(cols)

    rets_sub = rets[common_tickers].tail(lookback).dropna(how="all")
    if len(rets_sub) < 30:
        return pd.DataFrame(), pd.DataFrame()

    factor_rets_list = []
    for date, row in rets_sub.iterrows():
        r = row.values.astype(float)
        valid = ~np.isnan(r)
        if valid.sum() < MIN_STOCKS:
            continue
        B_v = B[valid]
        r_v = r[valid]
        try:
            # WLS: equal weight (simple OLS)
            BtB = B_v.T @ B_v
            Btr = B_v.T @ r_v
            # Ridge regularization to handle collinearity
            ridge = 1e-4 * np.eye(n_f)
            f_ret = np.linalg.solve(BtB + ridge, Btr)
            factor_rets_list.append(pd.Series(f_ret, index=cols, name=date))
        except np.linalg.LinAlgError:
            pass

    if not factor_rets_list:
        return pd.DataFrame(), pd.DataFrame()

    factor_rets = pd.DataFrame(factor_rets_list)
    factor_cov  = factor_rets.cov() * 252   # annualize

    return factor_rets, factor_cov


# ── Specific (idiosyncratic) risk ─────────────────────────────────────────────

def compute_specific_risk(
    rets:  pd.DataFrame,
    expo:  pd.DataFrame,
    f_rets: pd.DataFrame,
) -> pd.Series:
    """
    Specific variance = Var(r_i - B_i · f)
    where f is the factor return vector.
    Returns annualized specific vol per ticker.
    """
    if expo.empty or f_rets.empty:
        return pd.Series(dtype=float)

    common_tickers = list(set(expo.index) & set(rets.columns))
    common_dates   = list(set(rets.index) & set(f_rets.index))
    if not common_tickers or not common_dates:
        return pd.Series(dtype=float)

    B = expo.loc[common_tickers]
    r = rets.loc[common_dates, common_tickers].dropna(how="all")
    f = f_rets.loc[common_dates, B.columns].dropna(how="all")

    common_idx = r.index.intersection(f.index)
    r = r.loc[common_idx]
    f = f.loc[common_idx]

    specific_vols = {}
    for tk in common_tickers:
        if tk not in r.columns:
            continue
        r_tk  = r[tk].dropna()
        f_sub = f.loc[r_tk.index].values
        b_tk  = B.loc[tk].values
        resid = r_tk.values - f_sub @ b_tk
        specific_vols[tk] = float(np.std(resid) * np.sqrt(252))

    return pd.Series(specific_vols).rename("specific_vol")


# ── Portfolio risk decomposition ──────────────────────────────────────────────

def compute_portfolio_risk(
    weights:   pd.Series,
    expo:      pd.DataFrame,
    factor_cov:pd.DataFrame,
    spec_risk: pd.Series,
) -> dict:
    """
    Total variance = w' (B F B' + Δ) w
    factor_risk = w' B F B' w
    specific_risk = w' Δ w
    """
    tickers = list(set(weights.index) & set(expo.index))
    if not tickers:
        return {}

    w = weights.reindex(tickers).fillna(0.0).values
    B = expo.reindex(tickers)[factor_cov.columns].fillna(0.0).values
    F = factor_cov.values

    factor_var   = float(w @ B @ F @ B.T @ w)
    spec_var_vec = spec_risk.reindex(tickers).fillna(0.15).values ** 2  # annualized variance
    specific_var = float(w @ np.diag(spec_var_vec) @ w)

    total_var    = factor_var + specific_var
    total_vol    = float(np.sqrt(max(total_var, 0.0)))
    factor_vol   = float(np.sqrt(max(factor_var, 0.0)))
    specific_vol = float(np.sqrt(max(specific_var, 0.0)))

    # Factor contribution breakdown
    factor_contribs: dict[str, float] = {}
    for i, fname in enumerate(factor_cov.columns):
        b_f = B[:, i]
        fvar_contrib = float(w @ np.outer(b_f, b_f) @ w * F[i, i])
        factor_contribs[fname] = round(fvar_contrib / (total_var + 1e-12), 4)

    return {
        "total_annual_vol":    round(total_vol, 4),
        "factor_vol":          round(factor_vol, 4),
        "specific_vol":        round(specific_vol, 4),
        "factor_share":        round(factor_var / (total_var + 1e-12), 3),
        "specific_share":      round(specific_var / (total_var + 1e-12), 3),
        "factor_contributions": factor_contribs,
        "n_positions":         len(tickers),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(refresh: bool = False) -> dict:
    if not refresh and OUT_EXPO.exists():
        age = (datetime.now().timestamp() - OUT_EXPO.stat().st_mtime) / 86400
        if age < FRESHNESS_DAYS:
            print(f"  [FactorRisk] Exposures {age:.1f}d old — skipping. Use --refresh.")
            return {}

    all_tickers = sorted(set(tk for tks in SECTOR_TICKERS.values() for tk in tks))
    print(f"  [FactorRisk] Loading prices for {len(all_tickers)} tickers + SPY …")
    prices = load_prices(all_tickers, days=300)

    if prices.empty or "SPY" not in prices.columns:
        print("  [FactorRisk] No price data — skipping")
        return {}

    print(f"  [FactorRisk] Prices: {len(prices)} days × {len(prices.columns)} tickers")

    print("  [FactorRisk] Loading supplemental data …")
    mkt_caps  = load_market_caps()
    accruals  = load_accruals()
    revisions = load_analyst_revisions()
    short_int = load_short_interest()
    xbrl      = load_xbrl_fundamentals()
    if not xbrl.empty:
        print(f"  [FactorRisk] XBRL fundamentals: {len(xbrl)} tickers "
              f"({sum(1 for c in xbrl.columns if c.startswith('sig_'))} signals)")
    else:
        print("  [FactorRisk] No XBRL data — value/quality using fallback proxies")

    print("  [FactorRisk] Computing factor exposures …")
    expo = compute_factor_exposures(
        prices, mkt_caps, accruals, revisions, short_int, xbrl=xbrl if not xbrl.empty else None
    )
    if expo.empty:
        print("  [FactorRisk] Exposure matrix empty — skipping")
        return {}
    print(f"  [FactorRisk] Exposure matrix: {expo.shape}")

    rets = prices.pct_change(fill_method=None).dropna()
    print("  [FactorRisk] Computing factor returns via cross-sectional OLS …")
    factor_rets, factor_cov = compute_factor_returns_and_cov(rets, expo)
    if factor_cov.empty:
        print("  [FactorRisk] Factor covariance estimation failed — skipping")
        return {}
    print(f"  [FactorRisk] Factor cov matrix: {factor_cov.shape}")

    print("  [FactorRisk] Computing specific (idiosyncratic) risk …")
    spec_risk = compute_specific_risk(rets, expo, factor_rets)

    # ── Save outputs
    expo.reset_index().to_csv(OUT_EXPO, index=False)
    factor_cov.to_csv(OUT_FCOV)
    spec_risk.reset_index().rename(columns={"index":"ticker"}).to_csv(OUT_SRISK, index=False)
    print(f"  [FactorRisk] Saved: {OUT_EXPO.name}, {OUT_FCOV.name}, {OUT_SRISK.name}")

    # ── Portfolio risk decomposition (use paper_trading_log for current positions)
    result = {
        "expo": expo,
        "factor_cov": factor_cov,
        "spec_risk": spec_risk,
        "factor_rets": factor_rets,
    }

    _decompose_current_portfolio(expo, factor_cov, spec_risk)
    return result


def _decompose_current_portfolio(
    expo:       pd.DataFrame,
    factor_cov: pd.DataFrame,
    spec_risk:  pd.Series,
) -> None:
    """Decompose risk for the current paper portfolio."""
    log_path = ROOT / "paper_trading_log.csv"
    if not log_path.exists():
        return

    log = pd.read_csv(log_path)
    log["date"] = pd.to_datetime(log["date"], format="mixed", errors="coerce")
    latest = log[log["date"] == log["date"].max()]
    if latest.empty:
        return

    # Build weight vector: long = +1/N, short = -1/N
    long_col  = next((c for c in latest.columns if "long" in c.lower()), None)
    short_col = next((c for c in latest.columns if "short" in c.lower()), None)

    weights: dict[str, float] = {}
    if long_col and short_col:
        long_str  = str(latest.iloc[0].get(long_col, ""))
        short_str = str(latest.iloc[0].get(short_col, ""))
        longs  = [t.strip() for t in long_str.split(",") if t.strip()]
        shorts = [t.strip() for t in short_str.split(",") if t.strip()]
        n_l, n_s = len(longs), len(shorts)
        for t in longs:
            weights[t] = 1.0 / max(n_l, 1)
        for t in shorts:
            weights[t] = -1.0 / max(n_s, 1)

    if not weights:
        tickers_in_expo = expo.index.tolist()
        n = min(15, len(tickers_in_expo) // 2)
        for t in tickers_in_expo[:n]:
            weights[t] = 1.0 / n
        for t in tickers_in_expo[-n:]:
            weights[t] = -1.0 / n

    w_ser = pd.Series(weights)
    decomp = compute_portfolio_risk(w_ser, expo, factor_cov, spec_risk)
    decomp["date"] = datetime.now().strftime("%Y-%m-%d")
    pd.DataFrame([decomp]).to_csv(OUT_DECOMP, index=False)
    print(f"  [FactorRisk] Portfolio risk: vol={decomp.get('total_annual_vol',0):.2%}  "
          f"factor_share={decomp.get('factor_share',0):.1%}  "
          f"spec_share={decomp.get('specific_share',0):.1%}")


def write_report(expo: pd.DataFrame, factor_cov: pd.DataFrame,
                 spec_risk: pd.Series) -> None:
    if expo.empty:
        return

    style_factors = [c for c in expo.columns if not c.startswith("sector_")]

    # Average factor exposures for top/bottom quintile
    if spec_risk.empty:
        top_risk = expo.head(0)
    else:
        top_risk = spec_risk.nlargest(10)

    fac_rows = ""
    for f in style_factors:
        if f not in factor_cov.columns:
            continue
        ann_vol = float(np.sqrt(abs(factor_cov.loc[f, f])))
        avg_exp = expo[f].mean()
        fac_rows += f"| **{f}** | {avg_exp:+.3f} | {ann_vol:.2%} |\n"

    decomp_txt = ""
    if OUT_DECOMP.exists():
        d = pd.read_csv(OUT_DECOMP).iloc[-1]
        decomp_txt = (f"\n## Current Portfolio Risk\n\n"
                      f"| Metric | Value |\n|--------|------|\n"
                      f"| Total Annual Vol | {float(d.get('total_annual_vol',0)):.2%} |\n"
                      f"| Factor Vol | {float(d.get('factor_vol',0)):.2%} |\n"
                      f"| Specific Vol | {float(d.get('specific_vol',0)):.2%} |\n"
                      f"| Factor Share | {float(d.get('factor_share',0)):.1%} |\n"
                      f"| Specific Share | {float(d.get('specific_share',0)):.1%} |\n")

    report = f"""# Factor Risk Model Report — {datetime.now():%Y-%m-%d}

## Model Configuration

- Style factors: {len(style_factors)}
- Sector dummies: {len(SECTORS)}
- Total factors: {len(expo.columns)}
- Universe: {len(expo)} stocks
- Factor cov estimation: {LOOKBACK_DAYS}-day cross-sectional OLS

## Style Factor Summary

| Factor | Mean Exposure | Factor Annual Vol |
|--------|:-------------:|:-----------------:|
{fac_rows}
{decomp_txt}
## Top 10 Highest Idiosyncratic Risk Stocks

{top_risk.to_markdown() if not spec_risk.empty else 'N/A'}

## Interpretation

- **Factor share** > 60% = portfolio dominated by systematic risk
  (consider factor-neutralizing the book)
- **Specific share** > 60% = good: you are getting paid for stock picking
- A well-diversified L/S book should have ~40-55% factor share
- High beta + low specific risk = benchmark hugger (bad alpha signal)

---
*Built with free data. For production, replace size/value with Bloomberg fundamentals.*
"""
    OUT_REPORT.write_text(report)
    print(f"  [FactorRisk] Report → {OUT_REPORT.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor Risk Model")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Canyon v9 — Factor Risk Model  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60 + "\n")

    result = run(refresh=args.refresh)
    if result:
        write_report(result["expo"], result["factor_cov"], result["spec_risk"])

    print("\n" + "=" * 60)
    print("Step 88 Complete")
    print("=" * 60)
