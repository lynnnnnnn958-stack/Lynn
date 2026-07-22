"""
Factor Attribution & Alpha Regression
=======================================
Regresses portfolio returns against:
  - Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA)
  - Momentum factor (Mom)
Reports: Jensen's α, t-stat, factor loadings
Benchmark: SPY, RSP, stock-only, Full+TQQQ

This is the institutional gold standard for proving alpha.
α > 0 with t > 2 means returns NOT explained by known risk premia.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

def ols_alpha(y, X):
    """OLS regression with Newey-West HAC standard errors (4 lags)."""
    n, k = X.shape
    # Add intercept
    Xc = np.hstack([np.ones((n, 1)), X])
    # OLS
    beta  = np.linalg.lstsq(Xc, y, rcond=None)[0]
    resid = y - Xc @ beta
    # Newey-West HAC covariance (4 lags)
    S    = np.outer(resid, resid) * (Xc.T @ Xc) / n
    nw_cov = np.zeros((k+1, k+1))
    for lag in range(5):
        outer_ = Xc[lag:].T @ (resid[lag:, None] * resid[:n-lag, None] * Xc[:n-lag]) / n
        if lag == 0: nw_cov += outer_
        else:        nw_cov += (1 - lag/5) * (outer_ + outer_.T)
    xtx_inv = np.linalg.inv(Xc.T @ Xc / n)
    cov_hac = xtx_inv @ nw_cov @ xtx_inv / n
    se      = np.sqrt(np.diag(cov_hac))
    t_stats = beta / (se + 1e-12)
    r2      = 1 - np.var(resid) / np.var(y)
    return beta, t_stats, se, r2, resid

def run_regression(port_ret, bench_ret, ff, label="Portfolio", ann=12):
    """Full factor regression with multiple specifications."""
    # Align on common dates
    combined = pd.concat([port_ret.rename("port"), bench_ret.rename("bench"), ff], axis=1)
    combined = combined.dropna()
    if len(combined) < 24:
        print(f"  {label}: insufficient data ({len(combined)} obs)")
        return None

    port  = combined["port"].values
    bench = combined["bench"].values
    mktrf = combined["Mkt-RF"].values
    smb   = combined["SMB"].values
    hml   = combined["HML"].values
    rwm   = combined["RMW"].values if "RMW" in combined else np.zeros(len(port))
    cma   = combined["CMA"].values if "CMA" in combined else np.zeros(len(port))
    mom   = combined["Mom"].values if "Mom" in combined else np.zeros(len(port))
    rf    = combined["RF"].values

    excess_port  = port  - rf
    excess_bench = bench - rf

    print(f"\n  {'─'*60}")
    print(f"  {label}  (n={len(combined)}  {combined.index[0].date()} – "
          f"{combined.index[-1].date()})")
    print(f"  {'─'*60}")
    print(f"  {'Model':<18} {'α(ann%)':<10} {'t(α)':<7} {'β_mkt':<7} "
          f"{'β_smb':<7} {'β_hml':<7} {'β_mom':<7} {'R²'}")
    print(f"  {'─'*60}")

    results = {}
    # CAPM (1-factor)
    b, t, se, r2, _ = ols_alpha(excess_port, mktrf[:, None])
    alpha_ann = b[0] * ann
    print(f"  {'CAPM':<18} {alpha_ann*100:>+8.2f}%  {t[0]:>+6.2f}  "
          f"{b[1]:>+6.3f}  {'':>7}  {'':>7}  {'':>7}  {r2:.3f}")
    results["capm"] = (alpha_ann, t[0], b[1])

    # FF3 (3-factor)
    X3 = np.column_stack([mktrf, smb, hml])
    b, t, se, r2, _ = ols_alpha(excess_port, X3)
    alpha_ann = b[0] * ann
    print(f"  {'FF3':<18} {alpha_ann*100:>+8.2f}%  {t[0]:>+6.2f}  "
          f"{b[1]:>+6.3f}  {b[2]:>+6.3f}  {b[3]:>+6.3f}  {'':>7}  {r2:.3f}")
    results["ff3"] = (alpha_ann, t[0])

    # FF5 (5-factor)
    X5 = np.column_stack([mktrf, smb, hml, rwm, cma])
    b, t, se, r2, _ = ols_alpha(excess_port, X5)
    alpha_ann = b[0] * ann
    print(f"  {'FF5':<18} {alpha_ann*100:>+8.2f}%  {t[0]:>+6.2f}  "
          f"{b[1]:>+6.3f}  {b[2]:>+6.3f}  {b[3]:>+6.3f}  {'':>7}  {r2:.3f}")
    results["ff5"] = (alpha_ann, t[0])

    # FF6 (5 + momentum)
    X6 = np.column_stack([mktrf, smb, hml, rwm, cma, mom])
    b, t, se, r2, _ = ols_alpha(excess_port, X6)
    alpha_ann = b[0] * ann
    print(f"  {'FF6 (+Mom)':<18} {alpha_ann*100:>+8.2f}%  {t[0]:>+6.2f}  "
          f"{b[1]:>+6.3f}  {b[2]:>+6.3f}  {b[3]:>+6.3f}  {b[6]:>+6.3f}  {r2:.3f}")
    results["ff6"] = (alpha_ann, t[0], b[1:])
    factors_full = b[1:]

    # Information ratio vs benchmark
    active_ret = port - bench
    ir = active_ret.mean() / (active_ret.std() + 1e-9) * np.sqrt(ann)
    te = active_ret.std() * np.sqrt(ann)
    print(f"\n  vs benchmark:")
    print(f"    Tracking Error: {te*100:.1f}%/yr  "
          f"Information Ratio: {ir:+.3f}  "
          f"{'SIGNIFICANT' if abs(ir) > 0.4 else '(low)'}")

    if "ff6" in results:
        a, ta, fl = results["ff6"]
        significant = abs(ta) > 2 and a > 0
        print(f"\n  FF6 alpha: {a*100:+.2f}%/yr  t={ta:+.2f}  "
              f"{'*** SIGNIFICANT POSITIVE ALPHA ***' if significant else '(not significant)'}")
    return results

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading factor data...")

ff_monthly = None
ff_path = ROOT / "ff_factors_monthly.csv"
if ff_path.exists():
    ff_monthly = pd.read_csv(ff_path, index_col=0, parse_dates=True).sort_index()
    print(f"  FF monthly: {ff_monthly.shape}")
else:
    print("  ff_factors_monthly.csv not found — run download_fama_french.py first")

ff_daily = None
ffd_path = ROOT / "ff_factors_daily.csv"
if ffd_path.exists():
    ff_daily = pd.read_csv(ffd_path, index_col=0, parse_dates=True).sort_index()
    print(f"  FF daily:  {ff_daily.shape}")

# ── Load portfolio returns ────────────────────────────────────────────────────
p25   = pd.read_csv(ROOT / "sp500_price_25yr.csv", index_col=0, parse_dates=True).sort_index()
spy   = p25["SPY"]
HOLD_M = 21

# v21 OOS results
oos_t = None
oos_s = None
for fn in ["backtest_v21_oos.csv", "backtest_v20_oos.csv"]:
    if (ROOT / fn).exists():
        oos_t = pd.read_csv(ROOT / fn, parse_dates=["date"]).set_index("date")
        print(f"  Loaded: {fn}")
        break
for fn in ["backtest_v21_stock.csv", "backtest_v20_train.csv"]:
    if (ROOT / fn).exists():
        oos_s = pd.read_csv(ROOT / fn, parse_dates=["date"]).set_index("date")
        break

# Load RSP
rsp = None
if (ROOT / "sector_etf_prices.csv").exists():
    se  = pd.read_csv(ROOT / "sector_etf_prices.csv", index_col=0, parse_dates=True)
    rsp = se["RSP"] if "RSP" in se.columns else None

if ff_monthly is None:
    print("\nNo FF data available. Run download_fama_french.py first.")
    exit()

# ── Convert rebalancing returns to monthly ────────────────────────────────────
def rebal_to_monthly(df, price_series):
    """Convert rebalancing-period returns to calendar monthly."""
    if df is None: return None
    df2 = df.copy().reset_index()
    df2["year_month"] = pd.to_datetime(df2["date"]).dt.to_period("M")
    # Compound rebal returns within each month
    monthly = (df2.groupby("year_month")["ret"]
               .apply(lambda r: float(np.prod(1 + r.values) - 1))
               .reset_index())
    monthly["date"] = monthly["year_month"].dt.to_timestamp(how="end")
    monthly = monthly.set_index("date")["ret"]
    return monthly

port_monthly_t = rebal_to_monthly(oos_t, spy)
port_monthly_s = rebal_to_monthly(oos_s, spy)

# SPY and RSP monthly
spy_monthly = spy.resample("ME").last().pct_change().dropna()
rsp_monthly = rsp.resample("ME").last().pct_change().dropna() if rsp is not None else spy_monthly

# Align FF to 2018+
ff_oos = ff_monthly[ff_monthly.index >= "2018-01-01"]
spy_oos = spy_monthly[spy_monthly.index >= "2018-01-01"]

print("\n" + "═"*68)
print("  FAMA-FRENCH FACTOR ATTRIBUTION (OOS 2018-2026)")
print("═"*68)

if port_monthly_t is not None:
    run_regression(port_monthly_t, spy_oos, ff_oos,
                   label="v21 Full+TQQQ  vs SPY")

if port_monthly_s is not None:
    run_regression(port_monthly_s, spy_oos, ff_oos,
                   label="v21 Stock-only  vs SPY")
    if rsp is not None:
        run_regression(port_monthly_s, rsp_monthly[rsp_monthly.index>="2018-01-01"],
                       ff_oos, label="v21 Stock-only  vs RSP")

# ── FULL 2000-2026 analysis ───────────────────────────────────────────────────
train = None
for fn in ["backtest_v20_train.csv", "backtest_v21_stock.csv"]:
    if (ROOT / fn).exists():
        train = pd.read_csv(ROOT / fn, parse_dates=["date"]).set_index("date")
        break

if train is not None and port_monthly_t is not None:
    print("\n" + "═"*68)
    print("  FULL PERIOD 2000-2026 (TRAIN stock-only + TEST full)")
    print("═"*68)
    train_m = rebal_to_monthly(train, spy)
    combined_m = pd.concat([train_m, port_monthly_t]).sort_index()
    combined_m = combined_m[~combined_m.index.duplicated(keep="last")]
    ff_all = ff_monthly[ff_monthly.index >= "2000-01-01"]
    spy_all = spy_monthly[spy_monthly.index >= "2000-01-01"]
    run_regression(combined_m, spy_all, ff_all,
                   label="Combined 2000-2026  vs SPY")

# ── Factor exposure summary ───────────────────────────────────────────────────
print("\n" + "═"*68)
print("  FACTOR EXPOSURE INTERPRETATION")
print("═"*68)
print("""
  Factor Loadings Legend:
    β_mkt > 1.0  = levered market exposure (TQQQ effect)
    β_mkt ~ 0.5  = defensive, low market beta (quality/value tilt)
    β_smb > 0    = small-cap tilt (our 25-stock equal-weight)
    β_smb < 0    = large-cap tilt
    β_hml > 0    = value tilt
    β_hml < 0    = growth tilt (momentum tends to be growth-leaning)
    β_mom > 0    = momentum exposure (our strategy is momentum-based)
    β_mom < 0    = reversal / anti-momentum

  What a GOOD institutional result looks like:
    α (FF6) > +3%/yr with t > 2.0
    → Returns NOT explained by market/size/value/quality/momentum
    → Genuine portfolio construction skill

  What EXPLAINS poor annual beat rate vs SPY:
    If β_mkt < 1 and α > 0:
    → Strategy has LOWER market exposure than SPY
    → In strong bull markets (like 2018-2026) = underperforms
    → But delivers genuine alpha ON A RISK-ADJUSTED basis
    → SPY has β_mkt = 1 by definition; we shouldn't be judged on raw returns
""")
print("Done.")
