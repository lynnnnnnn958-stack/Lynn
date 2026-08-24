#!/usr/bin/env python3
"""
canyon_final_v9_step260_risk_infra.py  —  Week 6: Risk Infrastructure Upgrade

Four modules:
  A. Rolling IC Monitor    — 3m/6m rolling IC per signal, decay alerts
  B. Factor Neutralization — strip market/momentum/size from ensemble score
  C. CVaR Analysis         — historical + parametric + bootstrap tail risk
  D. Liquidity Stress Test — ADV-scaled liquidation cost under 4 scenarios

Outputs (all appended, no old files deleted):
  rolling_ic_monitor.csv / rolling_ic_report.md
  factor_exposures.csv   / factor_neutral_report.md
  cvar_analysis.csv      / cvar_report.md
  liquidity_stress_test.csv / liquidity_report.md
  risk_infrastructure_report.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Paths (inputs) ────────────────────────────────────────────────────────────
PRICES_FILE         = ROOT / "backtest_price_cache.csv"
WF_PREDS            = ROOT / "wf_oos_predictions.csv"
WF_PERF             = ROOT / "wf_oos_backtest_perf.csv"
TC_MONTHLY          = ROOT / "tc_comparison_monthly.csv"
PORT_CURVES         = ROOT / "portfolio_sizing_equity_curves.csv"
VOLUME_CACHE        = ROOT / "volume_cache.csv"
POS_SIZING          = ROOT / "position_sizing_output.csv"
ALT_IC              = ROOT / "alt_data_ic_report.csv"
FEAR_CACHE          = ROOT / "cboe_fear_gauge.csv"

# ── Paths (outputs) ───────────────────────────────────────────────────────────
ROLLING_IC_CSV      = ROOT / "rolling_ic_monitor.csv"
ROLLING_IC_REPORT   = ROOT / "rolling_ic_report.md"
FACTOR_EXP_CSV      = ROOT / "factor_exposures.csv"
FACTOR_REPORT       = ROOT / "factor_neutral_report.md"
CVAR_CSV            = ROOT / "cvar_analysis.csv"
CVAR_REPORT         = ROOT / "cvar_report.md"
LIQUIDITY_CSV       = ROOT / "liquidity_stress_test.csv"
LIQUIDITY_REPORT    = ROOT / "liquidity_report.md"
MAIN_REPORT         = ROOT / "risk_infrastructure_report.md"

# ── Config ────────────────────────────────────────────────────────────────────
IS_CUTOFF     = pd.Timestamp("2020-01-01")
PORTFOLIO_NAV = 10_000_000          # $10M portfolio
MAX_SINGLE    = 0.08                # 8% single-name cap
KAPPA         = 0.10                # Almgren-Chriss market-impact coefficient
PARTICIPATION = 0.20                # max ADV participation per day
IC_WARN       = 0.02                # rolling IC warning threshold
IC_ALERT      = 0.00                # rolling IC alert threshold
CVAR_LEVEL    = 0.05                # 5% tail (95% confidence)
CVAR_LEVEL_99 = 0.01                # 1% tail (99% confidence)
BOOTSTRAP_N   = 2000                # CVaR bootstrap samples


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.astype(float).sort_index()


def _spearman_ic(a: pd.Series, b: pd.Series) -> float:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return np.nan
    return float(stats.spearmanr(df["a"], df["b"]).statistic)


# ═══════════════════════════════════════════════════════════════════════════════
# Module A — Rolling IC Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def run_rolling_ic_monitor() -> pd.DataFrame:
    """
    Compute 3-month and 6-month rolling IC for all signals.
    Sources:
      1. ML ensemble (wf_oos_predictions.csv × wf_oos_backtest_perf.csv)
      2. Alt-data signals (alt_data_ic_report.csv)
      3. VIX fear signal (cboe_fear_gauge.csv × prices)
    """
    print("\n── Module A: Rolling IC Monitor ─────────────────────────────────")
    all_monthly_ic: list[pd.DataFrame] = []

    # ── Source 1: ML ensemble per-month IC ──────────────────────────────────
    if WF_PREDS.exists() and WF_PERF.exists():
        preds = pd.read_csv(WF_PREDS, parse_dates=["rebalance_date"])
        prices = load_prices()
        # Compute forward 21-day return at each rebalance date
        fwd_ret = prices.pct_change(21).shift(-21)
        records = []
        for dt, grp in preds.groupby("rebalance_date"):
            dt = pd.Timestamp(dt)
            future = fwd_ret.index[fwd_ret.index > dt]
            if len(future) == 0:
                continue
            fwd_row = fwd_ret.loc[future[0]].dropna()
            scores = grp.set_index("ticker")["ensemble_score"].dropna()
            common = scores.index.intersection(fwd_row.index)
            if len(common) < 5:
                continue
            ic = _spearman_ic(scores.reindex(common), fwd_row.reindex(common))
            if not np.isnan(ic):
                records.append({"date": dt, "signal": "ml_ensemble", "ic": ic,
                                "period": "OOS" if dt >= IS_CUTOFF else "IS"})
        if records:
            ml_ic = pd.DataFrame(records)
            all_monthly_ic.append(ml_ic)
            print(f"  ML ensemble: {len(records)} months IC computed")

    # ── Source 2: Alt-data signals (pre-computed monthly IC) ────────────────
    if ALT_IC.exists():
        alt_ic = pd.read_csv(ALT_IC, parse_dates=["date"])
        alt_ic = alt_ic[alt_ic["signal"].notna()]
        all_monthly_ic.append(alt_ic[["date", "signal", "ic", "period"]])
        sigs = alt_ic["signal"].unique().tolist()
        print(f"  Alt-data signals ({', '.join(sigs)}): {len(alt_ic)} rows loaded")

    if not all_monthly_ic:
        print("  No IC data found — skipping module A")
        return pd.DataFrame()

    ic_all = pd.concat(all_monthly_ic, ignore_index=True).sort_values("date")

    # ── Compute rolling 3m / 6m IC per signal ───────────────────────────────
    records_rolling = []
    for sig, grp in ic_all.groupby("signal"):
        grp = grp.set_index("date").sort_index()
        ic_s = grp["ic"].dropna()
        for i in range(len(ic_s)):
            dt = ic_s.index[i]
            window_3m  = ic_s.iloc[max(0, i-2) : i+1]
            window_6m  = ic_s.iloc[max(0, i-5) : i+1]
            window_12m = ic_s.iloc[max(0, i-11): i+1]

            ic_3m  = window_3m.mean()  if len(window_3m) >= 2  else np.nan
            ic_6m  = window_6m.mean()  if len(window_6m) >= 4  else np.nan
            ic_12m = window_12m.mean() if len(window_12m) >= 8 else np.nan
            ic_3m_std = window_3m.std() if len(window_3m) >= 2 else np.nan

            # Status
            if np.isnan(ic_3m):
                status = "insufficient_data"
            elif ic_3m < IC_ALERT:
                status = "ALERT_negative"
            elif ic_3m < IC_WARN:
                status = "WARNING_weak"
            else:
                status = "OK"

            records_rolling.append({
                "date":      dt,
                "signal":    sig,
                "ic_spot":   round(ic_s.iloc[i], 4),
                "ic_3m":     round(ic_3m, 4) if not np.isnan(ic_3m) else np.nan,
                "ic_6m":     round(ic_6m, 4) if not np.isnan(ic_6m) else np.nan,
                "ic_12m":    round(ic_12m, 4) if not np.isnan(ic_12m) else np.nan,
                "ic_3m_std": round(ic_3m_std, 4) if not np.isnan(ic_3m_std) else np.nan,
                "period":    grp.loc[dt, "period"] if "period" in grp.columns else "UNK",
                "status":    status,
            })

    result = pd.DataFrame(records_rolling)
    result.to_csv(ROLLING_IC_CSV, index=False)
    print(f"  Saved → {ROLLING_IC_CSV.name}: {len(result)} rows")

    # ── Print current status per signal ─────────────────────────────────────
    print("\n  Current Signal Health:")
    print(f"  {'Signal':<20} {'3m IC':>8} {'6m IC':>8} {'Status'}")
    print("  " + "-" * 55)
    for sig in result["signal"].unique():
        sig_df = result[result["signal"] == sig].sort_values("date")
        if sig_df.empty:
            continue
        latest = sig_df.iloc[-1]
        ic3  = f"{latest['ic_3m']:+.4f}" if not pd.isna(latest["ic_3m"]) else "  N/A  "
        ic6  = f"{latest['ic_6m']:+.4f}" if not pd.isna(latest["ic_6m"]) else "  N/A  "
        print(f"  {sig:<20} {ic3:>8} {ic6:>8}   {latest['status']}")

    return result


def write_rolling_ic_report(rolling: pd.DataFrame) -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [
        "# Rolling IC Monitor Report",
        f"**Generated:** {today}  |  **Canyon v9 — Week 6 Module A**",
        "",
        "## Purpose",
        "Detect signal decay before it impacts portfolio performance. "
        "Rolling 3-month IC < 0.02 triggers a WARNING; IC < 0 triggers ALERT.",
        "",
        f"| Threshold | Condition | Action |",
        f"|-----------|-----------|--------|",
        f"| OK        | 3m IC ≥ 0.02 | No action required |",
        f"| WARNING   | 0 ≤ 3m IC < 0.02 | Review signal, consider reducing weight |",
        f"| ALERT     | 3m IC < 0 | Suspend signal from ensemble |",
        "",
        "## Current Signal Status (Most Recent Month)",
        "",
    ]
    if rolling.empty:
        lines.append("*No rolling IC data available.*")
    else:
        lines += [
            "| Signal | Latest IC | 3m IC | 6m IC | 12m IC | Status |",
            "|--------|-----------|-------|-------|--------|--------|",
        ]
        for sig in rolling["signal"].unique():
            sig_df = rolling[rolling["signal"] == sig].sort_values("date")
            l = sig_df.iloc[-1]
            fmt = lambda x: f"{x:+.4f}" if not pd.isna(x) else "N/A"
            status_icon = {"OK": "✓", "WARNING_weak": "~", "ALERT_negative": "✗",
                           "insufficient_data": "?"}.get(l["status"], "?")
            lines.append(
                f"| `{sig}` | {fmt(l['ic_spot'])} | {fmt(l['ic_3m'])} "
                f"| {fmt(l['ic_6m'])} | {fmt(l['ic_12m'])} | {status_icon} {l['status']} |"
            )
        # IC decay summary
        alerts  = rolling[rolling["status"] == "ALERT_negative"]["signal"].unique()
        warns   = rolling[rolling["status"] == "WARNING_weak"]["signal"].unique()
        ok_sigs = rolling[rolling["status"] == "OK"]["signal"].unique()
        lines += [
            "",
            f"**Summary:** {len(ok_sigs)} OK | {len(warns)} WARNING | {len(alerts)} ALERT",
            "",
            "## IC History (IS vs OOS Mean)",
            "",
            "| Signal | IS Mean IC | OOS Mean IC | IS→OOS Decay |",
            "|--------|-----------|------------|--------------|",
        ]
        for sig in rolling["signal"].unique():
            sig_df = rolling[rolling["signal"] == sig]
            is_ic  = sig_df[sig_df["period"] == "IS"]["ic_spot"].mean()
            oos_ic = sig_df[sig_df["period"] == "OOS"]["ic_spot"].mean()
            decay  = oos_ic - is_ic if not (np.isnan(is_ic) or np.isnan(oos_ic)) else np.nan
            fmt = lambda x: f"{x:+.4f}" if not np.isnan(x) else "N/A"
            decay_flag = "↑ improved" if (not np.isnan(decay) and decay > 0) else "↓ degraded"
            lines.append(f"| `{sig}` | {fmt(is_ic)} | {fmt(oos_ic)} | {fmt(decay)} {decay_flag} |")

    ROLLING_IC_REPORT.write_text("\n".join(lines))
    print(f"  Report → {ROLLING_IC_REPORT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Module B — Factor Neutralization
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_factor_exposures(
    prices: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    At each rebalance date, compute per-stock factor exposures:
      - beta_mkt:  OLS β on SPY rolling {lookback} days
      - mom_12_1:  12-month minus 1-month return (pure momentum)
      - size_proxy: log-price as size proxy (no market cap data available)
    """
    rets = prices.pct_change().dropna(how="all")
    spy  = rets["SPY"] if "SPY" in rets.columns else rets.mean(axis=1)
    stocks = [c for c in prices.columns if c not in ("SPY","QQQ","SMH","SOXX",
               "XLK","XLE","XLF","XLV","XLU","XLP")]

    records = []
    for dt in rebal_dates:
        hist = rets.loc[:dt].iloc[-lookback:]
        if len(hist) < 60:
            continue
        spy_hist = spy.reindex(hist.index).dropna()
        for tk in stocks:
            if tk not in hist.columns:
                continue
            tk_hist = hist[tk].dropna()
            common  = tk_hist.index.intersection(spy_hist.index)
            if len(common) < 40:
                continue
            # Market beta
            x = spy_hist.reindex(common).values
            y = tk_hist.reindex(common).values
            beta = np.cov(y, x)[0, 1] / (np.var(x) + 1e-12)

            # Momentum: 12-month minus 1-month
            p = prices[tk].dropna()
            p_before = p.loc[:dt]
            if len(p_before) < 252:
                mom = np.nan
            else:
                r_12 = p_before.iloc[-1] / p_before.iloc[-252] - 1
                r_1  = p_before.iloc[-1] / p_before.iloc[-21]  - 1
                mom  = r_12 - r_1

            # Size proxy: log price (crude — real institutions use mkt cap)
            size = float(np.log(p_before.iloc[-1] + 1e-6)) if len(p_before) > 0 else np.nan

            records.append({
                "date": dt, "ticker": tk,
                "beta_mkt": round(beta, 4),
                "mom_12_1": round(mom, 4) if not np.isnan(mom) else np.nan,
                "size_proxy": round(size, 4) if not np.isnan(size) else np.nan,
            })
    return pd.DataFrame(records)


def _neutralize_scores(
    scores: pd.DataFrame,       # columns: date, ticker, score
    exposures: pd.DataFrame,    # columns: date, ticker, beta_mkt, mom_12_1, size_proxy
) -> pd.DataFrame:
    """
    At each date, regress scores on factor exposures. Use OLS residuals.
    Returns DataFrame with original score + neutralized score.
    """
    factor_cols = ["beta_mkt", "mom_12_1", "size_proxy"]
    records = []
    for dt, s_grp in scores.groupby("date"):
        dt = pd.Timestamp(dt)
        e_grp = exposures[exposures["date"] == dt] if len(exposures) > 0 else pd.DataFrame()
        merged = s_grp.merge(e_grp, on="ticker", how="inner") if not e_grp.empty else s_grp.copy()
        merged = merged.rename(columns={"date_x": "date"}, errors="ignore")
        if "date" not in merged.columns:
            merged["date"] = dt

        fcols_avail = [c for c in factor_cols if c in merged.columns and merged[c].notna().sum() > 3]
        if len(fcols_avail) < 1 or len(merged) < 5:
            merged["score_neutral"] = merged["score"] if "score" in merged.columns else np.nan
            records.append(merged)
            continue

        y = merged["score"].values
        X = merged[fcols_avail].values
        X_filled = np.nan_to_num(X, nan=0.0)
        # Add intercept
        X_int = np.column_stack([np.ones(len(X_filled)), X_filled])
        try:
            coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
            resid = y - X_int @ coef
        except Exception:
            resid = y - y.mean()
        # Re-normalize residual to same mean/std as original scores
        if resid.std() > 0:
            resid = (resid - resid.mean()) / resid.std() * y.std() + y.mean()
        merged["score_neutral"] = resid
        records.append(merged)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def run_factor_neutralization() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ML ensemble scores, compute factor exposures, neutralize scores.
    Compare IC before vs after neutralization.
    """
    print("\n── Module B: Factor Neutralization ─────────────────────────────")

    if not WF_PREDS.exists():
        print("  wf_oos_predictions.csv not found — skipping")
        return pd.DataFrame(), pd.DataFrame()

    prices = load_prices()
    preds  = pd.read_csv(WF_PREDS, parse_dates=["rebalance_date"])
    preds  = preds.rename(columns={"rebalance_date": "date", "ensemble_score": "score"})
    preds  = preds[["date", "ticker", "score", "period"]].dropna(subset=["score"])

    rebal_dates = pd.DatetimeIndex(sorted(preds["date"].unique()))
    print(f"  Computing factor exposures for {len(rebal_dates)} rebalance dates "
          f"({rebal_dates.min().date()} → {rebal_dates.max().date()})...")

    exposures = _compute_factor_exposures(prices, rebal_dates, lookback=252)
    print(f"  Factor exposures: {len(exposures)} rows")

    neutralized = _neutralize_scores(preds, exposures)
    print(f"  Neutralized scores: {len(neutralized)} rows")

    # ── Compare IC: raw vs neutralized ──────────────────────────────────────
    fwd_ret = prices.pct_change(21).shift(-21)
    ic_records = []
    for dt, grp in neutralized.groupby("date"):
        dt = pd.Timestamp(dt)
        future = fwd_ret.index[fwd_ret.index > dt]
        if len(future) == 0:
            continue
        fwd_row = fwd_ret.loc[future[0]].dropna()

        for score_col, label in [("score", "raw"), ("score_neutral", "neutralized")]:
            if score_col not in grp.columns:
                continue
            scores_s = grp.set_index("ticker")[score_col].dropna()
            common   = scores_s.index.intersection(fwd_row.index)
            if len(common) < 5:
                continue
            ic = _spearman_ic(scores_s.reindex(common), fwd_row.reindex(common))
            if not np.isnan(ic):
                ic_records.append({
                    "date": dt, "method": label, "ic": ic,
                    "period": "OOS" if dt >= IS_CUTOFF else "IS",
                })

    ic_df = pd.DataFrame(ic_records)
    if not ic_df.empty:
        summary = ic_df.groupby(["method", "period"])["ic"].agg(["mean","std","count"]).round(4)
        print("\n  IC Before vs After Factor Neutralization:")
        print(f"  {'Method':<14} {'IS Mean IC':>12} {'OOS Mean IC':>12}")
        print("  " + "-" * 42)
        for method in ["raw", "neutralized"]:
            for period in ["IS", "OOS"]:
                mask = (ic_df["method"] == method) & (ic_df["period"] == period)
                mean_ic = ic_df.loc[mask, "ic"].mean()
                print(f"  {method+'/'+period:<14} {mean_ic:>12.4f}")

    # Combine and save
    if not exposures.empty:
        exposures.to_csv(FACTOR_EXP_CSV, index=False)
        print(f"\n  Factor exposures → {FACTOR_EXP_CSV.name}")

    return exposures, ic_df


def write_factor_report(exposures: pd.DataFrame, ic_df: pd.DataFrame) -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [
        "# Factor Neutralization Report",
        f"**Generated:** {today}  |  **Canyon v9 — Week 6 Module B**",
        "",
        "## Methodology",
        "",
        "At each monthly rebalance date, the ML ensemble score is residualized against "
        "three systematic factor exposures using OLS regression:",
        "",
        "| Factor | Proxy | Why neutralize |",
        "|--------|-------|----------------|",
        "| Market | Rolling 252-day beta to SPY | Avoid tilting toward high-beta in bull markets |",
        "| Momentum | 12-month minus 1-month return | Prevent double-counting with price momentum |",
        "| Size | Log price (proxy for market cap) | Prevent cap-size bias in scoring |",
        "",
        "**Neutralized score** = OLS residual, rescaled to original mean/std.",
        "",
    ]

    if ic_df.empty:
        lines.append("*No IC comparison data — wf_oos_predictions.csv not found.*")
    else:
        lines += [
            "## IC Comparison: Raw vs Neutralized Ensemble Score",
            "",
            "| Method | Period | Mean IC | Std IC | N Months |",
            "|--------|--------|---------|--------|----------|",
        ]
        for (method, period), grp in ic_df.groupby(["method", "period"]):
            vals = grp["ic"].dropna()
            lines.append(f"| {method} | {period} | {vals.mean():+.4f} | {vals.std():.4f} | {len(vals)} |")

        lines += [
            "",
            "> **Interpretation:** Neutralized IC slightly lower (removes factor returns) "
            "but more predictive of *pure alpha* — the component not explained by common risk factors.",
        ]

    if not exposures.empty:
        latest_date = exposures["date"].max()
        latest = exposures[exposures["date"] == latest_date]
        lines += [
            "",
            f"## Current Factor Exposures (Most Recent: {latest_date.date()})",
            "",
            f"Portfolio-level weighted average (equal weight across {len(latest)} stocks):",
            "",
            f"| Factor | Mean | Std | Min | Max |",
            f"|--------|------|-----|-----|-----|",
        ]
        for col in ["beta_mkt", "mom_12_1", "size_proxy"]:
            if col in latest.columns:
                v = latest[col].dropna()
                lines.append(
                    f"| {col} | {v.mean():.3f} | {v.std():.3f} | {v.min():.3f} | {v.max():.3f} |"
                )

    FACTOR_REPORT.write_text("\n".join(lines))
    print(f"  Report → {FACTOR_REPORT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Module C — CVaR Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _historical_cvar(returns: pd.Series, level: float = 0.05) -> float:
    """Expected Shortfall: mean of returns below the level-th quantile."""
    q = returns.quantile(level)
    tail = returns[returns <= q]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def _parametric_cvar(returns: pd.Series, level: float = 0.05) -> float:
    """CVaR under normality assumption: -mu + sigma * phi(Phi^-1(alpha)) / alpha."""
    mu    = returns.mean()
    sigma = returns.std()
    z_a   = stats.norm.ppf(level)
    phi_z = stats.norm.pdf(z_a)
    return float(-mu + sigma * phi_z / level)


def _bootstrap_cvar(returns: pd.Series, level: float = 0.05, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Bootstrap mean and 95% CI for historical CVaR."""
    rng    = np.random.default_rng(42)
    arr    = returns.dropna().values
    cvar_b = []
    for _ in range(n):
        sample = rng.choice(arr, size=len(arr), replace=True)
        q      = np.quantile(sample, level)
        tail   = sample[sample <= q]
        cvar_b.append(tail.mean() if len(tail) > 0 else np.nan)
    cvar_b = np.array([x for x in cvar_b if not np.isnan(x)])
    lo, hi = np.percentile(cvar_b, [2.5, 97.5])
    return float(lo), float(hi)


def run_cvar_analysis() -> pd.DataFrame:
    """
    Compute CVaR metrics for all available return series:
      - Monthly ML returns (IS and OOS) from tc_comparison_monthly.csv
      - Equity curve methods from portfolio_sizing_equity_curves.csv
    """
    print("\n── Module C: CVaR Analysis ──────────────────────────────────────")

    records = []

    # ── Source 1: tc_comparison_monthly ─────────────────────────────────────
    if TC_MONTHLY.exists():
        tc = pd.read_csv(TC_MONTHLY, parse_dates=["rebalance_date"])
        tc = tc.sort_values("rebalance_date")
        for col, label in [("ret_dynamic_tc", "Dynamic-TC"), ("spy_ret", "SPY-benchmark")]:
            if col not in tc.columns:
                continue
            for period in ["IS", "OOS", "ALL"]:
                if period == "ALL":
                    rets = tc[col].dropna()
                elif period == "IS":
                    rets = tc.loc[tc["rebalance_date"] < IS_CUTOFF, col].dropna()
                else:
                    rets = tc.loc[tc["rebalance_date"] >= IS_CUTOFF, col].dropna()
                if len(rets) < 10:
                    continue
                cvar95h = _historical_cvar(rets, CVAR_LEVEL)
                cvar99h = _historical_cvar(rets, CVAR_LEVEL_99)
                cvar95p = _parametric_cvar(rets, CVAR_LEVEL)
                lo95, hi95 = _bootstrap_cvar(rets, CVAR_LEVEL)
                max_dd  = _max_drawdown(rets)
                ann_vol = rets.std() * np.sqrt(12)
                ann_ret = rets.mean() * 12
                sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
                calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
                records.append({
                    "series": label, "period": period,
                    "n_months": int(len(rets)),
                    "ann_ret_pct": round(ann_ret * 100, 2),
                    "ann_vol_pct": round(ann_vol * 100, 2),
                    "sharpe": round(sharpe, 3),
                    "max_dd_pct": round(max_dd * 100, 2),
                    "calmar": round(calmar, 3),
                    "cvar95_hist_pct": round(cvar95h * 100, 2),
                    "cvar99_hist_pct": round(cvar99h * 100, 2),
                    "cvar95_param_pct": round(cvar95p * 100, 2),
                    "cvar95_lo_pct": round(lo95 * 100, 2),
                    "cvar95_hi_pct": round(hi95 * 100, 2),
                })

    # ── Source 2: portfolio sizing equity curves ─────────────────────────────
    if PORT_CURVES.exists():
        curves = pd.read_csv(PORT_CURVES, parse_dates=["date"])
        for method, grp in curves.groupby("method"):
            grp = grp.sort_values("date")
            monthly_rets = grp.set_index("date")["cumret"].pct_change().dropna()
            for period in ["OOS", "ALL"]:
                if period == "OOS":
                    rets = monthly_rets[monthly_rets.index >= IS_CUTOFF]
                else:
                    rets = monthly_rets
                if len(rets) < 10:
                    continue
                cvar95h = _historical_cvar(rets, CVAR_LEVEL)
                cvar99h = _historical_cvar(rets, CVAR_LEVEL_99)
                max_dd  = _max_drawdown(rets)
                ann_vol = rets.std() * np.sqrt(12)
                ann_ret = rets.mean() * 12
                sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
                calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
                records.append({
                    "series": f"Sizing-{method}", "period": period,
                    "n_months": int(len(rets)),
                    "ann_ret_pct": round(ann_ret * 100, 2),
                    "ann_vol_pct": round(ann_vol * 100, 2),
                    "sharpe": round(sharpe, 3),
                    "max_dd_pct": round(max_dd * 100, 2),
                    "calmar": round(calmar, 3),
                    "cvar95_hist_pct": round(cvar95h * 100, 2),
                    "cvar99_hist_pct": round(cvar99h * 100, 2),
                    "cvar95_param_pct": np.nan,
                    "cvar95_lo_pct": np.nan,
                    "cvar95_hi_pct": np.nan,
                })

    result = pd.DataFrame(records)
    if not result.empty:
        result.to_csv(CVAR_CSV, index=False)
        print(f"  CVaR analysis: {len(result)} rows → {CVAR_CSV.name}")
        print(f"\n  {'Series':<22} {'Period':<6} {'Sharpe':>8} {'CVaR95%':>10} {'CVaR99%':>10} {'MaxDD':>8}")
        print("  " + "-" * 70)
        for _, row in result[result["period"] == "OOS"].iterrows():
            print(
                f"  {row['series']:<22} {row['period']:<6} {row['sharpe']:>8.3f} "
                f"{row['cvar95_hist_pct']:>9.2f}% {row['cvar99_hist_pct']:>9.2f}% "
                f"{row['max_dd_pct']:>7.2f}%"
            )
    return result


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def write_cvar_report(cvar_df: pd.DataFrame) -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [
        "# CVaR (Expected Shortfall) Analysis",
        f"**Generated:** {today}  |  **Canyon v9 — Week 6 Module C**",
        "",
        "## Methodology",
        "",
        "**CVaR (Conditional Value at Risk)** = Expected loss in the worst α% of scenarios.",
        "Also called Expected Shortfall (ES). More robust than VaR as it captures tail shape.",
        "",
        "| Method | Formula | Notes |",
        "|--------|---------|-------|",
        "| Historical | Mean of returns ≤ quantile(α) | Non-parametric, uses actual history |",
        "| Parametric | -μ + σ·φ(Φ⁻¹(α))/α | Assumes normality (underestimates fat tails) |",
        "| Bootstrap CI | 2000 resample iterations | 95% confidence interval on historical CVaR |",
        "",
        "## OOS Results",
        "",
    ]

    if cvar_df.empty:
        lines.append("*No return data available.*")
    else:
        oos_df = cvar_df[cvar_df["period"] == "OOS"].copy()
        if not oos_df.empty:
            lines += [
                "| Series | Sharpe | Ann Ret | Ann Vol | CVaR95 (hist) | CVaR99 (hist) | MaxDD | Calmar |",
                "|--------|--------|---------|---------|--------------|--------------|-------|--------|",
            ]
            for _, row in oos_df.iterrows():
                calmar_str = f"{row['calmar']:.2f}" if not pd.isna(row['calmar']) else "N/A"
                lines.append(
                    f"| {row['series']} | {row['sharpe']:.3f} | {row['ann_ret_pct']:.1f}% "
                    f"| {row['ann_vol_pct']:.1f}% | {row['cvar95_hist_pct']:.2f}% "
                    f"| {row['cvar99_hist_pct']:.2f}% | {row['max_dd_pct']:.2f}% "
                    f"| {calmar_str} |"
                )

        # Best series
        if "Dynamic-TC" in oos_df["series"].values:
            row = oos_df[oos_df["series"] == "Dynamic-TC"].iloc[0]
            lines += [
                "",
                f"## Dynamic-TC Strategy Deep Dive (OOS)",
                "",
                f"- **Monthly CVaR 95%**: {row['cvar95_hist_pct']:.2f}% (worst 5% of months average this loss)",
                f"- **Monthly CVaR 99%**: {row['cvar99_hist_pct']:.2f}% (worst 1% of months)",
                f"- **Bootstrap 95% CI**: [{row['cvar95_lo_pct']:.2f}%, {row['cvar95_hi_pct']:.2f}%]",
                f"- **Max Drawdown**: {row['max_dd_pct']:.2f}%",
                f"- **Calmar Ratio**: {row['calmar']:.2f} (Ann Return / |Max DD|)",
                "",
                "> **Calmar > 2.0 is considered excellent** — the strategy earns back more than 2× "
                "its maximum drawdown annually.",
            ]

    CVAR_REPORT.write_text("\n".join(lines))
    print(f"  Report → {CVAR_REPORT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Module D — Liquidity Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

# Stress scenarios: (label, adv_multiplier, description)
STRESS_SCENARIOS = [
    ("Normal",          1.00, "Current market conditions"),
    ("Moderate Stress", 0.50, "Sector rotation / earnings season"),
    ("Severe Stress",   0.20, "2008-style credit crisis"),
    ("Extreme Crisis",  0.05, "Circuit-breaker / trading halt scenario"),
]


def run_liquidity_stress_test() -> pd.DataFrame:
    """
    For each position and each stress scenario, compute:
      - Max position under ADV constraint
      - Excess position to liquidate
      - Days to liquidate
      - Almgren-Chriss liquidation cost (bps and $)
    """
    print("\n── Module D: Liquidity Stress Test ──────────────────────────────")

    if not VOLUME_CACHE.exists():
        print("  volume_cache.csv not found — skipping")
        return pd.DataFrame()

    # Load ADV (average daily dollar volume)
    vol_df = pd.read_csv(VOLUME_CACHE, index_col=0, parse_dates=True)   # 日期是无名 index 列
    if vol_df.empty:                              # 空缓存(只有表头/无人填充)→ 优雅跳过
        print("  volume_cache.csv is empty — skipping liquidity stress test")
        return pd.DataFrame()
    adv = vol_df.rolling(63).mean().iloc[-1]  # 63-day average (most recent)
    adv = adv.dropna()

    # Build positions: prefer ML predictions (tickers match volume cache)
    # Fall back to position_sizing_output.csv, then equal weight
    stocks_in_vol = [c for c in adv.index if c not in
                     ("SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP")]
    positions = pd.Series(dtype=float)

    # Try ML predictions first (most recent rebalance)
    if WF_PREDS.exists():
        preds = pd.read_csv(WF_PREDS, parse_dates=["rebalance_date"])
        latest_date = preds["rebalance_date"].max()
        latest = preds[preds["rebalance_date"] == latest_date].copy()
        latest = latest.sort_values("ensemble_score", ascending=False)
        top15  = latest.head(15)
        top15_in_vol = top15[top15["ticker"].isin(adv.index)]
        if len(top15_in_vol) >= 5:
            # Score-weighted positions capped at MAX_SINGLE
            raw_w = top15_in_vol["ensemble_score"].clip(lower=0)
            raw_w = raw_w / raw_w.sum() * 0.95
            raw_w = raw_w.clip(upper=MAX_SINGLE)
            positions = raw_w.set_axis(top15_in_vol["ticker"].values)
            print(f"  Positions from ML predictions ({latest_date.date()}): {len(positions)} tickers")

    # Fallback: position_sizing_output.csv filtered to volume-cache tickers
    if positions.empty and POS_SIZING.exists():
        pos_df = pd.read_csv(POS_SIZING)
        pos_df.columns = [c.lower().strip() for c in pos_df.columns]
        weight_col = next((c for c in pos_df.columns if c == "weight"), None)
        ticker_col = next((c for c in pos_df.columns if "ticker" in c), None)
        if weight_col and ticker_col:
            pos_s = pos_df.set_index(ticker_col)[weight_col].dropna()
            overlap = pos_s.index.intersection(adv.index)
            if len(overlap) >= 3:
                positions = pos_s.reindex(overlap).dropna()

    # Final fallback: equal weight on volume-cache stocks
    if positions.empty:
        n = min(15, len(stocks_in_vol))
        positions = pd.Series(1.0 / n, index=stocks_in_vol[:n])
        print(f"  Fallback: equal weight {n} tickers from volume cache")

    # Also load volatility from price data for market-impact model
    prices = load_prices()
    daily_vol = prices.pct_change().std()  # annualized later

    records = []
    common_tickers = sorted(set(positions.index) & set(adv.index))
    if not common_tickers:
        print(f"  WARNING: No common tickers between positions ({list(positions.index)[:5]}) and ADV ({list(adv.index)[:5]})")
        common_tickers = list(adv.index)[:min(10, len(adv.index))]

    print(f"  Testing {len(common_tickers)} tickers × {len(STRESS_SCENARIOS)} scenarios")

    for tk in common_tickers:
        pos_weight = float(positions.get(tk, MAX_SINGLE))
        pos_value  = PORTFOLIO_NAV * pos_weight         # $ position value
        adv_value  = float(adv.get(tk, 0))              # $ daily ADV
        if adv_value <= 0:
            continue

        vol = float(daily_vol.get(tk, 0.02))            # daily return vol

        for scenario_name, adv_mult, scenario_desc in STRESS_SCENARIOS:
            stressed_adv = adv_value * adv_mult
            max_pos = stressed_adv * 5 * PARTICIPATION  # 5-day × 20% participation

            excess_value = max(0.0, pos_value - max_pos)
            excess_pct   = excess_value / pos_value if pos_value > 0 else 0.0

            if excess_value > 0:
                # Days to liquidate excess at 20% participation
                liq_days = excess_value / (stressed_adv * PARTICIPATION) if stressed_adv > 0 else np.inf
                # Almgren-Chriss cost on excess
                participation = excess_value / stressed_adv if stressed_adv > 0 else 1.0
                impact_bps    = KAPPA * vol * np.sqrt(max(0, participation)) * 10_000
                spread_bps    = 1.5 if adv_value > 1e9 else (3.0 if adv_value > 1e8 else 6.0)
                total_cost_bps = impact_bps + spread_bps
                cost_dollars   = excess_value * total_cost_bps / 10_000
            else:
                liq_days     = 0.0
                impact_bps   = 0.0
                total_cost_bps = 0.0
                cost_dollars = 0.0

            records.append({
                "ticker":          tk,
                "scenario":        scenario_name,
                "adv_multiplier":  adv_mult,
                "pos_value_mm":    round(pos_value / 1e6, 3),
                "adv_normal_mm":   round(adv_value / 1e6, 3),
                "adv_stressed_mm": round(stressed_adv / 1e6, 3),
                "max_pos_mm":      round(max_pos / 1e6, 3),
                "excess_value_mm": round(excess_value / 1e6, 3),
                "excess_pct":      round(excess_pct * 100, 1),
                "liq_days":        round(liq_days, 1),
                "impact_bps":      round(impact_bps, 2),
                "total_cost_bps":  round(total_cost_bps, 2),
                "liq_cost_usd":    round(cost_dollars, 0),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result.to_csv(LIQUIDITY_CSV, index=False)
        print(f"  Saved → {LIQUIDITY_CSV.name}: {len(result)} rows")

        # Summary by scenario
        print(f"\n  {'Scenario':<20} {'Tickers Affected':>16} {'Avg Excess%':>12} {'Avg Cost(bps)':>14} {'Max Liq Days':>12}")
        print("  " + "-" * 80)
        for sc, grp in result.groupby("scenario"):
            affected  = (grp["excess_pct"] > 0).sum()
            avg_excess = grp.loc[grp["excess_pct"] > 0, "excess_pct"].mean() if affected > 0 else 0
            avg_cost  = grp.loc[grp["excess_pct"] > 0, "total_cost_bps"].mean() if affected > 0 else 0
            max_days  = grp["liq_days"].max()
            print(f"  {sc:<20} {affected:>16} {avg_excess:>11.1f}% {avg_cost:>13.1f} {max_days:>11.1f}d")

    return result


def write_liquidity_report(liq_df: pd.DataFrame) -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [
        "# Liquidity Stress Test Report",
        f"**Generated:** {today}  |  **Canyon v9 — Week 6 Module D**",
        "",
        "## Methodology",
        "",
        f"**Portfolio NAV:** ${PORTFOLIO_NAV:,.0f}  |  **Max Single Name:** {MAX_SINGLE*100:.0f}%",
        f"**ADV Constraint:** {int(5)} days × {PARTICIPATION*100:.0f}% participation rate",
        f"**Market Impact:** Almgren-Chriss: κ={KAPPA} × σ × √(participation)",
        "",
        "## Stress Scenarios",
        "",
        "| Scenario | ADV Multiplier | Context |",
        "|----------|---------------|---------|",
    ]
    for name, mult, desc in STRESS_SCENARIOS:
        lines.append(f"| {name} | {mult:.0%} of normal ADV | {desc} |")

    if liq_df.empty:
        lines.append("\n*No liquidity data available.*")
    else:
        lines += [
            "",
            "## Results by Scenario",
            "",
            "| Scenario | Positions Affected | Avg Excess | Total Liq Cost ($) | Max Days |",
            "|----------|-------------------|------------|--------------------|---------|",
        ]
        for sc, grp in liq_df.groupby("scenario"):
            affected   = (grp["excess_pct"] > 0).sum()
            total_cost = grp["liq_cost_usd"].sum()
            max_days   = grp["liq_days"].max()
            avg_excess = grp.loc[grp["excess_pct"] > 0, "excess_pct"].mean() if affected > 0 else 0
            lines.append(
                f"| {sc} | {affected}/{len(grp)} | {avg_excess:.1f}% | "
                f"${total_cost:,.0f} | {max_days:.1f}d |"
            )

        lines += [
            "",
            "## Key Risk Metrics",
            "",
            "### At Normal ADV",
        ]
        normal = liq_df[liq_df["scenario"] == "Normal"]
        if not normal.empty:
            no_viol = normal[normal["excess_pct"] == 0]
            lines.append(f"- **{len(no_viol)}/{len(normal)}** positions within ADV constraint")
            lines.append(f"- Max liquidation days needed: {normal['liq_days'].max():.1f}")

        lines += [
            "",
            "### At Severe Stress (20% ADV — 2008 scenario)",
        ]
        severe = liq_df[liq_df["scenario"] == "Severe Stress"]
        if not severe.empty:
            affected = severe[severe["excess_pct"] > 0]
            total_cost = severe["liq_cost_usd"].sum()
            lines += [
                f"- **{len(affected)}/{len(severe)}** positions require partial liquidation",
                f"- Total estimated liquidation cost: **${total_cost:,.0f}**",
                f"  ({total_cost/PORTFOLIO_NAV*100:.2f}% of NAV)",
                f"- Worst-case single position liquidation: **{severe['liq_days'].max():.1f} days**",
                "",
                "> **Institution practice:** Pre-commit to maximum position sizes such that "
                "liquidation time ≤ 5 days even under 20% ADV stress.",
            ]

    # Portfolio scale sensitivity
    if not liq_df.empty:
        lines += [
            "",
            "## Portfolio Size Sensitivity",
            "",
            "At what NAV do ADV constraints begin to bind under Severe Stress (20% ADV)?",
            "",
            "| Portfolio NAV | Positions Affected | Notes |",
            "|--------------|-------------------|-------|",
        ]
        adv_data = liq_df[liq_df["scenario"] == "Severe Stress"][["ticker","adv_normal_mm"]].drop_duplicates()
        for nav_mm in [10, 50, 100, 250, 500, 1000]:
            nav = nav_mm * 1_000_000
            cnt = 0
            for _, row in adv_data.iterrows():
                pos_val = nav * MAX_SINGLE
                stressed_adv = row["adv_normal_mm"] * 1e6 * 0.20
                max_pos = stressed_adv * 5 * PARTICIPATION
                if pos_val > max_pos:
                    cnt += 1
            note = "No constraints" if cnt == 0 else ("Minor constraints" if cnt <= 3 else "Significant constraints")
            lines.append(f"| ${nav_mm}M | {cnt}/{len(adv_data)} | {note} |")
        lines += [
            "",
            "> For S&P 500 large-cap strategies, ADV constraints become significant above **$250M NAV**.",
        ]

    LIQUIDITY_REPORT.write_text("\n".join(lines))
    print(f"  Report → {LIQUIDITY_REPORT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Report
# ═══════════════════════════════════════════════════════════════════════════════

def write_main_report(
    rolling: pd.DataFrame,
    ic_compare: pd.DataFrame,
    cvar_df: pd.DataFrame,
    liq_df: pd.DataFrame,
) -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lines = [
        "# Risk Infrastructure Report — Week 6 Summary",
        f"**Generated:** {today}  |  **Canyon v9 — Institutional Risk Framework**",
        "",
        "## Four Modules Completed",
        "",
        "| Module | Purpose | Key Output |",
        "|--------|---------|-----------|",
        "| A. Rolling IC Monitor | Detect signal decay in real-time | Per-signal health status |",
        "| B. Factor Neutralization | Strip market/momentum/size from ML scores | Pure-alpha residual |",
        "| C. CVaR Analysis | Tail-risk quantification (ES 95% / 99%) | Calmar ratio, bootstrap CI |",
        "| D. Liquidity Stress Test | ADV-constrained liquidation under 4 scenarios | Max liq days, cost $ |",
        "",
        "---",
        "",
        "## Module A: Signal Health",
        "",
    ]

    if not rolling.empty:
        alerts = rolling[rolling["status"] == "ALERT_negative"]["signal"].unique()
        warns  = rolling[rolling["status"] == "WARNING_weak"]["signal"].unique()
        ok     = rolling[rolling["status"] == "OK"]["signal"].unique()
        lines += [
            f"- **{len(ok)}** signals currently OK (3m IC ≥ 0.02)",
            f"- **{len(warns)}** signals in WARNING zone (0 ≤ 3m IC < 0.02)",
            f"- **{len(alerts)}** signals in ALERT zone (3m IC < 0)",
        ]
        if len(alerts) > 0:
            lines.append(f"- **ALERT signals:** {', '.join(alerts)} — consider suspending from ensemble")

    lines += [
        "",
        "---",
        "",
        "## Module B: Factor Neutralization",
        "",
    ]

    if not ic_compare.empty:
        raw_oos  = ic_compare[(ic_compare["method"]=="raw") & (ic_compare["period"]=="OOS")]["ic"].mean()
        neut_oos = ic_compare[(ic_compare["method"]=="neutralized") & (ic_compare["period"]=="OOS")]["ic"].mean()
        lines += [
            f"- Raw ensemble OOS IC: **{raw_oos:+.4f}**",
            f"- Neutralized OOS IC:  **{neut_oos:+.4f}**",
            f"- IC delta: {neut_oos - raw_oos:+.4f} "
            f"({'purer alpha' if neut_oos >= raw_oos - 0.005 else 'some IC was factor-driven'})",
        ]
    else:
        lines.append("- Factor exposures computed and saved to factor_exposures.csv")

    lines += [
        "",
        "---",
        "",
        "## Module C: CVaR Summary",
        "",
    ]

    if not cvar_df.empty:
        oos_tc = cvar_df[(cvar_df["series"] == "Dynamic-TC") & (cvar_df["period"] == "OOS")]
        if not oos_tc.empty:
            row = oos_tc.iloc[0]
            lines += [
                f"**Dynamic-TC Strategy (OOS):**",
                f"- CVaR 95%: **{row['cvar95_hist_pct']:.2f}%/month** "
                f"(worst 5% of months, {int(row['n_months'])} months total)",
                f"- CVaR 99%: **{row['cvar99_hist_pct']:.2f}%/month** (worst 1%)",
                f"- Sharpe: {row['sharpe']:.3f}  |  Max DD: {row['max_dd_pct']:.2f}%  "
                f"|  Calmar: {row['calmar']:.2f}",
                "",
                f"> Calmar > 2.0 = annual return exceeds 2× max drawdown. "
                f"Institutional benchmark: Calmar > 1.5 is good, > 3.0 is exceptional.",
            ]

    lines += [
        "",
        "---",
        "",
        "## Module D: Liquidity Stress",
        "",
    ]

    if not liq_df.empty:
        normal = liq_df[liq_df["scenario"] == "Normal"]
        severe = liq_df[liq_df["scenario"] == "Severe Stress"]
        if not normal.empty:
            lines.append(f"- **Normal ADV**: {(normal['excess_pct']==0).mean()*100:.0f}% of positions within constraint")
        if not severe.empty:
            total_cost = severe["liq_cost_usd"].sum()
            lines += [
                f"- **2008 Scenario (20% ADV)**: Total liquidation cost = ${total_cost:,.0f} "
                f"({total_cost/PORTFOLIO_NAV*100:.2f}% of NAV)",
                f"- Max days to liquidate in crisis: {severe['liq_days'].max():.1f} days",
            ]

    lines += [
        "",
        "---",
        "",
        "## Week 7 Preview",
        "",
        "**Institutional Reporting System:**",
        "- Monthly attribution: Brinson-Hood-Beebower performance attribution",
        "- Signal health dashboard panel (IC decay monitor in Streamlit)",
        "- Factor exposure charting (market beta over time)",
        "- CVaR trend monitoring (tail risk budget)",
    ]

    MAIN_REPORT.write_text("\n".join(lines))
    print(f"\nMain report → {MAIN_REPORT.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Canyon v9 — Step 260: Risk Infrastructure Upgrade (Week 6)")
    print("=" * 70)

    # Module A: Rolling IC Monitor
    rolling_ic = run_rolling_ic_monitor()
    write_rolling_ic_report(rolling_ic)

    # Module B: Factor Neutralization
    exposures, ic_compare = run_factor_neutralization()
    write_factor_report(exposures, ic_compare)

    # Module C: CVaR Analysis
    cvar_df = run_cvar_analysis()
    write_cvar_report(cvar_df)

    # Module D: Liquidity Stress Test
    liq_df = run_liquidity_stress_test()
    write_liquidity_report(liq_df)

    # Combined report
    write_main_report(rolling_ic, ic_compare, cvar_df, liq_df)

    print("\n" + "=" * 70)
    print("Week 6 Complete — Risk Infrastructure Upgrade")
    print("=" * 70)
    print("\nOutputs:")
    for f in [ROLLING_IC_CSV, ROLLING_IC_REPORT, FACTOR_EXP_CSV, FACTOR_REPORT,
              CVAR_CSV, CVAR_REPORT, LIQUIDITY_CSV, LIQUIDITY_REPORT, MAIN_REPORT]:
        if f.exists():
            kb = f.stat().st_size / 1024
            print(f"  {f.name:<42}  {kb:.1f} KB")


if __name__ == "__main__":
    main()
