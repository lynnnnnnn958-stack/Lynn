#!/usr/bin/env python3
"""
Canyon v9  Step 221 — Fama-French 5-Factor Attribution
=======================================================
Downloads FF5 daily factors from Ken French's website and runs
OLS regression to decompose portfolio returns into:

    R_p − Rf = α + β_Mkt·MKT + β_SMB·SMB + β_HML·HML
                              + β_RMW·RMW + β_CMA·CMA + ε

WHERE:
    R_p   = portfolio daily return (from sp500_price_cache + cvxpy weights)
    Rf    = daily risk-free rate (from FF5 file)
    MKT   = market excess return (CRSP value-weighted index − Rf)
    SMB   = Small Minus Big (size factor)
    HML   = High Minus Low (value factor)
    RMW   = Robust Minus Weak (profitability factor)
    CMA   = Conservative Minus Aggressive (investment factor)
    α     = Jensen's alpha (true excess return not explained by factors)

WHY THIS MATTERS
----------------
If α is close to 0 and t-stat < 2: you're not generating alpha,
you're loading on known risk factors (momentum, value, etc.).
Knowing your factor loadings tells you:
  - How much of your "edge" is just systematic factor premium?
  - Where your real concentration risk is?
  - Is the portfolio genuinely different from a factor ETF?

INPUTS
------
    sp500_price_cache.csv      — daily prices
    cvxpy_weights.csv          — current portfolio weights (from step220)
    ff5_daily.csv              — downloaded from Ken French (auto-fetched)

OUTPUTS
-------
    ff5_daily.csv              — cached FF5 factors
    factor_attribution.csv     — α, β, t-stats, R² for multiple windows
    factor_attribution_report.md — plain-English explanation
"""
from __future__ import annotations

import io
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FF5_CACHE  = ROOT / "ff5_daily.csv"
FF5_URL    = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
              "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
FF5_TTL_DAYS = 30   # re-download if cache > 30 days old


# =============================================================================
# 1. Download Fama-French 5 factors
# =============================================================================

def download_ff5() -> pd.DataFrame:
    """
    Download and parse FF5 daily factors from Ken French's data library.
    Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF (all in %)
    """
    import time as _time

    # Use cache if fresh
    if FF5_CACHE.exists():
        age_d = (_time.time() - FF5_CACHE.stat().st_mtime) / 86400
        if age_d < FF5_TTL_DAYS:
            df = pd.read_csv(FF5_CACHE, index_col=0, parse_dates=True)
            print(f"  FF5 cache hit ({age_d:.0f}d old) — {len(df)} trading days")
            return df

    print("  Downloading FF5 daily from Ken French data library …")
    import requests as _req
    resp = _req.get(FF5_URL, timeout=30, verify=True)
    resp.raise_for_status()
    raw = resp.content

    zf   = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
    content  = zf.read(csv_name).decode("utf-8", errors="replace")

    # Parse: skip header rows until we hit numeric date lines
    lines  = content.splitlines()
    data_lines = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_data:
                break
            continue
        # Check if first token looks like a date (YYYYMMDD)
        first = stripped.split(",")[0].strip()
        if first.isdigit() and len(first) == 8:
            in_data = True
            data_lines.append(stripped)
        elif in_data:
            break

    if not data_lines:
        raise ValueError("Could not parse FF5 file — format may have changed.")

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            date = pd.to_datetime(parts[0], format="%Y%m%d")
            vals = [float(p) for p in parts[1:7]]
            rows.append([date] + vals)
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows, columns=["date","Mkt-RF","SMB","HML","RMW","CMA","RF"])
    df = df.set_index("date").sort_index()

    # Values are already in percent — convert to decimal
    for col in df.columns:
        df[col] = df[col] / 100.0

    df.to_csv(FF5_CACHE)
    print(f"  Downloaded {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# =============================================================================
# 2. Portfolio return series
# =============================================================================

def build_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.Series,
    start_date: str = "2015-01-01",
) -> pd.Series:
    """
    Compute daily portfolio return series from price data and weights.
    Uses the cvxpy_weights alpha_mv column as static weights.
    (No rebalancing assumed — this is an approximation.)
    """
    tickers = [t for t in weights.index if t in prices.columns]
    w       = weights[tickers]
    w       = w / w.sum()   # renormalise

    px  = prices[tickers].loc[start_date:].copy()
    ret = px.pct_change().dropna()

    # Portfolio daily return = weighted sum of individual returns
    port_ret = (ret * w.values).sum(axis=1)
    return port_ret


# =============================================================================
# 3. OLS factor regression
# =============================================================================

def run_regression(
    port_ret: pd.Series,
    ff5: pd.DataFrame,
    label: str = "Full sample",
) -> dict:
    """
    Regress portfolio excess return on FF5 factors.
    Returns dict with alpha, betas, t-stats, R², p-values.
    """
    # Align dates
    common = port_ret.index.intersection(ff5.index)
    if len(common) < 30:
        return {"label": label, "n_obs": len(common), "status": "insufficient"}

    R  = port_ret[common]
    Rf = ff5.loc[common, "RF"]
    R_ex = R - Rf   # excess return

    factors = ff5.loc[common, ["Mkt-RF","SMB","HML","RMW","CMA"]]
    X = sm.add_constant(factors)

    model  = sm.OLS(R_ex, X).fit(cov_type="HC3")   # heteroskedasticity-robust

    alpha_ann = float(model.params["const"] * 252)
    alpha_t   = float(model.tvalues["const"])
    alpha_p   = float(model.pvalues["const"])

    betas = {f: float(model.params[f]) for f in ["Mkt-RF","SMB","HML","RMW","CMA"]}
    tstats= {f: float(model.tvalues[f]) for f in ["Mkt-RF","SMB","HML","RMW","CMA"]}

    return {
        "label":     label,
        "n_obs":     int(len(common)),
        "start":     str(common[0].date()),
        "end":       str(common[-1].date()),
        "status":    "OK",
        "alpha_ann": alpha_ann,
        "alpha_t":   alpha_t,
        "alpha_p":   alpha_p,
        "r_squared": float(model.rsquared),
        "adj_r2":    float(model.rsquared_adj),
        "betas":     betas,
        "tstats":    tstats,
        "residual_vol_ann": float(model.resid.std() * np.sqrt(252)),
        "info_ratio": alpha_ann / (model.resid.std() * np.sqrt(252)) if model.resid.std() > 0 else 0.0,
    }


# =============================================================================
# 4. Plain-English interpreter
# =============================================================================

def interpret_alpha(res: dict) -> str:
    if res.get("status") != "OK":
        return "Not enough data to interpret."
    a  = res["alpha_ann"]
    t  = res["alpha_t"]
    ir = res.get("info_ratio", 0)
    if abs(t) < 1.5:
        verdict = (f"Alpha is statistically indistinguishable from zero "
                   f"(t={t:.2f}). Your excess return is likely explained "
                   f"by factor exposures, not genuine skill.")
    elif t >= 1.5 and abs(a) > 0:
        verdict = (f"Weak positive signal: alpha = {a*100:+.2f}% annualised "
                   f"(t={t:.2f}). Encouraging but not statistically proven yet. "
                   f"Need t > 2.0 for 95% confidence.")
    else:
        verdict = (f"Strong alpha signal: {a*100:+.2f}% annualised (t={t:.2f}). "
                   f"This exceeds the 95% significance threshold.")
    return verdict


def interpret_betas(res: dict) -> list[str]:
    if res.get("status") != "OK":
        return []
    betas  = res["betas"]
    tstats = res["tstats"]
    lines  = []

    factor_meanings = {
        "Mkt-RF": ("Market", "broadly tracks the S&P 500"),
        "SMB":    ("Size (small vs big)", "exposure > 0 = tilted toward smaller companies"),
        "HML":    ("Value", "exposure > 0 = cheap stocks, < 0 = growth stocks"),
        "RMW":    ("Profitability", "exposure > 0 = high profit companies"),
        "CMA":    ("Investment", "exposure < 0 = aggressive growth spenders"),
    }
    for f, (name, meaning) in factor_meanings.items():
        b = betas[f]
        t = tstats[f]
        sig = " ✅ (significant)" if abs(t) > 2 else " (not significant)"
        lines.append(f"- **{name}** β={b:+.3f} (t={t:.2f}){sig}: {meaning}")
    return lines


# =============================================================================
# 5. Report writer
# =============================================================================

def write_report(results: list[dict], weights: pd.Series) -> None:
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Canyon v9 — Fama-French 5-Factor Attribution (Step 221)",
        f"Generated: {ts}\n",
        "## Why Factor Attribution Matters",
        "",
        "A portfolio can appear to generate alpha while actually just loading on",
        "known risk factors (market, size, value, profitability, investment).",
        "This report decomposes returns into **factor exposure** vs **true alpha (Jensen's α)**.",
        "",
        "If alpha t-stat < 2.0: the system is not proven to generate skill — it may just be a",
        "factor ETF in disguise. This is honest and expected at this stage.",
        "",
    ]

    for res in results:
        lines += [
            f"## Window: {res['label']}",
            f"**Period:** {res.get('start','?')} → {res.get('end','?')}  "
            f"(**{res.get('n_obs','?')} trading days**)\n",
        ]
        if res.get("status") != "OK":
            lines += ["> Insufficient data for this window.\n"]
            continue

        a   = res["alpha_ann"]
        t_a = res["alpha_t"]
        r2  = res["r_squared"]
        ir  = res["info_ratio"]

        lines += [
            "### Headline Numbers",
            "",
            f"| Metric | Value | Interpretation |",
            f"|--------|-------|----------------|",
            f"| Jensen's α (annualised) | {a*100:+.2f}% | True skill after factor adjustment |",
            f"| α t-statistic | {t_a:.2f} | Need > 2.0 for 95% confidence |",
            f"| R² | {r2:.3f} | How much return is explained by factors |",
            f"| Information Ratio | {ir:.2f} | α / tracking error |",
            "",
            "### Verdict",
            "",
            interpret_alpha(res),
            "",
            "### Factor Exposures (β)",
            "",
        ]
        lines += interpret_betas(res)
        lines.append("")

    # Current portfolio composition
    lines += [
        "## Current Portfolio (cvxpy Alpha-MV)",
        "",
        f"| Ticker | Weight | Sector |",
        f"|--------|--------|--------|",
    ]
    if not weights.empty:
        for tkr, w in weights[weights > 0.001].sort_values(ascending=False).items():
            lines.append(f"| {tkr} | {w*100:.2f}% | — |")

    lines += [
        "",
        "## Data Sources",
        "",
        "- **Factor data**: Kenneth R. French Data Library (mba.tuck.dartmouth.edu)",
        "  — Fama-French 5-Factor daily returns (freely available, updated monthly)",
        "- **Portfolio returns**: sp500_price_cache.csv (survivorship-biased — treat regression as approximate)",
        "- **Weights**: cvxpy_weights.csv (alpha_mv column from Step 220)",
        "",
        "⚠️ **Survivorship bias note**: Portfolio returns are computed from the current S&P 500",
        "constituents going back in time. This inflates apparent historical performance.",
        "Alpha estimates from this regression are hypothesis-grade, not production-grade.",
    ]

    (ROOT / "factor_attribution_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6. Main
# =============================================================================

def run() -> dict:
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 221: Factor Attribution  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"{'='*65}")

    # ── Load FF5 ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading Fama-French 5 factors …")
    try:
        ff5 = download_ff5()
    except Exception as e:
        print(f"  [ERROR] Could not load FF5 data: {e}")
        return {"status": "error", "msg": str(e)}

    # ── Load prices ───────────────────────────────────────────────────────────
    print("\n[2/5] Loading portfolio weights & prices …")
    price_path   = ROOT / "sp500_price_cache.csv"
    weights_path = ROOT / "cvxpy_weights.csv"

    if not price_path.exists():
        print("  [ERROR] sp500_price_cache.csv not found.")
        return {}
    if not weights_path.exists():
        print("  [ERROR] cvxpy_weights.csv not found — run step220 first.")
        return {}

    prices  = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    wdf     = pd.read_csv(weights_path)
    wdf     = wdf[wdf["alpha_mv"].notna()]
    weights = pd.Series(wdf["alpha_mv"].values, index=wdf["ticker"])
    weights = weights / weights.sum()
    print(f"  {len(weights)} tickers  (weights sum = {weights.sum():.4f})")

    # ── Build portfolio return series ─────────────────────────────────────────
    print("\n[3/5] Building portfolio return series …")
    port_ret = build_portfolio_returns(prices, weights, start_date="2018-01-01")
    print(f"  Portfolio returns: {len(port_ret)} days "
          f"({port_ret.index[0].date()} → {port_ret.index[-1].date()})")
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe  = (ann_ret - 0.053) / ann_vol if ann_vol > 0 else 0
    print(f"  Gross Sharpe (no factor adj.): {sharpe:.2f}  "
          f"ret={ann_ret*100:.1f}%  vol={ann_vol*100:.1f}%")

    # ── Run regressions on multiple windows ───────────────────────────────────
    print("\n[4/5] Running OLS factor regressions …")
    windows = [
        ("Full available", "2018-01-01"),
        ("3 years",        (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")),
        ("1 year",         (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")),
    ]

    results = []
    for label, start in windows:
        sub_ret = port_ret[port_ret.index >= start]
        res     = run_regression(sub_ret, ff5, label=label)
        results.append(res)
        if res.get("status") == "OK":
            a = res["alpha_ann"]
            t = res["alpha_t"]
            r2= res["r_squared"]
            print(f"  {label:15s}: α={a*100:+.2f}%/yr  t={t:.2f}  R²={r2:.3f}  "
                  f"({'✅ significant' if abs(t)>2 else '⚠ not proven'})")
        else:
            print(f"  {label:15s}: {res.get('status','?')}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for res in results:
        if res.get("status") != "OK":
            continue
        row = {
            "window":    res["label"],
            "start":     res["start"],
            "end":       res["end"],
            "n_obs":     res["n_obs"],
            "alpha_ann": res["alpha_ann"],
            "alpha_t":   res["alpha_t"],
            "alpha_p":   res["alpha_p"],
            "r_squared": res["r_squared"],
            "info_ratio":res["info_ratio"],
        }
        for f in ["Mkt-RF","SMB","HML","RMW","CMA"]:
            row[f"beta_{f.replace('-','_')}"] = res["betas"].get(f)
            row[f"t_{f.replace('-','_')}"]    = res["tstats"].get(f)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ROOT / "factor_attribution.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[5/5] Writing report …")
    write_report(results, weights)

    print(f"  [written] factor_attribution.csv")
    print(f"  [written] factor_attribution_report.md")

    # Final summary
    full_res = next((r for r in results if r["label"] == "Full available"
                     and r.get("status") == "OK"), None)
    if full_res:
        a = full_res["alpha_ann"]
        t = full_res["alpha_t"]
        b_mkt = full_res["betas"].get("Mkt-RF", 0)
        print(f"\n  ─── KEY FINDING ───────────────────────────────────────────")
        print(f"  Jensen's α = {a*100:+.2f}% annualised  (t={t:.2f})")
        if abs(t) < 2:
            print(f"  ⚠  Alpha is NOT statistically significant.")
            print(f"     Most of the portfolio return is factor exposure, not skill.")
        else:
            print(f"  ✅ Alpha IS statistically significant at 95% level.")
        print(f"  Market β = {b_mkt:.3f}  (R² = {full_res['r_squared']:.3f})")
        print(f"  ────────────────────────────────────────────────────────────\n")
        return {"status": "OK", "alpha_ann": a, "alpha_t": t, "r2": full_res["r_squared"]}

    return {"status": "OK"}


if __name__ == "__main__":
    import sys
    result = run()
    sys.exit(0 if result.get("status") == "OK" else 1)
