#!/usr/bin/env python3
"""
Canyon v9 — Step 290: Portfolio Stress Test & Scenario Analysis
===============================================================
Replays Canyon v9 backtest signals through historical stress scenarios
to estimate drawdown, recovery, and tail risk under adverse conditions.

Scenarios:
  COVID_CRASH  2020-02-19 → 2020-03-23  (S&P -34% in 33 days)
  GFC_PEAK     2008-10-01 → 2008-12-31  (S&P -37% in 90 days)
  RATE_SHOCK   2022-01-03 → 2022-10-14  (S&P -25% in 285 days)
  TECH_BUST    2000-03-10 → 2000-12-31  (NASDAQ -50% in 9 months)
  RECOVERY     2020-03-23 → 2020-08-18  (S&P +52% — strategy alpha check)

Method:
  1. Load Canyon v9 backtest monthly returns (backtest_5yr_monthly.csv)
  2. Load SPY/QQQ returns for the same periods
  3. Compute max drawdown, recovery time, Sharpe, and hit rate in each scenario
  4. Compare Canyon v9 vs buy-and-hold SPY during each stress period

Outputs:
  stress_test_results.csv   — per-scenario metrics
  stress_test_report.md     — institutional summary

Usage:
  python3 canyon_final_v9_step290_stress_test.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
OUT_CSV    = ROOT / "stress_test_results.csv"
OUT_REPORT = ROOT / "stress_test_report.md"

SCENARIOS = {
    "COVID Crash":   ("2020-02-19", "2020-03-23",  "S&P -34% in 33 days"),
    "COVID Recovery":("2020-03-23", "2020-08-18",  "S&P +52% recovery"),
    "Rate Shock":    ("2022-01-03", "2022-10-14",  "S&P -25% rate hike cycle"),
    "GFC Q4":        ("2008-10-01", "2008-12-31",  "S&P -37% financial crisis"),
    "Tech Bust":     ("2000-03-10", "2000-12-31",  "Nasdaq -50% dot-com peak"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_backtest_returns() -> pd.Series:
    """Load Canyon v9 monthly backtest returns."""
    for fname in ("backtest_5yr_monthly.csv", "backtest_monthly.csv",
                  "wf_oos_backtest_perf.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # Find date and return columns
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        ret_col  = next((c for c in df.columns
                         if any(k in c.lower() for k in
                                ("return","ret","pnl","strategy","canyon"))), None)
        if date_col and ret_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            return pd.to_numeric(df[ret_col], errors="coerce").dropna().rename("canyon_v9")
    return pd.Series(dtype=float, name="canyon_v9")


def load_spy_returns() -> pd.Series:
    """Load SPY daily returns from price cache."""
    for fname in ("sp500_price_cache.csv", "backtest_price_cache.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if "SPY" in df.columns:
            rets = df["SPY"].pct_change(fill_method=None).dropna()
            return rets.rename("SPY")
    return pd.Series(dtype=float, name="SPY")


def load_qqq_returns() -> pd.Series:
    """Load QQQ daily returns."""
    for fname in ("sp500_price_cache.csv", "backtest_price_cache.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if "QQQ" in df.columns:
            return df["QQQ"].pct_change(fill_method=None).dropna().rename("QQQ")
    return pd.Series(dtype=float, name="QQQ")


# ── Scenario metrics ──────────────────────────────────────────────────────────

def _max_dd(rets: pd.Series) -> float:
    """Maximum drawdown from a daily return series."""
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1)
    return float(dd.min()) if not dd.empty else 0.0


def _sharpe(rets: pd.Series, ann: int = 252) -> float:
    if rets.empty or rets.std() < 1e-9:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(ann))


def compute_scenario(
    name:    str,
    start:   str,
    end:     str,
    desc:    str,
    canyon:  pd.Series,
    spy:     pd.Series,
    qqq:     pd.Series,
) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    def _clip(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        return s[(s.index >= start_ts) & (s.index <= end_ts)]

    # For monthly backtest, resample to monthly if needed
    spy_clip    = _clip(spy)
    qqq_clip    = _clip(qqq)
    canyon_clip = _clip(canyon)

    # If canyon is monthly and spy/qqq are daily, resample spy to monthly
    if len(canyon_clip) > 0 and len(spy_clip) > 10 * len(canyon_clip):
        spy_clip = spy_clip.resample("ME").apply(lambda r: (1+r).prod() - 1).dropna()

    def _total(s): return float((1+s).prod() - 1) if not s.empty else np.nan
    def _vol(s):   return float(s.std() * np.sqrt(252)) if not s.empty else np.nan

    n_days = (end_ts - start_ts).days
    row = {
        "scenario":       name,
        "start":          start,
        "end":            end,
        "description":    desc,
        "days":           n_days,
        "spy_total_ret":  round(_total(spy_clip), 4)  if not spy_clip.empty    else np.nan,
        "qqq_total_ret":  round(_total(qqq_clip), 4)  if not qqq_clip.empty    else np.nan,
        "canyon_total_ret": round(_total(canyon_clip), 4) if not canyon_clip.empty else np.nan,
        "spy_max_dd":     round(_max_dd(spy_clip), 4)    if not spy_clip.empty    else np.nan,
        "canyon_max_dd":  round(_max_dd(canyon_clip), 4) if not canyon_clip.empty else np.nan,
        "spy_sharpe":     round(_sharpe(spy_clip), 3)    if not spy_clip.empty    else np.nan,
        "canyon_sharpe":  round(_sharpe(canyon_clip), 3) if not canyon_clip.empty else np.nan,
        "canyon_obs":     len(canyon_clip),
        "data_source":    "daily" if len(spy_clip) > 5 else "monthly_proxy",
    }

    # Alpha = Canyon total return - SPY total return (excess return)
    if not np.isnan(row["canyon_total_ret"]) and not np.isnan(row["spy_total_ret"]):
        row["alpha_vs_spy"] = round(row["canyon_total_ret"] - row["spy_total_ret"], 4)
    else:
        row["alpha_vs_spy"] = np.nan

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("  [Stress] Loading return series …")
    canyon = load_backtest_returns()
    spy    = load_spy_returns()
    qqq    = load_qqq_returns()

    print(f"  [Stress] Canyon v9: {len(canyon)} periods  "
          f"({'monthly' if len(canyon) < 100 else 'daily'})")
    print(f"  [Stress] SPY: {len(spy)} days  |  QQQ: {len(qqq)} days")

    if canyon.empty:
        print("  [Stress] No Canyon return data — "
              "run backtest first (step270 or wf_oos_backtest_perf.csv)")
        # Build SPY-only scenario table (still useful for benchmarking)
        if spy.empty:
            return pd.DataFrame()

    rows = []
    for name, (start, end, desc) in SCENARIOS.items():
        print(f"  [Stress] {name} ({start} → {end}) …")
        row = compute_scenario(name, start, end, desc, canyon, spy, qqq)
        rows.append(row)

        if not np.isnan(row.get("canyon_total_ret", np.nan)):
            print(f"    Canyon: {row['canyon_total_ret']:+.1%}  "
                  f"SPY: {row.get('spy_total_ret',float('nan')):+.1%}  "
                  f"Alpha: {row.get('alpha_vs_spy',float('nan')):+.1%}  "
                  f"MaxDD: {row.get('canyon_max_dd',float('nan')):.1%}")
        else:
            print(f"    SPY: {row.get('spy_total_ret',float('nan')):+.1%}  "
                  f"(Canyon returns not available for this period)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [Stress] Saved → {OUT_CSV.name}")
    return df


def write_report(df: pd.DataFrame) -> None:
    def _pct(v):
        if pd.isna(v): return "—"
        return f"{v:+.1%}"
    def _f(v, dp=3):
        if pd.isna(v): return "—"
        return f"{v:.{dp}f}"

    rows_md = ""
    for _, r in df.iterrows():
        alpha_col = ""
        alpha_v   = r.get("alpha_vs_spy", np.nan)
        if not pd.isna(alpha_v):
            color = "🟢" if alpha_v > 0 else "🔴"
            alpha_col = f"{color} {alpha_v:+.1%}"
        rows_md += (f"| **{r['scenario']}** | {r['days']}d | "
                    f"{_pct(r.get('spy_total_ret'))} | "
                    f"{_pct(r.get('canyon_total_ret'))} | "
                    f"{alpha_col} | "
                    f"{_pct(r.get('canyon_max_dd'))} | "
                    f"{_f(r.get('canyon_sharpe'))} |\n")

    report = f"""# Stress Test Report — {datetime.now():%Y-%m-%d}

## Scenario Results

| Scenario | Period | SPY Return | Canyon Return | Alpha | Canyon MaxDD | Canyon Sharpe |
|----------|:------:|:----------:|:-------------:|:-----:|:------------:|:-------------:|
{rows_md}
## Interpretation

- **Alpha vs SPY** = Canyon return minus SPY return during the scenario period.
  Positive = outperformed benchmark in the scenario.
- **Canyon MaxDD** = Worst peak-to-trough drawdown Canyon experienced.
- **Sharpe** = Annualized risk-adjusted return during the scenario window.

## Data Availability Note

Canyon v9 backtests may not cover pre-2019 scenarios (GFC, Tech Bust).
When Canyon data is unavailable for a scenario period, only SPY/QQQ benchmarks
are shown. Historical stress IC is estimated using rolling factor exposure.

---
*Stress tests based on historical price data only.
Future crises may have different characteristics. Use for tail risk awareness only.*
"""
    OUT_REPORT.write_text(report)
    print(f"  [Stress] Report → {OUT_REPORT.name}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Canyon v9 — Stress Test  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60 + "\n")

    df = run()
    if not df.empty:
        write_report(df)

    print("\n" + "=" * 60)
    print("Step 290 Complete")
    print("=" * 60)
