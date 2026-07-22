"""
W35: Backtest Credibility Scorecard
=====================================
Institutional investors evaluate backtest validity on multiple dimensions.
This scorecard quantifies each dimension and computes an overall credibility score.

Scorecard dimensions (Pardo 2008, Harvey et al. 2016):
  1. OOS Ratio       — fraction of backtest that is out-of-sample (target: ≥ 50%)
  2. OOS Sharpe      — OOS Sharpe ratio (target: ≥ 0.8)
  3. IS/OOS Gap      — difference in Sharpe: IS - OOS (target: < 0.5)
  4. Max Drawdown    — OOS max drawdown (target: < 25%)
  5. Calmar Ratio    — annualised return / max drawdown (target: ≥ 0.5)
  6. Turnover        — monthly portfolio turnover (target: < 60% for cost realism)
  7. Parameter Count — number of free parameters (fewer = more credible)
  8. Deflated Sharpe — Sharpe adjusted for multiple testing (Bailey & de Prado 2016)
  9. Hit Rate        — fraction of months with positive return
  10. PIT Compliance — no lookahead bias in fundamental data

Outputs:
  backtest_scorecard.csv  — dimension × score × commentary

Usage:
    from research.backtest_scorecard import run_scorecard
    scorecard = run_scorecard()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def _load_backtest_results() -> pd.DataFrame:
    """Load v11 backtest monthly results."""
    for fname in ("v11_backtest_monthly.csv", "walk_forward_monthly.csv"):
        p = ROOT / fname
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"] if "date" in pd.read_csv(p, nrows=0).columns else None)
    return pd.DataFrame()


def _load_backtest_summary() -> dict:
    """Load v11 backtest summary stats."""
    p = ROOT / "v11_backtest_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df.iloc[0].to_dict() if not df.empty else {}
    return {}


def _deflated_sharpe_ratio(sharpe_is: float, n_trials: int,
                            n_obs: int = 240) -> float:
    """
    Bailey & de Prado (2016) Deflated Sharpe Ratio.

    Adjusts IS Sharpe for the probability that it is spurious given n_trials tested.

    DSR = (SR_is - E[SR_max]) / sqrt(Var[SR_max])
    where E[SR_max] ≈ (1 - γ) × Z^{-1}(1 - 1/n) + γ × Z^{-1}(1 - 1/(n×e))

    Simplified approximation: penalise by sqrt(log(n_trials)).
    """
    from scipy.special import ndtri
    if n_trials <= 1:
        return sharpe_is
    # Expected max Sharpe under multiple testing
    e_max_sr = np.sqrt(2 * np.log(n_trials)) / np.sqrt(n_obs)
    sr_deflated = (sharpe_is - e_max_sr) / np.sqrt(1 + sharpe_is ** 2 / (2 * n_obs))
    return float(sr_deflated)


def _score_dimension(name: str, value: float,
                     target: float, threshold: float, higher_is_better: bool = True) -> dict:
    """
    Score a single dimension.

    Returns dict: name, value, target, score [0-10], status.
    """
    if higher_is_better:
        if value >= target:
            score  = 10.0
            status = "PASS"
        elif value >= threshold:
            score  = 5.0 + 5.0 * (value - threshold) / (target - threshold)
            status = "WARN"
        else:
            score  = max(0.0, 5.0 * value / (threshold + 1e-9))
            status = "FAIL"
    else:
        if value <= target:
            score  = 10.0
            status = "PASS"
        elif value <= threshold:
            score  = 5.0 + 5.0 * (threshold - value) / (threshold - target)
            status = "WARN"
        else:
            score  = max(0.0, 5.0 * threshold / (value + 1e-9))
            status = "FAIL"

    return {
        "dimension": name,
        "value":     round(value, 4),
        "target":    target,
        "threshold": threshold,
        "score":     round(score, 1),
        "status":    status,
    }


def run_scorecard(
    output_path: Optional[Path] = None,
    n_trials: int = 50,  # number of parameter combinations tested
) -> pd.DataFrame:
    """
    Run the backtest credibility scorecard.

    Args:
        output_path: Where to save scorecard CSV.
        n_trials:    Number of strategy variants tested (for deflated Sharpe).

    Returns DataFrame: dimension × score × commentary.
    """
    if output_path is None:
        output_path = ROOT / "backtest_scorecard.csv"

    bt_monthly = _load_backtest_results()
    summary    = _load_backtest_summary()

    print("\n[Scorecard] Backtest Credibility Scorecard")
    print("=" * 55)

    dimensions = []

    # 1. OOS Ratio
    oos_ratio = float(summary.get("oos_fraction", 0.5))
    if oos_ratio == 0.5 and not bt_monthly.empty:
        n_total = len(bt_monthly)
        n_oos   = int(summary.get("n_oos_months", n_total // 2))
        oos_ratio = n_oos / n_total if n_total > 0 else 0.0
    dimensions.append(_score_dimension("oos_ratio", oos_ratio, target=0.6, threshold=0.4))

    # 2. OOS Sharpe
    oos_sharpe = float(summary.get("oos_sharpe", 0.0))
    if oos_sharpe == 0.0 and not bt_monthly.empty:
        oos_col = "oos_return" if "oos_return" in bt_monthly.columns else \
                  "long_short_return" if "long_short_return" in bt_monthly.columns else \
                  "ls_return" if "ls_return" in bt_monthly.columns else None
        if oos_col:
            r = bt_monthly[oos_col].dropna()
            if len(r) > 12:
                oos_sharpe = float(r.mean() / r.std() * np.sqrt(12))
    dimensions.append(_score_dimension("oos_sharpe", oos_sharpe, target=0.80, threshold=0.40))

    # 3. IS/OOS Sharpe gap
    is_sharpe = float(summary.get("is_sharpe", oos_sharpe * 1.5))
    is_oos_gap = is_sharpe - oos_sharpe
    dimensions.append(_score_dimension("is_oos_gap", is_oos_gap,
                                       target=0.2, threshold=0.6, higher_is_better=False))

    # 4. Max Drawdown (OOS)
    max_dd = float(summary.get("oos_max_dd", 0.15))
    if max_dd == 0.15 and not bt_monthly.empty:
        oos_col = [c for c in bt_monthly.columns if "return" in c.lower() and
                   ("ls" in c.lower() or "oos" in c.lower())]
        if oos_col:
            r = bt_monthly[oos_col[0]].dropna()
            cum = (1 + r).cumprod()
            peak = cum.cummax()
            drawdown = (cum - peak) / peak
            max_dd = float(-drawdown.min()) if len(drawdown) > 0 else 0.15
    dimensions.append(_score_dimension("max_drawdown", max_dd,
                                       target=0.15, threshold=0.30, higher_is_better=False))

    # 5. Calmar Ratio
    ann_ret = float(summary.get("oos_annual_return", oos_sharpe * 0.10))
    calmar = ann_ret / (max_dd + 1e-9) if max_dd > 0 else 0.0
    dimensions.append(_score_dimension("calmar_ratio", calmar, target=0.80, threshold=0.30))

    # 6. Monthly Turnover (lower = more realistic)
    turnover = float(summary.get("monthly_turnover", 0.35))
    dimensions.append(_score_dimension("monthly_turnover", turnover,
                                       target=0.30, threshold=0.70, higher_is_better=False))

    # 7. Parameter count (fewer = more credible)
    # Canyon v11 has: beta_window(1), sector_cap(1), max_pos(1), IC_lookback(1),
    # top_n(1), VIX_threshold(1), short_borrow(1) = ~7 parameters
    n_params = int(summary.get("n_free_params", 7))
    dimensions.append(_score_dimension("n_free_params", n_params,
                                       target=5, threshold=15, higher_is_better=False))

    # 8. Deflated Sharpe (accounts for multiple testing)
    dsr = _deflated_sharpe_ratio(oos_sharpe, n_trials=n_trials, n_obs=240)
    dimensions.append(_score_dimension("deflated_sharpe", dsr, target=0.50, threshold=0.10))

    # 9. Hit Rate (fraction of months with positive OOS return)
    hit_rate = float(summary.get("oos_hit_rate", 0.55))
    if hit_rate == 0.55 and not bt_monthly.empty:
        oos_col = [c for c in bt_monthly.columns if "return" in c.lower() and
                   ("ls" in c.lower() or "oos" in c.lower())]
        if oos_col:
            r = bt_monthly[oos_col[0]].dropna()
            hit_rate = float((r > 0).mean())
    dimensions.append(_score_dimension("hit_rate", hit_rate, target=0.58, threshold=0.50))

    # 10. PIT Compliance
    pit_score = 1.0  # default: compliant if using EDGAR PIT data (W4-W5, W14)
    pit_path  = ROOT / "edgar_pit_fundamentals.csv"
    if pit_path.exists():
        try:
            pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
            violations = (pit_df["know_date"] <= pit_df["period_end"]).sum()
            pit_score = 1.0 if violations == 0 else 1.0 - violations / len(pit_df)
        except Exception:
            pit_score = 0.5  # no EDGAR = partial compliance
    dimensions.append(_score_dimension("pit_compliance", pit_score, target=1.0, threshold=0.90))

    # Compute overall score (weighted average)
    df = pd.DataFrame(dimensions)
    weights = {
        "oos_ratio":      1.5,
        "oos_sharpe":     2.0,
        "is_oos_gap":     2.0,
        "max_drawdown":   1.5,
        "calmar_ratio":   1.0,
        "monthly_turnover": 0.5,
        "n_free_params":  0.5,
        "deflated_sharpe": 2.0,
        "hit_rate":        1.0,
        "pit_compliance":  2.0,
    }
    df["weight"] = df["dimension"].map(weights).fillna(1.0)
    overall = float((df["score"] * df["weight"]).sum() / df["weight"].sum())

    # Print scorecard
    print(f"  {'Dimension':<22} {'Value':>10}  {'Target':>8}  {'Score':>6}  Status")
    print(f"  {'─'*65}")
    for _, row in df.iterrows():
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(row["status"], "?")
        print(f"  {row['dimension']:<22} {row['value']:>10.4f}  {row['target']:>8.3f}  "
              f"{row['score']:>6.1f}  {icon} {row['status']}")

    print(f"\n  {'─'*65}")
    print(f"  Overall Credibility Score: {overall:.1f} / 10.0")
    if overall >= 7.0:
        print("  ✓  INSTITUTIONAL GRADE (≥ 7.0)")
    elif overall >= 5.0:
        print("  ⚠  ACCEPTABLE (5.0-7.0) — improvements needed for institutional use")
    else:
        print("  ✗  BELOW STANDARD (< 5.0) — significant issues to address")

    df["overall_score"] = overall
    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")
    return df


if __name__ == "__main__":
    run_scorecard()
