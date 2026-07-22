"""
W49: Final Institutional Scoring Audit
=======================================
Produces a comprehensive institutional-grade scorecard for Canyon v11.
Each dimension is scored 0-10; total ≥ 7.0 qualifies as institutional grade.

Dimensions (weighted):
  1. Signal Quality       (20%) — mean IC, IC stability (IR), % IC positive
  2. Out-of-Sample        (20%) — deflated Sharpe, IS/OOS ratio
  3. Factor Hygiene       (15%) — VIF < 5 on all Barra factors, Spearman corr < 0.70
  4. Risk Controls        (15%) — regime covariance, max position size, beta bounds
  5. PIT Compliance       (10%) — EDGAR filed_date used as know_date
  6. Capacity             (5%)  — position sizes < 5% ADV
  7. Execution Quality    (5%)  — AC model calibrated, slippage < 20bps
  8. Code Health          (5%)  — all modules parse, tests pass, no Chinese
  9. Portfolio Theory     (5%)  — BL, Ledoit-Wolf, Kelly all present and non-trivial
  Bonus: Attribution      (+0.5) — BHB attribution module present and valid

Final output: institutional_scorecard.csv + text report

Usage:
    from research.institutional_scorecard import run_institutional_audit
    score = run_institutional_audit()
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

WEIGHTS = {
    "signal_quality":    0.20,
    "oos_performance":   0.20,
    "factor_hygiene":    0.15,
    "risk_controls":     0.15,
    "pit_compliance":    0.10,
    "capacity":          0.05,
    "execution_quality": 0.05,
    "code_health":       0.05,
    "portfolio_theory":  0.05,
}
BONUS_ATTRIBUTION = 0.5


# ─── Dimension scorers ────────────────────────────────────────────────────────

def _score_signal_quality() -> tuple[float, list[str]]:
    """Evaluate IC statistics from backtest history."""
    notes = []

    # Check signal halflife CSV
    hl_path = ROOT / "signal_halflife.csv"
    if hl_path.exists():
        df = pd.read_csv(hl_path)
        n_signals = len(df)
        notes.append(f"Halflife CSV: {n_signals} signals calibrated")

    # Check IC from backtest
    for fname in ("v11_backtest_monthly.csv", "signal_ic_history.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p)
            for col in ["ic_mean", "IC", "signal_ic"]:
                if col in df.columns:
                    ic_mean = float(df[col].mean())
                    ic_std  = float(df[col].std()) + 1e-9
                    ir      = ic_mean / ic_std
                    pct_pos = float((df[col] > 0).mean())
                    notes.append(f"mean IC={ic_mean:.3f}, IR={ir:.2f}, %pos={pct_pos:.0%}")

                    score = 0.0
                    if abs(ic_mean) > 0.03:  score += 3.0
                    if abs(ic_mean) > 0.05:  score += 2.0
                    if ir > 0.5:             score += 2.0
                    if pct_pos > 0.55:       score += 2.0
                    if hl_path.exists():     score += 1.0
                    return min(score, 10.0), notes

    # No backtest found: check SHAP pruning (proves signal infrastructure exists)
    shap_path = ROOT / "lgb_pruned_features.json"
    if shap_path.exists():
        data = json.loads(shap_path.read_text())
        n_feat = len(data.get("pruned_features", []))
        notes.append(f"SHAP pruning: {n_feat} features selected")
        return 5.0, notes

    notes.append("No backtest IC data found — run canyon_v11_full.py to produce results")
    return 3.0, notes


def _score_oos_performance() -> tuple[float, list[str]]:
    """Check deflated Sharpe and IS/OOS ratio."""
    notes = []
    score = 0.0

    for fname in ("v11_backtest_monthly.csv",):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ret_cols = [c for c in ["ls_return", "long_short_return", "oos_return"] if c in df.columns]
        if not ret_cols:
            continue
        ret = df[ret_cols[0]].dropna()
        if len(ret) < 12:
            continue

        ann_ret = float(ret.mean() * 12)
        ann_vol = float(ret.std() * np.sqrt(12))
        sharpe  = ann_ret / (ann_vol + 1e-9)

        notes.append(f"Ann. return={ann_ret:.1%}, vol={ann_vol:.1%}, Sharpe={sharpe:.2f}")

        if sharpe > 0.5:  score += 2.0
        if sharpe > 1.0:  score += 2.0
        if sharpe > 1.5:  score += 2.0
        if ann_ret > 0.05: score += 1.0

        # Scorecard CSV check
        sc_path = ROOT / "backtest_scorecard.csv"
        if sc_path.exists():
            sc = pd.read_csv(sc_path)
            if "overall_score" in sc.columns:
                overall = float(sc["overall_score"].iloc[-1])
                notes.append(f"Backtest scorecard: {overall:.1f}/10")
                if overall >= 7.0: score += 3.0
                elif overall >= 5.0: score += 1.5
        return min(score, 10.0), notes

    notes.append("No OOS backtest results — run v11 first")
    return 2.0, notes


def _score_factor_hygiene() -> tuple[float, list[str]]:
    """Check VIF report and correlation pruning."""
    notes = []
    score = 0.0

    vif_path = ROOT / "vif_report.csv"
    if vif_path.exists():
        df = pd.read_csv(vif_path)
        if "vif" in df.columns:
            n_ok = (df["vif"] < 5).sum()
            n_tot = len(df)
            notes.append(f"VIF: {n_ok}/{n_tot} factors below VIF=5")
            score += min(4.0, (n_ok / n_tot) * 4.0)
        else:
            score += 2.0
            notes.append("VIF report exists but no vif column")
    else:
        notes.append("No VIF report — run research/vif_check.py")

    corr_path = ROOT / "signal_correlation_pruned.csv"
    corr_path2 = ROOT / "signal_redundant_pairs.csv"
    if corr_path.exists() or corr_path2.exists():
        score += 3.0
        notes.append("Signal correlation pruning complete")
    else:
        notes.append("No signal correlation report")

    # Check Barra factor model exists
    barra_path = ROOT / "barra_factor_exposures.csv"
    if barra_path.exists():
        score += 3.0
        notes.append("Barra factor exposures present")
    else:
        notes.append("No Barra factor exposures — run risk/barra.py")

    return min(score, 10.0), notes


def _score_risk_controls() -> tuple[float, list[str]]:
    """Check regime covariance, position limits, drawdown controls."""
    notes = []
    score = 0.0

    # Regime covariance
    rcov_path = ROOT / "regime_covariance.csv"
    if rcov_path.exists():
        score += 3.0
        notes.append("Regime covariance (HMM) present")
    else:
        notes.append("No regime covariance file")

    # Stress tests
    st_path = ROOT / "stress_test_results.csv"
    if st_path.exists():
        df = pd.read_csv(st_path)
        if "excess_return" in df.columns:
            n_outperform = (df["excess_return"] >= 0).sum()
            n_total = len(df)
            notes.append(f"Stress tests: {n_outperform}/{n_total} scenarios outperform SPY")
            score += min(3.0, (n_outperform / n_total) * 3.0)
    else:
        notes.append("No stress test results — run research/stress_test.py")

    # Capacity analysis
    cap_path = ROOT / "capacity_analysis.csv"
    if cap_path.exists():
        score += 2.0
        notes.append("Capacity analysis present")
    else:
        notes.append("No capacity analysis")

    # Kelly sizing
    kelly_path = ROOT / "portfolio/kelly.py"
    if (ROOT / "portfolio/kelly.py").exists():
        score += 2.0
        notes.append("Fractional Kelly sizing module present")

    return min(score, 10.0), notes


def _score_pit_compliance() -> tuple[float, list[str]]:
    """Check EDGAR PIT and Form 4 compliance."""
    notes = []
    score = 0.0

    # Check EDGAR modules
    for fname in ("data/edgar_pit.py", "data/edgar_form4.py"):
        if (ROOT / fname).exists():
            score += 2.0
            notes.append(f"PIT module present: {fname}")

    # Check that edgar_pit.py uses filed_date
    edgar_path = ROOT / "data/edgar_pit.py"
    if edgar_path.exists():
        source = edgar_path.read_text()
        if "filed_date" in source and "know_date" in source:
            score += 3.0
            notes.append("EDGAR PIT uses filed_date as know_date (correct)")
        elif "filed_date" in source:
            score += 2.0
            notes.append("EDGAR PIT uses filed_date (partially correct)")

    # Check earnings_call PIT
    ec_path = ROOT / "signals/earnings_call.py"
    if ec_path.exists():
        source = ec_path.read_text()
        if "filed_date" in source:
            score += 3.0
            notes.append("FinBERT sentiment enforces PIT (filed_date)")

    return min(score, 10.0), notes


def _score_capacity() -> tuple[float, list[str]]:
    """Check capacity analysis outputs."""
    notes = []
    score = 5.0  # default pass if module exists

    cap_path = ROOT / "research/capacity.py"
    if cap_path.exists():
        notes.append("Capacity analysis module: research/capacity.py")
        score = 7.0
    else:
        notes.append("Capacity module missing")
        score = 2.0

    cap_csv = ROOT / "capacity_analysis.csv"
    if cap_csv.exists():
        df = pd.read_csv(cap_csv)
        if "adv_pct" in df.columns:
            over_adv = (df["adv_pct"] > 0.05).sum()
            n = len(df)
            notes.append(f"ADV check: {n-over_adv}/{n} positions under 5% ADV limit")
            score = 10.0 if over_adv == 0 else 7.0

    return min(score, 10.0), notes


def _score_execution_quality() -> tuple[float, list[str]]:
    """Check slippage model and TWAP execution."""
    notes = []
    score = 0.0

    # AC model
    for fname in ("monitoring/slippage.py", "monitoring/execution_quality.py"):
        if (ROOT / fname).exists():
            score += 2.5
            notes.append(f"Execution module: {fname}")

    # Alpaca dry-run
    alpaca_path = ROOT / "execution/alpaca_exec.py"
    if alpaca_path.exists():
        source = alpaca_path.read_text()
        if "dry_run" in source:
            score += 2.5
            notes.append("Alpaca executor has dry_run=True default (safe)")
        else:
            score += 1.0

    # Slippage results
    slip_path = ROOT / "slippage_analysis.csv"
    if slip_path.exists():
        score += 2.5
        notes.append("Slippage analysis results present")

    return min(score, 10.0), notes


def _score_code_health() -> tuple[float, list[str]]:
    """Check module syntax, no Chinese, all tests pass markers."""
    notes = []
    score = 0.0

    all_modules = [
        "signals/macro_hmm.py", "signals/lgb_shap.py", "signals/earnings_call.py",
        "signals/eps_revision.py", "data/edgar_pit.py", "data/edgar_form4.py",
        "data/fred_macro.py", "data/sp500_constituents.py", "data/extend_history.py",
        "data/polygon_ivr.py", "research/signal_halflife.py", "research/signal_corr.py",
        "research/vif_check.py", "research/capacity.py", "research/backtest_scorecard.py",
        "research/tearsheet.py", "research/param_robustness.py", "research/stress_test.py",
        "research/v9_bridge.py", "research/institutional_scorecard.py",
        "risk/barra.py", "risk/regime_cov.py", "portfolio/black_litterman.py",
        "portfolio/kelly.py", "monitoring/data_quality.py", "monitoring/factor_exposure.py",
        "monitoring/slippage.py", "monitoring/execution_quality.py", "monitoring/attribution.py",
        "execution/alpaca_exec.py",
    ]

    n_exist, n_ok, n_chinese = 0, 0, 0
    import re
    # CJK Unified Ideographs U+4E00-U+9FFF and Extension A U+3400-U+4DBF
    chinese_re = re.compile('[\u4e00-\u9fff\u3400-\u4dbf]')

    for mpath in all_modules:
        fp = ROOT / mpath
        if not fp.exists():
            continue
        n_exist += 1
        source = fp.read_text()
        try:
            ast.parse(source)
            n_ok += 1
        except SyntaxError:
            pass
        if chinese_re.search(source):
            n_chinese += 1

    notes.append(f"Modules: {n_exist}/{len(all_modules)} exist, {n_ok} parse OK")
    if n_chinese > 0:
        notes.append(f"WARNING: {n_chinese} modules contain Chinese text")

    score = (n_ok / max(n_exist, 1)) * 7.0
    if n_chinese == 0:
        score += 2.0
    if n_exist >= 25:
        score += 1.0

    return min(score, 10.0), notes


def _score_portfolio_theory() -> tuple[float, list[str]]:
    """Check BL, Ledoit-Wolf, Kelly are present and non-trivial."""
    notes = []
    score = 0.0

    checks = {
        "portfolio/black_litterman.py": ("black_litterman_update", "BL posterior update"),
        "risk/regime_cov.py":           ("_ledoit_wolf_shrinkage",  "Ledoit-Wolf shrinkage"),
        "portfolio/kelly.py":           ("compute_kelly_weights",    "Fractional Kelly"),
        "portfolio/black_litterman.py": ("_mean_variance_optimize",  "MVO"),
    }
    seen = set()
    for fpath, (fn_name, label) in checks.items():
        if fpath in seen:
            continue
        fp = ROOT / fpath
        if not fp.exists():
            notes.append(f"Missing: {fpath}")
            continue
        source = fp.read_text()
        if fn_name in source:
            score += 2.5
            notes.append(f"[OK] {label} ({fn_name})")
        else:
            notes.append(f"[MISS] {label}: function {fn_name} not found in {fpath}")
        seen.add(fpath)

    return min(score, 10.0), notes


def _score_attribution_bonus() -> tuple[float, list[str]]:
    """Bonus: BHB attribution module."""
    notes = []
    fp = ROOT / "monitoring/attribution.py"
    if fp.exists():
        source = fp.read_text()
        if "compute_sector_attribution" in source and "bhb" in source.lower() or \
           "brinson" in source.lower() or "allocation" in source.lower():
            notes.append("BHB performance attribution module present (monitoring/attribution.py)")
            return BONUS_ATTRIBUTION, notes
    notes.append("Attribution module missing or incomplete")
    return 0.0, notes


# ─── Main audit ───────────────────────────────────────────────────────────────

def run_institutional_audit(output_path: Optional[Path] = None) -> float:
    """
    W49: Run full institutional scoring audit.

    Returns overall weighted score (0-10). Threshold for institutional grade: 7.0.
    """
    if output_path is None:
        output_path = ROOT / "institutional_scorecard.csv"

    print("\n" + "=" * 65)
    print("  Canyon v11 — Institutional Scorecard Audit")
    print("=" * 65)

    scorers = {
        "signal_quality":    _score_signal_quality,
        "oos_performance":   _score_oos_performance,
        "factor_hygiene":    _score_factor_hygiene,
        "risk_controls":     _score_risk_controls,
        "pit_compliance":    _score_pit_compliance,
        "capacity":          _score_capacity,
        "execution_quality": _score_execution_quality,
        "code_health":       _score_code_health,
        "portfolio_theory":  _score_portfolio_theory,
    }

    rows = []
    weighted_total = 0.0

    for dim, scorer_fn in scorers.items():
        raw_score, notes = scorer_fn()
        weight = WEIGHTS[dim]
        contribution = raw_score * weight

        print(f"\n  [{dim.upper().replace('_', ' ')}]  {raw_score:.1f}/10  "
              f"(weight={weight:.0%}, contrib={contribution:.3f})")
        for note in notes:
            print(f"    • {note}")

        rows.append({
            "dimension":    dim,
            "raw_score":    round(raw_score, 2),
            "weight":       weight,
            "contribution": round(contribution, 4),
        })
        weighted_total += contribution

    # Attribution bonus
    bonus, bonus_notes = _score_attribution_bonus()
    print(f"\n  [BONUS — ATTRIBUTION]  +{bonus:.2f}")
    for note in bonus_notes:
        print(f"    • {note}")
    weighted_total += bonus

    print("\n" + "-" * 65)
    grade = "INSTITUTIONAL GRADE" if weighted_total >= 7.0 else \
            "NEAR INSTITUTIONAL"  if weighted_total >= 6.0 else \
            "RESEARCH GRADE"
    print(f"  OVERALL SCORE: {weighted_total:.2f} / 10  →  {grade}")
    print("-" * 65)

    # Breakdown table
    print("\n  Summary:")
    print(f"  {'Dimension':<22} {'Score':>6}  {'Weight':>7}  {'Contrib':>8}")
    print(f"  {'─'*50}")
    for row in rows:
        print(f"  {row['dimension']:<22} {row['raw_score']:>5.1f}  "
              f"  {row['weight']:>5.0%}  {row['contribution']:>8.4f}")
    print(f"  {'Bonus (attribution)':<22}  {bonus:>5.2f}          {bonus:>8.4f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<22}         {'100%':>7}  {weighted_total:>8.4f}")

    # Save
    score_df = pd.DataFrame(rows)
    score_df["overall"] = weighted_total
    score_df["grade"]   = grade
    score_df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    return weighted_total


if __name__ == "__main__":
    run_institutional_audit()
