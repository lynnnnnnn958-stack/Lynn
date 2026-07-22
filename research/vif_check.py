"""
W31: Factor Orthogonality VIF Check
=====================================
Variance Inflation Factor (VIF) measures multicollinearity among Barra factors.

VIF_k = 1 / (1 - R²_k)  where R²_k = R-squared of regressing factor k on all other factors.

Interpretation:
  VIF < 5:  Acceptable (factors are reasonably orthogonal)
  VIF 5-10: Moderate multicollinearity (consider orthogonalising)
  VIF > 10: Severe multicollinearity (factor is nearly redundant)

Institutional context:
  Barra USE3 factors are designed to be orthogonal; high VIF indicates
  the factor exposures are not well-estimated or the factor model is
  misspecified for this universe.

Outputs:
  vif_report.csv — factor × VIF score

Usage:
    from research.vif_check import run_vif_check
    report = run_vif_check()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent


def _compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute VIF for each column of X.
    VIF_k = 1 / (1 - R²_k).
    """
    vifs = {}
    cols = X.columns.tolist()
    for k, col in enumerate(cols):
        others = [c for c in cols if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        X_other = X[others].values
        y_k     = X[col].values

        # Handle NaN
        mask = ~(np.isnan(X_other).any(axis=1) | np.isnan(y_k))
        if mask.sum() < 10:
            vifs[col] = np.nan
            continue

        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_other[mask], y_k[mask])
            y_pred = reg.predict(X_other[mask])
            ss_res = np.sum((y_k[mask] - y_pred) ** 2)
            ss_tot = np.sum((y_k[mask] - y_k[mask].mean()) ** 2)
            r2     = 1 - ss_res / (ss_tot + 1e-12)
            r2     = min(r2, 0.9999)
            vifs[col] = 1.0 / (1.0 - r2)
        except Exception:
            vifs[col] = np.nan

    return pd.Series(vifs)


def run_vif_check(
    factor_exposures_path: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Run VIF check on Barra factor exposure matrix.

    Returns DataFrame: factor, VIF, interpretation
    """
    if factor_exposures_path is None:
        factor_exposures_path = ROOT / "barra_factor_exposures.csv"
    if output_path is None:
        output_path = ROOT / "vif_report.csv"

    p = Path(factor_exposures_path)
    if not p.exists():
        print(f"  [VIF] Factor exposures not found: {p}")
        print("  Run: python risk/barra.py to generate factor exposures")
        return pd.DataFrame()

    B = pd.read_csv(p, index_col=0).dropna()

    if B.empty or len(B) < 30:
        print("  [VIF] Insufficient data for VIF analysis")
        return pd.DataFrame()

    print(f"  [VIF] Computing VIF for {len(B.columns)} factors on {len(B)} stocks")
    vifs = _compute_vif(B)

    def interpret(v):
        if np.isnan(v):    return "N/A"
        if v < 5:          return "OK (orthogonal)"
        if v < 10:         return "MODERATE (watch)"
        return             "HIGH (multicollinear)"

    report = pd.DataFrame({
        "factor":        vifs.index,
        "VIF":           vifs.values.round(2),
        "interpretation": [interpret(v) for v in vifs.values],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    report.to_csv(output_path, index=False)

    print(f"\n  VIF Report:")
    print(f"  {'Factor':<18} {'VIF':>7}  Interpretation")
    print(f"  {'─'*50}")
    for _, row in report.iterrows():
        flag = " !" if float(row["VIF"]) > 5 else "  "
        print(f"  {row['factor']:<18} {row['VIF']:>7.2f}{flag}  {row['interpretation']}")

    print(f"\n  Saved → {output_path}")
    return report


if __name__ == "__main__":
    print("W31: Factor Orthogonality VIF Check")
    print("=" * 40)
    report = run_vif_check()
    if not report.empty:
        high_vif = report[report["VIF"] > 5]
        if not high_vif.empty:
            print(f"\n  ⚠  {len(high_vif)} factors with VIF > 5 (consider orthogonalizing)")
        else:
            print("\n  ✓  All factors have acceptable VIF < 5")
