#!/usr/bin/env python3
"""
Canyon — step_factor_correlation.py
=====================================
Compute factor correlation matrix and VIF to identify redundant factors.

Factors loaded
--------------
  momentum_scores.csv         -> momentum_score
  fundamental_quality_rank.csv -> quality_score
  earnings_revision_scores.csv -> revision_score
  short_interest_scores.csv   -> rank_squeeze
  accrual_scores.csv          -> accrual_score
  piotroski_scores.csv        -> piotroski_score
  alpha_scores.csv            -> alpha_score

Methodology
-----------
1. Merge all factor files on 'ticker'.
2. Compute Spearman correlation matrix.
3. Flag pairs with |corr| > 0.60.
4. Compute VIF for each factor (OLS R² method if statsmodels unavailable).

Saves:
  factor_correlation.csv     (ticker x factor matrix)
  factor_corr_matrix.json    (correlation dict + high_corr_pairs + vif_scores)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

HIGH_CORR_THRESHOLD = 0.60

def log(msg: str) -> None:
    print(f"  {msg}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── factor file definitions ───────────────────────────────────────────────────

FACTOR_FILES = [
    ("momentum_scores.csv",          "momentum_score",  "Momentum"),
    ("fundamental_quality_rank.csv", "quality_score",   "Quality"),
    ("earnings_revision_scores.csv", "revision_score",  "Revision"),
    ("short_interest_scores.csv",    "rank_squeeze",    "Squeeze"),
    ("accrual_scores.csv",           "accrual_score",   "Accruals"),
    ("piotroski_scores.csv",         "piotroski_score", "Piotroski"),
    ("alpha_scores.csv",             "alpha_score",     "Alpha"),
]

# ── load and merge factors ────────────────────────────────────────────────────

section("1. Loading factor files")

merged = None
loaded_factors: list[str] = []
skipped_factors: list[str] = []

for fname, col, label in FACTOR_FILES:
    fpath = ROOT / fname
    if not fpath.exists():
        log(f"SKIP: {fname} not found.")
        skipped_factors.append(label)
        continue

    try:
        df = pd.read_csv(fpath)

        if "ticker" not in df.columns:
            log(f"SKIP: {fname} has no 'ticker' column.")
            skipped_factors.append(label)
            continue

        if col not in df.columns:
            # Try to find a similar column
            candidates = [c for c in df.columns if label.lower() in c.lower() or "score" in c.lower() or "rank" in c.lower()]
            if candidates:
                col = candidates[0]
                log(f"  {fname}: '{col}' not found, using '{col}' instead.")
            else:
                log(f"SKIP: {fname} has no column '{col}' or similar.")
                skipped_factors.append(label)
                continue

        sub = df[["ticker", col]].rename(columns={col: label})
        sub = sub.dropna(subset=["ticker", label])
        sub["ticker"] = sub["ticker"].astype(str).str.upper().str.strip()

        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on="ticker", how="outer")

        loaded_factors.append(label)
        log(f"  Loaded {fname}: {len(sub):,} rows, factor '{col}' -> '{label}'")

    except Exception as e:
        log(f"  ERROR loading {fname}: {e}")
        skipped_factors.append(label)

if merged is None or len(loaded_factors) < 2:
    print("  FATAL: Need at least 2 factors to compute correlation.")
    raise SystemExit(1)

log(f"\nLoaded factors  : {loaded_factors}")
log(f"Skipped factors : {skipped_factors}")
log(f"Merged shape    : {merged.shape}")
log(f"Rows with all factors: {merged.dropna().shape[0]}")

# ── save factor matrix ────────────────────────────────────────────────────────

section("2. Saving factor_correlation.csv")

factor_matrix = merged.set_index("ticker")[loaded_factors]
factor_matrix.to_csv(ROOT / "factor_correlation.csv")
log(f"Saved factor_correlation.csv ({factor_matrix.shape})")

# ── compute spearman correlation matrix ──────────────────────────────────────

section("3. Computing Spearman correlation matrix")

# Use pairwise complete observations
factor_clean = factor_matrix.dropna()
log(f"Complete rows for correlation: {len(factor_clean)}")

def pairwise_spearman(factor_matrix: pd.DataFrame, factors: list) -> pd.DataFrame:
    """Compute pairwise Spearman correlation with safe scalar extraction."""
    n = len(factors)
    corr_arr = np.full((n, n), np.nan)
    for i, fa in enumerate(factors):
        corr_arr[i, i] = 1.0
        for j, fb in enumerate(factors):
            if i == j:
                continue
            both = factor_matrix[[fa, fb]].dropna()
            if len(both) < 5:
                continue
            try:
                result = spearmanr(both[fa].values, both[fb].values)
                # scipy returns SpearmanrResult; extract statistic
                c = float(result.statistic if hasattr(result, "statistic") else result[0])
                corr_arr[i, j] = c
            except Exception:
                pass
    return pd.DataFrame(corr_arr, index=factors, columns=factors)

if len(factor_clean) < 20:
    log("WARNING: Very few complete rows. Using pairwise correlations.")

corr_df = pairwise_spearman(factor_matrix, loaded_factors)

log("Correlation matrix:")
print()
print(corr_df.round(3).to_string())
print()

# ── flag high correlation pairs ───────────────────────────────────────────────

section("4. Flagging high correlation pairs (|corr| > 0.60)")

high_corr_pairs: list[dict] = []
for i, fa in enumerate(loaded_factors):
    for j, fb in enumerate(loaded_factors):
        if i >= j:
            continue
        c = corr_df.loc[fa, fb]
        if not np.isnan(c) and abs(c) > HIGH_CORR_THRESHOLD:
            high_corr_pairs.append({
                "factor_a"  : fa,
                "factor_b"  : fb,
                "spearman_r": round(float(c), 4),
                "abs_corr"  : round(abs(float(c)), 4),
                "flag"      : "HIGH CORRELATION",
            })

if high_corr_pairs:
    log(f"Found {len(high_corr_pairs)} high-correlation pair(s):")
    for p in sorted(high_corr_pairs, key=lambda x: -x["abs_corr"]):
        log(f"  {p['factor_a']} <-> {p['factor_b']}: r={p['spearman_r']:.3f}")
else:
    log("No pairs exceed |corr| > 0.60 threshold.")

# ── compute VIF ───────────────────────────────────────────────────────────────

section("5. Computing Variance Inflation Factor (VIF)")

vif_scores: dict[str, float] = {}

def compute_vif_r2(factor_matrix: pd.DataFrame, factors: list) -> dict:
    """Compute VIF using pairwise OLS R² for each factor as dependent variable."""
    results = {}
    for fname_vif in factors:
        # For this factor, use all tickers where this factor and at least 2 others are present
        others = [f for f in factors if f != fname_vif]
        if not others:
            results[fname_vif] = 1.0
            continue
        sub = factor_matrix[[fname_vif] + others].dropna(subset=[fname_vif])
        sub = sub.dropna(thresh=3)  # need at least 3 columns (including target)
        if len(sub) < 10:
            results[fname_vif] = float("nan")
            continue
        y = sub[fname_vif].rank(pct=True).values
        # Fill others with median for missing
        X_other = sub[others].rank(pct=True).fillna(0.5).values
        if X_other.shape[1] == 0:
            results[fname_vif] = 1.0
            continue
        X_mat = np.column_stack([np.ones(len(y)), X_other])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            y_pred = X_mat @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = max(0.0, min(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 0.9999))
            vif = 1.0 / (1.0 - r2)
            results[fname_vif] = round(float(vif), 4)
        except Exception:
            results[fname_vif] = float("nan")
    return results

# Try statsmodels on the full-complete matrix, fall back to pairwise R² approach
vif_computed = False
if len(factor_clean) >= 10:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        factor_ranked = factor_clean[loaded_factors].rank(pct=True)
        X = factor_ranked.values.astype(float)
        for i, fname_vif in enumerate(loaded_factors):
            try:
                vif = variance_inflation_factor(X, i)
                if np.isfinite(vif):
                    vif_scores[fname_vif] = round(float(vif), 4)
                else:
                    vif_scores[fname_vif] = float("nan")
            except Exception:
                vif_scores[fname_vif] = float("nan")
        if any(np.isfinite(v) for v in vif_scores.values() if v is not None):
            vif_computed = True
            log("VIF computed via statsmodels.")
    except Exception:
        pass

if not vif_computed:
    log("Using pairwise R² method for VIF (sparse factor coverage).")
    vif_scores = compute_vif_r2(factor_matrix, loaded_factors)

log("VIF scores:")
for k, v in sorted(vif_scores.items(), key=lambda x: -(x[1] or 0)):
    flag = " <-- HIGH MULTICOLLINEARITY" if (v is not None and not np.isnan(v) and v > 5) else ""
    log(f"  {k:20s}: VIF = {v:.2f}{flag}")

# ── identify most redundant factors ──────────────────────────────────────────

section("6. Redundancy analysis")

redundant_factors: list[str] = []
for fname_vif, vif in vif_scores.items():
    if vif is not None and not np.isnan(vif) and vif > 5:
        redundant_factors.append(fname_vif)

if redundant_factors:
    log(f"Most redundant (VIF > 5): {redundant_factors}")
    log("Consider removing or orthogonalizing these factors.")
else:
    log("No factor has VIF > 5. Multicollinearity is acceptable.")

# Also flag from high-correlation pairs
high_corr_factors = set()
for p in high_corr_pairs:
    high_corr_factors.update([p["factor_a"], p["factor_b"]])

if high_corr_factors:
    log(f"Factors appearing in high-corr pairs: {sorted(high_corr_factors)}")

# ── save json report ──────────────────────────────────────────────────────────

section("7. Saving factor_corr_matrix.json")

corr_dict = {
    factor: {
        other: round(float(corr_df.loc[factor, other]), 6)
        if not np.isnan(corr_df.loc[factor, other]) else None
        for other in loaded_factors
    }
    for factor in loaded_factors
}

report = {
    "generated_at"      : pd.Timestamp.now().isoformat(),
    "loaded_factors"    : loaded_factors,
    "skipped_factors"   : skipped_factors,
    "n_tickers"         : int(len(factor_matrix)),
    "n_complete_rows"   : int(len(factor_clean)),
    "correlation_matrix": corr_dict,
    "high_corr_pairs"   : high_corr_pairs,
    "high_corr_threshold": HIGH_CORR_THRESHOLD,
    "vif_scores"        : vif_scores,
    "redundant_factors_vif_gt_5": redundant_factors,
    "high_corr_factors" : sorted(high_corr_factors),
    "methodology"       : (
        "Spearman rank correlation on pairwise-complete observations. "
        "VIF computed via OLS R² method (or statsmodels if available). "
        f"High correlation threshold: |r| > {HIGH_CORR_THRESHOLD}."
    ),
}

out_path = ROOT / "factor_corr_matrix.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
log(f"Saved to {out_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  FACTOR CORRELATION SUMMARY")
print("="*60)
print(f"  Factors loaded  : {', '.join(loaded_factors)}")
print(f"  Tickers merged  : {len(factor_matrix)}")
print(f"  High-corr pairs : {len(high_corr_pairs)}")
for p in sorted(high_corr_pairs, key=lambda x: -x["abs_corr"]):
    print(f"    {p['factor_a']} <-> {p['factor_b']}: r={p['spearman_r']:.3f}")
print(f"  Redundant (VIF>5): {redundant_factors if redundant_factors else 'None'}")
print()
print("  Top VIF scores:")
for k, v in sorted(vif_scores.items(), key=lambda x: -(x[1] or 0)):
    print(f"    {k:20s}: {v:.2f}")
print()
print("  => factor_correlation.csv and factor_corr_matrix.json saved.")
