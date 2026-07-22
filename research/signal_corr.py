"""
W25: Signal Correlation Pruning
================================
Identifies and removes redundant signals that add little incremental alpha
due to high pairwise correlation with existing signals.

Method (Grinold-Kahn, Active Portfolio Management Ch.6):
  1. Load all signal CSVs from the current pipeline
  2. Compute cross-sectional rank-correlation matrix for today's signals
  3. Flag pairs with |ρ| > threshold (default 0.70) as redundant
  4. For redundant pairs, keep the signal with higher IC (or higher IC²)
  5. Output correlation heatmap data + recommended pruning

This prevents the IC² weighting from over-counting alpha that is already
captured by another signal (the "double-counting" problem that inflates
apparent diversification).

Threshold guidance:
  |ρ| > 0.70: definitely prune (signals carry same information)
  |ρ| > 0.50: consider pruning (diminishing marginal IC)
  |ρ| < 0.30: uncorrelated signals, keep both (genuine diversification)

Outputs:
  signal_correlation_matrix.csv  — full NxN Spearman correlation matrix
  signal_redundancy_report.csv   — redundant pairs with recommendation

Usage:
    from research.signal_corr import run_correlation_pruning, load_correlation_matrix
    report = run_correlation_pruning()
    print(report[report["redundant"]])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent

# All signals in the step87 pipeline (name → csv_file, score_column)
SIGNAL_SOURCES: dict[str, tuple[str, str]] = {
    "regime_ml":   ("regime_ml_scores.csv",         "predicted_score"),
    "quality":     ("fundamental_quality_rank.csv",  "quality_score"),
    "revision":    ("earnings_revision_scores.csv",  "revision_score"),
    "surprise":    ("earnings_surprise_scores.csv",  "rank_sue"),
    "sentiment":   ("finbert_sentiment.csv",         "rank_sentiment"),
    "squeeze":     ("short_interest_scores.csv",     "rank_squeeze"),
    "insider":     ("insider_signal_scores.csv",     "rank_insider"),
    "options":     ("options_signals.csv",           "rank_options"),
    "ml_ensemble": ("ml_signal_scores.csv",          "ensemble_score"),
    "momentum":    ("momentum_scores.csv",           "momentum_score"),
    "accruals":    ("accrual_scores.csv",            "accrual_score"),
    "piotroski":   ("piotroski_scores.csv",          "piotroski_score"),
    "ins_cluster": ("insider_cluster_scores.csv",    "cluster_score_rank"),
    "earnings_call": ("earnings_call_sentiment.csv", "sentiment_score"),
    "eps_revision":  ("eps_revision_scores.csv",     "revision_score"),
}

# IC² weights from config (fallback static values if config.yaml missing)
_IC_SQUARED_FALLBACK: dict[str, float] = {
    "ml_ensemble": 0.370 ** 2,
    "surprise":    0.229 ** 2,
    "regime_ml":   0.223 ** 2,
    "momentum":    0.168 ** 2,
    "revision":    0.142 ** 2,
    "insider":     0.108 ** 2,
    "sentiment":   0.089 ** 2,
    "quality":     0.082 ** 2,
    "options":     0.076 ** 2,
    "squeeze":     0.071 ** 2,
    "accruals":    0.065 ** 2,
    "piotroski":   0.058 ** 2,
    "ins_cluster": 0.055 ** 2,
    "earnings_call": 0.050 ** 2,
    "eps_revision":  0.048 ** 2,
}

REDUNDANCY_THRESHOLD = 0.70   # |ρ| above this = redundant


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signal loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_signals() -> pd.DataFrame:
    """
    Load all available signals into a single DataFrame indexed by ticker.
    Missing signals are silently skipped.

    Returns: DataFrame with columns = signal_name, index = ticker.
    """
    loaded: dict[str, pd.Series] = {}

    for sig_name, (fname, col) in SIGNAL_SOURCES.items():
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if "ticker" not in df.columns or col not in df.columns:
                continue
            series = df.set_index("ticker")[col].dropna()
            if len(series) >= 30:
                loaded[sig_name] = series
        except Exception:
            continue

    if not loaded:
        return pd.DataFrame()

    # Align on common tickers, forward-fill with neutral rank (50)
    all_tickers = sorted(set.union(*[set(s.index) for s in loaded.values()]))
    result = pd.DataFrame(index=all_tickers)
    for sig_name, series in loaded.items():
        result[sig_name] = series.reindex(all_tickers).fillna(50.0)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def _rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each signal to rank percentile [0, 1] for Spearman calculation."""
    return df.rank(pct=True)


def compute_correlation_matrix(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rank correlation matrix for all signals.

    Uses full cross-section (all tickers) for maximum statistical power.
    Returns DataFrame: signal × signal correlation matrix.
    """
    if signals.empty:
        return pd.DataFrame()

    ranked = _rank_normalize(signals)
    corr_matrix, _ = spearmanr(ranked.values)

    if isinstance(corr_matrix, float):
        # Only 2 signals — spearmanr returns a scalar
        n = ranked.shape[1]
        mat = np.eye(n)
        mat[0, 1] = mat[1, 0] = corr_matrix
        corr_matrix = mat

    return pd.DataFrame(corr_matrix, index=signals.columns, columns=signals.columns)


def _load_ic2_weights() -> dict[str, float]:
    """Load IC² weights from config.yaml or use static fallback."""
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / "config.yaml"))
        ic_cfg = cfg.get("signals", {}).get("ic_fallback", {})
        ic2 = {k: float(v) ** 2 for k, v in ic_cfg.items() if k.startswith("sig_")}
        # Rename sig_X to X
        ic2 = {k.replace("sig_", ""): v for k, v in ic2.items()}
        if ic2:
            return ic2
    except Exception:
        pass
    return dict(_IC_SQUARED_FALLBACK)


def find_redundant_pairs(
    corr_matrix: pd.DataFrame,
    ic2_weights: dict[str, float],
    threshold: float = REDUNDANCY_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify signal pairs with |ρ| > threshold.

    For each redundant pair, the signal with the lower IC² is flagged for pruning.

    Returns DataFrame with:
        sig_a, sig_b, correlation, ic2_a, ic2_b, prune_signal, reason
    """
    sigs = list(corr_matrix.columns)
    rows = []

    for i, sig_a in enumerate(sigs):
        for j, sig_b in enumerate(sigs):
            if j <= i:
                continue
            rho = float(corr_matrix.loc[sig_a, sig_b])
            if abs(rho) >= threshold:
                ic2_a = ic2_weights.get(sig_a, 0.0)
                ic2_b = ic2_weights.get(sig_b, 0.0)
                prune = sig_b if ic2_a >= ic2_b else sig_a
                keep  = sig_a if prune == sig_b else sig_b
                rows.append({
                    "sig_a":       sig_a,
                    "sig_b":       sig_b,
                    "correlation": round(rho, 3),
                    "ic2_a":       round(ic2_a, 5),
                    "ic2_b":       round(ic2_b, 5),
                    "prune_signal": prune,
                    "keep_signal":  keep,
                    "reason":      f"|ρ|={abs(rho):.2f} > {threshold}, prune lower IC²",
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_correlation_pruning(
    threshold: float = REDUNDANCY_THRESHOLD,
    corr_output: Optional[Path] = None,
    report_output: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run full signal correlation analysis and generate pruning recommendations.

    Args:
        threshold:     Redundancy threshold (default 0.70).
        corr_output:   Where to save the correlation matrix CSV.
        report_output: Where to save the redundancy report CSV.

    Returns:
        DataFrame with redundant signal pairs and pruning recommendations.
    """
    if corr_output  is None: corr_output  = ROOT / "signal_correlation_matrix.csv"
    if report_output is None: report_output = ROOT / "signal_redundancy_report.csv"

    print("[SignalCorr] Loading signals...")
    signals = _load_signals()
    n_loaded = len(signals.columns)
    n_tickers = len(signals)
    print(f"  Loaded {n_loaded} signals × {n_tickers} tickers")

    if signals.empty or n_loaded < 2:
        print("  Not enough signals loaded — returning empty report")
        return pd.DataFrame()

    print("[SignalCorr] Computing Spearman correlation matrix...")
    corr_matrix = compute_correlation_matrix(signals)
    corr_matrix.to_csv(corr_output)
    print(f"  Saved {n_loaded}×{n_loaded} matrix → {corr_output}")

    # Print correlation summary
    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    print(f"  Correlation stats: mean={np.mean(upper_tri):.2f}, "
          f"max={np.max(upper_tri):.2f}, min={np.min(upper_tri):.2f}")

    # Find redundant pairs
    ic2_weights = _load_ic2_weights()
    redundant_df = find_redundant_pairs(corr_matrix, ic2_weights, threshold)

    if redundant_df.empty:
        print(f"  No redundant signal pairs found (threshold={threshold:.2f})")
    else:
        print(f"\n  Found {len(redundant_df)} redundant pair(s) |ρ| > {threshold:.2f}:")
        for _, row in redundant_df.iterrows():
            print(f"    {row['sig_a']:15s} × {row['sig_b']:15s}  ρ={row['correlation']:+.2f}  "
                  f"→ prune {row['prune_signal']}")

    redundant_df.to_csv(report_output, index=False)
    print(f"\n  Saved redundancy report → {report_output}")

    # Print correlation heatmap (text form)
    print("\n  Signal Correlation Heatmap:")
    pd.set_option("display.float_format", "{:+.2f}".format)
    print(corr_matrix.round(2).to_string())
    pd.reset_option("display.float_format")

    return redundant_df


def load_correlation_matrix() -> pd.DataFrame:
    """Load precomputed correlation matrix from CSV."""
    p = ROOT / "signal_correlation_matrix.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


if __name__ == "__main__":
    print("W25: Signal Correlation Pruning")
    print("=" * 50)
    report = run_correlation_pruning()
    if not report.empty:
        print(f"\nPruning recommendations:")
        print(report[["sig_a", "sig_b", "correlation", "prune_signal"]].to_string())
    else:
        print("\nNo redundant signals detected — all signals provide incremental value.")
