#!/usr/bin/env python3
"""
canyon_final_v9_step104_signal_correlation.py
=============================================
Canyon v9 — Step 104: Signal Correlation & Diversification Monitor

Monitor whether the 9 alpha signals are providing independent information
or are all telling the same story (false diversification). Compute pairwise
signal correlations and effective signal count.

Outputs:
  signal_correlation.csv       — long-format: signal_a, signal_b, correlation, is_redundant
  signal_correlation_report.md — full diagnostic report

Usage:
  python3 canyon_final_v9_step104_signal_correlation.py
  python3 canyon_final_v9_step104_signal_correlation.py --threshold 0.60
  python3 canyon_final_v9_step104_signal_correlation.py --time-series
  python3 canyon_final_v9_step104_signal_correlation.py --threshold 0.70 --time-series
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Signal columns ─────────────────────────────────────────────────────────────
SIGNAL_COLS: list[str] = [
    # v9 L3/L4 signals (all 17 in IC_WEIGHTS as of v9.91)
    "ml_score",
    "factor_score",
    "smart_money",
    "accruals",
    "squeeze",
    "sig_fundamental",
    "finbert",
    "sig_10k",
    "sig_8k",
    "sig_pead",
    "sig_insider",
    "sig_revision",
    "sig_options",
    "alt_trends",
    "alt_wiki",
    "sig_cross_asset",
    "sig_crowd",
    # Legacy column aliases (backwards compat with old alpha_scores.csv)
    "sig_regime_ml",
    "sig_quality",
    "sig_surprise",
    "sig_sentiment",
    "sig_squeeze",
    "sig_ml_ensemble",
]

SIGNAL_LABELS: dict[str, str] = {
    "ml_score":        "ML Ensemble",
    "factor_score":    "Factor Score",
    "smart_money":     "Smart Money",
    "accruals":        "Accruals",
    "squeeze":         "Short Squeeze",
    "sig_fundamental": "XBRL Fundamentals",
    "finbert":         "FinBERT News",
    "sig_10k":         "10-K MD&A NLP",
    "sig_8k":          "8-K Earnings NLP",
    "sig_pead":        "PEAD / SUE Drift",
    "sig_insider":     "Insider Cluster",
    "sig_revision":    "Analyst Revision",
    "sig_options":     "Options IV/Flow",
    "alt_trends":      "Google Trends",
    "alt_wiki":        "Wikipedia Views",
    "sig_cross_asset": "Cross-Asset Momentum",
    "sig_crowd":       "13F Crowding",
    # Legacy aliases
    "sig_regime_ml":   "Regime ML (legacy)",
    "sig_quality":     "Quality (legacy)",
    "sig_surprise":    "Surprise (legacy)",
    "sig_sentiment":   "Sentiment (legacy)",
    "sig_squeeze":     "Squeeze (legacy)",
    "sig_ml_ensemble": "ML Ensemble (legacy)",
}

# ── I/O paths ──────────────────────────────────────────────────────────────────
ALPHA_SCORES_CSV  = ROOT / "alpha_scores.csv"
ALPHA_HISTORY_CSV = ROOT / "alpha_score_history.csv"
SCORE_HISTORY_CSV = ROOT / "score_history.csv"

OUT_CSV    = ROOT / "signal_correlation.csv"
OUT_REPORT = ROOT / "signal_correlation_report.md"

# ── Interpretation thresholds ──────────────────────────────────────────────────
DIV_SCORE_GOOD     = 70.0   # > 70 → Good diversity
DIV_SCORE_MODERATE = 50.0   # 50–70 → Moderate redundancy
# < 50 → High redundancy

REDUNDANCY_THRESHOLD_DEFAULT = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# [1/3] Core computation functions
# ─────────────────────────────────────────────────────────────────────────────

def _available_sig_cols(df: pd.DataFrame, sig_cols: list[str]) -> list[str]:
    """Return subset of sig_cols that exist and have enough non-NaN values."""
    present = [c for c in sig_cols if c in df.columns]
    if not present:
        return []
    # Keep only columns with at least 10 valid values
    valid = [
        c for c in present
        if df[c].dropna().shape[0] >= 10
    ]
    return valid


def pearson_corr_matrix(df: pd.DataFrame, sig_cols: list[str]) -> pd.DataFrame:
    """
    Compute N×N Pearson correlation matrix for signal columns.
    Uses pandas built-in which is pure numpy under the hood.
    Handles NaN via pairwise complete observations.
    """
    cols = _available_sig_cols(df, sig_cols)
    if not cols:
        return pd.DataFrame()

    sub = df[cols].copy()
    # Convert to numeric, coerce any non-numeric to NaN
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    corr = sub.corr(method="pearson", min_periods=10)
    return corr


def effective_signal_count(corr_matrix: pd.DataFrame) -> float:
    """
    Effective N = 1 / sum(w_i^2) where w_i = eigenvalue_i / sum(eigenvalues).
    Computed via numpy eigendecomposition of correlation matrix.

    Range: 1.0 (all identical) to N (fully independent).
    Example: 9 signals, effective N = 3.2 → signals are highly redundant.
    """
    if corr_matrix.empty:
        return 0.0

    mat = corr_matrix.values.astype(float)
    # Replace NaN diagonal/off-diagonal with 0/1 for stability
    np.fill_diagonal(mat, 1.0)
    mat = np.where(np.isnan(mat), 0.0, mat)

    # Eigenvalues (symmetric matrix → use eigh for numerical stability)
    eigenvalues = np.linalg.eigvalsh(mat)
    # Clip negatives from floating point noise
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0

    weights = eigenvalues / total
    # Avoid divide-by-zero in degenerate case
    sum_sq = float(np.sum(weights ** 2))
    if sum_sq <= 0:
        return float(len(corr_matrix))

    return 1.0 / sum_sq


def unique_information_ratio(
    corr_matrix: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    For each signal, estimate its unique contribution:
      UIR_i = 1 - max(|corr(i,j)|)  for j != i

    UIR close to 0 → very correlated with another signal (redundant).
    UIR close to 1 → unique signal.

    weights is accepted for API compatibility but not used in the computation
    (the formula is structural, not weight-dependent).
    """
    if corr_matrix.empty:
        return {}

    result: dict[str, float] = {}
    cols = list(corr_matrix.columns)

    for sig in cols:
        row = corr_matrix.loc[sig].copy()
        # Drop self-correlation
        off_diag = row.drop(labels=[sig], errors="ignore")
        max_abs_corr = float(off_diag.abs().max())
        if np.isnan(max_abs_corr):
            max_abs_corr = 0.0
        result[sig] = round(1.0 - max_abs_corr, 4)

    return result


def signal_redundancy_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = REDUNDANCY_THRESHOLD_DEFAULT,
) -> list[tuple[str, str, float]]:
    """
    Return list of (signal_a, signal_b, correlation) where |corr| > threshold.
    Each pair is reported once (upper triangle only).
    """
    if corr_matrix.empty:
        return []

    pairs: list[tuple[str, str, float]] = []
    cols = list(corr_matrix.columns)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            val = corr_matrix.loc[a, b]
            if pd.isna(val):
                continue
            if abs(val) > threshold:
                pairs.append((a, b, round(float(val), 4)))

    # Sort by descending |correlation|
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def diversification_score(eff_n: float, total_n: int) -> float:
    """
    Score = (eff_n / total_n) * 100.
    100 = fully independent signals.
    < 50 = high redundancy.
    """
    if total_n <= 0:
        return 0.0
    return round((eff_n / total_n) * 100.0, 2)


def _interpret_div_score(score: float) -> str:
    """Return interpretation string for diversification score."""
    if score > DIV_SCORE_GOOD:
        return "Good signal diversity"
    if score >= DIV_SCORE_MODERATE:
        return "Moderate redundancy — consider reviewing correlated pairs"
    return "High redundancy — multiple signals may be measuring the same thing"


def _corr_to_long(
    corr_matrix: pd.DataFrame,
    threshold: float,
    label: str = "cross_sectional",
) -> pd.DataFrame:
    """Convert a square correlation matrix to long format."""
    cols = list(corr_matrix.columns)
    rows: list[dict] = []

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            val = corr_matrix.loc[a, b]
            if pd.isna(val):
                continue
            rows.append({
                "signal_a":     a,
                "signal_b":     b,
                "correlation":  round(float(val), 4),
                "abs_corr":     round(abs(float(val)), 4),
                "is_redundant": abs(float(val)) > threshold,
                "source":       label,
            })

    if not rows:
        return pd.DataFrame(columns=["signal_a", "signal_b", "correlation",
                                      "abs_corr", "is_redundant", "source"])
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _format_corr_table(corr_matrix: pd.DataFrame) -> str:
    """Format correlation matrix as a markdown table with short labels."""
    if corr_matrix.empty:
        return "_No data available._"

    cols = list(corr_matrix.columns)
    short = {c: SIGNAL_LABELS.get(c, c) for c in cols}
    headers = [short[c] for c in cols]

    # Header row
    lines = ["| Signal | " + " | ".join(headers) + " |"]
    lines.append("|---|" + "---|" * len(headers))

    for row_col in cols:
        row_label = short[row_col]
        cells = []
        for col in cols:
            if row_col == col:
                cells.append("1.00")
            else:
                val = corr_matrix.loc[row_col, col]
                if pd.isna(val):
                    cells.append("—")
                else:
                    # Bold high correlations
                    formatted = f"{val:+.2f}"
                    if abs(val) >= 0.70:
                        formatted = f"**{formatted}**"
                    cells.append(formatted)
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# [2/3] Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_snapshot() -> Optional[pd.DataFrame]:
    """
    Load per-ticker cross-sectional signal snapshot.

    Priority:
      1. signals_snapshot_today.csv — step500 saves all 17 z-scored signals daily
      2. alpha_scores.csv           — older 9-signal schema (legacy fallback)
    """
    # Priority 1: step500 snapshot (has all 17 current signals as z-scores)
    snap_path = ROOT / "signals_snapshot_today.csv"
    if snap_path.exists():
        try:
            df = pd.read_csv(snap_path)
            if "ticker" in df.columns:
                sig_present = [c for c in SIGNAL_COLS if c in df.columns]
                if sig_present:
                    print(f"  [OK] signals_snapshot_today.csv — {len(df)} tickers, "
                          f"{len(sig_present)}/{len(SIGNAL_COLS)} signal columns")
                    return df
        except Exception as exc:
            print(f"  [WARN] signals_snapshot_today.csv read error: {exc}")

    # Priority 2: legacy alpha_scores.csv
    if not ALPHA_SCORES_CSV.exists():
        print(f"  [SKIP] alpha_scores.csv not found ({ALPHA_SCORES_CSV})")
        return None
    try:
        df = pd.read_csv(ALPHA_SCORES_CSV)
        if df.empty:
            print("  [SKIP] alpha_scores.csv is empty")
            return None
        sig_present = [c for c in SIGNAL_COLS if c in df.columns]
        if not sig_present:
            print("  [WARN] alpha_scores.csv has no matching signal columns")
            return None
        print(f"  [OK] alpha_scores.csv (legacy) — {len(df)} tickers, "
              f"{len(sig_present)} signal columns present")
        return df
    except Exception as exc:
        print(f"  [WARN] Could not read alpha_scores.csv: {exc}")
        return None


def load_history() -> Optional[pd.DataFrame]:
    """
    Load time-series signal history.
    Prefer alpha_score_history.csv; fall back to score_history.csv.
    alpha_score_history.csv only has (date, ticker, alpha_score) — no sig_* cols.
    score_history.csv has sig-like rank columns.
    Return None if neither has sig_* columns.
    """
    # Try alpha_score_history.csv first
    for path, label in [
        (ALPHA_HISTORY_CSV, "alpha_score_history.csv"),
        (SCORE_HISTORY_CSV, "score_history.csv"),
    ]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            sig_present = [c for c in SIGNAL_COLS if c in df.columns]
            if sig_present and "date" in df.columns:
                print(f"  [OK] {label} — {len(df)} rows, "
                      f"{df['date'].nunique() if 'date' in df.columns else '?'} dates, "
                      f"{len(sig_present)} sig_* columns")
                return df
            else:
                # Try to map rank columns from score_history to canonical names
                col_map: dict[str, str] = {
                    "rank_sue":       "sig_surprise",
                    "rank_revision":  "sig_revision",
                    "quality_score":  "sig_quality",
                    "rank_sentiment": "sig_sentiment",
                    "rank_options":   "sig_options",
                    "rank_squeeze":   "sig_squeeze",
                }
                mapped = {src: dst for src, dst in col_map.items()
                          if src in df.columns}
                if mapped and "date" in df.columns:
                    df = df.rename(columns=mapped)
                    print(f"  [OK] {label} (mapped columns) — {len(df)} rows, "
                          f"{df['date'].nunique()} dates")
                    return df
                print(f"  [SKIP] {label} has no sig_* columns and no mappable rank columns")
        except Exception as exc:
            print(f"  [WARN] Could not read {label}: {exc}")

    return None


def compute_timeseries_corr(history: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute time-series correlations: for each date, take cross-sectional
    mean of each signal, producing a (n_dates × n_signals) DataFrame.
    Then compute Pearson correlation across dates.
    This captures how signal levels move together over time.
    """
    sig_present = [c for c in SIGNAL_COLS if c in history.columns]
    if not sig_present or "date" not in history.columns:
        return None

    print(f"  Computing time-series correlation across "
          f"{history['date'].nunique()} dates ...")

    # Daily cross-sectional mean per signal
    daily = (
        history
        .groupby("date")[sig_present]
        .mean()
        .sort_index()
    )

    if len(daily) < 5:
        print("  [WARN] Fewer than 5 dates — time-series correlation skipped")
        return None

    # Convert to numeric
    for c in sig_present:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    corr = daily.corr(method="pearson", min_periods=5)
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# [3/3] Report writer and main runner
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(
    snapshot_corr:    Optional[pd.DataFrame],
    timeseries_corr:  Optional[pd.DataFrame],
    eff_n_xs:         float,
    eff_n_ts:         Optional[float],
    uir:              dict[str, float],
    redundant_pairs:  list[tuple[str, str, float]],
    div_score_xs:     float,
    div_score_ts:     Optional[float],
    total_n:          int,
    threshold:        float,
    n_tickers:        int,
    run_ts:           bool,
) -> None:
    """Write the full markdown diagnostic report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    interp_xs = _interpret_div_score(div_score_xs)

    lines: list[str] = []
    lines.append("# Canyon v9 — Signal Correlation & Diversification Report")
    lines.append(f"\n_Generated: {ts}_")
    lines.append(f"\n---\n")

    # ── Overview ──────────────────────────────────────────────────────────────
    lines.append("## Overview")
    lines.append(f"\n- **Signals monitored:** {total_n}")
    lines.append(f"- **Universe (tickers):** {n_tickers}")
    lines.append(f"- **Redundancy threshold (|r| >):** {threshold:.2f}")
    lines.append(f"- **Redundant pairs found:** {len(redundant_pairs)}")
    lines.append("")

    # ── Cross-sectional matrix ────────────────────────────────────────────────
    lines.append("## Cross-Sectional Correlation Matrix")
    lines.append(f"\n_(Pearson r across {n_tickers} tickers, current snapshot)_\n")
    if snapshot_corr is not None and not snapshot_corr.empty:
        lines.append(_format_corr_table(snapshot_corr))
    else:
        lines.append("_Data not available._")
    lines.append("")

    # ── Effective signal count ────────────────────────────────────────────────
    lines.append("## Effective Signal Count (Eigenvalue Decomposition)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total signals | {total_n} |")
    lines.append(f"| Effective N (cross-sectional) | **{eff_n_xs:.2f}** |")
    if eff_n_ts is not None:
        lines.append(f"| Effective N (time-series) | **{eff_n_ts:.2f}** |")
    lines.append(f"| Diversification score (cross-sectional) | **{div_score_xs:.1f} / 100** |")
    if div_score_ts is not None:
        lines.append(f"| Diversification score (time-series) | **{div_score_ts:.1f} / 100** |")
    lines.append("")
    lines.append(f"> **Interpretation:** {interp_xs}")
    lines.append("")
    lines.append(
        "_Effective N is computed from eigenvalue shares of the correlation matrix._  \n"
        "_N = 9 means all signals are fully independent. N = 1 means all signals are identical._"
    )
    lines.append("")

    # ── Diversification score bar ─────────────────────────────────────────────
    lines.append("## Diversification Score")
    lines.append("")
    filled = int(round(div_score_xs / 10))
    bar = "█" * filled + "░" * (10 - filled)
    lines.append(f"`[{bar}] {div_score_xs:.1f} / 100`")
    lines.append("")
    lines.append("| Range | Interpretation |")
    lines.append("|-------|----------------|")
    lines.append("| > 70 | Good signal diversity |")
    lines.append("| 50 – 70 | Moderate redundancy — consider reviewing correlated pairs |")
    lines.append("| < 50 | High redundancy — multiple signals may be measuring the same thing |")
    lines.append("")

    # ── Redundant pairs ───────────────────────────────────────────────────────
    lines.append("## Redundant Signal Pairs")
    lines.append(f"\n_(|r| > {threshold:.2f})_\n")
    if redundant_pairs:
        lines.append("| Signal A | Signal B | Correlation | Action |")
        lines.append("|----------|----------|-------------|--------|")
        for a, b, corr in redundant_pairs:
            la = SIGNAL_LABELS.get(a, a)
            lb = SIGNAL_LABELS.get(b, b)
            action = "Consider removing or reducing weight on one"
            lines.append(f"| {la} | {lb} | **{corr:+.3f}** | {action} |")
        lines.append("")
        lines.append(
            "> **Recommendation:** Signals with |r| > {:.2f} are largely measuring ".format(threshold)
            + "the same information. Consider reducing the weight of the lower-UIR signal "
            + "in the composite score to avoid double-counting."
        )
    else:
        lines.append(
            f"_No redundant pairs found at threshold |r| > {threshold:.2f}._  \n"
            "_All 9 signals are providing sufficiently independent information._"
        )
    lines.append("")

    # ── UIR ranking ───────────────────────────────────────────────────────────
    lines.append("## Unique Information Ratio (UIR) by Signal")
    lines.append(
        "\n_UIR = 1 - max|r| with any other signal.  \n"
        "UIR = 1.0 → fully unique. UIR = 0.0 → perfectly correlated with another signal._\n"
    )
    if uir:
        sorted_uir = sorted(uir.items(), key=lambda x: x[1], reverse=True)
        lines.append("| Rank | Signal | Label | UIR | Uniqueness |")
        lines.append("|------|--------|-------|-----|------------|")
        for rank, (sig, val) in enumerate(sorted_uir, 1):
            label = SIGNAL_LABELS.get(sig, sig)
            if val >= 0.70:
                uniqueness = "High"
            elif val >= 0.40:
                uniqueness = "Moderate"
            else:
                uniqueness = "Low — consider reviewing"
            lines.append(f"| {rank} | `{sig}` | {label} | {val:.3f} | {uniqueness} |")
    else:
        lines.append("_UIR data not available._")
    lines.append("")

    # ── Time-series section (if computed) ────────────────────────────────────
    if run_ts and timeseries_corr is not None and not timeseries_corr.empty:
        lines.append("## Time-Series Correlation Matrix")
        lines.append(
            "\n_(Pearson r across dates — daily cross-sectional means per signal)_\n"
        )
        lines.append(_format_corr_table(timeseries_corr))
        if div_score_ts is not None:
            interp_ts = _interpret_div_score(div_score_ts)
            lines.append(f"\n> **Time-series diversification:** {interp_ts}")
            lines.append(
                f"  Effective N (time-series) = {eff_n_ts:.2f} / {total_n} "
                f"(score = {div_score_ts:.1f})"
            )
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append(f"_Canyon v9 Step 104 · {ts}_")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Report written → {OUT_REPORT.name}")


def run_signal_correlation(threshold: float, run_ts: bool) -> None:
    """Main entry point for Step 104 Signal Correlation Monitor."""
    print("=" * 70)
    print("Canyon v9 — Step 104: Signal Correlation & Diversification Monitor")
    print("=" * 70)

    # ── [1/3] Load data ───────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    snapshot = load_snapshot()
    history: Optional[pd.DataFrame] = None

    if run_ts:
        print("  Loading time-series history ...")
        history = load_history()
    else:
        print("  [SKIP] Time-series mode not requested (use --time-series to enable)")

    if snapshot is None:
        print("\n[ERROR] alpha_scores.csv is required but not found. Exiting.")
        raise SystemExit(1)

    n_tickers = len(snapshot)
    sig_cols_present = _available_sig_cols(snapshot, SIGNAL_COLS)
    total_n = len(sig_cols_present)

    if total_n == 0:
        print("\n[ERROR] No valid sig_* columns found in alpha_scores.csv. Exiting.")
        raise SystemExit(1)

    print(f"  Signal columns found: {', '.join(sig_cols_present)}")

    # ── [2/3] Compute correlations ────────────────────────────────────────────
    print(f"\n[2/3] Computing correlations (redundancy threshold |r| > {threshold:.2f}) ...")

    # Cross-sectional correlation (across tickers, current snapshot)
    print(f"  Cross-sectional correlation ({n_tickers} tickers) ...")
    snapshot_corr = pearson_corr_matrix(snapshot, sig_cols_present)

    eff_n_xs   = effective_signal_count(snapshot_corr)
    div_score_xs = diversification_score(eff_n_xs, total_n)
    uir        = unique_information_ratio(snapshot_corr)
    redundant  = signal_redundancy_pairs(snapshot_corr, threshold)

    # Time-series correlation (optional)
    timeseries_corr: Optional[pd.DataFrame] = None
    eff_n_ts:        Optional[float]        = None
    div_score_ts:    Optional[float]        = None

    if run_ts and history is not None:
        print("  Time-series correlation (daily mean per signal across dates) ...")
        timeseries_corr = compute_timeseries_corr(history)
        if timeseries_corr is not None and not timeseries_corr.empty:
            eff_n_ts    = effective_signal_count(timeseries_corr)
            div_score_ts = diversification_score(eff_n_ts, total_n)

    # ── Print correlation matrix to stdout ────────────────────────────────────
    print("\n  Cross-sectional Pearson correlation matrix:")
    print("  " + "-" * 60)
    if not snapshot_corr.empty:
        # Pretty-print with short labels
        short_labels = {c: SIGNAL_LABELS.get(c, c)[:8] for c in sig_cols_present}
        display = snapshot_corr.rename(index=short_labels, columns=short_labels)
        # Format to 2 decimal places
        formatted_display = display.map(lambda x: f"{x:+.2f}" if pd.notna(x) else "  —  ")
        col_width = max(len(c) for c in formatted_display.columns) + 2
        header_line = " " * 14 + "".join(c.ljust(col_width) for c in formatted_display.columns)
        print("  " + header_line)
        for row_label in formatted_display.index:
            row_str = "  " + row_label.ljust(14) + "".join(
                str(v).ljust(col_width) for v in formatted_display.loc[row_label]
            )
            print(row_str)
    print("  " + "-" * 60)

    # ── Print key metrics ─────────────────────────────────────────────────────
    print(f"\n  Effective signal count: {eff_n_xs:.2f} / {total_n}")
    print(f"  Diversification score:  {div_score_xs:.1f} / 100")
    print(f"  Interpretation:         {_interpret_div_score(div_score_xs)}")

    if eff_n_ts is not None:
        print(f"\n  [Time-series] Effective N:          {eff_n_ts:.2f} / {total_n}")
        print(f"  [Time-series] Diversification score: {div_score_ts:.1f} / 100")

    if redundant:
        print(f"\n  Redundant pairs (|r| > {threshold:.2f}):")
        for a, b, corr in redundant:
            la = SIGNAL_LABELS.get(a, a)
            lb = SIGNAL_LABELS.get(b, b)
            print(f"    {la:14s} ↔ {lb:14s}  r = {corr:+.3f}")
    else:
        print(f"\n  No redundant pairs found at |r| > {threshold:.2f}.")

    print("\n  UIR ranking (most unique → least unique):")
    sorted_uir = sorted(uir.items(), key=lambda x: x[1], reverse=True)
    for sig, val in sorted_uir:
        label = SIGNAL_LABELS.get(sig, sig)
        bar_len = int(round(val * 20))
        bar = "▓" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:14s}  UIR={val:.3f}  [{bar}]")

    # ── [3/3] Write outputs ────────────────────────────────────────────────────
    print(f"\n[3/3] Writing outputs ...")

    # signal_correlation.csv (long format)
    long_df = _corr_to_long(snapshot_corr, threshold, label="cross_sectional")

    if timeseries_corr is not None and not timeseries_corr.empty:
        ts_long = _corr_to_long(timeseries_corr, threshold, label="time_series")
        long_df = pd.concat([long_df, ts_long], ignore_index=True)

    long_df.to_csv(OUT_CSV, index=False)
    print(f"  [OK] {OUT_CSV.name} — {len(long_df)} rows")

    # signal_correlation_report.md
    _write_report(
        snapshot_corr   = snapshot_corr,
        timeseries_corr = timeseries_corr,
        eff_n_xs        = eff_n_xs,
        eff_n_ts        = eff_n_ts,
        uir             = uir,
        redundant_pairs = redundant,
        div_score_xs    = div_score_xs,
        div_score_ts    = div_score_ts,
        total_n         = total_n,
        threshold       = threshold,
        n_tickers       = n_tickers,
        run_ts          = run_ts,
    )

    print("\n" + "=" * 70)
    print("Step 104 complete.")
    print(f"  {OUT_CSV.name}")
    print(f"  {OUT_REPORT.name}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canyon v9 Step 104 — Signal Correlation & Diversification Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=REDUNDANCY_THRESHOLD_DEFAULT,
        help="Redundancy correlation threshold (flag pairs with |r| above this value)",
    )
    p.add_argument(
        "--cross-sectional",
        dest="cross_sectional",
        action="store_true",
        default=True,
        help="Use current snapshot for cross-sectional correlation (default: True)",
    )
    p.add_argument(
        "--time-series",
        dest="time_series",
        action="store_true",
        default=False,
        help="Also compute time-series correlation if history file is available",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_signal_correlation(
        threshold=args.threshold,
        run_ts=args.time_series,
    )
