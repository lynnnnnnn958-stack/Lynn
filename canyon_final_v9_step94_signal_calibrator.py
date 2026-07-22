#!/usr/bin/env python3
"""
canyon_final_v9_step94_signal_calibrator.py
============================================
Canyon v9  Step 94 — IC-Based Signal Calibrator

Computes IC-calibrated weight multipliers for every signal used in Step 87
(Alpha Aggregator) and writes signal_weights.json for Step 87 to read at
runtime.

Algorithm overview:
  1. Load per-signal IC from live_ic_history.csv (ic_<name> columns).
     If those columns are absent, approximate IC by correlating each
     sig_* column in score_history / alpha_score_history with forward
     alpha score change.
  2. Convert raw ICs to multipliers: clip → shift → normalise → cap.
  3. Shrink toward equal (1.0) to prevent overfitting sparse histories.
  4. Apply multipliers to BASE_WEIGHTS and renormalise to sum=1.
  5. Write signal_weights.json (read by Step 87) and
     signal_calibration_report.md (human-readable audit trail).

Signal names match SIGNAL_CONFIG in Step 87 exactly:
  regime_ml, quality, revision, surprise, sentiment,
  squeeze, insider, options, ml_ensemble

Usage:
  python3 canyon_final_v9_step94_signal_calibrator.py
  python3 canyon_final_v9_step94_signal_calibrator.py --shrinkage 0.6
  python3 canyon_final_v9_step94_signal_calibrator.py --min-periods 5
  python3 canyon_final_v9_step94_signal_calibrator.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_NAMES: list[str] = [
    "regime_ml", "quality", "revision", "surprise", "sentiment",
    "squeeze", "insider", "options", "ml_ensemble",
]

# Base weights from Step 87 SIGNAL_CONFIG (fallback reference; Step 87 has
# regime-conditional overrides, but these are the documented neutral weights).
BASE_WEIGHTS: dict[str, float] = {
    "regime_ml":   0.28,
    "quality":     0.18,
    "revision":    0.14,
    "surprise":    0.10,
    "sentiment":   0.09,
    "squeeze":     0.08,
    "insider":     0.07,
    "options":     0.06,
    "ml_ensemble": 0.10,
}

# Column name mapping: signal name → score_history column (approximate IC proxy)
# These columns exist in score_history.csv as of Canyon v9 Step 87.
SCORE_HISTORY_COL: dict[str, str] = {
    "regime_ml":   "predicted_score",
    "quality":     "quality_score",
    "revision":    "rank_revision",
    "surprise":    "rank_sue",
    "sentiment":   "rank_sentiment",
    "squeeze":     "rank_squeeze",
    "insider":     None,   # not in score_history; falls back to neutral
    "options":     "rank_options",
    "ml_ensemble": None,   # not in score_history; falls back to neutral
}

# Paths
IC_HIST_PATH       = ROOT / "live_ic_history.csv"
SCORE_HIST_PATH    = ROOT / "score_history.csv"
ALPHA_HIST_PATH    = ROOT / "alpha_score_history.csv"
OUT_WEIGHTS_PATH   = ROOT / "signal_weights.json"
OUT_REPORT_PATH    = ROOT / "signal_calibration_report.md"


# ─────────────────────────────────────────────────────────────────────────────
# Regime helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_current_regime() -> str:
    """Read current market regime; default BULL."""
    json_path = ROOT / "regime_current.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            r = str(data.get("regime", "")).upper()
            if r:
                return r
        except Exception:
            pass

    hist_path = ROOT / "regime_history.csv"
    if hist_path.exists():
        try:
            df = pd.read_csv(hist_path)
            if "regime" in df.columns and not df.empty:
                return str(df["regime"].iloc[-1]).upper()
        except Exception:
            pass

    return "BULL"


# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation (no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation without scipy.
    Ranks ties with average method via pandas.
    """
    if len(x) < 4:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").values.astype(float)
    y_rank = pd.Series(y).rank(method="average").values.astype(float)
    # Pearson on ranks = Spearman
    xm = x_rank - x_rank.mean()
    ym = y_rank - y_rank.mean()
    denom = (np.sqrt((xm ** 2).sum()) * np.sqrt((ym ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Core: load per-signal IC
# ─────────────────────────────────────────────────────────────────────────────

def load_per_signal_ic(min_periods: int = 10) -> dict[str, float]:
    """
    Try to load per-signal IC from live_ic_history.csv (ic_<name> columns).

    If per-signal columns are absent, approximate IC by cross-sectionally
    correlating each sig_* score column in score_history.csv with the
    forward alpha_score change recorded in alpha_score_history.csv.

    Falls back to neutral (returns empty dict, meaning 1.0 multiplier for
    all signals) if insufficient data is available.

    Returns {signal_name: mean_ic}.  IC values may be negative.
    """
    # ── Attempt 1: per-signal IC columns in live_ic_history.csv ─────────────
    if IC_HIST_PATH.exists():
        try:
            ic_df = pd.read_csv(IC_HIST_PATH)
            per_sig_cols = {
                name: f"ic_{name}"
                for name in SIGNAL_NAMES
                if f"ic_{name}" in ic_df.columns
            }
            if per_sig_cols:
                result: dict[str, float] = {}
                for name, col in per_sig_cols.items():
                    vals = pd.to_numeric(ic_df[col], errors="coerce").dropna()
                    if len(vals) >= min_periods:
                        result[name] = float(vals.mean())
                    # else: signal absent — will become 1.0 multiplier
                if result:
                    n = len(ic_df)
                    print(f"    [ic] Loaded per-signal IC from live_ic_history.csv "
                          f"({len(result)}/{len(SIGNAL_NAMES)} signals, {n} periods)")
                    return result
        except Exception as e:
            print(f"    [WARN] live_ic_history.csv read error: {e}")

    # ── Attempt 2: approximate IC from score_history × alpha_score_history ──
    print("    [ic] Per-signal IC columns absent — approximating from "
          "score_history + alpha_score_history")

    if not SCORE_HIST_PATH.exists():
        print("    [ic] score_history.csv not found — using neutral multipliers")
        return {}

    if not ALPHA_HIST_PATH.exists():
        print("    [ic] alpha_score_history.csv not found — using neutral multipliers")
        return {}

    try:
        scores_df = pd.read_csv(SCORE_HIST_PATH)
        alpha_df  = pd.read_csv(ALPHA_HIST_PATH)

        if "date" not in scores_df.columns or "ticker" not in scores_df.columns:
            print("    [ic] score_history.csv missing date/ticker columns")
            return {}
        if "date" not in alpha_df.columns or "ticker" not in alpha_df.columns:
            print("    [ic] alpha_score_history.csv missing date/ticker columns")
            return {}

        scores_df["date"] = pd.to_datetime(scores_df["date"], errors="coerce")
        alpha_df["date"]  = pd.to_datetime(alpha_df["date"],  errors="coerce")

        # Build alpha_score lookup: date × ticker → alpha_score
        if "alpha_score" not in alpha_df.columns:
            print("    [ic] alpha_score_history.csv missing alpha_score column")
            return {}

        alpha_pivot = alpha_df.pivot_table(
            index="date", columns="ticker", values="alpha_score", aggfunc="last"
        )
        # Forward alpha_score change: next available date's score − current score
        alpha_shift = alpha_pivot.shift(-1) - alpha_pivot

        dates_sorted = sorted(scores_df["date"].dropna().unique())
        if len(dates_sorted) < 2:
            print("    [ic] Insufficient scoring dates in score_history.csv")
            return {}

        result_approx: dict[str, list[float]] = {n: [] for n in SIGNAL_NAMES}

        for dt in dates_sorted[:-1]:   # skip last date (no forward return)
            grp = scores_df[scores_df["date"] == dt]
            if grp.empty:
                continue

            # Forward alpha score change for this date
            fwd_dates = [d for d in alpha_pivot.index if d > dt]
            if not fwd_dates:
                continue
            next_dt = min(fwd_dates)
            if next_dt not in alpha_shift.index:
                continue
            fwd_row = alpha_shift.loc[next_dt].dropna()
            if len(fwd_row) < 5:
                continue

            for name in SIGNAL_NAMES:
                col = SCORE_HISTORY_COL.get(name)
                if col is None or col not in grp.columns:
                    continue

                sig = pd.to_numeric(grp.set_index("ticker")[col], errors="coerce").dropna()
                common = sig.index.intersection(fwd_row.index)
                if len(common) < 5:
                    continue

                ic_val = _spearman_ic(
                    sig[common].values.astype(float),
                    fwd_row[common].values.astype(float),
                )
                if not np.isnan(ic_val):
                    result_approx[name].append(ic_val)

        final: dict[str, float] = {}
        for name, vals in result_approx.items():
            if len(vals) >= min_periods:
                final[name] = float(np.mean(vals))

        if final:
            print(f"    [ic] Approximated IC for {len(final)}/{len(SIGNAL_NAMES)} signals "
                  f"from {len(dates_sorted)} scoring dates")
        else:
            print("    [ic] Insufficient periods for IC approximation — "
                  "using neutral multipliers")

        return final

    except Exception as e:
        print(f"    [WARN] IC approximation error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# IC → multipliers
# ─────────────────────────────────────────────────────────────────────────────

def ic_to_multipliers(
    ic_dict: dict[str, float],
    min_ic: float = -0.10,
    max_multiplier: float = 2.0,
) -> dict[str, float]:
    """
    Convert raw IC values to weight multipliers.

    Algorithm:
      1. Clip IC to [min_ic, +inf]  (reduce very negative signals, do not zero out)
      2. Shift so the minimum clipped IC maps to 0.5 (not zero)
      3. Normalise relative to mean of all signal ICs
      4. Cap at max_multiplier

    Signals with no IC data → multiplier = 1.0 (neutral).
    """
    if not ic_dict:
        return {name: 1.0 for name in SIGNAL_NAMES}

    # Step 1: clip
    clipped: dict[str, float] = {
        name: max(ic, min_ic) for name, ic in ic_dict.items()
    }

    # Fill missing signals with the mean of available clipped ICs (neutral proxy)
    clipped_vals = list(clipped.values())
    mean_clipped = float(np.mean(clipped_vals)) if clipped_vals else 0.0
    all_clipped = {
        name: clipped.get(name, mean_clipped) for name in SIGNAL_NAMES
    }

    # Step 2: shift so min maps to 0.5
    min_val = min(all_clipped.values())
    shifted = {name: v - min_val + 0.5 for name, v in all_clipped.items()}

    # Step 3: normalise relative to mean
    mean_shifted = float(np.mean(list(shifted.values())))
    if mean_shifted == 0:
        normalised = {name: 1.0 for name in SIGNAL_NAMES}
    else:
        normalised = {name: v / mean_shifted for name, v in shifted.items()}

    # Step 4: cap at max_multiplier; signals with no IC data get 1.0
    result: dict[str, float] = {}
    for name in SIGNAL_NAMES:
        if name not in ic_dict:
            result[name] = 1.0        # no data → neutral
        else:
            result[name] = round(min(normalised[name], max_multiplier), 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Shrinkage
# ─────────────────────────────────────────────────────────────────────────────

def shrink_multipliers(
    multipliers: dict[str, float],
    shrinkage: float = 0.40,
) -> dict[str, float]:
    """
    Shrink multipliers toward equal weight (1.0) to prevent overfitting.

    shrunk = shrinkage * 1.0 + (1 - shrinkage) * raw_multiplier
    """
    return {
        name: round(shrinkage * 1.0 + (1.0 - shrinkage) * m, 4)
        for name, m in multipliers.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Apply multipliers to base weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibrated_weights(
    base_weights: dict[str, float],
    multipliers: dict[str, float],
) -> dict[str, float]:
    """Apply shrunk multipliers to base weights, then renormalise to sum=1."""
    raw = {
        name: base_weights.get(name, 0.0) * multipliers.get(name, 1.0)
        for name in SIGNAL_NAMES
    }
    total = sum(raw.values())
    if total == 0:
        # Fallback: equal weights
        eq = 1.0 / len(SIGNAL_NAMES)
        return {name: round(eq, 4) for name in SIGNAL_NAMES}

    return {name: round(v / total, 4) for name, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────────────────────────────────────

def write_signal_weights(
    weights: dict[str, float],
    multipliers: dict[str, float],
    ic_dict: dict[str, float],
    regime: str,
    n_ic_periods: int,
    dry_run: bool = False,
) -> None:
    """
    Write signal_weights.json with structure:
    {
      "updated": "YYYY-MM-DD",
      "regime": regime_str,
      "n_ic_periods": int,
      "weights": {name: float, ...},        # calibrated weights (sum=1)
      "ic_multipliers": {name: float, ...}, # shrunk multipliers
      "raw_ic": {name: float, ...},         # raw IC values (None if absent)
      "note": "IC-calibrated via step94"
    }
    """
    payload = {
        "updated":      datetime.now().strftime("%Y-%m-%d"),
        "regime":       regime,
        "n_ic_periods": n_ic_periods,
        "weights":      weights,
        "ic_multipliers": multipliers,
        "raw_ic":       {name: round(ic_dict[name], 4) if name in ic_dict else None
                         for name in SIGNAL_NAMES},
        "note":         "IC-calibrated via step94",
    }

    if dry_run:
        print(f"\n  [dry-run] Would write {OUT_WEIGHTS_PATH}")
        print(json.dumps(payload, indent=2))
        return

    OUT_WEIGHTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"  [write] {OUT_WEIGHTS_PATH}")


def write_calibration_report(
    weights: dict[str, float],
    multipliers: dict[str, float],
    ic_dict: dict[str, float],
    regime: str,
    n_ic_periods: int,
    shrinkage: float,
    dry_run: bool = False,
) -> None:
    """Write signal_calibration_report.md — human-readable audit trail."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build per-signal rows
    rows: list[str] = []
    rows.append("| Signal | Base Weight | IC (raw) | Multiplier (shrunk) | Calibrated Weight |")
    rows.append("|--------|-------------|----------|---------------------|-------------------|")

    for name in SIGNAL_NAMES:
        bw  = BASE_WEIGHTS.get(name, 0.0)
        ic  = ic_dict.get(name)
        mlt = multipliers.get(name, 1.0)
        cw  = weights.get(name, 0.0)

        ic_str  = f"{ic:+.4f}" if ic is not None else "n/a"
        mlt_str = f"{mlt:.4f}"
        rows.append(
            f"| {name:<12} | {bw:.2f}        | {ic_str:>8} | {mlt_str:>19} | {cw:.4f}            |"
        )

    # Interpretation block
    if ic_dict:
        positive_signals = [n for n, v in ic_dict.items() if v > 0.03]
        weak_signals     = [n for n, v in ic_dict.items() if 0 < v <= 0.03]
        negative_signals = [n for n, v in ic_dict.items() if v <= 0]
        interp_lines = [
            "### Signal Interpretation",
            "",
            f"- **Strong positive IC (>0.03):** {', '.join(positive_signals) or 'none'}",
            f"- **Weak positive IC (0–0.03):** {', '.join(weak_signals) or 'none'}",
            f"- **Negative IC (<= 0):** {', '.join(negative_signals) or 'none'}",
            "",
            "Signals with negative IC have their multiplier reduced but not zeroed "
            "(min_ic floor = -0.10). Shrinkage further dampens extreme multipliers to "
            "guard against overfitting sparse IC histories.",
        ]
    else:
        interp_lines = [
            "### Signal Interpretation",
            "",
            "No IC data was available for any signal. All multipliers are set to 1.0 "
            "(neutral). Run Step 84 to accumulate live IC history, then re-run Step 94.",
        ]

    # Calibrated weight ranking
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    rank_lines = ["### Calibrated Weight Ranking"]
    for i, (name, w) in enumerate(ranked, 1):
        rank_lines.append(f"  {i}. **{name}** — {w:.2%}")

    lines = [
        "# Canyon v9 — Step 94: IC-Based Signal Calibration Report",
        f"Generated: {now}  |  Regime: {regime}  |  IC periods: {n_ic_periods}",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Step 94 computes information-coefficient (IC) based weight multipliers "
        "for the nine signals used by the Step 87 Alpha Aggregator.",
        "",
        "**Steps:**",
        "1. Load per-signal IC from `live_ic_history.csv` (columns `ic_<name>`). "
        "   If absent, approximate IC by correlating each signal's cross-sectional "
        "   ranks against the one-step-ahead change in alpha_score.",
        "2. Clip IC to [-0.10, +inf] — do not zero out underperforming signals; "
        "   just reduce their influence.",
        "3. Shift so the worst clipped IC maps to 0.5 (preserving partial credit).",
        "4. Normalise relative to mean IC across all signals.",
        "5. Cap each multiplier at 2.0 to prevent single-signal dominance.",
        f"6. Shrink toward 1.0 with shrinkage factor = **{shrinkage}** to limit "
        "   overfitting.",
        "7. Apply shrunk multipliers to Step 87 BASE_WEIGHTS and renormalise to sum=1.",
        "",
        "---",
        "",
        "## Per-Signal Table",
        "",
        *rows,
        "",
        "---",
        "",
        *interp_lines,
        "",
        "---",
        "",
        *rank_lines,
        "",
        "---",
        "",
        "## Output Files",
        "",
        "- `signal_weights.json` — machine-readable weights for Step 87 runtime",
        "- `signal_calibration_report.md` — this report",
        "",
        "_Step 94 — IC-Based Signal Calibrator_",
    ]

    text = "\n".join(lines) + "\n"

    if dry_run:
        print(f"\n  [dry-run] Would write {OUT_REPORT_PATH}")
        return

    OUT_REPORT_PATH.write_text(text)
    print(f"  [write] {OUT_REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    weights: dict[str, float],
    multipliers: dict[str, float],
    ic_dict: dict[str, float],
) -> None:
    """Print: signal | base_weight | ic_raw | multiplier | calibrated_weight."""
    header = (
        f"\n  {'Signal':<14}  {'Base W':>8}  {'IC (raw)':>9}  "
        f"{'Mult (shrunk)':>14}  {'Calib W':>8}"
    )
    sep = "  " + "-" * 62
    print(header)
    print(sep)

    for name in SIGNAL_NAMES:
        bw  = BASE_WEIGHTS.get(name, 0.0)
        ic  = ic_dict.get(name)
        mlt = multipliers.get(name, 1.0)
        cw  = weights.get(name, 0.0)

        ic_str = f"{ic:+.4f}" if ic is not None else "    n/a"
        print(
            f"  {name:<14}  {bw:>7.2%}  {ic_str:>9}  "
            f"{mlt:>14.4f}  {cw:>7.2%}"
        )

    print(sep)
    total_base = sum(BASE_WEIGHTS.values())
    total_cal  = sum(weights.values())
    print(
        f"  {'TOTAL':<14}  {total_base:>7.2%}  {'':>9}  "
        f"{'':>14}  {total_cal:>7.2%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main calibrate()
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    shrinkage: float = 0.40,
    min_periods: int = 10,
    dry_run: bool = False,
) -> None:
    """
    Full calibration pipeline:
      [1/4] Load IC data
      [2/4] Compute multipliers with shrinkage
      [3/4] Apply to BASE_WEIGHTS
      [4/4] Write outputs
    """
    print("\n" + "=" * 60)
    print("Canyon v9  Step 94 — IC-Based Signal Calibrator")
    print("=" * 60)

    # ── 1. Load IC data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading per-signal IC …")
    ic_dict = load_per_signal_ic(min_periods=min_periods)
    n_ic_periods = 0

    if ic_dict:
        print(f"  [ic] IC available for {len(ic_dict)}/{len(SIGNAL_NAMES)} signals")
        for name in SIGNAL_NAMES:
            if name in ic_dict:
                print(f"    {name:<14}  IC = {ic_dict[name]:+.4f}")
            else:
                print(f"    {name:<14}  IC = n/a  (will receive multiplier = 1.0)")
        # Estimate n_ic_periods from live_ic_history if it exists
        if IC_HIST_PATH.exists():
            try:
                n_ic_periods = len(pd.read_csv(IC_HIST_PATH))
            except Exception:
                n_ic_periods = 0
    else:
        print("  [ic] No IC data available — all multipliers set to neutral (1.0)")

    # ── 2. Compute multipliers ────────────────────────────────────────────────
    print("\n[2/4] Computing IC multipliers …")
    raw_multipliers  = ic_to_multipliers(ic_dict)
    shrunk_mults     = shrink_multipliers(raw_multipliers, shrinkage=shrinkage)
    print(f"  Shrinkage = {shrinkage:.0%}  "
          f"(multiplier range after shrinkage: "
          f"[{min(shrunk_mults.values()):.3f}, {max(shrunk_mults.values()):.3f}])")

    # ── 3. Apply to BASE_WEIGHTS ──────────────────────────────────────────────
    print("\n[3/4] Applying multipliers to base weights …")
    cal_weights = compute_calibrated_weights(BASE_WEIGHTS, shrunk_mults)
    regime      = _get_current_regime()
    print(f"  Regime detected: {regime}")

    print_comparison_table(cal_weights, shrunk_mults, ic_dict)

    # ── 4. Write outputs ──────────────────────────────────────────────────────
    print("\n[4/4] Writing outputs …")
    write_signal_weights(
        weights=cal_weights,
        multipliers=shrunk_mults,
        ic_dict=ic_dict,
        regime=regime,
        n_ic_periods=n_ic_periods,
        dry_run=dry_run,
    )
    write_calibration_report(
        weights=cal_weights,
        multipliers=shrunk_mults,
        ic_dict=ic_dict,
        regime=regime,
        n_ic_periods=n_ic_periods,
        shrinkage=shrinkage,
        dry_run=dry_run,
    )

    print("\n" + "=" * 60)
    if dry_run:
        print("Step 94 — dry-run complete (no files written)")
    else:
        print("Step 94 — calibration complete")
        print(f"  signal_weights.json         → {OUT_WEIGHTS_PATH}")
        print(f"  signal_calibration_report.md → {OUT_REPORT_PATH}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 94 — IC-Based Signal Calibrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Writes signal_weights.json (read by Step 87 at runtime) and\n"
            "signal_calibration_report.md (human-readable audit trail).\n\n"
            "Examples:\n"
            "  python3 canyon_final_v9_step94_signal_calibrator.py\n"
            "  python3 canyon_final_v9_step94_signal_calibrator.py --shrinkage 0.6\n"
            "  python3 canyon_final_v9_step94_signal_calibrator.py --dry-run"
        ),
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.40,
        metavar="FLOAT",
        help="Shrinkage toward equal weight 1.0 (default: 0.40). "
             "Higher = more conservative, closer to equal weights.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=10,
        metavar="INT",
        help="Minimum IC periods required before a signal receives a non-neutral "
             "multiplier (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print without writing any files.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.shrinkage <= 1.0):
        parser.error("--shrinkage must be between 0.0 and 1.0")
    if args.min_periods < 1:
        parser.error("--min-periods must be >= 1")

    calibrate(
        shrinkage=args.shrinkage,
        min_periods=args.min_periods,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
