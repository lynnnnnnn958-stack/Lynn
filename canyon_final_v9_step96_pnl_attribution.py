"""
Canyon v9 — Step 96: P&L Attribution Engine
============================================
Decomposes realized P&L from trade_journal.csv by signal source.
Answers: "Which signals (ML, options flow, insider, momentum, etc.)
contributed most to our wins and losses?"

Inputs:
  trade_journal.csv         — ticker, entry_date, exit_date, pnl_pct,
                              strategy_type, signal_grade, plan_adherence,
                              holding_days
  alpha_score_history.csv   — date, ticker, alpha_score, sig_regime_ml,
                              sig_quality, sig_revision, sig_surprise,
                              sig_sentiment, sig_squeeze, sig_insider,
                              sig_options, sig_ml_ensemble
  alpha_scores.csv          — latest scores (same sig_* columns), fallback

Outputs:
  pnl_attribution.csv         — one row per trade x signal
  pnl_attribution_report.md   — executive summary + ranked tables

Usage:
  python canyon_final_v9_step96_pnl_attribution.py [--min-trades 5] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

TRADE_JOURNAL      = ROOT / "trade_journal.csv"
HISTORY_CSV        = ROOT / "alpha_score_history.csv"
LATEST_CSV         = ROOT / "alpha_scores.csv"
OUTPUT_CSV         = ROOT / "pnl_attribution.csv"
OUTPUT_MD          = ROOT / "pnl_attribution_report.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNAL_NAMES = [
    "regime_ml",
    "quality",
    "revision",
    "surprise",
    "sentiment",
    "squeeze",
    "insider",
    "options",
    "ml_ensemble",
]

# Mapping: signal name -> column name in alpha_score CSVs
SIG_COLS = {name: f"sig_{name}" for name in SIGNAL_NAMES}

NEUTRAL = 50.0          # score at which a signal contributes nothing
DEFAULT_SCORE = 50.0    # score used when lookup fails


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trade_journal(path: Path) -> pd.DataFrame:
    """Load trade_journal.csv.  Returns empty DataFrame if missing/empty."""
    if not path.exists():
        print(f"[WARN] Trade journal not found: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        print(f"[WARN] Trade journal is empty: {path}")
        return df

    # Normalise required columns — use sensible defaults where columns are absent
    required = ["ticker", "entry_date", "exit_date", "pnl_pct",
                 "strategy_type", "signal_grade", "plan_adherence", "holding_days"]
    for col in required:
        if col not in df.columns:
            if col == "pnl_pct":
                # Try realized_return as alias
                if "realized_return" in df.columns:
                    df["pnl_pct"] = pd.to_numeric(df["realized_return"], errors="coerce")
                else:
                    print(f"[WARN] Column '{col}' missing and no alias found — defaulting to 0")
                    df["pnl_pct"] = 0.0
            elif col in ("strategy_type", "signal_grade", "plan_adherence"):
                df[col] = "unknown"
            elif col == "holding_days":
                df[col] = np.nan
            else:
                df[col] = np.nan

    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"],  errors="coerce")

    # Drop rows where we cannot determine pnl or ticker
    df = df.dropna(subset=["ticker"])

    return df.reset_index(drop=True)


def load_alpha_history(path: Path) -> pd.DataFrame:
    """Load alpha_score_history.csv.  Returns empty DataFrame if missing."""
    if not path.exists():
        print(f"[WARN] Alpha score history not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "ticker"])
        return df
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return pd.DataFrame()


def load_alpha_latest(path: Path) -> pd.DataFrame:
    """Load alpha_scores.csv (latest snapshot).  Returns empty DataFrame if missing."""
    if not path.exists():
        print(f"[WARN] Latest alpha scores not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core attribution functions
# ---------------------------------------------------------------------------

def load_signal_scores_at_entry(
    ticker: str,
    entry_date: pd.Timestamp,
    history_df: pd.DataFrame,
    latest_df: pd.DataFrame,
) -> dict[str, float]:
    """Look up sig_* scores for ticker at or before entry_date.

    Priority:
    1. alpha_score_history — most recent row for ticker with date <= entry_date
    2. alpha_scores (latest snapshot) if history lookup fails
    3. DEFAULT_SCORE (50.0) for any signal still missing

    Returns {signal_name: score}.
    """
    scores: dict[str, float] = {}

    # --- History lookup ---
    if not history_df.empty and pd.notna(entry_date):
        mask = (history_df["ticker"] == ticker) & (history_df["date"] <= entry_date)
        candidates = history_df.loc[mask]
        if not candidates.empty:
            row = candidates.sort_values("date").iloc[-1]
            for sig, col in SIG_COLS.items():
                if col in row.index and pd.notna(row[col]):
                    scores[sig] = float(row[col])

    # --- Latest snapshot fallback ---
    if len(scores) < len(SIGNAL_NAMES) and not latest_df.empty:
        lat = latest_df[latest_df["ticker"] == ticker]
        if not lat.empty:
            row = lat.iloc[0]
            for sig, col in SIG_COLS.items():
                if sig not in scores and col in row.index and pd.notna(row[col]):
                    scores[sig] = float(row[col])

    # --- Default for any remaining missing signals ---
    for sig in SIGNAL_NAMES:
        scores.setdefault(sig, DEFAULT_SCORE)

    return scores


def compute_attribution_shares(signal_scores: dict[str, float]) -> dict[str, float]:
    """Convert signal scores to attribution shares.

    share_X = (score_X - 50) / sum(|score_i - 50|)

    A signal at 50 (neutral) contributes 0.
    The shares sum to +-1 (or 0 if all signals are neutral).
    When total deviation is 0, all shares are 0 (no attribution possible).
    """
    deviations = {sig: score - NEUTRAL for sig, score in signal_scores.items()}
    total_abs  = sum(abs(d) for d in deviations.values())

    if total_abs < 1e-9:
        return {sig: 0.0 for sig in signal_scores}

    return {sig: dev / total_abs for sig, dev in deviations.items()}


def attribute_trade(
    trade_row: pd.Series,
    signal_scores: dict[str, float],
    attribution_shares: dict[str, float],
) -> dict[str, float]:
    """Compute attributed P&L (pct) per signal for one trade.

    attributed_pnl_X = pnl_pct × share_X
    """
    pnl = float(trade_row["pnl_pct"])
    return {sig: pnl * share for sig, share in attribution_shares.items()}


def aggregate_attribution(
    attributed_trades: list[dict],
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate attribution across all trades by signal.

    Returns DataFrame with columns:
      signal, total_attributed_pnl, avg_attributed_pnl,
      n_trades_contributed, win_attribution, loss_attribution,
      attribution_pct_of_total
    """
    if not attributed_trades:
        return pd.DataFrame(columns=[
            "signal", "total_attributed_pnl", "avg_attributed_pnl",
            "n_trades_contributed", "win_attribution", "loss_attribution",
            "attribution_pct_of_total",
        ])

    rows = []

    for sig in SIGNAL_NAMES:
        sig_pnls   = [t.get(sig, 0.0) for t in attributed_trades]
        total      = float(np.sum(sig_pnls))
        n_contrib  = int(np.sum([abs(v) > 1e-9 for v in sig_pnls]))
        avg        = total / n_contrib if n_contrib else 0.0
        win_attr   = float(np.sum([v for v in sig_pnls if v > 0]))
        loss_attr  = float(np.sum([v for v in sig_pnls if v < 0]))
        rows.append({
            "signal":               sig,
            "total_attributed_pnl": round(total, 4),
            "avg_attributed_pnl":   round(avg,   4),
            "n_trades_contributed": n_contrib,
            "win_attribution":      round(win_attr,  4),
            "loss_attribution":     round(loss_attr, 4),
        })

    df = pd.DataFrame(rows)
    total_gross = df["total_attributed_pnl"].abs().sum()
    if total_gross > 1e-9:
        df["attribution_pct_of_total"] = (
            df["total_attributed_pnl"].abs() / total_gross * 100
        ).round(1)
    else:
        df["attribution_pct_of_total"] = 0.0

    df = df.sort_values("total_attributed_pnl", ascending=False).reset_index(drop=True)
    return df


def attribution_by_strategy(
    attributed_trades: list[dict],
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attribution broken down by strategy_type."""
    if not attributed_trades or "strategy_type" not in trades_df.columns:
        return pd.DataFrame()

    strats = trades_df["strategy_type"].fillna("unknown").values
    unique_strats = sorted(set(strats))

    rows = []
    for strat in unique_strats:
        idx = [i for i, s in enumerate(strats) if s == strat]
        for sig in SIGNAL_NAMES:
            sig_pnls = [attributed_trades[i].get(sig, 0.0) for i in idx]
            rows.append({
                "strategy_type":        strat,
                "signal":               sig,
                "total_attributed_pnl": round(float(np.sum(sig_pnls)), 4),
                "n_trades":             len(idx),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["strategy_type", "total_attributed_pnl"], ascending=[True, False])
    return df.reset_index(drop=True)


def attribution_by_grade(
    attributed_trades: list[dict],
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attribution by signal_grade (A/B/C/D)."""
    if not attributed_trades or "signal_grade" not in trades_df.columns:
        return pd.DataFrame()

    grades = trades_df["signal_grade"].fillna("unknown").values
    unique_grades = sorted(set(grades))

    rows = []
    for grade in unique_grades:
        idx = [i for i, g in enumerate(grades) if g == grade]
        for sig in SIGNAL_NAMES:
            sig_pnls = [attributed_trades[i].get(sig, 0.0) for i in idx]
            rows.append({
                "signal_grade":         grade,
                "signal":               sig,
                "total_attributed_pnl": round(float(np.sum(sig_pnls)), 4),
                "n_trades":             len(idx),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["signal_grade", "total_attributed_pnl"], ascending=[True, False])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    """Format a P&L percentage value for markdown tables."""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def write_attribution_csv(
    attributed_trades: list[dict],
    trades_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write pnl_attribution.csv — one row per (trade x signal)."""
    rows = []
    for i, attr in enumerate(attributed_trades):
        base = {}
        for col in ["ticker", "entry_date", "exit_date", "pnl_pct",
                     "strategy_type", "signal_grade", "plan_adherence", "holding_days"]:
            if col in trades_df.columns:
                base[col] = trades_df.at[i, col]
        for sig, val in attr.items():
            rows.append({**base, "signal": sig, "attributed_pnl_pct": round(val, 6)})

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    """Render a subset of DataFrame columns as a markdown table string."""
    if df.empty:
        return "_No data_\n"
    fmt = fmt or {}
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if c in fmt:
                cells.append(fmt[c](val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    summary_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    grade_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    path: Path,
    min_trades: int,
) -> None:
    """Write pnl_attribution_report.md."""
    n_trades = len(trades_df)
    total_pnl = trades_df["pnl_pct"].sum()
    wins  = int((trades_df["pnl_pct"] > 0).sum())
    losses = int((trades_df["pnl_pct"] < 0).sum())

    lines = [
        "# Canyon v9 — P&L Attribution Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Trades analysed: {n_trades}  |  Wins: {wins}  |  Losses: {losses}",
        f"Total P&L: {_fmt_pct(total_pnl)}",
        "",
    ]

    # ---- Executive summary ----
    lines += ["## Executive Summary", ""]
    if summary_df.empty:
        lines += ["_Insufficient data for attribution._", ""]
    else:
        top_pos = summary_df[summary_df["total_attributed_pnl"] > 0].head(3)
        top_neg = summary_df[summary_df["total_attributed_pnl"] < 0].tail(3)

        if not top_pos.empty:
            leaders = ", ".join(
                f"**{r['signal']}** ({_fmt_pct(r['total_attributed_pnl'])})"
                for _, r in top_pos.iterrows()
            )
            lines.append(f"Top alpha drivers: {leaders}")
        if not top_neg.empty:
            detractors = ", ".join(
                f"**{r['signal']}** ({_fmt_pct(r['total_attributed_pnl'])})"
                for _, r in top_neg.iterrows()
            )
            lines.append(f"Top alpha detractors: {detractors}")
        lines.append("")

    # ---- Signal summary table ----
    lines += ["## Attribution by Signal", ""]
    if not summary_df.empty:
        fmt = {
            "total_attributed_pnl": _fmt_pct,
            "avg_attributed_pnl":   _fmt_pct,
            "win_attribution":      _fmt_pct,
            "loss_attribution":     _fmt_pct,
            "attribution_pct_of_total": lambda v: f"{v:.1f}%",
        }
        cols = [
            "signal", "total_attributed_pnl", "win_attribution",
            "loss_attribution", "n_trades_contributed", "attribution_pct_of_total",
        ]
        lines.append(_md_table(summary_df, cols, fmt))
    else:
        lines += ["_No data_", ""]

    # ---- By strategy ----
    lines += ["## Attribution by Strategy Type", ""]
    if not strat_df.empty:
        fmt = {"total_attributed_pnl": _fmt_pct}
        cols = ["strategy_type", "signal", "total_attributed_pnl", "n_trades"]
        lines.append(_md_table(strat_df.head(40), cols, fmt))
    else:
        lines += ["_No data_", ""]

    # ---- By grade ----
    lines += ["## Attribution by Signal Grade", ""]
    if not grade_df.empty:
        fmt = {"total_attributed_pnl": _fmt_pct}
        cols = ["signal_grade", "signal", "total_attributed_pnl", "n_trades"]
        lines.append(_md_table(grade_df.head(40), cols, fmt))
    else:
        lines += ["_No data_", ""]

    # ---- Recommendations ----
    lines += ["## Recommendations", ""]
    if not summary_df.empty:
        neg_sigs = summary_df[
            (summary_df["total_attributed_pnl"] < 0) &
            (summary_df["n_trades_contributed"] >= min_trades)
        ]
        if neg_sigs.empty:
            lines.append("No signals with consistently negative attribution detected.")
        else:
            for _, row in neg_sigs.iterrows():
                lines.append(
                    f"- **{row['signal']}**: total attribution = "
                    f"{_fmt_pct(row['total_attributed_pnl'])} across "
                    f"{row['n_trades_contributed']} trades. "
                    "Consider reducing weight or reviewing signal calibration."
                )
    else:
        lines.append("_Insufficient data for recommendations._")

    lines += ["", "---", "_Canyon v9 Step 96 — P&L Attribution Engine_"]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_attribution(args: argparse.Namespace) -> None:
    """End-to-end P&L attribution pipeline."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 96: P&L Attribution Engine ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ---- [1/7] Load trade journal ----
    print("[1/7] Loading trade journal …")
    trades_df = load_trade_journal(TRADE_JOURNAL)

    if trades_df.empty:
        print("[WARN] No trades found. Writing empty outputs and exiting.")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False)
        OUTPUT_MD.write_text(
            "# Canyon v9 — P&L Attribution Report\n\n"
            "_No trades found in trade_journal.csv — nothing to attribute._\n",
            encoding="utf-8",
        )
        sys.exit(0)

    n_trades = len(trades_df)
    print(f"  {n_trades} trades loaded")

    if n_trades < args.min_trades:
        print(
            f"[WARN] Only {n_trades} trades found; --min-trades={args.min_trades}. "
            "Report will be generated but results may not be statistically meaningful."
        )

    # ---- [2/7] Load alpha scores ----
    print("[2/7] Loading alpha score history …")
    history_df = load_alpha_history(HISTORY_CSV)
    latest_df  = load_alpha_latest(LATEST_CSV)

    if history_df.empty and latest_df.empty:
        print("[WARN] No alpha score data found. All signals will default to 50 (neutral).")
    elif history_df.empty:
        print(f"  History: empty — using latest snapshot only ({len(latest_df)} tickers)")
    else:
        print(
            f"  History: {len(history_df):,} rows  "
            f"({history_df['date'].min().date()} → {history_df['date'].max().date()})"
        )
        if not latest_df.empty:
            print(f"  Latest snapshot: {len(latest_df)} tickers (fallback)")

    # ---- [3/7] Per-trade signal lookup ----
    print("[3/7] Looking up signal scores at entry date …")
    trade_signal_scores: list[dict[str, float]] = []

    for idx, row in trades_df.iterrows():
        scores = load_signal_scores_at_entry(
            ticker=str(row["ticker"]),
            entry_date=row["entry_date"],
            history_df=history_df,
            latest_df=latest_df,
        )
        trade_signal_scores.append(scores)
        if args.verbose:
            print(f"  [{idx}] {row['ticker']} ({row['entry_date']}): "
                  + "  ".join(f"{k}={v:.1f}" for k, v in scores.items()))

    print(f"  Signal scores resolved for {len(trade_signal_scores)} trades")

    # ---- [4/7] Compute attribution shares and attributed P&L ----
    print("[4/7] Computing attribution shares …")
    trade_shares:      list[dict[str, float]] = []
    attributed_trades: list[dict[str, float]] = []

    for i, (idx, row) in enumerate(trades_df.iterrows()):
        scores = trade_signal_scores[i]
        shares = compute_attribution_shares(scores)
        attr   = attribute_trade(row, scores, shares)
        trade_shares.append(shares)
        attributed_trades.append(attr)

        if args.verbose:
            print(f"  [{idx}] {row['ticker']}  pnl={row['pnl_pct']:+.2f}%  "
                  + "  ".join(f"{k}={v:+.3f}" for k, v in attr.items()))

    # ---- [5/7] Aggregate ----
    print("[5/7] Aggregating by signal, strategy, grade …")
    summary_df = aggregate_attribution(attributed_trades, trades_df)
    strat_df   = attribution_by_strategy(attributed_trades, trades_df)
    grade_df   = attribution_by_grade(attributed_trades, trades_df)

    print(f"  Signal summary: {len(summary_df)} signals")
    print(f"  Strategy breakdown: {strat_df['strategy_type'].nunique() if not strat_df.empty else 0} strategy types")
    print(f"  Grade breakdown:    {grade_df['signal_grade'].nunique() if not grade_df.empty else 0} grades")

    # ---- [6/7] Write outputs ----
    print("[6/7] Writing output files …")
    write_attribution_csv(attributed_trades, trades_df, OUTPUT_CSV)
    write_report(summary_df, strat_df, grade_df, trades_df, OUTPUT_MD, args.min_trades)

    # ---- [7/7] Print summary table ----
    print("[7/7] Attribution summary:\n")
    _print_summary_table(summary_df, trades_df)

    print("\n>>> Done. Exit 0.")


def _print_summary_table(summary_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    """Print a clean fixed-width attribution table to stdout."""
    if summary_df.empty:
        print("  (no data to display)")
        return

    total_pnl = trades_df["pnl_pct"].sum()

    header  = f"  {'Signal':<14}  {'Total Attr PnL':>14}  {'Win Attr':>9}  {'Loss Attr':>10}  {'N Trades':>8}  {'Share%':>6}"
    divider = "  " + "-" * (len(header) - 2)
    print(divider)
    print(header)
    print(divider)

    for _, row in summary_df.iterrows():
        print(
            f"  {row['signal']:<14}  "
            f"{row['total_attributed_pnl']:>+14.2f}  "
            f"{row['win_attribution']:>+9.2f}  "
            f"{row['loss_attribution']:>+10.2f}  "
            f"{row['n_trades_contributed']:>8d}  "
            f"{row['attribution_pct_of_total']:>5.1f}%"
        )

    print(divider)
    print(f"  {'TOTAL P&L (unattributed sum)':<30}  {total_pnl:>+.2f}%")
    print(divider)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 96: P&L Attribution Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        metavar="N",
        help="Minimum trades required to include a signal in recommendations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trade signal scores and attribution shares",
    )
    args = parser.parse_args()
    run_attribution(args)


if __name__ == "__main__":
    main()
