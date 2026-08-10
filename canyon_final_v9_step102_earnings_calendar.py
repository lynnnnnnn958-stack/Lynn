#!/usr/bin/env python3
"""
Canyon v9 — Step 102: Earnings Calendar Risk Overlay
=====================================================
Scan current picks and portfolio for upcoming earnings events. Flag earnings
risk, identify pre-earnings IV expansion windows, and surface post-earnings
drift opportunities. For research use only. No live orders.

Earnings Risk Flags
-------------------
  HIGH      days <= 3   Earnings imminent — do not initiate new position
  MEDIUM    days <= 10  Within 2 weeks — size down, define risk with options
  LOW       days <= 21  Within a month — monitor
  CLEAR     days > 21   No near-term earnings risk

IV Expansion Quality (pre-earnings straddle assessment)
-------------------------------------------------------
  EXCELLENT  iv_rank < 40   IV cheap — good straddle/strangle entry
  GOOD       iv_rank 40-60  Moderate IV — acceptable entry
  POOR       iv_rank > 70   IV already elevated — straddle too expensive
  UNKNOWN    no data

Output files
------------
  earnings_calendar.csv        — per-ticker risk overlay table
  earnings_calendar_report.md  — markdown table sorted by days_until ascending

Usage
-----
  python3 canyon_final_v9_step102_earnings_calendar.py
  python3 canyon_final_v9_step102_earnings_calendar.py --top 30
  python3 canyon_final_v9_step102_earnings_calendar.py --days-ahead 14
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_CSV    = ROOT / "earnings_calendar.csv"
OUT_REPORT = ROOT / "earnings_calendar_report.md"

# ── Date parsing formats ──────────────────────────────────────────────────────
DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
]

TODAY = datetime.now().date()


# ─────────────────────────────────────────────────────────────────────────────
# [1/3]  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    """Coerce val to float; return default on failure or non-finite."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load CSV gracefully; normalise ticker column; return None on failure."""
    if not path.exists():
        print(f"  [SKIP] {label} not found ({path.name})")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {label} is empty")
            return None
        # Normalise ticker column
        for col in df.columns:
            if col.lower() == "ticker":
                df = df.rename(columns={col: "ticker"})
                break
        if "ticker" not in df.columns:
            print(f"  [SKIP] {label} has no ticker column")
            return None
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        return df
    except Exception as exc:
        print(f"  [WARN] {label} could not be read: {exc}")
        return None


def days_until_earnings(earnings_date_str: str) -> int | None:
    """
    Parse earnings_date_str (various formats: YYYY-MM-DD, MM/DD/YYYY, etc.)
    Return integer days from today. Negative = already passed. None = unparseable.
    """
    if not earnings_date_str or str(earnings_date_str).strip().lower() in ("", "nan", "none", "nat"):
        return None
    s = str(earnings_date_str).strip()
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(s, fmt).date()
            return (d - TODAY).days
        except ValueError:
            continue
    # Last attempt: pandas flexible parsing
    try:
        d = pd.to_datetime(s).date()
        return (d - TODAY).days
    except Exception:
        return None


def earnings_risk_flag(days: int | None) -> str:
    """
    days <= 3:        "HIGH"    (earnings imminent — do not initiate new position)
    days <= 10:       "MEDIUM"  (within 2 weeks — size down, define risk with options)
    days <= 21:       "LOW"     (within a month — monitor)
    days > 21 or None: "CLEAR"
    """
    if days is None:
        return "CLEAR"
    if days <= 3:
        return "HIGH"
    if days <= 10:
        return "MEDIUM"
    if days <= 21:
        return "LOW"
    return "CLEAR"


def pre_earnings_window(days: int | None) -> str:
    """
    5 >= days >= 2:       "PRE_EARNINGS_IV_PLAY"   (IV expansion window — buy straddle/strangle)
    days == 1:            "FINAL_DAY"
    days == 0:            "EARNINGS_TODAY"
    -5 <= days < 0:       "POST_EARNINGS_DRIFT"     (drift opportunity window)
    else:                 ""
    """
    if days is None:
        return ""
    if 2 <= days <= 5:
        return "PRE_EARNINGS_IV_PLAY"
    if days == 1:
        return "FINAL_DAY"
    if days == 0:
        return "EARNINGS_TODAY"
    if -5 <= days < 0:
        return "POST_EARNINGS_DRIFT"
    return ""


def iv_expansion_quality(iv_rank: float | None, atm_iv: float | None) -> str:
    """
    Assess straddle entry quality based on IV rank.

    iv_rank < 40:   "EXCELLENT"  (IV cheap, good straddle entry)
    iv_rank 40-60:  "GOOD"
    iv_rank > 70:   "POOR"       (IV already elevated, straddle too expensive)
    None:           "UNKNOWN"

    atm_iv is used as a secondary sanity check when iv_rank is not available.
    """
    if iv_rank is not None and not np.isnan(float(iv_rank)):
        r = float(iv_rank)
        if r < 40:
            return "EXCELLENT"
        if r <= 60:
            return "GOOD"
        if r > 70:
            return "POOR"
        return "GOOD"  # 60-70 range
    # Fallback: use atm_iv as a raw proxy (>60% ATM IV = elevated)
    if atm_iv is not None and not np.isnan(float(atm_iv)):
        v = float(atm_iv)
        if v < 0.30:
            return "EXCELLENT"
        if v < 0.50:
            return "GOOD"
        return "POOR"
    return "UNKNOWN"


def recommended_action(
    risk_flag: str,
    earnings_window: str,
    iv_qual: str,
    surprise_pct: float | None,
    in_portfolio: bool,
) -> str:
    """
    Derive a human-readable recommended action for the ticker.

    Priority order:
      1. HIGH risk + in portfolio → warn to hedge/exit
      2. PRE_EARNINGS_IV_PLAY + EXCELLENT IV → straddle opportunity
      3. POST_EARNINGS_DRIFT + positive surprise → momentum entry
      4. EARNINGS_TODAY → hold, monitor
      5. FINAL_DAY → final hedge check
      6. CLEAR → no action needed
    """
    if risk_flag == "HIGH" and in_portfolio:
        return "⚠️ Consider exiting or hedging before earnings"
    if earnings_window == "PRE_EARNINGS_IV_PLAY" and iv_qual == "EXCELLENT":
        return "📈 Pre-earnings straddle opportunity"
    if earnings_window == "PRE_EARNINGS_IV_PLAY" and iv_qual == "GOOD":
        return "📈 Pre-earnings IV play — acceptable entry"
    if earnings_window == "POST_EARNINGS_DRIFT":
        pct = _safe_float(surprise_pct)
        if not np.isnan(pct) and pct > 3.0:
            return "🚀 Consider post-earnings momentum entry"
        if not np.isnan(pct) and pct < -3.0:
            return "⬇️ Post-earnings gap down — watch for dead-cat bounce"
        return "👀 Post-earnings drift window — monitor price action"
    if earnings_window == "EARNINGS_TODAY":
        return "🔔 Earnings today — hold, do not add"
    if earnings_window == "FINAL_DAY":
        return "⏰ Final day before earnings — last chance to hedge"
    if risk_flag == "HIGH":
        return "⚠️ Earnings imminent — no new positions"
    if risk_flag == "MEDIUM":
        return "⚡ Earnings within 2 weeks — size down / define risk"
    if risk_flag == "LOW":
        return "📅 Earnings within 1 month — monitor"
    if risk_flag == "CLEAR":
        return "✅ No earnings risk in 3 weeks"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# [2/3]  DATA LOADING & CALENDAR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def load_ticker_universe(top_n: int) -> tuple[list[str], dict[str, dict]]:
    """
    Load tickers from daily_picks.csv and daily_final.csv.
    Deduplicate and return up to top_n tickers (sorted by alpha_score desc).
    Also returns a dict of ticker metadata: alpha_score, action, in_portfolio.
    """
    print("\n[1/3] Loading ticker universe ...")

    meta: dict[str, dict] = {}

    picks_df = _load_csv(ROOT / "daily_picks.csv", "daily_picks")
    final_df = _load_csv(ROOT / "daily_final.csv", "daily_final")

    def _ingest(df: pd.DataFrame | None, source: str, in_portfolio: bool) -> None:
        if df is None:
            return
        alpha_col = "alpha_score" if "alpha_score" in df.columns else None
        action_col = "action" if "action" in df.columns else None
        for _, row in df.iterrows():
            tk = str(row["ticker"]).upper().strip()
            if not tk or tk == "NAN":
                continue
            alpha = _safe_float(row[alpha_col]) if alpha_col else np.nan
            action = str(row[action_col]).strip() if action_col else ""
            if tk not in meta:
                meta[tk] = {
                    "alpha_score": alpha,
                    "action": action,
                    "in_portfolio": in_portfolio,
                    "source": source,
                }
            else:
                # Update if better alpha available
                existing = meta[tk]
                if np.isnan(existing["alpha_score"]) and not np.isnan(alpha):
                    existing["alpha_score"] = alpha
                if not existing["in_portfolio"] and in_portfolio:
                    existing["in_portfolio"] = True

    _ingest(picks_df,  "daily_picks", in_portfolio=False)
    _ingest(final_df,  "daily_final", in_portfolio=True)
    # 事件系统底库=标普500全体: 并入 alpha_scores.csv, 让 N(催化临近度) 覆盖全池
    alpha_df = _load_csv(ROOT / "alpha_scores.csv", "alpha_scores")
    _ingest(alpha_df, "alpha_scores", in_portfolio=False)

    # Sort by alpha_score descending (unknown alpha goes to end)
    sorted_tickers = sorted(
        meta.keys(),
        key=lambda t: meta[t]["alpha_score"] if not np.isnan(meta[t]["alpha_score"]) else -999,
        reverse=True,
    )
    selected = sorted_tickers[:top_n]

    n_portfolio = sum(1 for t in selected if meta[t]["in_portfolio"])
    n_picks     = len(selected) - n_portfolio
    print(f"  Universe: {len(selected)} tickers ({n_picks} picks, {n_portfolio} portfolio)")
    return selected, meta


def load_earnings_dates(tickers: list[str]) -> dict[str, tuple[str, float, float]]:
    """
    Load earnings dates from earnings_surprise_scores.csv and
    earnings_revision_scores.csv. For each ticker, take the latest
    (most future or least-past) earnings_date found across both files.

    Returns dict: ticker -> (earnings_date_str, surprise_pct, revision_score)
    """
    print("  Loading earnings date sources ...")

    surprise_df  = _load_csv(ROOT / "earnings_surprise_scores.csv", "earnings_surprise_scores")
    revision_df  = _load_csv(ROOT / "earnings_revision_scores.csv", "earnings_revision_scores")

    # Normalise date column names
    def _get_date_col(df: pd.DataFrame | None) -> str | None:
        if df is None:
            return None
        for c in df.columns:
            if c.lower() in ("earnings_date", "report_date"):
                return c
        return None

    # Build lookup: ticker -> list of (date_str, surprise_pct, revision_score)
    date_candidates: dict[str, list[tuple[str, float, float]]] = {}

    if surprise_df is not None:
        dcol = _get_date_col(surprise_df)
        for _, row in surprise_df.iterrows():
            tk = str(row["ticker"]).upper().strip()
            if tk not in tickers:
                continue
            ds = str(row[dcol]).strip() if dcol else ""
            sp = _safe_float(row.get("surprise_pct", np.nan))
            date_candidates.setdefault(tk, []).append((ds, sp, np.nan))

    if revision_df is not None:
        dcol = _get_date_col(revision_df)
        for _, row in revision_df.iterrows():
            tk = str(row["ticker"]).upper().strip()
            if tk not in tickers:
                continue
            ds = str(row[dcol]).strip() if dcol else ""
            rs = _safe_float(row.get("revision_score", np.nan))
            # Merge with existing entry if present
            if tk in date_candidates:
                # Find matching date entry or append
                matched = False
                for entry in date_candidates[tk]:
                    if entry[0] == ds:
                        # Update revision_score in-place via list replacement
                        idx = date_candidates[tk].index(entry)
                        date_candidates[tk][idx] = (entry[0], entry[1], rs)
                        matched = True
                        break
                if not matched:
                    date_candidates[tk].append((ds, np.nan, rs))
            else:
                date_candidates[tk] = [(ds, np.nan, rs)]

    # Per ticker: choose the entry with the latest (most upcoming) date
    result: dict[str, tuple[str, float, float]] = {}
    for tk, entries in date_candidates.items():
        best: tuple[str, float, float] | None = None
        best_days: int | None = None
        for ds, sp, rs in entries:
            d = days_until_earnings(ds)
            if d is None:
                continue
            # Prefer the most upcoming (largest days value, even if negative)
            if best_days is None or d > best_days:
                best_days = d
                best = (ds, sp, rs)
        if best is not None:
            result[tk] = best

    found = len(result)
    missing = len([t for t in tickers if t not in result])
    print(f"  Earnings dates found: {found} tickers  |  no date: {missing} tickers")
    return result


def load_options_data(tickers: list[str]) -> dict[str, dict]:
    """
    Load options_signals.csv; return dict: ticker -> {iv_rank, atm_iv}.
    Tries iv_pct (0-100) as iv_rank, and atm_iv directly.
    """
    opts_df = _load_csv(ROOT / "options_signals.csv", "options_signals")
    lookup: dict[str, dict] = {}
    if opts_df is None:
        return lookup

    # Determine the iv_rank column — try iv_rank, then iv_pct
    iv_rank_col = None
    for c in ("iv_rank", "iv_pct"):
        if c in opts_df.columns:
            iv_rank_col = c
            break
    atm_iv_col = "atm_iv" if "atm_iv" in opts_df.columns else None

    for _, row in opts_df.iterrows():
        tk = str(row["ticker"]).upper().strip()
        if tk not in tickers:
            continue
        iv_r = _safe_float(row[iv_rank_col]) if iv_rank_col else np.nan
        atm  = _safe_float(row[atm_iv_col])  if atm_iv_col  else np.nan
        lookup[tk] = {"iv_rank": iv_r, "atm_iv": atm}

    return lookup


def run_earnings_calendar(top_n: int = 50, days_ahead: int = 30) -> pd.DataFrame:
    """
    Full earnings calendar pipeline.

    1. Load tickers from daily_picks.csv + daily_final.csv (deduplicated, top_n)
    2. Load earnings dates from surprise + revision score files
    3. For each ticker: compute days_until, risk_flag, window, IV quality
    4. Cross-reference with options_signals.csv for IV data
    5. Build and write earnings_calendar.csv
    6. Write earnings_calendar_report.md
    7. Print summary

    Returns the completed DataFrame.
    """
    # ── Load universe ──
    tickers, meta = load_ticker_universe(top_n)

    # ── Load earnings dates ──
    earnings_lookup = load_earnings_dates(tickers)

    # ── Load options data ──
    opts_lookup = load_options_data(tickers)

    print("  Building earnings calendar rows ...")

    rows: list[dict] = []
    for tk in tickers:
        tm = meta[tk]
        alpha  = tm["alpha_score"]
        action = tm["action"]
        in_pf  = tm["in_portfolio"]

        # Earnings date info
        if tk in earnings_lookup:
            edate_str, surprise_pct, revision_score = earnings_lookup[tk]
            days = days_until_earnings(edate_str)
        else:
            edate_str     = ""
            surprise_pct  = np.nan
            revision_score = np.nan
            days          = None

        # Computed flags
        risk_flag   = earnings_risk_flag(days)
        earn_window = pre_earnings_window(days)

        # Options data
        iv_data  = opts_lookup.get(tk, {})
        iv_rank  = iv_data.get("iv_rank", np.nan)
        atm_iv   = iv_data.get("atm_iv",  np.nan)

        iv_rank_val  = iv_rank if not np.isnan(_safe_float(iv_rank)) else None
        atm_iv_val   = atm_iv  if not np.isnan(_safe_float(atm_iv))  else None
        iv_qual      = iv_expansion_quality(iv_rank_val, atm_iv_val)

        rec = recommended_action(risk_flag, earn_window, iv_qual, surprise_pct, in_pf)

        rows.append({
            "ticker":               tk,
            "earnings_date":        edate_str,
            "days_until":           days if days is not None else np.nan,
            "risk_flag":            risk_flag,
            "earnings_window":      earn_window,
            "iv_rank":              round(_safe_float(iv_rank), 1) if not np.isnan(_safe_float(iv_rank)) else np.nan,
            "iv_expansion_quality": iv_qual,
            "alpha_score":          round(alpha, 2) if not np.isnan(alpha) else np.nan,
            "action":               action,
            "surprise_pct_last":    round(_safe_float(surprise_pct), 2) if not np.isnan(_safe_float(surprise_pct)) else np.nan,
            "revision_score":       round(_safe_float(revision_score), 2) if not np.isnan(_safe_float(revision_score)) else np.nan,
            "recommended_action":   rec,
        })

    df = pd.DataFrame(rows)

    # Sort by days_until ascending (NaN at end), then alpha_score desc
    df["_days_sort"] = df["days_until"].apply(lambda x: x if not (isinstance(x, float) and np.isnan(x)) else 9999)
    df = df.sort_values(["_days_sort", "alpha_score"], ascending=[True, False]).drop(columns=["_days_sort"]).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# [3/3]  OUTPUT WRITERS & SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(df: pd.DataFrame) -> None:
    """Write earnings_calendar.csv with canonical column order."""
    col_order = [
        "ticker", "earnings_date", "days_until", "risk_flag", "earnings_window",
        "iv_rank", "iv_expansion_quality", "alpha_score", "action",
        "surprise_pct_last", "revision_score", "recommended_action",
    ]
    out_cols = [c for c in col_order if c in df.columns]
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"  CSV saved ({len(df)} rows): {OUT_CSV.name}")


def write_report(df: pd.DataFrame, days_ahead: int, top_n: int) -> None:
    """Write earnings_calendar_report.md — markdown table sorted by days_until."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Compute summary stats
    n_high    = (df["risk_flag"] == "HIGH").sum()
    n_medium  = (df["risk_flag"] == "MEDIUM").sum()
    n_low     = (df["risk_flag"] == "LOW").sum()
    n_clear   = (df["risk_flag"] == "CLEAR").sum()
    n_iv_play = (df["earnings_window"] == "PRE_EARNINGS_IV_PLAY").sum()
    n_drift   = (df["earnings_window"] == "POST_EARNINGS_DRIFT").sum()
    n_today   = (df["earnings_window"] == "EARNINGS_TODAY").sum()

    # Filter to days_ahead window for the main table (all rows in full table)
    display_df = df[
        df["days_until"].apply(
            lambda x: (not (isinstance(x, float) and np.isnan(x))) and float(x) <= days_ahead
        )
    ].copy()

    lines: list[str] = []
    lines.append("# Canyon v9 — Earnings Calendar Risk Overlay")
    lines.append(f"_Generated: {now_str} | Universe: top-{top_n} | Showing: ≤{days_ahead} days ahead_\n")
    lines.append("---")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Risk Level | Count | Description |")
    lines.append("|------------|-------|-------------|")
    lines.append(f"| HIGH | {n_high} | Earnings in ≤3 days — do not initiate |")
    lines.append(f"| MEDIUM | {n_medium} | Earnings in ≤10 days — size down |")
    lines.append(f"| LOW | {n_low} | Earnings in ≤21 days — monitor |")
    lines.append(f"| CLEAR | {n_clear} | No near-term earnings risk |")
    lines.append("")
    lines.append("| Opportunity | Count |")
    lines.append("|-------------|-------|")
    lines.append(f"| Pre-earnings IV plays | {n_iv_play} |")
    lines.append(f"| Post-earnings drift windows | {n_drift} |")
    lines.append(f"| Earnings today | {n_today} |")
    lines.append("")
    lines.append("---")

    # Main calendar table
    lines.append(f"## Earnings Calendar (next {days_ahead} days)")
    lines.append("")

    if display_df.empty:
        lines.append(f"_No earnings events within {days_ahead} days for current universe._\n")
    else:
        lines.append(
            "| # | Ticker | Earnings Date | Days | Risk Flag | Window | "
            "IV Rank | IV Quality | Alpha | Action | Surprise% | Rev Score | Recommended Action |"
        )
        lines.append(
            "|---|--------|---------------|------|-----------|--------|"
            "---------|------------|-------|--------|-----------|-----------|-------------------|"
        )

        for i, (_, row) in enumerate(display_df.iterrows(), 1):
            def _fmt(v, dec: int = 1) -> str:
                return f"{float(v):.{dec}f}" if pd.notna(v) and not (isinstance(v, float) and np.isnan(v)) else "—"

            days_str = _fmt(row["days_until"], 0) if pd.notna(row.get("days_until")) else "—"
            earn_dt  = str(row["earnings_date"]) if row["earnings_date"] else "—"
            window   = str(row["earnings_window"]) if row["earnings_window"] else "—"

            lines.append(
                f"| {i} "
                f"| **{row['ticker']}** "
                f"| {earn_dt} "
                f"| {days_str} "
                f"| {row['risk_flag']} "
                f"| {window} "
                f"| {_fmt(row.get('iv_rank'))} "
                f"| {row.get('iv_expansion_quality', '—')} "
                f"| {_fmt(row.get('alpha_score'))} "
                f"| {row.get('action') or '—'} "
                f"| {_fmt(row.get('surprise_pct_last'))} "
                f"| {_fmt(row.get('revision_score'))} "
                f"| {row.get('recommended_action', '')} |"
            )
        lines.append("")

    # High-priority section
    high_df = df[df["risk_flag"] == "HIGH"]
    if not high_df.empty:
        lines.append("---")
        lines.append("## High Risk Tickers (Earnings in ≤3 Days)")
        lines.append("")
        lines.append("> These tickers have earnings within 3 days. "
                     "**Do not initiate new positions.** Review existing exposure immediately.")
        lines.append("")
        for _, row in high_df.iterrows():
            days_val = int(row["days_until"]) if pd.notna(row.get("days_until")) else "?"
            pf_note  = " (IN PORTFOLIO)" if row.get("action", "") in ("STRONG BUY", "BUY") else ""
            lines.append(f"- **{row['ticker']}** — {row['earnings_date']} "
                         f"({days_val} days){pf_note}  →  {row['recommended_action']}")
        lines.append("")

    # IV play section
    iv_df = df[df["earnings_window"] == "PRE_EARNINGS_IV_PLAY"]
    if not iv_df.empty:
        lines.append("---")
        lines.append("## Pre-Earnings IV Play Candidates")
        lines.append("")
        lines.append("> IV expansion window (2–5 days before earnings). "
                     "Consider long straddle/strangle for EXCELLENT-rated IV only.")
        lines.append("")
        lines.append("| Ticker | Days | IV Rank | IV Quality | Recommended |")
        lines.append("|--------|------|---------|------------|-------------|")
        for _, row in iv_df.iterrows():
            def _fmtv(v, d=1):
                return f"{float(v):.{d}f}" if pd.notna(v) else "—"
            lines.append(
                f"| **{row['ticker']}** "
                f"| {_fmtv(row.get('days_until'), 0)} "
                f"| {_fmtv(row.get('iv_rank'))} "
                f"| {row.get('iv_expansion_quality', '—')} "
                f"| {row.get('recommended_action', '')} |"
            )
        lines.append("")

    # Post-earnings drift section
    drift_df = df[df["earnings_window"] == "POST_EARNINGS_DRIFT"]
    if not drift_df.empty:
        lines.append("---")
        lines.append("## Post-Earnings Drift Candidates")
        lines.append("")
        lines.append("> Within 5 days after earnings. "
                     "Watch for post-earnings momentum continuation.")
        lines.append("")
        lines.append("| Ticker | Days | Surprise% | IV Quality | Recommended |")
        lines.append("|--------|------|-----------|------------|-------------|")
        for _, row in drift_df.iterrows():
            def _fmtd(v, d=1):
                return f"{float(v):.{d}f}" if pd.notna(v) else "—"
            lines.append(
                f"| **{row['ticker']}** "
                f"| {_fmtd(row.get('days_until'), 0)} "
                f"| {_fmtd(row.get('surprise_pct_last'))} "
                f"| {row.get('iv_expansion_quality', '—')} "
                f"| {row.get('recommended_action', '')} |"
            )
        lines.append("")

    # Full table (all tickers, regardless of days_ahead filter)
    lines.append("---")
    lines.append("## Full Universe Table")
    lines.append("")
    lines.append("_All tickers sorted by days until earnings (ascending). "
                 "Tickers with no earnings date shown at end._")
    lines.append("")
    lines.append(
        "| # | Ticker | Earnings Date | Days | Risk Flag | Window | "
        "Alpha | Action | Surprise% | Rev Score | Recommended |"
    )
    lines.append(
        "|---|--------|---------------|------|-----------|--------|"
        "-------|--------|-----------|-----------|-------------|"
    )
    for i, (_, row) in enumerate(df.iterrows(), 1):
        def _fmtf(v, d=1):
            return f"{float(v):.{d}f}" if pd.notna(v) and not (isinstance(v, float) and np.isnan(v)) else "—"
        days_s = _fmtf(row["days_until"], 0) if pd.notna(row.get("days_until")) else "—"
        earn_dt = str(row["earnings_date"]) if row["earnings_date"] else "—"
        window  = str(row["earnings_window"]) if row["earnings_window"] else "—"
        lines.append(
            f"| {i} "
            f"| **{row['ticker']}** "
            f"| {earn_dt} "
            f"| {days_s} "
            f"| {row['risk_flag']} "
            f"| {window} "
            f"| {_fmtf(row.get('alpha_score'))} "
            f"| {row.get('action') or '—'} "
            f"| {_fmtf(row.get('surprise_pct_last'))} "
            f"| {_fmtf(row.get('revision_score'))} "
            f"| {row.get('recommended_action', '')} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("## Risk Definitions")
    lines.append("")
    lines.append("| Flag | Days Until Earnings | Action |")
    lines.append("|------|---------------------|--------|")
    lines.append("| HIGH | ≤ 3 days | Do not initiate new positions |")
    lines.append("| MEDIUM | ≤ 10 days | Size down; define risk with options |")
    lines.append("| LOW | ≤ 21 days | Monitor; be aware of event risk |")
    lines.append("| CLEAR | > 21 days | No near-term earnings risk |")
    lines.append("")
    lines.append("| Window | Description |")
    lines.append("|--------|-------------|")
    lines.append("| PRE_EARNINGS_IV_PLAY | 2–5 days before: IV expansion window |")
    lines.append("| FINAL_DAY | 1 day before earnings |")
    lines.append("| EARNINGS_TODAY | Day of earnings |")
    lines.append("| POST_EARNINGS_DRIFT | 1–5 days after: drift opportunity |")
    lines.append("")
    lines.append("| IV Quality | IV Rank | Straddle Assessment |")
    lines.append("|------------|---------|---------------------|")
    lines.append("| EXCELLENT | < 40 | IV is cheap — good straddle entry |")
    lines.append("| GOOD | 40–60 | IV moderate — acceptable |")
    lines.append("| POOR | > 70 | IV already elevated — straddle too expensive |")
    lines.append("| UNKNOWN | — | No IV data available |")
    lines.append("")
    lines.append("---")
    lines.append("_Canyon v9 — research use only. Not investment advice. No live orders._")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {OUT_REPORT.name}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary to stdout."""
    n_high   = (df["risk_flag"] == "HIGH").sum()
    n_medium = (df["risk_flag"] == "MEDIUM").sum()
    n_low    = (df["risk_flag"] == "LOW").sum()
    n_clear  = (df["risk_flag"] == "CLEAR").sum()
    n_iv     = (df["earnings_window"] == "PRE_EARNINGS_IV_PLAY").sum()
    n_drift  = (df["earnings_window"] == "POST_EARNINGS_DRIFT").sum()
    n_today  = (df["earnings_window"] == "EARNINGS_TODAY").sum()
    n_at_risk = n_high + n_medium + n_low

    print(f"\n{'='*70}")
    print("  CANYON v9 — EARNINGS CALENDAR RISK OVERLAY")
    print(f"{'='*70}")
    print(f"  Total universe: {len(df)} tickers")
    print(f"  At risk (HIGH/MEDIUM/LOW): {n_at_risk}")
    print(f"    HIGH   (≤3 days):  {n_high}")
    print(f"    MEDIUM (≤10 days): {n_medium}")
    print(f"    LOW    (≤21 days): {n_low}")
    print(f"  CLEAR (>21 days):   {n_clear}")
    print(f"  Pre-earnings IV plays:      {n_iv}")
    print(f"  Post-earnings drift windows:{n_drift}")
    print(f"  Earnings today:             {n_today}")
    print(f"{'='*70}")

    # Show high-risk tickers
    high_df = df[df["risk_flag"] == "HIGH"]
    if not high_df.empty:
        print("\n  HIGH RISK TICKERS (do not open new positions):")
        for _, row in high_df.iterrows():
            days_s = f"{int(row['days_until'])}d" if pd.notna(row.get("days_until")) else "?"
            print(f"    {row['ticker']:<8} {row['earnings_date']:<12} ({days_s})  "
                  f"{row['recommended_action']}")

    # Show IV plays
    iv_df = df[df["earnings_window"] == "PRE_EARNINGS_IV_PLAY"]
    if not iv_df.empty:
        print("\n  PRE-EARNINGS IV PLAY CANDIDATES:")
        for _, row in iv_df.iterrows():
            days_s = f"{int(row['days_until'])}d" if pd.notna(row.get("days_until")) else "?"
            iv_s   = f"iv={row['iv_rank']:.0f}" if pd.notna(row.get("iv_rank")) else "iv=?"
            print(f"    {row['ticker']:<8} {row['earnings_date']:<12} ({days_s})  "
                  f"{iv_s}  [{row['iv_expansion_quality']}]  {row['recommended_action']}")

    # Show drift opportunities
    drift_df = df[df["earnings_window"] == "POST_EARNINGS_DRIFT"]
    if not drift_df.empty:
        print("\n  POST-EARNINGS DRIFT WINDOWS:")
        for _, row in drift_df.iterrows():
            days_s = f"{int(row['days_until'])}d" if pd.notna(row.get("days_until")) else "?"
            sp_s   = f"surp={row['surprise_pct_last']:.1f}%" if pd.notna(row.get("surprise_pct_last")) else "surp=?"
            print(f"    {row['ticker']:<8} {row['earnings_date']:<12} ({days_s})  "
                  f"{sp_s}  {row['recommended_action']}")

    print(f"\n  Outputs:")
    print(f"    {OUT_CSV}")
    print(f"    {OUT_REPORT}")
    print(f"\n  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 102 — Earnings Calendar Risk Overlay (research only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of tickers to include from the pick/portfolio universe",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=30,
        metavar="N",
        help="Show earnings events within N days in the primary report table",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Canyon v9 — Step 102: Earnings Calendar Risk Overlay")
    print("  Research use only. No live orders.")
    print("=" * 60)
    print(f"  Settings: top={args.top}  days-ahead={args.days_ahead}")
    print(f"  Today: {TODAY}")

    df = run_earnings_calendar(top_n=args.top, days_ahead=args.days_ahead)

    print(f"\n[2/3] Writing output files ...")
    write_csv(df)
    write_report(df, days_ahead=args.days_ahead, top_n=args.top)

    print(f"\n[3/3] Summary ...")
    print_summary(df)


if __name__ == "__main__":
    main()
