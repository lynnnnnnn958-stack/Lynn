#!/usr/bin/env python3
"""
Canyon v9 — Step 101: Short / Sell Signal Generator
====================================================
Generates a SHORT CANDIDATES list — stocks where multiple independent signals
are simultaneously bearish. For research and paper trading only. No live orders.

Philosophy
----------
Canyon v9 is primarily long-focused. Short signals are generated ONLY when
MULTIPLE independent conditions all point bearish at the same time — never
on a single signal alone. Minimum 3 bearish conditions required.

Short condition weights
-----------------------
  low_alpha          2.0   Alpha score < 25 (bottom quartile)
  ml_bearish         2.0   ML regime-conditional score < 25
  earnings_cut       1.5   EPS estimates being revised down
  earnings_miss      1.5   Recent earnings miss
  bearish_options    1.5   High PCR + negative GEX
  insider_selling    1.5   Insider selling detected
  negative_sentiment 1.0   Negative NLP sentiment in filings
  weak_quality       1.0   Poor fundamentals (leverage, weak margins)
  seasonal_bear      0.5   Sector historically weak this month
  high_short_interest 0.5  Already heavily shorted (squeeze risk note)

  SHORT CANDIDATE  : weighted score >= 4.0  AND  n_conditions >= 3
  STRONG SHORT     : weighted score >= 6.0

Output files
------------
  short_candidates.csv        — ranked candidate list
  short_candidates_report.md  — markdown table + methodology + risk warnings

Usage
-----
  python3 canyon_final_v9_step101_short_signals.py
  python3 canyon_final_v9_step101_short_signals.py --top 30
  python3 canyon_final_v9_step101_short_signals.py --threshold 5.0
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

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_CSV    = ROOT / "short_candidates.csv"
OUT_REPORT = ROOT / "short_candidates_report.md"

# ── Condition registry ────────────────────────────────────────────────────────
SHORT_CONDITIONS: dict[str, dict] = {
    "low_alpha":           {"weight": 2.0, "desc": "Alpha score < 25 (bottom quartile)"},
    "ml_bearish":          {"weight": 2.0, "desc": "ML regime score < 25"},
    "earnings_cut":        {"weight": 1.5, "desc": "EPS estimates being revised down"},
    "earnings_miss":       {"weight": 1.5, "desc": "Recent earnings miss"},
    "bearish_options":     {"weight": 1.5, "desc": "Bearish options flow (high PCR, negative GEX)"},
    "insider_selling":     {"weight": 1.5, "desc": "Insider selling detected"},
    "negative_sentiment":  {"weight": 1.0, "desc": "Negative NLP sentiment in filings"},
    "weak_quality":        {"weight": 1.0, "desc": "Poor fundamentals (leverage, weak margins)"},
    "seasonal_bear":       {"weight": 0.5, "desc": "Sector historically weak this month"},
    "high_short_interest": {"weight": 0.5, "desc": "Already heavily shorted (crowded short risk)"},
}

# Thresholds
SHORT_THRESHOLD        = 4.0
STRONG_SHORT_THRESHOLD = 6.0
MIN_CONDITIONS         = 3   # must trigger at least this many conditions

# Pre-filter boundaries
UNIVERSE_ALPHA_MAX   = 35.0  # only consider tickers with alpha_score < 35
BUY_ALPHA_MIN        = 60.0  # never short tickers on active buy list
MAX_SHORT_PCT_FLOAT  = 30.0  # skip crowded shorts (squeeze risk > 30 % float)


# ─────────────────────────────────────────────────────────────────────────────
# [1/4]  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load CSV gracefully; return None if missing or unreadable."""
    if not path.exists():
        print(f"  [SKIP] {label} not found ({path.name})")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {label} is empty")
            return None
        # Normalise ticker column name
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


def _to_lookup(df: pd.DataFrame | None) -> dict[str, pd.Series]:
    """Convert a DataFrame into a dict of {ticker: row_as_Series}."""
    if df is None:
        return {}
    return {row["ticker"]: row for _, row in df.iterrows()}


def load_all_signal_files() -> dict[str, pd.DataFrame | None]:
    """Load every signal CSV; keys match the variable names used in evaluation."""
    print("\n[1/4] Loading signal files ...")
    files: dict[str, tuple[Path, str]] = {
        "alpha":     (ROOT / "alpha_scores.csv",            "Alpha scores"),
        "regime_ml": (ROOT / "regime_ml_scores.csv",        "Regime ML scores"),
        "options":   (ROOT / "options_signals.csv",         "Options signals"),
        # try short_squeeze_signals.csv first, fall back to short_interest_scores.csv
        "short_int": (ROOT / "short_squeeze_signals.csv",   "Short squeeze signals"),
        "revision":  (ROOT / "earnings_revision_scores.csv","Earnings revision scores"),
        "surprise":  (ROOT / "earnings_surprise_scores.csv","Earnings surprise scores"),
        "insider":   (ROOT / "insider_signal_scores.csv",   "Insider signal scores"),
        "sentiment": (ROOT / "finbert_sentiment.csv",       "FinBERT sentiment"),
        "quality":   (ROOT / "fundamental_quality_rank.csv","Fundamental quality rank"),
        "sector":    (ROOT / "sector_map.csv",              "Sector map"),
    }

    dfs: dict[str, pd.DataFrame | None] = {}
    for key, (path, label) in files.items():
        dfs[key] = _load_csv(path, label)

    # Fallback: short_interest_scores.csv if short_squeeze not present
    if dfs["short_int"] is None:
        alt = ROOT / "short_interest_scores.csv"
        dfs["short_int"] = _load_csv(alt, "Short interest scores (fallback)")

    return dfs


def load_sector_bias() -> dict[str, list[str]]:
    """
    Load current_month_sector_bias.json.
    Expected structure: {"bottom_sectors": ["Energy", "Utilities"], ...}
    Returns empty dict on missing/error.
    """
    path = ROOT / "current_month_sector_bias.json"
    if not path.exists():
        print("  [SKIP] current_month_sector_bias.json not found")
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        bottom = data.get("bottom_sectors", [])
        if bottom:
            print(f"  Sector bias loaded — bottom sectors: {bottom}")
        return data
    except Exception as exc:
        print(f"  [WARN] Could not load sector bias: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# [2/4]  UNIVERSE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_short_universe(
    alpha_df: pd.DataFrame | None,
    short_int_df: pd.DataFrame | None,
) -> list[str]:
    """
    Pre-filter candidates:
      - Only tickers with alpha_score < UNIVERSE_ALPHA_MAX (35)
      - Exclude tickers on the active BUY list (alpha_score >= BUY_ALPHA_MIN = 60)
      - Exclude crowded shorts: short_pct_float > MAX_SHORT_PCT_FLOAT (30%)
    Returns list of candidate tickers.
    """
    print("\n[2/4] Building short universe ...")

    if alpha_df is None or alpha_df.empty:
        print("  [ERROR] alpha_scores.csv required for universe build.")
        return []

    # Gate 1: alpha_score must be < UNIVERSE_ALPHA_MAX
    if "alpha_score" not in alpha_df.columns:
        print("  [ERROR] alpha_scores.csv missing 'alpha_score' column.")
        return []

    alpha_df = alpha_df.copy()
    alpha_df["alpha_score"] = pd.to_numeric(alpha_df["alpha_score"], errors="coerce")

    universe = alpha_df[alpha_df["alpha_score"] < UNIVERSE_ALPHA_MAX]["ticker"].tolist()
    print(f"  After alpha < {UNIVERSE_ALPHA_MAX} filter: {len(universe)} tickers")

    # Gate 2: exclude active buy list
    buy_set = set(
        alpha_df.loc[alpha_df["alpha_score"] >= BUY_ALPHA_MIN, "ticker"].tolist()
    )
    universe = [t for t in universe if t not in buy_set]
    print(f"  After excluding buy-list (alpha >= {BUY_ALPHA_MIN}): {len(universe)} tickers")

    # Gate 3: exclude crowded shorts (squeeze risk)
    if short_int_df is not None and "short_pct_float" in short_int_df.columns:
        short_int_df = short_int_df.copy()
        short_int_df["short_pct_float"] = pd.to_numeric(
            short_int_df["short_pct_float"], errors="coerce"
        )
        crowded = set(
            short_int_df.loc[
                short_int_df["short_pct_float"] * 100 > MAX_SHORT_PCT_FLOAT, "ticker"
            ].tolist()
        )
        removed = len([t for t in universe if t in crowded])
        universe = [t for t in universe if t not in crowded]
        if removed:
            print(f"  Excluded {removed} crowded-short tickers (float > {MAX_SHORT_PCT_FLOAT}%)")

    print(f"  Final short universe: {len(universe)} candidates")
    return universe


# ─────────────────────────────────────────────────────────────────────────────
# [3/4]  CONDITION EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_short_conditions(
    ticker: str,
    data: dict[str, pd.Series | None],
    bottom_sectors: list[str],
) -> tuple[float, list[str]]:
    """
    Evaluate all SHORT_CONDITIONS for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    data : dict[str, pd.Series | None]
        Keyed by file alias (alpha, regime_ml, options, short_int, revision,
        surprise, insider, sentiment, quality, sector).
    bottom_sectors : list[str]
        Sectors flagged bearish this month.

    Returns
    -------
    (total_score, triggered_condition_descriptions)
    """
    triggered: list[str] = []
    total_score = 0.0

    def _add(condition_key: str) -> None:
        nonlocal total_score
        info = SHORT_CONDITIONS[condition_key]
        total_score += info["weight"]
        triggered.append(f"{condition_key}({info['weight']:.1f}): {info['desc']}")

    alpha_row    = data.get("alpha")
    ml_row       = data.get("regime_ml")
    opt_row      = data.get("options")
    si_row       = data.get("short_int")
    rev_row      = data.get("revision")
    sur_row      = data.get("surprise")
    ins_row      = data.get("insider")
    sent_row     = data.get("sentiment")
    qual_row     = data.get("quality")
    sector_row   = data.get("sector")

    # ── Condition 1: low_alpha ──
    if alpha_row is not None:
        ascore = _safe_float(alpha_row.get("alpha_score", np.nan))
        if not np.isnan(ascore) and ascore < 25:
            _add("low_alpha")

    # ── Condition 2: ml_bearish ──
    # Check regime_ml_scores first, fall back to alpha_scores sig_regime_ml
    ml_score = np.nan
    if ml_row is not None:
        ml_score = _safe_float(ml_row.get("predicted_score", np.nan))
        if np.isnan(ml_score):
            ml_score = _safe_float(ml_row.get("signal_score", np.nan))
    if np.isnan(ml_score) and alpha_row is not None:
        ml_score = _safe_float(alpha_row.get("sig_regime_ml", np.nan))
    if not np.isnan(ml_score) and ml_score < 25:
        _add("ml_bearish")
    # Also catch explicit SELL signal
    elif ml_row is not None:
        sig = str(ml_row.get("signal", "")).upper()
        if sig == "SELL" and "ml_bearish" not in [t.split("(")[0] for t in triggered]:
            _add("ml_bearish")

    # ── Condition 3: earnings_cut ──
    if rev_row is not None:
        rev_score = _safe_float(rev_row.get("revision_score", np.nan))
        if np.isnan(rev_score):
            rev_score = _safe_float(rev_row.get("rank_revision", np.nan))
        if not np.isnan(rev_score) and rev_score < 30:
            _add("earnings_cut")

    # ── Condition 4: earnings_miss ──
    if sur_row is not None:
        rank_sue = _safe_float(sur_row.get("rank_sue", np.nan))
        if not np.isnan(rank_sue) and rank_sue < 25:
            _add("earnings_miss")

    # ── Condition 5: bearish_options ──
    if opt_row is not None:
        pcr    = _safe_float(opt_row.get("pcr_vol", np.nan))
        gex    = str(opt_row.get("gex_sign", "")).lower()
        bear_f = str(opt_row.get("uoa_bear_flag", "")).lower()
        strat  = str(opt_row.get("strategy", "")).upper()

        pcr_bear  = (not np.isnan(pcr)) and (pcr > 1.3)
        gex_bear  = gex == "negative"
        uoa_bear  = bear_f in ("1", "true", "yes")
        strad     = strat == "LONG_STRADDLE"  # uncertainty / hedging signal

        if pcr_bear or gex_bear or uoa_bear or strad:
            _add("bearish_options")

    # ── Condition 6: insider_selling ──
    if ins_row is not None:
        ins_signal = str(ins_row.get("insider_signal", "")).upper()
        rank_ins   = _safe_float(ins_row.get("rank_insider", np.nan))
        sell_count = _safe_float(ins_row.get("sell_count", np.nan))
        net_dir    = _safe_float(ins_row.get("net_direction", np.nan))

        is_selling = (
            "SELL" in ins_signal
            or (not np.isnan(rank_ins) and rank_ins < 25)
            or (not np.isnan(sell_count) and sell_count > 0 and
                (np.isnan(ins_row.get("buy_count", np.nan)) or
                 _safe_float(ins_row.get("buy_count", np.nan)) == 0))
            or (not np.isnan(net_dir) and net_dir < -0.1)
        )
        if is_selling:
            _add("insider_selling")

    # ── Condition 7: negative_sentiment ──
    if sent_row is not None:
        rank_sent = _safe_float(sent_row.get("rank_sentiment", np.nan))
        label     = str(sent_row.get("label", "")).upper()
        if (not np.isnan(rank_sent) and rank_sent < 25) or label == "BEARISH":
            _add("negative_sentiment")

    # ── Condition 8: weak_quality ──
    if qual_row is not None:
        qscore = _safe_float(qual_row.get("quality_score", np.nan))
        if not np.isnan(qscore) and qscore < 25:
            _add("weak_quality")

    # ── Condition 9: seasonal_bear ──
    if sector_row is not None and bottom_sectors:
        sector = str(sector_row.get("sector", "")).strip()
        if sector and sector in bottom_sectors:
            _add("seasonal_bear")

    # ── Condition 10: high_short_interest ──
    if si_row is not None:
        si_pct  = _safe_float(si_row.get("short_pct_float", np.nan))
        si_sig  = str(si_row.get("signal", "")).upper()
        rank_sq = _safe_float(si_row.get("rank_squeeze", np.nan))

        # High short interest = already heavily shorted (note: crowded, squeeze risk)
        # Only add for moderate short interest (10–30% float), crowded >30% was excluded upstream
        if (not np.isnan(si_pct) and 0.10 <= si_pct <= 0.30) or "SHORT" in si_sig:
            _add("high_short_interest")

    return total_score, triggered


# ─────────────────────────────────────────────────────────────────────────────
# [4/4]  MAIN COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_short_signals(top_n: int = 50, threshold: float = SHORT_THRESHOLD) -> pd.DataFrame:
    """
    Full short-signal pipeline.

    1. Load all signal files
    2. Build short universe (alpha < 35)
    3. Score each ticker against all conditions
    4. Filter: score >= threshold AND n_conditions >= MIN_CONDITIONS
    5. Classify: STRONG SHORT / SHORT
    6. Assign suggested vehicle
    7. Return sorted DataFrame (highest score first, top_n rows)
    """
    # ── Load data ──
    dfs = load_all_signal_files()
    sector_bias = load_sector_bias()
    bottom_sectors: list[str] = sector_bias.get("bottom_sectors", [])

    alpha_df    = dfs.get("alpha")
    short_int_df = dfs.get("short_int")

    # ── Build universe ──
    universe = build_short_universe(alpha_df, short_int_df)
    if not universe:
        print("\n[WARN] Empty short universe — no candidates to score.")
        return pd.DataFrame()

    # Build per-ticker lookups for all signal files
    lookups: dict[str, dict[str, pd.Series]] = {
        key: _to_lookup(df) for key, df in dfs.items()
    }

    print(f"\n[3/4] Scoring {len(universe)} universe tickers ...")

    rows: list[dict] = []
    for ticker in universe:
        # Assemble per-ticker data dict
        ticker_data = {key: lkp.get(ticker) for key, lkp in lookups.items()}

        score, conditions = evaluate_short_conditions(ticker, ticker_data, bottom_sectors)
        n_cond = len(conditions)

        if score < threshold or n_cond < MIN_CONDITIONS:
            continue  # does not meet short threshold

        # ── Classify action ──
        if score >= STRONG_SHORT_THRESHOLD:
            action = "STRONG SHORT"
        else:
            action = "SHORT"

        # ── Suggested vehicle ──
        has_high_si = any("high_short_interest" in c for c in conditions)
        if has_high_si:
            vehicle = "Long put spread only (avoid naked short)"
        elif action == "STRONG SHORT":
            vehicle = "Buy put options / short stock"
        else:
            vehicle = "Buy put spread (defined risk)"

        # ── Extract diagnostic columns ──
        alpha_row  = ticker_data.get("alpha")
        opt_row    = ticker_data.get("options")
        rev_row    = ticker_data.get("revision")
        sur_row    = ticker_data.get("surprise")
        sent_row   = ticker_data.get("sentiment")
        qual_row   = ticker_data.get("quality")
        sec_row    = ticker_data.get("sector")

        alpha_score  = _safe_float(alpha_row.get("alpha_score", np.nan)) if alpha_row is not None else np.nan
        sector       = str(sec_row.get("sector", "")) if sec_row is not None else ""
        if not sector and alpha_row is not None:
            sector = str(alpha_row.get("sector", ""))

        pcr_vol      = _safe_float(opt_row.get("pcr_vol", np.nan))    if opt_row  is not None else np.nan
        gex_sign     = str(opt_row.get("gex_sign", ""))               if opt_row  is not None else ""
        rev_score    = _safe_float(rev_row.get("revision_score", _safe_float(rev_row.get("rank_revision", np.nan)))) \
                       if rev_row is not None else np.nan
        sur_rank     = _safe_float(sur_row.get("rank_sue", np.nan))   if sur_row  is not None else np.nan
        sent_rank    = _safe_float(sent_row.get("rank_sentiment", np.nan)) if sent_row is not None else np.nan
        qual_score   = _safe_float(qual_row.get("quality_score", np.nan)) if qual_row is not None else np.nan

        rows.append({
            "ticker":          ticker,
            "short_score":     round(score, 2),
            "short_action":    action,
            "n_conditions":    n_cond,
            "sector":          sector,
            "alpha_score":     round(alpha_score, 2) if not np.isnan(alpha_score) else np.nan,
            "conditions_text": " | ".join(conditions),
            "suggested_vehicle": vehicle,
            "pcr_vol":         round(pcr_vol, 3) if not np.isnan(pcr_vol) else np.nan,
            "gex_sign":        gex_sign,
            "revision_score":  round(rev_score, 2) if not np.isnan(rev_score) else np.nan,
            "surprise_rank":   round(sur_rank, 2) if not np.isnan(sur_rank) else np.nan,
            "sentiment_rank":  round(sent_rank, 2) if not np.isnan(sent_rank) else np.nan,
            "quality_score":   round(qual_score, 2) if not np.isnan(qual_score) else np.nan,
        })

    if not rows:
        print("\n  No tickers met the short threshold.")
        return pd.DataFrame()

    result_df = (
        pd.DataFrame(rows)
        .sort_values("short_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, threshold: float, top_n: int) -> None:
    """Write short_candidates_report.md."""
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_strong = (df["short_action"] == "STRONG SHORT").sum() if not df.empty else 0
    n_short  = (df["short_action"] == "SHORT").sum() if not df.empty else 0

    lines: list[str] = []
    lines.append("# Canyon v9 — Short Candidates Report")
    lines.append(f"_Generated: {now_str} | Threshold: {threshold} | Top-N: {top_n}_\n")

    lines.append("---")
    lines.append("## Risk Warnings")
    lines.append("")
    lines.append("> **Short selling involves unlimited loss potential. "
                 "These are research signals only.**")
    lines.append(">")
    lines.append("> **High short interest stocks (>20% float) carry significant squeeze risk.**")
    lines.append(">")
    lines.append("> **Always use defined-risk structures (put spreads) rather than naked shorts.**")
    lines.append("")

    lines.append("---")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Stat | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Candidates | {len(df)} |")
    lines.append(f"| STRONG SHORT (score >= {STRONG_SHORT_THRESHOLD}) | {n_strong} |")
    lines.append(f"| SHORT (score {threshold}–{STRONG_SHORT_THRESHOLD}) | {n_short} |")
    lines.append(f"| Threshold applied | {threshold} |")
    lines.append(f"| Min conditions required | {MIN_CONDITIONS} |")
    lines.append("")

    if df.empty:
        lines.append("_No short candidates above threshold at this time._\n")
    else:
        lines.append("---")
        lines.append("## Short Candidates Table")
        lines.append("")

        # Table header
        lines.append("| # | Ticker | Score | Action | Conditions | Sector | "
                      "Alpha | PCR | GEX | Rev Score | Sur Rank | Sent Rank | "
                      "Quality | Vehicle |")
        lines.append("|---|--------|-------|--------|------------|--------|"
                     "-------|-----|-----|-----------|----------|-----------|"
                     "---------|---------|")

        for i, row in df.iterrows():
            def _fmt(v, dec=1):
                return f"{v:.{dec}f}" if pd.notna(v) else "—"

            lines.append(
                f"| {i+1} "
                f"| **{row['ticker']}** "
                f"| {row['short_score']:.2f} "
                f"| {row['short_action']} "
                f"| {row['n_conditions']} signals "
                f"| {row['sector'] or '—'} "
                f"| {_fmt(row.get('alpha_score'))} "
                f"| {_fmt(row.get('pcr_vol'), 2)} "
                f"| {row.get('gex_sign') or '—'} "
                f"| {_fmt(row.get('revision_score'))} "
                f"| {_fmt(row.get('surprise_rank'))} "
                f"| {_fmt(row.get('sentiment_rank'))} "
                f"| {_fmt(row.get('quality_score'))} "
                f"| {row['suggested_vehicle']} |"
            )
        lines.append("")

        # Conditions detail section
        lines.append("---")
        lines.append("## Conditions Detail")
        lines.append("")
        for _, row in df.iterrows():
            lines.append(f"### {row['ticker']}  (score: {row['short_score']:.2f})")
            for cond in row["conditions_text"].split(" | "):
                lines.append(f"- {cond}")
            lines.append("")

    lines.append("---")
    lines.append("## Methodology")
    lines.append("")
    lines.append("Short candidates are identified by a weighted multi-signal scoring system. "
                 "A ticker must satisfy **at least 3 independent bearish conditions** "
                 f"and reach a total weighted score of **{threshold}** or higher to appear in this list.")
    lines.append("")
    lines.append("### Condition Weights")
    lines.append("")
    lines.append("| Condition | Weight | Description |")
    lines.append("|-----------|--------|-------------|")
    for key, info in SHORT_CONDITIONS.items():
        lines.append(f"| `{key}` | {info['weight']:.1f} | {info['desc']} |")
    lines.append("")
    lines.append("### Pre-filter Rules")
    lines.append("")
    lines.append(f"- Universe: alpha_score < {UNIVERSE_ALPHA_MAX} only")
    lines.append(f"- Excluded: tickers with alpha_score >= {BUY_ALPHA_MIN} (active buy list)")
    lines.append(f"- Excluded: tickers with short interest > {MAX_SHORT_PCT_FLOAT}% float (squeeze risk)")
    lines.append("")
    lines.append("### Vehicle Guidance")
    lines.append("")
    lines.append("| Signal Level | Vehicle |")
    lines.append("|--------------|---------|")
    lines.append(f"| STRONG SHORT (score >= {STRONG_SHORT_THRESHOLD}) | Buy put options / short stock |")
    lines.append(f"| SHORT (score {threshold}–{STRONG_SHORT_THRESHOLD}) | Buy put spread (defined risk) |")
    lines.append("| High short interest triggered | Long put spread only (avoid naked short) |")
    lines.append("")
    lines.append("---")
    lines.append("_Canyon v9 — research use only. Not investment advice._")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    if df.empty:
        print("\n  No short candidates above threshold.")
        return

    print(f"\n{'='*90}")
    print(f"  CANYON v9 — SHORT CANDIDATES  ({len(df)} total)")
    print(f"{'='*90}")
    print(f"  {'#':<3} {'Ticker':<8} {'Score':>6} {'Action':<14} "
          f"{'Conds':>5} {'Sector':<22} {'Alpha':>6} {'Vehicle'}")
    print(f"  {'-'*3} {'-'*8} {'-'*6} {'-'*14} "
          f"{'-'*5} {'-'*22} {'-'*6} {'-'*35}")

    for i, row in df.iterrows():
        alpha_str = f"{row['alpha_score']:.1f}" if pd.notna(row.get("alpha_score")) else " —"
        sector    = (str(row.get("sector") or "")[:22]).ljust(22)
        print(
            f"  {i+1:<3} {row['ticker']:<8} {row['short_score']:>6.2f} "
            f"{row['short_action']:<14} {row['n_conditions']:>5} "
            f"{sector} {alpha_str:>6}  {row['suggested_vehicle']}"
        )

    strong = (df["short_action"] == "STRONG SHORT").sum()
    short  = (df["short_action"] == "SHORT").sum()
    print(f"\n  STRONG SHORT: {strong}   SHORT: {short}")
    print(f"{'='*90}")
    print(f"\n  WARNING: Research signals only — no live orders. "
          "Use defined-risk structures.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 101 — Short / Sell Signal Generator (research only)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of short candidates to output (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SHORT_THRESHOLD,
        metavar="SCORE",
        help=f"Minimum weighted score to qualify as a short candidate (default: {SHORT_THRESHOLD})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Canyon v9 — Step 101: Short Signal Generator")
    print("  For research / paper trading only. No live orders.")
    print("=" * 60)
    print(f"  Settings: top={args.top}  threshold={args.threshold}")

    df = compute_short_signals(top_n=args.top, threshold=args.threshold)

    print(f"\n[4/4] Writing outputs ...")

    if df.empty:
        empty_df = pd.DataFrame(columns=[
            "ticker", "short_score", "short_action", "n_conditions", "sector",
            "alpha_score", "conditions_text", "suggested_vehicle",
            "pcr_vol", "gex_sign", "revision_score", "surprise_rank",
            "sentiment_rank", "quality_score",
        ])
        empty_df.to_csv(OUT_CSV, index=False)
        print(f"  CSV saved (0 rows): {OUT_CSV.name}")
    else:
        # Enforce column order
        col_order = [
            "ticker", "short_score", "short_action", "n_conditions", "sector",
            "alpha_score", "conditions_text", "suggested_vehicle",
            "pcr_vol", "gex_sign", "revision_score", "surprise_rank",
            "sentiment_rank", "quality_score",
        ]
        out_cols = [c for c in col_order if c in df.columns]
        df[out_cols].to_csv(OUT_CSV, index=False)
        print(f"  CSV saved ({len(df)} rows): {OUT_CSV.name}")

    write_report(df, threshold=args.threshold, top_n=args.top)
    print_summary(df)

    print(f"\n  Outputs:")
    print(f"    {OUT_CSV}")
    print(f"    {OUT_REPORT}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
