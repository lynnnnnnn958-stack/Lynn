"""
canyon_final_v9_step89_trade_journal.py
Phase 6 – Trade Review System for Canyon v9

Reads closed paper-sim trades and auto-generates per-trade intelligence:
  entry_reason, exit_reason, strategy_type, plan_adherence, loss_reason,
  regime_at_entry, alpha_at_entry, signal_grade, holding_days, annualized_return

Outputs:
  trade_journal.csv         — enriched trade history
  trade_review_report.md    — post-mortem analysis report

Usage:
  python canyon_final_v9_step89_trade_journal.py [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent

TRADES_CSV          = ROOT / "paper_sim_trades.csv"
ALPHA_HIST_CSV      = ROOT / "alpha_score_history.csv"
REGIME_JSON         = ROOT / "regime_current.json"
OPTIONS_CSV         = ROOT / "options_signals.csv"
ALPHA_SCORES_CSV    = ROOT / "alpha_scores.csv"

JOURNAL_CSV         = ROOT / "trade_journal.csv"
REPORT_MD           = ROOT / "trade_review_report.md"


# ─────────────────────────────────────────────
# Helpers: loaders
# ─────────────────────────────────────────────

def load_trades(verbose: bool) -> pd.DataFrame | None:
    """Load paper_sim_trades.csv; return None if missing/empty."""
    if not TRADES_CSV.exists():
        print(f"[WARN] {TRADES_CSV.name} not found. Creating empty outputs.")
        return None
    try:
        df = pd.read_csv(TRADES_CSV)
    except Exception as e:
        print(f"[ERROR] Failed to read {TRADES_CSV.name}: {e}")
        return None
    if df.empty:
        print(f"[WARN] {TRADES_CSV.name} is empty. Creating empty outputs.")
        return None
    if verbose:
        print(f"  Loaded {len(df)} trade(s) from {TRADES_CSV.name}")
    return df


def load_alpha_history(verbose: bool) -> pd.DataFrame:
    """Load alpha_score_history.csv; returns empty DataFrame if missing."""
    if not ALPHA_HIST_CSV.exists():
        if verbose:
            print(f"  [WARN] {ALPHA_HIST_CSV.name} not found – alpha_at_entry will be NaN")
        return pd.DataFrame(columns=["date", "ticker", "alpha_score"])
    try:
        df = pd.read_csv(ALPHA_HIST_CSV, parse_dates=["date"])
        if verbose:
            print(f"  Loaded {len(df)} rows from {ALPHA_HIST_CSV.name}")
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {ALPHA_HIST_CSV.name}: {e}")
        return pd.DataFrame(columns=["date", "ticker", "alpha_score"])


def load_regime(verbose: bool) -> dict:
    """Load regime_current.json; returns empty dict if missing."""
    if not REGIME_JSON.exists():
        if verbose:
            print(f"  [WARN] {REGIME_JSON.name} not found – current regime unknown")
        return {}
    try:
        with open(REGIME_JSON) as f:
            data = json.load(f)
        if verbose:
            print(f"  Regime loaded: {data.get('regime', 'UNKNOWN')}")
        return data
    except Exception as e:
        print(f"  [WARN] Could not load {REGIME_JSON.name}: {e}")
        return {}


def load_options(verbose: bool) -> pd.DataFrame:
    """Load options_signals.csv; returns empty DataFrame if missing."""
    if not OPTIONS_CSV.exists():
        if verbose:
            print(f"  [WARN] {OPTIONS_CSV.name} not found – UOA flags unavailable")
        return pd.DataFrame()
    try:
        df = pd.read_csv(OPTIONS_CSV)
        if verbose:
            print(f"  Loaded {len(df)} options rows from {OPTIONS_CSV.name}")
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {OPTIONS_CSV.name}: {e}")
        return pd.DataFrame()


def load_alpha_scores(verbose: bool) -> pd.DataFrame:
    """Load alpha_scores.csv (latest); returns empty DataFrame if missing."""
    if not ALPHA_SCORES_CSV.exists():
        if verbose:
            print(f"  [WARN] {ALPHA_SCORES_CSV.name} not found")
        return pd.DataFrame()
    try:
        df = pd.read_csv(ALPHA_SCORES_CSV)
        if verbose:
            print(f"  Loaded {len(df)} rows from {ALPHA_SCORES_CSV.name}")
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {ALPHA_SCORES_CSV.name}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Classification logic
# ─────────────────────────────────────────────

def classify_exit_reason(notes: str) -> str:
    if not isinstance(notes, str) or notes.strip() == "":
        return "MANUAL"
    n = notes
    if "STOP" in n or "stop" in n:
        return "STOP_LOSS"
    if "TARGET" in n or "profit" in n:
        return "TARGET_REACHED"
    if "alpha" in n.lower():
        return "ALPHA_DEGRADATION"
    if "REGIME" in n or "regime" in n:
        return "REGIME_CHANGE"
    if "CROWD" in n:
        return "CROWDING_EXIT"
    if "rebalance" in n.lower():
        return "REBALANCE"
    return "MANUAL"


def classify_strategy_type(
    ml_score: float,
    notes: str,
    uoa_flag: int,
    pnl_pct: float,
    regime: str,
) -> str:
    notes_lower = notes.lower() if isinstance(notes, str) else ""
    notes_upper = notes.upper() if isinstance(notes, str) else ""

    if "insider" in notes_lower:
        return "INSIDER"
    if "earnings" in notes_lower:
        return "EARNINGS_PLAY"
    if "SQUEEZE" in notes_upper or "squeeze" in notes_lower:
        return "SHORT_SQUEEZE"
    if uoa_flag == 1:
        return "OPTIONS_FLOW"
    if ml_score >= 70:
        return "MOMENTUM" if pnl_pct > 0 else "QUALITY"
    if regime and regime.upper() not in ("UNKNOWN", ""):
        return "REGIME_TRADE"
    return "UNKNOWN"


def classify_loss_reason(
    exit_reason: str,
    holding_days: int,
    pnl_pct: float,
    notes: str,
    strategy_type: str,
) -> str:
    """Only called for losing trades (pnl < 0)."""
    notes_lower = notes.lower() if isinstance(notes, str) else ""
    if "REGIME" in exit_reason:
        return "MACRO_SHIFT"
    if "STOP" in exit_reason:
        return "STOP_TRIGGERED"
    if "sector" in notes_lower:
        return "SECTOR_ROTATION"
    if holding_days <= 3:
        return "SIGNAL_FAILURE"
    if holding_days <= 7 and abs(pnl_pct) > 5:
        if strategy_type in ("OPTIONS_FLOW", "EARNINGS_PLAY"):
            return "IV_CRUSH"
        return "EARNINGS_MISS"
    return "UNKNOWN"


def build_entry_reason(
    ml_score: float,
    notes: str,
    alpha_at_entry: float,
    regime: str,
    uoa_flag: int,
    top_signal: str,
) -> str:
    parts = []
    notes_str = notes if isinstance(notes, str) else ""
    top_signal_str = top_signal if isinstance(top_signal, str) else ""

    # ML score bucket
    if pd.notna(ml_score):
        ml_pct = int(round(float(ml_score) * 100)) if float(ml_score) <= 1.0 else int(round(float(ml_score)))
        if ml_pct >= 80:
            parts.append(f"ML High Confidence (score={ml_pct})")
        elif ml_pct >= 65:
            parts.append(f"ML Signal (score={ml_pct})")
        else:
            parts.append(f"System entry (ml_score={ml_pct})")
    else:
        parts.append("System entry (ml_score=N/A)")

    # Squeeze signal override/append
    if "SQUEEZE" in notes_str.upper() or "squeeze" in notes_str.lower() or \
       "SQUEEZE" in top_signal_str.upper():
        parts.append("Short Squeeze Signal")

    # Alpha quality
    if pd.notna(alpha_at_entry) and float(alpha_at_entry) >= 80:
        parts.append(f"Top-Tier Alpha (rank={alpha_at_entry:.1f})")

    # Regime context
    if regime and str(regime).upper() not in ("UNKNOWN", "", "NAN"):
        parts.append(f"{regime} regime")

    # Options UOA
    if uoa_flag == 1:
        parts.append("Options UOA detected")

    return ", ".join(parts) if parts else "System entry"


def score_plan_adherence(
    exit_reason: str,
    holding_days: int,
    pnl_pct: float,
    notes: str,
    alpha_at_entry: float,
) -> int:
    """
    Explicit four-component rubric (0–100).  Each component has hard thresholds
    so the score is reproducible and auditable.

    Component 1 — Documentation (0–20)
        Notes > 20 chars  → +20
        Notes 10-20 chars → +10
        Notes < 10 chars  → +0

    Component 2 — Entry quality (0–20)
        Alpha ≥ 80 (grade A) → +20
        Alpha ≥ 65 (grade B) → +15
        Alpha ≥ 50 (grade C) → +10
        Alpha < 50 (grade D) → +0

    Component 3 — Exit discipline (0–30)
        TARGET_PROFIT or STOP_LOSS   → +30  (pre-planned level honored)
        REGIME_CHANGE                → +25  (systematic exit)
        TIME_EXIT                    → +20  (time-based, planned)
        MANUAL with notes (≥10 chars)→ +15  (discretionary but documented)
        MANUAL without notes         → +0

    Component 4 — Holding period (0–30)
        3–21 days   → +30  (optimal swing window)
        22–30 days  → +20  (slightly extended)
        1–2 days    → +15  (short-term, not ideal)
        > 30 days   → +0   (overheld — plan drift)
        Intraday    →  tagged upstream as INTRADAY_FLIP, not scored
    """
    notes_str = notes if isinstance(notes, str) else ""
    n_len     = len(notes_str.strip())

    # ── Component 1: Documentation ────────────────────────────────────────────
    if n_len >= 20:
        doc_score = 20
    elif n_len >= 10:
        doc_score = 10
    else:
        doc_score = 0

    # ── Component 2: Entry quality ────────────────────────────────────────────
    try:
        alpha = float(alpha_at_entry) if pd.notna(alpha_at_entry) else np.nan
    except (TypeError, ValueError):
        alpha = np.nan

    if np.isnan(alpha):
        entry_score = 0
    elif alpha >= 80:
        entry_score = 20
    elif alpha >= 65:
        entry_score = 15
    elif alpha >= 50:
        entry_score = 10
    else:
        entry_score = 0

    # ── Component 3: Exit discipline ──────────────────────────────────────────
    er = str(exit_reason).upper()
    if er in ("TARGET_PROFIT", "STOP_LOSS"):
        exit_score = 30
    elif er == "REGIME_CHANGE":
        exit_score = 25
    elif er == "TIME_EXIT":
        exit_score = 20
    elif er == "MANUAL" and n_len >= 10:
        exit_score = 15
    else:
        exit_score = 0

    # ── Component 4: Holding period ───────────────────────────────────────────
    if holding_days <= 0:
        hold_score = 0    # intraday — should be tagged INTRADAY_FLIP upstream
    elif holding_days <= 2:
        hold_score = 15
    elif holding_days <= 21:
        hold_score = 30
    elif holding_days <= 30:
        hold_score = 20
    else:
        hold_score = 0

    total = doc_score + entry_score + exit_score + hold_score
    return int(np.clip(total, 0, 100))


def signal_grade(alpha_at_entry: float) -> str:
    if pd.isna(alpha_at_entry):
        return "D"
    v = float(alpha_at_entry)
    if v >= 80:
        return "A"
    if v >= 65:
        return "B"
    if v >= 50:
        return "C"
    return "D"


INTRADAY_TAG = "INTRADAY_FLIP"  # sentinel used for holding_days <= 1

def annualized_return(pnl_pct: float, holding_days: int) -> float:
    """
    Annualize a return.

    Returns NaN for:
      - invalid / missing inputs
      - intraday trades (holding_days <= 1) — multiplying by 365 would produce
        absurd numbers (e.g. +0.5% in 1 day → +182% annualised) that poison
        portfolio-level annualised-return averages.  Tag these trades as
        INTRADAY_FLIP and report the raw pnl_pct only.
    """
    if pd.isna(pnl_pct) or holding_days <= 0:
        return np.nan
    if holding_days <= 1:
        return np.nan   # caller should label as INTRADAY_FLIP
    return round((pnl_pct / 100) / holding_days * 365 * 100, 2)


# ─────────────────────────────────────────────
# Core enrichment
# ─────────────────────────────────────────────

def enrich_trades(
    trades: pd.DataFrame,
    alpha_hist: pd.DataFrame,
    regime_data: dict,
    options: pd.DataFrame,
    alpha_scores: pd.DataFrame,
    verbose: bool,
) -> pd.DataFrame:
    df = trades.copy()

    # Normalise ml_score to 0-100
    if "ml_score" in df.columns:
        df["ml_score"] = pd.to_numeric(df["ml_score"], errors="coerce").fillna(0)
        if df["ml_score"].max() <= 1.0:
            df["ml_score"] = (df["ml_score"] * 100).round(2)
    else:
        df["ml_score"] = 0.0

    # Parse dates
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # pnl_pct numeric
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct", 0), errors="coerce").fillna(0)
    df["pnl"]     = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0)

    # ── Build lookup: UOA flag by ticker ─────────────────────────────────────
    uoa_lookup: dict[str, int] = {}
    if not options.empty and "ticker" in options.columns:
        # prefer unusual_call_flag if present, else fall back to options_signal text
        if "unusual_call_flag" in options.columns:
            for _, row in options.iterrows():
                uoa_lookup[str(row["ticker"])] = int(row.get("unusual_call_flag", 0))
        elif "options_signal" in options.columns:
            for _, row in options.iterrows():
                sig = str(row.get("options_signal", "")).upper()
                uoa_lookup[str(row["ticker"])] = 1 if "UOA" in sig or "UNUSUAL" in sig else 0

    # ── Build lookup: alpha by (date, ticker) from history ───────────────────
    alpha_hist_lookup: dict[tuple, float] = {}
    if not alpha_hist.empty and "date" in alpha_hist.columns and "ticker" in alpha_hist.columns:
        alpha_hist["date"] = pd.to_datetime(alpha_hist["date"], errors="coerce")
        for _, row in alpha_hist.iterrows():
            key = (row["date"].date() if pd.notna(row["date"]) else None, str(row["ticker"]))
            if key[0] is not None:
                alpha_hist_lookup[key] = float(row["alpha_score"])

    # ── Build lookup: regime by (date, ticker) from history ──────────────────
    regime_hist_lookup: dict[tuple, str] = {}
    if not alpha_hist.empty and "regime" in alpha_hist.columns:
        for _, row in alpha_hist.iterrows():
            key = (row["date"].date() if pd.notna(row["date"]) else None, str(row["ticker"]))
            if key[0] is not None:
                regime_hist_lookup[key] = str(row["regime"])

    # ── Build latest alpha / top_signal lookup by ticker ─────────────────────
    latest_alpha_lookup: dict[str, float] = {}
    top_signal_lookup:   dict[str, str]   = {}
    if not alpha_scores.empty and "ticker" in alpha_scores.columns:
        for _, row in alpha_scores.iterrows():
            t = str(row["ticker"])
            latest_alpha_lookup[t] = float(row.get("alpha_score", np.nan))
            top_signal_lookup[t]   = str(row.get("top_signal", "")) if "top_signal" in row.index else ""

    # ── Per-row enrichment ────────────────────────────────────────────────────
    rows_enriched = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        entry_dt = row.get("entry_date")
        exit_dt  = row.get("exit_date")
        ml_score = float(row.get("ml_score", 0))
        pnl_pct  = float(row.get("pnl_pct", 0))
        pnl_val  = float(row.get("pnl", 0))
        notes    = str(row.get("notes", "")) if pd.notna(row.get("notes", "")) else ""

        # holding_days — allow 0 for same-day (intraday) trades
        if pd.notna(entry_dt) and pd.notna(exit_dt):
            holding_days = max(int((exit_dt - entry_dt).days), 0)
        else:
            holding_days = 1

        # alpha_at_entry from history (exact date match first, else latest)
        entry_key = (entry_dt.date() if pd.notna(entry_dt) else None, ticker)
        alpha_at_entry = alpha_hist_lookup.get(entry_key, np.nan)
        if pd.isna(alpha_at_entry):
            alpha_at_entry = latest_alpha_lookup.get(ticker, np.nan)

        # regime_at_entry
        regime_at_entry = regime_hist_lookup.get(entry_key, "")
        if not regime_at_entry:
            regime_at_entry = regime_data.get("regime", "UNKNOWN")

        # UOA flag
        uoa_flag = uoa_lookup.get(ticker, 0)

        # top_signal
        top_signal = top_signal_lookup.get(ticker, "")

        # Derived fields
        exit_reason    = classify_exit_reason(notes)
        # Override exit_reason for intraday flips so downstream analysis can filter
        if holding_days <= 1:
            exit_reason = INTRADAY_TAG
        strategy_type  = classify_strategy_type(ml_score, notes, uoa_flag, pnl_pct, regime_at_entry)
        entry_reason   = build_entry_reason(ml_score, notes, alpha_at_entry, regime_at_entry, uoa_flag, top_signal)
        plan_adherence = score_plan_adherence(exit_reason, holding_days, pnl_pct, notes, alpha_at_entry)
        s_grade        = signal_grade(alpha_at_entry)
        ann_ret        = annualized_return(pnl_pct, holding_days)  # NaN for intraday

        loss_reason = ""
        if pnl_val < 0:
            loss_reason = classify_loss_reason(exit_reason, holding_days, pnl_pct, notes, strategy_type)

        rows_enriched.append({
            **row.to_dict(),
            "entry_reason":      entry_reason,
            "exit_reason":       exit_reason,
            "strategy_type":     strategy_type,
            "plan_adherence":    plan_adherence,
            "loss_reason":       loss_reason,
            "regime_at_entry":   regime_at_entry,
            "alpha_at_entry":    round(alpha_at_entry, 2) if pd.notna(alpha_at_entry) else np.nan,
            "signal_grade":      s_grade,
            "holding_days":      holding_days,
            "annualized_return": ann_ret,
        })

    result = pd.DataFrame(rows_enriched)

    # Convert dates back to strings for clean CSV output
    for col in ("entry_date", "exit_date"):
        if col in result.columns:
            result[col] = result[col].dt.strftime("%Y-%m-%d")

    if verbose:
        print(f"  Enriched {len(result)} trade(s)")
    return result


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def rolling_win_rate(df: pd.DataFrame, window: int = 10) -> list[tuple[int, float]]:
    """Return list of (trade_index, rolling_win_rate_pct) for last N trades."""
    wins = (df["pnl"] > 0).astype(int)
    results = []
    for i in range(len(wins)):
        start = max(0, i - window + 1)
        w = wins.iloc[start : i + 1]
        results.append((i + 1, round(w.mean() * 100, 1)))
    return results


def build_report(df: pd.DataFrame) -> str:
    lines = []

    def h1(text):
        lines.append(f"# {text}\n")

    def h2(text):
        lines.append(f"\n## {text}\n")

    def h3(text):
        lines.append(f"\n### {text}\n")

    def row(label, value):
        lines.append(f"- **{label}:** {value}")

    h1("Canyon v9 Trade Review Report")
    lines.append(f"_Generated by canyon_final_v9_step89_trade_journal.py_\n")

    if df.empty:
        lines.append("No trades to analyse.\n")
        return "\n".join(lines)

    total = len(df)
    wins  = int((df["pnl"] > 0).sum())
    losses = int((df["pnl"] < 0).sum())
    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0
    avg_pnl  = round(df["pnl"].mean(), 2)
    avg_pnl_pct = round(df["pnl_pct"].mean(), 2) if "pnl_pct" in df.columns else np.nan
    avg_adh  = round(df["plan_adherence"].mean(), 1)

    h2("Overall Summary")
    row("Total Trades",        total)
    row("Wins / Losses",       f"{wins} / {losses}")
    row("Win Rate",            f"{win_rate}%")
    row("Avg PnL",             f"${avg_pnl:+,.2f}")
    row("Avg PnL %",           f"{avg_pnl_pct:+.2f}%")
    row("Avg Plan Adherence",  f"{avg_adh}/100")

    # ── By strategy_type ──────────────────────────────────────────────────────
    h2("Win Rate by Strategy Type")
    if "strategy_type" in df.columns:
        st_group = df.groupby("strategy_type")
        st_stats = st_group.agg(
            count=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
            avg_pnl=("pnl", "mean"),
        ).reset_index()
        st_stats["win_rate"] = (st_stats["wins"] / st_stats["count"] * 100).round(1)
        st_stats = st_stats.sort_values("avg_pnl", ascending=False)
        lines.append("")
        lines.append("| Strategy Type | Trades | Win Rate | Avg PnL |")
        lines.append("|---|---|---|---|")
        for _, r in st_stats.iterrows():
            lines.append(f"| {r['strategy_type']} | {r['count']} | {r['win_rate']}% | ${r['avg_pnl']:+,.2f} |")
        best_st = st_stats.iloc[0]
        lines.append(f"\n**Best Strategy:** {best_st['strategy_type']} (avg PnL ${best_st['avg_pnl']:+,.2f})")

    # ── By exit_reason ────────────────────────────────────────────────────────
    h2("Win Rate by Exit Reason")
    if "exit_reason" in df.columns:
        er_group = df.groupby("exit_reason")
        er_stats = er_group.agg(
            count=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
            avg_pnl=("pnl", "mean"),
        ).reset_index()
        er_stats["win_rate"] = (er_stats["wins"] / er_stats["count"] * 100).round(1)
        er_stats = er_stats.sort_values("win_rate", ascending=False)
        lines.append("")
        lines.append("| Exit Reason | Trades | Win Rate | Avg PnL |")
        lines.append("|---|---|---|---|")
        for _, r in er_stats.iterrows():
            lines.append(f"| {r['exit_reason']} | {r['count']} | {r['win_rate']}% | ${r['avg_pnl']:+,.2f} |")

    # ── Signal grade performance ──────────────────────────────────────────────
    h2("Signal Grade Performance (A/B/C/D)")
    if "signal_grade" in df.columns:
        sg_group = df.groupby("signal_grade")
        sg_stats = sg_group.agg(
            count=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
            avg_pnl=("pnl", "mean"),
        ).reset_index()
        sg_stats["win_rate"] = (sg_stats["wins"] / sg_stats["count"] * 100).round(1)
        sg_stats = sg_stats.sort_values("signal_grade")
        lines.append("")
        lines.append("| Grade | Trades | Win Rate | Avg PnL | Threshold |")
        lines.append("|---|---|---|---|---|")
        thresholds = {"A": "alpha>=80", "B": "alpha>=65", "C": "alpha>=50", "D": "alpha<50"}
        for _, r in sg_stats.iterrows():
            g = r["signal_grade"]
            lines.append(f"| {g} | {r['count']} | {r['win_rate']}% | ${r['avg_pnl']:+,.2f} | {thresholds.get(g, '')} |")

    # ── Loss analysis ─────────────────────────────────────────────────────────
    h2("Loss Analysis")
    losers = df[df["pnl"] < 0].copy()
    if losers.empty:
        lines.append("No losing trades in this period.")
    else:
        if "loss_reason" in losers.columns and losers["loss_reason"].notna().any():
            lr_counts = losers["loss_reason"].value_counts()
            lines.append("")
            lines.append("| Loss Reason | Count | % of Losses |")
            lines.append("|---|---|---|")
            for reason, cnt in lr_counts.items():
                pct = round(cnt / len(losers) * 100, 1)
                lines.append(f"| {reason} | {cnt} | {pct}% |")

    # ── Plan adherence detail ─────────────────────────────────────────────────
    h2("Plan Adherence")
    row("Average Score", f"{avg_adh}/100")
    if "plan_adherence" in df.columns:
        buckets = pd.cut(
            df["plan_adherence"],
            bins=[0, 60, 75, 90, 101],
            labels=["Poor (0-60)", "Fair (61-75)", "Good (76-90)", "Excellent (91-100)"],
            right=False,
        )
        bucket_counts = buckets.value_counts().sort_index()
        lines.append("")
        lines.append("| Adherence Bucket | Count |")
        lines.append("|---|---|")
        for label, cnt in bucket_counts.items():
            lines.append(f"| {label} | {cnt} |")

    # ── Rolling 10-trade win rate ─────────────────────────────────────────────
    h2("Rolling 10-Trade Win Rate (trend)")
    if len(df) >= 2:
        rolling = rolling_win_rate(df, window=10)
        lines.append("")
        lines.append("| Trade # | Rolling Win Rate |")
        lines.append("|---|---|")
        for idx, wr in rolling:
            lines.append(f"| {idx} | {wr}% |")
    else:
        lines.append("Insufficient trades for rolling analysis.")

    # ── Recommendations ───────────────────────────────────────────────────────
    h2("Recommendations")
    recs = []

    # Stop-loss recommendation
    if "exit_reason" in df.columns:
        stop_trades = df[df["exit_reason"] == "STOP_LOSS"]
        manual_trades = df[df["exit_reason"] == "MANUAL"]
        if len(stop_trades) > 0 and len(manual_trades) > 0:
            stop_wr   = (stop_trades["pnl"] > 0).mean() * 100
            manual_wr = (manual_trades["pnl"] > 0).mean() * 100
            if stop_wr > manual_wr:
                recs.append(
                    f"STOP_LOSS exits have {stop_wr:.0f}% win rate vs MANUAL {manual_wr:.0f}% "
                    f"→ use stops more consistently."
                )

    # Overheld deduction
    if "holding_days" in df.columns:
        overheld = df[df["holding_days"] > 30]
        if len(overheld) > 0:
            recs.append(
                f"{len(overheld)} trade(s) held >30 days – review exit timing and prevent overholding."
            )

    # Signal grade A vs D
    if "signal_grade" in df.columns:
        a_trades = df[df["signal_grade"] == "A"]
        d_trades = df[df["signal_grade"] == "D"]
        if len(a_trades) > 0 and len(d_trades) > 0:
            a_wr = (a_trades["pnl"] > 0).mean() * 100
            d_wr = (d_trades["pnl"] > 0).mean() * 100
            if a_wr > d_wr:
                recs.append(
                    f"Grade-A signal entries win {a_wr:.0f}% vs Grade-D {d_wr:.0f}% "
                    f"→ prioritise alpha>=80 setups."
                )

    # High plan_adherence correlation
    if "plan_adherence" in df.columns:
        high_adh = df[df["plan_adherence"] >= 80]
        low_adh  = df[df["plan_adherence"] < 60]
        if len(high_adh) > 0 and len(low_adh) > 0:
            h_wr = (high_adh["pnl"] > 0).mean() * 100
            l_wr = (low_adh["pnl"] > 0).mean() * 100
            recs.append(
                f"High-adherence trades (>=80) win {h_wr:.0f}% vs low-adherence {l_wr:.0f}% "
                f"→ document every entry and follow the plan."
            )

    if not recs:
        recs.append("Insufficient data for specific recommendations. Add more trades.")

    for rec in recs:
        lines.append(f"- {rec}")

    lines.append("\n---")
    lines.append("_Report generated by Phase 6 Trade Review System – Canyon v9_\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Summary box
# ─────────────────────────────────────────────

def print_summary_box(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n┌──────────────────────────────────────────┐")
        print("│          TRADE JOURNAL SUMMARY           │")
        print("│          No trades to analyse.           │")
        print("└──────────────────────────────────────────┘")
        return

    total     = len(df)
    wins      = int((df["pnl"] > 0).sum())
    losses    = int((df["pnl"] < 0).sum())
    win_rate  = round(wins / total * 100, 1) if total > 0 else 0.0
    avg_pnl   = round(df["pnl"].mean(), 2)
    avg_adh   = round(df["plan_adherence"].mean(), 1)
    avg_hold  = round(df["holding_days"].mean(), 1)

    lines = [
        "┌──────────────────────────────────────────┐",
        "│          TRADE JOURNAL SUMMARY           │",
        "├──────────────────────────────────────────┤",
        f"│  Total Trades      : {total:<20}│",
        f"│  Win / Loss        : {wins} / {losses:<17}│",
        f"│  Win Rate          : {win_rate:<20.1f}│",
        f"│  Avg PnL           : ${avg_pnl:<19,.2f}│",
        f"│  Avg Plan Adh.     : {avg_adh:<20.1f}│",
        f"│  Avg Holding Days  : {avg_hold:<20.1f}│",
        "└──────────────────────────────────────────┘",
    ]
    print()
    for ln in lines:
        print(ln)


# ─────────────────────────────────────────────
# Empty output writers
# ─────────────────────────────────────────────

def write_empty_outputs() -> None:
    empty_cols = [
        "ticker", "entry_date", "entry_price", "exit_date", "exit_price",
        "shares", "direction", "pnl", "pnl_pct", "status", "ml_score", "notes",
        "entry_reason", "exit_reason", "strategy_type", "plan_adherence",
        "loss_reason", "regime_at_entry", "alpha_at_entry", "signal_grade",
        "holding_days", "annualized_return",
    ]
    pd.DataFrame(columns=empty_cols).to_csv(JOURNAL_CSV, index=False)
    REPORT_MD.write_text(
        "# Canyon v9 Trade Review Report\n\nNo trades available to analyse.\n"
    )
    print(f"  Wrote empty {JOURNAL_CSV.name}")
    print(f"  Wrote empty {REPORT_MD.name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Phase 6 – Trade Journal & Review System"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress during processing",
    )
    args = parser.parse_args()
    verbose: bool = args.verbose

    print("=" * 50)
    print("Canyon v9  |  Step 89  |  Trade Journal")
    print("=" * 50)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\n[1] Loading input files…")
    trades = load_trades(verbose)
    if trades is None:
        write_empty_outputs()
        print_summary_box(pd.DataFrame())
        return 0

    alpha_hist  = load_alpha_history(verbose)
    regime_data = load_regime(verbose)
    options     = load_options(verbose)
    alpha_scores = load_alpha_scores(verbose)

    # ── Enrich ────────────────────────────────────────────────────────────────
    print("\n[2] Enriching trades…")
    journal = enrich_trades(trades, alpha_hist, regime_data, options, alpha_scores, verbose)

    # ── Write journal CSV ─────────────────────────────────────────────────────
    print("\n[3] Writing outputs…")
    journal.to_csv(JOURNAL_CSV, index=False)
    print(f"  trade_journal.csv → {JOURNAL_CSV}")

    # ── Build and write report ────────────────────────────────────────────────
    report_text = build_report(journal)
    REPORT_MD.write_text(report_text)
    print(f"  trade_review_report.md → {REPORT_MD}")

    # ── Print per-trade summary ───────────────────────────────────────────────
    if verbose:
        print("\n[4] Per-trade summary:")
        print(f"  {'Ticker':<8} {'Entry':>12} {'Exit':>12} {'PnL%':>7} {'Grade':<6} "
              f"{'ExitReason':<20} {'Strategy':<16} {'Adh':>5}")
        print("  " + "-" * 90)
        for _, r in journal.iterrows():
            print(
                f"  {str(r.get('ticker','')):<8} "
                f"{str(r.get('entry_date',''))[:10]:>12} "
                f"{str(r.get('exit_date',''))[:10]:>12} "
                f"{float(r.get('pnl_pct', 0)):>+7.2f}% "
                f"{str(r.get('signal_grade','')):<6} "
                f"{str(r.get('exit_reason','')):<20} "
                f"{str(r.get('strategy_type','')):<16} "
                f"{int(r.get('plan_adherence', 0)):>5}"
            )

    # ── Stats print ───────────────────────────────────────────────────────────
    if not journal.empty:
        print("\n[5] Quick stats:")
        total   = len(journal)
        wins    = int((journal["pnl"] > 0).sum())
        wr      = round(wins / total * 100, 1) if total > 0 else 0.0
        avg_pnl = round(journal["pnl"].mean(), 2)
        avg_adh = round(journal["plan_adherence"].mean(), 1)
        print(f"  Trades: {total}  |  Win rate: {wr}%  |  Avg PnL: ${avg_pnl:+,.2f}  |  Avg adherence: {avg_adh}/100")

        if "strategy_type" in journal.columns:
            print("\n  Win rate by strategy:")
            st_gb = journal.groupby("strategy_type")["pnl"].agg(
                count="count",
                win_count=lambda x: (x > 0).sum(),
            )
            for st, row_data in st_gb.iterrows():
                wr_st = round(row_data["win_count"] / row_data["count"] * 100, 1)
                print(f"    {str(st):<18} {row_data['count']:>3} trades   {wr_st:>6.1f}% win rate")

    print_summary_box(journal)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
