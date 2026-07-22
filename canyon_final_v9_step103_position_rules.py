#!/usr/bin/env python3
"""
Canyon v9 — Step 103: Position Rules Engine
============================================
Checks all open paper-sim positions against rule-based lifecycle management.
Generates URGENT / WARNING / MONITOR alerts for stop-loss, profit targets,
hold limits, overweight, alpha decay, earnings risk, and regime shifts.

For research and paper trading only. No live orders.

Output files
------------
  position_rules_alerts.csv   — one row per alert
  position_rules_report.md    — human-readable summary

Usage
-----
  python3 canyon_final_v9_step103_position_rules.py
  python3 canyon_final_v9_step103_position_rules.py --stop-loss -0.10
  python3 canyon_final_v9_step103_position_rules.py --profit-target 0.25 --max-hold 45
  python3 canyon_final_v9_step103_position_rules.py --max-weight 0.12 --min-alpha-exit 40
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_CSV    = ROOT / "position_rules_alerts.csv"
OUT_REPORT = ROOT / "position_rules_report.md"

# ── Default rules ─────────────────────────────────────────────────────────────
DEFAULT_RULES: dict[str, float] = {
    "stop_loss_pct":     -0.08,   # -8% from entry price
    "profit_target_pct":  0.20,   # +20% from entry price
    "max_hold_days":      63,     # ~3 months trading days
    "max_weight_pct":     0.15,   # 15% max single position
    "min_alpha_exit":     35.0,   # exit if alpha_score drops below 35
    "size_warn_pct":      0.12,   # warn if position > 12% (before hard limit)
}

# ── Alert types ───────────────────────────────────────────────────────────────
ALERT_TYPES: dict[str, str] = {
    "STOP_LOSS":      "Position has hit stop-loss level",
    "PROFIT_TARGET":  "Position has hit profit target",
    "MAX_HOLD":       "Position has exceeded maximum hold period",
    "OVERWEIGHT":     "Position exceeds maximum weight limit",
    "SIZE_WARNING":   "Position approaching weight limit",
    "ALPHA_DECAY":    "Alpha score dropped below exit threshold",
    "EARNINGS_RISK":  "Earnings within 3 days — define risk or exit",
    "REGIME_EXIT":    "Market regime shifted to BEAR — consider reducing",
}

# Severity ordering for sort
SEVERITY_ORDER = {"URGENT": 0, "WARNING": 1, "MONITOR": 2}


# ─────────────────────────────────────────────────────────────────────────────
# [1/4]  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    """Safely convert a value to float; return default if not possible."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_csv_safe(path: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV gracefully; return None if missing or unreadable."""
    if not path.exists():
        print(f"  [SKIP] {label} not found ({path.name})")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {label} is empty")
            return None
        return df
    except Exception as exc:
        print(f"  [WARN] {label} could not be read: {exc}")
        return None


def load_open_positions() -> pd.DataFrame:
    """
    Load open (active) trades from paper_sim_trades.csv.
    Open = rows where exit_date is NaN / blank (position not yet closed).
    Normalises ticker to uppercase.
    """
    path = ROOT / "paper_sim_trades.csv"
    df = _load_csv_safe(path, "paper_sim_trades.csv")
    if df is None:
        return pd.DataFrame()

    # Normalise ticker
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Filter to open trades — exit_date is missing or blank
    if "exit_date" in df.columns:
        open_mask = df["exit_date"].isna() | (df["exit_date"].astype(str).str.strip() == "")
        df = df[open_mask].copy()
    # Fallback: filter by status column
    elif "status" in df.columns:
        df = df[~df["status"].astype(str).str.upper().isin(["CLOSED", "EXITED"])].copy()

    df = df.reset_index(drop=True)
    print(f"  Open positions loaded: {len(df)} row(s)")
    return df


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """
    Read latest prices from sp500_price_cache.csv.
    The cache is wide-format: rows=dates, columns=tickers.
    Returns the last non-NaN price per requested ticker.
    """
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
        print(f"  [SKIP] sp500_price_cache.csv not found")
        return {}

    try:
        # Read with first column as index (dates)
        df = pd.read_csv(path, index_col=0)
        prices: dict[str, float] = {}
        for ticker in tickers:
            if ticker in df.columns:
                col = pd.to_numeric(df[ticker], errors="coerce")
                last_valid = col.dropna()
                if not last_valid.empty:
                    prices[ticker] = float(last_valid.iloc[-1])
                else:
                    print(f"  [WARN] No price data for {ticker} in cache")
            else:
                print(f"  [WARN] Ticker {ticker} not in price cache")
        return prices
    except Exception as exc:
        print(f"  [WARN] Could not read sp500_price_cache.csv: {exc}")
        return {}


def get_alpha_scores(tickers: list[str]) -> dict[str, float]:
    """
    Read current alpha_score per ticker from alpha_scores.csv.
    Returns dict {ticker: alpha_score}.
    """
    path = ROOT / "alpha_scores.csv"
    df = _load_csv_safe(path, "alpha_scores.csv")
    if df is None:
        return {}

    if "ticker" not in df.columns or "alpha_score" not in df.columns:
        print("  [WARN] alpha_scores.csv missing 'ticker' or 'alpha_score' column")
        return {}

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["alpha_score"] = pd.to_numeric(df["alpha_score"], errors="coerce")
    lookup = df.set_index("ticker")["alpha_score"].to_dict()
    return {t: _safe_float(lookup.get(t, np.nan)) for t in tickers}


def get_portfolio_weights(tickers: list[str]) -> dict[str, float]:
    """
    Read current portfolio weights from daily_final.csv (weight_pct column).
    weight_pct is expected as a percentage (e.g. 4.5 = 4.5%).
    Returns dict {ticker: weight_as_decimal}.
    """
    path = ROOT / "daily_final.csv"
    df = _load_csv_safe(path, "daily_final.csv")
    if df is None:
        return {}

    if "ticker" not in df.columns:
        print("  [WARN] daily_final.csv missing 'ticker' column")
        return {}

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Find weight column — prefer weight_pct, fall back to weight
    weight_col = None
    for candidate in ("weight_pct", "weight", "portfolio_weight"):
        if candidate in df.columns:
            weight_col = candidate
            break

    if weight_col is None:
        print("  [WARN] daily_final.csv has no weight column (weight_pct / weight)")
        return {}

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    lookup = df.set_index("ticker")[weight_col].to_dict()

    result: dict[str, float] = {}
    for t in tickers:
        raw = _safe_float(lookup.get(t, np.nan))
        if np.isnan(raw):
            result[t] = np.nan
        elif raw > 1.0:
            # stored as percentage (e.g. 4.5 → 0.045)
            result[t] = raw / 100.0
        else:
            result[t] = raw
    return result


def get_earnings_days(tickers: list[str]) -> dict[str, int | None]:
    """
    Load earnings_calendar.csv and compute calendar days until next earnings.
    Returns dict {ticker: days_until_earnings | None}.
    None means earnings date not found or in the past.
    """
    path = ROOT / "earnings_calendar.csv"
    df = _load_csv_safe(path, "earnings_calendar.csv")
    if df is None:
        return {t: None for t in tickers}

    # Normalise ticker
    for col in df.columns:
        if col.lower() == "ticker":
            df = df.rename(columns={col: "ticker"})
            break
    if "ticker" not in df.columns:
        return {t: None for t in tickers}

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Find earnings date column
    date_col = None
    for candidate in ("next_earnings", "earnings_date", "report_date", "date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        return {t: None for t in tickers}

    today = date.today()
    result: dict[str, int | None] = {}
    lookup = df.set_index("ticker")[date_col].to_dict()

    for t in tickers:
        raw = lookup.get(t)
        if raw is None or str(raw).strip() in ("", "nan", "NaT"):
            result[t] = None
            continue
        try:
            earn_dt = pd.to_datetime(raw).date()
            delta = (earn_dt - today).days
            result[t] = delta if delta >= 0 else None
        except Exception:
            result[t] = None

    return result


def get_current_regime() -> str | None:
    """
    Read current market regime from regime_current.json (step76 output).
    Returns regime string (e.g. 'BULL', 'BEAR', 'SIDEWAYS') or None.
    """
    path = ROOT / "regime_current.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        regime = str(data.get("regime", data.get("current_regime", ""))).upper()
        return regime if regime else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# [2/4]  RULE EVALUATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_pnl(entry_price: float, current_price: float) -> float:
    """Returns P&L as decimal (0.05 = +5%). Returns NaN if entry_price is 0."""
    if entry_price == 0 or np.isnan(entry_price) or np.isnan(current_price):
        return np.nan
    return (current_price - entry_price) / entry_price


def _compute_days_held(entry_date_raw) -> int | None:
    """Calendar days from entry_date to today."""
    if pd.isna(entry_date_raw) or str(entry_date_raw).strip() in ("", "nan"):
        return None
    try:
        entry_dt = pd.to_datetime(entry_date_raw).date()
        return (date.today() - entry_dt).days
    except Exception:
        return None


def check_position_rules(
    trade_row: pd.Series,
    current_price: float,
    current_alpha: float,
    current_weight: float,
    earnings_days: int | None,
    rules: dict,
    current_regime: str | None = None,
) -> list[dict]:
    """
    Check all position rules for one trade row.

    Parameters
    ----------
    trade_row       : pd.Series — one row from paper_sim_trades.csv
    current_price   : float — latest market price
    current_alpha   : float — current alpha_score (NaN if unavailable)
    current_weight  : float — current portfolio weight as decimal (NaN if unavailable)
    earnings_days   : int | None — calendar days until next earnings
    rules           : dict — rule parameters (stop_loss_pct, etc.)
    current_regime  : str | None — current market regime string

    Returns
    -------
    list of triggered alert dicts:
        {alert_type, severity, detail, recommended_action}
    """
    alerts: list[dict] = []
    ticker = str(trade_row.get("ticker", "UNKNOWN")).upper()

    entry_price = _safe_float(trade_row.get("entry_price", np.nan))
    entry_date  = trade_row.get("entry_date")
    pnl_pct     = compute_position_pnl(entry_price, current_price)
    days_held   = _compute_days_held(entry_date)

    def _add(alert_type: str, severity: str, detail: str, recommended_action: str) -> None:
        alerts.append({
            "ticker":             ticker,
            "alert_type":         alert_type,
            "severity":           severity,
            "detail":             detail,
            "recommended_action": recommended_action,
        })

    # ── Rule 1: Stop-loss ────────────────────────────────────────────────────
    stop_loss = rules.get("stop_loss_pct", DEFAULT_RULES["stop_loss_pct"])
    if not np.isnan(pnl_pct) and pnl_pct <= stop_loss:
        _add(
            alert_type="STOP_LOSS",
            severity="URGENT",
            detail=(
                f"P&L {pnl_pct*100:+.2f}% is at or below stop-loss threshold "
                f"({stop_loss*100:.0f}%). "
                f"Entry: ${entry_price:.2f}  Current: ${current_price:.2f}"
            ),
            recommended_action=(
                f"Exit position immediately. Stop-loss breached — "
                f"max loss limit of {abs(stop_loss)*100:.0f}% reached."
            ),
        )

    # ── Rule 2: Profit target ────────────────────────────────────────────────
    profit_target = rules.get("profit_target_pct", DEFAULT_RULES["profit_target_pct"])
    if not np.isnan(pnl_pct) and pnl_pct >= profit_target:
        _add(
            alert_type="PROFIT_TARGET",
            severity="WARNING",
            detail=(
                f"P&L {pnl_pct*100:+.2f}% has reached profit target "
                f"({profit_target*100:.0f}%). "
                f"Entry: ${entry_price:.2f}  Current: ${current_price:.2f}"
            ),
            recommended_action=(
                f"Consider trimming or closing position. Profit target of "
                f"{profit_target*100:.0f}% achieved — lock in gains or trail stop."
            ),
        )

    # ── Rule 3: Maximum hold period ──────────────────────────────────────────
    max_hold = int(rules.get("max_hold_days", DEFAULT_RULES["max_hold_days"]))
    if days_held is not None and days_held >= max_hold:
        _add(
            alert_type="MAX_HOLD",
            severity="WARNING",
            detail=(
                f"Position held for {days_held} calendar days, exceeding the "
                f"{max_hold}-day maximum hold limit."
            ),
            recommended_action=(
                f"Review thesis. Position exceeded {max_hold}-day hold limit — "
                "close unless fundamentals / alpha score strongly support extension."
            ),
        )

    # ── Rule 4: Overweight (hard limit) ──────────────────────────────────────
    max_weight = rules.get("max_weight_pct", DEFAULT_RULES["max_weight_pct"])
    if not np.isnan(current_weight) and current_weight >= max_weight:
        _add(
            alert_type="OVERWEIGHT",
            severity="URGENT",
            detail=(
                f"Position weight {current_weight*100:.2f}% exceeds maximum "
                f"weight limit ({max_weight*100:.0f}%)."
            ),
            recommended_action=(
                f"Reduce position. Current weight {current_weight*100:.2f}% is above "
                f"the {max_weight*100:.0f}% hard cap — trim to restore compliance."
            ),
        )

    # ── Rule 5: Size warning (approaching limit) ──────────────────────────────
    size_warn = rules.get("size_warn_pct", DEFAULT_RULES["size_warn_pct"])
    if (
        not np.isnan(current_weight)
        and current_weight >= size_warn
        and current_weight < max_weight
    ):
        _add(
            alert_type="SIZE_WARNING",
            severity="MONITOR",
            detail=(
                f"Position weight {current_weight*100:.2f}% is approaching the "
                f"maximum weight limit ({max_weight*100:.0f}%). "
                f"Warning threshold: {size_warn*100:.0f}%."
            ),
            recommended_action=(
                f"Monitor closely. Position is within {(max_weight - current_weight)*100:.2f}% "
                f"of the {max_weight*100:.0f}% hard cap — avoid adding to this position."
            ),
        )

    # ── Rule 6: Alpha decay ───────────────────────────────────────────────────
    min_alpha = rules.get("min_alpha_exit", DEFAULT_RULES["min_alpha_exit"])
    if not np.isnan(current_alpha) and current_alpha < min_alpha:
        _add(
            alert_type="ALPHA_DECAY",
            severity="WARNING",
            detail=(
                f"Current alpha score {current_alpha:.1f} has dropped below the "
                f"exit threshold ({min_alpha:.0f}). Alpha conviction no longer "
                f"supports holding this position."
            ),
            recommended_action=(
                f"Consider exiting. Alpha score {current_alpha:.1f} is below "
                f"minimum exit threshold of {min_alpha:.0f} — re-evaluate thesis."
            ),
        )

    # ── Rule 7: Earnings risk ─────────────────────────────────────────────────
    EARNINGS_WINDOW_DAYS = 3
    if earnings_days is not None and earnings_days <= EARNINGS_WINDOW_DAYS:
        severity = "URGENT" if earnings_days <= 1 else "WARNING"
        day_word = "day" if earnings_days == 1 else "days"
        _add(
            alert_type="EARNINGS_RISK",
            severity=severity,
            detail=(
                f"Earnings are in {earnings_days} {day_word}. "
                f"Open position carries binary event risk."
            ),
            recommended_action=(
                "Define earnings risk before the event: either reduce position size, "
                "hedge with defined-risk options structure, or exit before earnings."
            ),
        )

    # ── Rule 8: Regime exit (BEAR market) ────────────────────────────────────
    if current_regime is not None and "BEAR" in current_regime.upper():
        direction = str(trade_row.get("direction", "LONG")).upper()
        if direction == "LONG":
            _add(
                alert_type="REGIME_EXIT",
                severity="WARNING",
                detail=(
                    f"Market regime is currently {current_regime}. "
                    f"Long positions face elevated systemic downside risk."
                ),
                recommended_action=(
                    "Consider reducing long exposure. BEAR regime detected — "
                    "trim position or implement hedge until regime shifts."
                ),
            )

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# [3/4]  REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    alerts_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    rules: dict,
    run_ts: str,
) -> None:
    """Write position_rules_report.md."""
    lines: list[str] = []
    lines.append("# Canyon v9 — Position Rules Report (Step 103)")
    lines.append(f"_Generated: {run_ts}_\n")

    # ── Rule summary ──────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Rule Configuration")
    lines.append("")
    lines.append("| Rule | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Stop-loss threshold | {rules['stop_loss_pct']*100:.0f}% |")
    lines.append(f"| Profit target | {rules['profit_target_pct']*100:.0f}% |")
    lines.append(f"| Maximum hold days | {int(rules['max_hold_days'])} |")
    lines.append(f"| Maximum position weight | {rules['max_weight_pct']*100:.0f}% |")
    lines.append(f"| Size warning threshold | {rules['size_warn_pct']*100:.0f}% |")
    lines.append(f"| Minimum alpha to hold | {rules['min_alpha_exit']:.0f} |")
    lines.append("")

    # ── Portfolio overview ────────────────────────────────────────────────────
    n_positions = len(positions_df)
    lines.append("---")
    lines.append("## Portfolio Overview")
    lines.append("")
    lines.append(f"Open positions: **{n_positions}**\n")

    if alerts_df.empty:
        lines.append("No alerts triggered at this time. All positions within rules.")
        lines.append("")
    else:
        n_urgent  = (alerts_df["severity"] == "URGENT").sum()
        n_warning = (alerts_df["severity"] == "WARNING").sum()
        n_monitor = (alerts_df["severity"] == "MONITOR").sum()

        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| URGENT   | {n_urgent} |")
        lines.append(f"| WARNING  | {n_warning} |")
        lines.append(f"| MONITOR  | {n_monitor} |")
        lines.append(f"| **Total**| **{len(alerts_df)}** |")
        lines.append("")

        # ── Alert table ───────────────────────────────────────────────────────
        lines.append("---")
        lines.append("## Alerts (sorted by severity)")
        lines.append("")
        lines.append(
            "| Ticker | Alert Type | Severity | P&L % | Alpha | Days Held | Detail |"
        )
        lines.append(
            "|--------|------------|----------|-------|-------|-----------|--------|"
        )

        for _, row in alerts_df.iterrows():
            pnl_str   = f"{row['pnl_pct']*100:+.2f}%" if pd.notna(row.get("pnl_pct")) else "—"
            alpha_str = f"{row['current_alpha']:.1f}"  if pd.notna(row.get("current_alpha")) else "—"
            days_str  = str(int(row["days_held"]))     if pd.notna(row.get("days_held")) else "—"
            lines.append(
                f"| **{row['ticker']}** "
                f"| {row['alert_type']} "
                f"| {row['severity']} "
                f"| {pnl_str} "
                f"| {alpha_str} "
                f"| {days_str} "
                f"| {str(row['detail'])[:120]} |"
            )
        lines.append("")

        # ── Recommended actions ───────────────────────────────────────────────
        lines.append("---")
        lines.append("## Recommended Actions")
        lines.append("")
        for _, row in alerts_df.iterrows():
            sev_tag = f"[{row['severity']}]"
            lines.append(
                f"- **{row['ticker']}** {sev_tag} `{row['alert_type']}` — "
                f"{row['recommended_action']}"
            )
        lines.append("")

    # ── Position detail table ─────────────────────────────────────────────────
    if not positions_df.empty:
        lines.append("---")
        lines.append("## Open Position Detail")
        lines.append("")
        lines.append(
            "| Ticker | Entry Date | Entry Price | Days Held | P&L % | Alpha | Weight % |"
        )
        lines.append(
            "|--------|------------|-------------|-----------|-------|-------|----------|"
        )
        for _, row in positions_df.iterrows():
            pnl_str   = f"{row.get('pnl_pct', np.nan)*100:+.2f}%" if pd.notna(row.get("pnl_pct")) else "—"
            alpha_str = f"{row.get('current_alpha', np.nan):.1f}"  if pd.notna(row.get("current_alpha")) else "—"
            wt_str    = f"{row.get('current_weight', np.nan)*100:.2f}%" if pd.notna(row.get("current_weight")) else "—"
            days_str  = str(int(row["days_held"])) if pd.notna(row.get("days_held")) else "—"
            lines.append(
                f"| {row['ticker']} "
                f"| {row.get('entry_date', '—')} "
                f"| ${_safe_float(row.get('entry_price', np.nan)):.2f} "
                f"| {days_str} "
                f"| {pnl_str} "
                f"| {alpha_str} "
                f"| {wt_str} |"
            )
        lines.append("")

    # ── Alert type reference ──────────────────────────────────────────────────
    lines.append("---")
    lines.append("## Alert Type Reference")
    lines.append("")
    lines.append("| Alert Type | Description |")
    lines.append("|------------|-------------|")
    for k, v in ALERT_TYPES.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("---")
    lines.append("_Canyon v9 Step 103 — research use only. Not investment advice._")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# [4/4]  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def print_alert_table(alerts_df: pd.DataFrame) -> None:
    """Print a formatted alert table to stdout."""
    if alerts_df.empty:
        print("\n  No alerts triggered.")
        return

    w = 100
    print(f"\n{'='*w}")
    print(f"  CANYON v9 — POSITION RULES ALERTS  ({len(alerts_df)} total)")
    print(f"{'='*w}")
    print(
        f"  {'Ticker':<8} {'Alert Type':<16} {'Severity':<9} "
        f"{'P&L %':>7} {'Alpha':>6} {'Days':>5}  Detail"
    )
    print(
        f"  {'-'*8} {'-'*16} {'-'*9} "
        f"{'-'*7} {'-'*6} {'-'*5}  {'-'*40}"
    )

    for _, row in alerts_df.iterrows():
        pnl_str   = f"{row['pnl_pct']*100:+.2f}%" if pd.notna(row.get("pnl_pct")) else "  —   "
        alpha_str = f"{row['current_alpha']:.1f}"  if pd.notna(row.get("current_alpha")) else "  —  "
        days_str  = str(int(row["days_held"]))     if pd.notna(row.get("days_held")) else "  —"
        detail    = str(row.get("detail", ""))[:60]
        print(
            f"  {row['ticker']:<8} {row['alert_type']:<16} {row['severity']:<9} "
            f"{pnl_str:>7} {alpha_str:>6} {days_str:>5}  {detail}"
        )

    print(f"{'='*w}\n")

    # Severity summary line
    n_urgent  = (alerts_df["severity"] == "URGENT").sum()
    n_warning = (alerts_df["severity"] == "WARNING").sum()
    n_monitor = (alerts_df["severity"] == "MONITOR").sum()
    print(f"  URGENT: {n_urgent}   WARNING: {n_warning}   MONITOR: {n_monitor}")

    # Recommended actions
    print(f"\n  Recommended Actions:")
    for _, row in alerts_df.iterrows():
        print(f"    [{row['severity']}] {row['ticker']} / {row['alert_type']}: "
              f"{row['recommended_action']}")

    print(f"\n  Outputs:")
    print(f"    {OUT_CSV}")
    print(f"    {OUT_REPORT}")
    print()


def run_position_rules(rules: dict) -> None:
    """
    Main position rules pipeline.

    1. Load open positions from paper_sim_trades.csv
    2. Get current prices
    3. Get current alpha scores
    4. Get current portfolio weights from daily_final.csv
    5. Load earnings_calendar.csv if exists
    6. For each position: check all rules, collect alerts
    7. Sort alerts: URGENT first, then WARNING, then MONITOR
    8. Write position_rules_alerts.csv
    9. Write position_rules_report.md
    10. Print formatted alert table
    """
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n[1/4] Loading positions and market data ...")
    positions = load_open_positions()

    # ── Empty positions → write empty outputs and exit ────────────────────────
    if positions.empty:
        print("\n  No open positions to check.\n")

        empty_alerts = pd.DataFrame(columns=[
            "ticker", "alert_type", "severity", "detail",
            "recommended_action", "pnl_pct", "current_alpha", "days_held",
        ])
        empty_alerts.to_csv(OUT_CSV, index=False)
        print(f"  CSV saved (0 rows): {OUT_CSV.name}")

        empty_positions = pd.DataFrame()
        write_report(empty_alerts, empty_positions, rules, run_ts)
        return

    tickers = positions["ticker"].tolist()

    print("\n[2/4] Fetching current prices and scores ...")
    prices  = get_current_prices(tickers)
    alphas  = get_alpha_scores(tickers)
    weights = get_portfolio_weights(tickers)
    earnings_map = get_earnings_days(tickers)
    regime  = get_current_regime()

    if regime:
        print(f"  Current regime: {regime}")
    else:
        print("  [INFO] regime_current.json not found — regime exit rule disabled")

    print(f"\n[3/4] Evaluating position rules for {len(tickers)} open position(s) ...")

    all_alerts: list[dict] = []
    enriched_rows: list[dict] = []   # positions with computed fields

    for _, trade in positions.iterrows():
        ticker = str(trade.get("ticker", "")).upper()

        current_price  = prices.get(ticker, np.nan)
        current_alpha  = alphas.get(ticker, np.nan)
        current_weight = weights.get(ticker, np.nan)
        earn_days      = earnings_map.get(ticker)

        pnl_pct   = compute_position_pnl(
            _safe_float(trade.get("entry_price", np.nan)), current_price
        )
        days_held = _compute_days_held(trade.get("entry_date"))

        # Check rules
        triggered = check_position_rules(
            trade_row=trade,
            current_price=current_price,
            current_alpha=current_alpha,
            current_weight=current_weight,
            earnings_days=earn_days,
            rules=rules,
            current_regime=regime,
        )

        # Annotate each alert with diagnostic columns
        for alert in triggered:
            alert["pnl_pct"]       = pnl_pct
            alert["current_alpha"] = current_alpha
            alert["days_held"]     = days_held if days_held is not None else np.nan
            all_alerts.append(alert)

        print(
            f"    {ticker:<8} price={current_price if not np.isnan(current_price) else 'N/A':>8}  "
            f"pnl={pnl_pct*100:+.2f}%" if not np.isnan(pnl_pct) else
            f"    {ticker:<8} price=N/A  pnl=N/A",
            f"  alpha={current_alpha:.1f}" if not np.isnan(current_alpha) else "  alpha=N/A",
            f"  alerts={len(triggered)}",
        )

        enriched_rows.append({
            "ticker":         ticker,
            "entry_date":     trade.get("entry_date"),
            "entry_price":    _safe_float(trade.get("entry_price", np.nan)),
            "days_held":      days_held if days_held is not None else np.nan,
            "pnl_pct":        pnl_pct,
            "current_alpha":  current_alpha,
            "current_weight": current_weight,
        })

    enriched_df = pd.DataFrame(enriched_rows)

    # ── Build and sort alerts DataFrame ───────────────────────────────────────
    print(f"\n[4/4] Writing outputs ...")

    ALERT_COLUMNS = [
        "ticker", "alert_type", "severity", "detail",
        "recommended_action", "pnl_pct", "current_alpha", "days_held",
    ]

    if not all_alerts:
        alerts_df = pd.DataFrame(columns=ALERT_COLUMNS)
    else:
        alerts_df = pd.DataFrame(all_alerts)
        # Sort: URGENT → WARNING → MONITOR, then by ticker alphabetically
        alerts_df["_sev_order"] = alerts_df["severity"].map(SEVERITY_ORDER).fillna(99)
        alerts_df = (
            alerts_df
            .sort_values(["_sev_order", "ticker", "alert_type"])
            .drop(columns=["_sev_order"])
            .reset_index(drop=True)
        )
        # Enforce column order
        out_cols = [c for c in ALERT_COLUMNS if c in alerts_df.columns]
        alerts_df = alerts_df[out_cols]

    alerts_df.to_csv(OUT_CSV, index=False)
    print(f"  CSV saved ({len(alerts_df)} alert row(s)): {OUT_CSV.name}")

    write_report(alerts_df, enriched_df, rules, run_ts)
    print_alert_table(alerts_df)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Canyon v9 Step 103 — Position Rules Engine. "
            "Checks open paper-sim positions against lifecycle rules. "
            "Research and paper trading only. No live orders."
        )
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_RULES["stop_loss_pct"],
        metavar="PCT",
        help=f"Stop-loss as decimal (default: {DEFAULT_RULES['stop_loss_pct']})",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=DEFAULT_RULES["profit_target_pct"],
        metavar="PCT",
        help=f"Profit target as decimal (default: {DEFAULT_RULES['profit_target_pct']})",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=int(DEFAULT_RULES["max_hold_days"]),
        metavar="DAYS",
        help=f"Maximum hold days (default: {int(DEFAULT_RULES['max_hold_days'])})",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_RULES["max_weight_pct"],
        metavar="PCT",
        help=f"Maximum position weight as decimal (default: {DEFAULT_RULES['max_weight_pct']})",
    )
    parser.add_argument(
        "--min-alpha-exit",
        type=float,
        default=DEFAULT_RULES["min_alpha_exit"],
        metavar="SCORE",
        help=f"Exit if alpha_score drops below this (default: {DEFAULT_RULES['min_alpha_exit']})",
    )
    args = parser.parse_args()

    # Derive size_warn_pct as 80% of max_weight (always below hard limit)
    size_warn = min(args.max_weight * 0.80, DEFAULT_RULES["size_warn_pct"])

    rules = {
        "stop_loss_pct":     args.stop_loss,
        "profit_target_pct": args.profit_target,
        "max_hold_days":     args.max_hold,
        "max_weight_pct":    args.max_weight,
        "min_alpha_exit":    args.min_alpha_exit,
        "size_warn_pct":     size_warn,
    }

    print("=" * 65)
    print("  Canyon v9 — Step 103: Position Rules Engine")
    print("  Research / paper trading only. No live orders.")
    print("=" * 65)
    print(f"  Stop-loss:     {rules['stop_loss_pct']*100:.0f}%")
    print(f"  Profit target: {rules['profit_target_pct']*100:.0f}%")
    print(f"  Max hold:      {rules['max_hold_days']} days")
    print(f"  Max weight:    {rules['max_weight_pct']*100:.0f}%")
    print(f"  Size warn:     {rules['size_warn_pct']*100:.0f}%")
    print(f"  Min alpha:     {rules['min_alpha_exit']:.0f}")

    run_position_rules(rules)


if __name__ == "__main__":
    main()
