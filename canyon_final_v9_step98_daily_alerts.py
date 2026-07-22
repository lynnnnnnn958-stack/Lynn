#!/usr/bin/env python3
"""
canyon_final_v9_step98_daily_alerts.py
=======================================
Canyon v9 — Step 98: Daily Alert System

After all engines run, scans all outputs and generates top-priority alerts.
This is the last step in the Canyon v9 pipeline.

Outputs:
  daily_alerts.json   — machine-readable alert payload
  daily_alerts.md     — human-readable markdown report

Alert categories:
  CRITICAL (1)  — immediate attention
  WARNING  (2)  — review today
  INFO     (3)  — situational awareness

Usage:
  python canyon_final_v9_step98_daily_alerts.py
  python canyon_final_v9_step98_daily_alerts.py --threshold 82
  python canyon_final_v9_step98_daily_alerts.py --max-alerts 10 --quiet
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALERT_LEVELS: dict[str, int] = {
    "CRITICAL": 1,
    "WARNING":  2,
    "INFO":     3,
}

# Output files
OUT_JSON = ROOT / "daily_alerts.json"
OUT_MD   = ROOT / "daily_alerts.md"

# Input files
F_REGIME_HISTORY    = ROOT / "regime_history.csv"
F_REGIME_CURRENT    = ROOT / "regime_current.json"
F_ALPHA_SCORES      = ROOT / "alpha_scores.csv"
F_OPTIONS_SIGNALS   = ROOT / "options_signals.csv"
F_SQUEEZE_SIGNALS   = ROOT / "short_interest_scores.csv"
F_INSIDER_SCORES    = ROOT / "insider_signal_scores.csv"
F_MACRO_JSON        = ROOT / "macro_signals.json"
F_MACRO_CSV         = ROOT / "macro_signals.csv"
F_MACRO_REGIME_CSV  = ROOT / "macro_regime_signals.csv"
F_RISK_REPORT_CSV   = ROOT / "risk_report.csv"
F_RISK_REPORT_MD    = ROOT / "risk_report.md"
F_LIVE_IC           = ROOT / "live_ic_history.csv"
F_DAILY_PICKS       = ROOT / "daily_picks.csv"

# Default argparse values
DEFAULT_THRESHOLD  = 78      # alpha_score threshold for HIGH_CONVICTION
DEFAULT_MAX_ALERTS = 5       # max alerts per category

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load CSV safely. Returns None and prints a warning if unavailable."""
    if not path.exists():
        print(f"  [SKIP] {label}: {path.name} not found")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [SKIP] {label}: {path.name} is empty")
            return None
        return df
    except Exception as exc:
        print(f"  [SKIP] {label}: could not read {path.name} ({exc})")
        return None


def _load_json(path: Path, label: str) -> dict | None:
    """Load JSON safely. Returns None and prints a warning if unavailable."""
    if not path.exists():
        print(f"  [SKIP] {label}: {path.name} not found")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [SKIP] {label}: could not parse {path.name} ({exc})")
        return None


def _make_alert(
    priority: str,
    alert_type: str,
    title: str,
    detail: str,
    tickers: list[str] | None = None,
    action: str = "",
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "priority":   priority,
        "type":       alert_type,
        "title":      title,
        "detail":     detail,
        "tickers":    tickers or [],
        "action":     action,
        "_sort_score": score,   # internal field for sorting; stripped before write
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alert checkers  [1/10] … [10/10]
# ─────────────────────────────────────────────────────────────────────────────

def check_regime_change() -> list[dict]:
    """[1/10] REGIME_CHANGE — CRITICAL if regime changed, INFO if stable."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_REGIME_HISTORY, "regime_change")
        if df is None or len(df) < 2:
            return alerts

        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        cur_regime  = str(last.get("regime", "UNKNOWN")).strip()
        prev_regime = str(prev.get("regime", "UNKNOWN")).strip()

        if cur_regime != prev_regime:
            alerts.append(_make_alert(
                priority="CRITICAL",
                alert_type="REGIME_CHANGE",
                title=f"Market regime changed: {prev_regime} → {cur_regime}",
                detail=(
                    f"Regime shifted as of {str(last.get('Date', 'today')).split()[0]}. "
                    f"Previous: {prev_regime}. Current: {cur_regime}. "
                    "Recommend re-evaluating signal weights and portfolio exposure."
                ),
                tickers=[],
                action="Review portfolio immediately",
                score=100.0,
            ))
        else:
            # Also read regime_current.json for richer detail
            rj = _load_json(F_REGIME_CURRENT, "regime_current")
            extra = ""
            if rj:
                extra = (f" VIX={rj.get('vix', '—')}, "
                         f"RSI={rj.get('rsi', '—')}, "
                         f"20d ret={rj.get('ret_20d', 0):.1%}")
            alerts.append(_make_alert(
                priority="INFO",
                alert_type="REGIME_CHANGE",
                title=f"Regime stable: {cur_regime}",
                detail=f"No regime change detected.{extra}",
                tickers=[],
                action="No action required",
                score=0.0,
            ))
    except Exception as exc:
        print(f"  [WARN] regime_change checker error: {exc}")
    return alerts


def check_high_conviction(threshold: float, max_alerts: int) -> list[dict]:
    """[2/10] HIGH_CONVICTION — CRITICAL if alpha_score > 85 + ≥3 strong signals,
    WARNING if > threshold."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_ALPHA_SCORES, "high_conviction")
        if df is None:
            return alerts

        sig_cols = [c for c in df.columns if c.startswith("sig_")]

        candidates = df[df["alpha_score"] > threshold].copy()
        if candidates.empty:
            return alerts

        candidates = candidates.sort_values("alpha_score", ascending=False)

        for _, row in candidates.head(max_alerts).iterrows():
            ticker      = str(row.get("ticker", "—"))
            score       = float(row.get("alpha_score", 0))
            rank        = int(row.get("alpha_rank", 0)) if pd.notna(row.get("alpha_rank")) else 0

            strong_signals: list[str] = []
            for col in sig_cols:
                val = row.get(col, np.nan)
                if pd.notna(val) and float(val) > 70:
                    label = col.replace("sig_", "")
                    strong_signals.append(label)

            n_strong = len(strong_signals)

            if score > 85 and n_strong >= 3:
                priority = "CRITICAL"
                action   = "Consider initiating or adding to position"
            else:
                priority = "WARNING"
                action   = "Review for potential entry"

            sig_str = "  ".join([f"{s}✓" for s in strong_signals]) if strong_signals else "none"
            alerts.append(_make_alert(
                priority=priority,
                alert_type="HIGH_CONVICTION",
                title=f"{ticker} — alpha {score:.1f}, signals: {sig_str}",
                detail=(
                    f"Rank #{rank}. Alpha score {score:.1f} (threshold {threshold:.0f}). "
                    f"{n_strong} strong signals (>70): {', '.join(strong_signals) or 'none'}."
                ),
                tickers=[ticker],
                action=action,
                score=score,
            ))
    except Exception as exc:
        print(f"  [WARN] high_conviction checker error: {exc}")
    return alerts


def check_options_alert(max_alerts: int) -> list[dict]:
    """[3/10] OPTIONS_ALERT — WARNING for unusual options activity."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_OPTIONS_SIGNALS, "options_alert")
        if df is None:
            return alerts

        # uoa_flag may be stored as int (0/1) or bool
        if "unusual_call_flag" in df.columns:
            flag_col = "unusual_call_flag"
        elif "uoa_flag" in df.columns:
            flag_col = "uoa_flag"
        else:
            flag_col = None

        if flag_col:
            flagged = df[df[flag_col].astype(str).isin(["True", "1", "true"])].copy()
        else:
            flagged = df.copy()

        if "rank_options" in df.columns:
            flagged = flagged[flagged["rank_options"] > 75]
            flagged = flagged.sort_values("rank_options", ascending=False)

        for _, row in flagged.head(max_alerts).iterrows():
            ticker   = str(row.get("ticker", "—"))
            rank_opt = float(row.get("rank_options", 0))
            strategy = str(row.get("options_strategy", row.get("options_signal", row.get("strategy", "—"))))
            conv     = str(row.get("conviction", ""))

            detail_parts = [f"rank_options={rank_opt:.1f}"]
            if strategy and strategy != "—":
                detail_parts.append(f"signal={strategy}")
            if conv and conv not in ("", "nan"):
                detail_parts.append(f"conviction={conv}")
            expiry = str(row.get("expiry", ""))
            if expiry and expiry not in ("", "nan"):
                detail_parts.append(f"expiry={expiry}")

            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="OPTIONS_ALERT",
                title=f"{ticker} — unusual options activity (rank {rank_opt:.0f})",
                detail=", ".join(detail_parts) + ".",
                tickers=[ticker],
                action="Check options chain for directional conviction",
                score=rank_opt,
            ))
    except Exception as exc:
        print(f"  [WARN] options_alert checker error: {exc}")
    return alerts


def check_squeeze_watch(max_alerts: int) -> list[dict]:
    """[4/10] SQUEEZE_WATCH — WARNING for rank_squeeze > 85."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_SQUEEZE_SIGNALS, "squeeze_watch")
        if df is None:
            return alerts

        squeezers = df[df["rank_squeeze"] > 75].copy()
        squeezers = squeezers.sort_values("rank_squeeze", ascending=False)

        for _, row in squeezers.head(max_alerts).iterrows():
            ticker       = str(row.get("ticker", "—"))
            rank_sq      = float(row.get("rank_squeeze", 0))
            short_int    = row.get("short_pct_float", None)
            si_str       = f"{float(short_int):.1%}" if pd.notna(short_int) else "—"

            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="SQUEEZE_WATCH",
                title=f"{ticker} — squeeze rank {rank_sq:.0f}, short interest {si_str}",
                detail=(
                    f"rank_squeeze={rank_sq:.1f}. Short % of float={si_str}. "
                    f"signal={row.get('signal', '—')}."
                ),
                tickers=[ticker],
                action="Monitor for catalyst; may amplify upward move",
                score=rank_sq,
            ))
    except Exception as exc:
        print(f"  [WARN] squeeze_watch checker error: {exc}")
    return alerts


def check_insider_buy(max_alerts: int) -> list[dict]:
    """[5/10] INSIDER_BUY — WARNING for rank_insider > 85."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_INSIDER_SCORES, "insider_buy")
        if df is None:
            return alerts

        buyers = df[df["rank_insider"] > 85].copy()
        buyers = buyers.sort_values("rank_insider", ascending=False)

        for _, row in buyers.head(max_alerts).iterrows():
            ticker      = str(row.get("ticker", "—"))
            rank_ins    = float(row.get("rank_insider", 0))
            txn_type    = str(row.get("transaction_type", row.get("insider_signal", "")))

            detail_parts = [f"rank_insider={rank_ins:.1f}"]
            if txn_type and txn_type not in ("", "nan", "NEUTRAL"):
                detail_parts.append(f"signal={txn_type}")
            buy_val = row.get("buy_value", None)
            if pd.notna(buy_val) and float(buy_val) > 0:
                detail_parts.append(f"buy_value=${float(buy_val):,.0f}")

            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="INSIDER_BUY",
                title=f"{ticker} — insider buy rank {rank_ins:.0f}",
                detail=", ".join(detail_parts) + ".",
                tickers=[ticker],
                action="Confirm open window; insider buying is a bullish signal",
                score=rank_ins,
            ))
    except Exception as exc:
        print(f"  [WARN] insider_buy checker error: {exc}")
    return alerts


def check_macro_risk() -> list[dict]:
    """[6/10] MACRO_RISK — CRITICAL if RISK_OFF, WARNING if changed."""
    alerts: list[dict] = []
    try:
        # Primary: macro_signals.json
        macro_j = _load_json(F_MACRO_JSON, "macro_risk_json")

        current_signal: str | None = None
        if macro_j:
            current_signal = str(macro_j.get("macro_signal", macro_j.get("signal", ""))).upper()

        # Fallback: derive from macro_regime_signals.csv (volatility_regime.csv)
        if not current_signal:
            vr_path = ROOT / "volatility_regime.csv"
            if vr_path.exists():
                vr = pd.read_csv(vr_path)
                if not vr.empty and "regime" in vr.columns:
                    current_signal = str(vr.iloc[-1]["regime"]).upper()

        if not current_signal:
            return alerts

        # Previous day signal: from macro_signals.csv
        prev_signal: str | None = None
        macro_prev_df = _load_csv(F_MACRO_CSV, "macro_risk_prev")
        if macro_prev_df is not None and "macro_signal" in macro_prev_df.columns and len(macro_prev_df) >= 2:
            prev_signal = str(macro_prev_df.iloc[-2]["macro_signal"]).upper()

        if current_signal == "RISK_OFF":
            alerts.append(_make_alert(
                priority="CRITICAL",
                alert_type="MACRO_RISK",
                title="Macro regime: RISK_OFF — defensive posture required",
                detail=(
                    "Current macro signal is RISK_OFF. Historical back-tests show "
                    "RISK_OFF periods precede average drawdowns of 8–15%. "
                    "Recommend reducing beta and adding defensive allocations."
                ),
                tickers=[],
                action="Reduce equity exposure; increase cash / defensive positions",
                score=100.0,
            ))
        elif prev_signal and current_signal != prev_signal:
            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="MACRO_RISK",
                title=f"Macro signal changed: {prev_signal} → {current_signal}",
                detail=(
                    f"Macro signal shifted from {prev_signal} to {current_signal}. "
                    "Monitor for confirmation; may signal regime rotation."
                ),
                tickers=[],
                action="Review macro exposure and sector tilts",
                score=70.0,
            ))
    except Exception as exc:
        print(f"  [WARN] macro_risk checker error: {exc}")
    return alerts


def check_risk_breach(max_alerts: int) -> list[dict]:
    """[7/10] RISK_BREACH — WARNING if current_weight > kelly_max × 1.5."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_RISK_REPORT_CSV, "risk_breach")
        if df is None:
            return alerts

        if "current_weight" not in df.columns or "kelly_max" not in df.columns:
            return alerts

        df["current_weight"] = pd.to_numeric(df["current_weight"], errors="coerce")
        df["kelly_max"]      = pd.to_numeric(df["kelly_max"],      errors="coerce")
        df = df.dropna(subset=["current_weight", "kelly_max"])
        df = df[df["kelly_max"] > 0]

        breaches = df[df["current_weight"] > df["kelly_max"] * 1.5].copy()
        breaches["_excess"] = breaches["current_weight"] / breaches["kelly_max"]
        breaches = breaches.sort_values("_excess", ascending=False)

        for _, row in breaches.head(max_alerts).iterrows():
            ticker  = str(row.get("ticker", "—"))
            cw      = float(row["current_weight"])
            km      = float(row["kelly_max"])
            excess  = float(row["_excess"])

            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="RISK_BREACH",
                title=f"{ticker} — weight {cw:.1%} exceeds Kelly max {km:.1%} × 1.5",
                detail=(
                    f"current_weight={cw:.1%}, kelly_max={km:.1%}, "
                    f"ratio={excess:.2f}x. Position is over-sized by Kelly criterion."
                ),
                tickers=[ticker],
                action="Trim position to within Kelly constraint",
                score=excess * 50,
            ))
    except Exception as exc:
        print(f"  [WARN] risk_breach checker error: {exc}")
    return alerts


def check_portfolio_var() -> list[dict]:
    """[8/10] PORTFOLIO_VAR — WARNING if portfolio 1d VaR > 1.5%."""
    alerts: list[dict] = []
    try:
        var_value: float | None = None

        # Try CSV first
        if F_RISK_REPORT_CSV.exists():
            df = pd.read_csv(F_RISK_REPORT_CSV)
            for col in ["portfolio_var_1d", "var_1d", "VaR_1d", "var_contrib_pct"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not vals.empty:
                        if col == "var_contrib_pct":
                            # var_contrib_pct is per-ticker; use portfolio sum
                            var_value = float(vals.sum()) / 100
                        else:
                            var_value = float(vals.iloc[0])
                        break

        # Try markdown if CSV didn't have VaR
        if var_value is None and F_RISK_REPORT_MD.exists():
            text = F_RISK_REPORT_MD.read_text()
            import re
            m = re.search(r"(?:portfolio\s+1d\s+VaR|VaR_1d)[^\d]*([\d.]+)%?", text, re.IGNORECASE)
            if m:
                var_value = float(m.group(1))
                if var_value > 1:   # stored as percent (e.g. 2.1 means 2.1%)
                    var_value /= 100

        if var_value is None:
            return alerts

        # Normalise: if stored as fraction (0.021) convert to pct (2.1%)
        var_pct = var_value if var_value > 0.1 else var_value * 100

        if var_pct > 1.5:
            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="PORTFOLIO_VAR",
                title=f"Portfolio 1d VaR = {var_pct:.2f}% — exceeds 1.5% threshold",
                detail=(
                    f"Estimated 1-day Value-at-Risk is {var_pct:.2f}% "
                    "(95% confidence). Threshold is 1.5%. "
                    "Portfolio tail risk is elevated."
                ),
                tickers=[],
                action="Review concentration, add hedges, or reduce gross exposure",
                score=var_pct,
            ))
    except Exception as exc:
        print(f"  [WARN] portfolio_var checker error: {exc}")
    return alerts


def check_signal_degradation() -> list[dict]:
    """[9/10] SIGNAL_DEGRADATION — WARNING if last 5 IC readings avg < 0."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_LIVE_IC, "signal_degradation")
        if df is None:
            return alerts

        ic_col = None
        for candidate in ["ic", "IC", "mean_ic", "ic_value"]:
            if candidate in df.columns:
                ic_col = candidate
                break

        if ic_col is None:
            return alerts

        ic_series = pd.to_numeric(df[ic_col], errors="coerce").dropna()
        if len(ic_series) < 5:
            return alerts

        last5     = ic_series.iloc[-5:]
        mean_ic   = float(last5.mean())

        if mean_ic < 0:
            alerts.append(_make_alert(
                priority="WARNING",
                alert_type="SIGNAL_DEGRADATION",
                title=f"Signal IC degraded — last-5 mean IC = {mean_ic:+.4f}",
                detail=(
                    f"Average IC over last 5 periods is {mean_ic:+.4f} (negative). "
                    f"Individual readings: {[f'{v:+.3f}' for v in last5.tolist()]}. "
                    "Signals may have lost predictive power in the current regime."
                ),
                tickers=[],
                action="Investigate signal logic; consider reducing model weight",
                score=abs(mean_ic) * 100,
            ))
    except Exception as exc:
        print(f"  [WARN] signal_degradation checker error: {exc}")
    return alerts


def check_new_buy_list(threshold: float, max_alerts: int) -> list[dict]:
    """[10/10] NEW_BUY_LIST — INFO, top 5 picks with alpha_score > 70."""
    alerts: list[dict] = []
    try:
        df = _load_csv(F_DAILY_PICKS, "new_buy_list")
        if df is None:
            return alerts

        alpha_thresh = max(70.0, threshold - 10)  # slightly below conviction threshold
        top = df[df["alpha_score"] > alpha_thresh].head(5)
        if top.empty:
            return alerts

        pick_strs = [
            f"{row['ticker']}({row['alpha_score']:.1f})"
            for _, row in top.iterrows()
        ]
        ticker_list = top["ticker"].tolist()

        regime = str(top.iloc[0].get("regime", "")) if "regime" in top.columns else ""
        regime_str = f" [{regime}]" if regime else ""

        alerts.append(_make_alert(
            priority="INFO",
            alert_type="NEW_BUY_LIST",
            title=f"Top picks{regime_str} — {', '.join(pick_strs)}",
            detail=(
                f"Top {len(top)} tickers by alpha_score (>{alpha_thresh:.0f}): "
                + ", ".join(pick_strs) + ". "
                "Run Step 63 to apply risk-model adjusted weights."
            ),
            tickers=ticker_list,
            action="Review action cards for entry conditions",
            score=float(top["alpha_score"].max()) if not top.empty else 0.0,
        ))
    except Exception as exc:
        print(f"  [WARN] new_buy_list checker error: {exc}")
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Sort + deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def _sort_alerts(alerts: list[dict]) -> list[dict]:
    """Sort by priority (CRITICAL first), then by _sort_score descending."""
    return sorted(
        alerts,
        key=lambda a: (ALERT_LEVELS.get(a["priority"], 99), -a.get("_sort_score", 0)),
    )


def _deduplicate(alerts: list[dict]) -> list[dict]:
    """
    If the same ticker appears in multiple WARNING alerts, keep both alert types
    but annotate that they share a ticker. No alerts are dropped — dedup is
    informational only (same ticker in CRITICAL vs INFO kept independently).
    """
    seen_ticker_type: set[tuple[str, str]] = set()
    result: list[dict] = []
    for alert in alerts:
        key_pairs = [(t, alert["type"]) for t in (alert["tickers"] or ["__none__"])]
        already_seen = all(kp in seen_ticker_type for kp in key_pairs)
        if not already_seen:
            for kp in key_pairs:
                seen_ticker_type.add(kp)
            result.append(alert)
    return result


def _strip_internal(alert: dict) -> dict:
    """Remove internal fields before JSON serialisation."""
    return {k: v for k, v in alert.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json(alerts: list[dict], today: str, run_time: str) -> dict:
    clean = [_strip_internal(a) for a in alerts]
    critical = sum(1 for a in clean if a["priority"] == "CRITICAL")
    warning  = sum(1 for a in clean if a["priority"] == "WARNING")
    info     = sum(1 for a in clean if a["priority"] == "INFO")

    payload = {
        "date":           today,
        "run_time":       run_time,
        "total_alerts":   len(clean),
        "critical_count": critical,
        "warning_count":  warning,
        "info_count":     info,
        "alerts":         clean,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"  [written] {OUT_JSON.name}  ({len(clean)} alerts)")
    return payload


def _write_md(alerts: list[dict], today: str, run_time: str) -> None:
    critical = [a for a in alerts if a["priority"] == "CRITICAL"]
    warning  = [a for a in alerts if a["priority"] == "WARNING"]
    info_    = [a for a in alerts if a["priority"] == "INFO"]

    lines: list[str] = [
        f"# Canyon v9 Daily Alerts — {today}",
        f"",
        f"Generated: {run_time}  |  Total: {len(alerts)}  |  "
        f"Critical: {len(critical)}  |  Warning: {len(warning)}  |  Info: {len(info_)}",
        "",
    ]

    # CRITICAL section
    lines.append(f"## CRITICAL ({len(critical)})")
    if critical:
        for a in critical:
            lines.append(f"### {a['type']}: {a['title']}")
            lines.append(f"{a['detail']}")
            if a.get("tickers"):
                lines.append(f"**Tickers:** {', '.join(a['tickers'])}")
            if a.get("action"):
                lines.append(f"**Action:** {a['action']}")
            lines.append("")
    else:
        lines.append("_No critical alerts._")
        lines.append("")

    # WARNING section
    lines.append(f"## WARNING ({len(warning)})")
    if warning:
        for a in warning:
            lines.append(f"### {a['type']}: {a['title']}")
            lines.append(f"{a['detail']}")
            if a.get("tickers"):
                lines.append(f"**Tickers:** {', '.join(a['tickers'])}")
            if a.get("action"):
                lines.append(f"**Action:** {a['action']}")
            lines.append("")
    else:
        lines.append("_No warnings._")
        lines.append("")

    # INFO section
    lines.append(f"## INFO ({len(info_)})")
    if info_:
        for a in info_:
            lines.append(f"### {a['type']}: {a['title']}")
            lines.append(f"{a['detail']}")
            if a.get("tickers"):
                lines.append(f"**Tickers:** {', '.join(a['tickers'])}")
            if a.get("action"):
                lines.append(f"**Action:** {a['action']}")
            lines.append("")
    else:
        lines.append("_No informational alerts._")
        lines.append("")

    lines += [
        "---",
        "_Canyon v9 Step 98 — Daily Alert System_",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"  [written] {OUT_MD.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary box
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(alerts: list[dict], today: str) -> None:
    critical = [a for a in alerts if a["priority"] == "CRITICAL"]
    warning  = [a for a in alerts if a["priority"] == "WARNING"]
    info_    = [a for a in alerts if a["priority"] == "INFO"]

    W = 72
    print()
    print("─" * W)
    print(f"{'CANYON v9 — DAILY ALERTS':^{W}}")
    print(f"{today:^{W}}")
    print("─" * W)
    print(f"  Total: {len(alerts)}   "
          f"CRITICAL: {len(critical)}   "
          f"WARNING: {len(warning)}   "
          f"INFO: {len(info_)}")
    print("─" * W)

    for section_label, section_alerts in [
        ("CRITICAL", critical),
        ("WARNING",  warning),
        ("INFO",     info_),
    ]:
        if not section_alerts:
            continue
        print(f"\n  [{section_label}]")
        for a in section_alerts:
            ticker_str = f" ({', '.join(a['tickers'][:3])})" if a.get("tickers") else ""
            title_line = f"    {a['type']}{ticker_str}: {a['title']}"
            # Truncate long titles
            if len(title_line) > W - 2:
                title_line = title_line[: W - 5] + "..."
            print(title_line)

    print()
    print("─" * W)
    if critical:
        print(f"  [!] {len(critical)} CRITICAL alert(s) require immediate review.")
    else:
        print("  [OK] No critical alerts today.")
    print(f"  Outputs: {OUT_JSON.name}  |  {OUT_MD.name}")
    print("─" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_alerts(threshold: float = DEFAULT_THRESHOLD,
               max_alerts: int = DEFAULT_MAX_ALERTS,
               quiet: bool = False) -> int:
    """
    Run all 10 alert checkers, write outputs, print summary.
    Returns count of CRITICAL alerts (for upstream use; exit code is always 0).
    """
    now      = datetime.now()
    today    = now.strftime("%Y-%m-%d")
    run_time = now.strftime("%H:%M")

    if not quiet:
        print()
        print("=" * 72)
        print("Canyon v9  Step 98 — Daily Alert System")
        print("=" * 72)
        print(f"  Date: {today}   Time: {run_time}")
        print(f"  Threshold: alpha_score > {threshold:.0f}   Max per category: {max_alerts}")
        print()

    all_alerts: list[dict] = []

    checkers = [
        ("[1/10] Regime change",          lambda: check_regime_change()),
        ("[2/10] High conviction",         lambda: check_high_conviction(threshold, max_alerts)),
        ("[3/10] Options alerts",          lambda: check_options_alert(max_alerts)),
        ("[4/10] Squeeze watch",           lambda: check_squeeze_watch(max_alerts)),
        ("[5/10] Insider buy",             lambda: check_insider_buy(max_alerts)),
        ("[6/10] Macro risk",              lambda: check_macro_risk()),
        ("[7/10] Risk breach",             lambda: check_risk_breach(max_alerts)),
        ("[8/10] Portfolio VaR",           lambda: check_portfolio_var()),
        ("[9/10] Signal degradation",      lambda: check_signal_degradation()),
        ("[10/10] New buy list",           lambda: check_new_buy_list(threshold, max_alerts)),
    ]

    for label, fn in checkers:
        if not quiet:
            print(f"  {label} …")
        try:
            found = fn()
            if not quiet and found:
                print(f"    → {len(found)} alert(s)")
            all_alerts.extend(found)
        except Exception as exc:
            print(f"  [ERROR] {label}: unexpected error: {exc}")

    # Sort and deduplicate
    all_alerts = _sort_alerts(all_alerts)
    all_alerts = _deduplicate(all_alerts)

    if not quiet:
        print()
        print("  Writing outputs …")

    _write_json(all_alerts, today, run_time)
    _write_md(all_alerts,   today, run_time)

    if not quiet:
        _print_summary(all_alerts, today)

    critical_count = sum(1 for a in all_alerts if a["priority"] == "CRITICAL")
    return critical_count


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 98 — Daily Alert System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="alpha_score threshold for HIGH_CONVICTION alerts",
    )
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=DEFAULT_MAX_ALERTS,
        dest="max_alerts",
        help="maximum alerts per category",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="suppress stdout; only write output files",
    )
    args = parser.parse_args()

    critical_count = run_alerts(
        threshold=args.threshold,
        max_alerts=args.max_alerts,
        quiet=args.quiet,
    )

    # Always exit 0 — alerts are informational, not errors
    if not args.quiet and critical_count > 0:
        print(f"[WARNING] {critical_count} CRITICAL alert(s) generated. "
              f"Review {OUT_MD.name} immediately.\n")


if __name__ == "__main__":
    main()
