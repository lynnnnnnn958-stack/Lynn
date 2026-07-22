#!/usr/bin/env python3
"""
canyon_final_v9_step108_weekly_report.py
=========================================
Canyon v9 Step 108 — Weekly Performance Report Generator

Aggregates the last 5 trading days of data into a concise weekly summary:
  - Portfolio performance vs SPY benchmark
  - Top/bottom picks by alpha score change
  - Regime transitions during the week
  - Signal-level IC summary
  - P&L attribution breakdown
  - Risk metrics snapshot (VaR, beta, HHI)
  - Macro signal status
  - Upcoming earnings risk

Outputs:
  weekly_report.md     — full markdown report
  weekly_summary.json  — machine-readable summary dict

Usage:
  python canyon_final_v9_step108_weekly_report.py             # last 5 days
  python canyon_final_v9_step108_weekly_report.py --days 10   # last 10 days
  python canyon_final_v9_step108_weekly_report.py --dry-run   # print only
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# [1/4] Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_alpha_scores() -> pd.DataFrame:
    """Load current alpha_scores.csv."""
    path = ROOT / "alpha_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _load_alpha_history(days: int) -> pd.DataFrame:
    """Load alpha_score_history.csv, filtered to last `days` calendar days."""
    path = ROOT / "alpha_score_history.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns or "ticker" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days + 2)
        df = df[df["date"] >= cutoff].copy()
        return df
    except Exception:
        return pd.DataFrame()


def _load_daily_picks() -> pd.DataFrame:
    """Load daily_picks.csv — today's buy list."""
    path = ROOT / "daily_picks.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_price_returns(tickers: list[str], days: int) -> pd.DataFrame:
    """
    Load sp500_price_cache.csv (wide: date rows × ticker cols).
    Returns a DataFrame of close prices for the last `days+5` rows.
    """
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
        # Keep only relevant tickers + SPY
        keep = [t for t in (tickers + ["SPY"]) if t in df.columns]
        df = df[keep].tail(days + 5)
        return df
    except Exception:
        return pd.DataFrame()


def _load_regime() -> dict[str, str]:
    """Load current regime from regime_current.json."""
    path = ROOT / "regime_current.json"
    if not path.exists():
        return {"regime": "UNKNOWN", "confidence": "N/A"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"regime": "UNKNOWN", "confidence": "N/A"}


def _load_risk_report() -> dict[str, Any]:
    """Load risk_report.csv — returns dict of first-row metrics."""
    path = ROOT / "risk_report.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}


def _load_macro_signals() -> dict[str, Any]:
    """Load macro_signals.json."""
    path = ROOT / "macro_signals.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_pnl_attribution() -> pd.DataFrame:
    """Load pnl_attribution.csv."""
    path = ROOT / "pnl_attribution.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _load_earnings_calendar() -> pd.DataFrame:
    """Load earnings_calendar.csv."""
    path = ROOT / "earnings_calendar.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _load_signal_weights() -> dict[str, Any]:
    """Load signal_weights.json (IC calibration output)."""
    path = ROOT / "signal_weights.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_journal() -> pd.DataFrame:
    """Load trade_journal.csv."""
    path = ROOT / "trade_journal.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# [2/4] Computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_week_return(prices: pd.DataFrame, ticker: str, days: int) -> float:
    """Return simple % return over the last `days` trading rows, or NaN."""
    if ticker not in prices.columns:
        return float("nan")
    series = prices[ticker].dropna()
    if len(series) < 2:
        return float("nan")
    series = series.tail(days + 1)
    return float((series.iloc[-1] / series.iloc[0] - 1) * 100)


def _compute_alpha_change(history: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Return DataFrame with ticker, alpha_start, alpha_end, alpha_change
    over the past `days` calendar days.
    """
    if history.empty or "alpha_score" not in history.columns:
        return pd.DataFrame()
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")

    results = []
    for ticker, grp in history.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < 2:
            continue
        alpha_end   = float(grp["alpha_score"].iloc[-1])
        alpha_start = float(grp["alpha_score"].iloc[0])
        results.append({
            "ticker":       ticker,
            "alpha_start":  round(alpha_start, 1),
            "alpha_end":    round(alpha_end, 1),
            "alpha_change": round(alpha_end - alpha_start, 1),
        })
    return pd.DataFrame(results)


def _regime_transitions(history: pd.DataFrame) -> list[str]:
    """
    Look for regime column in alpha_score_history.
    Returns list of transition strings like 'SIDEWAYS → BULL'.
    """
    if history.empty or "regime" not in history.columns:
        return []
    regimes = (
        history.groupby("date")["regime"]
        .agg(lambda x: x.mode()[0] if not x.empty else "UNKNOWN")
        .sort_index()
    )
    transitions = []
    prev = None
    for date, reg in regimes.items():
        if prev is not None and reg != prev:
            transitions.append(
                f"{date.strftime('%Y-%m-%d')}: {prev} → {reg}"
            )
        prev = reg
    return transitions


def _portfolio_return(prices: pd.DataFrame, picks: pd.DataFrame, days: int) -> float:
    """
    Weighted portfolio return. Uses weight_pct column (0-100 scale) or equal weight.
    """
    if picks.empty or prices.empty:
        return float("nan")

    tickers_in = [t for t in picks.get("ticker", pd.Series()).tolist()
                  if t in prices.columns]
    if not tickers_in:
        return float("nan")

    weights = {}
    if "weight_pct" in picks.columns:
        for _, row in picks.iterrows():
            t = str(row.get("ticker", ""))
            if t in tickers_in:
                weights[t] = float(row.get("weight_pct", 0)) / 100.0
    else:
        eq = 1.0 / len(tickers_in)
        weights = {t: eq for t in tickers_in}

    total_w = sum(weights.values()) or 1.0
    port_ret = 0.0
    for t, w in weights.items():
        r = _compute_week_return(prices, t, days)
        if not math.isnan(r):
            port_ret += (w / total_w) * r
    return round(port_ret, 2)


def _grade_distribution(journal: pd.DataFrame) -> dict[str, int]:
    """Count trades by grade in the journal (last ~5 days by exit_date)."""
    if journal.empty or "grade" not in journal.columns:
        return {}
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if "exit_date" in journal.columns:
        recent = journal[journal["exit_date"].fillna("") >= cutoff]
    else:
        recent = journal
    counts = recent["grade"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# [3/4] Report builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(title: str) -> str:
    return f"\n## {title}\n\n"


def _build_performance_section(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    days: int,
    alpha_change: pd.DataFrame,
) -> str:
    spy_ret  = _compute_week_return(prices, "SPY", days)
    port_ret = _portfolio_return(prices, picks, days)

    lines = [_section_header("📈 Portfolio Performance")]

    spy_str  = f"{spy_ret:+.2f}%" if not math.isnan(spy_ret)  else "N/A"
    port_str = f"{port_ret:+.2f}%" if not math.isnan(port_ret) else "N/A"

    if not math.isnan(spy_ret) and not math.isnan(port_ret):
        excess = port_ret - spy_ret
        excess_str = f"{excess:+.2f}%"
        verdict = "✅ OUTPERFORM" if excess > 0 else ("⚠️ INLINE" if abs(excess) < 0.5 else "❌ UNDERPERFORM")
    else:
        excess_str = "N/A"
        verdict = "N/A"

    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Portfolio return ({days}d) | **{port_str}** |\n")
    lines.append(f"| SPY return ({days}d)       | {spy_str} |\n")
    lines.append(f"| Excess return             | {excess_str} |\n")
    lines.append(f"| Assessment                | {verdict} |\n")

    if not alpha_change.empty:
        top3 = alpha_change.nlargest(3, "alpha_change")
        bot3 = alpha_change.nsmallest(3, "alpha_change")
        lines.append("\n### Top Alpha Movers (Week)\n")
        lines.append("| Ticker | Alpha Start | Alpha End | Δ Alpha |\n")
        lines.append("|--------|------------|-----------|----------|\n")
        for _, row in top3.iterrows():
            sign = "+" if row["alpha_change"] >= 0 else ""
            lines.append(f"| {row['ticker']} | {row['alpha_start']:.1f} | {row['alpha_end']:.1f} | **{sign}{row['alpha_change']:.1f}** |\n")
        lines.append("\n### Bottom Alpha Movers (Week)\n")
        lines.append("| Ticker | Alpha Start | Alpha End | Δ Alpha |\n")
        lines.append("|--------|------------|-----------|----------|\n")
        for _, row in bot3.iterrows():
            sign = "+" if row["alpha_change"] >= 0 else ""
            lines.append(f"| {row['ticker']} | {row['alpha_start']:.1f} | {row['alpha_end']:.1f} | {sign}{row['alpha_change']:.1f} |\n")

    return "".join(lines)


def _build_regime_section(history: pd.DataFrame) -> str:
    regime_info = _load_regime()
    transitions = _regime_transitions(history)

    lines = [_section_header("🔄 Regime Status")]
    lines.append(f"**Current Regime:** `{regime_info.get('regime', 'UNKNOWN')}`  \n")
    conf = regime_info.get("confidence", regime_info.get("prob", "N/A"))
    lines.append(f"**Confidence:** {conf}\n\n")

    if transitions:
        lines.append("**Regime Transitions This Period:**\n")
        for t in transitions:
            lines.append(f"- {t}\n")
    else:
        lines.append("_No regime transitions detected this period._\n")

    return "".join(lines)


def _build_risk_section() -> str:
    risk = _load_risk_report()
    lines = [_section_header("🛡️ Risk Metrics Snapshot")]

    if not risk:
        lines.append("_Risk report not available — run step93 first._\n")
        return "".join(lines)

    metrics = [
        ("Portfolio VaR (95%)",   "var_95",    "{:.2f}%"),
        ("Portfolio CVaR (95%)",  "cvar_95",   "{:.2f}%"),
        ("Portfolio Beta",        "beta",      "{:.3f}"),
        ("HHI Concentration",     "hhi",       "{:.4f}"),
        ("Kelly Fraction",        "kelly",     "{:.2f}"),
        ("Positions Count",       "n_positions","{:.0f}"),
    ]

    lines.append("| Metric | Value |\n|--------|-------|\n")
    for label, key, fmt in metrics:
        val = risk.get(key, risk.get(key.replace("_", ""), "N/A"))
        if val != "N/A":
            try:
                val_str = fmt.format(float(val))
            except Exception:
                val_str = str(val)
        else:
            val_str = "N/A"
        lines.append(f"| {label} | {val_str} |\n")

    return "".join(lines)


def _build_macro_section() -> str:
    macro = _load_macro_signals()
    lines = [_section_header("🌍 Macro Signals")]

    if not macro:
        lines.append("_Macro signals not available — run step95 first._\n")
        return "".join(lines)

    composite = macro.get("composite_score", macro.get("macro_composite_score", None))
    if composite is not None:
        try:
            c = float(composite)
            trend = "🟢 BULLISH" if c >= 60 else ("🔴 BEARISH" if c <= 40 else "🟡 NEUTRAL")
            lines.append(f"**Macro Composite Score:** {c:.1f} / 100 — {trend}\n\n")
        except Exception:
            pass

    signals = macro.get("signals", {})
    if signals:
        lines.append("| Signal | Score | Interpretation |\n")
        lines.append("|--------|-------|----------------|\n")
        for name, data in signals.items():
            if isinstance(data, dict):
                score = data.get("score", "N/A")
                interp = data.get("interpretation", data.get("signal", ""))
            else:
                score = data
                interp = ""
            try:
                score_f = float(score)
                dot = "🟢" if score_f >= 60 else ("🔴" if score_f <= 40 else "🟡")
                score_str = f"{dot} {score_f:.1f}"
            except Exception:
                score_str = str(score)
            lines.append(f"| {name} | {score_str} | {interp} |\n")

    return "".join(lines)


def _build_attribution_section() -> str:
    attr = _load_pnl_attribution()
    lines = [_section_header("📊 P&L Attribution")]

    if attr.empty:
        lines.append("_P&L attribution not available — run step96 first._\n")
        return "".join(lines)

    # By signal
    if "signal_name" in attr.columns and "attributed_pnl" in attr.columns:
        by_sig = attr.groupby("signal_name")["attributed_pnl"].sum().sort_values(ascending=False)
        if not by_sig.empty:
            lines.append("**By Signal:**\n\n")
            lines.append("| Signal | Attributed P&L |\n|--------|----------------|\n")
            for sig, pnl in by_sig.head(8).items():
                sign = "+" if pnl >= 0 else ""
                lines.append(f"| {sig} | {sign}{pnl:.2f}% |\n")

    # By grade
    if "grade" in attr.columns and "attributed_pnl" in attr.columns:
        by_grade = attr.groupby("grade")["attributed_pnl"].sum().sort_values(ascending=False)
        if not by_grade.empty:
            lines.append("\n**By Trade Grade:**\n\n")
            lines.append("| Grade | Attributed P&L |\n|-------|----------------|\n")
            for grade, pnl in by_grade.items():
                sign = "+" if pnl >= 0 else ""
                lines.append(f"| {grade} | {sign}{pnl:.2f}% |\n")

    return "".join(lines)


def _build_ic_section() -> str:
    sw = _load_signal_weights()
    lines = [_section_header("🎯 Signal IC Summary")]

    if not sw:
        lines.append("_Signal calibration data not available — run step94 first._\n")
        return "".join(lines)

    raw_ic = sw.get("raw_ic", {})
    ic_mult = sw.get("ic_multipliers", {})
    regime = sw.get("regime", "N/A")
    updated = sw.get("updated", "N/A")

    lines.append(f"**Calibration Regime:** `{regime}`  \n")
    lines.append(f"**Last Updated:** {updated}\n\n")

    if raw_ic:
        lines.append("| Signal | IC | Multiplier | Interpretation |\n")
        lines.append("|--------|-----|------------|----------------|\n")
        for sig, ic_val in sorted(
            ((s, v) for s, v in raw_ic.items() if v is not None),
            key=lambda x: abs(float(x[1])), reverse=True
        ):
            try:
                ic_f    = float(ic_val)
                mult_f  = float(ic_mult.get(sig, 1.0))
                interp  = "✅ Strong" if abs(ic_f) > 0.05 else ("➡️ Modest" if abs(ic_f) > 0.02 else "⚠️ Weak")
                lines.append(f"| {sig} | {ic_f:.4f} | ×{mult_f:.2f} | {interp} |\n")
            except Exception:
                lines.append(f"| {sig} | {ic_val} | {ic_mult.get(sig, '—')} | — |\n")

    return "".join(lines)


def _build_earnings_section() -> str:
    ec = _load_earnings_calendar()
    lines = [_section_header("📅 Upcoming Earnings Risk")]

    if ec.empty:
        lines.append("_Earnings calendar not available — run step102 first._\n")
        return "".join(lines)

    # Filter to next 7 days
    risky = ec[ec.get("days_until_earnings", pd.Series(dtype=float)).fillna(999) <= 7] \
        if "days_until_earnings" in ec.columns else ec.head(10)

    if risky.empty:
        lines.append("_No portfolio holdings have earnings in the next 7 days._\n")
        return "".join(lines)

    lines.append("| Ticker | Days | Risk Level | IV Expansion | Action |\n")
    lines.append("|--------|------|------------|--------------|--------|\n")
    for _, row in risky.iterrows():
        ticker    = row.get("ticker", "?")
        days_     = row.get("days_until_earnings", "?")
        risk      = row.get("earnings_risk", "NORMAL")
        iv_exp    = row.get("iv_expansion_quality", "—")
        action    = row.get("recommended_action", "MONITOR")
        flag      = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟢")
        lines.append(f"| {ticker} | {days_} | {flag} {risk} | {iv_exp} | {action} |\n")

    return "".join(lines)


def _build_buys_section(picks: pd.DataFrame) -> str:
    lines = [_section_header("🎯 This Week's Buy List")]

    if picks.empty:
        lines.append("_No buy-list picks available._\n")
        return "".join(lines)

    buy_cols = ["ticker", "alpha_score", "signal", "regime_label", "sector"]
    show_cols = [c for c in buy_cols if c in picks.columns]

    sub = picks.head(15)[show_cols].copy()
    lines.append(sub.to_markdown(index=False))
    lines.append("\n")
    return "".join(lines)


def _build_journal_section(journal: pd.DataFrame) -> str:
    lines = [_section_header("📔 Recent Trades")]

    if journal.empty:
        lines.append("_Trade journal empty — run step89 to populate._\n")
        return "".join(lines)

    dist = _grade_distribution(journal)
    if dist:
        lines.append("**Grade Distribution (last 30 days):** ")
        lines.append("  ".join(f"{g}: {n}" for g, n in sorted(dist.items())))
        lines.append("\n\n")

    j_cols = ["exit_date", "ticker", "return_pct", "grade", "strategy"]
    show   = [c for c in j_cols if c in journal.columns]
    recent = journal.sort_values("exit_date", ascending=False).head(10)[show] \
             if "exit_date" in journal.columns else journal.head(10)[show]

    lines.append(recent.to_markdown(index=False))
    lines.append("\n")
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# [4/4] Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_weekly_report(days: int = 5, dry_run: bool = False) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("Canyon v9  Step 108 — Weekly Report Generator")
    print("=" * 60)
    print(f"  Period: last {days} trading days  (report date: {TODAY})")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data sources …")
    picks   = _load_daily_picks()
    history = _load_alpha_history(days)
    journal = _load_journal()
    print(f"  picks={len(picks)} rows  history={len(history)} rows  journal={len(journal)} rows")

    tickers = picks["ticker"].dropna().tolist() if not picks.empty and "ticker" in picks.columns else []
    prices  = _load_price_returns(tickers, days)
    print(f"  prices: {prices.shape}  tickers loaded: {len(tickers)}")

    # ── Compute ───────────────────────────────────────────────────────────────
    print("\n[2/4] Computing metrics …")
    alpha_change = _compute_alpha_change(history, days)
    spy_ret      = _compute_week_return(prices, "SPY", days)
    port_ret     = _portfolio_return(prices, picks, days)
    print(f"  SPY return: {spy_ret:+.2f}%  Portfolio return: {port_ret:+.2f}%"
          if not (math.isnan(spy_ret) or math.isnan(port_ret))
          else "  Returns: N/A (insufficient price data)")

    # ── Build report sections ─────────────────────────────────────────────────
    print("\n[3/4] Building report sections …")
    header = (
        f"# Canyon v9 — Weekly Performance Report\n\n"
        f"**Report Date:** {TODAY}  |  **Period:** Last {days} Trading Days\n\n"
        f"---\n"
    )

    sections = [
        header,
        _build_performance_section(picks, prices, days, alpha_change),
        _build_buys_section(picks),
        _build_regime_section(history),
        _build_risk_section(),
        _build_macro_section(),
        _build_attribution_section(),
        _build_ic_section(),
        _build_earnings_section(),
        _build_journal_section(journal),
    ]
    report_md = "\n".join(sections)

    # ── Summary dict ──────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "report_date":       TODAY,
        "days":              days,
        "portfolio_return":  round(port_ret, 2) if not math.isnan(port_ret) else None,
        "spy_return":        round(spy_ret, 2)  if not math.isnan(spy_ret)  else None,
        "n_picks":           len(picks),
        "n_alpha_tracked":   len(alpha_change),
        "regime":            _load_regime().get("regime", "UNKNOWN"),
    }

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n[4/4] Writing outputs …")
    if not dry_run:
        md_path   = ROOT / "weekly_report.md"
        json_path = ROOT / "weekly_summary.json"

        md_path.write_text(report_md, encoding="utf-8")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"  [written] {md_path}  ({len(report_md)} chars)")
        print(f"  [written] {json_path}")
    else:
        print("  [DRY-RUN] outputs not written")
        print("\n" + "─" * 60)
        print(report_md[:2000] + ("\n… [truncated]" if len(report_md) > 2000 else ""))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 108 — Weekly Report Generator"
    )
    parser.add_argument("--days", type=int, default=5,
                        help="Number of trading days to cover (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report preview; do not write files")
    args = parser.parse_args()

    summary = run_weekly_report(days=args.days, dry_run=args.dry_run)
    pr = summary.get("portfolio_return")
    sr = summary.get("spy_return")
    if pr is not None and sr is not None:
        excess = pr - sr
        print(f"\n✓ Weekly report complete — Portfolio: {pr:+.2f}%  SPY: {sr:+.2f}%  "
              f"Excess: {excess:+.2f}%\n")
    else:
        print(f"\n✓ Weekly report complete — return data unavailable\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
