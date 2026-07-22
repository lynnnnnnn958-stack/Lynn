#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 14 — Paper Learning Attribution

Purpose:
Reads CLOSED_PAPER / CLOSED_REAL trades from paper_portfolio_ledger.csv,
tallies which sleeve / risk_bucket / ticker / thesis performed well, and generates a learning report.

Principles:
- Only learns from CLOSED_PAPER or CLOSED_REAL.
- WATCHLIST / PAPER_CANDIDATE / OPEN_PAPER are excluded from learning.
- With fewer than 5 closed trades, only records — does not auto-adjust weights.
- No order submission, no broker connection, no market data downloads.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


ROOT = Path.cwd()

LEDGER_FILE = ROOT / "paper_portfolio_ledger.csv"

OUT_MD = ROOT / "learning_attribution_report.md"
OUT_CSV = ROOT / "learning_attribution_summary.csv"
OUT_ADJUST = ROOT / "learning_weight_suggestions.csv"


MIN_TRADES_FOR_ADJUSTMENT = 5


def pct(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def read_ledger() -> pd.DataFrame:
    if not LEDGER_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LEDGER_FILE, dtype=str)
    except Exception:
        return pd.DataFrame()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()

    c = df[df["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"])].copy()

    if c.empty:
        return c

    for col in ["pnl_pct", "entry_weight", "suggested_weight", "holding_days"]:
        if col in c.columns:
            c[col] = clean_numeric(c[col])

    if "pnl_pct" not in c.columns:
        c["pnl_pct"] = np.nan

    c["win"] = c["pnl_pct"] > 0
    c["loss"] = c["pnl_pct"] < 0

    # weighted contribution if entry_weight available
    if "entry_weight" in c.columns:
        c["weighted_pnl_contribution"] = c["entry_weight"].fillna(0) * c["pnl_pct"].fillna(0)
    else:
        c["weighted_pnl_contribution"] = np.nan

    return c


def summarize_group(c: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if c.empty or group_col not in c.columns:
        return pd.DataFrame()

    g = c.groupby(group_col, dropna=False).agg(
        trades=("trade_id", "count") if "trade_id" in c.columns else ("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
        win_rate=("win", "mean"),
        total_weighted_contribution=("weighted_pnl_contribution", "sum"),
    ).reset_index()

    g = g.sort_values(["avg_pnl", "win_rate"], ascending=[False, False])
    return g


def make_adjustments(c: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative suggestion engine:
    - <5 closed trades: NO_ADJUST
    - enough sample: sleeve/risk_bucket with negative avg pnl and win_rate <40% -> DOWNWEIGHT_REVIEW
    - positive avg pnl and win_rate >55% -> WATCH_FOR_UPWEIGHT, not auto-upweight
    """
    rows = []

    if c.empty:
        return pd.DataFrame([{
            "level": "SYSTEM",
            "key": "ALL",
            "trades": 0,
            "avg_pnl": np.nan,
            "win_rate": np.nan,
            "suggestion": "NO_DATA",
            "reason": "No closed paper/real trades."
        }])

    total_n = len(c)

    for level in ["sleeve", "risk_bucket", "ticker"]:
        if level not in c.columns:
            continue
        sg = summarize_group(c, level)
        for _, r in sg.iterrows():
            n = int(r["trades"])
            avg = float(r["avg_pnl"]) if pd.notna(r["avg_pnl"]) else np.nan
            wr = float(r["win_rate"]) if pd.notna(r["win_rate"]) else np.nan

            suggestion = "NO_ADJUST"
            reason = "Sample too small; record only."

            if total_n >= MIN_TRADES_FOR_ADJUSTMENT and n >= 3:
                if avg < 0 and wr < 0.40:
                    suggestion = "DOWNWEIGHT_REVIEW"
                    reason = "Negative average PnL and low win rate."
                elif avg > 0 and wr > 0.55:
                    suggestion = "WATCH_FOR_UPWEIGHT"
                    reason = "Positive average PnL and acceptable win rate; do not auto-upweight without more evidence."
                else:
                    suggestion = "KEEP_NEUTRAL"
                    reason = "Mixed evidence."

            rows.append({
                "level": level,
                "key": r[level],
                "trades": n,
                "avg_pnl": avg,
                "win_rate": wr,
                "suggestion": suggestion,
                "reason": reason,
            })

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data._"
    d = df.copy().head(max_rows)
    for col in d.columns:
        if col in ["avg_pnl", "median_pnl", "win_rate", "total_weighted_contribution"]:
            d[col] = d[col].apply(pct)
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(df: pd.DataFrame, c: pd.DataFrame, adjustments: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 14 — Learning Attribution Report")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Rule")
    md.append("")
    md.append("- Only CLOSED_PAPER and CLOSED_REAL trades are used for learning.")
    md.append("- WATCHLIST, PAPER_CANDIDATE, and OPEN_PAPER are ignored.")
    md.append(f"- At least {MIN_TRADES_FOR_ADJUSTMENT} closed trades are required before giving any real adjustment signal.")
    md.append("")

    if df.empty:
        md.append("## Missing Ledger")
        md.append("")
        md.append("`paper_portfolio_ledger.csv` was not found or could not be read.")
        return "\n".join(md)

    md.append("## Ledger Status")
    md.append("")
    if "status" in df.columns:
        status_counts = df["status"].value_counts(dropna=False).reset_index()
        status_counts.columns = ["status", "count"]
        md.append(format_table(status_counts))
    else:
        md.append("_No status column found._")
    md.append("")

    md.append("## Closed Trade Sample")
    md.append("")
    if c.empty:
        md.append("_No closed trades yet._")
        return "\n".join(md)

    pnl = c["pnl_pct"].dropna()
    win_rate = c["win"].mean() if len(c) else np.nan
    avg_pnl = pnl.mean() if len(pnl) else np.nan
    med_pnl = pnl.median() if len(pnl) else np.nan
    total_contribution = c["weighted_pnl_contribution"].sum(skipna=True)

    md.append(f"- Closed trades: **{len(c)}**")
    md.append(f"- Average PnL: **{pct(avg_pnl)}**")
    md.append(f"- Median PnL: **{pct(med_pnl)}**")
    md.append(f"- Win rate: **{pct(win_rate)}**")
    md.append(f"- Total weighted contribution: **{pct(total_contribution)}**")
    md.append("")

    if len(c) < MIN_TRADES_FOR_ADJUSTMENT:
        md.append("**Conclusion:** sample is too small. Record only; do not change strategy weights yet.")
    else:
        md.append("**Conclusion:** enough initial closed trades to begin conservative adjustment review.")
    md.append("")

    cols = [x for x in ["trade_id", "ticker", "sleeve", "risk_bucket", "entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "notes"] if x in c.columns]
    show = c[cols].copy()
    if "pnl_pct" in show.columns:
        show["pnl_pct"] = show["pnl_pct"].apply(pct)

    md.append("## Closed Trades")
    md.append("")
    md.append(format_table(show, max_rows=50))
    md.append("")

    for level in ["sleeve", "risk_bucket", "ticker"]:
        sg = summarize_group(c, level)
        md.append(f"## Attribution by {level}")
        md.append("")
        md.append(format_table(sg, max_rows=30))
        md.append("")

    md.append("## Weight Suggestions")
    md.append("")
    md.append(format_table(adjustments, max_rows=60))
    md.append("")

    md.append("## Next")
    md.append("")
    md.append("- Continue paper trading until at least 5 CLOSED_PAPER trades exist.")
    md.append("- Then Step 14 can start giving cautious downweight/upweight review signals.")
    md.append("- Do not let one fake test trade change the live system.")
    md.append("")

    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 14")
    print("Paper Learning Attribution")
    print("=" * 88)

    df = read_ledger()
    c = closed_trades(df)
    adjustments = make_adjustments(c)

    if not c.empty:
        summaries = []
        for level in ["sleeve", "risk_bucket", "ticker"]:
            sg = summarize_group(c, level)
            if not sg.empty:
                sg.insert(0, "level", level)
                sg = sg.rename(columns={level: "key"})
                summaries.append(sg)
        if summaries:
            summary = pd.concat(summaries, ignore_index=True)
        else:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()

    summary.to_csv(OUT_CSV, index=False)
    adjustments.to_csv(OUT_ADJUST, index=False)

    report = build_report(df, c, adjustments)
    OUT_MD.write_text(report, encoding="utf-8")

    print(f"Total ledger rows: {0 if df.empty else len(df)}")
    print(f"Closed trades used for learning: {0 if c.empty else len(c)}")

    if not c.empty:
        avg = c["pnl_pct"].mean()
        wr = c["win"].mean()
        print(f"Average PnL: {avg:.2%}")
        print(f"Win rate: {wr:.1%}")

    if len(c) < MIN_TRADES_FOR_ADJUSTMENT:
        print("Conclusion: sample too small; record only, no automatic adjustment.")
    else:
        print("Conclusion: enough trades for conservative adjustment review.")

    print("\nFiles generated:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_ADJUST}")

    print("\nNext Step 15: make a simple close/open paper trade helper so you do not have to edit CSVs manually.")


if __name__ == "__main__":
    main()
