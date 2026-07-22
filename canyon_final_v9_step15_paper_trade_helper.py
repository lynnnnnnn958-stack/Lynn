#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 15 — Paper Trade Helper

Purpose:
No need to manually edit CSV files. Manage the simulated trade ledger with simple commands.

Common commands:
    python3 -u canyon_final_v9_step15_paper_trade_helper.py list
    python3 -u canyon_final_v9_step15_paper_trade_helper.py enter GLD 300
    python3 -u canyon_final_v9_step15_paper_trade_helper.py close GLD 306
    python3 -u canyon_final_v9_step15_paper_trade_helper.py skip AMD
    python3 -u canyon_final_v9_step15_paper_trade_helper.py summary

Rules:
- Only modifies paper_portfolio_ledger.csv.
- No broker connection.
- No order submission.
- ENTER changes WATCHLIST/PAPER_CANDIDATE to OPEN_PAPER.
- CLOSE changes OPEN_PAPER to CLOSED_PAPER and calculates pnl_pct.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


ROOT = Path.cwd()
LEDGER_FILE = ROOT / "paper_portfolio_ledger.csv"
SUMMARY_FILE = ROOT / "paper_ledger_summary.md"

REQUIRED_COLS = [
    "trade_id", "created_at", "updated_at", "ticker", "side", "sleeve",
    "decision", "risk_bucket", "sector", "planned_weight", "approved_weight",
    "effective_weight", "suggested_weight", "suggested_action", "status",
    "entry_date", "entry_price", "entry_weight", "exit_date", "exit_price",
    "pnl_pct", "holding_days", "thesis", "risk_note", "manual_news_check",
    "earnings_date_check", "liquidity_check", "spread_check", "notes",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def load_ledger() -> pd.DataFrame:
    if not LEDGER_FILE.exists():
        raise SystemExit("paper_portfolio_ledger.csv not found. Please run Step 13 first.")

    df = pd.read_csv(LEDGER_FILE, dtype=str).fillna("")
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[REQUIRED_COLS].copy()


def save_ledger(df: pd.DataFrame) -> None:
    df.to_csv(LEDGER_FILE, index=False)


def safe_float(x, default=np.nan) -> float:
    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def find_trade(df: pd.DataFrame, ticker: str, statuses: list[str]) -> int:
    t = ticker.upper().strip()
    mask = (
        df["ticker"].astype(str).str.upper().eq(t)
        & df["status"].astype(str).str.upper().isin([s.upper() for s in statuses])
    )
    idx = df.index[mask].tolist()
    if not idx:
        raise SystemExit(f"No {statuses} trade found for {ticker}. Use 'list' to check.")
    return idx[0]


def build_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Canyon v9 Step 15 — Paper Trade Helper Summary")
    lines.append("")
    lines.append(f"Updated: {now()}")
    lines.append("")
    lines.append("## Status Summary")
    lines.append("")

    counts = df["status"].value_counts(dropna=False).reset_index()
    counts.columns = ["status", "count"]
    try:
        lines.append(counts.to_markdown(index=False))
    except Exception:
        lines.append(counts.to_string(index=False))
    lines.append("")

    open_df = df[df["status"].astype(str).str.upper().eq("OPEN_PAPER")].copy()
    closed_df = df[df["status"].astype(str).str.upper().eq("CLOSED_PAPER")].copy()

    lines.append("## Open Paper Trades")
    lines.append("")
    if open_df.empty:
        lines.append("_No open paper trades._")
    else:
        cols = ["trade_id", "ticker", "sleeve", "entry_date", "entry_price", "entry_weight", "risk_bucket", "notes"]
        show = open_df[cols].copy()
        try:
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("## Closed Paper Trades")
    lines.append("")
    if closed_df.empty:
        lines.append("_No closed paper trades._")
    else:
        closed_df["pnl_num"] = pd.to_numeric(closed_df["pnl_pct"], errors="coerce")
        cols = ["trade_id", "ticker", "sleeve", "entry_price", "exit_price", "pnl_pct", "holding_days", "notes"]
        show = closed_df[cols].copy()
        show["pnl_pct"] = pd.to_numeric(show["pnl_pct"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        try:
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append(show.to_string(index=False))
        lines.append("")
        pnl = closed_df["pnl_num"].dropna()
        if len(pnl):
            lines.append(f"- Closed count: **{len(pnl)}**")
            lines.append(f"- Average PnL: **{pnl.mean():.2%}**")
            lines.append(f"- Win rate: **{(pnl > 0).mean():.1%}**")
    lines.append("")

    lines.append("## Reminder")
    lines.append("")
    lines.append("- This is paper trading only.")
    lines.append("- No broker order was sent.")
    lines.append("- Learning should wait until at least 5 CLOSED_PAPER trades.")
    lines.append("")
    return "\n".join(lines)


def write_summary(df: pd.DataFrame) -> None:
    SUMMARY_FILE.write_text(build_summary(df), encoding="utf-8")


def cmd_list(args) -> None:
    df = load_ledger()
    cols = ["trade_id", "ticker", "status", "sleeve", "suggested_weight", "entry_price", "exit_price", "pnl_pct", "notes"]
    show = df[cols].copy()
    print(show.to_string(index=False))


def cmd_enter(args) -> None:
    df = load_ledger()
    i = find_trade(df, args.ticker, ["WATCHLIST", "PAPER_CANDIDATE"])

    price = safe_float(args.price)
    if not np.isfinite(price) or price <= 0:
        raise SystemExit("Entry price must be a positive number, e.g.: enter GLD 300")

    weight = args.weight
    if weight is None:
        weight = df.at[i, "suggested_weight"] or df.at[i, "effective_weight"] or "0.01"

    df.at[i, "status"] = "OPEN_PAPER"
    df.at[i, "entry_date"] = today()
    df.at[i, "entry_price"] = str(price)
    df.at[i, "entry_weight"] = str(weight)
    df.at[i, "manual_news_check"] = "YES"
    df.at[i, "earnings_date_check"] = "YES"
    df.at[i, "liquidity_check"] = "YES"
    df.at[i, "spread_check"] = "YES"
    df.at[i, "updated_at"] = now()
    note = args.note or "entered by Step15 helper"
    old = df.at[i, "notes"]
    df.at[i, "notes"] = (old + " | " + note).strip(" |")

    save_ledger(df)
    write_summary(df)

    print(f"OK: {args.ticker.upper()} -> OPEN_PAPER at {price}, weight={weight}")
    print(f"Open summary: {SUMMARY_FILE}")


def cmd_close(args) -> None:
    df = load_ledger()
    i = find_trade(df, args.ticker, ["OPEN_PAPER"])

    exit_price = safe_float(args.price)
    entry_price = safe_float(df.at[i, "entry_price"])

    if not np.isfinite(exit_price) or exit_price <= 0:
        raise SystemExit("Exit price must be a positive number, e.g.: close GLD 306")
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise SystemExit("This trade has no valid entry_price and cannot be closed.")

    side = str(df.at[i, "side"]).upper()
    direction = -1 if side in ["SHORT", "SELL"] else 1
    pnl = direction * (exit_price / entry_price - 1)

    try:
        d0 = datetime.strptime(str(df.at[i, "entry_date"])[:10], "%Y-%m-%d")
        holding_days = max(1, (datetime.now() - d0).days)
    except Exception:
        holding_days = 1

    df.at[i, "status"] = "CLOSED_PAPER"
    df.at[i, "exit_date"] = today()
    df.at[i, "exit_price"] = str(exit_price)
    df.at[i, "pnl_pct"] = str(pnl)
    df.at[i, "holding_days"] = str(holding_days)
    df.at[i, "updated_at"] = now()
    note = args.note or "closed by Step15 helper"
    old = df.at[i, "notes"]
    df.at[i, "notes"] = (old + " | " + note).strip(" |")

    save_ledger(df)
    write_summary(df)

    print(f"OK: {args.ticker.upper()} -> CLOSED_PAPER at {exit_price}; pnl={pnl:.2%}")
    print(f"Open summary: {SUMMARY_FILE}")


def cmd_skip(args) -> None:
    df = load_ledger()
    i = find_trade(df, args.ticker, ["WATCHLIST", "PAPER_CANDIDATE", "OPEN_PAPER"])

    df.at[i, "status"] = "SKIPPED"
    df.at[i, "updated_at"] = now()
    note = args.note or "skipped by Step15 helper"
    old = df.at[i, "notes"]
    df.at[i, "notes"] = (old + " | " + note).strip(" |")

    save_ledger(df)
    write_summary(df)
    print(f"OK: {args.ticker.upper()} -> SKIPPED")


def cmd_summary(args) -> None:
    df = load_ledger()
    write_summary(df)
    print(SUMMARY_FILE)


def main():
    parser = argparse.ArgumentParser(description="Canyon Step15 Paper Trade Helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="List ledger rows")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("enter", help="Enter a paper trade: enter GLD 300")
    p.add_argument("ticker")
    p.add_argument("price")
    p.add_argument("--weight", default=None)
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_enter)

    p = sub.add_parser("close", help="Close an open paper trade: close GLD 306")
    p.add_argument("ticker")
    p.add_argument("price")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_close)

    p = sub.add_parser("skip", help="Skip a candidate")
    p.add_argument("ticker")
    p.add_argument("--note", default=None)
    p.set_defaults(func=cmd_skip)

    p = sub.add_parser("summary", help="Regenerate summary")
    p.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
