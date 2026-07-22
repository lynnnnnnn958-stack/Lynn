#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 13 — Paper Portfolio Ledger

Purpose:
Converts candidates from the dashboard / execution gate / sizing recommendation into a
“simulated trade ledger”. This step does not touch real money, does not connect to a broker, and does not place orders automatically.

It does four things:
1. Reads position_sizing_recommendations.csv / execution_gate_review.csv
2. Creates or updates paper_portfolio_ledger.csv
3. Generates paper_trade_update_template.csv for you to manually fill in simulated entries/exits
4. If paper_trade_updates.csv exists, applies ENTER / CLOSE / SKIP updates

Key rules:
- WATCHLIST is not an order
- PAPER_CANDIDATE is not an order either
- Only when you write ENTER in paper_trade_updates.csv does it become OPEN_PAPER
- Only when you write CLOSE and fill in exit_price does it become CLOSED_PAPER
- The Learning Engine should only learn from CLOSED_PAPER or CLOSED_REAL in the future
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
import numpy as np


ROOT = Path.cwd()

SIZING_FILE = ROOT / "position_sizing_recommendations.csv"
GATE_FILE = ROOT / "execution_gate_review.csv"
ORDER_FILE = ROOT / "pre_trade_order_ticket.csv"

LEDGER_FILE = ROOT / "paper_portfolio_ledger.csv"
UPDATE_TEMPLATE_FILE = ROOT / "paper_trade_update_template.csv"
UPDATE_FILE = ROOT / "paper_trade_updates.csv"
SUMMARY_FILE = ROOT / "paper_ledger_summary.md"


LEDGER_COLUMNS = [
    "trade_id",
    "created_at",
    "updated_at",
    "ticker",
    "side",
    "sleeve",
    "decision",
    "risk_bucket",
    "sector",
    "planned_weight",
    "approved_weight",
    "effective_weight",
    "suggested_weight",
    "suggested_action",
    "status",
    "entry_date",
    "entry_price",
    "entry_weight",
    "exit_date",
    "exit_price",
    "pnl_pct",
    "holding_days",
    "thesis",
    "risk_note",
    "manual_news_check",
    "earnings_date_check",
    "liquidity_check",
    "spread_check",
    "notes",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def make_trade_id(ticker: str) -> str:
    return f"P{datetime.now().strftime('%y%m%d')}-{ticker.upper()}-{str(uuid.uuid4())[:4].upper()}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    for col in LEDGER_COLUMNS:
        if path == LEDGER_FILE and col not in df.columns:
            df[col] = ""
    df.to_csv(path, index=False)


def clean_weight(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
        if abs(v) > 1.0:
            return v / 100.0
        return v
    except Exception:
        return 0.0


def clean_price(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        v = float(str(x).replace("$", "").replace(",", "").strip())
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def pct(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def load_ledger() -> pd.DataFrame:
    if LEDGER_FILE.exists():
        df = read_csv(LEDGER_FILE)
        for col in LEDGER_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_candidates() -> pd.DataFrame:
    sizing = read_csv(SIZING_FILE)
    gate = read_csv(GATE_FILE)

    if not sizing.empty:
        df = sizing.copy()
    elif not gate.empty:
        df = gate.copy()
    else:
        return pd.DataFrame()

    # Normalize column names if needed.
    cols_lower = {c.lower(): c for c in df.columns}

    def col(name, fallback=""):
        return cols_lower.get(name.lower(), fallback)

    required = {
        "ticker": col("ticker"),
        "sleeve": col("sleeve"),
        "decision": col("decision"),
        "risk_bucket": col("risk_bucket"),
        "sector": col("sector"),
        "planned_weight": col("planned_weight"),
        "approved_weight": col("approved_weight"),
        "effective_weight": col("effective_weight"),
        "suggested_weight": col("suggested_weight"),
        "suggested_action": col("suggested_action"),
        "sizing_reason": col("sizing_reason"),
    }

    out = pd.DataFrame()
    out["ticker"] = df[required["ticker"]].astype(str).str.upper().str.strip() if required["ticker"] else ""
    out["sleeve"] = df[required["sleeve"]].astype(str).str.upper().str.strip() if required["sleeve"] else "UNKNOWN"
    out["decision"] = df[required["decision"]].astype(str) if required["decision"] else "UNKNOWN"
    out["risk_bucket"] = df[required["risk_bucket"]].astype(str) if required["risk_bucket"] else "UNKNOWN"
    out["sector"] = df[required["sector"]].astype(str) if required["sector"] else "UNKNOWN"
    out["planned_weight"] = df[required["planned_weight"]].apply(clean_weight) if required["planned_weight"] else 0.0
    out["approved_weight"] = df[required["approved_weight"]].apply(clean_weight) if required["approved_weight"] else 0.0
    out["effective_weight"] = df[required["effective_weight"]].apply(clean_weight) if required["effective_weight"] else out["approved_weight"]
    out["suggested_weight"] = df[required["suggested_weight"]].apply(clean_weight) if required["suggested_weight"] else out["effective_weight"]
    out["suggested_action"] = df[required["suggested_action"]].astype(str) if required["suggested_action"] else "REVIEW_ONLY"
    out["risk_note"] = df[required["sizing_reason"]].astype(str) if required["sizing_reason"] else ""

    # Filter only meaningful paper candidates.
    out = out[out["ticker"].astype(str).str.len() > 0].copy()
    out = out[out["suggested_weight"].fillna(0) > 0].copy()

    # Skip very tiny suggestions unless explicitly core/hedge.
    keep = (out["suggested_weight"] >= 0.002) | (out["sleeve"].isin(["CORE_HEDGE", "SECTOR_ROTATION"]))
    out = out[keep].copy()

    return out


def add_new_candidates_to_ledger(ledger: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return ledger

    existing_active = set(
        ledger[ledger["status"].astype(str).isin(["WATCHLIST", "PAPER_CANDIDATE", "OPEN_PAPER"])]
        ["ticker"].astype(str).str.upper().tolist()
    )

    rows = []
    for _, c in candidates.iterrows():
        ticker = str(c["ticker"]).upper().strip()
        if ticker in existing_active:
            continue

        status = "PAPER_CANDIDATE"
        action = str(c.get("suggested_action", "")).upper()
        if "SKIP" in action:
            status = "WATCHLIST"
        elif "REVIEW_ONLY" in action or "REDUCE" in action:
            status = "WATCHLIST"

        rows.append({
            "trade_id": make_trade_id(ticker),
            "created_at": now_str(),
            "updated_at": now_str(),
            "ticker": ticker,
            "side": "LONG",
            "sleeve": c.get("sleeve", "UNKNOWN"),
            "decision": c.get("decision", "UNKNOWN"),
            "risk_bucket": c.get("risk_bucket", "UNKNOWN"),
            "sector": c.get("sector", "UNKNOWN"),
            "planned_weight": c.get("planned_weight", 0.0),
            "approved_weight": c.get("approved_weight", 0.0),
            "effective_weight": c.get("effective_weight", 0.0),
            "suggested_weight": c.get("suggested_weight", 0.0),
            "suggested_action": c.get("suggested_action", ""),
            "status": status,
            "entry_date": "",
            "entry_price": "",
            "entry_weight": "",
            "exit_date": "",
            "exit_price": "",
            "pnl_pct": "",
            "holding_days": "",
            "thesis": f"{c.get('sleeve','')} candidate from Step 10 sizing recommendation",
            "risk_note": c.get("risk_note", ""),
            "manual_news_check": "NO",
            "earnings_date_check": "NO",
            "liquidity_check": "NO",
            "spread_check": "NO",
            "notes": "",
        })

    if rows:
        add = pd.DataFrame(rows)
        ledger = pd.concat([ledger, add], ignore_index=True)

    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = ""
    return ledger[LEDGER_COLUMNS]


def generate_update_template(ledger: pd.DataFrame) -> pd.DataFrame:
    active = ledger[ledger["status"].astype(str).isin(["WATCHLIST", "PAPER_CANDIDATE", "OPEN_PAPER"])].copy()

    rows = []
    for _, r in active.iterrows():
        rows.append({
            "trade_id": r.get("trade_id", ""),
            "ticker": r.get("ticker", ""),
            "current_status": r.get("status", ""),
            "suggested_weight": r.get("suggested_weight", ""),
            "action": "",  # ENTER / CLOSE / SKIP / NOTE
            "entry_price": "",
            "entry_weight": r.get("suggested_weight", ""),
            "exit_price": "",
            "manual_news_check": "NO",
            "earnings_date_check": "NO",
            "liquidity_check": "NO",
            "spread_check": "NO",
            "notes": "",
        })

    template = pd.DataFrame(rows)
    template.to_csv(UPDATE_TEMPLATE_FILE, index=False)
    return template


def apply_updates(ledger: pd.DataFrame) -> pd.DataFrame:
    if not UPDATE_FILE.exists():
        return ledger

    updates = read_csv(UPDATE_FILE)
    if updates.empty:
        return ledger

    ledger = ledger.copy()
    ledger["trade_id"] = ledger["trade_id"].astype(str)

    for _, u in updates.iterrows():
        tid = str(u.get("trade_id", "")).strip()
        action = str(u.get("action", "")).upper().strip()
        if not tid or not action:
            continue

        idx = ledger.index[ledger["trade_id"] == tid].tolist()
        if not idx:
            continue
        i = idx[0]

        if action == "ENTER":
            entry_price = clean_price(u.get("entry_price", np.nan))
            entry_weight = clean_weight(u.get("entry_weight", ledger.at[i, "suggested_weight"]))

            checks = [
                str(u.get("manual_news_check", "NO")).upper() == "YES",
                str(u.get("earnings_date_check", "NO")).upper() == "YES",
                str(u.get("liquidity_check", "NO")).upper() == "YES",
                str(u.get("spread_check", "NO")).upper() == "YES",
            ]

            if not all(checks):
                ledger.at[i, "notes"] = (str(ledger.at[i, "notes"]) + " | ENTER rejected: manual checks incomplete").strip(" |")
                ledger.at[i, "updated_at"] = now_str()
                continue

            if not np.isfinite(entry_price):
                ledger.at[i, "notes"] = (str(ledger.at[i, "notes"]) + " | ENTER rejected: missing entry_price").strip(" |")
                ledger.at[i, "updated_at"] = now_str()
                continue

            ledger.at[i, "status"] = "OPEN_PAPER"
            ledger.at[i, "entry_date"] = today_str()
            ledger.at[i, "entry_price"] = entry_price
            ledger.at[i, "entry_weight"] = entry_weight
            ledger.at[i, "manual_news_check"] = "YES"
            ledger.at[i, "earnings_date_check"] = "YES"
            ledger.at[i, "liquidity_check"] = "YES"
            ledger.at[i, "spread_check"] = "YES"
            ledger.at[i, "updated_at"] = now_str()
            ledger.at[i, "notes"] = str(u.get("notes", ledger.at[i, "notes"]))

        elif action == "CLOSE":
            exit_price = clean_price(u.get("exit_price", np.nan))
            entry_price = clean_price(ledger.at[i, "entry_price"])

            if not np.isfinite(exit_price) or not np.isfinite(entry_price):
                ledger.at[i, "notes"] = (str(ledger.at[i, "notes"]) + " | CLOSE rejected: missing entry/exit price").strip(" |")
                ledger.at[i, "updated_at"] = now_str()
                continue

            side = str(ledger.at[i, "side"]).upper()
            direction = -1 if side in ["SHORT", "SELL"] else 1
            pnl = direction * (exit_price / entry_price - 1)

            entry_date = str(ledger.at[i, "entry_date"])
            try:
                d0 = datetime.strptime(entry_date[:10], "%Y-%m-%d")
                holding_days = max(1, (datetime.now() - d0).days)
            except Exception:
                holding_days = ""

            ledger.at[i, "status"] = "CLOSED_PAPER"
            ledger.at[i, "exit_date"] = today_str()
            ledger.at[i, "exit_price"] = exit_price
            ledger.at[i, "pnl_pct"] = pnl
            ledger.at[i, "holding_days"] = holding_days
            ledger.at[i, "updated_at"] = now_str()
            ledger.at[i, "notes"] = str(u.get("notes", ledger.at[i, "notes"]))

        elif action == "SKIP":
            ledger.at[i, "status"] = "SKIPPED"
            ledger.at[i, "updated_at"] = now_str()
            ledger.at[i, "notes"] = str(u.get("notes", "Skipped by manual review"))

        elif action == "NOTE":
            ledger.at[i, "updated_at"] = now_str()
            ledger.at[i, "notes"] = str(u.get("notes", ledger.at[i, "notes"]))

    return ledger


def build_summary(ledger: pd.DataFrame) -> str:
    lines = []
    lines.append("# Canyon v9 Step 13 — Paper Portfolio Ledger")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This file tracks simulated/paper trades only. It is not connected to a broker.")
    lines.append("The Learning Engine should only learn from CLOSED_PAPER or CLOSED_REAL trades.")
    lines.append("")

    if ledger.empty:
        lines.append("No ledger rows yet.")
        return "\n".join(lines)

    status_counts = ledger["status"].value_counts(dropna=False).reset_index()
    status_counts.columns = ["status", "count"]

    lines.append("## Status Summary")
    lines.append("")
    lines.append(status_counts.to_markdown(index=False))
    lines.append("")

    open_rows = ledger[ledger["status"].astype(str) == "OPEN_PAPER"].copy()
    closed_rows = ledger[ledger["status"].astype(str) == "CLOSED_PAPER"].copy()

    lines.append("## Open Paper Trades")
    lines.append("")
    if open_rows.empty:
        lines.append("_No open paper trades._")
    else:
        cols = ["trade_id", "ticker", "sleeve", "entry_date", "entry_price", "entry_weight", "risk_bucket", "notes"]
        show = open_rows[[c for c in cols if c in open_rows.columns]].copy()
        lines.append(show.to_markdown(index=False))
    lines.append("")

    lines.append("## Closed Paper Learning Sample")
    lines.append("")
    if closed_rows.empty:
        lines.append("_No closed paper trades yet. Learning Engine should not adjust weights._")
    else:
        pnl = pd.to_numeric(closed_rows["pnl_pct"], errors="coerce").dropna()
        win_rate = (pnl > 0).mean() if len(pnl) else 0
        lines.append(f"- Closed trades: **{len(closed_rows)}**")
        lines.append(f"- Average PnL: **{pnl.mean():.2%}**")
        lines.append(f"- Win rate: **{win_rate:.1%}**")
        by_sleeve = closed_rows.copy()
        by_sleeve["pnl_pct"] = pd.to_numeric(by_sleeve["pnl_pct"], errors="coerce")
        s = by_sleeve.groupby("sleeve")["pnl_pct"].agg(["count", "mean"]).reset_index()
        s["mean"] = s["mean"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        lines.append("")
        lines.append(s.to_markdown(index=False))
    lines.append("")

    lines.append("## How to update")
    lines.append("")
    lines.append("1. Open `paper_trade_update_template.csv`.")
    lines.append("2. Copy rows you want to update into `paper_trade_updates.csv`.")
    lines.append("3. Fill `action`: ENTER, CLOSE, SKIP, or NOTE.")
    lines.append("4. For ENTER, fill entry_price and set all four manual checks to YES.")
    lines.append("5. For CLOSE, fill exit_price.")
    lines.append("6. Rerun Step 13.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 13")
    print("Paper Portfolio Ledger")
    print("=" * 88)

    ledger = load_ledger()
    candidates = load_candidates()

    print(f"Existing ledger rows: {len(ledger)}")
    print(f"Candidate rows from sizing/gate: {len(candidates)}")

    ledger = add_new_candidates_to_ledger(ledger, candidates)
    ledger = apply_updates(ledger)

    for col in LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = ""
    ledger = ledger[LEDGER_COLUMNS]

    ledger.to_csv(LEDGER_FILE, index=False)
    template = generate_update_template(ledger)
    summary = build_summary(ledger)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")

    print("\nLedger status:")
    if not ledger.empty:
        counts = ledger["status"].value_counts(dropna=False)
        for status, count in counts.items():
            print(f"  {status}: {count}")
    else:
        print("  No rows.")

    print("\nFiles generated:")
    print(f"  {LEDGER_FILE}")
    print(f"  {UPDATE_TEMPLATE_FILE}")
    print(f"  {SUMMARY_FILE}")

    if not UPDATE_FILE.exists():
        print("\nNo paper_trade_updates.csv found yet.")
        print("Next: copy rows from paper_trade_update_template.csv into paper_trade_updates.csv when you want to simulate ENTER/CLOSE/SKIP.")
    else:
        print(f"\nApplied updates from: {UPDATE_FILE}")

    print("\nNext Step 14: connect CLOSED_PAPER trades back into Learning Engine for sleeve/thesis performance attribution.")


if __name__ == "__main__":
    main()
