#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 8 — Daily PM Report
Aggregates regime / sleeve / evidence / SEC-event / journal / execution gate into a one-page PM report.

Principles:
1. No order submission.
2. No synthetic price data.
3. Only aggregates real data files that have already been generated and manual review status.
4. execution_gate_review.csv is a review sheet, not an order file.
5. pre_trade_order_ticket.csv defaults to DRAFT_NOT_SENT even if it exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit("pandas required. Run: pip install pandas") from exc

ROOT = Path.cwd()

RUN_FILES = [
    "run_v9_step3_fixed.txt",
    "run_v9_step4.txt",
    "run_v9_step5.txt",
    "run_v9_step6.txt",
    "run_v9_step7.txt",
]

CSV_FILES = {
    "journal": "canyon_trade_journal.csv",
    "update_template": "trade_update_template.csv",
    "execution_gate": "execution_gate_review.csv",
    "order_ticket": "pre_trade_order_ticket.csv",
}

OUTPUT_MD = ROOT / "daily_pm_report.md"
OUTPUT_TXT = ROOT / "daily_pm_report.txt"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def first_match(patterns: List[str], text: str) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return m.group(1).strip()
    return "not found"


def grep_lines(text: str, keywords: List[str], max_lines: int = 12) -> List[str]:
    lines = []
    for line in text.splitlines():
        if any(k in line for k in keywords):
            line = line.strip()
            if line:
                lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def read_csv(name: str) -> Optional[pd.DataFrame]:
    path = ROOT / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fmt_pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return str(x)


def summarize_gate(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "exists": False,
            "n": 0,
            "status_counts": {},
            "decision_counts": {},
            "rows_md": "No execution_gate_review.csv found.",
            "all_manual_ready": False,
        }

    status_col = "gate_status" if "gate_status" in df.columns else None
    decision_col = "decision" if "decision" in df.columns else None
    status_counts = df[status_col].value_counts().to_dict() if status_col else {}
    decision_counts = df[decision_col].value_counts().to_dict() if decision_col else {}

    check_cols = [
        c for c in df.columns
        if c in {
            "manual_news_check", "earnings_date_check", "liquidity_check",
            "spread_check", "thesis_confirmed", "risk_confirmed"
        }
    ]
    all_manual_ready = False
    if check_cols:
        all_manual_ready = bool((df[check_cols].astype(str).apply(lambda s: s.str.upper()) == "YES").all(axis=None))

    display_cols = [c for c in [
        "ticker", "sleeve", "planned_weight", "approved_weight",
        "approved_max_weight", "decision", "gate_status", "order_intent"
    ] if c in df.columns]
    if not display_cols:
        display_cols = df.columns[:8].tolist()

    rows_md = df[display_cols].head(12).to_markdown(index=False)
    return {
        "exists": True,
        "n": len(df),
        "status_counts": status_counts,
        "decision_counts": decision_counts,
        "rows_md": rows_md,
        "all_manual_ready": all_manual_ready,
    }


def summarize_orders(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty:
        return {"exists": False, "n": 0, "rows_md": "No order drafts. This is normal: no draft is generated before manual checks are complete."}
    display_cols = [c for c in ["ticker", "side", "order_type", "weight", "status", "order_intent"] if c in df.columns]
    if not display_cols:
        display_cols = df.columns[:8].tolist()
    return {"exists": True, "n": len(df), "rows_md": df[display_cols].head(12).to_markdown(index=False)}


def summarize_journal(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty:
        return {"exists": False, "n": 0, "closed": 0, "watchlist": 0, "rows_md": "No canyon_trade_journal.csv found or journal is empty."}
    status_col = "status" if "status" in df.columns else None
    closed = int((df[status_col].astype(str).str.upper() == "CLOSED").sum()) if status_col else 0
    watchlist = int((df[status_col].astype(str).str.upper() == "WATCHLIST").sum()) if status_col else 0
    display_cols = [c for c in ["ticker", "sleeve", "thesis_type", "status", "planned_weight", "decision"] if c in df.columns]
    if not display_cols:
        display_cols = df.columns[:8].tolist()
    return {"exists": True, "n": len(df), "closed": closed, "watchlist": watchlist, "rows_md": df[display_cols].head(12).to_markdown(index=False)}


def build_report() -> str:
    texts = {name: read_text(ROOT / name) for name in RUN_FILES}
    all_text = "\n".join(texts.values())

    data_rule = first_match([
        r"synthetic fallback:\s*([^\n]+)",
        r"all price data from real downloads[;；]([^\n]+)",
    ], all_text)

    data_loaded = first_match([
        r"\[Data\]\s*(✅[^\n]+)",
        r"Yahoo Finance\s*([^\n]+)",
    ], all_text)

    market_state = first_match([
        r"current market state[:：]\s*([^|\n]+)",
        r"market state[:：]\s*([^\n]+)",
        r"current state[:：]\s*([^\n]+)",
    ], all_text)

    sleeve_lines = grep_lines(all_text, ["TACTICAL", "CORE_HEDGE", "SECTOR_ROTATION"], max_lines=12)
    evidence_lines = grep_lines(all_text, ["decision=", "evidence:", "risk:", "check:"], max_lines=18)
    sec_lines = grep_lines(all_text, ["SEC=OK", "latest report", "Form4", "8-K", "revenue_yoy", "net_income"], max_lines=16)

    journal_df = read_csv(CSV_FILES["journal"])
    gate_df = read_csv(CSV_FILES["execution_gate"])
    order_df = read_csv(CSV_FILES["order_ticket"])

    journal = summarize_journal(journal_df)
    gate = summarize_gate(gate_df)
    orders = summarize_orders(order_df)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    risk_notes = []
    if gate["n"] == 0:
        risk_notes.append("No order draft generated after manual checks — default is no trade.")
    if gate.get("status_counts"):
        if any("PENDING" in str(k).upper() for k in gate["status_counts"].keys()):
            risk_notes.append("PENDING_MANUAL_CHECKS exist: news, earnings date, liquidity, spread, thesis not all confirmed.")
    if orders["n"] == 0:
        risk_notes.append("pre_trade_order_ticket.csv is empty or missing — no orders pending.")
    if journal["closed"] < 5:
        risk_notes.append("Fewer than 5 CLOSED real trades — Learning Engine should not auto-adjust weights.")
    if not risk_notes:
        risk_notes.append("No hard blocks detected, but manual review is still required.")

    md = f"""# Canyon v9 Daily PM Report

Generated: {now}

## 1. Data Authenticity

- Data status: {data_loaded}
- Synthetic data rule: {data_rule}
- Conclusion: the pipeline must use only real downloaded data; stop on download failure — do not fall back to synthetic data for decisions.

## 2. Current Market State and Three-Account Allocation

- Current market state: {market_state}

```text
{chr(10).join(sleeve_lines) if sleeve_lines else 'No sleeve lines found in report.'}
```

Interpretation: TACTICAL is a short-term candidate account, not an auto-trading account; CORE_HEDGE is the anchor; SECTOR_ROTATION is reviewed weekly, no intraday switching.

## 3. Evidence / SEC / Event Summary

### Price and Risk Evidence

```text
{chr(10).join(evidence_lines) if evidence_lines else 'No Evidence Card summary found.'}
```

### SEC / Event Evidence

```text
{chr(10).join(sec_lines) if sec_lines else 'No SEC/Event summary found.'}
```

Note: Form 4 count is not a buy signal. Must open the original Form 4 to determine buy, sell, option grant, tax sale, or 10b5-1 plan trade.

## 4. Trade Journal Status

- Journal candidate count: {journal['n']}
- WATCHLIST count: {journal['watchlist']}
- CLOSED real trade count: {journal['closed']}

{journal['rows_md']}

## 5. Execution Gate Status

- Review sheet candidate count: {gate['n']}
- gate_status distribution: {gate['status_counts']}
- decision distribution: {gate['decision_counts']}
- All manual checks complete: {gate['all_manual_ready']}

{gate['rows_md']}

## 6. Order Ticket Status

- Order draft count: {orders['n']}

{orders['rows_md']}

## 7. Daily Trading Discipline

{chr(10).join(f'- {x}' for x in risk_notes)}
- Evidence Card ALLOW only means eligible for review, not permission to place real orders.
- REVIEW_OR_REDUCE defaults to halving size or skipping.
- TACTICAL single-stock default cap 2.5%; correlated themes cannot all be at max.
- LIMIT_ONLY is the default order type; do not use market orders to chase price.
- All manual check items must be YES and order_intent must be PAPER or LIVE before generating an order draft.

## 8. Next Steps

Step 9 should run Portfolio Exposure Dashboard: aggregate semiconductor, tech, market-wide, single-stock, ETF overlapping exposure in table form to avoid hidden concentration within apparently diversified positions.
"""
    return md


def main():
    report = build_report()
    OUTPUT_MD.write_text(report, encoding="utf-8")
    OUTPUT_TXT.write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("CANYON v9 Step8 Daily PM Report")
    print("=" * 78)
    print(f"Generated: {OUTPUT_MD}")
    print(f"Generated: {OUTPUT_TXT}")
    print("\nNote: this is a PM review report, not an order command.")
    print("Open with: open daily_pm_report.md")
    print("\nPreview:")
    print("-" * 78)
    for line in report.splitlines()[:45]:
        print(line)


if __name__ == "__main__":
    main()
