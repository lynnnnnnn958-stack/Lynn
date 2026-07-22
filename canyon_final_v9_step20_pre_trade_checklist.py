#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 20 — Pre-trade Checklist

Purpose:
Turns “manual review” into a structured checklist and report, preventing
impulsive trading when a signal shows ALLOW.

Reads:
- position_sizing_recommendations.csv
- execution_gate_review.csv
- exposure_warnings.csv
- scenario_stress_results.csv
- paper_portfolio_ledger.csv

Outputs:
- pre_trade_checklist.csv
- pre_trade_checklist.md

Principles:
- No order submission
- No broker connection
- No market data download
- ALLOW is not a buy signal
- When Risk Light is RED, only PAPER_REVIEW is permitted, not LIVE
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path.cwd()

SIZING_FILE = ROOT / "position_sizing_recommendations.csv"
GATE_FILE = ROOT / "execution_gate_review.csv"
WARNINGS_FILE = ROOT / "exposure_warnings.csv"
STRESS_FILE = ROOT / "scenario_stress_results.csv"
LEDGER_FILE = ROOT / "paper_portfolio_ledger.csv"

OUT_CSV = ROOT / "pre_trade_checklist.csv"
OUT_MD = ROOT / "pre_trade_checklist.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=0.0):
    try:
        s = str(x).replace("%", "").strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def pct(x) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def classify_risk_light() -> tuple[str, str]:
    warnings = read_csv(WARNINGS_FILE)
    stress = read_csv(STRESS_FILE)

    high = med = 0
    if not warnings.empty and "level" in warnings.columns:
        levels = warnings["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())

    worst = None
    worst_name = "N/A"
    if not stress.empty and "estimated_pnl" in stress.columns:
        s = stress.copy()
        s["estimated_pnl_num"] = pd.to_numeric(s["estimated_pnl"], errors="coerce")
        s = s.dropna(subset=["estimated_pnl_num"]).sort_values("estimated_pnl_num")
        if not s.empty:
            worst = float(s.iloc[0]["estimated_pnl_num"])
            worst_name = str(s.iloc[0].get("scenario", "N/A"))

    if high > 0 or (worst is not None and worst <= -0.02):
        return "RED", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst_name} {pct(worst)}"
    if med >= 3 or (worst is not None and worst <= -0.01):
        return "AMBER", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst_name} {pct(worst)}"
    return "GREEN", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst_name} {pct(worst) if worst is not None else 'N/A'}"


def load_candidates() -> pd.DataFrame:
    sizing = read_csv(SIZING_FILE)
    gate = read_csv(GATE_FILE)

    if not sizing.empty:
        df = sizing.copy()
    elif not gate.empty:
        df = gate.copy()
    else:
        return pd.DataFrame()

    def col(candidates):
        lower = {c.lower(): c for c in df.columns}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    mapping = {
        "ticker": col(["ticker", "symbol"]),
        "sleeve": col(["sleeve"]),
        "decision": col(["decision"]),
        "risk_bucket": col(["risk_bucket"]),
        "sector": col(["sector"]),
        "effective_weight": col(["effective_weight"]),
        "suggested_weight": col(["suggested_weight"]),
        "suggested_action": col(["suggested_action"]),
        "sizing_reason": col(["sizing_reason"]),
    }

    out = pd.DataFrame()
    for k, c in mapping.items():
        if c:
            out[k] = df[c].astype(str)
        else:
            out[k] = ""

    out["ticker"] = out["ticker"].str.upper().str.strip()
    out = out[out["ticker"].str.len() > 0].copy()
    return out


def active_paper_status() -> dict[str, str]:
    ledger = read_csv(LEDGER_FILE)
    if ledger.empty or "ticker" not in ledger.columns or "status" not in ledger.columns:
        return {}

    # If multiple rows, prefer OPEN, then PAPER_CANDIDATE, then CLOSED/WATCHLIST.
    priority = {
        "OPEN_PAPER": 4,
        "OPEN_REAL": 4,
        "PAPER_CANDIDATE": 3,
        "WATCHLIST": 2,
        "CLOSED_PAPER": 1,
        "CLOSED_REAL": 1,
        "SKIPPED": 0,
    }
    status_map = {}
    best = {}

    for _, r in ledger.iterrows():
        t = str(r.get("ticker", "")).upper().strip()
        s = str(r.get("status", "")).upper().strip()
        if not t:
            continue
        p = priority.get(s, 0)
        if t not in best or p > best[t]:
            best[t] = p
            status_map[t] = s
    return status_map


def make_checklist() -> pd.DataFrame:
    risk_light, risk_detail = classify_risk_light()
    c = load_candidates()
    status_map = active_paper_status()

    rows = []

    if c.empty:
        return pd.DataFrame()

    for _, r in c.iterrows():
        ticker = str(r.get("ticker", "")).upper().strip()
        sleeve = str(r.get("sleeve", "")).upper().strip()
        decision = str(r.get("decision", "")).upper().strip()
        risk_bucket = str(r.get("risk_bucket", "")).upper().strip()
        suggested_action = str(r.get("suggested_action", "")).upper().strip()
        sizing_reason = str(r.get("sizing_reason", ""))

        sw = fnum(r.get("suggested_weight", ""), 0.0)
        ew = fnum(r.get("effective_weight", ""), 0.0)

        ledger_status = status_map.get(ticker, "")

        manual_news_check = "NO"
        earnings_date_check = "NO"
        liquidity_check = "NO"
        spread_check = "NO"
        duplicate_exposure_check = "REVIEW"
        stress_check = "REVIEW"
        live_allowed = "NO"
        paper_allowed = "NO"

        reasons = []

        # Base candidate status.
        if sw <= 0:
            base = "BLOCKED"
            reasons.append("suggested_weight is zero")
        elif "SKIP" in suggested_action:
            base = "BLOCKED"
            reasons.append("suggested_action contains SKIP")
        elif "REVIEW" in suggested_action or "REDUCE" in suggested_action:
            base = "REVIEW_REQUIRED"
            reasons.append(f"suggested_action={suggested_action}")
        elif "ALLOW" in suggested_action or "ALLOW" in decision:
            base = "PAPER_REVIEW"
            reasons.append("candidate can be reviewed for paper trading")
        else:
            base = "REVIEW_REQUIRED"
            reasons.append("unclear action; review required")

        # Risk-light rules.
        if risk_light == "RED":
            live_allowed = "NO"
            paper_allowed = "ONLY_SMALL_PAPER_AFTER_CHECKS"
            reasons.append("Risk Light RED; no live conversion")
        elif risk_light == "AMBER":
            live_allowed = "NO"
            paper_allowed = "SMALL_PAPER_AFTER_CHECKS"
            reasons.append("Risk Light AMBER; paper only")
        else:
            live_allowed = "NO"
            paper_allowed = "PAPER_AFTER_CHECKS"
            reasons.append("Risk Light GREEN still requires manual checks")

        # Overlap / concentration flags.
        reason_lower = sizing_reason.lower()
        if any(k in reason_lower for k in ["overlap", "semi exposure", "cluster", "concentration"]):
            duplicate_exposure_check = "FAIL_OR_REDUCE"
            reasons.append("overlap/concentration warning in sizing_reason")

        # Stress sensitive buckets.
        if risk_light == "RED" and risk_bucket in {"TECH_GROWTH", "SEMICONDUCTOR", "BROAD_BETA", "HIGH_BETA"}:
            stress_check = "FAIL_OR_TINY_SIZE"
            reasons.append(f"{risk_bucket} exposed under RED risk")
        elif risk_bucket in {"TECH_GROWTH", "SEMICONDUCTOR", "HIGH_BETA"}:
            stress_check = "REVIEW_REQUIRED"
            reasons.append(f"{risk_bucket} requires stress review")
        else:
            stress_check = "PASS_WITH_MANUAL_REVIEW"

        # Already closed/open logic.
        if ledger_status == "OPEN_PAPER":
            final_status = "ALREADY_OPEN_PAPER"
            reasons.append("already open in paper ledger")
        elif ledger_status in {"CLOSED_PAPER", "CLOSED_REAL"}:
            final_status = "ALREADY_CLOSED_DO_NOT_REPEAT"
            reasons.append("already closed; do not repeat automatically")
        elif base == "BLOCKED":
            final_status = "BLOCKED"
        else:
            final_status = "PENDING_MANUAL_CHECKS"

        rows.append({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "sleeve": sleeve,
            "decision": decision,
            "risk_bucket": risk_bucket,
            "effective_weight": ew,
            "suggested_weight": sw,
            "suggested_action": suggested_action,
            "ledger_status": ledger_status,
            "risk_light": risk_light,
            "risk_detail": risk_detail,
            "manual_news_check": manual_news_check,
            "earnings_date_check": earnings_date_check,
            "liquidity_check": liquidity_check,
            "spread_check": spread_check,
            "duplicate_exposure_check": duplicate_exposure_check,
            "stress_check": stress_check,
            "paper_allowed": paper_allowed,
            "live_allowed": live_allowed,
            "final_status": final_status,
            "reasons": "; ".join(reasons),
            "sizing_reason": sizing_reason,
        })

    out = pd.DataFrame(rows)

    # Sort most actionable first.
    order = {
        "PENDING_MANUAL_CHECKS": 0,
        "ALREADY_OPEN_PAPER": 1,
        "BLOCKED": 2,
        "ALREADY_CLOSED_DO_NOT_REPEAT": 3,
    }
    out["_sort"] = out["final_status"].map(order).fillna(9)
    out = out.sort_values(["_sort", "suggested_weight"], ascending=[True, False]).drop(columns=["_sort"])
    return out


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    d = df.head(max_rows).copy()
    for col in ["effective_weight", "suggested_weight"]:
        if col in d.columns:
            d[col] = d[col].map(lambda x: pct(x))
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(df: pd.DataFrame) -> str:
    risk_light, risk_detail = classify_risk_light()
    md = []
    md.append("# Canyon v9 Step 20 — Pre-trade Checklist")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Risk Gate")
    md.append("")
    md.append(f"- Risk Light: **{risk_light}**")
    md.append(f"- Detail: {risk_detail}")
    md.append("")
    if risk_light == "RED":
        md.append("**Rule:** No LIVE conversion. Paper trading only, and only after manual checks.")
    elif risk_light == "AMBER":
        md.append("**Rule:** Small paper trades only after manual checks.")
    else:
        md.append("**Rule:** Paper review allowed after manual checks. LIVE still blocked by default.")
    md.append("")

    if df.empty:
        md.append("No candidates found.")
        return "\n".join(md)

    md.append("## Status Summary")
    md.append("")
    summary = df["final_status"].value_counts().reset_index()
    summary.columns = ["final_status", "count"]
    md.append(summary.to_markdown(index=False))
    md.append("")

    md.append("## Actionable Rows")
    md.append("")
    actionable = df[df["final_status"] == "PENDING_MANUAL_CHECKS"].copy()
    cols = [
        "ticker", "sleeve", "risk_bucket", "suggested_weight", "suggested_action",
        "duplicate_exposure_check", "stress_check", "paper_allowed", "live_allowed", "reasons"
    ]
    md.append(md_table(actionable[cols], max_rows=20))
    md.append("")

    md.append("## Full Checklist")
    md.append("")
    full_cols = [
        "ticker", "sleeve", "ledger_status", "final_status", "risk_light",
        "suggested_weight", "manual_news_check", "earnings_date_check",
        "liquidity_check", "spread_check", "duplicate_exposure_check",
        "stress_check", "paper_allowed", "live_allowed"
    ]
    md.append(md_table(df[full_cols], max_rows=60))
    md.append("")

    md.append("## How to Use")
    md.append("")
    md.append("1. Do not trade anything marked BLOCKED or ALREADY_CLOSED_DO_NOT_REPEAT.")
    md.append("2. For PENDING_MANUAL_CHECKS, manually check news, earnings date, liquidity, and bid-ask spread.")
    md.append("3. If Risk Light is RED, do not convert to live. At most use tiny paper test.")
    md.append("4. Use Step 15 helper for paper trades; do not edit CSV manually.")
    md.append("")
    md.append("Example paper helper command:")
    md.append("")
    md.append("```bash")
    md.append("python3 -u canyon_final_v9_step15_paper_trade_helper.py enter TICKER PRICE")
    md.append("```")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 20")
    print("Pre-trade Checklist")
    print("=" * 88)

    df = make_checklist()
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_report(df), encoding="utf-8")

    print(f"Rows: {len(df)}")
    if not df.empty:
        counts = df["final_status"].value_counts()
        print("Status:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

        risk_light, risk_detail = classify_risk_light()
        print(f"Risk Light: {risk_light} | {risk_detail}")

    print("\nFiles generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")

    print("\nNext: open pre_trade_checklist.md")


if __name__ == "__main__":
    main()
