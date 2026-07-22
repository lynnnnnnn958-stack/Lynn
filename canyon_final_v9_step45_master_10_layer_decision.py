#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 45 — Master 10-Layer Decision Matrix

Combines available layer outputs into one ticker-level master decision table.
Missing layers are explicitly marked NO_DATA rather than ignored.

Outputs:
- master_10_layer_decision_matrix.csv
- master_10_layer_decision_report.md
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

FILES = {
    "pre": ROOT / "pre_trade_checklist.csv",
    "options": ROOT / "options_decision_matrix.csv",
    "cards": ROOT / "action_cards.csv",
    "sizing": ROOT / "position_sizing_recommendations.csv",
    "risk": ROOT / "exposure_warnings.csv",
    "stress": ROOT / "scenario_stress_results.csv",
    "learning": ROOT / "learning_weight_suggestions.csv",
    "ledger": ROOT / "paper_portfolio_ledger.csv",
    "layer_audit": ROOT / "canyon_layer_status_audit.csv",
}

OUT_CSV = ROOT / "master_10_layer_decision_matrix.csv"
OUT_MD = ROOT / "master_10_layer_decision_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan):
    try:
        s = str(x).replace("%", "").replace(",", "").strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def get_row(df: pd.DataFrame, ticker: str) -> dict:
    if df.empty or "ticker" not in df.columns:
        return {}
    m = df[df["ticker"].astype(str).str.upper().str.strip() == ticker]
    if m.empty:
        return {}
    return m.iloc[0].to_dict()


def all_tickers() -> list[str]:
    tickers = set()
    for key in ["pre", "options", "cards", "sizing", "ledger"]:
        df = read_csv(FILES[key])
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper().str.strip().tolist())
    return sorted([t for t in tickers if t and t not in {"CASH", "TACTICAL_CASH"}])


def portfolio_risk_state() -> tuple[str, str]:
    risk = read_csv(FILES["risk"])
    stress = read_csv(FILES["stress"])

    high = med = 0
    if not risk.empty and "level" in risk.columns:
        levels = risk["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())

    worst = np.nan
    if not stress.empty and "estimated_pnl" in stress.columns:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())

    if high > 0 or (np.isfinite(worst) and worst <= -0.02):
        return "RED", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst:.2%}" if np.isfinite(worst) else f"HIGH warnings={high}, MEDIUM warnings={med}"
    if med >= 3 or (np.isfinite(worst) and worst <= -0.01):
        return "AMBER", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst:.2%}" if np.isfinite(worst) else f"HIGH warnings={high}, MEDIUM warnings={med}"
    return "GREEN", f"HIGH warnings={high}, MEDIUM warnings={med}, worst={worst:.2%}" if np.isfinite(worst) else f"HIGH warnings={high}, MEDIUM warnings={med}"


def layer_gate_values(ticker: str, risk_state: str, risk_detail: str) -> dict:
    pre = read_csv(FILES["pre"])
    options = read_csv(FILES["options"])
    cards = read_csv(FILES["cards"])
    sizing = read_csv(FILES["sizing"])
    ledger = read_csv(FILES["ledger"])

    p = get_row(pre, ticker)
    o = get_row(options, ticker)
    c = get_row(cards, ticker)
    s = get_row(sizing, ticker)

    # L1 data integrity is still only basic.
    l1 = "PARTIAL"
    l1_note = "Basic local files exist, but no formal source/timestamp/stale-data table yet."

    # L2 macro not built.
    l2 = "NO_DATA"
    l2_note = "Macro/regime layer not formally built yet."

    # L3 sector rotation partial.
    sleeve = str(s.get("sleeve", p.get("sleeve", o.get("sleeve", ""))))
    l3 = "PARTIAL" if sleeve else "NO_DATA"
    l3_note = f"Sleeve/sector info: {sleeve or 'missing'}"

    # L4 fundamentals missing.
    l4 = "NO_DATA"
    l4_note = "Fundamental/valuation layer not built yet; do not use for long-term hold decision."

    # L5 event/news partial if pre has reasons, but no true news.
    pre_reasons = str(p.get("reasons", ""))
    l5 = "PARTIAL" if pre_reasons else "NO_DATA"
    l5_note = pre_reasons or "No event/news/earnings/insider evidence table yet."

    # L6 technical partial via sizing/decision.
    suggested_weight = fnum(s.get("suggested_weight", p.get("suggested_weight", np.nan)))
    l6 = "PARTIAL" if np.isfinite(suggested_weight) else "NO_DATA"
    l6_note = f"suggested_weight={suggested_weight:.2%}" if np.isfinite(suggested_weight) else "No technical signal table."

    # L7 options.
    opt_decision = str(o.get("final_options_decision", c.get("decision", ""))).upper()
    gamma_label = str(o.get("gamma_squeeze_label", ""))
    kill_label = str(o.get("option_kill_zone_label", c.get("kill_zone_label", "")))
    if opt_decision:
        l7 = opt_decision
        l7_note = f"{gamma_label}; {kill_label}"
    else:
        l7 = "NO_DATA"
        l7_note = "No options decision."

    # L8 risk.
    l8 = risk_state
    l8_note = risk_detail

    # L9 execution/pretrade.
    pre_status = str(p.get("final_status", "")).upper()
    if pre_status:
        l9 = pre_status
        l9_note = str(p.get("paper_allowed", "")) + " | " + str(p.get("live_allowed", "NO"))
    else:
        l9 = "NO_DATA"
        l9_note = "No pre-trade row."

    # L10 learning.
    if not ledger.empty and "ticker" in ledger.columns and "status" in ledger.columns:
        m = ledger[ledger["ticker"].astype(str).str.upper().str.strip() == ticker]
        if not m.empty:
            statuses = ", ".join(m["status"].astype(str).unique().tolist())
            l10 = "TRACE"
            l10_note = f"ledger status: {statuses}"
        else:
            l10 = "NO_SAMPLE"
            l10_note = "No paper sample for ticker."
    else:
        l10 = "NO_DATA"
        l10_note = "No ledger data."

    return {
        "L1_data": l1, "L1_note": l1_note,
        "L2_macro": l2, "L2_note": l2_note,
        "L3_sector": l3, "L3_note": l3_note,
        "L4_fundamental": l4, "L4_note": l4_note,
        "L5_event": l5, "L5_note": l5_note,
        "L6_price": l6, "L6_note": l6_note,
        "L7_options": l7, "L7_note": l7_note,
        "L8_risk": l8, "L8_note": l8_note,
        "L9_execution": l9, "L9_note": l9_note,
        "L10_learning": l10, "L10_note": l10_note,
    }


def master_action(row: dict) -> tuple[str, str]:
    l7 = str(row.get("L7_options", "")).upper()
    l8 = str(row.get("L8_risk", "")).upper()
    l9 = str(row.get("L9_execution", "")).upper()
    l4 = str(row.get("L4_fundamental", "")).upper()
    l2 = str(row.get("L2_macro", "")).upper()

    if "BLOCKED" in l9 or "SKIP" in l7:
        return "SKIP", "Blocked by execution/pre-trade or options decision."
    if "ALREADY_CLOSED" in l9:
        return "DO_NOT_REPEAT", "Already closed in paper; do not create fake repeated samples."
    if l8 == "RED" and l7 == "WAIT":
        return "WAIT_ONLY", "Portfolio risk is RED and options says WAIT. Observe trigger only."
    if l8 == "RED" and l7 == "PAPER_ONLY":
        return "TINY_PAPER_ONLY", "Risk RED: at most tiny stock/ETF paper, no options."
    if l7 == "WAIT":
        return "WAIT_TRIGGER", "Wait for trigger confirmation; no early short-dated options."
    if l7 == "PAPER_ONLY":
        return "PAPER_STOCK_ETF_ONLY", "Paper only via stock/ETF; avoid short-dated options."
    if l7 == "WATCH":
        return "WATCHLIST", "Watch only; wait for volume/price/event confirmation."
    if l4 == "NO_DATA":
        return "RESEARCH_REQUIRED", "Fundamental layer missing; not eligible for long-term hold."
    if l2 == "NO_DATA":
        return "MACRO_CONTEXT_MISSING", "Macro/regime layer missing; no sector-rotation conviction."
    return "REVIEW", "Manual review required."


def build_master() -> pd.DataFrame:
    risk_state, risk_detail = portfolio_risk_state()
    rows = []
    for t in all_tickers():
        layer_vals = layer_gate_values(t, risk_state, risk_detail)
        base = {"ticker": t}
        base.update(layer_vals)
        action, reason = master_action(base)
        base["master_action"] = action
        base["master_reason"] = reason
        rows.append(base)

    out = pd.DataFrame(rows)
    if not out.empty:
        order = {
            "TINY_PAPER_ONLY": 0,
            "PAPER_STOCK_ETF_ONLY": 1,
            "WAIT_ONLY": 2,
            "WAIT_TRIGGER": 3,
            "WATCHLIST": 4,
            "RESEARCH_REQUIRED": 5,
            "MACRO_CONTEXT_MISSING": 6,
            "DO_NOT_REPEAT": 7,
            "SKIP": 8,
            "REVIEW": 9,
        }
        out["_sort"] = out["master_action"].map(order).fillna(99)
        out = out.sort_values(["_sort", "ticker"]).drop(columns=["_sort"])
    return out


def md_table(df: pd.DataFrame, cols=None, max_rows=100) -> str:
    if df.empty:
        return "_No data._"
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    d = d.head(max_rows)
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 45 — Master 10-Layer Decision Report")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Important")
    md.append("")
    md.append("This report intentionally shows missing layers. A missing macro/fundamental layer should not be silently ignored.")
    md.append("")
    md.append("## Master actions")
    md.append("")
    if df.empty:
        md.append("_No rows._")
    else:
        s = df["master_action"].value_counts().reset_index()
        s.columns = ["master_action", "count"]
        md.append(s.to_markdown(index=False))
    md.append("")
    md.append("## Compact matrix")
    md.append("")
    compact_cols = [
        "ticker", "master_action", "master_reason",
        "L2_macro", "L4_fundamental", "L7_options", "L8_risk", "L9_execution", "L10_learning"
    ]
    md.append(md_table(df, compact_cols))
    md.append("")
    md.append("## Full 10-layer matrix")
    md.append("")
    md.append(md_table(df))
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- Options is only L7. It cannot override missing macro/fundamental/risk layers.")
    md.append("- RED risk means no live action.")
    md.append("- WAIT means wait for trigger; do not chase weekly OTM options.")
    md.append("- Missing L2/L4 means the system is not yet complete for sector rotation or long-term hold decisions.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 45")
    print("Master 10-Layer Decision Matrix")
    print("=" * 88)

    df = build_master()
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_report(df), encoding="utf-8")

    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker", "master_action", "L2_macro", "L4_fundamental", "L7_options", "L8_risk", "L9_execution"]].to_string(index=False))

    print()
    print("Files generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")
    print()
    print("Next: open master_10_layer_decision_report.md")


if __name__ == "__main__":
    main()
