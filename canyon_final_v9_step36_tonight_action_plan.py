#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 36 — Tonight Action Plan

Purpose:
Turn the current dashboard outputs into a short actionable PM plan:
- what to watch tonight
- what to paper only
- what to skip
- breakout trigger levels from call wall
- option kill-zone warnings
- no live order, no broker connection

Inputs:
- options_decision_matrix.csv
- gamma_squeeze_candidates.csv
- option_kill_zone_risk.csv
- exposure_warnings.csv
- scenario_stress_results.csv
- pre_trade_checklist.csv

Outputs:
- tonight_action_plan.md
- watch_triggers.csv
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

FILES = {
    "decision": ROOT / "options_decision_matrix.csv",
    "gamma": ROOT / "gamma_squeeze_candidates.csv",
    "kill": ROOT / "option_kill_zone_risk.csv",
    "warnings": ROOT / "exposure_warnings.csv",
    "stress": ROOT / "scenario_stress_results.csv",
    "pre": ROOT / "pre_trade_checklist.csv",
}

OUT_MD = ROOT / "tonight_action_plan.md"
OUT_CSV = ROOT / "watch_triggers.csv"


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
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def money(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"


def get_row(df: pd.DataFrame, ticker: str) -> dict:
    if df.empty or "ticker" not in df.columns:
        return {}
    m = df[df["ticker"].astype(str).str.upper().str.strip() == ticker]
    if m.empty:
        return {}
    return m.iloc[0].to_dict()


def build_triggers(decision: pd.DataFrame, gamma: pd.DataFrame, kill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if decision.empty or "ticker" not in decision.columns:
        return pd.DataFrame()

    for _, r in decision.iterrows():
        ticker = str(r.get("ticker", "")).upper().strip()
        if not ticker:
            continue

        g = get_row(gamma, ticker)
        k = get_row(kill, ticker)

        spot = fnum(r.get("spot", g.get("spot", np.nan)))
        call_wall = fnum(r.get("call_wall_strike", g.get("call_wall_strike", np.nan)))
        put_wall = fnum(r.get("put_wall_strike", g.get("put_wall_strike", np.nan)))
        max_pain = fnum(r.get("max_pain_proxy", k.get("max_pain_proxy", np.nan)))

        decision_label = str(r.get("final_options_decision", "")).upper()
        gamma_label = str(r.get("gamma_squeeze_label", ""))
        kill_label = str(r.get("option_kill_zone_label", ""))

        if np.isfinite(call_wall) and np.isfinite(spot):
            breakout_trigger = call_wall
            breakout_distance = call_wall / spot - 1
        else:
            breakout_trigger = np.nan
            breakout_distance = np.nan

        if np.isfinite(put_wall) and np.isfinite(spot):
            breakdown_trigger = put_wall
            breakdown_distance = put_wall / spot - 1
        else:
            breakdown_trigger = np.nan
            breakdown_distance = np.nan

        if decision_label == "PAPER_ONLY":
            action = "Equities/ETF paper only; no short-dated options"
            urgency = "MEDIUM"
        elif decision_label == "WAIT":
            action = "Wait for breakout/breakdown confirmation; no chasing weekly OTM"
            urgency = "HIGH" if "HIGH" in gamma_label else "MEDIUM"
        elif decision_label == "WATCH":
            action = "Add to watchlist; wait for price/volume confirmation"
            urgency = "MEDIUM"
        elif "SKIP" in decision_label:
            action = "Skip; pass tonight"
            urgency = "LOW"
        elif decision_label == "RESEARCH_ONLY":
            action = "Research only, no paper position"
            urgency = "LOW"
        else:
            action = "Manual review required"
            urgency = "MEDIUM"

        notes = []
        if "HIGH" in kill_label:
            notes.append("High kill-zone: pin/IV crush/theta bleed risk.")
        elif "MEDIUM" in kill_label:
            notes.append("Medium kill-zone: reduce option aggression.")
        if "HIGH" in gamma_label:
            notes.append("High gamma watch.")
        if np.isfinite(max_pain) and np.isfinite(spot):
            notes.append(f"Max pain proxy {max_pain:.2f}, distance {max_pain / spot - 1:.2%}.")

        rows.append({
            "ticker": ticker,
            "spot": spot,
            "decision": decision_label,
            "urgency": urgency,
            "action": action,
            "gamma_label": gamma_label,
            "kill_zone_label": kill_label,
            "call_wall_breakout_trigger": breakout_trigger,
            "call_wall_distance": breakout_distance,
            "put_wall_breakdown_trigger": breakdown_trigger,
            "put_wall_distance": breakdown_distance,
            "max_pain_proxy": max_pain,
            "live_allowed": r.get("live_allowed", "NO"),
            "notes": " | ".join(notes),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        decision_order = {"PAPER_ONLY": 0, "WAIT": 1, "WATCH": 2, "RESEARCH_ONLY": 3, "SKIP_OPTIONS": 4, "SKIP": 5}
        out["_urgency_sort"] = out["urgency"].map(order).fillna(9)
        out["_decision_sort"] = out["decision"].map(decision_order).fillna(9)
        out = out.sort_values(["_urgency_sort", "_decision_sort", "ticker"]).drop(columns=["_urgency_sort", "_decision_sort"])
    return out


def summarise_risk(warnings: pd.DataFrame, stress: pd.DataFrame) -> tuple[str, list[str]]:
    notes = []
    risk_light = "GREEN"

    if not warnings.empty and "level" in warnings.columns:
        levels = warnings["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())
        notes.append(f"Exposure warnings: HIGH={high}, MEDIUM={med}.")
        if high > 0:
            risk_light = "RED"
        elif med >= 3:
            risk_light = "AMBER"

    if not stress.empty and "estimated_pnl" in stress.columns:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())
            notes.append(f"Worst scenario estimated PnL: {worst:.2%}.")
            if worst <= -0.02:
                risk_light = "RED"
            elif worst <= -0.01 and risk_light != "RED":
                risk_light = "AMBER"

    return risk_light, notes


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 50) -> str:
    if df.empty:
        return "_No data._"
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    d = d.head(max_rows)

    for c in d.columns:
        if any(k in c for k in ["distance"]):
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        if any(k in c for k in ["spot", "trigger", "max_pain"]):
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_report(triggers: pd.DataFrame, decision: pd.DataFrame, risk_light: str, risk_notes: list[str]) -> str:
    md = []
    md.append("# Canyon v9 Step 36 — Tonight Action Plan")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Portfolio Risk")
    md.append("")
    md.append(f"Risk light: **{risk_light}**")
    md.append("")
    for n in risk_notes:
        md.append(f"- {n}")
    md.append("")
    md.append("## Tonight Priority")
    md.append("")
    if triggers.empty:
        md.append("_No watch triggers generated._")
    else:
        priority_cols = [
            "ticker", "decision", "urgency", "action",
            "call_wall_breakout_trigger", "call_wall_distance",
            "put_wall_breakdown_trigger", "put_wall_distance",
            "kill_zone_label", "notes",
        ]
        md.append(md_table(triggers, priority_cols, max_rows=80))
    md.append("")
    md.append("## Decision Breakdown")
    md.append("")
    if decision.empty or "final_options_decision" not in decision.columns:
        md.append("_No decision matrix._")
    else:
        s = decision["final_options_decision"].value_counts().reset_index()
        s.columns = ["decision", "count"]
        md.append(md_table(s))
    md.append("")
    md.append("## Rules For Tonight")
    md.append("")
    md.append("1. `WAIT` means wait. Do not buy short-dated options before breakout confirmation.")
    md.append("2. `PAPER_ONLY` means stock/ETF paper only, not weekly OTM calls/puts.")
    md.append("3. `HIGH_OPTION_KILL_ZONE` means pin / IV crush / theta bleed risk is too high for aggressive option buying.")
    md.append("4. `live_allowed` should remain NO unless you have a separate broker-level checklist.")
    md.append("5. If QQQ/SMH are high gamma + high kill-zone, they are observation triggers, not instant entries.")
    md.append("")
    md.append("## Next Manual Checks")
    md.append("")
    md.append("- Check earnings/event calendar.")
    md.append("- Check bid-ask spread and option liquidity manually.")
    md.append("- Check whether price is actually approaching call wall with volume.")
    md.append("- If no trigger, do nothing.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 36")
    print("Tonight Action Plan")
    print("=" * 88)

    decision = read_csv(FILES["decision"])
    gamma = read_csv(FILES["gamma"])
    kill = read_csv(FILES["kill"])
    warnings = read_csv(FILES["warnings"])
    stress = read_csv(FILES["stress"])

    triggers = build_triggers(decision, gamma, kill)
    triggers.to_csv(OUT_CSV, index=False)

    risk_light, risk_notes = summarise_risk(warnings, stress)
    OUT_MD.write_text(build_report(triggers, decision, risk_light, risk_notes), encoding="utf-8")

    print(f"Rows: {len(triggers)}")
    if not triggers.empty:
        print(triggers[["ticker", "decision", "urgency", "action", "call_wall_breakout_trigger", "kill_zone_label"]].to_string(index=False))

    print()
    print("Files generated:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_CSV}")
    print()
    print("Next: open tonight_action_plan.md")


if __name__ == "__main__":
    main()
