#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 39 — Action Cards Builder

Reads:
- watch_triggers.csv
- options_decision_matrix.csv

Writes:
- action_cards.csv
- action_cards.md

Purpose:
Make the current wide dashboard tables readable as IF/THEN action cards.

No broker. No live order. Research/paper only.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

WATCH = ROOT / "watch_triggers.csv"
DECISION = ROOT / "options_decision_matrix.csv"

OUT_CSV = ROOT / "action_cards.csv"
OUT_MD = ROOT / "action_cards.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def fmt_price(x):
    v = fnum(x)
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.2f}"


def fmt_pct(x):
    v = fnum(x)
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.2%}"


def get_decision_row(decision: pd.DataFrame, ticker: str) -> dict:
    if decision.empty or "ticker" not in decision.columns:
        return {}
    m = decision[decision["ticker"].astype(str).str.upper().str.strip() == ticker]
    if m.empty:
        return {}
    return m.iloc[0].to_dict()


def build_cards() -> pd.DataFrame:
    watch = read_csv(WATCH)
    decision = read_csv(DECISION)

    if watch.empty:
        return pd.DataFrame()

    rows = []

    for _, r in watch.iterrows():
        ticker = str(r.get("ticker", "")).upper().strip()
        if not ticker:
            continue

        d = get_decision_row(decision, ticker)

        decision_label = str(r.get("decision", d.get("final_options_decision", ""))).upper()
        urgency = str(r.get("urgency", "")).upper()
        kill = str(r.get("kill_zone_label", d.get("option_kill_zone_label", ""))).upper()
        gamma = str(d.get("gamma_squeeze_label", ""))
        score = str(d.get("gamma_squeeze_score", ""))

        spot = fmt_price(r.get("spot", d.get("spot", "")))
        breakout = fmt_price(r.get("call_wall_breakout_trigger", d.get("call_wall_strike", "")))
        breakout_dist = fmt_pct(r.get("call_wall_distance", d.get("call_wall_distance", "")))
        breakdown = fmt_price(r.get("put_wall_breakdown_trigger", d.get("put_wall_strike", "")))
        breakdown_dist = fmt_pct(r.get("put_wall_distance", d.get("put_wall_distance", "")))

        live_allowed = str(r.get("live_allowed", d.get("live_allowed", "NO")) or "NO").upper()
        notes = str(r.get("notes", d.get("explanation", "")))

        if decision_label == "WAIT":
            one_liner = f"Wait for breakout confirmation near {breakout}; do not chase weekly OTM before trigger."
            allowed_action = "Observe / wait for confirmation"
            forbidden_action = "Buying short-dated options early; chasing a false breakout"
            trigger_rule = f"Enter paper review only when price approaches or clears {breakout} with no contrary volume/news risk."
        elif decision_label == "PAPER_ONLY":
            one_liner = "Equities/ETF paper only; do not express via short-dated options."
            allowed_action = "Equities/ETF paper"
            forbidden_action = "Weekly OTM options; chasing full-position rallies"
            trigger_rule = f"Small paper observation OK; re-check Gamma/Kill Zone when price nears {breakout}."
        elif decision_label == "WATCH":
            one_liner = "Add to watchlist; no urgent action."
            allowed_action = "Observe"
            forbidden_action = "Building a position without a trigger"
            trigger_rule = f"Wait for {breakout} or valid price/volume confirmation."
        elif "SKIP" in decision_label:
            one_liner = "Skip tonight; preserve focus."
            allowed_action = "None"
            forbidden_action = "Forcing a trade rationale"
            trigger_rule = "No trigger condition; skip."
        elif decision_label == "RESEARCH_ONLY":
            one_liner = "Research only; no paper position."
            allowed_action = "Research"
            forbidden_action = "Opening a position / paper"
            trigger_rule = "Gather fundamental/event evidence first."
        else:
            one_liner = "Manual review required."
            allowed_action = "Manual review"
            forbidden_action = "Automatic order submission"
            trigger_rule = "Check news, earnings, liquidity, and spread first."

        if "HIGH_OPTION_KILL_ZONE" in kill:
            risk_note = "High Kill Zone: high risk of pin / IV crush / theta bleed."
        elif "MEDIUM_OPTION_KILL_ZONE" in kill:
            risk_note = "Medium Kill Zone: can observe, but options expression requires extreme caution."
        else:
            risk_note = "Kill Zone shows no high risk, but still requires manual check of spread and events."

        rows.append({
            "ticker": ticker,
            "decision": decision_label,
            "urgency": urgency,
            "spot": spot,
            "one_liner": one_liner,
            "allowed_action": allowed_action,
            "forbidden_action": forbidden_action,
            "breakout_trigger": breakout,
            "breakout_distance": breakout_dist,
            "breakdown_trigger": breakdown,
            "breakdown_distance": breakdown_dist,
            "gamma_label": gamma,
            "gamma_score": score,
            "kill_zone_label": kill,
            "risk_note": risk_note,
            "trigger_rule": trigger_rule,
            "live_allowed": live_allowed,
            "notes": notes,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        decision_order = {
            "WAIT": 0,
            "PAPER_ONLY": 1,
            "WATCH": 2,
            "RESEARCH_ONLY": 3,
            "SKIP_OPTIONS": 4,
            "SKIP": 5,
        }
        out["_u"] = out["urgency"].map(urgency_order).fillna(9)
        out["_d"] = out["decision"].map(decision_order).fillna(9)
        out = out.sort_values(["_u", "_d", "ticker"]).drop(columns=["_u", "_d"])

    return out


def build_md(cards: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 39 — Action Cards")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("These are readable IF/THEN cards generated from Tonight Plan and Options Decision.")
    md.append("")
    md.append("## Summary")
    md.append("")
    if cards.empty:
        md.append("_No action cards generated._")
        return "\n".join(md)

    summary = cards["decision"].value_counts().reset_index()
    summary.columns = ["decision", "count"]
    md.append(summary.to_markdown(index=False))
    md.append("")

    md.append("## Cards")
    md.append("")

    for _, r in cards.iterrows():
        md.append(f"### {r['ticker']} — {r['decision']} / {r['urgency']}")
        md.append("")
        md.append(f"**Now:** {r['one_liner']}")
        md.append("")
        md.append(f"- Spot: `{r['spot']}`")
        md.append(f"- Breakout trigger: `{r['breakout_trigger']}` ({r['breakout_distance']})")
        md.append(f"- Breakdown trigger: `{r['breakdown_trigger']}` ({r['breakdown_distance']})")
        md.append(f"- Gamma: `{r['gamma_label']}` score `{r['gamma_score']}`")
        md.append(f"- Kill Zone: `{r['kill_zone_label']}`")
        md.append(f"- Allowed: {r['allowed_action']}")
        md.append(f"- Forbidden: {r['forbidden_action']}")
        md.append(f"- Rule: {r['trigger_rule']}")
        md.append(f"- Risk note: {r['risk_note']}")
        md.append(f"- Live allowed: `{r['live_allowed']}`")
        if str(r["notes"]).strip():
            md.append(f"- Notes: {r['notes']}")
        md.append("")

    md.append("## Non-negotiable rules")
    md.append("")
    md.append("- WAIT means wait. No early weekly OTM chase.")
    md.append("- PAPER_ONLY means stock/ETF paper only.")
    md.append("- SKIP means no attention tonight.")
    md.append("- live_allowed should stay NO until a separate broker-level checklist exists.")
    md.append("")

    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 39")
    print("Action Cards Builder")
    print("=" * 88)

    cards = build_cards()
    cards.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_md(cards), encoding="utf-8")

    print(f"Cards: {len(cards)}")
    if not cards.empty:
        print(cards[["ticker", "decision", "urgency", "one_liner", "breakout_trigger", "kill_zone_label"]].to_string(index=False))

    print()
    print("Files generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")
    print()
    print("Next: open action_cards.md")


if __name__ == "__main__":
    main()
