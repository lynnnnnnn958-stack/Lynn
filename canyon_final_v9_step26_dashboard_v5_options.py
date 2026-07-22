#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 26 — PM Cockpit v5 with Options/Gamma/Kill-Zone
Local research dashboard only. No broker, no live order.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "daily_pm_report": ROOT / "daily_pm_report.md",
    "system_health_check": ROOT / "system_health_check.md",
    "pre_trade_checklist": ROOT / "pre_trade_checklist.csv",
    "pre_trade_md": ROOT / "pre_trade_checklist.md",
    "exposure_warnings": ROOT / "exposure_warnings.csv",
    "scenario_stress": ROOT / "scenario_stress_results.csv",
    "position_sizing": ROOT / "position_sizing_recommendations.csv",
    "paper_ledger": ROOT / "paper_portfolio_ledger.csv",
    "paper_md": ROOT / "paper_ledger_summary.md",
    "learning_md": ROOT / "learning_attribution_report.md",
    "learning_csv": ROOT / "learning_attribution_summary.csv",
    "options_fetch_md": ROOT / "yfinance_options_fetch_report.md",
    "options_chain": ROOT / "options_chain_snapshot.csv",
    "gamma_md": ROOT / "options_gamma_report.md",
    "gamma_csv": ROOT / "gamma_squeeze_candidates.csv",
    "kill_md": ROOT / "option_kill_zone_report.md",
    "kill_csv": ROOT / "option_kill_zone_risk.csv",
}

def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()

def read_md(path):
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Failed to read `{path.name}`: {e}_"

def num(x):
    try:
        s = str(x).replace("%", "").replace(",", "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"

def fmt(df):
    if df.empty:
        return df
    d = df.copy()
    for c in d.columns:
        lc = c.lower()
        if any(k in lc for k in ["weight", "pnl", "loss", "distance", "ratio", "spread", "gex"]):
            def f(v):
                vv = num(v)
                if vv is None:
                    return v
                if "gex" in lc:
                    return f"{vv:,.0f}"
                if any(k in lc for k in ["weight", "pnl", "loss", "distance", "ratio", "spread"]):
                    return f"{vv:.2%}"
                return v
            d[c] = d[c].apply(f)
    return d

def show(df, height=360):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(fmt(df), hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(fmt(df), hide_index=True, use_container_width=True, height=height)

def metrics():
    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_stress"])
    pre = read_csv(FILES["pre_trade_checklist"])
    ledger = read_csv(FILES["paper_ledger"])
    gamma = read_csv(FILES["gamma_csv"])
    kill = read_csv(FILES["kill_csv"])

    high = med = 0
    if "level" in warnings:
        levels = warnings["level"].str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())

    worst = None
    if "estimated_pnl" in stress:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())

    pending = blocked = 0
    if "final_status" in pre:
        fs = pre["final_status"].str.upper()
        pending = int((fs == "PENDING_MANUAL_CHECKS").sum())
        blocked = int((fs == "BLOCKED").sum())

    closed = 0
    if "status" in ledger:
        closed = int(ledger["status"].str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"]).sum())

    gamma_high = 0
    if "gamma_squeeze_label" in gamma:
        gamma_high = int(gamma["gamma_squeeze_label"].str.contains("HIGH", case=False, na=False).sum())

    kill_high = 0
    if "option_kill_zone_label" in kill:
        kill_high = int(kill["option_kill_zone_label"].str.contains("HIGH", case=False, na=False).sum())

    risk = "RED" if high > 0 or (worst is not None and worst <= -0.02) else ("AMBER" if med >= 3 else "GREEN")
    return risk, high, med, worst, pending, blocked, closed, gamma_high, kill_high

def overview():
    risk, high, med, worst, pending, blocked, closed, gamma_high, kill_high = metrics()
    st.markdown(f"### Portfolio Risk Light: **{risk}**")
    st.write("Research cockpit only. ALLOW / PAPER / Gamma Watch are NOT order commands.")

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Warnings", f"H{high}/M{med}")
    c2.metric("Worst", pct(worst) if worst is not None else "N/A")
    c3.metric("Pending", pending)
    c4.metric("Blocked", blocked)
    c5.metric("Closed Paper", closed)
    c6.metric("Gamma High", gamma_high)
    c7.metric("Kill-Zone High", kill_high)

    st.subheader("Action queue")
    show(pd.DataFrame([
        ["1", "Risk", "Start with the risk light and high-risk sources."],
        ["2", "Pre-trade", "PENDING means awaiting manual review, not a buy signal."],
        ["3", "Options/Gamma", "Check call wall / put wall / gamma squeeze score."],
        ["4", "Kill Zone", "Guard against weekly OTM pin / IV crush / theta bleed."],
        ["5", "Paper", "Paper trade only to accumulate track record."],
    ], columns=["step", "page", "task"]), 210)

def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v5", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 PM Cockpit v5")
    st.caption("No broker. No live order. Options gamma is a proxy, not dealer truth.")

    tabs = st.tabs(["Overview","Risk","Pre-trade","Options/Gamma","Option Kill Zone","Paper","Learning","Health","Reports","Helper"])

    with tabs[0]:
        overview()
    with tabs[1]:
        st.header("Risk")
        st.subheader("Exposure Warnings")
        show(read_csv(FILES["exposure_warnings"]), 300)
        st.subheader("Scenario Stress")
        show(read_csv(FILES["scenario_stress"]), 300)
        st.subheader("Position Sizing")
        show(read_csv(FILES["position_sizing"]), 420)
    with tabs[2]:
        st.header("Pre-trade")
        st.markdown(read_md(FILES["pre_trade_md"]))
        show(read_csv(FILES["pre_trade_checklist"]), 520)
    with tabs[3]:
        st.header("Options / Gamma")
        st.markdown(read_md(FILES["gamma_md"]))
        st.subheader("Gamma Candidates")
        show(read_csv(FILES["gamma_csv"]), 420)
        st.subheader("Options Fetch Report")
        st.markdown(read_md(FILES["options_fetch_md"]))
        st.subheader("Raw Chain")
        chain = read_csv(FILES["options_chain"])
        cols = [c for c in ["ticker","expiration_date","option_type","strike","spot","gamma","delta","implied_volatility","open_interest","volume","bid","ask","dte"] if c in chain.columns]
        show(chain[cols] if cols else chain, 420)
    with tabs[4]:
        st.header("Option Kill Zone")
        st.markdown(read_md(FILES["kill_md"]))
        show(read_csv(FILES["kill_csv"]), 520)
    with tabs[5]:
        st.header("Paper Ledger")
        st.markdown(read_md(FILES["paper_md"]))
        show(read_csv(FILES["paper_ledger"]), 520)
    with tabs[6]:
        st.header("Learning")
        st.markdown(read_md(FILES["learning_md"]))
        show(read_csv(FILES["learning_csv"]), 360)
    with tabs[7]:
        st.header("Health")
        st.markdown(read_md(FILES["system_health_check"]))
    with tabs[8]:
        st.header("Reports")
        st.markdown(read_md(FILES["daily_pm_report"]))
    with tabs[9]:
        st.header("Commands")
        st.code("""cd ~/Desktop/canyon_quant
source .venv/bin/activate

python3 -u canyon_final_v9_step25_options_daily_runner.py
streamlit run canyon_final_v9_step26_dashboard_v5_options.py

python3 -u canyon_final_v9_step27_full_daily_pipeline.py
""", language="bash")

if __name__ == "__main__":
    main()
