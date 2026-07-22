#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 21 — Dashboard v4 with Pre-trade Checklist + Health Check

This is still local-only:
- No broker connection
- No live order
- No auto-trade
- Reads local CSV/MD files only
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "daily_pm_report": ROOT / "daily_pm_report.md",
    "exposure_dashboard": ROOT / "exposure_dashboard.md",
    "stress_report": ROOT / "stress_position_sizing_report.md",
    "execution_gate": ROOT / "execution_gate_review.csv",
    "pre_trade_order": ROOT / "pre_trade_order_ticket.csv",
    "paper_ledger": ROOT / "paper_portfolio_ledger.csv",
    "paper_summary": ROOT / "paper_ledger_summary.md",
    "learning_report": ROOT / "learning_attribution_report.md",
    "learning_summary": ROOT / "learning_attribution_summary.csv",
    "learning_suggestions": ROOT / "learning_weight_suggestions.csv",
    "exposure_table": ROOT / "exposure_dashboard.csv",
    "exposure_warnings": ROOT / "exposure_warnings.csv",
    "scenario_results": ROOT / "scenario_stress_results.csv",
    "sizing_recommendations": ROOT / "position_sizing_recommendations.csv",
    "health_report": ROOT / "system_health_check.md",
    "health_table": ROOT / "system_health_check.csv",
    "pre_trade_checklist": ROOT / "pre_trade_checklist.csv",
    "pre_trade_report": ROOT / "pre_trade_checklist.md",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def read_md(path: Path) -> str:
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Failed to read `{path.name}`: {e}_"


def fnum(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).replace("%", "").replace(",", "").strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for col in d.columns:
        cl = col.lower()
        if any(k in cl for k in ["pnl", "loss", "weight", "reduction", "exposure", "shock"]):
            d[col] = d[col].apply(
                lambda x: pct(float(x))
                if str(x).strip() not in ["", "nan", "None"] and fnum(x) is not None
                else x
            )
    return d


def show_df(df: pd.DataFrame, height: int | None = None):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(fmt_table(df), hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(fmt_table(df), hide_index=True, use_container_width=True, height=height)


def get_metrics():
    gate = read_csv(FILES["execution_gate"])
    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_results"])
    ledger = read_csv(FILES["paper_ledger"])
    checklist = read_csv(FILES["pre_trade_checklist"])

    gate_n = len(gate)

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

    closed_n = 0
    open_n = 0
    avg_pnl = None
    win_rate = None
    if not ledger.empty and "status" in ledger.columns:
        status = ledger["status"].astype(str).str.upper()
        closed = ledger[status.isin(["CLOSED_PAPER", "CLOSED_REAL"])].copy()
        open_df = ledger[status.isin(["OPEN_PAPER", "OPEN_REAL"])].copy()
        closed_n = len(closed)
        open_n = len(open_df)
        if not closed.empty and "pnl_pct" in closed.columns:
            pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
            if len(pnl):
                avg_pnl = float(pnl.mean())
                win_rate = float((pnl > 0).mean())

    pending_checks = 0
    blocked = 0
    if not checklist.empty and "final_status" in checklist.columns:
        fs = checklist["final_status"].astype(str).str.upper()
        pending_checks = int((fs == "PENDING_MANUAL_CHECKS").sum())
        blocked = int((fs == "BLOCKED").sum())

    return {
        "gate_n": gate_n,
        "high": high,
        "med": med,
        "worst": worst,
        "worst_name": worst_name,
        "closed_n": closed_n,
        "open_n": open_n,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "pending_checks": pending_checks,
        "blocked": blocked,
    }


def risk_color(m):
    if m["high"] >= 1 or (m["worst"] is not None and m["worst"] <= -0.02):
        return "RED", "High risk: reduce concentration first, do not rush candidates into trades."
    if m["med"] >= 3 or (m["worst"] is not None and m["worst"] <= -0.01):
        return "AMBER", "Moderate risk: research OK but keep paper trade positions small."
    return "GREEN", "Relatively low risk: manual review still required."


def action_queue(m):
    color, msg = risk_color(m)
    rows = [
        ("1", "Risk", f"{color}: {msg}"),
        ("2", "Pre-trade Checklist", f"{m['pending_checks']} rows still need manual checks; {m['blocked']} blocked."),
        ("3", "Stress + Sizing", "Use suggested_weight, not raw model weight."),
        ("4", "Paper Ledger", f"Closed paper = {m['closed_n']}; need at least 5 before learning can adjust."),
        ("5", "Learning", "Record only until sample size improves."),
    ]
    return pd.DataFrame(rows, columns=["step", "section", "what_to_do"])


def display_metrics():
    m = get_metrics()
    color, msg = risk_color(m)

    st.markdown(
        f"""
        <div class="risk-card risk-{color.lower()}">
            <div class="risk-title">Portfolio Risk Light: {color}</div>
            <div>{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Gate", m["gate_n"])
    c2.metric("Warnings", f"H{m['high']}/M{m['med']}")
    c3.metric("Worst", pct(m["worst"]) if m["worst"] is not None else "N/A", m["worst_name"])
    c4.metric("Pending Checks", m["pending_checks"])
    c5.metric("Open Paper", m["open_n"])
    c6.metric("Closed Paper", m["closed_n"])
    c7.metric("Paper Avg PnL", pct(m["avg_pnl"]) if m["avg_pnl"] is not None else "N/A")

    st.subheader("Today's action queue")
    show_df(action_queue(m), height=210)


def tab_overview():
    display_metrics()

    st.subheader("Worst-case stress scenario ranking")
    stress = read_csv(FILES["scenario_results"])
    if not stress.empty and "estimated_pnl" in stress.columns:
        s = stress.copy()
        s["estimated_pnl_num"] = pd.to_numeric(s["estimated_pnl"], errors="coerce")
        s = s.sort_values("estimated_pnl_num").drop(columns=["estimated_pnl_num"])
        show_df(s, height=260)
    else:
        st.info("No scenario results yet.")

    st.subheader("Pre-trade status")
    checklist = read_csv(FILES["pre_trade_checklist"])
    if not checklist.empty:
        cols = [c for c in ["ticker", "sleeve", "final_status", "risk_light", "suggested_weight", "paper_allowed", "live_allowed", "reasons"] if c in checklist.columns]
        show_df(checklist[cols], height=320)
    else:
        st.info("No pre-trade checklist yet. Run Step 20.")


def tab_risk():
    st.header("Risk & Exposure")
    st.markdown("This page answers: whether the portfolio looks diversified on the surface but is concentrated on a single theme.")
    st.subheader("Exposure Warnings")
    show_df(read_csv(FILES["exposure_warnings"]), height=280)
    st.subheader("Exposure by Ticker / Bucket")
    show_df(read_csv(FILES["exposure_table"]), height=420)
    with st.expander("Full Markdown Report"):
        st.markdown(read_md(FILES["exposure_dashboard"]))


def tab_stress():
    st.header("Stress + Position Sizing")
    st.markdown("This page answers: potential loss in a bad scenario and recommended position reduction.")
    st.subheader("Scenario Stress Results")
    show_df(read_csv(FILES["scenario_results"]), height=260)
    st.subheader("Position Sizing Recommendations")
    sizing = read_csv(FILES["sizing_recommendations"])
    cols = [c for c in ["ticker", "sleeve", "decision", "risk_bucket", "effective_weight", "suggested_weight", "suggested_action", "sizing_reason"] if c in sizing.columns]
    show_df(sizing[cols] if cols else sizing, height=420)
    with st.expander("Full Markdown Report"):
        st.markdown(read_md(FILES["stress_report"]))


def tab_pre_trade():
    st.header("Pre-trade Checklist")
    st.markdown("This page is the pre-trade review. When Risk Light is RED, LIVE is not allowed — small paper review only.")
    st.markdown(read_md(FILES["pre_trade_report"]))
    st.subheader("Checklist Table")
    df = read_csv(FILES["pre_trade_checklist"])
    if not df.empty:
        cols = [c for c in [
            "ticker", "sleeve", "ledger_status", "final_status", "risk_light",
            "suggested_weight", "duplicate_exposure_check", "stress_check",
            "paper_allowed", "live_allowed", "reasons"
        ] if c in df.columns]
        show_df(df[cols], height=520)
    else:
        st.info("No pre-trade checklist found. Run Step 20.")


def tab_execution():
    st.header("Execution Gate")
    st.markdown("This is candidate review only, not an order page.")
    show_df(read_csv(FILES["execution_gate"]), height=420)
    st.subheader("Pre-trade Order Ticket")
    order = read_csv(FILES["pre_trade_order"])
    if order.empty:
        st.success("No order draft generated. This is expected unless manual checks are complete.")
    else:
        show_df(order, height=220)


def tab_paper():
    st.header("Paper Ledger")
    st.markdown("This shows simulated trades. Paper trade is not real-money trading.")
    st.markdown(read_md(FILES["paper_summary"]))
    st.subheader("Paper Portfolio Ledger")
    show_df(read_csv(FILES["paper_ledger"]), height=420)


def tab_learning():
    st.header("Learning Attribution")
    st.markdown("Learns from CLOSED_PAPER / CLOSED_REAL only. With fewer than 5 trades, records only — no weight adjustment.")
    st.markdown(read_md(FILES["learning_report"]))
    st.subheader("Learning Attribution Summary")
    show_df(read_csv(FILES["learning_summary"]), height=260)
    st.subheader("Learning Weight Suggestions")
    show_df(read_csv(FILES["learning_suggestions"]), height=300)


def tab_health():
    st.header("System Health")
    st.markdown(read_md(FILES["health_report"]))
    st.subheader("Health Table")
    show_df(read_csv(FILES["health_table"]), height=520)


def tab_reports():
    st.header("Reports")
    with st.expander("Daily PM Report", expanded=True):
        st.markdown(read_md(FILES["daily_pm_report"]))


def tab_helper():
    st.header("Commands")
    st.markdown("Use these commands going forward; avoid manually editing CSV files.")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Daily re-run
python3 -u canyon_final_v9_step12_daily_runner.py

# System health check
python3 -u canyon_final_v9_step19_health_check.py

# Pre-trade check
python3 -u canyon_final_v9_step20_pre_trade_checklist.py

# Open new dashboard
streamlit run canyon_final_v9_step21_dashboard_v4.py

# View paper ledger
python3 -u canyon_final_v9_step15_paper_trade_helper.py list

# Paper buy
python3 -u canyon_final_v9_step15_paper_trade_helper.py enter TICKER PRICE

# Paper sell
python3 -u canyon_final_v9_step15_paper_trade_helper.py close TICKER PRICE

# Re-run learning attribution
python3 -u canyon_final_v9_step14_learning_attribution.py
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v4", page_icon="🏔", layout="wide")

    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; max-width: 1450px; }
        .risk-card { padding: 1rem 1.2rem; border-radius: 16px; margin-bottom: 1rem; border: 1px solid #ddd; }
        .risk-title { font-size: 1.25rem; font-weight: 700; margin-bottom: .35rem; }
        .risk-red { background: #fff1f1; border-color: #ffb3b3; }
        .risk-amber { background: #fff8e6; border-color: #ffd580; }
        .risk-green { background: #effaf2; border-color: #9ad7a8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏔 Canyon v9 PM Cockpit v4")
    st.caption("Research cockpit only. No broker connection. No live order. Paper trading is simulated.")

    tabs = st.tabs([
        "Overview",
        "Risk",
        "Stress + Sizing",
        "Pre-trade",
        "Execution Gate",
        "Paper Ledger",
        "Learning",
        "Health",
        "Reports",
        "Helper",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_risk()
    with tabs[2]:
        tab_stress()
    with tabs[3]:
        tab_pre_trade()
    with tabs[4]:
        tab_execution()
    with tabs[5]:
        tab_paper()
    with tabs[6]:
        tab_learning()
    with tabs[7]:
        tab_health()
    with tabs[8]:
        tab_reports()
    with tabs[9]:
        tab_helper()


if __name__ == "__main__":
    main()
