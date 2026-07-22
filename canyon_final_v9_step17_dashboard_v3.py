#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 17 — Dashboard v3 Polished

Goal:
Make the website feel more like a real PM Cockpit:
1. Plain-English explanations + proper financial terminology
2. Risk traffic light: RED / AMBER / GREEN
3. Today's action queue: what to look at first, what to do next
4. Candidates, stress tests, paper trades, and learning attribution all in one view
5. No order submission, no broker connection, no market data downloads; reads local files only
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
            d[col] = d[col].apply(lambda x: pct(float(x)) if str(x).strip() not in ["", "nan", "None"] and fnum(x) is not None else x)
    return d


def show_df(df: pd.DataFrame, height: int | None = None):
    if df.empty:
        st.info("No data yet.")
    else:
        try:
            st.dataframe(fmt_table(df), hide_index=True, width="stretch", height=height)
        except TypeError:
            st.dataframe(fmt_table(df), hide_index=True, use_container_width=True, height=height)


def file_status_table():
    rows = []
    for k, p in FILES.items():
        rows.append({
            "file": p.name,
            "status": "FOUND" if p.exists() else "MISSING",
        })
    return pd.DataFrame(rows)


def get_metrics():
    gate = read_csv(FILES["execution_gate"])
    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_results"])
    ledger = read_csv(FILES["paper_ledger"])

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
    }


def risk_color(m):
    if m["high"] >= 1 or (m["worst"] is not None and m["worst"] <= -0.02):
        return "RED", "Elevated risk: reduce concentration first, do not rush candidates into trades."
    if m["med"] >= 3 or (m["worst"] is not None and m["worst"] <= -0.01):
        return "AMBER", "Moderate risk: research is fine, but keep paper trade sizes small."
    return "GREEN", "Risk is relatively contained: manual review still required."


def action_queue(m):
    actions = []

    color, msg = risk_color(m)
    actions.append(("1", "Check Risk First", f"{color}: {msg}"))

    if m["gate_n"] > 0:
        actions.append(("2", "Check Execution Gate", "Candidates are pending review only — they are not orders."))

    if m["worst"] is not None and m["worst"] <= -0.02:
        actions.append(("3", "Check Stress + Sizing", "Worst stress scenario exceeds -2%; prioritize reducing tech/semiconductor/duplicate ETF exposure."))
    else:
        actions.append(("3", "Check Stress + Sizing", "Confirm suggested_weight rather than relying on raw weight."))

    if m["closed_n"] < 5:
        actions.append(("4", "Continue Paper Trade", f"Current closed paper = {m['closed_n']}, fewer than 5 — Learning can only record, not adjust weights."))
    else:
        actions.append(("4", "Check Learning", "Closed paper sample is sufficient; start reviewing whether sleeve/thesis is degrading."))

    return pd.DataFrame(actions, columns=["step", "section", "what_to_do"])


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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Gate Candidates", m["gate_n"])
    c2.metric("Warnings", f"H{m['high']}/M{m['med']}")
    c3.metric("Worst Scenario", pct(m["worst"]) if m["worst"] is not None else "N/A", m["worst_name"])
    c4.metric("Open Paper", m["open_n"])
    c5.metric("Closed Paper", m["closed_n"])
    c6.metric("Paper Avg PnL", pct(m["avg_pnl"]) if m["avg_pnl"] is not None else "N/A", f"Win {pct(m['win_rate'])}" if m["win_rate"] is not None else "")

    st.subheader("Tonight's Priority Actions")
    show_df(action_queue(m), height=180)


def tab_overview():
    display_metrics()

    st.subheader("Worst Stress Scenarios Ranked")
    stress = read_csv(FILES["scenario_results"])
    if not stress.empty and "estimated_pnl" in stress.columns:
        s = stress.copy()
        s["estimated_pnl_num"] = pd.to_numeric(s["estimated_pnl"], errors="coerce")
        s = s.sort_values("estimated_pnl_num").drop(columns=["estimated_pnl_num"])
        show_df(s, height=260)
    else:
        st.info("No scenario results yet.")

    st.subheader("Current Paper Trade Loop")
    ledger = read_csv(FILES["paper_ledger"])
    if not ledger.empty and "status" in ledger.columns:
        status = ledger["status"].astype(str).str.upper()
        show = ledger[status.isin(["OPEN_PAPER", "CLOSED_PAPER", "CLOSED_REAL"])].copy()
        cols = [c for c in ["trade_id", "ticker", "status", "sleeve", "entry_price", "exit_price", "pnl_pct", "notes"] if c in show.columns]
        show_df(show[cols], height=220)
    else:
        st.info("No paper ledger yet.")


def tab_risk():
    st.header("Risk & Exposure")
    st.markdown("This page answers: Is the portfolio superficially diversified but actually concentrated in the same theme?")
    st.subheader("Exposure Warnings")
    show_df(read_csv(FILES["exposure_warnings"]), height=280)
    st.subheader("Exposure by Ticker / Bucket")
    show_df(read_csv(FILES["exposure_table"]), height=420)
    with st.expander("Full Markdown Report"):
        st.markdown(read_md(FILES["exposure_dashboard"]))


def tab_stress():
    st.header("Stress + Position Sizing")
    st.markdown("This page answers: How much could we lose in a bad scenario, and how much should positions be reduced?")
    st.subheader("Scenario Stress Results")
    show_df(read_csv(FILES["scenario_results"]), height=260)
    st.subheader("Position Sizing Recommendations")
    sizing = read_csv(FILES["sizing_recommendations"])
    cols = [c for c in ["ticker", "sleeve", "decision", "risk_bucket", "effective_weight", "suggested_weight", "suggested_action", "sizing_reason"] if c in sizing.columns]
    show_df(sizing[cols] if cols else sizing, height=420)
    with st.expander("Full Markdown Report"):
        st.markdown(read_md(FILES["stress_report"]))


def tab_execution():
    st.header("Execution Gate")
    st.markdown("This page is not an order entry screen. It is for reviewing candidates and confirming which ones are still awaiting manual checks.")
    show_df(read_csv(FILES["execution_gate"]), height=420)
    st.subheader("Pre-trade Order Ticket")
    order = read_csv(FILES["pre_trade_order"])
    if order.empty:
        st.success("No order draft generated. This is expected unless manual checks are complete.")
    else:
        show_df(order, height=220)


def tab_paper():
    st.header("Paper Ledger")
    st.markdown("This page shows simulated trades. Paper trades are not real-money transactions.")
    st.markdown(read_md(FILES["paper_summary"]))
    st.subheader("Paper Portfolio Ledger")
    show_df(read_csv(FILES["paper_ledger"]), height=420)


def tab_learning():
    st.header("Learning Attribution")
    st.markdown("Only learns from CLOSED_PAPER / CLOSED_REAL. With fewer than 5 trades, only records — does not adjust weights.")
    st.markdown(read_md(FILES["learning_report"]))
    st.subheader("Learning Attribution Summary")
    show_df(read_csv(FILES["learning_summary"]), height=260)
    st.subheader("Learning Weight Suggestions")
    show_df(read_csv(FILES["learning_suggestions"]), height=300)


def tab_reports():
    st.header("Reports")
    with st.expander("Daily PM Report", expanded=True):
        st.markdown(read_md(FILES["daily_pm_report"]))
    with st.expander("File Status"):
        show_df(file_status_table(), height=400)


def tab_helper():
    st.header("Commands")
    st.markdown("Going forward, use these commands rather than editing CSVs by hand.")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Re-run full daily system
python3 -u canyon_final_v9_step12_daily_runner.py

# Open new dashboard
streamlit run canyon_final_v9_step17_dashboard_v3.py

# View paper ledger
python3 -u canyon_final_v9_step15_paper_trade_helper.py list

# Simulate a buy
python3 -u canyon_final_v9_step15_paper_trade_helper.py enter TLT 90

# Simulate a sell
python3 -u canyon_final_v9_step15_paper_trade_helper.py close TLT 92

# Re-run learning attribution
python3 -u canyon_final_v9_step14_learning_attribution.py
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit", page_icon="🏔", layout="wide")

    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; max-width: 1400px; }
        .risk-card { padding: 1.0rem 1.2rem; border-radius: 16px; margin-bottom: 1rem; border: 1px solid #ddd; }
        .risk-title { font-size: 1.25rem; font-weight: 700; margin-bottom: .35rem; }
        .risk-red { background: #fff1f1; border-color: #ffb3b3; }
        .risk-amber { background: #fff8e6; border-color: #ffd580; }
        .risk-green { background: #effaf2; border-color: #9ad7a8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏔 Canyon v9 PM Cockpit")
    st.caption("Research dashboard only. No broker connection. No live order. Paper trading is simulated.")

    tabs = st.tabs([
        "Overview",
        "Risk",
        "Stress + Sizing",
        "Execution Gate",
        "Paper Ledger",
        "Learning",
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
        tab_execution()
    with tabs[4]:
        tab_paper()
    with tabs[5]:
        tab_learning()
    with tabs[6]:
        tab_reports()
    with tabs[7]:
        tab_helper()


if __name__ == "__main__":
    main()
