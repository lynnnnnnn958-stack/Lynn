#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 34 — PM Cockpit v6 Decision Dashboard

This is a clean standalone dashboard.
It does NOT patch old dashboard files.

Reads local files only:
- options_decision_matrix.csv / .md
- gamma_squeeze_candidates.csv / options_gamma_report.md
- option_kill_zone_risk.csv / option_kill_zone_report.md
- pre_trade_checklist.csv / .md
- exposure_warnings.csv
- scenario_stress_results.csv
- position_sizing_recommendations.csv
- paper_portfolio_ledger.csv
- learning_attribution_report.md
- system_health_check.md

No broker connection. No live orders.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "decision_csv": ROOT / "options_decision_matrix.csv",
    "decision_md": ROOT / "options_decision_matrix.md",
    "gamma_csv": ROOT / "gamma_squeeze_candidates.csv",
    "gamma_md": ROOT / "options_gamma_report.md",
    "kill_csv": ROOT / "option_kill_zone_risk.csv",
    "kill_md": ROOT / "option_kill_zone_report.md",
    "chain_csv": ROOT / "options_chain_snapshot.csv",
    "fetch_md": ROOT / "yfinance_options_fetch_report.md",
    "pre_csv": ROOT / "pre_trade_checklist.csv",
    "pre_md": ROOT / "pre_trade_checklist.md",
    "warnings_csv": ROOT / "exposure_warnings.csv",
    "stress_csv": ROOT / "scenario_stress_results.csv",
    "sizing_csv": ROOT / "position_sizing_recommendations.csv",
    "ledger_csv": ROOT / "paper_portfolio_ledger.csv",
    "paper_md": ROOT / "paper_ledger_summary.md",
    "learning_md": ROOT / "learning_attribution_report.md",
    "learning_csv": ROOT / "learning_attribution_summary.csv",
    "health_md": ROOT / "system_health_check.md",
    "daily_md": ROOT / "daily_pm_report.md",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()


def read_md(path: Path) -> str:
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Could not read `{path.name}`: {e}_"


def fnum(x):
    try:
        s = str(x).replace("%", "").replace(",", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    for col in out.columns:
        lc = col.lower()
        if any(k in lc for k in ["distance", "ratio", "spread", "weight", "pnl", "loss"]):
            def fmt_pct(v):
                vv = fnum(v)
                return f"{vv:.2%}" if vv is not None else v
            out[col] = out[col].apply(fmt_pct)
        elif "gex" in lc:
            def fmt_gex(v):
                vv = fnum(v)
                return f"{vv:,.0f}" if vv is not None else v
            out[col] = out[col].apply(fmt_gex)

    return out


def show_df(df: pd.DataFrame, height: int = 380):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(fmt_df(df), hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(fmt_df(df), hide_index=True, use_container_width=True, height=height)


def metric_counts():
    decision = read_csv(FILES["decision_csv"])
    pre = read_csv(FILES["pre_csv"])
    warn = read_csv(FILES["warnings_csv"])
    stress = read_csv(FILES["stress_csv"])
    gamma = read_csv(FILES["gamma_csv"])
    kill = read_csv(FILES["kill_csv"])

    high = med = 0
    if not warn.empty and "level" in warn.columns:
        levels = warn["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())

    worst = None
    if not stress.empty and "estimated_pnl" in stress.columns:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())

    pending = blocked = 0
    if not pre.empty and "final_status" in pre.columns:
        fs = pre["final_status"].astype(str).str.upper()
        pending = int((fs == "PENDING_MANUAL_CHECKS").sum())
        blocked = int((fs == "BLOCKED").sum())

    paper_review = watch = skip_options = wait = research = 0
    if not decision.empty and "final_options_decision" in decision.columns:
        fd = decision["final_options_decision"].astype(str).str.upper()
        paper_review = int(fd.str.contains("PAPER").sum())
        watch = int((fd == "WATCH").sum())
        wait = int((fd == "WAIT").sum())
        skip_options = int(fd.str.contains("SKIP").sum())
        research = int((fd == "RESEARCH_ONLY").sum())

    gamma_high = 0
    if not gamma.empty and "gamma_squeeze_label" in gamma.columns:
        gamma_high = int(gamma["gamma_squeeze_label"].astype(str).str.contains("HIGH", case=False, na=False).sum())

    kill_high = 0
    if not kill.empty and "option_kill_zone_label" in kill.columns:
        kill_high = int(kill["option_kill_zone_label"].astype(str).str.contains("HIGH", case=False, na=False).sum())

    risk = "RED" if high > 0 or (worst is not None and worst <= -0.02) else ("AMBER" if med >= 3 else "GREEN")

    return {
        "risk": risk, "high": high, "med": med, "worst": worst,
        "pending": pending, "blocked": blocked,
        "paper_review": paper_review, "watch": watch, "wait": wait,
        "skip_options": skip_options, "research": research,
        "gamma_high": gamma_high, "kill_high": kill_high,
    }


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def tab_overview():
    m = metric_counts()

    risk_msg = {
        "RED": "High risk: do not convert candidates directly to trades; check concentration / stress / kill-zone first.",
        "AMBER": "Moderate risk: research OK but keep positions small, continue paper trading.",
        "GREEN": "Relatively low risk: manual review still required.",
    }.get(m["risk"], "")

    bg = {"RED": "#fff1f1", "AMBER": "#fff8e6", "GREEN": "#effaf2"}.get(m["risk"], "#f5f5f5")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;border:1px solid #ddd;background:{bg};">
        <h3>Portfolio Risk Light: {m['risk']}</h3>
        <p>{risk_msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Warnings", f"H{m['high']}/M{m['med']}")
    c2.metric("Worst Stress", pct(m["worst"]) if m["worst"] is not None else "N/A")
    c3.metric("Pending Checks", m["pending"])
    c4.metric("Blocked", m["blocked"])
    c5.metric("Gamma High", m["gamma_high"])
    c6.metric("Kill-Zone High", m["kill_high"])
    c7.metric("Paper/Watch", f"{m['paper_review']}/{m['watch']}")

    st.subheader("Today's reading order")
    show_df(pd.DataFrame([
        ["1", "Options Decision", "Start with the final action: PAPER / WATCH / WAIT / SKIP."],
        ["2", "Option Kill Zone", "Check for pin risk, IV crush, and theta bleed."],
        ["3", "Options Gamma", "Check call wall, put wall, net GEX, and gamma score."],
        ["4", "Pre-trade", "Manual check: news, earnings, liquidity, spread."],
        ["5", "Risk", "Check portfolio concentration and worst-case stress scenario."],
    ], columns=["step", "page", "what_to_do"]), height=220)

    st.subheader("Options Decision Quick View")
    dec = read_csv(FILES["decision_csv"])
    if not dec.empty:
        cols = [c for c in [
            "ticker", "final_options_decision", "rule",
            "gamma_squeeze_label", "gamma_squeeze_score",
            "option_kill_zone_label", "option_kill_zone_score",
            "portfolio_risk_light", "live_allowed", "explanation"
        ] if c in dec.columns]
        show_df(dec[cols], height=420)
    else:
        st.info("Run Step 32 first: python3 -u canyon_final_v9_step32_options_decision_matrix.py")


def tab_decision():
    st.header("Options Decision Matrix")
    st.markdown("This page is the final readout: combines Gamma, Kill Zone, and Pre-trade into a single research action.")
    st.markdown(read_md(FILES["decision_md"]))
    st.subheader("Decision CSV")
    show_df(read_csv(FILES["decision_csv"]), height=560)


def tab_gamma():
    st.header("Options / Gamma")
    st.markdown("This is not real dealer positioning — OI/volume-based gamma pressure proxy only.")
    st.markdown(read_md(FILES["gamma_md"]))
    st.subheader("Gamma Candidates")
    show_df(read_csv(FILES["gamma_csv"]), height=520)

    with st.expander("Raw option chain"):
        chain = read_csv(FILES["chain_csv"])
        cols = [c for c in [
            "ticker", "expiration_date", "option_type", "strike", "spot",
            "gamma", "delta", "implied_volatility", "open_interest",
            "volume", "bid", "ask", "dte"
        ] if c in chain.columns]
        show_df(chain[cols] if cols else chain, height=520)

    with st.expander("Fetch report"):
        st.markdown(read_md(FILES["fetch_md"]))


def tab_kill():
    st.header("Option Kill Zone")
    st.markdown("This layer prevents chasing weekly OTM that could get pinned, IV-crushed, theta-bled, or spread-killed.")
    st.markdown(read_md(FILES["kill_md"]))
    st.subheader("Kill Zone Table")
    show_df(read_csv(FILES["kill_csv"]), height=560)


def tab_pretrade():
    st.header("Pre-trade")
    st.markdown(read_md(FILES["pre_md"]))
    show_df(read_csv(FILES["pre_csv"]), height=560)


def tab_risk():
    st.header("Risk")
    st.subheader("Exposure Warnings")
    show_df(read_csv(FILES["warnings_csv"]), height=320)
    st.subheader("Scenario Stress")
    show_df(read_csv(FILES["stress_csv"]), height=320)
    st.subheader("Position Sizing")
    show_df(read_csv(FILES["sizing_csv"]), height=440)


def tab_paper():
    st.header("Paper Ledger")
    st.markdown(read_md(FILES["paper_md"]))
    show_df(read_csv(FILES["ledger_csv"]), height=560)


def tab_learning():
    st.header("Learning")
    st.markdown(read_md(FILES["learning_md"]))
    show_df(read_csv(FILES["learning_csv"]), height=420)


def tab_health():
    st.header("Health")
    st.markdown(read_md(FILES["health_md"]))


def tab_reports():
    st.header("Daily PM Report")
    st.markdown(read_md(FILES["daily_md"]))


def tab_helper():
    st.header("Commands")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# 1. Options layer
python3 -u canyon_final_v9_step25_options_daily_runner.py

# 2. Final options decision matrix
python3 -u canyon_final_v9_step32_options_decision_matrix.py

# 3. Open v6 dashboard
streamlit run canyon_final_v9_step34_dashboard_v6_decision.py

# 4. Full system daily update
python3 -u canyon_final_v9_step27_full_daily_pipeline.py
python3 -u canyon_final_v9_step25_options_daily_runner.py
python3 -u canyon_final_v9_step32_options_decision_matrix.py
streamlit run canyon_final_v9_step34_dashboard_v6_decision.py
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v6", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 PM Cockpit v6")
    st.caption("Research only. No broker connection. No live order. Gamma is proxy, not true dealer position.")

    tabs = st.tabs([
        "Overview",
        "Options Decision",
        "Options Gamma",
        "Option Kill Zone",
        "Pre-trade",
        "Risk",
        "Paper",
        "Learning",
        "Health",
        "Reports",
        "Helper",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_decision()
    with tabs[2]:
        tab_gamma()
    with tabs[3]:
        tab_kill()
    with tabs[4]:
        tab_pretrade()
    with tabs[5]:
        tab_risk()
    with tabs[6]:
        tab_paper()
    with tabs[7]:
        tab_learning()
    with tabs[8]:
        tab_health()
    with tabs[9]:
        tab_reports()
    with tabs[10]:
        tab_helper()


if __name__ == "__main__":
    main()
