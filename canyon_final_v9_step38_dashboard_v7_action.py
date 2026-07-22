#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 38 — PM Cockpit v7 Action Dashboard

Standalone dashboard. Does not patch old files.

Focus:
- Tonight Plan
- Options Decision
- Watch Triggers
- Gamma
- Kill Zone
- Risk
- Reports

No broker connection. No live orders.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "tonight_md": ROOT / "tonight_action_plan.md",
    "watch_csv": ROOT / "watch_triggers.csv",
    "decision_md": ROOT / "options_decision_matrix.md",
    "decision_csv": ROOT / "options_decision_matrix.csv",
    "gamma_md": ROOT / "options_gamma_report.md",
    "gamma_csv": ROOT / "gamma_squeeze_candidates.csv",
    "kill_md": ROOT / "option_kill_zone_report.md",
    "kill_csv": ROOT / "option_kill_zone_risk.csv",
    "pre_md": ROOT / "pre_trade_checklist.md",
    "pre_csv": ROOT / "pre_trade_checklist.csv",
    "warnings_csv": ROOT / "exposure_warnings.csv",
    "stress_csv": ROOT / "scenario_stress_results.csv",
    "sizing_csv": ROOT / "position_sizing_recommendations.csv",
    "ledger_csv": ROOT / "paper_portfolio_ledger.csv",
    "learning_md": ROOT / "learning_attribution_report.md",
    "health_md": ROOT / "system_health_check.md",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        st.warning(f"Cannot read {path.name}: {e}")
        return pd.DataFrame()


def read_md(path: Path) -> str:
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Cannot read `{path.name}`: {e}_"


def fnum(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for col in d.columns:
        lc = col.lower()

        if any(k in lc for k in ["distance", "ratio", "spread", "weight", "pnl", "loss"]):
            def pct(v):
                vv = fnum(v)
                return f"{vv:.2%}" if vv is not None else v
            d[col] = d[col].apply(pct)

        elif "gex" in lc:
            def gex(v):
                vv = fnum(v)
                return f"{vv:,.0f}" if vv is not None else v
            d[col] = d[col].apply(gex)

        elif any(k in lc for k in ["trigger", "spot", "wall", "pain"]):
            def px(v):
                vv = fnum(v)
                return f"{vv:.2f}" if vv is not None else v
            d[col] = d[col].apply(px)

    return d


def show_df(df: pd.DataFrame, height: int = 420):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(fmt_df(df), hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(fmt_df(df), hide_index=True, use_container_width=True, height=height)


def calc_metrics():
    watch = read_csv(FILES["watch_csv"])
    decision = read_csv(FILES["decision_csv"])
    warn = read_csv(FILES["warnings_csv"])
    stress = read_csv(FILES["stress_csv"])

    high_urgency = 0
    wait = 0
    paper = 0
    skip = 0
    if not watch.empty:
        if "urgency" in watch.columns:
            high_urgency = int((watch["urgency"].astype(str).str.upper() == "HIGH").sum())
        if "decision" in watch.columns:
            d = watch["decision"].astype(str).str.upper()
            wait = int((d == "WAIT").sum())
            paper = int(d.str.contains("PAPER").sum())
            skip = int(d.str.contains("SKIP").sum())

    gamma_high = 0
    kill_high = 0
    if not decision.empty:
        if "gamma_squeeze_label" in decision.columns:
            gamma_high = int(decision["gamma_squeeze_label"].astype(str).str.contains("HIGH", case=False, na=False).sum())
        if "option_kill_zone_label" in decision.columns:
            kill_high = int(decision["option_kill_zone_label"].astype(str).str.contains("HIGH", case=False, na=False).sum())

    warn_high = warn_med = 0
    if not warn.empty and "level" in warn.columns:
        levels = warn["level"].astype(str).str.upper()
        warn_high = int((levels == "HIGH").sum())
        warn_med = int((levels == "MEDIUM").sum())

    worst = None
    if not stress.empty and "estimated_pnl" in stress.columns:
        vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
        if vals.notna().any():
            worst = float(vals.min())

    risk = "RED" if warn_high > 0 or (worst is not None and worst <= -0.02) else ("AMBER" if warn_med >= 3 else "GREEN")

    return {
        "risk": risk,
        "high_urgency": high_urgency,
        "wait": wait,
        "paper": paper,
        "skip": skip,
        "gamma_high": gamma_high,
        "kill_high": kill_high,
        "warn_high": warn_high,
        "warn_med": warn_med,
        "worst": worst,
    }


def tab_overview():
    m = calc_metrics()
    bg = {"RED": "#fff1f1", "AMBER": "#fff8e6", "GREEN": "#effaf2"}.get(m["risk"], "#f5f5f5")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;border:1px solid #ddd;background:{bg};">
        <h3>Action Dashboard Risk Light: {m['risk']}</h3>
        <p>This is a research panel, not an order panel. When RED, observation or paper only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("High Urgency", m["high_urgency"])
    c2.metric("WAIT", m["wait"])
    c3.metric("PAPER", m["paper"])
    c4.metric("SKIP", m["skip"])
    c5.metric("Gamma High", m["gamma_high"])
    c6.metric("Kill High", m["kill_high"])

    st.subheader("Tonight Action Snapshot")
    watch = read_csv(FILES["watch_csv"])
    cols = [c for c in [
        "ticker", "decision", "urgency", "action",
        "call_wall_breakout_trigger", "call_wall_distance",
        "put_wall_breakdown_trigger", "put_wall_distance",
        "kill_zone_label", "notes"
    ] if c in watch.columns]
    show_df(watch[cols] if cols else watch, height=450)


def tab_tonight():
    st.header("Tonight Plan")
    st.markdown(read_md(FILES["tonight_md"]))
    st.subheader("Watch Triggers")
    show_df(read_csv(FILES["watch_csv"]), height=560)


def tab_decision():
    st.header("Options Decision")
    st.markdown(read_md(FILES["decision_md"]))
    show_df(read_csv(FILES["decision_csv"]), height=560)


def tab_gamma():
    st.header("Options Gamma")
    st.markdown(read_md(FILES["gamma_md"]))
    show_df(read_csv(FILES["gamma_csv"]), height=560)


def tab_kill():
    st.header("Option Kill Zone")
    st.markdown(read_md(FILES["kill_md"]))
    show_df(read_csv(FILES["kill_csv"]), height=560)


def tab_pretrade():
    st.header("Pre-trade")
    st.markdown(read_md(FILES["pre_md"]))
    show_df(read_csv(FILES["pre_csv"]), height=560)


def tab_risk():
    st.header("Risk")
    st.subheader("Exposure Warnings")
    show_df(read_csv(FILES["warnings_csv"]), height=320)
    st.subheader("Stress")
    show_df(read_csv(FILES["stress_csv"]), height=320)
    st.subheader("Position Sizing")
    show_df(read_csv(FILES["sizing_csv"]), height=420)


def tab_paper():
    st.header("Paper Ledger")
    show_df(read_csv(FILES["ledger_csv"]), height=560)


def tab_learning():
    st.header("Learning")
    st.markdown(read_md(FILES["learning_md"]))


def tab_health():
    st.header("Health")
    st.markdown(read_md(FILES["health_md"]))


def tab_helper():
    st.header("Commands")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Stable daily action stack
python3 -u canyon_final_v9_step37_safe_options_action_runner.py

# Open v7
streamlit run canyon_final_v9_step38_dashboard_v7_action.py

# Manual fallback
python3 -u canyon_final_v9_step25_options_daily_runner.py
python3 -u canyon_final_v9_step32_options_decision_matrix.py
python3 -u canyon_final_v9_step36_tonight_action_plan.py
streamlit run canyon_final_v9_step38_dashboard_v7_action.py
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v7", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 PM Cockpit v7")
    st.caption("Action dashboard. Research only. No broker connection. No live order.")

    tabs = st.tabs([
        "Overview",
        "Tonight Plan",
        "Options Decision",
        "Options Gamma",
        "Option Kill Zone",
        "Pre-trade",
        "Risk",
        "Paper",
        "Learning",
        "Health",
        "Helper",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_tonight()
    with tabs[2]:
        tab_decision()
    with tabs[3]:
        tab_gamma()
    with tabs[4]:
        tab_kill()
    with tabs[5]:
        tab_pretrade()
    with tabs[6]:
        tab_risk()
    with tabs[7]:
        tab_paper()
    with tabs[8]:
        tab_learning()
    with tabs[9]:
        tab_health()
    with tabs[10]:
        tab_helper()


if __name__ == "__main__":
    main()
