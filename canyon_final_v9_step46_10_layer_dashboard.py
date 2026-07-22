#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 46 — 10-Layer System Dashboard

Standalone Streamlit dashboard for the full architecture.
No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "architecture": ROOT / "canyon_10_layer_architecture.md",
    "layer_audit": ROOT / "canyon_layer_status_audit.csv",
    "build_plan": ROOT / "canyon_layer_build_plan.md",
    "master_report": ROOT / "master_10_layer_decision_report.md",
    "master_csv": ROOT / "master_10_layer_decision_matrix.csv",
    "action_cards": ROOT / "action_cards.csv",
    "tonight": ROOT / "tonight_action_plan.md",
    "decision": ROOT / "options_decision_matrix.csv",
    "risk": ROOT / "exposure_warnings.csv",
    "stress": ROOT / "scenario_stress_results.csv",
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
        return f"_Could not read `{path.name}`: {e}_"


def show_df(df: pd.DataFrame, height=480):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(df, hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)


def color_layer_score(val):
    try:
        v = int(float(val))
    except Exception:
        return ""
    if v >= 5:
        return "background-color:#ddf5e6;color:#1e6b3a;font-weight:700;"
    if v >= 4:
        return "background-color:#eef6ff;color:#164a8b;font-weight:700;"
    if v >= 3:
        return "background-color:#fff8e6;color:#7a5200;font-weight:700;"
    if v >= 1:
        return "background-color:#f4edff;color:#5b2c83;font-weight:700;"
    return "background-color:#f4f4f4;color:#555;font-weight:700;"


def show_layer_audit():
    df = read_csv(FILES["layer_audit"])
    if df.empty:
        st.info("Run Step 44 first.")
        return
    show = df[[
        "layer_id", "layer_name", "maturity_score_0_5",
        "maturity_status", "missing_outputs", "next_build"
    ]].copy()
    styled = show.style.map(color_layer_score, subset=["maturity_score_0_5"])
    try:
        st.dataframe(styled, hide_index=True, width="stretch", height=520)
    except TypeError:
        st.dataframe(styled, hide_index=True, use_container_width=True, height=520)


def tab_overview():
    st.header("10-Layer Overview")
    audit = read_csv(FILES["layer_audit"])
    master = read_csv(FILES["master_csv"])

    if not audit.empty:
        avg = pd.to_numeric(audit["maturity_score_0_5"], errors="coerce").mean()
        missing_layers = int((pd.to_numeric(audit["maturity_score_0_5"], errors="coerce") <= 1).sum())
    else:
        avg = 0
        missing_layers = 10

    c1, c2, c3 = st.columns(3)
    c1.metric("Average layer maturity", f"{avg:.1f}/5")
    c2.metric("Weak/missing layers", missing_layers)
    c3.metric("Master rows", len(master))

    st.subheader("Layer audit")
    show_layer_audit()

    st.subheader("Master compact view")
    if not master.empty:
        cols = [c for c in ["ticker", "master_action", "master_reason", "L2_macro", "L4_fundamental", "L7_options", "L8_risk", "L9_execution"] if c in master.columns]
        show_df(master[cols], height=420)
    else:
        st.info("Run Step 45 first.")


def tab_architecture():
    st.header("Architecture")
    st.markdown(read_md(FILES["architecture"]))


def tab_master():
    st.header("Master 10-Layer Decision")
    st.markdown(read_md(FILES["master_report"]))
    show_df(read_csv(FILES["master_csv"]), height=600)


def tab_build_plan():
    st.header("Build Plan")
    st.markdown(read_md(FILES["build_plan"]))


def tab_action():
    st.header("Current Action Layer")
    st.markdown(read_md(FILES["tonight"]))
    st.subheader("Action Cards")
    show_df(read_csv(FILES["action_cards"]), height=500)
    st.subheader("Options Decision")
    show_df(read_csv(FILES["decision"]), height=500)


def tab_risk():
    st.header("Risk")
    st.subheader("Exposure Warnings")
    show_df(read_csv(FILES["risk"]), height=320)
    st.subheader("Stress")
    show_df(read_csv(FILES["stress"]), height=320)


def tab_helper():
    st.header("Commands")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Build/update 10-layer architecture
python3 -u canyon_final_v9_step44_layer_architecture_registry.py
python3 -u canyon_final_v9_step45_master_10_layer_decision.py

# Open 10-layer dashboard
streamlit run canyon_final_v9_step46_10_layer_dashboard.py

# Stable action workflow
python3 -u canyon_final_v9_step37_safe_options_action_runner.py
python3 -u canyon_final_v9_step39_action_cards.py
streamlit run canyon_final_v9_step40_dashboard_v8_cards.py
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 — 10 Layer System", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 — 10-Layer System Dashboard")
    st.caption("Options is only L7. This dashboard tracks the whole system.")

    tabs = st.tabs([
        "Overview",
        "Architecture",
        "Master Decision",
        "Build Plan",
        "Current Action",
        "Risk",
        "Helper",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_architecture()
    with tabs[2]:
        tab_master()
    with tabs[3]:
        tab_build_plan()
    with tabs[4]:
        tab_action()
    with tabs[5]:
        tab_risk()
    with tabs[6]:
        tab_helper()


if __name__ == "__main__":
    main()
