#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 40 — Dashboard v8 Cards

Standalone dashboard. Does not patch old files.
Shows action cards in a readable layout.

No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

ACTION_CARDS = ROOT / "action_cards.csv"
ACTION_MD = ROOT / "action_cards.md"
TONIGHT_MD = ROOT / "tonight_action_plan.md"
DECISION_CSV = ROOT / "options_decision_matrix.csv"
GAMMA_CSV = ROOT / "gamma_squeeze_candidates.csv"
KILL_CSV = ROOT / "option_kill_zone_risk.csv"
RISK_CSV = ROOT / "exposure_warnings.csv"
STRESS_CSV = ROOT / "scenario_stress_results.csv"


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


def badge_color(decision: str) -> str:
    d = str(decision).upper()
    if d == "WAIT":
        return "#fff8e6"
    if d == "PAPER_ONLY":
        return "#eef6ff"
    if "SKIP" in d:
        return "#f4f4f4"
    if d == "WATCH":
        return "#effaf2"
    return "#f7f2ff"


def border_color(decision: str) -> str:
    d = str(decision).upper()
    if d == "WAIT":
        return "#f0b429"
    if d == "PAPER_ONLY":
        return "#2f80ed"
    if "SKIP" in d:
        return "#999999"
    if d == "WATCH":
        return "#27ae60"
    return "#8e44ad"


def card_html(r: pd.Series) -> str:
    decision = r.get("decision", "")
    bg = badge_color(decision)
    border = border_color(decision)

    return f"""
    <div style="background:{bg}; border-left:7px solid {border}; padding:18px; border-radius:16px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <h3 style="margin-top:0;">{r.get('ticker','')} — {decision} / {r.get('urgency','')}</h3>
      <p style="font-size:17px;"><b>{r.get('one_liner','')}</b></p>
      <p>
        <b>Spot:</b> {r.get('spot','')} &nbsp; 
        <b>Breakout:</b> {r.get('breakout_trigger','')} ({r.get('breakout_distance','')}) &nbsp;
        <b>Breakdown:</b> {r.get('breakdown_trigger','')} ({r.get('breakdown_distance','')})
      </p>
      <p>
        <b>Gamma:</b> {r.get('gamma_label','')} score {r.get('gamma_score','')}<br>
        <b>Kill Zone:</b> {r.get('kill_zone_label','')}<br>
        <b>Risk:</b> {r.get('risk_note','')}
      </p>
      <p>
        <b>Allowed:</b> {r.get('allowed_action','')}<br>
        <b>Forbidden:</b> {r.get('forbidden_action','')}<br>
        <b>Rule:</b> {r.get('trigger_rule','')}
      </p>
      <p><b>Live allowed:</b> {r.get('live_allowed','NO')}</p>
    </div>
    """


def show_df(df: pd.DataFrame, height=420):
    if df.empty:
        st.info("No data yet.")
        return
    try:
        st.dataframe(df, hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v8", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 PM Cockpit v8 — Action Cards")
    st.caption("Research only. No broker connection. No live order.")

    tabs = st.tabs([
        "Action Cards",
        "Tonight Plan",
        "Decision Table",
        "Gamma",
        "Kill Zone",
        "Risk",
        "Helper",
    ])

    cards = read_csv(ACTION_CARDS)

    with tabs[0]:
        st.header("Action Cards")
        if cards.empty:
            st.warning("No action cards yet. Run Step 39 first.")
            st.code("python3 -u canyon_final_v9_step39_action_cards.py", language="bash")
        else:
            filters = ["ALL"] + sorted(cards["decision"].dropna().unique().tolist()) if "decision" in cards.columns else ["ALL"]
            choice = st.selectbox("Filter by decision", filters)
            view = cards.copy()
            if choice != "ALL":
                view = view[view["decision"] == choice]

            for _, row in view.iterrows():
                st.markdown(card_html(row), unsafe_allow_html=True)

    with tabs[1]:
        st.header("Tonight Plan")
        st.markdown(read_md(TONIGHT_MD))

    with tabs[2]:
        st.header("Decision Table")
        show_df(read_csv(DECISION_CSV), height=560)

    with tabs[3]:
        st.header("Gamma")
        show_df(read_csv(GAMMA_CSV), height=560)

    with tabs[4]:
        st.header("Kill Zone")
        show_df(read_csv(KILL_CSV), height=560)

    with tabs[5]:
        st.header("Risk")
        st.subheader("Exposure Warnings")
        show_df(read_csv(RISK_CSV), height=320)
        st.subheader("Stress")
        show_df(read_csv(STRESS_CSV), height=320)

    with tabs[6]:
        st.header("Commands")
        st.code(
            """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Stable action workflow
python3 -u canyon_final_v9_step37_safe_options_action_runner.py
python3 -u canyon_final_v9_step39_action_cards.py
streamlit run canyon_final_v9_step40_dashboard_v8_cards.py

# Open reports
open action_cards.md
open tonight_action_plan.md
open options_decision_matrix.md
""",
            language="bash",
        )


if __name__ == "__main__":
    main()
