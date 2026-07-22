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
        return "#eef6ff"
    if d == "PAPER_ONLY":
        return "#f4edff"
    if "SKIP" in d:
        return "#f4f4f4"
    if d == "WATCH":
        return "#effaf2"
    return "#f7f2ff"


def border_color(decision: str) -> str:
    d = str(decision).upper()
    if d == "WAIT":
        return "#2f80ed"
    if d == "PAPER_ONLY":
        return "#8e44ad"
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


def _decision_for_row(row):
    """Find the most useful decision/status label in a row."""
    candidate_cols = [
        "decision",
        "final_options_decision",
        "status",
        "final_status",
        "option_kill_zone_label",
        "gamma_squeeze_label",
        "urgency",
        "level",
    ]
    for col in candidate_cols:
        if col in row.index:
            val = str(row.get(col, "")).upper()
            if val and val not in {"NAN", "NONE"}:
                return val
    return ""


def _row_color(row):
    """Apply consistent decision colors across all table tabs."""
    label = _decision_for_row(row)

    # Main action colors
    if "WAIT" in label:
        return ["background-color: #eef6ff; border-left: 4px solid #2f80ed;" for _ in row]
    if "PAPER_ONLY" in label or "PAPER" in label:
        return ["background-color: #f4edff; border-left: 4px solid #8e44ad;" for _ in row]
    if "SKIP" in label or "BLOCKED" in label or "DO_NOT_REPEAT" in label:
        return ["background-color: #f4f4f4; border-left: 4px solid #999999;" for _ in row]
    if "WATCH" in label:
        return ["background-color: #effaf2; border-left: 4px solid #27ae60;" for _ in row]

    # Risk colors
    if "HIGH_OPTION_KILL_ZONE" in label or label == "HIGH":
        return ["background-color: #fff1f1; border-left: 4px solid #d64545;" for _ in row]
    if "MEDIUM_OPTION_KILL_ZONE" in label or label == "MEDIUM":
        return ["background-color: #fff8e6; border-left: 4px solid #f0b429;" for _ in row]
    if "LOW_OPTION_KILL_ZONE" in label or label == "LOW":
        return ["background-color: #f7f7f7; border-left: 4px solid #aaaaaa;" for _ in row]

    return ["" for _ in row]


def _highlight_key_cells(val):
    """Color individual decision-like cells too, so wide tables remain readable."""
    s = str(val).upper()
    if "WAIT" in s:
        return "background-color: #dcecff; color: #164a8b; font-weight: 700;"
    if "PAPER_ONLY" in s or s == "PAPER":
        return "background-color: #eadcff; color: #5b2c83; font-weight: 700;"
    if "SKIP" in s or "BLOCKED" in s or "DO_NOT_REPEAT" in s:
        return "background-color: #eeeeee; color: #555555; font-weight: 700;"
    if "WATCH" in s:
        return "background-color: #ddf5e6; color: #1e6b3a; font-weight: 700;"
    if "HIGH_OPTION_KILL_ZONE" in s or s == "HIGH":
        return "background-color: #ffe0e0; color: #922222; font-weight: 700;"
    if "MEDIUM_OPTION_KILL_ZONE" in s or s == "MEDIUM":
        return "background-color: #fff0c7; color: #7a5200; font-weight: 700;"
    return ""


def show_df(df: pd.DataFrame, height=420):
    if df.empty:
        st.info("No data yet.")
        return

    styled = (
        df.style
        .apply(_row_color, axis=1)
        .map(_highlight_key_cells)
    )

    try:
        st.dataframe(styled, hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(styled, hide_index=True, use_container_width=True, height=height)


def main():
    st.set_page_config(page_title="Canyon v9 PM Cockpit v8", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 PM Cockpit v8 — Action Cards")
    st.caption("Research only. No broker connection. No live order. Colors are now applied across tables.")

    st.markdown("""
    <div style="padding:12px 14px;border:1px solid #ddd;border-radius:14px;margin:10px 0 18px 0;background:#fafafa;">
      <b>Color legend across tabs:</b>
      <span style="background:#eef6ff;border-left:5px solid #2f80ed;padding:6px 10px;margin-left:10px;border-radius:8px;">Blue = WAIT: wait for breakout/breakdown confirmation</span>
      <span style="background:#f4edff;border-left:5px solid #8e44ad;padding:6px 10px;margin-left:10px;border-radius:8px;">Purple = PAPER_ONLY: equities/ETF paper only</span>
      <span style="background:#f4f4f4;border-left:5px solid #999999;padding:6px 10px;margin-left:10px;border-radius:8px;">Gray = SKIP/BLOCKED: pass tonight</span>
      <span style="background:#effaf2;border-left:5px solid #27ae60;padding:6px 10px;margin-left:10px;border-radius:8px;">Green = WATCH: observe</span>
      <span style="background:#fff1f1;border-left:5px solid #d64545;padding:6px 10px;margin-left:10px;border-radius:8px;">Red = HIGH risk / High Kill Zone</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:12px 14px;border:1px solid #ddd;border-radius:14px;margin:10px 0 18px 0;background:#fafafa;">
      <b>Color legend:</b>
      <span style="background:#f4edff;border-left:5px solid #8e44ad;padding:6px 10px;margin-left:10px;border-radius:8px;">Purple = PAPER_ONLY: equities/ETF paper only, no short-dated options</span>
      <span style="background:#eef6ff;border-left:5px solid #2f80ed;padding:6px 10px;margin-left:10px;border-radius:8px;">Blue = WAIT: wait for breakout/breakdown confirmation, no early chasing</span>
      <span style="background:#f4f4f4;border-left:5px solid #999999;padding:6px 10px;margin-left:10px;border-radius:8px;">Gray = SKIP: pass tonight, preserve focus</span>
      <span style="background:#effaf2;border-left:5px solid #27ae60;padding:6px 10px;margin-left:10px;border-radius:8px;">Green = WATCH: observe, wait for price/volume confirmation</span>
    </div>
    """, unsafe_allow_html=True)

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
