#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 55 — 10-Layer Dashboard v2

Reads Step 54 v2 matrix and scorecard.
Standalone dashboard. Does not patch old files.
No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path.cwd()

FILES = {
    "layer_audit": ROOT / "canyon_layer_status_audit.csv",
    "architecture": ROOT / "canyon_10_layer_architecture.md",
    "build_plan": ROOT / "canyon_layer_build_plan.md",

    "master_v2": ROOT / "master_10_layer_decision_matrix_v2.csv",
    "master_report_v2": ROOT / "master_10_layer_decision_report_v2.md",
    "scorecard": ROOT / "master_10_layer_scorecard.csv",

    "macro_report": ROOT / "macro_regime_report.md",
    "sector_report": ROOT / "sector_rotation_report.md",
    "fund_report": ROOT / "fundamental_report.md",
    "event_report": ROOT / "event_news_sec_insider_report.md",
    "tech_report": ROOT / "technical_microstructure_report.md",
    "action_cards": ROOT / "action_cards.csv",
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


def color_action(val):
    s = str(val).upper()
    if "TACTICAL_REVIEW" in s:
        return "background-color:#ddf5e6;color:#1e6b3a;font-weight:700;"
    if "LONG_TERM_REVIEW" in s:
        return "background-color:#e8f2ff;color:#164a8b;font-weight:700;"
    if "WAIT" in s:
        return "background-color:#eef6ff;color:#164a8b;font-weight:700;"
    if "PAPER" in s:
        return "background-color:#f4edff;color:#5b2c83;font-weight:700;"
    if "RISK" in s:
        return "background-color:#fff1f1;color:#922222;font-weight:700;"
    if "SKIP" in s or "DO_NOT_REPEAT" in s:
        return "background-color:#eeeeee;color:#555;font-weight:700;"
    if "RESEARCH" in s:
        return "background-color:#fff8e6;color:#7a5200;font-weight:700;"
    return ""


def color_state(val):
    s = str(val).upper()
    if s in {"OK", "RISK_ON", "LEADER", "QUALITY_HOLD_CANDIDATE", "EVENT_SUPPORT", "TACTICAL_CANDIDATE", "GREEN", "HAS_SAMPLE"}:
        return "background-color:#ddf5e6;color:#1e6b3a;font-weight:700;"
    if s in {"MIXED_RISK", "WATCH", "PARTIAL", "FUNDAMENTAL_WATCH", "NEUTRAL_OR_NO_EVENT", "AMBER", "PENDING_MANUAL_CHECKS"}:
        return "background-color:#fff8e6;color:#7a5200;font-weight:700;"
    if s in {"RED", "RISK_OFF_OR_CHOPPY", "EVENT_RISK", "NO_TECH_EDGE"}:
        return "background-color:#fff1f1;color:#922222;font-weight:700;"
    if "NO_DATA" in s or "NO_SAMPLE" in s:
        return "background-color:#eeeeee;color:#555;font-weight:700;"
    if "ETF_NOT_FUNDAMENTAL" in s:
        return "background-color:#eef6ff;color:#164a8b;font-weight:700;"
    if "WAIT" in s:
        return "background-color:#eef6ff;color:#164a8b;font-weight:700;"
    if "PAPER" in s:
        return "background-color:#f4edff;color:#5b2c83;font-weight:700;"
    if "SKIP" in s or "BLOCKED" in s:
        return "background-color:#eeeeee;color:#555;font-weight:700;"
    return ""


def show_df(df: pd.DataFrame, height=520, style=True):
    if df.empty:
        st.info("No data yet.")
        return
    obj = df
    if style:
        obj = df.style.map(color_state)
        if "master_action" in df.columns:
            obj = obj.map(color_action, subset=["master_action"])
    try:
        st.dataframe(obj, hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(obj, hide_index=True, use_container_width=True, height=height)


def tab_overview():
    st.header("10-Layer Overview v2")

    audit = read_csv(FILES["layer_audit"])
    master = read_csv(FILES["master_v2"])
    scorecard = read_csv(FILES["scorecard"])

    if not audit.empty:
        avg = pd.to_numeric(audit["maturity_score_0_5"], errors="coerce").mean()
        weak = int((pd.to_numeric(audit["maturity_score_0_5"], errors="coerce") <= 1).sum())
    else:
        avg, weak = 0, 10

    c1, c2, c3 = st.columns(3)
    c1.metric("Architecture maturity", f"{avg:.1f}/5")
    c2.metric("Weak/missing layers", weak)
    c3.metric("Master matrix rows", len(master))

    st.subheader("Layer scorecard from actual tickers")
    show_df(scorecard, height=300)

    st.subheader("Master compact matrix")
    if not master.empty:
        cols = [c for c in [
            "ticker", "master_action", "master_reason", "stack_score_avg",
            "L1_state", "L2_state", "L3_state", "L4_state", "L5_state",
            "L6_state", "L7_state", "L8_state", "L9_state", "L10_state"
        ] if c in master.columns]
        show_df(master[cols], height=600)
    else:
        st.warning("Run Step 54 first.")


def tab_master():
    st.header("Master 10-Layer Decision v2")
    st.markdown(read_md(FILES["master_report_v2"]))
    show_df(read_csv(FILES["master_v2"]), height=650)


def tab_layers():
    st.header("Architecture Layer Audit")
    audit = read_csv(FILES["layer_audit"])
    show_df(audit, height=600, style=False)


def tab_l2_l6():
    st.header("L2-L6 Reports")
    sub = st.tabs(["L2 Macro", "L3 Sector", "L4 Fundamental", "L5 Event", "L6 Technical"])
    with sub[0]:
        st.markdown(read_md(FILES["macro_report"]))
    with sub[1]:
        st.markdown(read_md(FILES["sector_report"]))
    with sub[2]:
        st.markdown(read_md(FILES["fund_report"]))
    with sub[3]:
        st.markdown(read_md(FILES["event_report"]))
    with sub[4]:
        st.markdown(read_md(FILES["tech_report"]))


def tab_action():
    st.header("Action Cards")
    show_df(read_csv(FILES["action_cards"]), height=650)


def tab_architecture():
    st.header("Architecture")
    st.markdown(read_md(FILES["architecture"]))


def tab_helper():
    st.header("Commands")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

# Full missing-layer update
python3 -u canyon_final_v9_step53_build_missing_layers_runner.py

# New v2 master matrix
python3 -u canyon_final_v9_step54_master_10_layer_decision_v2.py

# New dashboard
streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py

# Open reports
open master_10_layer_decision_report_v2.md
open macro_regime_report.md
open sector_rotation_report.md
open fundamental_report.md
open event_news_sec_insider_report.md
open technical_microstructure_report.md
""",
        language="bash",
    )


def main():
    st.set_page_config(page_title="Canyon v9 10-Layer v2", page_icon="🏔", layout="wide")
    st.title("🏔 Canyon v9 — 10-Layer System Dashboard v2")
    st.caption("Full-stack decision view. Options is only L7. No broker connection. No live order.")

    st.markdown("""
    <div style="padding:12px 14px;border:1px solid #ddd;border-radius:14px;margin:10px 0 18px 0;background:#fafafa;">
    <b>Color meaning:</b>
    <span style="background:#ddf5e6;border-left:5px solid #27ae60;padding:6px 10px;margin-left:10px;border-radius:8px;">Green = supportive</span>
    <span style="background:#fff8e6;border-left:5px solid #f0b429;padding:6px 10px;margin-left:10px;border-radius:8px;">Yellow = mixed / manual check</span>
    <span style="background:#eef6ff;border-left:5px solid #2f80ed;padding:6px 10px;margin-left:10px;border-radius:8px;">Blue = wait / ETF context</span>
    <span style="background:#f4edff;border-left:5px solid #8e44ad;padding:6px 10px;margin-left:10px;border-radius:8px;">Purple = paper only</span>
    <span style="background:#fff1f1;border-left:5px solid #d64545;padding:6px 10px;margin-left:10px;border-radius:8px;">Red = risk / hostile</span>
    <span style="background:#eeeeee;border-left:5px solid #999;padding:6px 10px;margin-left:10px;border-radius:8px;">Gray = no data / skip</span>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview",
        "Master v2",
        "Layer Audit",
        "L2-L6 Reports",
        "Action Cards",
        "Architecture",
        "Helper",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_master()
    with tabs[2]:
        tab_layers()
    with tabs[3]:
        tab_l2_l6()
    with tabs[4]:
        tab_action()
    with tabs[5]:
        tab_architecture()
    with tabs[6]:
        tab_helper()


if __name__ == "__main__":
    main()
