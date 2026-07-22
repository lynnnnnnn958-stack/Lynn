#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 11 — Local PM Dashboard

Purpose:
Consolidates all previously generated reports into a single local web page:
- daily_pm_report.md
- exposure_dashboard.md
- stress_position_sizing_report.md
- exposure_dashboard.csv
- exposure_warnings.csv
- scenario_stress_results.csv
- position_sizing_recommendations.csv
- canyon_trade_journal.csv
- execution_gate_review.csv
- pre_trade_order_ticket.csv

Principles:
- No market data downloads
- No automatic order submission
- No broker connection
- Read local files only
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Streamlit is not installed. Run: pip install streamlit"
    )


ROOT = Path.cwd()

FILES = {
    "PM Report": ROOT / "daily_pm_report.md",
    "Exposure Dashboard": ROOT / "exposure_dashboard.md",
    "Stress + Sizing": ROOT / "stress_position_sizing_report.md",
    "Execution Gate Review": ROOT / "execution_gate_review.csv",
    "Pre-trade Order Ticket": ROOT / "pre_trade_order_ticket.csv",
    "Trade Journal": ROOT / "canyon_trade_journal.csv",
    "Exposure Table": ROOT / "exposure_dashboard.csv",
    "Exposure Warnings": ROOT / "exposure_warnings.csv",
    "Scenario Stress Results": ROOT / "scenario_stress_results.csv",
    "Sizing Recommendations": ROOT / "position_sizing_recommendations.csv",
}


def read_text(path: Path) -> str:
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"_Failed to read `{path.name}`: {e}_"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fmt_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return x


def show_csv(title: str, path: Path):
    st.subheader(title)
    if not path.exists():
        st.info(f"{path.name} not found yet.")
        return
    df = read_csv(path)
    if df.empty:
        st.warning(f"{path.name} exists but is empty or unreadable.")
        return

    st.caption(f"{len(df)} rows · {path.name}")

    # display formatted copy
    df_show = df.copy()
    for col in df_show.columns:
        cl = col.lower()
        if any(k in cl for k in ["weight", "pnl", "loss", "shock", "reduction", "exposure"]):
            df_show[col] = df_show[col].apply(fmt_pct)

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.download_button(
        label=f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
    )


def status_badge(path: Path) -> str:
    return "FOUND" if path.exists() else "MISSING"


def build_summary():
    review = read_csv(FILES["Execution Gate Review"])
    warnings = read_csv(FILES["Exposure Warnings"])
    stress = read_csv(FILES["Scenario Stress Results"])
    sizing = read_csv(FILES["Sizing Recommendations"])
    journal = read_csv(FILES["Trade Journal"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Gate candidates", 0 if review.empty else len(review))
    with c2:
        if not warnings.empty:
            high = (warnings.get("level", pd.Series(dtype=str)).astype(str).str.upper() == "HIGH").sum()
            med = (warnings.get("level", pd.Series(dtype=str)).astype(str).str.upper() == "MEDIUM").sum()
            st.metric("Warnings", f"H{high}/M{med}")
        else:
            st.metric("Warnings", "N/A")
    with c3:
        if not stress.empty and "estimated_pnl" in stress.columns:
            worst = float(stress["estimated_pnl"].min())
            st.metric("Worst scenario", f"{worst:.2%}")
        else:
            st.metric("Worst scenario", "N/A")
    with c4:
        if not journal.empty:
            st.metric("Journal rows", len(journal))
        else:
            st.metric("Journal rows", 0)

    if not stress.empty and {"scenario", "estimated_pnl"}.issubset(stress.columns):
        st.markdown("### Worst Scenario Ranking")
        s = stress.sort_values("estimated_pnl").copy()
        s["estimated_pnl"] = s["estimated_pnl"].apply(fmt_pct)
        st.dataframe(s, use_container_width=True, hide_index=True)

    if not sizing.empty and "suggested_action" in sizing.columns:
        st.markdown("### Current Sizing Actions")
        cols = [c for c in [
            "ticker", "sleeve", "decision", "risk_bucket",
            "effective_weight", "suggested_weight",
            "suggested_action", "sizing_reason"
        ] if c in sizing.columns]
        s = sizing[cols].copy()
        for col in ["effective_weight", "suggested_weight"]:
            if col in s.columns:
                s[col] = s[col].apply(fmt_pct)
        st.dataframe(s, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Canyon v9 PM Dashboard",
        page_icon="🏔",
        layout="wide",
    )

    st.title("🏔 Canyon v9 PM Dashboard")
    st.caption("Local-only dashboard. It reads your generated Canyon reports and CSVs. It does not trade.")

    with st.sidebar:
        st.header("File Status")
        for name, path in FILES.items():
            st.write(f"**{name}:** {status_badge(path)}")
        st.divider()
        st.warning(
            "This dashboard is not an order system. WATCHLIST, REVIEW, and ALLOW are not trade instructions."
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "PM Report",
        "Exposure",
        "Stress + Sizing",
        "Execution Gate",
        "Journal",
    ])

    with tab1:
        build_summary()

    with tab2:
        st.markdown(read_text(FILES["PM Report"]))

    with tab3:
        st.markdown(read_text(FILES["Exposure Dashboard"]))
        st.divider()
        show_csv("Exposure Table", FILES["Exposure Table"])
        show_csv("Exposure Warnings", FILES["Exposure Warnings"])

    with tab4:
        st.markdown(read_text(FILES["Stress + Sizing"]))
        st.divider()
        show_csv("Scenario Stress Results", FILES["Scenario Stress Results"])
        show_csv("Position Sizing Recommendations", FILES["Sizing Recommendations"])

    with tab5:
        show_csv("Execution Gate Review", FILES["Execution Gate Review"])
        show_csv("Pre-trade Order Ticket", FILES["Pre-trade Order Ticket"])

    with tab6:
        show_csv("Trade Journal", FILES["Trade Journal"])
        st.markdown(
            """
            ### Journal Rule

            The learning engine should only learn from CLOSED trades.
            WATCHLIST and PLANNED rows are not performance data.
            """
        )


if __name__ == "__main__":
    main()
