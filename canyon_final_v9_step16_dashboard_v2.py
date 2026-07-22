#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 16 — Dashboard v2 with Paper Ledger + Learning

Purpose:
Upgrade the Step 11 dashboard:
- Add Paper Ledger
- Add Learning Attribution
- Add Paper Trade Helper usage instructions
- Display CLOSED_PAPER count, average return, and win rate
- No broker connection, no automatic order submission, no market data downloads
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Streamlit is not installed. Run: pip install streamlit")


ROOT = Path.cwd()

FILES = {
    "PM Report": ROOT / "daily_pm_report.md",
    "Exposure Dashboard": ROOT / "exposure_dashboard.md",
    "Stress + Sizing": ROOT / "stress_position_sizing_report.md",
    "Execution Gate Review": ROOT / "execution_gate_review.csv",
    "Pre-trade Order Ticket": ROOT / "pre_trade_order_ticket.csv",
    "Trade Journal": ROOT / "canyon_trade_journal.csv",
    "Paper Ledger Summary": ROOT / "paper_ledger_summary.md",
    "Paper Portfolio Ledger": ROOT / "paper_portfolio_ledger.csv",
    "Learning Report": ROOT / "learning_attribution_report.md",
    "Learning Summary": ROOT / "learning_attribution_summary.csv",
    "Learning Weight Suggestions": ROOT / "learning_weight_suggestions.csv",
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
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def to_float(s):
    try:
        return float(str(s).replace("%", "").strip())
    except Exception:
        return None


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

    df_show = df.copy()
    for col in df_show.columns:
        cl = col.lower()
        if any(k in cl for k in ["weight", "pnl", "loss", "shock", "reduction", "exposure"]):
            def maybe_fmt(v):
                try:
                    return f"{float(v):.2%}"
                except Exception:
                    return v
            df_show[col] = df_show[col].apply(maybe_fmt)

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.download_button(
        label=f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
    )


def build_metrics():
    review = read_csv(FILES["Execution Gate Review"])
    warnings = read_csv(FILES["Exposure Warnings"])
    stress = read_csv(FILES["Scenario Stress Results"])
    ledger = read_csv(FILES["Paper Portfolio Ledger"])

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Gate candidates", 0 if review.empty else len(review))

    with c2:
        if not warnings.empty and "level" in warnings.columns:
            high = (warnings["level"].astype(str).str.upper() == "HIGH").sum()
            med = (warnings["level"].astype(str).str.upper() == "MEDIUM").sum()
            st.metric("Warnings", f"H{high}/M{med}")
        else:
            st.metric("Warnings", "N/A")

    with c3:
        if not stress.empty and "estimated_pnl" in stress.columns:
            vals = pd.to_numeric(stress["estimated_pnl"], errors="coerce")
            worst = vals.min()
            st.metric("Worst scenario", f"{worst:.2%}" if pd.notna(worst) else "N/A")
        else:
            st.metric("Worst scenario", "N/A")

    with c4:
        if not ledger.empty and "status" in ledger.columns:
            closed = ledger[ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"])]
            st.metric("Closed paper/real", len(closed))
        else:
            st.metric("Closed paper/real", 0)

    with c5:
        if not ledger.empty and {"status", "pnl_pct"}.issubset(ledger.columns):
            closed = ledger[ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"])].copy()
            pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
            st.metric("Paper avg PnL", f"{pnl.mean():.2%}" if len(pnl) else "N/A")
        else:
            st.metric("Paper avg PnL", "N/A")


def show_overview():
    build_metrics()

    st.divider()

    st.subheader("Current worst scenario ranking")
    stress = read_csv(FILES["Scenario Stress Results"])
    if not stress.empty and "estimated_pnl" in stress.columns:
        s = stress.copy()
        s["estimated_pnl_num"] = pd.to_numeric(s["estimated_pnl"], errors="coerce")
        s = s.sort_values("estimated_pnl_num")
        s = s.drop(columns=["estimated_pnl_num"])
        if "estimated_pnl" in s.columns:
            s["estimated_pnl"] = pd.to_numeric(s["estimated_pnl"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        st.dataframe(s, use_container_width=True, hide_index=True)
    else:
        st.info("No scenario stress results yet.")

    st.subheader("Paper closed trades")
    ledger = read_csv(FILES["Paper Portfolio Ledger"])
    if not ledger.empty and "status" in ledger.columns:
        closed = ledger[ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"])].copy()
        cols = [c for c in ["trade_id", "ticker", "sleeve", "entry_price", "exit_price", "pnl_pct", "holding_days", "notes"] if c in closed.columns]
        if not closed.empty:
            show = closed[cols].copy()
            if "pnl_pct" in show.columns:
                show["pnl_pct"] = pd.to_numeric(show["pnl_pct"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("No closed paper trades yet.")
    else:
        st.info("Paper ledger not found.")


def show_helper_panel():
    st.subheader("Paper Trade Helper Commands")
    st.markdown(
        """
These commands only modify the local simulated ledger — no order submission, no broker connection.

```bash
cd ~/Desktop/canyon_quant
source .venv/bin/activate

# View ledger
python3 -u canyon_final_v9_step15_paper_trade_helper.py list

# Simulate a buy
python3 -u canyon_final_v9_step15_paper_trade_helper.py enter TLT 90

# Simulate a sell
python3 -u canyon_final_v9_step15_paper_trade_helper.py close TLT 92

# Re-run learning attribution
python3 -u canyon_final_v9_step14_learning_attribution.py
```

Rules:
- Only use `PAPER_CANDIDATE` or `WATCHLIST` status to simulate a buy.
- Do not close a ticker that is already `CLOSED_PAPER`.
- The Learning Engine only starts giving weight-adjustment suggestions after at least 5 `CLOSED_PAPER` trades.
"""
    )


def main():
    st.set_page_config(
        page_title="Canyon v9 Dashboard v2",
        page_icon="🏔",
        layout="wide",
    )

    st.title("🏔 Canyon v9 Dashboard v2")
    st.caption("Local-only PM dashboard with paper ledger and learning attribution. No broker connection. No live order.")

    with st.sidebar:
        st.header("File Status")
        for name, path in FILES.items():
            st.write(f"**{name}:** {'FOUND' if path.exists() else 'MISSING'}")
        st.divider()
        st.warning("ALLOW / REVIEW / PAPER are not live orders. This dashboard is a research cockpit only.")

    tabs = st.tabs([
        "Overview",
        "PM Report",
        "Exposure",
        "Stress + Sizing",
        "Execution Gate",
        "Paper Ledger",
        "Learning",
        "Helper",
    ])

    with tabs[0]:
        show_overview()

    with tabs[1]:
        st.markdown(read_text(FILES["PM Report"]))

    with tabs[2]:
        st.markdown(read_text(FILES["Exposure Dashboard"]))
        st.divider()
        show_csv("Exposure Table", FILES["Exposure Table"])
        show_csv("Exposure Warnings", FILES["Exposure Warnings"])

    with tabs[3]:
        st.markdown(read_text(FILES["Stress + Sizing"]))
        st.divider()
        show_csv("Scenario Stress Results", FILES["Scenario Stress Results"])
        show_csv("Position Sizing Recommendations", FILES["Sizing Recommendations"])

    with tabs[4]:
        show_csv("Execution Gate Review", FILES["Execution Gate Review"])
        show_csv("Pre-trade Order Ticket", FILES["Pre-trade Order Ticket"])

    with tabs[5]:
        st.markdown(read_text(FILES["Paper Ledger Summary"]))
        st.divider()
        show_csv("Paper Portfolio Ledger", FILES["Paper Portfolio Ledger"])

    with tabs[6]:
        st.markdown(read_text(FILES["Learning Report"]))
        st.divider()
        show_csv("Learning Attribution Summary", FILES["Learning Summary"])
        show_csv("Learning Weight Suggestions", FILES["Learning Weight Suggestions"])

    with tabs[7]:
        show_helper_panel()


if __name__ == "__main__":
    main()
