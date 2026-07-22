#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 19 — System Health Check

Purpose:
One-command system health check for Canyon — verifies:
- Key scripts exist
- Key output files exist
- Dashboard v3 exists
- Step 12 still points to the correct dashboard
- Paper ledger has CLOSED_PAPER entries
- Learning module has sufficient samples
- Exposure/stress test shows acceptable risk
- Execution gate has a real order draft

No data download, no order submission, no portfolio changes — diagnostic only.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path.cwd()

REQUIRED_SCRIPTS = [
    "canyon_final_v9_step3_realdata_tactical_fixed.py",
    "canyon_final_v9_step4_evidence_guard.py",
    "canyon_final_v9_step5_sec_event_layer.py",
    "canyon_final_v9_step6_journal_learning.py",
    "canyon_final_v9_step7_execution_gate.py",
    "canyon_final_v9_step8_pm_report.py",
    "canyon_final_v9_step9_exposure_dashboard.py",
    "canyon_final_v9_step10_stress_position_sizing.py",
    "canyon_final_v9_step12_daily_runner.py",
    "canyon_final_v9_step14_learning_attribution.py",
    "canyon_final_v9_step15_paper_trade_helper.py",
    "canyon_final_v9_step17_dashboard_v3.py",
]

REQUIRED_OUTPUTS = [
    "daily_pm_report.md",
    "execution_gate_review.csv",
    "pre_trade_order_ticket.csv",
    "exposure_dashboard.md",
    "exposure_dashboard.csv",
    "exposure_warnings.csv",
    "stress_position_sizing_report.md",
    "scenario_stress_results.csv",
    "position_sizing_recommendations.csv",
    "paper_portfolio_ledger.csv",
    "paper_ledger_summary.md",
    "learning_attribution_report.md",
    "learning_attribution_summary.csv",
    "learning_weight_suggestions.csv",
]

OUT_MD = ROOT / "system_health_check.md"
OUT_CSV = ROOT / "system_health_check.csv"


def exists(name: str) -> bool:
    return (ROOT / name).exists()


def read_csv(name: str) -> pd.DataFrame:
    p = ROOT / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def read_text(name: str) -> str:
    p = ROOT / name
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def status_level(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def build_checks():
    checks = []

    for s in REQUIRED_SCRIPTS:
        checks.append({
            "category": "script",
            "item": s,
            "status": status_level(exists(s)),
            "detail": "required script",
            "action": "" if exists(s) else "Download/copy this script into canyon_quant.",
        })

    for o in REQUIRED_OUTPUTS:
        checks.append({
            "category": "output",
            "item": o,
            "status": status_level(exists(o)),
            "detail": "required output/report",
            "action": "" if exists(o) else "Run Step 12 or related step again.",
        })

    if exists("canyon_final_v9_step17_dashboard_v3.py"):
        checks.append({
            "category": "dashboard",
            "item": "dashboard version",
            "status": "OK",
            "detail": "Dashboard v3 exists: Canyon v9 PM Cockpit.",
            "action": "Launch with: streamlit run canyon_final_v9_step17_dashboard_v3.py",
        })
    else:
        checks.append({
            "category": "dashboard",
            "item": "dashboard version",
            "status": "MISSING",
            "detail": "Dashboard v3 script not found.",
            "action": "Download Step 17.",
        })

    runner = read_text("canyon_final_v9_step12_daily_runner.py")
    if "step11_streamlit_dashboard" in runner and "step17_dashboard_v3" not in runner:
        checks.append({
            "category": "runner",
            "item": "Step 12 dashboard hint",
            "status": "WARN",
            "detail": "Step 12 still prints old Step 11 dashboard command.",
            "action": "Use Step 17 manually for now. Later update Step 12 dashboard command.",
        })
    else:
        checks.append({
            "category": "runner",
            "item": "Step 12 dashboard hint",
            "status": "OK",
            "detail": "Runner dashboard command appears current or not checked.",
            "action": "",
        })

    ledger = read_csv("paper_portfolio_ledger.csv")
    closed_count = 0

    if ledger.empty or "status" not in ledger.columns:
        checks.append({
            "category": "paper",
            "item": "paper ledger",
            "status": "WARN",
            "detail": "No readable paper ledger.",
            "action": "Run Step 13 / Step 15 flow.",
        })
    else:
        status = ledger["status"].astype(str).str.upper()
        closed_count = int(status.isin(["CLOSED_PAPER", "CLOSED_REAL"]).sum())
        openp = int(status.isin(["OPEN_PAPER", "OPEN_REAL"]).sum())
        watch = int(status.eq("WATCHLIST").sum())
        cand = int(status.eq("PAPER_CANDIDATE").sum())

        checks.append({
            "category": "paper",
            "item": "closed paper/real trades",
            "status": "OK" if closed_count >= 1 else "WARN",
            "detail": f"closed={closed_count}, open={openp}, watchlist={watch}, paper_candidate={cand}",
            "action": "Need at least 5 closed trades before learning can adjust weights." if closed_count < 5 else "Learning has enough initial sample for conservative review.",
        })

    checks.append({
        "category": "learning",
        "item": "learning sample size",
        "status": "WARN" if closed_count < 5 else "OK",
        "detail": f"closed trades used for learning = {closed_count}",
        "action": "Record only; no automatic adjustment." if closed_count < 5 else "Review learning_weight_suggestions.csv.",
    })

    warnings = read_csv("exposure_warnings.csv")
    high = med = 0
    if not warnings.empty and "level" in warnings.columns:
        levels = warnings["level"].astype(str).str.upper()
        high = int((levels == "HIGH").sum())
        med = int((levels == "MEDIUM").sum())
    checks.append({
        "category": "risk",
        "item": "exposure warnings",
        "status": "WARN" if high > 0 else ("CHECK" if med > 0 else "OK"),
        "detail": f"HIGH={high}, MEDIUM={med}",
        "action": "Do not convert candidates to live trades while high warnings exist." if high > 0 else "Review medium warnings before paper/live decisions.",
    })

    stress = read_csv("scenario_stress_results.csv")
    worst = None
    worst_name = "N/A"
    if not stress.empty and "estimated_pnl" in stress.columns:
        s = stress.copy()
        s["estimated_pnl_num"] = pd.to_numeric(s["estimated_pnl"], errors="coerce")
        s = s.dropna(subset=["estimated_pnl_num"]).sort_values("estimated_pnl_num")
        if not s.empty:
            worst = float(s.iloc[0]["estimated_pnl_num"])
            worst_name = str(s.iloc[0].get("scenario", "N/A"))
    if worst is None:
        checks.append({
            "category": "stress",
            "item": "worst scenario",
            "status": "WARN",
            "detail": "No readable stress result.",
            "action": "Run Step 10 or Step 12.",
        })
    else:
        checks.append({
            "category": "stress",
            "item": "worst scenario",
            "status": "WARN" if worst <= -0.02 else "OK",
            "detail": f"{worst_name}: {pct(worst)}",
            "action": "Risk light should be RED; reduce concentration before trading." if worst <= -0.02 else "Stress appears contained; still check execution gate.",
        })

    order = read_csv("pre_trade_order_ticket.csv")
    if order.empty:
        checks.append({
            "category": "execution",
            "item": "pre-trade order ticket",
            "status": "OK",
            "detail": "No order drafts generated.",
            "action": "Expected if manual checks are incomplete. No broker order sent.",
        })
    else:
        checks.append({
            "category": "execution",
            "item": "pre-trade order ticket",
            "status": "CHECK",
            "detail": f"{len(order)} order draft rows exist.",
            "action": "Verify PAPER/LIVE intent and manual checks before any action.",
        })

    return pd.DataFrame(checks)


def build_report(checks: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 19 — System Health Check")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Summary")
    md.append("")

    counts = checks["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    md.append(counts.to_markdown(index=False))
    md.append("")

    if (checks["status"] == "MISSING").any():
        md.append("**System status: NOT READY** — missing required scripts or outputs.")
    elif (checks["status"] == "WARN").any():
        md.append("**System status: USABLE BUT CAUTIOUS** — files exist, but risk/sample warnings remain.")
    else:
        md.append("**System status: OK** — no major health issue detected.")
    md.append("")

    md.append("## Checks")
    md.append("")
    md.append(checks.to_markdown(index=False))
    md.append("")

    md.append("## Recommended Next Actions")
    md.append("")
    if (checks["status"] == "MISSING").any():
        md.append("1. Fix missing files first.")
    else:
        md.append("1. Launch dashboard v3: `streamlit run canyon_final_v9_step17_dashboard_v3.py`.")
        md.append("2. Review Risk page before any paper trade.")
        md.append("3. Keep collecting CLOSED_PAPER trades until at least 5 samples.")
        md.append("4. Do not use LIVE orders while Risk Light is RED.")
        md.append("5. Later update Step 12 so it points to Step 17 dashboard instead of old Step 11.")
    md.append("")

    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 19")
    print("System Health Check")
    print("=" * 88)

    checks = build_checks()
    checks.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_report(checks), encoding="utf-8")

    counts = checks["status"].value_counts().to_dict()
    print("Status counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    missing = checks[checks["status"] == "MISSING"]
    warns = checks[checks["status"] == "WARN"]

    if not missing.empty:
        print("\nMissing items:")
        for _, r in missing.iterrows():
            print(f"  - {r['category']} / {r['item']}")
    if not warns.empty:
        print("\nWarnings:")
        for _, r in warns.iterrows():
            print(f"  - {r['category']} / {r['item']}: {r['detail']}")

    print("\nFiles generated:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_CSV}")

    print("\nNext: open system_health_check.md")


if __name__ == "__main__":
    main()
