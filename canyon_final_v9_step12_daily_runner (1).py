#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 12 — Daily One-Click Runner

Purpose:
Run Step 3 → Step 10 in one command to generate the full daily report,
trade review sheet, exposure panel, stress test, and position recommendations.
Step 11 Dashboard can be started standalone or via the --dashboard flag.

Run:
    python3 -u canyon_final_v9_step12_daily_runner.py

Optional:
    python3 -u canyon_final_v9_step12_daily_runner.py --dashboard

Principles:
- No automatic order submission
- No broker connection
- No bypassing manual review
- If any step fails, stop and prompt the user
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


ROOT = Path.cwd()
LOG_DIR = ROOT / "daily_runs"
LOG_DIR.mkdir(exist_ok=True)

STEPS = [
    ("Step 3",  "Real Data + Tactical",           "canyon_final_v9_step3_realdata_tactical_fixed.py", "run_v9_step3_fixed.txt"),
    ("Step 4",  "Evidence + Exposure Guard",      "canyon_final_v9_step4_evidence_guard.py",          "run_v9_step4.txt"),
    ("Step 5",  "SEC/Event Layer",                "canyon_final_v9_step5_sec_event_layer.py",         "run_v9_step5.txt"),
    ("Step 6",  "Trade Journal + Learning",       "canyon_final_v9_step6_journal_learning.py",        "run_v9_step6.txt"),
    ("Step 7",  "Broker / Execution Gate",        "canyon_final_v9_step7_execution_gate.py",          "run_v9_step7.txt"),
    ("Step 8",  "Daily PM Report",                "canyon_final_v9_step8_pm_report.py",               "run_v9_step8.txt"),
    ("Step 9",  "Exposure Dashboard",             "canyon_final_v9_step9_exposure_dashboard.py",      "run_v9_step9.txt"),
    ("Step 10", "Scenario Stress + Position Size", "canyon_final_v9_step10_stress_position_sizing.py", "run_v9_step10.txt"),
]

DASHBOARD_SCRIPT = "canyon_final_v9_step11_streamlit_dashboard.py"


def run_step(step_name: str, title: str, script: str, log_name: str) -> bool:
    script_path = ROOT / script
    log_path = ROOT / log_name
    archive_log_path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_name}"

    print("\n" + "=" * 88)
    print(f"{step_name}: {title}")
    print("=" * 88)

    if not script_path.exists():
        print(f"❌ Missing script: {script_path}")
        print("Action: download/copy this file into ~/Desktop/canyon_quant, then rerun Step 12.")
        return False

    cmd = [sys.executable, "-u", str(script_path)]

    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)

        rc = process.wait()

    try:
        archive_log_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    if rc != 0:
        print(f"\n❌ {step_name} failed with exit code {rc}.")
        print(f"Log file: {log_path}")
        return False

    print(f"\n✅ {step_name} completed.")
    print(f"Log file: {log_path}")
    return True


def file_status() -> None:
    important_outputs = [
        "daily_pm_report.md",
        "execution_gate_review.csv",
        "pre_trade_order_ticket.csv",
        "canyon_trade_journal.csv",
        "trade_update_template.csv",
        "exposure_dashboard.md",
        "exposure_dashboard.csv",
        "exposure_warnings.csv",
        "stress_position_sizing_report.md",
        "scenario_stress_results.csv",
        "position_sizing_recommendations.csv",
    ]

    print("\n" + "=" * 88)
    print("Output file status")
    print("=" * 88)

    for name in important_outputs:
        p = ROOT / name
        status = "FOUND" if p.exists() else "MISSING"
        print(f"{status:<8} {name}")


def launch_dashboard() -> None:
    dash = ROOT / DASHBOARD_SCRIPT
    if not dash.exists():
        print(f"\n⚠️ Dashboard script missing: {DASHBOARD_SCRIPT}")
        print("You can still open the markdown reports manually.")
        return

    print("\n" + "=" * 88)
    print("Launching Streamlit dashboard")
    print("=" * 88)
    print("If a browser does not open, copy the Local URL shown below.")
    print("Press Control+C in this Terminal to stop the dashboard.")
    subprocess.call(["streamlit", "run", str(dash)], cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after generating reports.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip Step 3–5 and only regenerate journal/gate/report/exposure/stress layers.",
    )
    args = parser.parse_args()

    print("\n🏔 CANYON v9 Step 12 — Daily One-Click Runner")
    print(f"Working folder: {ROOT}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRules:")
    print("  · No broker orders are sent.")
    print("  · WATCHLIST is not an order.")
    print("  · Execution Gate still requires manual checks.")
    print("  · If data download fails, the run stops rather than inventing data.")

    steps = STEPS
    if args.skip_data:
        steps = [s for s in STEPS if s[0] not in {"Step 3", "Step 4", "Step 5"}]

    missing = [script for _, _, script, _ in steps if not (ROOT / script).exists()]
    if missing:
        print("\n❌ Missing required scripts:")
        for m in missing:
            print(f"  - {m}")
        print("\nDownload/copy the missing files into ~/Desktop/canyon_quant, then rerun.")
        sys.exit(1)

    for step_name, title, script, log_name in steps:
        ok = run_step(step_name, title, script, log_name)
        if not ok:
            print("\nStopped. Fix the failed step first.")
            sys.exit(1)

    file_status()

    print("\n" + "=" * 88)
    print("Daily run completed")
    print("=" * 88)
    print("Open these files:")
    print("  open daily_pm_report.md")
    print("  open exposure_dashboard.md")
    print("  open stress_position_sizing_report.md")
    print("\nDashboard:")
    print(f"  streamlit run {DASHBOARD_SCRIPT}")

    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
