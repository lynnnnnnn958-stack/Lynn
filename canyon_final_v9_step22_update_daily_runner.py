#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 22 — Update Daily Runner Dashboard Hint

This only patches the Step 12 daily runner so the printed / launched dashboard
uses Step 21 Dashboard v4 instead of old Step 11.
"""

from pathlib import Path

ROOT = Path.cwd()
runner = ROOT / "canyon_final_v9_step12_daily_runner.py"

if not runner.exists():
    raise SystemExit("Missing canyon_final_v9_step12_daily_runner.py")

text = runner.read_text(encoding="utf-8")

text = text.replace(
    'DASHBOARD_SCRIPT = "canyon_final_v9_step11_streamlit_dashboard.py"',
    'DASHBOARD_SCRIPT = "canyon_final_v9_step21_dashboard_v4.py"',
)

text = text.replace(
    "streamlit run canyon_final_v9_step11_streamlit_dashboard.py",
    "streamlit run canyon_final_v9_step21_dashboard_v4.py",
)

text = text.replace(
    "Next Step 11: create a lightweight dashboard so you can view PM report, exposures, stress tests, and journal in one page.",
    "Dashboard: use canyon_final_v9_step21_dashboard_v4.py for the current PM Cockpit.",
)

backup = runner.with_suffix(".py.bak_step22")
backup.write_text(runner.read_text(encoding="utf-8"), encoding="utf-8")
runner.write_text(text, encoding="utf-8")

print("OK, Step 12 runner dashboard hint updated to Step 21 Dashboard v4.")
print(f"Backup saved: {backup}")
print("Run: python3 -u canyon_final_v9_step12_daily_runner.py --dashboard")
