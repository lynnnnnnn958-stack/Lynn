#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 33 — Patch Dashboard v5 to Show Options Decision Matrix

Directly modifies dashboard v5:
- Adds the Options Decision page
- Adds Step 32 command in the Helper
"""

from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
dash = ROOT / "canyon_final_v9_step26_dashboard_v5_options.py"

if not dash.exists():
    raise SystemExit("canyon_final_v9_step26_dashboard_v5_options.py not found")

backup = dash.with_name(dash.name + f".bak_step33_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
backup.write_text(dash.read_text(encoding="utf-8"), encoding="utf-8")

text = dash.read_text(encoding="utf-8")

# Add files if possible.
if '"decision_csv"' not in text and '"kill_csv"' in text:
    text = text.replace(
        '"kill_csv": ROOT / "option_kill_zone_risk.csv",',
        '"kill_csv": ROOT / "option_kill_zone_risk.csv",\n    "decision_md": ROOT / "options_decision_matrix.md",\n    "decision_csv": ROOT / "options_decision_matrix.csv",'
    )

tab_func = r