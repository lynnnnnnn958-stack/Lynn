#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 28 — Fix Step 7 Execution Gate

Error you encountered:
    IndexError: single positional indexer is out-of-bounds

This typically occurs when Step 7 forcefully calls df.iloc[0] even when a CSV has no candidate rows.
This patch will:
1. Back up the original canyon_final_v9_step7_execution_gate.py
2. Write a more robust Step 7
3. Generate empty execution_gate_review.csv and pre_trade_order_ticket.csv even when there are no candidates
4. No live orders, no broker connection
"""

from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
target = ROOT / "canyon_final_v9_step7_execution_gate.py"

if not target.exists():
    raise SystemExit("canyon_final_v9_step7_execution_gate.py not found. Make sure you are running from ~/Desktop/canyon_quant.")

backup = ROOT / f"canyon_final_v9_step7_execution_gate.py.bak_step28_{datetime.now():%Y%m%d_%H%M%S}"
backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

safe_step7 = r