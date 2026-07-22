#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 27 — Full Daily Pipeline
Runs core daily, health, pre-trade, options layer, learning.
"""

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "full_daily_pipeline_log.md"

STEPS = [
    ("Core daily", "canyon_final_v9_step12_daily_runner.py"),
    ("Health", "canyon_final_v9_step19_health_check.py"),
    ("Pre-trade", "canyon_final_v9_step20_pre_trade_checklist.py"),
    ("Options", "canyon_final_v9_step25_options_daily_runner.py"),
    ("Learning", "canyon_final_v9_step14_learning_attribution.py"),
]

def run(label, script):
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    if not (ROOT / script).exists():
        out = f"MISSING SCRIPT: {script}"
        print(out)
        return 1, out
    p = subprocess.run([sys.executable, "-u", script], cwd=ROOT, text=True, capture_output=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print(out)
    return p.returncode, out

def main():
    md = ["# Canyon v9 Step 27 — Full Daily Pipeline Log", "", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    ok = True
    for label, script in STEPS:
        code, out = run(label, script)
        md += [f"## {label}", "", f"Script: `{script}`", f"Exit code: `{code}`", "", "```text", out[-5000:], "```", ""]
        if code != 0:
            ok = False
            break
    LOG.write_text("\n".join(md), encoding="utf-8")
    print("\nFULL DAILY PIPELINE:", "OK" if ok else "FAILED")
    print("Open:")
    print("  open full_daily_pipeline_log.md")
    print("Dashboard:")
    print("  streamlit run canyon_final_v9_step26_dashboard_v5_options.py")
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
