#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 35 — Options Decision Daily Runner

Runs only the options decision stack:
1. Clean Step 25 options runner
2. Step 32 decision matrix

Then prints the v6 dashboard command.

No broker. No live order.
"""

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "options_decision_daily_runner_log.md"

STEPS = [
    ("Options runner", "canyon_final_v9_step25_options_daily_runner.py"),
    ("Options decision matrix", "canyon_final_v9_step32_options_decision_matrix.py"),
]

def run(label, script):
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    if not (ROOT / script).exists():
        out = f"MISSING SCRIPT: {script}"
        print(out)
        return 1, out
    p = subprocess.run([sys.executable, "-u", script], cwd=ROOT, text=True, capture_output=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print(out)
    return p.returncode, out

def main():
    md = [
        "# Canyon v9 Step 35 — Options Decision Daily Runner Log",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    ok = True
    for label, script in STEPS:
        code, out = run(label, script)
        md.extend([
            f"## {label}",
            "",
            f"Script: `{script}`",
            f"Exit code: `{code}`",
            "",
            "```text",
            out[-6000:],
            "```",
            "",
        ])
        if code != 0:
            ok = False
            break

    LOG.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"Options Decision Runner: {'OK' if ok else 'FAILED'}")
    print("=" * 88)
    print("Open:")
    print("  open options_decision_daily_runner_log.md")
    print("  open options_decision_matrix.md")
    print()
    print("Dashboard:")
    print("  streamlit run canyon_final_v9_step34_dashboard_v6_decision.py")

    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
