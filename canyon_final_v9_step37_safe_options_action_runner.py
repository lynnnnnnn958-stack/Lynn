#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 37 — Safe Options Action Runner

Runs the stable action stack only:
1. Step 25: options chain + gamma + kill-zone inputs
2. Step 32: options decision matrix
3. Step 36: tonight action plan

No broker connection. No live orders.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "safe_options_action_runner_log.md"

STEPS = [
    ("Options daily runner", "canyon_final_v9_step25_options_daily_runner.py"),
    ("Options decision matrix", "canyon_final_v9_step32_options_decision_matrix.py"),
    ("Tonight action plan", "canyon_final_v9_step36_tonight_action_plan.py"),
]

OUTPUTS = [
    "options_gamma_report.md",
    "option_kill_zone_report.md",
    "options_decision_matrix.md",
    "tonight_action_plan.md",
    "watch_triggers.csv",
]


def run(label: str, script: str) -> tuple[int, str]:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)

    path = ROOT / script
    if not path.exists():
        msg = f"MISSING SCRIPT: {script}"
        print(msg)
        return 1, msg

    proc = subprocess.run(
        [sys.executable, "-u", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    out = proc.stdout or ""
    if proc.stderr:
        out += "\n" + proc.stderr

    print(out)
    return proc.returncode, out


def main() -> None:
    log = [
        "# Canyon v9 Step 37 — Safe Options Action Runner Log",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    ok = True
    for label, script in STEPS:
        code, out = run(label, script)
        log.extend([
            f"## {label}",
            "",
            f"Script: `{script}`",
            f"Exit code: `{code}`",
            "",
            "```text",
            out[-7000:],
            "```",
            "",
        ])
        if code != 0:
            ok = False
            break

    log.extend(["## Output files", ""])
    for name in OUTPUTS:
        log.append(f"- {'FOUND' if (ROOT / name).exists() else 'MISSING'} `{name}`")

    LOG.write_text("\n".join(log), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"Safe Options Action Runner: {'OK' if ok else 'FAILED'}")
    print("=" * 88)

    print("Open:")
    print("  open tonight_action_plan.md")
    print("  open options_decision_matrix.md")
    print("  open safe_options_action_runner_log.md")
    print()
    print("Dashboard:")
    print("  streamlit run canyon_final_v9_step38_dashboard_v7_action.py")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
