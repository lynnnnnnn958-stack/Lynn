#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 56 — Full 10-Layer Daily Runner v2

Runs:
1. Step 60 pre-run output vault
2. Step 37 safe options/action stack
3. Step 39 action cards
4. Step 53 missing layer builder
5. Step 57 v8 research bridge
6. Step 58 v8 L9 execution gate
7. Step 59 v8 advanced risk bridge
8. Step 54 master 10-layer decision v2
9. Step 61 data source health monitor
10. Step 60 post-run shrinkage audit

Then tells you to open Step 55 dashboard.

No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "full_10_layer_daily_runner_v2_log.md"

STEPS = [
    ("Pre-run output vault", "canyon_final_v9_step60_output_vault.py", ["--label", "pre-run"]),
    ("Safe options action runner", "canyon_final_v9_step37_safe_options_action_runner.py", []),
    ("Action cards", "canyon_final_v9_step39_action_cards.py", []),
    ("Build missing layers", "canyon_final_v9_step53_build_missing_layers_runner.py", []),
    ("V8 research bridge", "canyon_final_v9_step57_v8_research_bridge.py", []),
    ("V8 L9 execution gate", "canyon_final_v9_step58_v8_l9_execution_gate.py", []),
    ("V8 advanced risk bridge", "canyon_final_v9_step59_v8_advanced_risk_bridge.py", []),
    ("Master 10-layer v2", "canyon_final_v9_step54_master_10_layer_decision_v2.py", []),
    ("Data source health", "canyon_final_v9_step61_data_source_health.py", []),
    ("Post-run output vault audit", "canyon_final_v9_step60_output_vault.py", ["--label", "post-run"]),
]


def run(label, script, args=None):
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    if not (ROOT / script).exists():
        out = f"MISSING SCRIPT: {script}"
        print(out)
        return 1, out
    p = subprocess.run([sys.executable, "-u", script] + list(args or []), cwd=ROOT, text=True, capture_output=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print(out)
    return p.returncode, out


def main():
    md = [
        "# Canyon v9 Step 56 — Full 10-Layer Daily Runner v2 Log",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    ok = True
    for label, script, args in STEPS:
        code, out = run(label, script, args)
        md.extend([
            f"## {label}",
            "",
            f"Script: `{script}`",
            f"Exit code: `{code}`",
            "",
            "```text",
            out[-8000:],
            "```",
            "",
        ])
        if code != 0:
            ok = False
            break

    LOG.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"Full 10-Layer Daily Runner v2: {'OK' if ok else 'FAILED'}")
    print("=" * 88)
    print("Open:")
    print("  open full_10_layer_daily_runner_v2_log.md")
    print("  open master_10_layer_decision_report_v2.md")
    print()
    print("Dashboard:")
    print("  streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
