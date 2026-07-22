#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 53 — Build Missing Layers Runner

Runs:
- Step 47 L1 Data Integrity
- Step 48 L2 Macro & Regime
- Step 49 L3 Sector Rotation
- Step 50 L4 Fundamental
- Step 51 L5 Event/News/SEC/Insider
- Step 52 L6 Technical/Microstructure
- Step 44 Architecture Registry
- Step 45 Master 10-Layer Decision

No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "build_missing_layers_runner_log.md"

STEPS = [
    ("L1 Data Integrity", "canyon_final_v9_step47_l1_data_integrity.py"),
    ("L2 Macro Regime", "canyon_final_v9_step48_l2_macro_regime.py"),
    ("L3 Sector Rotation", "canyon_final_v9_step49_l3_sector_rotation.py"),
    ("L4 Fundamental", "canyon_final_v9_step50_l4_fundamental_quality.py"),
    ("L5 Event News SEC Insider", "canyon_final_v9_step51_l5_event_news_sec_insider.py"),
    ("L6 Technical Microstructure", "canyon_final_v9_step52_l6_technical_microstructure.py"),
    ("10-Layer Architecture Registry", "canyon_final_v9_step44_layer_architecture_registry.py"),
    ("Master 10-Layer Decision", "canyon_final_v9_step45_master_10_layer_decision.py"),
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
        "# Canyon v9 Step 53 — Build Missing Layers Runner Log",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    ok = True
    for label, script in STEPS:
        code, out = run(label, script)
        md.extend([f"## {label}", "", f"Script: `{script}`", f"Exit code: `{code}`", "", "```text", out[-7000:], "```", ""])
        if code != 0:
            ok = False
            break

    LOG.write_text("\n".join(md), encoding="utf-8")
    print()
    print("=" * 88)
    print(f"Build Missing Layers Runner: {'OK' if ok else 'FAILED'}")
    print("=" * 88)
    print("Open:")
    print("  open build_missing_layers_runner_log.md")
    print("  open canyon_10_layer_architecture.md")
    print("  open master_10_layer_decision_report.md")
    print()
    print("Dashboard:")
    print("  streamlit run canyon_final_v9_step46_10_layer_dashboard.py")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
