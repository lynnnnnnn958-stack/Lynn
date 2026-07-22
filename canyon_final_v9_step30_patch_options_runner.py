#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 30 — Patch Options Runner to Include Step 29 Diagnostics/Fix

Purpose:
The Options/Gamma table in the v5 dashboard could appear empty.
The cause is that Step 23 sometimes filters too strictly; when Yahoo/yfinance OI/gamma fields are incomplete,
gamma_squeeze_candidates.csv becomes empty.

This patch updates Step 25:
Step 23B fetches the options chain
→ Step 23 raw Gamma Layer
→ Step 29 diagnostics/fix for empty table
→ Step 24 Option Kill Zone
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
target = ROOT / "canyon_final_v9_step25_options_daily_runner.py"

if not target.exists():
    raise SystemExit("canyon_final_v9_step25_options_daily_runner.py not found. Make sure you are running from ~/Desktop/canyon_quant.")

backup = ROOT / f"canyon_final_v9_step25_options_daily_runner.py.bak_step30_{datetime.now():%Y%m%d_%H%M%S}"
backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

patched = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 25 — Options Daily Runner, patched by Step 30

Runs:
1. Step 23B yfinance options fetcher
2. Step 23 options gamma layer
3. Step 29 gamma diagnostics/fix, prevents empty dashboard table when possible
4. Step 24 option kill-zone layer

Local research only. No broker, no live order.
"""

from pathlib import Path
from datetime import datetime
import subprocess
import sys

ROOT = Path.cwd()
LOG = ROOT / "options_daily_runner_log.md"

SCRIPTS = [
    "canyon_final_v9_step23b_yfinance_options_fetcher.py",
    "canyon_final_v9_step23_options_gamma_layer.py",
    "canyon_final_v9_step29_fix_empty_options_gamma.py",
    "canyon_final_v9_step24_option_kill_zone.py",
]

OUTPUTS = [
    "yfinance_options_fetch_report.md",
    "options_chain_snapshot.csv",
    "options_gamma_report.md",
    "gamma_squeeze_candidates.csv",
    "options_gamma_diagnostics.md",
    "option_kill_zone_report.md",
    "option_kill_zone_risk.csv",
]

def run(script):
    print("\n" + "=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    p = subprocess.run([sys.executable, "-u", script], cwd=ROOT, text=True, capture_output=True)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print(out)
    return p.returncode, out

def main():
    missing = [s for s in SCRIPTS if not (ROOT / s).exists()]
    if missing:
        print("Missing scripts:")
        for m in missing:
            print("  -", m)
        print("\nIf Step 29 is missing, download/move canyon_final_v9_step29_fix_empty_options_gamma.py first.")
        raise SystemExit(1)

    md = [
        "# Canyon v9 Step 25 — Options Daily Runner Log",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Patched by Step 30: includes Step 29 diagnostics/fix.",
        "",
    ]

    ok = True
    for s in SCRIPTS:
        code, out = run(s)
        md += [f"## {s}", "", f"Exit code: `{code}`", "", "```text", out[-6000:], "```", ""]
        if code != 0:
            ok = False
            break

    md += ["## Outputs", ""]
    for o in OUTPUTS:
        md.append(f"- {'FOUND' if (ROOT / o).exists() else 'MISSING'} `{o}`")

    LOG.write_text("\n".join(md), encoding="utf-8")

    print("\nOptions runner:", "OK" if ok else "FAILED")
    print("Open:")
    print("  open options_daily_runner_log.md")
    print("  open options_gamma_diagnostics.md")
    print("  open options_gamma_report.md")
    print("  open option_kill_zone_report.md")

    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
'''

target.write_text(patched, encoding="utf-8")

print("OK, Step 25 updated: will automatically include Step 29 going forward.")
print(f"Original Step 25 backed up to: {backup}")
print("")
print("Next, run:")
print("python3 -u canyon_final_v9_step25_options_daily_runner.py")
print("streamlit run canyon_final_v9_step26_dashboard_v5_options.py")
