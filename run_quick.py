#!/usr/bin/env python3
"""
Canyon — Quick Refresh Pipeline
=================================
Runs the fast subset of pipeline steps: price signals + social sentiment
+ push alerts + HTML rebuild.  Target: < 5 minutes total.

Useful for:
  • Intraday signal refresh (without waiting for full 30-90 min pipeline)
  • iPhone one-tap update via iOS Shortcuts pointing to this script
  • Testing dashboard changes without full run

Usage:
  .venv/bin/python run_quick.py
  make quick           (after: echo 'quick:\n\t.venv/bin/python run_quick.py' >> Makefile)
"""

from __future__ import annotations

import json
import os as _os
from pathlib import Path as _P

# Auto-load .env from project root (never commit .env to git)
_env_file = _P(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() and _v.strip() and _k.strip() not in _os.environ:
                _os.environ[_k.strip()] = _v.strip()
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = ROOT / ".venv" / "bin" / "python"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Only the fast steps — all should finish in < 2 min each
QUICK_STEPS = [
    # (label, script, critical, timeout_sec)
    ("Price & signals",   "step_daily_price_signals.py",  True,  120),
    ("Social sentiment",  "step_social_sentiment.py",     False,  120),
    ("Push alerts",       "step_push_alerts.py",          False,  30),
    ("HTML rebuild",      "update_research_html.py",      True,   90),
]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run_step(label: str, script: str, critical: bool, timeout: int) -> bool:
    path = ROOT / script
    if not path.exists():
        if critical:
            _log(f"{RED}MISS  {label} — {script} not found{RESET}")
            return False
        _log(f"{YELLOW}SKIP  {label} — optional, not found{RESET}")
        return True

    _log(f"{CYAN}START {BOLD}{label}{RESET}")
    t0 = time.time()
    import os as _os
    env = _os.environ.copy()
    env["RUN_DAILY_ACTIVE"] = "1"
    try:
        result = subprocess.run(
            [str(PYTHON), str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            _log(f"{GREEN}  OK  {label}  ({elapsed:.1f}s){RESET}")
            for ln in [l.strip() for l in result.stdout.strip().splitlines() if l.strip()][-2:]:
                _log(f"       {ln}")
            return True
        else:
            _log(f"{RED}  ERR {label}  rc={result.returncode}  ({elapsed:.1f}s){RESET}")
            for ln in (result.stderr or result.stdout or "").strip().splitlines()[-3:]:
                _log(f"       {ln.strip()}")
            return False
    except subprocess.TimeoutExpired:
        _log(f"{RED}  TIMEOUT {label} (>{timeout}s){RESET}")
        return False
    except Exception as e:
        _log(f"{RED}  EXCEPTION {label}: {e}{RESET}")
        return False


def main():
    if not PYTHON.exists():
        print(f"{RED}ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt{RESET}")
        sys.exit(1)

    t_start = time.time()
    _log(f"{BOLD}═══  Canyon Quick Refresh  ═══{RESET}")
    _log(f"  Steps: {len(QUICK_STEPS)}  (target < 5 min)")
    _log("")

    results = []
    for label, script, critical, timeout in QUICK_STEPS:
        ok = run_step(label, script, critical, timeout)
        results.append((label, ok))
        _log("")

    elapsed = time.time() - t_start
    passed  = sum(1 for _, ok in results if ok)
    failed  = len(results) - passed

    _log(f"{BOLD}═══  Done in {elapsed:.0f}s — {passed}/{len(results)} steps OK  ═══{RESET}")
    if failed:
        for label, ok in results:
            if not ok:
                _log(f"  {RED}✗  {label}{RESET}")
    _log(f"  Dashboard: http://localhost:8513")


if __name__ == "__main__":
    main()
