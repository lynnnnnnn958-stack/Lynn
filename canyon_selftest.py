#!/usr/bin/env python3
"""
canyon_selftest.py — 全系统部门自检
====================================
逐个跑每个模块(子进程+超时), 抓: 能否跑通 / 是否卡死 / 报错 / 输出文件是否刷新。
一个卡死不影响其他(独立子进程+timeout)。只读结果, 不改数据。
"""
import subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable

# (脚本, 超时秒, 期望输出文件, 是否需要网络)
MODULES = [
    ("canyon_macro_intel.py",       90,  "macro_intel_scorecard.json", False),
    ("canyon_sector_rotation.py",   90,  "sector_rotation.csv",        False),
    ("canyon_event_detect.py",      90,  "auto_event_candidates.csv",  False),
    ("canyon_lifecycle.py",         90,  "lifecycle_style.csv",        False),
    ("canyon_build_pool.py",        90,  "event_pool.csv",             False),
    ("canyon_event_system.py",      90,  "event_candidates.csv",       False),
    ("canyon_pools.py",             90,  "functional_pools.csv",       False),
    ("canyon_position_sizing.py",   90,  "concentrated_portfolio.csv", False),
    ("canyon_execution_costs.py",   90,  "execution_cost_plan.csv",    False),
    ("canyon_position_manager.py",  90,  "position_actions.csv",       False),
    ("canyon_review.py",            90,  "review_report.json",         False),
    ("canyon_event_validate.py",    150, "event_validation.json",      False),
    ("canyon_sector_etf.py",        120, "sector_etf_indicators.csv",  True),
    ("canyon_intraday.py",          180, "intraday_signals.json",      True),
    ("canyon_cftc_cot.py",          150, "cot_positioning.csv",        True),
    ("canyon_edgar_events.py",      420, "edgar_events.csv",           True),
    ("canyon_edgar_backtest.py",    300, "edgar_event_study.json",     True),  # 用缓存
    ("canyon_qqq_backtest.py",      180, "qqq_backtest.json",          False),
    ("canyon_qqq_leverage.py",      180, "qqq_leverage_grid.json",     False),
]


def run_one(script, timeout, outfile, needs_net):
    path = ROOT / script
    if not path.exists():
        return {"script": script, "status": "缺文件", "detail": ""}
    before = os.path.getmtime(ROOT / outfile) if (ROOT / outfile).exists() else 0
    t0 = time.time()
    try:
        r = subprocess.run([PY, str(path)], capture_output=True, text=True,
                           timeout=timeout, cwd=str(ROOT))
        dt = time.time() - t0
    except subprocess.TimeoutExpired:
        return {"script": script, "status": "🔴卡死/超时", "detail": f">{timeout}s"}
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines() or ["?"])[-1][:90]
        return {"script": script, "status": "🔴报错", "detail": f"rc={r.returncode} {err}"}
    after = os.path.getmtime(ROOT / outfile) if (ROOT / outfile).exists() else 0
    fresh = after > before
    exists = (ROOT / outfile).exists()
    if not exists:
        return {"script": script, "status": "🟡跑通但无输出", "detail": f"缺 {outfile} ({dt:.0f}s)"}
    return {"script": script, "status": "✅", "detail": f"{dt:.0f}s{'·输出已刷新' if fresh else '·输出未变'}"}


def main():
    only_net = "--net" in sys.argv
    only_core = "--core" in sys.argv
    print("=" * 70)
    print("Canyon 全系统部门自检")
    print("=" * 70)
    for script, timeout, outfile, needs_net in MODULES:
        if only_core and needs_net:
            continue
        if only_net and not needs_net:
            continue
        res = run_one(script, timeout, outfile, needs_net)
        print(f"  {res['status']:<12} {res['script']:<32} {res['detail']}", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
