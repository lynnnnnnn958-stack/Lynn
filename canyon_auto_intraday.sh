#!/bin/zsh
# Canyon 日内自动刷新(每30分钟)+ 数据过期自愈。由 launchd 触发。
cd /Users/renjingru/canyon_quant || exit 1
mkdir -p logs
# 系统 framework Python (/Library, 不被 iCloud 卸载); venv 在 iCloud 桌面会被清空导致卡死
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
[ -x "$PY" ] || PY=/Users/renjingru/canyon_quant/.venv/bin/python

# ── 过期自愈: 若核心数据 >20小时(1200分)没更新, 触发全量 run_daily(哪怕早6点漏跑也能补) ──
if [ -z "$(find event_candidates.csv -mmin -1200 2>/dev/null)" ]; then
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 数据过期, 自愈触发全量 run_daily =====" >> logs/auto_daily.log
  $PY run_daily.py >> logs/auto_daily.log 2>&1
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 自愈完成 (exit $?) =====" >> logs/auto_daily.log
fi

# ── 日内感知刷新(轻量)+ 刷新 dashboard ──
$PY canyon_intraday.py >> logs/auto_intraday.log 2>&1
$PY update_research_html.py >> logs/auto_intraday.log 2>&1
