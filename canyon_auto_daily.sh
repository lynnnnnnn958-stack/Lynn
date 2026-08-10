#!/bin/zsh
# Canyon 每日自动运行 — 跑全部引擎 + 刷新 dashboard。由 launchd 定时触发。
cd /Users/renjingru/Desktop/model/canyon_quant || exit 1
mkdir -p logs
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 开始每日自动运行 =====" >> logs/auto_daily.log
# 系统 framework Python (/Library, 不被 iCloud 卸载); venv 在 iCloud 桌面会被清空导致卡死
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
[ -x "$PY" ] || PY=/Users/renjingru/Desktop/model/canyon_quant/.venv/bin/python
"$PY" run_daily.py >> logs/auto_daily.log 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 完成 (exit $?) =====" >> logs/auto_daily.log
