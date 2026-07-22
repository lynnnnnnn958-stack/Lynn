#!/bin/bash
# Canyon v9 Dashboard 启动脚本
# 双击这个文件即可启动 Dashboard，然后在浏览器打开 http://localhost:8502
# Double-click to start, then open http://localhost:8502 in your browser

cd "$(dirname "$0")"

# 如果已经在运行，先停掉旧的
pkill -f "step86_dashboard_v3" 2>/dev/null
sleep 1

echo ""
echo "================================================"
echo "  Canyon v9 量化系统 Dashboard"
echo "================================================"
echo ""
echo "  正在启动..."
echo "  Starting dashboard..."
echo ""

# 启动 Streamlit (use framework Python for macOS GUI binding)
PYFW="/Library/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
if [ -f "$PYFW" ]; then
    "$PYFW" -m streamlit run canyon_final_v9_step86_dashboard_v3.py \
        --server.port 8502 \
        --server.headless false \
        --browser.gatherUsageStats false
else
    .venv/bin/python -m streamlit run canyon_final_v9_step86_dashboard_v3.py \
        --server.port 8502 \
        --server.headless false \
        --browser.gatherUsageStats false
fi

# 如果启动失败，保持窗口打开
echo ""
echo "Dashboard stopped. Press any key to close."
read -n 1
