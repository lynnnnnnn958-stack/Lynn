#!/usr/bin/env bash
set -e

cd ~/Desktop/canyon_quant
source .venv/bin/activate

echo "🏔 Starting Canyon v9 PM Cockpit..."
echo "Working folder: $(pwd)"
echo ""

streamlit run canyon_final_v9_step17_dashboard_v3.py
