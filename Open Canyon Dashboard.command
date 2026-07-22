#!/bin/bash
# Double-click this file to open Canyon Quant Dashboard
cd "$(dirname "$0")"

# Start the server in background if not already running
if ! lsof -i :8888 -t > /dev/null 2>&1; then
    echo "Starting Canyon dashboard server..."
    python3 serve_canyon.py &
    sleep 2
fi

# Open browser
open http://localhost:8888
echo "Dashboard opened at http://localhost:8888"
echo "Data refreshes automatically. You can close this window."
