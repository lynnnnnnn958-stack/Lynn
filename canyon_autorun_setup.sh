#!/bin/bash
# Canyon v9 — macOS Auto-Run Setup
# Creates a launchd agent that runs the daily runner at 17:30 every weekday
# (NYSE closes 16:00 ET; 17:30 local gives 90min buffer for yfinance data)
#
# Usage:
#   bash canyon_autorun_setup.sh           # install
#   bash canyon_autorun_setup.sh --remove  # uninstall
#   bash canyon_autorun_setup.sh --status  # check status

PLIST_NAME="com.canyon.daily_runner"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=$(which python3)
LOG_OUT="$SCRIPT_DIR/autorun_stdout.log"
LOG_ERR="$SCRIPT_DIR/autorun_stderr.log"

# ── Remove ────────────────────────────────────────────────────────────────────
if [[ "$1" == "--remove" ]]; then
    launchctl unload "$PLIST_PATH" 2>/dev/null
    rm -f "$PLIST_PATH"
    echo "✅  Canyon autorun removed."
    exit 0
fi

# ── Status ────────────────────────────────────────────────────────────────────
if [[ "$1" == "--status" ]]; then
    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "✅  Canyon autorun is ACTIVE"
        launchctl list "$PLIST_NAME" 2>/dev/null
    else
        echo "❌  Canyon autorun is NOT installed"
    fi
    exit 0
fi

# ── Install ───────────────────────────────────────────────────────────────────
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCRIPT_DIR}/canyon_final_v9_step70_daily_runner_all.py</string>
        <string>--fast</string>
    </array>

    <!-- Run Mon–Fri at 17:30 local time -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    </array>

    <!-- Weekly run on Sunday 08:00: full price cache + fundamentals -->
    <!-- Uncomment to enable:
    <key>StartCalendarInterval</key>
    <dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    -->

    <key>StandardOutPath</key>
    <string>${LOG_OUT}</string>

    <key>StandardErrorPath</key>
    <string>${LOG_ERR}</string>

    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <key>RunAtLoad</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl load "$PLIST_PATH"

echo ""
echo "✅  Canyon autorun installed!"
echo ""
echo "   Schedule : Mon–Fri 17:30 local time"
echo "   Command  : python3 step70_daily_runner_all.py --fast"
echo "   Stdout   : $LOG_OUT"
echo "   Stderr   : $LOG_ERR"
echo ""
echo "   Check status : bash canyon_autorun_setup.sh --status"
echo "   Remove       : bash canyon_autorun_setup.sh --remove"
echo ""
echo "   ⚠️  Weekly deep refresh (step75 + step78) runs manually:"
echo "       python3 canyon_final_v9_step70_daily_runner_all.py --weekly"
