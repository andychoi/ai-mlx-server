#!/bin/bash
# Uninstall ai-mlx-server macOS launchd user agent.
# Usage: bash packaging/uninstall-service.sh [--purge]
#
# By default, the config (~/.config/ai-mlx-server) is preserved.
# Pass --purge to also remove the config directory and log file.

set -euo pipefail

PLIST_NAME="com.andychoi.ai-mlx-server"
PLIST_FILE="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
BINARY="$HOME/.local/bin/ai-mlx-server"
LOG_FILE="$HOME/Library/Logs/ai-mlx-server.log"
CONFIG_DIR="$HOME/.config/ai-mlx-server"

PURGE=false
for arg in "$@"; do
    [[ "$arg" == "--purge" ]] && PURGE=true
done

# 1. Unload and remove the launchd plist
if launchctl list | grep -q "$PLIST_NAME" 2>/dev/null; then
    echo "Stopping service: $PLIST_NAME"
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
else
    echo "Service not loaded (already stopped or never started)"
fi

if [ -f "$PLIST_FILE" ]; then
    rm -f "$PLIST_FILE"
    echo "Removed plist: $PLIST_FILE"
else
    echo "Plist not found (already removed): $PLIST_FILE"
fi

# 2. Remove the binary
if [ -f "$BINARY" ]; then
    rm -f "$BINARY"
    echo "Removed binary: $BINARY"
else
    echo "Binary not found (already removed): $BINARY"
fi

# 3. Optionally remove config and logs
if $PURGE; then
    if [ -d "$CONFIG_DIR" ]; then
        rm -rf "$CONFIG_DIR"
        echo "Removed config directory: $CONFIG_DIR"
    fi
    if [ -f "$LOG_FILE" ]; then
        rm -f "$LOG_FILE"
        echo "Removed log file: $LOG_FILE"
    fi
else
    echo ""
    echo "Config preserved at: $CONFIG_DIR"
    echo "Log preserved at:    $LOG_FILE"
    echo "Re-run with --purge to remove these as well."
fi

echo ""
echo "Uninstall complete."
