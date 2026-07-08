#!/bin/bash
# Install mlx-server (LLM) or ai-mlx-imager (image-gen) as a macOS launchd user agent.
# Usage: bash packaging/install-service.sh           # LLM server on port 11434
#        bash packaging/install-service.sh --imager  # image server on port 11435

set -euo pipefail

MODE="server"
if [ "${1:-}" = "--imager" ]; then
    MODE="imager"
fi

if [ "$MODE" = "imager" ]; then
    PLIST_NAME="com.andychoi.ai-mlx-imager"
    EXEC_NAME="ai-mlx-imager"
    PORT="11435"
    LOG_NAME="ai-mlx-imager.log"
    EXTRA_ARGS_XML='        <string>--max-resident-models</string>
        <string>1</string>
        <string>--max-resident-gb</string>
        <string>30</string>'
else
    PLIST_NAME="com.andychoi.mlx-server"
    EXEC_NAME="mlx-server"
    PORT="11434"
    LOG_NAME="mlx-server.log"
    EXTRA_ARGS_XML=""
fi

LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"
CONFIG_DIR="$HOME/.config/mlx-server"

mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR" "$CONFIG_DIR"

# Create default models.yaml if it doesn't exist
if [ ! -f "$CONFIG_DIR/models.yaml" ]; then
    cat > "$CONFIG_DIR/models.yaml" <<'EOF'
models: []
# Add models like:
# models:
#   - id: mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ
#     role: chat
#   - id: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
#     role: embedding
#   - id: mlx-community/Qwen-Image-2512-4bit
#     role: image
EOF
    echo "Created default config: $CONFIG_DIR/models.yaml"
fi

# Generate the plist with actual home directory path
cat > "$LAUNCH_AGENTS/$PLIST_NAME.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>$HOME/.local/bin/$EXEC_NAME</string>
        <string>--models-config</string>
        <string>$CONFIG_DIR/models.yaml</string>
        <string>--port</string>
        <string>$PORT</string>
$EXTRA_ARGS_XML
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/$LOG_NAME</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/$LOG_NAME</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "Plist written to: $LAUNCH_AGENTS/$PLIST_NAME.plist"
echo ""
echo "To start the service:"
echo "  launchctl load $LAUNCH_AGENTS/$PLIST_NAME.plist"
echo ""
echo "To stop the service:"
echo "  launchctl unload $LAUNCH_AGENTS/$PLIST_NAME.plist"
echo ""
echo "Logs: $LOG_DIR/$LOG_NAME"
