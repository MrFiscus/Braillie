#!/usr/bin/env bash
# Double-click to start Braillie (the tutor and the website). Stop it with "Stop Braillie.command", Ctrl+C here, or by closing
# this window.
cd "$(dirname "$0")"
export PATH="$HOME/.local/node/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export TUTOR_PYTHON="$PWD/braille_tutor/.venv312/bin/python"
if pgrep -f "tutor_server.py" >/dev/null || pgrep -f "node_modules/.bin/vite" >/dev/null; then
    echo ""
    echo "  Braillie is already running. Double-click \"Stop Braillie.command\" first to restart it."
    echo ""
    read -r -p "  Press Return to close this window. " _
    exit 0
fi
echo ""
echo "  Starting Braillie. This can take up to 2 minutes while the phone link is set up."
echo "  The website opens by itself when it is ready."
exec ./run_all.sh "$@"
