#!/usr/bin/env bash
# run_all.sh — start the Braillie tutor backend and the React frontend together.
#
# Usage:
#   ./run_all.sh                               # default mode (menu — uses the website login flow)
#   ./run_all.sh --mode letters                # specific mode, no login required
#   ./run_all.sh --mock --no-mic               # offline/no-microphone testing
#
# --phone-camera (the "Connect your phone" QR code) and --paper (fall back to
# the printed sheet's own edges if the 4 ArUco markers can't be read) are
# always on: both are added automatically below unless already present in the
# arguments. --phone-camera needs the phone and laptop on the same Wi-Fi;
# neither has a --no-... flag to turn it off in tutor_server.py.
#
# All arguments are forwarded to tutor_server.py. The script itself only
# controls --host and --port (always 127.0.0.1:8000 for the dev server).
#
# TUTOR_PYTHON: override the Python interpreter (e.g. point at a venv).
#   Defaults to whatever `python3` resolves to in PATH.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${TUTOR_PYTHON:-python3}"
TUTOR_DIR="$REPO/braille_tutor"
FRONTEND_DIR="$REPO/website-frontend"
TUTOR_HOST="127.0.0.1"
TUTOR_PORT=8000

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PYTHON" >&2
    echo "Set TUTOR_PYTHON=/path/to/python3 to override." >&2
    exit 1
fi
if [ ! -f "$TUTOR_DIR/tutor_server.py" ]; then
    echo "ERROR: braille_tutor/tutor_server.py not found." >&2
    exit 1
fi
if [ ! -f "$FRONTEND_DIR/node_modules/.bin/vite" ]; then
    echo "ERROR: website-frontend/node_modules/.bin/vite not found." >&2
    echo "Run:  cd website-frontend && npm install" >&2
    exit 1
fi

# --phone-camera and --paper both run alongside --mode (they only affect the
# video source / page registration), so they just need to be forwarded like
# any other tutor_server.py flag below. Always on; skip re-adding either one
# the caller already passed explicitly.
ARGS=("$@")
PHONE_CAMERA=0
PAPER=0
for arg in "$@"; do
    [ "$arg" = "--phone-camera" ] && PHONE_CAMERA=1
    [ "$arg" = "--paper" ] && PAPER=1
done
if [ "$PHONE_CAMERA" -eq 0 ]; then
    ARGS+=("--phone-camera")
fi
if [ "$PAPER" -eq 0 ]; then
    ARGS+=("--paper")
fi

echo ""
echo "  NOTE: --phone-camera requires the phone and laptop on the same Wi-Fi"
echo "        network. The tutor backend will print a QR code (and a"
echo "        fallback https://<address>:<port> + code) to scan or open from"
echo "        the phone; watch the tutor log below once both servers start."

# ---------------------------------------------------------------------------
# Temp files — one PID list + per-process logs
# ---------------------------------------------------------------------------
PID_FILE=$(mktemp /tmp/braillie-pids.XXXXXX)
TUTOR_LOG=$(mktemp /tmp/braillie-tutor.XXXXXX)
FRONTEND_LOG=$(mktemp /tmp/braillie-frontend.XXXXXX)

_CLEANED=0
cleanup() {
    [ "$_CLEANED" -eq 1 ] && return; _CLEANED=1
    echo ""
    echo "  Stopping both servers..."
    if [ -f "$PID_FILE" ]; then
        # SIGTERM first
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        # Give processes a second to exit cleanly, then SIGKILL stragglers
        sleep 1
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done < "$PID_FILE"
    fi
    rm -f "$PID_FILE" "$TUTOR_LOG" "$FRONTEND_LOG"
    echo "  Done."
}

trap 'cleanup; exit 0' EXIT INT TERM HUP

# ---------------------------------------------------------------------------
# Start backend
# Must run from inside braille_tutor/ — bare module imports (detect, reader…)
# exec replaces the subshell so TUTOR_PID == the Python process directly.
# ARGS (script args, plus --phone-camera unless already given) are forwarded
# to tutor_server.py. --host and --port are injected here.
# ---------------------------------------------------------------------------
(
    cd "$TUTOR_DIR"
    exec "$PYTHON" tutor_server.py \
        --host "$TUTOR_HOST" \
        --port "$TUTOR_PORT" \
        "${ARGS[@]}"
) >"$TUTOR_LOG" 2>&1 &
TUTOR_PID=$!
echo "$TUTOR_PID" >> "$PID_FILE"

# ---------------------------------------------------------------------------
# Start frontend
# exec node directly — FRONTEND_PID IS the vite process, no npm wrapper
# that would become an orphan when killed.
# ---------------------------------------------------------------------------
(
    cd "$FRONTEND_DIR"
    exec node node_modules/.bin/vite
) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" >> "$PID_FILE"

# ---------------------------------------------------------------------------
# Wait for both to signal ready (poll logs, 30 s timeout each)
# ---------------------------------------------------------------------------
wait_for() {
    local log="$1" pattern="$2" name="$3"
    local n=0
    while ! grep -q "$pattern" "$log" 2>/dev/null; do
        sleep 0.5
        n=$((n+1))
        if [ "$n" -ge 60 ]; then
            echo "  WARNING: timed out waiting for $name to start" >&2
            tail -5 "$log" >&2
            return 1
        fi
    done
}

echo ""
echo "  Starting tutor backend ..."
wait_for "$TUTOR_LOG"    "Braillie tutor server" "tutor backend"

echo "  Starting React frontend ..."
wait_for "$FRONTEND_LOG" "ready in"              "React frontend"

# ---------------------------------------------------------------------------
# Extract actual URLs — Vite auto-increments port if 5173 is busy
# ---------------------------------------------------------------------------
TUTOR_URL=$(grep "Braillie tutor server:" "$TUTOR_LOG" \
    | grep -o 'http://[^ ]*' | head -1)
FRONTEND_URL=$(grep "Local:" "$FRONTEND_LOG" \
    | grep -o 'http://[^ ]*' | head -1)

TUTOR_URL="${TUTOR_URL:-http://$TUTOR_HOST:$TUTOR_PORT}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"

echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
printf "  │  %-56s│\n" "Tutor backend   →  $TUTOR_URL"
printf "  │  %-56s│\n" "React frontend  →  $FRONTEND_URL"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

# Open in browser on macOS (silently skip on other platforms)
if command -v open >/dev/null 2>&1; then
    open "$FRONTEND_URL" 2>/dev/null || true
    open "$TUTOR_URL"    2>/dev/null || true
fi

echo "  Press Ctrl+C to stop both servers."
echo ""

# ---------------------------------------------------------------------------
# Stream both logs to the terminal so crashes are immediately visible,
# then block until either server exits (crash or Ctrl+C).
# ---------------------------------------------------------------------------
tail -f "$TUTOR_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!
echo "$TAIL_PID" >> "$PID_FILE"

wait "$TUTOR_PID" 2>/dev/null || true
wait "$FRONTEND_PID" 2>/dev/null || true
