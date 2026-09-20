#!/usr/bin/env bash
# run_all.sh — start the Braillie tutor backend and the React frontend together.
#
# Usage:
#   ./run_all.sh [mode]          # mode: letters (default) | read | word-quiz
#   TUTOR_PYTHON=/path/to/python3 ./run_all.sh
#
# WHAT THIS DOES NOT DO:
#   These are two independent servers started side-by-side. The React frontend
#   has no button, link, or embed that opens the tutor session right now. That
#   integration is a separate, deliberate decision for later, not bundled here.
#
# TUTOR_PYTHON: override the Python interpreter (e.g. point at a venv's python3).
#   Defaults to whatever `python3` resolves to in your PATH.

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${TUTOR_PYTHON:-python3}"
TUTOR_DIR="$REPO/braille_tutor"
FRONTEND_DIR="$REPO/website-frontend"
TUTOR_PORT=8000
TUTOR_HOST="127.0.0.1"

# ---------------------------------------------------------------------------
# Mode argument
# ---------------------------------------------------------------------------
VALID_MODES="letters read word-quiz"
MODE="${1:-letters}"
if ! echo "$VALID_MODES" | grep -qw "$MODE"; then
    echo "ERROR: unknown mode '$MODE'. Valid: $VALID_MODES" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Preflight checks
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

# ---------------------------------------------------------------------------
# PID file — lets cleanup find children even when trap fires in a subshell
# ---------------------------------------------------------------------------
PID_FILE=$(mktemp /tmp/braillie-pids.XXXXXX)
TUTOR_LOG=$(mktemp /tmp/braillie-tutor.XXXXXX)
FRONTEND_LOG=$(mktemp /tmp/braillie-frontend.XXXXXX)

_CLEANED=0
cleanup() {
    [ "$_CLEANED" -eq 1 ] && return; _CLEANED=1
    echo ""
    echo "  Stopping both servers..."
    local pids
    # Read all stored child PIDs
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        # Grace period, then force-kill anything still alive
        sleep 1
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done < "$PID_FILE"
    fi
    rm -f "$PID_FILE" "$TUTOR_LOG" "$FRONTEND_LOG"
    echo "  Both servers stopped."
}

# Run cleanup on any exit (including Ctrl+C and kill)
trap 'cleanup; exit 0' EXIT INT TERM HUP

# ---------------------------------------------------------------------------
# Start backend
# tutor_server.py MUST run from inside braille_tutor/ (bare module imports)
# exec replaces the subshell so the stored PID is the Python process itself
# ---------------------------------------------------------------------------
(
    cd "$TUTOR_DIR"
    exec "$PYTHON" tutor_server.py \
        --mode "$MODE" \
        --host "$TUTOR_HOST" \
        --port "$TUTOR_PORT"
) >"$TUTOR_LOG" 2>&1 &
TUTOR_PID=$!
echo "$TUTOR_PID" >> "$PID_FILE"

# ---------------------------------------------------------------------------
# Start frontend
# exec node directly — FRONTEND_PID IS the vite process, no npm in the middle
# ---------------------------------------------------------------------------
(
    cd "$FRONTEND_DIR"
    exec node node_modules/.bin/vite
) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" >> "$PID_FILE"

# ---------------------------------------------------------------------------
# Wait for ready signals (poll log files, 30-second timeout each)
# ---------------------------------------------------------------------------
wait_for() {
    local log="$1" pattern="$2" name="$3"
    local n=0
    while ! grep -q "$pattern" "$log" 2>/dev/null; do
        sleep 0.5
        n=$((n+1))
        if [ "$n" -ge 60 ]; then
            echo "  WARNING: timed out waiting for $name" >&2
            tail -5 "$log" >&2
            return 1
        fi
    done
}

echo ""
echo "  Starting Braillie tutor backend (mode=$MODE) ..."
wait_for "$TUTOR_LOG"    "Braillie tutor server" "tutor backend"

echo "  Starting React frontend ..."
wait_for "$FRONTEND_LOG" "ready in"              "React frontend"

# ---------------------------------------------------------------------------
# Extract actual URLs (Vite auto-increments port if 5173 is busy)
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
echo "  NOTE: These are two independent servers. The React frontend"
echo "  has no button, link, or embed that opens the tutor session."
echo "  That integration is a separate, deliberate decision for later."
echo ""

# Open in browser on macOS (silently skip on other OS)
if command -v open >/dev/null 2>&1; then
    open "$TUTOR_URL"    2>/dev/null || true
    open "$FRONTEND_URL" 2>/dev/null || true
fi

echo "  Press Ctrl+C to stop both servers."
echo ""

# ---------------------------------------------------------------------------
# Stream combined log so crashes are visible, then wait for either to exit
# ---------------------------------------------------------------------------
tail -f "$TUTOR_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!
echo "$TAIL_PID" >> "$PID_FILE"

# Block until either server exits (crash or signal)
wait "$TUTOR_PID" 2>/dev/null || true
wait "$FRONTEND_PID" 2>/dev/null || true
