#!/usr/bin/env bash
# Double-click to stop Braillie: the tutor, the website and the phone's tunnel.
echo ""
echo "  Stopping Braillie..."
pkill -TERM -f "run_all.sh" 2>/dev/null       # (its own clean-up stops both servers)
sleep 2
pkill -TERM -f "tutor_server.py" 2>/dev/null
pkill -TERM -f "node_modules/.bin/vite" 2>/dev/null
pkill -TERM -f "cloudflared tunnel --url http://127.0.0.1" 2>/dev/null
sleep 1
pkill -KILL -f "tutor_server.py" 2>/dev/null
pkill -KILL -f "node_modules/.bin/vite" 2>/dev/null
pkill -KILL -f "cloudflared tunnel --url http://127.0.0.1" 2>/dev/null
if pgrep -f "tutor_server.py|node_modules/.bin/vite" >/dev/null; then
    echo "  Something is still running. Try again, or restart the laptop."
else
    echo "  Braillie is stopped."
fi
echo ""
