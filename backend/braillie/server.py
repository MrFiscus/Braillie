"""
WebSocket server for the Braillie backend.

Pushes real-time session state to the frontend.  All session changes
(mode switches, cell reads, narration, quiz results, stats) arrive as
JSON messages on the socket.

WHAT THE FRONTEND TEAMMATE NEEDS TO CONNECT
============================================
WebSocket URL:  ws://<host>:8765        (port overridable via WS_PORT env var)

Connect and listen.  No authentication, no handshake beyond the WS upgrade.

On connect: the server immediately sends a ``state`` snapshot so the UI
has something to render before the first event.

Message reference: see MESSAGE TYPES below.

Example (browser JS)::

    const ws = new WebSocket("ws://localhost:8765");
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
            case "state":   updateModeUI(msg);    break;
            case "cell":    updateCellDisplay(msg); break;
            case "narration": announceToScreenReader(msg.text); break;
            case "quiz_prompt": showTarget(msg.target); break;
            case "quiz_result": showResult(msg);  break;
            case "stats":   updateStatsBar(msg);  break;
            case "voice_command": flashCommand(msg.command); break;
            case "error":   showError(msg.message); break;
        }
    };

Optional client → server messages (currently no-ops; wired for future use)::

    ws.send(JSON.stringify({ type: "set_mode", mode: "quiz" }));
    ws.send(JSON.stringify({ type: "next" }));

MESSAGE TYPES (backend → frontend)
====================================

state
-----
Sent on connect and whenever the session mode or mic status changes.
{
    "type":       "state",
    "mode":       "idle" | "read" | "quiz",
    "listening":  true | false,     // Deepgram STT stream is open
    "mic_paused": true | false      // mic muted (e.g. while TTS plays)
}

cell
----
Sent when the tracking loop reports a stable finger position over a cell.
{
    "type":       "cell",
    "char":       "⠁",             // Unicode braille character
    "label":      "100000",        // 6-bit dot string (dot 1..6)
    "dots":       [1],             // active dot numbers as an array
    "x_mm":       23.4,            // page-coordinate position, mm
    "y_mm":       45.1,
    "confidence": 0.97
}

narration
---------
Sent whenever the backend calls speak().  Use to drive screen-reader
announcements for users who cannot hear the speaker, or to display
a subtitle.
{
    "type": "narration",
    "text": "The letter is A.",
    "mode": "normal" | "debrief"
}

quiz_prompt
-----------
Sent when a new quiz target is selected.
{
    "type":    "quiz_prompt",
    "target":  "A",           // uppercase letter to find
    "attempt": 1              // attempt number within this target (resets on advance)
}

quiz_result
-----------
Sent after "found it" is processed.
{
    "type":     "quiz_result",
    "correct":  true | false,
    "detected": "B",          // what the detector found at the finger's position
    "target":   "A"
}

stats
-----
Sent after every quiz_result.
{
    "type":        "stats",
    "total":       12,
    "correct":     9,
    "accuracy":    0.75,     // 0.0–1.0
    "streak":      3,
    "most_missed": ["d", "g", "j"]
}

voice_command
-------------
Sent whenever voice_io recognises a command.  Use to update a "listening"
indicator or highlight the recognised command in the UI.
{
    "type":    "voice_command",
    "command": "repeat" | "hint" | "found it" | "next" | "stop" | "start quiz"
}

error
-----
Sent when something goes wrong that the UI should surface.
{
    "type":    "error",
    "message": "DEEPGRAM_API_KEY is not set."
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

log = logging.getLogger("braillie.server")

WS_PORT: int = int(os.getenv("WS_PORT", "8765"))
WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")


class BraillieServer:
    """Async WebSocket server.  Call run() to start the event loop.

    Designed to be the main thread's asyncio entry point.  The voice-command
    and tracking-loop threads communicate with it via broadcast_from_thread().
    """

    def __init__(self) -> None:
        self._clients: set[ServerConnection] = set()
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_state: dict = {
            "type": "state",
            "mode": "idle",
            "listening": False,
            "mic_paused": False,
        }

    # ------------------------------------------------------------------
    # Thread-safe broadcast (call from any thread)
    # ------------------------------------------------------------------

    def broadcast_from_thread(self, msg: dict) -> None:
        """Push *msg* to all connected WebSocket clients.

        Thread-safe.  Call this from voice-command callbacks, the tracking
        loop, or anywhere else outside the asyncio event loop.
        """
        if self._loop is None or self._loop.is_closed():
            return
        # Cache state snapshots so late-connecting clients get the latest
        if msg.get("type") == "state":
            self._last_state = msg
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        log.info("Client connected  total=%d", len(self._clients))
        try:
            # Send current state immediately so the UI renders before any event
            await ws.send(json.dumps(self._last_state))
            # Keep the connection open; process optional client → server messages
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_client_msg(ws, msg)
                except json.JSONDecodeError:
                    log.warning("Ignoring non-JSON message from client")
        except Exception:
            log.debug("Client connection closed")
        finally:
            self._clients.discard(ws)
            log.info("Client disconnected  total=%d", len(self._clients))

    async def _handle_client_msg(self, ws: ServerConnection, msg: dict) -> None:
        """Optional client→server commands (set_mode, next).  No-op for MVP."""
        t = msg.get("type")
        if t in ("set_mode", "next"):
            log.debug("Client sent %r (not yet wired to session)", t)

    async def _broadcaster(self) -> None:
        while True:
            msg = await self._queue.get()
            payload = json.dumps(msg)
            dead: set[ServerConnection] = set()
            for ws in list(self._clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        host: str = WS_HOST,
        port: int = WS_PORT,
        on_ready: Any = None,
    ) -> None:
        """Start the asyncio event loop and serve forever.

        Parameters
        ----------
        on_ready:
            Optional callable invoked (in the loop thread) once the server
            is accepting connections.  Use to start voice_io after the loop
            is running.
        """
        asyncio.run(self._run(host, port, on_ready))

    async def _run(self, host: str, port: int, on_ready: Any) -> None:
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._broadcaster())

        async with serve(self._handler, host, port) as server:
            actual_port = server.sockets[0].getsockname()[1]
            log.info("WebSocket server listening on ws://%s:%d", host, actual_port)
            print(
                f"Braillie backend ready  ws://{host}:{actual_port}  "
                "(frontend: ws://localhost:{actual_port})",
                flush=True,
            )
            if on_ready is not None:
                on_ready()
            await server.serve_forever()
