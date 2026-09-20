#!/usr/bin/env python3
"""
Entry point for the Braillie backend.

    python3 run_server.py                  # real detector (needs braille_tutor + camera)
    python3 run_server.py --mock           # mock detector, no camera required
    VOICE_IO_MOCK=1 python3 run_server.py --mock   # full offline mode

Environment variables
---------------------
    DEEPGRAM_API_KEY   — required unless VOICE_IO_MOCK=1
    ELEVENLABS_API_KEY — required for debrief mode
    VOICE_IO_MOCK=1    — skip all audio APIs; print instead
    VOICE_IO_DEBUG=1   — print every raw Deepgram transcript
    WS_PORT=8765       — WebSocket port (default 8765)
    WS_HOST=0.0.0.0    — bind address (default all interfaces)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

# Make repo root importable (voice_io.py lives here)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Make backend/ importable as "braillie.*"
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("braillie.main")


# ---------------------------------------------------------------------------
# Mock tracking loop: simulates the hand-tracking module calling on_cell_read
# ---------------------------------------------------------------------------

def _start_mock_tracking(session, interval: float = 3.0) -> None:
    """Repeatedly call session.on_cell_read() to simulate finger movement.

    Runs in a daemon thread; stops automatically when the process exits.
    The mock detector cycles through its letter sequence on each call.
    """
    import time

    def _loop():
        x_mm, y_mm = 20.0, 20.0
        while True:
            time.sleep(interval)
            session.on_cell_read(x_mm, y_mm)
            x_mm = (x_mm + 6.5) % 120   # simulate moving across the page

    t = threading.Thread(target=_loop, daemon=True, name="mock-tracking")
    t.start()
    log.info("Mock tracking loop started (%.1fs interval)", interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Braillie backend server")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockDetector instead of braille_tutor (no camera needed)",
    )
    parser.add_argument(
        "--mock-interval",
        type=float,
        default=3.0,
        metavar="SECS",
        help="Seconds between simulated cell reads in mock mode (default: 3)",
    )
    parser.add_argument(
        "--quiz-letters",
        default="abcdefghij",
        metavar="LETTERS",
        help="Letters to include in quiz mode (default: abcdefghij)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Detector
    # ------------------------------------------------------------------
    if args.mock:
        from braillie.mock_detector import MockDetector
        detector = MockDetector(sequence=args.quiz_letters)
        log.info("Using MockDetector (letters: %s)", args.quiz_letters)
    else:
        try:
            from braillie_tutor_adapter import RealDetector  # type: ignore
            detector = RealDetector()
            log.info("Using RealDetector")
        except ImportError:
            log.error(
                "braillie_tutor_adapter not found.  Either:\n"
                "  - Run with --mock for development\n"
                "  - Create braillie_tutor_adapter.py implementing CellDetector\n"
                "    (see README.md for the spec)"
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Server
    # ------------------------------------------------------------------
    from braillie.server import BraillieServer
    server = BraillieServer()

    # ------------------------------------------------------------------
    # 3. Voice I/O
    # ------------------------------------------------------------------
    import voice_io

    mock_voice = os.getenv("VOICE_IO_MOCK", "0").strip() in ("1", "true", "yes")
    if not mock_voice:
        ok, msg = voice_io.check_api_key()
        if not ok:
            log.error("Deepgram auth failed: %s", msg)
            sys.exit(1)
        log.info("Deepgram API key OK")

    # ------------------------------------------------------------------
    # 4. Session
    # ------------------------------------------------------------------
    from braillie.session import TutorSession
    session = TutorSession(
        detector=detector,
        speak_fn=voice_io.speak,
        broadcast=server.broadcast_from_thread,
        quiz_letters=list(args.quiz_letters),
    )

    # ------------------------------------------------------------------
    # 5. on_ready: called once the asyncio loop + server socket are up
    # ------------------------------------------------------------------
    def on_ready():
        voice_io.start_listening()
        session.start()
        if args.mock:
            _start_mock_tracking(session, interval=args.mock_interval)
        log.info("All subsystems running")

    # ------------------------------------------------------------------
    # 6. Run (blocks until Ctrl-C)
    # ------------------------------------------------------------------
    try:
        server.run(on_ready=on_ready)
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        voice_io.stop_listening()
        session.stop()


if __name__ == "__main__":
    main()
