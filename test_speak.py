#!/usr/bin/env python3
"""
test_speak.py — Isolated test for voice_io.py's speak() and speak_debrief().

Tests all three TTS calls through Deepgram alone (narration + both debrief
tiers) without starting the speech-recognition listener.

Usage:
    python3 test_speak.py

Requires DEEPGRAM_API_KEY in the environment or .env file.
VOICE_IO_MOCK=1 will print instead of calling the API (useful for import checks).
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
import traceback


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pre-flight: ffmpeg
#    pydub (used by the retained-but-inactive ElevenLabs code path) needs
#    ffmpeg to decode MP3.  It's not required for the current Deepgram-only
#    path, but we still report it so the state is visible.
# ─────────────────────────────────────────────────────────────────────────────

_ffmpeg_path = shutil.which("ffmpeg")
if _ffmpeg_path:
    print(f"[preflight] ffmpeg found: {_ffmpeg_path}  (not needed for current TTS path)")
else:
    print("[preflight] ffmpeg not on PATH — not needed for Deepgram TTS")
    print("            (only required if ElevenLabs is re-enabled: brew install ffmpeg)")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Log capture
#    voice_io catches every exception internally and routes it through
#    log.error() or log.exception().  A custom handler lets us detect those
#    failures and print them verbatim rather than having them disappear.
# ─────────────────────────────────────────────────────────────────────────────

class _ErrorCapture(logging.Handler):
    """Buffers ERROR+ log records emitted by voice_io during one test step."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def drain(self) -> list[str]:
        """Return formatted messages and clear the buffer."""
        out = []
        for r in self.records:
            out.append(r.getMessage())
            if r.exc_info:
                out.append(
                    "".join(traceback.format_exception(*r.exc_info)).rstrip()
                )
        self.records.clear()
        return out


_capture = _ErrorCapture()

# Show WARNING+ from voice_io on stdout so nothing is hidden.
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.WARNING)
_console.setFormatter(logging.Formatter("  [voice_io log] %(levelname)s: %(message)s"))

_logger = logging.getLogger("voice_io")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_capture)
_logger.addHandler(_console)
_logger.propagate = False   # suppress double-printing via the root logger


# ─────────────────────────────────────────────────────────────────────────────
# 3. Import voice_io (loads .env, reads API keys into module-level constants)
# ─────────────────────────────────────────────────────────────────────────────

import voice_io  # noqa: E402 — must come after log handler setup


def _key_status(value: str, required: bool = True) -> str:
    if not value:
        return "NOT SET — this path will fail" if required else "not set (not required)"
    return f"set  ({len(value)} chars)"


print("Configuration")
print(f"  MOCK_MODE       : {voice_io.MOCK_MODE}")
print(f"  DEEPGRAM_API_KEY: {_key_status(voice_io.DEEPGRAM_API_KEY, required=True)}")
print(f"  Deepgram TTS    : {voice_io.DEEPGRAM_TTS_MODEL}  (Aura Asteria — all modes)")
print()

if voice_io.MOCK_MODE:
    print(
        "NOTE: VOICE_IO_MOCK=1 is set — speak() will print rather than call the API.\n"
        "      Unset VOICE_IO_MOCK to test real audio output.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Step runner
# ─────────────────────────────────────────────────────────────────────────────

_DIVIDER = "─" * 64

def _run(label: str, fn, *args, **kwargs) -> bool:
    """
    Run *fn* and report whether it succeeded.

    Returns True on success (no errors logged, no unhandled exception).

    voice_io catches all internal exceptions and routes them through
    log.error / log.exception.  We detect failures by watching for ERROR-level
    records, not by catching exceptions from speak() itself.

    Timing is shown as a proxy for "audio was actually played": a call that
    short-circuits (missing key, import error, etc.) returns in <0.1 s;
    a real API call + playback takes at minimum several hundred milliseconds.
    """
    print(_DIVIDER)
    print(f"STEP : {label}")
    print(f"USING: Deepgram TTS  {voice_io.DEEPGRAM_TTS_MODEL}")
    print(_DIVIDER)

    _capture.clear()
    t0 = time.monotonic()
    unhandled: Exception | None = None

    try:
        fn(*args, **kwargs)
    except Exception as exc:
        unhandled = exc

    elapsed = time.monotonic() - t0
    errors = _capture.drain()

    print(f"Elapsed: {elapsed:.2f}s")

    if unhandled is not None:
        print("UNEXPECTED EXCEPTION (voice_io should have caught this internally):")
        traceback.print_exc(file=sys.stdout)

    if errors:
        print("ERRORS captured from voice_io logger:")
        for msg in errors:
            for line in msg.splitlines():
                print(f"  {line}")
        print()
        print("RESULT: FAILED")
        print()
        return False

    if elapsed < 0.1 and not voice_io.MOCK_MODE:
        print(
            "WARNING: call returned in <0.1 s with no errors logged — "
            "this may mean it short-circuited silently (missing API key, "
            "import error that was swallowed, etc.)."
        )
        print("RESULT: UNCERTAIN (check configuration above)")
        print()
        return False

    print("Audio playback attempted — no errors logged.")
    print("RESULT: OK")
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Test steps — all three go through Deepgram TTS
# ─────────────────────────────────────────────────────────────────────────────

results: dict[str, bool] = {}

# Step 1: normal narration
results["deepgram_normal"] = _run(
    "speak(mode='normal') — in-session narration",
    voice_io.speak,
    text="Testing normal narration mode.",
    mode="normal",
)

# Step 2: debrief, high-accuracy tier (≥70 %)
# speak_debrief() takes accuracy as 0.0–1.0; 85 % = 0.85
results["deepgram_debrief_high"] = _run(
    "speak_debrief(accuracy=0.85) — 'good session' tier (≥70 %)",
    voice_io.speak_debrief,
    accuracy=0.85,
    missed_cells=["b", "k", "q"],
)

# Step 3: debrief, low-accuracy tier (<50 %)
results["deepgram_debrief_low"] = _run(
    "speak_debrief(accuracy=0.35) — 'tough session' tier (<50 %)",
    voice_io.speak_debrief,
    accuracy=0.35,
    missed_cells=["a", "e", "i", "o", "u"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("═" * 64)
print("SUMMARY")
print("═" * 64)
all_ok = True
for name, ok in results.items():
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name}")
    if not ok:
        all_ok = False

print()
if not voice_io.DEEPGRAM_API_KEY:
    print("  [!] DEEPGRAM_API_KEY not set — all steps will have failed")

print()
print("Overall:", "ALL PASSED ✓" if all_ok else "ONE OR MORE STEPS FAILED ✗")
