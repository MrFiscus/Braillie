"""
voice_io.py — Voice input/output module for the Braillie literacy tutor.

PUBLIC INTERFACE
================

Listening
---------
    register_command(command: str, callback: Callable[[], None]) -> None
        Register a handler for one of the recognised voice commands.
        Commands: "repeat", "hint", "found it", "next", "stop", "start quiz"

    start_listening() -> None
        Open the microphone and begin streaming audio to Deepgram.
        Recognised commands trigger the registered callbacks.

    stop_listening() -> None
        Tear down the audio stream and Deepgram websocket cleanly.

    pause_listening() -> None
        Mute the microphone input (call before the app speaks so the TTS
        output is not transcribed and fed back as a command).

    resume_listening() -> None
        Un-mute the microphone after the app has finished speaking.

Speaking
--------
    speak(text: str, mode: str = "normal") -> None
        Synthesise and play text.  Blocks until playback finishes, then
        automatically calls resume_listening() so the mic is always live
        after an utterance.

        mode values
        -----------
        "normal"   — Deepgram TTS (low-latency, used for all in-session
                     narration: letters, quiz prompts, right/wrong feedback)
        "debrief"  — ElevenLabs TTS (richer voice, used only for the end-
                     of-session debrief)

    speak_debrief(accuracy: float, missed_cells: list[str]) -> None
        Generate a short debrief script whose tone shifts with performance,
        then speak it via ElevenLabs (equivalent to calling speak(…, "debrief")).

        accuracy     — 0.0–1.0 (e.g. 0.82 → 82 %)
        missed_cells — list of braille cell labels that were most missed
                       (e.g. ["A", "B", "SH"])

ENVIRONMENT VARIABLES
---------------------
    DEEPGRAM_API_KEY    — Deepgram secret key
    ELEVENLABS_API_KEY  — ElevenLabs secret key

    VOICE_IO_MOCK=1     — Offline/mock mode: prints instead of hitting APIs.
                          Useful for dev/CI without spending credits.
    VOICE_IO_DEBUG=1    — Print every raw Deepgram transcript + match result.
    COMMAND_CONFIDENCE_THRESHOLD — Float 0–1, default 0.72.  Lower = more
                          sensitive; raise if you get false fires in noisy rooms.

QUICK START
-----------
    from voice_io import check_api_key, register_command, start_listening, speak

    ok, msg = check_api_key()
    if not ok:
        raise RuntimeError(msg)

    register_command("next", lambda: print("Moving to next cell"))
    register_command("repeat", lambda: speak("The letter is A"))
    start_listening()
    # … main tracking loop …
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import threading
import time
import wave
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so import errors surface with a clear
# message pointing at the install step.
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Read only the project-local file. Existing environment variables retain
# precedence, which keeps deployed configurations and CI secrets unchanged.
load_dotenv(Path(__file__).with_name(".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOCK_MODE: bool = os.getenv("VOICE_IO_MOCK", "0").strip() in ("1", "true", "yes")

# Debug mode: print every raw Deepgram transcript with confidence + match result.
DEBUG_MODE: bool = os.getenv("VOICE_IO_DEBUG", "0").strip() in ("1", "true", "yes")

DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

# ElevenLabs integration is retained in code but not used on the active run
# path. ELEVENLABS_API_KEY is not required; nothing in the normal flow checks
# or warns about it being absent.
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# Deepgram TTS. Aura 1 drops the last ~200–300 ms of a short line ("cat" -> "ca"), so we keep
# this model (it is the one that actually speaks in this demo) and add a pause + file tail below.
DEEPGRAM_TTS_MODEL: str = "aura-asteria-en"
TTS_TAIL = "..."  # synthesised as a pause, not a word: if anything still eats the end, this is what goes
WAV_SILENCE_SECONDS = 0.45  # extra quiet at the end of the file, for phone players that clip the last buffer

# ElevenLabs voice ID retained for future reference; not called at runtime.
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Deepgram STT model.
# nova-3 is Deepgram's current flagship real-time model (better accuracy,
# lower latency than nova-2, especially for short phrases in noisy rooms).
DEEPGRAM_STT_MODEL: str = "nova-3"

# Minimum average word-confidence before a transcript is eligible for matching.
# 0.72 is deliberately below 0.75 so that clearly-spoken commands near the
# threshold don't get silently dropped; raise via env var in very noisy rooms.
COMMAND_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("COMMAND_CONFIDENCE_THRESHOLD", "0.72")
)

# Explicit microphone device index.  None = fall back to system default.
# Override at runtime via the VOICE_IO_MIC_INDEX env var, or set it here
# to pin a specific device and avoid silent switches on demo day.
_mic_index_env = os.getenv("VOICE_IO_MIC_INDEX", "").strip()
MIC_DEVICE_INDEX: int | None = int(_mic_index_env) if _mic_index_env else None

# Audio capture settings.
SAMPLE_RATE: int = 16_000
CHANNELS: int = 1
CHUNK_MS: int = 100  # milliseconds of audio per microphone read

# ---------------------------------------------------------------------------
# Known commands
# ---------------------------------------------------------------------------

# Map of normalised command text → canonical command name.
# Phrases may appear as complete words anywhere in a final transcript, so
# "give me a hint" fires "hint" and "okay I found it" fires "found it".
_COMMAND_MAP: dict[str, str] = {
    # --- repeat ----------------------------------------------------------
    "repeat": "repeat",
    "say it again": "repeat",
    "say again": "repeat",
    # --- hint ------------------------------------------------------------
    # "hint" is consistently misheard as "hand", "int", "hamed", etc.
    # All four natural phrases + confirmed Deepgram misreads route here.
    "hint": "hint",
    "give me a hint": "hint",
    "clue": "hint",
    "help me": "hint",
    "hence": "hint",          # confirmed misread of "hint"
    "hen": "hint",            # confirmed misread of "hint"
    "hints": "hint",          # confirmed misread of "hint"
    "health": "hint",         # confirmed misread of "hint"
    # --- found it --------------------------------------------------------
    "found it": "found it",
    "i found it": "found it",
    "got it": "found it",     # natural shorthand: "got it" = located the cell
    "i got it": "found it",
    # --- navigation ------------------------------------------------------
    "next": "next",
    "skip": "next",
    "move on": "next",
    "continue": "next",
    "next page": "next page",  # turn to a new page: longer phrases win, so this is never heard as plain "next"
    "new page": "next page",
    "another page": "next page",
    # --- guided lessons (learn mode) -------------------------------------
    "explore": "explore",  # free exploring between lessons
    "free explore": "explore",
    "let me explore": "explore",
    "practice": "practice",  # adaptive review of the letters that need it most
    "practise": "practice",
    "review": "practice",
    "start": "start quiz",  # "say start to begin a lesson"
    "start lesson": "start quiz",
    "begin lesson": "start quiz",
    "start learning": "start quiz",
    # --- the menu: what do you want to do today? -------------------------
    "learn": "learn",
    "let's learn": "learn",
    "lets learn": "learn",
    "read": "read",
    "reading": "read",
    "let's read": "read",
    "lets read": "read",
    "quiz": "quiz",  # ("start quiz" is longer, so it still means "start" while a quiz is being set up)
    "take a quiz": "quiz",
    "quiz me": "quiz",
    "menu": "menu",
    "main menu": "menu",
    "change mode": "menu",
    "go back": "menu",
    # --- accessibility: ask what can be said, and how fast to go -----------
    "help": "help",  # ("help me" is longer and still means hint)
    "what can i say": "help",
    "what are my options": "help",
    "commands": "help",
    "slower": "slower",
    "speak slower": "slower",
    "slow down": "slower",
    "faster": "faster",
    "speak faster": "faster",
    "speed up": "faster",
    "take your time": "relaxed",
    "more time": "relaxed",
    "normal pace": "normal",
    "back to normal": "normal",
    "stop": "stop",
    "quit": "stop",
    "i'm done": "stop",
    "im done": "stop",        # apostrophe-free variant
    # --- quiz ------------------------------------------------------------
    "start quiz": "start quiz",
    "begin quiz": "start quiz",
}

# Test phrases longest-first. Testing each phrase in that order, rather than
# relying on one alternation regex, guarantees the longest matching command is
# selected even when a shorter phrase appears earlier in the utterance.
_COMMAND_PHRASES: tuple[str, ...] = tuple(
    sorted(_COMMAND_MAP, key=lambda phrase: (-len(phrase), phrase))
)
_COMMAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (phrase, re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"))
    for phrase in _COMMAND_PHRASES
)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_callbacks: dict[str, list[Callable[[], None]]] = {}
_listening_active: bool = False
_mic_paused: bool = False
_listener_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()

# Guards start_listening()/stop_listening() so two threads racing to start
# (e.g. two hub/menu transitions firing close together) can never both pass
# the "not already listening" check and end up with two concurrent pa.open()
# calls on the same device — that race is what segfaults inside PortAudio's
# native OpenStream, below anything a Python try/except can catch.
_lifecycle_lock = threading.RLock()


def _normalize_transcript(transcript: str) -> str:
    """Normalize user speech without altering words inside the transcript."""
    normalized = " ".join(transcript.casefold().strip().split())
    return normalized.rstrip(".,!?;:")


def _match_command(normalized_transcript: str) -> str | None:
    """Return the canonical command for the longest whole-phrase match."""
    for phrase, pattern in _COMMAND_PATTERNS:
        if pattern.search(normalized_transcript):
            return _COMMAND_MAP[phrase]
    return None


def _classify_transcript(
    transcript: str,
    confidence: float,
    is_final: bool,
) -> tuple[str, str | None, str | None]:
    """Normalize and decide whether one STT event is eligible to dispatch.

    Returns ``(normalized_transcript, command, ignored_reason)``. Exactly one
    command can be returned for an event.
    """
    normalized = _normalize_transcript(transcript)
    if not is_final:
        return normalized, None, "interim transcript"
    if _mic_paused:
        return normalized, None, "microphone is paused"
    if not normalized:
        return normalized, None, "empty transcript"
    if confidence < COMMAND_CONFIDENCE_THRESHOLD:
        return normalized, None, "confidence below threshold"

    command = _match_command(normalized)
    if command is None:
        return normalized, None, "no command phrase matched"
    return normalized, command, None


def _debug_transcript_event(
    transcript: str,
    confidence: float,
    is_final: bool,
    normalized: str,
    command: str | None,
    ignored_reason: str | None,
) -> None:
    """Emit safe, opt-in diagnostics for a single Deepgram transcript event."""
    if not DEBUG_MODE:
        return
    status = "final" if is_final else "interim"
    outcome = f"matched_command={command!r}" if command else "matched_command=None"
    reason = "" if ignored_reason is None else f" ignored_reason={ignored_reason!r}"
    print(
        "[VOICE_IO DEBUG]"
        f" transcript={transcript!r}"
        f" confidence={confidence:.3f}"
        f" status={status}"
        f" normalized={normalized!r}"
        f" {outcome}{reason}",
        flush=True,
    )


def _process_transcript_event(transcript: str, confidence: float, is_final: bool) -> str | None:
    """Classify one transcript event and dispatch at most one command."""
    normalized, command, ignored_reason = _classify_transcript(
        transcript, confidence, is_final
    )
    _debug_transcript_event(
        transcript, confidence, is_final, normalized, command, ignored_reason
    )
    if is_final and normalized and ignored_reason in (None, "no command phrase matched", "confidence below threshold"):
        _status["heard"] = {"n": _status["heard"]["n"] + 1, "text": normalized, "matched": command is not None, "reason": ignored_reason}
    if command is not None:
        log.info("Command recognised: %r (conf=%.2f)", command, confidence)
        _status["last_transcript"] = normalized
        _fire_command(command)
    return command


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class VoiceIOError(RuntimeError):
    """Raised by start_listening() when Deepgram mode cannot be set up.

    Callers should catch this to surface a clear startup failure rather than
    waiting for the listener thread to silently do nothing.
    """


# ---------------------------------------------------------------------------
# Listener status (readable from outside without touching internals)
# ---------------------------------------------------------------------------

_status: dict = {
    "mode": None,           # set to "mock" or "deepgram" on first start_listening()
    "connected": False,
    "mic_index": None,
    "mic_name": None,
    "last_error": None,
    "last_transcript": None,
    "commands_fired": 0,
    # the newest final utterance that was heard while the microphone was live, understood or not (for apps that want to say "I didn't
    # catch that"): n counts them, matched says whether a command was found, reason says why not
    "heard": {"n": 0, "text": "", "matched": False, "reason": None},
}


def get_mode() -> str:
    """Return "mock" or "deepgram" depending on VOICE_IO_MOCK."""
    return "mock" if MOCK_MODE else "deepgram"


def listener_status() -> dict:
    """Snapshot of listener state; safe to call from any thread."""
    return dict(_status)


# ---------------------------------------------------------------------------
# Public API — auth check
# ---------------------------------------------------------------------------


def check_api_key() -> tuple[bool, str]:
    """Verify DEEPGRAM_API_KEY is set and accepted by Deepgram's API.

    Makes a single lightweight GET request (no audio, no billing).
    Returns (True, info_message) on success, (False, error_message) on failure.
    Call this at app startup before start_listening() so a bad key surfaces
    immediately rather than silently after the first audio packet.

    Example::

        ok, msg = check_api_key()
        if not ok:
            raise RuntimeError(f"Deepgram auth failed: {msg}")
    """
    if MOCK_MODE:
        return True, "Mock mode — auth check skipped"
    if not DEEPGRAM_API_KEY:
        return False, (
            "DEEPGRAM_API_KEY is not set.  "
            "Export it or set VOICE_IO_MOCK=1 for offline testing."
        )
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            "https://api.deepgram.com/v1/auth/token",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return True, f"API key valid (HTTP {resp.status})"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "API key rejected (HTTP 401 Unauthorized) — check DEEPGRAM_API_KEY"
        return False, f"Deepgram auth endpoint returned HTTP {exc.code}"
    except Exception as exc:
        return False, f"Auth check failed (network?): {exc}"


# ---------------------------------------------------------------------------
# Public API — callbacks
# ---------------------------------------------------------------------------


def register_command(command: str, callback: Callable[[], None]) -> None:
    """Register *callback* to be called when *command* is recognised.

    Can be called multiple times for the same command to attach multiple
    handlers.  Handlers are called in registration order.
    """
    canonical = _COMMAND_MAP.get(command.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown command {command!r}. "
            f"Valid commands: {sorted(set(_COMMAND_MAP.values()))}"
        )
    _callbacks.setdefault(canonical, []).append(callback)
    log.debug("Registered handler for command %r", canonical)


def _fire_command(command: str) -> None:
    """Dispatch *command* to all registered callbacks (non-blocking)."""
    handlers = _callbacks.get(command, [])
    if not handlers:
        log.debug("Command %r has no registered handlers; ignoring.", command)
        return
    _status["commands_fired"] += 1
    for cb in handlers:
        try:
            cb()
        except Exception:
            log.exception("Error in handler for command %r", command)


# ---------------------------------------------------------------------------
# Public API — lifecycle
# ---------------------------------------------------------------------------


def start_listening() -> None:
    """Start continuous microphone capture and Deepgram streaming STT.

    Hard guard: refuses to start a second listener thread while one is
    already running for this process, even under concurrent calls (e.g. two
    hub/menu mode transitions firing close together). Two live listener
    threads would mean two concurrent pa.open() calls racing on the same
    microphone, which is a known crash path (see module docstring / crash
    report): it segfaults inside PortAudio's native OpenStream, a level
    below anything Python's exception handling can intercept. The lock makes
    the "is one already running" check-and-set atomic, and also checks the
    thread object itself (not just the flag), so a stale flag can never
    slip a second pa.open() through.
    """
    global _listening_active, _listener_thread, _stop_event

    with _lifecycle_lock:
        if _listening_active or (_listener_thread is not None and _listener_thread.is_alive()):
            log.warning("start_listening() called but already listening; ignoring.")
            return

        _stop_event = threading.Event()

        if MOCK_MODE:
            _listener_thread = threading.Thread(
                target=_mock_listener_loop, daemon=True, name="voice-listener"
            )
        else:
            _deepgram_preflight()           # raises VoiceIOError early if deps are missing
            _listener_thread = threading.Thread(
                target=_deepgram_listener_loop, daemon=True, name="voice-listener"
            )

        _status["mode"] = get_mode()
        _listening_active = True
        _listener_thread.start()
        log.info("Voice listener started (mock=%s)", MOCK_MODE)


def stop_listening() -> None:
    """Stop the microphone stream and close the Deepgram connection."""
    global _listening_active

    with _lifecycle_lock:
        if not _listening_active:
            return
        _stop_event.set()
        thread = _listener_thread

    if thread is not None:
        thread.join(timeout=5)

    with _lifecycle_lock:
        _listening_active = False
    log.info("Voice listener stopped.")


def pause_listening() -> None:
    """Mute microphone input.  Call before speak() to prevent feedback."""
    global _mic_paused
    _mic_paused = True
    log.debug("Microphone paused.")


def resume_listening() -> None:
    """Re-enable microphone input after speak() finishes."""
    global _mic_paused
    _mic_paused = False
    log.debug("Microphone resumed.")


# ---------------------------------------------------------------------------
# Public API — speaking
# ---------------------------------------------------------------------------


def speak(text: str, mode: str = "normal") -> None:
    """Synthesise *text* and play it via Deepgram TTS.

    Automatically pauses the microphone before speaking and resumes it
    after, so the system never transcribes its own narration.

    Args:
        text: The text to speak.
        mode: Accepted for API compatibility ("normal", "debrief") but both
              route through Deepgram TTS.  ElevenLabs code is retained in
              _elevenlabs_speak() but is not called from this path.
    """
    pause_listening()
    try:
        if MOCK_MODE:
            _mock_speak(text, mode)
        else:
            _deepgram_speak(text)
    except Exception:
        log.exception("speak() failed for mode=%r; text=%r", mode, text)
    finally:
        resume_listening()


def speak_debrief(accuracy: float, missed_cells: list[str]) -> None:
    """Generate and speak an end-of-session debrief via Deepgram TTS.

    Args:
        accuracy:     Session accuracy as a fraction 0.0–1.0.
        missed_cells: Braille cell labels the learner missed most often.
    """
    script = _build_debrief_script(accuracy, missed_cells)
    speak(script, mode="debrief")


# ---------------------------------------------------------------------------
# Debrief script generation
# ---------------------------------------------------------------------------


def _build_debrief_script(accuracy: float, missed_cells: list[str]) -> str:
    pct = round(accuracy * 100)

    if accuracy >= 0.90:
        opener = (
            f"Incredible work today! You finished with {pct} percent accuracy. "
            "You're really getting the feel of the cells under your fingers."
        )
        closer = "Keep this momentum going — you're well on your way to fluency."
    elif accuracy >= 0.70:
        opener = (
            f"Good session! You scored {pct} percent today. "
            "You're making solid progress."
        )
        closer = (
            "A little more practice each day and those tricky cells will start "
            "feeling natural."
        )
    elif accuracy >= 0.50:
        opener = (
            f"You finished with {pct} percent today. That's a decent start, "
            "and every session builds your muscle memory."
        )
        closer = (
            "Don't get discouraged — braille takes time, and you're putting in "
            "the work."
        )
    else:
        opener = (
            f"Today was a tough one — {pct} percent — but showing up is the "
            "most important step."
        )
        closer = (
            "Take a short break, then try again. The cells will click sooner "
            "than you think."
        )

    if missed_cells:
        cell_list = ", ".join(missed_cells[:-1])
        if len(missed_cells) > 1:
            cell_list += f", and {missed_cells[-1]}"
        else:
            cell_list = missed_cells[0]
        focus = (
            f"The cells that gave you the most trouble were {cell_list}. "
            "Those are worth a few extra minutes of focused practice next time."
        )
    else:
        focus = "You handled every cell without a pattern of errors — impressive."

    return f"{opener} {focus} {closer}"


# ---------------------------------------------------------------------------
# Deepgram STT — real implementation
# ---------------------------------------------------------------------------


def _deepgram_preflight() -> None:
    """Validate Deepgram dependencies before the listener thread starts.

    Raises VoiceIOError with an actionable message if anything is missing,
    so start_listening() fails loudly instead of spawning a silent no-op thread.
    """
    _hint = (
        "Run: pip install -r requirements-voice.txt, "
        "set DEEPGRAM_API_KEY, or use VOICE_IO_MOCK=1."
    )
    try:
        from deepgram import DeepgramClient                                      # noqa: F401
        from deepgram.listen.v1.types.listen_v1results import ListenV1Results   # noqa: F401
    except (ModuleNotFoundError, ImportError) as exc:
        raise VoiceIOError(
            f"deepgram-sdk is required but unavailable: {exc}. {_hint}"
        ) from exc
    try:
        import pyaudio  # noqa: F401
    except ModuleNotFoundError as exc:
        raise VoiceIOError(f"pyaudio is not installed: {exc}. {_hint}") from exc
    if not DEEPGRAM_API_KEY:
        raise VoiceIOError(f"DEEPGRAM_API_KEY is not set. {_hint}")


def _validate_input_device(pa, index: int | None) -> dict:
    """Raise VoiceIOError unless *index* is a real input device that can still
    open at SAMPLE_RATE/CHANNELS right now.

    Re-checked immediately before every pa.open() call, never cached from an
    earlier check: a device (especially a USB/Bluetooth mic) can disappear or
    change capabilities between one open and the next in the same session.
    Calling pa.open() on a stale or nonexistent index is what segfaults inside
    PortAudio's native OpenStream — a crash below anything Python's
    try/except can catch — so the goal here is to never make that call at all
    for a device that cannot presently support it.
    """
    import pyaudio  # type: ignore[import-untyped]

    try:
        count = pa.get_device_count()
    except Exception as exc:
        raise VoiceIOError(f"could not query audio devices: {exc}") from exc

    if index is None or index < 0 or index >= count:
        raise VoiceIOError(
            f"microphone device index {index!r} does not exist "
            f"(the system currently reports {count} audio device(s))"
        )

    try:
        info = pa.get_device_info_by_index(index)
    except Exception as exc:
        raise VoiceIOError(
            f"microphone device #{index} could not be queried (it may have just "
            f"disconnected): {exc}"
        ) from exc

    name = info.get("name", "unknown")
    max_in = int(info.get("maxInputChannels", 0) or 0)
    if max_in < CHANNELS:
        raise VoiceIOError(
            f"microphone device #{index} ({name}) has {max_in} input channel(s) "
            f"right now, needs at least {CHANNELS}"
        )

    try:
        pa.is_format_supported(
            SAMPLE_RATE,
            input_device=index,
            input_channels=CHANNELS,
            input_format=pyaudio.paInt16,
        )
    except ValueError as exc:
        raise VoiceIOError(
            f"microphone device #{index} ({name}) does not support "
            f"{SAMPLE_RATE} Hz / {CHANNELS} channel(s) input right now: {exc}"
        ) from exc

    return info


def _open_input_stream(pa, device_index: int | None, chunk_frames: int):
    """Validate then open the microphone input stream, one device at a time.

    Never calls pa.open() on a device that _validate_input_device() has not
    just approved. If *device_index* (the pinned VOICE_IO_MIC_INDEX, or None
    for "use the default") is invalid or fails to open, falls back to the
    system default input device — itself validated before opening, so a
    fully-gone audio subsystem raises VoiceIOError instead of segfaulting.

    Returns (audio_stream, active_index, fell_back_to_default).
    """
    import pyaudio  # type: ignore[import-untyped]

    open_kwargs: dict = dict(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=chunk_frames,
    )

    if device_index is not None:
        try:
            _validate_input_device(pa, device_index)
            return pa.open(input_device_index=device_index, **open_kwargs), device_index, False
        except (VoiceIOError, OSError) as exc:
            print(
                f"voice_io: ERROR — pinned mic device index {device_index} is unusable "
                f"right now ({exc}). Falling back to system default.",
                flush=True,
            )

    try:
        default_index = pa.get_default_input_device_info()["index"]
    except Exception as exc:
        raise VoiceIOError(f"no default input device is available: {exc}") from exc
    _validate_input_device(pa, default_index)  # raises VoiceIOError if even this is unusable
    return (
        pa.open(input_device_index=default_index, **open_kwargs),
        default_index,
        device_index is not None,
    )


def _deepgram_listener_loop() -> None:
    """Background thread: streams mic audio to Deepgram and dispatches commands."""
    try:
        import pyaudio  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        log.error("pyaudio is not installed.  Run: pip install pyaudio")
        return

    try:
        from deepgram import DeepgramClient  # type: ignore[import-untyped]
        from deepgram.core import EventType  # type: ignore[import-untyped]
        from deepgram.listen.v1.types.listen_v1results import (  # type: ignore[import-untyped]
            ListenV1Results,
        )
    except ModuleNotFoundError:
        log.error("deepgram-sdk is not installed.  Run: pip install 'deepgram-sdk==7.9.0'")
        return
    except ImportError as exc:
        log.error(
            "deepgram-sdk is installed but the expected API names were not found: %s\n"
            "This module requires deepgram-sdk==7.9.0.  "
            "Check the installed version with: pip show deepgram-sdk",
            exc,
        )
        return

    if not DEEPGRAM_API_KEY:
        log.error(
            "DEEPGRAM_API_KEY is not set. "
            "Set the environment variable or enable VOICE_IO_MOCK=1."
        )
        return

    backoff = 1.0

    while not _stop_event.is_set():
        pa = None
        audio_stream = None

        try:
            client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
            chunk_frames = int(SAMPLE_RATE * CHUNK_MS / 1000)

            with client.listen.v1.connect(
                model=DEEPGRAM_STT_MODEL,    # nova-3
                language="en-US",            # pin language; avoids detection latency
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                endpointing=300,             # 300 ms silence → finalise utterance
                                             # (was 500; shorter = faster command response
                                             #  without cutting off "start quiz" mid-phrase)
                vad_events=True,
                smart_format=True,           # handles capitalisation + punctuation;
                                             # our \b regex handles trailing periods fine
                interim_results=False,       # only final transcripts; interim results are
                                             # a common source of inaccurate word picks
                keyterm=[                    # boost command vocabulary in the acoustic model
                    "repeat", "hint", "next", "stop",
                    "found it", "got it",
                    "start quiz", "begin quiz",
                ],
            ) as socket:
                reader_done = threading.Event()
                connection_open = threading.Event()

                def _on_open(_event: object) -> None:
                    connection_open.set()

                def _on_message(msg: object) -> None:
                    """Process only transcript result events from the socket."""
                    if not isinstance(msg, ListenV1Results):
                        if DEBUG_MODE:
                            print(f"[DG:other] type={type(msg).__name__}", flush=True)
                        return
                    try:
                        alt = msg.channel.alternatives[0]
                        transcript: str = alt.transcript
                        words = alt.words or []
                        avg_conf = (
                            sum(w.confidence for w in words) / len(words)
                            if words else 1.0
                        )
                        _process_transcript_event(
                            transcript, avg_conf, bool(msg.is_final)
                        )
                    except Exception:
                        log.exception("Error processing transcript.")

                def _on_error(exc: object) -> None:
                    log.warning("Deepgram connection error: %s", exc)

                def _reader() -> None:
                    try:
                        socket.start_listening()
                    except Exception as exc:
                        log.warning("Deepgram reader exited: %s", exc)
                    finally:
                        reader_done.set()

                # Handlers must be registered before start_listening(), whose
                # first action is to emit EventType.OPEN.
                socket.on(EventType.OPEN, _on_open)
                socket.on(EventType.MESSAGE, _on_message)
                socket.on(EventType.ERROR, _on_error)
                reader_thread = threading.Thread(
                    target=_reader, daemon=True, name="dg-reader"
                )
                reader_thread.start()

                # Do not read or send microphone bytes until the socket's
                # listening loop has signalled a usable connection.
                if not connection_open.wait(timeout=5):
                    log.error("Deepgram connection did not open within 5 seconds.")
                    try:
                        socket.send_close_stream()
                    except Exception:
                        pass
                    reader_thread.join(timeout=3)
                    continue

                # --- microphone ------------------------------------------

                pa = pyaudio.PyAudio()

                device_index = MIC_DEVICE_INDEX
                if device_index is None:
                    print(
                        "voice_io: WARNING — MIC_DEVICE_INDEX is not set; "
                        "opening system default input device. "
                        "Set VOICE_IO_MIC_INDEX to pin a specific microphone.",
                        flush=True,
                    )

                audio_stream, active_index, _fell_back = _open_input_stream(
                    pa, device_index, chunk_frames
                )

                try:
                    _dev_name = pa.get_device_info_by_index(active_index)["name"]
                except Exception:
                    _dev_name = "unknown"

                _pin_note = (
                    f"pinned via VOICE_IO_MIC_INDEX={device_index}"
                    if device_index is not None and not _fell_back
                    else "system default (no VOICE_IO_MIC_INDEX set)"
                    if device_index is None
                    else f"VOICE_IO_MIC_INDEX={device_index} failed, fell back to default"
                )
                print(
                    f"voice_io: mic open  device=#{active_index} (\"{_dev_name}\")  {_pin_note}",
                    flush=True,
                )
                log.info(
                    "Microphone open on device #%s (%s); streaming to Deepgram.",
                    active_index, _dev_name,
                )
                _status["connected"] = True
                _status["mic_index"] = active_index
                _status["mic_name"] = _dev_name
                backoff = 1.0

                # --- send loop: mic → Deepgram ----------------------------
                # While paused (e.g. TTS playing) we still drain the mic
                # buffer every tick to prevent driver overflow, and send a
                # KeepAlive every 3 s so Deepgram doesn't close the idle
                # socket.  Deepgram closes connections after ~10 s of
                # silence; a full debrief runs 15-18 s.
                _last_keepalive = time.monotonic()
                while not _stop_event.is_set() and not reader_done.is_set():
                    try:
                        data = audio_stream.read(chunk_frames, exception_on_overflow=False)
                    except OSError as exc:
                        log.warning("Microphone read error: %s", exc)
                        break

                    if _mic_paused:
                        if time.monotonic() - _last_keepalive >= 3.0:
                            try:
                                socket.send_keep_alive()
                            except Exception as exc:
                                log.warning("send_keep_alive failed: %s", exc)
                            _last_keepalive = time.monotonic()
                        else:
                            time.sleep(0.05)
                        continue

                    socket.send_media(data)

                # --- graceful close --------------------------------------

                try:
                    socket.send_close_stream()
                except Exception:
                    pass
                reader_thread.join(timeout=3)

        except Exception as _exc:
            _status["connected"] = False
            _status["last_error"] = str(_exc) or type(_exc).__name__
            log.exception("Deepgram listener loop error; reconnecting in %.1fs.", backoff)
        finally:
            _status["connected"] = False
            if audio_stream is not None:
                try:
                    audio_stream.stop_stream()
                    audio_stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

        if not _stop_event.is_set():
            log.info("Reconnecting Deepgram STT in %.1fs …", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ---------------------------------------------------------------------------
# Deepgram TTS — real implementation
# ---------------------------------------------------------------------------


def _for_tts(text: str) -> str:
    """Text actually sent to Deepgram. Aura treats the last word of a short line as optional and
    drops it; a trailing ellipsis is a pause, so 'cat' is no longer the last thing it generates."""
    t = (text or "").rstrip()
    if not t:
        return text
    if t.endswith(TTS_TAIL):
        return t
    return t + TTS_TAIL


def _fix_wav_header(data: bytes) -> bytes:
    """Deepgram's streamed WAV header claims about 2 GB; rewrite the size fields to the real length."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    at = data.find(b"data", 12)
    if at < 0:
        return data
    body = len(data) - (at + 8)
    out = bytearray(data)
    out[4:8] = (len(data) - 8).to_bytes(4, "little")
    out[at + 4:at + 8] = body.to_bytes(4, "little")
    return bytes(out)


def _pad_wav_silence(data: bytes, seconds: float = WAV_SILENCE_SECONDS) -> bytes:
    """Append quiet samples so a player that drops the last buffer cannot eat the last consonant."""
    data = _fix_wav_header(data)
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            nch, sw, rate, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            comptype = wf.getcomptype()
            pcm = wf.readframes(nframes)
        if comptype != "NONE" or rate <= 0 or sw <= 0 or nch <= 0:
            return data
        extra = b"\x00" * int(rate * seconds) * sw * nch
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nch)
            out.setsampwidth(sw)
            out.setframerate(rate)
            out.writeframes(pcm + extra)
        return buf.getvalue()
    except (wave.Error, EOFError, OverflowError):
        return data


def _deepgram_speak(text: str) -> None:
    try:
        from deepgram import DeepgramClient  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        log.error("deepgram-sdk is not installed.  Run: pip install 'deepgram-sdk==7.9.0'")
        return
    except ImportError as exc:
        log.error(
            "deepgram-sdk API mismatch: %s — check: pip show deepgram-sdk", exc
        )
        return

    if not DEEPGRAM_API_KEY:
        log.error("DEEPGRAM_API_KEY is not set.")
        return

    try:
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        # Join the stream while the SDK's response is still open, then pad: the phone speaker
        # intercepts _play_audio_stream, so any silence written only inside pyaudio never reaches
        # the phone (that is why 'cat' was still heard as 'ca' after the laptop-side waits).
        raw = b"".join(
            client.speak.v1.audio.generate(
                text=_for_tts(text),
                model=DEEPGRAM_TTS_MODEL,
                encoding="linear16",  # request PCM; default is MP3 which wave.open() rejects
                container="wav",
            )
        )
        _play_audio_stream([_pad_wav_silence(raw)])
    except Exception as exc:
        log.error("Deepgram TTS failed: %s", exc)


# ---------------------------------------------------------------------------
# ElevenLabs TTS — real implementation
# ---------------------------------------------------------------------------


def _elevenlabs_speak(text: str) -> None:
    try:
        import pyaudio  # type: ignore[import-untyped]
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:
        log.error("Missing dependency for ElevenLabs TTS: %s", exc)
        return

    if not ELEVENLABS_API_KEY:
        log.error("ELEVENLABS_API_KEY is not set.")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15, stream=True)
        resp.raise_for_status()
    except Exception as exc:
        log.error("ElevenLabs TTS request failed: %s", exc)
        return

    _play_mp3_stream(resp.iter_content(chunk_size=4096))


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------


def _play_audio_stream(chunks) -> None:
    """Play raw PCM / WAV bytes from an iterable of chunks."""
    try:
        import pyaudio  # type: ignore[import-untyped]
        import wave  # stdlib
    except ImportError as exc:
        log.error("pyaudio not available for playback: %s", exc)
        return

    data = _fix_wav_header(b"".join(chunks))
    buf = io.BytesIO(data)
    buf.seek(0)

    try:
        with wave.open(buf) as wf:
            pa = pyaudio.PyAudio()
            try:
                pa.get_default_output_device_info()
            except Exception as exc:
                # No output device at all (e.g. it just disconnected): calling
                # pa.open() here is the same native-crash risk as the input
                # side, so refuse the call instead of attempting it.
                log.error("No default output device available for playback: %s", exc)
                pa.terminate()
                return
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            try:
                nframes, rate = wf.getnframes(), float(wf.getframerate() or 1)
                started = time.monotonic()
                chunk = wf.readframes(1024)
                while chunk:
                    stream.write(chunk)
                    chunk = wf.readframes(1024)
                remaining = (nframes / rate) - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(min(remaining, 8.0))  # never block the tutor on a lying WAV header
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()
    except Exception as exc:
        log.error("Audio playback failed: %s", exc)


def _play_mp3_stream(chunks) -> None:
    """Decode and play an MP3 byte stream (used for ElevenLabs output)."""
    try:
        import pyaudio  # type: ignore[import-untyped]
        from pydub import AudioSegment  # type: ignore[import-untyped]
        from pydub.playback import play  # type: ignore[import-untyped]
    except ImportError as exc:
        log.error(
            "Missing dependency for MP3 playback: %s. "
            "Install with: pip install pydub pyaudio",
            exc,
        )
        return

    buf = io.BytesIO(b"".join(chunks))
    buf.seek(0)
    try:
        seg = AudioSegment.from_mp3(buf)
        play(seg)
    except Exception as exc:
        log.error("MP3 playback failed: %s", exc)


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


def _mock_speak(text: str, mode: str) -> None:
    print(f"[SPEAK/Deepgram/{mode}] {text}")
    time.sleep(0.05)  # simulate minimal latency so callers aren't surprised


def _mock_listener_loop() -> None:
    """In mock mode, accept typed commands from stdin for testing."""
    print(
        "[VOICE_IO MOCK] Listener active. "
        "Type a command and press Enter to simulate speech recognition.\n"
        f"  Valid commands: {sorted(set(_COMMAND_MAP.values()))}\n"
        "  Type 'quit' to exit.\n"
    )
    import sys

    while not _stop_event.is_set():
        try:
            # Non-blocking stdin poll so we can respect _stop_event.
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            text = line.rstrip("\n")
            if text == "quit":
                _stop_event.set()
                break
            command = _process_transcript_event(text, 1.0, True)
            if command:
                print(f"[VOICE_IO MOCK] Command recognised: {command!r}")
            else:
                print(f"[VOICE_IO MOCK] Unmatched: {_normalize_transcript(text)!r}")
        except (EOFError, KeyboardInterrupt):
            break
