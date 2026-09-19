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

QUICK START
-----------
    from voice_io import register_command, start_listening, speak

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
from typing import Callable

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so import errors surface with a clear
# message pointing at the install step.
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOCK_MODE: bool = os.getenv("VOICE_IO_MOCK", "0").strip() in ("1", "true", "yes")

DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# Deepgram TTS voice — a clear, neutral voice suitable for accessibility.
DEEPGRAM_TTS_MODEL: str = "aura-asteria-en"

# ElevenLabs voice ID for the debrief — "Rachel" (calm, warm, expressive).
# Override via ELEVENLABS_VOICE_ID env var.
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Deepgram STT model.
DEEPGRAM_STT_MODEL: str = "nova-2"

# Minimum confidence for a recognised command (Deepgram word confidence).
COMMAND_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("COMMAND_CONFIDENCE_THRESHOLD", "0.75")
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
# Multiple phrasings can map to the same command.
_COMMAND_MAP: dict[str, str] = {
    "repeat": "repeat",
    "hint": "hint",
    "found it": "found it",
    "i found it": "found it",
    "next": "next",
    "stop": "stop",
    "start quiz": "start quiz",
    "begin quiz": "start quiz",
}

# Pre-compiled pattern that matches any command phrase anywhere in a transcript.
_COMMAND_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_COMMAND_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_callbacks: dict[str, list[Callable[[], None]]] = {}
_listening_active: bool = False
_mic_paused: bool = False
_listener_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()

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
    for cb in handlers:
        try:
            cb()
        except Exception:
            log.exception("Error in handler for command %r", command)


# ---------------------------------------------------------------------------
# Public API — lifecycle
# ---------------------------------------------------------------------------


def start_listening() -> None:
    """Start continuous microphone capture and Deepgram streaming STT."""
    global _listening_active, _listener_thread, _stop_event

    if _listening_active:
        log.warning("start_listening() called but already listening; ignoring.")
        return

    _stop_event = threading.Event()
    _listening_active = True

    if MOCK_MODE:
        _listener_thread = threading.Thread(
            target=_mock_listener_loop, daemon=True, name="voice-listener"
        )
    else:
        _listener_thread = threading.Thread(
            target=_deepgram_listener_loop, daemon=True, name="voice-listener"
        )

    _listener_thread.start()
    log.info("Voice listener started (mock=%s)", MOCK_MODE)


def stop_listening() -> None:
    """Stop the microphone stream and close the Deepgram connection."""
    global _listening_active

    if not _listening_active:
        return

    _stop_event.set()
    if _listener_thread is not None:
        _listener_thread.join(timeout=5)
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
    """Synthesise *text* and play it.

    Automatically pauses the microphone before speaking and resumes it
    after, so the system never transcribes its own narration.

    Args:
        text: The text to speak.
        mode: "normal" → Deepgram TTS; "debrief" → ElevenLabs TTS.
    """
    pause_listening()
    try:
        if MOCK_MODE:
            _mock_speak(text, mode)
        elif mode == "debrief":
            _elevenlabs_speak(text)
        else:
            _deepgram_speak(text)
    except Exception:
        log.exception("speak() failed for mode=%r; text=%r", mode, text)
    finally:
        resume_listening()


def speak_debrief(accuracy: float, missed_cells: list[str]) -> None:
    """Generate and speak an end-of-session debrief via ElevenLabs.

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


def _deepgram_listener_loop() -> None:
    """Background thread: streams mic audio to Deepgram and dispatches commands."""
    try:
        import pyaudio  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        log.error("pyaudio is not installed.  Run: pip install pyaudio")
        return

    try:
        from deepgram import DeepgramClient  # type: ignore[import-untyped]
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
                model=DEEPGRAM_STT_MODEL,
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                endpointing=500,
                vad_events=True,
                smart_format=True,
                interim_results=False,
            ) as socket:

                # --- reader thread: receives transcript messages ----------

                reader_done = threading.Event()

                def _reader() -> None:
                    try:
                        for msg in socket:
                            if _mic_paused:
                                continue
                            if not isinstance(msg, ListenV1Results):
                                continue
                            if not msg.is_final:
                                continue
                            try:
                                alt = msg.channel.alternatives[0]
                                transcript: str = alt.transcript.strip().lower()
                                if not transcript:
                                    continue

                                words = alt.words or []
                                avg_conf = (
                                    sum(w.confidence for w in words) / len(words)
                                    if words else 1.0
                                )

                                if avg_conf < COMMAND_CONFIDENCE_THRESHOLD:
                                    log.debug(
                                        "Low-confidence (%.2f): %r — skipping.",
                                        avg_conf, transcript,
                                    )
                                    continue

                                match = _COMMAND_PATTERN.search(transcript)
                                if match:
                                    phrase = match.group(1).lower()
                                    command = _COMMAND_MAP[phrase]
                                    log.info(
                                        "Command recognised: %r (conf=%.2f)",
                                        command, avg_conf,
                                    )
                                    _fire_command(command)
                                else:
                                    log.debug(
                                        "Unmatched speech (%.2f): %r",
                                        avg_conf, transcript,
                                    )
                            except Exception:
                                log.exception("Error processing transcript.")
                    except Exception as exc:
                        log.warning("Deepgram reader exited: %s", exc)
                    finally:
                        reader_done.set()

                reader_thread = threading.Thread(
                    target=_reader, daemon=True, name="dg-reader"
                )
                reader_thread.start()

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

                open_kwargs: dict = dict(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=chunk_frames,
                )
                if device_index is not None:
                    open_kwargs["input_device_index"] = device_index

                try:
                    audio_stream = pa.open(**open_kwargs)
                except OSError as exc:
                    print(
                        f"voice_io: ERROR — could not open device index {device_index}: {exc}. "
                        "Falling back to system default.",
                        flush=True,
                    )
                    open_kwargs.pop("input_device_index", None)
                    audio_stream = pa.open(**open_kwargs)

                active_index = (
                    device_index
                    if device_index is not None
                    else pa.get_default_input_device_info()["index"]
                )
                log.info(
                    "Microphone open on device #%s; streaming to Deepgram.",
                    active_index,
                )
                backoff = 1.0

                # --- send loop: mic → Deepgram ----------------------------

                while not _stop_event.is_set() and not reader_done.is_set():
                    if _mic_paused:
                        time.sleep(0.05)
                        continue
                    try:
                        data = audio_stream.read(chunk_frames, exception_on_overflow=False)
                        socket.send_media(data)
                    except OSError as exc:
                        log.warning("Microphone read error: %s", exc)
                        break

                # --- graceful close --------------------------------------

                try:
                    socket.send_close_stream()
                except Exception:
                    pass
                reader_thread.join(timeout=3)

        except Exception:
            log.exception("Deepgram listener loop error; reconnecting in %.1fs.", backoff)
        finally:
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
        chunks = client.speak.v1.audio.generate(
            text=text,
            model=DEEPGRAM_TTS_MODEL,
            container="wav",
        )
        _play_audio_stream(chunks)
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

    buf = io.BytesIO(b"".join(chunks))
    buf.seek(0)

    try:
        with wave.open(buf) as wf:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            try:
                chunk = wf.readframes(1024)
                while chunk:
                    stream.write(chunk)
                    chunk = wf.readframes(1024)
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
    backend = "ElevenLabs" if mode == "debrief" else "Deepgram"
    print(f"[SPEAK/{backend}] {text}")
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
            text = line.strip().lower()
            if text == "quit":
                _stop_event.set()
                break
            if _mic_paused:
                print("[VOICE_IO MOCK] Mic paused; ignoring input.")
                continue
            match = _COMMAND_PATTERN.search(text)
            if match:
                phrase = match.group(1).lower()
                command = _COMMAND_MAP[phrase]
                print(f"[VOICE_IO MOCK] Command recognised: {command!r}")
                _fire_command(command)
            else:
                print(f"[VOICE_IO MOCK] Unmatched: {text!r}")
        except (EOFError, KeyboardInterrupt):
            break
