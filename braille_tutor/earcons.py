"""Short, gentle sounds ("earcons") that tell the learner how they did without a word: a bright rising phrase for right, a soft falling
one for not quite (never a buzzer: this is for someone learning), a small blip when something is locked in, a fanfare for finishing.

They are made here as 16-bit mono WAV bytes and played through voice_io's own player (`_play_audio_stream`), so they go wherever the
tutor's voice goes (the laptop, or the phone when the phone is the speaker). No sound files are needed.
"""
from __future__ import annotations

import io
import wave
from typing import Callable, Optional

import numpy as np

RATE = 24000
LEVEL = 0.32  # of full scale: clearly audible, never loud

# note -> Hz (equal temperament, A4 = 440)
NOTE = {"G3": 196.0, "C4": 261.63, "E4": 329.63, "G4": 392.0, "A4": 440.0, "C5": 523.25, "E5": 659.25, "G5": 783.99, "A5": 880.0,
        "C6": 1046.5, "E6": 1318.5}

# kind -> [(note, seconds)]
PHRASES = {
    "ready": [("C5", 0.11), ("E5", 0.16)],
    "correct": [("C5", 0.08), ("E5", 0.08), ("G5", 0.08), ("C6", 0.20)],
    "wrong": [("E4", 0.16), ("C4", 0.26)],  # soft and falling: "not quite", not "no"
    "lost": [("G4", 0.10), ("E4", 0.10), ("C4", 0.20)],
    "locked": [("A5", 0.07)],
    "done": [("C5", 0.10), ("E5", 0.10), ("G5", 0.10), ("C6", 0.14), ("G5", 0.09), ("C6", 0.10), ("E6", 0.42)],
}
KINDS = tuple(PHRASES)


def _note(freq: float, seconds: float) -> np.ndarray:
    """One soft note: a sine with a little of its octave for warmth, and a quick rise and a smooth fall so it never clicks."""
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    wave_ = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    attack = np.minimum(1.0, t / 0.008)
    release = np.minimum(1.0, (seconds - t) / (0.6 * seconds))
    return wave_ * attack * np.clip(release, 0, 1) ** 1.5


def tone_samples(kind: str) -> np.ndarray:
    if kind not in PHRASES:
        raise ValueError(f"unknown tone {kind!r}; choose from {', '.join(KINDS)}")
    out = np.concatenate([_note(NOTE[n], s) for n, s in PHRASES[kind]])
    return LEVEL * out / max(1.0, float(np.abs(out).max()))


def tone_wav(kind: str) -> bytes:
    """The tone as a complete WAV file (16-bit mono)."""
    pcm = (tone_samples(kind) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Earcons:
    """Plays the tones through a voice_io-like module. Never raises: a tone that cannot play is not worth stopping a lesson for."""

    def __init__(self, voice, enabled: bool = True, log: Optional[Callable[[str], None]] = None):
        self.voice, self.enabled, self.log = voice, enabled, log or (lambda text: None)
        self.played: list = []  # for tests and displays

    def play(self, kind: str) -> None:
        if not self.enabled:
            return
        self.played.append(kind)
        try:
            if getattr(self.voice, "MOCK_MODE", False):
                print(f"[TONE] {kind}", flush=True)  # like the mock voice: shown, not played
                return
            data = tone_wav(kind)
            pause = getattr(self.voice, "pause_listening", None)
            resume = getattr(self.voice, "resume_listening", None)
            if pause:
                pause()  # the microphone must not hear it as speech
            try:
                self.voice._play_audio_stream([data])
            finally:
                if resume:
                    resume()
        except Exception as e:  # noqa: BLE001
            self.log(f"tone {kind!r} could not be played: {e}")
