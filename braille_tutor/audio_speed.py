"""Speak more slowly (or faster) without changing the voice's pitch: a time-stretch of the finished audio.

Deepgram's current voice has no speed setting (the API refuses one), and slower speech is one of the most useful things for a new learner or
an older listener. So the audio is stretched here, by WSOLA (waveform-similarity overlap-add): the speech is cut into short overlapping
frames, laid down further apart (slower) or closer together (faster), and each frame is nudged a few milliseconds to line up with what came
before, so the result stays smooth and the pitch does not drop. Works at 0.6x to 1.5x. Applied at the same seam as the phone speaker
(`voice._play_audio_stream`), so it affects everything the tutor says on the laptop or the phone, and NOT the little sounds (earcons).
"""
from __future__ import annotations

import contextlib
import io
import threading
import wave
from typing import Callable, Optional

import numpy as np

MIN_SPEED, MAX_SPEED = 0.6, 1.5
FRAME_MS, SEARCH_MS = 30.0, 8.0
_local = threading.local()


@contextlib.contextmanager
def raw():
    """Inside this block audio is played exactly as given (the earcons use it: a tone must not be stretched like speech)."""
    _local.raw = True
    try:
        yield
    finally:
        _local.raw = False


def clamp(speed: float) -> float:
    return float(min(MAX_SPEED, max(MIN_SPEED, speed)))


def time_stretch(x: np.ndarray, speed: float, rate: int) -> np.ndarray:
    """`x` (mono float samples) played at `speed` times its natural speed: 0.8 is slower (25% longer). Same pitch."""
    speed = clamp(speed)
    frame = int(rate * FRAME_MS / 1000) // 2 * 2
    if abs(speed - 1.0) < 0.01 or len(x) < 3 * frame:
        return x
    hop_out = frame // 2
    hop_in = hop_out * speed
    tol = int(rate * SEARCH_MS / 1000)
    window = np.hanning(frame + 1)[:frame].astype(np.float32)  # periodic Hann: overlapping frames at 50% sum to one
    x = np.concatenate([np.zeros(tol, np.float32), x.astype(np.float32), np.zeros(frame + 2 * tol, np.float32)])
    n_frames = int((len(x) - frame - 2 * tol) / hop_in)
    out = np.zeros((n_frames + 1) * hop_out + frame, np.float32)
    prev = tol  # where the previous frame was taken from
    out[:frame] += x[prev:prev + frame] * window
    for k in range(1, n_frames):
        natural = x[prev + hop_out:prev + hop_out + frame]  # what would have followed the previous frame
        centre = tol + int(round(k * hop_in))
        lo = max(0, centre - tol)
        region = x[lo:lo + frame + 2 * tol]
        if len(region) < frame + 2 * tol or len(natural) < frame:
            break
        score = np.correlate(region, natural, mode="valid")  # how well each candidate start lines up
        norm = np.sqrt(np.convolve(region * region, np.ones(frame, np.float32), mode="valid") + 1e-9)
        pos = lo + int(np.argmax(score / norm))
        out[k * hop_out:k * hop_out + frame] += x[pos:pos + frame] * window
        prev = pos
    end = int((len(x) - frame - 2 * tol) / speed)
    return out[:max(end, frame)]


def stretch_wav(data: bytes, speed: float) -> bytes:
    """A WAV (16-bit mono, as Deepgram sends) at `speed`; anything else, or speed 1, comes back unchanged."""
    speed = clamp(speed)
    if abs(speed - 1.0) < 0.01:
        return data
    try:
        from phonelink import fix_wav_header  # Deepgram streams a header that claims hours of audio: use the real sizes

        with wave.open(io.BytesIO(fix_wav_header(data))) as w:
            if (w.getnchannels(), w.getsampwidth()) != (1, 2):
                return data
            rate, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    except (wave.Error, EOFError, ValueError):
        return data
    y = time_stretch(pcm.astype(np.float32) / 32768.0, speed, rate)
    out = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())
    return buf.getvalue()


def install_speech_speed(voice, get_speed: Callable[[], float]) -> Callable:
    """Make everything voice_io plays go through `stretch_wav` at the speed `get_speed()` gives at that moment. Install it AFTER the phone
    speaker so the stretched audio is what goes to the phone. Returns a function that puts things back."""
    if not callable(getattr(voice, "_play_audio_stream", None)):
        raise RuntimeError("voice_io no longer has _play_audio_stream: cannot change the speech speed")
    original = voice._play_audio_stream

    def play(chunks) -> None:
        data = b"".join(chunks)
        if not getattr(_local, "raw", False):
            data = stretch_wav(data, get_speed())
        original([data])

    voice._play_audio_stream = play

    def restore() -> None:
        voice._play_audio_stream = original
    return restore
