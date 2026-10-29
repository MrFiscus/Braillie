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
        Synthesise and play text.  Blocks until playback finishes (or is cut
        off).  The mic stays live while it plays, so the learner can interrupt:
        any command (or "quiet", "wait", "stop") stops the line, and what the
        mic hears of the line itself is ignored.  VOICE_IO_BARGE_IN=0 mutes the
        mic while speaking instead.

    interrupt() -> bool
        Cut off the line being spoken.  interrupt_count() tells callers whether
        a cut happened since they queued a line, so they can drop stale lines.

        mode is "normal" or "debrief"; both are spoken by Deepgram.

    speak_debrief(accuracy: float, missed_cells: list[str]) -> None
        Generate a short debrief script whose tone shifts with performance,
        then speak it (equivalent to calling speak(…, "debrief")).

        accuracy     — 0.0–1.0 (e.g. 0.82 → 82 %)
        missed_cells — list of braille cell labels that were most missed
                       (e.g. ["A", "B", "SH"])

ENVIRONMENT VARIABLES
---------------------
    DEEPGRAM_API_KEY    — Deepgram secret key
    DEEPGRAM_TTS_MODEL  — the voice (default aura-2-thalia-en)

    VOICE_IO_MOCK=1     — Offline/mock mode: prints instead of hitting APIs.
                          Useful for dev/CI without spending credits.
    VOICE_IO_DEBUG=1    — Print every raw Deepgram transcript + match result.
    COMMAND_CONFIDENCE_THRESHOLD — Float 0–1, default 0.72, applied to the
                          command's own words.  Lower = more sensitive; raise
                          if you get false fires in noisy rooms.
    VOICE_IO_NOISE_GATE_DB=6 — How far (dB) speech must stand out above the
                          room's background to count.  Raise it in a loud room
                          where other voices get through; lower it if the
                          learner speaks softly and is missed.
    VOICE_IO_CONDITION=0  — Send the microphone as it is (no hum filter, no
                          gain for a quiet microphone).
    VOICE_IO_BARGE_IN=0   — Mute the microphone while the app speaks.

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

import collections
import difflib
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

try:  # only for listening in noise (levels and the microphone clean-up); without it those are simply skipped
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

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

# Deepgram TTS: Aura 2 (more natural than Aura 1, and it says the last word of a short line in full; checked by synthesising lines and
# transcribing them back). DEEPGRAM_TTS_MODEL in .env picks another voice. Aura 1 ("aura-asteria-en" and the like) drops the last
# ~200-300 ms of a short line ("cat" -> "ca"), so for an Aura 1 voice a pause is added after the last word (TTS_TAIL).
DEEPGRAM_TTS_MODEL: str = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
TTS_TAIL = "..."  # synthesised as a pause, not a word: if anything still eats the end, this is what goes
WAV_SILENCE_SECONDS = 0.45  # extra quiet at the end of the file, for phone players that clip the last buffer


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

# Barge-in: the microphone stays live while the app speaks, so the learner can cut it off ("quiet", "wait",
# "stop", or any command). What the microphone hears of the app's own voice is recognised and dropped (see
# _is_echo). VOICE_IO_BARGE_IN=0 goes back to muting the microphone for every line, as before.
BARGE_IN: bool = os.getenv("VOICE_IO_BARGE_IN", "1").strip() not in ("0", "false", "no")
ECHO_TAIL_SECONDS = 1.5  # after a line ends, the room (and Deepgram) may still be giving back its last words
PHONE_ECHO_TAIL_SECONDS = 3.5  # the same, when the phone plays the line and listens: its echo comes back later, over the network
# Set by the phone link (phonelink.install_phone_mic) while the sound heard comes from the phone's microphone. The laptop's echo
# canceller cannot take out a line the PHONE played (it never heard it, and the network delay keeps changing), so then the app's
# own words are taken out one by one (_without_echo), and only a real command cuts the app off: three stray words, which may well
# be its own voice misheard, do not.
MIC_FROM_PHONE: bool = False
SPEAKER_ON_PHONE: bool = False  # set by phonelink.install_phone_speaker while the lines are played on the phone (same problem)
CUT_OFF_BY = ("hush", "stop")  # with the echo not cancellable, only these cut a line off ("read", say, is also a word Braillo says)


def _echo_unreliable() -> bool:
    """Is the app's own voice coming back in a way the laptop's echo canceller cannot take out (played on, or heard through, the
    phone)? Then its words are taken out of what is heard one by one, and only "stop" or "quiet" cuts a line off."""
    return MIC_FROM_PHONE or SPEAKER_ON_PHONE
TALK_OVER_WORDS = 3  # fewer of the learner's own words than this while it talks: maybe an answer, maybe its own voice (OVER_SPEECH)
OVER_SPEECH = "heard over speech"  # the reason given for a short answer heard while the app talked (see _classify_transcript)
TALK_OVER_CONFIDENCE = 0.55  # a command said over the app's voice comes back less sure than one said in quiet: it is allowed less

# A spoken sentence often reaches us in pieces (a short pause ends a piece). Commands are acted on at the end of
# their piece, so they stay quick; anything else waits for the whole utterance (Deepgram's UtteranceEnd, after
# UTTERANCE_END_MS of no words), so a question with a pause in it is not cut in two.
UTTERANCE_END_MS: int = 1000
UTTERANCE_HOLD_SECONDS = 2.5  # ...and is let go after this long even if UtteranceEnd never arrives
QUICK_ANSWER_WORDS = 3  # an utterance this short is taken as soon as the speaker pauses (an answer, not the start of a question)

# Listening in a noisy room.
#  * Speech is only taken as meant for the app if it stands out from the room: at least NOISE_GATE_DB above the background
#    (the quiet between words), and not FAR_VOICE_DB quieter than the learner usually speaks (learned from the commands they
#    give). Talk across the room and the TV are quieter at the microphone than someone sitting at it.
#  * A command's confidence is that of its own words, so a clearly spoken "hint" is not sunk by murmur around it.
#  * A short utterance that sounds nearly like a command ("hent", "nex") is still taken for it (FUZZY_RATIO).
#  * With set_defer_long_utterances(True) (the app can work out free speech itself), a single command word in the middle of a
#    long sentence ("I need to stop by the shop") is not run: the sentence is passed on as heard, for the app to judge.
#  * The laptop microphone's signal is cleaned before it is sent: low hum and rumble filtered out, and a quiet microphone
#    gently turned up. (The phone's browser does its own noise suppression.)
NOISE_GATE_DB: float = float(os.getenv("VOICE_IO_NOISE_GATE_DB", "6"))
FAR_VOICE_DB: float = 15.0
LONG_UTTERANCE_WORDS = 6
FUZZY_RATIO = 0.8
CONDITION_AUDIO: bool = os.getenv("VOICE_IO_CONDITION", "1").strip() not in ("0", "false", "no")
HIGHPASS_HZ = 100.0
AGC_TARGET_DBFS = -26.0  # speech is turned up toward this level...
AGC_MAX_GAIN_DB = 12.0  # ...by at most this much (never down: a loud microphone is left as it is)

# Echo cancellation: what the laptop's speakers play (the app's voice and sounds) is taken back out of what its microphone hears,
# before Deepgram gets it, so the learner can be heard while the app is talking. Without it the app's own voice, loud and close,
# drowns the learner's and Deepgram writes down only the app. Uses WebRTC's audio processing (the `livekit` package; with it
# missing, the plain clean-up above is used instead), which also removes steady background noise. VOICE_IO_AEC=0 turns it off.
AEC: bool = os.getenv("VOICE_IO_AEC", "1").strip() not in ("0", "false", "no")
AEC_DELAY_MS = 60  # a first guess at how long the speakers take to reach the microphone; the canceller works out the rest
LEAK_SHARE = 0.8  # with the canceller on: a transcript this much made of the line's own words is the line leaking through

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
    "i'm stuck": "hint",  # (at once, no need to ask Claude what it meant)
    "im stuck": "hint",
    "i am stuck": "hint",
    "stuck": "hint",
    "stuck on this one": "hint",  # (longer than "this one", which means found it)
    "i'm stuck on this one": "hint",
    "im stuck on this one": "hint",
    "stuck on this": "hint",
    "i can't find it": "hint",
    "i cant find it": "hint",
    "where is it": "hint",
    "hence": "hint",          # confirmed misread of "hint"
    "hen": "hint",            # confirmed misread of "hint"
    "hints": "hint",          # confirmed misread of "hint"
    "health": "hint",         # confirmed misread of "hint"
    # --- found it --------------------------------------------------------
    "found it": "found it",
    "i found it": "found it",
    "got it": "found it",     # natural shorthand: "got it" = located the cell
    "i got it": "found it",
    # ...and what people say while pointing at their answer, without thinking of a command
    "this one": "found it",
    "that one": "found it",
    "this is it": "found it",
    "is this it": "found it",
    "is it this one": "found it",
    "this is the one": "found it",
    "here it is": "found it",
    "it's here": "found it",
    "its here": "found it",
    "right here": "found it",
    "over here": "found it",
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
    "i'm ready": "start quiz",  # (a lesson waiting for its sheet: "I'm ready", I have it)
    "i am ready": "start quiz",
    "im ready": "start quiz",
    "got the sheet": "start quiz",
    "i have the sheet": "start quiz",
    "i've got the sheet": "start quiz",
    "sheet is ready": "start quiz",
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
    "home": "menu",
    "go home": "menu",
    "home page": "menu",
    "home screen": "menu",
    # --- getting around: lessons, your page, signing out (from anywhere) ---
    "next lesson": "next lesson",
    "skip lesson": "next lesson",
    "skip this lesson": "next lesson",
    "previous lesson": "previous lesson",
    "last lesson": "previous lesson",
    "lesson before": "previous lesson",
    "go back a lesson": "previous lesson",
    "my page": "my page",
    "my profile": "my page",
    "profile": "my page",
    "my progress": "my page",
    "account": "my page",
    "my account": "my page",
    "go back": "back",  # the website: the page before (see VoiceControl)
    "back": "back",
    "go back a page": "back",
    "previous page": "back",
    "scroll down": "scroll down",
    "page down": "scroll down",
    "scroll up": "scroll up",
    "page up": "scroll up",
    "go to the top": "top",
    "top of the page": "top",
    "back to the top": "top",
    "back to top": "top",
    "sign out": "sign out",
    "log out": "sign out",
    "sign me out": "sign out",
    "log me out": "sign out",
    # --- accessibility: ask what can be said, and how fast to go -----------
    "help": "help",  # ("help me" is longer and still means hint)
    "what can i say": "help",
    "what are my options": "help",
    "what's on this page": "help",
    "whats on this page": "help",
    "what can i do here": "help",
    "read the buttons": "help",
    "more options": "more help",  # the whole list (help itself is short)
    "all options": "more help",
    "everything else": "more help",
    "list everything": "more help",
    "more help": "more help",
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
    "exit": "menu",  # (leaving an activity: back to choosing, like "menu")
    "leave": "menu",
    "quit lesson": "menu",
    # (other ways people say it: nobody has to remember the one word; all go back to the menu, with a word on how it went)
    "take me back": "menu",
    "take me home": "menu",
    "back to the menu": "menu",
    "back to menu": "menu",
    "go to the menu": "menu",
    "go to menu": "menu",
    "i want to go back": "menu",
    "let's go back": "menu",
    "lets go back": "menu",
    "something else": "menu",
    "do something else": "menu",
    "different mode": "menu",
    "change activity": "menu",
    "stop the lesson": "menu",
    "stop lesson": "menu",
    "end the lesson": "menu",
    "end lesson": "menu",
    "end this": "menu",
    "let's stop": "menu",
    "lets stop": "menu",
    "i want to stop": "menu",
    "that's enough": "menu",
    "thats enough": "menu",
    "that is enough": "menu",
    "that is enough for today": "menu",
    "enough for today": "menu",
    "i've had enough": "menu",
    "ive had enough": "menu",
    "done for today": "menu",
    "done for the day": "menu",
    "i'm done for today": "menu",
    "i'm done for the day": "menu",
    "im done for the day": "menu",
    "call it a day": "menu",
    "let's call it a day": "menu",
    "no more": "menu",
    "exit lesson": "menu",
    "leave lesson": "menu",
    "get me out of here": "menu",
    "get me out": "menu",
    "quit quiz": "menu",
    "stop quiz": "menu",
    "stop the quiz": "menu",
    "end quiz": "menu",
    "stop reading": "menu",
    "i'm done with this": "menu",
    "im done with this": "menu",
    "finish": "stop",
    "finished": "stop",
    "i'm finished": "stop",
    "im finished": "stop",
    "i'm done": "stop",
    "im done": "stop",        # apostrophe-free variant
    "done": "stop",
    # --- stop talking (cuts off the line being spoken; nothing else changes) ---
    "quiet": "hush",
    "be quiet": "hush",
    "hush": "hush",
    "shh": "hush",
    "shush": "hush",
    "stop talking": "hush",
    "wait": "hush",
    "hold on": "hush",
    "hang on": "hush",
    "pause": "hush",
    "enough": "hush",
    # --- quiz ------------------------------------------------------------
    "start quiz": "start quiz",
    "begin quiz": "start quiz",
    # --- ask the AI tutor (wake word; question may follow in the same utterance) ---
    "braillo": "braillo",
    "briello": "braillo",
    "brailo": "braillo",
    "barillo": "braillo",
    "braylo": "braillo",
    "brello": "braillo",
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
_user_muted: bool = False  # the learner turned the microphone off (a mute key): nothing is sent until they turn it on again
_listener_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()

# Guards start_listening()/stop_listening() so two threads racing to start
# (e.g. two hub/menu transitions firing close together) can never both pass
# the "not already listening" check and end up with two concurrent pa.open()
# calls on the same device — that race is what segfaults inside PortAudio's
# native OpenStream, below anything a Python try/except can catch.
_lifecycle_lock = threading.RLock()

# What the app is saying right now ("" when quiet), and the last line with the time its echo may still be heard.
_speaking_text: str = ""
_echo_text: str = ""
_echo_left: collections.Counter = collections.Counter()  # the line's words not yet heard coming back (see _without_echo)
_echo_until: float = 0.0
_interrupted = threading.Event()  # set by interrupt(): the line being played stops at its next chunk
_interrupts: int = 0  # how many lines have been cut off (callers drop lines they queued before a cut)
_speaker_thread = threading.local()  # .speaking is true only in the thread that is inside speak()

# The utterance being heard, in pieces: [(text, words)], and when the last piece arrived. A word is
# (text, confidence, start, end), with start/end in seconds of audio sent (Deepgram's clock); None where unknown.
Word = tuple
_pieces: list[tuple[str, list[Word]]] = []
_pieces_at: float = 0.0
_pieces_lock = threading.Lock()
_defer_long = False  # see set_defer_long_utterances()


def set_defer_long_utterances(on: bool) -> None:
    """The app can work out what free speech means (an AI helper): then a lone command word inside a long sentence is not run,
    and the sentence is reported as heard instead, for the app to judge ("I need to stop by the shop" is not "stop")."""
    global _defer_long
    _defer_long = bool(on)


class _RoomLevel:
    """How loud the microphone is, moment by moment, on Deepgram's clock (seconds of audio sent): the background level of the
    room (the quiet between words), and how loudly the learner speaks, so speech at the microphone can be told from the room."""

    def __init__(self, keep_seconds: float = 60.0) -> None:
        self.keep, self._lock = keep_seconds, threading.Lock()
        self.reset()

    def reset(self) -> None:
        """A new connection: Deepgram's clock starts again at zero. (The learner's voice level is kept.)"""
        with self._lock:
            self.t = 0.0
            self.marks: collections.deque = collections.deque()
            if not hasattr(self, "voice"):
                self.voice: collections.deque = collections.deque(maxlen=8)

    def add(self, pcm: bytes, rate: int = SAMPLE_RATE) -> None:
        """One chunk of what is being sent to Deepgram (16-bit mono)."""
        n = len(pcm) // 2
        if n == 0:
            return
        db = _dbfs(pcm) if np is not None else None
        with self._lock:
            if db is not None:
                self.marks.append((self.t, db))
                while self.marks and self.marks[0][0] < self.t - self.keep:
                    self.marks.popleft()
            self.t += n / rate

    def floor(self) -> float | None:
        """The room's background level (dBFS): the quieter moments of the last minute. None until there are 3 seconds to go on."""
        with self._lock:
            dbs = [d for _, d in self.marks]
        return float(np.percentile(dbs, 20)) if np is not None and len(dbs) >= 30 else None

    def level(self, start: float, end: float) -> float | None:
        """How loud the stretch [start, end] was (its louder moments, where the words are)."""
        with self._lock:
            dbs = [d for t, d in self.marks if start - 0.1 <= t <= end + 0.1]
        return float(np.percentile(dbs, 90)) if np is not None and dbs else None

    def check(self, start: float | None, end: float | None) -> str | None:
        """A reason to ignore speech in [start, end] as not meant for the app, or None if it stands out like the learner's."""
        if start is None or end is None:
            return None
        lvl, floor = self.level(start, end), self.floor()
        if lvl is None or floor is None:
            return None
        if lvl - floor < NOISE_GATE_DB:
            return "too quiet against the room"
        with self._lock:
            voice = list(self.voice)
        if len(voice) >= 3 and lvl < float(np.median(voice)) - FAR_VOICE_DB:
            return "much quieter than the learner's voice"
        return None

    def note_voice(self, start: float | None, end: float | None) -> None:
        """The learner just gave a command over [start, end]: remember how loud they speak."""
        if start is None or end is None:
            return
        lvl = self.level(start, end)
        if lvl is not None:
            with self._lock:
                self.voice.append(lvl)


def _dbfs(pcm: bytes) -> float:
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) / 32768.0 + 1e-9)) if x.size else -180.0


class _MicCleaner:
    """Clean the microphone signal before it is sent: a high-pass filter takes out hum, fans and traffic rumble (below
    HIGHPASS_HZ, where there is no speech worth having), and a slow, capped gain turns a quiet microphone up toward
    AGC_TARGET_DBFS. It never turns anything down, and it follows speech, not the room: only moments well above the background
    move it."""

    def __init__(self, rate: int = SAMPLE_RATE) -> None:
        # a second-order Butterworth high-pass (the "Audio EQ Cookbook" biquad, Q = 1/sqrt 2)
        w0 = 2 * np.pi * HIGHPASS_HZ / rate
        alpha = np.sin(w0) / np.sqrt(2)
        a0 = 1 + alpha
        self.b = ((1 + np.cos(w0)) / 2 / a0, -(1 + np.cos(w0)) / a0, (1 + np.cos(w0)) / 2 / a0)
        self.a = (-2 * np.cos(w0) / a0, (1 - alpha) / a0)
        self.state = [0.0, 0.0, 0.0, 0.0]  # x[n-1], x[n-2], y[n-1], y[n-2], carried from chunk to chunk
        self.speech_db: float | None = None
        self.gain_db = 0.0

    def highpass(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float64).tolist()
        (b0, b1, b2), (a1, a2) = self.b, self.a
        x1, x2, y1, y2 = self.state
        y = []
        for v in x:
            out = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, out
            y.append(out)
        self.state = [x1, x2, y1, y2]
        return np.clip(np.round(np.array(y)), -32768, 32767).astype(np.int16).tobytes()

    def gain(self, pcm: bytes, floor_db: float | None) -> bytes:
        db = _dbfs(pcm)
        if floor_db is not None and db > floor_db + NOISE_GATE_DB:  # this is speech: follow its level
            self.speech_db = db if self.speech_db is None else 0.9 * self.speech_db + 0.1 * db
        if self.speech_db is not None:
            want = min(max(AGC_TARGET_DBFS - self.speech_db, 0.0), AGC_MAX_GAIN_DB)
            self.gain_db += 0.2 * (want - self.gain_db)  # slowly: no pumping
        if self.gain_db < 0.1:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * (10 ** (self.gain_db / 20))
        return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


class _EchoCanceller:
    """WebRTC's echo canceller, noise suppressor and hum filter, fed in 10 ms frames: `played()` with what goes to the speakers (from
    the playback thread), `process()` with what the microphone heard (from the listening thread), which returns it with the app's
    own voice taken out. Both run in real time, so the frames line up closely; the canceller finds the remaining delay itself."""

    FRAME = SAMPLE_RATE // 100  # samples in 10 ms

    def __init__(self) -> None:
        from livekit import rtc  # type: ignore[import-untyped]

        self.rtc = rtc
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)
        self.apm.set_stream_delay_ms(AEC_DELAY_MS)
        self.far: collections.deque = collections.deque(maxlen=300)  # 10 ms frames being played, oldest first (3 s at most)
        self._far_rest = b""
        self._near_rest = b""
        self._lock = threading.Lock()

    def played(self, pcm: bytes, rate: int, channels: int, width: int) -> None:
        """Some audio just went to the speakers (any rate or channel count, 16-bit)."""
        if width != 2 or not pcm:
            return
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
        if channels > 1:
            x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
        if rate != SAMPLE_RATE and x.size:
            n = int(round(x.size * SAMPLE_RATE / rate))
            x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
        step = self.FRAME * 2
        with self._lock:
            data = self._far_rest + np.clip(x, -32768, 32767).astype(np.int16).tobytes()
            whole = len(data) - len(data) % step
            for i in range(0, whole, step):
                self.far.append(data[i:i + step])
            self._far_rest = data[whole:]

    def process(self, pcm: bytes) -> bytes:
        """What the microphone heard (16 kHz mono 16-bit), with the app's own sound and steady noise taken out."""
        step = self.FRAME * 2
        data = self._near_rest + pcm
        whole = len(data) - len(data) % step
        out = bytearray()
        for i in range(0, whole, step):
            with self._lock:
                far = self.far.popleft() if self.far else bytes(step)  # (nothing playing: silence)
            self.apm.process_reverse_stream(self.rtc.AudioFrame(far, SAMPLE_RATE, 1, self.FRAME))
            frame = self.rtc.AudioFrame(data[i:i + step], SAMPLE_RATE, 1, self.FRAME)
            self.apm.process_stream(frame)
            out += bytes(frame.data)
        self._near_rest = data[whole:]
        return bytes(out)


def _make_canceller() -> "_EchoCanceller | None":
    if not AEC or np is None:
        return None
    try:
        return _EchoCanceller()
    except Exception as exc:  # the package is missing or cannot load here: the plain clean-up is used
        log.warning("Echo cancellation is not available (%s); the app's own voice may drown the learner's while it talks.", exc)
        return None


_room = _RoomLevel()
_canceller: "_EchoCanceller | None" = None  # set while listening (see _deepgram_listener_loop)


def _span(words: list[Word] | None) -> tuple[float | None, float | None]:
    timed = [w for w in words or () if w[2] is not None and w[3] is not None]
    return (timed[0][2], timed[-1][3]) if timed else (None, None)


def _normalize_transcript(transcript: str) -> str:
    """Normalize user speech without altering words inside the transcript."""
    normalized = " ".join(transcript.casefold().strip().split())
    return normalized.rstrip(".,!?;:")


_LEAVING = ("menu", "stop", "sign out")  # (said after the wake word, these leave even in a question: "can we stop")
_QUESTION_WORDS = {"what", "how", "why", "where", "when", "who", "which", "is", "are", "can", "could", "does", "do", "did", "will",
                   "would", "should", "tell", "explain"}  # (after the wake word, a question goes to Braillo whatever words are in it)


def _match_phrase(normalized_transcript: str) -> tuple[str, str] | None:
    """(command, the phrase that matched) for the longest whole-phrase match, or None.

    The braillo wake word always wins when present, so "braillo how do I learn"
    asks the AI tutor instead of switching into Learn mode.
    """
    for phrase, pattern in _COMMAND_PATTERNS:
        found = pattern.search(normalized_transcript) if _COMMAND_MAP[phrase] == "braillo" else None
        if found:
            # ...except a short command after it, not a question: "hey Braillo, quit", "hey Braillo, I need a hint", "hey Braillo,
            # let's do read" are those commands (from real runs, where they were taken as questions)
            rest = normalized_transcript[found.end():]
            words = _words(rest)
            if 0 < len(words) <= 5 and (words[0] not in _QUESTION_WORDS or any(_COMMAND_MAP[p] in _LEAVING and q.search(rest)
                                                                            for p, q in _COMMAND_PATTERNS)):
                for other, other_pattern in _COMMAND_PATTERNS:
                    if _COMMAND_MAP[other] != "braillo" and other_pattern.search(rest):
                        return _COMMAND_MAP[other], other
            return "braillo", phrase
    for phrase, pattern in _COMMAND_PATTERNS:
        if pattern.search(normalized_transcript):
            return _COMMAND_MAP[phrase], phrase
    return None


def _match_command(normalized_transcript: str) -> str | None:
    """Return the canonical command for the longest whole-phrase match."""
    found = _match_phrase(normalized_transcript)
    return found[0] if found else None


_FUZZY_PHRASES = tuple(p for p in _COMMAND_MAP if len(p.replace(" ", "")) >= 4)


def _consonants(text: str) -> str:
    """The shape of a word with its vowels blurred: speech recognition in noise mostly gets the vowel wrong ("hent", "repeet")."""
    return re.sub(r"[aeiouy]+", "*", text)


def _fuzzy_phrase(normalized: str) -> tuple[str, str] | None:
    """A short utterance (at most 3 words) that sounds nearly like a command: (command, phrase). Either the same consonants
    with other vowels ("hent" is hint, "repeet" is repeat; "shop" is not stop), or nearly the same letters ("nex")."""
    heard = " ".join(_words(normalized))
    if not heard or len(heard.split()) > 3:
        return None
    for phrase in _FUZZY_PHRASES:
        if _consonants(heard) == _consonants(phrase):
            return _COMMAND_MAP[phrase], phrase
    # (nearly the same letters is too loose for leaving: "one more" is not "no more", from a real run)
    near = [p for p in _FUZZY_PHRASES if _COMMAND_MAP[p] not in ("menu", "sign out", "stop")]
    if not near:
        return None
    best = max(near, key=lambda p: difflib.SequenceMatcher(None, heard, p).ratio())
    return (_COMMAND_MAP[best], best) if difflib.SequenceMatcher(None, heard, best).ratio() >= FUZZY_RATIO else None


def _phrase_confidence(phrase: str, words: list[Word] | None, fallback: float) -> float:
    """The confidence of the words that make up `phrase` (so murmur around a clear command does not sink it)."""
    want = phrase.split()
    heard = [(_words(w[0] or "") or [""])[0] for w in words or ()]
    for i in range(len(heard) - len(want) + 1):
        if heard[i:i + len(want)] == want:
            confs = [w[1] for w in words[i:i + len(want)]]
            return sum(confs) / len(confs)
    return fallback


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def _echo_words(text: str) -> list[str]:
    """The words, for telling the app's own line from the learner's: every way the speech service writes the app's name ("brello",
    "brailo") counts as "braillo", so the name in its own line ("Hi, I'm Braillo"), coming back misspelt, is not taken for the wake word."""
    return ["braillo" if _COMMAND_MAP.get(w) == "braillo" else w for w in _words(text)]


def _without_echo(normalized: str, now: float | None = None, consume: bool = False) -> str:
    """What is left of a transcript once the words of the line the app is saying (or just said) are taken out.

    While the app talks the microphone hears both voices, and the transcript mixes them ("the letter d dots one
    wait what"): only the learner's own words are left ("what" ... "wait"). Each word of the line is taken out only
    as many times as the line says it, counting what has already come back (`consume`, for final transcripts): so
    when the line says "say guest" and the learner also says "guest", one "guest" is the app's and the other is
    theirs. (If the learner says it before the app has, theirs is taken for the app's at first, and the app's own,
    arriving later, is then left over: it still comes out as heard, just a moment late.)"""
    now = time.monotonic() if now is None else now
    line = _speaking_text or (_echo_text if now < _echo_until else "")
    if not line:
        return normalized
    left = _echo_left if consume else collections.Counter(_echo_left)
    own = []
    for w, key in zip(_words(normalized), _echo_words(normalized)):
        if left[key] > 0:
            left[key] -= 1  # the app's own word, coming back: each word of the line is only taken out as often as the line says it
        else:
            own.append(w)
    return " ".join(own)


def _learner_words(normalized: str, now: float | None = None, consume: bool = False) -> str:
    """The learner's part of what was heard while the app talks (or just did).

    With the echo canceller running, the app's voice is already taken out of the sound: only a transcript that is plainly the
    app's own line leaking through (three words or more, nearly all of them the line's) is dropped, and nothing is picked out of
    what the learner says, whose words may well be the line's own ("continue as a guest"). Without it, the app's words are taken
    out of the mixed transcript one by one (see _without_echo)."""
    if _canceller is None or _echo_unreliable():
        return _without_echo(normalized, now, consume)
    now = time.monotonic() if now is None else now
    line = _speaking_text or (_echo_text if now < _echo_until else "")
    heard = _echo_words(normalized)
    if line and len(heard) >= 3:
        said = set(_echo_words(line))
        if sum(w in said for w in heard) / len(heard) >= LEAK_SHARE:
            return ""
    return normalized


def _is_echo(normalized: str, now: float | None = None) -> bool:
    """Is this only what the app itself is saying (or just said), picked up by the microphone?"""
    return bool(_words(normalized)) and not _learner_words(normalized, now)


def _app_talking(now: float | None = None) -> bool:
    """The app is speaking, or has only just stopped (what is heard may still be partly its own voice)."""
    now = time.monotonic() if now is None else now
    return bool(_speaking_text) or now < _echo_until


def is_speaking() -> bool:
    """Is a line being played right now?"""
    return bool(_speaking_text)


_playing: dict | None = None  # the line now playing: {"text", "started" (wall clock, s), "seconds" (its audio, at normal speed)}


SPOKEN_WORDS_PER_SECOND = 2.9  # how fast the voice talks (Aura 2): a line still arriving is guessed at this, then corrected


def _wav_seconds(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(_fix_wav_header(wav))) as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


def _mark_playing(text: str, wav: bytes | None) -> None:
    """The audio for `text` is about to play: note when, and how long it lasts, so a display can follow along as it is spoken.
    With no `wav` (it is still arriving, see _streamed) the length is guessed from the words, and corrected once it has all come."""
    global _playing
    seconds = _wav_seconds(wav) if wav is not None else len(text.split()) / SPOKEN_WORDS_PER_SECOND
    _playing = {"text": text, "started": time.time(), "seconds": round(seconds, 3)}


ECHO_WINDOW_BEFORE, ECHO_WINDOW_AFTER = 4.0, 0.6  # a word heard this long after the app said it (or just before): its own echo


def _said_just_now(phrase: str, now: float | None = None) -> bool:
    """Did the app itself say this phrase within the last few seconds (by where it falls in the line playing, or just played)?
    A command word that comes back that soon after the app said it is its own voice, heard back ("Learn teaches the letters"
    heard as "learn"); the learner saying it at another moment is not. Words are placed evenly over the line's length."""
    line = _playing
    if not line:
        return False
    now = time.time() if now is None else now
    words, target = _words(line["text"]), _words(phrase)
    if not words or not target:
        return False
    started, seconds = line["started"], max(float(line["seconds"]), 0.1)
    for i in range(len(words) - len(target) + 1):
        if words[i:i + len(target)] == target:
            said_at = started + (i + 0.5) / len(words) * seconds
            if now - ECHO_WINDOW_BEFORE <= said_at <= now + ECHO_WINDOW_AFTER:
                return True
    return False


def now_playing() -> dict | None:
    """The line being spoken right now, with when it started (time.time()) and its length in seconds at normal speed (a display
    divides by the speech speed setting), or None when quiet. Lets a page show each sentence, word by word, as it is said."""
    return dict(_playing) if _playing and _speaking_text == _playing["text"] else None


def interrupt() -> bool:
    """Cut off the line being spoken (barge-in). Returns True if something was playing."""
    global _interrupts
    if not _speaking_text:
        return False
    _interrupts += 1
    _interrupted.set()
    return True


def interrupt_count() -> int:
    """How many lines have been cut off so far: a caller that queued a line before the latest cut can drop it."""
    return _interrupts


def playback_interrupted() -> bool:
    """For the audio players: should the clip playing in THIS thread stop now? (Earcons and other sounds played
    from other threads are never cut.)"""
    return getattr(_speaker_thread, "speaking", False) and _interrupted.is_set()


def _classify_transcript(
    transcript: str,
    confidence: float,
    is_final: bool,
    words: list[Word] | None = None,
) -> tuple[str, str | None, str | None]:
    """Normalize and decide whether one STT event is eligible to dispatch.

    Returns ``(normalized_transcript, command, ignored_reason)``. Exactly one
    command can be returned for an event. `confidence` is the whole utterance's;
    with `words` the command is judged on its own words, and the utterance on how
    loud it was against the room.
    """
    normalized = _normalize_transcript(transcript)
    if not is_final:
        return normalized, None, "interim transcript"
    if _mic_paused or _user_muted:
        return normalized, None, "microphone is paused"
    if not normalized:
        return normalized, None, "empty transcript"
    mixed = _app_talking()
    if mixed:  # keep only the learner's own words: the rest is the app's voice coming back
        normalized = _learner_words(normalized, consume=True)
        if not normalized:
            return normalized, None, "echo of the app's own voice"
    far = _room.check(*_span(words))
    if far:
        return normalized, None, far

    found = _match_phrase(normalized) or _fuzzy_phrase(normalized)
    if found is not None and mixed and found[0] not in CUT_OFF_BY and _said_just_now(found[1]):
        return normalized, None, "echo of the app's own voice"  # (a command word it had just said itself, heard back)
    if found is not None:
        command, phrase = found
        many = len(normalized.split()) > LONG_UTTERANCE_WORDS
        if _defer_long and many and " " not in phrase and command not in ("braillo", "hush"):
            found = None  # a lone command word inside a long sentence: let the app judge the sentence
        elif _phrase_confidence(phrase, words, confidence) < (TALK_OVER_CONFIDENCE if mixed else COMMAND_CONFIDENCE_THRESHOLD):
            return normalized, None, "confidence below threshold"
    if found is None:
        if mixed and len(normalized.split()) < TALK_OVER_WORDS:
            # a word or two over the app's voice: an answer ("guest", "yes") or the app itself misheard. Kept, marked as such:
            # the app acts on it if it makes sense there, and lets it go quietly if not. It does not stop the app talking.
            return normalized, None, OVER_SPEECH
        if confidence < COMMAND_CONFIDENCE_THRESHOLD and not mixed:  # (the app's voice lowers the whole transcript's score)
            return normalized, None, "confidence below threshold"
        return normalized, None, "no command phrase matched"
    if command == "stop" and _speaking_text and normalized.strip(" ,.!") in ("stop", "stop stop", "okay stop", "ok stop"):
        command = "hush"  # "stop" while it is talking means stop talking, not end the lesson
    return normalized, command, None


def _barge_in(transcript: str, confidence: float, words: list[Word] | None = None) -> None:
    """An interim (still being spoken) transcript while the app talks: if it is a command, cut the line off now,
    without waiting for the end of the utterance. The command itself is run when the final transcript arrives."""
    if not _speaking_text or _mic_paused or _user_muted:
        return
    # Strictly: the words of the line being said are taken out one by one even with the echo canceller running, because a line that
    # leaks past it comes back slightly misheard ("that's me, now the word C" for "That's E. Now the word C"), and cutting Braillo
    # off over his own voice is worse than a learner having to say "stop". What they really say over it still gets through.
    own = _without_echo(_normalize_transcript(transcript))
    if not own or _room.check(*_span(words)):
        return
    found = _match_phrase(own)
    command = found is not None and _phrase_confidence(found[1], words, confidence) >= TALK_OVER_CONFIDENCE
    if _echo_unreliable():  # (its own voice, misheard, must not cut it off: only a clear "stop" or "quiet" does)
        command = command and found[0] in CUT_OFF_BY
    # Only a command (or Braillo's name) cuts a line off. Other talk is heard and answered after the line, not over it: in a
    # room with other people in it, or with its own voice coming back misheard, "three words of their own" stopped it mid-sentence.
    if command and interrupt():
        log.info("Barge-in: %r cut the app off.", own)
        print(f"voice_io: barge-in: {own!r} cut the line off", flush=True)  # (worth seeing in the tutor's log: it stops Braillo mid-line)


def _debug_transcript_event(
    transcript: str,
    confidence: float,
    is_final: bool,
    normalized: str,
    command: str | None,
    ignored_reason: str | None,
    words: list[Word] | None = None,
) -> None:
    """Emit safe, opt-in diagnostics for a single Deepgram transcript event."""
    if not DEBUG_MODE or (not is_final and not normalized):  # (an empty partial result is just silence)
        return
    status = "final" if is_final else "interim"
    outcome = f"matched_command={command!r}" if command else "matched_command=None"
    reason = "" if ignored_reason is None else f" ignored_reason={ignored_reason!r}"
    start, end = _span(words)
    level, floor = (_room.level(start, end) if start is not None else None), _room.floor()
    loud = f" level={level:.0f}dB room={floor:.0f}dB" if level is not None and floor is not None else ""
    print(
        "[VOICE_IO DEBUG]"
        f" transcript={transcript!r}"
        f" confidence={confidence:.3f}"
        f" status={status}"
        f" normalized={normalized!r}"
        f" {outcome}{reason}{loud}",
        flush=True,
    )


def _process_transcript_event(transcript: str, confidence: float, is_final: bool,
                              words: list[Word] | None = None) -> str | None:
    """Classify one transcript event and dispatch at most one command."""
    normalized, command, ignored_reason = _classify_transcript(
        transcript, confidence, is_final, words
    )
    _debug_transcript_event(
        transcript, confidence, is_final, normalized, command, ignored_reason, words
    )
    if not is_final:
        _barge_in(transcript, confidence, words)
        return None
    if normalized and ignored_reason in (None, "no command phrase matched", "confidence below threshold", OVER_SPEECH):
        _status["heard"] = {"n": _status["heard"]["n"] + 1, "text": normalized, "matched": command is not None, "reason": ignored_reason}
        print(f"voice_io: heard {normalized!r}" + (f" -> {command}" if command else ""), flush=True)  # (so the log shows the conversation)
        # (no interrupt for talk that is not a command: it is answered once the line has finished, see _barge_in)
    if command is not None:
        log.info("Command recognised: %r (conf=%.2f)", command, confidence)
        _status["last_transcript"] = normalized
        _room.note_voice(*_span(words))
        interrupt()  # a command while it talks: stop talking first (a no-op if it is quiet)
        _fire_command(command)
    return command


def _as_words(words: list) -> list[Word]:
    """Words as (text, confidence, start, end); a bare confidence (older callers) becomes ("", c, None, None)."""
    return [w if isinstance(w, tuple) else ("", float(w), None, None) for w in words]


def _take_utterance() -> tuple[str, float, list[Word]] | None:
    """Join the pieces heard so far into one utterance (text, mean word confidence, words) and start a new one."""
    global _pieces
    with _pieces_lock:
        pieces, _pieces = _pieces, []
    if not pieces:
        return None
    words = [w for _, ws in pieces for w in ws]
    confs = [w[1] for w in words]
    return " ".join(t for t, _ in pieces), (sum(confs) / len(confs) if confs else 1.0), words


def _flush_utterance() -> str | None:
    utterance = _take_utterance()
    if not utterance:
        return None
    text, confidence, words = utterance
    return _process_transcript_event(text, confidence, True, words)


def _final_piece(transcript: str, words: list, speech_final: bool) -> str | None:
    """One finalised piece of what is being said (`words`: (text, confidence, start, end) each, or bare confidences). At the
    end of a piece (speech_final) a command is acted on at once; anything else (a question, a sentence) waits for the rest of
    the utterance (see UTTERANCE_END_MS)."""
    global _pieces_at
    if transcript.strip():
        with _pieces_lock:
            _pieces.append((transcript, _as_words(words)))
            _pieces_at = time.monotonic()
    if not speech_final:
        return None
    with _pieces_lock:
        text = " ".join(t for t, _ in _pieces)
    normalized = _normalize_transcript(text)
    found = _match_phrase(normalized) or _fuzzy_phrase(normalized)
    if found is not None and found[0] != "braillo":  # (a question may follow the wake word: hear it out)
        return _flush_utterance()
    if found is None and 0 < len(normalized.split()) <= QUICK_ANSWER_WORDS:
        return _flush_utterance()  # a short answer ("guest", "yes", a name) is complete: no need to wait for more
    return None


def _utterance_end() -> str | None:
    """Deepgram heard no more words for UTTERANCE_END_MS: the utterance is complete."""
    return _flush_utterance()


def _flush_stale_utterance(now: float | None = None) -> str | None:
    """Let a held utterance go if no UtteranceEnd arrived for it (called from the microphone loop)."""
    now = time.monotonic() if now is None else now
    with _pieces_lock:
        stale = bool(_pieces) and now - _pieces_at >= UTTERANCE_HOLD_SECONDS
    return _flush_utterance() if stale else None


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
    """Snapshot of listener state; safe to call from any thread. room_db: the room's background level at the microphone
    (dBFS, None until known)."""
    floor = _room.floor()
    return {**_status, "room_db": None if floor is None else round(floor, 1), "muted": _user_muted}


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


def set_muted(muted: bool) -> None:
    """The learner's own mute key: while muted nothing they say is sent to Deepgram or acted on. Separate from
    pause_listening()/resume_listening(), which the app uses around its own sounds and would otherwise undo it."""
    global _user_muted
    _user_muted = bool(muted)
    log.info("Microphone %s by the learner.", "muted" if _user_muted else "turned back on")


def is_muted() -> bool:
    return _user_muted


# ---------------------------------------------------------------------------
# Public API — speaking
# ---------------------------------------------------------------------------


def speak(text: str, mode: str = "normal") -> None:
    """Synthesise *text* and play it via Deepgram TTS.

    The microphone stays live (barge-in, see BARGE_IN): the learner can cut the
    line off, and what the microphone hears of the line itself is dropped. With
    BARGE_IN off the microphone is paused while speaking and resumed after.

    Args:
        text: The text to speak.
        mode: "normal" or "debrief" (both are spoken by Deepgram).
    """
    global _speaking_text, _echo_text, _echo_until
    if not BARGE_IN:
        pause_listening()
    _interrupted.clear()
    _speaking_text = _echo_text = text
    _echo_left.clear()
    _echo_left.update(_echo_words(text))
    _speaker_thread.speaking = True
    try:
        if MOCK_MODE:
            _mock_speak(text, mode)
        else:
            _deepgram_speak(text)
    except Exception:
        log.exception("speak() failed for mode=%r; text=%r", mode, text)
    finally:
        _speaker_thread.speaking = False
        _echo_until = time.monotonic() + (PHONE_ECHO_TAIL_SECONDS if _echo_unreliable() else ECHO_TAIL_SECONDS)
        _speaking_text = ""
        if not BARGE_IN:
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
    global _canceller
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
                endpointing=300,             # 300 ms silence → finalise a piece
                                             # (was 500; shorter = faster command response
                                             #  without cutting off "start quiz" mid-phrase)
                utterance_end_ms=UTTERANCE_END_MS,  # ...and this long with no words ends the
                                             # utterance, so a question with a pause is heard whole
                vad_events=True,
                smart_format=True,           # handles capitalisation + punctuation;
                                             # our \b regex handles trailing periods fine
                interim_results="true",      # needed for UtteranceEnd, and lets a command cut the
                                             # app off mid-line; commands still run on final text only
                keyterm=[                    # boost command vocabulary in the acoustic model
                    "repeat", "hint", "next", "stop", "finish", "done",
                    "found it", "got it",
                    "start quiz", "begin quiz",
                    "braillo", "learn", "read", "quiz", "menu",
                    "quiet", "wait", "help", "skip", "slower", "faster",
                    "practice", "explore", "next page",
                ],
            ) as socket:
                reader_done = threading.Event()
                connection_open = threading.Event()

                def _on_open(_event: object) -> None:
                    connection_open.set()

                def _on_message(msg: object) -> None:
                    """Transcript results build up an utterance; UtteranceEnd completes it."""
                    if not isinstance(msg, ListenV1Results):
                        if getattr(msg, "type", None) == "UtteranceEnd":
                            try:
                                _utterance_end()
                            except Exception:
                                log.exception("Error processing utterance end.")
                        elif DEBUG_MODE:
                            print(f"[DG:other] type={type(msg).__name__}", flush=True)
                        return
                    try:
                        alt = msg.channel.alternatives[0]
                        transcript: str = alt.transcript
                        words = [(getattr(w, "word", ""), w.confidence, getattr(w, "start", None), getattr(w, "end", None))
                                 for w in (alt.words or [])]
                        if not msg.is_final:
                            avg_conf = sum(w[1] for w in words) / len(words) if words else 1.0
                            _process_transcript_event(transcript, avg_conf, False, words)
                            return
                        # (a result without speech_final, as from older servers, is a whole utterance)
                        _final_piece(transcript, words, getattr(msg, "speech_final", True) is not False)
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
                _room.reset()  # Deepgram's clock (word times) starts again with this connection
                _canceller = _canceller or _make_canceller()  # (kept across reconnections: it has learnt the room)
                cleaner = _MicCleaner() if CONDITION_AUDIO and np is not None else None
                while not _stop_event.is_set() and not reader_done.is_set():
                    try:
                        data = audio_stream.read(chunk_frames, exception_on_overflow=False)
                    except OSError as exc:
                        log.warning("Microphone read error: %s", exc)
                        break

                    _flush_stale_utterance()

                    if _mic_paused or _user_muted:  # (no audio goes to Deepgram: it only keeps the connection open)
                        if time.monotonic() - _last_keepalive >= 3.0:
                            try:
                                socket.send_keep_alive()
                            except Exception as exc:
                                log.warning("send_keep_alive failed: %s", exc)
                            _last_keepalive = time.monotonic()
                        else:
                            time.sleep(0.05)
                        continue

                    if _canceller is not None:
                        data = _canceller.process(data)  # the app's own voice out, before anything else looks at it
                    if cleaner is not None:
                        if _canceller is None:
                            data = cleaner.highpass(data)  # (the canceller has its own hum filter)
                        _room.add(data)  # (measured before the gain, so near and far voices keep their difference)
                        data = cleaner.gain(data, _room.floor())
                    else:
                        _room.add(data)
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


# Names the voice says wrongly as they are written, and how to write them so it says them right (found by having the voice say each
# spelling and transcribing it back: "Braillo" came back as "brolo"; "Braylow" as "braylo"). Only what is sent to the voice changes:
# what is shown on screen keeps the real spelling.
SAY_AS = ((re.compile(r"\bBraillo\b", re.IGNORECASE), "Braylow"),)


def _for_tts(text: str) -> str:
    """Text actually sent to Deepgram. Aura 1 treats the last word of a short line as optional and
    drops it; a trailing ellipsis is a pause, so 'cat' is no longer the last thing it generates."""
    for name, said in SAY_AS:
        text = name.sub(said, text or "")
    t = (text or "").rstrip()
    if not t or DEEPGRAM_TTS_MODEL.startswith("aura-2"):  # (Aura 2 keeps the last word: no pause needed, and none dragging each line out)
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
        with _speech_cache_lock:
            raw = _SPEECH_CACHE.get(text)
        if raw is not None:  # made already (see prefetch_speech): it plays at once
            _mark_playing(text, raw)
            _play_audio_stream([_pad_wav_silence(raw)])
            return
        # Otherwise it plays as it arrives: Deepgram sends the first audio within a fraction of a second but takes about half the
        # line's own length to send all of it, so waiting for the whole of a long line before playing meant seconds of silence.
        _mark_playing(text, None)
        _play_audio_stream(_with_first_deadline(_streamed(text, DeepgramClient), FIRST_AUDIO_SECONDS))
    except Exception as exc:  # (no answer from Deepgram in time, or none at all: the line is still said, by this computer)
        log.error("Deepgram TTS failed: %s", exc)
        print(f"voice_io: Deepgram did not answer ({type(exc).__name__}): saying it with this computer's own voice", flush=True)
        _local_say(text)


# Lines the app says again and again (the welcome, the menu, "Sorry, I did not catch that") are synthesised once and kept, so they
# start at once instead of waiting about a second for Deepgram each time. prefetch_speech() makes them ahead of time.
_SPEECH_CACHE: "collections.OrderedDict[str, bytes]" = collections.OrderedDict()
SPEECH_CACHE_LINES = 64
_speech_cache_lock = threading.Lock()


def _synthesise(text: str, client_class=None) -> bytes:
    """The WAV for `text`, from the cache or from Deepgram (and then kept)."""
    with _speech_cache_lock:
        if text in _SPEECH_CACHE:
            _SPEECH_CACHE.move_to_end(text)
            return _SPEECH_CACHE[text]
    if client_class is None:
        from deepgram import DeepgramClient as client_class  # type: ignore[import-untyped]
    client = client_class(api_key=DEEPGRAM_API_KEY)
    # Join the stream while the SDK's response is still open, then pad: the phone speaker
    # intercepts _play_audio_stream, so any silence written only inside pyaudio never reaches
    # the phone (that is why 'cat' was still heard as 'ca' after the laptop-side waits).
    #
    # timeout_in_seconds: without a bound, a hung network call here hangs forever -- and say()
    # (voice_io callers) is routinely called while a caller holds its own session lock (see
    # tutor.py's TutorSession.lock), so one stuck TTS request can freeze every voice command and
    # button that needs that same lock, not just narration. 12s is generous for even a long
    # debrief sentence under a normal connection, but still a hard ceiling instead of none.
    raw = b"".join(
        client.speak.v1.audio.generate(
            text=_for_tts(text),
            model=DEEPGRAM_TTS_MODEL,
            encoding="linear16",  # request PCM; default is MP3 which wave.open() rejects
            container="wav",
            request_options={"timeout_in_seconds": 12},
        )
    )
    with _speech_cache_lock:
        _SPEECH_CACHE[text] = raw
        while len(_SPEECH_CACHE) > SPEECH_CACHE_LINES:
            _SPEECH_CACHE.popitem(last=False)
    return raw


FIRST_AUDIO_SECONDS = 4.0  # no sound from Deepgram this long after asking: the network is in trouble, say it another way


def _with_first_deadline(source, seconds: float):
    """`source`'s chunks, read on a thread of their own, raising TimeoutError if the first does not come within `seconds` (a
    request to Deepgram on a bad network can hang for tens of seconds, and everything said after it waits behind it)."""
    items: "queue.Queue" = queue.Queue()
    end = object()

    def pump() -> None:
        try:
            for chunk in source:
                items.put(chunk)
        except Exception as exc:  # (handed over, raised where it is read)
            items.put(exc)
        finally:
            items.put(end)

    threading.Thread(target=pump, daemon=True, name="speech-pump").start()
    wait = seconds
    while True:
        try:
            item = items.get(timeout=wait)
        except queue.Empty:
            raise TimeoutError(f"no audio from Deepgram within {wait:.0f} s") from None
        wait = 30.0
        if item is end:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _local_say(text: str) -> None:
    """Say a line with this computer's own voice (macOS `say`), when Deepgram cannot: plainer, but never silent. Cut off like any
    other line (barge-in)."""
    import shutil
    import subprocess

    if not shutil.which("say"):
        return
    try:
        proc = subprocess.Popen(["say", "-r", "185", _for_tts(text)])
    except OSError as exc:
        log.error("local voice failed: %s", exc)
        return
    while proc.poll() is None:
        if playback_interrupted():
            proc.terminate()
            break
        time.sleep(0.05)


def _streamed(text: str, client_class):
    """The WAV for `text`, chunk by chunk as Deepgram sends it, then a little silence (see WAV_SILENCE_SECONDS). Once it has all come
    it is kept (as _synthesise keeps it), and the length noted for the captions is corrected. Stops early if the line is cut off."""
    client = client_class(api_key=DEEPGRAM_API_KEY)
    got, fmt = bytearray(), None
    for chunk in client.speak.v1.audio.generate(text=_for_tts(text), model=DEEPGRAM_TTS_MODEL, encoding="linear16", container="wav",
                                                request_options={"timeout_in_seconds": 15}):
        if playback_interrupted():
            return
        got += chunk
        if fmt is None:
            fmt = _wav_format(bytes(got[:4096]))
        yield chunk
    raw = bytes(got)
    with _speech_cache_lock:
        _SPEECH_CACHE[text] = raw
        while len(_SPEECH_CACHE) > SPEECH_CACHE_LINES:
            _SPEECH_CACHE.popitem(last=False)
    playing = _playing
    if playing is not None and playing["text"] == text:
        playing["seconds"] = round(_wav_seconds(raw), 3)
    if fmt is not None:
        channels, width, rate, _ = fmt
        yield b"\x00" * (int(rate * WAV_SILENCE_SECONDS) * channels * width)


def _wav_format(head: bytes) -> tuple[int, int, int, int] | None:
    """(channels, bytes per sample, sample rate, where the samples start) from the start of a WAV, or None if more is needed. The
    sizes in the header are not used: while Deepgram streams, they claim hours of audio."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    at, fmt = 12, None
    while at + 8 <= len(head):
        kind, size = head[at:at + 4], int.from_bytes(head[at + 4:at + 8], "little")
        if kind == b"fmt ":
            if at + 24 > len(head):
                return None
            fmt = (int.from_bytes(head[at + 10:at + 12], "little"), int.from_bytes(head[at + 22:at + 24], "little") // 8,
                   int.from_bytes(head[at + 12:at + 16], "little"))
        elif kind == b"data":
            return (*fmt, at + 8) if fmt else None
        at += 8 + size + (size & 1)
    return None


def prefetch_speech(lines: list[str]) -> None:
    """Synthesise these lines in the background (the ones not kept already), so that saying them later starts at once."""
    if MOCK_MODE or not DEEPGRAM_API_KEY:
        return

    def run() -> None:
        for line in lines:
            try:
                _synthesise(line)
            except Exception as exc:  # it will simply be made when it is said
                log.info("prefetching speech failed (%s); it will be made when needed", exc)
                return
    threading.Thread(target=run, daemon=True, name="speech-prefetch").start()


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------


PLAY_AHEAD_SECONDS = 0.25  # a streamed line starts once this much of it has arrived, so a slow moment on the network does not stutter
PLAYBACK_FRAMES = 2048  # the speaker's buffer, in samples (about 85 ms at 24 kHz): with the default (a few ms) any small delay before
                        # the next write left it empty for an instant, and that was heard as crackling


def _play_audio_stream(chunks) -> None:
    """Play WAV bytes from an iterable of chunks, as they arrive: a line still coming from Deepgram (see _streamed) starts playing
    straight away instead of after all of it has come. The header's sizes are not trusted (a streamed one claims hours)."""
    try:
        import pyaudio  # type: ignore[import-untyped]
    except ImportError as exc:
        log.error("pyaudio not available for playback: %s", exc)
        return

    source = iter(chunks)
    head, fmt = bytearray(), None
    for chunk in source:
        head += chunk
        fmt = _wav_format(bytes(head[:4096]))
        if fmt is not None:
            break
    if fmt is None:
        if head:
            log.error("Audio playback failed: not a WAV this can play")
        return
    channels, width, rate, at = fmt
    frame = channels * width
    pending = bytearray(head[at:])

    try:
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
        stream = pa.open(format=pa.get_format_from_width(width), channels=channels, rate=rate, output=True,
                         frames_per_buffer=PLAYBACK_FRAMES)
        lock, arrived, finished = threading.Lock(), threading.Event(), threading.Event()
        # The echo canceller is told what went to the speakers on a thread of its own: it shares a lock with the microphone side, and
        # waiting for that between two writes would leave the speaker empty for a moment (a crackle).
        heard_out: "queue.Queue[bytes | None]" = queue.Queue()

        def tell_canceller() -> None:
            while (piece := heard_out.get()) is not None:
                if _canceller is not None:
                    _canceller.played(piece, rate, channels, width)

        threading.Thread(target=tell_canceller, daemon=True, name="speech-echo").start()

        def fetch() -> None:  # the rest of the line keeps arriving while the start of it plays
            try:
                for chunk in source:
                    with lock:
                        pending.extend(chunk)
                    arrived.set()
                    if playback_interrupted():
                        break
            except Exception as exc:
                log.error("Deepgram TTS stopped part-way: %s", exc)
            finally:
                finished.set()
                arrived.set()

        threading.Thread(target=fetch, daemon=True, name="speech-fetch").start()
        try:
            played, started, ahead = 0, None, int(rate * PLAY_AHEAD_SECONDS) * frame
            while not playback_interrupted():
                with lock:
                    have = len(pending)
                if not finished.is_set() and (have < frame or (started is None and have < ahead)):
                    arrived.wait(0.05)
                    arrived.clear()
                    continue
                with lock:
                    usable = len(pending) - len(pending) % frame
                    piece = bytes(pending[:min(usable, PLAYBACK_FRAMES * frame)])
                    del pending[:len(piece)]
                if not piece:
                    break
                if started is None:
                    started = time.monotonic()
                stream.write(piece)
                played += len(piece)
                heard_out.put(piece)  # what the speakers play is what to take back out of the microphone
            if started is not None:
                remaining = played / (rate * frame) - (time.monotonic() - started)
                if remaining > 0 and not playback_interrupted():
                    time.sleep(min(remaining, 8.0))  # let what is buffered finish
        finally:
            heard_out.put(None)
            stream.stop_stream()
            stream.close()
            pa.terminate()
    except Exception as exc:
        log.error("Audio playback failed: %s", exc)


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
