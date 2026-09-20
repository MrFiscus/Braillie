"""
TutorSession: orchestrates read mode, quiz mode, voice I/O, and WebSocket state.

Wired between the detection module (CellDetector), voice I/O (voice_io.py),
and the WebSocket server (server.py).  All three are injected so the session
is testable without hardware.

PUBLIC API (what the main loop calls)
======================================

    session = TutorSession(
        detector  = my_detector,          # CellDetector impl or MockDetector
        speak_fn  = voice_io.speak,       # Callable[[str], None]
        broadcast = server.broadcast,     # Callable[[dict], None]
    )
    session.start()                        # registers voice commands, enters read mode

    # In the tracking loop (called once per stable finger position):
    session.on_cell_read(x_mm, y_mm)

    session.stop()                         # clean shutdown

MODES
=====
    "idle"  — not narrating, not quizzing.  Listens for "start quiz".
    "read"  — narrates the braille letter under the finger on every on_cell_read().
    "quiz"  — asks the user to find a target letter; checks on voice "found it".
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from braillie.interfaces import Cell, CellDetector, make_redetect
from braillie.word_correction import correct_word_read_mode

log = logging.getLogger("braillie.session")

# ---------------------------------------------------------------------------
# Braille alphabet — dots and human-readable descriptions used for hints
# ---------------------------------------------------------------------------

_DOT_NAME = {
    1: "top-left",     2: "middle-left",  3: "bottom-left",
    4: "top-right",    5: "middle-right", 6: "bottom-right",
}

_ALPHABET: dict[str, frozenset] = {
    'a': frozenset({1}),          'b': frozenset({1, 2}),
    'c': frozenset({1, 4}),       'd': frozenset({1, 4, 5}),
    'e': frozenset({1, 5}),       'f': frozenset({1, 2, 4}),
    'g': frozenset({1, 2, 4, 5}), 'h': frozenset({1, 2, 5}),
    'i': frozenset({2, 4}),       'j': frozenset({2, 4, 5}),
    'k': frozenset({1, 3}),       'l': frozenset({1, 2, 3}),
    'm': frozenset({1, 3, 4}),    'n': frozenset({1, 3, 4, 5}),
    'o': frozenset({1, 3, 5}),    'p': frozenset({1, 2, 3, 4}),
    'q': frozenset({1, 2, 3, 4, 5}), 'r': frozenset({1, 2, 3, 5}),
    's': frozenset({2, 3, 4}),    't': frozenset({2, 3, 4, 5}),
    'u': frozenset({1, 3, 6}),    'v': frozenset({1, 2, 3, 6}),
    'w': frozenset({2, 4, 5, 6}), 'x': frozenset({1, 3, 4, 6}),
    'y': frozenset({1, 3, 4, 5, 6}), 'z': frozenset({1, 3, 5, 6}),
}

# Default quiz sequence — first 10 letters so early sessions aren't overwhelming.
DEFAULT_QUIZ_LETTERS = list("abcdefghij")


def _dots_to_char(dots: frozenset) -> str:
    return chr(0x2800 + sum(1 << (d - 1) for d in dots))


def _hint_text(letter: str) -> str:
    dots = sorted(_ALPHABET[letter])
    n = len(dots)
    positions = ", ".join(_DOT_NAME[d] for d in dots)
    plural = "dots" if n > 1 else "dot"
    return f"The letter {letter.upper()} has {n} {plural}: {positions}."


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    total: int = 0
    correct: int = 0
    streak: int = 0
    missed: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def record(self, letter: str, correct: bool) -> None:
        self.total += 1
        if correct:
            self.correct += 1
            self.streak += 1
        else:
            self.streak = 0
            self.missed.append(letter)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 3),
            "streak": self.streak,
            "most_missed": _top_missed(self.missed),
        }


def _top_missed(missed: list, n: int = 5) -> list[str]:
    from collections import Counter
    return [letter for letter, _ in Counter(missed).most_common(n)]


# ---------------------------------------------------------------------------
# TutorSession
# ---------------------------------------------------------------------------

class TutorSession:
    """Coordinates detection, voice I/O, and WebSocket broadcast.

    Parameters
    ----------
    detector:
        Any object implementing CellDetector (real or MockDetector).
    speak_fn:
        Called to narrate text, e.g. ``voice_io.speak``.
        Signature: ``(text: str) -> None``.
    broadcast:
        Called to push a JSON-serialisable dict to all WebSocket clients.
        Thread-safe; called from voice-command thread and tracking loop.
        Signature: ``(msg: dict) -> None``.
    quiz_letters:
        Letters to cycle through in quiz mode.  Defaults to a–j.
    """

    def __init__(
        self,
        detector: CellDetector,
        speak_fn: Callable[[str], None],
        broadcast: Callable[[dict], None],
        quiz_letters: list[str] = DEFAULT_QUIZ_LETTERS,
    ):
        self._detector = detector
        self._speak = speak_fn
        self._broadcast = broadcast
        self._quiz_letters = [l.lower() for l in quiz_letters]

        self._mode: str = "idle"
        self._last_pos: tuple[float, float] = (0.0, 0.0)
        self._last_cell: Optional[Cell] = None
        self._last_narration: str = ""

        self._quiz_idx: int = 0
        self._quiz_target: str = ""          # current target letter
        self._quiz_wrong_streak: int = 0

        self._stats = SessionStats()
        self._lock = threading.Lock()        # guards _mode, _last_pos, _last_cell

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register voice commands and switch to read mode.

        Call once after voice_io.start_listening().
        """
        self._register_voice_commands()
        self._set_mode("read")
        log.info("TutorSession started in read mode")

    def stop(self) -> None:
        """Clean shutdown; does not stop voice_io (caller's responsibility)."""
        self._set_mode("idle")
        log.info("TutorSession stopped")

    def on_cell_read(self, x_mm: float, y_mm: float) -> None:
        """Called by the tracking loop when the finger has settled on a cell.

        In read mode: narrates the letter.
        In quiz mode: stores position (check happens on "found it" voice command).
        In idle mode: no-op.
        """
        with self._lock:
            self._last_pos = (x_mm, y_mm)
            cell = self._detector.get_cell_at(x_mm, y_mm)
            self._last_cell = cell
            mode = self._mode

        if cell is None:
            return

        self._broadcast({
            "type": "cell",
            "char": cell["char"],
            "label": cell["label"],
            "dots": sorted(cell["dots"]),
            "x_mm": round(cell["x"], 1),
            "y_mm": round(cell["y"], 1),
            "confidence": round(cell["confidence"], 3),
        })

        if mode == "read":
            self._read_cell(cell, x_mm, y_mm)
        elif mode == "quiz":
            # Position stored; don't narrate the cell (that would give the answer).
            # The user confirms by saying "found it".
            pass

    # ------------------------------------------------------------------
    # Voice command handlers (registered with voice_io)
    # ------------------------------------------------------------------

    def _register_voice_commands(self) -> None:
        try:
            import voice_io
            voice_io.register_command("repeat",     self._cmd_repeat)
            voice_io.register_command("hint",       self._cmd_hint)
            voice_io.register_command("found it",   self._cmd_found_it)
            voice_io.register_command("next",       self._cmd_next)
            voice_io.register_command("stop",       self._cmd_stop)
            voice_io.register_command("start quiz", self._cmd_start_quiz)
        except ImportError:
            log.warning("voice_io not importable — voice commands won't fire")

    def _cmd_repeat(self) -> None:
        self._broadcast({"type": "voice_command", "command": "repeat"})
        if self._last_narration:
            self._narrate(self._last_narration, announce=False)
        elif self._mode == "quiz" and self._quiz_target:
            self._narrate(f"Find the letter {self._quiz_target.upper()}.", announce=False)

    def _cmd_hint(self) -> None:
        self._broadcast({"type": "voice_command", "command": "hint"})
        if self._mode == "quiz" and self._quiz_target:
            self._narrate(_hint_text(self._quiz_target))
        elif self._mode == "read":
            with self._lock:
                cell = self._last_cell
            if cell:
                letter = _char_to_letter(cell["char"])
                if letter:
                    self._narrate(_hint_text(letter))

    def _cmd_found_it(self) -> None:
        self._broadcast({"type": "voice_command", "command": "found it"})
        if self._mode != "quiz":
            return
        with self._lock:
            x_mm, y_mm = self._last_pos
            target = self._quiz_target
        cell = self._detector.get_cell_at(x_mm, y_mm)
        if cell is None:
            self._narrate("I didn't catch a cell there. Try repositioning your finger.")
            return
        self._check_quiz_answer(cell, target, x_mm, y_mm)

    def _cmd_next(self) -> None:
        self._broadcast({"type": "voice_command", "command": "next"})
        if self._mode == "quiz":
            self._advance_quiz(skipped=True)
        elif self._mode == "read":
            self._narrate("Okay.")

    def _cmd_stop(self) -> None:
        self._broadcast({"type": "voice_command", "command": "stop"})
        self._set_mode("idle")
        self._narrate("Session paused. Say start quiz to begin a quiz.")

    def _cmd_start_quiz(self) -> None:
        self._broadcast({"type": "voice_command", "command": "start quiz"})
        if self._mode != "quiz":
            self._quiz_idx = 0
            self._quiz_wrong_streak = 0
        self._set_mode("quiz")
        self._prompt_quiz_target()

    # ------------------------------------------------------------------
    # Internal read-mode logic
    # ------------------------------------------------------------------

    def _read_cell(self, cell: Cell, x_mm: float, y_mm: float) -> None:
        raw_char = cell["char"]
        redetect = make_redetect(self._detector, x_mm, y_mm)
        result = correct_word_read_mode(raw_char, redetect=redetect)
        letter = _char_to_letter(result.word) or result.word
        text = f"The letter is {letter.upper()}." if len(letter) == 1 else f"The word is {letter}."
        self._narrate(text)

    # ------------------------------------------------------------------
    # Internal quiz-mode logic
    # ------------------------------------------------------------------

    def _prompt_quiz_target(self) -> None:
        target = self._quiz_letters[self._quiz_idx % len(self._quiz_letters)]
        self._quiz_target = target
        text = f"Find the letter {target.upper()}."
        self._broadcast({
            "type": "quiz_prompt",
            "target": target.upper(),
            "attempt": self._quiz_wrong_streak + 1,
        })
        self._narrate(text)

    def _check_quiz_answer(self, cell: Cell, target: str, x_mm: float, y_mm: float) -> None:
        target_dots = _ALPHABET.get(target, frozenset())
        detected_dots = cell["dots"]
        detected_letter = _char_to_letter(cell["char"]) or "?"
        correct = detected_dots == target_dots

        if not correct and self._quiz_wrong_streak < 1:
            # One silent re-detect before counting wrong
            redetect_fn = make_redetect(self._detector, x_mm, y_mm)
            reread_char = redetect_fn()
            if reread_char:
                reread_dots = _char_to_dots(reread_char)
                if reread_dots == target_dots:
                    correct = True
                    detected_letter = target

        self._stats.record(target, correct)
        result_msg = {
            "type": "quiz_result",
            "correct": correct,
            "detected": detected_letter.upper(),
            "target": target.upper(),
        }
        self._broadcast(result_msg)
        self._broadcast({"type": "stats", **self._stats.to_dict()})

        if correct:
            self._quiz_wrong_streak = 0
            self._narrate(f"Correct! That's the letter {target.upper()}.")
            self._advance_quiz(skipped=False)
        else:
            self._quiz_wrong_streak += 1
            if self._quiz_wrong_streak >= 3:
                self._narrate(
                    f"That's the letter {detected_letter.upper()}, not {target.upper()}. "
                    f"Let's move on."
                )
                self._quiz_wrong_streak = 0
                self._advance_quiz(skipped=False)
            else:
                self._narrate(f"Not quite. Try again. Say hint if you need help.")

    def _advance_quiz(self, skipped: bool) -> None:
        self._quiz_idx = (self._quiz_idx + 1) % len(self._quiz_letters)
        self._quiz_wrong_streak = 0
        self._prompt_quiz_target()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _narrate(self, text: str, announce: bool = True) -> None:
        self._last_narration = text
        if announce:
            self._broadcast({"type": "narration", "text": text, "mode": "normal"})
        try:
            self._speak(text)
        except Exception:
            log.exception("speak_fn raised")

    def _set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode
        try:
            import voice_io
            listening = voice_io._listening_active
            paused = voice_io._mic_paused
        except ImportError:
            listening, paused = False, False
        self._broadcast({
            "type": "state",
            "mode": mode,
            "listening": listening,
            "mic_paused": paused,
        })
        log.info("Mode → %s", mode)


# ---------------------------------------------------------------------------
# Utility: braille char ↔ letter
# ---------------------------------------------------------------------------

_CHAR_TO_LETTER = {_dots_to_char(dots): letter for letter, dots in _ALPHABET.items()}


def _char_to_letter(char: str) -> str:
    """Unicode braille char → English letter, or '' if not in Grade-1 alphabet."""
    return _CHAR_TO_LETTER.get(char, "")


def _char_to_dots(char: str) -> frozenset:
    code = ord(char) - 0x2800
    return frozenset(i + 1 for i in range(6) if code & (1 << i))
