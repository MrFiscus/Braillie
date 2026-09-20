"""
Mock implementation of CellDetector for development and smoke-testing.

Use this when the real braille_tutor detector isn't available (no camera,
no model weights, teammate's code not yet integrated).

Usage::

    from braillie.mock_detector import MockDetector

    det = MockDetector()          # cycles A→Z as finger "moves"
    det = MockDetector("bcdfgh")  # custom letter sequence
    det = MockDetector(cycle=False, letter="m")  # always returns "m"

The mock ignores (x_mm, y_mm) — position doesn't affect the result.
Advance to the next letter by calling det.advance() or by calling
get_cell_at() repeatedly (advances on each call when cycle=True).
"""

from __future__ import annotations

import threading
from typing import Optional

from braillie.interfaces import Cell

# Standard English braille Grade-1 alphabet: letter → active dot numbers.
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


def _dots_to_char(dots: frozenset) -> str:
    return chr(0x2800 + sum(1 << (d - 1) for d in dots))


def _dots_to_label(dots: frozenset) -> str:
    return "".join("1" if i in dots else "0" for i in range(1, 7))


def _make_cell(letter: str, x_mm: float = 20.0, y_mm: float = 20.0) -> Cell:
    dots = _ALPHABET[letter.lower()]
    return Cell(
        x=x_mm, y=y_mm, w=6.0, h=10.0,
        label=_dots_to_label(dots),
        char=_dots_to_char(dots),
        dots=dots,
        confidence=0.99,
        row=0, col=0,
    )


class MockDetector:
    """Cycles through a letter sequence on each get_cell_at() call.

    Thread-safe: the session calls get_cell_at() from multiple threads.
    """

    def __init__(self, sequence: str = "abcdefghij", *, cycle: bool = True):
        """
        Parameters
        ----------
        sequence:
            Letters to cycle through.  Each call to get_cell_at() advances
            to the next letter (wrapping when cycle=True).
        cycle:
            If False, stay on the last letter indefinitely.
        """
        if not sequence or any(c not in _ALPHABET for c in sequence.lower()):
            raise ValueError(
                f"sequence must be non-empty letters a–z; got {sequence!r}"
            )
        self._seq = sequence.lower()
        self._cycle = cycle
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def current_letter(self) -> str:
        """The letter that will be returned on the next get_cell_at() call."""
        with self._lock:
            return self._seq[self._idx]

    def advance(self) -> None:
        """Manually advance to the next letter in the sequence."""
        with self._lock:
            if self._cycle:
                self._idx = (self._idx + 1) % len(self._seq)
            else:
                self._idx = min(self._idx + 1, len(self._seq) - 1)

    def get_cell_at(self, x_mm: float, y_mm: float) -> Optional[Cell]:
        """Return the current letter as a Cell, then advance to the next one."""
        with self._lock:
            letter = self._seq[self._idx]
            if self._cycle:
                self._idx = (self._idx + 1) % len(self._seq)
            else:
                self._idx = min(self._idx + 1, len(self._seq) - 1)
        return _make_cell(letter, x_mm, y_mm)

    def set_next(self, letter: str) -> None:
        """Force the next get_cell_at() to return a specific letter."""
        letter = letter.lower()
        if letter not in _ALPHABET:
            raise ValueError(f"unsupported letter {letter!r}")
        if letter not in self._seq:
            raise ValueError(f"{letter!r} not in this detector's sequence {self._seq!r}")
        with self._lock:
            self._idx = self._seq.index(letter)
