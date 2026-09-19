"""Word validation for braille cell detector output.

When a detected word is not valid, the module asks the caller to re-run
detection on a fresh frame instead of guessing a correction.
"""
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import logging

from spellchecker import SpellChecker

log = logging.getLogger("braillie.word_correction")

Redetect = Callable[[], str]   # grabs a fresh frame, re-runs cell detection, returns the word


@dataclass(frozen=True)
class ReadResult:
    word: str            # word to narrate
    raw: str             # first detection
    corrected: bool      # True when a re-read replaced the raw detection
    source: str          # "exact" | "redetect" | "none"
    attempts: int        # number of re-detections performed


@dataclass(frozen=True)
class QuizResult:
    correct: bool
    detected: str        # first detection
    target: str
    corrected: bool      # True when correct only thanks to a re-read
    attempts: int


def _norm(s):
    return s.strip().lower()


@lru_cache(maxsize=1)
def _spell():
    return SpellChecker(language="en")


def is_valid_word(word):
    w = _norm(word)
    return bool(w) and w in _spell()


def _retry(redetect, max_redetects, accept):
    """Call `redetect` until `accept` returns True for the normalized re-read.

    Returns (accepted_value_or_None, attempts). Stops early if redetect raises.
    """
    attempts = 0
    while redetect is not None and attempts < max_redetects:
        attempts += 1
        try:
            value = _norm(redetect())
        except Exception as exc:
            log.warning("redetect failed on attempt %d: %s", attempts, exc)
            break
        if accept(value):
            return value, attempts
    return None, attempts


def correct_word_read_mode(detected_word, *, redetect=None, max_redetects=2):
    """Return the word to narrate, re-detecting instead of correcting."""
    raw = _norm(detected_word)
    if is_valid_word(raw):
        return ReadResult(raw, raw, False, "exact", 0)
    reread, attempts = _retry(redetect, max_redetects, is_valid_word)
    if reread is not None:
        log.info("read-mode re-detect raw=%r reread=%r attempt=%d", raw, reread, attempts)
        return ReadResult(reread, raw, True, "redetect", attempts)
    log.info("read-mode: no valid re-read for raw=%r after %d attempt(s)", raw, attempts)
    return ReadResult(raw, raw, False, "none", attempts)


def check_word_quiz_mode(detected_word, target_word, *, redetect=None, max_redetects=2):
    """Check the detected word against the quiz target by re-detecting."""
    detected = _norm(detected_word)
    target = _norm(target_word)
    if detected == target:
        return QuizResult(True, detected, target, False, 0)
    reread, attempts = _retry(redetect, max_redetects, lambda w: w == target)
    if reread is not None:
        log.info("quiz-mode re-detect matched detected=%r target=%r attempt=%d",
                 detected, target, attempts)
        return QuizResult(True, detected, target, True, attempts)
    log.debug("quiz-mode: no match detected=%r target=%r attempts=%d",
              detected, target, attempts)
    return QuizResult(False, detected, target, False, attempts)
