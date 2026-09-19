"""Grade 2 (contracted) English braille: turn a word's cells into English text.

Published braille is not spelled out letter by letter. One cell can be a whole word ("b" alone means "but", the cell 2346 means
"the") or a group of letters (16 = "ch", 346 = "ing"), and where a contraction may appear depends on where the cell sits in the
word. Read letter by letter, a page of this looks like nonsense even when every cell is right.

This covers the core rules of Unified English Braille: alphabetic wordsigns, strong and lower wordsigns, strong groupsigns, lower
groupsigns, initial groupsigns, the capital sign, the number sign, and trailing punctuation. It does NOT cover shortforms
("abt" = about), the two-cell initial-letter contractions, or technical notation. Where a word has more than one legal reading
it asks `is_word` (a spell checker) which are real words, so a reading is only as good as that dictionary.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Optional

from detect import _ALPHABET


def _key(dots) -> str:
    return "".join(str(d) for d in sorted(dots))


LETTERS = {"".join(sorted(v)): k for k, v in _ALPHABET.items()}  # "145" -> "d"
DIGITS = {"".join(sorted(_ALPHABET[c])): str(i) for i, c in zip(list(range(1, 10)) + [0], "abcdefghij")}  # after a number sign

ALPHA_WORDSIGNS = {"b": "but", "c": "can", "d": "do", "e": "every", "f": "from", "g": "go", "h": "have", "j": "just",
                   "k": "knowledge", "l": "like", "m": "more", "n": "not", "p": "people", "q": "quite", "r": "rather",
                   "s": "so", "t": "that", "u": "us", "v": "very", "w": "will", "x": "it", "y": "you", "z": "as"}
STRONG_WORDSIGNS = {"12346": "and", "123456": "for", "12356": "of", "2346": "the", "23456": "with", "16": "child",
                    "146": "shall", "1456": "this", "156": "which", "1256": "out", "34": "still"}
LOWER_WORDSIGNS = {"35": "in", "26": "enough", "23": "be", "235": "to", "2356": "were", "236": "his", "356": "was"}
STRONG_GROUPSIGNS = {"12346": "and", "123456": "for", "12356": "of", "2346": "the", "23456": "with", "16": "ch", "126": "gh",
                     "146": "sh", "1456": "th", "156": "wh", "1246": "ed", "12456": "er", "1256": "ou", "246": "ow",
                     "34": "st", "345": "ar"}
NOT_FIRST = {"346": "ing", "3456": "ble"}  # a word cannot begin with these
LOWER_MIDDLE = {"2": "ea", "23": "bb", "25": "cc", "235": "ff", "2356": "gg"}  # only between other cells, never first or last
LOWER_ANYWHERE = {"26": "en", "35": "in"}
INITIAL = {"23": "be", "25": "con", "256": "dis", "36": "com"}  # only at the start of a longer word
TRAILING_PUNCTUATION = {"2": ",", "256": ".", "235": "!", "236": "?", "23": ";", "25": ":", "356": '"'}
CAPITAL_SIGN, NUMBER_SIGN, OPENING_QUOTE = "6", "3456", "236"
MAX_CANDIDATES = 300


@dataclass(frozen=True)
class Decoded:
    text: str  # the chosen reading
    valid: bool  # whether `is_word` accepted it (False: a best guess from the rules alone)
    candidates: tuple  # every legal reading that was considered, best first


def _options(key: str, i: int, n: int) -> list:
    """Every legal reading of one cell in a multi-cell word: [(text, is_contraction)]."""
    out = []
    if key in LETTERS:
        out.append((LETTERS[key], 0))
    if key in STRONG_GROUPSIGNS:
        out.append((STRONG_GROUPSIGNS[key], 1))
    if i > 0 and key in NOT_FIRST:
        out.append((NOT_FIRST[key], 1))
    if 0 < i < n - 1 and key in LOWER_MIDDLE:
        out.append((LOWER_MIDDLE[key], 1))
    if key in LOWER_ANYWHERE:
        out.append((LOWER_ANYWHERE[key], 1))
    if i == 0 and key in INITIAL:
        out.append((INITIAL[key], 1))
    if key == "3" and 0 < i < n - 1:
        out.append(("'", 0))  # apostrophe inside a word: don't
    return out


def _readings(cells: list) -> list:
    """All spellings of a run of cells, as [(text, contractions)], capped so pathological words stay cheap."""
    n = len(cells)
    partial = [("", 0)]
    for i, key in enumerate(cells):
        opts = _options(key, i, n)
        if not opts:
            opts = [("?", 0)]  # not any known cell: keep the position visible
        partial = [(t + o, c + k) for t, c in partial for o, k in opts][:MAX_CANDIDATES * 4]
        if len(partial) > MAX_CANDIDATES:  # keep the ones with fewer contractions: they are the likelier
            partial = sorted(partial, key=lambda tc: tc[1])[:MAX_CANDIDATES]
    return partial


def _single_cell(key: str) -> list:
    """A one-cell word is usually a wordsign: 'b' alone is 'but', 2346 alone is 'the'."""
    out = []
    letter = LETTERS.get(key)
    if letter in ALPHA_WORDSIGNS:
        out.append((ALPHA_WORDSIGNS[letter], 1))
    if key in STRONG_WORDSIGNS:
        out.append((STRONG_WORDSIGNS[key], 1))
    if key in LOWER_WORDSIGNS:
        out.append((LOWER_WORDSIGNS[key], 1))
    if letter in ("a", "i"):
        out.append((letter, 0))  # "a" and "I" are words on their own
    return out


@functools.lru_cache(maxsize=1)
def default_is_word() -> Optional[Callable]:
    """A dictionary check if pyspellchecker is installed (the backend already needs it), else None."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return None
    checker = SpellChecker(language="en")
    return lambda w: bool(w) and w.lower() in checker


def decode_word(cells, is_word: Optional[Callable] = None) -> Decoded:
    """One word (a list of dot sets or dot-number strings) to English. is_word(str) -> bool picks between legal readings."""
    keys = [c if isinstance(c, str) else _key(c) for c in cells]
    if not keys:
        return Decoded("", True, ())
    is_word = is_word or default_is_word()
    prefix, upper_all, upper_first = "", False, False
    if keys[0] == NUMBER_SIGN and len(keys) > 1:
        return Decoded("".join(DIGITS.get(k, "?") for k in keys[1:]), True, ())
    if len(keys) > 2 and keys[0] == CAPITAL_SIGN and keys[1] == CAPITAL_SIGN:
        upper_all, keys = True, keys[2:]
    elif len(keys) > 1 and keys[0] == CAPITAL_SIGN:
        upper_first, keys = True, keys[1:]
    if len(keys) > 1 and keys[0] == OPENING_QUOTE:
        prefix, keys = '"', keys[1:]
    suffix = ""
    while len(keys) > 1 and keys[-1] in TRAILING_PUNCTUATION:  # punctuation stuck to the end of the word
        suffix = TRAILING_PUNCTUATION[keys[-1]] + suffix
        keys = keys[:-1]
    if len(keys) == 1:  # a wordsign outranks the bare letter: "b" on its own means "but", not the letter b
        readings = _single_cell(keys[0]) or [(LETTERS.get(keys[0]) or TRAILING_PUNCTUATION.get(keys[0]) or "?", 0)]
    else:
        readings = _readings(keys)
    seen, ranked = set(), []
    for text, contractions in readings:
        if text in seen:
            continue
        seen.add(text)
        ranked.append((0 if is_word and is_word(text) else 1, contractions, text))
    ranked.sort(key=lambda r: (r[0], r[1]))  # dictionary words first; among those, fewest contractions
    best_valid, _, best = ranked[0]
    if upper_all:
        best = best.upper()
    elif upper_first:
        best = best[:1].upper() + best[1:]
    return Decoded(prefix + best + suffix, best_valid == 0 if is_word else False, tuple(r[2] for r in ranked[:8]))


def decode_words(words: list, is_word: Optional[Callable] = None) -> list:
    """Decode several words: [[cell, ...], ...] -> ["the", "cat", ...]."""
    return [decode_word(w, is_word).text for w in words]
