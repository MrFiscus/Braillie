"""Turn detected cells into letters and words, so the word-checking backend can read them.

Plain (uncontracted) letters only: a cell that is not a letter reads as "?", which the spell checker rejects,
and that in turn makes the backend ask for a fresh detection instead of guessing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from detect import Cell, letter_of, nearest_cell, rows_of


def word_text(cells: list) -> str:
    """Letters of a run of cells, with "?" for any cell that isn't a plain letter."""
    return "".join(letter_of(c["dots"]) or "?" for c in cells)


def typical_pitch(cells: list) -> Optional[float]:
    """The normal centre-to-centre spacing of neighbouring cells, judged over the whole page.

    Judging it row by row fails on a line made of one-cell words ("the child can go"), where every gap is a word gap and so
    looks normal. The lower quartile of all gaps is dominated by cells inside words; a cell's box is about 0.54 of its pitch,
    which caps the estimate when a page has almost no multi-cell words."""
    gaps = [g for row in rows_of(cells) for g in np.diff([c["x"] for c in row])]
    if not gaps:
        return None
    widths = float(np.median([c["w"] for c in cells]))
    return min(float(np.percentile(gaps, 25)), 1.25 * widths / 0.54)


def split_words(row: list, gap_factor: float = 1.6, typical: Optional[float] = None) -> list:
    """Split one row (cells in reading order) into words wherever the gap to the next cell is much bigger than usual.

    typical: the normal cell spacing (see typical_pitch); by default the median gap in this row."""
    if len(row) < 2:
        return [row] if row else []
    gaps = np.diff([c["x"] for c in row])
    typical = typical if typical else float(np.median(gaps))
    words, current = [], [row[0]]
    for cell, gap in zip(row[1:], gaps):
        if gap > gap_factor * typical:
            words.append(current)
            current = []
        current.append(cell)
    words.append(current)
    return words


def read_lines(cells: list) -> list:
    """Every row as a line of space-separated words, e.g. ["cap cat", "dog"]."""
    pitch = typical_pitch(cells)
    return [" ".join(word_text(w) for w in split_words(row, typical=pitch)) for row in rows_of(cells)]


def word_at(cells: list, x_mm: float, y_mm: float, decode: bool = False, is_word=None) -> Optional[str]:
    """The word whose cell is nearest the page point (x, y), or None if the point isn't on a cell.

    decode=True reads it as contracted braille (see contractions.py); the default reads plain letters."""
    hit = nearest_cell(cells, x_mm, y_mm)
    if hit is None:
        return None
    pitch = typical_pitch(cells)
    for row in rows_of(cells):
        for word in split_words(row, typical=pitch):
            if any(c["row"] == hit["row"] and c["col"] == hit["col"] for c in word):
                if decode:
                    from contractions import decode_word
                    return decode_word([c["dots"] for c in word], is_word).text
                return word_text(word)
    return None


def decode_lines(cells: list, is_word=None) -> list:
    """Every row as English text with contractions decoded, e.g. ["the cat", "but you"].

    Published braille is contracted (one cell can be a word or a group of letters), so word_text() alone reads it as gibberish.
    is_word(str) -> bool is a dictionary check used to choose between legal readings; see contractions.py for what is covered."""
    from contractions import decode_word

    pitch = typical_pitch(cells)
    return [" ".join(decode_word([c["dots"] for c in word], is_word).text for word in split_words(row, typical=pitch))
            for row in rows_of(cells)]
