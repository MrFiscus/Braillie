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


def split_words(row: list, gap_factor: float = 1.6) -> list:
    """Split one row (cells in reading order) into words wherever the gap to the next cell is much bigger than usual."""
    if len(row) < 2:
        return [row] if row else []
    gaps = np.diff([c["x"] for c in row])
    typical = float(np.median(gaps))
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
    return [" ".join(word_text(w) for w in split_words(row)) for row in rows_of(cells)]


def word_at(cells: list, x_mm: float, y_mm: float) -> Optional[str]:
    """The word whose cell is nearest the page point (x, y), or None if the point isn't on a cell."""
    hit = nearest_cell(cells, x_mm, y_mm)
    if hit is None:
        return None
    for row in rows_of(cells):
        for word in split_words(row):
            if any(c["row"] == hit["row"] and c["col"] == hit["col"] for c in word):
                return word_text(word)
    return None
