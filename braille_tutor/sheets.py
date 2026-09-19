"""The printable practice sheets: what each cell on the page means, and where it sits.

Every sheet is the same A4 layout (four markers, page 150 x 237 mm), so calibration and page registration work for all of
them. A sheet is a few rows of characters; one character = one braille cell, a space = an empty slot (a word gap):

  letters a-z    the plain letter                   digits 0-9   the digit's cell (the same dots as a-j; put a "#" before it)
  # ^ . , ? ! ' - ;   number sign, capital sign, period, comma, question mark, exclamation mark, apostrophe, hyphen, semicolon
"""
from __future__ import annotations

from dataclasses import dataclass

from detect import _ALPHABET, Cell, _make_cell, dots_to_label
from page import PAGE_W_MM

DOT_MM = 6.0  # spacing between dots in a cell (real braille is 2.5, this is big on purpose)
DOT_R_MM = 1.6  # drawn dot radius


@dataclass(frozen=True)
class Symbol:
    """What a cell means: `key` identifies it, `spoken` is how the tutor says it, `short` labels it on the printout."""
    key: str
    spoken: str
    short: str
    dots: frozenset


def _dots(digits: str) -> frozenset:
    return frozenset(int(d) for d in digits)


_SIGNS = {  # English braille punctuation, as (dots, spoken name, short label)
    "#": ("3456", "the number sign", "NUM"), "^": ("6", "the capital sign", "CAP"), ".": ("256", "the period", "."),
    ",": ("2", "the comma", ","), "?": ("236", "the question mark", "?"), "!": ("235", "the exclamation mark", "!"),
    "'": ("3", "the apostrophe", "'"), "-": ("36", "the hyphen", "-"), ";": ("23", "the semicolon", ";"),
}
_DIGIT_LETTER = dict(zip("1234567890", "abcdefghij"))  # in braille a digit is the letter a-j after a number sign


def letter_symbol(letter: str) -> Symbol:
    """The symbol for a plain letter a-z."""
    return Symbol(letter, f"the letter {letter.upper()}", letter.upper(), _dots(_ALPHABET[letter]))


def symbol_for(ch: str) -> Symbol:
    """The Symbol a layout character stands for."""
    if ch in _ALPHABET:
        return letter_symbol(ch)
    if ch in _DIGIT_LETTER:
        return Symbol(ch, f"the number {ch}", ch, _dots(_ALPHABET[_DIGIT_LETTER[ch]]))
    if ch in _SIGNS:
        dots, spoken, short = _SIGNS[ch]
        return Symbol(ch, spoken, short, _dots(dots))
    raise ValueError(f"no braille symbol for {ch!r}")


@dataclass(frozen=True)
class SheetSpec:
    name: str
    title: str
    note: str  # printed on the poke template for the person poking, and explains the sheet
    rows: tuple
    pitch_y: float
    y0: float
    pitch_x: float = 19.0


SPECS = {s.name: s for s in (
    SheetSpec("alphabet", "ALPHABET A-Z", "Every letter once, in order, 8 per row.",
              ("abcdefgh", "ijklmnop", "qrstuvwx", "yz"), pitch_y=45.0, y0=50.0),
    SheetSpec("words", "WORDS", "Twelve short words for read mode and the word quiz: cat dog sun hat red cup bed pig fish egg "
              "bird bee. A blank slot separates words.",
              ("cat dog", "sun hat", "red cup", "bed pig", "fish egg", "bird bee"), pitch_y=30.0, y0=46.0),
    SheetSpec("numbers", "NUMBERS AND SIGNS", "Digits 1-9 then 0, each after its number sign (NUM). Bottom row: capital sign, "
              "period, comma, question mark, exclamation mark, apostrophe, hyphen, semicolon.",
              ("#1 #2 #3", "#4 #5 #6", "#7 #8 #9", "#0", "^.,?!'-;"), pitch_y=37.0, y0=46.0),
    SheetSpec("lookalikes", "LOOK-ALIKES", "Rows 1-3: pairs that differ by ONE dot (a/k b/l c/m d/n e/o f/p g/q h/r). "
              "Row 4: mirror images (d/f e/i h/j), the pairs learners mix up most.",
              ("ak bl cm", "dn eo fp", "gq hr", "df ei hj"), pitch_y=45.0, y0=50.0),
)}
SHEET_NAMES = tuple(SPECS)


class Sheet:
    """A built sheet: `cells` (page mm, same structure as scan_page) and `names` mapping (row, col) -> Symbol."""

    def __init__(self, spec: SheetSpec):
        self.spec, self.cells, self.names = spec, [], {}
        slots = max(len(r) for r in spec.rows)
        x0 = (PAGE_W_MM - (slots - 1) * spec.pitch_x) / 2  # centre of the first slot, page mm
        for r, row in enumerate(spec.rows):
            for c, ch in enumerate(row):
                if ch == " ":
                    continue
                sym = symbol_for(ch)
                self.cells.append(_make_cell(x0 + c * spec.pitch_x, spec.y0 + r * spec.pitch_y, DOT_MM + 2 * DOT_R_MM,
                                             2 * DOT_MM + 2 * DOT_R_MM, dots_to_label(sym.dots), 1.0, r, c))
                self.names[(r, c)] = sym

    def symbol(self, cell: Cell) -> Symbol:
        return self.names[(cell["row"], cell["col"])]


def get_sheet(name: str = "alphabet") -> Sheet:
    """Build a sheet by name (see SHEET_NAMES)."""
    if name not in SPECS:
        raise ValueError(f"unknown sheet {name!r}; choose from {', '.join(SHEET_NAMES)}")
    return Sheet(SPECS[name])
