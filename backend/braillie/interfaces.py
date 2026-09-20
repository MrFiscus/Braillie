"""
Canonical detection interface for the Braillie backend.

This is the single contract between the session logic (read mode, quiz mode)
and the detection module (braille_tutor or any replacement).  Import from here;
do not import from braille_tutor directly in session or server code.

WHAT THE DETECTION TEAMMATE NEEDS TO IMPLEMENT
===============================================
Implement the CellDetector protocol below.  Your class does not need to inherit
from anything — Python's structural typing (Protocol) means any object with a
``get_cell_at`` method of the right signature satisfies it.

Minimal example::

    class MyDetector:
        def get_cell_at(self, x_mm: float, y_mm: float) -> Optional[Cell]:
            # 1. Get the latest voted cell list from your CellVoter/Detector.
            # 2. Call braille_tutor.detect.nearest_cell(cells, x_mm, y_mm).
            # 3. Return the result (a Cell dict) or None.
            ...

Then pass an instance to TutorSession::

    session = TutorSession(detector=MyDetector(), ...)

The Cell dict your code returns must match the Cell TypedDict below.
braille_tutor.detect.Cell already has this exact shape, so no conversion needed.
"""

from __future__ import annotations

from typing import Optional, Protocol, TypedDict


class Cell(TypedDict):
    """One braille cell in page-coordinate millimetres (y-axis pointing down).

    This is the same shape as ``braille_tutor.detect.Cell``; the two are
    structurally compatible without any conversion.
    """
    x: float          # centre x, page mm from top-left corner
    y: float          # centre y, page mm from top-left corner
    w: float          # bounding box width, page mm
    h: float          # bounding box height, page mm
    label: str        # 6-char binary string, dot 1..6; e.g. "100000" = dot-1 only = letter A
    char: str         # Unicode braille character e.g. "⠁"; use for display and quiz matching
    dots: frozenset   # active dot numbers as frozenset[int], e.g. frozenset({1}) for A
    confidence: float # detector confidence 0.0–1.0
    row: int          # 0-based row in the page grid (top row = 0)
    col: int          # 0-based column in the page grid (left column = 0)


class CellDetector(Protocol):
    """What the backend session logic calls to read the braille page.

    Implement this with your braille_tutor.detect machinery.
    A mock implementation is in braillie.mock_detector.MockDetector.

    Contract
    --------
    - ``get_cell_at`` must be thread-safe (the session may call it from the
      voice-command thread as well as the main tracking loop).
    - It should return the *latest available* result — do not block waiting
      for a new frame.  The tracking loop controls when to call this.
    - Returning ``None`` means "no cell found near that point right now";
      the session will silently skip narration rather than crashing.
    """

    def get_cell_at(self, x_mm: float, y_mm: float) -> Optional[Cell]:
        """Return the braille cell nearest to (x_mm, y_mm) in the latest scan.

        Parameters
        ----------
        x_mm, y_mm:
            Fingertip position in page millimetres, as reported by the hand-
            tracking module.  Origin is the top-left corner of the registered
            page.

        Returns
        -------
        Cell dict if a cell exists within a reasonable proximity threshold,
        None otherwise (finger is off the page or between cells).
        """
        ...


# ---------------------------------------------------------------------------
# Bridge helper: CellDetector → word_correction.Redetect
# ---------------------------------------------------------------------------

def make_redetect(detector: CellDetector, x_mm: float, y_mm: float):
    """Build the ``Redetect = Callable[[], str]`` that word_correction expects.

    Call this once per cell-read event, capturing the current fingertip
    position.  Pass the returned callable to ``correct_word_read_mode`` or
    ``check_word_quiz_mode``::

        redetect = make_redetect(detector, x_mm, y_mm)
        result = correct_word_read_mode(raw_char, redetect=redetect)

    The callable re-queries the detector at the same page position and returns
    the new cell's ``char`` field, or ``""`` if no cell is found.
    """
    def _redetect() -> str:
        cell = detector.get_cell_at(x_mm, y_mm)
        return cell["char"] if cell else ""
    return _redetect
