"""Temporal voting: pool detections from the last few scans so labels stop flickering from frame to frame."""
from __future__ import annotations

import collections
from collections import deque

import numpy as np

from detect import Cell, _assign_grid, _make_cell


def keep_inside(cells: list, w_mm: float, h_mm: float) -> list:
    """Drop cells whose centre is off the page (the receipt, the binder, a neighbouring page)."""
    return [c for c in cells if 0 <= c["x"] <= w_mm and 0 <= c["y"] <= h_mm]


def first_rows(cells: list, n: int) -> list:
    """Keep only the first n rows (lines of braille) of a row-numbered cell list."""
    return [c for c in cells if c["row"] < n]


class CellVoter:
    """Collects cells (in page mm) from the last `frames` scans; result() gives one stable cell per position.

    A cell must show up in at least `min_presence` of the scans to count, which removes one-frame ghosts. Its label
    is the one with the most total confidence; its confidence is (share of votes for that label) x (share of scans
    it appeared in).
    """

    def __init__(self, frames: int = 8, min_presence: float = 0.4):
        self.history: deque = deque(maxlen=frames)
        self.min_presence = min_presence

    def add(self, cells: list) -> None:
        """Record one scan's cells (page mm)."""
        self.history.append(cells)

    def reset(self) -> None:
        """Forget everything (the page moved or was lost)."""
        self.history.clear()

    def result(self) -> list:
        """One voted Cell per position, row/col numbered."""
        pool = [(i, c) for i, scan in enumerate(self.history) for c in scan]
        if not pool:
            return []
        radius = 0.4 * float(np.median([c["w"] for _, c in pool]))
        clusters: list = []
        for i, c in sorted(pool, key=lambda p: -p[1]["confidence"]):
            for cl in clusters:
                if np.hypot(c["x"] - cl[0][1]["x"], c["y"] - cl[0][1]["y"]) < radius:
                    cl.append((i, c))
                    break
            else:
                clusters.append([(i, c)])
        need = self.min_presence * len(self.history) if len(self.history) >= 3 else 0
        out = []
        for cl in clusters:
            scans = {i for i, _ in cl}
            if len(scans) < need:
                continue
            votes: collections.Counter = collections.Counter()
            for _, c in cl:
                votes[c["label"]] += c["confidence"]
            label, best = votes.most_common(1)[0]
            conf = (best / sum(votes.values())) * (len(scans) / len(self.history))
            out.append(_make_cell(float(np.mean([c["x"] for _, c in cl])), float(np.mean([c["y"] for _, c in cl])),
                                  float(np.mean([c["w"] for _, c in cl])), float(np.mean([c["h"] for _, c in cl])),
                                  label, conf))
        return _assign_grid(out)
