"""Score a reader against simulated ground truth: how many cells were found, and how many got the exact dot pattern."""
from __future__ import annotations

import numpy as np


def score(cells: list, truth: list, origin_mm=(10, 10), dot_mm=2.5, pitch_mm=6.0, line_mm=10.0, tol_mm=None) -> dict:
    """cells: reader output in page mm. truth: [(row, col, dotset)] from sim_embossed. Matches by nearest centre."""
    tol = tol_mm or 0.45 * pitch_mm
    found = exact = one_off = 0
    used = set()
    for r, c, dots in truth:
        cx = origin_mm[0] + c * pitch_mm + dot_mm / 2
        cy = origin_mm[1] + r * line_mm + dot_mm
        best, bd = None, tol
        for i, cell in enumerate(cells):
            d = np.hypot(cell["x"] - cx, cell["y"] - cy)
            if d < bd and i not in used:
                best, bd = i, d
        if best is not None:
            used.add(best)
            found += 1
            got = cells[best]["dots"]
            exact += got == dots
            one_off += len(got ^ dots) == 1
    n = len(truth)
    return {"truth": n, "found": found, "exact": exact, "recall": found / n, "exact_rate": exact / n,
            "exact_of_found": exact / max(found, 1), "one_dot_off": one_off, "false_cells": len(cells) - len(used)}
