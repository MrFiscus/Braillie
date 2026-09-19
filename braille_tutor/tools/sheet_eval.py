"""Score the detector on OUR generated sheets, seen as embossed (poked) dots through a simulated camera."""
from __future__ import annotations

import numpy as np

import sim_embossed as sim
import sheets
from page import SHEET_ORIGIN_MM

A4 = (210.0, 297.0)


def sheet_relief(name: str, px_per_mm: float = 8.0, contrast: float = 24.0, seed: int = 0):
    """(flat paper image, truth cells in PAPER mm) for one of our sheets, drawn as embossed dots."""
    sh = sheets.get_sheet(name)
    spec = sh.spec
    ncols = max(len(r) for r in spec.rows)
    rows = []
    for r, row in enumerate(spec.rows):
        rows.append([sheets.symbol_for(ch).dots if ch != " " else None for ch in row] + [None] * (ncols - len(row)))
    slots = ncols
    x0 = (150.0 - (slots - 1) * spec.pitch_x) / 2  # centre of slot 0 in page mm, as Sheet builds it
    origin = (SHEET_ORIGIN_MM[0] + x0 - sheets.DOT_MM / 2, SHEET_ORIGIN_MM[1] + spec.y0 - sheets.DOT_MM)
    flat, _ = sim.render_relief(rows, sheets.DOT_MM, spec.pitch_x, spec.pitch_y, px_per_mm, A4, origin_mm=origin,
                                contrast=contrast, seed=seed)
    truth = [{**c, "x": c["x"] + SHEET_ORIGIN_MM[0], "y": c["y"] + SHEET_ORIGIN_MM[1]} for c in sh.cells]
    return flat, truth


def camera(flat, fill: float, blur: float, seed: int, noise: float = 2.0):
    ph = int(1080 * fill)
    width = int(ph * A4[0] / A4[1])
    x0, y0 = (1920 - width) // 2, (1080 - ph) // 2
    quad = [[x0, y0], [x0 + width, y0 + 6], [x0 + width - 4, y0 + ph], [x0 + 3, y0 + ph - 4]]
    return sim.camera_view(flat, out_size=(1920, 1080), quad=quad, blur=blur, noise=noise, seed=seed)[0]
