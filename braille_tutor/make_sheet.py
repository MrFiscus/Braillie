"""Printable poke-through braille sheet (A4). Run: python make_sheet.py

Writes sheet/poke_template.png (print at 100%, poke from this side) and sheet/face_preview.png (a simulation of
the finished face, for checking). Quiz code gets matching cells from sheet_cells().
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from detect import Cell, cells_from_layout
from make_markers import SHEET_MARKER_MM as MARKER_MM
from page import ARUCO_DICT, MARKER_POS_MM, PAGE_H_MM, PAGE_W_MM

OUT = Path("sheet")
DPI = 300
MM = DPI / 25.4  # pixels per mm
A4_MM = (210.0, 297.0)
ORIGIN = ((A4_MM[0] - PAGE_W_MM) / 2, (A4_MM[1] - PAGE_H_MM) / 2)  # paper position of marker 0's centre

ROWS = ["abcdefgh", "ijklmnop", "qrstuvwx", "yz"]
DOT_MM = 6.0  # spacing between dots in a cell (real braille is 2.5, this is big on purpose)
DOT_R_MM = 1.6  # drawn dot radius
PITCH_X, PITCH_Y = 19.0, 45.0  # cell centre to cell centre
X0 = (PAGE_W_MM - (max(map(len, ROWS)) - 1) * PITCH_X) / 2  # centre of the top-left cell, page mm
Y0 = 50.0


def sheet_cells() -> list[Cell]:
    """The sheet's cells in page mm, same structure as scan_page; use this if the detector isn't reliable."""
    return cells_from_layout(ROWS, X0, Y0, PITCH_X, PITCH_Y, cell_w=DOT_MM + 2 * DOT_R_MM, cell_h=2 * DOT_MM + 2 * DOT_R_MM)


def dot_points(cell: Cell) -> list[tuple[float, float]]:
    """Page-mm centres of the raised dots in a cell (dots 1-3 left column, 4-6 right, top to bottom)."""
    return [(cell["x"] + ((d > 3) - 0.5) * DOT_MM, cell["y"] + ((d - 1) % 3 - 1) * DOT_MM) for d in sorted(cell["dots"])]


def _px(x_mm: float, y_mm: float, mirror: bool) -> tuple[int, int]:
    """Page mm -> pixel on the paper. Mirrored = left/right swapped, for the side you poke from."""
    px = ORIGIN[0] + x_mm
    if mirror:
        px = A4_MM[0] - px
    return int(round(px * MM)), int(round((ORIGIN[1] + y_mm) * MM))


def _text(img, s, xy, scale=0.9, thick=2, center=False):
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = xy
    cv2.putText(img, s, (x - w // 2 if center else x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)


def _canvas() -> np.ndarray:
    return np.full((int(A4_MM[1] * MM), int(A4_MM[0] * MM)), 255, np.uint8)


def render_poke_template() -> np.ndarray:
    """Mirrored guide: black dot = poke here. Letters are left readable for the person poking."""
    img = _canvas()
    _text(img, "POKE SIDE. Print at 100% (no scaling). Place on foam, poke each black dot with a blunt stylus,",
          (int(12 * MM), int(58 * MM)), 0.7, 1)
    _text(img, "then FLIP THE SHEET OVER: raised dots on the back now read correctly, markers go on the front.",
          (int(12 * MM), int(65 * MM)), 0.7, 1)
    for c in sheet_cells():
        for x, y in dot_points(c):
            cv2.circle(img, _px(x, y, True), int(DOT_R_MM * MM), 0, -1, cv2.LINE_AA)
        letter = ROWS[c["row"]][c["col"]]  # cells_from_layout keeps col = position in the string
        _text(img, f"{letter.upper()}: {''.join(map(str, sorted(c['dots'])))}",
              _px(c["x"], c["y"] + 2 * DOT_MM + 2, True), 0.8, 2, center=True)
    for i, (mx, my) in MARKER_POS_MM.items():  # pin-prick crosses at the corners of each marker sticker
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = _px(mx + sx * MARKER_MM / 2, my + sy * MARKER_MM / 2, True)
                cv2.drawMarker(img, (x, y), 0, cv2.MARKER_CROSS, int(4 * MM), 1)
        _text(img, f"marker {i}", _px(mx, my - 2, True), 0.7, 1, center=True)
        _text(img, "stick on FRONT", _px(mx, my + 3, True), 0.6, 1, center=True)
    x1, y = int(55 * MM), int(282 * MM)  # scale check bar between the bottom markers: must measure 100 mm
    cv2.line(img, (x1, y), (x1 + int(100 * MM), y), 0, 3)
    for t in (0, 100):
        cv2.line(img, (x1 + int(t * MM), y - 12), (x1 + int(t * MM), y + 12), 0, 3)
    _text(img, "this bar must measure 100 mm", (x1, y - 30), 0.7, 1)
    return img


def render_face_preview() -> np.ndarray:
    """Simulated finished sheet seen from the camera: markers at the corners, dots as dark discs."""
    img = _canvas()
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    m = int(MARKER_MM * MM)
    for i, (mx, my) in MARKER_POS_MM.items():
        cx, cy = _px(mx, my, False)
        img[cy - m // 2 : cy - m // 2 + m, cx - m // 2 : cx - m // 2 + m] = cv2.aruco.generateImageMarker(d, i, m)
    for c in sheet_cells():
        for x, y in dot_points(c):
            cv2.circle(img, _px(x, y, False), int(DOT_R_MM * MM), 60, -1, cv2.LINE_AA)
    return img


def save_png(path: Path, img: np.ndarray) -> None:
    """Write a PNG with a pHYs chunk so print dialogs know it is 300 dpi (plain cv2.imwrite has no DPI)."""
    png = bytes(cv2.imencode(".png", img)[1])
    ppm = round(DPI / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body))
    path.write_bytes(png[:33] + chunk + png[33:])  # 8-byte signature + 25-byte IHDR chunk come first


def main() -> None:
    OUT.mkdir(exist_ok=True)
    save_png(OUT / "poke_template.png", render_poke_template())
    save_png(OUT / "face_preview.png", render_face_preview())
    n = len(sheet_cells())
    print(f"wrote {OUT}/poke_template.png and {OUT}/face_preview.png ({n} cells). Print markers with make_markers.py.")
    print(f"cells_from_layout({ROWS}, x0={X0}, y0={Y0}, pitch_x={PITCH_X}, pitch_y={PITCH_Y}); PAGE {PAGE_W_MM}x{PAGE_H_MM} mm")


if __name__ == "__main__":
    main()
