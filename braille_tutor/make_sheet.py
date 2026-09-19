"""Printable poke-through braille sheets (A4). Run: python make_sheet.py [alphabet words numbers lookalikes]

Every sheet comes in two versions that share one layout, so the tutor treats them the same:
  sheet/poke_<name>.png              WITH corner markers (you stick four printed markers on the front)
  sheet/nomarkers/poke_<name>.png    WITHOUT markers (the camera finds the paper's own four edges: run the tutor with --paper)
plus face_<name>.png simulations of the finished face, for checking. sheet/print/ collects the four to print for a demo.
Quiz code gets the matching cells from sheet_cells(name); see sheets.py.
"""
from __future__ import annotations

import struct
import sys
import textwrap
import zlib
from pathlib import Path

import cv2
import numpy as np

from detect import Cell
from make_markers import SHEET_MARKER_MM as MARKER_MM
import page
from page import ARUCO_DICT, MARKER_POS_MM, PAGE_H_MM, PAGE_W_MM
from sheets import DOT_MM, DOT_R_MM, SHEET_NAMES, SPECS, get_sheet

OUT = Path("sheet")
DPI = 300
MM = DPI / 25.4  # pixels per mm
A4_MM = page.A4_MM
ORIGIN = page.SHEET_ORIGIN_MM  # paper position of marker 0's centre (and of page-mm (0, 0))
PRINT_PACK = ("alphabet", "words")  # the two designs collected into sheet/print/, each with and without markers

_ALPHABET_SPEC = SPECS["alphabet"]
ROWS = list(_ALPHABET_SPEC.rows)  # the alphabet sheet's layout, kept under these names for older code
PITCH_X, PITCH_Y, Y0 = _ALPHABET_SPEC.pitch_x, _ALPHABET_SPEC.pitch_y, _ALPHABET_SPEC.y0
X0 = (PAGE_W_MM - (max(map(len, ROWS)) - 1) * PITCH_X) / 2  # centre of the top-left cell, page mm


def sheet_cells(name: str = "alphabet") -> list[Cell]:
    """A sheet's cells in page mm, same structure as scan_page; use this if the detector isn't reliable."""
    return get_sheet(name).cells


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


def render_poke_template(name: str = "alphabet", markers: bool = True) -> np.ndarray:
    """Mirrored guide: black dot = poke here. Labels are left readable for the person poking.

    markers=False leaves out the marker pin-pricks and instructions: the camera uses the paper's own edges instead."""
    sheet = get_sheet(name)
    img = _canvas()
    _text(img, f"{sheet.spec.title}   (sheet: {name}, {'with' if markers else 'NO'} markers)", (int(58 * MM), int(22 * MM)), 1.0, 2)
    for i, line in enumerate(textwrap.wrap(sheet.spec.note, 78)[:4]):
        _text(img, line, (int(58 * MM), int((29 + 5 * i) * MM)), 0.55, 1)
    _text(img, "POKE SIDE. Print at 100% (no scaling). Place on foam, poke each black dot with a blunt stylus,",
          (int(12 * MM), int(58 * MM)), 0.7, 1)
    _text(img, "then FLIP THE SHEET OVER: raised dots on the back now read correctly, markers go on the front."
          if markers else "then FLIP THE SHEET OVER: the raised dots now read correctly. Use it on a plain DARK surface, "
          "all four paper edges in view.", (int(12 * MM), int(65 * MM)), 0.7 if markers else 0.6, 1)
    for c in sheet.cells:
        for x, y in dot_points(c):
            cv2.circle(img, _px(x, y, True), int(DOT_R_MM * MM), 0, -1, cv2.LINE_AA)
        _text(img, f"{sheet.symbol(c).short}: {''.join(map(str, sorted(c['dots'])))}",
              _px(c["x"], c["y"] + 2 * DOT_MM + 2, True), 0.8 if len(sheet.symbol(c).short) == 1 else 0.6, 2, center=True)
    for i, (mx, my) in (MARKER_POS_MM.items() if markers else ()):  # pin-prick crosses at the corners of each marker sticker
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


def render_face_preview(name: str = "alphabet", markers: bool = True) -> np.ndarray:
    """Simulated finished sheet seen from the camera: markers at the corners (if any), dots as dark discs."""
    img = _canvas()
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    m = int(MARKER_MM * MM)
    for i, (mx, my) in (MARKER_POS_MM.items() if markers else ()):
        cx, cy = _px(mx, my, False)
        img[cy - m // 2 : cy - m // 2 + m, cx - m // 2 : cx - m // 2 + m] = cv2.aruco.generateImageMarker(d, i, m)
    for c in sheet_cells(name):
        for x, y in dot_points(c):
            cv2.circle(img, _px(x, y, False), int(DOT_R_MM * MM), 60, -1, cv2.LINE_AA)
    return img


def render_flat_test(name: str = "alphabet", markers: bool = True) -> np.ndarray:
    """A print-and-use test sheet: the cells as solid black dots the right way round, so the camera side can be checked
    without poking anything. Not tactile - it is for testing the reader, not for a learner to feel."""
    img = render_face_preview(name, markers)
    img[(img > 30) & (img < 200)] = 0  # the preview draws dots in grey; print them solid black
    _text(img, f"{get_sheet(name).spec.title} - FLAT TEST SHEET (not tactile; print at 100%)",
          (int(20 * MM), int(20 * MM)), 0.8, 2)
    return img


def save_png(path: Path, img: np.ndarray) -> None:
    """Write a PNG with a pHYs chunk so print dialogs know it is 300 dpi (plain cv2.imwrite has no DPI)."""
    png = bytes(cv2.imencode(".png", img)[1])
    ppm = round(DPI / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body))
    path.write_bytes(png[:33] + chunk + png[33:])  # 8-byte signature + 25-byte IHDR chunk come first


def main() -> None:
    names = sys.argv[1:] or list(SHEET_NAMES)
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown sheet {name!r}; choose from {', '.join(SHEET_NAMES)}")
    (OUT / "nomarkers").mkdir(parents=True, exist_ok=True)
    for name in names:
        for markers, folder in ((True, OUT), (False, OUT / "nomarkers")):
            save_png(folder / f"poke_{name}.png", render_poke_template(name, markers))
            save_png(folder / f"face_{name}.png", render_face_preview(name, markers))
        print(f"{name:11s} {len(sheet_cells(name)):3d} cells -> {OUT}/poke_{name}.png and {OUT}/nomarkers/poke_{name}.png")
    if all(n in names for n in PRINT_PACK):
        (OUT / "print").mkdir(exist_ok=True)
        for i, (name, markers) in enumerate(((n, m) for n in PRINT_PACK for m in (True, False)), 1):
            save_png(OUT / "print" / f"{i}_{name}_{'WITH' if markers else 'NO'}_markers.png", render_poke_template(name, markers))
        for name in PRINT_PACK:  # print-and-use test sheets: check the camera side before poking anything
            save_png(OUT / "print" / f"flat_test_{name}.png", render_flat_test(name))
        print(f"the four to print: {OUT}/print/ (print at 100%; the WITH-markers ones also need markers/sheet.png cut out and stuck on)")
    print(f"PAGE {PAGE_W_MM}x{PAGE_H_MM} mm")


if __name__ == "__main__":
    main()
