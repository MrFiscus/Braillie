"""Printable braille sheets (A4).

  python make_sheet.py        the three Braillie sheets (alphabet, words, look-alikes), each printed on both sides: a reading
                              side with the dots in black, and a poking side with a circle round each dot to push a pen through.
                              sheet/braillie-sheets.pdf to print, and a PNG of each side. (See make_pack.)

  python make_sheet.py NAME…  the older one-sided poke templates, in two versions that share one layout:
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
from sheets import DOT_MM, DOT_R_MM, SHEET_BUMPS, SHEET_NAMES, SPECS, get_sheet

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


# ---- the printed sheets: Braillie branded, printed on both sides ----------------------------------------------------------------
# Each sheet is one piece of paper. The READING side has every dot printed solid black where the bump will be (and the letter under
# each cell), with the Braillie wordmark, Braillo and the sheet's name; the POKING side, on the back, is its mirror image, with a
# circle round each dot to push a pen through. Printed double-sided (flipped on the long edge), every circle sits right behind its
# black dot, so each bump comes up on the dot it belongs to. The dots are where they have always been (sheet_cells), so the tutor
# reads these sheets exactly as before, finding the page by its own four edges (--paper).

PACK = ("alphabet", "words", "lookalikes", "numbers")  # Learn, Read, Quiz, and numbers and signs for going further
PACK_TITLES = {"alphabet": ("The alphabet", "for Learn"), "words": ("Words", "for Read"), "lookalikes": ("Look-alikes", "for Quiz"),
               "numbers": ("Numbers and signs", "for advanced practice")}
ID_Y_MM, ID_RIGHT_MM, ID_PITCH_MM = 38.0, 192.0, 7.0  # the sheet's raised ID dots (see sheets.SHEET_BUMPS): top right, in a row
BRAND = Path(__file__).parent / "sheet" / "brand"  # Braillo, drawn from the website's own picture of him
FONT_WOFF2 = Path(__file__).parent.parent / "website-frontend" / "public" / "fonts" / "atkinson-hyperlegible-next.woff2"
ACCENT = (11, 58, 176)  # the website's blue
SOFT = (110, 110, 118)  # rules and small print
RING_R_MM = 2.4  # the poking circle: a target a little bigger than the dot
PDF_OUT = OUT / "braillie-sheets.pdf"


def _font(size_mm: float, weight: int = 400):
    """The website's typeface (Atkinson Hyperlegible Next) at this size and weight. It is converted from the website's own font
    file once, into a cache next to this script (that needs the brotli package)."""
    from PIL import ImageFont

    ttf = Path(__file__).parent / "sheet" / ".cache" / "atkinson-hyperlegible-next.ttf"
    if not ttf.exists():
        from fontTools.ttLib import TTFont

        ttf.parent.mkdir(parents=True, exist_ok=True)
        f = TTFont(str(FONT_WOFF2))
        f.flavor = None
        f.save(str(ttf))
    font = ImageFont.truetype(str(ttf), int(round(size_mm * MM)))
    try:
        font.set_variation_by_axes([weight])
    except (OSError, AttributeError):  # (not a variable font here: one weight it is)
        pass
    return font


def _page():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (int(round(A4_MM[0] * MM)), int(round(A4_MM[1] * MM))), "white")
    return img, ImageDraw.Draw(img)


def _mm(x: float, y: float) -> tuple[int, int]:
    return int(round(x * MM)), int(round(y * MM))


def _header(img, draw, name: str, side: str) -> None:
    """Braillo, the wordmark, the sheet's name, and which side this is. On the poking side it is laid out the other way round (Braillo
    and the words at the right), because the sheet's ID dots come through at the top left there."""
    from PIL import Image

    title, use = PACK_TITLES[name]
    braillo = Image.open(BRAND / "braillo.png").convert("RGBA")
    h = int(round(40 * MM))
    braillo = braillo.resize((int(braillo.width * h / braillo.height), h), Image.LANCZOS)
    left = side == "reading"
    bw = braillo.width / MM
    img.paste(braillo, _mm(14 if left else 196 - bw, 12), braillo)
    x = 14 + bw + 4 if left else 196 - bw - 4  # where the words start (reading side) or end (poking side)
    anchor = "la" if left else "ra"
    word = _font(8.5, 800)
    wordmark_w = draw.textlength("Braillie.", font=word) / MM
    wx = x if left else x - wordmark_w
    draw.text(_mm(wx, 13), "Braillie", font=word, fill="black")
    draw.text((_mm(wx, 13)[0] + int(draw.textlength("Braillie", font=word)), _mm(wx, 13)[1]), ".", font=word, fill=ACCENT)
    draw.text(_mm(x, 25), title, font=_font(11, 800), fill="black", anchor=anchor)
    number = PACK.index(name) + 1
    draw.text(_mm(x, 40), f"Sheet {number} of {len(PACK)}  ·  {use}", font=_font(4.6, 600), fill=SOFT, anchor=anchor)
    # which side, as a label at the top corner away from Braillo: solid for the reading side, outlined for the poking side
    label = "Reading side" if side == "reading" else "Poking side"
    f = _font(4.4, 800)
    w = draw.textlength(label, font=f) / MM + 10
    x0 = 196 - w if left else 14
    box = [_mm(x0, 14), _mm(x0 + w, 25)]
    if left:
        draw.rounded_rectangle(box, radius=int(5.5 * MM), fill=ACCENT)
        draw.text(_mm(x0 + w / 2, 19.5), label, font=f, fill="white", anchor="mm")
    else:
        draw.rounded_rectangle(box, radius=int(5.5 * MM), outline=ACCENT, width=int(0.6 * MM))
        draw.text(_mm(x0 + w / 2, 19.5), label, font=f, fill=ACCENT, anchor="mm")
    draw.line([_mm(14, 58), _mm(196, 58)], fill=SOFT, width=max(1, int(0.3 * MM)))


def id_points(name: str) -> list[tuple[float, float]]:
    """Paper-mm centres of the sheet's ID dots, as seen on the reading side: a row along the top right (one per sheet number)."""
    return [(ID_RIGHT_MM - k * ID_PITCH_MM, ID_Y_MM) for k in range(SHEET_BUMPS[name])]


def _id_mark(draw, name: str, side: str) -> None:
    """The raised dots that tell which sheet this is (and which way up), with a word to say so for whoever prints it."""
    n = SHEET_BUMPS[name]
    for x, y in id_points(name):
        if side == "reading":
            r = DOT_R_MM * MM
            cx, cy = _mm(x, y)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")
        else:  # (behind it: mirrored, ringed like every other dot)
            cx, cy = _mm(A4_MM[0] - x, y)
            ring, point = RING_R_MM * MM, 0.45 * MM
            draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline="black", width=max(2, int(0.4 * MM)))
            draw.ellipse([cx - point, cy - point, cx + point, cy + point], fill="black")
    words = f"{n} bump{'s' if n > 1 else ''}: sheet {n}"
    if side == "reading":
        draw.text(_mm(ID_RIGHT_MM + 2.4, ID_Y_MM + 6), words, font=_font(3.4, 600), fill=SOFT, anchor="ra")
    else:
        draw.text(_mm(A4_MM[0] - ID_RIGHT_MM - 2.4, ID_Y_MM + 6), words, font=_font(3.4, 600), fill=SOFT, anchor="la")


def _labels(draw, sheet, mirror: bool) -> None:
    """The letter under each cell, readable on both sides (on the poking side it goes under the mirrored cell)."""
    f = _font(5.2, 800)
    for c in sheet.cells:
        draw.text(_px(c["x"], c["y"] + 2 * DOT_MM + 2.5, mirror), sheet.symbol(c).short, font=f, fill="black", anchor="mm")


def render_reading_side(name: str):
    """The side that is felt (and that the camera sees): every dot printed solid black where its bump comes up."""
    sheet = get_sheet(name)
    img, draw = _page()
    _header(img, draw, name, "reading")
    _id_mark(draw, name, "reading")
    r = DOT_R_MM * MM
    for c in sheet.cells:
        for x, y in dot_points(c):
            cx, cy = _px(x, y, False)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")
    _labels(draw, sheet, False)
    draw.line([_mm(14, 272), _mm(196, 272)], fill=SOFT, width=max(1, int(0.3 * MM)))
    draw.text(_mm(14, 277), "Feel the dots. Hear the letters.", font=_font(4.4, 800), fill="black")
    draw.text(_mm(196, 277), f"Which sheet? Count the bumps at the top right: {SHEET_BUMPS[name]}.", font=_font(4, 400), fill=SOFT,
              anchor="ra")
    return img


def render_poking_side(name: str):
    """The back: the mirror image, with a circle round each dot (and a point in its middle) to push a pen through."""
    sheet = get_sheet(name)
    img, draw = _page()
    _header(img, draw, name, "poking")
    _id_mark(draw, name, "poking")
    ring, point, width = RING_R_MM * MM, 0.45 * MM, max(2, int(0.4 * MM))
    for c in sheet.cells:
        for x, y in dot_points(c):
            cx, cy = _px(x, y, True)
            draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline="black", width=width)
            draw.ellipse([cx - point, cy - point, cx + point, cy + point], fill="black")
    _labels(draw, sheet, True)
    draw.line([_mm(14, 250), _mm(196, 250)], fill=SOFT, width=max(1, int(0.3 * MM)))
    steps = ("Lay this side up on something soft: a folded towel, foam or cork.",
             "Push a blunt pen through the middle of every circle, the ones in the top corner too.",
             "Turn the sheet over: each bump comes up on a black dot.")
    step = _font(4.4, 600)
    for i, text in enumerate(steps):
        y = 256 + i * 8
        cx, cy = _mm(18, y + 2.6)
        draw.ellipse([cx - 2.6 * MM, cy - 2.6 * MM, cx + 2.6 * MM, cy + 2.6 * MM], fill=ACCENT)
        draw.text((cx, cy), str(i + 1), font=_font(3.8, 800), fill="white", anchor="mm")
        draw.text(_mm(24, y), text, font=step, fill="black")
    # print-size check: 10 cm, at the right
    x0, y = 96, 284.5
    draw.line([_mm(x0, y), _mm(x0 + 100, y)], fill="black", width=max(2, int(0.35 * MM)))
    for t in (0, 100):
        draw.line([_mm(x0 + t, y - 1.8), _mm(x0 + t, y + 1.8)], fill="black", width=max(2, int(0.35 * MM)))
    draw.text(_mm(x0 + 50, y - 1.5), "Printed at 100%? This line is 10 cm.", font=_font(3.4, 400), fill=SOFT, anchor="mb")
    return img


def make_pack() -> list[Path]:
    """The three sheets, both sides: one PNG per side (300 dpi) and one PDF to print double-sided, reading side then poking side."""
    pages, paths = [], []
    for i, name in enumerate(PACK, 1):
        for side, render in (("reading", render_reading_side), ("poking", render_poking_side)):
            img = render(name)
            path = OUT / f"{i}_{name}_{side}.png"
            img.save(path, dpi=(DPI, DPI))
            pages.append(img)
            paths.append(path)
    pages[0].save(PDF_OUT, save_all=True, append_images=pages[1:], resolution=DPI)
    return paths + [PDF_OUT]


def save_png(path: Path, img: np.ndarray) -> None:
    """Write a PNG with a pHYs chunk so print dialogs know it is 300 dpi (plain cv2.imwrite has no DPI)."""
    png = bytes(cv2.imencode(".png", img)[1])
    ppm = round(DPI / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body))
    path.write_bytes(png[:33] + chunk + png[33:])  # 8-byte signature + 25-byte IHDR chunk come first


def main() -> None:
    if not sys.argv[1:]:  # the four branded sheets, printed on both sides (the usual thing to make)
        OUT.mkdir(exist_ok=True)
        for path in make_pack():
            print(path)
        print("print the PDF double-sided at 100% (no scaling), flipping on the long edge")
        return
    names = sys.argv[1:]  # (named sheets: the older one-sided poke templates, with and without markers)
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
