"""Braille cell detection in page coordinates (mm), using pretrained DotNeuralNet YOLOv8 weights."""
from __future__ import annotations

import argparse
import json
import textwrap
import threading
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, TypedDict

import cv2
import numpy as np

from page import PAGE_H_MM, PAGE_W_MM, diagnose_markers, page_homography, to_image, to_page
from pagefind import AutoPage, find_page_homography
from tracker import load_calibration

WEIGHTS = Path(__file__).parent / "third_party/DotNeuralNet/weights/yolov8_braille.pt"

# Standard English braille letters -> dot numbers (used only by cells_from_layout).
_ALPHABET = dict(zip("abcdefghijklmnopqrstuvwxyz", (
    "1 12 14 145 15 124 1245 125 24 245 13 123 134 1345 135 1234 12345 1235 234 2345 136 1236 2456 1346 13456 1356"
).split()))


class Cell(TypedDict):
    """One braille cell. Geometry is in page millimeters, y pointing down."""
    x: float
    y: float
    w: float
    h: float
    label: str  # detector class, e.g. "100110"
    char: str  # Unicode braille, e.g. "⠙"
    dots: frozenset  # dot numbers 1..6
    confidence: float
    row: int
    col: int


# ---- label <-> dots <-> character -------------------------------------------------------------

def label_to_dots(label: str) -> frozenset:
    """'100110' -> frozenset({1, 4, 5}). Character i of the label is dot i+1."""
    if len(label) != 6 or set(label) - {"0", "1"}:
        raise ValueError(f"bad braille label: {label!r}")
    return frozenset(i + 1 for i, c in enumerate(label) if c == "1")


def dots_to_label(dots: Iterable[int]) -> str:
    """frozenset({1, 4, 5}) -> '100110'."""
    dots = set(dots)  # materialise once: callers may pass a generator
    return "".join("1" if i in dots else "0" for i in range(1, 7))


def dots_to_char(dots: Iterable[int]) -> str:
    """Dot set -> Unicode braille character (U+2800 block); the empty set is the blank cell U+2800."""
    dots = set(dots)
    if not dots <= {1, 2, 3, 4, 5, 6}:
        raise ValueError(f"dots must be in 1..6: {sorted(dots)}")
    return chr(0x2800 + sum(1 << (d - 1) for d in dots))


def dot_distance(a: Iterable[int], b: Iterable[int]) -> int:
    """Number of dots that differ between two cells (d vs f -> 2)."""
    return len(set(a) ^ set(b))


def _make_cell(x: float, y: float, w: float, h: float, label: str, conf: float, row: int = 0, col: int = 0) -> Cell:
    dots = label_to_dots(label)
    return Cell(x=x, y=y, w=w, h=h, label=label, char=dots_to_char(dots), dots=dots,
                confidence=conf, row=row, col=col)


# ---- detection --------------------------------------------------------------------------------

_model = None


def enhance_paper(frame: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Stretch and equalise the bright (paper) part of a frame, so faint embossed shading stands out.

    Helps a lot when the dots are low-contrast or small in the frame, and can hurt on a grainy or already-clean photo,
    which is why scan_page's "auto" mode tries both and keeps whichever the detector is more confident about."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    level, _ = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = gray[gray >= level]
    if paper.size < 0.02 * gray.size:  # hardly any bright area: nothing that looks like paper
        return frame
    lo, hi = np.percentile(paper, (2, 98))
    if hi - lo < 1:
        return frame
    out = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(out), cv2.COLOR_GRAY2BGR)


TARGET_BOX_PX = 8.5  # a braille cell's detected box is about 0.54 of its pitch; ~8.5 px at the model's input reads best
_SIZE_CACHE: dict = {}  # frame shape -> [calls so far, chosen input size]


def choose_imgsz(boxes: list, frame_side: int, current: int = 640) -> int:
    """The model input size that makes braille cells about TARGET_BOX_PX wide, from the boxes a quick first pass found.

    The model shrinks the whole frame to imgsz, so on a large camera frame the cells can end up too small to read (a 1080p
    frame at 640 shrinks them to a third). Measured on simulated pages: 64-80% correct at 640 became 90-98% at the right size."""
    if len(boxes) < 5:
        return current
    width = float(np.median([b[2] - b[0] for b in boxes]))
    return int(np.clip(round(TARGET_BOX_PX * frame_side / max(width, 1.0) / 32) * 32, 640, 1280))


def _run_enhance(frame: np.ndarray, conf: float, imgsz: int, enhance: str) -> list:
    if enhance == "off":
        return _detect_pixels(frame, conf, imgsz)
    if enhance == "on":
        return _detect_pixels(enhance_paper(frame), conf, imgsz)
    plain = _detect_pixels(frame, conf, imgsz)
    boosted = _detect_pixels(enhance_paper(frame), conf, imgsz)
    return boosted if sum(b[5] for b in boosted) > sum(b[5] for b in plain) else plain


def detect_boxes(frame: np.ndarray, conf: float = 0.15, imgsz="auto", enhance: str = "auto") -> list:
    """Pixel boxes for one frame. enhance: "off" (as-is), "on" (enhanced), "auto" (both, keep the more confident).

    imgsz: a number, or "auto" to size the model input to the braille in view (re-checked every 12th call, so it costs one
    extra quick pass now and then, not on every frame)."""
    if imgsz != "auto":
        return _run_enhance(frame, conf, int(imgsz), enhance)
    entry = _SIZE_CACHE.setdefault(frame.shape[:2], [0, 640])
    entry[0] += 1
    if entry[0] % 12 == 1:
        probe = _detect_pixels(frame, conf, 640)
        if len(probe) < 5:  # cells too small for 640 to find at all: look again at a larger size
            probe = _detect_pixels(frame, conf, 1024)
        entry[1] = choose_imgsz(probe, max(frame.shape[:2]), entry[1])
    return _run_enhance(frame, conf, entry[1], enhance)


def reading_quality(boxes: list) -> tuple:
    """(ok, message): whether a scan looks trustworthy, and what to change if not. Judged on how sure the detector was."""
    if not boxes:
        return False, "nothing detected: is the braille in view, lit from the side, and filling the frame?"
    mean_conf = sum(b[5] for b in boxes) / len(boxes)
    width = float(np.median([b[2] - b[0] for b in boxes]))
    if width < 12:
        return False, f"cells are only {width:.0f} px wide: move the camera closer, they need about 20 px"
    if mean_conf < 0.3:
        return False, f"the detector is unsure (average {mean_conf:.0%}): try stronger light from one side, at a low angle"
    if mean_conf < 0.45:
        return True, f"readings may be shaky (average confidence {mean_conf:.0%})"
    return True, ""


def _detect_pixels(frame: np.ndarray, conf: float, imgsz: int) -> list[tuple]:
    """Run YOLO once. Returns (x1, y1, x2, y2, label, confidence) in image pixels."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # imported lazily so page/layout code works without loading torch

        _model = YOLO(str(WEIGHTS))
    r = _model.predict(frame, conf=conf, imgsz=imgsz, device="cpu", agnostic_nms=True, max_det=1000, verbose=False)[0]
    return [(*b.xyxy[0].tolist(), _model.names[int(b.cls)], float(b.conf)) for b in r.boxes]


def _overlap(a: Cell, b: Cell) -> float:
    """Intersection area as a fraction of the smaller box (catches half-shifted duplicates that IoU misses)."""
    ix = min(a["x"] + a["w"] / 2, b["x"] + b["w"] / 2) - max(a["x"] - a["w"] / 2, b["x"] - b["w"] / 2)
    iy = min(a["y"] + a["h"] / 2, b["y"] + b["h"] / 2) - max(a["y"] - a["h"] / 2, b["y"] - b["h"] / 2)
    inter = max(ix, 0) * max(iy, 0)
    return inter / (min(a["w"] * a["h"], b["w"] * b["h"]) + 1e-9)


BOX_H_OVER_D = 2.87  # the detector's cell box is this many dot spacings tall, at every sheet size measured (2.84-2.91)


def _shift_dots(dots: frozenset, rows: int, cols: int) -> Optional[frozenset]:
    """The dots moved down `rows` and right `cols` (dot number = column * 3 + row + 1), or None if that leaves the cell."""
    out = set()
    for n in dots:
        col, row = (n - 1) // 3 + cols, (n - 1) % 3 + rows
        if col > 1 or row > 2:
            return None
        out.add(col * 3 + row + 1)
    return frozenset(out)


def _mode_pitch(values: np.ndarray, d: float) -> Optional[float]:
    """The commonest spacing among sorted 1-D positions: the cell pitch along a row, or the line pitch down a column."""
    gaps = np.diff(np.sort(values))
    gaps = gaps[gaps > 1.4 * d]  # closer than that is the same cell seen twice, or noise
    if len(gaps) < 2:
        return None
    bins = np.round(gaps / (0.25 * d)).astype(int)
    common = np.bincount(bins).argmax()
    return float(np.mean(gaps[np.abs(bins - common) <= 1]))


def align_shifted(cells: list, max_share: float = 0.45) -> list:
    """Undo the detector's row and column shift on cells that are missing their top row or left column.

    The detector reads a dot pattern relative to its box, and for a cell with no dots in the top row (a comma, a period, the
    capital sign) it places the box one dot spacing too low, so the pattern is read as if it were top-aligned: `235` comes out
    as `124`. Lines of braille sit on a regular grid, so a box that is one spacing below its line's top is telling us which rows
    are empty. Only clear whole-spacing offsets are acted on, and nothing changes if too many cells would move."""
    if len(cells) < 8:
        return cells
    d = float(np.median([c["h"] for c in cells])) / BOX_H_OVER_D
    tops = np.array([c["y"] - c["h"] / 2 for c in cells])
    lefts = np.array([c["x"] - c["w"] / 2 for c in cells])

    def offsets(values: np.ndarray, slots: int) -> Optional[np.ndarray]:
        """How many spacings each value sits past the grid line before it (0 = on the line), or None where unclear."""
        order = np.sort(values)
        clusters, start = [], 0
        for i in range(1, len(order) + 1):
            if i == len(order) or order[i] - order[i - 1] > 0.3 * d:
                clusters.append(order[start:i])
                start = i
        centres = [float(c.mean()) for c in clusters]
        strong = [float(c.mean()) for c in clusters if len(c) >= max(3, 0.12 * len(values))]
        pitch = _mode_pitch(np.array(strong), d) if len(strong) >= 3 else None
        if pitch is None or pitch < (slots + 0.6) * d:
            return None

        def on_grid(ref: float) -> int:  # how many values sit exactly on the grid line this reference implies
            return int((np.abs((values - ref + pitch / 2) % pitch - pitch / 2) < 0.3 * d).sum())

        ref = max(centres, key=on_grid)  # the reference is whichever line explains the most cells, not simply the biggest cluster
        residue = (values - ref) % pitch
        steps = residue / d
        rounded = np.round(steps)
        clear = (np.abs(steps - rounded) < 0.3) & (rounded <= slots)
        out = np.where(clear, rounded, 0).astype(int)
        out[residue > pitch - 0.6 * d] = 0  # the top of the next line, not a shift
        return out

    row_shift, col_shift = offsets(tops, 2), offsets(lefts, 1)
    row_shift = np.zeros(len(cells), int) if row_shift is None else row_shift
    col_shift = np.zeros(len(cells), int) if col_shift is None else col_shift
    moved = (row_shift > 0) | (col_shift > 0)
    if moved.sum() > max_share * len(cells):
        return cells  # implausibly many: the grid estimate is probably wrong, so leave the detector's answer alone
    out = []
    for c, r, k in zip(cells, row_shift, col_shift):
        shifted = _shift_dots(c["dots"], int(r), int(k)) if (r or k) else None
        if shifted is None:
            out.append(c)
            continue
        out.append({**c, "dots": shifted, "label": dots_to_label(shifted), "char": dots_to_char(shifted),
                    "y": c["y"] - r * d, "x": c["x"] - k * d})
    return out


def cells_from_boxes(boxes: list[tuple], H: np.ndarray, max_overlap: float = 0.3, align: bool = True) -> list[Cell]:
    """Pixel boxes (x1, y1, x2, y2, label, conf) -> deduplicated, row/col-numbered Cells in mm."""
    cells = []
    for x1, y1, x2, y2, label, conf in boxes:
        pts = [to_page(H, x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = to_page(H, (x1 + x2) / 2, (y1 + y2) / 2)  # centre in pixels, then to mm
        cells.append(_make_cell(cx, cy, max(xs) - min(xs), max(ys) - min(ys), label, conf))
    kept: list[Cell] = []  # greedy dedupe: highest confidence wins, overlapping boxes are dropped
    for c in sorted(cells, key=lambda c: -c["confidence"]):
        if all(_overlap(c, k) < max_overlap for k in kept):
            kept.append(c)
    return _assign_grid(align_shifted(kept) if align else kept)


def _assign_grid(cells: list[Cell]) -> list[Cell]:
    """Group into rows by y (gap > half a median cell height starts a new row), order by x."""
    if not cells:
        return []
    row_gap = 0.5 * float(np.median([c["h"] for c in cells]))
    rows: list[list[Cell]] = []
    for c in sorted(cells, key=lambda c: c["y"]):
        if rows and c["y"] - np.mean([r["y"] for r in rows[-1]]) <= row_gap:
            rows[-1].append(c)
        else:
            rows.append([c])
    out = []
    for r, row in enumerate(rows):
        for col, c in enumerate(sorted(row, key=lambda c: c["x"])):
            out.append({**c, "row": r, "col": col})
    return out


def scan_page(frame: np.ndarray, H: np.ndarray, conf: float = 0.15, imgsz="auto", enhance: str = "auto") -> list[Cell]:
    """Detect all cells in one frame (no hand in view). Cells come back in page mm, sorted by row, col."""
    return cells_from_boxes(detect_boxes(frame, conf, imgsz, enhance), H)


def rows_of(cells: list[Cell]) -> list[list[Cell]]:
    """Split a Cell list into rows (lists of Cells in column order)."""
    n = max((c["row"] for c in cells), default=-1) + 1
    return [sorted((c for c in cells if c["row"] == r), key=lambda c: c["col"]) for r in range(n)]


# ---- lookup and fallback ----------------------------------------------------------------------

def nearest_cell(cells: list[Cell], x_mm: float, y_mm: float, max_dist: Optional[float] = None) -> Optional[Cell]:
    """Closest cell to a page point, or None if farther than max_dist (default: half the median neighbor spacing)."""
    if not cells:
        return None
    xy = np.array([[c["x"], c["y"]] for c in cells])
    d = np.hypot(*(xy - [x_mm, y_mm]).T)
    if max_dist is None and len(cells) > 1:
        pair = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
        np.fill_diagonal(pair, np.inf)
        max_dist = 0.5 * float(np.median(pair.min(axis=1)))
    i = int(d.argmin())
    return cells[i] if max_dist is None or d[i] <= max_dist else None


def _median_spacing_mm(cells: list[Cell]) -> float:
    if len(cells) < 2:
        return 19.0
    xy = np.array([[c["x"], c["y"]] for c in cells], dtype=float)
    pair = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(pair, np.inf)
    return float(np.median(pair.min(axis=1)))


def _axis_spacing_mm(cells: list[Cell]) -> tuple[float, float]:
    """Typical (pitch_x, pitch_y) for a printed grid — words are denser vertically than the alphabet sheet."""
    if len(cells) < 2:
        return 19.0, 45.0
    by_row: dict = {}
    by_col: dict = {}
    for c in cells:
        by_row.setdefault(c["row"], []).append(float(c["x"]))
        by_col.setdefault(c["col"], []).append((int(c["row"]), float(c["y"])))
    dxs = []
    for xs in by_row.values():
        xs = sorted(xs)
        dxs.extend(xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > 1.0)
    dys = []
    for col in by_col.values():
        ys = [y for _, y in sorted(col)]
        dys.extend(ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1.0)
    px = float(np.median(dxs)) if dxs else 19.0
    py = float(np.median(dys)) if dys else 45.0
    return px, py


def cell_at(cells: list[Cell], x_mm: float, y_mm: float, pad_mm: Optional[float] = None) -> Optional[Cell]:
    """The cell whose drawn box contains the point, else the nearest cell within one grid step.

    Uses separate horizontal/vertical pitch so the denser words sheet (pitch_y ~30 mm) and lookalike
    pairs hit-test like the alphabet sheet. A tip in a letter's square must resolve to that letter."""
    if not cells:
        return None
    pitch_x, pitch_y = _axis_spacing_mm(cells)
    spacing = min(pitch_x, pitch_y)
    if pad_mm is None:
        pad_x, pad_y = pitch_x * 0.42, pitch_y * 0.42
    else:
        pad_x = pad_y = pad_mm
    hit, best = None, 1e9
    for c in cells:
        hw = min(max(float(c.get("w") or 0) / 2.0, pitch_x * 0.22) + pad_x, pitch_x * 0.55)
        hh = min(max(float(c.get("h") or 0) / 2.0, pitch_y * 0.22) + pad_y, pitch_y * 0.55)
        dx, dy = abs(x_mm - c["x"]), abs(y_mm - c["y"])
        if dx <= hw and dy <= hh:
            d = max(dx / max(hw, 1e-6), dy / max(hh, 1e-6))
            if d < best:
                best, hit = d, c
    if hit is not None:
        return hit
    return nearest_cell(cells, x_mm, y_mm, max_dist=spacing)


def cells_from_layout(rows_of_text: list[str], x0: float, y0: float, pitch_x: float, pitch_y: float,
                      cell_w: Optional[float] = None, cell_h: Optional[float] = None) -> list[Cell]:
    """Build Cells from a hardcoded letter grid. (x0, y0) is the centre of the top-left cell, in page mm.

    cell_w / cell_h are the box size (default 0.6 of the pitch).
    Spaces leave an empty slot. Same structure as scan_page, so quiz code can't tell the difference.
    """
    cells = []
    for r, text in enumerate(rows_of_text):
        for c, ch in enumerate(text.lower()):
            if ch == " ":
                continue
            if ch not in _ALPHABET:
                raise ValueError(f"unsupported character {ch!r}; use a-z or space")
            label = dots_to_label(int(d) for d in _ALPHABET[ch])
            cells.append(_make_cell(x0 + c * pitch_x, y0 + r * pitch_y, cell_w or 0.6 * pitch_x,
                                    cell_h or 0.6 * pitch_y, label, 1.0, r, c))
    return cells


def save_cells(cells: list[Cell], path: str) -> None:
    """Write cells to a JSON file (dots stored as sorted lists)."""
    Path(path).write_text(json.dumps([{**c, "dots": sorted(c["dots"])} for c in cells], indent=1, ensure_ascii=False))


def load_cells(path: str) -> list[Cell]:
    """Read cells written by save_cells."""
    return [{**c, "dots": frozenset(c["dots"])} for c in json.loads(Path(path).read_text())]


# ---- CLI --------------------------------------------------------------------------------------

def annotate(frame: np.ndarray, cells: list[Cell], H: np.ndarray) -> np.ndarray:
    """Draw each cell's box (as a page-aligned quad, so tilt is respected) and its dot numbers."""
    out = frame.copy()
    for c in cells:
        quad = np.int32([to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2)
                         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        cv2.polylines(out, [quad], True, (0, 200, 0), 2)
        text = "".join(map(str, sorted(c["dots"]))) or "-"
        cv2.putText(out, text, tuple(quad[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out


def print_rows(cells: list[Cell]) -> None:
    for r, row in enumerate(rows_of(cells)):
        print(f"row {r}: " + "".join(c["char"] for c in row))


def _has_frames(cap: cv2.VideoCapture, seconds: float = 4.0) -> bool:
    """True once the camera delivers a frame. Cameras (especially a phone via Continuity Camera) need a moment to start."""
    end = time.time() + seconds
    while cap.isOpened() and time.time() < end:
        if cap.read()[0]:
            return True
        time.sleep(0.1)
    return False


def _camera_works(index: int) -> bool:
    cap = cv2.VideoCapture(index)
    ok = _has_frames(cap, 2.0)
    cap.release()
    return ok


def configure_camera(cap: cv2.VideoCapture) -> str:
    """Ask a local camera for continuous autofocus. Many drivers ignore this (macOS often does); says what happened."""
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
        return "camera: autofocus requested and accepted"
    return "camera: the driver doesn't let software control focus (the camera may still autofocus by itself)"


_PHONE_ENDPOINTS = {"torch_on": "/enabletorch", "torch_off": "/disabletorch", "focus": "/focus"}  # Android "IP Webcam" style


def phone_command(base_url: str, action: str) -> str:
    """Send torch/focus to a phone camera app's web server (Android IP Webcam style URLs). Returns a status message.

    Only plain http(s) addresses are used. The endpoint names come from that app's documentation; other apps differ."""
    if not base_url.startswith(("http://", "https://")):
        return f"phone {action}: not sent, --phone-url must start with http:// or https://"
    url = base_url.rstrip("/") + _PHONE_ENDPOINTS[action]
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return f"phone {action}: sent ({r.status})"
    except Exception as e:
        return f"phone {action}: FAILED ({e}). Is the phone app's server running at {base_url}?"


def open_camera(src: Optional[str] = None) -> cv2.VideoCapture:
    """Open a camera by index, stream URL or video file; None picks the first camera that gives frames."""
    for s in ([src] if src is not None else [str(i) for i in range(4)]):
        cap = cv2.VideoCapture(int(s) if s.isdigit() else s)
        if _has_frames(cap):
            if s.isdigit():  # a local camera, not a file or stream
                print(configure_camera(cap))
            return cap
        cap.release()
    raise SystemExit(f"no frames from camera {src if src is not None else '0-3'}. Cameras that work: "
                     f"{[i for i in range(4) if _camera_works(i)] or 'none'}. Try --camera N (a number from that list), "
                     "and check macOS camera permission for your terminal/IDE.")


GREEN, RED, AMBER, CYAN = (0, 210, 0), (0, 0, 255), (0, 165, 255), (255, 255, 0)


def page_status(frame: np.ndarray, page_src=None) -> tuple:
    """(H, message, ok): the page homography for this frame and a plain-English status.

    page_src is a tracker.load_calibration() result, or None to use the four markers."""
    if page_src is not None:
        H = page_src.homography(frame)
        if H is None:
            return None, f"PAGE NOT FOUND: {page_src.status}", False
        return H, page_src.status, "LOST" not in page_src.status
    H, note = diagnose_markers(frame)
    if H is None:
        return None, f"PAGE NOT FOUND: {note}", False
    return H, f"page OK ({note})" if note else "page OK (4 markers)", True


def draw_hud(view: np.ndarray, lines: list, y: int = 28, scale: float = 0.65) -> None:
    """Draw (text, colour) status lines top-left with a dark outline, wrapped to the frame width."""
    width = max(20, int(view.shape[1] / (20 * scale)))
    for text, color in lines:
        for part in textwrap.wrap(text, width) or [""]:
            cv2.putText(view, part, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, part, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            y += int(38 * scale)


class _Detector(threading.Thread):
    """Runs the model on the newest frame in the background, so the video stays smooth."""

    def __init__(self, conf: float, imgsz="auto", enhance: str = "auto", sheet: Optional[list] = None):
        super().__init__(daemon=True)
        self.conf, self.imgsz, self.enhance, self.frame, self.lock = conf, imgsz, enhance, None, threading.Lock()
        self.sheet = sheet  # cells of a known printed sheet: when given and the page is registered, read it by observation
        self.observed, self.match = [], None  # in that mode: the observed cells, and sheetread.compare() against the sheet
        self.boxes: list = []  # latest (x1, y1, x2, y2, label, confidence) in image pixels
        self.seconds, self.error, self.version, self.H_used, self.H_next = 0.0, "", 0, None, None
        self.start()

    def submit(self, frame: np.ndarray, H: Optional[np.ndarray] = None) -> None:
        """Hand over the newest frame, and the page homography that goes with it (None if the page isn't registered)."""
        with self.lock:
            self.frame, self.H_next = frame, H

    def snapshot(self) -> tuple:
        """(boxes, homography of the frame they came from, version) - consistent, and version changes per scan."""
        with self.lock:
            return self.boxes, self.H_used, self.version

    def run(self) -> None:
        while True:
            with self.lock:
                frame, self.frame, H = self.frame, None, self.H_next
            if frame is None:
                time.sleep(0.02)
                continue
            t0 = time.time()
            try:
                if self.sheet is not None and H is not None:
                    from sheetread import compare, display_boxes, observe  # the sheet's dot positions are known: see which are raised
                    observed = observe(frame, H, self.sheet)
                    boxes, error = display_boxes(observed, H), ""
                    self.observed, self.match = observed, compare(observed, self.sheet)
                else:
                    boxes, error = detect_boxes(frame, self.conf, self.imgsz, self.enhance), ""
            except Exception as e:  # e.g. weights missing: show it on screen instead of dying silently
                boxes, error = [], f"DETECTOR ERROR: {e}"
            with self.lock:
                self.boxes, self.error, self.H_used, self.version = boxes, error, H, self.version + 1
            self.seconds = time.time() - t0


def page_source_from_args(a):
    """--paper -> the A4 sheet's own edges; --auto-page W H -> AutoPage; --calib FILE -> saved calibration;
    otherwise the markers, kept alive by RobustPage when they go out of view (--markers-only for the plain behaviour)."""
    from page import A4_MM, SHEET_ORIGIN_MM
    from tracker import RobustPage

    paper = None
    if getattr(a, "paper", False):
        paper = AutoPage(*A4_MM, origin=(-SHEET_ORIGIN_MM[0], -SHEET_ORIGIN_MM[1]), track=False)
        return paper if getattr(a, "markers_only", False) else RobustPage(paper_fallback=paper)
    if a.auto_page:
        return AutoPage(*a.auto_page)
    if a.calib:
        return load_calibration(a.calib)
    return None if getattr(a, "markers_only", False) else RobustPage()


def draw_page_outline(view: np.ndarray, H: np.ndarray, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0)) -> None:
    """Outline the page the program believes it found (magenta), so you can see if it matches the real page."""
    x0, y0 = origin
    pts = np.int32([to_image(H, x, y) for x, y in ((x0, y0), (x0 + w_mm, y0), (x0 + w_mm, y0 + h_mm), (x0, y0 + h_mm))])
    cv2.polylines(view, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)


# Where each dot sits inside a cell box, as (across, down) fractions; dots 1-3 left column, 4-6 right.
DOT_UV = {1: (0.25, 0.15), 2: (0.25, 0.5), 3: (0.25, 0.85), 4: (0.75, 0.15), 5: (0.75, 0.5), 6: (0.75, 0.85)}
_LETTER = {frozenset(int(ch) for ch in dots): letter for letter, dots in _ALPHABET.items()}  # plain-letter reading, if any


def letter_of(dots: Iterable[int]) -> Optional[str]:
    """The plain (uncontracted) letter a dot set spells, or None if it isn't a letter (a-z)."""
    return _LETTER.get(frozenset(dots))


def cell_letter_text(cell: Cell, printed: Optional[str] = None) -> str:
    """What to draw over a cell: the printed sheet's letter when we have one, else the live-read letter, else "?".

    On a known sheet the printed layout (from the sheet photos / first scan) is ground truth. A finger that hides a
    dot still produces some other letter-shaped pattern; that must not rename the cell on the video."""
    if printed:
        return printed.upper()
    return (letter_of(cell["dots"]) or "?").upper()


def draw_cell(view: np.ndarray, quad, cell: Cell, color, labels: bool = False, dots: bool = True, letters: bool = False,
              letter_text: Optional[str] = None) -> None:
    """Draw a cell's box (quad = TL, TR, BR, BL in pixels) plus red dots where the detector says the dots are.

    The red dots should sit on the real dots underneath: a real dot with no red dot on it is a miss.
    `letter_text` is the printed letter to draw when the live-read dots don't spell a letter (see cell_letter_text)."""
    tl, tr, br, bl = [np.float32(p) for p in quad]
    cv2.polylines(view, [np.int32([tl, tr, br, bl])], True, color, 1, cv2.LINE_AA)
    r = max(2, int(np.linalg.norm(tr - tl) / 9))
    for dot in (cell["dots"] if dots else ()):
        u, v = DOT_UV[dot]
        p = tl + (tr - tl) * u + (bl - tl) * v
        cv2.circle(view, (int(p[0]), int(p[1])), r, RED, -1, cv2.LINE_AA)
    if labels:
        cv2.putText(view, "".join(map(str, sorted(cell["dots"]))), (int(tl[0]), int(tl[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
    if letters:  # the reading itself, large and outlined so it can be read on any background
        text = cell_letter_text(cell, letter_text)
        scale = float(np.clip(np.linalg.norm(tr - tl) / 30.0, 0.5, 1.1))
        pos = (int(tl[0]), int(tl[1]) - int(6 + 8 * scale))
        cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)


def draw_detections(view: np.ndarray, boxes: list, color=None, letters: bool = True) -> int:
    """Draw what the detector read straight onto the camera image (pixel space, so no page registration is needed):
    a box, red dots and the letter for each cell. Returns how many cells were drawn."""
    cells = cells_from_boxes(boxes, np.eye(3))
    for c in cells:
        x1, y1, x2, y2 = c["x"] - c["w"] / 2, c["y"] - c["h"] / 2, c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
        draw_cell(view, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], c, color or CYAN, letters=letters)
    return len(cells)


_DECODED: dict = {"key": None, "lines": []}


def decoded_lines(cells: list) -> list:
    """The cells read as contracted English, one string per row. Cached, because the readout redraws every video frame."""
    key = tuple((c["row"], c["col"], c["label"]) for c in cells)
    if _DECODED["key"] != key:
        try:
            from reader import decode_lines
            _DECODED["lines"] = decode_lines(cells)
        except Exception:  # decoding is a convenience: never let it break the live view
            _DECODED["lines"] = []
        _DECODED["key"] = key
    return _DECODED["lines"]


def render_readout(cells: list, max_rows: int = 12, max_cols: int = 40) -> np.ndarray:
    """A picture of what was detected: each cell as its 2x3 dot pattern, row by row, with the plain letter under it.

    Unsure cells (low detector confidence, or scans that disagree when voting) get an orange background.
    Letters assume plain (uncontracted) braille; "?" means the pattern isn't a plain letter."""
    rows = [r[:max_cols] for r in rows_of(cells)[:max_rows]]
    pitch, top = 46, 34
    text = decoded_lines(cells)[:8]
    glyph_bottom = top + max(len(rows), 1) * 66 + 10
    img = np.full((glyph_bottom + (34 + 22 * len(text) if text else 0),
                   max(620 if text else 420, 60 + pitch * max((len(r) for r in rows), default=0)), 3), 255, np.uint8)
    if text:  # the same cells read as contracted English: what the page actually says, unlike the letter guesses above
        cv2.putText(img, "reads as contracted English (spell-checked; contractions covered are listed in contractions.py):",
                    (8, glyph_bottom + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
        for i, line in enumerate(text):
            cv2.putText(img, f"{i}: {line[:78]}", (8, glyph_bottom + 34 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"detected: {len(cells)} cells, {len(rows_of(cells))} rows (orange = unsure)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    for ri, row in enumerate(rows):
        y0 = top + ri * 66
        cv2.putText(img, str(ri), (6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        for ci, c in enumerate(row):
            x0 = 34 + ci * pitch
            if c["confidence"] < 0.6:
                cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + 30, y0 + 58), (200, 230, 255), -1)
            for dot, (u, _) in DOT_UV.items():
                center = (x0 + 6 + int(u > 0.5) * 14, y0 + 6 + ((dot - 1) % 3) * 14)
                on = dot in c["dots"]
                cv2.circle(img, center, 4 if on else 2, (0, 0, 0) if on else (200, 200, 200), -1, cv2.LINE_AA)
            cv2.putText(img, _LETTER.get(c["dots"], "?"), (x0 + 6, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def sheet_check_line(match: tuple) -> tuple:
    """(text, colour) saying how many cells of a known sheet were read as the sheet says, and the first few that were not."""
    good, total, bad = match
    if not bad:
        return f"sheet check: all {total} cells read correctly", GREEN
    detail = "; ".join(f"row {r + 1} col {c + 1}: expected {dots_to_char(e)} read {dots_to_char(o) if o else 'blank'}" for r, c, e, o in bad[:3])
    return f"sheet check: {good}/{total} cells correct ({detail})", AMBER if good >= 0.8 * total else RED


def locked_check_line(results: list, expected: list) -> tuple:
    """(text, colour) for a known sheet read through a CellLocker: how many cells are locked in as correct, how many are still
    being read, and the first few that are wrong."""
    total = len(expected)
    locked = sum(1 for r in results if r.get("locked"))
    wrong = [(e["row"], e["col"], e["dots"], r["dots"]) for r, e in zip(results, expected) if not r.get("locked") and r["dots"] != e["dots"]]
    if locked == total:
        return f"sheet check: all {total} cells locked in correct (u = read again)", GREEN
    reading = total - locked - len(wrong)
    text = f"sheet check: {locked}/{total} locked"
    if reading:
        text += f", {reading} still reading"
    if wrong:
        text += f", {len(wrong)} wrong (" + "; ".join(
            f"row {r + 1} col {c + 1}: expected {dots_to_char(e)} read {dots_to_char(o) if o else 'blank'}" for r, c, e, o in wrong[:3]) + ")"
    return text, RED if len(wrong) > 0.2 * total else AMBER


def braille_status(det: "_Detector", n_cells: int) -> tuple:
    """(text, colour) line describing what the detector is doing."""
    if det.error:
        return det.error, RED
    if det.seconds == 0.0:
        return "loading the braille detector...", CYAN
    if n_cells == 0:
        return f"no braille detected in view ({det.seconds:.1f} s/scan)", AMBER
    ok, note = reading_quality(det.boxes)
    if note:
        return f"{n_cells} cells, but {note}", GREEN if ok else AMBER
    return f"{n_cells} braille cells detected ({det.seconds:.1f} s/scan)", GREEN


def _identity_if_none(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    H = page_homography(frame)
    return (H, True) if H is not None else (np.eye(3), False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="photo to scan")
    ap.add_argument("--annotate", action="store_true", help="save IMAGE with boxes drawn (IMAGE_annotated.png)")
    ap.add_argument("--live", action="store_true", help="webcam with live green boxes on detected braille; s saves cells.json, q quits")
    ap.add_argument("--vote", type=int, default=8, help="in --live, pool this many recent scans so labels stop flickering (1 = off)")
    ap.add_argument("--rows", type=int, default=0, help="in --live, show and save only the first N lines (rows) of braille")
    ap.add_argument("--phone-url", help="phone camera app's address, e.g. http://192.168.1.5:8080 (Android IP Webcam style); "
                    "used for the video (BASE/video) and for the focus and torch commands")
    ap.add_argument("--torch", action="store_true", help="with --phone-url: switch the phone's flash/torch on (off again on exit)")
    ap.add_argument("--print", action="store_true", help="in --live, also stream the detected rows to the terminal as braille characters")
    ap.add_argument("--no-dots", action="store_true", help="in --live, don't draw the red predicted dots on the video")
    ap.add_argument("--verified", metavar="CELLS_JSON",
                    help="in --live, start with the cells in this file already locked (ones you checked against the page by eye)")
    ap.add_argument("--labels", action="store_true", help="in --live, also print each box's dot numbers")
    ap.add_argument("--camera", default=None, help="camera index, stream URL or video file (default: first that works)")
    ap.add_argument("--calib", help="calibration.json from calibrate.py, instead of the four markers (tracks the page if calibration.png is beside it)")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"),
                    help="no markers: find the page edges automatically; give the page's real width and height in mm")
    ap.add_argument("--markers-only", action="store_true",
                    help="require all four markers in every frame (default: keep following the sheet when they go out of view)")
    ap.add_argument("--sheet", choices=("alphabet", "words", "numbers", "lookalikes"),
                    help="read one of our printed sheets by looking at its known dot positions (needs the page located); "
                         "much more reliable than the general detector, and shows how many cells match the sheet")
    ap.add_argument("--paper", action="store_true",
                    help="no markers: find the edges of the A4 printed sheet itself (dark plain desk, all four edges in view)")
    ap.add_argument("--enhance", choices=("auto", "on", "off"), default="auto",
                    help="boost faint embossed shading before reading: auto tries both and keeps the more confident (default)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=lambda v: v if v == "auto" else int(v), default="auto",
                    help="model input size in pixels, or auto (default): sized to the braille in view")
    a = ap.parse_args()

    if a.phone_url:
        if a.camera is None:
            a.camera = a.phone_url.rstrip("/") + "/video"
        print(phone_command(a.phone_url, "focus"))
        if a.torch:
            print(phone_command(a.phone_url, "torch_on"))
    if a.live:
        from vote import CellLocker, CellVoter, first_rows, keep_inside  # imported here because vote.py imports this module

        from sheets import get_sheet  # a known printed sheet, if one was named: read it by observation instead of detection

        sheet_cells = get_sheet(a.sheet).cells if a.sheet else None
        cap, page_src, det = open_camera(a.camera), page_source_from_args(a), _Detector(a.conf, a.imgsz, a.enhance, sheet=sheet_cells)
        voter, seen_version, last_text, last_print = CellVoter(a.vote), 0, "", 0.0
        locker = CellLocker()  # holds a reading once it can be trusted, so a shaky frame cannot flicker it
        if a.verified:  # cells checked by eye against the page: locked from the very start
            locker.prelock(load_cells(a.verified))
            print(f"{locker.locked_count} verified cells locked from {a.verified}", flush=True)
        stable: list = []
        size = tuple(a.auto_page) if a.auto_page else (None if a.calib else (PAGE_W_MM, PAGE_H_MM))  # page size in mm, if known
        while True:
            ok, frame = cap.read()
            if not ok:
                print("the camera stopped giving frames")
                break
            H, msg, page_ok = page_status(frame, page_src)
            det.submit(frame, H if page_ok else None)
            boxes, H_scan, version = det.snapshot()
            if version != seen_version:  # a new scan finished: add it to the vote, in page mm so camera motion doesn't matter
                seen_version = version
                if H_scan is None:
                    voter.reset()  # (a lost page never unlocks anything: locked cells wait for the page to come back)
                elif sheet_cells is not None and det.observed:
                    stable = locker.update_known(det.observed, sheet_cells)  # known sheet: lock once it reads as the sheet says
                else:
                    scan = cells_from_boxes(boxes, H_scan)
                    voter.add(keep_inside(scan, *size) if size else scan)
                    stable = locker.update(voter.result())  # unknown page: lock on a steady majority
            view = frame.copy()  # draw on a copy so the overlay never reaches the detector
            if page_ok and H is not None:
                cells = stable if (sheet_cells is not None or a.verified) else _assign_grid(list(stable))
                cells = first_rows(cells, a.rows) if a.rows else cells
                for c in cells:
                    quad = [to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2)
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    draw_cell(view, quad, c, GREEN if c.get("locked") else AMBER, a.labels, not a.no_dots, letters=True)  # green = locked in
                if isinstance(page_src, AutoPage):
                    draw_page_outline(view, H, *page_src.size_mm, page_src.origin)
            else:  # no page yet: show the raw detections in image pixels
                cells = cells_from_boxes(boxes, np.eye(3))
                cells = first_rows(cells, a.rows) if a.rows else cells
                for c in cells:
                    x1, y1, x2, y2 = c["x"] - c["w"] / 2, c["y"] - c["h"] / 2, c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
                    draw_cell(view, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], c, GREEN, a.labels, not a.no_dots, letters=True)
            keys = "s: save cells.json   u: unlock   p: save screenshot   q: quit" + ("   r: re-find page" if isinstance(page_src, AutoPage) else "") + ("   f: refocus phone" if a.phone_url else "")
            lines = [(msg, (AMBER if "warning" in msg else GREEN) if page_ok else RED), braille_status(det, len(cells))]
            if sheet_cells is not None and stable and page_ok:
                lines.append(locked_check_line(stable, sheet_cells))
            elif locker.locked_count and page_ok:
                lines.append((f"{locker.locked_count} cells locked in (u = read again)", GREEN))
            draw_hud(view, lines + [(keys, CYAN)])
            cv2.imshow("braille", view)
            cv2.imshow("detected braille", render_readout(cells))
            text_rows = decoded_lines(cells)
            rows_text = "\n".join(f"row {r}: " + "".join(c["char"] for c in row) + (f"   = {text_rows[r]}" if r < len(text_rows) else "")
                                  for r, row in enumerate(rows_of(cells)))
            if a.print and rows_text != last_text and time.time() - last_print > 1.0:  # stream real braille to the terminal
                print("\n----- detected -----\n" + rows_text, flush=True)
                last_text, last_print = rows_text, time.time()
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):  # for bug reports: the raw camera frame and exactly what the screen showed
                cv2.imwrite("frame.png", frame)
                cv2.imwrite("screenshot.png", view)
                print(f"saved frame.png and screenshot.png. Page status was: {msg}")
            if k == ord("f") and a.phone_url:
                print(phone_command(a.phone_url, "focus"))  # re-trigger autofocus after the page distance changed
            if k == ord("u"):  # let go of everything and read again from scratch
                locker.reset()
                voter.reset()
                stable = []
            if k == ord("r") and isinstance(page_src, AutoPage):
                page_src.unlock()
                voter.reset()
            if k == ord("s"):
                saved = list(stable) if page_ok and H is not None else []
                saved = first_rows(saved, a.rows) if a.rows else saved
                if not page_ok or H is None:
                    print(f"not saved: {msg}")
                elif not saved:
                    print("not saved: no braille detected yet")
                else:
                    print_rows(saved)
                    save_cells(saved, "cells.json")
                    print(f"{len(saved)} cells saved to cells.json (label agreement is stored in each cell's confidence)")
        cap.release()
        cv2.destroyAllWindows()
        if a.phone_url and a.torch:
            print(phone_command(a.phone_url, "torch_off"))
    elif a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            raise SystemExit(f"cannot read {a.image}")
        if a.paper:
            from page import A4_MM, SHEET_ORIGIN_MM
            H, why = find_page_homography(frame, *A4_MM, origin=(-SHEET_ORIGIN_MM[0], -SHEET_ORIGIN_MM[1]))
        elif a.auto_page:
            H, why = find_page_homography(frame, *a.auto_page)
        elif a.calib:
            H, why = load_calibration(a.calib).homography(frame), "calibration"
        else:
            H, why = page_homography(frame), "no page markers found"
        found = H is not None
        if not found:
            print(f"page not registered ({why}): coordinates below are image pixels, not mm")
            H = np.eye(3)
        boxes = detect_boxes(frame, a.conf, a.imgsz, a.enhance)
        cells = cells_from_boxes(boxes, H)
        print_rows(cells)
        ok, note = reading_quality(boxes)
        if note:
            print(('' if ok else 'poor reading conditions: ') + note)
        if a.annotate:
            out = str(Path(a.image).with_name(Path(a.image).stem + "_annotated.png"))
            shown = annotate(frame, cells, H)
            if (a.auto_page or a.paper) and found:
                size, origin = ((210.0, 297.0), (-30.0, -30.0)) if a.paper else (tuple(a.auto_page), (0.0, 0.0))
                draw_page_outline(shown, H, *size, origin)
            cv2.imwrite(out, shown)
            print(f"{len(cells)} cells; wrote {out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
