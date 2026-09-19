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


def cells_from_boxes(boxes: list[tuple], H: np.ndarray, max_overlap: float = 0.3) -> list[Cell]:
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
    return _assign_grid(kept)


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


def scan_page(frame: np.ndarray, H: np.ndarray, conf: float = 0.15, imgsz: int = 640) -> list[Cell]:
    """Detect all cells in one frame (no hand in view). Cells come back in page mm, sorted by row, col."""
    return cells_from_boxes(_detect_pixels(frame, conf, imgsz), H)


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
    return H, "page OK (4 markers)" + (f" - warning: {note}" if note else ""), True


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

    def __init__(self, conf: float, imgsz: int):
        super().__init__(daemon=True)
        self.conf, self.imgsz, self.frame, self.lock = conf, imgsz, None, threading.Lock()
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
                boxes, error = _detect_pixels(frame, self.conf, self.imgsz), ""
            except Exception as e:  # e.g. weights missing: show it on screen instead of dying silently
                boxes, error = [], f"DETECTOR ERROR: {e}"
            with self.lock:
                self.boxes, self.error, self.H_used, self.version = boxes, error, H, self.version + 1
            self.seconds = time.time() - t0


def page_source_from_args(a):
    """--auto-page W H  ->  AutoPage;  --calib FILE  ->  saved calibration;  neither  ->  None (use the four markers)."""
    if a.auto_page:
        return AutoPage(*a.auto_page)
    return load_calibration(a.calib) if a.calib else None


def draw_page_outline(view: np.ndarray, H: np.ndarray, w_mm: float, h_mm: float) -> None:
    """Outline the page the program believes it found (magenta), so you can see if it matches the real page."""
    pts = np.int32([to_image(H, x, y) for x, y in ((0, 0), (w_mm, 0), (w_mm, h_mm), (0, h_mm))])
    cv2.polylines(view, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)


# Where each dot sits inside a cell box, as (across, down) fractions; dots 1-3 left column, 4-6 right.
DOT_UV = {1: (0.25, 0.15), 2: (0.25, 0.5), 3: (0.25, 0.85), 4: (0.75, 0.15), 5: (0.75, 0.5), 6: (0.75, 0.85)}
_LETTER = {frozenset(int(ch) for ch in dots): letter for letter, dots in _ALPHABET.items()}  # plain-letter reading, if any


def draw_cell(view: np.ndarray, quad, cell: Cell, color, labels: bool = False, dots: bool = True) -> None:
    """Draw a cell's box (quad = TL, TR, BR, BL in pixels) plus red dots where the detector says the dots are.

    The red dots should sit on the real dots underneath: a real dot with no red dot on it is a miss."""
    tl, tr, br, bl = [np.float32(p) for p in quad]
    cv2.polylines(view, [np.int32([tl, tr, br, bl])], True, color, 1, cv2.LINE_AA)
    r = max(2, int(np.linalg.norm(tr - tl) / 9))
    for dot in (cell["dots"] if dots else ()):
        u, v = DOT_UV[dot]
        p = tl + (tr - tl) * u + (bl - tl) * v
        cv2.circle(view, (int(p[0]), int(p[1])), r, RED, -1, cv2.LINE_AA)
    if labels:
        cv2.putText(view, "".join(map(str, sorted(cell["dots"]))), (int(tl[0]), int(tl[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)


def render_readout(cells: list, max_rows: int = 12, max_cols: int = 40) -> np.ndarray:
    """A picture of what was detected: each cell as its 2x3 dot pattern, row by row, with the plain letter under it.

    Unsure cells (low detector confidence, or scans that disagree when voting) get an orange background.
    Letters assume plain (uncontracted) braille; "?" means the pattern isn't a plain letter."""
    rows = [r[:max_cols] for r in rows_of(cells)[:max_rows]]
    pitch, top = 46, 34
    img = np.full((top + max(len(rows), 1) * 66 + 10, max(420, 60 + pitch * max((len(r) for r in rows), default=0)), 3), 255, np.uint8)
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


def braille_status(det: "_Detector", n_cells: int) -> tuple:
    """(text, colour) line describing what the detector is doing."""
    if det.error:
        return det.error, RED
    if det.seconds == 0.0:
        return "loading the braille detector...", CYAN
    if n_cells == 0:
        return f"no braille detected in view ({det.seconds:.1f} s/scan)", AMBER
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
    ap.add_argument("--labels", action="store_true", help="in --live, also print each box's dot numbers")
    ap.add_argument("--camera", default=None, help="camera index, stream URL or video file (default: first that works)")
    ap.add_argument("--calib", help="calibration.json from calibrate.py, instead of the four markers (tracks the page if calibration.png is beside it)")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"),
                    help="no markers: find the page edges automatically; give the page's real width and height in mm")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input size; try 1280 if small cells are missed")
    a = ap.parse_args()

    if a.phone_url:
        if a.camera is None:
            a.camera = a.phone_url.rstrip("/") + "/video"
        print(phone_command(a.phone_url, "focus"))
        if a.torch:
            print(phone_command(a.phone_url, "torch_on"))
    if a.live:
        from vote import CellVoter, first_rows, keep_inside  # imported here because vote.py imports this module

        cap, page_src, det = open_camera(a.camera), page_source_from_args(a), _Detector(a.conf, a.imgsz)
        voter, seen_version, last_text, last_print = CellVoter(a.vote), 0, "", 0.0
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
                    voter.reset()
                else:
                    scan = cells_from_boxes(boxes, H_scan)
                    voter.add(keep_inside(scan, *size) if size else scan)
            view = frame.copy()  # draw on a copy so the overlay never reaches the detector
            if page_ok and H is not None:
                cells = voter.result()
                cells = first_rows(cells, a.rows) if a.rows else cells
                for c in cells:
                    quad = [to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2)
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    draw_cell(view, quad, c, GREEN if c["confidence"] >= 0.6 else AMBER, a.labels, not a.no_dots)  # amber = unsure
                if a.auto_page:
                    draw_page_outline(view, H, *a.auto_page)
            else:  # no page yet: show the raw detections in image pixels
                cells = cells_from_boxes(boxes, np.eye(3))
                cells = first_rows(cells, a.rows) if a.rows else cells
                for c in cells:
                    x1, y1, x2, y2 = c["x"] - c["w"] / 2, c["y"] - c["h"] / 2, c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
                    draw_cell(view, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], c, GREEN, a.labels, not a.no_dots)
            keys = "s: save cells.json   p: save screenshot   q: quit" + ("   r: re-find page" if a.auto_page else "") + ("   f: refocus phone" if a.phone_url else "")
            draw_hud(view, [(msg, (AMBER if "warning" in msg else GREEN) if page_ok else RED),
                            braille_status(det, len(cells)), (keys, CYAN)])
            cv2.imshow("braille", view)
            cv2.imshow("detected braille", render_readout(cells))
            rows_text = "\n".join(f"row {r}: " + "".join(c["char"] for c in row) for r, row in enumerate(rows_of(cells)))
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
            if k == ord("r") and isinstance(page_src, AutoPage):
                page_src.unlock()
                voter.reset()
            if k == ord("s"):
                saved = voter.result() if page_ok and H is not None else []
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
        if a.auto_page:
            H, why = find_page_homography(frame, *a.auto_page)
        elif a.calib:
            H, why = load_calibration(a.calib).homography(frame), "calibration"
        else:
            H, why = page_homography(frame), "no page markers found"
        found = H is not None
        if not found:
            print(f"page not registered ({why}): coordinates below are image pixels, not mm")
            H = np.eye(3)
        cells = scan_page(frame, H, a.conf, a.imgsz)
        print_rows(cells)
        if a.annotate:
            out = str(Path(a.image).with_name(Path(a.image).stem + "_annotated.png"))
            shown = annotate(frame, cells, H)
            if a.auto_page and found:
                draw_page_outline(shown, H, *a.auto_page)
            cv2.imwrite(out, shown)
            print(f"{len(cells)} cells; wrote {out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
