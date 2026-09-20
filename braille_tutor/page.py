"""Page registration: map webcam pixels to page millimeters using four ArUco corner markers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Marker centre to marker centre distances. These match the sheet from make_sheet.py (A4, markers 30 mm
# in from each edge). If you make your own sheet, MEASURE IT and edit these.
PAGE_W_MM = 150.0
PAGE_H_MM = 237.0

# The printed sheets are A4, and the marker rectangle sits centred on it, so paper coordinates and page coordinates differ
# by a fixed shift: the top-left marker's centre is SHEET_ORIGIN_MM in from the paper's top-left corner.
A4_MM = (210.0, 297.0)
SHEET_ORIGIN_MM = ((A4_MM[0] - PAGE_W_MM) / 2, (A4_MM[1] - PAGE_H_MM) / 2)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
# id 0 top left, 1 top right, 2 bottom right, 3 bottom left -> page coordinates (mm), y down.
MARKER_POS_MM = {0: (0.0, 0.0), 1: (PAGE_W_MM, 0.0), 2: (PAGE_W_MM, PAGE_H_MM), 3: (0.0, PAGE_H_MM)}
MARKER_SIZE_MM = 40.0  # printed marker side; must match make_markers.py's SHEET_MARKER_MM
# fewer point pairs than this makes findHomography's fit too loosely constrained to trust
MIN_MARKERS_FOR_HOMOGRAPHY = 3

_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
)


def visible_marker_corners(frame: np.ndarray) -> dict:
    """Pixel corners (4x2, ArUco's own corner order) of the corner markers (ids 0-3) currently visible, keyed by id."""
    corners, ids, _ = _detector.detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in MARKER_POS_MM}


def visible_markers(frame: np.ndarray) -> dict:
    """Pixel centres of the corner markers (ids 0-3) currently visible, keyed by id."""
    return {i: c.mean(axis=0) for i, c in visible_marker_corners(frame).items()}


def _marker_corners_mm(marker_id: int) -> np.ndarray:
    """The four corners of one printed marker in page mm, in ArUco's own corner order (top-left, top-right,
    bottom-right, bottom-left of the marker AS PRINTED). make_sheet.py pastes every marker unrotated, so that
    printed orientation is also its orientation on the page, whatever angle the camera views it from."""
    cx, cy = MARKER_POS_MM[marker_id]
    half = MARKER_SIZE_MM / 2
    return np.float32([(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])


def homography_from_marker_corners(seen: dict) -> Optional[np.ndarray]:
    """A homography fit from whichever markers are visible (id -> its 4 corners in image px, as from
    visible_marker_corners), using every corner as its own point correspondence rather than collapsing each
    marker to just its centre. With only 3 of the 4 corner markers in view -- typically because a hand is
    reaching in to point at a cell near one of them -- this still gives an exact, geometry-based registration
    from the other 3 (12 point pairs, well over the 4 a homography needs), instead of losing marker registration
    outright and falling back to much noisier frame-to-frame feature tracking (see PageTracker). The redundant
    point pairs also make the ordinary 4-marker case more robust: RANSAC can shrug off one corner's detection
    jitter instead of fitting it exactly, the way a plain 4-point transform must."""
    if len(seen) < MIN_MARKERS_FOR_HOMOGRAPHY:
        return None
    ids = sorted(seen)
    src = np.concatenate([seen[i] for i in ids]).astype(np.float32)
    dst = np.concatenate([_marker_corners_mm(i) for i in ids]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


def _homography_from_centers(centers: dict) -> np.ndarray:
    src = np.float32([centers[i] for i in range(4)])
    dst = np.float32([MARKER_POS_MM[i] for i in range(4)])
    return cv2.getPerspectiveTransform(src, dst)


def page_homography(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the 3x3 homography (image px -> page mm), or None unless all four markers are visible."""
    centers = visible_markers(frame)
    return _homography_from_centers(centers) if len(centers) == 4 else None


def diagnose_markers(frame: np.ndarray) -> tuple:
    """(homography, note). H is None when the page can't be registered and note says why in plain English.
    When H is present, note is an optional warning (empty if all is well)."""
    corners, ids, rejected = _detector.detectMarkers(frame)
    mean = float(frame.mean())
    light = " Image is very dark." if mean < 40 else " Image is washed out." if mean > 242 else ""
    seen, others, dup = {}, [], []
    for c, i in zip(corners, ids.ravel() if ids is not None else []):
        i = int(i)
        if i not in MARKER_POS_MM:
            others.append(i)
        elif i in seen:
            dup.append(i)
        else:
            seen[i] = c.reshape(4, 2)
    if not seen:
        if others:
            return None, f"found marker ids {sorted(set(others))} but need ids 0-3 (wrong markers?)." + light
        if len(rejected):
            return None, ("square shapes seen but none decode: markers too small, blurry, glared, mirrored, "
                          "or not 4x4 ArUco." + light)
        return None, "no markers in view. Are the 4 printed markers visible, matte and well lit?" + light
    if dup:
        return None, f"marker id {sorted(set(dup))} is seen more than once."
    missing = [i for i in MARKER_POS_MM if i not in seen]
    if missing and len(seen) < MIN_MARKERS_FOR_HOMOGRAPHY:
        return None, f"missing marker(s) {missing}, found {sorted(seen)}." + light
    ctr = {i: seen[i].mean(axis=0) for i in seen}
    for i in seen:  # 0 -> 1 -> 2 -> 3 must turn the same way every time (top left, top right, bottom right, bottom left),
        j, k = (i + 1) % 4, (i + 2) % 4  # checked at every vertex where both of its neighbours are also visible
        if j in ctr and k in ctr:
            a, b = ctr[j] - ctr[i], ctr[k] - ctr[j]
            if a[0] * b[1] - a[1] * b[0] <= 0:
                return None, "markers are in the wrong corners, or the sheet is rotated or flipped (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left)."
    H = homography_from_marker_corners(seen)
    if H is None:
        return None, f"missing marker(s) {missing}, found {sorted(seen)}." + light
    side = min(float(np.linalg.norm(c[0] - c[1])) for c in seen.values())
    parts = ["4 markers"] if not missing else [f"{len(seen)} of 4 markers, {missing} hidden"]
    if side < 25:
        parts.append(f"markers small ({side:.0f} px): move the camera closer.")
    return H, "; ".join(parts) if missing or side < 25 else ""


def _apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = M @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def to_page(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Map an image pixel (px, py) to page millimeters."""
    return _apply(H, px, py)


def to_image(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map page millimeters (x, y) back to an image pixel."""
    return _apply(np.linalg.inv(H), x, y)


def homography_from_corners(corners_px, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0)) -> np.ndarray:
    """Marker-free homography from 4 clicked pixels: top left, top right, bottom right, bottom left of a w x h mm
    rectangle whose top-left corner sits at page position `origin` (mm)."""
    x0, y0 = origin
    dst = np.float32([[x0, y0], [x0 + w_mm, y0], [x0 + w_mm, y0 + h_mm], [x0, y0 + h_mm]])
    return cv2.getPerspectiveTransform(np.float32(corners_px), dst)


def save_homography(H: np.ndarray, path: str) -> None:
    """Store a homography as JSON (used by calibrate.py)."""
    Path(path).write_text(json.dumps(H.tolist()))


def load_homography(path: str) -> np.ndarray:
    """Read a homography written by save_homography."""
    return np.array(json.loads(Path(path).read_text()), dtype=np.float64)
