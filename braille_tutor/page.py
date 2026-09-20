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

_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
)


def visible_markers(frame: np.ndarray) -> dict:
    """Pixel centres of the corner markers (ids 0-3) currently visible, keyed by id."""
    corners, ids, _ = _detector.detectMarkers(frame)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.ravel(), corners) if int(i) in MARKER_POS_MM}


# Real registrations of this sheet, from markers or from tracking, come out around 1e4 to 1e5 (the mapping carries both the
# mm-per-pixel scale and the page offset). A fit onto a line or a point comes out around 1e18, so anything past this is broken.
MAX_CONDITION = 1e10


def usable_homography(H: Optional[np.ndarray]) -> bool:
    """Is this mapping safe to use at all?

    Four marker centres that are almost in a line (the sheet seen edge-on, a marker read a few pixels off) still fit a
    matrix, but one that squashes the whole page onto a line or a point. Nothing detects that later: to_page and to_image
    quietly divide by ~0 and return positions in the millions, or NaN, and whatever they are handed to -- the reader, the
    overlay, the fingertip -- then raises deep inside OpenCV or numpy. Catch it here, where there is still a good answer
    available (the last known position), rather than there, where there is not.
    """
    if H is None:
        return False
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return False
    spread = np.linalg.svd(H, compute_uv=False)
    return bool(spread[-1] > 0 and spread[0] / spread[-1] < MAX_CONDITION)


def _homography_from_centers(centers: dict) -> Optional[np.ndarray]:
    """Image px -> page mm from the four marker centres, or None if they do not place the page usably (see usable_homography)."""
    src = np.float32([centers[i] for i in range(4)])
    dst = np.float32([MARKER_POS_MM[i] for i in range(4)])
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:  # four points that cannot be told apart at all
        return None
    return H if usable_homography(H) else None


def page_homography(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the 3x3 homography (image px -> page mm), or None unless all four markers are visible and place the page usably."""
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
    if missing:
        return None, f"missing marker(s) {missing}, found {sorted(seen)}." + light
    ctr = [seen[i].mean(axis=0) for i in range(4)]
    for k in range(4):  # 0 -> 1 -> 2 -> 3 must turn the same way every time (top left, top right, bottom right, bottom left)
        a, b = ctr[(k + 1) % 4] - ctr[k], ctr[(k + 2) % 4] - ctr[(k + 1) % 4]
        if a[0] * b[1] - a[1] * b[0] <= 0:
            return None, "markers are in the wrong corners, or the sheet is rotated or flipped (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left)."
    H = _homography_from_centers(dict(zip(range(4), ctr)))
    if H is None:
        return None, "all four markers are in view but too squashed together to place the page: face the sheet more squarely." + light
    side = min(float(np.linalg.norm(c[0] - c[1])) for c in seen.values())
    return H, (f"markers small ({side:.0f} px): move the camera closer." if side < 25 else "")


def _apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = M @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def to_page(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Map an image pixel (px, py) to page millimeters."""
    return _apply(H, px, py)


def to_image(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map page millimeters (x, y) back to an image pixel."""
    return _apply(np.linalg.inv(H), x, y)


def homography_from_corners(corners_px, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0)) -> Optional[np.ndarray]:
    """Marker-free homography from 4 clicked pixels: top left, top right, bottom right, bottom left of a w x h mm
    rectangle whose top-left corner sits at page position `origin` (mm). None if those four points place the page
    nowhere usable (see usable_homography)."""
    x0, y0 = origin
    dst = np.float32([[x0, y0], [x0 + w_mm, y0], [x0 + w_mm, y0 + h_mm], [x0, y0 + h_mm]])
    try:
        H = cv2.getPerspectiveTransform(np.float32(corners_px), dst)
    except cv2.error:
        return None
    return H if usable_homography(H) else None


def save_homography(H: np.ndarray, path: str) -> None:
    """Store a homography as JSON (used by calibrate.py)."""
    Path(path).write_text(json.dumps(H.tolist()))


def load_homography(path: str) -> np.ndarray:
    """Read a homography written by save_homography."""
    return np.array(json.loads(Path(path).read_text()), dtype=np.float64)
