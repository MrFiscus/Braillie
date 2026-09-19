"""Automatic page detection without markers: find the page as the big bright four-sided shape against the desk.

Assumes the page sits upright in the picture (top of the page towards the top of the image) and is roughly flat.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from page import homography_from_corners
from tracker import PageTracker

WORK_W = 640  # detection runs on a copy this wide
MIN_AREA = 0.10  # the page must cover at least this fraction of the frame
ASPECT_TOL = 0.40  # allowed relative error of the page's width/height ratio (perspective squeezes it a little)
MIN_CONTRAST = 25  # gray levels (0-255) between the page and the desk around it
MIN_EDGE_SUPPORT = 0.4  # each page side must have this fraction of its length on a clean straight edge
STEADY_FRAMES = 5  # edges must hold still this many frames before we lock on


def order_corners(pts) -> np.ndarray:
    """Four points -> top left, top right, bottom right, bottom left (page upright in the image)."""
    p = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = p.sum(axis=1), p[:, 1] - p[:, 0]
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def _candidates(gray: np.ndarray):
    """Yield contours of large regions found by several thresholds (both polarities) and by edges."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    masks = []
    for t in (otsu, *np.percentile(blur, (35, 50, 65))):
        masks += [(blur > t).astype(np.uint8) * 255, (blur <= t).astype(np.uint8) * 255]
    masks.append(cv2.morphologyEx(cv2.Canny(blur, 30, 90), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yield from cnts


def _contrast(gray: np.ndarray, quad: np.ndarray) -> float:
    """Mean brightness inside the quad minus mean of a band just outside it, in gray levels (positive = brighter page)."""
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.int32(quad), 255)
    k = max(3, gray.shape[1] // 60)
    core = cv2.erode(inside, np.ones((k, k), np.uint8))
    ring = cv2.dilate(inside, np.ones((4 * k, 4 * k), np.uint8)) & ~cv2.dilate(inside, np.ones((k, k), np.uint8))
    if not core.any() or not ring.any():
        return 0.0
    return float(gray[core > 0].mean()) - float(gray[ring > 0].mean())


def _refine(gray: np.ndarray, q: np.ndarray, search: float, samples: int = 40) -> tuple:
    """Snap a rough quad to the real page edges: fit a line to the strongest edge along each side, intersect neighbours.

    Returns (corners, support) where support[i] is the fraction of sample points on side i that sit on a strong edge
    within a few pixels of the fitted straight line (low = not a clean straight edge)."""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    R = max(4, int(search * w))
    offs = np.arange(-R, R + 1, dtype=np.float32)
    lines, support = [], []
    tol = max(3.0, 0.004 * w)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-6)
        n = np.array([-d[1], d[0]], np.float32)
        t = np.linspace(0.1, 0.9, samples, dtype=np.float32)[:, None]
        centres = a + (b - a) * t  # samples x 2
        xs = centres[:, 0:1] + offs[None] * n[0]
        ys = centres[:, 1:2] + offs[None] * n[1]
        prof = cv2.remap(g, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        grad = np.abs(np.diff(prof, axis=1))
        k = grad.argmax(axis=1)
        strong = grad[np.arange(samples), k] > 6
        if strong.sum() < 10:
            return q, [0.0] * 4
        pts = np.stack([xs[np.arange(samples), k], ys[np.arange(samples), k]], axis=1)[strong].astype(np.float32)
        line = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()  # vx, vy, x0, y0
        lines.append(line)
        dist = np.abs((pts[:, 0] - line[2]) * line[1] - (pts[:, 1] - line[3]) * line[0])
        support.append(float((dist < tol).sum()) / samples)
    out = []
    for i in range(4):
        (vx0, vy0, x0, y0), (vx1, vy1, x1, y1) = lines[i - 1], lines[i]
        det = vx0 * (-vy1) - vy0 * (-vx1)
        if abs(det) < 1e-3:
            return q, support
        tt = ((x1 - x0) * (-vy1) - (y1 - y0) * (-vx1)) / det
        out.append([x0 + tt * vx0, y0 + tt * vy0])
    out = np.float32(out)
    return (out, support) if np.abs(out - q).max() < 2 * R else (q, support)


def find_page(frame: np.ndarray, w_mm: float, h_mm: float) -> tuple:
    """(corners, reason): the page's four corners in image pixels (TL, TR, BR, BL), or (None, why not)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    s = WORK_W / gray.shape[1]
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H_s, W_s = small.shape
    want = w_mm / h_mm
    best, best_score, worst_shape, low_contrast, border, seen_big, darker = None, 0.0, None, -255.0, False, False, False
    for c in _candidates(small):
        area = cv2.contourArea(c)
        if area < MIN_AREA * small.size:
            continue
        seen_big = True
        x, y, w, h = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + w >= W_s - 2 or y + h >= H_s - 2:
            border = True  # a region running off the frame is the desk or a cropped page, not a whole page
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                break
        else:
            continue
        q = order_corners(approx)
        w_px = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h_px = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        ratio = w_px / max(h_px, 1e-6)
        if abs(ratio / want - 1) > ASPECT_TOL:
            worst_shape = ratio
            continue
        contrast = _contrast(small, q)
        if contrast < MIN_CONTRAST:  # paper is brighter than what it lies on; a darker rectangle is a folder, mat or desk
            if contrast < 0:
                darker = True
            low_contrast = max(low_contrast, contrast)
            continue
        score = cv2.contourArea(hull) * contrast
        if score > best_score:
            best, best_score = q / s, score
    if best is not None:
        for search in (0.04, 0.015):  # coarse then fine snap to the true edges
            best, support = _refine(gray, best, search)
        names = ("top", "right", "bottom", "left")
        weak = [n for n, sup in zip(names, support) if sup < MIN_EDGE_SUPPORT]
        if weak:
            return None, (f"the {'/'.join(weak)} edge of the page isn't a clean straight line (something touching the page, "
                          "a curled edge, or a hand?). Clear the area around the page, or click the corners with calibrate.py.")
        return best, ""
    if worst_shape is not None:
        return None, (f"found a four-sided shape with width:height {worst_shape:.2f}, but the page is {want:.2f}. "
                      "Two pages in view, page rotated sideways, wrong --auto-page size, or the page blends into a light desk?")
    if low_contrast > 0:
        return None, (f"a page-shaped region is there but only {low_contrast:.0f} gray levels brighter than the desk "
                      f"(need {MIN_CONTRAST}). Put a plain, clearly darker surface under the book.")
    if darker:
        return None, ("the only rectangles found are darker than their surroundings (a folder or mat, not a white page). "
                      "The page must be brighter than what is around it.")
    if border:
        return None, "only regions running off the edge of the frame were found: page not fully in view, or nothing but desk. Move the camera back so the whole page and some desk show."
    if seen_big:
        return None, "found a big shape but its outline isn't a clean four-sided page (hand over an edge? page curled?)."
    return None, "no large page-shaped region: page out of view, or too little contrast with the desk."


def find_page_homography(frame: np.ndarray, w_mm: float, h_mm: float) -> tuple:
    """(H, reason): image px -> page mm (top-left page corner = 0, 0), or (None, why not)."""
    q, why = find_page(frame, w_mm, h_mm)
    return (None, why) if q is None else (homography_from_corners(q, w_mm, h_mm), "")


class AutoPage:
    """Page source for `--auto-page`: looks for the page edges, locks on once they hold steady, then tracks the page."""

    def __init__(self, w_mm: float, h_mm: float):
        self.size_mm, self.tracker, self.prev, self.steady, self.reason = (w_mm, h_mm), None, None, 0, ""

    def unlock(self) -> None:
        """Forget the current page and look for the edges again (bound to the r key)."""
        self.tracker, self.prev, self.steady = None, None, 0

    @property
    def status(self) -> str:
        if self.tracker is not None:
            return self.tracker.status
        if self.steady:
            return f"page edges found, holding steady ({self.steady}/{STEADY_FRAMES})"
        return f"looking for the page edges: {self.reason}"

    def homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Image px -> page mm for this frame, or None while no page has been found."""
        if self.tracker is not None:
            return self.tracker.homography(frame)
        q, self.reason = find_page(frame, *self.size_mm)
        if q is None:
            self.prev, self.steady = None, 0
            return None
        moved = self.prev is None or np.abs(q - self.prev).max() > 4
        self.steady, self.prev = 1 if moved else self.steady + 1, q
        H = homography_from_corners(q, *self.size_mm)
        if self.steady >= STEADY_FRAMES:
            self.tracker = PageTracker(frame, H)  # from now on follow the page's texture, not its edges
        return H
