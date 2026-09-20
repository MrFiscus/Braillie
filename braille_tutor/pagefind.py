"""Automatic page detection without markers: find the page as the big bright four-sided shape against the desk.

Assumes the page sits upright in the picture (top of the page towards the top of the image) and is roughly flat.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from page import homography_from_corners
from tracker import PageTracker

WORK_W = 640  # detection runs on a copy this wide
MIN_AREA = 0.10  # the page must cover at least this fraction of the frame
ASPECT_TOL = 0.40  # allowed relative error of the page's width/height ratio (perspective squeezes it a little)
MIN_CONTRAST = 25  # gray levels (0-255) between the page and the desk around it
MIN_SIDE_CONTRAST = 6  # along every side, the paper just inside must be this much brighter than the desk just outside
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


def _find_page_by_regions(frame: np.ndarray, w_mm: float, h_mm: float) -> tuple:
    """Strategy 1: find the page as a bright region and approximate its outline by four corners."""
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


def _canonical_lines(edges: np.ndarray, width: int) -> tuple:
    """Straight lines in an edge map, as (nx, ny, rho, votes): horizontal-ish ones and vertical-ish ones, near-duplicates merged.

    Each line is n . p = rho with the normal n pointing down (horizontal-ish) or right (vertical-ish)."""
    found = cv2.HoughLines(edges, 1, np.pi / 360, int(0.14 * width))  # half-degree steps: tilted edges keep their votes together
    horizontal, vertical = [], []
    if found is None:
        return horizontal, vertical
    for rho, theta in found[:, 0][:250]:
        nx, ny = float(np.cos(theta)), float(np.sin(theta))
        family, ok = (horizontal, abs(nx) < 0.5) if abs(ny) >= abs(nx) else (vertical, abs(ny) < 0.5)
        if not ok:
            continue
        if (family is horizontal and ny < 0) or (family is vertical and nx < 0):
            nx, ny, rho = -nx, -ny, -float(rho)
        angle = np.arctan2(ny, nx)
        if all(abs(rho - r2) > 0.02 * width or abs(angle - a2) > np.radians(6) for _, _, r2, a2 in family):
            family.append((nx, ny, float(rho), angle))
    return [(nx, ny, r) for nx, ny, r, _ in horizontal[:6]], [(nx, ny, r) for nx, ny, r, _ in vertical[:6]]


def _find_page_by_lines(gray: np.ndarray, small: np.ndarray, s: float, want: float) -> Optional[np.ndarray]:
    """Strategy 2: fit the page's four straight edges directly. Survives an object hiding part of an edge or a corner,
    because each edge is judged by how much of it is really there, not by a blob outline."""
    H_s, W_s = small.shape
    blur = cv2.GaussianBlur(small, (11, 11), 0)  # smooth away the page's own texture, keep its outline
    horizontal, vertical = _canonical_lines(cv2.Canny(blur, 30, 90), W_s)
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    def y_at_centre(line):
        return (line[2] - line[0] * W_s / 2) / line[1]

    def x_at_centre(line):
        return (line[2] - line[1] * H_s / 2) / line[0]

    def meet(h, v):
        x, y = np.linalg.solve([[h[0], h[1]], [v[0], v[1]]], [h[2], v[2]])
        return [x, y]

    horizontal.sort(key=y_at_centre)
    vertical.sort(key=x_at_centre)
    found = []
    for i, top in enumerate(horizontal):
        for bottom in horizontal[i + 1:]:
            if y_at_centre(bottom) - y_at_centre(top) < 0.3 * H_s:
                continue
            for j, left in enumerate(vertical):
                for right in vertical[j + 1:]:
                    if x_at_centre(right) - x_at_centre(left) < 0.3 * W_s:
                        continue
                    try:
                        q = np.float32([meet(top, left), meet(top, right), meet(bottom, right), meet(bottom, left)])
                    except np.linalg.LinAlgError:
                        continue
                    if not (q[:, 0].min() > 2 and q[:, 1].min() > 2 and q[:, 0].max() < W_s - 3 and q[:, 1].max() < H_s - 3):
                        continue
                    area = cv2.contourArea(q)
                    w_px = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
                    h_px = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
                    if (area < MIN_AREA * small.size or not cv2.isContourConvex(q.reshape(-1, 1, 2))
                            or abs(w_px / max(h_px, 1e-6) / want - 1) > ASPECT_TOL):
                        continue
                    contrast = _contrast(small, q)
                    if contrast >= MIN_CONTRAST and min(_side_contrasts(small, q)) >= MIN_SIDE_CONTRAST:
                        found.append((area * contrast, q))
    best, best_score = None, 0.0
    for _, q in sorted(found, key=lambda f: -f[0])[:8]:  # the biggest, brightest few are checked edge by edge
        q, support = q / s, None
        for search in (0.04, 0.015):
            q, support = _refine(gray, q, search)
        if min(support) < MIN_EDGE_SUPPORT:
            continue
        score = cv2.contourArea(q * s) * _contrast(small, q * s) * min(support) * _fill(small, q * s) ** 10
        if score > best_score:
            best, best_score = q, score
    return best


def _side_contrasts(gray: np.ndarray, quad: np.ndarray, samples: int = 40) -> list:
    """For each side of the quad (top, right, bottom, left): mean brightness just inside minus just outside it."""
    h, w = gray.shape
    k = max(3.0, 0.012 * w)
    out = []
    for i in range(4):
        a, b = np.float32(quad[i]), np.float32(quad[(i + 1) % 4])
        d = b - a
        inward = np.float32([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)  # the quad is ordered clockwise on screen
        t = np.linspace(0.1, 0.9, samples, dtype=np.float32)[:, None]
        pts = a + d * t
        def sample(offset):
            p = pts + inward * offset
            x = np.clip(p[:, 0], 0, w - 1).astype(int)
            y = np.clip(p[:, 1], 0, h - 1).astype(int)
            return gray[y, x].astype(np.float32)
        out.append(float(np.mean(sample(k) - sample(-k))))
    return out


def _fill(gray: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of the quad's interior that is as bright as the page itself (1.0 = uniformly paper; a strip of dark desk
    inside the quad lowers it). Braille shadows are tiny, so a true page stays above about 0.9."""
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.int32(quad), 255)
    values = gray[cv2.erode(inside, np.ones((5, 5), np.uint8)) > 0]
    if values.size == 0:
        return 0.0
    page_level = float(np.percentile(values, 60))  # a typical paper pixel, whatever the exposure
    return float((values > 0.6 * page_level).mean())


def _quality(gray: np.ndarray, small: np.ndarray, s: float, q: np.ndarray) -> float:
    """How page-like a quad (full-resolution corners) is: 0 if it fails a check, else a score that rewards a big, bright,
    uniformly paper-filled area whose four sides each sit on a clean straight edge."""
    qs = q * s
    contrast = _contrast(small, qs)
    _, support = _refine(gray, q, 0.015)
    if contrast < MIN_CONTRAST or min(_side_contrasts(small, qs)) < MIN_SIDE_CONTRAST or min(support) < MIN_EDGE_SUPPORT:
        return 0.0
    return cv2.contourArea(qs) * contrast * min(support) * _fill(small, qs) ** 10


def find_page(frame: np.ndarray, w_mm: float, h_mm: float) -> tuple:
    """(corners, reason): the page's four corners in image pixels (TL, TR, BR, BL), or (None, why not).

    Two strategies run: one finds the page as a bright region and approximates its outline; the other fits the four edge
    lines directly (which survives something hiding part of an edge or a corner). The better quad, by one shared quality
    score, wins; if neither is clean the region strategy's explanation is returned."""
    q_region, why = _find_page_by_regions(frame, w_mm, h_mm)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    s = WORK_W / gray.shape[1]
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    q_lines = _find_page_by_lines(gray, small, s, w_mm / h_mm)
    scored = [(_quality(gray, small, s, q), q) for q in (q_region, q_lines) if q is not None]
    scored = [(score, q) for score, q in scored if score > 0]
    return (max(scored, key=lambda sq: sq[0])[1], "") if scored else (None, why)


def find_page_homography(frame: np.ndarray, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0)) -> tuple:
    """(H, reason): image px -> page mm, or (None, why not). The page's top-left corner sits at `origin` (default 0, 0)."""
    q, why = find_page(frame, w_mm, h_mm)
    if q is None:
        return None, why
    H = homography_from_corners(q, w_mm, h_mm, origin)
    return (H, "") if H is not None else (None, "the edges found are too squashed to place the page: face the sheet more squarely")


class AutoPage:
    """Page source for `--auto-page` / `--paper`: finds the page edges, then either tracks the page or keeps re-finding it.

    track=True   lock on once the edges hold steady, then follow the page's texture (good for a page full of braille)
    track=False  keep re-finding the edges about 3 times a second and hold the last good position while they are hidden
                 (good for a sparse printed sheet, where texture tracking has too little to lock onto)
    origin       where the page's top-left corner sits in page mm (the printed sheets use -SHEET_ORIGIN_MM)
    """

    def __init__(self, w_mm: float, h_mm: float, origin: tuple = (0.0, 0.0), track: bool = True,
                 refresh_seconds: float = 0.3, hold_seconds: float = 5.0):
        self.size_mm, self.origin, self.track = (w_mm, h_mm), origin, track
        self.refresh, self.hold = refresh_seconds, hold_seconds
        self.tracker, self.prev, self.steady, self.reason = None, None, 0, ""
        self.last_H: Optional[np.ndarray] = None
        self.last_seen, self.last_try = 0.0, 0.0

    def unlock(self) -> None:
        """Forget the current page and look for the edges again (bound to the r key)."""
        self.tracker, self.prev, self.steady, self.last_H, self.last_seen, self.last_try = None, None, 0, None, 0.0, 0.0

    @property
    def status(self) -> str:
        if self.tracker is not None:
            return self.tracker.status
        if not self.track:
            if self.last_H is None:
                return f"looking for the page edges: {self.reason}"
            if self.reason:
                return f"page edges hidden, holding the last position ({self.reason})"
            return "page edges found"
        if self.steady:
            return f"page edges found, holding steady ({self.steady}/{STEADY_FRAMES})"
        return f"looking for the page edges: {self.reason}"

    def homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Image px -> page mm for this frame, or None while no page has been found."""
        if not self.track:
            return self._edges_only(frame)
        if self.tracker is not None:
            return self.tracker.homography(frame)
        q, self.reason = find_page(frame, *self.size_mm)
        if q is None:
            self.prev, self.steady = None, 0
            return None
        H = homography_from_corners(q, *self.size_mm, self.origin)
        if H is None:  # edges in a line, or nearly: no better than not having found them
            self.prev, self.steady, self.reason = None, 0, "the edges found are too squashed to place the page"
            return None
        moved = self.prev is None or np.abs(q - self.prev).max() > 4
        self.steady, self.prev = 1 if moved else self.steady + 1, q
        if self.steady >= STEADY_FRAMES:
            self.tracker = PageTracker(frame, H)  # from now on follow the page's texture, not its edges
        return H

    def _edges_only(self, frame: np.ndarray) -> Optional[np.ndarray]:
        now = time.time()
        if self.last_H is not None and now - self.last_try < self.refresh:
            return self.last_H
        self.last_try = now
        q, self.reason = find_page(frame, *self.size_mm)
        H = homography_from_corners(q, *self.size_mm, self.origin) if q is not None else None
        if H is not None:
            self.last_H, self.last_seen = H, now
            return self.last_H
        if q is not None:
            self.reason = "the edges found are too squashed to place the page"
        if self.last_H is not None and now - self.last_seen < self.hold:
            return self.last_H  # hidden for a moment (a hand over an edge): keep the last good position
        self.last_H = None
        return None
