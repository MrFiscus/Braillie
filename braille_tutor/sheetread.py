"""Read a KNOWN printed sheet by looking for raised dots at the exact places the sheet puts them.

The general detector has to find cells anywhere and classify them, and it has systematic weaknesses (it drops right-column dots,
misplaces cells with no top row). On our own sheets none of that is necessary: page registration puts every dot position within a
fraction of a millimetre, so the question for each of a cell's six slots is only "is a dot raised here?". That is measured from
the image, not copied from the layout: a missed poke, or the wrong sheet on the desk, shows up as a mismatch.

    rectify   flatten the registered page into a top-down picture in millimetres
    respond   a map of how strongly each pixel looks like a dot (a highlight or a shadow, so embossed and printed dots alike)
    align     find how far the registration is off (a printer that shifts the print, a page cut a bit crooked, paper edges found
              a millimetre out): a search for the one shift at which the layout's dots line up best with the picture
    observe   sample the six slots of every cell, split "raised" from "flat" with a threshold learned from the page itself
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from detect import Cell, dots_to_char, dots_to_label
from page import PAGE_H_MM, PAGE_W_MM
from sheets import DOT_MM, DOT_R_MM

PX_PER_MM = 5.0  # a 3.2 mm dot is 16 px wide; 10 px/mm read no better and took 3 s a scan (measured: same accuracy, 13x faster)
MARGIN_MM = 6.0
SLOT_OFFSETS = {n: (((n - 1) // 3 - 0.5) * DOT_MM, (((n - 1) % 3) - 1) * DOT_MM) for n in range(1, 7)}  # dot n from the cell centre


def rectify(frame: np.ndarray, H: np.ndarray, px_per_mm: float = PX_PER_MM) -> np.ndarray:
    """The registered page as a flat top-down grayscale picture, page (0, 0) at MARGIN_MM in from the top-left."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    size = (int((PAGE_W_MM + 2 * MARGIN_MM) * px_per_mm), int((PAGE_H_MM + 2 * MARGIN_MM) * px_per_mm))
    S = np.array([[px_per_mm, 0, MARGIN_MM * px_per_mm], [0, px_per_mm, MARGIN_MM * px_per_mm], [0, 0, 1]], np.float64)
    return cv2.warpPerspective(gray, S @ H, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def respond(flat: np.ndarray, px_per_mm: float = PX_PER_MM) -> np.ndarray:
    """How dot-like each pixel is: bright or dark features about the size of a dot, whatever the paper's own brightness."""
    dia = 2 * DOT_R_MM * px_per_mm
    k = int(round(1.8 * dia)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = flat.astype(np.float32)
    both = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel) + cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    return cv2.GaussianBlur(both, (0, 0), dia / 4)


SEARCH_MM = 4.5  # how far off the registration may be and still be corrected (the dots are 6 mm apart, so more is ambiguous)
SEARCH_STEP_MM = 0.2
PULL = 0.15  # tie-break: a shift of SEARCH_MM costs this share of the score of a perfectly lined-up sheet


def _slots(cells: list) -> tuple:
    """Every slot of every cell, page mm: (raised as an (n, 2) array, flat as an (m, 2) array), per the layout."""
    on, off = [], []
    for c in cells:
        for n, (dx, dy) in SLOT_OFFSETS.items():
            (on if n in c["dots"] else off).append((c["x"] + dx, c["y"] + dy))
    return np.array(on, np.float64).reshape(-1, 2), np.array(off, np.float64).reshape(-1, 2)


def align(resp: np.ndarray, cells: list, px_per_mm: float = PX_PER_MM, search_mm: float = SEARCH_MM,
          step_mm: float = SEARCH_STEP_MM) -> tuple:
    """(dx, dy) in mm: where the page's dots really are relative to where the registration says, found by trying every shift
    within `search_mm` and keeping the one at which the picture responds most on the slots the layout raises and least on the
    ones it leaves flat. A single shift for the whole page, so it cannot bend the reading toward what is expected: a missed poke
    or the wrong sheet still reads wrong. Returns (0, 0) if the layout has nothing to line up."""
    on, off = _slots(cells)
    if len(on) < 4:
        return 0.0, 0.0
    cap = float(np.percentile(resp, 99.5)) + 1e-6
    r = cv2.GaussianBlur(np.minimum(resp, cap), (0, 0), 0.5 * DOT_R_MM * px_per_mm)  # smooth: the score should slope toward the peak
    to_px = lambda p: np.rint((p + MARGIN_MM) * px_per_mm).astype(int)
    pon, poff = to_px(on), to_px(off)
    h, w = r.shape
    steps = np.arange(-round(search_mm / step_mm), round(search_mm / step_mm) + 1) * step_mm
    best, best_score = (0.0, 0.0), -np.inf
    for oy in steps:
        for ox in steps:
            sx, sy = int(round(ox * px_per_mm)), int(round(oy * px_per_mm))
            xs, ys = np.clip(pon[:, 0] + sx, 0, w - 1), np.clip(pon[:, 1] + sy, 0, h - 1)
            raised = float(r[ys, xs].mean())
            score = raised - PULL * raised * np.hypot(ox, oy) / search_mm
            if len(poff):
                score -= float(r[np.clip(poff[:, 1] + sy, 0, h - 1), np.clip(poff[:, 0] + sx, 0, w - 1)].mean())
            if score > best_score:
                best, best_score = (float(ox), float(oy)), score
    return best


def _sample(resp: np.ndarray, x_mm: float, y_mm: float, radius_px: int, px_per_mm: float) -> float:
    cx, cy = int(round((x_mm + MARGIN_MM) * px_per_mm)), int(round((y_mm + MARGIN_MM) * px_per_mm))
    patch = resp[max(0, cy - radius_px):cy + radius_px + 1, max(0, cx - radius_px):cx + radius_px + 1]
    return float(patch.max()) if patch.size else 0.0


def _split(values: np.ndarray, floor: float) -> float:
    """Threshold between 'flat' and 'raised' slots: the best two-class split (Otsu), never below `floor`."""
    hist, edges = np.histogram(values, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = hist.astype(np.float64)
    best, best_t = -1.0, floor
    for i in range(1, 63):
        w0, w1 = weight[:i].sum(), weight[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (weight[:i] * centres[:i]).sum() / w0, (weight[i:] * centres[i:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, (centres[i - 1] + centres[i]) / 2
    return max(best_t, floor)


def observe(frame: np.ndarray, H: np.ndarray, cells: list, px_per_mm: float = PX_PER_MM, radius_mm: float = 1.0,
            floor_sigmas: float = 2.0, self_align: bool = True) -> list:
    """For each cell of a known sheet, the dots that are actually raised in the image. Returns Cells (same layout, observed
    dots); each cell's `confidence` is how far its weakest decision was from the threshold, from 0 (a coin flip) to 1.
    With `self_align` the registration is first corrected by up to SEARCH_MM (see align()); the returned cells then sit where
    the dots really are, and carry that correction as `shift` (mm)."""
    return observe_response(respond(rectify(frame, H, px_per_mm), px_per_mm), cells, px_per_mm, radius_mm, floor_sigmas, self_align)


def observe_response(resp: np.ndarray, cells: list, px_per_mm: float = PX_PER_MM, radius_mm: float = 1.0,
                     floor_sigmas: float = 2.0, self_align: bool = True) -> list:
    """observe(), starting from an already computed response map (so settings can be compared without redoing the image work)."""
    shift = align(resp, cells, px_per_mm) if self_align else (0.0, 0.0)
    if shift != (0.0, 0.0):
        cells = [{**c, "x": c["x"] + shift[0], "y": c["y"] + shift[1]} for c in cells]
    radius = int(round(radius_mm * px_per_mm))  # registration is good to ~0.3 mm, so a small patch: less noise to take the max of
    ev = {(i, n): _sample(resp, c["x"] + dx, c["y"] + dy, radius, px_per_mm)
          for i, c in enumerate(cells) for n, (dx, dy) in SLOT_OFFSETS.items()}
    # what an empty patch of the page scores: between the lines, where nothing is ever placed
    ys = sorted({round(c["y"], 1) for c in cells})
    pitch_y = float(np.median(np.diff(ys))) if len(ys) > 1 else 40.0
    quiet = [_sample(resp, c["x"], c["y"] + pitch_y / 2 + 0.5 * DOT_MM, radius, px_per_mm) for c in cells]
    floor = float(np.mean(quiet) + floor_sigmas * (np.std(quiet) + 1e-6))
    values = np.array(list(ev.values()))
    threshold = _split(values, floor)
    scale = max(float(np.percentile(values, 95)) - threshold, 1e-6)
    out = []
    for i, c in enumerate(cells):
        dots = frozenset(n for n in range(1, 7) if ev[(i, n)] >= threshold)
        margin = min(abs(ev[(i, n)] - threshold) for n in range(1, 7)) / scale
        out.append({**c, "dots": dots, "label": dots_to_label(dots), "char": dots_to_char(dots), "confidence": float(min(1.0, margin)), "shift": shift})
    return out


def compare(observed: list, expected: list) -> tuple:
    """(matching, total, mismatches): how many observed cells have exactly the dots the sheet says, and the ones that don't as
    [(row, col, expected dots, observed dots)]. A missed poke or the wrong sheet on the desk shows up here."""
    bad = [(o["row"], o["col"], e["dots"], o["dots"]) for o, e in zip(observed, expected) if o["dots"] != e["dots"]]
    return len(expected) - len(bad), len(expected), bad


IDENTIFY_MIN_SCORE = 0.75  # a sheet is recognised when at least this share of its cells read exactly as it says...
IDENTIFY_MARGIN = 0.15  # ...and it beats the runner-up by this much (different sheets put their cells in different places)


def identify_sheet(frame: np.ndarray, H: np.ndarray, candidates: dict) -> tuple:
    """Which known sheet is on the desk? `candidates` maps a name to that sheet's cells. Returns (name or None, scores), scores
    being the share of each sheet's cells whose dots read exactly as that sheet says. None when nothing is clearly ahead: a blank
    page, a hand in the way, or a page that is none of them."""
    resp = respond(rectify(frame, H))  # the expensive part, once for all of them
    scores = {}
    for name, cells in candidates.items():
        good, total, _ = compare(observe_response(resp, cells), cells)
        scores[name] = good / total if total else 0.0
    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    if scores[best] >= IDENTIFY_MIN_SCORE and scores[best] - runner_up >= IDENTIFY_MARGIN:
        return best, scores
    return None, scores


def display_boxes(observed: list, H: np.ndarray) -> list:
    """Observed cells as pixel boxes (x1, y1, x2, y2, label, confidence), the same form the detector returns, so the live view
    can draw and vote on them exactly like detections. Cells with no raised dot are skipped."""
    from page import to_image

    out = []
    for c in observed:
        if not c["dots"]:
            continue
        pts = [to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2) for sx in (-1, 1) for sy in (-1, 1)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys), c["label"], max(0.05, c["confidence"])))
    return out
