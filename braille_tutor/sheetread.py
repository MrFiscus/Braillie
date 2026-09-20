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
LOCAL_SEARCH_MM = 4.5  # per-cell fine tuning after align(): how much each cell may drift from the global shift on its own
LOCAL_PULL = 0.25  # tie-break: a per-cell shift only wins if it clearly beats the global position (see refine_per_cell)
LOCAL_MIN_DOTS = 3  # a cell with fewer expected dots is too weak a signature to refine independently -- see refine_per_cell


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


def refine_per_cell(resp: np.ndarray, cells: list, px_per_mm: float = PX_PER_MM, search_mm: float = LOCAL_SEARCH_MM) -> list:
    """Nudge each cell by up to ~half a pitch so it sits on its own dots, on top of align()'s single global shift.

    align() finds the one shift that lines the whole page up best; on a curved sheet, one whose corner markers were
    stuck a millimetre or two off ideal, or one registered from only 3 of the 4 markers (the fourth briefly covered),
    that one shift is wrong in different amounts in different parts of the page. This picks up the residual per cell.

    For each cell with at least LOCAL_MIN_DOTS raised dots we score its expected raised slots minus its expected empty
    slots across every candidate shift within `search_mm`, biased toward not moving (LOCAL_PULL) so a cell only drifts
    if the picture clearly prefers a shifted position. The search stays under one pitch, so a cell cannot latch onto
    its neighbour's dots. A weaker cell ('a' has one dot, 'b' has two) can't do this alone -- it might slide off its
    own dot -- so it inherits the shift of its nearest refined neighbour, matching how much the image is corrected in
    that region. That keeps the sampling threshold consistent across the page: if only strong cells were nudged, their
    on-dot samples would jump up and pull the threshold with them, and every weak cell's dots would read as blank."""
    if not cells:
        return cells
    cap = float(np.percentile(resp, 99.5)) + 1e-6
    r = cv2.GaussianBlur(np.minimum(resp, cap), (0, 0), 0.5 * DOT_R_MM * px_per_mm)  # same smoothing as align()
    h, w = r.shape
    n = int(round(search_mm * px_per_mm))  # pixels of search each side; the search step is one pixel = 1 / px_per_mm mm
    k = 2 * n + 1
    offs = np.arange(-n, n + 1) / px_per_mm  # (k,), mm
    dist = np.hypot(offs[None, :], offs[:, None])  # (k, k), mm from no-shift for the tie-break

    def patch(cx_px: int, cy_px: int) -> np.ndarray:
        """A k*k patch of `r` centred on (cx_px, cy_px); zero-padded where it would fall outside the image."""
        p = np.zeros((k, k), np.float64)
        y0, y1 = cy_px - n, cy_px + n + 1
        x0, x1 = cx_px - n, cx_px + n + 1
        iy0, iy1, ix0, ix1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
        if iy1 > iy0 and ix1 > ix0:
            p[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = r[iy0:iy1, ix0:ix1]
        return p

    shifts: dict = {}  # (row, col) -> (dx_mm, dy_mm), only for cells refined on their own dots
    for c in cells:
        if len(c["dots"]) < LOCAL_MIN_DOTS:
            continue
        raised = [SLOT_OFFSETS[i] for i in c["dots"]]
        empty = [SLOT_OFFSETS[i] for i in range(1, 7) if i not in c["dots"]]
        cx_px = int(round((c["x"] + MARGIN_MM) * px_per_mm))
        cy_px = int(round((c["y"] + MARGIN_MM) * px_per_mm))
        on = sum(patch(cx_px + int(round(dx * px_per_mm)), cy_px + int(round(dy * px_per_mm))) for dx, dy in raised) / len(raised)
        off = sum(patch(cx_px + int(round(dx * px_per_mm)), cy_px + int(round(dy * px_per_mm))) for dx, dy in empty) / len(empty) if empty else 0.0
        score = on - off - LOCAL_PULL * on * dist / search_mm
        iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
        shifts[(c["row"], c["col"])] = (float(offs[ix]), float(offs[iy]))

    if not shifts:  # nothing was strong enough to refine on; keep the global-only positions
        return cells
    keys = list(shifts)
    key_pos = np.array([(cells[i]["x"], cells[i]["y"]) for i, c in enumerate(cells) if (c["row"], c["col"]) in shifts], np.float64)
    out = []
    for c in cells:
        rc = (c["row"], c["col"])
        if rc in shifts:
            dx, dy = shifts[rc]
        else:  # weak cell: take the shift of the nearest refined neighbour, in page mm
            nearest = int(np.argmin(np.hypot(key_pos[:, 0] - c["x"], key_pos[:, 1] - c["y"])))
            dx, dy = shifts[keys[nearest]]
        out.append({**c, "x": c["x"] + dx, "y": c["y"] + dy})
    return out


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
    With `self_align` the registration is first corrected by up to SEARCH_MM by a single global shift (see align()), then
    each cell is nudged by up to LOCAL_SEARCH_MM on its own (see refine_per_cell) to catch a warped sheet or corner markers
    stuck a little out; the returned cells then sit where the dots really are, and carry the global correction as `shift`."""
    return observe_response(respond(rectify(frame, H, px_per_mm), px_per_mm), cells, px_per_mm, radius_mm, floor_sigmas, self_align)


def observe_response(resp: np.ndarray, cells: list, px_per_mm: float = PX_PER_MM, radius_mm: float = 1.0,
                     floor_sigmas: float = 2.0, self_align: bool = True) -> list:
    """observe(), starting from an already computed response map (so settings can be compared without redoing the image work)."""
    shift = align(resp, cells, px_per_mm) if self_align else (0.0, 0.0)
    if shift != (0.0, 0.0):
        cells = [{**c, "x": c["x"] + shift[0], "y": c["y"] + shift[1]} for c in cells]
    radius = int(round(radius_mm * px_per_mm))  # registration is good to ~0.3 mm, so a small patch: less noise to take the max of
    sample = lambda cs: {(i, n): _sample(resp, c["x"] + dx, c["y"] + dy, radius, px_per_mm)
                         for i, c in enumerate(cs) for n, (dx, dy) in SLOT_OFFSETS.items()}
    # what an empty patch of the page scores: between the lines, where nothing is ever placed. Computed from the
    # globally-shifted positions (not per-cell refined ones) so per-cell refinement can only lift a cell's on-dot
    # samples above the threshold, never push a weak cell's dots below it by raising the bar for everyone.
    ys = sorted({round(c["y"], 1) for c in cells})
    pitch_y = float(np.median(np.diff(ys))) if len(ys) > 1 else 40.0
    quiet = [_sample(resp, c["x"], c["y"] + pitch_y / 2 + 0.5 * DOT_MM, radius, px_per_mm) for c in cells]
    floor = float(np.mean(quiet) + floor_sigmas * (np.std(quiet) + 1e-6))
    threshold = _split(np.array(list(sample(cells).values())), floor)
    if self_align:
        cells = refine_per_cell(resp, cells, px_per_mm)  # residual per cell after the global shift
    ev = sample(cells)
    values = np.array(list(ev.values()))
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
