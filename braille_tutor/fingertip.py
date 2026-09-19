"""Fingertip tracking: where is the learner's pointing finger on the page? opencv + numpy only, no model, no training.

How it works, in three steps:
  1. skin mask     skin colour (YCrCb) that is not the white/grey of the paper (saturation), cleaned up and cut to the largest blob
  2. tip           the hand comes in from the edge of the picture, so the fingertip is the point of the hand's outline that is
                   farthest from where the arm enters (ignoring points on the border itself, where the arm is cut off); the
                   palm (the fattest part of the blob) only tells how far the finger sticks out, which is the confidence
  3. tracker       smoothed in PAGE millimetres (so a shaking camera does not shake the finger), needs two frames in a row before
                   it reports, and holds the last position through a short dropout

Limits (be honest about them): it is tuned on synthetic hands, not yet on real ones. Very dark or very light skin under unusual
lighting may need `skin_ranges` adjusted; a skin-coloured desk defeats it; gloves do not work. The tip found is the end of the
finger, the touching pad is a few millimetres behind it, which is small next to the 19 mm cell spacing of our sheets. Several
fingers out: the farthest one wins, which is usually the pointing one. Clicking the video still overrides it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from page import to_page

WORK_WIDTH = 320  # the mask is made at this width: fast enough for every frame, sharp enough for a fingertip
BORDER_PX = 3  # outline points this close to the picture edge are where the arm is cut off, not a fingertip
MIN_HAND_FRACTION = 0.004  # a skin blob smaller than this share of the picture is not a hand
MAX_HAND_FRACTION = 0.45  # a bigger one is more likely a skin-coloured desk
SKIN_CR = (135, 180)
SKIN_CB = (85, 130)
MIN_SATURATION = 40  # paper, however warm the light, is far less saturated than skin
MIN_VALUE = 50


@dataclass(frozen=True)
class Tip:
    """One fingertip sighting. x, y are image pixels of the ORIGINAL frame; confidence is 0..1."""
    x: float
    y: float
    confidence: float
    palm: tuple  # (x, y) of the palm centre, image pixels
    reach: float  # tip distance divided by palm radius: about 2 or more for a pointing finger, about 1 for a fist


def skin_mask(bgr: np.ndarray, skin_cr: tuple = SKIN_CR, skin_cb: tuple = SKIN_CB) -> np.ndarray:
    """A 0/255 mask of skin-coloured pixels that are not paper-coloured. Expects a small (already blurred) image."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(ycc, (0, skin_cr[0], skin_cb[0]), (255, skin_cr[1], skin_cb[1]))
    mask &= cv2.inRange(hsv, (0, MIN_SATURATION, MIN_VALUE), (180, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)  # speckle
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)  # holes (knuckle creases, glare)


def _largest_hand(mask: np.ndarray) -> Optional[np.ndarray]:
    """The largest blob as its own filled mask, or None if there isn't one of a plausible size."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    share = stats[best, cv2.CC_STAT_AREA] / mask.size
    if not MIN_HAND_FRACTION <= share <= MAX_HAND_FRACTION:
        return None
    hand = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(hand)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)  # no holes, so the palm centre is not pulled off by a crease
    return filled


def find_fingertip(frame: np.ndarray, skin_ranges: Optional[tuple] = None) -> Optional[Tip]:
    """The fingertip in one BGR frame, or None if no hand is in view. Pixels are those of `frame`."""
    h0, w0 = frame.shape[:2]
    scale = min(1.0, WORK_WIDTH / w0)
    small = cv2.resize(frame, (max(1, int(w0 * scale)), max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    small = cv2.GaussianBlur(small, (5, 5), 0)
    mask = skin_mask(small, *(skin_ranges or (SKIN_CR, SKIN_CB)))
    hand = _largest_hand(mask)
    if hand is None:
        return None
    h, w = hand.shape
    dist = cv2.distanceTransform(hand, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)  # the fattest part of the hand: the palm
    radius = float(dist[cy, cx])
    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    edge = np.zeros_like(hand, dtype=bool)
    edge[:BORDER_PX + 1, :] = edge[-BORDER_PX - 1:, :] = edge[:, :BORDER_PX + 1] = edge[:, -BORDER_PX - 1:] = True
    entry_px = np.argwhere((hand > 0) & edge)  # where the arm leaves the picture
    touches_border = len(entry_px) > 0  # a real hand is attached to an arm that leaves the picture
    entry = entry_px[:, ::-1].mean(axis=0) if touches_border else np.array([cx, cy], np.float64)
    inside = (pts[:, 0] > BORDER_PX) & (pts[:, 0] < w - 1 - BORDER_PX) & (pts[:, 1] > BORDER_PX) & (pts[:, 1] < h - 1 - BORDER_PX)
    if not inside.any() or radius < 2:
        return None
    pts = pts[inside]
    d = np.hypot(pts[:, 0] - entry[0], pts[:, 1] - entry[1])  # the finger points away from where the arm comes in
    tip = pts[d >= 0.97 * d.max()].mean(axis=0)  # average the very end of the finger: one noisy outline pixel cannot move it
    reach = float(np.hypot(tip[0] - cx, tip[1] - cy)) / radius
    confidence = float(np.clip((reach - 1.2) / 1.3, 0.0, 1.0)) * (1.0 if touches_border else 0.4)
    return Tip(float(tip[0] / scale), float(tip[1] / scale), confidence, (float(cx / scale), float(cy / scale)), reach)


class FingerTracker:
    """Turns per-frame sightings into a steady page position. Feed it every frame with update(); read .position.

    * needs `confirm` sightings in a row before reporting (a flash of skin colour does not move the finger)
    * smooths in page mm with an exponential average (`smooth` = weight of the newest sighting), unless the finger jumped
      farther than `jump_mm`, which is taken as a new place rather than smoothed toward
    * holds the last position for `hold_seconds` after the hand leaves, then reports None"""

    def __init__(self, min_confidence: float = 0.3, confirm: int = 2, smooth: float = 0.5, jump_mm: float = 40.0,
                 hold_seconds: float = 0.6, clock: Callable = time.monotonic, find: Callable = find_fingertip):
        self.min_confidence, self.confirm, self.smooth, self.jump_mm = min_confidence, confirm, smooth, jump_mm
        self.hold_seconds, self.clock, self.find = hold_seconds, clock, find
        self.reset()

    def reset(self) -> None:
        self.position: Optional[tuple] = None  # page mm, or image px when no page is registered
        self.tip: Optional[Tip] = None  # the newest raw sighting
        self._streak, self._seen = 0, 0.0

    def update(self, frame: np.ndarray, H: Optional[np.ndarray]) -> Optional[tuple]:
        """Process one frame. Returns the fingertip in page mm (needs H), or None."""
        tip = self.find(frame)
        now = self.clock()
        if tip is None or tip.confidence < self.min_confidence:
            self.tip, self._streak = None, 0
            if self.position is not None and now - self._seen > self.hold_seconds:
                self.position = None
            return self.position if H is not None else None
        self.tip, self._streak, self._seen = tip, self._streak + 1, now
        if H is None:  # without a page there is no mm: keep nothing that would mean the wrong thing
            self.position = None
            return None
        raw = to_page(H, tip.x, tip.y)
        if self._streak < self.confirm and self.position is None:
            return None
        if self.position is None or np.hypot(raw[0] - self.position[0], raw[1] - self.position[1]) > self.jump_mm:
            self.position = raw
        else:
            a = self.smooth
            self.position = (a * raw[0] + (1 - a) * self.position[0], a * raw[1] + (1 - a) * self.position[1])
        return self.position


def main() -> None:
    """Try the tracker on a real hand: python fingertip.py [--camera 0|path]. Shows the video, the skin mask and the tip."""
    import argparse

    from detect import GREEN, RED, draw_hud, open_camera, page_source_from_args, page_status

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--calib", help="calibration.json: also show the fingertip in page millimetres")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"))
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--markers-only", action="store_true")
    a = ap.parse_args()
    cap = open_camera(a.camera)
    src = page_source_from_args(a) if (a.calib or a.auto_page or a.paper or a.markers_only) else None
    tracker = FingerTracker()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H = page_status(frame, src)[0] if src is not None else None
        pos = tracker.update(frame, H)
        view, tip = frame.copy(), tracker.tip
        small = cv2.GaussianBlur(cv2.resize(frame, (WORK_WIDTH, int(frame.shape[0] * WORK_WIDTH / frame.shape[1]))), (5, 5), 0)
        cv2.imshow("skin mask", skin_mask(small))
        if tip is not None:
            colour = GREEN if tip.confidence >= tracker.min_confidence else RED
            cv2.circle(view, (int(tip.x), int(tip.y)), 12, colour, 3)
            cv2.circle(view, (int(tip.palm[0]), int(tip.palm[1])), 6, (255, 128, 0), -1)
        lines = [(f"tip confidence {tip.confidence:.2f}  reach {tip.reach:.1f}" if tip else "no hand found", GREEN if tip else RED)]
        if pos is not None:
            lines.append((f"page position {pos[0]:.0f}, {pos[1]:.0f} mm", GREEN))
        elif src is not None:
            lines.append(("no page registered: no millimetres", RED))
        draw_hud(view, lines + [("q: quit", (255, 255, 0))])
        cv2.imshow("fingertip", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
