"""Marker-free page tracking: match each camera frame to a reference photo so the calibration follows the camera."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from page import load_homography

WORK_W = 960  # features are found on a copy this wide, to keep it fast on a CPU


class FixedPage:
    """A calibration that never changes: the camera and page must not move."""
    status = "fixed calibration"

    def __init__(self, H: np.ndarray):
        self.H = H

    def homography(self, frame: np.ndarray) -> np.ndarray:
        """Image px -> page mm (always the same matrix)."""
        return self.H


class PageTracker:
    """Keeps the image px -> page mm homography up to date as the camera moves, using SIFT matches to a reference."""

    def __init__(self, ref_frame: np.ndarray, H_ref: np.ndarray, min_inliers: int = 25):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.ref_kp, self.ref_des, self.ref_S = self._features(ref_frame)
        self.H_ref, self.min_inliers, self.inliers, self.reason = H_ref, min_inliers, 0, ""
        self.last: Optional[np.ndarray] = H_ref  # the camera starts where the reference was taken

    @property
    def status(self) -> str:
        return f"tracking page ({self.inliers} matches)" if not self.reason else f"tracking LOST: {self.reason} (holding last position)"

    def _features(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = WORK_W / gray.shape[1]
        kp, des = self.sift.detectAndCompute(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
        return kp, des, np.diag([s, s, 1.0])

    def homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Image px -> page mm for this frame; keeps the last good answer if matching fails (hand in the way, blur)."""
        kp, des, S = self._features(frame)
        self.inliers, self.reason = 0, ""
        if des is None or len(kp) < 2 or self.ref_des is None:
            self.reason = "too little texture in view to match the reference photo"
            return self.last
        pairs = cv2.BFMatcher().knnMatch(des, self.ref_des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]  # Lowe ratio test
        if len(good) < self.min_inliers:
            self.reason = f"only {len(good)} matches with the reference photo (need {self.min_inliers}); view blurry, blocked or moved too far"
            return self.last
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        self.inliers = int(mask.sum()) if mask is not None else 0
        if M is None or self.inliers < self.min_inliers:
            self.reason = f"matches to the reference photo are inconsistent ({self.inliers} agree, need {self.min_inliers})"
            return self.last
        candidate = self.H_ref @ np.linalg.inv(self.ref_S) @ M @ S  # frame px -> frame small -> ref small -> ref px -> mm
        if not np.isfinite(candidate).all():
            # a near-degenerate point spread (matches bunched along one edge, say) can fit a singular matrix that
            # findHomography's own checks don't catch: using it would divide by ~0 in to_page/to_image later and
            # crash whatever called them, so it is rejected here at the source, same as any other bad fit.
            self.reason = "matches to the reference photo fit no usable mapping"
            return self.last
        self.last = candidate
        return self.last


def load_calibration(path: str) -> Union[PageTracker, FixedPage]:
    """calibration.json -> a tracker if calibration.png (the reference photo) sits beside it, else a fixed page."""
    H, ref = load_homography(path), Path(path).with_suffix(".png")
    img = cv2.imread(str(ref)) if ref.exists() else None
    return PageTracker(img, H) if img is not None else FixedPage(H)


class RobustPage:
    """Markers when they are visible; keeps working from the sheet's own appearance when they are not.

    Exact marker registration only needs 3 of the 4 corner markers readable (each contributes its own 4 corners,
    not just its centre: see page.homography_from_marker_corners), so a hand covering one of them -- typically
    the one nearest whatever cell is being pointed at -- does not by itself lose exact registration. This still
    keeps the last marker-registered frame as a reference and, whenever fewer than min_markers are readable,
    matches the current frame to it, so a sheet can be moved, tilted or covered more than that without the page
    being lost outright -- just less precisely followed.

    paper_fallback: an optional AutoPage (from --paper) used if even matching fails.
    """

    def __init__(self, hold_seconds: float = 4.0, refresh_seconds: float = 1.5, paper_fallback=None,
                 min_markers: int = 3):
        from page import MARKER_POS_MM

        self.marker_pos, self.hold, self.refresh = MARKER_POS_MM, hold_seconds, refresh_seconds
        self.paper_fallback, self.min_markers = paper_fallback, min_markers
        self.tracker: Optional[PageTracker] = None
        self.last_H: Optional[np.ndarray] = None
        self.last_good, self.last_ref, self.source, self.detail = 0.0, 0.0, "none", ""

    @property
    def status(self) -> str:
        if self.source == "markers":
            return f"page OK ({self.detail})" if self.detail else "page OK (4 markers)"
        if self.source == "tracking":
            return f"page OK - markers hidden, following the sheet ({self.detail})"
        if self.source == "paper":
            return "page OK - markers hidden, using the sheet's edges"
        if self.source == "holding":
            return f"markers hidden and the sheet cannot be followed; holding its last position ({self.detail})"
        return f"PAGE NOT FOUND: {self.detail or 'no markers and nothing to follow yet'}"

    def unlock(self) -> None:
        """Forget everything and re-register from scratch."""
        self.tracker, self.last_H, self.last_good, self.last_ref = None, None, 0.0, 0.0
        self.source, self.detail = "none", ""
        if self.paper_fallback is not None:
            self.paper_fallback.unlock()

    def homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Image px -> page mm for this frame, or None if the page cannot be placed at all."""
        from page import homography_from_marker_corners, visible_marker_corners

        now = time.time()
        seen = visible_marker_corners(frame)
        H = homography_from_marker_corners(seen) if len(seen) >= self.min_markers else None
        if H is not None:
            self.last_H, self.last_good, self.source = H, now, "markers"
            self.detail = "" if len(seen) == 4 else f"{len(seen)} of 4 markers"
            if now - self.last_ref > self.refresh:  # keep the reference fresh so tracking starts from a recent view
                self.tracker, self.last_ref = PageTracker(frame, H), now
            return H
        if self.tracker is not None:
            H = self.tracker.homography(frame)
            if H is not None and not self.tracker.reason:
                self.last_H, self.last_good = H, now
                self.source, self.detail = "tracking", f"{self.tracker.inliers} matches"
                return H
        if self.paper_fallback is not None:
            H = self.paper_fallback.homography(frame)
            if H is not None:
                self.last_H, self.last_good, self.source, self.detail = H, now, "paper", ""
                return H
        if self.last_H is not None and now - self.last_good < self.hold:
            self.source, self.detail = "holding", f"{len(seen)}/4 markers, {now - self.last_good:.0f}s"
            return self.last_H
        from page import diagnose_markers

        self.source = "none"
        self.detail = diagnose_markers(frame)[1] or "no markers in view"  # the detailed reason, for the person aiming
        return None
