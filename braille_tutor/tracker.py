"""Marker-free page tracking: match each camera frame to a reference photo so the calibration follows the camera."""
from __future__ import annotations

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
        self.last = self.H_ref @ np.linalg.inv(self.ref_S) @ M @ S  # frame px -> frame small -> ref small -> ref px -> mm
        return self.last


def load_calibration(path: str) -> Union[PageTracker, FixedPage]:
    """calibration.json -> a tracker if calibration.png (the reference photo) sits beside it, else a fixed page."""
    H, ref = load_homography(path), Path(path).with_suffix(".png")
    img = cv2.imread(str(ref)) if ref.exists() else None
    return PageTracker(img, H) if img is not None else FixedPage(H)
