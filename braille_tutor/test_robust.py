"""Tests for RobustPage: the page must survive markers leaving the frame, being covered, or motion blur.

This is the failure the sheet actually hits in use — a marker drifts out of view as the sheet is moved, and with plain
marker registration everything stops at once.
"""
import time
import unittest
from unittest import mock

import cv2
import numpy as np

import make_sheet
import page
import sheets
from tracker import PageTracker, RobustPage

SHEET = cv2.cvtColor(make_sheet.render_flat_test("alphabet"), cv2.COLOR_GRAY2BGR)
CELLS = sheets.get_sheet("alphabet").cells
BASE = [[300, 90], [980, 120], [1000, 870], [280, 850]]


def view(quad, blur=0.6, noise=0.0, gain=1.0, cover=None, seed=0):
    """A camera view of the sheet on a dark desk. cover=(centre, axes) paints a hand-like blob over it."""
    h, w = SHEET.shape[:2]
    T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T, (1280, 960))
    img = np.where(mask[..., None] > 0, cv2.warpPerspective(SHEET, T, (1280, 960)),
                   np.full((960, 1280, 3), 70, np.uint8)).astype(np.float32)
    if cover:
        cv2.ellipse(img, cover[0], cover[1], 20, 0, 360, (120, 150, 190), -1)
    img = cv2.GaussianBlur(img * gain, (0, 0), blur)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def dot_error(img, H):
    """90th-percentile distance from each predicted dot to the nearest printed dot: small = the overlay still fits."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dist = cv2.distanceTransform(1 - (gray < 120).astype(np.uint8), cv2.DIST_L2, 3)
    errs = [dist[int(py), int(px)] for c in CELLS for x, y in make_sheet.dot_points(c)
            for px, py in [page.to_image(H, x, y)] if 0 <= px < 1280 and 0 <= py < 960]
    return float(np.percentile(errs, 90)) if errs else 999.0


class RobustPageTests(unittest.TestCase):
    def test_survives_markers_leaving_the_frame(self):
        rp = RobustPage()
        moves = [("all four", BASE), ("one off-frame", [[640, 90], [1330, 120], [1350, 870], [620, 850]]),
                 ("two off-frame", [[860, 80], [1560, 110], [1580, 860], [840, 840]]),
                 ("tilted, moved up", [[520, -40], [1180, 10], [1210, 760], [500, 730]]), ("back", BASE)]
        sources = []
        for label, quad in moves:
            img = view(quad, noise=2.0, seed=1)
            H = rp.homography(img)
            self.assertIsNotNone(H, f"{label}: page lost ({rp.status})")
            self.assertLess(dot_error(img, H), 4.0, f"{label}: overlay drifted off the dots")
            sources.append(rp.source)
        self.assertEqual(sources[0], "markers")
        self.assertIn("tracking", sources[1:4], f"tracking never took over: {sources}")
        self.assertEqual(sources[-1], "markers", "should return to markers once they are visible again")

    def test_survives_a_hand_over_the_markers(self):
        rp = RobustPage()
        self.assertIsNotNone(rp.homography(view(BASE)))
        img = view(BASE, cover=((640, 480), (430, 330)))  # a big hand across the middle and two markers
        H = rp.homography(img)
        self.assertIsNotNone(H, rp.status)
        self.assertIn(rp.source, ("markers", "tracking", "holding"))

    def test_holds_briefly_then_reports_the_page_lost(self):
        rp = RobustPage(hold_seconds=0.4)
        self.assertIsNotNone(rp.homography(view(BASE)))
        blank = np.full((960, 1280, 3), 70, np.uint8)  # sheet gone entirely: nothing to see or match
        self.assertIsNotNone(rp.homography(blank), "should hold the last position briefly")
        self.assertEqual(rp.source, "holding")
        self.assertIn("holding its last position", rp.status)
        time.sleep(0.5)
        self.assertIsNone(rp.homography(blank))
        self.assertIn("PAGE NOT FOUND", rp.status)

    def test_failure_message_still_explains_what_is_wrong(self):
        rp = RobustPage()
        self.assertIsNone(rp.homography(np.full((480, 640, 3), 255, np.uint8)))
        self.assertIn("no markers in view", rp.status)

    def test_unlock_forgets_everything(self):
        rp = RobustPage()
        rp.homography(view(BASE))
        self.assertIsNotNone(rp.tracker)
        rp.unlock()
        self.assertIsNone(rp.tracker)
        self.assertIsNone(rp.last_H)
        self.assertEqual(rp.source, "none")

    def test_tracking_is_as_accurate_as_the_markers_it_replaces(self):
        """Register from markers, then hide them and check the answer barely moves."""
        rp = RobustPage()
        img = view(BASE, noise=2.0, seed=3)
        H_markers = rp.homography(img)
        covered = img.copy()
        for mx, my in page.MARKER_POS_MM.values():  # blank out each marker, leaving the rest of the sheet
            x, y = page.to_image(H_markers, mx, my)
            cv2.circle(covered, (int(x), int(y)), 70, (235, 235, 235), -1)
        H_tracked = rp.homography(covered)
        self.assertEqual(rp.source, "tracking", rp.status)
        worst = max(np.hypot(*(np.array(page.to_image(H_markers, x, y)) - page.to_image(H_tracked, x, y)))
                    for x, y in ((0, 0), (page.PAGE_W_MM, 0), (page.PAGE_W_MM, page.PAGE_H_MM), (0, page.PAGE_H_MM)))
        print(f"\n  markers hidden: corners moved at most {worst:.1f} px")
        self.assertLess(worst, 8.0)


FLAT = [[300, 500], [700, 500], [1000, 500], [280, 500]]  # four "marker centres" in one line: the whole page onto a line


class DegenerateFitTests(unittest.TestCase):
    """A fit that squashes the page onto a line or a point must be turned down where it is made.

    It reads as a perfectly ordinary 3x3 matrix, and every use of it afterwards divides by ~0: positions in the millions,
    or NaN, and then an exception from inside OpenCV or numpy in whichever thread was unlucky enough to ask.
    """

    def test_a_real_registration_is_accepted_and_a_flattened_one_is_not(self):
        self.assertTrue(page.usable_homography(page.page_homography(view(BASE))))
        self.assertFalse(page.usable_homography(None))
        self.assertFalse(page.usable_homography(np.full((3, 3), np.nan)))
        self.assertFalse(page.usable_homography(np.zeros((3, 3))))
        self.assertIsNone(page._homography_from_centers(dict(enumerate(FLAT))))
        self.assertIsNone(page.homography_from_corners(FLAT, page.PAGE_W_MM, page.PAGE_H_MM))

    def test_markers_that_place_the_page_nowhere_leave_it_not_found(self):
        rp = RobustPage()
        with mock.patch("page.visible_markers", return_value=dict(enumerate(np.float32(FLAT)))):
            self.assertIsNone(rp.homography(view(BASE)), "a flattened fit must not be handed out as a position")
        self.assertEqual(rp.source, "none")

    def test_markers_that_place_the_page_nowhere_fall_back_to_following_the_sheet(self):
        rp = RobustPage()
        img = view(BASE)
        self.assertIsNotNone(rp.homography(img))  # registered properly first, so there is a reference to follow
        with mock.patch("page.visible_markers", return_value=dict(enumerate(np.float32(FLAT)))):
            H = rp.homography(img)
        self.assertIsNotNone(H, rp.status)
        self.assertEqual(rp.source, "tracking")
        self.assertLess(dot_error(img, H), 4.0)

    def test_a_flattened_tracking_fit_keeps_the_last_position_instead(self):
        img = view(BASE)
        H_ref = page.page_homography(img)
        tracker = PageTracker(img, H_ref)
        good = tracker.homography(img)
        self.assertTrue(page.usable_homography(good))

        def flat_fit(src, dst, method, thresh):  # RANSAC agreeing on a mapping that collapses the page
            return np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]), np.ones((len(src), 1), np.uint8)

        with mock.patch("cv2.findHomography", side_effect=flat_fit):
            held = tracker.homography(img)
        np.testing.assert_allclose(held, good, err_msg="should hold the last good position")
        self.assertIn("no usable position", tracker.reason)
        self.assertIn("tracking LOST", tracker.status)


if __name__ == "__main__":
    unittest.main()
