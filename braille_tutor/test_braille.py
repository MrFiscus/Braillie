"""Quick checks. Run: python -m unittest -v"""
import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import make_sheet
import page
from detect import (WEIGHTS, cells_from_boxes, cells_from_layout, dot_distance, dots_to_char, dots_to_label,
                    label_to_dots, load_cells, nearest_cell, rows_of, save_cells, scan_page)
from make_sheet import dot_points, sheet_cells

REPO = Path(__file__).parent / "third_party/DotNeuralNet"


class LabelTests(unittest.TestCase):
    def test_known_letters(self):
        for label, dots, char in [("100000", {1}, "⠁"), ("110000", {1, 2}, "⠃"), ("100110", {1, 4, 5}, "⠙"),
                                  ("110100", {1, 2, 4}, "⠋"), ("000000", set(), "⠀"), ("111111", {1, 2, 3, 4, 5, 6}, "⠿")]:
            self.assertEqual(label_to_dots(label), frozenset(dots))
            self.assertEqual(dots_to_char(label_to_dots(label)), char)

    def test_all_64_round_trip(self):
        for n in range(64):
            label = format(n, "06b")
            dots = label_to_dots(label)
            self.assertIsInstance(dots, frozenset)
            self.assertEqual(dots_to_label(dots), label)
            self.assertEqual(len(set(dots_to_char(label_to_dots(format(m, "06b"))) for m in range(64))), 64)

    def test_bad_input(self):
        for bad in ("10000", "1000000", "10a000", ""):
            with self.assertRaises(ValueError):
                label_to_dots(bad)
        with self.assertRaises(ValueError):
            dots_to_char({7})

    def test_dot_distance(self):
        d, f = label_to_dots("100110"), label_to_dots("110100")
        self.assertEqual(dot_distance(d, f), 2)  # d={1,4,5} f={1,2,4}
        self.assertEqual(dot_distance(d, d), 0)
        self.assertEqual(dot_distance({1}, {1, 2}), 1)
        self.assertEqual(dot_distance(set(), {1, 2, 3, 4, 5, 6}), 6)

    @unittest.skipUnless((REPO / "src/utils/alphabet_map.json").exists(), "repo not cloned")
    def test_letter_table_matches_repo_alphabet_data(self):
        ref = json.loads((REPO / "src/utils/alphabet_map.json").read_text())
        cells = cells_from_layout(["".join(ref)], 0, 0, 10, 10)
        self.assertEqual([c["label"] for c in cells], list(ref.values()))


class HomographyTests(unittest.TestCase):
    S, MARGIN, MK = 5, 25, int(page.MARKER_SIZE_MM)  # canvas px per mm, canvas margin mm, printed marker mm

    def render(self, skip=()):
        """White page canvas with ArUco markers centred on the page corners, then warped by a known tilt."""
        W, H, S, M = page.PAGE_W_MM, page.PAGE_H_MM, self.S, self.MARGIN
        canvas = np.full((int((H + 2 * M) * S), int((W + 2 * M) * S)), 255, np.uint8)
        d = cv2.aruco.getPredefinedDictionary(page.ARUCO_DICT)
        m = int(self.MK * S)
        for i, (x, y) in page.MARKER_POS_MM.items():
            if i in skip:
                continue
            cx, cy = int((x + M) * S), int((y + M) * S)
            canvas[cy - m // 2 : cy + m // 2, cx - m // 2 : cx + m // 2] = cv2.aruco.generateImageMarker(d, i, m)
        h, w = canvas.shape
        quad = np.float32([[150, 80], [1100, 140], [1180, 880], [100, 820]])  # tilted, perspective
        T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
        return cv2.warpPerspective(canvas, T, (1280, 960), borderValue=255), T

    def truth_px(self, T, x, y):
        """Where page point (x, y) mm really landed in the warped image."""
        p = T @ np.array([(x + self.MARGIN) * self.S, (y + self.MARGIN) * self.S, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def test_recovers_page_mapping(self):
        img, T = self.render()
        H = page.page_homography(img)
        self.assertIsNotNone(H)
        worst = 0.0
        for x, y in [(0, 0), (page.PAGE_W_MM, page.PAGE_H_MM), (95, 125), (40, 200), (150, 30)]:
            px, py = self.truth_px(T, x, y)
            gx, gy = page.to_page(H, px, py)
            worst = max(worst, abs(gx - x), abs(gy - y))
            ix, iy = page.to_image(H, x, y)
            self.assertLess(np.hypot(ix - px, iy - py), 2.0)  # px
        print(f"\n  homography worst error on synthetic tilt: {worst:.3f} mm")
        self.assertLess(worst, 1.0)
        rx, ry = page.to_image(H, *page.to_page(H, 321.0, 456.0))
        self.assertAlmostEqual(rx, 321.0, places=3)
        self.assertAlmostEqual(ry, 456.0, places=3)

    def test_visible_markers_counts_ids(self):
        img, _ = self.render()
        self.assertEqual(sorted(page.visible_markers(img)), [0, 1, 2, 3])
        img, _ = self.render(skip=(2, 3))
        self.assertEqual(sorted(page.visible_markers(img)), [0, 1])
        self.assertEqual(page.visible_markers(np.full((480, 640, 3), 255, np.uint8)), {})

    def test_none_unless_all_four(self):
        img, _ = self.render(skip=(2,))
        self.assertIsNone(page.page_homography(img))
        self.assertIsNone(page.page_homography(np.full((480, 640, 3), 255, np.uint8)))


def _boxes_for(cells_mm, H_img_to_page, jitter=0.0, seed=0):
    """Fake detector output: pixel boxes (x1,y1,x2,y2,label,conf) around given mm centres."""
    rnd, inv, out = random.Random(seed), np.linalg.inv(H_img_to_page), []
    for x, y, label, conf in cells_mm:
        px, py = page.to_page(inv, x, y)
        px, py = px + rnd.uniform(-jitter, jitter), py + rnd.uniform(-jitter, jitter)
        out.append((px - 20, py - 30, px + 20, py + 30, label, conf))
    rnd.shuffle(out)
    return out


class DiagnoseTests(unittest.TestCase):
    """The on-screen reason when the page can't be registered must name the real problem."""

    def note(self, img):
        H, note = page.diagnose_markers(img)
        return H, note

    def test_all_good(self):
        img, _ = HomographyTests().render()
        H, note = self.note(img)
        self.assertIsNotNone(H)
        self.assertEqual(note, "")

    def test_no_markers(self):
        H, note = self.note(np.full((480, 640, 3), 255, np.uint8))
        self.assertIsNone(H)
        self.assertIn("no markers in view", note)
        self.assertIn("very dark", self.note(np.full((480, 640, 3), 5, np.uint8))[1])

    def test_three_markers_still_register(self):
        """One marker hidden (a hand reaching in to point near it, typically) must not lose page registration
        outright: the other 3 markers' 12 corners are enough for an exact homography fit."""
        img, T = HomographyTests().render(skip=(2,))
        H, note = self.note(img)
        self.assertIsNotNone(H)
        self.assertIn("3 of 4 markers", note)
        self.assertIn("[2]", note)
        worst = 0.0
        for x, y in [(0, 0), (page.PAGE_W_MM, page.PAGE_H_MM), (95, 125), (40, 200), (150, 30)]:
            px, py = HomographyTests().truth_px(T, x, y)
            gx, gy = page.to_page(H, px, py)
            worst = max(worst, abs(gx - x), abs(gy - y))
        self.assertLess(worst, 1.0)

    def test_too_few_markers_named(self):
        img, _ = HomographyTests().render(skip=(1, 2))
        H, note = self.note(img)
        self.assertIsNone(H)
        self.assertIn("missing marker(s)", note)
        self.assertIn("[1, 2]", note)

    def test_wrong_marker_ids(self):
        d = cv2.aruco.getPredefinedDictionary(page.ARUCO_DICT)
        img = np.full((400, 400), 255, np.uint8)
        img[100:300, 100:300] = cv2.aruco.generateImageMarker(d, 7, 200)
        H, note = self.note(img)
        self.assertIsNone(H)
        self.assertIn("[7]", note)

    def test_mirrored_markers_are_reported_as_undecodable(self):
        img, _ = HomographyTests().render()
        H, note = self.note(cv2.flip(img, 1))
        self.assertIsNone(H)
        self.assertIn("none decode", note)

    def test_markers_in_wrong_corners(self):
        """Markers 1 and 3 swapped: all four are readable but the sheet can't be right."""
        d = cv2.aruco.getPredefinedDictionary(page.ARUCO_DICT)
        img = np.full((900, 700), 255, np.uint8)
        for i, (cx, cy) in {0: (150, 150), 1: (150, 750), 2: (550, 750), 3: (550, 150)}.items():
            img[cy - 60 : cy + 60, cx - 60 : cx + 60] = cv2.aruco.generateImageMarker(d, i, 120)
        H, note = self.note(img)
        self.assertIsNone(H)
        self.assertIn("wrong corners", note)

    def test_small_markers_warn_but_still_register(self):
        img, _ = HomographyTests().render()
        small = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)  # markers ~20 px wide
        H, note = self.note(small)
        self.assertIsNotNone(H)
        self.assertIn("markers small", note)

    def test_tracker_reasons(self):
        from tracker import PageTracker
        tr = PageTracker(TrackerTests.textured_page(), np.diag([0.1, 0.1, 1.0]))
        tr.homography(np.full((900, 1200, 3), 128, np.uint8))
        self.assertIn("LOST", tr.status)
        self.assertIn("texture", tr.status)


class PageFinderTests(unittest.TestCase):
    """Automatic page-edge detection (--auto-page) on a synthetic desk scene with a textured page."""
    W_MM, H_MM = 200.0, 190.0
    QUAD = [[260, 90], [980, 120], [1010, 860], [230, 840]]

    @staticmethod
    def scene(desk=60, quad=None, page=True, hand=None, receipt_behind=None):
        content = TrackerTests.textured_page()  # 1600 x 1200 px "page" with texture
        rng = np.random.default_rng(3)
        out = cv2.GaussianBlur(np.full((960, 1280, 3), desk, np.float32) + rng.normal(0, 6, (960, 1280, 1)), (0, 0), 2).astype(np.uint8)
        if receipt_behind:  # a bright slip of paper lying UNDER the page, poking out past one corner
            cv2.rectangle(out, receipt_behind[0], receipt_behind[1], (225, 225, 225), -1)
        if page:
            ch, cw = content.shape[:2]
            T = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), np.float32(quad or PageFinderTests.QUAD))
            mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), T, (1280, 960))
            out = np.where(mask[..., None] > 0, cv2.warpPerspective(content, T, (1280, 960)), out)
        if hand:
            cv2.ellipse(out, (620, 850), (150, 90), 10, 0, 360, (120, 150, 190), -1)
        return out

    def test_order_corners(self):
        import pagefind
        q = pagefind.order_corners([(900, 800), (100, 100), (880, 90), (110, 820)])
        np.testing.assert_array_equal(q, np.float32([[100, 100], [880, 90], [900, 800], [110, 820]]))

    def test_finds_page_accurately(self):
        import pagefind
        for kw in ({}, {"hand": True}, {"desk": 120}):
            q, why = pagefind.find_page(self.scene(**kw), self.W_MM, self.H_MM)
            self.assertIsNotNone(q, why)
            self.assertLess(np.abs(q - np.float32(self.QUAD)).max(), 3.0, kw)

    def test_white_object_touching_the_page_is_rejected_not_misregistered(self):
        """A bright receipt overlapping a corner merges with the page; a wrong quad is worse than no quad."""
        import pagefind
        img = self.scene()
        cv2.rectangle(img, (150, 40), (420, 150), (215, 215, 215), -1)  # receipt sticking out past the top-left corner
        q, why = pagefind.find_page(img, self.W_MM, self.H_MM)
        if q is not None:  # if it does return a quad, it must still be the true page
            self.assertLess(np.abs(q - np.float32(self.QUAD)).max(), 6.0)
        else:
            self.assertIn("edge", why)

    def test_bright_slip_lying_under_a_page_corner_does_not_fool_the_finder(self):
        """The real overview photo had a white receipt behind the page's top-left corner; the corner must stay the page's."""
        import pagefind
        for slip in (((150, 30), (420, 150)), ((230, 20), (480, 130)), ((180, 60), (330, 200))):
            q, why = pagefind.find_page(self.scene(receipt_behind=slip), self.W_MM, self.H_MM)
            self.assertIsNotNone(q, (slip, why))
            self.assertLess(np.abs(q - np.float32(self.QUAD)).max(), 6.0, slip)

    def test_darker_rectangle_is_not_the_page(self):
        """A dark folder on a lighter table must not be mistaken for the (white) page."""
        import pagefind
        img = self.scene(desk=170, page=False)
        cv2.fillConvexPoly(img, np.int32([[260, 90], [980, 120], [1010, 860], [230, 840]]), (40, 40, 45))
        q, why = pagefind.find_page(img, self.W_MM, self.H_MM)
        self.assertIsNone(q)

    def test_no_false_positive_on_empty_desk(self):
        import pagefind
        q, why = pagefind.find_page(self.scene(page=False), self.W_MM, self.H_MM)
        self.assertIsNone(q)
        self.assertNotEqual(why, "")

    def test_wrong_page_size_is_explained(self):
        import pagefind
        q, why = pagefind.find_page(self.scene(), 400.0, 100.0)
        self.assertIsNone(q)
        self.assertIn("width:height", why)

    def test_homography_maps_corners_to_mm(self):
        import pagefind
        H, why = pagefind.find_page_homography(self.scene(), self.W_MM, self.H_MM)
        self.assertIsNotNone(H, why)
        for (px, py), (mx, my) in zip(self.QUAD, [(0, 0), (200, 0), (200, 190), (0, 190)]):
            gx, gy = page.to_page(H, px, py)
            self.assertAlmostEqual(gx, mx, delta=1.0)
            self.assertAlmostEqual(gy, my, delta=1.0)

    def test_autopage_locks_then_tracks_a_moved_camera(self):
        import pagefind
        ap = pagefind.AutoPage(self.W_MM, self.H_MM)
        frame = self.scene()
        self.assertIsNone(pagefind.AutoPage(self.W_MM, self.H_MM).homography(self.scene(page=False)))
        for _ in range(pagefind.STEADY_FRAMES):
            H = ap.homography(frame)
            self.assertIsNotNone(H)
        self.assertIsNotNone(ap.tracker, "should have locked after steady frames")
        R = np.vstack([cv2.getRotationMatrix2D((640, 480), 6, 1.25), [0, 0, 1]])  # camera moves closer and turns
        moved = cv2.warpPerspective(frame, R, (1280, 960), borderMode=cv2.BORDER_REPLICATE)
        H2 = ap.homography(moved)
        corner = R @ [*self.QUAD[2], 1.0]  # page's bottom-right corner in the moved view
        gx, gy = page.to_page(H2, corner[0] / corner[2], corner[1] / corner[2])
        self.assertAlmostEqual(gx, 200, delta=1.5)
        self.assertAlmostEqual(gy, 190, delta=1.5)
        ap.unlock()
        self.assertIsNone(ap.tracker)


class VoteTests(unittest.TestCase):
    """vote.py: pooled scans must give stable labels, drop ghosts, and respect the page."""

    @staticmethod
    def scan(labels, conf=0.8, extra=()):
        """A scan of a 1 x N row of cells 10 mm apart, with the given labels (plus any extra (x, y, label))."""
        from detect import _make_cell
        cells = [_make_cell(10.0 * i, 5.0, 6.0, 9.0, lab, conf) for i, lab in enumerate(labels)]
        cells += [_make_cell(x, y, 6.0, 9.0, lab, conf) for x, y, lab in extra]
        return cells

    def test_majority_label_wins_over_flicker(self):
        from vote import CellVoter
        v = CellVoter(frames=6)
        for lab in ("100110", "100110", "110100", "100110", "100110", "100110"):  # d, d, f, d, d, d
            v.add(self.scan(["100000", lab]))
        out = v.result()
        self.assertEqual([c["label"] for c in out], ["100000", "100110"])
        self.assertGreater(out[0]["confidence"], 0.95)
        self.assertAlmostEqual(out[1]["confidence"], 5 / 6, delta=0.02)  # 5 of 6 votes agree

    def test_one_frame_ghost_is_dropped(self):
        from vote import CellVoter
        v = CellVoter(frames=8)
        for i in range(8):
            v.add(self.scan(["100000", "110000"], extra=[(50.0, 5.0, "111111")] if i == 3 else []))
        self.assertEqual(len(v.result()), 2)

    def test_small_position_jitter_is_one_cell(self):
        from vote import CellVoter
        v = CellVoter(frames=5)
        for i in range(5):
            v.add([c | {"x": c["x"] + (i - 2) * 0.4} for c in self.scan(["100000", "110000"])])
        out = v.result()
        self.assertEqual(len(out), 2)
        self.assertEqual([(c["row"], c["col"]) for c in out], [(0, 0), (0, 1)])

    def test_reset_and_empty(self):
        from vote import CellVoter
        v = CellVoter()
        self.assertEqual(v.result(), [])
        v.add(self.scan(["100000"]))
        v.reset()
        self.assertEqual(v.result(), [])

    def test_keep_inside_and_first_rows(self):
        from vote import first_rows, keep_inside
        from detect import _make_cell, _assign_grid
        cells = _assign_grid([_make_cell(x, y, 6.0, 9.0, "100000", 0.9) for x, y in [(10, 10), (30, 10), (10, 40), (500, 10), (10, -20)]])
        self.assertEqual(len(keep_inside(cells, 100, 100)), 3)  # x=500 and y=-20 are off the page
        self.assertEqual({c["row"] for c in first_rows(_assign_grid(keep_inside(cells, 100, 100)), 1)}, {0})


class PhoneCommandTests(unittest.TestCase):
    """phone_command must hit the right paths on a phone app's web server, and fail politely otherwise."""

    def test_sends_focus_and_torch_requests(self):
        import http.server
        import threading
        from detect import phone_command
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}/"
        try:
            for action in ("focus", "torch_on", "torch_off"):
                self.assertIn("sent (200)", phone_command(base, action))
        finally:
            srv.shutdown()
        self.assertEqual(seen, ["/focus", "/enabletorch", "/disabletorch"])

    def test_failures_are_reported_not_raised(self):
        from detect import phone_command
        self.assertIn("FAILED", phone_command("http://127.0.0.1:9", "focus"))  # nothing listens on port 9
        self.assertIn("must start with http", phone_command("file:///etc/passwd", "focus"))


class CalibrationTests(unittest.TestCase):
    def test_clicked_corners_give_same_mapping_as_markers(self):
        """Clicking the four marker centres must reproduce what marker detection gives."""
        img, T = HomographyTests().render()
        H_markers = page.page_homography(img)
        clicks = [page.to_image(H_markers, *page.MARKER_POS_MM[i]) for i in range(4)]
        H = page.homography_from_corners(clicks, page.PAGE_W_MM, page.PAGE_H_MM)
        for x, y in [(0, 0), (40, 80), (page.PAGE_W_MM, page.PAGE_H_MM)]:
            px, py = page.to_image(H_markers, x, y)
            gx, gy = page.to_page(H, px, py)
            self.assertAlmostEqual(gx, x, delta=0.01)
            self.assertAlmostEqual(gy, y, delta=0.01)

    def test_clicking_paper_corners_reproduces_sheet_coordinates(self):
        """calibrate.py --sheet: the four A4 corners must land the layout dots exactly where the markers would."""
        face = make_sheet.render_face_preview()
        h, w = face.shape[:2]
        quad = np.float32([[160, 60], [1000, 120], [1080, 880], [90, 840]])  # camera view of the sheet
        T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
        H = page.homography_from_corners(quad, *make_sheet.A4_MM, origin=(-make_sheet.ORIGIN[0], -make_sheet.ORIGIN[1]))
        for c in sheet_cells():
            for x, y in dot_points(c):
                p = T @ np.array([(make_sheet.ORIGIN[0] + x) * make_sheet.MM, (make_sheet.ORIGIN[1] + y) * make_sheet.MM, 1.0])
                gx, gy = page.to_page(H, p[0] / p[2], p[1] / p[2])
                self.assertAlmostEqual(gx, x, delta=0.1)  # 0.1 mm: the rendered image is 209.97 mm wide, not exactly 210
                self.assertAlmostEqual(gy, y, delta=0.1)

    def test_save_load_round_trip(self):
        H = page.homography_from_corners([(10, 20), (300, 30), (310, 200), (5, 190)], 120, 60)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "c.json")
            page.save_homography(H, p)
            np.testing.assert_allclose(page.load_homography(p), H)


class TrackerTests(unittest.TestCase):
    @staticmethod
    def textured_page():
        """A synthetic 'page' with plenty of distinct texture (random blobs), deterministic."""
        rng = np.random.default_rng(1)
        img = np.full((1200, 1600), 200, np.uint8)
        for _ in range(900):
            cv2.circle(img, (int(rng.integers(0, 1600)), int(rng.integers(0, 1200))), int(rng.integers(3, 14)),
                       int(rng.integers(40, 250)), -1)
        return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.COLOR_GRAY2BGR)

    def test_follows_zoom_and_rotation(self):
        from tracker import PageTracker
        ref = self.textured_page()
        Href = np.diag([0.1, 0.1, 1.0])
        tr = PageTracker(ref, Href)
        for zoom, ang in [(1.0, 0), (1.6, 10), (2.0, -15), (0.7, 25)]:
            R = np.vstack([cv2.getRotationMatrix2D((800, 600), ang, zoom), [0, 0, 1]])
            frame = cv2.warpPerspective(ref, R, (1600, 1200), borderMode=cv2.BORDER_REPLICATE)
            H = tr.homography(frame)
            for x, y in [(700, 500), (900, 650)]:  # page points that stay in view
                p = R @ [x, y, 1.0]
                gx, gy = page.to_page(H, p[0] / p[2], p[1] / p[2])
                self.assertAlmostEqual(gx, x * 0.1, delta=0.2)
                self.assertAlmostEqual(gy, y * 0.1, delta=0.2)

    def test_holds_last_good_when_lost(self):
        from tracker import PageTracker
        tr = PageTracker(self.textured_page(), np.diag([0.1, 0.1, 1.0]))
        H = tr.homography(np.full((900, 1200, 3), 128, np.uint8))  # featureless frame: nothing to match
        np.testing.assert_allclose(H, np.diag([0.1, 0.1, 1.0]))
        self.assertIn("LOST", tr.status)


class GroupingTests(unittest.TestCase):
    def test_rows_cols_dedupe_and_mm(self):
        H = np.diag([0.25, 0.25, 1.0])  # 1 px = 0.25 mm
        grid = [(100 * c, 120 * r, format(1 + 3 * r + c, "06b"), 0.9) for r in range(2) for c in range(3)]
        boxes = _boxes_for([(x * 0.25, y * 0.25, l, cf) for x, y, l, cf in grid], H, jitter=5)
        x1, y1, x2, y2, *_ = boxes[0]  # near-identical duplicates of a real box, lower confidence, other labels
        boxes.append((x1 + 2, y1 + 3, x2 + 2, y2 + 3, "111111", 0.30))
        boxes.append((x1 - 3, y1 - 2, x2 - 3, y2 - 2, "000000", 0.10))
        cells = cells_from_boxes(boxes, H)
        self.assertEqual(len(cells), 6)
        self.assertEqual([(c["row"], c["col"]) for c in cells], [(r, c) for r in range(2) for c in range(3)])
        self.assertEqual([c["label"] for c in cells], [format(1 + i, "06b") for i in range(6)])  # duplicates lost
        self.assertAlmostEqual(cells[1]["x"] - cells[0]["x"], 25.0, delta=3)  # 100 px * 0.25 mm/px
        self.assertAlmostEqual(cells[3]["y"] - cells[0]["y"], 30.0, delta=3)
        self.assertAlmostEqual(cells[0]["w"], 10.0, delta=0.01)
        self.assertEqual([len(r) for r in rows_of(cells)], [3, 3])

    def test_tilted_camera(self):
        """Rows still come out horizontal because grouping happens in page mm, not pixels."""
        rot = cv2.getRotationMatrix2D((0, 0), 12, 4.0)  # page mm -> pixels: rotated 12 deg, 4 px/mm
        H = np.linalg.inv(np.vstack([rot, [0, 0, 1]]))  # pixels -> page mm
        grid = [(15 * c, 20 * r, format(c + 4 * r, "06b"), 0.8) for r in range(3) for c in range(4)]
        cells = cells_from_boxes(_boxes_for(grid, H), H)
        self.assertEqual(len(cells), 12)
        self.assertEqual([(c["row"], c["col"]) for c in cells], [(r, c) for r in range(3) for c in range(4)])
        self.assertEqual([c["label"] for c in cells], [format(i, "06b") for i in range(12)])

    def test_empty(self):
        self.assertEqual(cells_from_boxes([], np.eye(3)), [])


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.cells = cells_from_layout(["abc", "def"], x0=10, y0=10, pitch_x=20, pitch_y=30)

    def test_nearest(self):
        self.assertEqual(nearest_cell(self.cells, 31, 12)["char"], "⠃")  # b
        self.assertEqual(nearest_cell(self.cells, 48, 41)["char"], "⠋")  # f
        self.assertEqual(nearest_cell(self.cells, 10, 10)["char"], "⠁")

    def test_default_max_dist_is_half_spacing(self):
        # nearest-neighbour spacing is 20 mm, so the cutoff is 10 mm
        self.assertIsNotNone(nearest_cell(self.cells, 10, 10 + 9.9))  # rows are 30 mm apart, so probe downward
        self.assertIsNone(nearest_cell(self.cells, 10, 10 + 10.5))
        self.assertIsNone(nearest_cell(self.cells, 500, 500))
        self.assertIsNotNone(nearest_cell(self.cells, 500, 500, max_dist=1e9))

    def test_edge_cases(self):
        self.assertIsNone(nearest_cell([], 0, 0))
        self.assertIsNotNone(nearest_cell(self.cells[:1], 999, 999))  # single cell: no spacing, no cutoff

    def test_layout_matches_scan_structure(self):
        c = self.cells
        self.assertEqual([(x["row"], x["col"]) for x in c], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        self.assertEqual((c[4]["x"], c[4]["y"]), (30, 40))
        self.assertEqual(dots_to_char(c[3]["dots"]), "⠙")  # d
        # same keys as detector output
        det = cells_from_boxes([(0, 0, 10, 10, "100000", 0.9)], np.eye(3))[0]
        self.assertEqual(set(det), set(c[0]))
        with self.assertRaises(ValueError):
            cells_from_layout(["a1"], 0, 0, 1, 1)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "cells.json")
            save_cells(self.cells, p)
            self.assertEqual(load_cells(p), self.cells)
            self.assertIsInstance(load_cells(p)[0]["dots"], frozenset)


class SheetTests(unittest.TestCase):
    """make_sheet.py: the printed geometry, the homography and sheet_cells() must all agree."""

    def test_layout_content(self):
        cells = sheet_cells()
        self.assertEqual(len(cells), 26)
        self.assertEqual("".join(c["char"] for c in rows_of(cells)[0]), "⠁⠃⠉⠙⠑⠋⠛⠓")  # a-h
        self.assertEqual("".join(c["char"] for c in rows_of(cells)[3]), "⠽⠵")  # y z

    def test_dots_fit_page_and_clear_markers(self):
        half = make_sheet.MARKER_MM / 2 + 3
        for c in sheet_cells():
            for x, y in dot_points(c):
                self.assertTrue(0 < x < page.PAGE_W_MM and 0 < y < page.PAGE_H_MM)
                for mx, my in page.MARKER_POS_MM.values():
                    self.assertFalse(abs(x - mx) < half and abs(y - my) < half, (c["char"], x, y))

    def test_mirror_is_left_right_flip_only(self):
        for x, y in [(0, 0), (13.5, 50), (150, 237)]:
            (mx, my), (fx, fy) = make_sheet._px(x, y, True), make_sheet._px(x, y, False)
            self.assertEqual(my, fy)
            self.assertAlmostEqual(mx + fx, 210 * make_sheet.MM, delta=1)

    def test_face_image_dots_land_where_the_homography_says(self):
        """Find the markers in the rendered face, then check dark pixels sit exactly on the layout's dots."""
        img = make_sheet.render_face_preview()
        H = page.page_homography(img)
        self.assertIsNotNone(H)
        n = 0
        for c in sheet_cells():
            for x, y in dot_points(c):
                px, py = page.to_image(H, x, y)
                self.assertLess(img[int(round(py)), int(round(px))], 100, (c["char"], x, y))
                n += 1
            missing = {1, 2, 3, 4, 5, 6} - c["dots"]
            for d in list(missing)[:1]:  # an unraised position must be blank paper
                x, y = c["x"] + ((d > 3) - 0.5) * make_sheet.DOT_MM, c["y"] + ((d - 1) % 3 - 1) * make_sheet.DOT_MM
                px, py = page.to_image(H, x, y)
                self.assertEqual(img[int(round(py)), int(round(px))], 255)
        self.assertGreater(n, 60)

    def test_png_has_300_dpi(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.png"
            make_sheet.save_png(p, np.zeros((10, 10), np.uint8))
            self.assertIn(b"pHYs", p.read_bytes())
            self.assertIsNotNone(cv2.imread(str(p)))


@unittest.skipUnless(WEIGHTS.exists() and (REPO / "assets/alpha-numeric.jpeg").exists(), "weights/example missing")
class ModelSmokeTest(unittest.TestCase):
    def test_scan_repo_example_image(self):
        img = cv2.imread(str(REPO / "assets/alpha-numeric.jpeg"))
        cells = scan_page(img, np.eye(3))
        print(f"\n  model found {len(cells)} cells on the repo's example photo")
        self.assertGreater(len(cells), 5)
        self.assertTrue(all(len(c["char"]) == 1 and 0x2800 <= ord(c["char"]) <= 0x283F for c in cells))


if __name__ == "__main__":
    unittest.main()
