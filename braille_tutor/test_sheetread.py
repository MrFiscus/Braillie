"""Tests for reading our known sheets by observation (sheetread.py), the actual poke_*.png files, and the row/column shift fix.

The simulated photos come from tools/: embossed dots with side lighting, paper grain, blur and uneven light, seen by a 1080p camera.
"""
import glob
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "tools"))

import detect
import make_sheet
import poke_files as pf
import sheet_eval as se
import sheetread
import sheets
import sim_embossed as sim
from detect import _make_cell, align_shifted
from pagefind import find_page_homography
from page import SHEET_ORIGIN_MM, page_homography

SHIFT = np.array([[1, 0, -SHEET_ORIGIN_MM[0]], [0, 1, -SHEET_ORIGIN_MM[1]], [0, 0, 1.0]])  # paper mm -> page mm
POKE_FILES = {n: Path(__file__).parent / "sheet" / f"poke_{n}.png" for n in sheets.SHEET_NAMES}


def wrong_cells(observed, expected):
    return [(o["row"], o["col"]) for o, e in zip(observed, expected) if o["dots"] != e["dots"]]


class PokeFileTests(unittest.TestCase):
    def test_every_poke_file_has_exactly_the_dots_the_layout_needs(self):
        """The template PNGs are the ground truth for what gets poked; check them against the layout, independently."""
        for folder in ("", "nomarkers"):
            for name in sheets.SHEET_NAMES:
                path = Path(__file__).parent / "sheet" / folder / f"poke_{name}.png"
                if not path.exists():
                    self.skipTest(f"{path} not generated")
                self.assertEqual(len(pf.guide_dots(str(path))), sum(len(c["dots"]) for c in sheets.get_sheet(name).cells), path)

    def test_flipped_guide_dots_land_on_the_layouts_dot_positions(self):
        """After poking and flipping, each guide dot must sit on a slot of the layout (to within a fraction of a millimetre)."""
        import make_sheet as ms
        for name in sheets.SHEET_NAMES:
            path = POKE_FILES[name]
            if not path.exists():
                self.skipTest("poke files not generated")
            wanted = [(x + SHEET_ORIGIN_MM[0], y + SHEET_ORIGIN_MM[1]) for c in sheets.get_sheet(name).cells for x, y in ms.dot_points(c)]
            for x, y in pf.after_flip(pf.guide_dots(str(path))):
                self.assertLess(min(np.hypot(x - wx, y - wy) for wx, wy in wanted), 0.5, (name, x, y))


class PhotoCase(unittest.TestCase):
    def photo_of_file(self, name, fill=0.7, blur=1.0, contrast=24, seed=0, drop=None):
        pts = pf.after_flip(pf.guide_dots(str(POKE_FILES[name])))
        if drop:
            pts = [p for p in pts if p not in drop]
        flat = sim.render_relief_points(pts, sheets.DOT_R_MM, 8.0, (210, 297), contrast=contrast, seed=seed)
        cam = se.camera(flat, fill, blur, seed)
        H, why = find_page_homography(cam, 210, 297)
        self.assertIsNotNone(H, why)
        return cam, SHIFT @ H


class ObservationTests(PhotoCase):
    @unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
    def test_reads_the_actual_poke_files_correctly_in_several_conditions(self):
        total = wrong = 0
        for name in sheets.SHEET_NAMES:
            expected = sheets.get_sheet(name).cells
            for fill, blur, contrast in [(0.95, 1.0, 24), (0.70, 1.8, 24), (0.70, 1.2, 12)]:
                cam, H = self.photo_of_file(name, fill, blur, contrast)
                bad = wrong_cells(sheetread.observe(cam, H, expected), expected)
                total += len(expected)
                wrong += len(bad)
        print(f"\n  poke files through the camera: {total - wrong}/{total} cells read correctly")
        self.assertLessEqual(wrong, 0.01 * total)

    @unittest.skipUnless(POKE_FILES["alphabet"].exists(), "poke files not generated")
    def test_a_dot_that_was_not_poked_is_reported_as_a_mismatch(self):
        expected = sheets.get_sheet("alphabet").cells
        target = expected[3]  # the letter d: dots 1, 4, 5
        dots = {(x + SHEET_ORIGIN_MM[0], y + SHEET_ORIGIN_MM[1]) for x, y in make_sheet.dot_points(target)}
        pts = pf.after_flip(pf.guide_dots(str(POKE_FILES["alphabet"])))
        gone = [p for p in pts if min(np.hypot(p[0] - dx, p[1] - dy) for dx, dy in dots) < 0.5][:1]  # pts are already flipped to the face
        cam, H = self.photo_of_file("alphabet", drop=gone)
        matching, total, bad = sheetread.compare(sheetread.observe(cam, H, expected), expected)
        self.assertEqual(total, 26)
        self.assertEqual(matching, 25)
        self.assertEqual((bad[0][0], bad[0][1]), (target["row"], target["col"]))
        self.assertLess(len(bad[0][3]), len(bad[0][2]))  # it saw one dot fewer than the sheet says

    @unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
    def test_the_wrong_sheet_on_the_desk_is_obvious(self):
        cam, H = self.photo_of_file("numbers")
        matching, total, bad = sheetread.compare(sheetread.observe(cam, H, sheets.get_sheet("alphabet").cells), sheets.get_sheet("alphabet").cells)
        self.assertLess(matching, 0.5 * total)

    def test_flat_printed_test_sheets_read_correctly_too(self):
        for name in sheets.SHEET_NAMES:
            face = cv2.cvtColor(make_sheet.render_flat_test(name), cv2.COLOR_GRAY2BGR)
            h, w = face.shape[:2]
            width = 560
            ph = int(width * h / w)
            quad = [[400, 40], [400 + width, 50], [400 + width - 8, 40 + ph], [406, 32 + ph]]
            T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
            mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T, (1280, 960))
            img = np.where(mask[..., None] > 0, cv2.warpPerspective(face, T, (1280, 960)), np.full((960, 1280, 3), 70, np.uint8))
            img = cv2.GaussianBlur(img, (0, 0), 0.9)
            H = page_homography(img)
            self.assertIsNotNone(H, name)
            expected = sheets.get_sheet(name).cells
            self.assertEqual(wrong_cells(sheetread.observe(img, H, expected), expected), [], name)

    def test_threshold_split_handles_two_classes_and_respects_the_floor(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(6, 0.8, 300), rng.normal(20, 2, 100)])
        t = sheetread._split(values, floor=0.0)
        self.assertTrue(9 < t < 17, t)
        self.assertGreaterEqual(sheetread._split(values, floor=18.0), 18.0)


def misregister(dx=0.0, dy=0.0, scale=1.0, rot=0.0):
    """A registration that is off by a known amount (mm about the page centre), to apply as `P @ H`."""
    c, s, (cx, cy) = np.cos(np.radians(rot)), np.sin(np.radians(rot)), (75.0, 118.0)
    move = lambda x, y: np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])
    return move(dx, dy) @ move(cx, cy) @ np.array([[scale * c, -scale * s, 0], [scale * s, scale * c, 0], [0, 0, 1.0]]) @ move(-cx, -cy)


@unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
class AlignmentTests(PhotoCase):
    """Without markers the page is placed from the paper's edges, which can be a millimetre or three away from where the printer
    put the dots. The reader finds that offset from the dots themselves. Before this, 2 mm off read 16/26 and 3 mm off 0/26."""

    def read(self, name, P, drop=None):
        cam, H = self.photo_of_file(name, drop=drop)
        expected = sheets.get_sheet(name).cells
        return sheetread.observe(cam, P @ H, expected), expected

    def test_registration_off_by_up_to_four_mm_still_reads_every_cell(self):
        for name in ("alphabet", "lookalikes"):
            for dx, dy in [(1.5, 0.9), (2.0, 1.2), (3.0, 1.8), (-3.0, -3.0), (4.0, 0.0), (0.0, -4.0)]:
                with self.subTest(sheet=name, dx=dx, dy=dy):
                    observed, expected = self.read(name, misregister(dx, dy))
                    self.assertEqual(wrong_cells(observed, expected), [])

    def test_the_offset_is_found_and_reported(self):
        base, _ = self.read("alphabet", misregister())  # the paper-edge finder is itself a little off, even on a clean photo
        moved, expected = self.read("alphabet", misregister(2.0, -1.0))
        (bx, by), (sx, sy) = base[0]["shift"], moved[0]["shift"]
        self.assertAlmostEqual(sx - bx, 2.0, delta=0.4)  # registration moved by (+2, -1) mm: the dots appear that much further on
        self.assertAlmostEqual(sy - by, -1.0, delta=0.4)
        self.assertAlmostEqual(moved[0]["x"], expected[0]["x"] + sx, places=6)  # cells report where the dots really are

    def test_scale_and_rotation_errors_still_read(self):
        for P in (misregister(scale=1.02), misregister(rot=1.0), misregister(1.5, 1.0, 1.01, 0.5)):
            observed, expected = self.read("alphabet", P)
            self.assertEqual(wrong_cells(observed, expected), [])

    def test_a_missed_poke_is_still_a_mismatch_when_misregistered(self):
        """Aligning must not bend the reading toward what is expected."""
        expected = sheets.get_sheet("alphabet").cells
        target = expected[3]
        dots = {(x + SHEET_ORIGIN_MM[0], y + SHEET_ORIGIN_MM[1]) for x, y in make_sheet.dot_points(target)}
        pts = pf.after_flip(pf.guide_dots(str(POKE_FILES["alphabet"])))
        gone = [p for p in pts if min(np.hypot(p[0] - dx, p[1] - dy) for dx, dy in dots) < 0.5][:1]
        observed, expected = self.read("alphabet", misregister(2.5, 1.5), drop=gone)
        self.assertEqual(wrong_cells(observed, expected), [(target["row"], target["col"])])

    def test_the_wrong_sheet_is_still_obvious_when_misregistered(self):
        cam, H = self.photo_of_file("numbers")
        expected = sheets.get_sheet("alphabet").cells
        matching, total, _ = sheetread.compare(sheetread.observe(cam, misregister(2.5, 1.5) @ H, expected), expected)
        self.assertLess(matching, 0.5 * total)

    def test_align_off_leaves_positions_alone_and_is_the_old_behaviour(self):
        cam, H = self.photo_of_file("alphabet")
        expected = sheets.get_sheet("alphabet").cells
        observed = sheetread.observe(cam, H, expected, self_align=False)
        self.assertEqual(observed[0]["shift"], (0.0, 0.0))
        self.assertEqual(observed[0]["x"], expected[0]["x"])

    def test_a_read_is_fast_enough_to_keep_up_with_the_camera(self):
        import time
        cam, H = self.photo_of_file("alphabet")
        expected = sheets.get_sheet("alphabet").cells
        start = time.time()
        sheetread.observe(cam, H, expected)
        self.assertLess(time.time() - start, 1.5)  # was about 3 s before the working resolution came down


class HelperTests(unittest.TestCase):
    @staticmethod
    def cells():
        return [_make_cell(10.0 + 19 * i, 20.0, 9.2, 15.2, lab, 1.0, 0, i) for i, lab in enumerate(["100000", "110000", "100100"])]

    def test_compare_lists_only_the_cells_that_differ(self):
        expected = self.cells()
        observed = [{**c} for c in expected]
        observed[1] = {**observed[1], "dots": frozenset({1})}
        good, total, bad = sheetread.compare(observed, expected)
        self.assertEqual((good, total, [(r, c) for r, c, _, _ in bad]), (2, 3, [(0, 1)]))

    def test_sheet_check_line_wording_and_colour(self):
        text, colour = detect.sheet_check_line((26, 26, []))
        self.assertIn("all 26 cells read correctly", text)
        self.assertEqual(colour, detect.GREEN)
        text, colour = detect.sheet_check_line((25, 26, [(1, 2, frozenset({1, 4, 5}), frozenset({1, 4}))]))
        self.assertIn("25/26", text)
        self.assertIn("row 2 col 3", text)
        self.assertEqual(colour, detect.AMBER)
        self.assertEqual(detect.sheet_check_line((10, 26, [(0, 0, frozenset({1}), frozenset())]))[1], detect.RED)

    def test_display_boxes_skips_blank_cells_and_uses_pixels(self):
        cells = [{**c, "dots": frozenset(), "label": "000000", "confidence": 0.0} if i == 1 else {**c, "confidence": 0.9}
                 for i, c in enumerate(self.cells())]
        boxes = sheetread.display_boxes(cells, np.diag([0.1, 0.1, 1.0]))  # page mm -> image px with H = 0.1 mm/px, so px = mm * 10
        self.assertEqual(len(boxes), 2)
        x1, y1, x2, y2, label, conf = boxes[0]
        self.assertAlmostEqual((x2 - x1), 92.0, delta=1)  # a 9.2 mm cell is 92 px here


class ShiftFixTests(unittest.TestCase):
    """align_shifted: the detector places the box one dot spacing too low for cells with no top row, and reads the pattern as if
    it were top-aligned. Lines of braille are a regular grid, so the offset gives it away."""
    D, H, W, PITCH = 6.0, 17.3, 11.0, 37.0

    def cell(self, line, col, label, extra_down=0.0):
        top = 30.0 + line * self.PITCH + extra_down
        c = _make_cell(20.0 + col * 19.0, top + self.H / 2, self.W, self.H, label, 0.9, line, col)
        return c

    def page(self, strip_label="110100", strip_shift=6.0):
        cells = [self.cell(line, col, "100100") for line in range(4) for col in range(6)]
        cells += [self.cell(4, col, strip_label, strip_shift) for col in range(6)]
        return cells

    def test_a_row_of_shifted_boxes_is_moved_back_down_a_row(self):
        fixed = align_shifted(self.page("110100"))
        strip = [c for c in fixed if c["row"] == 4]
        self.assertTrue(all(c["label"] == "011010" for c in strip), [c["label"] for c in strip])  # 124 -> 235
        self.assertTrue(all(abs(c["y"] - (30 + 4 * 37 + self.H / 2)) < 0.5 for c in strip))       # and the centre back on its line
        self.assertTrue(all(c["label"] == "100100" for c in fixed if c["row"] < 4))               # normal cells untouched

    def test_two_rows_of_shift(self):
        fixed = align_shifted(self.page("100000", strip_shift=12.0))
        self.assertTrue(all(c["label"] == "001000" for c in fixed if c["row"] == 4))              # dot 1 -> dot 3

    def test_nothing_moves_when_every_box_is_normal(self):
        cells = [self.cell(line, col, "100110") for line in range(5) for col in range(6)]
        self.assertEqual([c["label"] for c in align_shifted(cells)], [c["label"] for c in cells])

    def test_a_shift_that_would_leave_the_cell_is_ignored(self):
        cells = self.page("001001", strip_shift=6.0)  # dots 3 and 6 are already on the bottom row: cannot move down
        strip = [c for c in align_shifted(cells) if c["row"] == 4]
        self.assertTrue(all(c["label"] == "001001" for c in strip))

    def test_too_many_moves_means_the_grid_estimate_is_wrong_so_nothing_changes(self):
        cells = [self.cell(line, col, "110100", 6.0 if (line + col) % 2 else 0.0) for line in range(5) for col in range(6)]
        self.assertEqual([c["label"] for c in align_shifted(cells)], [c["label"] for c in cells])

    def test_small_jitter_is_not_mistaken_for_a_shift(self):
        rng = np.random.default_rng(1)
        cells = [self.cell(line, col, "100100", float(rng.normal(0, 0.6))) for line in range(5) for col in range(6)]
        self.assertTrue(all(c["label"] == "100100" for c in align_shifted(cells)))

    def test_too_few_cells_are_returned_unchanged(self):
        cells = [self.cell(0, i, "110100", 6.0) for i in range(4)]
        self.assertEqual(align_shifted(cells), cells)


if __name__ == "__main__":
    unittest.main()
