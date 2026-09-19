"""Tests for the printable sheets (sheets.py, make_sheet.py) and the tutor's use of them. Run: python -m unittest test_sheets"""
import argparse
import random
import unittest

import cv2
import numpy as np

import make_sheet
import page
import reader
import sheets
import tutor
from detect import dot_distance
from test_tutor import WC, FakeVoice


class SheetContentTests(unittest.TestCase):
    def test_every_sheet_builds_with_a_name_for_every_cell(self):
        for name in sheets.SHEET_NAMES:
            sh = sheets.get_sheet(name)
            self.assertGreater(len(sh.cells), 10, name)
            spots = {(c["row"], c["col"]) for c in sh.cells}
            self.assertEqual(len(spots), len(sh.cells), f"{name}: two cells in one slot")
            self.assertEqual(spots, set(sh.names), name)
            for c in sh.cells:
                self.assertEqual(sh.symbol(c).dots, c["dots"], (name, c))

    def test_everything_fits_the_page_and_clears_the_markers(self):
        half = make_sheet.MARKER_MM / 2 + 3
        for name in sheets.SHEET_NAMES:
            cells = sheets.get_sheet(name).cells
            for c in cells:
                for x, y in make_sheet.dot_points(c):
                    self.assertTrue(0 < x < page.PAGE_W_MM and 0 < y < page.PAGE_H_MM, (name, x, y))
                    for mx, my in page.MARKER_POS_MM.values():
                        self.assertFalse(abs(x - mx) < half and abs(y - my) < half, (name, c["row"], c["col"]))
                # the printed label under a cell must stay clear of the bottom markers too
                self.assertLess(c["y"] + 2 * sheets.DOT_MM + 6, page.PAGE_H_MM - make_sheet.MARKER_MM / 2, name)

    def test_cells_do_not_overlap(self):
        for name in sheets.SHEET_NAMES:
            cells = sheets.get_sheet(name).cells
            for i, a in enumerate(cells):
                for b in cells[i + 1:]:
                    self.assertTrue(abs(a["x"] - b["x"]) >= a["w"] + 2 or abs(a["y"] - b["y"]) >= a["h"] + 2, (name, a, b))

    def test_numbers_and_signs_have_the_right_dots(self):
        sh = sheets.get_sheet("numbers")
        by_key = {}
        for c in sh.cells:
            by_key.setdefault(sh.symbol(c).key, c["dots"])
        expect = {"#": {3, 4, 5, 6}, "1": {1}, "2": {1, 2}, "3": {1, 4}, "9": {2, 4}, "0": {2, 4, 5}, "^": {6}, ".": {2, 5, 6},
                  ",": {2}, "?": {2, 3, 6}, "!": {2, 3, 5}, "'": {3}, "-": {3, 6}, ";": {2, 3}}
        for key, dots in expect.items():
            self.assertEqual(by_key[key], frozenset(dots), key)
        self.assertEqual(sum(1 for c in sh.cells if sh.symbol(c).key == "#"), 10)  # a number sign before each digit

    def test_number_sign_comes_before_every_digit(self):
        sh = sheets.get_sheet("numbers")
        for c in sh.cells:
            if sh.symbol(c).key.isdigit():
                self.assertEqual(sh.names[(c["row"], c["col"] - 1)].key, "#")

    def test_words_sheet_reads_back_as_real_words(self):
        sh = sheets.get_sheet("words")
        lines = reader.read_lines(sh.cells)
        self.assertEqual(lines, ["cat dog", "sun hat", "red cup", "bed pig", "fish egg", "bird bee"])
        if WC is not None:
            for word in " ".join(lines).split():
                self.assertTrue(WC.is_valid_word(word), word)

    def test_lookalike_pairs_are_what_the_note_claims(self):
        sh = sheets.get_sheet("lookalikes")
        pairs = []
        for r, row in enumerate(sh.spec.rows):
            for group in row.split():
                cells = [next(c for c in sh.cells if c["row"] == r and c["col"] == row.index(group) + i) for i in range(2)]
                pairs.append((r, cells[0]["dots"], cells[1]["dots"]))
        self.assertEqual(len(pairs), 11)
        for r, a, b in pairs[:8]:
            self.assertEqual(dot_distance(a, b), 1, "rows 1-3 differ by one dot")
        mirror = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}
        for r, a, b in pairs[8:]:
            self.assertEqual(frozenset(mirror[d] for d in a), b, "row 4 are mirror images")

    def test_unknown_sheet_and_symbol(self):
        with self.assertRaises(ValueError):
            sheets.get_sheet("nope")
        with self.assertRaises(ValueError):
            sheets.symbol_for("%")


class SheetRenderTests(unittest.TestCase):
    def test_dots_land_where_the_homography_says_for_every_sheet(self):
        for name in sheets.SHEET_NAMES:
            img = make_sheet.render_face_preview(name)
            H = page.page_homography(img)
            self.assertIsNotNone(H, name)
            for c in sheets.get_sheet(name).cells:
                for x, y in make_sheet.dot_points(c):
                    px, py = page.to_image(H, x, y)
                    self.assertLess(img[int(round(py)), int(round(px))], 100, (name, x, y))

    def test_poke_templates_are_mirrored_and_labelled(self):
        for name in sheets.SHEET_NAMES:
            img = make_sheet.render_poke_template(name)
            self.assertEqual(img.shape, (int(297 * make_sheet.MM), int(210 * make_sheet.MM)))
            face = make_sheet.render_face_preview(name)
            # the dots of the poke side are the left-right mirror of the face (ignoring markers/labels): compare dot centres
            for c in sheets.get_sheet(name).cells[:6]:
                for x, y in make_sheet.dot_points(c):
                    mx, my = make_sheet._px(x, y, True)
                    self.assertLess(img[my, mx], 60, (name, x, y))

    def test_main_writes_both_versions_and_the_print_pack(self):
        import os
        import sys
        import tempfile
        old, argv = make_sheet.OUT, sys.argv
        with tempfile.TemporaryDirectory() as d:
            make_sheet.OUT = make_sheet.Path(d)
            sys.argv = ["make_sheet.py"]
            try:
                make_sheet.main()
            finally:
                make_sheet.OUT, sys.argv = old, argv
            files = sorted(str(p.relative_to(d)) for p in make_sheet.Path(d).rglob("*.png"))
        expect = []
        for n in sheets.SHEET_NAMES:
            expect += [f"poke_{n}.png", f"face_{n}.png", f"nomarkers/poke_{n}.png", f"nomarkers/face_{n}.png"]
        expect += ["print/1_alphabet_WITH_markers.png", "print/2_alphabet_NO_markers.png",
                   "print/3_words_WITH_markers.png", "print/4_words_NO_markers.png",
                   "print/flat_test_alphabet.png", "print/flat_test_words.png"]
        self.assertEqual(files, sorted(expect))

    def test_no_marker_version_has_the_same_dots_and_no_marker_marks(self):
        for name in sheets.SHEET_NAMES:
            with_m, without = make_sheet.render_poke_template(name, True), make_sheet.render_poke_template(name, False)
            for c in sheets.get_sheet(name).cells:
                for x, y in make_sheet.dot_points(c):
                    mx, my = make_sheet._px(x, y, True)
                    self.assertLess(with_m[my, mx], 60)
                    self.assertLess(without[my, mx], 60)  # same dots, same places
            for mx, my in page.MARKER_POS_MM.values():
                for sx in (-1, 1):
                    for sy in (-1, 1):  # the pin-prick cross at each marker corner is gone
                        x, y = make_sheet._px(mx + sx * make_sheet.MARKER_MM / 2, my + sy * make_sheet.MARKER_MM / 2, True)
                        self.assertEqual(int(without[y, x]), 255)
                        self.assertLess(int(with_m[y, x]), 60)

    def test_flat_test_sheet_is_readable_by_the_page_finder(self):
        """The print-and-use test sheet must register from its markers and put the layout on its printed dots."""
        img = cv2.cvtColor(make_sheet.render_flat_test("alphabet"), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])))
        H = page.page_homography(img)
        self.assertIsNotNone(H)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for c in sheets.get_sheet("alphabet").cells:
            for x, y in make_sheet.dot_points(c):
                px, py = page.to_image(H, x, y)
                self.assertLess(gray[int(round(py)), int(round(px))], 100, (x, y))

    def test_no_marker_face_has_no_markers(self):
        self.assertIsNone(page.page_homography(make_sheet.render_face_preview("alphabet", markers=False)))
        self.assertIsNotNone(page.page_homography(make_sheet.render_face_preview("alphabet", markers=True)))


def paper_scene(name="alphabet", desk=55, quad=((330, 60), (930, 90), (960, 900), (300, 880))):
    """A marker-free sheet lying on a dark desk, seen at an angle. Returns (image, warp T from paper pixels to image)."""
    face = make_sheet.render_face_preview(name, markers=False)
    h, w = face.shape[:2]
    quad = np.float32(quad)
    T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(np.full((960, 1280), desk, np.float32) + rng.normal(0, 6, (960, 1280)), (0, 0), 2).astype(np.uint8)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T, (1280, 960))
    img = np.where(mask > 0, cv2.warpPerspective(face, T, (1280, 960)), bg).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), T


class PaperDetectionTests(unittest.TestCase):
    ORIGIN = (-page.SHEET_ORIGIN_MM[0], -page.SHEET_ORIGIN_MM[1])

    def assert_dots_dark(self, img, H, name):
        bad = 0
        for c in sheets.get_sheet(name).cells:
            for x, y in make_sheet.dot_points(c):
                px, py = page.to_image(H, x, y)
                bad += img[int(round(py)), int(round(px)), 0] > 100
        self.assertEqual(bad, 0, f"{bad} predicted dot positions were not on a dot")

    def test_paper_edges_register_the_sheet_for_every_design(self):
        import pagefind
        for name in sheets.SHEET_NAMES:
            img, T = paper_scene(name)
            H, why = pagefind.find_page_homography(img, *page.A4_MM, origin=self.ORIGIN)
            self.assertIsNotNone(H, (name, why))
            self.assert_dots_dark(img, H, name)  # no markers anywhere, yet every layout dot lands on a real dot

    def test_edge_only_autopage_holds_then_gives_up_then_recovers(self):
        import time
        import pagefind
        img, _ = paper_scene()
        desk_only = paper_scene(quad=((0, 0), (1, 0), (1, 1), (0, 1)))[0]  # no paper in view
        ap = pagefind.AutoPage(*page.A4_MM, origin=self.ORIGIN, track=False, refresh_seconds=0, hold_seconds=0.4)
        self.assertIsNone(ap.homography(desk_only))
        self.assertIn("looking for the page edges", ap.status)
        H1 = ap.homography(img)
        self.assertIsNotNone(H1)
        self.assertEqual(ap.status, "page edges found")
        held = ap.homography(desk_only)  # the paper vanished (hand over it): keep the last position for a while
        np.testing.assert_allclose(held, H1)
        self.assertIn("holding the last position", ap.status)
        time.sleep(0.5)
        self.assertIsNone(ap.homography(desk_only))  # ...but not forever
        self.assertIsNotNone(ap.homography(img))  # and it comes back by itself
        ap.unlock()
        self.assertIsNone(ap.last_H)

    def test_edge_only_autopage_follows_a_moved_sheet(self):
        import pagefind
        img1, T1 = paper_scene()
        img2, T2 = paper_scene(quad=((380, 100), (990, 110), (1000, 890), (350, 900)))
        ap = pagefind.AutoPage(*page.A4_MM, origin=self.ORIGIN, track=False, refresh_seconds=0)
        self.assert_dots_dark(img1, ap.homography(img1), "alphabet")
        self.assert_dots_dark(img2, ap.homography(img2), "alphabet")  # the sheet moved; no tracker to lose

    def test_paper_flag_builds_the_marker_free_page_source(self):
        import pagefind
        import tracker
        from detect import page_source_from_args

        def args(**kw):
            base = dict(paper=False, auto_page=None, calib=None, markers_only=False)
            base.update(kw)
            return argparse.Namespace(**base)

        plain = page_source_from_args(args(paper=True, markers_only=True))
        self.assertIsInstance(plain, pagefind.AutoPage)
        self.assertEqual((plain.size_mm, plain.origin, plain.track), (page.A4_MM, self.ORIGIN, False))
        # by default the paper finder becomes the fallback behind the markers
        combined = page_source_from_args(args(paper=True))
        self.assertIsInstance(combined, tracker.RobustPage)
        self.assertIsInstance(combined.paper_fallback, pagefind.AutoPage)
        # markers: robust by default, plain (None -> the caller uses the markers directly) with --markers-only
        self.assertIsInstance(page_source_from_args(args()), tracker.RobustPage)
        self.assertIsNone(page_source_from_args(args(markers_only=True)))


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class PaperCameraLoopTests(unittest.TestCase):
    def test_quiz_without_any_markers(self):
        import os
        import tempfile
        import time
        import pagefind
        img, T = paper_scene("alphabet")
        path = os.path.join(tempfile.mkdtemp(), "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 960))
        for _ in range(60):
            vw.write(img)
        vw.release()
        sh = sheets.get_sheet("alphabet")
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, sh.cells, finger=lambda: None, questions=1, rng=random.Random(5), names=sh.names)
        callbacks, step = [], [0]
        real = {k: getattr(cv2, k) for k in ("imshow", "waitKey", "namedWindow", "setMouseCallback", "destroyAllWindows")}
        cv2.imshow = cv2.namedWindow = cv2.destroyAllWindows = lambda *a, **k: None
        cv2.setMouseCallback = lambda name, cb: callbacks.append(cb)

        def key(ms):
            step[0] += 1
            time.sleep(0.2)
            if step[0] == 3:
                return ord("s")
            if step[0] == 8:
                c = s._target_cell(s.items[0])
                p = T @ np.array([(make_sheet.ORIGIN[0] + c["x"]) * make_sheet.MM, (make_sheet.ORIGIN[1] + c["y"]) * make_sheet.MM, 1.0])
                callbacks[0](cv2.EVENT_LBUTTONDOWN, int(p[0] / p[2]), int(p[1] / p[2]))
                return ord("f")
            return -1

        cv2.waitKey = key
        src = pagefind.AutoPage(*page.A4_MM, origin=(-page.SHEET_ORIGIN_MM[0], -page.SHEET_ORIGIN_MM[1]), track=False,
                                refresh_seconds=0)
        try:
            tutor.run_camera(s, tutor.open_camera(path), src, sh.cells, labels={k: v.short for k, v in sh.names.items()})
        finally:
            for k, fn in real.items():
                setattr(cv2, k, fn)
        self.assertTrue(s.finished.is_set(), voice.said)
        self.assertTrue(any(t.startswith("Correct!") for t in voice.said), voice.said)
        self.assertEqual(voice.debriefs[-1], (1.0, []))


def session_for(sheet_name, mode="letters", **kw):
    sh = sheets.get_sheet(sheet_name)
    voice = FakeVoice()
    s = tutor.TutorSession(voice, WC, sh.cells, finger=lambda: None, mode=mode, rng=random.Random(2), names=sh.names, **kw)
    return s, voice, sh


def at(s, sh, key, which=0):
    cells = [c for c in sh.cells if sh.symbol(c).key == key]
    c = cells[which]
    s.finger = lambda: (c["x"], c["y"])


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class TutorOnSheetsTests(unittest.TestCase):
    def test_number_quiz(self):
        s, v, sh = session_for("numbers", questions=3)
        s.items = ["3", "#", "."]
        s.state = "asking"
        s._ask()
        self.assertEqual(v.said[-1], "Find the number 3.")
        at(s, sh, "#")
        s.on_found_it()
        self.assertIn("That's the number sign, not the number 3", v.said[-1])
        s.on_hint()
        self.assertEqual(v.said[-1], "The number 3 has 2 dots: 1, 4.")
        at(s, sh, "3")
        s.on_found_it()
        self.assertEqual(v.said[-2], "Correct! That's the number 3.")
        self.assertEqual(v.said[-1], "Find the number sign.")
        at(s, sh, "#", which=5)  # any of the ten number signs counts
        s.on_found_it()
        self.assertEqual(v.said[-1], "Find the period.")
        s.on_hint()
        self.assertEqual(v.said[-1], "The period has 3 dots: 2, 5, 6.")
        at(s, sh, "^")
        s.on_found_it()
        self.assertIn("That's the capital sign, not the period", v.said[-1])
        s.on_next()
        self.assertTrue(s.finished.is_set())
        acc, missed = v.debriefs[-1]
        self.assertAlmostEqual(acc, 2 / 3)
        self.assertEqual(missed, [".", "3"])  # short labels, most slips first (the period slipped twice, the 3 once)

    def test_lookalike_duplicates_are_all_accepted_and_hint_says_so(self):
        s, v, sh = session_for("lookalikes")
        s.items = ["d"]
        s.state = "asking"
        s._ask()
        self.assertEqual(v.said[-1], "Find the letter D.")
        s.on_hint()
        s.on_hint()
        self.assertIn("One of them is in row", v.said[-1])
        at(s, sh, "f")  # the mirror image: the 2-dot confusion
        s.on_found_it()
        self.assertIn("not the letter D. Very close, it differs by 2 dots.", v.said[-1])
        at(s, sh, "d", which=1)  # the second copy, in the mirror-pair row
        s.on_found_it()
        self.assertIn("Correct!", " ".join(v.said))

    def test_quiz_only_asks_what_is_on_the_sheet(self):
        for name in sheets.SHEET_NAMES:
            s, v, sh = session_for(name, questions=50)
            s.on_start()
            self.assertEqual(set(s.items), {sh.symbol(c).key for c in sh.cells}, name)


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class WordsSheetTutorTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(mode="letters", sheet=None, detect=False, words=[], questions=5)
        base.update(kw)
        return argparse.Namespace(**base), argparse.ArgumentParser()

    def test_setup_defaults(self):
        a, ap = self.args()
        st = tutor.setup_from_args(a, ap)
        self.assertEqual((len(st.cells), st.layout_scan), (26, False))  # letters mode: alphabet by default
        a, ap = self.args(mode="read")
        st = tutor.setup_from_args(a, ap)
        self.assertEqual((st.cells, st.layout_scan), ([], False))  # real page: use the detector

    def test_word_quiz_on_the_words_sheet_uses_its_words_and_layout(self):
        a, ap = self.args(mode="word-quiz", sheet="words")
        st = tutor.setup_from_args(a, ap)
        self.assertTrue(st.layout_scan)
        self.assertEqual(len(st.words), 12)
        self.assertIn("cat", st.words)
        a, ap = self.args(mode="word-quiz", sheet="words", detect=True)
        self.assertFalse(tutor.setup_from_args(a, ap).layout_scan)

    def test_word_quiz_needs_words_without_the_words_sheet(self):
        a, ap = self.args(mode="word-quiz", sheet="numbers")
        with self.assertRaises(SystemExit):
            tutor.setup_from_args(a, ap)

    def test_read_and_quiz_on_the_printed_layout(self):
        sh = sheets.get_sheet("words")
        voice = FakeVoice()
        cat = next(c for c in sh.cells if c["row"] == 0 and c["col"] == 0)
        dog = next(c for c in sh.cells if c["row"] == 0 and c["col"] == 4)
        pos = {"c": cat}
        s = tutor.TutorSession(voice, WC, sh.cells, finger=lambda: (pos["c"]["x"], pos["c"]["y"]), scan=lambda: sh.cells,
                               mode="read", names=sh.names)
        s.on_start()
        s.on_found_it()
        self.assertEqual(voice.said[-1], "The word is cat.")
        pos["c"] = dog
        s.on_found_it()
        self.assertEqual(voice.said[-1], "The word is dog.")
        q = tutor.TutorSession(voice, WC, sh.cells, finger=lambda: (pos["c"]["x"], pos["c"]["y"]), scan=lambda: sh.cells,
                               mode="word-quiz", words=("dog",), questions=1, names=sh.names)
        q.on_start()
        q.on_found_it()
        self.assertIn("Correct! The word is dog.", voice.said)


if __name__ == "__main__":
    unittest.main()
