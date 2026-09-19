"""Tests for the embossed-braille reading path: enhancement, the auto choice, and quality warnings.

The accuracy test renders a simulated photo of real-size embossed braille (tools/sim_embossed.py) with known ground
truth, so the whole pipeline (find the page -> read it) is scored end to end rather than eyeballed.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "tools"))

import sim_embossed as sim
from detect import WEIGHTS, cells_from_boxes, detect_boxes, enhance_paper, reading_quality, scan_page
from pagefind import find_page_homography
from score import score

PAGE_MM = (170.0, 160.0)


def photo(contrast=25, width=900, blur=1.2, noise=2.0, seed=1, lines=10, cells=20):
    """(camera image, truth) for a simulated embossed page lying on a dark desk."""
    rows = sim.random_text_rows(lines, cells, seed=seed)
    flat, truth = sim.render_relief(rows, 2.5, 6.0, 10.0, 8, PAGE_MM, contrast=contrast, seed=seed)
    inset = (1280 - width) // 2
    quad = [[inset, 40], [inset + width, 60], [inset + width - 10, 40 + int(width * 160 / 170)],
            [inset + 10, 40 + int(width * 158 / 170)]]
    img, _ = sim.camera_view(flat, out_size=(1280, 960), quad=quad, blur=blur, noise=noise, seed=seed)
    return img, truth


class EnhanceTests(unittest.TestCase):
    def test_enhance_keeps_shape_and_survives_odd_input(self):
        img, _ = photo()
        out = enhance_paper(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        for odd in (np.zeros((40, 40, 3), np.uint8), np.full((40, 40, 3), 255, np.uint8),
                    np.full((40, 40), 128, np.uint8), np.zeros((5, 5, 3), np.uint8)):
            self.assertIsNotNone(enhance_paper(odd))

    def test_enhance_raises_contrast_on_a_faint_page(self):
        img, _ = photo(contrast=10)
        page = (slice(200, 700), slice(300, 900))
        before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[page].std()
        after = cv2.cvtColor(enhance_paper(img), cv2.COLOR_BGR2GRAY)[page].std()
        self.assertGreater(after, before * 1.5)


@unittest.skipUnless(WEIGHTS.exists(), "detector weights missing")
class ReadingTests(unittest.TestCase):
    def read(self, img, **kw):
        H, why = find_page_homography(img, *PAGE_MM)
        self.assertIsNotNone(H, why)
        return H

    def test_reads_a_simulated_embossed_page_accurately(self):
        """Good conditions: this guards the whole pipeline, not just the model."""
        img, truth = photo(contrast=30, width=900)
        s = score(scan_page(img, self.read(img)), truth)
        print(f"\n  simulated embossed page, good light: found {s['recall']:.0%}, exact dots {s['exact_rate']:.0%}")
        self.assertGreater(s["recall"], 0.90)
        self.assertGreater(s["exact_rate"], 0.80)
        self.assertLess(s["false_cells"], 0.1 * s["truth"])

    def test_auto_enhance_is_no_worse_than_plain_when_the_page_is_faint(self):
        img, truth = photo(contrast=12, width=760)
        H = self.read(img)
        plain = score(cells_from_boxes(detect_boxes(img, enhance="off"), H), truth)["exact_rate"]
        auto = score(cells_from_boxes(detect_boxes(img, enhance="auto"), H), truth)["exact_rate"]
        print(f"  faint page: plain {plain:.0%} -> auto {auto:.0%}")
        self.assertGreaterEqual(auto, plain)

    def test_the_three_enhance_modes_all_work(self):
        img, truth = photo()
        H = self.read(img)
        for mode in ("off", "on", "auto"):
            cells = cells_from_boxes(detect_boxes(img, enhance=mode), H)
            self.assertGreater(len(cells), 50, mode)


class QualityTests(unittest.TestCase):
    @staticmethod
    def boxes(n=30, conf=0.7, w=25):
        return [(i * 40.0, 0.0, i * 40.0 + w, 30.0, "100000", conf) for i in range(n)]

    def test_messages(self):
        ok, note = reading_quality(self.boxes())
        self.assertTrue(ok)
        self.assertEqual(note, "")
        ok, note = reading_quality([])
        self.assertFalse(ok)
        self.assertIn("nothing detected", note)
        ok, note = reading_quality(self.boxes(w=8))
        self.assertFalse(ok)
        self.assertIn("closer", note)
        ok, note = reading_quality(self.boxes(conf=0.2))
        self.assertFalse(ok)
        self.assertIn("light", note)
        ok, note = reading_quality(self.boxes(conf=0.4))
        self.assertTrue(ok)
        self.assertIn("shaky", note)



class LiveReadingOverlayTests(unittest.TestCase):
    """The reading must be drawable straight onto the camera image, with no page registration."""

    @staticmethod
    def boxes():
        # (x1, y1, x2, y2, label, confidence): an 'a' (dot 1), a 'd' (dots 1,4,5) and a non-letter (all six dots)
        return [(100, 100, 130, 150, "100000", 0.9), (200, 100, 230, 150, "100110", 0.8), (300, 100, 330, 150, "111111", 0.7)]

    def test_draw_detections_marks_every_cell_without_a_homography(self):
        from detect import draw_detections
        img = np.full((300, 500, 3), 200, np.uint8)
        before = img.copy()
        n = draw_detections(img, self.boxes())
        self.assertEqual(n, 3)
        self.assertFalse(np.array_equal(img, before))
        for x1, y1, x2, y2 in ((100, 100, 130, 150), (200, 100, 230, 150), (300, 100, 330, 150)):
            self.assertFalse(np.array_equal(img[y1 - 40:y2 + 5, x1 - 5:x2 + 5], before[y1 - 40:y2 + 5, x1 - 5:x2 + 5]))

    def test_letters_flag_controls_the_text(self):
        from detect import draw_detections
        with_letters, without = (np.full((300, 500, 3), 200, np.uint8) for _ in range(2))
        draw_detections(with_letters, self.boxes(), letters=True)
        draw_detections(without, self.boxes(), letters=False)
        above = (slice(55, 94), slice(90, 340))  # where the letters go, clear of the boxes below
        self.assertFalse(np.array_equal(with_letters[above], without[above]))
        self.assertTrue(np.array_equal(np.full((39, 250, 3), 200, np.uint8), without[above]))

    def test_empty_detections_draw_nothing(self):
        from detect import draw_detections
        img = np.full((100, 100, 3), 7, np.uint8)
        self.assertEqual(draw_detections(img, []), 0)
        self.assertTrue((img == 7).all())


class CameraFeedReadingTests(unittest.TestCase):
    """CameraFeed shows the live reading when the page is not registered (and only then, unless asked)."""

    class FakeDetector:
        def __init__(self, boxes):
            self.boxes, self.submitted = boxes, 0

        def submit(self, frame, H=None):
            self.submitted += 1

    def feed(self, registered, always=False):
        import tutor
        det = self.FakeDetector(LiveReadingOverlayTests.boxes())
        feed = tutor.CameraFeed(None, None, None, detector=det, always_reading=always)
        feed.frame = np.full((300, 500, 3), 200, np.uint8)
        feed.H = np.eye(3) if registered else None
        return feed, det

    def test_shown_when_the_page_is_not_registered(self):
        feed, det = self.feed(registered=False)
        self.assertFalse(np.array_equal(feed.render(), feed.frame))

    def test_hidden_when_registered_unless_always(self):
        feed, _ = self.feed(registered=True)
        self.assertTrue(np.array_equal(feed.render(), feed.frame))
        feed, _ = self.feed(registered=True, always=True)
        self.assertFalse(np.array_equal(feed.render(), feed.frame))

    def test_update_hands_frames_to_the_detector(self):
        import tutor
        det = self.FakeDetector([])
        feed = tutor.CameraFeed(None, None, None, detector=det)
        feed.update(np.full((300, 500, 3), 200, np.uint8))
        self.assertEqual(det.submitted, 1)

    def test_no_detector_means_no_change(self):
        import tutor
        feed = tutor.CameraFeed(None, None, None)
        feed.frame, feed.H = np.full((100, 100, 3), 9, np.uint8), None
        self.assertTrue(np.array_equal(feed.render(), feed.frame))



class AutoSizeTests(unittest.TestCase):
    """The model input size follows the braille in view: too-small cells at 640 were the main loss on 1080p webcam frames."""

    @staticmethod
    def boxes(width, n=30):
        return [(i * 50.0, 0.0, i * 50.0 + width, 40.0, "100000", 0.8) for i in range(n)]

    def test_choose_imgsz_scales_with_cell_size(self):
        from detect import choose_imgsz
        self.assertEqual(choose_imgsz(self.boxes(30), 1920), 640)   # cells already big: 640 is plenty
        self.assertEqual(choose_imgsz(self.boxes(17), 1920), 960)   # a third of the frame budget lost: size up
        self.assertEqual(choose_imgsz(self.boxes(12), 1920), 1280)  # small cells: the cap
        self.assertEqual(choose_imgsz(self.boxes(4), 1920), 1280)   # never beyond the cap
        sizes = [choose_imgsz(self.boxes(w), 1920) for w in (30, 22, 17, 12)]
        self.assertEqual(sizes, sorted(sizes))                       # smaller cells never ask for a smaller input

    def test_choose_imgsz_keeps_the_current_size_when_it_has_too_little_to_go_on(self):
        from detect import choose_imgsz
        self.assertEqual(choose_imgsz([], 1920, current=800), 800)
        self.assertEqual(choose_imgsz(self.boxes(20, n=3), 1920, current=800), 800)

    def test_sizes_are_multiples_of_32_within_bounds(self):
        from detect import choose_imgsz
        for w in range(6, 60, 3):
            s = choose_imgsz(self.boxes(w), 1920)
            self.assertEqual(s % 32, 0)
            self.assertTrue(640 <= s <= 1280)

    @unittest.skipUnless(WEIGHTS.exists(), "detector weights missing")
    def test_auto_beats_fixed_640_when_the_page_is_small_in_a_1080p_frame(self):
        import detect
        rows = sim.random_text_rows(12, 24, seed=3)
        flat, truth = sim.render_relief(rows, 2.5, 6.0, 10.0, 10, (160, 140), contrast=22)
        width, ph = 620, int(620 * 140 / 160)
        x0, y0 = (1920 - width) // 2, (1080 - ph) // 2
        quad = [[x0, y0], [x0 + width, y0 + 8], [x0 + width - 8, y0 + ph], [x0 + 6, y0 + ph - 6]]
        img, _ = sim.camera_view(flat, out_size=(1920, 1080), quad=quad, blur=1.4, noise=2.0)
        H, why = find_page_homography(img, 160, 140)
        self.assertIsNotNone(H, why)
        detect._SIZE_CACHE.clear()
        fixed = score(cells_from_boxes(detect_boxes(img, imgsz=640, enhance="off"), H), truth)["exact_rate"]
        auto = score(cells_from_boxes(detect_boxes(img, imgsz="auto", enhance="off"), H), truth)["exact_rate"]
        print(f"\n  1080p frame, small page: fixed 640 {fixed:.0%} -> auto {auto:.0%} (input {detect._SIZE_CACHE[img.shape[:2]][1]})")
        self.assertGreater(auto, fixed + 0.10)

    @unittest.skipUnless(WEIGHTS.exists(), "detector weights missing")
    def test_probe_runs_on_the_first_call_then_only_now_and_then(self):
        import detect
        calls = []
        real = detect._detect_pixels
        detect._detect_pixels = lambda frame, conf, size: (calls.append(size) or [])
        try:
            detect._SIZE_CACHE.clear()
            frame = np.zeros((300, 400, 3), np.uint8)
            for _ in range(13):
                detect.detect_boxes(frame, imgsz="auto", enhance="off")
        finally:
            detect._detect_pixels = real
            detect._SIZE_CACHE.clear()
        # call 1 probes (640, then 1024 because nothing was found) and reads; calls 2-12 only read; call 13 probes again
        self.assertEqual(len(calls), (2 + 1) + 11 + (2 + 1))



class WordSplittingTests(unittest.TestCase):
    """A line of one-cell words ("the child can go") must still split, which judging each row by its own median cannot do."""

    @staticmethod
    def line(labels_by_word, row=0, pitch=6.0):
        from detect import _make_cell
        cells, col, x = [], 0, 10.0
        for labels in labels_by_word:
            for lab in labels:
                cells.append({**_make_cell(x, 10.0 + row * 12, 3.2, 5.0, lab, 1.0, row, col),
                              "dots": frozenset(i + 1 for i, ch in enumerate(lab) if ch == "1")})
                x += pitch
                col += 1
            x += pitch  # the blank cell between words
            col += 1
        return cells

    def test_single_cell_words_split_using_the_page_wide_spacing(self):
        import reader
        one_cell = self.line([["011101"], ["100000"], ["100100"]], row=0)      # 2346 | 1 | 14: three one-cell words
        multi = self.line([["100100", "100000", "011110"], ["110000", "100010"]], row=1)  # c a t | b e: sets the normal pitch
        cells = one_cell + multi
        pitch = reader.typical_pitch(cells)
        self.assertAlmostEqual(pitch, 6.0, delta=0.5)
        words = reader.split_words([c for c in cells if c["row"] == 0], typical=pitch)
        self.assertEqual(len(words), 3)
        self.assertEqual(len(reader.split_words([c for c in cells if c["row"] == 1], typical=pitch)), 2)

    def test_without_the_page_wide_value_a_row_of_single_cell_words_does_not_split(self):
        """The failure this guards against: every gap is a word gap, so the row median mistakes them for normal."""
        import reader
        row = self.line([["011101"], ["100000"], ["100100"]])
        self.assertEqual(len(reader.split_words(row)), 1)

    def test_pitch_is_capped_when_the_page_has_no_multi_cell_words(self):
        import reader
        cells = self.line([["011101"], ["100000"], ["100100"], ["110000"]])  # every gap is 12 mm; a box is 3.2 mm wide
        self.assertLess(reader.typical_pitch(cells), 8.0)

    def test_decode_lines_and_word_at_use_the_same_splitting(self):
        import reader
        cells = self.line([["011101"], ["100000"], ["100100"]]) + self.line([["100100", "100000", "011110"]], row=1)
        y = 10.0
        self.assertEqual(reader.word_at(cells, 10.0, y), "?")           # 2346 has no plain-letter reading
        self.assertEqual(reader.word_at(cells, 10.0, y, decode=True, is_word={"the", "a", "can"}.__contains__), "the")
        self.assertEqual(reader.decode_lines(cells, {"the", "a", "can", "cat"}.__contains__), ["the a can", "cat"])


@unittest.skipUnless(WEIGHTS.exists(), "detector weights missing")
class ContractedEnglishEndToEndTests(unittest.TestCase):
    """A simulated embossed page of real contracted English through the whole pipeline: find the page, read, decode."""

    def test_decoding_contractions_turns_nonsense_into_english(self):
        import contracted_page as cp
        import detect
        import reader
        flat, truth, words, (w_mm, h_mm) = cp.build(contrast=28)
        width = 1000
        ph = int(width * h_mm / w_mm)
        img, _ = sim.camera_view(flat, out_size=(1280, 960), quad=[[140, 100], [1140, 110], [1132, 100 + ph], [146, 94 + ph]],
                                 blur=1.0, noise=2.0)
        H, why = find_page_homography(img, w_mm, h_mm)
        self.assertIsNotNone(H, why)
        detect._SIZE_CACHE.clear()
        cells = cells_from_boxes(detect_boxes(img, enhance="off"), H)
        self.assertGreater(len(cells), 0.9 * len(truth))
        plain, total = cp.words_right(reader.read_lines(cells), words)
        decoded, _ = cp.words_right(reader.decode_lines(cells), words)
        print(f"\n  contracted English page: {plain}/{total} words right as plain letters -> {decoded}/{total} decoded")
        self.assertLess(plain, 0.15 * total)         # read letter by letter, contracted braille is nonsense
        self.assertGreater(decoded, 0.70 * total)    # decoded, most of it is English again


if __name__ == "__main__":
    unittest.main()
