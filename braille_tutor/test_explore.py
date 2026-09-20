"""Tests for explore mode (speak what the camera detects under the finger), the .env key loader and the voice self-check.

Time is passed in by the tests (no sleeping) except in the one thread test."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sheets
import tutor
from detect import _make_cell, dots_to_label
from test_sheetread import POKE_FILES, PhotoCase, wrong_cells
from test_tutor import WC, FakeVoice

SHEET = sheets.get_sheet("alphabet")


def explore_session(cells=None, finger=None, scan=None, names="sheet"):
    voice = FakeVoice()
    cells = SHEET.cells if cells is None else cells
    s = tutor.TutorSession(voice, WC, cells, finger=finger or (lambda: None), scan=scan, mode="explore",
                           names=SHEET.names if names == "sheet" else names)
    s.state, s._ex = "exploring", s._new_explore_state()
    return s, voice


def centre(letter):
    c = next(c for c in SHEET.cells if SHEET.names[(c["row"], c["col"])].key == letter)
    return c["x"], c["y"]


class Finger:
    def __init__(self, pos=None):
        self.pos = pos

    def __call__(self):
        return self.pos


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class DescribeTests(unittest.TestCase):
    def cell(self, dots, row=0, col=0):
        return _make_cell(10.0, 10.0, 9.2, 15.2, dots_to_label(frozenset(dots)), 1.0, row, col)

    def test_full_and_short_descriptions(self):
        s, _ = explore_session()
        d = SHEET.cells[3]  # the letter d
        self.assertEqual(s.describe(d), "The letter D. Dots 1, 4 and 5: top-left, top-right and middle-right.")
        self.assertEqual(s.describe(d, full=False), "The letter D. Dots 1, 4 and 5.")
        self.assertEqual(s.describe(SHEET.cells[0]), "The letter A. Dot 1: top-left.")
        self.assertEqual(s.describe(SHEET.cells[1]), "The letter B. Dots 1 and 2: top-left and middle-left.")

    def test_what_is_said_keeps_the_sheet_name_when_a_finger_hides_a_dot(self):
        """The sheet says D (1 4 5) but the camera sees only 1 and 4: it must still say D, not C."""
        s, _ = explore_session()
        seen = {**SHEET.cells[3], "dots": frozenset({1, 4})}
        self.assertEqual(s.describe(seen, full=False), "The letter D. Dots 1, 4 and 5.")

    def test_no_dots_and_unknown_patterns(self):
        s, _ = explore_session(names=None)
        self.assertIn("no raised dots", s.describe(self.cell([])))
        text = s.describe(self.cell([1, 2, 3, 4, 5, 6]), full=False)  # not a letter
        self.assertEqual(text, "A cell I don't recognise. Dots 1, 2, 3, 4, 5 and 6.")

    def test_signs_on_the_numbers_sheet_keep_their_names_when_the_camera_agrees(self):
        sh = sheets.get_sheet("numbers")
        s = tutor.TutorSession(FakeVoice(), WC, sh.cells, lambda: None, mode="explore", names=sh.names)
        num = next(c for c in sh.cells if sh.names[(c["row"], c["col"])].key == "#")
        self.assertEqual(s.describe(num, full=False), "The number sign. Dots 3, 4, 5 and 6.")


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class DwellTests(unittest.TestCase):
    def test_speaks_once_when_the_finger_settles_then_stays_quiet(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        for t in (0.0, 0.3, 0.6):
            s.explore_tick(t)
        self.assertEqual(v.said, [], "still deciding whether it is resting")
        s.explore_tick(0.8)
        self.assertEqual(v.said, ["The letter A. Dot 1: top-left."])
        for t in (1.0, 2.0, 5.0):
            s.explore_tick(t)
        self.assertEqual(len(v.said), 1, "no repeating while the finger stays")

    def test_small_movements_inside_the_cell_do_not_repeat_it(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        s.explore_tick(0.0)
        s.explore_tick(0.8)
        x, y = centre("a")
        for i, t in enumerate((1.0, 1.4, 1.8, 2.2)):
            finger.pos = (x + (3 if i % 2 else -3), y + (4 if i % 2 else -4))  # wobble
            s.explore_tick(t)
        self.assertEqual(len(v.said), 1)

    def test_a_moving_finger_says_nothing_until_it_rests(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        x, y = centre("a")
        for i in range(30):  # sweeping along the row: 6 mm every 0.1 s
            finger.pos = (x + 6 * i * 0.3, y)
            s.explore_tick(0.1 * i)
        self.assertEqual(v.said, [])

    def test_next_cell_is_announced_and_a_revisit_is_brief(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        s.explore_tick(0.0); s.explore_tick(0.8)
        finger.pos = centre("b")
        s.explore_tick(1.0); s.explore_tick(1.8)
        finger.pos = centre("a")
        s.explore_tick(2.0); s.explore_tick(2.8)
        self.assertEqual(v.said, ["The letter A. Dot 1: top-left.",
                                  "The letter B. Dots 1 and 2: top-left and middle-left.",
                                  "The letter A. Dot 1."])  # heard before: just the name and the dots

    def test_lost_finger_is_reported_once_and_the_cell_is_read_again_on_return(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        s.explore_tick(0.0); s.explore_tick(0.8)
        finger.pos = None
        for t in (1.0, 2.0, 3.0, 4.0, 9.0):
            s.explore_tick(t)
        self.assertEqual(v.said[1:], ["I can't see your finger."])
        finger.pos = centre("a")
        s.explore_tick(10.0); s.explore_tick(10.8)
        self.assertEqual(v.said[-1], "The letter A. Dot 1.")

    def test_never_saw_a_finger_stays_silent(self):
        s, v = explore_session(finger=Finger(None))
        for t in range(0, 20):
            s.explore_tick(float(t))
        self.assertEqual(v.said, [])

    def test_resting_on_blank_paper_is_said_once(self):
        finger = Finger((5.0, 5.0))  # nowhere near a cell
        s, v = explore_session(finger=finger)
        for t in (0.0, 0.8, 2.0):
            s.explore_tick(t)
        self.assertEqual(v.said, [])
        s.explore_tick(2.6)
        s.explore_tick(4.0)
        self.assertEqual(v.said, ["I don't see any braille there."])

    def test_it_speaks_the_printed_letter_when_a_dot_is_hidden(self):
        """A finger covering a poke must not rename the cell: explore says the sheet's letter, not the live misread."""
        seen = [{**c, "dots": frozenset({1, 4}) if c["row"] == 0 and c["col"] == 3 else c["dots"]} for c in SHEET.cells]
        finger = Finger(centre("d"))
        s, v = explore_session(finger=finger, scan=lambda: seen)
        s.explore_tick(0.0); s.explore_tick(0.8)
        self.assertTrue(v.said[0].startswith("The letter D. Dots 1, 4 and 5"), v.said)

    def test_falls_back_to_the_layout_if_the_scan_finds_nothing(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger, scan=lambda: [])
        s.explore_tick(0.0); s.explore_tick(0.8)
        self.assertEqual(v.said, ["The letter A. Dot 1: top-left."])


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class CommandTests(unittest.TestCase):
    def test_voice_commands_in_explore_mode(self):
        finger = Finger(centre("d"))
        s, v = explore_session(finger=finger)
        s.state = "idle"
        s.on_start()
        self.assertEqual(s.state, "exploring")
        self.assertIn("Explore mode", v.said[0])
        s.on_found_it()  # don't wait for the dwell
        self.assertEqual(v.said[-1], "The letter D. Dots 1, 4 and 5: top-left, top-right and middle-right.")
        s.on_hint()  # always in full
        self.assertEqual(v.said[-1], "The letter D. Dots 1, 4 and 5: top-left, top-right and middle-right.")
        s.on_found_it()  # heard before: brief
        self.assertEqual(v.said[-1], "The letter D. Dots 1, 4 and 5.")
        s.on_repeat()
        self.assertEqual(v.said[-1], "The letter D. Dots 1, 4 and 5.")
        s.on_next()
        self.assertIn("move your finger", v.said[-1])
        finger.pos = None
        s.on_found_it()
        self.assertIn("can't see your finger", v.said[-1])
        s.on_stop()
        self.assertEqual((s.state, s.finished.is_set()), ("done", True))

    def test_the_real_loop_speaks_when_a_finger_rests_and_stops_on_stop(self):
        finger = Finger(centre("c"))
        s, v = explore_session(finger=finger)
        s.state = "idle"
        s.on_start()
        deadline = time.time() + 5
        while time.time() < deadline and not any("The letter C" in t for t in v.said):
            time.sleep(0.05)
        self.assertTrue(any("The letter C. Dots 1 and 4" in t for t in v.said), v.said)
        threads = [t for t in threading.enumerate() if t.name == "explore"]
        self.assertEqual(len(threads), 1)
        s.on_start()  # starting again must not start a second loop
        self.assertEqual(len([t for t in threading.enumerate() if t.name == "explore"]), 1)
        s.on_stop()
        time.sleep(0.4)
        self.assertEqual([t for t in threading.enumerate() if t.name == "explore" and t.is_alive()], [])

    def test_speech_never_overlaps(self):
        class SlowVoice(FakeVoice):
            def __init__(self):
                super().__init__()
                self.now = self.peak = 0
                self.lock = threading.Lock()

            def speak(self, text, mode="normal"):
                with self.lock:
                    self.now += 1
                    self.peak = max(self.peak, self.now)
                time.sleep(0.03)
                with self.lock:
                    self.now -= 1

        voice = SlowVoice()
        s = tutor.TutorSession(voice, WC, SHEET.cells, lambda: None, mode="explore")
        threads = [threading.Thread(target=s.say, args=(f"line {i}",)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(voice.peak, 1)


@unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class SheetObservationTests(PhotoCase):
    def test_feed_scan_reports_what_the_camera_sees_on_the_sheet(self):
        cam, H = self.photo_of_file("alphabet", drop=None)
        feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEET.cells)
        feed.frame, feed.H = cam, H
        seen = feed.scan()
        self.assertEqual(len(seen), 26)
        self.assertEqual(wrong_cells(seen, SHEET.cells), [])

    def test_a_missing_poke_is_spoken_as_what_is_actually_there(self):
        """End to end: photo of the sheet with one dot not poked -> the finger on that cell hears the detected pattern."""
        import poke_files as pf
        import make_sheet
        from page import SHEET_ORIGIN_MM
        target = SHEET.cells[3]  # d: 1, 4, 5
        dots = {(x + SHEET_ORIGIN_MM[0], y + SHEET_ORIGIN_MM[1]) for x, y in make_sheet.dot_points(target)}
        pts = pf.after_flip(pf.guide_dots(str(POKE_FILES["alphabet"])))
        gone = [p for p in pts if min(np.hypot(p[0] - dx, p[1] - dy) for dx, dy in dots) < 0.5][:1]
        cam, H = self.photo_of_file("alphabet", drop=gone)
        feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEET.cells)
        feed.frame, feed.H = cam, H
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, SHEET.cells, lambda: (target["x"], target["y"]), feed.scan, "explore", names=SHEET.names)
        s.state, s._ex = "exploring", s._new_explore_state()
        s.explore_tick(0.0); s.explore_tick(0.8)
        self.assertEqual(len(voice.said), 1)
        self.assertIn("letter D", voice.said[0])  # a missing poke must not rename the cell


@unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ReadingBoxesTests(PhotoCase):
    """The video shows what the camera reads: a box per cell, the detected dots and the letter, green once locked in."""

    def run_feed(self, drop=None, scans=4):
        cam, H = self.photo_of_file("alphabet", drop=drop)
        feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEET.cells)
        feed.frame, feed.H = cam, H
        for _ in range(scans):  # the reader thread scans in the background: feed it and wait for each new scan
            before = feed.reader.snapshot()[2]
            feed.reader.submit(cam, H)
            deadline = time.time() + 20
            while time.time() < deadline and feed.reader.snapshot()[2] == before:
                time.sleep(0.02)
            feed._absorb_reading()
        return feed, cam

    def test_every_cell_gets_locked_and_drawn_green(self):
        feed, cam = self.run_feed()
        self.assertEqual(sum(1 for c in feed.stable if c["locked"]), 26)
        view = feed.render()
        self.assertEqual(view.shape, cam.shape)
        green = (view[..., 1] > 150) & (view[..., 0] < 60) & (view[..., 2] < 60)
        raw_green = (cam[..., 1] > 150) & (cam[..., 0] < 60) & (cam[..., 2] < 60)
        self.assertGreater(green.sum(), raw_green.sum() + 400, "boxes and status text were drawn")
        red = (view[..., 2] > 200) & (view[..., 1] < 60) & (view[..., 0] < 60)
        self.assertGreater(red.sum(), 150, "the detected dots are drawn in red")

    def test_what_is_spoken_is_the_same_reading_that_is_drawn(self):
        feed, _ = self.run_feed()
        self.assertEqual([c["label"] for c in feed.scan()], [c["label"] for c in feed.stable])

    def test_a_missing_poke_does_not_rename_the_printed_letter(self):
        import poke_files as pf
        import make_sheet
        from page import SHEET_ORIGIN_MM
        target = SHEET.cells[3]
        dots = {(x + SHEET_ORIGIN_MM[0], y + SHEET_ORIGIN_MM[1]) for x, y in make_sheet.dot_points(target)}
        pts = pf.after_flip(pf.guide_dots(str(POKE_FILES["alphabet"])))
        gone = [p for p in pts if min(np.hypot(p[0] - dx, p[1] - dy) for dx, dy in dots) < 0.5][:1]
        feed, _ = self.run_feed(drop=gone)
        self.assertEqual(sum(1 for c in feed.stable if c["locked"]), 26)
        held = next(c for c in feed.stable if (c["row"], c["col"]) == (target["row"], target["col"]))
        self.assertEqual(held["dots"], target["dots"], "D stays D even if a poke is hidden")
        import detect
        text, _ = detect.locked_check_line(feed.stable, SHEET.cells)
        self.assertTrue("26" in text and "locked" in text, text)

    def test_hidden_detections_and_no_page_draw_nothing_extra(self):
        feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEET.cells, show_reading=False)
        self.assertIsNone(feed.reader)
        frame = np.full((240, 320, 3), 90, np.uint8)
        feed.frame = frame
        self.assertTrue(np.array_equal(feed.render(), frame))
        feed2 = tutor.CameraFeed(None, None, None, observe_sheet=SHEET.cells)  # reader on but no page registered: says so, draws no boxes
        feed2.frame, feed2.message = frame, "PAGE NOT FOUND: test"
        view = feed2.render()
        self.assertTrue(feed2.stable, "demo sheets start prelocked from the printed layout")
        self.assertIsNone(feed2.H)
        self.assertFalse(np.array_equal(view, frame), "the status line is drawn")
        # Without a page lock, boxes are not drawn even though the layout is known.
        self.assertLess(np.count_nonzero(view != frame), view.size * 0.15)


class NextPageVoiceTests(unittest.TestCase):
    """"next page" must be heard as itself, never as plain "next" (which means something else)."""

    def heard(self, transcript):
        import voice_io
        return voice_io._match_command(voice_io._normalize_transcript(transcript))

    def test_phrases(self):
        import tutor
        tutor.load_teammate_modules(mock=True)
        for said in ("next page", "Next page.", "okay next page please", "new page", "another page", "let's go to the next page now"):
            self.assertEqual(self.heard(said), "next page", said)
        self.assertEqual(self.heard("next"), "next")
        self.assertEqual(self.heard("next question"), "next")
        self.assertEqual(self.heard("found it"), "found it")


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class NextPageCommandTests(unittest.TestCase):
    def test_it_forgets_the_finger_position_calls_the_hook_and_says_so(self):
        finger = Finger(centre("a"))
        s, v = explore_session(finger=finger)
        calls = []
        s.new_page = lambda: calls.append(1)
        s.explore_tick(0.0); s.explore_tick(0.8)
        self.assertEqual(len(v.said), 1)
        s.on_next_page()
        self.assertEqual(calls, [1])
        self.assertIn("next page", v.said[-1])
        s.explore_tick(1.0); s.explore_tick(1.8)  # same spot of the finger, but it is a new page: read again
        self.assertEqual(len(v.said), 3)
        self.assertEqual(s.state, "exploring")

    def test_wording_depends_on_whether_it_will_work_out_the_sheet(self):
        s, v = explore_session()
        s.on_next_page()
        self.assertEqual(v.said[-1], "Okay. Ready for the next page.")
        s.identifies_sheets = True
        s.on_next_page()
        self.assertIn("Show me the next page", v.said[-1])

    def test_a_quiz_in_progress_is_not_disturbed(self):
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, SHEET.cells, lambda: None, mode="letters", questions=3, names=SHEET.names)
        s.on_start()
        state, index, said = s.state, s.index, len(voice.said)
        hook = []
        s.new_page = lambda: hook.append(1)
        s.on_next_page()
        self.assertEqual((s.state, s.index, hook), (state, index, [1]))
        self.assertEqual(len(voice.said), said + 1)

    def test_set_sheet_switches_what_the_cells_mean(self):
        s, _ = explore_session()
        words = sheets.get_sheet("words")
        s.set_sheet(words)
        self.assertEqual((len(s.cells), s.names), (len(words.cells), words.names))
        self.assertEqual(s._ex["cell"], None)

    def test_the_command_is_registered_and_reachable_from_the_server_api(self):
        import tutor_server
        self.assertEqual(tutor_server.COMMANDS["next page"], "on_next_page")
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, SHEET.cells, lambda: None, mode="explore")
        s.attach()
        self.assertIn("next page", voice.commands)
        voice.commands["next page"]()
        self.assertEqual(len(voice.said), 1)


@unittest.skipUnless(all(p.exists() for p in POKE_FILES.values()), "poke files not generated")
@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class SheetIdentificationTests(PhotoCase):
    def all_cells(self):
        return {n: sheets.get_sheet(n).cells for n in sheets.SHEET_NAMES}

    def test_each_of_the_four_sheets_is_recognised_from_its_photo(self):
        import sheetread
        for name in sheets.SHEET_NAMES:
            cam, H = self.photo_of_file(name)
            found, scores = sheetread.identify_sheet(cam, H, self.all_cells())
            self.assertEqual(found, name, (name, scores))
            others = [v for k, v in scores.items() if k != name]
            self.assertGreaterEqual(scores[name] - max(others), 0.15)

    def test_a_blank_page_is_not_mistaken_for_a_sheet(self):
        import sheetread
        cam, H = self.photo_of_file("alphabet", drop=set(pf_points("alphabet")))  # nothing poked at all
        found, scores = sheetread.identify_sheet(cam, H, self.all_cells())
        self.assertIsNone(found, scores)

    def test_recognition_survives_the_registration_being_a_couple_of_mm_out(self):
        import sheetread
        from test_sheetread import misregister
        for name in ("words", "lookalikes"):
            cam, H = self.photo_of_file(name)
            found, _ = sheetread.identify_sheet(cam, misregister(2.5, 1.5) @ H, self.all_cells())
            self.assertEqual(found, name)

    def make_feed(self, first, later):
        """A feed reading `first`, then given "next page" and shown `later`. Returns (feed, session, voice, said_by_feed, cam, H)."""
        cam1, H1 = self.photo_of_file(first)
        feed = tutor.CameraFeed(None, None, None, observe_sheet=sheets.get_sheet(first).cells, show_reading=True,
                                known_sheets={n: sheets.get_sheet(n) for n in sheets.SHEET_NAMES}, sheet_name=first)
        voice = FakeVoice()
        session = tutor.TutorSession(voice, WC, sheets.get_sheet(first).cells, lambda: None, mode="explore", names=sheets.get_sheet(first).names)
        told = []
        tutor.wire_new_page(session, feed)
        feed.announce = told.append  # synchronous, for the test
        for _ in range(4):  # lock in the first page
            before = feed.reader.snapshot()[2]
            feed.frame, feed.H = cam1, H1
            feed.reader.submit(cam1, H1)
            deadline = time.time() + 20
            while time.time() < deadline and feed.reader.snapshot()[2] == before:
                time.sleep(0.02)
            feed._absorb_reading()
        return feed, session, voice, told, cam1, H1

    def test_next_page_forgets_the_old_page_and_switches_to_the_new_sheet(self):
        feed, session, voice, told, _, _ = self.make_feed("alphabet", "words")
        self.assertEqual(sum(1 for c in feed.stable if c["locked"]), 26)
        session.on_next_page()
        self.assertEqual(feed.stable, [], "the old page's locked cells are gone")
        self.assertIsNone(feed.H)
        self.assertGreater(feed.identify_until, 0)
        cam2, H2 = self.photo_of_file("words")  # the words sheet is put down
        feed.frame, feed.H = cam2, H2
        feed._identify(cam2, H2)
        self.assertEqual(feed.sheet_name, "words")
        self.assertEqual(told, ["This is the words sheet."])
        self.assertEqual(len(session.cells), len(sheets.get_sheet("words").cells), "the tutor now knows the words sheet's cells")
        self.assertEqual(feed.identify_until, 0)
        for _ in range(4):  # and the new page is read and locked
            before = feed.reader.snapshot()[2]
            feed.reader.submit(cam2, H2)
            deadline = time.time() + 20
            while time.time() < deadline and feed.reader.snapshot()[2] == before:
                time.sleep(0.02)
            feed._absorb_reading()
        self.assertEqual(sum(1 for c in feed.stable if c["locked"]), len(sheets.get_sheet("words").cells))

    def test_the_same_sheet_again_is_just_read_again(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "alphabet")
        session.on_next_page()
        feed._identify(cam, H)  # the OLD page is still in view: recognising it proves nothing yet
        self.assertEqual(told, [])
        self.assertGreater(feed.identify_until, 0, "still waiting for the page to be turned")
        feed._changed = True  # ...then something moves in view (a hand, the page being swapped)
        feed._identify(cam, H)
        self.assertEqual(told, ["This looks like the alphabet sheet again."])
        self.assertEqual(feed.sheet_name, "alphabet")

    def test_a_different_sheet_is_accepted_at_once_because_that_proves_the_page_changed(self):
        feed, session, voice, told, _, _ = self.make_feed("alphabet", "words")
        session.on_next_page()
        self.assertFalse(feed._changed)
        cam2, H2 = self.photo_of_file("words")
        feed._identify(cam2, H2)
        self.assertEqual((feed.sheet_name, told), ("words", ["This is the words sheet."]))

    def test_scene_change_is_noticed_from_the_picture_or_from_losing_the_page(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "alphabet")
        feed.frame = cam
        session.on_next_page()
        feed.H = H
        feed._maybe_identify(cam, now=feed._identify_from + 0.5)
        self.assertFalse(feed._changed, "the same picture is not a change")
        hand = cam.copy()
        hand[:, : hand.shape[1] // 3] = 20  # a big dark shape moves in
        feed._maybe_identify(hand, now=feed._identify_from + 1.0)
        self.assertTrue(feed._changed)
        session.on_next_page()
        self.assertFalse(feed._changed, "a new command starts over")
        feed.H = None
        for i in range(3):  # the page is lost for a few frames
            feed._maybe_identify(cam, now=feed._identify_from + 0.1 * i)
        self.assertTrue(feed._changed)

    def test_if_nothing_changes_it_says_so_and_carries_on_with_the_same_sheet_soon(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "alphabet")
        feed.frame = cam
        session.on_next_page()
        feed.H = H
        feed._maybe_identify(cam, now=feed._identify_from + tutor.IDENTIFY_UNCHANGED_SECONDS + 1)
        self.assertEqual(feed.identify_until, 0)
        self.assertEqual(told, ["The page didn't change, so I am still using the alphabet sheet."])

    def test_while_between_pages_nothing_from_the_old_sheet_is_drawn_or_locked(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "words")
        session.on_next_page()
        cam2, H2 = self.photo_of_file("words")
        before = feed.reader.snapshot()[2]
        feed.reader.submit(cam2, H2)  # the reader is still set to the alphabet: it would read the words page as garbage
        deadline = time.time() + 20
        while time.time() < deadline and feed.reader.snapshot()[2] == before:
            time.sleep(0.02)
        feed._absorb_reading()
        self.assertEqual(feed.stable, [])

    def test_giving_up_says_so_and_stays_on_the_old_sheet(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "words")
        session.on_next_page()
        feed.H = None
        feed._maybe_identify(cam, now=feed.identify_until + 1)
        self.assertEqual(feed.identify_until, 0)
        self.assertEqual(feed.sheet_name, "alphabet")
        self.assertIn("still using the alphabet sheet", told[-1])

    def test_a_late_answer_after_giving_up_is_ignored(self):
        feed, session, voice, told, cam, H = self.make_feed("alphabet", "words")
        session.on_next_page()
        feed.identify_until = 0.0  # already resolved or cancelled
        feed.use_sheet("words")
        self.assertEqual(feed.sheet_name, "alphabet")

    def test_page_source_is_asked_to_look_again(self):
        class Src:
            unlocked = 0
            def unlock(self):
                Src.unlocked += 1
        feed = tutor.CameraFeed(Src(), None, None, observe_sheet=SHEET.cells)
        feed.new_page()
        self.assertEqual(Src.unlocked, 1)


def pf_points(name):
    import poke_files as pf
    return set(pf.after_flip(pf.guide_dots(str(POKE_FILES[name]))))


class EnvAndVoiceCheckTests(unittest.TestCase):
    def test_env_file_is_loaded_without_overriding_and_without_returning_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text('# comment\n\nBRAILLIE_T1=abc123\nexport BRAILLIE_T2="quoted value"\nBRAILLIE_T3=\nnot a line\nBRAILLIE_T4=keep-me\n')
            with mock.patch.dict(os.environ, {"BRAILLIE_T4": "already set"}, clear=False):
                names = tutor.load_env_files([p, Path(d) / "missing.env"])
                self.assertEqual(sorted(names), ["BRAILLIE_T1", "BRAILLIE_T2"])
                self.assertEqual(os.environ["BRAILLIE_T1"], "abc123")
                self.assertEqual(os.environ["BRAILLIE_T2"], "quoted value")
                self.assertEqual(os.environ["BRAILLIE_T4"], "already set")
                self.assertNotIn("BRAILLIE_T3", os.environ)
                for k in ("BRAILLIE_T1", "BRAILLIE_T2"):
                    os.environ.pop(k, None)

    def voice(self, elevenlabs=False, **kw):
        """A voice module; `elevenlabs` makes its speak() call ElevenLabs for the debrief, as the older voice_io did."""
        if elevenlabs:
            def speak(text, mode="normal"):
                return _elevenlabs_speak(text)  # noqa: F821
        else:
            def speak(text, mode="normal"):
                return None
        return type("V", (), {"MOCK_MODE": False, "DEEPGRAM_API_KEY": "k", "ELEVENLABS_API_KEY": "", "speak": staticmethod(speak), **kw})

    def test_mock_voice_is_reported_as_silent(self):
        c = tutor.voice_check(self.voice(MOCK_MODE=True))
        self.assertEqual((c["mode"], c["ok"]), ("mock", False))
        self.assertIn("not played", c["problems"][0])

    def test_missing_key_is_a_problem_naming_the_fix(self):
        c = tutor.voice_check(self.voice(DEEPGRAM_API_KEY=""))
        self.assertFalse(c["ok"])
        self.assertTrue(any("DEEPGRAM_API_KEY" in p and ".env" in p for p in c["problems"]))

    def test_missing_packages_are_problems_and_a_missing_debrief_key_only_a_warning(self):
        with mock.patch("importlib.util.find_spec", side_effect=lambda name: None if name == "pyaudio" else object()):
            c = tutor.voice_check(self.voice())
        self.assertFalse(c["ok"])
        self.assertTrue(any("pyaudio" in p for p in c["problems"]))
        with mock.patch("importlib.util.find_spec", return_value=object()):
            c = tutor.voice_check(self.voice(elevenlabs=True))  # a voice module that really does use ElevenLabs
        self.assertTrue(c["ok"])
        self.assertTrue(any("ELEVENLABS_API_KEY" in w for w in c["warnings"]))

    def test_no_elevenlabs_warning_when_the_voice_module_only_uses_deepgram(self):
        """The current voice_io speaks everything through Deepgram: a missing ElevenLabs key is not worth a warning."""
        with mock.patch("importlib.util.find_spec", return_value=object()):
            c = tutor.voice_check(self.voice(elevenlabs=False))
        self.assertEqual((c["ok"], c["warnings"]), (True, []))

    def test_a_docstring_that_only_mentions_elevenlabs_is_not_a_use_of_it(self):
        """The current voice_io says 'ElevenLabs code is retained in _elevenlabs_speak() but is not called': that must not warn."""
        def speak(text, mode="normal"):
            """Routes through Deepgram. ElevenLabs code is retained in _elevenlabs_speak() but is not called from this path."""
            return None
        self.assertFalse(tutor._speech_uses_elevenlabs(type("V", (), {"speak": staticmethod(speak)})))
        self.assertFalse(tutor._speech_uses_elevenlabs(object()), "no speak() at all")

    def test_report_is_loud_when_silent_and_quiet_when_fine(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tutor.report_voice({"mode": "live", "ok": False, "problems": ["no key"], "warnings": []})
        self.assertIn("YOU WILL NOT HEAR ANYTHING", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tutor.report_voice({"mode": "live", "ok": True, "problems": [], "warnings": []})
        self.assertIn("Speech will be played", out.getvalue())

    def test_explore_mode_setup_uses_the_camera_not_the_layout(self):
        import argparse
        ap = argparse.ArgumentParser()
        tutor.add_setup_args(ap)
        a = ap.parse_args(["--mode", "explore", "--sheet", "words"])
        setup = tutor.setup_from_args(a, ap)
        self.assertTrue(setup.observed)
        self.assertFalse(setup.layout_scan)
        self.assertEqual(len(setup.cells), len(sheets.get_sheet("words").cells))
        a = ap.parse_args(["--mode", "explore"])
        self.assertEqual(len(tutor.setup_from_args(a, ap).cells), 26)  # defaults to the alphabet sheet


if __name__ == "__main__":
    unittest.main()
