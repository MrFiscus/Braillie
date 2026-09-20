"""Tests for the tutor glue: reader.py + tutor.py against the real voice_io (mock mode) and backend word_correction.

Run: python -m unittest test_tutor -v     (skipped if pyspellchecker is not installed)
"""
import contextlib
import io
import os
import random
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

import make_sheet
import page
import reader
import tutor
from detect import cells_from_layout, dots_to_char

try:
    VOICE, WC = tutor.load_teammate_modules(mock=True)
except ImportError as e:  # e.g. pyspellchecker missing
    VOICE = WC = None
    SKIP_REASON = str(e)


class FakeVoice:
    """Stands in for voice_io: records what would have been said."""

    def __init__(self):
        self.said, self.debriefs, self.commands = [], [], {}

    def speak(self, text, mode="normal"):
        self.said.append(text)

    def speak_debrief(self, accuracy, missed):
        self.debriefs.append((accuracy, missed))

    def register_command(self, name, callback):
        self.commands[name] = callback


class PlanningClient:
    """Deterministic stand-in for OpenAI's structured target-plan reply."""

    model = "fake-adaptive-model"

    def __init__(self, error=None):
        self.calls, self.error = [], error

    def chat_json(self, _system, user, _schema):
        self.calls.append(__import__("json").loads(user))
        if self.error:
            raise self.error
        payload = self.calls[-1]
        candidates = payload["candidate_targets"]
        history = payload["miss_history"]
        ordered = sorted(candidates, key=lambda key: (-history.get(key, {}).get("misses", 0), key))
        return {"targets": ordered[:payload["target_count"]], "reason": "recent misses first, then variety"}


def session(mode="letters", cells=None, scan=None, finger=None, **kw):
    voice = FakeVoice()
    cells = make_sheet.sheet_cells() if cells is None else cells
    s = tutor.TutorSession(voice, WC, cells, finger=finger or (lambda: None), scan=scan, mode=mode,
                           rng=random.Random(3), **kw)
    return s, voice


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ReaderTests(unittest.TestCase):
    def test_words_and_gaps(self):
        cells = cells_from_layout(["cap dog", "a"], x0=10, y0=10, pitch_x=12, pitch_y=20)
        self.assertEqual(reader.read_lines(cells), ["cap dog", "a"])

    def test_word_at_and_off_page(self):
        cells = cells_from_layout(["cap dog"], x0=10, y0=10, pitch_x=12, pitch_y=20)
        self.assertEqual(reader.word_at(cells, 22, 12), "cap")
        self.assertEqual(reader.word_at(cells, 70, 10), "dog")
        self.assertIsNone(reader.word_at(cells, 500, 500))

    def test_non_letters_read_as_question_marks(self):
        from detect import _make_cell
        cells = [_make_cell(10, 10, 6, 9, "100000", 1.0, 0, 0), _make_cell(20, 10, 6, 9, "111111", 1.0, 0, 1)]
        self.assertEqual(reader.word_text(cells), "a?")


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class LetterQuizTests(unittest.TestCase):
    def cell_pos(self, s, letter):
        c = s._target_cell(letter)
        return (c["x"], c["y"])

    def test_full_flow_with_hint_wrong_then_right_then_skip(self):
        pos = {"now": None}
        s, v = session(questions=2, finger=lambda: pos["now"])
        s.on_start()
        first = s.items[0]
        self.assertEqual(v.said[-1], f"Find the letter {first.upper()}.")
        s.on_found_it()
        self.assertIn("can't see your finger", v.said[-1])
        pos["now"] = (500, 500)
        s.on_found_it()
        self.assertIn("not on a cell", v.said[-1])
        wrong = "a" if first != "a" else "b"
        pos["now"] = self.cell_pos(s, wrong)
        s.on_found_it()
        self.assertIn(f"not the letter {first.upper()}", v.said[-1])
        s.on_hint()
        self.assertIn("dot", v.said[-1])
        s.on_hint()
        self.assertIn("row", v.said[-1])
        pos["now"] = self.cell_pos(s, first)
        s.on_found_it()
        self.assertTrue(any(t.startswith("Correct!") for t in v.said))
        self.assertEqual(s.index, 1)
        s.on_repeat()
        self.assertEqual(v.said[-1], f"Find the letter {s.items[1].upper()}.")
        s.on_next()  # skip the second
        self.assertTrue(s.finished.is_set())
        acc, missed = v.debriefs[-1]
        self.assertAlmostEqual(acc, 0.5)  # 1 of 2 asked
        self.assertEqual(set(missed), {s.items[0].upper(), s.items[1].upper()})

    def test_close_miss_is_flagged(self):
        s, v = session(questions=1, finger=lambda: None)
        s.on_start()
        target = s.items[0]
        from detect import dot_distance
        close = next(c for c in s.cells if dot_distance(c["dots"], s._target_cell(target)["dots"]) == 1)
        s.finger = lambda: (close["x"], close["y"])
        s.on_found_it()
        self.assertIn("differs by 1 dot", v.said[-1])

    def test_gives_up_after_max_tries_and_reveals_position(self):
        s, v = session(questions=1, max_tries=2)
        s.on_start()
        target = s.items[0]
        other = "a" if target != "a" else "b"
        s.finger = lambda: self.cell_pos(s, other)
        s.on_found_it()
        s.on_found_it()
        self.assertIn("Let's move on", v.said[-1])
        self.assertTrue(s.finished.is_set())
        self.assertEqual(v.debriefs[-1][0], 0.0)
        self.assertEqual(v.debriefs[-1][1], [target.upper()])

    def test_commands_before_start_and_stop_without_questions(self):
        s, v = session()
        s.on_found_it()
        s.on_hint()
        s.on_next()
        s.on_repeat()
        self.assertTrue(all("start quiz" in t for t in v.said))
        s.on_stop()
        self.assertEqual(v.said[-1], "Goodbye.")
        self.assertEqual(v.debriefs, [])
        self.assertTrue(s.finished.is_set())


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class AdaptiveQuizTests(unittest.TestCase):
    @staticmethod
    def wait_for_plan(planner):
        for _ in range(100):
            if planner._pending and planner._pending.done.is_set():
                return
            time.sleep(0.01)
        raise AssertionError("adaptive plan did not complete")

    def test_missed_letter_is_prioritized_in_the_next_round(self):
        client = PlanningClient()
        planner = tutor.AdaptiveTargetPlanner(client)
        s, _ = session(questions=1, max_tries=1, adaptive_planner=planner)
        s.on_start()  # first plan is not ready, so this deliberately uses fallback
        missed = s.items[0]
        wrong = "a" if missed != "a" else "b"
        cell = s._target_cell(wrong)
        s.finger = lambda: (cell["x"], cell["y"])
        s.on_found_it()  # completes the round and prefetches the history-aware plan
        self.wait_for_plan(planner)
        self.assertFalse(s.finished.is_set(), "adaptive mode must allow another live round")

        s.on_start()
        self.assertEqual(s.miss_history[missed]["misses"], 1)
        self.assertEqual(s.items, [missed])
        self.assertEqual(s.last_selection["source"], "openai")
        self.assertEqual(client.calls[-1]["miss_history"][missed]["misses"], 1)
        s.on_stop()
        self.assertTrue(s.finished.is_set())

    def test_word_quiz_uses_the_same_history_aware_planner(self):
        client = PlanningClient()
        planner = tutor.AdaptiveTargetPlanner(client)
        s, _ = session(mode="word-quiz", cells=[], words=("cap", "dog", "sun"), questions=2,
                       max_tries=1, scan=lambda: [], adaptive_planner=planner)
        s.on_start()
        self.assertEqual(s.items, ["cap", "dog"])
        s._check_word((0, 0))  # miss cap; the fake scan has no matching word
        s._check_word((0, 0))  # miss dog and finish the round
        self.wait_for_plan(planner)

        s.on_start()
        self.assertEqual(s.items, ["cap", "dog"])
        self.assertEqual(s.last_selection["source"], "openai")
        self.assertEqual(s.miss_history["cap"]["misses"], 1)
        self.assertEqual(s.miss_history["dog"]["misses"], 1)

    def test_failed_openai_plan_uses_the_existing_fallback(self):
        planner = tutor.AdaptiveTargetPlanner(PlanningClient(error=ConnectionError("bad API key")))
        s, _ = session(questions=1, adaptive_planner=planner)
        s.miss_history = {"a": {"misses": 3, "last_missed_quiz": 1}}
        candidates = s._candidate_targets()
        planner.prefetch(candidates, 1, s.mode, s._selection_history(False))
        self.wait_for_plan(planner)
        picked = s._pick_items()
        self.assertEqual(len(picked), 1)
        self.assertEqual(s.last_selection["source"], "fallback")
        self.assertIn("ConnectionError", s.last_selection["reason"])


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class WordModeTests(unittest.TestCase):
    """The real backend word_correction decides when to re-detect; the fake scan plays back what the camera 'saw'."""

    @staticmethod
    def scans(*lines):
        """A scan() that returns cells for each given text line in turn, then keeps repeating the last."""
        it = iter(lines)
        state = {"last": None, "calls": 0}

        def scan():
            state["calls"] += 1
            state["last"] = next(it, state["last"])
            return cells_from_layout([state["last"]], x0=10, y0=10, pitch_x=12, pitch_y=20)

        return scan, state

    def test_read_mode_redetects_a_bad_word(self):
        scan, st = self.scans("cax dog", "cap dog")
        s, v = session(mode="read", cells=[], scan=scan, finger=lambda: (22, 12))
        s.on_start()
        s.on_found_it()
        self.assertEqual(v.said[-1], "The word is cap.")
        self.assertEqual(st["calls"], 2)  # first read + one re-detect

    def test_read_mode_valid_word_needs_no_redetect(self):
        scan, st = self.scans("cap dog")
        s, v = session(mode="read", cells=[], scan=scan, finger=lambda: (22, 12))
        s.on_start()
        s.on_found_it()
        self.assertEqual((v.said[-1], st["calls"]), ("The word is cap.", 1))

    def test_read_mode_never_guesses(self):
        scan, st = self.scans("cax dog")
        s, v = session(mode="read", cells=[], scan=scan, finger=lambda: (22, 12))
        s.on_start()
        s.on_found_it()
        self.assertIn("couldn't read that clearly", v.said[-1])
        self.assertIn("c, a, x", v.said[-1])
        self.assertEqual(st["calls"], 1 + 2)  # first read + max_redetects

    def test_quiz_accepts_after_redetect(self):
        scan, st = self.scans("cax dog", "cap dog")
        s, v = session(mode="word-quiz", cells=[], scan=scan, finger=lambda: (22, 12), words=("cap",), questions=1)
        s.on_start()
        self.assertEqual(v.said[-1], "Find the word cap.")
        s.on_hint()
        self.assertEqual(v.said[-1], "The word has 3 letters and starts with c.")
        s.on_found_it()
        self.assertIn("Correct! The word is cap.", v.said)
        self.assertEqual(v.debriefs[-1], (1.0, []))

    def test_quiz_wrong_word_then_gives_up(self):
        scan, st = self.scans("cap dog")
        s, v = session(mode="word-quiz", cells=[], scan=scan, finger=lambda: (70, 10), words=("cap",), questions=1, max_tries=2)
        s.on_start()
        s.on_found_it()
        self.assertEqual(v.said[-1], "I read dog. Try again.")
        s.on_found_it()
        self.assertIn("The word was cap", v.said[-1])
        self.assertEqual(v.debriefs[-1], (0.0, ["CAP"]))

    def test_no_word_under_finger(self):
        scan, _ = self.scans("cap dog")
        s, v = session(mode="read", cells=[], scan=scan, finger=lambda: (500, 500))
        s.on_start()
        s.on_found_it()
        self.assertIn("don't see a word", v.said[-1])


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class RealVoiceTests(unittest.TestCase):
    """Drive the tutor through teammates' real voice_io (offline mock mode): commands in, speech out."""

    def setUp(self):
        VOICE._callbacks.clear()

    def tearDown(self):
        VOICE._callbacks.clear()

    def test_voice_commands_drive_a_whole_session(self):
        self.assertTrue(VOICE.MOCK_MODE)
        pos = {"now": None}
        s = tutor.TutorSession(VOICE, WC, make_sheet.sheet_cells(), finger=lambda: pos["now"], questions=1, rng=random.Random(1))
        s.attach()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            VOICE._fire_command("start quiz")
            target = s.items[0]
            c = s._target_cell(target)
            pos["now"] = (c["x"], c["y"])
            VOICE._fire_command("repeat")
            VOICE._fire_command("hint")
            VOICE._fire_command("found it")
        text = out.getvalue()
        self.assertIn(f"[SPEAK/Deepgram] Find the letter {target.upper()}.", text)
        self.assertIn("[SPEAK/Deepgram] Correct!", text)
        self.assertIn("[SPEAK/ElevenLabs]", text)  # the debrief goes through the ElevenLabs voice
        self.assertTrue(s.finished.is_set())

    def test_every_voice_command_is_registered(self):
        s = tutor.TutorSession(VOICE, WC, [], finger=lambda: None)
        s.attach()
        self.assertEqual(set(VOICE._callbacks), {"start quiz", "repeat", "hint", "found it", "next", "stop"})


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class CameraLoopTests(unittest.TestCase):
    """The whole camera app: markers -> page registration -> click as the finger -> 'found it' -> spoken feedback."""

    def test_end_to_end_from_a_video(self):
        face = make_sheet.render_face_preview()
        h, w = face.shape[:2]
        quad = np.float32([[160, 60], [1000, 120], [1080, 880], [90, 840]])
        T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
        frame = cv2.warpPerspective(face, T, (1280, 960), borderValue=255)
        path = os.path.join(tempfile.mkdtemp(), "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 960))
        for _ in range(40):
            vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        vw.release()

        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, make_sheet.sheet_cells(), finger=lambda: None, questions=1, rng=random.Random(5))
        callbacks, step = [], [0]
        real = {k: getattr(cv2, k) for k in ("imshow", "waitKey", "namedWindow", "setMouseCallback", "destroyAllWindows")}
        cv2.imshow = cv2.namedWindow = cv2.destroyAllWindows = lambda *a, **k: None
        cv2.setMouseCallback = lambda name, cb: callbacks.append(cb)

        def key(ms):
            step[0] += 1
            time.sleep(0.15)
            if step[0] == 2:
                return ord("s")
            if step[0] == 6:  # click on the target cell as seen by the camera, then say "found it"
                c = s._target_cell(s.items[0])
                p = T @ np.array([(make_sheet.ORIGIN[0] + c["x"]) * make_sheet.MM, (make_sheet.ORIGIN[1] + c["y"]) * make_sheet.MM, 1.0])
                callbacks[0](cv2.EVENT_LBUTTONDOWN, int(p[0] / p[2]), int(p[1] / p[2]))
                return ord("f")
            return -1

        cv2.waitKey = key
        try:
            tutor.run_camera(s, tutor.open_camera(path), None, make_sheet.sheet_cells())
        finally:
            for k, fn in real.items():
                setattr(cv2, k, fn)
        self.assertTrue(s.finished.is_set(), voice.said)
        self.assertTrue(voice.said[0].startswith("Find the letter"))
        self.assertTrue(any(t.startswith("Correct!") for t in voice.said), voice.said)
        self.assertEqual(voice.debriefs[-1], (1.0, []))


if __name__ == "__main__":
    unittest.main()
