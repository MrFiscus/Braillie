"""Tests for learn mode wired into the real tutor: TutorSession, its loop, saved progress, the server's endpoints and the options."""
import argparse
import contextlib
import io
import json
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import make_sheet
import sheets
import tutor
import tutor_server
from learn import DOTS, name
from progress import Progress
from test_server import http
from test_tutor import WC, FakeVoice

ALPHABET = sheets.get_sheet("alphabet")


def pos_of(letter):
    c = next(c for c in ALPHABET.cells if frozenset(c["dots"]) == DOTS[letter])
    return c["x"], c["y"]


class Finger:
    def __init__(self):
        self.pos = None

    def __call__(self):
        return self.pos


def learn_session(progress=None, path=None, coach=None, finger=None, tones=False):
    voice = FakeVoice()
    finger = finger or Finger()
    s = tutor.TutorSession(voice, WC, ALPHABET.cells, finger=finger, scan=None, mode="learn", names=ALPHABET.names, coach=coach,
                           rng=random.Random(1), progress=progress, progress_file=path, tones=tones)
    return s, voice, finger


def wait_for(pred, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class SessionTests(unittest.TestCase):
    def test_learn_mode_creates_the_journey_and_registers_every_command(self):
        s, voice, _ = learn_session()
        s.attach()
        self.assertIsNotNone(s.journey)
        expected = {"start quiz", "repeat", "hint", "found it", "next", "next page", "explore", "practice", "learn", "read", "quiz", "menu",
                    "help", "slower", "faster", "take your time", "normal pace", "stop"}  # (the phrases they are registered under)
        self.assertLessEqual(expected, set(voice.commands))
        self.assertIsNone(tutor.TutorSession(FakeVoice(), WC, [], lambda: None, mode="explore").journey)
        self.assertEqual(s.learning_status()["phase"], "idle")
        self.assertIsNone(tutor.TutorSession(FakeVoice(), WC, [], lambda: None, mode="letters").learning_status())

    def test_the_real_loop_teaches_and_the_finger_answers_and_progress_is_saved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            s, voice, finger = learn_session(path=path, tones=True)  # (the fake voice has no speaker: the tone is counted, not heard)
            s.attach()
            voice.commands["start quiz"]()
            self.assertEqual(s.state, "learning")
            self.assertIn("six places for raised dots", " ".join(voice.said))
            self.assertEqual(s.learning_status()["target"], "a")
            self.assertTrue(path.exists(), "the session was recorded straight away")
            finger.pos = pos_of("a")  # the learner rests a finger on A
            self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"), voice.said[-3:])
            self.assertTrue(any("That's the letter A." in t for t in voice.said))
            self.assertIn("Earcons", type(s.earcons).__name__)
            self.assertIn("correct", s.earcons.played)
            s.on_stop()
            self.assertTrue(s.finished.is_set())
            saved = Progress.load(path)
            self.assertEqual(saved.sessions, 1)

    def test_a_reading_spoiled_by_the_hand_does_not_fail_a_right_answer(self):
        """The reported bug, through the real session: with a live reader running, every letter touched came back as
        "a cell I don't recognise" -- because the hand that answered was over the cell being read."""
        s, voice, finger = learn_session()
        s.scan = lambda: [{**c, "dots": frozenset({1, 2, 3, 4, 5, 6})} for c in ALPHABET.cells]
        s.attach()
        voice.commands["start quiz"]()
        self.assertEqual(s.learning_status()["target"], "a")
        finger.pos = pos_of("a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"), voice.said[-3:])
        self.assertFalse(any("recognise" in t for t in voice.said), voice.said)
        s.on_stop()

    def test_start_after_stop_begins_again_and_the_loop_runs_again(self):
        s, voice, finger = learn_session()
        s.attach()
        voice.commands["start quiz"]()
        voice.commands["stop"]()
        self.assertTrue(s.finished.is_set())
        self.assertTrue(wait_for(lambda: not s._learn_thread.is_alive(), 3))
        voice.commands["start quiz"]()
        self.assertFalse(s.finished.is_set())
        self.assertTrue(s._learn_thread.is_alive())
        finger.pos = pos_of("a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        s.on_stop()

    def test_progress_from_an_earlier_session_is_used(self):
        p = Progress()
        p.sessions, p.streak, p.last_day = 3, 2, "2000-01-01"
        p.lesson_done("l1", 1.0)
        for l in "abc":
            for _ in range(3):
                p.record(l, True)
        s, voice, _ = learn_session(progress=p)
        s.attach()
        voice.commands["start quiz"]()
        self.assertIn("Welcome back", voice.said[0])
        self.assertEqual(s.learning_status()["lesson"]["id"], "l2", "carries on from the next lesson")
        s.on_stop()

    def test_explore_inside_learn_mode_speaks_what_is_under_the_finger(self):
        s, voice, finger = learn_session()
        s.attach()
        voice.commands["start quiz"]()
        voice.commands["explore"]()
        self.assertEqual(s.learning_status()["phase"], "explore")
        finger.pos = pos_of("d")
        self.assertTrue(wait_for(lambda: any("The letter D." in t for t in voice.said)), voice.said[-3:])
        voice.commands["next"]()
        self.assertEqual(s.learning_status()["phase"], "teach")
        s.on_stop()

    def test_explore_and_practice_in_other_modes_say_where_they_belong(self):
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, ALPHABET.cells, lambda: None, mode="letters")
        s.on_explore()
        s.on_practice()
        self.assertIn("--mode learn", voice.said[0])
        self.assertIn("--mode learn", voice.said[1])

    def test_a_current_sheet_can_be_asked_for_and_next_page_recognition_is_used_to_switch(self):
        s, voice, _ = learn_session()
        self.assertEqual(s.current_sheet(), "alphabet")
        calls = []
        s.new_page = lambda: calls.append(1)
        s.journey.host.request_sheet("words")
        self.assertEqual(calls, [1])

    def test_the_quiz_modes_still_work_exactly_as_before(self):
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, ALPHABET.cells, lambda: None, mode="letters", questions=1, names=ALPHABET.names, rng=random.Random(2))
        s.on_start()
        self.assertEqual(s.state, "asking")
        self.assertIn("Find", voice.said[-1])


class OptionTests(unittest.TestCase):
    def parse(self, *argv):
        ap = argparse.ArgumentParser()
        tutor.add_setup_args(ap)
        return ap.parse_args(list(argv)), ap

    def test_learn_mode_reads_the_sheet_by_observation_and_can_switch_sheets(self):
        a, ap = self.parse("--mode", "learn")
        setup = tutor.setup_from_args(a, ap)
        self.assertTrue(setup.observed)
        self.assertEqual(len(setup.cells), 26)
        self.assertEqual(sorted(tutor.known_sheets_for(a, setup)), sorted(sheets.SHEET_NAMES))
        a2, ap2 = self.parse("--mode", "letters")
        self.assertIsNone(tutor.known_sheets_for(a2, tutor.setup_from_args(a2, ap2)))

    def test_the_ai_coach_turns_itself_on_in_learn_mode_only_when_there_is_a_key(self):
        class Client:
            model = "test-model"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with mock.patch.object(tutor.LLMClient, "from_env", return_value=None):
                a, _ = self.parse("--mode", "learn")
                self.assertIsNone(tutor.make_coach(a))
            self.assertIn("OPENAI_API_KEY", out.getvalue())
            with mock.patch.object(tutor.LLMClient, "from_env", return_value=Client()):
                a, _ = self.parse("--mode", "learn")
                self.assertIsNotNone(tutor.make_coach(a))
                a, _ = self.parse("--mode", "learn", "--no-llm")
                self.assertIsNone(tutor.make_coach(a), "can be switched off")
                a, _ = self.parse("--mode", "letters")
                self.assertIsNone(tutor.make_coach(a), "other modes unchanged: only with --llm")
                a, _ = self.parse("--mode", "letters", "--llm")
                self.assertIsNotNone(tutor.make_coach(a))

    def test_progress_is_kept_only_in_learn_mode_and_per_profile(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(tutor, "progress_path", lambda profile="default": Path(d) / f"progress-{profile}.json"):
                a, _ = self.parse("--mode", "explore")
                self.assertEqual(tutor.progress_for(a), (None, None))
                p = Progress()
                p.record("a", True)
                p.save(Path(d) / "progress-ana.json")
                with contextlib.redirect_stdout(io.StringIO()):
                    a, _ = self.parse("--mode", "learn", "--profile", "ana")
                    progress, path = tutor.progress_for(a)
                    self.assertEqual(progress.practised(), ["a"])
                    a, _ = self.parse("--mode", "learn", "--profile", "ben")
                    self.assertEqual(tutor.progress_for(a)[0].practised(), [], "someone else's progress is separate")

    def test_the_lesson_phrases_are_understood_by_the_voice_module(self):
        voice, _ = tutor.load_teammate_modules(mock=True)
        heard = lambda text: voice._match_command(voice._normalize_transcript(text))
        for said, want in (("explore", "explore"), ("let me explore", "explore"), ("practice", "practice"), ("let's practise", "practice"),
                           ("review", "practice"), ("start", "start quiz"), ("start lesson", "start quiz"), ("next", "next"),
                           ("next page", "next page"), ("stop", "stop")):
            self.assertEqual(heard(said), want, said)


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        face = make_sheet.render_face_preview()
        h, w = face.shape[:2]
        quad = np.float32([[160, 60], [1000, 120], [1080, 880], [90, 840]])
        T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
        frame = cv2.cvtColor(cv2.warpPerspective(face, T, (1280, 960), borderValue=255), cv2.COLOR_GRAY2BGR)
        path = os.path.join(tempfile.mkdtemp(), "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 960))
        for _ in range(12):
            vw.write(frame)
        vw.release()
        cls.dir = tempfile.mkdtemp()
        cls.progress_file = Path(cls.dir) / "p.json"
        cls.fake = FakeVoice()
        cls.voice = tutor_server.RecordingVoice(cls.fake)
        cls.feed = tutor.CameraFeed(None, ALPHABET.cells)
        cls.session = tutor.TutorSession(cls.voice, WC, ALPHABET.cells, cls.feed.finger, cls.feed.scan, "learn", names=ALPHABET.names,
                                         rng=random.Random(5), progress=Progress(), progress_file=cls.progress_file, tones=False)
        cls.session.attach()
        cls.rt = tutor_server.TutorRuntime(tutor.open_camera(path), cls.feed, cls.session, cls.voice, loop_file=True)
        cls.rt.start()
        cls.server = tutor_server.serve(cls.rt, port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        # a server that is NOT in learn mode
        cls.plain = tutor.TutorSession(FakeVoice(), WC, ALPHABET.cells, lambda: None, mode="letters")
        cls.plain_rt = tutor_server.TutorRuntime(tutor.open_camera(path), tutor.CameraFeed(None, ALPHABET.cells), cls.plain, tutor_server.RecordingVoice(FakeVoice()), loop_file=True)
        cls.plain_rt.start()
        cls.plain_server = tutor_server.serve(cls.plain_rt, port=0)
        cls.plain_base = f"http://127.0.0.1:{cls.plain_server.server_address[1]}"
        threading.Thread(target=cls.plain_server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.session.on_stop()
        for rt, server in ((cls.rt, cls.server), (cls.plain_rt, cls.plain_server)):
            rt.stop()
            server.shutdown()
            server.server_close()

    def test_state_carries_the_lesson_and_the_progress(self):
        s = http(self.base + "/api/state")[2]
        self.assertEqual(s["config"]["mode"], "learn")
        self.assertEqual(s["learning"]["phase"], "idle")
        self.assertEqual(s["progress"]["sessions"], 0)
        self.assertIn("explore", s["config"]["commands"])
        self.assertIn("practice", s["config"]["commands"])
        plain = http(self.plain_base + "/api/state")[2]
        self.assertEqual((plain["learning"], plain["progress"]), (None, None))

    def test_the_commands_reach_the_lessons(self):
        self.assertEqual(http(self.base + "/api/command", {"command": "start"})[0], 400, "'start' is a voice phrase, the API takes the command name")
        self.assertEqual(http(self.base + "/api/command", {"command": "start quiz"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["learning"]["phase"] == "teach"))
        s = http(self.base + "/api/state")[2]
        self.assertEqual((s["learning"]["target"], s["learning"]["target_dots"], s["tutor"]["state"]), ("a", [1], "learning"))
        self.assertEqual(http(self.base + "/api/command", {"command": "hint"})[0], 202)
        self.assertEqual(http(self.base + "/api/command", {"command": "explore"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["learning"]["phase"] == "explore"))
        self.assertEqual(http(self.base + "/api/command", {"command": "next"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["learning"]["phase"] == "teach"))

    def test_progress_can_be_read_and_an_account_copy_merged_in(self):
        code, _, body = http(self.base + "/api/progress")
        self.assertEqual(code, 200)
        self.assertIn("summary", body)
        remote = Progress()
        for _ in range(3):
            remote.record("q", True)
        remote.lesson_done("l6", 0.9)  # (a lesson the other tests are not waiting to start)
        code, _, merged = http(self.base + "/api/progress", {"data": remote.to_dict()})
        self.assertEqual(code, 200)
        self.assertEqual(merged["summary"]["mastery"]["q"], 0.75)
        self.assertEqual(merged["summary"]["lessons"]["l6"]["best"], 0.9)
        self.assertEqual(Progress.load(self.progress_file).lessons["l6"]["best"], 0.9, "and it was saved")
        code, _, again = http(self.base + "/api/progress", {"data": remote.to_dict()})
        self.assertEqual(again["summary"]["mastery"]["q"], 0.75, "merging the same copy twice changes nothing")

    def test_bad_progress_bodies_are_refused_and_other_modes_have_no_progress(self):
        for bad in ({"data": "nope"}, {}, {"data": [1, 2]}):
            self.assertEqual(http(self.base + "/api/progress", bad)[0], 400, bad)
        self.assertEqual(http(self.plain_base + "/api/progress")[0], 404)
        self.assertEqual(http(self.plain_base + "/api/progress", {"data": {}})[0], 404)
        self.assertEqual(http(self.base + "/api/progress", {"data": {}}, origin="https://evil.example")[0], 403)


if __name__ == "__main__":
    unittest.main()
