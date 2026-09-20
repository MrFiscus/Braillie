"""Tests for the menu-driven tutor: who is here (Google keeps progress, a guest keeps nothing), the greeting, and Learn / Read / Quiz."""
import argparse
import contextlib
import io
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

import learn
import make_sheet
import phonelink
import sheets
import tutor
import tutor_server
from learn import DOTS
from progress import Progress
from test_server import http
from test_tutor import WC, FakeVoice

SHEETS = {n: sheets.get_sheet(n) for n in sheets.SHEET_NAMES}


def cell_pos(sheet, letter):
    c = next(c for c in SHEETS[sheet].cells if frozenset(c["dots"]) == DOTS[letter])
    return c["x"], c["y"]


class Finger:
    def __init__(self):
        self.pos = None

    def __call__(self):
        return self.pos


def wait_for(pred, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.03)
    return False


def fast_dwell(seconds=None):
    return learn.Dwell(seconds=0.15)  # (whatever length was asked for)


def hub(tmpdir=None, phone=None):
    """A menu-driven session with a real (camera-less) feed, a fake voice and a controllable finger."""
    voice, finger = FakeVoice(), Finger()
    feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEETS["alphabet"].cells, known_sheets=dict(SHEETS), sheet_name="alphabet")
    s = tutor.TutorSession(voice, WC, SHEETS["alphabet"].cells, finger=finger, scan=feed.scan, mode="menu", names=SHEETS["alphabet"].names,
                           rng=random.Random(3), tones=False)
    tutor.wire_new_page(s, feed)
    if phone is not None:
        s.phone_ready_fn = phone
    s.attach()
    return s, voice, finger, feed


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class UserTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.patch = mock.patch.object(tutor, "progress_path", lambda profile="default": Path(self.dir) / f"progress-{profile}.json")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_names_are_made_safe_to_show_and_to_say(self):
        self.assertEqual(tutor.clean_name("  Ana   Maria "), "Ana Maria")
        self.assertEqual(tutor.clean_name("Zoë O'Neil-Smith"), "Zoë O'Neil-Smith")
        self.assertEqual(tutor.clean_name("<b>Bob</b>"), "bBobb", "markup characters are dropped")
        self.assertEqual((tutor.clean_name(""), tutor.clean_name(None), tutor.clean_name("!!!")), ("friend",) * 3)
        self.assertEqual(len(tutor.clean_name("x" * 200)), 40)
        self.assertEqual(tutor.greeting("Ana"), "What do you want to do today, Ana? You can say learn, read, or quiz.")

    def test_a_guest_keeps_nothing_not_even_on_this_computer(self):
        s, voice, finger, feed = hub()
        s.set_user("Guest Gary", "guest")
        self.assertEqual((s.user["kind"], s.user["profile"], s.progress_file), ("guest", None, None))
        s.set_mode("learn")
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        s.progress.record("q", True)  # whatever happens...
        s.save_progress()
        s.set_mode("menu")
        self.assertEqual(os.listdir(self.dir), [], "...nothing was written")
        self.assertGreaterEqual(s.progress.sessions, 1, "(it is remembered in memory for this session, so practice can adapt)")

    def test_a_google_learner_is_remembered_across_sessions_under_their_own_name(self):
        s, voice, finger, feed = hub()
        s.set_user("Ana", "google", "user-abc-123")
        self.assertEqual((s.user["kind"], s.progress_file.name), ("google", "progress-user-abc-123.json"))
        s.set_mode("learn")
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        s.set_mode("menu")
        self.assertTrue((Path(self.dir) / "progress-user-abc-123.json").exists())
        # a new session for the same account picks it up; a different account does not
        s2, *_ = hub()
        s2.set_user("Ana", "google", "user-abc-123")
        self.assertEqual(s2.progress.sessions, 1)
        s3, *_ = hub()
        s3.set_user("Ben", "google", "user-xyz-789")
        self.assertEqual(s3.progress.sessions, 0)

    def test_a_google_session_without_an_account_id_is_treated_as_a_guest(self):
        s, *_ = hub()
        s.set_user("Ana", "google", None)
        self.assertEqual((s.user["kind"], s.progress_file), ("guest", None))
        self.assertEqual(os.listdir(self.dir), [])

    def test_the_same_user_twice_is_ignored_and_a_new_user_starts_fresh(self):
        s, voice, *_ = hub()
        s.set_user("Ana", "guest")
        first = s.progress
        s.set_user("Ana", "guest")
        self.assertIs(s.progress, first)
        s.set_user("Ben", "guest")
        self.assertIsNot(s.progress, first)
        self.assertEqual(s.hub_status()["user"], {"name": "Ben", "kind": "guest", "saves": False})


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class GreetingTests(unittest.TestCase):
    def said(self, voice):
        return [t for t in voice.said if "What do you want to do today" in t]

    def test_nothing_is_said_until_we_know_who_it_is(self):
        s, voice, *_ = hub()
        s.on_phone_ready()
        time.sleep(0.2)
        self.assertEqual(self.said(voice), [])
        self.assertEqual(s.hub_status()["user"], None)

    def test_without_a_phone_the_greeting_comes_as_soon_as_we_know_the_name(self):
        s, voice, *_ = hub()
        s.set_user("Ana", "guest")
        self.assertTrue(wait_for(lambda: self.said(voice)))
        self.assertEqual(self.said(voice), ["What do you want to do today, Ana? You can say learn, read, or quiz."])
        self.assertEqual(s.state, "menu")

    def test_with_a_phone_it_waits_for_the_phone_to_be_linked_and_says_it_once(self):
        linked = [False]
        s, voice, *_ = hub(phone=lambda: linked[0])
        s.set_user("Ana", "guest")
        time.sleep(0.3)
        self.assertEqual(self.said(voice), [], "the phone is not linked yet")
        linked[0] = True
        s.on_phone_ready()
        self.assertTrue(wait_for(lambda: self.said(voice)))
        s.on_phone_ready()
        s.on_phone_ready()
        time.sleep(0.3)
        self.assertEqual(len(self.said(voice)), 1, "not repeated")

    def test_a_phone_linked_before_the_user_signs_in_still_gets_the_greeting_at_sign_in(self):
        s, voice, *_ = hub(phone=lambda: True)
        s.on_phone_ready()
        time.sleep(0.2)
        s.set_user("Ana", "guest")
        self.assertTrue(wait_for(lambda: self.said(voice)))

    def test_a_different_user_is_greeted_by_their_own_name(self):
        s, voice, *_ = hub()
        s.set_user("Ana", "guest")
        self.assertTrue(wait_for(lambda: len(self.said(voice)) == 1))
        s.set_user("Ben", "guest")
        self.assertTrue(wait_for(lambda: len(self.said(voice)) == 2))
        self.assertIn("Ben", self.said(voice)[1])

    def test_someone_signing_in_while_a_lesson_is_running_is_not_greeted_over_it(self):
        s, voice, *_ = hub()
        s.set_mode("learn")
        s.set_user("Ana", "guest")  # a new person arrives while a mode is already running
        time.sleep(0.5)
        self.assertEqual(self.said(voice), [])
        s.set_mode("menu")

    def test_no_greeting_while_a_mode_is_running(self):
        s, voice, *_ = hub()
        s.set_user("Ana", "guest")
        self.assertTrue(wait_for(lambda: self.said(voice)))
        s.set_mode("learn")
        s.on_phone_ready()
        time.sleep(0.3)
        self.assertEqual(len(self.said(voice)), 1)
        s.set_mode("menu")


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ModeTests(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.object(tutor, "Dwell", fast_dwell)  # answering by resting takes 0.15 s here, not 1.2 s
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_the_menu_is_the_starting_point_and_all_menu_commands_are_registered(self):
        s, voice, *_ = hub()
        self.assertEqual((s.mode, s.hub_mode, s.journey), ("menu", "menu", None))
        for c in ("learn", "read", "quiz", "menu"):
            self.assertIn(c, voice.commands)
        self.assertEqual(s.hub_status()["modes"], {"learn": {"title": "Learn", "sheet": "alphabet"}, "read": {"title": "Read", "sheet": "words"},
                                                   "quiz": {"title": "Quiz", "sheet": "lookalikes"}})
        with self.assertRaises(ValueError):
            s.set_mode("dance")

    def test_learn_starts_at_a_on_the_alphabet_sheet(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        self.assertEqual((feed.sheet_name, s.hub_status()["mode"], s.mode, s.state), ("alphabet", "learn", "learn", "learning"))
        self.assertEqual(s.learning_status()["target"], "a")
        self.assertEqual(s.learning_status()["lesson"]["title"], "The first five letters")
        self.assertIn("The letter A. One dot, at the top-left.", voice.said[-1])
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        s.set_mode("menu")

    def test_quiz_uses_the_look_alikes_sheet_and_resting_a_finger_answers(self):
        s, voice, finger, feed = hub()
        s.set_mode("quiz")
        self.assertEqual((feed.sheet_name, s.mode, s.state, s.questions), ("lookalikes", "letters", "asking", 8))
        self.assertEqual(len(s.cells), len(SHEETS["lookalikes"].cells), "the session now knows the look-alikes sheet's cells")
        self.assertIn("look-alikes sheet", " ".join(voice.said))
        prompt = s.status()["prompt"]
        self.assertRegex(prompt, r"Find the letter [A-R]\.")
        target = prompt.split("letter ")[1][0].lower()
        finger.pos = cell_pos("lookalikes", target)
        self.assertTrue(wait_for(lambda: s.status()["correct"] == 1), voice.said[-3:])
        self.assertTrue(any("Correct" in t for t in voice.said))
        s.set_mode("menu")

    def test_a_wrong_touch_in_the_quiz_says_what_it_was_and_the_finger_must_move_before_the_next_answer(self):
        s, voice, finger, feed = hub()
        s.set_mode("quiz")
        target = s.status()["prompt"].split("letter ")[1][0].lower()
        wrong = next(l for l in "akblcmdn" if l != target and DOTS[l] != DOTS[target])
        finger.pos = cell_pos("lookalikes", wrong)
        self.assertTrue(wait_for(lambda: s.status()["tries"] >= 1))
        self.assertTrue(any("not" in t for t in voice.said[-2:]))
        n = len(voice.said)
        time.sleep(0.6)
        self.assertEqual(len(voice.said), n, "resting on the same wrong cell is not counted again")
        s.set_mode("menu")

    def test_a_finished_quiz_goes_back_to_the_menu(self):
        s, voice, finger, feed = hub()
        s.set_mode("quiz")
        for _ in range(8):
            target = s.status()["prompt"].split("letter ")[1][0].lower()
            finger.pos = None
            time.sleep(0.15)
            finger.pos = cell_pos("lookalikes", target)
            before = s.status()["asked"]
            self.assertTrue(wait_for(lambda: s.hub_mode == "menu" or s.status()["asked"] > before, 6), voice.said[-3:])
            if s.hub_mode == "menu":
                break
        self.assertTrue(wait_for(lambda: s.hub_mode == "menu", 6))
        text = " ".join(voice.said)
        self.assertIn("You can say learn, read, or quiz", text)
        self.assertEqual((s.mode, s.state), ("menu", "menu"))

    def test_read_uses_the_words_sheet_and_reads_the_word_under_a_resting_finger(self):
        s, voice, finger, feed = hub()
        words = SHEETS["words"]
        s.scan = lambda: words.cells  # what the camera would report for the words sheet
        s.set_mode("read")
        self.assertEqual((feed.sheet_name, s.mode, s.state), ("words", "read", "reading"))
        self.assertIn("words sheet", voice.said[-1])
        first = min(words.cells, key=lambda c: (c["row"], c["col"]))
        finger.pos = (first["x"], first["y"])
        self.assertTrue(wait_for(lambda: any("The word is cat" in t for t in voice.said)), voice.said[-3:])
        s.set_mode("menu")

    def test_switching_mode_stops_the_old_one_completely(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        lesson_loop = s._learn_thread
        self.assertTrue(lesson_loop.is_alive())
        s.set_mode("quiz")
        self.assertTrue(wait_for(lambda: not lesson_loop.is_alive(), 3), "the lesson loop ended")
        self.assertIsNone(s.journey)
        n = len(voice.said)
        finger.pos = None
        time.sleep(0.5)
        self.assertEqual(len(voice.said), n, "nothing from the lesson is said any more")
        s.set_mode("menu")
        self.assertTrue(wait_for(lambda: not any(t.name == "dwell" and t.is_alive() for t in threading.enumerate()), 3))

    def test_stop_in_a_mode_gives_the_recap_then_the_menu(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        voice.commands["stop"]()
        self.assertEqual((s.hub_mode, s.state), ("menu", "menu"))
        self.assertIn("end of this session", " ".join(voice.said))
        self.assertIn("You can say learn, read, or quiz", voice.said[-1])

    def test_the_voice_commands_switch_modes(self):
        s, voice, finger, feed = hub()
        voice.commands["quiz"]()
        self.assertEqual(s.hub_mode, "quiz")
        voice.commands["menu"]()
        self.assertEqual(s.hub_mode, "menu")
        voice.commands["learn"]()
        self.assertEqual((s.hub_mode, feed.sheet_name), ("learn", "alphabet"))
        voice.commands["read"]()
        self.assertEqual((s.hub_mode, feed.sheet_name), ("read", "words"))
        voice.commands["menu"]()

    def test_choosing_a_mode_in_a_non_menu_tutor_explains_and_changes_nothing(self):
        voice = FakeVoice()
        s = tutor.TutorSession(voice, WC, SHEETS["alphabet"].cells, lambda: None, mode="letters", names=SHEETS["alphabet"].names)
        s.on_mode("learn")
        self.assertIn("menu", voice.said[-1])
        self.assertEqual((s.mode, s.hub_status()), ("letters", None))

    def test_the_feed_can_switch_sheets_directly_and_tells_the_session(self):
        s, voice, finger, feed = hub()
        feed.select_sheet("words")
        self.assertEqual((feed.sheet_name, len(feed.observe_sheet), len(s.cells)), ("words", len(SHEETS["words"].cells), len(SHEETS["words"].cells)))
        self.assertEqual(feed.stable, [])
        self.assertEqual(feed.reader.sheet, SHEETS["words"].cells)


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
        cls.patch = mock.patch.object(tutor, "progress_path", lambda profile="default": Path(cls.dir) / f"progress-{profile}.json")
        cls.patch.start()
        cls.voice = tutor_server.RecordingVoice(FakeVoice())
        cls.feed = tutor.CameraFeed(None, None, None, observe_sheet=SHEETS["alphabet"].cells, known_sheets=dict(SHEETS), sheet_name="alphabet")
        cls.session = tutor.TutorSession(cls.voice, WC, SHEETS["alphabet"].cells, cls.feed.finger, cls.feed.scan, "menu",
                                         names=SHEETS["alphabet"].names, rng=random.Random(1), tones=False)
        tutor.wire_new_page(cls.session, cls.feed)
        cls.session.attach()
        cls.rt = tutor_server.TutorRuntime(tutor.open_camera(path), cls.feed, cls.session, cls.voice, loop_file=True)
        cls.rt.start()
        cls.server = tutor_server.serve(cls.rt, port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.plain = tutor.TutorSession(FakeVoice(), WC, SHEETS["alphabet"].cells, lambda: None, mode="letters")
        cls.plain_rt = tutor_server.TutorRuntime(tutor.open_camera(path), tutor.CameraFeed(None, None), cls.plain, tutor_server.RecordingVoice(FakeVoice()), loop_file=True)
        cls.plain_rt.start()
        cls.plain_server = tutor_server.serve(cls.plain_rt, port=0)
        cls.plain_base = f"http://127.0.0.1:{cls.plain_server.server_address[1]}"
        threading.Thread(target=cls.plain_server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.session.set_mode("menu")
        for rt, server in ((cls.rt, cls.server), (cls.plain_rt, cls.plain_server)):
            rt.stop()
            server.shutdown()
            server.server_close()
        cls.patch.stop()

    def test_state_has_the_hub(self):
        s = http(self.base + "/api/state")[2]
        self.assertEqual(s["hub"]["mode"], "menu")
        self.assertEqual(sorted(s["hub"]["modes"]), ["learn", "quiz", "read"])
        self.assertEqual(s["config"]["mode"], "menu")
        for c in ("learn", "read", "quiz", "menu"):
            self.assertIn(c, s["config"]["commands"])
        self.assertIsNone(http(self.plain_base + "/api/state")[2]["hub"])

    def test_the_session_endpoint_validates_and_records_who_is_here(self):
        bad = [{}, {"kind": "admin", "name": "x"}, {"kind": "guest"}, {"kind": "guest", "name": "   "}, {"kind": "guest", "name": 5},
               {"kind": "google", "name": "Ana"}, {"kind": "google", "name": "Ana", "profile": "../etc"}, {"kind": "google", "name": "Ana", "profile": "a b"},
               {"kind": "google", "name": "Ana", "profile": "x" * 200}]
        for body in bad:
            self.assertEqual(http(self.base + "/api/session", body)[0], 400, body)
        code, _, body = http(self.base + "/api/session", {"kind": "guest", "name": "Gina <b>"})
        self.assertEqual(code, 200)
        self.assertEqual(body["hub"]["user"], {"name": "Gina b", "kind": "guest", "saves": False})
        code, _, body = http(self.base + "/api/session", {"kind": "google", "name": "Ana", "profile": "user-1"})
        self.assertEqual(body["hub"]["user"], {"name": "Ana", "kind": "google", "saves": True})
        self.assertTrue(wait_for(lambda: any("What do you want to do today, Ana" in x["text"] for x in http(self.base + "/api/state")[2]["said"])))

    def test_the_mode_endpoint_switches_and_returns(self):
        self.assertEqual(http(self.base + "/api/mode", {"mode": "dance"})[0], 400)
        self.assertEqual(http(self.base + "/api/mode", {})[0], 400)
        self.assertEqual(http(self.base + "/api/mode", {"mode": "quiz"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["hub"]["mode"] == "quiz"))
        s = http(self.base + "/api/state")[2]
        self.assertEqual((s["config"]["sheet"], s["tutor"]["state"]), ("lookalikes", "asking"))
        self.assertEqual(http(self.base + "/api/mode", {"mode": "menu"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["hub"]["mode"] == "menu"))

    def test_the_prompt_endpoint_speaks_only_what_it_is_allowed_to(self):
        self.assertEqual(http(self.base + "/api/prompt", {"name": "sing me a song"})[0], 400)
        self.assertEqual(http(self.base + "/api/prompt", {})[0], 400)
        self.assertEqual(http(self.base + "/api/prompt", {"name": "welcome"})[0], 202)
        self.assertTrue(wait_for(lambda: any("Welcome to Braillie. On this page you can sign in" in x["text"] for x in http(self.base + "/api/state")[2]["said"])))

    def test_a_prompt_can_carry_a_cleaned_name_and_nothing_else(self):
        n = len(http(self.base + "/api/state")[2]["said"])
        self.assertEqual(http(self.base + "/api/prompt", {"name": "confirm_name", "who": "Sam <script>"})[0], 202)
        self.assertTrue(wait_for(lambda: any("Is your name Sam script? Say yes" in x["text"] for x in http(self.base + "/api/state")[2]["said"][n - 1:])))
        self.assertEqual(http(self.base + "/api/prompt", {"name": "go_guest_anon", "who": "ignored"})[0], 202)

    def test_the_sign_in_conversation_silences_commands_and_the_did_not_catch_that_reply(self):
        s = self.session
        s.set_mode("menu")
        self.assertEqual(http(self.base + "/api/dialogue", {"open": "yes"})[0], 400)
        self.assertEqual(http(self.base + "/api/dialogue", {})[0], 400)
        self.assertEqual(http(self.base + "/api/dialogue", {"open": True})[2], {"open": True})
        self.assertFalse(s._listening_state())  # a name that no command matches is not "I did not catch that"
        commands = getattr(s.voice, "voice", s.voice).commands  # what the voice module would run for "learn", "read", "quiz"
        for name in ("learn", "read", "quiz"):
            commands[name]()
        self.assertEqual(s.hub_mode, "menu")  # a name that sounds like a command must not start a lesson before anyone has signed in
        self.assertEqual(http(self.base + "/api/dialogue", {"open": False})[2], {"open": False})
        self.assertTrue(s._listening_state())

    def test_who_is_here_ends_the_sign_in_conversation_and_it_expires_by_itself(self):
        s = self.session
        s.dialogue(True)
        s.set_user("Ivy", "guest")
        self.assertFalse(s.dialogue_active())
        s.dialogue(True)
        with mock.patch.object(tutor.time, "time", return_value=time.time() + tutor.DIALOGUE_SECONDS + 1):
            self.assertFalse(s.dialogue_active())  # the tab was closed: the tutor is not left deaf

    def test_state_says_whether_the_tutor_is_talking_and_what_the_microphone_last_heard(self):
        s = http(self.base + "/api/state")[2]
        self.assertIn("speaking", s)
        self.assertIn("heard", s)

    def test_these_endpoints_belong_to_the_menu_tutor_only_and_to_local_pages_only(self):
        for path, body in (("/api/session", {"kind": "guest", "name": "x"}), ("/api/mode", {"mode": "learn"}), ("/api/prompt", {"name": "welcome"})):
            self.assertEqual(http(self.plain_base + path, body)[0], 404, path)
            self.assertEqual(http(self.base + path, body, origin="https://evil.example")[0], 403, path)


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class PhoneHookTests(unittest.TestCase):
    def args(self, *extra):
        ap = argparse.ArgumentParser()
        phonelink.add_phone_args(ap)
        return ap.parse_args(["--phone-camera", "--phone-host", "127.0.0.1", "--phone-port", "0", *extra])

    def test_on_ready_replaces_the_generic_announcements_and_fires_once_per_connection(self):
        said, ready = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            link, cam, server = phonelink.start_phone(self.args(), announce=said.append, on_ready=lambda: ready.append(1))
        try:
            link.verbose = False
            frame = cv2.imencode(".jpg", np.full((120, 160, 3), 100, np.uint8))[1].tobytes()
            link.push(frame)
            self.assertTrue(wait_for(lambda: ready))
            link.push(frame)
            link.set_audio_ready()
            time.sleep(0.3)
            self.assertEqual(ready, [1], "once, however many frames and sound-ready signals follow")
            self.assertEqual(said, [], "no generic 'phone connected' speech: the app speaks to the user instead")
        finally:
            server.stop()

    def test_without_on_ready_the_old_announcements_are_unchanged(self):
        said = []
        with contextlib.redirect_stdout(io.StringIO()):
            link, cam, server = phonelink.start_phone(self.args(), announce=said.append)
        try:
            link.verbose = False
            link.push(cv2.imencode(".jpg", np.full((120, 160, 3), 100, np.uint8))[1].tobytes())
            self.assertTrue(wait_for(lambda: any("Phone camera connected" in t for t in said)))
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
