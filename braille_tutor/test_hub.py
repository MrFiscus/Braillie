"""Tests for the menu-driven tutor: who is here (Google keeps progress, a guest keeps nothing), the greeting, and Learn / Read / Quiz."""
import os
os.environ["BRAILLIE_NO_PHONE_BACKUP"] = "1"  # start_phone must never open a real public tunnel from a test
import argparse
import contextlib
import io
import random
import tempfile
import threading
import time
import json
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


_ANSWERED = mock.patch("learn.Journey._new_learner", return_value=False)


def setUpModule():
    """A new learner is first asked how much braille they know (test_learn.py tests that); here they have answered "new"."""
    _ANSWERED.start()


def tearDownModule():
    _ANSWERED.stop()

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
    s.page_ok_fn = lambda: True  # (the sheet is under the camera: a lesson waiting for its sheet starts. Tests of a missing sheet set it.)
    s.sheet_in_view_fn = None  # (no camera frames: which sheet is there is not made out. Tests of the wrong sheet set it.)
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
        hello = tutor.greeting("Ana")
        self.assertTrue(hello.endswith(" " + tutor.MENU_LINE) and "Ana" in hello, hello)
        self.assertIn(hello[: -len(tutor.MENU_LINE) - 1], [g.format(name="Ana") for g in tutor.GREETINGS])
        self.assertNotEqual(tutor.greeting("Ana"), hello, "never the same opening twice running")
        self.assertEqual(len(tutor.GREETINGS), 10)
        self.assertIn("Choose an option", tutor.MENU_LINE)
        self.assertIn("Learn teaches", tutor.MENU_LINE)
        self.assertIn("Read says a word", tutor.MENU_LINE)
        self.assertIn("Quiz names a letter", tutor.MENU_LINE)

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
        return [t for t in voice.said if tutor.MENU_LINE in t]  # (the opening question comes in a few ways; the choices are always the same)

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
        self.assertEqual(len(self.said(voice)), 1)
        self.assertIn("Ana", self.said(voice)[0])
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
        self.patch = mock.patch.object(tutor, "Dwell", fast_dwell)  # answering by resting takes 0.15 s here, not DWELL_SECONDS
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
        self.assertIn("the letter A. One dot, at the top-left.", voice.said[-1])
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        s.set_mode("menu")

    def test_a_new_learner_says_how_much_braille_they_know_and_starts_there(self):
        with mock.patch("learn.Journey._new_learner", lambda self: not self.progress.level):  # (as for a real new learner)
            s, voice, finger, feed = hub()
            heard = {"n": 0, "text": "", "matched": False, "reason": None}
            s.heard_fn = lambda: heard
            s.watch_once(0.0)
            s.set_mode("learn")
            self.assertEqual(s.learning_status()["phase"], "placing")
            self.assertIn("how familiar are you with braille?", voice.said[-1])
            heard.update(n=1, text="banana")  # (not an answer: asked again, briefly)
            s.watch_once(1.0)
            self.assertEqual(voice.said[-1], learn.RETRY_LEVEL)
            heard.update(n=2, text="I know the whole alphabet")
            s.watch_once(2.0)
            self.assertEqual((s.learning_status()["lesson"]["id"], s.progress.level), ("l6", "alphabet"))
            heard.update(n=3, text="can I change my level")
            s.watch_once(3.0)
            self.assertEqual(s.learning_status()["phase"], "placing")
            s.choose_level("new")  # (the same, pressed on the screen)
            self.assertEqual(s.learning_status()["target"], "a")
            with self.assertRaises(ValueError):
                s.choose_level("expert")
            s.set_mode("menu")

    def test_a_guest_is_asked_too_and_so_is_the_next_guest(self):
        """A guest's progress is never kept, so every guest is asked where to start, and one guest's answer is not the next one's."""
        with mock.patch("learn.Journey._new_learner", lambda self: not self.progress.level and not self.progress.lessons):
            s, voice, finger, feed = hub()
            s.set_user("Guest", "guest", "visit1")
            s.set_mode("learn")
            self.assertEqual(s.learning_status()["phase"], "placing")
            self.assertIn("how familiar are you with braille?", voice.said[-1])
            s.choose_level("alphabet")
            self.assertEqual(s.learning_status()["lesson"]["id"], "l6")
            s.set_mode("menu")
            s.set_user("Guest", "guest", "visit1")  # (the same guest, after a page reload: their lesson is kept)
            self.assertEqual(s.progress.level, "alphabet")
            self.assertIsNone(s.user["profile"], "nothing is saved for a guest")
            s.set_user("Guest", "guest", "visit2")  # (the next guest)
            s.set_mode("learn")
            self.assertEqual((s.learning_status()["phase"], s.progress.level), ("placing", ""))
            s.set_mode("menu")

    def test_while_a_lesson_waits_for_its_sheet_the_lesson_says_what_the_camera_saw(self):
        p = Progress()
        for lid in ("l1", "l2", "l3", "l4", "l5", "l6", "l7"):
            p.lesson_done(lid, 1.0)
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        s.journey.progress = s.progress = p
        s._learn("go_to_lesson", 7)
        self.assertEqual(s.learning_status()["phase"], "await_sheet")
        s.sheet_news("This looks like the alphabet sheet again.")
        self.assertTrue(voice.said[-1].startswith("That's the alphabet sheet. This lesson needs the numbers sheet"), voice.said[-1])
        self.assertNotIn("This looks like the alphabet sheet again.", voice.said)
        s.set_mode("read")
        s.sheet_news("This is the words sheet.")  # (an activity with no lesson waiting: said as it is)
        self.assertTrue(wait_for(lambda: voice.said[-1] == "This is the words sheet."))
        s.set_mode("menu")
        n = len(voice.said)
        s.sheet_news("This is the alphabet sheet.")  # (nothing chosen yet: not a word about the sheet)
        time.sleep(0.2)
        self.assertNotIn("This is the alphabet sheet.", voice.said[n:])

    def test_stopping_an_activity_goes_back_to_the_menu_without_a_goodbye(self):
        s, voice, finger, feed = hub()
        for mode in ("read", "learn", "quiz"):
            s.set_mode(mode)
            n = len(voice.said)
            s.on_stop()
            s.on_stop()  # (asked first; said again, that is a yes)
            wait_for(lambda: s.hub_mode == "menu")
            time.sleep(0.2)
            said = " ".join(voice.said[n:])
            self.assertNotIn("Goodbye", said, mode)
            self.assertNotIn("See you next time", said, mode)
            self.assertNotIn("click the picture", " ".join(voice.said), "a blind learner cannot click a picture")

    def test_leaving_an_activity_part_way_is_one_line_with_how_it_went(self):
        with mock.patch("learn.Journey._new_learner", return_value=False):
            for word in ("menu", "stop", "exit"):
                s, voice, finger, feed = hub()
                s.set_mode("learn")
                time.sleep(0.2)
                finger.pos = cell_pos("alphabet", "a")
                self.assertTrue(wait_for(lambda: s.learning_status()["target"] != "a"))
                finger.pos = None
                time.sleep(0.3)
                n = len(voice.said)
                leave = s.on_stop if word == "stop" else s.on_mode_menu
                leave()
                leave()  # (the first time it asks; said again, that is a yes)
                self.assertTrue(wait_for(lambda: s.hub_mode == "menu"))
                time.sleep(0.2)
                said = voice.said[n:]
                self.assertEqual(said[0], tutor.LEAVE_QUESTION)
                self.assertEqual(len(said), 2, (word, said))
                self.assertIn("You found 1 of 1 letters today.", said[1])
                self.assertIn("Say learn, read, or quiz.", said[1])

    def test_signing_out_is_one_goodbye_and_the_front_page_does_not_start_over(self):
        s, voice, finger, feed = hub()
        s.website = True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.set_user("Guest", "guest", "v1")
        self.assertTrue(wait_for(lambda: any("Guest" in t for t in voice.said)))  # (the greeting first)
        s.guided = True  # (they had asked to be talked through the site)
        n = len(voice.said)
        s.on_sign_out()
        self.assertIsNone(s.user, "the tutor knows nobody is signed in now")
        self.assertEqual(s.nav["to"], "signout")
        s.set_page("landing", ["Start learning"], arrived=True)
        time.sleep(0.3)
        said = voice.said[n:]
        self.assertEqual(len(said), 1, said)
        self.assertIn("start learning", said[0])
        self.assertNotIn(tutor.LANDING_WELCOME, said)

    def test_leaving_part_way_is_asked_first_and_no_or_silence_carries_on(self):
        with mock.patch("learn.Journey._new_learner", return_value=False):
            s, voice, finger, feed = hub()
            s.website = True
            heard = {"n": 0, "text": "", "matched": False, "reason": None}
            s.heard_fn = lambda: heard
            s.watch_once(0.0)
            s.set_mode("learn")
            self.assertEqual(s.learning_status()["phase"], "teach")
            s.on_mode_menu()  # (said by accident, mid-lesson)
            self.assertEqual((voice.said[-1], s.hub_mode), (tutor.LEAVE_QUESTION, "learn"))
            heard.update(n=1, text="no keep going")
            s.watch_once(1.0)
            self.assertEqual(s.hub_mode, "learn")
            self.assertTrue(wait_for(lambda: "Okay, let's keep going." in voice.said[-2:] and "letter A" in voice.said[-1]),
                            "and on with what they were asked")
            s.on_stop()
            heard.update(n=2, text="yes please")
            s.watch_once(2.0)
            self.assertTrue(wait_for(lambda: s.hub_mode == "menu"))
            s.set_mode("learn")
            s.on_stop()
            s._leave_asked["at"] -= tutor.LEAVE_WAIT + 1  # (no answer for a while)
            heard.update(n=3, text="yes")
            s.watch_once(3.0)
            self.assertEqual(s.hub_mode, "learn", "too late: it was taken as said by accident")
            s.pressed("on_stop")  # (the Finish key on the screen: pressed on purpose, nothing asked)
            self.assertTrue(wait_for(lambda: s.hub_mode == "menu"))
            s.set_mode("menu")

    def test_quitting_pauses_everything_asks_and_a_yes_sums_up_then_goes_home(self):
        """How quitting should feel: whatever Braillo was doing stops, he asks yes or no (and nothing else is said meanwhile), a
        no carries on, and a yes says how it went and goes back to the menu."""
        with mock.patch("learn.Journey._new_learner", return_value=False):
            s, voice, finger, feed = hub()
            heard = {"n": 0, "text": "", "matched": False, "reason": None}
            s.heard_fn = lambda: heard
            s.watch_once(0.0)
            s.set_mode("learn")
            finger.pos = cell_pos("alphabet", "a")
            self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
            finger.pos = None
            voice.commands["stop"]()  # "I wanna quit"
            self.assertEqual(voice.said[-1], tutor.LEAVE_QUESTION)
            n = len(voice.said)
            s.say("A hint that came due meanwhile.")
            s._learn("on_hint")
            self.assertEqual(len(voice.said), n, "the lesson is paused while it asks")
            heard.update(n=1, text="what about the cat")  # (neither yes nor no: asked once more)
            s.watch_once(1.0)
            self.assertEqual(voice.said[-1], tutor.LEAVE_AGAIN)
            heard.update(n=2, text="yeah i'm done")
            s.watch_once(2.0)
            self.assertTrue(wait_for(lambda: s.hub_mode == "menu"))
            self.assertTrue(wait_for(lambda: "You found 1 of 1 letters today." in voice.said[-1]))
            self.assertTrue(voice.said[-1].startswith("Okay, let's stop there."))
            self.assertIn("Say learn, read, or quiz", voice.said[-1])

    def test_a_level_said_with_a_command_word_in_it_is_the_answer(self):
        """From a real run: "let's start with new today" has "start" in it, a command, and the question was asked again."""
        with mock.patch("learn.Journey._new_learner", lambda self: not self.progress.level):
            s, voice, finger, feed = hub()
            heard = {"n": 0, "text": "", "matched": False, "reason": None}
            s.heard_fn = lambda: heard
            s.watch_once(0.0)
            s.set_mode("learn")
            self.assertEqual(s.learning_status()["phase"], "placing")
            heard.update(n=1, text="let's start with new today", matched=True)
            voice.commands["start quiz"]()
            self.assertEqual((s.progress.level, s.learning_status()["target"]), ("new", "a"))
            s.set_mode("menu")

    def test_a_late_reply_from_braillo_does_not_start_the_lesson_over(self):
        """From a real run: "learn" started the lesson (and its familiarity question) while Braillo was still answering for the menu.
        His late "say learn, read or quiz", with the action learn, started the lesson over and asked the question again."""
        s, voice, finger, feed = hub()
        s.set_user("Gina", "guest")

        class Late:
            enabled = True

            def respond(self, text, context=None, on_first=None):
                s.set_mode("learn")  # (the command ran while this reply was on its way)
                on_first("Say learn, read, or quiz, and I will take it from there.")
                return {"action": "learn", "say": "Say learn, read, or quiz, and I will take it from there."}

        s.ask = Late()
        self.assertTrue(wait_for(lambda: s._greeted_for is not None))
        time.sleep(0.1)
        voice.said.clear()
        s._talk("okay let's learn then")
        started = len(voice.said)
        self.assertEqual(started, 1, voice.said)  # (the lesson's opening: once)
        self.assertFalse(any("learn, read, or quiz, and I will" in x for x in voice.said))
        s.set_mode("learn")  # (and asked again, by a button too: still the one lesson, not started over)
        self.assertEqual(len(voice.said), started)
        s.set_mode("menu")

    def test_signing_out_is_not_undone_by_the_website_sending_them_straight_back(self):
        """From a real run: signed out by voice, the website (not yet told) sent the same guest again, who was greeted again."""
        s, voice, finger, feed = hub()
        s.website = True
        s.set_user("Gina", "guest", "visit1")
        self.assertTrue(wait_for(lambda: s._greeted_for is not None))
        s.on_sign_out()
        n = len(voice.said)
        self.assertIs(s.set_user("Gina", "guest", "visit1"), False)
        time.sleep(0.2)
        self.assertIsNone(s.user)
        self.assertEqual(len(voice.said), n, "no greeting")
        s.phone_news("I've lost the phone's camera.")
        self.assertEqual(len(voice.said), n, "the phone is nobody's news now")
        s.set_user("Ben", "guest", "visit2")  # (someone else can sign in at once)
        self.assertEqual(s.user["name"], "Ben")

    def test_the_sign_in_page_asking_for_its_welcome_means_that_is_where_they_are(self):
        """From a real run: the website's report of the sign-in page came late, so a "hi" there was answered as if on the front page
        ("You're on the front page..."). The sign-in welcome is only asked for by the sign-in page: so that is where they are."""
        s, voice, finger, feed = hub()
        s.website = True
        s.set_page("landing", ["Start learning"])
        s.speak_prompt("welcome")
        self.assertEqual(s.page, "login")
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        heard.update(n=1, text="hi there how are you")
        self.assertFalse(s._wants_talk(heard), "the sign-in page's own conversation answers that")
        s.speak_prompt("connect_phone")
        self.assertEqual(s.page, "connect-phone")

    def test_the_quiz_waits_for_its_own_sheet_and_says_when_another_is_there(self):
        """From a real run: the quiz started with the words sheet under the camera. Now it waits for the look-alikes sheet."""
        s, voice, finger, feed = hub()
        under = {"sheet": "words"}
        s.sheet_in_view_fn = lambda: under["sheet"]
        watched = []
        s.watch_sheet = watched.append
        s.set_mode("quiz")
        self.assertEqual(s.state, "await_sheet")
        self.assertIn("Get the look-alikes sheet ready", voice.said[-1])
        self.assertTrue(wait_for(lambda: any("That's the words sheet. The quiz needs the look-alikes sheet" in x for x in voice.said)))
        self.assertEqual(s.state, "await_sheet", "not on the wrong sheet")
        s.on_start()  # ("I'm ready")
        self.assertIn("I'm waiting for the look-alikes sheet", voice.said[-1])
        under["sheet"] = "lookalikes"
        self.assertTrue(wait_for(lambda: s.state == "asking"))
        self.assertTrue(any("I can see the sheet" in x for x in voice.said))
        self.assertIn("lookalikes", watched)
        self.assertIsNone(watched[-1], "the check stops once it has started")
        s.set_mode("menu")

    def test_a_page_the_camera_cannot_name_starts_on_their_word_but_a_wrong_one_does_not(self):
        s, voice, finger, feed = hub()
        under = {"sheet": "words"}
        s.sheet_in_view_fn = lambda: under["sheet"]
        s.set_mode("quiz")
        s.on_start()  # ("I'm ready", on the words sheet: no)
        self.assertEqual(s.state, "await_sheet")
        under["sheet"] = None  # (a page, but which one cannot be made out)
        s.on_start()
        self.assertEqual(s.state, "asking")
        s.set_mode("menu")

    def test_a_lesson_waits_for_its_own_sheet_not_just_any_page(self):
        s, voice, finger, feed = hub()
        under = {"sheet": "words"}
        s.sheet_in_view_fn = lambda: under["sheet"]
        s.set_mode("learn")
        self.assertEqual(s.learning_status()["phase"], "await_sheet")
        self.assertTrue(wait_for(lambda: any("That's the words sheet" in x for x in voice.said)))
        under["sheet"] = "alphabet"
        self.assertTrue(wait_for(lambda: s.learning_status()["phase"] == "teach"))
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

    def test_quiz_says_when_it_cannot_see_a_finger(self):
        s, voice, finger, feed = hub()
        self.patch_nag = mock.patch.object(tutor, "NO_FINGER_SECONDS", 0.2)
        self.patch_nag.start()
        self.addCleanup(self.patch_nag.stop)
        self.patch_again = mock.patch.object(tutor, "NO_FINGER_AGAIN", 0.2)
        self.patch_again.start()
        self.addCleanup(self.patch_again.stop)
        s.set_mode("quiz")
        finger.pos = None
        self.assertTrue(wait_for(lambda: any("can't see your finger" in t for t in voice.said), 4), voice.said[-3:])
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
        self.assertEqual(s.progress.confusions.get(f"{wrong}>{target}"), 1, "the mix-up is kept for the lessons to teach again")
        s.set_mode("menu")

    def test_with_the_website_voice_starts_nothing_before_sign_in_or_before_the_phone(self):
        linked = [False]
        s, voice, finger, feed = hub(phone=lambda: linked[0])
        s.website = True
        commands = getattr(s.voice, "voice", s.voice).commands
        n = s.nav["n"]
        commands["quiz"]()  # ("start learning" on the front page, say): to the sign-in page, which speaks for itself
        self.assertEqual(s.hub_mode, "menu")
        self.assertEqual((s.nav["n"], s.nav["to"]), (n + 1, "login"))
        s.set_page("login")
        commands["quiz"]()  # (on the sign-in page already: it says what to do)
        self.assertIn("signed in first", voice.said[-1])
        s.set_user("Sam", "guest")
        commands["quiz"]()
        self.assertEqual(s.hub_mode, "menu")
        self.assertIn("connect your phone first", voice.said[-1])
        linked[0] = True
        commands["quiz"]()
        self.assertEqual(s.hub_mode, "quiz")
        s.set_mode("menu")  # (the quiz's loop ends with it)

    def test_braillo_knows_the_page_the_camera_the_finger_and_what_he_is_waiting_for(self):
        s, voice, finger, feed = hub(phone=lambda: True)
        s.website = True
        s.set_page("connect-phone")
        ctx = s.talk_context()
        self.assertEqual(ctx["page"], "connect-phone")
        self.assertIn("scan the QR code", ctx["page_is_for"])
        self.assertIn("sign in", ctx["waiting_for"])  # nobody signed in yet
        s.set_user("Sam", "guest")
        self.assertIn("choose", s.talk_context()["waiting_for"])
        self.assertNotIn("Sam", json.dumps(s.talk_context()), "no names go to the model")
        s.set_mode("learn")
        finger.pos = cell_pos("alphabet", "a")
        ctx = s.talk_context()
        self.assertEqual(ctx["finger"], "on the letter A (dots 1)")
        self.assertIn("find the letter A", ctx["waiting_for"])
        self.assertEqual(ctx["target_letter"], "A")
        self.assertIn("letters_met", ctx["progress"])
        s.set_page("nowhere")  # (not a page it knows: ignored)
        self.assertEqual(s.page, "connect-phone")

    def test_the_front_page_asks_first_and_a_yes_gets_the_welcome(self):
        s, voice, finger, feed = hub()
        s.website = True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        said = lambda line: sum(t == line for t in voice.said)
        s.set_page("landing", ["Start learning"], arrived=True)
        self.assertTrue(wait_for(lambda: said(tutor.ARRIVAL_LINES["landing"]) == 1 and not s._speech.locked()))  # "would you like me to…?"
        s.set_page("landing", ["Start learning"], arrived=True)  # (reloaded straight away: asked once)
        time.sleep(0.2)
        self.assertEqual(said(tutor.ARRIVAL_LINES["landing"]), 1)
        heard.update(n=1, text="yes please")
        s.watch_once(1.0)
        self.assertEqual((s.guided, said(tutor.LANDING_WELCOME)), (True, 1))
        s._announced.clear()
        s.set_page("landing", arrived=True)  # back later: the welcome again (it says what to do), not the question again
        self.assertTrue(wait_for(lambda: said(tutor.LANDING_WELCOME) == 2))
        s.set_page("landing", listen=True)  # the Listen key: the whole welcome, and the controls are kept
        self.assertTrue(wait_for(lambda: said(tutor.LANDING_WELCOME) == 3))
        self.assertEqual(s.controls, ["Start learning"])
        s.on_repeat()
        self.assertTrue(wait_for(lambda: said(tutor.LANDING_WELCOME) == 4))
        self.assertEqual(said(tutor.ARRIVAL_LINES["landing"]), 1, "the question is asked once")

    def test_braillo_only_suggests_what_works_on_the_page_they_are_on(self):
        """On the front page, nobody signed in: "start learning" is the way in; "Google or guest" belongs to the sign-in page."""
        s, voice, finger, feed = hub()
        s.website = True
        s.set_page("landing", ["Start learning"])
        self.assertIn("say start learning", s.talk_context()["waiting_for"])
        self.assertIn("start learning", s.talk_context()["page_is_for"])
        s.set_page("login", ["Sign in with Google", "Continue as guest"])
        self.assertTrue(s.talk_context()["waiting_for"].startswith("them to sign in: say Google, or guest"))
        self.assertNotIn("first name", s.talk_context()["page_is_for"], "guests are not asked for a name any more")

    def test_braillo_lets_begin_on_the_front_page_begins_instead_of_explaining(self):
        class Braillo:  # (a fake Claude: "let's begin" is the start action; a plain question just gets an answer)
            enabled = True

            def respond(self, text, context=None):
                return {"action": "start", "say": "", "lesson": 0} if "begin" in text else {"action": "none", "say": "", "lesson": 0}

            def answer(self, question, context=None):
                return "Braillie teaches braille by touch."

        s, voice, finger, feed = hub()
        s.website, s.ask = True, Braillo()
        s.page = "landing"
        n = s.nav["n"]
        s.heard_fn = lambda: {"n": 1, "text": "braillo let's begin", "matched": True}
        s.on_braillo()
        self.assertEqual((s.nav["n"], s.nav["to"]), (n + 1, "login"), "straight to the sign-in page")
        s.heard_fn = lambda: {"n": 2, "text": "braillo what is this website about", "matched": True}
        s.on_braillo()
        self.assertEqual((s.nav["n"], voice.said[-1]), (n + 1, "Braillie teaches braille by touch."))

    def test_the_way_in_without_scanning_is_only_told_when_they_say_it_is_not_working(self):
        s, voice, finger, feed = hub(phone=lambda: False)
        s.website, s.phone_code_fn = True, lambda: "407268"
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.page = "connect-phone"
        self.assertTrue(all("4 0 7 2 6 8" not in v and "browser" not in v for v in tutor.PROMPTS["connect_phone"]),
                        "the usual instructions leave it out")
        heard.update(n=1, text="it won't scan")
        s.watch_once(1.0)
        self.assertTrue(voice.said[-1].startswith("Let's sort it out. If the code won't scan, open the address on the screen in your phone's "
                                                  "browser, and type the code 4 0 7 2 6 8."), "no backup link: how to type it in")

    def test_the_qr_page_is_quiet_unless_they_ask(self):
        """From a real run: it talked a lot on the QR page. There, stray speech gets no answer; "how do I do this" gets the steps."""
        s, voice, finger, feed = hub(phone=lambda: False)
        s.website = True
        s.set_user("Gina", "guest")
        s.ask = type("Talker", (), {"enabled": True, "respond": lambda self, *a, **k: {"action": "none", "say": "Chatty reply."}})()
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.page = "connect-phone"
        time.sleep(0.2)
        voice.said.clear()
        heard.update(n=1, text="okay hold on let me grab my phone")
        s.watch_once(1.0)
        heard.update(n=2, text="i think the camera is open now")
        s.watch_once(2.0)
        self.assertEqual(voice.said, [], "busy with the phone: no reply")
        heard.update(n=3, text="how do i do this")
        s.watch_once(3.0)
        self.assertEqual(voice.said[-1], tutor.PAGE_HELP["connect-phone"])

    def test_the_backup_code_is_shown_when_they_say_it_is_not_working_and_only_then(self):
        import phonelink
        s, voice, finger, feed = hub(phone=lambda: False)
        s.website, s.phone_code_fn = True, lambda: "407268"
        shown = []
        s.phone_backup_fn = lambda: shown.append(True) or True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.page = "connect-phone"
        s.watch_once(1.0)
        self.assertEqual(shown, [], "not by itself")
        heard.update(n=1, text="it's not working")
        s.watch_once(2.0)
        self.assertEqual((shown, voice.said[-1]), ([True], phonelink.BACKUP_LINE))
        heard.update(n=2, text="still not working")
        s.watch_once(3.0)
        self.assertTrue(voice.said[-1].startswith("Let's sort it out. If the code won't scan"), "the second time: type it in instead")

    def test_google_or_guest_said_on_the_front_page_opens_the_sign_in_page(self):
        s, voice, finger, feed = hub()
        s.website = True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.page = "landing"
        n = s.nav["n"]
        heard.update(n=1, text="google")
        s.watch_once(1.0)
        self.assertEqual((s.nav["n"], s.nav["to"]), (n + 1, "login"))
        s.page = "login"  # (there, the sign-in page's own conversation takes it: nothing more from here)
        heard.update(n=2, text="guest")
        s.watch_once(2.0)
        self.assertEqual(s.nav["n"], n + 1)

    def test_a_no_on_the_front_page_keeps_braillo_quiet_there(self):
        s, voice, finger, feed = hub()
        s.website = True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.set_page("landing", arrived=True)
        self.assertTrue(wait_for(lambda: tutor.ARRIVAL_LINES["landing"] in voice.said and not s._speech.locked()))
        heard.update(n=1, text="no thanks")
        s.watch_once(1.0)
        self.assertEqual((s.guided, voice.said[-1]), (False, tutor.LANDING_NO))
        s._announced.clear()
        n = len(voice.said)
        s.set_page("landing", arrived=True)
        time.sleep(0.2)
        self.assertEqual(len(voice.said), n, "quiet on the front page from now on")

    def test_its_own_name_coming_back_while_it_asks_does_not_answer_the_front_pages_question(self):
        """From a real run: "Hi, I'm Braillo... say yes, or no" came back through the laptop's microphone as the wake word, and the
        whole welcome followed over the learner's "no". The open question is answered only by a yes or a no."""
        s, voice, finger, feed = hub()
        s.website = True
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        s.set_page("landing", arrived=True)
        self.assertTrue(wait_for(lambda: tutor.ARRIVAL_LINES["landing"] in voice.said and not s._speech.locked()))
        heard.update(n=1, text="hi i'm braillo", matched=True)
        s.on_braillo()
        self.assertNotIn(tutor.LANDING_WELCOME, voice.said)
        heard.update(n=2, text="no", matched=False)
        s.watch_once(1.0)
        self.assertEqual((s.guided, voice.said[-1]), (False, tutor.LANDING_NO))
        heard.update(n=3, text="braillo", matched=True)
        s.on_braillo()  # (after the no, "if you want me, just say Braillo")
        self.assertEqual(voice.said[-1], tutor.LANDING_WELCOME)

    def test_help_fits_the_page_of_the_way_in(self):
        s, voice, finger, feed = hub()
        s.website = True
        s.set_page("login")
        s.on_help()
        self.assertIn("Say Google to sign in, or guest", voice.said[-1])
        self.assertNotIn("learn, read", voice.said[-1])
        s.set_page("connect-phone")
        s.on_help()
        self.assertIn("point it at the code on the screen", voice.said[-1])
        s.on_more_help()
        self.assertIn("point it at the code on the screen", voice.said[-1])
        self.assertNotIn("Learn teaches", voice.said[-1])

    def test_any_button_on_the_page_can_be_used_by_saying_its_name(self):
        s, voice, finger, feed = hub()
        s.website = True
        s.set_page("landing", ["Start learning", "How it works", "Download all sheets", "Display", "Dark", "Privacy policy"])
        self.assertTrue(wait_for(lambda: tutor.ARRIVAL_LINES["landing"] in voice.said and not s._speech.locked()))  # (it asks first)
        heard = {"n": 0, "text": "", "matched": False, "reason": None}
        s.heard_fn = lambda: heard
        s.watch_once(0.0)
        n = s.nav["n"]
        heard.update(n=1, text="press the download all sheets button")
        s.watch_once(1.0)
        self.assertEqual((s.nav["n"], s.nav["to"], s.nav["target"]), (n + 1, "click", "Download all sheets"))
        # a button's own name wins over a voice command with the same words ("start learning" would start a lesson)
        commands = getattr(s.voice, "voice", s.voice).commands
        heard.update(n=2, text="start learning", matched=True)
        commands["start quiz"]()
        self.assertEqual((s.nav["to"], s.nav["target"], s.hub_mode), ("click", "Start learning", "menu"))
        # what can I say, here: the page's own controls
        s.on_help()  # short: the page's main few, and how to hear the rest
        self.assertIn("Start learning, How it works, Download all sheets, or Privacy policy", voice.said[-1])
        self.assertNotIn("Dark", voice.said[-1])
        self.assertIn("more options", voice.said[-1])
        s.on_more_help()  # everything
        self.assertIn("Dark", voice.said[-1])
        self.assertIn("go back", voice.said[-1])
        # getting around
        s.on_back()
        self.assertEqual(s.nav["to"], "back")
        s.on_scroll_down()
        self.assertEqual(s.nav["to"], "scroll-down")

    def test_lessons_can_be_chosen_by_voice_from_anywhere(self):
        s, voice, finger, feed = hub()
        s.on_lesson(2)  # "lesson three", said at the menu: learn starts, at that lesson
        self.assertEqual((s.hub_mode, s.journey.lesson.id), ("learn", "l3"))
        self.assertIn("K to O", " ".join(voice.said[-2:]))
        s.on_next_lesson()
        self.assertEqual(s.journey.lesson.id, "l4")
        s.on_previous_lesson()
        s.on_previous_lesson()
        self.assertEqual(s.journey.lesson.id, "l2")
        s.on_lesson(0)
        s.on_previous_lesson()
        self.assertEqual(voice.said[-1], "This is the first lesson.")

    def test_my_page_and_sign_out_ask_the_website_to_go_there(self):
        s, voice, finger, feed = hub()
        s.set_mode("quiz")
        n = s.nav["n"]
        s.on_my_page()
        self.assertEqual((s.nav["n"], s.nav["to"], s.hub_mode), (n + 1, "account", "menu"))  # the quiz stops; the page speaks next
        s.on_sign_out()
        self.assertEqual((s.nav["n"], s.nav["to"]), (n + 2, "signout"))
        self.assertIn("You're signed out", voice.said[-1])

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
        self.assertGreaterEqual(len(s.progress.practised()), 1, "the quiz's answers count towards the letters the lessons know")
        text = " ".join(voice.said)
        self.assertIn("Choose an option", text)
        self.assertIn("Say learn, read, or quiz", text)
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
        said = len(voice.said)
        voice.commands["read"]()  # already in read: reads the word again, does not replay the intro
        self.assertTrue(wait_for(lambda: len(voice.said) > said), voice.said[-3:])
        self.assertIn("The word is cat", voice.said[-1])
        self.assertNotIn("Read mode", voice.said[-1])
        s.set_mode("menu")

    def test_switching_to_read_or_quiz_resets_the_finger_tracker(self):
        """A held tip from Learn must not keep grading after the demo sheet is swapped."""
        from fingertip import FingerTracker, Tip
        s, voice, finger, feed = hub()
        feed.tracker = FingerTracker()
        feed.tracker.position = (11.0, 22.0)
        feed.tracker.tip = Tip(40, 30, 0.9, (40, 50), 2.0)
        feed.tracker._streak = 9
        s.set_mode("read")
        self.assertIsNone(feed.tracker.position)
        self.assertIsNone(feed.tracker.tip)
        self.assertEqual(feed.tracker._streak, 0)
        feed.tracker.position = (33.0, 44.0)
        s.set_mode("quiz")
        self.assertIsNone(feed.tracker.position)
        self.assertEqual(feed.sheet_name, "lookalikes")
        s.set_mode("menu")

    def test_stop_leaves_read_and_quiz_even_when_the_adaptive_planner_is_on(self):
        """With the adaptive planner on, a plain _finish() only ended the round: "stop" said "Round complete" and stayed in the mode, however
        many times it was said."""
        from test_tutor import PlanningClient
        for mode, sheet in (("read", "words"), ("quiz", "lookalikes")):
            s, voice, finger, feed = hub()
            s.adaptive_planner = tutor.AdaptiveTargetPlanner(PlanningClient())
            s.scan = lambda sheet=sheet: SHEETS[sheet].cells
            s.set_user("Sam", "guest")
            s.set_mode(mode)
            self.assertEqual(s.hub_mode, mode)
            s.voice.commands["stop"]()  # ("do you want to stop?")
            self.assertIn(tutor.LEAVE_QUESTION, voice.said)
            s.voice.commands["stop"]()  # (said again: yes)
            self.assertTrue(wait_for(lambda: s.hub_mode == "menu", 5), f"{mode}: stop did not leave it: {voice.said[-3:]}")
            self.assertEqual((s.mode, s.state), ("menu", "menu"))
            self.assertNotIn("Round complete", " ".join(voice.said), mode)
            said = len(voice.said)
            for _ in range(3):  # and saying it again at the menu just repeats the menu, it does not start anything
                s.voice.commands["stop"]()
            self.assertEqual((s.hub_mode, s.state), ("menu", "menu"))
            self.assertTrue(all("learn, read, or quiz" in t for t in voice.said[said:]), voice.said[said:])
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

    def test_stop_in_a_mode_gives_how_it_went_and_the_menu_in_one_line(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        finger.pos = cell_pos("alphabet", "a")
        self.assertTrue(wait_for(lambda: s.learning_status()["target"] == "b"))
        voice.commands["stop"]()
        voice.commands["stop"]()  # (asked first: said again, that is a yes)
        self.assertEqual((s.hub_mode, s.state), ("menu", "menu"))
        self.assertIn("You found 1 of 1 letters today.", voice.said[-1])
        self.assertIn("Say learn, read, or quiz", voice.said[-1])
        self.assertIn("Choose an option", voice.said[-1])

    def test_the_voice_commands_switch_modes(self):
        s, voice, finger, feed = hub()
        voice.commands["quiz"]()
        self.assertEqual(s.hub_mode, "quiz")
        voice.commands["menu"]()  # (mid-quiz: asked first)
        voice.commands["menu"]()
        self.assertEqual(s.hub_mode, "menu")
        voice.commands["learn"]()
        self.assertEqual((s.hub_mode, feed.sheet_name), ("learn", "alphabet"))
        voice.commands["read"]()
        self.assertEqual((s.hub_mode, feed.sheet_name), ("read", "words"))
        voice.commands["menu"]()  # (mid-reading: asked first)
        voice.commands["menu"]()
        self.assertEqual(s.hub_mode, "menu")

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
        self.assertEqual(len(feed.stable), len(SHEETS["words"].cells))
        self.assertTrue(all(c["locked"] for c in feed.stable))
        self.assertEqual(feed.reader.sheet, SHEETS["words"].cells)


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class SpokenAnswerTests(unittest.TestCase):
    def test_a_letter_named_on_its_own_is_an_answer(self):
        cases = {"E": "e", "is this E?": "e", "it's the letter bee": "b", "Is it see": "c", "that's G": "g", "okay it's D here": "d",
                 "the letter double you": "w", "I think it's F": "f"}
        for said, want in cases.items():
            self.assertEqual(tutor.spoken_letter(said), want, said)
        for said in ("and it's not falling then it lets go", "see you later", "what is this letter", ""):
            self.assertIsNone(tutor.spoken_letter(said), said)

    def test_pointing_words_mean_found_it(self):
        import voice_io
        for said in ("this one", "Is this it?", "It's here.", "right here", "okay that one"):
            self.assertEqual(voice_io._match_command(voice_io._normalize_transcript(said)), "found it", said)


class LessonByNameTests(unittest.TestCase):
    def test_a_lesson_is_asked_for_by_number_or_name(self):
        cases = {"lesson three": 2, "go to lesson 2": 1, "let's do K to O": 2, "the look-alikes lesson": 5, "second lesson": 1,
                 "do U to Z": 4}
        for said, want in cases.items():
            self.assertEqual(tutor.lesson_named(said), want, said)
        for said in ("K to O", "lesson nine", "what is the letter K", ""):  # (a name alone may be about the letters, not a lesson)
            self.assertIsNone(tutor.lesson_named(said), said)


class ProfileLineTests(unittest.TestCase):
    def test_the_progress_is_said_in_plain_words(self):
        self.assertEqual(tutor.progress_line({"learned": 3, "streak": 2, "sessions": 4, "practised": 9}),
                         "You know 3 of 26 letters. You have practised 2 days in a row, over 4 sessions.")
        self.assertEqual(tutor.progress_line({"learned": 0, "streak": 1, "sessions": 1, "practised": 2}),
                         "You do not know any letters yet. You have practised 1 day in a row, over 1 session.")
        self.assertIn("not had a lesson yet", tutor.progress_line({}))

    def test_the_letters_are_listed_known_then_still_learning(self):
        self.assertEqual(tutor.letters_line({}, ["a", "b"], ["a", "b", "c"]), "You know A and B. You are still learning C. 23 letters are still to try.")
        self.assertEqual(tutor.letters_line({}, [], ["z"]), "You are still learning Z. 25 letters are still to try.")
        self.assertEqual(tutor.letters_line({}, [], []), "You have not practised any letters yet.")


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
        self.assertTrue(wait_for(lambda: any("Ana" in x["text"] and tutor.MENU_LINE in x["text"] for x in http(self.base + "/api/state")[2]["said"])))

    def test_a_reloaded_website_with_nobody_signed_in_makes_the_tutor_forget_an_old_guest(self):
        """From a real run: a guest from before a reload stayed in the tutor, so the sign-in page was talked over as if someone was in."""
        http(self.base + "/api/session", {"kind": "guest", "name": "Gina"})
        said = len(http(self.base + "/api/state")[2]["said"])
        code, _, body = http(self.base + "/api/session", {"kind": "none"})
        self.assertEqual(code, 200)
        self.assertIsNone(body["hub"]["user"])
        self.assertEqual(body["hub"]["mode"], "menu")
        self.assertEqual(http(self.base + "/api/session", {"kind": "none"})[0], 200, "nobody to forget: still fine")
        time.sleep(0.3)
        self.assertFalse(any("signed out" in x["text"] for x in http(self.base + "/api/state")[2]["said"][said:]), "forgotten quietly")

    def test_the_level_buttons_endpoint_answers_the_level_question(self):
        """From the final check: the Learn page's level buttons posted to /api/level, which answered 404, so they did nothing."""
        with mock.patch("learn.Journey._new_learner", lambda j: not j.progress.level):
            http(self.base + "/api/session", {"kind": "guest", "name": "Lev", "profile": "lev1"})
            http(self.base + "/api/mode", {"mode": "learn"})
            self.assertTrue(wait_for(lambda: (http(self.base + "/api/state")[2]["learning"] or {}).get("phase") == "placing"))
            self.assertEqual(http(self.base + "/api/level", {"level": "dance"})[0], 400)
            self.assertEqual(http(self.base + "/api/level", {"level": "new"})[0], 202)
            self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["learning"]["phase"] != "placing"))
            http(self.base + "/api/mode", {"mode": "menu"})

    def test_the_mode_endpoint_switches_and_returns(self):
        self.assertEqual(http(self.base + "/api/mode", {"mode": "dance"})[0], 400)
        self.assertEqual(http(self.base + "/api/mode", {})[0], 400)
        self.assertEqual(http(self.base + "/api/mode", {"mode": "quiz"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["hub"]["mode"] == "quiz"))
        s = http(self.base + "/api/state")[2]
        # (the video shows the alphabet sheet: the quiz waits for the look-alikes sheet rather than starting on the wrong one)
        self.assertEqual((s["config"]["sheet"], s["tutor"]["state"]), ("lookalikes", "await_sheet"))
        self.assertEqual(http(self.base + "/api/mode", {"mode": "menu"})[0], 202)
        self.assertTrue(wait_for(lambda: http(self.base + "/api/state")[2]["hub"]["mode"] == "menu"))

    def test_the_lines_heard_at_every_start_come_in_a_few_versions_and_never_twice_running(self):
        for name in ("welcome", "welcome_back", "ask_name", "go_guest", "go_guest_anon", "connect_phone"):
            self.assertEqual(len(tutor.PROMPTS[name]), 10, name)
            said = [tutor.next_prompt(name) for _ in range(12)]
            self.assertTrue(all(a != b for a, b in zip(said, said[1:])), f"{name} repeated itself")
            self.assertGreater(len(set(said)), 1)
        self.assertEqual(tutor.next_prompt("retry_back"), tutor.PROMPTS["retry_back"])  # a one-line reply stays one line
        planned = tutor.planned_prompt("welcome")
        self.assertIn(planned, tutor.common_lines())  # only the version said next is prepared ahead...
        self.assertEqual(sum(1 for v in tutor.PROMPTS["welcome"] if v in tutor.common_lines()), 1)  # ...not all of them
        self.assertEqual(tutor.next_prompt("welcome"), planned)  # and it is the one said

    def test_the_prompt_endpoint_speaks_only_what_it_is_allowed_to(self):
        self.assertEqual(http(self.base + "/api/prompt", {"name": "sing me a song"})[0], 400)
        self.assertEqual(http(self.base + "/api/prompt", {})[0], 400)
        self.assertEqual(http(self.base + "/api/prompt", {"name": "welcome"})[0], 202)
        welcomes = tutor.PROMPTS["welcome"]
        self.assertTrue(wait_for(lambda: any(x["text"] in welcomes for x in http(self.base + "/api/state")[2]["said"])))

    def test_a_prompt_can_carry_a_cleaned_name_and_nothing_else(self):
        n = len(http(self.base + "/api/state")[2]["said"])
        self.assertEqual(http(self.base + "/api/prompt", {"name": "confirm_name", "who": "Sam <script>"})[0], 202)
        self.assertTrue(wait_for(lambda: any("Is your name Sam script? Say yes" in x["text"] for x in http(self.base + "/api/state")[2]["said"][n - 1:])))
        self.assertEqual(http(self.base + "/api/prompt", {"name": "go_guest_anon", "who": "ignored"})[0], 202)

    def test_your_page_reads_out_your_own_progress_from_the_tutors_record(self):
        self.assertEqual(http(self.base + "/api/prompt", {"name": "account", "who": "Sam"})[0], 202)
        self.assertEqual(http(self.base + "/api/prompt", {"name": "account_letters"})[0], 202)
        lines = lambda: [x["text"] for x in http(self.base + "/api/state")[2]["said"]]  # noqa: E731  (the last few lines said)
        self.assertTrue(wait_for(lambda: any("Sam" in t and "letters" in t for t in lines())), lines())
        said = lines()
        self.assertFalse(any("{" in t for t in said), said)  # every blank filled in

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
        self.assertFalse(s._listening_state())  # back at the menu, which never says "I did not catch that" to the room
        s.hub_mode, s.state = "read", "reading"
        try:
            self.assertTrue(s._listening_state())  # ...but once an activity runs, closing the conversation gives listening back
        finally:
            s.hub_mode, s.state = "menu", "menu"

    def test_the_mute_key_turns_the_microphone_off_and_back_on(self):
        s = self.session
        self.assertEqual(http(self.base + "/api/mic", {"on": "no"})[0], 400)
        self.assertEqual(http(self.base + "/api/mic", {"on": False})[2], {"on": False})
        try:
            state = http(self.base + "/api/state")[2]
            self.assertTrue(state["muted"])
            self.assertIsNone(state["heard"])  # the pages show the microphone as off
            commands = getattr(s.voice, "voice", s.voice).commands
            s.set_mode("menu")
            commands["quiz"]()
            self.assertEqual(s.hub_mode, "menu")  # muted: a word heard does nothing
            s.hub_mode, s.state = "read", "reading"
            self.assertFalse(s._listening_state())  # and there is no "I did not catch that"
        finally:
            s.hub_mode, s.state = "menu", "menu"
            self.assertEqual(http(self.base + "/api/mic", {"on": True})[2], {"on": True})
        self.assertFalse(http(self.base + "/api/state")[2]["muted"])

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
