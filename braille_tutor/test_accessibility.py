"""Tests for accessibility.py and its place in the tutor: telling the learner when the sheet is lost, when they were not understood, what
they can say, and letting them set the pace. Time is passed in by the tests."""
import json
import random
import threading
import time
import unittest
from unittest import mock

import accessibility as acc
import audio_speed
import learn
import sheets
import tutor
import tutor_server
from accessibility import Awareness, Hearing, Settings, help_text
from test_hub import SHEETS, cell_pos, fast_dwell, hub, wait_for
from test_learn_session import learn_session
from test_server import http
from test_tutor import WC, FakeVoice


class AwarenessTests(unittest.TestCase):
    def run_it(self, a, seq):
        """seq: [(seconds, page_ok, active)] -> messages by time"""
        out = []
        for t, ok, active in seq:
            m = a.update(t, ok, active)
            if m:
                out.append((t, m))
        return out

    def test_it_says_so_after_a_moment_not_at_the_first_flicker(self):
        a = Awareness()
        self.assertEqual(self.run_it(a, [(0, False, True), (1, False, True), (2.9, False, True)]), [])
        self.assertEqual(self.run_it(a, [(3.1, False, True)]), [(3.1, acc.LOST)])

    def test_a_brief_loss_that_recovers_is_never_mentioned(self):
        a = Awareness()
        self.assertEqual(self.run_it(a, [(0, False, True), (2, False, True), (2.5, True, True), (10, True, True)]), [])

    def test_while_still_lost_it_repeats_now_and_then_but_not_constantly(self):
        a = Awareness()
        out = self.run_it(a, [(t / 2, False, True) for t in range(0, 140)])  # 70 seconds lost
        self.assertEqual([m for _, m in out], [acc.LOST, acc.STILL_LOST, acc.STILL_LOST])
        self.assertGreaterEqual(out[1][0] - out[0][0], 25)

    def test_it_says_when_the_sheet_is_back_only_once_it_is_steadily_back(self):
        a = Awareness()
        self.run_it(a, [(0, False, True), (5, False, True)])  # told at 5
        self.assertEqual(self.run_it(a, [(6, True, True), (6.5, False, True)]), [], "a flicker back is not 'back'")
        out = self.run_it(a, [(7, True, True), (8, True, True), (8.6, True, True), (9, True, True)])
        self.assertEqual(out, [(8.6, acc.FOUND)])
        self.assertEqual(self.run_it(a, [(20, True, True)]), [], "and it does not say it again")

    def test_nothing_is_said_when_no_activity_is_running_and_it_starts_fresh_afterwards(self):
        a = Awareness()
        self.assertEqual(self.run_it(a, [(t, False, False) for t in range(0, 40)]), [], "at the menu the sheet does not matter")
        self.assertEqual(self.run_it(a, [(40, False, True)]), [], "the clock starts when the activity does, not before")
        self.assertEqual(self.run_it(a, [(44.5, False, True)]), [(44.5, acc.LOST)])
        a2 = Awareness()
        self.run_it(a2, [(0, False, True), (5, False, True)])
        a2.update(6, False, False)  # activity ended
        self.assertEqual(self.run_it(a2, [(7, True, True), (10, True, True)]), [], "nothing about a sheet from an old activity")


class HearingTests(unittest.TestCase):
    def heard(self, n, text, matched=False, reason=Hearing.UNMATCHED):
        return {"n": n, "text": text, "matched": matched, "reason": reason}

    def test_something_said_but_not_understood_gets_a_gentle_reply(self):
        h = Hearing()
        self.assertIsNone(h.update(self.heard(0, ""), 0, True), "history before we started is ignored")
        self.assertEqual(h.update(self.heard(1, "what is the weather"), 1, True), acc.NOT_UNDERSTOOD)

    def test_unclear_speech_is_asked_to_be_repeated(self):
        h = Hearing()
        h.update(self.heard(0, ""), 0, True)
        self.assertEqual(h.update(self.heard(1, "mumble mumble", reason=Hearing.UNCLEAR), 1, True), acc.NOT_CLEAR)

    def test_understood_commands_and_stray_syllables_get_no_reply(self):
        h = Hearing()
        h.update(self.heard(0, ""), 0, True)
        self.assertIsNone(h.update(self.heard(1, "next page", matched=True, reason=None), 1, True))
        self.assertIsNone(h.update(self.heard(2, "uh"), 2, True))
        self.assertIsNone(h.update(self.heard(3, "hm"), 3, True))

    def test_a_room_full_of_talk_does_not_make_it_nag(self):
        h = Hearing(cooldown=20)
        h.update(self.heard(0, ""), 0, True)
        said = [h.update(self.heard(i, "we should go and get dinner soon"), float(i), True) for i in range(1, 60)]
        self.assertEqual(sum(1 for m in said if m), 3, "once every 20 seconds at most, over a minute")

    def test_it_stays_quiet_when_nothing_is_listening_or_the_learner_switched_it_off(self):
        h = Hearing()
        h.update(self.heard(0, ""), 0, True)
        self.assertIsNone(h.update(self.heard(1, "some talk here"), 1, active=False))
        self.assertIsNone(h.update(self.heard(2, "some more talk"), 30, True, enabled=False))
        self.assertEqual(h.update(self.heard(3, "and more talk"), 60, True), acc.NOT_UNDERSTOOD)

    def test_no_voice_module_or_nothing_heard_yet(self):
        h = Hearing()
        self.assertIsNone(h.update(None, 0, True))
        self.assertIsNone(h.update({}, 1, True))


class SettingsTests(unittest.TestCase):
    def test_valid_changes_apply_and_invalid_ones_change_nothing(self):
        s = Settings()
        s.update({"speech_speed": 0.8, "pace": "relaxed", "tones": False})
        self.assertEqual(s.to_dict(), {"speech_speed": 0.8, "pace": "relaxed", "tones": False, "hearing_feedback": True})
        before = s.to_dict()
        for bad in ({"speech_speed": 0.2}, {"speech_speed": 9}, {"speech_speed": "fast"}, {"speech_speed": True}, {"pace": "hurried"},
                    {"tones": "yes"}, {"hearing_feedback": 1}, {"volume": 11}, {"speech_speed": 1.0, "pace": "bogus"}):
            with self.assertRaises(ValueError, msg=bad):
                s.update(bad)
            self.assertEqual(s.to_dict(), before, f"{bad} must change nothing, not even the valid part")

    def test_relaxed_means_a_longer_rest_and_later_help(self):
        s = Settings()
        self.assertEqual((s.dwell_scale, s.hint_scale), (1.0, 1.0))
        s.update({"pace": "relaxed"})
        self.assertEqual((s.dwell_scale, s.hint_scale), (acc.RELAXED_DWELL, acc.RELAXED_HINTS))


class HelpTests(unittest.TestCase):
    def test_help_matches_where_you_are(self):
        self.assertIn("learn, read, or quiz", help_text("menu", "menu", "menu"))
        self.assertIn("hint for help", help_text("learn", "learn", "learning"))
        self.assertIn("next to skip a question", help_text("quiz", "letters", "asking"))
        self.assertIn("rest a finger on a word", help_text("read", "read", "reading"))
        self.assertIn("Rest a finger on any cell", help_text(None, "explore", "exploring"))
        for h in (help_text("menu", "menu", "menu"), help_text("learn", "learn", "learning")):
            self.assertIn("slower or faster", h)
            self.assertIn("take your time", h)


class PageSeenTests(unittest.TestCase):
    """page_seen_now: really in view, not remembered. Fake page sources, so no camera is needed."""

    class Feed:
        def __init__(self, src=None, page_ok=True, identify_until=0.0):
            self.page_src, self.page_ok, self.identify_until = src, page_ok, identify_until

    class Src:
        def __init__(self, source="", last_seen=None, paper=None):
            self.source = source
            if last_seen is not None:
                self.last_seen = last_seen
            if paper is not None:
                self.paper_fallback = paper

    class Paper:
        def __init__(self, seen):
            self.last_seen = seen

    def test_not_ok_or_holding_or_none_is_not_seen(self):
        F, S = self.Feed, self.Src
        self.assertFalse(tutor.page_seen_now(F(S("markers"), page_ok=False)))
        self.assertFalse(tutor.page_seen_now(F(S("holding"))))
        self.assertFalse(tutor.page_seen_now(F(S("none"))))

    def test_markers_and_tracking_count_as_seen(self):
        F, S = self.Feed, self.Src
        self.assertTrue(tutor.page_seen_now(F(S("markers"))))
        self.assertTrue(tutor.page_seen_now(F(S("tracking"))))

    def test_the_paper_finder_is_seen_only_while_its_edges_are_recent(self):
        F, S, P = self.Feed, self.Src, self.Paper
        self.assertTrue(tutor.page_seen_now(F(S("paper", paper=P(time.time() - 0.4)))))
        self.assertFalse(tutor.page_seen_now(F(S("paper", paper=P(time.time() - 3.0)))), "held from memory for a few seconds: not seen")
        self.assertTrue(tutor.page_seen_now(F(S("paper", paper=P(0.0)))), "never used: nothing to judge by")
        self.assertFalse(tutor.page_seen_now(F(S("", last_seen=time.time() - 4.0))), "a page finder with no source, judged by its own last_seen")
        self.assertTrue(tutor.page_seen_now(F(S("", last_seen=time.time() - 0.3))))

    def test_between_pages_and_no_page_source_never_complain(self):
        F, S, P = self.Feed, self.Src, self.Paper
        self.assertTrue(tutor.page_seen_now(F(S("holding"), identify_until=99.0)))
        self.assertTrue(tutor.page_seen_now(F(None)), "no page source at all (a fixed test setup): trust page_ok")
        self.assertTrue(tutor.page_seen_now(F(S("paper", paper=P(time.time() - 9)), page_ok=True, identify_until=5.0)))


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class SessionTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(tutor, "Dwell", fast_dwell)
        p.start()
        self.addCleanup(p.stop)

    def test_every_recognised_command_first_plays_a_small_heard_you_sound(self):
        s, voice, *_ = hub()
        s.settings.tones = True
        s.earcons.enabled = True
        voice.commands["quiz"]()
        self.assertEqual(s.earcons.played[0], "locked")
        n = len(s.earcons.played)
        voice.commands["menu"]()
        self.assertEqual(s.earcons.played[n], "locked")
        s.apply_settings({"tones": False})
        n = len(s.earcons.played)
        voice.commands["learn"]()
        self.assertEqual(len(s.earcons.played), n, "sounds off: silent")
        s.set_mode("menu")

    def test_help_speaks_what_can_be_said_here(self):
        s, voice, *_ = hub()
        voice.commands["help"]()
        self.assertIn("learn, read, or quiz", voice.said[-1])
        s.set_mode("learn")
        voice.commands["help"]()
        self.assertIn("hint for help", voice.said[-1])
        s.set_mode("menu")

    def test_slower_and_faster_change_the_speech_speed_and_stop_at_the_ends(self):
        s, voice, *_ = hub()
        voice.commands["slower"]()
        self.assertAlmostEqual(s.settings.speech_speed, 0.85)
        self.assertIn("more slowly", voice.said[-1])
        for _ in range(5):
            voice.commands["slower"]()
        self.assertAlmostEqual(s.settings.speech_speed, 0.6)
        self.assertIn("as slowly as I can go", voice.said[-1])
        for _ in range(8):
            voice.commands["faster"]()
        self.assertAlmostEqual(s.settings.speech_speed, 1.5)
        self.assertIn("as fast as I go", voice.said[-1])

    def test_take_your_time_slows_the_lesson_down_and_normal_pace_undoes_it(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        self.assertEqual((s.journey.dwell.seconds, s.journey.hint_scale), (learn.DWELL_SECONDS, 1.0))
        voice.commands["take your time"]()
        self.assertEqual(s.settings.pace, "relaxed")
        self.assertAlmostEqual(s.journey.dwell.seconds, learn.DWELL_SECONDS * acc.RELAXED_DWELL)
        self.assertEqual(s.journey.hint_scale, acc.RELAXED_HINTS)
        s.set_mode("menu")
        s.set_mode("learn")  # a lesson started later is relaxed too
        self.assertEqual(s.journey.hint_scale, acc.RELAXED_HINTS)
        voice.commands["normal pace"]()
        self.assertEqual((s.settings.pace, s.journey.hint_scale), ("normal", 1.0))
        s.set_mode("menu")

    def test_relaxed_pace_really_delays_the_help_by_itself(self):
        s, voice, finger, feed = hub()
        s.set_mode("learn")
        j = s.journey
        j.on_next()  # skip to a letter to ask for
        base = j.last_activity
        s.apply_settings({"pace": "relaxed"})
        j.tick(base + learn.HINT_AFTER[0] + 1)
        self.assertEqual(j.hints, 0, "12 s is not enough when relaxed")
        j.tick(base + learn.HINT_AFTER[0] * acc.RELAXED_HINTS + 1)
        self.assertEqual(j.hints, 1)
        s.set_mode("menu")

    def test_the_lost_sheet_is_announced_only_during_an_activity(self):
        s, voice, finger, feed = hub()
        seen = [False]
        s.page_ok_fn = lambda: seen[0]
        s.heard_fn = None
        self.assertIsNone(s.watch_once(0.0))
        self.assertIsNone(s.watch_once(10.0), "at the menu the sheet does not matter")
        s.set_mode("learn")
        n = len(voice.said)
        s.watch_once(20.0)
        self.assertIsNone(s.watch_once(22.5))
        self.assertEqual(s.watch_once(23.5), acc.LOST)
        self.assertEqual(voice.said[-1], acc.LOST)
        seen[0] = True
        s.watch_once(26.0)
        self.assertEqual(s.watch_once(28.0), acc.FOUND)
        s.set_mode("menu")

    def test_it_does_not_speak_over_itself(self):
        s, voice, *_ = hub()
        s.page_ok_fn = lambda: False
        s.heard_fn = None
        s.set_mode("learn")
        s.watch_once(0.0)
        with s._speech:  # the tutor is talking
            self.assertIsNone(s.watch_once(10.0))
        s.set_mode("menu")

    def test_something_not_understood_is_answered_through_the_watcher(self):
        s, voice, *_ = hub()
        heard = [{"n": 0, "text": "", "matched": False, "reason": None}]
        s.heard_fn = lambda: heard[0]
        s.page_ok_fn = None
        s.set_user("Ana", "guest")
        self.assertTrue(wait_for(lambda: any("What do you want" in t for t in voice.said)))
        s.watch_once(0.0)
        heard[0] = {"n": 1, "text": "what is the weather", "matched": False, "reason": Hearing.UNMATCHED}
        self.assertEqual(s.watch_once(1.0), acc.NOT_UNDERSTOOD)
        self.assertIn("Say help", voice.said[-1])
        s.apply_settings({"hearing_feedback": False})
        heard[0] = {"n": 2, "text": "another sentence here", "matched": False, "reason": Hearing.UNMATCHED}
        self.assertIsNone(s.watch_once(60.0))

    def test_the_real_voice_module_reports_what_was_heard(self):
        voice, _ = tutor.load_teammate_modules(mock=True)
        s = tutor.TutorSession(voice, WC, SHEETS["alphabet"].cells, lambda: None, mode="menu")
        self.assertIsNone(s._voice_heard(), "the mock voice has nothing to report")

    def test_a_live_voice_module_reports_what_it_heard_through_the_session(self):
        class Live:  # what voice_io looks like when it is listening for real (not the mock)
            MOCK_MODE = False
            heard = {"n": 3, "text": "some talk", "matched": False, "reason": "no command phrase matched"}

            @classmethod
            def listener_status(cls):
                return {"heard": cls.heard, "connected": True}
        s = tutor.TutorSession(Live, WC, SHEETS["alphabet"].cells, lambda: None, mode="menu")
        self.assertEqual(s._voice_heard()["text"], "some talk")
        wrapped = type("W", (), {"voice": Live})()  # the server wraps the module (RecordingVoice keeps it as .voice)
        s2 = tutor.TutorSession(wrapped, WC, SHEETS["alphabet"].cells, lambda: None, mode="menu")
        self.assertEqual(s2._voice_heard()["n"], 3)

    def test_the_watcher_thread_starts_once_and_keeps_going(self):
        s, *_ = hub()
        s.start_watching()
        first = s._watcher
        s.start_watching()
        self.assertIs(s._watcher, first)
        self.assertTrue(first.is_alive())


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_hub import ServerTests as HubServer
        HubServer.setUpClass.__func__(cls)  # the same camera-less menu-driven server

    @classmethod
    def tearDownClass(cls):
        from test_hub import ServerTests as HubServer
        HubServer.tearDownClass.__func__(cls)

    def test_settings_can_be_read_and_changed_over_the_api(self):
        code, _, s = http(self.base + "/api/settings")
        self.assertEqual((code, s["speech_speed"], s["pace"], s["hearing_feedback"]), (200, 1.0, "normal", True))  # (tones: this test server has them off)
        self.assertEqual(sorted(s), ["hearing_feedback", "pace", "speech_speed", "tones"])
        code, _, s = http(self.base + "/api/settings", {"speech_speed": 0.8, "pace": "relaxed"})
        self.assertEqual((code, s["speech_speed"], s["pace"]), (200, 0.8, "relaxed"))
        self.assertEqual(http(self.base + "/api/state")[2]["settings"]["speech_speed"], 0.8)
        http(self.base + "/api/settings", {"speech_speed": 1.0, "pace": "normal"})

    def test_bad_settings_are_refused_with_a_reason_and_change_nothing(self):
        for bad in ({"speech_speed": 5}, {"pace": "hurried"}, {"nonsense": 1}, {"tones": "no"}):
            code, _, body = http(self.base + "/api/settings", bad)
            self.assertEqual(code, 400, bad)
            self.assertIn("error", body)
        self.assertEqual(http(self.base + "/api/settings")[2]["speech_speed"], 1.0)

    def test_settings_are_for_local_pages_only(self):
        self.assertEqual(http(self.base + "/api/settings", {"tones": False}, origin="https://evil.example")[0], 403)

    def test_the_new_commands_are_available_by_button_too(self):
        cmds = http(self.base + "/api/state")[2]["config"]["commands"]
        for c in ("help", "slower", "faster", "relaxed", "normal"):
            self.assertIn(c, cmds)
        self.assertEqual(http(self.base + "/api/command", {"command": "help"})[0], 202)


if __name__ == "__main__":
    unittest.main()
