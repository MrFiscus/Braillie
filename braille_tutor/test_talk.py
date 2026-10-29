"""Tests for how the tutor talks: reminders that do not repeat forever, lines dropped after the learner cuts in, "quiet", and
free speech worked out by Braillo. Run: python -m unittest test_talk -v

Voice and LLM are faked; nothing here needs a microphone, a camera or a network.
"""
import random
import threading
import time
import unittest

import learn
import sheets
import tutor
from test_tutor import FakeVoice


class CuttableVoice(FakeVoice):
    """A voice the learner can cut off: like voice_io, it counts the cuts."""

    def __init__(self):
        super().__init__()
        self.cuts = 0

    def interrupt_count(self):
        return self.cuts


class Talker:
    """A fake Braillo: respond() returns `reply`; records what it was given."""
    enabled = True

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def respond(self, text, context=None):
        self.calls.append((text, context))
        return self.reply

    def answer(self, question, context=None):
        return None


def session(voice=None, mode="letters"):
    sh = sheets.get_sheet("alphabet")
    voice = voice or CuttableVoice()
    s = tutor.TutorSession(voice, None, sh.cells, finger=lambda: None, mode=mode, rng=random.Random(1), names=sh.names)
    return s, voice


class NagTests(unittest.TestCase):
    def test_a_reminder_comes_round_less_and_less_often_then_stops(self):
        s, voice = session()
        line = "I can't see your finger."
        said_at = [t for t in range(0, 600, 5) if s.nag(line, now=float(t))]
        self.assertEqual(said_at, [0, 30, 90, 210])  # gaps of 30, 60 and 120 seconds, then quiet
        self.assertEqual(voice.said.count(line), tutor.NAG_MAX)

    def test_different_reminders_are_counted_separately(self):
        s, _ = session()
        self.assertTrue(s.nag("one", now=0.0))
        self.assertTrue(s.nag("two", now=1.0))
        self.assertFalse(s.nag("one", now=2.0))

    def test_saying_a_command_lets_reminders_start_over(self):
        s, voice = session()
        s.attach()
        s.nag("I can't see your finger.", now=0.0)
        self.assertFalse(s.nag("I can't see your finger.", now=1.0))
        voice.commands["slower"]()
        self.assertTrue(s.nag("I can't see your finger.", now=2.0))

    def test_the_lessons_send_their_reminders_through_the_host(self):
        class Host:
            def __init__(self):
                self.said, self.nagged = [], []

            def say(self, text):
                self.said.append(text)

            def nag(self, text):
                self.nagged.append(text)

        host = Host()
        j = learn.Journey(host, progress=None)
        j._nag("still waiting")
        self.assertEqual((host.said, host.nagged), ([], ["still waiting"]))
        del Host.nag
        j._nag("a host without nag just says it")
        self.assertEqual(host.said, ["a host without nag just says it"])


class LongSentenceTests(unittest.TestCase):
    def test_the_voice_leaves_long_sentences_to_braillo_only_when_braillo_is_on(self):
        class Voice(CuttableVoice):
            defer = None

            def set_defer_long_utterances(self, on):
                self.defer = on

        for ask, want in ((Talker({}), True), (None, False)):
            s, voice = session(Voice())
            s.ask = ask
            s.attach()
            self.assertIs(voice.defer, want)


class CutInTests(unittest.TestCase):
    def test_a_line_queued_before_the_learner_cut_in_is_dropped(self):
        s, voice = session()
        with s._speech:  # the tutor is talking; another line is waiting its turn
            waiting = threading.Thread(target=s.say, args=("an old reminder",))
            waiting.start()
            time.sleep(0.2)  # (it is now waiting for its turn)
            voice.cuts += 1  # the learner said "quiet"
        waiting.join(2)
        s.say("the reply to what they asked")
        self.assertEqual(voice.said, ["the reply to what they asked"])

    def test_quiet_with_nothing_after_it_says_nothing(self):
        s, voice = session()
        s.ask = Talker({"action": "none", "say": "should not be used"})
        s.heard_fn = lambda: {"n": 1, "text": "quiet", "matched": True}
        s.on_hush()
        self.assertEqual(voice.said, [])
        self.assertEqual(s.ask.calls, [])

    def test_wait_then_a_question_is_answered(self):
        s, voice = session()
        s.ask = Talker({"action": "none", "say": "D is dots one, four and five."})
        s.heard_fn = lambda: {"n": 1, "text": "wait which dots does d have", "matched": True}
        s.on_hush()
        self.assertEqual(s.ask.calls[0][0], "which dots does d have")
        self.assertEqual(voice.said, ["D is dots one, four and five."])


class FreeSpeechTests(unittest.TestCase):
    def setUp(self):
        self.s, self.voice = session()
        self.heard = {"n": 0, "text": "", "matched": False, "reason": None}
        self.s.heard_fn = lambda: self.heard
        self.s.watch_once(0.0)  # (the first look only notes what was heard before)

    def hear(self, text, n):
        self.heard = {"n": n, "text": text, "matched": False, "reason": "no command phrase matched"}

    def test_free_speech_runs_the_command_it_means(self):
        self.s.ask = Talker({"action": "help", "say": ""})
        self.hear("what am I supposed to say", 1)
        self.s.watch_once(1.0)
        self.assertIn("you can say found it", self.voice.said[-1])  # the quiz help line

    def test_a_reply_is_spoken_and_the_moment_is_passed_along(self):
        self.s.ask = Talker({"action": "none", "say": "A braille cell has six dots."})
        self.s.say("Find the letter K.")
        self.hear("how many dots are there again", 1)
        self.assertEqual(self.s.watch_once(1.0), "A braille cell has six dots.")
        context = self.s.ask.calls[0][1]
        self.assertEqual(context["you_said_last"], ["Find the letter K."])
        self.assertIn("activity", context)

    def test_an_answer_is_not_followed_by_a_command_that_says_it_again(self):
        self.s.ask = Talker({"action": "repeat", "say": "D has dots one, four and five."})
        self.hear("which dots does it have again", 1)
        self.s.watch_once(1.0)
        self.assertEqual(self.voice.said, ["D has dots one, four and five."])

    def test_asking_for_help_in_other_words_gets_the_apps_own_hint_counted_after_a_warm_word(self):
        self.s.ask = Talker({"action": "hint", "say": "No worries!"})
        self.s.on_start()  # (a quiz under way: a letter is being asked for)
        self.voice.said.clear()
        self.hear("i really cannot work this one out", 1)
        before = self.s.hints
        self.s.watch_once(1.0)
        self.assertEqual(self.voice.said[0], "No worries!")
        self.assertRegex(self.voice.said[-1], r"has \d dots?:", "then the hint itself, from the app's ladder")
        self.assertEqual(self.s.hints, before + 1, "and it counts")

    def test_a_reply_and_a_command_that_changes_something_both_happen(self):
        self.s.ask = Talker({"action": "menu", "say": "Sure."})
        self.s.hub = True
        self.hear("let's do something else", 1)
        self.s.watch_once(1.0)
        self.assertEqual(self.voice.said[0], "Sure.")
        self.assertEqual(self.s.hub_mode, "menu")
        self.assertIn("Choose an option", self.voice.said[-1])

    def test_speech_not_meant_for_the_tutor_gets_no_reply(self):
        self.s.ask = Talker({"action": "none", "say": ""})
        self.hear("did you feed the cat", 1)
        self.assertIsNone(self.s.watch_once(1.0))
        self.assertEqual(self.voice.said, [])

    def test_one_word_is_not_sent_to_braillo(self):
        self.s.ask = Talker({"action": "hint", "say": ""})
        self.hear("um", 1)
        self.s.watch_once(1.0)
        self.assertEqual(self.s.ask.calls, [])

    def test_when_braillo_fails_the_plain_reply_is_used(self):
        self.s.ask = Talker(None)
        self.s.state = "reading"
        self.hear("what is going on here", 1)
        self.assertEqual(self.s.watch_once(1.0), tutor.acc.NOT_UNDERSTOOD)

    def test_the_same_utterance_is_handled_once(self):
        self.s.ask = Talker({"action": "none", "say": "Sure."})
        self.hear("can you help me out", 1)
        self.s.watch_once(1.0)
        self.s.watch_once(1.5)
        self.assertEqual(len(self.s.ask.calls), 1)



if __name__ == "__main__":
    unittest.main()
