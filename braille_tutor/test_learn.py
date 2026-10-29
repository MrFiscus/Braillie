"""Tests for learn.py: the teaching text, resting-to-answer, and whole lessons run against a fake finger, voice and clock."""
import random
import unittest

import learn
import sheets
from learn import (DOTS, Dwell, HINT_AFTER, Journey, LESSONS, MAX_TRIES, LESSON_BY_ID, cap, compare_pair, contrast, dot_words, letter_of_dots,
                   name, relation, welcome_first_time)
from progress import Progress

ALPHABET = sheets.get_sheet("alphabet")
LOOKALIKES = sheets.get_sheet("lookalikes")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeCoach:
    enabled = True

    def __init__(self):
        self.prefetched, self.aids = [], {}
        self._n = 0

    def prefetch(self, cells):
        self.prefetched.append(cells)

    def aid(self, name_):
        return self.aids.get(name_)

    def next_praise(self):
        self._n += 1
        return "Brilliant work."


class Host:
    """Everything the journey needs from outside, recorded."""

    def __init__(self, sheet="alphabet"):
        self.said, self.tones, self.requests, self.explore_ticks, self.explore_resets, self.finished = [], [], [], 0, 0, False
        self.finger_pos, self.sheet = None, sheet
        self.sheets = {"alphabet": ALPHABET, "lookalikes": LOOKALIKES, "numbers": sheets.get_sheet("numbers")}

    # -- the host interface --
    def say(self, text):
        self.said.append(text)

    def tone(self, kind):
        self.tones.append(kind)

    def finger(self):
        return self.finger_pos

    def cells(self):
        return self.sheets[self.sheet].cells

    def known_cells(self):
        return self.sheets[self.sheet].cells

    def sheet_name(self):
        return self.sheet

    def request_sheet(self, name_):
        self.requests.append(name_)

    def explore_reset(self):
        self.explore_resets += 1

    def explore_tick(self, now):
        self.explore_ticks += 1

    def finish(self):
        self.finished = True

    def symbol_key(self, cell):
        """(Only when the test asks for it: what the sheet has printed there, as the real tutor knows.)"""
        if not self.knows_symbols:
            return None
        return sheets.get_sheet(self.sheet).symbol(cell).key

    knows_symbols = False

    # -- helpers for the tests --
    def pos(self, letter):
        if self.knows_symbols:
            c = next(c for c in self.cells() if self.symbol_key(c) == letter)
        else:
            c = next(c for c in self.cells() if frozenset(c["dots"]) == DOTS[letter])
        return c["x"], c["y"]

    def last(self, n=1):
        return " | ".join(self.said[-n:])


def make(progress=None, coach=None, sheet="alphabet", seed=1):
    clock, host = Clock(), Host(sheet)
    if progress is None:  # (a new learner who has said they are new to braille: the question itself is tested in LevelTests)
        progress = Progress(clock=lambda: clock.t)
        progress.level, progress.start_lesson = "new", "l1"
    p = progress
    j = Journey(host, p, coach=coach, rng=random.Random(seed), clock=clock)
    return j, host, clock, p


def run(j, host, clock, seconds, step=0.1):
    for _ in range(int(round(seconds / step))):
        clock.t += step
        j.tick(clock.t)


def answer(j, host, clock, letter, settle=None):
    """The learner puts a finger on `letter` and rests it there for long enough to count as an answer."""
    from learn import DWELL_SECONDS
    host.finger_pos = host.pos(letter)
    run(j, host, clock, DWELL_SECONDS + 0.2 if settle is None else settle)


def leave(j, host, clock):
    host.finger_pos = (500.0, 500.0)
    run(j, host, clock, 0.3)
    host.finger_pos = None
    run(j, host, clock, 0.2)


def target_letter(j):
    return j.target


class TeachingTextTests(unittest.TestCase):
    def test_dot_words(self):
        self.assertEqual(dot_words("a"), "One dot, at the top-left.")
        self.assertEqual(dot_words("d"), "Three dots: top-left, top-right and middle-right.")
        self.assertEqual(dot_words("g"), "Four dots: top-left, middle-left, top-right and middle-right.")

    def test_a_letter_is_explained_from_one_the_learner_knows(self):
        self.assertEqual(relation("b", ["a"]), "It's A with one more dot, at the middle-left.")
        self.assertEqual(relation("k", list("abcde")), "It's A with one more dot, at the bottom-left.")
        self.assertEqual(relation("w", list("abcdefghij")), "It's J with one more dot, at the bottom-right.")
        self.assertIsNone(relation("a", list("bcde")), "nothing is a start of A")
        self.assertIsNone(relation("b", []))
        self.assertIsNone(relation("b", ["b", "q"]), "not itself, and q (12345) is not a start of b")

    def test_the_nearest_known_letter_with_the_fewest_extra_dots_is_chosen(self):
        self.assertIn("It's C ", relation("d", list("abc")))  # d (145) grows from c (14), one dot; a (1) would need two
        self.assertIn("It's A ", relation("d", ["a"]))
        self.assertIn("two more dots", relation("d", ["a"]))

    def test_every_letter_a_to_z_can_be_described_and_its_relation_is_true(self):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self.assertTrue(dot_words(letter).endswith("."))
            rel = relation(letter, [l for l in "abcdefghijklmnopqrstuvwxyz" if l < letter])
            if rel:  # whatever it says, the named letter must really be a start of this one
                named = rel.split("It's ")[1][0].lower()
                self.assertLess(DOTS[named], DOTS[letter], (letter, rel))

    def test_contrast_says_what_the_finger_is_on_and_what_differs(self):
        self.assertEqual(contrast("f", "d"), "That's the letter F. The letter D has a dot at the middle-right, and no dot at the middle-left.")
        self.assertEqual(contrast("a", "b"), "That's the letter A. The letter B has a dot at the middle-left.")
        self.assertEqual(contrast("b", "a"), "That's the letter B. The letter A has no dot at the middle-left.")
        self.assertEqual(contrast("a", "g"), "That's the letter A. The letter G has dots at the middle-left, top-right and middle-right.")
        self.assertEqual(contrast("g", "a"), "That's the letter G. The letter A has no dots at the middle-left, top-right and middle-right.")

    def test_capital_letters_keep_their_case_for_the_voice(self):
        self.assertEqual(cap(name("d")), "The letter D")
        for text in (contrast("f", "d"), compare_pair("d", "f"), dot_words("d")):
            self.assertNotIn("letter d", text)
        self.assertIn("The letter D:", compare_pair("d", "f"))

    def test_first_time_welcome_explains_the_cell(self):
        text = welcome_first_time()
        for part in ("room for six dots", "two columns of three", "find it on the sheet with your finger"):
            self.assertIn(part, text)

    def test_letter_of_dots(self):
        self.assertEqual(letter_of_dots([1, 4, 5]), "d")
        self.assertIsNone(letter_of_dots([1, 2, 3, 4, 5, 6]))


class LessonDataTests(unittest.TestCase):
    def test_lessons_are_sound(self):
        seen = []
        for lesson in LESSONS:
            self.assertTrue(all(l in learn.SYMBOL_DOTS for l in lesson.letters), lesson.id)
            self.assertEqual(len(lesson.letters), len(set(lesson.letters)), lesson.id)
            sheet = sheets.get_sheet(lesson.sheet)
            printed = {sheet.symbol(c).key for c in sheet.cells}
            self.assertTrue(set(lesson.letters) <= printed, f"{lesson.id} needs symbols its sheet does not have")
            if lesson.sheet == "alphabet":
                seen += list(lesson.letters)
        self.assertEqual("".join(seen), "abcdefghijklmnopqrstuvwxyz", "the alphabet lessons cover a to z once, in order")
        self.assertEqual(len({l.id for l in LESSONS}), len(LESSONS))

    def test_the_insights_are_true_braille(self):
        add3 = lambda l: DOTS[l] | {3}
        self.assertTrue(all(DOTS[k] == add3(a) for k, a in zip("klmno", "abcde")))
        self.assertTrue(all(DOTS[p] == add3(f) for p, f in zip("pqrst", "fghij")))
        add36 = lambda l: DOTS[l] | {3, 6}
        self.assertTrue(all(DOTS[u] == add36(a) for u, a in zip("uvxyz", "abcde")))
        self.assertEqual(DOTS["w"], DOTS["j"] | {6})
        self.assertTrue(all(DOTS[l] <= {1, 2, 4, 5} for l in "abcdefghij"))
        pairs = "akblcmdneofpgqhr"
        self.assertTrue(all(DOTS[pairs[i + 1]] == DOTS[pairs[i]] | {3} for i in range(0, len(pairs), 2)))


class DwellTests(unittest.TestCase):
    def test_answers_once_after_resting_then_waits_for_the_finger_to_leave(self):
        d = Dwell(seconds=1.0)
        self.assertIsNone(d.update((10, 10), 0.0))
        self.assertIsNone(d.update((11, 10), 0.5))
        self.assertEqual(d.update((10, 11), 1.1), (10, 11))
        for t in (1.2, 2.0, 5.0):
            self.assertIsNone(d.update((10, 10), t), "does not answer again while it stays")
        self.assertIsNone(d.update((30, 10), 6.0), "moved away: re-armed but must rest again")
        self.assertIsNone(d.update((30, 10), 6.5))
        self.assertEqual(d.update((30, 10), 7.1), (30, 10))

    def test_moving_restarts_the_timing_and_losing_the_finger_resets(self):
        d = Dwell(seconds=1.0)
        d.update((0, 0), 0.0)
        self.assertIsNone(d.update((20, 0), 0.9))  # moved: timing starts again from here
        self.assertIsNone(d.update((20, 0), 1.5))
        self.assertEqual(d.update((20, 0), 1.95), (20, 0))
        d.update(None, 3.0)
        d.update(None, 3.6)  # gone longer than the dropout gap: the rest is forgotten
        self.assertIsNone(d.update((20, 0), 3.7), "after the finger was lost it must rest again, even on the same spot")
        self.assertEqual(d.update((20, 0), 4.8), (20, 0))


    def test_a_few_millimetres_of_tracking_jitter_does_not_restart_the_rest(self):
        """A fingertip on a cell wobbles a little under the camera; that must not reset the 3-second answer timer."""
        d = Dwell(seconds=1.0)
        d.update((0, 0), 0.0)
        self.assertIsNone(d.update((8, 0), 0.5))  # 8 mm: inside STILL_MM (half a cell), still counting
        self.assertEqual(d.update((7, 1), 1.05), (7, 1))

    def test_a_brief_tracking_dropout_does_not_restart_the_rest(self):
        d = Dwell(seconds=1.0)
        d.update((0, 0), 0.0)
        d.update(None, 0.3)  # one missed frame
        self.assertEqual(d.update((1, 0), 1.05), (1, 0))


class AdvancedLessonTests(unittest.TestCase):
    """After the letters: numbers, then capitals and punctuation, on the numbers sheet, in the same adaptive journey."""

    def numbers_lesson(self):
        p = Progress()
        for lesson in LESSONS[:6]:
            p.lesson_done(lesson.id, 1.0)
        j, host, clock, _ = make(progress=p, sheet="numbers")
        host.knows_symbols = True  # (as the real tutor does: what is printed where)
        j.session_started = True
        j.on_start()
        return j, host, clock, p

    def test_after_the_letters_come_numbers_taught_as_letters_after_the_number_sign(self):
        j, host, clock, p = self.numbers_lesson()
        self.assertEqual((j.lesson.id, j.target), ("l7", "#"))
        self.assertIn("the number sign. Four dots", host.last())
        answer(j, host, clock, "#")
        leave(j, host, clock)
        self.assertEqual(j.target, "1")
        self.assertIn("the number 1", host.last())
        self.assertIn("after a number sign, and has the dots of the letter A", host.last())

    def test_a_digit_is_graded_by_what_is_printed_there_not_mistaken_for_its_letter(self):
        j, host, clock, p = self.numbers_lesson()
        answer(j, host, clock, "#")
        leave(j, host, clock)
        answer(j, host, clock, "1")  # (the cell with dot 1 only: on this sheet it is the number 1, not the letter A)
        self.assertIn("That's 1.", host.last())
        self.assertEqual(j.target, "2")

    def test_punctuation_follows_the_numbers(self):
        p = Progress()
        for lesson in LESSONS[:7]:
            p.lesson_done(lesson.id, 1.0)
        j, host, clock, _ = make(progress=p, sheet="numbers")
        host.knows_symbols = True
        j.session_started = True
        j.on_start()
        self.assertEqual((j.lesson.id, j.target), ("l8", "^"))
        self.assertIn("the capital sign", host.last())


class WordedHintTests(unittest.TestCase):
    """The app decides what a hint gives away (and counts it); Claude, if on, puts it for this learner."""

    def test_the_hint_knows_what_they_touched_and_how_it_differs_and_the_app_counts_it(self):
        j, host, clock, _ = make()
        seen = []
        host.hint_words = lambda hint, facts: seen.append((hint, facts)) or "You were on A. D is A plus two dots on the right."
        j.on_start()
        answer(j, host, clock, "a")
        leave(j, host, clock)
        answer(j, host, clock, "b")
        leave(j, host, clock)
        answer(j, host, clock, "c")
        leave(j, host, clock)
        self.assertEqual(j.target, "d")
        answer(j, host, clock, "a")  # (a wrong touch: A, when D was asked for)
        leave(j, host, clock)
        j.on_hint()
        hint, facts = seen[-1]
        self.assertEqual((facts["looking_for"], facts["just_touched"], facts["hint_number"]), ("the letter D", "the letter A", 1))
        self.assertEqual(facts["how_it_differs_from_what_they_touched"], "Unlike A, it has dots at the top-right and middle-right.")
        self.assertEqual(host.last(), "You were on A. D is A plus two dots on the right.")
        self.assertEqual(j.hints, 1, "counted")

    def test_without_claude_the_hint_is_the_apps_own(self):
        j, host, clock, _ = make()
        host.hint_words = lambda hint, facts: None
        j.on_start()
        j.on_hint()
        self.assertTrue(host.last().startswith("The letter A has one dot, at the top-left."), host.last())


class SheetSeenTests(unittest.TestCase):
    """While a lesson waits for its sheet, what the camera makes of the page is said by the lesson, once, in its own words."""

    def waiting(self):
        p = Progress()
        for lesson in LESSONS[:7]:
            p.lesson_done(lesson.id, 1.0)
        j, host, clock, _ = make(progress=p)  # (the alphabet sheet is on the desk; lesson 8 needs the numbers sheet)
        j.on_start()
        self.assertEqual((j.phase, j.lesson.id), ("await_sheet", "l8"))
        return j, host, clock

    def test_the_wrong_sheet_is_named_with_the_one_that_is_needed(self):
        j, host, clock = self.waiting()
        n = len(host.said)
        j.sheet_seen("alphabet")
        self.assertEqual(len(host.said), n + 1)
        self.assertEqual(host.said[-1], "That's the alphabet sheet. This lesson needs the numbers sheet, the one with four bumps in "
                                        "its top right corner.")
        j.sheet_seen(None)
        self.assertTrue(host.said[-1].startswith("I can't tell which sheet that is yet. This lesson needs the numbers sheet"))
        j.sheet_seen("numbers")  # (the right one: nothing to say here, the lesson begins on its own)
        self.assertEqual(len(host.said), n + 2)

    def test_when_the_camera_can_tell_sheets_apart_the_reminder_leaves_it_to_the_camera(self):
        j, host, clock = self.waiting()
        host.identifies_sheets = lambda: True
        n, asked = len(host.said), len(host.requests)
        run(j, host, clock, learn.REMIND_EVERY + 1)
        self.assertEqual(len(host.said), n, "no second voice: what the camera sees is said instead (sheet_seen)")
        self.assertGreater(len(host.requests), asked, "but it does look again")


class LevelTests(unittest.TestCase):
    """Before a new learner's first lesson: how much braille do they know already? The lessons start from there."""

    def fresh(self, sheet="alphabet"):
        clock = Clock()
        j, host, clock, p = make(progress=Progress(clock=lambda: clock.t), sheet=sheet)
        return j, host, clock, p

    def test_a_new_learner_is_asked_first_and_nothing_is_taught_until_they_answer(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        self.assertEqual((j.phase, j.target), ("placing", None))
        self.assertIn("how familiar are you with braille?", host.last())
        j.on_start()  # ("start" again: the question again, briefly)
        self.assertEqual(j.phase, "placing")
        self.assertEqual(host.said[-1], learn.ASK_LEVEL_AGAIN)

    def test_new_starts_at_the_beginning_with_how_braille_works(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        j.on_level("new")
        self.assertEqual((j.lesson.id, j.target, p.level), ("l1", "a", "new"))
        self.assertIn("start at the very beginning", host.last())
        self.assertIn("First, how braille works", host.last())
        self.assertEqual(host.last().count("start"), 2, "Then we'll start ... Let's start with: not 'before we start' as well")

    def test_knowing_the_alphabet_starts_at_the_look_alikes_and_asks_for_that_sheet(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        j.on_level("alphabet")
        self.assertEqual((j.lesson.id, j.phase, p.level, p.start_lesson), ("l6", "await_sheet", "alphabet", "l6"))
        self.assertIn("you've learned it before. We'll start at lesson 6,", host.last())
        self.assertEqual(host.last().count("Look-alikes"), 1, "the lesson is named once, not twice")
        self.assertIn("look-alikes sheet", host.last())
        self.assertEqual(p.next_lesson(LESSONS).id, "l6")
        self.assertEqual(Progress.from_dict(p.to_dict()).level, "alphabet")

    def test_the_earlier_letters_are_practised_too_so_what_they_said_is_checked(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        j.on_level("some")
        self.assertEqual(j.lesson.id, "l3")
        self.assertEqual(set(j.taught), set("abcdefghij"))

    def test_next_skips_the_question_and_starts_from_the_beginning(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        j.on_next()
        self.assertEqual((j.lesson.id, p.level), ("l1", "new"))

    def test_no_answer_brings_the_question_back_after_a_while(self):
        j, host, clock, p = self.fresh()
        j.on_start()
        n = len(host.said)
        run(j, host, clock, learn.REMIND_EVERY + 1)
        self.assertGreater(len(host.said), n)
        self.assertEqual(host.said[-1], learn.ASK_LEVEL)

    def test_someone_who_has_had_lessons_is_not_asked(self):
        p = Progress()
        p.lesson_done("l1", 1.0)
        j, host, clock, _ = make(progress=p)
        j.on_start()
        self.assertNotEqual(j.phase, "placing")

    def test_the_level_can_be_changed_in_the_middle_of_a_lesson(self):
        j, host, clock, p = make()
        j.on_start()
        self.assertEqual(j.target, "a")
        j.ask_level()
        self.assertEqual((j.phase, j.target), ("placing", None))
        self.assertEqual(host.last(), learn.ASK_LEVEL_AGAIN, "not 'before we start': they have started")
        j.on_level("numbers")
        self.assertEqual((j.lesson.id, p.level), ("l8", "numbers"))

    def test_someone_who_started_further_on_is_pointed_back_to_the_first_lessons_at_the_end(self):
        p = Progress()
        p.level, p.start_lesson = "numbers", "l8"
        p.lesson_done("l8", 1.0)
        j, host, clock, _ = make(progress=p)
        j.on_start()
        self.assertIn("finished every lesson from where you started. Say lesson one", host.last())

    def test_a_first_hint_while_teaching_does_not_repeat_what_was_just_said(self):
        j, host, clock, _ = make()
        j.on_start()
        answer(j, host, clock, "a")
        leave(j, host, clock)
        self.assertIn("It's A with one more dot, at the middle-left", host.last())  # (B is introduced by how it grows from A)
        j.on_hint()
        self.assertNotIn("It's A with one more dot", host.last())
        self.assertIn("Feel along each row", host.last())

    def test_the_warm_up_says_how_a_letter_differs_from_its_mix_up_without_naming_the_wrong_one_as_this(self):
        self.assertEqual(learn.unlike("d", "a"), "Unlike D, it has no dots at the top-right and middle-right.")

    def test_signs_are_not_counted_as_letters(self):
        self.assertEqual(learn._letters(["a", "b"]), " letters")
        self.assertEqual(learn._letters(["^", "."]), "")
        self.assertEqual(learn._letters(["a", "1"]), "")

    def test_what_people_say_is_understood(self):
        for said, level in [("I'm new", "new"), ("nothing at all", "new"), ("not sure", "new"), ("beginner", "new"),
                            ("I know a few letters", "some"), ("some", "some"), ("just a little", "some"),
                            ("alphabet", "alphabet"), ("I know all the letters", "alphabet"), ("a to z", "alphabet"),
                            ("I know the alphabet but no numbers", "alphabet"), ("numbers", "numbers"),
                            ("letters and numbers too", "numbers"), ("I'm advanced", "numbers"), ("banana", None)]:
            self.assertEqual(learn.level_of(said), level, said)
        with self.assertRaises(ValueError):
            make()[0].on_level("expert")


class FirstLessonTests(unittest.TestCase):
    def test_a_first_time_learner_is_welcomed_then_taught_the_first_letter(self):
        j, host, clock, p = make()
        j.on_start()
        text = " ".join(host.said)
        self.assertIn("room for six dots", text)  # the first-ever welcome
        self.assertIn("your first five letters, A to E", text)
        self.assertIn("just four dots to feel for", text)  # the insight
        self.assertEqual(j.phase, "teach")
        self.assertEqual(j.target, "a")
        self.assertEqual(len(host.said), 1)  # the welcome, the lesson and its first letter: one line, no gaps between them
        self.assertTrue(host.last().endswith("Let's start with the letter A. One dot, at the top-left. Find it, and rest your finger on it."))
        self.assertEqual(host.tones, ["ready"])
        self.assertEqual(p.sessions, 1)

    def test_resting_in_the_letter_square_counts_even_off_the_centre(self):
        """The green ring in A's box must register A, not only a hit on the exact cell centre."""
        j, host, clock, p = make()
        j.on_start()
        cell = next(c for c in host.cells() if frozenset(c["dots"]) == DOTS["a"])
        host.finger_pos = (cell["x"] + cell["w"] / 2 - 0.3, cell["y"] + cell["h"] / 2 - 0.3)
        run(j, host, clock, 3.2)
        self.assertEqual(j.target, "b", host.last(3))

    def test_a_whole_lesson_teach_practise_recap_and_the_next_lesson_offered(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":  # teach: each letter is found by resting a finger on it
            self.assertEqual(j.target, letter)
            answer(j, host, clock, letter)
            leave(j, host, clock)
        self.assertEqual(j.phase, "practice", host.last(3))
        self.assertIn("Now let's practise", " ".join(host.said))
        asked = []
        while j.phase == "practice":
            asked.append(j.target)
            answer(j, host, clock, j.target)
            leave(j, host, clock)
        self.assertEqual(sorted(asked), list("abcde"), "each letter asked once, in a fresh order")
        self.assertNotEqual("".join(asked), "abcde", "not simply the same order")
        # it went well, so: a real word to spell, letter by letter
        self.assertEqual(j.phase, "word")
        word = j.word
        self.assertIn(f"let's spell a real word, {word.upper()}", host.last())
        spelled = []
        while j.phase == "word":
            spelled.append(j.target)
            answer(j, host, clock, j.target)
            leave(j, host, clock)
        self.assertEqual("".join(spelled), word)
        self.assertEqual(j.phase, "recap")
        self.assertIn(f"You just spelled {word.upper()} in braille", host.said[-1])
        self.assertIn("Lesson complete", host.said[-1])
        self.assertIn("5 of 5", host.said[-1])
        self.assertIn("You've met 5 of the 26 letters so far", host.said[-1])
        self.assertIn("say next for F to J", host.said[-1])
        self.assertEqual(host.tones.count("correct"), 10 + len(word))
        self.assertEqual(host.tones[-1], "done")
        self.assertTrue(p.lesson_mastered("l1"))
        self.assertTrue(all(p.box(l) >= 1 for l in "abcde"), "found clean in practice: each moved up a box")
        self.assertEqual(len(p.practised()), 5)
        # and "next" carries on to the next lesson
        j.on_next()
        self.assertEqual((j.phase, j.lesson.id, j.target), ("teach", "l2", "f"))

    def test_letters_that_went_wrong_are_taught_again_at_the_start_of_the_next_lesson(self):
        p = Progress()
        p.lesson_done("l1", 1.0)  # lesson 1 passed, but C was missed and D was mixed up with F twice (in a lesson or the quiz)
        for l in "abe":
            p.record(l, True)
        p.record("c", False)
        p.record("d", True)
        p.record_confusion("f", "d")
        p.record_confusion("f", "d")
        j, host, clock, p = make(progress=p)
        j.on_start()
        self.assertEqual((j.phase, j.lesson.id), ("warmup", "l2"), host.last(3))
        self.assertIn("a quick warm-up with a few that tripped you up last time: C and D", " ".join(host.said))
        self.assertEqual(j.target, "c", "the most troublesome first: missed beats found")
        answer(j, host, clock, "c")
        leave(j, host, clock)
        self.assertEqual(j.target, "d")
        self.assertIn("It's been getting mixed up with F", host.last(), "taught with the letter it was confused with")
        answer(j, host, clock, "d")
        leave(j, host, clock)
        self.assertEqual((j.phase, j.target), ("teach", "f"), "then the new letters, as usual")
        self.assertEqual(p.box("c"), 1, "the warm-up counts as practice: C found cleanly moved up")

    def test_a_lesson_with_nothing_gone_wrong_starts_straight_away(self):
        p = Progress()
        p.lesson_done("l1", 1.0)
        for l in "abcde":
            p.record(l, True)
        j, host, clock, p = make(progress=p)
        j.on_start()
        self.assertEqual((j.phase, j.target), ("teach", "f"))

    def test_the_letters_that_went_wrong_come_first_in_the_next_lessons_practice(self):
        p = Progress()
        p.lesson_done("l1", 1.0)
        for l in "abde":
            p.record(l, True)
        p.record("c", False)
        j, host, clock, p = make(progress=p)
        j.on_start()
        self.assertEqual((j.phase, j.target), ("warmup", "c"))
        j.on_next()  # skipped in the warm-up too: still not known
        for letter in "fghij":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        self.assertEqual(j.phase, "practice")
        asked = []
        while j.phase == "practice":
            asked.append(j.target)
            answer(j, host, clock, j.target)
            leave(j, host, clock)
        self.assertIn("I've mixed in the ones that went wrong before", " ".join(host.said))
        self.assertIn("c", asked, "the letter that went wrong is asked for again in the practice")
        self.assertTrue(set("fghij") <= set(asked))

    def test_the_teaching_of_each_letter_builds_on_the_ones_before(self):
        j, host, clock, p = make()
        j.on_start()
        answer(j, host, clock, "a")
        # the praise and the next letter are one line (no gap between them), and A's dots are not said again
        self.assertIn("That's A.", host.last())
        self.assertIn("the letter B. Two dots: top-left and middle-left. It's A with one more dot, at the middle-left.", host.last())
        self.assertNotIn("One dot, at the top-left", host.last())
        leave(j, host, clock)
        answer(j, host, clock, "b")
        leave(j, host, clock)
        answer(j, host, clock, "c")
        leave(j, host, clock)
        self.assertIn("It's C with one more dot, at the middle-right.", host.last())  # D is taught from C, the closest start


class WrongAnswerTests(unittest.TestCase):
    def start_practice(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        return j, host, clock, p

    def test_a_wrong_touch_says_what_it_is_and_how_it_differs(self):
        j, host, clock, p = self.start_practice()
        want = j.target
        wrong = next(l for l in "abcde" if l != want)
        answer(j, host, clock, wrong)
        said = host.last()
        self.assertIn(f"That's the letter {wrong.upper()}.", said)
        self.assertIn(f"The letter {want.upper()}", said)
        self.assertEqual(host.tones[-1], "wrong")
        self.assertEqual(j.tries, 1)
        self.assertEqual(j.target, want, "still asking for the same letter")
        self.assertEqual(p.confusions.get(f"{wrong}>{want}"), 1)

    def test_moving_to_the_right_letter_after_a_mistake_counts_without_leaving_first(self):
        """After 'that's L', resting on B must register — the learner should not have to lift off and wait again."""
        j, host, clock, p = make()
        j.on_start()
        answer(j, host, clock, "a")
        leave(j, host, clock)
        self.assertEqual(j.target, "b")
        answer(j, host, clock, "l")
        self.assertEqual(j.tries, 1)
        answer(j, host, clock, "b")  # no leave() in between: they slide to the right cell
        self.assertEqual(j.target, "c", host.last(3))
        self.assertEqual(host.tones[-1], "correct")

    def test_already_on_the_right_letter_when_the_correction_finishes_counts(self):
        """The correction speech blocks the rest-timer; if they found B during it, take it when speech ends."""
        j, host, clock, p = make()
        j.on_start()
        answer(j, host, clock, "a")
        leave(j, host, clock)
        self.assertEqual(j.target, "b")
        original_say = host.say

        def say_and_move(text):
            original_say(text)
            if j.tries >= 1 and j.target == "b":
                host.finger_pos = host.pos("b")

        host.say = say_and_move
        answer(j, host, clock, "l")
        self.assertEqual(j.target, "c", host.last(3))

    def test_a_right_answer_after_a_wrong_one_counts_but_not_as_clean(self):
        j, host, clock, p = self.start_practice()
        want = j.target
        wrong = next(l for l in "abcde" if l != want)
        answer(j, host, clock, wrong)
        leave(j, host, clock)
        answer(j, host, clock, want)
        self.assertEqual(host.tones[-1], "correct")
        s = p.letters[want]
        self.assertEqual((s.attempts, s.correct, s.clean, s.box), (1, 1, 0, 0), "found, but it took a second go: stays in box 0")

    def test_three_wrong_touches_shows_the_place_and_moves_on(self):
        j, host, clock, p = self.start_practice()
        want = j.target
        wrongs = [l for l in "abcde" if l != want]
        for i in range(MAX_TRIES):
            answer(j, host, clock, wrongs[i % len(wrongs)])
            leave(j, host, clock)
        said = " | ".join(host.said[-3:])
        self.assertIn("Let me show you", said)
        self.assertRegex(said, r"row \d, column \d")
        self.assertNotEqual(j.target, want, "moved on to the next letter")
        s = p.letters[want]
        self.assertEqual((s.attempts, s.correct, s.box), (1, 0, 0))
        self.assertGreaterEqual(host.tones.count("wrong"), MAX_TRIES)

    def test_resting_on_blank_paper_is_gently_pointed_out_once(self):
        from learn import DWELL_SECONDS
        j, host, clock, p = self.start_practice()
        n = len(host.said)
        host.finger_pos = (5.0, 5.0)  # nowhere near a cell
        run(j, host, clock, DWELL_SECONDS + 1.3)
        said = host.said[n:]
        self.assertEqual(sum("don't feel any braille" in s for s in said), 1, said)
        run(j, host, clock, 2.0)
        self.assertEqual(sum("don't feel any braille" in s for s in host.said[n:]), 1, "not repeated while the finger rests there")
        self.assertEqual(j.tries, 0, "not a wrong answer")


class HintTests(unittest.TestCase):
    def test_help_arrives_by_itself_and_gets_more_specific(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        n = len(host.said)
        want = j.target
        host.finger_pos = None  # (finger off the page: nothing to help with yet, so no hint however long it takes)
        run(j, host, clock, HINT_AFTER[2] + 5)
        self.assertEqual(j.hints, 0)
        host.finger_pos = (-20.0, -20.0)  # (searching: on the paper, between cells)
        run(j, host, clock, HINT_AFTER[0] + 1)
        first = host.said[n:]
        self.assertTrue(any("Here's a hint" in s and learn.has_dots(want) in s for s in first), first)
        run(j, host, clock, HINT_AFTER[1] + 1)
        run(j, host, clock, HINT_AFTER[2] + 1)
        text = " | ".join(host.said[n:])
        self.assertRegex(text, r"row \d, column \d", "in the end it says where it is")
        self.assertEqual(j.hints, 3)
        run(j, host, clock, 60)
        self.assertEqual(j.hints, 3, "no endless nagging")

    def test_asking_for_a_hint_gives_the_next_tier(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        want = j.target
        j.on_hint()
        self.assertIn(learn.has_dots(want), host.last())
        j.on_hint()
        j.on_hint()
        self.assertRegex(host.last(), r"row \d, column \d")

    def test_a_hint_means_a_right_answer_is_not_clean(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        want = j.target
        j.on_hint()
        answer(j, host, clock, want)
        s = p.letters[want]
        self.assertEqual((s.correct, s.clean, s.hints, s.box), (1, 0, 1, 0))

    def test_the_ai_coachs_memory_aid_is_used_when_it_has_arrived(self):
        coach = FakeCoach()
        coach.aids["the letter B"] = "B is A with a friend underneath."
        j, host, clock, p = make(coach=coach)
        j.on_start()
        self.assertEqual(len(coach.prefetched), 1, "asked for in the background as the lesson starts")
        self.assertEqual([c["name"] for c in coach.prefetched[0]], [name(l) for l in "abcde"])
        answer(j, host, clock, "a")
        leave(j, host, clock)
        self.assertEqual(j.target, "b")
        j.on_hint()
        self.assertEqual(host.last(), "B is A with a friend underneath.")


class CommandTests(unittest.TestCase):
    def test_repeat_found_it_and_next(self):
        j, host, clock, p = make()
        j.on_start()
        j.on_repeat()
        self.assertIn("The letter A. One dot, at the top-left.", host.last())
        host.finger_pos = None
        j.on_found_it()
        self.assertIn("can't see your finger", host.last())
        host.finger_pos = host.pos("a")
        j.on_found_it()  # immediately, without waiting to rest
        self.assertEqual((host.tones[-1], j.target), ("correct", "b"))
        j.on_next()  # skip B
        self.assertEqual(j.target, "c")
        self.assertIn("skipping", host.said[-2] + host.said[-1])

    def test_before_a_lesson_starts_the_commands_point_the_way(self):
        j, host, clock, p = make()
        for command in (j.on_hint, j.on_found_it):
            command()
            self.assertIn("Say start", host.last())

    def test_stop_gives_a_recap_saves_nothing_it_should_not_and_finishes(self):
        j, host, clock, p = make()
        j.on_start()
        answer(j, host, clock, "a")
        j.on_stop()
        self.assertTrue(host.finished)
        self.assertEqual(j.phase, "done")
        self.assertIn("You found 1 of 1", host.last())
        j2, host2, _, _ = make()
        j2.on_stop()
        self.assertIn("Come back soon", host2.last())

    def test_resume_after_stop_or_at_a_recap_carries_on(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        j.on_repeat()  # mid-practice: asks again
        self.assertIn("Find the letter", host.last())


class RecapAndRetryTests(unittest.TestCase):
    def test_a_shaky_lesson_goes_over_the_tricky_letters_and_is_not_mastered_until_they_are_solid(self):
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        # practice: get two wrong three times (given up on), the rest right
        fails = {}
        while j.phase == "practice" and j.round_no == 1:
            t = j.target
            if len(fails) < 3 and t not in fails:
                fails[t] = True
                for wrong in [l for l in "abcde" if l != t][:MAX_TRIES]:
                    answer(j, host, clock, wrong)
                    leave(j, host, clock)
            else:
                answer(j, host, clock, t)
                leave(j, host, clock)
        self.assertEqual(j.round_no, 2)
        self.assertIn("Let's go over the tricky ones", " ".join(host.said))
        self.assertEqual(sorted(j.queue + [j.target]), sorted(fails))
        while j.phase in ("practice", "word"):  # (put right in round 2, it went well enough for the word at the end)
            answer(j, host, clock, j.target)
            leave(j, host, clock)
        self.assertEqual(j.phase, "recap")
        # 2 found first time in round 1 + the 3 fixed in round 2 only count if clean: they were clean this time
        self.assertTrue(p.lessons["l1"]["last"] > 0)

    def test_a_lesson_the_learner_struggled_with_is_never_called_complete(self):
        """Being told "Lesson complete!" after failing would be the worst possible feedback."""
        j, host, clock, p = make()
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        guard = 0
        while j.phase == "practice" and guard < 40:  # every letter is given up on, in both rounds
            guard += 1
            t = j.target
            for wrong in [l for l in "abcde" if l != t][:MAX_TRIES]:
                answer(j, host, clock, wrong)
                leave(j, host, clock)
        self.assertEqual(j.phase, "recap")
        text = " ".join(host.said[-3:])
        self.assertNotIn("Lesson complete", text)
        self.assertIn("need more practice", text)
        self.assertIn("say next to try this lesson again", host.said[-1])
        self.assertFalse(p.lesson_mastered("l1"))
        self.assertEqual(host.tones[-1], "ready", "the finishing fanfare is only for a lesson that was really learned")
        self.assertNotEqual(host.tones[-1], "done")
        j.on_next()
        self.assertEqual(j.lesson.id, "l1", "so it comes round again")

    def test_an_unmastered_lesson_comes_round_again_instead_of_moving_on(self):
        j, host, clock, p = make()
        p.lesson_done("l1", 0.2)
        j.session_started = True
        j.on_start()
        self.assertEqual(j.lesson.id, "l1")

    def test_after_the_last_lesson_it_suggests_practice_and_exploring(self):
        j, host, clock, p = make()
        for lesson in LESSONS:
            p.lesson_done(lesson.id, 1.0)
        j.on_start()
        self.assertIn("finished every lesson", host.last())
        self.assertEqual(j.phase, "recap")


class SheetTests(unittest.TestCase):
    def test_a_lesson_on_another_sheet_waits_for_it_and_then_starts(self):
        j, host, clock, p = make()
        for lesson in LESSONS[:5]:
            p.lesson_done(lesson.id, 1.0)
        j.on_start()
        self.assertEqual(j.phase, "await_sheet")
        self.assertEqual(host.requests, ["lookalikes"])
        self.assertIn("look-alikes sheet", host.last())
        run(j, host, clock, 3)
        self.assertEqual(j.phase, "await_sheet", "still the alphabet sheet on the desk")
        host.sheet = "lookalikes"  # the learner swaps it (the tutor recognises it)
        run(j, host, clock, 0.5)
        self.assertEqual(j.phase, "await_sheet", "only once it stays in view")
        run(j, host, clock, 2)
        self.assertEqual(j.phase, "teach")
        self.assertEqual(j.target, "a")
        self.assertIn("Look-alikes", " ".join(host.said))

    def test_a_lesson_waits_until_its_sheet_is_really_under_the_camera(self):
        """They are told which sheet, and given time to get it: the lesson starts by itself once the camera can see it."""
        j, host, clock, p = make()
        host.page_visible = lambda: False  # (the right sheet is chosen, but nothing is under the camera yet)
        j.on_start()
        self.assertEqual(j.phase, "await_sheet")
        self.assertIn("Get the alphabet sheet ready", host.last())
        self.assertIn("I'll start as soon as I can see it", host.last())
        run(j, host, clock, 10)
        self.assertEqual(j.phase, "await_sheet")
        j.on_start()  # "I'm ready", but it is not in view: what it is waiting for
        self.assertEqual(j.phase, "await_sheet")
        self.assertIn("waiting for the alphabet sheet", host.last())
        host.page_visible = lambda: True
        run(j, host, clock, 3)
        self.assertEqual((j.phase, j.target), ("teach", "a"))
        self.assertTrue(any("I can see the sheet" in x for x in host.said))
        self.assertEqual(sum("alphabet sheet" in x for x in host.said), 2, "the sheet is named when asked for, not again in the intro")

    def test_saying_ready_with_the_sheet_in_view_starts_at_once(self):
        j, host, clock, p = make()
        host.page_visible = lambda: False
        j.on_start()
        host.page_visible = lambda: True
        j.on_found_it()  # ("got it")
        self.assertEqual((j.phase, j.target), ("teach", "a"))

    def test_it_reminds_them_now_and_then_while_waiting(self):
        j, host, clock, p = make()
        for lesson in LESSONS[:5]:
            p.lesson_done(lesson.id, 1.0)
        j.on_start()
        n = len(host.said)
        run(j, host, clock, 60)
        self.assertGreaterEqual(sum("waiting for the look-alikes sheet" in s for s in host.said[n:]), 2)
        self.assertGreaterEqual(len(host.requests), 3, "asks the tutor to look for the sheet again each time")


class ExploreAndReviewTests(unittest.TestCase):
    def test_explore_hands_the_clock_to_the_host_and_next_returns_to_the_lessons(self):
        j, host, clock, p = make()
        j.on_start()
        j.on_explore()
        self.assertEqual((j.phase, host.explore_resets), ("explore", 1))
        run(j, host, clock, 1.0)
        self.assertEqual(host.explore_ticks, 10)
        j.on_next()
        self.assertEqual(j.phase, "teach")

    def test_practice_needs_something_to_practise_and_then_favours_what_is_shaky(self):
        j, host, clock, p = make()
        j.on_practice()
        self.assertIn("lesson first", host.last())
        for l in "abcdefghijklmnop":  # 16 letters practised, so choosing 8 of them is a real choice
            p.record(l, True)
            p.record(l, True)
            p.record(l, True)
        p.record("d", False)  # d is shaky
        p.record("f", False)
        counts = {l: 0 for l in "abcdefghijklmnop"}
        for seed in range(40):
            jj, hh, cc, pp = make(progress=p, seed=seed)
            jj.session_started = True
            jj.on_practice()
            self.assertEqual(jj.phase, "review")
            for l in [jj.target] + jj.queue:
                counts[l] += 1
        self.assertGreater(counts["d"] + counts["f"], 2 * (counts["a"] + counts["b"]))
        self.assertGreater(counts["d"], 30, "the shaky letters come up nearly every time")

    def test_a_review_recaps_and_drills_a_pair_that_was_mixed_up_twice(self):
        j, host, clock, p = make()
        for l in "abcdef":
            p.record(l, True)
        j.session_started = True
        j.on_practice()
        j.queue = ["d", "d", "a"]  # force a scenario: d is asked twice and F is touched both times
        j.target = j.queue.pop(0)
        j.tries = j.hints = 0
        for _ in range(2):
            if j.target != "d":
                break
            answer(j, host, clock, "f")
            leave(j, host, clock)
            if j.target == "d":
                pass
        # finish whatever is left correctly
        guard = 0
        while j.phase == "review" and guard < 30:
            guard += 1
            answer(j, host, clock, j.target)
            leave(j, host, clock)
        text = " ".join(host.said)
        self.assertIn("Practice done", text)
        self.assertRegex(text, r"Let's compare the letter [DF] and the letter [DF]\.")  # the pair, in the order they were confused
        self.assertIn("The letter D:", text)
        self.assertIn("The letter F:", text)


class MiscTests(unittest.TestCase):
    def test_a_returning_learner_is_welcomed_back_with_their_streak(self):
        j, host, clock, p = make()
        p.sessions, p.streak, p.last_day = 4, 3, "2000-01-01"
        p.lesson_done("l1", 1.0)
        for l in "abc":
            for _ in range(3):
                p.record(l, True)
        j.on_start()
        self.assertIn("Welcome back", host.said[0])
        self.assertIn("You know 3 letters well", host.said[0])
        self.assertNotIn("room for six dots", host.said[0])

    def test_praise_can_come_from_the_ai_coach(self):
        coach = FakeCoach()
        j, host, clock, p = make(coach=coach, seed=3)
        j.on_start()
        for letter in "abcde":
            answer(j, host, clock, letter)
            leave(j, host, clock)
        self.assertTrue(any("Brilliant work." in s for s in host.said))

    def test_the_finger_gone_for_a_while_is_mentioned_once_in_a_while(self):
        j, host, clock, p = make()
        j.on_start()
        n = len(host.said)
        host.finger_pos = None
        run(j, host, clock, 9)
        self.assertEqual(sum("can't see your finger" in s for s in host.said[n:]), 1)

    def test_the_finger_reminder_backs_off_instead_of_nagging(self):
        j, host, clock, p = make()
        j.on_start()
        n = len(host.said)
        host.finger_pos = None
        times = []
        for _ in range(int(150 / 0.1)):
            before = len(host.said)
            clock.t += 0.1
            j.tick(clock.t)
            if any("can't see your finger" in s for s in host.said[before:]):
                times.append(round(clock.t - 1000.0, 1))
        gaps = [round(b - a) for a, b in zip(times, times[1:])]
        self.assertLessEqual(len(times), 6, times)
        self.assertEqual(gaps[:3], [15, 30, 60], gaps)  # 6 s, then 15, 30, then a minute

    def test_the_finger_reminder_starts_over_after_the_finger_was_seen(self):
        j, host, clock, p = make()
        j.on_start()
        host.finger_pos = None
        run(j, host, clock, 40)  # reminded a few times
        host.finger_pos = (500.0, 500.0)
        run(j, host, clock, 0.5)
        n = len(host.said)
        host.finger_pos = None
        run(j, host, clock, 7)
        self.assertEqual(sum("can't see your finger" in s for s in host.said[n:]), 1, "back to the first, short interval")

    def test_nothing_is_said_about_the_finger_while_the_sheet_itself_is_missing(self):
        j, host, clock, p = make()
        j.on_start()
        host.page_visible = lambda: False
        n = len(host.said)
        host.finger_pos = None
        run(j, host, clock, 20)
        self.assertEqual(sum("can't see your finger" in s for s in host.said[n:]), 0, "the sheet is the problem, and that is said elsewhere")
        host.page_visible = lambda: True
        run(j, host, clock, 3)
        self.assertEqual(sum("can't see your finger" in s for s in host.said[n:]), 1)

    def test_status_describes_where_the_learner_is(self):
        j, host, clock, p = make()
        self.assertEqual(j.status()["phase"], "idle")
        j.on_start()
        st = j.status()
        self.assertEqual((st["phase"], st["target"], st["target_dots"], st["lesson"]["title"], st["remaining"]), ("teach", "a", [1], "The first five letters", 5))
        answer(j, host, clock, "a")
        self.assertEqual(j.status()["remaining"], 4)
        import json
        json.dumps(j.status())


if __name__ == "__main__":
    unittest.main()
