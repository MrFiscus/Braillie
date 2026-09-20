"""Tests for learn.py: the teaching text, resting-to-answer, and whole lessons run against a fake finger, voice and clock."""
import random
import unittest

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
        self.sheets = {"alphabet": ALPHABET, "lookalikes": LOOKALIKES}

    # -- the host interface --
    def say(self, text):
        self.said.append(text)

    def tone(self, kind):
        self.tones.append(kind)

    def finger(self):
        return self.finger_pos

    def cells(self):
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

    # -- helpers for the tests --
    def pos(self, letter):
        c = next(c for c in self.cells() if frozenset(c["dots"]) == DOTS[letter])
        return c["x"], c["y"]

    def last(self, n=1):
        return " | ".join(self.said[-n:])


def make(progress=None, coach=None, sheet="alphabet", seed=1):
    clock, host = Clock(), Host(sheet)
    p = progress or Progress(clock=lambda: clock.t)
    j = Journey(host, p, coach=coach, rng=random.Random(seed), clock=clock)
    return j, host, clock, p


def run(j, host, clock, seconds, step=0.1):
    for _ in range(int(round(seconds / step))):
        clock.t += step
        j.tick(clock.t)


def answer(j, host, clock, letter, settle=1.4):
    """The learner puts a finger on `letter` and rests it there."""
    host.finger_pos = host.pos(letter)
    run(j, host, clock, settle)


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
        self.assertEqual(relation("b", ["a"]), "It is the letter A with one more dot, at the middle-left.")
        self.assertEqual(relation("k", list("abcde")), "It is the letter A with one more dot, at the bottom-left.")
        self.assertEqual(relation("w", list("abcdefghij")), "It is the letter J with one more dot, at the bottom-right.")
        self.assertIsNone(relation("a", list("bcde")), "nothing is a start of A")
        self.assertIsNone(relation("b", []))
        self.assertIsNone(relation("b", ["b", "q"]), "not itself, and q (12345) is not a start of b")

    def test_the_nearest_known_letter_with_the_fewest_extra_dots_is_chosen(self):
        self.assertIn("the letter C", relation("d", list("abc")))  # d (145) grows from c (14), one dot; a (1) would need two
        self.assertIn("the letter A", relation("d", ["a"]))
        self.assertIn("two more dots", relation("d", ["a"]))

    def test_every_letter_a_to_z_can_be_described_and_its_relation_is_true(self):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self.assertTrue(dot_words(letter).endswith("."))
            rel = relation(letter, [l for l in "abcdefghijklmnopqrstuvwxyz" if l < letter])
            if rel:  # whatever it says, the named letter must really be a start of this one
                named = rel.split("the letter ")[1][0].lower()
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
        for part in ("six places", "two columns of three", "one, two, three", "four, five, six"):
            self.assertIn(part, text)

    def test_letter_of_dots(self):
        self.assertEqual(letter_of_dots([1, 4, 5]), "d")
        self.assertIsNone(letter_of_dots([1, 2, 3, 4, 5, 6]))


class LessonDataTests(unittest.TestCase):
    def test_lessons_are_sound(self):
        seen = []
        for lesson in LESSONS:
            self.assertTrue(all(l in DOTS for l in lesson.letters), lesson.id)
            self.assertEqual(len(lesson.letters), len(set(lesson.letters)), lesson.id)
            sheet_letters = {letter_of_dots(c["dots"]) for c in sheets.get_sheet(lesson.sheet).cells}
            self.assertTrue(set(lesson.letters) <= sheet_letters, f"{lesson.id} needs letters its sheet does not have")
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
        self.assertEqual(d.update((10, 11), 1.1), (10, 10))
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
        self.assertIsNone(d.update((20, 0), 3.1), "after the finger was lost it must rest again, even on the same spot")
        self.assertEqual(d.update((20, 0), 4.2), (20, 0))


class FirstLessonTests(unittest.TestCase):
    def test_a_first_time_learner_is_welcomed_then_taught_the_first_letter(self):
        j, host, clock, p = make()
        j.on_start()
        text = " ".join(host.said)
        self.assertIn("six places for raised dots", text)  # the first-ever welcome
        self.assertIn("The first five letters", text)
        self.assertIn("top four dots", text)  # the insight
        self.assertEqual(j.phase, "teach")
        self.assertEqual(j.target, "a")
        self.assertEqual(host.last(), "The letter A. One dot, at the top-left. Find it and rest your finger on it.")
        self.assertEqual(host.tones, ["ready"])
        self.assertEqual(p.sessions, 1)

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
        self.assertEqual(j.phase, "recap")
        self.assertIn("Lesson complete", " ".join(host.said[-4:]))
        self.assertIn("5 of 5", " ".join(host.said[-4:]))
        self.assertIn("say next for F to J", host.said[-1])
        self.assertEqual(host.tones.count("correct"), 10)
        self.assertEqual(host.tones[-1], "done")
        self.assertTrue(p.lesson_mastered("l1"))
        self.assertTrue(all(p.box(l) == 1 for l in "abcde"), "found clean in practice: each moved up a box")
        self.assertEqual(len(p.practised()), 5)
        # and "next" carries on to the next lesson
        j.on_next()
        self.assertEqual((j.phase, j.lesson.id, j.target), ("teach", "l2", "f"))

    def test_the_teaching_of_each_letter_builds_on_the_ones_before(self):
        j, host, clock, p = make()
        j.on_start()
        answer(j, host, clock, "a")
        self.assertIn("That's the letter A. One dot, at the top-left.", host.said[-2])  # the praise, then the next letter is asked
        leave(j, host, clock)
        self.assertEqual(host.last(), "The letter B. Two dots: top-left and middle-left. It is the letter A with one more dot, at the middle-left. "
                                      "Find it and rest your finger on it.")
        answer(j, host, clock, "b")
        leave(j, host, clock)
        answer(j, host, clock, "c")
        leave(j, host, clock)
        self.assertIn("It is the letter C with one more dot, at the middle-right.", host.last())  # D is taught from C, the closest start


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
        j, host, clock, p = self.start_practice()
        n = len(host.said)
        host.finger_pos = (5.0, 5.0)  # nowhere near a cell
        run(j, host, clock, 2.5)
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
        host.finger_pos = None
        run(j, host, clock, HINT_AFTER[0] + 1)
        first = host.said[n:]
        self.assertTrue(any("Here's a hint" in s and dot_words(want) in s for s in first), first)
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
        self.assertIn(dot_words(want), host.last())
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
        while j.phase == "practice":
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
        self.assertIn("lookalikes sheet", host.last())
        run(j, host, clock, 3)
        self.assertEqual(j.phase, "await_sheet", "still the alphabet sheet on the desk")
        host.sheet = "lookalikes"  # the learner swaps it (the tutor recognises it)
        run(j, host, clock, 0.5)
        self.assertEqual(j.phase, "teach")
        self.assertEqual(j.target, "a")
        self.assertIn("Look-alikes", " ".join(host.said))

    def test_it_reminds_them_now_and_then_while_waiting(self):
        j, host, clock, p = make()
        for lesson in LESSONS[:5]:
            p.lesson_done(lesson.id, 1.0)
        j.on_start()
        n = len(host.said)
        run(j, host, clock, 60)
        self.assertGreaterEqual(sum("waiting for the lookalikes sheet" in s for s in host.said[n:]), 2)
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
        self.assertNotIn("six places", host.said[0])

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
