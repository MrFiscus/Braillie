"""Tests for progress.py: mastery boxes, confusions, lessons, streaks, merging, and safe storage."""
import datetime as dt
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from progress import BOXES, MASTERED_BOX, Progress, progress_path, today_str


class Clock:
    def __init__(self, day="2026-03-10", hour=12):
        self.t = dt.datetime.fromisoformat(f"{day}T{hour:02d}:00:00").timestamp()

    def __call__(self):
        return self.t

    def advance(self, days=0, seconds=0):
        self.t += days * 86400 + seconds


class Lesson:
    def __init__(self, id):
        self.id = id


class BoxTests(unittest.TestCase):
    def test_clean_answers_climb_and_help_or_mistakes_do_not(self):
        p = Progress(Clock())
        for want in (1, 2, 3, 4, 4):
            p.record("a", True)
            self.assertEqual(p.box("a"), want)  # tops out at the last box
        p.record("a", True, hints=1)
        self.assertEqual(p.box("a"), 4, "help needed: stays where it is")
        p.record("a", True, tries=2)
        self.assertEqual(p.box("a"), 4)
        p.record("a", False)
        self.assertEqual(p.box("a"), 0, "a miss sends it back to the start")
        s = p.letters["a"]
        self.assertEqual((s.attempts, s.correct, s.clean, s.hints), (8, 7, 5, 1))

    def test_mastery_and_learned(self):
        p = Progress(Clock())
        self.assertEqual((p.mastery("z"), p.is_learned("z"), p.learned()), (0.0, False, []))
        for _ in range(MASTERED_BOX):
            p.record("d", True)
        self.assertTrue(p.is_learned("d"))
        self.assertAlmostEqual(p.mastery("d"), MASTERED_BOX / (BOXES - 1))
        p.record("e", False)
        self.assertEqual((p.learned(), p.practised()), (["d"], ["d", "e"]))


class ConfusionTests(unittest.TestCase):
    def test_pairs_are_counted_and_ranked_and_looked_up_either_way_round(self):
        p = Progress(Clock())
        for _ in range(3):
            p.record_confusion("f", "d")
        p.record_confusion("d", "f")
        p.record_confusion("k", "a")
        p.record_confusion("a", "a")  # touching the right one is not a confusion
        p.record_confusion("", "a")
        self.assertEqual(p.top_confusions(2), [("f", "d", 3), ("d", "f", 1)])
        self.assertEqual(p.confused_with("d"), ["f"])
        self.assertEqual(p.confused_with("a"), ["k"])
        self.assertEqual(p.confused_with("q"), [])


class LessonTests(unittest.TestCase):
    def test_score_best_and_the_next_lesson(self):
        p = Progress(Clock())
        lessons = [Lesson("l1"), Lesson("l2"), Lesson("l3")]
        self.assertIs(p.next_lesson(lessons), lessons[0])
        p.lesson_done("l1", 0.6)
        self.assertFalse(p.lesson_mastered("l1"))
        self.assertIs(p.next_lesson(lessons), lessons[0], "not mastered yet: stays on it")
        p.lesson_done("l1", 1.0)
        p.lesson_done("l1", 0.4)  # a worse later attempt does not undo it
        self.assertEqual((p.lessons["l1"]["best"], p.lessons["l1"]["last"], p.lessons["l1"]["times"]), (1.0, 0.4, 3))
        self.assertIs(p.next_lesson(lessons), lessons[1])
        p.lesson_done("l2", 0.8)
        p.lesson_done("l3", 0.9)
        self.assertIsNone(p.next_lesson(lessons))


class StreakTests(unittest.TestCase):
    def test_days_in_a_row(self):
        c = Clock()
        p = Progress(c)
        p.start_session()
        self.assertEqual((p.sessions, p.streak), (1, 1))
        c.advance(seconds=3600)
        p.start_session()  # same day
        self.assertEqual((p.sessions, p.streak), (2, 1))
        c.advance(days=1)
        p.start_session()
        c.advance(days=1)
        p.start_session()
        self.assertEqual((p.streak, p.best_streak), (3, 3))
        c.advance(days=3)
        p.start_session()
        self.assertEqual((p.streak, p.best_streak), (1, 3), "a gap starts again, the best is kept")


class PracticePickingTests(unittest.TestCase):
    def test_shaky_letters_are_picked_far_more_often_than_solid_ones(self):
        c = Clock()
        p = Progress(c)
        for _ in range(4):
            p.record("a", True)  # solid
        p.record("b", False)  # shaky
        c.advance(seconds=10)
        picks = Counter()
        rng = random.Random(1)
        for _ in range(600):
            picks[p.pick_practice("ab", 1, rng)[0]] += 1
        self.assertGreater(picks["b"], 4 * picks["a"])
        self.assertGreater(picks["a"], 0, "solid ones still come up now and then")

    def test_confused_letters_are_boosted(self):
        p = Progress(Clock())
        for l in "abc":
            p.record(l, True)
        for _ in range(5):
            p.record_confusion("b", "c")
        picks = Counter()
        rng = random.Random(2)
        for _ in range(600):
            picks[p.pick_practice("abc", 1, rng)[0]] += 1
        self.assertGreater(picks["b"] + picks["c"], 2 * picks["a"])

    def test_distinct_and_bounded_and_deterministic_with_a_seed(self):
        p = Progress(Clock())
        got = p.pick_practice("abcdef", 4, random.Random(5))
        self.assertEqual(len(got), 4)
        self.assertEqual(len(set(got)), 4)
        self.assertEqual(got, p.pick_practice("abcdef", 4, random.Random(5)))
        self.assertEqual(sorted(p.pick_practice("abc", 10, random.Random(0))), ["a", "b", "c"])
        self.assertEqual(p.pick_practice("", 3), [])
        self.assertEqual(p.pick_practice("aab", 5, random.Random(0)).count("a"), 1, "duplicates in the pool are ignored")


class StorageTests(unittest.TestCase):
    def full(self):
        c = Clock()
        p = Progress(c)
        p.start_session()
        for l in "abc":
            p.record(l, True)
        p.record_confusion("d", "f")
        p.lesson_done("l1", 0.8)
        return p

    def test_round_trip_through_a_file(self):
        p = self.full()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "p.json"
            p.save(path)
            q = Progress.load(path)
            self.assertEqual(q.to_dict(), p.to_dict())
            self.assertEqual([f.name for f in path.parent.iterdir()], ["p.json"], "no temp file left behind")

    def test_a_missing_file_is_a_fresh_start(self):
        with tempfile.TemporaryDirectory() as d:
            q = Progress.load(Path(d) / "nope.json")
            self.assertEqual((q.sessions, q.letters), (0, {}))

    def test_a_damaged_file_is_set_aside_not_lost_or_crashed_on(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            path.write_text("{ this is not json")
            q = Progress.load(path)
            self.assertEqual(q.letters, {})
            self.assertTrue((Path(d) / "p.damaged").exists())
            self.assertEqual((Path(d) / "p.damaged").read_text(), "{ this is not json")

    def test_junk_in_the_data_is_ignored_safely(self):
        q = Progress.from_dict({"letters": {"a": {"box": 99, "attempts": "3"}, "zz": {}, 5: {}, "b": "no"}, "confusions": {"x": 1, "d>f": "2"},
                                "lessons": {"l1": "bad", "l2": {"best": "0.5"}}, "sessions": "4"})
        self.assertEqual(list(q.letters), ["a"])
        self.assertEqual((q.letters["a"].box, q.letters["a"].attempts), (BOXES - 1, 3))
        self.assertEqual(q.confusions, {"d>f": 2})
        self.assertEqual((q.lessons, q.sessions), ({"l2": {"best": 0.5, "last": 0.0, "times": 0}}, 4))
        self.assertEqual(Progress.from_dict("garbage").letters, {})

    def test_profile_names_make_safe_file_names(self):
        self.assertEqual(progress_path("Ana Maria", Path("/x")).name, "progress-anamaria.json")
        self.assertEqual(progress_path("../../etc/passwd", Path("/x")).name, "progress-etcpasswd.json")
        self.assertEqual(progress_path("", Path("/x")).name, "progress-default.json")


class MergeTests(unittest.TestCase):
    def test_merging_your_own_copy_changes_nothing(self):
        p = StorageTests().full()
        before = p.to_dict()
        p.merge(Progress.from_dict(p.to_dict()))
        self.assertEqual(p.to_dict(), before, "nothing counted twice")

    def test_two_devices_combine(self):
        c1, c2 = Clock("2026-03-10"), Clock("2026-03-11")
        laptop, account = Progress(c1), Progress(c2)
        laptop.record("a", True)
        laptop.record("b", False)
        laptop.record_confusion("b", "a")
        laptop.start_session()
        c1.advance(days=1)
        account.record("b", True)
        account.record("b", True)  # more recent and further on for b
        account.record("c", True)
        account.record_confusion("b", "a")
        account.record_confusion("b", "a")
        account.lesson_done("l1", 0.9)
        account.start_session()
        account.start_session()
        laptop.merge(account)
        self.assertEqual(sorted(laptop.letters), ["a", "b", "c"])
        self.assertEqual(laptop.letters["b"].box, 2, "the more recent copy of b wins")
        self.assertEqual(laptop.letters["a"].box, 1)
        self.assertEqual(laptop.confusions["b>a"], 2)
        self.assertEqual(laptop.lessons["l1"]["best"], 0.9)
        self.assertEqual(laptop.sessions, 2)
        self.assertEqual(laptop.last_day, "2026-03-11")

    def test_summary_is_json_ready(self):
        p = StorageTests().full()
        s = p.summary()
        json.dumps(s)
        self.assertEqual((s["practised"], s["sessions"], s["streak"]), (3, 1, 1))
        self.assertEqual(s["confusions"], [{"touched": "d", "wanted": "f", "count": 1}])


if __name__ == "__main__":
    unittest.main()
