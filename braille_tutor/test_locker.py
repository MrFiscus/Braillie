"""Tests for CellLocker: a cell that has been read right must STAY right through camera shake, blur and dropouts."""
import unittest

import numpy as np

import sheets
from detect import _make_cell
from vote import CellLocker


def observed_like(expected, labels):
    """A scan of the known sheet where cell i reads labels[i] (a 6-char string)."""
    return [{**e, "label": lab, "dots": frozenset(i + 1 for i, ch in enumerate(lab) if ch == "1"), "char": "?", "confidence": 0.8}
            for e, lab in zip(expected, labels)]


def cells_at(labels, y=10.0, x0=10.0, pitch=12.0):
    return [_make_cell(x0 + i * pitch, y, 6.0, 9.0, lab, 0.8, 0, i) for i, lab in enumerate(labels)]


class KnownSheetTests(unittest.TestCase):
    def setUp(self):
        self.expected = sheets.get_sheet("alphabet").cells
        self.right = [e["label"] for e in self.expected]

    def scan(self, locker, labels):
        return locker.update_known(observed_like(self.expected, labels), self.expected)

    def test_locks_as_soon_as_it_reads_as_the_sheet_says(self):
        lk = CellLocker(lock_after=2)
        self.scan(lk, self.right)
        self.assertEqual(lk.locked_count, 0)
        out = self.scan(lk, self.right)
        self.assertEqual(lk.locked_count, 26)
        self.assertTrue(all(c["locked"] for c in out))

    def test_known_sheet_centres_stay_on_the_printed_layout(self):
        """Live refine under a finger must not drag demo-sheet boxes onto neighbours."""
        lk = CellLocker(lock_after=1)
        shifted = [{**c, "x": c["x"] + 4.0, "y": c["y"] - 3.0} for c in observed_like(self.expected, self.right)]
        out = lk.update_known(shifted, self.expected)
        for got, want in zip(out, self.expected):
            self.assertAlmostEqual(got["x"], want["x"], places=3)
            self.assertAlmostEqual(got["y"], want["y"], places=3)

    def test_a_wrong_reading_never_locks(self):
        lk = CellLocker()
        wrong = ["111111"] * 26
        for _ in range(10):
            self.scan(lk, wrong)
        self.assertEqual(lk.locked_count, 0)

    def test_camera_shake_cannot_unlock_a_correct_cell(self):
        """The reported problem: right, then a bump of the camera scrambles the reading. It must not flicker."""
        lk = CellLocker(lock_after=2, unlock_after=6)
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        garbled = list(self.right)
        garbled[3] = "000001"  # the letter d suddenly reads as a lone dot 6
        garbled[10] = "000000"  # and k reads as nothing
        for _ in range(5):  # five bad scans in a row: still inside the hold
            out = self.scan(lk, garbled)
            self.assertEqual(out[3]["label"], self.right[3])
            self.assertEqual(out[10]["label"], self.right[10])
            self.assertTrue(out[3]["locked"])

    def test_alternating_good_and_bad_scans_stay_locked_once_locked(self):
        lk = CellLocker(lock_after=2, unlock_after=4)
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        rng = np.random.default_rng(0)

        def garbage(true):  # shake garbles a cell differently every time
            while True:
                lab = "".join(rng.choice(["0", "1"], 6))
                if lab != true:
                    return lab

        for i in range(60):
            noisy = [lab if rng.random() > 0.4 else garbage(lab) for lab in self.right]  # 40% of cells wrong in every scan
            out = self.scan(lk, noisy)
            self.assertEqual([c["label"] for c in out], self.right, f"flickered at scan {i}")

    def test_a_scan_with_nothing_seen_changes_nothing(self):
        lk = CellLocker()
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        out = self.scan(lk, ["000000"] * 26)
        self.assertEqual([c["label"] for c in out], self.right)

    def test_a_finger_covering_a_dot_does_not_rename_a_known_cell(self):
        """Once the alphabet sheet is known, covering a dot (B looking like A) must not change the name."""
        lk = CellLocker(lock_after=2, unlock_after=4)
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        covered = list(self.right)
        covered[1] = self.right[0]  # B, with a dot hidden, reads as A
        for _ in range(20):
            out = self.scan(lk, covered)
            self.assertEqual(out[1]["label"], self.right[1], "B must stay B")
            self.assertTrue(out[1]["locked"])

    def test_it_keeps_the_sheet_name_even_when_a_different_pattern_persists(self):
        """A long run of misreads is a finger or shadow, not a new sheet. Names change only on reset / a new page."""
        lk = CellLocker(lock_after=2, unlock_after=4)
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        changed = list(self.right)
        changed[5] = "101010"
        for _ in range(12):
            out = self.scan(lk, changed)
            self.assertTrue(out[5]["locked"])
            self.assertEqual(out[5]["label"], self.right[5])

    def test_swapping_the_whole_sheet_does_not_rename_until_reset(self):
        lk = CellLocker(lock_after=2, unlock_after=3)
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        other = ["110011"] * 26
        for _ in range(8):
            self.scan(lk, other)
        self.assertEqual(lk.locked_count, 26)
        self.assertEqual([c["label"] for c in lk.result()], self.right)

    def test_reset(self):
        lk = CellLocker()
        self.scan(lk, self.right)
        self.scan(lk, self.right)
        lk.reset()
        self.assertEqual(lk.locked_count, 0)
        self.assertEqual(lk.result(), [])


class UnknownPageTests(unittest.TestCase):
    LABELS = ["100000", "110000", "100100", "100110", "100010"]

    def test_locks_on_a_steady_majority(self):
        lk = CellLocker(lock_after=3, agree=0.7)
        for _ in range(2):
            lk.update(cells_at(self.LABELS))
        self.assertEqual(lk.locked_count, 0)
        lk.update(cells_at(self.LABELS))
        self.assertEqual(lk.locked_count, 5)

    def test_an_unsteady_reading_does_not_lock(self):
        lk = CellLocker(lock_after=3, agree=0.7)
        for lab in ["100000", "110000", "100100", "100110", "100000", "110000"]:
            lk.update(cells_at([lab]))
        self.assertEqual(lk.locked_count, 0)

    def test_locked_cells_survive_dropouts_and_noise(self):
        lk = CellLocker(lock_after=3, unlock_after=6)
        for _ in range(4):
            lk.update(cells_at(self.LABELS))
        for _ in range(4):
            lk.update([])  # the detector saw nothing at all
        for _ in range(4):
            lk.update(cells_at(["111111"] * 5))  # then garbage, but not for long enough to count
        out = lk.result()
        self.assertEqual([c["label"] for c in out], self.LABELS)
        self.assertTrue(all(c["locked"] for c in out))

    def test_small_position_jitter_stays_the_same_cell(self):
        lk = CellLocker(lock_after=3)
        rng = np.random.default_rng(2)
        for _ in range(6):
            lk.update(cells_at(self.LABELS, x0=10.0 + float(rng.normal(0, 0.4)), y=10.0 + float(rng.normal(0, 0.4))))
        self.assertEqual(len(lk.slots), 5)
        self.assertEqual(lk.locked_count, 5)

    def test_unlocked_ghosts_age_out_but_locked_cells_do_not(self):
        lk = CellLocker(lock_after=2, window=4)
        lk.update(cells_at(["100000", "110000"]))
        lk.update(cells_at(["100000", "110000"]))  # both locked
        lk.update(cells_at(["100000", "110000", "111000"], pitch=12.0))  # a one-off third cell
        self.assertEqual(len(lk.slots), 3)
        for _ in range(6):
            lk.update(cells_at(["100000", "110000"]))
        self.assertEqual(len(lk.slots), 2)  # the ghost is gone, the two real cells remain

    def test_prelock_holds_verified_cells_from_the_start(self):
        lk = CellLocker()
        lk.prelock(cells_at(self.LABELS))
        self.assertEqual(lk.locked_count, 5)
        for _ in range(3):
            lk.update(cells_at(["111111"] * 5))
        self.assertEqual([c["label"] for c in lk.result()], self.LABELS)


if __name__ == "__main__":
    unittest.main()
