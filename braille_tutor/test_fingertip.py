"""Fingertip tracker tests on SYNTHETIC hands (paper + skin-coloured arm, palm and finger). Not real hands: see fingertip.py."""
import unittest

import cv2
import numpy as np

import fingertip
from fingertip import FingerTracker, Tip, find_fingertip

W, H_PX = 640, 480
SKIN_TONES = {"light": (150, 170, 225), "medium": (105, 145, 200), "tan": (80, 120, 175), "dark": (55, 85, 130)}


def hand_frame(tip=(320, 150), tone="medium", angle_deg=0.0, paper=(235, 238, 240), noise=0.0, brightness=1.0, seed=0,
               extra_finger=False):
    """Paper with a hand entering from the bottom edge, its index finger ending exactly at `tip` (before rotation about it)."""
    img = np.full((H_PX, W, 3), paper, np.uint8)
    skin = SKIN_TONES[tone]
    hand = np.zeros((H_PX, W), np.uint8)
    tx, ty = tip
    cv2.rectangle(hand, (tx - 55, ty + 210), (tx + 55, H_PX), 255, -1)  # forearm, cut off by the bottom edge
    cv2.ellipse(hand, (tx, ty + 170), (75, 70), 0, 0, 360, 255, -1)  # palm
    cv2.line(hand, (tx, ty + 130), (tx, ty + 18), 255, 36)  # index finger; the round cap of a thick line ends exactly at (tx, ty)
    if extra_finger:  # a shorter second finger beside it
        cv2.line(hand, (tx + 55, ty + 130), (tx + 55, ty + 90), 255, 30)
    M = cv2.getRotationMatrix2D((tx, ty + 200), angle_deg, 1.0)
    hand = cv2.warpAffine(hand, M, (W, H_PX))
    img[hand > 0] = skin
    img = cv2.GaussianBlur(img, (3, 3), 0)
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * brightness + (rng.normal(0, noise, img.shape) if noise else 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def truth_after_rotation(tip, angle_deg):
    tx, ty = tip
    M = cv2.getRotationMatrix2D((tx, ty + 200), angle_deg, 1.0)
    p = M @ np.array([tx, ty, 1.0])
    return float(p[0]), float(p[1])


class TestFindFingertip(unittest.TestCase):
    def assert_tip(self, frame, truth, tol=8.0):
        tip = find_fingertip(frame)
        self.assertIsNotNone(tip)
        self.assertLess(np.hypot(tip.x - truth[0], tip.y - truth[1]), tol, f"found {(tip.x, tip.y)} expected {truth}")
        return tip

    def test_no_hand_no_tip(self):
        self.assertIsNone(find_fingertip(np.full((H_PX, W, 3), (235, 238, 240), np.uint8)))

    def test_finds_the_end_of_the_finger(self):
        tip = self.assert_tip(hand_frame(), (320, 150))
        self.assertGreater(tip.confidence, 0.7)
        self.assertGreater(tip.reach, 2.0)

    def test_across_skin_tones(self):
        for tone in SKIN_TONES:
            with self.subTest(tone=tone):
                self.assert_tip(hand_frame(tone=tone), (320, 150))

    def test_across_positions_and_angles(self):
        for pos in ((150, 120), (500, 200), (320, 260)):
            for angle in (-30, 0, 25):
                with self.subTest(pos=pos, angle=angle):
                    self.assert_tip(hand_frame(pos, angle_deg=angle), truth_after_rotation(pos, angle), tol=10)

    def test_lighting_and_noise(self):
        for brightness in (0.75, 1.0):
            with self.subTest(brightness=brightness):
                self.assert_tip(hand_frame(brightness=brightness, noise=6, seed=3), (320, 150), tol=10)

    def test_warm_paper_is_not_skin(self):
        """Paper under a warm light (flash / lamp) must not be taken for a hand."""
        self.assertIsNone(find_fingertip(np.full((H_PX, W, 3), (170, 205, 235), np.uint8)))

    def test_farthest_finger_wins(self):
        self.assert_tip(hand_frame(extra_finger=True), (320, 150), tol=10)

    def test_a_blob_not_attached_to_the_border_is_doubted(self):
        img = np.full((H_PX, W, 3), (235, 238, 240), np.uint8)
        cv2.circle(img, (320, 240), 60, SKIN_TONES["medium"], -1)  # a skin-coloured coin-like thing floating on the page
        tip = find_fingertip(img)
        self.assertTrue(tip is None or tip.confidence < 0.3)

    def test_huge_skin_region_is_not_a_hand(self):
        self.assertIsNone(find_fingertip(np.full((H_PX, W, 3), SKIN_TONES["medium"], np.uint8)))

    def test_works_at_other_resolutions(self):
        big = cv2.resize(hand_frame(), (1280, 960))
        self.assert_tip(big, (640, 300), tol=16)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def identity_like():  # page mm == image px / 4
    return np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1.0]])


class TestFingerTracker(unittest.TestCase):
    def tracker(self, **kw):
        clock = FakeClock()
        return FingerTracker(clock=clock, **kw), clock

    def test_needs_two_frames_in_a_row(self):
        tr, _ = self.tracker()
        frame = hand_frame()
        self.assertIsNone(tr.update(frame, identity_like()))
        pos = tr.update(frame, identity_like())
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 80, delta=3)  # 320 px / 4
        self.assertAlmostEqual(pos[1], 37.5, delta=3)

    def test_no_position_without_a_page(self):
        tr, _ = self.tracker()
        for _ in range(3):
            self.assertIsNone(tr.update(hand_frame(), None))

    def test_holds_through_a_short_dropout_then_lets_go(self):
        tr, clock = self.tracker(hold_seconds=0.6)
        empty = np.full((H_PX, W, 3), (235, 238, 240), np.uint8)
        H = identity_like()
        for _ in range(3):
            tr.update(hand_frame(), H)
        clock.t = 0.3
        self.assertIsNotNone(tr.update(empty, H))  # hand vanished a moment ago: still held
        clock.t = 1.0
        self.assertIsNone(tr.update(empty, H))  # gone for good

    def test_camera_shake_does_not_move_the_finger_in_page_mm(self):
        """Same finger on the page, camera shifted 20 px: page registration moves with it, so the page position stays put."""
        tr, _ = self.tracker()
        H1 = identity_like()
        shifted = np.array([[0.25, 0, -5.0], [0, 0.25, 0], [0, 0, 1.0]])  # camera moved: same point maps to different mm... 
        f1 = hand_frame((320, 150))
        f2 = hand_frame((340, 150))  # ...because the finger appears 20 px to the right in the image
        for _ in range(3):
            tr.update(f1, H1)
        p1 = tr.position
        for _ in range(6):
            tr.update(f2, shifted)
        # (340 px * 0.25) - 5 = 80 mm: the same page spot as before
        self.assertLess(np.hypot(tr.position[0] - p1[0], tr.position[1] - p1[1]), 3.0)

    def test_smooths_jitter_but_follows_a_real_move(self):
        tr, _ = self.tracker()
        H = identity_like()
        for i in range(4):
            tr.update(hand_frame((320 + (4 if i % 2 else -4), 150)), H)
        self.assertLess(abs(tr.position[0] - 80), 2.0)  # +-4 px = +-1 mm jitter averaged away
        for _ in range(3):
            tr.update(hand_frame((520, 250)), H)  # a real jump of 50 mm+: taken as a new place at once
        self.assertAlmostEqual(tr.position[0], 130, delta=3)

    def test_low_confidence_sightings_are_ignored(self):
        weak = lambda frame: Tip(10, 10, 0.1, (0, 0), 1.2)
        tr = FingerTracker(find=weak)
        for _ in range(4):
            self.assertIsNone(tr.update(np.zeros((10, 10, 3), np.uint8), identity_like()))


class TestFeedIntegration(unittest.TestCase):
    def tracked_feed(self):
        """A feed with a registered page and a hand in view, so feed.finger() has something of its own to report."""
        import tutor
        feed = tutor.CameraFeed(None, None, None, track_finger=True)
        feed.page_src, feed.H = None, identity_like()
        for _ in range(3):
            feed.tracker.update(hand_frame(), feed.H)
        return feed

    def test_tracked_finger_feeds_the_tutor_and_a_click_stands_in_for_it(self):
        feed = self.tracked_feed()
        pos = feed.finger()
        self.assertAlmostEqual(pos[0], 80, delta=3)
        self.assertEqual(feed.finger_source(), "camera")
        feed.set_finger_mm(10, 20)
        self.assertEqual((feed.finger(), feed.finger_source()), ((10, 20), "posted"))
        feed.clear_finger()
        self.assertAlmostEqual(feed.finger()[0], 80, delta=3)
        self.assertEqual(feed.finger_source(), "camera")
        feed.H = None
        self.assertIsNone(feed.finger())
        self.assertIsNone(feed.finger_source())

    def test_a_click_hands_the_finger_back_to_the_camera_instead_of_pinning_it_for_ever(self):
        """The reported bug: one stray tap on the video and the tutor named the clicked cell for every letter touched after it."""
        import tutor
        feed = self.tracked_feed()
        feed.set_finger_mm(10, 20)
        self.assertEqual(feed.finger(), (10, 20))
        feed.finger_posted -= tutor.POSTED_FINGER_SECONDS + 0.1  # as if the click were that long ago
        self.assertAlmostEqual(feed.finger()[0], 80, delta=3, msg="a stale click must not outrank the camera")
        self.assertEqual(feed.finger_source(), "camera")
        feed.set_finger_mm(30, 40)  # clicking again is how you keep pointing
        self.assertEqual(feed.finger(), (30, 40))

    def test_a_stale_click_outlives_a_rest_that_counts_as_an_answer(self):
        """It has to last: the learner clicks, the tutor waits for the finger to settle, and only then is it an answer."""
        import tutor
        self.assertGreater(tutor.POSTED_FINGER_SECONDS, tutor.DWELL_SECONDS * 2)

    def test_tracking_is_off_unless_asked_for(self):
        import tutor
        self.assertIsNone(tutor.CameraFeed(None, None, None).tracker)


if __name__ == "__main__":
    unittest.main()
