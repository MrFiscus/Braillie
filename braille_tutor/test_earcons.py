"""Tests for earcons.py: the tones are valid, gentle, distinct and routed through the voice player."""
import io
import unittest
import wave

import numpy as np

import earcons
from earcons import Earcons, KINDS, tone_samples, tone_wav


def dominant_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / earcons.RATE)[spectrum.argmax()])


class ToneTests(unittest.TestCase):
    def test_every_tone_is_a_valid_gentle_wav(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                data = tone_wav(kind)
                with wave.open(io.BytesIO(data)) as w:
                    self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, earcons.RATE))
                    seconds = w.getnframes() / w.getframerate()
                self.assertTrue(0.05 < seconds < 1.5, seconds)
                x = tone_samples(kind)
                self.assertLessEqual(float(np.abs(x).max()), earcons.LEVEL + 1e-6, "never louder than the set level")
                self.assertGreater(float(np.abs(x).max()), 0.1, "clearly audible")
                self.assertLess(abs(float(x[0])), 0.02, "starts from silence: no click")
                self.assertLess(abs(float(x[-1])), 0.02, "ends in silence: no click")

    def test_right_sounds_higher_and_brighter_than_wrong(self):
        right, wrong = tone_samples("correct"), tone_samples("wrong")
        self.assertGreater(dominant_hz(right[-int(0.15 * earcons.RATE):]), 900)  # ends on the high C
        self.assertLess(dominant_hz(wrong[-int(0.15 * earcons.RATE):]), 400)  # ends low
        self.assertLess(len(wrong) / earcons.RATE, 0.6, "and short: not a scolding")

    def test_the_tones_are_different_from_each_other(self):
        seen = {kind: tone_wav(kind) for kind in KINDS}
        self.assertEqual(len(set(seen.values())), len(KINDS))

    def test_an_unknown_tone_is_refused(self):
        with self.assertRaises(ValueError):
            tone_wav("fanfare-of-doom")


class PlayTests(unittest.TestCase):
    def voice(self, mock=False):
        class V:
            MOCK_MODE = mock

            def __init__(self):
                self.calls = []

            def pause_listening(self):
                self.calls.append("pause")

            def resume_listening(self):
                self.calls.append("resume")

            def _play_audio_stream(self, chunks):
                self.calls.append(("play", len(b"".join(chunks))))

            def _play_mp3_stream(self, chunks):
                self.calls.append(("play-mp3", len(b"".join(chunks))))
        return V()

    def test_plays_through_the_voice_player_with_the_microphone_muted_around_it(self):
        v = self.voice()
        Earcons(v).play("correct")
        self.assertEqual([c if isinstance(c, str) else c[0] for c in v.calls], ["pause", "play", "resume"])
        self.assertGreater(v.calls[1][1], 1000)

    def test_mock_voice_shows_the_tone_instead_of_playing_it(self):
        import contextlib
        v = self.voice(mock=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            e = Earcons(v)
            e.play("wrong")
        self.assertIn("[TONE] wrong", out.getvalue())
        self.assertEqual(v.calls, [])
        self.assertEqual(e.played, ["wrong"])

    def test_a_failing_player_never_raises_and_the_microphone_is_always_unmuted(self):
        v = self.voice()
        v._play_audio_stream = lambda chunks: 1 / 0
        messages = []
        Earcons(v, log=messages.append).play("ready")
        self.assertEqual(v.calls, ["pause", "resume"])
        self.assertTrue(messages)

    def test_disabled_is_silent(self):
        v = self.voice()
        e = Earcons(v, enabled=False)
        e.play("correct")
        self.assertEqual((v.calls, e.played), ([], []))

    def test_routes_to_the_phone_when_the_phone_is_the_speaker(self):
        """The same seam the tutor's voice uses (phonelink.install_phone_speaker), so a tone follows the voice."""
        import phonelink
        v = self.voice()
        link = phonelink.PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        phonelink.install_phone_speaker(v, link)
        Earcons(v).play("correct")  # phone not ready: falls back to the laptop's player
        self.assertIn("play", [c if isinstance(c, str) else c[0] for c in v.calls])


if __name__ == "__main__":
    unittest.main()
