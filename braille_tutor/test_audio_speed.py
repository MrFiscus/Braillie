"""Tests for audio_speed.py: slower and faster speech keeps its pitch and its shape, and only speech is touched."""
import io
import unittest
import wave

import numpy as np

import audio_speed
from audio_speed import clamp, install_speech_speed, stretch_wav, time_stretch

RATE = 24000


def sine(hz, seconds=1.0, amp=0.4):
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def speechlike(seconds=2.0):
    """Syllable-like bursts of a voice-ish tone (200 Hz + harmonics) with gaps: enough structure to see timing preserved."""
    x = np.zeros(int(RATE * seconds), np.float32)
    t = np.arange(len(x)) / RATE
    tone = sum(a * np.sin(2 * np.pi * 200 * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3))).astype(np.float32) * 0.2
    for start in np.arange(0.1, seconds - 0.3, 0.45):
        i, j = int(start * RATE), int((start + 0.28) * RATE)
        env = np.hanning(j - i).astype(np.float32)
        x[i:j] = tone[i:j] * env
    return x


def wav_of(x, rate=RATE, lie=False):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    data = bytearray(buf.getvalue())
    if lie:  # the way Deepgram streams it: a header that claims about 12 hours
        data[4:8] = (2147418148).to_bytes(4, "little")
        at = bytes(data).find(b"data")
        data[at + 4:at + 8] = (2147418112).to_bytes(4, "little")
    return bytes(data)


def read(data):
    with wave.open(io.BytesIO(data)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, w.getframerate()


def dominant_hz(x):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[spectrum.argmax()])


class StretchTests(unittest.TestCase):
    def test_slower_is_longer_and_faster_is_shorter_by_the_asked_amount(self):
        x = speechlike(2.0)
        for speed in (0.6, 0.75, 0.85, 1.15, 1.3, 1.5):
            with self.subTest(speed=speed):
                y = time_stretch(x, speed, RATE)
                self.assertAlmostEqual(len(y) / len(x), 1 / speed, delta=0.04)

    def test_the_pitch_does_not_change(self):
        x = sine(220, 1.5)
        for speed in (0.7, 0.85, 1.25):
            with self.subTest(speed=speed):
                y = time_stretch(x, speed, RATE)
                middle = y[len(y) // 4: 3 * len(y) // 4]
                self.assertAlmostEqual(dominant_hz(middle), 220, delta=6, msg=f"speed {speed}")

    def test_the_shape_of_the_speech_is_kept_syllables_stay_and_gaps_stay_gaps(self):
        x = speechlike(2.0)
        y = time_stretch(x, 0.8, RATE)
        def bursts(sig):
            env = np.convolve(np.abs(sig), np.ones(int(0.02 * RATE)) / int(0.02 * RATE), mode="same")
            on = env > 0.25 * env.max()
            return int(np.sum(np.diff(on.astype(int)) == 1))
        self.assertEqual(bursts(y), bursts(x), "the same number of syllables")
        gap_fraction = lambda s: float(np.mean(np.abs(s) < 0.01))
        self.assertAlmostEqual(gap_fraction(y), gap_fraction(x), delta=0.06, msg="the pauses are stretched too, not squeezed out")

    def test_no_clipping_no_nans_and_the_level_is_kept(self):
        rng = np.random.default_rng(0)
        x = (0.3 * rng.standard_normal(RATE)).clip(-1, 1).astype(np.float32)
        for speed in (0.6, 0.9, 1.4):
            y = time_stretch(x, speed, RATE)
            self.assertTrue(np.isfinite(y).all())
            self.assertLessEqual(float(np.abs(y).max()), 1.05 * float(np.abs(x).max()) + 0.05)
            self.assertAlmostEqual(float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(x ** 2))), delta=0.06)

    def test_silence_stays_silent(self):
        y = time_stretch(np.zeros(RATE, np.float32), 0.8, RATE)
        self.assertLess(float(np.abs(y).max()), 1e-6)

    def test_normal_speed_and_very_short_clips_are_returned_untouched(self):
        x = speechlike(1.0)
        self.assertIs(time_stretch(x, 1.0, RATE), x)
        self.assertIs(time_stretch(x, 1.005, RATE), x)
        short = sine(300, 0.02)
        self.assertIs(time_stretch(short, 0.7, RATE), short)

    def test_speeds_outside_the_range_are_clamped(self):
        self.assertEqual((clamp(0.1), clamp(9.0), clamp(1.0)), (0.6, 1.5, 1.0))
        x = speechlike(1.5)
        self.assertAlmostEqual(len(time_stretch(x, 0.2, RATE)) / len(x), 1 / 0.6, delta=0.05)

    def test_it_is_fast_enough_not_to_hold_up_the_tutor(self):
        import time
        x = speechlike(8.0)  # a long instruction
        start = time.time()
        time_stretch(x, 0.75, RATE)
        self.assertLess(time.time() - start, 2.5, "well under the time it takes to say it")


class WavTests(unittest.TestCase):
    def test_a_wav_comes_out_as_a_valid_wav_at_the_new_speed(self):
        data = wav_of(speechlike(2.0))
        out = stretch_wav(data, 0.8)
        y, rate = read(out)
        self.assertEqual(rate, RATE)
        self.assertAlmostEqual(len(y) / RATE, 2.0 / 0.8, delta=0.1)

    def test_the_header_deepgram_sends_is_handled(self):
        data = wav_of(speechlike(1.0), lie=True)
        y, _ = read(stretch_wav(data, 0.75))
        self.assertAlmostEqual(len(y) / RATE, 1.0 / 0.75, delta=0.08)

    def test_normal_speed_returns_the_same_bytes(self):
        data = wav_of(speechlike(1.0))
        self.assertIs(stretch_wav(data, 1.0), data)

    def test_things_it_cannot_stretch_are_returned_unchanged_never_broken(self):
        for junk in (b"", b"not audio at all", b"RIFF" + b"\x00" * 40):
            self.assertEqual(stretch_wav(junk, 0.8), junk)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:  # stereo
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"\x00\x00" * 2 * RATE)
        self.assertEqual(stretch_wav(buf.getvalue(), 0.8), buf.getvalue())


class InstallTests(unittest.TestCase):
    def voice(self):
        class V:
            def __init__(self):
                self.played = []

            def _play_audio_stream(self, chunks):
                self.played.append(b"".join(chunks))
        return V()

    def test_speech_is_stretched_at_the_speed_asked_for_at_that_moment(self):
        v, speed = self.voice(), [1.0]
        restore = install_speech_speed(v, lambda: speed[0])
        data = wav_of(speechlike(1.0))
        v._play_audio_stream([data])
        self.assertEqual(v.played[-1], data, "normal speed: untouched")
        speed[0] = 0.75
        v._play_audio_stream([data[:len(data) // 2], data[len(data) // 2:]])  # arrives in pieces, as it does from Deepgram
        y, _ = read(v.played[-1])
        self.assertAlmostEqual(len(y) / RATE, 1.0 / 0.75, delta=0.08)
        restore()
        v._play_audio_stream([data])
        self.assertEqual(v.played[-1], data)

    def test_a_tone_inside_raw_is_never_stretched(self):
        v = self.voice()
        install_speech_speed(v, lambda: 0.6)
        tone = wav_of(sine(660, 0.3))
        with audio_speed.raw():
            v._play_audio_stream([tone])
        self.assertEqual(v.played[-1], tone)
        v._play_audio_stream([tone])
        self.assertNotEqual(v.played[-1], tone, "but speech-like audio outside raw() is")

    def test_earcons_use_raw_so_a_slow_voice_does_not_slow_the_little_sounds(self):
        import earcons
        v = self.voice()
        v.MOCK_MODE = False
        install_speech_speed(v, lambda: 0.6)
        earcons.Earcons(v).play("correct")
        self.assertEqual(v.played[-1], earcons.tone_wav("correct"))

    def test_it_refuses_a_voice_module_without_the_player(self):
        with self.assertRaises(RuntimeError):
            install_speech_speed(type("V", (), {})(), lambda: 1.0)

    def test_it_works_on_top_of_the_phone_speaker_so_the_phone_gets_the_slow_version(self):
        import phonelink
        v = self.voice()
        v._play_mp3_stream = lambda chunks: None
        link = phonelink.PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        link.push(__import__("cv2").imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))[1].tobytes())
        link.set_audio_ready()
        phonelink.install_phone_speaker(v, link)
        install_speech_speed(v, lambda: 0.75)  # installed after: what goes to the phone is already slowed
        sent = []
        import threading, time
        def fake_phone():
            while not sent:
                offer = link.next_clip()
                if offer:
                    sent.append(link.take_clip(offer["id"])["data"])
                    link.clip_done(offer["id"])
                time.sleep(0.01)
        threading.Thread(target=fake_phone, daemon=True).start()
        v._play_audio_stream([wav_of(speechlike(1.0))])
        y, _ = read(sent[0])
        self.assertAlmostEqual(len(y) / RATE, 1.0 / 0.75, delta=0.08)


if __name__ == "__main__":
    unittest.main()
