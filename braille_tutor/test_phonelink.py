"""Tests for phonelink.py (phone as camera) and its place in tutor_server. No real phone: uploads are made by the tests, over real
HTTP and real HTTPS with the certificate the code generates. The phone page's JavaScript is NOT run here (no browser)."""
import json
import os
import random
import ssl
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import make_sheet
import phonelink
import tutor
import tutor_server
from phonelink import PhoneCamera, PhoneLink, PhoneServer
from test_tutor import WC, FakeVoice


def jpeg(color=(40, 120, 200), size=(160, 120)) -> bytes:
    img = np.full((size[1], size[0], 3), color, np.uint8)
    cv2.circle(img, (size[0] // 2, size[1] // 2), 20, (255, 255, 255), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def decode_qr(img) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.QRCodeDetector().detectAndDecode(img)[0]


def request(url, data=None, method=None, headers=None, context=None):
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"), headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5, context=context) as r:
            body = r.read()
            return r.status, r.headers, (json.loads(body) if "json" in r.headers.get("Content-Type", "") else body)
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, e.headers, (json.loads(body) if "json" in e.headers.get("Content-Type", "") else body)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class QrTests(unittest.TestCase):
    URL = "https://192.168.1.23:8443/phone?t=482917"

    def test_qr_image_decodes_back_to_the_link(self):
        for size in (240, 480, 900):
            self.assertEqual(decode_qr(phonelink.qr_image(self.URL, size)), self.URL, size)

    def test_qr_png_is_a_png_that_decodes(self):
        png = phonelink.qr_png(self.URL)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(decode_qr(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)), self.URL)

    def test_terminal_qr_has_the_same_pattern_as_the_image(self):
        text = phonelink.qr_terminal(self.URL)
        rows = text.split("\n")
        self.assertGreater(len(rows), 15)
        self.assertEqual(len({len(r) for r in rows}), 1)
        self.assertTrue(set("".join(rows)) <= set(" ▀▄█"))
        # rebuild the module grid from the characters and compare with the encoder's own grid
        dark = phonelink.qr_modules(self.URL)
        rebuilt = np.array([[ch in "▀█" for ch in r] for r in rows]), np.array([[ch in "▄█" for ch in r] for r in rows])
        grid = np.empty((len(rows) * 2, len(rows[0])), bool)
        grid[0::2], grid[1::2] = rebuilt
        h, w = dark.shape
        self.assertTrue(np.array_equal(grid[2:2 + h, 2:2 + w], dark))

    def test_spelled_out_address_is_readable_by_a_voice(self):
        self.assertEqual(phonelink.spell_out("10.0.0.5"), "1 0 dot 0 dot 0 dot 5")

    def test_lan_ip_is_a_dotted_address(self):
        parts = phonelink.lan_ip().split(".")
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(p.isdigit() and 0 <= int(p) <= 255 for p in parts))


class LinkTests(unittest.TestCase):
    def link(self, **kw):
        self.clock = FakeClock()
        return PhoneLink("192.168.1.23", 8443, clock=self.clock, **kw)

    def test_code_is_six_digits_and_in_the_link(self):
        link = self.link()
        self.assertRegex(link.code, r"^\d{6}$")
        self.assertEqual(link.url, f"https://192.168.1.23:8443/phone?t={link.code}")
        self.assertNotEqual(self.link().code, self.link().code)  # (random: a repeat is a one in a million fluke)

    def test_spoken_instructions_give_address_and_code_digit_by_digit(self):
        link = self.link(code="048213")
        text = link.spoken_instructions()
        self.assertIn("1 9 2 dot 1 6 8 dot 1 dot 2 3", text)
        self.assertIn("8 4 4 3", text)
        self.assertIn("0 4 8 2 1 3", text)
        self.assertIn("not private", text)

    def test_wrong_codes_are_refused_and_lock_out_guessing(self):
        link = self.link(code="123456")
        self.assertTrue(link.check_code("123456"))
        for _ in range(phonelink.MAX_BAD_CODES):
            self.assertFalse(link.check_code("000000"))
        self.assertFalse(link.check_code("123456"), "locked out: even the right code waits")
        self.clock.t += phonelink.LOCKOUT_SECONDS + 1
        self.assertTrue(link.check_code("123456"))

    def test_a_right_code_resets_the_count_of_wrong_ones(self):
        link = self.link(code="123456")
        for _ in range(phonelink.MAX_BAD_CODES - 1):
            link.check_code("x")
        self.assertTrue(link.check_code("123456"))
        for _ in range(phonelink.MAX_BAD_CODES - 1):
            link.check_code("x")
        self.assertTrue(link.check_code("123456"))

    def test_missing_or_empty_code_is_refused(self):
        link = self.link(code="123456")
        self.assertFalse(link.check_code(None))
        self.assertFalse(link.check_code(""))

    def test_only_ordinary_images_are_taken(self):
        link = self.link()
        self.assertFalse(link.push(b""))
        self.assertFalse(link.push(b"not an image at all"))
        self.assertFalse(link.push(jpeg()[:200]))  # cut off
        self.assertFalse(link.push(b"\xff" * (phonelink.MAX_FRAME_BYTES + 1)))
        noise = np.random.default_rng(0).integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
        big = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
        self.assertGreater(len(big), phonelink.MAX_FRAME_BYTES, "test precondition: a VALID image over the limit")
        self.assertFalse(link.push(big))
        self.assertFalse(link.push(jpeg(size=(8, 8))))  # too small to be a camera picture
        self.assertEqual((link.frames, link.connected), (0, False))
        self.assertTrue(link.push(jpeg()))
        self.assertEqual((link.frames, link.connected), (1, True))

    def test_connection_goes_stale_and_callbacks_fire_once_each(self):
        events = []
        link = self.link(on_connect=lambda: events.append("up"), on_disconnect=lambda: events.append("down"))
        link.push(jpeg())
        link.push(jpeg())
        time.sleep(0.05)
        self.assertEqual(events, ["up"])
        self.clock.t += phonelink.STALE_SECONDS + 1
        self.assertFalse(link.connected)
        link.check_connection()
        link.check_connection()
        time.sleep(0.05)
        self.assertEqual(events, ["up", "down"])
        link.push(jpeg())
        time.sleep(0.05)
        self.assertEqual(events, ["up", "down", "up"])

    def test_wait_frame_returns_the_newest_and_times_out(self):
        link = self.link()
        self.assertEqual(link.wait_frame(0, 0.01)[0], None)
        link.push(jpeg())
        frame, seq = link.wait_frame(0, 0.01)
        self.assertEqual((frame.shape, seq), ((120, 160, 3), 1))
        start = time.time()
        frame2, seq2 = link.wait_frame(seq, 0.15)  # nothing newer: waits, then gives the same frame back
        self.assertGreaterEqual(time.time() - start, 0.1)
        self.assertEqual(seq2, seq)


class CameraTests(unittest.TestCase):
    def test_setup_screen_is_shown_until_a_phone_connects_and_its_qr_scans(self):
        link = PhoneLink("192.168.1.23", 8443, code="123456")
        cam = PhoneCamera(link)
        ok, frame = cam.read()
        self.assertTrue(ok)
        self.assertEqual(decode_qr(frame), link.url)  # the video window is itself the setup screen: scan it
        link.push(jpeg())
        ok, frame = cam.read()
        self.assertEqual(frame.shape, (120, 160, 3))  # now the phone's picture
        self.assertTrue(cam.isOpened())
        cam.release()
        self.assertFalse(cam.isOpened())
        self.assertEqual(cam.read(), (False, None))

    def test_falls_back_to_the_setup_screen_when_the_phone_goes_quiet(self):
        clock = FakeClock()
        link = PhoneLink("192.168.1.23", 8443, clock=clock)
        cam = PhoneCamera(link)
        link.push(jpeg())
        self.assertEqual(cam.read()[1].shape, (120, 160, 3))
        clock.t += phonelink.STALE_SECONDS + 1
        self.assertEqual(decode_qr(cam.read()[1]), link.url)


class ServerTests(unittest.TestCase):
    """Plain HTTP here (port 0) so the routes can be tested quickly; TLS has its own test below."""

    @classmethod
    def setUpClass(cls):
        cls.link = PhoneLink("127.0.0.1", 0, code="123456", tls=False)
        cls.server = PhoneServer(cls.link, host="127.0.0.1").start()
        cls.base = cls.link.base

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_phone_page_is_served(self):
        for path in ("/phone", "/phone?t=123456", "/"):
            code, headers, body = request(self.base + path)
            self.assertEqual(code, 200, path)
            self.assertIn("text/html", headers["Content-Type"])
            self.assertIn(b"getUserMedia", body)
            self.assertEqual(headers["Cache-Control"], "no-store")

    def test_only_two_kinds_of_thing_are_served(self):
        for path in ("/api/state", "/api/video", "/phone.html", "/../phonelink.py", "/etc/passwd", "/phone/frame"):
            self.assertEqual(request(self.base + path)[0], 404, path)
        self.assertEqual(request(self.base + "/api/command", b"{}")[0], 404)

    def test_check_accepts_only_the_code(self):
        self.assertEqual(request(self.base + "/phone/check?t=123456")[0], 200)
        self.assertEqual(request(self.base + "/phone/check?t=654321")[0], 403)
        self.assertEqual(request(self.base + "/phone/check")[0], 403)
        self.link._bad = 0

    def test_frames_need_the_code_and_arrive(self):
        before = self.link.frames
        code, _, body = request(self.base + "/phone/frame?t=000000", jpeg(), headers={"Content-Type": "image/jpeg"})
        self.assertEqual(code, 403)
        self.assertEqual(self.link.frames, before)
        self.link.status = {"page_ok": True, "message": "page locked"}
        code, headers, body = request(self.base + "/phone/frame?t=123456", jpeg(), headers={"Content-Type": "image/jpeg"})
        self.assertEqual((code, body["ok"], body["page_ok"], body["message"]), (200, True, True, "page locked"))
        self.assertEqual(self.link.frames, before + 1)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"), "no cross-origin access: the page is same-origin")
        self.link._bad = 0

    def test_bad_uploads_are_rejected_and_not_stored(self):
        before = self.link.frames
        url = self.base + "/phone/frame?t=123456"
        self.assertEqual(request(url, b"GIF89a not really", headers={"Content-Type": "image/jpeg"})[0], 400)
        self.assertEqual(request(url, b"", method="POST")[0], 400)
        req = urllib.request.Request(url, data=b"x", method="POST", headers={"Content-Length": str(phonelink.MAX_FRAME_BYTES + 5)})
        code = None
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            code = e.code
        except OSError:
            code = 413  # the server may close the connection instead of waiting for a body that never comes
        self.assertEqual(code, 413)
        self.assertEqual(self.link.frames, before)
        self.link._bad = 0


class DiagnosisTests(unittest.TestCase):
    """Where did a phone that 'doesn't work' get stuck? The laptop must be able to say."""

    def link(self):
        self.clock = FakeClock()
        link = PhoneLink("192.168.1.23", 8443, clock=self.clock, tls=False)
        link.verbose = False
        return link

    def test_the_diagnosis_follows_how_far_the_phone_got(self):
        link = self.link()
        self.assertIsNone(link.diagnose(), "too early to say anything")
        self.clock.t += 20
        self.assertIn("No phone has reached this laptop", link.diagnose())
        self.assertIn("hotspot", link.diagnose())
        link.contact("192.168.1.50")
        self.assertIn("never opened the page", link.diagnose())
        link.page_opens += 1
        self.assertIn("no video has arrived", link.diagnose())
        link.camera_error = "NotAllowedError: Permission denied"
        self.assertIn("NotAllowedError", link.diagnose())
        link.push(jpeg())
        self.assertIsNone(link.diagnose(), "connected: nothing to diagnose")
        self.clock.t += phonelink.STALE_SECONDS + 1
        self.assertIn("stopped sending", link.diagnose())

    def test_events_record_each_step_and_contacts_are_noted_once_per_device(self):
        link = self.link()
        link.contact("10.0.0.7")
        link.contact("10.0.0.7")
        link.contact("10.0.0.8")
        link.push(jpeg())
        texts = [t for _, t in link.events]
        self.assertEqual(sum("reached the laptop" in t for t in texts), 2)
        self.assertEqual(sum("video is arriving" in t for t in texts), 1)

    def test_the_phone_page_can_report_its_own_errors_and_the_laptop_records_them(self):
        link = PhoneLink("127.0.0.1", 0, code="123456", tls=False)
        link.verbose = False
        server = PhoneServer(link, host="127.0.0.1").start()
        try:
            url = f"{link.base}/phone/log?t=123456"
            code, _, _ = request(url, json.dumps({"event": "camera-error", "detail": "NotAllowedError: Permission denied"}).encode(),
                                 headers={"Content-Type": "application/json"})
            self.assertEqual(code, 200)
            self.assertEqual(link.camera_error, "NotAllowedError: Permission denied")
            self.assertTrue(any("phone says camera-error" in t for _, t in link.events))
            self.assertEqual(request(f"{link.base}/phone/log?t=000000", b"{}", headers={"Content-Type": "application/json"})[0], 403)
            request(url, json.dumps({"event": "x" * 500, "detail": "y" * 5000}).encode())  # oversized: ignored or cut, never stored whole
            self.assertTrue(all(len(t) < 300 for _, t in link.events))
            request(f"{link.base}/phone")
            self.assertEqual(link.page_opens, 1)
            self.assertTrue(any("opened the phone page" in t for _, t in link.events))
        finally:
            server.stop()

    def test_a_busy_port_moves_to_the_next_one_and_the_link_knows(self):
        first = PhoneLink("127.0.0.1", 0, tls=False)
        first.verbose = False
        s1 = PhoneServer(first, host="127.0.0.1").start()
        try:
            second = PhoneLink("127.0.0.1", first.port, tls=False)
            second.verbose = False
            s2 = PhoneServer(second, host="127.0.0.1").start()
            try:
                self.assertNotEqual(second.port, first.port)
                self.assertIn(str(second.port), second.url)  # the QR code carries the port actually used
                self.assertEqual(request(f"{second.base}/phone")[0], 200)
            finally:
                s2.stop()
        finally:
            s1.stop()

    def test_state_exposes_the_diagnosis_and_events(self):
        import tutor_server
        link = self.link()
        link.contact("10.0.0.7")
        rt = type("RT", (), {"phone": link})()
        info = tutor_server.TutorRuntime.phone_info(rt)
        self.assertIn("never opened the page", info["diagnosis"])
        self.assertTrue(any("reached the laptop" in e for e in info["events"]))


def wav_bytes(seconds=0.3, rate=8000) -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class FakePhone(threading.Thread):
    """Plays the part of the phone page: fetches every clip the link offers and reports it played (or failed)."""

    def __init__(self, link, error="", delay=0.05):
        super().__init__(daemon=True)
        self.link, self.error, self.delay, self.got, self.stop = link, error, delay, [], False
        self.start()

    def run(self):
        while not self.stop:
            offer = self.link.next_clip()
            if offer:
                clip = self.link.take_clip(offer["id"])
                self.got.append((offer["id"], clip["type"], clip["data"]))
                time.sleep(self.delay)  # "playing"
                self.link.clip_done(offer["id"], self.error)
            time.sleep(0.01)


class SoundTests(unittest.TestCase):
    def ready_link(self, **kw):
        link = PhoneLink("127.0.0.1", 0, tls=False, **kw)
        link.verbose = False
        link.push(jpeg())
        link.set_audio_ready()
        return link

    def test_audio_length_is_exact_for_wav_and_estimated_for_mp3(self):
        self.assertAlmostEqual(phonelink.audio_seconds(wav_bytes(0.5), "audio/wav"), 0.5, places=2)
        self.assertAlmostEqual(phonelink.audio_seconds(b"x" * 32000, "audio/mpeg"), 2.0, places=2)
        self.assertGreater(phonelink.audio_seconds(b"not a wav", "audio/wav"), 0)  # unreadable: still gives a usable estimate

    def lying_wav(self, seconds=0.3):
        """A WAV the way Deepgram streams it: header claims ~2 GB, the file is small."""
        data = bytearray(wav_bytes(seconds))
        data[4:8] = (2147418148).to_bytes(4, "little")
        at = bytes(data).find(b"data")
        data[at + 4:at + 8] = (2147418112).to_bytes(4, "little")
        return bytes(data)

    def test_a_wav_header_that_claims_hours_is_repaired_and_timed_from_the_real_bytes(self):
        import io
        import wave
        lying = self.lying_wav(0.3)
        with wave.open(io.BytesIO(lying)) as w:
            self.assertGreater(w.getnframes() / w.getframerate(), 1000, "test precondition: the header really lies")
        self.assertAlmostEqual(phonelink.audio_seconds(lying, "audio/wav"), 0.3, places=2)
        fixed = phonelink.fix_wav_header(lying)
        with wave.open(io.BytesIO(fixed)) as w:
            self.assertEqual(w.getnframes(), int(0.3 * 8000))
        self.assertEqual(phonelink.fix_wav_header(fixed), fixed, "an already-correct file is unchanged")
        self.assertEqual(phonelink.fix_wav_header(b"not a wav"), b"not a wav")

    def test_what_is_sent_to_the_phone_has_the_repaired_header(self):
        class V:
            MOCK_MODE = False
            def _play_audio_stream(self, chunks): pass
            def _play_mp3_stream(self, chunks): pass
        v, link = V(), self.ready_link()
        phonelink.install_phone_speaker(v, link)
        phone = FakePhone(link)
        try:
            v._play_audio_stream([self.lying_wav(0.2)])
            import io
            import wave
            with wave.open(io.BytesIO(phone.got[0][2])) as w:
                self.assertEqual(w.getnframes(), int(0.2 * 8000))
        finally:
            phone.stop = True

    def test_nothing_is_sent_unless_the_phone_can_play_it(self):
        link = PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        self.assertFalse(link.play_audio(wav_bytes()), "no phone yet")
        link.push(jpeg())
        self.assertFalse(link.play_audio(wav_bytes()), "connected but its sound is not on")
        link.set_audio_ready()
        clock = FakeClock()
        link.clock = clock
        link.last_frame_time = clock() - 100
        self.assertFalse(link.play_audio(wav_bytes()), "sound on but the phone went quiet")

    def test_a_clip_is_delivered_and_the_call_waits_until_it_has_played(self):
        link = self.ready_link()
        phone = FakePhone(link, delay=0.3)
        try:
            data = wav_bytes(0.2)
            start = time.time()
            self.assertTrue(link.play_audio(data))
            self.assertGreaterEqual(time.time() - start, 0.3, "returns only after the phone reports it played")
            self.assertEqual(phone.got, [(1, "audio/wav", data)])
            self.assertIsNone(link.next_clip())
            self.assertTrue(link.play_audio(wav_bytes(0.1), "audio/mpeg"))
            self.assertEqual([g[0] for g in phone.got], [1, 2])  # in order
            self.assertEqual(phone.got[1][1], "audio/mpeg")
        finally:
            phone.stop = True

    def test_a_phone_that_cannot_play_sends_speech_back_to_the_laptop(self):
        link = self.ready_link()
        phone = FakePhone(link, error="NotAllowedError")
        try:
            self.assertFalse(link.play_audio(wav_bytes(0.1)))
            self.assertFalse(link.audio_ready, "stays on the laptop until the phone says its sound is on again")
            self.assertFalse(link.sound_ready())
        finally:
            phone.stop = True

    def test_a_phone_that_never_picks_the_clip_up_times_out_and_the_laptop_takes_over(self):
        link = self.ready_link()
        link.audio_slack = 0.3
        start = time.time()
        self.assertFalse(link.play_audio(wav_bytes(0.1)))
        self.assertLess(time.time() - start, 2.0)
        self.assertFalse(link.audio_ready)

    def test_a_phone_that_took_the_clip_is_trusted_even_if_it_never_reports_back(self):
        """Replaying on the laptop would say the line twice, so once the phone has it, it is assumed to play it."""
        link = self.ready_link()
        link.audio_slack = 0.3
        offer = []
        threading.Thread(target=lambda: (time.sleep(0.05), offer.append(link.take_clip(1))), daemon=True).start()  # fetches, never says done
        start = time.time()
        self.assertTrue(link.play_audio(wav_bytes(0.1)))
        self.assertGreaterEqual(time.time() - start, 0.3)  # waited its length plus the slack, then trusted it
        self.assertTrue(link.audio_ready)

    def test_a_blip_in_the_video_while_the_phone_is_playing_does_not_cause_a_second_playback(self):
        link = self.ready_link()
        clock = FakeClock()
        link.clock = clock
        link.last_frame_time = clock()
        result = []
        threading.Thread(target=lambda: result.append(link.play_audio(wav_bytes(0.1))), daemon=True).start()
        time.sleep(0.15)
        link.take_clip(1)  # the phone starts playing...
        clock.t += phonelink.STALE_SECONDS + 1  # ...and its video stalls for a few seconds
        time.sleep(0.3)
        self.assertEqual(result, [], "still waiting for the phone to finish")
        link.clip_done(1)
        deadline = time.time() + 2
        while not result and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(result, [True])

    def test_a_phone_that_vanishes_before_picking_the_clip_up_gives_it_to_the_laptop(self):
        link = self.ready_link()
        clock = FakeClock()
        link.clock = clock
        link.last_frame_time = clock()
        result = []
        threading.Thread(target=lambda: result.append(link.play_audio(wav_bytes(0.1))), daemon=True).start()
        time.sleep(0.15)
        clock.t += phonelink.STALE_SECONDS + 1
        deadline = time.time() + 2
        while not result and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(result, [False])

    def test_the_phone_is_told_about_a_waiting_clip_in_its_frame_reply(self):
        link = PhoneLink("127.0.0.1", 0, code="123456", tls=False)
        link.verbose = False
        server = PhoneServer(link, host="127.0.0.1").start()
        try:
            hdr = {"Content-Type": "image/jpeg"}
            self.assertIsNone(request(f"{link.base}/phone/frame?t=123456", jpeg(), headers=hdr)[2]["say"])
            self.assertEqual(request(f"{link.base}/phone/audio-ready?t=000000", b"")[0], 403)
            self.assertEqual(request(f"{link.base}/phone/audio-ready?t=123456", b"")[0], 200)
            self.assertTrue(link.sound_ready())
            result = []
            threading.Thread(target=lambda: result.append(link.play_audio(wav_bytes(0.1))), daemon=True).start()
            deadline = time.time() + 3
            reply = None
            while time.time() < deadline:
                reply = request(f"{link.base}/phone/frame?t=123456", jpeg(), headers=hdr)[2]
                if reply["say"]:
                    break
                time.sleep(0.05)
            self.assertEqual(reply["say"], {"id": 1, "type": "audio/wav"})
            self.assertEqual(request(f"{link.base}/phone/audio?t=000000&id=1")[0], 403)
            self.assertEqual(request(f"{link.base}/phone/audio?t=123456&id=99")[0], 404)
            code, headers, body = request(f"{link.base}/phone/audio?t=123456&id=1")
            self.assertEqual((code, headers["Content-Type"], body[:4]), (200, "audio/wav", b"RIFF"))
            self.assertIsNone(request(f"{link.base}/phone/frame?t=123456", jpeg(), headers=hdr)[2]["say"], "offered only once")
            self.assertEqual(request(f"{link.base}/phone/audio-done?t=123456&id=1", b"")[0], 200)
            deadline = time.time() + 3
            while not result and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(result, [True])
        finally:
            server.stop()

    def test_audio_ready_fires_its_callback_once(self):
        calls = []
        link = PhoneLink("127.0.0.1", 0, tls=False, on_audio_ready=lambda: calls.append(1))
        link.verbose = False
        link.set_audio_ready()
        link.set_audio_ready()
        time.sleep(0.05)
        self.assertEqual(calls, [1])


def pcm(n_samples, value=100) -> bytes:
    return (int(value).to_bytes(2, "little", signed=True)) * n_samples


class PhoneMicTests(unittest.TestCase):
    def link(self):
        self.clock = FakeClock()
        link = PhoneLink("127.0.0.1", 0, tls=False, clock=self.clock)
        link.verbose = False
        return link

    def test_audio_comes_out_in_order_in_the_chunks_asked_for(self):
        link = self.link()
        self.assertIsNone(link.mic_take(3200), "nothing yet")
        link.mic_push(pcm(1600, 1) + pcm(1600, 2))
        self.assertEqual(link.mic_take(3200), pcm(1600, 1))
        self.assertEqual(link.mic_take(3200), pcm(1600, 2))
        self.assertIsNone(link.mic_take(3200), "drained: the caller uses the laptop microphone for that moment")
        self.assertTrue(link.mic_live())

    def test_only_the_newest_audio_is_kept_so_a_command_is_never_heard_late(self):
        link = self.link()
        for i in range(20):  # 20 x 0.25 s = 5 s arrives without being taken
            link.mic_push(pcm(4000, i))
        kept = phonelink.MIC_RATE * 2 * phonelink.MIC_MAX_BACKLOG_SECONDS
        self.assertLessEqual(len(link._mic), kept)
        first = int.from_bytes(link.mic_take(3200)[:2], "little", signed=True)
        self.assertGreaterEqual(first, 17, "only the last 0.6 s (chunks 17-19) is left: the older audio was dropped")

    def test_it_goes_quiet_when_the_phone_stops_and_stale_audio_is_thrown_away(self):
        link = self.link()
        link.mic_push(pcm(4000))
        self.clock.t += phonelink.MIC_LIVE_SECONDS + 1
        self.assertFalse(link.mic_live())
        self.assertIsNone(link.mic_take(3200), "old audio is not sent after a gap")
        link.mic_push(pcm(4000, 7))
        self.assertTrue(link.mic_live())
        self.assertEqual(link.mic_take(3200)[:2], (7).to_bytes(2, "little", signed=True))

    def test_bad_chunks_are_refused(self):
        link = self.link()
        self.assertFalse(link.mic_push(b""))
        self.assertFalse(link.mic_push(b"\x00" * 3), "odd length: not 16-bit")
        self.assertFalse(link.mic_push(b"\x00" * (phonelink.MAX_MIC_CHUNK_BYTES + 2)))
        self.assertFalse(link.mic_live())

    def test_the_tutors_own_voice_heard_by_the_phone_microphone_is_dropped(self):
        link = PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        link.push(jpeg())
        link.set_audio_ready()
        phone = FakePhone(link, delay=0.1)
        try:
            link.mic_push(pcm(3200, 5))
            self.assertTrue(link.play_audio(wav_bytes(0.1)))
            self.assertIsNone(link.mic_take(3200), "what was waiting when the tutor spoke is discarded")
            link.mic_push(pcm(3200, 6))  # arrives right after it finished speaking: its tail
            self.assertIsNone(link.mic_take(3200))
            self.assertTrue(link.mic_live(), "the microphone itself is still known to be on")
            time.sleep(phonelink.MIC_TAIL_SECONDS + 0.1)
            link.mic_push(pcm(3200, 8))
            self.assertEqual(link.mic_take(3200)[:2], (8).to_bytes(2, "little", signed=True))
        finally:
            phone.stop = True

    def test_the_mic_route_needs_the_code_and_real_pcm(self):
        link = PhoneLink("127.0.0.1", 0, code="123456", tls=False)
        link.verbose = False
        server = PhoneServer(link, host="127.0.0.1").start()
        try:
            url = f"{link.base}/phone/mic?t=123456"
            hdr = {"Content-Type": "application/octet-stream"}
            self.assertEqual(request(f"{link.base}/phone/mic?t=000000", pcm(1600), headers=hdr)[0], 403)
            self.assertEqual(request(url, pcm(1600), headers=hdr)[0], 200)
            self.assertTrue(link.mic_live())
            self.assertEqual(request(url, b"\x00" * 3, headers=hdr)[0], 400)
            self.assertEqual(link.mic_take(3200), pcm(1600))
        finally:
            server.stop()

    def fake_pyaudio(self):
        """A stand-in for the pyaudio module: an input stream whose read() returns the laptop microphone (value 1)."""
        import types
        mod = types.ModuleType("pyaudio")

        class Stream:
            def read(self, num_frames, exception_on_overflow=True):
                return pcm(num_frames, 1)
        mod.Stream = Stream
        return mod

    def voice(self, rate=16000, channels=1):
        return type("V", (), {"SAMPLE_RATE": rate, "CHANNELS": channels})()

    def test_voice_io_hears_the_phone_microphone_when_it_is_live_and_the_laptop_one_otherwise(self):
        import sys
        from unittest import mock
        link = self.link()
        with mock.patch.dict(sys.modules, {"pyaudio": self.fake_pyaudio()}):
            import pyaudio
            original = pyaudio.Stream.read
            restore = phonelink.install_phone_mic(self.voice(), link)
            stream = pyaudio.Stream()
            self.assertEqual(stream.read(1600), pcm(1600, 1), "phone silent: the laptop microphone, as before")
            link.mic_push(pcm(1600, 9))
            self.assertEqual(stream.read(1600), pcm(1600, 9), "phone streaming: its audio")
            self.assertEqual(stream.read(1600), pcm(1600, 1), "phone ran dry for a moment: laptop for that chunk")
            restore()
            link.mic_push(pcm(1600, 9))
            self.assertEqual(stream.read(1600), pcm(1600, 1), "restored: only the laptop microphone")
            self.assertIs(pyaudio.Stream.read, original)

    def test_it_refuses_when_the_formats_do_not_match_or_pyaudio_is_missing(self):
        import sys
        from unittest import mock
        link = self.link()
        with mock.patch.dict(sys.modules, {"pyaudio": self.fake_pyaudio()}):
            with self.assertRaises(RuntimeError) as ctx:
                phonelink.install_phone_mic(self.voice(rate=44100), link)
            self.assertIn("--mic laptop", str(ctx.exception))
            with self.assertRaises(RuntimeError):
                phonelink.install_phone_mic(self.voice(channels=2), link)
        with mock.patch.dict(sys.modules, {"pyaudio": None}):  # import fails
            with self.assertRaises(RuntimeError):
                phonelink.install_phone_mic(self.voice(), link)

    def test_the_real_voice_io_asks_for_exactly_the_format_the_phone_sends(self):
        import tutor
        voice, _ = tutor.load_teammate_modules(mock=True)
        self.assertEqual((voice.SAMPLE_RATE, voice.CHANNELS), (phonelink.MIC_RATE, 1))

    def test_a_voice_module_without_the_playback_functions_does_not_stop_the_tutor_starting(self):
        import argparse
        import contextlib
        import io
        ap = argparse.ArgumentParser()
        phonelink.add_phone_args(ap)
        v = self.voice()
        v.MOCK_MODE = False  # has neither _play_audio_stream nor _play_mp3_stream
        a = ap.parse_args(["--phone-camera", "--phone-host", "127.0.0.1", "--phone-port", "0", "--mic", "laptop"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            link, cam, server = phonelink.start_phone(a, voice=v)
        server.stop()
        self.assertIn("will stay on the laptop", out.getvalue())

    def test_start_phone_installs_the_mic_only_when_asked(self):
        import argparse
        import contextlib
        import io
        import sys
        from unittest import mock
        ap = argparse.ArgumentParser()
        phonelink.add_phone_args(ap)
        with mock.patch.dict(sys.modules, {"pyaudio": self.fake_pyaudio()}):
            import pyaudio
            for extra, mock_voice, expect in (([], False, True), (["--mic", "laptop"], False, False), (["--mic", "phone"], True, False)):
                original = pyaudio.Stream.read
                v = self.voice()
                v.MOCK_MODE = mock_voice
                a = ap.parse_args(["--phone-camera", "--phone-host", "127.0.0.1", "--phone-port", "0"] + extra)
                with contextlib.redirect_stdout(io.StringIO()):
                    link, cam, server = phonelink.start_phone(a, voice=v)
                try:
                    self.assertEqual(pyaudio.Stream.read is not original, expect, (extra, mock_voice))
                finally:
                    server.stop()
                    pyaudio.Stream.read = original


class SpeakerRedirectTests(unittest.TestCase):
    def voice(self):
        class V:
            MOCK_MODE = False

            def __init__(self):
                self.local = []

            def _play_audio_stream(self, chunks):
                self.local.append(("wav", b"".join(chunks)))

            def _play_mp3_stream(self, chunks):
                self.local.append(("mp3", b"".join(chunks)))

            def speak(self, text, mode="normal"):  # what voice_io does: synthesize, then hand the audio to the player
                if mode == "debrief":
                    self._play_mp3_stream([b"mp3-", text.encode()])
                else:
                    self._play_audio_stream([wav_bytes(0.05), b""])
        return V()

    def test_speech_goes_to_the_phone_when_it_is_ready_and_to_the_laptop_otherwise(self):
        v = self.voice()
        link = PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        restore = phonelink.install_phone_speaker(v, link)
        v.speak("hello")
        self.assertEqual([k for k, _ in v.local], ["wav"], "phone not ready: the laptop plays it, exactly as before")
        v.local.clear()
        link.push(jpeg())
        link.set_audio_ready()
        phone = FakePhone(link)
        try:
            v.speak("hello")
            v.speak("the debrief", mode="debrief")
            self.assertEqual(v.local, [], "the laptop stayed silent")
            self.assertEqual([(g[0], g[1]) for g in phone.got], [(1, "audio/wav"), (2, "audio/mpeg")])
            self.assertEqual(phone.got[1][2], b"mp3-the debrief")
        finally:
            phone.stop = True
        restore()
        v.speak("hello")
        self.assertEqual([k for k, _ in v.local], ["wav"])  # put back

    def test_it_refuses_to_install_over_a_voice_module_it_does_not_understand(self):
        with self.assertRaises(RuntimeError) as ctx:
            phonelink.install_phone_speaker(type("V", (), {"speak": lambda *a: None})(), PhoneLink("127.0.0.1", 0, tls=False))
        self.assertIn("--sound laptop", str(ctx.exception))

    def test_the_real_voice_io_still_has_the_two_functions_we_redirect(self):
        import tutor
        voice, _ = tutor.load_teammate_modules(mock=True)
        for name in ("_play_audio_stream", "_play_mp3_stream"):
            self.assertTrue(callable(getattr(voice, name, None)), f"voice_io.{name} is gone: the phone speaker redirect would break")

    @unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
    def test_the_tutors_own_speech_lines_are_heard_on_the_phone_one_at_a_time(self):
        v = self.voice()
        link = PhoneLink("127.0.0.1", 0, tls=False)
        link.verbose = False
        phonelink.install_phone_speaker(v, link)
        link.push(jpeg())
        link.set_audio_ready()
        phone = FakePhone(link, delay=0.1)
        session = tutor.TutorSession(v, WC, [], lambda: None, mode="explore")
        try:
            threads = [threading.Thread(target=session.say, args=(f"line {i}",)) for i in range(4)]
            [x.start() for x in threads]
            [x.join(10) for x in threads]
            self.assertEqual(len(phone.got), 4)
            self.assertEqual(v.local, [])
        finally:
            phone.stop = True

    def test_start_phone_installs_the_redirect_only_when_asked_and_only_for_real_voices(self):
        import argparse
        import contextlib
        import io
        ap = argparse.ArgumentParser()
        phonelink.add_phone_args(ap)
        for extra, mock, expect in ((["--sound", "phone"], False, True), ([], False, True), (["--sound", "laptop"], False, False), (["--sound", "phone"], True, False)):
            v = self.voice()
            v.MOCK_MODE = mock
            original = v._play_audio_stream
            a = ap.parse_args(["--phone-camera", "--phone-host", "127.0.0.1", "--phone-port", "0"] + extra)
            with contextlib.redirect_stdout(io.StringIO()):
                link, cam, server = phonelink.start_phone(a, voice=v)
            try:
                self.assertEqual(v._play_audio_stream != original, expect, (extra, mock))
            finally:
                server.stop()


@unittest.skipUnless(Path("/usr/bin/openssl").exists() or os.system("openssl version > /dev/null 2>&1") == 0, "no openssl command")
class TlsTests(unittest.TestCase):
    def test_a_real_https_upload_with_the_generated_certificate_verified(self):
        directory = Path(tempfile.mkdtemp())
        cert, key = phonelink.ensure_cert("127.0.0.1", directory)
        self.assertEqual((directory / Path(key).name).stat().st_mode & 0o077, 0, "the private key must not be readable by others")
        self.assertEqual(phonelink.ensure_cert("127.0.0.1", directory), (cert, key), "made once, then reused")
        link = PhoneLink("127.0.0.1", 0, code="123456")
        server = PhoneServer(link, host="127.0.0.1", cert=(cert, key)).start()
        try:
            ctx = ssl.create_default_context(cafile=cert)  # a phone would need to be told to trust it; here we do, and check the address matches
            code, _, body = request(f"https://127.0.0.1:{link.port}/phone/frame?t=123456", jpeg(), headers={"Content-Type": "image/jpeg"}, context=ctx)
            self.assertEqual((code, body["ok"]), (200, True))
            self.assertTrue(link.connected)
            with self.assertRaises(urllib.error.URLError):  # without trusting it, the connection is refused: it really is TLS
                urllib.request.urlopen(f"https://127.0.0.1:{link.port}/phone", timeout=5, context=ssl.create_default_context())
            with self.assertRaises(Exception):  # and plain http to the TLS port does not work
                urllib.request.urlopen(f"http://127.0.0.1:{link.port}/phone", timeout=3)
        finally:
            server.stop()


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class EndToEndTests(unittest.TestCase):
    """A 'phone' uploads photos of the printed sheet; the tutor must find the page from them and tell the phone so."""

    @classmethod
    def setUpClass(cls):
        face = make_sheet.render_face_preview()
        h, w = face.shape[:2]
        T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32([[160, 60], [1000, 120], [1080, 880], [90, 840]]))
        cls.photo = cv2.imencode(".jpg", cv2.warpPerspective(face, T, (1280, 960), borderValue=255))[1].tobytes()
        cls.link = PhoneLink("127.0.0.1", 0, code="123456", tls=False)
        cls.server = PhoneServer(cls.link, host="127.0.0.1").start()
        cells = make_sheet.sheet_cells()
        cls.voice = tutor_server.RecordingVoice(FakeVoice())
        cls.feed = tutor.CameraFeed(None, cells)
        cls.session = tutor.TutorSession(cls.voice, WC, cells, cls.feed.finger, cls.feed.scan, "letters", 1, rng=random.Random(5))
        cls.session.attach()
        cls.rt = tutor_server.TutorRuntime(PhoneCamera(cls.link), cls.feed, cls.session, cls.voice, phone=cls.link)
        cls.rt.start()
        cls.api = tutor_server.serve(cls.rt, port=0)
        cls.api_base = f"http://127.0.0.1:{cls.api.server_address[1]}"
        threading.Thread(target=cls.api.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.rt.stop()
        cls.api.shutdown()
        cls.api.server_close()
        cls.server.stop()

    def test_setup_screen_then_phone_frames_then_page_found(self):
        state = request(self.api_base + "/api/state")[2]
        self.assertFalse(state["phone"]["connected"])
        self.assertEqual(state["phone"]["code"], "123456")
        self.assertIn("qr", state["phone"])
        code, headers, png = request(self.api_base + "/api/phone/qr.png")
        self.assertEqual((code, headers["Content-Type"]), (200, "image/png"))
        self.assertEqual(decode_qr(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)), self.link.url)
        # the frontend's video already shows the QR before any phone connects
        with urllib.request.urlopen(self.api_base + "/api/video", timeout=5) as r:
            self.assertIn("multipart", r.headers["Content-Type"])
        # the phone starts uploading
        deadline, reply = time.time() + 15, None
        while time.time() < deadline:
            code, _, reply = request(f"{self.server.link.base}/phone/frame?t=123456", self.photo, headers={"Content-Type": "image/jpeg"})
            self.assertEqual(code, 200)
            if reply["page_ok"]:
                break
            time.sleep(0.1)
        self.assertTrue(reply["page_ok"], reply)  # the phone is told the page was found (it says so out loud)
        state = request(self.api_base + "/api/state")[2]
        self.assertTrue(state["phone"]["connected"])
        self.assertTrue(state["page"]["ok"], state["page"])
        self.assertTrue(state["camera"]["ok"])

    def test_without_a_phone_camera_the_endpoint_says_so(self):
        feed = tutor.CameraFeed(None, None)
        rt = tutor_server.TutorRuntime(PhoneCamera(PhoneLink("127.0.0.1", 0)), feed, self.session, self.voice)  # no phone= given
        server = tutor_server.serve(rt, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            self.assertEqual(request(base + "/api/phone/qr.png")[0], 404)
            self.assertIsNone(request(base + "/api/state")[2]["phone"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
