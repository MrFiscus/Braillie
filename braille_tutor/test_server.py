"""Tests for tutor_server.py: the HTTP API a browser frontend uses. Run: python -m unittest test_server -v"""
import json
import os
import random
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

import make_sheet
import tutor
import tutor_server
from detect import letter_of
from test_tutor import WC, FakeVoice


def http(url, body=None, origin=None, method=None, raw=None):
    """(status, headers, parsed body) for a request; HTTP errors are returned, not raised."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            payload = r.read()
            return r.status, r.headers, (json.loads(payload) if "json" in r.headers.get("Content-Type", "") else payload)
    except urllib.error.HTTPError as e:
        payload = e.read()
        return e.code, e.headers, (json.loads(payload) if "json" in e.headers.get("Content-Type", "") else payload)


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        face = make_sheet.render_face_preview()
        h, w = face.shape[:2]
        quad = np.float32([[160, 60], [1000, 120], [1080, 880], [90, 840]])
        cls.T = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
        frame = cv2.cvtColor(cv2.warpPerspective(face, cls.T, (1280, 960), borderValue=255), cv2.COLOR_GRAY2BGR)
        path = os.path.join(tempfile.mkdtemp(), "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 960))
        for _ in range(12):
            vw.write(frame)
        vw.release()
        cls.fake = FakeVoice()
        cls.voice = tutor_server.RecordingVoice(cls.fake)
        cells = make_sheet.sheet_cells()
        cls.feed = tutor.CameraFeed(None, cells)
        cls.session = tutor.TutorSession(cls.voice, WC, cells, cls.feed.finger, cls.feed.scan, "letters", 1, rng=random.Random(5))
        cls.session.attach()
        cls.rt = tutor_server.TutorRuntime(tutor.open_camera(path), cls.feed, cls.session, cls.voice, loop_file=True)
        cls.rt.start()
        cls.server = tutor_server.serve(cls.rt, port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        end = time.time() + 10
        while time.time() < end and not (cls.feed.frames > 3 and cls.feed.page_ok):
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.rt.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def wait_for(self, predicate, seconds=5.0):
        end = time.time() + seconds
        while time.time() < end:
            state = http(self.base + "/api/state")[2]
            if predicate(state):
                return state
            time.sleep(0.05)
        self.fail(f"condition not reached; last state: {state}")

    # ---- reading ----
    def test_state_snapshot(self):
        code, _, s = http(self.base + "/api/state")
        self.assertEqual(code, 200)
        self.assertTrue(s["page"]["ok"], s["page"])
        self.assertTrue(s["camera"]["ok"])
        self.assertEqual(s["config"]["mode"], "letters")
        self.assertEqual(sorted(s["config"]["commands"]), sorted(tutor_server.COMMANDS))
        self.assertIn("state", s["tutor"])

    def test_cells_endpoint(self):
        code, _, body = http(self.base + "/api/cells")
        self.assertEqual(code, 200)
        self.assertEqual(len(body["cells"]), 26)
        self.assertEqual(body["cells"][0]["letter"], "a")
        self.assertEqual(body["cells"][0]["dots"], [1])

    def test_video_stream_delivers_a_jpeg(self):
        with urllib.request.urlopen(self.base + "/api/video", timeout=5) as r:
            self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
            head = r.read(2000)
            m = re.search(rb"Content-Length: (\d+)\r\n\r\n", head)
            self.assertIsNotNone(m)
            size = int(m.group(1))
            jpeg = head[m.end():] + r.read(size - len(head[m.end():]))
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        self.assertLessEqual(img.shape[1], tutor_server.STREAM_WIDTH)

    def test_event_stream_pushes_state(self):
        with urllib.request.urlopen(self.base + "/api/events", timeout=5) as r:
            self.assertIn("text/event-stream", r.headers["Content-Type"])
            line = r.readline()
        self.assertTrue(line.startswith(b"data: "))
        self.assertIn("tutor", json.loads(line[6:]))

    def test_console_page_and_unknown_paths(self):
        code, headers, page = http(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn(b"Braillie tutor console", page)
        self.assertEqual(http(self.base + "/nope")[0], 404)
        self.assertEqual(http(self.base + "/../../etc/passwd")[0], 404)  # only fixed routes exist
        self.assertEqual(http(self.base + "/api/command", raw=b"{}", method="PUT")[0] in (404, 501), True)

    # ---- validation and security ----
    def test_bad_requests_are_rejected(self):
        self.assertEqual(http(self.base + "/api/command", raw=b"not json")[0], 400)
        self.assertEqual(http(self.base + "/api/command", raw=b"[1,2]")[0], 400)
        code, _, body = http(self.base + "/api/command", {"command": "self destruct"})
        self.assertEqual(code, 400)
        self.assertIn("valid", body)
        self.assertEqual(http(self.base + "/api/command", raw=b"x" * 5000)[0], 413)
        for bad in ({"u": 2, "v": 0.5}, {"u": "abc", "v": 0.5}, {"u": float("nan"), "v": 0.5}, {"x_mm": 1e9, "y_mm": 0}, {}):
            self.assertEqual(http(self.base + "/api/finger", bad)[0], 400, bad)

    def test_cors_only_for_local_origins(self):
        code, headers, _ = http(self.base + "/api/command", {"command": "nope"}, origin="http://localhost:5173")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        code, headers, _ = http(self.base + "/api/state", origin="http://127.0.0.1:3000")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:3000")
        code, headers, _ = http(self.base + "/api/command", {"command": "start quiz"}, origin="https://evil.example")
        self.assertEqual(code, 403)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))
        self.assertEqual(self.session.state, "idle")  # the forbidden request did not run the command
        code, headers, _ = http(self.base + "/api/command", origin="http://localhost:5173", method="OPTIONS")
        self.assertEqual(code, 204)
        self.assertIn("POST", headers["Access-Control-Allow-Methods"])
        self.assertEqual(http(self.base + "/api/command", origin="https://evil.example", method="OPTIONS")[0], 403)

    # ---- a whole quiz, over HTTP only (runs last: it changes the session) ----
    def test_zz_full_quiz_over_http(self):
        self.assertEqual(http(self.base + "/api/command", {"command": "found it"})[0], 202)  # before start: polite refusal
        self.wait_for(lambda s: any("start quiz" in m["text"] for m in s["said"]))
        self.assertEqual(http(self.base + "/api/command", {"command": "start_quiz"})[0], 202)  # underscore form is accepted
        s = self.wait_for(lambda s: s["tutor"]["state"] == "asking")
        target = re.search(r"letter (\w)\.", s["tutor"]["prompt"]).group(1).lower()
        cell = next(c for c in make_sheet.sheet_cells() if letter_of(c["dots"]) == target)
        p = self.T @ np.array([(make_sheet.ORIGIN[0] + cell["x"]) * make_sheet.MM, (make_sheet.ORIGIN[1] + cell["y"]) * make_sheet.MM, 1.0])
        self.assertEqual(http(self.base + "/api/finger", {"u": p[0] / p[2] / 1280, "v": p[1] / p[2] / 960})[0], 200)
        s = self.wait_for(lambda s: s["finger"]["cell"] and s["finger"]["cell"]["letter"] == target)
        self.assertIsNotNone(s["finger"]["page_mm"])
        http(self.base + "/api/command", {"command": "found it"})
        s = self.wait_for(lambda s: s["tutor"]["state"] == "done")
        self.assertTrue(any(m["text"].startswith("Correct!") for m in s["said"]), s["said"])
        self.assertEqual(s["debrief"], {"accuracy": 1.0, "missed": []})
        self.assertEqual(self.fake.debriefs[-1], (1.0, []))  # and it really went to the voice module
        self.assertEqual(http(self.base + "/api/finger", {"clear": True})[0], 200)
        self.wait_for(lambda s: s["finger"]["page_mm"] is None)


if __name__ == "__main__":
    unittest.main()
