"""Local web server for the tutor, so a browser frontend can show the camera and drive the session. Standard library only.

    python tutor_server.py --mock --mode letters          then open http://127.0.0.1:8000

  GET  /                 a small built-in console page (web/index.html)
  GET  /api/state        JSON snapshot: page status, quiz state, finger, what was said
  GET  /api/events       the same snapshot as a Server-Sent Events stream, pushed when it changes
  GET  /api/video        the camera as an MJPEG stream (use it as an <img src>)
  GET  /api/cells        the letter-quiz layout cells in page mm (for drawing your own overlay)
  POST /api/command      {"command": "start quiz" | "repeat" | "hint" | "found it" | "next" | "stop"}
  POST /api/finger       {"u": 0..1, "v": 0..1} (a point on the video, as fractions)  |  {"x_mm":..,"y_mm":..}  |  {"clear": true}

Only pages served from localhost / 127.0.0.1 may call the POST endpoints, and the server binds to 127.0.0.1 by default.
See API.md for how to use it from the React frontend.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2

import tutor
from detect import letter_of, nearest_cell, open_camera, page_source_from_args

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 4096
COMMANDS = {"start quiz": "on_start", "repeat": "on_repeat", "hint": "on_hint", "found it": "on_found_it",
            "next": "on_next", "stop": "on_stop"}
LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
FPS, STREAM_WIDTH = 15, 960


class RecordingVoice:
    """Wraps voice_io: everything is still spoken, and also kept so a display can show what was said."""

    def __init__(self, voice):
        self.voice, self.said, self.debrief = voice, collections.deque(maxlen=30), None

    def speak(self, text: str, mode: str = "normal") -> None:
        self.said.append({"t": time.time(), "kind": "speech", "text": text})
        self.voice.speak(text, mode)

    def speak_debrief(self, accuracy: float, missed: list) -> None:
        self.debrief = {"accuracy": accuracy, "missed": list(missed)}
        self.said.append({"t": time.time(), "kind": "debrief", "text": f"Debrief: {round(accuracy * 100)}% correct"
                          + (f", missed {', '.join(missed)}" if missed else "")})
        self.voice.speak_debrief(accuracy, missed)

    def register_command(self, name: str, callback) -> None:
        self.voice.register_command(name, callback)


class TutorRuntime:
    """Runs the camera in a background thread and keeps the newest annotated JPEG and state for the web handlers."""

    def __init__(self, cap, feed: tutor.CameraFeed, session: tutor.TutorSession, voice: RecordingVoice,
                 loop_file: bool = False):
        self.cap, self.feed, self.session, self.voice, self.loop_file = cap, feed, session, voice, loop_file
        self.cond, self.jpeg, self.seq, self.stopped = threading.Condition(), None, 0, False
        self.camera_ok = True

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="camera").start()

    def stop(self) -> None:
        self.stopped = True
        with self.cond:
            self.cond.notify_all()

    def _loop(self) -> None:
        while not self.stopped:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.loop_file:  # a recorded video: play it again from the start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.camera_ok = False
                time.sleep(0.5)
                continue
            self.camera_ok = True
            self.feed.update(frame)
            view = self.feed.render()
            if view.shape[1] > STREAM_WIDTH:
                view = cv2.resize(view, (STREAM_WIDTH, int(view.shape[0] * STREAM_WIDTH / view.shape[1])))
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self.cond:
                    self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                    self.cond.notify_all()
            time.sleep(max(0.0, 1 / FPS - (time.time() - t0)))

    def wait_frame(self, last_seq: int, timeout: float = 2.0) -> tuple:
        """Block until a frame newer than last_seq exists; returns (jpeg or None, seq)."""
        with self.cond:
            if self.seq == last_seq and not self.stopped:
                self.cond.wait(timeout)
            return (self.jpeg, self.seq) if self.seq != last_seq else (None, last_seq)

    def snapshot(self) -> dict:
        """Everything a display needs, as plain JSON-able data."""
        feed, session = self.feed, self.session
        pos = feed.finger()
        cell = None
        if pos is not None and session.cells:
            hit = nearest_cell(session.cells, *pos)
            if hit is not None:
                cell = {"letter": letter_of(hit["dots"]), "dots": sorted(hit["dots"]), "row": hit["row"], "col": hit["col"]}
        return {"config": {"mode": session.mode, "commands": list(COMMANDS)},
                "camera": {"ok": self.camera_ok, "frames": feed.frames},
                "page": {"ok": feed.page_ok, "message": feed.message},
                "tutor": session.status(),
                "finger": {"page_mm": None if pos is None else [round(pos[0], 1), round(pos[1], 1)], "cell": cell},
                "said": list(self.voice.said)[-15:], "debrief": self.voice.debrief}

    def cells(self) -> list:
        return [{"x": round(c["x"], 2), "y": round(c["y"], 2), "w": round(c["w"], 2), "h": round(c["h"], 2),
                 "letter": letter_of(c["dots"]), "dots": sorted(c["dots"]), "row": c["row"], "col": c["col"]}
                for c in self.session.cells]

    def command(self, name: str) -> None:
        """Run a tutor command in its own thread (speaking blocks, and the request must not)."""
        threading.Thread(target=getattr(self.session, COMMANDS[name]), daemon=True).start()

    def finger_from_json(self, body: dict) -> None:
        if body.get("clear") is True:
            return self.feed.clear_finger()
        if "u" in body and "v" in body:
            u, v = float(body["u"]), float(body["v"])
            frame = self.feed.frame
            if not (0 <= u <= 1 and 0 <= v <= 1) or frame is None:
                raise ValueError("u and v must be between 0 and 1, and the camera must be running")
            return self.feed.set_finger_px(u * frame.shape[1], v * frame.shape[0])
        if "x_mm" in body and "y_mm" in body:
            x, y = float(body["x_mm"]), float(body["y_mm"])
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000):
                raise ValueError("x_mm / y_mm out of range")
            return self.feed.set_finger_mm(x, y)
        raise ValueError('send {"u":..,"v":..}, {"x_mm":..,"y_mm":..} or {"clear": true}')


def make_handler(rt: TutorRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the terminal for the tutor's own output
            pass

        # -- helpers --
        def _origin(self) -> Optional[str]:
            origin = self.headers.get("Origin")
            return origin if origin and LOCAL_ORIGIN.match(origin) else None

        def _cors(self) -> None:
            origin = self._origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Optional[dict]:
            """The JSON object in the request body, or None after sending the error response."""
            if self.headers.get("Origin") and not self._origin():
                self._json(403, {"error": "origin not allowed"})
                return None
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._json(413 if length > MAX_BODY else 400, {"error": f"body must be 0-{MAX_BODY} bytes"})
                return None
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": "body is not valid JSON"})
                return None
            if not isinstance(body, dict):
                self._json(400, {"error": "body must be a JSON object"})
                return None
            return body

        # -- routes --
        def do_OPTIONS(self):
            if self.headers.get("Origin") and not self._origin():
                return self._json(403, {"error": "origin not allowed"})
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                page = (WEB_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif path == "/api/state":
                self._json(200, rt.snapshot())
            elif path == "/api/cells":
                self._json(200, {"cells": rt.cells()})
            elif path == "/api/video":
                self._stream_video()
            elif path == "/api/events":
                self._stream_events()
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/api/command", "/api/finger"):
                return self._json(404, {"error": "not found"})
            body = self._read_json()
            if body is None:
                return
            if path == "/api/command":
                name = str(body.get("command", "")).replace("_", " ").strip().lower()
                if name not in COMMANDS:
                    return self._json(400, {"error": f"unknown command {name!r}", "valid": list(COMMANDS)})
                rt.command(name)
                return self._json(202, {"accepted": name})
            try:
                rt.finger_from_json(body)
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(200, {"ok": True})

        def _stream_video(self):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while not rt.stopped:
                    jpeg, seq = rt.wait_frame(seq)
                    if jpeg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _stream_events(self):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, idle = None, 0
            try:
                while not rt.stopped:
                    data = json.dumps(rt.snapshot())
                    if data != last or idle >= 20:  # push on change, plus a heartbeat every ~5 s
                        self.wfile.write(b"data: " + data.encode() + b"\n\n")
                        self.wfile.flush()
                        last, idle = data, 0
                    idle += 1
                    time.sleep(0.25)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def serve(rt: TutorRuntime, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create the HTTP server (not yet serving). port=0 picks a free port."""
    server = ThreadingHTTPServer((host, port), make_handler(rt))
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("letters", "read", "word-quiz"), default="letters")
    ap.add_argument("--words", nargs="+", default=[])
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="offline voice: speech is printed instead of played")
    ap.add_argument("--no-mic", action="store_true", help="don't listen on the microphone; use the HTTP commands only")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--calib")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "word-quiz" and not a.words:
        ap.error("--mode word-quiz needs --words")
    voice, wc = tutor.load_teammate_modules(a.mock)
    rv = RecordingVoice(voice)
    cells: list = []
    if a.mode == "letters":
        import make_sheet
        cells = make_sheet.sheet_cells()
    feed = tutor.CameraFeed(page_source_from_args(a), cells or None)
    session = tutor.TutorSession(rv, wc, cells, feed.finger, feed.scan, a.mode, a.questions, a.words, rng=random.Random(a.seed))
    session.attach()
    rt = TutorRuntime(open_camera(a.camera), feed, session, rv, loop_file=bool(a.camera) and Path(a.camera).is_file())
    rt.start()
    if not a.no_mic:
        voice.start_listening()
    server = serve(rt, a.host, a.port)
    print(f"Braillie tutor server: http://{a.host}:{server.server_address[1]}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()
        if not a.no_mic:
            voice.stop_listening()


if __name__ == "__main__":
    main()
