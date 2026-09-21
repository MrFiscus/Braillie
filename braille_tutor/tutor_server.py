"""Local web server for the tutor, so a browser frontend can show the camera and drive the session. Standard library only.

    python tutor_server.py --mock --mode letters          then open http://127.0.0.1:8000

  GET  /                 a small built-in console page (web/index.html)
  GET  /api/state        JSON snapshot: page status, quiz state, finger, what was said
  GET  /api/events       the same snapshot as a Server-Sent Events stream, pushed when it changes
  GET  /api/video        the camera as an MJPEG stream (use it as an <img src>)
  GET  /api/cells        the printed sheet's cells in page mm, with their names (for drawing your own overlay)
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
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2

import phonelink
import tutor
from progress import Progress
from detect import cell_at, letter_of, open_camera, page_source_from_args

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 4096
COMMANDS = {"start quiz": "on_start", "repeat": "on_repeat", "hint": "on_hint", "found it": "on_found_it",
            "next": "on_next", "next page": "on_next_page", "explore": "on_explore", "practice": "on_practice",
            "learn": "on_mode_learn", "read": "on_mode_read", "quiz": "on_mode_quiz", "menu": "on_mode_menu",
            "help": "on_help", "slower": "on_slower", "faster": "on_faster", "relaxed": "on_relaxed", "normal": "on_normal",
            "stop": "on_stop", "braillo": "on_braillo"}
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
                 loop_file: bool = False, phone: Optional[phonelink.PhoneLink] = None):
        self.cap, self.feed, self.session, self.voice, self.loop_file = cap, feed, session, voice, loop_file
        self.phone = phone  # when the camera is a phone: its link, so the page can show the QR code and tell the phone what we see
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
            try:
                self.feed.update(frame)
                if self.phone is not None:  # the phone speaks this to whoever is holding it ("page found", "hold the phone higher")
                    self.phone.status = {"page_ok": bool(self.feed.page_ok and self.phone.connected), "message": self.feed.message}
                # _maybe_greet() is cheap and self-guarding (it no-ops once greeted, or until its
                # preconditions hold) -- polling it here, rather than relying only on the one-shot
                # on_phone_ready() callback, means a transient hiccup during the phone's connect handshake
                # (a brief drop in link.connected right when that single callback fires) can no longer
                # permanently skip the "what do you want to do today" menu greeting for the whole session.
                self.session._maybe_greet()
                view = self.feed.render()
                if view.shape[1] > STREAM_WIDTH:
                    view = cv2.resize(view, (STREAM_WIDTH, int(view.shape[0] * STREAM_WIDTH / view.shape[1])))
                ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with self.cond:
                        self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                        self.cond.notify_all()
            except Exception:
                # One bad frame (e.g. a degenerate homography from an extreme camera angle) must never take the
                # whole camera thread down with it -- that would silently freeze the video forever, with nothing
                # left running to recover. Log it and carry on to the next frame.
                traceback.print_exc()
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
            grade = getattr(session, "grade_cells", None)
            cells = grade() if callable(grade) else session.cells
            hit = cell_at(cells or session.cells, *pos)
            if hit is not None:
                sym = session._symbol(hit)
                cell = {"letter": letter_of(hit["dots"]), "label": sym.short if sym else None, "name": sym.spoken if sym else None,
                        "dots": sorted(hit["dots"]), "row": hit["row"], "col": hit["col"]}
        return {"config": {"mode": session.mode, "commands": list(COMMANDS), "llm": session.coach.status if session.coach else "off",
                   "ask": session.ask.status if session.ask else "off",
                   "voice": session.voice_status, "sheet": self.feed.sheet_name},
                "reading": {"locked": sum(1 for c in feed.stable if c.get("locked")), "total": len(feed.observe_sheet or []),
                            "between_pages": bool(feed.identify_until)},
                "settings": session.settings.to_dict(),
                "hub": session.hub_status(),
                "learning": session.learning_status(),
                "progress": session.progress.summary() if session.journey is not None else None,
                "camera": {"ok": self.camera_ok, "frames": feed.frames}, "phone": self.phone_info(),
                "page": {"ok": feed.page_ok, "message": feed.message},
                "tutor": session.status(),
                "finger": {"page_mm": None if pos is None else [round(pos[0], 1), round(pos[1], 1)], "cell": cell,
                           "source": feed.finger_source()},
                "speaking": session.is_speaking(), "heard": session._voice_heard(),
                "said": list(self.voice.said)[-15:], "debrief": session.last_debrief or self.voice.debrief}

    def hub_request(self, path: str, body: dict) -> tuple:
        """/api/session, /api/mode, /api/prompt and /api/dialogue: returns (status, json). Speech and mode changes run in a thread so the answer is quick."""
        session = self.session
        if path == "/api/session":
            kind, name, profile = body.get("kind"), body.get("name"), body.get("profile")
            if kind not in ("google", "guest"):
                raise ValueError('"kind" must be "google" or "guest"')
            if not isinstance(name, str) or not name.strip():
                raise ValueError('"name" must be a non-empty string')
            if kind == "google" and not (isinstance(profile, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", profile)):
                raise ValueError('a google session needs a "profile" (the account id: letters, digits, - and _)')
            session.set_user(name, kind, profile if kind == "google" else None)
            return 200, {"ok": True, "hub": session.hub_status()}
        if path == "/api/mode":
            mode = str(body.get("mode", "")).lower()
            if mode != "menu" and mode not in tutor.HUB_MODES:
                raise ValueError(f'"mode" must be one of: menu, {", ".join(tutor.HUB_MODES)}')
            threading.Thread(target=session.set_mode, args=(mode,), daemon=True).start()
            return 202, {"accepted": mode}
        if path == "/api/dialogue":
            if not isinstance(body.get("open"), bool):
                raise ValueError('"open" must be true or false')
            session.dialogue(body["open"])
            return 200, {"open": session.dialogue_active()}
        name = str(body.get("name", ""))
        if name not in tutor.PROMPTS:
            raise ValueError(f'"name" must be one of: {", ".join(tutor.PROMPTS)}')
        threading.Thread(target=session.speak_prompt, args=(name, body.get("who")), daemon=True).start()
        return 202, {"accepted": name}

    def merge_progress(self, data) -> None:
        """Fold a saved copy of the learner's progress (e.g. the one in their account) into the live one, and keep the result."""
        if not isinstance(data, dict):
            raise ValueError('body must be {"data": <progress object>}')
        with self.session.lock:
            self.session.progress.merge(Progress.from_dict(data))
            self.session.save_progress()

    def phone_info(self) -> Optional[dict]:
        """None unless the camera is a phone; then what a setup screen needs: show `qr` (an image) until `connected`."""
        link = self.phone
        if link is None:
            return None
        return {"connected": link.connected, "url": link.url, "address": link.base, "code": link.code, "qr": "/api/phone/qr.png",
                "tunnel": link.tunnelled, "trusted": link.no_warning, "instructions": link.spoken_instructions(), "diagnosis": link.diagnose(), "sound": link.sound_ready(), "mic": link.mic_live(),
                "events": [text for _, text in list(link.events)[-8:]]}

    def cells(self) -> list:
        out = []
        for c in self.session.cells:
            sym = self.session._symbol(c)
            out.append({"x": round(c["x"], 2), "y": round(c["y"], 2), "w": round(c["w"], 2), "h": round(c["h"], 2),
                        "letter": letter_of(c["dots"]), "label": sym.short if sym else None, "name": sym.spoken if sym else None,
                        "dots": sorted(c["dots"]), "row": c["row"], "col": c["col"]})
        return out

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
            elif path == "/api/phone/qr.png":
                if rt.phone is None:
                    return self._json(404, {"error": "the camera is not a phone (start with --phone-camera)"})
                png = phonelink.qr_png(rt.phone.url)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(png)
            elif path == "/api/settings":
                self._json(200, rt.session.settings.to_dict())
            elif path == "/api/progress":
                if rt.session.journey is None:
                    return self._json(404, {"error": "progress is kept in learn mode (start with --mode learn)"})
                self._json(200, {"summary": rt.session.progress.summary(), "data": rt.session.progress.to_dict()})
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
            if path not in ("/api/command", "/api/finger", "/api/progress", "/api/session", "/api/mode", "/api/prompt", "/api/dialogue", "/api/settings"):
                return self._json(404, {"error": "not found"})
            body = self._read_json()
            if body is None:
                return
            if path == "/api/settings":  # speech speed, pace, sounds, "I didn't catch that": send only what should change
                try:
                    return self._json(200, rt.session.apply_settings(body))
                except (ValueError, TypeError) as e:
                    return self._json(400, {"error": str(e)})
            if path in ("/api/session", "/api/mode", "/api/prompt", "/api/dialogue"):
                if not rt.session.hub:
                    return self._json(404, {"error": "these belong to the menu-driven tutor (start tutor_server.py without --mode)"})
                try:
                    return self._json(*rt.hub_request(path, body))
                except (ValueError, TypeError) as e:
                    return self._json(400, {"error": str(e)})
            if path == "/api/progress":
                if rt.session.journey is None:
                    return self._json(404, {"error": "progress is kept in learn mode (start with --mode learn)"})
                try:
                    rt.merge_progress(body.get("data"))
                except (ValueError, TypeError) as e:
                    return self._json(400, {"error": str(e)})
                return self._json(200, {"summary": rt.session.progress.summary(), "data": rt.session.progress.to_dict()})
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
    tutor.add_setup_args(ap)
    ap.set_defaults(mode="menu")  # the website asks who is here and what they want to do; --mode learn, explore ... still work
    ap.add_argument("--mock", action="store_true", help="offline voice: speech is printed instead of played")
    ap.add_argument("--no-mic", action="store_true", help="don't listen on the microphone; use the HTTP commands only")
    ap.add_argument("--camera", default=None)
    phonelink.add_phone_args(ap)
    ap.add_argument("--calib")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"))
    ap.add_argument("--markers-only", action="store_true", help="require all four markers in every frame")
    ap.add_argument("--paper", action="store_true", help="no markers: find the printed A4 sheet's own edges (see README)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    setup = tutor.setup_from_args(a, ap)
    voice, wc = tutor.load_teammate_modules(a.mock)
    rv = RecordingVoice(voice)
    feed = tutor.CameraFeed(page_source_from_args(a), setup.cells or None, setup.labels,
                            detector=tutor._Detector(0.15, "auto") if a.show_detections else None,
                            track_finger=not a.no_finger_tracking, observe_sheet=setup.cells if setup.observed else None,
                            show_reading=not a.hide_detections, known_sheets=tutor.known_sheets_for(a, setup),
                            sheet_name=a.sheet or "alphabet")
    progress, progress_file = tutor.progress_for(a)
    session = tutor.TutorSession(rv, wc, setup.cells, feed.finger, feed.scan, a.mode, a.questions, setup.words,
                                 rng=random.Random(a.seed), names=setup.names, coach=tutor.make_coach(a),
                                 contracted=setup.contracted, progress=progress, progress_file=progress_file, tones=not a.no_tones,
                                 ask=tutor.make_ask_tutor(a))
    if setup.layout_scan:
        session.scan = lambda: session.cells  # word modes read the printed sheet's known layout
    tutor.wire_new_page(session, feed)
    session.start_watching()  # says so if the camera loses the sheet, or if it did not understand what was said
    session.voice_status = tutor.voice_check(voice)
    tutor.report_voice(session.voice_status)  # silence must never be a mystery
    session.attach()
    phone = phonelink.start_phone(a, announce=lambda text: session.say(text), voice=voice,
                                  on_ready=session.on_phone_ready if session.hub else None)
    tutor.install_speed(voice, session, a)  # after the phone speaker, so the phone gets the slowed speech too
    link = phone[0] if phone else None
    if link is not None and session.hub:
        session.phone_ready_fn = lambda: link.connected  # the greeting waits until the phone is linked
    rt = TutorRuntime(phone[1] if phone else open_camera(a.camera), feed, session, rv,
                      loop_file=not phone and bool(a.camera) and Path(a.camera).is_file(), phone=link)
    rt.start()
    def opening() -> None:
        if link is not None:  # someone who cannot see the QR code hears how to connect
            session.say(link.spoken_instructions())
        if a.mode in ("explore", "learn"):  # nothing to wait for: it starts by itself
            session.on_start()

    threading.Thread(target=opening, daemon=True).start()
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
        if phone:
            phone[2].stop()
        if not a.no_mic:
            voice.stop_listening()


if __name__ == "__main__":
    main()
