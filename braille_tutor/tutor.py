"""Braillie tutor: ties braille detection to the voice I/O module and the word-checking backend.

Modes (python tutor.py --mode ...):
  letters    voice quiz "Find the letter D" on the printed A-Z sheet; the finger's page position is matched to a cell
  read       "found it" reads the word under the finger aloud, re-detecting when the reading isn't a real word
  word-quiz  "Find the word cap": the word under the finger is checked against the target, re-detecting on a miss

Voice commands: start quiz, repeat, hint, found it, next, stop. Window keys are shortcuts for the same:
  s start quiz   r repeat   h hint   f found it   n next   x stop   q quit
The finger is not tracked yet, so click on the camera view where the fingertip is (or plug a tracker into `finger`).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

import reader
from detect import GREEN, RED, dot_distance, draw_hud, letter_of, nearest_cell, open_camera, page_source_from_args, page_status
from page import to_page

ROOT = Path(__file__).resolve().parents[1]


def load_teammate_modules(mock: bool = False):
    """Import voice_io (repo root) and braillie.word_correction (backend/) without installing anything."""
    if mock:
        os.environ["VOICE_IO_MOCK"] = "1"  # voice_io reads this when it is imported
    for p in (ROOT, ROOT / "backend"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import voice_io
    from braillie import word_correction

    return voice_io, word_correction


class TutorSession:
    """The quiz/read logic. All outside pieces are passed in, so it can be tested without a camera or a microphone.

    voice   - module with speak(text), speak_debrief(accuracy, missed) and register_command(name, callback)
    wc      - the backend word_correction module
    cells   - the page's cells with known dots (e.g. make_sheet.sheet_cells()) for the letter quiz
    finger  - callable returning the fingertip's page position in mm as (x, y), or None if it isn't visible
    scan    - callable that grabs a FRESH frame, runs detection, and returns cells (for the word modes)
    """

    def __init__(self, voice, wc, cells: list, finger: Callable, scan: Optional[Callable] = None, mode: str = "letters",
                 questions: int = 5, words: tuple = (), max_tries: int = 3, rng: Optional[random.Random] = None):
        self.voice, self.wc, self.cells, self.finger, self.scan = voice, wc, cells, finger, scan
        self.mode, self.questions, self.words, self.max_tries = mode, questions, tuple(words), max_tries
        self.rng, self.lock, self.finished = rng or random.Random(), threading.RLock(), threading.Event()
        self.state, self.items, self.index, self.tries, self.hints = "idle", [], 0, 0, 0
        self.asked, self.correct, self.slips = 0, 0, {}

    # ---- wiring ---------------------------------------------------------------------------------
    def attach(self) -> None:
        """Register the voice commands."""
        for name, handler in (("start quiz", self.on_start), ("repeat", self.on_repeat), ("hint", self.on_hint),
                              ("found it", self.on_found_it), ("next", self.on_next), ("stop", self.on_stop)):
            self.voice.register_command(name, handler)

    def say(self, text: str) -> None:
        self.voice.speak(text)

    def status(self) -> dict:
        """A snapshot for displays (no lock: commands hold it while they speak, and a display must not wait for that)."""
        asking = self.state == "asking"
        try:
            prompt = self._prompt() if asking else ""
        except IndexError:  # a command finished the quiz while we were reading
            prompt = ""
        return {"mode": self.mode, "state": self.state, "prompt": prompt, "question": self.index + 1 if asking else 0,
                "total": len(self.items), "asked": self.asked, "correct": self.correct, "tries": self.tries}

    # ---- commands -------------------------------------------------------------------------------
    def on_start(self) -> None:
        with self.lock:
            self.asked = self.correct = self.index = 0
            self.slips = {}
            if self.mode == "read":
                self.state = "reading"
                return self.say("Read mode. Put your finger on a word, then say found it.")
            self.items = self._pick_items()
            if not self.items:
                return self.say("I have nothing to quiz on yet.")
            self.state = "asking"
            self._ask()

    def on_repeat(self) -> None:
        with self.lock:
            if self.state == "asking":
                self.say(self._prompt())
            elif self.state == "reading":
                self.say("Put your finger on a word, then say found it.")
            else:
                self.say("Say start quiz to begin.")

    def on_hint(self) -> None:
        with self.lock:
            if self.state != "asking":
                return self.say("Say start quiz to begin.")
            self.hints += 1
            item = self.items[self.index]
            if self.mode == "word-quiz":
                return self.say(f"The word has {len(item)} letters and starts with {item[0]}.")
            cell = self._target_cell(item)
            dots = sorted(cell["dots"])
            if self.hints == 1:
                self.say(f"The letter {item.upper()} has {len(dots)} dot{'s' * (len(dots) != 1)}: {', '.join(map(str, dots))}.")
            else:
                self.say(f"It is in row {cell['row'] + 1}, column {cell['col'] + 1}.")

    def on_found_it(self) -> None:
        with self.lock:
            if self.state == "idle" and self.mode != "read":
                return self.say("Say start quiz to begin.")
            pos = self.finger()
            if pos is None:
                return self.say("I can't see your finger yet. Put it on the page.")
            if self.mode == "read":
                self._read_word(pos)
            elif self.mode == "word-quiz":
                self._check_word(pos)
            else:
                self._check_letter(pos)

    def on_next(self) -> None:
        with self.lock:
            if self.state != "asking":
                return self.say("Say start quiz to begin.")
            self._record(False)
            self._advance()

    def on_stop(self) -> None:
        with self.lock:
            self._finish()

    # ---- quiz internals -------------------------------------------------------------------------
    def _pick_items(self) -> list:
        if self.mode == "word-quiz":
            return list(self.words)[: self.questions]
        letters = sorted({letter_of(c["dots"]) for c in self.cells} - {None})
        return self.rng.sample(letters, min(self.questions, len(letters)))

    def _prompt(self) -> str:
        item = self.items[self.index]
        return f"Find the word {item}." if self.mode == "word-quiz" else f"Find the letter {item.upper()}."

    def _target_cell(self, letter: str) -> dict:
        return next(c for c in self.cells if letter_of(c["dots"]) == letter)

    def _ask(self) -> None:
        self.tries = self.hints = 0
        self.say(self._prompt())

    def _record(self, ok: bool, slip: bool = True) -> None:
        """Count one finished question; a miss also goes in the slips list that feeds the debrief."""
        self.asked += 1
        self.correct += ok
        if not ok and slip:
            item = self.items[self.index]
            self.slips[item] = self.slips.get(item, 0) + 1

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.items):
            self._finish()
        else:
            self._ask()

    def _finish(self) -> None:
        if self.asked:
            missed = [k.upper() for k, _ in sorted(self.slips.items(), key=lambda kv: -kv[1])]
            self.voice.speak_debrief(self.correct / self.asked, missed)
        else:
            self.say("Goodbye.")
        self.state = "done"
        self.finished.set()

    def _check_letter(self, pos) -> None:
        target = self.items[self.index]
        cell = nearest_cell(self.cells, *pos)
        if cell is None:
            return self.say("You're not on a letter yet. Keep feeling around.")
        wanted = self._target_cell(target)
        if dot_distance(cell["dots"], wanted["dots"]) == 0:
            self.say(f"Correct! That's the letter {target.upper()}.")
            self._record(True)
            return self._advance()
        self.tries += 1
        self.slips[target] = self.slips.get(target, 0) + 1
        got = letter_of(cell["dots"])
        got_text = f"the letter {got.upper()}" if got else "a mark"
        if self.tries >= self.max_tries:
            self.say(f"That was {got_text}. The letter {target.upper()} is in row {wanted['row'] + 1}, "
                     f"column {wanted['col'] + 1}. Let's move on.")
            self._record(False, slip=False)  # this wrong answer was already counted in slips just above
            return self._advance()
        near = dot_distance(cell["dots"], wanted["dots"])
        close = f" Very close, it differs by {near} dot{'s' * (near != 1)}." if near <= 2 else ""
        self.say(f"That's {got_text}, not {target.upper()}.{close} Try again.")

    def _word_at(self, pos) -> str:
        """The word under the finger, from a fresh detection ("" if there is none)."""
        if self.scan is None:
            return ""
        return reader.word_at(self.scan(), *pos) or ""

    def _read_word(self, pos) -> None:
        if self.scan is None:
            return self.say("I can't scan the page right now.")
        raw = self._word_at(pos)
        if not raw:
            return self.say("I don't see a word there.")
        result = self.wc.correct_word_read_mode(raw, redetect=lambda: self._word_at(pos))
        if result.source in ("exact", "redetect"):
            self.say(f"The word is {result.word}.")
        else:  # the backend never guesses; say what was read, letter by letter
            self.say("I couldn't read that clearly. I think it says " + ", ".join(result.raw.replace("?", "unknown")) + ".")

    def _check_word(self, pos) -> None:
        target = self.items[self.index]
        raw = self._word_at(pos)
        result = self.wc.check_word_quiz_mode(raw, target, redetect=lambda: self._word_at(pos))
        if result.correct:
            self.say(f"Correct! The word is {target}.")
            self._record(True)
            return self._advance()
        self.tries += 1
        if self.tries >= self.max_tries:
            self.say(f"That word reads {result.detected or 'nothing'}. The word was {target}. Let's move on.")
            self._record(False)
            return self._advance()
        self.say(f"I read {result.detected or 'nothing'}. Try again.")


# ---- the camera app ---------------------------------------------------------------------------------

KEYS = {"s": "on_start", "r": "on_repeat", "h": "on_hint", "f": "on_found_it", "n": "on_next", "x": "on_stop"}


class CameraFeed:
    """Newest camera frame + page registration + a 'finger'; shared by the window app and the web server.

    Until a real fingertip tracker exists, the finger is whatever was last clicked (image pixels) or set in page mm."""

    def __init__(self, page_src, overlay_cells: Optional[list] = None):
        self.page_src, self.overlay_cells = page_src, overlay_cells
        self.frame, self.H, self.frames = None, None, 0
        self.message, self.page_ok = "waiting for the camera", False
        self.finger_px: Optional[tuple] = None
        self.finger_mm: Optional[tuple] = None

    def update(self, frame: np.ndarray) -> None:
        """Take a new camera frame and re-register the page on it."""
        H, self.message, self.page_ok = page_status(frame, self.page_src)
        self.H = H if self.page_ok else None
        self.frame, self.frames = frame, self.frames + 1

    def set_finger_px(self, x: float, y: float) -> None:
        self.finger_px, self.finger_mm = (float(x), float(y)), None

    def set_finger_mm(self, x: float, y: float) -> None:
        self.finger_mm, self.finger_px = (float(x), float(y)), None

    def clear_finger(self) -> None:
        self.finger_px = self.finger_mm = None

    def finger(self) -> Optional[tuple]:
        """The fingertip's page position in mm, or None if unknown or the page isn't registered."""
        if self.finger_mm is not None:
            return self.finger_mm
        if self.finger_px is None or self.H is None:
            return None
        return to_page(self.H, *self.finger_px)

    def scan(self) -> list:
        """A FRESH detection on the newest frame (used by the word modes); [] if there is no registered page."""
        from detect import scan_page
        frame, H = self.frame, self.H
        return scan_page(frame.copy(), H) if frame is not None and H is not None else []

    def render(self) -> Optional[np.ndarray]:
        """The newest frame with the layout overlay (if any) and the finger marker drawn on it."""
        from check_sheet import draw_overlay  # imported here: it pulls in make_sheet
        frame, H = self.frame, self.H
        if frame is None:
            return None
        view = draw_overlay(frame, H, self.overlay_cells) if (H is not None and self.overlay_cells) else frame.copy()
        if self.finger_px is not None:
            cv2.circle(view, (int(self.finger_px[0]), int(self.finger_px[1])), 10, (255, 128, 0), 3)
        return view


def run_camera(session: TutorSession, cap, page_src, overlay_cells: Optional[list], title: str = "tutor") -> None:
    """Show the camera with the page status, let clicks stand in for the fingertip, and dispatch key shortcuts."""
    feed = CameraFeed(page_src, overlay_cells)
    session.finger, session.scan = feed.finger, feed.scan
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, lambda ev, x, y, *_: feed.set_finger_px(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while not session.finished.is_set():
        ok, frame = cap.read()
        if not ok:
            print("the camera stopped giving frames")
            break
        feed.update(frame)
        view = feed.render()
        state = f"{session.mode}: {session.state}" + (f"  (question {session.index + 1}/{len(session.items)})" if session.state == "asking" else "")
        draw_hud(view, [(feed.message, GREEN if feed.page_ok else RED), (state, (255, 255, 0)),
                        ("click = finger   s start  r repeat  h hint  f found it  n next  x stop  q quit", (255, 255, 0))])
        cv2.imshow(title, view)
        code = cv2.waitKey(1) & 0xFF
        key = chr(code) if code < 128 else ""
        if key == "q":
            break
        if key in KEYS:
            threading.Thread(target=getattr(session, KEYS[key]), daemon=True).start()  # speaking blocks; keep the video moving
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("letters", "read", "word-quiz"), default="letters")
    ap.add_argument("--words", nargs="+", default=[], help="target words for --mode word-quiz")
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="offline voice: prints speech, and you type the commands")
    ap.add_argument("--autostart", action="store_true", help="begin straight away instead of waiting for 'start quiz'")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--calib", help="calibration.json (for the A-Z sheet: python calibrate.py --sheet --no-track)")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "word-quiz" and not a.words:
        ap.error("--mode word-quiz needs --words")
    voice, wc = load_teammate_modules(a.mock)
    overlay = None
    cells: list = []
    if a.mode == "letters":
        import make_sheet
        cells = overlay = make_sheet.sheet_cells()
    session = TutorSession(voice, wc, cells, finger=lambda: None, mode=a.mode, questions=a.questions, words=a.words,
                           rng=random.Random(a.seed))
    session.attach()
    cap, page_src = open_camera(a.camera), page_source_from_args(a)
    voice.start_listening()
    try:
        if a.autostart:
            session.on_start()
        else:
            session.say("Welcome to Braillie. Say start quiz to begin.")
        run_camera(session, cap, page_src, overlay)
    finally:
        voice.stop_listening()


if __name__ == "__main__":
    main()
