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
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np

import reader
from detect import GREEN, RED, _Detector, braille_status, dot_distance, draw_detections, draw_hud, draw_page_outline, letter_of, nearest_cell, open_camera, page_source_from_args, page_status
from llm import Coach, LLMClient
from fingertip import FingerTracker
from page import to_image, to_page
from sheets import Symbol, get_sheet, letter_symbol

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
    names   - what each cell means, from the printed sheet (sheets.Sheet.names); leave None for a plain A-Z layout
    finger  - callable returning the fingertip's page position in mm as (x, y), or None if it isn't visible
    scan    - callable that grabs a FRESH frame, runs detection, and returns cells (for the word modes)
    """

    def __init__(self, voice, wc, cells: list, finger: Callable, scan: Optional[Callable] = None, mode: str = "letters",
                 questions: int = 5, words: tuple = (), max_tries: int = 3, rng: Optional[random.Random] = None,
                 names: Optional[dict] = None, coach: Optional[Coach] = None, contracted: bool = False):
        self.contracted = contracted  # read words as contracted (Grade 2) braille: for a real page; printed sheets are plain letters
        self.coach, self.confusions, self.last_debrief = coach, {}, None  # coach = optional LLM extras; see llm.py
        self.names = names  # (row, col) -> sheets.Symbol for every cell of a printed sheet; None = every letter cell is a letter
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
            if self.coach:
                self.coach.prefetch(self._aid_requests())  # background: ready by the time a hint or praise is wanted
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
            spoken = self._symbol(cell).spoken
            aid = self.coach.aid(spoken) if (self.coach and self.hints == 3) else None  # third hint: a memory aid, if it arrived
            if self.hints == 1:
                self.say(f"{spoken[0].upper() + spoken[1:]} has {len(dots)} dot{'s' * (len(dots) != 1)}: {', '.join(map(str, dots))}.")
            elif aid:
                self.say(aid)
            else:
                copies = sum(1 for c in self.cells if getattr(self._symbol(c), "key", None) == item)
                self.say(f"{'One of them is' if copies > 1 else 'It is'} in row {cell['row'] + 1}, column {cell['col'] + 1}.")

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
                self._check_symbol(pos)

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
    def _symbol(self, cell: dict) -> Optional[Symbol]:
        """What a cell means: from the sheet's names if there are any, otherwise a plain letter (or None)."""
        if self.names is not None:
            return self.names.get((cell["row"], cell["col"]))
        letter = letter_of(cell["dots"])
        return letter_symbol(letter) if letter else None

    _POSITIONS = {1: "top-left", 2: "middle-left", 3: "bottom-left", 4: "top-right", 5: "middle-right", 6: "bottom-right"}

    def _aid_requests(self) -> list:
        """What the LLM helper is asked to write memory aids for: each quiz cell's name, dots and where they sit."""
        if self.mode == "word-quiz":
            return []
        out = []
        for key in self.items:
            cell = self._target_cell(key)
            dots = sorted(cell["dots"])
            out.append({"name": self._symbol(cell).spoken, "dots": dots, "positions": [self._POSITIONS[d] for d in dots]})
        return out

    def _tail(self, line: Optional[str]) -> str:
        return f" {line}" if line else ""

    def _debrief_facts(self, accuracy: float, order: list) -> dict:
        """Facts (and only facts) the LLM may use for the debrief; every number in its reply must appear here."""
        trouble = []
        for key in order[:3]:
            if self.mode == "word-quiz":
                trouble.append({"name": key})
                continue
            cell = self._target_cell(key)
            trouble.append({"name": self._symbol(cell).spoken, "dots": sorted(cell["dots"]),
                            "mistaken_for": sorted(self.confusions.get(key, ()))})
        return {"accuracy_percent": round(accuracy * 100), "questions_answered": self.asked, "correct": self.correct,
                "trouble": trouble}

    def _pick_items(self) -> list:
        if self.mode == "word-quiz":
            return list(self.words)[: self.questions]
        keys = sorted({s.key for s in map(self._symbol, self.cells) if s is not None})
        return self.rng.sample(keys, min(self.questions, len(keys)))

    def _prompt(self) -> str:
        item = self.items[self.index]
        if self.mode == "word-quiz":
            return f"Find the word {item}."
        return f"Find {self._symbol(self._target_cell(item)).spoken}."

    def _target_cell(self, key: str) -> dict:
        """The first cell on the sheet that means `key`."""
        return next(c for c in self.cells if getattr(self._symbol(c), "key", None) == key)

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
            order = [k for k, _ in sorted(self.slips.items(), key=lambda kv: -kv[1])]
            missed = [k.upper() if self.mode == "word-quiz" else self._symbol(self._target_cell(k)).short for k in order]
            accuracy = self.correct / self.asked
            self.last_debrief = {"accuracy": accuracy, "missed": missed}
            facts = self._debrief_facts(accuracy, order)
            pending = self.coach.debrief_async(facts) if self.coach else None  # starts now; the lead-in hides the wait
            if pending is not None:
                self.say("Let's see how you did.")
            text = self.coach.debrief_text(pending, facts) if pending is not None else None
            if text:
                self.last_debrief["text"] = text
                self.voice.speak(text, "debrief")
            else:
                self.voice.speak_debrief(accuracy, missed)  # the built-in debrief (also the fallback if the LLM is slow or off)
        else:
            self.say("Goodbye.")
        self.state = "done"
        self.finished.set()

    def _check_symbol(self, pos) -> None:
        target = self.items[self.index]
        cell = nearest_cell(self.cells, *pos)
        if cell is None:
            return self.say("You're not on a cell yet. Keep feeling around.")
        wanted = self._target_cell(target)
        want_text = self._symbol(wanted).spoken
        found = self._symbol(cell)
        if found is not None and found.key == target:
            self.say(f"Correct! That's {want_text}.{self._tail(self.coach.next_praise() if self.coach else None)}")
            self._record(True)
            return self._advance()
        self.tries += 1
        self.slips[target] = self.slips.get(target, 0) + 1
        got_text = found.spoken if found else "a mark"
        self.confusions.setdefault(target, set()).add(got_text)
        if self.tries >= self.max_tries:
            self.say(f"That was {got_text}. {want_text[0].upper() + want_text[1:]} is in row {wanted['row'] + 1}, "
                     f"column {wanted['col'] + 1}. Let's move on.")
            self._record(False, slip=False)  # this wrong answer was already counted in slips just above
            return self._advance()
        near = dot_distance(cell["dots"], wanted["dots"])
        close = f" Very close, it differs by {near} dot{'s' * (near != 1)}." if near <= 2 else ""
        cheer = self.coach.next_cheer() if (self.coach and self.tries == 1) else None
        self.say(f"That's {got_text}, not {want_text}.{close}{self._tail(cheer)} Try again.")

    def _word_at(self, pos) -> str:
        """The word under the finger, from a fresh detection ("" if there is none)."""
        if self.scan is None:
            return ""
        return reader.word_at(self.scan(), *pos, decode=self.contracted) or ""

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
            self.say(f"Correct! The word is {target}.{self._tail(self.coach.next_praise() if self.coach else None)}")
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

    The finger is the tracked fingertip (fingertip.py) when `track_finger` is on; a click on the video or a position set in page
    mm overrides it until cleared, so a demo can always fall back to pointing with the mouse."""

    def __init__(self, page_src, overlay_cells: Optional[list] = None, labels: Optional[dict] = None, detector=None,
                 always_reading: bool = False, track_finger: bool = False):
        self.page_src, self.overlay_cells, self.labels = page_src, overlay_cells, labels
        self.detector, self.always_reading = detector, always_reading  # detector: optional live reading; needs no page
        self.frame, self.H, self.frames = None, None, 0
        self.message, self.page_ok = "waiting for the camera", False
        self.finger_px: Optional[tuple] = None
        self.finger_mm: Optional[tuple] = None
        self.tracker = FingerTracker() if track_finger else None

    def update(self, frame: np.ndarray) -> None:
        """Take a new camera frame and re-register the page on it."""
        H, self.message, self.page_ok = page_status(frame, self.page_src)
        self.H = H if self.page_ok else None
        self.frame, self.frames = frame, self.frames + 1
        if self.tracker is not None:
            self.tracker.update(frame, self.H)
        if self.detector is not None:
            self.detector.submit(frame, None)

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
        if self.finger_px is None:
            return self.tracker.position if self.tracker is not None and self.H is not None else None
        if self.H is None:
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
        view = draw_overlay(frame, H, self.overlay_cells, self.labels) if (H is not None and self.overlay_cells) else frame.copy()
        if self.detector is not None and (H is None or self.always_reading):
            draw_detections(view, self.detector.boxes)  # what the camera reads, shown even when the page isn't registered
        if H is not None and hasattr(self.page_src, "size_mm"):  # outline the page the edge finder found
            draw_page_outline(view, H, *self.page_src.size_mm, self.page_src.origin)
        if self.finger_px is not None:
            cv2.circle(view, (int(self.finger_px[0]), int(self.finger_px[1])), 10, (255, 128, 0), 3)
        elif self.tracker is not None and self.tracker.position is not None and H is not None:
            x, y = to_image(H, *self.tracker.position)  # drawn from the smoothed page position: what the tutor is really using
            cv2.circle(view, (int(x), int(y)), 12, (0, 255, 0), 3)
            cv2.circle(view, (int(x), int(y)), 3, (0, 255, 0), -1)
        return view


def run_camera(session: TutorSession, cap, page_src, overlay_cells: Optional[list], title: str = "tutor",
               labels: Optional[dict] = None, layout_scan: bool = False, detector=None, track_finger: bool = False) -> None:
    """Show the camera with the page status, let clicks stand in for the fingertip, and dispatch key shortcuts.

    layout_scan=True makes word modes read from the session's known cells instead of running the detector."""
    feed = CameraFeed(page_src, overlay_cells, labels, detector, track_finger=track_finger)
    session.finger = feed.finger
    session.scan = (lambda: session.cells) if layout_scan else feed.scan
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
        lines = [(feed.message, GREEN if feed.page_ok else RED), (state, (255, 255, 0))]
        if detector is not None:
            lines.append(braille_status(detector, len(feed.detector.boxes)))
        lines.append(("click = finger" if not track_finger else "your finger is tracked (click to override)", (255, 255, 0)))
        lines.append(("s start  r repeat  h hint  f found it  n next  x stop  q quit", (255, 255, 0)))
        draw_hud(view, lines)
        cv2.imshow(title, view)
        code = cv2.waitKey(1) & 0xFF
        key = chr(code) if code < 128 else ""
        if key == "q":
            break
        if key in KEYS:
            threading.Thread(target=getattr(session, KEYS[key]), daemon=True).start()  # speaking blocks; keep the video moving
    cap.release()
    cv2.destroyAllWindows()


class Setup(NamedTuple):
    """What a run needs, chosen from the command line: the cells with known meaning and how word modes read."""
    cells: list
    names: Optional[dict]
    labels: Optional[dict]
    layout_scan: bool
    words: list
    contracted: bool = False  # a real page is contracted braille; a printed sheet is spelled out


def add_setup_args(ap: argparse.ArgumentParser) -> None:
    """The options shared by tutor.py and tutor_server.py that choose the mode and the printed sheet."""
    from sheets import SHEET_NAMES
    ap.add_argument("--mode", choices=("letters", "read", "word-quiz"), default="letters")
    ap.add_argument("--sheet", choices=SHEET_NAMES, help="a printed sheet (see sheets.py). Letters mode quizzes on it "
                    "(default alphabet); word modes then read its known layout instead of running the detector")
    ap.add_argument("--detect", action="store_true", help="with --sheet in a word mode: read with the camera detector anyway")
    ap.add_argument("--words", nargs="+", default=[], help="target words for --mode word-quiz (default: the words sheet's words)")
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--show-detections", action="store_true",
                    help="also run the detector and draw what it reads (boxes + letters) on the video; works without a locked page")
    ap.add_argument("--no-finger-tracking", action="store_true",
                    help="do not track the fingertip with the camera (fingertip.py); click the video to point instead")
    ap.add_argument("--llm", action="store_true", help="optional LLM extras (memory aids, praise, personal debrief); "
                    "needs OPENAI_API_KEY. Never slows the tutor: see llm.py")
    ap.add_argument("--llm-model", default=None, help="model name for the LLM helper (default gpt-4o-mini, or $BRAILLIE_LLM_MODEL)")


def make_coach(a) -> Optional[Coach]:
    """The LLM helper if --llm was given and a key is set, else None (the tutor then behaves exactly as before)."""
    if not getattr(a, "llm", False):
        return None
    client = LLMClient.from_env(a.llm_model)
    if client is None:
        print("--llm was given but OPENAI_API_KEY is not set: the LLM helper stays off.", flush=True)
        return None
    print(f"LLM helper on ({client.model}). Only lesson facts (cell names, dot numbers, scores) are sent, never camera images "
          "or audio. If the service is slow or fails, the built-in wording is used.", flush=True)
    return Coach(client)


def setup_from_args(a, ap: argparse.ArgumentParser) -> Setup:
    """Resolve --mode / --sheet / --words / --detect into cells, names and reading behaviour."""
    sheet = get_sheet(a.sheet or "alphabet") if (a.mode == "letters" or a.sheet) else None
    words = list(a.words)
    if a.mode == "word-quiz" and not words:
        if sheet is not None and sheet.spec.name == "words":
            words = [w for line in reader.read_lines(sheet.cells) for w in line.split()]
        else:
            ap.error("--mode word-quiz needs --words (or --sheet words)")
    if sheet is None:
        return Setup([], None, None, False, words, contracted=a.mode in ("read", "word-quiz"))
    layout = a.mode != "letters" and not a.detect
    if layout:
        print(f"Reading words from the known layout of the printed '{sheet.spec.name}' sheet, not from camera detection "
              "(add --detect to use the detector).", flush=True)
    return Setup(sheet.cells, sheet.names, {k: s.short for k, s in sheet.names.items()}, layout, words)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_setup_args(ap)
    ap.add_argument("--mock", action="store_true", help="offline voice: prints speech, and you type the commands")
    ap.add_argument("--autostart", action="store_true", help="begin straight away instead of waiting for 'start quiz'")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--calib", help="calibration.json (for a printed sheet: python calibrate.py --sheet --no-track)")
    ap.add_argument("--auto-page", type=float, nargs=2, metavar=("W_MM", "H_MM"))
    ap.add_argument("--markers-only", action="store_true", help="require all four markers in every frame")
    ap.add_argument("--paper", action="store_true", help="no markers: find the printed A4 sheet's own edges (see README)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    setup = setup_from_args(a, ap)
    voice, wc = load_teammate_modules(a.mock)
    session = TutorSession(voice, wc, setup.cells, finger=lambda: None, mode=a.mode, questions=a.questions, words=setup.words,
                           rng=random.Random(a.seed), names=setup.names, coach=make_coach(a), contracted=setup.contracted)
    session.attach()
    cap, page_src = open_camera(a.camera), page_source_from_args(a)
    voice.start_listening()
    try:
        if a.autostart:
            session.on_start()
        else:
            session.say("Welcome to Braillie. Say start quiz to begin.")
        run_camera(session, cap, page_src, setup.cells or None, labels=setup.labels, layout_scan=setup.layout_scan,
                   detector=_Detector(0.15, "auto") if a.show_detections else None,
                   track_finger=not a.no_finger_tracking)
    finally:
        voice.stop_listening()


if __name__ == "__main__":
    main()
