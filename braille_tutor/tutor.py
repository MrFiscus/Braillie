"""Braillie tutor: ties braille detection to the voice I/O module and the word-checking backend.

Modes (python tutor.py --mode ...):
  letters    voice quiz "Find the letter D" on the printed A-Z sheet; the finger's page position is matched to a cell
  read       "found it" reads the word under the finger aloud, re-detecting when the reading isn't a real word
  word-quiz  "Find the word cap": the word under the finger is checked against the target, re-detecting on a miss
  explore    no quiz: rest a finger on a cell and it says what the camera sees there, with the dots ("The letter D. Dots 1, 4 and 5")

Voice commands: start quiz, repeat, hint, found it, next, next page, stop. ("next page" forgets the page that was being read, and
in explore mode works out which printed sheet is now on the desk, so the next page can be read.) Window keys are shortcuts for the same:
  s start quiz   r repeat   h hint   f found it   n next   p next page   x stop   q quit
The fingertip is tracked from the camera (fingertip.py); clicking the video overrides it.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np

import reader
from detect import AMBER, GREEN, RED, _Detector, braille_status, dot_distance, draw_cell, draw_detections, draw_hud, draw_page_outline, letter_of, locked_check_line, nearest_cell, open_camera, page_source_from_args, page_status
from llm import Coach, LLMClient
import phonelink
from fingertip import FingerTracker
from page import to_image, to_page
from sheets import Symbol, get_sheet, letter_symbol
from vote import CellLocker

ROOT = Path(__file__).resolve().parents[1]

EXPLORE_DWELL_SECONDS = 0.7  # a finger resting this long on one spot is "feeling" it
EXPLORE_STILL_MM = 4.0  # ...where "one spot" means it moved less than this
EXPLORE_LOST_SECONDS = 2.0  # finger gone this long: say so, once
EXPLORE_OFF_CELL_SECONDS = 2.5  # resting this long on blank paper: say so, once
IDENTIFY_SECONDS = 30.0  # after "next page": how long to keep trying to work out which sheet is on the desk
IDENTIFY_EVERY_SECONDS = 1.0
IDENTIFY_UNCHANGED_SECONDS = 8.0  # after "next page", if nothing at all changes in view for this long, carry on with the same sheet
SCENE_CHANGE = 6.0  # mean brightness difference (0-255) of a small copy of the picture that counts as "something moved in view"
SHEET_SPOKEN = {"alphabet": "the alphabet sheet", "words": "the words sheet", "numbers": "the numbers and signs sheet",
                "lookalikes": "the look-alikes sheet"}
EXPLORE_MARGIN_MM = 3.0  # after a cell is announced, stay quiet until the finger is this far outside it


def _and(items: list) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def load_env_files(paths: Optional[list] = None) -> list:
    """Read KEY=VALUE lines from the git-ignored .env files (repo root, then braille_tutor/) into the environment, so API keys
    live in a file that is never committed. Variables already set win. Returns the NAMES loaded; values are never printed."""
    loaded = []
    for path in paths if paths is not None else [ROOT / ".env", Path(__file__).resolve().parent / ".env"]:
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def _speech_uses_elevenlabs(voice) -> bool:
    """Does voice_io's speak() actually CALL ElevenLabs? Read from the names its code uses, not its text, so a docstring or comment
    that merely mentions ElevenLabs ("retained but not called") does not count, and it stays true if the routing changes."""
    code = getattr(getattr(voice, "speak", None), "__code__", None)
    return code is not None and "_elevenlabs_speak" in code.co_names


def voice_check(voice) -> dict:
    """Will speech actually be HEARD? {"mode": "mock"|"live", "ok": bool, "problems": [...], "warnings": [...]}.

    voice_io swallows speech errors into its log, so without this a missing key or package just means silence. `problems` make
    all speech silent; `warnings` only affect the end-of-session debrief voice (ElevenLabs)."""
    import importlib.util
    import shutil

    if getattr(voice, "MOCK_MODE", False):
        return {"mode": "mock", "ok": False, "warnings": [],
                "problems": ["mock voice: speech is printed in the terminal, not played (drop --mock and unset VOICE_IO_MOCK to hear it)"]}
    problems, warnings = [], []
    if not getattr(voice, "DEEPGRAM_API_KEY", ""):
        problems.append("DEEPGRAM_API_KEY is not set (put it in a .env file at the repo root: DEEPGRAM_API_KEY=...)")
    for module, why in (("pyaudio", "plays the speech and listens to the microphone"), ("deepgram", "the Deepgram speech client (pip install deepgram-sdk==7.9.0)")):
        if importlib.util.find_spec(module) is None:
            problems.append(f"the Python package {module!r} is not installed: {why}")
    if _speech_uses_elevenlabs(voice):  # older voice_io sent the debrief to ElevenLabs; the current one uses Deepgram for everything
        if not getattr(voice, "ELEVENLABS_API_KEY", ""):
            warnings.append("ELEVENLABS_API_KEY is not set: the end-of-session debrief will not be spoken")
        elif importlib.util.find_spec("pydub") is None or shutil.which("ffmpeg") is None:
            warnings.append("the debrief voice needs pydub and the ffmpeg program: the debrief will not be spoken")
    return {"mode": "live", "ok": not problems, "problems": problems, "warnings": warnings}


def report_voice(check: dict) -> None:
    """Print the voice check where it cannot be missed."""
    if check["ok"] and not check["warnings"]:
        return print("Voice: on (Deepgram). Speech will be played.", flush=True)
    bar = "=" * 78
    print(bar, flush=True)
    print("VOICE: " + ("speech will play, with a warning" if check["ok"] else "YOU WILL NOT HEAR ANYTHING") , flush=True)
    for line in check["problems"] + check["warnings"]:
        print("  - " + line, flush=True)
    print(bar, flush=True)


def wire_new_page(session: "TutorSession", feed: "CameraFeed") -> None:
    """Connect "next page": the session asks the feed to forget the old page; what the feed then works out is spoken and applied."""
    session.new_page = feed.new_page
    session.identifies_sheets = bool(feed.known_sheets)
    feed.on_sheet = session.set_sheet
    feed.announce = lambda text: threading.Thread(target=session.say, args=(text,), daemon=True).start()  # speaking blocks


def known_sheets_for(a, setup) -> Optional[dict]:
    """Explore mode on a printed sheet can switch between all of them when the page is turned; other modes stay on their sheet."""
    from sheets import SHEET_NAMES
    return {name: get_sheet(name) for name in SHEET_NAMES} if (a.mode == "explore" and setup.observed) else None


def load_teammate_modules(mock: bool = False):
    """Import voice_io (repo root) and braillie.word_correction (backend/) without installing anything."""
    if mock:
        os.environ["VOICE_IO_MOCK"] = "1"  # voice_io reads this when it is imported
    else:
        load_env_files()  # API keys from the git-ignored .env, before voice_io reads them at import
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
        self._speech = threading.Lock()  # one voice at a time: the explore loop, commands and the phone announcer never talk over each other
        self._heard, self._last_said, self._ex = set(), "", None
        self.voice_status: Optional[dict] = None  # set by the apps from voice_check(): whether speech will actually be heard
        self.new_page: Optional[Callable] = None  # set by the apps: forgets the old page (its reading and its registration)
        self.identifies_sheets = False  # set by the apps: "next page" also works out which printed sheet is now on the desk

    # ---- wiring ---------------------------------------------------------------------------------
    def attach(self) -> None:
        """Register the voice commands."""
        for name, handler in (("start quiz", self.on_start), ("repeat", self.on_repeat), ("hint", self.on_hint),
                              ("found it", self.on_found_it), ("next", self.on_next), ("next page", self.on_next_page),
                              ("stop", self.on_stop)):
            try:
                self.voice.register_command(name, handler)
            except ValueError:  # an older voice_io that does not know this phrase: the others still work
                print(f"voice: this voice_io does not know the command {name!r}", flush=True)

    def say(self, text: str) -> None:
        with self._speech:
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
            if self.mode == "explore":
                running = self.state == "exploring"
                self.state, self._ex = "exploring", self._new_explore_state()
                self.say("Explore mode. Rest a finger on any cell and I will tell you what it is. Say stop when you are done.")
                if not running:
                    threading.Thread(target=self._explore_loop, daemon=True, name="explore").start()
                return
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
            if self.state == "exploring":
                self.say(self._last_said or "Rest a finger on a cell and I will tell you what it is.")
            elif self.state == "asking":
                self.say(self._prompt())
            elif self.state == "reading":
                self.say("Put your finger on a word, then say found it.")
            else:
                self.say("Say start quiz to begin.")

    def on_hint(self) -> None:
        with self.lock:
            if self.state == "exploring":
                return self._explore_here(full=True)
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
            if self.state == "exploring":
                return self._explore_here(full=False)  # don't wait for the finger to settle
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
            if self.state == "exploring":
                return self.say("Just move your finger to another cell. Say stop when you are done.")
            if self.state != "asking":
                return self.say("Say start quiz to begin.")
            self._record(False)
            self._advance()

    def on_stop(self) -> None:
        with self.lock:
            self._finish()

    def on_next_page(self) -> None:
        """"Next page": forget the page that was being read, so the one now on the desk is read fresh (a locked-in reading
        otherwise holds on to the old page on purpose). Does not disturb a quiz in progress."""
        with self.lock:
            self._last_said = ""
            if self._ex is not None:
                self._ex = self._new_explore_state()  # the finger will be read again wherever it is
            if self.new_page is not None:
                self.new_page()
            self.say("Okay. Show me the next page and hold it in view." if self.identifies_sheets else "Okay. Ready for the next page.")

    def set_sheet(self, sheet) -> None:
        """Switch to another printed sheet (what the cells are, and what they mean)."""
        with self.lock:
            self.cells, self.names = sheet.cells, sheet.names
            if self._ex is not None:
                self._ex = self._new_explore_state()

    # ---- explore mode ---------------------------------------------------------------------------
    @staticmethod
    def _new_explore_state() -> dict:
        return {"anchor": None, "since": 0.0, "cell": None, "lost": None, "told_lost": False, "seen": False, "off_told": False}

    def _explore_cells(self) -> list:
        """The cells to look for the finger on: a FRESH reading of the page if there is one, else the known layout."""
        cells = self.scan() if self.scan is not None else []
        return cells or self.cells

    def _explore_symbol(self, cell: dict) -> Optional[Symbol]:
        """What the DETECTED dots mean: the sheet's name for the cell only if the camera agrees with it, else a plain letter."""
        sym = self._symbol(cell)
        if sym is not None and self.names is not None and sym.dots != frozenset(cell["dots"]):
            letter = letter_of(cell["dots"])
            return letter_symbol(letter) if letter else None
        return sym

    def describe(self, cell: dict, full: bool = True) -> str:
        """What a blind learner needs to hear for a cell: its name, the dot numbers and (`full`) where those dots sit."""
        dots = sorted(cell["dots"])
        if not dots:
            return "The camera sees no raised dots in this cell."
        sym = self._explore_symbol(cell)
        lead = (sym.spoken[0].upper() + sym.spoken[1:]) if sym else "A cell I don't recognise"
        text = f"{lead}. {'Dot' if len(dots) == 1 else 'Dots'} {_and([str(d) for d in dots])}"
        if full:
            text += ": " + _and([self._POSITIONS[d] for d in dots])
        return text + "."

    def _explore_announce(self, pos, now: float, force: bool = False, full: Optional[bool] = None) -> None:
        ex = self._ex
        cell = nearest_cell(self._explore_cells(), *pos)
        if cell is None:
            if (force or now - ex["since"] >= EXPLORE_OFF_CELL_SECONDS) and not ex["off_told"]:
                ex["off_told"] = True
                self._last_said = "I don't see any braille there."
                self.say(self._last_said)
            return
        sym = self._explore_symbol(cell)
        key = sym.key if sym else tuple(sorted(cell["dots"]))
        first = key not in self._heard  # the first time a symbol is met it is described in full; after that, briefly
        self._heard.add(key)
        ex["cell"], ex["off_told"] = cell, False
        self._last_said = self.describe(cell, full=first if full is None else full)
        self.say(self._last_said)

    def _explore_here(self, full: bool) -> None:
        pos = self.finger()
        if pos is None:
            return self.say("I can't see your finger yet. Put it on the page.")
        self._ex["since"] = time.monotonic()
        self._explore_announce(pos, time.monotonic(), force=True, full=True if full else None)

    def explore_tick(self, now: float) -> None:
        """One look at the finger (about ten a second): speak when it has settled on a cell, and not again until it moves to another."""
        ex, pos = self._ex, self.finger()
        if pos is None:
            if ex["lost"] is None:
                ex["lost"] = now
            if ex["seen"] and not ex["told_lost"] and now - ex["lost"] >= EXPLORE_LOST_SECONDS:
                ex["told_lost"] = True
                self.say("I can't see your finger.")
            ex["anchor"] = ex["cell"] = None
            return
        ex["lost"], ex["told_lost"], ex["seen"] = None, False, True
        cell = ex["cell"]
        if cell is not None:  # this cell was announced: quiet until the finger has left it
            if abs(pos[0] - cell["x"]) <= cell["w"] / 2 + EXPLORE_MARGIN_MM and abs(pos[1] - cell["y"]) <= cell["h"] / 2 + EXPLORE_MARGIN_MM:
                return
            ex["cell"] = ex["anchor"] = None
        if ex["anchor"] is None or np.hypot(pos[0] - ex["anchor"][0], pos[1] - ex["anchor"][1]) > EXPLORE_STILL_MM:
            ex["anchor"], ex["since"], ex["off_told"] = pos, now, False  # moving: start timing again from here
            return
        if now - ex["since"] >= EXPLORE_DWELL_SECONDS:
            self._explore_announce(ex["anchor"], now)

    def _explore_loop(self) -> None:
        while not self.finished.is_set():
            with self.lock:
                if self.state != "exploring":
                    return
                try:
                    self.explore_tick(time.monotonic())
                except Exception:  # keep exploring: one bad frame must not silence the tutor
                    traceback.print_exc()
            time.sleep(0.1)

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

KEYS = {"s": "on_start", "r": "on_repeat", "h": "on_hint", "f": "on_found_it", "n": "on_next", "p": "on_next_page", "x": "on_stop"}


class CameraFeed:
    """Newest camera frame + page registration + a 'finger'; shared by the window app and the web server.

    The finger is the tracked fingertip (fingertip.py) when `track_finger` is on; a click on the video or a position set in page
    mm overrides it until cleared, so a demo can always fall back to pointing with the mouse."""

    def __init__(self, page_src, overlay_cells: Optional[list] = None, labels: Optional[dict] = None, detector=None,
                 always_reading: bool = False, track_finger: bool = False, observe_sheet: Optional[list] = None,
                 show_reading: bool = True, known_sheets: Optional[dict] = None, sheet_name: str = "alphabet"):
        self.page_src, self.overlay_cells, self.labels = page_src, overlay_cells, labels
        self.detector, self.always_reading = detector, always_reading  # detector: optional live reading; needs no page
        self.frame, self.H, self.frames = None, None, 0
        self.message, self.page_ok = "waiting for the camera", False
        self.finger_px: Optional[tuple] = None
        self.finger_mm: Optional[tuple] = None
        self.tracker = FingerTracker() if track_finger else None
        self.observe_sheet = observe_sheet  # a known sheet's cells: scan() then reports the dots the camera SEES on them
        # ...and (show_reading) that reading runs continuously, is locked in cell by cell as it proves steady, and is drawn on the
        # video as a box per cell with the detected dots and letter: green = locked in, amber = still reading
        self.reader = _Detector(0.15, "auto", sheet=observe_sheet) if (observe_sheet is not None and show_reading) else None
        self.locker, self.stable, self._seen = CellLocker(), [], 0
        # "next page": which of these sheets is on the desk now? (name -> Sheet); on_sheet(Sheet) and announce(text) are set by the app
        self.known_sheets, self.sheet_name = known_sheets, sheet_name
        self.on_sheet: Optional[Callable] = None
        self.announce: Callable = lambda text: None
        self.identify_until, self._identifying, self._last_identify = 0.0, False, 0.0
        self._changed, self._thumb0, self._identify_from, self._lost_frames = False, None, 0.0, 0

    def update(self, frame: np.ndarray) -> None:
        """Take a new camera frame and re-register the page on it."""
        H, self.message, self.page_ok = page_status(frame, self.page_src)
        self.H = H if self.page_ok else None
        self.frame, self.frames = frame, self.frames + 1
        if self.tracker is not None:
            self.tracker.update(frame, self.H)
        if self.reader is not None:
            self.reader.submit(frame, self.H)
            self._absorb_reading()
        if self.identify_until:
            self._maybe_identify(frame)
        if self.detector is not None:
            self.detector.submit(frame, None)

    def new_page(self, now: Optional[float] = None) -> None:
        """Forget the page that was being read: the locked-in reading, and where the page was registered. With several known sheets
        it then works out which one is on the desk now (for up to IDENTIFY_SECONDS, while you swap them)."""
        self.locker.reset()
        self.stable = []
        if self.reader is not None:
            self._seen = self.reader.snapshot()[2]  # a scan already under way belongs to the old page
        if hasattr(self.page_src, "unlock"):
            self.page_src.unlock()
        self.H, self.page_ok = None, False
        if self.known_sheets and self.reader is not None:
            now = time.monotonic() if now is None else now
            self.identify_until, self._identify_from, self._last_identify = now + IDENTIFY_SECONDS, now, 0.0
            # the old page is usually still in view for a moment: recognising IT does not mean the page was turned. Remember what the
            # scene looked like, and only accept "same sheet again" once something in view has changed.
            self._thumb0 = self._thumb(self.frame) if self.frame is not None else None
            self._changed, self._lost_frames = False, 0

    @staticmethod
    def _thumb(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _maybe_identify(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if not self._changed:
            self._lost_frames = self._lost_frames + 1 if self.H is None else 0
            moved = self._thumb0 is not None and float(np.abs(self._thumb(frame) - self._thumb0).mean()) > SCENE_CHANGE
            self._changed = moved or self._lost_frames >= 3  # a hand or the page moved through, or the page was lost for a moment
        spoken = SHEET_SPOKEN.get(self.sheet_name, self.sheet_name)
        if not self._changed and now > self._identify_from + IDENTIFY_UNCHANGED_SECONDS:
            self.identify_until = 0.0  # nothing in view changed: it is the same page
            self.announce(f"The page didn't change, so I am still using {spoken}.")
        elif now > self.identify_until:  # gave up: stay with the sheet that was being read
            self.identify_until = 0.0
            self.announce(f"I couldn't tell which sheet this is, so I am still using {spoken}.")
        elif self.H is not None and not self._identifying and now - self._last_identify >= IDENTIFY_EVERY_SECONDS:
            self._identifying, self._last_identify = True, now
            threading.Thread(target=self._identify, args=(frame.copy(), self.H), daemon=True, name="identify").start()

    def _identify(self, frame: np.ndarray, H: np.ndarray) -> None:
        from sheetread import identify_sheet
        try:
            name, _ = identify_sheet(frame, H, {n: sh.cells for n, sh in self.known_sheets.items()})
        except Exception:  # keep trying: one bad frame is not the end of "next page"
            traceback.print_exc()
            name = None
        finally:
            self._identifying = False
        if name is not None:
            self.use_sheet(name)

    def use_sheet(self, name: str) -> None:
        """The sheet on the desk is `name`: read that one from now on (and say so)."""
        if not self.identify_until:  # answered too late: the command was cancelled or already resolved
            return
        if name == self.sheet_name and not self._changed:
            return  # that is the OLD page, still in view: keep waiting for the page to be turned
        self.identify_until = 0.0
        same = name == self.sheet_name
        if not same:
            sheet = self.known_sheets[name]
            self.sheet_name, self.observe_sheet = name, sheet.cells
            self.reader.sheet = sheet.cells
            self.locker.reset()
            self.stable = []
            if self.on_sheet is not None:
                self.on_sheet(sheet)
        spoken = SHEET_SPOKEN.get(name, name)
        self.announce(f"This looks like {spoken} again." if same else f"This is {spoken}.")

    def _absorb_reading(self) -> None:
        """Take the reader's newest scan (if there is one) into the locker: cells lock in once read right, twice in a row."""
        _, H_scan, version = self.reader.snapshot()
        if version != self._seen:
            self._seen = version
            if self.identify_until:  # between pages: what is on the desk is not the sheet these cells belong to, yet
                return
            if H_scan is not None and self.reader.observed:
                self.stable = self.locker.update_known(self.reader.observed, self.observe_sheet)

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
        if frame is None or H is None:
            return []
        if self.reader is not None and self.stable:  # what is on screen right now: the same reading the boxes show
            return list(self.stable)
        if self.observe_sheet is not None:  # our own sheet: look at exactly the places its dots would be (accurate, and fast)
            from sheetread import observe
            return observe(frame.copy(), H, self.observe_sheet)
        return scan_page(frame.copy(), H)

    def render(self, hud: bool = True) -> Optional[np.ndarray]:
        """The newest frame with the reading (boxes) or layout overlay (if any) and the finger marker drawn on it."""
        from check_sheet import draw_overlay  # imported here: it pulls in make_sheet
        frame, H = self.frame, self.H
        if frame is None:
            return None
        view = draw_overlay(frame, H, self.overlay_cells, self.labels) if (H is not None and self.overlay_cells and self.reader is None) else frame.copy()
        if self.reader is not None:
            if H is not None:
                for c in self.stable:  # one box per cell, in page mm mapped back to the picture
                    quad = [to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    draw_cell(view, quad, c, GREEN if c.get("locked") else AMBER, False, True, letters=True)
            if hud:
                lines = [(self.message, GREEN if H is not None else RED)]
                if self.identify_until:
                    lines.append(("looking for the next page...", AMBER))
                elif H is not None and self.stable:
                    lines.append(locked_check_line(self.stable, self.observe_sheet))
                elif H is not None:
                    lines.append(("reading the sheet...", AMBER))
                draw_hud(view, lines)
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
               labels: Optional[dict] = None, layout_scan: bool = False, detector=None, track_finger: bool = False,
               observe_sheet: Optional[list] = None, show_reading: bool = True, known_sheets: Optional[dict] = None,
               sheet_name: str = "alphabet") -> None:
    """Show the camera with the page status, let clicks stand in for the fingertip, and dispatch key shortcuts.

    layout_scan=True makes word modes read from the session's known cells instead of running the detector."""
    feed = CameraFeed(page_src, overlay_cells, labels, detector, track_finger=track_finger, observe_sheet=observe_sheet,
                      show_reading=show_reading, known_sheets=known_sheets, sheet_name=sheet_name)
    wire_new_page(session, feed)
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
        view = feed.render(hud=False)
        state = f"{session.mode}: {session.state}" + (f"  (question {session.index + 1}/{len(session.items)})" if session.state == "asking" else "")
        lines = [(feed.message, GREEN if feed.page_ok else RED), (state, (255, 255, 0))]
        if detector is not None:
            lines.append(braille_status(detector, len(feed.detector.boxes)))
        if feed.reader is not None:
            lines.append(locked_check_line(feed.stable, observe_sheet) if feed.stable and feed.H is not None else ("reading the sheet...", AMBER))
        lines.append(("click = finger" if not track_finger else "your finger is tracked (click to override)", (255, 255, 0)))
        lines.append(("s start  r repeat  h hint  f found it  n next  p next page  x stop  q quit", (255, 255, 0)))
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
    observed: bool = False  # explore mode on a printed sheet: speak the dots the camera sees, read with sheetread


def add_setup_args(ap: argparse.ArgumentParser) -> None:
    """The options shared by tutor.py and tutor_server.py that choose the mode and the printed sheet."""
    from sheets import SHEET_NAMES
    ap.add_argument("--mode", choices=("letters", "read", "word-quiz", "explore"), default="letters",
                    help="explore: no quiz, it says what the camera detects under your finger, with the dots (starts by itself)")
    ap.add_argument("--sheet", choices=SHEET_NAMES, help="a printed sheet (see sheets.py). Letters mode quizzes on it "
                    "(default alphabet); word modes then read its known layout instead of running the detector")
    ap.add_argument("--detect", action="store_true", help="with --sheet in a word mode: read with the camera detector anyway")
    ap.add_argument("--words", nargs="+", default=[], help="target words for --mode word-quiz (default: the words sheet's words)")
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--show-detections", action="store_true",
                    help="also run the detector and draw what it reads (boxes + letters) on the video; works without a locked page")
    ap.add_argument("--hide-detections", action="store_true",
                    help="explore mode draws a box, the detected dots and the letter for every cell on the video; this turns that off")
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
    sheet = get_sheet(a.sheet or "alphabet") if (a.mode in ("letters", "explore") or a.sheet) else None
    words = list(a.words)
    if a.mode == "word-quiz" and not words:
        if sheet is not None and sheet.spec.name == "words":
            words = [w for line in reader.read_lines(sheet.cells) for w in line.split()]
        else:
            ap.error("--mode word-quiz needs --words (or --sheet words)")
    if sheet is None:
        return Setup([], None, None, False, words, contracted=a.mode in ("read", "word-quiz"))
    if a.mode == "explore":  # speak what the camera SEES on the sheet, never the layout copied from the file
        return Setup(sheet.cells, sheet.names, {k: s.short for k, s in sheet.names.items()}, False, words, observed=True)
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
    phonelink.add_phone_args(ap)
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
    session.voice_status = voice_check(voice)
    report_voice(session.voice_status)  # silence must never be a mystery
    session.attach()
    phone = phonelink.start_phone(a, announce=lambda text: session.say(text), voice=voice)  # the video window then shows the QR code until a phone connects
    cap, page_src = (phone[1] if phone else open_camera(a.camera)), page_source_from_args(a)
    voice.start_listening()
    try:
        if phone:  # someone who cannot see the QR code on screen hears how to connect
            session.say(phone[0].spoken_instructions())
        if a.autostart or a.mode == "explore":  # explore has nothing to wait for
            session.on_start()
        else:
            session.say("Welcome to Braillie. Say start quiz to begin.")
        run_camera(session, cap, page_src, setup.cells or None, labels=setup.labels, layout_scan=setup.layout_scan,
                   detector=_Detector(0.15, "auto") if a.show_detections else None,
                   track_finger=not a.no_finger_tracking, observe_sheet=setup.cells if setup.observed else None,
                   show_reading=not a.hide_detections, known_sheets=known_sheets_for(a, setup), sheet_name=a.sheet or "alphabet")
    finally:
        voice.stop_listening()
        if phone:
            phone[2].stop()


if __name__ == "__main__":
    main()
