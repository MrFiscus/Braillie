"""The guided learning journey: short lessons that teach a few letters at a time, by touch, with spoken feedback that explains itself.

  lesson   teach each new letter (describe its dots, and how it relates to letters already known), the learner finds it and rests a
           finger on it; then practice them in a fresh order (plus a couple of older letters that need it); then a spoken recap
  review   adaptive practice: the letters that are shaky or recently mixed up come round more often (see progress.py)
  explore  free exploring, as in explore mode, and back

The learner speaks or stays silent; the tutor never leaves them stuck: hints arrive by themselves, get more specific, and in the end the
tutor simply shows where the letter is. A wrong answer is never just "wrong": it says what the finger IS on and how that differs from
what was asked for ("That's F. D has a dot at the middle-right that F doesn't.").

Everything the engine needs from the outside comes through a small `host` (say, tone, finger, cells, sheet), so whole lessons run in
tests with no camera, no voice and no clock. The tutor session is the real host (tutor.py).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from detect import _ALPHABET, nearest_cell
from progress import LESSON_MASTERY, Progress

POSITION = {1: "top-left", 2: "middle-left", 3: "bottom-left", 4: "top-right", 5: "middle-right", 6: "bottom-right"}
DOTS = {letter: frozenset(int(d) for d in digits) for letter, digits in _ALPHABET.items()}
NUMBER_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}

DWELL_SECONDS = 1.2  # resting a finger this long on a cell is "this is my answer"
STILL_MM = 4.0
LEAVE_MM = 9.0
HINT_AFTER = (12.0, 15.0, 20.0)  # seconds of no answer before each hint the tutor offers by itself (12 s, then 15 s more, then 20 s more)
MAX_TRIES = 3  # wrong touches before the tutor shows where the letter is and moves on
REMIND_EVERY = 25.0  # while waiting for the learner to say what to do next
LOST_AFTER = 6.0
OFF_PAGE_AFTER = 3.0
REVIEW_LENGTH = 8
EXTRA_REVIEW = 2  # older letters mixed into a lesson's practice


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    sheet: str
    letters: tuple
    intro: str
    insight: str = ""  # a fact about the braille system that makes this lesson easier


LESSONS = (
    Lesson("l1", "The first five letters", "alphabet", tuple("abcde"),
           "In this lesson you will learn the first five letters: A, B, C, D and E.",
           "They only use the top four dots, the top two rows of the cell."),
    Lesson("l2", "F to J", "alphabet", tuple("fghij"),
           "In this lesson you will learn F, G, H, I and J.",
           "These also use only the top two rows. Between them, A to J are the ten letters everything else is built from."),
    Lesson("l3", "K to O", "alphabet", tuple("klmno"),
           "In this lesson you will learn K, L, M, N and O.",
           "Good news: K to O are just A to E with one more dot added at the bottom-left, dot 3. If you know A to E, you nearly know these."),
    Lesson("l4", "P to T", "alphabet", tuple("pqrst"),
           "In this lesson you will learn P, Q, R, S and T.",
           "The same trick again: P to T are F to J with dot 3 added at the bottom-left."),
    Lesson("l5", "U to Z", "alphabet", tuple("uvwxyz"),
           "In this lesson you will learn U, V, W, X, Y and Z.",
           "U, V, X, Y and Z are A to E with the bottom two dots added, dots 3 and 6. W is the odd one out: it is J with dot 6 added."),
    Lesson("l6", "Look-alikes", "lookalikes", tuple("akblcmdneofpgqhr"),
           "This lesson is about telling look-alike letters apart. Each pair differs by exactly one dot.",
           "In every pair, the second letter is the first with dot 3 added at the bottom-left."),
)
LESSON_BY_ID = {l.id: l for l in LESSONS}

PRAISE = ("Yes, that's it.", "Exactly right.", "You found it.", "That's the one.", "Well done.", "Perfect.", "Nice, you've got it.")
NEARLY = ("Not quite.", "Close, but not that one.", "Almost.", "Good try.")
AGAIN = ("Have another feel.", "Try again.", "Feel around and try once more.")


# ---- teaching text: every sentence is built from the actual dots, so it can never disagree with the braille ---------------------

def name(letter: str) -> str:
    return f"the letter {letter.upper()}"


def cap(text: str) -> str:
    """Capitalise only the first character ("the letter D" -> "The letter D"; the built-in would make it "The letter d")."""
    return text[:1].upper() + text[1:]


def _and(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def positions(dots) -> str:
    return _and([POSITION[d] for d in sorted(dots)])


def dot_words(letter: str) -> str:
    """"One dot, at the top-left." / "Three dots: top-left, middle-left and top-right." """
    dots = sorted(DOTS[letter])
    if len(dots) == 1:
        return f"One dot, at the {POSITION[dots[0]]}."
    return f"{NUMBER_WORD[len(dots)]} dots: {positions(dots)}."


def relation(letter: str, known) -> Optional[str]:
    """How this letter grows out of one the learner already knows: "It is A with one more dot, at the middle-left." None if there is
    no such letter (a letter that shares no start with any known one)."""
    mine = DOTS[letter]
    best = None
    for k in known:
        theirs = DOTS.get(k)
        if theirs is None or k == letter or not theirs < mine:
            continue
        extra = mine - theirs
        rank = (len(extra), abs(ord(k) - ord(letter)))  # the fewest extra dots, then the nearest letter in the alphabet
        if best is None or rank < best[0]:
            best = (rank, k, extra)
    if best is None:
        return None
    _, k, extra = best
    if len(extra) == 1:
        return f"It is {name(k)} with one more dot, at the {POSITION[next(iter(extra))]}."
    return f"It is {name(k)} with {NUMBER_WORD[len(extra)].lower()} more dots: {positions(extra)}."


def contrast(touched: str, wanted: str) -> str:
    """What the finger is on, and what to feel for instead, told from the wanted letter's side so it is easy to follow by ear:
    "That's the letter E. The letter C has a dot at the top-right, and no dot at the middle-right." """
    a, b = DOTS[touched], DOTS[wanted]
    have, lack = b - a, a - b  # the wanted letter has `have` that the touched one lacks, and lacks `lack` that the touched one has
    parts = []
    if have:
        parts.append(f"a dot at the {positions(have)}" if len(have) == 1 else f"dots at the {positions(have)}")
    if lack:
        parts.append(f"no dot at the {positions(lack)}" if len(lack) == 1 else f"no dots at the {positions(lack)}")
    return f"That's {name(touched)}. {cap(name(wanted))} has {' and '.join(parts) if len(parts) == 1 else ', and '.join(parts)}."


def compare_pair(a: str, b: str) -> str:
    """A side-by-side description of two letters the learner keeps mixing up."""
    return f"Let's compare {name(a)} and {name(b)}. {cap(name(a))}: {dot_words(a)} {cap(name(b))}: {dot_words(b)}"


def welcome_first_time() -> str:
    return ("Welcome to Braillie. A braille letter is a small cell with six places for raised dots: two columns of three. "
            "The dots are numbered down the left side, one, two, three, and down the right side, four, five, six. "
            "I will describe each letter by which dots it has, and you will find it with your finger.")


# ---- resting = answering ------------------------------------------------------------------------------------------------------

class Dwell:
    """Feed it the finger position every moment; it returns the position ONCE when the finger has stayed put for `seconds`, and then
    nothing until the finger has moved well away (or vanished): resting on a cell answers once, not repeatedly."""

    def __init__(self, seconds: float = DWELL_SECONDS, still_mm: float = STILL_MM, leave_mm: float = LEAVE_MM):
        self.seconds, self.still_mm, self.leave_mm = seconds, still_mm, leave_mm
        self.reset()

    def reset(self) -> None:
        self.anchor: Optional[tuple] = None
        self.since = 0.0
        self.fired: Optional[tuple] = None

    def update(self, pos: Optional[tuple], now: float) -> Optional[tuple]:
        if pos is None:
            self.reset()
            return None
        if self.fired is not None:
            if np.hypot(pos[0] - self.fired[0], pos[1] - self.fired[1]) > self.leave_mm:
                self.fired, self.anchor, self.since = None, pos, now
            return None
        if self.anchor is None or np.hypot(pos[0] - self.anchor[0], pos[1] - self.anchor[1]) > self.still_mm:
            self.anchor, self.since = pos, now
            return None
        if now - self.since >= self.seconds:
            self.fired = self.anchor
            return self.anchor
        return None


# ---- the journey --------------------------------------------------------------------------------------------------------------

class Journey:
    """The lesson state machine. `host` supplies: say(text), tone(kind), finger() -> (x, y) mm or None, cells() -> the cells currently
    read (dicts with dots/row/col/x/y), sheet_name() -> str, request_sheet(name), explore_reset(), explore_tick(now), finish()."""

    def __init__(self, host, progress: Progress, lessons=LESSONS, coach=None, rng: Optional[random.Random] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.host, self.progress, self.lessons, self.coach = host, progress, tuple(lessons), coach
        self.rng, self.clock = rng or random.Random(), clock
        self.dwell = Dwell()
        self.phase, self.lesson = "idle", None
        self.queue: list = []
        self.target: Optional[str] = None
        self.tries = self.hints = 0
        self.asked_at = self.last_activity = self.last_lost = 0.0
        self.off_page_since = -1e9
        self.round_no, self.lesson_results, self.session_results = 1, {}, []
        self.in_a_row, self.session_started, self.taught = 0, False, []
        self.review_confusions: dict = {}
        self._await_since = 0.0
        self._pending_review: Optional[list] = None  # a practice session waiting for the right sheet to be put down

    # ---- public commands (voice) --------------------------------------------------------------
    def on_start(self) -> None:
        """"Start": begin, or continue where we were."""
        if self.phase in ("teach", "practice", "review"):
            return self.on_repeat()
        if self.phase in ("recap", "explore", "idle", "done"):
            return self._begin_or_continue()
        if self.phase == "await_sheet":
            return self._remind_sheet()

    def on_repeat(self) -> None:
        if self.phase in ("teach", "practice", "review") and self.target:
            self._ask(repeat=True)
        elif self.phase == "await_sheet":
            self._remind_sheet()
        elif self.phase == "recap":
            self._say_options()
        elif self.phase == "explore":
            self.host.say("You are exploring freely. Rest a finger on a cell and I'll tell you what it is. Say next to go back to the lessons.")
        else:
            self.host.say("Say start to begin a lesson.")

    def on_hint(self) -> None:
        if self.phase in ("teach", "practice", "review") and self.target:
            self._hint()
        else:
            self.host.say("Say start to begin a lesson.")

    def on_found_it(self) -> None:
        if self.phase in ("teach", "practice", "review") and self.target:
            pos = self.host.finger()
            if pos is None:
                return self.host.say("I can't see your finger yet. Put it on the page.")
            self._judge(pos)
        else:
            self.host.say("Say start to begin a lesson.")

    def on_next(self) -> None:
        """"Next": skip this letter, or move on from a recap, or leave free exploring."""
        if self.phase in ("teach", "practice", "review") and self.target:
            self._give_up(skipped=True)
        elif self.phase in ("recap", "explore", "idle", "done"):
            self._begin_or_continue()
        elif self.phase == "await_sheet":
            self._remind_sheet()

    def on_explore(self) -> None:
        if self.phase in ("teach", "practice", "review", "await_sheet"):
            self._abandon_round()
        self.phase, self.target = "explore", None
        self.host.explore_reset()
        self.host.tone("ready")
        self.host.say("Free exploring. Rest a finger on any cell and I will tell you what it is. Say next to go back to the lessons, or practice for a review.")

    def on_practice(self) -> None:
        """"Practice": an adaptive review of the letters that need it most."""
        if not self.progress.practised():
            return self.host.say("Let's do a lesson first, so there is something to practise. Say start.")
        if self.phase in ("teach", "practice", "review", "await_sheet"):
            self._abandon_round()
        pool = [l for l in self.progress.practised()]
        self._start_review(self.progress.pick_practice(pool, REVIEW_LENGTH, self.rng))

    def on_stop(self) -> None:
        self._save()
        self._say_session_recap()
        self.phase, self.target = "done", None
        self.host.finish()

    # ---- the clock ----------------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> None:
        """Call about ten times a second."""
        now = self.clock() if now is None else now
        if self.phase == "explore":
            return self.host.explore_tick(now)
        if self.phase == "await_sheet":
            return self._tick_sheet(now)
        if self.phase == "recap":
            if now - self.last_activity >= REMIND_EVERY:
                self.last_activity = now
                self._say_options()
            return
        if self.phase not in ("teach", "practice", "review") or not self.target:
            return
        pos = self.host.finger()
        if pos is None:
            if now - self.last_lost >= LOST_AFTER and now - self.asked_at >= LOST_AFTER:
                self.last_lost = now
                self.host.say("I can't see your finger. Put it on the page.")
            self.dwell.update(None, now)
        else:
            self.last_lost = now
            answer = self.dwell.update(pos, now)
            if answer is not None:
                return self._judge(answer)
        # the learner is not answering: help arrives by itself, a little more specific each time
        level = self.hints
        if level < len(HINT_AFTER) and now - self.last_activity >= HINT_AFTER[level]:
            self.last_activity = now
            self._hint(auto=True)

    # ---- flow ---------------------------------------------------------------------------------
    def _save(self) -> None:
        save = getattr(self.host, "save", None)
        if save is not None:
            save()

    def _begin_or_continue(self) -> None:
        if not self.session_started:
            self.session_started = True
            self.progress.start_session()
            self._save()
            self._greet()
        lesson = self.progress.next_lesson(self.lessons)
        if lesson is None:
            self.phase = "recap"
            self.last_activity = self.clock()
            self.host.say("You have finished every lesson. Say practice for a mixed review, explore to feel the letters freely, or stop.")
            return
        self._start_lesson(lesson)

    def _greet(self) -> None:
        p = self.progress
        if p.sessions <= 1 and not p.lessons:
            self.host.say(welcome_first_time())
            return
        learned = len(p.learned())
        text = "Welcome back."
        if p.streak >= 2:
            text += f" That's {p.streak} days in a row."
        if learned:
            text += f" You know {learned} letter{'s' if learned != 1 else ''} well so far."
        self.host.say(text)

    def _start_lesson(self, lesson: Lesson) -> None:
        self.lesson, self.round_no, self.lesson_results = lesson, 1, {}
        self.taught = [l for prev in self.lessons[: self.lessons.index(lesson)] for l in prev.letters if l in DOTS]
        if self.host.sheet_name() != lesson.sheet:
            self.phase, self._await_since = "await_sheet", self.clock()
            self.host.request_sheet(lesson.sheet)
            self.host.say(f"{lesson.title}. This one uses the {lesson.sheet} sheet. Put it in front of the camera and hold it steady.")
            return
        self._lesson_intro()

    def _remind_sheet(self) -> None:
        self._await_since = self.clock()
        self.host.request_sheet(self.lesson.sheet)
        self.host.say(f"I'm waiting for the {self.lesson.sheet} sheet. Put it in front of the camera and hold it steady.")

    def _tick_sheet(self, now: float) -> None:
        if self.host.sheet_name() == self.lesson.sheet and self.host.cells():
            if self._pending_review is not None:
                return self._start_review(self._pending_review)
            return self._lesson_intro()
        if now - self._await_since >= REMIND_EVERY:
            self._remind_sheet()

    def _lesson_intro(self) -> None:
        lesson = self.lesson
        self.host.tone("ready")
        self.host.say(f"{lesson.title}. {lesson.intro} {lesson.insight}")
        if self.coach is not None:
            self.coach.prefetch(self._aid_requests(lesson.letters))  # background: memory aids arrive by the time a hint wants one
        self.phase, self.queue = "teach", list(lesson.letters)
        self._next_item()

    def _next_item(self) -> None:
        if not self.queue:
            return self._phase_done()
        self.target = self.queue.pop(0)
        self.tries = self.hints = 0
        self.dwell.reset()
        self._ask()

    def _ask(self, repeat: bool = False) -> None:
        t = self.target
        if self.phase == "teach":
            rel = relation(t, self.taught) if self.lesson else None
            text = f"{cap(name(t))}. {dot_words(t)} {rel + ' ' if rel else ''}Find it and rest your finger on it."
            if repeat:
                text = f"{cap(name(t))}. {dot_words(t)} Find it and rest your finger on it."
        else:
            text = f"Find {name(t)}."
        self.host.say(text)
        self.asked_at = self.last_activity = self.clock()

    def _judge(self, pos: tuple) -> None:
        """The learner rested on `pos`: is that the target?"""
        cell = nearest_cell(self.host.cells(), *pos)
        now = self.clock()
        if cell is None:
            if now - self.off_page_since >= OFF_PAGE_AFTER:  # (once per rest: the dwell stays "answered" until the finger moves away)
                self.off_page_since = now
                self.host.say("I don't feel any braille there. Move your finger over the sheet.")
            return
        touched = letter_of_dots(cell["dots"])
        if touched == self.target:
            return self._correct()
        self.tries += 1
        self.last_activity = now
        if touched:
            self.progress.record_confusion(touched, self.target)
            self.review_confusions[(touched, self.target)] = self.review_confusions.get((touched, self.target), 0) + 1
        self.host.tone("wrong")
        if self.tries >= MAX_TRIES:
            return self._give_up(skipped=False, touched=touched)
        lead = self.rng.choice(NEARLY)
        detail = contrast(touched, self.target) if touched else "That's a cell I don't recognise."
        self.host.say(f"{lead} {detail} {self.rng.choice(AGAIN)}")

    def _mark_taught(self, letter: str) -> None:
        if self.phase == "teach" and letter not in self.taught:
            self.taught.append(letter)  # from now on other letters can be explained in terms of this one

    def _correct(self) -> None:
        t = self.target
        self._mark_taught(t)
        self.host.tone("correct")
        clean = self.hints == 0 and self.tries == 0
        self.in_a_row = self.in_a_row + 1 if clean else 0
        self._record(t, True, clean)
        praise = self.coach.next_praise() if (self.coach is not None and self.rng.random() < 0.5) else None
        text = praise or self.rng.choice(PRAISE)
        if self.phase == "teach":
            text += f" That's {name(t)}. {dot_words(t)}"
        if self.phase != "teach" and self.in_a_row in (3, 5, 8):  # (teaching is meant to be easy: a streak means something in practice)
            text += f" That's {self.in_a_row} in a row!"
        self.host.say(text)
        self._next_item()

    def _give_up(self, skipped: bool, touched: Optional[str] = None) -> None:
        """Too many wrong touches, or "next": show where it is, count it as missed, and move on."""
        t = self.target
        self._mark_taught(t)
        cell = self._cell_of(t)
        where = f" It is in row {cell['row'] + 1}, column {cell['col'] + 1}." if cell else ""
        self._record(t, False, False)
        self.in_a_row = 0
        prefix = "Okay, skipping it." if skipped else "Let me show you."
        self.host.say(f"{prefix} {cap(name(t))}: {dot_words(t)}{where} We'll come back to it.")
        self._next_item()

    def _hint(self, auto: bool = False) -> None:
        t = self.target
        self.hints += 1
        self.last_activity = self.clock()
        cell = self._cell_of(t)
        rel = relation(t, self.taught)
        if self.hints == 1 and self.phase != "teach":
            text = f"{cap(name(t))}: {dot_words(t)}"
        elif self.hints <= 2 and rel:
            aid = self.coach.aid(name(t)) if self.coach is not None else None
            text = aid or rel
        elif cell:
            text = f"It is in row {cell['row'] + 1}, column {cell['col'] + 1}. {dot_words(t)}"
        else:
            text = f"{cap(name(t))}: {dot_words(t)} Feel along each row."
        self.host.say(("Here's a hint. " if auto and self.hints == 1 else "") + text)

    # ---- bookkeeping --------------------------------------------------------------------------
    def _record(self, letter: str, correct: bool, clean: bool) -> None:
        if self.phase in ("practice", "review"):
            self.progress.record(letter, correct, hints=self.hints, tries=self.tries + 1)
            self._save()
        elif self.phase == "teach" and correct:
            self.progress.stat(letter)  # introduced: it now appears in the record, at box 0, until it is practised
        self.session_results.append((letter, correct, self.hints, self.tries))
        if self.phase == "practice":
            self.lesson_results[letter] = self.lesson_results.get(letter, False) or (correct and clean)

    def _cell_of(self, letter: str) -> Optional[dict]:
        want = DOTS[letter]
        return next((c for c in self.host.cells() if frozenset(c["dots"]) == want), None)

    def _abandon_round(self) -> None:
        self.target, self.queue, self._pending_review = None, [], None

    def _phase_done(self) -> None:
        self.target = None
        if self.phase == "teach":
            return self._start_practice()
        if self.phase == "practice":
            return self._lesson_recap()
        if self.phase == "review":
            return self._review_recap()

    def _start_practice(self) -> None:
        self.phase = "practice"
        letters = list(self.lesson.letters)
        older = [l for l in self.taught if l not in letters and self.host_has(l)]
        extra = self.progress.pick_practice(older, EXTRA_REVIEW, self.rng) if older else []
        self.rng.shuffle(letters)
        self.queue = letters + [l for l in extra if l not in letters]
        self.rng.shuffle(self.queue)
        self.host.tone("ready")
        self.host.say("Now let's practise. I'll ask for the letters in a different order." + (" A couple of older ones are mixed in to keep them fresh." if extra else ""))
        self._next_item()

    def host_has(self, letter: str) -> bool:
        return self._cell_of(letter) is not None

    def _lesson_recap(self) -> None:
        letters = self.lesson.letters
        got = [l for l in letters if self.lesson_results.get(l)]
        score = len(got) / len(letters)
        tricky = [l for l in letters if not self.lesson_results.get(l)]
        if tricky and self.round_no == 1 and score < LESSON_MASTERY:
            self.round_no += 1
            self.phase, self.queue = "practice", list(tricky)
            self.host.say(f"You found {len(got)} of {len(letters)} first time. Let's go over the tricky ones: {_and([l.upper() for l in tricky])}.")
            return self._next_item()
        entry = self.progress.lesson_done(self.lesson.id, score)
        self._save()
        mastered = entry["last"] >= LESSON_MASTERY or self.progress.lesson_mastered(self.lesson.id)
        self.host.tone("done" if mastered else "ready")
        nxt = self.progress.next_lesson(self.lessons)
        if mastered:
            text = f"Lesson complete! You found {len(got)} of {len(letters)} letters first time, with no help."
            text += f" The tricky ones were {_and([l.upper() for l in tricky])}." if tricky else " Every one."
        else:
            text = f"That's the end of the lesson. You found {len(got)} of {len(letters)} first time; {_and([l.upper() for l in tricky])} need more practice. We'll come back to them."
        confused = self.progress.top_confusions(1)
        if confused and confused[0][2] >= 2:
            text += f" You have been mixing up {name(confused[0][0])} and {name(confused[0][1])}: practice will focus on those."
        self.host.say(text)
        self.phase, self.last_activity = "recap", self.clock()
        self._say_options(after_lesson=True, next_lesson=nxt)

    def _say_options(self, after_lesson: bool = False, next_lesson=None) -> None:
        nxt = next_lesson or self.progress.next_lesson(self.lessons)
        parts = []
        if nxt is not None:
            repeat = self.lesson is not None and nxt.id == self.lesson.id  # the lesson just done was not mastered: it comes round again
            parts.append("say next to try this lesson again" if repeat else f"say next for {nxt.title}")
        parts += ["practice for a review", "explore to feel the letters freely", "or stop"]
        self.host.say("You can " + ", ".join(parts) + ".")

    # ---- review (adaptive practice) -----------------------------------------------------------
    def _start_review(self, letters: list) -> None:
        if self.host.sheet_name() != "alphabet":  # every letter of every lesson is on the alphabet sheet
            self.phase, self._await_since, self._pending_review = "await_sheet", self.clock(), letters
            self.lesson = next(l for l in self.lessons if l.sheet == "alphabet")
            self.host.request_sheet("alphabet")
            self.host.say("Practice uses the alphabet sheet. Put it in front of the camera and hold it steady.")
            return
        self._pending_review = None
        self.phase, self.review_confusions = "review", {}
        self.taught = [l for lesson in self.lessons for l in lesson.letters if l in DOTS]
        self.queue = list(letters)
        self.host.tone("ready")
        self.host.say(f"Practice time: {len(letters)} letters, chosen because they need it most.")
        self._next_item()

    def _review_recap(self) -> None:
        recent = self.session_results[-REVIEW_LENGTH:]
        clean = sum(1 for _, ok, h, t in recent if ok and h == 0 and t == 0)
        self.host.tone("done" if clean == len(recent) else "ready")
        self.host.say(f"Practice done: {clean} of {len(recent)} found first time.")
        pair = max(self.review_confusions.items(), key=lambda kv: kv[1], default=None)
        if pair and pair[1] >= 2:
            (a, b), _ = pair
            self.phase, self.queue = "review", [a, b, a, b]
            self.host.say(compare_pair(a, b) + " Now find each in turn.")
            self.review_confusions = {}
            return self._next_item()
        self.phase, self.last_activity = "recap", self.clock()
        self._say_options()

    def _say_session_recap(self) -> None:
        rs = self.session_results
        if not rs:
            self.host.say("Goodbye. Come back soon.")
            return
        clean = sum(1 for _, ok, h, t in rs if ok and h == 0 and t == 0)
        got = sum(1 for _, ok, _h, _t in rs if ok)
        p = self.progress
        text = f"That's the end of this session. You found {got} of {len(rs)} letters, {clean} of them with no help."
        if p.learned():
            text += f" You now know {len(p.learned())} letters well."
        confused = p.top_confusions(1)
        if confused:
            text += f" Worth watching: {name(confused[0][0])} and {name(confused[0][1])} get mixed up."
        self.host.tone("done")
        self.host.say(text + " Well done for practising. See you next time.")

    def _aid_requests(self, letters) -> list:
        return [{"name": name(l), "dots": sorted(DOTS[l]), "positions": [POSITION[d] for d in sorted(DOTS[l])]} for l in letters if l in DOTS]

    # ---- display ------------------------------------------------------------------------------
    def status(self) -> dict:
        lesson = self.lesson
        total = len(lesson.letters) if lesson else 0
        return {"phase": self.phase,
                "lesson": None if lesson is None else {"id": lesson.id, "title": lesson.title, "index": self.lessons.index(lesson) + 1,
                                                        "of": len(self.lessons), "sheet": lesson.sheet},
                "target": self.target, "target_dots": sorted(DOTS[self.target]) if self.target else [],
                "remaining": len(self.queue) + (1 if self.target else 0), "letters": total,
                "in_a_row": self.in_a_row, "hints": self.hints, "tries": self.tries,
                "round": self.round_no}


def letter_of_dots(dots) -> Optional[str]:
    """The plain letter a dot set spells, or None."""
    want = frozenset(dots)
    return next((l for l, d in DOTS.items() if d == want), None)
