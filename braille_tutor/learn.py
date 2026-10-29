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
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from detect import _ALPHABET, cell_at, nearest_cell
from progress import LESSON_MASTERY, Progress
from sheets import feel_for, symbol_for

POSITION = {1: "top-left", 2: "middle-left", 3: "bottom-left", 4: "top-right", 5: "middle-right", 6: "bottom-right"}
DOTS = {letter: frozenset(int(d) for d in digits) for letter, digits in _ALPHABET.items()}
# Every symbol a lesson can teach: the letters, then the digits (a number sign, then the dots of a letter a-j) and the signs
# (number sign, capital sign, punctuation). Lessons, hints and grading all go by these keys.
SYMBOL_DOTS = {**DOTS, **{ch: symbol_for(ch).dots for ch in "1234567890#^.,?!'-;"}}
NUMBER_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}

DWELL_SECONDS = 3.0  # resting a finger this long on a cell is "this is my answer" (long enough to feel the dots before it commits)
STILL_MM = 12.0  # resting drift allowed (~half a denser words-sheet cell); tip jitter used to reset the 3 s timer forever at 4 mm
LEAVE_MM = 18.0  # after an answer, the finger must move about one cell away before the next rest can count
DWELL_GAP_SECONDS = 0.5  # a one-frame tracking dropout must not restart the 3 s rest
HINT_AFTER = (20.0, 25.0, 30.0)  # seconds of searching (finger on the page, no answer) before each hint offered unasked: time to work it out
MAX_TRIES = 3  # wrong touches before the tutor shows where the letter is and moves on
SHEET_UNSURE = 15.0  # a page in view that the camera cannot name this long: they are told they can say "I'm ready"
SHEET_SETTLE = 1.5  # seconds the sheet a lesson waits for must stay in view before the lesson starts
REMIND_EVERY = 25.0  # while waiting for the learner to say what to do next
LOST_AFTER = 6.0  # the finger has been out of sight this long: say so once...
LOST_BACKOFF = (6.0, 15.0, 30.0, 60.0)  # ...then less and less often (someone with both hands on the sheet does not need reminding every few seconds)
OFF_PAGE_AFTER = 3.0
REVIEW_LENGTH = 8
EXTRA_REVIEW = 2  # older letters mixed into a lesson's practice
WARMUP_MAX = 3  # letters that went wrong before, taught again at the start of the next lesson (see _lesson_intro)


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
           "Let's learn your first five letters, A to E.",
           "They only use the top two rows of the cell, so there are just four dots to feel for."),
    Lesson("l2", "F to J", "alphabet", tuple("fghij"),
           "Today it's F to J.",
           "Like A to E, they stay in the top two rows. Learn these and you'll have the ten letters the rest of the alphabet is built on."),
    Lesson("l3", "K to O", "alphabet", tuple("klmno"),
           "Now for K to O.",
           "Here's the good news: each one is A to E with one extra dot at the bottom-left. So you already nearly know them."),
    Lesson("l4", "P to T", "alphabet", tuple("pqrst"),
           "Next up, P to T.",
           "Same trick as before: they're F to J with one extra dot at the bottom-left."),
    Lesson("l5", "U to Z", "alphabet", tuple("uvwxyz"),
           "The last few letters now, U to Z.",
           "U, V, X, Y and Z are A to E with both bottom dots added. W is the odd one out: it's J with a dot at the bottom-right."),
    Lesson("l6", "Look-alikes", "lookalikes", tuple("akblcmdneofpgqhr"),
           "This one is about letters that look alike.",
           "Each pair differs by just one dot: the second letter has an extra dot at the bottom-left."),
    Lesson("l7", "Numbers", "numbers", ("#", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"),
           "Now for numbers.",
           "Braille numbers reuse the first ten letters. The number sign, dots 3, 4, 5 and 6, says a number comes next: then A is 1, "
           "B is 2, and so on up to J, which is 0."),
    Lesson("l8", "Capitals and punctuation", "numbers", ("^", ".", ",", "?", "!", "'", "-", ";"),
           "Last of all: capitals and punctuation.",
           "These sit low in the cell. The capital sign, dot 6 on its own, goes just before a letter to make it a capital."),
)
LESSON_BY_ID = {l.id: l for l in LESSONS}

ASKING = ("teach", "warmup", "practice", "review", "word")  # the phases in which a letter is being asked for

PRAISE = ("Yes, that's it.", "Exactly right.", "You found it.", "That's the one.", "Well done.", "Perfect.", "Nice, you've got it.",
          "Spot on.", "Lovely, that's it.", "Yes! Great feeling.")
AFTER_HELP = ("You got there!", "There it is. Well worked out.", "Yes! Good persistence.", "That's it. You stuck with it.")
STREAK = {3: "Three in a row!", 5: "Five in a row, you're flying!", 8: "Eight in a row! Brilliant.", 12: "Twelve in a row. Incredible!"}
WORD_STEP = ("Yes!", "Got it.", "That's it.", "Nice.")

# A real word to spell at the end of each lesson, from letters known by then: the moment the dots become reading. One is picked
# whose every letter is on the sheet in front of the camera.
WORDS = {"l1": ("bed", "bad", "cab", "dab", "bead"), "l2": ("fig", "hide", "jab", "chief", "badge"),
         "l3": ("milk", "lemon", "book", "make", "cold"), "l4": ("star", "frog", "plant", "smile", "toast"),
         "l5": ("zebra", "wave", "fox", "quiz", "yellow"), "l6": ("hope", "bread", "pond", "frog")}

FACTS = ("Here's something fun: Louis Braille invented this system when he was just fifteen.",
         "Here's something fun: braille isn't a language. It's a way of writing, so the same dots work in English, French and many more.",
         "Here's something fun: a braille cell is about the size of a fingertip, so you can feel a whole letter at once.",
         "Here's something fun: skilled readers can read braille about as fast as many people read print.",
         "Here's something fun: you can find braille on lift buttons, medicine boxes and train tickets.",
         "Here's something fun: the six dots give sixty-three patterns, enough for letters, numbers, and punctuation too.")
NEARLY = ("Not quite.", "Not that one.", "Good try.", "Nice try.")  # (nothing that says it was close: often it is not)
AGAIN = ("Have another feel.", "Try again.", "Feel around and try once more.")
NEXT = ("Next, {l}.", "Now {l}.", "On to {l}.")  # (a new letter in a lesson, after the last one was found)
FIND = ("Can you find it?", "See if you can find it.", "Have a feel for it.")  # (asking after the first letter: the "rest" part is known by then)


# ---- teaching text: every sentence is built from the actual dots, so it can never disagree with the braille ---------------------

def sheet_words(sheet: str) -> str:
    """A sheet's name as it is said ("lookalikes" -> "look-alikes")."""
    return {"lookalikes": "look-alikes"}.get(sheet, sheet)


def name(letter: str) -> str:
    """How a symbol is said: "the letter A", "the number 1", "the number sign"."""
    return f"the letter {letter.upper()}" if letter in DOTS else symbol_for(letter).spoken


def short(key: str) -> str:
    """A symbol said on its own, as in "That's A." or "That's the number sign.": a letter, a digit, or a sign's name."""
    return key.upper() if key in DOTS else key if key.isdigit() else symbol_for(key).spoken


def cap(text: str) -> str:
    """Capitalise only the first character ("the letter D" -> "The letter D"; the built-in would make it "The letter d")."""
    return text[:1].upper() + text[1:]


def _and(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def positions(dots) -> str:
    return _and([POSITION[d] for d in sorted(dots)])


def dot_words(letter: str) -> str:
    """"One dot, at the top-left." / "Three dots: top-left, middle-left and top-right." """
    dots = sorted(SYMBOL_DOTS[letter])
    if len(dots) == 1:
        return f"One dot, at the {POSITION[dots[0]]}."
    return f"{NUMBER_WORD[len(dots)]} dots: {positions(dots)}."


def has_dots(letter: str) -> str:
    """dot_words to follow "has": "one dot, at the top-left." / "three dots: top-left, middle-left and top-right." """
    words = dot_words(letter)
    return words[:1].lower() + words[1:]


def relation(letter: str, known) -> Optional[str]:
    """How this letter grows out of one the learner already knows: "It's A with one more dot, at the middle-left." None if there is
    no such letter (a letter that shares no start with any known one)."""
    if letter.isdigit():  # (a digit is a letter's dots after a number sign: that IS how it relates)
        return f"It comes just after a number sign, and has the dots of the letter {'ABCDEFGHIJ'['1234567890'.index(letter)]}."
    if letter not in DOTS:
        return None  # (signs: their own shapes, best learned as they are)
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
        return f"It's {k.upper()} with one more dot, at the {POSITION[next(iter(extra))]}."
    return f"It's {k.upper()} with {NUMBER_WORD[len(extra)].lower()} more dots: {positions(extra)}."


def _differences(other: str, wanted: str) -> str:
    """What `wanted` has that `other` lacks, and lacks that it has: "a dot at the top-right, and no dot at the middle-right"."""
    a, b = SYMBOL_DOTS[other], SYMBOL_DOTS[wanted]
    have, lack = b - a, a - b  # the wanted letter has `have` that the other one lacks, and lacks `lack` that the other one has
    parts = []
    if have:
        parts.append(f"a dot at the {positions(have)}" if len(have) == 1 else f"dots at the {positions(have)}")
    if lack:
        parts.append(f"no dot at the {positions(lack)}" if len(lack) == 1 else f"no dots at the {positions(lack)}")
    return " and ".join(parts) if len(parts) == 1 else ", and ".join(parts)


def contrast(touched: str, wanted: str) -> str:
    """What the finger is on, and what to feel for instead, told from the wanted letter's side so it is easy to follow by ear:
    "That's the letter E. The letter C has a dot at the top-right, and no dot at the middle-right." """
    return f"That's {name(touched)}. {cap(name(wanted))} has {_differences(touched, wanted)}."


def unlike(other: str, wanted: str) -> str:
    """How `wanted` differs from a letter it gets mixed up with, before the finger is on either: "Unlike D, it has no dots at the
    top-right and middle-right." """
    return f"Unlike {short(other)}, it has {_differences(other, wanted)}."


def compare_pair(a: str, b: str) -> str:
    """A side-by-side description of two letters the learner keeps mixing up."""
    return f"Let's compare {name(a)} and {name(b)}. {cap(name(a))}: {dot_words(a)} {cap(name(b))}: {dot_words(b)}"


# ---- how much braille they know already ----------------------------------------------------------------------------------------
# Asked once, before a new learner's first lesson, so someone who reads braille already does not start from A. The lessons before
# their start are still there ("lesson one" goes back), and practice mixes in letters from them, so what they said is checked as
# they go: a letter they miss comes back in the next warm-up.
LEVELS = {"new": "l1", "some": "l3", "alphabet": "l6", "numbers": "l8"}  # what they said -> the lesson to start at
ASK_LEVEL = ("Before we start, how familiar are you with braille? Are you a complete beginner, have you tried it a little, have you "
             "learned it before, or are you pretty confident with it? Not sure? Just say beginner.")
RETRY_LEVEL = ("Sorry, I didn't catch that. Are you a complete beginner, have you tried it a little, have you learned it before, or "
               "are you pretty confident? Not sure? Just say beginner.")
# What Braillo says back, so they hear what he understood (beginners hear how braille works instead, see on_level)
LEVEL_REPLIES = {"some": "Nice, you've had a go before. We'll pick up at {where}, and I'll slip in a few of the first letters to "
                         "check them.",
                 "alphabet": "Great, you've learned it before. We'll start at {where}, and I'll check a few earlier letters along the way.",
                 "numbers": "Brilliant, you're confident with it. We'll start at {where}, and I'll check some letters and numbers along "
                            "the way."}
ASK_LEVEL_AGAIN = ("Sure. How familiar are you with braille: a complete beginner, tried it a little, learned it before, or pretty "
                   "confident?")  # ("change my level", later on)
_NOT_KNOWN = re.compile(r"\b(?:no|not|don'?t know|do not know|never learn(?:ed|t)|without)\s+(?:any\s+|the\s+)?(?:numbers?|alphabet|letters?)\b"
                        r"|\bnot (?:very |that |so |too |really )?(?:confident|good|fluent)\b")


def level_of(text: str) -> Optional[str]:
    """How familiar someone says they are with braille ("complete beginner", "tried it a little", "learned it before", "pretty
    confident"; or the older "new", "some", "alphabet", "numbers"): a key of LEVELS, or None."""
    said = " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())
    if re.search(r"\b(never|first time|complete beginner|absolute beginner|total beginner)\b", said):
        return "new"  # ("never tried it": the "tried" in it does not make it some)
    t = _NOT_KNOWN.sub(" ", said)
    if re.search(r"\bnot (?:very |that |so |too |really )?(?:confident|good|fluent)\b", said) and not re.search(r"\bnumbers?\b", t):
        return "some"  # ("I'm not very confident": they have had a go)
    if re.search(r"\b(numbers?|advanced|expert|everything|fluent|a lot|lots|confident|pretty good|really good|i read braille|"
                 r"i can read braille)\b", t):
        return "numbers"
    if re.search(r"\b(some|few|little|bit|basics?|not much|tried|dabbled|rusty|had a go|a go)\b", t):
        return "some"
    if re.search(r"\b(alphabet|a to z|(all|the|every|my) letters?|intermediate|learned|learnt|studied|know (it|braille)|before)\b", t):
        return "alphabet"
    if re.search(r"\b(new|nothing|none|no|nope|beginner|never|not sure|don'?t know|zero|beginning|scratch|first time|start(ing)? out)\b", t):
        return "new"
    return None


def _letters(keys) -> str:
    """ " letters" when they are all letters (counting "of 8 letters"), else nothing: numbers and signs are not letters."""
    return " letters" if keys and all(k in DOTS for k in keys) else ""


def welcome_first_time(lead: str = "Before we start, here's") -> str:
    return (f"{lead} how braille works. Each letter is a small cell with room for six dots, in two columns of three. "
            "I'll tell you which dots a letter has, and you find it on the sheet with your finger.")


# ---- resting = answering ------------------------------------------------------------------------------------------------------

class Dwell:
    """Feed it the finger position every moment; it returns the position ONCE when the finger has stayed put for `seconds`, and then
    nothing until the finger has moved well away (or vanished): resting on a cell answers once, not repeatedly."""

    def __init__(self, seconds: float = DWELL_SECONDS, still_mm: float = STILL_MM, leave_mm: float = LEAVE_MM,
                 gap_seconds: float = DWELL_GAP_SECONDS):
        self.seconds, self.still_mm, self.leave_mm, self.gap_seconds = seconds, still_mm, leave_mm, gap_seconds
        self.reset()

    def reset(self) -> None:
        self.anchor: Optional[tuple] = None
        self.since = 0.0
        self.fired: Optional[tuple] = None
        self._gone_since: Optional[float] = None

    def update(self, pos: Optional[tuple], now: float) -> Optional[tuple]:
        if pos is None:
            if self._gone_since is None:
                self._gone_since = now
            if now - self._gone_since > self.gap_seconds:
                self.reset()
            return None
        self._gone_since = None
        if self.fired is not None:
            if np.hypot(pos[0] - self.fired[0], pos[1] - self.fired[1]) > self.leave_mm:
                self.fired, self.anchor, self.since = None, pos, now
            return None
        if self.anchor is None or np.hypot(pos[0] - self.anchor[0], pos[1] - self.anchor[1]) > self.still_mm:
            self.anchor, self.since = pos, now
            return None
        if now - self.since >= self.seconds:
            self.fired = pos
            return pos
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
        self.hint_scale = 1.0  # "take your time": help by itself arrives later (see set_pace)
        self.phase, self.lesson = "idle", None
        self.queue: list = []
        self.target: Optional[str] = None
        self.tries = self.hints = 0
        self.last_touched: Optional[str] = None  # the wrong symbol touched last while looking for this one (for the hint)
        self._last_tick = 0.0  # (when tick last ran: see the hint clock in tick)
        self._sheet_told: Optional[str] = None  # the sheet they were last told to use (see _lesson_intro)
        self.asked_at = self.last_activity = self.last_lost = 0.0
        self.lost_notices = 0
        self.off_page_since = -1e9
        self.round_no, self.lesson_results, self.session_results = 1, {}, []
        self._mixed_now: set = set()  # the mix-ups in the current lesson (see _start_lesson)
        self.in_a_row, self.session_started, self.taught = 0, False, []
        self.review_confusions: dict = {}
        self._await_since = 0.0
        self._seen_since: Optional[float] = None  # when the sheet a lesson waits for came into view (see _tick_sheet)
        self._told_seen: Optional[str] = None  # the wrong sheet last said to be under the camera while waiting
        self._unsure_since: Optional[float] = None  # a page in view that cannot be named, since (see _tick_sheet)
        self._unsure_told = False
        self._pending_review: Optional[list] = None  # a practice session waiting for the right sheet to be put down
        self._lead = ""  # words to say at the start of the next line (praise, "okay, skipping it"), so they flow on as one line
        self.word: Optional[str] = None  # the word being spelled at the end of the lesson (see _start_word)
        self.correct_total = 0  # right answers this session (a display cheers each one)
        self._word_done = False
        self._asked_in_lesson = 0  # letters asked for so far in this part: the first one gets the whole "rest your finger" explanation

    def set_pace(self, dwell_scale: float = 1.0, hint_scale: float = 1.0) -> None:
        """A slower pace: a finger may take longer to rest on a cell to answer, and the tutor waits longer before helping by itself."""
        self.dwell.seconds = DWELL_SECONDS * dwell_scale
        self.hint_scale = hint_scale

    def _hold(self, text: str) -> None:
        """Say this at the start of the next line instead of on its own: "Yes, that's A. Next, B. ..." is one line, with no gap between."""
        self._lead = f"{self._lead} {text}".strip()

    def _speak(self, text: str) -> None:
        """Say a line, starting with anything held for it (see _hold)."""
        lead, self._lead = self._lead, ""
        self.host.say(f"{lead} {text}".strip() if lead else text)

    def _flush(self) -> None:
        """Say anything still held (when nothing followed it)."""
        if self._lead:
            self._speak("")

    def _nag(self, text: str) -> None:
        """A reminder nobody asked for: the host may hold it back if it has been said too often (a host without nag() just says it)."""
        nag = getattr(self.host, "nag", None)
        (nag or self.host.say)(text)

    # ---- public commands (voice) --------------------------------------------------------------
    def on_start(self) -> None:
        """"Start": begin, or continue where we were."""
        if self.phase == "placing":  # ("start", with no level in it: asked again, briefly)
            return self._speak(ASK_LEVEL_AGAIN)
        if self.phase in ASKING:
            return self.on_repeat()
        if self.phase in ("recap", "explore", "idle", "done"):
            return self._begin_or_continue()
        if self.phase == "await_sheet":  # ("I'm ready"): on, if the sheet is in view (or a page it cannot name); else what it waits for
            return self._sheet_ready() if self._sheet_in_view(self.lesson.sheet) or self._unnamed_page() else self._remind_sheet()

    def on_repeat(self) -> None:
        if self.phase in ASKING and self.target:
            self._ask(repeat=True)
        elif self.phase == "await_sheet":
            self._remind_sheet()
        elif self.phase == "placing":
            self._speak(ASK_LEVEL)
        elif self.phase == "recap":
            self._say_options()
        elif self.phase == "explore":
            self.host.say("You are exploring freely. Rest a finger on a cell and I'll tell you what it is. Say next to go back to the lessons.")
        else:
            self.host.say("Say start to begin a lesson.")

    def on_hint(self) -> None:
        if self.phase in ASKING and self.target:
            self._hint()
        else:
            self.host.say("Say start to begin a lesson.")

    def on_found_it(self) -> None:
        if self.phase == "await_sheet":  # ("got it": the sheet, they mean)
            return self.on_start()
        if self.phase in ASKING and self.target:
            pos = self.host.finger()
            if pos is None:
                return self.host.say("I can't see your finger yet. Put it on the page.")
            self._judge(pos)
        else:
            self.host.say("Say start to begin a lesson.")

    def on_next(self) -> None:
        """"Next": skip this letter, or move on from a recap, or leave free exploring."""
        if self.phase in ASKING and self.target:
            self._give_up(skipped=True)
        elif self.phase == "placing":  # (skipping the question: from the beginning, then)
            self.on_level("new")
        elif self.phase in ("recap", "explore", "idle", "done"):
            self._begin_or_continue()
        elif self.phase == "await_sheet":
            self._remind_sheet()

    def go_to_lesson(self, index: int) -> None:
        """"Lesson three", "next lesson": leave whatever is going on and start that lesson (counted from 0; kept in range)."""
        if not self.session_started:
            self.session_started = True
            self.progress.start_session()
            self._save()
        if self.phase in ASKING + ("await_sheet",):
            self._abandon_round()
        self._lead = ""
        self._start_lesson(self.lessons[max(0, min(index, len(self.lessons) - 1))])

    def resume_visit(self) -> None:
        """Back to the lessons later in the same visit (after reading, say): no welcome again, and not counted as a new session."""
        self.session_started = True
        self._hold("Back to the lessons.")

    def ask_level(self) -> None:
        """"Change my level": leave whatever is going on and ask again how much braille they know."""
        if self.phase in ASKING + ("await_sheet",):
            self._abandon_round()
        self._lead = ""
        self.phase, self.target, self.last_activity = "placing", None, self.clock()
        self._speak(ASK_LEVEL if self._new_learner() else ASK_LEVEL_AGAIN)

    def on_level(self, level: str) -> None:
        """Their answer to ASK_LEVEL (said, or pressed on the screen): keep it, and start the lessons from there."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        if not self.session_started:
            self.session_started = True
            self.progress.start_session()
        if self.phase in ASKING + ("await_sheet",):
            self._abandon_round()
        self.progress.level, self.progress.start_lesson = level, LEVELS[level]
        self._save()
        start = next(l for l in self.lessons if l.id == LEVELS[level])
        where = f"lesson {self.lessons.index(start) + 1}"  # (its title comes next, with the lesson itself)
        if level == "new":
            self._hold(f"Then we'll start at the very beginning. {welcome_first_time('First,')}")
        else:
            self._hold(LEVEL_REPLIES[level].format(where=where))
        self._start_lesson(start)

    def _new_learner(self) -> bool:
        """Nothing practised, no lesson done and no level given yet: someone to ask where to start."""
        p = self.progress
        return not p.level and not p.lessons and not p.letters

    def lesson_index(self) -> Optional[int]:
        """Which lesson is on now (from 0), or None between lessons."""
        return self.lessons.index(self.lesson) if self.lesson in self.lessons and self.phase != "idle" else None

    def on_explore(self) -> None:
        if self.phase in ASKING + ("await_sheet",):
            self._abandon_round()
        self.phase, self.target = "explore", None
        self.host.explore_reset()
        self.host.tone("ready")
        self.host.say("Free exploring. Rest a finger on any cell and I will tell you what it is. Say next to go back to the lessons, or practice for a review.")

    def on_practice(self) -> None:
        """"Practice": an adaptive review of the letters that need it most."""
        if not self.progress.practised():
            return self.host.say("Let's do a lesson first, so there is something to practise. Say start.")
        if self.phase in ASKING + ("await_sheet",):
            self._abandon_round()
        pool = [l for l in self.progress.practised() if l in DOTS]  # (the review is on the alphabet sheet: letters)
        self._start_review(self.progress.pick_practice(pool, REVIEW_LENGTH, self.rng))

    def brief_recap(self) -> str:
        """A few words on how it went, for leaving a lesson early ("You found 5 of 6 letters today."), or "" if nothing was asked."""
        rs = self.session_results
        if not rs:
            return ""
        got = sum(1 for _, ok, _h, _t in rs if ok)
        return f"You found {got} of {len(rs)}{_letters([r[0] for r in rs])} today."

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
                self._say_options(nag=True)
            return
        if self.phase == "placing":
            if now - self.last_activity >= REMIND_EVERY:
                self.last_activity = now
                self._nag(ASK_LEVEL)
            return
        if self.phase not in ASKING or not self.target:
            return
        pos = self.host.finger()
        if pos is None:
            gap = LOST_BACKOFF[min(self.lost_notices, len(LOST_BACKOFF) - 1)]
            visible = getattr(self.host, "page_visible", None)
            if now - self.last_lost >= gap and now - self.asked_at >= LOST_AFTER and (visible is None or visible()):
                self.last_lost = now
                self.lost_notices += 1
                self._nag("I can't see your finger. Put it on the page.")
            self.dwell.update(None, now)
        else:
            self.last_lost, self.lost_notices = now, 0
            answer = self.dwell.update(pos, now)
            if answer is not None:
                return self._judge(answer)
        # the learner is not answering: help arrives by itself, a little more specific each time, but only while they are searching
        # (finger on the page): with the finger off the page there is nothing to help with yet
        if pos is None:
            gap = now - self._last_tick
            if 0.0 < gap < 1.0:
                self.last_activity += gap  # (the clock for hints stops while the finger is off the page)
            self._last_tick = now
            return
        self._last_tick = now
        level = self.hints
        if level < len(HINT_AFTER) and now - self.last_activity >= HINT_AFTER[level] * self.hint_scale:
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
            if self._new_learner():
                self.phase, self.target, self.last_activity = "placing", None, self.clock()
                return self._speak(ASK_LEVEL)
            self._greet()
        lesson = self.progress.next_lesson(self.lessons)
        if lesson is None:
            self.phase = "recap"
            self.last_activity = self.clock()
            skipped = [l for l in self.lessons if l.id == self.progress.start_lesson]
            if skipped and self.lessons.index(skipped[0]) > 0:  # (they started further on: the first ones are still there)
                return self._speak("You have finished every lesson from where you started. Say lesson one to go over the earlier ones, "
                                   "practice for a mixed review, or explore to feel the letters freely.")
            self._speak("You have finished every lesson. Say practice for a mixed review, explore to feel the letters freely, or stop.")
            return
        self._start_lesson(lesson)

    def _greet(self) -> None:
        p = self.progress
        if p.sessions <= 1 and not p.lessons:
            self._hold(welcome_first_time())  # (it flows on into the lesson, as one line)
            return
        learned = len(p.learned())
        text = "Welcome back."
        if p.streak >= 2:
            text += f" That's {p.streak} days in a row."
        if learned:
            text += f" You know {learned} letter{'s' if learned != 1 else ''} well so far."
        self._hold(text)

    def _start_lesson(self, lesson: Lesson) -> None:
        self.lesson, self.round_no, self.lesson_results = lesson, 1, {}
        self._mixed_now = set()  # the mix-ups in this lesson (the end of the lesson mentions only these)
        self.word, self._word_done = None, False
        self.taught = [l for prev in self.lessons[: self.lessons.index(lesson)] for l in prev.letters if l in SYMBOL_DOTS]
        if self._sheet_in_view(lesson.sheet):
            return self._lesson_intro()  # (the right sheet is under the camera already: they have it)
        # The lesson waits for its sheet: it starts by itself once the camera can see it, so they have time to find it and lay it down.
        self.phase, self._await_since, self._seen_since, self._told_seen = "await_sheet", self.clock(), None, None
        self._unsure_since, self._unsure_told = None, False
        self._sheet_told = lesson.sheet
        if self.host.sheet_name() != lesson.sheet:
            self.host.request_sheet(lesson.sheet)
        self._watch(lesson.sheet)
        self._speak(f"{lesson.title} next. Get the {sheet_words(lesson.sheet)} sheet ready, the one with {feel_for(lesson.sheet)}. "
                    "Take your time: lay it flat under the camera, and I'll start as soon as I can see it.")

    def _sheet_in_view(self, sheet: str) -> bool:
        """Is this sheet under the camera now? When the camera can make out which sheet it is, that one (another one there is said,
        once); otherwise the one it is reading for, with the page in view."""
        checks = getattr(self.host, "checks_sheets", None)
        if checks is not None and checks():
            seen = self.host.sheet_under_camera()
            if seen is not None and seen != sheet and seen != self._told_seen and self.phase == "await_sheet":
                self._told_seen = seen
                self.sheet_seen(seen)  # "That's the words sheet. This lesson needs..."
            return seen == sheet
        visible = getattr(self.host, "page_visible", None)
        return self.host.sheet_name() == sheet and bool(self.host.cells()) and (visible is None or visible())

    def _unnamed_page(self) -> bool:
        """A page is in view, but the camera cannot tell which sheet it is (light, glare): their word is taken for it."""
        checks = getattr(self.host, "checks_sheets", None)
        visible = getattr(self.host, "page_visible", None)
        return bool(checks is not None and checks() and self.host.sheet_under_camera() is None and (visible is None or visible()))

    def _watch(self, sheet: Optional[str]) -> None:
        watch = getattr(self.host, "watch_sheet", None)
        if watch is not None:
            watch(sheet)

    def _sheet_ready(self) -> None:
        """The sheet the lesson waited for is in view: on with it."""
        self._seen_since = None
        self._watch(None)
        if self._pending_review is not None:
            return self._start_review(self._pending_review)
        self._hold("Got it, I can see the sheet.")
        self._lesson_intro()

    def _remind_sheet(self, nag: bool = False) -> None:
        self._await_since = self.clock()
        self.host.request_sheet(self.lesson.sheet)
        identifies = getattr(self.host, "identifies_sheets", None)
        if nag and identifies is not None and identifies():
            return  # (the camera is having another look: what it sees is said then, see sheet_seen, so this would only say it twice)
        text = (f"I'm waiting for the {sheet_words(self.lesson.sheet)} sheet, the one with {feel_for(self.lesson.sheet)}. Put it in "
                "front of the camera and hold it steady.")
        if nag:
            self._nag(text)
        else:
            self.host.say(text)

    def sheet_seen(self, sheet: Optional[str]) -> None:
        """The camera made out which sheet is in view (None: it could not tell) while this lesson waits for its own. The lesson says
        it, in its own words: the wrong one, and which is needed. (The right one starts the lesson by itself, see _tick_sheet.)"""
        if self.phase != "await_sheet" or self.lesson is None or sheet == self.lesson.sheet:
            return
        self._await_since = self.clock()  # (just said: the reminder need not follow straight after)
        want = f"This lesson needs the {sheet_words(self.lesson.sheet)} sheet, the one with {feel_for(self.lesson.sheet)}."
        self._speak(f"That's the {sheet_words(sheet)} sheet. {want}" if sheet else f"I can't tell which sheet that is yet. {want}")

    def _tick_sheet(self, now: float) -> None:
        if self._sheet_in_view(self.lesson.sheet):
            if self._seen_since is None:
                self._seen_since = now
            if now - self._seen_since < SHEET_SETTLE:
                return  # (seen for a moment while it is being put down: wait until it stays in view)
            return self._sheet_ready()
        self._seen_since = None
        if self._unnamed_page():
            if self._unsure_since is None:
                self._unsure_since = now
            if not self._unsure_told and now - self._unsure_since >= SHEET_UNSURE:
                self._unsure_told, self._await_since = True, now
                self._speak(f"I can see a sheet, but I can't tell which one yet. If it's the {sheet_words(self.lesson.sheet)} sheet, "
                            "say I'm ready.")
            return
        self._unsure_since = None
        if now - self._await_since >= REMIND_EVERY:
            self._remind_sheet(nag=True)

    def _lesson_intro(self, warmed: bool = False) -> None:
        """Start the lesson. First, if letters from earlier went wrong (here or in the quiz), a short warm-up teaches them again, each
        with how it differs from the letter it was mixed up with; then the new letters."""
        lesson = self.lesson
        if not warmed and self._sheet_told != lesson.sheet:  # (which sheet, said whenever it is a new one for them)
            self._sheet_told = lesson.sheet
            self._hold(f"You'll need the {sheet_words(lesson.sheet)} sheet for this, the one with {feel_for(lesson.sheet)}.")
        if not warmed:
            weak = [l for l in self.progress.needs_reteaching([l for l in self.taught if l not in lesson.letters], WARMUP_MAX)
                    if self.host_has(l)]
            if weak:
                self.host.tone("ready")
                self._hold(f"{lesson.intro} First, a quick warm-up with "
                           f"{'one' if len(weak) == 1 else 'a few'} that tripped you up last time: {_and([short(l) for l in weak])}.")
                self.phase, self.queue, self._asked_in_lesson = "warmup", list(weak), 0
                return self._next_item()
        self.host.tone("ready")
        self._hold(f"{lesson.insight}" if warmed else f"{lesson.intro} {lesson.insight}")
        if self.coach is not None:
            self.coach.prefetch(self._aid_requests(lesson.letters))  # background: memory aids arrive by the time a hint wants one
        self.phase, self.queue, self._asked_in_lesson = "teach", list(lesson.letters), 0
        self._next_item()

    def _next_item(self) -> None:
        if not self.queue:
            return self._phase_done()
        self.target = self.queue.pop(0)
        self.tries = self.hints = 0
        self.last_touched = None
        self.dwell.reset()
        self._ask()

    def _ask(self, repeat: bool = False) -> None:
        t = self.target
        first = self._asked_in_lesson == 0 and not repeat
        find = "Find it, and rest your finger on it." if first or repeat else self.rng.choice(FIND)
        if not repeat:
            self._asked_in_lesson += 1
        if self.phase == "word":  # spelling: the letters in turn, with the word kept in mind
            if repeat:
                text = f"We're spelling {self.word.upper()}. Find {name(t)}."
            else:
                text = f"Find {name(t)}." if first else f"Now {short(t)}."
            self._speak(text)
            self.asked_at = self.last_activity = self.clock()
            return
        if self.phase == "warmup":  # taught again: the dots, and what it was mixed up with
            mixed = self.progress.confused_with(t)
            careful = f" It's been getting mixed up with {short(mixed[0])}. {unlike(mixed[0], t)}" if mixed and not repeat else ""
            text = f"{cap(name(t))}. {dot_words(t)}{careful} {find}"
        elif self.phase == "teach":
            rel = relation(t, self.taught) if self.lesson and not repeat else None
            opener = f"Let's start with {name(t)}." if first else cap(name(t)) + "." if repeat else self.rng.choice(NEXT).format(l=name(t))
            text = f"{opener} {dot_words(t)} {rel + ' ' if rel else ''}{find}"
        else:
            text = f"Find {name(t)}."
        self._speak(text)
        self.asked_at = self.last_activity = self.clock()

    def _judge(self, pos: tuple) -> None:
        """The learner rested on `pos`: is that the target?

        Uses the sheet's own printed layout (known_cells), not a live camera reading: the finger being
        graded is resting on exactly the cell that reading would need a clear view of, which a finger
        or hand nearby routinely spoils (see vote.py's CellLocker -- a run of misreads there can even
        unlock a cell that was read correctly moments earlier, before the hand arrived)."""
        cell = cell_at(self.host.known_cells(), *pos)
        now = self.clock()
        if cell is None:
            # Last resort: any known letter within a generous reach of the tip (warped page / lag).
            cells = self.host.known_cells()
            if cells:
                cell = nearest_cell(cells, *pos, max_dist=30.0)
        if cell is None:
            if now - self.off_page_since >= OFF_PAGE_AFTER:
                self.off_page_since = now
                self._nag("I don't feel any braille there. Move your finger over the sheet.")
            return
        touched = self._symbol_at(cell)
        if touched is None and self.host.known_cells():
            # Prefer the sheet's letter at this box even if live dots were empty/garbled.
            printed = next((c for c in self.host.known_cells()
                            if c.get("row") == cell.get("row") and c.get("col") == cell.get("col")), cell)
            touched = letter_of_dots(printed["dots"])
        if touched == self.target:
            return self._correct()
        self.tries += 1
        self.last_activity = now
        if touched:
            self.last_touched = touched
            self.progress.record_confusion(touched, self.target)
            self._mixed_now.add((touched, self.target))
            self.review_confusions[(touched, self.target)] = self.review_confusions.get((touched, self.target), 0) + 1
        self.host.tone("wrong")
        if self.tries >= MAX_TRIES:
            return self._give_up(skipped=False, touched=touched)
        lead = self.rng.choice(NEARLY)
        detail = contrast(touched, self.target) if touched else "That's a cell I don't recognise."
        self.host.say(f"{lead} {detail} {self.rng.choice(AGAIN)}")
        self.dwell.reset()
        self._accept_if_on_target()

    def _accept_if_on_target(self) -> None:
        """After a correction, take the letter the finger is on now if it is the one we asked for."""
        pos = self.host.finger()
        if pos is None or not self.target:
            return
        cell = cell_at(self.host.known_cells(), *pos)
        if cell is not None and self._symbol_at(cell) == self.target:
            self._correct()

    def _mark_taught(self, letter: str) -> None:
        if self.phase == "teach" and letter not in self.taught:
            self.taught.append(letter)  # from now on other letters can be explained in terms of this one

    def _correct(self) -> None:
        t = self.target
        self.correct_total += 1
        self._mark_taught(t)
        self.host.tone("correct")
        clean = self.hints == 0 and self.tries == 0
        self.in_a_row = self.in_a_row + 1 if clean else 0
        self._record(t, True, clean)
        if self.phase == "word":
            text = self.rng.choice(WORD_STEP)
        elif not clean:
            text = self.rng.choice(AFTER_HELP)
        else:
            praise = self.coach.next_praise() if (self.coach is not None and self.rng.random() < 0.5) else None
            text = praise or self.rng.choice(PRAISE)
        if self.phase in ("teach", "warmup"):
            text += f" That's {short(t)}."  # (not its dots again: they were just said, and just felt)
        if self.phase in ("practice", "review") and self.in_a_row in STREAK:  # (teaching is meant to be easy: a streak means more here)
            text += f" {STREAK[self.in_a_row]}"
        self._hold(text)  # it flows straight on into the next letter, or the end of the round, as one line
        self._next_item()
        self._flush()

    def _give_up(self, skipped: bool, touched: Optional[str] = None) -> None:
        """Too many wrong touches, or "next": show where it is, count it as missed, and move on."""
        t = self.target
        self._mark_taught(t)
        cell = self._cell_of(t)
        where = f" It is in row {cell['row'] + 1}, column {cell['col'] + 1}." if cell else ""
        self._record(t, False, False)
        self.in_a_row = 0
        prefix = "Okay, skipping it." if skipped else "Let me show you."
        self._hold(f"{prefix} {cap(name(t))} has {has_dots(t)}{where} We'll come back to it.")
        self._next_item()
        self._flush()

    def _hint(self, auto: bool = False) -> None:
        t = self.target
        self.hints += 1
        self.last_activity = self.clock()
        cell = self._cell_of(t)
        rel = relation(t, self.taught)
        said_rel = self.phase == "teach"  # (introducing a letter already says how it grows from one they know: not again as a hint)
        aid = self.coach.aid(name(t)) if (self.coach is not None and self.hints <= 2) else None
        if self.hints == 1 and not (said_rel and aid):
            text = f"{cap(name(t))} has {has_dots(t)}" + (" Feel along each row, one cell at a time." if said_rel else "")
        elif self.hints <= 2 and (aid or (rel and not said_rel)):
            text = aid or rel
        elif cell:
            text = f"It is in row {cell['row'] + 1}, column {cell['col'] + 1}. {dot_words(t)}"
        else:
            text = f"{cap(name(t))} has {has_dots(t)} Feel along each row."
        worded = getattr(self.host, "hint_words", None)  # (Claude puts it for this learner, keeping every dot: see hint_facts)
        if callable(worded):
            text = worded(text, self.hint_facts()) or text
        self.host.say(("Here's a hint. " if auto and self.hints == 1 else "") + text)

    def hint_facts(self) -> dict:
        """What a hint can be put in terms of, for this learner now: what they are looking for, what they touched instead (and how
        the two differ, worked out here so it is exact), what they keep mixing it up with, and how much braille they know."""
        t = self.target
        facts = {"looking_for": name(t), "its_dots": has_dots(t), "hint_number": self.hints,
                 "level": self.progress.level or "not said", "lesson": self.lesson.title if self.lesson else ""}
        if self.last_touched and self.last_touched != t and self.last_touched in SYMBOL_DOTS:
            facts["just_touched"] = name(self.last_touched)
            facts["how_it_differs_from_what_they_touched"] = unlike(self.last_touched, t)
        mixed = [m for m in self.progress.confused_with(t) if m in SYMBOL_DOTS and m != self.last_touched][:1]
        if mixed:
            facts["often_mixed_up_with"] = name(mixed[0])
            facts["how_it_differs_from_that"] = unlike(mixed[0], t)
        return facts

    # ---- bookkeeping --------------------------------------------------------------------------
    def _record(self, letter: str, correct: bool, clean: bool) -> None:
        if self.phase in ("practice", "review", "warmup", "word"):
            self.progress.record(letter, correct, hints=self.hints, tries=self.tries + 1)
            self._save()
        elif self.phase == "teach" and correct:
            self.progress.stat(letter)  # introduced: it now appears in the record, at box 0, until it is practised
        self.session_results.append((letter, correct, self.hints, self.tries))
        if self.phase == "practice":
            self.lesson_results[letter] = self.lesson_results.get(letter, False) or (correct and clean)

    def _symbol_at(self, cell: dict) -> Optional[str]:
        """What is printed at this cell: the sheet says (on the numbers sheet a digit has the dots of a letter); else what its dots spell."""
        key = getattr(self.host, "symbol_key", None)
        found = key(cell) if callable(key) else None
        return found if found is not None else letter_of_dots(cell["dots"])

    def _cell_of(self, letter: str) -> Optional[dict]:
        """Where a symbol is on the sheet in front of the camera, or None if it is not on it."""
        if callable(getattr(self.host, "symbol_key", None)):
            return next((c for c in self.host.known_cells() if self._symbol_at(c) == letter), None)
        want = SYMBOL_DOTS[letter]
        return next((c for c in self.host.cells() if frozenset(c["dots"]) == want), None)

    def _abandon_round(self) -> None:
        self.target, self.queue, self._pending_review = None, [], None

    def _phase_done(self) -> None:
        self.target = None
        if self.phase == "warmup":
            self._hold("Good. Now, the new letters.")
            return self._lesson_intro(warmed=True)
        if self.phase == "teach":
            return self._start_practice()
        if self.phase == "practice":
            return self._lesson_recap()
        if self.phase == "word":
            self.host.tone("done")
            self._hold(f"You just spelled {self.word.upper()} in braille, by touch! That's a real word, and you read every letter of it.")
            return self._lesson_recap()
        if self.phase == "review":
            return self._review_recap()

    def _start_practice(self) -> None:
        self.phase = "practice"
        letters = list(self.lesson.letters)
        older = [l for l in self.taught if l not in letters and self.host_has(l)]
        # the letters that went wrong before come back here too, first; then a couple of others to keep them fresh
        weak = self.progress.needs_reteaching(older, WARMUP_MAX)
        rest = [l for l in older if l not in weak]
        extra = weak + (self.progress.pick_practice(rest, EXTRA_REVIEW, self.rng) if rest else [])
        self.rng.shuffle(letters)
        self.queue = letters + [l for l in extra if l not in letters]
        self.rng.shuffle(self.queue)
        self.host.tone("ready")
        mixed = (" I've mixed in the ones that went wrong before, and a couple of others to keep them fresh." if weak
                 else " I've mixed in a couple of older ones too." if extra else "")
        self._hold(f"You've met all {NUMBER_WORD.get(len(letters), str(len(letters))).lower() if len(self.lesson.letters) <= 6 else 'of them'}. "
                   "Now let's practise them, in a different order." + mixed)
        self._next_item()

    def _start_word(self) -> bool:
        """Spell a real word from the lesson's letters (and earlier ones): its letters are found one by one. False if no word fits the
        sheet in front of the camera (then the lesson just ends)."""
        fits = [w for w in WORDS.get(self.lesson.id, ()) if all(self.host_has(c) for c in w)]
        if not fits:
            return False
        self.word = self.rng.choice(fits)
        self.phase, self.queue, self._asked_in_lesson = "word", list(self.word), 0
        self.host.tone("ready")
        self._hold(f"Now for something fun: let's spell a real word, {self.word.upper()}. That's {', '.join(self.word.upper())}. "
                   "Find each letter in turn, and rest your finger on it.")
        self._next_item()
        return True

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
            self._hold(f"You found {len(got)} of {len(letters)} first time. Let's go over the tricky ones again: {_and([short(l) for l in tricky])}.")
            return self._next_item()
        if not self._word_done and score >= LESSON_MASTERY:  # it went well: one real word to spell before the wrap-up (not as
            self._word_done = True                                 # one more test for someone who found the lesson hard)
            if self._start_word():
                return
        entry = self.progress.lesson_done(self.lesson.id, score)
        self._save()
        mastered = entry["last"] >= LESSON_MASTERY or self.progress.lesson_mastered(self.lesson.id)
        self.host.tone("done" if mastered else "ready")
        nxt = self.progress.next_lesson(self.lessons)
        if mastered:
            text = f"Lesson complete! You found {len(got)} of {len(letters)}{_letters(letters)} first time, with no help."
            text += (f" The tricky {'one was' if len(tricky) == 1 else 'ones were'} {_and([short(l) for l in tricky])}." if tricky
                     else " Every one.")
        else:
            text = f"That's the end of the lesson. You found {len(got)} of {len(letters)} first time; {_and([short(l) for l in tricky])} need more practice. I'll bring them back at the start of the next lesson."
        confused = [c for c in self.progress.top_confusions(5) if (c[0], c[1]) in self._mixed_now][:1]  # (this lesson's, not old news)
        if confused and confused[0][2] >= 2:
            text += f" You have been mixing up {name(confused[0][0])} and {name(confused[0][1])}: practice will focus on those."
        known, met = len(self.progress.learned()), len(self.progress.practised())
        if known >= 26:
            text += " You know the whole alphabet!"
        elif known:
            text += f" That's {known} of the 26 letters you know well now."
        elif met:
            text += f" You've met {met} of the 26 letters so far."
        if not mastered and self.lesson.id == self.progress.start_lesson and self.lessons.index(self.lesson) > 0:
            text += " If that felt too hard, say lesson one, and we'll go back to the beginning."  # (they started further on)
        text += " " + self.rng.choice(FACTS)
        self._hold(text)
        self.phase, self.last_activity = "recap", self.clock()
        self._say_options(after_lesson=True, next_lesson=nxt)

    def _say_options(self, after_lesson: bool = False, next_lesson=None, nag: bool = False) -> None:
        nxt = next_lesson or self.progress.next_lesson(self.lessons)
        parts = []
        if nxt is not None:
            repeat = self.lesson is not None and nxt.id == self.lesson.id  # the lesson just done was not mastered: it comes round again
            parts.append("say next to try this lesson again" if repeat else f"say next for {nxt.title}")
        parts += ["practice for a review", "explore to feel the letters freely", "or stop"]
        text = "You can " + ", ".join(parts) + "."
        if nag:
            self._nag(text)
        else:
            self._speak(text)

    # ---- review (adaptive practice) -----------------------------------------------------------
    def _start_review(self, letters: list) -> None:
        if self.host.sheet_name() != "alphabet":  # every letter of every lesson is on the alphabet sheet
            self.phase, self._await_since, self._pending_review = "await_sheet", self.clock(), letters
            self.lesson = next(l for l in self.lessons if l.sheet == "alphabet")
            self.host.request_sheet("alphabet")
            self._watch("alphabet")
            self._speak(f"Practice uses the alphabet sheet, the one with {feel_for('alphabet')}. Put it in front of the camera and hold "
                        "it steady.")
            return
        self._pending_review = None
        self.phase, self.review_confusions = "review", {}
        self.taught = [l for lesson in self.lessons for l in lesson.letters if l in DOTS]  # (practice is on the alphabet sheet)
        self.queue = list(letters)
        self.host.tone("ready")
        self._hold(f"Practice time: {len(letters)} letters, the ones that need it most.")
        self._asked_in_lesson = 0
        self._next_item()

    def _review_recap(self) -> None:
        recent = self.session_results[-REVIEW_LENGTH:]
        clean = sum(1 for _, ok, h, t in recent if ok and h == 0 and t == 0)
        self.host.tone("done" if clean == len(recent) else "ready")
        self._hold(f"Practice done: {clean} of {len(recent)} found first time.")
        pair = max(self.review_confusions.items(), key=lambda kv: kv[1], default=None)
        if pair and pair[1] >= 2:
            (a, b), _ = pair
            self.phase, self.queue = "review", [a, b, a, b]
            self._hold(compare_pair(a, b) + " Now find each in turn.")
            self.review_confusions = {}
            return self._next_item()
        self.phase, self.last_activity = "recap", self.clock()
        self._say_options()

    def _say_session_recap(self) -> None:
        rs = self.session_results
        staying = bool(getattr(self.host, "staying", lambda: False)())  # (back to the menu, not leaving: no goodbye)
        if not rs:
            if not staying:
                self.host.say("Goodbye. Come back soon.")
            return
        clean = sum(1 for _, ok, h, t in rs if ok and h == 0 and t == 0)
        got = sum(1 for _, ok, _h, _t in rs if ok)
        p = self.progress
        text = f"That's the end of this session. You found {got} of {len(rs)}{_letters([r[0] for r in rs])}, {clean} of them with no help."
        if p.learned():
            text += f" You now know {len(p.learned())} letters well."
        confused = p.top_confusions(1)
        if confused:
            text += f" Worth watching: {name(confused[0][0])} and {name(confused[0][1])} get mixed up."
        self.host.tone("done")
        self.host.say(text + (" Well done for practising." if staying else " Well done for practising. See you next time."))

    def _aid_requests(self, letters) -> list:
        return [{"name": name(l), "dots": sorted(SYMBOL_DOTS[l]), "positions": [POSITION[d] for d in sorted(SYMBOL_DOTS[l])]}
                for l in letters if l in SYMBOL_DOTS]

    # ---- display ------------------------------------------------------------------------------
    def status(self) -> dict:
        lesson = self.lesson
        total = len(lesson.letters) if lesson else 0
        if self.phase == "word" and self.word:
            total = len(self.word)
        return {"phase": self.phase, "word": self.word.upper() if self.phase == "word" and self.word else None, "correct": self.correct_total,
                "lesson": None if lesson is None else {"id": lesson.id, "title": lesson.title, "index": self.lessons.index(lesson) + 1,
                                                        "of": len(self.lessons), "sheet": lesson.sheet},
                "target": self.target, "target_dots": sorted(SYMBOL_DOTS[self.target]) if self.target else [],
                "target_label": symbol_for(self.target).short if self.target else None,
                "remaining": len(self.queue) + (1 if self.target else 0), "letters": total,
                "in_a_row": self.in_a_row, "hints": self.hints, "tries": self.tries,
                "round": self.round_no}


def letter_of_dots(dots) -> Optional[str]:
    """The plain letter a dot set spells, or None."""
    want = frozenset(dots)
    return next((l for l, d in DOTS.items() if d == want), None)
