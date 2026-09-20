"""What the learner has learned so far: per-letter mastery, the letters they mix up, lessons finished, and how regularly they come back.

Mastery uses Leitner boxes (the classic spaced-repetition idea): every letter sits in a box from 0 (new or shaky) to 4 (solid).
Finding it first time with no help moves it up a box; needing help leaves it where it is; getting it wrong drops it back to 0.
Practice sessions then favour the low boxes and the letters that were recently mistaken for each other, so time goes where it helps.

Everything is plain data (`to_dict` / `from_dict`), so it can be kept in a local file and in the learner's Supabase account, and two
copies (this laptop and the account) can be `merge`d without counting anything twice.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

BOXES = 5  # box 0 (new or shaky) ... box 4 (solid)
MASTERED_BOX = 3  # a letter counts as learned from this box up
LESSON_MASTERY = 0.8  # share of a lesson's letters found first time, with no help, to count the lesson as mastered
DEFAULT_DIR = Path.home() / ".braillie"


@dataclass
class LetterStat:
    attempts: int = 0  # times it was asked for in practice
    correct: int = 0  # ...and found (with or without help)
    clean: int = 0  # ...and found first time with no help
    hints: int = 0  # hints it took in total
    box: int = 0
    last_seen: float = 0.0  # unix time of the last practice of it


def today_str(now: Optional[float] = None) -> str:
    return dt.date.fromtimestamp(time.time() if now is None else now).isoformat()


class Progress:
    """The learner's record. Not thread-safe by itself: the tutor session holds its lock while it records."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.letters: dict = {}  # letter -> LetterStat
        self.confusions: dict = {}  # "touched>wanted" -> how many times that letter was touched when the other was asked for
        self.lessons: dict = {}  # lesson id -> {"best": 0..1, "last": 0..1, "times": n}
        self.sessions, self.streak, self.best_streak, self.last_day = 0, 0, 0, ""

    # ---- recording ----------------------------------------------------------------------------
    def stat(self, letter: str) -> LetterStat:
        return self.letters.setdefault(letter, LetterStat())

    def record(self, letter: str, correct: bool, hints: int = 0, tries: int = 1) -> LetterStat:
        """One practice answer for `letter`. `hints` and `tries` (wrong touches before the right one) say how much help it took."""
        s = self.stat(letter)
        s.attempts += 1
        s.hints += hints
        s.last_seen = self.clock()
        if correct:
            s.correct += 1
            if hints == 0 and tries <= 1:
                s.clean += 1
                s.box = min(BOXES - 1, s.box + 1)  # no help needed: it moves up
            # help was needed: it stays where it is, so it comes round again soon
        else:
            s.box = 0
        return s

    def record_confusion(self, touched: str, wanted: str) -> None:
        if touched and wanted and touched != wanted:
            key = f"{touched}>{wanted}"
            self.confusions[key] = self.confusions.get(key, 0) + 1

    def lesson_done(self, lesson_id: str, score: float) -> dict:
        """A lesson finished with `score` (share of its letters found first time, no help)."""
        entry = self.lessons.setdefault(lesson_id, {"best": 0.0, "last": 0.0, "times": 0})
        entry["last"], entry["times"] = round(score, 3), entry["times"] + 1
        entry["best"] = round(max(entry["best"], score), 3)
        return entry

    def start_session(self, now: Optional[float] = None) -> None:
        """Count a session and keep the streak of days in a row."""
        today = today_str(self.clock() if now is None else now)
        self.sessions += 1
        if self.last_day == today:
            pass
        elif self.last_day and (dt.date.fromisoformat(today) - dt.date.fromisoformat(self.last_day)).days == 1:
            self.streak += 1
        else:
            self.streak = 1
        self.last_day = today
        self.best_streak = max(self.best_streak, self.streak)

    # ---- questions ----------------------------------------------------------------------------
    def box(self, letter: str) -> int:
        return self.letters[letter].box if letter in self.letters else 0

    def mastery(self, letter: str) -> float:
        """0 (never practised) to 1 (solid)."""
        return self.box(letter) / (BOXES - 1) if letter in self.letters and self.letters[letter].attempts else 0.0

    def is_learned(self, letter: str) -> bool:
        return letter in self.letters and self.letters[letter].box >= MASTERED_BOX

    def learned(self) -> list:
        return sorted(l for l in self.letters if self.is_learned(l))

    def practised(self) -> list:
        return sorted(l for l, s in self.letters.items() if s.attempts)

    def lesson_mastered(self, lesson_id: str) -> bool:
        return self.lessons.get(lesson_id, {}).get("best", 0.0) >= LESSON_MASTERY

    def next_lesson(self, lessons: Iterable) -> Optional[object]:
        """The first lesson not yet mastered, or None when they all are."""
        return next((l for l in lessons if not self.lesson_mastered(l.id)), None)

    def top_confusions(self, n: int = 3) -> list:
        """[(touched, wanted, count)], the pairs mixed up most."""
        rows = [(*k.split(">"), c) for k, c in self.confusions.items()]
        return sorted(rows, key=lambda r: (-r[2], r[0], r[1]))[:n]

    def confused_with(self, letter: str) -> list:
        """Letters this one has been mixed up with, either way round, most often first."""
        score: dict = {}
        for touched, wanted, c in self.top_confusions(len(self.confusions)):
            if touched == letter:
                score[wanted] = score.get(wanted, 0) + c
            elif wanted == letter:
                score[touched] = score.get(touched, 0) + c
        return sorted(score, key=lambda l: (-score[l], l))

    def days_since(self, letter: str) -> float:
        s = self.letters.get(letter)
        return 30.0 if s is None or not s.last_seen else max(0.0, (self.clock() - s.last_seen) / 86400)

    def pick_practice(self, pool: Iterable, n: int, rng: Optional[random.Random] = None) -> list:
        """`n` distinct letters from `pool` to practise: mostly the shaky and the recently confused, with a few solid ones to keep them
        solid. Weighted sampling, so it is varied, not a fixed order."""
        rng = rng or random.Random()
        pool = list(dict.fromkeys(pool))
        weights = {}
        for l in pool:
            s = self.letters.get(l)
            base = (BOXES - (s.box if s else 0)) ** 2  # box 0 -> 25 ... box 4 -> 1
            confused = 3.0 * sum(c for k, c in self.confusions.items() if l in k.split(">"))
            weights[l] = base + confused + min(self.days_since(l), 7.0)  # not seen for a while: a little more likely
        out = []
        for _ in range(min(n, len(pool))):
            total = sum(weights[l] for l in pool if l not in out)
            r, acc = rng.random() * total, 0.0
            for l in pool:
                if l in out:
                    continue
                acc += weights[l]
                if r <= acc:
                    out.append(l)
                    break
            else:
                out.append(next(l for l in pool if l not in out))
        return out

    def summary(self) -> dict:
        """What a display needs: counts and a per-letter mastery."""
        return {"learned": len(self.learned()), "practised": len(self.practised()), "sessions": self.sessions, "streak": self.streak,
                "best_streak": self.best_streak,
                "mastery": {l: round(self.mastery(l), 2) for l in sorted(self.letters)},
                "confusions": [{"touched": t, "wanted": w, "count": c} for t, w, c in self.top_confusions(5)],
                "lessons": dict(self.lessons)}

    # ---- storage ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1, "letters": {l: asdict(s) for l, s in sorted(self.letters.items())}, "confusions": dict(self.confusions),
                "lessons": {k: dict(v) for k, v in self.lessons.items()}, "sessions": self.sessions, "streak": self.streak,
                "best_streak": self.best_streak, "last_day": self.last_day}

    @classmethod
    def from_dict(cls, data: dict, clock: Callable[[], float] = time.time) -> "Progress":
        p = cls(clock)
        if not isinstance(data, dict):
            return p
        for l, s in (data.get("letters") or {}).items():
            if isinstance(l, str) and len(l) == 1 and isinstance(s, dict):
                p.letters[l] = LetterStat(**{k: type(getattr(LetterStat(), k))(s.get(k, 0)) for k in asdict(LetterStat())})
                p.letters[l].box = max(0, min(BOXES - 1, p.letters[l].box))
        p.confusions = {str(k): int(v) for k, v in (data.get("confusions") or {}).items() if isinstance(k, str) and ">" in k}
        p.lessons = {str(k): {"best": float(v.get("best", 0)), "last": float(v.get("last", 0)), "times": int(v.get("times", 0))}
                     for k, v in (data.get("lessons") or {}).items() if isinstance(v, dict)}
        p.sessions, p.streak = int(data.get("sessions", 0)), int(data.get("streak", 0))
        p.best_streak, p.last_day = int(data.get("best_streak", 0)), str(data.get("last_day", ""))
        return p

    def merge(self, other: "Progress") -> "Progress":
        """Fold another copy of the same learner's record into this one (e.g. the one kept in their account) without counting anything
        twice: per letter the more recently practised copy wins, everything else takes the larger value."""
        for l, s in other.letters.items():
            mine = self.letters.get(l)
            if mine is None or s.last_seen > mine.last_seen or (s.last_seen == mine.last_seen and s.attempts > mine.attempts):
                self.letters[l] = LetterStat(**asdict(s))
        for k, c in other.confusions.items():
            self.confusions[k] = max(self.confusions.get(k, 0), c)
        for k, v in other.lessons.items():
            mine = self.lessons.get(k)
            if mine is None:
                self.lessons[k] = dict(v)
            else:
                mine["best"], mine["times"] = max(mine["best"], v["best"]), max(mine["times"], v["times"])
                if v["times"] > mine["times"] or (v["times"] == mine["times"] and v["last"]):
                    mine["last"] = v["last"]
        self.sessions = max(self.sessions, other.sessions)
        if other.last_day > self.last_day:
            self.last_day, self.streak = other.last_day, other.streak
        self.best_streak = max(self.best_streak, other.best_streak, self.streak)
        return self

    def save(self, path: Path) -> None:
        """Write atomically (a crash mid-write must not lose what was learned)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=1)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path, clock: Callable[[], float] = time.time) -> "Progress":
        """The saved record, or a fresh one if there is none. A damaged file is set aside (never deleted) and a fresh record started."""
        path = Path(path)
        try:
            return cls.from_dict(json.loads(path.read_text()), clock)
        except FileNotFoundError:
            return cls(clock)
        except (ValueError, OSError, TypeError, KeyError):
            try:
                path.replace(path.with_suffix(".damaged"))
            except OSError:
                pass
            return cls(clock)


def progress_path(profile: str = "default", directory: Path = DEFAULT_DIR) -> Path:
    safe = "".join(ch for ch in profile.lower() if ch.isalnum() or ch in "-_") or "default"
    return Path(directory) / f"progress-{safe}.json"
