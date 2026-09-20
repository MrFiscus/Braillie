"""What makes the tutor feel like a teacher you can talk to, for someone who cannot see the screen: it tells you when it cannot see your
sheet, tells you when it did not understand you, tells you what you can say, and lets you set the pace.

Pure pieces (no camera, no voice, no threads), so they are tested with a fake clock; the tutor session runs them (tutor.py).

  Awareness   the camera lost the sheet -> say so (after a moment, then now and then), and say when it is back
  Hearing     something was said but no command was understood -> say so, without nagging
  Settings    speech speed, pace ("relaxed" = wait longer, hints later), sounds on or off, "I didn't catch that" on or off
  help_text   what can be said right now
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import audio_speed

SLOWER_FASTER_STEP = 0.15
RELAXED_DWELL = 1.6  # how much longer a finger may take to rest on a cell to answer
RELAXED_HINTS = 2.0  # how much later the tutor offers help by itself

LOST = "I can't see the sheet. Hold it flat, with the whole page in front of the camera."
STILL_LOST = "I still can't see the sheet. Check that the whole page is in front of the camera and that there is enough light."
FOUND = "Got it, I can see the sheet again."
NOT_UNDERSTOOD = "Sorry, I didn't catch a command there. Say help to hear what you can say."
NOT_CLEAR = "I couldn't hear that clearly. Could you say it again?"


class Awareness:
    """Watches whether the camera can see the sheet while an activity is running, and says so. Call update() about twice a second."""

    def __init__(self, lost_after: float = 3.0, repeat_every: float = 25.0, found_after: float = 1.5):
        self.lost_after, self.repeat_every, self.found_after = lost_after, repeat_every, found_after
        self.lost_since: Optional[float] = None  # when the sheet was last seen going missing
        self.told_at: Optional[float] = None  # when we last said it is missing (None: not told, or it is back)
        self.back_since: Optional[float] = None

    def reset(self) -> None:
        self.lost_since = self.told_at = self.back_since = None

    def update(self, now: float, page_ok: bool, active: bool) -> Optional[str]:
        """What to say now, or None. `active`: an activity is running (nothing is said at the menu or between activities)."""
        if not active:
            self.reset()
            return None
        if page_ok:
            self.lost_since = None
            if self.told_at is None:
                self.back_since = None
                return None
            if self.back_since is None:
                self.back_since = now
            if now - self.back_since >= self.found_after:  # steadily back, not a flicker
                self.told_at = self.back_since = None
                return FOUND
            return None
        self.back_since = None
        if self.lost_since is None:
            self.lost_since = now
        if self.told_at is None and now - self.lost_since >= self.lost_after:
            self.told_at = now
            return LOST
        if self.told_at is not None and now - self.told_at >= self.repeat_every:
            self.told_at = now
            return STILL_LOST
        return None


class Hearing:
    """Says "I didn't catch that" when something was said but no command was understood (see voice_io's listener_status()["heard"]).
    Not every time: a room with other people talking must not make the tutor nag, so at most once per `cooldown` seconds, and never for a
    stray syllable or while it is itself talking."""

    UNMATCHED = "no command phrase matched"
    UNCLEAR = "confidence below threshold"

    def __init__(self, cooldown: float = 20.0):
        self.cooldown = cooldown
        self.seen: Optional[int] = None  # the count of utterances already looked at (the history before we started is ignored)
        self.last_said = -1e9

    def update(self, heard: Optional[dict], now: float, active: bool, enabled: bool = True) -> Optional[str]:
        if not heard:
            return None
        n = int(heard.get("n", 0))
        if self.seen is None:
            self.seen = n
            return None
        if n <= self.seen:
            return None
        self.seen = n  # this one is dealt with, whatever happens below
        if not (active and enabled) or heard.get("matched") or now - self.last_said < self.cooldown:
            return None
        text = str(heard.get("text", ""))
        if len(text.split()) < 2 and len(text) < 5:
            return None  # "uh", "hm": not worth a reply
        reason = heard.get("reason")
        if reason == self.UNCLEAR:
            self.last_said = now
            return NOT_CLEAR
        if reason == self.UNMATCHED:
            self.last_said = now
            return NOT_UNDERSTOOD
        return None


@dataclass
class Settings:
    """Things a learner may want to change. All safe to change at any time."""
    speech_speed: float = 1.0  # 0.6 (slow) .. 1.5 (fast)
    pace: str = "normal"  # "relaxed": wait longer before helping, allow a longer rest to answer
    tones: bool = True  # the little sounds for right, wrong, heard
    hearing_feedback: bool = True  # "I didn't catch that"

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, changes: dict) -> None:
        """Apply the given changes, or raise ValueError (and change nothing) if any is not allowed."""
        known = {"speech_speed", "pace", "tones", "hearing_feedback"}
        unknown = set(changes) - known
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}; choose from {', '.join(sorted(known))}")
        new = dict(changes)
        if "speech_speed" in new:
            v = new["speech_speed"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not audio_speed.MIN_SPEED <= v <= audio_speed.MAX_SPEED:
                raise ValueError(f"speech_speed must be a number from {audio_speed.MIN_SPEED} to {audio_speed.MAX_SPEED}")
            new["speech_speed"] = round(float(v), 2)
        if "pace" in new and new["pace"] not in ("normal", "relaxed"):
            raise ValueError('pace must be "normal" or "relaxed"')
        for k in ("tones", "hearing_feedback"):
            if k in new and not isinstance(new[k], bool):
                raise ValueError(f"{k} must be true or false")
        for k, v in new.items():
            setattr(self, k, v)

    @property
    def dwell_scale(self) -> float:
        return RELAXED_DWELL if self.pace == "relaxed" else 1.0

    @property
    def hint_scale(self) -> float:
        return RELAXED_HINTS if self.pace == "relaxed" else 1.0


def help_text(hub_mode: Optional[str], mode: str, state: str) -> str:
    """What can be said right now, in a sentence or two (the tutor speaks it)."""
    ends = ("You can also say braillo and then ask me a question, slower or faster to change how quickly I talk, "
            "and take your time if you would like more time.")
    if hub_mode == "menu":
        return ("Choose an option. Learn teaches the letters. Read reads words aloud. Quiz asks you to find a letter. "
                f"Say learn, read, or quiz. {ends}")
    if hub_mode == "learn" or mode == "learn":
        return ("In a lesson you can say hint for help, repeat to hear it again, found it to answer straight away, next to skip a letter, "
                f"explore to feel the letters freely, practice for a review, or stop or finish. Say menu to choose something else. {ends}")
    if hub_mode == "quiz" or mode == "letters":
        return f"In the quiz you can say found it to answer, hint, repeat, next to skip a question, or stop or finish. If I cannot see your finger, click the picture where it is. Say menu to choose something else. {ends}"
    if hub_mode == "read" or mode == "read":
        return (f"When reading, rest a finger on a word and I will read it. You can also say read or found it, or repeat to hear "
                f"the last word again. Say stop or finish when you are done, or menu to choose something else. {ends}")
    if mode == "explore":
        return f"Rest a finger on any cell and I will tell you what it is. You can say repeat, hint, next page, or stop or finish. {ends}"
    return f"You can say start, repeat, hint, found it, next, or stop or finish. {ends}"
