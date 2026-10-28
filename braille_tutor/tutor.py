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
import collections
import difflib
import json
import logging
import os
import random
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np

import reader
from detect import AMBER, GREEN, RED, _Detector, braille_status, cell_at, dot_distance, draw_cell, draw_detections, draw_hud, draw_page_outline, letter_of, locked_check_line, open_camera, page_source_from_args, page_status
from drill import drill as _drill
from llm import AskTutor, Coach, LLMClient, Pending, looks_like_question, strip_ask_wake, ASK_CANCEL
import accessibility as acc
import audio_speed
import phonelink
from earcons import Earcons
from fingertip import FingerTracker, best_finder
from learn import DOTS as LETTER_DOTS, DWELL_SECONDS, LESSONS, LEVELS, RETRY_LEVEL, Dwell, Journey, level_of
from page import to_image, to_page
from progress import Progress, progress_path
from sheets import Symbol, feel_for, get_sheet, letter_symbol
from vote import CellLocker

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


class AdaptiveTargetPlanner:
    """Asynchronously ask Claude to order the next quiz's known targets."""

    _SYSTEM = (
        "You are an adaptive braille-literacy quiz planner. Choose and order the next "
        "targets from the supplied candidate list. Prioritize targets with more misses, "
        "especially recent misses, but include variety so one or two hard targets do not "
        "crowd out all other practice. Return only the requested structured data."
    )
    _SCHEMA = {
        "name": "braillie_adaptive_targets",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["targets", "reason"],
            "additionalProperties": False,
        },
    }

    def __init__(self, client: LLMClient):
        self.client = client
        self._pending: Optional[Pending] = None
        self._signature: Optional[str] = None
        self.last_response: Optional[dict] = None

    @staticmethod
    def _signature_for(candidates: list[str], count: int, mode: str, history: dict) -> str:
        return json.dumps(
            {"candidates": candidates, "count": count, "mode": mode, "history": history},
            sort_keys=True,
        )

    def prefetch(self, candidates: list[str], count: int, mode: str, history: dict) -> None:
        """Start the next-plan request without holding up the learner."""
        signature = self._signature_for(candidates, count, mode, history)
        if self._pending is not None and self._signature == signature:
            return
        payload = json.dumps({
            "mode": mode,
            "candidate_targets": candidates,
            "target_count": count,
            "miss_history": history,
        }, sort_keys=True)
        self._signature = signature
        self._pending = Pending(lambda: self.client.chat_json(self._SYSTEM, payload, self._SCHEMA))

    def take_ready(self, candidates: list[str], count: int, mode: str, history: dict) -> tuple[Optional[list[str]], str]:
        """Return a validated ready plan, or a reason to use the local fallback."""
        signature = self._signature_for(candidates, count, mode, history)
        if self._pending is None or self._signature != signature:
            return None, "no matching Claude plan is ready"
        if not self._pending.done.is_set():
            return None, "Claude plan is still pending"
        result = self._pending.value
        if isinstance(result, Exception):
            return None, f"Claude request failed: {type(result).__name__}: {result}"
        if not isinstance(result, dict):
            return None, "Claude returned no adaptive plan"
        targets = result.get("targets")
        if not isinstance(targets, list) or len(targets) != count:
            return None, "Claude returned the wrong number of targets"
        if any(not isinstance(target, str) for target in targets):
            return None, "Claude returned a non-string target"
        if len(set(targets)) != len(targets) or any(target not in candidates for target in targets):
            return None, "Claude returned unknown or duplicate targets"
        self.last_response = result
        return targets, "Claude adaptive plan"

SPEECH_TIMEOUT = 8.0  # a single say() call gives up after this long (see TutorSession.say): no legitimate
# utterance takes anywhere near this, so past it something downstream (network, audio device) is stuck
SPEECH_SECONDS_PER_WORD = 0.5  # ...plus this much per word, so a long answer from Braillo is not cut short by that limit
# Reminders nobody asked for ("I can't see your finger", "I'm waiting for the sheet"): said at once the first time, then only after
# these gaps while nothing changes, and not at all after NAG_MAX times, until the learner says something (see TutorSession.nag).
NAG_GAPS = (0.0, 30.0, 60.0, 120.0)
NAG_MAX = 4
TALK_MIN_WORDS = 2  # free speech shorter than this is not sent to Braillo to work out (a cough, "um")
OVER_SPEECH = "heard over speech"  # voice_io's reason for a word or two caught while the tutor was talking
EXPLORE_DWELL_SECONDS = 0.7  # a finger resting this long on one spot is "feeling" it
NO_FINGER_SECONDS = 4.0  # in quiz/read: how long with no fingertip before the tutor says it cannot see one
NO_FINGER_AGAIN = 12.0  # ...and how long between those reminders
EXPLORE_STILL_MM = 4.0  # ...where "one spot" means it moved less than this
EXPLORE_LOST_SECONDS = 2.0  # finger gone this long: say so, once
EXPLORE_OFF_CELL_SECONDS = 2.5  # resting this long on blank paper: say so, once
# The three things a learner can choose at the menu, and what each one needs: which kind of session, and which printed sheet.
HUB_MODES = {"learn": {"mode": "learn", "sheet": "alphabet", "title": "Learn"},
             "read": {"mode": "read", "sheet": "words", "title": "Read"},
             "quiz": {"mode": "letters", "sheet": "lookalikes", "title": "Quiz"}}
MENU_LINE = ("Choose an option. Learn teaches the letters a few at a time on the alphabet sheet. "
             "Read says a word aloud when you rest your finger on it on the words sheet. "
             "Quiz names a letter on the look-alikes sheet and you find it. "
             "Say learn, read, or quiz.")
# What each mode does is explained in full once, in the opening greeting (see greeting() / _maybe_greet());
# every later return to the menu (from "stop"/"finish", or saying it again while already there) uses this
# short prompt instead, so the same three explanations are not repeated on every round of a demo.
MENU_LINE_SHORT = "Choose an option. Say learn, read, or quiz."
PROMPTS = {  # canned things the website may ask the tutor to say (it cannot make the tutor say anything else; {name} is a cleaned first name)
    # The lines heard at every start have a few versions, each saying the same thing in other words, so it never sounds like one script
    # read again and again (next_prompt picks; never the same one twice running). The short replies stay one line each.
    "welcome": (
        "Hi, I'm Braillo, and I'll be teaching you braille. Would you like to sign in with Google, so I can remember "
        "your progress, or carry on as a guest? Just say Google, or guest.",
        "Welcome to Braillie! I'm Braillo. If you sign in with Google, I'll keep track of how you're doing. If you'd "
        "rather not, you can come in as a guest. Say Google, or guest.",
        "Hello, and welcome to Braillie. I'm Braillo, your braille tutor. To save your progress, say Google and sign "
        "in. To just try it out, say guest. You can also use the tab key to move between the options.",
        "Hi there, welcome to Braillie. You can sign in with Google to keep your progress, or come in as a guest, "
        "without an account. Say Google, or say guest.",
        "Welcome! I'm Braillo. Before we begin, would you like to sign in? Say Google to save your progress, or guest "
        "to jump straight in.",
        "Hello! This is Braillie, and I'm Braillo. Say Google if you'd like me to remember your progress, or say guest "
        "to start without an account.",
        "Hi, I'm Braillo. Welcome to Braillie. There are two ways in: say Google to sign in and save your progress, or "
        "say guest to start right away.",
        "Welcome to Braillie. I'm Braillo, and I'm here to help you learn braille by touch. Say Google to sign in, or "
        "guest to carry on without an account.",
        "Hello, welcome to Braillie! Would you like to sign in with Google, so your progress is saved? Say Google, or "
        "say guest to come in as a guest. The tab key works too.",
        "Hi! Braillo here. Welcome to Braillie. Sign in with Google and I'll remember where you got to, or say guest to"
        " start fresh, without an account.",
    ),
    "welcome_back": (
        "Welcome back, {name}! Say continue to pick up where you left off, or switch if this isn't you.",
        "Hi again, {name}. Shall we carry on? Say continue, or say switch to use a different account.",
        "Good to see you again, {name}. Say continue to keep going, or switch if you're someone else.",
        "Hello again, {name}! Ready to carry on? Just say continue. Or say switch to sign in another way.",
        "Nice to have you back, {name}. Say continue, and we'll pick up where we left off. Or say switch.",
        "{name}, welcome back! Say continue to carry on, or switch if this isn't you.",
        "Hey, {name}, good to hear from you again. Say continue when you're ready, or switch to change accounts.",
        "Welcome back! Is that you, {name}? Say continue if it is, or switch if it's someone else.",
        "Hi, {name}, back for more? Say continue to pick up where you left off, or say switch.",
        "There you are, {name}! Say continue to keep going, or switch to sign in another way.",
    ),
    "retry_back": "Sorry, I didn't catch that. Say continue, or switch.",
    "again_name": "Okay. What's your first name, then? Or say skip.",
    "retry_choice": "Sorry, I didn't catch that. Say Google to sign in, or guest to carry on without an account.",
    "ask_name": (
        "Okay, no account. Just so you know, your progress won't be saved this time. What's your first name? Or say "
        "skip.",
        "Sure, we'll go without an account, so nothing gets saved. What should I call you? You can also say skip.",
        "Alright, as a guest, then. Your progress won't be kept, but that's fine. What's your first name? Or just say "
        "skip.",
        "No account, no problem. Remember, nothing is saved this way. What's your first name? Say skip if you'd rather "
        "not.",
        "Great, let's start as a guest. Your progress won't be saved. What would you like me to call you? Or say skip.",
        "Okay, guest it is. Nothing will be saved, just so you know. What's your first name? You can say skip, too.",
        "Got it, no account. Your progress won't be kept this time. What's your first name? Or say skip.",
        "Fine by me. Without an account, I won't keep your progress. Could you tell me your first name? Or say skip.",
        "Okay, let's skip the account. Your progress won't be saved. What's your first name? Or say skip.",
        "Sure thing. As a guest, nothing gets saved. What's your first name? If you'd rather not say, just say skip.",
    ),
    "confirm_name": "Is your name {name}? Say yes, or tell me your name again.",
    "retry_name": "Sorry, I didn't catch that. What's your first name? Or say skip.",
    "go_guest": (
        "Nice to meet you, {name}! Now let's connect your phone.",
        "Lovely to meet you, {name}. Next, let's get your phone connected.",
        "Hi, {name}! Let's connect your phone next.",
        "Great to meet you, {name}. Now for your phone.",
        "Hello, {name}! Next up, your phone.",
        "Welcome, {name}. Let's get your phone set up.",
        "Nice one, {name}. Now let's link your phone.",
        "Pleased to meet you, {name}. Let's set up your phone.",
        "Hi there, {name}! Time to connect your phone.",
        "Good to meet you, {name}. Let's get your phone ready.",
    ),
    "go_guest_anon": (  # (a guest is not asked for a name: just "guest". The next page says what to do: connect the phone)
        "Hi, guest!",
        "Hello, guest!",
        "Welcome, guest!",
        "Hi there, guest!",
        "Welcome in, guest!",
        "Hello there, guest!",
        "Hi, guest, good to have you!",
        "Welcome aboard, guest!",
        "Hey there, guest!",
        "Hi, guest! Glad you're here.",
    ),
    "go_google": "Opening Google now. From here, use your keyboard or screen reader on the Google page.",
    "give_up": "I'll stop listening for now. You can use the buttons on the screen instead: press tab to move between "
               "them.",
    "connect_phone": (
        "Now let's connect your phone. Open its camera, point it at the code on the screen, and tap the link that "
        "pops up. Then tap the big Start button, and I'll say hello.",
        "Next, your phone. Point its camera at the code on the screen and tap the link that appears. When the page "
        "opens, tap the big Start button, and you'll hear me say hello.",
        "Let's link your phone. Open the camera, aim it at the code on the screen, and tap the link. On the page that"
        " opens, tap the big Start button.",
        "Time to connect your phone. Hold its camera up to the code on the screen, then tap the link it shows. Tap "
        "the big Start button, and I'll greet you.",
        "Now for your phone. Scan the code on the screen with its camera, and tap the link. When the page loads, tap "
        "the big Start button.",
        "Let's get your phone ready. Point its camera at the code on the screen until a link appears, and tap it. "
        "Then tap the big Start button, and I'll say hello.",
        "Your phone is next. Open the camera, point it at the code on the screen, and tap the link that comes up. On "
        "the new page, tap the big Start button.",
        "Okay, now the phone. Use its camera to scan the code on the screen, and tap the link. When the page loads, "
        "tap the big Start button.",
        "Now we'll connect your phone. Hold its camera up to the code on the screen, and tap the link that appears. "
        "Then press the big Start button, and I'll say hello.",
        "Let's bring in your phone. Point its camera at the code on the screen, and tap the link. On the page that "
        "opens, tap the big Start button, and I'll greet you.",
    ),
    # (the phone dropped out for good, mid-way: the website shows the code again, see usePhoneWatch.ts)
    "reconnect_phone": "I've lost your phone. Scan the code on the screen with its camera again, tap Start, and we'll carry on where we were.",
    # (only when they say it is not working: the way in without scanning, which most people never need to hear)
    "connect_help": "If it won't scan, open the address on the screen in your phone's browser, and type the code {code}.",
}
DIALOGUE_SECONDS = 30.0  # the sign-in page keeps the conversation open by pinging; if it stops (tab closed) the tutor goes back to normal
# Your page (the profile). Only the tutor says these, never the browser's own voice: {progress} and {letters} are filled in from the
# tutor's record of the learner (see progress_line and letters_line), never from the page.
PROMPTS.update({
    "account": (
        "This is your page, {name}. {progress} Say continue to go home, letters to hear your alphabet, or sign out.",
        "Here's how you're getting on, {name}. {progress} Say letters to hear which ones you know, continue to go home, or sign out.",
        "Hi, {name}, this is your page. {progress} To hear your alphabet, say letters. Say continue to go home, or say sign out.",
        "{name}, here's your progress. {progress} Say continue to carry on, letters for your alphabet, or sign out.",
        "Welcome to your page, {name}. {progress} You can say letters, continue, or sign out.",
    ),
    "account_letters": "{letters} Say continue to go home, or say learn to practise.",
    "account_help": "On this page, say letters to hear your alphabet, continue to go home, learn, read or quiz to start one of them, "
                    "or sign out. Say repeat to hear your page again.",
    "account_retry": "Sorry, I did not catch that. Say continue, letters, or sign out.",
    "account_bye": "Signing you out. See you soon, {name}.",
})


def clean_name(name) -> str:
    """A name safe to show and to say: letters, digits, spaces and a few marks, at most 40 characters; "friend" if nothing is left."""
    text = re.sub(r"[^\w .'\-]", "", str(name or ""), flags=re.UNICODE)
    return " ".join(text.split())[:40].strip() or "friend"


# How a letter's name comes back from speech recognition when it is said on its own ("B" is often written "be" or "bee").
LETTER_SOUNDS = {"ay": "a", "bee": "b", "be": "b", "see": "c", "sea": "c", "dee": "d", "ee": "e", "eff": "f", "gee": "g", "aitch": "h",
                 "eye": "i", "jay": "j", "kay": "k", "el": "l", "elle": "l", "em": "m", "en": "n", "oh": "o", "pee": "p", "cue": "q",
                 "queue": "q", "are": "r", "ess": "s", "tea": "t", "tee": "t", "you": "u", "vee": "v", "ex": "x", "why": "y", "zed": "z",
                 "zee": "z", "double you": "w", "double u": "w"}
_SPOKEN_LETTER = re.compile(
    r"^(?:(?:um|uh|so|okay|ok|yes|yeah)\s+)?(?:(?:is (?:this|it|that)|it'?s|it is|this is|that'?s|that is|i think (?:it'?s|this is)?|here'?s)\s+)?"
    r"(?:the\s+)?(?:letter\s+)?(?P<l>[a-z]|double (?:you|u)|[a-z]{2,6})(?:\s+(?:here|right|now))?$")


# The website's pages, as Braillo is told about them (the website says which is open, see set_page).
WEBSITE_PAGES = {
    "landing": "the front page of the website, about Braillie; nothing is running yet. To begin they say start learning, which opens "
               "the sign-in page",
    "login": "signing in: they say Google to sign in (progress saved) or guest (nothing saved)",
    "connect-phone": "connecting the phone: they scan the QR code on the laptop screen with the phone's camera, open the link, and tap "
                     "Start; the phone then becomes the camera that watches the sheet",
    "modes": "home: they choose Learn (the letters, a few at a time), Read (words read aloud) or Quiz (find look-alike letters)",
    "practice": "the activity they chose, with the sheet under the phone's camera",
    "account": "their own page: how many letters they know, days in a row, sessions, and their alphabet",
    "privacy": "the privacy policy: what Braillie keeps and sends, and what it does not",
    "terms": "the terms of use",
    "accessibility": "the accessibility statement: how Braillie works with screen readers, keyboards and voice",
}
# Pages that have no voice of their own: arriving on one, Braillo says where they are and how to hear what they can do there.
ARRIVAL_LINES = {
    "landing": "Hi, I'm Braillo. Would you like me to talk you through this website by voice? Say yes, or no.",
    "privacy": "This is the privacy policy. Say go back to return, or more options to hear the links on this page.",
    "terms": "These are the terms of use. Say go back to return, or more options to hear the links on this page.",
    "accessibility": "This is the accessibility statement. Say go back to return, or more options to hear the links on this page.",
}
# The front page asks first (above); a "yes" gets the welcome, and later visits a short reminder; a "no" and it stays quiet there.
LANDING_WELCOME = ("Welcome to Braillie. Here you learn braille by touch, with me, Braillo, talking you through it. First, print the "
                   "practice sheets, and press each dot through from the back with a pen, so the dots stand up. If you can't do that "
                   "yourself, ask someone to print and poke them for you. When you're ready, say start learning to begin.")
LANDING_NO = "Okay, I'll stay quiet. If you want me, just say Braillo."
_YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|please|okay|ok|yes please|go on|go ahead|of course|i would|i do|please do)\b")
_NO = re.compile(r"^(?:no|nope|nah|no thanks|no thank you|not now|i'?m fine|that'?s okay)\b")
# "It's not working", "I can't scan it", "nothing happens": on the connect-phone page, the way in without scanning (see _cant_connect)
_CANT_CONNECT = re.compile(r"\b(?:can'?t|cannot|can not|won'?t|doesn'?t|does not|isn'?t|is not|didn'?t|not)\s+(?:\w+\s+){0,2}?"
                           r"(?:scan\w*|connect\w*|work\w*|open\w*|load\w*|find\w*|see\w*)\b|\bnothing (?:is )?happen\w*\b|"
                           r"\b(?:having |got )?(?:trouble|problems?)\b|\bstuck\b")
_SIGN_IN_WORDS = re.compile(r"\b(?:go+g+(?:le|el|ol|al)s?|guest|sign ?in|log ?in)\b")
_CHANGE_LEVEL = re.compile(r"\b(?:change|set|reset|pick|choose|switch|redo)\b.*\blevel\b|\bmy level\b")
# "Help" on the pages of the way in, where the activities' own help does not apply yet (nothing can be learned before signing in).
PAGE_HELP = {
    "login": "Say Google to sign in, or guest to carry on without an account.",
    "connect-phone": ("Scan the code on the screen with your phone's camera, open the link, and tap Start. If it won't scan, type the "
                      "address on the screen into the phone's browser."),
    "account": "Say letters to hear your alphabet, continue to go home, or sign out.",
}
HELP_CONTROLS = 4  # "help" on a page names this many of its controls (its main ones: the website lists those first)
UTILITY_CONTROLS = {"braillie, home", "sign in", "display", "standard", "large", "larger", "largest", "match my device", "light", "dark",
                    "black on white", "yellow on black"}  # (on every page: not what "help" there is about)


def _or_list(names: list) -> str:
    """ "A", "A or B", "A, B, or C" """
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])}, or {names[-1]}"


ARRIVAL_AGAIN_SECONDS = 10.0  # (the same page reported twice in quick succession, by two quick reloads say, is announced once)
_CONTROL_LEAD = re.compile(r"^(?:(?:please|can you|could you|okay|ok|now)\s+)*(?:press|click|tap|select|choose|open|go to|hit|pick|use)\s+(?:the\s+)?")
_CONTROL_TAIL = re.compile(r"\s+(?:button|link|option|please|for me)$")
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6}


def lesson_named(text: str) -> Optional[int]:
    """The lesson a short request names, counted from 0: "lesson three", "go to lesson 2", "let's do K to O", "the look-alikes
    lesson". None if it names none (a name alone, without "lesson" or a word like "go to", is not taken: it may be about letters)."""
    t = " ".join(re.sub(r"[^a-z0-9' ]", " ", str(text or "").lower().replace("-", " ")).split())
    if not t or len(t.split()) > 9:
        return None
    m = re.search(r"\blesson (?:number )?(\d|one|two|three|four|five|six)\b|\b(first|second|third|fourth|fifth|sixth) lesson\b", t)
    if m:
        word = m.group(1) or m.group(2)
        number = int(word) if word.isdigit() else _NUMBERS[word]
        return number - 1 if 1 <= number <= len(LESSONS) else None
    if not ("lesson" in t or re.match(r"^(?:go to|do|start|open|switch to|let'?s do|let'?s learn|teach me|take me to|can we do)\b", t)):
        return None
    for i, lesson in enumerate(LESSONS):
        names = {lesson.title.lower().replace("-", " "), f"{lesson.letters[0]} to {lesson.letters[-1]}"}
        if lesson.id == "l6":
            names |= {"look alikes", "lookalikes", "look alike"}
        if any(re.search(rf"\b{re.escape(name)}\b", t) for name in names):
            return i
    return None


def spoken_letter(text: str) -> Optional[str]:
    """The letter a short answer names ("E", "is this E?", "it's the letter bee"), or None if it does not name one on its own."""
    t = " ".join(re.sub(r"[^a-z' ]", " ", str(text or "").lower()).split())
    m = _SPOKEN_LETTER.match(t)
    if not m:
        return None
    word = m.group("l")
    return word if len(word) == 1 else LETTER_SOUNDS.get(word)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _spoken_list(letters) -> str:
    """"A", "A and B", "A, B and C": letters as they are said."""
    up = [l.upper() for l in letters]
    return up[0] if len(up) == 1 else f"{', '.join(up[:-1])} and {up[-1]}"


def progress_line(summary: dict) -> str:
    """The learner's progress in a sentence or two, for their page: letters known, days in a row, sessions."""
    learned, streak, sessions = summary.get("learned", 0), summary.get("streak", 0), summary.get("sessions", 0)
    if not sessions and not summary.get("practised", 0):
        return "You have not had a lesson yet. Say learn to start your first one."
    known = "You do not know any letters yet." if not learned else f"You know {learned} of 26 letters."
    return f"{known} You have practised {_count(streak, 'day', 'days')} in a row, over {_count(sessions, 'session', 'sessions')}."


def letters_line(summary: dict, learned=(), practised=()) -> str:
    """Which letters the learner knows, which they are still learning, and how many are left to try."""
    learning = [l for l in practised if l not in learned]
    left = 26 - len(set(learned) | set(practised))
    parts = []
    if learned:
        parts.append(f"You know {_spoken_list(learned)}.")
    if learning:
        parts.append(f"You are still learning {_spoken_list(learning)}.")
    if not parts:
        return "You have not practised any letters yet."
    if left:
        parts.append(f"{_count(left, 'letter is', 'letters are')} still to try.")
    return " ".join(parts)


_prompt_rng = random.Random()
_planned: dict = {}  # prompt name -> the version it will say next (chosen ahead, so only that one is prepared: see common_lines)
_planned_lock = threading.Lock()


def _versions(name: str) -> tuple:
    v = PROMPTS[name]
    return (v,) if isinstance(v, str) else v


def planned_prompt(name: str) -> str:
    """The version of a prepared line that will be said next."""
    with _planned_lock:
        if name not in _planned:
            _planned[name] = _prompt_rng.choice(_versions(name))
        return _planned[name]


def next_prompt(name: str) -> str:
    """The version to say now (the planned one), and the one after chosen at once: another version, never the same twice running."""
    text = planned_prompt(name)
    with _planned_lock:
        others = [v for v in _versions(name) if v != text] or [text]
        _planned[name] = _prompt_rng.choice(others)
    return text


def common_lines() -> list:
    """The tutor's fixed lines, the ones it says most (the sign-in conversation, the menu, "I didn't catch that"), to prepare ahead.
    (Only the version of each that will be said next: preparing every version would cost several times as much speech.)"""
    lines = [t for t in (planned_prompt(n) for n in PROMPTS) if "{" not in t]
    lines += [f"Okay. {MENU_LINE_SHORT}", MENU_LINE_SHORT, acc.NOT_UNDERSTOOD, acc.NOT_CLEAR, acc.LOST, acc.FOUND, "Okay.", "What's your question?"]
    return lines


GREETINGS = (
    "What do you want to do today, {name}?",
    "Hi, {name}! What would you like to do?",
    "Welcome, {name}. What shall we do today?",
    "Great, {name}, you're all set. What would you like to do?",
    "Hello, {name}! What are we doing today?",
    "Okay, {name}, we're ready. What would you like to try?",
    "Lovely, {name}, everything is ready. What shall we do?",
    "Right, {name}. What do you feel like doing today?",
    "All set, {name}! Where would you like to start?",
    "Nice to have you here, {name}. What would you like to do?",
)
_last_greeting: list = [None]


def greeting(name: str) -> str:
    """The opening question, in one of a few ways (not the same as last time), then the choices in full."""
    opener = _prompt_rng.choice([g for g in GREETINGS if g != _last_greeting[0]])
    _last_greeting[0] = opener
    return f"{opener.format(name=name)} {MENU_LINE}"


IDENTIFY_SECONDS = 30.0  # after "next page": how long to keep trying to work out which sheet is on the desk
IDENTIFY_EVERY_SECONDS = 1.0
IDENTIFY_UNCHANGED_SECONDS = 8.0  # after "next page", if nothing at all changes in view for this long, carry on with the same sheet
SCENE_CHANGE = 6.0  # mean brightness difference (0-255) of a small copy of the picture that counts as "something moved in view"
SHEET_SPOKEN = {"alphabet": "the alphabet sheet", "words": "the words sheet", "numbers": "the numbers and signs sheet",
                "lookalikes": "the look-alikes sheet"}
EXPLORE_MARGIN_MM = 3.0  # after a cell is announced, stay quiet until the finger is this far outside it

# How long a point posted to /api/finger (a click or tap on the video) stands in for the camera's own fingertip tracking.
# It must outlast a rest that counts as an answer (DWELL_SECONDS, plus time to say "found it"), and it must expire: a single
# stray tap used to pin the finger to one cell for the rest of the session, and the tutor then insisted, letter after letter,
# that the finger was on whatever had been clicked.
POSTED_FINGER_SECONDS = 10.0


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


def voice_check(voice) -> dict:
    """Will speech actually be HEARD? {"mode": "mock"|"live", "ok": bool, "problems": [...], "warnings": [...]}.

    voice_io swallows speech errors into its log, so without this a missing key or package just means silence. `problems` make
    all speech silent; `warnings` are for things that leave speech on but not quite right (none at the moment)."""
    import importlib.util

    if getattr(voice, "MOCK_MODE", False):
        return {"mode": "mock", "ok": False, "warnings": [],
                "problems": ["mock voice: speech is printed in the terminal, not played (drop --mock and unset VOICE_IO_MOCK to hear it)"]}
    problems, warnings = [], []
    if not getattr(voice, "DEEPGRAM_API_KEY", ""):
        problems.append("DEEPGRAM_API_KEY is not set (put it in a .env file at the repo root: DEEPGRAM_API_KEY=...)")
    for module, why in (("pyaudio", "plays the speech and listens to the microphone"), ("deepgram", "the Deepgram speech client (pip install deepgram-sdk==7.9.0)")):
        if importlib.util.find_spec(module) is None:
            problems.append(f"the Python package {module!r} is not installed: {why}")
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


SEEN_WITHIN_SECONDS = 1.5  # the page edges were really found this recently


def page_seen_now(feed: "CameraFeed") -> bool:
    """Is the sheet really in view right now, not merely remembered? The page finders keep reporting the last position for a few seconds after
    the sheet leaves, to ride out a wobble ("holding"; the paper finder holds its edges for 5 s): fine for drawing, but too slow to tell
    a learner who cannot see that the sheet has gone."""
    if feed.identify_until:  # between pages the sheet is meant to be missing
        return True
    if not feed.page_ok:
        return False
    src = feed.page_src
    source = getattr(src, "source", "")
    if source in ("holding", "none"):
        return False
    paper = getattr(src, "paper_fallback", None) or (src if hasattr(src, "last_seen") else None)
    if paper is not None and (source == "paper" or not source):
        seen = getattr(paper, "last_seen", 0.0)
        return seen == 0.0 or time.time() - seen < SEEN_WITHIN_SECONDS
    return True


def wire_new_page(session: "TutorSession", feed: "CameraFeed") -> None:
    """Connect "next page": the session asks the feed to forget the old page; what the feed then works out is spoken and applied."""
    session.new_page = feed.new_page
    session.select_sheet = feed.select_sheet if feed.known_sheets else None
    session.page_ok_fn = lambda: page_seen_now(feed)  # (see page_seen_now: really seen, not just remembered)
    session.current_sheet = lambda: feed.sheet_name
    session.identifies_sheets = bool(feed.known_sheets)
    # Grading uses the same centres the video draws. On known demo sheets those centres stay on the
    # printed layout (photos): live refine under a finger used to drag boxes on words / lookalikes.
    session.grade_cells = lambda: _cells_for_grading(session.cells, feed.stable)
    feed.on_sheet = session.set_sheet
    feed.announce = lambda text: threading.Thread(target=session.sheet_news, args=(text,), daemon=True).start()  # speaking blocks


def _cells_for_grading(printed: list, stable: list) -> list:
    """Printed-sheet letters at printed centres (demo photos), or the last drawn stable boxes if no layout yet.

    Known hub sheets (alphabet / words / lookalikes) keep the layout centres from the sheet files so a
    covering finger cannot drag the graded square onto a neighbour."""
    if printed:
        return list(printed)
    return list(stable) if stable else []


def known_sheets_for(a, setup) -> Optional[dict]:
    """Explore mode on a printed sheet can switch between all of them when the page is turned; other modes stay on their sheet."""
    from sheets import SHEET_NAMES
    return {name: get_sheet(name) for name in SHEET_NAMES} if (a.mode in ("explore", "learn", "menu") and setup.observed) else None


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


class _JourneyHost:
    """What the lesson engine (learn.py) needs from the outside, provided by the running tutor session."""

    def __init__(self, session: "TutorSession"):
        self.s = session

    def say(self, text: str) -> None:
        self.s.say(text)

    def nag(self, text: str) -> None:
        self.s.nag(text)

    def tone(self, kind: str) -> None:
        self.s.earcons.play(kind)

    def finger(self):
        return self.s.finger()

    def cells(self) -> list:
        return self.s._explore_cells()  # a fresh reading of the page, or the sheet's layout if there is none yet

    def known_cells(self) -> list:
        """The active sheet's own printed letters at the printed centres from the demo-sheet layout.

        Unlike cells() above (for explore mode, where hearing what the camera actually sees is the point),
        grading a lesson answer should not depend on a clean live reading of the exact cell the learner's
        finger is resting on: that finger is, by definition, occluding it right then. Centres stay on the
        sheet photos so denser words / lookalikes boxes are not dragged onto neighbours."""
        grade = getattr(self.s, "grade_cells", None)
        if callable(grade):
            cells = grade()
            if cells:
                return cells
        return self.s.cells

    def sheet_name(self) -> str:
        return self.s.current_sheet()

    def identifies_sheets(self) -> bool:
        return bool(getattr(self.s, "identifies_sheets", False))

    def staying(self) -> bool:
        """Is stopping a lesson only going back to the menu (the menu-driven app), not leaving?"""
        return bool(self.s.hub)

    def symbol_key(self, cell: dict):
        """What the sheet has printed in this cell (on the numbers sheet, the cell with dot 1 is the number 1, not A)."""
        sym = self.s._symbol(cell)
        return sym.key if sym else None

    def request_sheet(self, name: str) -> None:
        if self.s.new_page is not None:
            self.s.new_page()  # forget the old page and work out which sheet is on the desk now (the tutor announces it)

    def explore_reset(self) -> None:
        self.s._ex = self.s._new_explore_state()

    def explore_tick(self, now: float) -> None:
        self.s.explore_tick(now)

    def page_visible(self) -> bool:
        """Can the camera see the sheet right now? (When it cannot, the finger is not the problem worth mentioning.)"""
        return self.s.page_ok_fn() if self.s.page_ok_fn is not None else True

    def finish(self) -> None:
        self.s.state = "done"
        self.s.finished.set()

    def save(self) -> None:
        self.s.save_progress()


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
                 names: Optional[dict] = None, coach: Optional[Coach] = None, contracted: bool = False,
                 adaptive_planner: Optional[AdaptiveTargetPlanner] = None,
                 progress: Optional[Progress] = None, progress_file=None, tones: bool = True,
                 ask: Optional[AskTutor] = None):
        self.contracted = contracted  # read words as contracted (Grade 2) braille: for a real page; printed sheets are plain letters
        self.coach, self.confusions, self.last_debrief = coach, {}, None  # coach = optional LLM extras; see llm.py
        self.ask = ask  # optional Claude Q&A: say "braillo" then ask anything
        self._ask_armed = False
        self.nav = {"n": 0, "to": None, "target": None}  # what the website is asked to do (a page, scrolling, a control): see _go
        self.controls: list = []  # the names of the buttons, links and options on the open page (see set_page)
        self._announced: dict = {}  # page -> when it was last announced
        self.guided: Optional[bool] = None  # asked on the front page: talk them through it? (None: not answered yet)
        self._offer_n: Optional[int] = None  # what had been heard when that was asked: an answer comes after it
        self.website = False  # the website signs people in (tutor_server sets this): until it has, voice starts nothing
        self.page: Optional[str] = None  # which page of the website is open (see set_page): Braillo knows where the learner is
        self._ask_heard_n = 0
        self.names = names  # (row, col) -> sheets.Symbol for every cell of a printed sheet; None = every letter cell is a letter
        self.voice, self.wc, self.cells, self.finger, self.scan = voice, wc, cells, finger, scan
        self.mode, self.questions, self.words, self.max_tries = mode, questions, tuple(words), max_tries
        self.rng, self.lock, self.finished = rng or random.Random(), threading.RLock(), threading.Event()
        self.state, self.items, self.index, self.tries, self.hints = "idle", [], 0, 0, 0
        self.asked, self.correct, self.slips = 0, 0, {}
        self.quiz_correct: dict[str, int] = {}     # per-symbol correct count for the running quiz
        self.miss_history: dict[str, dict[str, int]] = {}
        self.quiz_number = 0
        self.adaptive_planner = adaptive_planner
        self.last_selection = {"source": "unselected", "reason": ""}
        self._dialogue_at = -1e9
        self._speech = threading.Lock()  # one voice at a time: the explore loop, commands and the phone announcer never talk over each other
        self._heard, self._last_said, self._ex = set(), "", None
        self._nags: dict = {}  # reminder text -> (when last said, how many times), cleared when the learner says something
        self._recent: collections.deque = collections.deque(maxlen=6)  # the last lines said, for Braillo's sense of the moment
        self.voice_status: Optional[dict] = None  # set by the apps from voice_check(): whether speech will actually be heard
        self.new_page: Optional[Callable] = None  # set by the apps: forgets the old page (its reading and its registration)
        self.identifies_sheets = False  # set by the apps: "next page" also works out which printed sheet is now on the desk
        self.current_sheet: Callable = lambda: "alphabet"  # set by the apps: which printed sheet the camera is reading now
        self.progress, self.progress_file = progress or Progress(), progress_file  # what the learner has learned; saved as it changes
        self.settings = acc.Settings(tones=tones)  # speech speed, pace, sounds, "I didn't catch that": see accessibility.py
        self.earcons = Earcons(getattr(voice, "voice", voice), enabled=self.settings.tones)  # (a wrapped voice keeps the real module as .voice)
        self.awareness, self.hearing = acc.Awareness(), acc.Hearing()
        self.page_ok_fn: Optional[Callable] = None  # set by the apps: can the camera see the sheet right now?
        self.heard_fn: Optional[Callable] = self._voice_heard  # what the voice module last heard (tests may replace it)
        self._watcher: Optional[threading.Thread] = None
        self.journey: Optional[Journey] = None
        self._lessons_for: Optional[str] = None  # who has had lessons already this visit (see set_mode)
        if mode == "learn":  # the guided lessons (learn.py)
            self.journey = self._new_journey()
        # the menu: who is using it, and which of learn / read / quiz they chose (see set_user, set_mode)
        self.hub = mode == "menu"
        self.hub_mode = "menu"
        self.user: Optional[dict] = None  # {"name", "kind": "google" | "guest", "profile"}
        self.select_sheet: Optional[Callable] = None  # set by the apps: read this printed sheet now (no need to recognise it)
        self.phone_ready_fn: Optional[Callable] = None  # set by the apps when a phone is the camera: is it connected?
        self.phone_code_fn: Optional[Callable] = None  # set by the apps when a phone is the camera: the code to type on it
        self.phone_backup_fn: Optional[Callable] = None  # set by the apps: show the backup link (a tunnel) in the QR code; True if it now is
        self._backup_shown = False
        self._greeted_for: Optional[str] = None
        self.muted = False  # the learner turned the microphone off with the mute key (see set_muted)
        self._epoch = 0  # bumped whenever the mode changes: the loops of the previous mode notice and stop
        self._learn_epoch = -1

    # ---- wiring ---------------------------------------------------------------------------------
    def attach(self) -> None:
        """Register the voice commands. Each one plays a tiny sound first, so a learner who cannot see the screen knows they were heard."""
        def heard(handler):
            def run() -> None:
                if self.muted:
                    return
                getting_around = handler in (self.on_back, self.on_scroll_down, self.on_scroll_up, self.on_top)
                if self.dialogue_active() and not getting_around:  # answering the sign-in page (a name must not start a lesson)
                    return
                if self.website and not self.dialogue_active():  # a button's own name on the open page wins over a command
                    heard_now = self.heard_fn() if self.heard_fn is not None else None
                    control = self._control_for(str((heard_now or {}).get("text") or ""))
                    if control is not None:
                        return self._press(control)
                if handler in self._starts_something() and self._not_ready_to_start():
                    return
                if self._ask_armed and handler is not self.on_braillo:
                    self._ask_armed = False  # another command: drop the pending question
                self._nags.clear()  # the learner is here and talking: reminders may start over
                self.earcons.play("locked")
                handler()
            return run

        for name, handler in self._commands().items():
            try:
                self.voice.register_command(name, heard(handler))
            except ValueError:  # an older voice_io that does not know this phrase: the others still work
                print(f"voice: this voice_io does not know the command {name!r}", flush=True)
        defer = getattr(self._voice_module(), "set_defer_long_utterances", None)
        if callable(defer):  # Braillo can judge a long sentence: a command word buried in one is not run blindly
            defer(self.ask is not None and self.ask.enabled)

    def _starts_something(self) -> tuple:
        """The commands that start an activity or go somewhere (the rest only talk, or change how the tutor talks)."""
        return (self.on_start, self.on_mode_learn, self.on_mode_read, self.on_mode_quiz, self.on_next_lesson, self.on_previous_lesson,
                self.on_my_page, self.on_practice, self.on_explore, self.on_found_it, self.on_next, self.on_next_page)

    def _not_ready_to_start(self) -> bool:
        """Said by voice before anyone has signed in, or before the phone (the camera) is linked: nothing starts yet; it says why. (The
        website's own buttons are not stopped: they only appear once both are done.)"""
        if not self.hub or not self.website:
            return False
        if self.user is None:  # ("start learning" on the front page, say): that begins with signing in, so off to it
            if self.page == "login":
                self.say("Let's get you signed in first. Say Google, or guest.")
            else:
                self._go("login")  # (the sign-in page then speaks for itself)
            return True
        if self.phone_ready_fn is not None and not self.phone_ready_fn():
            self.say("Let's connect your phone first, so I can see the sheet. Scan the code on the screen with its camera.")
            return True
        return False

    def _commands(self) -> dict:
        """Voice command name -> handler (also what Braillo's understanding of free speech can run, see _talk)."""
        return {"start quiz": self.on_start, "repeat": self.on_repeat, "hint": self.on_hint, "found it": self.on_found_it,
                "next": self.on_next, "next page": self.on_next_page, "explore": self.on_explore, "practice": self.on_practice,
                "learn": self.on_mode_learn, "read": self.on_mode_read, "quiz": self.on_mode_quiz, "menu": self.on_mode_menu,
                "help": self.on_help, "slower": self.on_slower, "faster": self.on_faster, "take your time": self.on_relaxed,
                "normal pace": self.on_normal, "stop": self.on_stop, "hush": self.on_hush, "braillo": self.on_braillo,
                "next lesson": self.on_next_lesson, "previous lesson": self.on_previous_lesson, "my page": self.on_my_page,
                "sign out": self.on_sign_out, "more help": self.on_more_help, "back": self.on_back, "scroll down": self.on_scroll_down, "scroll up": self.on_scroll_up,
                "top": self.on_top}

    def _voice_module(self):
        return getattr(self.voice, "voice", self.voice)  # (a wrapped voice keeps the real module as .voice)

    def say(self, text: str) -> None:
        # say() is called from almost every command handler while self.lock is held (set_mode, on_stop,
        # _check_symbol...), so a voice.speak() call that never returns -- a hung network request, or a
        # stuck audio-device write, neither of which is guaranteed to have its own timeout -- would freeze
        # every voice command and button in the whole session, not just narration. Running it in its own
        # thread and giving up after a time limit bounds that: a hang is logged and the caller carries on;
        # the abandoned thread is a daemon, so it cannot block shutdown, and _speech (below) still keeps two
        # calls from writing to the speaker at once in the normal, non-hung case.
        cuts = getattr(self._voice_module(), "interrupt_count", None)
        asked_at = cuts() if callable(cuts) else 0
        with self._speech:
            if callable(cuts) and cuts() != asked_at:
                return  # the learner cut the tutor off after this line was queued: it is out of date, drop it
            self._recent.append(text)
            done = threading.Event()

            def run() -> None:
                try:
                    self.voice.speak(text)
                finally:
                    done.set()

            limit = SPEECH_TIMEOUT + SPEECH_SECONDS_PER_WORD * len(text.split()) / max(self.settings.speech_speed, 0.5)
            threading.Thread(target=run, daemon=True, name="speak").start()
            if not done.wait(limit):
                log.warning("say() timed out after %.0fs; giving up so the session is not stuck (text=%r)", limit, text)

    def nag(self, text: str, now: Optional[float] = None) -> bool:
        """Say a reminder nobody asked for, unless it has been said too recently or too often (see NAG_GAPS): the same line over and
        over is what makes a tutor sound broken. Returns True if it was said."""
        now = time.monotonic() if now is None else now
        last, times = self._nags.get(text, (-1e9, 0))
        if times >= NAG_MAX or now - last < NAG_GAPS[min(times, len(NAG_GAPS) - 1)]:
            return False
        self._nags[text] = (now, times + 1)
        self.say(text)
        return True

    def status(self) -> dict:
        """A snapshot for displays (no lock: commands hold it while they speak, and a display must not wait for that)."""
        asking = self.state == "asking"
        try:
            prompt = self._prompt() if asking else ""
        except IndexError:  # a command finished the quiz while we were reading
            prompt = ""
        return {"mode": self.mode, "state": self.state, "prompt": prompt, "question": self.index + 1 if asking else 0,
                "total": len(self.items), "asked": self.asked, "correct": self.correct, "tries": self.tries}

    def _new_journey(self) -> Journey:
        j = Journey(_JourneyHost(self), self.progress, coach=self.coach, rng=self.rng)
        j.set_pace(self.settings.dwell_scale, self.settings.hint_scale)
        return j

    # ---- accessibility: settings, help, and speaking up when something is wrong -----------------
    def apply_settings(self, changes: dict) -> dict:
        """Change speech speed / pace / sounds / "I didn't catch that" (raises ValueError, changing nothing, if a value is not allowed)."""
        self.settings.update(changes)
        self.earcons.enabled = self.settings.tones
        if self.journey is not None:
            self.journey.set_pace(self.settings.dwell_scale, self.settings.hint_scale)
        return self.settings.to_dict()

    def set_muted(self, muted: bool) -> bool:
        """The mute key: turn the microphone off (nothing is sent to Deepgram or acted on) or back on. Returns the new state."""
        self.muted = bool(muted)
        mute = getattr(self._voice_module(), "set_muted", None)
        if callable(mute):
            mute(self.muted)
        return self.muted

    def _voice_heard(self) -> Optional[dict]:
        if self.muted:  # not listening: the pages show the microphone as off
            return None
        module = getattr(self.voice, "voice", self.voice)
        status = getattr(module, "listener_status", None)
        if status is None or getattr(module, "MOCK_MODE", False):
            return None
        return (status() or {}).get("heard")

    def _listening_state(self) -> bool:
        if self.dialogue_active() or self._ask_armed or self.muted:
            return False
        # At the choose-learn/read/quiz screen: stay quiet. Only those three commands (and help/settings)
        # should get a reply; do not say "I didn't catch that" for every other sound in the room.
        if self.hub and self.hub_mode == "menu":
            return False
        return self.state in ("learning", "exploring", "asking", "reading")

    def _activity_running(self) -> bool:
        """An activity is under way (a lesson, a quiz, reading, exploring): the sheet matters. Not at the menu, not while waiting for a sheet."""
        if self.state not in ("learning", "exploring", "asking", "reading") or self.finished.is_set():
            return False
        if self.hub and self.hub_mode == "menu":
            return False
        return not (self.journey is not None and self.journey.phase in ("idle", "await_sheet", "done", "placing"))

    def watch_once(self, now: float) -> Optional[str]:
        """One look at what is going on, speaking up if needed (called about twice a second by the watcher; testable on its own)."""
        if self._speech.locked():  # it is talking already: what it would say may be out of date by the time it finishes
            return None
        if self._ask_armed and self.heard_fn is not None:
            heard = self.heard_fn()
            if heard and int(heard.get("n", 0)) > self._ask_heard_n:
                self._ask_armed = False
                self.hearing.seen = int(heard["n"])  # so Hearing does not also nag about this utterance
                text = str(heard.get("text") or "").strip()
                if not text or text.casefold() in ASK_CANCEL:
                    self.say("Okay.")
                    return "Okay."
                self._answer_ask(strip_ask_wake(text) or text)
                return None
        page_ok = self.page_ok_fn() if self.page_ok_fn is not None else True
        msg = self.awareness.update(now, bool(page_ok), self._activity_running() and self.page_ok_fn is not None)
        if msg == acc.FOUND:
            self._nags.pop(acc.LOST, None)
            self._nags.pop(acc.STILL_LOST, None)
            self.say(msg)
            return msg
        if msg:
            return msg if self.nag(msg, now) else None
        if self.heard_fn is not None:
            heard = self.heard_fn()
            answer = self._answers_offer(heard)
            if answer is not None:  # "yes" (or "no") to "would you like me to talk you through this website?"
                self.hearing.seen, self.guided, self._offer_n = int(heard["n"]), answer, None
                self.say(LANDING_WELCOME if answer else LANDING_NO)
                return None
            if self._cant_connect(heard):  # "it won't scan": then (and only then) the backup code, or else the way in without scanning
                self.hearing.seen = int(heard["n"])
                if not self._backup_shown and self.phone_backup_fn is not None and self.phone_backup_fn():
                    from phonelink import BACKUP_LINE
                    self._backup_shown = True
                    self.say(BACKUP_LINE)
                else:  # (no backup to show, or it is up already and still no luck: typing the address in)
                    self.speak_prompt("connect_help")
                return None
            if self._wants_to_sign_in(heard):  # "Google" on the front page: the sign-in page (which then asks for itself)
                self.hearing.seen = int(heard["n"])
                self._go("login")
                return None
            control = self._names_a_control(heard)
            if control is not None:  # "privacy policy", "press dark": that button, link or option on the open page
                self.hearing.seen = int(heard["n"])
                self._press(control)
                return None
            if self._asks_to_change_level(heard):  # "change my level" (while being asked already: asked again)
                self.hearing.seen = int(heard["n"])
                if self.journey is not None and (not self.hub or self.hub_mode == "learn"):
                    self._learn("ask_level")
                else:
                    self.say("Your level belongs to the lessons. Say learn, then change my level.")
                return None
            level = self._answers_level(heard)
            if level is not None:  # "I know the alphabet", to "how much braille do you know already?" ("": not understood)
                self.hearing.seen = int(heard["n"])
                if level:
                    self._learn("on_level", level)
                elif heard.get("reason") != OVER_SPEECH:  # (a word or two of its own voice, misheard: not worth a "sorry")
                    self.say(RETRY_LEVEL)
                return None
            index = self._asks_for_lesson(heard)
            if index is not None:  # "lesson three", "go to K to O"
                self.hearing.seen = int(heard["n"])
                self.on_lesson(index)
                return None
            if self._answers_by_name(heard):  # "is this E?" with a finger on a cell: that is an answer, so check it
                self.hearing.seen = int(heard["n"])
                self.on_found_it()
                return None
            if self._wants_talk(heard):
                self.hearing.seen = int(heard["n"])  # dealt with here: Hearing must not also say "I didn't catch that"
                return self._talk(str(heard.get("text") or ""))
            msg = self.hearing.update(heard, now, self._listening_state(), self.settings.hearing_feedback)
        if msg:
            return msg if self.nag(msg, now) else None
        return None

    def _cant_connect(self, heard: Optional[dict]) -> bool:
        """On the connect-phone page, before the phone is in: they say it is not working (the QR code will not scan, say)."""
        if not self.website or self.page != "connect-phone" or not self._new_heard(heard):
            return False
        if self.phone_ready_fn is not None and self.phone_ready_fn():
            return False
        return bool(_CANT_CONNECT.search(str(heard.get("text") or "").lower()))

    def _wants_to_sign_in(self, heard: Optional[dict]) -> bool:
        """"Google", "guest" or "sign in" said on a page before the sign-in page (the front page, say): they want the sign-in page."""
        if not (self.hub and self.website) or self.user is not None or self.page in (None, "login") or not self._new_heard(heard):
            return False
        return bool(_SIGN_IN_WORDS.search(str(heard.get("text") or "").lower()))

    def _names_a_control(self, heard: Optional[dict]) -> Optional[str]:
        """Something new was said (no command) that names a control on the open page: which one."""
        if not self.website or not heard or heard.get("matched") or self.dialogue_active() or self.muted:
            return None
        n = int(heard.get("n", 0))
        if self.hearing.seen is None or n <= self.hearing.seen:
            return None
        return self._control_for(str(heard.get("text") or ""))

    def _asks_for_lesson(self, heard: Optional[dict]) -> Optional[int]:
        """Something new was said that asks for a lesson by number or name ("lesson three", "go to K to O"): which one (from 0)."""
        if not self.hub or not heard or heard.get("matched") or self.dialogue_active() or self.muted:
            return None
        n = int(heard.get("n", 0))
        if self.hearing.seen is None or n <= self.hearing.seen:
            return None
        return lesson_named(str(heard.get("text") or ""))

    def _new_heard(self, heard: Optional[dict]) -> bool:
        """Something new was said that is not a command, while nobody is signing in and the microphone is on."""
        if not heard or heard.get("matched") or self.dialogue_active() or self.muted:
            return False
        return self.hearing.seen is not None and int(heard.get("n", 0)) > self.hearing.seen

    def _answers_level(self, heard: Optional[dict]) -> Optional[str]:
        """While "how much braille do you know already?" is asked: the level said (a key of LEVELS), "" if not understood, else None."""
        if self.journey is None or self.journey.phase != "placing" or not self._new_heard(heard):
            return None
        return level_of(str(heard.get("text") or "")) or ""

    def _asks_to_change_level(self, heard: Optional[dict]) -> bool:
        return self._new_heard(heard) and bool(_CHANGE_LEVEL.search(str(heard.get("text") or "").lower()))

    def choose_level(self, level: str) -> None:
        """How much braille they know, picked on the screen (the same as saying it): the lessons start from there."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        if self.journey is None:
            raise ValueError("the level belongs to the lessons (learn mode)")
        self._learn("on_level", level)

    def _answers_by_name(self, heard: Optional[dict]) -> bool:
        """Something new was said that names a letter ("E", "is this E?", "it's the letter E") while a letter is being asked for:
        the learner is answering with their finger on it, so it counts as "found it" (the finger decides; the name is not graded)."""
        if not heard or heard.get("matched") or self.dialogue_active() or self.muted:
            return False
        n = int(heard.get("n", 0))
        if self.hearing.seen is None or n <= self.hearing.seen:
            return False
        waiting = (self.journey is not None and self.journey.target is not None) or (
            self.journey is None and self.state == "asking" and self.mode not in ("read", "word-quiz"))
        return waiting and spoken_letter(str(heard.get("text") or "")) is not None

    # ---- talking with Braillo: free speech that is not a command ------------------------------------
    def _wants_talk(self, heard: Optional[dict]) -> bool:
        """Something new was said that matched no command, and Braillo is on to work out what was meant."""
        if not heard or heard.get("matched") or self.ask is None or not self.ask.enabled or self.dialogue_active() or self.muted:
            return False
        if self.hub and self.website and self.user is None:  # before signing in, free speech is not acted on (a quiz on the sign-in page)
            return False
        if heard.get("reason") == OVER_SPEECH:  # a word or two caught while it was talking: only an answer the page expects counts
            return False
        n = int(heard.get("n", 0))
        if self.hearing.seen is None or n <= self.hearing.seen:
            return False
        return len(str(heard.get("text") or "").split()) >= TALK_MIN_WORDS

    def set_page(self, page: str, controls: Optional[list] = None, arrived: bool = False, listen: bool = False) -> None:
        """Which page of the website is open, and the names of the buttons, links and options on it (it says so whenever they change):
        Braillo knows where the learner is, and any of them can be used by voice. A page with no voice of its own is announced when
        they arrive on it (`arrived`: opened, reloaded, or its tab brought to the front), or at once when they ask (`listen`)."""
        if page not in WEBSITE_PAGES:
            return
        arrived = arrived or page != self.page
        self.page = page
        if controls is not None:
            self.controls = [" ".join(str(c).split())[:60] for c in controls if str(c).strip()][:60]
        if (arrived or listen) and page in ARRIVAL_LINES and not self.dialogue_active():
            now = time.monotonic()
            if listen or now - self._announced.get(page, -1e9) >= ARRIVAL_AGAIN_SECONDS:
                self._announced[page] = now
                if listen:
                    self.cut_speech()  # (asked for: it replaces whatever was being said)
                line = self._arrival_line(page, listen)
                if line:
                    threading.Thread(target=self.say, args=(line,), daemon=True).start()

    def _arrival_line(self, page: str, listen: bool) -> Optional[str]:
        """What to say on arriving at a page with no voice of its own. The front page asks first whether to talk them through it."""
        if page != "landing":
            return ARRIVAL_LINES[page]
        if listen:  # (the Listen key: they asked for it)
            return LANDING_WELCOME
        if self.guided is None:
            self._offer_n = self._heard_n()
            return ARRIVAL_LINES["landing"]  # "would you like me to talk you through it?"
        return LANDING_WELCOME if self.guided else None  # (back again: the whole welcome, which says what to do, not a short reminder)

    def _heard_n(self) -> int:
        heard = self.heard_fn() if self.heard_fn is not None else None
        return int((heard or {}).get("n", 0))

    def _answers_offer(self, heard: Optional[dict]) -> Optional[bool]:
        """A yes or a no, said to the front page's question (whether to talk them through it): which, or None."""
        if self._offer_n is None or self.page != "landing" or not heard or heard.get("matched"):
            return None
        if int(heard.get("n", 0)) <= self._offer_n:
            return None
        t = " ".join(re.sub(r"[^a-z' ]", " ", str(heard.get("text") or "").lower()).split())
        return True if _YES.match(t) else False if _NO.match(t) else None

    def _control_for(self, text: str) -> Optional[str]:
        """The control on the open page that `text` names ("start learning", "press the dark button"), or None."""
        if not self.controls:
            return None
        said = " ".join(re.sub(r"[^\w' ]", " ", str(text or "").lower()).split())
        said = _CONTROL_TAIL.sub("", _CONTROL_LEAD.sub("", said)).strip()
        if not said or len(said.split()) > 8:
            return None
        names = {" ".join(re.sub(r"[^\w' ]", " ", c.lower()).split()): c for c in self.controls}
        if said in names:
            return names[said]
        close = difflib.get_close_matches(said, list(names), n=1, cutoff=0.86)  # (a word misheard: "dork" for "dark")
        return names[close[0]] if close else None

    def _press(self, control: str) -> None:
        """Use a control on the open page by voice: the website presses it (see VoiceControl)."""
        self.earcons.play("locked")
        self._go("click", control)

    def talk_context(self) -> dict:
        """What is happening right now, for Braillo: where the learner is, what the camera sees, how they are getting on, and what he
        is waiting for them to do (lesson facts only: no names, images or audio). No lock: a command may hold it while it speaks."""
        ctx: dict = {"activity": self.hub_mode if self.hub else self.mode, "state": self.state}
        if self.page:
            ctx["page"], ctx["page_is_for"] = self.page, WEBSITE_PAGES[self.page]
        if self.hub:
            ctx["signed_in"] = "no" if self.user is None else ("with Google (progress saved)" if self.user["kind"] == "google"
                                                               else "as a guest (progress not saved: they have been told, so do "
                                                                    "not say it again unless they ask)")
        if self.phone_ready_fn is not None:
            ctx["phone_connected"] = bool(self.phone_ready_fn())
        try:
            ctx["sheet"] = self.current_sheet()
        except Exception:
            pass
        if self.page_ok_fn is not None:
            ctx["camera_sees_the_sheet"] = bool(self.page_ok_fn())
        ctx["finger"] = self._finger_words()
        j = self.journey
        if j is not None:
            st = j.status()
            ctx["lesson_phase"] = st["phase"]
            if st["lesson"]:
                ctx["lesson"] = f"{st['lesson']['title']} (lesson {st['lesson']['index']} of {st['lesson']['of']})"
            if st.get("word"):
                ctx["spelling_word"] = st["word"]
            if st["target"]:
                ctx.update(target_letter=st["target"].upper(), target_dots=st["target_dots"], hints_given=st["hints"],
                           wrong_tries=st["tries"], letters_left_in_this_part=st["remaining"])
        elif self.state == "asking":
            try:
                ctx["quiz_prompt"] = self._prompt()
                ctx["question"] = f"{self.index + 1} of {len(self.items)}"
            except (IndexError, StopIteration, AttributeError):
                pass
        if self.hub and self.user is not None:
            p = self.progress
            nxt = p.next_lesson(LESSONS)
            ctx["progress"] = {"letters_known_well": len(p.learned()), "letters_met": len(p.practised()), "starting_level": p.level or "not asked yet",
                               "lessons_done": [l.title for l in LESSONS if l.id in p.lessons],
                               "next_lesson": nxt.title if nxt is not None else "all done", "days_in_a_row": p.streak}
        ctx["waiting_for"] = self._waiting_for()
        recent = list(self._recent)[-3:]
        if self.user is not None and self.user.get("name"):  # (no names go to the model: the learner's is taken out of what was said)
            name = re.compile(rf"\b{re.escape(self.user['name'])}\b", re.IGNORECASE)
            recent = [name.sub("you", line) for line in recent]
        ctx["you_said_last"] = recent
        ctx["commands"] = acc.more_help_text(self.hub_mode if self.hub else None, self.mode, self.state)
        if self.controls:
            ctx["buttons_on_this_page"] = self.controls
        return ctx

    def _finger_words(self) -> str:
        """Where the learner's finger is, in words: on which letter, or that it is not on a cell, or not seen."""
        try:
            pos = self.finger()
        except Exception:
            pos = None
        if pos is None:
            return "not seen by the camera"
        cells = self.grade_cells() if callable(getattr(self, "grade_cells", None)) else self.cells
        hit = cell_at(cells or self.cells or [], *pos)
        if hit is None:
            return "on the sheet, but not on a letter"
        letter = letter_of(hit["dots"])
        dots = ", ".join(str(d) for d in sorted(hit["dots"]))
        return f"on the letter {letter.upper()} (dots {dots})" if letter else f"on a cell with dots {dots}"

    def _waiting_for(self) -> str:
        """What the tutor expects the learner to do next, in a sentence."""
        if self.hub and self.website and self.user is None:
            if self.page == "login":
                return "them to sign in: say Google, or guest"
            if self.page in ("landing", "privacy", "terms", "accessibility"):
                return ("them to say start learning (if they say they want to begin, start is the action: do it, do not tell them "
                    "to say it); the next page then speaks for itself, so do not describe it or its choices")
            return "them to sign in first: saying Google, or guest, takes them to the sign-in page"
        if self.hub and self.phone_ready_fn is not None and not self.phone_ready_fn():
            return "them to connect their phone: scan the QR code on the screen with the phone's camera, then tap Start"
        if self.hub and self.hub_mode == "menu":
            return "them to choose: learn, read or quiz (or say my page to hear how they are doing)"
        j = self.journey
        if j is not None:
            st = j.status()
            if st["phase"] == "placing":
                return "them to say how much braille they know already: new, some, alphabet, or numbers"
            if st["phase"] == "await_sheet" and st["lesson"]:
                return f"them to put the {st['lesson']['sheet']} sheet under the camera, flat and all in view"
            if st["phase"] == "word" and st["target"]:
                return f"them to find the letter {st['target'].upper()} (spelling the word {st.get('word')}) and rest a finger on it"
            if st["target"]:
                return (f"them to find the letter {st['target'].upper()} (dots {', '.join(map(str, st['target_dots']))}) on the sheet and "
                        "rest a finger on it, or point and say this one")
            if st["phase"] in ("recap", "done"):
                return "them to say next for the next lesson, practice for a review, explore, or home"
            if st["phase"] == "explore":
                return "them to rest a finger on any letter, and you say what it is"
        if self.state == "asking":
            return "them to point at the letter asked for and say this one, or rest a finger on it"
        if self.state == "reading":
            return "them to rest a finger on a word on the words sheet, and you read it aloud"
        return "them to say what they would like to do"

    def _talk(self, text: str) -> Optional[str]:
        """Let Braillo work out what free speech meant: run the command it names and/or say its reply. Falls back to the plain
        "I didn't catch that" when Braillo cannot answer just now."""
        reply = self.ask.respond(text, self.talk_context())
        if reply is None:
            msg = acc.NOT_UNDERSTOOD if self._listening_state() and self.settings.hearing_feedback else None
            return msg if msg and self.nag(msg) else None
        return self._act_on(reply)

    def _act_on(self, reply: dict) -> Optional[str]:
        """Do what Braillo made of what was said: say his line, and run the command it names (if it may run now)."""
        self._nags.clear()  # they are talking to us
        action, line = reply["action"], reply["say"]
        if line:
            self.say(line)
            if action in ("repeat", "hint", "help"):
                return line  # those only talk, and Braillo has just answered: running one too would say it all twice
        if action == "lesson":  # "let's do K to O": the lesson by its number
            if not self._not_ready_to_start():
                self.on_lesson(reply.get("lesson", 1) - 1)
            return line or None
        handler = self._commands().get("start quiz" if action == "start" else action)
        if handler is not None and action not in ("none", "braillo", "hush") and not (
                handler in self._starts_something() and self._not_ready_to_start()):
            handler()
        return line or None

    def cut_speech(self) -> None:
        """Stop the line being spoken now (the voice module cuts its playback; lines queued behind it are dropped, see say())."""
        cut = getattr(self._voice_module(), "interrupt", None)
        if callable(cut):
            cut()

    def on_hush(self) -> None:
        """"Quiet" / "wait" / "stop" while it talks: the line is cut off (a spoken one already has been, by the voice module). If
        more was said after the word ("wait, which dots does D have?"), that is worked out like any other free speech."""
        self.cut_speech()
        heard = self.heard_fn() if self.heard_fn is not None else None
        text = str((heard or {}).get("text") or "")
        rest = re.sub(r"^\W*(?:be quiet|stop talking|hold on|hang on|quiet|hush|shh|shush|wait|pause|enough|stop)\b\W*", "", text,
                      flags=re.IGNORECASE)
        if len(rest.split()) >= 3 and self.ask is not None and self.ask.enabled:
            self._talk(rest)

    def start_watching(self) -> None:
        """Start the background watcher (once): it tells the learner when the camera loses the sheet, and when they were not understood."""
        if self._watcher is not None and self._watcher.is_alive():
            return

        def loop() -> None:
            while True:
                time.sleep(0.5)
                try:
                    self.watch_once(time.monotonic())
                except Exception:  # one bad moment must not stop the watcher
                    traceback.print_exc()
        self._watcher = threading.Thread(target=loop, daemon=True, name="watcher")
        self._watcher.start()

    def on_braillo(self) -> None:
        """Wake word for Claude Q&A: answer a question in the same utterance, or wait for the next one."""
        heard = self.heard_fn() if self.heard_fn is not None else None
        text = str((heard or {}).get("text") or "")
        question = strip_ask_wake(text)
        if looks_like_question(question):
            self._ask_armed = False
            self._answer_ask(question)
            return
        if self.website and self.page == "landing" and not self.guided:  # (on the front page, "Braillo" alone: talk me through it)
            self.guided, self._offer_n = True, None
            return self.say(LANDING_WELCOME)
        self._ask_heard_n = int((heard or {}).get("n", 0))
        self._ask_armed = True
        self.say("What's your question?")

    def _answer_ask(self, question: str) -> None:
        """Fetch a short Claude answer and speak it. Network wait is outside the speech lock."""
        q = " ".join(str(question or "").split()).strip()
        if not q:
            self.say("I did not catch a question. Say braillo, then ask again.")
            return
        if self.ask is None or not self.ask.enabled:
            self.say("I cannot answer questions right now. Add an Anthropic API key to the .env file to turn Braillo on.")
            return
        if self.website and self.hub and callable(getattr(self.ask, "respond", None)):
            # On the website, "Braillo, let's begin" should begin, not explain how to: what was said may be something to do as well
            # as something to answer. (Only a plain question with nothing to do gets the longer answer below.)
            reply = self.ask.respond(q, self.talk_context())
            if reply is not None and (reply["action"] != "none" or reply["say"]):
                self._act_on(reply)
                return
        answer = self.ask.answer(q, context=self.talk_context())
        if not answer:
            self.say("Sorry, I could not get an answer just then. Try again in a moment.")
            return
        self.say(answer)

    def on_help(self) -> None:
        """"Help": in a sentence or so, the few things that matter right now (on a page of the website, its main buttons). Short on
        purpose: someone who asked must not be left listening for a minute. "More options" gives everything."""
        if self.website and self.page in PAGE_HELP and (self.user is None or self.page != "login"):
            return self.say(f"{PAGE_HELP[self.page]} Say more options for everything else.")
        if self.website and self.page in ARRIVAL_LINES:  # (a page of the website, not an activity: its own controls are what matter)
            main = [c for c in self.controls if c.lower() not in UTILITY_CONTROLS][:HELP_CONTROLS]
            if not main:
                return self.say("Say go back to return, or more options for everything else.")
            return self.say(f"Here you can say {_or_list(main)}. Say more options for everything else.")
        self.say(acc.help_text(self.hub_mode if self.hub else None, self.mode, self.state))

    def on_more_help(self) -> None:
        """"More options": everything that can be said here, the page's buttons and links included."""
        parts = []
        if self.website and self.controls:
            parts.append(f"On this page you can say {_or_list(self.controls)}. You can also say go back, scroll down, or scroll up.")
        if self.website and self.page in PAGE_HELP:
            parts.insert(0, PAGE_HELP[self.page])
        elif not (self.website and self.page in ARRIVAL_LINES):
            parts.append(acc.more_help_text(self.hub_mode if self.hub else None, self.mode, self.state))
        self.say(" ".join(parts) or acc.more_help_text(self.hub_mode if self.hub else None, self.mode, self.state))

    def on_slower(self) -> None:
        s = self.settings
        if s.speech_speed <= audio_speed.MIN_SPEED + 0.001:
            return self.say("That's as slowly as I can go.")
        self.apply_settings({"speech_speed": max(audio_speed.MIN_SPEED, round(s.speech_speed - acc.SLOWER_FASTER_STEP, 2))})
        self.say("Okay, I'll speak more slowly.")  # (already at the new speed)

    def on_faster(self) -> None:
        s = self.settings
        if s.speech_speed >= audio_speed.MAX_SPEED - 0.001:
            return self.say("That's as fast as I go.")
        self.apply_settings({"speech_speed": min(audio_speed.MAX_SPEED, round(s.speech_speed + acc.SLOWER_FASTER_STEP, 2))})
        self.say("Okay, a bit faster.")

    def on_relaxed(self) -> None:
        self.apply_settings({"pace": "relaxed"})
        self.say("Okay, take your time. I'll wait longer before offering help, and you can rest your finger a little longer to answer.")

    def on_normal(self) -> None:
        self.apply_settings({"pace": "normal"})
        self.say("Back to the normal pace.")

    def learning_status(self) -> Optional[dict]:
        """Where the learner is in the lessons (None unless --mode learn)."""
        return None if self.journey is None else self.journey.status()

    def save_progress(self) -> None:
        """Keep what was learned (called as it changes; a small atomic write)."""
        if self.progress_file is not None:
            try:
                self.progress.save(self.progress_file)
            except OSError as e:
                print(f"could not save progress to {self.progress_file}: {e}", flush=True)

    def _learn(self, command: str, *args) -> None:
        """Run a voice command in learn mode: the lesson engine handles it, and its clock loop is kept running."""
        with self.lock:
            if command in ("on_start", "go_to_lesson"):
                self.finished.clear()  # "start" after "stop" begins again
                self.state = "learning"
                alive = getattr(self, "_learn_thread", None) is not None and self._learn_thread.is_alive()
                if not alive or self._learn_epoch != self._epoch:
                    self._learn_epoch = self._epoch
                    self._learn_thread = threading.Thread(target=self._learn_loop, args=(self._epoch,), daemon=True, name="learn")
                    self._learn_thread.start()
            getattr(self.journey, command)(*args)

    def _learn_loop(self, epoch: int) -> None:
        while not self.finished.is_set() and epoch == self._epoch:
            with self.lock:
                if epoch != self._epoch or self.journey is None:
                    return
                try:
                    self.journey.tick()
                except Exception:  # keep teaching: one bad moment must not silence the tutor
                    traceback.print_exc()
            time.sleep(0.1)

    # ---- the menu: who is here, and what they chose -----------------------------------------------
    def set_user(self, name, kind: str, profile: Optional[str] = None) -> None:
        """Who is using the tutor. A "google" learner's progress is kept (in a file named for their account, and by the website in their
        account); a "guest" gets fresh, unsaved progress: nothing they do is written anywhere. (A guest's `profile` is the website visit's
        id: the same guest after a page reload keeps their lesson, and the next guest, another visit, starts afresh.)"""
        name = clean_name(name)
        key = f"{kind}:{profile or name}"
        with self.lock:
            if self.user and self.user.get("key") == key and self.user["name"] == name:
                return
            saves = kind == "google" and bool(profile)
            self.user = {"name": name, "kind": "google" if saves else "guest", "profile": profile if saves else None, "key": key}
            if saves:
                self.progress_file = progress_path(profile)
                self.progress = Progress.load(self.progress_file)
            else:
                self.progress_file, self.progress = None, Progress()  # not saved, not even on this computer
            if self.journey is not None:
                self.journey.progress = self.progress
            self._greeted_for = None
            self._dialogue_at = -1e9
        threading.Thread(target=self._maybe_greet, daemon=True, name="greet").start()

    def on_phone_ready(self) -> None:
        """The phone is linked (and its sound is on): the moment to speak to the user."""
        threading.Thread(target=self._maybe_greet, daemon=True, name="greet").start()

    def _maybe_greet(self) -> None:
        """"What do you want to do today, <name>?" once, when we know who it is and (if a phone is the camera) the phone is linked."""
        # Polled every camera frame (see TutorRuntime._loop) so a hiccup in the one-shot phone-ready callback
        # can't permanently skip the greeting; this fast, lock-free check keeps that polling from contending
        # for self.lock ~15 times a second once greeted, which is the overwhelmingly common case.
        if not self.hub or self.user is None or self.hub_mode != "menu" or self._greeted_for == self.user["key"]:
            return
        with self.lock:
            if not self.hub or self.user is None or self.hub_mode != "menu" or self._greeted_for == self.user["key"]:
                return
            if self.phone_ready_fn is not None and not self.phone_ready_fn():
                return
            self._greeted_for = self.user["key"]
            self.state = "menu"
            self.say(greeting(self.user["name"]))

    def hub_status(self) -> Optional[dict]:
        """For displays: None unless this is the menu-driven tutor."""
        if not self.hub:
            return None
        u = self.user
        nxt = self.progress.next_lesson(LESSONS) if u is not None else None  # what "Continue" on the home screen starts
        return {"mode": self.hub_mode, "modes": {k: {"title": v["title"], "sheet": v["sheet"]} for k, v in HUB_MODES.items()},
                "user": None if u is None else {"name": u["name"], "kind": u["kind"], "saves": u["kind"] == "google"},
                "greeted": self._greeted_for is not None,
                "next_lesson": None if nxt is None else {"id": nxt.id, "title": nxt.title, "index": LESSONS.index(nxt) + 1,
                                                         "of": len(LESSONS), "letters": [l.upper() for l in nxt.letters]}}

    def speak_prompt(self, name: str, who=None) -> None:
        text = next_prompt(name) if name in PROMPTS else None
        if text:
            code = self.phone_code_fn() if self.phone_code_fn is not None else ""
            spelled = " ".join(code) if code else "shown on the screen"  # one digit at a time: "6 7 9 9 7 0", not a large number
            if "{progress}" in text or "{letters}" in text:  # (your page: the numbers come from the tutor's own record of you)
                p = self.progress
                text = text.replace("{progress}", progress_line(p.summary())).replace("{letters}", letters_line(p.summary(), p.learned(), p.practised()))
            self.say(text.replace("{name}", clean_name(who)).replace("{code}", spelled))

    def dialogue(self, open_: bool) -> None:
        """The sign-in page is talking with the user (or has finished). While it is: no "I did not catch that", and no voice command runs."""
        self._dialogue_at = time.time() if open_ else -1e9

    def dialogue_active(self) -> bool:
        return time.time() - self._dialogue_at < DIALOGUE_SECONDS

    def is_speaking(self) -> bool:
        return self._speech.locked()

    def on_mode(self, mode: str) -> None:
        """"Learn", "read", "quiz" or "menu" (by voice or from the website)."""
        if not self.hub:
            return self.say("Choosing a mode is part of the menu: start the tutor without --mode, or use --mode learn, explore or another.")
        self.set_mode(mode)

    # ---- getting around by voice, from anywhere: lessons, your page, signing out --------------------------------
    def on_lesson(self, index: int) -> None:
        """"Lesson three", "K to O": that lesson, now (learn mode starts first if it is not on)."""
        if not self.hub:
            return self.say("Choosing a lesson is part of the menu: start the tutor without --mode.")
        if self.hub_mode != "learn":
            return self.set_mode("learn", lesson=index)
        self._learn("go_to_lesson", index)

    def on_next_lesson(self) -> None:
        self._lesson_step(+1)

    def on_previous_lesson(self) -> None:
        self._lesson_step(-1)

    def _lesson_step(self, step: int) -> None:
        here = self.journey.lesson_index() if (self.hub_mode == "learn" and self.journey is not None) else None
        if here is None:  # not in a lesson: "next" is the one due, "previous" the one before it
            due = self.progress.next_lesson(LESSONS)
            base = LESSONS.index(due) if due is not None else len(LESSONS)
            index = base if step > 0 else base - 1
        else:
            index = here + step
        if index < 0:
            return self.say("This is the first lesson.")
        if index >= len(LESSONS):
            return self.say("That was the last lesson. Say practice for a review, or explore to feel the letters freely.")
        self.on_lesson(index)

    def _go(self, to: str, target: Optional[str] = None) -> None:
        """Ask the website to do something ("login", "account", "signout", "back", "home", "scroll-down", "scroll-up", "top", or "click" a
        control named `target`): it watches `nav` in the state."""
        self.nav = {"n": self.nav["n"] + 1, "to": to, "target": target}

    def on_back(self) -> None:
        self._go("back")

    def on_scroll_down(self) -> None:
        self._go("scroll-down")

    def on_scroll_up(self) -> None:
        self._go("scroll-up")

    def on_top(self) -> None:
        self._go("top")

    def _leave_quietly(self) -> None:
        """Stop whatever is running, back at the menu, without the menu's spoken prompt (another page is about to speak)."""
        with self.lock:
            self._epoch += 1
            self.journey, self._ex, self.items, self.index = None, None, [], 0
            self.mode, self.hub_mode, self.state = "menu", "menu", "menu"

    def on_my_page(self) -> None:
        """"My page": the learner's own page (it reads out how they are getting on)."""
        if not self.hub:
            return
        self._leave_quietly()
        self._go("account")

    def on_sign_out(self) -> None:
        if not self.hub:
            return
        self._leave_quietly()
        self.say("Signing you out. See you soon.")
        self._go("signout")

    def on_mode_learn(self) -> None:
        self.on_mode("learn")

    def on_mode_read(self) -> None:
        if self.hub and self.hub_mode == "read" and self.state == "reading":
            return self._read_word_now()  # already in read: "read" means read this word, not the intro again
        self.on_mode("read")

    def on_mode_quiz(self) -> None:
        self.on_mode("quiz")

    def on_mode_menu(self) -> None:
        """"Home" / "menu": back to choosing what to do (before signing in, the website's front page)."""
        if self.hub and self.website:
            if self.user is None:
                return self._go("home")
            self._go("home")
        return self._on_mode_menu()

    def _on_mode_menu(self) -> None:
        self.on_mode("menu")

    def _read_word_now(self) -> None:
        """Speak the word under the finger right now (read mode), or say how to place the finger."""
        pos = self.finger()
        if pos is None:
            return self.say("I can't see your finger. Rest it on a word.")
        self._read_word(pos)

    def set_mode(self, mode: str, lesson: Optional[int] = None) -> None:
        """Switch to learn, read or quiz (each on its own printed sheet), or back to the menu. Whatever was running stops. With
        `lesson` (learn only), that lesson starts instead of the next one due."""
        if mode != "menu" and mode not in HUB_MODES:
            raise ValueError(f"unknown mode {mode!r}; choose menu, {', '.join(HUB_MODES)}")
        with self.lock:
            self._epoch += 1  # the previous mode's loops see this and end
            self.journey, self._ex, self.items, self.index, self.state = None, None, [], 0, "idle"
            self.finished.clear()
            if mode == "menu":
                self.mode, self.hub_mode, self.state = "menu", "menu", "menu"
                return self.say(f"Okay. {MENU_LINE_SHORT}")
            cfg = HUB_MODES[mode]
            self.mode, self.hub_mode = cfg["mode"], mode
            if self.select_sheet is not None:
                self.select_sheet(cfg["sheet"])  # switches what the camera reads for, and (through set_sheet) which cells this session knows
            if mode == "learn":
                self.journey = self._new_journey()
                who = (self.user or {}).get("key")
                if who is not None and self._lessons_for == who:
                    self.journey.resume_visit()  # (they have had lessons already this visit: no "welcome back", no new session)
                self._lessons_for = who
                return self._learn("go_to_lesson", lesson) if lesson is not None else self._learn("on_start")
            self.contracted = False  # the printed sheets are plain letters and words
            self._start_dwell_loop()
            if mode == "read":
                self.state = "reading"
                return self.say(f"Read mode. Put the words sheet, the one with {feel_for('words')}, in front of the camera. Rest a finger on "
                                "a word and I will read it aloud. "
                                "You can also say read or found it. Say menu to choose something else.")
            self.questions, self.state = 8, "idle"
            self.say(f"Quiz time. Put the look-alikes sheet, the one with {feel_for('lookalikes')}, in front of the camera. I will name a "
                     "letter: find it and rest your finger on it. Say menu to stop.")
            self.on_start()

    def _start_dwell_loop(self) -> None:
        threading.Thread(target=self._dwell_loop, args=(self._epoch,), daemon=True, name="dwell").start()

    def _dwell_loop(self, epoch: int) -> None:
        """In read and quiz mode, resting a finger on a spot is the answer (as in the lessons): a quiz answer, or a word to read out."""
        dwell = Dwell(seconds=DWELL_SECONDS * self.settings.dwell_scale)
        lost_since = None
        last_nag = 0.0
        while epoch == self._epoch:
            now = time.monotonic()
            grade = None
            nag = False
            with self.lock:
                if epoch != self._epoch:
                    return
                try:
                    pos = dwell.update(self.finger(), now)
                    waiting = ((self.mode == "letters" and self.state == "asking")
                               or (self.mode == "read" and self.state == "reading"))
                    if pos is not None:
                        if self.mode == "letters" and self.state == "asking":
                            grade = ("symbol", pos)
                        elif self.mode == "read" and self.state == "reading":
                            grade = ("word", pos)
                    if waiting and self.finger() is None:
                        if lost_since is None:
                            lost_since = now
                        elif now - lost_since >= NO_FINGER_SECONDS and now - last_nag >= NO_FINGER_AGAIN:
                            nag, last_nag = True, now
                    else:
                        lost_since = None
                except Exception:
                    traceback.print_exc()
            try:
                if grade is not None:
                    kind, pos = grade
                    with self.lock:
                        if kind == "symbol" and self.mode == "letters" and self.state == "asking":
                            self._check_symbol(pos)
                        elif kind == "word" and self.mode == "read" and self.state == "reading":
                            self._read_word(pos)
                elif nag and not self.is_speaking():
                    self.nag("I can't see your finger. Rest it on the page.")
            except Exception:
                traceback.print_exc()
            time.sleep(0.1)

    # ---- commands -------------------------------------------------------------------------------
    def on_start(self) -> None:
        if self.journey is not None:
            return self._learn("on_start")
        with self.lock:
            self._merge_slips_into_history()
            self.asked = self.correct = self.index = 0
            self.slips = {}
            self.quiz_number += 1
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
        if self.website and self.page in ARRIVAL_LINES and (self.hub_mode == "menu" or not self.hub):
            return self.say(LANDING_WELCOME if self.page == "landing" else ARRIVAL_LINES[self.page])  # (that page's welcome, again)
        if self.journey is not None:
            return self._learn("on_repeat")
        with self.lock:
            if self.state == "exploring":
                self.say(self._last_said or "Rest a finger on a cell and I will tell you what it is.")
            elif self.state == "asking":
                self.say(self._prompt())
            elif self.state == "reading":
                if self._last_said.startswith("The word is"):
                    return self.say(self._last_said)
                return self._read_word_now()
            else:
                self.say("Say start quiz to begin.")

    def on_hint(self) -> None:
        if self.journey is not None:
            return self._learn("on_hint")
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
        if self.journey is not None:
            return self._learn("on_found_it")
        if self.hub and self.hub_mode == "menu":
            return self.say(MENU_LINE_SHORT)  # ("this one" at the menu: nothing is being looked for yet)
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
        if self.journey is not None:
            return self._learn("on_next")
        with self.lock:
            if self.state == "exploring":
                return self.say("Just move your finger to another cell. Say stop when you are done.")
            if self.state != "asking":
                return self.say("Say start quiz to begin.")
            self._record(False)
            self._advance()

    def on_stop(self) -> None:
        if self.journey is not None:
            self._learn("on_stop")
            if self.hub:
                self.set_mode("menu")
            return
        with self.lock:
            if self.hub and self.hub_mode == "menu":
                return self.say(MENU_LINE_SHORT)
            self._finish(stop=True)  # "stop" always ends it (with the adaptive planner on, a plain _finish() only ends the ROUND and stays in the mode)

    def on_explore(self) -> None:
        """"Explore": free exploring in learn mode."""
        if self.journey is not None:
            return self._learn("on_explore")
        self.say("Free exploring is part of the guided lessons: start with --mode learn, or use --mode explore." if self.mode != "explore"
                 else "You are already exploring. Rest a finger on any cell.")

    def on_practice(self) -> None:
        """"Practice": an adaptive review of the letters that need it, in learn mode."""
        if self.journey is not None:
            return self._learn("on_practice")
        self.say("Practice is part of the guided lessons: start with --mode learn.")

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

    def sheet_news(self, text: str) -> None:
        """What the camera made of the page in view, after "next page" or a lesson asking for its sheet. While a lesson waits for its
        sheet, the lesson says it in its own words (which sheet that is, and which it needs), so two voices never talk over each
        other; otherwise it is said as it is."""
        j = self.journey
        if self.hub and (self.hub_mode == "menu" or (j is not None and j.phase in ("placing", "idle"))):
            return  # (nothing chosen yet, or still saying how much braille they know: the sheet is not their business yet)
        if j is not None and j.phase == "await_sheet":
            seen = None if text.startswith("I couldn't tell") else self.current_sheet()
            return self._learn("sheet_seen", seen)
        self.say(text)

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
        """The printed sheet's name for this cell. A finger covering a dot must not rename it to the live misread."""
        sym = self._symbol(cell)
        if sym is not None:
            return sym
        letter = letter_of(cell["dots"])
        return letter_symbol(letter) if letter else None

    def describe(self, cell: dict, full: bool = True) -> str:
        """What a blind learner needs to hear for a cell: its name, the dot numbers and (`full`) where those dots sit."""
        sym = self._explore_symbol(cell)
        dots = sorted(sym.dots if sym is not None else cell["dots"])
        if not dots:
            return "The camera sees no raised dots in this cell."
        lead = (sym.spoken[0].upper() + sym.spoken[1:]) if sym else "A cell I don't recognise"
        text = f"{lead}. {'Dot' if len(dots) == 1 else 'Dots'} {_and([str(d) for d in dots])}"
        if full:
            text += ": " + _and([self._POSITIONS[d] for d in dots])
        return text + "."

    def _explore_announce(self, pos, now: float, force: bool = False, full: Optional[bool] = None) -> None:
        ex = self._ex
        cell = cell_at(self._explore_cells(), *pos)
        if cell is None:
            if (force or now - ex["since"] >= EXPLORE_OFF_CELL_SECONDS) and not ex["off_told"]:
                ex["off_told"] = True
                self._last_said = "I don't see any braille there."
                if force:  # asked for ("found it"): always answered
                    self.say(self._last_said)
                else:
                    self.nag(self._last_said)
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
                self.nag("I can't see your finger.")
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
        candidates = self._candidate_targets()
        count = min(self.questions, len(candidates))
        history = self._selection_history(include_current=False)
        if self.adaptive_planner is not None and count:
            planned, reason = self.adaptive_planner.take_ready(candidates, count, self.mode, history)
            if planned is not None:
                self.last_selection = {"source": "claude", "reason": self.adaptive_planner.last_response.get("reason", "")}
                log.info("Adaptive target selection used Claude: %s", self.last_selection["reason"])
                self.adaptive_planner.prefetch(candidates, count, self.mode, history)
                return planned
            self.adaptive_planner.prefetch(candidates, count, self.mode, history)
            self.last_selection = {"source": "fallback", "reason": reason}
            log.warning("Adaptive target selection fallback: %s", reason)
        return self._fallback_items(candidates, count)

    def _candidate_targets(self) -> list[str]:
        if self.mode == "word-quiz":
            return list(dict.fromkeys(self.words))
        return sorted({s.key for s in map(self._symbol, self.cells) if s is not None})

    def _fallback_items(self, candidates: list[str], count: int) -> list:
        """Miss-weighted selection (LetterDrill) when no Claude plan is ready.

        word-quiz: unchanged — ordered slice of self.words.
        letters:   weighted draw via (1 + 2*misses)/(1 + correct); falls back to
                   uniform random when the pool has no history (first quiz).
        Returns list[str] of exactly min(count, len(candidates)) symbols,
        same shape as the former rng.sample() call.
        """
        if self.mode == "word-quiz":
            return list(self.words)[:count]
        stats = self._selection_history(include_current=False)
        # exclude_first: last item from the previous quiz so we don't open with
        # the same letter twice in a row; self.items still holds the old list here.
        exclude_first = self.items[-1] if self.items else None
        return _drill.pick(candidates, count, stats, self.rng, exclude_first=exclude_first)

    def _merge_slips_into_history(self) -> None:
        for key, misses in self.slips.items():
            record = self.miss_history.setdefault(key, {"misses": 0, "correct": 0, "last_missed_quiz": 0})
            record["misses"] += misses
            record["last_missed_quiz"] = self.quiz_number
        for key, correct in self.quiz_correct.items():
            record = self.miss_history.setdefault(key, {"misses": 0, "correct": 0, "last_missed_quiz": 0})
            record["correct"] = record.get("correct", 0) + correct
        self.quiz_correct = {}      # reset for the next quiz

    def _selection_history(self, include_current: bool) -> dict[str, dict[str, int]]:
        history = {key: dict(value) for key, value in self.miss_history.items()}
        if include_current:
            for key, misses in self.slips.items():
                record = history.setdefault(key, {"misses": 0, "correct": 0, "last_missed_quiz": 0})
                record["misses"] += misses
                record["last_missed_quiz"] = self.quiz_number
            for key, correct in self.quiz_correct.items():
                record = history.setdefault(key, {"misses": 0, "correct": 0, "last_missed_quiz": 0})
                record["correct"] = record.get("correct", 0) + correct
        return history

    def _prefetch_adaptive_selection(self) -> None:
        if self.adaptive_planner is None or self.mode == "read":
            return
        candidates = self._candidate_targets()
        self.adaptive_planner.prefetch(
            candidates, min(self.questions, len(candidates)), self.mode,
            self._selection_history(include_current=True),
        )

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
        item = self.items[self.index]
        if ok:
            self.quiz_correct[item] = self.quiz_correct.get(item, 0) + 1
        elif slip:
            self.slips[item] = self.slips.get(item, 0) + 1
        if self._is_letter(item):  # the quiz counts towards what the lessons know: a letter missed here is taught again in the next lesson
            self.progress.record(item, ok, hints=self.hints, tries=self.tries + (1 if ok else 0))
            self.save_progress()

    def _is_letter(self, key) -> bool:
        return self.mode == "letters" and isinstance(key, str) and len(key) == 1 and key in LETTER_DOTS

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.items):
            self._finish()
        else:
            self._ask()

    def _finish(self, stop: bool = False) -> None:
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
        elif not (self.hub and self.hub_mode != "menu"):  # (in the menu-driven app they are only going back to the menu: no goodbye)
            self.say("Goodbye.")
        self._prefetch_adaptive_selection()
        if self.adaptive_planner is not None and not stop:
            self.state = "idle"
            return self.say("Round complete. Say start quiz when you're ready for the next round.")
        self.state = "done"
        self.finished.set()
        if self.hub and self.hub_mode != "menu":
            self.set_mode("menu")  # back to the choice: learn, read or quiz again

    def _check_symbol(self, pos) -> None:
        target = self.items[self.index]
        grade = getattr(self, "grade_cells", None)
        cells = grade() if callable(grade) else self.cells
        cell = cell_at(cells or self.cells, *pos)
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
        if found is not None and self._is_letter(target) and self._is_letter(found.key):
            self.progress.record_confusion(found.key, target)  # (the lessons go over this pair again)
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
        """The word under the finger, from a fresh detection ("" if there is none).

        Hit-testing uses the same drawn boxes as the green ring (grade_cells / stable overlay) so resting
        inside a letter square counts. On a known printed sheet (plain mode), the sheet's own letters are
        passed as a fallback so a cell the finger is covering still contributes its printed letter --
        otherwise a finger over the 'c' in "cat" turns the reading into "?at" and the tutor stumbles
        instead of just reading "cat"."""
        if self.scan is None:
            return ""
        printed = None
        if not self.contracted and self.cells:
            printed = {(c["row"], c["col"]): letter_of(c["dots"]) for c in self.cells}
        grade = getattr(self, "grade_cells", None)
        cells = grade() if callable(grade) else None
        if not cells:
            cells = self.scan()
        return reader.word_at(cells, *pos, decode=self.contracted, printed=printed) or ""

    def _read_word(self, pos) -> None:
        if self.scan is None:
            return self.say("I can't scan the page right now.")
        raw = self._word_at(pos)
        if not raw:
            return self.say("I don't see a word there.")
        result = self.wc.correct_word_read_mode(raw, redetect=lambda: self._word_at(pos))
        if result.source in ("exact", "redetect"):
            self._last_said = f"The word is {result.word}."
            self.say(self._last_said)
        else:  # the backend never guesses; say what was read, letter by letter -- "unknown" for any cell that isn't a plain letter
            spoken = ", ".join("unknown" if ch == "?" else ch for ch in result.raw)
            self._last_said = f"I couldn't read that clearly. I think it says {spoken}."
            self.say(self._last_said)

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

KEYS = {"s": "on_start", "r": "on_repeat", "h": "on_hint", "f": "on_found_it", "n": "on_next", "p": "on_next_page", "e": "on_explore", "v": "on_practice", "x": "on_stop"}


class CameraFeed:
    """Newest camera frame + page registration + a 'finger'; shared by the window app and the web server.

    The finger is the tracked fingertip (fingertip.py) when `track_finger` is on; a click on the video or a position set in page
    mm stands in for it for POSTED_FINGER_SECONDS (or until cleared), so a demo can always fall back to pointing with the mouse."""

    def __init__(self, page_src, overlay_cells: Optional[list] = None, labels: Optional[dict] = None, detector=None,
                 always_reading: bool = False, track_finger: bool = False, observe_sheet: Optional[list] = None,
                 show_reading: bool = True, known_sheets: Optional[dict] = None, sheet_name: str = "alphabet",
                 finder: Optional[Callable] = None):
        self.page_src, self.overlay_cells, self.labels = page_src, overlay_cells, labels
        self.detector, self.always_reading = detector, always_reading  # detector: optional live reading; needs no page
        self.frame, self.H, self.frames = None, None, 0
        self.message, self.page_ok = "waiting for the camera", False
        self.finger_px: Optional[tuple] = None
        self.finger_mm: Optional[tuple] = None
        self.finger_posted = 0.0  # when, so a click stands in for the fingertip only while it is fresh
        # finder: how a fingertip is found in a frame (the apps pass best_finder(), MediaPipe's hand model; default the skin-colour one)
        self.tracker = (FingerTracker(find=finder) if finder is not None else FingerTracker()) if track_finger else None
        self.observe_sheet = observe_sheet  # a known sheet's cells: scan() then reports the dots the camera SEES on them
        # ...and (show_reading) that reading runs continuously, is locked in cell by cell as it proves steady, and is drawn on the
        # video as a box per cell with the detected dots and letter: green = locked in, amber = still reading
        self.reader = _Detector(0.15, "auto", sheet=observe_sheet) if (observe_sheet is not None and show_reading) else None
        self.locker, self.stable, self._seen = CellLocker(), [], 0
        self._remember_sheet(observe_sheet)
        # "next page": which of these sheets is on the desk now? (name -> Sheet); on_sheet(Sheet) and announce(text) are set by the app
        self.known_sheets, self.sheet_name = known_sheets, sheet_name
        self.on_sheet: Optional[Callable] = None
        self.announce: Callable = lambda text: None
        self.identify_until, self._identifying, self._last_identify = 0.0, False, 0.0
        self._changed, self._thumb0, self._identify_from, self._lost_frames = False, None, 0.0, 0

    def _remember_sheet(self, cells: Optional[list]) -> None:
        """Pin every cell's name to the printed sheet (the photos / first layout). Live frames only refine where the boxes sit."""
        self.locker.reset()
        if cells:
            self.locker.prelock(cells)
            self.stable = self.locker.result()
        else:
            self.stable = []

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

    def select_sheet(self, name: str) -> None:
        """Read `name` from now on, because the learner chose an activity that uses it (nothing to recognise: they were told to put it
        down). Forgets the old page's reading; the tutor session is told which cells it now has."""
        sheet = self.known_sheets[name]
        self.identify_until = 0.0
        if name != self.sheet_name or self.observe_sheet is not sheet.cells:
            self.sheet_name, self.observe_sheet = name, sheet.cells
            if self.reader is not None:
                self.reader.sheet = sheet.cells
        self._remember_sheet(sheet.cells)
        # Fresh finger for the new sheet: a held tip from Learn must not grade Read/Quiz for seconds.
        if self.tracker is not None:
            self.tracker.reset()
        self.clear_finger()
        if self.reader is not None:
            self._seen = self.reader.snapshot()[2]  # a scan already under way belongs to the old sheet
        if self.on_sheet is not None:
            self.on_sheet(sheet)

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
            self._remember_sheet(sheet.cells)
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
        self.finger_px, self.finger_mm, self.finger_posted = (float(x), float(y)), None, time.monotonic()

    def set_finger_mm(self, x: float, y: float) -> None:
        self.finger_mm, self.finger_px, self.finger_posted = (float(x), float(y)), None, time.monotonic()

    def clear_finger(self) -> None:
        self.finger_px = self.finger_mm = None
        self.finger_posted = 0.0

    def _posted_finger(self) -> Optional[tuple]:
        """The point last posted to /api/finger, in page mm, while it is still fresh (see POSTED_FINGER_SECONDS)."""
        if time.monotonic() - self.finger_posted > POSTED_FINGER_SECONDS:
            return None
        if self.finger_mm is not None:
            return self.finger_mm
        if self.finger_px is not None and self.H is not None:
            return to_page(self.H, *self.finger_px)
        return None

    def finger(self) -> Optional[tuple]:
        """The fingertip's page position in mm, or None if unknown or the page isn't registered.

        A posted point wins while it is fresh, then the camera has it back: whoever clicked meant "it is here, now", not
        "it is here for the rest of the session", and a click nobody remembers making is indistinguishable from a tutor
        that has decided every letter is the same one.

        When the green ring is on a camera tip, grade the finger *pad* a little behind that tip (toward the
        palm) so resting on a cell of the denser words / lookalikes demo sheets matches what the ring shows.
        A held smoothed position only fills in through brief tip dropouts — never outranks a live on-page tip."""
        posted = self._posted_finger()
        if posted is not None:
            return posted
        if self.tracker is None or self.H is None:
            return None
        tip = self.tracker.tip
        if tip is not None:
            raw = self.tracker.contact_mm(self.H, tip)
            if raw is not None:
                return raw
        return self.tracker.position

    def finger_source(self) -> Optional[str]:
        """Where the position now comes from: "posted" (a click is standing in), "camera", or None if there is none."""
        if self._posted_finger() is not None:
            return "posted"
        return "camera" if self.finger() is not None else None

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
                # The sheet's printed layout is the name we keep: a finger over a cell hides dots and would
                # otherwise make the live reading spell a different letter. Red dots still show what the camera sees.
                printed = {(c["row"], c["col"]): letter_of(c["dots"]) for c in (self.observe_sheet or [])}
                for c in self.stable:  # one box per cell, in page mm mapped back to the picture
                    quad = [to_image(H, c["x"] + sx * c["w"] / 2, c["y"] + sy * c["h"] / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    draw_cell(view, quad, c, GREEN if c.get("locked") else AMBER, False, True, letters=True,
                              letter_text=printed.get((c["row"], c["col"])))
            if hud:
                lines = [(self.message, GREEN if H is not None else RED)]
                if self.identify_until:
                    lines.append(("looking for the next page...", AMBER))
                elif H is not None and self.stable:
                    lines.append(locked_check_line(self.stable, self.observe_sheet))
                elif H is not None:
                    lines.append(("reading the sheet...", AMBER))
                if self.tracker is not None and self._posted_finger() is None and self.finger() is None:
                    if self.tracker.tip is not None and H is None:
                        lines.append(("finger seen — lock the page (markers) or click the picture", AMBER))
                    elif self.tracker.tip is not None:
                        lines.append(("finger seen — hold still on a cell", AMBER))
                    else:
                        lines.append(("no finger — click the picture where it is", RED))
                draw_hud(view, lines)
        if self.detector is not None and (H is None or self.always_reading):
            draw_detections(view, self.detector.boxes)  # what the camera reads, shown even when the page isn't registered
        if H is not None and hasattr(self.page_src, "size_mm"):  # outline the page the edge finder found
            draw_page_outline(view, H, *self.page_src.size_mm, self.page_src.origin)
        if self.finger_px is not None and self._posted_finger() is not None:  # the orange ring goes when the click stops counting
            cv2.circle(view, (int(self.finger_px[0]), int(self.finger_px[1])), 14, (255, 128, 0), 3)
            cv2.circle(view, (int(self.finger_px[0]), int(self.finger_px[1])), 4, (255, 128, 0), -1)
        elif self.tracker is not None and self.tracker.tip is not None:
            # Always on the camera fingertip (image pixels), not the smoothed page point used for grading.
            x, y = int(self.tracker.tip.x), int(self.tracker.tip.y)
            cv2.circle(view, (x, y), 16, (0, 255, 0), 3)
            cv2.circle(view, (x, y), 4, (0, 255, 0), -1)
        return view


def run_camera(session: TutorSession, cap, page_src, overlay_cells: Optional[list], title: str = "tutor",
               labels: Optional[dict] = None, layout_scan: bool = False, detector=None, track_finger: bool = False,
               observe_sheet: Optional[list] = None, show_reading: bool = True, known_sheets: Optional[dict] = None,
               sheet_name: str = "alphabet") -> None:
    """Show the camera with the page status, let clicks stand in for the fingertip, and dispatch key shortcuts.

    layout_scan=True makes word modes read from the session's known cells instead of running the detector."""
    feed = CameraFeed(page_src, overlay_cells, labels, detector, track_finger=track_finger, observe_sheet=observe_sheet,
                      show_reading=show_reading, known_sheets=known_sheets, sheet_name=sheet_name,
                      finder=best_finder() if track_finger else None)
    wire_new_page(session, feed)
    session.start_watching()  # says so if the camera loses the sheet, or if it did not understand what was said
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
    ap.add_argument("--mode", choices=("letters", "read", "word-quiz", "explore", "learn", "menu"), default="letters",
                    help="learn: guided lessons (teach a few letters at a time, practise, recap, adaptive review, free exploring), "
                         "starts by itself and remembers progress; explore: no quiz, it says what the camera detects under your finger; "
                         "menu: the website picks who is using it and whether they learn, read or take a quiz (the default of tutor_server.py)")
    ap.add_argument("--profile", default="default", help="learn mode: whose progress to keep (one file per name in ~/.braillie)")
    ap.add_argument("--no-tones", action="store_true", help="learn mode: no little sounds for right and wrong answers, only speech")
    ap.add_argument("--speech-speed", type=float, default=1.0, help="how fast the tutor talks, 0.6 (slow) to 1.5 (fast); the learner can also say slower or faster")
    ap.add_argument("--no-llm", action="store_true", help="learn mode uses the AI coach automatically when ANTHROPIC_API_KEY is set; this stops that")
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
                    "needs ANTHROPIC_API_KEY. Never slows the tutor: see llm.py")
    ap.add_argument("--llm-model", default=None, help="model name for the LLM helper (default claude-haiku-5-5, or $BRAILLIE_LLM_MODEL)")
    ap.add_argument("--adaptive", action="store_true", help="use Claude to prioritize targets from this session's miss history; "
                    "falls back safely when no plan is ready")


def make_coach(a) -> Optional[Coach]:
    """The LLM helper if asked for (--llm) or, in learn mode, whenever ANTHROPIC_API_KEY is set (--no-llm stops that); else None and the
    tutor uses its built-in wording. Only lesson facts are ever sent."""
    explicit = getattr(a, "llm", False)
    automatic = getattr(a, "mode", "") in ("learn", "menu") and not getattr(a, "no_llm", False)
    if not (explicit or automatic):
        return None
    client = LLMClient.from_env(getattr(a, "llm_model", None))
    if client is None:
        if explicit:
            print("--llm was given but ANTHROPIC_API_KEY is not set: the LLM helper stays off.", flush=True)
        else:
            print("AI coach off: add ANTHROPIC_API_KEY=... to the .env file to turn it on (the built-in coaching works without it).", flush=True)
        return None
    print(f"AI coach on ({client.model}). Only lesson facts (letter names, dot numbers, scores) are sent, never camera images "
          "or audio. If the service is slow or fails, the built-in wording is used.", flush=True)
    return Coach(client)


def make_ask_tutor(a=None) -> Optional[AskTutor]:
    """Claude Q&A for the braillo wake word. On whenever ANTHROPIC_API_KEY is set, unless --no-llm."""
    if a is not None and getattr(a, "no_llm", False):
        return None
    model = getattr(a, "llm_model", None) if a is not None else None
    client = LLMClient.from_env(model)
    if client is None:
        print("Braillo Q&A off: add ANTHROPIC_API_KEY=... to the .env file, then say braillo and ask a question.", flush=True)
        return None
    ask = AskTutor(client)
    print(f"Braillo Q&A on ({client.model}). Say braillo, then ask anything.", flush=True)
    return ask


def make_adaptive_planner(a) -> Optional[AdaptiveTargetPlanner]:
    """Return the opt-in, bounded-latency Claude target planner."""
    if not getattr(a, "adaptive", False):
        return None
    client = LLMClient.from_env(a.llm_model)
    if client is None:
        log.warning("Adaptive target selection is enabled but ANTHROPIC_API_KEY is not set; using fallback selection.")
        return None
    client.timeout = min(client.timeout, 2.5)
    return AdaptiveTargetPlanner(client)


def install_speed(voice, session: "TutorSession", a=None) -> None:
    """Make the tutor's speech follow session.settings.speech_speed (a time-stretch of Deepgram's audio; see audio_speed.py). Call it AFTER the
    phone speaker is installed, so what goes to the phone is slowed too. Nothing to do with the mock voice, which only prints."""
    if a is not None and getattr(a, "speech_speed", 1.0) != 1.0:
        session.apply_settings({"speech_speed": a.speech_speed})
    if getattr(voice, "MOCK_MODE", False):
        return
    try:
        audio_speed.install_speech_speed(voice, lambda: session.settings.speech_speed)
    except RuntimeError as e:  # voice_io changed: speech still works, just not at a chosen speed
        print(f"speech speed will not be adjustable: {e}", flush=True)


def progress_for(a) -> tuple:
    """(Progress, file) for the learner in learn mode, loaded from ~/.braillie; (None, None) in other modes."""
    if getattr(a, "mode", "") == "menu":  # progress belongs to whoever signs in: nothing is loaded or saved until then
        return None, None
    if getattr(a, "mode", "") != "learn":
        return None, None
    path = progress_path(getattr(a, "profile", "default"))
    progress = Progress.load(path)
    print(f"Progress for '{getattr(a, 'profile', 'default')}': {len(progress.learned())} letters learned, {progress.sessions} sessions ({path})", flush=True)
    return progress, path


def setup_from_args(a, ap: argparse.ArgumentParser) -> Setup:
    """Resolve --mode / --sheet / --words / --detect into cells, names and reading behaviour."""
    sheet = get_sheet(a.sheet or "alphabet") if (a.mode in ("letters", "explore", "learn", "menu") or a.sheet) else None
    words = list(a.words)
    if a.mode == "word-quiz" and not words:
        if sheet is not None and sheet.spec.name == "words":
            words = [w for line in reader.read_lines(sheet.cells) for w in line.split()]
        else:
            ap.error("--mode word-quiz needs --words (or --sheet words)")
    if sheet is None:
        return Setup([], None, None, False, words, contracted=a.mode in ("read", "word-quiz"))
    if a.mode in ("explore", "learn", "menu"):  # speak what the camera SEES on the sheet, never the layout copied from the file
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
    progress, progress_file = progress_for(a)
    session = TutorSession(voice, wc, setup.cells, finger=lambda: None, mode=a.mode, questions=a.questions, words=setup.words,
                           rng=random.Random(a.seed), names=setup.names, coach=make_coach(a), contracted=setup.contracted,
                           adaptive_planner=make_adaptive_planner(a),
                           progress=progress, progress_file=progress_file, tones=not a.no_tones, ask=make_ask_tutor(a))
    session.voice_status = voice_check(voice)
    report_voice(session.voice_status)  # silence must never be a mystery
    session.attach()
    phone = phonelink.start_phone(a, announce=lambda text: session.say(text), voice=voice)  # the video window then shows the QR code until a phone connects
    install_speed(voice, session, a)
    cap, page_src = (phone[1] if phone else open_camera(a.camera)), page_source_from_args(a)
    voice.start_listening()
    try:
        if phone:  # someone who cannot see the QR code on screen hears how to connect
            session.say(phone[0].spoken_instructions())
        if a.autostart or a.mode in ("explore", "learn"):  # these have nothing to wait for
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
