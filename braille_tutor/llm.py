"""Optional LLM helper for the tutor, using Claude (the Anthropic Python SDK).

Rules that keep it from ever hurting the app:
  * off unless asked for (--llm) and a key is set (ANTHROPIC_API_KEY); with it off the tutor behaves exactly as before
  * nothing the learner waits for depends on it: extras are fetched in the background and are a lookup when needed; the
    one call that is waited on (the debrief) has a hard time limit and falls back to the built-in debrief
  * only lesson facts are sent (cell names, dot numbers, scores): never camera images, audio or names. Braillo (AskTutor) also
    gets the text of what the learner said to it: a question after the wake word, or speech that matched no command
  * replies are checked before use: no digits that were not in the facts, no position words that don't apply, length limits
  * after the first error or timeout the coach switches itself off for the rest of the session; Braillo does after three in a row,
    or at once if the key is refused
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional
from urllib.parse import urlparse

import anthropic

DEFAULT_BASE_URL = "https://api.anthropic.com"
# The cheapest current Claude model, and fast: plenty for short spoken replies. Override with --llm-model or BRAILLIE_LLM_MODEL
# (claude-sonnet-5-5 or claude-opus-5-5 for smarter, costlier answers).
DEFAULT_MODEL = "claude-haiku-5-5"
POSITION_WORDS = {"top", "middle", "bottom", "left", "right"}
# Braillo's conversation (answering questions, working out what was meant): the fast model. The questions are short and the facts
# that matter are given to him (BRAILLE_FACTS, BRAILLE_HISTORY), so it answers well, and starts speaking in about a second (streamed).
# BRAILLIE_TALK_MODEL=claude-opus-5-5 gives richer answers, about half a second later.
TALK_MODEL = DEFAULT_MODEL
REPLY_EFFORT = "low"  # everything asked for here is a short spoken line or a small plan: low effort answers soonest
MAX_TOKENS = 4000  # the model's thinking counts toward this too; the prompts themselves keep what is said short
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # a declined request is re-run on Anthropic's recommended fallback model
NO_FALLBACK_PREFIXES = ("claude-haiku",)
QUICK_WITHOUT_THINKING = ("claude-haiku",)  # models that may skip thinking at low effort (Opus 5.5 cannot: a 400)  # Haiku has no server-side fallback model: the request is sent without one


class LLMError(RuntimeError):
    """The model answered, but not with something usable (declined, or cut off)."""


class LLMClient:
    """A minimal Claude client (Messages API). Only https (or a localhost address) is accepted as the endpoint."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, timeout: float = 8.0):
        url = urlparse(base_url)
        if url.scheme != "https" and url.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError("the LLM base URL must be https (or localhost)")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        # one quick retry at most: a spoken reply that arrives much later is no use to anyone
        self._client = anthropic.Anthropic(api_key=api_key, base_url=self.base_url, max_retries=1)

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> Optional["LLMClient"]:
        """A client using ANTHROPIC_API_KEY (or CLAUDE_API_KEY; and optional BRAILLIE_LLM_MODEL / BRAILLIE_LLM_BASE_URL), or None
        if there is no key."""
        key = (os.environ.get("ANTHROPIC_API_KEY", "").strip() or os.environ.get("CLAUDE_API_KEY", "").strip())
        if not key:
            return None
        return cls(key, model or os.environ.get("BRAILLIE_LLM_MODEL", DEFAULT_MODEL),
                   os.environ.get("BRAILLIE_LLM_BASE_URL", DEFAULT_BASE_URL))

    def _text(self, system: str, user: str, max_tokens: Optional[int], output: Optional[dict] = None,
              timeout: Optional[float] = None, quick: bool = False) -> str:
        """One request; the reply's text. Raises on any error, and on a reply that was declined or cut off. `quick`: a small job
        (a line put another way, a hint) done without thinking first where the model allows it, which takes a third of the time."""
        request = dict(model=self.model, max_tokens=max_tokens or MAX_TOKENS, system=system,
                       messages=[{"role": "user", "content": user}],
                       output_config={"effort": REPLY_EFFORT, **(output or {})}, timeout=timeout or self.timeout)
        if quick and self.model.startswith(QUICK_WITHOUT_THINKING):
            request["thinking"] = {"type": "disabled"}
        # (a request with its own short time limit is not retried: a retry would double the wait it is limiting, and the caller
        # already has something to say instead)
        client = self._client.with_options(max_retries=0) if timeout else self._client
        if self.model.startswith(NO_FALLBACK_PREFIXES):
            response = client.messages.create(**request)
        else:
            response = client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"no usable reply (stop_reason={response.stop_reason})")
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None, timeout: Optional[float] = None,
             quick: bool = False) -> str:
        """One request; returns the reply text. Raises on any error (a `timeout` in seconds, if given, for this request only)."""
        return self._text(system, user, max_tokens, timeout=timeout, quick=quick)

    def chat_json_stream(self, system: str, user: str, schema: dict, on_text) -> dict:
        """chat_json, streamed: `on_text(text so far)` is called as the reply arrives (so the start of an answer can be spoken
        before the rest has come), then the whole reply is returned as a dict."""
        shape = schema.get("schema", schema)
        request = dict(model=self.model, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}],
                       output_config={"effort": REPLY_EFFORT, "format": {"type": "json_schema", "schema": shape}},
                       timeout=self.timeout)
        if self.model.startswith(NO_FALLBACK_PREFIXES):
            stream_ctx = self._client.messages.stream(**request)
        else:
            stream_ctx = self._client.beta.messages.stream(**request, betas=[FALLBACK_BETA], fallbacks="default")
        text = ""
        with stream_ctx as stream:
            for chunk in stream.text_stream:
                text += chunk
                on_text(text)
            final = stream.get_final_message()
        if final.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"no usable reply (stop_reason={final.stop_reason})")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Claude returned a non-object reply")
        return parsed

    def chat_json(self, system: str, user: str, schema: dict, quick: bool = False) -> dict:
        """A reply shaped exactly by `schema` (a JSON schema, or {"name", "schema"} around one), as a dict."""
        shape = schema.get("schema", schema)
        parsed = json.loads(self._text(system, user, None, {"format": {"type": "json_schema", "schema": shape}}, quick=quick))
        if not isinstance(parsed, dict):
            raise ValueError("Claude returned a non-object reply")
        return parsed


class Pending:
    """A background call whose result may or may not arrive in time."""

    def __init__(self, fn):
        self.value, self.done = None, threading.Event()
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn) -> None:
        try:
            self.value = fn()
        except Exception as e:  # the caller decides what a failure means
            self.value = e
        self.done.set()

    def get(self, wait: float):
        """The result if it arrived within `wait` seconds, else None (also None if the call failed)."""
        self.done.wait(max(wait, 0.0))
        return None if isinstance(self.value, Exception) or not self.done.is_set() else self.value


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def clean_line(text, max_chars: int, allow_digits: bool = False, allowed_positions: Optional[set] = None) -> Optional[str]:
    """A reply fit to be spoken, or None. One line, no markdown, no digits unless allowed, no unrelated position words."""
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text or len(text) > max_chars or re.search(r"[*_`#<>{}\[\]|~]|https?:", text):
        return None
    if not allow_digits and re.search(r"\d", text):
        return None
    if allowed_positions is not None and (_words(text) & POSITION_WORDS) - allowed_positions:
        return None
    return text


class Coach:
    """Extras for the tutor: memory aids, praise lines and a personal debrief. Every method is safe to call at any time."""

    def __init__(self, client: Optional[LLMClient] = None, debrief_wait: float = 2.0):
        self.client, self.debrief_wait, self.off_reason = client, debrief_wait, ""
        self.aids: dict = {}
        self.praise: list = []
        self.cheer: list = []
        self._pi = self._ci = 0

    @property
    def enabled(self) -> bool:
        return self.client is not None and not self.off_reason

    @property
    def status(self) -> str:
        if self.client is None:
            return "off"
        return f"off after an error: {self.off_reason}" if self.off_reason else f"on ({self.client.model})"

    def _ask(self, system: str, user: str) -> Optional[str]:
        """Blocking call that never raises; the first failure switches the helper off for the session."""
        if not self.enabled:
            return None
        try:
            return self.client.chat(system, user)
        except Exception as e:
            self.off_reason = f"{type(e).__name__}: {e}"[:120]
            return None

    # ---- prefetched extras (background; a lookup when used) -------------------------------------
    def prefetch(self, cells: list) -> None:
        """Start fetching memory aids for `cells` ([{"name","dots","positions"}]) and praise lines, without waiting."""
        if self.enabled:
            threading.Thread(target=self._prefetch, args=(cells,), daemon=True).start()

    def _prefetch(self, cells: list) -> None:
        system = ("You help teach braille by touch to people who cannot see the page. Reply with JSON only, no other text. "
                  'Shape: {"aids": {"<cell name>": "<memory aid>"}, "praise": ["..."], "cheer": ["..."]}. '
                  "A memory aid is ONE spoken sentence of at most 14 words that helps someone feel that cell's pattern. Describe "
                  "positions only with the position words given for that cell (top, middle, bottom, left, right); never write "
                  "digits. Praise: 10 different celebrations of a correct answer, at most 4 words each. Cheer: 6 different "
                  "kind encouragements after a wrong answer, at most 7 words each. No digits, emoji or markdown anywhere.")
        raw = self._ask(system, json.dumps({"cells": cells}))
        try:
            data = json.loads(re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip())
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        for cell in cells:
            allowed = {w for p in cell.get("positions", []) for w in re.findall(r"[a-z]+", p)} & POSITION_WORDS
            line = clean_line((data.get("aids") or {}).get(cell["name"]), 110, allowed_positions=allowed)
            if line:
                self.aids[cell["name"]] = line
        self.praise = [t for t in (clean_line(x, 30) for x in data.get("praise") or []) if t]
        self.cheer = [t for t in (clean_line(x, 50) for x in data.get("cheer") or []) if t]

    def aid(self, name: str) -> Optional[str]:
        """The memory aid for a cell name, if one has arrived and passed the checks."""
        return self.aids.get(name)

    def next_praise(self) -> Optional[str]:
        if not self.praise:
            return None
        self._pi += 1
        return self.praise[self._pi % len(self.praise)]

    def next_cheer(self) -> Optional[str]:
        if not self.cheer:
            return None
        self._ci += 1
        return self.cheer[self._ci % len(self.cheer)]

    # ---- the one call that is waited on ---------------------------------------------------------
    def debrief_async(self, facts: dict) -> Optional[Pending]:
        """Start writing a personal debrief from `facts`; returns a Pending, or None if the helper is off."""
        if not self.enabled:
            return None
        system = ("You write the short spoken end-of-session message for a braille tutor, for someone learning by touch. "
                  "Two or three warm sentences, at most 55 words, plain speech: no markdown, emoji or lists. Use ONLY the "
                  "facts you are given; never invent scores, letters or dot numbers. If they missed something, name at "
                  "most two and give one concrete tip using the dot patterns provided.")
        return Pending(lambda: self._ask(system, json.dumps(facts)))

    def debrief_text(self, pending: Optional[Pending], facts: dict) -> Optional[str]:
        """The debrief if it arrived within the time limit and passes the checks (every number in it must be in `facts`)."""
        text = pending.get(self.debrief_wait) if pending else None
        if text is None:
            if pending is not None and not self.off_reason:
                self.off_reason = "the debrief did not arrive in time"
            return None
        allowed = set(re.findall(r"\d+", json.dumps(facts)))
        line = clean_line(text, 420, allow_digits=True)
        if line is None or set(re.findall(r"\d+", line)) - allowed:
            return None
        return line


# ---- spoken Q&A: say "braillo" then ask anything -------------------------------------------------

ASK_WAKE = "braillo"
# Deepgram often reshapes the name; any of these arms the tutor the same way.
ASK_WAKE_ALIASES = (ASK_WAKE, "briello", "brailo", "barillo", "braylo", "brello")
_ASK_WAKE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in ASK_WAKE_ALIASES) + r")\b[,:]?\s*",
    re.IGNORECASE,
)
ASK_CANCEL = frozenset({"stop", "cancel", "never mind", "nevermind", "nothing", "forget it"})
def _braille_facts() -> str:
    """The alphabet's dots and where each dot sits, so even a small model never has to remember them (and get them wrong)."""
    first = {"a": "1", "b": "12", "c": "14", "d": "145", "e": "15", "f": "124", "g": "1245", "h": "125", "i": "24", "j": "245"}
    dots = dict(first)
    dots.update({chr(ord(k) + 10): v + "3" for k, v in first.items()})  # k to t: a to j plus dot 3
    dots.update({u: first[a] + "36" for u, a in zip("uvxyz", "abcde")})  # u v x y z: a to e plus dots 3 and 6
    dots["w"] = "2456"
    letters = "; ".join(f"{k} {', '.join(sorted(v))}" for k, v in sorted(dots.items()))
    return ("Braille facts (use these, never guess): dot 1 top-left, dot 2 middle-left, dot 3 bottom-left, dot 4 top-right, "
            f"dot 5 middle-right, dot 6 bottom-right. Letters and their dots: {letters}.")


BRAILLE_FACTS = _braille_facts()
# What Braillo talks about: he is a braille tutor, not a general assistant. Asked for a brownie recipe, he says so kindly and steers
# back, rather than answering it.
ON_TOPIC = (
    "Stay on your subject: braille and reading and writing by touch, its history and Louis Braille, blindness and low vision as "
    "they relate to reading and learning, tips for learning, and this app and how to use it. If they ask about anything else (a "
    "recipe, homework, the news, other topics), do not answer it or give any of it: say kindly, in one short sentence, that you are "
    "only here to help with braille, and offer to carry on with it. Small talk about how they are feeling, or a thank you, is fine "
    "to answer briefly."
)
# What a curious learner asks about ("when was braille invented?", "who made it?"): the well-established facts, so the answer is
# right rather than remembered. Anything beyond them, Braillo answers from what he knows, and says so if he is not sure.
BRAILLE_HISTORY = (
    "About braille (well-established facts: use them, and if asked something they do not cover, answer from what you know and say "
    "if you are not sure): Braille was invented by Louis Braille, born on 4 January 1809 in Coupvray, near Paris, France. He lost "
    "his sight as a small child after an accident with a tool in his father's leather workshop, at about three years old. At the "
    "Royal Institute for Blind Youth in Paris he met Charles Barbier's night writing, a code of raised dots made for soldiers to "
    "read in the dark, and simplified it into a cell of six dots. He had the system working by 1824, when he was about fifteen, and "
    "first published it in 1829. He died on 6 January 1852; France officially adopted braille in 1854, and it spread around the "
    "world. World Braille Day is on 4 January, his birthday. A cell has six dots in two columns of three, which gives 64 patterns "
    "(63 with at least one dot raised). Braille is a way of writing, not a language: it is used for many languages. Uncontracted "
    "braille (often called grade 1) spells every letter; contracted braille (grade 2) uses short forms for common words and letter "
    "groups, such as one cell for and or the. Unified English Braille (UEB) is the standard English code in most English-speaking "
    "countries; the United States adopted it in 2016. Maths can be written in Nemeth code (in the United States) or UEB, and there "
    "is braille for music. Braille is written with a slate and stylus, a braille writer such as the Perkins Brailler (from 1951), "
    "or printed by braille embossers, and read on refreshable braille displays that raise pins electronically."
)
PERSONA = (
    "You are Braillo, a friendly little robot shaped like a braille cell, and the voice of Braillie, an app that teaches braille by "
    "touch to people who are often blind or have low vision. You talk like a warm, patient, upbeat teacher: short, natural spoken "
    "sentences, a little playful, never patronising. Celebrate effort as well as success. If they sound frustrated or lost, be kind, "
    "and offer the next small step."
)
APP_MAP = (
    "How Braillie works: the learner prints practice sheets (the alphabet, words, look-alike letters, and numbers and signs), presses "
    "each dot through from the back so it is raised, and lays the sheet under their phone's camera. The phone streams video to the "
    "laptop, which finds the sheet and their fingertip. The pages, in order: the front page (say start learning to begin), sign-in "
    "(Google, or guest), connect the phone (scan the QR code with its camera), home (choose Learn, Read or Quiz), the activity "
    "itself, and their own page (their progress). Learn has eight short lessons: A to E, F to J, K to O, P to T, U to Z, look-alikes, "
    "numbers, then capitals and punctuation; before the first one, a new learner says how much braille they know (new, some, "
    "alphabet or numbers) and starts there. Each letter is described by its dots, and the learner finds it and rests a finger "
    "on it for about three seconds; then a mixed practice; and, if it went well, they spell a real word. Read: rest a finger on a "
    "word on the words sheet and it is read aloud. Quiz: you name a letter on the look-alikes sheet, and they point at it and say "
    "this one (or its name), or rest a finger on it. At any time they can say home, my page, next lesson, lesson three, hint, "
    "repeat, slower, faster, or sign out."
)
RULES = (
    "With each message you get app_now: the page they are on and what it is for, whether the phone and camera can see the sheet, "
    "which letter their finger is on, their progress, and what you are waiting for them to do. Use it: answer about the moment they "
    "are in, and gently steer them toward what you are waiting for. Only tell them to say words that work on the page they are on "
    "now (what you are waiting for, and the buttons on this page): never the words of a page they have not reached yet, and do not "
    "describe what the next page will ask (it says so itself). When they clearly want to do what you are waiting for, do it rather "
    "than telling them which words to say. Do not tell them again what they have just been told (you_said_last), such as that a guest's "
    "progress is not saved. Never "
    "claim to see what app_now does not say, and never state facts about the app beyond the description above. Do not end every "
    "reply with a nudge about what to say next: add one only when it helps, and never the same one twice running. If the camera "
    "cannot see the sheet or their finger, say so, and how to fix it: the whole sheet in view and flat, good light, the phone about "
    "thirty centimetres above. Teaching: describe letters by their dots. Do not give away where a letter is on the sheet (its row "
    "and column) unless they have already had two hints, or ask for it outright. You do not know how the sheet is laid out, so "
    "never say where a letter is compared with another (to its left, the next one along): guide them by the dots instead. Spoken words only: no markdown, lists, emoji, web "
    "addresses or braille symbols; say letters and dot numbers in words. Do not repeat your last line word for word. "
)
ASK_SYSTEM = (
    PERSONA + " " + APP_MAP + " " + RULES + "They asked you a question: answer it like a real tutor who knows the topic and enjoys "
    "it, in two to four spoken sentences (under 70 words) unless they clearly want more. " + ON_TOPIC + " For braille and its "
    "history, use the facts below. " + BRAILLE_FACTS + " " + BRAILLE_HISTORY
)
# What free speech (no exact command phrase) may be turned into; the tutor runs the matching command.
TALK_ACTIONS = ("none", "repeat", "hint", "found it", "next", "next page", "start", "stop", "menu", "learn", "read", "quiz",
                "explore", "practice", "help", "slower", "faster", "next lesson", "previous lesson", "lesson", "my page", "sign out")
TALK_SYSTEM = (
    PERSONA + " " + APP_MAP + " " + RULES + "You get what the microphone just heard (speech to text, so it may be garbled) and the "
    "last few exchanges. Decide what to do:\n"
    "- addressed: false if this was not said to you (other people talking, the TV, noise, half a word). Then action none, say "
    "empty, lesson 0.\n"
    "- action: the app command they want, if they clearly want one. For example: 'what was that letter again' is repeat; 'I'm "
    "stuck' or 'where is it' is hint; 'this one' or 'I'm on it now' is found it; 'I'm ready' or 'I've got the sheet' (while waiting for a sheet) is start; 'skip this one' is next; 'take me home' or 'let's "
    "do something else' is menu; wanting to stop, quit, finish or take a break ('I want to stop', 'I'm done', 'can we stop now', 'I need a break') is menu too: the app then asks them to confirm, so never just say you are stopping, or tell them to say home, without the action; 'I want to practise reading' is read; 'can we move on to the next lesson' is next lesson; 'go "
    "back a lesson' is previous lesson; 'let's do K to O' or 'lesson three' is lesson (with its number); 'how am I doing' or 'show "
    "my progress' is my page; 'I'm done for today, log me out' is sign out. Otherwise none.\n"
    "- lesson: the lesson number (1 to 8) when the action is lesson; otherwise 0.\n"
    "- say: what to say back: usually one or two short sentences, under 40 words. Leave it empty when the action speaks for itself "
    "(repeat, hint, next, a lesson or a page already talk). When they ask a question (about braille, its history, the letters, the "
    "app or the moment), answer it properly, like a tutor who enjoys the topic: up to four spoken sentences, under 70 words. If "
    "they did not understand your last line, explain it differently. " + ON_TOPIC + " " + BRAILLE_FACTS + " " + BRAILLE_HISTORY
)
SPEAKING_ACTIONS = ("hint", "next lesson", "previous lesson", "lesson", "my page", "menu", "learn", "read", "quiz", "start", "practice",
                    "explore", "next", "next page", "sign out", "stop")  # actions that say their own piece once they run
TALK_SCHEMA = {
    "name": "braillo_reply",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "addressed": {"type": "boolean"},
            "action": {"type": "string", "enum": list(TALK_ACTIONS)},
            "lesson": {"type": "integer"},
            "say": {"type": "string"},
        },
        "required": ["addressed", "action", "lesson", "say"],
        "additionalProperties": False,
    },
}
HINT_TIMEOUT = 2.5  # seconds: a hint not worded by then is given as the app wrote it
# A hint for this learner: the app decides what the hint gives away (the dots first, then how the letter grows from one they know,
# then where it is) and counts it; Claude only puts it for them, in the light of what they touched and what they mix up.
HINT_SYSTEM = (
    PERSONA + " A learner is looking for a braille symbol on their sheet and needs a hint. You get `hint`, the hint the app is about "
    "to give; `keep`, words it must still contain exactly (dot positions, row and column); `facts`, what they are looking for, what "
    "they just touched instead and exactly how the two differ, what they often mix it up with, and their level; and `you_said_last`, "
    "your last few lines. Give the same hint your own way, for this learner, right now: if they just touched something else, start "
    "from it, using the difference given (never work dots out yourself); if they often mix it up with another, say so; for a "
    "beginner be gentle and simple, for someone who knows more be brief. Encouraging, never discouraging. Keep every word in `keep`. "
    "Do not say where it is on the sheet (rows, columns, left or right of anything) unless `hint` already does, and give away "
    "nothing more than `hint` does. One or two short spoken sentences, under 40 words, not worded like your last lines. Spoken "
    "words only: no markdown, lists or quotation marks."
)
VARIANTS_SYSTEM = (
    PERSONA + " You get `line`, something the app says aloud for you again and again, and `keep`, words it must still contain exactly "
    "as written (what the learner can say back, names of sheets, letters, numbers). Write `how_many` different ways you might say "
    "it, each warm and natural, as you would in conversation, and each clearly different from the others. Keep every instruction "
    "and every word in `keep`; add nothing the learner must do and no new facts. Never longer than the line by more than a third. "
    "Spoken words only: no markdown, lists, emoji or quotation marks."
)
VARIANTS_SCHEMA = {"type": "object", "properties": {"versions": {"type": "array", "items": {"type": "string"}}},
                   "required": ["versions"], "additionalProperties": False}
HISTORY_TURNS = 6  # exchanges kept, so "what about the next one?" means something
LESSON_COUNT = 8  # (learn.LESSONS: A to E ... capitals and punctuation)
PHRASE_TIMEOUT = 2.5  # seconds: a line Braillo would say in his own words is said as written if Claude has not answered by then
# Braillo in his own words: the app knows WHAT to say (the facts, the instructions, the words the learner can say back); Claude says
# it the way Braillo would right now, so the same moment does not sound the same every time, and what was just said is not said again.
PHRASE_SYSTEM = (
    PERSONA + " " + APP_MAP + " You get `line`, something the app is about to say aloud for you; `keep`, words it must still contain "
    "exactly as written (what the learner can say back, names of sheets, letters, numbers, codes); `app_now`, what is happening; and "
    "`you_said_last`, your last few lines. Say the line your own way, as you would say it in this moment: warm, natural, a little "
    "playful, never stiff or robotic, and not the same words as before. Keep every instruction and every word in `keep`. Add nothing "
    "the learner must do that the line does not ask, and no facts that are in neither the line nor app_now. If they heard something "
    "in the line a moment ago (in you_said_last), say it more briefly or leave it out, but keep every word in `keep`. Never longer "
    "than the line by more than a third. Spoken words only: no markdown, lists, emoji or quotation marks. Reply with the line only."
)


def strip_ask_wake(text: str) -> str:
    """Remove the first braillo wake word (and aliases) from a transcript; leftover is the question."""
    if not isinstance(text, str):
        return ""
    return _ASK_WAKE_RE.sub("", text, count=1).strip().strip(",:;!")


def looks_like_question(text: str) -> bool:
    """True when the leftover after the wake word is enough to send to the model."""
    words = text.split()
    return bool(words) and (len(words) >= 2 or text.endswith("?") or len(text) >= 8)


def _keeps(said: str, keep) -> bool:
    """Does a line said another way still contain each of these words or phrases, as whole words?"""
    low = said.casefold()
    return all(re.search(r"(?<![a-z0-9])" + re.escape(k.casefold()) + r"(?![a-z0-9])", low) for k in keep)


def _partial_field(text: str, name: str) -> Optional[str]:
    """A field of a JSON object still arriving: a finished string or boolean value, or None if it is not all there yet."""
    m = re.search(rf'"{name}"\s*:\s*("(?:[^"\\]|\\.)*"|true|false)', text)
    return None if m is None else m.group(1).strip('"')


def first_sentence_said(text: str) -> Optional[str]:
    """In a reply still arriving ({"addressed", "action", "lesson", "say"}): the first sentence of "say", once it is complete (a
    full stop, question or exclamation mark followed by more, or the end of the string), else None."""
    m = re.search(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)(")?', text)
    if m is None:
        return None
    raw = re.sub(r"\\+$", "", m.group(1))  # (an escape cut in half)
    try:
        said = json.loads('"' + raw + '"')
    except ValueError:
        return None
    first = re.match(r"\s*(.+?[.!?])\s+\S", said)
    if first:
        return first.group(1)
    return said.strip() if m.group(2) and said.strip() else None


def _spoken(raw) -> Optional[str]:
    """A reply fit to speak: as it is if it passes clean_line, else with the markup stripped."""
    line = clean_line(raw, 480, allow_digits=True)
    if line:
        return line
    plain = " ".join(str(raw or "").replace("\n", " ").split())
    plain = re.sub(r"[*_`#<>{}\[\]|~]", "", plain)
    return plain[:480] if plain else None


class AskTutor:
    """Fast spoken Q&A via Claude, aware of what is happening in the app and of the last few exchanges.

    answer()  a question after the braillo wake word
    respond() anything else the learner said that was not a command: what to do (a command) and what to say back, or
              nothing at all when it was not said to the tutor"""

    def __init__(self, client: Optional[LLMClient] = None, timeout: float = 10.0, fast: Optional[LLMClient] = None):
        self.client, self.off_reason = client, ""
        self.fast = fast or client  # (for phrase: a line in his own words has to come quickly)
        self.history: list = []  # [(what they said, what was said back)], newest last
        self.failures = 0
        if self.client is not None and hasattr(self.client, "timeout"):
            self.client.timeout = min(self.client.timeout, timeout)

    @property
    def enabled(self) -> bool:
        return self.client is not None and not self.off_reason

    @property
    def status(self) -> str:
        if self.client is None:
            return "off"
        return f"off after an error: {self.off_reason}" if self.off_reason else f"on ({self.client.model})"

    def _failed(self, e: Exception) -> None:
        """A key that is refused switches the helper off for the session; a slow or dropped request only does after three in a row."""
        self.failures += 1
        if self.failures >= 3 or getattr(e, "status_code", None) in (401, 403):
            self.off_reason = f"{type(e).__name__}: {e}"[:120]

    def _remember(self, heard: str, said: str) -> None:
        self.history = (self.history + [(heard, said)])[-HISTORY_TURNS:]

    def _prompt(self, text: str, context: Optional[dict]) -> str:
        if not context and not self.history:
            return text
        return json.dumps({"app_now": context or {}, "earlier": [{"learner": h, "you": s} for h, s in self.history],
                           "learner_said": text})

    def answer(self, question: str, context: Optional[dict] = None) -> Optional[str]:
        """Blocking Claude reply fit to speak, or None. `context`: what is happening in the app right now."""
        if not self.enabled:
            return None
        q = " ".join(str(question or "").split()).strip()
        if not q:
            return None
        try:
            raw = self.client.chat(ASK_SYSTEM, self._prompt(q, context))
        except Exception as e:
            self._failed(e)
            return None
        self.failures = 0
        line = _spoken(raw)  # (if the model used markup, still speak a plain version)
        if line:
            self._remember(q, line)
        return line

    def hint(self, hint: str, keep: list, facts: dict, recent: Optional[list] = None) -> Optional[str]:
        """The app's hint, put for this learner (see HINT_SYSTEM), or None: the app's own wording is used then."""
        if not self.enabled or not hint:
            return None
        prompt = json.dumps({"hint": hint, "keep": list(keep), "facts": facts, "you_said_last": list(recent or [])[-4:]})
        try:
            raw = self.fast.chat(HINT_SYSTEM, prompt, timeout=HINT_TIMEOUT, quick=True)
        except Exception as e:
            self._failed(e)
            return None
        self.failures = 0
        said = _spoken(raw)
        if not said or not _keeps(said, keep) or len(said.split()) > 50:
            return None
        if not re.search(r"\b(?:row|column)\b", hint, re.IGNORECASE) and re.search(r"\b(?:row|column)s?\b", said, re.IGNORECASE):
            return None  # (it gave away where it is before the hint ladder does)
        return said

    def variants(self, line: str, keep: list, n: int = 3) -> list:
        """A few ways Braillo might say a line that comes up again and again (made ahead of time, so saying one later costs no
        wait): each keeps every word of `keep` and is about as long as the line. [] if the helper is off or the reply is not usable."""
        if not self.enabled or not line:
            return []
        prompt = json.dumps({"line": line, "keep": list(keep), "how_many": n})
        try:
            reply = self.fast.chat_json(VARIANTS_SYSTEM, prompt, VARIANTS_SCHEMA, quick=True)
        except Exception as e:
            self._failed(e)
            return []
        self.failures = 0
        out = []
        for raw in reply.get("versions") or []:
            said = _spoken(raw)
            if said and _keeps(said, keep) and len(said.split()) <= len(line.split()) * 4 / 3 + 6 and said not in out:
                out.append(said)
        return out[:n]

    def phrase(self, line: str, keep: list, context: Optional[dict] = None, recent: Optional[list] = None) -> Optional[str]:
        """`line` in Braillo's own words for this moment, or None (helper off, slow, or a reply that lost a word of `keep` or ran
        long): the caller then says `line` as written."""
        if not self.enabled or not line:
            return None
        prompt = json.dumps({"line": line, "keep": list(keep), "app_now": context or {}, "you_said_last": list(recent or [])[-5:]})
        try:
            raw = self.fast.chat(PHRASE_SYSTEM, prompt, timeout=PHRASE_TIMEOUT, quick=True)
        except Exception as e:
            self._failed(e)
            return None
        self.failures = 0
        said = _spoken(raw)
        if not said:
            return None
        if not _keeps(said, keep):
            return None  # (lost something the learner needs: say it as written)
        if len(said.split()) > len(line.split()) * 4 / 3 + 6:
            return None
        return said

    def respond(self, heard: str, context: Optional[dict] = None, on_first=None) -> Optional[dict]:
        """What to do about free speech: {"action": one of TALK_ACTIONS, "say": text or ""}, or None if the helper is off or
        failed (the caller then falls back to its own "I didn't catch that"). Speech not meant for the tutor gives action none
        and nothing to say. With `on_first(sentence)`, the reply is streamed: its first sentence is handed over (to be spoken) as
        soon as it is complete, and "say" then holds only the rest."""
        if not self.enabled:
            return None
        text = " ".join(str(heard or "").split()).strip()
        if not text:
            return None
        spoken: list = []  # the first sentence, if it was already said while the rest arrived

        def started(so_far: str) -> None:
            if spoken or on_first is None:
                return
            first = first_sentence_said(so_far)
            if first is None:
                return
            action = _partial_field(so_far, "action")
            if _partial_field(so_far, "addressed") == "false" or action is None or action in SPEAKING_ACTIONS:
                spoken.append("")  # (nothing to say ahead: not for us, or the action speaks for itself)
                return
            line = _spoken(first)
            spoken.append(line or "")
            if line:
                on_first(line)

        try:
            streaming = on_first is not None and callable(getattr(self.client, "chat_json_stream", None))
            reply = (self.client.chat_json_stream(TALK_SYSTEM, self._prompt(text, context), TALK_SCHEMA, started) if streaming
                     else self.client.chat_json(TALK_SYSTEM, self._prompt(text, context), TALK_SCHEMA))
        except Exception as e:
            self._failed(e)
            return None
        self.failures = 0
        action = reply.get("action") if reply.get("action") in TALK_ACTIONS else "none"
        if reply.get("addressed") is False:
            return {"action": "none", "say": "", "lesson": 0}
        lesson = reply.get("lesson") if isinstance(reply.get("lesson"), int) else 0
        if action == "lesson" and not 1 <= lesson <= LESSON_COUNT:
            action = "none"  # (a lesson with no number it could name: nothing to go to)
        say = _spoken(reply.get("say")) or ""
        if action in SPEAKING_ACTIONS and say:  # the lesson or page says its own piece next: only a short "sure, let's go" before it
            say = re.split(r"(?<=[.!?])\s+", say)[0]
            say = say if len(say.split()) <= 8 else ""
        self._remember(text, say or f"({action})")
        if spoken and spoken[0] and say.startswith(spoken[0]):
            say = say[len(spoken[0]):].strip()  # (the first sentence has been said already)
        reply = {"action": action, "say": say, "lesson": lesson}
        if spoken and spoken[0]:
            reply["spoke"] = True  # (its first sentence was said already, as it arrived)
        return reply
