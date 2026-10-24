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
REPLY_EFFORT = "low"  # everything asked for here is a short spoken line or a small plan: low effort answers soonest
MAX_TOKENS = 4000  # the model's thinking counts toward this too; the prompts themselves keep what is said short
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # a declined request is re-run on Anthropic's recommended fallback model
NO_FALLBACK_PREFIXES = ("claude-haiku",)  # Haiku has no server-side fallback model: the request is sent without one


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

    def _text(self, system: str, user: str, max_tokens: Optional[int], output: Optional[dict] = None) -> str:
        """One request; the reply's text. Raises on any error, and on a reply that was declined or cut off."""
        request = dict(model=self.model, max_tokens=max_tokens or MAX_TOKENS, system=system,
                       messages=[{"role": "user", "content": user}],
                       output_config={"effort": REPLY_EFFORT, **(output or {})}, timeout=self.timeout)
        if self.model.startswith(NO_FALLBACK_PREFIXES):
            response = self._client.messages.create(**request)
        else:
            response = self._client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"no usable reply (stop_reason={response.stop_reason})")
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        """One request; returns the reply text. Raises on any error."""
        return self._text(system, user, max_tokens)

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """A reply shaped exactly by `schema` (a JSON schema, or {"name", "schema"} around one), as a dict."""
        shape = schema.get("schema", schema)
        parsed = json.loads(self._text(system, user, None, {"format": {"type": "json_schema", "schema": shape}}))
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
ASK_SYSTEM = (
    "You are Braillo, the spoken tutor inside Braillie, a braille learning app for people who may be blind or have "
    "low vision. Answer like a real tutor who knows the topic. Prefer plain speech for text-to-speech: no markdown, "
    "bullet lists, emoji, URLs, or braille symbols (they cannot be spoken: name the letters and dots instead). Keep answers under 60 words unless they clearly ask for more detail. Help with "
    "braille, reading, and the app when relevant, but answer any honest question. You may be given what is happening "
    "in the app right now (the activity, the letter being asked for and its dots, what you said last): use it so your "
    "answer fits the moment, and never just repeat your last line word for word. " + BRAILLE_FACTS
)
# What free speech (no exact command phrase) may be turned into; the tutor runs the matching command.
TALK_ACTIONS = ("none", "repeat", "hint", "found it", "next", "next page", "start", "stop", "menu", "learn", "read", "quiz",
                "explore", "practice", "help", "slower", "faster")
TALK_SYSTEM = (
    "You are Braillo, the voice of Braillie, a braille tutor for people who may be blind or have low vision. You get what "
    "the microphone just heard (speech to text, so it may be garbled), what is happening in the app, and the last few "
    "exchanges. Decide what to do:\n"
    "- addressed: false if this was not said to you (other people talking, the TV, noise, half a word). Then action none, say empty.\n"
    "- action: the app command they want, if they clearly want one (for example 'what was that letter again' is repeat, "
    "'I'm stuck' or 'where is it' is hint, 'I'm on it now' is found it, 'skip this one' is next, 'let's do something "
    "else' is menu, 'I want to practise reading' is read). Otherwise none.\n"
    "- say: what to say back, for text to speech: at most two short sentences, under 40 words, plain words, no lists or "
    "markdown, and no braille symbols (say the letters and dot numbers instead). Leave it empty when the action speaks for itself (repeat, hint, next and so on already talk). Answer "
    "questions about braille, the letters and the app directly. If they did not understand your last line, explain it "
    "differently instead of repeating it. Only use dot numbers and letters from the facts you are given or that you are "
    "certain of (a is dot 1, b is dots 1 2, c is dots 1 4, and so on)." + " " + BRAILLE_FACTS
)
TALK_SCHEMA = {
    "name": "braillo_reply",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "addressed": {"type": "boolean"},
            "action": {"type": "string", "enum": list(TALK_ACTIONS)},
            "say": {"type": "string"},
        },
        "required": ["addressed", "action", "say"],
        "additionalProperties": False,
    },
}
HISTORY_TURNS = 6  # exchanges kept, so "what about the next one?" means something


def strip_ask_wake(text: str) -> str:
    """Remove the first braillo wake word (and aliases) from a transcript; leftover is the question."""
    if not isinstance(text, str):
        return ""
    return _ASK_WAKE_RE.sub("", text, count=1).strip().strip(",:;!")


def looks_like_question(text: str) -> bool:
    """True when the leftover after the wake word is enough to send to the model."""
    words = text.split()
    return bool(words) and (len(words) >= 2 or text.endswith("?") or len(text) >= 8)


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

    def __init__(self, client: Optional[LLMClient] = None, timeout: float = 10.0):
        self.client, self.off_reason = client, ""
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

    def respond(self, heard: str, context: Optional[dict] = None) -> Optional[dict]:
        """What to do about free speech: {"action": one of TALK_ACTIONS, "say": text or ""}, or None if the helper is off or
        failed (the caller then falls back to its own "I didn't catch that"). Speech not meant for the tutor gives action none
        and nothing to say."""
        if not self.enabled:
            return None
        text = " ".join(str(heard or "").split()).strip()
        if not text:
            return None
        try:
            reply = self.client.chat_json(TALK_SYSTEM, self._prompt(text, context), TALK_SCHEMA)
        except Exception as e:
            self._failed(e)
            return None
        self.failures = 0
        action = reply.get("action") if reply.get("action") in TALK_ACTIONS else "none"
        if reply.get("addressed") is False:
            return {"action": "none", "say": ""}
        say = _spoken(reply.get("say")) or ""
        self._remember(text, say or f"({action})")
        return {"action": action, "say": say}
