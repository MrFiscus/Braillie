"""Optional LLM helper for the tutor (any OpenAI-compatible chat API). Standard library only.

Rules that keep it from ever hurting the app:
  * off unless asked for (--llm) and a key is set (OPENAI_API_KEY); with it off the tutor behaves exactly as before
  * nothing the learner waits for depends on it: extras are fetched in the background and are a lookup when needed; the
    one call that is waited on (the debrief) has a hard time limit and falls back to the built-in debrief
  * only lesson facts are sent (cell names, dot numbers, scores): never camera images, audio or names
  * replies are checked before use: no digits that were not in the facts, no position words that don't apply, length limits
  * after the first error or timeout the helper switches itself off for the rest of the session
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from typing import Optional
from urllib.parse import urlparse

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"  # override with --llm-model or BRAILLIE_LLM_MODEL: use any model your key can call
POSITION_WORDS = {"top", "middle", "bottom", "left", "right"}


class LLMClient:
    """A minimal chat-completions client. Only https (or a localhost address) is accepted."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, timeout: float = 8.0):
        url = urlparse(base_url)
        if url.scheme != "https" and url.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError("the LLM base URL must be https (or localhost)")
        self.api_key, self.model, self.base_url, self.timeout = api_key, model, base_url.rstrip("/"), timeout

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> Optional["LLMClient"]:
        """A client using OPENAI_API_KEY (and optional BRAILLIE_LLM_MODEL / BRAILLIE_LLM_BASE_URL), or None if no key."""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        return cls(key, model or os.environ.get("BRAILLIE_LLM_MODEL", DEFAULT_MODEL),
                   os.environ.get("BRAILLIE_LLM_BASE_URL", DEFAULT_BASE_URL))

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """One request; returns the reply text. Raises on any error."""
        payload: dict = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                           {"role": "user", "content": user}]}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """Request a strictly schema-shaped JSON reply from Chat Completions."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": schema},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            result = json.loads(r.read())["choices"][0]["message"]["content"]
        parsed = json.loads(result)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI returned a non-object adaptive plan")
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
ASK_SYSTEM = (
    "You are Braillo, the spoken tutor inside Braillie, a braille learning app for people who may be blind or have "
    "low vision. Answer like a real tutor who knows the topic. Prefer plain speech for text-to-speech: no markdown, "
    "bullet lists, emoji, or URLs. Keep answers under 60 words unless they clearly ask for more detail. Help with "
    "braille, reading, and the app when relevant, but answer any honest question."
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


class AskTutor:
    """Fast spoken Q&A via ChatGPT. Armed by the braillo wake word; answers are kept short for TTS."""

    def __init__(self, client: Optional[LLMClient] = None, timeout: float = 6.0):
        self.client, self.off_reason = client, ""
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

    def answer(self, question: str) -> Optional[str]:
        """Blocking ChatGPT reply fit to speak, or None. Failures switch this helper off for the session."""
        if not self.enabled:
            return None
        q = " ".join(str(question or "").split()).strip()
        if not q:
            return None
        try:
            raw = self.client.chat(ASK_SYSTEM, q, max_tokens=160, temperature=0.4)
        except Exception as e:
            self.off_reason = f"{type(e).__name__}: {e}"[:120]
            return None
        line = clean_line(raw, 480, allow_digits=True)
        if line:
            return line
        # Model used forbidden markup: still speak a compressed plain version if we can.
        plain = " ".join(str(raw or "").replace("\n", " ").split())
        plain = re.sub(r"[*_`#<>{}\[\]|~]", "", plain)
        return plain[:480] if plain else None
