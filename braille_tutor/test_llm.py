"""Tests for llm.py and the tutor's use of it. Run: python -m unittest test_llm -v

The LLM is always faked: these tests never call a real service. What they prove is that the helper can add polish
without ever making the tutor wait, break, or say something invented.
"""
import argparse
import http.server
import io
import json
import os
import random
import threading
import time
import unittest
from contextlib import redirect_stdout

import llm
import sheets
import tutor
from test_tutor import WC, FakeVoice


class Scripted:
    """A fake LLM client: records every request and answers with whatever `reply` returns (or raises)."""
    model = "fake-model"

    def __init__(self, reply=None, delay=0.0, error=None):
        self.reply, self.delay, self.error, self.calls = reply, delay, error, []

    def chat(self, system, user):
        self.calls.append((system, user))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reply(system, user) if callable(self.reply) else self.reply


def aids_reply(system, user):
    """What a well-behaved model returns for the prefetch request."""
    cells = json.loads(user)["cells"]
    return json.dumps({"aids": {c["name"]: f"Feel the {c['positions'][0].replace('-', ' ')} dot first." for c in cells},
                       "praise": ["Well done!", "Brilliant!", "Spot on!"], "cheer": ["Keep going!", "You are close!"]})


class ModeVoice(FakeVoice):
    def __init__(self):
        super().__init__()
        self.modes = []

    def speak(self, text, mode="normal"):
        self.modes.append(mode)
        super().speak(text, mode)


def make_session(coach, questions=2, voice=None):
    sh = sheets.get_sheet("alphabet")
    voice = voice or ModeVoice()
    s = tutor.TutorSession(voice, WC, sh.cells, finger=lambda: None, questions=questions, rng=random.Random(3),
                           names=sh.names, coach=coach)
    return s, voice, sh


def touch(s, sh, key):
    c = next(c for c in sh.cells if sh.symbol(c).key == key)
    s.finger = lambda: (c["x"], c["y"])


def wait_until(fn, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.02)
    return False


class ClientTests(unittest.TestCase):
    def test_request_shape_and_reply(self):
        seen = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"], seen["auth"] = self.path, self.headers["Authorization"]
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = json.dumps({"choices": [{"message": {"content": "  hello  "}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            client = llm.LLMClient("sk-test", "some-model", f"http://127.0.0.1:{srv.server_address[1]}/v1")
            self.assertEqual(client.chat("be brief", "hi"), "hello")
        finally:
            srv.shutdown()
        self.assertEqual((seen["path"], seen["auth"]), ("/v1/chat/completions", "Bearer sk-test"))
        self.assertEqual(set(seen["body"]), {"model", "messages"})  # nothing else is sent
        self.assertEqual([m["role"] for m in seen["body"]["messages"]], ["system", "user"])

    def test_only_https_or_localhost(self):
        with self.assertRaises(ValueError):
            llm.LLMClient("k", "m", "http://api.example.com/v1")
        llm.LLMClient("k", "m", "https://api.example.com/v1")
        llm.LLMClient("k", "m", "http://localhost:9999/v1")

    def test_from_env(self):
        old = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "BRAILLIE_LLM_MODEL", "BRAILLIE_LLM_BASE_URL")}
        try:
            self.assertIsNone(llm.LLMClient.from_env())
            os.environ["OPENAI_API_KEY"] = "sk-x"
            c = llm.LLMClient.from_env()
            self.assertEqual((c.model, c.base_url), (llm.DEFAULT_MODEL, llm.DEFAULT_BASE_URL))
            os.environ["BRAILLIE_LLM_MODEL"] = "other"
            self.assertEqual(llm.LLMClient.from_env().model, "other")
            self.assertEqual(llm.LLMClient.from_env("explicit").model, "explicit")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_make_coach(self):
        os.environ.pop("OPENAI_API_KEY", None)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertIsNone(tutor.make_coach(argparse.Namespace(llm=False, llm_model=None)))
            self.assertIsNone(tutor.make_coach(argparse.Namespace(llm=True, llm_model=None)))  # no key: stays off
        self.assertIn("OPENAI_API_KEY is not set", out.getvalue())
        os.environ["OPENAI_API_KEY"] = "sk-x"
        try:
            with redirect_stdout(out):
                coach = tutor.make_coach(argparse.Namespace(llm=True, llm_model="m1"))
            self.assertTrue(coach.enabled)
            self.assertIn("never camera images", out.getvalue())
        finally:
            del os.environ["OPENAI_API_KEY"]


class GuardTests(unittest.TestCase):
    def test_clean_line(self):
        ok = llm.clean_line
        self.assertEqual(ok("  Nice   work! ", 30), "Nice work!")
        self.assertIsNone(ok("Two dots: 1 and 4", 50))  # digits not allowed
        self.assertEqual(ok("Dots 1 and 4", 50, allow_digits=True), "Dots 1 and 4")
        for bad in ("**bold**", "see https://x.example", "a `b`", "x" * 200, "", None, 5, "line\n\nbreak" * 30):
            self.assertIsNone(ok(bad, 60), bad)
        self.assertIsNone(ok("Feel the bottom dot", 60, allowed_positions={"top", "left"}))
        self.assertEqual(ok("Feel the top dot", 60, allowed_positions={"top", "left"}), "Feel the top dot")

    def test_pending_timeout_and_error(self):
        self.assertEqual(llm.Pending(lambda: "x").get(1.0), "x")
        self.assertIsNone(llm.Pending(lambda: time.sleep(0.5) or "late").get(0.05))
        self.assertIsNone(llm.Pending(lambda: 1 / 0).get(1.0))


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class CoachInSessionTests(unittest.TestCase):
    def test_prefetch_gives_aids_praise_and_cheers_without_blocking(self):
        client = Scripted(aids_reply, delay=0.3)
        coach = llm.Coach(client)
        s, v, sh = make_session(coach)
        t0 = time.time()
        s.on_start()
        self.assertLess(time.time() - t0, 0.2, "starting the quiz must not wait for the LLM")
        self.assertTrue(wait_until(lambda: coach.praise and coach.aids), "background fetch never finished")
        # the request contained only lesson facts
        payload = json.loads(client.calls[0][1])
        self.assertEqual(set(payload), {"cells"})
        for cell in payload["cells"]:
            self.assertEqual(set(cell), {"name", "dots", "positions"})
        # hint 1 = facts, hint 2 = position, hint 3 = the memory aid
        target = s.items[0]
        s.on_hint()
        s.on_hint()
        s.on_hint()
        self.assertTrue(v.said[-1].startswith("Feel the "), v.said[-3:])
        s.on_hint()  # a fourth falls back to the position hint
        self.assertIn("row", v.said[-1])
        # praise on a right answer, cheer only on the first wrong one
        wrong = "a" if target != "a" else "b"
        touch(s, sh, wrong)
        s.on_found_it()
        self.assertTrue(v.said[-1].endswith(" Try again.") and any(c in v.said[-1] for c in ("Keep going!", "You are close!")), v.said[-1])
        s.on_found_it()
        self.assertFalse(any(c in v.said[-1] for c in ("Keep going!", "You are close!")), "cheer only after the first miss")
        touch(s, sh, target)
        s.on_found_it()
        self.assertTrue(any(f"Correct! That's the letter {target.upper()}. " in t and t.rstrip().endswith(("!", ".")) for t in v.said))
        self.assertTrue(any(p in " ".join(v.said) for p in ("Well done!", "Brilliant!", "Spot on!")))

    def test_bad_replies_are_filtered_not_used(self):
        def reply(system, user):
            cells = json.loads(user)["cells"]
            aids = {}
            for i, c in enumerate(cells):
                aids[c["name"]] = ["Feel the bottom dot first.",  # a position that isn't in this cell
                                   "Think of dots 1 and 4.",  # digits
                                   "x" * 300,  # too long
                                   f"Feel the {c['positions'][0].replace('-', ' ')} dot first."][i % 4]
            return "```json\n" + json.dumps({"aids": aids, "praise": ["Great!", "Number 1!", "**bold**"], "cheer": []}) + "\n```"

        coach = llm.Coach(Scripted(reply))
        s, v, sh = make_session(coach, questions=8)
        s.on_start()
        self.assertTrue(wait_until(lambda: coach.praise))
        self.assertEqual(coach.praise, ["Great!"])  # digits and markdown dropped
        cells = {c["name"]: c for c in s._aid_requests()}
        self.assertGreater(len(cells), len(coach.aids), "some replies must have been filtered out")
        self.assertGreater(len(coach.aids), 0)
        for name, aid in coach.aids.items():
            self.assertNotRegex(aid, r"\d")
            self.assertLess(len(aid), 111)
            used = {w for w in ("top", "middle", "bottom", "left", "right") if w in aid.lower().split() or w in aid.lower()}
            allowed = {w for p in cells[name]["positions"] for w in p.split("-")}
            self.assertLessEqual(used, allowed, f"{name}: {aid!r} mentions a position this cell doesn't have")

    def test_failure_switches_the_helper_off_and_the_tutor_carries_on(self):
        coach = llm.Coach(Scripted(error=ConnectionError("no wifi")))
        s, v, sh = make_session(coach, questions=1)
        s.on_start()
        self.assertTrue(wait_until(lambda: not coach.enabled))
        self.assertIn("ConnectionError", coach.status)
        touch(s, sh, s.items[0])
        s.on_found_it()
        self.assertTrue(v.said[-1].startswith("Correct!"))
        self.assertEqual(v.said[-1], f"Correct! That's the letter {s.items[0].upper()}.")  # exactly the built-in wording
        self.assertEqual(len(v.debriefs), 1)  # and the built-in debrief

    def test_a_slow_llm_never_delays_answers_and_the_debrief_falls_back(self):
        coach = llm.Coach(Scripted(aids_reply, delay=3.0), debrief_wait=0.4)
        s, v, sh = make_session(coach, questions=1)
        s.on_start()
        touch(s, sh, s.items[0])
        t0 = time.time()
        s.on_found_it()  # the debrief starts here, and the LLM is slow
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.5, f"the debrief wait must be capped, took {elapsed:.1f}s")
        self.assertIn("Let's see how you did.", v.said)  # the lead-in that hides the wait
        self.assertEqual(len(v.debriefs), 1)  # fell back to the built-in debrief
        self.assertFalse(coach.enabled)
        self.assertIn("did not arrive in time", coach.status)
        self.assertNotIn("debrief", v.modes)

    def test_personal_debrief_is_spoken_in_the_debrief_voice(self):
        seen = {}

        def reply(system, user):
            if system.startswith("You write the short spoken"):
                seen["facts"] = json.loads(user)
                return "You got everything right today, which is lovely to hear. Keep it up!"
            return aids_reply(system, user)

        s, v, sh = make_session(llm.Coach(Scripted(reply)), questions=1)
        s.on_start()
        touch(s, sh, s.items[0])
        s.on_found_it()
        self.assertIn("You got everything right today, which is lovely to hear. Keep it up!", v.said)
        self.assertEqual(v.modes[-1], "debrief")
        self.assertEqual(v.debriefs, [])  # the LLM text replaced the built-in script
        self.assertEqual(s.last_debrief["accuracy"], 1.0)
        self.assertIn("lovely", s.last_debrief["text"])
        self.assertEqual(set(seen["facts"]), {"accuracy_percent", "questions_answered", "correct", "trouble"})

    def test_debrief_facts_carry_the_confusions(self):
        seen = {}

        def reply(system, user):
            if system.startswith("You write the short spoken"):
                seen["facts"] = json.loads(user)
                return "Good effort. Try feeling the shapes slowly."
            return aids_reply(system, user)

        s, v, sh = make_session(llm.Coach(Scripted(reply)), questions=1, voice=ModeVoice())
        s.max_tries = 1
        s.on_start()
        target = s.items[0]
        touch(s, sh, "a" if target != "a" else "b")
        s.on_found_it()
        trouble = seen["facts"]["trouble"][0]
        self.assertEqual(trouble["name"], f"the letter {target.upper()}")
        self.assertEqual(len(trouble["mistaken_for"]), 1)
        self.assertEqual(seen["facts"]["accuracy_percent"], 0)

    def test_invented_numbers_in_the_debrief_are_rejected(self):
        def reply(system, user):
            if system.startswith("You write the short spoken"):
                return "Almost! Remember the letter has dots 1, 2 and 6, and you scored 99 percent."  # not in the facts
            return aids_reply(system, user)

        s, v, sh = make_session(llm.Coach(Scripted(reply)), questions=1)
        s.on_start()
        touch(s, sh, s.items[0])
        s.on_found_it()
        self.assertEqual(len(v.debriefs), 1)  # built-in debrief instead
        self.assertFalse(any("99 percent" in t for t in v.said))

    def test_coach_off_changes_nothing(self):
        s, v, sh = make_session(None, questions=1)
        s.on_start()
        s.on_hint()
        s.on_hint()
        s.on_hint()
        self.assertIn("row", v.said[-1])
        touch(s, sh, s.items[0])
        s.on_found_it()
        self.assertEqual(v.said[-1], f"Correct! That's the letter {s.items[0].upper()}.")
        self.assertEqual(len(v.debriefs), 1)


if __name__ == "__main__":
    unittest.main()
