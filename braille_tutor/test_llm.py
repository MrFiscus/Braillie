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

    def chat(self, system, user, max_tokens=None, temperature=None, timeout=None, quick=False):
        self.calls.append((system, user, max_tokens, temperature))
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


def fake_messages_api(reply_text="  hello  ", stop_reason="end_turn", status=200):
    """A local stand-in for Claude's Messages API: (server, what it saw). Answers every request with one text block."""
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["key"], seen["beta"] = self.path, self.headers["Authorization"], self.headers["anthropic-beta"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = {"id": "msg_1", "type": "message", "role": "assistant", "model": seen["body"]["model"],
                     "content": [{"type": "text", "text": reply_text}], "stop_reason": stop_reason, "stop_sequence": None,
                     "usage": {"input_tokens": 1, "output_tokens": 1}}
            if status != 200:
                reply = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
            out = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


class ClientTests(unittest.TestCase):
    def client(self, srv):
        return llm.LLMClient("sk-ant-test", "some-model", f"http://127.0.0.1:{srv.server_address[1]}")

    def test_request_shape_and_reply(self):
        srv, seen = fake_messages_api()
        try:
            self.assertEqual(self.client(srv).chat("be brief", "hi"), "hello")
        finally:
            srv.shutdown()
        self.assertEqual((seen["path"], seen["key"]), ("/v1/messages?beta=true", "Bearer sk-ant-test"))
        self.assertIn(llm.FALLBACK_BETA, seen["beta"])
        body = seen["body"]
        self.assertEqual(set(body), {"model", "max_tokens", "system", "messages", "output_config", "fallbacks"})  # nothing else is sent
        self.assertEqual((body["system"], body["messages"]), ("be brief", [{"role": "user", "content": "hi"}]))
        self.assertEqual((body["output_config"], body["fallbacks"]), ({"effort": llm.REPLY_EFFORT}, "default"))

    def test_haiku_is_asked_without_a_fallback_model(self):
        srv, seen = fake_messages_api()
        try:
            client = llm.LLMClient("sk-ant-test", "claude-haiku-5-5", f"http://127.0.0.1:{srv.server_address[1]}")
            self.assertEqual(client.chat("be brief", "hi"), "hello")
        finally:
            srv.shutdown()
        self.assertEqual(seen["path"], "/v1/messages")
        self.assertNotIn("fallbacks", seen["body"])

    def test_json_reply_follows_the_schema(self):
        srv, seen = fake_messages_api('{"action": "hint"}')
        try:
            self.assertEqual(self.client(srv).chat_json("s", "u", llm.TALK_SCHEMA), {"action": "hint"})
        finally:
            srv.shutdown()
        self.assertEqual(seen["body"]["output_config"]["format"], {"type": "json_schema", "schema": llm.TALK_SCHEMA["schema"]})

    def test_a_declined_or_cut_off_reply_is_an_error(self):
        for stop in ("refusal", "max_tokens"):
            srv, _ = fake_messages_api("half a sent", stop_reason=stop)
            try:
                with self.assertRaises(llm.LLMError):
                    self.client(srv).chat("s", "u")
            finally:
                srv.shutdown()

    def test_a_refused_key_switches_braillo_off_at_once(self):
        srv, _ = fake_messages_api(status=401)
        try:
            ask = llm.AskTutor(self.client(srv))
            self.assertIsNone(ask.answer("hello there"))
        finally:
            srv.shutdown()
        self.assertFalse(ask.enabled)

    def test_only_https_or_localhost(self):
        with self.assertRaises(ValueError):
            llm.LLMClient("k", "m", "http://api.example.com/v1")
        llm.LLMClient("k", "m", "https://api.example.com/v1")
        llm.LLMClient("k", "m", "http://localhost:9999/v1")

    def test_from_env(self):
        old = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "BRAILLIE_LLM_MODEL", "BRAILLIE_LLM_BASE_URL")}
        try:
            self.assertIsNone(llm.LLMClient.from_env())
            os.environ["ANTHROPIC_API_KEY"] = "sk-x"
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
        os.environ.pop("ANTHROPIC_API_KEY", None)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertIsNone(tutor.make_coach(argparse.Namespace(llm=False, llm_model=None)))
            self.assertIsNone(tutor.make_coach(argparse.Namespace(llm=True, llm_model=None)))  # no key: stays off
        self.assertIn("ANTHROPIC_API_KEY is not set", out.getvalue())
        os.environ["ANTHROPIC_API_KEY"] = "sk-x"
        try:
            with redirect_stdout(out):
                coach = tutor.make_coach(argparse.Namespace(llm=True, llm_model="m1"))
            self.assertTrue(coach.enabled)
            self.assertIn("never camera images", out.getvalue())
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


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


class AskStripTests(unittest.TestCase):
    def test_strip_wake_and_question_shape(self):
        self.assertEqual(llm.strip_ask_wake("braillo what is dot one"), "what is dot one")
        self.assertEqual(llm.strip_ask_wake("Briello, how many dots?"), "how many dots?")
        self.assertEqual(llm.strip_ask_wake("braillo"), "")
        self.assertTrue(llm.looks_like_question("what is a"))
        self.assertFalse(llm.looks_like_question(""))
        self.assertFalse(llm.looks_like_question("hi"))


class AskTutorTests(unittest.TestCase):
    def test_answer_is_short_spoken_text(self):
        client = Scripted("Dot one is top left on the cell.")
        ask = llm.AskTutor(client)
        self.assertEqual(ask.answer("what is dot one"), "Dot one is top left on the cell.")
        self.assertIsNone(client.calls[0][2], "no tight token cap: the model's thinking counts toward it")
        self.assertIn("Braillo", client.calls[0][0])

    def test_repeated_failures_turn_ask_off(self):
        client = Scripted(error=RuntimeError("down"))
        ask = llm.AskTutor(client)
        self.assertIsNone(ask.answer("hello there"))
        self.assertTrue(ask.enabled)  # one dropped request is not the end of Braillo
        ask.answer("hello again")
        ask.answer("and again")
        self.assertFalse(ask.enabled)
        self.assertIsNone(ask.answer("one more"))  # stays off; no more calls
        self.assertEqual(len(client.calls), 3)

    def test_answer_carries_the_moment_and_the_last_exchanges(self):
        client = Scripted("D is dots one, four and five.")
        ask = llm.AskTutor(client)
        ask.answer("what is d", context={"target_letter": "d", "target_dots": [1, 4, 5]})
        ask.answer("and the one after it", context={"target_letter": "d"})
        sent = json.loads(client.calls[1][1])
        self.assertEqual(sent["learner_said"], "and the one after it")
        self.assertEqual(sent["app_now"], {"target_letter": "d"})
        self.assertEqual(sent["earlier"], [{"learner": "what is d", "you": "D is dots one, four and five."}])


class JsonScripted(Scripted):
    """A fake client for respond(): chat_json returns `reply` (a dict) or raises."""

    def chat_json(self, system, user, schema):
        self.calls.append((system, user, schema, None))
        if self.error:
            raise self.error
        return self.reply(system, user) if callable(self.reply) else self.reply


class RespondTests(unittest.TestCase):
    def test_free_speech_becomes_a_command(self):
        ask = llm.AskTutor(JsonScripted({"addressed": True, "action": "repeat", "say": ""}))
        self.assertEqual(ask.respond("what was that letter again", {"state": "learning"}), {"action": "repeat", "say": "", "lesson": 0})

    def test_speech_not_meant_for_the_tutor_is_ignored(self):
        ask = llm.AskTutor(JsonScripted({"addressed": False, "action": "hint", "say": "Here is a hint."}))
        self.assertEqual(ask.respond("did you feed the cat"), {"action": "none", "say": "", "lesson": 0})

    def test_an_unknown_action_is_none_and_the_reply_is_spoken_plainly(self):
        ask = llm.AskTutor(JsonScripted({"addressed": True, "action": "dance", "say": "**Dot one** is top left."}))
        self.assertEqual(ask.respond("which dot is one"), {"action": "none", "say": "Dot one is top left.", "lesson": 0})

    def test_a_lesson_is_asked_for_by_its_number(self):
        ask = llm.AskTutor(JsonScripted({"addressed": True, "action": "lesson", "lesson": 3, "say": ""}))
        self.assertEqual(ask.respond("let's do K to O"), {"action": "lesson", "say": "", "lesson": 3})
        ask = llm.AskTutor(JsonScripted({"addressed": True, "action": "lesson", "lesson": 9, "say": ""}))
        self.assertEqual(ask.respond("lesson nine")["action"], "none", "no such lesson: nowhere to go")

    def test_before_an_action_that_talks_only_a_short_go_ahead_is_said(self):
        ask = llm.AskTutor(JsonScripted({"addressed": True, "action": "next lesson", "lesson": 0,
                                         "say": "Sure, let's move on! Lesson two covers F to J, so you'll find new dots."}))
        self.assertEqual(ask.respond("next one please")["say"], "Sure, let's move on!")

    def test_braillo_is_told_the_app_and_what_to_expect(self):
        for system in (llm.TALK_SYSTEM, llm.ASK_SYSTEM):
            for part in ("connect the phone", "waiting for", "Do not give away where a letter is", "my page"):
                self.assertIn(part, system)

    def test_failure_gives_none_so_the_tutor_falls_back(self):
        ask = llm.AskTutor(JsonScripted(error=RuntimeError("down")))
        self.assertIsNone(ask.respond("what now"))


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class BrailloSessionTests(unittest.TestCase):
    def test_same_utterance_answers_immediately(self):
        client = Scripted("Braille cells have six dots.")
        s, voice, _ = make_session(None)
        s.ask = llm.AskTutor(client)
        s.heard_fn = lambda: {"n": 3, "text": "braillo how many dots are in a cell", "matched": True}
        s.attach()
        voice.commands["braillo"]()
        self.assertIn("six dots", voice.said[-1])
        self.assertFalse(s._ask_armed)

    def test_wake_alone_waits_then_answers_next_utterance(self):
        client = Scripted("The letter A is only dot one.")
        s, voice, _ = make_session(None)
        s.ask = llm.AskTutor(client)
        heard = {"n": 1, "text": "braillo", "matched": True}
        s.heard_fn = lambda: heard
        s.on_braillo()
        self.assertTrue(s._ask_armed)
        self.assertIn("question", voice.said[-1].lower())
        heard.update({"n": 2, "text": "what is the letter a", "matched": False, "reason": "no command phrase matched"})
        s.watch_once(time.time())
        self.assertIn("dot one", voice.said[-1].lower())
        self.assertFalse(s._ask_armed)

    def test_no_key_explains_how_to_turn_it_on(self):
        s, voice, _ = make_session(None)
        s.ask = None
        s.heard_fn = lambda: {"n": 1, "text": "braillo what is braille", "matched": True}
        s.on_braillo()
        self.assertIn("Anthropic API key", voice.said[-1])

    def test_make_ask_tutor(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                self.assertIsNone(tutor.make_ask_tutor(argparse.Namespace(no_llm=False, llm_model=None)))
            self.assertIn("Braillo Q&A off", out.getvalue())
            os.environ["ANTHROPIC_API_KEY"] = "sk-x"
            with redirect_stdout(out):
                ask = tutor.make_ask_tutor(argparse.Namespace(no_llm=False, llm_model="m1"))
            self.assertTrue(ask.enabled)
            with redirect_stdout(out):
                ask = tutor.make_ask_tutor(argparse.Namespace(no_llm=False, llm_model=None))
            self.assertEqual((ask.client.model, ask.fast.model), (llm.TALK_MODEL, os.environ.get("BRAILLIE_LLM_MODEL", llm.DEFAULT_MODEL)),
                             "his conversation on the capable model, his own words for a line on the fast one")
            self.assertIsNone(tutor.make_ask_tutor(argparse.Namespace(no_llm=True, llm_model=None)))
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


def wait_for(pred, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class OwnWordsTests(unittest.TestCase):
    """Braillo says a line his own way, but never loses what the learner needs to hear or say."""
    LINE = "Choose an option. Say learn, read, or quiz."

    def test_his_own_words_are_used_when_they_keep_what_matters(self):
        ask = llm.AskTutor(Scripted("Right then, what'll it be? Say learn, read, or quiz."))
        self.assertEqual(ask.phrase(self.LINE, ["learn", "read", "quiz"], {"page": "modes"}, ["Hi, Guest!"]),
                         "Right then, what'll it be? Say learn, read, or quiz.")
        sent = json.loads(ask.client.calls[0][1])
        self.assertEqual((sent["line"], sent["keep"], sent["you_said_last"]), (self.LINE, ["learn", "read", "quiz"], ["Hi, Guest!"]))

    def test_a_reply_that_drops_a_word_or_runs_long_is_not_used(self):
        self.assertIsNone(llm.AskTutor(Scripted("Pick learn or read!")).phrase(self.LINE, ["learn", "read", "quiz"]))
        long = "Say learn, read, or quiz. " + "And so on and so forth " * 6
        self.assertIsNone(llm.AskTutor(Scripted(long)).phrase(self.LINE, ["learn", "read", "quiz"]))
        self.assertIsNone(llm.AskTutor(Scripted(error=TimeoutError("slow"))).phrase(self.LINE, ["learn"]))


@unittest.skipIf(WC is None, "backend dependencies missing (pip install pyspellchecker)")
class HintWordsTests(unittest.TestCase):
    HINT = "The letter D has three dots: top-left, top-right and middle-right."

    def keep(self):
        return tutor.hint_keep(self.HINT, {"looking_for": "the letter D"})

    def test_what_a_hint_must_keep(self):
        self.assertEqual(self.keep(), ["top-left", "top-right", "middle-right", "D"])

    def test_a_worded_hint_is_used_only_if_it_keeps_every_dot_and_gives_nothing_more_away(self):
        good = "You were on A. D is A plus two more: top-left, top-right and middle-right."
        self.assertEqual(llm.AskTutor(Scripted(good)).hint(self.HINT, self.keep(), {}), good)
        self.assertIsNone(llm.AskTutor(Scripted("D has top-left and top-right.")).hint(self.HINT, self.keep(), {}), "lost a dot")
        leak = "D is in row 1, column 4: top-left, top-right and middle-right."
        self.assertIsNone(llm.AskTutor(Scripted(leak)).hint(self.HINT, self.keep(), {}), "where it is, before the ladder says so")


class OwnWordsSessionTests(unittest.TestCase):
    def test_menu_lines_are_his_own_but_lesson_feedback_and_claudes_replies_are_exact(self):
        class Braillo:  # (a fake Claude: two ways of saying a line, made ahead of time; a line with numbers put another way now)
            enabled = True

            def __init__(self):
                self.asked_for_ways = 0

            def variants(self, line, keep, n=3):
                self.asked_for_ways += 1
                return ["Right then, what next? Say learn, read, or quiz.", "So, what shall we do? Say learn, read, or quiz."]

            def phrase(self, line, keep, context=None, recent=None):
                return "Nice work, five of six! " + line.split(". ", 1)[1]

        s, voice, _ = make_session(None)
        s.ask = Braillo()
        menu = "Choose an option. Say learn, read, or quiz."
        s.say(menu)
        self.assertEqual(voice.said[-1], menu, "the first time: as written, no wait (his own ways are made meanwhile)")
        self.assertTrue(wait_for(lambda: bool(s._ways.get(menu))))
        s.say(menu)
        s.say(menu)
        self.assertEqual(set(voice.said[-2:]), {"Right then, what next? Say learn, read, or quiz.", "So, what shall we do? Say learn, read, or quiz."},
                         "after that, his own ways, not the same one twice running")
        self.assertEqual(s.ask.asked_for_ways, 1, "made once")
        s.say("Okay. You found 5 of 6 letters today. Choose an option. Say learn, read, or quiz.")
        self.assertTrue(voice.said[-1].startswith("Nice work, five of six!"), "a line with this moment's numbers: put his way now")
        s.say(menu, natural=False)  # (a line Claude wrote already)
        self.assertEqual(voice.said[-1], menu)
        s.say("Okay.")  # (short: as it is)
        self.assertEqual(voice.said[-1], "Okay.")
        s.state = "asking"  # (in the middle of a quiz: feedback must be exact and instant)
        s.say("Correct! That's the letter J. Find the letter G next, please.")
        self.assertEqual(voice.said[-1], "Correct! That's the letter J. Find the letter G next, please.")
        self.assertEqual(tutor.keep_words("Say go back to return, or more options to hear the links."), ["go back", "more options"])

    def test_his_ways_of_saying_a_line_each_keep_what_matters(self):
        class Fast:
            model = "fast"

            def chat_json(self, system, user, schema, quick=False):
                return {"versions": ["What next? Say learn, read, or quiz.", "Pick one!", "Say learn, read, or quiz, whichever you like."]}

        ask = llm.AskTutor(Scripted("x"), fast=Fast())
        self.assertEqual(ask.variants("Choose an option. Say learn, read, or quiz.", ["learn", "read", "quiz"]),
                         ["What next? Say learn, read, or quiz.", "Say learn, read, or quiz, whichever you like."])



if __name__ == "__main__":
    unittest.main()
