"""Offline checks for talking over the app (barge-in), ignoring its own voice, and hearing a sentence whole.

Run with: python3 test_voice_barge_in.py
"""
from __future__ import annotations

import threading

import voice_io


class _Fired:
    """Swap in recording callbacks and a clean speaking state; put everything back afterwards."""

    def __init__(self, speaking: str = "") -> None:
        self.speaking, self.fired = speaking, []

    def __enter__(self) -> list[str]:
        self.saved = (voice_io._callbacks, voice_io._mic_paused, voice_io._speaking_text, voice_io._echo_text,
                      voice_io._echo_until, voice_io._pieces, dict(voice_io._status["heard"]))
        voice_io._status["heard"] = {"n": 0, "text": "", "matched": False, "reason": None}  # (each check starts with nothing heard)
        voice_io._callbacks = {c: [lambda c=c: self.fired.append(c)]
                               for c in ("hint", "repeat", "stop", "hush", "braillo", "next")}
        voice_io._mic_paused = False
        voice_io._speaking_text = voice_io._echo_text = self.speaking
        voice_io._echo_left.clear()
        voice_io._echo_left.update(voice_io._words(self.speaking))  # (as speak() does when a line starts)
        voice_io._echo_until = 0.0
        voice_io._pieces = []
        voice_io._interrupted.clear()
        return self.fired

    def __exit__(self, *_args: object) -> None:
        (voice_io._callbacks, voice_io._mic_paused, voice_io._speaking_text, voice_io._echo_text,
         voice_io._echo_until, voice_io._pieces, voice_io._status["heard"]) = self.saved
        voice_io._interrupted.clear()


LINE = "The letter D. Dots one, four and five. Say repeat to hear it again."


def test_its_own_voice_is_ignored() -> None:
    with _Fired(LINE) as fired:
        assert voice_io._process_transcript_event("dots one four and five", 0.99, True) is None
        assert voice_io._process_transcript_event("say repeat to hear it again", 0.99, True) is None
        assert fired == [] and not voice_io._interrupted.is_set()


def test_a_command_cuts_it_off_and_runs() -> None:
    with _Fired(LINE) as fired:
        cuts = voice_io.interrupt_count()
        assert voice_io._process_transcript_event("give me a hint", 0.99, True) == "hint"
        assert fired == ["hint"]
        assert voice_io._interrupted.is_set() and voice_io.interrupt_count() == cuts + 1


def test_stop_while_talking_only_stops_the_talking() -> None:
    with _Fired(LINE) as fired:
        assert voice_io._process_transcript_event("Stop.", 0.99, True) == "hush"
        assert fired == ["hush"]
    with _Fired("") as fired:  # quiet: "stop" ends the lesson as before
        assert voice_io._process_transcript_event("stop", 0.99, True) == "stop"
    with _Fired(LINE) as fired:  # "I'm done" still means done
        assert voice_io._process_transcript_event("I'm done", 0.99, True) == "stop"


def test_quiet_words_are_commands() -> None:
    for text in ("quiet", "be quiet please", "wait", "hold on", "stop talking"):
        with _Fired(LINE) as fired:
            voice_io._process_transcript_event(text, 0.99, True)
            assert fired == ["hush"], (text, fired)


def test_an_interim_command_cuts_it_off_at_once_but_runs_only_when_final() -> None:
    with _Fired(LINE) as fired:
        voice_io._process_transcript_event("hint", 0.99, False)
        assert voice_io._interrupted.is_set() and fired == []
    with _Fired(LINE) as fired:  # its own voice coming back, or a stray word, does not cut it off
        voice_io._process_transcript_event("dots one four", 0.99, False)
        voice_io._process_transcript_event("the letter d uh", 0.99, False)
        assert not voice_io._interrupted.is_set()
    with _Fired(LINE) as fired:  # the learner talking (three words of their own) does
        voice_io._process_transcript_event("the letter d hang on what was", 0.99, False)
        assert voice_io._interrupted.is_set() and fired == []


def test_the_learner_is_heard_through_the_apps_own_voice() -> None:
    """While it talks, the transcript mixes both voices. Only the app's words are taken out: what is left is the learner."""
    with _Fired(LINE) as fired:  # a command in the middle of the app's line
        assert voice_io._process_transcript_event("the letter d dots one four and give me a hint", 0.6, True) == "hint"
        assert fired == ["hint"] and voice_io._interrupted.is_set()
    with _Fired(LINE) as fired:  # plain speech over the app: it stops, and the learner's words are passed on as heard
        assert voice_io._process_transcript_event("the letter d can you explain that please dots one", 0.5, True) is None
        assert voice_io._interrupted.is_set()
        assert voice_io._status["heard"]["text"] == "can you explain that please"
        assert voice_io._status["heard"]["matched"] is False
    with _Fired(LINE) as fired:  # the app misheard by a word or two: kept as heard, marked, and it carries on talking
        assert voice_io._process_transcript_event("the letter d dots one for and fife", 0.9, True) is None
        assert not voice_io._interrupted.is_set() and fired == []
        assert voice_io._status["heard"]["text"] == "for fife"
        assert voice_io._status["heard"]["reason"] == voice_io.OVER_SPEECH


def test_a_word_the_line_also_says_is_still_heard_from_the_learner() -> None:
    welcome = "To sign in, say Google. To carry on without an account, say guest."
    with _Fired(welcome) as fired:
        # the app's own "say guest" comes back through the microphone first: that one is the app's
        assert voice_io._process_transcript_event("to carry on without an account say guest", 0.9, True) is None
        assert voice_io._status["heard"]["text"] != "guest"
        # then the learner says "guest" over the rest of the line: the line has only one "guest", already heard, so this one is theirs
        assert voice_io._process_transcript_event("guest", 0.9, True) is None
        assert voice_io._status["heard"]["text"] == "guest"
        assert voice_io._status["heard"]["reason"] == voice_io.OVER_SPEECH
        assert not voice_io._interrupted.is_set()  # it keeps talking; the page decides it was an answer
    with _Fired(welcome) as fired:  # said before the app reached that word: taken for the app's at first...
        voice_io._process_transcript_event("guest", 0.9, True)
        # ...and the app's own "guest", arriving after, is then the one left over: it still comes out as heard
        voice_io._process_transcript_event("without an account say guest", 0.9, True)
        assert voice_io._status["heard"]["text"] == "guest"


def test_echo_tail_after_the_line_ends() -> None:
    with _Fired("") as fired:
        voice_io._echo_text = LINE
        voice_io._echo_left.update(voice_io._words(LINE))  # (set when the line started)
        voice_io._echo_until = voice_io.time.monotonic() + 5
        assert voice_io._process_transcript_event("say repeat", 0.99, True) is None  # (would be "repeat" if it were the learner)
        voice_io._echo_until = voice_io.time.monotonic() - 1
        assert voice_io._process_transcript_event("repeat", 0.99, True) == "repeat"


def test_only_the_speaking_thread_is_cut() -> None:
    with _Fired(LINE):
        voice_io.interrupt()
        assert not voice_io.playback_interrupted()  # (this thread is not inside speak(): an earcon, say)
        seen = []

        def speaker() -> None:
            voice_io._speaker_thread.speaking = True
            seen.append(voice_io.playback_interrupted())

        t = threading.Thread(target=speaker)
        t.start()
        t.join()
        assert seen == [True]


def test_a_sentence_in_pieces_is_heard_whole() -> None:
    with _Fired("") as fired:
        heard_before = voice_io._status["heard"]["n"]
        assert voice_io._final_piece("braillo why is d", [0.9], True) is None  # wake word: hear the question out
        assert voice_io._final_piece("different from f", [0.95], True) is None
        assert voice_io._utterance_end() == "braillo"
        assert fired == ["braillo"]
        assert voice_io._status["heard"]["text"] == "braillo why is d different from f"
        assert voice_io._status["heard"]["n"] == heard_before + 1


def test_a_command_piece_is_acted_on_without_waiting() -> None:
    with _Fired("") as fired:
        assert voice_io._final_piece("next", [0.99], True) == "next"
        assert fired == ["next"] and voice_io._pieces == []


def test_a_short_answer_is_taken_as_soon_as_the_speaker_pauses() -> None:
    with _Fired("") as fired:
        before = voice_io._status["heard"]["n"]
        voice_io._final_piece("guest", [0.95], True)  # no waiting for the end of the utterance: it is an answer
        assert voice_io._status["heard"]["n"] == before + 1 and voice_io._status["heard"]["text"] == "guest"
        assert voice_io._pieces == []
        voice_io._final_piece("what is the letter", [0.95], True)  # longer: it may go on, so it waits
        assert voice_io._status["heard"]["text"] == "guest" and voice_io._pieces != []
        voice_io._pieces = []


def test_a_piece_without_utterance_end_is_let_go() -> None:
    with _Fired("") as fired:
        voice_io._final_piece("what does this one", [0.9], True)
        assert voice_io._flush_stale_utterance(now=voice_io._pieces_at + 0.5) is None
        voice_io._flush_stale_utterance(now=voice_io._pieces_at + voice_io.UTTERANCE_HOLD_SECONDS + 0.1)
        assert voice_io._pieces == [] and voice_io._status["heard"]["text"] == "what does this one"


def test_with_the_echo_canceller_the_learners_words_are_kept_whole() -> None:
    """The canceller has already taken the app's voice out of the sound, so nothing is picked out of what the learner says, even
    words the line also says. Only a transcript that is plainly the line leaking through is dropped."""
    welcome = "To sign in, say Google. To carry on without an account, say guest."
    saved = voice_io._canceller
    voice_io._canceller = object()  # (stands in for a running canceller)
    try:
        with _Fired(welcome):
            assert voice_io._process_transcript_event("i would like to carry on as a guest", 0.9, True) is None
            assert voice_io._status["heard"]["text"] == "i would like to carry on as a guest"  # (most of it is the line's words too)
            assert voice_io._interrupted.is_set()  # they are talking to it: it stops to listen
        with _Fired(welcome):
            assert voice_io._process_transcript_event("guest", 0.9, True) is None
            assert voice_io._status["heard"]["text"] == "guest"  # the one word the line also says: still theirs
        with _Fired(welcome):
            before = voice_io._status["heard"]["n"]
            voice_io._process_transcript_event("to carry on without an account say guest", 0.9, True)
            assert voice_io._status["heard"]["n"] == before and not voice_io._interrupted.is_set()  # the line itself, leaking
    finally:
        voice_io._canceller = saved


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"Passed {len(tests)} offline barge-in and utterance checks.")


if __name__ == "__main__":
    main()
