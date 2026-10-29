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
        voice_io._echo_left.update(voice_io._echo_words(self.speaking))  # (as speak() does when a line starts)
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
    with _Fired(LINE) as fired:  # nor does talk that is not a command (other people in the room, say): it is answered afterwards
        voice_io._process_transcript_event("the letter d did you see the game last night", 0.99, False)
        assert not voice_io._interrupted.is_set() and fired == []
    with _Fired(LINE) as fired:  # its name does: they want it
        voice_io._process_transcript_event("braillo wait", 0.99, False)
        assert voice_io._interrupted.is_set()


def test_the_learner_is_heard_through_the_apps_own_voice() -> None:
    """While it talks, the transcript mixes both voices. Only the app's words are taken out: what is left is the learner."""
    with _Fired(LINE) as fired:  # a command in the middle of the app's line
        assert voice_io._process_transcript_event("the letter d dots one four and give me a hint", 0.6, True) == "hint"
        assert fired == ["hint"] and voice_io._interrupted.is_set()
    with _Fired(LINE) as fired:  # plain speech over the app: it carries on, and the learner's words are passed on as heard
        assert voice_io._process_transcript_event("the letter d can you explain that please dots one", 0.5, True) is None
        assert not voice_io._interrupted.is_set()
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


def test_its_own_name_coming_back_misspelt_is_not_the_wake_word() -> None:
    """From a real run: "Hi, I'm Braillo" came back through the laptop's microphone written another way, and was taken for the wake
    word. Every way of writing the name counts as the line's own; the learner saying it after is still heard."""
    offer = "Hi, I'm Braillo. Would you like me to talk you through this website by voice? Say yes, or no."
    with _Fired(offer) as fired:
        assert voice_io._process_transcript_event("hi i'm brello would you like", 0.9, True) is None
        assert fired == [] and not voice_io._interrupted.is_set()
        assert voice_io._process_transcript_event("braillo", 0.9, True) == "braillo"  # (the line has one name, already heard back)


def test_quitting_after_the_wake_word_is_quitting_and_a_question_is_still_a_question() -> None:
    """From a real run: "hey Braillo, quit" asked Braillo a question ("What's your question?") instead of quitting."""
    classify = lambda t: voice_io._classify_transcript(t, 0.95, True, None)[1]  # noqa: E731
    assert classify("hey, braillo. quit") == "stop"
    assert classify("hey braillo i wanna quit") == "stop"
    assert classify("braillo go home") == "menu"
    assert classify("braillo how do i learn") == "braillo"
    assert classify("braillo what does stop mean in braille please tell me") == "braillo"
    assert classify("one more") is None  # (not "no more": leaving needs a close match)
    assert classify("hey, braillo. i need a hint") == "hint"
    assert classify("hey, braillo. let's do read") == "read"
    assert classify("hey braillo what is a hint") == "braillo"
    assert classify("hey braillo can we stop") == "stop"


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
            assert not voice_io._interrupted.is_set()  # (heard, and answered once the line is done: not cut off)
        with _Fired(welcome):
            assert voice_io._process_transcript_event("guest", 0.9, True) is None
            assert voice_io._status["heard"]["text"] == "guest"  # the one word the line also says: still theirs
        with _Fired(welcome):
            before = voice_io._status["heard"]["n"]
            voice_io._process_transcript_event("to carry on without an account say guest", 0.9, True)
            assert voice_io._status["heard"]["n"] == before and not voice_io._interrupted.is_set()  # the line itself, leaking
    finally:
        voice_io._canceller = saved


def test_through_the_phone_only_a_command_cuts_it_off_and_its_own_words_are_taken_out() -> None:
    """The phone plays the line and listens: the laptop's canceller cannot take that echo out, and what comes back is often
    misheard. So three stray words do not cut the line off there (a command still does), and the line's own words are taken out
    of what is heard, one by one, even with the canceller running."""
    welcome = "Learn teaches the letters a few at a time. Read says a word aloud. Say learn, read, or quiz."
    saved = voice_io._canceller, voice_io.MIC_FROM_PHONE
    voice_io._canceller, voice_io.MIC_FROM_PHONE = object(), True
    try:
        with _Fired(welcome):
            voice_io._process_transcript_event("lean teaches the ladders a few", 0.9, False)  # (its own voice, misheard)
            assert not voice_io._interrupted.is_set(), "echo-like chatter does not cut Braillo off through the phone"
        with _Fired(welcome) as fired:
            voice_io._process_transcript_event("stop", 0.95, False)
            assert voice_io._interrupted.is_set(), "a command still does"
        with _Fired(welcome):
            assert voice_io._learner_words("say learn read or quiz please") == "please"  # (the line's words come out)
    finally:
        voice_io._canceller, voice_io.MIC_FROM_PHONE = saved


def test_played_on_the_phone_a_mode_name_in_its_own_line_does_not_cut_it_off() -> None:
    """The real case: the menu line says "Read says a word aloud", the phone plays it, the microphone hears "read says" back, and
    "read" is a command. With the sound on the phone that is its own voice: taken out, and nothing is cut off. A clear "stop" still
    cuts it off, whichever microphone hears it."""
    line = "Learn teaches the letters a few at a time. Read says a word aloud when you rest your finger on it. Say learn, read, or quiz."
    saved = voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE
    voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE = object(), False, True  # (laptop mic, phone speaker)
    try:
        with _Fired(line):
            voice_io._process_transcript_event("read says", 0.95, False)
            assert not voice_io._interrupted.is_set(), "its own words, played on the phone, do not cut it off"
        with _Fired(line):
            voice_io._process_transcript_event("quiz", 0.95, False)  # (a command, but one the line says: not a reason to stop)
            assert not voice_io._interrupted.is_set()
        with _Fired(line):
            voice_io._process_transcript_event("be quiet", 0.95, False)
            assert voice_io._interrupted.is_set(), "quiet still does"
    finally:
        voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE = saved


def test_its_own_line_leaking_past_the_canceller_misheard_does_not_cut_it_off() -> None:
    """From a real run: the laptop's canceller let "That's E. Now the word C" through as "that's me now the word c". That is its own
    voice, not the learner: it must not stop the line. The learner talking over it still does."""
    line = "That's E. Now the word CAB. Find the letter C."
    saved = voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE
    voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE = object(), False, False  # (the laptop does both)
    try:
        with _Fired(line):
            voice_io._process_transcript_event("that's me now the word c", 0.95, False)
            assert not voice_io._interrupted.is_set()
        with _Fired(line):
            voice_io._process_transcript_event("be quiet", 0.95, False)
            assert voice_io._interrupted.is_set(), "the learner, telling it to stop"
    finally:
        voice_io._canceller, voice_io.MIC_FROM_PHONE, voice_io.SPEAKER_ON_PHONE = saved


def test_a_command_word_it_has_just_said_is_its_own_echo_but_the_learner_saying_it_later_is_not() -> None:
    """From a real run: "Learn teaches the letters..." came back as "learn", and Braillo took it as the learner choosing Learn.
    A command word heard within a few seconds of Braillo saying it is his own voice; the same word at another moment is theirs."""
    import time as _t
    line = "Learn teaches the letters a few at a time. Read says a word aloud. Quiz names a letter. Say learn, read, or quiz."
    saved = voice_io._playing, voice_io._canceller
    voice_io._canceller = object()  # (the laptop's speaker and microphone, with its echo canceller: the usual setup)
    try:
        with _Fired(line) as fired:
            voice_io._playing = {"text": line, "started": _t.time() - 0.8, "seconds": 9.0}  # (just past "Learn teaches")
            assert voice_io._process_transcript_event("learn", 0.95, True) is None
            assert "learn" not in fired, "its own 'Learn', heard back"
        with _Fired(line) as fired:
            voice_io._callbacks["quiz"] = [lambda: fired.append("quiz")]
            voice_io._playing = {"text": line, "started": _t.time() - 0.8, "seconds": 9.0}  # ("quiz" is not said until the end)
            voice_io._process_transcript_event("quiz", 0.95, True)
            assert "quiz" in fired, "the learner choosing Quiz early"
    finally:
        voice_io._playing, voice_io._canceller = saved


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"Passed {len(tests)} offline barge-in and utterance checks.")


if __name__ == "__main__":
    main()
