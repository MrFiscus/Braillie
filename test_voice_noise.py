"""Offline checks for listening in a noisy room: speech that does not stand out from the room is ignored, a command is judged
on its own words, near-misses are caught, a command word buried in someone's sentence is left for the app to judge, and the
microphone clean-up takes out hum and turns a quiet microphone up.

Run with: python3 test_voice_noise.py
"""
from __future__ import annotations

import numpy as np

import voice_io

RATE = voice_io.SAMPLE_RATE


def tone(db: float, seconds: float, hz: float = 440.0, seed: int = 0) -> bytes:
    """A noisy tone at about `db` dBFS: stands in for speech (loud) or room noise (quiet)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    x = np.sin(2 * np.pi * hz * t) + 0.3 * rng.standard_normal(t.size)
    x *= 10 ** (db / 20) * 32768 / np.sqrt(np.mean(x * x))
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


def feed(room: voice_io._RoomLevel, pcm: bytes) -> None:
    step = int(RATE * voice_io.CHUNK_MS / 1000) * 2
    for i in range(0, len(pcm), step):
        room.add(pcm[i:i + step])


def noisy_room() -> voice_io._RoomLevel:
    """5 s of room noise at -50 dBFS, then speech at -20 (5-6 s), then a far voice at -46 (6-7 s), then the room again."""
    room = voice_io._RoomLevel()
    for db, secs in ((-50, 5.0), (-20, 1.0), (-46, 1.0), (-50, 2.0)):
        feed(room, tone(db, secs))
    return room


class _State:
    """Clean module state for one check: recording callbacks, a given room, no speaking; all put back afterwards."""

    def __init__(self, room=None, defer=False):
        self.room, self.defer, self.fired = room or voice_io._RoomLevel(), defer, []

    def __enter__(self):
        self.saved = (voice_io._callbacks, voice_io._room, voice_io._defer_long, voice_io._speaking_text, voice_io._echo_text,
                      voice_io._mic_paused, dict(voice_io._status["heard"]))
        voice_io._callbacks = {c: [lambda c=c: self.fired.append(c)] for c in set(voice_io._COMMAND_MAP.values())}
        voice_io._room, voice_io._defer_long = self.room, self.defer
        voice_io._speaking_text = voice_io._echo_text = ""
        voice_io._mic_paused = False
        return self

    def __exit__(self, *_):
        (voice_io._callbacks, voice_io._room, voice_io._defer_long, voice_io._speaking_text, voice_io._echo_text,
         voice_io._mic_paused, heard) = self.saved
        voice_io._status["heard"] = heard


def words(text: str, conf: float = 0.95, start: float | None = None, each: float = 0.2, confs: dict | None = None) -> list:
    out, t = [], start
    for w in text.split():
        out.append((w, (confs or {}).get(w, conf), t, None if t is None else t + each))
        t = None if t is None else t + each
    return out


def test_the_room_level_and_a_near_voice() -> None:
    room = noisy_room()
    assert abs(room.floor() - (-50)) < 2, room.floor()
    assert abs(room.level(5.1, 5.9) - (-20)) < 2
    assert room.check(5.1, 5.9) is None  # someone at the microphone
    assert room.check(6.1, 6.9) == "too quiet against the room"  # talk across the room, 4 dB over the noise
    assert room.check(None, None) is None  # no timing: never ignored


def test_a_voice_much_quieter_than_the_learners_is_ignored() -> None:
    room = voice_io._RoomLevel()
    for db, secs in ((-60, 4.0), (-20, 1.0), (-38, 1.0)):
        feed(room, tone(db, secs))
    for _ in range(3):
        room.note_voice(4.1, 4.9)  # the learner speaks at about -20
    assert room.check(5.1, 5.9) == "much quieter than the learner's voice"  # 22 dB over the room, but far quieter than them


def test_far_speech_is_not_a_command_and_not_heard() -> None:
    with _State(noisy_room()) as s:
        before = voice_io._status["heard"]["n"]
        assert voice_io._process_transcript_event("hint", 0.99, True, words("hint", start=6.2)) is None
        assert s.fired == [] and voice_io._status["heard"]["n"] == before  # (so no "I didn't catch that" either)
        assert voice_io._process_transcript_event("hint", 0.99, True, words("hint", start=5.2)) == "hint"


def test_a_clear_command_is_not_sunk_by_murmur_around_it() -> None:
    with _State() as s:
        heard = words("um uh give me a hint", confs={"um": 0.2, "uh": 0.3})
        avg = sum(w[1] for w in heard) / len(heard)
        assert avg < voice_io.COMMAND_CONFIDENCE_THRESHOLD  # the old check would have dropped it
        assert voice_io._process_transcript_event("um uh give me a hint", avg, True, heard) == "hint"
        mumbled = words("hint", conf=0.4)
        assert voice_io._process_transcript_event("hint", 0.4, True, mumbled) is None  # a mumbled command still is not enough


def test_near_misses_are_caught_only_in_short_utterances() -> None:
    cases = {"hent": "hint", "nex": "next", "repeet": "repeat", "brailo please": "braillo",
             "the weather": None, "hello": None, "text": None, "shop": None,
             "it was a really nice hent in the end": None}
    with _State():
        for text, want in cases.items():
            found = voice_io._match_phrase(text) or voice_io._fuzzy_phrase(text)
            assert (found[0] if found else None) == want, (text, found, want)


def test_a_command_word_inside_someone_elses_sentence_is_left_to_the_app() -> None:
    sentence = "i need to stop by the shop later today"
    with _State(defer=True) as s:
        assert voice_io._process_transcript_event(sentence, 0.95, True, words(sentence)) is None
        assert voice_io._status["heard"] == {**voice_io._status["heard"], "text": sentence, "matched": False}
        long_ask = "okay could you please give me a hint now"  # a whole command phrase is still a command
        assert voice_io._process_transcript_event(long_ask, 0.95, True, words(long_ask)) == "hint"
        assert voice_io._process_transcript_event("stop", 0.95, True, words("stop")) == "stop"  # short: as before
    with _State(defer=False) as s:  # with nobody to judge the sentence, it is taken as before
        assert voice_io._process_transcript_event(sentence, 0.95, True, words(sentence)) == "stop"


def test_far_speech_does_not_cut_the_app_off() -> None:
    with _State(noisy_room()):
        voice_io._speaking_text = "Find the letter D."
        voice_io._echo_left.clear()
        voice_io._echo_left.update(voice_io._words(voice_io._speaking_text))
        voice_io._interrupted.clear()
        voice_io._process_transcript_event("next", 0.99, False, words("next", start=6.2))
        assert not voice_io._interrupted.is_set()
        voice_io._process_transcript_event("next", 0.99, False, words("next", start=5.2))
        assert voice_io._interrupted.is_set()
        voice_io._interrupted.clear()


def sine(db: float, seconds: float, hz: float) -> bytes:
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * hz * t) * 10 ** (db / 20) * 32768 * np.sqrt(2)).astype(np.int16).tobytes()


def test_the_cleaner_takes_out_hum_and_keeps_speech() -> None:
    step = int(RATE * voice_io.CHUNK_MS / 1000) * 2
    for hz, most_db, least_db in ((50, -31, -99), (1000, -19.5, -20.5), (300, -19.5, -21)):
        c, pcm = voice_io._MicCleaner(), sine(-20, 1.0, hz)
        out = b"".join(c.highpass(pcm[i:i + step]) for i in range(0, len(pcm), step))  # chunk by chunk, as it is sent
        db = voice_io._dbfs(out[len(out) // 2:])  # (after the filter has settled)
        assert least_db <= db <= most_db, (hz, db)  # hum: down 11 dB or more; speech frequencies: untouched


def test_the_cleaner_turns_a_quiet_microphone_up_but_never_down() -> None:
    c = voice_io._MicCleaner()
    quiet = tone(-44, 0.1, hz=1000)
    out = quiet
    for _ in range(60):
        out = c.gain(quiet, floor_db=-70)
    assert abs(c.gain_db - voice_io.AGC_MAX_GAIN_DB) < 0.5  # wanted +18 dB, capped at +12
    assert abs(voice_io._dbfs(out) - (-44 + voice_io.AGC_MAX_GAIN_DB)) < 1
    loud = tone(-10, 0.1, hz=1000)
    c = voice_io._MicCleaner()
    for _ in range(30):
        assert c.gain(loud, floor_db=-60) == loud
    c = voice_io._MicCleaner()
    noise = tone(-55, 0.1)
    for _ in range(30):
        c.gain(noise, floor_db=-56)  # the room itself (not speech) does not move the gain
    assert c.speech_db is None and c.gain_db == 0.0


def test_the_echo_canceller_takes_the_apps_voice_out_of_the_microphone() -> None:
    try:
        c = voice_io._EchoCanceller()
    except ImportError:
        print("  (skipped: the livekit package is not installed)")
        return
    rng = np.random.default_rng(1)
    far_rate, secs = 24000, 6
    t = np.arange(far_rate * secs) / far_rate
    far = (np.sin(2 * np.pi * 220 * t * (1 + 0.3 * np.sin(2 * np.pi * 0.7 * t))) + 0.5 * rng.standard_normal(t.size)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 9000
    far16 = np.interp(np.linspace(0, far.size - 1, RATE * secs), np.arange(far.size), far)
    delay = int(0.045 * RATE)
    echo = np.zeros_like(far16)
    echo[delay:] = 0.5 * far16[:-delay]  # the speakers, heard by the microphone a moment later and quieter
    near = np.clip(echo + rng.standard_normal(echo.size) * 60, -32768, 32767).astype(np.int16).tobytes()
    far_b = np.clip(far, -32768, 32767).astype(np.int16).tobytes()
    out, fi = bytearray(), 0
    for i in range(0, len(near), 3200):  # in step, as playback and listening run: 100 ms of each at a time
        while fi < (i // 3200 + 1) * 4800 and fi < len(far_b):
            c.played(far_b[fi:fi + 2048], far_rate, 1, 2)
            fi += 2048
        out += c.process(near[i:i + 3200])
    settled = slice(len(out) // 2, len(out))  # (after it has learnt the room)
    reduced = voice_io._dbfs(near[settled]) - voice_io._dbfs(bytes(out)[settled])
    assert reduced >= 12, f"the app's voice was only reduced by {reduced:.1f} dB"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"Passed {len(tests)} offline noise-handling checks.")


if __name__ == "__main__":
    main()
