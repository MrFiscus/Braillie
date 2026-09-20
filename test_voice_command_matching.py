"""Offline checks for final-transcript voice-command recognition.

Run with: python3 test_voice_command_matching.py
"""
from __future__ import annotations

import voice_io


def _run_event(text: str, confidence: float = 0.99, is_final: bool = True) -> list[str]:
    fired: list[str] = []
    original_callbacks = voice_io._callbacks
    original_paused = voice_io._mic_paused
    try:
        voice_io._callbacks = {
            command: [lambda command=command: fired.append(command)]
            for command in ("hint", "found it", "start quiz", "stop", "braillo", "learn")
        }
        voice_io._mic_paused = False
        voice_io._process_transcript_event(text, confidence, is_final)
    finally:
        voice_io._callbacks = original_callbacks
        voice_io._mic_paused = original_paused
    return fired


def main() -> None:
    cases = [
        ("hint", 0.99, True, ["hint"]),
        ("give me a hint", 0.99, True, ["hint"]),
        ("I found it", 0.99, True, ["found it"]),
        ("start quiz", 0.99, True, ["start quiz"]),
        ("please start quiz now", 0.99, True, ["start quiz"]),
        ("  HINT?!  ", 0.99, True, ["hint"]),
        ("hint", 0.10, True, []),
        ("hint", 0.99, False, []),
        ("I enjoy practicing braille today", 0.99, True, []),
        ("stopping", 0.99, True, []),
        ("finish", 0.99, True, ["stop"]),
        ("I'm finished", 0.99, True, ["stop"]),
        ("done", 0.99, True, ["stop"]),
        ("braillo", 0.99, True, ["braillo"]),
        ("braillo how do I learn", 0.99, True, ["braillo"]),  # wake word wins over "learn"
        ("briello what is quiz mode", 0.99, True, ["braillo"]),
    ]
    for text, confidence, is_final, expected in cases:
        actual = _run_event(text, confidence, is_final)
        assert actual == expected, (
            f"{text!r}, confidence={confidence}, is_final={is_final}: "
            f"expected {expected}, got {actual}"
        )
    print(f"Passed {len(cases)} offline command-matching checks.")


if __name__ == "__main__":
    main()
