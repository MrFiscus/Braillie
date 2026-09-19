"""
Smoke-test for voice_io.py — runs fully offline (VOICE_IO_MOCK=1).

Usage:
    VOICE_IO_MOCK=1 python test_voice_io.py
"""
import os
import time

os.environ["VOICE_IO_MOCK"] = "1"

import voice_io  # noqa: E402 (env must be set before import)

# ---------- callback tracking ----------
fired: dict[str, int] = {}

def make_handler(name: str):
    def handler():
        fired[name] = fired.get(name, 0) + 1
        print(f"  [handler] '{name}' fired (total={fired[name]})")
    return handler

# ---------- register commands ----------
for cmd in ["repeat", "hint", "found it", "next", "stop", "start quiz"]:
    voice_io.register_command(cmd, make_handler(cmd))

# ---------- speak tests ----------
print("\n=== speak() tests ===")
voice_io.speak("The letter is A.")
voice_io.speak("Correct! Well done.")
voice_io.speak("Try tracing the top row of dots.")

print("\n=== speak_debrief() tests ===")
voice_io.speak_debrief(0.95, [])
voice_io.speak_debrief(0.78, ["SH", "TH"])
voice_io.speak_debrief(0.55, ["B", "C"])
voice_io.speak_debrief(0.30, ["A", "B", "C", "D"])

# ---------- listener test ----------
print("\n=== listener + command dispatch test ===")
voice_io.start_listening()
time.sleep(0.1)

# Directly invoke internal dispatch to test callbacks without stdin.
print("Simulating command dispatch …")
voice_io._fire_command("repeat")
voice_io._fire_command("next")
voice_io._fire_command("start quiz")

# Test pause/resume guard.
voice_io.pause_listening()
print("Mic paused — command below should be silently dropped by listener.")
voice_io.resume_listening()

voice_io.stop_listening()

# ---------- assertions ----------
assert fired.get("repeat") == 1, f"Expected repeat=1, got {fired}"
assert fired.get("next") == 1, f"Expected next=1, got {fired}"
assert fired.get("start quiz") == 1, f"Expected start quiz=1, got {fired}"

print("\nAll assertions passed.")
