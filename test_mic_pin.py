"""
Live microphone integration test for voice_io.py.

Usage:
    python3 test_mic_pin.py                  # run until Ctrl-C
    VOICE_IO_DEBUG=1 python3 test_mic_pin.py # also print every raw Deepgram transcript
    VOICE_IO_TEST_DURATION=60 python3 test_mic_pin.py  # auto-stop after N seconds
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import voice_io
from voice_io import register_command, start_listening, stop_listening

DEBUG = os.getenv("VOICE_IO_DEBUG", "0").strip() in ("1", "true", "yes")
DURATION = int(os.getenv("VOICE_IO_TEST_DURATION", "0"))

COMMANDS = ["repeat", "hint", "found it", "next", "stop", "start quiz"]

fired: dict[str, int] = {}

def make_handler(name: str):
    def handler():
        fired[name] = fired.get(name, 0) + 1
        print(f"✅ Command detected: {name!r}  (total fires: {fired[name]})", flush=True)
    return handler

for cmd in COMMANDS:
    register_command(cmd, make_handler(cmd))

print()
print("=" * 60)
if DEBUG:
    print("  DEBUG MODE ON  (VOICE_IO_DEBUG=1)")
    print("  Every Deepgram transcript will be printed as [DG] ...")
else:
    print("  Run with VOICE_IO_DEBUG=1 to see raw Deepgram output")
if DURATION:
    print(f"  Auto-stopping after {DURATION}s  (VOICE_IO_TEST_DURATION={DURATION})")
print("=" * 60)
print()
print("Commands registered:", COMMANDS)
print()

for i in (3, 2, 1):
    print(f"  Starting in {i}...", flush=True)
    time.sleep(1)
print("  🎙  LISTENING — say a command now\n", flush=True)

start_listening()

try:
    if DURATION:
        time.sleep(DURATION)
        raise KeyboardInterrupt
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

stop_listening()

print("\n" + "=" * 60)
print("SUMMARY — commands fired this session:")
if fired:
    for cmd, count in sorted(fired.items()):
        print(f"  {cmd!r}: {count}x")
else:
    print("  (none)")
print("=" * 60)
