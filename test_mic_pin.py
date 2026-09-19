# test_mic_pin.p:y
import time
from voice_io import register_command, start_listening, stop_listening

COMMANDS = ["repeat", "hint", "found it", "next", "stop", "start quiz"]

def make_handler(name):
    def handler():
        print(f"✅ Command detected: {name}")
    return handler

for cmd in COMMANDS:
    register_command(cmd, make_handler(cmd))

print("Listening... say a command (e.g. 'hint'). Press Ctrl-C to stop.")
start_listening()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_listening()
    print("\nStopped.")
