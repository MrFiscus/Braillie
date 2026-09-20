# Braillie

A braille literacy tutor using a smartphone camera, hand tracking, and voice I/O.
The user keeps both hands on a physical braille page throughout — all input is voice.

---

## Repo layout

```
braille_tutor/     Braille cell detection (YOLOv8, camera, page registration)
backend/
  braillie/
    interfaces.py      Canonical detection contract (read this first)
    mock_detector.py   Stub detector for dev/testing without a camera
    session.py         Read mode + quiz mode logic
    server.py          WebSocket server (frontend connection point)
    word_correction.py Braille word validation and re-detection
voice_io.py        Deepgram STT/TTS + ElevenLabs debrief (standalone module)
run_server.py      Entry point — starts everything
website-frontend/  React/Vite frontend (connect to ws://localhost:8765)
```

---

## Running it today (no camera, no frontend needed)

```bash
# 1. Install backend deps
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cd ..

# 2. Set API keys (or skip audio entirely with VOICE_IO_MOCK=1)
export DEEPGRAM_API_KEY=your_key_here
export ELEVENLABS_API_KEY=your_key_here   # only needed for end-of-session debrief

# 3. Start the server with mock detector and mock voice
VOICE_IO_MOCK=1 python3 run_server.py --mock

# With real audio (Deepgram):
python3 run_server.py --mock
```

The server prints `Braillie backend ready  ws://0.0.0.0:8765` when it's up.
A mock tracking loop runs automatically in `--mock` mode, feeding one letter every
3 seconds to simulate a finger moving across the page.

```bash
# Override the mock interval or quiz letter set:
python3 run_server.py --mock --mock-interval 5 --quiz-letters abcdefghij
```

### Quick WebSocket smoke-test from the terminal

```bash
pip install websockets   # if not already installed
python3 - <<'EOF'
import asyncio, json
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        for _ in range(5):
            msg = json.loads(await ws.recv())
            print(msg)

asyncio.run(main())
EOF
```

---

## For the detection teammate

**File to read:** `backend/braillie/interfaces.py`

You need to implement the `CellDetector` protocol — one method:

```python
def get_cell_at(self, x_mm: float, y_mm: float) -> Optional[Cell]:
    ...
```

### What it receives
- `x_mm`, `y_mm`: fingertip position in **page millimetres**, origin at the
  top-left corner of the registered page.  Provided by the hand-tracking module.

### What it must return
A `Cell` dict (or `None` if no cell is close enough):

```python
{
    "x": 23.4,         # cell centre x, page mm
    "y": 45.1,         # cell centre y, page mm
    "w": 6.0,          # bounding box width, page mm
    "h": 10.0,         # bounding box height, page mm
    "label": "100000", # 6-char binary dot string; dot 1 = index 0
    "char": "⠁",       # Unicode braille character (chr(0x2800 + dot_bits))
    "dots": frozenset({1}),  # active dot numbers 1..6
    "confidence": 0.97,
    "row": 0,          # 0-based row in the page grid
    "col": 0,          # 0-based column
}
```

`braille_tutor.detect.Cell` already has this exact shape — return it directly.
No conversion needed.

### Minimal concrete implementation

```python
# braillie_tutor_adapter.py  (place at repo root, imported by run_server.py)
from typing import Optional
from braillie.interfaces import Cell
from braille_tutor.detect import nearest_cell   # your module
from braille_tutor.vote import CellVoter        # your module

class RealDetector:
    def __init__(self):
        self._voter = CellVoter()
        self._latest_cells: list = []

    def push_scan(self, cells: list) -> None:
        """Call this from your camera/detection loop after each scan."""
        self._voter.add(cells)
        self._latest_cells = self._voter.result()

    def get_cell_at(self, x_mm: float, y_mm: float) -> Optional[Cell]:
        return nearest_cell(self._latest_cells, x_mm, y_mm)
```

Then pass `RealDetector()` to `TutorSession` and call `push_scan()` from your
camera loop.

### Thread safety
`get_cell_at` may be called from two threads (tracking loop + voice command
thread).  Using a voted snapshot list as above is fine — list reads in CPython
are GIL-protected.  Add a `threading.Lock` if you need stronger guarantees.

### No inheritance needed
Python Protocols are structural.  Your class does not need to import or extend
anything from `braillie.interfaces`.  If it has a `get_cell_at` method with the
right signature, it satisfies `CellDetector`.

---

## For the frontend teammate

**WebSocket URL:** `ws://localhost:8765`  (default port; change via `WS_PORT=NNNN`)

Connect and listen.  The backend pushes JSON whenever state changes.
No authentication.  No required client→server messages for the voice-only MVP.

### Connecting

```typescript
const ws = new WebSocket("ws://localhost:8765");

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data) as BraillieMessage;
    switch (msg.type) {
        case "state":        handleState(msg);       break;
        case "cell":         handleCell(msg);        break;
        case "narration":    announce(msg.text);     break;  // → screen reader
        case "quiz_prompt":  showTarget(msg.target); break;
        case "quiz_result":  showResult(msg);        break;
        case "stats":        updateStats(msg);       break;
        case "voice_command": flashCommand(msg.command); break;
        case "error":        showError(msg.message); break;
    }
};
```

### Message reference

#### `state` — mode + mic status
Sent immediately on connect and whenever anything changes.
```json
{
    "type":       "state",
    "mode":       "idle",
    "listening":  true,
    "mic_paused": false
}
```
| field | values |
|---|---|
| `mode` | `"idle"` / `"read"` / `"quiz"` |
| `listening` | Deepgram STT stream is open |
| `mic_paused` | mic muted while TTS is playing |

#### `cell` — fingertip landed on a braille cell
```json
{
    "type":       "cell",
    "char":       "⠁",
    "label":      "100000",
    "dots":       [1],
    "x_mm":       23.4,
    "y_mm":       45.1,
    "confidence": 0.97
}
```

#### `narration` — the backend is about to speak this text
Use to drive screen-reader announcements (`aria-live`) or subtitles.
```json
{
    "type": "narration",
    "text": "The letter is A.",
    "mode": "normal"
}
```
`mode` is `"normal"` (Deepgram TTS) or `"debrief"` (ElevenLabs, end-of-session only).

#### `quiz_prompt` — new quiz target
```json
{
    "type":    "quiz_prompt",
    "target":  "A",
    "attempt": 1
}
```

#### `quiz_result` — answer checked
```json
{
    "type":     "quiz_result",
    "correct":  true,
    "detected": "A",
    "target":   "A"
}
```

#### `stats` — session accuracy (sent after every quiz result)
```json
{
    "type":        "stats",
    "total":       12,
    "correct":     9,
    "accuracy":    0.75,
    "streak":      3,
    "most_missed": ["d", "g"]
}
```

#### `voice_command` — echo of recognised command
Use to animate a microphone indicator or highlight the command in the UI.
```json
{
    "type":    "voice_command",
    "command": "hint"
}
```

#### `error`
```json
{
    "type":    "error",
    "message": "DEEPGRAM_API_KEY is not set."
}
```

### Optional client→server messages
These are wired but currently no-ops.  Implement them in `server.py` when
keyboard/button control is needed:

```json
{ "type": "set_mode", "mode": "quiz" }
{ "type": "next" }
```

### CORS / same-origin
The WebSocket server does not enforce origin checks.  The Vite dev server runs
on a different port, so connect with the full `ws://localhost:8765` URL.  For
production, set `WS_HOST=127.0.0.1` to restrict to localhost.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | — | Deepgram secret key (required unless `VOICE_IO_MOCK=1`) |
| `ELEVENLABS_API_KEY` | — | ElevenLabs key (required for end-of-session debrief only) |
| `VOICE_IO_MOCK=1` | off | Disable all audio APIs; print instead |
| `VOICE_IO_DEBUG=1` | off | Print every raw Deepgram transcript + confidence |
| `VOICE_IO_MIC_INDEX` | system default | Pin a specific microphone device index |
| `COMMAND_CONFIDENCE_THRESHOLD` | `0.72` | Lower = more sensitive STT command matching |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_HOST` | `0.0.0.0` | WebSocket bind address |

---

## Testing

```bash
# All backend unit tests (14 tests, no hardware needed)
cd backend && source .venv/bin/activate && pytest

# Full offline smoke-test (mock detector + mock voice)
VOICE_IO_MOCK=1 python3 test_voice_io.py

# Live mic command test (needs DEEPGRAM_API_KEY)
VOICE_IO_DEBUG=1 VOICE_IO_TEST_DURATION=60 python3 test_mic_pin.py
```
