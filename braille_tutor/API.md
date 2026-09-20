# Tutor server API (for the frontend)

Start the server (from `braille_tutor/`, venv active):

```
python tutor_server.py --mock --mode letters            # offline voice, speech is printed
python tutor_server.py --mode letters                   # real Deepgram / ElevenLabs voice (needs the API keys)
```

It listens on `http://127.0.0.1:8000`. Open that address for a built-in console page (no Node needed), or call the API
from the React app. Options: `--camera N|URL|video.mp4`, `--auto-page W H`, `--paper` (no markers: find the printed sheet's edges), `--calib calibration.json`, `--mode read`,
`--sheet alphabet|words|numbers|lookalikes`, `--mode word-quiz --words cap cat`, `--llm`, `--questions N`, `--no-mic`, `--port`. Standard library only, no extra install.

Only pages from `localhost` / `127.0.0.1` (any port, so `npm run dev` on 5173 works) may call the API. The server does not
listen on the network unless you pass `--host 0.0.0.0` (don't, on a shared network).

## Endpoints

| | | |
|---|---|---|
| GET | `/api/state` | one JSON snapshot (below) |
| GET | `/api/events` | the same snapshot as Server-Sent Events, sent when it changes (and every ~5 s) |
| GET | `/api/video` | the camera as an MJPEG stream: `<img src="http://127.0.0.1:8000/api/video">` |
| GET | `/api/cells` | the printed sheet's cells in page mm: `{cells:[{x,y,w,h,letter,label,name,dots,row,col}]}` (`name` is e.g. "the number 3") |
| POST | `/api/command` | `{"command": "start quiz"}`; others: `repeat`, `hint`, `found it`, `next`, `stop` (`found_it` also works). Replies `202` |
| POST | `/api/finger` | `{"u":0.4,"v":0.7}` a point on the video as fractions, or `{"x_mm":..,"y_mm":..}`, or `{"clear":true}` |

Errors are JSON `{"error": "..."}` with 400 (bad input), 403 (foreign origin), 404, or 413 (body over 4 KB).

## `/api/state`

```json
{
  "config":  {"mode": "letters", "llm": "off", "commands": ["start quiz", "repeat", "hint", "found it", "next", "stop"]},
  "camera":  {"ok": true, "frames": 812},
  "page":    {"ok": true, "message": "page OK (4 markers)"},
  "tutor":   {"mode": "letters", "state": "asking", "prompt": "Find the letter H.", "question": 1, "total": 5,
              "asked": 0, "correct": 0, "tries": 0},
  "finger":  {"page_mm": [62.5, 51.0], "cell": {"letter": "h", "label": "H", "name": "the letter H", "dots": [1, 2, 5], "row": 0, "col": 7}},
  "said":    [{"t": 1758290000.1, "kind": "speech", "text": "Find the letter H."}],
  "debrief": null
}
```

`tutor.state` is `idle`, `asking` (quiz running), `reading` (read mode) or `done`. When the session ends, `debrief` becomes
`{"accuracy": 0.5, "missed": ["J", "H"]}`. `page.ok` false means the page can't be located; `page.message` says why.
`finger` is the camera-tracked fingertip (`fingertip.py`; off with `--no-finger-tracking`) unless something was posted to `/api/finger`, which overrides it until `{"clear": true}`. `finger.cell` is filled in when a printed
sheet is loaded (letters mode, or `--sheet`).

## From React (TypeScript)

```tsx
import { useEffect, useState } from 'react'

const API = 'http://127.0.0.1:8000'

export type TutorState = {
  page: { ok: boolean; message: string }
  tutor: { state: string; prompt: string; question: number; total: number; asked: number; correct: number }
  finger: { page_mm: [number, number] | null; cell: { letter: string | null; dots: number[] } | null }
  said: { kind: 'speech' | 'debrief'; text: string }[]
  debrief: { accuracy: number; missed: string[] } | null
}

export function useTutor() {
  const [state, setState] = useState<TutorState | null>(null)
  useEffect(() => {
    const es = new EventSource(`${API}/api/events`)
    es.onmessage = (e) => setState(JSON.parse(e.data))
    return () => es.close()
  }, [])
  const command = (c: string) =>
    fetch(`${API}/api/command`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command: c }) })
  const setFinger = (u: number, v: number) =>
    fetch(`${API}/api/finger`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ u, v }) })
  return { state, command, setFinger }
}

// <img src={`${API}/api/video`} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setFinger((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height) }} />
```
