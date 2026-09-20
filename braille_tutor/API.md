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
| GET | `/api/phone/qr.png` | the QR code (PNG) that connects a phone camera. 404 unless started with `--phone-camera` |

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
`finger` is the camera-tracked fingertip (`fingertip.py`; off with `--no-finger-tracking`) unless something was posted to `/api/finger`, which overrides it until `{"clear": true}`. `GET /api/settings` returns `{"speech_speed": 1.0, "pace": "normal", "tones": true, "hearing_feedback": true}`; `POST /api/settings` with any of those keys changes them (400 with an `error` and NO change if any value is not allowed: speech_speed 0.6-1.5, pace normal|relaxed, tones and hearing_feedback true|false). `state.settings` carries the same. The commands `help`, `slower`, `faster`, `relaxed`, `normal` are in `config.commands`. The menu-driven tutor (`tutor_server.py` without `--mode`, or `--mode menu`): `state.hub` is `{"mode": "menu|learn|read|quiz", "modes": {"learn": {"title", "sheet"}, ...}, "user": {"name", "kind": "google|guest", "saves": bool} | null, "greeted": bool}` (null in other modes). `POST /api/session {"kind": "google", "name": "Ana", "profile": "<account id>"}` or `{"kind": "guest", "name": "Gina"}` says who is using it (a guest's progress is never saved; the tutor speaks "What do you want to do today, <name>?" once the phone is linked). `POST /api/mode {"mode": "learn"|"read"|"quiz"|"menu"}` chooses what to do (202). `POST /api/prompt {"name": "welcome"}` has the tutor say one of its prepared lines (`PROMPTS` in `tutor.py`: the sign-in conversation: `welcome`, `welcome_back`, `ask_name`, `confirm_name`, ...); a line with `{name}` takes `"who": "Sam"` (cleaned like any name). `POST /api/dialogue {"open": true}` says the sign-in page is talking with the user: while it is open (30 s, so the page pings every 10 s; closed by `{"open": false}` or by `/api/session`) no voice command runs and the tutor does not say "I did not catch that" over a name being answered. `state.speaking` is true while the tutor talks, and `state.heard` is `{"n", "text", "matched", "reason"}` for the last thing its microphone made out (`n` counts up; null when the tutor has no live microphone): together they let a page have a spoken conversation. All four answer 404 outside the menu-driven tutor. Voice commands `learn`, `read`, `quiz`, `menu` are also in `config.commands`. With `--mode learn`: `state.learning` is `{"phase": "teach|practice|review|recap|explore|await_sheet|idle|done", "lesson": {"id", "title", "index", "of", "sheet"}, "target": "d", "target_dots": [1,4,5], "remaining", "letters", "in_a_row", "hints", "tries", "round"}` and `state.progress` is `{"learned", "practised", "sessions", "streak", "best_streak", "mastery": {"a": 0.25, ...}, "confusions": [{"touched","wanted","count"}], "lessons": {"l1": {"best","last","times"}}}` (both null in other modes). `GET /api/progress` returns `{"summary", "data"}`; `POST /api/progress {"data": <saved copy>}` merges a copy in (idempotent) and returns the same. Both 404 outside learn mode. The commands `explore` and `practice` are added. `POST /api/command` also takes `"next page"` (forget the page being read and, in explore mode, work out which printed sheet is now on the desk): `state.reading` = `{"locked", "total", "between_pages"}` and `state.config.sheet` follow it. With `--phone-camera` the tutor's voice is played on the phone (`--sound laptop` to keep it here): nothing to do in the frontend. `phone.diagnosis` (string or null) says in plain words why no video has arrived yet, and `phone.events` lists the last few steps the phone got through: show the diagnosis on the setup screen. In `explore` mode the `/api/video` frames carry the detection boxes and a status line. `config.voice` is `{"mode": "live"|"mock", "ok": bool, "problems": [...], "warnings": [...]}`: `ok` false means the user will hear nothing, and `problems` says why (show it). `config.mode` may be `explore` (no quiz: `tutor.state` is `exploring`, and what is said is what the camera detects under the finger). `phone` is null unless the server was started with `--phone-camera`; then it is `{"connected": false, "address": "https://192.168.1.5:8443", "code": "482917", "qr": "/api/phone/qr.png", "instructions": "..."}`: show the `qr` image (and `address` + `code` as text, or speak `instructions`) until `connected` is true. Until a phone connects, `/api/video` itself shows the QR code and steps. `finger.cell` is filled in when a printed
sheet is loaded (letters mode, or `--sheet`).

Posted fingertip positions expire after 10 seconds (or immediately after `{"clear": true}`), then camera tracking resumes.
`finger.source` is `"posted"` or `"camera"` for the active position, and null when there is no position.

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
