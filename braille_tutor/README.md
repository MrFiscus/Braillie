# Braille tutor: detection module

Finds braille cells on a tactile page seen by a top-down webcam and reports them in **page millimeters**,
so a tracked fingertip can be matched to a cell with `nearest_cell`.

## Credits

Detection uses the pretrained YOLOv8-medium weights from
[DotNeuralNet](https://github.com/snoop2head/DotNeuralNet) by snoop2head (MIT License, Copyright (c) 2023 snoop2head).
Cloned to `third_party/DotNeuralNet`; nothing is trained or fine-tuned here, and none of its scripts are run.
The class-to-braille conversion in `detect.py` is our own. Class labels are 6-character strings where
character *i* is dot *i+1* (`"100110"` = dots 1, 4, 5 = d).

## Setup

```
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install ultralytics opencv-python numpy
git clone https://github.com/snoop2head/DotNeuralNet third_party/DotNeuralNet
```

## Use

```
python make_markers.py                 # markers/marker_0..3.png + markers/sheet.png; print and stick on page corners
python make_sheet.py                   # every printable sheet, WITH markers (sheet/) and WITHOUT (sheet/nomarkers/); sheet/print/ = the four to print
# then set PAGE_W_MM / PAGE_H_MM in page.py to your measured marker-centre to marker-centre distances

python detect.py photo.jpg --annotate  # writes photo_annotated.png, prints each row as braille
python detect.py --live                # live green boxes on detected braille + page status; s = save cells.json, q = quit
python detect.py --live --auto-page 200 190 --rows 1   # --rows N keeps only the first N lines; --vote N pools N scans (default 8) so labels stay steady; amber box = unsure.
# no markers: finds the page edges itself (page width and height in mm); r = re-find
python detect.py photo.jpg --auto-page 200 190 --annotate   # same on a photo; draws the found page outline
python detect.py --live --print        # also shows a 'detected braille' window and streams the rows to the terminal as braille
python check_sheet.py                  # overlay the A-Z sheet layout on the camera, click a cell to name it
python calibrate.py --sheet --no-track # no markers: click the 4 paper corners, then add --calib calibration.json to the above
python -m unittest -v                  # tests
```

Marker ids: 0 top left, 1 top right, 2 bottom right, 3 bottom left.

**Markers going out of view no longer lose the page.** Once the four markers have been seen, the sheet keeps being followed
by its own appearance, so it can be moved, tilted, or have a marker covered or pushed off the frame and the overlay stays
put. The status line says which is in use: `page OK (4 markers)`, `markers hidden, following the sheet (N matches)`, or
`holding its last position` for a few seconds before the page is finally reported lost. Press `r` to re-register.
Use `--markers-only` for the old behaviour (all four markers required in every frame).

```python
from detect import scan_page, nearest_cell, cells_from_layout, load_cells
from page import page_homography, to_page

H = page_homography(frame)                    # None unless all 4 markers visible
cells = scan_page(frame, H)                   # no hand in view; list of Cell dicts in mm
# fallback if the detector struggles on your sheet: same structure, hardcoded
cells = cells_from_layout(["abcde", "fghij"], x0=20, y0=30, pitch_x=15, pitch_y=25)

x, y = to_page(H, finger_px, finger_py)
cell = nearest_cell(cells, x, y)              # None if the finger is not on a cell
```

## The tutor (detection + voice + word checking)

`tutor.py` ties this module to the team's `voice_io.py` and `backend/braillie/word_correction.py`. It imports them from the
repo, so there is nothing to install except the spell checker: `pip install pyspellchecker`.

```
python tutor.py --mock --autostart                      # letter quiz on the A-Z sheet; speech is printed, type commands
python tutor.py --mock --sheet numbers --autostart      # quiz on the numbers-and-signs sheet (or: words, lookalikes)
python tutor.py --mock --sheet words --mode read        # word modes read a printed sheet from its known layout
python tutor.py --mock --mode read --auto-page 280 292  # "found it" reads the word under the finger, re-detecting bad reads
python tutor.py --mock --mode word-quiz --words cap cat # "find the word cap"
```

Drop `--mock` to use the real Deepgram voice (needs `DEEPGRAM_API_KEY`, see `voice_io.py`). Voice commands: start quiz,
repeat, hint, found it, next, next page, stop. The same things are on the window keys s r h f n p x (q quits).

- **Next page** (say "next page", "new page" or "another page"; window key `p`; `POST /api/command {"command": "next page"}`): forgets the
  page that was being read so the one now on the desk is read fresh. Locked-in readings hold on purpose (so a shaky frame cannot make
  them flicker), which is also why a page you swapped stayed stuck; this lets go of them and makes the tutor find the page again. In
  **explore mode** it also works out WHICH of the four printed sheets (alphabet, words, numbers, look-alikes) is now on the desk and
  switches to it: "This is the words sheet." Because the old page is usually still in view for a moment, recognising it does not
  count as a change: a different sheet is accepted at once, the same sheet again only after something in view changes (a hand, the
  page being swapped), and if nothing changes for 8 s it says "The page didn't change, so I am still using the alphabet sheet."
  If it cannot tell in 30 s it says so and keeps the old sheet. It never disturbs a quiz in progress. `state.reading` shows
  `locked`/`total` cells and `between_pages`. Pages that are not one of the four printed sheets (a real book page) are not
  recognised: explore mode only reads the printed sheets. The phrase was added to `voice_io.py`'s command list (longest phrase wins,
  so "next page" is never heard as plain "next").
- **Finger:** the fingertip is tracked from the camera (`fingertip.py`, see below); clicking the video overrides it, and
  `--no-finger-tracking` turns tracking off. Any other tracker can plug into `TutorSession.finger` (a function returning the
  page position in mm, or None).
- **Printed sheets** (`sheets.py`): alphabet, words, numbers (with signs) and lookalikes. All share one A4 layout, so calibration
  and page registration work for any of them. `--sheet NAME` picks one; the quiz asks for whatever is printed on it (letters,
  numbers, punctuation) and the word modes read its known layout instead of running the detector (`--detect` overrides).
- **With or without markers.** Every sheet comes in two versions with the same layout. The marker version needs the four
  printed markers stuck on the front (`markers/sheet.png`). The no-marker version needs nothing: run with `--paper` and the camera
  finds the paper's own four edges (`--paper` works with `detect.py`, `tutor.py`, `tutor_server.py` and `check_sheet.py`).
  It needs a plain dark surface, all four paper edges in view, the sheet upright, and nothing touching the paper; if it can't
  find the sheet the status line says why. It re-checks the edges about three times a second and holds the last position for
  5 seconds while an edge is hidden. Press `r` in the window to make it look again.
- **Letter quiz** uses the printed sheet's known layout. **Word modes** use fresh detections, turned into letters and words by `reader.py`
  (plain letters only; any other cell reads as "?", which makes the backend re-detect instead of guessing).

## Optional LLM extras (`--llm`)

`llm.py` can add polish with any OpenAI-compatible chat API, without ever sitting on the critical path:

```
export OPENAI_API_KEY=...            # never commit this; set BRAILLIE_LLM_MODEL (or --llm-model) to a model your key can call
python tutor.py --mock --llm --autostart
```

- **Off by default.** Without `--llm` and a key, the tutor behaves exactly as before.
- **Memory aids and praise lines** are fetched in the background when a quiz starts; a hint, a "well done" or a "keep going" is then
  a lookup, not a network call. The third hint uses a memory aid if one has arrived.
- **The debrief** is written from the session's facts. Its request starts before "Let's see how you did", so that lead-in hides
  the wait, and a hard time limit (2 s) falls back to the built-in debrief.
- **Only facts are sent** (cell names, dot numbers, scores). Never camera images, audio or names.
- **Replies are checked**: no digits that were not in the facts, no position words that don't apply to the cell, length limits.
- The first error or timeout switches the helper off for the rest of the session. The server's `/api/state` shows `config.llm`.

## Web server for the frontend

`tutor_server.py` exposes the tutor over HTTP so a browser can show the camera and drive the session:

```
python tutor_server.py --mock          # then open http://127.0.0.1:8000
```

It serves a built-in console page and a small JSON / MJPEG / event-stream API. See `API.md` for the endpoints and a React hook.

## Seeing the reading on screen, with or without a locked page

Two different things can be drawn on the camera view, and only one needs the page:

- **The layout overlay** (green rings and letters, from `check_sheet.py` and the tutor) comes from the sheet's *known layout*
  projected through the page registration, so it needs the page located.
- **The live reading** is what the detector sees in the camera image: a box per cell, red dots where it thinks the dots are, and
  the letter above (`?` if the pattern isn't a plain letter). It needs no registration, so it stays on screen when the page is lost.

`check_sheet.py` and `detect.py --live` show the live reading by default (`check_sheet.py --no-detect` turns it off; press `d`
to keep it on or hide it while the page is registered). `tutor.py` and `tutor_server.py` show it with `--show-detections`.
It is only as good as the detector: on embossed braille it is usually right, on big flat printed dots it often is not.

## Reading our printed / poked sheets correctly (`--sheet`)

The general detector has to find cells anywhere and classify them, and it has measured weaknesses (it drops right-column dots and
misplaces cells with no top row). For OUR sheets none of that is needed: page registration puts every dot position within a
fraction of a millimetre, so `sheetread.py` just looks at each of a cell's six slots and measures whether a dot is raised there.
That is observation, not copying the layout, so a missed poke or the wrong sheet on the desk shows up.

```
python check_sheet.py --sheet alphabet       # words | numbers | lookalikes; add --paper if the sheet has no markers
python detect.py --live --sheet alphabet --print
```

**Self-alignment.** Without markers the page is placed from the paper's edges, and a printer's margin or scaling puts the printed
dots a millimetre or three away from where the A4 outline says. Sampling a dot slot 2 mm off reads garbage (measured: alphabet
sheet 26/26 at 1 mm off, 16/26 at 2 mm, 0/26 at 3 mm; the markers never had this because they are printed with the dots). So
`sheetread.align` first finds the one shift (up to 4.5 mm) at which the layout's dots line up best with the picture, then reads.
It is a single shift for the whole page, so it cannot bend the reading toward what is expected: a missed poke still shows as
one wrong cell and the wrong sheet still reads 0/26. Simulated: every cell right for shifts up to 5.5 mm, and with 2% scale or 1
degree of rotation error; not recovered: 2.5 mm shift AND 2% scale AND 1 degree together (about 80% of cells). The working
resolution also came down from 10 to 5 px/mm: same accuracy, 245 ms a scan instead of 3.2 s.

The window shows `sheet check: all 26 cells read correctly`, or `sheet check: 25/26 cells correct (row 2 col 4: expected d read
blank)` with the offending cell ringed in red. Measured on simulated photos (embossed dots, side light, blur, faint contrast,
1080p camera): the actual `poke_*.png` files, taken back out of the PNGs, flipped and read, came out **99.9% of cells correct** (the
general detector alone: 95.3%); your flat printed test sheets 100% (detector alone: 88%). Real cameras and real poking are not
measured yet: the on-screen check line is how to measure them.

The general detector also got a fix for the same family of errors: a cell with no dots in its top row (a comma, period, capital
sign) was read one row too high; `align_shifted` corrects that from the page's line grid (numbers-and-signs sheet 72% to 93%).

## Reading a real embossed book page

What matters for reading the actual book, and what was measured (on simulated embossed pages with known truth, in
`tools/`, plus your real photos where there is no truth):

- **Input size is chosen automatically** (`--imgsz auto`, the default). The model shrinks every frame to a fixed width, so
  on a 1080p webcam frame small braille got too small to read: 64-80% correct at 640 became 90-98% at the right size. It
  measures the cells from a quick first pass and sizes the input so a cell is about 15-17 px wide. Cost: a scan can take
  up to about 1.2 s when the page is small in the frame.
- **Faint pages are boosted** (`--enhance auto`, the default): it tries the frame as-is and with the paper's contrast
  stretched, and keeps whichever the detector is more confident about. `--enhance off` disables it.
- **Contractions are decoded** (`contractions.py`, `reader.decode_lines`). Published braille is contracted: one cell can be a
  whole word ("b" alone is "but") or a group ("th", "ing"). Read letter by letter it is nonsense even when every cell is right:
  on a simulated contracted-English page 1 of 41 words came out right as plain letters and 34 of 41 once decoded. It covers
  the core UEB rules (wordsigns, groupsigns, capitals, numbers, punctuation), not shortforms or technical notation, and it
  uses a spell checker to choose between readings, so one misread cell still gives a wrong word. The live readout window and
  `--print` show the decoded text; `tutor.py --mode read` on a real page uses it.
- **Word gaps** are judged against the page-wide cell spacing, so a line of one-cell words ("the child can go") splits.
- **A quality note** appears on the live view when the reading looks unreliable: cells too small, or the detector unsure.

Known weakness, measured: the detector favours left-column dots. On random patterns it drops right-column dots (4, 5, 6)
19-32% of the time and invents left-column ones 25-30% of the time. Right-heavy contractions ("th", "er", "ou") suffer most.
A correction that re-checks each dot against the image gained on some pages and lost on others, so it is not included.
Good raking light and a sharp, close photo help more than any setting.

## Hearing what the camera detects: explore mode (`--mode explore`)

```
python tutor_server.py --mode explore --sheet alphabet --paper        # or tutor.py --mode explore ...
```

No quiz and no commands to give: it starts by itself. Rest a finger on a cell for about three quarters of a second and it says what
the camera SEES there, with the dots, so a blind learner can feel the pattern while hearing it:

> "The letter D. Dots 1, 4 and 5: top-left, top-right and middle-right."

- The first time a symbol is met it is described in full (where each dot sits); after that briefly ("The letter D. Dots 1, 4 and 5.").
  It speaks once per cell, and stays quiet while the finger wobbles inside it or sweeps across, until it rests on another cell.
- It reads the dots the camera observes (`sheetread.py`), NOT the layout copied from the file: if a dot was not poked it says what is
  actually there ("The letter C. Dots 1 and 4" where the sheet wants a D). On a sheet with signs it names them ("The number sign").
- Off the braille for 2.5 s: "I don't see any braille there." Finger gone for 2 s: "I can't see your finger." Each once.
- Voice commands still work: *found it* (say the cell now), *hint* (always the full description), *repeat*, *stop*.
- Speech never overlaps: one voice at a time, whatever asks for it.
- **You can see it working:** the video shows a small box for every cell with the dots the camera detected (red) and the letter it
  reads (yellow above the box). A box turns green once its reading has held steady and matches the sheet (locked in, so it will not
  flicker), and stays amber while still reading or when it does not match (a dot not poked). The top line shows the page status and
  `sheet check: all 26 cells locked in correct` / `25/26 locked, 1 wrong (row 1 col 4: expected D read C)`. What is spoken comes from
  this same reading. `--hide-detections` turns the boxes off. On the phone flow the boxes appear the moment the page is in view.
- Only sheets so far. On a real book page the cell is described by its dots and its plain-letter reading; contractions are not spoken.

### Voice setup (Deepgram, `voice_io.py`)

The tutor reads API keys from a **git-ignored `.env` file** at the repo root (`*.env` is in `.gitignore`), never from source or
`.gitignore` itself (that file is committed, so a key put in it would be published):

```
DEEPGRAM_API_KEY=...          # speech for everything the tutor says
# (no ElevenLabs key needed: the current voice_io speaks everything, the end-of-session debrief included, through Deepgram)
```

Packages: `pip install deepgram-sdk==7.9.0 pydub pyaudio`. On a Mac without Homebrew `pyaudio` cannot build; PortAudio can be built
from source into your home directory (`./configure --prefix=$HOME/.local/portaudio && make install`, copy `include/pa_mac_core.h`
next to the installed headers, then `CFLAGS=-I$HOME/.local/portaudio/include LDFLAGS="-L$HOME/.local/portaudio/lib -Wl,-rpath,$HOME/.local/portaudio/lib" pip install pyaudio`).
Python 3.10+ is what `deepgram-sdk` supports (the `.venv312` environment has all of this).

`voice_io` swallows speech errors into its log, so a missing key or package used to mean silence with no explanation. Both apps now
print a banner at start ("VOICE: YOU WILL NOT HEAR ANYTHING" plus the reasons: mock mode, missing key, missing package) and
`GET /api/state` has `config.voice` with `ok`, `problems` and `warnings`. `--mock` (or `VOICE_IO_MOCK=1`) still prints speech instead
of playing it: without `--mock` and with a key, speech is played.

Bug fixed in `voice_io.py` on the way: it asked Deepgram for `container="wav"` without `encoding="linear16"`; Deepgram then
defaults to mp3 and answers HTTP 400 ("container is not applicable when encoding=mp3"), so no speech ever played. One added argument.

## Learning mode: guided lessons (`--mode learn`)

```
python tutor_server.py --mode learn --sheet alphabet --paper            # add --profile ana to keep someone's progress separate
```

It starts by itself and teaches braille the way a good teacher would, out loud, by touch:

1. **Lessons** (`learn.py`): six short lessons, A-E, F-J, K-O, P-T, U-Z and the look-alike pairs. Each one *teaches* a letter at a time
   ("The letter B. Two dots: top-left and middle-left. It is the letter A with one more dot, at the middle-left. Find it and rest your
   finger on it."), then *practises* them in a fresh order with a couple of older shaky letters mixed in, then gives a spoken *recap*.
   The explanations are built from the actual dot patterns, so they cannot disagree with the braille, and they use true patterns of
   the system (K-O are A-E with dot 3 added, and so on) to make the letters stick.
2. **Resting a finger is the answer**: hold the finger on a cell for about a second (or say "found it"). Nothing to press.
3. **A mistake is never just "wrong"**: it says what the finger IS on and what to feel for instead ("That's the letter E. The letter C has a
   dot at the top-right, and no dot at the middle-right."). Three misses and it shows where the letter is and moves on.
4. **It never leaves you stuck**: hints arrive by themselves after 12 s, then 15 s, then 20 s, each more specific, ending with the
   row and column. "Hint" asks for the next one at once.
5. **Adaptive review** ("practice"): 8 letters chosen by spaced repetition (`progress.py`): shaky and recently confused letters come
   round far more often, solid ones now and then. If two letters keep being mixed up it stops to compare them side by side.
6. **Free exploring** ("explore") between lessons, and back ("next"). Different sheets: a lesson that needs another sheet asks for it and
   recognises it ("next page" machinery).
7. **Gentle sounds** (`earcons.py`): a bright rising phrase for right, a soft falling one for not quite (never a buzzer), a little
   chime when a lesson starts, a fanfare for a lesson really learned. Played through the same speaker as the voice; `--no-tones` turns them off.

Voice commands: **start** (or "start lesson"), **repeat**, **hint**, **found it**, **next** (skip a letter, move on from a recap), **explore**,
**practice** (or "review"), **next page**, **stop** (spoken recap of the session). The same are buttons on the website.

Progress (letters and how solid, mix-ups, lessons, days in a row) is saved after every answer to `~/.braillie/progress-<profile>.json`
(written atomically; a damaged file is set aside, never deleted). `GET /api/progress` returns it, `POST /api/progress {"data": ...}`
MERGES a copy in (the more recently practised letter wins, counts take the larger value, so merging the same copy twice changes nothing).
The website uses that to keep it in the learner's Supabase account: run `supabase_progress.sql` once (a `learning_progress` table with
row level security), and the Learn screen loads the account's copy on start and saves back a few seconds after each change. Signed out or
offline it just says progress is kept on this computer.

**AI coach**: with `OPENAI_API_KEY=...` in the repo's `.env` learn mode turns the coach on by itself (`--no-llm` stops that): it writes
memory aids for the letters in a lesson (used as the second hint), varied praise, and a personal debrief. Only lesson facts (letter names, dot
numbers, scores) are ever sent, never pictures or audio; if it is slow or fails, the built-in wording is used. Without a key, everything
above works the same with the built-in wording.

What is tested and what is not: the lesson engine, the progress rules, the tones, the wiring and the account sync are covered by tests
(`test_learn.py`, `test_progress.py`, `test_earcons.py`, `test_learn_session.py`, and `node --test "tests/*.test.mjs"` in `website-frontend`),
and a whole lesson was run through the real server with a simulated camera and a scripted learner. NOT tested: a real hand on a real poked
sheet (the timings, such as the 1.2 s rest, are educated guesses to tune with real learners), the AI coach with a real key, and the account
sync against a real Supabase project.

## The website flow (React frontend)

With the tutor running (`python tutor_server.py --mode explore --sheet alphabet --paper`, add `--phone-camera` for a phone) and the site
running (`cd website-frontend && npm run dev`), the flow after sign-in is:

1. **Get to Know** (`/user-information`), then **Next** goes to
2. **Connect your camera** (`/connect-phone`): a QR code and the typed address + code when the tutor was started with `--phone-camera`
   (Continue unlocks once the phone's video arrives, and a plain-words hint says where a phone got stuck); with the laptop camera it says
   so and Continue is available at once. Then
3. **Practice / Learn** (`/practice`): with `--mode learn` it becomes the learning screen (lesson, the letter with its dot diagram, progress
   through the lesson, a map of how well each letter is known, streak, and the progress sync); the live camera with a box on every detected cell (red dots = what it sees, letter above, green = locked in),
   a status line ("All 26 cells read. Rest a finger on a cell to hear it."), the sheet, what the finger is on, buttons for the voice
   commands (next page, repeat, hint, found it, stop), and what the tutor said. A warning box explains if speech will not be heard.
   Clicking the video stands in for the fingertip. The tutor itself does the speaking.

The pages talk to the tutor on `http://127.0.0.1:8000` (`VITE_TUTOR_API` overrides it); shared code is in `website-frontend/src/tutor/`.
The site needs `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` and `VITE_GOOGLE_CLIENT_ID` in a git-ignored `website-frontend/.env` to log in.

## Phone as the camera: scan a QR code (`--phone-camera`)

```
python tutor_server.py --phone-camera --sheet alphabet --paper       # or: python tutor.py --phone-camera ...
```

The video (the tutor window, or `/api/video` for the frontend) shows a QR code and the steps. The phone scans it with its normal
camera app, opens the link, allows the camera, and streams to the laptop. Nothing to install on the phone. Both must be on the
same Wi-Fi. The QR code is also printed in the terminal, and `GET /api/phone/qr.png` serves it as an image (see API.md;
`state.phone` says whether a phone is connected, so a setup screen can show the code until it is).

For someone who cannot see the screen: the laptop SPEAKS how to connect (the address and a 6-digit code, digit by digit; the
phone page has a big code box), an iPhone with VoiceOver reads a QR code aloud from its Camera app, and once connected the phone
itself talks: "Page found, you are ready" or "I cannot see the page yet, hold the phone higher". The phone page is one large
button and one status line, built for VoiceOver and TalkBack. The laptop announces when the phone connects or drops.

**Sound on the phone** (`--sound phone`, the default with `--phone-camera`; `--sound laptop` keeps it on the laptop): the person is
next to the phone, not the laptop, so the tutor's voice plays there. The laptop still makes the speech with your friend's Deepgram
voice, then sends the finished audio to the phone page, which plays it and reports back; the laptop
waits for that, so lines never overlap and the microphone stays muted while it talks. The one thing the user does: **tap the big
button on the phone once** (phones refuse to play sound until a tap; the same tap starts the camera). Until the phone has done that,
and whenever it drops off, speech plays on the laptop as before, so nothing is ever silent (the opening instructions are spoken
there, before any phone is connected). If the phone takes a clip it is trusted to play it, even through a hiccup in the video, so a
line is never said twice. The debrief (MP3) is decoded by the phone, so it needs no ffmpeg on the laptop when played there.
Only the two playback functions in `voice_io` are redirected (`_play_audio_stream`, `_play_mp3_stream`); a test fails if they are
renamed. Bug found on the way: Deepgram's streamed WAV declares a length of about 12 hours in its header, so the tutor rewrites the
header with the true size before sending. Not done: the phone's MICROPHONE is not used, so spoken commands still go to the laptop's
microphone (fine if the user is near the laptop; streaming the phone's mic is the next step if not).

**If the phone does not connect**, the laptop says where it got stuck: the terminal prints `[phone] ...` lines as they happen (a
device reached the laptop; the phone page was opened; the phone's browser reported a camera error; video is arriving) and, while
nothing is connected, a `[phone] HINT:` in plain words. The same text is in `state.phone.diagnosis` (the setup screens show it).
"No phone has reached this laptop yet" means the network (same Wi-Fi? many shared networks block devices from talking to each
other: use the phone's hotspot for both); "could not finish the secure connection (TLSV1_ALERT_UNKNOWN_CA)" means the phone refused
the certificate (choose Advanced, then continue); "camera did not start (NotAllowedError)" means camera permission was denied. If
port 8443 is busy (an earlier run still going) it moves to the next free port and the QR code carries it. Only ONE copy of the
tutor should run at a time: stop the old one (Ctrl+C) before starting a new one.

How it works and what to know:
- Phone browsers only give a page the camera over HTTPS, so the laptop makes a self-signed certificate once (`openssl`, saved in
  `~/.braillie`). The first time, the phone says the connection is not private: choose Advanced, then continue. **That warning
  is the one step that cannot go away without a real domain name; a helper may need to tap it.**
- Same-Wi-Fi only, and many campus, hotel and hackathon networks block devices from talking to each other. If the phone cannot
  reach the address, join both to the phone's hotspot or the laptop's. `--phone-host IP` overrides the address it guesses,
  `--phone-port` the port (8443).
- Only two things are reachable from the network (the phone page and frame uploads), both need the session's code, wrong codes are
  throttled (20 in a row locks guessing out for a minute), uploads are size limited and must decode as an image. The tutor's own
  API stays on localhost.
- Streams about 8 frames a second at up to 1280 px, plus the phone's continuous focus and torch where the browser allows them
  (Android Chrome does; iPhone Safari does not offer the torch).
- Tested: the whole chain with a simulated phone over real HTTPS (certificate verified), a real running server, QR round trips,
  code throttling, upload limits, and the tutor finding the page from uploaded photos. NOT tested: a real phone, its browser, or the
  phone page's JavaScript running (there is no browser here; the script was only syntax-checked). Expect to fix small things there.

## Locking in what is read right

A reading that is right stays right: a cell is **locked** once it has been read correctly on `lock_after` (2) scans in a row (known
sheet: equal to what the sheet says; other pages: a steady majority), and after that shaking the camera, a blurred frame or a
finger passing over it cannot change it. It only unlocks after `unlock_after` (6) scans in a row that all agree on the same
DIFFERENT reading (the page really changed), or when you press `u`. Locked cells are drawn solid green, still-reading ones amber,
wrong ones red (`check_sheet.py`); in `detect.py --live` locked is green and unlocked amber. Losing the page never unlocks anything.
`python detect.py --live --verified cells.json` starts with the cells in that file already locked (save one with `s`, fix any
label by hand, then reuse it): the way to make a demo page rock solid.

## Fingertip tracking (`fingertip.py`)

opencv + numpy only. Skin colour (YCrCb, minus paper-coloured pixels) gives the hand; the fingertip is the outline point
farthest from where the arm enters the picture; the tracker smooths it in PAGE millimetres (camera shake does not shake the
finger), needs two frames in a row before reporting, and holds the position for 0.6 s if the hand blinks out. The tutor uses it
automatically (`tutor.py`, `tutor_server.py`; the green ring on the video is the tracked finger, an orange ring is a click).

```
python fingertip.py --camera 0                 # try it: video, skin mask, tip and confidence
python fingertip.py --camera 0 --paper         # plus the page position in mm (any page source flag works)
```

Tested on synthetic hands only (four skin tones, angles, positions, noise, dimmer light, warm paper, two fingers, a skin-coloured
blob that is not attached to an arm): the tip lands within about 10 px. NOT tested on a real hand yet. Known limits: a
skin-coloured desk or sleeve-less forearm filling the picture defeats it, gloves do not work, unusual lighting may need
`SKIN_CR` / `SKIN_CB` adjusted, and the tip found is the end of the finger, a few millimetres past where the pad touches (small
next to the 19 mm cell pitch, but it is why a finger held flat may read one cell over).

## Known limits

The pretrained model confuses some letters (on a clean example image it read d as f, i as e and missed y).
Use `dot_distance` to treat near-miss pairs leniently, and keep `cells_from_layout` ready as the fallback.
