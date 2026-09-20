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

Drop `--mock` to use the real Deepgram / ElevenLabs voices (needs their API keys, see `voice_io.py`). Voice commands: start quiz,
repeat, hint, found it, next, stop. The same things are on the window keys s r h f n x (q quits).

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
