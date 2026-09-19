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
python tutor.py --mock --mode read --auto-page 280 292  # "found it" reads the word under the finger, re-detecting bad reads
python tutor.py --mock --mode word-quiz --words cap cat # "find the word cap"
```

Drop `--mock` to use the real Deepgram / ElevenLabs voices (needs their API keys, see `voice_io.py`). Voice commands: start quiz,
repeat, hint, found it, next, stop. The same things are on the window keys s r h f n x (q quits).

- **Finger:** nobody tracks the fingertip yet, so click on the camera view where the fingertip is. Plug a tracker into
  `TutorSession.finger` (a function returning the page position in mm, or None).
- **Letter quiz** uses the A-Z sheet's known layout. **Word modes** use fresh detections, turned into letters and words by `reader.py`
  (plain letters only; any other cell reads as "?", which makes the backend re-detect instead of guessing).

## Web server for the frontend

`tutor_server.py` exposes the tutor over HTTP so a browser can show the camera and drive the session:

```
python tutor_server.py --mock          # then open http://127.0.0.1:8000
```

It serves a built-in console page and a small JSON / MJPEG / event-stream API. See `API.md` for the endpoints and a React hook.

## Known limits

The pretrained model confuses some letters (on a clean example image it read d as f, i as e and missed y).
Use `dot_distance` to treat near-miss pairs leniently, and keep `cells_from_layout` ready as the fallback.
