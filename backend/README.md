# Braillie backend

Word validation for braille cell detector output. When a detected word isn't
valid, the module asks the caller to re-run detection on a fresh frame instead
of guessing a correction — it never guesses; it falls back to the raw word.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```bash
pytest
```

## Usage

```python
from braillie.word_correction import correct_word_read_mode, check_word_quiz_mode

# Read mode: validate a detected word; re-detect on a fresh frame if invalid.
result = correct_word_read_mode(
    "cax",
    redetect=lambda: detector.read_word(camera.latest_frame()),
)
# ReadResult(word="cap", raw="cax", corrected=True, source="redetect", attempts=1)

# Quiz mode: re-detect until the reading matches the target word.
result = check_word_quiz_mode(
    "cax", "cap",
    redetect=lambda: detector.read_word(camera.latest_frame()),
)
# QuizResult(correct=True, detected="cax", target="cap", corrected=True, attempts=1)
```

`redetect` is optional — pass `max_redetects` to bound the retries (default 2).
Quiz mode never accepts anything but an exact (normalized) match, whether from
the first detection or a re-read.
