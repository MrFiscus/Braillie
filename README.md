# Braillie

**Learn braille by touch, with Braillo: a tutor who talks you through it and hears you back.**

Braillie watches a printed braille sheet through your phone's camera, follows your finger, and works out which letter you're
touching. Then Braillo, its friendly braille-cell tutor, teaches you, reads to you, or quizzes you out loud. It's built for people
who can't see the screen: from the front page to your progress, **everything can be done by voice.**

<p align="center"><img src="docs/img/screenshot-landing.png" alt="The Braillie front page: 'Feel the dots. Hear the letters.' beside a drawing of a fingertip reading the word braillie in braille, with Start learning and How it works buttons." width="820"></p>

## What it does

- **Voice first, from the very first page.** Braillo offers to talk you through the website as soon as you arrive. Say any
  button's name to press it ("start learning", "dark", "privacy policy"), or "go back", "scroll down", "home", "my page",
  "sign out". "Help" says what you can do right there, in a sentence; "more options" lists everything.
- **Reads a real printed braille sheet.** A YOLOv8 detector finds every raised dot and registers the page, so it knows which
  cell your finger is on. **MediaPipe's hand model** finds your index fingertip, across skin tones and lighting.
- **Lessons that feel like lessons.** Six short lessons (A to E, F to J, K to O, P to T, U to Z, then look-alikes). Each letter is
  described by its dots and how it grows from one you know; you find it and rest your finger on it, or point and say "this one".
  Then a mixed practice, and, if it went well, you **spell a real word** by touch. Mistakes are explained ("That's F. D has a dot at
  the middle-right…"), and letters that went wrong come back at the start of the next lesson.
- **Read and Quiz.** **Read** speaks a word aloud when you rest a finger on it. **Quiz** names a look-alike letter, and you point
  at it and say "this one", or its name.
- **Braillo knows what's going on.** Say "Braillo" and ask anything, or just talk: with an Anthropic API key, Claude works out what
  you meant, knowing the page you're on, what the camera sees, which letter your finger is on, your progress, and what he's
  waiting for you to do. ("I can't find it" gets a hint; "can we move on?" goes to the next lesson.)
- **A phone is the camera.** Scan a QR code and your phone becomes Braillie's eyes, speaker and microphone. On Wi-Fi that blocks
  it, a backup link over a tunnel takes over by itself.
- **Speech that starts straight away.** Braillo's voice (Deepgram Aura 2) plays as it's made, on the laptop or the phone, with
  captions that follow along sentence by sentence.
- **Built for blind and low-vision users.** Screen-reader-friendly markup, every action on a key as well as by voice, WCAG AAA
  colour contrast, light, dark and two high-contrast themes, and four text sizes.
- **Remembers your progress.** Google sign-in keeps it across sessions. Guests go straight in, and nothing of theirs is saved.

<table>
<tr>
<td width="50%"><img src="docs/img/screenshot-signin.png" alt="Sign-in: Braillo beside 'You can learn new letters, read some words, or take a quiz', and two big keys: Sign in with Google, or Continue as guest."><br><sub><b>Sign in</b> by voice ("Google" or "guest") or with the keys.</sub></td>
<td width="50%"><img src="docs/img/screenshot-connect.png" alt="Connect your phone: Braillo saying 'To connect your phone, scan the code on the screen with its camera, open the link, and tap Start' beside a large QR code."><br><sub><b>Connect your phone</b>: scan the code; Braillo talks you through it.</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/img/screenshot-modes.png" alt="Home: Braillo listening, with 'Good morning, Guest! What shall we do?', and three big keys: Learn (F to J next), Read and Quiz."><br><sub><b>Home</b>: say learn, read or quiz.</sub></td>
<td width="50%"><img src="docs/img/screenshot-lesson.png" alt="A lesson: Braillo saying 'G is dots 1, 2, 4, 5', the letter G with its braille cell, progress dots, and Found it, Hint, Next and Finish keys."><br><sub><b>A lesson</b>: find the letter by touch, with hints when you need them.</sub></td>
</tr>
<tr>
<td colspan="2" align="center"><img src="docs/img/screenshot-account.png" alt="Your page: 'Hi, Guest', Braillo, three numbers (letters known, days in a row, sessions) and the alphabet, each letter with a bar showing how well it is known." width="620"><br><sub><b>Your page</b>: Braillo reads out how you're doing; say "letters" to hear your alphabet.</sub></td>
</tr>
</table>

## Try it in a few minutes

You'll need Python 3.10+, Node 20+, a webcam or a phone, and a free [Deepgram](https://deepgram.com) API key (it does the
speaking and listening). An [Anthropic](https://console.anthropic.com) API key is optional: it lets Braillo answer questions and
understand free speech; everything else works without it.

```bash
git clone https://github.com/MrFiscus/Braillie.git
cd Braillie

# Python side: detection, finger tracking, the tutor, the phone link
python3 -m venv braille_tutor/.venv && source braille_tutor/.venv/bin/activate
pip install -r braille_tutor/requirements.txt -r requirements-voice.txt
git clone https://github.com/snoop2head/DotNeuralNet braille_tutor/third_party/DotNeuralNet

# Website
cd website-frontend && npm install && cd ..

# Your keys (never commit this file)
cp .env.example .env   # then fill in DEEPGRAM_API_KEY (and ANTHROPIC_API_KEY for Braillo's questions)
```

Then, from the repo root, with the venv active:

```bash
./run_all.sh
```

This starts the tutor and the website together and opens **http://localhost:5173**. Braillo greets you there. Sign in (or go in as
a guest), scan the QR code with your phone, and choose Learn, Read or Quiz. MediaPipe's hand model (about 8 MB) is downloaded by
itself the first time; without it, Braillie falls back to finding the finger by skin colour.

On a network that blocks phone-to-laptop traffic (most public Wi-Fi), Braillie switches to a **backup link** by itself: if
[`cloudflared`](https://github.com/cloudflare/cloudflared) is installed, a tunnel is made ready at start, and when no phone has reached
the laptop 40 seconds after the code is shown, the QR code changes to it (or press the globe key on the Connect Phone page). For a demo on
poor Wi-Fi, the most reliable setup is still the phone's hotspot for both devices. `./run_all.sh --phone-tunnel` uses the tunnel from
the start; see [`braille_tutor/README.md`](braille_tutor/README.md#phone-as-the-camera-scan-a-qr-code---phone-camera) for every
connection mode and what each trades off.

Only one browser tab talks to Braillo at a time: the one you're using.

## What you can say

| Anywhere | In a lesson or the quiz | Getting around |
|---|---|---|
| **help** (a sentence) · **more options** (everything) | **this one** / **found it** (or the letter's name) | any button's name, e.g. **start learning** |
| **repeat** · **quiet** · **slower** · **faster** | **hint** · **next** (skip) · **repeat** | **go back** · **scroll down** · **scroll up** |
| **Braillo**, then any question | **next lesson** · **previous lesson** · **lesson three** | **home** · **my page** · **sign out** |
| **learn** · **read** · **quiz** | **explore** · **practice** · **stop** | |

## Print a sheet and try it today

The four practice sheets are already generated and checked into the repo. Click a thumbnail below to open and download the
full-resolution, print-ready PNG (A4, 300 DPI, print at **100% / no scaling**).

<table>
<tr>
<td align="center" width="25%">
<a href="braille_tutor/sheet/nomarkers/poke_alphabet.png"><img src="docs/img/sheet-alphabet.png" width="200" alt="Alphabet A to Z: a close-up of the raised-dot pattern for each letter"></a><br>
<b>Alphabet A–Z</b><br><sub>used by <b>Learn</b> · <a href="braille_tutor/sheet/nomarkers/poke_alphabet.png">download</a></sub>
</td>
<td align="center" width="25%">
<a href="braille_tutor/sheet/nomarkers/poke_words.png"><img src="docs/img/sheet-words.png" width="200" alt="Words sheet: a close-up of a few printed words in braille"></a><br>
<b>Words</b><br><sub>used by <b>Read</b> · <a href="braille_tutor/sheet/nomarkers/poke_words.png">download</a></sub>
</td>
<td align="center" width="25%">
<a href="braille_tutor/sheet/nomarkers/poke_lookalikes.png"><img src="docs/img/sheet-lookalikes.png" width="200" alt="Look-alike letters: a close-up of letters that are easy to mix up"></a><br>
<b>Look-alikes</b><br><sub>used by <b>Quiz</b> · <a href="braille_tutor/sheet/nomarkers/poke_lookalikes.png">download</a></sub>
</td>
<td align="center" width="25%">
<a href="braille_tutor/sheet/nomarkers/poke_numbers.png"><img src="docs/img/sheet-numbers.png" width="200" alt="Numbers and signs: a close-up of digits 0 to 9 and the number sign"></a><br>
<b>Numbers &amp; signs</b><br><sub>extra practice · <a href="braille_tutor/sheet/nomarkers/poke_numbers.png">download</a></sub>
</td>
</tr>
</table>

**To make a physical page:** print one at 100% scale, tape it face-down onto a corkboard, foam sheet or thick cardboard, and poke
each black dot through from the back with anything blunt (a used ballpoint pen, a knitting needle, an unfolded paperclip). Flip it
over: the raised bumps on the front are what you feel and what the camera reads. If you can't do this yourself, ask someone to print
and poke a sheet for you. Hold the whole page flat under the camera on a plain, contrasting surface.

These are the *no-markers* sheets: `./run_all.sh` already runs with `--paper`, so Braillie finds the page from its own four edges.
(A version with corner registration markers also exists in [`braille_tutor/sheet/`](braille_tutor/sheet/) for more robust tracking in
tricky lighting; see [`braille_tutor/README.md`](braille_tutor/README.md#reading-our-printed--poked-sheets-correctly---sheet), plus
`--mode explore` for reading a real embossed braille book page.)

## How it works

The phone (or a webcam) streams frames to the laptop. A YOLOv8 detector finds every braille cell and its raised dots, a
page-registration step maps pixels to real page millimetres, and MediaPipe's hand landmarks give the index fingertip, tracked and
smoothed in page millimetres. The Python tutor (`braille_tutor/tutor.py`, lessons in `learn.py`) decides what to say and drives Learn,
Read and Quiz. Deepgram (`voice_io.py`, at the root) turns its words into speech, streamed as it's made, and your voice into commands;
anything that isn't a command goes to Claude (`llm.py`), which is told what's happening and what's expected next.

The React website polls the tutor's local HTTP API for its state, and tells it which page is open and the names of the controls on
it, so any of them can be used by voice. Braillo, drawn in SVG, breathes, blinks, looks around, leans in to listen, and hops for joy at
a right answer; everything holds still for anyone who has asked their device for less motion.

## Repo layout

```
braille_tutor/        Detection (YOLOv8 + page registration), finger tracking (MediaPipe), the tutor and its lessons,
                       the phone link (QR pairing, certificates, tunnel, streamed speech), the local HTTP API,
                       Braillo's Claude helper (llm.py), and sheet/: the printable practice sheets above.
                       See braille_tutor/README.md (full detail) and API.md (the HTTP API).
website-frontend/     React + TypeScript + Vite: the landing page, voice-driven sign-in, phone pairing, home, the
                       activities and your page; src/voice/ holds the voice control for the whole site.
voice_io.py            Deepgram speech in and out (streamed), used by the tutor.
run_all.sh              Starts the tutor and the website together (the way to run it above).
requirements-voice.txt  Root-level Python deps for voice_io.py.
backend/, run_server.py An earlier prototype (a separate WebSocket server); superseded by
                       braille_tutor/tutor_server.py + run_all.sh above and no longer used.
```

## Documentation

- [`braille_tutor/README.md`](braille_tutor/README.md): the full technical write-up: every CLI flag and mode, how page
  registration and locking work, reading a real embossed book page, the accessibility features, the phone link's connection
  modes, the AI coach, and known limits.
- [`braille_tutor/API.md`](braille_tutor/API.md): the tutor's local HTTP API, for anyone building another frontend against it.

## Testing

```bash
cd braille_tutor && python -m unittest discover -p "test_*.py"   # 640+ tests: detection, finger tracking, lessons, tutor, phone link, voice
cd website-frontend && node --test "tests/*.test.mjs"            # sign-in voice, profile voice, colour contrast, progress sync
```

(The `test_*.py` files at the repo root are hand-run voice scripts: `test_speak.py` speaks through Deepgram and
`test_mic_pin.py` opens the live microphone, so run them on purpose, not with pytest.)

## Team

Built at HACKMIT by **Smaran Pokharel** (braille detection, the tutor, phone pairing, the accessible website), **Tenzing
Gurung** (voice I/O, word-checking, demo-night reliability fixes), **Reagan Spurlock** (the first version of the website), and
**Sage Peterson** (the physical braille sheets, video and technical support).

## Credits

Detection uses the pretrained YOLOv8 weights from [DotNeuralNet](https://github.com/snoop2head/DotNeuralNet) by snoop2head (MIT
License), cloned separately, not modified. Finger tracking uses Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe)
hand landmarker (Apache License 2.0). See [`braille_tutor/README.md`](braille_tutor/README.md#credits) for the full attribution.
