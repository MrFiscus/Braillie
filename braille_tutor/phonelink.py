"""Use a phone as the camera: the laptop shows a QR code, the phone scans it, and the phone's camera streams to the tutor.

    laptop                                                     phone
    PhoneCamera (looks like cv2.VideoCapture)                  scan the QR (or type the address + code) -> web/phone.html
    PhoneLink   (newest frame + who is connected)   <- JPEG frames over HTTPS, about 8 a second, with the laptop's page
    PhoneServer (HTTPS on the local network)        -> status back ("page found" / "hold the phone higher"), which the phone speaks

Why HTTPS with a certificate the phone has to be told to trust: phone browsers only give a web page the camera on HTTPS. The
certificate is made once with the `openssl` command (no extra packages) and lives in ~/.braillie. The phone shows a "connection
is not private" warning the first time; tap Advanced / Show details, then continue. That is the one step that cannot be
removed without a real domain name, and it is the part a helper may need to do.

Safety: this is the only part of the tutor reachable from the network, so it is small and locked down. It serves exactly two
things (the phone page, and frame uploads), both need the session's 6-digit code, wrong codes are throttled, uploads are size
limited and must decode as ordinary images, and the tutor's own API stays on localhost.

Sound: the laptop still makes the speech (your friend's Deepgram voice in voice_io) but the finished audio is sent to the phone
and played there, where the user is (`--sound phone`, the default with the phone camera; `--sound laptop` keeps it on the laptop).
A phone page may only play sound after a tap, so the phone page's one big button starts the camera AND unlocks the sound. Until the
phone says its sound is ready (and whenever it is not connected) speech plays on the laptop as before, so nothing is ever silent.

Microphone: the phone's microphone is used for the spoken commands too (`--mic phone`, the default with the phone camera). The page
sends 16 kHz mono 16-bit audio; the laptop wraps the one place voice_io reads its microphone (pyaudio's `Stream.read`, every
100 ms) so that the chunk sent on to Deepgram comes from the phone while it is streaming audio and from the laptop microphone
otherwise. Nothing else in voice_io changes: same Deepgram connection, same command matching, same muting while it speaks, and the
laptop microphone is the automatic fallback.

For someone who cannot see a QR code: the laptop says a short "scan the QR on the screen" line (`spoken_instructions`);
an iPhone with VoiceOver can also read a QR code aloud from its Camera app. Someone sighted can scan it for them.
Details (address, code, privacy warning) stay on the laptop screen.
"""
from __future__ import annotations

import collections
import errno
import os
import hmac
import json
import re
import secrets
import socket
import io
import atexit
import datetime as dt
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

DEFAULT_PORT = 8443
CERT_DIR = Path.home() / ".braillie"
PHONE_PAGE = Path(__file__).parent / "web" / "phone.html"
MAX_FRAME_BYTES = 3_000_000
MAX_SIDE_PX = 4096
STALE_SECONDS = 3.0  # no frame for this long: the phone counts as disconnected
MIC_RATE = 16000  # what the phone page sends and voice_io (Deepgram STT) expects: mono, 16-bit little-endian
MIC_LIVE_SECONDS = 1.5  # no audio for this long: the phone microphone counts as off
MIC_MAX_BACKLOG_SECONDS = 0.6  # keep the newest audio only, so a command is never heard late
MAX_MIC_CHUNK_BYTES = 200_000
MIC_TAIL_SECONDS = 0.8  # after the phone has spoken, its microphone audio is dropped for this long (the tail of the tutor's own voice)
MAX_BAD_CODES = 20  # this many wrong codes in a row locks phone connections out for LOCKOUT_SECONDS
LOCKOUT_SECONDS = 60.0
TUNNEL_URL = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
TUNNEL_SECONDS = 30.0  # how long to wait for the tunnel to say its address
LOCAL_IP_DOMAIN = "local-ip.sh"  # a public name service: 192-168-1-22.local-ip.sh points at 192.168.1.22, with a real certificate
LOCAL_IP_CERT_URL, LOCAL_IP_KEY_URL = f"https://{LOCAL_IP_DOMAIN}/server.pem", f"https://{LOCAL_IP_DOMAIN}/server.key"
TRUSTED_DIR = CERT_DIR / "local-ip"
CERT_REFRESH_DAYS = 7  # a saved copy of that certificate is used for this long before a fresh one is fetched
CERT_MIN_DAYS_LEFT = 2  # a certificate that runs out sooner than this is not used


def lan_ip() -> str:
    """This computer's address on the local network (the one a phone on the same Wi-Fi can reach), best effort."""
    for target in ("8.8.8.8", "10.255.255.255", "192.168.255.255"):  # connecting a UDP socket sends nothing: it just picks the route
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            continue
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def spell_out(text: str) -> str:
    """Digits and dots spoken one at a time, for a screen reader or a voice: '10.0.0.5' -> '1 0 dot 0 dot 0 dot 5'."""
    return " ".join("dot" if ch == "." else ch for ch in text)


# ---- the QR code --------------------------------------------------------------------------------

def qr_modules(text: str) -> np.ndarray:
    """The QR code for `text` as a boolean grid (True = dark), one entry per module, with a 2-module quiet border."""
    img = cv2.QRCodeEncoder.create().encode(text)
    if img is None or img.size == 0:
        raise ValueError("could not make a QR code (text too long?)")
    return img < 128


def qr_image(text: str, size: int = 480) -> np.ndarray:
    """A grayscale picture of the QR code, `size` px square (nearest-neighbour scaled so the edges stay sharp), with the full
    4-module quiet zone scanners like."""
    dark = qr_modules(text)
    scale = max(1, size // (dark.shape[0] + 4))
    img = np.where(dark, 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(img, 2 * scale, 2 * scale, 2 * scale, 2 * scale, cv2.BORDER_CONSTANT, value=255)


def qr_png(text: str, size: int = 480) -> bytes:
    ok, buf = cv2.imencode(".png", qr_image(text, size))
    if not ok:
        raise ValueError("could not encode the QR code")
    return buf.tobytes()


def qr_terminal(text: str) -> str:
    """The QR code drawn with block characters, for a terminal (two rows of modules per line)."""
    dark = np.pad(qr_modules(text), 2, constant_values=False)
    if dark.shape[0] % 2:
        dark = np.pad(dark, ((0, 1), (0, 0)), constant_values=False)
    lines = []
    for y in range(0, dark.shape[0], 2):
        lines.append("".join({(False, False): " ", (True, False): "▀", (False, True): "▄", (True, True): "█"}[(bool(a), bool(b))]
                             for a, b in zip(dark[y], dark[y + 1])))
    return "\n".join(lines)  # dark = block: best on a light-on-dark terminal, which most are; inverted terminals still scan


def fix_wav_header(data: bytes) -> bytes:
    """Deepgram streams its WAV with a header that claims a length of about 2 GB (about 12 hours of audio) while the file is a few
    hundred KB. Players may trust that, and so did the length calculation. Rewrite the two size fields to what is really there."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    at = data.find(b"data", 12)
    if at < 0:
        return data
    body = len(data) - (at + 8)
    out = bytearray(data)
    out[4:8] = (len(data) - 8).to_bytes(4, "little")
    out[at + 4:at + 8] = body.to_bytes(4, "little")
    return bytes(out)


def audio_seconds(data: bytes, kind: str) -> float:
    """How long a clip plays: from the bytes actually present for WAV (never from a header that may lie), an estimate
    (128 kbit/s) for MP3 and anything unreadable."""
    if kind == "audio/wav":
        try:
            data = fix_wav_header(data)
            with wave.open(io.BytesIO(data)) as wf:
                return wf.getnframes() / float(wf.getframerate())
        except (wave.Error, EOFError, ZeroDivisionError):
            pass
    return len(data) / 16000.0


def install_phone_speaker(voice, link: "PhoneLink") -> Callable:
    """Redirect voice_io's finished audio (Deepgram WAV, ElevenLabs MP3) to the phone whenever the phone can play it, else leave it
    to voice_io's own laptop playback. Only the two playback functions are swapped; the synthesis (your friend's voices) is
    untouched. Returns a function that puts everything back."""
    names = ("_play_audio_stream", "_play_mp3_stream")
    missing = [n for n in names if not callable(getattr(voice, n, None))]
    if missing:
        raise RuntimeError(f"voice_io no longer has {', '.join(missing)}: cannot move the sound to the phone (use --sound laptop)")
    original = {n: getattr(voice, n) for n in names}

    def routed(name: str, kind: str):
        def play(chunks) -> None:
            data = fix_wav_header(b"".join(chunks)) if kind == "audio/wav" else b"".join(chunks)
            if not link.play_audio(data, kind):
                original[name]([data])  # the phone is not ready or could not play it: the laptop does
        return play

    voice._play_audio_stream = routed("_play_audio_stream", "audio/wav")
    voice._play_mp3_stream = routed("_play_mp3_stream", "audio/mpeg")

    def restore() -> None:
        for n in names:
            setattr(voice, n, original[n])
    return restore


def install_phone_mic(voice, link: "PhoneLink") -> Callable:
    """Make voice_io hear the phone's microphone whenever the phone is streaming it (laptop microphone otherwise). voice_io reads its
    microphone with pyaudio's Stream.read, one 100 ms chunk at a time, and sends each chunk to Deepgram: that read is wrapped so the
    chunk it returns comes from the phone. The laptop read still happens (it paces the loop and is the fallback). Returns a function
    that undoes it. Raises RuntimeError if this cannot be done safely (then use --mic laptop)."""
    if getattr(voice, "SAMPLE_RATE", None) != MIC_RATE or getattr(voice, "CHANNELS", None) != 1:
        raise RuntimeError(f"voice_io listens at {getattr(voice, 'SAMPLE_RATE', '?')} Hz x {getattr(voice, 'CHANNELS', '?')} channel(s), "
                           f"but the phone microphone is {MIC_RATE} Hz mono: use --mic laptop")
    try:
        import pyaudio
    except ImportError as e:
        raise RuntimeError(f"pyaudio is not installed, so voice_io has no microphone loop to redirect ({e})") from e
    stream_class = getattr(pyaudio, "Stream", None)
    if stream_class is None or not callable(getattr(stream_class, "read", None)):
        raise RuntimeError("pyaudio.Stream.read is missing: cannot move the microphone to the phone (use --mic laptop)")
    original = stream_class.read

    def read(self, num_frames, exception_on_overflow=True):
        data = original(self, num_frames, exception_on_overflow)  # the laptop microphone: paces the loop, and is the fallback
        phone = link.mic_take(len(data))
        return phone if phone is not None and len(phone) == len(data) else data

    stream_class.read = read

    def restore() -> None:
        stream_class.read = original
    return restore


# ---- state shared between the phone's uploads and the tutor -----------------------------------------

class TunnelError(RuntimeError):
    """The tunnel could not be started; the message says why and what to do."""


class Tunnel:
    """A public https address that reaches the phone server, with a real certificate from Cloudflare's quick tunnels, so the phone shows no
    "connection is not private" warning and the laptop and phone need not be on the same network. It is the `cloudflared` program (free, no
    account) running next to the tutor; it stops when the tutor does. The address is random and only in the QR code; everything the phone
    sends still needs the session code."""

    def __init__(self, proc: subprocess.Popen, url: str, reader: Optional[threading.Thread] = None):
        self.proc, self.url, self.reader = proc, url, reader

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        _end(self.proc, self.reader)


# The tunnel program runs under this small shell script, which also watches the tutor: if the tutor goes away in ANY way (Ctrl+C, being
# killed, the terminal closing, a crash) the tunnel is shut down too, so a public address is never left open behind it. ($0 is the program, $1
# the local address; stopping the script stops the program.)
_WATCHDOG = """
"$0" tunnel --url "$1" --no-autoupdate &
child=$!
trap 'kill $child 2>/dev/null; exit' TERM INT HUP
( while kill -0 $PPID 2>/dev/null && kill -0 $child 2>/dev/null; do sleep 1; done; kill $child 2>/dev/null ) &
wait $child
"""


def _end(proc: subprocess.Popen, reader: Optional[threading.Thread] = None) -> None:
    """Stop the program and tidy up after it (wait for it, let the reader finish, close the pipe), so nothing is left running or open."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if reader is not None:
        reader.join(timeout=2)
    if proc.stdout is not None and (reader is None or not reader.is_alive()):
        proc.stdout.close()


def find_cloudflared() -> Optional[str]:
    """The cloudflared program: on the PATH, or a copy in ~/.braillie."""
    return shutil.which("cloudflared") or (str(CERT_DIR / "cloudflared") if (CERT_DIR / "cloudflared").is_file() else None)


INSTALL_HINT = ("Install it once with `brew install cloudflared` (macOS), or download it from "
                "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ and put it on your PATH.")


def start_tunnel(port: int, binary: Optional[str] = None, timeout: float = TUNNEL_SECONDS) -> Tunnel:
    """Start a quick tunnel to http://127.0.0.1:`port` and return it once it has its public address. Raises TunnelError."""
    binary = binary or find_cloudflared()
    if not binary:
        raise TunnelError("the phone tunnel needs the `cloudflared` program, which is not installed. " + INSTALL_HINT)
    try:
        proc = subprocess.Popen(["/bin/sh", "-c", _WATCHDOG, binary, f"http://127.0.0.1:{port}"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as e:
        raise TunnelError(f"could not run {binary}: {e}") from e
    found, tail = threading.Event(), collections.deque(maxlen=6)
    box: list = []

    def read() -> None:  # keeps reading for as long as the program runs, so its output never fills the pipe and stalls it
        for line in proc.stdout:
            tail.append(line.strip())
            if not box and (m := TUNNEL_URL.search(line)):
                box.append(m.group(0))
                found.set()
        found.set()  # the program ended

    reader = threading.Thread(target=read, daemon=True, name="tunnel-output")
    reader.start()
    found.wait(timeout)
    if not box:
        stopped = proc.poll() is not None
        _end(proc, reader)
        why = "it stopped: " + " | ".join(t for t in tail if t)[-300:] if stopped and tail else "it printed no address in time"
        raise TunnelError(f"the phone tunnel did not start ({why}). Does this computer have an internet connection?")
    tunnel = Tunnel(proc, box[0], reader)
    atexit.register(tunnel.stop)
    return tunnel


class PhoneLink:
    """The newest frame from the phone, whether it is connected, and the session code. Thread-safe."""

    def __init__(self, ip: Optional[str] = None, port: int = DEFAULT_PORT, code: Optional[str] = None, tls: bool = True,
                 on_connect: Optional[Callable] = None, on_disconnect: Optional[Callable] = None, clock: Callable = time.monotonic,
                 on_audio_ready: Optional[Callable] = None):
        self.ip, self.port, self.tls = ip or lan_ip(), port, tls
        self.public_base: Optional[str] = None  # set when a tunnel gives the phone a public https address (see Tunnel)
        self.trusted_name: Optional[str] = None  # set when a name with a real certificate points at this computer (see local_ip_certificate)
        self.code = code or f"{secrets.randbelow(10**6):06d}"
        self.on_connect, self.on_disconnect, self.clock = on_connect, on_disconnect, clock
        self.cond = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.seq, self.last_frame_time, self.frames, self._connected = 0, -1e9, 0, False
        self.status: dict = {"page_ok": False, "message": "waiting for the camera"}  # what the phone is told: set by the tutor
        self._bad, self._locked_until = 0, 0.0
        self._lock = threading.Lock()
        # what has happened so far, to tell WHERE a phone that "doesn't work" got stuck (see diagnose())
        self.events: collections.deque = collections.deque(maxlen=50)
        self.contacts: set = set()  # addresses of devices that have reached the phone port at all
        self.page_opens, self.camera_error, self.started, self.verbose = 0, "", clock(), True
        # sound played on the phone: clips waiting for it, and whether its page has unlocked sound (needs a tap on iOS)
        self.audio_ready, self.on_audio_ready, self.audio_slack = False, on_audio_ready, 5.0
        self._mic = bytearray()  # audio from the phone waiting to be sent on to Deepgram
        self._mic_lock = threading.Lock()
        self.mic_last, self._mic_was_live, self.mic_bytes, self._mic_mute_until = -1e9, False, 0, 0.0
        self._clips: dict = {}  # id -> {"data", "type", "fetched", "done", "error"}
        self._clip_id = 0
        self._clip_lock = threading.Lock()

    # -- what happened --
    def note(self, text: str) -> None:
        """Record (and print, so the laptop's terminal shows it) one step of the phone's progress."""
        self.events.append((self.clock(), text))
        if self.verbose:
            print(f"[phone] {text}", flush=True)

    def contact(self, ip: str) -> None:
        if ip not in self.contacts:
            self.contacts.add(ip)
            self.note(f"a device at {ip} reached the laptop")

    def diagnose(self) -> Optional[str]:
        """Plain words for why no video is arriving yet, from how far the phone got. None while it is connected (or too early to say)."""
        if self.connected:
            return None
        if self.frames:
            return "The phone stopped sending video. Keep its screen on and the page open, then scan the code again if it does not come back."
        if self.camera_error:
            return (f"The phone opened the page but its camera did not start ({self.camera_error}). Allow camera access for the page "
                    "in the phone's browser settings, then reload it.")
        if self.page_opens:
            return "The phone opened the page but no video has arrived yet. If it asks to use the camera, choose Allow."
        if self.tunnelled:
            if self.clock() - self.started > 15:
                return ("No phone has opened the link yet. Scan the code with the phone's camera and open the link. The phone needs "
                        "an internet connection (Wi-Fi or mobile data); it does not have to be on the same network as the laptop.")
            return None
        if self.trusted_name:
            if self.frames == 0 and self.contacts and not self.page_opens:
                return "A device reached the laptop but never opened the page. Open the link from the QR code again."
            if not self.contacts and self.clock() - self.started > 15:
                return ("No phone has reached this laptop yet. The phone and laptop must be on the same Wi-Fi, and many shared networks block "
                        "that. Some routers also refuse names that point at a local address (\"DNS rebinding protection\"): try the phone's "
                        f"hotspot for both, or start the tutor with --phone-tunnel. The address is {self.base}.")
            return None
        if self.contacts:
            return ("A device reached the laptop but never opened the page. If your phone showed a privacy warning, choose Advanced (or "
                    "Show Details), then continue to the website.")
        if self.clock() - self.started > 15:
            return ("No phone has reached this laptop yet. The phone and laptop must be on the same Wi-Fi, and many shared networks block "
                    f"that. Try the phone's hotspot for both, or open {self.base} in the phone's browser to test.")
        return None

    # -- addresses --
    @property
    def base(self) -> str:
        if self.trusted_name:
            return f"https://{self.trusted_name}:{self.port}"
        return self.public_base or f"{'https' if self.tls else 'http'}://{self.ip}:{self.port}"

    @property
    def tunnelled(self) -> bool:
        return self.public_base is not None

    @property
    def no_warning(self) -> bool:
        """True when the phone will not be shown a "connection is not private" page: a real certificate, from a tunnel or for a real name."""
        return self.tunnelled or bool(self.trusted_name)

    @property
    def url(self) -> str:
        """The address in the QR code: the phone page with the code already in it."""
        return f"{self.base}/phone?t={self.code}"

    def spoken_instructions(self) -> str:
        """Short line spoken on the QR screen: just tell them to scan. Details stay on screen for sighted helpers."""
        return "Scan the QR code on the screen using your mobile phone."

    # -- codes --
    def check_code(self, given: Optional[str]) -> bool:
        """Is `given` this session's code? Throttled: after MAX_BAD_CODES wrong ones in a row nothing is accepted for a minute."""
        with self._lock:
            now = self.clock()
            if now < self._locked_until:
                return False
            if given and hmac.compare_digest(given.encode(), self.code.encode()):
                self._bad = 0
                return True
            self._bad += 1
            if self._bad >= MAX_BAD_CODES:
                self._locked_until, self._bad = now + LOCKOUT_SECONDS, 0
            return False

    # -- frames --
    @property
    def connected(self) -> bool:
        return self.clock() - self.last_frame_time < STALE_SECONDS

    def push(self, jpeg: bytes) -> bool:
        """Take one uploaded JPEG. False if it is too big or not an ordinary image (nothing is stored then)."""
        if not jpeg or len(jpeg) > MAX_FRAME_BYTES:
            return False
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > MAX_SIDE_PX or min(frame.shape[:2]) < 32:
            return False
        was = self._connected
        if not self.frames:
            self.note("video is arriving from the phone")
        with self.cond:
            self.frame, self.seq, self.last_frame_time = frame, self.seq + 1, self.clock()
            self.frames += 1
            self._connected = True
            self.cond.notify_all()
        if not was and self.on_connect:
            threading.Thread(target=self.on_connect, daemon=True).start()
        return True

    # -- the phone's microphone --
    def mic_live(self) -> bool:
        """Is the phone sending microphone audio right now?"""
        return self.clock() - self.mic_last < MIC_LIVE_SECONDS

    def mic_push(self, pcm: bytes) -> bool:
        """Take a chunk of the phone's microphone (16 kHz mono int16). Only the newest MIC_MAX_BACKLOG_SECONDS are kept."""
        if not pcm or len(pcm) > MAX_MIC_CHUNK_BYTES or len(pcm) % 2:
            return False
        with self._mic_lock:
            if not self.mic_live():
                self._mic.clear()  # audio that was waiting from before a gap is stale
            self.mic_last = self.clock()  # still alive, even while its audio is being dropped below
            if self.clock() < self._mic_mute_until:
                return True  # the tutor was just speaking on this phone: what the microphone heard is its own voice
            self._mic += pcm
            keep = int(MIC_RATE * 2 * MIC_MAX_BACKLOG_SECONDS)
            if len(self._mic) > keep:
                del self._mic[:len(self._mic) - keep]
            self.mic_bytes += len(pcm)
        if not self._mic_was_live:
            self._mic_was_live = True
            self.note("the phone's microphone is on: spoken commands are heard through it")
        return True

    def mic_take(self, nbytes: int) -> Optional[bytes]:
        """The next `nbytes` of phone audio, or None when the phone is not streaming (or has not yet sent enough): the caller
        then uses the laptop microphone's chunk for that moment instead."""
        with self._mic_lock:
            if self._mic_was_live and not self.mic_live():
                self._mic_was_live = False
                self._mic.clear()
                self.note("the phone's microphone stopped: the laptop microphone is used again")
            if not self.mic_live() or len(self._mic) < nbytes:
                return None
            out = bytes(self._mic[:nbytes])
            del self._mic[:nbytes]
            return out

    # -- sound on the phone --
    def sound_ready(self) -> bool:
        """Can speech go to the phone right now? Its page must have unlocked sound and still be sending video."""
        return self.audio_ready and self.connected

    def set_audio_ready(self) -> None:
        first = not self.audio_ready
        self.audio_ready = True
        if first:
            self.note("the phone's sound is on: speech now plays on the phone")
            if self.on_audio_ready:
                threading.Thread(target=self.on_audio_ready, daemon=True).start()

    def next_clip(self) -> Optional[dict]:
        """The oldest clip the phone has not fetched yet, as {"id", "type"} (told to the phone in every frame reply), or None."""
        with self._clip_lock:
            for cid in sorted(self._clips):
                if not self._clips[cid]["fetched"]:
                    return {"id": cid, "type": self._clips[cid]["type"]}
        return None

    def take_clip(self, cid: int) -> Optional[dict]:
        with self._clip_lock:
            clip = self._clips.get(cid)
            if clip:
                clip["fetched"], clip["fetched_at"] = True, self.clock()
            return clip

    def clip_done(self, cid: int, error: str = "") -> None:
        with self._clip_lock:
            clip = self._clips.get(cid)
        if clip:
            clip["error"] = error
            clip["done"].set()

    def play_audio(self, data: bytes, kind: str = "audio/wav") -> bool:
        """Send one finished clip to the phone and WAIT until it has played (speech blocks while it talks, so the microphone stays
        muted and lines never overlap). False means it did NOT play there (the phone is not ready, never picked the clip up, or could
        not play it): the caller then plays it on the laptop instead. Once the phone has the clip it is trusted to play it, even
        through a blip in the video connection: replaying it on the laptop would say every line twice."""
        if not self.sound_ready():
            return False
        with self._clip_lock:
            self._clip_id += 1
            cid = self._clip_id
            clip = self._clips[cid] = {"data": data, "type": kind, "fetched": False, "done": threading.Event(), "error": ""}
        length, offered = audio_seconds(data, kind), self.clock()
        try:
            while not clip["done"].wait(0.1):
                if clip["fetched"]:
                    if self.clock() > clip["fetched_at"] + length + self.audio_slack:
                        self.note("the phone took a clip but never said it finished: assuming it played")
                        return True
                elif self.clock() > offered + self.audio_slack or not self.connected:
                    self.audio_ready = False  # it never picked the clip up: the laptop speaks until the phone says its sound is on again
                    self.note("the phone did not pick up a clip: speech goes back to the laptop until its sound is on again")
                    return False
            if clip["error"]:
                self.audio_ready = False
                self.note(f"the phone could not play sound ({clip['error']}): speech goes back to the laptop")
                return False
            return True
        finally:
            with self._clip_lock:
                self._clips.pop(cid, None)
            with self._mic_lock:  # the phone's microphone just heard the end of this clip: do not let it be taken for a command
                self._mic.clear()
                self._mic_mute_until = self.clock() + MIC_TAIL_SECONDS

    def check_connection(self) -> None:  # (also notes the first frame)
        """Notice a phone that went quiet (call now and then); fires on_disconnect once."""
        if self._connected and not self.connected:
            self._connected = False
            self.note("the phone stopped sending video")
            if self.on_disconnect:
                threading.Thread(target=self.on_disconnect, daemon=True).start()

    def wait_frame(self, last_seq: int, timeout: float = 0.3) -> tuple:
        """(frame, seq): the newest frame, waiting up to `timeout` for one newer than last_seq. frame is None if there is none."""
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            return (self.frame, self.seq) if self.frame is not None else (None, self.seq)


def waiting_frame(link: PhoneLink, width: int = 1280, height: int = 720, note: str = "") -> np.ndarray:
    """The picture shown while no phone is connected: the QR code and what to do, so the video window itself is the setup screen."""
    img = np.full((height, width, 3), 255, np.uint8)
    qr = cv2.cvtColor(qr_image(link.url, min(height - 80, 560)), cv2.COLOR_GRAY2BGR)
    y0 = (height - qr.shape[0]) // 2
    img[y0:y0 + qr.shape[0], 40:40 + qr.shape[1]] = qr
    x = 80 + qr.shape[1]
    lines = [("Connect your phone camera", 1.4, 3), ("1. Scan this code with your phone's camera", 0.9, 2), ("2. Open the link it shows", 0.9, 2),
             ("3. If asked, allow the camera", 0.9, 2), ("", 0.5, 1),
             (f"Or open {link.base}", 0.7, 1), (f"and enter the code {link.code}", 0.7, 1),
             ("(warning about privacy: choose Advanced, then continue)", 0.55, 1)]
    if note:
        lines += [("", 0.5, 1), (note, 0.7, 2)]
    y = y0 + 40
    for text, scale, thick in lines:
        if text:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thick, cv2.LINE_AA)
        y += int(46 * max(scale, 0.7))
    return img


class PhoneCamera:
    """Looks like a cv2.VideoCapture, so the tutor's camera loops need no changes: read() gives the phone's newest frame, or the
    setup screen (QR code) while no phone is connected."""

    def __init__(self, link: PhoneLink):
        self.link, self._seq, self._waiting, self._released = link, 0, None, False
        self.width = self.height = 0

    def isOpened(self) -> bool:
        return not self._released

    def read(self) -> tuple:
        self.link.check_connection()
        if self._released:
            return False, None
        if self.link.connected:
            frame, self._seq = self.link.wait_frame(self._seq, 0.3)
            if frame is not None:
                return True, frame.copy()
        if self._waiting is None:
            self._waiting = waiting_frame(self.link)
        time.sleep(0.1)
        return True, self._waiting.copy()

    def get(self, prop) -> float:
        return 0.0

    def set(self, prop, value) -> bool:
        return False

    def release(self) -> None:
        self._released = True


# ---- HTTPS ----------------------------------------------------------------------------------------

def ensure_cert(ip: str, directory: Path = CERT_DIR) -> tuple:
    """(certfile, keyfile) for a self-signed certificate valid for `ip`, made with the openssl command if it isn't there yet."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = "phone-" + re.sub(r"[^0-9a-z]", "_", ip.lower())
    cert, key = directory / f"{stem}.pem", directory / f"{stem}.key"
    if cert.exists() and key.exists():
        return str(cert), str(key)
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                        "-days", "365", "-subj", "/CN=braillie", "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1"],
                       check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError("could not make the HTTPS certificate the phone camera needs (is the openssl command installed?): "
                           f"{e}") from e
    key.chmod(0o600)
    return str(cert), str(key)


class TrustedCertError(RuntimeError):
    """A trusted certificate could not be set up; the message says why."""


def local_ip_name(ip: str) -> str:
    """192.168.1.22 -> 192-168-1-22.local-ip.sh (which a public DNS server answers with 192.168.1.22)."""
    if not re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", ip or ""):
        raise TrustedCertError(f"{ip!r} is not an IPv4 address, so it has no {LOCAL_IP_DOMAIN} name")
    return ip.replace(".", "-") + "." + LOCAL_IP_DOMAIN


def _openssl_x509(certfile: str, *args: str) -> str:
    try:
        return subprocess.run(["openssl", "x509", "-in", certfile, "-noout", *args], check=True, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise TrustedCertError(f"could not read the certificate {certfile} (is the openssl command installed?): {e}") from e


def cert_days_left(certfile: str, now: Optional[float] = None) -> float:
    """Days until the (first) certificate in the file expires (negative if it has)."""
    text = _openssl_x509(certfile, "-enddate").strip()  # notAfter=Nov 17 14:51:54 2026 GMT
    try:
        end = dt.datetime.strptime(text.split("=", 1)[1].replace(" GMT", ""), "%b %d %H:%M:%S %Y").replace(tzinfo=dt.timezone.utc)
    except (IndexError, ValueError) as e:
        raise TrustedCertError(f"could not read when the certificate expires: {text!r}") from e
    return (end.timestamp() - (time.time() if now is None else now)) / 86400


def cert_covers(certfile: str, name: str) -> bool:
    """Does the certificate's list of names include `name` (a *.example.com entry covers one label, like a browser would)?"""
    names = re.findall(r"DNS:([^\s,]+)", _openssl_x509(certfile, "-text"))
    name = name.lower()
    return any(n.lower() == name or (n.startswith("*.") and "." in name and name.split(".", 1)[1] == n[2:].lower()) for n in names)


def check_cert_pair(cert: str, key: str, name: str) -> None:
    """The certificate must be for `name`, have days left, and belong to the key. Raises TrustedCertError."""
    if not cert_covers(cert, name):
        raise TrustedCertError(f"the certificate {cert} is not for {name}")
    left = cert_days_left(cert)
    if left < CERT_MIN_DAYS_LEFT:
        raise TrustedCertError(f"the certificate {cert} {'has expired' if left <= 0 else f'runs out in {left:.1f} days'}")
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)
    except (ssl.SSLError, OSError) as e:
        raise TrustedCertError(f"the key {key} does not belong to the certificate {cert}: {e}") from e


def fetch_file(url: str, timeout: float = 15.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # (certificate-checked: this is a normal https download)
        return r.read(200_000)


def local_ip_certificate(ip: str, directory: Optional[Path] = None, fetch: Optional[Callable] = None, say: Callable = print) -> tuple:
    """(name, (certfile, keyfile)) for a name that points at `ip`, with a REAL certificate, so the phone shows no warning and its video and
    sound stay on the local network. It is the public wildcard certificate that local-ip.sh publishes (issued by Let's Encrypt): it is
    kept in ~/.braillie/local-ip, refreshed after a week, and checked (right name, days left, key matches) before use. Because that key is
    public, it gives no more protection against someone actively interfering on the same Wi-Fi than a self-signed certificate does."""
    name, directory, fetch = local_ip_name(ip), directory or TRUSTED_DIR, fetch or fetch_file
    cert, key = directory / "server.pem", directory / "server.key"

    def usable() -> bool:
        try:
            check_cert_pair(str(cert), str(key), name)
            return True
        except (TrustedCertError, OSError):
            return False

    if usable() and time.time() - cert.stat().st_mtime < CERT_REFRESH_DAYS * 86400:
        return name, (str(cert), str(key))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        new_cert, new_key = directory / "server.pem.new", directory / "server.key.new"
        new_cert.write_bytes(fetch(LOCAL_IP_CERT_URL))
        new_key.write_bytes(fetch(LOCAL_IP_KEY_URL))
        new_key.chmod(0o600)
        check_cert_pair(str(new_cert), str(new_key), name)  # only a good pair replaces what was saved
        os.replace(new_key, key)
        os.replace(new_cert, cert)
    except (OSError, urllib.error.URLError, TrustedCertError, ValueError) as e:
        for leftover in (directory / "server.pem.new", directory / "server.key.new"):
            leftover.unlink(missing_ok=True)
        if not usable():
            raise TrustedCertError(f"could not get the trusted certificate from {LOCAL_IP_DOMAIN} ({e}). It needs an internet connection "
                                   "the first time, and once a week after that.") from e
        say(f"[phone] could not refresh the saved {LOCAL_IP_DOMAIN} certificate ({e}); using the saved one")
    return name, (str(cert), str(key))


def own_domain_certificate(domain: str, cert: Optional[str], key: Optional[str]) -> tuple:
    """(name, (certfile, keyfile)) for your own domain name, which must point at this computer (an A record with its address on the local
    network) and have a certificate from a real authority (for example from Let's Encrypt). Checked before use."""
    if not (cert and key):
        raise TrustedCertError("--phone-domain needs --phone-cert and --phone-key (the certificate and its key, as .pem files)")
    for f in (cert, key):
        if not Path(f).is_file():
            raise TrustedCertError(f"{f} does not exist")
    check_cert_pair(cert, key, domain)
    return domain, (cert, key)


def points_here(name: str, ip: str) -> Optional[str]:
    """None if `name` resolves to `ip`; otherwise a sentence saying what happens instead (for the terminal)."""
    try:
        found = socket.gethostbyname(name)
    except OSError as e:
        return f"{name} could not be looked up from this computer ({e}): check the internet connection"
    return None if found == ip else f"{name} points at {found}, not at this computer ({ip}): the phone would not reach it"


def make_handler(link: PhoneLink):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            import json
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _query(self) -> dict:
            return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/phone"):
                # the page itself carries no secret; it asks for the code (or takes it from the QR's link) before doing anything
                link.page_opens += 1
                agent = self.headers.get("User-Agent", "")
                device = "an iPhone or iPad" if re.search(r"iPhone|iPad", agent) else "an Android phone" if "Android" in agent else "a browser"
                link.note(f"{device} opened the phone page")
                self._send(200, PHONE_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/phone/audio":  # the phone fetches a clip it was told about in a frame reply
                q = self._query()
                if not link.check_code(q.get("t")):
                    return self._json(403, {"error": "wrong code"})
                clip = link.take_clip(int(q["id"])) if q.get("id", "").isdigit() else None
                if clip is None:
                    return self._send(404, b"no such clip")
                self._send(200, clip["data"], clip["type"])
            elif path == "/phone/check":  # the page asks whether a typed code is right, before turning the camera on
                ok = link.check_code(self._query().get("t"))
                self._json(200 if ok else 403, {"ok": ok})
            else:
                self._send(404, b"not found")

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/phone/log":  # the phone page reports what its own browser said (camera blocked, no camera API, ...)
                if not link.check_code(self._query().get("t")):
                    return self._json(403, {"error": "wrong code"})
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                    body = json.loads(self.rfile.read(length)) if 0 < length <= 2000 else {}
                except (ValueError, OSError):
                    body = {}
                event, detail = str(body.get("event", ""))[:40], str(body.get("detail", ""))[:200]
                if event:
                    link.note(f"phone says {event}: {detail}" if detail else f"phone says {event}")
                    if event == "camera-error":
                        link.camera_error = detail
                return self._json(200, {"ok": True})
            if path == "/phone/mic":  # microphone audio from the phone page: raw 16 kHz mono int16, a few hundred ms at a time
                if not link.check_code(self._query().get("t")):
                    return self._json(403, {"error": "wrong code"})
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length <= 0 or length > MAX_MIC_CHUNK_BYTES:
                    return self._json(413 if length > MAX_MIC_CHUNK_BYTES else 400, {"error": "bad size"})
                if not link.mic_push(self.rfile.read(length)):
                    return self._json(400, {"error": "audio must be 16 kHz mono 16-bit"})
                return self._json(200, {"ok": True})
            if path in ("/phone/audio-ready", "/phone/audio-done"):
                q = self._query()
                if not link.check_code(q.get("t")):
                    return self._json(403, {"error": "wrong code"})
                if path == "/phone/audio-ready":
                    link.set_audio_ready()
                elif q.get("id", "").isdigit():
                    link.clip_done(int(q["id"]), q.get("error", "")[:80])
                return self._json(200, {"ok": True})
            if path != "/phone/frame":
                return self._send(404, b"not found")
            if not link.check_code(self._query().get("t")):
                return self._json(403, {"error": "wrong code"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_FRAME_BYTES:
                return self._json(413 if length > MAX_FRAME_BYTES else 400, {"error": "bad size"})
            if not link.push(self.rfile.read(length)):
                return self._json(400, {"error": "not a usable image"})
            self._json(200, {"ok": True, **link.status, "say": link.next_clip()})  # the reply tells the phone what to say and play

    return Handler


class _LinkServer(ThreadingHTTPServer):
    """Does the TLS handshake in each connection's own thread (a phone that stalls mid-handshake cannot block the others), and
    records who connected and why a handshake failed: the difference between 'never reached us' and 'refused our certificate'."""
    daemon_threads = True
    link: PhoneLink
    ctx: Optional[ssl.SSLContext] = None

    def process_request_thread(self, request, client_address):
        self.link.contact(client_address[0])
        if self.ctx is not None:
            try:
                request.settimeout(10)
                request = self.ctx.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError) as e:
                reason = str(getattr(e, "reason", None) or e)[:60]
                if reason == "HTTP_REQUEST":
                    self.link.note(f"the device at {client_address[0]} used plain http: the address must start with https://")
                else:
                    self.link.note(f"the device at {client_address[0]} could not finish the secure connection ({reason}); if that is "
                                   "the phone, it refused the certificate: choose Advanced (or Show Details), then continue")
                return self.shutdown_request(request)
        super().process_request_thread(request, client_address)

    def handle_error(self, request, client_address):  # a phone hanging up mid-request is normal, not a traceback
        pass


class PhoneServer:
    """The HTTPS server the phone talks to. Bound to the local network on its own port; the tutor's own API is not on it."""

    def __init__(self, link: PhoneLink, host: str = "0.0.0.0", cert: Optional[tuple] = None):
        self.link = link
        wanted = link.port
        for port in ([wanted] + [wanted + i for i in range(1, 10)] if wanted else [0]):
            try:
                self.httpd = _LinkServer((host, port), make_handler(link))
                break
            except OSError as e:  # busy: most likely an earlier run of the tutor that is still going
                if e.errno != errno.EADDRINUSE or not wanted:
                    raise
        else:
            raise OSError(errno.EADDRINUSE, f"ports {wanted}-{wanted + 9} are all busy: is another copy of the tutor still running?")
        if wanted and self.httpd.server_address[1] != wanted:
            print(f"[phone] port {wanted} is busy (another copy of the tutor still running?): using {self.httpd.server_address[1]} instead. "
                  "The QR code has the right address.", flush=True)
        self.httpd.link = link
        if link.tls:
            cert = cert or ensure_cert(link.ip)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(*cert)
            self.httpd.ctx = ctx
        link.port = self.httpd.server_address[1]  # in case port 0 was asked for

    def start(self) -> "PhoneServer":
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="phone-server").start()
        return self

    def stop(self) -> None:
        tunnel = getattr(self, "tunnel", None)
        if tunnel is not None:
            tunnel.stop()
        self.httpd.shutdown()
        self.httpd.server_close()


# ---- command line ---------------------------------------------------------------------------------

def _hint_loop(link: PhoneLink) -> None:
    """While no phone is connected, print the diagnosis (again whenever it changes) so the laptop's terminal says what is wrong."""
    last = None
    while True:
        time.sleep(3)
        text = link.diagnose()
        if text and text != last:
            print(f"[phone] HINT: {text}", flush=True)
        last = text


def add_phone_args(ap) -> None:
    ap.add_argument("--phone-camera", action="store_true",
                    help="use a phone as the camera: shows a QR code to scan (see phonelink.py). Needs the phone on the same Wi-Fi")
    ap.add_argument("--phone-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--sound", choices=("phone", "laptop"), default="phone",
                    help="with --phone-camera: where the tutor's voice plays. phone (default): the phone's speaker once it is on, the "
                         "laptop until then; laptop: always the laptop")
    ap.add_argument("--mic", choices=("phone", "laptop"), default="phone",
                    help="with --phone-camera: where spoken commands are heard. phone (default): the phone's microphone while it is "
                         "streaming one, the laptop's otherwise; laptop: always the laptop's")
    ap.add_argument("--phone-host", default=None, help="this computer's address for the phone to use (default: found automatically)")
    ap.add_argument("--phone-trusted", action="store_true",
                    help="with --phone-camera: give the phone a real, trusted certificate (a *.local-ip.sh name that points at this computer), so it "
                         "shows no \"connection is not private\" warning while the video and sound stay on your Wi-Fi. Needs internet for the "
                         "first run, and the phone and laptop on the same Wi-Fi")
    ap.add_argument("--phone-domain", default=None, help="with --phone-camera: your own domain name, pointing at this computer, with --phone-cert and --phone-key")
    ap.add_argument("--phone-cert", default=None, help="the certificate (.pem) for --phone-domain")
    ap.add_argument("--phone-key", default=None, help="the key (.pem) for --phone-domain")
    ap.add_argument("--phone-tunnel", action="store_true",
                    help="with --phone-camera: reach the phone through a public https tunnel (needs the cloudflared program and internet), so the phone "
                         "shows no \"connection is not private\" warning and can be on any network")


def start_phone(a, announce: Optional[Callable] = None, voice=None, on_ready: Optional[Callable] = None) -> Optional[tuple]:
    """If --phone-camera was given: start the phone server and return (link, PhoneCamera, server); else None.
    `announce(text)` is called (in a thread) when a phone connects or drops, to speak it. With `voice` (the voice_io module) and
    --sound phone (the default) the speech is played on the phone once its sound is on. `on_ready()` (if given) replaces the generic
    "phone connected" announcements: it is called once per connection when the phone is connected and its sound is on (or 6 s have
    passed without it), which is the moment the app wants to speak to the user."""
    if not getattr(a, "phone_camera", False):
        return None
    say = announce or (lambda text: None)
    hold = "Hold the phone upright, above the page, about thirty centimetres up."
    sound_on_phone = voice is not None and getattr(a, "sound", "phone") == "phone" and not getattr(voice, "MOCK_MODE", False)

    ready_flag = [False]  # on_ready has been called for this connection

    def ready() -> None:
        if on_ready is not None and not ready_flag[0]:
            ready_flag[0] = True
            on_ready()

    def lost() -> None:
        ready_flag[0] = False
        say("The phone camera disconnected.")

    def connected() -> None:
        if on_ready is not None:
            if sound_on_phone:  # give the phone a moment to turn its sound on, so the greeting comes out of it
                deadline = time.time() + 6
                while not link.audio_ready and link.connected and time.time() < deadline:
                    time.sleep(0.1)
            return ready()
        if not sound_on_phone:
            return say(f"Phone camera connected. {hold}")
        time.sleep(6)  # the phone's tap turns its sound on a moment after the video starts: that greeting is the one to say
        if not link.audio_ready and link.connected:
            say("Phone camera connected, but its sound is off, so I am speaking here. Tap the big button on the phone to hear me there.")

    link = PhoneLink(a.phone_host, a.phone_port, on_connect=connected, on_disconnect=lost,
                     on_audio_ready=lambda: ready() if on_ready is not None else say(f"Phone connected. You will hear me on this phone now. {hold}"))
    if sound_on_phone:
        try:
            install_phone_speaker(voice, link)
        except RuntimeError as e:  # voice_io changed: keep working, on the laptop's speakers
            print(f"[phone] the tutor's voice will stay on the laptop: {e}", flush=True)
            sound_on_phone = False
    if voice is not None and getattr(a, "mic", "phone") == "phone" and not getattr(voice, "MOCK_MODE", False):
        try:
            install_phone_mic(voice, link)
        except RuntimeError as e:  # e.g. no pyaudio: speech still works, commands use the laptop microphone
            print(f"[phone] the phone microphone will not be used: {e}", flush=True)
    server = None
    if getattr(a, "phone_tunnel", False):
        link.tls = False  # the tunnel's own certificate is what the phone sees; this server is only for the tunnel, on this computer
        server = PhoneServer(link, host="127.0.0.1").start()
        try:
            server.tunnel = start_tunnel(link.port)
            link.public_base = server.tunnel.url
        except TunnelError as e:
            print(f"\n[phone] {e}\n[phone] Using the local network instead (the phone will warn the connection is not private).", flush=True)
            server.stop()
            server, link.tls = None, True
    if server is None and (getattr(a, "phone_trusted", False) or getattr(a, "phone_domain", None)):
        try:
            if getattr(a, "phone_domain", None):
                name, cert = own_domain_certificate(a.phone_domain, getattr(a, "phone_cert", None), getattr(a, "phone_key", None))
            else:
                name, cert = local_ip_certificate(link.ip)
            link.trusted_name = name
            server = PhoneServer(link, cert=cert).start()
            problem = points_here(name, link.ip)
            if problem:
                print(f"[phone] WARNING: {problem}", flush=True)
        except (TrustedCertError, RuntimeError) as e:
            link.trusted_name = None
            print(f"\n[phone] {e}\n[phone] Using the self-signed certificate instead (the phone will warn the connection is not private).", flush=True)
    if server is None:
        server = PhoneServer(link).start()
    threading.Thread(target=_hint_loop, args=(link,), daemon=True, name="phone-hints").start()
    print(f"\nPhone camera: scan this with your phone's camera, or open {link.url}\n", flush=True)
    print(qr_terminal(link.url), flush=True)
    print(f"\n(no camera to scan with? on the phone open {link.base} and enter the code {link.code})", flush=True)
    if link.tunnelled:
        print("(this address works on any network and shows no privacy warning; it is random and needs the code above)", flush=True)
    elif link.trusted_name:
        print("(this address has a real certificate, so the phone shows no privacy warning; it must be on the same Wi-Fi)", flush=True)
    return link, PhoneCamera(link), server
