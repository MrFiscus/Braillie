#!/usr/bin/env python3
"""
audio_device_check.py — Microphone diagnostic for voice_io.py

Usage
-----
    python3 audio_device_check.py                  # device table + voice_io info
    python3 audio_device_check.py --list-only       # device table only, no recording
    python3 audio_device_check.py <index>           # device table + 5-sec level meter
    python3 audio_device_check.py <index> --list-only  # --list-only always wins
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time

# ---------------------------------------------------------------------------
# Constants pulled from voice_io.py so this script stays in sync.
# ---------------------------------------------------------------------------
VOICE_IO_FORMAT_WIDTH = 2        # paInt16 → 2 bytes per sample
VOICE_IO_SAMPLE_RATE  = 16_000
VOICE_IO_CHANNELS     = 1
VOICE_IO_CHUNK_MS     = 100

METER_WIDTH    = 50              # character width of the level bar
METER_DURATION = 5.0             # seconds to record when metering
METER_SCALE    = 6.0             # amplify RMS so normal speech fills the bar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_pyaudio():
    try:
        import pyaudio  # type: ignore[import-untyped]
        return pyaudio
    except ImportError:
        sys.exit(
            "ERROR: pyaudio is not installed.\n"
            "Install it with:  pip install pyaudio\n"
            "(On macOS you may also need: brew install portaudio)"
        )


def _rms(data: bytes) -> float:
    """Root-mean-square amplitude of a block of int16 PCM bytes (0.0 – 1.0)."""
    count = len(data) // VOICE_IO_FORMAT_WIDTH
    if count == 0:
        return 0.0
    samples = struct.unpack(f"{count}h", data)
    mean_sq = sum(s * s for s in samples) / count
    return math.sqrt(mean_sq) / 32768.0


def _bar(level: float) -> str:
    """Return a fixed-width bar string for *level* in [0.0, 1.0]."""
    filled = min(int(level * METER_WIDTH), METER_WIDTH)
    empty  = METER_WIDTH - filled
    return "#" * filled + "-" * empty


def _device_rows(pa) -> list[dict]:
    """Return a list of dicts describing every input-capable device."""
    rows = []
    count = pa.get_device_count()
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] < 1:
            continue
        rows.append({
            "index":      i,
            "name":       info["name"],
            "rate":       int(info["defaultSampleRate"]),
            "host_api":   pa.get_host_api_info_by_index(info["hostApi"])["name"],
        })
    return rows


def _default_input_index(pa) -> int:
    try:
        return pa.get_default_input_device_info()["index"]
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Feature: device table
# ---------------------------------------------------------------------------

def print_device_table(pa) -> None:
    rows        = _device_rows(pa)
    default_idx = _default_input_index(pa)

    if not rows:
        print("No input devices found.")
        return

    # Column widths
    name_w = max(len(r["name"]) for r in rows)
    name_w = max(name_w, len("Name"))
    api_w  = max(len(r["host_api"]) for r in rows)
    api_w  = max(api_w, len("Host API"))

    header = (
        f"{'Idx':>3}  {'Default':7}  "
        f"{'Name':<{name_w}}  {'Rate':>8}  {'Host API':<{api_w}}"
    )
    sep = "-" * len(header)

    print("\nAUDIO INPUT DEVICES")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        marker = "  <---  " if r["index"] == default_idx else "         "
        print(
            f"{r['index']:>3}  {marker}"
            f"{r['name']:<{name_w}}  {r['rate']:>8}  {r['host_api']:<{api_w}}"
        )
    print(sep)

    # voice_io.py explanation
    print("\nVOICE_IO.PY DEVICE SELECTION")
    print(
        "  voice_io.py calls pa.open(input=True) with no input_device_index,\n"
        "  so PyAudio opens whichever device is the system default at that moment."
    )
    if default_idx >= 0:
        default_info = pa.get_device_info_by_index(default_idx)
        print(
            f"  → Device #{default_idx}: \"{default_info['name']}\"  "
            f"({int(default_info['defaultSampleRate'])} Hz)\n"
            f"     voice_io.py will request {VOICE_IO_SAMPLE_RATE} Hz from this device."
        )
    else:
        print("  → Could not determine the system default input device.")
    print()


# ---------------------------------------------------------------------------
# Feature: level meter
# ---------------------------------------------------------------------------

def run_level_meter(pa, device_index: int) -> None:
    # Validate the index before opening.
    try:
        info = pa.get_device_info_by_index(device_index)
    except Exception:
        sys.exit(f"ERROR: Device index {device_index} does not exist.")

    if info["maxInputChannels"] < 1:
        sys.exit(
            f"ERROR: Device #{device_index} (\"{info['name']}\") "
            "has no input channels."
        )

    print(
        f"LEVEL METER — Device #{device_index}: \"{info['name']}\"\n"
        f"  Sample rate : {VOICE_IO_SAMPLE_RATE} Hz  "
        f"(voice_io.py setting)\n"
        f"  Channels    : {VOICE_IO_CHANNELS}\n"
        f"  Duration    : {METER_DURATION:.0f} seconds\n"
        f"  Bar width   : {METER_WIDTH} chars  (each '#' ≈ {100/METER_WIDTH:.1f}% of scale)\n"
        "\nPress Ctrl-C to stop early.\n"
    )

    chunk_frames = int(VOICE_IO_SAMPLE_RATE * VOICE_IO_CHUNK_MS / 1000)

    try:
        stream = pa.open(
            format=pa.get_format_from_width(VOICE_IO_FORMAT_WIDTH),
            channels=VOICE_IO_CHANNELS,
            rate=VOICE_IO_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_frames,
        )
    except OSError as exc:
        sys.exit(
            f"ERROR: Could not open device #{device_index} "
            f"(\"{info['name']}\"):\n  {exc}\n"
            "Common causes: device is busy, permissions denied, or the requested\n"
            f"sample rate ({VOICE_IO_SAMPLE_RATE} Hz) is not supported by this device."
        )

    start = time.monotonic()
    peak  = 0.0

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= METER_DURATION:
                break

            try:
                data = stream.read(chunk_frames, exception_on_overflow=False)
            except OSError as exc:
                print(f"\nERROR: Microphone read failed: {exc}")
                break

            rms     = _rms(data)
            scaled  = min(rms * METER_SCALE, 1.0)
            peak    = max(peak, scaled)
            bar     = _bar(scaled)
            remaining = max(0.0, METER_DURATION - elapsed)

            # Overwrite the same terminal line.
            print(
                f"\r  {remaining:4.1f}s  [{bar}]  rms={rms:.4f}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()

    print()  # newline after the meter line

    # Summary
    if peak == 0.0:
        verdict = "No signal detected. Check the device and try speaking."
    elif peak < 0.05:
        verdict = "Very faint signal. Device may be too far away or muted."
    elif peak < 0.25:
        verdict = "Quiet but present. May struggle in a noisy environment."
    elif peak < 0.75:
        verdict = "Good signal level for voice_io.py."
    else:
        verdict = "Strong signal — you may want to reduce input gain slightly."

    print(f"  Peak (scaled): {peak:.0%}  →  {verdict}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose which microphone voice_io.py is capturing from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 audio_device_check.py --list-only\n"
            "  python3 audio_device_check.py 1\n"
        ),
    )
    parser.add_argument(
        "device_index",
        nargs="?",
        type=int,
        default=None,
        metavar="INDEX",
        help="Device index to run the level meter against (optional).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the device table and exit without recording.",
    )
    args = parser.parse_args()

    pyaudio = _require_pyaudio()
    pa = pyaudio.PyAudio()

    try:
        print_device_table(pa)

        if args.list_only:
            return

        if args.device_index is None:
            print(
                "Tip: pass a device index to run the level meter, e.g.:\n"
                f"  python3 {sys.argv[0]} <index>\n"
                f"  python3 {sys.argv[0]} --list-only\n"
            )
            return

        run_level_meter(pa, args.device_index)

    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
