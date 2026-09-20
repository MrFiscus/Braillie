"""Offline event-model test for voice_io's Deepgram 7.9 streaming loop.

Run with: python3 test_deepgram_stt_mock.py
"""
from __future__ import annotations

import sys
import types

import voice_io


class _FakeEventType:
    OPEN = "open"
    MESSAGE = "message"
    ERROR = "error"


class _FakeResults:
    def __init__(self, transcript: str, confidence: float, is_final: bool) -> None:
        word = types.SimpleNamespace(confidence=confidence)
        alternative = types.SimpleNamespace(transcript=transcript, words=[word])
        self.channel = types.SimpleNamespace(alternatives=[alternative])
        self.is_final = is_final


class _FakeConnection:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self.handlers: dict[str, object] = {}

    def on(self, event: str, handler: object) -> None:
        self.handlers[event] = handler
        self.state.setdefault("handler_events", []).append(event)

    def start_listening(self) -> None:
        self.state["start_listening"] = True
        self.handlers[_FakeEventType.OPEN](None)  # type: ignore[index, operator]
        self.handlers[_FakeEventType.MESSAGE](object())  # type: ignore[index, operator]
        self.handlers[_FakeEventType.MESSAGE](  # type: ignore[index, operator]
            _FakeResults("hint", 0.99, False)
        )
        self.handlers[_FakeEventType.MESSAGE](  # type: ignore[index, operator]
            _FakeResults("Hint!", 0.99, True)
        )
        STATE["reader_release"].wait(timeout=1)  # type: ignore[index, union-attr]

    def send_media(self, data: bytes) -> None:
        self.state.setdefault("media", []).append(data)

    def send_close_stream(self) -> None:
        self.state["closed"] = True
        self.state["reader_release"].set()  # type: ignore[index, union-attr]


class _FakeConnect:
    def __init__(self, state: dict[str, object]) -> None:
        self.connection = _FakeConnection(state)

    def __enter__(self) -> _FakeConnection:
        return self.connection

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeClient:
    def __init__(self, *, api_key: str) -> None:
        STATE["api_key"] = api_key
        self.listen = types.SimpleNamespace(
            v1=types.SimpleNamespace(connect=self._connect)
        )

    def _connect(self, **options: object) -> _FakeConnect:
        STATE["options"] = options
        return _FakeConnect(STATE)


class _FakeStream:
    def read(self, _frames: int, **_kwargs: object) -> bytes:
        voice_io._stop_event.set()
        return b"mic-chunk"

    def stop_stream(self) -> None:
        STATE["stream_stopped"] = True

    def close(self) -> None:
        STATE["stream_closed"] = True


class _FakePyAudioInstance:
    def open(self, **kwargs: object) -> _FakeStream:
        STATE["mic_options"] = kwargs
        assert STATE.get("start_listening") is True
        return _FakeStream()

    def get_default_input_device_info(self) -> dict[str, int]:
        return {"index": 0}

    def terminate(self) -> None:
        STATE["terminated"] = True


STATE: dict[str, object] = {}


def main() -> None:
    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "deepgram",
            "deepgram.core",
            "deepgram.listen",
            "deepgram.listen.v1",
            "deepgram.listen.v1.types",
            "deepgram.listen.v1.types.listen_v1results",
            "pyaudio",
        )
    }
    original_key = voice_io.DEEPGRAM_API_KEY
    original_stop_event = voice_io._stop_event
    original_callbacks = voice_io._callbacks
    try:
        STATE.clear()
        STATE["reader_release"] = voice_io.threading.Event()
        deepgram = types.ModuleType("deepgram")
        deepgram.DeepgramClient = _FakeClient  # type: ignore[attr-defined]
        core = types.ModuleType("deepgram.core")
        core.EventType = _FakeEventType  # type: ignore[attr-defined]
        results = types.ModuleType("deepgram.listen.v1.types.listen_v1results")
        results.ListenV1Results = _FakeResults  # type: ignore[attr-defined]
        sys.modules.update({
            "deepgram": deepgram,
            "deepgram.core": core,
            "deepgram.listen": types.ModuleType("deepgram.listen"),
            "deepgram.listen.v1": types.ModuleType("deepgram.listen.v1"),
            "deepgram.listen.v1.types": types.ModuleType("deepgram.listen.v1.types"),
            "deepgram.listen.v1.types.listen_v1results": results,
            "pyaudio": types.SimpleNamespace(
                PyAudio=_FakePyAudioInstance, paInt16=8
            ),
        })
        voice_io.DEEPGRAM_API_KEY = "test-key"
        voice_io._stop_event = voice_io.threading.Event()
        fired: list[str] = []
        voice_io._callbacks = {"hint": [lambda: fired.append("hint")]}

        voice_io._deepgram_listener_loop()

        assert STATE["api_key"] == "test-key"
        assert STATE["options"]["interim_results"] is False  # type: ignore[index]
        assert STATE["handler_events"] == ["open", "message", "error"]
        assert STATE["start_listening"] is True
        assert STATE["media"] == [b"mic-chunk"]
        assert STATE["closed"] is True
        assert STATE["stream_stopped"] is True
        assert STATE["stream_closed"] is True
        assert STATE["terminated"] is True
        assert fired == ["hint"], "only the final transcript may dispatch"
        print("Passed offline Deepgram event startup, transcript, and shutdown checks.")
    finally:
        voice_io.DEEPGRAM_API_KEY = original_key
        voice_io._stop_event = original_stop_event
        voice_io._callbacks = original_callbacks
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    main()
