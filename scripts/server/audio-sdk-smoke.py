"""Opt-in OpenAI Python SDK interoperability, invoked by audio-sdk-smoke.ts."""
import hashlib
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

import openai
from openai import OpenAI, Stream

client = OpenAI(base_url=os.environ["MLX_AUDIO_SDK_BASE_URL"], api_key=os.environ["MLX_AUDIO_SDK_TOKEN"], max_retries=0, timeout=60)
fixture, output = map(Path, sys.argv[1:3])
asr, tts = os.environ["ASR_MODEL"], os.environ["TTS_MODEL"]
voice = os.environ["TTS_VOICE"]
speech = dict(model=tts, voice=voice, input="Hello. This is a local speech service.")
sampling = dict(seed=42, temperature=0, max_duration_seconds=12)
baseline = Path(os.environ["MLX_AUDIO_SDK_PCM"]).read_bytes()
results = []


def check(name, action):
    started = time.perf_counter()
    try:
        detail = action()
        results.append(dict(name=name, status="passed", ms=(time.perf_counter()-started)*1000, detail=detail))
        print("PASS", name, flush=True)
    except Exception as error:
        results.append(dict(name=name, status="failed", ms=(time.perf_counter()-started)*1000, detail=str(error)))
        print("FAIL", name, repr(error), flush=True)


def models():
    ids = [m.id for m in client.models.list()]
    assert asr in ids and tts in ids
    return dict(models=ids)


def transcription(response_format="json"):
    with fixture.open("rb") as audio:
        result = client.audio.transcriptions.create(model=asr, file=audio, response_format=response_format)
    if response_format == "text":
        assert result.strip()
        return dict(text=result)
    assert result.text.strip()
    assert result.finish_reason == "stop"
    return result.model_dump()


def revisions():
    events = []
    with fixture.open("rb") as audio:
        stream = client.audio.transcriptions.create(model=asr, file=audio, stream=True, extra_body={
            "sampling": json.dumps({"max_tokens": 128}), "stream_options": json.dumps({"chunk_seconds": 1}),
        })
        for event in stream:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            assert data["type"] == "transcript"
            assert data["revision"] > (events[-1]["revision"] if events else -1)
            assert isinstance(data["stable"], str) and isinstance(data["provisional"], str)
            events.append(data)
    assert len(events) > 1 and events[-1]["is_final"] and events[-1]["provisional"] == ""
    return dict(events=len(events), final=events[-1], semantics="replace whole snapshot")


def wav_audio():
    response = client.audio.speech.with_raw_response.create(**speech, response_format="wav", extra_body={"sampling": sampling})
    content = response.parse().content
    with wave.open(io.BytesIO(content)) as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        pcm = wav.readframes(wav.getnframes())
    assert pcm == baseline
    assert rate == int(response.headers["x-sample-rate"])
    assert channels == int(response.headers["x-channels"])
    assert len(pcm)/(2*rate*channels) == float(response.headers["x-audio-seconds"])
    return dict(bytes=len(content), exactPcm=True, sampleRate=rate, channels=channels)


def pcm_audio():
    with client.audio.speech.with_streaming_response.create(**speech, response_format="pcm", extra_body={"sampling": sampling}) as response:
        chunks = list(response.iter_bytes(chunk_size=4096))
        pcm = b"".join(chunks)
        assert pcm == baseline
        return dict(bytes=len(pcm), chunks=len(chunks), exactPcm=True, sha256=hashlib.sha256(pcm).hexdigest(), trailerDeclaration=response.headers["trailer"])


def speech_events():
    # The generated speech method returns binary content even for SSE. Use
    # the SDK's generic streaming transport with our declared extension type.
    stream = client.post("/audio/speech", cast_to=dict, body={**speech, "response_format": "pcm", "stream_format": "sse", "sampling": sampling}, stream=True, stream_cls=Stream[dict])
    pieces, done = [], None
    import base64
    for event in stream:
        if event["type"] == "speech.audio.delta":
            assert done is None
            pieces.append(base64.b64decode(event["audio"], validate=True))
        else:
            assert event["type"] == "speech.audio.done" and done is None
            done = event
    assert done and b"".join(pieces) == baseline
    return dict(chunks=len(pieces), exactPcm=True, done=done)


def rejected_format():
    try:
        client.audio.speech.create(**speech, response_format="mp3")
    except openai.BadRequestError as error:
        assert error.code == "unsupported_response_format" and error.param == "response_format"
        return dict(status=error.status_code, code=error.code, param=error.param)
    raise AssertionError("Expected BadRequestError")


try:
    for name, action in [
        ("python/models", models), ("python/asr-json", transcription),
        ("python/asr-text", lambda: transcription("text")), ("python/asr-revision-sse", revisions),
        ("python/tts-wav", wav_audio), ("python/tts-pcm-stream", pcm_audio),
        ("python/tts-sse-sdk-decoder", speech_events), ("python/unsupported-format-error", rejected_format),
    ]:
        check(name, action)
finally:
    client.close()
    output.write_text(json.dumps(dict(python=sys.version.split()[0], sdk=openai.__version__, results=results), indent=2)+"\n")
sys.exit(1 if any(r["status"] == "failed" for r in results) else 0)
