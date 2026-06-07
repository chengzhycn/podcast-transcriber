"""
Speech-to-text backends:
  - local: local FunASR paraformer-zh+cam++ via Docker (best Chinese accuracy, speaker labels)
  - openai: OpenAI Whisper API (cloud fallback)
"""

from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# Local FunASR (Docker on Mac) — paraformer-zh + cam++ speaker diarization
#
# Outputs speaker-labeled transcript:
#   [SPK0] 今天我们来聊一个话题...
#   [SPK1] 对，这个问题确实很重要...
# Speaker labels help the LLM organize the blog into dialogue structure.
# ---------------------------------------------------------------------------

def transcribe_local(
    audio_path: Path,
    base_url: str = "http://localhost:18902",
) -> str:
    """POST local audio file to the local FunASR Docker server."""
    print(f"[funasr] Sending {audio_path.name} ({audio_path.stat().st_size // 1024 // 1024} MB) to local server ...")
    with open(audio_path, "rb") as f:
        resp = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, f, "application/octet-stream")},
            data={"model": "paraformer-zh"},
            timeout=1800,
        )
    resp.raise_for_status()
    return resp.json()["text"]


# ---------------------------------------------------------------------------
# OpenAI Whisper API (cloud fallback)
# ---------------------------------------------------------------------------

_WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24 MB (API limit is 25 MB)


def transcribe_openai(
    audio_path: Path,
    api_key: str,
    model: str = "gpt-4o-mini-transcribe",
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    size = audio_path.stat().st_size

    if size <= _WHISPER_MAX_BYTES:
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(model=model, file=f).text

    return _transcribe_openai_chunked(audio_path, client, model)


def _transcribe_openai_chunked(audio_path: Path, client, model: str) -> str:
    """Split audio into ~20-min chunks and transcribe each."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required for large file chunking: pip install pydub")

    print(f"[whisper] File is large ({audio_path.stat().st_size // 1024 // 1024} MB), splitting into chunks ...")
    audio = AudioSegment.from_file(audio_path)

    chunk_ms = 20 * 60 * 1000
    chunks_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    chunks_dir.mkdir(exist_ok=True)

    parts = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = chunks_dir / f"chunk_{i:03d}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="128k")
        print(f"[whisper] Transcribing chunk {i + 1} ...")
        with open(chunk_path, "rb") as f:
            text = client.audio.transcriptions.create(model=model, file=f).text
        parts.append(text)

    return " ".join(parts)
