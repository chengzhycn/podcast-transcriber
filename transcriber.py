"""
Speech-to-text backends:
  - self-hosted: faster-whisper-server via SSH (server downloads audio from CDN, no local upload)
  - openai: OpenAI Whisper API (cloud fallback)
"""

import json
import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Self-hosted faster-whisper-server — SSH remote execution (primary)
#
# Flow: local → SSH → server downloads audio from CDN → server calls localhost ASR → text back
# Avoids slow local-to-server upload; xyzcdn.net CDN is fast from China servers.
# ---------------------------------------------------------------------------

def transcribe_self_hosted(
    audio_url: str,
    ssh_host: str,
    asr_port: int = 8000,
    model: str = "Systran/faster-whisper-tiny",
    language: str = "zh",
    ssh_key: str | None = None,
    ssh_user: str = "root",
) -> str:
    """
    SSH into the ASR server, download audio from CDN there, transcribe via localhost API.
    Returns plain transcript text.
    """
    ssh_target = f"{ssh_user}@{ssh_host}"
    remote_path = "/tmp/podcast_asr_audio.m4a"

    remote_cmd = (
        f'curl -sL -o {remote_path} '
        f'-H "User-Agent: Mozilla/5.0" '
        f'"{audio_url}" && '
        f'curl -s -X POST http://localhost:{asr_port}/v1/audio/transcriptions '
        f'-F "file=@{remote_path}" '
        f'-F "model={model}" '
        f'-F "language={language}"'
    )

    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    if ssh_key:
        ssh_cmd += ["-i", os.path.expanduser(ssh_key)]
    ssh_cmd += [ssh_target, remote_cmd]

    print(f"[asr] Connecting to {ssh_target}, downloading audio and transcribing ...")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        raise RuntimeError(
            f"SSH transcription failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:1000]}\n"
            f"stderr: {result.stderr[:1000]}"
        )

    # Response is JSON: {"text": "..."}
    try:
        return json.loads(result.stdout)["text"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Unexpected ASR response: {result.stdout[:500]}") from e


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
