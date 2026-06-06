"""
Speech-to-text backends:
  - self-hosted: faster-whisper-server (OpenAI-compatible, default)
  - openai: OpenAI Whisper API (cloud fallback)
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Self-hosted faster-whisper-server (primary)
# ---------------------------------------------------------------------------

def transcribe_self_hosted(
    audio_path: Path,
    base_url: str,
    model: str = "Systran/faster-whisper-tiny",
    language: str = "zh",
) -> str:
    """POST audio file to a self-hosted faster-whisper-server instance."""
    from openai import OpenAI

    # OpenAI client works against any compatible endpoint
    client = OpenAI(api_key="not-needed", base_url=base_url)
    print(f"[asr] Sending {audio_path.name} ({audio_path.stat().st_size // 1024 // 1024} MB) to {base_url} ...")
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
        )
    return result.text


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
