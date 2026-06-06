"""
Speech-to-text: Aliyun ASR (primary) or OpenAI Whisper (fallback).

Aliyun ASR endpoint reference:
  https://help.aliyun.com/zh/isi/developer-reference/api-nls-2019-02-28-createtoken
  File transcription: https://help.aliyun.com/zh/isi/developer-reference/recording-file-recognition
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Aliyun ASR — file transcription (录音文件识别)
# ---------------------------------------------------------------------------

NLS_META_URL = "https://nls-meta.cn-shanghai.aliyuncs.com/pop/2018-05-18/tokens"
NLS_FILETRANS_URL = "https://filetrans.cn-shanghai.aliyuncs.com/api/speech/fileTrans"

_TERMINAL_STATUS = {"SUCCESS", "SUCCESS_WITH_NO_VALID_FRAGMENT"}
_FAILED_STATUS = {"FAILED", "UNSUPPORTED_FORMAT", "EXCEED_MAX_DURATION", "INTERNAL_ERROR"}
_POLL_INTERVAL = 15  # seconds


def get_aliyun_token(access_key_id: str, access_key_secret: str) -> str:
    """Fetch a short-lived NLS access token."""
    params = {
        "AccessKeyId": access_key_id,
        "Action": "CreateToken",
        "Format": "JSON",
        "RegionId": "cn-shanghai",
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": uuid.uuid4().hex,
        "SignatureVersion": "1.0",
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "Version": "2019-02-28",
    }
    sorted_query = urllib.parse.urlencode(sorted(params.items()))
    string_to_sign = (
        "POST"
        + "&" + urllib.parse.quote("/", safe="")
        + "&" + urllib.parse.quote(sorted_query, safe="")
    )
    key = (access_key_secret + "&").encode()
    sig = base64.b64encode(
        hmac.new(key, string_to_sign.encode(), hashlib.sha1).digest()
    ).decode()
    params["Signature"] = sig

    resp = httpx.post(NLS_META_URL, data=params, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if "Token" not in body:
        raise RuntimeError(f"Aliyun token error: {body}")
    return body["Token"]["Id"]


def transcribe_aliyun(
    audio_url: str,
    appkey: str,
    access_key_id: str,
    access_key_secret: str,
) -> str:
    """Submit audio URL to Aliyun file transcription, poll until done, return text."""
    token = get_aliyun_token(access_key_id, access_key_secret)
    headers = {
        "X-NLS-Token": token,
        "Content-Type": "application/json",
    }

    # Submit
    submit_body = {
        "appkey": appkey,
        "file_link": audio_url,
        "version": "4.0",
        "enable_words": False,
    }
    resp = httpx.post(NLS_FILETRANS_URL, json=submit_body, headers=headers, timeout=30)
    resp.raise_for_status()
    submit_data = resp.json()
    status = submit_data.get("StatusText", "")
    if status not in ("RUNNING", "QUEUEING"):
        raise RuntimeError(f"Aliyun submit failed: {submit_data}")
    task_id = submit_data["TaskId"]
    print(f"[aliyun] Submitted task {task_id}, polling every {_POLL_INTERVAL}s ...")

    # Poll
    query_body = {"appkey": appkey, "task_id": task_id, "version": "4.0"}
    while True:
        time.sleep(_POLL_INTERVAL)
        resp = httpx.post(NLS_FILETRANS_URL, json=query_body, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("StatusText", "")
        print(f"[aliyun] Status: {status}")
        if status in _TERMINAL_STATUS:
            return _extract_text(result)
        if status in _FAILED_STATUS:
            raise RuntimeError(f"Aliyun transcription failed: {result}")


def _extract_text(result: dict) -> str:
    sentences = result.get("Result", {}).get("SentenceList", [])
    return "".join(s.get("Text", "") for s in sentences)


# ---------------------------------------------------------------------------
# Whisper fallback (OpenAI)
# ---------------------------------------------------------------------------

_WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24 MB (API limit is 25 MB)


def transcribe_whisper(audio_path: Path, api_key: str, model: str = "gpt-4o-mini-transcribe") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    size = audio_path.stat().st_size

    if size <= _WHISPER_MAX_BYTES:
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(model=model, file=f).text

    return _transcribe_whisper_chunked(audio_path, client, model)


def _transcribe_whisper_chunked(audio_path: Path, client, model: str) -> str:
    """Split audio into ~20-min chunks and transcribe each."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required for large file chunking: pip install pydub")

    print(f"[whisper] File is large ({audio_path.stat().st_size // 1024 // 1024} MB), splitting into chunks...")
    audio = AudioSegment.from_file(audio_path)

    # Aim for 20-minute chunks at 128 kbps ≈ 19 MB
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
