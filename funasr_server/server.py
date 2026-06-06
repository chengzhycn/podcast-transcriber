"""
OpenAI-compatible transcription server backed by FunASR paraformer-zh + cam++ (speaker diarization).

Endpoint: POST /v1/audio/transcriptions
  - file: audio file (multipart)
  - model: ignored (always uses paraformer-zh)

Response: {"text": "[SPK0] ...\n[SPK1] ...", "segments": [...]}
"""

import logging
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from funasr import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="FunASR Transcription Server")

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", "/root/.cache/modelscope")

log.info("Loading FunASR models (first run downloads ~640MB, subsequent runs use cache)...")
_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    disable_update=True,
)
log.info("Models loaded.")


@app.get("/health")
def health():
    return {"status": "ok", "model": "paraformer-zh+cam++"}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form("zh"),
):
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    audio_bytes = await file.read()
    log.info("Received %.1f MB audio, transcribing...", len(audio_bytes) / 1024 / 1024)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = _model.generate(
            input=tmp_path,
            batch_size_s=300,   # process up to 300s of audio per batch
            merge_vad=True,
            merge_length_s=15,  # merge short VAD segments into ≤15s chunks
        )
    finally:
        os.unlink(tmp_path)

    if not result or not result[0]:
        return JSONResponse({"text": "", "segments": []})

    segments = result[0]
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        spk = seg.get("spk", "")
        lines.append(f"[{spk}] {text}" if spk else text)

    full_text = "\n".join(lines)
    log.info("Done. %d segments, %d chars", len(segments), len(full_text))
    return JSONResponse({"text": full_text, "segments": segments})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
