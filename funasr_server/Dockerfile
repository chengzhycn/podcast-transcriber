FROM python:3.11-slim

# ffmpeg: audio format conversion (m4a/mp3/ogg → wav for FunASR)
# libsndfile1: soundfile dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (prevents pulling the huge CUDA variant)
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# FunASR + API server deps
RUN pip install --no-cache-dir \
    funasr \
    modelscope \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY server.py .

# Models are downloaded to this path on first run and cached via volume mount
ENV MODELSCOPE_CACHE=/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
