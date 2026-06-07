FROM python:3.11-slim

# ── apt: Tsinghua mirror (has Debian 13 trixie) ───────────────────────────────
RUN sed -i \
    -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
    -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
    /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── pip: Aliyun mirror ────────────────────────────────────────────────────────
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# torch on aarch64 pulls nvidia-cublas (~540MB, Jetson build),
# but CPU inference still works fine — CUDA libs are simply unused.
RUN pip install --no-cache-dir torch torchaudio

RUN pip install --no-cache-dir \
    funasr \
    modelscope \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY server.py .

# Models downloaded on first run, persisted via Docker volume
ENV MODELSCOPE_CACHE=/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
