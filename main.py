"""
小宇宙播客 → 语音转文字 → Blog 文章

Usage:
    python main.py "https://www.xiaoyuzhoufm.com/episode/xxxx"
    python main.py <url> --transcript transcript.txt   # skip STT
    python main.py <url> --stt whisper                 # use Whisper instead of Aliyun
    python main.py <url> --audio-only                  # only download audio
    python main.py <url> --llm-model gpt-4o-mini       # cheaper LLM
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

import config
from fetcher import download_audio, fetch_episode, format_duration
from organizer import organize
from searcher import format_results, search  # no token needed — uses DuckDuckGo
from transcriber import transcribe_local, transcribe_self_hosted, transcribe_openai

app = typer.Typer(add_completion=False)


def _safe_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s)[:60]


@app.command()
def run(
    url: str = typer.Argument(..., help="小宇宙单集页面 URL"),
    stt: str = typer.Option(config.get("STT_PROVIDER", "local"), help="STT provider: local | self-hosted | openai"),
    transcript: Optional[Path] = typer.Option(None, help="已有转录稿文件，跳过 STT"),
    audio_only: bool = typer.Option(False, help="仅下载音频，不转录"),
    llm_model: str = typer.Option(config.get("LLM_MODEL", "gpt-4o"), help="OpenAI 模型（整理用）"),
    output: Optional[Path] = typer.Option(None, help="输出 Markdown 文件路径"),
) -> None:
    # 1. Fetch episode metadata
    typer.echo("Fetching episode metadata ...")
    meta = fetch_episode(url)
    typer.echo(
        f"  [{meta['podcast_title']}] {meta['title']} "
        f"({format_duration(meta['duration_sec'])})"
    )

    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"{_safe_filename(meta['podcast_title'])}_{_safe_filename(meta['title'])}_{date_str}"

    # 2. Download audio
    audio_dir = config.OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(meta["audio_url"].split("?")[0]).suffix or ".mp3"
    audio_path = audio_dir / f"{stem}{suffix}"

    if not meta["audio_url"]:
        typer.echo("ERROR: Could not find audio URL in episode page.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Audio URL: {meta['audio_url']}")
    download_audio(meta["audio_url"], audio_path)
    typer.echo(f"Saved audio: {audio_path}")

    if audio_only:
        typer.echo("Done (audio only).")
        return

    # 3. Transcribe
    if transcript:
        typer.echo(f"Loading transcript from {transcript} ...")
        transcript_text = transcript.read_text(encoding="utf-8")
    elif stt == "local":
        local_url = config.get("ASR_LOCAL_URL", "http://localhost:18902")
        typer.echo(f"Transcribing with local FunASR ({local_url}) ...")
        transcript_text = transcribe_local(audio_path, base_url=local_url)
    elif stt == "self-hosted":
        ssh_host = config.require("ASR_SSH_HOST")
        asr_port = int(config.get("ASR_PORT", "8000"))
        asr_model = config.get("ASR_MODEL", "Systran/faster-whisper-tiny")
        ssh_key = config.get("ASR_SSH_KEY")
        ssh_user = config.get("ASR_SSH_USER", "root")
        transcript_text = transcribe_self_hosted(
            meta["audio_url"],
            ssh_host=ssh_host,
            asr_port=asr_port,
            model=asr_model,
            ssh_key=ssh_key,
            ssh_user=ssh_user,
        )
    elif stt == "openai":
        typer.echo("Transcribing with OpenAI Whisper ...")
        openai_key = config.require("OPENAI_API_KEY")
        transcript_text = transcribe_openai(audio_path, openai_key)
    else:
        typer.echo(f"Unknown STT provider: {stt} (valid: local, self-hosted, openai)", err=True)
        raise typer.Exit(1)

    # Save raw transcript alongside the blog
    transcript_path = config.OUTPUT_DIR / f"{stem}_transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    typer.echo(f"Saved transcript: {transcript_path}")

    # 4. Organize into blog post
    openai_key = config.require("OPENAI_API_KEY")
    openai_base_url = config.get("OPENAI_BASE_URL")
    blog_md = organize(transcript_text, meta, api_key=openai_key, model=llm_model, base_url=openai_base_url)

    # 5. Save output
    out_path = output or (config.OUTPUT_DIR / f"{stem}_blog.md")
    out_path.write_text(blog_md, encoding="utf-8")
    typer.echo(f"\nBlog saved: {out_path}")


@app.command()
def search_cmd(
    keyword: str = typer.Argument(..., help="搜索关键词（播客名、主播名、话题等）"),
    limit: int = typer.Option(10, help="最多返回结果数"),
) -> None:
    """搜索小宇宙播客和单集，输出可直接使用的 URL（通过 DuckDuckGo，无需登录）。"""
    typer.echo(f'搜索 "{keyword}" ...')
    results = search(keyword, limit=limit)
    format_results(results)


if __name__ == "__main__":
    app()
