import json
import re
from pathlib import Path

import httpx
from tqdm import tqdm

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch_episode(url: str) -> dict:
    """Parse episode metadata from a xiaoyuzhoufm.com episode page."""
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()

    match = re.search(
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(\{.*?\})</script>',
        resp.text,
        re.DOTALL,
    )
    if not match:
        # Fallback: try the assignment pattern
        match = re.search(
            r'__NEXT_DATA__\s*=\s*(\{.*?\})\s*</script>',
            resp.text,
            re.DOTALL,
        )
    if not match:
        raise ValueError("Could not find __NEXT_DATA__ in page — the page structure may have changed.")

    data = json.loads(match.group(1))

    try:
        ep = data["props"]["pageProps"]["episode"]
    except KeyError:
        raise ValueError("Unexpected __NEXT_DATA__ structure — 'episode' key not found.")

    podcast = ep.get("podcast", {})
    # audio URL: ep["enclosure"]["url"] (public) or ep["media"]["source"]["url"]
    enclosure = ep.get("enclosure") or {}
    media_source = (ep.get("media") or {}).get("source") or {}
    audio_url = enclosure.get("url") or media_source.get("url") or ""

    return {
        "title": ep.get("title", "unknown"),
        "podcast_title": podcast.get("title", "unknown"),
        "audio_url": audio_url,
        "duration_sec": ep.get("duration"),  # seconds
        "hosts": _extract_hosts(ep),
        "description": ep.get("description", ""),
        "pub_date": ep.get("pubDate", ""),
    }


def _extract_hosts(ep: dict) -> str:
    podcast = ep.get("podcast", {})
    hosts = []
    for author in podcast.get("author", "").split(","):
        name = author.strip()
        if name:
            hosts.append(name)
    if not hosts:
        hosts = [podcast.get("title", "")]
    return "、".join(hosts)


def download_audio(audio_url: str, dest: Path) -> Path:
    """Download audio file with a progress bar. Returns the saved path."""
    if dest.exists():
        return dest

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        with client.stream("GET", audio_url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {dest.name}",
            ) as bar:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def format_duration(sec: int | None) -> str:
    if not sec:
        return "未知"
    h, remainder = divmod(int(sec), 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
