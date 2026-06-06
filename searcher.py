"""
Search xiaoyuzhou for podcasts or episodes.

Uses DuckDuckGo web search (site:xiaoyuzhoufm.com) — no token required.
The native app search API is iOS-only and not accessible from web clients.
"""

import re
from urllib.parse import unquote

import httpx

WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def search(keyword: str, limit: int = 10) -> list[dict]:
    """
    Search for podcasts/episodes using DuckDuckGo site:xiaoyuzhoufm.com.
    Returns a list of {type, title, url, id}.
    """
    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": f'site:xiaoyuzhoufm.com "{keyword}"', "kl": "cn-zh"},
        headers=WEB_HEADERS,
        timeout=15,
        follow_redirects=True,
    )
    resp.raise_for_status()

    results = []
    seen_urls: set[str] = set()

    # DDG HTML: <a class="result__a" href="//duckduckgo.com/l/?uddg=URL-encoded-url&rut=...">Title</a>
    for m in re.finditer(
        r'class="result__a"[^>]*href="[^"]*uddg=([^&"]+)[^"]*"[^>]*>(.*?)</a>',
        resp.text,
        re.DOTALL,
    ):
        url = unquote(m.group(1))
        title = re.sub(r'<[^>]+>', '', m.group(2)).strip()

        # Keep only xiaoyuzhoufm.com podcast/episode pages
        kind_m = re.search(r'xiaoyuzhoufm\.com/(podcast|episode)/([a-f0-9]+)', url)
        if not kind_m:
            continue
        kind, id_ = kind_m.group(1), kind_m.group(2)
        clean_url = f"https://www.xiaoyuzhoufm.com/{kind}/{id_}"
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)
        results.append({"type": kind, "url": clean_url, "id": id_, "title": title})
        if len(results) >= limit:
            break

    return results


def format_results(results: list[dict]) -> None:
    podcasts = [r for r in results if r["type"] == "podcast"]
    episodes = [r for r in results if r["type"] == "episode"]

    if podcasts:
        print("\n=== 播客 ===")
        for p in podcasts:
            print(f"  {p['title']}")
            print(f"    {p['url']}")

    if episodes:
        print("\n=== 单集 ===")
        for e in episodes:
            print(f"  {e['title']}")
            print(f"    {e['url']}")

    if not results:
        print("未找到结果（尝试换个关键词）")
