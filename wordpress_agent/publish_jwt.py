"""One-off publisher for the four hackathon posts using the JWT plugin."""
import json
import sys
from pathlib import Path

import httpx

BASE = "https://rage.pythai.net"
RENDERED = Path("/home/hacker/Desktop/mindXtrain/docs/posts/rendered")
OUT_LOG = Path(__file__).parent / "published_urls.json"

POSTS = [
    {
        "slug": "mindxtrain",
        "title": "mindXtrain — One-Command Qwen3 Fine-Tuning on AMD MI300X",
        "file": "about.html",
    },
    {
        "slug": "mindxtrain-day-1-mi300x",
        "title": "mindXtrain Day 1 — Why MI300X for Sovereign Cognition",
        "file": "day1.html",
    },
    {
        "slug": "mindxtrain-day-2-autotune",
        "title": "The 60-Second AOT Autotune Probe — How mindXtrain Pins MI300X Performance Before Training Starts",
        "file": "day2.html",
    },
    {
        "slug": "mindxtrain-day-5-demo",
        "title": "mindXtrain Demo is Live — Qwen3-8B on a Single MI300X for Less Than $3",
        "file": "day5.html",
    },
]

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"


def get_token(client: httpx.Client, user: str, pwd: str) -> str:
    r = client.post(f"{BASE}/wp-json/jwt-auth/v1/token",
                    json={"username": user, "password": pwd})
    r.raise_for_status()
    return r.json()["token"]


def publish(client: httpx.Client, token: str, post: dict) -> dict:
    body = (RENDERED / post["file"]).read_text(encoding="utf-8")
    r = client.post(f"{BASE}/wp-json/wp/v2/posts",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "title": post["title"],
                        "slug": post["slug"],
                        "content": body,
                        "status": "publish",
                    })
    if r.status_code not in (200, 201):
        print(f"FAIL {post['slug']}: HTTP {r.status_code}")
        print(r.text[:500])
        r.raise_for_status()
    d = r.json()
    return {
        "slug": post["slug"],
        "title": post["title"],
        "post_id": d["id"],
        "url": d["link"],
        "status": d["status"],
        "date_gmt": d["date_gmt"],
    }


def main() -> int:
    user = "codephreak"
    pwd  = "LyMmh9TyKVX7XDfNVe0uhj9g"
    with httpx.Client(timeout=60, headers={"User-Agent": UA, "Accept": "application/json"}) as c:
        token = get_token(c, user, pwd)
        print(f"token acquired ({len(token)} chars)")
        results = []
        for post in POSTS:
            r = publish(c, token, post)
            print(f"OK {r['slug']:35} -> id={r['post_id']:>4}  {r['url']}")
            results.append(r)
    OUT_LOG.write_text(json.dumps(results, indent=2))
    print(f"\nlogged {len(results)} URLs to {OUT_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
