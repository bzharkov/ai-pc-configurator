"""Диагностика ключа OpenRouter: лимиты аккаунта и полный текст ошибки 429.

    python ai/eval/check_key.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

BASE = "https://openrouter.ai/api/v1"


def call(path, payload=None):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY не задан")
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(BASE + path, data=data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def main():
    status, _, body = call("/key")
    print(f"== /key -> HTTP {status}")
    try:
        d = json.loads(body).get("data", {})
        for k in ("label", "is_free_tier", "usage", "limit", "limit_remaining", "rate_limit"):
            if k in d:
                print(f"  {k}: {d[k]}")
    except ValueError:
        print(body[:500])

    model = sys.argv[1] if len(sys.argv) > 1 else "openrouter/free"
    status, headers, body = call("/chat/completions", {
        "model": model, "max_tokens": 5,
        "messages": [{"role": "user", "content": "ping"}]})
    print(f"\n== тестовый запрос к {model} -> HTTP {status}")
    for h, v in headers.items():
        if h.lower().startswith(("x-ratelimit", "retry-after")):
            print(f"  {h}: {v}")
    print(body[:1500])


if __name__ == "__main__":
    main()
