"""Оценка качества разбора на эталонных запросах.

Запуск (нужен OPENROUTER_API_KEY):
    python ai/eval/run_eval.py                       # модели из окружения
    python ai/eval/run_eval.py --models m1,m2 --delay 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "userPrompt"))
from request_parser import ParseError, get_client, parse_request  # noqa: E402

CASES = Path(__file__).with_name("cases.json")


def score(case: dict, req) -> dict:
    b = req.budget == case["budget"]
    t = set(req.tasks) == set(case["tasks"])
    r = req.resolution == case["resolution"]
    return {"budget": b, "tasks": t, "resolution": r, "all": b and t and r}


def evaluate(models, delay=0.0, limit=None, client=None, verbose=True):
    cases = json.loads(CASES.read_text(encoding="utf-8"))[:limit]
    totals = {"budget": 0, "tasks": 0, "resolution": 0, "all": 0, "errors": 0}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        try:
            req = parse_request(case["q"], client=client, model=models)
            s = score(case, req)
            for k, v in s.items():
                totals[k] += v
            if verbose and not s["all"]:
                print(f"[{i}] MISMATCH {case['q']!r}\n    ожидалось: b={case['budget']} t={case['tasks']} "
                      f"r={case['resolution']}\n    получено:  b={req.budget} t={req.tasks} r={req.resolution}")
        except ParseError as e:
            totals["errors"] += 1
            if verbose:
                print(f"[{i}] ERROR {case['q']!r}: {e}")
        if delay:
            time.sleep(delay)
    n = len(cases)
    return {"n": n, "seconds": round(time.time() - t0, 1),
            **{k: f"{v}/{n}" for k, v in totals.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="модели через запятую (по умолчанию из окружения)")
    ap.add_argument("--delay", type=float, default=2.0, help="пауза между запросами (лимиты free-моделей)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    try:
        client = get_client()
    except ParseError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(evaluate(models, a.delay, a.limit, client), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
