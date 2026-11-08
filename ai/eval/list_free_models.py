"""Список бесплатных моделей OpenRouter с поддержкой структурированного вывода.

    python ai/eval/list_free_models.py
Выберите несколько и пропишите через запятую в OPENROUTER_MODELS.
"""
import json
import urllib.request


def free_models(models: list[dict]) -> list[dict]:
    out = []
    for m in models:
        p = m.get("pricing") or {}
        if str(p.get("prompt")) == "0" and str(p.get("completion")) == "0":
            params = m.get("supported_parameters") or []
            out.append({
                "id": m["id"],
                "context": m.get("context_length"),
                "json_schema": "structured_outputs" in params,
                "json_object": "response_format" in params,
            })
    return sorted(out, key=lambda x: (not x["json_schema"], not x["json_object"], x["id"]))


def main():
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        data = json.load(r)["data"]
    for m in free_models(data):
        print(f'{m["id"]:60} ctx={m["context"]} schema={m["json_schema"]} json={m["json_object"]}')


if __name__ == "__main__":
    main()
