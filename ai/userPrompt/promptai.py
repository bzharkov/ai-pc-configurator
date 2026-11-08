"""CLI для ручной проверки: python ai/userPrompt/promptai.py "ПК для игр в 2К до 120к"."""
import sys
from pathlib import Path

from request_parser import ParseError, parse_request

OUTPUT = Path(__file__).with_name("AIRequest.json")


def main() -> int:
    query = " ".join(sys.argv[1:]) or input("Опишите, какой ПК нужен: ")
    try:
        req = parse_request(query)
    except ParseError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    OUTPUT.write_text(req.model_dump_json(indent=4), encoding="utf-8")
    print(req.model_dump_json(indent=4))
    if req.missing:
        print(f"Нужно уточнить: {', '.join(req.missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
