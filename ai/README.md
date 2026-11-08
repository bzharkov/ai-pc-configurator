# AI-модуль

Разбор текстового запроса пользователя в структурированные требования
(`budget`, `tasks`, `resolution`, `preferences`, `missing`).

## Установка
```bash
cd ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # вписать OPENROUTER_API_KEY
```

## Использование из кода (бэкенд)
```python
from request_parser import parse_request, ParseError   # путь: ai/userPrompt
req = parse_request("ПК для игр в 2К до 120к")          # -> Requirements
req.missing   # поля, которые надо уточнить у пользователя (["budget"], ["tasks"])
```

## Ручная проверка
```bash
python ai/userPrompt/promptai.py "ПК для игр в 2К и монтажа до 120 000 ₽"
```
Результат печатается и сохраняется в `ai/userPrompt/AIRequest.json`.

## Тесты (без сети)
```bash
cd ai && python -m pytest tests
```

## Формат результата
```json
{"budget": 120000, "tasks": ["Games", "Montage"], "resolution": "1440p",
 "preferences": "красивый корпус", "missing": []}
```
`tasks`: Games, Montage, Office, 3D, Study. `resolution`: 1080p, 1440p, 4K или null.
