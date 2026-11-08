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

## Устойчивость к ошибкам бесплатных моделей
- `OPENROUTER_MODELS` — список моделей по приоритету; при сбое берётся следующая.
- Режимы ответа пробуются по порядку: `json_schema` → `json_object` → только промпт
  (многие free-модели не поддерживают `response_format`). Рабочий режим запоминается.
- Невалидный ответ отправляется модели обратно с текстом ошибки (одна попытка исправления).
- Жёсткий лимит 8 обращений к API на запрос; итоговая ошибка — `ParseError` с причинами.
- В промпте есть примеры (few-shot) и защита от prompt injection.

Подбор бесплатных моделей:
```bash
python ai/eval/list_free_models.py
```

## Оценка качества (30 эталонных запросов, `ai/eval/cases.json`)
```bash
python ai/eval/run_eval.py --models model1:free,model2:free --delay 3
```
Показывает точность по `budget`, `tasks`, `resolution`, полное совпадение и число ошибок.
Запустите для 2–3 моделей и оставьте лучшую первой в `OPENROUTER_MODELS`.

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
