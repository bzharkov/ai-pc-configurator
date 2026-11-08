"""Детерминированная страховка: явные значения из текста запроса.

Нужна, когда слабая модель пропускает очевидное (бюджет «200к», «4k», «для танков»).
Правила консервативные: заполняют только то, что явно есть в тексте.
"""
import re
from typing import Optional

_RES_PATTERNS = [
    ("4K", re.compile(r"(?<![\d.,])\b4\s*[kк]\b|\buhd\b|\b2160p\b", re.I)),
    ("1440p", re.compile(r"(?<![\d.,])\b2\s*[kк]\b|\bqhd\b|\b1440p\b|\bwqhd\b", re.I)),
    ("1080p", re.compile(r"\bfull\s*hd\b|\bfhd\b|\b1080p\b|\bфулл\s*хд\b", re.I)),
]

_TASK_KEYWORDS = {
    "Games": r"игр|гейм|game|gaming|танк|fortnite|фортнайт|cs\s?2?|кс\b|dota|дот[аыу]|minecraft|майнкрафт|cyberpunk|warzone|valorant",
    "Montage": r"монтаж|рендерин\w*\s+видео|видеоредакт|premiere|davinci|resolve|after\s*effects|видеомонтаж|video\s*edit|editing",
    "Office": r"офис|office|бухгалт|интернет|ютуб|youtube|программир|работ[аыуе]\b",
    "3D": r"\b3d\b|3д|blender|maya|рендер|render|3ds\s*max|zbrush|cinema\s*4d",
    "Study": r"учёб|учеб|студент|школ|универ|study",
}

_MULT = (("млн", 1_000_000), ("милл", 1_000_000), ("тыс", 1000), ("k", 1000), ("к", 1000))

_RES_STRIP = re.compile(
    r"(?<![\d.,])\b[248]\s*[kк]\b|\b\d{3,4}p\b|\bqhd\b|\buhd\b", re.I)
_NUM = re.compile(
    r"(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*(млн|милл\w*|тыс\w*|[kк])?(?![\w])", re.I)


def extract_resolution(text: str) -> Optional[str]:
    for value, pat in _RES_PATTERNS:
        if pat.search(text):
            return value
    return None


def extract_budget(text: str) -> Optional[int]:
    t = _RES_STRIP.sub(" ", text.lower())
    if re.search(r"\bмиллион\b", t) and not re.search(r"\d\s*млн", t):
        return 1_000_000
    for m in _NUM.finditer(t):
        raw, suffix = m.group(1), (m.group(2) or "")
        rest = t[m.end():m.end() + 6]
        if re.match(r"\s*(гб|gb|fps|гц|hz|вт|w\b|мб|тб|tb|ядер|core)", rest):
            continue  # характеристики, а не деньги
        num = raw.strip().replace(" ", "").replace(",", ".")
        if not re.fullmatch(r"\d+(\.\d+)?", num):
            num = re.sub(r"[^\d]", "", num)
            if not num:
                continue
        factor = next((f for k, f in _MULT if suffix.startswith(k)), 1)
        value = int(float(num) * factor)
        has_currency = bool(re.search(r"₽|руб|\bр\b", t[m.start():m.end() + 6]))
        if factor > 1 or has_currency or value >= 10000:
            if value >= 5000:
                return value
    return None


def extract_tasks(text: str) -> list[str]:
    t = text.lower()
    return [task for task, pat in _TASK_KEYWORDS.items() if re.search(pat, t)]


def apply_heuristics(text: str, req):
    """Дополнить/поправить результат модели явными значениями из текста."""
    update = {}
    budget = extract_budget(text)
    if budget is not None and req.budget != budget:
        update["budget"] = budget  # явное число в тексте надёжнее ответа модели
    if req.resolution is None:
        res = extract_resolution(text)
        if res:
            update["resolution"] = res
    if not req.tasks:
        tasks = extract_tasks(text)
        if tasks:
            update["tasks"] = tasks
    if not update:
        return req
    return req.model_copy(update=update).with_missing()
