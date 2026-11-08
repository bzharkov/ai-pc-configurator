"""Схема структурированных требований пользователя (контракт ИИ -> бэкенд)."""
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Task = Literal["Games", "Montage", "Office", "3D", "Study"]
Resolution = Literal["1080p", "1440p", "4K"]


class Requirements(BaseModel):
    budget: Optional[int] = Field(None, description="Бюджет в рублях, null если не указан")
    tasks: list[Task] = Field(default_factory=list)
    resolution: Optional[Resolution] = Field(None, description="Целевое разрешение для игр")
    preferences: str = ""
    # Что нужно уточнить у пользователя (заполняется кодом, не моделью)
    missing: list[str] = Field(default_factory=list)

    @field_validator("budget", mode="before")
    @classmethod
    def _parse_budget(cls, v):
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            v = int(v)
        elif isinstance(v, str):
            m = re.search(r"(\d[\d\s.,]*)\s*(млн|миллион\w*|тыс\w*|k|к)?", v.lower())
            if not m:
                return None
            num = m.group(1).strip()
            suffix = m.group(2) or ""
            factor = next((f for key, f in (("млн", 1_000_000), ("милл", 1_000_000), ("тыс", 1000), ("k", 1000), ("к", 1000)) if suffix.startswith(key)), 1)
            if factor > 1:
                v = int(float(num.replace(" ", "").replace(",", ".")) * factor)
            else:
                v = int(re.sub(r"[^\d]", "", num))
        else:
            raise ValueError("budget must be a number")
        if v <= 0:
            return None
        return v

    @field_validator("tasks", mode="before")
    @classmethod
    def _dedupe_tasks(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return list(dict.fromkeys(v))

    @field_validator("resolution", mode="before")
    @classmethod
    def _norm_resolution(cls, v):
        if v is None:
            return None
        key = str(v).strip().lower().replace(" ", "")
        aliases = {
            "1080p": "1080p", "fullhd": "1080p", "fhd": "1080p",
            "1440p": "1440p", "2k": "1440p", "qhd": "1440p",
            "4k": "4K", "2160p": "4K", "uhd": "4K",
        }
        return aliases.get(key)  # неизвестное значение -> None

    @field_validator("preferences", mode="before")
    @classmethod
    def _prefs(cls, v):
        return "" if v is None else str(v).strip()

    def with_missing(self) -> "Requirements":
        missing = []
        if self.budget is None:
            missing.append("budget")
        if not self.tasks:
            missing.append("tasks")
        return self.model_copy(update={"missing": missing})
