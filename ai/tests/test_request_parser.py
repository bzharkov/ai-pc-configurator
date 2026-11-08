import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "userPrompt"))
from request_parser import ParseError, parse_request  # noqa: E402


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def test_basic():
    c = FakeClient(['{"budget": 120000, "tasks": ["Games","Games","Montage"], "resolution": "2K", "preferences": null}'])
    r = parse_request("ПК для игр в 2К и монтажа до 120к", client=c, model="m")
    assert r.budget == 120000 and r.tasks == ["Games", "Montage"]
    assert r.resolution == "1440p" and r.preferences == "" and r.missing == []


def test_missing_budget_and_string_budget():
    c = FakeClient(['{"budget": "120 000 ₽", "tasks": [], "resolution": null, "preferences": "тихий"}'])
    r = parse_request("x", client=c, model="m")
    assert r.budget == 120000 and r.missing == ["tasks"]
    c = FakeClient(['{"budget": null, "tasks": ["Office"]}'])
    assert parse_request("x", client=c, model="m").missing == ["budget"]


def test_markdown_fence():
    c = FakeClient(['```json\n{"budget": 50000, "tasks": ["Office"]}\n```'])
    assert parse_request("x", client=c, model="m").budget == 50000


def test_retry_then_success():
    c = FakeClient(["не json", '{"budget": 1, "tasks": ["Bogus"]}', '{"budget": 70000, "tasks": ["Study"]}'])
    assert parse_request("x", client=c, model="m").tasks == ["Study"]
    assert c.calls == 3


def test_fails_after_retries():
    c = FakeClient(["bad"] * 3)
    with pytest.raises(ParseError):
        parse_request("x", client=c, model="m")


def test_empty_and_long():
    with pytest.raises(ParseError):
        parse_request("   ", client=FakeClient([]))
    with pytest.raises(ParseError):
        parse_request("a" * 3000, client=FakeClient([]))
