import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import OpenAIError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "userPrompt"))
sys.path.insert(0, str(ROOT / "eval"))
import request_parser  # noqa: E402
from request_parser import ParseError, parse_request  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cache():
    request_parser._mode_cache.clear()


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        self.history = getattr(self, "history", []) + [kwargs]
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
    c = FakeClient(["bad"] * 20)
    with pytest.raises(ParseError):
        parse_request("x", client=c, model="m")


def test_empty_and_long():
    with pytest.raises(ParseError):
        parse_request("   ", client=FakeClient([]))
    with pytest.raises(ParseError):
        parse_request("a" * 3000, client=FakeClient([]))


@pytest.mark.parametrize("raw,expected", [
    ("50к", 50000), ("120 000 ₽", 120000), ("1.5 млн", 1500000),
    ("80 тыс", 80000), ("100k", 100000), ("нет", None), (0, None), (75000, 75000),
])
def test_budget_formats(raw, expected):
    from schemas import Requirements
    assert Requirements(budget=raw).budget == expected


GOOD = '{"budget": 70000, "tasks": ["Study"], "resolution": null, "preferences": ""}'


def test_api_error_falls_back_to_next_mode_and_remembers():
    c = FakeClient([OpenAIError("response_format unsupported"), GOOD])
    assert parse_request("x", client=c, model="m").budget == 70000
    assert c.history[1]["response_format"] == {"type": "json_object"}
    assert request_parser._mode_cache["m"] == "json_object"
    c2 = FakeClient([GOOD])
    parse_request("x", client=c2, model="m")
    assert c2.history[0]["response_format"] == {"type": "json_object"}  # сразу рабочий режим


def test_fallback_to_next_model():
    c = FakeClient([OpenAIError("429")] * 3 + [GOOD])
    parse_request("x", client=c, model=["bad", "good"])
    assert c.history[-1]["model"] == "good"


def test_prompt_mode_has_no_response_format():
    c = FakeClient([OpenAIError("a"), OpenAIError("b"), GOOD])
    parse_request("x", client=c, model="m")
    assert "response_format" not in c.history[2]


def test_error_feedback_is_sent_on_retry():
    c = FakeClient(['{"budget": 1, "tasks": ["Bogus"]}', GOOD])
    parse_request("x", client=c, model="m")
    msgs = c.history[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and "Bogus" in msgs[-2]["content"]
    assert "невалиден" in msgs[-1]["content"]


def test_empty_choices_does_not_crash():
    class Empty:
        choices = None
    c = FakeClient([])
    c.replies = []
    def create(**k):
        c.calls += 1
        return Empty() if c.calls == 1 else SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=GOOD))])
    c.chat.completions.create = create
    assert parse_request("x", client=c, model="m").budget == 70000


def test_call_limit():
    c = FakeClient([OpenAIError("x")] * 50)
    with pytest.raises(ParseError):
        parse_request("x", client=c, model=["a", "b", "c", "d"])
    assert c.calls <= request_parser.MAX_CALLS


def test_eval_cases_are_valid_and_score():
    import json
    import run_eval
    from schemas import Requirements
    cases = json.loads(run_eval.CASES.read_text(encoding="utf-8"))
    assert len(cases) >= 30
    for case in cases:  # эталон проходит собственную схему
        Requirements(budget=case["budget"], tasks=case["tasks"], resolution=case["resolution"])
    ok = Requirements(budget=1, tasks=["Games"], resolution="4K")
    assert run_eval.score({"budget": 1, "tasks": ["Games"], "resolution": "4K"}, ok)["all"]
    assert not run_eval.score({"budget": 2, "tasks": ["Games"], "resolution": "4K"}, ok)["all"]


def test_free_models_filter():
    import list_free_models as lf
    data = [
        {"id": "a:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["response_format"]},
        {"id": "b", "pricing": {"prompt": "0.1", "completion": "0"}},
        {"id": "c:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["structured_outputs", "response_format"]},
    ]
    assert [m["id"] for m in lf.free_models(data)] == ["c:free", "a:free"]
