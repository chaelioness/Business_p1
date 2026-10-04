"""lab/llm.py: 캐시·한도·키 오류. 실제 API 는 부르지 않음."""
import pytest

from lab import llm


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE", tmp_path)
    monkeypatch.setattr(llm, "USAGE", tmp_path / "usage.json")
    for k in llm._session:
        monkeypatch.setitem(llm._session, k, 0)
    calls = []

    def call(model, messages, params):
        calls.append(messages)
        return {"text": f"ok{len(calls)}", "prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setitem(llm._state, "client", object())
    monkeypatch.setitem(llm._state, "call", call)
    return calls


def test_cache_hits_do_not_call(fake):
    assert llm.complete("hi", temperature=0) == "ok1"
    assert llm.complete("hi", temperature=0) == "ok1"
    assert llm.complete("hi", temperature=0.5) == "ok2"     # 파라미터가 다르면 새 호출
    assert len(fake) == 2
    u = llm.usage()
    assert u["session"]["calls"] == 2 and u["session"]["cached"] == 1
    assert u["total"]["prompt_tokens"] == 20


def test_budget(fake):
    llm.complete("a", max_calls=1)
    with pytest.raises(llm.BudgetExceeded):
        llm.complete("b", max_calls=1)
    assert llm.estimate(["a", "b", "c"]) == 2
    with pytest.raises(llm.BudgetExceeded):
        llm.estimate(["a", "b", "c"], max_calls=1)


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with pytest.raises(llm.MissingKey, match="wandb login"):
        llm._key()


def test_project_name():
    assert llm.project() == "BI_finance_project/finance-llm"
