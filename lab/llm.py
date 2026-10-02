"""팀 공용 LLM 호출. W&B Inference (OpenAI 호환) 를 부르고 호출을 Weave 에 남김.

    from lab.llm import complete, estimate, usage

    msgs = [[{"role": "user", "content": f"이 기사는 실적·가이던스·소송·M&A 같은 큰 사건인가? {t}"}]
            for t in titles]
    estimate(msgs, max_calls=500)          # 새로 부를 호출 수를 출력, 한도 넘으면 멈춤
    out = [complete(m, temperature=0, max_calls=500) for m in msgs]
    usage()                                # 이번 실행 / 누적 호출 수·토큰 수

- Weave 프로젝트는 lab/wandb.json 의 entity / weave_project (BI_finance_project/finance-llm).
- 키는 WANDB_API_KEY 또는 `uv run wandb login` 으로 저장한 것. 키는 출력하지 않음.
- 같은 (모델, 메시지, 파라미터) 는 cache/llm/ 에서 꺼내고 다시 부르지 않음.
- 누적 호출 수·토큰 수는 cache/llm/usage.json. 대량 호출 전에는 팀에 먼저 말할 것.

원칙
- LLM 은 학습 기간 데이터 분석·라벨링에만 씀. src/model.py 는 LLM 을 부르지 않음.
- 입력에 정답(day.y) 이나 cutoff 이후 정보를 넣지 않음. 뉴스는 day.news(...) 로 꺼낸 것만.
- 검증·홀드아웃 기간 기사로 프롬프트를 고르지 않음.
"""

import hashlib
import json
import os
import time
import warnings

from src.paths import ROOT

from .tracking import HOST, settings

BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "OpenPipe/Qwen3-14B-Instruct"
CACHE = ROOT / "cache" / "llm"
USAGE = CACHE / "usage.json"

_state = {"client": None, "call": None}
_session = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}


class MissingKey(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def _key():
    """WANDB_API_KEY, 없으면 wandb login 이 저장한 netrc. 값은 반환만 하고 출력하지 않음."""
    k = os.environ.get("WANDB_API_KEY")
    if k:
        return k
    import netrc
    from pathlib import Path
    for name in (".netrc", "_netrc"):
        try:
            a = netrc.netrc(Path.home() / name).authenticators(HOST)
            if a and a[2]:
                return a[2]
        except (FileNotFoundError, netrc.NetrcParseError, OSError):
            pass
    raise MissingKey("W&B API 키가 없습니다. `uv run wandb login` 을 하거나 "
                     "환경변수 WANDB_API_KEY 를 설정하세요 (키를 코드·파일에 적지 말 것).")


def project():
    s = settings()
    return f"{s['entity']}/{s['weave_project']}"


def _setup():
    """OpenAI 클라이언트와 weave.init 은 처음 한 번만."""
    if _state["client"] is not None:
        return
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai 가 없습니다. `uv sync` 로 개발 의존성을 설치하세요.") from e
    _state["client"] = OpenAI(base_url=BASE_URL, api_key=_key(), project=project())

    def call(model: str, messages: list, params: dict) -> dict:
        r = _state["client"].chat.completions.create(model=model, messages=messages, **params)
        u = r.usage
        return {"text": r.choices[0].message.content,
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0}

    try:
        import weave
        weave.init(project())
        call = weave.op(call, name="complete")
    except ImportError:
        warnings.warn("weave 가 없어 호출을 Weave 에 남기지 않습니다 (호출은 그대로 함).")
    except Exception as e:
        warnings.warn(f"weave.init 실패, 기록 없이 계속함: {type(e).__name__}: {e}")
    _state["call"] = call


def _norm(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


def cache_key(messages, model=DEFAULT_MODEL, **params):
    blob = json.dumps({"model": model, "messages": _norm(messages), "params": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(k):
    return CACHE / k[:2] / f"{k}.json"


def _add_usage(prompt, completion):
    CACHE.mkdir(parents=True, exist_ok=True)
    total = json.loads(USAGE.read_text(encoding="utf-8")) if USAGE.exists() else {}
    for k, v in (("calls", 1), ("prompt_tokens", prompt), ("completion_tokens", completion)):
        total[k] = total.get(k, 0) + v
    USAGE.write_text(json.dumps(total, indent=2), encoding="utf-8")


def complete(messages, model=DEFAULT_MODEL, *, max_calls=None, max_tokens_total=None,
             cache=True, **params) -> str:
    """메시지(또는 문자열 하나)를 보내고 답 문자열을 돌려줌.

    params 는 chat.completions.create 에 그대로 (temperature, max_tokens ...).
    max_calls / max_tokens_total 은 이번 실행에서 새로 부르는 호출·토큰 한도. 넘으면 BudgetExceeded.
    """
    messages = _norm(messages)
    k = cache_key(messages, model, **params)
    p = _path(k)
    if cache and p.exists():
        _session["cached"] += 1
        return json.loads(p.read_text(encoding="utf-8"))["text"]

    if max_calls is not None and _session["calls"] >= max_calls:
        raise BudgetExceeded(f"호출 한도 {max_calls}회에 닿았습니다.")
    used = _session["prompt_tokens"] + _session["completion_tokens"]
    if max_tokens_total is not None and used >= max_tokens_total:
        raise BudgetExceeded(f"토큰 한도 {max_tokens_total} 에 닿았습니다 (사용 {used}).")

    _setup()
    r = _state["call"](model, messages, params)
    _session["calls"] += 1
    _session["prompt_tokens"] += r["prompt_tokens"]
    _session["completion_tokens"] += r["completion_tokens"]
    _add_usage(r["prompt_tokens"], r["completion_tokens"])
    if cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": model, "messages": messages, "params": params,
                                 "created": time.strftime("%Y-%m-%d %H:%M:%S"), **r},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return r["text"]


def estimate(batch, model=DEFAULT_MODEL, max_calls=None, **params) -> int:
    """대량 호출 전에: 캐시에 없는(새로 부를) 호출 수를 출력. max_calls 를 넘으면 멈춤."""
    new = sum(not _path(cache_key(m, model, **params)).exists() for m in batch)
    print(f"LLM 예상 호출: 전체 {len(batch)} 중 새 호출 {new} (캐시 {len(batch) - new}), 모델 {model}")
    if max_calls is not None and new > max_calls:
        raise BudgetExceeded(f"새 호출 {new} 회가 한도 {max_calls} 회를 넘습니다.")
    return new


def usage() -> dict:
    total = json.loads(USAGE.read_text(encoding="utf-8")) if USAGE.exists() else {}
    return {"session": dict(_session), "total": total}
