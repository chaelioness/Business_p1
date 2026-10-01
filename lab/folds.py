"""팀 공통 walk-forward 폴드.

    from lab.folds import load_folds
    for f in load_folds(ds):
        f.name, f.train, f.val          # Day 리스트

구간은 lab/folds.json 에 날짜로 박혀 있음. 모두 같은 파일을 읽으니 폴드가 갈릴 일이 없음.
다시 만들 때만 `uv run python -m lab.folds` 를 돌림(데이터가 같으면 결과도 같음).

설계
    홀드아웃   2026-06-01 이후 (target 기준). 여기는 폴드에 안 들어감
    val       폴드마다 60거래일(약 3달), 홀드아웃 앞 240거래일을 4개로 나눔
    embargo   train 과 val 사이 12거래일을 비움. 실제 평가도 데이터 끝(9/14)과
              평가 시작(10/1) 사이에 12거래일이 비어 있어서 그 간격을 그대로 따름
    train     embargo 앞 전부 (expanding)
    holdout   마지막에 한 번만 보는 구간. 실제 상황과 같게 train 끝과 홀드아웃 사이에도
              12거래일을 비움 (ds.split("2026-06-01") 을 그대로 쓰면 이 간격이 없어서
              점수가 실제보다 좋게 나옴). load_folds(ds, holdout=True) 일 때만 나옴
    unseen    평가에 처음 보는 종목 10개가 나오므로, 50개 중 10개를 고정해 둠.
              학습할 때 이 종목을 빼고 val 점수를 seen / unseen 으로 나눠 봄

시차 기준 (슬라이드 7~9쪽)
    - 기준일 d 의 정답은 target(다음 거래일) 16:00 에 확정, 피처는 target 09:30 에서 잘림
    - 그래서 경계는 target 으로 셈: train 마지막 target 과 val 첫 target 사이에
      거래일이 정확히 EMBARGO 개 비어 있어야 함 (make_spec 에서 assert)
    - 실제: 마지막 정답 9/14 -> 평가 첫 target 10/1, 사이 9/15~9/30 이 12거래일
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HOLDOUT = "2026-06-01"
N_FOLDS = 4
VAL_DAYS = 60
EMBARGO = 12
N_UNSEEN = 10
SEED = 2026
PATH = Path(__file__).with_name("folds.json")


@dataclass
class Fold:
    name: str
    train: list          # Day 리스트
    val: list            # Day 리스트

    def __repr__(self):
        return (f"<{self.name} train {len(self.train)}일 "
                f"({self.train[0].date:%Y-%m-%d}~{self.train[-1].date:%Y-%m-%d}) | "
                f"val {len(self.val)}일 ({self.val[0].date:%Y-%m-%d}~{self.val[-1].date:%Y-%m-%d})>")


def make_spec(ds) -> dict:
    """데이터의 거래일로 폴드 날짜를 정함. 기준일(date) 범위로 적어 둠."""
    dev, test = ds.split(HOLDOUT)
    n = len(dev)
    trading = ds.dates()

    def gap(train, first):
        """train 마지막 target 과 first 의 target 사이에 낀 거래일 수."""
        a, b = train[-1].target, first.target
        return sum(1 for x in trading if a < x < b)

    folds = []
    for k in range(N_FOLDS):
        v0 = n - (N_FOLDS - k) * VAL_DAYS
        v1 = v0 + VAL_DAYS
        tr = dev[: v0 - EMBARGO]
        va = dev[v0:v1]
        assert gap(tr, va[0]) == EMBARGO, (k, gap(tr, va[0]))
        folds.append({
            "name": f"fold{k + 1}",
            "train": [f"{tr[0].date:%Y-%m-%d}", f"{tr[-1].date:%Y-%m-%d}"],
            "embargo": [f"{dev[v0 - EMBARGO].date:%Y-%m-%d}", f"{dev[v0 - 1].date:%Y-%m-%d}"],
            "val": [f"{va[0].date:%Y-%m-%d}", f"{va[-1].date:%Y-%m-%d}"],
            "n_train": len(tr), "n_val": len(va),
        })
    tr = dev[: n - EMBARGO + 1]
    while gap(tr, test[0]) < EMBARGO:
        tr = tr[:-1]
    assert gap(tr, test[0]) == EMBARGO
    holdout = {
        "name": "holdout",
        "train": [f"{tr[0].date:%Y-%m-%d}", f"{tr[-1].date:%Y-%m-%d}"],
        "embargo": [f"{dev[len(tr)].date:%Y-%m-%d}", f"{dev[-1].date:%Y-%m-%d}"],
        "val": [f"{test[0].date:%Y-%m-%d}", f"{test[-1].date:%Y-%m-%d}"],
        "n_train": len(tr), "n_val": len(test),
    }

    syms = sorted(ds.symbols)
    unseen = sorted(random.Random(SEED).sample(syms, N_UNSEEN))
    return {
        "holdout_from": HOLDOUT,
        "rule": f"val {VAL_DAYS}거래일 x {N_FOLDS}, embargo {EMBARGO}거래일, expanding train. "
                "날짜는 전부 기준일(day.date) 기준, 양끝 포함",
        "unseen_symbols": unseen,
        "folds": folds,
        "holdout": holdout,
    }


def load_spec() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def load_folds(ds, spec=None, holdout=False) -> list[Fold]:
    """folds.json 대로 Day 리스트를 나눠 돌려줌. holdout=True 면 홀드아웃 폴드 하나만."""
    spec = spec or load_spec()
    days = ds.days()

    def pick(lo, hi):
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        return [d for d in days if lo <= d.date <= hi]

    chosen = [spec["holdout"]] if holdout else spec["folds"]
    return [Fold(f["name"], pick(*f["train"]), pick(*f["val"])) for f in chosen]


def unseen_symbols(spec=None) -> list[str]:
    return (spec or load_spec())["unseen_symbols"]


if __name__ == "__main__":
    from src import Dataset

    spec = make_spec(Dataset())
    PATH.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(PATH)
    for f in [*spec["folds"], spec["holdout"]]:
        print(f"  {f['name']}  train {f['train'][0]}~{f['train'][1]} ({f['n_train']}일)"
              f"  embargo {f['embargo'][0]}~{f['embargo'][1]}"
              f"  val {f['val'][0]}~{f['val'][1]} ({f['n_val']}일)")
    print("  unseen", spec["unseen_symbols"])
