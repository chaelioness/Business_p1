"""B 피처 누수 검사.

    uv run python work/b/test_leak.py

임의의 기준일 5개에서 두 가지를 확인한다.
1. 감시 래퍼: day.daily/price/earnings/analyst 가 돌려준 모든 행이 known_at < cutoff 이고,
   build_b 가 day.y / day.symbols(정답표에서 나옴) / day.ds 를 건드리지 않는다.
2. 교란: cutoff 이후 행의 숫자 값을 전부 바꾼 Dataset 으로 다시 만들어도 피처가 같다.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from src import Dataset  # noqa: E402
from features_b import B_FEATURES, build_b  # noqa: E402
from folds import TEST_START  # noqa: E402

N_DAYS, SEED = 5, 0


class GuardDay:
    """Day 를 감싸 허용된 접근만 통과시키고 반환 행의 known_at 을 검사한다."""

    ALLOWED = {"date", "target", "cutoff", "start_of"}
    TABLES = {"daily", "price", "earnings", "analyst"}

    def __init__(self, day):
        object.__setattr__(self, "_day", day)
        object.__setattr__(self, "calls", [])

    def __getattr__(self, name):
        day = object.__getattribute__(self, "_day")
        if name in self.ALLOWED:
            return getattr(day, name)
        if name in self.TABLES:
            def call(*a, **kw):
                df = getattr(day, name)(*a, **kw)
                self.calls.append((name, len(df)))
                if len(df):
                    assert (df.known_at < day.cutoff).all(), f"{name}: cutoff 이후 행"
                return df
            return call
        raise AssertionError(f"build_b 가 금지된 속성 day.{name} 을 씀")


class PerturbedDataset(Dataset):
    """table() 이 cutoff(=T) 이후 행의 숫자 값을 바꿔서 읽힌다."""

    T = None

    def table(self, name, since=None, until=None, **kw):
        df = super().table(name, since, None, **kw)
        if self.T is not None and len(df):
            late = df.known_at >= self.T
            num = df.select_dtypes("number").columns
            df[num] = df[num].astype(float)
            df.loc[late, num] = df.loc[late, num] * 1.37 + 11
        if until is not None:
            df = df[df.known_at < pd.Timestamp(until)]
        return df.reset_index(drop=True)


def main():
    ds = Dataset()
    pds = PerturbedDataset()
    pds.labels()                                   # 정답표는 교란 전 값으로 고정
    days = [d for d in ds.days() if d.target < pd.Timestamp(TEST_START)]
    picked = sorted(random.Random(SEED).sample(range(len(days)), N_DAYS))
    ok = True
    for i in picked:
        day = days[i]
        g = GuardDay(day)
        a = build_b(g)
        pds.T = day.cutoff
        b = build_b(pds.day(day.date))
        pds.T = None
        same = (a.symbol.tolist() == b.symbol.tolist()
                and np.allclose(a[B_FEATURES].to_numpy(float), b[B_FEATURES].to_numpy(float),
                                equal_nan=True))
        ok &= same
        print(f"{day.date:%Y-%m-%d} -> {day.target:%Y-%m-%d}  감시 통과 {dict(g.calls)}  "
              f"교란 후 동일 {same}")
    print("누수 검사 통과" if ok else "누수 검사 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
