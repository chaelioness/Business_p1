"""규칙 기준선을 W&B 에 run 으로 남김:  uv run python scripts/log_baselines.py

검증: A 폴드 4개 (lab/folds.json). 주 지표 fold4, fold1~3 은 참고 (2026-10-02 팀 결정, 안 2).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda.cv_check_ovn_gap import fit_ovn_gap, fit_pre_gap  # noqa: E402
from lab.baselines import fit_flat, fit_yesterday  # noqa: E402
from lab.cv import run_cv  # noqa: E402
from src import Dataset  # noqa: E402

METHOD = "A-folds4 (주 지표 fold4)"


class Random:
    """날짜마다 같은 시드로 무작위 등급. p 가 없으면 5등급 균등."""

    def __init__(self, p=None, seed=0):
        self.p, self.seed = p, seed

    def predict(self, day):
        rng = np.random.default_rng([self.seed, int(day.date.strftime("%Y%m%d"))])
        return pd.DataFrame({"symbol": day.symbols,
                             "label": rng.choice(5, size=len(day.symbols), p=self.p)})


def fit_random_uniform(train_days, exclude):
    return Random()


def fit_random_prior(train_days, exclude):
    """train 정답 비율대로 무작위 (학습한 종목만)."""
    y = pd.concat([d.y[~d.y.symbol.isin(exclude)].label for d in train_days]).astype(int)
    return Random(np.bincount(y, minlength=5) / len(y))


RUNS = [
    ("random_uniform", fit_random_uniform, "5등급 균등 무작위 (seed 0)"),
    ("random_prior", fit_random_prior, "train 정답 비율대로 무작위 (seed 0)"),
    ("flat", fit_flat, "전부 보합"),
    ("yesterday", fit_yesterday, "전날 등락률의 label 그대로"),
    ("rule_pre_move", fit_pre_gap, "프리마켓 안 등락(pre_move) 구간 나누기, 경계는 train 에서 score 최대"),
    ("rule_ovn_gap", fit_ovn_gap, "전일 종가 → 마지막 시간외 봉 갭 구간 나누기, 경계는 train 에서 score 최대"),
]


def main():
    ds = Dataset()
    for name, fit, desc in RUNS:
        print(name)
        res = run_cv(fit, ds, name=name, method=METHOD, tags=["baseline", "rule"],
                     config={"model": name, "description": desc, "main_metric": "fold4/all/score"})
        print(res.summary())


if __name__ == "__main__":
    main()
