"""B 규칙을 run_cv 용 fit 으로 감쌈.

    fit_gapz       gap_z 규칙만 (λ = 0)
    fit_gapz_ext   B 최종안: gap_z · exp(−λ·x̃(ext_range_z)), λ·경계는 train 에서
"""

import json

import numpy as np

from lab.cv import feature_table

from . import gapz_rule as R


class GapzRule:
    def __init__(self, params):
        self.params = params

    def predict(self, day):
        return R.predict(day, self.params)

    def save(self, path):
        path.with_suffix(".json").write_text(json.dumps(self.params, indent=2), encoding="utf-8")


def _table(train_days, exclude):
    t = feature_table(train_days, R.build_gapz, name="gapz", verbose=False)
    t = t[~t.symbol.isin(exclude)].copy()
    t["label"] = t.label.astype(int)
    return t


def fit_gapz_ext(train_days, exclude):
    return GapzRule(R.fit_table(_table(train_days, exclude)))


def fit_gapz(train_days, exclude):
    old = R.LAMS
    R.LAMS = [0.0]
    try:
        return GapzRule(R.fit_table(_table(train_days, exclude)))
    finally:
        R.LAMS = old
