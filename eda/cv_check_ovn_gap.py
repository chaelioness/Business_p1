"""EDA 핵심 발견(ovn_gap) 을 폴드 CV 로 확인:  uv run python -m eda.cv_check_ovn_gap

lab/baselines.py 의 gap 기준선과 똑같은 절차(폴드마다 train 에서 임계값 선택)로, 갭 정의만
바꿔서 비교함. EDA 는 train 구간만 봤고, 여기서 val 점수를 보는 것은 팀이 정한 검증 절차임.
val 결과를 보고 임계값을 다시 손대지 말 것. 결과: eda/out/cv_ovn_gap/
"""

import numpy as np
import pandas as pd

from lab.baselines import Gap, _cut, build_gap
from lab.cv import feature_table, run_cv
from src import score

from eda.common import Section

S = Section("cv_ovn_gap")


def build_ovn_gap(day):
    """전일(기준일) 종가 → cutoff 전 마지막 시간외 체결가 %. EDA 의 ovn_gap 과 같은 값."""
    p = day.price(hours=20, columns=["symbol", "datetime", "close", "session"])
    last = p[p.session != "regular"].sort_values("datetime").groupby("symbol").close.last()
    d = day.daily(days=2, columns=["symbol", "date_et", "close"])
    prev = d[d.date_et == day.date].set_index("symbol").close
    return ((last / prev - 1) * 100).rename("gap").reset_index()


class OvnGap(Gap):
    def predict(self, day):
        x = pd.DataFrame({"symbol": day.symbols}).merge(build_ovn_gap(day), on="symbol", how="left")
        return pd.DataFrame({"symbol": x.symbol, "label": _cut(x.gap.to_numpy(), self.a, self.b)})


def _fit(t, exclude, cls):
    t = t[~t.symbol.isin(exclude)].dropna(subset=["gap"])
    g, y = t.gap.to_numpy(), t.label.astype(int).to_numpy()
    qs = np.quantile(np.abs(g), [.1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, .975, .99])
    best = max(((score(y, _cut(g, a, b))["score"], a, b)
                for i, a in enumerate(qs) for b in qs[i + 1:]), key=lambda r: r[0])
    return cls(best[1], best[2])


def fit_ovn_gap(train_days, exclude):
    return _fit(feature_table(train_days, build_ovn_gap, name="ovn_gap", verbose=False), exclude, OvnGap)


def fit_pre_gap(train_days, exclude):
    """기존 기준선과 같은 정의, 같은 (넓힌) 분위 후보. 후보 차이로 인한 점수 차를 없애려는 것."""
    return _fit(feature_table(train_days, build_gap, name="gap", verbose=False), exclude, Gap)


def main():
    rows = []
    for name, fit in [("pre_move(기존 gap 정의)", fit_pre_gap), ("ovn_gap(전일 종가 기준)", fit_ovn_gap)]:
        print(name)
        res = run_cv(fit)
        s = res.summary()
        s.columns = [f"{name} {c}" for c in s.columns]
        rows.append(s)
    t = pd.concat(rows, axis=1)
    S.tab(t, "tab_cv_gap_definitions")
    print(t)
    S.save()


if __name__ == "__main__":
    main()
