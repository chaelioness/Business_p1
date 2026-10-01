"""규칙 기반 기준선. 폴드 점수가 이걸 못 넘으면 모델이 아무것도 못 배운 것.

    flat        전부 보합. 정의상 0점
    yesterday   전날 등락률의 label 을 그대로 (교수님 노트북 기준선)
    gap         개장 전 시간외 등락률 하나로 다섯 구간을 나눔. 구간 경계는 train 에서
                score 가 가장 높게 나오는 값을 고름 (폴드마다 다시 고름)
"""

import numpy as np
import pandas as pd

from src import FLAT, label_of, score

from .cv import feature_table


class Const:
    def predict(self, day):
        return pd.DataFrame({"symbol": day.symbols, "label": FLAT})


def fit_flat(train_days, exclude):
    return Const()


class Yesterday:
    def predict(self, day):
        d = day.daily(days=3, columns=["symbol", "date_et", "ret"])
        last = d.sort_values("date_et").groupby("symbol")["ret"].last() * 100
        lab = label_of(last).astype("float").reindex(day.symbols).fillna(FLAT).astype(int)
        return pd.DataFrame({"symbol": day.symbols, "label": lab.values})


def fit_yesterday(train_days, exclude):
    return Yesterday()


# ---------------------------------------------------------------- gap

def build_gap(day):
    """대상일 개장 전(pre 세션) 등락률 %. hours=20 이면 대상일 pre 봉만 잡힘."""
    p = day.price(hours=20, columns=["symbol", "datetime", "close", "session"])
    pre = p[p.session == "pre"].sort_values("datetime").groupby("symbol")["close"]
    g = pre.agg(lambda s: (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan)
    return g.rename("gap").reset_index()


def _cut(g, a, b):
    """|gap| < a 보합, a~b 상승/하락, b 이상 급등락."""
    out = np.full(len(g), FLAT)
    out[g >= a] = 3
    out[g >= b] = 4
    out[g <= -a] = 1
    out[g <= -b] = 0
    out[np.isnan(g)] = FLAT
    return out


class Gap:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def predict(self, day):
        x = pd.DataFrame({"symbol": day.symbols}).merge(build_gap(day), on="symbol", how="left")
        return pd.DataFrame({"symbol": x.symbol, "label": _cut(x.gap.to_numpy(), self.a, self.b)})


def fit_gap(train_days, exclude):
    t = feature_table(train_days, build_gap, name="gap", verbose=False)
    t = t[~t.symbol.isin(exclude)].dropna(subset=["gap"])
    g, y = t.gap.to_numpy(), t.label.astype(int).to_numpy()
    qs = np.quantile(np.abs(g), [.5, .6, .7, .8, .85, .9, .95, .975, .99])
    best = max(((score(y, _cut(g, a, b))["score"], a, b)
                for i, a in enumerate(qs) for b in qs[i + 1:]), key=lambda r: r[0])
    return Gap(best[1], best[2])


BASELINES = {"flat": fit_flat, "yesterday": fit_yesterday, "gap": fit_gap}
