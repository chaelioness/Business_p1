"""제출 모델 v1: gap_z 규칙 + ext_range_z (B 최종안, work/a/gapz_rule.py 를 옮김).

    load_model()      artifacts/gapz_rule.json (학습 결과 숫자 5개) 만 읽음
    Model.predict()   Day 하나 → symbol / label (0~4)

규칙
    gap         = cutoff 전 마지막 시간외 봉 종가 ÷ 기준일 종가 − 1   (기준일 16:00 이후 post/pre 봉)
    vol20       = 최근 20거래일 일간 수익률 표준편차
    gap_z       = gap ÷ vol20
    ext_range_z = (시간외 봉 최고가 − 최저가) ÷ 기준일 종가 ÷ vol20
    x̃ = (ext_range_z − med) ÷ sd, ±5 로 자름, 결측 0              (med·sd 는 train)
    s = gap_z · exp(−λ·x̃)                                        (λ 는 train score 최대)
    s ≥ b → 4 / s ≥ a → 3 / s ≤ −a → 1 / s ≤ −b → 0 / 그 밖 → 2

갭을 못 구한 종목(시간외 봉 누락, 기준일 일봉 없음 등)은 보합(2). 방향 정보가 없을 때 급등락을 찍으면
반대로 틀렸을 때 벌점이 커서 보합이 가장 안전함.
학습은 scripts/a_train_v1.py (채점 때 다시 돌리지 않음). numpy, pandas, json 만 씀.
"""

import json

import numpy as np
import pandas as pd

from .data import FLAT
from .paths import ARTIFACTS

PARAMS_FILE = "gapz_rule.json"
CLOSE = pd.Timedelta(hours=16)


def build_gapz(day):
    """symbol / gap_z / ext_range_z. 기준일 일봉이 없으면 빈 표."""
    cols = ["symbol", "gap_z", "ext_range_z"]
    d = day.daily(days=40, columns=["symbol", "date_et", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=cols)
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    C = d.pivot_table(index="date_et", columns="symbol", values="close", aggfunc="last").sort_index()
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    close = C.loc[last]

    p = day.price(since=day.date, columns=["symbol", "datetime", "high", "low", "close", "session"])
    ext = p[(p.datetime >= day.date + CLOSE) & p.session.isin(["post", "pre"])].sort_values("datetime")
    g = ext.groupby("symbol")
    out = pd.DataFrame(index=pd.Index(close.index, name="symbol"))
    out["gap_z"] = (g.close.last() / close - 1) / vol20
    out["ext_range_z"] = (g.high.max() - g.low.min()) / close / vol20
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out[cols]


def strength(gap_z, ext_range_z, params):
    x = np.nan_to_num(np.clip((ext_range_z - params["med"]) / params["sd"], -5, 5))
    return gap_z * np.exp(-params["lam"] * x)


def cut(s, a, b):
    o = np.full(len(s), FLAT)
    o[s >= a], o[s >= b], o[s <= -a], o[s <= -b] = 3, 4, 1, 0
    o[np.isnan(s)] = FLAT
    return o


class Model:
    def __init__(self, params):
        self.params = params

    def predict(self, day):
        """day.symbols 전부에 label. 피처가 없는 종목은 보합."""
        out = pd.DataFrame({"symbol": list(day.symbols)})
        x = build_gapz(day)
        if len(x):
            s = strength(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), self.params)
            x = x.assign(label=cut(s, self.params["a"], self.params["b"]))
            out = out.merge(x[["symbol", "label"]], on="symbol", how="left")
        else:
            out["label"] = FLAT
        out["label"] = out.label.fillna(FLAT).astype(int)
        return out


def load_model():
    with open(ARTIFACTS / PARAMS_FILE, encoding="utf-8") as f:
        return Model(json.load(f))
