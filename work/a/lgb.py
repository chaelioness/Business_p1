"""LightGBM 5클래스 + 디코딩 비교 (A).

    from work.a.lgb import make_fit
    run_cv(make_fit("match_train"), ...)

피처: B 의 B_FEATURES_V1 26개 (work/a/features_b.py). 표는 cache/b_v1.parquet (guard 아래 계산).
파라미터: B 기본값. 반복 수는 학습 구간 마지막 40 기준일로 조기 종료해 정하고 전체로 다시 학습.
디코딩 (확률 p → 등급)
    argmax          가장 큰 확률
    match_train     기대 등급 e = Σ i·p_i 를 train 정답 비율 분위로 자름 (B 의 분포 맞춤)
    match_recent    같은데 비율은 train 마지막 60 기준일 정답 비율
    aggr            train 비율에서 급등락 ×k, 보합 ÷k 후 분위로 자름. k 는 조기 종료용 40일 구간에서 score 최대
경계(e 분위)는 언제나 train 예측으로만 정함.
"""

import numpy as np
import pandas as pd

from lab.cv import feature_table
from src import score
from src.paths import ROOT

from .features_b import B_FEATURES_V1, build_b

FEATS = B_FEATURES_V1
PARAMS = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, seed=0, verbose=-1, num_threads=4)
VALID_DAYS = 40
RECENT_DAYS = 60
KS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LV = np.arange(5)

TEXT_TABLE = ROOT / "cache" / "text_v1_dev" / "train_text_features.parquet"
TEXT = {"news": ["news_log_count_overnight", "news_spike20"],
        "reddit": ["reddit_log_count_overnight", "reddit_spike20", "reddit_persistence3"],
        "joint": ["text_both_spike"], "context": ["text_overnight_hours"]}

_models = {}      # 같은 폴드·피처는 디코딩만 바꿀 때 다시 학습하지 않음
_feats = {}       # 기준일 → 피처 (predict 가 같은 날을 여러 번 계산하지 않게)
_text = None


def text_table():
    """C 텍스트 피처 (work/c run.py 로 dev 데이터에서 다시 만든 표). symbol + target 으로 붙임."""
    global _text
    if _text is None:
        _text = pd.read_parquet(TEXT_TABLE).drop(columns=["date", "cutoff"])
    return _text


def table(days):
    t = feature_table(days, build_b, name="b_v1", verbose=False)
    tg = pd.DataFrame({"date": [d.date for d in days], "target": [d.target for d in days]})
    return t.merge(tg, on="date", how="left")


def with_text(t):
    out = t.merge(text_table(), on=["symbol", "target"], how="left", validate="one_to_one")
    assert len(out) == len(t)
    return out


def _train(t, feats, rounds=None):
    import lightgbm as lgb

    ds = lambda d: lgb.Dataset(d[feats], d.label)  # noqa: E731
    if rounds is None:
        cut = np.sort(t.date.unique())[-VALID_DAYS]
        fit, val = t[t.date < cut], t[t.date >= cut]
        m = lgb.train(PARAMS, ds(fit), num_boost_round=2000, valid_sets=[ds(val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(m.best_iteration, 20)
        return m, rounds, val
    return lgb.train(PARAMS, ds(t), num_boost_round=rounds), rounds, None


def thresholds(e_ref, q):
    """e_ref 분포에서 등급 비율 q 가 되도록 하는 경계 4개."""
    q = np.asarray(q, float) / np.sum(q)
    return np.quantile(e_ref, np.cumsum(q)[:-1])


def cut(e, th):
    return np.searchsorted(th, e, side="right")


def aggr_q(q, k):
    q = np.asarray(q, float).copy()
    q[[0, 4]] *= k
    q[2] /= k
    return q / q.sum()


def props(y):
    return np.bincount(np.asarray(y, int), minlength=5) / len(y)


class LGBModel:
    def __init__(self, booster, decode, th=None, info=None, feats=FEATS):
        self.booster, self.decode, self.th, self.info, self.feats = booster, decode, th, info or {}, feats

    def proba(self, x):
        return self.booster.predict(x[self.feats])

    def predict(self, day):
        if day.date not in _feats:
            _feats[day.date] = build_b(day)
        x = _feats[day.date]
        if any(f not in x for f in self.feats):          # 텍스트 피처: C 표에서 (symbol, target) 로
            x = x.assign(target=day.target).merge(text_table(), on=["symbol", "target"], how="left")
        x = x.set_index("symbol").reindex(day.symbols)
        label = pd.Series(2, index=x.index)
        ok = x[FEATS].notna().any(axis=1)
        if ok.any():
            p = self.proba(x.loc[ok])
            label[ok] = p.argmax(1) if self.decode == "argmax" else cut(p @ LV, self.th)
        return pd.DataFrame({"symbol": x.index, "label": label.astype(int).values})

    def feature_importance(self):
        return pd.Series(self.booster.feature_importance("gain"), index=self.feats)

    def save(self, path):
        self.booster.save_model(str(path.with_suffix(".txt")))


def fit_all(train_days, exclude, feats=FEATS):
    """폴드 하나: 모델 + 네 가지 디코딩의 경계."""
    key = (train_days[0].date, train_days[-1].date, tuple(exclude), tuple(feats))
    if key in _models:
        return _models[key]
    t = table(train_days)
    if any(f not in t for f in feats):
        t = with_text(t)
    t = t[~t.symbol.isin(exclude)].copy()
    t["label"] = t.label.astype(int)
    m_es, rounds, val = _train(t, feats)
    m, _, _ = _train(t, feats, rounds)

    e_tr = m.predict(t[feats]) @ LV
    q_tr = props(t.label)
    recent = t[t.date >= np.sort(t.date.unique())[-RECENT_DAYS]]
    q_re = props(recent.label)

    e_es_tr = m_es.predict(t[t.date < val.date.min()][feats]) @ LV
    e_val = m_es.predict(val[feats]) @ LV
    ks = {k: score(val.label, cut(e_val, thresholds(e_es_tr, aggr_q(q_tr, k))))["score"] for k in KS}
    k = max(ks, key=ks.get)

    out = {"booster": m, "rounds": rounds, "k": k, "k_scores": ks,
           "th": {"match_train": thresholds(e_tr, q_tr), "match_recent": thresholds(e_tr, q_re),
                  "aggr": thresholds(e_tr, aggr_q(q_tr, k))},
           "q_train": q_tr, "q_recent": q_re}
    _models[key] = out
    return out


def make_fit(decode, feats=FEATS):
    feats = list(feats)

    def fit(train_days, exclude):
        f = fit_all(train_days, exclude, feats)
        return LGBModel(f["booster"], decode, f["th"].get(decode),
                        {"rounds": f["rounds"], "k": f["k"]}, feats)
    fit.__qualname__ = f"fit_lgb_{decode}"
    return fit
