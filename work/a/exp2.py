"""v2 실험: LLM 없는 텍스트 피처 + 여러 모델·손실. 모두 run_cv (A 폴드, 주 지표 fold4) 로 W&B 기록.

피처 표 (master, 종목·기준일당 한 줄, 전부 guard 아래에서 계산된 캐시)
    B 피처 전체 (cache/b_v1.parquet: gap_z, ext_range_z, vol20 ...)
    C 텍스트 7개 (cache/text_v1_dev)
    사전(lexicon) 사건 수: 장 마감 후 기사 제목에서 미리 정한 단어 목록을 셈 (아래 EVENTS, 결과 보기 전에 정함)
    학습 텍스트 점수: 폴드 train 기사 제목으로만 학습한 TF-IDF 로지스틱 → 종목·날짜별 평균 (text_big, text_dir)

실험용이라 predict 는 master 에서 그날 행을 꺼냄 (master 자체는 guard 아래에서 만든 값).
채택된 것만 src/model.py 에서 day 로 바로 계산하도록 옮김.
"""

import re

import numpy as np
import pandas as pd

from lab.cv import CACHE, feature_table
from src import score

from . import gapz_rule as R
from .features_b import build_b
from .news_titles import titles

LV = np.arange(5)
QS = R.QS
TEXT_C = ["news_log_count_overnight", "news_spike20", "reddit_log_count_overnight", "reddit_spike20",
          "reddit_persistence3", "text_both_spike", "text_overnight_hours"]

# 사건 단어 목록. 점수를 보기 전에 정함 (바꾸면 worklog 에 기록).
EVENTS = {
    "ev_earn": r"\b(?:earnings|results|quarter|q[1-4]|eps|revenue|sales|profit)\b",
    "ev_guide": r"\b(?:guidance|outlook|forecast|reaffirm\w*)\b",
    "ev_analyst": r"\b(?:upgrade\w*|downgrade\w*|price target|initiat\w*)\b",
    "ev_legal": r"\b(?:lawsuit|sues?|sued|probe|investigation|antitrust|recall\w*|subpoena)\b",
    "ev_deal": r"\b(?:acquir\w*|acquisition|merger|buyout|takeover)\b",
    "ev_fda": r"\b(?:fda|approval|approved|phase \d|trial)\b",
    "ev_mgmt": r"\b(?:ceo|cfo|resign\w*|steps down|layoffs?|job cuts)\b",
    "ev_pos": r"\b(?:beats?|tops|surpass\w*|raises|soar\w*|surge\w*|jump\w*|rall(?:y|ies))\b",
    "ev_neg": r"\b(?:miss(?:es)?|falls short|cuts|lowers|plunge\w*|slump\w*|tumble\w*|sink\w*)\b",
}
EV = list(EVENTS)
TEXT_L = ["text_big", "text_big_max", "text_dir"]
LEX = [f"lex_{k}" for k in EV] + ["lex_n", "lex_event", "lex_tone"]

_master = None


# ---------------------------------------------------------------- 표

def lexicon(t):
    """제목 → 종목·날짜별 사건 수 (log1p)."""
    low = t.title.fillna("").str.lower()
    x = pd.DataFrame({"date": t.date, "symbol": t.symbol})
    for k, pat in EVENTS.items():
        x[k] = low.str.contains(pat, regex=True).astype(float)
    x["tone"] = t.tone
    g = x.groupby(["date", "symbol"])
    out = np.log1p(g[EV].sum()).add_prefix("lex_")
    out["lex_n"] = np.log1p(g.size())
    out["lex_event"] = np.log1p(g[["ev_earn", "ev_guide", "ev_deal", "ev_legal", "ev_fda", "ev_mgmt"]].sum().sum(1))
    out["lex_tone"] = g.tone.mean()
    return out.reset_index()


def master(days):
    """dev 날짜 전부의 피처 표 (+ label, ret_pct). 한 번 만들면 메모리에 둠."""
    global _master
    if _master is None:
        t = feature_table(days, build_b, name="b_v1", verbose=False)
        tg = pd.DataFrame({"date": [d.date for d in days], "target": [d.target for d in days]})
        t = t.merge(tg, on="date", how="left")
        c = pd.read_parquet(CACHE / "text_v1_dev" / "train_text_features.parquet")
        t = t.merge(c[["symbol", "target", *TEXT_C]], on=["symbol", "target"], how="left")
        tt = titles(days, verbose=False)
        t = t.merge(lexicon(tt), on=["date", "symbol"], how="left")
        for k in LEX:
            if k != "lex_tone":
                t[k] = t[k].fillna(0.0)
        t["lex_posneg"] = t.lex_ev_pos - t.lex_ev_neg
        t["abs_mkt_gap"] = t.mkt_gap.abs()
        t["label"] = t.label.astype(int)
        t["yz"] = t.ret_pct / 100 / t.vol20          # 자기 변동성 단위 수익률 (회귀 목표)
        _master = t
    return _master


# ---------------------------------------------------------------- 학습 텍스트 점수 (폴드마다)

MAX_TITLES = 20      # 종목·날짜당 학습에 쓰는 제목 수 (기사 많은 종목이 학습을 독차지하지 않게)


def _titles_capped(dates, symbols_out):
    tt = titles_cache()
    tt = tt[tt.date.isin(dates) & ~tt.symbol.isin(symbols_out)]
    tt = tt.drop_duplicates(["date", "symbol", "title"])
    return tt.groupby(["date", "symbol"]).tail(MAX_TITLES)


_tc = None


_tbd = None


def titles_by_date():
    global _tbd
    if _tbd is None:
        _tbd = dict(tuple(titles_cache().groupby("date")))
    return _tbd


def titles_cache():
    global _tc
    if _tc is None:
        _tc = pd.read_parquet(CACHE / "news_titles.parquet", columns=["date", "symbol", "title"])
    return _tc


class TextScorer:
    """제목 TF-IDF → 로지스틱 2개: 급등락인가(text_big), 급등락이면 위인가(text_dir). train 날짜로만 학습."""

    def __init__(self, train_dates, exclude, y):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression

        self.vec = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False,
                                     norm=None, stop_words="english")
        tt = _titles_capped(set(train_dates), set(exclude)).merge(y, on=["date", "symbol"])
        X = self.vec.transform(tt.title.fillna(""))
        self.tf = TfidfTransformer(sublinear_tf=True).fit(X)
        X = self.tf.transform(X)
        big = (tt.label.isin([0, 4])).astype(int).to_numpy()
        self.big = LogisticRegression(C=0.5, max_iter=300).fit(X, big)
        m = tt.label.isin([0, 1, 3, 4]).to_numpy()
        self.dir = LogisticRegression(C=0.5, max_iter=300).fit(X[m], (tt.label[m] >= 3).astype(int))

    def features(self, dates):
        by = titles_by_date()
        parts = [by[d] for d in dates if d in by]
        if not parts:
            return pd.DataFrame(columns=["date", "symbol", "text_big", "text_big_max", "text_dir"])
        tt = pd.concat(parts).drop_duplicates(["date", "symbol", "title"])
        X = self.tf.transform(self.vec.transform(tt.title.fillna("")))
        x = tt[["date", "symbol"]].assign(text_big=self.big.predict_proba(X)[:, 1],
                                          text_dir=self.dir.predict_proba(X)[:, 1] - 0.5)
        g = x.groupby(["date", "symbol"])
        return pd.concat([g.text_big.mean(), g.text_big.max().rename("text_big_max"),
                          g.text_dir.mean()], axis=1).reset_index()


# ---------------------------------------------------------------- 공통: 연속 점수 s → 등급

def fit_cut(s, y):
    """|s| 분위 후보에서 train score 최대인 경계 (a, b). B 규칙과 같은 방식."""
    ok = ~np.isnan(s)
    q = np.quantile(np.abs(s[ok]), QS)
    return max(((score(y[ok], R.cut(s[ok], a, b))["score"], a, b)
                for i, a in enumerate(q) for b in q[i + 1:]), key=lambda r: r[0])


def zfit(x):
    x = np.asarray(x, float)
    return np.nanmedian(x), np.nanstd(x) or 1.0


def zapply(x, ms):
    return np.nan_to_num(np.clip((np.asarray(x, float) - ms[0]) / ms[1], -5, 5))


class TableModel:
    """predict(day): master 에서 그날 행을 꺼내 self.labels(df) 로 등급."""

    def __init__(self, labels, info=None, importance=None, text=None):
        self.labels, self.info, self.text = labels, info or {}, text
        if importance is not None:
            self.feature_importance = importance

    def predict(self, day):
        m = _master
        x = m[m.date == day.date]
        if self.text is not None:
            x = x.merge(self.text.features([day.date]), on=["date", "symbol"], how="left")
            x[TEXT_L] = x[TEXT_L].astype(float)
        out = pd.DataFrame({"symbol": list(day.symbols)})
        if len(x):
            x = x.assign(label_pred=self.labels(x))
            out = out.merge(x[["symbol", "label_pred"]], on="symbol", how="left")
        out["label"] = out.get("label_pred", pd.Series(dtype=float)).fillna(2).astype(int)
        return out[["symbol", "label"]]


def train_rows(train_days, exclude):
    m = _master
    dates = {d.date for d in train_days}
    return m[m.date.isin(dates) & ~m.symbol.isin(exclude)].copy()
