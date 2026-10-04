"""v2 모델들. 전부 '연속 점수 s → |s| 경계 (a, b) 로 5등급' 구조 (B 규칙과 같은 디코딩).

    make_fit(kind, feats, text=False, **kw) → run_cv 용 fit

kind
    rule        s = gap_z · exp(−λ·z(ext_range_z) + γ·z(f))       (f = feats[0], 없으면 B 규칙)
    ridge       선형 회귀 (제곱 손실) 로 yz = 수익률/vol20 예측
    huber       선형 회귀 (Huber 손실, 큰 수익률에 덜 끌려감)
    logit       다항 로지스틱 → 기대 등급 − 2
    lgb_l2 / lgb_huber / lgb_l1     LightGBM 회귀 (yz), gap_z 에 단조 증가 제약
    lgb_ord     LightGBM 회귀 (label − 2, 순서형 근사)
    lgb_wmulti  LightGBM 다항 + 점수식 가중 (급등락 행 가중 4, 보합 0.25) → 기대 등급 − 2
    hgb_l1      sklearn HistGradientBoosting 회귀 (절대 오차 손실)
경계·λ·γ·반복 수는 train 에서만. 트리 모델은 train 마지막 40일을 떼어 반복 수와 경계를 정하고 전체로 다시 학습.
text=True 면 폴드 train 기사 제목으로 학습한 text_big / text_dir 를 붙임
(train 행의 값은 날짜 4구간 교차 적합으로 만들어 제 자신을 외운 값이 아님).
"""

import numpy as np
import pandas as pd

from src import score

from . import exp2 as E

VALID_DAYS = 40
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4]
GAMS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
CLASS_W = {0: 4.0, 1: 1.0, 2: 0.25, 3: 1.0, 4: 4.0}
LGB = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0, verbose=-1, num_threads=4)


# ---------------------------------------------------------------- 텍스트 점수 붙이기

def add_text(t, train_dates, exclude):
    """train 행: 날짜 4구간 교차 적합. 반환한 scorer 는 train 전체로 학습 (val 용)."""
    y = t[["date", "symbol", "label"]]
    dates = np.sort(np.asarray(list(train_dates)))
    parts = []
    for blk in np.array_split(dates, 4):
        other = [d for d in dates if d not in set(blk)]
        sc = E.TextScorer(other, exclude, y[y.date.isin(other)])
        parts.append(sc.features(blk))
    t = t.merge(pd.concat(parts), on=["date", "symbol"], how="left")
    t[E.TEXT_L] = t[E.TEXT_L].astype(float)
    return t, E.TextScorer(dates, exclude, y)


# ---------------------------------------------------------------- 점수 함수들

def _xy(t, feats):
    X = t[feats].to_numpy(float)
    return X, np.clip(t.yz.to_numpy(float), -10, 10)


def fit_rule(t, feats):
    gz, y = t.gap_z.to_numpy(float), t.label.to_numpy(int)
    me = E.zfit(t.ext_range_z)
    xe = E.zapply(t.ext_range_z, me)
    f = feats[0] if feats else None
    mf = E.zfit(t[f]) if f else None
    xf = E.zapply(t[f], mf) if f else 0.0
    best = None
    for lam in LAMS:
        for gam in (GAMS if f else [0.0]):
            sc, a, b = E.fit_cut(gz * np.exp(-lam * xe + gam * xf), y)
            if best is None or sc > best[0]:
                best = (sc, lam, gam, a, b)
    sc, lam, gam, a, b = best

    def s(x):
        xf_ = E.zapply(x[f], mf) if f else 0.0
        return x.gap_z.to_numpy(float) * np.exp(-lam * E.zapply(x.ext_range_z, me) + gam * xf_)
    return s, (a, b), {"lam": lam, "gam": gam, "train_score": sc}


def _lin(t, feats, kind):
    from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X, y = _xy(t, feats)
    if kind == "logit":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=500)).fit(X, t.label)
        s = lambda x: m.predict_proba(x[feats].to_numpy(float)) @ E.LV - 2  # noqa: E731
    else:
        reg = Ridge(alpha=10.0) if kind == "ridge" else HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), reg).fit(X, y)
        s = lambda x: m.predict(x[feats].to_numpy(float))  # noqa: E731
    sc, a, b = E.fit_cut(s(t), t.label.to_numpy(int))
    return s, (a, b), {"train_score": sc}


DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
BETAS = [-0.2, -0.1, 0.0, 0.1, 0.2]


def fit_rule_size(t, feats):
    """B 규칙에서 급등락 경계만 피처로 움직임: |s| ≥ b·exp(−δ·z(f)) 이면 급등락.
    방향(부호)과 보합 경계 a 는 B 규칙 그대로. f 가 크면(사건 기사·실적) 급등락으로 더 쉽게 찍게 할지 train 이 정함."""
    s1, (a0, b0), i1 = fit_rule(t, [])
    f = feats[0]
    mf = E.zfit(t[f])
    y = t.label.to_numpy(int)
    v = s1(t)
    xf = E.zapply(t[f], mf)
    q = np.quantile(np.abs(v[~np.isnan(v)]), E.QS)

    def lab(v, xf, a, b, d):
        o = E.R.cut(v, a, np.inf)
        bb = b * np.exp(-d * xf)
        o[v >= bb] = 4
        o[v <= -bb] = 0
        o[np.isnan(v)] = 2
        return o
    best = max(((score(y, lab(v, xf, a, b, d))["score"], a, b, d)
                for i, a in enumerate(q) for b in q[i + 1:] for d in DELTAS), key=lambda r: r[0])
    sc, a, b, d = best

    def labels(x):
        return lab(np.asarray(s1(x), float), E.zapply(x[f], mf), a, b, d)
    return labels, {"lam": i1["lam"], "delta": d, "a": a, "b": b, "train_score": sc}


def fit_rule_add(t, feats):
    """B 규칙 + 텍스트 방향 점수 더하기: s = s_rule + β·sd(s_rule)·z(f)."""
    s1, _, i1 = fit_rule(t, [])
    f = feats[0]
    mf = E.zfit(t[f])
    y = t.label.to_numpy(int)
    v, sd = s1(t), np.nanstd(s1(t))
    best = max(((*E.fit_cut(v + bt * sd * E.zapply(t[f], mf), y), bt) for bt in BETAS), key=lambda r: r[0])
    sc, a, b, bt = best
    return (lambda x: s1(x) + bt * sd * E.zapply(x[f], mf)), (a, b), {"beta": bt, "lam": i1["lam"], "train_score": sc}


KAPPAS = [-1.0, -0.75, -0.5, -0.25, 0.0]
THETAS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def fit_rule_gen(t, feats):
    """s = (gap + κ·mkt_gap + θ·pre_move) / vol20 · exp(−λ·z(ext_range_z)).
    κ<0 이면 시장 전체 갭을 덜 믿음 (A EDA: 시장 갭은 장중 되돌림), θ 는 장전 움직임 비중. 모두 train."""
    me = E.zfit(t.ext_range_z)
    y = t.label.to_numpy(int)

    def sfun(x, k, th, lam):
        g = x.gap.to_numpy(float) + k * x.mkt_gap.to_numpy(float) + th * np.nan_to_num(x.pre_move.to_numpy(float))
        return g / x.vol20.to_numpy(float) * np.exp(-lam * E.zapply(x.ext_range_z, me))
    use_k = KAPPAS if "kappa" in feats else [0.0]
    use_t = THETAS if "theta" in feats else [0.0]
    best = max(((*E.fit_cut(sfun(t, k, th, lam), y), k, th, lam)
                for k in use_k for th in use_t for lam in LAMS), key=lambda r: r[0])
    sc, a, b, k, th, lam = best
    return (lambda x: sfun(x, k, th, lam)), (a, b), {"kappa": k, "theta": th, "lam": lam, "train_score": sc}


ENS_W = [0.0, 0.25, 0.5, 1.0, 2.0]


def fit_ens(t, feats):
    """B 규칙 s 와 Huber 선형 예측을 각자 train 표준편차로 나눠 더함: s_rule + w·s_huber. w 는 train."""
    s1, _, i1 = fit_rule(t, [])
    s2, _, _ = _lin(t, feats, "huber")
    y = t.label.to_numpy(int)
    v1, v2 = s1(t), s2(t)
    d1, d2 = np.nanstd(v1), np.nanstd(v2)
    best = max(((*E.fit_cut(v1 / d1 + w * v2 / d2, y), w) for w in ENS_W), key=lambda r: r[0])
    sc, a, b, w = best
    return (lambda x: s1(x) / d1 + w * s2(x) / d2), (a, b), {"w": w, "lam": i1["lam"], "train_score": sc}


def _lgb_params(kind, feats):
    p = dict(LGB)
    if kind == "lgb_wmulti":
        p.update(objective="multiclass", num_class=5)
    else:
        p.update(objective={"lgb_l2": "regression", "lgb_huber": "huber", "lgb_l1": "regression_l1",
                            "lgb_ord": "regression"}[kind])
        if kind not in ("lgb_ord", "lgb_l1"):     # l1 은 단조 제약을 못 씀
            p["monotone_constraints"] = [1 if f == "gap_z" else 0 for f in feats]
    return p


def _tree(t, feats, kind):
    """마지막 40일로 반복 수·경계, 전체로 다시 학습."""
    cut_d = np.sort(t.date.unique())[-VALID_DAYS]
    fit, val = t[t.date < cut_d], t[t.date >= cut_d]
    if kind == "hgb_l1":
        from sklearn.ensemble import HistGradientBoostingRegressor
        mk = lambda: HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=15,  # noqa: E731
                                                   min_samples_leaf=300, learning_rate=0.05,
                                                   max_iter=300, random_state=0)
        m_es = mk().fit(*_xy(fit, feats))
        m = mk().fit(*_xy(t, feats))
        pr = lambda mm, x: mm.predict(x[feats].to_numpy(float))  # noqa: E731
        imp = None
    else:
        import lightgbm as lgb
        p = _lgb_params(kind, feats)
        if kind == "lgb_wmulti":
            tgt = lambda d: d.label  # noqa: E731
            w = lambda d: d.label.map(CLASS_W)  # noqa: E731
        else:
            tgt = (lambda d: d.label - 2) if kind == "lgb_ord" else (lambda d: np.clip(d.yz, -10, 10))
            w = lambda d: None  # noqa: E731
        ds = lambda d: lgb.Dataset(d[feats], tgt(d), weight=w(d))  # noqa: E731
        m_es = lgb.train(p, ds(fit), 2000, valid_sets=[ds(val)],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(m_es.best_iteration, 20)
        m = lgb.train(p, ds(t), rounds)
        if kind == "lgb_wmulti":
            pr = lambda mm, x: mm.predict(x[feats]) @ E.LV - 2  # noqa: E731
        else:
            pr = lambda mm, x: mm.predict(x[feats])  # noqa: E731
        imp = pd.Series(m.feature_importance("gain"), index=feats)
    sc, a, b = E.fit_cut(pr(m_es, val), val.label.to_numpy(int))
    return (lambda x: pr(m, x)), (a, b), {"valid_score": sc}, imp


def make_fit(kind, feats, text=False):
    feats = list(feats)

    def fit(train_days, exclude):
        t = E.train_rows(train_days, exclude)
        scorer = None
        if text:
            t, scorer = add_text(t, [d.date for d in train_days], exclude)
        imp = None
        if kind == "rule_size":
            labels, info = fit_rule_size(t, feats)
            return E.TableModel(labels, info, None, scorer)
        if kind == "winvote":                    # 전체·최근 250·최근 120 기준일 규칙의 등급 중간값
            ud = np.sort(t.date.unique())
            fits = [fit_rule(t[t.date >= ud[-min(n, len(ud))]], []) for n in (10 ** 6, 250, 120)]

            def vote(x):
                ls = []
                for s_, (a_, b_), _ in fits:
                    v = np.asarray(s_(x), float)
                    v[np.isnan(x.gap_z.to_numpy(float))] = np.nan
                    ls.append(E.R.cut(v, a_, b_))
                return np.median(np.vstack(ls), axis=0).astype(int)
            return E.TableModel(vote, {"lams": [f[2]["lam"] for f in fits]}, None, scorer)
        if kind.startswith("recent"):            # 최근 N 기준일만으로 B 규칙
            n = int(kind[6:])
            ud = np.sort(t.date.unique())
            t = t[t.date >= ud[-min(n, len(ud))]]
            s, (a, b), info = fit_rule(t, [])
        elif kind == "rule_gen":
            s, (a, b), info = fit_rule_gen(t, feats)
        elif kind == "rule_add":
            s, (a, b), info = fit_rule_add(t, feats)
        elif kind == "rule":
            s, (a, b), info = fit_rule(t, feats)
        elif kind == "ens_huber":
            s, (a, b), info = fit_ens(t, feats)
        elif kind in ("ridge", "huber", "logit"):
            s, (a, b), info = _lin(t, feats, kind)
        else:
            s, (a, b), info, imp = _tree(t, feats, kind)

        def labels(x):
            v = np.asarray(s(x), float)
            v[np.isnan(x.gap_z.to_numpy(float))] = np.nan      # 갭이 없으면 보합 (B 규칙과 같게)
            return E.R.cut(v, a, b)
        return E.TableModel(labels, {**info, "a": a, "b": b}, imp, scorer)
    fit.__qualname__ = f"fit_{kind}"
    return fit
