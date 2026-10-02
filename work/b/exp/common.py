"""B 피처 실험 공용 — 경로, 표 불러오기, 점수·규칙, 피처를 gap_z 에 붙이는 방식, 모델 도우미.

실험 파일(r01_… ~ r12_…)은 맨 위에서 `import common` 으로 경로를 잡고 여기 함수를 씀.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BDIR = Path(__file__).resolve().parents[1]      # work/b
ROOT = BDIR.parents[1]                          # 저장소 루트
for p in (str(ROOT), str(BDIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import Dataset  # noqa: E402
from src.data import WEIGHT  # noqa: E402
from lab.cv import feature_table  # noqa: E402
from lab.folds import load_folds, unseen_symbols  # noqa: E402
from features_b import build_b, build_b_extra  # noqa: E402

OUT = BDIR / "cache"                            # 실험 결과 csv
SEEDS = (0, 1, 2)
VALID_DAYS = 40                                 # LightGBM 조기 종료용 train 마지막 구간
BASE = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=15,
            min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, verbose=-1, num_threads=4)


# ------------------------------------------------------------------ 표
def build_all(day):
    return build_b(day).merge(build_b_extra(day), on="symbol", how="left")


def fold_days(folds):
    return sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)


def load_table():
    """A 폴드의 모든 날 × B 피처 72개 (cache/b_v2.parquet). label 은 int."""
    folds = load_folds(Dataset())
    t = feature_table(fold_days(folds), build_all, name="b_v2")
    t["label"] = t.label.astype(int)
    return t, folds


SKIP = {"date", "symbol", "label", "ret_pct", "gap", "gap_z"}   # gap_z 에 붙일 후보에서 뺄 칼럼


def engineered(t):
    """표 칼럼으로 만든 새 후보 (같은 행 값만 씀 → 시점 문제 없음)."""
    sg = np.sign(t.gap)
    return {
        "new_pre_z": t.pre_move / t.vol20,                                    # 장전 움직임 ÷ 평소 변동성
        "new_post_z": t.post_ret / t.vol20,                                   # 전일 장후 움직임 ÷ 평소 변동성
        "new_agree3": np.sign(t.post_ret).fillna(0) + np.sign(t.pre_move).fillna(0) + sg,  # 장후·장전·갭 방향 일치 수
        "new_idio_gap_z": (t.gap - t.beta60 * t.mkt_gap) / t.vol20,           # 베타로 시장 몫을 뺀 갭
        "new_ext_liq": np.log1p(t.ext_n),                                     # 시간외 봉 수 (유동성)
    }


def add_engineered(t):
    for k, v in engineered(t).items():
        t[k] = v.replace([np.inf, -np.inf], np.nan)
    return t


def full_table():
    """b_v2 + 새 후보 + b_v3(r09) + b_v4(r11) + 과거 결과 피처. 후보 칼럼 목록도 돌려줌."""
    from builders import past_feats
    table, folds = load_table()
    add_engineered(table)
    for name in ("b_v3", "b_v4"):
        c = feature_table(fold_days(folds), None, name=name)     # r09 / r11 이 먼저 만들어 둔 것
        table = table.merge(c.drop(columns=["label", "ret_pct"]), on=["date", "symbol"], how="left")
    table = past_feats(table)
    feats = [c for c in table.columns if c not in SKIP and pd.api.types.is_numeric_dtype(table[c])]
    for c in feats:
        table[c] = pd.to_numeric(table[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return table, folds, feats


def split(table, f, unseen):
    """폴드 f 의 (train, val). 처음 보는 종목은 train 에서 뺌."""
    tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
    va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
    return tr, va


# ------------------------------------------------------------------ 점수·규칙
QS = [.5, .6, .7, .8, .85, .9, .95, .975, .99]   # 경계 후보 분위 (A 방식)


def fscore(y, p):
    """src.score 의 score 와 같은 값 (빠른 버전)."""
    o = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (WEIGHT * o).sum() / (WEIGHT * e).sum()


def cut(x, a, b):
    """x ≥ b → 4, ≥ a → 3, ≤ −a → 1, ≤ −b → 0, 그 밖·결측 → 2."""
    o = np.full(len(x), 2)
    o[x >= a], o[x >= b], o[x <= -a], o[x <= -b] = 3, 4, 1, 0
    o[np.isnan(x)] = 2
    return o


def fit_ab(x, y, qs=QS):
    """|x| 분위 후보 중 train score 최대인 경계 (a, b)."""
    ok = ~np.isnan(x)
    q = np.quantile(np.abs(x[ok]), qs)
    return max(((fscore(y[ok], cut(x[ok], a, b)), a, b) for i, a in enumerate(q) for b in q[i + 1:]))[1:]


def fit_eval(s_tr, y_tr, s_va, y_va):
    a, b = fit_ab(s_tr, y_tr)
    return fscore(y_va, cut(s_va, a, b))


def inner_splits(dates, inner=60):
    """train 날짜 안의 내부 검증 두 구간 (마지막 120일을 60일씩): [(학습 마스크, 검증 마스크)]. 5거래일 띄움."""
    u = np.sort(np.unique(dates))
    out = []
    for k in (2, 1):
        v0, v1 = u[-k * inner], (u[-(k - 1) * inner] if k > 1 else None)
        tr = dates < u[-k * inner - 5]
        va = (dates >= v0) & ((dates < v1) if v1 is not None else True)
        out.append((tr, va))
    return out


# ------------------------------------------------------------------ gap_z 에 피처 붙이기
BETAS = {"더하기": [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
         "곱하기": [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5],
         "같은방향": [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]}


def apply(form, b, g, x, sd):
    """더하기 g + b·sd·x / 곱하기 g·exp(b·x) / 같은방향 g·exp(b·x·sign(g))."""
    if form == "더하기":
        return g + b * sd * x
    if form == "곱하기":
        return g * np.exp(b * x)
    return g * np.exp(b * x * np.sign(g))


def stdz(ref, x):
    """train(ref) 중앙값·표준편차로 표준화, ±5 로 자르고 결측 0. 표준편차가 0 이면 None."""
    med, sd = np.nanmedian(ref), np.nanstd(ref)
    if not sd > 0:
        return None
    return np.nan_to_num(np.clip((x - med) / sd, -5, 5))


def prep(table, folds, feats, unseen):
    """폴드마다 gap_z, 표준화한 피처, 정답을 미리 만들어 둠 (one / judge 입력)."""
    data = {}
    for f in folds:
        tr, va = split(table, f, unseen)
        gz_tr = tr.gap_z.values
        x = {}
        for c in feats:
            xt = stdz(tr[c].values, tr[c].values)
            if xt is not None:
                x[c] = (xt, stdz(tr[c].values, va[c].values))
        data[f.name] = {"gz_tr": gz_tr, "gz_va": va.gap_z.values, "sd": np.nanstd(gz_tr), "x": x,
                        "y_tr": tr.label.values, "y": va.label.values, "u": va.symbol.isin(unseen).values}
    return data


def one(name, form, folds_data):
    """피처 하나 × 방식 하나: β 후보마다 (train score, val score). β=0 은 gap_z 규칙."""
    out = []
    for fn, d in folds_data.items():
        xt, xv = d["x"][name]
        gt, gv, sd = d["gz_tr"], d["gz_va"], d["sd"]
        for b in BETAS[form]:
            st, sv = apply(form, b, gt, xt, sd), apply(form, b, gv, xv, sd)
            lo, hi = fit_ab(st, d["y_tr"])
            pv = cut(sv, lo, hi)
            out.append({"피처": name, "방식": form, "fold": fn, "beta": b,
                        "train": fscore(d["y_tr"], cut(st, lo, hi)), "all": fscore(d["y"], pv),
                        "unseen": fscore(d["y"][d["u"]], pv[d["u"]])})
    return out


def judge(r):
    """one 결과 → 폴드마다 train 최대 β 로 고른 val Δ, 채택 여부 (평균 +0.005, 3폴드 상승, fold4 하락 없음)."""
    base = r[(r.beta == 0) & (r["방식"] == "더하기")].groupby("fold")["all"].mean()
    ga = r.loc[r.groupby(["피처", "방식", "fold"])["train"].idxmax()]
    da = ga.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(base, axis=1)
    ok = (da.mean(axis=1) >= 0.005) & ((da > 0).sum(axis=1) >= 3) & (da["fold4"] >= 0)
    return ok, da


# ------------------------------------------------------------------ LightGBM
def train_predict(table, folds, feats, seed, params=None):
    """다중분류. 폴드마다 (train 확률, val 확률, train label, val 행). 처음 보는 종목은 학습에서 뺌."""
    import lightgbm as lgb
    p = {**BASE, "seed": seed, **(params or {})}
    unseen = set(unseen_symbols())
    out = {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        cutd = np.sort(tr.date.unique())[-VALID_DAYS]
        a, v = tr[tr.date < cutd], tr[tr.date >= cutd]
        m = lgb.train(p, lgb.Dataset(a[feats], a.label), 2000, valid_sets=[lgb.Dataset(v[feats], v.label)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b = lgb.train(p, lgb.Dataset(tr[feats], tr.label), max(m.best_iteration, 20))
        out[f.name] = (b.predict(tr[feats]), b.predict(va[feats]), tr.label.values, va)
    return out


def lgb_binary(tr, feats, y, seed, leaf):
    """이진 분류 (조기 종료 후 train 전체로 다시 학습)."""
    import lightgbm as lgb
    p = {**BASE, "objective": "binary", "seed": seed, "min_data_in_leaf": leaf}
    p.pop("num_class")
    cutd = np.sort(tr.date.unique())[-VALID_DAYS]
    a, v = tr.date < cutd, tr.date >= cutd
    m = lgb.train(p, lgb.Dataset(tr.loc[a, feats], y[a]), 2000, valid_sets=[lgb.Dataset(tr.loc[v, feats], y[v])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, lgb.Dataset(tr[feats], y), max(m.best_iteration, 20))


def lgb_regress(tr, va, cols, seed, mono=None):
    """목표 z = 다음날 수익률 ÷ vol20 (±6) 회귀 → (train 예측, val 예측)."""
    import lightgbm as lgb
    p = {**BASE, "objective": "regression", "seed": seed}
    p.pop("num_class")
    if mono is not None:
        p.update(monotone_constraints=mono, monotone_constraints_method="advanced")
    z = np.clip(np.nan_to_num(tr.ret_pct.values / 100 / tr.vol20.values), -6, 6)
    cutd = np.sort(tr.date.unique())[-VALID_DAYS]
    a, v = tr.date.values < cutd, tr.date.values >= cutd
    m = lgb.train(p, lgb.Dataset(tr.loc[a, cols], z[a]), 2000, valid_sets=[lgb.Dataset(tr.loc[v, cols], z[v])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(p, lgb.Dataset(tr[cols], z), max(m.best_iteration, 20))
    return m.predict(tr[cols]), m.predict(va[cols])
