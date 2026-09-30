"""2단계: B_FEATURES_V1 점검.

    uv run python work/b/v1_check.py diag      # LightGBM 보합 쏠림 원인 비교 + V1 점수
    uv run python work/b/v1_check.py holdout   # 학습에 없던 종목 10개로 평가
    uv run python work/b/v1_check.py universe  # 20종목만 보일 때 시장 평균 피처 변화

점수는 src.data.score 그대로. 폴드는 folds.py.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src import Dataset, score  # noqa: E402
from features_b import B_FEATURES, B_FEATURES_V1, build_b_table  # noqa: E402
from folds import FOLD_STARTS, TEST_START, folds  # noqa: E402

CACHE = HERE / "cache"
SEED = 0
VALID_DAYS = 40
METRICS = ["score", "accuracy", "direction", "big_recall", "big_prec"]
BASE = dict(objective="multiclass", num_class=5, learning_rate=0.05,
            num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            seed=SEED, verbose=-1, num_threads=4)
MKT_FEATS = ["mkt_ret1", "mkt_ret5", "mkt_vol20", "mkt_gap", "mkt_gap_z",
             "gap_rel_z", "beta60", "rel_ret1_z", "rel_ret5_z"]


def load_b():
    b = pd.read_parquet(CACHE / "b_features.parquet")
    b["label"] = b.label.astype(int)
    return b


# ---------------------------------------------------------------- 학습

def class_weight(y):
    """클래스 빈도의 역수. 평균이 1 이 되게 맞춤."""
    f = pd.Series(y).value_counts(normalize=True)
    w = pd.Series(y).map(1 / f).to_numpy()
    return w / w.mean()


def train_lgb(tr, feats, params=None, weighted=False, rounds=None, valid_days=VALID_DAYS):
    """rounds 가 없으면 학습 구간 마지막 valid_days 기준일로 조기 종료 반복수를 정한다."""
    import lightgbm as lgb
    p = {**BASE, **(params or {})}

    def ds(d):
        return lgb.Dataset(d[feats], d.label, weight=class_weight(d.label) if weighted else None)

    if rounds is None:
        cut = np.sort(tr.date.unique())[-valid_days]
        fit, val = tr[tr.date < cut], tr[tr.date >= cut]
        m = lgb.train(p, ds(fit), num_boost_round=2000, valid_sets=[ds(val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(m.best_iteration, 20)
    return lgb.train(p, ds(tr), num_boost_round=rounds), rounds


def match_decode(p_tr, y_tr, p_te):
    """기대 등급 Σ i·p_i 를 학습 구간 label 비율에 맞춘 분위로 자름 (비교용, 디코딩은 A 담당)."""
    lv = np.arange(5)
    e_tr, e_te = p_tr @ lv, p_te @ lv
    cum = np.cumsum(np.bincount(y_tr, minlength=5) / len(y_tr))[:-1]
    return np.searchsorted(np.quantile(e_tr, cum), e_te, side="right")


def metrics(y, pr):
    r = score(y, pr)
    return {k: r[k] for k in METRICS} | {"flat%": float((pr == 2).mean() * 100)}


def summary(rows, by="model"):
    r = pd.DataFrame(rows)
    fold = r.pivot_table(index=by, columns="fold", values="score", sort=False).round(3)
    mean = r.groupby(by, sort=False)[METRICS + ["flat%"]].mean().round(3)
    if "iter" in r:
        mean["iter"] = r.groupby(by, sort=False)["iter"].mean().round(0)
    return pd.concat([fold, mean], axis=1)


# ---------------------------------------------------------------- 1. 진단

VARIANTS = {
    "all  기본": (B_FEATURES, {}, False, None),
    "V1   기본": (B_FEATURES_V1, {}, False, None),
    "V1   클래스 가중": (B_FEATURES_V1, {}, True, None),
    "V1   min_leaf 100": (B_FEATURES_V1, {"min_data_in_leaf": 100}, False, None),
    "V1   반복 300 고정": (B_FEATURES_V1, {}, False, 300),
    "V1   가중+반복 300": (B_FEATURES_V1, {}, True, 300),
}


def diag():
    b = load_b()
    rows = []
    for name, tr_m, te_m in folds(targets=b.target):
        tr, te = b[tr_m], b[te_m]
        for model, (feats, params, w, rounds) in VARIANTS.items():
            m, n = train_lgb(tr, feats, params, w, rounds)
            p = m.predict(te[feats])
            rows.append({"fold": name, "model": model, "iter": n, **metrics(te.label, p.argmax(1))})
            if model == "V1   기본":
                pd_ = match_decode(m.predict(tr[feats]), tr.label.values, p)
                rows.append({"fold": name, "model": "V1   기본+분포맞춤 디코딩", "iter": n,
                             **metrics(te.label, pd_)})
                imp = pd.Series(m.feature_importance("gain"), index=feats)
                rows[-1]["_imp"] = imp / imp.sum()
        print(f"{name} 완료", flush=True)
    imp = pd.concat([r.pop("_imp") for r in rows if "_imp" in r], axis=1).mean(1)
    print("\n== LightGBM 변형 비교 (폴드별 score, 평균 지표, 보합 예측 비율 %) ==")
    print(summary(rows).to_string())
    print("\n실제 평가 구간 보합 비율 %:",
          round(float(np.mean(np.concatenate([b[m].label.values == 2 for _, _, m in folds(targets=b.target)])) * 100), 1))
    print("\n== V1 기본 피처 중요도 (gain 비율, 폴드 평균) ==")
    print((imp.sort_values(ascending=False) * 100).round(1).to_string())
    pd.DataFrame(rows).to_csv(CACHE / "v1_diag.csv", index=False)


# ---------------------------------------------------------------- 2. 새 종목

def holdout(n_hold=10, seeds=(0, 1, 2)):
    """종목 10개를 학습에서 빼고, 평가 구간에서 그 10개 / 나머지 40개 점수를 따로 본다.
    비교 기준: 50종목 전부로 학습한 모델의 같은 10개 점수."""
    b = load_b()
    syms = sorted(b.symbol.unique())
    rows = []
    for s in seeds:
        hold = set(random.Random(s).sample(syms, n_hold))
        for name, tr_m, te_m in folds(targets=b.target):
            tr, te = b[tr_m], b[te_m]
            m_all, _ = train_lgb(tr, B_FEATURES_V1)
            m_seen, _ = train_lgb(tr[~tr.symbol.isin(hold)], B_FEATURES_V1)
            h, o = te[te.symbol.isin(hold)], te[~te.symbol.isin(hold)]
            for model, mdl, part in [("40종목 학습 → 처음 보는 10", m_seen, h),
                                     ("40종목 학습 → 학습한 40", m_seen, o),
                                     ("50종목 학습 → 같은 10", m_all, h)]:
                pr = mdl.predict(part[B_FEATURES_V1]).argmax(1)
                rows.append({"seed": s, "fold": name, "model": model, **metrics(part.label, pr)})
        print(f"seed {s} 제외 종목 {sorted(hold)}", flush=True)
    r = pd.DataFrame(rows)
    print("\n== 새 종목 대응 (V1, 기본 LGB, 3개 제외 조합 × 5폴드 평균) ==")
    print(summary(r.assign(fold=r.fold)).to_string())
    print("\n조합(seed)별 score 평균")
    print(r.pivot_table(index="model", columns="seed", values="score", sort=False).round(3).to_string())
    r.to_csv(CACHE / "v1_holdout.csv", index=False)


# ---------------------------------------------------------------- 3. 20종목

def universe(n=20, seed=0):
    """Dataset(symbols=20개) 로 평가 구간 피처를 다시 만들어 50종목 캐시와 비교한다."""
    b = load_b()
    syms = sorted(random.Random(seed).sample(sorted(b.symbol.unique()), n))
    f = CACHE / f"b_features_u{n}.parquet"
    if f.exists():
        u = pd.read_parquet(f)
    else:
        ds = Dataset(symbols=syms)
        days = [d for d in ds.days()
                if pd.Timestamp(FOLD_STARTS[0]) <= d.target < pd.Timestamp(TEST_START)]
        t = time.time()
        u = build_b_table(days, verbose=False)
        u["target"] = u.date.map({d.date: d.target for d in days})
        u.to_parquet(f, index=False)
        print(f"20종목 피처 {u.shape} {time.time() - t:.0f}s", flush=True)
    u["label"] = u.label.astype(int)
    print("종목:", syms)

    j = u.merge(b, on=["symbol", "date"], suffixes=("_20", "_50"))
    rows = []
    for c in B_FEATURES:
        a, z = j[f"{c}_20"], j[f"{c}_50"]
        ok = a.notna() & z.notna()
        if not ok.any():
            continue
        diff = (a - z)[ok].abs()
        if diff.max() < 1e-9:
            continue
        rows.append({"feature": c, "V1": c in B_FEATURES_V1,
                     "corr": a[ok].corr(z[ok]), "spearman": a[ok].corr(z[ok], method="spearman"),
                     "|차이| 중앙값 / 50종목 sd": diff.median() / z[ok].std(),
                     "|차이| 90% / sd": diff.quantile(0.9) / z[ok].std()})
    print("\n== 값이 달라진 피처 (나머지 피처는 완전히 같음) ==")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 같은 모델(50종목 캐시로 학습)로 20종목을 예측: 50종목 기준 피처 vs 20종목 기준 피처
    res = []
    for name, tr_m, te_m in folds(targets=b.target):
        te_lo, te_hi = b[te_m].target.min(), b[te_m].target.max()
        m, _ = train_lgb(b[tr_m], B_FEATURES_V1)
        part = j[(j.target_50 >= te_lo) & (j.target_50 <= te_hi)]
        for model, sfx in [("50종목 기준 피처", "_50"), ("20종목 기준 피처", "_20")]:
            x = part[[f"{c}{sfx}" for c in B_FEATURES_V1]].set_axis(B_FEATURES_V1, axis=1)
            res.append({"fold": name, "model": model,
                        **metrics(part.label_50, m.predict(x).argmax(1))})
    print("\n== 같은 20종목 · 같은 모델, 피처 계산 기준만 다름 (V1) ==")
    print(summary(res).to_string())
    pr = pd.DataFrame(res)
    pr.to_csv(CACHE / "v1_universe.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    {"diag": diag, "holdout": holdout, "universe": universe}[sys.argv[1]]()
