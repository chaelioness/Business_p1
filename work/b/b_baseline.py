"""B 피처 베이스라인 비교 (walk-forward).

    uv run python work/b/b_baseline.py

(a) 전부 보합  (b) gap_z 규칙  (c) sample_model RandomForest(원래 8개 피처)
(d) LightGBM 다중분류, B 피처 전체  (e) (d) + 비용 최소 디코딩(비교용)

점수는 src.data.score 를 그대로 쓴다. 폴드는 folds.py 한 곳에서 정한다.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "example"))

from src import FLAT, WEIGHT, Dataset, score  # noqa: E402
from features_b import B_FEATURES  # noqa: E402
from folds import TEST_START, folds  # noqa: E402

CACHE = HERE / "cache"
SEED = 0
VALID_DAYS = 40            # 조기 종료용: 학습 구간의 마지막 40 기준일
LGB_PARAMS = dict(objective="multiclass", num_class=5, learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=SEED, verbose=-1, num_threads=4)
METRICS = ["score", "accuracy", "direction", "big_recall", "big_prec"]


def load_b():
    return pd.read_parquet(CACHE / "b_features.parquet")


def load_sample():
    """sample_model 의 8개 피처. 없으면 한 번 만들어 캐시한다."""
    f = CACHE / "sample_features.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from sample_model import build_table
    t = time.time()
    days = [d for d in Dataset().days() if d.target < pd.Timestamp(TEST_START)]
    tab = build_table(days, with_label=False, verbose=False)
    tab.to_parquet(f, index=False)
    print(f"sample 피처 캐시 {tab.shape} {time.time() - t:.0f}s")
    return tab


# ---------------------------------------------------------------- 모델

def rule_gap(tr, te):
    """gap_z 하나로 5등급. 학습 구간 label 비율에 맞춰 분위 임계값을 정한다.
    방향(부호)도 학습 구간 Spearman 으로 정한다."""
    s = np.sign(tr.gap_z.corr(tr.ret_pct, method="spearman")) or 1.0
    x_tr, x_te = s * tr.gap_z.fillna(0), s * te.gap_z.fillna(0)
    cum = np.cumsum(tr.label.value_counts(normalize=True).sort_index().values)[:-1]
    cuts = np.quantile(x_tr, cum)
    return np.searchsorted(cuts, x_te.values, side="right")


def rf_sample(tr, te):
    from sklearn.ensemble import RandomForestClassifier
    from sample_model import FEATURES
    tr = tr.dropna(subset=FEATURES)
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                 n_jobs=4, random_state=SEED)
    clf.fit(tr[FEATURES], tr.label.astype(int))
    ok = te[FEATURES].notna().all(axis=1).values
    pred = np.full(len(te), FLAT)
    pred[ok] = clf.predict(te.loc[ok, FEATURES])
    return pred


def lgb_proba(tr, te, feats):
    """학습 구간 마지막 VALID_DAYS 기준일로 조기 종료 반복수를 정한 뒤 학습 전체로 다시 맞춘다."""
    import lightgbm as lgb
    dates = np.sort(tr.date.unique())
    cut = dates[-VALID_DAYS]
    fit, val = tr[tr.date < cut], tr[tr.date >= cut]
    m = lgb.train(LGB_PARAMS, lgb.Dataset(fit[feats], fit.label.astype(int)),
                  num_boost_round=2000,
                  valid_sets=[lgb.Dataset(val[feats], val.label.astype(int))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(m.best_iteration, 20)
    m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[feats], tr.label.astype(int)), num_boost_round=n)
    return m.predict(te[feats]), n, m


def cost_decode(p):
    """argmin_j Σ_i p_i · WEIGHT[i][j]."""
    return np.argmin(p @ WEIGHT, axis=1)


# ---------------------------------------------------------------- 실행

def main():
    b = load_b()
    b["label"] = b.label.astype(int)
    s = load_sample()
    s = s.merge(b[["symbol", "date", "target", "label", "ret_pct"]], on=["symbol", "date"])

    rows, dist, pooled = [], {}, {}
    for name, tr_m, te_m in folds(targets=b.target):
        tr, te = b[tr_m], b[te_m]
        s_tr = s[s.target <= tr.target.max()]           # 같은 학습 구간 (embargo 포함)
        s_te = s[(s.target >= te.target.min()) & (s.target <= te.target.max())]
        p, n_iter, _ = lgb_proba(tr, te, B_FEATURES)
        preds = {
            "(a) 전부 보합": (te.label, np.full(len(te), FLAT)),
            "(b) gap_z 규칙": (te.label, rule_gap(tr, te)),
            "(c) RF sample 8": (s_te.label, rf_sample(s_tr, s_te)),
            "(d) LGB B 전체": (te.label, p.argmax(1)),
            "(e) (d)+비용최소": (te.label, cost_decode(p)),
        }
        for model, (y, pr) in preds.items():
            r = score(y, pr)
            rows.append({"fold": name, "model": model, **{k: r[k] for k in METRICS}, "n": r["n"]})
            dist.setdefault(model, []).append(np.bincount(pr, minlength=5))
            pooled.setdefault(model, []).append((np.asarray(y, dtype=int), np.asarray(pr, dtype=int)))
        print(f"{name}: 학습 {tr_m.sum():,} (대상일 ~{tr.target.max():%Y-%m-%d}) "
              f"평가 {te_m.sum():,} ({te.target.min():%Y-%m-%d}~)  LGB 반복 {n_iter}", flush=True)

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n폴드별")
    print(r.pivot(index="model", columns="fold", values="score").round(4).to_string())
    print("\n합쳐서 채점 (전체 평가 행으로 한 번 계산, 주 지표) + 폴드 평균 score")
    pool = {}
    for model, parts in pooled.items():
        sc = score(np.concatenate([y for y, _ in parts]), np.concatenate([p for _, p in parts]))
        pool[model] = {k: sc[k] for k in METRICS}
    pool = pd.DataFrame(pool).T
    pool["폴드평균"] = r.groupby("model").score.mean()
    print(pool.round(4).to_string())
    pool.to_csv(HERE / "cache" / "baseline_pooled.csv")
    print("\n예측 label 분포 (전체 평가, %)")
    d = pd.DataFrame({k: np.sum(v, axis=0) for k, v in dist.items()}).T
    print((d.div(d.sum(1), axis=0) * 100).round(1).to_string())
    print("\n실제 label 분포 (평가, %)")
    all_te = np.concatenate([b[m].label.values for _, _, m in folds(targets=b.target)])
    print((np.bincount(all_te, minlength=5) / len(all_te) * 100).round(1))
    r.to_csv(HERE / "cache" / "baseline_results.csv", index=False)


if __name__ == "__main__":
    main()
