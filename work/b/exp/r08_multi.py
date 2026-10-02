"""gap_z 규칙에 피처 여러 개 — 고르는 것은 모두 train 안에서만. A 폴드 기준.

    uv run python work/b/exp/r08_multi.py

M1 차례로 추가: 7차의 225개 조합(피처 × 더하기/곱하기/같은방향)을 하나씩 더함. train 안 내부 검증
   (마지막 120일을 60일씩 두 구간, 각 구간 앞까지로 학습)에서 평균이 0.003 이상 오를 때만 추가, 최대 6개.
M2 Ridge: gap_z + 피처 75개 × (x̃, x̃·sign(gz), x̃·gz) 를 z = 다음날 수익률 ÷ vol20 에 맞춤.
   예측값에 A 방식 경계. 규제 강도는 내부 검증으로 고름.
M3 LightGBM 회귀: 같은 목표, 피처 75개 + gap_z (gap_z 는 단조 증가 제약). 예측값에 A 방식 경계. 시드 3개.
비교: gap_z 규칙 (4폴드 0.438, fold4 0.496).
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import BASE, BETAS, OUT, SEEDS, SKIP, VALID_DAYS, apply, cut, engineered, fit_ab as ffit, fit_eval, fscore, inner_splits, load_table  # noqa: E402

INNER = 60          # 내부 검증 구간 길이 (거래일)
MIN_GAIN = 0.003
MAX_STEPS = 6
ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10]


def stdz(tr, others, feats):
    med, std = tr[feats].median(), tr[feats].std().replace(0, np.nan)
    f = lambda z: np.nan_to_num(np.clip(((z[feats] - med) / std).values, -5, 5))  # noqa: E731
    return f(tr), [f(z) for z in others]


# ------------------------------------------------------------------ M1
def m1_candidate(c, form, j, g_cur, X, y, splits, sd):
    best = (-9, None)
    for b in BETAS[form]:
        if b == 0:
            continue
        s = apply(form, b, g_cur, X[:, j], sd)
        v = np.mean([fit_eval(s[t], y[t], s[w], y[w]) for t, w in splits])
        if v > best[0]:
            best = (v, b)
    return c, form, best[1], best[0]


def m1(tr, va, feats, y_tr):
    X_tr, (X_va,) = stdz(tr, [va], feats)
    g_tr, g_va = tr.gap_z.values.copy(), va.gap_z.values.copy()
    sd = np.nanstd(g_tr)
    splits = inner_splits(tr.date.values)
    cur = np.mean([fit_eval(g_tr[t], y_tr[t], g_tr[w], y_tr[w]) for t, w in splits])
    chosen = []
    for _ in range(MAX_STEPS):
        res = Parallel(n_jobs=-1)(delayed(m1_candidate)(c, form, j, g_tr, X_tr, y_tr, splits, sd)
                                  for j, c in enumerate(feats) for form in BETAS)
        c, form, b, v = max(res, key=lambda r: r[3])
        if v - cur < MIN_GAIN:
            break
        j = feats.index(c)
        g_tr, g_va = apply(form, b, g_tr, X_tr[:, j], sd), apply(form, b, g_va, X_va[:, j], sd)
        chosen.append(f"{c}·{form}({b:g}) 내부 {v - cur:+.4f}")
        cur = v
    a, bb = ffit(g_tr, y_tr)
    return cut(g_va, a, bb), chosen


# ------------------------------------------------------------------ M2
def design(X, gz):
    g = np.nan_to_num(gz)
    sg = np.sign(g)[:, None]
    return np.column_stack([g, X, X * sg, X * np.clip(g, -6, 6)[:, None]])


def ridge(A, z, alpha):
    mu, sdv = A.mean(0), A.std(0) + 1e-9
    An = (A - mu) / sdv
    n, p = An.shape
    w = np.linalg.solve(An.T @ An + alpha * n * np.eye(p), An.T @ (z - z.mean()))
    return lambda B: ((B - mu) / sdv) @ w + z.mean(), w / sdv


def m2(tr, va, feats, y_tr):
    X_tr, (X_va,) = stdz(tr, [va], feats)
    A_tr, A_va = design(X_tr, tr.gap_z.values), design(X_va, va.gap_z.values)
    z = np.clip(np.nan_to_num(tr.ret_pct.values / 100 / tr.vol20.values), -6, 6)
    splits = inner_splits(tr.date.values)
    inner = {al: np.mean([fit_eval(ridge(A_tr[t], z[t], al)[0](A_tr[t]), y_tr[t], ridge(A_tr[t], z[t], al)[0](A_tr[w]), y_tr[w])
                          for t, w in splits]) for al in ALPHAS}
    al = max(inner, key=inner.get)
    f, coef = ridge(A_tr, z, al)
    s_tr, s_va = f(A_tr), f(A_va)
    a, b = ffit(s_tr, y_tr)
    names = ["gap_z"] + feats + [c + "·sign" for c in feats] + [c + "·gz" for c in feats]
    sc = pd.Series(coef * A_tr.std(0), index=names)
    top = sc.abs().sort_values(ascending=False).head(8)
    return cut(s_va, a, b), al, {k: round(sc[k], 3) for k in top.index}


# ------------------------------------------------------------------ M3
def m3(tr, va, feats, y_tr, seed):
    import lightgbm as lgb
    cols = ["gap_z"] + [c for c in feats if c != "gap_z"]
    p = {**BASE, "objective": "regression", "seed": seed,
         "monotone_constraints": [1] + [0] * (len(cols) - 1), "monotone_constraints_method": "advanced"}
    p.pop("num_class")
    z = np.clip(np.nan_to_num(tr.ret_pct.values / 100 / tr.vol20.values), -6, 6)
    cutd = np.sort(tr.date.unique())[-VALID_DAYS]
    a, v = tr.date.values < cutd, tr.date.values >= cutd
    m = lgb.train(p, lgb.Dataset(tr.loc[a, cols], z[a]), 2000, valid_sets=[lgb.Dataset(tr.loc[v, cols], z[v])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(p, lgb.Dataset(tr[cols], z), max(m.best_iteration, 20))
    s_tr, s_va = m.predict(tr[cols]), m.predict(va[cols])
    lo, hi = ffit(s_tr, y_tr)
    return cut(s_va, lo, hi)


def run():
    table, folds = load_table()
    for k, v in engineered(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in table.columns if c not in SKIP and pd.api.types.is_numeric_dtype(table[c])]
    unseen = set(unseen_symbols())

    # fscore 확인
    t = table.dropna(subset=["gap_z"]).head(5000)
    pr = cut(t.gap_z.values, 0.5, 1.5)
    assert abs(fscore(t.label.values, pr) - score(t.label.values, pr)["score"]) < 1e-9

    rows, notes = [], []

    def rec(name, fold, seed, y, pred, u):
        rows.append({"model": name, "fold": fold, "seed": seed, "all": fscore(y, pred), "unseen": fscore(y[u], pred[u]),
                     "pred_big%": np.isin(pred, [0, 4]).mean() * 100})

    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
        va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
        y_tr, y, u = tr.label.values, va.label.values, va.symbol.isin(unseen).values
        a, b = ffit(tr.gap_z.values, y_tr)
        rec("gap_z 규칙", f.name, 0, y, cut(va.gap_z.values, a, b), u)

        pred, chosen = m1(tr, va, feats, y_tr)
        rec("M1 차례로 추가", f.name, 0, y, pred, u)
        notes.append(f"[{f.name}] M1 고른 것: {chosen or '없음 (gap_z 그대로)'}")

        pred, al, top = m2(tr, va, feats, y_tr)
        rec("M2 Ridge", f.name, 0, y, pred, u)
        notes.append(f"[{f.name}] M2 규제 {al}, 큰 계수: {top}")

        for s in SEEDS:
            rec("M3 LightGBM 회귀 (gap_z 단조)", f.name, s, y, m3(tr, va, feats, y_tr, s), u)
        print(f"  {f.name} 완료", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_multi.csv", index=False)
    m = r.groupby(["model", "fold"])[["all", "unseen", "pred_big%"]].mean()
    w = m["all"].unstack()
    w["fold1~3"] = w[["fold1", "fold2", "fold3"]].mean(axis=1)
    w["4폴드 평균"] = w[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    w = w.join(m.groupby(level=0).mean()[["unseen", "pred_big%"]])
    d = w[["fold1", "fold2", "fold3", "fold4"]].sub(w.loc["gap_z 규칙", ["fold1", "fold2", "fold3", "fold4"]])
    w["상승폴드"] = (d > 0).sum(axis=1)
    pd.set_option("display.width", 250)
    print("\n== 여러 피처 (고르기는 train 안에서만) ==")
    print(w.round(4).to_string())
    sd = r[r.model.str.startswith("M3")].groupby("seed")["all"].mean().std()
    print(f"\nM3 시드 간 표준편차 {sd:.4f}\n")
    print("\n".join(notes))


if __name__ == "__main__":
    run()
