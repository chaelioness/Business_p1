"""10차: 미리 정한 소수 피처 조합 — 데이터로 고르지 않음. A 폴드 기준.

    uv run python work/b/exp/r10_prior.py

피처와 부호는 경제적 근거로 실험 전에 정함 (순서대로 하나씩 늘림, S1 ~ S6):
  ext_range_z  곱하기   −  시간외 출렁임 큰 갭은 덜 믿음
  new_pre_z    같은방향 +  장전 흐름이 갭과 같은 방향이면 더 믿음
  mkt_gap      더하기   +  시장 전체 갭 방향으로 보탬
  new_ext_liq  곱하기   +  시간외 봉이 많으면(유동성) 더 믿음
  earn_now     곱하기   +  실적일 갭은 정보가 있으니 더 믿음
  vol_5_20     같은방향 +  최근 변동성 커진 시기엔 더 믿음
점수 s = gz · exp(λ·Σ곱하기 부호·x̃ + λ·Σ같은방향 부호·x̃·sign(gz)) + (λ/2)·σ(gz)·Σ더하기 부호·x̃
(가) λ = 0.2 고정 (학습 없음)   (나) λ 하나만 train score 최대로   경계는 둘 다 A 방식.
(다) LightGBM 회귀 (목표 수익률÷vol20), 같은 피처 + gap_z, 방향 피처(gap_z, new_pre_z, mkt_gap) 단조 증가 제약, 시드 3개.
"""


import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from lab.folds import unseen_symbols  # noqa: E402
from common import BASE, OUT, SEEDS, VALID_DAYS, cut, engineered, fit_ab as ffit, fscore, load_table  # noqa: E402

PRIOR = [("ext_range_z", "곱하기", -1), ("new_pre_z", "같은방향", +1), ("mkt_gap", "더하기", +1),
         ("new_ext_liq", "곱하기", +1), ("earn_now", "곱하기", +1), ("vol_5_20", "같은방향", +1)]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def std_x(tr, va, c):
    med, sd = np.nanmedian(tr[c]), np.nanstd(tr[c])
    f = lambda z: np.nan_to_num(np.clip((z[c].values - med) / sd, -5, 5))  # noqa: E731
    return f(tr), f(va)


def combo(g, xs, spec, lam, sd):
    m = np.zeros(len(g))
    a = np.zeros(len(g))
    sg = np.sign(np.nan_to_num(g))
    for (c, form, s), x in zip(spec, xs):
        if form == "곱하기":
            m += s * x
        elif form == "같은방향":
            m += s * x * sg
        else:
            a += s * x
    return g * np.exp(lam * m) + lam / 2 * sd * a


def lgb_pred(tr, va, cols, mono, seed):
    import lightgbm as lgb
    p = {**BASE, "objective": "regression", "seed": seed, "monotone_constraints": mono,
         "monotone_constraints_method": "advanced"}
    p.pop("num_class")
    z = np.clip(np.nan_to_num(tr.ret_pct.values / 100 / tr.vol20.values), -6, 6)
    cd = np.sort(tr.date.unique())[-VALID_DAYS]
    a, v = tr.date.values < cd, tr.date.values >= cd
    m = lgb.train(p, lgb.Dataset(tr.loc[a, cols], z[a]), 2000, valid_sets=[lgb.Dataset(tr.loc[v, cols], z[v])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(p, lgb.Dataset(tr[cols], z), max(m.best_iteration, 20))
    s_tr, s_va = m.predict(tr[cols]), m.predict(va[cols])
    lo, hi = ffit(s_tr, tr.label.values)
    return cut(s_va, lo, hi)


def run():
    table, folds = load_table()
    for k, v in engineered(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    unseen = set(unseen_symbols())
    rows, lam_pick = [], {}

    def rec(name, fold, seed, y, p, u):
        rows.append({"model": name, "fold": fold, "seed": seed, "all": fscore(y, p), "unseen": fscore(y[u], p[u]),
                     "pred_big%": np.isin(p, [0, 4]).mean() * 100})

    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
        va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
        y_tr, y, u = tr.label.values, va.label.values, va.symbol.isin(unseen).values
        g_tr, g_va = tr.gap_z.values, va.gap_z.values
        sd = np.nanstd(g_tr)
        a, b = ffit(g_tr, y_tr)
        rec("gap_z 규칙", f.name, 0, y, cut(g_va, a, b), u)
        X = [std_x(tr, va, c) for c, _, _ in PRIOR]
        for k in range(1, len(PRIOR) + 1):
            spec, xs_tr, xs_va = PRIOR[:k], [x[0] for x in X[:k]], [x[1] for x in X[:k]]
            name = f"S{k} +" + PRIOR[k - 1][0]
            # (가) 고정
            st, sv = combo(g_tr, xs_tr, spec, 0.2, sd), combo(g_va, xs_va, spec, 0.2, sd)
            a, b = ffit(st, y_tr)
            rec(f"(가) 고정 λ=0.2 | {name}", f.name, 0, y, cut(sv, a, b), u)
            # (나) λ 하나 train
            best = None
            for lam in LAMS:
                st = combo(g_tr, xs_tr, spec, lam, sd)
                a, b = ffit(st, y_tr)
                s = fscore(y_tr, cut(st, a, b))
                if best is None or s > best[0]:
                    best = (s, lam, a, b)
            lam_pick[(name, f.name)] = best[1]
            rec(f"(나) λ train | {name}", f.name, 0, y, cut(combo(g_va, xs_va, spec, best[1], sd), best[2], best[3]), u)
        # (다) LightGBM 소수 피처
        cols = ["gap_z"] + [c for c, _, _ in PRIOR]
        mono = [1, 0, 1, 1, 0, 0, 0]
        for s in SEEDS:
            rec("(다) LightGBM 회귀 · 미리 정한 6개 + gap_z", f.name, s, y, lgb_pred(tr, va, cols, mono, s), u)
        print(f"  {f.name} 완료", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_prior.csv", index=False)
    m = r.groupby(["model", "fold"])[["all", "unseen", "pred_big%"]].mean()
    w = m["all"].unstack()
    base = w.loc["gap_z 규칙"]
    F = ["fold1", "fold2", "fold3", "fold4"]
    w["4폴드 평균"] = w[F].mean(axis=1)
    w["Δ평균"] = w["4폴드 평균"] - base.mean()
    w["상승폴드"] = (w[F].sub(base) > 1e-12).sum(axis=1)
    w = w.join(m.groupby(level=0).mean()[["unseen", "pred_big%"]])
    order = ["gap_z 규칙"] + [i for i in w.index if i.startswith("(가)")] + [i for i in w.index if i.startswith("(나)")] + \
            [i for i in w.index if i.startswith("(다)")]
    pd.set_option("display.width", 250)
    print("\n== 미리 정한 조합 ==")
    print(w.loc[order].round(4).to_string())
    lp = pd.Series(lam_pick).unstack()
    print("\n(나) 에서 train 이 고른 λ:")
    print(lp.to_string())
    w.to_csv(OUT / "exp_prior_summary.csv")


if __name__ == "__main__":
    run()
