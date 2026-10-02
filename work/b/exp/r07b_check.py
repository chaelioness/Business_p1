"""exp_gapfeat_all 통과 조합 점검.

    uv run python work/b/exp/r07b_check.py

1) 우연 기준선: 모든 피처 값을 행 사이에서 무작위로 섞어(관계 끊음) 같은 절차를 돌리고, 채택 기준을 몇 개가 통과하는지.
   2회 반복 (225 × 2 = 450 조합).
2) 통과한 셋을 함께: β 는 폴드마다 train score 최대로 하나씩 차례로 고름.
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)
from common import BETAS, SKIP, apply, cut, engineered, fit_ab, judge, load_table, one, prep  # noqa: E402
from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402

PASSED = [("dow", "더하기"), ("vol_5_20", "같은방향"), ("ext_range_z", "곱하기")]


def run():
    table, folds = load_table()
    for k, v in engineered(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in table.columns if c not in SKIP and pd.api.types.is_numeric_dtype(table[c])]
    unseen = set(unseen_symbols())

    # 1) 우연 기준선
    rng = np.random.default_rng(0)
    passes = []
    for rep in range(2):
        sh = table.copy()
        for c in feats:
            sh[c] = rng.permutation(sh[c].values)
        data = prep(sh, folds, feats, unseen)
        fs = [c for c in feats if all(c in d["x"] for d in data.values())]
        res = Parallel(n_jobs=-1)(delayed(one)(c, form, data) for c in fs for form in BETAS)
        ok, da = judge(pd.DataFrame([row for p in res for row in p]))
        passes.append(int(ok.sum()))
        print(f"섞은 데이터 {rep + 1}회차: 채택 기준 통과 {ok.sum()}개 / {len(ok)}개, "
              f"Δ평균 최대 {da.mean(axis=1).max():.4f}, 통과: {list(ok[ok].index)}", flush=True)

    # 2) 통과한 셋 함께
    data = prep(table, folds, feats, unseen)
    rows = []
    for fn, d in data.items():
        st, sv = d["gz_tr"].copy(), d["gz_va"].copy()
        sd = d["sd"]
        lo, hi = fit_ab(st, d["y_tr"])
        rows.append({"fold": fn, "단계": "gap_z 규칙", "all": score(d["y"], cut(sv, lo, hi))["score"],
                     "unseen": score(d["y"][d["u"]], cut(sv, lo, hi)[d["u"]])["score"]})
        added = []
        for c, form in PASSED:
            xt, xv = d["x"][c]
            best = None
            for b in BETAS[form]:
                t2 = apply(form, b, st, xt, sd)
                lo, hi = fit_ab(t2, d["y_tr"])
                trs = score(d["y_tr"], cut(t2, lo, hi))["score"]
                if best is None or trs > best[0]:
                    best = (trs, b)
            st, sv = apply(form, best[1], st, xt, sd), apply(form, best[1], sv, xv, sd)
            added.append(f"{c}({best[1]:g})")
            lo, hi = fit_ab(st, d["y_tr"])
            pv = cut(sv, lo, hi)
            rows.append({"fold": fn, "단계": " + ".join(a.split("(")[0] for a in added), "β": " ".join(added),
                         "all": score(d["y"], pv)["score"], "unseen": score(d["y"][d["u"]], pv[d["u"]])["score"]})
    r = pd.DataFrame(rows)
    w = r.pivot_table(index="단계", columns="fold", values="all", sort=False)
    w["4폴드 평균"] = w.mean(axis=1)
    w["unseen"] = r.groupby("단계", sort=False)["unseen"].mean()
    pd.set_option("display.width", 250)
    print("\n== 통과한 셋 차례로 더하기 ==")
    print(w.round(4).to_string())
    print("\n폴드별 β:")
    print(r.dropna(subset=["β"]).pivot_table(index="단계", columns="fold", values="β", aggfunc="first", sort=False).to_string())


if __name__ == "__main__":
    run()
