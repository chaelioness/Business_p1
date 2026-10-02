"""gap_z 규칙 다음 단계 — 분모 바꾸기, 규칙과 1단계 모델 섞기. A 폴드 기준.

    uv run python work/b/exp/r05_gapz2.py

1) 분모: gap ÷ (vol5 | vol20 | vol60 | atr14 | √(vol20·vol60)), gap ÷ vol20^p (p=0 은 원래 갭),
   시장 갭을 뺀 gap_rel_z. 경계는 모두 A 방식(분위 후보 중 train score 최대).
2) 섞기: c = w·순위(갭 피처) + (1−w)·순위(1단계 모델 기대 등급). 순위는 train 분포 기준.
   c 를 "갭 규칙이 train 에서 찍은 등급 비율"대로 자름. w 는 fold1~3 으로 고름. 시드 3개.
"""


import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from features_b import B_FEATURES_V1  # noqa: E402
from common import OUT, SEEDS, cut, fit_ab, load_table, train_predict  # noqa: E402

WS = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]
COLS = ["all", "unseen", "big_recall", "big_prec", "pred_big%"]


def variants(t):
    g = t.gap
    return {
        "gap (p=0)": g,
        "gap ÷ vol20^0.5": g / np.sqrt(t.vol20),
        "gap ÷ vol20 (gap_z)": g / t.vol20,
        "gap ÷ vol20^1.5": g / t.vol20 ** 1.5,
        "gap ÷ vol5": g / t.vol5,
        "gap ÷ vol60": g / t.vol60,
        "gap ÷ atr14": g / t.atr14,
        "gap ÷ √(vol20·vol60)": g / np.sqrt(t.vol20 * t.vol60),
        "gap_rel_z (시장 갭 뺌)": t.gap_rel_z,
    }


def ecdf(ref, x):
    """train 분포 기준 백분위."""
    ref = np.sort(ref[~np.isnan(ref)])
    return np.searchsorted(ref, x, side="right") / len(ref)


def summarize(r, keys):
    m = r.groupby(keys + ["fold"])[COLS].mean()
    w = m["all"].unstack("fold")
    w["fold1~3"] = w[["fold1", "fold2", "fold3"]].mean(axis=1)
    w["4폴드 평균"] = w[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    return w.join(m.groupby(level=list(range(len(keys)))).mean()[COLS[1:]])


def run():
    table, folds = load_table()
    unseen = set(unseen_symbols())
    for k, v in variants(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    names = list(variants(table))

    def rec(rows, extra, y, pred, u):
        sc = score(y, pred)
        rows.append({**extra, "all": sc["score"], "unseen": score(y[u], pred[u])["score"], "big_recall": sc["big_recall"],
                     "big_prec": sc["big_prec"], "pred_big%": np.isin(pred, [0, 4]).mean() * 100})

    # 1) 분모
    rows = []
    ctx = {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        y_tr, y, u = tr.label.values, va.label.values, va.symbol.isin(unseen).values
        for n in names:
            a, b = fit_ab(tr[n].values, y_tr)
            rec(rows, {"fold": f.name, "피처": n}, y, cut(va[n].values, a, b), u)
            ctx[(f.name, n)] = (a, b)
    w1 = summarize(pd.DataFrame(rows), ["피처"]).sort_values("fold1~3", ascending=False)
    pd.set_option("display.width", 250)
    print("== 1) 갭 규칙의 분모 (fold1~3 순) ==")
    print(w1.round(4).to_string())
    best = w1.index[0]
    print(f"\nfold1~3 으로 고른 갭 피처: {best}")
    w1.to_csv(OUT / "exp_gapz2_denom.csv")

    # 2) 섞기: 고른 갭 피처 + 1단계 모델
    rows = []
    for s in SEEDS:
        tp = train_predict(table, folds, B_FEATURES_V1, s)
        for f in folds:
            p_tr, p_va, y_tr, va = tp[f.name]
            tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
            y, u = va.label.values, va.symbol.isin(unseen).values
            a, b = ctx[(f.name, best)]
            x_tr, x_va = tr[best].values, va[best].values
            pi_t = np.bincount(cut(x_tr, a, b), minlength=5) / len(x_tr)  # 규칙이 train 에서 찍은 비율
            e_tr, e_va = p_tr @ np.arange(5), p_va @ np.arange(5)
            gx_tr, gx_va = ecdf(x_tr, np.nan_to_num(x_tr)), ecdf(x_tr, np.nan_to_num(x_va))
            ge_tr, ge_va = ecdf(e_tr, e_tr), ecdf(e_tr, e_va)
            rec(rows, {"seed": s, "fold": f.name, "방식": "규칙 그대로", "w": np.nan}, y, cut(x_va, a, b), u)
            for w in WS:
                c_tr, c_va = w * gx_tr + (1 - w) * ge_tr, w * gx_va + (1 - w) * ge_va
                th = np.quantile(c_tr, np.cumsum(pi_t)[:-1])
                rec(rows, {"seed": s, "fold": f.name, "방식": "섞기 (규칙 비율로 자름)", "w": w}, y,
                    np.searchsorted(th, c_va, side="right"), u)
        print(f"  seed {s} 완료", flush=True)
    r = pd.DataFrame(rows)
    w2 = summarize(r, ["방식", "w"])
    print("\n== 2) 갭 규칙 × 1단계 모델 섞기 (w = 갭 비중, 시드 3개 평균) ==")
    print(w2.round(4).to_string())
    mix = w2.loc["섞기 (규칙 비율로 자름)"]
    bw = mix["fold1~3"].idxmax()
    print(f"\nfold1~3 으로 고른 w = {bw} → fold4 {mix.loc[bw, 'fold4']:.4f}, 4폴드 {mix.loc[bw, '4폴드 평균']:.4f}")
    per_seed = r[(r["방식"] == "섞기 (규칙 비율로 자름)") & (r.w == bw)].groupby("seed")["all"].mean()
    print(f"고른 w 의 시드 간 표준편차 {per_seed.std():.4f}")
    d = r[(r["방식"] == "섞기 (규칙 비율로 자름)") & (r.w == bw)].groupby("fold")["all"].mean() - \
        r[r["방식"] == "규칙 그대로"].groupby("fold")["all"].mean()
    print("규칙 대비 폴드별 차이:", d.round(4).to_dict())
    w2.to_csv(OUT / "exp_gapz2_blend.csv")


if __name__ == "__main__":
    run()
