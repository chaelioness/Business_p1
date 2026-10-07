"""찍는 공격성(급등락·보합을 몇 % 찍을지) 실험 — A 폴드 기준.

    uv run python work/b/exp/r02_decode.py grid     # 디코딩 격자, 갭 규칙과 비교
    uv run python work/b/exp/r02_decode.py feats    # 고른 디코딩으로 크기 후보 다시 비교

배경: A 의 EDA 에서 급등락을 실제(13.6%)보다 많이(약 34%) 찍을 때 score 가 가장 높았음.
B 모델은 train 비율(13.6%)대로 찍어서 같은 폴드에서 갭 규칙(score 최대 경계)보다 낮았음.

디코딩 (모델 확률 p 로부터)
  E  : 기대 등급 Σ i·p_i 를 목표 비율의 분위로 자름
  DS : 급등락 여부는 P(0)+P(4) 상위 K, 방향은 P(3)+P(4)−P(0)−P(1) 로 나눔
목표 비율: 급등락 K, 보합 F (train 값 또는 지정), 나머지는 train 의 하락:상승 비율로 나눔.
고르는 법: fold1~3 평균 score 로 (방식, K, F) 를 고르고, fold4(2026-03~05, EDA 와 안 겹침)로 확인.
모델 학습은 폴드·시드마다 한 번, val 피처는 guard 아래에서 만든 표(cache/b_v2.parquet)에서 읽음.
"""

import sys

import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from features_b import B_FEATURES_V1  # noqa: E402
from common import OUT, SEEDS, load_table, train_predict  # noqa: E402

KS = ["train", 0.20, 0.27, 0.34, 0.40]
FS = ["train", 0.20, 0.10]


def target_props(pi, K, F):
    K = pi[0] + pi[4] if K == "train" else K
    F = pi[2] if F == "train" else F
    rest = 1 - K - F
    return np.array([K * pi[0] / (pi[0] + pi[4]), rest * pi[1] / (pi[1] + pi[3]), F,
                     rest * pi[3] / (pi[1] + pi[3]), K * pi[4] / (pi[0] + pi[4])])


def decode(p_tr, p_va, pi_t, kind):
    if kind == "E":
        e_tr, e_va = p_tr @ np.arange(5), p_va @ np.arange(5)
        return np.searchsorted(np.quantile(e_tr, np.cumsum(pi_t)[:-1]), e_va, side="right")
    d = lambda p: p[:, 3] + p[:, 4] - p[:, 0] - p[:, 1]  # noqa: E731
    m = lambda p: p[:, 0] + p[:, 4]  # noqa: E731
    t_big = np.quantile(m(p_tr), 1 - (pi_t[0] + pi_t[4]))
    mid = pi_t[1:4] / pi_t[1:4].sum()
    d_mid = d(p_tr)[m(p_tr) <= t_big]
    q1, q2 = np.quantile(d_mid, [mid[0], mid[0] + mid[1]])
    dv, mv = d(p_va), m(p_va)
    lab = np.where(dv < q1, 1, np.where(dv < q2, 2, 3))
    return np.where(mv > t_big, np.where(dv > 0, 4, 0), lab)


def scores(y, pred, va, unseen):
    s = score(y, pred)
    u = va.symbol.isin(unseen).values
    return {"all": s["score"], "unseen": score(y[u], pred[u])["score"], "big_recall": s["big_recall"],
            "big_prec": s["big_prec"], "pred_big%": np.isin(pred, [0, 4]).mean() * 100, "pred_flat%": (pred == 2).mean() * 100}


def gap_rule(table, folds):
    """A 방식 갭 규칙: |gap| 분위 후보 중 train score 가 최대인 경계 a<b (gap 은 전일 종가 기준)."""
    unseen = set(unseen_symbols())
    rows = []
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].dropna(subset=["gap"])
        va = table[table.date.isin({d.date for d in f.val})]
        g, y = tr.gap.values, tr.label.values
        qs = np.quantile(np.abs(g), [.5, .6, .7, .8, .85, .9, .95, .975, .99])

        def cut(x, a, b):
            o = np.full(len(x), 2)
            o[x >= a], o[x >= b], o[x <= -a], o[x <= -b] = 3, 4, 1, 0
            o[np.isnan(x)] = 2
            return o
        best = max(((score(y, cut(g, a, b))["score"], a, b) for i, a in enumerate(qs) for b in qs[i + 1:]))
        pred = cut(va.gap.values, best[1], best[2])
        rows.append({"fold": f.name, **scores(va.label.values, pred, va, unseen)})
    return pd.DataFrame(rows).set_index("fold")


def grid():
    table, folds = load_table()
    unseen = set(unseen_symbols())
    rows = []
    for s in SEEDS:
        tp = train_predict(table, folds, B_FEATURES_V1, s)
        for fold, (p_tr, p_va, y_tr, va) in tp.items():
            pi = np.bincount(y_tr, minlength=5) / len(y_tr)
            for kind in ["E", "DS"]:
                for K in KS:
                    for F in FS:
                        pred = decode(p_tr, p_va, target_props(pi, K, F), kind)
                        rows.append({"seed": s, "fold": fold, "kind": kind, "K": str(K), "F": str(F),
                                     **scores(va.label.values, pred, va, unseen)})
        print(f"  seed {s} 완료", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_decode_grid.csv", index=False)
    m = r.groupby(["kind", "K", "F", "fold"])[["all", "unseen", "big_recall", "big_prec", "pred_big%", "pred_flat%"]].mean()
    wide = m["all"].unstack("fold")
    wide["fold1~3"] = wide[["fold1", "fold2", "fold3"]].mean(axis=1)
    wide["4폴드 평균"] = wide[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    extra = m.groupby(level=[0, 1, 2]).mean()[["unseen", "big_recall", "big_prec", "pred_big%", "pred_flat%"]]
    wide = wide.join(extra).sort_values("fold1~3", ascending=False)
    pd.set_option("display.width", 250)
    print("\n== 디코딩 격자 (시드 3개 평균, fold1~3 평균 순) ==")
    print(wide.round(4).head(12).to_string())
    print("\n기존 (E, train, train):")
    print(wide.loc[("E", "train", "train")].round(4).to_string())
    best = wide.index[0]
    print(f"\nfold1~3 으로 고른 디코딩: {best} → fold4 {wide.loc[best, 'fold4']:.4f}")
    g = gap_rule(table, folds)
    print("\n== A 방식 갭 규칙 (같은 폴드, |gap| 경계를 train score 최대로) ==")
    print(g.round(4).to_string())
    print("평균", g["all"].mean().round(4), "| fold1~3", g.loc[["fold1", "fold2", "fold3"], "all"].mean().round(4))
    wide.to_csv(OUT / "exp_decode_summary.csv")


def feats():
    """고른 디코딩으로 크기 후보를 다시 비교."""
    w = pd.read_csv(OUT / "exp_decode_summary.csv", index_col=[0, 1, 2])
    kind, K, F = w.index[0]
    K = K if K == "train" else float(K)
    F = F if F == "train" else float(F)
    print(f"디코딩: {kind}, 급등락 {K}, 보합 {F}")
    table, folds = load_table()
    unseen = set(unseen_symbols())
    variants = {"V1": B_FEATURES_V1, **{f"+{c}": B_FEATURES_V1 + [c] for c in
                ["earn_hist_absr", "mkt_absgap", "gap_disp", "big_yday", "earn_timing", "ext_pos", "gap_rank"]}}
    rows = []
    for name, fs in variants.items():
        for s in SEEDS:
            for fold, (p_tr, p_va, y_tr, va) in train_predict(table, folds, fs, s).items():
                pi = np.bincount(y_tr, minlength=5) / len(y_tr)
                pred = decode(p_tr, p_va, target_props(pi, K, F), kind)
                rows.append({"variant": name, "seed": s, "fold": fold, **scores(va.label.values, pred, va, unseen)})
        print(f"  {name} 완료", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_decode_feats.csv", index=False)
    m = r.groupby(["variant", "fold"])[["all", "unseen"]].mean()
    a = m["all"].unstack()
    d = a.sub(a.loc["V1"], axis=1)
    out = pd.DataFrame({"평균": a.mean(axis=1), "Δ평균": d.mean(axis=1), "Δfold4": d["fold4"], "상승 폴드": (d > 0).sum(axis=1),
                        "unseen": m["unseen"].unstack().mean(axis=1)}).join(a)
    out["채택"] = (out["Δ평균"] >= 0.005) & (out["상승 폴드"] >= 3) & (out["Δfold4"] >= 0)
    sd = r[r.variant == "V1"].groupby("seed")["all"].mean().std()
    print(f"\nV1 시드 간 표준편차 {sd:.4f}")
    print(out.round(4).to_string())


if __name__ == "__main__":
    {"grid": grid, "feats": feats}[sys.argv[1]]()
