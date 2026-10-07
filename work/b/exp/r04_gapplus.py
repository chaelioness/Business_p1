"""갭 규칙 + α — 갭 방향은 그대로 두고 경계만 바꾸는 세 가지. A 폴드 기준.

    uv run python work/b/exp/r04_gapplus.py

R0  갭 규칙: |gap| 분위 후보 중 train score 최대인 경계 a<b (A 방식)
R1a gap_z 규칙: 같은 방법을 gap_z(갭 ÷ vol20)로
R1b 변동성 구간별 규칙: train 의 vol20 3분위 구간마다 gap 경계를 따로
R2  내리기: R0 이 급등락으로 찍은 것 중 크기 모델 P(급등락)이 낮은 것(train 기준 하위 q)을 상승/하락으로 한 칸 내림
R3  실적일 경계: earn_now=1 인 날은 경계에 s 를 곱함 (낮추면 급등락을 더 쉽게 찍음)
조합: 위에서 fold1~3 으로 고른 것들을 함께.
q, s 는 fold1~3 평균으로 고르고 fold4 로 확인. R2 는 크기 모델 때문에 시드 3개 평균.
"""


import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import OUT, SEEDS, cut, fit_ab, lgb_binary, load_table  # noqa: E402
from r03_twostage import MAG  # noqa: E402

DEMOTE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EARN_S = [1.0, 0.75, 0.5, 0.25]


def cut_scaled(x, a, b, s_mask, s):
    """s_mask 인 행은 경계에 s 를 곱함."""
    o = cut(x, a, b)
    if s != 1.0 and s_mask.any():
        o[s_mask] = cut(x[s_mask], a * s, b * s)
    return o


def demote(pred, pb, thr):
    """급등락 예측 중 P(급등락) < thr 를 한 칸 안쪽으로."""
    o = pred.copy()
    low = pb < thr
    o[(pred == 4) & low] = 3
    o[(pred == 0) & low] = 1
    return o


def run():
    table, folds = load_table()
    unseen = set(unseen_symbols())
    rows = []

    def rec(seed, fold, name, param, y, pred, u):
        sc = score(y, pred)
        rows.append({"seed": seed, "fold": fold, "model": name, "param": param, "all": sc["score"],
                     "unseen": score(y[u], pred[u])["score"], "big_recall": sc["big_recall"], "big_prec": sc["big_prec"],
                     "pred_big%": np.isin(pred, [0, 4]).mean() * 100})

    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
        va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
        y_tr, y = tr.label.values, va.label.values
        u = va.symbol.isin(unseen).values
        g_tr, g = tr.gap.values, va.gap.values

        # R0
        a, b = fit_ab(g_tr, y_tr)
        r0 = cut(g, a, b)
        rec(0, f.name, "R0 갭 규칙", "", y, r0, u)

        # R1a gap_z
        az, bz = fit_ab(tr.gap_z.values, y_tr)
        rec(0, f.name, "R1a gap_z 규칙", "", y, cut(va.gap_z.values, az, bz), u)

        # R1b 변동성 3구간별
        edges = np.nanquantile(tr.vol20, [1 / 3, 2 / 3])
        bt, bv = np.digitize(tr.vol20, edges), np.digitize(va.vol20, edges)
        r1b = np.full(len(va), 2)
        for k in range(3):
            ak, bk = fit_ab(g_tr[bt == k], y_tr[bt == k])
            r1b[bv == k] = cut(g[bv == k], ak, bk)
        rec(0, f.name, "R1b 변동성 구간별 규칙", "", y, r1b, u)

        # R3 실적일 경계 배율
        e_va = va.earn_now.values == 1
        for s in EARN_S:
            rec(0, f.name, "R3 실적일 경계 × s", s, y, cut_scaled(g, a, b, e_va, s), u)

        # R2 내리기 (크기 모델, 시드 3개)
        r0_tr = cut(g_tr, a, b)
        for seed in SEEDS:
            m = lgb_binary(tr, MAG, np.isin(y_tr, [0, 4]).astype(int), seed, 100)
            pb_tr, pb = m.predict(tr[MAG]), m.predict(va[MAG])
            ext_tr = np.isin(r0_tr, [0, 4])
            for q in DEMOTE:
                thr = np.quantile(pb_tr[ext_tr], q) if q > 0 else -np.inf
                rec(seed, f.name, "R2 급등락 내리기 (하위 q)", q, y, demote(r0, pb, thr), u)
            # 조합은 아래에서 고른 값으로 다시 만듦 → 확률 저장
            rows.append({"seed": seed, "fold": f.name, "model": "_cache", "pb": pb, "thr_by_q": {q: (np.quantile(pb_tr[ext_tr], q) if q > 0 else -np.inf) for q in DEMOTE}})
        rows.append({"fold": f.name, "model": "_ctx", "a": a, "b": b, "e_va": e_va, "g": g, "y": y, "u": u})
        print(f"  {f.name} 완료", flush=True)

    cache = [r for r in rows if r["model"] in ("_cache", "_ctx")]
    r = pd.DataFrame([r for r in rows if r["model"] not in ("_cache", "_ctx")])
    m = r.groupby(["model", "param", "fold"])[["all", "unseen", "big_recall", "big_prec", "pred_big%"]].mean()
    w = m["all"].unstack("fold")
    w["fold1~3"] = w[["fold1", "fold2", "fold3"]].mean(axis=1)
    w["4폴드 평균"] = w[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    w = w.join(m.groupby(level=[0, 1]).mean()[["unseen", "big_recall", "big_prec", "pred_big%"]])

    # 고르기 (fold1~3)
    best_q = w.loc["R2 급등락 내리기 (하위 q)"]["fold1~3"].idxmax()
    best_s = w.loc["R3 실적일 경계 × s"]["fold1~3"].idxmax()
    use_r1b = w.loc[("R1b 변동성 구간별 규칙", ""), "fold1~3"] > w.loc[("R0 갭 규칙", ""), "fold1~3"]
    print(f"\nfold1~3 으로 고른 값: R2 q = {best_q}, R3 s = {best_s}, R1b 사용 = {use_r1b}")

    # 조합: R0(또는 R1b 는 경계가 구간별이라 여기선 R0) + R3(s) + R2(q)
    ctx = {c["fold"]: c for c in cache if c["model"] == "_ctx"}
    comb = []
    for c in cache:
        if c["model"] != "_cache":
            continue
        k = ctx[c["fold"]]
        pred = cut_scaled(k["g"], k["a"], k["b"], k["e_va"], float(best_s))
        pred = demote(pred, c["pb"], c["thr_by_q"][float(best_q)])
        sc = score(k["y"], pred)
        comb.append({"seed": c["seed"], "fold": c["fold"], "all": sc["score"], "unseen": score(k["y"][k["u"]], pred[k["u"]])["score"],
                     "big_recall": sc["big_recall"], "big_prec": sc["big_prec"], "pred_big%": np.isin(pred, [0, 4]).mean() * 100})
    cm = pd.DataFrame(comb).groupby("fold")[["all", "unseen", "big_recall", "big_prec", "pred_big%"]].mean()
    row = cm["all"].to_dict()
    row.update({"fold1~3": cm.loc[["fold1", "fold2", "fold3"], "all"].mean(), "4폴드 평균": cm["all"].mean(),
                **cm[["unseen", "big_recall", "big_prec", "pred_big%"]].mean().to_dict()})
    w.loc[(f"조합 R3(s={best_s}) + R2(q={best_q})", ""), :] = pd.Series(row)

    pd.set_option("display.width", 250)
    show = w.reset_index()
    show["param"] = show["param"].astype(str)
    print("\n== 갭 규칙 + α (fold1~3 / fold4 / 4폴드 평균) ==")
    print(show.round(4).to_string(index=False))
    w.to_csv(OUT / "exp_gapplus.csv")


if __name__ == "__main__":
    run()
