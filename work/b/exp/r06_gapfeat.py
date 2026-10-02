"""gap_z 규칙에 피처를 하나씩 더하기 — A 폴드 기준.

    uv run python work/b/exp/r06_gapfeat.py

점수 s = gap_z + β·σ(gap_z)·x̃,  x̃ = (x − train 중앙값) / train 표준편차 (±5 로 자름, 결측은 0).
s 에 A 방식 경계(분위 후보 중 train score 최대)를 그대로 적용.
β 고르기: (가) 폴드마다 train score 최대인 β (val 안 봄), (나) fold1~3 평균 val score 최대인 β 하나.
채택: 4폴드 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음 — (가) 기준.
"""

import sys

import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import OUT, cut, fit_ab, load_table  # noqa: E402

BETAS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
CANDS = [
    # 시간외·장전 흐름
    "pre_move", "pre_late", "post_ret", "ext_pos", "ext_range_z", "mkt_gap", "gap_rank",
    # 전날·최근 가격 (되돌림/추세)
    "ret1_z", "ret5_z", "ret20_z", "rel_ret1_z", "intraday1", "gap_open1", "last_hour_z",
    "dist_ma20_z", "rsi14", "bb_pctb", "macd_hist", "streak", "pos60", "mkt_ret1",
    # 실적·애널리스트
    "earn_surprise_now", "last_surprise", "an_now_net", "an_tgt_gap", "an_tgtchg30",
]


def run(only=None):
    table, folds = load_table()
    unseen = set(unseen_symbols())
    cands = [c for c in (only or CANDS) if c in table.columns]
    miss = sorted(set(only or CANDS) - set(cands))
    if miss:
        print("표에 없는 후보:", miss)
    rows, corr = [], []
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        y_tr, y, u = tr.label.values, va.label.values, va.symbol.isin(unseen).values
        gz_tr, gz_va = tr.gap_z.values, va.gap_z.values
        sd = np.nanstd(gz_tr)
        for c in cands:
            med, std = np.nanmedian(tr[c]), np.nanstd(tr[c])
            if not std > 0:
                continue
            xt = np.nan_to_num(np.clip((tr[c].values - med) / std, -5, 5))
            xv = np.nan_to_num(np.clip((va[c].values - med) / std, -5, 5))
            corr.append({"fold": f.name, "피처": c, "corr": pd.Series(tr[c].values).corr(pd.Series(gz_tr), method="spearman")})
            for b in BETAS:
                s_tr, s_va = gz_tr + b * sd * xt, gz_va + b * sd * xv
                lo, hi = fit_ab(s_tr, y_tr)
                pv = cut(s_va, lo, hi)
                rows.append({"fold": f.name, "피처": c, "beta": b, "train": score(y_tr, cut(s_tr, lo, hi))["score"],
                             "all": score(y, pv)["score"], "unseen": score(y[u], pv[u])["score"]})
        print(f"  {f.name} 완료", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_gapfeat.csv", index=False)
    base = r[r.beta == 0].groupby("fold")["all"].mean()  # gap_z 규칙
    print("\ngap_z 규칙:", base.round(4).to_dict(), "평균", round(base.mean(), 4))

    # (가) 폴드마다 train 최대 β
    ga = r.loc[r.groupby(["피처", "fold"])["train"].idxmax()]
    da = ga.pivot(index="피처", columns="fold", values="all").sub(base, axis=1)
    ba = ga.pivot(index="피처", columns="fold", values="beta")
    # (나) fold1~3 평균 val 최대 β
    m = r.pivot_table(index=["피처", "beta"], columns="fold", values="all")
    m13 = m[["fold1", "fold2", "fold3"]].mean(axis=1)
    bb = m13.groupby(level=0).idxmax()
    dn = m.loc[bb.values].droplevel("beta").sub(base, axis=1)

    cr = pd.DataFrame(corr).groupby("피처")["corr"].mean()
    out = pd.DataFrame({
        "gap_z 상관": cr,
        "(가) β": ba.apply(lambda s: "/".join(f"{v:g}" for v in s), axis=1),
        "(가) Δ평균": da.mean(axis=1), "(가) Δfold4": da["fold4"], "(가) 상승폴드": (da > 0).sum(axis=1),
        "(나) β": bb.map(lambda t: t[1]),
        "(나) Δfold1~3": dn[["fold1", "fold2", "fold3"]].mean(axis=1), "(나) Δfold4": dn["fold4"],
    }).sort_values("(가) Δ평균", ascending=False)
    out["채택"] = (out["(가) Δ평균"] >= 0.005) & (out["(가) 상승폴드"] >= 3) & (out["(가) Δfold4"] >= 0)
    pd.set_option("display.width", 250)
    print("\n== gap_z 규칙 + 피처 하나 (Δ = gap_z 규칙 대비) ==")
    print(out.round(4).to_string())
    out.to_csv(OUT / "exp_gapfeat_summary.csv")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
