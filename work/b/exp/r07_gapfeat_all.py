"""gap_z 규칙 + 피처 하나 — 표의 모든 피처 × 세 가지 붙이는 방식. A 폴드 기준.

    uv run python work/b/exp/r07_gapfeat_all.py

x̃ = (x − train 중앙값) / train 표준편차, ±5 로 자름, 결측 0.
  더하기    s = gz + β·σ(gz)·x̃            — 방향 정보를 보탬
  곱하기    s = gz · exp(β·x̃)              — 그날 갭을 얼마나 믿을지 (크기·신뢰 피처)
  같은방향  s = gz · exp(β·x̃·sign(gz))     — 피처가 갭과 같은 방향이면 갭을 더 믿음
경계는 A 방식(분위 후보 중 train score 최대). β 는 폴드마다 train score 최대 (val 안 봄).
채택: 4폴드 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음.
조합 수가 많아(≈225) 우연히 통과하는 것이 나올 수 있음 → 통과한 것은 β 부호가 폴드마다 같은지도 봄.
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from lab.folds import unseen_symbols  # noqa: E402
from common import BETAS, OUT, SKIP, engineered, load_table, one  # noqa: E402


def run():
    t0 = time.time()
    table, folds = load_table()
    for k, v in engineered(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in table.columns if c not in SKIP and pd.api.types.is_numeric_dtype(table[c])]
    unseen = set(unseen_symbols())
    data = {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        gz_tr = tr.gap_z.values
        x = {}
        for c in feats:
            med, std = np.nanmedian(tr[c]), np.nanstd(tr[c])
            if std > 0:
                x[c] = tuple(np.nan_to_num(np.clip((z[c].values - med) / std, -5, 5)) for z in (tr, va))
        data[f.name] = {"gz_tr": gz_tr, "gz_va": va.gap_z.values, "sd": np.nanstd(gz_tr), "x": x,
                        "y_tr": tr.label.values, "y": va.label.values, "u": va.symbol.isin(unseen).values}
    feats = [c for c in feats if all(c in d["x"] for d in data.values())]
    jobs = [(c, form) for c in feats for form in BETAS]
    print(f"피처 {len(feats)}개 × 방식 3 = {len(jobs)}개 조합", flush=True)
    res = Parallel(n_jobs=-1, verbose=5)(delayed(one)(c, form, data) for c, form in jobs)
    r = pd.DataFrame([row for part in res for row in part])
    r.to_csv(OUT / "exp_gapfeat_all.csv", index=False)

    base = r[(r.beta == 0) & (r["방식"] == "더하기")].groupby("fold")["all"].mean()
    print("\ngap_z 규칙:", base.round(4).to_dict(), "평균", round(base.mean(), 4))
    ga = r.loc[r.groupby(["피처", "방식", "fold"])["train"].idxmax()]
    da = ga.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(base, axis=1)
    du = ga.pivot_table(index=["피처", "방식"], columns="fold", values="unseen").mean(axis=1)
    bt = ga.pivot_table(index=["피처", "방식"], columns="fold", values="beta")
    out = pd.DataFrame({
        "β": bt.apply(lambda s: "/".join(f"{v:g}" for v in s), axis=1),
        "β부호 일정": bt.apply(lambda s: (s > 0).all() or (s < 0).all(), axis=1),
        "Δ평균": da.mean(axis=1), "Δfold4": da["fold4"], "상승폴드": (da > 0).sum(axis=1), "unseen": du,
    }).join(da[["fold1", "fold2", "fold3"]].add_prefix("Δ"))
    out["채택"] = (out["Δ평균"] >= 0.005) & (out["상승폴드"] >= 3) & (out["Δfold4"] >= 0)
    out = out.sort_values("Δ평균", ascending=False)
    out.to_csv(OUT / "exp_gapfeat_all_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print("\n== 상위 30 (Δ = gap_z 규칙 대비, β 는 train 으로 고름) ==")
    print(out.head(30).round(4).to_string())
    print(f"\n채택 {out['채택'].sum()}개 / {len(out)}개.  Δ평균 > 0 인 조합 {(out['Δ평균'] > 0).sum()}개, "
          f"β=0 만 고른 조합 {(bt == 0).all(axis=1).sum()}개")
    print("\n방식별 최고:")
    print(out.reset_index().groupby("방식").head(3).round(4).to_string(index=False))
    print(f"\n{time.time() - t0:.0f}초")


if __name__ == "__main__":
    run()
