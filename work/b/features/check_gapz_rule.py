"""gapz_rule.py 확인 — 실험 결과(4폴드 0.447)가 그대로 나오는지, 피처 값이 실험 표와 같은지, 누수가 없는지.

    uv run python work/b/features/check_gapz_rule.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp"))

from src import score as src_score  # noqa: E402
from lab.cv import feature_table  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from lab.leak import guard  # noqa: E402
from common import load_table  # noqa: E402
import gapz_rule as G  # noqa: E402


def main():
    table, folds = load_table()          # 실험에서 쓴 표 (cache/b_v2.parquet)
    unseen = set(unseen_symbols())

    # 1) 피처 값: build_gapz (guard 아래) vs 실험 표
    days = [folds[0].train[100], folds[1].val[10], folds[3].val[-1]]
    for day in days:
        x = G.build_gapz(guard(day)).set_index("symbol")
        ref = table[table.date == day.date].set_index("symbol")[["gap_z", "ext_range_z"]]
        j = ref.join(x, rsuffix="_new", how="inner")
        d = max(np.nanmax(np.abs(j.gap_z - j.gap_z_new)), np.nanmax(np.abs(j.ext_range_z - j.ext_range_z_new)))
        print(f"피처 비교 {day.date.date()}: 종목 {len(j)}개, 최대 차이 {d:.2e}")

    # 2) 누수: 전 기간을 guard 아래에서 다시 만든 표와 같은지 (feature_table 이 guard 를 씌움)
    all_days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    t = feature_table(all_days, G.build_gapz, name="gapz_rule", verbose=False)
    m = t.merge(table[["date", "symbol", "gap_z", "ext_range_z"]], on=["date", "symbol"], suffixes=("", "_ref"))
    print(f"전 기간 피처: {len(m)}행, gap_z 최대 차이 {np.nanmax(np.abs(m.gap_z - m.gap_z_ref)):.2e}, "
          f"ext_range_z 최대 차이 {np.nanmax(np.abs(m.ext_range_z - m.ext_range_z_ref)):.2e}")

    # 3) score 함수가 src.score 와 같은지
    y = t.label.astype(int).values[:3000]
    p = G.cut(t.gap_z.values[:3000], 0.5, 1.5)
    print(f"score 비교: {G.score(y, p):.6f} vs src {src_score(y, p)['score']:.6f}")

    # 4) A 폴드 성능
    t["label"] = t.label.astype(int)
    rows = []
    for f in folds:
        tr = t[t.date.isin({d.date for d in f.train}) & ~t.symbol.isin(unseen)]
        va = t[t.date.isin({d.date for d in f.val})]
        prm = G.fit_table(tr)
        pr = G.predict_table(va, prm)
        u = va.symbol.isin(unseen).values
        rows.append({"fold": f.name, "score": G.score(va.label, pr), "unseen": G.score(va.label[u], pr[u]),
                     "λ": prm["lam"], "a": round(prm["a"], 3), "b": round(prm["b"], 3)})
    r = pd.DataFrame(rows).set_index("fold")
    print("\n", r.round(4).to_string())
    print(f"4폴드 평균 {r.score.mean():.4f} (기대값 0.4469), unseen {r.unseen.mean():.4f}")

    # 5) predict(day) 형태 확인
    day = folds[3].val[0]
    prm = G.fit_table(t[t.date.isin({d.date for d in folds[3].train}) & ~t.symbol.isin(unseen)])
    out = G.predict(guard(day), prm)
    print(f"\npredict({day.date.date()}): {len(out)}행 / day.symbols {len(day.symbols)}개, "
          f"label 분포 {out.label.value_counts().sort_index().to_dict()}, 빠진 종목 {set(day.symbols) - set(out.symbol)}")


if __name__ == "__main__":
    main()
