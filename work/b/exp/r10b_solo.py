"""피처 하나만으로 — gap_z 규칙 없이 각 피처의 단독 성능. A 폴드 기준.

    uv run python work/b/exp/r10b_solo.py

혼자 규칙: s = 부호 × (x − train 중앙값), 부호는 train score 로 고름, 경계는 A 방식. 4폴드 val score.
방향 IC: 날짜별 순위상관(x, 다음날 수익률), 크기 IC: 날짜별 순위상관(x, |수익률|). 4폴드 val 날짜 전체, t 값 포함.
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from lab.cv import feature_table  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import OUT, SKIP, cut, engineered, fit_ab as ffit, fscore, load_table  # noqa: E402
from builders import build_more, past_feats  # noqa: E402

GAPLIKE = {"gap_z", "gap", "abs_gap_z", "gap_rel_z", "gap_onz", "gap_rank", "new_idio_gap_z", "idio_gz",
           "gap_first_z", "brk_z", "rng_pos", "ext_pos", "pre_move", "new_pre_z", "pre_late", "mkt_gap", "mkt_gap_z", "peer_gz"}


def solo(c, folds_data):
    out = {}
    for fn, (x_tr, y_tr, x_va, y_va) in folds_data.items():
        med = np.nanmedian(x_tr)
        best = None
        for sgn in (1, -1):
            s = sgn * (x_tr - med)
            a, b = ffit(s, y_tr)
            v = fscore(y_tr, cut(s, a, b))
            if best is None or v > best[0]:
                best = (v, sgn, a, b)
        _, sgn, a, b = best
        out[fn] = (fscore(y_va, cut(sgn * (x_va - med), a, b)), sgn)
    return c, out


def ic(df, c, y):
    r = df[["date", c, y]].dropna()
    r = r.assign(a=r.groupby("date")[c].rank(), b=r.groupby("date")[y].rank())
    s = r.groupby("date").apply(lambda g: g.a.corr(g.b) if g.a.nunique() > 1 else np.nan, include_groups=False).dropna()
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 else np.nan


def run():
    table, folds = load_table()
    for k, v in engineered(table).items():
        table[k] = v.replace([np.inf, -np.inf], np.nan)
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    c3 = feature_table(days, build_more, name="b_v3")
    table = past_feats(table.merge(c3.drop(columns=["label", "ret_pct"]), on=["date", "symbol"], how="left"))
    feats = [c for c in table.columns if c not in (SKIP - {"gap", "gap_z"}) and c not in ("date", "symbol", "label", "ret_pct")
             and pd.api.types.is_numeric_dtype(table[c])]
    for c in feats:
        table[c] = table[c].replace([np.inf, -np.inf], np.nan)
    unseen = set(unseen_symbols())
    data, vals = {}, []
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        vals.append(va)
        data[f.name] = {c: (tr[c].values.astype(float), tr.label.values, va[c].values.astype(float), va.label.values) for c in feats}
    res = Parallel(n_jobs=-1)(delayed(solo)(c, {fn: d[c] for fn, d in data.items()}) for c in feats)
    V = pd.concat(vals).assign(absret=lambda d: d.ret_pct.abs())
    rows = []
    for c, o in res:
        sc = [o[fn][0] for fn in ["fold1", "fold2", "fold3", "fold4"]]
        di, dt = ic(V, c, "ret_pct")
        mi, mt = ic(V, c, "absret")
        rows.append({"피처": c, "갭 계열": c in GAPLIKE, "혼자 규칙 4폴드": np.mean(sc), "fold4": sc[3],
                     "부호": "/".join("+" if o[fn][1] > 0 else "−" for fn in ["fold1", "fold2", "fold3", "fold4"]),
                     "방향 IC": di, "방향 t": dt, "크기 IC": mi, "크기 t": mt})
    r = pd.DataFrame(rows).set_index("피처")
    r.to_csv(OUT / "exp_solo.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("== 혼자 규칙 점수 순 (상위 25) ==")
    print(r.sort_values("혼자 규칙 4폴드", ascending=False).head(25).round(4).to_string())
    print("\n== 갭 계열 빼고, 혼자 규칙 점수 순 (상위 15) ==")
    print(r[~r["갭 계열"]].sort_values("혼자 규칙 4폴드", ascending=False).head(15).round(4).to_string())
    print("\n== 갭 계열 빼고, |방향 t| 순 (상위 12) ==")
    nr = r[~r["갭 계열"]]
    print(nr.reindex(nr["방향 t"].abs().sort_values(ascending=False).index).head(12).round(4).to_string())
    print("\n== 크기 IC 순 (상위 12) ==")
    print(r.sort_values("크기 IC", ascending=False).head(12).round(4).to_string())


if __name__ == "__main__":
    run()
