"""12차: 두세 개 조합을 빠짐없이 — 고르는 것은 train 안에서만. A 폴드 기준.

    uv run python work/b/exp/r12_combo.py

후보: 지금까지 만든 모든 피처(b_v2 + 새 피처 + b_v3 + b_v4) × 더하기/곱하기/같은방향.
폴드마다
  1) train 내부 검증(마지막 120일을 60일씩 두 구간)으로 각 (피처, 방식)의 β 와 내부 점수를 구함.
     피처마다 가장 좋은 방식 하나만 남기고, 내부 점수 상위 TOP 개를 후보로.
  2) 후보 안에서 2개 조합 전부, 3개 조합 전부. β 는 1)에서 정한 값 그대로 (자유도 늘리지 않음).
  3) 내부 점수가 가장 좋은 조합(1개·2개·3개 중)을 골라 val 로 확인.
  4) 모든 조합의 val 결과 분포도 봄 (꾸준히 오르는 조합이 있는지).
우연 기준선: 피처 값을 섞어서 같은 절차 2번.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from lab.folds import unseen_symbols  # noqa: E402
from common import BETAS, OUT, apply, cut, fit_ab as ffit, fscore, full_table, inner_splits, stdz  # noqa: E402

TOP = 12
FORMS = list(BETAS)


def inner_eval(parts, gz, y, splits, sd):
    """parts = [(x, form, b)], 내부 두 구간 평균 score."""
    s = gz.copy()
    for x, form, b in parts:
        s = apply(form, b, s, x, sd)
    out = []
    for t, w in splits:
        a, bb = ffit(s[t], y[t])
        out.append(fscore(y[w], cut(s[w], a, bb)))
    return np.mean(out)


def single(c, x, gz, y, splits, sd):
    best = []
    for form in FORMS:
        bb = max(((inner_eval([(x, form, b)], gz, y, splits, sd), b) for b in BETAS[form] if b != 0))
        best.append((bb[0], c, form, bb[1]))
    return max(best)


def fold_run(tr, va, feats, unseen_mask):
    y_tr, y_va = tr.label.values, va.label.values
    gz_tr, gz_va = tr.gap_z.values, va.gap_z.values
    sd = np.nanstd(gz_tr)
    splits = inner_splits(tr.date.values)
    X_tr, X_va = {}, {}
    for c in feats:
        a = stdz(tr[c].values, tr[c].values)
        if a is not None:
            X_tr[c], X_va[c] = a, stdz(tr[c].values, va[c].values)
    base_in = inner_eval([], gz_tr, y_tr, splits, sd)
    sing = Parallel(n_jobs=-1)(delayed(single)(c, X_tr[c], gz_tr, y_tr, splits, sd) for c in X_tr)
    sing = sorted(sing, reverse=True)[:TOP]
    cands = [(c, form, b) for _, c, form, b in sing]

    def val_of(combo):
        st, sv = gz_tr.copy(), gz_va.copy()
        for c, form, b in combo:
            st, sv = apply(form, b, st, X_tr[c], sd), apply(form, b, sv, X_va[c], sd)
        a, bb = ffit(st, y_tr)
        p = cut(sv, a, bb)
        return fscore(y_va, p), fscore(y_va[unseen_mask], p[unseen_mask])

    combos = [(k,) for k in cands] + list(combinations(cands, 2)) + list(combinations(cands, 3))
    ins = Parallel(n_jobs=-1)(delayed(inner_eval)([(X_tr[c], f, b) for c, f, b in cb], gz_tr, y_tr, splits, sd)
                              for cb in combos)
    vals = Parallel(n_jobs=-1)(delayed(val_of)(cb) for cb in combos)
    a, bb = ffit(gz_tr, y_tr)
    pb = cut(gz_va, a, bb)
    base_val = fscore(y_va, pb)
    rows = [{"조합": " + ".join(f"{c}·{f}({b:g})" for c, f, b in cb), "개수": len(cb), "내부Δ": i - base_in,
             "valΔ": v[0] - base_val, "unseen": v[1]} for cb, i, v in zip(combos, ins, vals)]
    return pd.DataFrame(rows), base_val, fscore(y_va[unseen_mask], pb[unseen_mask])


def procedure(table, folds, feats, unseen, tag):
    res = {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
        va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
        res[f.name] = fold_run(tr, va, feats, va.symbol.isin(unseen).values)
        print(f"  [{tag}] {f.name} 완료", flush=True)
    return res


def summarize(res):
    pick = {}
    for fn, (r, base, _) in res.items():
        best = {k: r[r["개수"] == k].sort_values("내부Δ", ascending=False).iloc[0] for k in (1, 2, 3)}
        best["전체"] = r.sort_values("내부Δ", ascending=False).iloc[0]
        pick[fn] = best
    return pick


def run():
    table, folds, feats = full_table()
    unseen = set(unseen_symbols())
    print(f"후보 피처 {len(feats)}개 × 방식 3", flush=True)
    res = procedure(table, folds, feats, unseen, "진짜")
    pick = summarize(res)
    F = list(res)
    base = pd.Series({fn: res[fn][1] for fn in F})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(f"\ngap_z 규칙: {base.round(4).to_dict()} 평균 {base.mean():.4f}")
    print("\n== 내부 검증으로 고른 조합 → val (Δ = gap_z 규칙 대비) ==")
    for k in (1, 2, 3, "전체"):
        d = pd.Series({fn: pick[fn][k]["valΔ"] for fn in F})
        print(f"\n[{k}개]" if k != "전체" else "\n[1~3개 중 내부 최고]",
              f"Δ평균 {d.mean():+.4f}, 폴드별 {d.round(4).to_dict()}, 상승 {int((d > 0).sum())}폴드")
        for fn in F:
            p = pick[fn][k]
            print(f"   {fn}: {p['조합']}  (내부 {p['내부Δ']:+.4f}, val {p['valΔ']:+.4f})")

    print("\n== 모든 조합의 val 분포 (폴드별) ==")
    for fn in F:
        r = res[fn][0]
        for k in (1, 2, 3):
            q = r[r["개수"] == k]["valΔ"]
            print(f"  {fn} {k}개 {len(q):>3}개 조합: val 오른 비율 {(q > 0).mean():.2f}, 평균 {q.mean():+.4f}, "
                  f"최대 {q.max():+.4f}, 내부·val 상관 {r[r['개수'] == k][['내부Δ', 'valΔ']].corr().iloc[0, 1]:+.2f}")
    # 4폴드 모두 후보에 든 조합
    allr = pd.concat([res[fn][0].assign(fold=fn) for fn in F])
    allr["key"] = allr["조합"].str.replace(r"\([^)]*\)", "", regex=True)
    common4 = allr.groupby("key").filter(lambda g: g.fold.nunique() == 4)
    if len(common4):
        cs = common4.groupby("key").agg(Δ평균=("valΔ", "mean"), 상승폴드=("valΔ", lambda s: int((s > 0).sum())),
                                       Δfold4=("valΔ", lambda s: s.iloc[-1]))
        print("\n== 4폴드 모두 후보에 든 조합 (Δ평균 순) ==")
        print(cs.sort_values("Δ평균", ascending=False).head(15).round(4).to_string())
    allr.to_csv(OUT / "exp_combo.csv", index=False)

    rng = np.random.default_rng(3)
    for rep in range(2):
        sh = table.copy()
        for c in feats:
            sh[c] = rng.permutation(sh[c].values)
        rs = procedure(sh, folds, feats, unseen, f"가짜 {rep + 1}")
        ps = summarize(rs)
        for k in (2, 3, "전체"):
            d = pd.Series({fn: ps[fn][k]["valΔ"] for fn in F})
            print(f"섞은 가짜 {rep + 1}회차 [{k}]: Δ평균 {d.mean():+.4f}, 상승 {int((d > 0).sum())}폴드", flush=True)
        mx = pd.concat([rs[fn][0].assign(fold=fn) for fn in F])
        mx["key"] = mx["조합"].str.replace(r"\([^)]*\)", "", regex=True)
        cm = mx.groupby("key").filter(lambda g: g.fold.nunique() == 4).groupby("key")["valΔ"].mean()
        print(f"  4폴드 공통 조합 Δ평균 최대 {cm.max() if len(cm) else float('nan'):+.4f}", flush=True)


if __name__ == "__main__":
    run()
