"""9차: 최대한 다양하게 — 규칙 변형, 지난 결과로 만든 피처, 원본에서 새로 만든 피처. A 폴드 기준.

    uv run python work/b/exp/r09_more.py

A 규칙 자체 (피처 추가 없음, 고르는 것은 train 안에서만)
  A1 상승·하락 경계를 따로          A2 경계 후보 25개(촘촘히)
  A3 최근 250/500 거래일로만 경계    A4 경계 부트스트랩 30번의 중앙값
  A5 정규화 갭 평균 (÷vol20, ÷vol60, ÷atr14, ÷√(vol20·vol60))
  A6 gap_z 30구간마다 등급을 train score 로 직접 정함
  A7 gap_z 20구간 × 보조 피처 3구간 표 (보조 피처는 train 내부 검증으로 고름)
B 지난 결과(정답은 대상일 전 것만)로 만든 피처
  peer_gz      과거 60일 수익률 상관 상위 5종목의 오늘 gap_z 평균
  idio_gz      gap_z − peer_gz
  rel_mkt20    지난 20일 날짜별 순위상관(gap_z, 수익률) 평균 — 최근 갭이 잘 맞았나 (시장)
  follow_mkt20 지난 20일 수익률÷vol20 을 gap_z 에 회귀한 기울기 (시장, 이어짐/되돌림)
  follow_sym60 같은 것을 종목별 지난 60번으로
  hit_sym60    종목별 지난 60번 중 갭 방향 = 수익률 방향 비율 (|gap_z| > 0.3 만)
C 원본(일봉·시간외 봉)에서 새로 만든 피처 — guard 아래에서 만듦 (cache/b_v3.parquet)
  brk_z        갭이 전일 고가 위/저가 아래로 벗어난 만큼 ÷ vol20 (범위 안이면 0)
  rng_pos      마지막 시간외 가격의 전일 고저 범위 안 위치 (1 넘으면 고가 위)
  gap_first_z  첫 시간외 봉 가격의 갭 ÷ vol20 (첫 반응)
  gap_drift_z  gap_z − gap_first_z (밤사이 갭이 커졌나 줄었나)
  ovn5_z, ovn20_z  지난 5/20일 밤사이 수익률(시가÷전일 종가) 합 ÷ vol20
  intra20_z    지난 20일 장중 수익률(종가÷시가) 합 ÷ vol20
B·C 평가: 7차와 같음 (더하기/곱하기/같은방향, β 는 폴드마다 train score 최대) + 값을 섞은 가짜 피처 3번.
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from lab.cv import feature_table  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import BETAS, OUT, QS, cut, fscore, inner_splits, load_table, one, prep  # noqa: E402
from builders import NEW_B, NEW_C, build_more, past_feats  # noqa: E402


QS_FINE = list(np.round(np.linspace(0.40, 0.995, 25), 4))
AUX = ["vol_5_20", "ext_range_z", "mkt_absgap", "vol20", "rsi14", "mkt_ret5", "ret5_z", "earn_now"]


# ------------------------------------------------------------------ A: 규칙 변형
def fit_q(x, y, qs=QS):
    ok = ~np.isnan(x)
    q = np.quantile(np.abs(x[ok]), qs)
    return max(((fscore(y[ok], cut(x[ok], a, b)), a, b) for i, a in enumerate(q) for b in q[i + 1:]))[1:]


def cut2(x, au, bu, ad, bd):
    o = np.full(len(x), 2)
    o[x >= au], o[x >= bu], o[x <= -ad], o[x <= -bd] = 3, 4, 1, 0
    o[np.isnan(x)] = 2
    return o


def fit_asym(x, y):
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    q = np.quantile(np.abs(x), QS)
    a, b = fit_q(x, y)
    p = [a, b, a, b]
    for _ in range(2):
        for side in (0, 2):
            best = None
            for i, u in enumerate(q):
                for v in q[i + 1:]:
                    t = p.copy()
                    t[side], t[side + 1] = u, v
                    s = fscore(y, cut2(x, *t))
                    if best is None or s > best[0]:
                        best = (s, t)
            p = best[1]
    return p


def bins_of(x_tr, n):
    e = np.unique(np.nanquantile(x_tr, np.linspace(0, 1, n + 1)[1:-1]))
    return lambda x: np.searchsorted(e, np.nan_to_num(x), side="right"), len(e) + 1


def fit_table(cell_tr, y, init, ncell):
    """칸마다 등급을 train score 가 오르는 쪽으로 바꿈 (좌표 상승 2바퀴)."""
    lab = np.array([np.bincount(init[cell_tr == k], minlength=5).argmax() if (cell_tr == k).any() else 2
                    for k in range(ncell)])
    cur = fscore(y, lab[cell_tr])
    for _ in range(2):
        for k in range(ncell):
            if not (cell_tr == k).any():
                continue
            for j in range(5):
                if j == lab[k]:
                    continue
                old = lab[k]
                lab[k] = j
                s = fscore(y, lab[cell_tr])
                if s > cur:
                    cur = s
                else:
                    lab[k] = old
    return lab


def table_rule(tr, va, aux, nb=20):
    """gap_z nb 구간 (× aux 3구간) 표."""
    g_tr, y = tr.gap_z.values, tr.label.values
    fb, ng = bins_of(g_tr, nb)
    a, b = fit_q(g_tr, y)
    init = cut(g_tr, a, b)
    if aux is None:
        c_tr, c_va, n = fb(g_tr), fb(va.gap_z.values), ng
    else:
        fa, na = bins_of(tr[aux].values, 3)
        c_tr = fb(g_tr) * na + fa(tr[aux].values)
        c_va = fb(va.gap_z.values) * na + fa(va[aux].values)
        n = ng * na
    lab = fit_table(c_tr, y, init, n)
    out = lab[c_va]
    out[np.isnan(va.gap_z.values)] = 2
    return out


def rule_variants(tr, va):
    y = tr.label.values
    g_tr, g_va = tr.gap_z.values, va.gap_z.values
    out = {}
    a, b = fit_q(g_tr, y)
    out["A0 gap_z 규칙"] = cut(g_va, a, b)
    out["A1 상승·하락 경계 따로"] = cut2(g_va, *fit_asym(g_tr, y))
    out["A2 경계 후보 25개"] = cut(g_va, *fit_q(g_tr, y, QS_FINE))
    ud = np.sort(tr.date.unique())
    for n in (250, 500):
        m = tr.date.values >= ud[-n] if len(ud) > n else np.ones(len(tr), bool)
        out[f"A3 최근 {n}일로 경계"] = cut(g_va, *fit_q(g_tr[m], y[m]))
    rng = np.random.default_rng(0)
    ab = []
    for _ in range(30):
        pick = rng.choice(ud, len(ud))
        idx = np.concatenate([np.flatnonzero(tr.date.values == d) for d in pick])
        ab.append(fit_q(g_tr[idx], y[idx]))
    out["A4 경계 부트스트랩 중앙값"] = cut(g_va, *np.median(np.array(ab), axis=0))

    def norm(df):
        g = df.gap
        return [g / df.vol20, g / df.vol60, g / df.atr14, g / np.sqrt(df.vol20 * df.vol60)]
    ntr, nva = norm(tr), norm(va)
    sds = [np.nanstd(v.replace([np.inf, -np.inf], np.nan)) for v in ntr]
    comb = lambda L: np.nanmean(np.column_stack([(v.replace([np.inf, -np.inf], np.nan) / s).values for v, s in zip(L, sds)]), 1)  # noqa: E731
    c_tr, c_va = comb(ntr), comb(nva)
    out["A5 정규화 갭 4개 평균"] = cut(c_va, *fit_q(c_tr, y))
    out["A6 gap_z 30구간 표"] = table_rule(tr, va, None, 30)

    # A7: 보조 피처를 train 내부 검증으로 고름 (None = 1차원 20구간)
    sp = inner_splits(tr.date.values)
    sc = {}
    for aux in [None, *AUX]:
        sc[aux] = np.mean([fscore(tr.label.values[w], table_rule(tr[t], tr[w], aux)) for t, w in sp])
    best = max(sc, key=sc.get)
    out["A7 gap_z × 보조 피처 표"] = table_rule(tr, va, best)
    return out, best


def run():
    table, folds = load_table()
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    c = feature_table(days, build_more, name="b_v3")
    table = table.merge(c.drop(columns=["label", "ret_pct"]), on=["date", "symbol"], how="left")
    table = past_feats(table)
    for k in NEW_B + NEW_C:
        table[k] = table[k].replace([np.inf, -np.inf], np.nan)
    print("새 피처 결측 비율:", table[NEW_B + NEW_C].isna().mean().round(3).to_dict(), flush=True)
    unseen = set(unseen_symbols())
    pd.set_option("display.width", 250)

    # ---- A
    rows, aux_pick = [], {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)].reset_index(drop=True)
        va = table[table.date.isin({d.date for d in f.val})].reset_index(drop=True)
        y, u = va.label.values, va.symbol.isin(unseen).values
        preds, aux_pick[f.name] = rule_variants(tr, va)
        for k, p in preds.items():
            rows.append({"model": k, "fold": f.name, "all": fscore(y, p), "unseen": fscore(y[u], p[u]),
                         "pred_big%": np.isin(p, [0, 4]).mean() * 100})
        print(f"  A {f.name} 완료", flush=True)
    r = pd.DataFrame(rows)
    w = r.pivot(index="model", columns="fold", values="all")
    base = w.loc["A0 gap_z 규칙"]
    w["4폴드 평균"] = w[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    w["Δ평균"] = w["4폴드 평균"] - base.mean()
    w["상승폴드"] = (w[["fold1", "fold2", "fold3", "fold4"]] > base + 1e-12).sum(axis=1)
    w = w.join(r.groupby("model")[["unseen", "pred_big%"]].mean())
    print("\n== A. 규칙 자체 바꾸기 ==")
    print(w.round(4).to_string())
    print("A7 이 고른 보조 피처:", aux_pick)
    w.to_csv(OUT / "exp_more_A.csv")

    # ---- B·C
    feats = NEW_B + NEW_C
    data = prep(table, folds, feats, unseen)
    fs = [c for c in feats if all(c in d["x"] for d in data.values())]
    res = Parallel(n_jobs=-1)(delayed(one)(c, form, data) for c in fs for form in BETAS)
    rr = pd.DataFrame([row for p in res for row in p])
    b0 = rr[(rr.beta == 0) & (rr["방식"] == "더하기")].groupby("fold")["all"].mean()
    ga = rr.loc[rr.groupby(["피처", "방식", "fold"])["train"].idxmax()]
    da = ga.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(b0, axis=1)
    bt = ga.pivot_table(index=["피처", "방식"], columns="fold", values="beta")
    o = pd.DataFrame({"β": bt.apply(lambda s: "/".join(f"{v:g}" for v in s), axis=1),
                      "Δ평균": da.mean(axis=1), "Δfold4": da["fold4"], "상승폴드": (da > 0).sum(axis=1)})
    o["채택"] = (o["Δ평균"] >= 0.005) & (o["상승폴드"] >= 3) & (o["Δfold4"] >= 0)
    o = o.sort_values("Δ평균", ascending=False)
    print(f"\n== B·C. 새 피처 {len(fs)}개 × 3방식 ==")
    print(o.round(4).to_string())
    o.to_csv(OUT / "exp_more_BC.csv")

    rng = np.random.default_rng(1)
    for rep in range(3):
        sh = table.copy()
        for c in fs:
            sh[c] = rng.permutation(sh[c].values)
        dsh = prep(sh, folds, fs, unseen)
        res = Parallel(n_jobs=-1)(delayed(one)(c, form, dsh) for c in fs for form in BETAS)
        q = pd.DataFrame([row for p in res for row in p])
        g2 = q.loc[q.groupby(["피처", "방식", "fold"])["train"].idxmax()]
        d2 = g2.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(b0, axis=1)
        ok = (d2.mean(axis=1) >= 0.005) & ((d2 > 0).sum(axis=1) >= 3) & (d2["fold4"] >= 0)
        print(f"섞은 가짜 {rep + 1}회차: 통과 {ok.sum()}개 / {len(ok)}, Δ평균 최대 {d2.mean(axis=1).max():.4f}", flush=True)


if __name__ == "__main__":
    run()
