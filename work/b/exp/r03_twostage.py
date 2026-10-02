"""2단계 구조 실험 — 방향과 크기를 따로 정함. A 폴드 기준.

    uv run python work/b/exp/r03_twostage.py

구조
  T1 규칙 + 크기 모델 : 방향·보합은 갭 규칙(경계 train score 최대). 규칙이 보합이 아닌 종목 중
                         크기 모델 P(급등락) 상위를 급등락(0/4, 갭 부호)으로, 나머지는 하락/상승(1/3)으로.
  T2 방향 모델 + 크기 모델 : 방향 모델 P(상승). 급등락 = P(급등락) × 방향 확신도 |2P(상승)−1| 상위,
                         나머지는 P(상승) 을 하락/보합/상승 비율로 자름.
  T3 갭 방향 + 크기 × 갭 확신도 : 방향 = gap 부호. 급등락 = P(급등락) × |gap_z| 순위 상위, 나머지는 gap_z 로 자름.
비교: 갭 규칙 하나, 1단계 모델(V1 다중분류 + 급등락 34% 디코딩).
급등락 비율 K, 보합 비율 F 는 fold1~3 평균으로 고르고 fold4 로 확인. 시드 3개 평균.
"""


import numpy as np
import pandas as pd

import common  # noqa: E402,F401  (경로 설정)

from src import score  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from features_b import B_FEATURES_V1  # noqa: E402
from common import OUT, SEEDS, lgb_binary, load_table, train_predict  # noqa: E402
from r02_decode import decode, target_props  # noqa: E402

MAG = ["vol5", "vol20", "vol60", "atr14", "range1", "bigfreq60", "beta60", "abs_gap_z", "mkt_vol20",
       "mkt_absgap", "gap_disp", "earn_now", "earn_timing", "earn_hist_absr", "days_since_earn",
       "big_yday", "an_now_n", "gap_days", "ret1", "intraday1"]
DIR = ["gap", "gap_z", "gap_rel_z", "pre_move", "post_ret", "mkt_gap", "ext_pos", "gap_rank",
       "ret1", "intraday1", "last_hour_z", "mkt_ret1", "earn_now"]
KS = [0.15, 0.20, 0.27, 0.34, 0.40]
FS = [0.35, 0.25, 0.20]


def gap_cuts(tr):
    """A 방식: |gap| 분위 후보 중 train score 최대인 a<b."""
    g, y = tr.gap.values, tr.label.values
    ok = ~np.isnan(g)
    qs = np.quantile(np.abs(g[ok]), [.5, .6, .7, .8, .85, .9, .95, .975, .99])
    best = max(((score(y[ok], rule(g[ok], a, b))["score"], a, b) for i, a in enumerate(qs) for b in qs[i + 1:]))
    return best[1], best[2]


def rule(g, a, b):
    o = np.full(len(g), 2)
    o[g >= a], o[g >= b], o[g <= -a], o[g <= -b] = 3, 4, 1, 0
    o[np.isnan(g)] = 2
    return o


def top_share(s, share, base_mask=None):
    """s 상위 share(전체 행 기준)를 True. base_mask 가 있으면 그 안에서만 고름."""
    s = np.where(np.isnan(s), -np.inf, s)
    if base_mask is not None:
        s = np.where(base_mask, s, -np.inf)
    k = int(round(share * len(s)))
    out = np.zeros(len(s), bool)
    if k > 0:
        out[np.argsort(-s)[:k]] = True
    return out & np.isfinite(s)


def split3(x, F, pi13):
    """x 를 하락/보합/상승으로. 보합 F, 나머지는 train 하락:상승 비율."""
    lo = (1 - F) * pi13[0]
    q1, q2 = np.nanquantile(x, [lo, lo + F])
    return np.where(x < q1, 1, np.where(x < q2, 2, 3))


def run():
    table, folds = load_table()
    unseen = set(unseen_symbols())
    rows = []
    for s in SEEDS:
        base_tp = train_predict(table, folds, B_FEATURES_V1, s)
        for f in folds:
            tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
            va = table[table.date.isin({d.date for d in f.val})]
            y_va = va.label.values
            u = va.symbol.isin(unseen).values
            pi = np.bincount(tr.label, minlength=5) / len(tr)
            pi13 = np.array([pi[1], pi[3]]) / (pi[1] + pi[3])

            def rec(name, K, F, pred):
                sc = score(y_va, pred)
                rows.append({"seed": s, "fold": f.name, "model": name, "K": K, "F": F, "all": sc["score"],
                             "unseen": score(y_va[u], pred[u])["score"], "big_recall": sc["big_recall"],
                             "big_prec": sc["big_prec"], "pred_big%": np.isin(pred, [0, 4]).mean() * 100})

            # 비교군
            a, b = gap_cuts(tr)
            g_rule = rule(va.gap.values, a, b)
            rec("갭 규칙", np.nan, np.nan, g_rule)
            p_tr, p_va, y_tr, _ = base_tp[f.name]
            rec("1단계 모델 (급등락 34%)", 0.34, np.nan, decode(p_tr, p_va, target_props(pi, 0.34, "train"), "E"))

            # 크기 모델, 방향 모델
            big_y = tr.label.isin([0, 4]).astype(int).values
            pb = lgb_binary(tr, MAG, big_y, s, 100).predict(va[MAG])
            pu = lgb_binary(tr, DIR, (tr.ret_pct > 0).astype(int).values, s, 300).predict(va[DIR])
            conf = np.abs(2 * pu - 1)
            gz = va.gap_z.values
            sgn_gap = np.sign(np.nan_to_num(va.gap.values))
            for K in KS:
                # T1: 규칙이 보합 아닌 종목 중 P(급등락) 상위
                nonflat = g_rule != 2
                big = top_share(pb, K, nonflat)
                t1 = np.where(nonflat, np.where(big, np.where(sgn_gap > 0, 4, 0), np.where(sgn_gap > 0, 3, 1)), 2)
                rec("T1 규칙 + 크기 모델", K, np.nan, t1)
                for F in FS:
                    # T2
                    big2 = top_share(pb * conf, K)
                    mid2 = split3(pu, F / (1 - K), pi13)
                    t2 = np.where(big2, np.where(pu > 0.5, 4, 0), mid2)
                    rec("T2 방향 모델 + 크기 모델", K, F, t2)
                    # T3
                    rk = pd.Series(np.abs(gz)).rank(pct=True).values
                    big3 = top_share(pb * rk, K)
                    mid3 = split3(np.nan_to_num(gz), F / (1 - K), pi13)
                    t3 = np.where(big3, np.where(sgn_gap > 0, 4, 0), mid3)
                    rec("T3 갭 방향 + 크기 × 갭 확신도", K, F, t3)
        print(f"  seed {s} 완료", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exp_twostage.csv", index=False)
    m = r.groupby(["model", "K", "F", "fold"], dropna=False)[["all", "unseen", "big_recall", "big_prec", "pred_big%"]].mean()
    w = m["all"].unstack("fold")
    w["fold1~3"] = w[["fold1", "fold2", "fold3"]].mean(axis=1)
    w["4폴드 평균"] = w[["fold1", "fold2", "fold3", "fold4"]].mean(axis=1)
    w = w.join(m.groupby(level=[0, 1, 2], dropna=False).mean()[["unseen", "big_recall", "big_prec", "pred_big%"]])
    pd.set_option("display.width", 250)
    best = w.sort_values("fold1~3", ascending=False).groupby(level=0, sort=False).head(1)
    print("\n== 구조별로 fold1~3 평균이 가장 좋은 설정 (시드 3개 평균) ==")
    print(best.round(4).to_string())
    w.to_csv(OUT / "exp_twostage_summary.csv")
    sd = r[r.model == "갭 규칙"].groupby("seed")["all"].mean().std()
    sd2 = r[r.model == "1단계 모델 (급등락 34%)"].groupby("seed")["all"].mean().std()
    print(f"\n시드 간 표준편차: 갭 규칙 {sd:.4f} (규칙이라 0), 1단계 모델 {sd2:.4f}")


if __name__ == "__main__":
    run()
