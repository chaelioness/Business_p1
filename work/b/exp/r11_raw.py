"""11차: 원본에서 아직 안 쓴 정보 — 전날 장중 흐름·거래량, 애널리스트 세부, 실적 세부. A 폴드 기준.

    uv run python work/b/exp/r11_raw.py

새 피처 (guard 아래에서 만듦, cache/b_v4.parquet)
  전날 정규장 시간봉
    am_ret_z     전날 오전(12:30 전 봉) 수익률 ÷ vol20
    pm_ret_z     전날 오후(12:30 이후 봉) 수익률 ÷ vol20
    pm_vol_rel   전날 마지막 2시간 거래량 비중 − 이 종목 지난 20일 평균 비중
    buy_press    Σ(봉 수익률 부호 × 거래량) ÷ Σ거래량
    vwap_dev_z   (종가 − 거래량 가중 평균가) ÷ 종가 ÷ vol20
    close_loc    전날 고저 범위 안 종가 위치 (0~1)
  애널리스트
    an_init30, an_reit30   최근 30일 신규 커버리지 / 재확인 수
    an_firms30             최근 30일 의견 낸 증권사 수
    an_init_now            밤사이(기준일 16:00 이후) 신규 커버리지 수
    an_now_tgtchg          밤사이 목표가 변화율 평균
    an_init_tgt_gap        최근 90일 신규 커버리지 목표가 중앙값 ÷ 종가 − 1
  실적
    beat_streak   연속으로 예상을 웃돈 횟수 (기준일 16:00 전 발표까지)
    eps_yoy       최근 발표 EPS vs 4번 전 발표 EPS 증가 여부 (+1/−1)
평가: gap_z 규칙에 더하기/곱하기/같은방향 (β 는 train), 혼자 규칙 점수, 값을 섞은 가짜 3번.
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common  # noqa: E402,F401  (경로 설정)

from lab.cv import feature_table  # noqa: E402
from lab.folds import unseen_symbols  # noqa: E402
from common import BETAS, OUT, load_table, one, prep  # noqa: E402
from r10b_solo import solo  # noqa: E402
from builders import NEW, build_raw  # noqa: E402


def run():
    table, folds = load_table()
    days = sorted({d.date: d for f in folds for d in (*f.train, *f.val)}.values(), key=lambda d: d.date)
    c = feature_table(days, build_raw, name="b_v4")
    table = table.merge(c.drop(columns=["label", "ret_pct"]), on=["date", "symbol"], how="left")
    for k in NEW:
        table[k] = pd.to_numeric(table[k], errors="coerce").replace([np.inf, -np.inf], np.nan)
    print("결측 비율:", table[NEW].isna().mean().round(3).to_dict())
    print("0 이 아닌 비율:", (table[NEW].fillna(0) != 0).mean().round(3).to_dict(), flush=True)
    unseen = set(unseen_symbols())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    # gap_z 에 붙이기
    data = prep(table, folds, NEW, unseen)
    fs = [c for c in NEW if all(c in d["x"] for d in data.values())]
    res = Parallel(n_jobs=-1)(delayed(one)(c, form, data) for c in fs for form in BETAS)
    rr = pd.DataFrame([row for p in res for row in p])
    b0 = rr[(rr.beta == 0) & (rr["방식"] == "더하기")].groupby("fold")["all"].mean()
    ga = rr.loc[rr.groupby(["피처", "방식", "fold"])["train"].idxmax()]
    da = ga.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(b0, axis=1)
    bt = ga.pivot_table(index=["피처", "방식"], columns="fold", values="beta")
    o = pd.DataFrame({"β": bt.apply(lambda s: "/".join(f"{v:g}" for v in s), axis=1),
                      "Δ평균": da.mean(axis=1), "Δfold4": da["fold4"], "상승폴드": (da > 0).sum(axis=1)})
    o["채택"] = (o["Δ평균"] >= 0.005) & (o["상승폴드"] >= 3) & (o["Δfold4"] >= 0)
    print(f"\n== gap_z 규칙({b0.mean():.4f}) + 새 피처 {len(fs)}개 × 3방식 ==")
    print(o.sort_values("Δ평균", ascending=False).head(20).round(4).to_string())
    print(f"채택 {o['채택'].sum()}개")
    o.to_csv(OUT / "exp_raw_gapz.csv")

    # 혼자
    sd = {}
    for f in folds:
        tr = table[table.date.isin({d.date for d in f.train}) & ~table.symbol.isin(unseen)]
        va = table[table.date.isin({d.date for d in f.val})]
        sd[f.name] = {c: (tr[c].values.astype(float), tr.label.values, va[c].values.astype(float), va.label.values) for c in fs}
    so = Parallel(n_jobs=-1)(delayed(solo)(c, {fn: d[c] for fn, d in sd.items()}) for c in fs)
    s = pd.DataFrame({c: {"혼자 규칙 4폴드": np.mean([v[0] for v in o_.values()]),
                          "부호": "/".join("+" if v[1] > 0 else "−" for v in o_.values())} for c, o_ in so}).T
    print("\n== 혼자 규칙 (gap_z 없이) ==")
    print(s.sort_values("혼자 규칙 4폴드", ascending=False).to_string())

    # 섞은 가짜
    rng = np.random.default_rng(2)
    for rep in range(3):
        sh = table.copy()
        for c in fs:
            sh[c] = rng.permutation(sh[c].values)
        dsh = prep(sh, folds, fs, unseen)
        q = pd.DataFrame([row for p in Parallel(n_jobs=-1)(delayed(one)(c, form, dsh) for c in fs for form in BETAS) for row in p])
        g2 = q.loc[q.groupby(["피처", "방식", "fold"])["train"].idxmax()]
        d2 = g2.pivot_table(index=["피처", "방식"], columns="fold", values="all").sub(b0, axis=1)
        ok = (d2.mean(axis=1) >= 0.005) & ((d2 > 0).sum(axis=1) >= 3) & (d2["fold4"] >= 0)
        print(f"섞은 가짜 {rep + 1}회차: 통과 {ok.sum()}개 / {len(ok)}, Δ평균 최대 {d2.mean(axis=1).max():.4f}", flush=True)


if __name__ == "__main__":
    run()
