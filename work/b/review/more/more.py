"""규칙 자체를 바꿔 보는 실험 — 분모, 갭 측정, 디코딩, 학습 방식, 달력.

    uv run python work/b/review/more/more.py

사전 등록 (실행 전에 고정, 결과를 보고 바꾸지 않음)
----------------------------------------------------
기본: 제출 규칙 (gap_z + ext_range_z, λ·a·b 를 train 점수 최대로). 모든 파라미터는 폴드마다 train 으로만.

V 변동성 분모 (gap_z 와 ext_range_z 의 분모를 같이 바꿈, 나머지 절차 같음)
  V1 EWMA 변동성 (λ_ewm 0.94, 최근 수익률에 더 무게)
  V2 Parkinson 20일 변동성 (일봉 고가·저가)
  V3 vol20 과 V1 의 평균
P 갭의 "현재 가격" (지금: 대상일 08:00 봉 종가)
  P1 마지막 시간외 봉 2개 종가 평균
  P2 시간외 봉 전체 종가의 중앙값
  P3 장전(pre) 봉 종가 평균
D 디코딩 (s 는 기본 규칙 그대로)
  D1 점수식 최적 디코딩: s 를 train 분위 20칸으로 나누고, 칸마다 정답 분포로
     "기대 벌점 − ρ × 기대 분모"가 가장 작은 등급을 고름 (ρ = 점수 비율, train 에서 반복 갱신. Dinkelbach)
  D2 날마다 순위로 자르기: 그날 종목들의 |s| 순위 백분위로 경계 (두 경계 백분위를 train 점수 최대로)
L 학습 방식
  L1 최근 가중: train 행 무게 0.5^(경과일/120) 로 가중 점수를 최대로 (λ·a·b)
  L2 λ 평균: s = 평균_{λ∈0,0.1,0.2,0.3,0.4} gap_z·exp(−λ·x̃), a·b 만 고름
C 달력 (그날 급등락 경계를 이동, 형태 S, δ ∈ −0.3…0.3)
  C1 옵션 만기일 (대상일이 그 달 셋째 금요일)
  C2 월말·월초 (대상일이 그 달 마지막 또는 첫 거래일)
  C3 연휴 전날 (대상일 다음 거래일까지 평일이 하루 이상 비는 날)
  대조: 같은 비율로 무작위로 고른 날 5회
판정: 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음 (C 는 + 가짜 최대보다 큼)
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402

OUT = HERE / "out"
SEED, N_FAKE = 2026, 5
DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


# ------------------------------------------------------------------ 데이터
def build(t):
    d = pd.read_parquet(ROOT / "dataset" / "daily.parquet", columns=["symbol", "date_et", "high", "low", "close", "ret"])
    d = d.sort_values(["symbol", "date_et"])
    g = d.groupby("symbol")
    d["vol_ewm"] = g.ret.transform(lambda s: s.ewm(alpha=0.06, min_periods=20).std())
    d["pk"] = np.log(d.high / d.low) ** 2 / (4 * np.log(2))
    d["vol_pk"] = np.sqrt(g.pk.transform(lambda s: s.rolling(20, min_periods=10).mean()))
    d = d.rename(columns={"date_et": "date", "close": "prev_close"})[["symbol", "date", "vol_ewm", "vol_pk", "prev_close"]]
    t = t.merge(d, on=["symbol", "date"], how="left")
    t["vol_mix"] = (t.vol20 + t.vol_ewm) / 2

    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"], w["cutoff"] = w.date + pd.Timedelta(hours=16), w.target + pd.Timedelta(hours=9, minutes=30)
    con = duckdb.connect()
    con.register("w", w)
    p = con.execute(f"""
        SELECT w.date, p.symbol, p.datetime, p.session, p.close
        FROM read_parquet('{ROOT / "dataset" / "price.parquet"}') p
        JOIN w ON p.datetime >= w.since AND p.known_at < w.cutoff
        WHERE p.session IN ('post', 'pre')""").df().sort_values("datetime")
    con.close()
    gp = p.groupby(["date", "symbol"]).close
    px = pd.DataFrame({"px_last": gp.last(), "px_last2": gp.apply(lambda s: s.iloc[-2:].mean()),
                       "px_med": gp.median()})
    px = px.join(p[p.session == "pre"].groupby(["date", "symbol"]).close.mean().rename("px_pre")).reset_index()
    t = t.merge(px, on=["date", "symbol"], how="left")
    t["gap_raw"] = t.ext_range_z * 0 + (t.px_last / t.prev_close - 1)
    t["range_raw"] = t.ext_range_z * t.vol20            # (최고 − 최저) ÷ 전일 종가

    # 달력 (대상일 기준)
    dates = pd.DatetimeIndex(sorted(t.target.dropna().unique()))
    cal = pd.DataFrame({"target": dates})
    cal["opex"] = (cal.target.dt.weekday == 4) & cal.target.dt.day.between(15, 21)
    ym = cal.target.dt.to_period("M")
    cal["month_edge"] = (ym != ym.shift(-1)) | (ym != ym.shift(1))
    nxt = cal.target.shift(-1)
    gapdays = (nxt - cal.target).dt.days
    cal["pre_holiday"] = ((cal.target.dt.weekday < 4) & (gapdays > 1)) | ((cal.target.dt.weekday == 4) & (gapdays > 3))
    t = t.merge(cal, on="target", how="left")
    for c in ["opex", "month_edge", "pre_holiday"]:
        t[c] = t[c].fillna(False).astype(float)
    return t


def check(t):
    d = (t.gap_raw / t.vol20 - t.gap_z).abs()
    print(f"갭 재계산 확인: 최대 차이 {d.max():.2e}")


# ------------------------------------------------------------------ 변형들
def variant_table(x, vol=None, price=None):
    """gap_z, ext_range_z 를 다른 분모·가격으로 다시 만든 표."""
    x = x.copy()
    v = x[vol] if vol else x.vol20
    gap = x[price] / x.prev_close - 1 if price else x.gap_raw
    x["gap_z"] = gap / v
    x["ext_range_z"] = x.range_raw / v
    return x


def wscore(y, p, w):
    o = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), weights=w, minlength=25).reshape(5, 5)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (G.WEIGHT * o).sum() / (G.WEIGHT * e).sum()


def fit_weighted(tr, half=120):
    age = (tr.date.max() - tr.date).dt.days.to_numpy()
    w = 0.5 ** (age / half)
    gz, er, y = tr.gap_z.to_numpy(float), tr.ext_range_z.to_numpy(float), tr.label.to_numpy(int)
    med, sd = float(np.nanmedian(er)), float(np.nanstd(er))
    best = None
    for lam in G.LAMS:
        s = G._score_s(gz, er, med, sd, lam)
        ok = ~np.isnan(s)
        q = np.quantile(np.abs(s[ok]), G.QS)
        for i, a in enumerate(q):
            for b in q[i + 1:]:
                sc = wscore(y[ok], G.cut(s[ok], a, b), w[ok])
                if best is None or sc > best[0]:
                    best = (sc, lam, a, b)
    _, lam, a, b = best
    return {"lam": lam, "med": med, "sd": sd, "a": a, "b": b}


def s_base(x, P, lam=None):
    return G._score_s(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), P["med"], P["sd"],
                      P["lam"] if lam is None else lam)


def bayes_decode(s_tr, y, s_va, nbins=20, iters=20):
    ok = ~np.isnan(s_tr)
    edges = np.quantile(s_tr[ok], np.linspace(0, 1, nbins + 1)[1:-1])
    btr, bva = np.searchsorted(edges, s_tr[ok]), np.searchsorted(edges, s_va)
    yt = y[ok]
    cnt = np.zeros((nbins, 5))
    np.add.at(cnt, (btr, yt), 1)
    q = np.bincount(yt, minlength=5) / len(yt)
    rho, lab = 0.5, np.full(nbins, 2)
    for _ in range(iters):
        num = cnt @ G.WEIGHT                 # 칸 b 에서 j 로 찍을 때 분자 기여 (행 수 가중)
        den = cnt.sum(1, keepdims=True) * (q @ G.WEIGHT)[None, :]
        lab = np.argmin(num - rho * den, axis=1)
        p = lab[btr]
        new = 1 - G.score(yt, p)
        if abs(new - rho) < 1e-6:
            break
        rho = new
    out = lab[bva]
    out[np.isnan(s_va)] = 2
    return out, lab


def daily_rank_fit(tr, va, s_tr, s_va):
    grid = [.2, .3, .4, .5, .6, .7, .8, .9, .95]
    def lab(df, s, pa, pb):
        r = pd.Series(np.abs(s)).groupby(df.date.to_numpy()).rank(pct=True).to_numpy()
        o = np.full(len(s), 2)
        o[(r >= pa) & (s > 0)], o[(r >= pa) & (s < 0)] = 3, 1
        o[(r >= pb) & (s > 0)], o[(r >= pb) & (s < 0)] = 4, 0
        o[np.isnan(s)] = 2
        return o
    y = tr.label.to_numpy(int)
    best = max(((G.score(y, lab(tr, s_tr, pa, pb)), pa, pb) for i, pa in enumerate(grid) for pb in grid[i + 1:]))
    return lab(va, s_va, best[1], best[2]), (best[1], best[2])


def size_fit(tr, va, s_tr, s_va, col):
    y = tr.label.to_numpy(int)
    m, sd = tr[col].mean(), tr[col].std()
    z_tr = np.nan_to_num(((tr[col] - m) / sd).to_numpy(float)) if sd > 0 else np.zeros(len(tr))
    z_va = np.nan_to_num(((va[col] - m) / sd).to_numpy(float)) if sd > 0 else np.zeros(len(va))
    q = np.quantile(np.abs(s_tr[~np.isnan(s_tr)]), G.QS)

    def cut(s, a, b, z, d):
        o = G.cut(s, a, np.inf)
        bb = b * np.exp(-d * z)
        o[s >= bb], o[s <= -bb] = 4, 0
        o[np.isnan(s)] = 2
        return o
    best = max(((G.score(y, cut(s_tr, a, b, z_tr, d)), a, b, d) for i, a in enumerate(q) for b in q[i + 1:] for d in DELTAS),
               key=lambda r: (round(r[0], 10), -abs(r[3])))
    _, a, b, d = best
    return cut(s_va, a, b, z_va, d), d


# ------------------------------------------------------------------ 실행
def run(t):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        y, u = va.label.to_numpy(int), va.unseen.to_numpy()
        yt = tr.label.to_numpy(int)

        def rec(lab, p, prm=""):
            rows.append({"fold": name, "실험": lab, "score": G.score(y, p), "unseen": G.score(y[u], p[u]),
                         "급등락비율": float(np.isin(p, [0, 4]).mean()), "param": prm})

        P = G.fit_table(tr)
        rec("기본 규칙", G.predict_table(va, P), P["lam"])
        for lab, vol in [("V1 EWMA 변동성", "vol_ewm"), ("V2 Parkinson 변동성", "vol_pk"), ("V3 vol20·EWMA 평균", "vol_mix")]:
            tr2, va2 = variant_table(tr, vol=vol), variant_table(va, vol=vol)
            P2 = G.fit_table(tr2)
            rec(lab, G.predict_table(va2, P2), P2["lam"])
        for lab, px in [("P1 마지막 2봉 평균 가격", "px_last2"), ("P2 시간외 중앙값 가격", "px_med"), ("P3 장전 봉 평균 가격", "px_pre")]:
            tr2, va2 = variant_table(tr, price=px), variant_table(va, price=px)
            P2 = G.fit_table(tr2)
            rec(lab, G.predict_table(va2, P2), P2["lam"])
        s_tr, s_va = s_base(tr, P), s_base(va, P)
        p, labs = bayes_decode(s_tr, yt, s_va)
        rec("D1 점수식 최적 디코딩", p, "".join(map(str, labs)))
        p, prm = daily_rank_fit(tr, va, s_tr, s_va)
        rec("D2 날마다 순위로 자르기", p, prm)
        Pw = fit_weighted(tr)
        rec("L1 최근 가중 (반감기 120일)", G.predict_table(va, Pw), Pw["lam"])
        s_avg_tr = np.mean([s_base(tr, P, lam) for lam in [0, .1, .2, .3, .4]], axis=0)
        s_avg_va = np.mean([s_base(va, P, lam) for lam in [0, .1, .2, .3, .4]], axis=0)
        _, a, b = G._fit_ab(s_avg_tr, yt)
        rec("L2 λ 평균", G.cut(s_avg_va, a, b), "")
        for lab, col in [("C1 옵션 만기일", "opex"), ("C2 월말·월초", "month_edge"), ("C3 연휴 전날", "pre_holiday")]:
            p, d = size_fit(tr, va, s_tr, s_va, col)
            rec(lab, p, d)
            rate = pd.concat([tr, va]).groupby("date")[col].first().mean()
            for k in range(N_FAKE):
                days = pd.concat([tr, va]).date.unique()
                fake = pd.Series(rng.random(len(days)) < rate, index=days).astype(float)
                trf, vaf = tr.assign(fk=tr.date.map(fake)), va.assign(fk=va.date.map(fake))
                p, d = size_fit(trf, vaf, s_tr, s_va, "fk")
                rec(f"가짜 {lab[:2]}-{k}", p, d)
        print(f"  {name} 끝", flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    t, _ = E.load()
    t = build(t)
    check(t)
    r = run(t)
    base = r[r.실험 == "기본 규칙"].set_index("fold")
    r["delta"] = r.score - r.fold.map(base.score)
    r["d_unseen"] = r.unseen - r.fold.map(base.unseen)
    r.to_csv(OUT / "more.csv", index=False, encoding="utf-8-sig")
    fake = r[r.실험.str.startswith("가짜")].groupby("실험").delta.mean()
    out = []
    for c, g in r[~r.실험.str.startswith("가짜") & (r.실험 != "기본 규칙")].groupby("실험", sort=False):
        g = g.set_index("fold")
        ref = fake[fake.index.str.contains(c[:2])].max() if c.startswith("C") else -np.inf
        ok = bool(g.delta.mean() >= 0.005 and (g.delta > 0).sum() >= 3 and g.delta["fold4"] >= 0 and g.delta.mean() > ref)
        out.append({"실험": c, **{f: round(g.delta[f], 4) for f in ["fold1", "fold2", "fold3", "fold4"]},
                    "평균": round(g.delta.mean(), 4), "처음 보는 종목": round(g.d_unseen.mean(), 4),
                    "급등락비율": round(g.급등락비율.mean(), 3), "가짜 최대": None if ref == -np.inf else round(ref, 4),
                    "채택": ok, "param": " / ".join(str(x) for x in g.param)})
    j = pd.DataFrame(out)
    j.to_csv(OUT / "judge.csv", index=False, encoding="utf-8-sig")
    print("기본", base.score.round(4).to_dict(), "급등락비율", round(base.급등락비율.mean(), 3))
    print(j.drop(columns=["param"]).to_string(index=False))
    print("\n파라미터\n", j[["실험", "param"]].to_string(index=False))


if __name__ == "__main__":
    main()
