"""규칙 수정 후보 — 09 오답 분석에서 나온 아이디어를 4폴드로 시험.

    uv run python work/b/review/tweak/tweak.py

⚠️ 아이디어가 val 을 본 분석(09, 03, 06)에서 나왔으므로 같은 val 로 잰 점수는 낙관적임.
그래서 기준을 엄격하게 고정 (실행 전):
  - 4폴드 모두 상승 + 평균 +0.005 이상 + fold4 하락 없음 + 가짜 대조보다 큼
  - 통과해도 제출 규칙은 바꾸지 않고, 홀드아웃 1회 채점 후보로만

후보 (모든 파라미터는 폴드마다 train 으로만)
  M1 변동성 큰 종목은 급등락 경계를 높임      |s| ≥ b·exp(δ·z(vol20)),  δ ∈ {−0.3 … 0.3}     (09: vol20 상위 집중도 1.55)
  M2 경계를 부트스트랩 중앙값으로 (흔들림 줄이기) train 날짜를 30번 다시 뽑아 고른 a, b 의 중앙값       (09: 경계 갓 넘은 행 집중도 1.35)
  M3 시장 전체 갭이 큰 날은 출렁임 보정 끄기   |시장 평균 gap_z| ≥ train 날짜의 80 분위면 λ = 0  (03: 그런 날 보정이 손해)
  M4 λ 0.2 고정                                                                              (06)
  M5 = M2 + M4,  M6 = M3 + M4,  M7 = M1 + M4
  대조: M1 은 vol20 을 같은 날 안에서 섞은 가짜 5회, M3 는 같은 비율로 무작위로 고른 날 5회
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap", "work/b/review/batch"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402

OUT = HERE / "out"
SEED, N_BOOT, N_FAKE = 2026, 30, 5
DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


def s_of(x, P, lam=None, off=None):
    lam = P["lam"] if lam is None else lam
    s = G._score_s(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), P["med"], P["sd"], lam)
    if off is not None:                       # off 인 행은 λ = 0
        s0 = x.gap_z.to_numpy(float)
        s = np.where(off, s0, s)
    return s


def fit_ab_boot(s, y, dates, rng):
    ud = np.unique(dates)
    idx_by = {d: np.where(dates == d)[0] for d in ud}
    A, Bv = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(ud, len(ud))
        ii = np.concatenate([idx_by[d] for d in pick])
        _, a, b = G._fit_ab(s[ii], y[ii])
        A.append(a)
        Bv.append(b)
    return float(np.median(A)), float(np.median(Bv))


def size_cut(s, a, b, z, d):
    o = G.cut(s, a, np.inf)
    bb = b * np.exp(d * z)                    # d > 0 이면 z 가 큰 행(변동성 큰 종목)의 급등락 경계가 높아짐
    o[s >= bb], o[s <= -bb] = 4, 0
    o[np.isnan(s)] = 2
    return o


def fit_size(tr, va, col, P):
    y = tr.label.to_numpy(int)
    s_tr, s_va = s_of(tr, P), s_of(va, P)
    m, sd = np.nanmean(tr[col]), np.nanstd(tr[col])
    z_tr, z_va = np.nan_to_num(((tr[col] - m) / sd).to_numpy(float)), np.nan_to_num(((va[col] - m) / sd).to_numpy(float))
    q = np.quantile(np.abs(s_tr[~np.isnan(s_tr)]), G.QS)
    best = max(((G.score(y, size_cut(s_tr, a, b, z_tr, d)), a, b, d) for i, a in enumerate(q) for b in q[i + 1:]
                for d in DELTAS), key=lambda r: (round(r[0], 10), -abs(r[3])))
    _, a, b, d = best
    return size_cut(s_va, a, b, z_va, d), d


def mkt_off(tr, va, frac=None, rng=None):
    mk = pd.concat([tr, va]).groupby("date").gap_z.mean().abs()
    thr = mk[mk.index.isin(tr.date)].quantile(0.8)
    if frac is None:
        f = lambda x: x.date.map(mk >= thr).to_numpy(bool)  # noqa: E731
    else:                                      # 가짜: 같은 비율로 무작위 날
        days = mk.index.to_numpy()
        fake = pd.Series(rng.random(len(days)) < 0.2, index=days)
        f = lambda x: x.date.map(fake).to_numpy(bool)  # noqa: E731
    return f(tr), f(va)


def run():
    t, _ = E.load()
    folds = NG.folds_of(t)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, tr, va, _, _ in folds:
        y, yt = va.label.to_numpy(int), tr.label.to_numpy(int)
        u = va.unseen.to_numpy()
        P = G.fit_table(tr)
        P2 = E.fit_fixed(tr, 0.2)

        def rec(label, p, prm=""):
            rows.append({"fold": name, "후보": label, "score": G.score(y, p), "unseen": G.score(y[u], p[u]),
                         "급등락 예측 비율": float(np.isin(p, [0, 4]).mean()), "param": prm})

        rec("기본 규칙", G.predict_table(va, P), P["lam"])
        # M1, M7
        p, d = fit_size(tr, va, "vol20", P)
        rec("M1 변동성 큰 종목 경계 높이기", p, d)
        p, d = fit_size(tr, va, "vol20", P2)
        rec("M7 M1 + λ 0.2", p, d)
        # M2, M5
        for lab, PP in [("M2 경계 부트스트랩 중앙값", P), ("M5 M2 + λ 0.2", P2)]:
            a, b = fit_ab_boot(s_of(tr, PP), yt, tr.date.to_numpy(), rng)
            rec(lab, G.cut(s_of(va, PP), a, b), f"a {a:.3f} b {b:.3f} (원래 {PP['a']:.3f} {PP['b']:.3f})")
        # M3, M6
        off_tr, off_va = mkt_off(tr, va)
        for lab, PP in [("M3 시장 갭 큰 날 보정 끄기", P), ("M6 M3 + λ 0.2", P2)]:
            sc, a, b = G._fit_ab(s_of(tr, PP, off=off_tr), yt)
            rec(lab, G.cut(s_of(va, PP, off=off_va), a, b), f"off {off_va.mean():.2f}")
        rec("M4 λ 0.2 고정", G.predict_table(va, P2), 0.2)
        # 가짜
        for k in range(N_FAKE):
            trf, vaf = tr.copy(), va.copy()
            for df in (trf, vaf):
                df["vol_fake"] = df.groupby("date").vol20.transform(lambda s: rng.permutation(s.to_numpy()))
            p, d = fit_size(trf, vaf, "vol_fake", P)
            rec(f"가짜 M1-{k}", p, d)
            ftr, fva = mkt_off(tr, va, frac=0.2, rng=rng)
            sc, a, b = G._fit_ab(s_of(tr, P, off=ftr), yt)
            rec(f"가짜 M3-{k}", G.cut(s_of(va, P, off=fva), a, b), "")
        print(f"  {name} 끝", flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    r = run()
    base = r[r.후보 == "기본 규칙"].set_index("fold")
    r["delta"] = r.score - r.fold.map(base.score)
    r["d_unseen"] = r.unseen - r.fold.map(base.unseen)
    r.to_csv(OUT / "tweak.csv", index=False, encoding="utf-8-sig")
    fake = r[r.후보.str.startswith("가짜")].groupby("후보").delta.mean()
    f_m1 = fake[fake.index.str.contains("M1")].max()
    f_m3 = fake[fake.index.str.contains("M3")].max()
    out = []
    for c, g in r[~r.후보.str.startswith("가짜") & (r.후보 != "기본 규칙")].groupby("후보", sort=False):
        g = g.set_index("fold")
        ref = f_m1 if "M1" in c or "M7" in c else (f_m3 if "M3" in c or "M6" in c else 0.0)
        ok = bool((g.delta > 0).all() and g.delta.mean() >= 0.005 and g.delta.mean() > ref)
        out.append({"후보": c, **{f: round(g.delta[f], 4) for f in ["fold1", "fold2", "fold3", "fold4"]},
                    "평균": round(g.delta.mean(), 4), "처음 보는 종목": round(g.d_unseen.mean(), 4),
                    "급등락 예측 비율": round(g["급등락 예측 비율"].mean(), 3), "가짜 최대": round(ref, 4),
                    "통과": ok, "param": " / ".join(str(x) for x in g.param)})
    j = pd.DataFrame(out)
    j.to_csv(OUT / "judge.csv", index=False, encoding="utf-8-sig")
    print("기본 규칙", base.score.round(4).to_dict(), "급등락 예측 비율", round(base["급등락 예측 비율"].mean(), 3))
    print(j.drop(columns=["param"]).to_string(index=False))
    print("\n파라미터\n", j[["후보", "param"]].to_string(index=False))
    print("\n가짜 평균 차이", fake.round(4).to_dict())


if __name__ == "__main__":
    main()
