"""홀드아웃 기준일 후보 비교. 정답(label) 분포와 실적 일정만 봄. feature-정답 관계는 안 봄."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "dataset"
START = pd.Timestamp("2024-08-13")
CUTS = (-2.5, -0.6, 0.6, 2.5)
L = np.arange(5)
W = ((L[:, None] - L[None, :]) ** 2) * (np.abs(L - 2)[:, None] ** 2)

d = pd.read_parquet(DATA / "daily.parquet").dropna(subset=["ret"])
d["ret_pct"] = d["ret"] * 100
d["label"] = pd.cut(d.ret_pct, [-np.inf, *CUTS, np.inf], right=False, labels=False).astype(int)
d = d.rename(columns={"date_et": "date"})
d = d[d.date > START]            # target 기준 (기준일 08-13 의 target 은 그 다음 거래일)
dates = np.array(sorted(d.date.unique()))
print(f"target 일수 {len(dates)}  {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}  종목 {d.symbol.nunique()}")

# ---------------------------------------------------------------- 월별
d["big"] = d.label.isin([0, 4])
m = d.groupby(d.date.dt.to_period("M")).agg(
    days=("date", "nunique"), n=("label", "size"),
    crash=("label", lambda s: (s == 0).mean() * 100),
    down=("label", lambda s: (s == 1).mean() * 100),
    flat=("label", lambda s: (s == 2).mean() * 100),
    up=("label", lambda s: (s == 3).mean() * 100),
    surge=("label", lambda s: (s == 4).mean() * 100),
    big=("big", lambda s: s.mean() * 100),
    abs_ret=("ret_pct", lambda s: s.abs().mean()),
    mkt=("ret_pct", "mean"))
print("\n[월별 label 분포 %]")
print(m.round(1).to_string())

# 실적 발표 (월별 건수)
e = pd.read_parquet(DATA / "earnings.parquet")
e = e[e.known_at >= START]
print("\n[월별 실적 발표 건수]")
print(e.groupby(e.known_at.dt.to_period("M")).size().to_string())


# ---------------------------------------------------------------- 점수 잡음
def score(t, p):
    o = np.zeros((5, 5))
    np.add.at(o, (t, p), 1)
    ex = np.outer(o.sum(1), o.sum(0)) / len(t)
    den = (W * ex).sum()
    return 1 - (W * o).sum() / den if den else np.nan


rng = np.random.default_rng(0)
prior = np.bincount(d.label, minlength=5) / len(d)


def noisy_pred(t, skill):
    """skill 확률로 정답, 아니면 전체 분포에서 무작위. 실력이 고정된 가상의 모델."""
    hit = rng.random(len(t)) < skill
    return np.where(hit, t, rng.choice(5, size=len(t), p=prior))


by_day = {k: g.label.to_numpy() for k, g in d.groupby("date")}


def window_scores(ds_, skill, reps=300):
    t = np.concatenate([by_day[x] for x in ds_])
    return np.array([score(t, noisy_pred(t, skill)) for _ in range(reps)])


print("\n[홀드아웃 후보]  skill=0.10 인 가상 모델의 score 평균±표준편차 (같은 실력인데 구간만 다름)")
rows = []
for on in ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01",
           "2026-07-01", "2026-08-01"]:
    on_ts = pd.Timestamp(on)
    te = dates[dates >= on_ts]
    tr = dates[dates < on_ts]
    sub = d[d.date >= on_ts]
    s = window_scores(te, 0.10)
    ne = ((e.known_at >= on_ts)).sum()
    rows.append(dict(split=on, train_days=len(tr), test_days=len(te), n=len(sub),
                     big_n=int(sub.big.sum()), big_pct=round(sub.big.mean() * 100, 1),
                     crash_pct=round((sub.label == 0).mean() * 100, 1),
                     surge_pct=round((sub.label == 4).mean() * 100, 1),
                     earnings=int(ne), score_mean=round(s.mean(), 3), score_sd=round(s.std(), 3)))
print(pd.DataFrame(rows).to_string(index=False))

# 구간 길이별 잡음: 길이 k 거래일 창을 전 구간에서 굴렸을 때
print("\n[창 길이별] 같은 실력(skill=0.10) 모델 score 의 창 간 표준편차 + 창 안 잡음")
for k in [20, 40, 60, 73, 90, 120]:
    starts = range(0, len(dates) - k + 1, max(1, k // 4))
    means, sds = [], []
    for s0 in starts:
        sc = window_scores(dates[s0:s0 + k], 0.10, reps=60)
        means.append(sc.mean()); sds.append(sc.std())
    print(f"  {k:>3}일  창 안 sd {np.mean(sds):.3f}   창 사이 sd {np.std(means):.3f}"
          f"   big% 범위 {min(d[d.date.isin(dates[s:s+k])].big.mean()*100 for s in starts):.1f}"
          f"~{max(d[d.date.isin(dates[s:s+k])].big.mean()*100 for s in starts):.1f}")

# ---------------------------------------------------------------- 종목별 일봉 시작
first = pd.read_parquet(DATA / "daily.parquet", columns=["symbol", "date_et"]).groupby("symbol").date_et.min()
print("\n[종목별 일봉 시작일 분포]")
print(first.value_counts().sort_index().to_string())
sj = DATA / "symbols.json"
if sj.exists():
    print("\nsymbols.json:", sj.read_text()[:1500])
