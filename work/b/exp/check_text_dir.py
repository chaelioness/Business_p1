"""텍스트(LLM·감성) 방향 피처가 '갭이 모르는 방향'을 맞히는지 확인.

    uv run python work/b/exp/check_text_dir.py <피처 파일> --dir <방향 열> [옵션]

    예) uv run python work/b/exp/check_text_dir.py work/c/llm_features.parquet --dir llm_dir
        uv run python work/b/exp/check_text_dir.py feats.csv --dir sentiment --where "relevant == 1" --mag llm_big

입력 파일: csv 또는 parquet. symbol 열 + 날짜 열 하나 (target = 예측 대상일, 또는 date = 기준일).
  --dir   방향 열. 숫자(양수 = 좋음)나 글자(좋음/나쁨/중립, good/bad/neutral, positive/negative,
          up/down, bullish/bearish) 모두 받음. 0·중립·결측은 '판단 없음'으로 빼고 셈.
  --where pandas query 조건 (여러 번 가능). 예: "relevant == 1", "novel == 'new'"
  --mag   크기 열 (선택). 남은 움직임 크기·급등락과의 관계를 같이 봄.
  --period train(기본) | all. train = fold4 train 대상일까지(~2026-02-13). 판정은 train 으로만 하고
          val 기간은 run_cv 로 확인 (val 을 보고 기준을 고치지 않기 위함).

무엇과 비교하나: 남은 움직임 = 9시 시간외 가격 → 당일 종가 (갭이 모르는 부분).
하루 수익률과 비교하면 갭이 이미 아는 부분 때문에 좋아 보이므로 참고로만 같이 출력.
판정: 부호 일치율 ≥ 0.55 이고 날짜 묶음 부트스트랩 95% 하한 > 0.5 이고 섞은 기준선 p < 0.05 → 쓸 만함.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C  # noqa: F401  (경로 설정)
from common import fold_days, load_table  # noqa: E402

WORDS = {1: ["좋음", "호재", "긍정", "good", "positive", "pos", "up", "bullish", "buy", "beat"],
         -1: ["나쁨", "악재", "부정", "bad", "negative", "neg", "down", "bearish", "sell", "miss"],
         0: ["중립", "무관", "neutral", "none", "mixed", "unrelated", "na", "n/a", ""]}
RULE = (0.55, 0.5, 0.05)   # 일치율, 부트스트랩 하한, 섞은 기준선 p


def to_sign(s):
    if pd.api.types.is_numeric_dtype(s):
        return np.sign(s.astype(float))
    m = {w: k for k, ws in WORDS.items() for w in ws}
    low = s.astype(str).str.strip().str.lower()
    out = low.map(m)
    bad = sorted(low[out.isna() & s.notna()].unique())[:10]
    if bad:
        print(f"  ⚠ 방향으로 못 읽은 값 (판단 없음 처리): {bad}")
    return out.astype(float)


def read(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def boot_ci(df, n=500, seed=0):
    """날짜 묶음 부트스트랩: 같은 날의 종목들은 같이 움직이므로 날짜 단위로 다시 뽑음."""
    g = df.groupby("target").agg(k=("agree", "sum"), n=("agree", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    k, m = g.k.values[idx].sum(1), g.n.values[idx].sum(1)
    return np.quantile(k / m, [0.025, 0.975])


def shuffle_p(df, n=500, seed=0):
    """같은 날 안에서 방향 값을 섞어도 이만큼 맞는지 (날짜 효과·쏠림 제거)."""
    rng = np.random.default_rng(seed)
    obs = df.agree.mean()
    groups = list(df.groupby("target").indices.values())     # 위치 (df 는 reset_index 된 상태)
    d, r = df.d.values, df.r_sign.values
    hits = 0
    for _ in range(n):
        dd = d.copy()
        for ix in groups:
            dd[ix] = rng.permutation(dd[ix])
        hits += (np.mean(dd == r) >= obs)
    return (hits + 1) / (n + 1)


def date_ic(df, x, y):
    r = df[["target", x, y]].dropna()
    r = r[r.groupby("target")[x].transform("nunique") > 1]
    ic = r.groupby("target").apply(lambda d: d[x].rank().corr(d[y].rank()), include_groups=False).dropna()
    if len(ic) < 5:
        return np.nan, np.nan, len(ic)
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), len(ic)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--where", action="append", default=[])
    ap.add_argument("--mag")
    ap.add_argument("--period", choices=["train", "all"], default="train")
    a = ap.parse_args()

    t, folds = load_table()
    tmap = {d.date: d.target for d in fold_days(folds)}
    t["target"] = t.date.map(tmap)
    t["resid"] = ((1 + t.ret_pct / 100) / (1 + t.gap) - 1) * 100      # 9시 가격 → 종가 (%)
    t["big"] = t.label.isin([0, 4])
    t = t.dropna(subset=["gap", "target"])

    f = read(a.file)
    key = "target" if "target" in f.columns else ("date" if "date" in f.columns else None)
    if key is None or "symbol" not in f.columns:
        sys.exit(f"파일에 symbol 과 target(또는 date) 열이 있어야 함. 지금 열: {list(f.columns)[:15]}")
    f[key] = pd.to_datetime(f[key]).dt.normalize()
    for q in a.where:
        f = f.query(q)
    cols = [c for c in {a.dir, a.mag} if c]
    m = t.merge(f[["symbol", key, *cols]].drop_duplicates(["symbol", key]), on=["symbol", key], how="inner")
    if a.period == "train":
        end = max(d.target for d in folds[-1].train)
        m = m[m.target <= end]
    print(f"파일 {len(f):,}행 → 정형 표와 맞춘 행 {len(m):,} (기간 {a.period}: {m.target.min():%Y-%m-%d} ~ {m.target.max():%Y-%m-%d})")
    if not len(m):
        sys.exit("맞춘 행이 없음 — 날짜 열이 target(대상일)인지 date(기준일)인지 확인")

    m["d"] = to_sign(m[a.dir])
    m["r_sign"] = np.sign(m.resid)
    m["g_sign"] = np.sign(m.gap)
    use = m[(m.d != 0) & m.d.notna() & (m.r_sign != 0)].reset_index(drop=True)
    print(f"판단 있음(좋음/나쁨) {len(use):,}행 = {len(use) / len(m):.1%}, 좋음 비율 {(use.d > 0).mean():.1%}\n")
    if len(use) < 100:
        print("⚠ 판단 있는 행이 100개 미만 — 결과가 크게 흔들릴 수 있음")

    use["agree"] = use.d == use.r_sign
    lo, hi = boot_ci(use)
    p = shuffle_p(use)
    rate = use.agree.mean()
    day_rate = (use.d == np.sign(use.ret_pct)).mean()
    gap_same = (use.d == use.g_sign).mean()
    print("== 방향 ==")
    print(f"  남은 움직임(9시→종가)과 부호 일치  {rate:.3f}  [95% {lo:.3f} ~ {hi:.3f}]  섞은 기준선 p={p:.3f}")
    print(f"  (참고) 하루 수익률과 부호 일치     {day_rate:.3f}  ← 갭이 아는 부분이 섞여 부풀려짐")
    print(f"  (참고) 갭 방향과 같은 비율          {gap_same:.3f}  ← 높으면 갭을 다시 말하는 것뿐일 수 있음")
    if pd.api.types.is_numeric_dtype(m[a.dir]):
        ic, tv, nd = date_ic(m, a.dir, "resid")
        print(f"  날짜별 순위상관 (점수 → 남은 움직임) {ic:+.4f}  (t {tv:+.2f}, {nd}일)")

    print("\n== 갭 크기별 (남은 움직임과 부호 일치) ==")
    use["갭크기"] = pd.cut(use.gap_z.abs(), [0, 0.5, 1, 2, np.inf], labels=["<0.5", "0.5~1", "1~2", "2+"], include_lowest=True)
    use["갭과"] = np.where(use.d == use.g_sign, "갭과 같은 방향", "갭과 반대")
    print(use.groupby("갭크기", observed=True).agree.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "일치율"}).round(3).to_string())
    print(use.groupby("갭과").agree.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "일치율"}).round(3).to_string())

    if a.mag:
        mm = m.dropna(subset=[a.mag])
        ic, tv, nd = date_ic(mm.assign(absr=mm.resid.abs()), a.mag, "absr")
        q = pd.qcut(mm[a.mag].rank(method="first"), 3, labels=["낮음", "중간", "높음"])
        print(f"\n== 크기 ({a.mag}) ==\n  날짜별 순위상관 (→ |남은 움직임|) {ic:+.4f} (t {tv:+.2f})")
        print(mm.groupby(q, observed=True).agg(n=("big", "size"), 급등락pct=("big", lambda s: s.mean() * 100),
                                               남은움직임크기=("resid", lambda s: s.abs().mean())).round(2).to_string())

    ok = rate >= RULE[0] and lo > RULE[1] and p < RULE[2]
    print(f"\n판정: {'✅ 쓸 만함 → run_cv 로 B 규칙 위에 얹어 W&B 에서 vs_gapz_ext 확인' if ok else '❌ 갭이 모르는 방향 정보로 보기 어려움'}"
          f"  (기준: 일치율 ≥ {RULE[0]}, 95% 하한 > {RULE[1]}, p < {RULE[2]})")


if __name__ == "__main__":
    main()
