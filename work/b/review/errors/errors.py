"""갭 규칙 오답 분석 — 4폴드 val 예측(12,000행)을 합쳐 벌점이 어디서 나오는지.

    uv run python work/b/review/errors/errors.py

사후 분석 (규칙을 고치지 않음). 폴드마다 train 으로 학습한 규칙으로 그 폴드 val 을 예측한 뒤 합침.
행 벌점 = WEIGHT[정답, 예측], 벌점 비중 = 그 행들의 벌점 합 / 전체 벌점 합 (점수식 분자).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
LAB = ["급하락", "하락", "보합", "상승", "급상승"]


def predict_all():
    t, _ = E.load()
    t, _ = NG.build(t)          # 뉴스 그룹, post (09:00 → 종가)
    rows = []
    for name, tr, va, _, _ in NG.folds_of(t):
        P = G.fit_table(tr)
        x = _s_parts(va, P)
        rows.append(va.assign(fold=name, pred=G.predict_table(va, P), s=x, a=P["a"], b=P["b"]))
    v = pd.concat(rows, ignore_index=True)
    v["pen"] = G.WEIGHT[v.label.to_numpy(int), v.pred.to_numpy(int)]
    v["kind"] = np.select(
        [((v.label == 0) & (v.pred >= 3)) | ((v.label == 4) & (v.pred <= 1)),
         v.label.isin([0, 4]) & (v.pred == 2),
         v.label.isin([0, 4]) & (np.sign(v.label - 2) == np.sign(v.pred - 2)) & ~v.pred.isin([0, 4]),
         v.pen > 0],
        ["① 급등락을 정반대로", "② 급등락을 보합으로", "③ 급등락을 한 칸 약하게", "④ 그 밖 (급등락 아닌 날 오답)"],
        "벌점 없음")
    return t, v


def _s_parts(va, P):
    return G._score_s(va.gap_z.to_numpy(float), va.ext_range_z.to_numpy(float), P["med"], P["sd"], P["lam"])


def share_table(v, col, order=None):
    tot = v.pen.sum()
    g = v.groupby(col, observed=True)
    out = pd.DataFrame({"행 수": g.size(), "행 비중": g.size() / len(v), "벌점 비중": g.pen.sum() / tot,
                        "정반대 비율": g.kind.apply(lambda s: (s == "① 급등락을 정반대로").mean()),
                        "score": g.apply(lambda x: G.score(x.label, x.pred))})
    out["집중도"] = out["벌점 비중"] / out["행 비중"]
    return out.reindex(order) if order else out


def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    t, v = predict_all()
    print("전체", len(v), "행, score", round(G.score(v.label, v.pred), 4))

    # 1. 벌점이 나오는 칸
    tot = v.pen.sum()
    cell = v.groupby(["label", "pred"]).pen.agg(["size", "sum"]).reset_index()
    cell["벌점 비중"] = cell["sum"] / tot
    mat = cell.pivot(index="label", columns="pred", values="벌점 비중").reindex(index=range(5), columns=range(5)).fillna(0)
    kinds = v.groupby("kind").agg(행=("pen", "size"), 벌점=("pen", "sum"))
    kinds["행 비중"], kinds["벌점 비중"] = kinds.행 / len(v), kinds.벌점 / tot
    print("\n[1] 오답 종류\n", kinds.round(3).to_string())

    # 2. 어떤 행에 몰리나 (그룹별)
    v["|gap_z|"] = pd.cut(v.gap_z.abs(), [0, 0.1, 0.25, 0.5, 1, 2, np.inf], labels=["<0.1", "0.1~0.25", "0.25~0.5", "0.5~1", "1~2", "2+"])
    v["갭 방향"] = np.where(v.gap_z > 0, "위로 갭", np.where(v.gap_z < 0, "아래로 갭", "갭 없음"))
    v["경계와 거리"] = pd.cut(v.s.abs() / v.b, [0, 0.5, 1, 1.5, 2, 4, np.inf], labels=["<0.5b", "0.5~1b", "1~1.5b", "1.5~2b", "2~4b", "4b+"])
    v["실적 밤"] = np.where(v.earn, "있음", "없음")
    v["시장 갭 큰 날"] = np.where(v.mkt_big, "상위 20%", "나머지")
    v["종목"] = np.where(v.unseen, "처음 보는 10", "학습한 40")
    v["뉴스"] = v.grp
    v["요일(대상일)"] = v.target.dt.day_name().str[:3]
    v["vol20 5분위"] = pd.qcut(v.vol20, 5, labels=["1 낮음", "2", "3", "4", "5 높음"])
    groups = {}
    for col in ["|gap_z|", "경계와 거리", "갭 방향", "실적 밤", "시장 갭 큰 날", "종목", "뉴스", "vol20 5분위", "요일(대상일)", "fold"]:
        groups[col] = share_table(v, col)
        print(f"\n[2] {col}\n", groups[col].round(3).to_string())
    pd.concat(groups, names=["기준"]).to_csv(OUT / "groups.csv", encoding="utf-8-sig")

    # 3. 날 단위 쏠림
    day = v.groupby("target").agg(pen=("pen", "sum"), n=("pen", "size"),
                                  opp=("kind", lambda s: (s == "① 급등락을 정반대로").sum()),
                                  mkt_gap=("gap_z", "mean"), mkt_post=("post", "mean"))
    day = day.sort_values("pen", ascending=False)
    day["누적 벌점 비중"] = day.pen.cumsum() / tot
    nd = len(day)
    top = {k: float(day["누적 벌점 비중"].iloc[int(np.ceil(nd * k)) - 1]) for k in [0.05, 0.10, 0.20]}
    print("\n[3] 날 수", nd, "| 벌점 상위 5%·10%·20% 날의 벌점 비중", {k: round(x, 3) for k, x in top.items()})
    day["시장 뒤집힘"] = np.sign(day.mkt_post) != np.sign(day.mkt_gap)
    print("상위 10% 날 중 시장 평균이 갭과 반대로 간 날 비율", round(day.head(int(nd * 0.1))["시장 뒤집힘"].mean(), 3),
          "| 전체 날", round(day["시장 뒤집힘"].mean(), 3))
    print(day.head(10)[["pen", "opp", "mkt_gap", "mkt_post", "누적 벌점 비중"]].round(3).to_string())
    day.to_csv(OUT / "days.csv", encoding="utf-8-sig")

    # 4. 정반대 오답 행의 아침 모습 vs 맞힌 급등락
    opp = v[v.kind == "① 급등락을 정반대로"]
    hit = v[v.label.isin([0, 4]) & (v.label == v.pred)]
    v["day_rev"] = v.target.map(day["시장 뒤집힘"])
    cols = {"|gap_z|": lambda x: x.gap_z.abs().median(), "s/b": lambda x: (x.s.abs() / x.b).median(),
            "ext_range_z": lambda x: x.ext_range_z.median(), "vol20(%)": lambda x: x.vol20.median() * 100,
            "실적 밤": lambda x: x.earn.mean(), "기사 없음": lambda x: (x.grp == NG.GROUPS[0]).mean(),
            "시장 갭과 같은 방향": lambda x: (np.sign(x.gap_z) == np.sign(x.target.map(day.mkt_gap))).mean(),
            "그날 시장이 뒤집힘": lambda x: x.target.map(day["시장 뒤집힘"]).mean(),
            "09:00 이후 |움직임|/vol20": lambda x: (x.post.abs() / x.vol20).median()}
    prof = pd.DataFrame({k: {"① 정반대 오답": f(opp), "맞힌 급등락": f(hit), "전체": f(v)} for k, f in cols.items()}).T
    print("\n[4] 정반대 오답의 모습\n", prof.round(3).to_string())
    prof.to_csv(OUT / "profile.csv", encoding="utf-8-sig")

    # 5. 놓친 급등락: 갭이 작아서 못 찍은 급등락
    big = v[v.label.isin([0, 4])]
    miss = big.assign(gapsmall=big.gap_z.abs() < 0.25)
    print("\n[5] 실제 급등락", len(big), "행 중 |gap_z| < 0.25 비율", round(miss.gapsmall.mean(), 3),
          "| 그중 규칙이 급등락으로 찍은 비율", round(miss[miss.gapsmall].pred.isin([0, 4]).mean(), 3))
    ups, downs = v[v.gap_z > 0], v[v.gap_z < 0]
    print("위로 갭 정반대 비율", round((ups.kind == "① 급등락을 정반대로").mean(), 4),
          "| 아래로 갭", round((downs.kind == "① 급등락을 정반대로").mean(), 4))

    # 그림
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax = axes[0]
    im = ax.imshow(mat.values * 100, cmap="Blues")
    for i in range(5):
        for j in range(5):
            val = mat.values[i, j] * 100
            if val >= 0.5:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if val > 15 else E.INK)
    ax.set_xticks(range(5), LAB, fontsize=8)
    ax.set_yticks(range(5), LAB, fontsize=8)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.grid(False)
    ax.set_title("벌점이 나온 칸 (전체 벌점 대비)", loc="left")
    ax = axes[1]
    g = groups["경계와 거리"]
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["행 비중"] * 100, 0.38, color=E.GRID, edgecolor=E.INK2, label="행 비중")
    ax.bar(x + 0.2, g["벌점 비중"] * 100, 0.38, color=E.C2, label="벌점 비중")
    ax.set_xticks(x, g.index, fontsize=8)
    ax.set_xlabel("|s| ÷ 급등락 경계 b (1 이상이면 급등락으로 찍음)")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.set_title("경계에서 얼마나 떨어진 행이 벌점을 내나", loc="left")
    ax = axes[2]
    xs = np.arange(1, nd + 1) / nd * 100
    ax.plot(xs, day["누적 벌점 비중"].to_numpy() * 100, color=E.C1, linewidth=2)
    ax.plot([0, 100], [0, 100], color=E.INK2, linewidth=0.8, linestyle="--")
    for k, val in top.items():
        ax.scatter([k * 100], [val * 100], color=E.C2, zorder=3)
        ax.annotate(f"상위 {int(k * 100)}% 날 → {val:.0%}", (k * 100, val * 100), xytext=(8, -12),
                    textcoords="offset points", fontsize=8, color=E.INK)
    ax.set_xlabel("벌점 큰 순서로 센 날 (%)")
    ax.set_ylabel("누적 벌점 비중 (%)")
    ax.set_title(f"벌점이 몇몇 날에 몰리나 ({nd}일)", loc="left")
    fig.suptitle("갭 규칙 오답 분석 — 4폴드 val 12,000행", x=0.01, ha="left", fontsize=11, fontweight="bold", color=E.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "errors.png", dpi=160)
    plt.close(fig)
    cell.to_csv(OUT / "cells.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
