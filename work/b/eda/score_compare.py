"""간단한 규칙들의 val 점수 비교 그림 → figs/10_score_compare.png

    uv run python work/b/eda/score_compare.py

- 전부 보합: 한 칸만 계속 찍으면 정확히 0점
- 어제 따라가기: 기준일(어제) 수익률의 등급을 그대로 찍음
- 교수님 예시 RandomForest: example/sample_model.py 의 8개 피처 (b_baseline.py 결과)
- pre 세션만 규칙: 대상일 개장 전 pre 세션 안의 움직임(pre_move) 하나로 등급을 나눔 (교수님 예시의 pre_ret 과 같은 정보)
- gap_z 규칙: 전날 종가 → 개장 직전 시간외 가격(gap) ÷ vol20 하나로 등급을 나눔
규칙 둘은 train 의 등급 비율대로 값을 잘라 경계 4개를 정함 (b_baseline.rule_gap 과 같은 방식).
점수는 src.data.score, val = work/common/folds.py 의 val (대상일 2026-03-02 ~ 05-29).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work" / "common"))

from src import label_of, score  # noqa: E402
from folds import split  # noqa: E402

CACHE, FIGS = HERE.parent / "cache", HERE / "figs"


def rule(tr, te, col):
    """col 하나로 5등급. train 등급 비율로 경계를 정하고, 방향은 train 상관 부호."""
    s = np.sign(tr[col].corr(tr.ret_pct, method="spearman")) or 1.0
    x_tr, x_te = s * tr[col].fillna(0), s * te[col].fillna(0)
    cum = np.cumsum(tr.label.value_counts(normalize=True).sort_index().values)[:-1]
    return np.searchsorted(np.quantile(x_tr, cum), x_te.values, side="right")


def main():
    b = pd.read_parquet(CACHE / "b_features.parquet")
    b["label"] = b.label.astype(int)
    tr_m, va_m = split(b.target)
    tr, va = b[tr_m], b[va_m]

    rows = {
        "전부 보합": score(va.label, np.full(len(va), 2))["score"],
        "어제 따라가기": score(va.label, label_of(va.ret1 * 100).astype(int))["score"],
        "pre 세션만 규칙": score(va.label, rule(tr, va, "pre_move"))["score"],
        "gap_z 규칙": score(va.label, rule(tr, va, "gap_z"))["score"],
    }
    pooled = pd.read_csv(CACHE / "baseline_pooled.csv", index_col=0)
    rows["교수님 예시 RandomForest"] = float(pooled.loc["(c) RF sample 8", "score"])
    order = ["전부 보합", "어제 따라가기", "교수님 예시 RandomForest", "pre 세션만 규칙", "gap_z 규칙"]
    s = pd.Series(rows)[order]
    print(s.round(3).to_string())

    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in have), "sans-serif"),
                         "axes.unicode_minus": False})
    SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    BLUE, GRAY, RED = "#2a78d6", "#9a988f", "#e34948"
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(s))[::-1]
    colors = [BLUE if k == "gap_z 규칙" else (RED if v < 0 else GRAY) for k, v in s.items()]
    ax.barh(y, s.values, height=0.55, color=colors)
    span = max(s.max(), 0.1) - min(s.min(), 0)
    for yi, (k, v) in zip(y, s.items()):
        off = span * 0.02
        ax.text(v + off if v >= 0 else off, yi, f"{v:+.2f}" if v < 0 else f"{v:.2f}", va="center", ha="left",
                color=INK if k == "gap_z 규칙" else INK2, fontsize=13 if k == "gap_z 규칙" else 11,
                fontweight="bold" if k == "gap_z 규칙" else "normal")
    ax.set_yticks(y, s.index, fontsize=11, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.set_xlim(min(s.min(), 0) - span * 0.05, s.max() + span * 0.18)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.set_title("아무 생각 없이 찍으면 0점, 피처 하나로 0.48", loc="left", color=INK, fontsize=14, pad=24)
    ax.text(0, 1.03, "val 2026-03~05 score (0 = 무작위, 1 = 완벽)", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "10_score_compare.png", dpi=150)
    print("저장", FIGS / "10_score_compare.png")


if __name__ == "__main__":
    main()
