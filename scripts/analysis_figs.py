"""발표용 분석 a, c~h 와 그림.  uv run python scripts/analysis_figs.py
읽는 것: cache/analysis/runs.json (analysis_runs.py), cache/analysis/preds_*.parquet (analysis_params.py),
        cache/b_v1.parquet (guard 아래 계산된 피처, 기준일 ≤ 2026-05-28). 홀드아웃 날짜는 쓰지 않음 (lab/folds.json 만).
쓰는 것: docs/presentation/figs/*.png, cache/analysis/numbers.json (슬라이드에 들어가는 숫자 전부)
W&B: 숫자 요약을 tags=["analysis"] run 하나로 기록 (키가 없으면 건너뜀).
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import tracking  # noqa: E402
from lab.folds import HOLDOUT, load_folds, unseen_symbols  # noqa: E402
from src import Dataset, score  # noqa: E402
from work.a import exp2 as E  # noqa: E402
from work.a import gapz_rule as R  # noqa: E402
from work.a import lgb as L  # noqa: E402
from work.a import models2 as M  # noqa: E402
from work.a.rules import fit_gapz_ext  # noqa: E402

AN = ROOT / "cache" / "analysis"
FIG = ROOT / "docs" / "presentation" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
W = R.WEIGHT
NAMES = ["급하락", "하락", "보합", "상승", "급상승"]

# 색: 역할별로 고정 (모든 그림 공통)
C_RULE, C_LGB, C_TEXT, C_REF = "#1A7F55", "#D97706", "#2B6CB0", "#8A8F98"
INK, MUTED, GRID = "#1F2328", "#5B616B", "#E4E7EB"
LABEL_COLORS = ["#B42318", "#F0A39B", "#D0D5DD", "#8CC9A8", "#1A7F55"]   # 급하락 → 급상승 (발산: 빨강-회색-녹색)

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})
NUM = {}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("fig", name, flush=True)


def conf(y, p):
    return np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)


def decompose(y, p):
    o = conf(y, p)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return o, e, (W * o).sum(1), (W * e).sum(1)


def preds(name):
    """run_cv 예측 행. 그 'date' 열은 day.y 의 날짜 = 대상일(target) 이라 기준일(base) 을 붙여 둠."""
    p = pd.read_parquet(AN / f"preds_{name}.parquet").rename(columns={"date": "target"})
    return p.assign(date=p.target.map(T2D))


# ---------------------------------------------------------------- 공통 데이터
ds = Dataset()
dev = ds.split(HOLDOUT)[0]
folds = load_folds(ds)
f4 = folds[-1]
UNSEEN = set(unseen_symbols())
master = E.master(dev)
assert master.date.max() < pd.Timestamp(HOLDOUT), "홀드아웃 행"
T2D = {d.target: d.date for d in dev}
tr4 = {d.date for d in f4.train}
va4 = {d.date for d in f4.val}
runs = {r["name"]: r for r in json.loads((AN / "runs.json").read_text(encoding="utf-8"))
        if "failed_empty" not in r["tags"]}


# ---------------------------------------------------------------- 0. 가중치 행렬
def fig_weight():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(np.sqrt(W), cmap="Greens", vmin=0, vmax=10)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(W[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if W[i, j] >= 36 else INK, fontweight="bold" if W[i, j] >= 36 else None)
    ax.set_xticks(range(5), NAMES)
    ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.grid(False)
    ax.set_title("벌점 가중치 w = (i-j)² × (|i-2|)²")
    save(fig, "00_weight")


# ---------------------------------------------------------------- d. 올려 찍기 계산
def fig_upgrade():
    """상승(3)으로 찍던 행을 급상승(4)으로 올리면 정답별로 벌점이 얼마나 바뀌나."""
    p = preds("rule_gapz_ext")
    delta = W[:, 4] - W[:, 3]                       # 정답 i 일 때 3→4 로 바꾼 벌점 변화
    d1 = W[:, 1] - W[:, 0]                          # 대칭: 1→0
    rows3 = p[p.label_pred == 3]
    dist = np.bincount(rows3.label, minlength=5)
    NUM["upgrade"] = {"delta_3to4": delta.tolist(), "fold4_rows_pred3": int(len(rows3)),
                      "truth_of_pred3": dist.tolist(),
                      "numerator_change_if_all_upgraded": float((dist * delta).sum())}
    o, e, on, en = decompose(p.label, p.label_pred)
    NUM["upgrade"]["fold4_rule_numerator"] = float(on.sum())
    NUM["upgrade"]["fold4_rule_denominator"] = float(en.sum())
    p2 = p.assign(label_pred=p.label_pred.replace({3: 4}))
    NUM["upgrade"]["score_if_all_3_to_4"] = score(p2.label, p2.label_pred)["score"]
    NUM["upgrade"]["score_rule"] = score(p.label, p.label_pred)["score"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = [C_RULE if v < 0 else ("#B42318" if v > 0 else C_REF) for v in delta]
    ax.bar(range(5), delta, color=cols, width=0.55)
    for i, v in enumerate(delta):
        ax.text(i, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                color=INK, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(5), [f"정답 {n}" for n in NAMES])
    ax.set_ylabel("벌점 변화 (한 행)")
    ax.set_ylim(-10, 34)
    ax.set_title("'상승' 대신 '급상승'으로 찍으면: 맞으면 -4, 정반대면 +28")
    save(fig, "05_upgrade")
    return d1


# ---------------------------------------------------------------- c. 점수식 분해
def fig_decompose():
    out = {}
    for n in ["rule_gapz_ext", "lgb_v1_aggr", "lgb_v1_argmax"]:
        p = preds(n)
        o, e, on, en = decompose(p.label, p.label_pred)
        out[n] = {"O_by_true": on.tolist(), "E_by_true": en.tolist(), "score": 1 - on.sum() / en.sum(),
                  "pred_dist": (o.sum(0) / o.sum()).tolist(), "true_dist": (o.sum(1) / o.sum()).tolist(),
                  "confusion": o.astype(int).tolist()}
    NUM["decompose"] = out
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    x = np.arange(5)
    for ax, n, c, t in [(axes[0], "rule_gapz_ext", C_RULE, "B 규칙 (fold4 0.515)"),
                        (axes[1], "lgb_v1_aggr", C_LGB, "LightGBM aggr (fold4 0.495)")]:
        d = out[n]
        ax.bar(x - 0.2, d["E_by_true"], 0.38, color=C_REF, label="무작위 기대 벌점 (분모)")
        ax.bar(x + 0.2, d["O_by_true"], 0.38, color=c, label="실제 벌점 (분자)")
        for i in x:
            ax.text(i + 0.2, d["O_by_true"][i], f"{d['O_by_true'][i] / 1000:.1f}k", ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(x, [f"정답\n{n_}" for n_ in NAMES])
        ax.set_title(t)
        ax.legend(loc="upper center", fontsize=9)
    axes[0].set_ylabel("Σ w·O (정답 등급별)")
    save(fig, "04_decompose")

    # 차이가 생기는 칸: (규칙 - LightGBM) w·O / 분모
    r, g = out["rule_gapz_ext"], out["lgb_v1_aggr"]
    wo_r = W * np.array(r["confusion"]) / sum(r["E_by_true"])
    wo_g = W * np.array(g["confusion"]) / sum(g["E_by_true"])
    diff = wo_g - wo_r
    NUM["decompose"]["cell_diff_lgb_minus_rule"] = diff.round(4).tolist()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    lim = np.abs(diff).max()
    ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(5):
        for j in range(5):
            if W[i, j]:
                ax.text(j, i, f"{diff[i, j] * 100:+.1f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(5), NAMES)
    ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.grid(False)
    ax.set_title("LightGBM - 규칙, 칸별 벌점 비중 (%p)\n빨강 = LightGBM 이 더 잃은 칸")
    save(fig, "A_decompose_cells")


# ---------------------------------------------------------------- e. gap_z 구간별 실제 등급
def fig_gapz_bins():
    t = master[master.date.isin(tr4) & ~master.symbol.isin(UNSEEN)].copy()
    m = fit_gapz_ext([d for d in f4.train], sorted(UNSEEN)).params
    NUM["rule_fold4_params"] = m
    x = np.nan_to_num(np.clip((t.ext_range_z - m["med"]) / m["sd"], -5, 5))
    t["s"] = t.gap_z * np.exp(-m["lam"] * x)
    edges = [-np.inf, -m["b"], -m["a"], m["a"], m["b"], np.inf]
    bins = np.quantile(t.s.dropna(), np.linspace(0, 1, 13))
    t["bin"] = pd.cut(t.s, bins, include_lowest=True)
    tab = pd.crosstab(t.bin, t.label, normalize="index").reindex(columns=range(5), fill_value=0)
    mids = [iv.mid for iv in tab.index]
    NUM["gapz_bins"] = {"edges": [str(i) for i in tab.index], "dist": tab.round(4).values.tolist(),
                        "a": m["a"], "b": m["b"], "n": int(t.s.notna().sum()),
                        "share_inside_a": float((t.s.abs() < m["a"]).mean()),
                        "share_beyond_b": float((t.s.abs() >= m["b"]).mean())}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bottom = np.zeros(len(tab))
    for k in range(5):
        ax.bar(range(len(tab)), tab[k], bottom=bottom, color=LABEL_COLORS[k], width=0.86,
               edgecolor="white", linewidth=1.5, label=NAMES[k])
        bottom += tab[k].values
    ax.set_xticks(range(len(tab)), [f"{iv.left:.2f}\n~{iv.right:.2f}" for iv in tab.index], fontsize=8)
    ax.set_xlabel("규칙 점수 s = gap_z · exp(-λ·ext_z)  구간 (fold4 train, 각 구간 같은 행 수)")
    ax.set_ylabel("실제 등급 비율")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    for v in edges[1:-1]:
        pos = np.interp(v, mids, range(len(mids)))
        ax.axvline(pos, color=INK, lw=1.2, ls="--")
    ax.text(np.interp(m["b"], mids, range(len(mids))), 1.01, f" b={m['b']:.2f}", fontsize=9, color=INK)
    ax.text(np.interp(m["a"], mids, range(len(mids))), 1.01, f" a={m['a']:.2f}", fontsize=9, color=INK)
    save(fig, "02_gapz_bins")


# ---------------------------------------------------------------- f. 노이즈 바닥
def fig_noise():
    p = preds("rule_gapz_ext")
    y = p.label.to_numpy()
    rng = np.random.default_rng(0)
    q = np.bincount(master[master.date.isin(tr4) & ~master.symbol.isin(UNSEEN)].label, minlength=5)
    q = q / q.sum()
    uni = [score(y, rng.integers(0, 5, len(y)))["score"] for _ in range(1000)]
    pri = [score(y, rng.choice(5, len(y), p=q))["score"] for _ in range(1000)]

    # 날짜 블록 bootstrap: 규칙 fold4 점수, recent250 - 규칙 (짝지은 차이)
    p2 = preds("v3_recent250").set_index(["date", "symbol"]).label_pred.rename("p2")
    pp = p.set_index(["date", "symbol"]).join(p2).reset_index()
    days = pp.date.unique()
    g = {d: x for d, x in pp.groupby("date")}
    boot, diff = [], []
    for _ in range(1000):
        s = pd.concat([g[d] for d in rng.choice(days, len(days))])
        a = score(s.label, s.label_pred)["score"]
        boot.append(a)
        diff.append(score(s.label, s.p2)["score"] - a)

    # 가짜 피처: 행을 섞은 atr14 로 '급등락 경계 이동' 규칙 (v2_rsize 와 같은 절차) 을 여러 번
    t = master[master.date.isin(tr4) & ~master.symbol.isin(UNSEEN)].copy()
    v = master[master.date.isin(va4)].copy()
    base = score(v.label, _rule_labels(t, v))["score"]
    fake = []
    for i in range(30):
        r = np.random.default_rng(100 + i)
        t["fake"] = r.permutation(t.atr14.to_numpy())
        v["fake"] = r.permutation(v.atr14.to_numpy())
        labels, info = M.fit_rule_size(t, ["fake"])
        lab = labels(v)
        lab[np.isnan(v.gap_z.to_numpy(float))] = 2
        fake.append(score(v.label, lab)["score"] - base)
        print("fake", i, round(fake[-1], 4), info["delta"], flush=True)
    NUM["noise"] = {
        "random_uniform_sd": float(np.std(uni)), "random_uniform_p95": float(np.quantile(np.abs(uni), .95)),
        "random_prior_sd": float(np.std(pri)),
        "rule_fold4_boot_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "rule_fold4_boot_sd": float(np.std(boot)),
        "recent250_minus_rule_ci": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))],
        "recent250_minus_rule_p_le0": float(np.mean(np.array(diff) <= 0)),
        "fake_feature_diffs": fake, "fake_max": float(np.max(fake)),
        "fake_share_ge_0005": float(np.mean(np.array(fake) >= 0.005)),
        "fake_share_gt0": float(np.mean(np.array(fake) > 0)), "base_rule_fold4_master": base}

    wb = pd.read_csv(ROOT / "docs" / "presentation" / "experiment_table_raw.csv").dropna(subset=["vs_gapz_ext_fold4"])
    real = wb[wb.run.str.startswith(("v2_", "v3_"))].drop_duplicates("run", keep="last")
    real = real[real.vs_gapz_ext_fold4.abs() > 1e-9]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ax.hist(uni, bins=40, color=C_REF, alpha=.9, label="5등급 균등 무작위")
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(f"무작위 예측 1,000번: fold4 score 표준편차 {np.std(uni):.3f}")
    ax.set_xlabel("score")
    ax.set_ylabel("횟수")
    ax = axes[1]
    yy = np.zeros(len(fake))
    ax.scatter(fake, yy + 1, s=40, color=C_REF, label="가짜 피처 (행 섞음) 30회", zorder=3)
    ax.scatter(real.vs_gapz_ext_fold4, np.zeros(len(real)), s=40, color=C_TEXT, label="실제 v2·v3 변형", zorder=3)
    ax.annotate("recent250·120·winvote", (0.006, 0), xytext=(-60, -22), textcoords="offset points",
                fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(0.005, color=C_RULE, lw=1, ls="--")
    ax.text(0.0052, 1.35, "+0.005", color=C_RULE, fontsize=9)
    ax.set_yticks([0, 1], ["실제", "가짜"])
    ax.set_ylim(-0.7, 1.6)
    ax.set_xlabel("fold4 score - B 규칙")
    ax.set_title("규칙에 무언가 더했을 때 fold4 변화")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "06_noise")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(diff, bins=40, color=C_TEXT)
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(f"recent250 - 규칙, 날짜 bootstrap 1,000회\n0 이하일 확률 {np.mean(np.array(diff) <= 0):.0%}")
    ax.set_xlabel("fold4 score 차이")
    save(fig, "A_recent_boot")


def _rule_labels(t, v):
    s, (a, b), _ = M.fit_rule(t, [])
    x = np.asarray(s(v), float)
    x[np.isnan(v.gap_z.to_numpy(float))] = np.nan
    return R.cut(x, a, b)


# ---------------------------------------------------------------- g. 오차 분석
def fig_errors():
    p = preds("rule_gapz_ext").merge(master[["date", "symbol", "gap_z", "abs_gap_z", "earn_now", "vol20",
                                             "ext_range_z"]], on=["date", "symbol"], how="left")
    w = W[p.label, p.label_pred]
    p["pen"] = w
    big = ((p.label == 0) & (p.label_pred >= 3)) | ((p.label == 4) & (p.label_pred <= 1))
    miss = p.label.isin([0, 4]) & (p.label_pred == 2)
    tot = w.sum()
    by_day = p.groupby("date").pen.sum().sort_values(ascending=False)
    unseen = p.symbol.isin(UNSEEN)
    vol_u = p[unseen].groupby("symbol").vol20.mean()
    NUM["errors"] = {
        "n": int(len(p)), "n_big_wrong": int(big.sum()), "share_rows_big_wrong": float(big.mean()),
        "share_penalty_big_wrong": float(w[big].sum() / tot),
        "n_missed_big_as_flat": int(miss.sum()), "share_penalty_missed_flat": float(w[miss].sum() / tot),
        "earn_rate_all": float(p.earn_now.fillna(0).gt(0).mean()),
        "earn_rate_big_wrong": float(p[big].earn_now.fillna(0).gt(0).mean()),
        "abs_gapz_median_all": float(p.abs_gap_z.median()), "abs_gapz_median_big_wrong": float(p[big].abs_gap_z.median()),
        "unseen_rate_all": float(unseen.mean()), "unseen_rate_big_wrong": float(p[big].symbol.isin(UNSEEN).mean()),
        "top5_days": [[str(d.date()), float(v / tot)] for d, v in by_day.head(5).items()],
        "top5_days_share": float(by_day.head(5).sum() / tot), "n_days": int(len(by_day)),
        "big_rate_seen": float(p[~unseen].label.isin([0, 4]).mean()),
        "big_rate_unseen": float(p[unseen].label.isin([0, 4]).mean()),
        "vol20_seen": float(p[~unseen].vol20.mean()), "vol20_unseen": float(p[unseen].vol20.mean()),
        "unseen_score_by_symbol": {s: float(score(g.label, g.label_pred)["score"]) for s, g in p[unseen].groupby("symbol")},
        "unseen_vol20": vol_u.round(4).to_dict(),
    }
    # 정답이 급등락인 행: 규칙 점수 크기별로 맞힌 방향 비율
    bigrows = p[p.label.isin([0, 4])].copy()
    bigrows["sign_ok"] = np.sign(bigrows.gap_z) == np.where(bigrows.label == 4, 1, -1)
    bigrows["bin"] = pd.cut(bigrows.abs_gap_z, [0, .25, .5, 1, 2, np.inf])
    sg = bigrows.groupby("bin", observed=True).agg(n=("sign_ok", "size"), ok=("sign_ok", "mean"))
    NUM["errors"]["big_sign_by_absgapz"] = {str(k): [int(r.n), float(r.ok)] for k, r in sg.iterrows()}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ax.bar(range(len(sg)), sg.ok, color=C_RULE, width=0.6)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    for i, (k, r) in enumerate(sg.iterrows()):
        ax.text(i, r.ok + 0.02, f"{r.ok:.0%}\n(n={r.n})", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(sg)), [f"{k.left:g}~{k.right:g}" for k in sg.index])
    ax.set_xlabel("|gap_z| 구간")
    ax.set_ylabel("갭 부호가 맞은 비율")
    ax.set_ylim(0, 1.15)
    ax.set_title("정답이 급등락인 날: 갭이 작으면 방향은 동전 던지기")
    ax = axes[1]
    cats = ["행 수", "벌점"]
    vals = [[big.mean(), miss.mean(), 1 - big.mean() - miss.mean()],
            [w[big].sum() / tot, w[miss].sum() / tot, 1 - (w[big].sum() + w[miss].sum()) / tot]]
    cols = ["#B42318", C_REF, "#D0D5DD"]
    labs = ["급등락을 반대 방향으로", "급등락을 보합으로", "나머지"]
    for i, vv in enumerate(vals):
        left = 0
        for k, v in enumerate(vv):
            ax.barh(i, v, left=left, color=cols[k], edgecolor="white", height=0.55, label=labs[k] if i == 0 else None)
            if v > 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white" if k == 0 else INK)
            left += v
    ax.set_yticks([0, 1], cats)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.set_title("fold4 B 규칙: 행 3%가 벌점의 절반")
    ax.grid(axis="y", visible=False)
    save(fig, "07_errors")


# ---------------------------------------------------------------- h. 월별
def fig_monthly():
    mm = master.assign(month=pd.to_datetime(master.target).dt.strftime("%Y-%m"))   # W&B 월별과 같게 대상일 월
    big = mm.groupby("month").label.apply(lambda s: s.isin([0, 4]).mean())
    big = big[big.index >= "2024-08"]

    def mon(n):
        t = runs[n]["tables"]["monthly"]
        return pd.Series(t["score"], index=t["month"])
    rule, rec, lgb = mon("rule_gapz_ext"), mon("v3_recent250"), mon("lgb_v1_aggr")
    NUM["monthly"] = {"big_rate": big.round(4).to_dict(), "rule": rule.round(4).to_dict(),
                      "recent250": rec.round(4).to_dict(), "lgb_aggr": lgb.round(4).to_dict(),
                      "big_rate_train_fold4": float(master[master.date.isin(tr4)].label.isin([0, 4]).mean()),
                      "big_rate_val_fold4": float(master[master.date.isin(va4)].label.isin([0, 4]).mean())}
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    months = list(big.index)
    ax = axes[0]
    ax.bar(range(len(months)), big.values, color=C_REF, width=0.7)
    ax.set_ylabel("급등락 비율")
    ax.set_title("월별 급등락 비율 (대상일 월, 전체 dev 기간)")
    ax = axes[1]
    for s, c, lab in [(rule, C_RULE, "B 규칙"), (lgb, C_LGB, "LightGBM aggr"), (rec, C_TEXT, "recent250")]:
        xi = [months.index(m) for m in s.index if m in months]
        ax.plot(xi, s.values[:len(xi)], marker="o", ms=5, lw=2, color=c, label=lab)
    ax.set_ylabel("val score (월)")
    ax.set_title("월별 검증 점수 (4폴드 val 이어 붙임)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_xticks(range(len(months)), [m[2:] for m in months], rotation=90, fontsize=8)
    save(fig, "08_monthly")


# ---------------------------------------------------------------- a. 중요도 + SHAP(pred_contrib)
def fig_importance():
    imp = {}
    for n, r in runs.items():
        t = r["tables"]
        if "importance/fold1" not in t:
            continue
        per = pd.DataFrame({f: pd.Series(t[f"importance/{f}"]["importance"], index=t[f"importance/{f}"]["feature"])
                            for f in ["fold1", "fold2", "fold3", "fold4"] if f"importance/{f}" in t})
        rk = per.rank(ascending=False)
        rho = rk.corr(method="spearman").values[np.triu_indices(per.shape[1], 1)].mean()
        top = per.mean(1).sort_values(ascending=False)
        textf = [c for c in per.index if c in L.TEXT["news"] + L.TEXT["reddit"] + L.TEXT["joint"]
                 + L.TEXT["context"] or c.startswith(("lex_", "text_"))]
        imp[n] = {"top10": top.head(10).round(4).to_dict(), "rank_spearman_mean": float(rho),
                  "top10_rank_by_fold": rk.loc[top.head(10).index].astype(int).to_dict("index"),
                  "text_share": float(per.loc[textf].sum().mean()) if textf else 0.0, "text_feats": textf}
    NUM["importance"] = imp

    # fold4 LightGBM (lgb_v1) 다시 학습 → pred_contrib
    f = L.fit_all(f4.train, sorted(UNSEEN))
    b = f["booster"]
    v = master[master.date.isin(va4)]
    X = v[L.FEATS]
    c = b.predict(X, pred_contrib=True).reshape(len(X), 5, len(L.FEATS) + 1)
    direction = c[:, 4, :-1] - c[:, 0, :-1]                 # 급상승 vs 급하락 logit 으로 미는 정도
    size = -c[:, 2, :-1]                                    # 보합 logit 을 깎는 정도 (+ 면 급등락·등락 쪽)
    dmean = pd.Series(np.abs(direction).mean(0), index=L.FEATS).sort_values(ascending=False)
    smean = pd.Series(np.abs(size).mean(0), index=L.FEATS).sort_values(ascending=False)
    gain = pd.Series(b.feature_importance("gain"), index=L.FEATS)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    gi = L.FEATS.index("gap_z")
    corr_dir = {k: float(np.corrcoef(X[k].fillna(X[k].median()), direction[:, L.FEATS.index(k)])[0, 1])
                for k in dmean.head(6).index}
    NUM["shap"] = {"direction_top": dmean.head(8).round(4).to_dict(), "size_top": smean.head(8).round(4).to_dict(),
                   "gain_top": gain.head(8).round(4).to_dict(), "corr_value_vs_direction": corr_dir,
                   "rounds": f["rounds"], "k": f["k"]}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s, col, t in [(axes[0], gain.head(8), C_REF, "gain 중요도 (크기만)"),
                          (axes[1], dmean.head(8), C_RULE, "방향 기여 |급상승-급하락|"),
                          (axes[2], smean.head(8), C_LGB, "크기 기여 |보합을 깎는 정도|")]:
        ax.barh(range(len(s))[::-1], s.values, color=col, height=0.6)
        ax.set_yticks(range(len(s))[::-1], s.index, fontsize=9)
        ax.set_title(t)
        ax.grid(axis="y", visible=False)
    save(fig, "09_importance")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ok = X.gap_z.notna()
    ax.scatter(np.clip(X.gap_z[ok], -4, 4), direction[ok.values, gi], s=6, alpha=.35, color=C_RULE, edgecolors="none")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("gap_z (±4 로 자름)")
    ax.set_ylabel("gap_z 의 기여 (급상승 - 급하락 logit)")
    ax.set_title("LightGBM fold4: gap_z 가 예측을 미는 방향")
    save(fig, "A_shap_gapz")


# ---------------------------------------------------------------- 실험 여정 (묶음별 fold4)
GROUP = [("기준선", lambda n: n in ("random_uniform", "random_prior", "flat", "yesterday", "rule_pre_move", "rule_ovn_gap")),
         ("B 규칙", lambda n: n in ("rule_gapz", "rule_gapz_ext")),
         ("LightGBM 디코딩", lambda n: n.startswith("lgb_v1")),
         ("C 텍스트 ablation", lambda n: n.startswith("ablation")),
         ("v2 선형·트리·손실", lambda n: n.startswith(("v2_ridge", "v2_huber", "v2_logit", "v2_lgb", "v2_hgb"))),
         ("v2 규칙+텍스트·앙상블", lambda n: n.startswith(("v2_rule", "v2_rsize", "v2_radd", "v2_ens"))),
         ("v3 갭·국면·창", lambda n: n.startswith("v3_"))]


def fig_journey():
    wb = pd.read_csv(ROOT / "docs" / "presentation" / "experiment_table_raw.csv").drop_duplicates("run", keep="last")
    base = wb.set_index("run").loc["rule_gapz_ext", "fold4"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for gi, (g, f) in enumerate(GROUP):
        d = wb[wb.run.map(f)].sort_values("fold4")
        col = {"B 규칙": C_RULE, "LightGBM 디코딩": C_LGB, "기준선": C_REF}.get(g, C_TEXT)
        jit = np.linspace(-0.25, 0.25, len(d)) if len(d) > 1 else [0]
        ax.scatter(d.fold4, gi + np.asarray(jit), s=34, color=col, zorder=3, edgecolors="white", linewidths=0.8)
        best = d.iloc[-1]
        ax.text(max(best.fold4, 0) + 0.006, gi + jit[-1], f"{best.run}  {best.fold4:.3f}", fontsize=8, va="center",
                color=INK)
    ax.axvline(base, color=C_RULE, lw=1.4, ls="--")
    ax.text(base, len(GROUP) - 0.35, f" B 규칙 {base:.3f}", color=C_RULE, fontsize=9)
    ax.axvspan(-0.06, 0.06, color=C_REF, alpha=0.12, lw=0)
    ax.text(0, -0.75, "무작위 ±2σ", ha="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(GROUP)), [g for g, _ in GROUP])
    ax.invert_yaxis()
    ax.set_xlim(-0.16, 0.66)
    ax.set_xlabel("fold4 score (val 2026-03-04 ~ 05-28)")
    ax.set_title("53개 run: 무엇을 해도 B 규칙 근처에서 멈춤")
    ax.grid(axis="y", visible=False)
    save(fig, "03_journey")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fx = ["fold1", "fold2", "fold3", "fold4"]
    for n, c, lab in [("rule_gapz_ext", C_RULE, "B 규칙"), ("lgb_v1_aggr", C_LGB, "LightGBM aggr"),
                      ("rule_pre_move", C_REF, "pre_move 규칙"), ("random_prior", "#C4C8CE", "무작위(정답 비율)")]:
        r = wb.set_index("run").loc[n]
        ax.plot(fx, r[fx].astype(float), marker="o", lw=2, ms=6, color=c, label=lab)
        ax.text(3.08, float(r["fold4"]), lab, fontsize=9, va="center", color=INK)
    ax.set_xlim(-0.2, 3.9)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("score")
    ax.set_title("폴드별 점수: 순위는 네 폴드 모두 같음")
    save(fig, "A_folds")


def log_wandb():
    tr = tracking.Tracker(name="analysis_presentation", tags=["analysis"], job_type="analysis",
                          config={"git": tracking.git_info(), "script": "scripts/analysis_figs.py"})
    if tr.on:
        flat = {}
        for k in ["noise", "errors", "upgrade"]:
            for kk, vv in NUM[k].items():
                if isinstance(vv, (int, float)):
                    flat[f"{k}/{kk}"] = vv
        tr.run.summary.update(flat)
        for p in sorted(FIG.glob("*.png")):
            tr.run.log({f"fig/{p.stem}": tr.wandb.Image(str(p))})
    tr.finish()


if __name__ == "__main__":
    fig_weight()
    fig_journey()
    fig_upgrade()
    fig_decompose()
    fig_gapz_bins()
    fig_errors()
    fig_monthly()
    fig_importance()
    fig_noise()
    (AN / "numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log_wandb()
