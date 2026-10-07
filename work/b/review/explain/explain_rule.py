"""제출 규칙(gap_z + ext_range_z) 설명가능성 분석 — 강의 w6-1 적용. 02 문서 1~3번.

    uv run python work/b/review/explain/explain_rule.py

1. 예측 분해표 (w6-1 p5, p7): fold4 val 예시 3행을 단계별로 + 출렁임 보정이 바꾼 행 집계
2. 블록 permutation importance (w6-1 p15~18): 4폴드 val, 30회, 섞는 법 2가지, gap_z·ext_range_z·둘 묶음
3. 그룹별 효과 (w6-1 p20~22): λ 를 바꿔 가며 그룹별 val 점수 + 출렁임 구간별 갭 방향 적중률

입력: cache/gapz_rule.parquet (build_gapz 를 guard 아래 전 기간 돌린 표, 22,450행), lab/folds.json,
      dataset/daily.parquet (거래일·vol20), dataset/earnings.parquet (실적 발표 밤 표시)
출력: work/b/review/explain/out/*.csv, figs/*.png
그룹·섞는 법·반복 수·λ 격자는 실행 전에 정함. val 점수를 보고 바꾸지 않음.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))
import gapz_rule as G  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
SEED, N_REP, BLOCK = 2026, 30, 5
LAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
LABELS = ["급하락", "하락", "보합", "상승", "급상승"]

# 차트 색 (dataviz 기본 팔레트 1~3번, 밝은 바탕)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


# ------------------------------------------------------------------ 데이터
def load():
    t = pd.read_parquet(ROOT / "cache" / "gapz_rule.parquet")
    t["label"] = t.label.astype(int)
    spec = json.loads((ROOT / "lab" / "folds.json").read_text(encoding="utf-8"))
    unseen = set(spec["unseen_symbols"])

    d = pd.read_parquet(ROOT / "dataset" / "daily.parquet", columns=["symbol", "date_et", "ret"])
    dates = pd.DatetimeIndex(sorted(d.date_et.unique()))
    nxt = pd.Series(dates[1:], index=dates[:-1])
    t["target"] = t.date.map(nxt)

    # vol20: build_gapz 와 같은 정의 (일간 수익률 20일 표준편차, 최소 10일)
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    vol = R.rolling(20, min_periods=10).std().stack().rename("vol20").reset_index()
    t = t.merge(vol.rename(columns={"date_et": "date"}), on=["date", "symbol"], how="left")
    t["gap"] = t.gap_z * t.vol20

    # 그룹 (라벨을 쓰지 않고 정의)
    e = pd.read_parquet(ROOT / "dataset" / "earnings.parquet", columns=["symbol", "known_at"])
    t["since"], t["cutoff"] = t.date + pd.Timedelta(hours=16), t.target + pd.Timedelta(hours=9, minutes=30)
    m = t[["symbol", "date", "since", "cutoff"]].merge(e, on="symbol")
    hit = m[(m.known_at >= m.since) & (m.known_at < m.cutoff)][["symbol", "date"]].drop_duplicates()
    t["earn"] = t.set_index(["symbol", "date"]).index.isin(hit.set_index(["symbol", "date"]).index)
    t["unseen"] = t.symbol.isin(unseen)
    mk = t.groupby("date").gap_z.mean().abs()
    t["mkt_big"] = t.date.map(mk >= mk.quantile(0.8))

    folds = []
    for f in spec["folds"]:
        tr = t[t.date.between(*f["train"]) & ~t.unseen]
        va = t[t.date.between(*f["val"])].copy()
        folds.append((f["name"], tr, va))
    return t.drop(columns=["since", "cutoff"]), folds


def fit_fixed(tr, lam):
    """λ 를 고정하고 a, b 만 train score 최대로."""
    gz, er, y = tr.gap_z.to_numpy(float), tr.ext_range_z.to_numpy(float), tr.label.to_numpy(int)
    med, sd = float(np.nanmedian(er)), float(np.nanstd(er))
    sc, a, b = G._fit_ab(G._score_s(gz, er, med, sd, lam), y)
    return {"lam": lam, "med": med, "sd": sd, "a": float(a), "b": float(b), "train_score": float(sc)}


def pen(y, p):
    return G.WEIGHT[np.asarray(y, int), np.asarray(p, int)]


# ------------------------------------------------------------------ 1. 예측 분해표
def part1(folds):
    name, tr, va = folds[3]
    P, P0 = G.fit_table(tr), fit_fixed(tr, 0.0)
    x = np.nan_to_num(np.clip((va.ext_range_z.to_numpy() - P["med"]) / P["sd"], -5, 5))
    va = va.assign(x_tilde=x, shrink=np.exp(-P["lam"] * x))
    va["s"] = va.gap_z * va.shrink
    va["pred"] = G.cut(va.s.to_numpy(), P["a"], P["b"])
    va["pred_gz"] = G.predict_table(va, P0)
    va["pen"], va["pen_gz"] = pen(va.label, va.pred), pen(va.label, va.pred_gz)

    # 예시 3행 (고르는 규칙을 먼저 정함)
    def middle(df, key):
        df = df.sort_values(key)
        return df.iloc[len(df) // 2]
    hits = va[(va.label == va.pred) & va.label.isin([0, 4])]
    opp = va[((va.label == 0) & (va.pred >= 3)) | ((va.label == 4) & (va.pred <= 1))]
    chg = va[(va.pred != va.pred_gz) & (va.pen < va.pen_gz)]
    ex = [("급등락을 맞힌 행 (|s| 중앙)", middle(hits.assign(k=hits.s.abs()), "k")),
          ("방향을 정반대로 찍은 행 (|gap_z| 중앙)", middle(opp.assign(k=opp.gap_z.abs()), "k")),
          ("출렁임 보정으로 등급이 바뀌어 벌점이 준 행 (x̃ 최대)", chg.sort_values("x_tilde").iloc[-1])]
    rows = []
    for title, r in ex:
        rows.append({
            "예시": title, "종목": r.symbol, "기준일": f"{r.date:%Y-%m-%d}", "대상일": f"{r.target:%Y-%m-%d}",
            "갭": f"{r.gap:+.2%}", "vol20": f"{r.vol20:.2%}", "gap_z": f"{r.gap_z:+.3f}",
            "ext_range_z": f"{r.ext_range_z:.3f}", "x̃": f"{r.x_tilde:+.2f}",
            "보정 exp(−λx̃)": f"{r.shrink:.3f}", "s": f"{r.s:+.3f}",
            "예측": LABELS[int(r.pred)], "보정 없을 때 예측": LABELS[int(r.pred_gz)],
            "정답": LABELS[int(r.label)], "실제 등락": f"{r.ret_pct:+.2f}%", "벌점": int(r.pen),
        })
    ex_df = pd.DataFrame(rows)

    changed = va[va.pred != va.pred_gz]
    summ = {
        "fold": name, "lam": P["lam"], "a": P["a"], "b": P["b"], "a_gz": P0["a"], "b_gz": P0["b"],
        "val_rows": len(va), "score": G.score(va.label, va.pred), "score_gz_only": G.score(va.label, va.pred_gz),
        "changed_rows": len(changed), "changed_share": len(changed) / len(va),
        "changed_pen_full": int(changed.pen.sum()), "changed_pen_gz": int(changed.pen_gz.sum()),
        "changed_better": int((changed.pen < changed.pen_gz).sum()),
        "changed_worse": int((changed.pen > changed.pen_gz).sum()),
        "changed_same": int((changed.pen == changed.pen_gz).sum()),
        "changed_mean_x_tilde": float(changed.x_tilde.mean()),
    }
    # 바뀐 방향: 보정 없을 때 → 보정 후
    trans = changed.groupby(["pred_gz", "pred"]).size().rename("n").reset_index()
    trans["바뀜"] = trans.pred_gz.map(lambda i: LABELS[i]) + " → " + trans.pred.map(lambda i: LABELS[i])
    return ex_df, summ, trans[["바뀜", "n"]].sort_values("n", ascending=False)


# ------------------------------------------------------------------ 2. permutation importance
def _perm_index(va, scheme, rng):
    """va 행 순서 기준의 새 인덱스 (값을 어디서 가져올지)."""
    idx = np.arange(len(va))
    out = idx.copy()
    if scheme == "같은 날 안에서 종목끼리":
        for _, g in va.reset_index(drop=True).groupby("date").groups.items():
            g = np.asarray(g)
            out[g] = rng.permutation(g)
    else:  # 종목 안에서 5일 블록 순서 섞기
        v = va.reset_index(drop=True)
        for _, g in v.groupby("symbol").groups.items():
            g = np.asarray(sorted(g, key=lambda i: v.date.iat[i]))
            blocks = [g[i:i + BLOCK] for i in range(0, len(g), BLOCK)]
            order = rng.permutation(len(blocks))
            src = np.concatenate([blocks[j] for j in order])
            # 블록 길이가 달라도 위치를 맞추기 위해 앞에서부터 채움
            out[g] = src[:len(g)]
    return out


def part2(folds):
    rng = np.random.default_rng(SEED)
    schemes = ["같은 날 안에서 종목끼리", "종목 안에서 5일 블록"]
    feats = {"gap_z": ["gap_z"], "ext_range_z": ["ext_range_z"], "둘 묶음": ["gap_z", "ext_range_z"]}
    rows = []
    for name, tr, va in folds:
        P = G.fit_table(tr)
        va = va.sort_values(["date", "symbol"]).reset_index(drop=True)
        y = va.label.to_numpy()
        base = G.score(y, G.predict_table(va, P))
        for sch in schemes:
            for rep in range(N_REP):
                ix = _perm_index(va, sch, rng)
                for fname, cols in feats.items():
                    v = va.copy()
                    for c in cols:
                        v[c] = va[c].to_numpy()[ix]
                    rows.append({"fold": name, "scheme": sch, "rep": rep, "feature": fname,
                                 "base": base, "importance": base - G.score(y, G.predict_table(v, P))})
    r = pd.DataFrame(rows)
    summ = (r.groupby(["feature", "scheme", "fold"]).importance.agg(["mean", "std"]).unstack("fold"))
    mean4 = r.groupby(["feature", "scheme", "rep"]).importance.mean().groupby(["feature", "scheme"])
    neg = r[r.feature == "ext_range_z"].groupby(["scheme", "fold"]).importance.apply(lambda s: (s <= 0).mean())
    return r, summ, mean4.agg(["mean", "std", "min", "max"]), neg


def fig2(r):
    m = r.groupby(["feature", "scheme", "rep"]).importance.mean().reset_index()  # 반복마다 4폴드 평균
    schemes = list(dict.fromkeys(r.scheme))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    for ax, feats, title in [(axes[0], ["gap_z", "둘 묶음"], "gap_z · 둘 묶음 (큰 축)"),
                             (axes[1], ["ext_range_z"], "ext_range_z (작은 축)")]:
        pos, ticks = 0, []
        for f in feats:
            for k, (sch, col) in enumerate(zip(schemes, [C1, C2])):
                v = m[(m.feature == f) & (m.scheme == sch)].importance
                bp = ax.boxplot(v, positions=[pos + k * 0.42], widths=0.32, patch_artist=True,
                                medianprops={"color": INK, "linewidth": 1.4},
                                whiskerprops={"color": INK2}, capprops={"color": INK2},
                                flierprops={"markeredgecolor": INK2, "markersize": 3})
                bp["boxes"][0].set(facecolor=col, edgecolor=SURF, alpha=0.9)
            ticks.append(pos + 0.21)
            pos += 1.2
        ax.set_xticks(ticks, feats)
        ax.set_title(title, loc="left")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_ylabel("점수 하락폭 (섞기 전 - 섞은 후)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [C1, C2]]
    fig.legend(h, schemes, loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("블록 permutation importance — 30회, 4폴드 val 평균", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "2_permutation.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ 3. 그룹별 효과
GROUPS = [("earn", "실적 발표 밤", "있음", "없음"),
          ("unseen", "처음 보는 종목", "처음 보는 10종목", "학습한 40종목"),
          ("mkt_big", "시장 전체 갭", "큰 날 (상위 20%)", "나머지")]


def part3(t, folds):
    preds = []
    for lam in LAM_GRID:
        for name, tr, va in folds:
            P = fit_fixed(tr, lam)
            preds.append(va.assign(lam=lam, fold=name, pred=G.predict_table(va, P)))
    p = pd.concat(preds, ignore_index=True)
    rows = []
    for lam, g in p.groupby("lam"):
        rows.append({"lam": lam, "group": "전체", "value": "전체", "n": len(g) // 1,
                     "score": G.score(g.label, g.pred)})
        for col, gname, yes, no in GROUPS:
            for flag, lab in [(True, yes), (False, no)]:
                h = g[g[col] == flag]
                rows.append({"lam": lam, "group": gname, "value": lab, "n": len(h),
                             "score": G.score(h.label, h.pred)})
    sweep = pd.DataFrame(rows)
    base = sweep[sweep.lam == 0].set_index(["group", "value"]).score
    sweep["delta_vs_lam0"] = sweep.score - sweep.set_index(["group", "value"]).index.map(base)

    # 출렁임 구간별 갭 방향 적중률 (보합이 아닌 날, 전 기간, 라벨은 결과로만 씀)
    d = t[(t.label != 2) & t.gap_z.notna() & (t.gap_z != 0)].copy()
    d["bin"] = pd.qcut(d.ext_range_z, 5, labels=[1, 2, 3, 4, 5])
    d["hit"] = np.sign(d.gap_z) == np.sign(d.ret_pct)
    hr = []
    for col, gname, yes, no in [("전체", "전체", "전체", None)] + GROUPS:
        for flag, lab in ([(None, "전체")] if col == "전체" else [(True, yes), (False, no)]):
            h = d if flag is None else d[d[col] == flag]
            s = h.groupby("bin", observed=True).hit.agg(["mean", "size"]).reset_index()
            s["group"], s["value"] = gname, lab
            hr.append(s)
    hit = pd.concat(hr, ignore_index=True).rename(columns={"mean": "hit_rate", "size": "n"})
    edges = d.ext_range_z.quantile([0, .2, .4, .6, .8, 1]).round(2).tolist()
    return sweep, hit, edges


def fig3(sweep, hit, lam_fit):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey="row")
    for j, (col, gname, yes, no) in enumerate(GROUPS):
        ax = axes[0, j]
        for lab, c in [(yes, C2), (no, C1)]:
            s = sweep[(sweep.group == gname) & (sweep.value == lab)]
            ax.plot(s.lam, s.delta_vs_lam0, color=c, linewidth=2, marker="o", markersize=4,
                    label=f"{lab} (n={int(s.n.iloc[0]):,})")
        ax.legend(fontsize=8, loc="lower left")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvspan(min(lam_fit), max(lam_fit), color=GRID, alpha=0.6, linewidth=0)
        ax.set_title(gname, loc="left")
        ax.set_xlabel("λ (출렁임 보정 세기)")
        if j == 0:
            ax.set_ylabel("val 점수 변화 (λ=0 대비)")
        ax = axes[1, j]
        for lab, c in [(yes, C2), (no, C1)]:
            s = hit[(hit.group == gname) & (hit.value == lab)]
            ax.plot(s.bin.astype(int), s.hit_rate, color=c, linewidth=2, marker="o", markersize=4, label=lab)
        ax.set_xticks([1, 2, 3, 4, 5], ["1\n작음", "2", "3", "4", "5\n큼"])
        ax.set_xlabel("ext_range_z 5분위 (시간외 출렁임)")
        if j == 0:
            ax.set_ylabel("갭 방향 = 실제 방향 비율")
    fig.suptitle("그룹별 효과\n위: λ 를 바꿨을 때 val 점수 변화 (4폴드 합산, 회색 띠 = 폴드별로 고른 λ 범위)\n"
                 "아래: 보합이 아닌 날, 시간외 출렁임 구간별로 갭 방향이 실제 방향과 같았던 비율 (색은 위와 같음)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGS / "3_groups.png", dpi=160)
    plt.close(fig)


def mechanism(t):
    """보정이 어디서 효과를 내는지: |gap_z| 3분위 × ext_range_z 5분위 표 (전 기간, 설명용)."""
    d = t[t.gap_z.notna() & (t.gap_z != 0)].copy()
    d["출렁임 5분위"] = pd.qcut(d.ext_range_z, 5, labels=[1, 2, 3, 4, 5])
    d["|gap_z| 3분위"] = pd.qcut(d.gap_z.abs(), 3, labels=["작음", "중간", "큼"])
    d["same_big"] = ((d.label == 4) & (d.gap_z > 0)) | ((d.label == 0) & (d.gap_z < 0))
    d["opp_big"] = ((d.label == 0) & (d.gap_z > 0)) | ((d.label == 4) & (d.gap_z < 0))
    out = {}
    for v, name in [("same_big", "갭과 같은 방향 급등락 비율"), ("opp_big", "갭과 반대 방향 급등락 비율")]:
        out[name] = d.pivot_table(index="|gap_z| 3분위", columns="출렁임 5분위", values=v,
                                  aggfunc="mean", observed=True)
    out["행 수"] = d.pivot_table(index="|gap_z| 3분위", columns="출렁임 5분위", values="same_big",
                               aggfunc="size", observed=True)
    rho = d.gap_z.abs().corr(d.ext_range_z, method="spearman")
    return out, rho


# ------------------------------------------------------------------ 실행
def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t, folds = load()
    print("표", t.shape, "| 실적 발표 밤", int(t.earn.sum()), "| 시장 갭 큰 날 행", int(t.mkt_big.sum()))

    lam_fit = []
    for name, tr, va in folds:
        P = G.fit_table(tr)
        lam_fit.append(P["lam"])
        print(f"{name}: λ {P['lam']}, a {P['a']:.3f}, b {P['b']:.3f}, val {G.score(va.label, G.predict_table(va, P)):.4f}")

    ex, summ, trans = part1(folds)
    ex.to_csv(OUT / "1_examples.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(OUT / "1_changed_transitions.csv", index=False, encoding="utf-8-sig")
    (OUT / "1_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[1]", json.dumps(summ, ensure_ascii=False))
    print(ex.T.to_string())
    print(trans.to_string(index=False))

    r, s2, m4, neg = part2(folds)
    r.to_csv(OUT / "2_permutation_raw.csv", index=False, encoding="utf-8-sig")
    s2.to_csv(OUT / "2_permutation_by_fold.csv", encoding="utf-8-sig")
    m4.to_csv(OUT / "2_permutation_mean4.csv", encoding="utf-8-sig")
    fig2(r)
    print("\n[2] 4폴드 평균 (30회)\n", m4.round(4).to_string())
    print(s2.round(4).to_string())
    print("ext_range_z 하락폭 ≤ 0 비율\n", neg.round(2).to_string())

    sweep, hit, edges = part3(t, folds)
    sweep.to_csv(OUT / "3_lambda_sweep.csv", index=False, encoding="utf-8-sig")
    hit.to_csv(OUT / "3_hit_by_ext_bin.csv", index=False, encoding="utf-8-sig")
    fig3(sweep, hit, lam_fit)
    print("\n[3] ext_range_z 5분위 경계", edges)
    print(sweep.pivot_table(index=["group", "value"], columns="lam", values="delta_vs_lam0").round(4).to_string())
    print(sweep[sweep.lam == 0][["group", "value", "n", "score"]].round(4).to_string(index=False))
    print(hit.pivot_table(index=["group", "value"], columns="bin", values="hit_rate", observed=True).round(3).to_string())

    mech, rho = mechanism(t)
    print(f"\n[3+] |gap_z| 와 ext_range_z 순위상관 {rho:.3f}")
    with open(OUT / "3_mechanism.csv", "w", encoding="utf-8-sig") as f:
        for name, tab in mech.items():
            f.write(f"# {name}\n")
            tab.round(3).to_csv(f)
            print(f"-- {name}\n{tab.round(3).to_string()}")


if __name__ == "__main__":
    main()
