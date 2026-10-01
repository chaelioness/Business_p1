"""6. 평가지표 관점의 의사결정 + 처음 보는 종목(unseen) 점검.

점수 = 1 − Σw·O / Σw·E. 분모 E 는 예측의 열 합(얼마나 자주 각 등급을 찍나)에 따라 변하므로
"얼마나 공격적으로 급등락을 찍을지" 자체가 의사결정 변수임. 여기서는 EDA 구간 안에서
신호 하나(ovn_gap)로 그 지형을 그려 봄. ※ 이 구간에서 고른 임계값을 그대로 쓰면 안 됨.
실제 임계값은 폴드마다 train 에서만 다시 골라야 함 (README 피처 규칙 6).
"""

from eda.common import *  # noqa: F403

S = Section("s06_metric_unseen")


def cut(x, a, b):
    out = np.full(len(x), 2)
    out[x >= a] = 3
    out[x >= b] = 4
    out[x <= -a] = 1
    out[x <= -b] = 0
    return out


def contributions(d):
    g = d.dropna(subset=["ovn_gap"])
    t, p = g.label.to_numpy(), label_of(g.ovn_gap).astype(int).to_numpy()
    O = np.zeros((5, 5))
    np.add.at(O, (t, p), 1)
    E = np.outer(O.sum(1), O.sum(0)) / len(t)
    wo, we = WEIGHT * O, WEIGHT * E
    rows = pd.DataFrame({"정답": LABEL_NAMES, "Σw·O (벌점)": wo.sum(1), "Σw·E (기대 벌점)": we.sum(1)})
    rows["벌점 비중 %"] = rows["Σw·O (벌점)"] / wo.sum() * 100
    S.tab(rows.set_index("정답"), "tab_penalty_by_true_row_naive_gap")
    S.note("penalty_share_from_big_rows", (wo[0].sum() + wo[4].sum()) / wo.sum())


def aggressiveness(d):
    """pred = label_of(k · ovn_gap). k 가 크면 급등락을 더 자주 찍음."""
    g = d.dropna(subset=["ovn_gap"])
    ks = [.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
    rows = []
    for k in ks:
        r = score(g.label, label_of(k * g.ovn_gap).astype(int))
        r["k"] = k
        r["pred_big_share"] = np.isin(label_of(k * g.ovn_gap).astype(int), [0, 4]).mean()
        rows.append(r)
    t = pd.DataFrame(rows).set_index("k")
    S.tab(t, "tab_score_vs_aggressiveness")

    # (a, b) 대칭 임계값 지형
    A = np.array([.1, .2, .3, .4, .5, .6, .8, 1.0])
    B = np.array([.6, .8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    grid = np.full((len(A), len(B)), np.nan)
    x, y = g.ovn_gap.to_numpy(), g.label.to_numpy()
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if b > a:
                grid[i, j] = score(y, cut(x, a, b))["score"]
    gz = pd.DataFrame(grid, index=[f"a={v}" for v in A], columns=[f"b={v}" for v in B])
    S.tab(gz, "tab_threshold_grid_ovn_gap")
    i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
    S.note("best_in_sample_threshold_ab", [float(A[i]), float(B[j]), float(grid[i, j])])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t.index, t.score, "o-", label="score")
    ax[0].plot(t.index, t.big_recall, "s--", label="big_recall")
    ax[0].plot(t.index, t.big_prec, "^--", label="big_prec")
    ax[0].plot(t.index, t.pred_big_share, ":", color="k", label="급등락 예측 비율")
    ax[0].set(xscale="log", title="공격성 k: pred = label_of(k·ovn_gap)", xlabel="k (log)")
    ax[0].legend(fontsize=7)
    im = ax[1].imshow(grid, cmap="viridis", origin="lower", aspect="auto")
    ax[1].set_xticks(range(len(B)), B)
    ax[1].set_yticks(range(len(A)), A)
    ax[1].set(xlabel="b: 급등락 경계 (|gap| ≥ b)", ylabel="a: 상승/하락 경계 (|gap| ≥ a)",
              title="ovn_gap 임계값 지형 (EDA 구간, in-sample)")
    ax[1].grid(False)
    for (ii, jj), v in np.ndenumerate(grid):
        if not np.isnan(v):
            ax[1].text(jj, ii, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax[1])
    S.fig(fig, "fig_metric_landscape")
    return A[i], B[j]


def vol_scaled(d, a, b):
    """같은 갭이라도 고변동 종목이면 급등락 가능성이 큼 → 갭/σ 로 경계를 잡으면?"""
    g = d.dropna(subset=["ovn_gap", "vol20"]).copy()
    y = g.label.to_numpy()
    base = score(y, cut(g.ovn_gap.to_numpy(), a, b))["score"]
    best = (-9, None)
    z = g.ovn_gap_z.to_numpy()
    for az in [.05, .1, .15, .2, .25, .3, .4]:
        for bz in [.4, .5, .6, .75, 1, 1.25, 1.5]:
            if bz > az:
                s = score(y, cut(z, az, bz))["score"]
                best = max(best, (s, (az, bz)))
    S.note("score_raw_gap_best_vs_z_gap_best", [base, best[0], list(best[1])])

    # 실적 플래그로 급등락 쪽으로 밀어 보기: 실적일은 갭 부호대로 급등락
    p = cut(g.ovn_gap.to_numpy(), a, b)
    e = g.earn.to_numpy()
    p2 = p.copy()
    p2[e & (g.ovn_gap.to_numpy() > 0.5)] = 4
    p2[e & (g.ovn_gap.to_numpy() < -0.5)] = 0
    S.note("score_gap_vs_gap_plus_earn_override", [base, score(y, p2)["score"]])


def stability(d, a, b):
    g = d.dropna(subset=["ovn_gap"]).copy()
    g["pred"] = cut(g.ovn_gap.to_numpy(), a, b)
    m = g.groupby("month").apply(lambda x: pd.Series(score(x.label, x.pred)), include_groups=False)
    m["p_big"] = g.groupby("month").label.apply(lambda s: s.isin([0, 4]).mean())
    S.tab(m, "tab_monthly_score_gap_rule")
    S.note("monthly_score_min_max", [m.score.min(), m.score.max()])
    S.note("corr_monthly_score_vs_pbig", m.score.corr(m.p_big))

    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.bar(m.index, m.score, color="#4c72b0", label="월별 score (갭 규칙)")
    ax2 = ax.twinx()
    ax2.plot(m.index, m.p_big * 100, "k-o", ms=3, label="급등락 %")
    ax2.grid(False)
    ax.set(title="같은 규칙도 월마다 score 가 크게 흔들림 — 국면 의존", ylabel="score")
    ax2.set_ylabel("급등락 %")
    ax.tick_params(axis="x", rotation=70)
    S.fig(fig, "fig_monthly_score")

    rows = {}
    for name, x in [("seen 40", g[~g.unseen]), ("unseen 10", g[g.unseen])]:
        rows[name] = score(x.label, x.pred)
    S.tab(pd.DataFrame(rows).T, "tab_seen_vs_unseen_score_gap_rule")


def unseen_shift(d):
    """unseen 10종목의 피처 분포가 seen 과 다른가 (KS). 처음 보는 종목 일반화의 위험 신호."""
    rows = {}
    for c in ["ret_pct", "vol20", "atr14", "ovn_gap", "news_n", "dvol20", "r20", "volu_ratio"]:
        a, b = d.loc[~d.unseen, c].dropna(), d.loc[d.unseen, c].dropna()
        ks = stats.ks_2samp(a, b)
        rows[c] = {"seen_median": a.median(), "unseen_median": b.median(), "KS": ks.statistic, "p": ks.pvalue}
    S.tab(pd.DataFrame(rows).T, "tab_unseen_distribution_shift")
    # 종목 하나만 있는 피처(가격 수준, 거래대금)는 unseen 에서 범위를 벗어날 수 있음
    S.note("price_level_range_seen", [d.loc[~d.unseen, "close_B"].min(), d.loc[~d.unseen, "close_B"].max()])


def main():
    print("[s06] 평가지표·unseen")
    d = panel()
    contributions(d)
    a, b = aggressiveness(d)
    vol_scaled(d, a, b)
    stability(d, a, b)
    unseen_shift(d)
    S.save()


if __name__ == "__main__":
    main()
