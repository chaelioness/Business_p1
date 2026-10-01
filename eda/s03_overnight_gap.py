"""3. 시간외(overnight) 갭 — 09:30 에 알 수 있는 가장 강한 정보.

대상일 수익률 = (시가/전일종가) 갭 + (종가/시가) 장중. 갭의 상당 부분은 개장 전 시간외
체결가로 이미 드러남. 볼 것:
  - 수익률 분산 중 갭이 차지하는 몫, 그 갭을 08:00 pre 봉이 얼마나 미리 보여 주나
  - 갭 정의 비교: 전일 종가 기준(ovn_gap) vs pre 세션 안에서의 등락(pre_move, 현재 기준선)
  - 갭 이후 장중에 이어지나(모멘텀) 되돌리나(반전), 시장 갭과 개별 갭이 다르게 움직이나
"""

from eda.common import *  # noqa: F403

S = Section("s03_overnight_gap")

GAPS = ["ovn_gap", "post_ret", "pre_vs_post", "pre_move", "ovn_gap_z", "mkt_gap", "rel_gap"]
GAP_DESC = {
    "ovn_gap": "전일 종가 → cutoff 직전 마지막 시간외 체결가",
    "post_ret": "전일 종가 → 전일 post 세션 마지막가",
    "pre_vs_post": "post 마지막가 → 당일 pre 마지막가",
    "pre_move": "당일 pre 첫 봉 → 마지막 봉 (현재 gap 기준선 정의)",
    "ovn_gap_z": "ovn_gap / vol20 (변동성 정규화)",
    "mkt_gap": "50종목 평균 ovn_gap (시장 갭)",
    "rel_gap": "ovn_gap − 시장 갭 (개별 갭)",
}


def decomposition(d):
    v = d.ret_pct.var()
    S.note("var_share_open_gap", d.y_gap_open.var() / v)
    S.note("var_share_intraday", d.y_intraday.var() / v)
    S.note("corr_open_gap_intraday", d.y_gap_open.corr(d.y_intraday))
    S.note("r2_ovn_gap_to_open_gap", ols_r2(d.ovn_gap, d.y_gap_open))
    S.note("r2_ovn_gap_to_ret", ols_r2(d.ovn_gap, d.ret_pct))
    big = d.label.isin([0, 4])
    S.note("big_days_share_where_|open_gap|>1.5", (d.y_gap_open[big].abs() > 1.5).mean())

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    lim = 12
    ax[0].scatter(d.ovn_gap.clip(-lim, lim), d.y_gap_open.clip(-lim, lim), s=2, alpha=.3)
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    ax[0].set(title=f"08:00 시간외 갭 vs 실제 시가 갭 (R²={ols_r2(d.ovn_gap, d.y_gap_open):.2f})",
              xlabel="ovn_gap %", ylabel="시가 갭 %")
    ax[1].scatter(d.ovn_gap.clip(-lim, lim), d.ret_pct.clip(-lim, lim), s=2, alpha=.3, c="#55a868")
    for c in RET_CUTS:
        ax[1].axhline(c, color="r", lw=.6, ls=":")
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    ax[1].set(title=f"ovn_gap vs 대상일 수익률 (R²={ols_r2(d.ovn_gap, d.ret_pct):.2f})",
              xlabel="ovn_gap %", ylabel="ret %")
    ax[2].scatter(d.y_gap_open.clip(-lim, lim), d.y_intraday.clip(-lim, lim), s=2, alpha=.3, c="#c44e52")
    ax[2].set(title=f"시가 갭 vs 장중 수익률 (corr={d.y_gap_open.corr(d.y_intraday):.3f})",
              xlabel="시가 갭 %", ylabel="장중 %")
    S.fig(fig, "fig_gap_decomposition")


def compare_definitions(d):
    from sklearn.metrics import roc_auc_score
    rows = []
    for c in GAPS:
        ic, _ = daily_ic(d, c)
        g = d[[c, "label", "ret_pct"]].dropna()
        lab = g.label
        rows.append({
            "feature": c, "설명": GAP_DESC[c],
            "spearman_pooled": pooled_corr(d, c),
            "daily_IC": ic["ic_mean"], "IC_t": ic["t"], "IC_hit": ic["hit"],
            "AUC_surge(4 vs rest)": roc_auc_score(lab.eq(4), g[c]),
            "AUC_crash(0 vs rest)": roc_auc_score(lab.eq(0), -g[c]),
            "spearman_|x|_vs_|ret|": pooled_corr(d.assign(a=d[c].abs(), b=d.ret_pct.abs()), "a", "b"),
            "coverage": d[c].notna().mean(),
        })
    t = pd.DataFrame(rows).set_index("feature")
    S.tab(t, "tab_gap_definitions")

    # 반기별 안정성
    st = pd.DataFrame({h: {c: daily_ic(g, c)[0]["ic_mean"] for c in GAPS} for h, g in d.groupby("half")})
    S.tab(st, "tab_gap_ic_by_half")


def buckets(d):
    edges = [-np.inf, -5, -2.5, -1.5, -0.6, -0.3, 0.3, 0.6, 1.5, 2.5, 5, np.inf]
    names = ["<-5", "-5~-2.5", "-2.5~-1.5", "-1.5~-0.6", "-0.6~-0.3", "±0.3",
             "0.3~0.6", "0.6~1.5", "1.5~2.5", "2.5~5", ">5"]
    g = d.dropna(subset=["ovn_gap"]).copy()
    g["b"] = pd.cut(g.ovn_gap, edges, labels=names)
    m = g.groupby("b", observed=True).label.value_counts(normalize=True).unstack().fillna(0)
    n = g.groupby("b", observed=True).size()
    extra = g.groupby("b", observed=True).agg(ret=("ret_pct", "mean"), intraday=("y_intraday", "mean"),
                                              open_gap=("y_gap_open", "mean"))
    t = (m * 100).round(1)
    t.columns = LABEL_NAMES
    S.tab(t.join(extra).assign(n=n), "tab_label_by_ovn_gap_bucket")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    bottom = np.zeros(len(m))
    for k in range(5):
        ax[0].bar(range(len(m)), m[k], bottom=bottom, color=LABEL_COLORS[k], label=LABEL_NAMES[k])
        bottom += m[k].values
    for i, v in enumerate(n):
        ax[0].text(i, 1.01, f"{v}", ha="center", fontsize=6)
    ax[0].set_xticks(range(len(m)), m.index, rotation=45)
    ax[0].set(title="ovn_gap 구간별 대상일 label 구성", xlabel="ovn_gap % 구간")
    ax[0].legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.42))

    # 장중 이어지나/되돌리나 — 시장 갭 vs 개별 갭
    for col, lab, c in [("rel_gap", "개별 갭(rel_gap)", "#4c72b0"), ("mkt_gap", "시장 갭(mkt_gap)", "#dd8452")]:
        q = g.dropna(subset=[col]).copy()
        q["q"] = pd.qcut(q[col].rank(method="first"), 10, labels=False)
        r = q.groupby("q").agg(x=(col, "mean"), intra=("y_intraday", "mean"),
                               se=("y_intraday", lambda s: s.std() / np.sqrt(len(s))))
        ax[1].errorbar(r.x, r.intra, yerr=1.96 * r.se, fmt="o-", ms=3, label=lab, color=c)
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set(title="갭 10분위 → 장중(시가→종가) 평균 수익률", xlabel="갭 % (분위 평균)", ylabel="장중 %")
    ax[1].legend()
    S.fig(fig, "fig_gap_buckets_and_intraday")

    # 갭 크기를 종목 σ 로 보면 반전/지속이 갈리나
    q = g.copy()
    q["zq"] = pd.cut(q.ovn_gap_z, [-np.inf, -2, -1, -.5, .5, 1, 2, np.inf])
    t2 = q.groupby("zq", observed=True).agg(n=("ret_pct", "size"), gap=("ovn_gap", "mean"),
                                            open_gap=("y_gap_open", "mean"), intraday=("y_intraday", "mean"),
                                            ret=("ret_pct", "mean"),
                                            keep_sign=("ret_pct", lambda s: np.nan))
    keep = q.groupby("zq", observed=True).apply(
        lambda x: (np.sign(x.ret_pct) == np.sign(x.ovn_gap)).mean(), include_groups=False)
    t2["keep_sign"] = keep
    S.tab(t2, "tab_gap_z_continuation")


def naive_mapping(d):
    """갭을 label 경계에 그대로 대입했을 때 (튜닝 없음) — 신호의 '원석' 점수."""
    rows = {}
    for c in ["ovn_gap", "pre_vs_post", "pre_move", "post_ret"]:
        g = d.dropna(subset=[c])
        pred = label_of(g[c]).astype(int)
        rows[c] = score(g.label, pred)
    t = pd.DataFrame(rows).T
    S.tab(t, "tab_naive_gap_to_label_score")
    g = d.dropna(subset=["ovn_gap"])
    cm = pd.crosstab(g.label, label_of(g.ovn_gap).astype(int), normalize="index")
    cm.index = [f"정답 {x}" for x in LABEL_NAMES]
    cm.columns = [f"예측 {x}" for x in LABEL_NAMES]
    S.tab((cm * 100).round(1), "tab_naive_ovn_gap_confusion_rownorm")
    S.note("naive_ovn_gap_score", rows["ovn_gap"]["score"])
    S.note("naive_pre_move_score", rows["pre_move"]["score"])


def event_split(d):
    """실적 발표가 걸린 날 vs 아닌 날 갭의 설명력."""
    rows = {}
    for name, g in [("실적일", d[d.earn]), ("일반일", d[~d.earn])]:
        rows[name] = {"n": len(g), "r2_ovn_gap_ret": ols_r2(g.ovn_gap, g.ret_pct),
                      "mean_|gap|": g.ovn_gap.abs().mean(), "p_big": g.label.isin([0, 4]).mean(),
                      "naive_score": score(g.label, label_of(g.ovn_gap.fillna(0)).astype(int))["score"]}
    S.tab(pd.DataFrame(rows).T, "tab_gap_earnings_vs_normal")


def main():
    print("[s03] 시간외 갭")
    d = panel()
    decomposition(d)
    compare_definitions(d)
    buckets(d)
    naive_mapping(d)
    event_split(d)
    S.save()


if __name__ == "__main__":
    main()
