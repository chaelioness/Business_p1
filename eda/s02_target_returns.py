"""2. 타깃과 수익률의 성질 (stylized facts).

점수는 사실상 급등락(label 0/4) 행에서 결정됨. 그래서 볼 것:
  - 급등락이 '누구에게(종목)', '언제(국면·요일)', '어떻게(시장 전체 vs 개별)' 일어나나
  - 수익률은 정규분포가 아니고(두꺼운 꼬리), 변동성은 뭉쳐 다님(volatility clustering)
  - 방향(부호)은 거의 예측 불가, 크기(|ret|)는 예측 가능 → 모델 설계의 출발점
"""

from eda.common import *  # noqa: F403

S = Section("s02_target_returns")


def distribution(d):
    t = pd.DataFrame({"EDA 구간": label_dist(d.label)})
    for h, g in d.groupby("half"):
        t[h] = label_dist(g.label)
    t["seen"] = label_dist(d[~d.unseen].label)
    t["unseen"] = label_dist(d[d.unseen].label)
    S.tab((t * 100).round(2), "tab_label_dist")

    r = d.ret_pct
    S.note("ret_mean_sd_skew_kurt", [round(r.mean(), 4), round(r.std(), 4),
                                     round(stats.skew(r), 3), round(stats.kurtosis(r), 2)])
    z = (r - r.mean()) / r.std()
    S.note("tail_ratio_|z|>3_vs_normal", round(float((z.abs() > 3).mean() / (2 * stats.norm.sf(3))), 2))

    # 월별 라벨 구성 + 급등락 비율
    m = d.groupby("month").label.value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    bottom = np.zeros(len(m))
    for k in range(5):
        ax.bar(m.index, m[k], bottom=bottom, color=LABEL_COLORS[k], label=LABEL_NAMES[k])
        bottom += m[k].values
    ax2 = ax.twinx()
    ax2.plot(m.index, (m[0] + m[4]) * 100, "k-o", ms=3, label="급등락 %")
    ax2.set_ylabel("급등락 비율 %")
    ax2.grid(False)
    ax.set(title="월별 label 구성 — 급등락 비중이 국면에 따라 2배 이상 변함", ylabel="비율")
    ax.tick_params(axis="x", rotation=70)
    ax.legend(ncol=5, fontsize=7, loc="lower left")
    S.fig(fig, "fig_label_by_month")
    S.tab((m * 100).round(2).assign(big=lambda x: x[0] + x[4]), "tab_label_by_month")

    # 수익률 분포 vs 정규
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    bins = np.linspace(-10, 10, 161)
    ax[0].hist(r.clip(-10, 10), bins=bins, density=True, color="#4c72b0", alpha=.7, label="실제")
    xs = np.linspace(-10, 10, 400)
    ax[0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "k--", lw=1, label="정규(같은 σ)")
    for c in RET_CUTS:
        ax[0].axvline(c, color="r", lw=.8, ls=":")
    ax[0].set(title="대상일 수익률 분포와 label 경계", xlabel="ret %", yscale="log", ylim=(1e-4, 1))
    ax[0].legend()
    (osm, osr), _ = stats.probplot(z.sample(5000, random_state=0), dist="norm")
    ax[1].scatter(osm, osr, s=3)
    ax[1].plot([-4, 4], [-4, 4], "k--", lw=.8)
    ax[1].set(title="QQ plot (표준화 수익률)", xlabel="정규 분위", ylabel="실제 분위")
    S.fig(fig, "fig_return_distribution")


def by_symbol(d):
    t = d.groupby("symbol").agg(vol=("ret_pct", "std"), p_big=("label", lambda s: s.isin([0, 4]).mean()),
                                p_crash=("label", lambda s: s.eq(0).mean()),
                                p_surge=("label", lambda s: s.eq(4).mean()),
                                p_flat=("label", lambda s: s.eq(2).mean()),
                                unseen=("unseen", "first")).sort_values("vol")
    S.tab(t, "tab_symbol_label_profile")
    big = d[d.label.isin([0, 4])]
    share = big.symbol.value_counts(normalize=True)
    S.note("big_moves_share_top10_symbols", round(float(share.head(10).sum()), 3))
    S.note("p_big_min_max_by_symbol", [round(t.p_big.min(), 3), round(t.p_big.max(), 3)])
    S.note("corr_symbol_vol_vs_pbig", round(float(t.vol.corr(t.p_big)), 3))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    c = np.where(t.unseen, "#dd8452", "#4c72b0")
    ax[0].scatter(t.vol, t.p_big * 100, c=c, s=22)
    for s, row in t.iterrows():
        ax[0].annotate(s, (row.vol, row.p_big * 100), fontsize=6, alpha=.75)
    ax[0].set(title="종목 일변동성 vs 급등락 빈도 (주황=unseen)", xlabel="일수익률 σ %", ylabel="급등락 %")
    lo = t.sort_values("p_big")
    ax[1].barh(lo.index, lo.p_crash * 100, color=LABEL_COLORS[0], label="급하락")
    ax[1].barh(lo.index, lo.p_surge * 100, left=lo.p_crash * 100, color=LABEL_COLORS[4], label="급상승")
    ax[1].set(title="종목별 급등락 비율 — 고정 ±2.5% 임계값이라 고변동 종목에 몰림", xlabel="%")
    ax[1].tick_params(axis="y", labelsize=6)
    ax[1].legend()
    S.fig(fig, "fig_symbol_vol_vs_big")


def clustering(d):
    """크기는 예측 가능(변동성 군집), 방향은 거의 불가."""
    d = d.sort_values(["symbol", "target"])
    lags = range(1, 11)
    acf_r, acf_a = [], []
    for k in lags:
        acf_r.append(d.groupby("symbol").ret_pct.apply(lambda s: s.autocorr(k)).mean())
        acf_a.append(d.groupby("symbol").ret_pct.apply(lambda s: s.abs().autocorr(k)).mean())
    ci = 1.96 / np.sqrt(d.groupby("symbol").size().mean())
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].bar(np.array(lags) - .2, acf_r, .4, label="ret")
    ax[0].bar(np.array(lags) + .2, acf_a, .4, label="|ret|")
    ax[0].axhspan(-ci, ci, color="grey", alpha=.2)
    ax[0].set(title="종목 평균 자기상관 — 방향 ≈0, 크기 >0", xlabel="lag(거래일)")
    ax[0].legend()
    S.note("acf_ret_lag1", round(acf_r[0], 4))
    S.note("acf_absret_lag1_5_10", [round(acf_a[0], 3), round(acf_a[4], 3), round(acf_a[9], 3)])

    # vol20 분위별 급등락 확률
    d = d.copy()
    d["big"] = d.label.isin([0, 4])
    d["q_vol"] = pd.qcut(d.vol20, 10, labels=False) + 1
    t = d.groupby("q_vol").agg(vol20=("vol20", "mean"), p_big=("big", "mean"),
                               p_crash=("label", lambda s: s.eq(0).mean()),
                               p_surge=("label", lambda s: s.eq(4).mean()), p_flat=("label", lambda s: s.eq(2).mean()))
    S.tab(t, "tab_pbig_by_vol20_decile")
    ax[1].plot(t.vol20, t.p_big * 100, "o-", label="급등락")
    ax[1].plot(t.vol20, t.p_flat * 100, "s--", color="grey", label="보합")
    ax[1].set(title="직전 20일 변동성 10분위 → 다음날 급등락 확률", xlabel="vol20 평균 %", ylabel="%")
    ax[1].legend()
    S.note("p_big_vol20_decile1_vs_10", [round(t.p_big.iloc[0], 3), round(t.p_big.iloc[-1], 3)])

    # 변동성 정규화 수익률: 고정 임계값을 종목 σ 로 나누면?
    d["thr_in_sigma"] = 2.5 / d.vol20
    ax[2].hist(d.thr_in_sigma.clip(0, 6), bins=60, color="#8172b2")
    ax[2].set(title="±2.5% 임계값이 종목 σ 의 몇 배인가", xlabel="2.5 / vol20")
    S.note("thr_in_sigma_median", round(float(d.thr_in_sigma.median()), 2))
    S.fig(fig, "fig_vol_clustering")

    # AUC: 변동성 피처만으로 급등락 여부를 얼마나 가르나 (방향 없이)
    from sklearn.metrics import roc_auc_score
    rows = {}
    for c in ["vol20", "vol60", "vol5", "atr14", "range1"]:
        g = d[[c, "big"]].dropna()
        rows[c] = roc_auc_score(g.big, g[c])
    S.tab(pd.Series(rows, name="AUC_big").to_frame(), "tab_auc_big_vol_features")
    S.note("auc_big_atr14", round(rows["atr14"], 3))


def market_factor(d):
    """급등락이 시장 전체 이벤트인가 개별 이벤트인가."""
    r2 = d.groupby("symbol").apply(lambda g: ols_r2(g.y_mkt, g.ret_pct), include_groups=False)
    S.note("r2_on_market_median", round(float(r2.median()), 3))
    var_share = d.y_mkt.var() / d.ret_pct.var()
    S.note("market_var_share_pooled", round(float(var_share), 3))

    day = d.groupby("target").agg(mkt=("y_mkt", "first"),
                                  n_big=("label", lambda s: s.isin([0, 4]).sum()),
                                  disp=("ret_pct", "std"))
    day["mkt_big"] = day.mkt.abs() > 1.5
    S.note("share_big_moves_on_|mkt|>1.5_days", round(float(day.loc[day.mkt_big, "n_big"].sum() / day.n_big.sum()), 3))
    S.note("share_days_|mkt|>1.5", round(float(day.mkt_big.mean()), 3))
    # 급등락이 집중되는 날: 상위 5% 날이 전체 급등락의 몇 %?
    top = day.n_big.sort_values(ascending=False)
    S.note("share_big_in_top5pct_days", round(float(top.head(int(len(top) * .05)).sum() / top.sum()), 3))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].scatter(day.mkt, day.n_big, s=8, alpha=.6)
    ax[0].set(title="시장(등가중) 수익률 vs 그날 급등락 종목 수", xlabel="시장 수익률 %", ylabel="급등락 종목 수 (/50)")
    ax[1].plot(day.index, day.disp.rolling(10).mean(), label="횡단면 분산(σ) 10일")
    ax[1].plot(day.index, day.mkt.abs().rolling(10).mean(), label="|시장| 10일")
    ax[1].set(title="시장 변동 vs 종목 간 분산 — 개별 이벤트 비중")
    ax[1].legend()
    S.fig(fig, "fig_market_vs_idio")

    # 상관 행렬 (군집 순서)
    w = d.pivot(index="target", columns="symbol", values="ret_pct")
    c = w.corr()
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = c.columns[leaves_list(linkage(1 - c.values[np.triu_indices(len(c), 1)], "average"))]
    c = c.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(c)), c.index, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, shrink=.7)
    ax.set_title("종목 일수익률 상관 (계층 군집 순서) — 반도체·금융·방어주 블록")
    S.fig(fig, "fig_corr_matrix")
    S.note("avg_pairwise_corr", round(float(c.values[np.triu_indices(len(c), 1)].mean()), 3))


def calendar(d):
    t = d.groupby("weekday").agg(p_big=("label", lambda s: s.isin([0, 4]).mean()),
                                 ret=("ret_pct", "mean"), absret=("ret_pct", lambda s: s.abs().mean()), n=("ret_pct", "size"))
    t.index = ["월", "화", "수", "목", "금"]
    S.tab(t, "tab_weekday")


def transitions(d):
    d = d.sort_values(["symbol", "target"])
    prev = d.groupby("symbol").label.shift()
    m = pd.crosstab(prev.dropna().astype(int), d.label[prev.notna()], normalize="index")
    m.index = LABEL_NAMES
    m.columns = LABEL_NAMES
    S.tab((m * 100).round(1), "tab_label_transition")


def main():
    print("[s02] 타깃·수익률")
    d = panel()
    distribution(d)
    by_symbol(d)
    clustering(d)
    market_factor(d)
    calendar(d)
    transitions(d)
    S.save()


if __name__ == "__main__":
    main()
