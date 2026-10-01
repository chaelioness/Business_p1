"""4. 일봉 기반 기술적·횡단면 피처 — 방향 vs 크기, 그리고 갭을 넘어서는 추가 정보.

각 피처를 세 가지 타깃에 대해 봄.
  ret_pct      대상일 수익률 (방향)
  y_intraday   시가→종가 (갭이 이미 반영한 부분을 뺀 나머지. 갭 피처와 겹치지 않는 정보)
  |ret_pct|    크기 (급등락 여부)
일별 횡단면 Spearman IC 의 평균과 t값(Fama-MacBeth 식)으로 봄. 풀링 상관은 시장 전체가
움직이는 날에 끌려가서 과대평가되기 쉬움.
"""

from sklearn.metrics import roc_auc_score

from eda.common import *  # noqa: F403

S = Section("s04_technical")

FEATS = {
    "r1": "전일 수익률", "z1": "전일 수익률/σ20", "rel_r1": "전일 시장 대비 수익률",
    "gap1": "전일 시가 갭", "intra1": "전일 장중 수익률",
    "r5": "5일 수익률", "rel_r5": "5일 시장 대비", "r20": "20일 수익률", "rel_r20": "20일 시장 대비",
    "r60": "60일 수익률", "rsi14": "RSI14", "dist_ma20": "MA20 괴리", "dist_ma50": "MA50 괴리",
    "dist_hi20": "20일 고점 대비", "streak": "연속 상승/하락 일수",
    "vol5": "5일 σ", "vol20": "20일 σ", "vol60": "60일 σ", "atr14": "ATR14 %", "range1": "전일 고저폭",
    "vol_ratio": "σ5/σ60", "volu_ratio": "전일 거래량/20일 평균", "dvol20": "20일 평균 거래대금",
    "mkt_r1": "전일 시장 수익률", "ovn_gap": "(참고) 시간외 갭",
}


def ic_table(d):
    d = d.assign(absret=d.ret_pct.abs(), big=d.label.isin([0, 4]))
    # 갭으로 설명되는 부분을 뺀 잔차: 갭 외의 추가 정보인지 보려는 것
    g = d.dropna(subset=["ovn_gap"])
    beta = np.polyfit(g.ovn_gap.clip(-10, 10), g.ret_pct, 1)
    d["ret_resid"] = d.ret_pct - np.polyval(beta, d.ovn_gap.clip(-10, 10).fillna(0))
    rows = []
    for c, desc in FEATS.items():
        r = {"feature": c, "설명": desc}
        for tgt, name in [("ret_pct", "IC_ret"), ("y_intraday", "IC_intraday"),
                          ("ret_resid", "IC_resid_after_gap"), ("absret", "IC_absret")]:
            ic, _ = daily_ic(d, c, tgt)
            r[name], r[name + "_t"] = ic["ic_mean"], ic["t"]
        x = d[[c, "big"]].dropna()
        r["AUC_big"] = roc_auc_score(x.big, x[c])
        r["AUC_big(sym)"] = max(r["AUC_big"], 1 - r["AUC_big"])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("feature")
    S.tab(t, "tab_feature_ic")

    # 반기별 IC 안정성 (방향 / 크기)
    for tgt, name in [("ret_pct", "ret"), ("absret", "absret")]:
        st = pd.DataFrame({h: {c: daily_ic(x, c, tgt)[0]["ic_mean"] for c in FEATS}
                           for h, x in d.groupby("half")})
        S.tab(st, f"tab_feature_ic_by_half_{name}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    t2 = t.drop(index="ovn_gap").sort_values("IC_ret")
    y = np.arange(len(t2))
    ax[0].barh(y - .2, t2.IC_ret, .4, label="ret (방향)")
    ax[0].barh(y + .2, t2.IC_intraday, .4, label="장중 (갭 이후)")
    ax[0].set_yticks(y, [f"{i} {FEATS[i]}" for i in t2.index], fontsize=7)
    ax[0].axvline(0, color="k", lw=.6)
    ax[0].set(title="방향 IC (일별 횡단면 평균) — 대부분 |IC|<0.03, 반전 성향", xlabel="IC")
    ax[0].legend()
    t3 = t.drop(index="ovn_gap").sort_values("IC_absret")
    ax[1].barh(np.arange(len(t3)), t3.IC_absret, color="#c44e52")
    ax[1].set_yticks(np.arange(len(t3)), [f"{i} {FEATS[i]}" for i in t3.index], fontsize=7)
    ax[1].set(title="크기 IC (|ret|) — 변동성 계열이 0.2~0.3", xlabel="IC")
    S.fig(fig, "fig_feature_ic")
    return d


def regime_reversal(d):
    """단기 반전이 고변동 국면에서 더 강한가 (유동성 공급 프리미엄)."""
    day = d.groupby("target").ret_pct.std()
    hi = day > day.median()
    d = d.assign(hi_disp=d.target.map(hi))
    rows = {}
    for c in ["r1", "rel_r1", "r5", "rel_r5", "dist_ma20"]:
        rows[c] = {
            "IC_ret_전체": daily_ic(d, c)[0]["ic_mean"],
            "IC_ret_전일분산高": daily_ic(d[d.hi_disp], c)[0]["ic_mean"],
            "IC_ret_전일분산低": daily_ic(d[~d.hi_disp], c)[0]["ic_mean"]}
    # (주의) hi_disp 는 대상일 당일 분산이라 조건부 분석용. 피처로 쓰려면 전일 값을 써야 함.
    S.tab(pd.DataFrame(rows).T, "tab_reversal_by_dispersion_regime_ex_post")


def buckets(d):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for a, c in zip(ax.flat, ["r1", "r5", "dist_ma20", "atr14", "volu_ratio", "rsi14"]):
        t = bucket_table(d, c, 10)
        a.plot(t.x_mean, t.p_crash * 100, "o-", color=LABEL_COLORS[0], label="급하락")
        a.plot(t.x_mean, t.p_surge * 100, "o-", color=LABEL_COLORS[4], label="급상승")
        a2 = a.twinx()
        a2.bar(t.x_mean, t.ret_mean, width=(t.x_mean.max() - t.x_mean.min()) / 25, alpha=.25, color="grey")
        a2.axhline(0, color="grey", lw=.5)
        a2.grid(False)
        a.set(title=f"{c} ({FEATS[c]}) 10분위", ylabel="확률 %")
        a2.set_ylabel("평균 ret % (막대)", fontsize=7)
        if c == "r1":
            a.legend(fontsize=7)
        S.tab(t, f"tab_bucket_{c}")
    S.fig(fig, "fig_feature_buckets")


def main():
    print("[s04] 기술적 피처")
    d = panel()
    d = ic_table(d)
    regime_reversal(d)
    buckets(d)
    S.save()


if __name__ == "__main__":
    main()
