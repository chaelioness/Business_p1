"""5. 이벤트(실적·애널리스트)와 텍스트(뉴스·Reddit).

투자 실무에서 급등락의 대부분은 '알려진 이벤트' 에서 나옴. 볼 것:
  - 실적 발표가 걸린 대상일의 급등락 확률, 서프라이즈 부호와 방향, 갭이 이미 반영하나
  - 애널리스트 의견 변화가 갭 이상의 정보를 주나
  - 뉴스 기사량(특히 장 마감 후)·논조가 크기/방향과 관계 있나 — 갭을 통제한 뒤에도
  - Reddit 활동량이 시장 전체 변동성의 선행 지표인가 (종목 구분이 없는 시황 데이터)
Reddit score / n_comments 는 36시간 뒤 값이라 쓰지 않음. 개수와 시각만 씀.
"""

import re

from sklearn.metrics import roc_auc_score

from eda.common import *  # noqa: F403

S = Section("s05_events_text")


# ---------------------------------------------------------------- 실적

def earnings(d):
    e = d[d.earn]
    S.note("earn_rows", len(e))
    S.note("earn_p_big_vs_normal", [e.label.isin([0, 4]).mean(), d[~d.earn].label.isin([0, 4]).mean()])
    S.note("earn_mean_absret_vs_normal", [e.ret_pct.abs().mean(), d[~d.earn].ret_pct.abs().mean()])
    S.note("earn_share_of_all_big_moves", e.label.isin([0, 4]).sum() / d.label.isin([0, 4]).sum())
    t = pd.DataFrame({"실적일": label_dist(e.label), "일반일": label_dist(d[~d.earn].label)}) * 100
    S.tab(t.round(1), "tab_earn_label_dist")

    s = e.dropna(subset=["earn_surprise"]).copy()
    s["sgn"] = np.sign(s.earn_surprise)
    rows = {}
    for k, g in s.groupby("sgn"):
        rows[{-1: "미달", 0: "부합", 1: "상회"}[k]] = {
            "n": len(g), "ret_mean": g.ret_pct.mean(), "p_up(>0)": (g.ret_pct > 0).mean(),
            "gap_mean": g.ovn_gap.mean(), "intraday_mean": g.y_intraday.mean(),
            "p_big": g.label.isin([0, 4]).mean()}
    S.tab(pd.DataFrame(rows).T, "tab_earn_by_surprise_sign")
    S.note("earn_spearman_surprise_ret", s.earn_surprise.corr(s.ret_pct, method="spearman"))
    S.note("earn_spearman_gap_ret", s.ovn_gap.corr(s.ret_pct, method="spearman"))
    # 갭과 서프라이즈 중 무엇이 더 많이 말하나 (잔차)
    b = np.polyfit(s.ovn_gap, s.ret_pct, 1)
    S.note("earn_spearman_surprise_vs_ret_resid_after_gap",
           s.earn_surprise.corr(s.ret_pct - np.polyval(b, s.ovn_gap), method="spearman"))
    S.tab(pd.crosstab(e.earn_when, e.label, normalize="index").round(3), "tab_earn_timing_label")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(s.earn_surprise.clip(-50, 50), s.ret_pct, s=12, alpha=.6)
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].axvline(0, color="k", lw=.5)
    ax[0].set(title="EPS 서프라이즈 vs 대상일 수익률", xlabel="surprise % (±50 클립)", ylabel="ret %")
    ax[1].scatter(s.ovn_gap, s.ret_pct, s=12, alpha=.6, c="#55a868")
    ax[1].plot([-20, 20], [-20, 20], "k--", lw=.7)
    for c in RET_CUTS:
        ax[1].axhline(c, color="r", lw=.5, ls=":")
    ax[1].set(title="실적일: 시간외 갭 vs 수익률 — 갭이 반응의 대부분을 선반영", xlabel="ovn_gap %", ylabel="ret %")
    S.fig(fig, "fig_earnings")

    # 실적 '예정' 정보: 직전 실적 이후 경과일 → 다음 발표 근접 (분기 ~63거래일 주기)
    dd = d.sort_values(["symbol", "target"]).copy()
    dd["last_earn"] = dd.target.where(dd.earn)
    dd["last_earn"] = dd.groupby("symbol").last_earn.ffill()
    dd["days_since_earn"] = (dd.target - dd.last_earn).dt.days
    # 발표 당일은 0 이라 '그 이전 발표 기준' 으로 한 행 밀어서 봄 (cutoff 기준으로 알 수 있는 값)
    dd["days_since_prev"] = dd.groupby("symbol").days_since_earn.shift(1) + (
        dd.target - dd.groupby("symbol").target.shift(1)).dt.days
    x = dd.dropna(subset=["days_since_prev"])
    x = x.assign(bin=pd.cut(x.days_since_prev, [0, 20, 40, 60, 80, 85, 90, 95, 100, 200]))
    t = x.groupby("bin", observed=True).agg(n=("earn", "size"), p_earn=("earn", "mean"),
                                            p_big=("label", lambda s: s.isin([0, 4]).mean()))
    S.tab(t, "tab_days_since_prev_earnings")


# ---------------------------------------------------------------- 애널리스트

def analyst(d):
    d = d.assign(an_net=d.an_up - d.an_down, an_tgt_net=d.an_raise - d.an_lower)
    rows = {}
    for name, m in [("상향(up)", d.an_up > 0), ("하향(down)", d.an_down > 0), ("신규(init)", d.an_init > 0),
                    ("목표가↑ 우세", d.an_tgt_net > 0), ("목표가↓ 우세", d.an_tgt_net < 0), ("이벤트 없음", d.an_n == 0)]:
        g = d[m]
        rows[name] = {"n": len(g), "ret_mean": g.ret_pct.mean(), "gap_mean": g.ovn_gap.mean(),
                      "intraday_mean": g.y_intraday.mean(), "p_big": g.label.isin([0, 4]).mean(),
                      "p_up": (g.ret_pct > 0).mean()}
    S.tab(pd.DataFrame(rows).T, "tab_analyst_effect")
    # 실적 다음날 목표가 조정 러시는 실적 반응과 섞임 → 실적 제외 버전
    ne = d[~d.earn]
    S.note("analyst_up_minus_down_intraday_excl_earn",
           ne[ne.an_up > 0].y_intraday.mean() - ne[ne.an_down > 0].y_intraday.mean())
    S.note("analyst_events_on_earn_days_share", d[d.earn].an_n.sum() / d.an_n.sum())


# ---------------------------------------------------------------- 뉴스

def news(d):
    d = d.assign(absret=d.ret_pct.abs(), big=d.label.isin([0, 4]),
                 log_news=np.log1p(d.news_n), log_news_ovn=np.log1p(d.news_n_ovn), abs_gap=d.ovn_gap.abs())
    g = d.dropna(subset=["ovn_gap"])
    b = np.polyfit(g.abs_gap.clip(0, 10), g.absret, 1)
    d["absret_resid"] = d.absret - np.polyval(b, d.abs_gap.clip(0, 10).fillna(0))
    b2 = np.polyfit(g.ovn_gap.clip(-10, 10), g.ret_pct, 1)
    d["ret_resid"] = d.ret_pct - np.polyval(b2, d.ovn_gap.clip(-10, 10).fillna(0))
    rows = []
    for c in ["log_news", "log_news_ovn", "news_abn", "news_tone", "news_tone_ovn", "news_neg", "news_pos"]:
        r = {"feature": c}
        for tgt in ["ret_pct", "ret_resid", "absret", "absret_resid"]:
            ic, _ = daily_ic(d, c, tgt)
            r[f"IC_{tgt}"], r[f"t_{tgt}"] = ic["ic_mean"], ic["t"]
        x = d[[c, "big"]].dropna()
        r["AUC_big"] = roc_auc_score(x.big, x[c])
        r["coverage"] = d[c].notna().mean()
        rows.append(r)
    S.tab(pd.DataFrame(rows).set_index("feature"), "tab_news_ic")

    t = bucket_table(d.dropna(subset=["news_abn"]), "news_abn", 10)
    S.tab(t, "tab_bucket_news_abn")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(t.x_mean, t.p_big * 100, "o-", label="급등락 %")
    ax[0].plot(t.x_mean, t.absret_mean * 10, "s--", label="|ret| ×10")
    ax[0].set(title="비정상 기사량(log, 종목 20일 평균 대비) 10분위", xlabel="news_abn")
    ax[0].legend()
    q = d.dropna(subset=["news_tone_ovn"]).copy()
    q = q[q.news_n_ovn >= 5]
    t2 = bucket_table(q, "news_tone_ovn", 10)
    ax[1].plot(t2.x_mean, t2.ret_mean, "o-")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set(title="장 마감 후 기사 논조 10분위 → 평균 ret (기사≥5건)", xlabel="평균 tone", ylabel="ret %")
    S.fig(fig, "fig_news")
    S.tab(t2, "tab_bucket_news_tone_ovn")


# ---------------------------------------------------------------- Reddit

TICKER_SKIP = {"V", "GE", "MS", "KO", "PM", "DE", "CB", "GS", "NOW", "LIN", "COST", "DELL", "UBER"}


def reddit(d):
    """시장 레벨: 장 마감~개장 전 댓글 수 (종목 평균 |ret| 와 비교). 종목 레벨: 글 제목 티커 언급."""
    dates = pd.DatetimeIndex(np.sort(d.target.unique()))
    from eda.build_panel import map_to_target
    all_dates = pd.DatetimeIndex(np.sort(read("daily", ["date_et"]).date_et.unique()))

    day = d.groupby("target").agg(mkt_abs=("y_mkt", lambda s: abs(s.iloc[0])),
                                  mean_abs=("ret_pct", lambda s: s.abs().mean()),
                                  n_big=("label", lambda s: s.isin([0, 4]).sum()),
                                  mkt_gap_abs=("mkt_gap", lambda s: abs(s.iloc[0])),
                                  mkt_r1_abs=("mkt_r1", lambda s: abs(s.iloc[0])))
    for sub in subreddits():
        r = read_reddit(sub, "comments", ["created_et"])
        r = r[r.created_et >= EDA_FIRST_TARGET - pd.Timedelta(days=60)]
        T, B, ovn = map_to_target(r.created_et.astype("datetime64[us]"), all_dates)
        cnt = pd.Series(1, index=T[ovn.values].values).groupby(level=0).sum()
        day[f"rd_{sub}"] = np.log1p(cnt.reindex(day.index).fillna(0))
    rd = [c for c in day.columns if c.startswith("rd_")]
    # 수준 대신 직전 20일 대비 변화 (추세 제거)
    for c in rd:
        day[c] = day[c] - day[c].shift(1).rolling(20, min_periods=10).mean()
    day["rd_all"] = day[rd].mean(axis=1)
    rows = {}
    for c in rd + ["rd_all"]:
        x = day[[c, "mean_abs", "n_big", "mkt_abs"]].dropna()
        # 전일 |시장| 과 |시장 갭| 을 통제한 부분상관 (이미 아는 정보 이상인가)
        ctrl = day.loc[x.index, ["mkt_r1_abs", "mkt_gap_abs"]].fillna(0).values
        A = np.c_[np.ones(len(x)), ctrl]
        res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        rows[c] = {"spearman_mean_abs": x[c].corr(x.mean_abs, method="spearman"),
                   "spearman_n_big": x[c].corr(x.n_big, method="spearman"),
                   "spearman_|mkt|": x[c].corr(x.mkt_abs, method="spearman"),
                   "partial_mean_abs|ctrl": np.corrcoef(res(x[c].values), res(x.mean_abs.values))[0, 1],
                   "n_days": len(x)}
    S.tab(pd.DataFrame(rows).T, "tab_reddit_market_activity")

    # 종목 언급 (글 제목, 대문자 또는 $티커). 흔한 단어와 겹치는 티커는 $ 만 인정
    syms = sorted(d.symbol.unique())
    pat = re.compile(r"(?<![A-Za-z$])\$?(" + "|".join(re.escape(s) for s in syms) + r")(?![A-Za-z])")
    rows = []
    for sub in subreddits():
        p = read_reddit(sub, "posts", ["created_et", "title"])
        p = p[p.created_et >= EDA_FIRST_TARGET - pd.Timedelta(days=60)]
        for t, title in zip(p.created_et, p.title.fillna("")):
            for m in pat.finditer(title):
                s = m.group(1)
                if s in TICKER_SKIP and not m.group(0).startswith("$"):
                    continue
                rows.append((t, s))
    m = pd.DataFrame(rows, columns=["created_et", "symbol"])
    T, B, ovn = map_to_target(m.created_et.astype("datetime64[us]"), all_dates)
    m["target"] = T.values
    cnt = m.groupby(["symbol", "target"]).size().rename("rd_mentions").reset_index()
    x = d.merge(cnt, on=["symbol", "target"], how="left").fillna({"rd_mentions": 0})
    x = x.sort_values(["symbol", "target"])
    x["rd_mention_abn"] = np.log1p(x.rd_mentions) - np.log1p(
        x.groupby("symbol").rd_mentions.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()))
    x["absret"] = x.ret_pct.abs()
    S.note("reddit_mentions_total", int(m.shape[0]))
    S.tab(m.symbol.value_counts().rename("mentions").to_frame(), "tab_reddit_mentions_by_symbol")
    out = {}
    for c in ["rd_mentions", "rd_mention_abn"]:
        out[c] = {"IC_ret": daily_ic(x, c)[0]["ic_mean"], "IC_absret": daily_ic(x, c, "absret")[0]["ic_mean"],
                  "IC_absret_t": daily_ic(x, c, "absret")[0]["t"]}
    S.tab(pd.DataFrame(out).T, "tab_reddit_mentions_ic")


def main():
    print("[s05] 이벤트·텍스트")
    d = panel()
    earnings(d)
    analyst(d)
    news(d)
    reddit(d)
    S.save()


if __name__ == "__main__":
    main()
