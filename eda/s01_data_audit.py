"""1. 데이터 감사: 커버리지, 결측, 시각 정합성, 이상치.

투자 데이터에서 모델보다 먼저 볼 것: 가격이 맞는지(일봉 vs 시간봉), 빠진 날, 분할·이상
수익률, 그리고 각 표의 정보가 '언제' 들어오는지(known_at 분포).
"""

from eda.common import *  # noqa: F403

S = Section("s01_data_audit")


def coverage():
    rows = []
    for name, cols in [("daily", ["symbol"]), ("price", ["symbol"]), ("earnings", ["symbol"]),
                       ("analyst", ["symbol"]), ("news", ["symbols"])]:
        d = read(name, cols)
        rows.append({"table": name, "rows": len(d), "first": d.known_at.min(), "last": d.known_at.max(),
                     "rows_in_eda_window": int((d.known_at >= EDA_FIRST_TARGET - pd.Timedelta(days=1)).sum())})
    for sub in subreddits():
        for kind in ["posts", "comments"]:
            d = read_reddit(sub, kind, ["created_et"])
            rows.append({"table": f"reddit/{sub}.{kind}", "rows": len(d), "first": d.created_et.min(),
                         "last": d.created_et.max(),
                         "rows_in_eda_window": int((d.created_et >= EDA_FIRST_TARGET).sum())})
    S.tab(pd.DataFrame(rows), "tab_coverage", index=False)


def daily_checks():
    d = read("daily")
    d = d[(d.date_et >= EDA_FIRST_TARGET)]
    per = d.groupby("symbol").size()
    S.note("daily_days_per_symbol_min_max", [int(per.min()), int(per.max())])
    S.note("daily_missing_ret", int(d.ret.isna().sum()))
    # ret 이 prev_close→close 와 맞는지
    calc = d.close / d.prev_close - 1
    S.note("daily_ret_mismatch_gt_1bp", int(((calc - d.ret).abs() > 1e-4).sum()))
    # 이상 수익률 (분할·오류 후보)
    ext = d.loc[d.ret.abs() > 0.12, ["symbol", "date_et", "prev_close", "open", "close", "ret", "volume"]]
    S.tab(ext.sort_values("ret"), "tab_extreme_returns", index=False)
    S.note("n_abs_ret_gt_12pct", len(ext))

    # 시간봉 마지막 정규장 봉 종가 == 일봉 종가 ?
    p = read("price", ["symbol", "datetime", "close", "session"])
    p = p[(p.session == "regular") & (p.datetime >= EDA_FIRST_TARGET)]
    last = p.sort_values("datetime").groupby(["symbol", p.datetime.dt.normalize()]).close.last()
    last.index.names = ["symbol", "date_et"]
    m = d.set_index(["symbol", "date_et"]).close.to_frame().join(last.rename("hourly_close"), how="inner")
    diff = (m.hourly_close / m.close - 1).abs()
    S.note("hourly_vs_daily_close_median_absdiff_bp", round(float(diff.median() * 1e4), 2))
    S.note("hourly_vs_daily_close_share_gt_10bp", round(float((diff > 1e-3).mean()), 4))


def price_bars():
    p = read("price", ["symbol", "datetime", "close", "volume", "session"])
    p = p[p.datetime >= EDA_FIRST_TARGET]
    p["hm"] = p.datetime.dt.strftime("%H:%M")
    t = p.groupby(["session", "hm"]).agg(n=("close", "size"), vol_pos=("volume", lambda v: (v > 0).mean()))
    p = p.sort_values(["symbol", "datetime"])
    p["moved"] = p.groupby("symbol").close.diff().ne(0)
    t["price_changed"] = p.groupby(["session", "hm"]).moved.mean()
    t["usable_at_cutoff"] = [
        "yes" if s == "post" or (s == "pre" and h <= "08:00") else ("NO (known 09:30)" if s == "pre" else "NO")
        for s, h in t.index]
    S.tab(t, "tab_hourly_bar_layout")
    S.note("ext_session_volume_recorded_share", round(float((p[p.session != "regular"].volume > 0).mean()), 5))


def info_timing():
    """정보가 언제 들어오나 (ET 시각). 장 마감 후~개장 전 정보가 예측에 쓰이는 몫."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
    n = read("news", ["symbols"])
    n = n[n.known_at >= EDA_FIRST_TARGET]
    h = n.known_at.dt.hour.value_counts(normalize=True).sort_index()
    ax[0].bar(h.index, h.values, color="#4c72b0")
    ax[0].set(title=f"뉴스 known_at 시각 분포 (n={len(n):,})", xlabel="ET 시", ylabel="비율")
    S.note("news_known_at_minute_values_top", n.known_at.dt.strftime("%M").value_counts().head(5).to_dict())
    S.note("news_share_after_close_or_premarket",
           round(float(((n.known_at.dt.hour >= 16) | (n.known_at.dt.hour < 9)).mean()), 3))

    a = read("analyst", ["symbol"])
    a = a[a.known_at >= EDA_FIRST_TARGET]
    h = a.known_at.dt.hour.value_counts(normalize=True).sort_index()
    ax[1].bar(h.index, h.values, color="#55a868")
    ax[1].set(title=f"애널리스트 의견 시각 (n={len(a):,})", xlabel="ET 시")

    e = read("earnings", ["symbol"])
    e = e[e.known_at >= EDA_FIRST_TARGET - pd.Timedelta(days=1)]
    h = e.known_at.dt.strftime("%H:%M").value_counts().sort_index()
    ax[2].bar(h.index, h.values, color="#c44e52")
    ax[2].set(title=f"실적 발표 시각 (n={len(e)})", xlabel="ET")
    ax[2].tick_params(axis="x", rotation=60)
    S.fig(fig, "fig_info_timing")

    # 뉴스·Reddit 일별 양 (데이터 수집 결함·급변 확인)
    daily_news = n.groupby(n.known_at.dt.normalize()).size()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(daily_news.index, daily_news.values, lw=.7)
    ax[0].plot(daily_news.rolling(20).median(), color="k", lw=1.2)
    ax[0].set(title="뉴스 일별 기사 수 (검정=20일 중앙값)", yscale="log")
    for sub in subreddits():
        r = read_reddit(sub, "comments", ["created_et"])
        r = r[r.created_et >= EDA_FIRST_TARGET]
        s = r.groupby(r.created_et.dt.to_period("W").dt.start_time).size()
        ax[1].plot(s.index, s.values, lw=1, label=sub)
    ax[1].set(title="Reddit 주별 댓글 수", yscale="log")
    ax[1].legend(fontsize=7, ncol=4)
    S.fig(fig, "fig_news_reddit_volume_over_time")
    S.note("news_daily_count_median", float(daily_news.median()))


def per_symbol():
    d = panel()
    t = d.groupby("symbol").agg(n=("ret_pct", "size"), vol_daily=("ret_pct", "std"),
                                news_per_day=("news_n", "mean"), analyst_events=("an_n", "sum"),
                                earnings=("earn", "sum"), price=("close_B", "last"),
                                dollar_vol_mn=("dvol20", lambda s: s.mean() / 1e6),
                                unseen=("unseen", "first")).sort_values("vol_daily")
    S.tab(t, "tab_per_symbol_coverage")


def main():
    print("[s01] 데이터 감사")
    coverage()
    daily_checks()
    price_bars()
    info_timing()
    per_symbol()
    S.save()


if __name__ == "__main__":
    main()
