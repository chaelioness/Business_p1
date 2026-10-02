"""원본에서 새로 만드는 피처 (r09, r11 이 cache 에 표로 만들어 둠). guard 아래에서만 돌림.

    build_more(day)  → NEW_C   (cache/b_v3.parquet)   시간외 세부·전일 범위·밤사이 누적
    past_feats(table)→ NEW_B                          지난 결과(정답은 대상일 전 것만)로 만든 피처
    build_raw(day)   → NEW     (cache/b_v4.parquet)   전날 장중 흐름·거래량, 애널리스트·실적 세부
"""

import numpy as np
import pandas as pd

import common  # noqa: F401  (경로 설정)
from features_b import B_CLOSE

NEW_C = ["brk_z", "rng_pos", "gap_first_z", "gap_drift_z", "ovn5_z", "ovn20_z", "intra20_z"]
NEW_B = ["peer_gz", "idio_gz", "rel_mkt20", "follow_mkt20", "follow_sym60", "hit_sym60"]
NEW = ["am_ret_z", "pm_ret_z", "pm_vol_rel", "buy_press", "vwap_dev_z", "close_loc",
       "an_init30", "an_reit30", "an_firms30", "an_init_now", "an_now_tgtchg", "an_init_tgt_gap",
       "beat_streak", "eps_yoy"]
NOON = pd.Timedelta(hours=12, minutes=30)


def build_more(day):
    d = day.daily(days=90, columns=["symbol", "date_et", "open", "high", "low", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=["symbol", *NEW_C])
    w = {c: d.pivot_table(index="date_et", columns="symbol", values=c, aggfunc="last").sort_index()
         for c in ["open", "high", "low", "close", "ret"]}
    O, H, L, C, R = w["open"], w["high"], w["low"], w["close"], w["ret"]
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    ovn = O / C.shift() - 1
    intra = C / O - 1
    out = pd.DataFrame(index=pd.Index(R.columns, name="symbol"))
    out["ovn5_z"] = ovn.rolling(5, min_periods=3).sum().loc[last] / vol20
    out["ovn20_z"] = ovn.rolling(20, min_periods=10).sum().loc[last] / vol20
    out["intra20_z"] = intra.rolling(20, min_periods=10).sum().loc[last] / vol20

    p = day.price(since=day.date, columns=["symbol", "datetime", "close", "session"])
    ext = p[(p.datetime >= day.date + B_CLOSE) & p.session.isin(["post", "pre"])].sort_values("datetime")
    g = ext.groupby("symbol")
    c0, h0, l0 = C.loc[last], H.loc[last], L.loc[last]
    lp, fp = g.close.last().reindex(out.index), g.close.first().reindex(out.index)
    out["brk_z"] = np.where(lp > h0, (lp - h0) / c0, np.where(lp < l0, (lp - l0) / c0, 0.0)) / vol20
    out.loc[lp.isna(), "brk_z"] = np.nan
    out["rng_pos"] = ((lp - l0) / (h0 - l0)).where(h0 > l0)
    out["gap_first_z"] = (fp / c0 - 1) / vol20
    out["gap_drift_z"] = (lp / c0 - 1) / vol20 - out.gap_first_z
    return out.reset_index()[["symbol", *NEW_C]]


def past_feats(t):
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = np.sort(t.date.unique())
    gz = t.pivot(index="date", columns="symbol", values="gap_z").reindex(dates)
    r1 = t.pivot(index="date", columns="symbol", values="ret1").reindex(dates)
    zr = (t.ret_pct / 100 / t.vol20).clip(-6, 6)
    zr = t.assign(zr=zr).pivot(index="date", columns="symbol", values="zr").reindex(dates)
    ret = t.pivot(index="date", columns="symbol", values="ret_pct").reindex(dates)

    # peer: 과거 60행 ret1 상관 상위 5 (ret1 은 기준일 수익률 = 대상일 전에 앎)
    peer = pd.DataFrame(np.nan, index=dates, columns=gz.columns)
    for i in range(30, len(dates)):
        c = r1.iloc[max(0, i - 59):i + 1].corr(min_periods=30)
        cv = c.values.copy()
        np.fill_diagonal(cv, np.nan)
        c = pd.DataFrame(cv, index=c.index, columns=c.columns)
        g = gz.iloc[i]
        for s in gz.columns:
            if g[s] == g[s]:
                top = c[s].dropna().nlargest(5).index
                v = g[top].dropna()
                if len(v):
                    peer.iat[i, gz.columns.get_loc(s)] = v.mean()

    # 시장: 날짜별 순위상관, 기울기 → 대상일 전 20일 평균 (shift 1)
    ic = pd.Series([gz.loc[d].corr(ret.loc[d], method="spearman") for d in dates], index=dates)
    num = (gz * zr).sum(1)
    den = (gz ** 2).sum(1)
    rel_mkt20 = ic.shift(1).rolling(20, min_periods=10).mean()
    follow_mkt20 = num.shift(1).rolling(20, min_periods=10).sum() / den.shift(1).rolling(20, min_periods=10).sum()

    # 종목: 지난 60번 (shift 1)
    gzs, zrs = gz.shift(1), zr.shift(1)
    follow_sym = (gzs * zrs).rolling(60, min_periods=30).sum() / (gzs ** 2).rolling(60, min_periods=30).sum()
    m = gzs.abs() > 0.3
    hit = (np.sign(gzs) == np.sign(zrs)).where(m & zrs.notna()).astype(float)
    hit = hit.where(m & zrs.notna())
    hit_sym = hit.rolling(60, min_periods=15).mean()

    def long(df, name):
        return df.stack(future_stack=True).rename(name)

    out = pd.concat([long(peer, "peer_gz"), long(follow_sym, "follow_sym60"), long(hit_sym, "hit_sym60")], axis=1)
    out = out.reset_index().rename(columns={"level_0": "date"})
    out.columns = ["date", "symbol", "peer_gz", "follow_sym60", "hit_sym60"]
    t = t.merge(out, on=["date", "symbol"], how="left")
    t["idio_gz"] = t.gap_z - t.peer_gz
    t["rel_mkt20"] = t.date.map(rel_mkt20)
    t["follow_mkt20"] = t.date.map(follow_mkt20)
    return t


def build_raw(day):
    d = day.daily(days=40, columns=["symbol", "date_et", "high", "low", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=["symbol", *NEW])
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    t0 = d[d.date_et == last].set_index("symbol")
    out = pd.DataFrame(index=pd.Index(R.columns, name="symbol"))
    out["close_loc"] = ((t0.close - t0.low) / (t0.high - t0.low)).where(t0.high > t0.low)

    # 정규장 시간봉 (지난 30일: 오후 거래량 비중의 평소 값)
    p = day.price(since=day.date - pd.Timedelta(days=30),
                  columns=["symbol", "datetime", "open", "close", "volume", "session"])
    p = p[p.session == "regular"].copy()
    p["day"] = p.datetime.dt.normalize()
    p["tod"] = p.datetime - p.day
    p["late"] = p.tod >= pd.Timedelta(hours=14)
    share = p.groupby(["symbol", "day"]).apply(
        lambda g: g.loc[g.late, "volume"].sum() / g.volume.sum() if g.volume.sum() > 0 else np.nan,
        include_groups=False).unstack("symbol")
    if day.date in share.index:
        out["pm_vol_rel"] = share.loc[day.date] - share[share.index < day.date].tail(20).mean()
    t = p[p.day == day.date].sort_values("datetime")
    if len(t):
        g = t.groupby("symbol")
        am, pm = t[t.tod < NOON].groupby("symbol"), t[t.tod >= NOON].groupby("symbol")
        out["am_ret_z"] = (am.close.last() / am.open.first() - 1) / vol20
        out["pm_ret_z"] = (pm.close.last() / pm.open.first() - 1) / vol20
        sv = (np.sign(t.close - t.open) * t.volume).groupby(t.symbol).sum()
        out["buy_press"] = sv / g.volume.sum().replace(0, np.nan)
        vwap = (t.close * t.volume).groupby(t.symbol).sum() / g.volume.sum().replace(0, np.nan)
        cl = g.close.last()
        out["vwap_dev_z"] = (cl - vwap) / cl / vol20

    # 애널리스트
    a = day.analyst(since=day.date - pd.Timedelta(days=90))
    close = t0.close
    w30 = a[a.known_at >= day.target - pd.Timedelta(days=30)]
    g30 = w30.groupby("symbol")
    out["an_init30"] = g30.action.apply(lambda s: (s == "init").sum()).reindex(out.index).fillna(0)
    out["an_reit30"] = g30.action.apply(lambda s: (s == "reit").sum()).reindex(out.index).fillna(0)
    out["an_firms30"] = g30.firm.nunique().reindex(out.index).fillna(0)
    now = a[a.known_at >= day.date + B_CLOSE]
    out["an_init_now"] = now[now.action == "init"].groupby("symbol").size().reindex(out.index).fillna(0)
    chg = (now.target_current / now.target_prior.where(now.target_prior > 0) - 1).clip(-1, 1)
    out["an_now_tgtchg"] = chg.groupby(now.symbol).mean().reindex(out.index).fillna(0)
    ini = a[(a.action == "init") & (a.target_current > 0)].groupby("symbol").target_current.median()
    out["an_init_tgt_gap"] = (ini.reindex(out.index) / close.reindex(out.index) - 1).clip(-1, 1)

    # 실적
    e = day.earnings(since=day.date - pd.Timedelta(days=900))
    e = e[e.known_at < day.date + B_CLOSE].sort_values("known_at")
    bs, yy = {}, {}
    for s, g in e.groupby("symbol"):
        beat = (g.eps_reported > g.eps_estimate).values[::-1]
        bs[s] = int(np.argmin(beat)) if not beat.all() else len(beat)
        r = g.eps_reported.dropna().values
        yy[s] = np.sign(r[-1] - r[-5]) if len(r) >= 5 else np.nan
    out["beat_streak"] = pd.Series(bs).reindex(out.index)
    out["eps_yoy"] = pd.Series(yy).reindex(out.index)
    return out.reset_index()[["symbol", *NEW]]
