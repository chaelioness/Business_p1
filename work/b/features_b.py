"""B 파트 피처: daily, price(시간외 포함), earnings, analyst.

    x = build_b(day)            # symbol + B_FEATURES, 종목당 한 줄
    t = build_b_table(days)     # 여러 날을 쌓은 학습용 표 (+ label, ret_pct)

이 파일은 src/model.py 에 그대로 붙여 넣을 수 있게 자급으로 만든다.
import 는 표준 라이브러리, numpy, pandas 만 쓴다.

시간 구간 원칙
- 모든 데이터는 day.daily / day.price / day.earnings / day.analyst 로만 꺼낸다.
  이 함수들은 known_at < day.cutoff(대상일 09:30) 인 행만 돌려준다.
- "기준일 장 마감 이후" 는 기준일 16:00 부터 cutoff 전까지를 말한다.
- 종목 이름·날짜를 코드에 박지 않는다. 크기가 종목마다 다른 값은
  자기 변동성으로 나눈 z-score 로 만든다.
- 시장 평균은 그날 daily 에 보이는 종목들의 동일가중 평균이다.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 상수

B_LOOKBACK = 130            # 일봉 길이(거래일). 60일 변동성·베타에 여유를 둠
B_CLOSE = pd.Timedelta(hours=16)        # 정규장 마감
B_LAST_HOUR = pd.Timedelta(hours=14, minutes=30)
B_EARN_DAYS = 400           # 실적 이력 (달력일)
B_ANALYST_DAYS = 90         # 애널리스트 이력 (달력일)
B_WINDOW = 30               # 애널리스트 최근 창 (달력일)
B_BIG = 0.025               # 급등락 기준 (label 0/4 경계)
B_SURPRISE_CLIP = 100.0     # surprise_pct 극단값 자름 (%)
B_TARGET_CLIP = 1.0         # 목표가 변경률·괴리율 자름

B_DAILY = [
    "ret1", "ret5", "ret20", "ret1_z", "ret5_z", "ret20_z",
    "vol5", "vol20", "vol60", "vol_5_20", "vol_20_60",
    "atr14", "range1", "range1_z", "gap_open1", "intraday1", "volu_ratio",
    "dist_ma20_z", "dist_ma50_z", "rsi14", "bb_pctb", "macd_hist",
    "bigfreq60", "streak", "pos60",
    "mkt_ret1", "mkt_ret5", "mkt_vol20", "rel_ret1_z", "rel_ret5_z", "beta60",
]
B_PRICE = [
    "post_ret", "post_n", "pre_move", "pre_n", "ext_n",
    "gap", "gap_z", "abs_gap_z", "gap_rel_z", "mkt_gap", "mkt_gap_z",
    "ext_range_z", "last_hour_z",
]
B_EARN = ["earn_now", "earn_surprise_now", "earn_prev", "days_since_earn",
          "last_surprise", "surprise_mean4"]
B_ANALYST = ["an_n30", "an_up30", "an_down30", "an_raise30", "an_lower30",
             "an_tgtchg30", "an_tgt_gap", "an_now_n", "an_now_net"]
B_CAL = ["gap_days", "dow"]

B_FEATURES = B_DAILY + B_PRICE + B_EARN + B_ANALYST + B_CAL

# v1: 시간외 갭·변동성·모멘텀·시장 대비 네 그룹에서 학습 구간 신호가 확인된 것만.
# 기준: 방향 |Spearman| >= 0.02 이고 5개 기간 중 4개 이상 부호 유지,
# 또는 크기 |Spearman| >= 0.10 이고 5개 기간 모두 유지. 실적·애널리스트·달력은 뺌.
# 주의: 근거였던 이전 EDA 는 삭제함. 팀 데이터(2026-06-01 이전)로 다시 확인해야 함 (임시 목록).
# mkt_gap_z 는 종목 구성(vol20 중앙값)에 따라 달라지고 mkt_gap 과 겹쳐서 뺌.
B_V1_GAP = ["gap", "gap_z", "abs_gap_z", "gap_rel_z", "pre_move", "post_ret", "mkt_gap"]
B_V1_VOL = ["vol5", "vol20", "vol60", "atr14", "range1", "bigfreq60"]
B_V1_MOM = ["ret1", "ret5", "ret1_z", "ret5_z", "intraday1", "gap_open1",
            "last_hour_z", "rsi14", "dist_ma20_z", "macd_hist"]
B_V1_MKT = ["beta60", "mkt_ret1", "mkt_vol20"]
B_FEATURES_V1 = B_V1_GAP + B_V1_VOL + B_V1_MOM + B_V1_MKT


# ---------------------------------------------------------------- daily

def _b_daily(day):
    """일봉 피처. 기준일 종가까지(known_at 16:00)만 쓴다.

    일봉을 (날짜 x 종목) 넓은 표로 펼쳐 한 번에 계산하고 마지막 행(기준일)만 뽑는다.
    기준일에 거래가 없던 종목은 NaN 이 된다.
    """
    d = day.daily(days=B_LOOKBACK, columns=["symbol", "date_et", "open", "high",
                                            "low", "close", "volume", "ret"])
    if not len(d):
        return pd.DataFrame(columns=["symbol", *B_DAILY]), pd.Series(dtype=float), pd.Series(dtype=float)
    w = {c: d.pivot_table(index="date_et", columns="symbol", values=c, aggfunc="last")
         for c in ["open", "high", "low", "close", "volume", "ret"]}
    C, O, H, L, V, R = (w[k].sort_index() for k in ["close", "open", "high", "low", "volume", "ret"])

    mkt = R.mean(axis=1)                         # 보이는 종목 동일가중
    vol5, vol20, vol60 = (R.rolling(n, min_periods=max(3, n // 2)).std() for n in (5, 20, 60))
    ret5 = C / C.shift(5) - 1
    ret20 = C / C.shift(20) - 1

    tr = np.maximum(H - L, np.maximum((H - C.shift()).abs(), (L - C.shift()).abs()))
    atr14 = tr.rolling(14, min_periods=7).mean() / C
    rng = (H - L) / C
    ma20, ma50 = C.rolling(20, min_periods=10).mean(), C.rolling(50, min_periods=25).mean()

    diff = C.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    sd20 = C.rolling(20, min_periods=10).std()
    pctb = (C - (ma20 - 2 * sd20)) / (4 * sd20)
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    hist = (macd - macd.ewm(span=9, adjust=False).mean()) / C

    sgn = np.sign(R.fillna(0).to_numpy())
    st = np.zeros_like(sgn)
    for i in range(len(sgn)):                    # 연속 상승(+)/하락(-) 일수
        prev = st[i - 1] if i else np.zeros(sgn.shape[1])
        st[i] = np.where((np.sign(prev) == sgn[i]) & (sgn[i] != 0), prev + sgn[i], sgn[i])
    streak = pd.DataFrame(st, index=R.index, columns=R.columns)
    lo60, hi60 = C.rolling(60, min_periods=20).min(), C.rolling(60, min_periods=20).max()

    m_var = mkt.rolling(60, min_periods=30).var()
    beta = R.apply(lambda s: s.rolling(60, min_periods=30).cov(mkt)).div(m_var, axis=0)
    mkt_vol20 = mkt.rolling(20, min_periods=10).std()
    rel1, rel5 = R.sub(mkt, axis=0), ret5.sub(ret5.mean(axis=1), axis=0)

    last = C.index[-1]
    if pd.Timestamp(last) != day.date:           # 기준일 일봉이 없으면 쓰지 않음
        return pd.DataFrame(columns=["symbol", *B_DAILY]), pd.Series(dtype=float), pd.Series(dtype=float)

    def at(x):
        return x.loc[last]

    v20 = at(vol20)
    s5 = np.sqrt(5)
    out = pd.DataFrame({
        "ret1": at(R), "ret5": at(ret5), "ret20": at(ret20),
        "ret1_z": at(R) / v20, "ret5_z": at(ret5) / (v20 * s5), "ret20_z": at(ret20) / (v20 * np.sqrt(20)),
        "vol5": at(vol5), "vol20": v20, "vol60": at(vol60),
        "vol_5_20": at(vol5) / v20, "vol_20_60": v20 / at(vol60),
        "atr14": at(atr14), "range1": at(rng),
        "range1_z": at(rng) / at(rng.rolling(20, min_periods=10).mean()),
        "gap_open1": at(O) / at(C.shift()) - 1, "intraday1": at(C) / at(O) - 1,
        "volu_ratio": np.log(at(V) / at(V.rolling(20, min_periods=10).mean())),
        "dist_ma20_z": (at(C) / at(ma20) - 1) / v20, "dist_ma50_z": (at(C) / at(ma50) - 1) / v20,
        "rsi14": at(rsi), "bb_pctb": at(pctb), "macd_hist": at(hist),
        "bigfreq60": at((R.abs() > B_BIG).astype(float).where(R.notna()).rolling(60, min_periods=20).mean()),
        "streak": at(streak), "pos60": (at(C) - at(lo60)) / (at(hi60) - at(lo60)),
        "mkt_ret1": mkt.loc[last], "mkt_ret5": ret5.loc[last].mean(), "mkt_vol20": mkt_vol20.loc[last],
        "rel_ret1_z": at(rel1) / v20, "rel_ret5_z": at(rel5) / (v20 * s5), "beta60": at(beta),
    })
    out.index.name = "symbol"
    return out.reset_index(), at(C), v20


# ---------------------------------------------------------------- price

def _b_price(day, close, vol20):
    """시간외 피처. 기준일 16:00 이후 post/pre 봉과 기준일 정규장 마지막 봉.

    day.price(hours=20) 은 월요일 대상일에 금요일 post 를 놓치므로 since=기준일로 받는다.
    시간외 봉은 volume 이 0 이라 거래량 대신 봉 개수와 가격 범위를 쓴다.
    """
    cols = ["symbol", *B_PRICE]
    if close.empty:
        return pd.DataFrame(columns=cols)
    p = day.price(since=day.date, columns=["symbol", "datetime", "open", "high", "low",
                                           "close", "session", "known_at"])
    after = day.date + B_CLOSE
    ext = p[(p.datetime >= after) & p.session.isin(["post", "pre"])].sort_values("datetime")
    post = ext[(ext.session == "post") & (ext.datetime.dt.normalize() == day.date)]
    pre = ext[(ext.session == "pre") & (ext.datetime.dt.normalize() == day.target)]
    reg = p[(p.session == "regular") & (p.datetime.dt.normalize() == day.date)
            & (p.datetime >= day.date + B_LAST_HOUR)].sort_values("datetime")

    out = pd.DataFrame(index=close.index)
    g_post, g_pre, g_ext = post.groupby("symbol"), pre.groupby("symbol"), ext.groupby("symbol")
    out["post_ret"] = g_post.close.last() / close - 1
    out["post_n"] = g_post.size().reindex(out.index).fillna(0)
    out["pre_move"] = g_pre.close.last() / g_pre.open.first() - 1
    out["pre_n"] = g_pre.size().reindex(out.index).fillna(0)
    out["ext_n"] = g_ext.size().reindex(out.index).fillna(0)
    out["gap"] = g_ext.close.last() / close - 1
    out["gap_z"] = out.gap / vol20
    out["abs_gap_z"] = out.gap_z.abs()
    mkt_gap = out.gap.mean()
    out["gap_rel_z"] = (out.gap - mkt_gap) / vol20
    out["mkt_gap"] = mkt_gap
    out["mkt_gap_z"] = mkt_gap / vol20.median()
    out["ext_range_z"] = (g_ext.high.max() - g_ext.low.min()) / close / vol20
    g_reg = reg.groupby("symbol")
    out["last_hour_z"] = (g_reg.close.last() / g_reg.open.first() - 1) / vol20
    out.index.name = "symbol"
    return out.reset_index()[cols]


# ---------------------------------------------------------------- earnings

def _b_earnings(day, symbols):
    """실적 피처. '지금' = 기준일 16:00 ~ cutoff 사이 발표(대상일 시가에 반영될 것)."""
    e = day.earnings(since=day.date - pd.Timedelta(days=B_EARN_DAYS))
    out = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    after = day.date + B_CLOSE
    prev_start = day.start_of(days=2)            # 직전 거래일
    prev_start = (prev_start + B_CLOSE) if prev_start is not None else after
    e = e.assign(s=e.surprise_pct.clip(-B_SURPRISE_CLIP, B_SURPRISE_CLIP)).sort_values("known_at")
    now, past = e[e.known_at >= after], e[e.known_at < after]
    out["earn_now"] = now.groupby("symbol").size().reindex(out.index).fillna(0).clip(upper=1)
    out["earn_surprise_now"] = now.groupby("symbol").s.last().reindex(out.index).fillna(0)
    out["earn_prev"] = (past[past.known_at >= prev_start].groupby("symbol").size()
                        .reindex(out.index).fillna(0).clip(upper=1))
    g = past.groupby("symbol")
    last_at = g.known_at.max().reindex(out.index)
    out["days_since_earn"] = (day.target - last_at.dt.normalize()).dt.days
    out["last_surprise"] = g.s.last().reindex(out.index)
    out["surprise_mean4"] = g.s.apply(lambda s: s.tail(4).mean()).reindex(out.index)
    return out.reset_index()


# ---------------------------------------------------------------- analyst

def _b_analyst(day, close):
    """애널리스트 피처. 최근 30일 의견 변경과 목표가, 기준일 장 마감 후 신규 의견.

    target_prior 가 없거나 0 이하이면 목표가 변경률 계산에서 뺀다.
    """
    a = day.analyst(since=day.date - pd.Timedelta(days=B_ANALYST_DAYS))
    out = pd.DataFrame(index=pd.Index(close.index, name="symbol"))
    ta = a.target_action.fillna("").str.lower()
    a = a.assign(
        up=(a.action == "up").astype(int), down=(a.action == "down").astype(int),
        raise_=(ta == "raises").astype(int), lower=(ta == "lowers").astype(int),
        chg=(a.target_current / a.target_prior.where(a.target_prior > 0) - 1)
            .clip(-B_TARGET_CLIP, B_TARGET_CLIP))
    a["net"] = a.up + a.raise_ - a.down - a.lower
    w30 = a[a.known_at >= day.target - pd.Timedelta(days=B_WINDOW)]
    g = w30.groupby("symbol")
    out["an_n30"] = g.size()
    out["an_up30"], out["an_down30"] = g.up.sum(), g.down.sum()
    out["an_raise30"], out["an_lower30"] = g.raise_.sum(), g.lower.sum()
    for c in ["an_n30", "an_up30", "an_down30", "an_raise30", "an_lower30"]:
        out[c] = out[c].fillna(0)
    out["an_tgtchg30"] = g.chg.mean()
    tgt = a[a.target_current > 0].groupby("symbol").target_current.median()
    out["an_tgt_gap"] = (tgt.reindex(out.index) / close - 1).clip(-B_TARGET_CLIP, B_TARGET_CLIP)
    now = a[a.known_at >= day.date + B_CLOSE].groupby("symbol")
    out["an_now_n"] = now.size().reindex(out.index).fillna(0)
    out["an_now_net"] = now.net.sum().reindex(out.index).fillna(0)
    return out.reset_index()


# ---------------------------------------------------------------- 묶기

def build_b(day):
    """하루치 B 피처. symbol + B_FEATURES. 그날 daily 에 보이는 종목마다 한 줄."""
    x, close, vol20 = _b_daily(day)
    if not len(x):
        return pd.DataFrame(columns=["symbol", *B_FEATURES])
    x = x.merge(_b_price(day, close, vol20), on="symbol", how="left")
    x = x.merge(_b_earnings(day, x.symbol), on="symbol", how="left")
    x = x.merge(_b_analyst(day, close), on="symbol", how="left")
    x["gap_days"] = (day.target - day.date).days
    x["dow"] = day.target.dayofweek
    x = x[["symbol", *B_FEATURES]]
    num = x[B_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return pd.concat([x[["symbol"]], num], axis=1)


# ---------------------------------------------------------------- 추가 후보 (2026-10-02 실험용)

B_EXTRA_HIST_DAYS = 900     # 과거 실적 반응을 볼 달력일 (일봉이 2023-10 부터 있음)
B_EXTRA = [
    # 크기
    "earn_hist_absr",   # 이 종목의 이전 실적 발표 다음날 |ret| 평균 (%) — 이번 발표 전 것만
    "earn_timing",      # 밤사이 실적 발표: 0 없음 / 1 개장 전 / 2 장 마감 후
    "absret1_z",        # 기준일 |ret| ÷ vol60
    "big_yday",         # 기준일 |ret| > 2.5%
    "bigfreq20",        # 최근 20일 급등락 비율
    "mkt_absgap",       # 그날 보이는 종목들의 |gap_z| 평균 (밤사이 시장 요동)
    "gap_disp",         # 그날 종목 간 gap_z 표준편차
    # 방향
    "gap_onz",          # gap ÷ 과거 60일 밤사이 수익률(시가/전일 종가) 표준편차
    "gap_rank",         # 그날 보이는 종목 중 gap_z 백분위 (0~1)
    "pre_late",         # 대상일 마지막 pre 봉 2개의 움직임 ÷ vol20
    "ext_pos",          # 시간외 최고·최저 사이 마지막 가격 위치 (0~1)
]


def build_b_extra(day):
    """B_EXTRA 후보. build_b 와 같은 규칙 (day.* 로만 읽음, cutoff 전, 종목 ID 없음)."""
    d = day.daily(since=day.date - pd.Timedelta(days=B_EXTRA_HIST_DAYS),
                  columns=["symbol", "date_et", "open", "close", "ret"])
    if not len(d) or pd.Timestamp(d.date_et.max()) != day.date:
        return pd.DataFrame(columns=["symbol", *B_EXTRA])
    O = d.pivot_table(index="date_et", columns="symbol", values="open", aggfunc="last").sort_index()
    C = d.pivot_table(index="date_et", columns="symbol", values="close", aggfunc="last").sort_index()
    R = d.pivot_table(index="date_et", columns="symbol", values="ret", aggfunc="last").sort_index()
    last = R.index[-1]
    vol20 = R.rolling(20, min_periods=10).std().loc[last]
    vol60 = R.rolling(60, min_periods=30).std().loc[last]
    ovn_sd = (O / C.shift() - 1).rolling(60, min_periods=30).std().loc[last]
    r1 = R.loc[last]
    out = pd.DataFrame(index=pd.Index(R.columns, name="symbol"))
    out["absret1_z"] = r1.abs() / vol60
    out["big_yday"] = (r1.abs() > B_BIG).astype(float).where(r1.notna())
    out["bigfreq20"] = (R.abs() > B_BIG).astype(float).where(R.notna()).rolling(20, min_periods=10).mean().loc[last]

    # 시간외: 기준일 16:00 이후 post/pre 봉 (build_b 와 같은 창)
    p = day.price(since=day.date, columns=["symbol", "datetime", "open", "high", "low", "close", "session"])
    ext = p[(p.datetime >= day.date + B_CLOSE) & p.session.isin(["post", "pre"])].sort_values("datetime")
    g = ext.groupby("symbol")
    close0 = C.loc[last]
    gap = g.close.last() / close0 - 1
    gap_z = gap / vol20
    out["gap_onz"] = gap / ovn_sd
    out["gap_rank"] = gap_z.rank(pct=True)
    out["mkt_absgap"] = gap_z.abs().mean()
    out["gap_disp"] = gap_z.std()
    hi, lo = g.high.max(), g.low.min()
    out["ext_pos"] = ((g.close.last() - lo) / (hi - lo)).where(hi > lo)
    pre = ext[(ext.session == "pre") & (ext.datetime.dt.normalize() == day.target)]
    late = pre.groupby("symbol").tail(2).groupby("symbol")
    out["pre_late"] = (late.close.last() / late.open.first() - 1) / vol20

    # 실적: 이번 발표 시각, 이전 발표들의 다음날 반응 크기
    e = day.earnings(since=day.date - pd.Timedelta(days=B_EXTRA_HIST_DAYS), columns=["symbol", "known_at"])
    after = day.date + B_CLOSE
    now = e[e.known_at >= after]
    hm = now.known_at.dt.hour * 60 + now.known_at.dt.minute
    timing = pd.Series(np.where(hm >= 16 * 60, 2.0, 1.0), index=now.symbol.values).groupby(level=0).max()
    out["earn_timing"] = timing.reindex(out.index).fillna(0.0)
    past = e[e.known_at < after]
    days_idx = R.index.values
    react = []
    for sym, kt in zip(past.symbol, past.known_at):
        d0 = np.datetime64(kt.normalize())
        k = np.searchsorted(days_idx, d0, side="right" if kt.hour * 60 + kt.minute >= 9 * 60 + 30 else "left")
        if k < len(days_idx) and sym in R.columns:
            v = R.iat[k, R.columns.get_loc(sym)]
            if v == v:
                react.append((sym, abs(v) * 100))
    if react:
        rr = pd.DataFrame(react, columns=["symbol", "absr"]).groupby("symbol").absr.agg(["mean", "size"])
        out["earn_hist_absr"] = rr["mean"].where(rr["size"] >= 2).reindex(out.index)
    else:
        out["earn_hist_absr"] = np.nan
    out = out.reset_index()[["symbol", *B_EXTRA]]
    num = out[B_EXTRA].astype(float).replace([np.inf, -np.inf], np.nan)
    return pd.concat([out[["symbol"]], num], axis=1)


def build_b_table(days, with_label=True, verbose=True):
    """여러 날을 쌓는다. with_label 이면 day.y 의 label, ret_pct 를 붙임(학습용)."""
    rows = []
    for i, day in enumerate(days, 1):
        x = build_b(day)
        if not len(x):
            continue
        if with_label:
            x = x.merge(day.y[["symbol", "label", "ret_pct"]], on="symbol")
        x.insert(1, "date", day.date)
        rows.append(x)
        if verbose and i % 20 == 0:
            print(f"  {i}/{len(days)}일", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
