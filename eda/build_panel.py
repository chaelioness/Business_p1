"""(종목, 대상일 T) 패널 생성 → cache/eda_panel.parquet

한 행 = 한 종목의 한 예측 문제. B = T 의 직전 거래일(기준일).
피처는 전부 T 09:30(cutoff) 이전에 알 수 있는 값만 씀. 결과(y_*)는 T 16:00 확정값.

    일봉 피처      B 까지의 일봉 (known_at = B 16:00)
    시간외 피처    B 의 post 봉(16~20시) + T 의 pre 봉 중 known_at < T 09:30
                   → T 09:00 pre 봉은 known_at 이 09:30 이라 제외됨 (마지막은 08:00 봉)
    이벤트         known_at 이 [B 09:30, T 09:30) 에 들어온 실적·애널리스트·뉴스
                   (B 16:00 이후 = overnight, 그 전 = B 장중)

EDA 전용이라 벡터화로 한꺼번에 계산함. 모델용 피처는 반드시 build(day) 로 다시 만들고
check_no_leak 을 통과시킬 것.
"""

import numpy as np
import pandas as pd

from eda.common import (CACHE, EDA_FIRST_TARGET, EDA_LAST_TARGET, UNSEEN, label_of, read)

CUT = pd.Timedelta(hours=9, minutes=30)
CLOSE = pd.Timedelta(hours=16)


# ---------------------------------------------------------------- 일봉

def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def daily_features():
    # 마지막 대상일의 정답(16:00 확정)까지 필요해서 일봉만 그날 16:00 까지 읽음.
    # 피처는 B 행에서 만들고 T 의 값은 결과(y)로만 붙으므로 피처에는 섞이지 않음.
    d = read("daily", until=EDA_LAST_TARGET + CLOSE + pd.Timedelta(minutes=1)).sort_values(["symbol", "date_et"]).reset_index(drop=True)
    d = d.rename(columns={"date_et": "date"})
    g = d.groupby("symbol", group_keys=False)

    f = pd.DataFrame({"symbol": d.symbol, "date": d.date})
    r = d.ret
    f["r1"] = r * 100
    f["r5"] = (d.close / g.close.shift(5) - 1) * 100
    f["r20"] = (d.close / g.close.shift(20) - 1) * 100
    f["r60"] = (d.close / g.close.shift(60) - 1) * 100
    f["vol20"] = g.ret.transform(lambda s: s.rolling(20, min_periods=15).std()) * 100
    f["vol60"] = g.ret.transform(lambda s: s.rolling(60, min_periods=40).std()) * 100
    f["vol5"] = g.ret.transform(lambda s: s.rolling(5, min_periods=5).std()) * 100
    tr = pd.concat([d.high - d.low, (d.high - d.prev_close).abs(),
                    (d.low - d.prev_close).abs()], axis=1).max(axis=1) / d.prev_close
    f["atr14"] = tr.groupby(d.symbol).transform(lambda s: s.rolling(14, min_periods=10).mean()) * 100
    f["range1"] = (d.high - d.low) / d.prev_close * 100
    f["gap1"] = (d.open / d.prev_close - 1) * 100            # B 의 시가 갭
    f["intra1"] = (d.close / d.open - 1) * 100               # B 의 장중 수익률
    f["rsi14"] = g.close.transform(_rsi)
    f["dist_ma20"] = (d.close / g.close.transform(lambda s: s.rolling(20).mean()) - 1) * 100
    f["dist_ma50"] = (d.close / g.close.transform(lambda s: s.rolling(50).mean()) - 1) * 100
    f["dist_hi20"] = (d.close / g.high.transform(lambda s: s.rolling(20).max()) - 1) * 100
    vavg = g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    f["volu_ratio"] = d.volume / vavg
    f["dvol20"] = (d.close * d.volume).groupby(d.symbol).transform(
        lambda s: s.rolling(20, min_periods=15).mean())       # 20일 평균 거래대금
    f["z1"] = f.r1 / f.vol20
    f["vol_ratio"] = f.vol5 / f.vol60                        # 단기/장기 변동성
    f["close_B"] = d.close
    f["vavg20"] = g.volume.transform(lambda s: s.rolling(20, min_periods=15).mean())
    sign = np.sign(r).fillna(0)
    f["streak"] = sign.groupby([d.symbol, (sign != sign.groupby(d.symbol).shift()).cumsum()]
                               ).cumsum()

    # 결과(대상일 기준) : 다음 행의 값
    y = pd.DataFrame({"symbol": d.symbol, "target": d.date,
                      "ret_pct": d.ret * 100,
                      "y_gap_open": (d.open / d.prev_close - 1) * 100,
                      "y_intraday": (d.close / d.open - 1) * 100,
                      "y_range": (d.high - d.low) / d.prev_close * 100,
                      "y_volume": d.volume})
    f["target"] = g.date.shift(-1)
    return f.dropna(subset=["target"]), y.dropna(subset=["ret_pct"])


# ---------------------------------------------------------------- 시간외

def extended_features(dates):
    p = read("price", ["symbol", "datetime", "close", "open", "volume", "session"])
    p = p[p.session != "regular"].copy()
    p["day"] = p.datetime.dt.normalize()
    nxt = pd.Series(dates[1:], index=dates[:-1])
    # post 봉은 다음 거래일 T 에, pre 봉은 그 날짜 T 에 귀속
    p["target"] = np.where(p.session == "post", p.day.map(nxt), p.day)
    p["target"] = pd.to_datetime(p.target)
    p = p.dropna(subset=["target"])
    p = p[p.known_at < p.target + CUT]                       # cutoff 이전에 닫힌 봉만
    p = p.sort_values(["symbol", "target", "datetime"])

    # 시간외 봉은 거래량이 기록돼 있지 않음(99.99% 가 0). 가격은 정상적으로 움직이므로
    # 거래량으로 거르지 않음. 시간외 거래량 피처는 만들 수 없음.
    live = p
    agg = p.groupby(["symbol", "target"]).agg(n_ext_bars=("close", "size"))
    last = live.groupby(["symbol", "target"]).close.last().rename("ext_last")
    post = live[live.session == "post"].groupby(["symbol", "target"]).close.last().rename("post_last")
    pre = live[live.session == "pre"].groupby(["symbol", "target"])
    pre_first = pre.close.first().rename("pre_first")
    pre_last = pre.close.last().rename("pre_last")
    return pd.concat([agg, last, post, pre_first, pre_last], axis=1).reset_index()


# ---------------------------------------------------------------- 이벤트 → T 매핑

def map_to_target(ts, dates):
    """known_at 을 그 정보를 처음 쓸 수 있는 대상일 T 로. T 09:30 > known_at 인 첫 거래일."""
    cut = dates + CUT
    i = np.searchsorted(cut.values, ts.values, side="right")
    ok = i < len(dates)
    T = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[us]")
    T[ok] = dates[i[ok]]
    B = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[us]")
    ok2 = ok & (i > 0)
    B[ok2] = dates[i[ok2] - 1]
    overnight = ts >= (B + CLOSE)
    return T, B, overnight


def earnings_features(dates):
    e = read("earnings").copy()
    e["T"], e["B"], e["overnight"] = map_to_target(e.known_at, dates)
    e["when"] = np.select([e.known_at.dt.normalize() == e["T"], e.overnight],
                          ["pre_open", "after_close"], "intraday_B")
    out = e.rename(columns={"T": "target"})[
        ["symbol", "target", "when", "surprise_pct", "eps_estimate", "eps_reported", "known_at"]]
    out = out.dropna(subset=["target"]).sort_values("known_at").drop_duplicates(
        ["symbol", "target"], keep="last")
    return out.rename(columns={"when": "earn_when", "surprise_pct": "earn_surprise",
                               "known_at": "earn_known_at"})[
        ["symbol", "target", "earn_when", "earn_surprise", "earn_known_at"]]


def analyst_features(dates):
    a = read("analyst").copy()
    a = a[a.known_at >= pd.Timestamp("2023-09-01")]
    a["target"], a["B"], a["overnight"] = map_to_target(a.known_at, dates)
    ta = a.target_action.str.lower()
    a["tgt_chg"] = (a.target_current / a.target_prior - 1) * 100
    a = a.assign(an_up=a.action.eq("up"), an_down=a.action.eq("down"),
                 an_init=a.action.eq("init"), an_raise=ta.eq("raises"),
                 an_lower=ta.eq("lowers"), an_n=1)
    return a.groupby(["symbol", "target"]).agg(
        an_n=("an_n", "sum"), an_up=("an_up", "sum"), an_down=("an_down", "sum"),
        an_init=("an_init", "sum"), an_raise=("an_raise", "sum"),
        an_lower=("an_lower", "sum"), an_tgt_chg=("tgt_chg", "mean")).reset_index()


def news_features(dates, symbols):
    """기사 하나에 종목이 여러 개 붙어 있어 explode. 50종목만 남김."""
    n = read("news", ["symbols", "tone", "positive", "negative"])
    n = n[n.known_at >= EDA_FIRST_TARGET - pd.Timedelta(days=60)]
    n["target"], n["B"], n["overnight"] = map_to_target(n.known_at, dates)
    n = n.dropna(subset=["target"])
    n["n_tags"] = n.symbols.fillna("").str.count(",") + 1
    n["symbol"] = n.symbols.fillna("").str.split(",")
    n = n.drop(columns="symbols").explode("symbol")
    n = n[n.symbol.isin(symbols)]
    n["solo"] = n.n_tags == 1                               # 이 종목만 다룬 기사
    grp = n.groupby(["symbol", "target"])
    out = grp.agg(news_n=("tone", "size"), news_tone=("tone", "mean"),
                  news_neg=("negative", "mean"), news_pos=("positive", "mean"),
                  news_solo=("solo", "sum")).reset_index()
    ovn = n[n.overnight].groupby(["symbol", "target"]).agg(
        news_n_ovn=("tone", "size"), news_tone_ovn=("tone", "mean")).reset_index()
    return out.merge(ovn, on=["symbol", "target"], how="left")


# ---------------------------------------------------------------- 조립

def build():
    feats, y = daily_features()
    dates = pd.DatetimeIndex(np.sort(y.target.unique()))
    symbols = sorted(y.symbol.unique())

    df = feats.merge(y, on=["symbol", "target"], how="inner")
    df = df[(df.target >= EDA_FIRST_TARGET) & (df.target <= EDA_LAST_TARGET)]
    df = df.rename(columns={"date": "base"})

    print("  시간외 봉...", flush=True)
    df = df.merge(extended_features(dates), on=["symbol", "target"], how="left")
    df["ovn_gap"] = (df.ext_last / df.close_B - 1) * 100            # 전일 종가 → cutoff 직전 체결가
    df["post_ret"] = (df.post_last / df.close_B - 1) * 100
    df["pre_move"] = (df.pre_last / df.pre_first - 1) * 100          # lab/baselines.py 의 gap 과 같은 정의
    df["pre_vs_post"] = (df.pre_last / df.post_last - 1) * 100
    df["ovn_gap_z"] = df.ovn_gap / df.vol20

    print("  실적·애널리스트·뉴스...", flush=True)
    df = df.merge(earnings_features(dates), on=["symbol", "target"], how="left")
    df["earn"] = df.earn_when.notna() & df.earn_when.ne("intraday_B")
    df = df.merge(analyst_features(dates), on=["symbol", "target"], how="left")
    for c in ["an_n", "an_up", "an_down", "an_init", "an_raise", "an_lower"]:
        df[c] = df[c].fillna(0)
    df = df.merge(news_features(dates, symbols), on=["symbol", "target"], how="left")
    df["news_n"] = df.news_n.fillna(0)
    df["news_n_ovn"] = df.news_n_ovn.fillna(0)
    df["news_solo"] = df.news_solo.fillna(0)

    # 종목별 과거 대비 비정상 기사량 (직전 20개 대상일 평균, 당일 제외)
    df = df.sort_values(["symbol", "target"])
    base_n = df.groupby("symbol").news_n.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    df["news_abn"] = np.log1p(df.news_n) - np.log1p(base_n)

    # 시장(50종목 등가중) 피처와 결과
    m = df.groupby("target")
    df["mkt_gap"] = m.ovn_gap.transform("mean")
    df["rel_gap"] = df.ovn_gap - df.mkt_gap
    df["mkt_r1"] = m.r1.transform("mean")
    df["rel_r1"] = df.r1 - df.mkt_r1
    df["rel_r5"] = df.r5 - m.r5.transform("mean")
    df["rel_r20"] = df.r20 - m.r20.transform("mean")
    df["y_mkt"] = m.ret_pct.transform("mean")
    df["y_idio"] = df.ret_pct - df.y_mkt

    df["label"] = label_of(df.ret_pct).astype(int)
    df["unseen"] = df.symbol.isin(UNSEEN)
    df["weekday"] = df.target.dt.dayofweek
    df["month"] = df.target.dt.to_period("M").astype(str)
    df["half"] = df.target.dt.year.astype(str) + np.where(df.target.dt.month <= 6, "H1", "H2")

    df = df.sort_values(["target", "symbol"]).reset_index(drop=True)
    CACHE.mkdir(exist_ok=True)
    df.to_parquet(CACHE / "eda_panel.parquet", index=False)
    print(f"  패널 {df.shape}  대상일 {df.target.min():%Y-%m-%d} ~ {df.target.max():%Y-%m-%d}")
    return df


if __name__ == "__main__":
    build()
