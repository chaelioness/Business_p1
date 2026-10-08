"""뉴스·시장 실험: ① 뉴스 있는 갭 vs 없는 갭, ② 시장 전체가 뒤집힌 날, ③ 뉴스 수 baseline 과 LLM 피처 비교.

    uv run python work/b/review/news_gap/news_gap.py

사전 등록 (실행 전에 고정, 결과를 보고 바꾸지 않음)
----------------------------------------------------
공통
- 표: cache/gapz_rule.parquet (guard 아래 만든 gap_z, ext_range_z), A 폴드 lab/folds.json.
- 갭 이후 움직임 post = (1 + ret) / (1 + gap) − 1   (09:00 시간외 가격 → 당일 종가, 갭이 모르는 부분)
- 뉴스·Reddit 은 기준일 16:00 ≤ 시각 < 대상일 09:00 만 셈 (갭 가격이 09:00 공개라 같은 시점까지만).
- 발견 기간 = 대상일 ≤ 2026-02-12 (EDA·fold4 train), 확인 기간 = fold4 val (기준일 2026-03-04 ~ 05-28).

① 뉴스 있는 갭 vs 없는 갭 (가설: 10/3 B 문서 6절 "뉴스 없이 혼자 튄 갭이 더 많이 되돌려지는가")
- 대상 행: |gap_z| ≥ 0.25, post ≠ 0
- 그룹: G0 태그 기사 없음 / G1 태그 기사는 있지만 제목에 기업명 없음 /
        G2 기업명이 나온 기사 있음 (G3 아님) / G3 채민 LLM 이 기대 비교(above·below·in_line·mixed)를 확인한 관련 기사 있음
  기업명 판정은 채민 v6 후보 필터와 같은 규칙 (회사명, 티커는 대문자, 짧거나 일반어 티커는 $·거래소 표기만).
- 지표: 갭 방향 유지율 = P(sign(post) = sign(gap)), 갭 방향 급등락 비율
- 판정: (G2∪G3) − G0 의 유지율 차이가 |차이| ≥ 0.05 이고 날짜 묶음 부트스트랩 95% 구간이 0 을 안 걸침 (발견 기간),
        확인 기간에서 같은 부호 → "차이 있음"

③ 뉴스 수 baseline vs LLM 피처 (규칙에 붙여 4폴드)
- 피처: f_count = log1p(태그 기사 수) [baseline], f_name = 1[기업명 기사 있음],
        f_llm_rel = 1[LLM 관련 기사 있음], f_llm_cmp = 1[LLM 기대 비교 기사 있음]. train 기준 표준화.
- 형태 T (갭 신뢰도): s = gap_z·exp(−λ·x̃ + γ·z(f)), λ 는 기본 규칙 값, γ ∈ GAMMAS, a·b 다시 고름
- 형태 S (급등락 경계): |s| ≥ b·exp(−δ·z(f)) 이면 급등락, δ ∈ DELTAS, a·b 다시 고름 (A fit_rule_size 와 같은 방식)
- γ·δ·a·b 는 train score 최대 (동점이면 0 에 가까운 값). 처음 보는 10종목은 학습에서 뺌.
- 채택 기준 (B·A 공통): 4폴드 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음.
- 대조: f_count 를 같은 날 안에서 섞은 가짜 피처 5회 (형태별).

② 시장 전체가 뒤집힌 날 (날 단위)
- 결과: 그날 |gap_z| ≥ 0.25 인 종목 중 갭과 반대로 간 비율 rev_share
- 예측 후보 (09:00 전에 알 수 있는 것): 밤사이 전체 뉴스 수(log), 그 이상치(직전 20일 평균 대비),
  Reddit 글 수(log), 그 이상치, |시장 평균 gap_z|, gap_z 의 종목 간 표준편차, 창 길이(시간)
- 판정: 발견 기간 Spearman p < 0.05 이고 확인 기간에서 같은 부호 → "통과". 통과한 후보만 형태 S 로 4폴드.
"""

import html
import json
import re
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))
sys.path.insert(0, str(ROOT / "work" / "b" / "review" / "explain"))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402  (load, 차트 스타일 재사용)

OUT, FIGS = HERE / "out", HERE / "figs"
LLM_DIR = ROOT / "cache" / "llm_news_v6"     # 채민 llm_news_v6_dev_handoff_final.zip 의 csv 2개
SEED, N_BOOT, N_FAKE = 2026, 1000, 5
GAP_MIN = 0.25
DISC_END = pd.Timestamp("2026-02-12")
GAMMAS = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
GROUPS = ["G0 기사 없음", "G1 태그만", "G2 기업명 기사", "G3 LLM 기대비교"]

# 채민 v6 후보 필터와 같은 기업명 규칙 (BA_llm_1차.ipynb 셀 39)
NAMES = {
    "ADI": ["Analog Devices"], "AMD": ["Advanced Micro Devices"], "AMGN": ["Amgen"], "AMZN": ["Amazon"],
    "AVGO": ["Broadcom"], "AXP": ["American Express"], "BA": ["Boeing"], "BKNG": ["Booking Holdings"],
    "BRK-B": ["Berkshire Hathaway"], "CB": ["Chubb"], "COP": ["ConocoPhillips"], "COST": ["Costco"],
    "CRM": ["Salesforce"], "CSCO": ["Cisco"], "CVS": ["CVS Health", "CVS"], "DE": ["Deere", "John Deere"],
    "DELL": ["Dell"], "DHR": ["Danaher"], "GE": ["GE Aerospace", "General Electric"], "GILD": ["Gilead"],
    "GLW": ["Corning"], "GS": ["Goldman Sachs"], "IBKR": ["Interactive Brokers"], "INTC": ["Intel"],
    "JNJ": ["Johnson & Johnson", "Johnson and Johnson"], "KO": ["Coca-Cola", "Coca Cola"], "LIN": ["Linde"],
    "LLY": ["Eli Lilly"], "MCD": ["McDonald's", "McDonald’s", "McDonalds"], "MS": ["Morgan Stanley"],
    "MSFT": ["Microsoft"], "NEM": ["Newmont"], "NFLX": ["Netflix"], "NOW": ["ServiceNow"], "ORCL": ["Oracle"],
    "PFE": ["Pfizer"], "PLD": ["Prologis"], "PM": ["Philip Morris"], "QCOM": ["Qualcomm"],
    "RTX": ["RTX Corporation", "Raytheon"], "TJX": ["TJX"], "TMO": ["Thermo Fisher"],
    "TMUS": ["T-Mobile", "T Mobile"], "UBER": ["Uber"], "UNP": ["Union Pacific"], "V": ["Visa"],
    "VZ": ["Verizon"], "WELL": ["Welltower"], "WMT": ["Walmart"], "XOM": ["Exxon", "ExxonMobil", "Exxon Mobil"],
}
AMBIGUOUS = {"COST", "NOW", "WELL", "COP", "LIN"}


def name_regex(sym):
    parts = [r"(?i:(?<!\w)" + re.escape(n) + r"(?!\w))" for n in NAMES[sym]]
    forms = [sym] + (["BRK.B"] if sym == "BRK-B" else [])
    for tk in forms:
        e = re.escape(tk)
        if len(sym) <= 2 or sym in AMBIGUOUS:
            parts.append(r"\$" + e + r"(?!\w)|(?:NASDAQ|NYSE)\s*:\s*" + e + r"(?!\w)")
        else:
            parts.append(r"(?<![\w$])" + e + r"(?!\w)")
    return re.compile("|".join(parts))


# ------------------------------------------------------------------ 데이터
def build(t):
    """종목·기준일별 뉴스·LLM 피처, 날짜별 시장 피처."""
    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"] = w.date + pd.Timedelta(hours=16)
    w["cut9"] = w.target + pd.Timedelta(hours=9)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("w", w)
    con.register("u", pd.DataFrame({"symbol": sorted(t.symbol.unique())}))
    news = str(ROOT / "dataset" / "news.parquet")
    lo, hi = str(w.since.min()), str(w.cut9.max())
    tagged = con.execute(f"""
        WITH n AS (SELECT symbols, title, known_at FROM read_parquet('{news}')
                   WHERE known_at >= '{lo}' AND known_at < '{hi}' AND title IS NOT NULL),
             x AS (SELECT trim(tag) AS symbol, title, known_at FROM n, UNNEST(string_split(symbols, ',')) AS s(tag))
        SELECT w.date, x.symbol, x.title FROM x JOIN u USING (symbol)
        JOIN w ON x.known_at >= w.since AND x.known_at < w.cut9""").df()
    total = con.execute(f"""
        SELECT w.date, count(*) AS news_total FROM read_parquet('{news}') n
        JOIN w ON n.known_at >= w.since AND n.known_at < w.cut9 GROUP BY w.date""").df()
    rd = str(ROOT / "dataset" / "reddit" / "*.posts.parquet")
    reddit = con.execute(f"""
        SELECT w.date, count(*) AS reddit_total FROM read_parquet('{rd}') r
        JOIN w ON r.created_et >= w.since AND r.created_et < w.cut9 GROUP BY w.date""").df()
    con.close()

    tagged["title"] = tagged.title.map(lambda s: html.unescape(str(s)).strip())
    tagged["key"] = tagged.title.str.replace(r"\s+", " ", regex=True).str.casefold()
    tagged = tagged.drop_duplicates(["date", "symbol", "key"])
    tagged["name"] = False
    for sym, idx in tagged.groupby("symbol").groups.items():
        rx = name_regex(sym)
        tagged.loc[idx, "name"] = tagged.loc[idx, "title"].map(lambda s: bool(rx.search(s)))
    per = tagged.groupby(["date", "symbol"]).agg(n_tag=("key", "size"), n_name=("name", "sum")).reset_index()

    v = pd.read_csv(LLM_DIR / "valid_articles.csv")
    for c in ["date", "target", "known_at"]:
        v[c] = pd.to_datetime(v[c], format="mixed").astype("datetime64[us]")   # 확장분은 "00:00:00" 이 붙어 있음
    if v.date.isna().any():
        raise ValueError("valid_articles 날짜 변환 실패")
    a = v.annotation_json.map(json.loads)
    v["rel"] = a.map(lambda x: x.get("relevance") in ("direct", "indirect"))
    v["cmp"] = v.rel & a.map(lambda x: x.get("surprise") in ("above", "below", "in_line", "mixed"))
    v = v[v.known_at < v.target + pd.Timedelta(hours=9)]
    llm = v.groupby(["date", "symbol"]).agg(llm_rel=("rel", "any"), llm_cmp=("cmp", "any")).reset_index()

    t = t.merge(per, on=["date", "symbol"], how="left").merge(llm, on=["date", "symbol"], how="left")
    for c in ["n_tag", "n_name"]:
        t[c] = t[c].fillna(0)
    for c in ["llm_rel", "llm_cmp"]:
        t[c] = t[c].astype("boolean").fillna(False).astype(bool)
    t["post"] = (1 + t.ret_pct / 100) / (1 + t.gap) - 1
    t["grp"] = np.select([t.llm_cmp, t.n_name > 0, t.n_tag > 0], GROUPS[3:0:-1], GROUPS[0])
    t["f_count"] = np.log1p(t.n_tag)
    t["f_name"] = (t.n_name > 0).astype(float)
    t["f_llm_rel"] = t.llm_rel.astype(float)
    t["f_llm_cmp"] = t.llm_cmp.astype(float)

    day = w.merge(total, on="date", how="left").merge(reddit, on="date", how="left").sort_values("date")
    day[["news_total", "reddit_total"]] = day[["news_total", "reddit_total"]].fillna(0)
    day["win_hours"] = (day.cut9 - day.since).dt.total_seconds() / 3600
    for c in ["news_total", "reddit_total"]:
        lg = np.log1p(day[c])
        day[f"log_{c}"] = lg
        day[f"abn_{c}"] = lg - lg.shift(1).rolling(20, min_periods=10).mean()
    return t, day


def folds_of(t):
    spec = json.loads((ROOT / "lab" / "folds.json").read_text(encoding="utf-8"))
    out = []
    for f in spec["folds"]:
        out.append((f["name"], t[t.date.between(*f["train"]) & ~t.unseen], t[t.date.between(*f["val"])],
                    pd.Timestamp(f["val"][0]), pd.Timestamp(f["val"][1])))
    return out


# ------------------------------------------------------------------ ① 그룹별 갭 방향 유지율
def part1(t, f4):
    d = t[(t.gap_z.abs() >= GAP_MIN) & (t.post != 0) & t.gap_z.notna()].copy()
    d["cont"] = np.sign(d.post) == np.sign(d.gap)
    d["ext_same"] = ((d.label == 4) & (d.gap > 0)) | ((d.label == 0) & (d.gap < 0))
    d["period"] = np.where(d.target <= DISC_END, "발견", np.where(d.date.between(f4[0], f4[1]), "확인", "기타"))
    tab = (d[d.period != "기타"].groupby(["period", "grp"])
           .agg(n=("cont", "size"), 유지율=("cont", "mean"), 갭방향_급등락=("ext_same", "mean"),
                abs_gap_z=("gap_z", lambda s: s.abs().median())).reset_index())

    rng = np.random.default_rng(SEED)
    boot = {}
    for per in ["발견", "확인"]:
        x = d[d.period == per]
        dates = x.date.unique()
        byd = {k: g for k, g in x.groupby("date")}

        def diff(df, a, b):
            return df[df.grp.isin(a)].cont.mean() - df[df.grp.isin(b)].cont.mean()
        comps = {"(G2∪G3) − G0": (GROUPS[2:], GROUPS[:1]), "G3 − G0": (GROUPS[3:], GROUPS[:1]),
                 "G1 − G0": (GROUPS[1:2], GROUPS[:1]), "G2 − G0": (GROUPS[2:3], GROUPS[:1])}
        for name, (a, b) in comps.items():
            est = diff(x, a, b)
            bs = []
            for _ in range(N_BOOT):
                s = pd.concat([byd[k] for k in rng.choice(dates, len(dates))])
                bs.append(diff(s, a, b))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            boot[(per, name)] = {"차이": est, "95%_하한": lo, "95%_상한": hi}
    boot = pd.DataFrame(boot).T.rename_axis(["period", "비교"]).reset_index()
    return d, tab, boot


# ------------------------------------------------------------------ ③ 규칙에 붙이기 (T, S)
def _z(train, x):
    m, s = float(np.nanmean(train)), float(np.nanstd(train))
    return np.nan_to_num((x - m) / s) if s > 0 else np.zeros(len(x))


def _s0(x, P):
    return G._score_s(x.gap_z.to_numpy(float), x.ext_range_z.to_numpy(float), P["med"], P["sd"], P["lam"])


def _cut_size(s, a, b, zf, d):
    o = G.cut(s, a, np.inf)
    bb = b * np.exp(-d * zf)
    o[s >= bb], o[s <= -bb] = 4, 0
    o[np.isnan(s)] = 2
    return o


def fit_variant(tr, va, feat, form, P):
    y = tr.label.to_numpy(int)
    s_tr, s_va = _s0(tr, P), _s0(va, P)
    z_tr, z_va = _z(tr[feat].to_numpy(float), tr[feat].to_numpy(float)), _z(tr[feat].to_numpy(float), va[feat].to_numpy(float))
    if form == "T":
        best = max(((*G._fit_ab(s_tr * np.exp(g * z_tr), y), g) for g in GAMMAS),
                   key=lambda r: (round(r[0], 10), -abs(r[3])))
        sc, a, b, g = best
        return G.cut(s_va * np.exp(g * z_va), a, b), g
    ok = ~np.isnan(s_tr)
    q = np.quantile(np.abs(s_tr[ok]), G.QS)
    best = max(((G.score(y, _cut_size(s_tr, a, b, z_tr, d)), a, b, d)
                for i, a in enumerate(q) for b in q[i + 1:] for d in DELTAS),
               key=lambda r: (round(r[0], 10), -abs(r[3])))
    sc, a, b, d = best
    return _cut_size(s_va, a, b, z_va, d), d


def part3(t, folds, feats, rng):
    rows = []
    for name, tr, va, _, _ in folds:
        P = G.fit_table(tr)
        base = G.predict_table(va, P)
        u = va.unseen.to_numpy()
        yb = va.label.to_numpy(int)
        rows.append({"fold": name, "feature": "기본 규칙", "form": "-", "param": P["lam"],
                     "score": G.score(yb, base), "unseen": G.score(yb[u], base[u])})
        for feat in feats:
            for form in ["T", "S"]:
                p, prm = fit_variant(tr, va, feat, form, P)
                rows.append({"fold": name, "feature": feat, "form": form, "param": prm,
                             "score": G.score(yb, p), "unseen": G.score(yb[u], p[u])})
        for k in range(N_FAKE):
            trf, vaf = tr.copy(), va.copy()
            for df in (trf, vaf):
                df["f_fake"] = df.groupby("date").f_count.transform(lambda s: rng.permutation(s.to_numpy()))
            for form in ["T", "S"]:
                p, prm = fit_variant(trf, vaf, "f_fake", form, P)
                rows.append({"fold": name, "feature": f"가짜{k}", "form": form, "param": prm,
                             "score": G.score(yb, p), "unseen": G.score(yb[u], p[u])})
    r = pd.DataFrame(rows)
    base = r[r.feature == "기본 규칙"].set_index("fold").score
    r["delta"] = r.score - r.fold.map(base)
    return r


def judge(r):
    out = []
    for (feat, form), g in r[r.feature != "기본 규칙"].groupby(["feature", "form"], sort=False):
        g = g.set_index("fold")
        out.append({"feature": feat, "form": form,
                    **{f: g.delta.get(f) for f in ["fold1", "fold2", "fold3", "fold4"]},
                    "평균": g.delta.mean(), "상승폴드": int((g.delta > 0).sum()),
                    "파라미터": " / ".join(str(x) for x in g.param),
                    "채택": bool(g.delta.mean() >= 0.005 and (g.delta > 0).sum() >= 3 and g.delta["fold4"] >= 0)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ ② 날 단위
PRED = ["log_news_total", "abn_news_total", "log_reddit_total", "abn_reddit_total",
        "abs_mkt_gap", "gap_disp", "win_hours"]


def part2(t, day, f4):
    d = t[t.gap_z.notna()].copy()
    big = d[(d.gap_z.abs() >= GAP_MIN) & (d.post != 0)]
    agg = d.groupby("date").agg(mkt_gap=("gap_z", "mean"), gap_disp=("gap_z", "std"))
    agg["abs_mkt_gap"] = agg.mkt_gap.abs()
    agg["rev_share"] = big.groupby("date").apply(lambda g: (np.sign(g.post) != np.sign(g.gap)).mean())
    agg["n_big"] = big.groupby("date").size()
    x = day.merge(agg.reset_index(), on="date", how="inner")
    x = x[x.n_big >= 5]
    x["period"] = np.where(x.target <= DISC_END, "발견", np.where(x.date.between(f4[0], f4[1]), "확인", "기타"))
    rows = []
    for p in PRED:
        r = {"예측 후보": p}
        for per in ["발견", "확인"]:
            y = x[x.period == per][[p, "rev_share"]].dropna()
            rho, pv = stats.spearmanr(y[p], y.rev_share)
            r[f"{per}_rho"], r[f"{per}_p"], r[f"{per}_n"] = rho, pv, len(y)
        r["통과"] = bool(r["발견_p"] < 0.05 and np.sign(r["발견_rho"]) == np.sign(r["확인_rho"]))
        rows.append(r)
    return x, pd.DataFrame(rows)


# ------------------------------------------------------------------ 그림
def fig1(tab, boot):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, per in zip(axes, ["발견", "확인"]):
        s = tab[tab.period == per].set_index("grp").reindex(GROUPS)
        cols = [E.C1, E.C1, E.C2, E.C3]
        ax.bar(range(4), s["유지율"], color=cols, width=0.62, edgecolor=E.SURF, linewidth=2)
        for i, (v, n) in enumerate(zip(s["유지율"], s["n"])):
            if pd.notna(v):
                ax.annotate(f"{v:.1%}\nn={int(n):,}", (i, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=E.INK)
        ax.axhline(0.5, color=E.INK2, linewidth=0.8, linestyle="--")
        ax.set_xticks(range(4), [g.replace(" ", "\n", 1) for g in GROUPS], fontsize=8)
        ax.set_ylim(0.4, 0.75)
        title = "발견 기간 (대상일 ~2026-02-12)" if per == "발견" else "확인 기간 (fold4 val)"
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("09:00 이후 갭 방향 유지율")
    fig.suptitle(f"① 뉴스가 있는 갭은 덜 되돌려지는가  (|gap_z| ≥ {GAP_MIN})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=E.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "1_news_groups.png", dpi=160)
    plt.close(fig)


def fig3(r):
    real = r[(r.feature != "기본 규칙") & ~r.feature.str.startswith("가짜")]
    fake = r[r.feature.str.startswith("가짜")]
    labels = {"f_count": "뉴스 수 (baseline)", "f_name": "기업명 기사 있음",
              "f_llm_rel": "LLM 관련 기사 있음", "f_llm_cmp": "LLM 기대비교 있음"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, form, title in [(axes[0], "T", "형태 T: 갭 신뢰도 조절"), (axes[1], "S", "형태 S: 급등락 경계 이동")]:
        rows = list(labels)
        for i, feat in enumerate(rows):
            g = real[(real.feature == feat) & (real.form == form)]
            ax.scatter(g.delta, [i] * len(g), color=E.C1, s=28, zorder=3, label="폴드별" if i == 0 else None)
            ax.scatter([g.delta.mean()], [i], color=E.C2, marker="D", s=46, zorder=4,
                       label="4폴드 평균" if i == 0 else None)
        fk = fake[fake.form == form].groupby("feature").delta.mean()
        ax.axvspan(fk.min(), fk.max(), color=E.GRID, alpha=0.8, linewidth=0, label="가짜 피처 평균 범위")
        ax.axvline(0, color=E.INK2, linewidth=0.8)
        ax.axvline(0.005, color=E.INK2, linewidth=0.8, linestyle=":")
        ax.set_yticks(range(len(rows)), [labels[k] for k in rows])
        ax.set_title(title, loc="left")
        ax.set_xlabel("기본 규칙 대비 val 점수 차이")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("③ 뉴스 수 baseline vs LLM 피처 — 규칙에 붙였을 때 (점선 = 채택 기준 +0.005)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold", color=E.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "3_baseline.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ 실행
def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    t, _ = E.load()
    t, day = build(t)
    folds = folds_of(t)
    f4 = (folds[3][3], folds[3][4])
    print("표", t.shape, "| 그룹별 행 수", t.grp.value_counts().to_dict())

    d, tab, boot = part1(t, f4)
    tab.to_csv(OUT / "1_groups.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "1_boot.csv", index=False, encoding="utf-8-sig")
    fig1(tab, boot)
    print("\n[①]\n", tab.round(3).to_string(index=False), "\n", boot.round(3).to_string(index=False))

    x, cor = part2(t, day, f4)
    cor.to_csv(OUT / "2_day_corr.csv", index=False, encoding="utf-8-sig")
    print("\n[②] 날 수", x.period.value_counts().to_dict(), "\n", cor.round(3).to_string(index=False))
    passed = [p for p in cor[cor["통과"]]["예측 후보"]]

    rng = np.random.default_rng(SEED)
    feats = ["f_count", "f_name", "f_llm_rel", "f_llm_cmp"]
    for p in passed:   # ② 통과한 날 단위 후보만 형태 S 로
        t[f"day_{p}"] = t.date.map(x.set_index("date")[p])
        feats.append(f"day_{p}")
    folds = folds_of(t)
    r = part3(t, folds, feats, rng)
    r.to_csv(OUT / "3_cv.csv", index=False, encoding="utf-8-sig")
    j = judge(r)
    j.to_csv(OUT / "3_judge.csv", index=False, encoding="utf-8-sig")
    fig3(r[~r.feature.str.startswith("day_")])
    print("\n[③] 기본 규칙", r[r.feature == "기본 규칙"][["fold", "score", "unseen"]].round(4).to_string(index=False))
    print(j.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
