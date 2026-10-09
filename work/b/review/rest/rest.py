"""01·02 문서에서 남은 항목을 한 번에 — 02 의 5·6·8·12번, 01 의 집계 규칙·재보도 수·Reddit 작성자·별칭 효과.

    uv run python work/b/review/rest/rest.py            # 전부
    uv run python work/b/review/rest/rest.py lgb        # LightGBM (5·6번)만

사전 등록 (실행 전에 고정, 결과를 보고 바꾸지 않음)
----------------------------------------------------
[02-5] LightGBM 상관 피처 묶음 중요도
  - A 의 LightGBM (B 피처 26개, 같은 파라미터·조기 종료·공격성 k 디코딩)을 cache/b_v2.parquet 로 폴드마다 다시 학습
  - SHAP(pred_contrib)을 묶음별로 합침: 방향 몫 = 급상승 − 급하락 logit, 크기 몫 = −보합 logit
  - 묶음 permutation: 같은 날 안에서 묶음 전체를 같이 섞기 10회, gap·gap_z·gap_rel_z 를 하나씩 섞은 것과 비교
  - 묶음: 갭 7 / 변동성 8 / 추세·수익률 11 (아래 GROUPS)
[02-6] faithfulness (w6-1 p23)
  - val SHAP 절댓값 합 상위 k 개(k = 1, 3, 5)를 빼고 다시 학습 vs 무작위 k 개 5회
  - 판정: 상위 k 를 뺄 때의 점수 하락이 무작위 k 하락 범위 밖(더 큼)이면 "설명이 충실"
[02-8] 실적 서프라이즈 × 갭 방향
  - 실적 발표 밤, EPS 서프라이즈 부호와 갭 부호가 같음 / 반대. 09:00 이후 갭 방향 유지율, 갭 방향 급등락 비율
  - 표본이 작아 관찰만 (규칙에 안 넣음)
[02-12] 처음 보는 종목
  - (a) 처음 보는 10종목을 학습에 넣었을 때 vs 뺐을 때, 그 10종목의 val 점수 (4폴드). 차이 없으면 "종목 과적합 아님"
  - (b) 종목별 val 점수의 폴드 간 순위상관 (같은 종목이 계속 낮은가)
  - (c) 낮은 종목 · 높은 종목의 시간외 봉 수, |gap_z|, ext_range_z, 실적일 수, 갭 되돌림 비율
[01] 집계 규칙 2개 (w5-1 p16–17): LLM 뉴스·Reddit 방향을 비율 대신 "있다/없다" 이진으로 (형태 D)
[01] 재보도 수 (w4-2 p17, p45): 같은 종목·날 제목을 문자 5-gram Jaccard ≥ 0.5 로 묶어 고유 사건 수 · 재보도 비율 (형태 T·S)
[01] Reddit 작성자 (w4-2 p44): 종목 언급 글의 작성자 수, 데이터에 처음 나온 지 30일 안 된 작성자 비율 (형태 T·S)
[01] 별칭 넓히기: 뉴스 기대 비교 후보를 Reddit v4 별칭으로 다시 세면 몇 건 늘어나나 (API 없음, 세기만)
  - 01 피처 실험의 판정은 06 문서와 같음: 채택 기준 + 06 가짜 피처 95 분위수(+0.0004)
"""

import html
import json
import re
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap", "work/b/review/batch"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402
import batch as B  # noqa: E402
from features_b import B_FEATURES_V1 as FEATS  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
SEED = 2026
NULL95 = 0.0004            # 06 문서 가짜 피처 95 분위수
DISC_END = pd.Timestamp("2026-02-12")
PARAMS = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=15, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0, verbose=-1,
              num_threads=4)
KS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LV = np.arange(5)
GROUPS = {
    "갭": ["gap", "gap_z", "abs_gap_z", "gap_rel_z", "mkt_gap", "pre_move", "post_ret"],
    "변동성": ["vol5", "vol20", "vol60", "atr14", "range1", "bigfreq60", "mkt_vol20", "beta60"],
    "추세·수익률": ["ret1", "ret5", "ret1_z", "ret5_z", "intraday1", "gap_open1", "last_hour_z", "rsi14",
                "dist_ma20_z", "macd_hist", "mkt_ret1"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATS)


# ------------------------------------------------------------------ LightGBM (A lgb.py 와 같은 절차)
def _props(y):
    return np.bincount(np.asarray(y, int), minlength=5) / len(y)


def _th(e, q):
    q = np.asarray(q, float) / np.sum(q)
    return np.quantile(e, np.cumsum(q)[:-1])


def _aggr(q, k):
    q = np.asarray(q, float).copy()
    q[[0, 4]] *= k
    q[2] /= k
    return q / q.sum()


def lgb_fit(tr, feats):
    ds = lambda d: lgb.Dataset(d[feats], d.label)  # noqa: E731
    cut = np.sort(tr.date.unique())[-40]
    fit, val = tr[tr.date < cut], tr[tr.date >= cut]
    m_es = lgb.train(PARAMS, ds(fit), 2000, valid_sets=[ds(val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(m_es.best_iteration, 20)
    m = lgb.train(PARAMS, ds(tr), rounds)
    q = _props(tr.label)
    e_es = m_es.predict(fit[feats]) @ LV
    ev = m_es.predict(val[feats]) @ LV
    k = max(KS, key=lambda k: G.score(val.label.to_numpy(), np.searchsorted(_th(e_es, _aggr(q, k)), ev, side="right")))
    th = _th(m.predict(tr[feats]) @ LV, _aggr(q, k))
    return m, th


def lgb_pred(m, th, x, feats):
    return np.searchsorted(th, m.predict(x[feats]) @ LV, side="right")


def part_lgb(folds, rng):
    shap_rows, perm_rows, faith_rows = [], [], []
    for name, tr, va, _, _ in folds:
        m, th = lgb_fit(tr, FEATS)
        y = va.label.to_numpy(int)
        base = G.score(y, lgb_pred(m, th, va, FEATS))
        faith_rows.append({"fold": name, "kind": "기본", "k": 0, "rep": 0, "score": base})
        c = m.predict(va[FEATS], pred_contrib=True).reshape(len(va), 5, len(FEATS) + 1)[:, :, :-1]
        dirc, magc = c[:, 4, :] - c[:, 0, :], -c[:, 2, :]
        tot = np.abs(c).sum(axis=1).mean(axis=0)
        for g, cols in GROUPS.items():
            idx = [FEATS.index(f) for f in cols]
            shap_rows.append({"fold": name, "group": g, "방향": np.abs(dirc[:, idx].sum(1)).mean(),
                              "크기": np.abs(magc[:, idx].sum(1)).mean(),
                              "방향_개별합": np.abs(dirc[:, idx]).mean(0).sum()})
        # permutation: 묶음 / 갭 피처 하나씩
        vv = va.reset_index(drop=True)
        units = {f"[묶음] {g}": cols for g, cols in GROUPS.items()}
        units.update({f"[하나] {f}": [f] for f in ["gap", "gap_z", "gap_rel_z"]})
        for unit, cols in units.items():
            for rep in range(10):
                x = vv.copy()
                ix = np.arange(len(x))
                for _, gi in x.groupby("date").groups.items():
                    gi = np.asarray(gi)
                    ix[gi] = rng.permutation(gi)
                x[cols] = vv[cols].to_numpy()[ix]
                perm_rows.append({"fold": name, "unit": unit, "rep": rep,
                                  "fall": base - G.score(y, lgb_pred(m, th, x, FEATS))})
        # faithfulness
        order = [FEATS[i] for i in np.argsort(-tot)]
        for k in [1, 3, 5]:
            keep = [f for f in FEATS if f not in order[:k]]
            m2, th2 = lgb_fit(tr, keep)
            faith_rows.append({"fold": name, "kind": "상위", "k": k, "rep": 0, "removed": ",".join(order[:k]),
                               "score": G.score(y, lgb_pred(m2, th2, va, keep))})
            for rep in range(5):
                rm = list(rng.choice(FEATS, k, replace=False))
                keep = [f for f in FEATS if f not in rm]
                m2, th2 = lgb_fit(tr, keep)
                faith_rows.append({"fold": name, "kind": "무작위", "k": k, "rep": rep, "removed": ",".join(rm),
                                   "score": G.score(y, lgb_pred(m2, th2, va, keep))})
        print(f"  LightGBM {name}: val {base:.4f}, 상위 5 피처 {order[:5]}", flush=True)
    f = pd.DataFrame(faith_rows)
    f["fall"] = f.fold.map(f[f.kind == "기본"].set_index("fold").score) - f.score
    return pd.DataFrame(shap_rows), pd.DataFrame(perm_rows), f


# ------------------------------------------------------------------ 02-8 실적 × 갭
def part_earn(t):
    e = pd.read_parquet(ROOT / "dataset" / "earnings.parquet")
    x = t[t.earn & t.gap_z.notna() & (t.post != 0)].copy()
    x["since"], x["cutoff"] = x.date + pd.Timedelta(hours=16), x.target + pd.Timedelta(hours=9, minutes=30)
    m = x.merge(e[["symbol", "known_at", "surprise_pct"]], on="symbol")
    m = m[(m.known_at >= m.since) & (m.known_at < m.cutoff)].drop_duplicates(["date", "symbol"])
    m = m[m.surprise_pct.notna() & (m.surprise_pct != 0) & (m.gap != 0)]
    m["관계"] = np.where(np.sign(m.surprise_pct) == np.sign(m.gap), "서프라이즈 = 갭 방향", "서프라이즈 ≠ 갭 방향")
    m["유지"] = np.sign(m.post) == np.sign(m.gap)
    m["급등락_갭방향"] = ((m.label == 4) & (m.gap > 0)) | ((m.label == 0) & (m.gap < 0))
    m["급등락_반대"] = ((m.label == 0) & (m.gap > 0)) | ((m.label == 4) & (m.gap < 0))
    tab = m.groupby("관계").agg(n=("유지", "size"), 유지율=("유지", "mean"), 갭방향_급등락=("급등락_갭방향", "mean"),
                              반대_급등락=("급등락_반대", "mean"), abs_gap_z=("gap_z", lambda s: s.abs().median())).reset_index()
    a, b = m[m.관계.str.contains("=")].유지, m[m.관계.str.contains("≠")].유지
    p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1] if len(b) else np.nan
    return m, tab, p


# ------------------------------------------------------------------ 02-12 처음 보는 종목
def part_unseen(t, folds):
    spec = json.loads((ROOT / "lab" / "folds.json").read_text(encoding="utf-8"))
    rows, per = [], []
    for (name, tr, va, _, _), f in zip(folds, spec["folds"]):
        tr_all = t[t.date.between(*f["train"])]
        u = va.unseen.to_numpy()
        y = va.label.to_numpy(int)
        p_ex = G.predict_table(va, G.fit_table(tr))
        p_in = G.predict_table(va, G.fit_table(tr_all))
        rows.append({"fold": name, "뺐을 때": G.score(y[u], p_ex[u]), "넣었을 때": G.score(y[u], p_in[u]),
                     "학습한 40종목 (뺐을 때 규칙)": G.score(y[~u], p_ex[~u])})
        va = va.assign(pred=p_ex)
        for sym, g in va.groupby("symbol"):
            per.append({"fold": name, "symbol": sym, "unseen": bool(g.unseen.iloc[0]),
                        "score": G.score(g.label, g.pred)})
    a = pd.DataFrame(rows)
    per = pd.DataFrame(per)
    piv = per.pivot(index="symbol", columns="fold", values="score")
    rk = piv.rank()
    corr = rk.corr(method="spearman")
    pair = [corr.iloc[i, j] for i in range(4) for j in range(i + 1, 4)]
    useen = sorted(t[t.unseen].symbol.unique())
    rk_u = piv.loc[useen].rank().corr(method="spearman")
    pair_u = [rk_u.iloc[i, j] for i in range(4) for j in range(i + 1, 4)]
    # 종목 성질 (전 기간, 설명용)
    b2 = pd.read_parquet(ROOT / "cache" / "b_v2.parquet", columns=["date", "symbol", "ext_n"])
    x = t.merge(b2, on=["date", "symbol"], how="left")
    big = x[(x.gap_z.abs() >= 0.25) & (x.post != 0)]
    prop = pd.DataFrame({
        "시간외 봉 수": x.groupby("symbol").ext_n.mean(),
        "|gap_z| 중앙": x.groupby("symbol").gap_z.apply(lambda s: s.abs().median()),
        "ext_range_z 중앙": x.groupby("symbol").ext_range_z.median(),
        "실적일 수": x.groupby("symbol").earn.sum(),
        "갭 되돌림 비율": big.groupby("symbol").apply(lambda g: (np.sign(g.post) != np.sign(g.gap)).mean()),
        "급등락 비율": x.groupby("symbol").label.apply(lambda s: s.isin([0, 4]).mean()),
    })
    prop["4폴드 평균 점수"] = piv.mean(axis=1)
    prop["처음 보는 종목"] = prop.index.isin(useen)
    pc = prop.drop(columns=["처음 보는 종목"]).corr(method="spearman")["4폴드 평균 점수"].drop("4폴드 평균 점수")
    return a, per, piv, (np.mean(pair), np.min(pair), np.max(pair)), (np.mean(pair_u), np.min(pair_u), np.max(pair_u)), prop, pc


# ------------------------------------------------------------------ 01 텍스트 피처
def _grams(s, n=5):
    s = re.sub(r"\s+", " ", s.casefold())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def text_features(t):
    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"], w["cut9"] = w.date + pd.Timedelta(hours=16), w.target + pd.Timedelta(hours=9)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("w", w)
    con.register("u", pd.DataFrame({"symbol": sorted(t.symbol.unique())}))
    news = str(ROOT / "dataset" / "news.parquet")
    lo, hi = str(w.since.min()), str(w.cut9.max())
    n = con.execute(f"""
        WITH n AS (SELECT symbols, title, known_at FROM read_parquet('{news}')
                   WHERE known_at >= '{lo}' AND known_at < '{hi}' AND title IS NOT NULL),
             x AS (SELECT trim(tag) AS symbol, title, known_at FROM n, UNNEST(string_split(symbols, ',')) AS s(tag))
        SELECT w.date, x.symbol, x.title FROM x JOIN u USING (symbol)
        JOIN w ON x.known_at >= w.since AND x.known_at < w.cut9""").df()
    rd = str(ROOT / "dataset" / "reddit" / "*.posts.parquet")
    posts = con.execute(f"""
        SELECT w.date, r.title, r.author, r.created_et FROM read_parquet('{rd}') r
        JOIN w ON r.created_et >= w.since AND r.created_et < w.cut9""").df()
    first = con.execute(f"SELECT author, min(created_et) AS first_seen FROM read_parquet('{rd}') GROUP BY author").df()
    con.close()

    # 재보도: 문자 5-gram Jaccard ≥ 0.5 로 묶기 (같은 종목·날)
    n["title"] = n.title.map(lambda s: html.unescape(str(s)).strip())
    n = n.drop_duplicates(["date", "symbol", "title"])
    ev = []
    for (d, sym), g in n.groupby(["date", "symbol"]):
        reps = []
        for s in g.title:
            gs = _grams(s)
            if not any(len(gs & r) / len(gs | r) >= 0.5 for r in reps):
                reps.append(gs)
        ev.append((d, sym, len(g), len(reps)))
    ev = pd.DataFrame(ev, columns=["date", "symbol", "n_titles", "n_events"])
    ev["events_n"] = np.log1p(ev.n_events)
    ev["reprint_ratio"] = 1 - ev.n_events / ev.n_titles

    # Reddit 작성자: 티커($ 또는 대문자, 짧거나 일반어 티커는 $ 만) 또는 회사명 언급
    posts = posts[posts.author.notna() & ~posts.author.isin(["[deleted]", "AutoModerator"])]
    posts = posts.merge(first, on="author", how="left")
    posts["new"] = (posts.created_et - posts.first_seen) < pd.Timedelta(days=30)
    au = []
    for sym in sorted(t.symbol.unique()):
        names = [re.escape(x) for x in NG.NAMES[sym]]
        tk = re.escape(sym.replace("-", "."))
        tick = r"\$" + tk + r"\b" if (len(sym) <= 2 or sym in NG.AMBIGUOUS) else r"(?<![\w$])\$?" + tk + r"\b"
        rx = re.compile(r"(?i:(?<!\w)(?:" + "|".join(names) + r")(?!\w))|" + tick)
        hit = posts[posts.title.astype(str).str.contains(rx, regex=True)]
        if len(hit):
            g = hit.groupby("date").agg(rd_authors=("author", "nunique"), rd_new_share=("new", "mean")).reset_index()
            au.append(g.assign(symbol=sym))
    au = pd.concat(au)
    au["rd_authors"] = np.log1p(au.rd_authors)

    # 집계 규칙 2: 있다/없다 (채민 기사·글 단위)
    v = pd.read_csv(ROOT / "cache" / "llm_news_v6" / "valid_articles.csv")
    v["target"] = pd.to_datetime(v.target, format="mixed")
    a = v.annotation_json.map(json.loads)
    rel = a.map(lambda x: x.get("relevance") in ("direct", "indirect"))
    v["ab"], v["be"] = rel & a.map(lambda x: x.get("surprise") == "above"), rel & a.map(lambda x: x.get("surprise") == "below")
    nb = v.groupby(["symbol", "target"]).agg(ab=("ab", "any"), be=("be", "any")).reset_index()
    nb["llm_dir_bin"] = nb.ab.astype(int) - nb.be.astype(int)
    rp = pd.read_csv(ROOT / "cache" / "llm_reddit_v4" / "valid_posts.csv")
    rp["target"] = pd.to_datetime(rp.target, format="mixed")
    rp = rp[rp.relevance == "direct"]
    rb = rp.groupby(["symbol", "target"]).stance.agg(lambda s: int((s == "bullish").any()) - int((s == "bearish").any()))
    rb = rb.rename("rd_dir_bin").reset_index()

    t = t.merge(ev[["date", "symbol", "events_n", "reprint_ratio"]], on=["date", "symbol"], how="left")
    t = t.merge(au, on=["date", "symbol"], how="left")
    t = t.merge(nb[["symbol", "target", "llm_dir_bin"]], on=["symbol", "target"], how="left")
    t = t.merge(rb, on=["symbol", "target"], how="left")
    return t, ev


TESTS01 = [("llm_dir_bin", "D", "LLM 상회/하회 있다·없다"), ("rd_dir_bin", "D", "Reddit 상승/하락 의견 있다·없다"),
           ("events_n", "T", "고유 사건 수"), ("events_n", "S", "고유 사건 수"),
           ("reprint_ratio", "T", "재보도 비율"), ("reprint_ratio", "S", "재보도 비율"),
           ("rd_authors", "T", "Reddit 작성자 수"), ("rd_authors", "S", "Reddit 작성자 수"),
           ("rd_new_share", "T", "Reddit 새 작성자 비율"), ("rd_new_share", "S", "Reddit 새 작성자 비율")]


def part_text_cv(t, folds):
    rows = []
    for name, tr, va, _, _ in folds:
        P = G.fit_table(tr)
        y = va.label.to_numpy(int)
        base = G.score(y, G.predict_table(va, P))
        for feat, form, lab in TESTS01:
            p, prm = B.fit_form(tr, va, feat, form, P)
            rows.append({"fold": name, "test": lab, "form": form, "param": prm, "delta": G.score(y, p) - base})
    r = pd.DataFrame(rows)
    out = []
    for (lab, form), g in r.groupby(["test", "form"], sort=False):
        g = g.set_index("fold")
        crit = g.delta.mean() >= 0.005 and (g.delta > 0).sum() >= 3 and g.delta["fold4"] >= 0
        out.append({"실험": lab, "형태": form, **{f: g.delta[f] for f in ["fold1", "fold2", "fold3", "fold4"]},
                    "평균": g.delta.mean(), "파라미터": " / ".join(str(x) for x in g.param),
                    "채택": bool(crit and g.delta.mean() > NULL95)})
    return pd.DataFrame(out)


def part_alias():
    """뉴스 기대 비교 후보: 채민 v6 기업명 규칙 vs Reddit v4 별칭(더 넓음). 세기만."""
    plan = json.loads(open(ROOT / "cache" / "llm_reddit_v4" / "plan.json", encoding="utf-8").read()) \
        if (ROOT / "cache" / "llm_reddit_v4" / "plan.json").exists() else None
    wide = plan["aliases"] if plan else None
    if wide is None:
        return None
    t, _ = E.load()
    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"], w["cutoff"] = w.date + pd.Timedelta(hours=16), w.target + pd.Timedelta(hours=9, minutes=30)
    con = duckdb.connect()
    con.register("w", w)
    con.register("u", pd.DataFrame({"symbol": sorted(t.symbol.unique())}))
    news = str(ROOT / "dataset" / "news.parquet")
    c = con.execute(f"""
        WITH n AS (SELECT symbols, title, known_at FROM read_parquet('{news}')
                   WHERE regexp_matches(title, '(?i)\\b(expect[a-z]*|estimat[a-z]*|consensus|forecast[a-z]*)\\b')),
             x AS (SELECT trim(tag) AS symbol, title, known_at FROM n, UNNEST(string_split(symbols, ',')) AS s(tag))
        SELECT DISTINCT w.target, x.symbol, x.title FROM x JOIN u USING (symbol)
        JOIN w ON x.known_at >= w.since AND x.known_at < w.cutoff""").df()
    con.close()
    c["title"] = c.title.map(lambda s: html.unescape(str(s)).strip())
    c = c.drop_duplicates(["target", "symbol", "title"])
    narrow = np.zeros(len(c), bool)
    widem = np.zeros(len(c), bool)
    for sym, idx in c.groupby("symbol").groups.items():
        rx_n = NG.name_regex(sym)
        names = sorted(set(wide.get(sym, [])) | set(NG.NAMES[sym]), key=len, reverse=True)
        rx_w = re.compile("|".join([r"(?i:(?<!\w)" + re.escape(x) + r"(?!\w))" for x in names] + [rx_n.pattern]))
        ti = c.loc[idx, "title"]
        narrow[c.index.get_indexer(idx)] = ti.map(lambda s: bool(rx_n.search(s))).to_numpy()
        widem[c.index.get_indexer(idx)] = ti.map(lambda s: bool(rx_w.search(s))).to_numpy()
    c["narrow"], c["wide"] = narrow, widem
    add = c[c.wide & ~c.narrow]
    return {"기대 표현 태그 기사": len(c), "채민 v6 규칙 통과": int(c.narrow.sum()), "넓힌 별칭 통과": int(c.wide.sum()),
            "늘어난 기사": len(add), "늘어난 종목·날짜": int(add.groupby(["symbol", "target"]).ngroups),
            "많이 늘어난 종목": add.symbol.value_counts().head(6).to_dict(),
            "예시": add.title.head(6).tolist()}


# ------------------------------------------------------------------ 그림
def fig_lgb(shap, perm, faith):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    s = shap.groupby("group")[["방향", "크기"]].mean().reindex(list(GROUPS))
    ax = axes[0]
    xx = np.arange(len(s))
    ax.bar(xx - 0.18, s.방향, 0.34, color=E.C1, label="방향 몫 (급상승 - 급하락)", edgecolor=E.SURF)
    ax.bar(xx + 0.18, s.크기, 0.34, color=E.C2, label="크기 몫 (보합 깎기)", edgecolor=E.SURF)
    ax.set_xticks(xx, s.index)
    ax.set_title("SHAP 묶음 합 (|logit| 평균, 4폴드)", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    p = perm.groupby(["unit", "rep"]).fall.mean().reset_index()
    units = [f"[묶음] {g}" for g in GROUPS] + ["[하나] gap", "[하나] gap_z", "[하나] gap_rel_z"]
    data = [p[p.unit == u].fall.to_numpy() for u in units]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5, medianprops={"color": E.INK})
    for b, u in zip(bp["boxes"], units):
        b.set(facecolor=E.C1 if u.startswith("[묶음]") else E.C3, edgecolor=E.SURF)
    ax.set_yticks(range(1, len(units) + 1), units, fontsize=8)
    ax.axvline(0, color=E.INK2, linewidth=0.8)
    ax.set_title("permutation (10회, 4폴드 평균 하락)", loc="left")
    ax = axes[2]
    for k, off in zip([1, 3, 5], [0, 1, 2]):
        r = faith[(faith.k == k) & (faith.kind == "무작위")].groupby("rep").fall.mean()
        top = faith[(faith.k == k) & (faith.kind == "상위")].fall.mean()
        ax.scatter(r, [off] * len(r), color=E.GRID, edgecolors=E.INK2, s=30, label="무작위 k개 빼기" if k == 1 else None)
        ax.scatter([top], [off], color=E.C2, s=60, marker="D", label="SHAP 상위 k개 빼기" if k == 1 else None)
    ax.set_yticks([0, 1, 2], ["k = 1", "k = 3", "k = 5"])
    ax.axvline(0, color=E.INK2, linewidth=0.8)
    ax.set_xlabel("다시 학습한 val 점수 하락 (4폴드 평균)")
    ax.set_title("faithfulness", loc="left")
    ax.set_ylim(-0.6, 2.9)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("LightGBM (B 피처 26개) — 묶음 중요도와 설명의 충실도", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=E.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "lgb.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ 실행
def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    rng = np.random.default_rng(SEED)
    t, _ = E.load()
    t["post"] = (1 + t.ret_pct / 100) / (1 + t.gap) - 1
    b2 = pd.read_parquet(ROOT / "cache" / "b_v2.parquet", columns=["date", "symbol"] + FEATS)
    chk = t.merge(b2[["date", "symbol", "gap", "vol20"]], on=["date", "symbol"], suffixes=("", "_b2"))
    print(f"b_v2 와 같은 값인지: gap 최대 차이 {(chk.gap - chk.gap_b2).abs().max():.2e}, "
          f"vol20 최대 차이 {(chk.vol20 - chk.vol20_b2).abs().max():.2e}")
    t = t.merge(b2.drop(columns=[c for c in t.columns if c in b2 and c not in ("date", "symbol")]),
                on=["date", "symbol"], how="left")
    folds = NG.folds_of(t)

    if only in ("all", "lgb"):
        shap, perm, faith = part_lgb(folds, rng)
        shap.to_csv(OUT / "5_shap_groups.csv", index=False, encoding="utf-8-sig")
        perm.to_csv(OUT / "5_perm.csv", index=False, encoding="utf-8-sig")
        faith.to_csv(OUT / "6_faith.csv", index=False, encoding="utf-8-sig")
        fig_lgb(shap, perm, faith)
        print("\n[02-5] SHAP 묶음 (4폴드 평균)\n", shap.groupby("group")[["방향", "크기", "방향_개별합"]].mean().round(4))
        print(perm.groupby(["unit", "rep"]).fall.mean().groupby("unit").agg(["mean", "std"]).round(4))
        print("\n[02-6] faithfulness\n", faith.groupby(["kind", "k"]).fall.agg(["mean", "min", "max"]).round(4))
        print(faith[faith.kind == "상위"][["fold", "k", "removed", "fall"]].round(4).to_string(index=False))
    if only == "lgb":
        return

    if only == "text":
        return text_main(t)
    m, tab, p = part_earn(t)
    tab.to_csv(OUT / "8_earn.csv", index=False, encoding="utf-8-sig")
    print("\n[02-8] 실적 밤\n", tab.round(3).to_string(index=False), f"\n유지율 차이 Fisher p = {p:.3f}")

    a, per, piv, cor_all, cor_u, prop, pc = part_unseen(t, folds)
    a.to_csv(OUT / "12_inout.csv", index=False, encoding="utf-8-sig")
    prop.to_csv(OUT / "12_props.csv", encoding="utf-8-sig")
    print("\n[02-12] (a) 처음 보는 10종목 점수\n", a.round(4).to_string(index=False))
    print(f"(b) 종목 점수 순위 폴드 간 상관: 50종목 평균 {cor_all[0]:.2f} [{cor_all[1]:.2f}, {cor_all[2]:.2f}], "
          f"처음 보는 10종목 {cor_u[0]:.2f} [{cor_u[1]:.2f}, {cor_u[2]:.2f}]")
    print("(c) 종목 성질과 4폴드 평균 점수의 순위상관\n", pc.round(2).to_string())
    print(prop[prop["처음 보는 종목"]].sort_values("4폴드 평균 점수").round(3).to_string())

    text_main(t)


def text_main(t):
    t2, ev = text_features(t)
    folds2 = NG.folds_of(t2)
    j = part_text_cv(t2, folds2)
    j.to_csv(OUT / "01_text_cv.csv", index=False, encoding="utf-8-sig")
    print("\n[01] 재보도: 제목→고유 사건", f"{ev.n_titles.sum():,} → {ev.n_events.sum():,}",
          f"(재보도 비율 평균 {ev.reprint_ratio.mean():.3f})")
    print(j.round(4).to_string(index=False))

    al = part_alias()
    (OUT / "01_alias.json").write_text(json.dumps(al, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n[01] 별칭 넓히기\n", json.dumps(al, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
