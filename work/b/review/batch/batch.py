"""다양한 실험 배치 — 처음 시도하는 피처 31개를 같은 틀로 제출 규칙에 붙여 보고, 가짜 피처 20개로 우연 기준선을 만든다.

    uv run python work/b/review/batch/batch.py

사전 등록 (실행 전에 고정, 결과를 보고 바꾸지 않음)
----------------------------------------------------
틀: 제출 규칙(gap_z + ext_range_z)에 피처 하나를 붙임. 파라미터는 폴드마다 train 으로만 (동점이면 0 에 가까운 값).
    처음 보는 10종목은 학습에서 뺌. A 폴드 4개.
형태
  T  갭 신뢰도   s = s0 · exp(γ·z(f)),            γ ∈ GAMMAS
  S  급등락 경계 |s0| ≥ b·exp(−δ·z(f)) 이면 급등락, δ ∈ DELTAS
  D  방향 더하기 s = s0 + β·sd(s0)·z(f),           β ∈ BETAS     (방향이 있는 피처만)
  R  ext_range_z 를 다른 범위로 바꿔 규칙 전체를 다시 학습
  L  λ 를 0.2 로 고정
  (s0 = 기본 규칙 점수, z = train 기준 표준화, 결측은 0 = 평균)

피처 (모두 대상일 09:00 전 정보. N5·N6 만 09:00~09:30, cutoff 09:30 이전이라 사용 가능)
  가격  P1 rng_close   봉 종가들의 최고 − 최저 (고가·저가 대신)                         R
        P2 rng_ex16    16시 봉을 뺀 고가·저가                                         R
        P3 gap_age_h   마지막 시간외 봉이 09:00 보다 몇 시간 이른가                      T S
        P4 lam_fixed   λ = 0.2 고정 (재은 할 일)                                       L
  뉴스  N1 tone_mean   밤사이 태그 기사 논조 평균 (원본 tone)                             D
        N2 tone_sd     논조 표준편차 (기사 2개 이상, w4-2 p44 의견 분산)                   T S
        N3 src_n       출처 수 log(1+n)                                               T S
        N4 novelty     최근 5거래일에 없던 제목 비율 (w5-1 p22 새로움)                     T S
        N5 late_n      09:00~09:30 기사 수 log(1+n) (갭 가격 뒤에 나온 뉴스)             T S
        N6 late_tone   09:00~09:30 기사 논조 평균                                      D
  LLM   L1 llm_dir     뉴스 above − below 비율                                         D
        L2 llm_guid    기업 전망(가이던스) 사건 비율                                    T S
        L3 llm_legal   규제·법적 사건 비율                                             T S
        L4 llm_ma      인수합병 사건 비율                                              T S
        L5 llm_unsure  의혹 + 가능성 상태 비율                                         T S
  Reddit R1 rd_dir     bullish − bearish 비율 (채민 v4)                                D
        R2 rd_cnt      후보 글 수 log(1+n)                                             T S
        R3 rd_question 질문 글 비율                                                    T S
        R4 rd_unsure   불확실성 표현 비율                                              T S
  → 시험 31개 (D 4 + T·S 12×2 + R 2 + L 1)

판정
  - 채택 기준 (B·A 공통): 4폴드 평균 +0.005 이상, 3폴드 이상 상승, fold4 하락 없음
  - 그리고 우연 기준선: 가짜 피처(진짜 피처를 같은 날 안에서 섞은 것) 20개를 같은 형태로 돌린 결과의
    평균 차이 95 분위수보다 커야 함
  - 방향 피처(D)는 별도로 train 기간 "09:00 이후 움직임과 부호 일치율"도 봄 (판정에는 안 씀)
"""

import html
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "work" / "b" / "features"))
sys.path.insert(0, str(ROOT / "work" / "b" / "review" / "explain"))
sys.path.insert(0, str(ROOT / "work" / "b" / "review" / "news_gap"))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"
SEED, N_FAKE = 2026, 20
GAMMAS = NG.GAMMAS
DELTAS = NG.DELTAS
BETAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
DISC_END = pd.Timestamp("2026-02-12")

TESTS = [  # (이름, 묶음, 형태들)
    ("rng_close", "가격", ["R"]), ("rng_ex16", "가격", ["R"]), ("gap_age_h", "가격", ["T", "S"]),
    ("lam_fixed", "가격", ["L"]),
    ("tone_mean", "뉴스", ["D"]), ("tone_sd", "뉴스", ["T", "S"]), ("src_n", "뉴스", ["T", "S"]),
    ("novelty", "뉴스", ["T", "S"]), ("late_n", "뉴스", ["T", "S"]), ("late_tone", "뉴스", ["D"]),
    ("llm_dir", "LLM 뉴스", ["D"]), ("llm_guid", "LLM 뉴스", ["T", "S"]), ("llm_legal", "LLM 뉴스", ["T", "S"]),
    ("llm_ma", "LLM 뉴스", ["T", "S"]), ("llm_unsure", "LLM 뉴스", ["T", "S"]),
    ("rd_dir", "Reddit", ["D"]), ("rd_cnt", "Reddit", ["T", "S"]), ("rd_question", "Reddit", ["T", "S"]),
    ("rd_unsure", "Reddit", ["T", "S"]),
]
LABEL = {
    "rng_close": "시간외 범위: 봉 종가 기준", "rng_ex16": "시간외 범위: 16시 봉 제외", "gap_age_h": "갭 가격 오래됨",
    "lam_fixed": "λ 0.2 고정", "tone_mean": "뉴스 논조 평균", "tone_sd": "뉴스 논조 흩어짐", "src_n": "뉴스 출처 수",
    "novelty": "새 제목 비율", "late_n": "09:00~09:30 기사 수", "late_tone": "09:00~09:30 기사 논조",
    "llm_dir": "LLM 상회 - 하회", "llm_guid": "LLM 가이던스 사건", "llm_legal": "LLM 규제·법적 사건",
    "llm_ma": "LLM 인수합병", "llm_unsure": "LLM 의혹·가능성", "rd_dir": "Reddit 상승 - 하락 의견",
    "rd_cnt": "Reddit 글 수", "rd_question": "Reddit 질문 비율", "rd_unsure": "Reddit 불확실성 표현",
}


# ------------------------------------------------------------------ 피처 만들기
def build(t):
    w = t[["date", "target"]].drop_duplicates().dropna().copy()
    w["since"] = w.date + pd.Timedelta(hours=16)
    w["cut9"] = w.target + pd.Timedelta(hours=9)
    w["cutoff"] = w.target + pd.Timedelta(hours=9, minutes=30)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("w", w)
    con.register("u", pd.DataFrame({"symbol": sorted(t.symbol.unique())}))

    # 가격: build_gapz 와 같은 창 (기준일 16:00 이후 post/pre 봉, known_at < cutoff)
    price, daily = str(ROOT / "dataset" / "price.parquet"), str(ROOT / "dataset" / "daily.parquet")
    p = con.execute(f"""
        SELECT w.date, p.symbol, p.datetime, p.high, p.low, p.close
        FROM read_parquet('{price}') p JOIN u USING (symbol)
        JOIN w ON p.datetime >= w.since AND p.known_at < w.cutoff
        WHERE p.session IN ('post', 'pre')""").df()
    dc = con.execute(f"SELECT symbol, date_et AS date, close AS prev_close FROM read_parquet('{daily}')").df()
    p = p.sort_values("datetime")
    g = p.groupby(["date", "symbol"])
    px = pd.DataFrame({"hi": g.high.max(), "lo": g.low.min(), "cmax": g.close.max(), "cmin": g.close.min(),
                       "last": g.datetime.max()}).reset_index()
    ex = p[p.datetime.dt.hour != 16].groupby(["date", "symbol"]).agg(hi16=("high", "max"), lo16=("low", "min"))
    px = px.merge(ex.reset_index(), on=["date", "symbol"], how="left").merge(dc, on=["date", "symbol"], how="left")

    # 뉴스 원본: 기준일 16:00 ~ 대상일 09:30, 09:00 기준으로 앞·뒤 나눔
    news = str(ROOT / "dataset" / "news.parquet")
    lo, hi = str(w.since.min()), str(w.cutoff.max())
    n = con.execute(f"""
        WITH n AS (SELECT symbols, title, source, tone, known_at FROM read_parquet('{news}')
                   WHERE known_at >= '{lo}' AND known_at < '{hi}' AND title IS NOT NULL),
             x AS (SELECT trim(tag) AS symbol, title, source, tone, known_at
                   FROM n, UNNEST(string_split(symbols, ',')) AS s(tag))
        SELECT w.date, w.cut9, x.symbol, x.title, x.source, x.tone, x.known_at FROM x JOIN u USING (symbol)
        JOIN w ON x.known_at >= w.since AND x.known_at < w.cutoff""").df()
    con.close()
    n["key"] = n.title.map(lambda s: html.unescape(str(s)).strip()).str.replace(r"\s+", " ", regex=True).str.casefold()
    n = n.drop_duplicates(["date", "symbol", "key"])
    pre, late = n[n.known_at < n.cut9], n[n.known_at >= n.cut9]
    gp = pre.groupby(["date", "symbol"])
    nf = pd.DataFrame({"tone_mean": gp.tone.mean(), "tone_sd": gp.tone.std(), "src_n": np.log1p(gp.source.nunique())})
    gl = late.groupby(["date", "symbol"])
    nf = nf.join(pd.DataFrame({"late_n": np.log1p(gl.size()), "late_tone": gl.tone.mean()}), how="outer").reset_index()

    # 새로움: 같은 종목의 직전 5거래일 창에 없던 제목 비율
    dates = sorted(w.date)
    pos = {d: i for i, d in enumerate(dates)}
    keys = pre.groupby(["symbol", "date"]).key.apply(set).to_dict()
    nov = []
    for (sym, d), ks in keys.items():
        i = pos[d]
        seen = set().union(*[keys.get((sym, dd), set()) for dd in dates[max(0, i - 5):i]])
        nov.append((d, sym, np.mean([k not in seen for k in ks])))
    nf = nf.merge(pd.DataFrame(nov, columns=["date", "symbol", "novelty"]), on=["date", "symbol"], how="left")

    # 채민 LLM (뉴스 공백 보정본, Reddit v4). 피처가 없으면 0 (= 그런 기사·글 없음)
    ln = pd.read_csv(ROOT / "cache" / "llm_news_v6" / "daily_features.csv")
    ln["target"] = pd.to_datetime(ln.target, format="mixed")
    ln = ln.assign(llm_dir=ln.feature_surprise_above_share - ln.feature_surprise_below_share,
                   llm_guid=ln.feature_event_company_guidance_share,
                   llm_legal=ln.feature_event_regulation_legal_share,
                   llm_ma=ln.feature_event_merger_acquisition_share,
                   llm_unsure=ln.feature_state_alleged_share + ln.feature_state_proposed_possible_share)
    rd = pd.read_csv(ROOT / "cache" / "llm_reddit_v4" / "daily_features.csv")
    rd["target"] = pd.to_datetime(rd.target, format="mixed")
    rd = rd.assign(rd_dir=rd.feature_stance_bullish_ratio - rd.feature_stance_bearish_ratio,
                   rd_cnt=np.log1p(rd.candidate_count), rd_question=rd.feature_content_type_question_ratio,
                   rd_unsure=rd.feature_uncertainty_explicit_ratio)

    t = t.merge(px, on=["date", "symbol"], how="left").merge(nf, on=["date", "symbol"], how="left")
    t = t.merge(ln[["symbol", "target", "llm_dir", "llm_guid", "llm_legal", "llm_ma", "llm_unsure"]],
                on=["symbol", "target"], how="left")
    t = t.merge(rd[["symbol", "target", "rd_dir", "rd_cnt", "rd_question", "rd_unsure"]],
                on=["symbol", "target"], how="left")
    t["ext_check"] = (t.hi - t.lo) / t.prev_close / t.vol20
    t["rng_close"] = (t.cmax - t.cmin) / t.prev_close / t.vol20
    t["rng_ex16"] = ((t.hi16 - t.lo16) / t.prev_close / t.vol20).fillna(t.rng_close)
    t["gap_age_h"] = ((t.target + pd.Timedelta(hours=8)) - t["last"]).dt.total_seconds() / 3600   # 08:00 봉이면 0
    t["post"] = (1 + t.ret_pct / 100) / (1 + t.gap) - 1
    return t


# ------------------------------------------------------------------ 형태
def _s0(x, P):
    return NG._s0(x, P)


def fit_form(tr, va, feat, form, P):
    if form in ("T", "S"):
        return NG.fit_variant(tr, va, feat, form, P)
    if form == "D":
        y = tr.label.to_numpy(int)
        s_tr, s_va = _s0(tr, P), _s0(va, P)
        f_tr = tr[feat].to_numpy(float)
        z_tr, z_va = NG._z(f_tr, f_tr), NG._z(f_tr, va[feat].to_numpy(float))
        sd = float(np.nanstd(s_tr))
        best = max(((*G._fit_ab(s_tr + b * sd * z_tr, y), b) for b in BETAS),
                   key=lambda r: (round(r[0], 10), -abs(r[3])))
        _, a, bb, beta = best
        return G.cut(s_va + beta * sd * z_va, a, bb), beta
    if form == "R":
        P2 = G.fit_table(tr.assign(ext_range_z=tr[feat]))
        return G.predict_table(va.assign(ext_range_z=va[feat]), P2), P2["lam"]
    if form == "L":
        P2 = E.fit_fixed(tr, 0.2)
        return G.predict_table(va, P2), 0.2


def run(t, folds, rng):
    rows = []
    fake_src = [n for n, _, forms in TESTS if forms[0] in ("T", "S", "D")]
    fakes = [(f"가짜{k:02d}", fake_src[k % len(fake_src)]) for k in range(N_FAKE)]
    for name, tr, va, _, _ in folds:
        P = G.fit_table(tr)
        yb, u = va.label.to_numpy(int), va.unseen.to_numpy()
        base = G.predict_table(va, P)
        rows.append({"fold": name, "test": "기본 규칙", "group": "-", "form": "-", "param": P["lam"],
                     "score": G.score(yb, base), "unseen": G.score(yb[u], base[u])})
        for feat, grp, forms in TESTS:
            for form in forms:
                p, prm = fit_form(tr, va, feat, form, P)
                rows.append({"fold": name, "test": feat, "group": grp, "form": form, "param": prm,
                             "score": G.score(yb, p), "unseen": G.score(yb[u], p[u])})
        for fk, src in fakes:
            forms = dict((n, f) for n, _, f in TESTS)[src]
            trf, vaf = tr.copy(), va.copy()
            for df in (trf, vaf):
                df[fk] = df.groupby("date")[src].transform(lambda s: rng.permutation(s.to_numpy()))
            for form in forms:
                p, prm = fit_form(trf, vaf, fk, form, P)
                rows.append({"fold": name, "test": fk, "group": "가짜", "form": form, "param": prm,
                             "score": G.score(yb, p), "unseen": G.score(yb[u], p[u])})
    r = pd.DataFrame(rows)
    base = r[r.test == "기본 규칙"].set_index("fold").score
    r["delta"] = r.score - r.fold.map(base)
    return r


def judge(r):
    fk = r[r.group == "가짜"].groupby(["test", "form"]).delta.mean()
    null95 = float(np.quantile(fk, 0.95))
    out = []
    for (test, form), g in r[(r.group != "가짜") & (r.test != "기본 규칙")].groupby(["test", "form"], sort=False):
        g = g.set_index("fold")
        crit = g.delta.mean() >= 0.005 and (g.delta > 0).sum() >= 3 and g.delta["fold4"] >= 0
        out.append({"test": test, "이름": LABEL[test], "group": g.group.iloc[0], "form": form,
                    **{f: g.delta[f] for f in ["fold1", "fold2", "fold3", "fold4"]},
                    "평균": g.delta.mean(), "상승폴드": int((g.delta > 0).sum()),
                    "unseen_평균": (g.unseen - r[r.test == "기본 규칙"].set_index("fold").unseen).mean(),
                    "파라미터": " / ".join(str(x) for x in g.param),
                    "기준통과": bool(crit), "가짜95초과": bool(g.delta.mean() > null95),
                    "채택": bool(crit and g.delta.mean() > null95)})
    j = pd.DataFrame(out)
    nulls = pd.DataFrame({"평균": fk.values}).assign(form=[f for _, f in fk.index])
    return j, nulls, null95


def direction_check(t):
    d = t[(t.target <= DISC_END) & (t.post != 0)]
    rows = []
    for f in ["tone_mean", "late_tone", "llm_dir", "rd_dir"]:
        x = d[d[f].notna() & (d[f] != 0)]
        hit = (np.sign(x[f]) == np.sign(x.post)).mean()
        rows.append({"피처": LABEL[f], "판단 있는 행": len(x), "09시 이후 부호 일치": hit,
                     "갭과 같은 방향": (np.sign(x[f]) == np.sign(x.gap)).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 그림
COL = {"가격": E.C1, "뉴스": E.C2, "LLM 뉴스": E.C3, "Reddit": "#4a3aa7"}


def fig(j, nulls, null95):
    j = j.copy()
    j["row"] = j.이름 + " · " + j.form
    j = j.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 9.2))
    ax.axvspan(nulls.평균.min(), null95, color=E.GRID, alpha=0.9, linewidth=0, label="가짜 피처 20개 (최소 ~ 95 분위수)")
    ax.axvline(0, color=E.INK2, linewidth=0.8)
    ax.axvline(0.005, color=E.INK2, linewidth=0.8, linestyle=":")
    for grp, c in COL.items():
        s = j[j.group == grp]
        ax.scatter(s.평균, s.index, color=c, s=36, zorder=3, label=grp, edgecolors=E.SURF, linewidths=1)
    for i, rr in j.iterrows():
        ax.hlines(i, min(rr[["fold1", "fold2", "fold3", "fold4"]]), max(rr[["fold1", "fold2", "fold3", "fold4"]]),
                  color=COL[rr.group], alpha=0.35, linewidth=2)
    ax.set_yticks(j.index, j.row, fontsize=8)
    ax.set_xlabel("기본 규칙 대비 val 점수 차이 (점 = 4폴드 평균, 선 = 폴드별 최소~최대)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("배치 실험 31개 — 점선 = 채택 기준 +0.005", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "batch.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ 실행
def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    t, _ = E.load()
    t = build(t)
    d = (t.ext_check - t.ext_range_z).abs()
    print(f"ext_range_z 재계산 확인: 최대 차이 {d.max():.2e} (행 {int(d.notna().sum())})")
    print("결측 비율", t[[n for n, _, f in TESTS if f[0] != "L"]].isna().mean().round(3).to_dict())
    folds = NG.folds_of(t)
    r = run(t, folds, np.random.default_rng(SEED))
    r.to_csv(OUT / "cv.csv", index=False, encoding="utf-8-sig")
    j, nulls, null95 = judge(r)
    j.to_csv(OUT / "judge.csv", index=False, encoding="utf-8-sig")
    fig(j, nulls, null95)
    dc = direction_check(t)
    dc.to_csv(OUT / "direction.csv", index=False, encoding="utf-8-sig")
    print(f"\n가짜 피처 평균 차이: 최소 {nulls.평균.min():+.4f}, 95분위 {null95:+.4f}, 최대 {nulls.평균.max():+.4f}, "
          f"채택 기준 통과 {int(((nulls.평균 >= 0.005)).sum())}/{len(nulls)}")
    print(j.drop(columns=["test"]).round(4).to_string(index=False))
    print("\n방향 피처 (train)\n", dc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
