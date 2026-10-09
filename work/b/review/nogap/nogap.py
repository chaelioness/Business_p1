"""갭 없이 맞히면? — 시간외 가격 피처를 전부 뺀 금융·텍스트 데이터로 여러 구성을 LightGBM 으로.

    uv run python work/b/review/nogap/nogap.py

사전 등록 (실행 전에 고정)
--------------------------
- 모델: A lgb.py 와 같은 LightGBM 절차 (파라미터, train 마지막 40일 조기 종료, 공격성 k 디코딩). 처음 보는 10종목은 학습에서 뺌.
- "갭 제외" = 기준일 16:00 이후 시간외 가격에서 나온 피처를 모두 뺌 (EXT 목록). 일봉·정규장·실적·애널리스트·달력은 남김.
- 구성 (피처 묶음)
    R   갭 규칙 (제출본, 비교 기준)
    A1  B 26개 (A 와 같음, 갭 포함)                     ← 재현 확인
    F   금융 (갭 제외): 수익률·추세·변동성·정규장·시장 + 실적·애널리스트·달력
    F0  금융 (갭 제외) 중 실적·애널리스트 빼고 가격만
    E   실적·애널리스트만
    T   텍스트만 (원본 뉴스 수·논조·출처·새로움·09:00~09:30 뉴스, 채민 LLM 뉴스, 채민 Reddit)
    FT  금융 (갭 제외) + 텍스트
    GT  갭 묶음 + 텍스트
    ALL 금융 + 갭 묶음 + 텍스트
- 지표: score (4폴드 + 처음 보는 종목), 급등락 포착률 (정답 급등락 중 급등락으로 찍은 비율),
        방향 적중 (보합 아닌 예측·정답에서 부호가 같은 비율)
- 비교 기준선: 무작위 예측의 score 는 0 ± 0.03 (A 발표 11장)
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in ["work/b/features", "work/b/review/explain", "work/b/review/news_gap", "work/b/review/batch",
          "work/b/review/rest"]:
    sys.path.insert(0, str(ROOT / p))
import gapz_rule as G  # noqa: E402
import explain_rule as E  # noqa: E402
import news_gap as NG  # noqa: E402
import batch as B  # noqa: E402
import rest as RS  # noqa: E402

OUT, FIGS = HERE / "out", HERE / "figs"

GAP = ["gap", "gap_z", "abs_gap_z", "gap_rel_z", "mkt_gap", "mkt_gap_z", "pre_move", "post_ret", "post_n", "pre_n",
       "ext_n", "ext_range_z", "mkt_absgap", "gap_disp", "gap_onz", "gap_rank", "pre_late", "ext_pos"]
PRICE = ["ret1", "ret5", "ret20", "ret1_z", "ret5_z", "ret20_z", "vol5", "vol20", "vol60", "vol_5_20", "vol_20_60",
         "atr14", "range1", "range1_z", "gap_open1", "intraday1", "volu_ratio", "dist_ma20_z", "dist_ma50_z", "rsi14",
         "bb_pctb", "macd_hist", "bigfreq60", "bigfreq20", "streak", "pos60", "mkt_ret1", "mkt_ret5", "mkt_vol20",
         "rel_ret1_z", "rel_ret5_z", "beta60", "last_hour_z", "absret1_z", "big_yday", "gap_days", "dow"]
EARN = ["earn_now", "earn_surprise_now", "earn_prev", "days_since_earn", "last_surprise", "surprise_mean4",
        "earn_hist_absr", "earn_timing", "an_n30", "an_up30", "an_down30", "an_raise30", "an_lower30", "an_tgtchg30",
        "an_tgt_gap", "an_now_n", "an_now_net"]
TEXT = ["n_tag", "n_name", "f_llm_rel", "f_llm_cmp", "tone_mean", "tone_sd", "src_n", "novelty", "late_n", "late_tone",
        "llm_dir", "llm_guid", "llm_legal", "llm_ma", "llm_unsure", "rd_dir", "rd_cnt", "rd_question", "rd_unsure"]
CONFIGS = {
    "A1  B 26개 (갭 포함, A 재현)": RS.FEATS,
    "F   금융 (갭 제외) 전체": PRICE + EARN,
    "F0  금융 (갭 제외) 가격만": PRICE,
    "E   실적·애널리스트만": EARN,
    "T   텍스트만": TEXT,
    "FT  금융 (갭 제외) + 텍스트": PRICE + EARN + TEXT,
    "GT  갭 묶음 + 텍스트": GAP + TEXT,
    "ALL 금융 + 갭 + 텍스트": PRICE + EARN + GAP + TEXT,
}


def metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, int)
    big = np.isin(y, [0, 4])
    nz = (y != 2) & (p != 2)
    return {"score": G.score(y, p), "급등락_포착": float(np.isin(p[big], [0, 4]).mean()),
            "방향_적중": float((np.sign(y[nz] - 2) == np.sign(p[nz] - 2)).mean()),
            "급등락_예측비율": float(np.isin(p, [0, 4]).mean())}


def main():
    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    t, _ = E.load()
    t, _ = NG.build(t)
    t = B.build(t)
    b2 = pd.read_parquet(ROOT / "cache" / "b_v2.parquet")
    t = t.merge(b2.drop(columns=[c for c in b2 if c in t and c not in ("date", "symbol")]), on=["date", "symbol"], how="left")
    miss = [c for f in CONFIGS.values() for c in f if c not in t]
    assert not miss, miss
    assert not set(GAP) & set(PRICE + EARN + TEXT)
    folds = NG.folds_of(t)
    rows = []
    for name, tr, va, _, _ in folds:
        y, u = va.label.to_numpy(int), va.unseen.to_numpy()
        p = G.predict_table(va, G.fit_table(tr))
        rows.append({"fold": name, "config": "R   갭 규칙 (제출본)", "n_feat": 2, **metrics(y, p),
                     "unseen": G.score(y[u], p[u])})
        for cname, feats in CONFIGS.items():
            m, th = RS.lgb_fit(tr, list(feats))
            p = RS.lgb_pred(m, th, va, list(feats))
            rows.append({"fold": name, "config": cname, "n_feat": len(feats), **metrics(y, p),
                         "unseen": G.score(y[u], p[u])})
        print(f"  {name} 끝", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "configs.csv", index=False, encoding="utf-8-sig")
    s = r.pivot_table(index="config", columns="fold", values="score")
    s["평균"] = s.mean(axis=1)
    extra = r.groupby("config")[["unseen", "급등락_포착", "방향_적중", "급등락_예측비율", "n_feat"]].mean()
    tab = s.join(extra).sort_values("평균", ascending=False)
    tab.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print(tab.round(3).to_string())

    order = list(tab.index)[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    gapcfg = {"R   갭 규칙 (제출본)", "A1  B 26개 (갭 포함, A 재현)", "GT  갭 묶음 + 텍스트", "ALL 금융 + 갭 + 텍스트"}
    for i, c in enumerate(order):
        rr = r[r.config == c]
        col = E.C1 if c in gapcfg else E.C2
        ax.hlines(i, rr.score.min(), rr.score.max(), color=col, alpha=0.35, linewidth=3)
        ax.scatter([tab.loc[c, "평균"]], [i], color=col, s=48, zorder=3, edgecolors=E.SURF)
        ax.annotate(f"{tab.loc[c, '평균']:.3f}", (tab.loc[c, "평균"], i), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=E.INK)
    ax.axvspan(-0.03, 0.03, color=E.GRID, alpha=0.9, linewidth=0, label="무작위 예측 범위 (±0.03)")
    ax.axvline(0, color=E.INK2, linewidth=0.8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("val score (점 = 4폴드 평균, 선 = 폴드별 최소~최대)")
    ax.scatter([], [], color=E.C1, label="갭 정보 있음")
    ax.scatter([], [], color=E.C2, label="갭 정보 없음")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("갭 없이 맞히면? — 구성별 LightGBM", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "configs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
