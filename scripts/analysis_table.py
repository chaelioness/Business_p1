"""실험 검증표: W&B 점수 + 코드에서 확인한 '바꾼 것 하나' + 폴드별 학습된 파라미터.
    uv run python scripts/analysis_table.py
입력: docs/presentation/experiment_table_raw.csv (analysis_runs.py), cache/analysis/params.json (analysis_params.py)
출력: docs/presentation/experiment_table.csv, experiment_table.md, params_table.md
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "presentation"
AN = ROOT / "cache" / "analysis"

# (묶음, 비교 대상, 바꾼 것 하나, 근거 코드)
CHANGE = {
    "random_uniform": ("기준선", "-", "5등급 균등 무작위", "scripts/log_baselines.py Random()"),
    "random_prior": ("기준선", "random_uniform", "무작위 비율을 train 정답 비율로", "log_baselines.fit_random_prior"),
    "flat": ("기준선", "-", "전부 보합", "lab/baselines.fit_flat"),
    "yesterday": ("기준선", "-", "어제 등락 label 그대로", "lab/baselines.fit_yesterday"),
    "rule_pre_move": ("기준선", "flat", "피처: 프리마켓 안 등락(pre_move), |s| 경계 train", "eda/cv_check_ovn_gap.fit_pre_gap"),
    "rule_ovn_gap": ("기준선", "rule_pre_move", "피처: 전일 종가 기준 갭 (A)", "eda/cv_check_ovn_gap.fit_ovn_gap"),
    "rule_gapz": ("규칙", "rule_ovn_gap", "피처: 갭 ÷ vol20 (B)", "work/a/rules.fit_gapz (λ=0)"),
    "rule_gapz_ext": ("규칙", "rule_gapz", "피처: × exp(−λ·ext_range_z 표준화), λ∈{0..0.5} train (B 최종 = 제출 v1)", "work/a/rules.fit_gapz_ext"),
    "lgb_v1_argmax": ("LightGBM", "rule_gapz_ext", "모델: LightGBM 다항, B 피처 26개, 디코딩 argmax", "work/a/lgb.py"),
    "lgb_v1_match_train": ("LightGBM", "lgb_v1_argmax", "디코딩: 기대 등급을 train 정답 비율 분위로", "lgb.thresholds(e_tr, q_tr)"),
    "lgb_v1_match_recent": ("LightGBM", "lgb_v1_match_train", "디코딩: 비율을 train 마지막 60일 정답 비율로", "lgb q_recent"),
    "lgb_v1_aggr": ("LightGBM", "lgb_v1_match_train", "디코딩: 급등락 비율 ×k, 보합 ÷k (k 는 train 마지막 40일)", "lgb.aggr_q"),
    "ablation_struct_aggr": ("ablation", "lgb_v1_aggr", "피처: + text_overnight_hours (C)", "scripts/a_ablation_text.py"),
    "ablation_struct+news_aggr": ("ablation", "ablation_struct_aggr", "피처: + C 뉴스 2개", "a_ablation_text GROUPS"),
    "ablation_struct+reddit_aggr": ("ablation", "ablation_struct_aggr", "피처: + C Reddit 3개", "a_ablation_text GROUPS"),
    "ablation_struct+text_all_aggr": ("ablation", "ablation_struct_aggr", "피처: + C 텍스트 7개 전부", "a_ablation_text GROUPS"),
    "v2_rule_base": ("v2 규칙", "rule_gapz_ext", "λ 후보만 {0..0.4} 로 (models2.LAMS) — 재구현 기준", "models2.fit_rule feats=[]"),
    "v2_rule_lex_event": ("v2 규칙", "v2_rule_base", "갭 신뢰도 γ·z(사건 단어 수)", "models2.fit_rule"),
    "v2_rule_lex_n": ("v2 규칙", "v2_rule_base", "갭 신뢰도 γ·z(기사 제목 수)", "models2.fit_rule"),
    "v2_rule_news_c": ("v2 규칙", "v2_rule_base", "갭 신뢰도 γ·z(C 장 마감 후 기사 수)", "models2.fit_rule"),
    "v2_rule_text_big": ("v2 규칙", "v2_rule_base", "갭 신뢰도 γ·z(TF-IDF 급등락 확률)", "models2.fit_rule + add_text"),
    "v2_ridge_core": ("v2 선형", "v2_rule_base", "모델: Ridge(제곱 손실), 핵심 9개 → 수익률/vol20", "models2._lin ridge"),
    "v2_huber_core": ("v2 선형", "v2_ridge_core", "손실: 제곱 → Huber", "models2._lin huber"),
    "v2_huber_core_text": ("v2 선형", "v2_huber_core", "피처: + 사전 단어 6 + C 텍스트 3", "a_v2 LEXT+CT"),
    "v2_logit_core": ("v2 선형", "v2_ridge_core", "손실: 다항 로지스틱 → 기대 등급−2", "models2._lin logit"),
    "v2_lgb_l2_b26": ("v2 트리", "lgb_v1_argmax", "목표: 등급 → 수익률/vol20 회귀(L2), gap_z 단조, 디코딩 |s| 경계", "models2._tree lgb_l2"),
    "v2_lgb_huber_b26": ("v2 트리", "v2_lgb_l2_b26", "손실: L2 → Huber", "models2._tree lgb_huber"),
    "v2_lgb_l1_b26": ("v2 트리", "v2_lgb_l2_b26", "손실: L2 → L1 (단조 제약 없음)", "models2._tree lgb_l1"),
    "v2_lgb_ord_b26": ("v2 트리", "v2_lgb_l2_b26", "목표: label−2 회귀 (순서형 근사)", "models2._tree lgb_ord"),
    "v2_lgb_wmulti_b26": ("v2 트리", "lgb_v1_argmax", "손실: 다항 + 행 가중(급등락 4, 보합 0.25), 디코딩 |s| 경계", "models2.CLASS_W"),
    "v2_hgb_l1_b26": ("v2 트리", "v2_lgb_l1_b26", "모델: sklearn HistGB (L1)", "models2._tree hgb_l1"),
    "v2_lgb_huber_b26_lex": ("v2 트리", "v2_lgb_huber_b26", "피처: + 사전 단어 6 + C 텍스트 3", "a_v2 EXPS"),
    "v2_lgb_huber_b26_lex_tfidf": ("v2 트리", "v2_lgb_huber_b26_lex", "피처: + TF-IDF 점수 2", "a_v2 EXPS"),
    "v2_ens_rule_huber_core": ("v2 앙상블", "v2_rule_base", "s_rule + w·s_huber(핵심 9)", "models2.fit_ens"),
    "v2_ens_rule_huber_text": ("v2 앙상블", "v2_ens_rule_huber_core", "Huber 피처 + 텍스트 9", "models2.fit_ens"),
    "v2_ens_rule_huber_tfidf": ("v2 앙상블", "v2_ens_rule_huber_text", "Huber 피처 + TF-IDF 2", "models2.fit_ens"),
    "v2_rsize_earn_now": ("v2 규칙", "v2_rule_base", "급등락 경계 b·exp(−δ·z(실적 발표))", "models2.fit_rule_size"),
    "v2_rsize_lex_event": ("v2 규칙", "v2_rule_base", "급등락 경계 ← 사건 단어 수", "models2.fit_rule_size"),
    "v2_rsize_lex_earn": ("v2 규칙", "v2_rule_base", "급등락 경계 ← 실적 단어 수", "models2.fit_rule_size"),
    "v2_rsize_news_c": ("v2 규칙", "v2_rule_base", "급등락 경계 ← C 기사 수", "models2.fit_rule_size"),
    "v2_rsize_text_big": ("v2 규칙", "v2_rule_base", "급등락 경계 ← TF-IDF 급등락 확률", "models2.fit_rule_size"),
    "v2_rsize_atr14": ("v2 규칙", "v2_rule_base", "급등락 경계 ← atr14", "models2.fit_rule_size"),
    "v2_radd_text_dir": ("v2 규칙", "v2_rule_base", "s + β·sd·z(TF-IDF 방향)", "models2.fit_rule_add"),
    "v2_radd_lex_posneg": ("v2 규칙", "v2_rule_base", "s + β·sd·z(호재−악재 단어)", "models2.fit_rule_add"),
    "v3_gen_kappa": ("v3", "v2_rule_base", "갭 = gap + κ·mkt_gap", "models2.fit_rule_gen"),
    "v3_gen_theta": ("v3", "v2_rule_base", "갭 = gap + θ·pre_move", "models2.fit_rule_gen"),
    "v3_gen_kappa_theta": ("v3", "v2_rule_base", "κ, θ 둘 다", "models2.fit_rule_gen"),
    "v3_rsize_mkt_vol20": ("v3", "v2_rule_base", "급등락 경계 ← 시장 변동성", "models2.fit_rule_size"),
    "v3_rsize_bigfreq60": ("v3", "v2_rule_base", "급등락 경계 ← 종목 60일 급등락 빈도", "models2.fit_rule_size"),
    "v3_rsize_abs_mkt_gap": ("v3", "v2_rule_base", "급등락 경계 ← |시장 갭|", "models2.fit_rule_size"),
    "v3_recent250": ("v3", "v2_rule_base", "학습 창: 최근 250 기준일", "models2.make_fit recent250"),
    "v3_recent120": ("v3", "v2_rule_base", "학습 창: 최근 120 기준일", "models2.make_fit recent120"),
    "v3_winvote": ("v3", "v2_rule_base", "전체·250·120 창 규칙의 등급 중간값", "models2.make_fit winvote"),
}

PKEYS = ["lams", "lam", "gam", "delta", "beta", "w", "kappa", "theta", "k", "rounds", "a", "b"]


def param_summary(seen):
    """폴드 4개 info → 'λ 0.2/0.1/..' 같은 짧은 문자열."""
    out = []
    for k in PKEYS:
        vals = [s.get(k) for s in seen]
        if all(v is None for v in vals):
            continue
        out.append(f"{k} " + "/".join("-" if v is None else f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals))
    return "; ".join(out)


def main():
    raw = pd.read_csv(DOC / "experiment_table_raw.csv").drop_duplicates("run", keep="last")
    params = json.loads((AN / "params.json").read_text(encoding="utf-8")) if (AN / "params.json").exists() else {}
    rows = []
    for _, r in raw.iterrows():
        g, ref, ch, code = CHANGE[r.run]
        f = [r.fold1, r.fold2, r.fold3, r.fold4]
        base = raw.set_index("run").loc["rule_gapz_ext"]
        up = sum(x > y + 1e-9 for x, y in zip(f, [base.fold1, base.fold2, base.fold3, base.fold4]))
        adopt = (r.fold4 >= base.fold4 - 1e-9) and up >= 3 and (r.mean4 - base.mean4 >= 0.005)
        rows.append({"run": r.run, "묶음": g, "비교 대상": ref, "바꾼 것": ch, "코드": code,
                     "fold1": r.fold1, "fold2": r.fold2, "fold3": r.fold3, "fold4": r.fold4, "4폴드 평균": r.mean4,
                     "unseen fold4": r.fold4_unseen,
                     "3월": r.get("month/2026-03"), "4월": r.get("month/2026-04"), "5월": r.get("month/2026-05"),
                     "vs 규칙 fold4": r.fold4 - base.fold4, "vs 규칙 평균": r.mean4 - base.mean4,
                     "B규칙 대비 상승 폴드": up, "채택 기준": "통과" if adopt else "",
                     "학습된 파라미터 (fold1/2/3/4)": param_summary(params[r.run]) if r.run in params else ""})
    df = pd.DataFrame(rows)
    df.to_csv(DOC / "experiment_table.csv", index=False, encoding="utf-8-sig")

    fmt = lambda v: "" if pd.isna(v) else f"{v:+.4f}" if isinstance(v, float) and abs(v) < 0.1 and v != 0 and False else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["# 실험 검증표 (W&B finance-direction, 점수 run 53개)", "",
             "점수: src.data.score, A 폴드(lab/folds.json). 월은 **대상일 월**. 기준 = `rule_gapz_ext` (B 최종, 제출 v1).",
             "채택 기준(worklog §1): fold4 하락 없음 + 4폴드 중 3폴드 상승 + 평균 +0.005. '바꾼 것'은 config 와 코드로 확인.", ""]
    for g in df["묶음"].unique():
        d = df[df["묶음"] == g]
        lines += [f"## {g}", "", "| run | 비교 대상 | 바꾼 것 | f1 | f2 | f3 | **f4** | 평균 | unseen | 3월 | 4월 | 5월 | vs 규칙 f4 | 상승 폴드 | 학습된 파라미터 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join([r.run, r["비교 대상"], r["바꾼 것"]] + [fmt(r[c]) for c in
                         ["fold1", "fold2", "fold3", "fold4", "4폴드 평균", "unseen fold4", "3월", "4월", "5월"]]
                         + [f"{r['vs 규칙 fold4']:+.4f}", str(r["B규칙 대비 상승 폴드"]), r["학습된 파라미터 (fold1/2/3/4)"]]) + " |")
        lines.append("")
    (DOC / "experiment_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(df[["run", "fold4", "vs 규칙 fold4", "B규칙 대비 상승 폴드", "채택 기준"]].to_string())


if __name__ == "__main__":
    main()
