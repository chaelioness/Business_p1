"""v2 실험 묶음: LLM 없는 텍스트 + 여러 모델·손실.  uv run python scripts/a_v2.py [이름 ...]
모든 run 은 W&B finance-direction, 태그 v2. 비교 기준은 B 규칙 (vs_gapz_ext, vs_gapz_ext_fold4).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.cv import run_cv  # noqa: E402
from lab.folds import HOLDOUT  # noqa: E402
from src import Dataset  # noqa: E402
from work.a import exp2 as E  # noqa: E402
from work.a.features_b import B_FEATURES_V1  # noqa: E402
from work.a.models2 import make_fit  # noqa: E402

METHOD = "A-folds4 (주 지표 fold4)"
CORE = ["gap_z", "ext_range_z", "abs_gap_z", "vol20", "atr14", "mkt_gap", "gap_rel_z", "post_ret", "pre_move"]
LEXT = ["lex_event", "lex_n", "lex_ev_earn", "lex_ev_pos", "lex_ev_neg", "lex_ev_analyst"]
CT = ["news_log_count_overnight", "news_spike20", "reddit_log_count_overnight"]
LT = ["text_big", "text_dir"]

EXPS = {   # 이름: (kind, feats, text)
    # 규칙 + 텍스트로 갭 신뢰도 조절
    "v2_rule_base": ("rule", [], False),
    "v2_rule_lex_event": ("rule", ["lex_event"], False),
    "v2_rule_lex_n": ("rule", ["lex_n"], False),
    "v2_rule_news_c": ("rule", ["news_log_count_overnight"], False),
    "v2_rule_text_big": ("rule", ["text_big"], True),
    # 선형 (손실 비교)
    "v2_ridge_core": ("ridge", CORE, False),
    "v2_huber_core": ("huber", CORE, False),
    "v2_huber_core_text": ("huber", CORE + LEXT + CT, False),
    "v2_logit_core": ("logit", CORE, False),
    # 트리 (손실 비교)
    "v2_lgb_l2_b26": ("lgb_l2", B_FEATURES_V1, False),
    "v2_lgb_huber_b26": ("lgb_huber", B_FEATURES_V1, False),
    "v2_lgb_l1_b26": ("lgb_l1", B_FEATURES_V1, False),
    "v2_lgb_ord_b26": ("lgb_ord", B_FEATURES_V1, False),
    "v2_lgb_wmulti_b26": ("lgb_wmulti", B_FEATURES_V1, False),
    "v2_hgb_l1_b26": ("hgb_l1", B_FEATURES_V1, False),
    # 트리 + 텍스트
    "v2_lgb_huber_b26_lex": ("lgb_huber", B_FEATURES_V1 + LEXT + CT, False),
    "v2_lgb_huber_b26_lex_tfidf": ("lgb_huber", B_FEATURES_V1 + LEXT + CT + LT, True),
    # 규칙: 급등락 경계만 피처로 움직이기 / 텍스트 방향 더하기
    "v2_rsize_earn_now": ("rule_size", ["earn_now"], False),
    "v2_rsize_lex_event": ("rule_size", ["lex_event"], False),
    "v2_rsize_lex_earn": ("rule_size", ["lex_ev_earn"], False),
    "v2_rsize_news_c": ("rule_size", ["news_log_count_overnight"], False),
    "v2_rsize_text_big": ("rule_size", ["text_big"], True),
    "v2_rsize_atr14": ("rule_size", ["atr14"], False),
    "v2_radd_text_dir": ("rule_add", ["text_dir"], True),
    "v2_radd_lex_posneg": ("rule_add", ["lex_posneg"], False),
    # v3: 갭 구성 바꾸기, 시장 국면 경계, 최근 기간만
    "v3_gen_kappa": ("rule_gen", ["kappa"], False),
    "v3_gen_theta": ("rule_gen", ["theta"], False),
    "v3_gen_kappa_theta": ("rule_gen", ["kappa", "theta"], False),
    "v3_rsize_mkt_vol20": ("rule_size", ["mkt_vol20"], False),
    "v3_rsize_bigfreq60": ("rule_size", ["bigfreq60"], False),
    "v3_rsize_abs_mkt_gap": ("rule_size", ["abs_mkt_gap"], False),
    "v3_recent250": ("recent250", [], False),
    "v3_recent120": ("recent120", [], False),
    "v3_winvote": ("winvote", [], False),
    # 앙상블: B 규칙 + Huber 선형
    "v2_ens_rule_huber_core": ("ens_huber", CORE, False),
    "v2_ens_rule_huber_text": ("ens_huber", CORE + LEXT + CT, False),
    "v2_ens_rule_huber_tfidf": ("ens_huber", CORE + LEXT + CT + LT, True),
}

if __name__ == "__main__":
    ds = Dataset()
    E.master(ds.split(HOLDOUT)[0])
    names = sys.argv[1:] or list(EXPS)
    for n in names:
        kind, feats, text = EXPS[n]
        t0 = time.time()
        res = run_cv(make_fit(kind, feats, text), ds, name=n, method=METHOD, tags=["v2", kind],
                     config={"model": kind, "features": feats, "n_features": len(feats), "text_model": text,
                             "decode": "|s| 경계 a,b (train score 최대)", "events": E.EVENTS,
                             "main_metric": "fold4/all/score"}, verbose=False)
        t = res.table[res.table.group == "all"].set_index("fold").score
        u = res.table[res.table.group == "unseen"].set_index("fold").score
        print(f"{n:30s} fold4 {t.iloc[-1]:.3f} unseen4 {u.iloc[-1]:.3f} | "
              + " ".join(f"{v:.3f}" for v in t) + f" | mean {t.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
