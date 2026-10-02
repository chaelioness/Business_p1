"""C 텍스트 피처 ablation: 정형만 → +뉴스 → +Reddit → +전부 (7개). 같은 폴드, 같은 LightGBM, 같은 디코딩.
text_overnight_hours 는 C 안내대로 모든 비교군에 공통으로 넣음.
    uv run python scripts/a_ablation_text.py [decode]      # 기본 decode = 앞 실험에서 fold4 가 가장 좋은 것
텍스트 표: cache/text_v1_dev/train_text_features.parquet (work/c run.py --data dataset, 홀드아웃 행 삭제)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.cv import run_cv  # noqa: E402
from src import Dataset  # noqa: E402
from work.a import lgb as L  # noqa: E402

METHOD = "A-folds4 (주 지표 fold4)"
T = L.TEXT
GROUPS = {
    "struct": L.FEATS + T["context"],
    "struct+news": L.FEATS + T["context"] + T["news"],
    "struct+reddit": L.FEATS + T["context"] + T["reddit"],
    "struct+text_all": L.FEATS + T["context"] + T["news"] + T["reddit"] + T["joint"],
}

if __name__ == "__main__":
    dec = sys.argv[1] if len(sys.argv) > 1 else "match_train"
    ds = Dataset()
    for g, feats in GROUPS.items():
        res = run_cv(L.make_fit(dec, feats), ds, name=f"ablation_{g}_{dec}", method=METHOD,
                     tags=["ablation", "text", "C"],
                     config={"model": "lightgbm multiclass", "group": g, "features": feats,
                             "n_features": len(feats), "params": L.PARAMS, "decode": dec,
                             "text_source": "C text_baseline v1 (dev 재생성)", "main_metric": "fold4/all/score"})
        print(g); print(res.summary(), flush=True)
