"""LightGBM(B 피처 26개) + 디코딩 4가지를 W&B 에 올림.  uv run python scripts/a_lgb.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.cv import run_cv  # noqa: E402
from lab.folds import HOLDOUT  # noqa: E402
from lab.leak import check_no_leak  # noqa: E402
from src import Dataset  # noqa: E402
from work.a import lgb as L  # noqa: E402
from work.a.features_b import build_b  # noqa: E402

METHOD = "A-folds4 (주 지표 fold4)"

if __name__ == "__main__":
    ds = Dataset()
    days = ds.split(HOLDOUT)[0]           # 홀드아웃(대상일 2026-06-01~) 은 표에도 넣지 않음
    check_no_leak(build_b, [days[100], days[300], days[-5]])
    print("피처 표", L.table(days).shape, flush=True)   # 449일 전부 캐시
    for dec in ["argmax", "match_train", "match_recent", "aggr"]:
        res = run_cv(L.make_fit(dec), ds, name=f"lgb_v1_{dec}", method=METHOD, tags=["lgb", "B_V1"],
                     config={"model": "lightgbm multiclass", "features": L.FEATS, "n_features": len(L.FEATS),
                             "params": L.PARAMS, "valid_days": L.VALID_DAYS, "decode": dec,
                             "recent_days": L.RECENT_DAYS, "k_grid": L.KS, "main_metric": "fold4/all/score"})
        print(dec); print(res.summary(), flush=True)
    for key, f in L._models.items():
        print(f"{key[1]:%Y-%m-%d}  rounds {f['rounds']}  k {f['k']}  q_train {f['q_train'].round(3)}  q_recent {f['q_recent'].round(3)}")
