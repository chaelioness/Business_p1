"""전체 기준일의 B 피처를 캐시로 저장한다.

    uv run python work/b/early/build_cache.py            # 이어서 돌림
    uv run python work/b/early/build_cache.py --force    # 처음부터

월 단위로 work/b/cache/parts/YYYY-MM.parquet 에 저장하고, 있으면 건너뛴다.
대상일이 TEST_START(2026-06-01) 이후인 날은 만들지 않는다 (팀 최종 테스트 구간).
마지막에 합쳐서 work/b/cache/b_features.parquet 로 쓴다.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "features"))
sys.path.insert(0, str(HERE.parents[2] / "work" / "common"))

from src import Dataset  # noqa: E402
from features_b import build_b_table  # noqa: E402
from folds import TEST_START  # noqa: E402

CACHE = HERE.parent / "cache"
OUT = CACHE / "b_features.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    parts = CACHE / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    days = [d for d in ds.days() if d.target < pd.Timestamp(TEST_START)]
    by_month = {}
    for d in days:
        by_month.setdefault(f"{d.date:%Y-%m}", []).append(d)

    t0 = time.time()
    for i, (m, ds_m) in enumerate(by_month.items(), 1):
        f = parts / f"{m}.parquet"
        if f.exists() and not args.force:
            continue
        t = time.time()
        tab = build_b_table(ds_m, verbose=False)
        tmap = {d.date: d.target for d in ds_m}
        tab["target"] = tab.date.map(tmap)
        tab.to_parquet(f, index=False)
        print(f"[{i}/{len(by_month)}] {m}  {len(ds_m)}일 {len(tab)}행  "
              f"{time.time() - t:.0f}s (누적 {time.time() - t0:.0f}s)", flush=True)

    tab = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))],
                    ignore_index=True)
    tab.to_parquet(OUT, index=False)
    print(f"저장 {OUT}  {tab.shape}  기준일 {tab.date.min():%Y-%m-%d}~{tab.date.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
