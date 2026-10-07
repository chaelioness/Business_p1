"""train / val 로 나눈 데이터를 따로 저장한다.

    uv run python work/common/make_split_data.py <train 원본 폴더>

1. 원본 표: dataset/ 의 모든 표를 train 마지막 대상일의 장 마감(16:00, 그날 정답 확정)까지만 잘라
   <train 원본 폴더> 에 저장. 그 뒤(그날 밤 시간외 등)는 embargo 구간의 피처라 뺀다.
   EDA · 피처 선정은 Dataset(folder=<train 원본 폴더>) 로 하면 val 이 섞일 수 없다.
   자르는 기준은 Dataset 과 같음: known_at (Reddit 은 created_et).
2. B 피처 캐시: work/b/cache/b_features.parquet 를 b_features_train.parquet / b_features_val.parquet 로 나눔.

val 은 피처를 만들 때 과거가 필요해서 원본은 dataset/ (449일 전체) 를 그대로 쓴다.
큰 파일(Reddit 댓글)은 한 번에 읽지 않고 묶음 단위로 나눠 쓴다.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.paths import DATASET  # noqa: E402
from folds import split, train_end  # noqa: E402

B_CACHE = ROOT / "work" / "b" / "cache"


def cut_file(src, dst, col, until):
    """col < until 인 행만 dst 로. 묶음 단위로 읽고 씀."""
    pf = pq.ParquetFile(src)
    ts = pa.scalar(until.to_pydatetime(), type=pf.schema_arrow.field(col).type)
    writer, n_in, n_out = None, 0, 0
    for batch in pf.iter_batches(batch_size=200_000):
        t = pa.Table.from_batches([batch])
        n_in += t.num_rows
        t = t.filter(pc.less(t[col], ts))
        if writer is None:
            writer = pq.ParquetWriter(dst, pf.schema_arrow, compression="zstd")   # 원본과 같은 압축
        if t.num_rows:
            writer.write_table(t)
            n_out += t.num_rows
    if writer is None:
        pq.write_table(pf.schema_arrow.empty_table(), dst)
    else:
        writer.close()
    return n_in, n_out


def main():
    out = Path(sys.argv[1]).resolve()
    days = pd.read_parquet(DATASET / "daily.parquet", columns=["date_et"]).date_et
    te = train_end(days[days >= "2024-08-14"])
    until = te + pd.Timedelta(hours=16, minutes=1)      # 16:00 에 확정되는 그날 일봉(정답)까지 포함
    print(f"train 마지막 대상일 {te:%Y-%m-%d} → 원본 표는 {until} 전까지")
    shutil.rmtree(out, ignore_errors=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "reddit").mkdir(exist_ok=True)
    for f in sorted(DATASET.glob("*.parquet")):
        n_in, n_out = cut_file(f, out / f.name, "known_at", until)
        print(f"  {f.name:28s} {n_in:>12,} → {n_out:>12,}", flush=True)
    for f in sorted((DATASET / "reddit").glob("*.parquet")):
        n_in, n_out = cut_file(f, out / "reddit" / f.name, "created_et", until)
        print(f"  reddit/{f.name:28s} {n_in:>12,} → {n_out:>12,}", flush=True)
    for f in DATASET.glob("*.json"):
        shutil.copy(f, out / f.name)

    b = pd.read_parquet(B_CACHE / "b_features.parquet")
    tr, va = split(b.target)
    b[tr].to_parquet(B_CACHE / "b_features_train.parquet", index=False)
    b[va].to_parquet(B_CACHE / "b_features_val.parquet", index=False)
    print(f"B 피처 캐시: train {tr.sum():,}행 (대상일 ~{b.target[tr].max():%Y-%m-%d}), "
          f"val {va.sum():,}행 ({b.target[va].min():%Y-%m-%d} ~ {b.target[va].max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
