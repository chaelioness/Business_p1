"""홀드아웃 앞쪽만 담은 dataset 을 만듦. 팀원 공유용.

    uv run python scripts/make_dev_dataset.py <출력 폴더> [--until 2026-06-01]

`ds.split(until)` 의 학습 쪽이 볼 수 있는 행만 남김. 기준은 src/data.py 와 같은
known_at (Reddit 은 created_et) 임. 만든 폴더를 dataset/ 으로 쓰면 Dataset() 이
그대로 돌아가고, 홀드아웃 구간은 아예 읽히지 않음.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "dataset"


def cut(src: Path, dst: Path, col: str, until: pd.Timestamp):
    """row group 단위로 읽어 col < until 인 행만 옮김."""
    pf = pq.ParquetFile(src)
    typ = pf.schema_arrow.field(col).type
    if getattr(typ, "tz", None):
        until = until.tz_localize("America/New_York")
    bound = pa.scalar(until.to_pydatetime(), type=typ)
    codec = pf.metadata.row_group(0).column(0).compression.lower()
    kept = total = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dst, pf.schema_arrow, compression=codec) as w:
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i)
            total += t.num_rows
            t = t.filter(pc.less(t[col], bound))
            if t.num_rows:
                w.write_table(t)
                kept += t.num_rows
    print(f"  {src.name:32} {kept:>10,} / {total:>10,}행")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--until", default="2026-06-01", help="홀드아웃 시작일. 이 날부터는 빠짐")
    args = ap.parse_args()
    until, out = pd.Timestamp(args.until), Path(args.out)

    for name in ("daily", "price", "news", "earnings", "analyst"):
        cut(SRC / f"{name}.parquet", out / f"{name}.parquet", "known_at", until)
    for f in sorted((SRC / "reddit").glob("*.parquet")):
        cut(f, out / "reddit" / f.name, "created_et", until)
    shutil.copy(SRC / "symbols.json", out / "symbols.json")


if __name__ == "__main__":
    main()
