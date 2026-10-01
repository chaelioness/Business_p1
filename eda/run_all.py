"""EDA 전체 실행:  uv run python -m eda.run_all  [--rebuild]

--rebuild 를 주면 cache/eda_panel.parquet 를 다시 만듦. 결과는 eda/out/<섹션>/ 에 저장되고
섹션별 핵심 수치는 eda/out/findings.json 하나로도 모임.
"""

import json
import sys
import time

from eda import (build_panel, s01_data_audit, s02_target_returns, s03_overnight_gap,
                 s04_technical, s05_events_text, s06_metric_unseen)
from eda.common import CACHE, EDA_FIRST_TARGET, EDA_LAST_TARGET, OUT

SECTIONS = [s01_data_audit, s02_target_returns, s03_overnight_gap,
            s04_technical, s05_events_text, s06_metric_unseen]


def main():
    print(f"EDA 구간: 대상일 {EDA_FIRST_TARGET:%Y-%m-%d} ~ {EDA_LAST_TARGET:%Y-%m-%d} (fold4 train)")
    if "--rebuild" in sys.argv or not (CACHE / "eda_panel.parquet").exists():
        build_panel.build()
    for mod in SECTIONS:
        t = time.time()
        mod.main()
        print(f"  ({time.time() - t:.0f}s)\n")
    allf = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT.glob("*/findings.json"))}
    (OUT / "findings.json").write_text(json.dumps(allf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료 → {OUT}")


if __name__ == "__main__":
    main()
