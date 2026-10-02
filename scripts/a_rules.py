"""B 규칙 두 개를 W&B 에 올리고, 최종안을 기준선 캐시(gapz_ext)로 저장.
    uv run python scripts/a_rules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.cv import run_cv, save_baseline  # noqa: E402
from src import Dataset  # noqa: E402
from work.a.rules import fit_gapz, fit_gapz_ext  # noqa: E402

METHOD = "A-folds4 (주 지표 fold4)"

if __name__ == "__main__":
    ds = Dataset()
    for name, fit, desc in [
        ("rule_gapz", fit_gapz, "gap/vol20 규칙 (B)"),
        ("rule_gapz_ext", fit_gapz_ext, "B 최종안: gap_z · exp(-λ·ext_range_z 표준화), λ·경계 train"),
    ]:
        res = run_cv(fit, ds, name=name, method=METHOD, tags=["baseline", "rule", "B"], log_model=True,
                     config={"model": name, "description": desc, "source": "B gapz_rule.py 2026-10-03",
                             "features": ["gap_z", "ext_range_z"] if "ext" in name else ["gap_z"],
                             "main_metric": "fold4/all/score"})
        print(name); print(res.summary())
    print(save_baseline(fit_gapz_ext, key="gapz_ext", ds=ds))
