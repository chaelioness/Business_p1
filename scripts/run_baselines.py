"""기준선을 폴드마다 돌려 점수를 냄.

    uv run python scripts/run_baselines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from lab.baselines import BASELINES  # noqa: E402
from lab.cv import run_cv  # noqa: E402

out = {}
for name, fit in BASELINES.items():
    print(f"\n[{name}]")
    res = run_cv(fit)
    print(res.summary().to_string())
    out[name] = res.summary()["all"]

print("\n[요약] 폴드별 score (all)")
print(pd.DataFrame(out).to_string())
