"""제출 v1 학습: B 최종 규칙을 홀드아웃 폴드의 train 구간(~2026-05-11, 공백 12일)으로 맞추고
artifacts/gapz_rule.json 저장, W&B 에 Artifact(type=model) 로 올림.
    uv run python scripts/a_train_v1.py
처음 보는 종목 제외 없이 50종목 전부로 학습 (제출 모델이므로). 홀드아웃 val 은 읽지 않음.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import tracking  # noqa: E402
from lab.folds import load_folds  # noqa: E402
from src import Dataset  # noqa: E402
from src.paths import ARTIFACTS  # noqa: E402
from work.a.rules import fit_gapz_ext  # noqa: E402

if __name__ == "__main__":
    ds = Dataset()
    train = load_folds(ds, holdout=True)[0].train
    m = fit_gapz_ext(train, [])
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "gapz_rule.json"
    path.write_text(json.dumps(m.params, indent=2), encoding="utf-8")
    print(path, m.params)
    cfg = {"model": "rule_gapz_ext", "train": [f"{train[0].date:%Y-%m-%d}", f"{train[-1].date:%Y-%m-%d}"],
           "n_train_days": len(train), "exclude": [], "git": tracking.git_info(), "params": m.params}
    tr = tracking.Tracker(name="submit_v1_gapz_ext", config=cfg, tags=["submit", "v1"], job_type="train")
    tr.model({"gapz_rule": m}, "submit-v1-gapz-ext", cfg)
    tr.finish()
