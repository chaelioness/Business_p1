"""features_b.py 를 model.py 에 그대로 붙여 넣어도 도는지 시험한다.

    uv run python work/b/test_paste.py <임시 폴더>

실제 src/model.py 는 건드리지 않는다. 임시 폴더에 src/ 와 main.py 를 복사하고,
src/model.py 를 [features_b.py 원문 + LightGBM 래퍼] 로 바꾼 뒤
main.py check 를 (1) 전체 종목, (2) 20종목만 보이게 해서 돌린다.
모델은 마지막 폴드 시작 전 대상일로 학습한 V1 LightGBM (기본 파라미터, argmax).
팀 데이터는 2026-06-01 앞까지만 있으므로 check 구간도 그 안(마지막 폴드)으로 잡는다.
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from features_b import B_FEATURES_V1  # noqa: E402
from folds import FOLD_STARTS  # noqa: E402
from v1_check import load_b, train_lgb  # noqa: E402

SPLIT = FOLD_STARTS[-1]

WRAPPER = '''

# ---------------------------------------------------------------- 모델 (붙여 넣기 시험용)

import lightgbm as lgb

from .data import FLAT
from .paths import ARTIFACTS


class Model:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, day):
        x = build_b(day).set_index("symbol").reindex(day.symbols)
        label = pd.Series(FLAT, index=x.index)
        ok = x[B_FEATURES_V1].notna().any(axis=1)
        if ok.any():
            label[ok] = self.booster.predict(x.loc[ok, B_FEATURES_V1]).argmax(1)
        return pd.DataFrame({"symbol": x.index, "label": label.astype(int).values})


def load_model():
    return Model(lgb.Booster(model_file=str(ARTIFACTS / "b_v1.txt")))
'''


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "main.py", out / "main.py")
    (out / "artifacts").mkdir()

    src = (HERE / "features_b.py").read_text(encoding="utf-8")
    (out / "src" / "model.py").write_text('"""붙여 넣기 시험용 model.py."""\n\n' + src + WRAPPER,
                                          encoding="utf-8")

    b = load_b()
    m, n = train_lgb(b[b.target < pd.Timestamp(SPLIT)], B_FEATURES_V1)
    m.save_model(str(out / "artifacts" / "b_v1.txt"))
    print(f"모델 저장 (반복 {n})")

    syms = sorted(random.Random(1).sample(sorted(b.symbol.unique()), 20))
    data = str(ROOT / "dataset")
    base = [sys.executable, "main.py", "check", "--split", SPLIT, "--data", data]
    code = 0
    for tag, extra in [("전체 종목", []), ("20종목", ["--symbols", ",".join(syms)])]:
        print(f"\n== main.py check ({tag}) ==", flush=True)
        r = subprocess.run(base + extra, cwd=out, env={**__import__("os").environ,
                                                        "PYTHONIOENCODING": "utf-8"})
        code |= r.returncode
    print("\n결과:", "통과" if code == 0 else "실패")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
