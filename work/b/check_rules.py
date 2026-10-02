"""gapz_rule.py 가 팀 피처 규칙 8개를 지키는지 확인.

    uv run python work/b/check_rules.py

1·3·4·8  check_no_leak 으로 build_gapz, predict 를 여러 날 돌림 (cutoff 이후 조회, day.y, Reddit score 쓰면 LeakError)
         + ds.table / day.y / reddit 를 코드에서 직접 부르는지 소스 검사
2        build_gapz 결과가 symbol + 피처, 종목당 한 줄인지
5        종목 이름을 코드에 적었는지, 피처에 문자열·종목 ID 가 들어갔는지
6        fit 결과(λ, 중앙값, 표준편차, 경계)가 train 날짜만으로 정해지는지: val 날짜를 바꿔도 같은지
7        gapz_rule.py 를 src/model.py 에 붙여 넣은 복사본 저장소에서 main.py check 가 통과하는지
"""

import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src import Dataset  # noqa: E402
from lab.folds import load_folds  # noqa: E402
from lab.leak import check_no_leak, guard  # noqa: E402
import gapz_rule as G  # noqa: E402

SRC = (HERE / "gapz_rule.py").read_text(encoding="utf-8")
ok_all = True


def report(name, ok, msg=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'통과' if ok else '실패'}] {name}  {msg}")


def main():
    ds = Dataset()
    folds = load_folds(ds)
    rng = np.random.default_rng(0)
    pool = [d for f in folds for d in f.val]
    days = [pool[i] for i in rng.choice(len(pool), 40, replace=False)]

    # 1·3·4 소스 검사: 예측 경로(build_gapz, predict)에서 금지된 접근
    tree = ast.parse(SRC)
    funcs = {n.name: ast.get_source_segment(SRC, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    pred_path = funcs["build_gapz"] + funcs["predict"] + funcs["predict_table"] + funcs["_score_s"] + funcs["cut"]
    bad = [p for p in (r"\.table\(", r"\.y\b", r"reddit", r"n_comments", r"\bscore\b\s*[\]\[]") if re.search(p, pred_path)]
    report("규칙 1·3·4 소스: 예측 경로에 ds.table / day.y / reddit 없음", not bad, f"찾은 것 {bad}" if bad else "")
    reads = sorted(set(re.findall(r"day\.(\w+)\(", funcs["build_gapz"])))
    report("규칙 1 읽는 함수", set(reads) <= {"daily", "price", "news", "earnings", "analyst", "reddit"}, f"day.{reads}")

    # 8 누수 검사 (실제 실행)
    prm = G.fit_table(G.build_table(folds[3].train[-120:]))   # 학습은 정답(day.y)을 봐야 하므로 guard 없이
    for name, fn in [("build_gapz", G.build_gapz), ("predict", lambda day: G.predict(day, prm))]:
        try:
            check_no_leak(fn, days)
            report(f"규칙 8 check_no_leak({name}) {len(days)}일", True)
        except Exception as e:  # noqa: BLE001
            report(f"규칙 8 check_no_leak({name})", False, repr(e)[:200])

    # 2 모양
    shapes = []
    for d in days[:15]:
        x = G.build_gapz(guard(d))
        shapes.append((list(x.columns) == ["symbol", "gap_z", "ext_range_z"], x.symbol.is_unique,
                       set(x.symbol) <= set(d.symbols) | set(x.symbol)))
    report("규칙 2 build(day) → [symbol, gap_z, ext_range_z], 종목당 한 줄", all(a and b for a, b, _ in shapes))
    out = G.predict(guard(days[0]), prm)
    report("규칙 2 predict → day.symbols 전부, 한 번씩, label 0~4",
           sorted(out.symbol) == sorted(days[0].symbols) and out.symbol.is_unique and out.label.between(0, 4).all())

    # 5 종목 이름·ID
    syms = set(pd.read_parquet(ROOT / "dataset" / "daily.parquet", columns=["symbol"]).symbol.unique())
    named = [s for s in syms if re.search(rf"['\"]{re.escape(s)}['\"]", SRC)]
    x = G.build_gapz(guard(days[0]))
    feat_types = x.drop(columns="symbol").dtypes
    report("규칙 5 종목 이름을 코드에 안 씀", not named, f"{named}" if named else "")
    report("규칙 5 피처가 숫자뿐 (종목 ID·문자열 없음, 변동성으로 나눈 비율)", all(np.issubdtype(t, np.floating) for t in feat_types))

    # 6 train 에서만 fit: 같은 train 으로 두 번, val 을 바꿔도 결과가 같은지 + fit 에 val 이 안 들어가는지
    f = folds[3]
    tr_days = f.train[-200:]
    p1 = G.fit(tr_days)
    p2 = G.fit(tr_days)
    t = G.build_table(tr_days)
    report("규칙 6 fit 은 넘겨준 train 날짜만 씀 (같은 입력 → 같은 값)", p1 == p2 and t.shape[0] > 0,
           f"train {len(tr_days)}일, λ={p1['lam']}, a={p1['a']:.3f}, b={p1['b']:.3f}")
    va_dates = {d.date for d in f.val}
    tr_dates = {d.date for d in tr_days}
    report("규칙 6 train 날짜와 val 날짜가 안 겹침", not (va_dates & tr_dates))

    # 7 붙여 넣기
    tmp = Path(tempfile.mkdtemp(prefix="paste_gapz_"))
    try:
        shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "main.py", tmp / "main.py")
        body = SRC.split('"""', 2)[2]          # 맨 위 설명 빼고
        model_py = ('"""제출 파일 (gapz_rule 붙여 넣기 확인용)."""\n' + body + '''

from .paths import ARTIFACTS


class Model:
    def __init__(self, params):
        self.params = params

    def predict(self, day):
        return predict(day, self.params)


def load_model():
    return Model(load(ARTIFACTS / "gapz_rule.json"))
''')
        (tmp / "src" / "model.py").write_text(model_py, encoding="utf-8")
        art = tmp / "artifacts"
        art.mkdir(exist_ok=True)
        G.save(p1, art / "gapz_rule.json")
        imports = sorted({n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(model_py))
                          if isinstance(n, ast.Import)} | {n.module.split(".")[0] for n in ast.walk(ast.parse(model_py))
                                                           if isinstance(n, ast.ImportFrom) and n.module and n.level == 0})
        report("규칙 7 붙여 넣은 model.py 의 외부 import", set(imports) <= {"json", "numpy", "pandas"}, f"{imports}")
        split = str(f.val[0].target.date())                     # fold4 val 첫 대상일 (fit 은 그 전 train)
        syms = sorted(rng.choice(sorted(syms), 20, replace=False))
        base = [sys.executable, "main.py", "check", "--split", split, "--data", str(ROOT / "dataset")]
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        for tag, extra in [("전체 종목", []), ("20종목만", ["--symbols", ",".join(syms)])]:
            r = subprocess.run(base + extra, cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=1800, env=env)
            tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()][-4:]
            report(f"규칙 7 src/model.py 에 붙여 넣고 main.py check ({tag}, split {split})", r.returncode == 0,
                   " | ".join(tail))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n전체:", "모두 통과" if ok_all else "실패 있음")


if __name__ == "__main__":
    main()
